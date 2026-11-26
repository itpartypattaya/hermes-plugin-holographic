"""Opt-in, bounded task outcomes. No truth votes, no model calls, no copy in the consumer."""
import contextlib
import json
import logging
import re
import sqlite3
import time
import uuid

from .screen import _recall_problem
from .history import source_of

TTL = 30 * 86400
MAX_ATTEMPTS = 1000
MAX_REVISIONS = 10
MAX_REFS = 100
# The journal is optional: don't hold the answer behind another process's writer.
_ATTEMPT_BUSY_MS = 50
_ATTEMPT_MAX_TRIES = 2
_SLOW_ATTEMPT_MS = 25
# A file rewrite another process blocked is retried by a receipt write at most this often: while a reader elsewhere
# stays, each VACUUM would add the whole database to the WAL again.
_SCRUB_RETRY_S = 60
logger = logging.getLogger(__name__)
_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
OUTCOMES = frozenset(("useful", "dead_end", "corrected", "unknown"))
REASONS = frozenset(("wrong_entity", "stale_version", "irrelevant", "missing_evidence", "answer_error", "other"))

# Lazy, additive migration. DELETE triggers also work when an older provider forgets a fact.
# Every referenced candidate invalidates the entire attempt: no unindexed text survives a partial erase.
SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_attempts (
 attempt_id TEXT PRIMARY KEY, created_at REAL NOT NULL, kind TEXT NOT NULL,
 session_key TEXT NOT NULL, stages TEXT NOT NULL, emitted TEXT NOT NULL, trace_status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_attempt_refs (
 attempt_id TEXT NOT NULL, fact_id INTEGER NOT NULL, version_id INTEGER NOT NULL,
 PRIMARY KEY (attempt_id, fact_id, version_id)
);
CREATE INDEX IF NOT EXISTS memory_refs_fact ON memory_attempt_refs(fact_id);
CREATE INDEX IF NOT EXISTS memory_refs_version ON memory_attempt_refs(version_id);
CREATE TABLE IF NOT EXISTS memory_outcomes (
 event_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL, created_at REAL NOT NULL,
 origin_signal_id TEXT NOT NULL UNIQUE, supersedes TEXT UNIQUE, payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_outcomes_attempt ON memory_outcomes(attempt_id);
CREATE TRIGGER IF NOT EXISTS memory_attempt_delete AFTER DELETE ON memory_attempts BEGIN
 DELETE FROM memory_outcomes WHERE attempt_id = old.attempt_id;
 DELETE FROM memory_attempt_refs WHERE attempt_id = old.attempt_id;
END;
CREATE TRIGGER IF NOT EXISTS memory_fact_delete BEFORE DELETE ON facts
WHEN EXISTS (SELECT 1 FROM fact_versions WHERE fact_id=old.fact_id AND status='current')
 OR NOT EXISTS (SELECT 1 FROM fact_versions WHERE fact_id=old.fact_id) BEGIN
 DELETE FROM memory_attempts WHERE attempt_id IN
 (SELECT attempt_id FROM memory_attempt_refs WHERE fact_id = old.fact_id);
END;
CREATE TRIGGER IF NOT EXISTS memory_version_delete BEFORE DELETE ON fact_versions BEGIN
 DELETE FROM memory_attempts WHERE attempt_id IN
 (SELECT attempt_id FROM memory_attempt_refs WHERE version_id = old.version_id);
END;
"""


def identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Expected an opaque identifier (1-80 ASCII letters, numbers, _ or -)")
    return value


def current_outcomes(rows):
    events = [json.loads(r["payload"]) for r in rows]
    replaced = {e["supersedes"] for e in events if e.get("supersedes")}
    return [{**e, "active": e["event_id"] not in replaced} for e in events]


class ExperienceMixin:
    def experience_ready(self):
        return bool(self._one("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_attempts'"))

    def _experience_schema(self):
        if not self.experience_ready():
            # DDL is transactional too; executescript would commit an outer transaction.
            with self._tx():
                for statement in _schema_statements():
                    self._conn.execute(statement)
                self._set_meta("experience_schema", "1")
        if self._meta("experience_schema") != "1":
            raise ValueError("Unsupported experience schema")

    def _prune_experiences(self):
        outcomes = self._one("SELECT count(*) AS n FROM memory_outcomes")["n"]
        self._conn.execute("DELETE FROM memory_attempts WHERE created_at <= ?", (time.time() - TTL,))
        self._conn.execute("DELETE FROM memory_attempts WHERE attempt_id IN (SELECT attempt_id FROM "
                           "memory_attempts ORDER BY created_at DESC, rowid DESC LIMIT -1 OFFSET ?)", (MAX_ATTEMPTS,))
        if self._one("SELECT count(*) AS n FROM memory_outcomes")["n"] < outcomes:
            self._mark_scrub()  # an outcome's free text went: the file is rewritten as after a forget

    def prune_experiences(self):
        """Retention when a writer opens the store, collection on or off: outcome text must not outlive the limits
        because nothing records any more. Never creates the journal; returns the number of receipts removed."""
        with self._lock:
            if not self.experience_ready():
                return 0
            due = self._one("SELECT count(*) AS n, coalesce(sum(created_at <= ?), 0) AS expired FROM memory_attempts",
                            (time.time() - TTL,))
            if not due["expired"] and due["n"] <= MAX_ATTEMPTS:
                if self._meta("scrub_pending") is not None:
                    self.scrub()  # an outcome a receipt write removed and could not scrub then
                return 0  # the normal open: a read, no write transaction
            prior_timeout = self._conn.execute("PRAGMA busy_timeout").fetchone()[0]
            self._conn.execute(f"PRAGMA busy_timeout={min(prior_timeout, _ATTEMPT_BUSY_MS)}")
            try:
                with self._tx():
                    self._prune_experiences()
                    left = self._one("SELECT count(*) AS n FROM memory_attempts")["n"]
            finally:
                self._conn.execute(f"PRAGMA busy_timeout={prior_timeout}")
            self.scrub()  # blocked by another process: the mark stays and the next writer open retries (0.20.2)
            return due["n"] - left

    def record_attempt(self, kind, emitted, stages, session_key, trace=False):
        # This connection is shared by providers in the process, so hold its lock
        # until the original timeout is restored, even when a transaction fails.
        started = time.perf_counter()
        lock_ms = schema_ms = transaction_ms = 0.0
        timings = {}
        status = "error"
        tries = 0
        result = None
        with self._lock:
            lock_ms = (time.perf_counter() - started) * 1000
            prior_timeout = self._conn.execute("PRAGMA busy_timeout").fetchone()[0]
            self._conn.execute(f"PRAGMA busy_timeout={min(prior_timeout, _ATTEMPT_BUSY_MS)}")
            try:
                for tries in range(1, _ATTEMPT_MAX_TRIES + 1):
                    timings.clear()
                    try:
                        phase = time.perf_counter()
                        try:
                            self._experience_schema()
                        finally:
                            schema_ms += (time.perf_counter() - phase) * 1000
                        phase = time.perf_counter()
                        try:
                            result = self._record_attempt(kind, emitted, stages, session_key, trace, timings)
                        finally:
                            transaction_ms += (time.perf_counter() - phase) * 1000
                        status = "ok"
                        break
                    except sqlite3.OperationalError as exc:
                        code = getattr(exc, "sqlite_errorcode", None)
                        busy = (code is not None and (code & 0xff) == sqlite3.SQLITE_BUSY) or (
                            code is None and str(exc) == "database is locked")
                        if not busy or tries == _ATTEMPT_MAX_TRIES or self._conn.in_transaction:
                            raise
            finally:
                self._conn.execute(f"PRAGMA busy_timeout={prior_timeout}")
                total_ms = (time.perf_counter() - started) * 1000
                if total_ms >= _SLOW_ATTEMPT_MS:
                    logger.warning("holographic: experience slow lock_ms=%.1f schema_ms=%.1f begin_ms=%s "
                                   "work_ms=%s commit_ms=%s transaction_ms=%.1f total_ms=%.1f tries=%d status=%s",
                                   lock_ms, schema_ms, *(f"{timings[k]:.1f}" if k in timings else "-"
                                                         for k in ("begin", "work", "commit")),
                                   transaction_ms, total_ms, tries, status)
        self._scrub_after_receipt()
        return result

    def _scrub_after_receipt(self):
        """After the receipt's transaction, not inside it: rewrite the file when retention removed an outcome (or a
        forget left the rewrite pending), on this open connection — without it the correction text stayed in the
        main file or the WAL until the next open. Only when the mark is set, so a normal receipt pays one meta read.
        Busy timeout 0 as at forget (scrub): another process reading the database makes it come back at once with
        the mark kept, and a receipt write retries it at most every _SCRUB_RETRY_S; a writer open retries too."""
        try:
            with self._lock:
                if self._meta("scrub_pending") is None:
                    self._entry["receipt_scrub_failed"] = None  # served elsewhere (an open, a forget): no wait
                    return
                failed = self._entry.get("receipt_scrub_failed")
                if failed is not None and time.monotonic() - failed < _SCRUB_RETRY_S:
                    return
                started = time.perf_counter()
                done = self.scrub()
                self._entry["receipt_scrub_failed"] = None if done else time.monotonic()
                logger.info("holographic: file rewrite after journal retention %s in %.1f ms",
                            "done" if done else "deferred", (time.perf_counter() - started) * 1000)
        except sqlite3.Error as exc:  # the receipt is committed; the mark stays for the next try
            logger.warning("holographic: file rewrite after journal retention deferred (%s)", exc)

    @contextlib.contextmanager
    def _timed_experience_tx(self, timings):
        started = time.perf_counter()
        work_done = None
        try:
            with self._tx():
                work_start = time.perf_counter()
                timings["begin"] = (work_start - started) * 1000
                try:
                    yield
                finally:
                    work_done = time.perf_counter()
                    timings["work"] = (work_done - work_start) * 1000
        finally:
            end = time.perf_counter()
            if "begin" not in timings:
                timings["begin"] = (end - started) * 1000
            elif work_done is not None:
                timings["commit"] = (end - work_done) * 1000

    def _record_attempt(self, kind, emitted, stages, session_key, trace=False, timings=None):
        with self._timed_experience_tx(timings if timings is not None else {}):
            self._prune_experiences()
            refs = {}

            def ref(fact, strict=True):
                fid = int(fact["fact_id"])
                version = self._current_version(fid)
                if version is None or version["content"] != fact.get("content"):
                    if not strict:
                        return None
                    raise ValueError("Memory changed during retrieval; attempt was not recorded")
                entry = {"fact_id": fid, "version_id": int(version["version_id"])}
                refs[(fid, entry["version_id"])] = entry
                return entry

            issued = [ref(f) for f in emitted[:MAX_REFS]]
            if trace:  # why each came in (0.32.0): a kind and numbers, never text; a copy, the stages share the refs
                issued = [dict(r, why=f["_why_text"]) if f.get("_why_text") else r
                          for r, f in zip(issued, emitted[:MAX_REFS])]
            observations = []
            if trace:
                # A candidate another writer changed or forgot meanwhile (a rerank call can take seconds) is left out
                # and counted: the emission stays exact and the trace never names a version it did not see.
                for name, facts, *status in stages[:16]:
                    seen = [ref(f, strict=False) for f in facts[:MAX_REFS]]
                    observation = {"stage": name, "facts": [r for r in seen if r], "truncated": len(facts) > MAX_REFS}
                    if status:  # the rerank stage (0.28.1): the call's outcome, a word from Rescorer.outcome()
                        observation["status"] = status[0]
                    if None in seen:
                        observation["changed"] = seen.count(None)
                    observations.append(observation)
            aid = uuid.uuid4().hex
            self._conn.execute("INSERT INTO memory_attempts VALUES (?,?,?,?,?,?,?)",
                               (aid, time.time(), kind, session_key, json.dumps(observations), json.dumps(issued),
                                "recorded" if trace else "not_recorded"))
            self._conn.executemany("INSERT INTO memory_attempt_refs VALUES (?,?,?)",
                                   [(aid, r["fact_id"], r["version_id"]) for r in refs.values()])
            self._prune_experiences()
            return aid

    def record_outcome(self, attempt_id, event_id, outcome, reason, *, correction="", supersedes=None,
                       actor="agent", session_key="", task="", correction_refs=None):
        identifier(attempt_id)
        identifier(event_id)
        if supersedes is not None:
            identifier(supersedes)
        if outcome not in OUTCOMES or reason not in REASONS:
            raise ValueError("Unknown outcome or reason")
        if actor != "agent":
            raise ValueError("Human verification is unsupported; agent reports cannot assert it")
        if not isinstance(correction, str) or len(correction) > 400 or (correction and _recall_problem(correction)):
            raise ValueError("Correction must be safe plain text, at most 400 characters")
        if not isinstance(task, str) or len(task) > 160 or (task and _recall_problem(task)):
            raise ValueError("Task must be safe plain text, at most 160 characters")
        refs = correction_refs or []
        if not isinstance(refs, list) or len(refs) > 20 or any(
                not isinstance(r, dict) or set(r) != {"fact_id", "version_id"} or
                any(type(v) is not int or v <= 0 for v in r.values()) for r in refs):
            raise ValueError("Correction references must name up to 20 fact/version pairs")
        if not self.experience_ready():
            raise ValueError("Attempt unavailable")
        payload = {"event_id": event_id, "outcome": outcome, "reason": reason, "correction": correction,
                   "task": task, "correction_refs": refs,
                   "actor": "agent", "human_verified": False, "supersedes": supersedes}
        encoded = json.dumps(payload, sort_keys=True)
        with self._tx():
            attempt = self._one("SELECT * FROM memory_attempts WHERE attempt_id=?", (attempt_id,))
            if not attempt or attempt["created_at"] <= time.time() - TTL or attempt["session_key"] != session_key:
                raise ValueError("Attempt unavailable")
            previous = self._one("SELECT * FROM memory_outcomes WHERE event_id=?", (event_id,))
            if previous:
                if previous["attempt_id"] != attempt_id or previous["payload"] != encoded:
                    raise ValueError("Event identifier conflict")
                return {"recorded": True, "already": True, "event_id": event_id}
            rows = self._conn.execute("SELECT * FROM memory_outcomes WHERE attempt_id=? ORDER BY rowid",
                                      (attempt_id,)).fetchall()
            active = [e for e in current_outcomes(rows) if e["active"]]
            if len(rows) >= MAX_REVISIONS or (active and supersedes != active[0]["event_id"]) \
                    or (not active and supersedes is not None):
                raise ValueError("Outcome revision conflict or revision limit")
            for ref in refs:
                version = self._version(ref["version_id"])
                if version is None or version["fact_id"] != ref["fact_id"]:
                    raise ValueError("Correction reference unavailable")
                self._conn.execute("INSERT OR IGNORE INTO memory_attempt_refs VALUES (?,?,?)",
                                   (attempt_id, ref["fact_id"], ref["version_id"]))
            # One root signal per real attempt. Revisions replace its active outcome; legacy feedback is not imported.
            signal = f"{attempt_id}:{len(rows)}"
            self._conn.execute("INSERT INTO memory_outcomes VALUES (?,?,?,?,?,?)",
                               (event_id, attempt_id, time.time(), signal, supersedes, encoded))
            return {"recorded": True, "event_id": event_id, "human_verified": False}

    def experience_report(self, attempt_id, session_key=None):
        identifier(attempt_id)
        with self._lock:
            if not self.experience_ready():
                return {"status": "not_recorded"}
            row = self._one("SELECT * FROM memory_attempts WHERE attempt_id=?", (attempt_id,))
            if not row or (session_key is not None and row["session_key"] != session_key):
                return {"status": "not_recorded"}
            if row["created_at"] <= time.time() - TTL:
                return {"status": "expired"}
            events = self._conn.execute("SELECT * FROM memory_outcomes WHERE attempt_id=? ORDER BY rowid",
                                        (attempt_id,)).fetchall()
            emitted = json.loads(row["emitted"])
            # Source metadata only: never copy the fact text or expand another store.
            for ref in emitted:
                version = self._version(ref["version_id"])
                if version is None:
                    return {"status": "unavailable"}
                ref["version_status"] = version["status"]
                ref["origin"] = version["src_origin"]
                ref["source"] = source_of(version) or "unknown"
            emitted_ids = {r["fact_id"] for r in emitted}
            stages = json.loads(row["stages"])
            decisions = []
            if stages:
                observed = {r["fact_id"]: r for stage in stages for r in stage["facts"]}
                coverage = {r["fact_id"] for stage in stages if stage["stage"] == "coverage_admitted"
                            for r in stage["facts"]}
                for fid, ref in observed.items():
                    # Do not invent a precise gate reason when that branch wasn't observed.
                    rejected = {s["stage"] for s in stages if s["stage"].startswith("rejected_")
                                and any(r["fact_id"] == fid for r in s["facts"])}
                    reason = "emitted" if fid in emitted_ids else sorted(rejected)[0] if rejected else \
                        "coverage_rejected" if row["kind"] == "prefetch" \
                        and any(s["stage"] == "coverage_admitted" for s in stages) and fid not in coverage \
                        and any(fid == r["fact_id"] for s in stages if s["stage"] == "lexical_candidates"
                                for r in s["facts"]) else "not_emitted_reason_not_recorded"
                    decisions.append({**ref, "decision": reason})
            return {"status": "ok", "schema": 1, "attempt_id": attempt_id, "kind": row["kind"],
                    "trace_status": row["trace_status"], "stages": stages, "decisions": decisions,
                    "emitted": emitted, "outcomes": current_outcomes(events),
                    "interpretation": "Emission is not proof that the answer used the fact; outcomes are agent reports"}

    def recent_attempts(self, session_key):
        with self._lock:
            if not self.experience_ready():
                return []
            return [dict(r) for r in self._conn.execute("SELECT attempt_id,kind,created_at FROM memory_attempts "
                    "WHERE session_key=? AND created_at>? ORDER BY created_at DESC,rowid DESC LIMIT 20",
                    (session_key, time.time() - TTL))]


def _schema_statements():
    import sqlite3
    statement = ""
    for line in SCHEMA.splitlines():
        statement += line + "\n"
        if sqlite3.complete_statement(statement):
            yield statement
            statement = ""
