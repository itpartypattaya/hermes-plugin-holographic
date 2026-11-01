"""Fact history (0.8.0): versions with valid time and provenance, the five ways a fact stops being current, and
forgetting for good. Mixed into MemoryStore; docs/history.md is the contract.

``facts`` stays the projection of current versions (no new column, so older versions of the plugin and tools that
read it keep working); ``fact_versions`` holds every version, the current one included, with its own FTS index that
only ``as_of`` reads."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
import unicodedata
from datetime import date, datetime, timezone

from .entity_names import entity_key

logger = logging.getLogger(__name__)


def error_kind(exc: BaseException) -> str:
    """What failed, for a log line or stats: the exception's type and SQLite's error name (SQLITE_BUSY), never its
    text. SQLite quotes a column's value in some messages ("Could not decode to UTF-8 column 'content' with text
    '...'"), so the text could carry a fact or a message past the screen."""
    name = getattr(exc, "sqlite_errorname", None)
    return f"{type(exc).__name__} ({name})" if name else type(exc).__name__


HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_versions (
    version_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id        INTEGER NOT NULL,
    seq            INTEGER NOT NULL,
    content        TEXT NOT NULL,
    category       TEXT DEFAULT 'general',
    tags           TEXT DEFAULT '',
    valid_from     TEXT,
    valid_to       TEXT,
    valid_to_exact INTEGER NOT NULL DEFAULT 1,
    status         TEXT NOT NULL DEFAULT 'current',
    recorded_at    TEXT NOT NULL,
    recorded_on    TEXT NOT NULL,
    closed_at      TEXT,
    supersedes     INTEGER,
    superseded_by  INTEGER,
    trust_score    REAL,
    src_via        TEXT,
    src_platform   TEXT,
    src_chat_type  TEXT,
    src_chat_id    TEXT,
    src_thread_id  TEXT,
    src_session_id TEXT,
    src_user_id    TEXT,
    src_user_name  TEXT,
    src_origin     TEXT,  -- 0.15.0: said / confirmed / inferred; a file an older version created gets it on migration
    UNIQUE (fact_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_versions_fact  ON fact_versions(fact_id);
CREATE INDEX IF NOT EXISTS idx_versions_valid ON fact_versions(status, valid_from, valid_to);
CREATE UNIQUE INDEX IF NOT EXISTS idx_versions_current ON fact_versions(fact_id) WHERE status = 'current';

CREATE VIRTUAL TABLE IF NOT EXISTS fact_versions_fts
    USING fts5(content, tags, content=fact_versions, content_rowid=version_id);
CREATE TRIGGER IF NOT EXISTS fact_versions_ai AFTER INSERT ON fact_versions BEGIN
    INSERT INTO fact_versions_fts(rowid, content, tags) VALUES (new.version_id, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS fact_versions_ad AFTER DELETE ON fact_versions BEGIN
    INSERT INTO fact_versions_fts(fact_versions_fts, rowid, content, tags)
        VALUES ('delete', old.version_id, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS fact_versions_au AFTER UPDATE OF content, tags ON fact_versions BEGIN
    INSERT INTO fact_versions_fts(fact_versions_fts, rowid, content, tags)
        VALUES ('delete', old.version_id, old.content, old.tags);
    INSERT INTO fact_versions_fts(rowid, content, tags) VALUES (new.version_id, new.content, new.tags);
END;

-- Tombstones: the id and keyed hashes of every version's text, never the text.
CREATE TABLE IF NOT EXISTS forgotten (
    fact_id      INTEGER PRIMARY KEY,
    forgotten_at TEXT NOT NULL,
    digests      TEXT NOT NULL
);

-- 0.15.0: each time a person confirmed a fact, once per session (fact_feedback confirmed, or the same text added
-- again as said). No text; forget deletes the rows with the fact.
CREATE TABLE IF NOT EXISTS fact_confirmations (
    fact_id      INTEGER NOT NULL,
    confirmed_at TEXT NOT NULL,
    session_id   TEXT,
    user_id      TEXT,
    via          TEXT
);
CREATE INDEX IF NOT EXISTS idx_confirmations_fact ON fact_confirmations(fact_id);

-- 0.19.0: a fact merged into another (merge). The copy's last version stays under its own id with status 'merged';
-- this row says which fact it is part of now, and which version of that fact it was merged into.
CREATE TABLE IF NOT EXISTS fact_merges (
    from_fact_id    INTEGER PRIMARY KEY,
    into_fact_id    INTEGER NOT NULL,
    into_version_id INTEGER NOT NULL,
    merged_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_merges_into ON fact_merges(into_fact_id);

-- 0.22.0: the names declared on a fact while a version was current, written when change / correct / update
-- closes that version, so revert links them again — a new text given with names replaces them, and a name no text
-- says could not be told otherwise. Every record has a row with an empty name (the record exists, even when no name
-- is left in it); closed_by is the version that closed it (a record another closing made is stale). A revert to the
-- version uses the record up. d_key / d_fold / d_old: the name's keyed hashes in the three forms a tombstone of
-- delete_entity is written under (_name_digest_forms), so a tombstone finds every spelling it covers by index.
-- Additive: an older release ignores the table (its revert keeps the names as before), and its forget and its
-- delete_entity remove the rows too, through the triggers in the file (VERSION_NAMES_TRIGGERS).
CREATE TABLE IF NOT EXISTS version_names (
    version_id INTEGER NOT NULL,
    name       TEXT NOT NULL,
    name_key   TEXT NOT NULL,
    closed_by  INTEGER NOT NULL,
    d_key      TEXT,
    d_fold     TEXT,
    d_old      TEXT,
    PRIMARY KEY (version_id, name_key)
);
CREATE INDEX IF NOT EXISTS idx_version_names_key ON version_names(name_key);
CREATE INDEX IF NOT EXISTS idx_version_names_d_key ON version_names(d_key);
CREATE INDEX IF NOT EXISTS idx_version_names_d_fold ON version_names(d_fold);
CREATE INDEX IF NOT EXISTS idx_version_names_d_old ON version_names(d_old);
CREATE TRIGGER IF NOT EXISTS version_names_ad AFTER DELETE ON fact_versions BEGIN
    DELETE FROM version_names WHERE version_id = old.version_id;
END;
"""

# Run after the entity schema (entity_tombstones): a tombstone written by any writer, an older release's
# delete_entity included, removes in its own transaction every name a version kept that the tombstone covers, before
# that writer rewrites the file. Nothing is left for an open to reconcile, so no open hashes the kept names.
VERSION_NAMES_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS version_names_tombstoned AFTER INSERT ON entity_tombstones BEGIN
    DELETE FROM version_names
     WHERE name_key != '' AND (d_key = new.digest OR d_fold = new.digest OR d_old = new.digest);
END;
"""

# The history schema this version knows. 1: 0.8.0-0.18.x (0.15.0's additions were additive). 2 (0.19.0): a file that
# holds merges — a writer of 0.15-0.18 would forget a keep and leave its copies' text behind, so it refuses the file
# (HistoryTooNew). The mark is raised by the first merge, not on open: a file without merges stays at 1 and an older
# release still writes it.
HISTORY_SCHEMA_VERSION = 2
_HISTORY_SCHEMA_BASE = 1  # what a file without merges needs
_MERGE_MAX = 20  # copies one merge takes
_ERASE_BATCH = 400  # ids per pass of _erase: twice that many parameters stay under SQLite's old limit of 999
# Who stood behind a version (0.15.0): the person said it, the person confirmed the agent's conclusion, or the agent
# concluded it. No value: not known (everything stored before 0.15.0, mirrored memory entries).
ORIGINS = ("said", "confirmed", "inferred")
# The trust a new or changed fact starts with, as a step from default_trust: what the person said or confirmed above
# an unknown origin, the agent's own conclusion below it (measured: 40 of 40 conflicting pairs, see CHANGELOG 0.15.0).
ORIGIN_TRUST = {"said": 0.1, "confirmed": 0.1, "inferred": -0.1}
SOURCE_FIELDS = ("via", "platform", "chat_type", "chat_id", "thread_id", "session_id", "user_id", "user_name",
                 "origin")
_SOURCE_COLUMNS = tuple(f"src_{f}" for f in SOURCE_FIELDS)
CLOSED = ("changed", "ended")  # past versions that really held; 'corrected' ones never did
_TS = "%Y-%m-%d %H:%M:%S"  # SQLite CURRENT_TIMESTAMP format, UTC


class HistoryUnavailable(RuntimeError):
    """History is read in a context that may not migrate (a subagent or cron job) before any writer set it up."""


class NotFound(KeyError):
    """No such fact or entity: an answer to give the agent, not a missing argument (a KeyError from args[...])."""


class HistoryTooNew(RuntimeError):
    """The file carries a history schema this version does not know: a newer release wrote it. A writer refuses
    to open it (its migration would stamp the schema back and could mis-read new columns); readers still read."""


class ChangedSince(ValueError):
    """A conditional write (merge with `expect`): a participant's current version is not the one the caller saw.
    Nothing was written; *current* maps each named fact to its current version id (None: no current version)."""

    def __init__(self, current: dict):
        super().__init__("changed_since: " + ", ".join(f"#{f} is at version id {v}" for f, v in current.items()))
        self.current = current


class NotOwned(ValueError):
    """A merge limited to one session's facts (a scheduled job, cron_writes): the current version of each fact in
    *ids* was not stored by that session. Checked inside the merge's transaction; nothing was written."""

    def __init__(self, ids):
        self.ids = [int(i) for i in ids]
        super().__init__("not stored in this session: " + ", ".join(f"#{i}" for i in self.ids))


def normalize(text: str) -> str:
    """What two texts must share to count as the same fact for tombstones: NFKC, casefold, whitespace collapsed,
    end punctuation dropped."""
    return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split()).strip(" .!?;,")


def parse_day(value, *, field: str, today: str) -> str | None:
    """A date argument: YYYY-MM-DD, not after *today*. None / "" -> None (unknown)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    text = str(value).strip()
    try:
        day = date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{field} must be a date as YYYY-MM-DD (got {text!r})") from None
    if len(text) != 10:
        raise ValueError(f"{field} must be a date as YYYY-MM-DD (got {text!r})")
    if day.isoformat() > today:
        raise ValueError(f"{field} {day.isoformat()} is in the future; only dates up to today ({today}) are accepted")
    return day.isoformat()


def version_state(row, day: str) -> tuple[str, str]:
    """('holds' | 'undetermined' | 'no', reason) of a version on *day* (YYYY-MM-DD), by the rules of docs/history.md:
    an unknown start is known only from the day the version was recorded; an inexact end is an upper bound."""
    if row["status"] == "corrected":
        return "no", "corrected: it never held"
    start, end = row["valid_from"], row["valid_to"]
    if start and day < start:
        return "no", f"held from {start}"
    if end and day >= end:
        return "no", f"held until {end}" if row["valid_to_exact"] else f"no longer held by {end}"
    if not start and day < row["recorded_on"]:
        return "undetermined", f"start unknown: recorded on {row['recorded_on']}"
    if end and not row["valid_to_exact"] and day > row["recorded_on"]:
        return "undetermined", f"held on {row['recorded_on']}; changed at some point before {end}"
    return "holds", ""


def source_of(row, prefix: str = "src_") -> dict | None:
    """The provenance of a version row as a dict (empty fields left out); None for facts stored before 0.8.0."""
    present = row.keys()  # a file an older version created has no src_origin until a writer migrates it
    src = {f: row[prefix + f] for f in SOURCE_FIELDS if prefix + f in present and row[prefix + f] not in (None, "")}
    return None if not src or src.get("via") == "legacy" else src


def _check_after_end(start: str | None, ended) -> None:
    """A resumed version starts after the ended one it follows: not before its end when that is exact, and never on
    or before its start (the two would both hold)."""
    if not start or ended is None:
        return
    if ended["valid_to_exact"] and ended["valid_to"] and start < ended["valid_to"]:
        raise ValueError(f"valid_from {start} is before the fact ended ({ended['valid_to']}); use correct or revert to "
                         "change how it ended")
    if ended["valid_from"] and start <= ended["valid_from"]:
        raise ValueError(f"valid_from {start} is not after the ended version's start ({ended['valid_from']}): both "
                         "would hold; use correct or revert to change how it ended")


class HistoryMixin:
    """The history half of MemoryStore. Expects _conn, _tx, _one, _write, _meta, _set_meta, _entry, default_trust, tz and the
    fact-row helpers of MemoryStore."""

    tz = None  # tzinfo of Hermes's configured timezone; None = UTC

    # -- clocks ----------------------------------------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime(_TS)

    def _today(self) -> str:
        return datetime.now(self.tz or timezone.utc).date().isoformat()

    def today(self) -> str:
        """Today in Hermes's timezone: the latest date a fact's history accepts (parse_day)."""
        return self._today()

    def _local_day(self, timestamp) -> str:
        """Local date of a UTC timestamp written by SQLite; today when it is missing or unreadable."""
        try:
            ts = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return self._today()
        ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(self.tz or timezone.utc).date().isoformat()

    # -- set-up and reconciliation ---------------------------------------------------------------------------------

    def history_ready(self) -> bool:
        return self._meta("history_schema") is not None

    def require_history(self) -> None:
        if not self.history_ready():
            raise HistoryUnavailable("fact history is not set up in this database yet: the main agent sets it up "
                                     "the first time it opens the store (subagents and cron jobs only read)")

    def _tombstone_key(self) -> bytes:
        key = self._meta("tombstone_key")
        if key is None:
            key = secrets.token_hex(16)
            self._write("INSERT OR IGNORE INTO meta (key, value) VALUES ('tombstone_key', ?)", (key,))
            key = self._meta("tombstone_key")
        return bytes.fromhex(key)

    def digest(self, text: str) -> str:
        return hmac.new(self._tombstone_key(), normalize(text).encode("utf-8"), hashlib.sha256).hexdigest()

    def _migrate_history(self) -> None:
        """Inside the migration transaction: tombstone key, versions for every fact, reconciliation, schema mark."""
        found = self._meta("history_schema")
        if found is not None and found.strip().isdigit() and int(found) > HISTORY_SCHEMA_VERSION:
            raise HistoryTooNew(f"this database was written by a newer holographic (history schema {found}, this "
                                f"version knows {HISTORY_SCHEMA_VERSION}): update the plugin; not opened for writing")
        self._tombstone_key()
        self._reconcile_history()
        # The mark says what the file holds, not who opened it: a file without merges keeps 1 (an older release
        # still writes it; a downgrade costs nothing), one a merge raised to 2 keeps 2 (merge_facts stamps it).
        if found is None or not found.strip().isdigit() or int(found) < _HISTORY_SCHEMA_BASE:
            self._set_meta("history_schema", str(_HISTORY_SCHEMA_BASE))

    def _reconcile_history(self) -> None:
        """Bring the versions in line with ``facts`` after an older version of the plugin wrote the file (see
        docs/history.md, Migration). Three indexed queries; nothing to do is the normal case. 0.19.0: a keep an older
        release forgot (a tombstone, or neither a row nor versions) takes its merged copies with it (_reconcile_merges)."""
        missing = self._conn.execute(
            "SELECT f.fact_id, f.content, f.category, f.tags, COALESCE(f.updated_at, f.created_at) AS ts FROM facts f "
            "WHERE NOT EXISTS (SELECT 1 FROM fact_versions v WHERE v.fact_id = f.fact_id AND v.status = 'current')").fetchall()
        for f in missing:
            self._insert_version(f["fact_id"], f["content"], f["category"], f["tags"], valid_from=None,
                                 source={"via": "legacy"}, recorded_at=f["ts"])
        drifted = self._conn.execute(
            "SELECT f.fact_id, f.content, f.category, f.tags, COALESCE(f.updated_at, f.created_at) AS ts, f.trust_score, "
            "v.version_id, v.recorded_on FROM facts f JOIN fact_versions v ON v.fact_id = f.fact_id AND v.status = 'current' "
            "WHERE f.content != v.content").fetchall()
        for f in drifted:  # an older `update content=`: the old text held until some point before the update
            bound = max(f["recorded_on"], self._local_day(f["ts"]))
            self._close_version(f["version_id"], "changed", valid_to=bound, exact=False, trust=f["trust_score"])
            new = self._insert_version(f["fact_id"], f["content"], f["category"], f["tags"], valid_from=None,
                                       source={"via": "legacy"}, recorded_at=f["ts"], supersedes=f["version_id"])
            self._write("UPDATE fact_versions SET superseded_by = ? WHERE version_id = ?", (new, f["version_id"]))
        self._write("UPDATE fact_versions SET tags = (SELECT tags FROM facts WHERE fact_id = fact_versions.fact_id), "
                    "category = (SELECT category FROM facts WHERE fact_id = fact_versions.fact_id) "
                    "WHERE status = 'current' AND EXISTS (SELECT 1 FROM facts f WHERE f.fact_id = fact_versions.fact_id "
                    "AND (f.tags IS NOT fact_versions.tags OR f.category IS NOT fact_versions.category))")
        gone = [r[0] for r in self._conn.execute(
            "SELECT fact_id FROM fact_versions v WHERE status = 'current' "
            "AND NOT EXISTS (SELECT 1 FROM facts f WHERE f.fact_id = v.fact_id)")]
        for fact_id in gone:  # an older `remove` or `supersedes`: the user wanted it gone, history included
            self._erase([fact_id, *self._merge_group(fact_id)])  # with the facts merged into it (0.19.0)
        # Always, not only when nothing above was gone: both kinds of deletion can wait for the same open (review 2)
        cascaded = self._reconcile_merges()
        if gone or cascaded:
            self._mark_scrub()

    def _reconcile_merges(self) -> bool:
        """0.19.0, after the step above: a merged copy whose keep is gone for good — forgotten (a tombstone: by an
        older writer that was open before the file's mark rose, or by this version's forget of a keep that the step
        above erased), or without a row and without versions — is erased too, with a tombstone of its own, as
        forget of the keep would have done. A merge row whose copy an older writer already forgot (its versions are
        gone) is dropped without a word. True when something was erased (the file wants a scrub)."""
        if not self._merges_ready():
            return False
        orphans = [r[0] for r in self._conn.execute(
            "SELECT from_fact_id FROM fact_merges m WHERE NOT EXISTS (SELECT 1 FROM fact_versions v WHERE v.fact_id = "
            "m.from_fact_id)")]
        for from_id in orphans:
            self._write("DELETE FROM fact_merges WHERE from_fact_id = ?", (from_id,))
        dead = [r[0] for r in self._conn.execute(
            "SELECT DISTINCT into_fact_id FROM fact_merges m WHERE EXISTS (SELECT 1 FROM forgotten t WHERE t.fact_id = "
            "m.into_fact_id) OR (NOT EXISTS (SELECT 1 FROM facts f WHERE f.fact_id = m.into_fact_id) AND NOT EXISTS "
            "(SELECT 1 FROM fact_versions v WHERE v.fact_id = m.into_fact_id))")]
        erased = False
        for keep in dead:
            copies = self._merge_group(keep)
            if copies:
                self._erase(copies)
                erased = True
            self._write("DELETE FROM fact_merges WHERE into_fact_id = ?", (keep,))
        return erased

    def _merges_ready(self) -> bool:
        """Has a writer of 0.19.0 created fact_merges? A reader may open a file no such writer has touched."""
        if not self._entry.get("merges"):
            self._entry["merges"] = bool(self._one(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'fact_merges'"))
        return self._entry["merges"]

    def _merge_group(self, keep: int) -> list[int]:
        """The ids merged into *keep* (chains are rebound to the root, so one level is the whole group)."""
        if not self._merges_ready():
            return []
        return [r[0] for r in self._conn.execute(
            "SELECT from_fact_id FROM fact_merges WHERE into_fact_id = ? ORDER BY from_fact_id", (keep,))]

    def merge_target(self, fact_id: int):
        """The fact_merges row of a copy (sqlite3.Row with into_fact_id, into_version_id, merged_at), or None."""
        if not self._merges_ready():
            return None
        return self._one("SELECT from_fact_id, into_fact_id, into_version_id, merged_at FROM fact_merges "
                         "WHERE from_fact_id = ?", (fact_id,))

    # -- version rows ----------------------------------------------------------------------------------------------

    def _insert_version(self, fact_id: int, content: str, category, tags, *, valid_from: str | None,
                        source: dict | None = None, recorded_at: str | None = None, supersedes: int | None = None,
                        status: str = "current") -> int:
        seq = (self._one("SELECT MAX(seq) FROM fact_versions WHERE fact_id = ?", (fact_id,))[0] or 0) + 1
        recorded_at = str(recorded_at or self._now())
        src = source or {}
        values = [src.get(f) if src.get(f) not in ("", None) else None for f in SOURCE_FIELDS]
        values = [str(v) if v is not None else None for v in values]
        cur = self._write(
            f"INSERT INTO fact_versions (fact_id, seq, content, category, tags, valid_from, status, recorded_at, recorded_on, "
            f"supersedes, {', '.join(_SOURCE_COLUMNS)}) VALUES ({', '.join('?' * (10 + len(_SOURCE_COLUMNS)))})",
            (fact_id, seq, content, category or "general", tags or "", valid_from, status, recorded_at,
             self._local_day(recorded_at), supersedes, *values))
        return int(cur.lastrowid)

    def _close_version(self, version_id: int, status: str, *, valid_to: str | None = None, exact: bool = True,
                       trust: float | None = None) -> None:
        self._write("UPDATE fact_versions SET status = ?, valid_to = COALESCE(?, valid_to), valid_to_exact = ?, "
                    "closed_at = ?, trust_score = COALESCE(?, trust_score) WHERE version_id = ?",
                    (status, valid_to, 1 if exact else 0, self._now(), trust, version_id))

    def _current_version(self, fact_id: int):
        return self._one("SELECT * FROM fact_versions WHERE fact_id = ? AND status = 'current'", (fact_id,))

    def _head_version(self, fact_id: int):
        """The newest version that is not a correction: current, or ended for a fact that left the present. A revert
        leaves a corrected version above it, so the highest seq is not the head."""
        return self._one("SELECT * FROM fact_versions WHERE fact_id = ? AND status != 'corrected' ORDER BY seq DESC "
                         "LIMIT 1", (fact_id,))

    def _version(self, version_id):
        return self._one("SELECT * FROM fact_versions WHERE version_id = ?", (version_id,)) if version_id else None

    def _replace_current(self, fact_id: int, closing: str, content: str, category, tags, *, valid_from: str | None,
                         source: dict | None, valid_to: str | None = None, exact: bool = True,
                         trust: float | None = None) -> int:
        """Close the current version as *closing* and make a new current one; returns the new version's seq."""
        cur = self._current_version(fact_id)
        if cur is None:  # a writer of 0.8.0 keeps every fact versioned; an older one is reconciled before writes
            raise KeyError(f"fact_id {fact_id} has no current version")
        self._close_version(cur["version_id"], closing, valid_to=valid_to, exact=exact, trust=trust)
        new = self._insert_version(fact_id, content, category, tags, valid_from=valid_from, source=source,
                                   supersedes=cur["version_id"])
        self._write("UPDATE fact_versions SET superseded_by = ? WHERE version_id = ?", (new, cur["version_id"]))
        self._record_version_names(cur["version_id"], fact_id, new)  # the links are still the closed version's
        self._fix_predecessor(new)
        return int(self._one("SELECT seq FROM fact_versions WHERE version_id = ?", (new,))[0])

    def _ended_before(self, version) -> dict | None:
        """The ended version *version* resumed (corrected ones skipped), or None when it did not resume one."""
        link = version
        while link is not None and link["supersedes"]:
            prev = self._version(link["supersedes"])
            if prev is None or prev["status"] != "corrected":
                return prev if prev is not None and prev["status"] == "ended" else None
            link = prev
        return None

    def _fix_predecessor(self, version_id: int) -> None:
        """Keep the end of the version that *version_id* (now current) replaced in step with its start: the version
        it replaced, skipping corrected ones, ends where it begins (exactly), or at the day the change was recorded
        (an upper bound) when its start is unknown."""
        current = self._version(version_id)
        link, change_point = current, None
        while link is not None and link["supersedes"]:
            prev = self._version(link["supersedes"])
            if prev is None:
                return
            if prev["status"] != "corrected":
                if prev["status"] != "changed":
                    return
                change_point = self._version(prev["superseded_by"]) or link
                break
            link = prev
        else:
            return
        if current["valid_from"]:
            end, exact = current["valid_from"], True
        else:
            end, exact = change_point["recorded_on"], False
        if exact and prev["valid_from"] and end <= prev["valid_from"]:  # an upper bound may equal the start
            raise ValueError(f"the previous version (#{prev['seq']}) started on {prev['valid_from']}: the date would leave "
                             "it no time at all; correct that version's date first, or revert")
        self._write("UPDATE fact_versions SET valid_to = ?, valid_to_exact = ? WHERE version_id = ?",
                    (end, 1 if exact else 0, prev["version_id"]))

    # -- what happened to a fact -----------------------------------------------------------------------------------

    def change_fact(self, fact_id: int, content: str, *, valid_from=None, category: str | None = None,
                    tags: str | None = None, entities=(), source: dict | None = None) -> dict:
        """The fact changed in the world: the current version is closed as `changed` and *content* becomes the
        current version of the same fact. An ended fact holds again in this new form. If another current fact
        already says exactly *content*, this one is ended and that one returned."""
        content = (content or "").strip()
        if not content:
            raise ValueError("content must not be empty")
        self.require_history()
        today = self._today()
        start = parse_day(valid_from, field="valid_from", today=today)
        with self._tx():
            self._reconcile_history()
            fact = self._one("SELECT * FROM facts WHERE fact_id = ?", (fact_id,))
            if fact is None:
                return self._resume(fact_id, content, start, category, tags, entities, source)
            cur = self._current_version(fact_id)
            if content == fact["content"]:
                raise ValueError("this is already what the fact says: nothing changed. A wrong start date is fixed "
                                 "with correct {fact_id, valid_from}")
            if start and cur["valid_from"] and start <= cur["valid_from"]:
                raise ValueError(f"valid_from {start} is not after the current version's start ({cur['valid_from']}): the "
                                 "current text never held then. If it was simply wrong, use correct")
            end, exact = (start, True) if start else (today, False)
            other = self._one("SELECT fact_id FROM facts WHERE content = ? AND fact_id != ?", (content, fact_id))
            if other is None:  # the text of a copy merged into another fact: that fact says it (0.19.0)
                other = self._merged_head(content, fact_id)
            if other is not None:  # what it changed into is already a fact of its own
                self._end(fact, cur, end, exact)
                self._declare(int(other["fact_id"]), entities)  # names given with it still count (0.17.0)
                return {"fact_id": int(other["fact_id"]), "status": "exists", "ended": fact_id}
            new_category = category or fact["category"]
            new_tags = fact["tags"] if tags is None else tags
            seq = self._replace_current(fact_id, "changed", content, new_category, new_tags, valid_from=start,
                                        source=source, valid_to=end, exact=exact, trust=fact["trust_score"])
            # A changed fact has not earned the old text's feedback (as `supersedes` did by making a new fact), nor
            # its confirmations: they confirmed the old claim (review 2026-10-15). It starts at its origin's trust.
            self._rewrite_fact(fact_id, content, new_category, new_tags, entities=entities, old_category=fact["category"],
                               reset_trust=True, trust=self.start_trust((source or {}).get("origin")))
            self._drop_confirmations(fact_id)
            return {"fact_id": fact_id, "status": "changed", "version": seq,
                    "previous_until": end if exact else f"before {end}"}

    def _resume(self, fact_id, content, start, category, tags, entities, source) -> dict:
        last = self._head_version(fact_id)
        if last is None or last["status"] != "ended":
            raise NotFound(f"fact_id {fact_id} not found")
        _check_after_end(start, last)
        other = self._one("SELECT fact_id FROM facts WHERE content = ?", (content,))
        if other is None:
            other = self._merged_head(content, fact_id)
        if other is not None:  # what it holds as again is already a fact of its own
            self._declare(int(other["fact_id"]), entities)  # names given with it still count (0.17.0)
            return {"fact_id": int(other["fact_id"]), "status": "exists"}
        new = self._insert_version(fact_id, content, category or last["category"], last["tags"] if tags is None else tags,
                                   valid_from=start, source=source, supersedes=last["version_id"])
        self._write("UPDATE fact_versions SET superseded_by = ? WHERE version_id = ?", (new, last["version_id"]))
        row = self._version(new)
        start_trust = self.start_trust((source or {}).get("origin"))
        self._insert_fact_row(fact_id, row, trust=self.default_trust if start_trust is None else start_trust,
                              entities=entities)
        self._drop_confirmations(fact_id)  # they confirmed what it said before it ended
        return {"fact_id": fact_id, "status": "resumed", "version": int(row["seq"])}

    def end_fact(self, fact_id: int, *, valid_to=None) -> dict:
        """The fact stopped holding and nothing replaces it: the current version is closed as `ended` and the fact
        leaves the current store; its history stays."""
        self.require_history()
        today = self._today()
        end = parse_day(valid_to, field="valid_to", today=today)
        with self._tx():
            self._reconcile_history()
            fact = self._one("SELECT * FROM facts WHERE fact_id = ?", (fact_id,))
            if fact is None:
                last = self._head_version(fact_id)
                if last is not None and last["status"] == "ended":
                    raise ValueError(f"fact #{fact_id} already ended ({last['valid_to']})")
                raise NotFound(f"fact_id {fact_id} not found")
            cur = self._current_version(fact_id)
            if end and cur["valid_from"] and end <= cur["valid_from"]:
                raise ValueError(f"valid_to {end} is not after the fact's start ({cur['valid_from']})")
            self._end(fact, cur, end or today, end is not None)
            return {"fact_id": fact_id, "status": "ended", "valid_to": end or today, "exact": end is not None}

    def _end(self, fact, cur, end: str, exact: bool) -> None:
        self._close_version(cur["version_id"], "ended", valid_to=end, exact=exact, trust=fact["trust_score"])
        self._delete_fact_row(fact["fact_id"], fact["category"])

    def correct_fact(self, fact_id: int, content: str | None = None, *, valid_from=None, entities=(),
                     source: dict | None = None, tags: str | None = None) -> dict:
        """It was wrong: the current version is closed as `corrected` (never returned as the past) and the corrected
        text and/or start date become the current version. *tags*: the corrected version's tags (None keeps the
        fact's; issue #73: keywords given with a correction were dropped)."""
        today = self._today()
        start = parse_day(valid_from, field="valid_from", today=today) if valid_from not in (None, "") else None
        text = (content or "").strip() if content is not None else None
        if text is not None and not text:
            raise ValueError("content must not be empty")
        self.require_history()
        with self._tx():
            self._reconcile_history()
            fact = self._one("SELECT * FROM facts WHERE fact_id = ?", (fact_id,))
            if fact is None:
                last = self._head_version(fact_id)
                if last is not None and last["status"] == "ended":
                    raise ValueError(f"fact #{fact_id} has ended; revert the end first, then correct it")
                raise NotFound(f"fact_id {fact_id} not found")
            cur = self._current_version(fact_id)
            if start is not None and start != cur["valid_from"]:  # a resume's start may not reach into its end
                _check_after_end(start, self._ended_before(cur))
            new_text = text if text is not None else fact["content"]
            new_start = start if start is not None else cur["valid_from"]
            if new_text == fact["content"]:  # the text stays: names given with it still count (0.17.0)
                self._declare(fact_id, entities)
            if new_text == fact["content"] and new_start == cur["valid_from"]:
                out = {"fact_id": fact_id, "status": "unchanged"}
                if entities:  # the text and date stay, the names were linked all the same (review 0.17.0, Fable)
                    out["entities_linked"] = [n for n in entities if isinstance(n, str) and n.strip()]
                return out
            source = self._with_origin(source, cur)
            new_tags = fact["tags"] if tags is None else tags
            seq = self._replace_current(fact_id, "corrected", new_text, fact["category"], new_tags,
                                        valid_from=new_start, source=source, trust=fact["trust_score"])
            if new_text != fact["content"]:
                # A new text is a new claim: what the wrong one earned (helpful marks, recalls, confirmations) is not
                # its; it starts at its origin's trust, as a change does (F1 research, 2026-10-29). The wrong text's
                # trust stays in its closed version: a revert brings it back. A new date alone keeps everything.
                self._rewrite_fact(fact_id, new_text, fact["category"], new_tags, entities=entities,
                                   old_category=fact["category"], reset_trust=True,
                                   trust=self.start_trust((source or {}).get("origin")))
                self._drop_confirmations(fact_id)
            elif new_tags != fact["tags"]:  # a new date alone, with keywords: the current fact and its index take them
                self._write("UPDATE facts SET tags = ? WHERE fact_id = ?", (new_tags, fact_id))
            return {"fact_id": fact_id, "status": "corrected", "version": seq}

    def revert_fact(self, fact_id: int, screen=None) -> dict:
        """Undo the last change, correction or end: the last version is closed as `corrected` and the one before it
        is current again (an ended fact is current again). *screen*: why a text must not be current again, or None
        (the provider's write screen: an older version may hold what later rules refuse; Codex review, #66380)."""
        self.require_history()
        with self._tx():
            self._reconcile_history()
            fact = self._one("SELECT * FROM facts WHERE fact_id = ?", (fact_id,))
            last = self._head_version(fact_id)
            if last is None:
                raise NotFound(f"fact_id {fact_id} not found")
            if fact is None:  # undo an end
                if last["status"] != "ended":
                    raise NotFound(f"fact_id {fact_id} not found")
                if self._one("SELECT 1 FROM facts WHERE content = ?", (last["content"],)):
                    raise ValueError("another current fact already says this; forget one of them instead")
                self._screened(last["content"], screen)
                self._write("UPDATE fact_versions SET status = 'current', valid_to = NULL, valid_to_exact = 1, "
                            "closed_at = NULL, superseded_by = NULL WHERE version_id = ?", (last["version_id"],))
                self._insert_fact_row(fact_id, self._version(last["version_id"]),
                                      trust=last["trust_score"] if last["trust_score"] is not None else self.default_trust)
                return {"fact_id": fact_id, "status": "reverted", "current_version": int(last["seq"])}
            cur = self._current_version(fact_id)
            prev = self._version(cur["supersedes"])
            if prev is None:
                if self._one("SELECT 1 FROM fact_versions WHERE fact_id = ? AND version_id != ?",
                             (fact_id, cur["version_id"])):  # the other versions were reverted away (corrected)
                    raise ValueError(f"fact #{fact_id}: nothing to revert; version {cur['seq']} is the one before "
                                     "the reverted ones, and a revert is not undone by another revert (change it "
                                     "again instead)")
                raise ValueError(f"fact #{fact_id} has a single version: nothing to revert (forget deletes it)")
            self._close_version(cur["version_id"], "corrected", trust=fact["trust_score"])
            if prev["status"] == "ended":  # undo a resume: the fact is ended again
                self._delete_fact_row(fact_id, fact["category"])
                return {"fact_id": fact_id, "status": "reverted", "ended": prev["valid_to"]}
            if self._one("SELECT 1 FROM facts WHERE content = ? AND fact_id != ?", (prev["content"], fact_id)):
                raise ValueError("another current fact already says what the previous version said; forget one of them")
            self._screened(prev["content"], screen)
            self._write("UPDATE fact_versions SET status = 'current', valid_to = NULL, valid_to_exact = 1, closed_at = NULL, "
                        "superseded_by = NULL WHERE version_id = ?", (prev["version_id"],))
            self._fix_predecessor(prev["version_id"])
            trust = prev["trust_score"] if prev["trust_score"] is not None else fact["trust_score"]
            # The names the version had come back (version_names); a version closed before this release: names only
            # the undone text gave go with it (_relink_fact)
            self._rewrite_fact(fact_id, prev["content"], prev["category"], prev["tags"], old_category=fact["category"],
                               trust=trust, restore=self._take_version_names(prev["version_id"], cur["version_id"]))
            return {"fact_id": fact_id, "status": "reverted", "current_version": int(prev["seq"])}

    @staticmethod
    def _screened(content: str, screen) -> None:
        """Raise ValueError when *screen* refuses the text a revert would make current again."""
        problem = screen(content) if screen is not None else None
        if problem:
            raise ValueError(f"not reverted: the earlier version {problem}")

    def forget_fact(self, fact_id: int) -> dict:
        """The user asked to forget it: the fact and every version are deleted for good (see docs/history.md,
        Forget means gone) and a tombstone without text is kept. Forgetting it again retries a file rewrite that
        another process blocked last time. 0.19.0: forgetting a fact others were merged into forgets them too (a
        tombstone each: they are the same fact; `merged_copies` lists them); forgetting a merged copy erases its own
        history only, and the fact it was merged into keeps saying the same (`merged_into`)."""
        self.require_history()
        with self._tx():
            self._reconcile_history()
            known = self._one("SELECT 1 FROM facts WHERE fact_id = ?", (fact_id,)) or \
                self._one("SELECT 1 FROM fact_versions WHERE fact_id = ?", (fact_id,))
            already = not known and self._one("SELECT 1 FROM forgotten WHERE fact_id = ?", (fact_id,)) is not None
            if not known and not already:
                raise NotFound(f"fact_id {fact_id} not found")
            versions, copies, into = 0, [], None
            if not already:
                copies = self._merge_group(fact_id)
                target = self.merge_target(fact_id)
                into = int(target["into_fact_id"]) if target is not None else None
                versions = self._erase([fact_id, *copies])
                self._mark_scrub()
        result = {"fact_id": fact_id, "forgotten": True, "versions": versions,
                  "scrub": "done" if self.scrub() else "pending"}
        if already:
            result["already"] = True
        if copies:
            result["merged_copies"] = copies
        if into is not None:
            result["merged_into"] = into
        return result

    def _erase(self, fact_ids) -> int:
        """Delete the fact(s), their versions and links, rebuild both indexes, write a tombstone for each id (in a
        transaction). A list (0.19.0): a keep with its merged copies, in one pass over the indexes."""
        ids = [int(fact_ids)] if isinstance(fact_ids, int) else [int(i) for i in fact_ids]
        if not ids:
            return 0
        if len(ids) > _ERASE_BATCH:  # a large group: within SQLite's bound-parameter limit (one transaction still)
            return sum(self._erase(ids[i:i + _ERASE_BATCH]) for i in range(0, len(ids), _ERASE_BATCH))
        marks = ",".join("?" * len(ids))
        texts: dict[int, list[str]] = {i: [] for i in ids}
        for r in self._conn.execute(f"SELECT fact_id, content FROM fact_versions WHERE fact_id IN ({marks})", ids):
            texts[r["fact_id"]].append(r["content"])
        rows = {r["fact_id"]: r for r in self._conn.execute(
            f"SELECT fact_id, content, category FROM facts WHERE fact_id IN ({marks})", ids)}
        named = [r[0] for r in self._conn.execute(
            f"SELECT DISTINCT entity_id FROM fact_entities WHERE fact_id IN ({marks})", ids)]
        for fact_id, row in rows.items():
            texts[fact_id].append(row["content"])
        # the "Y" of its "X aka Y", read while the text is still here (0.16.0)
        said = self._aka_alias_keys(t for ts in texts.values() for t in ts)
        for fact_id, row in rows.items():
            self._delete_fact_row(fact_id, row["category"])
        # A name only the forgotten fact gave goes even if it has aliases (_drop_orphan_entities keeps those).
        for entity_id in named:
            if not self._one("SELECT 1 FROM fact_entities WHERE entity_id = ?", (entity_id,)):
                self._drop_entity_row(entity_id)
                self._bump()  # the cached index of known names must not hand out the deleted id (review 0.16.0)
        if said:  # an alias of unknown origin that only this fact said goes with it, as one from its text does
            self._prune_aka_aliases(legacy_keys=said)
            self._drop_orphan_entities()
        versions = self._write(f"DELETE FROM fact_versions WHERE fact_id IN ({marks})", ids).rowcount
        if self._entry.get("migrated"):  # the row may be gone already (another writer): its vector goes regardless
            self._write(f"DELETE FROM fact_embeddings WHERE fact_id IN ({marks})", ids)
        # Also during the migration (the cascade of _reconcile_merges runs there, after the orphan sweep): who
        # confirmed a forgotten fact goes with it (review 2 of #36)
        if self._confirmations_ready():
            self._write(f"DELETE FROM fact_confirmations WHERE fact_id IN ({marks})", ids)
        if self._merges_ready():  # as a copy, and as a keep (its copies are in *ids* when the caller forgets it)
            self._write(f"DELETE FROM fact_merges WHERE from_fact_id IN ({marks}) OR into_fact_id IN ({marks})",
                        [*ids, *ids])
        now = self._now()
        for fact_id in ids:
            digests = sorted({self.digest(t) for t in texts[fact_id]})
            old = self._one("SELECT digests FROM forgotten WHERE fact_id = ?", (fact_id,))
            if old is not None:
                digests = sorted(set(digests) | set(json.loads(old["digests"] or "[]")))
            self._write("INSERT INTO forgotten (fact_id, forgotten_at, digests) VALUES (?, ?, ?) ON CONFLICT(fact_id) "
                        "DO UPDATE SET forgotten_at = excluded.forgotten_at, digests = excluded.digests",
                        (fact_id, now, json.dumps(digests)))
        self._write("DELETE FROM memory_banks")  # superpositions that included the fact
        # An FTS5 delete only marks the row; the words stay in older index segments until the index is rebuilt.
        self._write("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
        self._write("INSERT INTO fact_versions_fts(fact_versions_fts) VALUES ('rebuild')")
        return max(versions, 0)

    # -- merge (0.19.0) --------------------------------------------------------------------------------------------

    def merge_facts(self, keep: int, from_ids, *, tags=None, expect: dict | None = None,
                    session_only: str | None = None) -> dict:
        """Two or more current facts say the same thing: the others (*from_ids*) become part of *keep*. Each copy's
        current version is closed as `merged` (valid time untouched; as_of clips it at the merge date), a fact_merges
        row says where it went, its entity links move to keep, its row leaves the current projection (with its FTS
        entry, vector and bank), its history and confirmations stay under its own id. keep: its own trust, the
        counts of the group added (an aggregate of the group's past), *tags* when given (the provider's keyword
        union), origin and category untouched. Copies merged earlier into one of the *from_ids* are rebound to
        keep (no chains, no cycles). One transaction: an error on any copy leaves nothing written. *expect*
        ({fact_id: version_id}): the current version of each named participant must be that one (ChangedSince
        otherwise). The same ids again (all already part of keep) is a success without changes. *session_only*: every
        copy's current version must have been stored by that session (NotOwned otherwise: a scheduled job). *tags* may
        be a function (keep's tags, [each copy's tags]) -> the keep's new tags or None: it runs inside the transaction,
        after reconciliation, so a tag or keyword another writer gave keep a moment earlier is part of what it joins.
        Copies merged into keep's current version follow keep's new current one; copies merged into an earlier
        version of the fact merged now keep that link (they said its claim of then: past_version, review 2)."""
        keep = int(keep)
        ids = [int(i) for i in (from_ids or ())]
        if not ids:
            raise ValueError("merge needs the ids of the facts to merge into this one in 'from'")
        if len(ids) > _MERGE_MAX:
            raise ValueError(f"merge takes at most {_MERGE_MAX} facts at a time (got {len(ids)})")
        if len(set(ids)) != len(ids):
            raise ValueError("'from' names the same fact twice")
        if keep in ids:
            raise ValueError(f"fact #{keep} cannot be merged into itself")
        self.require_history()
        self._require_writer()
        with self._tx():
            self._reconcile_history()
            target = self.merge_target(keep)
            if target is not None:
                raise ValueError(f"fact #{keep} was merged into #{target['into_fact_id']}; merge into that one")
            fact = self._one("SELECT * FROM facts WHERE fact_id = ?", (keep,))
            if fact is None:
                self._refuse_absent(keep, "merge into")
            copies, done = [], []
            for fact_id in ids:
                row = self._one("SELECT * FROM facts WHERE fact_id = ?", (fact_id,))
                if row is not None:
                    copies.append(row)
                    continue
                target = self.merge_target(fact_id)
                if target is not None and int(target["into_fact_id"]) == keep:
                    done.append(fact_id)  # the same merge again: nothing to add
                    continue
                if target is not None:
                    raise ValueError(f"fact #{fact_id} was merged into #{target['into_fact_id']} already; forget it "
                                     "there, or merge that fact")
                self._refuse_absent(fact_id, "merge")
            if session_only is not None and copies:
                sources = self.current_sources([r["fact_id"] for r in copies])
                foreign = [int(r["fact_id"]) for r in copies
                           if (sources.get(int(r["fact_id"])) or {}).get("session_id") != session_only]
                if foreign:
                    raise NotOwned(foreign)
            if callable(tags):  # read here, in the transaction: what is there now, not what the caller read before
                tags = tags(fact["tags"] or "", [r["tags"] or "" for r in copies])
            versions = {int(r["fact_id"]): self._current_version(r["fact_id"]) for r in (fact, *copies)}
            if expect:
                seen = {int(f): (None if v is None else int(v)) for f, v in expect.items()}
                current = {}
                for f in seen:  # a copy merged into keep already: its head is the merged version the caller saw
                    head = versions[f] if f in versions else self._head_version(f)
                    current[f] = int(head["version_id"]) if head is not None else None
                if any(current[f] != seen[f] for f in seen):
                    raise ChangedSince({f: current[f] for f in seen if current[f] != seen[f]})
            if not copies:
                return {"merged": [], "into": keep, "already": done, "counts_added": {"retrieval": 0, "helpful": 0},
                        "merged_confirmations": self._merged_confirmations(keep)}
            now = self._now()
            marks = ",".join("?" * len(copies))
            # Facts merged earlier into a copy now belong to keep. Those merged into the copy's current version said
            # its current claim, which is keep's now; those merged into an earlier version (the copy changed since)
            # said a past claim and keep pointing at it — rebound, they would confirm keep's claim by old words.
            heads = [int(versions[int(r["fact_id"])]["version_id"]) for r in copies if versions[int(r["fact_id"])]]
            self._write(f"UPDATE fact_merges SET into_fact_id = ?, into_version_id = CASE WHEN into_version_id IN "
                        f"({','.join('?' * len(heads))}) THEN ? ELSE into_version_id END "
                        f"WHERE into_fact_id IN ({marks})",
                        (keep, *heads, int(versions[keep]["version_id"]), *[r["fact_id"] for r in copies]))
            retrieval = helpful = 0
            moved: list[int] = []  # the copies' names, to be keep's
            for row in copies:
                fact_id = int(row["fact_id"])
                cur = versions[fact_id]
                if cur is None:  # reconciled above: every current fact has one
                    raise KeyError(f"fact_id {fact_id} has no current version")
                self._close_version(cur["version_id"], "merged", trust=row["trust_score"])
                self._write("INSERT INTO fact_merges (from_fact_id, into_fact_id, into_version_id, merged_at) "
                            "VALUES (?, ?, ?, ?)", (fact_id, keep, int(versions[keep]["version_id"]), now))
                moved += [r[0] for r in self._conn.execute(
                    "SELECT entity_id FROM fact_entities WHERE fact_id = ?", (fact_id,)).fetchall()]
                retrieval += int(row["retrieval_count"] or 0)
                helpful += int(row["helpful_count"] or 0)
                # The row leaves the current projection (as _delete_fact_row does, the orphan sweep apart)
                if self._entry.get("migrated"):
                    self._write("DELETE FROM fact_embeddings WHERE fact_id = ?", (fact_id,))
                self._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
                self._write("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
                self._invalidate_bank(row["category"])
            # Aliases a copy's "X aka Y" gave go with its row (no current text says them), and dropping one links the
            # facts of that entity again from their text — so the copies' names are given to keep after that, and
            # only then do entities without facts go: the names only the copies had stay, as keep's.
            self._prune_aka_aliases()
            for entity_id in sorted(set(moved)):
                self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) SELECT ?, entity_id FROM "
                            "entities WHERE entity_id = ?", (keep, entity_id))
            self._drop_orphan_entities()
            sets = ["retrieval_count = retrieval_count + ?", "helpful_count = helpful_count + ?"]
            params: list = [retrieval, helpful]
            if tags is not None:
                sets.append("tags = ?")
                params.append(tags)
                self._write("UPDATE fact_versions SET tags = ? WHERE version_id = ?",
                            (tags, versions[keep]["version_id"]))
            self._write(f"UPDATE facts SET {', '.join(sets)} WHERE fact_id = ?", (*params, keep))
            self._compute_hrr_vector(keep, fact["content"])  # its roles changed with the links
            self._invalidate_bank(fact["category"])
            self._bump()
            found = self._meta("history_schema")
            if not (found and found.strip().isdigit() and int(found) >= HISTORY_SCHEMA_VERSION):
                self._set_meta("history_schema", str(HISTORY_SCHEMA_VERSION))
            return {"merged": [int(r["fact_id"]) for r in copies], "into": keep, "already": done,
                    "counts_added": {"retrieval": retrieval, "helpful": helpful},
                    "merged_confirmations": self._merged_confirmations(keep)}

    def _refuse_absent(self, fact_id: int, what: str) -> None:
        """Why a fact cannot take part in a merge: it ended, or there is no such fact."""
        last = self._head_version(fact_id)
        if last is not None and last["status"] == "ended":
            raise ValueError(f"fact #{fact_id} has ended ({last['valid_to']}); revert the end first to {what} it")
        raise NotFound(f"fact_id {fact_id} not found")

    def _merged_confirmations(self, keep: int) -> int:
        """How many confirmations the copies merged into *keep* hold (they stay under the copies' ids)."""
        if not self._merges_ready() or not self._confirmations_ready():
            return 0
        return int(self._one("SELECT COUNT(*) FROM fact_confirmations WHERE fact_id IN (SELECT from_fact_id FROM "
                             "fact_merges WHERE into_fact_id = ?)", (keep,))[0])

    def _merged_head(self, content: str, fact_id: int | None = None):
        """{fact_id} of the current fact a copy that said exactly *content* was merged into (a row with fact_id), or
        None; a copy of *fact_id* itself does not count (the fact may take its old text back)."""
        found = self.merged_match(content)
        if found is None or (fact_id is not None and found["fact_id"] == fact_id):
            return None
        return {"fact_id": found["fact_id"]}

    def merged_match(self, text: str) -> dict | None:
        """{fact_id: the keep, merged_from: the copy, merged_at} when exactly *text* is the last text of a fact
        merged into a current fact (the version with status 'merged'), else None. Only that version: a copy's
        earlier changed or ended text is a past version, a corrected one never held, and once the keep is gone
        nothing says the text. Only while the keep is still at the version it had when the copy was merged in
        (into_version_id): after a change or an end of the keep the copy's text is the keep's past claim
        (past_version), not its new one."""
        if not self.history_ready() or not self._merges_ready():
            return None
        row = self.read("SELECT m.into_fact_id, v.fact_id, m.merged_at FROM fact_versions v JOIN fact_merges m "
                        "ON m.from_fact_id = v.fact_id WHERE v.content = ? AND v.status = 'merged' AND EXISTS "
                        "(SELECT 1 FROM facts f WHERE f.fact_id = m.into_fact_id) AND EXISTS (SELECT 1 FROM fact_versions k WHERE "
                        "k.version_id = m.into_version_id AND k.status = 'current') ORDER BY v.version_id DESC LIMIT 1",
                        ((text or "").strip(),))
        if not row:
            return None
        return {"fact_id": int(row[0][0]), "merged_from": int(row[0][1]), "merged_at": row[0][2]}

    def _mark_scrub(self) -> None:
        """Ask for a file rewrite after text was deleted. Each mark is new (a random token), so a scrub clears only the
        mark it read: one set by another process meanwhile, for text the rewrite may have missed, stays (Codex review
        2026-10-30)."""
        self._set_meta("scrub_pending", secrets.token_hex(8))

    def scrub(self) -> bool:
        """Rewrite the file and empty the WAL so freed pages keep no copy of forgotten text. Best effort: another
        process holding the database can block it; then the mark stays and the next open or forget retries."""
        with self._lock:
            mark = self._meta("scrub_pending")
            if self._conn.in_transaction or mark is None:
                return mark is None
            # Without waiting on another process (busy timeout 0 for every statement here, as the recall counts do;
            # restored last): a reader blocks the checkpoint for as long as it reads, every provider operation waits
            # behind a forget, and a receipt write runs this on the search path, so "pending" must come back at once
            # rather than after the 10 s timeout — the closing checkpoint too (Codex review 2026-10-30).
            timeout = self._conn.execute("PRAGMA busy_timeout").fetchone()[0]
            self._conn.execute("PRAGMA busy_timeout = 0")
            try:
                self._conn.execute("VACUUM")
                busy = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if busy is not None and busy[0]:
                    return False
                # only the mark this rewrite served: a newer one asks for text written after the rewrite began
                self._conn.execute("DELETE FROM meta WHERE key = 'scrub_pending' AND value = ?", (mark,))
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # the cleared mark; text is already gone
                return True
            except sqlite3.Error as exc:
                logger.warning("holographic: scrub after forget deferred (%s)", exc)
                return False
            finally:
                self._conn.execute(f"PRAGMA busy_timeout = {int(timeout)}")

    # -- reads -----------------------------------------------------------------------------------------------------

    def forgotten_match(self, text: str) -> dict | None:
        """The tombstone whose versions include *text* (normalised), or None."""
        if not self.history_ready():
            return None
        row = self.read("SELECT fact_id, forgotten_at FROM forgotten WHERE digests LIKE ?", (f'%"{self.digest(text)}"%',))
        return dict(row[0]) if row else None

    def past_version(self, text: str, statuses=CLOSED) -> dict | None:
        """A closed version with exactly *text* (the newest), with whether its fact is still current."""
        if not self.history_ready():
            return None
        marks = ",".join("?" * len(statuses))
        row = self.read(f"SELECT v.fact_id, v.seq, v.status, v.valid_to, v.valid_to_exact, "
                        f"EXISTS (SELECT 1 FROM facts f WHERE f.fact_id = v.fact_id) AS is_current FROM fact_versions v "
                        f"WHERE v.content = ? AND v.status IN ({marks}) ORDER BY v.version_id DESC LIMIT 1",
                        ((text or "").strip(), *statuses))
        if row:
            return dict(row[0])
        if not self._merges_ready():
            return None
        # The last text of a merged copy whose keep has moved on (its version at the merge closed with one of
        # *statuses*): it said the keep's claim of then, so it is that version of the keep (0.19.0).
        row = self.read(f"SELECT k.fact_id, k.seq, k.status, k.valid_to, k.valid_to_exact, "
                        f"EXISTS (SELECT 1 FROM facts f WHERE f.fact_id = k.fact_id) AS is_current FROM fact_versions v "
                        f"JOIN fact_merges m ON m.from_fact_id = v.fact_id JOIN fact_versions k ON k.version_id = "
                        f"m.into_version_id WHERE v.content = ? AND v.status = 'merged' AND k.status IN ({marks}) "
                        f"ORDER BY v.version_id DESC LIMIT 1", ((text or "").strip(), *statuses))
        return dict(row[0]) if row else None

    def history(self, fact_id: int) -> dict:
        self.require_history()
        rows = self.read("SELECT * FROM fact_versions WHERE fact_id = ? ORDER BY seq", (fact_id,))
        if not rows:
            gone = self.read("SELECT forgotten_at FROM forgotten WHERE fact_id = ?", (fact_id,))
            if gone:
                return {"fact_id": fact_id, "forgotten": True, "forgotten_at": gone[0][0], "versions": []}
            fact = self.read("SELECT content, COALESCE(updated_at, created_at) FROM facts WHERE fact_id = ?",
                             (fact_id,))
            if fact:  # written by another program or an older release: no version yet (the next write adds it)
                return {"fact_id": fact_id, "current": True, "versions": [
                    {"version": 1, "content": fact[0][0], "status": "current", "valid_from": "unknown",
                     "valid_to": None, "recorded_at": fact[0][1], "source": "legacy (written outside this version; "
                     "its history starts with the next change)"}]}
            raise NotFound(f"fact_id {fact_id} not found")
        out = {"fact_id": fact_id, "current": any(r["status"] == "current" for r in rows),
               "versions": self._version_entries(rows)}
        target = self.merge_target(fact_id)
        if target is not None:  # a merged copy: where it went (0.19.0)
            out["merged_into"] = int(target["into_fact_id"])
            out["merged_at"] = target["merged_at"]
        copies = self._merge_group(fact_id)
        if copies:  # a keep: the histories of the facts merged into it, under their own ids
            merged_rows = self.read("SELECT m.merged_at, v.* FROM fact_merges m JOIN fact_versions v ON v.fact_id = "
                                    "m.from_fact_id WHERE m.into_fact_id = ? ORDER BY v.fact_id, v.seq", (fact_id,))
            by_copy: dict[int, list] = {}
            for r in merged_rows:
                by_copy.setdefault(int(r["fact_id"]), []).append(r)
            out["merged_from"] = [{"fact_id": copy, "merged_at": rows_of[0]["merged_at"],
                                   "versions": self._version_entries(rows_of)} for copy, rows_of in by_copy.items()]
            out["merged_confirmations"] = self._merged_confirmations(fact_id)
        return out

    @staticmethod
    def _version_entries(rows) -> list[dict]:
        seqs = {r["version_id"]: r["seq"] for r in rows}
        versions = []
        for r in rows:
            entry = {"version": r["seq"], "version_id": r["version_id"], "content": r["content"], "status": r["status"],
                     "valid_from": r["valid_from"] or "unknown", "valid_to": r["valid_to"],
                     "recorded_at": r["recorded_at"], "source": source_of(r) or "unknown"}
            if r["valid_to"] and not r["valid_to_exact"]:
                entry["valid_to"] = f"before {r['valid_to']}"
            if r["closed_at"]:
                entry["closed_at"] = r["closed_at"]
            if r["supersedes"] in seqs:
                entry["replaces_version"] = seqs[r["supersedes"]]
            if r["superseded_by"] in seqs:
                entry["replaced_by_version"] = seqs[r["superseded_by"]]
            versions.append(entry)
        return versions

    def merged_gone_by(self, day: str) -> list[int]:
        """version_ids of merged versions that no longer hold on *day* (merged on it or before, by local day): an
        as_of search leaves them out of its corpus before ranking, so they take no place in its pool."""
        if not self._merges_ready():
            return []
        rows = self.read("SELECT v.version_id, m.merged_at FROM fact_versions v JOIN fact_merges m ON "
                         "m.from_fact_id = v.fact_id WHERE v.status = 'merged'")
        return [int(r[0]) for r in rows if self._local_day(r[1]) <= day]

    def clip_merged(self, row) -> dict | None:
        """A version row as a dict, a merged one with its effective end (0.19.0): from the local day of the merge the
        truth about the claim is the keep's history, so a merged version holds on [valid_from, min(valid_to, that
        day)) — and `merged_into` names the keep. Rows with status 'merged' need the merge row to be read at all; a
        copy whose merge row is gone (its keep was forgotten by an older writer) is None (the next writer open erases
        it)."""
        out = dict(row)
        if row["status"] != "merged":
            return out
        merged_at = out.get("merged_at")
        if merged_at is None:
            target = self.merge_target(row["fact_id"])
            if target is None:
                return None
            merged_at, out["merged_into"] = target["merged_at"], int(target["into_fact_id"])
        merged_on = self._local_day(merged_at)
        if not out.get("valid_to") or merged_on < out["valid_to"]:
            out["valid_to"], out["valid_to_exact"] = merged_on, 1
        return out

    def versions_on(self, day: str, category: str | None = None, min_trust: float = 0.0, limit: int = 50) -> dict:
        """What held on *day* (`list as_of`): versions known to hold, and those that may have (undetermined). A
        merged version (0.19.0) holds until the day it was merged, under its own id, with `merged_into`."""
        self.require_history()
        merges = self._merges_ready()
        rows = self.read(
            "SELECT v.*, COALESCE(f.trust_score, v.trust_score, ?) AS trust"
            + (", m.merged_at, m.into_fact_id AS merged_into" if merges else "")
            + " FROM fact_versions v LEFT JOIN facts f ON f.fact_id = v.fact_id AND v.status = 'current' "
            + ("LEFT JOIN fact_merges m ON m.from_fact_id = v.fact_id AND v.status = 'merged' " if merges else "")
            + "WHERE v.status != 'corrected' AND (v.valid_from IS NULL OR v.valid_from <= ?) "
            "AND (v.valid_to IS NULL OR v.valid_to > ?)" + (" AND v.category = ?" if category else "")
            + " ORDER BY v.fact_id, v.seq", (self.default_trust, day, day, *([category] if category else [])))
        held, maybe = {}, {}
        for raw in rows:
            if raw["trust"] < min_trust:
                continue
            r = self.clip_merged(raw)
            if r is None:
                continue
            state, why = version_state(r, day)
            entry = {"fact_id": r["fact_id"], "version": r["seq"], "content": r["content"], "category": r["category"],
                     "valid_from": r["valid_from"] or "unknown", "valid_to": r["valid_to"], "status": r["status"]}
            if r["status"] == "merged":
                entry["merged_into"] = r["merged_into"]
            if state == "holds":
                held[r["fact_id"]] = entry
            elif state == "undetermined":
                maybe.setdefault(r["fact_id"], {**entry, "reason": why})
        undetermined = [e for fid, e in maybe.items() if fid not in held]
        return {"as_of": day, "facts": list(held.values())[:limit], "undetermined": undetermined[:limit]}

    def history_counts(self) -> dict:
        if not self.history_ready():
            return {}
        counts = {r[0]: r[1] for r in self.read("SELECT status, COUNT(*) FROM fact_versions GROUP BY status")}
        # A fact that ended for good: no current version, and not merged since (ended -> resumed -> merged, 0.19.0)
        ended = self.read("SELECT COUNT(DISTINCT fact_id) FROM fact_versions v WHERE status = 'ended' AND NOT EXISTS "
                          "(SELECT 1 FROM fact_versions w WHERE w.fact_id = v.fact_id AND w.status IN ('current', "
                          "'merged'))")[0][0]
        merged = self.read("SELECT COUNT(DISTINCT fact_id) FROM fact_versions WHERE status = 'merged'")[0][0]
        return {"past_versions": counts.get("changed", 0) + counts.get("ended", 0), "corrected": counts.get("corrected", 0),
                "ended_facts": ended, "merged_facts": merged, "forgotten": self.read("SELECT COUNT(*) FROM forgotten")[0][0]}

    def _source_columns(self) -> tuple:
        """The source columns this file has: a reader may open a file an older writer created, without src_origin."""
        cols = self._entry.get("source_columns")
        if cols is None or len(cols) < len(_SOURCE_COLUMNS):  # a writer elsewhere may have added it since
            have = {r[1] for r in self._conn.execute("PRAGMA table_info(fact_versions)")}
            cols = self._entry["source_columns"] = tuple(c for c in _SOURCE_COLUMNS if c in have)
        return cols

    def _version_names_ready(self) -> bool:
        if not self._entry.get("version_names"):  # with closed_by: a table without it is not one this code writes
            self._entry["version_names"] = {"closed_by", "d_fold"} <= {
                r[1] for r in self._conn.execute("PRAGMA table_info(version_names)")}
        return self._entry["version_names"]

    def _record_version_names(self, version_id: int, fact_id: int, closed_by: int) -> None:
        """Keep the names declared on the fact now as those of *version_id*, which version *closed_by* just closed
        (version_names): the empty-name row says the record exists, whatever delete_entity leaves of it later."""
        if not self._version_names_ready():
            return
        names: dict[str, str] = {"": ""}
        for r in self._conn.execute("SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
                                    "WHERE fe.fact_id = ? AND e.entity_type = 'declared' ORDER BY e.name", (fact_id,)):
            names.setdefault(entity_key(r[0]), r[0])
        self._write("DELETE FROM version_names WHERE version_id = ?", (version_id,))
        for key, name in names.items():
            forms = self._name_digest_forms(name) if key else (None, None, None)
            self._write("INSERT INTO version_names (version_id, name, name_key, closed_by, d_key, d_fold, d_old) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)", (version_id, name, key, closed_by, *forms))

    def _take_version_names(self, version_id: int, closed_by: int) -> list[str] | None:
        """The names declared on the fact while *version_id* was current, for a revert that undoes *closed_by*; the
        record is used up (the version is current again, and the next closing records it anew). None when there is
        no record that closing made: a version closed before this release, by an older one, or closed again by an
        older one after a revert (its record would be stale). A name delete_entity removed since is not given back."""
        if not self._version_names_ready():
            return None
        rows = self._conn.execute("SELECT name, name_key, closed_by FROM version_names WHERE version_id = ? "
                                  "ORDER BY name", (version_id,)).fetchall()
        self._write("DELETE FROM version_names WHERE version_id = ?", (version_id,))
        marker = [r for r in rows if not r["name_key"]]
        if not marker or int(marker[0]["closed_by"]) != int(closed_by):
            return None
        return [r["name"] for r in rows if r["name_key"] and not self._tombstoned(r["name"])]

    _COVERED = "name_key != '' AND (d_key IN ({m}) OR d_fold IN ({m}) OR d_old IN ({m}))"

    def _version_history_names(self, name: str) -> list[str]:
        """The spellings version_names keeps of *name* as its tombstone would cover them (the digests of
        _name_digests: "Jose" for "José", "Lukasz" for "Łukasz"), for a name no entity has any more."""
        if not entity_key(name or "") or not self._version_names_ready():
            return []
        digests = sorted(self._name_digests(name))
        marks = ",".join("?" * len(digests))
        return sorted({r[0] for r in self._conn.execute(
            "SELECT name FROM version_names WHERE " + self._COVERED.format(m=marks), digests * 3)})

    def _drop_version_names(self, names) -> int:
        """Delete what versions kept of *names*, every spelling their tombstones cover (the trigger does the same for a
        tombstone that is new; this also covers one written before). Returns how many rows went."""
        digests = sorted({d for n in names if entity_key(n or "") for d in self._name_digests(n)})
        if not digests or not self._version_names_ready():
            return 0
        marks = ",".join("?" * len(digests))
        return max(self._write("DELETE FROM version_names WHERE " + self._COVERED.format(m=marks), digests * 3).rowcount, 0)

    def _confirmations_ready(self) -> bool:
        if not self._entry.get("confirmations"):
            self._entry["confirmations"] = bool(self._one(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'fact_confirmations'"))
        return self._entry["confirmations"]

    def origins(self, fact_ids) -> dict[int, dict]:
        """{fact_id: {"origin": said | confirmed | inferred | None, "confirmations": n}} of current facts. A conclusion
        a person has confirmed since counts as confirmed; the version keeps what it was stored as."""
        ids = [int(i) for i in fact_ids]
        if not ids or not self.history_ready():
            return {}
        has_origin = "src_origin" in self._source_columns()
        counted = self._confirmations_ready()
        out = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            confirmations = {r[0]: r[1] for r in self.read(
                f"SELECT fact_id, COUNT(*) FROM fact_confirmations WHERE fact_id IN ({marks}) GROUP BY fact_id",
                chunk)} if counted else {}
            stored = {r[0]: r[1] for r in self.read(
                f"SELECT fact_id, src_origin FROM fact_versions WHERE status = 'current' AND fact_id IN ({marks})",
                chunk)} if has_origin else {}
            for fact_id in chunk:
                origin, n = stored.get(fact_id), confirmations.get(fact_id, 0)
                if n and origin not in ("said", "confirmed"):
                    origin = "confirmed"
                out[fact_id] = {"origin": origin if origin in ORIGINS else None, "confirmations": n}
        return out

    def start_trust(self, origin) -> float | None:
        """The trust a new or changed fact starts with by its origin (ORIGIN_TRUST), clamped to [0, 1]; None when the
        origin is not given (the default applies)."""
        if origin not in ORIGIN_TRUST:
            return None
        return max(0.0, min(1.0, float(self.default_trust) + ORIGIN_TRUST[origin]))

    def _with_origin(self, source: dict | None, version) -> dict | None:
        """A correction that names no origin keeps its version's: fixing a typo does not make a fact nobody's."""
        if (source or {}).get("origin") or version is None or "src_origin" not in version.keys() \
                or not version["src_origin"]:
            return source
        return {**(source or {}), "origin": version["src_origin"]}

    def _drop_confirmations(self, fact_id: int) -> None:
        """The confirmations of a fact whose claim changed (change, correct and update of the text, a resume): they
        were of the old claim."""
        if self._confirmations_ready():
            self._write("DELETE FROM fact_confirmations WHERE fact_id = ?", (fact_id,))

    def drop_orphan_confirmations(self) -> None:
        """Confirmations of facts an older version's forget erased (it deletes the versions, not these rows)."""
        if self._confirmations_ready():
            self._write("DELETE FROM fact_confirmations WHERE fact_id NOT IN (SELECT fact_id FROM fact_versions)")

    def stale_inferences(self, stale_days: int, limit: int) -> list[int]:
        """Fact ids whose current version is the agent's own conclusion, stored more than *stale_days* ago, that
        nobody has confirmed since: the oldest first, at most *limit* (review)."""
        if not self.history_ready() or "src_origin" not in self._source_columns():
            return []
        unconfirmed = ("AND NOT EXISTS (SELECT 1 FROM fact_confirmations c WHERE c.fact_id = v.fact_id) "
                       if self._confirmations_ready() else "")
        return [r[0] for r in self.read(
            "SELECT v.fact_id FROM fact_versions v JOIN facts f ON f.fact_id = v.fact_id WHERE v.status = 'current' "
            f"AND v.src_origin = 'inferred' AND julianday('now') - julianday(v.recorded_at) > ? {unconfirmed}"
            "ORDER BY v.recorded_at LIMIT ?", (stale_days, limit))]

    def record_confirmation(self, fact_id: int, source: dict | None, *, bump: float = 0.05) -> dict:
        """A person confirmed a current fact: one row per session — or per day where there is no session (a CLI
        call, a script), so repeating the call cannot pile rows up (review 2026-10-15). Trust rises by *bump* only
        for a person who has not confirmed this fact before: in a group chat one member repeating "yes, true"
        session after session raises it once (a known id; without one, once per session). Returns {fact_id,
        confirmed: a new row, confirmations, trust_score, origin}. Raises KeyError if the fact is not current."""
        src = source or {}
        session = str(src.get("session_id") or "") or f"day:{self._today()}"
        user = str(src.get("user_id") or "")
        with self._tx():
            row = self._one("SELECT trust_score FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                raise NotFound(f"fact_id {fact_id} not found")
            # The group (0.19.0): a person who confirmed a copy before it was merged in has confirmed this claim —
            # only copies merged into the keep's current version: after a change of the keep a copy's confirmations
            # were of the old claim, as the keep's own were (change clears those)
            group, of = "fact_id = ?", [fact_id]
            if self._merges_ready():  # a subquery, not a list: a group has no size limit
                group = ("(fact_id = ? OR fact_id IN (SELECT m.from_fact_id FROM fact_merges m JOIN fact_versions k ON "
                         "k.version_id = m.into_version_id WHERE m.into_fact_id = ? AND k.status = 'current'))")
                of = [fact_id, fact_id]
            again = bool(self._one(f"SELECT 1 FROM fact_confirmations WHERE {group} AND session_id = ?",
                                   (*of, session)))
            trust = row["trust_score"]
            if not again:
                known = bool(user) and bool(self._one(
                    f"SELECT 1 FROM fact_confirmations WHERE {group} AND user_id = ?", (*of, user)))
                self._write("INSERT INTO fact_confirmations (fact_id, confirmed_at, session_id, user_id, via) "
                            "VALUES (?, ?, ?, ?, ?)", (fact_id, self._now(), session, user or None,
                                                       str(src.get("via") or "") or None))
                if not known:
                    trust = min(1.0, round(trust + bump, 6))
                    self._write("UPDATE facts SET trust_score = ? WHERE fact_id = ?", (trust, fact_id))
        info = self.origins([fact_id]).get(fact_id, {})
        return {"fact_id": fact_id, "confirmed": not again, "confirmations": info.get("confirmations", 0),
                "trust_score": trust, "origin": info.get("origin")}

    def recorded_times(self, fact_ids) -> dict[int, str]:
        """{fact_id: when its current version was recorded (UTC)}: the age of what the fact says now, which temporal
        decay counts from. Feedback, recall counts, keywords and tags do not move it; a new text or date does (a new
        version), and a revert brings back the time of the version it restores. {} without history."""
        ids = [int(i) for i in fact_ids]
        if not ids or not self.history_ready():
            return {}
        out = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            out.update((r[0], r[1]) for r in self.read(
                f"SELECT fact_id, recorded_at FROM fact_versions WHERE status = 'current' "
                f"AND fact_id IN ({','.join('?' * len(chunk))})", chunk))
        return out

    def current_sources(self, fact_ids) -> dict[int, dict | None]:
        """{fact_id: source of its current version} (None: unknown, stored before 0.8.0)."""
        ids = [int(i) for i in fact_ids]
        if not ids or not self.history_ready():
            return {}
        cols = ", ".join(self._source_columns())
        out = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            for r in self.read(f"SELECT fact_id, recorded_on, {cols} FROM fact_versions WHERE status = 'current' "
                               f"AND fact_id IN ({','.join('?' * len(chunk))})", chunk):
                src = source_of(r)
                if src is not None:
                    src["recorded_on"] = r["recorded_on"]
                out[r["fact_id"]] = src
        return out


__all__ = ["HISTORY_SCHEMA", "VERSION_NAMES_TRIGGERS", "HistoryMixin", "HistoryUnavailable", "HistoryTooNew", "ChangedSince", "NotOwned",
           "NotFound", "CLOSED", "ORIGINS", "SOURCE_FIELDS", "normalize", "parse_day", "version_state", "source_of"]
