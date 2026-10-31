"""Condition 8 (issue #59): the duplicate metric, before and after the gate — structural and read-only.

A key is ``(session, fact_id, revision_text)``: the revision text is the line's text after ``]``, so two
revisions of one fact are two keys — differing text under one ``#N`` is **not** a duplicate but a fact whose
wording changed (counted separately, as information). A duplicate is one revision showing in two or more rows
that are live now. Lines are read structurally only: ``<memory-context>`` blocks and the plugin's
``- #N [trust] `` line shape, never prose; a cut rendering (``… (cut)``) counts as its own text, because that
is what the message actually carried. Twins (both rows share the same ``api_content``) stay in the count, per
the maintainer. The cause between a pair's rows — a summary row (``_compressed_summary = 1``) of the **same
session** strictly between the two row ids means compaction-preceded, otherwise window expiry ("preceded by",
not "caused by").

Output is aggregates only: counts, a store fingerprint (sha256 of the path, not the path), row totals, and a
sanitized run label (control characters stripped, capped at 64) — no session ids, no fact texts, no file paths.
The database is opened read-only and every select shares one transaction (one consistent store view).

    python tools/duplicate_count.py [--db PATH] [--label TEXT]

``--db`` defaults to ``$HERMES_HOME/state.db``. Run it on one store before the gate goes on and after, and compare.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import os
import re
import sqlite3
from pathlib import Path

_BLOCK = re.compile(r"<memory-context>(.*?)</memory-context>", re.S)
_LINE = re.compile(r"^- #(\d+) \[[^\]]*\] (.*)$", re.M)
_SUFFIX = re.compile(r" \(recorded [^)]*\)$")  # the recall suffix is presentation, not the revision


def _default_db() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "state.db"


def _load(db: Path):
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        con.execute("BEGIN")  # one read transaction: every select sees the same store snapshot
        rows = con.execute("""
            SELECT id, session_id, active, api_content
            FROM messages
            WHERE api_content LIKE '%<memory-context>%' AND api_content IS NOT NULL
            ORDER BY id
        """).fetchall()
        compaction_by_session: dict[str, set[int]] = collections.defaultdict(set)
        for cid, sess in con.execute("SELECT id, session_id FROM messages WHERE _compressed_summary = 1"):
            compaction_by_session[sess].add(cid)
        sessions_all = con.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()[0]
        messages_all = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        con.execute("COMMIT")
    finally:
        con.close()
    return rows, compaction_by_session, sessions_all, messages_all


def main() -> int:
    ap = argparse.ArgumentParser(description="structural duplicate metric for issue #59, aggregates only")
    ap.add_argument("--db", type=Path, default=_default_db())
    ap.add_argument("--label", default="", help="run label printed with the numbers (e.g. 'before' / 'after')")
    args = ap.parse_args()
    db: Path = args.db.expanduser()
    label = re.sub(r"[\x00-\x1f]", "", args.label)[:64]  # printed, not executed: a label carries no control chars

    rows, compaction_by_session, sessions_all, messages_all = _load(db)

    # (session, fact_id, revision_text) -> [(row_id, active, api_content)]
    injections: dict[tuple[str, int, str], list] = collections.defaultdict(list)
    texts_per_fact: dict[tuple[str, int], set[str]] = collections.defaultdict(set)
    for rid, sess, active, api in rows:
        for block in _BLOCK.findall(api):
            for fid, text in _LINE.findall(block):
                text = text.strip()
                if text.endswith("(cut)"):
                    continue  # a cut rendering carries no stable revision text: it must never merge
                text = _SUFFIX.sub("", text)  # suffix changes do not split one revision either
                injections[(sess, int(fid), text)].append((rid, active, api))
                if active == 1:  # the wording info is over LIVE rows only (its label says so)
                    texts_per_fact[(sess, int(fid))].add(text)

    # Primary grain is the KEY: one revision with >=2 live rows is a duplicate. Twins and cause are per key;
    # cause consults compaction rows of the SAME session only (another session's compaction between two ids
    # says nothing about this one). Pair totals follow as secondary detail.
    keys = twin_keys = 0
    pairs = twin_pairs = 0
    dup_sessions: set[str] = set()
    cause_keys: collections.Counter = collections.Counter()
    cause_pairs: collections.Counter = collections.Counter()
    for (sess, _fid, _text), items in injections.items():
        live = sorted({(rid, api) for rid, active, api in items if active == 1})
        if len(live) < 2:
            continue
        keys += 1
        dup_sessions.add(sess)
        if len({a for _, a in live}) == 1:
            twin_keys += 1
        ids = [rid for rid, _ in live]
        cids = compaction_by_session.get(sess, ())
        between = any(ids[0] < c < ids[-1] for c in cids)
        cause_keys["compaction_preceded" if between else "window_expiry_or_other"] += 1
        for a, b in itertools.combinations(live, 2):
            pairs += 1
            if a[1] == b[1]:
                twin_pairs += 1
            p_between = any(a[0] < c < b[0] for c in cids)
            cause_pairs["compaction_preceded" if p_between else "window_expiry_or_other"] += 1

    # Information, not duplicates: facts whose wording changed across live rows (each text is its own key).
    multi_text_facts = sum(1 for v in texts_per_fact.values() if len(v) > 1)

    fingerprint = hashlib.sha256(str(db).encode("utf-8")).hexdigest()[:12]
    label_print = f" [{label}]" if label else ""
    print(f"duplicate_count{label_print}")
    print(f"  store fingerprint: {fingerprint} | messages: {messages_all} | sessions in store: {sessions_all}")
    print(f"  rows carrying <memory-context>: {len(rows)} | compaction rows: "
          f"{sum(len(v) for v in compaction_by_session.values())}")
    print(f"  duplicate keys (session, fact_id, revision text) with >=2 live rows: {keys} | "
          f"sessions involved: {len(dup_sessions)}")
    print(f"  byte-identical twin keys (rows share api_content): {twin_keys} - kept in the count")
    print(f"  cause, key-level (compaction rows scoped to the key's session): {dict(cause_keys)}")
    print(f"  facts with >1 revision text among live rows (wording changed, NOT a duplicate): {multi_text_facts}")
    print(f"  secondary, live row pairs: {pairs} | twins: {twin_pairs} | cause: {dict(cause_pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
