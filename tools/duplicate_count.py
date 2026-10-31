"""Condition 8 (issue #59): the duplicate metric, before and after the gate — structural and read-only.

A key is ``(session, fact_id)``; a duplicate is a key whose fact line shows in two or more rows that are live now.
Lines are read structurally only: ``<memory-context>`` blocks and the plugin's ``- #N [trust] `` line shape, never
prose. Pair reports: same line text vs differing text under one ``#N`` (revision or two stores), byte-identical
twins (both rows share the same ``api_content`` — kept in, per the maintainer), and the cause between a pair's
rows — a summary row (``_compressed_summary = 1``) strictly between the two row ids means compaction-preceded,
otherwise window expiry ("preceded by", not "caused by").

Output is aggregates only: counts, a store fingerprint (sha256 of the path, not the path), row totals, and an
optional run label — no session ids, no fact texts, no file paths. The database is opened read-only.

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


def _default_db() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "state.db"


def _load(db: Path):
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("""
            SELECT id, session_id, active, api_content
            FROM messages
            WHERE api_content LIKE '%<memory-context>%' AND api_content IS NOT NULL
            ORDER BY id
        """).fetchall()
        compaction_ids = {r[0] for r in con.execute("SELECT id FROM messages WHERE _compressed_summary = 1")}
        sessions_all = con.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()[0]
        messages_all = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        con.close()
    return rows, compaction_ids, sessions_all, messages_all


def main() -> int:
    ap = argparse.ArgumentParser(description="structural duplicate metric for issue #59, aggregates only")
    ap.add_argument("--db", type=Path, default=_default_db())
    ap.add_argument("--label", default="", help="run label printed with the numbers (e.g. 'before' / 'after')")
    args = ap.parse_args()
    db: Path = args.db.expanduser()

    rows, compaction_ids, sessions_all, messages_all = _load(db)

    # (session, fact_id) -> [(row_id, active, line_text, api_content)]
    injections: dict[tuple[str, int], list] = collections.defaultdict(list)
    for rid, sess, active, api in rows:
        for block in _BLOCK.findall(api):
            for fid, text in _LINE.findall(block):
                injections[(sess, int(fid))].append((rid, active, text.strip(), api))

    # Primary grain is the KEY (a key = (session, fact_id) with >=2 live rows), matching the baseline definition:
    # same/diff text and twins are per key; a key is compaction-preceded when a summary row falls strictly between
    # its first and last live row. Pair totals follow as secondary detail.
    keys = same_text_keys = diff_text_keys = twin_keys = 0
    pairs = same_text_pairs = diff_text_pairs = twin_pairs = 0
    dup_sessions: set[str] = set()
    cause_keys: collections.Counter = collections.Counter()
    cause_pairs: collections.Counter = collections.Counter()
    for (sess, _fid), items in injections.items():
        live = sorted({(rid, text, api) for rid, active, text, api in items if active == 1})
        if len(live) < 2:
            continue
        keys += 1
        dup_sessions.add(sess)
        texts = {t for _, t, _ in live}
        apis = {a for _, _, a in live}
        if len(texts) == 1:
            same_text_keys += 1
        else:
            diff_text_keys += 1
        if len(apis) == 1:
            twin_keys += 1
        ids = [rid for rid, _, _ in live]
        between = any(ids[0] < c < ids[-1] for c in compaction_ids)
        cause_keys["compaction_preceded" if between else "window_expiry_or_other"] += 1
        for a, b in itertools.combinations(live, 2):
            pairs += 1
            if a[1] == b[1]:
                same_text_pairs += 1
            else:
                diff_text_pairs += 1
            if a[2] == b[2]:
                twin_pairs += 1
            p_between = any(a[0] < c < b[0] for c in compaction_ids)
            cause_pairs["compaction_preceded" if p_between else "window_expiry_or_other"] += 1

    fingerprint = hashlib.sha256(str(db).encode("utf-8")).hexdigest()[:12]
    label = f" [{args.label}]" if args.label else ""
    print(f"duplicate_count{label}")
    print(f"  store fingerprint: {fingerprint} | messages: {messages_all} | sessions in store: {sessions_all}")
    print(f"  rows carrying <memory-context>: {len(rows)} | compaction rows: {len(compaction_ids)}")
    print(f"  duplicate keys (session, fact_id) with >=2 live rows: {keys} | sessions involved: {len(dup_sessions)}")
    print(f"  keys, same line text / differing text under one #N: {same_text_keys} / {diff_text_keys}")
    print(f"  byte-identical twin keys (rows share api_content): {twin_keys} — kept in the count")
    print(f"  cause, key-level: {dict(cause_keys)}")
    print(f"  secondary, live row pairs: {pairs} | same text: {same_text_pairs} | differing: {diff_text_pairs}"
          f" | twins: {twin_pairs} | cause: {dict(cause_pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
