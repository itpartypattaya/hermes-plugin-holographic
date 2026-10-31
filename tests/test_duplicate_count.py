"""The duplicate metric (issue #59, review round 2): revision keys - suffix stability, cut exclusion,
live-only wording info. Aggregates only; the store is synthetic, created in tmp_path."""
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "duplicate_count.py"
_SUFFIX = " (recorded 2026-10-01 \u00b7 sample \u00b7 this chat)"


def _run(tmp_path, rows):
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, active INTEGER,
                   api_content TEXT, _compressed_summary INTEGER DEFAULT 0)""")
    for rid, sess, active, api in rows:
        con.execute("INSERT INTO messages VALUES (?, ?, ?, ?, 0)", (rid, sess, active, api))
    con.commit()
    con.close()
    out = subprocess.run([sys.executable, str(_TOOL), "--db", str(db)],
                         capture_output=True, text=True, check=True).stdout
    dup_keys = int(re.search(r"duplicate keys .*: (\d+)", out).group(1))
    multi = int(re.search(r"wording changed, NOT a duplicate\): (\d+)", out).group(1))
    return dup_keys, multi


def test_suffix_change_is_one_revision_and_cut_never_merges(tmp_path):
    """A suffix change (recorded-on differs) must NOT split one revision; cut renderings must NOT merge
    different revisions; inactive rows do not count as live wording changes."""
    def block(lines):
        return "<memory-context>\n" + "\n".join(lines) + "\n</memory-context>"

    rows = [
        # one revision, two suffix variants: still ONE duplicate key after the fix
        (1, "s", 1, block([f"- #1 [0.5] stable fact text{_SUFFIX}"])),
        (2, "s", 1, block(["- #1 [0.5] stable fact text (recorded 2026-10-02 \u00b7 sample \u00b7 this chat)"])),
        # two different revisions, both cut to the same prefix: must not merge into a duplicate
        (3, "s", 1, block(["- #2 [0.5] The nightly backup \u2026 (cut)"])),
        (4, "s", 1, block(["- #2 [0.5] The nightly archive \u2026 (cut)"])),
        # one live text + one inactive other text: the fact is not a multi-text fact among LIVE rows
        (5, "s", 1, block(["- #3 [0.5] live wording"])),
        (6, "s", 0, block(["- #3 [0.5] dead wording"])),
    ]
    dup_keys, multi = _run(tmp_path, rows)
    assert dup_keys == 1  # only the suffix pair is a duplicate
    assert multi == 0  # inactive wording does not count
