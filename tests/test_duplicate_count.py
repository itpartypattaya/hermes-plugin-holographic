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
    dup_keys = int(re.search(r"duplicate keys [^:\n]*: (\d+)", out).group(1))
    multi = int(re.search(r"wording changed, NOT a duplicate\): (\d+)", out).group(1))
    return dup_keys, multi


def _cut(ending):
    """A long fact as prefetch renders it: cut to the line cap by the plugin's own _one_line."""
    from hermes_plugin_holographic import _one_line

    return _one_line("The nightly backup of the family photo archive runs from the attic server " * 5 + ending)


def test_suffix_change_is_one_revision_and_cut_never_merges(tmp_path):
    """A suffix change (recorded-on differs) must NOT split one revision; cut renderings must NOT merge
    different revisions; inactive rows do not count as live wording changes."""
    def block(lines):
        return "<memory-context>\n" + "\n".join(lines) + "\n</memory-context>"

    rows = [
        # one revision, two suffix variants: still ONE duplicate key after the fix
        (1, "s", 1, block([f"- #1 [0.5] stable fact text{_SUFFIX}"])),
        (2, "s", 1, block(["- #1 [0.5] stable fact text (recorded 2026-10-02 \u00b7 sample \u00b7 this chat)"])),
        # two different revisions, both cut to the same prefix: must not merge into a duplicate (the plugin's own
        # rendering - the earlier "\u2026 (cut)" rows were a form prefetch never writes; review of 0.26.0)
        (3, "s", 1, block([f"- #2 [0.5] {_cut('to the blue disk')}"])),
        (4, "s", 1, block([f"- #2 [0.5] {_cut('to the cloud bucket')}"])),
        # one live text + one inactive other text: the fact is not a multi-text fact among LIVE rows
        (5, "s", 1, block(["- #3 [0.5] live wording"])),
        (6, "s", 0, block(["- #3 [0.5] dead wording"])),
    ]
    dup_keys, multi = _run(tmp_path, rows)
    assert dup_keys == 1  # only the suffix pair is a duplicate
    assert multi == 0  # inactive wording does not count


def test_real_cut_renderings_of_two_revisions_never_merge(tmp_path):
    """Prefetch cuts a long fact with `_one_line` - an ellipsis at the line cap and no other marker. Two revisions
    that share their first 299 characters render the same line; the counter must not call that one revision shown
    twice (it did, waiting for a "(cut)" marker the plugin never writes)."""
    from hermes_plugin_holographic import _one_line

    def block(lines):
        return "<memory-context>\n" + "\n".join(lines) + "\n</memory-context>"

    stem = "The nightly backup of the family photo archive runs from the attic server " * 5
    first, second = stem + "to the blue disk", stem + "to the cloud bucket"
    assert _one_line(first) == _one_line(second) and _one_line(first).endswith("…")
    rows = [
        (1, "s", 1, block([f"- #2 [0.5] {_one_line(first)}{_SUFFIX}"])),
        (2, "s", 1, block([f"- #2 [0.5] {_one_line(second)}{_SUFFIX}"])),
    ]
    dup_keys, multi = _run(tmp_path, rows)
    assert dup_keys == 0


def test_recall_marks_do_not_split_one_revision(tmp_path):
    """`(by meaning)` and `(inferred)` say how the line was found or what it is, like the recorded-on suffix: one
    revision shown twice, once found by words and once by meaning, is one duplicate, not a wording change."""
    def block(lines):
        return "<memory-context>\n" + "\n".join(lines) + "\n</memory-context>"

    rows = [
        (1, "s", 1, block([f"- #4 [0.5] Lena moved to Lyon{_SUFFIX}"])),
        (2, "s", 1, block([f"- #4 [0.5] Lena moved to Lyon (by meaning){_SUFFIX}"])),
        (3, "s", 1, block(["- #5 [0.3] Lena likes trains (inferred)"])),
        (4, "s", 1, block(["- #5 [0.3] Lena likes trains (by meaning) (inferred)"])),
    ]
    dup_keys, multi = _run(tmp_path, rows)
    assert dup_keys == 2
    assert multi == 0
