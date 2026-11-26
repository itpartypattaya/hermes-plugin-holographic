"""0.32.0 (recall_history, on by default): a recalled fact whose text was recently changed or corrected says so.

The requirement: a model that sees "The observatory budget is 600 euros" right after the person corrected it from 500
should know the value was corrected (and when), so it does not argue for the old one it may still see earlier in the
conversation. The old text itself is never shown: a corrected text was wrong, and a cut one loses its "not".
"""
import json
import sqlite3
from datetime import date, timedelta

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider


@pytest.fixture
def make(tmp_path):
    made = []

    def _make(**config):
        p = HolographicMemoryProvider({"db_path": str(tmp_path / f"facts{len(made)}.db"), **config})
        p.initialize(session_id="s", platform="cli")
        made.append(p)
        return p

    yield _make
    for p in made:
        p.shutdown()


def store(p, **args):
    return json.loads(p.handle_tool_call("fact_store", args))


def line_of(block, fact_id):
    lines = [ln for ln in block.splitlines() if ln.startswith(f"- #{fact_id} ")]
    assert lines, f"fact #{fact_id} is not in the block:\n{block}"
    return lines[0]


def today(p):
    return p._store.today()


def test_a_corrected_fact_says_corrected_and_never_shows_the_wrong_text(make):
    p = make()
    fid = store(p, action="add", content="The observatory budget is 500 euros", origin="said")["fact_id"]
    store(p, action="correct", fact_id=fid, content="The observatory budget is 600 euros", origin="said")
    block = p.prefetch("observatory budget", session_id="s")
    assert line_of(block, fid).endswith(f"The observatory budget is 600 euros (corrected {today(p)})")
    assert "500" not in block


def test_a_changed_fact_says_changed_without_the_old_value(make):
    p = make()
    fid = store(p, action="add", content="Alice Smith lives in Paris", origin="said")["fact_id"]
    store(p, action="change", fact_id=fid, content="Alice Smith lives in Berlin", origin="said")
    block = p.prefetch("Where does Alice Smith live?", session_id="s")
    assert line_of(block, fid).endswith(f"Alice Smith lives in Berlin (changed {today(p)})")
    assert "Paris" not in block


def test_a_fact_with_one_version_carries_no_mark_while_a_changed_one_does(make):
    p = make()
    plain = store(p, action="add", content="Bob Jones runs the observatory night shift", origin="said")["fact_id"]
    moved = store(p, action="add", content="Bob Jones parks at the observatory gate", origin="said")["fact_id"]
    store(p, action="change", fact_id=moved, content="Bob Jones parks at the observatory garage", origin="said")
    block = p.prefetch("Bob Jones observatory", session_id="s")
    assert line_of(block, plain).endswith("night shift")
    assert "(changed " in line_of(block, moved)


def test_a_revert_does_not_describe_the_undone_change(make):
    """Revert makes the version before the change current again: that version came from no change or correction."""
    p = make()
    fid = store(p, action="add", content="Bob Jones runs the observatory night shift", origin="said")["fact_id"]
    store(p, action="change", fact_id=fid, content="Bob Jones runs the observatory day shift", origin="said")
    store(p, action="revert", fact_id=fid)
    other = store(p, action="add", content="Bob Jones parks at the observatory gate", origin="said")["fact_id"]
    store(p, action="correct", fact_id=other, content="Bob Jones parks at the observatory garage", origin="said")
    block = p.prefetch("Bob Jones observatory", session_id="s")
    assert line_of(block, fid).endswith("Bob Jones runs the observatory night shift")
    assert "(corrected " in line_of(block, other)  # the mark works in this very block


def test_a_revert_to_a_changed_version_says_changed(make):
    """A changed to B, B corrected to C, C reverted: B is current again, and B came from a change."""
    p = make()
    fid = store(p, action="add", content="Alice Smith lives in Paris", origin="said")["fact_id"]
    store(p, action="change", fact_id=fid, content="Alice Smith lives in Berlin", origin="said")
    store(p, action="correct", fact_id=fid, content="Alice Smith lives in Bern", origin="said")
    store(p, action="revert", fact_id=fid)
    block = p.prefetch("Where does Alice Smith live?", session_id="s")
    assert line_of(block, fid).endswith(f"Alice Smith lives in Berlin (changed {today(p)})")
    assert "Bern " not in block and "Paris" not in block


def test_a_date_only_correction_leaves_the_text_unmarked(make):
    """Only the start date was wrong: the text the model reads is the same claim, nothing about it was corrected."""
    p = make()
    fid = store(p, action="add", content="Alice Smith lives in Berlin", origin="said")["fact_id"]
    store(p, action="correct", fact_id=fid, valid_from="2026-01-05")
    other = store(p, action="add", content="Alice Smith works in Potsdam", origin="said")["fact_id"]
    store(p, action="correct", fact_id=other, content="Alice Smith works in Spandau", origin="said")
    block = p.prefetch("Alice Smith Berlin Spandau", session_id="s")
    assert line_of(block, fid).endswith("Alice Smith lives in Berlin")
    assert "(corrected " in line_of(block, other)


def test_an_old_correction_is_no_longer_marked(make):
    p = make()
    fid = store(p, action="add", content="The observatory budget is 500 euros", origin="said")["fact_id"]
    store(p, action="correct", fact_id=fid, content="The observatory budget is 600 euros", origin="said")
    fresh = store(p, action="add", content="The observatory roof budget is 80 euros", origin="said")["fact_id"]
    store(p, action="correct", fact_id=fresh, content="The observatory roof budget is 90 euros", origin="said")
    old_day = (date.fromisoformat(today(p)) - timedelta(days=91)).isoformat()
    con = sqlite3.connect(p._store.db_path)
    con.execute("UPDATE fact_versions SET recorded_on = ? WHERE fact_id = ? AND status = 'current'", (old_day, fid))
    con.commit()
    con.close()
    block = p.prefetch("observatory budget", session_id="s")
    assert line_of(block, fid).endswith("The observatory budget is 600 euros")
    assert "(corrected " in line_of(block, fresh)


def test_a_store_whose_current_text_left_its_history_is_not_marked(make):
    """A writer older than 0.8.0 rewrote the fact row without a version: the history no longer describes this text."""
    p = make()
    fid = store(p, action="add", content="The observatory budget is 500 euros", origin="said")["fact_id"]
    store(p, action="correct", fact_id=fid, content="The observatory budget is 600 euros", origin="said")
    other = store(p, action="add", content="The observatory roof budget is 80 euros", origin="said")["fact_id"]
    store(p, action="correct", fact_id=other, content="The observatory roof budget is 90 euros", origin="said")
    con = sqlite3.connect(p._store.db_path)
    con.execute("UPDATE facts SET content = 'The observatory budget is 700 euros' WHERE fact_id = ?", (fid,))
    con.commit()
    con.close()
    block = p.prefetch("observatory budget", session_id="s")
    assert line_of(block, fid).endswith("The observatory budget is 700 euros")
    assert "(corrected " in line_of(block, other)


def test_with_recall_source_short_the_mark_takes_the_place_of_recorded(make):
    p = make(recall_source="short")
    fid = store(p, action="add", content="The observatory budget is 500 euros", origin="said")["fact_id"]
    store(p, action="correct", fact_id=fid, content="The observatory budget is 600 euros", origin="said")
    plain = store(p, action="add", content="The observatory opens at night", origin="said")["fact_id"]
    block = p.prefetch("observatory budget night", session_id="s")
    assert line_of(block, fid).endswith(f"600 euros (corrected {today(p)})")
    assert line_of(block, plain).endswith(f"opens at night (recorded {today(p)})")


def test_recall_history_off_gives_the_plain_line(make):
    p = make(recall_history="off")
    fid = store(p, action="add", content="The observatory budget is 500 euros", origin="said")["fact_id"]
    store(p, action="correct", fact_id=fid, content="The observatory budget is 600 euros", origin="said")
    block = p.prefetch("observatory budget", session_id="s")
    assert line_of(block, fid).endswith("The observatory budget is 600 euros")
    on = make()
    same = store(on, action="add", content="The observatory budget is 500 euros", origin="said")["fact_id"]
    store(on, action="correct", fact_id=same, content="The observatory budget is 600 euros", origin="said")
    assert "(corrected " in line_of(on.prefetch("observatory budget", session_id="s"), same)


def test_the_mark_counts_against_the_token_budget(make):
    """prefetch_max_tokens leaves a line out whole: the mark is part of the line, so a line that fits without it and
    not with it is left out — measured against the same fact without history."""
    plain_p = make(prefetch_max_tokens=10_000)
    store(plain_p, action="add", content="The observatory budget is 600 euros", origin="said")
    unmarked = plain_p.prefetch("observatory budget", session_id="s")
    budget = plain_p._budget_last["tokens"]
    p = make(prefetch_max_tokens=budget)
    fid = store(p, action="add", content="The observatory budget is 500 euros", origin="said")["fact_id"]
    store(p, action="correct", fact_id=fid, content="The observatory budget is 600 euros", origin="said")
    assert "600 euros" in unmarked
    assert p.prefetch("observatory budget", session_id="s") == ""
