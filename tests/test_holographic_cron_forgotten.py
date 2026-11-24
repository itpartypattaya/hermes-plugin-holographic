"""Review 2026-11-17, H3: a scheduled job does not bring back what the user asked to forget, by any write.

Requirement: after "forget it", a cron job (hermes-dreaming's nightly agent re-reading old messages) does not put the
forgotten text back into the store — not by add, and not by turning another fact into it with change, add with
'supersedes', update {content} or correct {content}. Before, only a plain add checked the forget tombstones: a cron
change of another fact to the forgotten text answered "changed". The answer is the plain add's: status 'forgotten',
nothing stored, which hermes-dreaming reads as stored-nothing-on-purpose. The other fact stays as it was. A cron job
may still change a fact to any other text, and the person in a chat may still say the forgotten thing again.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider

FORGOTTEN = "Zoe Quinn owes Leon 500 baht"
OTHER = "Zoe Quinn owes Leon 300 baht"


def _provider(db, **init):
    p = HolographicMemoryProvider(config={"db_path": str(db), "hrr_dim": 512})
    p.initialize(session_id=init.pop("session_id", "s1"), **init)
    return p


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


@pytest.fixture
def setup(tmp_path):
    """A forgotten fact and another current one, then a cron job's provider on the same store."""
    db = tmp_path / "memory_store.db"
    main = _provider(db)
    gone = _call(main, action="add", content=FORGOTTEN)["fact_id"]
    _call(main, action="forget", fact_id=gone)
    other = _call(main, action="add", content=OTHER)["fact_id"]
    main.shutdown()
    cron = _provider(db, session_id="cron1", agent_context="cron", platform="cron")
    yield cron, gone, other, db
    cron.shutdown()


def _content(provider, fact_id):
    return _call(provider, action="history", fact_id=fact_id)["versions"][-1]["content"]


@pytest.mark.parametrize("write", [
    lambda other: {"action": "change", "fact_id": other, "content": FORGOTTEN},
    lambda other: {"action": "add", "content": FORGOTTEN, "supersedes": other},
    lambda other: {"action": "update", "fact_id": other, "content": FORGOTTEN},
    lambda other: {"action": "correct", "fact_id": other, "content": FORGOTTEN},
    lambda other: {"action": "change", "fact_id": other, "content": f"  {FORGOTTEN} "},
], ids=["change", "add+supersedes", "update", "correct", "change-spaces"])
def test_a_cron_write_does_not_turn_a_fact_into_forgotten_text(setup, write):
    cron, gone, other, _ = setup
    result = _call(cron, **write(other))
    assert result.get("status") == "forgotten" and result.get("stored") is False, result
    assert "superseded" not in result and result.get("updated") is not True, result
    assert _content(cron, other) == OTHER
    assert FORGOTTEN not in [r["content"] for r in _call(cron, action="list")["facts"]]


def test_the_plain_add_answers_as_before(setup):
    cron, gone, _, _ = setup
    again = _call(cron, action="add", content=FORGOTTEN)
    assert again["status"] == "forgotten" and again["stored"] is False and again["fact_id"] == gone


def test_a_cron_job_still_changes_a_fact_to_any_other_text(setup):
    cron, _, other, _ = setup
    assert _call(cron, action="change", fact_id=other, content="Zoe Quinn owes Leon 200 baht")["status"] == "changed"
    assert _content(cron, other) == "Zoe Quinn owes Leon 200 baht"


def test_the_person_in_a_chat_may_say_it_again(setup):
    cron, _, other, db = setup
    main = _provider(db, session_id="s2")
    try:
        assert _call(main, action="change", fact_id=other, content=FORGOTTEN)["status"] == "changed"
    finally:
        main.shutdown()
