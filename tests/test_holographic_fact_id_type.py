"""A fact id that is not a whole number is refused, never read as some other fact (QA audit 2026-11-17).

Hermes core coerces only string arguments to the schema ("12" -> 12); a native JSON ``true`` or ``1.9`` reaches the
plugin as it is. ``int(True)`` is 1 and ``int(1.9)`` is 1, so ``fact_store(action="forget", fact_id=true)`` erased
fact #1 for good and ``change`` rewrote it. ``merge`` and the recall counts already refused a bool; the other
actions read ``int(a["fact_id"])``. What a model may still send stays accepted: an integer, its string ("12"), and
an integral float (12.0).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider

FIRST = "Яна любит жасминовый чай."


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "ids.db"), "hrr_dim": 1024})
    p.initialize(session_id="ids")
    assert p._store.add_fact(FIRST) == 1
    yield p
    p.shutdown()


def _first(provider):
    rows = provider._store.read("SELECT content FROM facts WHERE fact_id = 1")
    return rows[0][0] if rows else None


def _call(provider, tool, **args):
    return json.loads(provider.handle_tool_call(tool, args))


@pytest.mark.parametrize("bad", [True, False, 1.9, 0.5])
@pytest.mark.parametrize("action,extra", [
    ("forget", {}),
    ("remove", {}),
    ("change", {"content": "Яна любит зелёный чай."}),
    ("correct", {"content": "Яна любит зелёный чай."}),
    ("end", {}),
])
def test_a_fact_store_action_refuses_a_non_integer_id_and_changes_nothing(provider, action, extra, bad):
    out = _call(provider, "fact_store", action=action, fact_id=bad, **extra)
    assert "error" in out, out
    assert "fact_id" in out["error"]
    assert _first(provider) == FIRST
    assert provider._store.history(1)["versions"][-1]["status"] == "current"


@pytest.mark.parametrize("bad", [True, 1.9])
def test_feedback_refuses_a_non_integer_id(provider, bad):
    before = provider._store.read("SELECT trust_score FROM facts WHERE fact_id = 1")[0][0]
    out = _call(provider, "fact_feedback", action="unhelpful", fact_id=bad)
    assert "error" in out, out
    assert provider._store.read("SELECT trust_score FROM facts WHERE fact_id = 1")[0][0] == before


def test_add_refuses_a_bool_supersedes(provider):
    out = _call(provider, "fact_store", action="add", content="Яна любит зелёный чай.", supersedes=True)
    assert "error" in out, out
    assert "supersedes" in out["error"]
    assert _first(provider) == FIRST


@pytest.mark.parametrize("good", [1, "1", 1.0])
def test_a_whole_number_id_still_works(provider, good):
    out = _call(provider, "fact_store", action="change", fact_id=good, content="Яна любит зелёный чай.")
    assert "error" not in out, out
    assert _first(provider) == "Яна любит зелёный чай."
