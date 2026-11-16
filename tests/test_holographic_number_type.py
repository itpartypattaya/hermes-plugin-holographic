"""A number argument that is no real number is refused, never read as some other value (QA audit 2026-11-17).

Hermes core coerces only string arguments to the schema ("0.1" -> 0.1) and leaves "nan" / "inf" as strings (they are
no JSON numbers); a native JSON ``true`` reaches the plugin as it is. ``float(True)`` is 1.0 and ``float("nan")`` is
NaN, and the trust clamp ``max(0, min(1, x))`` reads NaN as 1.0: ``update {trust_delta: true}`` or ``"nan"`` raised a
fact's trust to the top. ``min_trust: true`` read as 1.0 hid every fact from search and list, and ``limit: true`` gave
one result. What a model may still send stays accepted: a number, its string, an integral float for an integer.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider

FACTS = ("Яна любит жасминовый чай.", "Яна живёт в Паттайе.", "Яна пьёт чай по утрам.")


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "numbers.db"), "hrr_dim": 1024})
    p.initialize(session_id="numbers")
    for text in FACTS:
        p._store.add_fact(text)
    yield p
    p.shutdown()


def _trust(provider, fact_id=1):
    return provider._store.read("SELECT trust_score FROM facts WHERE fact_id = ?", (fact_id,))[0][0]


def _call(provider, tool, **args):
    return json.loads(provider.handle_tool_call(tool, args))


@pytest.mark.parametrize("bad", [True, False, float("nan"), "nan", "NaN", float("inf"), "-inf"])
def test_update_refuses_a_trust_delta_that_is_no_number_and_keeps_the_trust(provider, bad):
    before = _trust(provider)
    out = _call(provider, "fact_store", action="update", fact_id=1, trust_delta=bad)
    assert "error" in out, out
    assert "trust_delta" in out["error"]
    assert _trust(provider) == before


@pytest.mark.parametrize("good,expected", [(0.1, 0.1), ("0.1", 0.1), (-0.2, -0.2), (1, 1.0)])
def test_update_still_moves_trust_by_a_real_number(provider, good, expected):
    before = _trust(provider)
    out = _call(provider, "fact_store", action="update", fact_id=1, trust_delta=good)
    assert out.get("updated") is True, out
    assert _trust(provider) == pytest.approx(max(0.0, min(1.0, before + expected)))


@pytest.mark.parametrize("bad", [True, float("nan"), "nan"])
@pytest.mark.parametrize("action,extra", [
    ("search", {"query": "Яна чай"}),
    ("list", {}),
    ("probe", {"entity": "Яна"}),
])
def test_a_read_refuses_a_min_trust_that_is_no_number(provider, action, extra, bad):
    out = _call(provider, "fact_store", action=action, min_trust=bad, **extra)
    assert "error" in out, out
    assert "min_trust" in out["error"]


@pytest.mark.parametrize("good", [0.0, "0.2", 0])
def test_list_still_takes_a_real_min_trust(provider, good):
    out = _call(provider, "fact_store", action="list", min_trust=good)
    assert out.get("count") == len(FACTS), out


@pytest.mark.parametrize("bad", [True, False, 2.5])
@pytest.mark.parametrize("action,extra", [("list", {}), ("search", {"query": "Яна чай"})])
def test_a_read_refuses_a_limit_that_is_no_whole_number(provider, action, extra, bad):
    out = _call(provider, "fact_store", action=action, limit=bad, **extra)
    assert "error" in out, out
    assert "limit" in out["error"]


@pytest.mark.parametrize("good", [2, "2", 2.0])
def test_list_still_takes_a_whole_number_limit(provider, good):
    out = _call(provider, "fact_store", action="list", limit=good)
    assert out.get("count") == 2, out
