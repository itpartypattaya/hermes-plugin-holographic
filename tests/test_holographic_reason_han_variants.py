"""reason with a Chinese name finds a fact written in the other set of characters (review 2026-11-17, H17).

Search reads Traditional and Simplified characters as one (text.fold: 醫生 is 医生), so search("醫生") finds «明天下午去看
医生». reason(["醫生"]), for a name that is no known entity, took search's candidates and then kept only the facts
whose text holds the name exactly as typed, so it dropped that very fact: reason answered "nothing" where search
found the answer.

What the user needs: reason with an unknown Chinese name finds the facts search finds for it, in either set of
characters, and still only those that hold every name it is given. The facts are synthetic.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider


@pytest.fixture
def p(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "han.db"), "hrr_dim": 1024})
    provider.initialize(session_id="han")
    yield provider
    provider.shutdown()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _add(provider, content):
    result = _call(provider, action="add", content=content)
    assert "fact_id" in result, result
    return result["fact_id"]


def _ids(result):
    assert "error" not in result, result
    return {f["fact_id"] for f in result.get("results", result.get("facts", []))}


@pytest.mark.parametrize("fact, name", [
    ("明天下午去看医生。", "醫生"),  # a Simplified fact, a Traditional name
    ("明天下午去看醫生。", "医生"),  # and the other way round
])
def test_reason_finds_what_search_finds_in_the_other_characters(p, fact, name):
    _add(p, "周末去公园跑步。")
    wanted = _add(p, fact)
    assert wanted in _ids(_call(p, action="search", query=name))
    assert wanted in _ids(_call(p, action="reason", entities=[name]))


def test_reason_still_requires_every_name(p):
    both = _add(p, "医生说孩子下周要打疫苗。")
    doctor_only = _add(p, "明天下午去看医生。")
    found = _ids(_call(p, action="reason", entities=["醫生", "疫苗"]))
    assert both in found
    assert doctor_only not in found
