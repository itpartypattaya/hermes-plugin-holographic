"""Issue #123: the category a write gives is one of the four the fact_store schema declares.

Requirements:
- add, change, correct, update and merge with a category outside user_pref / project / tool / general are refused, the
  answer names the four, and nothing is written (the schema told the model four values while any string was stored,
  so a schema-validating client and the handler disagreed);
- each of the four is stored as given;
- a category that is no string is refused the same way;
- search and list still filter by any category a store holds (an older release stored whatever it was given, and
  those facts stay reachable).
The facts are synthetic.
"""

from __future__ import annotations

import json

import pytest

from hermes_plugin_holographic import FACT_STORE_SCHEMA, HolographicMemoryProvider

FOUR = ("user_pref", "project", "tool", "general")


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64})
    p.initialize(session_id="s1", hermes_home=str(tmp_path))
    yield p
    p.shutdown()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _categories(provider):
    return {f["content"]: f["category"] for f in provider._store.list_facts(limit=50)}


def test_the_schema_declares_the_four(provider):
    assert tuple(FACT_STORE_SCHEMA["parameters"]["properties"]["category"]["enum"]) == FOUR


@pytest.mark.parametrize("category", FOUR)
def test_each_declared_category_is_stored(provider, category):
    answer = _call(provider, action="add", content="The office plant is a ficus.", category=category)
    assert "error" not in answer, answer
    assert _categories(provider) == {"The office plant is a ficus.": category}


@pytest.mark.parametrize("category", ["rule", "incident", "infra", 5, ["general"]])
def test_add_with_an_undeclared_category_is_refused_and_stores_nothing(provider, category):
    answer = _call(provider, action="add", content="The office plant is a ficus.", category=category)
    assert "error" in answer, answer
    assert all(name in answer["error"] for name in FOUR), answer
    assert provider._store.list_facts(limit=50) == []


@pytest.mark.parametrize("action,extra", [
    ("change", {"content": "The office plant is a palm."}),
    ("correct", {"content": "The office plant is a palm."}),
    ("update", {}),
])
def test_a_write_to_a_fact_with_an_undeclared_category_leaves_it_as_it_was(provider, action, extra):
    fact_id = _call(provider, action="add", content="The office plant is a ficus.", category="general")["fact_id"]
    answer = _call(provider, action=action, fact_id=fact_id, category="incident", **extra)
    assert "error" in answer, answer
    assert _categories(provider) == {"The office plant is a ficus.": "general"}


def test_merge_with_an_undeclared_category_is_refused(provider):
    a = _call(provider, action="add", content="The office plant is a ficus.", category="general")["fact_id"]
    b = _call(provider, action="add", content="The office plant is a ficus tree.", category="general")["fact_id"]
    answer = _call(provider, action="merge", fact_id=a, **{"from": [b]}, category="workflow")
    assert "error" in answer, answer
    assert len(provider._store.list_facts(limit=50)) == 2


def test_search_and_list_still_reach_a_category_an_older_release_stored(provider):
    provider._store.add_fact("The deploy runbook lives in the wiki.", category="runbook")
    listed = _call(provider, action="list", category="runbook")
    found = _call(provider, action="search", query="deploy runbook", category="runbook")
    assert [f["content"] for f in listed["facts"]] == ["The deploy runbook lives in the wiki."]
    assert [f["content"] for f in found["results"]] == ["The deploy runbook lives in the wiki."]
