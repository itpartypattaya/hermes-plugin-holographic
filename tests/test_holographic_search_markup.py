"""Markdown glued to a word does not hide the word from search (review 2026-11-17, H11).

An agent writes the way it writes everywhere else: a host name in a code span (`pve-01`), a word in bold (**pve-01**),
a trailing star (scooter*), a list dash with no space after it (-server). The query reader stripped punctuation from
the edges of a word but not these marks, so the term became "`pve-01`" or "**pve-01**", which no fact holds, and search
answered "nothing" for a fact that the bare word finds; an agent that hears "nothing" stores the fact again. The same
marks in a fact's own text hid its word from a bare query.

What the user needs: a word wrapped in or prefixed by markup characters finds what the bare word finds, in the query
and in the fact. What must not change: the marks are no FTS5 syntax (a query of them is answered, not an error), a
CLI flag («--force») is still a word of its own, and marks inside a word («fact_store», «pve-01») stay part of it.
The facts are synthetic.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider

FACTS = {
    "backup": "Бэкап сервера pve-01 запускается в три часа ночи.",
    "scooter": "We rented a scooter for a month from the neighbour.",
    "server": "Nginx on the server in Bangkok proxies the dashboard.",
    "force": "Never run git push --force on the main branch.",
    "tool": "The fact_store tool takes an action and a query.",
    # fillers, so that the words above are rare in the store
    "coffee": "Антон пьёт чёрный кофе без сахара.",
    "rubbish": "Мусор вывозят во вторник и в пятницу.",
    "dentist": "The dentist sees patients on Wednesdays after lunch.",
    "wifi": "The wifi password is written on the fridge.",
    "bike": "Fill the bike up at the station by the market.",
}


@pytest.fixture
def p(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "markup.db"), "hrr_dim": 1024})
    provider.initialize(session_id="markup")
    back = {}
    for key, text in FACTS.items():
        result = json.loads(provider.handle_tool_call("fact_store", {"action": "add", "content": text}))
        back[result["fact_id"]] = key
    provider.back = back
    yield provider
    provider.shutdown()


def _found(provider, query):
    result = json.loads(provider.handle_tool_call("fact_store", {"action": "search", "query": query}))
    assert "error" not in result, result
    return {provider.back.get(f["fact_id"], f["fact_id"]) for f in result.get("results", [])}


@pytest.mark.parametrize("bare, marked", [
    ("pve-01", "`pve-01`"),
    ("pve-01", "**pve-01**"),
    ("pve-01", "*pve-01*"),
    ("scooter", "scooter*"),
    ("scooter", "**scooter**"),
    ("scooter", "`scooter`"),
    ("server", "-server"),
    ("fact_store", "`fact_store`"),
])
def test_a_word_with_markup_finds_what_the_bare_word_finds(p, bare, marked):
    expected = _found(p, bare)
    assert expected, bare  # the bare word finds its fact: the comparison below means something
    assert _found(p, marked) == expected


@pytest.mark.parametrize("text, query", [
    ("Бэкап хоста `pve-02` идёт по воскресеньям.", "pve-02"),
    ("Ключ от **гаража** висит у двери.", "гараж"),
    ("Rent the *kayak* at the pier.", "kayak"),
])
def test_markup_in_a_fact_does_not_hide_its_word(p, text, query):
    fact_id = json.loads(p.handle_tool_call("fact_store", {"action": "add", "content": text}))["fact_id"]
    assert fact_id in _found(p, query)


@pytest.mark.parametrize("query", ["*", "**", "`", "-", "`*`", '"`pve-01`"', "**pve-01** NEAR(`x`)", "-*server*"])
def test_a_query_of_markup_and_fts_syntax_is_answered_not_an_error(p, query):
    _found(p, query)  # asserts no error


def test_a_cli_flag_is_still_a_word_of_its_own(p):
    assert "force" in _found(p, "--force")
