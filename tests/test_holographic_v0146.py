"""0.14.6: fixes from the Antigravity CLI review of 0.14.5 — token shapes and the borrowed-word table."""
from __future__ import annotations

import json

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, _fact_problem, _recall_problem, _secret_in
from hermes_plugin_holographic import text as T

# Made-up tokens, put together at run time so that secret scanners do not flag the test file.
BOT_TOKEN = "123456789" + ":" + "AA" + "y" * 33


@pytest.mark.parametrize("text", [
    f"Webhook: https://api.telegram.org/bot{BOT_TOKEN}/setWebhook",  # the Bot API address: "bot" glued to the id
    f"curl api.telegram.org/bot{BOT_TOKEN}/getMe",
    "The STS session key is " + "ASIA" + "IOSFODNN7EXAMPL1",  # a temporary AWS key, not only AKIA
])
def test_more_token_shapes_are_credentials(text):
    assert _fact_problem(text) and _recall_problem(text) and _secret_in(text)


@pytest.mark.parametrize("text", ["The maximum tokens for this model is 8192", "Total tokens: 2048",
                                  "Max output tokens is 128,000", "Лимит токенов: 4096"])
def test_a_token_count_is_a_fact(text):
    assert _fact_problem(text) is None and _recall_problem(text) is None


@pytest.mark.parametrize("text", ["The door PIN is 4821", "пин-код 4821", "password: 12345678",
                                  "The API token is 7f3a9c21d4"])
def test_digits_after_other_words_stay_secret(text):
    assert _fact_problem(text)


@pytest.fixture
def plain(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": 64})
    p.initialize(session_id="s1")
    yield p
    p.shutdown()


def _found(p, query):
    raw = p.handle_tool_call("fact_store", {"action": "search", "query": query, "limit": 5})
    return {f["content"] for f in json.loads(raw)["results"]}


@pytest.mark.parametrize("fact, query", [
    ("Завтра я уберу в своей комнате", "uber"),  # убрать: Uber is out of the table
    ("Убери игрушки до ужина", "uber"),
    ("Ich lese ein Buch über die Geschichte", "убер"),  # German über folds to uber
    ("Mon chat dort sur le lit", "чат"),  # French: a cat
    ("Los botes de pesca salen temprano", "бот"),  # Spanish: boats
    ("Ботинки стоят у двери", "bot"),  # a short form is a whole word, not a prefix
    ("The bottle is on the shelf", "бот"),
])
def test_no_meeting_across_languages_or_by_prefix(plain, fact, query):
    plain._store.add_fact(fact)
    assert fact not in _found(plain, query)


@pytest.mark.parametrize("fact, query", [
    ("The family bots answer in two chats", "бот"),
    ("Робин — бот семьи", "bots"),
    ("Old chats are archived monthly", "чатах"),
    ("Мы переписываемся в чатах", "chat"),
])
def test_short_borrowed_words_still_meet(plain, fact, query):
    plain._store.add_fact(fact)
    assert fact in _found(plain, query)


def test_short_forms_reach_fts5_as_whole_words():
    expr = T.query_terms("bot")[0].fts(T.FTS_OPERATORS)
    assert '"бот"' in expr and '"бот"*' not in expr and '"бота"' in expr and 'loan' not in expr and 'fts' not in expr
    assert '"сервер"*' in T.query_terms("server")[0].fts(T.FTS_OPERATORS)  # long enough: still a prefix
    assert '"ботинк"*' in T.query_terms("ботинки")[0].fts(T.FTS_OPERATORS)  # not a borrowed word: unchanged
