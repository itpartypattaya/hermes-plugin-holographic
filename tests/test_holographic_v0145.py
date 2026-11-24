"""0.14.5: parallel searches, the secret shapes a fact must not hold, and loanwords written in Latin or Cyrillic."""
from __future__ import annotations

import json
import random
import threading
import time

import pytest

np = pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider, embeddings  # noqa: E402
from test_holographic_v0140_embeddings import FILLERS, ON, FakeEndpoint, fake_vector  # noqa: E402

FACTS = ["The nightly backup runs from the OES box at 2 am.", "Lena's birthday is on June 3.",
         "Mark plays the violin on Sundays.", "Tom is allergic to peanut satay."]
QUERIES = ["Which machine copies our data while we sleep?", "When do we celebrate the special day?",
           "Who plays an instrument?", "Which dish must Tom avoid?", "What happens at night on the server?",
           "Is there an archive of our data?"]


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class _Jittery(FakeEndpoint):
    """Answers after a random pause, so requests and searches interleave."""

    def open(self, request, timeout=None):
        time.sleep(random.uniform(0.0, 0.05))
        return super().open(request, timeout)


def test_parallel_searches_never_share_a_query_vector(tmp_path, monkeypatch):
    """Issue #3 (0.14.4): six searches sent at once, one returned a fact by meaning with the similarity it has to
    another of the six. Each result must carry its own query's closeness."""
    monkeypatch.setattr(embeddings, "_OPENER", _Jittery())
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": 64, **ON})
    p.initialize(session_id="s1")
    try:
        texts = {p._store.add_fact(f"The {w} note number {i} is kept on the {w} shelf"): None
                 for i, w in enumerate(FILLERS)}
        for f in FACTS:
            texts[p._store.add_fact(f)] = f
        for _ in range(5):
            p._catch_up_vectors()
        contents = {r[0]: r[1] for r in p._store.read("SELECT fact_id, content FROM facts")}
        got, errors = {}, []

        def run(q):
            try:
                for _ in range(3):
                    raw = p.handle_tool_call("fact_store", {"action": "search", "query": q, "limit": 3})
                    got.setdefault(q, []).append(json.loads(raw)["results"])
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(q,)) for q in QUERIES]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert not errors
        checked = 0
        for q, runs in got.items():
            qv = _unit(fake_vector(q))
            for results in runs:
                for f in results:
                    if f.get("match") == "meaning":
                        want = float(qv @ _unit(fake_vector(contents[f["fact_id"]])))
                        assert f["similarity"] == pytest.approx(want, abs=1e-3), (q, f["fact_id"])
                        checked += 1
        assert checked  # the test saw facts by meaning at all
    finally:
        p.shutdown()


# -- a token whose shape says it is one --------------------------------------------------------------------------------

from hermes_plugin_holographic import _fact_problem, _recall_problem, _secret_in  # noqa: E402

# Made-up tokens, put together at run time so that secret scanners do not flag the test file (GitHub raised an alert
# on the literal bot token of 0.14.5).
BOT_TOKEN = "7123456789" + ":" + "AA" + "x" * 33
JWT = "eyJ" + "hbGciOiJIUzI1NiJ9" + ".eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0" + "." + "s" * 43
TOKENS = [f"The family bot runs on {BOT_TOKEN}", f"Session cookie {JWT}", "Deploy uses " + "ghp_" + "a" * 24,
          "The OpenAI one is " + "sk-proj-" + "b" * 20, "Key file: -----BEGIN OPENSSH PRIVATE " + "KEY----- b3BlbnNzaC1r"]


@pytest.mark.parametrize("text", TOKENS)
def test_a_bare_token_is_never_stored_recalled_or_sent(text):
    assert _fact_problem(text) and _recall_problem(text) and _secret_in(text)  # bot token and JWT: none of them before


@pytest.mark.parametrize("text", ["The meeting moved to 12:30 on Friday", "Robin's bot id is 7123456789",
                                  "Robin runs with MAX_TOKENS=128000", "Order 123456:7 shipped",
                                  "The token expires in 30 days", "eyJ is how a JWT begins"])
def test_ordinary_text_is_still_a_fact(text):
    assert _fact_problem(text) is None and _recall_problem(text) is None


def test_a_stored_token_is_left_out_of_recall(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": 64})
    p.initialize(session_id="s1")
    try:
        written_before = p._store.add_fact(f"The family bot token is kept as {BOT_TOKEN}")  # bypasses the write check
        clean = p._store.add_fact("The family bot is called Robin")
        block = p.prefetch("what about the family bot", session_id="q")
        assert f"#{clean} " in block and f"#{written_before} " not in block
    finally:
        p.shutdown()


# -- one word in Latin and Cyrillic ------------------------------------------------------------------------------------

from hermes_plugin_holographic import text as T  # noqa: E402


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
    ("The nightly backup runs from the OES box at 2 am", "когда бэкап"),
    ("Ночной бэкап делает OES в 2 часа", "backup schedule"),
    ("Ночные бекапы лежат на диске D", "backups"),
    ("The family chat lives in Telegram", "в каком телеграме чат"),
    ("Мы общаемся в телеграмме", "telegram"),
    ("The server reboots on Sundays", "когда перезагружается сервер"),
    ("Логи лежат на серверах в Германии", "server logs"),
    ("Яна смотрит YouTube после школы", "ютуб"),
    ("Для Таиланда нужна виза ED", "visa"),
    ("Visa runs are done in Laos", "визы"),
    ("Подключаемся через VPN в Казахстане", "впн"),
    ("Отели в Паттайе бронируем на Booking", "hotel"),
    ("The hotel is near the beach", "отелей"),
])
def test_a_borrowed_word_meets_its_other_spelling(plain, fact, query):
    plain._store.add_fact(fact)
    assert fact in _found(plain, query)


@pytest.mark.parametrize("fact, query", [
    ("Ток в розетке 220 вольт", "token"),  # Snowball cuts токен to ток: forms, not stems
    ("Визит к врачу во вторник", "visa"),  # виза and визит share the stem виз
    ("Такса Боня спит у двери", "taxi"),
    ("Крона дерева закрыла окно", "cron"),
    ("Бутылка стоит на окне", "бот"),
])
def test_no_meeting_through_a_stem_or_a_lookalike(plain, fact, query):
    plain._store.add_fact(fact)
    assert fact not in _found(plain, query)


def test_the_mark_never_reaches_an_fts_query():
    expr = T.query_terms("бэкап")[0].fts(T.FTS_OPERATORS)
    assert "loan" not in expr and '"backup"*' in expr and '"бекап"*' in expr
