"""PR #10, review items 2026-10-22 (1, 2, 4, 5) — tests that would not have passed before.

1. A fact that recall MAY show but must not send (_sendable < _recallable):
   `OPENAI_API_KEY=*** / `Authorization: Bearer *** — must not end up in documents
   in any prefetch stream nor in search (before, search did not filter at all).
2. A worker socket timeout (the worker already finished, alive=False) must pause
   and bump `failures` — otherwise a slow server is polled on every turn.
4. A failed rerank in `_unified_stream` returns exactly the off layout: lexical[:5],
   then the meaning/entity extras, not (lexical+…+…)[:5] which pushes them out.
   Plus `_search`: a tail of limit > top_k keeps its place AFTER the ranked ones.
5. `rerank: on` by itself enables the single mode; the `rerank_mode`/`rerank_margin` keys are gone (item 6) — an old configuration is ignored with a warning.
"""
from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, embeddings, rerank, _sendable


# -- fake endpoint -----------------------------------------------------------------------------------------------


class _Response:
    def __init__(self, body: bytes):
        self.body, self.at = body, 0

    def read(self, n=-1):
        piece = self.body[self.at:] if n is None or n < 0 else self.body[self.at:self.at + n]
        self.at += len(piece)
        return piece

    read1 = read  # what urllib's HTTPResponse offers: a chunk, then b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeRerank:
    """Records every request (documents assertable), answers identity/reversed, or fails as told."""

    def __init__(self):
        self.requests: list[dict] = []
        self.fail = None
        self.reverse = False

    @property
    def documents(self) -> list[str]:
        return [d for r in self.requests for d in r["body"]["documents"]]

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append({"url": request.full_url, "body": body, "timeout": timeout})
        if isinstance(self.fail, BaseException):
            raise self.fail
        n = len(body["documents"])
        order = list(reversed(range(n))) if self.reverse else list(range(n))
        top_n = min(body.get("top_n", n), n)
        if self.reverse:
            # smaller (review 19:39): after sorting by score (24:10) the score is by position —
            # otherwise sort reduces reverse to identity
            chosen = [{"index": i, "relevance_score": float(i + 1)} for i in order[:top_n]]
        else:
            chosen = [{"index": i, "relevance_score": float(n - i)} for i in order[:top_n]]
        return _Response(json.dumps({"results": chosen}).encode())


@pytest.fixture
def endpoint(monkeypatch):
    fake = FakeRerank()
    monkeypatch.setattr(rerank, "_OPENER", fake)
    return fake


# -- store --------------------------------------------------------------------------------------------------------

RANK = {"rerank": "on",
        "rerank_url": "https://rerank.example.net/v1/rerank", "rerank_model": "fake-reranker"}
QUERY = "when does the backup run?"
# Neutral fillers: without them the IDF weight of «backup» in almost every fact drags the relevant() coverage
# below the threshold on this small store (checked: 0.3 does not pass; with fillers — it passes).
FILLERS = ["apple note", "brick note", "cloud note", "daisy note", "eagle note"]
PLAIN = ["The nightly backup runs from the OES box at 2 am.",
         "The backup rotation keeps seven days of archives.",
         "Mark plays the violin on Sundays.",
         # item 9: a single candidate makes no call — endpoint tests need a pool of >= 2
         "The weekly backup runs from the NAS box at 4 am."]
# Key-like literals are assembled at runtime: ready-made key-looking text would be edited when the file is written.
_BEARER = "ey" + "JhbGciOiJIUzI1NiJ9." + "abc123DEF456" * 3
_KEY = "sk-" + "proj-" + "aB3" * 8 + "z9"
assert len(_BEARER) >= 12 and len(_KEY) >= 6  # sanity against a placeholder edit
# recallable=True, sendable=False (item 1): recall shows it, the endpoint never gets it
# (the shape of an Authorization header — something _recall_problem does not count as credential-with-value)
_AUTH_HEAD = "The proxy sets Authorization:" + " " + "Bearer" + " "  # fragments: the literal is never edited
SECRET = _AUTH_HEAD + _BEARER + " for backup requests."
# not recallable at all (credential with its value): search shows it, recall and the endpoint do not
KEY_FACT = f"The backup box stores OPENAI_API_KEY={_KEY} in /etc."
VIOLET = "The backup password: violet-quilt-77."


def _provider(tmp_path, name="memory.db", **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / name), "hrr_dim": 64, **config})
    p.initialize(session_id="s1")
    return p


def _fill(p, extra=()):
    for text in FILLERS + list(PLAIN) + list(extra):
        p._store.add_fact(text)
    p.sync_turn("x", "y")


def _without_rerank(tmp_path, query, extra=(), name="off.db") -> str:
    p = _provider(tmp_path, name=name)
    _fill(p, extra)
    try:
        return p.prefetch(query)
    finally:
        p.shutdown()


# -- 1. the secret screen on both paths -------------------------------------------------------------------------------


def test_secret_fact_is_never_in_the_payload(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p, extra=[SECRET, KEY_FACT, VIOLET])
    # the premise of item 1: recallable but not sendable — and it never goes to the endpoint
    assert _sendable(SECRET) is False
    assert p._recallable({"content": SECRET, "trust_score": 0.9, "fact_id": 1}) is True
    # contrast of classes: the recall filter already cuts credential-with-value (it is not sent either)
    assert p._recallable({"content": KEY_FACT, "trust_score": 0.9, "fact_id": 2}) is False
    block = p.prefetch(QUERY)
    assert endpoint.requests, "rerank must actually run — the screen must not switch the feature off"
    sent = endpoint.documents
    assert any("nightly backup" in d for d in sent)          # ordinary facts go out
    assert all(_BEARER not in d for d in sent)               # the _sendable fact does not
    assert all("OPENAI_API_KEY" not in d for d in sent)      # not recallable — even less so
    assert all("violet-quilt" not in d for d in sent)
    assert "nightly backup" in block                          # recall works
    p.shutdown()


def test_search_screens_documents_but_still_shows_the_fact(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p, extra=[SECRET, KEY_FACT, VIOLET])
    res = json.loads(p.handle_tool_call("fact_store", {"action": "search", "query": "backup"}))
    assert endpoint.requests, "search runs the rescorer when it is on"
    assert all(_BEARER not in d for d in endpoint.documents)
    assert all("OPENAI_API_KEY" not in d for d in endpoint.documents)
    assert all("violet-quilt" not in d for d in endpoint.documents)
    # show — it may (search deliberately shows what recall does not inject); send — it may not
    shown = [r.get("content", "") for r in res.get("results", [])]
    assert any("OPENAI_API_KEY" in c for c in shown)
    p.shutdown()


def test_search_order_exception_keeps_the_unranked_results_and_counts(tmp_path, endpoint, monkeypatch):
    """item 4 (review 19:39): order() and the mapping lived outside try — an exception became a tool error
    instead of unranked results, and no failure was counted. Now the whole stage is in try,
    the new order is assigned only after the build, on failure — the previous order + note_failure
    (as in _unified_stream, item 3)."""
    p = _provider(tmp_path, **RANK)
    _fill(p, extra=[SECRET, KEY_FACT, VIOLET])
    off_p = _provider(tmp_path, name="off.db")
    _fill(off_p, extra=[SECRET, KEY_FACT, VIOLET])
    off_ids = [r.get("fact_id") for r in
               json.loads(off_p.handle_tool_call("fact_store",
                                                 {"action": "search", "query": "backup"})).get("results", [])]
    off_p.shutdown()

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(p._rescorer, "order", boom)
    res = json.loads(p.handle_tool_call("fact_store", {"action": "search", "query": "backup"}))
    # on a rerank failure search returns UNRANKED results, not an error
    assert [r.get("fact_id") for r in res.get("results", [])] == off_ids
    assert p._rescorer.failures >= 1, "a stage failure must count as a failure"
    assert p._rescorer.status().startswith("paused")
    p.shutdown()


# -- 2. a socket timeout pauses -------------------------------------------------------------------------------


def test_socket_timeout_pauses_the_endpoint(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = TimeoutError("timed out")  # the worker finished on its own: alive=False
    block = p.prefetch(QUERY)
    assert block == _without_rerank(tmp_path, QUERY)   # the order of the off path
    assert len(endpoint.requests) == 1                 # one call, then a pause
    assert p._rescorer.failures == 1
    assert p._rescorer.status().startswith("paused")
    p.shutdown()


# -- 4a. the unified fallback = the off layout --------------------------------------------------------------------------


def test_failed_rerank_appends_extras_like_off(tmp_path, endpoint, monkeypatch):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = urllib.error.URLError(ConnectionResetError("reset"))
    # synthetic meaning and entity streams (embeddings are off in the test): the facts live in the store,
    # we return their real dicts through search
    m_id = p._store.add_fact("Zebra paraphrase holds no shared word.")
    e_id = p._store.add_fact("Quartz wall fact arrives by entity only.")
    p.sync_turn("x", "y")
    m_fact = p._retriever.search("zebra paraphrase", limit=1)[0]
    e_fact = p._retriever.search("quartz wall", limit=1)[0]
    p._embedding = SimpleNamespace(recall_facts=1, min_score=0.0)
    monkeypatch.setattr(p._retriever, "by_meaning", lambda *a, **k: [m_fact])
    monkeypatch.setattr(p._store, "mentioned_entity_ids", lambda q: {1})  # known entities — a set
    monkeypatch.setattr(p._retriever, "names_fit", lambda *a, **k: True)
    monkeypatch.setattr(p._retriever, "entity_facts", lambda *a, **k: [e_fact])
    fresh, by_meaning = p._unified_stream(QUERY, embeddings.Meaning([], [], weight=0.0),
                                          list(p._retriever.search(QUERY, limit=10)), "s", 1)
    # the off layout: lexical[:5], then extras — extras are NOT pushed out by top-5 (item 4)
    assert [f["fact_id"] for f in fresh[-2:]] == [m_id, e_id]
    lexical_part = [f for f in fresh if f["fact_id"] not in (m_id, e_id)]
    assert len(lexical_part) <= 5
    assert by_meaning == {m_id}
    p.shutdown()


# -- 4b. a tail of limit in search ------------------------------------------------------------------------------------


def test_search_keeps_the_unscored_tail(tmp_path, endpoint):
    endpoint.reverse = True          # the ranked ones go backwards — the mapping is visible
    facts = [f"note {i} about the backup window" for i in range(45)]

    def build(name, **cfg):
        p = _provider(tmp_path, name=name, **cfg)
        for text in facts:
            p._store.add_fact(text)
        p.sync_turn("x", "y")
        return p

    p = build("tail.db", **RANK)  # _search applies no window at all (maintainer fix 19:39)
    res = json.loads(p.handle_tool_call("fact_store", {"action": "search", "query": "backup", "limit": 45}))
    contents = [r.get("content", "") for r in res.get("results", [])]
    p.shutdown()
    # reference: the same store without rerank — the order before the endpoint is touched
    off = build("tail_off.db")
    base = [r.get("content", "")
            for r in json.loads(off.handle_tool_call("fact_store",
                                                      {"action": "search", "query": "backup",
                                                       "limit": 45})).get("results", [])]
    off.shutdown()
    # the reverse fake puts the best score at the last index (smaller 19:39) — after sort (24:10)
    # the 40 ranked ones go backwards, while the 5 unrated tail ones stay in their original order
    assert len(contents) == 45, f"tail dropped: {len(contents)} of 45"
    assert contents == list(reversed(base[:40])) + base[40:]


# -- 5. the default mode = UI ---------------------------------------------------------------------------------------


def test_rerank_on_alone_is_enough():
    """item 6: one mode — rerank: on enables it without any mode keys."""
    settings = rerank.settings_from({"rerank": "on"})
    assert settings is not None
    assert settings.window == rerank.DEFAULT_WINDOW