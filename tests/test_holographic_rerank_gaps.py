"""The four gaps the 2026-10-22 review found in the rerank suite, in one new file.

Existing tests, conftest and production code are untouched. Nothing reaches a real endpoint: ``rerank._OPENER``
is replaced by a fake, so every request the endpoint *would* have seen is recorded and can be asserted on.

1. ``Rescorer._request``'s HTTP branches — HTTPError (500 / 429), URLError, a body json cannot parse — were
   covered only for ConnectionRefused and an out-of-range index.
2. No test ever looked at WHAT was sent: ``FakeRerank.documents`` is asserted here on BOTH paths of prefetch —
   the unified stream — so a stored
   credential and a fact injected one turn ago are provably cut before the endpoint sees the pool.
3. ``_unified_stream`` had no direct test at all: the union of the three streams, the safety / de-dup it does
   itself, the top five strictly by the rank the endpoint answered, and the fall back to the old order when
   ``scores()`` returns None.
4. A config with rerank on and an address rerank.py refuses: recall must not fail and must come out in the
   order it has without rerank. Plus the optional one: a database file that is not a database."""
from __future__ import annotations

import json
import logging
import urllib.error
from types import SimpleNamespace

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, embeddings, rerank

# -- a fake endpoint --------------------------------------------------------------------------------------------


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
    """Stands in for rerank._OPENER: records every request (so documents can be asserted), answers with a fixed
    or reversed order of what it was sent, or fails as told."""

    def __init__(self):
        self.requests: list[dict] = []
        self.fail = None  # an exception to raise, or bytes to answer with
        self.reverse = False  # answer the pool backwards: worst rank first
        self.scores = None  # optional relevance_scores to answer with (else n, n-1, ...)

    @property
    def documents(self) -> list[str]:
        return [d for r in self.requests for d in r["body"]["documents"]]

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body,
                              "timeout": timeout})
        if isinstance(self.fail, BaseException):
            raise self.fail
        if isinstance(self.fail, bytes):
            return _Response(self.fail)
        n = len(body["documents"])
        order = list(reversed(range(n))) if self.reverse else list(range(n))
        top_n = min(body.get("top_n", n), n)
        if self.scores is not None:
            chosen = [{"index": i, "relevance_score": self.scores[i % len(self.scores)]} for i in order[:top_n]]
        elif self.reverse:
            # smaller (review 19:39): after sorting by score (24:10) the answer must be genuinely
            # flipped — score by position (best at the last index), otherwise sort reduces reverse
            # to identity and the test stops verifying the flip
            chosen = [{"index": i, "relevance_score": float(i + 1)} for i in order[:top_n]]
        else:
            chosen = [{"index": i, "relevance_score": float(n - i)} for i in order[:top_n]]
        return _Response(json.dumps({"results": chosen, "model": body["model"]}).encode())


@pytest.fixture
def endpoint(monkeypatch):
    fake = FakeRerank()
    monkeypatch.setattr(rerank, "_OPENER", fake)
    return fake


# -- a store to run on -------------------------------------------------------------------------------------------

RANK = {"rerank": "on", "rerank_url": "https://rerank.example.net/v1/rerank",
        "rerank_model": "fake-reranker"}  # one mode since item 6: rerank_mode is gone
QUERY = "when does the backup run?"
FACTS = ["The nightly backup runs from the OES box at 2 am.", "Lena's birthday is on June 3.",
         "Mark plays the violin on Sundays.", "Tom is allergic to peanut satay.",
         # item 9: a single candidate makes no call — endpoint tests need a pool of >= 2
         "The weekly backup runs from the NAS box at 4 am."]
FILLERS = ["apple note", "brick note", "cloud note", "daisy note", "eagle note"]
SECRET = "The backup password: violet-quilt-77."  # stored, but recall refuses it (a credential with its value)
NOTE_QUERY = "what about the note?"  # every note fact is a candidate of it, in every turn
EXTRA_NOTES = ["fjord note number", "garnet note number"]  # more note facts than one turn's top five holds
MEANING = "Violin practice happens every Sunday at noon."  # shares no word with QUERY, linked to one entity
ENTITY = "Green paint covers every wall."  # ditto: only the entity stream can bring it in


def _provider(tmp_path, name="memory_store.db", **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / name), "hrr_dim": 64, **config})
    p.initialize(session_id="s1")
    return p


def _fill(p):
    for f in FILLERS:
        p._store.add_fact(f"{f} number")
    for f in FACTS:
        p._store.add_fact(f)
    p.sync_turn("x", "y")


def _block(p, query):
    return p.prefetch(query)


def _without_rerank(tmp_path, query, name="baseline.db") -> str:
    """The same store and the same query with rerank off: the order every failure must fall back to."""
    p = _provider(tmp_path, name=name)
    _fill(p)
    try:
        return _block(p, query)
    finally:
        p.shutdown()


def _sent(fake, index: int) -> list[str]:
    return fake.requests[index]["body"]["documents"]


def _shown(block: str) -> set[str]:
    """The fact contents a recall block injects (its lines are "- #12 [0.5] the fact")."""
    return {line.split("] ", 1)[1] for line in (block or "").splitlines()
            if line.startswith("- #") and "] " in line}


# -- 1. the HTTP branches of Rescorer._request --------------------------------------------------------------------

@pytest.mark.parametrize("code", [500, 429])
def test_http_error_keeps_order_and_pauses(tmp_path, endpoint, caplog, code):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = urllib.error.HTTPError("https://rerank.example.net/v1/rerank", code, "boom", {}, None)
    with caplog.at_level(logging.WARNING):
        block = _block(p, QUERY)
    assert block == _without_rerank(tmp_path, QUERY)  # the order is the one without rerank
    assert len(endpoint.requests) == 1  # one attempt, then the endpoint goes to backoff
    assert p._rescorer.failures == 1
    assert p._rescorer.status().startswith("paused")
    assert f"rerank endpoint failed (HTTP {code})" in caplog.text
    p.shutdown()


def test_urlerror_keeps_order_and_pauses(tmp_path, endpoint, caplog):
    """The branch a real opener takes when the socket breaks under a reachable address (an error whose reason
    is neither a timeout nor HTTP): the reason's type is what the debug line reports."""
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = urllib.error.URLError(ConnectionResetError("reset by peer"))
    with caplog.at_level(logging.WARNING):
        block = _block(p, QUERY)
    assert block == _without_rerank(tmp_path, QUERY)
    assert len(endpoint.requests) == 1
    assert p._rescorer.failures == 1
    assert p._rescorer.status().startswith("paused")
    assert "cannot reach the endpoint (ConnectionResetError)" in caplog.text
    p.shutdown()


@pytest.mark.parametrize("body", [b"not json at all", b'{"results": [', b""],
                         ids=["garbage", "truncated", "empty"])
def test_broken_body_keeps_order_and_pauses(tmp_path, endpoint, caplog, body):
    """A 200 whose body json cannot parse (the answer never reaches _ranked): the same graceful refusal."""
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = body
    with caplog.at_level(logging.WARNING):
        block = _block(p, QUERY)
    assert block == _without_rerank(tmp_path, QUERY)
    assert len(endpoint.requests) == 1
    assert p._rescorer.failures == 1
    assert p._rescorer.status().startswith("paused")
    assert "no answer (JSONDecodeError)" in caplog.text
    p.shutdown()


# -- 2. what never leaves the machine: safety and de-dup BEFORE the endpoint --------------------------------------

def test_a_stored_credential_is_never_sent(tmp_path, endpoint):
    """A fact recall refuses (a credential) IS a candidate of the query — only the safety check before the
    fork keeps it out of the request."""
    p = _provider(tmp_path, **RANK)
    _fill(p)
    p._store.add_fact(SECRET)
    pool = [c["content"] for c in p._retriever.search(QUERY, min_trust=p._min_trust, limit=60)]
    assert FACTS[0] in pool and SECRET in pool  # it is a candidate: the search is not what keeps it out
    block = _block(p, QUERY)
    assert block and FACTS[0] in block and SECRET not in block  # recall withholds it from the prompt
    assert endpoint.requests  # ...and the rest of the pool WAS scored
    assert SECRET not in endpoint.documents  # ...without the secret
    assert FACTS[0] in endpoint.documents  # the safe fact went out: it is filtering, not an empty pool
    p.shutdown()


def test_a_fact_injected_one_turn_ago_is_never_sent_again(tmp_path, endpoint):
    """prefetch_repeat_turns: seven facts answer the message, only five fit into one turn's top. The next turn
    finds all seven again — the five already injected are cut BEFORE the endpoint sees the pool, the two fresh
    ones go out."""
    p = _provider(tmp_path, **RANK)
    _fill(p)
    for note in EXTRA_NOTES:
        p._store.add_fact(note)
    notes = [f"{f} number" for f in FILLERS] + EXTRA_NOTES
    first = _block(p, NOTE_QUERY)
    assert first and endpoint.requests
    pool = {c["content"] for c in p._retriever.search(NOTE_QUERY, min_trust=p._min_trust, limit=60)}
    assert set(notes) <= pool  # next turn the search finds all seven again: that is not what keeps them out
    assert set(notes) <= set(_sent(endpoint, 0))  # turn 1 sent every one: nothing was injected back then
    injected = _shown(first)
    assert injected and injected <= set(notes)
    second = _block(p, NOTE_QUERY)
    assert len(endpoint.requests) == 2  # the second turn WAS scored — with another pool
    sent = _sent(endpoint, 1)
    assert sent and set(sent) <= set(notes)  # the fresh facts of the same query went out ...
    assert set(sent).isdisjoint(injected)  # ... and the injected ones did not: cut before the request
    assert second and _shown(second).isdisjoint(injected)  # nor repeated in the prompt either
    p.shutdown()


# -- 3. _unified_stream, directly ---------------------------------------------------------------------------------

def _unified_store(tmp_path, endpoint, monkeypatch):
    """A store whose three streams are separable: stream 1 is what QUERY's own words find, streams 2 and 3 can
    only bring in the two facts that share no word with it (both linked to one entity the query does not name),
    and the credential fact that must never be scored. Returns (provider, the meaning stream's fact)."""
    p = _provider(tmp_path, **RANK)
    _fill(p)
    p._store.add_fact(SECRET)
    p._store.add_fact_status(MEANING, entities=["Zorblax"])
    p._store.add_fact_status(ENTITY, entities=["Zorblax"])
    # The entity stream asks which entities the message names; the message names none, so the store is told
    # the one entity that exists (names_fit then asks whether the meaning fact is linked to it — it is).
    known = p._store.mentioned_entity_ids("what about Zorblax today?")
    assert len(known) == 1
    entity_id = next(iter(known))
    monkeypatch.setattr(p._store, "mentioned_entity_ids", lambda q: {entity_id})
    monkeypatch.setattr(p, "_embedding", SimpleNamespace(recall_facts=1, min_score=None))
    meaning_fact = next(f for f in p._retriever.search("violin practice", min_trust=p._min_trust, limit=10)
                        if f["content"] == MEANING)
    secret_fact = next(f for f in p._retriever.search("backup password", min_trust=p._min_trust, limit=10)
                       if f["content"] == SECRET)

    def by_meaning(meaning, *, exclude=(), **kwargs):  # the real contract: never offer what is already in
        return [f for f in (secret_fact, meaning_fact) if f["fact_id"] not in set(exclude)]

    monkeypatch.setattr(p._retriever, "by_meaning", by_meaning)
    return p, meaning_fact


def _stream_arguments(p):
    """(query, meaning, candidates, key, turn) as prefetch passes them: candidates already through the safety
    and de-dup it does before the fork (nothing is recent on the first turn)."""
    key, turn = f"s1|{p._scope_key or ''}", 1
    candidates = [f for f in p._retriever.search(QUERY, min_trust=p._min_trust, limit=60)
                  if p._recallable(f) and p._not_recent(key, f, turn)]
    return QUERY, embeddings.Meaning([], [], weight=0.0), candidates, key, turn


def test_unified_stream_sends_the_union_and_ranks_it_strictly(tmp_path, endpoint, monkeypatch):
    p, meaning_fact = _unified_store(tmp_path, endpoint, monkeypatch)
    query, meaning, candidates, key, turn = _stream_arguments(p)
    lexical = [f["content"] for f in p._retriever.relevant(query, candidates, p._min_coverage)]
    # stream 1: the words alone decide membership (item 9: the backup request pool is now two facts)
    assert lexical == [FACTS[0], FACTS[4]]
    endpoint.reverse = True  # the endpoint answers the pool backwards: worst rank first
    fresh, by_meaning = p._unified_stream(query, meaning, candidates, key, turn)
    # What the endpoint saw: both streams first, then the lexical pool — the credential the meaning stream
    # offered was dropped by _unified_stream itself, and the entity fact the meaning stream had already
    # admitted was not offered twice.
    assert endpoint.documents == [MEANING, ENTITY, FACTS[0], FACTS[4]]
    assert endpoint.documents.count(MEANING) == 1
    assert by_meaning == {meaning_fact["fact_id"]}
    # Score semantics since 24:10 (smaller, maintainer: «a sort would close the case»): _ranked
    # sorts by relevance_score, list position is no longer authoritative. The reverse fake now
    # puts the best score at the last index (smaller 19:39) — a flipped answer really
    # stays flipped after sort too: F4(4.0) -> F0(3.0) -> ENTITY(2.0) -> MEANING(1.0);
    # the window δ=0.05 cuts below best-δ (0.982 and 0.953 pass, 0.881 does not): the top two.
    assert [f["content"] for f in fresh] == [FACTS[4], FACTS[0]]
    p.shutdown()


def test_unified_stream_falls_back_to_the_old_order_when_scores_is_none(tmp_path, endpoint, monkeypatch):
    p, meaning_fact = _unified_store(tmp_path, endpoint, monkeypatch)
    query, meaning, candidates, key, turn = _stream_arguments(p)
    endpoint.fail = ConnectionRefusedError("refused")  # scores() returns None
    fresh, by_meaning = p._unified_stream(query, meaning, candidates, key, turn)
    # lexical first (both backup facts, item 9), streams after: as in off
    assert [f["content"] for f in fresh] == [FACTS[0], FACTS[4], MEANING, ENTITY]
    assert by_meaning == {meaning_fact["fact_id"]}  # the marks are kept so the prompt can still say "(by meaning)"
    p.shutdown()


# -- 4. rerank on, the endpoint unusable -------------------------------------------------------------------------

def test_dead_endpoint_degrades_to_the_order_without_rerank(tmp_path, endpoint):
    """rerank on, but an address rerank.py refuses outright: the provider must not fail, must not send
    anything, and must recall exactly what it recalls with rerank off."""
    p = _provider(tmp_path, **{**RANK, "rerank_url": "ftp://nowhere.invalid/v1/rerank"})
    _fill(p)
    assert p._rescorer is not None  # the config asks for rerank ...
    assert p._rescorer.unavailable()  # ... and rerank.py refuses the address before any request
    assert not p._rescorer.usable()
    block = _block(p, QUERY)
    assert block == _without_rerank(tmp_path, QUERY)  # degraded to the order without rerank
    assert endpoint.requests == []  # not one request was even attempted
    assert p._rescorer.status().startswith("unavailable")
    assert p._rescorer.failures == 0  # refused, not failed: nothing to back off from
    p.shutdown()


# -- 5. a database file that is not a database --------------------------------------------------------------------

def test_a_file_that_is_not_a_database_takes_recall_off(tmp_path):
    path = tmp_path / "memory_store.db"
    path.write_bytes(b"this is definitely not a sqlite database " * 8)
    p = HolographicMemoryProvider(config={"db_path": str(path), "hrr_dim": 64})
    p.initialize(session_id="s1")
    assert p._store is None  # the store could not be opened ...
    assert p.prefetch(QUERY) == ""  # ... so recall is off and the turn goes on
    prompt = p.system_prompt_block()
    assert prompt.startswith("# Holographic Memory\nUnavailable this session")
    assert "file is not a database" in prompt
    assert "recover it with: sqlite3" in prompt  # the way out is named, not just the failure
    p.shutdown()



# -- the fallback IS the off layout (review 19:39, item 3) -------------------------------------------------------

def test_fallback_block_equals_off_block_when_the_target_sits_past_candidate_15(tmp_path, endpoint):
    """The fallback took top-5 from the WIDENED pool (max(15×4, top_k)) and without the «shown recently» filter,
    while off searches 15: candidates 1–15 are shown, the target sits at 17th place — off returns empty, the old fallback
    would have returned five from the wide pool. Now the fallback calls _off_layout WITHOUT candidates (a repeated
    search with the off limit) — the blocks must match EXACTLY. The «shown recently» mechanism from the
    maintainer's scenario: three runs show the top-15 fillers, the fourth — empty."""
    query = "nightly backup schedule"

    def store_provider(name, **cfg):
        p = _provider(tmp_path, name=name, **cfg)
        # the fillers share all three query words with a high TF — BM25 keeps them above the short target,
        # which falls to 17th place, beyond the off search limit=15
        for k in range(20):
            p._store.add_fact(f"nightly nightly backup backup backup schedule schedule schedule of subsystem {k}"
                              f" pending in the apple tree of configuration layer {k}")
        target_id = p._store.add_fact("The nightly backup schedule runs at 2 am.")
        p.sync_turn("x", "y")
        pool = p._retriever.search(query, min_trust=p._min_trust, limit=60, meaning=None)
        position = next(i for i, fact in enumerate(pool) if fact["fact_id"] == target_id)
        assert position >= 15, f"construction broke: the target is at position {position}, need >= 15"
        return p

    off = store_provider("off.db")
    for _ in range(3):  # the top-15 fillers are shown as recent
        off.prefetch(query)
    block_off = off.prefetch(query)
    assert block_off == "", f"off must return empty (all 15 are shown): {block_off!r}"
    off.shutdown()

    on = store_provider("on.db", **RANK)
    endpoint.fail = RuntimeError("endpoint down")  # every call goes to the fallback
    for _ in range(3):  # the fallback must advance exactly like off (the recent filter + limit 15)
        on.prefetch(query)
    block_fallback = on.prefetch(query)
    on.shutdown()
    assert block_fallback == block_off, ("the fallback must be the off layout: "
                                         f"off={block_off!r}, fallback={block_fallback!r}")
