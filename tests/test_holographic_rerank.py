"""Rescoring by a cross-encoder (rerank.py, RFC #9) — optional, offline-safe, off by default.

A fake endpoint stands in for the /v1/rerank server: it answers with a fixed or reversed order of the documents
sent, or fails as told.

One mode since the 2026-10-22 review (item 6): the cross-encoder reorders the pool, admission stays with coverage.
`rerank_window` (δ) optionally cuts the tail below δ of the best score before the top five (coverage ran
earlier, on the lexical stream);
`rerank_weight` > 0 blends the score into the blend. rerank_mode / rerank_margin are gone — a test covers that
they are ignored with a warning. Any failure keeps the current order and pauses the endpoint."""
from __future__ import annotations

import http.server
import json
import threading
import logging
import re
import time

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, rerank
from hermes_plugin_holographic import _bottom_cut, _window_keep, blend_union


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
    """Stands in for rerank._OPENER: records every request, answers with a fixed or reversed order, or fails."""

    def __init__(self):
        self.requests: list[dict] = []
        self.fail = None  # an exception to raise, or bytes to answer with
        self.reverse = False  # answer with the documents reversed (worst ranker)
        self.delay = 0.0  # seconds to sleep in the "network" call
        self.scores = None  # optional relevance_scores to answer with (else n, n-1, ...)

    @property
    def documents(self) -> list[str]:
        return [d for r in self.requests for d in r["body"]["documents"]]

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body,
                              "timeout": timeout})
        if self.delay:
            time.sleep(self.delay)
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


RANK = {"rerank": "on", "rerank_url": "https://rerank.example.net/v1/rerank",
        "rerank_model": "fake-reranker"}


def _provider(tmp_path, name="memory_store.db", **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / name), "hrr_dim": 64, **config})
    p.initialize(session_id="s1")
    return p


FACTS = ["The nightly backup runs from the OES box at 2 am.", "Lena's birthday is on June 3.",
         "Mark plays the violin on Sundays.", "Tom is allergic to peanut satay.",
         # item 9: a single candidate makes no call — endpoint tests need a pool of >= 2
         "The weekly backup runs from the NAS box at 4 am."]
FILLERS = ["apple note", "brick note", "cloud note", "daisy note", "eagle note"]


def _fill(p):
    for f in FILLERS:
        p._store.add_fact(f"{f} number")
    ids = [p._store.add_fact(f) for f in FACTS]
    p.sync_turn("x", "y")
    return ids


def _block(p, query):
    return p.prefetch(query)


# -- off by default -------------------------------------------------------------------------------------------

def test_off_by_default_no_calls_same_order(tmp_path, endpoint):
    p = _provider(tmp_path)
    _fill(p)
    assert p._rescorer is None and endpoint.requests == []
    assert p.prefetch("when does the backup run?") != ""
    assert endpoint.requests == []
    p.shutdown()


def test_config_schema_defaults(tmp_path):
    p = _provider(tmp_path)
    schema = {f["key"]: f.get("default") for f in p.get_config_schema()
              if str(f["key"]).startswith("rerank")}
    assert schema["rerank"] == "off"
    assert schema["rerank_window"] == "0.05"  # item 6: one mode, the window is the key
    # items 1+3 (13:06): no window_min/window_named keys anymore — at least two are built
    # into the window; the scale is explicit, default logit (llama-server), no auto and no probe
    assert schema["rerank_scale"] == "logit"
    assert "rerank_window_min" not in schema and "rerank_window_named" not in schema
    assert schema["rerank_weight"] == "0.0"
    p.shutdown()


# -- the call: payload shape, order, failure ----------------------------------------------------------------

def test_order_only_reorders_and_sends_cohere_payload(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    block = _block(p, "when does the backup run?")
    assert block  # facts recalled
    req = endpoint.requests[0]
    assert req["body"]["query"]
    assert req["body"]["model"] == "fake-reranker"
    assert isinstance(req["body"]["documents"], list) and req["body"]["documents"]
    assert req["body"]["top_n"] >= 1
    p.shutdown()


def test_window_cut_uses_ranking(tmp_path, endpoint):
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    _fill(p)
    endpoint.scores = [1.0, 0.2]  # a confident top: the margin admits the pool
    block = _block(p, "when does the backup run?")
    assert block
    assert endpoint.requests  # the gate IS the reranker: it was called
    p.shutdown()


def test_bottom_cut_default_keeps_the_window(tmp_path, endpoint):
    """rerank_window cuts the pool FROM THE BOTTOM: everything within δ of the best score survives, the top
    always survives (a window can't reject the winner); recall still works after the cut."""
    assert _bottom_cut([0.99, 0.96, 0.93, 0.5], 0.05) == [0, 1]  # gap 0.03 in, 0.06 and 0.49 out
    assert _bottom_cut([0.4], 0.05) == [0]  # a lone candidate always survives
    assert _bottom_cut([], 0.05) == []
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    _fill(p)
    endpoint.scores = [1.0, 0.9, 0.89, 0.88, 0.87]  # flat tail — the DEFAULT has no ratio veto
    assert _block(p, "what about the note?") != ""  # unlike the enabled veto in the test above
    p.shutdown()


def test_window_keeps_two_scored_and_refills_missing(tmp_path):
    """Built-in window minimum (items 1+4 of the 13:06 review, no keys): the cut never drops
    # below two scored candidates; the fill-in goes by RANK positions that are missing —
    # keep does not have to be a prefix (`_window_keep([1], 3)` -> [0, 1], not [1, 1])."""
    assert _window_keep([0], 4) == [0, 1]        # tops up with the next one by rank
    assert _window_keep([1], 3) == [0, 1]        # item 4(4): its example — keep is not a prefix
    assert _window_keep([0, 1], 4) == [0, 1]     # already not below two
    assert _window_keep([0], 1) == [0]           # the single scored one is not resurrected
    assert _window_keep(None, 4) is None         # disabled -> not revived
    assert _window_keep([], 4) == []             # an empty cut is not revived
    # no keys anymore: Settings does not read them, the default scale is logit (item 3)
    s = rerank.Settings({"rerank": "on"})
    assert not hasattr(s, "window_min") and not hasattr(s, "window_named")
    assert s.scale == "logit"


def test_window_never_cuts_below_two(tmp_path, endpoint):
    """Integration (items 1+4(1) of the 13:06 review; fix at 14:01): 3+ candidates and an explicit assert
    # on what remains — on two candidates the test would pass even if the window cut nothing."""
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    _fill(p)
    endpoint.scores = [1.0, 0.0, 0.0, 0.0, 0.0]  # 5 notes: only the top ones stay within δ
    block = _block(p, "what about the note?")
    req = endpoint.requests[0]["body"]["documents"]
    assert len(req) >= 3  # otherwise the window has nothing to cut at all
    lines = [ln for ln in block.splitlines() if ln.startswith("- #")]
    assert len(lines) == 2  # the window cut 5 -> 1, the built-in minimum brought the second back
    kept = [ln.split("] ", 1)[1] for ln in lines]
    assert kept == req[:2]  # exactly the first two by rank remain (ids are visible in the lines)
    p.shutdown()


def test_gate_reads_score_falls_back_to_coverage_on_failure(tmp_path, endpoint):
    """A reranker outage must not empty recall: the pipeline falls back to the coverage gate."""
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    _fill(p)
    endpoint.fail = ConnectionRefusedError("refused")
    block = _block(p, "when does the backup run?")
    assert block  # the coverage gate still recalls the fact that answers the words
    p.shutdown()


def test_gate_reads_score_single_candidate_uses_coverage(tmp_path, endpoint):
    """A single candidate goes to the coverage gate like any other pool."""
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    _fill(p)
    # "apple" matches exactly one filler ("apple note number"); the reranker finds it irrelevant (-10)
    calls = []
    original = p._retriever.relevant

    def spy(query, candidates, min_coverage):
        calls.append(len(candidates))
        return original(query, candidates, min_coverage)

    p._retriever.relevant = spy
    endpoint.scores = [-10.0]
    _block(p, "what about apple?")
    assert calls == [1]  # the single candidate went to the coverage gate, not straight through
    p.shutdown()


def test_failure_keeps_current_order_and_pauses(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = ConnectionRefusedError("refused")
    before = p.prefetch("when does the backup run?")
    # backoff: the endpoint is paused, so no further call is made
    calls = len(endpoint.requests)
    # a different session (the repeat window would drop the same fact) and a fresh query
    after = p.prefetch("what does Tom avoid?")
    assert len(endpoint.requests) == calls  # paused: no second request
    assert before and after  # both prefetches worked (order kept, not empty)
    assert p._rescorer.status().startswith("paused")  # the failure is counted
    p.shutdown()


def test_malformed_answer_keeps_order(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = b'{"results": [{"index": 99, "relevance_score": 1.0}]}'
    block = _block(p, "when does the backup run?")
    assert block  # the malformed answer did not take recall down
    assert p._rescorer.status().startswith("paused")  # counted as a failure
    p.shutdown()


def test_huge_score_overflow_counts_as_failed(tmp_path, endpoint):
    """10**400 — valid JSON: math.isfinite(int) raises OverflowError, which bypasses
    except ValueError in _request and used to kill the call as not_called instead of failed — a fail-fast run
    never saw such an answer. Now it is a malformed answer, like any other invalid score.
    (item 1 of the maintainer's pre-merge checklist, PR #10)."""
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = ('{"results": [{"index": 0, "relevance_score": %d}]}' % (10 ** 400)).encode()
    block = _block(p, "when does the backup run?")
    assert block  # the order is not lost
    assert p._rescorer.last_status.startswith("failed: malformed answer")
    assert p._rescorer.status().startswith("paused")  # the error counted
    p.shutdown()


def test_window_keeps_unscored_tail_in_block(tmp_path, endpoint):
    """The window does not lose facts the reranker never saw: the union tail (non-sendable facts and positions past
    top_k) enters fresh after keep — the _sendable_docs docstring promises «the pool composition for recall does not
    change». Maintainer 24:10, item 1: rerank_top_k=2 with a pool of three — the third fact is not sent
    and must survive the window in the BLOCK (the block is checked, not the payload)."""
    p = _provider(tmp_path, **{**RANK, "rerank_top_k": "2"})
    facts = ["The nightly backup runs from the OES box at 2 am.",
             "The weekly backup runs from the NAS box at 4 am.",
             "The weekly backup runs from the tapes in the garage."]
    for f in facts:
        p._store.add_fact(f)
    p.sync_turn("x", "y")
    block = _block(p, "when does the backup run?")
    assert endpoint.requests, "the rerank call must happen (pool of three)"
    sent = endpoint.requests[-1]["body"]["documents"]
    assert len(sent) == 2, "top_k=2: only two go to the endpoint"
    for f in facts:
        assert f in block, f"a fact outside ranked must survive the window in the block: {f[:30]}"
    p.shutdown()


def test_stage_exception_keeps_order_and_counts(tmp_path, endpoint, monkeypatch):
    """Any exception in the rerank stage (call, mapping, window) = a failed call: the off layout in
    the block, the endpoint pauses, failures == 1, failed in last_status. Previously the exception fell into the
    prefetch handler — the turn went out WITHOUT facts, failures stayed 0, the endpoint was never paused.
    Maintainer 24:10, item 3 (Codex reproduced Thread.start() -> RuntimeError)."""
    p = _provider(tmp_path, **RANK)
    _fill(p)

    def boom(*args, **kwargs):
        raise RuntimeError("Thread.start() exploded")

    monkeypatch.setattr(p._rescorer, "scores", boom)
    block = _block(p, "when does the backup run?")
    assert block, "an exception in the stage must not take the turn's block away"
    assert "The nightly backup runs from the OES box at 2 am." in block  # the off layout, not emptiness
    assert p._rescorer.failures == 1, "a stage failure must count as a failure"
    assert p._rescorer.status().startswith("paused"), "the endpoint must pause"
    assert p._rescorer.last_status.startswith("failed: rerank stage")
    p.shutdown()


def test_no_second_worker_while_one_is_alive(tmp_path, endpoint):
    """item 5: after join(timeout) the worker keeps living (the server sends one byte per 100 ms — up to
    MAX_RESPONSE), and after the pause the next call used to start a SECOND one. Now the previous worker
    is tracked: while it is alive no new request starts and that counts as a failure — the minimal
    variant sanctioned by the maintainer 24:10 (a full counterpart of embeddings._abandoned/
    _reap_locked can be fleshed out separately if needed)."""
    p = _provider(tmp_path, **{**RANK, "rerank_timeout": "0.5"})
    _fill(p)
    endpoint.delay = 1.0  # above the 0.5 clamp: the worker hung inside open()
    block = _block(p, "when does the backup run?")
    assert block  # the turn was not hurt
    r = p._rescorer
    assert r.failures == 1, "the timeout counted"
    assert r._worker is not None and r._worker.is_alive(), "a live worker must be tracked"
    with r._lock:  # lift the pause to reach a new call, as after it expires
        r._pause = r._paused_until = 0.0
    sent = len(endpoint.requests)
    ranked = r.scores("when does the backup run?",
                      ["The nightly backup runs from the OES box at 2 am.",
                       "The weekly backup runs from the NAS box at 4 am."])
    assert ranked is None, "there must be no new answer"
    assert len(endpoint.requests) == sent, "the second request must not start while the worker is alive"
    assert r.failures == 2, "a failure with a live worker must count"
    r._worker.join(3)  # the test leaves no thread behind
    p.shutdown()


def test_secret_straddling_the_2000_cut_is_screened(tmp_path, endpoint):
    """The secret screen runs over the WHOLE request: a key starting before the 2000th character hit the [:2000] slice
    as a fragment («pas») and passed the screen, while the first letters went to the endpoint. Maintainer 24:10,
    item 2 (the same bug in main — their PR #21). A direct scores() call: through prefetch a straddle request
    also drops coverage («TopSecret123» as an identifier), and the bug lives precisely in the
    last line of defence of rerank.py."""
    p = _provider(tmp_path, **RANK)
    _fill(p)
    base = "when does the backup run? "
    query = (base + " backup" * 300)[:1996] + " password: TopSecret123"
    assert len(query) > 2000 and query.index("password") < 2000 < query.index("TopSecret")
    docs = ["The nightly backup runs from the OES box at 2 am.",
            "The weekly backup runs from the NAS box at 4 am."]
    ranked = p._rescorer.scores(query, docs)
    assert ranked is None, "a truncated key must not pass the screen"
    assert p._rescorer.last_status == "skipped_secret"
    assert endpoint.requests == [], "a request carrying a key fragment must not go out"
    p.shutdown()


def test_secret_in_query_no_call(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    block = _block(p, "password: Qwerty123 what is it?")
    assert endpoint.requests == []  # what recall must not inject is never sent
    assert block is not None
    p.shutdown()


def test_timeout_keeps_order(tmp_path, endpoint, monkeypatch):
    """Maintainer 24:10: the old version did not catch the timeout — rerank_timeout 0.2 clamps to 0.5, while
    delay was exactly 0.5, and in its three runs the status was answered. Now delay is far above
    the clamp, and the test catches the timeout itself: pause + failures == 1, not just a live block."""
    p = _provider(tmp_path, **{**RANK, "rerank_timeout": "0.2"})
    _fill(p)
    endpoint.delay = 2.0  # well above the 0.5 clamp (0.2 -> 0.5)
    start = time.perf_counter()
    block = _block(p, "when does the backup run?")
    elapsed = time.perf_counter() - start
    assert block  # recall did not hang
    assert elapsed < 1.0  # the caller waited rerank_timeout (clamped to 0.5), not the delay
    r = p._rescorer
    assert r.failures == 1, "the timeout must count as a failure"
    assert r.status().startswith("paused"), "the endpoint must pause"
    assert r.last_status.startswith("failed")
    if r._worker is not None:
        r._worker.join(3)  # the worker sleeps out its own delay — the test leaves no thread behind
    p.shutdown()


# -- a real slow server ------------------------------------------------------------------------------------------

class _TrickleHandler(http.server.BaseHTTPRequestHandler):
    """Serves the body BYTE BY BYTE with a pause — the «slow body» from the 19:39 review (item 1)."""

    payload = b""
    delay = 0.1

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        for byte in self.payload:
            self.wfile.write(bytes([byte]))
            self.wfile.flush()
            time.sleep(self.delay)

    def log_message(self, *a):
        pass


def test_slow_body_returns_at_rerank_timeout_not_at_the_body(tmp_path):
    """Slow body server (maintainer 19:39, item 1): 21 bytes at 100 ms each = 2.1 s with rerank_timeout 0.5.

    Caller-side close() waited for the worker stuck in read(): scores() returned at 2.0 s (21 bytes) and
    10.1 s (101 bytes), the provider lock was held for the whole turn, the core (>8 s) dropped recall. Fake close()s
    do not wait and return at once — only a real server catches this. The worker must cut itself off by
    its own deadline (read1 + with on its own thread); the caller waits only for join(timeout)."""
    handler = type("H", (_TrickleHandler,), {"payload": b'{"results":[{"i":0}]}', "delay": 0.1})
    class _Quiet(http.server.ThreadingHTTPServer):
        # the worker drops the connection on its own deadline — a break in the handler is expected,
        # no traceback needed here
        def handle_error(self, *a):
            pass

    server = _Quiet(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        p = _provider(tmp_path, **{**RANK,
                                   "rerank_url": f"http://127.0.0.1:{server.server_address[1]}/v1/rerank",
                                   "rerank_timeout": "0.5"})
        _fill(p)
        start = time.perf_counter()
        _block(p, "when does the backup run?")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.5, f"caller waited for the server body: {elapsed:.2f} s instead of ~rerank_timeout (0.5)"
        r = p._rescorer
        assert r.failures == 1, "a break at the worker's deadline must count as a failure"
        assert r.status().startswith("paused"), "the endpoint pauses"
        if r._worker is not None:
            r._worker.join(3)  # the worker closes the connection itself and dies — no thread is left behind
        p.shutdown()
    finally:
        server.shutdown()
        server.server_close()


# -- blend (weight > 0) --------------------------------------------------------------------------------------

def test_weight_zero_is_order_only(tmp_path, endpoint):
    p = _provider(tmp_path, **{**RANK, "rerank_weight": "0.0"})
    _fill(p)
    _block(p, "when does the backup run?")
    assert endpoint.requests  # still called: order-only is the default
    p.shutdown()


def test_weight_blend_runs_and_keeps_recall(tmp_path, endpoint):
    p = _provider(tmp_path, **{**RANK, "rerank_weight": "0.5"})
    _fill(p)
    block = _block(p, "when does the backup run?")
    assert block and endpoint.requests  # blend recomputes score and sorts; recall still works
    p.shutdown()


# -- the endpoint's own API ---------------------------------------------------------------------------------

def test_settings_validation():
    assert rerank.settings_from({}) is None
    assert rerank.settings_from({"rerank": "off"}) is None
    s = rerank.Settings({"rerank": "on", "rerank_top_k": "15", "rerank_window": "0.1"})
    assert (s.top_k, s.window) == (15, 0.1)
    assert rerank.Settings({"rerank": "on"}).window == rerank.DEFAULT_WINDOW  # the window default

    # item 4 (maintainer 24:10) + item 2 (19:39): the core budget — the provider prefetch is 8 s, the sum of
    # embedding_timeout + rerank_timeout must fit inside it; emb counts ONLY
    # with embeddings: on (irrelevant tuning must not cut rerank), and the default rerank_timeout
    # also passes the cap (_number returns the default WITHOUT clamping — their finding).
    assert rerank.Settings({"rerank": "on", "embeddings": "on",
                             "embedding_timeout": "6", "rerank_timeout": "10"}).timeout == 1.5
    assert rerank.Settings({"rerank": "on", "embeddings": "on",
                             "embedding_timeout": "0.5", "rerank_timeout": "10"}).timeout == 7.0
    assert rerank.Settings({"rerank": "on", "embeddings": "on",
                             "embedding_timeout": "6"}).timeout == 1.5  # the default 3 is capped too
    assert rerank.Settings({"rerank": "on",
                             "embedding_timeout": "6", "rerank_timeout": "10"}).timeout == 7.5  # emb off
    assert rerank.Settings({"rerank": "on"}).timeout == 3.0  # emb off: the default 3 fits into 7.5


def test_removed_keys_are_ignored_with_a_warning(caplog):
    """rerank_mode / rerank_margin are gone (item 6): an old configuration breaks nothing, the key is ignored."""
    with caplog.at_level(logging.WARNING):
        s = rerank.Settings({"rerank": "on", "rerank_mode": "gate_reads_score", "rerank_margin": "1.5"})
    assert s.window == rerank.DEFAULT_WINDOW
    assert "rerank_mode was removed" in caplog.text and "rerank_margin was removed" in caplog.text


def test_ranked_shape():
    full = {"results": [{"index": 2, "relevance_score": -0.239},
                        {"index": 1, "relevance_score": -1.5},
                        {"index": 0, "relevance_score": -2.877}]}
    assert rerank._ranked(full, 3) == [(2, -0.239), (1, -1.5), (0, -2.877)]
    # the answer order is not trusted: a shuffled server list is sorted by score
    shuffled = {"results": [{"index": 1, "relevance_score": -1.5},
                            {"index": 2, "relevance_score": -0.239},
                            {"index": 0, "relevance_score": -2.877}]}
    assert rerank._ranked(shuffled, 3) == [(2, -0.239), (1, -1.5), (0, -2.877)]
    # item 4(3): a partial answer (fewer indices than documents) is an error, not «a smaller pool»:
    # the request always ships with top_n >= the number of documents, the server must answer them all.
    with pytest.raises(ValueError):
        rerank._ranked({"results": [{"index": 2, "relevance_score": -0.239},
                                     {"index": 0, "relevance_score": -2.877}]}, 5)  # 2 out of 5
    with pytest.raises(ValueError):
        rerank._ranked({"results": [{"index": 9, "relevance_score": 1.0}]}, 5)  # out of range
    with pytest.raises(ValueError):
        rerank._ranked({"results": [{"index": 0, "relevance_score": "x"}]}, 5)  # not a number
    with pytest.raises(ValueError):
        # 10**400 — valid JSON; math.isfinite(int) raises OverflowError, bypassing the except
        # ValueError in _request — before the fix such an answer was not counted as failed (item 1 of the checklist).
        rerank._ranked({"results": [{"index": 0, "relevance_score": 10 ** 400}]}, 5)  # isfinite overflow
    with pytest.raises(ValueError):
        rerank._ranked({"results": []}, 5)  # empty


def test_endpoint_validation():
    local = rerank._endpoint("http://127.0.0.1:8080/v1/rerank")
    assert local[1] is True  # local
    assert rerank._endpoint("http://example.com/v1/rerank")[0] is None  # public http refused
    assert rerank._endpoint("ftp://x")[0] is None  # wrong scheme


# -- fallback = off, the endpoint scale (item 3 of the 13:06 review), blend on one scale ------------------------------

def test_scale_is_explicit_and_never_probes(tmp_path, endpoint):
    """Item 3 of the 13:06 review: the scale is explicit (default logit), no latch and no probe — not a single
    # extra request: a probe could latch wrongly, was resolved without a lock (a race) and
    added a second request to the turn of up to 2 × rerank_timeout. An answer in [0,1] does not flip the scale —
    it is read on the same one."""
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    rescorer = p._rescorer
    assert rescorer.settings.scale == "logit"
    first = rescorer.normalized([(0, 5.0), (1, -3.0)])  # logits -> sigmoid
    assert first == pytest.approx([rerank._sigmoid(5.0), rerank._sigmoid(-3.0)])
    second = rescorer.normalized([(0, 0.9), (1, 0.7)])  # in [0,1] — the same scale
    assert second == pytest.approx([rerank._sigmoid(0.9), rerank._sigmoid(0.7)])
    assert endpoint.requests == []  # there was and will be no probe
    p.shutdown()


def test_scale_unit_is_literal_and_bad_value_names_the_key(tmp_path):
    """Items 2+3 (13:06): unit is honoured literally; a value outside {unit, logit} is
    a named config problem (memory off), not a silent fallback to auto."""
    unit = _provider(tmp_path, "unit.db", **{**RANK, "rerank_scale": "unit"})
    assert unit._rescorer.settings.scale == "unit"
    assert unit._rescorer.normalized([(0, -5.0), (1, -3.0)]) == [-5.0, -3.0]  # unit forced: as is
    unit.shutdown()
    bad = _provider(tmp_path, "bad.db", **{**RANK, "rerank_scale": "banana"})
    assert any("rerank_scale" in problem for problem in bad._config_problems)
    bad.shutdown()


def test_blend_union_scores_every_member_on_one_scale():
    """item 4(4): with rerank_weight > 0 the blend is built for ALL pool members: unscored ones
    (beyond top_k) get half of the min-max scale instead of staying on the raw first score — otherwise
    the union is sorted by two scales (the weight test only covered a fully scored pool)."""
    union = [{"score": 0.4}, {"score": 0.6}, {"score": 0.9}, {"score": 0.8}, {"score": 0.5}]
    out = blend_union(union, [(0, 1.0), (1, 0.0)], 0.5)  # the endpoint scored only two
    # 0: 0.5*0.4+0.5*1.0 = 0.7 (the normalized maximum), 1: 0.5*0.6+0.5*0 = 0.3,
    # 2-4: 0.5*orig + 0 (half of the scale) -> 0.45 / 0.4 / 0.25
    assert [round(f["score"], 3) for f in out] == [0.7, 0.45, 0.4, 0.3, 0.25]
    # The old behaviour left 2-4 on the raw 0.9/0.8/0.5 — they would have ranked ahead of 0.7.


def test_failed_call_shows_exactly_the_off_block(tmp_path, endpoint):
    """item 4(1): a failed call falls back to the off layout, including the fact that the entity stream
    adds to off beyond the five preserved lexical ones (it used to get lost in the fallback:
    the extras were computed over the whole lexicon)."""
    lena_facts = ["Lena's birthday is on June 3.", "Lena drinks tea with lemon.",
                  "Lena works at the archive on Mondays.", "Lena's brother is Andrey.",
                  "Lena dislikes strong coffee.", "Lena runs the book club in the library."]

    def fill(p):
        for f in FILLERS:
            p._store.add_fact(f"{f} number")
        for text in lena_facts:
            p._store.add_fact_status(text, entities=("Lena",))  # the link is set explicitly (harness)

    off = _provider(tmp_path, "off.db")
    fill(off)
    down = _provider(tmp_path, "down.db", **RANK)
    fill(down)
    endpoint.fail = ConnectionRefusedError("refused")
    off_block = _block(off, "what about Lena?")
    down_block = _block(down, "what about Lena?")
    endpoint.fail = None
    assert off_block == down_block  # the block on failure is exactly what a disabled rerank would show
    assert len(re.findall(r"- #\d+", off_block)) == 6  # five lexical ones + the sixth from the entity stream
    off.shutdown()
    down.shutdown()
