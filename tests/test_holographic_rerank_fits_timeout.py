"""A rerank call asks no more of the endpoint than it answers within rerank_timeout (2026-10-29).

Requirement: a cross-encoder on a CPU scores the pairs at a steady speed, so its time grows with everything it is
sent. A llama-server (jina-reranker-v2, 2 threads) answers 5 short facts in 0.4 s, 15 in 1.2 s and 40 in 3.3 s —
over the default rerank_timeout of 3.0 s. The call failed, recall went on without the reranker and the endpoint was
paused for 60 s and longer, on exactly the large pools a big store gives. Once the endpoint's speed is known, a call
sends only as many of the first candidates as it scores in half the timeout; the rest keep their order behind the
scored ones, as the tail beyond rerank_top_k always did. A timeout after the connection was ready teaches the same:
the next call sends about half of what did not come back in time. A timeout before the connection was ready teaches
nothing. A fast endpoint is sent the whole pool, as before.

The fake endpoint below stands in for the opener: it marks the connection ready, as net's connection does, then
takes `per_char` seconds per character of every query-document pair it was sent."""
from __future__ import annotations

import json
import logging
import re
import socket
import threading
import time

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, rerank

RANK = {"rerank": "on", "rerank_url": "https://rerank.example.net/v1/rerank", "rerank_model": "fake-reranker"}
QUERY = "when does the backup run?"
BOXES = ["alder", "birch", "cedar", "dogwood", "elm", "fir", "ginkgo", "hazel", "juniper", "larch", "maple", "oak",
         "pine", "quince", "rowan", "spruce", "teak", "walnut", "willow", "yew", "acacia", "banyan", "cypress",
         "ebony", "hemlock", "laurel", "magnolia", "olive", "poplar", "sequoia", "tamarind", "aspen", "baobab",
         "catalpa", "eucalyptus", "hickory", "linden", "mahogany", "palm", "sycamore"]
DOCS = [f"The backup of the {box} box runs at {1 + i % 12} am." for i, box in enumerate(BOXES)]  # 40, ~40 chars
LONG = [f"The backup of the {box} box runs at night and copies the shared folders. " * 12 for box in BOXES[:2]]


class _Response:
    def __init__(self, body: bytes):
        self.body, self.at = body, 0

    def read1(self, n=-1):
        piece = self.body[self.at:] if n is None or n < 0 else self.body[self.at:self.at + n]
        self.at += len(piece)
        return piece

    read = read1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class SlowByText:
    """Stands in for rerank._OPENER: connected at once, then `per_char` seconds per character of the pairs; answers
    best first, by position (`steep`: scores 5, 0, -5, ... — the window keeps only the top, topped up to two)."""

    def __init__(self, per_char: float, steep: bool = False):
        self.per_char, self.steep = per_char, steep
        self.requests: list[dict] = []
        self.release = threading.Event()  # set: a sleeping answer ends at once (its caller has given up on it)

    def open(self, request, timeout=None):
        request.holder["connected"] = time.perf_counter()  # what net's connection marks once it is ready
        body = json.loads(request.data)
        self.requests.append(body)
        if self.per_char:
            self.release.wait(self.per_char * self.chars(body))
        n = len(body["documents"])
        score = (lambda i: 5.0 - 5.0 * i) if self.steep else (lambda i: float(n - i))
        return _Response(json.dumps({"results": [{"index": i, "relevance_score": score(i)}
                                                 for i in range(n)]}).encode())

    @staticmethod
    def chars(body: dict) -> int:
        return sum(len(body["query"]) + len(d) for d in body["documents"])


@pytest.fixture
def slow(monkeypatch):
    fake = SlowByText(0.001)  # that server's speed scaled down: 2 short facts ~0.13 s, 40 ~2.7 s
    monkeypatch.setattr(rerank, "_OPENER", fake)
    yield fake
    fake.release.set()


def _rescorer(**config) -> rerank.Rescorer:
    return rerank.Rescorer(rerank.settings_from({**RANK, "rerank_timeout": 1.0, **config}))


def _let_go(r: rerank.Rescorer, fake: SlowByText) -> None:
    """End an abandoned answer, so the next call is not refused as 'an earlier request is still running'."""
    fake.release.set()
    if r._worker is not None:
        r._worker.join(5)
    fake.release = threading.Event()


def test_a_large_pool_on_a_slow_endpoint_is_answered_in_part_not_failed(slow):
    r = _rescorer()
    assert r.scores(QUERY, DOCS[:2]) is not None, "a small pool is answered: the endpoint's speed is known"
    started = time.perf_counter()
    ranked = r.scores(QUERY, DOCS)
    took = time.perf_counter() - started
    assert ranked is not None and r.last_status == "answered", r.last_status
    assert r.failures == 0 and r.usable(), "a slow endpoint that answers is not paused"
    sent = slow.requests[-1]["documents"]
    assert 2 <= len(sent) < len(DOCS), f"{len(sent)} of {len(DOCS)} sent"
    assert sent == DOCS[:len(sent)], "the first candidates go out, in the order given"
    assert took < 1.0, f"the call took {took:.2f} s, over rerank_timeout"
    assert {i for i, _ in ranked} == set(range(len(sent)))


def test_a_fast_endpoint_keeps_getting_the_whole_pool(monkeypatch):
    fake = SlowByText(0.0)
    monkeypatch.setattr(rerank, "_OPENER", fake)
    r = _rescorer()
    for _ in range(4):  # each answer must come within half the timeout for the next to be whole: 0.5 s of slack
        assert r.scores(QUERY, DOCS) is not None
        assert len(fake.requests[-1]["documents"]) == len(DOCS) == r.settings.top_k


def test_a_timeout_after_connecting_halves_what_the_next_call_sends(slow, monkeypatch):
    """No answer has come yet, so the first call sends the whole pool and times out. The server had the whole wait and
    did not answer, so it is at least that slow: each next call sends about half of the one before, until one is
    answered."""
    monkeypatch.setattr(rerank, "PAUSE_FIRST", 0.0)  # retry at once instead of a minute later
    r = _rescorer()
    sent, answered = [], None
    for _ in range(5):
        answered = r.scores(QUERY, DOCS)
        sent.append(SlowByText.chars(slow.requests[-1]))
        if answered is not None:
            break
        assert r.last_status == "failed: no answer (timeout)", r.last_status
        _let_go(r, slow)
    assert len(slow.requests[0]["documents"]) == len(DOCS)
    assert all(later <= earlier * 0.55 for earlier, later in zip(sent, sent[1:])), sent
    assert answered is not None, f"never answered; characters sent {sent}"


def test_a_timeout_before_the_connection_is_ready_teaches_nothing():
    """A real socket: the server accepts TCP and never answers the TLS handshake, so the wait ends before a request
    went out. That says nothing of how fast the endpoint scores — the next call is still sent the whole pool."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    try:
        r = _rescorer(rerank_url=f"https://127.0.0.1:{server.getsockname()[1]}/v1/rerank", rerank_timeout=0.5)
        assert r.scores(QUERY, DOCS) is None
        assert r.last_status == "failed: no answer (timeout)", r.last_status
        assert r.fitting(QUERY, DOCS) == len(DOCS)
        if r._worker is not None:
            r._worker.join(5)
        assert r.fitting(QUERY, DOCS) == len(DOCS), "nor once the abandoned handshake has ended"
    finally:
        server.close()


def test_an_endpoint_slower_than_two_documents_is_still_sent_two(slow):
    """Two pairs in 1.2 s of a 2.0 s timeout: more than half the timeout. A call still sends two documents — one
    would not be reordered, none would be no request."""
    slow.per_char = 1.2 / SlowByText.chars({"query": QUERY, "documents": DOCS[:2]})
    r = _rescorer(rerank_timeout=2.0)
    assert r.scores(QUERY, DOCS[:2]) is not None
    assert r.scores(QUERY, DOCS) is not None, r.last_status
    assert slow.requests[-1]["documents"] == DOCS[:2]


def test_the_slowest_of_the_last_five_answers_decides(slow):
    r = _rescorer()
    assert r.scores(QUERY, DOCS[:2]) is not None  # one slow answer
    assert r.fitting(QUERY, DOCS) < len(DOCS)
    slow.per_char = 0.0  # the server is idle again; long pairs keep each fast answer's slack at ~0.3 s
    for fast in range(1, 6):
        assert r.scores(QUERY, LONG) is not None
        if fast < 5:
            assert r.fitting(QUERY, DOCS) < len(DOCS), f"after {fast} fast answers the slow one still counts"
    assert r.fitting(QUERY, DOCS) == len(DOCS), "after five fast answers the slow one no longer counts"


def test_recall_on_a_slow_endpoint_keeps_the_unsent_candidates_and_traces_what_is_sent(tmp_path, slow, caplog):
    slow.steep = True
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64,
                                          **RANK, "rerank_timeout": 1.0})
    p.initialize(session_id="s1")
    for fact in DOCS[:25] + ["Mark plays the violin on Sundays.", "Lena plays the violin at the school concert."]:
        p._store.add_fact(fact)
    p.sync_turn("x", "y")
    assert p.prefetch("who plays the violin?"), "a two-fact pool: the endpoint answers and its speed is known"
    assert len(slow.requests) == 1 and len(slow.requests[0]["documents"]) == 2
    caplog.set_level(logging.DEBUG, logger="hermes_plugin_holographic")
    block = p.prefetch(QUERY)
    r = p._rescorer
    assert r.last_status == "answered" and r.failures == 0, r.last_status
    sent = [re.search(r"the (\w+) box", d).group(1) for d in slow.requests[-1]["documents"]]
    assert 2 <= len(sent) < 25
    shown = re.findall(r"backup of the (\w+) box", block)
    # The window keeps the two best scored (steep scores), and the candidates that were not sent fill the rest.
    assert shown[:2] == sent[:2] and len(shown) == 5, (shown, sent)
    assert not set(shown[2:]) & set(sent), f"{shown[2:]} should be candidates that were not sent ({sent})"
    traced = [int(m.group(1)) for m in (re.search(r"rescore prefetch: pool \d+, candidates (\d+)", rec.getMessage())
                                        for rec in caplog.records) if m]
    assert traced == [len(sent)], f"the trace says {traced}, the request held {len(sent)}"
    p.shutdown()
