# -*- coding: utf-8 -*-
"""PR #10, item 9: one sendable candidate — no call, the output matches
the reranked path (maintainer, 2026-10-23):

    "No call for a single candidate: with one sendable document there is nothing
    to reorder, and the window cannot drop the only one. Please add it with a test
    that the output matches the reranked path."
"""
from __future__ import annotations

import json

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, rerank

RANK = {"rerank": "on",
        "rerank_url": "https://rerank.example.net/v1/rerank", "rerank_model": "fake-reranker"}
FACT = "The nightly backup runs from the OES box at 2 am."
QUERY = "when does the backup run?"


# -- fake endpoint (pattern from tests/..._rerank_review; conftest — a follow-up after the merge) --------


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
    """Records every request; answers identity (or fails as told)."""

    def __init__(self):
        self.requests: list[dict] = []
        self.fail = None
        self.score = None  # set -> the server answers with this logit (for a single doc)

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append({"url": request.full_url, "body": body, "timeout": timeout})
        if isinstance(self.fail, BaseException):
            raise self.fail
        n = len(body["documents"])
        if self.score is not None and n == 1:
            chosen = [{"index": 0, "relevance_score": float(self.score)}]
        else:
            chosen = [{"index": i, "relevance_score": float(n - i)}
                       for i in range(min(body.get("top_n", n), n))]
        return _Response(json.dumps({"results": chosen}).encode())


@pytest.fixture
def endpoint(monkeypatch):
    fake = FakeRerank()
    monkeypatch.setattr(rerank, "_OPENER", fake)
    return fake


def _rescorer():
    settings = rerank.settings_from(RANK)
    assert settings is not None
    return rerank.Rescorer(settings)


# -- a single candidate: no call -------------------------------------------------------


def test_single_candidate_scores_make_no_call(endpoint):
    rescorer = _rescorer()
    assert rescorer.scores(QUERY, [FACT]) == [(0, 1.0)]
    assert rescorer.order(QUERY, [FACT]) == [0]
    assert rescorer.calls == 0
    assert endpoint.requests == [], "one document must not reach the endpoint"


def test_two_candidates_still_call_the_endpoint(endpoint):
    rescorer = _rescorer()
    ranked = rescorer.scores(QUERY, [FACT, "Another note about tapes."])
    assert ranked is not None and len(ranked) == 2
    assert len(endpoint.requests) == 1, "the shortcut must not swallow real pools"


# -- the output matches the reranked path ---------------------------------------------------


def _build(tmp_path, name, **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / name), "hrr_dim": 64, **config})
    p.initialize(session_id="s1")
    p._store.add_fact(FACT)  # exactly one fact => a sendable pool of length 1
    p.sync_turn("x", "y")
    return p


@pytest.mark.parametrize("extra", [
    {"rerank_weight": "0.0", "rerank_window": "0.05"},  # gate mode: rank + window
    {"rerank_weight": "0.5"},                          # blend
    {"rerank_weight": "0.0", "rerank_window": "0.0"},  # order without a cut
])
def test_single_candidate_output_matches_the_ranked_path(tmp_path, endpoint, extra, monkeypatch):
    # path A = the reranked path, as BEFORE the fix: the call is made, the server answers
    # with any logit (–2.5: outside [0,1], sigmoid normalization kicks in)
    def endpoint_path_scores(self, query, documents):
        if not documents or not self.usable():
            return None
        if self._secret and self._secret((query or "")[:2000]):
            return None
        limit = int(self.settings.top_k)
        docs = [str(documents[i])[:2000] for i in range(min(limit, len(documents)))]
        return self._request({"model": self.settings.model, "query": (query or "")[:2000],
                              "documents": docs, "top_n": limit})
    original = rerank.Rescorer.scores
    monkeypatch.setattr(rerank.Rescorer, "scores", endpoint_path_scores)
    endpoint.score = -2.5
    tag = f"{extra['rerank_weight']}_{extra.get('rerank_window', 'x')}"
    ranked_provider = _build(tmp_path, f"one_{tag}.db", **RANK, **extra)
    block_ranked = ranked_provider.prefetch(QUERY)
    assert len(endpoint.requests) == 1, "sanity: the ranked path really calls"
    ranked_provider.shutdown()

    # path B: the shortcut is on, the call is forbidden (it fails unless it is avoided)
    monkeypatch.setattr(rerank.Rescorer, "scores", original)
    endpoint.requests.clear()
    endpoint.fail = AssertionError("single candidate must not be sent")
    skip_provider = _build(tmp_path, f"one_skip_{tag}.db", **RANK, **extra)
    block_skip = skip_provider.prefetch(QUERY)
    skip_provider.shutdown()

    assert endpoint.requests == [], "no call for a single candidate"
    assert block_skip == block_ranked, "the skipped output must match the reranked path"
    assert block_ranked.strip(), "the single fact must survive (the window keeps the top)"