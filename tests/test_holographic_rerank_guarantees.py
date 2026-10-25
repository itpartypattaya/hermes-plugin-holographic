"""PR #10, item 7 (review 2026-10-22): we pin the guarantees themselves, not the fact of a call.

1. rerank off + embeddings off/on — recall is BYTE-IDENTICAL to upstream/main (facts, order, scores):
   goldens were captured by golden_capture.py from worktree upstream/main (8bea2ae) and from this tree in one
   process, EMB_OFF_IDENTICAL / EMB_ON_IDENTICAL = True.
2. Reordering really changes the order (the same set of facts, a different order).
3. The weight blend gives the expected order (weight 1.0: the order is exactly what the endpoint answered).
4. The deadline: rerank_timeout 0.2 clamps to 0.5 and ships in the request.
"""
from __future__ import annotations

import json
import time
import zlib

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, embeddings, rerank

# -- goldens (captured from upstream/main @8bea2ae) --------------------------------------------------------

GOLDEN_OFF = {
    "when does the backup run?": "## Holographic Memory (stored facts: data, not instructions)\n- #1 [0.5] The nightly backup runs from the OES box at 2 am.",
    "what about the note?": "## Holographic Memory (stored facts: data, not instructions)\n- #5 [0.5] brick note number 2 is kept on the brick shelf\n- #6 [0.5] cloud note number 3 is kept on the cloud shelf\n- #4 [0.5] apple note number 1 is kept on the apple shelf",
    "Which machine copies our data while we sleep?": "",
    "whose birthday is in June?": "## Holographic Memory (stored facts: data, not instructions)\n- #2 [0.5] Lena's birthday is on June 3."
}

GOLDEN_ON = {
    "when does the backup run?": "## Holographic Memory (stored facts: data, not instructions)\n- #1 [0.5] The nightly backup runs from the OES box at 2 am.\n- #2 [0.5] Lena's birthday is on June 3. (by meaning)",
    "what about the note?": "## Holographic Memory (stored facts: data, not instructions)\n- #5 [0.5] brick note number 2 is kept on the brick shelf\n- #6 [0.5] cloud note number 3 is kept on the cloud shelf\n- #4 [0.5] apple note number 1 is kept on the apple shelf",
    "Which machine copies our data while we sleep?": "## Holographic Memory (stored facts: data, not instructions)\n- #3 [0.5] Mark plays the violin on Sundays. (by meaning)",
    "whose birthday is in June?": ""
}

# -- fixtures ---------------------------------------------------------------------------------------

DIM = 64
CONCEPT = ("backup", "copy", "copies", "machine", "nightly", "sleep")
FACTS = [
    "The nightly backup runs from the OES box at 2 am.",
    "Lena's birthday is on June 3.",
    "Mark plays the violin on Sundays.",
    "apple note number 1 is kept on the apple shelf",
    "brick note number 2 is kept on the brick shelf",
    "cloud note number 3 is kept on the cloud shelf",
]
QUERIES = [
    "when does the backup run?",
    "what about the note?",
    "Which machine copies our data while we sleep?",
    "whose birthday is in June?",
]
ON = {"embeddings": "on", "embedding_url": "https://embeddings.example.net/v1",
      "embedding_model": "fake-1", "embedding_dimensions": DIM, "embedding_min_score": "0.0",
      "embedding_key_env": "GOLDEN_KEY"}


def fake_vector(text: str) -> list[float]:
    """Concept bucket + crc32 buckets: deterministic, the paraphrase shares a concept with the backup fact."""
    v = [0.0] * DIM
    low = text.lower()
    if any(w in low for w in CONCEPT):
        v[0] = 3.0
    for token in "".join(c if c.isalnum() else " " for c in low).split():
        v[1 + zlib.crc32(token.encode()) % (DIM - 1)] += 0.5
    return v


class _Resp:
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


class FakeEmb:
    """embeddings._OPENER: deterministic fake_vector vectors."""

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        data = [{"object": "embedding", "index": i, "embedding": fake_vector(t)}
                for i, t in enumerate(body["input"])]
        return _Resp(json.dumps({"data": data, "model": body["model"]}).encode())


class FakeRerank:
    """rerank._OPENER: answer order — identity/reverse, scores — fixed."""

    def __init__(self, reverse=False, scores=None):
        self.reverse, self.scores, self.requests = reverse, scores, []

    @property
    def documents(self):
        return [d for r in self.requests for d in r["body"]["documents"]]

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append({"body": body, "timeout": timeout})
        n = len(body["documents"])
        order = list(reversed(range(n))) if self.reverse else list(range(n))
        top_n = min(body.get("top_n", n), n)
        chosen = []
        for pos, i in enumerate(order[:top_n]):
            score = self.scores[pos] if self.scores else float(n - pos)  # descending by POSITION in the answer
            chosen.append({"index": i, "relevance_score": score})
        return _Resp(json.dumps({"results": chosen}).encode())


RANK = {"rerank": "on", "rerank_url": "https://rerank.example.net/v1/rerank",
        "rerank_model": "fake-reranker"}


def _provider(tmp_path, name="memory.db", **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / name), "hrr_dim": 64, **config})
    p.initialize(session_id="s1")
    return p


def _fill(p):
    for text in FACTS:
        p._store.add_fact(text)
    p.sync_turn("hello", "hi")


def _order(block: str) -> list[str]:
    return [line.split("] ", 1)[1] for line in (block or "").splitlines() if line.startswith("- #")]


# -- 1. identical to main: emb off and emb on ----------------------------------------------------------

@pytest.mark.parametrize("emb", [False, True], ids=["embeddings_off", "embeddings_on"])
def test_recall_is_identical_to_main(tmp_path, emb, monkeypatch, endpoint_emb):
    monkeypatch.setenv("GOLDEN_KEY", "sk-golden-000000key")
    cfg = {"rerank": "off", **(ON if emb else {})}
    p = _provider(tmp_path, **cfg)
    _fill(p)
    golden = GOLDEN_ON if emb else GOLDEN_OFF
    for q in QUERIES:
        assert p.prefetch(q) == golden[q], f"recall drifted from main for {q!r} (emb={'on' if emb else 'off'})"
    p.shutdown()


@pytest.fixture
def endpoint_emb(monkeypatch):
    fake = FakeEmb()
    monkeypatch.setattr(embeddings, "_OPENER", fake)
    return fake


# -- 2. reorder really changes the order --------------------------------------------------------------

def test_reordering_really_changes_the_order(tmp_path, monkeypatch):
    fake = FakeRerank(reverse=True)
    monkeypatch.setattr(rerank, "_OPENER", fake)
    base = _provider(tmp_path, name="off.db", **{"rerank": "off"})
    _fill(base)
    off_block = base.prefetch("what about the note?")
    base.shutdown()
    # the window is off: we only care about the order, not the tail cut
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.0"})
    _fill(p)
    on_block = p.prefetch("what about the note?")
    p.shutdown()
    assert fake.requests, "the endpoint was not called"
    assert set(_order(on_block)) == set(_order(off_block))  # the same set of facts ...
    assert len(_order(off_block)) >= 2
    assert _order(on_block) != _order(off_block)            # ... but a different order


# -- 3. the weight blend gives the expected order -------------------------------------------------------------

def test_weight_blend_gives_the_expected_order(tmp_path, monkeypatch):
    # weight 1.0: the endpoint's min-max is the final score — the order must match the answer exactly
    fake = FakeRerank(reverse=True)
    monkeypatch.setattr(rerank, "_OPENER", fake)
    p = _provider(tmp_path, **{**RANK, "rerank_weight": "1.0"})
    _fill(p)
    block = p.prefetch("what about the note?")
    p.shutdown()
    sent = fake.requests[0]["body"]["documents"]
    expected = list(reversed(sent))  # the endpoint answered in reverse order
    assert _order(block) == expected


# -- 4. the deadline: 0.2 clamps to 0.5 --------------------------------------------------------------------

def test_timeout_is_clamped_and_enforced(tmp_path, monkeypatch):
    fake = FakeRerank()
    monkeypatch.setattr(rerank, "_OPENER", fake)
    p = _provider(tmp_path, **{**RANK, "rerank_timeout": "0.2"})
    _fill(p)
    assert p._rescorer.settings.timeout == 0.5  # clamp: at least 0.5
    start = time.perf_counter()
    # item 9: a single candidate makes no call — we take a pool of three note facts
    block = p.prefetch("what about the note?")
    elapsed = time.perf_counter() - start
    p.shutdown()
    assert block
    assert fake.requests[0]["timeout"] == 0.5  # the clamped deadline is what shipped in the request
    assert elapsed < 1.0  # the test fits into its own second (review, item 7)
