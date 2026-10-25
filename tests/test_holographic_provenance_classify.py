"""tools/provenance_probe: classify() and the rank mapping.

Class E («cut by the window»): the target is in the rank top-5 BEFORE the rerank_window cut, but not in recall —
before, such a miss lied with «below top-5 (pool#N)» (review 2026-10-23, Andrey's case).
Mapping: ranked indexes the send list of documents, not the pool — fact_id is computed
via union + send_idx, otherwise the meaning/entity streams put foreign ids in the JSONL.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import provenance_probe as pp  # noqa: E402 - the path to tools/ is added above


def _ctx(**over):
    base = {"tags": {}, "blocked": {}, "gate_called": True,
            "pool": [11, 12, 13, 14, 15, 16], "gate_order": [11, 12, 13, 14, 15, 16],
            "rank_order": [], "rerank_top5": [], "window": 0.0, "weight": 0.0}
    base.update(over)
    return base


def test_none_and_dropped():
    assert pp.classify(None, _ctx()) == "-"
    assert pp.classify(7, _ctx(tags={7: ["recent"]})) == "dropped:recent"


def test_window_cut_is_class_e():
    # it passed coverage and stood in the rank top-5, but the window cut it before the top-5 was served
    ctx = _ctx(rank_order=[17, 13, 14], rerank_top5=[17, 13, 14], window=0.05)
    assert pp.classify(17, ctx) == "E: срезан окном (ранг #0)"
    assert pp.classify(13, ctx) == "E: срезан окном (ранг #1)"


def test_no_window_no_e():
    # window 0 (reorder_only): position >= 5 honestly reads «below top-5 (rank)»
    ctx = _ctx(rank_order=[17, 13, 14, 18, 19, 20], window=0.0)
    assert pp.classify(20, ctx) == "C: ниже top-5 (ранг #5)"
    # weight > 0: the window does not cut (see _unified_stream) — E never triggers
    ctx = _ctx(rank_order=[17], window=0.05, weight=0.5)
    assert not pp.classify(17, ctx).startswith("E:")


def test_pool_classes():
    ctx = _ctx(pool=[11, 12], gate_order=[11, 12], blocked={12: 1})
    assert pp.classify(99, ctx) == "A: нет в словесном пуле"
    assert pp.classify(12, ctx) == "D: вырезан coverage (пул#1)"
    ctx = _ctx(gate_called=False, gate_order=[11, 12])
    assert pp.classify(11, ctx) == "gate: отвергнут margin (в top-5 порядка!)"


def test_probe_rank_ids_follow_send_map():
    rescorer = SimpleNamespace(
        scores=lambda query, docs: [(1, 0.9), (0, 0.2)],
        settings=SimpleNamespace(window=0.05, weight=0.0))
    prov = SimpleNamespace(
        _retriever=SimpleNamespace(),
        _rescorer=rescorer,
        _sendable_docs=pp.HolographicMemoryProvider._sendable_docs,
        prefetch=lambda query: "## block")
    probe = pp.Probe(prov)
    union = [{"fact_id": 7, "content": "alpha"},
             {"fact_id": 9, "content": "OPENAI_API_KEY=secret"},
             {"fact_id": 11, "content": "gamma"}]
    idx, docs = prov._sendable_docs(union)
    assert idx == [0, 2] and docs == ["alpha", "gamma"]  # the secret never reaches the endpoint
    rescorer.scores("q", docs)
    # ranked — indices into the send list: 1 -> union[2] = 11, 0 -> union[0] = 7
    assert probe.ctx["rank_order"] == [11, 7]
    assert probe.ctx["rerank_top5"] == [11, 7]
    # the window/weight are taken off the rescorer settings on every prefetch
    _, ctx = probe.prefetch("q")
    assert ctx["window"] == 0.05 and ctx["weight"] == 0.0