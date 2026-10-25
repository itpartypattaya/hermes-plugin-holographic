# -*- coding: utf-8 -*-
"""Item 9: the household set is frozen and self-consistent (the maintainer's recipe in PR #10,
2026-10-22: one house ~200 facts, en/ru/zh, dev/test, a label for every question).

self_check inside household.py is the full check (the targets exist, the slices and
counters, nooverlap without shared words). Here — freeze pinning: a change to the
data without re-consensus must be visible in the test.
"""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("household", REPO / "tools" / "household.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_self_check_passes():
    _load().self_check()  # SystemExit = a data invariant is violated


def test_counts_are_frozen():
    m = _load()
    assert len(m.FACTS) == 199            # ~200 facts (the recipe)
    assert len(m.QUERIES) == 70           # dev 40 + test 30
    assert len(m.CHITCHAT) == 45          # dev 30 + test 15
    assert m.LANGS == ("en", "ru", "zh")


def test_every_labelled_target_exists_and_noanswer_is_empty():
    m = _load()
    for qid, q in m.QUERIES.items():
        assert all(t in m.FACTS for t in q["targets"]), qid
        assert (q["slice"] == "noanswer") == (q["targets"] == []), qid


def test_translations_share_the_split():
    # one split for the whole id — question translations always live in the same split (the recipe)
    m = _load()
    for qid, q in m.QUERIES.items():
        assert set(q) >= {"split", "slice", "targets", "en", "ru", "zh"}, qid
    assert {q["split"] for q in m.QUERIES.values()} == {"dev", "test"}


def test_confused_labels_static_and_disjoint_from_targets():
    # the maintainer's fix list (PR #10, 10:05): the wrong-answer label is —
    # a static fact key, it exists and does not intersect the targets
    m = _load()
    for qid, q in m.QUERIES.items():
        conf = set(q.get("confused", []))
        assert conf <= set(m.FACTS), qid
        assert not conf & set(q["targets"]), qid
    labelled = [q for q in m.QUERIES.values() if q.get("confused")]
    assert len(labelled) == 14


def test_confusion_and_named_extra():
    # a query about Olya: a foreign fact in the top-5 => confusion 1; the correct answer about
    # Olya lands in named_extra, the foreign «Lena» does not (the name is not mentioned)
    m = _load()
    q = m.QUERIES["q002"]
    assert m.confusion(q, ["olga_job", "lena_job"]) == ["lena_job"]
    assert m.confusion(q, ["olga_job"]) == []
    assert m.named_extra(
        "en", q["en"], ["olga_job", "olya_coffee", "lena_job"], {"olga_job"}
    ) == ["olya_coffee"]
    # namesakes: Sergey Ivanovich against Sergey Petrovich
    q46 = m.QUERIES["q046"]
    assert m.confusion(q46, ["sergey_g_retired", "sergey_colleague"]) == ["sergey_colleague"]
    assert m.named_extra(
        "en", q46["en"], ["sergey_g_retired", "sergey_colleague"], set(q46["targets"])
    ) == []


def test_q023_names_the_neighbour():
    # the question names the neighbour: both of her facts are targets, the same-name coach is the label
    m = _load()
    q = m.QUERIES["q023"]
    assert q["targets"] == ["marina_flat", "walk_dog_neighbour"]
    assert q["confused"] == ["marina_pool"]
