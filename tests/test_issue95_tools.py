"""Issue #95 tooling: gate G1 in tools/mcnemar_gate.py (--pair embed_text) and the probe's arms.

Gate G1 was fixed before any run (2026-11-16): two arms that differ only in what fact vectors are made from, on one
head, one embedding model and one keywords file and coverage, every store fully covered. The gain passes exact McNemar
on rescued (b) against lost (c) top-five hits, p < 0.05 with b > c; every lost target is listed by name; noise and
confusion count as rows added minus rows removed, each at most 0. Anything that would make the two arms more than one
change refuses the files instead of reporting.

The probe side: --embed-text and the keywords never outlive their run and stamp every row and the file name; half
coverage is the same half for the same seed; each run's stores live in a folder of their own; frozen keywords are
stored as written; a store left short of vectors stops the run (it used to report "ready" after 60 tries)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("mcnemar_gate_95", _REPO / "tools" / "mcnemar_gate.py")
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import provenance_probe as pp  # noqa: E402

HEAD = {"mode": "off", "emb": "on", "scale": "logit", "git_sha": "abc1234", "endpoint": "mock", "margin": 1.0,
        "emb_model": "http://127.0.0.1:1241/v1/bge-m3", "emb_dim": 1024, "kw_file": "0123456789ab",
        "kw_coverage": "half:seed=95", "vectors": 199, "vectors_expected": 199, "split": "test",
        "names_fit": "on", "kw_facts": 99}


def _real(arm, qid, keys, *, targets=("t",), confused=False, lang="en", set_="household", **over):
    keys = list(keys)
    row = {**HEAD, "set": set_, "lang": lang, "qid": qid, "embed_text": arm, "kind": "real", "q": f"q {qid}",
           "targets": list(targets), "keys": keys, "hit5": bool(set(targets) & set(keys[:5])),
           "hit": bool(set(targets) & set(keys)), "confused": ["x"] if confused else [], "out_n": len(keys),
           "sources": {}}
    row.update(over)
    return row


def _chit(arm, qid, keys=(), **over):
    row = {**HEAD, "set": "household", "lang": "en", "qid": qid, "embed_text": arm, "kind": "chit",
           "q": f"chit {qid}", "targets": [], "keys": list(keys), "hit5": None, "hit": None, "confused": [],
           "out_n": len(keys), "sources": {}}
    row.update(over)
    return row


def _six_rescued():
    """Six questions the keywords arm rescues (b=6, c=0, p=0.03125) and two chit rows with no output on either side."""
    base, prop = [], []
    for i in range(6):
        base.append(_real("content", f"r{i}", ["f1", "f2", "f3", "f4", "f5"]))
        prop.append(_real("content+keywords", f"r{i}", ["t", "f1", "f2", "f3", "f4"]))
    for i in range(2):
        base.append(_chit("content", f"c{i}"))
        prop.append(_chit("content+keywords", f"c{i}"))
    return base, prop


def _files(base, prop, tmp_path):
    a, b = tmp_path / "etcontent_on.jsonl", tmp_path / "etcontent-keywords_on.jsonl"
    a.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in base), encoding="utf-8")
    b.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in prop), encoding="utf-8")
    return [str(a), str(b)]


# --- gate G1 ---


def test_six_rescues_and_no_loss_pass(tmp_path, capsys):
    """Control: the fixture can pass, so a failure below comes from the one thing each test changes."""
    base, prop = _six_rescued()
    assert gate.report_embed_text(_files(base, prop, tmp_path)) is True
    out = capsys.readouterr().out
    assert "b=6 rescued, c=0 lost, exact McNemar p=0.0312" in out
    assert "gate G1 PASSES" in out


def test_a_lost_target_is_named_and_does_not_veto_a_real_gain(tmp_path, capsys):
    """G1, unlike the #17 gate, lets a loss stand when the gain outweighs it — but names it: set qid:target, lang."""
    base, prop = _six_rescued()
    for i in range(3):  # three more rescues keep p < 0.05 with one loss (b=9, c=1, p=0.021)
        base.append(_real("content", f"s{i}", ["f1", "f2", "f3", "f4", "f5"]))
        prop.append(_real("content+keywords", f"s{i}", ["t", "f1", "f2", "f3", "f4"]))
    base.append(_real("content", "q7", ["dentist", "f1"], targets=["dentist"], lang="ru"))
    prop.append(_real("content+keywords", "q7", ["f1", "f2", "f3", "f4", "f5", "dentist"], targets=["dentist"],
                      lang="ru"))
    assert gate.report_embed_text(_files(base, prop, tmp_path)) is True
    out = capsys.readouterr().out
    assert "lost targets in the top five (1): household q7:dentist, ru" in out
    # the target is still in the output, below the fifth line: the whole-output count shows that separately
    assert "target anywhere in the output: 1/10 -> 10/10; b=9, c=0" in out
    assert "hit@5: 1/10 -> 9/10; b=9 rescued, c=1 lost" in out


def test_a_significant_loss_does_not_pass_as_a_gain(tmp_path, capsys):
    """p < 0.05 alone is two-sided: six LOST and none rescued is as significant as the reverse."""
    base, prop = _six_rescued()
    for r in base + prop:  # swap the arms' answers, keep the labels
        if r["kind"] == "real":
            r["keys"] = ["f1", "f2", "f3", "f4", "f5"] if r["embed_text"] == "content+keywords" else ["t", "f1"]
            r["hit5"] = r["hit"] = "t" in r["keys"]
    assert gate.report_embed_text(_files(base, prop, tmp_path)) is False
    out = capsys.readouterr().out
    assert "FAIL  gain: McNemar p < 0.05 with b > c (b=0, c=6, p=0.031)" in out


def test_noise_counts_rows_added_minus_rows_removed(tmp_path, capsys):
    """One chit row gains output and another loses it: net 0 passes. A third that gains output makes it +1: fails."""
    base, prop = _six_rescued()
    base[-1]["keys"], base[-1]["out_n"] = ["f9"], 1   # c1 had noise in the content arm...
    prop[-2]["keys"], prop[-2]["out_n"] = ["f8"], 1   # ...c0 gets noise in the keywords arm
    assert gate.report_embed_text(_files(base, prop, tmp_path)) is True
    assert "PASS  noise: added 1 - removed 1 <= 0" in capsys.readouterr().out

    base.append(_chit("content", "c2"))
    prop.append(_chit("content+keywords", "c2", keys=["f7"]))
    assert gate.report_embed_text(_files(base, prop, tmp_path)) is False
    assert "FAIL  noise: added 2 - removed 1 <= 0" in capsys.readouterr().out


def test_a_no_answer_question_with_output_is_noise(tmp_path, capsys):
    base, prop = _six_rescued()
    base.append(_real("content", "n0", [], targets=()))
    prop.append(_real("content+keywords", "n0", ["f1"], targets=()))
    assert gate.report_embed_text(_files(base, prop, tmp_path)) is False
    assert "FAIL  noise: added 1 - removed 0 <= 0" in capsys.readouterr().out


def test_rising_confusion_fails(tmp_path, capsys):
    base, prop = _six_rescued()
    prop[0]["confused"] = ["x"]
    assert gate.report_embed_text(_files(base, prop, tmp_path)) is False
    assert "FAIL  confusion: added 1 - removed 0 <= 0" in capsys.readouterr().out


@pytest.mark.parametrize("field, other", [("emb_model", "https://api.openai.com/v1/text-embedding-3-small"),
                                          ("emb_dim", 1536), ("kw_coverage", "all"), ("kw_file", "ffffffffffff"),
                                          ("margin", 0.6), ("git_sha", "def5678"), ("names_fit", "off")])
def test_a_second_change_beside_the_vector_text_is_refused(tmp_path, field, other):
    """Two models, two vector sizes, two keyword coverages or files, two margins or two heads are not one change."""
    base, prop = _six_rescued()
    for r in prop:
        r[field] = other
    with pytest.raises(SystemExit, match=field):
        gate.report_embed_text(_files(base, prop, tmp_path))


@pytest.mark.parametrize("field, other", [("split", "dev"), ("kw_facts", 199)])
def test_a_pair_whose_stores_differ_is_refused(tmp_path, field, other):
    """Inside a pair: another split, or stores holding a different number of facts with keywords."""
    base, prop = _six_rescued()
    prop[0][field] = other
    with pytest.raises(SystemExit, match="unpaired"):
        gate.report_embed_text(_files(base, prop, tmp_path))


def test_rows_of_an_uncommitted_tree_are_refused(tmp_path):
    base, prop = _six_rescued()
    for r in base + prop:
        r["git_sha"] = "abc1234-dirty"
    with pytest.raises(SystemExit, match="uncommitted"):
        gate.report_embed_text(_files(base, prop, tmp_path))


def test_dev_rows_are_refused(tmp_path):
    base, prop = _six_rescued()
    for r in base + prop:
        r["split"] = "dev"
    with pytest.raises(SystemExit, match="dev rows"):
        gate.report_embed_text(_files(base, prop, tmp_path))


def test_rows_without_the_model_stamp_are_refused(tmp_path):
    base, prop = _six_rescued()
    for r in base + prop:
        r.pop("emb_model")
    with pytest.raises(SystemExit, match="emb_model missing"):
        gate.report_embed_text(_files(base, prop, tmp_path))


def test_a_store_short_of_vectors_is_refused(tmp_path):
    base, prop = _six_rescued()
    prop[0]["vectors"] = 198
    with pytest.raises(SystemExit, match="incomplete vector coverage"):
        gate.report_embed_text(_files(base, prop, tmp_path))


def test_rows_without_embeddings_are_refused(tmp_path):
    base, prop = _six_rescued()
    for r in base + prop:
        r["emb"] = "off"
    with pytest.raises(SystemExit, match="embeddings on"):
        gate.report_embed_text(_files(base, prop, tmp_path))


def test_a_question_missing_from_one_arm_is_refused(tmp_path):
    base, prop = _six_rescued()
    prop.pop(0)
    with pytest.raises(SystemExit, match="unpaired"):
        gate.report_embed_text(_files(base, prop, tmp_path))


# --- the probe's arms ---


def _stub_run(set_name, modes, emb, endpoint, extra=None):
    """One contrast-shaped row; records what the run itself saw: its store folder, vector text and keywords."""
    _stub_run.db_dirs.append(pp.DB_DIR)
    _stub_run.seen.append((pp._holographic_embeddings.EMBED_TEXT, pp.KEYWORDS))
    return [{"set": set_name, "lang": set_name, "qid": "p0q0", "split": "all", "kind": "real", "q": "query",
             "keys": ["t"], "targets": ["t"], "hit": True, "hit5": True, "wrong": False, "sources": {"1": ["words"]},
             "miss_cls": "-", "mode": modes[0], "emb": emb, "out_n": 1}]


def _main(monkeypatch, tmp_path, *extra):
    _stub_run.db_dirs, _stub_run.seen = [], []
    monkeypatch.setattr(pp, "run_contrast", _stub_run)
    return pp.main(["--set", "contrast_en", "--modes", "off", "--out", str(tmp_path), *extra])


def test_the_vector_text_override_does_not_outlive_its_run(monkeypatch, tmp_path):
    monkeypatch.setattr(pp._holographic_embeddings, "EMBED_TEXT", "content")
    rows = _main(monkeypatch, tmp_path, "--embed-text", "content+keywords")
    assert rows[0]["embed_text"] == "content+keywords"
    assert _stub_run.seen[0][0] == "content+keywords"  # in force while the run ran, not only in the stamp
    assert pp._holographic_embeddings.EMBED_TEXT == "content"
    again = _main(monkeypatch, tmp_path, "--embed-text", "content")
    assert again[0]["embed_text"] == "content"
    names = sorted(p.name for p in tmp_path.glob("*.jsonl"))
    assert names == ["provenance_contrast_en_etcontent-keywords_off.jsonl", "provenance_contrast_en_etcontent_off.jsonl"]


def test_keywords_are_stamped_and_put_back(monkeypatch, tmp_path):
    kw = tmp_path / "kw.json"
    kw.write_text(json.dumps({"contrast_en": {"a": ["x", "y", "z"], "b": ["u", "v", "w"]}}), encoding="utf-8")
    rows = _main(monkeypatch, tmp_path, "--keywords", str(kw), "--kw-coverage", "half", "--kw-seed", "7")
    assert rows[0]["kw_coverage"] == "half:seed=7" and len(rows[0]["kw_file"]) == 12
    assert _stub_run.seen[0][1] == {"contrast_en": pp._load_keywords(str(kw), "half", 7)["contrast_en"]}
    assert len(_stub_run.seen[0][1]["contrast_en"]) == 1
    assert pp.KEYWORDS is None
    assert (tmp_path / "provenance_contrast_en_kwhalf7_off.jsonl").exists()
    plain = _main(monkeypatch, tmp_path)
    assert plain[0]["kw_coverage"] == "none" and plain[0]["kw_file"] is None


def test_each_run_has_stores_of_its_own_and_leaves_none_behind(monkeypatch, tmp_path):
    first = _main(monkeypatch, tmp_path, "--embed-text", "content") and _stub_run.db_dirs[0]
    second = _main(monkeypatch, tmp_path, "--embed-text", "content+keywords") and _stub_run.db_dirs[0]
    assert first and second and first != second
    assert pp.DB_DIR is None
    assert not first.exists() and not second.exists()


def test_half_coverage_is_the_same_half_for_the_same_seed(tmp_path):
    kw = tmp_path / "kw.json"
    by_key = {f"k{i:02d}": ["a", "b", "c"] for i in range(40)}
    kw.write_text(json.dumps({"en": by_key, "ru": by_key}), encoding="utf-8")
    one = pp._load_keywords(str(kw), "half", 95)
    assert one == pp._load_keywords(str(kw), "half", 95)
    assert len(one["en"]) == 20 and len(one["ru"]) == 20
    assert set(one["en"]) != set(pp._load_keywords(str(kw), "half", 96)["en"])
    assert pp._load_keywords(str(kw), "all", 95) == {"en": by_key, "ru": by_key}


@pytest.fixture
def provider(tmp_path):
    p = pp.HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": 64})
    p.initialize(session_id="kw")
    yield p
    p.shutdown()


def test_frozen_keywords_are_stored_as_written(provider, monkeypatch):
    from hermes_plugin_holographic import _split_tags
    monkeypatch.setattr(pp, "KEYWORDS", {"en": {"dacha": ["country house", "summer"]}})
    back = pp.store_facts(provider, "en", {"dacha": ("Grandma spends July at the dacha.", []),
                                           "car": ("The car is on level two.", [])})
    by_key = {key: fid for fid, key in back.items()}
    assert _split_tags(provider._store.fact_tags(by_key["dacha"]))[1] == ["country house", "summer"]
    assert _split_tags(provider._store.fact_tags(by_key["car"]))[1] == []


def test_keywords_the_plugin_would_cut_stop_the_run(provider, monkeypatch):
    monkeypatch.setattr(pp, "KEYWORDS", {"en": {"dacha": ["x" * 41, "summer", "july"]}})
    with pytest.raises(SystemExit, match="not stored as written"):
        pp.store_facts(provider, "en", {"dacha": ("Grandma spends July at the dacha.", [])})


def test_a_keywords_file_of_another_set_stops_the_run(provider, monkeypatch):
    """No entry for the run's language: both arms would run without keywords and look like a measured nothing."""
    monkeypatch.setattr(pp, "KEYWORDS", {"contrast_en": {"dacha": ["a", "b", "c"]}})
    with pytest.raises(SystemExit, match="not this set's file"):
        pp.store_facts(provider, "en", {"dacha": ("Grandma spends July at the dacha.", [])})


def test_the_store_counts_its_facts_with_keywords(provider, monkeypatch):
    monkeypatch.setattr(pp, "KEYWORDS", {"en": {"dacha": ["country house", "summer", "july"]}})
    pp.store_facts(provider, "en", {"dacha": ("Grandma spends July at the dacha.", []),
                                    "car": ("The car is on level two.", [])})
    provider._store.add_fact("Tagged by the person", tags="family\nkeywords: kin, relatives, clan")
    assert pp._kw_facts(provider) == 2


def test_keywords_for_a_fact_the_set_lacks_stop_the_run(provider, monkeypatch):
    monkeypatch.setattr(pp, "KEYWORDS", {"en": {"gone": ["a", "b", "c"]}})
    with pytest.raises(SystemExit, match="does not have"):
        pp.store_facts(provider, "en", {"dacha": ("Grandma spends July at the dacha.", [])})


class _Down:
    """An embeddings endpoint that never answers."""

    def open(self, request, timeout=None):
        raise OSError("connection refused")


class _Up:
    def open(self, request, timeout=None):
        body = json.loads(request.data)

        class _R:
            def __init__(self, raw):
                self.raw = raw

            def read(self, n=-1):
                piece, self.raw = (self.raw, b"") if n is None or n < 0 else (self.raw[:n], self.raw[n:])
                return piece

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        data = [{"object": "embedding", "index": i, "embedding": [1.0, 0.0, 0.0, float(i)]}
                for i, _ in enumerate(body["input"])]
        return _R(json.dumps({"data": data, "model": body["model"]}).encode())


def _emb_provider(tmp_path, monkeypatch, opener):
    monkeypatch.setattr(pp._holographic_embeddings, "_OPENER", opener)
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    p = pp.HolographicMemoryProvider(config={
        "db_path": str(tmp_path / "e.db"), "hrr_dim": 64, "embeddings": "on",
        "embedding_url": "https://embeddings.example.net/v1", "embedding_model": "fake-1",
        "embedding_key_env": "TEST_EMBEDDINGS_KEY"})
    p.initialize(session_id="kw")
    for i in range(3):
        p._store.add_fact(f"Fact number {i} about the garden shed")
    return p


def test_a_store_left_short_of_vectors_stops_the_run(tmp_path, monkeypatch):
    p = _emb_provider(tmp_path, monkeypatch, _Down())
    try:
        with pytest.raises(SystemExit, match="incomplete vector coverage: 0 of 3"):
            pp.catch_up_vectors(p)
    finally:
        p.shutdown()


def test_a_covered_store_reports_its_model_and_counts(tmp_path, monkeypatch):
    p = _emb_provider(tmp_path, monkeypatch, _Up())
    try:
        got = pp.catch_up_vectors(p)
    finally:
        p.shutdown()
    assert got == {"emb_model": p._embedding.tag, "emb_dim": 4, "vectors": 3, "vectors_expected": 3,
                   "never_sent": 0}
