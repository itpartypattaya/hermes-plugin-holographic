"""embeddings.RECALL_CHECK_MARGIN: a lower margin whose facts a check has to confirm (a measurement, issue #121).

With the value set and rerank off, a fact that stands out at the lower margin but not at RECALL_MARGIN is offered on
meaning only on the check's yes (recall_check.py); one that stands out at RECALL_MARGIN is offered as before, unasked.
Unset, the default, changes nothing. The meaning is given (no request), as in test_holographic_names_fit_cases; the
facts are made up and the check is a stand-in. The value and the provider's check are set with raising=False, so that
on main, where neither exists, a test fails on what it asserts rather than on the setup.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider, embeddings, rerank  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import mcnemar_gate as mcnemar  # noqa: E402
import provenance_probe as pp  # noqa: E402

ON = {"embeddings": "on", "embedding_url": "https://embeddings.example.net/v1", "embedding_model": "m",
      "embedding_key_env": "TEST_EMBEDDINGS_KEY"}

KEY = "The spare key hangs behind the blue flowerpot on the balcony."  # what the meaning stream may add
FILLERS = [f"Shelf {n} of the archive holds the folders of year {1990 + n}." for n in range(32)]
ASKS = "Whereabouts of the hidden gadget?"  # no word of a fact, no name: the words leave the turn silent


def _rc():
    """recall_check, imported where a test needs it: main has none, and the test fails on that line."""
    return importlib.import_module("hermes_plugin_holographic.recall_check")


class _NoNetwork:
    def open(self, *a, **k):
        raise AssertionError("no request in these tests: meaning and the verdict are given")


class _Check:
    """A check that gives one answer (or raises it) and remembers every question."""

    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def __call__(self, message, fact):
        self.asked.append((message, fact))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _provider(tmp_path, monkeypatch, **extra):
    monkeypatch.setattr(embeddings, "_OPENER", _NoNetwork())
    monkeypatch.setattr(rerank, "_OPENER", _NoNetwork())
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64, **ON, **extra})
    p.initialize(session_id="s1")
    return p


@pytest.fixture
def provider(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    yield p
    p.shutdown()


def _store(provider, monkeypatch, sims=None):
    """KEY among the fillers; at 0.30 it stands out more than 0.5 over the chance maximum and less than
    RECALL_MARGIN. *sims*: other similarities by text, for facts stored after KEY."""
    sims = {KEY: 0.30, **(sims or {})}
    ids = {}
    for text in (*sims, *FILLERS):
        result = json.loads(provider.handle_tool_call("fact_store", {"action": "add", "content": text}))
        assert "fact_id" in result, result
        ids[text] = result["fact_id"]
    order = list(ids)
    meaning = embeddings.Meaning([ids[t] for t in order],
                                 np.array([sims.get(t, 0.10 + 0.01 * (i % 12)) for i, t in enumerate(order)]), 0.3,
                                 [embeddings.text_hash(t) for t in order])
    for text in sims:  # every fact given a similarity here is between the two margins
        assert meaning.standout(ids[text], 0.5, None) and not meaning.standout(ids[text], 1.0, None)
    monkeypatch.setattr(provider, "_meaning_of", lambda q, before_turn: meaning)
    return ids[KEY]


def _checked(provider, monkeypatch, answer, margin=0.5):
    """The provider's check replaced by a stand-in that gives *answer*, and the value set unless *margin* is None."""
    check = _Check(answer)
    monkeypatch.setattr(provider, "_recall_check", check, raising=False)
    if margin is not None:
        monkeypatch.setattr(embeddings, "RECALL_CHECK_MARGIN", margin, raising=False)
    return check


def _by_meaning(block: str, fact_id: int) -> bool:
    return any(f"#{fact_id} " in line and line.endswith("(by meaning)") for line in block.splitlines())


# ---------------------------------------------------------------- recall

def test_unset_leaves_recall_as_it_is_and_makes_no_call(provider, monkeypatch):
    _store(provider, monkeypatch)
    check = _checked(provider, monkeypatch, ("yes", 5.0), margin=None)
    assert embeddings.RECALL_CHECK_MARGIN is None  # the default
    assert provider.prefetch(ASKS) == "" and check.asked == []


def test_an_eligible_turn_makes_exactly_one_call(provider, monkeypatch):
    """Two facts between the margins: one question, about the closest; after its no the other is not asked about."""
    second = "The spare bulb lies behind the green bucket in the cellar."
    _store(provider, monkeypatch, sims={KEY: 0.40, second: 0.39})  # two this close widen the scale: both between
    check = _checked(provider, monkeypatch, ("no", -1.0))
    assert provider.prefetch(ASKS) == "" and check.asked == [(ASKS, KEY)]


def test_a_yes_lets_the_fact_in(provider, monkeypatch):
    key = _store(provider, monkeypatch)
    check = _checked(provider, monkeypatch, ("yes", 2.0))
    assert _by_meaning(provider.prefetch(ASKS), key)
    assert check.asked == [(ASKS, KEY)]  # the message, and the fact as recall shows it


def test_a_no_keeps_the_fact_out(provider, monkeypatch):
    _store(provider, monkeypatch)
    check = _checked(provider, monkeypatch, ("no", -2.0))
    assert provider.prefetch(ASKS) == "" and check.asked == [(ASKS, KEY)]


@pytest.mark.parametrize("answer", [("not_answered", None), RuntimeError("endpoint down"), ("maybe", 0.9)])
def test_a_failed_call_keeps_the_baseline_result(provider, monkeypatch, answer):
    """Not answered, a check that raises, an answer that is neither yes nor no: the block of the value unset."""
    _store(provider, monkeypatch)
    words = "Which shelf of the archive holds the folders of year 1995?"  # a block of word matches
    baseline = provider.prefetch(words)
    assert "year 1995" in baseline
    provider.on_session_switch("s2", reset=True)  # the same turn again, nothing recent
    check = _checked(provider, monkeypatch, answer)
    assert provider.prefetch(words) == baseline and check.asked == [(words, KEY)]


def test_the_check_runs_with_rerank_off_only(tmp_path, monkeypatch):
    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    off = _provider(tmp_path / "off", monkeypatch)
    on = _provider(tmp_path / "on", monkeypatch, rerank="on", rerank_url="http://127.0.0.1:9/v1/rerank")
    try:
        for p in (off, on):
            _store(p, monkeypatch)
        monkeypatch.setattr(on._rescorer, "scores", lambda *a, **k: None)  # its order kept: no network here
        unset = on.prefetch(ASKS)
        on.on_session_switch("s2", reset=True)
        asked = {p: _checked(p, monkeypatch, ("yes", 2.0)).asked for p in (off, on)}
        off.prefetch(ASKS)
        assert len(asked[off]) == 1  # rerank off: the check is asked
        assert on.prefetch(ASKS) == unset and asked[on] == []  # rerank on: recall as with the value unset
    finally:
        off.shutdown()
        on.shutdown()


def test_a_run_can_swap_in_another_check(provider, monkeypatch):
    """recall_check.CHECK: one function (message, fact) -> (verdict, raw score), used instead of the cross-encoder;
    no rerank_url is needed for it."""
    key = _store(provider, monkeypatch)
    check = _Check(("yes", 0.97))
    monkeypatch.setattr(_rc(), "CHECK", check)
    monkeypatch.setattr(embeddings, "RECALL_CHECK_MARGIN", 0.5)
    assert _by_meaning(provider.prefetch(ASKS), key) and check.asked == [(ASKS, KEY)]


# ---------------------------------------------------------------- the guards of the check

SECRET = "OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl"


@pytest.mark.parametrize("message,content,shown,outcome", [
    (f"Where is {SECRET} kept?", KEY, KEY, "skipped_secret"),  # the message
    (ASKS, f"The router login: {SECRET}", f"The router login: {SECRET}", "skipped_secret"),  # the fact
    (ASKS, KEY + " " * 400 + SECRET, KEY, "skipped_secret"),  # the whole fact, not only the line recall shows
    ("where " * 90, KEY, KEY, "skipped_long"),  # a message over rerank.RERANK_MAX_QUERY_CHARS
    ("where " * 70, "spare " * 120, "spare " * 120, "skipped_long"),  # a pair over rerank.RERANK_PAIR_CHARS
])
def test_the_screens_send_nothing(message, content, shown, outcome):
    rc, check = _rc(), _Check(("yes", 5.0))
    assert rc.ask(check, message, content, shown) == (outcome, None) and check.asked == []


def test_the_check_fits_the_prefetch_budget_with_the_embedding_call(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch, embedding_timeout="6", rerank_url="http://127.0.0.1:9/v1/rerank",
                  rerank_timeout="10")
    try:
        monkeypatch.setattr(embeddings, "RECALL_CHECK_MARGIN", 0.5, raising=False)
        client = p._recall_checker().args[0]  # the default: the cross-encoder on the rerank_* keys, rerank off
        assert p._rescorer is None and client.settings.url == "http://127.0.0.1:9/v1/rerank"
        assert p._embedding.timeout + client.settings.timeout == rerank._PREFETCH_BUDGET_S  # 6 + 1.5
    finally:
        p.shutdown()


def test_a_slow_endpoint_costs_one_timeout_then_is_left_alone(tmp_path, monkeypatch):
    class _Slow:
        opened = 0

        def open(self, request, timeout=None):
            _Slow.opened += 1
            raise TimeoutError("timed out")  # what the socket says of an answer slower than rerank_timeout

    p = _provider(tmp_path, monkeypatch, rerank_url="http://127.0.0.1:9/v1/rerank")
    try:
        monkeypatch.setattr(rerank, "_OPENER", _Slow())
        monkeypatch.setattr(embeddings, "RECALL_CHECK_MARGIN", 0.5, raising=False)
        check = p._recall_checker()
        assert check(ASKS, KEY) == ("not_answered", None) and _Slow.opened == 1
        assert check(ASKS, KEY) == ("not_answered", None) and _Slow.opened == 1  # paused: no second request
    finally:
        p.shutdown()


def test_one_document_is_scored_only_when_asked_for(monkeypatch):
    client = rerank.Rescorer(rerank.Settings({"rerank_url": "http://127.0.0.1:9/v1/rerank"}))
    sent = []
    monkeypatch.setattr(client, "_request", lambda body: sent.append(body) or [(0, 1.5)])
    assert client.scores("question", ["document"]) == [(0, 1.0)] and client.last_status == "skipped_single"
    assert sent == []  # the reranker's own path: one document is not sent
    assert client.scores("question", ["document"], single=True) == [(0, 1.5)] and client.last_status == "answered"
    assert [body["documents"] for body in sent] == [["document"]]


# ---------------------------------------------------------------- the probe

def _stub_run(langs, modes, emb, endpoint):
    return [{"set": "multi", "lang": "en", "qid": "0", "split": "all", "kind": "real", "q": "query",
             "keys": ["f1"], "targets": ["t"], "hit": True, "hit5": True, "wrong": [], "wrong5": [],
             "sources": {"f1": ["words"]}, "miss_cls": "-", "mode": modes[0], "emb": emb, "out_n": 1}]


def _rows_in(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _probe(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(pp, "run_multilingual", _stub_run)
    monkeypatch.setattr(pp._holographic_embeddings, "RECALL_MARGIN", 1.0)
    return pp.main(["--set", "multi", "--modes", "off", "--emb", "on", "--out", str(tmp_path),
                    "--endpoint", "http://a/v1/rerank", *extra])


def test_the_probe_stamps_both_arms_and_names_their_files(monkeypatch, tmp_path):
    instruction = tmp_path / "instruction.txt"
    instruction.write_bytes(b"Decide whether the document answers the query.")
    checked = _probe(monkeypatch, tmp_path, "--recall-check-margin", "0.5", "--check-model", "Qwen3-Reranker-0.6B",
                     "--check-instruction", str(instruction))[0]
    assert (checked["margin"], checked["check_margin"], checked["check_model"]) == (1.0, 0.5, "Qwen3-Reranker-0.6B")
    assert checked["check_instruction_sha256"] == hashlib.sha256(instruction.read_bytes()).hexdigest()
    assert pp._holographic_embeddings.RECALL_CHECK_MARGIN is None and pp.CHECK_MODEL is None  # gone after the run
    off = _probe(monkeypatch, tmp_path, "--recall-check-margin", "off")[0]
    assert (off["margin"], off["check_margin"], off["check_model"], off["check_instruction_sha256"]) == \
        (1.0, "off", None, None)
    assert _probe(monkeypatch, tmp_path)[0]["check_margin"] == "off"  # a run without the flag ran without the check
    assert [r["check_margin"] for name in ("c0.5", "coff") for r in _rows_in(tmp_path / f"provenance_multi_{name}_on.jsonl")] \
        == [0.5, "off"]
    assert (tmp_path / "provenance_multi_on.jsonl").exists()


@pytest.mark.parametrize("extra,refusal", [
    (("--recall-check-margin", "0.5"), "name the check"),
    (("--recall-check-margin", "0.5", "--modes", "off,gate_only"), "rerank off"),
    (("--recall-check-margin", "1.0"), "not below the recall margin"),
])
def test_the_probe_refuses_a_checked_arm_that_cannot_check(monkeypatch, tmp_path, extra, refusal):
    instruction = tmp_path / "instruction.txt"
    instruction.write_bytes(b"Decide.")
    named = () if refusal == "name the check" else ("--check-model", "m", "--check-instruction", str(instruction))
    with pytest.raises(SystemExit, match=refusal):
        _probe(monkeypatch, tmp_path, *extra, *named)


def test_the_probe_records_each_turn_of_the_check(provider, monkeypatch):
    _store(provider, monkeypatch)
    _checked(provider, monkeypatch, ("no", -1.5))
    probe = pp.Probe(provider)
    block, ctx = probe.prefetch(ASKS)
    assert block == ""  # a turn that shows nothing still says what the check did
    record = ctx["check"]
    assert (record["check_calls"], record["check_outcome"], record["check_score"]) == (1, "no", -1.5)
    assert isinstance(record["check_ms"], float)


def test_the_probe_stops_on_a_check_that_did_not_answer(provider, monkeypatch):
    _store(provider, monkeypatch)
    _checked(provider, monkeypatch, ("not_answered", None))
    with pytest.raises(SystemExit, match="not answered"):
        pp.Probe(provider).prefetch(ASKS)


# ---------------------------------------------------------------- the gate

SHA = hashlib.sha256(b"Decide whether the document answers the query.").hexdigest()


def _mrow(margin, qid, keys, hit5, **over):
    row = {"set": "multi", "lang": "en", "qid": qid, "margin": margin, "kind": "real", "q": "query",
           "split": "all", "mode": "off", "emb": "on", "scale": "logit", "git_sha": "958de48",
           "endpoint": "http://127.0.0.1:1237/v1/rerank", "targets": ["t"], "keys": list(keys),
           "hit5": hit5, "wrong": [], "wrong5": [], "out_n": len(keys)}
    row.update(over)
    return row


def _files(tmp_path, *arms):
    paths = []
    for i, rows in enumerate(arms):
        path = tmp_path / f"arm{i}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        paths.append(str(path))
    return paths


def _arms(**checked):
    """Six questions the off arm misses and the checked arm finds, both at the ordinary margin 1.0."""
    off = [_mrow(1.0, f"r{i}", ["f1", "f2"], False, check_margin="off") for i in range(6)]
    on = [_mrow(1.0, f"r{i}", ["f1", "f2", "t"], True, check_margin=0.5, check_model="Qwen3-Reranker-0.6B",
                check_instruction_sha256=SHA, check_calls=1, check_outcome="yes", check_score=2.0, check_ms=41.0)
          for i in range(6)]
    for r in on:
        r.update(checked)
    return off, on


def _gate(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["mcnemar_gate.py", *argv])
    mcnemar.main()


def test_the_gate_pairs_the_off_arm_with_the_checked_one(tmp_path, monkeypatch, capsys):
    _gate(monkeypatch, "--pair", "check_margin", *_files(tmp_path, *_arms()))
    out = capsys.readouterr().out
    assert (f"== check_margin off (default) vs 0.5 (proposal) | margin 1.0 | checker Qwen3-Reranker-0.6B, "
            f"instruction sha256 {SHA} |") in out
    assert "check: 6 calls in 6 turns, {'yes': 6}, slowest 41.0 ms" in out and "=> gate PASSES" in out


def test_the_plain_pairs_refuse_runs_that_differ_in_check_margin(tmp_path, monkeypatch):
    off, on = _arms()
    for r in on:
        r["margin"] = 0.5  # the plain margin pair, one of its runs checked
    with pytest.raises(SystemExit, match="check_margin"):
        _gate(monkeypatch, "--pair", "margin", *_files(tmp_path, off, on))
    for rows, text in ((off, "content"), (on, "content+keywords")):
        for r in rows:
            r.update(margin=1.0, embed_text=text)
    with pytest.raises(SystemExit, match="check_margin"):
        _gate(monkeypatch, "--pair", "embed_text", *_files(tmp_path, off, on))


@pytest.mark.parametrize("checked,refusal", [
    ({"check_model": None}, "one checker"),
    ({"check_instruction_sha256": None}, "one checker"),
    ({"git_sha": "958de48-dirty"}, "uncommitted"),
])
def test_the_gate_refuses_a_checked_arm_it_cannot_name(tmp_path, monkeypatch, checked, refusal):
    off, on = _arms(**checked)
    if "git_sha" in checked:  # one head: both arms from the same tree
        for r in off:
            r["git_sha"] = checked["git_sha"]
    with pytest.raises(SystemExit, match=refusal):
        _gate(monkeypatch, "--pair", "check_margin", *_files(tmp_path, off, on))
