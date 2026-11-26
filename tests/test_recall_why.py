"""0.32.0: why a fact was recalled — in the task-outcome journal (experience_trace on) and in fact_store search with
explain: true, never in the recall block.

The requirement: when someone asks "why did the agent recall this?" (or why not), the journal of that turn answers by
which path each emitted fact came in — the words of the message (how many), the meaning, a name the message gave, the
memory graph — and, when the reranker scored it, its rank. The journal keeps no text, so the reason is a kind and a
number; the words themselves appear only in an explicit search with explain, whose answer already shows the facts.
"""
import json

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, embeddings, rerank
from test_holographic_v0140_embeddings import BACKUP, BIRTHDAY, ON, PARAPHRASE, FakeEndpoint, _fill
from test_holographic_rerank_single import RANK, FakeRerank


@pytest.fixture
def make(tmp_path):
    made = []

    def _make(**config):
        p = HolographicMemoryProvider({"db_path": str(tmp_path / f"facts{len(made)}.db"), "hrr_dim": 64,
                                       "experience": "on", "experience_trace": "on", **config})
        p.initialize(session_id="s", platform="cli")
        made.append(p)
        return p

    yield _make
    for p in made:
        p.shutdown()


def store(p, **args):
    return json.loads(p.handle_tool_call("fact_store", args))


def emitted(p):
    """The last attempt's emitted facts: {fact_id: why}."""
    attempt = json.loads(p.handle_tool_call("fact_experience", {"action": "recent"}))["attempts"][0]
    report = json.loads(p.handle_tool_call("fact_experience", {"action": "explain",
                                                                "attempt_id": attempt["attempt_id"]}))
    return {e["fact_id"]: e.get("why") for e in report["emitted"]}


def test_a_fact_found_by_words_says_how_many_of_the_message_words_it_holds(make):
    p = make()
    fid = store(p, action="add", content="The observatory budget is 600 euros", origin="said")["fact_id"]
    block = p.prefetch("observatory budget", session_id="s")
    assert f"#{fid} " in block
    assert emitted(p) == {fid: "words:2"}
    assert "words:" not in block and "why" not in block  # never in what the model reads


def test_a_fact_brought_by_a_name_the_message_gives_says_entity(make):
    p = make()
    by_words = store(p, action="add", content="Lena plays the violin", origin="said", entities=["Lena"])["fact_id"]
    by_name = store(p, action="add", content="Lena moved to Porto in May", origin="said",
                    entities=["Lena"])["fact_id"]
    p.prefetch("Does Lena still play the violin?", session_id="s")
    why = emitted(p)
    assert why[by_words].startswith("words:")
    assert why[by_name] == "entity"


def test_a_fact_found_by_meaning_says_meaning(make, monkeypatch):
    fake = FakeEndpoint()
    monkeypatch.setattr(embeddings, "_OPENER", fake)
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    p = make(**ON)
    _, (backup, _) = _fill(p, BACKUP, BIRTHDAY)
    p.sync_turn("hello", "hi")
    block = p.prefetch(PARAPHRASE, session_id="s")
    assert f"#{backup} " in block
    assert emitted(p)[backup] == "meaning"


def test_a_fact_the_reranker_scored_carries_its_rank(make, monkeypatch):
    fake = FakeRerank()
    monkeypatch.setattr(rerank, "_OPENER", fake)
    p = make(**RANK, rerank_weight="0.0", rerank_window="0.0")
    first = store(p, action="add", content="The nightly backup runs at 2 am", origin="said")["fact_id"]
    second = store(p, action="add", content="The nightly backup copies the mail server", origin="said")["fact_id"]
    p.prefetch("when does the nightly backup run", session_id="s")
    assert fake.requests, "the reranker was asked"
    why = emitted(p)
    assert set(why) == {first, second}
    assert sorted(w.split()[-1] for w in why.values()) == ["rerank:1", "rerank:2"]
    assert all(w.split()[0].startswith("words:") for w in why.values())


def test_without_the_trace_the_journal_keeps_ids_only(make):
    p = make(experience_trace="off")
    fid = store(p, action="add", content="The observatory budget is 600 euros", origin="said")["fact_id"]
    p.prefetch("observatory budget", session_id="s")
    assert emitted(p) == {fid: None}
    traced = make()
    other = store(traced, action="add", content="The observatory budget is 600 euros", origin="said")["fact_id"]
    traced.prefetch("observatory budget", session_id="s")
    assert emitted(traced) == {other: "words:2"}


def test_search_with_explain_names_the_words_and_without_it_says_nothing(make):
    p = make()
    fid = store(p, action="add", content="The observatory budget is 600 euros", origin="said")["fact_id"]
    plain = store(p, action="search", query="observatory budget")
    assert plain["results"][0]["fact_id"] == fid
    assert all(not k.startswith("_") and k != "why" for r in plain["results"] for k in r)
    explained = store(p, action="search", query="observatory budget", explain=True)
    assert explained["results"][0]["why"] == {"kind": "words", "words": ["budget", "observatory"]}


def test_a_fact_the_graph_reaches_is_marked_graph(tmp_path):
    """The fixture of test_a_multi_word_entity_name_seeds_expansion: "textnode keeps the parser small" shares no word
    with "Realm Forge" and is reached only through the Realm Forge–textnode edge; every fact the expansion adds says
    so, the candidate the words found does not."""
    pytest.importorskip("numpy")
    from hermes_plugin_holographic.retrieval import FactRetriever
    from hermes_plugin_holographic.store import MemoryStore

    store = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=512)
    try:
        seed, _ = store.add_fact_status("Realm Forge uses a text backend", category="general",
                                        entities=["Realm Forge", "textnode"])
        reached, _ = store.add_fact_status("textnode keeps the parser small", category="general")
        retriever = FactRetriever(store, hrr_dim=512)
        candidates = {seed: {"fact_id": seed, "content": "Realm Forge uses a text backend"}}
        expanded = retriever._expand_by_graph("Realm Forge", candidates, None, 0.3)
        assert reached in expanded
        assert all(candidates[fid]["_why"] == "graph" for fid in expanded)
        assert "_why" not in candidates[seed]
    finally:
        store.close()


def test_explain_reads_the_graph_mark_and_strips_it_from_the_answer(make):
    """The provider's side of the graph mark: what search found through the graph says kind graph under explain,
    and the internal mark never reaches the answer."""
    p = make()
    fid = store(p, action="add", content="The observatory budget is 600 euros", origin="said")["fact_id"]
    real_search = p._retriever.search

    def search_through_graph(*args, **kwargs):  # the retriever as the graph test above shows it behaving
        out = real_search(*args, **kwargs)
        for f in out:
            f["_why"] = "graph"
        return out

    p._retriever.search = search_through_graph
    explained = store(p, action="search", query="observatory budget", explain=True)["results"]
    assert explained[0]["fact_id"] == fid and explained[0]["why"] == {"kind": "graph"}
    plain = store(p, action="search", query="observatory budget")["results"]
    assert all(not k.startswith("_") and k != "why" for r in plain for k in r)
