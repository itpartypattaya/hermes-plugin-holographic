"""0.15.0: who stands behind a fact (said / confirmed / inferred) and confirmations by the person."""
from __future__ import annotations

import json
import sqlite3

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider


def _provider(tmp_path, session="s1", **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": 64, **config})
    p.initialize(session_id=session)
    return p


@pytest.fixture
def p(tmp_path):
    provider = _provider(tmp_path)
    yield provider
    provider.shutdown()


def _call(p, tool="fact_store", **args):
    return json.loads(p.handle_tool_call(tool, args))


def _add(p, content, origin=None):
    return _call(p, action="add", content=content, **({"origin": origin} if origin else {}))


def _trust(p, fact_id):
    return p._store.read("SELECT trust_score FROM facts WHERE fact_id = ?", (fact_id,))[0][0]


def test_the_origin_sets_the_starting_trust_and_is_kept_with_the_version(p):
    said = _add(p, "Anton prefers green tea", "said")["fact_id"]
    mine = _add(p, "Anton probably drinks tea every morning", "inferred")["fact_id"]
    plain = _add(p, "The office is on the third floor")["fact_id"]
    assert (_trust(p, said), _trust(p, mine), _trust(p, plain)) == pytest.approx((0.6, 0.4, 0.5))
    history = _call(p, action="history", fact_id=said)
    assert history["versions"][0]["source"]["origin"] == "said"
    assert p._store.origins([said, mine, plain]) == {said: {"origin": "said", "confirmations": 0},
                                                      mine: {"origin": "inferred", "confirmations": 0},
                                                      plain: {"origin": None, "confirmations": 0}}


def test_an_unknown_origin_is_refused(p):
    assert "origin must be" in _call(p, action="add", content="Zelda is a cat", origin="guessed")["error"]


def test_said_again_in_a_later_session_is_a_confirmation_once_per_session(tmp_path):
    first = _provider(tmp_path, "s1")
    fact = _add(first, "Irma's school starts at 8", "said")["fact_id"]
    again = _add(first, "Irma's school starts at 8", "said")  # the session that stored it: the agent repeating itself
    assert again["status"] == "exists" and "confirmations" not in again and _trust(first, fact) == pytest.approx(0.6)
    first.shutdown()
    second = _provider(tmp_path, "s2")
    assert _add(second, "Irma's school starts at 8", "said")["confirmations"] == 1
    assert _add(second, "Irma's school starts at 8", "said")["confirmations"] == 1  # once per session
    assert _trust(second, fact) == pytest.approx(0.65)
    assert "confirmations" not in _add(second, "Irma's school starts at 8", "inferred")  # re-inferring is not confirming
    second.shutdown()


def test_without_a_session_a_confirmation_counts_once_a_day(p):
    fact = _add(p, "The gym opens at 6", "said")["fact_id"]
    for _ in range(5):
        p._store.record_confirmation(fact, {"via": "feedback"})  # a script with no session id
    assert p._store.origins([fact])[fact]["confirmations"] == 1 and _trust(p, fact) == pytest.approx(0.65)


def test_one_person_raises_trust_once_however_many_sessions(p):
    fact = _add(p, "The meeting room is on floor 2", "inferred")["fact_id"]
    for session in ("a", "b", "c"):
        p._store.record_confirmation(fact, {"session_id": session, "user_id": "42", "via": "feedback"})
    assert p._store.origins([fact])[fact]["confirmations"] == 3 and _trust(p, fact) == pytest.approx(0.45)
    p._store.record_confirmation(fact, {"session_id": "d", "user_id": "43", "via": "feedback"})  # another person
    assert _trust(p, fact) == pytest.approx(0.5)


def test_change_starts_over_with_the_new_origin(p):
    fact = _add(p, "Alice lives in Paris", "said")["fact_id"]
    _call(p, "fact_feedback", action="confirmed", fact_id=fact)
    _call(p, action="change", fact_id=fact, content="Alice probably moved to Berlin", origin="inferred")
    assert p._store.origins([fact])[fact] == {"origin": "inferred", "confirmations": 0}  # they confirmed Paris
    assert _trust(p, fact) == pytest.approx(0.4)
    p._injected.clear()
    assert "(inferred)" in p.prefetch("where does Alice live, Berlin?", session_id="q9")


def test_correct_keeps_the_origin_it_does_not_name(p):
    fact = _add(p, "Zelda was born in March 2026", "said")["fact_id"]
    _call(p, action="correct", fact_id=fact, content="Zelda was born in April 2026")
    assert p._store.origins([fact])[fact]["origin"] == "said"
    _call(p, action="update", fact_id=fact, content="Zelda was born on April 2, 2026")  # update content= is correct
    assert p._store.origins([fact])[fact]["origin"] == "said"


def test_review_counts_the_age_of_the_inference_not_of_the_fact(p):
    fact = _add(p, "Boris works at a bank", "said")["fact_id"]
    p._store._write("UPDATE facts SET created_at = datetime('now', '-90 days'), retrieval_count = 1 WHERE fact_id = ?",
                    (fact,))
    _call(p, action="change", fact_id=fact, content="Boris probably works at a startup now", origin="inferred")
    reasons = {c["fact_id"]: c["reason"] for c in _call(p, action="review")["candidates"]}
    assert "inference" not in reasons.get(fact, "")  # it became an inference today


def test_a_writer_opened_later_drops_confirmations_an_older_forget_left(tmp_path, p):
    fact = _add(p, "The car is parked in the garage", "said")["fact_id"]
    _call(p, "fact_feedback", action="confirmed", fact_id=fact)
    p._store._write("DELETE FROM facts WHERE fact_id = ?", (fact,))  # what 0.14.x forget does, rows of this table aside
    p._store._write("DELETE FROM fact_versions WHERE fact_id = ?", (fact,))
    from hermes_plugin_holographic.store import MemoryStore
    again = MemoryStore(db_path=str(tmp_path / "m.db"), hrr_dim=64)  # the same process: no migration, a writer open
    try:
        assert again.read("SELECT COUNT(*) FROM fact_confirmations")[0][0] == 0
    finally:
        again.close()


def test_the_starting_trust_stays_within_bounds(tmp_path):
    p = _provider(tmp_path, default_trust=0.95)
    try:
        assert _trust(p, _add(p, "Anton lives in Jomtien", "said")["fact_id"]) == pytest.approx(1.0)
    finally:
        p.shutdown()


def test_a_confirmed_inference_reads_as_confirmed(p):
    fact = _add(p, "The family flies to Hua Hin in spring", "inferred")["fact_id"]
    assert "(inferred)" in p.prefetch("when does the family fly to Hua Hin", session_id="q1")
    got = _call(p, "fact_feedback", action="confirmed", fact_id=fact)
    assert got["confirmed"] and got["origin"] == "confirmed" and got["trust_score"] == pytest.approx(0.45)
    assert _call(p, "fact_feedback", action="confirmed", fact_id=fact)["confirmed"] is False  # same session
    p._injected.clear()
    assert "(inferred)" not in p.prefetch("when does the family fly to Hua Hin", session_id="q2")
    found = _call(p, action="search", query="Hua Hin flight")["results"]
    assert found[0]["origin"] == "confirmed" and found[0]["confirmations"] == 1


def test_forget_takes_the_confirmations_with_it(p):
    fact = _add(p, "Anton's bike is a Honda Forza", "said")["fact_id"]
    _call(p, "fact_feedback", action="confirmed", fact_id=fact)
    _call(p, action="forget", fact_id=fact)
    assert p._store.read("SELECT COUNT(*) FROM fact_confirmations")[0][0] == 0


def test_review_lists_an_old_unconfirmed_inference(p):
    old = _add(p, "Ksenia likes jazz concerts", "inferred")["fact_id"]
    fresh = _add(p, "Ksenia likes the sea", "inferred")["fact_id"]
    said = _add(p, "Ksenia speaks English at work", "said")["fact_id"]
    p._store._write("UPDATE facts SET created_at = datetime('now', '-90 days'), retrieval_count = 1 WHERE fact_id IN (?, ?)",
                    (old, said))
    p._store._write("UPDATE fact_versions SET recorded_at = datetime('now', '-90 days') WHERE fact_id IN (?, ?)",
                    (old, said))
    reasons = {c["fact_id"]: c["reason"] for c in _call(p, action="review")["candidates"]}
    assert "inference nobody confirmed" in reasons.get(old, "") and fresh not in reasons and said not in reasons


def test_the_tool_tells_the_agent_what_leaving_origin_out_does(p):
    # issue #3: 10 of 24 new facts on a live store had no origin; the description named the three values but not what
    # happens without one, so nothing nudged the agent to pass it
    schema = next(s for s in p.get_tool_schemas() if s["name"] == "fact_store")
    text = schema["parameters"]["properties"]["origin"]["description"]
    assert "Pass it whenever you know" in text
    assert "'unknown' in stats" in text and "default trust" in text
    assert "keeps the version's" in text


def test_leaving_origin_out_does_what_the_tool_says(p):
    plain = _add(p, "The office is on the third floor")["fact_id"]
    assert _trust(p, plain) == pytest.approx(0.5)
    assert _call(p, action="stats")["origin"]["unknown"] == 1
    p.on_session_switch("s2", reset=True)
    again = _add(p, "The office is on the third floor")
    assert again["fact_id"] == plain and "confirmations" not in again
    said = _add(p, "The office is on the third floor", "said")
    assert said["confirmations"] == 1
    # the same text without an origin is not a new fact: the stored one keeps its origin and trust
    tea = _add(p, "Anton prefers green tea", "said")["fact_id"]
    assert _add(p, "Anton prefers green tea")["fact_id"] == tea
    assert p._store.origins([tea])[tea]["origin"] == "said" and _trust(p, tea) == pytest.approx(0.6)


def test_stats_count_the_origins(p):
    _add(p, "Zelda is a British cat", "said")
    _add(p, "Zelda likes the sofa", "inferred")
    _add(p, "Zelda was born in 2026")
    assert _call(p, action="stats")["origin"] == {"said": 1, "confirmed": 0, "inferred": 1, "unknown": 1,
                                                  "confirmations": 0}


def test_a_reader_opens_a_file_an_older_writer_made(tmp_path):
    """0.14.x files have no src_origin and no fact_confirmations: a read-only context (no migration) still reads
    them, and the next writer adds both."""
    writer = _provider(tmp_path)
    fact = _add(writer, "The router sits in the hallway", "said")["fact_id"]
    writer.shutdown()
    con = sqlite3.connect(tmp_path / "m.db")
    con.execute("DROP TABLE fact_confirmations")
    con.execute("ALTER TABLE fact_versions DROP COLUMN src_origin")
    con.commit()
    con.close()
    reader = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": 64})
    reader.initialize(session_id="r", agent_context="subagent")
    try:
        assert f"#{fact} " in reader.prefetch("where is the router", session_id="q")
        assert _call(reader, action="search", query="router")["results"][0]["fact_id"] == fact
        assert _call(reader, action="history", fact_id=fact)["versions"][0]["source"] == "unknown" \
            or "origin" not in _call(reader, action="history", fact_id=fact)["versions"][0]["source"]
        assert _call(reader, action="stats")["origin"]["unknown"] == 1
    finally:
        reader.shutdown()
    writer = _provider(tmp_path, "s2")
    try:
        assert _call(writer, "fact_feedback", action="confirmed", fact_id=fact)["confirmations"] == 1
        assert _add(writer, "The printer sits in the office", "inferred")["status"] == "added"
    finally:
        writer.shutdown()


def test_auto_extracted_sentences_are_said(tmp_path):
    p = _provider(tmp_path, auto_extract="true")
    try:
        p.sync_turn("I prefer oat milk in my coffee", "Noted.", session_id="s1")
        p.on_session_end([{"role": "user", "content": "I prefer oat milk in my coffee"}])
        ids = [r[0] for r in p._store.read("SELECT fact_id FROM facts")]
        assert ids and all(o["origin"] == "said" for o in p._store.origins(ids).values())
    finally:
        p.shutdown()
