"""A refresh of facts the session already holds records no new attempt.

With the presence gate off (the default), ``prefetch_repeat_turns`` is a safety net: a fact this session
injected comes back once the window expires, in case the context was trimmed without a compaction event. If
the earlier block is still in context, the core (NousResearch/hermes-agent#148069) drops the repeated bullets,
and a fresh ``Memory attempt:`` line, the only line the window does not carry, kept a heading-only block alive
every few turns and recorded an attempt for facts the model had already seen. The earlier block, with its own
attempt line, is what the model still reads, so a refresh-only block carries no attempt line.
"""
import json
import sqlite3

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider

FACT = "Alice Smith works on the observatory project"
OTHER = "Bob Jones runs the observatory night shift"
QUERY = "Alice observatory"


@pytest.fixture
def make(tmp_path):
    made = []

    def _make(**config):
        p = HolographicMemoryProvider({"db_path": str(tmp_path / f"facts{len(made)}.db"), "experience": "on",
                                       **config})
        p.initialize(session_id="s")
        made.append(p)
        return p

    yield _make
    for p in made:
        p.shutdown()


def add(p, content):
    return json.loads(p.handle_tool_call("fact_store", {"action": "add", "content": content, "origin": "said"}))


def attempts(p):
    return json.loads(p.handle_tool_call("fact_experience", {"action": "recent"}))["attempts"]


def expire_window(p):
    """Advance this session past ``prefetch_repeat_turns`` without injecting anything in between."""
    for key in p._turns:
        p._turns[key] += p._repeat_turns + 1


def test_a_refresh_after_the_window_expires_carries_no_attempt_line(make):
    p = make()
    add(p, FACT)
    first = p.prefetch(QUERY, session_id="s")
    assert FACT in first and "Memory attempt:" in first
    recorded = len(attempts(p))

    expire_window(p)
    refresh = p.prefetch(QUERY, session_id="s")

    assert FACT in refresh  # the safety net still re-delivers the fact
    assert "Memory attempt:" not in refresh
    assert len(attempts(p)) == recorded  # and no attempt names facts the model already holds


def test_a_refresh_after_a_failed_journal_write_records_its_attempt(make, monkeypatch):
    """The first injection's attempt was never written (the journal was locked): nothing names the fact yet, so
    its next delivery records the attempt instead of counting as a refresh."""
    p = make()
    add(p, FACT)
    written = p._store.record_attempt

    def locked(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(p._store, "record_attempt", locked)
    first = p.prefetch(QUERY, session_id="s")
    assert FACT in first and "Memory attempt:" not in first
    monkeypatch.setattr(p._store, "record_attempt", written)
    expire_window(p)

    again = p.prefetch(QUERY, session_id="s")

    assert FACT in again and "Memory attempt:" in again
    assert len(attempts(p)) == 1


def test_a_block_with_a_new_fact_records_its_attempt(make):
    p = make()
    add(p, FACT)
    p.prefetch(QUERY, session_id="s")
    expire_window(p)
    add(p, OTHER)

    block = p.prefetch("Alice Bob observatory", session_id="s")

    assert OTHER in block and FACT in block
    assert "Memory attempt:" in block


def test_after_a_compaction_the_re_delivery_records_its_attempt(make):
    """Gate off, a compaction clears what the session injected: the fact comes back as a new delivery."""
    p = make()
    add(p, FACT)
    p.prefetch(QUERY, session_id="s")
    p.on_pre_compress([{"role": "user", "content": QUERY}])

    block = p.prefetch(QUERY, session_id="s")

    assert FACT in block and "Memory attempt:" in block


def test_with_the_gate_on_and_no_view_a_refresh_carries_no_attempt_line(make):
    """Gate on before any view of the context: the window decides alone, as with the gate off."""
    p = make(prefetch_presence_gate="true")
    add(p, FACT)
    p.prefetch(QUERY, session_id="s")
    expire_window(p)

    refresh = p.prefetch(QUERY, session_id="s")

    assert FACT in refresh and "Memory attempt:" not in refresh


def test_with_the_gate_on_a_compaction_since_the_injection_records_its_attempt(make):
    """Gate on, a compaction keeps the record (no view yet to say what left): the fact may be gone, so its return
    is a real re-delivery."""
    p = make(prefetch_presence_gate="true")
    add(p, FACT)
    p.prefetch(QUERY, session_id="s")
    p.on_pre_compress([{"role": "user", "content": QUERY}])
    expire_window(p)

    block = p.prefetch(QUERY, session_id="s")

    assert FACT in block and "Memory attempt:" in block


def test_with_the_gate_on_a_fact_shown_gone_records_its_attempt(make):
    """Gate on, a view of the context that no longer shows the fact proves it left: a real re-delivery."""
    p = make(prefetch_presence_gate="true")
    add(p, FACT)
    p.prefetch(QUERY, session_id="s")
    expire_window(p)
    p.sync_turn("u", "a", session_id="s", messages=[{"role": "user", "content": "unrelated"}])

    block = p.prefetch(QUERY, session_id="s")

    assert FACT in block and "Memory attempt:" in block
