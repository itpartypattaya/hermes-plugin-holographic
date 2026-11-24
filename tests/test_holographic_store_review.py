"""Store / provider fixes from the review of 0.6.2–0.9.0: a failed COMMIT no longer wedges the connection, a forget
does not wait on a foreign reader, credentials are never stored, short CJK sentences are extracted, a missing space
after a full stop still splits sentences, user_id_alt names the user store, revert explains itself after a revert, and
a file written by a newer release is not opened for writing."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider, _RE_SENTENCE
from hermes_plugin_holographic.history import HistoryTooNew
from hermes_plugin_holographic.store import MemoryStore

DIM = 64


def _provider(tmp_path, config=None, **init):
    cfg = {"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": DIM, **(config or {})}
    p = HolographicMemoryProvider(config=cfg)
    p.initialize(session_id=init.pop("session_id", "s1"), **init)
    return p


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _contents(provider):
    return [f["content"] for f in provider._store.list_facts(limit=100)]


def _journal(store, mode: str) -> None:
    """Pin the journal mode for the test: the core picks WAL or DELETE by the SQLite it runs on."""
    assert store._conn.execute(f"PRAGMA journal_mode={mode}").fetchone()[0].lower() == mode.lower()


def _reader(db) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Another process in the middle of a long query: a statement left open holds its read lock."""
    other = sqlite3.connect(str(db), isolation_level=None)
    cur = other.execute("SELECT fact_id, content FROM facts")
    cur.fetchone()
    return other, cur


# --- a failed COMMIT ------------------------------------------------------------------------------------------------


def test_a_failed_commit_is_rolled_back_and_the_next_write_works(tmp_path):
    """Rollback-journal mode: a foreign SHARED lock makes COMMIT fail with "database is locked". The transaction
    must not stay open, or every later write would join it and nothing would ever be committed again."""
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    _journal(store, "DELETE")
    for i in range(5):  # several rows: Python's cursor steps one row ahead, and a finished statement holds no lock
        store.add_fact(f"Alice Smith lives in Paris, flat {i}")
    other, cur = _reader(db)
    store._conn.execute("PRAGMA busy_timeout = 200")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store.add_fact("Bob Brown lives in Rome")
        assert not store._conn.in_transaction  # rolled back, not dangling
    finally:
        cur.close()
        other.close()
    fid = store.add_fact("Bob Brown lives in Rome")
    assert fid > 0 and store.count_facts() == 6
    assert sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 6  # committed for real
    store.close()


# --- forget and a foreign reader ------------------------------------------------------------------------------------


def test_forget_does_not_wait_for_a_foreign_reader(tmp_path):
    """WAL: the reader blocks only the checkpoint. The forget must report `pending` at once (every provider
    operation queues behind it), not after the connection's 10 s busy timeout, and leave the timeout as it was."""
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    _journal(store, "WAL")
    for i in range(50):
        store.add_fact(f"Person{i} Surname{i} lives in city number {i}")
    timeout = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    other, cur = _reader(db)
    try:
        started = time.monotonic()
        result = store.forget_fact(50)
        elapsed = time.monotonic() - started
    finally:
        cur.close()
        other.close()
    assert result["forgotten"] is True and result["scrub"] == "pending" and elapsed < 3.0
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == timeout
    assert store.read("SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH 'Surname49'")[0][0] == 0
    again = store.forget_fact(50)
    assert again["already"] is True and again["scrub"] == "done"
    assert store.read("SELECT COUNT(*) FROM meta WHERE key = 'scrub_pending'")[0][0] == 0
    store.close()


# --- credentials ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sentence", [
    "I use password hunter2 for the home wifi.",
    "The wifi password is Qwerty!23 at home.",
    "My passphrase for the vault is correct-horse-battery",
    "Пароль от wifi: Qwerty123, не забудь.",
    "My API key is sk-live-9f8e7d6c5b4a for prod.",
    "The door pin code is 4821 for guests.",
])
def test_a_credential_with_its_value_is_never_stored(tmp_path, sentence):
    provider = _provider(tmp_path, config={"auto_extract": "true"})
    provider.on_session_end([{"role": "user", "content": sentence}])
    assert _contents(provider) == []
    result = _call(provider, action="add", content=sentence)
    assert "credential" in result["error"]
    provider.shutdown()


@pytest.mark.parametrize("sentence", [
    "I prefer tea.",
    "I use a password manager for everything.",
    "I always forget my password for the bank.",
    "We decided the token expires after thirty days.",
    "I prefer a PIN over a fingerprint.",
])
def test_the_word_alone_is_not_a_credential(tmp_path, sentence):
    provider = _provider(tmp_path, config={"auto_extract": "true"})
    provider.on_session_end([{"role": "user", "content": sentence}])
    assert _contents(provider) == [sentence]
    provider.shutdown()


# --- sentences ------------------------------------------------------------------------------------------------------


def test_short_cjk_sentences_are_extracted(tmp_path):
    provider = _provider(tmp_path, config={"auto_extract": "true"})
    provider.on_session_end([{"role": "user", "content": "我们决定用Go。"}, {"role": "user", "content": "我偏好簡潔的解法。"},
                             {"role": "user", "content": "I use Go."}])  # nine Latin characters: too short still
    assert _contents(provider) == ["我们决定用Go。", "我偏好簡潔的解法。"]
    provider.shutdown()


def test_a_missing_space_after_a_full_stop_still_splits(tmp_path):
    assert [m.group(0) for m in _RE_SENTENCE.finditer("I prefer tea.I use vim daily.")] == ["I prefer tea.", "I use vim daily."]
    assert [m.group(0) for m in _RE_SENTENCE.finditer("We pin Python 3.12, keep notes in topic.md and run on qx-vps.local.")] \
        == ["We pin Python 3.12, keep notes in topic.md and run on qx-vps.local."]
    provider = _provider(tmp_path, config={"auto_extract": "true"})
    provider.on_session_end([{"role": "user", "content": "I prefer tea.I use vim daily."}])
    assert sorted(_contents(provider)) == ["I prefer tea.", "I use vim daily."]
    provider.shutdown()


# --- user_id_alt, revert, a newer schema ----------------------------------------------------------------------------


def test_user_id_alt_names_the_user_store(tmp_path):
    provider = _provider(tmp_path, config={"scope": "user"}, platform="telegram", user_id_alt="999")
    assert _call(provider, action="add", content="Alice Smith lives in Paris")["status"] == "added"
    provider.shutdown()
    assert (tmp_path / "holographic-users" / "telegram-999.db").exists()


def test_revert_after_a_revert_explains_itself(tmp_path):
    provider = _provider(tmp_path)
    fid = _call(provider, action="add", content="Frank Hill prefers green tea")["fact_id"]
    _call(provider, action="change", fact_id=fid, content="Frank Hill prefers black coffee")
    assert _call(provider, action="revert", fact_id=fid)["status"] == "reverted"
    error = _call(provider, action="revert", fact_id=fid)["error"]
    assert "not undone by another revert" in error and "single version" not in error
    lone = _call(provider, action="add", content="Gina Ray lives in Oslo")["fact_id"]
    assert "single version" in _call(provider, action="revert", fact_id=lone)["error"]
    provider.shutdown()


def test_a_file_from_a_newer_release_is_not_opened_for_writing(tmp_path):
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    store.add_fact("Alice Smith lives in Paris")
    store._conn.execute("UPDATE meta SET value = '99' WHERE key = 'history_schema'")
    store.close()
    with pytest.raises(HistoryTooNew, match="update the plugin"):
        MemoryStore(str(db), hrr_dim=DIM)
    assert sqlite3.connect(str(db)).execute("SELECT value FROM meta WHERE key = 'history_schema'").fetchone()[0] == "99"
    reader = MemoryStore(str(db), hrr_dim=DIM, migrate=False)
    assert reader.count_facts() == 1
    reader.close()
    provider = _provider(tmp_path)
    assert "update the plugin" in provider.system_prompt_block() and ".recover" not in provider.system_prompt_block()
    provider.shutdown()
