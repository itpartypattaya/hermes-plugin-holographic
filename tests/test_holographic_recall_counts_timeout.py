"""The recall-count flush leaves the connection's lock wait as it found it (code review 2026-11-17).

add_retrieval_counts writes with busy timeout 0 so a flush never waits on another process's write lock, then puts
the timeout back. It put back a fixed 10000 ms; the scrub after forget (history.py) and the experience prune
(experience.py) restore what they read. The store's connection is shared by every provider of the process, so a
fixed value overwrote whatever wait the connection had when the flush ran.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider


def _provider(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "counts.db"), "hrr_dim": 1024})
    provider.initialize(session_id="counts")
    return provider


def _timeout(store) -> int:
    return store._conn.execute("PRAGMA busy_timeout").fetchone()[0]


def test_flush_keeps_the_connections_own_timeout(tmp_path):
    provider = _provider(tmp_path)
    store = provider._store
    fact_id = store.add_fact("Яна любит жасминовый чай.")
    store._conn.execute("PRAGMA busy_timeout = 3000")
    assert store.add_retrieval_counts({fact_id: 2}) is True
    assert store.read("SELECT retrieval_count FROM facts WHERE fact_id = ?", (fact_id,))[0][0] == 2
    assert _timeout(store) == 3000
    provider.shutdown()


def test_busy_flush_returns_at_once_and_keeps_the_timeout(tmp_path):
    provider = _provider(tmp_path)
    store = provider._store
    fact_id = store.add_fact("Яна любит жасминовый чай.")
    store._conn.execute("PRAGMA busy_timeout = 3000")
    other = sqlite3.connect(str(tmp_path / "counts.db"), isolation_level=None)
    other.execute("BEGIN IMMEDIATE")  # another process holds the write lock
    try:
        assert store.add_retrieval_counts({fact_id: 1}) is False  # deferred, not waited 3 s for
    finally:
        other.execute("ROLLBACK")
        other.close()
    assert _timeout(store) == 3000
    provider.shutdown()
