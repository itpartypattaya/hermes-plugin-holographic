"""Migration from the 0.6.1 line, which retired a fact in place (facts.retired_at) instead of replacing the row.

This line treats `facts` as the current set and never reads that column, so a store written by the old line answered
with the retired text next to the fact that superseded it until the migration dropped it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore

DIM = 64


def test_a_retired_row_is_dropped_on_open(tmp_path):
    path = str(tmp_path / "legacy.db")
    store = MemoryStore(path, hrr_dim=DIM)
    keeper, _ = store.add_fact_status("keeper fact about the weather", category="general")
    replaced, _ = store.add_fact_status("an older claim about the weather", category="general")
    with store._tx():
        store._write("ALTER TABLE facts ADD COLUMN retired_at TEXT")
        store._write("ALTER TABLE facts ADD COLUMN retired_by INTEGER")
        store._write("UPDATE facts SET retired_at = '2026-11-12 03:41:39', retired_by = ? WHERE fact_id = ?",
                     (keeper, replaced))
    store.close()

    reopened = MemoryStore(path, hrr_dim=DIM)
    try:
        ids = {r["fact_id"] for r in reopened.read("SELECT fact_id FROM facts")}
        assert keeper in ids
        assert replaced not in ids, "a row the old line retired came back as a current fact"
        assert not reopened.read("SELECT fact_id FROM fact_entities WHERE fact_id = ?", (replaced,)), \
            "the dropped fact kept its entity links"
        hits = [f["fact_id"] for f in FactRetriever(reopened, hrr_dim=DIM).search("older claim about the weather",
                                                                                  limit=5)]
        assert replaced not in hits, "the dropped fact still comes back from search"
    finally:
        reopened.close()


def test_a_retired_column_with_no_retired_rows_drops_nothing(tmp_path):
    """The column existing is not a reason to delete: only rows actually retired go."""
    path = str(tmp_path / "mixed.db")
    store = MemoryStore(path, hrr_dim=DIM)
    fact, _ = store.add_fact_status("an absolutely current fact", category="general")
    with store._tx():
        store._write("ALTER TABLE facts ADD COLUMN retired_at TEXT")
    store.close()

    reopened = MemoryStore(path, hrr_dim=DIM)
    try:
        assert fact in {r["fact_id"] for r in reopened.read("SELECT fact_id FROM facts")}
    finally:
        reopened.close()


def test_a_store_this_line_wrote_is_untouched(tmp_path):
    """No retired_at column: the migration must not invent one or delete anything."""
    path = str(tmp_path / "fresh.db")
    store = MemoryStore(path, hrr_dim=DIM)
    fact, _ = store.add_fact_status("a current fact", category="general")
    store.close()
    reopened = MemoryStore(path, hrr_dim=DIM)
    try:
        assert fact in {r["fact_id"] for r in reopened.read("SELECT fact_id FROM facts")}
        assert "retired_at" not in {r[1] for r in reopened._conn.execute("PRAGMA table_info(facts)")}
    finally:
        reopened.close()
