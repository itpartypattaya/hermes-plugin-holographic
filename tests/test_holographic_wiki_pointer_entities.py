"""Wiki pointer facts: the namespace and page slug of "Wiki page: <path>.md" are linked as entities.

Bulk wiki ingest writes each page through the normal add_fact path, but its pointer line carries no capitalised
phrase the entity extractor can see. Without these links a re-ingested page is reachable only by the words of its
body: it is never a topic of its own, and an entity view of the wiki shows nothing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import store as store_module
from hermes_plugin_holographic.store import MemoryStore

DIM = 512

POINTER = "[WIKI HINT - consult when relevant] Wiki page: {path} | type: entity"


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)


def _names(store, fact_id):
    return {r["name"].lower() for r in store.read(
        "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id WHERE fe.fact_id = ?",
        (fact_id,))}


def test_pointer_fact_links_namespace_and_page_topic(tmp_path):
    """The path is the only proper noun on a pointer line: namespace and page slug come from it."""
    store = _store(tmp_path)
    fact, _ = store.add_fact_status(POINTER.format(path="ai-agents/entities/llama-cpp.md"), category="wiki")
    assert _names(store, fact) == {"ai agents", "llama cpp"}


def test_nested_path_takes_namespace_and_last_segment(tmp_path):
    """A section directory is structure: the page is the last segment, not the folder in the middle."""
    store = _store(tmp_path)
    fact, _ = store.add_fact_status(POINTER.format(path="gaming/openmw/tamriel-rebuilt.md"), category="wiki")
    assert _names(store, fact) == {"gaming", "tamriel rebuilt"}


def test_structural_slugs_are_not_entities(tmp_path):
    """index/schema/log pages are the wiki's shape, not topics: only the namespace is linked."""
    store = _store(tmp_path)
    fact, _ = store.add_fact_status(POINTER.format(path="operations/SCHEMA.md"), category="schema")
    assert _names(store, fact) == {"operations"}


def test_single_segment_path_names_no_page(tmp_path):
    """A path with no namespace is not a pointer to a wiki page: nothing is taken from it."""
    store = _store(tmp_path)
    fact, _ = store.add_fact_status(POINTER.format(path="llama-cpp.md"), category="wiki")
    assert _names(store, fact) == set()


def test_a_deleted_name_is_not_recreated_from_a_path(tmp_path):
    """A path is text: text never brings back a name delete_entity removed (see _tombstoned)."""
    store = _store(tmp_path)
    store.add_fact_status(POINTER.format(path="ai-agents/entities/llama-cpp.md"), category="wiki")
    store.delete_entity("llama cpp")
    fact, _ = store.add_fact_status(
        "[WIKI HINT] Wiki page: ai-agents/entities/llama-cpp.md | type: entity; second ingest", category="wiki")
    assert "llama cpp" not in _names(store, fact)
    assert "ai agents" in _names(store, fact)


def test_rebuild_under_new_entity_rules_links_existing_pointer_facts(tmp_path):
    """A store indexed before these rules gets the links on its first writer open, not on a re-ingest."""
    path = str(tmp_path / "legacy.db")
    store = MemoryStore(path, hrr_dim=DIM)
    fact, _ = store.add_fact_status(POINTER.format(path="ai-agents/entities/llama-cpp.md"), category="wiki")
    # The index an older release wrote: pointer links absent, rules recorded as 4.
    with store._tx():
        store._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact,))
        store._write("UPDATE meta SET value = '4' WHERE key = 'entity_rules'")
    store.close()

    reopened = MemoryStore(path, hrr_dim=DIM)
    assert reopened.entity_rules() == store_module._ENTITY_RULES == 5
    assert _names(reopened, fact) == {"ai agents", "llama cpp"}
    reopened.close()
