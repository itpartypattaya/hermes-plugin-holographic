"""Memory graph: the co_occurs write path, bounded one-hop expansion, interleaving, and repair.

Regression guards for a measured failure: a single generic query token resolved to a hub entity, the unbounded
one-hop expansion admitted 110 weakly related facts, and a flat relevance bonus pushed them into ranks 2-8 above
genuine text hits.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import retrieval as retrieval_mod
from hermes_plugin_holographic.graph import GRAPH_MAX_ENTITY_DEGREE, GRAPH_NEIGHBOUR_CAP
from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore

DIM = 512


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)


def _names(store, fact_id):
    return {r["name"].lower() for r in store.read(
        "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id WHERE fe.fact_id = ?",
        (fact_id,))}


def test_a_fact_linking_two_entities_creates_their_edge(tmp_path):
    """The write path lives in the schema: a fact naming X and Y makes X-Y a graph neighbour."""
    store = _store(tmp_path)
    store.add_fact_status("portlandship pairs with rivernode", category="general",
                          entities=["portlandship", "rivernode"])
    left, right = store.entity_ids("portlandship"), store.entity_ids("rivernode")
    assert left and right
    assert store.entity_neighbors(left) == set(right)
    assert store.entity_degrees([*left, *right]) == {left[0]: 1, right[0]: 1}


def test_edges_survive_an_entity_merge(tmp_path):
    """Merging two entities repoints their edges instead of orphaning them."""
    store = _store(tmp_path)
    store.add_fact_status("mergeship links firstnode", category="general",
                          entities=["mergeship", "firstnode"])
    store.add_fact_status("Mergeship and secondnode", category="general", entities=["secondnode"])
    store.add_alias("mergeship", "mergeship2")  # no edge of its own: resolving the alias must still reach the node
    hubs = store.entity_ids("mergeship")
    assert hubs
    assert store.entity_ids("mergeship2") == hubs
    neighbours = store.entity_neighbors(hubs)
    assert len(neighbours) >= 1


def test_deleted_entity_leaves_no_edges(tmp_path):
    """delete_entity drops the edges it owned, so no relation points at a name that is gone."""
    store = _store(tmp_path)
    store.add_fact_status("goner links staynode", category="general", entities=["goner", "staynode"])
    store.delete_entity("goner")
    assert store.read("SELECT COUNT(*) c FROM entity_relations")[0]["c"] == 0


def test_rebuild_graph_recreates_edges_from_fact_entities(tmp_path):
    """The graph is derived data: rebuilding from fact_entities restores exactly the implied edges."""
    store = _store(tmp_path)
    store.add_fact_status("rebuildship links alphaleaf betaleaf", category="general",
                          entities=["rebuildship", "alphaleaf", "betaleaf"])
    with store._tx():
        store._write("DELETE FROM entity_relations")
    assert store.rebuild_graph() == 3  # the three pairs the one fact implies
    assert store.rebuild_graph() == 0  # idempotent


def test_prune_graph_drops_edges_no_fact_implies(tmp_path):
    """The graph is derived data: an edge whose fact stopped linking the pair is pruned, a live one is kept."""
    store = _store(tmp_path)
    store.add_fact_status("pruneship links keptleaf", category="general", entities=["pruneship", "keptleaf"])
    store.add_fact_status("pruneship links dropleaf", category="general", entities=["pruneship", "dropleaf"])
    with store._tx():  # simulate a fact whose links a later write removed (the trigger never deletes edges)
        store._write("DELETE FROM fact_entities WHERE entity_id = (SELECT entity_id FROM entities WHERE name = 'dropleaf')")
    assert store.prune_graph() >= 1
    left = store.entity_ids("pruneship")
    assert store.entity_ids("keptleaf")[0] in store.entity_neighbors(left)
    assert store.entity_ids("dropleaf")[0] not in store.entity_neighbors(left)


def test_expansion_is_capped_by_neighbour_and_fact_limits(tmp_path):
    """A hub with many neighbours must not flood the candidate pool."""
    store = _store(tmp_path)
    leaves = [f"graphnbr{i:02d}" for i in range(40)]
    store.add_fact_status("hubnode bridges " + " ".join(leaves), category="general",
                          entities=["hubnode", *leaves])
    for i in range(len(leaves)):
        store.add_fact_status(f"filler{i:02d} unrelated payload wording", category="general",
                              entities=[leaves[i]])

    hub = store.entity_ids("hubnode")
    neighbours = store.entity_neighbors(hub, limit=GRAPH_NEIGHBOUR_CAP, max_degree=GRAPH_MAX_ENTITY_DEGREE)
    assert len(neighbours) == GRAPH_NEIGHBOUR_CAP  # the fan-out itself is capped, not just the facts it yields

    expanded = FactRetriever(store, hrr_dim=DIM)._expand_by_graph("hubnode", {}, None, 0.3)
    assert expanded, "the fixture must actually expand, or the caps prove nothing"
    assert len(expanded) <= retrieval_mod._GRAPH_FACT_CAP


def test_hub_entities_are_not_traversed(tmp_path):
    """Hub/namespace entities are neither expanded from nor walked into."""
    store = _store(tmp_path)
    leaves = [f"leaf{i:02d}" for i in range(45)]
    store.add_fact_status("hubspace topics " + " ".join(leaves), entities=["hubspace", *leaves])
    for i in range(len(leaves)):
        store.add_fact_status(f"leaf fact {i:02d}", category="general", entities=[leaves[i]])
    store.add_fact_status("smallspace links tinynode", entities=["smallspace", "tinynode"])

    hub = store.entity_ids("hubspace")
    assert store.entity_degrees(hub)[hub[0]] > GRAPH_MAX_ENTITY_DEGREE
    # A leaf never walks back into the hub.
    assert store.entity_neighbors(store.entity_ids("leaf00"), limit=25,
                                 max_degree=GRAPH_MAX_ENTITY_DEGREE) == set()
    # Ordinary entities still traverse.
    assert store.entity_neighbors(store.entity_ids("smallspace"), limit=25,
                                  max_degree=GRAPH_MAX_ENTITY_DEGREE)
    # A query that resolves only to a hub does not expand at all.
    assert FactRetriever(store, hrr_dim=DIM)._expand_by_graph("hubspace", {}, None, 0.3) == set()


def test_graph_discoveries_are_interleaved_not_promoted(tmp_path):
    """A related discovery surfaces below the direct hit; a zero-evidence discovery is never shown."""
    store = _store(tmp_path)
    store.add_fact_status("hubnode bridging satellite device", category="general",
                          entities=["hubnode", "satellite"])
    strong, _ = store.add_fact_status("hubnode review gate blocks unverified commits", category="general")
    related, _ = store.add_fact_status("review gate notes about weather", category="general",
                                       entities=["satellite"])
    junk, _ = store.add_fact_status("completely different topic filler", category="general",
                                    entities=["satellite"])

    ranked = [f["fact_id"] for f in FactRetriever(store, hrr_dim=DIM).search("hubnode review gate", limit=10)]
    assert related in ranked                       # recall: reachable through the graph
    assert ranked.index(strong) < ranked.index(related)  # precision: below the direct text hit
    assert junk not in ranked                      # a co-occurrence edge alone is not evidence


def test_wiki_pointer_fact_links_namespace_and_page_topic(tmp_path):
    """A wiki pointer line carries no capitalised name: the path supplies namespace and topic."""
    store = _store(tmp_path)
    fact, _ = store.add_fact_status(
        "[WIKI HINT - consult when relevant] Wiki page: ai-agents/entities/llama-cpp.md | type: entity",
        category="wiki")
    assert _names(store, fact) == {"ai agents", "llama cpp"}
    # The pair the trigger derives from that link is one edge.
    assert store.read("SELECT COUNT(*) c FROM entity_relations")[0]["c"] == 1


def test_wiki_structural_slugs_are_not_entities(tmp_path):
    """index/schema/log pages are structure, not topics: only the namespace is linked."""
    store = _store(tmp_path)
    fact, _ = store.add_fact_status(
        "[WIKI HINT - consult when relevant] Wiki page: operations/SCHEMA.md | type: schema", category="wiki")
    assert _names(store, fact) == {"operations"}
