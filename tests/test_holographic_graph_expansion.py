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


def test_edges_follow_an_entity_merge(tmp_path):
    """Merging two entities repoints their edges onto the survivor instead of orphaning them."""
    store = _store(tmp_path)
    store.add_fact_status("mergeship links firstnode", category="general", entities=["mergeship", "firstnode"])
    store.add_fact_status("secondnode stands alone", category="general", entities=["secondnode"])
    source = store.entity_ids("mergeship")[0]
    target = store.entity_ids("secondnode")[0]
    firstnode = store.entity_ids("firstnode")[0]
    assert firstnode in store.entity_neighbors([source])
    with store._tx():
        store._merge_into(source, target)
    assert store.read("SELECT COUNT(*) c FROM entity_relations WHERE src_entity_id = ? OR dst_entity_id = ?",
                      (source, source))[0]["c"] == 0, "an edge still points at the merged-away entity"
    assert firstnode in store.entity_neighbors([target]), "the merged entity's edges did not move to the survivor"


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


def test_a_multi_word_entity_name_seeds_expansion(tmp_path):
    """Seeding must use the names the query mentions: word-by-word lookup finds nothing for "Realm Forge"."""
    store = _store(tmp_path)
    store.add_fact_status("Realm Forge uses a text backend", category="general",
                          entities=["Realm Forge", "textnode"])
    store.add_fact_status("textnode keeps the parser small", category="general")
    expanded = FactRetriever(store, hrr_dim=DIM)._expand_by_graph("Realm Forge", {}, None, 0.3)
    assert expanded, "a multi-word entity name did not seed the graph"
    assert store.entity_ids("textnode")[0] in {
        r["entity_id"] for r in store.read("SELECT DISTINCT entity_id FROM fact_entities WHERE fact_id IN ("
                                           + ",".join("?" * len(expanded)) + ")", list(expanded))}


def test_graph_discoveries_fill_the_tail_and_never_displace(tmp_path):
    """A discovery appears only where the text side left room, below every text hit; junk is never shown."""
    store = _store(tmp_path)
    store.add_fact_status("hubnode bridging satellite device", category="general",
                          entities=["hubnode", "satellite"])
    strong, _ = store.add_fact_status("hubnode review gate blocks unverified commits", category="general")
    related, _ = store.add_fact_status("satellite review gate notes about weather", category="general",
                                       entities=["satellite"])
    junk, _ = store.add_fact_status("completely different topic filler", category="general",
                                    entities=["satellite"])

    retriever = FactRetriever(store, hrr_dim=DIM)
    roomy = [f["fact_id"] for f in retriever.search("hubnode review gate", limit=10)]
    assert related in roomy                     # recall: reachable through the graph
    assert roomy.index(strong) < roomy.index(related)  # precision: below every text hit
    assert junk not in roomy                    # a co-occurrence edge alone is not evidence

    tight = [f["fact_id"] for f in retriever.search("hubnode review gate", limit=1)]
    assert tight == [strong], "a discovery displaced the only text hit"


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


def test_edges_are_backfilled_for_a_store_written_before_the_table_existed(tmp_path):
    """A store with fact_entities but no entity_relations gets the graph on its first writer open."""
    path = str(tmp_path / "legacy.db")
    store = MemoryStore(path, hrr_dim=DIM)
    store.add_fact_status("alphaword links betaword", category="general", entities=["alphaword", "betaword"])
    assert store.read("SELECT COUNT(*) c FROM entity_relations")[0]["c"] == 1
    with store._tx():  # what a pre-graph release left behind: links, no table, no trigger
        store._write("DROP TRIGGER IF EXISTS trg_fact_entities_edge")
        store._write("DROP TABLE entity_relations")
    store.close()

    reopened = MemoryStore(path, hrr_dim=DIM)
    try:
        assert reopened.read("SELECT COUNT(*) c FROM entity_relations")[0]["c"] == 1
        assert reopened.entity_neighbors(reopened.entity_ids("alphaword")) == set(reopened.entity_ids("betaword"))
    finally:
        reopened.close()


def test_graph_calls_are_no_ops_without_the_table(tmp_path):
    """Every graph write and read degrades to a no-op rather than failing a caller's transaction."""
    store = _store(tmp_path)
    with store._tx():
        store._write("DROP TRIGGER IF EXISTS trg_fact_entities_edge")
        store._write("DROP TABLE entity_relations")
    assert store.entity_neighbors([]) == set()
    assert store.entity_degrees([]) == {}
    assert store.rebuild_graph() == 0
    assert store.prune_graph() == 0
    store.forget_relations(1)  # no table: must not raise


def test_the_fact_cap_counts_eligible_facts_not_raw_links(tmp_path):
    """Forty links in another category must not spend the budget before an eligible discovery is reached."""
    store = _store(tmp_path)
    store.add_fact_status("hubnode bridges satellite", category="general", entities=["hubnode", "satellite"])
    for i in range(45):
        store.add_fact_status(f"other-category note {i:02d}", category="othercat", entities=["satellite"])
    eligible, _ = store.add_fact_status("satellite keeps a general note", category="general",
                                        entities=["satellite"])
    expanded = FactRetriever(store, hrr_dim=DIM)._expand_by_graph("hubnode", {}, category="general", min_trust=0.3)
    assert eligible in expanded, "the cap was spent on facts the category filter then dropped"
    already_found = FactRetriever(store, hrr_dim=DIM)._expand_by_graph(
        "hubnode", {eligible: {"content": "satellite keeps a general note"}}, category="general", min_trust=0.3)
    assert eligible not in already_found, "a fact the text search already returned was offered again as a discovery"


def test_graph_expansion_stops_at_the_fact_cap(tmp_path):
    """The discovery budget is an upper bound: more eligible neighbours do not flood the tail."""
    store = _store(tmp_path)
    seed, _ = store.add_fact_status("caphub bridges cappage", category="general", entities=["caphub", "cappage"])
    made = {store.add_fact_status(f"eligible note {i:02d}", category="general", entities=["cappage"])[0]
            for i in range(45)}
    expanded = FactRetriever(store, hrr_dim=DIM)._expand_by_graph("caphub", {}, category="general", min_trust=0.3)
    assert len(expanded) == 40, "the cap did not bound the discoveries"
    assert expanded <= made | {seed}, "a discovery came from outside the neighbour's facts"


def test_wiki_pointer_paths_need_a_namespace_and_a_page():
    """A bare file name is not a namespace/topic pair: <wiki>/page.md is the shortest accepted form."""
    wiki_entities = MemoryStore._wiki_page_entities
    assert wiki_entities("Wiki page: README.md | type: summary") == ()
    assert wiki_entities("Wiki page: index") == ()
    assert wiki_entities("no pointer here") == ()
    assert wiki_entities("Wiki page: ai-agents/llama-cpp.md") == ("ai agents", "llama cpp")
