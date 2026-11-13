"""Memory graph: the co_occurs write path, bounded one-hop expansion, interleaving, and repair.

Regression guards for a measured failure: a single generic query token resolved to a hub entity, the unbounded
one-hop expansion admitted 110 weakly related facts, and a flat relevance bonus pushed them into ranks 2-8 above
genuine text hits.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import retrieval as retrieval_mod
from hermes_plugin_holographic.graph import GRAPH_MAX_ENTITY_DEGREE, GRAPH_NEIGHBOUR_CAP
from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore

DIM = 512


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)


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


def test_prune_graph_repairs_a_stale_edge(tmp_path):
    """A link removed by an older release left its edge behind: prune_graph repairs it, a live edge is kept."""
    store = _store(tmp_path)
    store.add_fact_status("pruneship links keptleaf", category="general", entities=["pruneship", "keptleaf"])
    store.add_fact_status("pruneship links dropleaf", category="general", entities=["pruneship", "dropleaf"])
    pruneship, dropleaf = store.entity_ids("pruneship")[0], store.entity_ids("dropleaf")[0]
    with store._tx():  # the old state: links gone, edge still there
        store._write("DELETE FROM fact_entities WHERE entity_id = ?", (dropleaf,))
        store._write("INSERT OR IGNORE INTO entity_relations (src_entity_id, dst_entity_id, relation) VALUES (?, ?, ?)",
                     (min(pruneship, dropleaf), max(pruneship, dropleaf), "co_occurs"))
    assert store.prune_graph() >= 1
    assert store.entity_ids("keptleaf")[0] in store.entity_neighbors([pruneship])
    assert store.entity_ids("dropleaf")[0] not in store.entity_neighbors([pruneship])


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


def test_edges_are_backfilled_for_a_store_written_before_the_table_existed(tmp_path):
    """A store with fact_entities but no entity_relations gets the graph on its first writer open."""
    path = str(tmp_path / "legacy.db")
    store = MemoryStore(path, hrr_dim=DIM)
    store.add_fact_status("alphaword links betaword", category="general", entities=["alphaword", "betaword"])
    assert store.read("SELECT COUNT(*) c FROM entity_relations")[0]["c"] == 1
    with store._tx():  # what a pre-graph release left behind: links, no table, no triggers, no mark
        store._write("DROP TRIGGER IF EXISTS trg_fact_entities_edge")
        store._write("DROP TRIGGER IF EXISTS trg_fact_entities_unedge")
        store._write("DROP TABLE entity_relations")
        store._write("DELETE FROM meta WHERE key = 'graph_backfilled'")
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



def test_forget_removes_the_edge_it_implied(tmp_path):
    """Forget has to reach derived data: the association the fact created goes with it."""
    store = _store(tmp_path)
    link, _ = store.add_fact_status("Alice Cooper works with Bob Dylan", category="general")
    store.add_fact_status("Alice Cooper likes tea", category="general")
    store.add_fact_status("Bob Dylan likes coffee", category="general")
    assert store.read("SELECT COUNT(*) c FROM entity_relations")[0]["c"] == 1
    store.forget_fact(link)
    assert not store.read("SELECT 1 FROM entity_relations"), "the edge the forgotten fact implied survived"


def test_an_edge_survives_while_another_fact_links_the_pair(tmp_path):
    """Only the pair's LAST linking fact removes the edge."""
    store = _store(tmp_path)
    first, _ = store.add_fact_status("Carol King meets Dan Brown", category="general")
    store.add_fact_status("Carol King and Dan Brown share a publisher", category="general")
    store.forget_fact(first)
    assert store.read("SELECT COUNT(*) c FROM entity_relations")[0]["c"] == 1, \
        "an edge still linked by a current fact was removed"


def test_a_relink_drops_the_edge_the_old_links_implied(tmp_path):
    """An update is a DELETE plus an INSERT: the edges of the old links go with them."""
    store = _store(tmp_path)
    fact, _ = store.add_fact_status("Erin Gray knows Finn Hale", category="general")
    store.add_fact_status("Erin Gray knows Gus Hart", category="general")
    with store._tx():
        store._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact,))
        store._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                     (fact, store.entity_ids("gus hart")[0]))
    erin, gus = store.entity_ids("Erin Gray")[0], store.entity_ids("Gus Hart")[0]
    assert store.entity_neighbors([erin]) == {gus}
    assert store.entity_ids("finn hale")[0] not in store.entity_neighbors([erin])


def test_text_hit_order_is_identical_with_and_without_the_graph(tmp_path):
    """A discovery must never change how the text hits rank; the coverage test runs on text candidates only."""
    store = _store(tmp_path)
    store.add_fact_status("hubnode bridging satellite device", category="general",
                          entities=["hubnode", "satellite"])
    store.add_fact_status("satellite weirdness in the review gate notes", category="general", entities=["satellite"])
    store.add_fact_status("hubnode review gate blocks unverified commits", category="general")
    store.add_fact_status("review gate wording", category="general")
    retriever = FactRetriever(store, hrr_dim=DIM)
    retriever.graph_expansion = False
    text_only = [f["fact_id"] for f in retriever.search("hubnode review gate", limit=10)]
    retriever.graph_expansion = True
    with_graph = [f["fact_id"] for f in retriever.search("hubnode review gate", limit=10)]
    assert with_graph[:len(text_only)] == text_only, "the graph reordered the text hits"


def test_a_read_only_open_creates_no_graph_schema(tmp_path):
    """migrate=False is a reader: it must not create tables, indexes or triggers."""
    fresh = str(tmp_path / "fresh_reader.db")
    reader = MemoryStore(fresh, hrr_dim=DIM, migrate=False)
    try:
        assert reader.read("SELECT name FROM sqlite_master WHERE name IN "
                           "('entity_relations', 'trg_fact_entities_edge', 'trg_fact_entities_unedge')") == []
    finally:
        reader.close()


def test_backfill_is_keyed_on_a_meta_mark_not_on_an_empty_table(tmp_path):
    """Edges written before the first migrating writer must not make the legacy pairs look backfilled."""
    path = str(tmp_path / "legacy.db")
    store = MemoryStore(path, hrr_dim=DIM)
    store.add_fact_status("alphaword links betaword", category="general", entities=["alphaword", "betaword"])
    store.add_fact_status("gammaword links deltaword", category="general", entities=["gammaword", "deltaword"])
    with store._tx():  # what an older writer of this version could leave: some edges, no mark
        store._write("DELETE FROM meta WHERE key = 'graph_backfilled'")
        store._write("DELETE FROM entity_relations WHERE src_entity_id = ?",
                     (min(store.entity_ids("gammaword")[0], store.entity_ids("deltaword")[0]),))
    store.close()
    reopened = MemoryStore(path, hrr_dim=DIM)
    try:
        assert reopened.read("SELECT COUNT(*) c FROM entity_relations")[0]["c"] == 2
        assert reopened._meta("graph_backfilled") is not None
    finally:
        reopened.close()


def test_neighbour_facts_are_read_in_one_query(tmp_path, monkeypatch):
    """Prefetch runs expansion every turn: N neighbour facts must not cost N reads."""
    store = _store(tmp_path)
    store.add_fact_status("onehub bridges onepage", category="general", entities=["onehub", "onepage"])
    for i in range(5):
        store.add_fact_status(f"note {i} on the page", category="general", entities=["onepage"])
    retriever = FactRetriever(store, hrr_dim=DIM)
    calls = []
    original = retriever._rows

    def counting(ids, *args, **kwargs):
        calls.append(list(ids))
        return original(ids, *args, **kwargs)

    monkeypatch.setattr(retriever, "_rows", counting)
    expanded = retriever._expand_by_graph("onehub", {}, None, 0.3)
    assert len(expanded) > 1, "the fixture must yield several discoveries"
    assert len(calls) == 1, f"neighbour facts cost {len(calls)} reads"


def test_edge_trust_does_not_disable_the_graph(tmp_path):
    """Edges default to 0.5; comparing that with min_trust switched the graph off for min_trust > 0.5."""
    store = _store(tmp_path)
    store.add_fact_status("highhub bridges highpage", category="general", entities=["highhub", "highpage"], trust=0.9)
    discovered, _ = store.add_fact_status("highpage note worth remembering", category="general",
                                          entities=["highpage"], trust=0.9)
    expanded = FactRetriever(store, hrr_dim=DIM)._expand_by_graph("highhub", {}, None, min_trust=0.8)
    assert discovered in expanded, "an edge's default trust filtered a fact the fact-level min_trust admits"


def test_neighbour_facts_beyond_the_sqlite_parameter_limit(tmp_path):
    """The facts behind 25 neighbours have no ceiling: reading them in one IN list fails on a 999-parameter SQLite."""
    store = MemoryStore(str(tmp_path / "big.db"), hrr_dim=DIM)
    for i in range(20):
        store.add_fact_status(f"Seed Person works with Partner Number{i}", category="general")
        for k in range(60):
            store.add_fact_status(f"Partner Number{i} shipped release {k}", category="general")
    store._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    retriever = FactRetriever(store, hrr_dim=DIM)
    retriever.graph_expansion = True
    hits = retriever.search("Seed Person", limit=15)
    assert hits, "the text hit must still come back"
    assert len(hits) <= 15


def test_graph_on_reader_of_a_pre_graph_store_does_not_raise(tmp_path):
    """graph_expansion on a store opened without migrating returns the text hits, not an OperationalError."""
    path = tmp_path / "pre.db"
    writer = MemoryStore(str(path), hrr_dim=DIM)
    writer.add_fact_status("Alice Cooper met Bob Dylan", category="general")
    with writer._tx():  # what a store written by a pre-graph release looks like
        writer._write("DROP TRIGGER IF EXISTS trg_fact_entities_edge")
        writer._write("DROP TRIGGER IF EXISTS trg_fact_entities_unedge")
        writer._write("DROP TABLE IF EXISTS entity_relations")
        writer._write("DELETE FROM meta WHERE key = 'graph_backfilled'")
    writer.close()

    reader = MemoryStore(str(path), hrr_dim=DIM, migrate=False)
    try:
        retriever = FactRetriever(reader, hrr_dim=DIM)
        retriever.graph_expansion = True
        hits = retriever.search("Alice Cooper", limit=5)
        assert hits, "a read-only open lost the text hit"
    finally:
        reader.close()
