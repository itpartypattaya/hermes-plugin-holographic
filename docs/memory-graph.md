# Memory graph (retrieval Phase 4)

Two entities a fact links become neighbours. A fact that says "portlandship pairs with rivernode" links both names,
and every pair of entities one fact links gets a `co_occurs` edge in `entity_relations`. Search then walks one hop out
of the entities the query resolves to, so a fact about an entity next to the query's is reachable without sharing a
word with the query.

The table is not written by the retrieval path and is not a source of truth. `fact_entities` is: the edge is created
by `trg_fact_entities_edge` (in `GRAPH_SCHEMA`, part of the schema init, so a fresh store has it), a merge repoints
the edges of the entity it merges away (`_repoint_relations`), a delete drops them (`forget_relations`), and
`rebuild_graph()` recreates every edge from `fact_entities` at any time. A store written before the table existed gets
that backfill on its first writer open (`_migrate_graph`). `prune_graph()` removes the edges no current fact implies —
stale, not wrong: they only widen one hop slightly, so it is not run automatically.

## The bounds

The hop is the part that goes wrong, so both ends of it are bounded:

- `GRAPH_MAX_ENTITY_DEGREE` (40): an entity touching more others than this is a hub — a namespace ("ai agents", a
  project name), a wiki root, a word every fact mentions. Expanding from a hub, or into one, offers arbitrary
  neighbours rather than related ones. Hubs are neither expanded from nor walked into.
- `GRAPH_NEIGHBOUR_CAP` (25): neighbour entities traversed per query.
- 40 facts admitted per query from those neighbours.

Measured on an 806-fact store: one generic query token resolved to an entity of degree 57, and an unbounded expansion
admitted 110 weakly related facts into the candidate pool.

## Discoveries fill the tail, they never displace

A graph-expanded fact is a *discovery*, not a better text match. It is shown only where the text side left room: the
result list is the text hits first, then discoveries up to `limit`, each at least `_GRAPH_MIN_REL_RATIO` (0.5) of the
**weakest shown text hit**. When the text hits already fill the list, the graph is inert for that query.

That is a measurement, not a preference. On a live 807-fact store, discoveries score ~0.05 (`fts_rank` is 0 by
construction, so they carry only the jaccard and HRR terms, times trust) while text hits score 0.17–0.42. A ratio
against the top hit (0.5) therefore never fired — 0 of 15 realistic queries surfaced a discovery. Interleaving at slot
4 with the ratio removed did fire, and pushed genuine hits down to make room for filler ("operations/code-review" under
a Realm Forge query). Tail-fill is the only rule of the three that adds recall without subtracting any.

The feature is therefore live where recall is sparse (few text hits, or a `limit` above what the words match) and silent
where the words already answer. That is the case it exists for: a fact about an entity next to the query's, reachable
when nothing in the query matches it directly.

Expansion runs only over the current facts and only in the mode a user's query takes; `as_of` searches and write-time
conflict checks leave the graph alone.

## Wiki pointer facts

Bulk wiki ingest writes facts like "[WIKI HINT] Wiki page: ai-agents/entities/llama-cpp.md": no capitalised phrase,
so the entity extractor found nothing and every re-ingested page stayed invisible to the graph. Such a line now
contributes its namespace (`ai agents`) and page slug (`llama cpp`); structural slugs (index, schema, log, readme,
overview, changelog, notes, todo) are dropped — they are the wiki's structure, not topics.

## What the graph does not do

- It does not invent relations: only `co_occurs` exists, and only where one fact links both entities.
- It does not override text relevance: a discovery below half the top score is never shown.
- It does not survive an entity reindex by itself: the first open of a store indexed under older entity rules
  rebuilds the entity index (upstream behaviour, `meta.entity_rules`), and the graph follows the new links — fewer,
  more accurate edges, not the old ones.
