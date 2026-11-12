"""Memory graph: entity relations, one-hop expansion, hub gating (retrieval Phase 4).

A fact that links two entities creates the co_occurs edge between them (the
``trg_fact_entities_edge`` trigger in :data:`GRAPH_SCHEMA`), so the graph follows the
same truth as ``fact_entities``: whoever links a fact to an entity links the entities
to each other. Retrieval then walks ONE hop out of the entities a query resolves to,
which reaches a fact about a structurally connected entity that shares no word with
the query.

The hop is bounded on purpose. A namespace-shaped entity ("llama.cpp", a project
name, a wiki root) touches a whole corpus; expanding from it, or into it, offers
arbitrary neighbours rather than related ones (observed: one generic token
contributed 110 facts). :data:`GRAPH_MAX_ENTITY_DEGREE` drops such hubs,
:data:`GRAPH_NEIGHBOUR_CAP` bounds the fan-out per query.

Edges are additive and cheap; nothing here is a source of truth. ``rebuild_graph``
recreates them from ``fact_entities`` at any time, which is what a store written
before this table existed gets on its first writer open.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_relations (
    src_entity_id INTEGER REFERENCES entities(entity_id),
    dst_entity_id INTEGER REFERENCES entities(entity_id),
    relation      TEXT NOT NULL DEFAULT 'co_occurs',
    weight        REAL DEFAULT 1.0,
    trust_score   REAL DEFAULT 0.5,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (src_entity_id, dst_entity_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_er_src ON entity_relations(src_entity_id);
CREATE INDEX IF NOT EXISTS idx_er_dst ON entity_relations(dst_entity_id);
CREATE INDEX IF NOT EXISTS idx_er_relation ON entity_relations(relation);

-- The write path, in the schema init and not only in a migration: a fresh store
-- without this trigger has an empty graph, and two entities linked by the same fact
-- then never become neighbours.
CREATE TRIGGER IF NOT EXISTS trg_fact_entities_edge
AFTER INSERT ON fact_entities
BEGIN
  INSERT OR IGNORE INTO entity_relations (src_entity_id, dst_entity_id, relation, weight)
  SELECT fe1.entity_id, fe2.entity_id, 'co_occurs', 1.0
  FROM fact_entities fe1, fact_entities fe2
  WHERE fe1.fact_id = NEW.fact_id AND fe2.fact_id = NEW.fact_id
    AND fe1.entity_id < fe2.entity_id;
END;
"""

GRAPH_NEIGHBOUR_CAP = 25       # neighbour entities traversed per query
GRAPH_MAX_ENTITY_DEGREE = 40   # entities with more edges than this are hubs: never traversed


class GraphMixin:
    """The graph half of MemoryStore. Expects read, _write, _tx, _one and _bump of MemoryStore."""

    def entity_neighbors(self, entity_ids: list[int], min_trust: float = 0.0, limit: int = 50,
                         max_degree: int | None = None) -> set[int]:
        """Entity ids one hop from *entity_ids* (excluding them).

        *max_degree* drops hub entities on both sides: an entity touching more than
        that many others says nothing about which neighbour is related to a query.
        With *max_degree* the rows are filtered before the cap, so the cap is not spent
        on hubs that the filter then discards; without it the SQL limit applies.
        """
        ids = [int(e) for e in entity_ids or ()]
        if not ids:
            return set()
        marks = ", ".join("?" * len(ids))
        sql = ("SELECT dst_entity_id AS eid FROM entity_relations "
               f"WHERE src_entity_id IN ({marks}) AND trust_score >= ? "
               "UNION ALL "
               "SELECT src_entity_id AS eid FROM entity_relations "
               f"WHERE dst_entity_id IN ({marks}) AND trust_score >= ? "
               "ORDER BY eid")
        params: list = [*ids, min_trust, *ids, min_trust]
        if max_degree is None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.read(sql, params)
        found = {int(r["eid"]) for r in rows} - set(ids)
        if max_degree is not None and found:
            degree = self.entity_degrees(list(found))
            found = {eid for eid in found if degree.get(eid, 0) <= max_degree}
        return set(sorted(found)[:limit])

    def entity_degrees(self, entity_ids: list[int]) -> dict[int, int]:
        """{entity_id: number of graph edges touching it}."""
        ids = [int(e) for e in entity_ids or ()]
        if not ids:
            return {}
        marks = ", ".join("?" * len(ids))
        return {int(r["eid"]): int(r["deg"]) for r in self.read(
            "SELECT eid, COUNT(*) AS deg FROM ("
            f"SELECT src_entity_id AS eid FROM entity_relations WHERE src_entity_id IN ({marks}) "
            "UNION ALL "
            f"SELECT dst_entity_id AS eid FROM entity_relations WHERE dst_entity_id IN ({marks})"
            ") GROUP BY eid", [*ids, *ids])}

    def _graph_ready(self) -> bool:
        """Whether this store has the graph table: a store written before it existed has none until a writer migrates
        it, and every graph write must stay a no-op until then rather than fail the caller's transaction."""
        return bool(self._one("SELECT 1 FROM sqlite_master WHERE type='table' AND name='entity_relations'"))

    def rebuild_graph(self) -> int:
        """Create the co_occurs edges ``fact_entities`` implies; returns how many were added."""
        if not self._graph_ready():
            return 0
        before = int(self._one("SELECT COUNT(*) FROM entity_relations")[0])
        with self._tx():
            self._write(
                "INSERT OR IGNORE INTO entity_relations (src_entity_id, dst_entity_id, relation, weight) "
                "SELECT fe1.entity_id, fe2.entity_id, 'co_occurs', 1.0 "
                "FROM fact_entities fe1 JOIN fact_entities fe2 ON fe1.fact_id = fe2.fact_id "
                "WHERE fe1.entity_id < fe2.entity_id")
            self._bump()
        return int(self._one("SELECT COUNT(*) FROM entity_relations")[0]) - before

    def prune_graph(self) -> int:
        """Delete the edges no current fact implies (a fact's links changed, or it was forgotten); returns the count.

        The graph is derived data, so this is safe at any time; it is not run automatically because the full scan
        costs as much as a rebuild and the stale edges are harmless: they only widen one hop slightly.
        """
        if not self._graph_ready():
            return 0
        before = int(self._one("SELECT COUNT(*) FROM entity_relations")[0])
        with self._tx():
            self._write(
                "DELETE FROM entity_relations WHERE NOT EXISTS ("
                "SELECT 1 FROM fact_entities a JOIN fact_entities b ON a.fact_id = b.fact_id "
                "WHERE a.entity_id = entity_relations.src_entity_id "
                "AND b.entity_id = entity_relations.dst_entity_id)")
            self._bump()
        return before - int(self._one("SELECT COUNT(*) FROM entity_relations")[0])

    def _migrate_graph(self) -> None:
        """Backfill the graph once for a store written before ``entity_relations`` existed (inside _migrate's tx)."""
        if int(self._one("SELECT COUNT(*) FROM entity_relations")[0]) or \
                not int(self._one("SELECT COUNT(*) FROM fact_entities")[0]):
            return
        added = self.rebuild_graph()
        if added:
            logger.info("holographic: memory graph backfilled with %d co-occurrence edges", added)

    def _repoint_relations(self, source: int, target: int) -> None:
        """Move *source*'s edges onto *target* (entity merge; inside the caller's transaction).

        Every edge touching *source* is read, deleted and re-inserted as one canonical direction
        (smaller id first) with INSERT OR IGNORE, so a self-loop, a duplicate or the reverse of an
        edge *target* already has collapses into one row instead of surviving as a half-move.
        """
        if source == target or not self._graph_ready():
            return
        rows = self.read(
            "SELECT src_entity_id AS a, dst_entity_id AS b, relation, weight, trust_score "
            "FROM entity_relations WHERE src_entity_id = ? OR dst_entity_id = ?", (source, source))
        self._write("DELETE FROM entity_relations WHERE src_entity_id = ? OR dst_entity_id = ?", (source, source))
        for row in rows:
            a = target if int(row["a"]) == source else int(row["a"])
            b = target if int(row["b"]) == source else int(row["b"])
            if a == b:
                continue
            self._write(
                "INSERT OR IGNORE INTO entity_relations (src_entity_id, dst_entity_id, relation, weight, trust_score) "
                "VALUES (?, ?, ?, ?, ?)", (min(a, b), max(a, b), row["relation"], row["weight"], row["trust_score"]))

    def forget_relations(self, entity_id: int) -> None:
        """Drop every edge of a deleted entity (inside the caller's transaction)."""
        if not self._graph_ready():
            return
        self._write("DELETE FROM entity_relations WHERE src_entity_id = ? OR dst_entity_id = ?",
                    (entity_id, entity_id))
