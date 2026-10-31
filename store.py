"""SQLite-backed fact store with entity resolution and trust scoring (single-user Hermes memory plugin)."""

import contextlib
from collections import Counter
import logging
import os
import sqlite3
import threading
from pathlib import Path

from . import holographic as hrr
from .embeddings import text_hash
from .entities import _GIVEN_ALIAS, _MAX_COUNTED_IDS, _SPLITS, EntityMixin
from .entity_names import _clean_given, entity_key, fold
from .entity_names import (  # noqa: F401 - re-exported: they lived here, and the provider and tests import them from store
    _RE_TOKEN, _capitalized_phrases, _entity_key_0160, _name_shaped, _outside_plain_quotes, _script_pieces, _valid_entity,
    _without_code, unusable_names)
from .history import HISTORY_SCHEMA, VERSION_NAMES_TRIGGERS, HistoryMixin, NotFound, error_kind, parse_day
from .experience import ExperienceMixin

logger = logging.getLogger(__name__)

# FTS is re-indexed only when indexed text changes: counters, trust and vectors are updated far more often.
_FACTS_AU = """CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF content, tags ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

""" + _FACTS_AU + """

-- 0.4.0. Older versions ignore these tables; see _migrate() for how their writes are reconciled on open.
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_key TEXT PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entities(entity_id),
    alias     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity ON entity_aliases(entity_id);

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
# 0.8.0 history tables (history.py) are created by a writer in _migrate(): a read-only context creates nothing.

# 0.14.0 search by meaning (embeddings.py): at most one vector per current fact, made by *model* from the text whose
# hash is *text_hash*. Created by every writer, whether or not the feature is on, so that deleting a fact always deletes
# its vector; older versions never read it.
_EMBEDDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_id    INTEGER PRIMARY KEY,
    model      TEXT NOT NULL,
    text_hash  TEXT NOT NULL,
    vector     BLOB NOT NULL,
    revision   TEXT NOT NULL DEFAULT '',  -- random, new with every write: a vector made again is read again
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_VERSION = 2  # 1 = 0.1.x-0.3.x layout, 2 = + meta, entities.name_key, entity_aliases
# The rules entities are guessed from a fact's text by (meta.entity_rules; none recorded = 1, before 0.16.0). A store
# indexed under older rules is rebuilt on a writer's first open (rebuild_entity_index), up to the fact count below; a
# larger one on request. Raise it with every change of _extract_entities / _name_shaped that drops what they took.
_ENTITY_RULES = 4  # 2: 0.16.0; 3: 0.16.1 (script seams, line breaks, leading symbols, counts in quotes);
# 4: 0.16.2 (Chinese names that begin with a number, Chinese counts)
_AUTO_REBUILD_MAX_FACTS = 2000
_AUTO_REBUILD_MAX_CHARS = 500_000  # and in all; the work grows with the text, not the count (review 0.16.0)
# 0.16.0: names delete_entity removed, as keyed hashes (see _name_digest). Created by a writer in _migrate().
_ENTITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_tombstones (
    digest     TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
_HELPFUL_DELTA, _UNHELPFUL_DELTA = 0.05, -0.10

_ENTITY_NAMES_SQL = "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?"


class DuplicateContentError(ValueError):
    """update_fact would make a fact identical to another one (content is UNIQUE)."""

    def __init__(self, fact_id: int):
        super().__init__(f"another fact already has this content (fact_id {fact_id})")
        self.fact_id = fact_id


def _clamp_trust(value: float) -> float:
    return max(0.0, min(1.0, value))


_UNRESTRICTED: set[str] = set()  # files whose owner-only mode could not be set, warned about once each


def _restrict_permissions(db_path: Path) -> None:
    """Owner-only access to the database and its WAL side files (POSIX; best effort, but a failure is logged once per
    file: the README promises 0600, and a store readable by others must not stay so without a word)."""
    if os.name != "posix":
        return
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{db_path}{suffix}")
        try:
            if path.exists() and path.stat().st_mode & 0o077:
                os.chmod(path, 0o600)
        except OSError as exc:
            if str(path) not in _UNRESTRICTED:
                _UNRESTRICTED.add(str(path))
                logger.warning("holographic: could not make %s readable by its owner only (%s)", path, exc)


class MemoryStore(EntityMixin, HistoryMixin, ExperienceMixin):
    """SQLite-backed fact store with entity resolution and trust scoring.

    Process-wide shared connection registry: SQLite allows one writer at a time and several providers
    coexist per process (main agent + every delegate_task subagent), so all instances for the same database
    share ONE connection and ONE re-entrant lock — writes are fully serialized and "database is locked" is
    impossible within a process. Refcounted: closing one instance never tears the connection out from under a
    sibling. Every multi-statement write runs in one transaction, so a failure leaves nothing half-written."""

    _shared: dict = {}
    _shared_guard = threading.Lock()

    def __init__(self, db_path: "str | Path | None" = None, default_trust: float = 0.5, hrr_dim: int = 1024, *,
                 migrate: bool = True, tz=None) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust, self.hrr_dim = _clamp_trust(default_trust), hrr_dim
        self._hrr_override: bool | None = None  # _hrr_available asks numpy on first use, not at open (#145156)
        self.tz = tz  # local dates of the history (recorded_on, "today"); None = UTC
        try:  # resolve() so symlinked/relative paths to the same file share ONE connection
            self._key = str(self.db_path.resolve())
        except OSError:
            self._key = str(self.db_path)
        with MemoryStore._shared_guard:
            entry = MemoryStore._shared.get(self._key)
            if entry is None:
                # Autocommit outside explicit transactions (_tx): a failed statement never leaves a dangling
                # transaction holding the write lock.
                conn = sqlite3.connect(self._key, check_same_thread=False, timeout=10.0, isolation_level=None)
                conn.row_factory = sqlite3.Row
                conn.create_function("hfold", 1, fold, deterministic=True)
                conn.execute("PRAGMA secure_delete = ON")  # freed pages and cells are zeroed: forgotten text leaves no copy
                entry = MemoryStore._shared[self._key] = {"conn": conn, "lock": threading.RLock(), "refs": 0, "ready": False,
                                                          "migrated": False, "generation": 0}
            entry["refs"] += 1
            self._entry, self._conn, self._lock = entry, entry["conn"], entry["lock"]
        with self._lock:  # schema initialised once per shared connection; migrated by the first writer
            try:
                if not entry["ready"]:
                    self._init_db()
                    entry["ready"] = True
                if migrate and not entry["migrated"]:
                    self._migrate()
                    entry["migrated"] = True
                    self._maybe_rebuild_entities()
                    self.scrub()  # a forget whose file rewrite was blocked last time
                elif migrate:
                    self._check_vectors()
                    self._drop_orphan_vectors()  # facts an older version deleted while this process held the file
                    self.drop_orphan_confirmations()  # the same for confirmations (review 2026-10-15)
                    self.scrub()  # a forget whose file rewrite another process blocked: retried by every writer open
            except BaseException:
                self.close()  # drop our reference; a broken connection must not stay registered
                raise
        _restrict_permissions(self.db_path)

    def _init_db(self) -> None:
        """Create schema, enable WAL via the shared fallback helper (NFS/SMB/FUSE degrade gracefully), add hrr_vector
        to pre-HRR DBs, and narrow the FTS update trigger of DBs created by older versions."""
        from hermes_state_wal import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="memory_store.db (holographic)")
        self._conn.executescript(_SCHEMA)
        if "hrr_vector" not in {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}:
            from hermes_cli.sqlite_util import add_column_if_missing
            add_column_if_missing(self._conn, "facts", "hrr_vector", "hrr_vector BLOB")
        trigger = self._one("SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'facts_au'")
        if trigger is not None and "UPDATE OF" not in (trigger["sql"] or "").upper():
            with self._tx():
                self._conn.execute("DROP TRIGGER facts_au")
                self._conn.execute(_FACTS_AU)

    def _migrate(self) -> None:
        """Bring the store to SCHEMA_VERSION, idempotently, on every open. Additive only, so every older version
        keeps working on the file; what an older version wrote in between is reconciled here:
        - entities it created have no ``name_key``: keys are filled in, and entities that turn out to share a key
          are merged (links and aliases moved to the oldest one);
        - aliases it may have stored in the legacy ``entities.aliases`` column are copied into ``entity_aliases``;
        - HRR vectors are re-encoded when ``hrr_dim`` differs from the dimension they were written with (recorded in
          ``meta``; before 0.4.0, inferred from the stored blobs), so a changed setting no longer breaks queries.
        0.8.0 adds the history tables (executescript commits, so before the transaction) and versions every fact; 0.14.0
        the vectors of search by meaning, whose rows for facts an older version deleted are dropped here; 0.15.0 the
        origin of a version and the confirmations of a fact, whose rows an older version's forget left are dropped;
        0.16.0 where an alias came from (entity_aliases.source: older rows and an older version's writes count as given
        by the user) and the tombstones of deleted names. The entity index is rebuilt after the migration, by
        _maybe_rebuild_entities. An older version reads such a store as before; one that writes to it as well may
        create a deleted name again from a new fact's text (its next open by this version does not notice that)."""
        self._conn.executescript(HISTORY_SCHEMA + _EMBEDDINGS_SCHEMA + _ENTITY_SCHEMA + VERSION_NAMES_TRIGGERS)
        with self._tx():
            if "source" not in {r[1] for r in self._conn.execute("PRAGMA table_info(entity_aliases)")}:
                # 0.16.0. 'legacy': written before, or by an older release since; nothing says whether someone gave it
                self._conn.execute("ALTER TABLE entity_aliases ADD COLUMN source TEXT DEFAULT 'legacy'")
            self._entry.pop("alias_source", None)
            self._conn.execute("DELETE FROM fact_embeddings WHERE fact_id NOT IN (SELECT fact_id FROM facts)")
            if "src_origin" not in {r[1] for r in self._conn.execute("PRAGMA table_info(fact_versions)")}:
                self._conn.execute("ALTER TABLE fact_versions ADD COLUMN src_origin TEXT")
            self._entry.pop("source_columns", None)
            self._conn.execute("DELETE FROM fact_confirmations WHERE fact_id NOT IN (SELECT fact_id FROM fact_versions)")
            if "name_key" not in {r[1] for r in self._conn.execute("PRAGMA table_info(entities)")}:
                self._conn.execute("ALTER TABLE entities ADD COLUMN name_key TEXT")
            # Aliases an older version deleted the entity of, or that point nowhere: drop before anything resolves them.
            self._conn.execute("DELETE FROM entity_aliases WHERE entity_id NOT IN (SELECT entity_id FROM entities)")
            # Legacy comma-separated aliases are imported BEFORE keys are reconciled: a merge deletes the source row.
            names = {entity_key(r["name"]) for r in self._conn.execute("SELECT name FROM entities")}
            for row in self._conn.execute("SELECT entity_id, aliases FROM entities WHERE aliases IS NOT NULL AND aliases != ''").fetchall():
                for alias in row["aliases"].split(","):
                    key = entity_key(alias)
                    if key and key not in names:
                        self._conn.execute("INSERT OR IGNORE INTO entity_aliases (alias_key, entity_id, alias) VALUES (?, ?, ?)",
                                           (key, row["entity_id"], alias.strip()))
            self._reconcile_entity_keys()
            self._rekey_entities()
            self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_key ON entities(name_key) WHERE name_key IS NOT NULL")
            self._sync_hrr_dim()
            self._migrate_history()
            self._bump()
            self._set_meta("schema_version", str(SCHEMA_VERSION))

    def _meta(self, key: str) -> str | None:
        row = self._one("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._write("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value))

    @property
    def _hrr_available(self) -> bool:
        """Whether HRR vectors can be computed: imports numpy on the first ask (has_numpy), so a store that is only
        opened and read never loads it. Assignable, for a store that must not compute them (and for tests)."""
        override = getattr(self, "_hrr_override", None)
        return hrr.has_numpy() if override is None else override

    @_hrr_available.setter
    def _hrr_available(self, value: bool) -> None:
        self._hrr_override = bool(value)

    def _vectors_possible(self) -> bool:
        """The open-time gate of the vector repairs: numpy present, asked without importing it. A repair imports numpy
        only when there is a vector to write, and does nothing when that import fails."""
        override = getattr(self, "_hrr_override", None)
        return hrr.numpy_installed() if override is None else override

    def _check_vectors(self) -> None:
        """Re-encode vectors whose size does not match hrr_dim (another version or setting wrote them)."""
        if not self._vectors_possible():
            return
        sizes = (4 + 4 * self.hrr_dim, 8 * self.hrr_dim)
        stale = self._conn.execute("SELECT fact_id, content FROM facts WHERE hrr_vector IS NULL OR LENGTH(hrr_vector) NOT IN (?, ?)",
                                   sizes).fetchall()
        if stale and self._hrr_available:  # work to do: numpy is imported now, and a broken one leaves them as they are
            with self._tx():
                for row in stale:
                    self._compute_hrr_vector(row["fact_id"], row["content"])

    def _sync_hrr_dim(self) -> None:
        """Record hrr_dim; re-encode every vector when the configured dimension differs from the stored one."""
        if not self._vectors_possible():
            return
        stored = self._meta("hrr_dim")
        if stored is None:  # before 0.4.0: infer from a stored blob (prefixed float32 or legacy float64)
            blob = self._one("SELECT hrr_vector FROM facts WHERE hrr_vector IS NOT NULL LIMIT 1")
            if blob is not None:
                data = bytes(blob["hrr_vector"])
                stored = str((len(data) - 4) // 4 if data.startswith(b"HRR1") else len(data) // 8)
        if stored is not None and stored != str(self.hrr_dim):
            if not self._hrr_available:  # found but not importable: keep the old size on record, nothing re-encoded
                return
            logger.warning("holographic: hrr_dim changed (%s -> %s), re-encoding stored vectors", stored, self.hrr_dim)
            for row in self._conn.execute("SELECT fact_id, content FROM facts").fetchall():
                self._compute_hrr_vector(row["fact_id"], row["content"])
            self._conn.execute("DELETE FROM memory_banks")
        self._check_vectors()  # the record is not proof: an older writer may have stored other sizes since
        self._set_meta("hrr_dim", str(self.hrr_dim))

    def _one(self, sql: str, params=()):
        return self._conn.execute(sql, params).fetchone()

    def _write(self, sql: str, params=()) -> sqlite3.Cursor:
        """One write statement: autocommitted on its own, or part of the enclosing _tx()."""
        return self._conn.execute(sql, params)

    @contextlib.contextmanager
    def _tx(self):
        """One write transaction (BEGIN IMMEDIATE ... COMMIT, ROLLBACK on error); nested calls join the outer one.

        The COMMIT is inside the guard: in rollback-journal mode (the core keeps DELETE on a SQLite it deems
        WAL-reset-vulnerable, and on NFS / SMB) a reader in another process makes COMMIT fail with "database is
        locked" after the busy timeout, and a commit that failed leaves the transaction open; without the ROLLBACK
        every later _tx() would join that dangling transaction and nothing would ever be committed again."""
        with self._lock:
            if self._conn.in_transaction:
                yield
                return
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._conn.execute("COMMIT")
                if getattr(_SPLITS, "pending", None):  # committed: what collect_splits reports (0.17.1)
                    _SPLITS.found.update(_SPLITS.pending)
                    _SPLITS.pending = []
            except BaseException:
                if getattr(_SPLITS, "pending", None):
                    _SPLITS.pending = []  # rolled back: it never happened
                if self._conn.in_transaction:
                    try:
                        self._conn.execute("ROLLBACK")
                    except sqlite3.Error as exc:  # the connection is broken: the error raised below says so
                        logger.warning("holographic: rollback after a failed write failed too (%s)", exc)
                self._bump()  # anything cached inside the transaction never happened
                raise

    def read(self, sql: str, params=()) -> list:
        """Run a read under the store lock (a consistent snapshot against this process's writers)."""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def add_fact(self, content: str, category: str = "general", tags: str = "") -> int:
        """Insert a fact and return its fact_id; on duplicate content (UNIQUE) return the existing fact_id untouched."""
        return self.add_fact_status(content, category=category, tags=tags)[0]

    def add_fact_status(self, content: str, category: str = "general", tags: str = "", *,
                        trust: float | None = None, entities=(), valid_from=None,
                        source: dict | None = None) -> tuple[int, bool]:
        """(fact_id, created). Links extracted, declared (*entities*) and already known entities and stores the HRR
        vector and the first version (*valid_from*: the date it started to hold, None = unknown; *source*: where it
        was written) in the same transaction. *trust* overrides the default trust of a new fact."""
        content = (content or "").strip()
        if not content:
            raise ValueError("content must not be empty")
        start = parse_day(valid_from, field="valid_from", today=self._today())
        with self._tx():
            try:
                fact_id: int = self._write(
                    "INSERT INTO facts (content, category, tags, trust_score) VALUES (?, ?, ?, ?)",
                    (content, category, tags, _clamp_trust(self.default_trust if trust is None else trust))).lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                existing = int(self._one("SELECT fact_id FROM facts WHERE content = ?", (content,))["fact_id"])
                self._declare(existing, entities)  # who it is about still counts (0.17.0; dropped until 0.16.3)
                return existing, False
            self._link_entities(fact_id, content, entities)
            self._compute_hrr_vector(fact_id, content)
            self._invalidate_bank(category)
            if self.history_ready():
                self._insert_version(fact_id, content, category, tags, valid_from=start, source=source)
            return fact_id, True

    def update_fact(self, fact_id: int, content: str | None = None, trust_delta: float | None = None,
                    tags: str | None = None, category: str | None = None, *, entities=(), touch: bool = True,
                    source: dict | None = None) -> bool:
        """Partially update a fact (trust clamped to [0, 1]). Returns True if the row existed. Raises ValueError for
        empty content and DuplicateContentError when another fact already has the new content. touch=False keeps
        updated_at (temporal decay and "recently changed" read it): for search aids that do not change the fact.

        A new text is a correction (as history.correct_fact): the old text stays as a `corrected` version, and the new
        one starts at its origin's trust with no helpful marks, recalls or confirmations. Tags and category change the
        current version in place."""
        if content is not None:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")
        with self._tx():
            row = self._one("SELECT fact_id, trust_score, category, content, tags FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                return False
            history = self.history_ready()
            if history:
                self._reconcile_history()
            # A new text is a new claim (as correct_fact): it starts at its origin's trust, without the helpful marks,
            # recalls and confirmations the old text earned; trust_delta, when given, moves that start
            new_claim = content is not None and content != row["content"]
            base = row["trust_score"]
            if new_claim and history:
                other = self._one("SELECT fact_id FROM facts WHERE content = ? AND fact_id != ?", (content, fact_id))
                if other is None:  # the last text of a copy merged into another fact: that fact says it (0.19.0)
                    other = self._merged_head(content, fact_id)
                if other is not None:
                    raise DuplicateContentError(int(other["fact_id"]))
                cur = self._current_version(fact_id)
                source = self._with_origin(source, cur)
                self._replace_current(fact_id, "corrected", content, category or row["category"],
                                      row["tags"] if tags is None else tags, valid_from=cur["valid_from"],
                                      source=source, trust=row["trust_score"])
            elif history and (tags is not None or category is not None):
                self._write("UPDATE fact_versions SET tags = COALESCE(?, tags), category = COALESCE(?, category) "
                            "WHERE fact_id = ? AND status = 'current'", (tags, category, fact_id))
            if new_claim:
                start = self.start_trust((source or {}).get("origin"))  # no history: the origin given still counts
                base = self.default_trust if start is None else start
            changes = {col: val for col, val in {
                "content": content, "tags": tags, "category": category,
                "trust_score": _clamp_trust(base + (trust_delta or 0.0)) if trust_delta is not None or new_claim else None,
                "helpful_count": 0 if new_claim else None, "retrieval_count": 0 if new_claim else None,
            }.items() if val is not None}
            if new_claim:
                self._drop_confirmations(fact_id)
            assignments = ", ".join((["updated_at = CURRENT_TIMESTAMP"] if touch else []) + [f"{col} = ?" for col in changes])
            if not assignments and not entities:
                return True
            try:
                if assignments:  # entities alone (touch=False) change no column (review 0.17.0: they were dropped)
                    self._write(f"UPDATE facts SET {assignments} WHERE fact_id = ?", [*changes.values(), fact_id])
            except sqlite3.IntegrityError:
                raise self._duplicate(content) from None
            if content is not None:  # re-extract entities and recompute the vector; with a new text, names given replace
                # the declared ones and a declared name only the old text gave goes (_relink_fact); with the same text
                # names are added, as by entities alone
                rewritten = content != row["content"]
                self._relink_fact(fact_id, content, entities, replace_declared=rewritten,
                                  left_text=row["content"] if rewritten else None)
                self._compute_hrr_vector(fact_id, content)
                self._drop_orphan_entities()
            elif entities:  # declare who the fact is about without rewording it
                current = self._one("SELECT content FROM facts WHERE fact_id = ?", (fact_id,))["content"]
                self._link_declared(fact_id, entities)
                self._compute_hrr_vector(fact_id, current)
            self._invalidate_bank(row["category"])
            if category is not None and category != row["category"]:
                self._invalidate_bank(category)
            return True

    def remove_fact(self, fact_id: int) -> bool:
        """0.7.0 `remove`, now `forget` (history.forget_fact): the fact, its history, links and entities left without
        facts are deleted for good and a tombstone is kept. Returns True if the fact existed (current or ended)."""
        if not self.history_ready():  # a store no writer of 0.8.0 has migrated: the plain delete
            with self._tx():
                row = self._one("SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,))
                if row is None:
                    return False
                self._delete_fact_row(fact_id, row["category"])
                return True
        try:
            return not self.forget_fact(fact_id).get("already")
        except NotFound:  # only that: a KeyError from a fault is not "there was no such fact" (0.17.8)
            return False

    # -- rows of the current projection (inside a transaction) ---------------------------------------------------

    def _delete_fact_row(self, fact_id: int, category: str | None) -> None:
        if self._entry.get("migrated"):  # the vector goes with the fact (forget, end, remove)
            self._write("DELETE FROM fact_embeddings WHERE fact_id = ?", (fact_id,))
        self._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
        self._write("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
        self._drop_orphan_entities()
        self._invalidate_bank(category)

    def _rewrite_fact(self, fact_id: int, content: str, category, tags, *, entities=(), old_category=None,
                      reset_trust: bool = False, trust: float | None = None, restore=None) -> None:
        """The current text of a fact changed (change / correct / revert): the row, its entity links (_relink_fact:
        names given with the new text replace the declared ones; without names those stay, but for one only the old
        text gave; *restore*: a revert's names of the version it brings back), vector and banks follow. reset_trust: a
        changed fact starts over (default trust, no counts)."""
        old = self._one("SELECT content FROM facts WHERE fact_id = ?", (fact_id,))
        sets = ["content = ?", "category = ?", "tags = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list = [content, category or "general", tags or ""]
        if reset_trust:  # *trust*, when given, is where it starts over (its origin's, 0.15.0)
            sets += ["trust_score = ?", "helpful_count = 0", "retrieval_count = 0"]
            params.append(self.default_trust if trust is None else _clamp_trust(trust))
        elif trust is not None:
            sets.append("trust_score = ?")
            params.append(_clamp_trust(trust))
        try:
            self._write(f"UPDATE facts SET {', '.join(sets)} WHERE fact_id = ?", [*params, fact_id])
        except sqlite3.IntegrityError:
            raise self._duplicate(content) from None
        self._relink_fact(fact_id, content, entities, replace_declared=True,
                          left_text=old["content"] if old is not None else None, restore=restore)
        self._compute_hrr_vector(fact_id, content)
        self._drop_orphan_entities()
        self._invalidate_bank(old_category)
        if category != old_category:
            self._invalidate_bank(category)

    def _duplicate(self, content: str) -> DuplicateContentError:
        """The error for a write that would give a second fact *content* (the UNIQUE column refused it)."""
        other = self._one("SELECT fact_id FROM facts WHERE content = ?", (content,))
        return DuplicateContentError(int(other["fact_id"]) if other else -1)

    def _relink_fact(self, fact_id: int, content: str, entities=(), *, replace_declared: bool = False,
                     left_text: str | None = None, restore=None) -> None:
        """Link a fact to its entities again from *content*, *entities* declared with it. Names someone declared on it
        before stay (a rewording without names: "my teacher" stays the person someone named), unless
        *replace_declared* and *entities* hold a usable name: a new text given with names says who it is about, and the
        earlier declarations go (they used to stay: "Leo's teacher is Ben Cole." changed into "... is Omar Reid."
        with entities [Leo, Omar Reid] was still found by probe "Ben Cole"). Names the new text itself gives are
        linked either way. *left_text*: the text the fact stops saying (change, correct, update, revert): without
        such names, a declared name it named and *content* does not goes with it, and one no text names stays; named
        is read as entity identity reads names (case folding, accents, aliases: _named_in). *restore*: the names a
        revert brings back (version_names), declared instead of the current ones."""
        rows = self._conn.execute(
            "SELECT e.entity_id, e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
            "WHERE fe.fact_id = ? AND e.entity_type = 'declared'", (fact_id,)).fetchall()
        if restore is not None:
            rows = [(None, name) for name in restore]
        elif replace_declared and any(_valid_entity(_clean_given(n), given=True) for n in entities or ()):
            rows = []  # a list of blanks or junk replaces nothing: those names are reported as not linked
        elif left_text is not None and rows:
            ids = [r[0] for r in rows]
            undone = self._named_in(left_text, ids) - self._named_in(content, ids)
            rows = [r for r in rows if r[0] not in undone]
        self._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
        self._link_entities(fact_id, content, [*(r[1] for r in rows), *(entities or ())])

    def _insert_fact_row(self, fact_id: int, version, *, trust: float, entities=()) -> None:
        """Put an ended fact back into the current projection under its own id (resume / revert of an end)."""
        first = self._one("SELECT MIN(recorded_at) FROM fact_versions WHERE fact_id = ?", (fact_id,))[0]
        try:
            self._write("INSERT INTO facts (fact_id, content, category, tags, trust_score, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)", (fact_id, version["content"], version["category"], version["tags"],
                                                      _clamp_trust(trust), first or self._now()))
        except sqlite3.IntegrityError:
            raise self._duplicate(version["content"]) from None
        self._link_entities(fact_id, version["content"], entities)
        self._compute_hrr_vector(fact_id, version["content"])
        self._invalidate_bank(version["category"])

    # -- vectors of search by meaning (embeddings.py) -------------------------------------------------------------

    def put_vectors(self, model: str, rows) -> int:
        """Store vectors made by *model* from (fact_id, text hash, blob); a fact that is gone, or whose text changed
        while its vector was being made, is skipped. Returns how many were stored."""
        stored = 0
        with self._tx():
            for fact_id, digest, blob in rows:
                row = self._one("SELECT content FROM facts WHERE fact_id = ?", (fact_id,))
                if row is None or text_hash(row["content"]) != digest:
                    continue
                self._write("INSERT INTO fact_embeddings (fact_id, model, text_hash, vector, revision) "
                            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(fact_id) DO UPDATE SET model = excluded.model, "
                            "text_hash = excluded.text_hash, vector = excluded.vector, revision = excluded.revision, "
                            "created_at = CURRENT_TIMESTAMP", (fact_id, model, digest, blob, os.urandom(8).hex()))
                stored += 1
        return stored

    def _drop_orphan_vectors(self) -> None:
        """Vectors of facts that are gone (an older version deleted them: it knows no vector table)."""
        with self._tx():
            self._write("DELETE FROM fact_embeddings WHERE fact_id NOT IN (SELECT fact_id FROM facts)")

    def drop_vectors(self) -> int:
        """Delete every vector (search by meaning is off: nothing derived from the facts by an outside model stays in
        the file). Returns how many there were."""
        with self._lock:
            if not self._entry.get("migrated") or not self._one("SELECT 1 FROM fact_embeddings LIMIT 1"):
                return 0
            with self._tx():
                return max(self._write("DELETE FROM fact_embeddings").rowcount, 0)

    def list_facts(self, category: str | None = None, min_trust: float = 0.0, limit: int = 50) -> list[dict]:
        """Browse facts ordered by trust_score descending, optionally filtered by category / min trust."""
        category_clause = "AND category = ? " if category is not None else ""
        params = [min_trust] + ([category] if category is not None else []) + [limit]
        sql = ("SELECT fact_id, content, category, tags, trust_score, retrieval_count, helpful_count, "
               f"created_at, updated_at FROM facts WHERE trust_score >= ? {category_clause}"
               "ORDER BY trust_score DESC LIMIT ?")
        return [dict(r) for r in self.read(sql, params)]

    def count_facts(self) -> int:
        return int(self.read("SELECT COUNT(*) FROM facts")[0][0])

    def facts_version(self) -> tuple:
        """A key that changes whenever a fact's text may have: a write on this connection other than the recall
        counts (total_changes less those), a rollback (generation) or a commit by another connection (PRAGMA
        data_version). For caches of what the facts hold: the recall counts are written every few searches."""
        with self._lock:
            return (self._entry.get("generation", 0), self._conn.total_changes - self._entry.get("counted", 0),
                    self._one("PRAGMA data_version")[0])

    def fact_tags(self, fact_id: int) -> str | None:
        """The tags column of a current fact ("" when empty), None when no current fact has that id."""
        rows = self.read("SELECT tags FROM facts WHERE fact_id = ?", (fact_id,))
        return (rows[0]["tags"] or "") if rows else None

    # A fact has keywords when its tags start with the provider's marker line, or have it after a line break.
    _WITHOUT_KEYWORDS = "COALESCE(tags, '') NOT LIKE ? AND COALESCE(tags, '') NOT LIKE ?"

    def facts_without_keywords(self, mark: str, limit: int) -> list[dict]:
        """Current facts with no *mark* line in their tags, the most recalled first."""
        return self.read(f"SELECT fact_id, content FROM facts WHERE {self._WITHOUT_KEYWORDS} "
                         "ORDER BY retrieval_count DESC, fact_id LIMIT ?", (mark + "%", "%\n" + mark + "%", limit))

    def count_without_keywords(self, mark: str) -> int:
        """How many current facts have no *mark* line in their tags."""
        return int(self.read(f"SELECT COUNT(*) FROM facts WHERE {self._WITHOUT_KEYWORDS}",
                             (mark + "%", "%\n" + mark + "%"))[0][0])

    def record_retrievals(self, fact_ids) -> int:
        """retrieval_count += 1 for facts handed to the model; returns the number of rows touched.

        Only the counter changes: ``updated_at`` drives temporal decay and recall is not an edit (and the FTS
        trigger ignores the column). Only integer ids count (``True`` and ``1.9`` do not); duplicates count once."""
        ids = sorted({int(raw) for raw in fact_ids or ()
                      if (isinstance(raw, int) and not isinstance(raw, bool)) or (isinstance(raw, str) and raw.isdigit())})
        ids = ids[:_MAX_COUNTED_IDS]
        if not ids:
            return 0
        with self._lock:
            before = self._conn.total_changes
            cur = self._write(f"UPDATE facts SET retrieval_count = retrieval_count + 1 "
                              f"WHERE fact_id IN ({','.join('?' * len(ids))})", ids)
            self._entry["counted"] = self._entry.get("counted", 0) + self._conn.total_changes - before
            return max(cur.rowcount, 0)

    def add_retrieval_counts(self, counts: dict) -> bool:
        """Add buffered recall counts ({fact_id: n}) in one transaction without waiting on another process's write
        lock (busy timeout 0 for this write only). False when the database was busy: the caller keeps the counts."""
        rows = [(int(n), int(i)) for i, n in counts.items()
                if isinstance(i, int) and not isinstance(i, bool) and isinstance(n, int) and n > 0]
        if not rows:
            return True
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout = 0")
            try:
                before = self._conn.total_changes
                with self._tx():
                    self._conn.executemany("UPDATE facts SET retrieval_count = retrieval_count + ? WHERE fact_id = ?", rows)
                self._entry["counted"] = self._entry.get("counted", 0) + self._conn.total_changes - before
                return True
            except sqlite3.OperationalError as exc:
                logger.debug("holographic: recall counts deferred (%s)", error_kind(exc))
                return False
            finally:
                self._conn.execute("PRAGMA busy_timeout = 10000")

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Adjust trust asymmetrically: helpful -> +0.05 and helpful_count += 1; unhelpful -> -0.10.
        Returns {fact_id, old_trust, new_trust, helpful_count}. Raises KeyError if fact_id is unknown.

        ``updated_at`` stays: a vote on a fact is not a change of it (until 0.22.2 it was stamped, and an old fact
        marked unhelpful came out "younger" for temporal decay and "recently changed")."""
        with self._tx():
            row = self._one("SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                raise NotFound(f"fact_id {fact_id} not found")
            old_trust: float = row["trust_score"]
            new_trust = _clamp_trust(old_trust + (_HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA))
            increment = 1 if helpful else 0
            self._write("UPDATE facts SET trust_score = ?, helpful_count = helpful_count + ? WHERE fact_id = ?",
                        (new_trust, increment, fact_id))
            return {"fact_id": fact_id, "old_trust": old_trust, "new_trust": new_trust, "helpful_count": row["helpful_count"] + increment}

    def entity_rules(self) -> int:
        """The extraction rules the store's entity index was built with (1: before 0.16.0, which recorded none)."""
        try:
            return int(self._meta("entity_rules") or 1)
        except ValueError:
            return 1

    def rebuild_entity_index(self) -> dict:
        """Link every fact to its entities again under this release's extraction rules: names an older release
        guessed that these rules do not, and links only they gave, go. What someone named stays as it is: declared
        entities, entities with an alias someone gave (or of unknown origin), and their links. First pass: the facts
        in the order they were stored, each read as adding it would read it — so "X aka Y" makes Y an alias of X only
        while no earlier fact has named Y and nobody gave Y; a text never merges a name already in use into another
        (review 0.16.0: a stored "Mallory Jones aka Alice Smith" must not take over the facts about Alice Smith).
        Second pass: the known names each fact mentions, against the whole index the first left, so an earlier fact
        gets a name a later one brought. Entities that stay keep their id and age; vectors are made again only for
        facts whose entities changed. Records the rules. ValueError when a newer release indexed the store."""
        self._require_writer()
        with self._tx():
            if self.entity_rules() > _ENTITY_RULES:
                raise ValueError(f"a newer release indexed this store's entities (rules {self.entity_rules()}, this one "
                                 f"knows {_ENTITY_RULES}); it is left as it is")
            before = self._one("SELECT COUNT(*) FROM entities")[0]
            given = f" WHERE {_GIVEN_ALIAS}" if self._alias_sources() else ""
            protected = {r[0] for r in self._conn.execute("SELECT entity_id FROM entities WHERE entity_type = 'declared'")}
            protected |= {r[0] for r in self._conn.execute(f"SELECT entity_id FROM entity_aliases{given}")}
            old = {(r[0], r[1]) for r in self._conn.execute("SELECT fact_id, entity_id FROM fact_entities")}
            # A name two facts held before the rebuild is no "Y" an "X aka Y" may take: fact ids stay through change
            # and update, so "earlier id" is not "said earlier" (review 0.16.1). Not kept by this alone: junk two
            # facts held still goes when the rules no longer guess it.
            held = Counter(entity_id for _, entity_id in old)
            established = {entity_id for entity_id, n in held.items() if n >= 2}
            facts = self._conn.execute("SELECT fact_id, content FROM facts ORDER BY fact_id").fetchall()
            first: set[tuple[int, int]] = set()
            named: set[int] = set()  # entities an earlier fact of this pass named
            for f in facts:
                for left, right in self._aka_pairs(f["content"]):
                    # Y left over from the old index and not named yet: the alias adding this fact would have made
                    if not set(self.entity_ids(right)) & (protected | named | established):
                        try:
                            self.add_alias(left, right, source="aka")
                        except ValueError:  # Y is another entity's alias already
                            pass
                ids = self._text_entity_ids(f["content"])
                named |= ids
                first |= {(f["fact_id"], e) for e in ids}
            current = {(r[0], r[1]) for r in self._conn.execute("SELECT fact_id, entity_id FROM fact_entities")}
            kept = {link for link in current if link[1] in protected}
            for link in sorted(current - first - kept):  # merges above moved links: diff against the table, not *old*
                self._write("DELETE FROM fact_entities WHERE fact_id = ? AND entity_id = ?", link)
            for link in sorted(first - current):
                self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", link)
            self._drop_orphan_entities(keep_declared=True)  # entities only a dropped link held
            self._bump()
            second = {(f["fact_id"], e) for f in facts for e in self.mentioned_entity_ids(f["content"])}
            for link in sorted(second - first - kept):
                self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", link)
            new = {(r[0], r[1]) for r in self._conn.execute("SELECT fact_id, entity_id FROM fact_entities")}
            vectors = self._reencode({fact_id for fact_id, _ in old ^ new})
            self._set_meta("entity_rules", str(_ENTITY_RULES))
            self._bump()
            return {"facts": len(facts), "entities_before": before,
                    "entities_after": self._one("SELECT COUNT(*) FROM entities")[0],
                    "links_removed": len(old - new), "links_added": len(new - old), "vectors": vectors}

    def entity_index_state(self) -> str:
        """"current", or what a store indexed under older rules (too large to rebuild on open) needs."""
        rules = self.entity_rules()
        if rules >= _ENTITY_RULES:
            return "current"
        return f"built with extraction rules {rules} of {_ENTITY_RULES}: run fact_store rebuild_entities"

    def _maybe_rebuild_entities(self) -> None:
        """On a writer's first open: a store indexed under older rules is rebuilt, up to _AUTO_REBUILD_MAX_FACTS facts
        (the write lock is held throughout, and other processes wait at most 10 s for it); a larger one says so in
        stats and is rebuilt on request. A failure leaves the old index and is logged: the store works as before."""
        if self.entity_rules() >= _ENTITY_RULES:
            return
        facts, chars = self._one("SELECT COUNT(*), COALESCE(SUM(LENGTH(content)), 0) FROM facts")
        if facts > _AUTO_REBUILD_MAX_FACTS or chars > _AUTO_REBUILD_MAX_CHARS:
            logger.info("holographic: %d facts, %d characters; the entity index is rebuilt on request "
                        "(fact_store rebuild_entities)", facts, chars)
            return
        try:
            result = self.rebuild_entity_index()
        except Exception as exc:  # noqa: BLE001 - the old index works; never fail an open over it
            logger.warning("holographic: entity index not rebuilt (%s); the old one stays", exc)
            return
        if result["links_removed"] or result["links_added"]:
            logger.info("holographic: entity index rebuilt under rules %d: %d -> %d entities, %d links removed, %d added",
                        _ENTITY_RULES, result["entities_before"], result["entities_after"], result["links_removed"],
                        result["links_added"])

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store the HRR vector for a fact (linked entities as roles). No-op without numpy."""
        if not self._hrr_available:
            return
        entities = [row["name"] for row in self._conn.execute(_ENTITY_NAMES_SQL, (fact_id,)).fetchall()]
        blob = hrr.phases_to_bytes(hrr.encode_fact(content, entities, self.hrr_dim))
        self._write("UPDATE facts SET hrr_vector = ? WHERE fact_id = ?", (blob, fact_id))

    def _invalidate_bank(self, category: str | None) -> None:
        """Drop the category's memory bank. Retrieval no longer reads banks; deleting (instead of rebuilding the
        whole category on every write) keeps a downgrade safe: an older version that finds no bank falls back to
        the per-fact vectors, and rebuilds the bank itself on its next write."""
        if category is not None:
            self._write("DELETE FROM memory_banks WHERE bank_name = ?", (f"cat:{category}",))

    @classmethod
    def release_all_under(cls, directory: "str | Path") -> int:
        """Force-close every shared connection whose database lives under ``directory``; returns the count.
        close() is refcount-driven, so a live holder (e.g. an agent's provider) keeps a profile's SQLite handle
        open, which on Windows makes rmtree of the profile fail. The directory is going away, so later use by a
        stale holder is expected to fail.

        That is exactly what a profile delete must break on Windows: the desktop's main ``serve`` process
        opens ``memory_store.db`` for every known profile, and ``rmtree`` of the profile directory fails
        with ``WinError 32`` while any of those handles is open (#88347). In a process that holds none (e.g.
        the CLI deleting from outside serve) this is a harmless no-op returning 0.
        """
        root = os.path.normcase(str(Path(directory).expanduser().resolve())) + os.sep
        with cls._shared_guard:
            doomed = [cls._shared.pop(key) for key in list(cls._shared) if os.path.normcase(key).startswith(root)]
            for entry in doomed:
                try:
                    with entry["lock"]:
                        entry["conn"].close()
                except Exception:  # noqa: BLE001 - an already-closed/broken connection must not abort releasing siblings
                    pass
        return len(doomed)

    def close(self) -> None:
        """Release this instance's reference; the connection closes with the last holder. Idempotent."""
        with MemoryStore._shared_guard:
            entry = getattr(self, "_entry", None)
            if entry is None:
                return
            entry["refs"] -= 1
            if entry["refs"] <= 0:
                try:
                    entry["conn"].close()
                finally:
                    # Pop only OUR entry: after release_all_under() a same-path store may have
                    # registered a FRESH entry under this key; a stale late close() must not evict it.
                    # See #88347.
                    if MemoryStore._shared.get(self._key) is entry:
                        MemoryStore._shared.pop(self._key, None)
            self._entry = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
