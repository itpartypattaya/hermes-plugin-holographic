"""Entities and aliases: names resolved to entities, facts linked to them, aliases someone gave or a fact's
"X aka Y" gave, and names deleted or renamed (tombstones). Mixed into MemoryStore, as history.py is; the rules
of what a name is are in entity_names.py."""

import contextlib
import itertools
import logging
import re
import threading
import unicodedata

from .entity_names import (_AKA_MAX, _AKA_WINDOW, _ENTITY_MAX_CHARS, _QUOTED_MAX_CHARS, _QUOTED_MAX_WORDS, _RE_AKA,
                           _RE_QUOTED, _aka_side, _capitalized_phrases, _cased_word_in, _clean_given, _entity_key_0160,
                           _found_between_cuts, _inline_gap, _legacy_aliases, _mention_stretches, _mixed, _name_shaped,
                           _outside_plain_quotes, _straight_quoted, _uncased, _valid_entity, _without_code, entity_key, fold)
from .history import NotFound
from .text import STOPWORDS, UNSEGMENTED, analyze, is_chinese, name_keys, zh_word_cuts
from .text import fold as text_fold

logger = logging.getLogger(__name__)

_RE_WIKI_PAGE = re.compile(r"Wiki page:\s*([^\s|]+)")
_WIKI_STRUCTURAL_SLUGS = frozenset({
    "index", "schema", "log", "readme", "wiki index", "wiki log", "overview", "changelog", "notes", "todo",
})

# Where an alias came from (entity_aliases.source, 0.16.0): 'user' (alias, rename, a merge someone asked for), 'aka'
# ("X aka Y" in a fact's text: it goes with the last fact that says it) or 'legacy' (written before 0.16.0 or by an
# older release since: kept like a given one, and dropped only when a forgotten fact said it as "X aka Y").
_GIVEN_ALIAS = "COALESCE(source, 'legacy') != 'aka'"
# What declared names split off aliases a fact's text gave during one operation (MemoryStore.collect_splits, 0.17.0):
# per thread, open only while the provider runs one tool call.
_SPLITS = threading.local()
_DELETE_MAX = 100  # names in one delete_entity call
_MAX_COUNTED_IDS = 500  # record_retrievals batch cap (far under SQLite's bound-parameter limit)


def _key_words(text: str) -> list[str]:
    """The words of *text* as entity identity compares names (entity_key), accents folded: what _named_in matches."""
    return [text_fold(w) for w in re.findall(r"\w+", entity_key(text))]


def _in_a_row(words: list[str], name: list[str]) -> bool:
    n = len(name)
    return n > 0 and any(words[i:i + n] == name for i in range(len(words) - n + 1))


class EntityMixin:
    """The entity half of MemoryStore. Expects _conn, _lock, _entry, _tx, _one, _write, read, _set_meta,
    _hrr_available, _compute_hrr_vector and _relink_fact of MemoryStore, and digest and scrub of HistoryMixin.
    rebuild_entity_index, entity_index_state and _maybe_rebuild_entities stay in store.py with the limits they read
    (_ENTITY_RULES, _AUTO_REBUILD_*)."""

    def _rekey_entities(self) -> None:
        """Keys and alias keys written before entity_key changed (0.16.1: ".NET" was "net", "‘Alice’" was "‘alice’") are
        moved to today's; an entity whose new key another one holds is merged into it, an alias row whose new key is
        taken goes (the other row already says it)."""
        for r in self._conn.execute("SELECT entity_id, name, name_key, entity_type FROM entities "
                                    "WHERE name_key IS NOT NULL ORDER BY entity_id").fetchall():
            key = entity_key(r["name"])
            if not key or key == r["name_key"]:
                continue
            owner = self._one("SELECT entity_id FROM entities WHERE name_key = ? AND entity_id != ?", (key, r["entity_id"]))
            if owner is None:
                self._conn.execute("UPDATE entities SET name_key = ? WHERE entity_id = ?", (key, r["entity_id"]))
            else:
                self._merge_into(r["entity_id"], owner[0], declared=r["entity_type"] == "declared")
        for r in self._conn.execute("SELECT alias_key, entity_id, alias FROM entity_aliases").fetchall():
            key = entity_key(r["alias"])
            if not key or key == r["alias_key"]:
                continue
            if self._one("SELECT 1 FROM entity_aliases WHERE alias_key = ?", (key,)) is None:
                self._conn.execute("UPDATE entity_aliases SET alias_key = ? WHERE alias_key = ?", (key, r["alias_key"]))
            else:
                self._conn.execute("DELETE FROM entity_aliases WHERE alias_key = ?", (r["alias_key"],))

    def _reconcile_entity_keys(self) -> None:
        """Fill missing name keys; merge entities whose keys collide into the oldest of them."""
        rows = self._conn.execute("SELECT entity_id, name, name_key, entity_type FROM entities ORDER BY entity_id").fetchall()
        if all(r["name_key"] for r in rows):
            return
        canonical: dict[str, int] = {}
        for r in rows:
            if r["name_key"]:
                canonical.setdefault(r["name_key"], r["entity_id"])
        for r in rows:
            if r["name_key"]:
                continue
            key = entity_key(r["name"]) or f"#{r['entity_id']}"
            keep = canonical.get(key)
            if keep is None:
                canonical[key] = r["entity_id"]
                self._conn.execute("UPDATE entities SET name_key = ? WHERE entity_id = ?", (key, r["entity_id"]))
            else:
                self._merge_into(r["entity_id"], keep, declared=r["entity_type"] == "declared")

    def _merge_into(self, source: int, target: int, *, declared: bool | None = None, alias_source: str = "user") -> None:
        """Make entity *source* part of *target* (inside a transaction): links, aliases (both representations) and
        the declared flag move, *source*'s own name stays findable as an alias (from *alias_source*: "aka" when a
        fact's "X aka Y" merged them, 0.16.0), the vectors of every affected fact are re-encoded with the surviving
        name, and category banks are dropped."""
        src = self._one("SELECT name, name_key, aliases, entity_type FROM entities WHERE entity_id = ?", (source,))
        tgt = self._one("SELECT name, name_key, aliases FROM entities WHERE entity_id = ?", (target,))
        if src is None or tgt is None or source == target:
            return
        affected = [r[0] for r in self._conn.execute("SELECT fact_id FROM fact_entities WHERE entity_id IN (?, ?)", (source, target))]
        self._repoint_relations(source, target)  # the graph follows the merge: edges of *source* become *target*'s
        self._conn.execute("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) SELECT fact_id, ? FROM fact_entities "
                           "WHERE entity_id = ?", (target, source))
        self._conn.execute("DELETE FROM fact_entities WHERE entity_id = ?", (source,))
        self._conn.execute("UPDATE OR IGNORE entity_aliases SET entity_id = ? WHERE entity_id = ?", (target, source))
        self._conn.execute("DELETE FROM entity_aliases WHERE entity_id = ?", (source,))
        target_key = tgt["name_key"] or entity_key(tgt["name"])
        moved = [a for a in (src["aliases"] or "").split(",") if a.strip()]
        if entity_key(src["name"]) != target_key:
            moved.append(src["name"])
            if self._alias_sources():
                self._conn.execute("INSERT OR IGNORE INTO entity_aliases (alias_key, entity_id, alias, source) "
                                   "VALUES (?, ?, ?, ?)", (entity_key(src["name"]), target, src["name"], alias_source))
            else:
                self._conn.execute("INSERT OR IGNORE INTO entity_aliases (alias_key, entity_id, alias) VALUES (?, ?, ?)",
                                   (entity_key(src["name"]), target, src["name"]))
        legacy = [a for a in (tgt["aliases"] or "").split(",") if a.strip()]
        known = {entity_key(a) for a in legacy} | {target_key}
        legacy += [a.strip() for a in moved if entity_key(a) not in known]
        self._conn.execute("UPDATE entities SET aliases = ? WHERE entity_id = ?", (_legacy_aliases(legacy), target))
        if declared if declared is not None else src["entity_type"] == "declared":
            self._conn.execute("UPDATE entities SET entity_type = 'declared' WHERE entity_id = ?", (target,))
        self._conn.execute("DELETE FROM entities WHERE entity_id = ?", (source,))
        self._reencode(affected)  # without numpy: emptied, not left with the old name's role (review 0.16.0)
        self._bump()

    def _bump(self) -> None:
        """Invalidate per-connection caches (the known-entity index) after an entity change or a rollback."""
        self._entry["generation"] = self._entry.get("generation", 0) + 1

    def entity_ids(self, name: str) -> list[int]:
        """Entities whose name or an alias has the identity key of *name* (see entity_key). Entities an older
        version created since this store was opened have no key yet; they are matched by casefolded name."""
        key = entity_key(name or "")
        if not key:
            return []
        with self._lock:
            if not self._keyed():  # a file no writer of 0.4.0 or later has migrated: names only
                return sorted({r[0] for r in self._conn.execute("SELECT entity_id FROM entities WHERE hfold(name) = ?", (fold(name),))})
            ids = {r[0] for r in self._conn.execute(
                "SELECT entity_id FROM entities WHERE name_key = ? "
                "UNION SELECT ea.entity_id FROM entity_aliases ea JOIN entities e ON e.entity_id = ea.entity_id WHERE ea.alias_key = ? "
                "UNION SELECT entity_id FROM entities WHERE name_key IS NULL AND hfold(name) = ?", (key, key, fold(name)))}
        return sorted(ids)

    def add_alias(self, name: str, alias: str, *, source: str = "user") -> dict:
        """Make *alias* another name of the entity *name* (created if needed). If *alias* is already an entity of its
        own, it is merged into *name*: links and aliases move, and its name becomes an alias; *name* stays the name
        (rename_entity changes it). *source* (0.16.0): "user" for the alias action, "aka" for "X aka Y" found in a
        fact's text. Only an alias someone gave makes the entity established, keeps it without facts and brings back
        a name delete_entity removed."""
        if not isinstance(name, str) or not isinstance(alias, str):
            raise ValueError("an alias needs two names")
        if source == "user":
            name, alias = _clean_given(name), _clean_given(alias)
        key, alias_key = entity_key(name), entity_key(alias)
        if not key or not alias_key or key == alias_key:
            raise ValueError("an alias needs two different non-empty names")
        for side in (name, alias):  # a name someone gives passes what every given name does (review 0.16.0)
            if not _valid_entity(side.strip(), given=True):
                raise ValueError(f"'{side.strip()[:80]}' cannot be a name (2-{_ENTITY_MAX_CHARS} characters, mostly "
                                 "letters; no paths, URLs, key=value, commas or flags)")
        with self._tx():
            if source == "user":
                textual = self._text_alias(name)  # alias {entity: <a name only a text gave>} (0.17.0)
                if textual is not None:
                    raise ValueError(self._text_alias_refusal(name, textual))
                self._lift_tombstones(name, alias)
            target = self._resolve_entity(name.strip())
            own = self._one("SELECT name FROM entities WHERE entity_id = ?", (target,))
            if alias_key == entity_key(own["name"]):  # alias("Bob", "Alice") with Bob already Alice's: no-op (review)
                return {"entity_id": target, "alias": alias.strip(), "merged": []}
            owner = self._one("SELECT ea.entity_id, e.name FROM entity_aliases ea JOIN entities e ON e.entity_id = ea.entity_id "
                              "WHERE ea.alias_key = ?", (alias_key,))
            was = self._alias_row(alias) if source == "user" else None
            if owner is not None and owner["entity_id"] != target:
                if was is not None and was["source"] == "aka":
                    raise ValueError(f"'{alias.strip()}' is a name only a fact's text gave '{owner['name']}' (\"X aka "
                                     f"Y\"), and nobody confirmed it. If it belongs to '{name.strip()}' instead: split "
                                     f"{{entity: '{alias.strip()}'}} first, then alias")
                raise ValueError(f"'{alias.strip()}' is already an alias of '{owner['name']}'; merge the two entities "
                                 f"(alias '{name.strip()}' -> '{owner['name']}') if they are the same")
            confirmed = was is not None and was["source"] == "aka" and was["entity_id"] == target
            other = [r[0] for r in self._conn.execute("SELECT entity_id FROM entities WHERE name_key = ? AND entity_id != ?",
                                                      (alias_key, target))]
            for merged in other:
                self._merge_into(merged, target, alias_source=source)
            self._put_alias(alias_key, target, alias.strip(), source)
            # Older versions resolve aliases from the comma-separated column: keep it in step.
            row = self._one("SELECT aliases FROM entities WHERE entity_id = ?", (target,))
            legacy = [a for a in (row["aliases"] or "").split(",") if a.strip()]
            if alias_key not in {entity_key(a) for a in legacy}:
                self._conn.execute("UPDATE entities SET aliases = ? WHERE entity_id = ?", (_legacy_aliases(legacy + [alias.strip()]), target))
            self._bump()
            result = {"entity_id": target, "alias": alias.strip(), "merged": other}
            if confirmed:  # a name a fact's text gave, now confirmed by someone (0.17.0)
                result["confirmed"] = True
            return result

    def _put_alias(self, alias_key: str, entity_id: int, alias: str, source: str = "user") -> None:
        """Insert or move an alias row. An alias someone gave stays theirs when the same name is found in text later."""
        if self._entry.get("migrated"):  # the source column is added by this version's migration
            self._conn.execute("INSERT INTO entity_aliases (alias_key, entity_id, alias, source) VALUES (?, ?, ?, ?) "
                               "ON CONFLICT(alias_key) DO UPDATE SET entity_id = excluded.entity_id, alias = excluded.alias, "
                               "source = CASE WHEN entity_aliases.source = 'user' THEN 'user' ELSE excluded.source END",
                               (alias_key, entity_id, alias, source))
        else:
            self._conn.execute("INSERT INTO entity_aliases (alias_key, entity_id, alias) VALUES (?, ?, ?) "
                               "ON CONFLICT(alias_key) DO UPDATE SET entity_id = excluded.entity_id, alias = excluded.alias",
                               (alias_key, entity_id, alias))

    def _keyed(self) -> bool:
        """Whether the file has the entity keys of 0.4.0 (name_key, and entity_aliases kept in step): a writer of 0.4.0
        or later migrated it at some point. Reads use them in every context, also one that does not migrate — a
        subagent found no entity by an alias before (review 0.16.0); writes still wait for this process's migration."""
        if self._entry.get("migrated"):
            return True
        known = self._entry.get("keyed")
        if known is None:
            with self._lock:
                known = self._entry["keyed"] = "name_key" in {
                    r[1] for r in self._conn.execute("PRAGMA table_info(entities)")}
        return known

    def _alias_sources(self) -> bool:
        """Whether entity_aliases says where each alias came from (a store this version migrated)."""
        known = self._entry.get("alias_source")
        if known is None:
            with self._lock:
                known = self._entry["alias_source"] = "source" in {
                    r[1] for r in self._conn.execute("PRAGMA table_info(entity_aliases)")}
        return known

    def _alias_row(self, name: str):
        """The entity_aliases row of *name*'s key, with where it came from ('legacy' when the file does not say) and
        the name of the entity it belongs to; None when no alias has that key or the file has no keys (0.17.0)."""
        key = entity_key(name or "")
        if not key or not self._keyed():
            return None
        source = "COALESCE(ea.source, 'legacy')" if self._alias_sources() else "'legacy'"
        with self._lock:
            return self._one(f"SELECT ea.alias_key, ea.entity_id, ea.alias, {source} AS source, e.name AS owner "
                             "FROM entity_aliases ea JOIN entities e ON e.entity_id = ea.entity_id WHERE ea.alias_key = ?",
                             (key,))

    def _text_alias(self, name: str):
        """The alias row when *name* names one entity only through an alias that "X aka Y" in a fact's text gave and
        nobody confirmed, else None (0.17.0). Such a name is a hint for text, never the target of an action someone
        takes with a name: a declaration splits it off (_link_declared), alias, rename and delete_entity refuse it."""
        row = self._alias_row(name)
        if row is None or row["source"] != "aka" or self.entity_ids(name) != [row["entity_id"]]:
            return None
        return row

    @staticmethod
    def _text_alias_refusal(name: str, row) -> str:
        return (f"'{name.strip()}' is a name only a fact's text gave '{row['owner']}' (\"X aka Y\"), and nobody "
                f"confirmed it. If it is the same one: alias {{entity: '{row['owner']}', content: '{row['alias']}'}}; "
                f"if they are two: split {{entity: '{row['alias']}'}}; then try again")

    @contextlib.contextmanager
    def collect_splits(self):
        """{declared name: what split_alias answers} for every alias a fact's text gave that a name declared during
        this operation split off (0.17.0). Read it only after the operation succeeded; nested in one thread, the
        outer one collects."""
        outer = getattr(_SPLITS, "found", None)
        if outer is not None:
            yield outer
            return
        found: dict = {}
        _SPLITS.found, _SPLITS.pending = found, []
        try:
            yield found
        finally:
            _SPLITS.found, _SPLITS.pending = None, []

    def _drop_alias_row(self, alias_key: str, entity_id: int) -> None:
        """Remove one alias from both representations (the entity_aliases row and the legacy column)."""
        self._write("DELETE FROM entity_aliases WHERE alias_key = ? AND entity_id = ?", (alias_key, entity_id))
        row = self._one("SELECT aliases FROM entities WHERE entity_id = ?", (entity_id,))
        if row is not None and row["aliases"]:
            kept = [a for a in row["aliases"].split(",") if a.strip() and entity_key(a) != alias_key]
            self._write("UPDATE entities SET aliases = ? WHERE entity_id = ?", (_legacy_aliases(kept), entity_id))

    def _detach_alias(self, row, *, declaring: int | None = None) -> dict:
        """Make the alias *row* (see _alias_row) an entity of its own, inside the caller's transaction (0.17.0).

        1. The alias leaves the entity it belonged to (both representations) and becomes an entity (declared).
        2. Aliases "X aka Y" gave that no current fact backs any more are dropped, again and again until none is
           left: in a chain "Alice aka Bob", "Bob aka Carl", "Carl aka Dan", Carl hung on Bob and Dan on Carl; one
           pass left Dan with Alice (review 0.17.0, Codex P1.2). Only aliases are dropped here, no link yet.
        3. The facts of every entity that lost an alias are linked again from their text, the facts that say
           "X aka Y" first (so a chain is given again to whoever its text now names: "Bob aka Carl" makes Carl Bob's
           before "Carl likes tea" is read). A link to such an entity stays when the text names it; when it does not,
           a declared one stays unless the text names one of the names it lost (that name is now someone else's:
           "Bob aka Carl" no longer keeps Alice, review 0.17.0 Codex P1.1), and a fact it stays on whose text names it
           not is listed under "facts_to_check": why a declared link exists the file does not record. Not the fact
           *declaring* when it is being declared on the owner too ("They met", entities=[Alice, Bob]).
        An owner someone gave the alias to (alias {entity: X, content: Y}) was named by that person: it counts as
        declared from here, so it does not go when Y takes its facts (review 0.17.0, Fable: X vanished with its alias).
        The answer is read from the links at the end (review 0.17.0: it described a state in between); "from_dropped"
        says the old entity went with its last fact."""
        owner, alias, alias_key = row["entity_id"], row["alias"], row["alias_key"]
        lost: dict[int, set[str]] = {owner: {alias}}  # entity -> names it lost
        self._drop_alias_row(alias_key, owner)
        if row["source"] == "user":
            self._write("UPDATE entities SET entity_type = 'declared' WHERE entity_id = ?", (owner,))
        new = self._resolve_entity(alias)
        self._write("UPDATE entities SET entity_type = 'declared' WHERE entity_id = ?", (new,))
        self._bump()
        for _ in range(self._one("SELECT COUNT(*) FROM entity_aliases")[0] + 1):
            rows = self._conn.execute("SELECT alias_key, entity_id, alias FROM entity_aliases WHERE source = 'aka'").fetchall()
            backed = self._aka_backing(r["entity_id"] for r in rows) if rows else set()
            gone = [r for r in rows if (r["entity_id"], r["alias_key"]) not in backed]
            if not gone:
                break
            for r in gone:
                self._drop_alias_row(r["alias_key"], r["entity_id"])
                lost.setdefault(r["entity_id"], set()).add(r["alias"])
            self._bump()
        kinds = {r[0]: r[1] for r in self._conn.execute(
            f"SELECT entity_id, entity_type FROM entities WHERE entity_id IN ({','.join('?' * len(lost))})",
            tuple(lost)).fetchall()}
        affected = sorted({r[0] for r in self._conn.execute(
            f"SELECT fact_id FROM fact_entities WHERE entity_id IN ({','.join('?' * len(lost))})", tuple(lost)).fetchall()})
        texts = {r[0]: r[1] for r in self._conn.execute(
            f"SELECT fact_id, content FROM facts WHERE fact_id IN ({','.join('?' * len(affected))})",
            tuple(affected)).fetchall()} if affected else {}
        before = {f: {r[0] for r in self._conn.execute("SELECT entity_id FROM fact_entities WHERE fact_id = ?", (f,))}
                  for f in affected}
        check: set[int] = set()
        for fact_id in sorted(affected, key=lambda f: (not self._aka_pairs(texts.get(f) or ""), f)):
            content = texts.get(fact_id)
            if content is None:
                continue
            current = {r[0]: r[1] for r in self._conn.execute(
                "SELECT fe.entity_id, e.entity_type FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
                "WHERE fe.fact_id = ?", (fact_id,)).fetchall()}
            named = self._text_entity_ids(content) | self.mentioned_entity_ids(content)
            links = set(named) | {e for e, kind in current.items() if kind == "declared" and e not in lost}
            for entity_id in lost:
                if entity_id not in current or entity_id in named or kinds.get(entity_id) != "declared":
                    continue
                moved_to = {i for name in lost[entity_id] for i in self.entity_ids(name)} - {entity_id}
                if not named & moved_to:
                    links.add(entity_id)  # declared there, and the text says nothing about the names it lost
                    check.add(fact_id)
            self._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
            for entity_id in sorted(links):
                self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (fact_id, entity_id))
        self._reencode(affected)
        self._drop_orphan_entities(keep_declared=True)
        self._bump()
        after = {f: {r[0] for r in self._conn.execute("SELECT entity_id FROM fact_entities WHERE fact_id = ?", (f,))}
                 for f in affected}
        out = {"entity": alias, "entity_id": new, "from": row["owner"], "was": row["source"],
               "facts_moved": [f for f in affected if owner in before[f] and owner not in after[f]],
               "facts_shared": [f for f in affected if owner in after[f] and new in after[f]],
               "facts_to_check": sorted(f for f in check if owner in after.get(f, ()) and f != declaring)}
        if self._one("SELECT 1 FROM entities WHERE entity_id = ?", (owner,)) is None:
            out["from_dropped"] = True
        return out

    def split_alias(self, name: str) -> dict:
        """Make an alias an entity of its own (the split action, 0.17.0): one a fact's text gave, one someone gave or
        one of unknown origin; the new entity counts as declared. See _detach_alias for the facts. KeyError for an
        unknown name; ValueError for an entity's own name or a name of two entities."""
        self._require_writer()
        if not isinstance(name, str) or not name.strip():
            raise ValueError("split needs the alias in 'entity'")
        with self._tx():
            ids = self.entity_ids(name)
            if not ids:
                raise NotFound(f"no entity is named '{name.strip()}'")
            if len(ids) > 1:
                raise ValueError(f"'{name.strip()}' names {len(ids)} entities; fact_store entities shows them")
            row = self._alias_row(name)
            if row is None or row["entity_id"] != ids[0]:
                raise ValueError(f"'{name.strip()}' is the name of an entity, not an alias; rename or delete_entity "
                                 "changes a name")
            self._lift_tombstones(name)  # someone named it, as a declaration does (review 0.17.0)
            return self._detach_alias(row)

    def unconfirmed_text_aliases(self, limit: int = 20) -> list[dict]:
        """Aliases "X aka Y" in a fact's text gave that nobody confirmed (0.17.0), with the facts that say them: for
        review. All of them, however old: what matters is that a person looks, not when the text said it."""
        if not self._alias_sources():
            return []
        out = []
        for r in self.read("SELECT ea.alias, ea.alias_key, ea.entity_id, e.name AS owner FROM entity_aliases ea "
                           "JOIN entities e ON e.entity_id = ea.entity_id WHERE ea.source = 'aka' "
                           "ORDER BY e.name, ea.alias")[:limit]:
            facts = []
            for f in self.read("SELECT f.fact_id, f.content FROM facts f JOIN fact_entities fe ON fe.fact_id = f.fact_id "
                               "WHERE fe.entity_id = ? AND (f.content LIKE '%aka%' OR f.content LIKE '%a.k.a.%' "
                               "OR f.content LIKE '%also%known%as%') ORDER BY f.fact_id", (r["entity_id"],)):
                if any(entity_key(right) == r["alias_key"] and r["entity_id"] in self.entity_ids(left)
                       for left, right in self._aka_pairs(f["content"])):
                    facts.append(f["fact_id"])
            out.append({"entity": r["owner"], "alias": r["alias"], "facts": facts,
                        "reason": "a name a fact's text gave (\"X aka Y\"); nobody confirmed it is the same one"})
        return out

    def _entity_rows(self) -> list[dict]:
        """Every entity with its aliases, number of facts and standing, most linked first (0.16.0). Established: someone
        named it (declared with entities=[...], renamed, or given an alias with the alias action), or two or more
        current facts are linked to it; provisional: a name the extraction rules guessed from one fact, nobody
        confirmed and no other fact named. A standing, not a verdict: it drops back when the second fact goes.
        A name without capitals (Chinese, Japanese, Thai...) that the facts only ever put in quotes stays provisional
        however many facts quote it (0.16.2, issue #4): Chinese quotes a button, a menu item or a step as readily as a
        name, and two facts quoting "自动添加任务并解析元数据" made it established. Then it takes a person naming it, or
        two linked facts of which one names it outside plain quotes — in title marks 《》, in "X aka Y", or in
        running text as mentions are found (_used_outside_quotes)."""
        given = _GIVEN_ALIAS if self._alias_sources() else "1"
        rows = self.read("SELECT e.entity_id, e.name, e.entity_type, e.created_at, COUNT(fe.fact_id) AS facts FROM entities e "
                         "LEFT JOIN fact_entities fe ON fe.entity_id = e.entity_id GROUP BY e.entity_id "
                         "ORDER BY facts DESC, e.name")
        aliases: dict[int, list[str]] = {}
        textual: dict[int, list[str]] = {}  # aliases "X aka Y" in a fact's text gave, nobody confirmed (0.17.0)
        named: set[int] = set()
        for r in self.read(f"SELECT entity_id, alias, {given} AS given FROM entity_aliases ORDER BY alias"):
            aliases.setdefault(r["entity_id"], []).append(r["alias"])
            if r["given"]:
                named.add(r["entity_id"])
            elif self._alias_sources():
                textual.setdefault(r["entity_id"], []).append(r["alias"])
        outside = self._used_outside_quotes({
            r["entity_id"]: aliases.get(r["entity_id"], []) for r in rows
            if r["facts"] >= 2 and r["entity_id"] not in named and r["entity_type"] != "declared" and _uncased(r["name"])})
        out = []
        for r in rows:
            declared = r["entity_type"] == "declared"
            established = declared or r["entity_id"] in named or (
                r["facts"] >= 2 and (not _uncased(r["name"]) or r["entity_id"] in outside))
            out.append({"entity_id": r["entity_id"], "name": r["name"], "facts": r["facts"], "declared": declared,
                        "aliases": aliases.get(r["entity_id"], []),
                        "text_aliases": textual.get(r["entity_id"], []),
                        "status": "established" if established else "provisional",
                        "_since": r["created_at"] or ""})
        return out

    def _used_outside_quotes(self, candidates: dict) -> set[int]:
        """Of *candidates* ({entity_id: its aliases}), those a linked current fact names outside plain quotes (see
        _entity_rows): on the fact's text with plain quotes blanked (_outside_plain_quotes), what the linking readers
        find — mentioned_entity_ids, and the names _extract_entities finds (title marks, "X aka Y") resolved by
        entity_ids — or an alias with case written there as a word, compared as entity keys compare (Straße =
        STRASSE: a one-word alias is not looked up in running text, review 0.16.2). Each fact's text is read once
        and kept, with what it names, for the current facts only and until an entity changes (review 0.16.2: 1,000
        names on two long facts took 11 s; a forgotten fact's text must not stay in memory)."""
        if not candidates:
            return set()
        with self._lock:
            version = (self._entry.get("generation", 0), self._one("PRAGMA data_version")[0])
            cached = self._entry.get("outside")
            previous = cached[1] if cached is not None and cached[0] == version else {}
            current: dict[int, tuple] = {}
            links: dict[int, list[int]] = {}
            ordered = sorted(candidates)
            for k in range(0, len(ordered), _MAX_COUNTED_IDS):
                chunk = ordered[k:k + _MAX_COUNTED_IDS]
                for row in self._conn.execute(
                        f"SELECT entity_id, fact_id FROM fact_entities WHERE entity_id IN ({','.join('?' * len(chunk))})",
                        chunk):
                    links.setdefault(row["fact_id"], []).append(row["entity_id"])
            fact_ids = sorted(links)
            for k in range(0, len(fact_ids), _MAX_COUNTED_IDS):
                chunk = fact_ids[k:k + _MAX_COUNTED_IDS]
                for row in self._conn.execute(
                        f"SELECT fact_id, content FROM facts WHERE fact_id IN ({','.join('?' * len(chunk))})", chunk):
                    content = row["content"] or ""
                    seen = previous.get(row["fact_id"])
                    if seen is None or seen[0] != content:
                        bare = _outside_plain_quotes(content)
                        ids = set(self.mentioned_entity_ids(bare))
                        for name in self._extract_entities(bare):
                            ids.update(self.entity_ids(name))
                        seen = (content, bare.casefold(), frozenset(ids))
                    current[row["fact_id"]] = seen
            self._entry["outside"] = (version, current)
            found: set[int] = set()
            for fact_id, (_, bare, ids) in current.items():
                for entity_id in links[fact_id]:
                    if entity_id in ids or any(not _uncased(a) and _cased_word_in(
                            bare, unicodedata.normalize("NFKC", a).casefold()) for a in candidates[entity_id]):
                        found.add(entity_id)
            return found

    def list_entities(self, limit: int = 50, status: str | None = None) -> list[dict]:
        """Entities with their aliases, number of facts and standing (see _entity_rows), most linked first. *status*
        "established" or "provisional" keeps those only; provisional ones come oldest first, the first to clean up."""
        if status not in (None, "all", "established", "provisional"):
            raise ValueError("status must be 'all', 'established' or 'provisional'")
        rows = self._entity_rows()
        if status in ("established", "provisional"):
            rows = [r for r in rows if r["status"] == status]
        if status == "provisional":
            rows.sort(key=lambda r: (r["_since"], r["entity_id"]))
        return [{k: v for k, v in r.items() if k != "_since"} for r in rows[:limit]]

    def entity_counts(self) -> dict:
        """{"entities": all, "provisional": guessed names nobody confirmed (see _entity_rows), "text_aliases": aliases
        a fact's text gave that nobody confirmed (0.17.0)}."""
        rows = self._entity_rows()
        return {"entities": len(rows), "provisional": sum(r["status"] == "provisional" for r in rows),
                "text_aliases": sum(len(r["text_aliases"]) for r in rows)}

    def stale_provisional_entities(self, days: int = 60, limit: int = 20) -> list[dict]:
        """Provisional entities older than *days*, oldest first: clean-up candidates for review (0.16.0)."""
        old = {r[0] for r in self.read("SELECT entity_id FROM entities WHERE julianday('now') - julianday(created_at) > ?",
                                       (days,))}
        rows = sorted((r for r in self._entity_rows() if r["status"] == "provisional" and r["entity_id"] in old),
                      key=lambda r: (r["_since"], r["entity_id"]))
        return [{"entity_id": r["entity_id"], "entity": r["name"], "facts": r["facts"],
                 "reason": (f"a name guessed from one fact; nobody declared it or named it again in {days}+ days"
                            if r["facts"] < 2 else
                            f"found only in quotes, in {r['facts']} facts; nobody declared it or used it as a name in "
                            f"{days}+ days")}
                for r in rows[:limit]]

    def _extract_entities(self, text: str) -> list[str]:
        """Entity candidates (see the pattern table) shaped like a name (_name_shaped) and not deleted (delete_entity),
        deduplicated by casefold in first-seen order. The text is read in NFC, so a letter written with a combining
        accent is one letter ("José" either way), and with code spans and blocks blanked out (0.16.1)."""
        text = _without_code(unicodedata.normalize("NFC", text or ""))
        quoted = _straight_quoted(text) + [m.group(1) for pattern in _RE_QUOTED for m in pattern.finditer(text)]
        found = [(n, False) for n in _capitalized_phrases(text)] + [
            (q, True) for q in quoted if len(q) <= _QUOTED_MAX_CHARS and len(q.split()) <= _QUOTED_MAX_WORDS]
        for left, right in self._aka_pairs(text):
            found += [(left, False), (right, False)]
        uniq: dict[str, str] = {}  # casefolded key -> first-seen spelling, insertion-ordered
        for raw, quoted in found:
            name = raw.strip(" ,;:!?").rstrip(".") if raw else ""  # a leading dot stays: ".NET" (0.16.1)
            if name and fold(name) not in uniq and _name_shaped(name, quoted=quoted) and not self._tombstoned(name):
                uniq[fold(name)] = name
        return list(uniq.values())

    def _aka_pairs(self, text: str) -> list[tuple[str, str]]:
        """(name, alias) for the first _AKA_MAX "X aka Y" in *text* (NFC, code blanked: see _extract_entities) whose
        sides are shaped like names and not deleted.
        Each side is read within _AKA_WINDOW characters of its "aka": reading the whole text before and after every
        "aka" made a 10,000-character chain of them take a second per fact (review 0.16.0). A name is at most
        _ENTITY_MAX_CHARS long, far under the window, so a run the window cuts is too long to be a name anyway."""
        text = _without_code(unicodedata.normalize("NFC", text or ""))
        pairs = []
        for m in itertools.islice(_RE_AKA.finditer(text), _AKA_MAX):
            before, after = text[max(0, m.start() - _AKA_WINDOW):m.start()], text[m.end():m.end() + _AKA_WINDOW]
            left_text, right_text = before.rstrip(), after.lstrip()
            if not (_inline_gap(before[len(left_text):]) and _inline_gap(after[:len(after) - len(right_text)])):
                continue  # "aka" on a line of its own
            left = _aka_side(left_text, from_end=True)
            right = _aka_side(right_text, from_end=False)
            left, right = (n.strip(" ,;:!?").rstrip(".") if n else "" for n in (left, right))
            if text_fold(left.lower()) in STOPWORDS or text_fold(right.lower()) in STOPWORDS:
                continue  # "Bruce Banner aka the Hulk": "the" is no alias (review 0.16.1)
            if _name_shaped(left, quoted=_mixed(left)) and _name_shaped(right, quoted=_mixed(right)) \
                    and entity_key(left) != entity_key(right) and not self._tombstoned(left) and not self._tombstoned(right):
                pairs.append((left, right))
        return pairs

    def _text_entity_ids(self, content: str, declared_keys=frozenset()) -> set[int]:
        """Entities *content* itself names: extracted names (created if new) and "X aka Y" (Y an alias of X rather than
        a second entity, unless Y is already one, or is a name declared with this very fact: *declared_keys*, 0.17.0)."""
        pairs = self._aka_pairs(content)
        alias_only = {entity_key(right) for _, right in pairs}
        ids = {self._resolve_entity(name) for name in self._extract_entities(content)
               if entity_key(name) not in alias_only or self.entity_ids(name)}
        for left, right in pairs:
            if not self.entity_ids(right) and entity_key(right) not in declared_keys:
                ids.add(self.add_alias(left, right, source="aka")["entity_id"])
        return ids

    @staticmethod
    def _wiki_page_entities(text: str) -> tuple[str, ...]:
        """Entities for a wiki pointer fact: "<wiki>/[section/]page-slug.md" yields the wiki namespace and the page slug.

        Bulk wiki ingest writes facts through the same add_fact path but with no names the extractor can see: a line
        like "Wiki page: ai-agents/entities/llama-cpp.md" carries no capitalised phrase, so without this every
        re-ingested page stays invisible to the memory graph. The namespace and the slug are the only proper nouns such
        a line has. Hopeless slugs (index, schema, log, readme, the wiki root) are dropped: structural, not topics.
        """
        m = _RE_WIKI_PAGE.search(text or "")
        if m is None:
            return ()
        path = m.group(1).strip().strip("<>[]()`'\"")
        if path.lower().endswith(".md"):
            path = path[:-3]
        parts = [p for p in re.split(r"[/\\]+", path) if p and p not in {".", ".."}]
        if len(parts) < 2:
            return ()
        named = [parts[0].replace("-", " ").replace("_", " "), parts[-1].replace("-", " ").replace("_", " ")]
        out: list[str] = []
        for name in (n.strip() for n in named):
            key = entity_key(name)
            if not key or key in _WIKI_STRUCTURAL_SLUGS or not _valid_entity(name):
                continue
            if key not in {entity_key(x) for x in out}:
                out.append(name)
        return tuple(out)

    def _link_entities(self, fact_id: int, content: str, declared=()) -> None:
        """Link the fact to its entities: names extracted from the text, names the caller declared (a one-word name
        such as "Masha" is accepted here), and entities already known that the text mentions in any word form.
        A wiki pointer line contributes its namespace and page slug (see _wiki_page_entities)."""
        declared_keys = frozenset(entity_key(_clean_given(n)) for n in declared or ())
        ids = self._text_entity_ids(content, declared_keys) | self.mentioned_entity_ids(content)
        ids |= {self._resolve_entity(name) for name in self._wiki_page_entities(content)}
        for entity_id in sorted(ids):
            self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (fact_id, entity_id))
        self._link_declared(fact_id, declared)

    def _declare(self, fact_id: int, declared) -> None:
        """Declare names on a fact whose text stays as it is (0.17.0: an add of a text already stored, a correct that
        changes no text, a change into a text another fact holds): until 0.16.3 such declarations were dropped."""
        if not declared:
            return
        self._link_declared(fact_id, declared)
        self._reencode([fact_id])  # its roles changed: the vector again (emptied without numpy), banks go (0.17.1)

    def _link_declared(self, fact_id: int, declared) -> None:
        """Link names the caller declared (one-word names allowed) and mark them declared. A declared name is someone
        naming it: a tombstone of delete_entity on it is lifted. A name that is only an alias "X aka Y" in a fact's
        text gave (nobody confirmed it) is not resolved to X: it becomes an entity of its own first (0.17.0, see
        _detach_alias; the operation's collect_splits says so), since a text must not decide who a name someone gives
        belongs to."""
        for name in declared or ():
            name = _clean_given(name)
            if not _valid_entity(name, given=True):
                logger.info("holographic: a declared name of %d characters is no name; not linked", len(name))
            else:
                self._lift_tombstones(name)
                row = self._text_alias(name) if self._entry.get("migrated") else None
                if row is not None:
                    also = any(row["entity_id"] in self.entity_ids(_clean_given(other))
                               for other in declared if _clean_given(other) != name)
                    detached = self._detach_alias(row, declaring=fact_id if also else None)
                    if getattr(_SPLITS, "found", None) is not None:  # reported once the transaction commits
                        _SPLITS.pending.append((name, detached))
                entity_id = self._resolve_entity(name)
                if self._write("UPDATE entities SET entity_type = 'declared' WHERE entity_id = ? AND entity_type != 'declared'",
                               (entity_id,)).rowcount:
                    self._bump()
                self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (fact_id, entity_id))

    def _known_entities(self) -> dict[str, list[tuple[int, tuple]]]:
        """{key of the first word: [(entity_id, key sets of a name's words)]} for entities safe to match in free text:
        multi-word names, and one-word names only when declared (older stores hold one-word junk such as a quoted
        "Done", which would otherwise attach itself to every fact saying "done"). Cached per connection and rebuilt
        after any entity change here, a rollback, or a commit by another connection (PRAGMA data_version)."""
        with self._lock:
            version = (self._entry.get("generation", 0), self._one("PRAGMA data_version")[0])
            cached = self._entry.get("entities")
            if cached is None or cached[0] != version:
                index: dict[str, list[tuple[int, tuple]]] = {}
                names: dict[int, list[str]] = {}
                forms: set[str] = set()
                if self._keyed():
                    for row in self._conn.execute("SELECT ea.entity_id, ea.alias FROM entity_aliases ea "
                                                  "JOIN entities e ON e.entity_id = ea.entity_id"):
                        names.setdefault(row["entity_id"], []).append(row["alias"])
                for row in self._conn.execute("SELECT entity_id, name, aliases, entity_type FROM entities"):
                    for name in [row["name"], *((row["aliases"] or "").split(",")), *names.get(row["entity_id"], [])]:
                        name = unicodedata.normalize("NFKC", name)  # as mentioned_entity_ids reads text
                        if name.strip().startswith(".") and len(name.strip()) > 1:  # ".NET": with its dot, below
                            index.setdefault(".", []).append((int(row["entity_id"]), name.strip().lower()))
                            continue
                        key = name_keys(name)
                        forms.update(k for word_forms, exact in key if not exact for k in word_forms)
                        if len(key) > 1 or (key and row["entity_type"] == "declared"):
                            for first in key[0][0]:
                                index.setdefault(first, []).append((int(row["entity_id"]), key))
                        glued = name.strip().lower()
                        if UNSEGMENTED.search(glued) and len(glued) >= 2 and (len(key) > 1 or row["entity_type"] == "declared"):
                            index.setdefault("", []).append((int(row["entity_id"]), glued))  # "" = substring names
                cached = self._entry["entities"] = (version, index, frozenset(forms))
            return cached[1]

    def known_name_forms(self) -> frozenset:
        """Keys of the words of every entity's name and aliases read as names (name_forms): «ян» and «яна» of «Яна»
        or of «Яна Соколова». One-word names count whether declared or guessed, unlike the index above: a key of
        this set only lets a case form of a name stand in for the word in recall's coverage (FactRetriever.relevant),
        it links nothing to the entity. Names in Latin script give no key (they match as written). Cached with the
        index."""
        with self._lock:
            self._known_entities()
            return self._entry["entities"][2]

    def mentioned_entity_ids(self, text: str) -> set[int]:
        """Known entities whose name (or alias) occurs in *text* as a run of words: a Latin-script name as written
        (or with 's), a name in another script by its keys, so "Маши" and "Машу" mention "Маша" and "لإيرما" mentions
        "إيرما", while "jeans" does not mention "Jean" and "машина" not "Маша". Looked up by each word's keys, not by
        scanning names. The text is read in NFKC, as names are indexed, and stretch by stretch between separators
        (0.16.1): the words of one name stand together on one line, never across a line break, a comma or a bracket
        ("Alice\nSmith" and "Alice, Smith" do not mention Alice Smith)."""
        text = unicodedata.normalize("NFKC", text or "")
        index = self._known_entities()
        found = set()
        for stretch in _mention_stretches(text):
            sequence = analyze(stretch).words
            for i, word in enumerate(sequence):
                for first in word.keys:
                    for entity_id, key in index.get(first, ()):
                        if entity_id not in found and len(sequence) - i >= len(key) and all(
                                sequence[i + j].text in forms if exact else not forms.isdisjoint(sequence[i + j].keys)
                                for j, (forms, exact) in enumerate(key)):
                            found.add(entity_id)
        lowered_text = text.lower()
        for entity_id, name in index.get(".", ()):  # ".NET" with its dot, not the word "net" (0.16.1)
            if entity_id not in found and re.search(r"(?<![\w.])" + re.escape(name) + r"(?!\w)", lowered_text):
                found.add(entity_id)
        substring_names = index.get("", ())
        if substring_names and UNSEGMENTED.search(text):  # a name in Chinese/Thai text has no spaces around it
            if is_chinese(text):
                # From word cut to word cut only (0.16.1: 和服 was found in 我和服务员, "I and the waiter"), Traditional
                # and Simplified as one
                folded = analyze(text).text
                cuts = zh_word_cuts(folded)
                for entity_id, name in substring_names:
                    if entity_id not in found and _found_between_cuts(folded, text_fold(name), cuts):
                        found.add(entity_id)
            else:
                lowered = text.lower()
                for entity_id, name in substring_names:
                    if name in lowered:
                        found.add(entity_id)
        return found

    def _resolve_entity(self, name: str) -> int:
        """Return the entity_id for a casefold name or alias match, creating the entity if absent."""
        ids = self.entity_ids(name)
        if ids:
            return ids[0]
        self._bump()
        if not self._entry.get("migrated"):
            return int(self._write("INSERT INTO entities (name) VALUES (?)", (name,)).lastrowid)  # type: ignore[arg-type]
        return int(self._write("INSERT INTO entities (name, name_key) VALUES (?, ?)", (name, entity_key(name))).lastrowid)  # type: ignore[arg-type]

    def _drop_entity_row(self, entity_id: int) -> None:
        """The entity's row and its aliases; the caller unlinks its facts first and bumps the index generation."""
        if self._entry.get("migrated"):  # a store no writer has migrated yet may lack entity_aliases
            self._write("DELETE FROM entity_aliases WHERE entity_id = ?", (entity_id,))
        self.forget_relations(entity_id)  # a deleted entity leaves no edges behind (a merge already repointed them)
        self._write("DELETE FROM entities WHERE entity_id = ?", (entity_id,))

    def _drop_orphan_entities(self, keep_declared: bool = False) -> None:
        """Entities no fact links to go — but not one the user gave an alias (alias may come before any fact names
        it; an unrelated rewrite used to wipe it with its aliases). Forget drops the names its fact gave itself.
        First, aliases "X aka Y" gave (0.16.0) go with the last current fact that says so (forget, end, a new text): an
        alias is the text's, and a forgotten fact must not live on as one; such an alias keeps nothing. *keep_declared*:
        a declared entity stays without facts too (the rebuild leaves what someone named as it is)."""
        self._prune_aka_aliases()
        keep = (f" AND entity_id NOT IN (SELECT entity_id FROM entity_aliases WHERE {_GIVEN_ALIAS})"
                if self._entry.get("migrated") else "")
        if keep_declared:
            keep += " AND COALESCE(entity_type, '') != 'declared'"
        if self._write("DELETE FROM entities WHERE entity_id NOT IN (SELECT entity_id FROM fact_entities)"
                       + keep).rowcount:
            self._bump()
        if self._entry.get("migrated"):
            self._write("DELETE FROM entity_aliases WHERE entity_id NOT IN (SELECT entity_id FROM entities)")
        if self._graph_ready():  # a store written before the graph table existed has none to prune
            self._write("DELETE FROM entity_relations WHERE src_entity_id NOT IN (SELECT entity_id FROM entities) "
                        "OR dst_entity_id NOT IN (SELECT entity_id FROM entities)")

    def _aka_backing(self, entity_ids) -> set[tuple[int, str]]:
        """(entity_id, alias key) of every "X aka Y" that a current fact linked to one of *entity_ids* says."""
        out: set[tuple[int, str]] = set()
        for entity_id in sorted(set(entity_ids)):
            for row in self._conn.execute(
                    "SELECT f.content FROM facts f JOIN fact_entities fe ON fe.fact_id = f.fact_id WHERE fe.entity_id = ? "
                    "AND (f.content LIKE '%aka%' OR f.content LIKE '%a.k.a.%' OR f.content LIKE '%also%known%as%')",
                    (entity_id,)).fetchall():
                for left, right in self._aka_pairs(row["content"]):
                    if entity_id in self.entity_ids(left):
                        out.add((entity_id, entity_key(right)))
        return out

    def _prune_aka_aliases(self, legacy_keys=frozenset()) -> None:
        """Drop aliases "X aka Y" gave that no current fact says any more, from both representations; with
        *legacy_keys* (the "Y" a forgotten fact said), also aliases of unknown origin ('legacy') with those keys. The
        facts the alias linked are linked again from their text, so a name only the gone text gave (the "X") does not
        stay on them either: it then goes as an entity without facts (review 0.16.0)."""
        if not self._entry.get("migrated"):
            return
        rows = self._conn.execute("SELECT alias_key, entity_id, alias, source FROM entity_aliases WHERE source = 'aka'"
                                  + (f" OR (source = 'legacy' AND alias_key IN ({','.join('?' * len(legacy_keys))}))"
                                     if legacy_keys else ""), tuple(sorted(legacy_keys))).fetchall()
        if not rows:
            return
        backed = self._aka_backing(r["entity_id"] for r in rows)
        gone = [r for r in rows if (r["entity_id"], r["alias_key"]) not in backed]
        for r in gone:
            self._drop_alias_row(r["alias_key"], r["entity_id"])
        if gone:
            self._bump()
            self._relink({f for r in gone for f, in self._conn.execute(
                "SELECT fact_id FROM fact_entities WHERE entity_id = ?", (r["entity_id"],)).fetchall()})

    def _aka_alias_keys(self, texts) -> set[str]:
        """Keys of the "Y" of every "X aka Y" in *texts* (what a forgotten fact said; see _prune_aka_aliases)."""
        return {entity_key(right) for text in texts if text for _, right in self._aka_pairs(text)}

    def _named_in(self, text: str, entity_ids) -> set[int]:
        """Which of *entity_ids* *text* names: a mention as recall reads it (mentioned_entity_ids), or the words of the
        entity's name or one of its aliases in a row, compared as entity identity compares names (entity_key: NFKC,
        casefold, so "strasse" names "Straße") with accents folded ("Jose" names "José")."""
        wanted = set(entity_ids)
        found = self.mentioned_entity_ids(text) & wanted
        words = _key_words(text)
        for entity_id in sorted(wanted - found):
            row = self._one("SELECT name, aliases FROM entities WHERE entity_id = ?", (entity_id,))
            if row is None:
                continue
            names = [row["name"], *(row["aliases"] or "").split(",")]
            if self._entry.get("migrated"):
                names += [r[0] for r in self._conn.execute("SELECT alias FROM entity_aliases WHERE entity_id = ?",
                                                           (entity_id,))]
            if any(_in_a_row(words, _key_words(name)) for name in names if name and name.strip()):
                found.add(entity_id)
        return found

    def _relink(self, fact_ids) -> None:
        """Link facts to their entities again from their own text, as a new text does (declared links stay)."""
        for fact_id in sorted(set(fact_ids)):
            row = self._one("SELECT content FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                continue
            self._relink_fact(fact_id, row["content"])
        self._reencode(fact_ids)

    # -- names someone removed, renamed or wants indexed again (0.16.0) ------------------------------------------------

    def _require_writer(self) -> None:
        """delete_entity, rename_entity and rebuild_entity_index write what only a migrated store holds (tombstones,
        where an alias came from, the scrub mark): a store opened without migrating refuses them (review 0.16.0: it
        answered "scrub": "done" for a file it had not rewritten)."""
        if not self._entry.get("migrated"):
            raise ValueError("this store was opened read-only (without migrating); entity changes need the main agent")

    def _name_digest(self, name: str) -> str:
        """A deleted name is kept as a keyed hash, as the text of a forgotten fact is: the file never says it."""
        return self.digest("entity " + entity_key(name))

    def _name_digests(self, name: str) -> set[str]:
        """The digests a deleted name is kept under: its key, and its key folded as search folds words (Traditional and
        Simplified Chinese as one, accents dropped), so 台湾大学 is suppressed after 臺灣大學 was deleted (review 0.16.1).
        The first is what 0.16.0 wrote."""
        return set(self._name_digest_forms(name))

    def _name_digest_forms(self, name: str) -> tuple[str, str, str]:
        """The three digests of _name_digests in order (key, folded key, key of 0.16.0), as version_names keeps them."""
        key = entity_key(name)
        return tuple(self.digest("entity " + k) for k in (key, text_fold(key), _entity_key_0160(name)))

    def _tombstoned(self, name: str) -> bool:
        """Whether delete_entity removed this name: text never creates it again."""
        if not self._entry.get("migrated") or not entity_key(name) \
                or self._one("SELECT 1 FROM entity_tombstones LIMIT 1") is None:
            return False
        digests = sorted(self._name_digests(name))
        return self._one(f"SELECT 1 FROM entity_tombstones WHERE digest IN ({','.join('?' * len(digests))})",
                         digests) is not None

    def _lift_tombstones(self, *names: str) -> None:
        """Someone named it (entities=[...], alias, rename): a deleted name may be created from text again. What a
        version kept of it went when the tombstone was written (trigger version_names_tombstoned): naming it again
        brings none of that back."""
        if not self._entry.get("migrated") or self._one("SELECT 1 FROM entity_tombstones LIMIT 1") is None:
            return
        for name in names:
            if entity_key(name or ""):
                for digest in sorted(self._name_digests(name)):
                    self._write("DELETE FROM entity_tombstones WHERE digest = ?", (digest,))

    def _reencode(self, fact_ids) -> int:
        """Vectors of facts whose entities changed: computed again; without numpy emptied, so that the next writer that
        has it computes them (_check_vectors) instead of keeping roles of names that are gone. Banks go too."""
        done = 0
        for fact_id in sorted(set(fact_ids)):
            row = self._one("SELECT content FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                continue
            if self._hrr_available:
                self._compute_hrr_vector(fact_id, row["content"])
            else:
                self._write("UPDATE facts SET hrr_vector = NULL WHERE fact_id = ?", (fact_id,))
            done += 1
        self._write("DELETE FROM memory_banks")
        return done

    def delete_entity(self, name: str) -> dict:
        """Delete an entity by its name or an alias: the name, its aliases and its links to facts. The facts stay as
        they are: this cleans the index of names and forgets nothing (fact_store forget does that). The name and its
        aliases get a tombstone, so a fact's text never creates them again; an explicit name (entities=[...], alias,
        rename) lifts it. Aliases "X aka Y" gave go without one ("text_aliases_dropped", 0.17.1): a text decided they
        were X's, and Y may be someone else. The file is rewritten afterwards, as after forget, so freed pages keep no copy of a name no
        fact says ("scrub": "pending" when another process blocked it; the next open retries). KeyError when no entity
        has that name or alias."""
        result = self._delete_entity(name)
        result["scrub"] = "done" if self.scrub() else "pending"
        return result

    def delete_entities(self, names) -> dict:
        """delete_entity for several names in one transaction and one rewrite of the file (0.16.2, issue #4: cleaning
        up 23 provisional names took 23 calls). *names*: a list of 1 to _DELETE_MAX non-empty strings, checked whole
        before anything changes (review 0.16.2: a list with a number in it, or a dict, deleted the names it held).
        Each name must name one entity: one that names two (a name of one and an alias of another, in a store an older
        version wrote) refuses the call, as an unknown name alone does not. Answer: "deleted" (as delete_entity),
        "covered" (names of an entity this call deleted already: an alias given after its name), "not_found"; KeyError
        when none is known, ValueError for input that is no such list or for a name of two entities."""
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError("entities must be a list of names")
        if len(names) > _DELETE_MAX:
            raise ValueError(f"at most {_DELETE_MAX} names at once, {len(names)} were given")
        if not all(isinstance(n, str) and n.strip() for n in names):
            raise ValueError("every name in entities must be a non-empty string")
        wanted = list(dict.fromkeys(n.strip() for n in names))
        self._require_writer()
        deleted, covered, missing = [], [], []
        with self._tx():
            plan, ambiguous, textual = [], [], []
            for name in wanted:
                ids = self.entity_ids(name)
                if not ids and self._version_history_names(name):
                    plan.append((name, None))  # only what a version kept of it is left (_delete_history_name)
                elif not ids:
                    missing.append(name)
                elif len(ids) > 1:
                    ambiguous.append(name)
                else:
                    if self._text_alias(name) is not None:
                        textual.append(name)
                    plan.append((name, ids[0]))
            if ambiguous:
                raise ValueError("these names belong to more than one entity: " + ", ".join(f"'{n}'" for n in ambiguous)
                                 + "; fact_store entities shows them — delete each by a name or alias only it has")
            if textual:  # checked whole, before anything changes (0.17.0)
                raise ValueError(self._text_alias_refusal(textual[0], self._text_alias(textual[0]))
                                 + (f" (also: {', '.join(repr(n) for n in textual[1:])})" if textual[1:] else ""))
            if not plan:
                raise NotFound("no entity is named " + ", ".join(f"'{n}'" for n in missing))
            done: set = set()
            for name, entity_id in plan:
                seen = entity_id if entity_id is not None else ("history", entity_key(name))
                if seen in done:
                    covered.append(name)
                    continue
                deleted.append(self._delete_entity(name))
                done.add(seen)
        return {"deleted": deleted, "covered": covered, "not_found": missing,
                "scrub": "done" if self.scrub() else "pending"}

    def _delete_entity(self, name: str) -> dict:
        self._require_writer()
        with self._tx():
            ids = self.entity_ids(name or "")
            if not ids and self._version_history_names(name):
                return self._delete_history_name(name)
            if not ids:
                raise NotFound(f"no entity is named '{(name or '').strip()}'")
            if len(ids) > 1:  # one entity's name and another's alias, in a store an older version wrote (review 0.16.2)
                raise ValueError(f"'{name.strip()}' belongs to more than one entity; fact_store entities shows them — "
                                 "delete each by a name or alias only it has")
            textual = self._text_alias(name)  # a text must not decide whose name is deleted (0.17.0)
            if textual is not None:
                raise ValueError(self._text_alias_refusal(name, textual))
            names: list[str] = []
            textual_names: list[str] = []  # aliases "X aka Y" gave: they go with the entity, without a tombstone
            facts: set[int] = set()
            aliases = 0
            source = "COALESCE(source, 'legacy')" if self._alias_sources() else "'legacy'"
            for entity_id in ids:
                row = self._one("SELECT name, aliases FROM entities WHERE entity_id = ?", (entity_id,))
                rows = self._conn.execute(f"SELECT alias, {source} FROM entity_aliases WHERE entity_id = ?",
                                          (entity_id,)).fetchall()
                own = [r[0] for r in rows]
                legacy = [a.strip() for a in (row["aliases"] or "").split(",") if a.strip()]
                # A text decided that "Y" is X's, not someone: deleting X must not keep Y from ever being found again
                # (review 0.17.0, Fable: "Bob Brown lives in Berlin" then linked no entity at all).
                text_keys = {entity_key(r[0]) for r in rows if r[1] == "aka"} - {entity_key(row["name"])}
                textual_names += [a for a in dict.fromkeys(a for a in (*own, *legacy) if entity_key(a) in text_keys)]
                names += [n for n in (row["name"], *own, *legacy) if entity_key(n) not in text_keys]
                aliases += len({entity_key(a) for a in (*own, *legacy)} - {entity_key(row["name"])})
                facts |= {r[0] for r in self._conn.execute("SELECT fact_id FROM fact_entities WHERE entity_id = ?", (entity_id,))}
                self._write("DELETE FROM fact_entities WHERE entity_id = ?", (entity_id,))
                self._drop_entity_row(entity_id)
            if self._version_names_ready():  # what versions kept of these names goes, every spelling a tombstone covers
                self._drop_version_names(names)
                for key in sorted({entity_key(n) for n in textual_names if entity_key(n)}):
                    self._write("DELETE FROM version_names WHERE name_key = ?", (key,))
            if self._entry.get("migrated"):
                for digest in sorted({d for n in names if entity_key(n) for d in self._name_digests(n)}):
                    self._write("INSERT OR IGNORE INTO entity_tombstones (digest) VALUES (?)", (digest,))
                self._mark_scrub()
            self._reencode(facts)
            self._bump()
            result = {"entity": names[0], "deleted": True, "facts_unlinked": len(facts), "aliases_dropped": aliases,
                      "facts_deleted": 0}
            if textual_names:
                result["text_aliases_dropped"] = sorted({entity_key(n): n for n in textual_names}.values())
            return result

    def _delete_history_name(self, name: str) -> dict:
        """delete_entity of a name no entity has any more but a version kept (version_names: declared on a fact whose
        text changed since, so a revert would bring it back): those rows go, the name gets its tombstone, and the file
        is rewritten, as for an entity."""
        spelt = self._version_history_names(name)
        versions = self._drop_version_names([name, *spelt])
        if self._entry.get("migrated"):
            for digest in sorted({d for n in (name, *spelt) for d in self._name_digests(n)}):
                self._write("INSERT OR IGNORE INTO entity_tombstones (digest) VALUES (?)", (digest,))
        self._mark_scrub()
        self._bump()
        return {"entity": spelt[0] if spelt else name.strip(), "deleted": True, "facts_unlinked": 0, "aliases_dropped": 0,
                "facts_deleted": 0, "past_versions_cleared": versions}

    def rename_entity(self, name: str, new_name: str) -> dict:
        """Give an entity another name; its id, links and aliases stay, the old name becomes an alias, and the entity
        counts as declared (someone named it). Only another spelling ("phoenix" -> "Phoenix") changes the name alone;
        the entity's own alias swaps places with its name. A name another entity has, as its name or an alias, is
        refused: two entities become one only through alias, never through a rename (a typo would merge two people).
        KeyError for an unknown name; ValueError for an empty or unusable new name, or one that names two entities."""
        self._require_writer()
        new = _clean_given(new_name)
        new_key = entity_key(new)
        if not new_key or not _valid_entity(new, given=True):
            raise ValueError(f"'{str(new_name)[:80]}' cannot be a name (2-{_ENTITY_MAX_CHARS} characters, mostly letters; "
                             "no paths, URLs, key=value, commas or flags)")
        with self._tx():
            ids = self.entity_ids(name or "")
            if not ids:
                raise NotFound(f"no entity is named '{(name or '').strip()}'")
            if len(ids) > 1:
                raise ValueError(f"'{name.strip()}' names {len(ids)} entities; rename one by a name only it has")
            textual = self._text_alias(name)  # a text must not decide whose name is changed (0.17.0)
            if textual is not None:
                raise ValueError(self._text_alias_refusal(name, textual))
            entity_id = ids[0]
            taken = self._text_alias(new)  # the hint below would merge two people a text tied (review 0.17.0, Fable)
            if taken is not None and taken["entity_id"] != entity_id:
                raise ValueError(self._text_alias_refusal(new, taken))
            others = [i for i in self.entity_ids(new) if i != entity_id]
            if others:
                other = self._one("SELECT name FROM entities WHERE entity_id = ?", (others[0],))["name"]
                raise ValueError(f"'{new}' already names the entity '{other}'. If both are one, alias {{entity: '{other}', "
                                 f"content: '{name.strip()}'}} makes them one and keeps '{other}' as the name; rename "
                                 "after that if the other name should stay.")
            row = self._one("SELECT name, aliases FROM entities WHERE entity_id = ?", (entity_id,))
            old = row["name"]
            old_key = entity_key(old)
            legacy = [a.strip() for a in (row["aliases"] or "").split(",") if a.strip()]
            self._lift_tombstones(new)
            if new_key != old_key:
                self._write("DELETE FROM entity_aliases WHERE alias_key = ? AND entity_id = ?", (new_key, entity_id))
                self._put_alias(old_key, entity_id, old)
                legacy = [a for a in legacy if entity_key(a) != new_key]
                if old_key not in {entity_key(a) for a in legacy}:
                    legacy.append(old)
            self._write("UPDATE entities SET name = ?, aliases = ?, entity_type = 'declared' WHERE entity_id = ?",
                        (new, _legacy_aliases(legacy), entity_id))
            if self._entry.get("migrated"):
                self._write("UPDATE entities SET name_key = ? WHERE entity_id = ?", (new_key, entity_id))
            self._reencode([r[0] for r in self._conn.execute("SELECT fact_id FROM fact_entities WHERE entity_id = ?",
                                                             (entity_id,))])
            self._bump()
            aliases = [r[0] for r in self._conn.execute("SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY alias",
                                                        (entity_id,))]
            return {"entity_id": entity_id, "name": new, "previous": old, "aliases": aliases}
