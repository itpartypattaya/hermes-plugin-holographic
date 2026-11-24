# Fact history, provenance and forgetting (0.8.0)

Status: implemented in 0.8.0. This note is the contract; the code and its tests follow it.

Until 0.7.0 the store lived in an eternal present. `add supersedes=` stored the new text as a new fact and
deleted the old one; `update content=` overwrote the text in place; nothing said where a fact came from. So the
store could not answer "where did Alice live in August?", could not tell a fact that changed from one that was
wrong, and "forget this" deleted the row while copies of the text stayed in the search index's free pages.

0.8.0 keeps the history of every fact, records where each version was written, and makes "forget" final.

## Two kinds of time

| | meaning | stored as | used by |
|---|---|---|---|
| **valid time** | when the fact held in the world | `valid_from`, `valid_to` (dates) | `as_of` |
| **recorded time** | when the agent learned it / closed it | `recorded_at`, `closed_at` (UTC timestamps), `recorded_on` (local date) | shown in `history`; "what did we know on a date" is not a v1 query |

On 2026-10-12 the user says "I changed jobs on 1 September": the new version gets `valid_from = 2026-09-01` and
`recorded_at = 2026-10-12 …`. `as_of=2026-09-15` finds the new job; `history` shows that it was learned later.

- Intervals are half-open, `[valid_from, valid_to)`: on the day of a change the new version holds.
- The granularity is a date (`YYYY-MM-DD`). The local date is taken in Hermes's configured timezone (UTC when none
  is set). Dates in the future are refused.

## Nothing is invented

A fact's start is often unknown, and every fact stored before 0.8.0 has no start at all. The store does not make
one up:

- `valid_from = NULL` means *unknown*. The fact is known to hold from the day it was recorded: `as_of` on or after
  `recorded_on` finds it; an earlier date reports it as **undetermined** ("history unknown"), never as true.
- A change without a date closes the old version with an *upper bound*: `valid_to` = the day the change was
  recorded, `valid_to_exact = 0`. The old version held on the day it was recorded; between that day and the bound
  it is undetermined; from the bound on it does not hold.
- A dated change closes it exactly: `valid_to = valid_from` of the new version.

So every `as_of` answer has two lists: `results` (known to hold on that date) and `undetermined` (may have held;
the reason says why). When `results` is empty the answer says that the history for that date is unknown rather
than returning today's state.

## Six ways a fact stops being current

They are different actions in the API because they mean different things:

| action | meaning | what happens | in `as_of` |
|---|---|---|---|
| `change` | the world changed (moved, new job, new price) | current version closed as `changed`, new version current; **same `fact_id`** | old version for dates before the change |
| `end` | it stopped holding and nothing replaces it ("no longer works at X") | current version closed as `ended`; the fact leaves the current store | until `valid_to` |
| `correct` | it was wrong (typo, misheard, wrong date) | current version closed as `corrected`, new version inherits `valid_from` (or takes the corrected one) | the wrong version is **never** returned |
| `revert` | undo the last `change` / `correct` / `end` (e.g. junk from auto-extraction) | last version → `corrected`, the previous one is current again; an ended fact is current again | as if the undone edit never happened |
| `forget` | the user asked to forget it | the fact and **all** its versions are physically deleted; a tombstone without text stays | gone |
| `merge` (0.19.0) | it says the same as another fact (a duplicate) | its current version closed as `merged`, a `fact_merges` row points to the keep; the fact leaves the present, its versions, sources and confirmations stay under its own id | until the local day of the merge, under its own id with `merged_into` |

A correction may also fix only the date: `correct {fact_id, valid_from}` moves the start, and the end of the
previous version moves with it when that version was closed exactly at the old start.

Changing tags, keywords, category or trust is not a new version: it edits the current version in place.

### Who a new text is about (0.22.0)

`change`, `correct` and `update` with a new text link the fact again from that text. Names given with it in
`entities` replace the names declared on the fact before: changed from "Leo's guitar teacher is Ben Cole." into
"... is Omar Reid." with `entities: ["Leo", "Omar Reid"]`, the fact is no longer found by `probe {entity: "Ben Cole"}`
(before, the old declaration stayed and attributed the new text to the old person). A name the new text itself
gives stays linked whether or not it is given again. Without `entities`, a declared name the old text named and
the new text does not goes with the old text (the same change without names is not found under Ben Cole either);
a declared name no text says stays: a fact about "my teacher" declared as Ben Cole keeps him across a rewording.
"Named" is read as entity identity reads names: case folded ("strasse" names Straße, "МАША" names Маша), accents
folded ("Jose" names José), and an alias counts as the name. A list whose names are all unusable (blank, a path, a
URL) replaces nothing and is read as no names. With the same text (`correct` of the date only, `update` with the
text it already has, a change into a text another fact holds, `merge`) names given are added, as before.

`revert` brings back the names the version had. When `change`, `correct` or `update` closes a version, the names
declared on the fact then are kept for it (`version_names`), and a revert to it declares exactly those again: one-word
names ("Masha") and names no text says ("my teacher" declared as Ben Cole and replaced by Pia Lund) included. The
record is used up by that revert; the next closing of the version records it anew. A name `delete_entity` removed
since is not given back, not even when someone names it again later, and when it was the only name kept, the revert
brings back none. `delete_entity` also finds a name only such a record still holds (its entity went with its last
link): the record loses it, the name gets a tombstone and the file is rewritten (`past_versions_cleared` in the
answer). A version closed before this release, by an older one, or closed again by an older one after a revert has
no record of that closing: its revert links the restored text from itself and keeps a declared name unless the
undone text named it and the restored one does not.

### Merge (0.19.0)

`merge {fact_id: <keep>, from: [ids], keywords?, expect?}` is for duplicates, where `forget` was the only tool and
took the copy's history and the person's confirmations with it. The keep is not touched as a claim: its text, trust,
origin, category and user tags stay; its keywords are joined with the copies' (or replaced by the list given) and
the group's `retrieval_count` / `helpful_count` are added once. Each copy keeps its versions, sources and
confirmations under its own id; `history {keep}` lists them under `merged_from`, `history {copy}` answers
`merged_into`. A copy's last text answers `exists` with the keep when it is added, changed into or resumed again
and auto-extraction skips it — only while the keep is still at the version it had at the merge (`into_version_id`);
once the keep is changed or ended, the copy's text is the keep's past claim (`past_version`), not its new one, and
auto-extraction skips it after a change, end or correction of the keep, as it skips the keep's own closed
versions. Its earlier `changed` versions stay past versions as before, a `corrected` one is nothing. Chains do not
form: merging a keep into another fact moves its copies to the new root, and a merge into a copy is refused; a copy
that was merged into an earlier version of that keep (the keep changed since) keeps pointing at that version, so it
stays a past claim under the new root. `expect: {"<id>": <version_id>}` makes the merge conditional on each named
fact being at that version, so an approval of the texts someone saw merges exactly those texts (`changed_since`
otherwise, nothing written). A cron job merges only facts it stored in the same session. Confirmations of a copy
are not counted in the keep's `origins` (0.19.0), but a person who confirmed a copy does not raise the keep's trust
a second time. `forget` of the keep forgets its copies too, a tombstone for each; `forget` of a copy erases it
alone. There is no unmerge.

### Backward compatibility of the API

| old call | 0.8.0 meaning |
|---|---|
| `add {content, supersedes: X}` | `change X` (the id stays X — before 0.8.0 the new text got a new id) |
| `update {fact_id, content}` | `correct` |
| `update {fact_id, keywords / tags / category / trust_delta}` | in place, no new version |
| `remove {fact_id}` | `forget` |
| built-in memory `replace` (with `mirror_memory_writes`) | `change`, no date |
| built-in memory `remove` (with `mirror_memory_writes`) | `forget` |

If a change would make the fact say exactly what another current fact already says, the fact is *ended* and the
other fact is returned (the old `supersedes` did the same by deleting it).

## Default search sees only the present

`facts` stays the projection of current versions: the same table, columns and triggers as before, so `search`,
`probe`, `related`, `reason`, prefetch, the n-gram and word scans, the entity index, HRR vectors and banks never see
a closed version. A past text cannot leak into an ordinary answer, not even through a word that only the old text
contained.

History lives in `fact_versions` with its **own** FTS5 index, `fact_versions_fts`. Only `as_of` reads it:

- the search corpus is the versions that may hold on the date (`corrected` never qualify); candidates, IDF, the
  relevance floor and `limit` are computed over that corpus, so `limit=1` returns the best version *for that date*;
- HRR does not take part (vectors exist for current facts only);
- `as_of` is supported by `search` and `list`; `probe`, `related` and `reason` refuse it explicitly.

### A fact that comes back

X → Y → X is a third version of the same fact, not a new fact and not a duplicate. `add X` while X is a past
version of fact #N stores nothing and answers `status: "past_version"` with a hint to use `change {fact_id: N}`.
The write check also lists close (not identical) past versions under `past_versions`. Corrected versions are never
offered. Auto-extraction skips a sentence that equals any past version.

## Provenance

Each version records where it was written, i.e. "written while answering this person in this chat" — not "who
said it":

| field | example |
|---|---|
| `via` | `tool`, `auto_extract`, `mirror`, `legacy` (stored before 0.8.0) |
| `platform`, `chat_type` | `telegram`, `group` |
| `chat_id`, `thread_id` | `-1001234567890`, `7` |
| `session_id` | the Hermes session |
| `user_id`, `user_name` | the author of the turn |
| `origin` (0.15.0) | who stands behind it: `said` (the person said it), `confirmed` (the agent's conclusion, the person agreed), `inferred` (the agent's own conclusion); empty: not known |

Chat names are not stored (they change and they leak). Facts from before 0.8.0 say "source unknown".

Confirmations (0.15.0) live in their own table, one row per fact per session (per day without a session): when, the
session, the person's id, and through what (`feedback`, `tool`), never text. A conclusion with a confirmation counts as
`confirmed`. `change` and a resume drop a fact's confirmations (they were of the old claim), `forget` deletes them.

- `list`, `history` and `export` show the source in full.
- Prefetch does not show it by default (`recall_source: off`). `recall_source: short` appends
  "recorded 2026-10-12 · Leon · this chat / another chat" to each recalled fact, without ids.
- `forget` deletes the source with the fact.

## Forget means gone

`forget` runs in one transaction: the fact row, every version, both FTS entries, entity links, entities and aliases
left without facts, the category banks, and the fact's vector of search by meaning (0.14.0; an embeddings provider
that made it has seen the text, which no forget reaches). Then, to remove copies the database itself keeps:

1. both FTS indexes are rebuilt from their content tables (an FTS5 delete only adds a marker; the words stay in old
   segments until a rebuild);
2. the connection runs with `PRAGMA secure_delete = ON`, so freed pages and cells are zeroed;
3. `VACUUM` rewrites the file (free space left by earlier versions of the plugin is not zeroed otherwise);
4. `PRAGMA wal_checkpoint(TRUNCATE)` empties the write-ahead log.

Steps 3–4 can be blocked by another process holding the database (a reader in the middle of a long query). The
store does not wait for it (busy timeout 0 for these two statements; every other memory operation of the agent
queues behind a forget): the result says `scrub: "pending"` at once and the store retries at its next open or next
forget. The acceptance test greps the raw bytes of the `.db` and `-wal` files for a word that only the forgotten
fact contained.

In rollback-journal mode (the core keeps `journal_mode=DELETE` instead of WAL on a SQLite it considers
WAL-reset-vulnerable, and on network file systems) a reader in another process blocks the *commit* itself: the
forget then fails with "database is locked" after the busy timeout, nothing is changed, no tombstone is written,
and the connection is rolled back so the next write works. Retry the forget.

### Tombstones

`forgotten(fact_id, forgotten_at, digests)` keeps the id (ids are never reused: `AUTOINCREMENT`) and a keyed hash
(HMAC-SHA256 with a per-database random key) of the normalised text of every version — **never the text**.
Tombstones are kept forever.

A consumer that holds a text and wants to know whether it was forgotten can compute the digest itself — the
format is stable:

- `digests` is a JSON list of lower-case hex strings;
- each is `HMAC-SHA256(key, normalize(text))`, where `key` is the bytes of `meta.tombstone_key` (hex, 16 random
  bytes, created the first time a writer opens the file; every per-user / per-chat database has its own) and
  `normalize` = Unicode NFKC → `casefold()` → whitespace collapsed to single spaces → leading and trailing
  `" .!?;,"` stripped (`history.normalize`).

A paraphrase does not match: only the texts the fact's versions actually had are covered. Tombstones are used for
two things:

- auto-extraction silently skips a sentence that matches a tombstone, so a forgotten fact does not come back from
  old messages; an `add` from a cron job (a nightly extraction re-reading old messages) is refused with
  `status: "forgotten"`; a manual `add` by the main agent is stored, with a hint that it was forgotten earlier;
- consumers that copied a fact elsewhere clean their copies: `SELECT fact_id, forgotten_at FROM forgotten`.

What the plugin cannot reach is said plainly. The copies other software makes (a dreaming skill's caches and diary,
a backup of the database file, a git history of dumps) are cleaned by that software, by fact id. Hermes's own
session database keeps the messages the fact was learned from, and `hermes backup` archives are not rewritten.

## Schema

```sql
CREATE TABLE fact_versions (
    version_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id        INTEGER NOT NULL,            -- stable id of the fact (facts.fact_id while it is current)
    seq            INTEGER NOT NULL,            -- 1, 2, 3 … within the fact
    content, category, tags,                    -- the version's text and metadata
    valid_from     TEXT,                        -- YYYY-MM-DD or NULL (unknown)
    valid_to       TEXT,                        -- YYYY-MM-DD, exclusive; NULL = open
    valid_to_exact INTEGER NOT NULL DEFAULT 1,  -- 0: valid_to is an upper bound
    status         TEXT NOT NULL,               -- current | changed | ended | corrected | merged (0.19.0)
    recorded_at, recorded_on, closed_at,
    supersedes, superseded_by,                  -- version_ids
    trust_score,                                -- snapshot when the version was closed
    src_via, src_platform, src_chat_type, src_chat_id, src_thread_id, src_session_id, src_user_id, src_user_name,
    src_origin,                                 -- 0.15.0, added by the first 0.15.0 writer to an older file
    UNIQUE (fact_id, seq)
);
-- 0.15.0
CREATE TABLE fact_confirmations (fact_id INTEGER NOT NULL, confirmed_at TEXT NOT NULL, session_id TEXT, user_id TEXT,
                                 via TEXT);
-- at most one current version per fact
CREATE UNIQUE INDEX idx_versions_current ON fact_versions(fact_id) WHERE status = 'current';
CREATE VIRTUAL TABLE fact_versions_fts USING fts5(content, tags, content=fact_versions, content_rowid=version_id);
CREATE TABLE forgotten (fact_id INTEGER PRIMARY KEY, forgotten_at TEXT NOT NULL, digests TEXT NOT NULL);
-- 0.19.0; status also takes 'merged'
CREATE TABLE fact_merges (from_fact_id INTEGER PRIMARY KEY, into_fact_id INTEGER NOT NULL,
                          into_version_id INTEGER NOT NULL, merged_at TEXT NOT NULL);
-- 0.22.0: names declared on the fact while the version was current, kept when change / correct / update
-- closes it; a row with an empty name marks every record (it exists, even with no name left in it); closed_by is
-- the version that closed it. A revert that undoes exactly that closing declares the names again and deletes the
-- record. d_key / d_fold / d_old: the name's keyed hashes in the three forms a tombstone is written under (no
-- name in them), indexed. Deleted with the version and when a tombstone covers the name (triggers, which an older
-- release's forget and delete_entity fire too).
CREATE TABLE version_names (version_id INTEGER NOT NULL, name TEXT NOT NULL, name_key TEXT NOT NULL,
                            closed_by INTEGER NOT NULL, d_key TEXT, d_fold TEXT, d_old TEXT,
                            PRIMARY KEY (version_id, name_key));
CREATE TRIGGER version_names_ad AFTER DELETE ON fact_versions
    BEGIN DELETE FROM version_names WHERE version_id = old.version_id; END;
CREATE TRIGGER version_names_tombstoned AFTER INSERT ON entity_tombstones
    BEGIN DELETE FROM version_names WHERE name_key != ''
          AND (d_key = new.digest OR d_fold = new.digest OR d_old = new.digest); END;
```

`version_names` is additive: any writer creates it on open, an older release ignores it (its own change and revert
keep names as before, a version it closes gets no record) and still removes its rows, since the triggers are part
of the file: its forget fires `version_names_ad`, and its `delete_entity` writes tombstones, which fire
`version_names_tombstoned` inside its own transaction, before it rewrites the file — so the rows go with every
spelling the tombstone covers ("Jose" with "José"), whoever deleted the name and whichever connection is open, and no
open has anything to reconcile or hash. `delete_entity` of this release also deletes by the same digests what a
tombstone written earlier covers. A record whose `closed_by` is not the version a revert undoes (an older release
reverted and changed the fact in between) is not used.

`facts` gets no new column, so older versions of the plugin and tools that read `facts` directly (hermes-dreaming)
keep working. Every write is one transaction.

## Migration and older versions on the same file

- The first writer of 0.8.0 gives every fact a version: `seq 1`, `current`, `valid_from NULL`, `recorded_at` = the
  fact's `updated_at` (the latest moment its text is known to have been true), `via legacy`. A context that may
  write migrates: the main agent, and a cron job with `cron_writes` on (the default). Read-only contexts
  (subagents, flush, cron with `cron_writes: false`) do not; `as_of` and `history` there answer with a clear error
  until a writer has.
- A file whose `meta.history_schema` is higher than this version knows was written by a newer release: a writer
  refuses to open it ("update the plugin"), so its migration cannot stamp the schema back; readers still read.
- An older version may write the file in between (a downgrade, a second profile). Each writer open reconciles:
  - a fact without a current version → a `legacy` version;
  - a fact whose text differs from its current version (an older `update`) → the version is closed as `changed`
    with an upper-bound end, and the new text becomes a `legacy` version;
  - tags or category that differ → synced in place;
  - a current version whose fact is gone (an older `remove` or `supersedes`) → treated as `forget`: history
    deleted, tombstone written. An `ended` fact is legitimately absent and is left alone.
- **History schema 2 (0.19.0).** The first `merge` in a file raises `meta.history_schema` to 2; a file without a
  merge stays at 1. From then on writers of 0.15–0.18 refuse the file (the rule above), so going back below 0.19.0
  after a merge leaves the main agent without memory until 0.19.0 or a backup from before the merge is back; setting
  the schema back by hand would let an older `forget` of a keep leave its copies' text. A writer of an older release
  already open when the schema rose is not stopped: on its next open, 0.19.0 erases the copies of a keep that is gone
  (a tombstone each) and drops a merge row whose copy is gone. Readers of 0.15–0.18 see the keep only and show a
  copy in `as_of` under its own id without the merge-day clip; releases before 0.15.0 do not read the schema.
- CI runs 0.8.0 → v0.1.1 → 0.8.0 and 0.8.0 → 0.7.0 → 0.8.0 on one database; 0.19.0 tests a merged file against
  0.18.2 and v0.7.0 from their tags.

## Who may write

The main agent writes. A cron job writes only through its explicit `fact_store` / `fact_feedback` calls
(`cron_writes`, on by default: memory-maintenance jobs exist to make them), and nothing automatic happens there
(auto-extraction, mirroring, recall counts). Subagents and flush contexts read only. A writer that can write also
migrates the file; one that cannot never does and creates no history table (as before 0.8.0, a reader opening a
file still sets WAL mode and the base schema on it; it changes no data).

## Scopes

`scope: user | chat | thread` keeps a database per user or chat; each has its own history and its own tombstones.

## Not in v1

"What did we know on a date" (a query by recorded time); time of day; `as_of` for `probe` / `related` / `reason`;
future `valid_from`; "who said it" as distinct from "whom we were answering"; cleaning Hermes's session database and
`hermes backup` archives.
