# Holographic Memory Provider

Local memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent): a SQLite fact store with
FTS5 search, trust scoring, entity resolution and HRR (holographic reduced representation) retrieval. By default
everything stays on your machine: no API key, no network, no model calls. [Search by
meaning](#search-by-meaning-optional-online) is an optional online add-on, off unless you turn it on.

**Maintained by [itpartypattaya](https://github.com/itpartypattaya)** since October 2026. This is the provider that
shipped inside Hermes core as `plugins/memory/holographic/`; Nous Research
[handed it off](https://github.com/NousResearch/hermes-plugin-holographic/blob/main/HANDOFF.md) when the bundled
copy was scheduled to leave core (announced for October 15, 2026). Same provider name, same config, same database:
users who have `memory.provider: holographic` keep their facts. See [CHANGELOG.md](CHANGELOG.md) for what changed.

## How it works

![Architecture: before each turn Hermes core sends the message through the relevance gate and search; writes pass a
validator that refuses instructions and credentials and land in one SQLite transaction with the fact's history; recall
and history tools read the same file; search by meaning is an optional online add-on](diagram/architecture.svg)

![Recall before a turn: greetings recall nothing; the message's words become keys; 15 candidates come from FTS5,
character n-grams or a word scan, reordered by meaning if that is on; a relevance gate, a repeat window and the screen
decide; up to five facts by their words, one by meaning alone and two of named entities go into the turn as
data](diagram/recall.svg)

## Install

While your Hermes still bundles `holographic`, the bundled copy is used and this plugin stays dormant. After core
drops it, `hermes update` installs the catalog plugin named `holographic` for every profile that has it configured.
To install by hand:

```bash
hermes plugins install holographic          # from the plugin catalog
hermes config set memory.provider holographic
```

or `hermes memory setup` and pick `holographic`.

The catalog entry may still pin an older release while a version bump waits for review; to install a given release
exactly, pass its commit: `hermes plugins install holographic --ref <40-character sha of the tag> --force`.

### Memory went off after `hermes update`

Core dropped its bundled copy on 2026-10-11 (Hermes v0.21.7) and installs this plugin in its place. If that
install fails (a dependency resolution error in the plugin workspace, a network error), Hermes reports "Memory
provider 'holographic' is not installed, so external memory is off", and a failed install may leave `memory.provider`
empty and `holographic` listed under `plugins.disabled` in `config.yaml`. Your facts are not touched:
`memory_store.db` stays where it was. To bring memory back:

1. `hermes plugins install holographic` (add `--force` if a copy is already in `~/.hermes/plugins/holographic/`);
   if it fails on dependencies, update Hermes first — the cause is usually in core's lock, not in this plugin.
2. In `config.yaml`: remove `holographic` from `plugins.disabled`, and set `memory.provider: holographic`.
3. Restart the gateway (or start a new CLI session) and check with `hermes memory status`.

## Config

In `config.yaml` under `plugins.hermes-memory-store`:

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite database path |
| `auto_extract` | `false` | Store user sentences that state a preference or decision (English, Russian, Chinese) at session end and before context compression, at trust 0.4; questions and restatements of a stored fact are skipped |
| `mirror_memory_writes` | `false` | Also store every built-in memory `add` (MEMORY.md / USER.md) as a fact |
| `default_trust` | `0.5` | Trust score of a new fact |
| `min_trust_threshold` | `0.3` | Facts below this trust are not recalled |
| `temporal_decay_half_life` | `0` | Days; `0` turns recency decay off |
| `hrr_dim` | `1024` | HRR vector dimensions |
| `hrr_weight` | `0.3` | Weight of the HRR term in search ranking |
| `prefetch_min_coverage` | `0.3` | Share of a message's informative words (rarer words weigh more) a fact must match to be recalled before the turn; a fact sharing one word of a longer message is recalled only when no other fact holds that word or it may be a name (capitalized inside the sentence, or in a script without capitals) |
| `prefetch_repeat_turns` | `6` | A fact recalled before a turn is not injected again for this many turns of the session |
| `recall_source` | `off` | `short`: each fact recalled before a turn also says where it was written (`recorded 2026-10-12 · Anton · this chat`; no ids). Off by default: it costs tokens and tells the room which chat a fact came from. `list`, `history` and `export` always show the source in full |
| `cron_writes` | `true` | A cron job's explicit `fact_store` / `fact_feedback` calls may write (nightly memory jobs such as hermes-dreaming's extraction and keywords exist to make them). Automatic writes (auto-extraction, mirroring, recall counts) stay off in cron, subagents and flush contexts; with `embeddings` on, a cron job that may write also makes missing fact vectors; `false` makes cron read-only too |
| `recall_indicator` | `true` | Report recalled facts to the core's recall indicator ("🧠 Holographic — recalled 2 memories"). On Telegram and other messaging platforms that line arrives as a separate message before the reply; `false` turns it off, recall itself is unchanged. Like every key here it is read when the gateway starts: a change takes effect after a restart |
| `embeddings` | `off` | `on`: search by meaning through an embeddings API you name (`embedding_url`, `embedding_model`, `embedding_key_env`, …) — ONLINE: fact texts and messages are sent there. See [Search by meaning](#search-by-meaning-optional-online) |
| `rerank` | `off` | `on`: rescore the candidate pool before a turn with a cross-encoder (`rerank_url`, `rerank_model`, `rerank_window`, …) — ONLINE: candidate texts and the message are sent there. See [Rescoring by a cross-encoder](#rescoring-by-a-cross-encoder-optional-online) |
| `scope` | `profile` | `user`: a separate database per user (`holographic-users/<platform>-<user id>.db` next to `db_path`); in a shared chat the store follows the author of each turn, and with no user identity nothing is recalled or written. `chat`: a separate database per chat (`holographic-chats/<platform>-<chat id>.db`) — for one bot serving many unrelated groups; `thread`: per forum topic / thread (`…-t<thread id>.db`), the chat's own database outside threads. Without a chat (CLI, cron, subagents) nothing is recalled or written. File names: an id made only of `A-Z a-z 0-9 _ . -` is used as is (Telegram's `-1001234567890`); any other character becomes `_`, the name is cut to 60 characters, and `-<first 10 hex of sha256(the original "platform-id")>` is appended, so distinct ids never share a file |

A number that is not one (`hrr_dim: abc`, `min_trust_threshold: high`), and a value outside a key's fixed set
(`scope: users`, `recall_source: full`, `embeddings: maybe`, `auto_extract: sure`; 0.17.8 — before, it was read as the
default, and a mistyped `scope` put everyone in one store) turn memory off for the session: the system prompt and every
`fact_store` call name the key and what it must be, never the value, and no database is opened or written until it is
fixed. The `embedding_*` numbers are replaced by their default, or brought into their range, with a warning in the
gateway log that names the key. A `config.yaml` the core cannot parse: Hermes serves the copy it last read whole, and
the plugin uses its settings from it with a warning; with no such copy (or a core that raises) nothing tells what
`scope` was, so memory is off until the file reads (0.17.8; it used to run on the defaults). Warnings give the error's
type only: a YAML error quotes the line it stopped at, which may hold a key.

## Tools

| Tool | Description |
|------|-------------|
| `fact_store` | add, search, probe, related, reason, contradict, change, end, correct, revert, forget, history, update, remove, list, stats, export, review, entities, alias, rename, delete_entity, split, rebuild_entities, enrich |
| `fact_feedback` | Rate a fact helpful or unhelpful (moves its trust score), or record that the person confirmed it (0.15.0) |

- **search**: keyword candidates in any word form (`Маши` finds `Маша`, `deploys` finds `deploy`, `kucingnya` finds
  `kucing`; see [Languages](#languages)), reranked by word overlap and HRR similarity, weighted by trust.
- **Before each turn** the provider recalls up to five facts: search candidates that pass a relevance gate (the fact
  must cover enough of the message's informative words; "ok, thanks" recalls nothing) plus the top facts of entities
  the message names. A fact already injected in this session is not repeated for `prefetch_repeat_turns` turns.
  Facts are framed as data (`#id [trust] text`), one line each.
- **add** takes `entities` (who the fact is about; needed for one-word names such as a first name, so later facts
  mentioning it in any form link to it), `keywords` (see [Synonyms](#synonyms-and-other-roots-keywords)) and
  `valid_from` (the date it started to hold, when the message says). A name in `entities` that cannot be one (a
  path, a URL, `key=value`, a single letter, not a string) is not linked, and the answer of add, update, change and
  correct lists it under `entities_not_linked` by its position in the list (`index`, from 0) with the reason, not
  the name itself (a refused name may be `KEY=secret`); the fact is stored all the same. Its answer lists `similar` facts,
  `possible_conflicts` (same subject, a different claim) and `past_versions`; the agent decides. Text that is not a
  fact (JSON, tool or chat-role markup, instruction-like phrasing, a credential with its value or a token whose shape
  alone says it is one — an API key prefix, a Telegram bot token (also inside a Bot API address), a JWT, a private key
  (0.14.5) —; instructions and credentials in all eighteen languages of [Languages](#languages), also glued
  to Chinese text (0.15.1); an environment variable that holds a key, a token or a password, by its name — any in
  capitals ending like `…_KEY` or `…_TOKEN` and 310 popular ones of model and service providers — and an address
  with a password in it (0.15.2) —, an image (markdown or
  HTML), invisible characters, long blobs, >1000 characters) is refused with the reason. A name that begins with a
  credential word joined to the rest by a hyphen is no credential when the whole word names a thing (0.17.2, issue
  #6): a file with a common extension (`token-saver-stats.jsonl`, `plugins/token-saver/filters/web.yaml`), a host of
  three labels and more (`password-reset.example.com`), a package with a version after `==` or `@`
  (`token-saver==1.2`, `token-saver@1.2.3`); a file name shaped like a password is none (`password-Secret2026.conf`,
  `password-P@ssw0rd.txt`: a capital with a digit, or "@"), nor is an e-mail address. A credential word that stands alone is followed by its value,
  whatever it looks like: `password Summer2026.txt`, `password: secret/hunter2.txt`, `Пароль файла: Лето2026.txt`,
  `password-wifi: Qwerty123`, `Password wifi: correct-horse`, and `the token counter.py module` and `Token file:
  tokens.json` too; so is a dot that is
  no extension (`password-Qwerty.123`), a one-letter or a key file's extension (`token-Secret.key`), a two-label host
  (`password-letmein.co`), a tag after `:` (`password-wifi:192.168.1.1`, and so `token-service:1.4`). The cost,
  chosen: `password-hunter2.txt` is not caught. A request to send a credential stays refused whatever follows the word ("send credentials-backup.json",
  and so "share the token-saver report" too); a Russian credential word counts as a word, not as the start of one
  (`секретарю`, `ключевые цифры`, `учётная политика`, `токенизатор`, "пин на карте" are none; `пин-код`, "пин от
  карты", `секретная фраза`, `фраза восстановления` are). Recall
  checks a stored fact again before it goes into a prompt (0.13.0): one an older release, or another program, wrote
  past today's rules (markup, instructions, credentials, invisible characters, images) is left out of recall
  and logged once; `search`, `list` and `history` still show it, so it can be changed or forgotten. A quoted
  instruction to the assistant (a note quoted as telling it to drop its rules) may be stored as a report of what
  something says, but recall leaves it out too (0.14.3): in the prompt it reads the same as the bare instruction.
- **stats** (what is stored and how much of it was ever recalled; `recall_error` while recall before a turn fails, so a
  broken store does not pass for one with nothing relevant), **export** (Markdown view), **review**
  (clean-up candidates: low trust, never recalled for 60 days, near-duplicates), **enrich** (facts without keywords,
  most recalled first, to give keywords with `update`).
- **entities** lists names with their aliases, fact counts and standing (0.16.0): `established` when someone named
  it (`entities=[...]`, `alias`, `rename`) or two or more facts name it, `provisional` when the rules below guessed it
  from one fact and nobody confirmed it; `status` lists one kind, and the answer says how many provisional names the
  store holds. A name whose letters have no case (Chinese, Japanese, Thai, Arabic...) that the facts only ever put in plain quotes stays
  provisional however many facts quote it (0.16.2): in Chinese a button or a step goes in quotes as readily as a name.
  Then it takes a person naming it, or two facts of which one uses it in title marks 《》, in "X aka Y" or in
  running text where mentions are found (a name of several words, or one a person declared: a one-word Chinese name
  nobody declared is not looked up in running text, where a quoted label is written bare as often). **alias** gives an entity a second name (or merges two entities; `entity` stays the name).
  **rename** gives it another name (the old one becomes an alias; a name another entity has is refused, so a typo
  never merges two). **delete_entity** removes a name that is not one — a phrase, a command, a label — with its aliases and
  links (`entities: [...]` for up to 100 in one call, 0.16.2); every fact stays, a tombstone (a keyed hash, not the name) keeps text from creating it again until someone
  names it explicitly, and the file is rewritten as after forget. An alias "X aka Y" in a fact's text gave goes with
  the last fact that says it, and so does X on the facts only that alias linked; an alias written before 0.16.0 is of
  unknown origin, kept like a given one and dropped only when a forgotten fact said it. A name with a comma is no
  name (the legacy aliases column is comma-separated). **review** lists provisional names older than 60 days under
  `entity_candidates`.
  Spelling variants are one entity (`Hermes Agent` = `hermes-agent`), and `X aka Y` makes `Y` an alias.
- **An alias a fact's text gave is a hint, not a decision** (0.17.0). "X aka Y" lets later texts that say Y link to X,
  and `probe Y` find X's facts — but no action someone takes with a name is resolved through it: a name declared
  (`entities=[...]` in add, update, change, correct) that is only such an alias becomes an entity of its own, and the
  answer says so under `split`, with the `alias` call that makes them one again; `alias`, `rename` and `delete_entity`
  by it are refused, and so is `rename` of someone else into it (its hint would have merged the two). Deleting X
  drops such aliases without keeping Y from being found again (`text_aliases_dropped`). Until 0.16.3, "Alice Smith aka Bob Brown" after facts about Alice made a later
  `entities=["Bob Brown"]` file the second person's facts under her, and `rename` or `delete_entity` by "Bob Brown"
  renamed or deleted her. `alias {entity: X, content: Y}` confirms such a name; **split** `{entity: <an alias>}` makes
  any alias an entity of its own. Facts whose text names Y then go to Y; one whose text names neither stays with X if
  someone declared X on it, and the answer lists it under `facts_to_check` — why a declared link exists the file does
  not record. If someone gave X the alias (`alias`), X stays an entity after the split even without facts; otherwise
  the answer says `from_dropped` when X went with its last fact. `entities` shows such names as `text_aliases`,
  `review` lists them under `alias_candidates` however old (up to its `limit`), `stats` gives their number. What this
  does not do: a text that says Y still links to X, so recall can bring X's facts for a message about Y until someone
  splits them; it guards what this version writes — an older version writing the same file still resolves a declared
  name through such an alias; and an alias written before 0.16.0 is of unknown origin even if a text gave it: it is
  shown under `aliases`, not refused, and `split` separates it. A name declared with a text already stored (an
  `add` of the same text, a `correct` of the date only, a `change` into another fact's text) now counts too; until
  0.16.3 it was dropped.
- **probe**: every fact linked to a named entity, from the entity index (name or alias, any letter case).
  **reason**: facts linked to *all* of several entities; when a name is not a known entity, keyword search that still
  requires every name. Entities are linked when a fact is stored: capitalized names of two to five words in any cased
  script on one line, ending where the script changes (`Alice Smith`, `Мария Иванова`, `Jean-Luc Picard`; "Docker
  Hub直连可用" gives `Docker Hub`), short quoted terms (`"Phoenix"`, `«Сбер Бизнес»`, `“华为”`, `「华为」`, `《平凡的世界》`,
  straight quotes beside Chinese characters too) and both sides of `X aka Y` (Y becomes X's alias only while no fact has named Y on its own);
  code spans and blocks give none, and a known name of several words is mentioned only as written together on one
  line (in Chinese, never across a seam between two words); a name begins with a letter or a digit, holds no invisible
  character, passes these checks in its NFKC form too, and flags, `key=value`, paths, URLs, IPs and commas are not names, and
  neither is a version number with nothing beside it that names a thing ("0.17.0 installed"; "Python 3.12" is one), a
  quoted span whose letters with case are all lower case and that holds no identifier with a letter in it ('count',
  '14 days', «быстро беги»; 'kanban.db' and 'gpt-4o' are), a quoted greeting ("Good morning", «Спасибо большое»,
  “谢谢”), a count ("3个月", "15分钟", "1 小时": an Arabic number and units; 360安全卫士 and 115网盘 are names),
  a quoted Chinese span that reads as something said — ending in a sentence-final particle (“知道了”) or with
  three or more function words besides 的 (“先去搜索再告诉我答案”; titles and brands with one or two, 平凡的世界,
  饿了么, 有赞科技有限公司, are names) — or one of more than 16 Han characters. A name someone gives (`entities`,
  `alias`, `rename`) skips these rules and needs only a letter ("3M", "C++"). A one-word name is not linked; probing it
  falls back to keyword search. When these rules change, a writer rebuilds the index on its first open (up to 2,000
  facts and 500,000 characters; a larger store says so in `stats` and is rebuilt with **rebuild_entities**), reading
  the facts in the order they were stored as adding them would; declared names, aliases someone gave and their links
  stay as they are.
- **related**: facts structurally tied to an entity in their HRR vector, above a noise floor.
- Every recall action honours `min_trust` (default `min_trust_threshold`).
- **contradict**: candidate conflicts, i.e. pairs about the same subject that make a similar but not identical
  claim ("lives in Paris" / "lives in London"). It suggests; it never edits.

### History, sources and forgetting (0.8.0)

A fact that stops being current is not overwritten. The agent says what happened, and the store keeps the history:

| action | when | effect |
|---|---|---|
| `change {fact_id, content, valid_from?}` | it changed (moved, new job, new price) | the old text becomes a past version of the **same** fact id |
| `end {fact_id, valid_to?}` | it no longer holds and nothing replaces it | the fact leaves the current store; its history stays |
| `correct {fact_id, content?, valid_from?}` | it was wrong | the wrong version is never shown as the past |
| `revert {fact_id}` | undo the last change, correction or end | the previous version is current again |
| `forget {fact_id}` | the user asks to forget it | deleted for good, every version included; only a tombstone without text stays |

- **Ordinary recall sees only the present.** Past versions live in their own table and index; `search` and `list`
  read them only with `as_of=YYYY-MM-DD` ("what held on that date"). The answer separates `results` (known to hold
  then) from `undetermined` (may have held: a start or an end that is not known to the day). The store invents no
  dates: a fact without `valid_from` is known to hold from the day it was recorded, and so is every fact stored
  before 0.8.0.
- **`history {fact_id}`** lists the versions with their dates, status and source.
- **Source.** Each version records where it was written: via (`tool`, `auto_extract`, `mirror`), platform, chat
  type, chat and thread id, session, and the author of the turn. Chat names are not stored.
- **Forget means gone.** The fact, its versions, links and orphaned entities are deleted, both search indexes are
  rebuilt, the file is rewritten (`VACUUM`, with `secure_delete` on) and the write-ahead log emptied, so no copy of the
  text stays in free pages. A tombstone (the id and keyed hashes, never the text) stops auto-extraction from bringing
  the fact back from old messages and tells other tools which ids to clean from their own copies.
- Older prompts keep working: `add supersedes=X` is `change X` (the id now stays `X`), `update content=` is `correct`,
  `remove` is `forget`.

### Who stands behind a fact (0.15.0)

The source says where a fact was written; it did not say whether the person said it or the agent concluded it. Six
months on, a rule the user set and a guess the agent made looked the same.

- **Origin.** `add`, `change` and `correct` take `origin`: `said` (the person said it), `confirmed` (the agent's
  conclusion, and the person agreed) or `inferred` (the agent's own conclusion). It is kept with the version, shown
  by `history`, `list`, `export` and `search`, and counted in `stats`. Auto-extracted sentences are `said` (they are
  the user's own words); facts stored before 0.15.0 and mirrored memory entries have no origin. A `correct` that
  names no origin keeps the version's (fixing a typo does not make a fact nobody's); a `change` is a new claim and
  takes only the origin it is given. A session opened before an update keeps the tools it was given and offers no
  `origin` until a new session starts: its facts land without one. A new fact or a changed version stored with no
  origin has none (`unknown` in `stats`), and the tool description says so; an `add` of a text already stored
  keeps that fact's origin and trust. In the database the origin is kept with
  the version, in `fact_versions.src_origin` of the current row (`status = 'current'`), not on `facts`. Stored
  times are UTC (`fact_versions.recorded_at`, `facts.created_at` / `updated_at`); for a window in local days use
  `fact_versions.recorded_on`, the same moment as a date in Hermes's timezone (UTC when none is set).
- **Starting trust.** What the person said or confirmed starts 0.1 above `default_trust`, the agent's conclusion 0.1
  below it (within 0 and 1), a fact without an origin at `default_trust` as before; a `change` starts over at its
  new origin's trust. Measured on 40 pairs of conflicting facts about the
  same person (one said, one inferred, stored in alternating order): without origins the one said came first in 17;
  with a step of 0.05, 0.1 or 0.15, in 40 of 40, in `search` and before a turn alike. An inference with no rival
  was still recalled every time (10 of 10). 0.1 keeps the order after a helpful vote or two on the inference.
- **Confirmations.** `fact_feedback {action: confirmed, fact_id}` records that the person confirmed a fact — said it
  again or agreed when asked — and the same text added again with `origin: said` in a later session counts too (not
  in the session that stored it: that is the agent repeating itself). At most one confirmation per fact per session,
  or per day where there is no session. Trust rises by 0.05 for each person who confirms a fact the first time:
  one group-chat member saying "yes, true" session after session raises it once (without a known id, once per
  session). An inference that has been confirmed counts as `confirmed`; a `change` drops the confirmations, which
  were of the old claim. A nightly job that finds a person's message backing a fact can confirm it the same way.
- **Recall says what is a guess.** Before a turn, an inference nobody confirmed ends with `(inferred)`, so the model
  does not repeat it as something it was told. `review` lists inferences nobody confirmed in 60 days.

On a family store the agent stored 22 facts in four months and never the same one twice, so confirmation by
repetition alone will be rare there: the explicit `confirmed` and a nightly job are where confirmations come from.

The full contract, including what the plugin cannot clean (Hermes's own session database, `hermes backup`
archives): [docs/history.md](docs/history.md).

### Nightly dreaming: hermes-dreaming

The store keeps what the agent was told; it does not decide what deserves to become a permanent memory.
[hermes-dreaming](https://github.com/itpartypattaya/hermes-dreaming) (in the Hermes plugin catalog as
`hermes-dreaming`) does that at night, deterministically and without its own model calls: an optional extraction job
stores candidate facts here through `fact_store`, and the nightly pass reads this store read-only, corroborates each
fact with what people actually said in their chats, proposes the few worth writing to `MEMORY.md` / `USER.md`, asks
whether old entries still hold, and gives facts without keywords their keywords. With 0.8.0 the two work as one: a
human's "no longer true", "that was wrong" or "forget it" to a dream question becomes `end`, `correct` or `forget`
here, and after a `forget` the dream removes its own copies of the fact (cache, diary lines, its cron logs) by the
tombstone's id. Use hermes-dreaming 2.4.0 or later with this release: cron jobs may write facts only from 0.8.0 on
(`cron_writes`).

## Languages

![How a word is matched across languages: words, language, function words, keys, match; examples in eighteen languages
that match (German compounds, Russian and English spellings of one word, Traditional and Simplified Chinese, Japanese
spelling variants) and examples that do not](diagram/languages.svg)

A word matches another when they share its plain form or its stem in the language of the text:

- **Stems**: Snowball for English, Russian, Spanish, French, Portuguese, Indonesian, German, Hindi, Italian, Turkish,
  Polish and Dutch, with two plural rules it lacks (Portuguese `ação` / `ações`, Spanish `lápiz` / `lápices`); for
  Arabic, Larkey's light stemmer (the one Lucene ships) next to Snowball's, plus the word without a glued conjunction or
  preposition before an alef (`لإيفا`, "for Eva"; `ولإيفا`, "and for Eva"); for Bengali, a light suffix stripper
  (Snowball has none) that leaves at least three letters, so `মাটি` (soil) is not `মা` (mother). Italian, Turkish,
  Polish and Dutch words (0.11.0) are stemmed as written and folded: Snowball's Turkish and Polish read the letters
  folding drops (`çocuğun` is `çocuk`, `domów` is `dom`), and a word typed without its marks still meets one typed with
  them. Turkish also gets the noun without the `s` of a possessive that Snowball leaves (`kedisi` → `kedi`); Dutch keeps
  a `ge-` that Snowball takes off (`gezin`, family, is not `zin`, sense). Polish needs snowballstemmer 3.1 or newer
  (current Hermes installs 3.1.1); with 2.x Polish words compare as written and the rest of Polish (function words,
  greetings) works.
- **Which language**: the script decides for Cyrillic, Arabic, Devanagari and Bengali. Latin text is English, Spanish,
  French, Portuguese, Indonesian, German, Italian, Turkish, Polish or Dutch by the function words it uses ("the", "el",
  "le", "o", "yang", "der", "il", "bir", "się", "het"). Text with none of them is stemmed in English, Spanish, French,
  Portuguese, Indonesian and German, and in Turkish, Polish or Dutch only when it has their letters (`ı ğ ş`; `ą ę ł ń
  ś ź ż ć`; `ij`): Polish stems of English words join unrelated ones (`army` / `arm`, `apply` / `apple`), so a
  one-word Polish or Italian question gets its stems only with a function word or such a letter. A word is dropped
  as a function word only when the other words show that language, or when it is the only sign of it in a sentence of
  four words or more: "son" alone, or in "where does my son study", is a search word; in "il boit son café" it is
  French "his". With no other evidence the English list applies, as it did to all Latin text before 0.5.0. Function
  words of the four newer languages that are also everyday words, names or abbreviations elsewhere (Italian `come`,
  `dove`, `per`; Turkish `kim`, `ben`; Polish `bo`; Dutch `door`, `met`, `van`, `wat`) do not tell the language and are
  dropped only where that language alone fits the other words: "Ana come carne" stays Spanish, "Kim Lane lives in Oslo"
  keeps `kim`. Italian `mia` (my) is not on the list: it is a common name.
- **Folding**: Latin accents and ligatures (`esta` finds `está`, `coeur` finds `cœur`), Arabic vowel marks and letter
  variants, the Indic nukta, and native digits (`2027` finds `২০২৭`, `HR24` finds `HR২৪`). Folding also merges what
  only an accent tells apart (Spanish `año` / `ano`), as the FTS5 index already did. Turkish `ı` and Polish `ł` compare
  as `i` and `l` (`İstanbul`, `ISTANBUL` and `istanbul` meet, `KIZIM` meets `kızım`, `szkola` meets `szkoła`); since the
  FTS5 index keeps those two letters, such a word is also searched as written.
- **One word, two scripts** (0.14.5): Russian text writes a borrowed word either way, and `бэкап` finds `backup`,
  `телеграме` finds `Telegram`, `впн` finds `VPN`, `визы` finds `visa`, and the other way round — as the same word,
  without search by meaning. 55 such words are listed (apps and services, devices, `server`, `backup`, `token`,
  `hotel`, `visa`, `chat`, `site`, `bot`…; `_LOANWORDS` in `text.py`). A word belongs to its group by its exact form
  — the spelling, an English plural, a Russian case ending — never by its stem: Snowball cuts `токен` to `ток`
  (electric current) and `виза` to `виз`, as in `визит`, and neither meets `token` or `visa`. A Latin spelling joins
  its group only in English or in text of no known language (0.14.6): French `chat` is a cat, German `über` folds to
  `uber`, Spanish `botes` are boats. A spelling whose Russian forms are another word is not listed (`убер` gives
  `уберу`, of `убрать`), and a short one is searched by its forms as whole words, so `бот` does not bring in
  `ботинки`. Words two languages share by meaning, not spelling (`password` / `пароль`), are not listed: that is what
  keywords are for.
- **Hyphens**: French elision and inversion (`d'Eva`, `prend-il`, `est-ce`) and Portuguese enclisis (`chama-se`) are
  taken apart in text of that language, reduplication (`anak-anak`) in any; a hyphen between scripts splits
  (`hr-ovh-এর`). A letters-only hyphenated token is an identifier in English text or text of no determined language
  (`hr-ovh`, `carry-on`): a message naming it is about that one, and a fact about `hr-other` is not recalled for it.
  In Spanish, French, Portuguese, Indonesian and German text it is a compound word (`week-end`, `sexta-feira`,
  `IT-Treffen`), so a host name in a question in those languages is matched like any word, not enforced.
- **Apostrophes**: Italian elision as French (`dell'anno`, `l'ospedale`, `quest'anno`, `dov'è`, `anch'io`; not before
  a lone `s`, so English `dev's` stays). A Turkish case ending after a name comes off in text that may be Turkish
  (`Ankara'da`, `Ayşe'nin`, `HR24'ün`, `hr-ovh'de` are Ankara, Ayşe, HR24, hr-ovh), so the name is mentioned and an
  identifier stays required; only Turkish endings, so `Eva's`, `o'clock` and `rock'n'roll` are left alone. In Dutch
  text the plural and possessive `'s` comes off (`pinda's`).
- **German compounds** (0.11.0): in text determined as German, a compound is split by a dictionary of German lexemes
  from Wikidata (CC0): parts of three letters or more joined by German linking elements (`-s-`, `-es-`, `-n-`, `-en-`,
  `-er-`, `-e-`, a dropped `-e`), the analysis with the fewest parts, Wikidata's own analysis where it has one
  (`Zahnarzttermin` = `Zahn` + `Arzt` + `Termin`, `Blumentopferde` = `Blumen` + `Topf` + `Erde`); two analyses that cut
  differently leave the word whole unless one is clearly likelier (`Staubecken`, `Wachstube` stay whole). Both
  directions: the last part of a compound in a fact matches like the word itself (`Katze` finds `Familienkatze`), and
  a compound in a question finds a fact that writes its parts apart (`Zahnarzttermin` finds `Termin beim Zahnarzt`).
  The parts before the last count 0.6 and never recall a fact alone, and so does a last part of three letters
  (`Tag` in `Alltag`); a question compound whose last part alone is in a fact counts 0.6 too. Words the dictionary knows as one word stay whole (`Matrose` is no `Rose`, `Mitglied` no
  `Lied`, `Ausland` no `Land`); 0.6.0's suffix test stays for words it does not know (`Arbeitslaptop`). On Wikidata's
  own compounds held out of the build, 94.9 % are split, 99.9 % of the splits as Wikidata cuts them; of 204 simple
  words that look splittable, 4 are still split (`Plattform`, `Tablette`, `Kommode`, `Zoophobie`). Compounds whose
  meaning is not their parts' are never split (0.12.0: `Hochzeit` is no `Zeit`, `Frühstück` no `Stück`,
  `Weihnachten` no `Nacht`, `Augenblick`, `Verstand`, `Reisende`, `Wochenende`, `Flugzeug` …, nor their forms;
  `Tablette` is in that list too), in the table as well since its rebuild (0.17.0): the build keeps those words
  whole and cuts the other nouns as before, by Wikidata's analysis where it has one and else by the splitter's own
  judgement with stricter limits (0.12.0 had made every noun without an analysis one word; the table it would have
  built no longer split 15 of 33 everyday compounds, and was never shipped). The table of 0.17.0 was built without
  the verb forms and lacked 1,530 verbs that keep their forms whole: it cut 1,515 forms (`gehörte` as `geh|orte`,
  `stelltest` as `stell|test`); 0.17.1 is built with them, and verbs, adjectives and verb forms are read as in 0.16.3
  again. A real ambiguity
  stays: `Kopfschuppen` (dandruff) is `Kopf` + `Schuppen`, and `Schuppen` is also a shed. `de_compounds.tsv.gz`
  (212 KB, rebuilt 2026-10-17), loaded on the first German word (0.15 s, 7 MB); `_DE_COMPOUNDS = False` turns it off.
- **German and Spanish verb forms** (0.11.0): Snowball keeps `ging` apart from `gehen`, `fue` from `ir` and `ser`,
  `hizo` from `hacer`. A table of such forms from Wikidata's lexemes (the 2,000 German and 4,000 Spanish verbs a
  stand-in for frequency ranks first) gives a form its lemmas' stems, in text of that language only (not in a tie
  with English: "he went"). Such a match counts 0.6, like a root, and never recalls a fact alone: a form can belong to
  two verbs (`fue` is `ir` and `ser`), the source has errors (`tuviese` under twelve verbs, dropped: a form under
  more than three is left out; `gelegt` under `liegen`), and a name can be spelled like a form (`Lucía` / `lucir`).
  Forms spelled like another word are left out (`Macht`, `vino`, `cuenta`, `hecho`), and so are forms spelled like a
  regular form of another verb (`fuhren` / `führen`). Separable verbs are not joined (`rief … an` / `anrufen`). German
  text is told by its function words, and the list lacks `zum`, `ins`, `wohin`: a short German sentence without others
  stays undetermined and gets neither compounds nor verb forms. `de_es_forms.tsv.gz` (251 KB): German 0.03 s and 2 MB, Spanish 0.05–0.09 s and
  11 MB on the first word of the language; `_LEMMA_FORMS = False` turns it off.
- Dutch compounds are not taken apart (`_NL_COMPOUNDS`, off): on 10 questions that ask by the end of
  a compound and 7 that use that end in its own sense, it found 2 more right facts and recalled 4 more wrong ones
  (`huis` in `ziekenhuis`, `weer` in `brandweer`, `kast` in `koelkast`).
- **Words of the same root** (Russian and English): Snowball removes inflection but keeps derivations apart. Russian
  pairs come from a dictionary (0.10.0): DerivBase.Ru, its direct derivation links as stem pairs, kept when the pair
  has the shape of a derivation (`ночные` / `ночь`, `городской` / `город`, `учительница` / `учитель`, `дружеский` /
  `друг`; not `полный` / `пол` or `задача` / `дача`) — 46,756 pairs, Apache-2.0
  ([licence](LICENSES/DerivBase.Ru-Apache-2.0.txt)), loaded on the first Russian word (about 0.3 s, 12 MB). English
  pairs come from Open English Wordnet 2025 (0.11.0): its derivation and pertainym links between single words, as stem
  pairs of the same shape (`teacher` / `teaches`, `pay` / `payment`, `deliver` / `delivery`, `decide` / `decision`,
  `angry` / `anger`, `allergic` / `allergy`; not `Peter` / `pet`, `meters` / `met` or `classic` / `class`, which
  0.9.0's suffix rules joined). A link is kept only when it is the derived word's commonest sense (`butter` is food
  before it is one who butts, `tower` a building before one who tows, `letter` writing before one who lets) and no
  stem is another, unrelated word (`pearly` stems to `pear`) — 5,428 pairs, CC BY 4.0 on Princeton WordNet 3.1
  ([licence](LICENSES/OEWN-CC-BY-4.0.md), [WordNet notice](LICENSES/OEWN-WNDB-License.txt)), 30 KB, loaded on the
  first English word (0.04 s, 2.4 MB). Missing from it: `dentist` / `dental`, `anxiety` / `anxious`, `runner` /
  `run`, `rental` / `rent`, `medical` / `medicine` (left out by those filters); `pronunciation` / `pronounce` is in
  it but its spelling change is too large for the shape filter. `_EN_DERIVATION = "rules"` (and `_RU_DERIVATION`) restores 0.9.0's rules. Russian
  stems of six letters also match on their first five (`покупа` / `покупк`). Only in text whose language is
  determined (English by its function words; Cyrillic without Ukrainian or Belarusian letters), and only within that
  language. A root counts 0.6 of the word itself and **never recalls a fact on its own**: suffix rules cannot tell a
  derivation from a look-alike (`войной` / `вой`, `полный` / `пол`), so automatic recall needs a word of the message
  itself, and roots add to it. In `search` a root brings in facts the word's forms missed and ranks them below facts
  with the word. Not joined: alternating consonants and prefixes (`друг` / `дружба`, `страховка` / `застрахован`),
  four-letter beginnings (`большое` is not `больница`). Duplicate and conflict checks do not use roots.
- **Courtesy** (0.9.0): a message whose every clause is greetings, thanks, sign-offs, reactions or "how are you",
  at least one of them more than padding, recalls nothing ("Hi, good morning!", "Thanks, that's all for now",
  "спасибо, на этом всё", "Grazie mille, a domani!", "Wielkie dzięki, do jutra!", "Heel erg bedankt, tot morgen!",
  and any combination of the whole-message greetings of every language). Pieces start and end
  at word boundaries; padding alone ("fine?", "сегодня?") is a question. Russian question words (`какой`, `который`,
  `сколько`, `почему` …) are left out of a query's terms, like English `what` / `which` / `how`, but kept in facts.
- **Names** in Latin script are matched as written (and with an English `'s`): `Jean` is not mentioned by `jeans`.
  Names in other scripts carry their inflections (`Маши` mentions `Маша`, `لإيفا` mentions `إيفا`).
- **Chinese, Japanese, Korean, Thai** (no spaces between words): character n-grams, bigrams for Chinese, Japanese and
  Korean, trigrams for Thai. Grammar weighs less: Chinese function characters (的 是 了), Japanese particles (は が を
  に の), Thai function words (ของ ที่ และ); a message that is only a filler (はい, 네, โอเค) recalls nothing. Chinese
  characters (also in Japanese) count one by one as well; kana, Hangul and Thai letters, which are sounds, do not.
  Traditional characters compare as Simplified ones (0.9.0): `學校` finds `学校`, `醫院` finds `医院`, so a fact a Taiwan
  or Hong Kong user wrote meets a mainland user's question and the other way round; the stored text keeps its script.
  The table is OpenCC's (`cjk_fold.py`, 3,222 characters, Apache-2.0, [licence](LICENSES/OpenCC-Apache-2.0.txt)). The
  n-gram scan reads the facts' folded text, analysed once per fact: at 5,000 Chinese facts the first search after start
  takes about 1.5 s, later ones 0.12 s with recall (0.42 s in 0.8.0, whose scan ran in SQL).
- **Japanese spelling variants** (0.11.0): one word written in kanji or kana, with more or less okurigana, or as
  a loanword with or without the long vowel compares as its representative spelling: `子ども` finds `子供`, `引越し` and
  `引越` find `引っ越し`, `申込` finds `申し込み`, `コンピュータ` finds `コンピューター`, `ウィルス` finds `ウイルス`. Only in text
  that has kana, so Chinese text is never touched; the stored text keeps its spelling. The spellings come from
  SudachiDict's synonym source (`ja_variants.tsv.gz`, 5,167 variants of 4,011 words, Apache-2.0,
  [licence](LICENSES/SudachiDict-Apache-2.0.txt)), spellings of one word only: no synonyms, abbreviations (`スマホ` is
  not `スマートフォン`), translations or romaji. Japanese has no spaces, so a variant is replaced only where its edges can
  be word edges: hiragana next to a particle, punctuation, katakana or a suffix (`たち`, `さん`), not inside other kana
  or right after a kanji (`べんきょう` holds no `きょう`, `白い草` no `い草`); a short katakana word not inside a longer one
  (`アイス` holds no `イス`). Left out: hiragana of two letters (`いす`, `かに`), hiragana verbs Sudachi marks as ambiguous
  (`はなす`: 話す, 離す, 放す), words whose representative is one kanji (`たまご` / `卵`: a lone kanji has no bigram, so
  two facts that both wrote `たまご` would stop meeting), and `いつも` / `だいぶ` (`何時`, `大分` are other words). Verbs
  meet in the forms the source lists and through their noun (`引越しします`). Loaded on the first text with kana:
  0.03 s, 2 MB.
- **Chinese words** (0.11.0): character pairs cannot tell a word from the halves of two words, so text whose Han is
  Chinese (no kana, no Hangul) is also split into words, the way jieba cuts without its HMM: the most probable split
  by word frequency, a character that begins no word standing alone (`南京市长江大桥` is `南京市 | 长江大桥`, `我和服务员`
  is `我 | 和 | 服务员`). The words add to the pairs, they do not replace them. Where two dictionary words meet, or one
  meets a function character standing alone, the pair across that seam is read as such: a question's `和服` (kimono)
  is not found in `我和服务员` ("me and the waiter"), `研究生` (graduate student) not in `研究生命` ("to research life"),
  `校长` (principal) not in `学校长期`, and a character a fact holds only as the half of such a split pair does not count
  one by one either. Next to a content character standing alone the split is not sure enough: `看 | 牙医` still holds
  `看牙`, `续 | 保险` still `续保`. Every pair keeps its weight, so the share of a question a fact covers is counted as
  before. A one-character word is also searched alone, wherever it stands in a fact: `猫几岁了？` finds `橘猫`,
  `药一天吃几次？` finds `降压药`; not function characters, numerals, units of the calendar, the commonest words (`过`,
  `看`, `去`), characters jieba tags as a person name (most surnames), or two side by side in a question, which are more
  likely a name the dictionary lacks (`怡君`, `思琪`: found by their pair, as before). Traditional text is split after
  folding; Japanese and Korean text is compared as before. The dictionary is jieba's (`zh_words.txt.gz`, all 347,549
  words of its `dict.txt`, folded to Simplified, frequencies kept to a quarter of a natural-log step; 1.45 MB; MIT,
  [licence](LICENSES/jieba-MIT.txt); its author has said the frequencies were counted on the People's Daily 1998 and
  MSR segmentation corpora, which carry research terms of their own — the dictionary is used as jieba publishes it);
  the segmenter is the plugin's own, jieba is not a dependency. Loaded on the first
  Chinese text: 0.2 s, 30 MB. Splitting every fact once adds 0.3–0.7 s to the first search after start at 5,000
  facts; later searches are unchanged within measurement noise. `_ZH_WORDS = False` in `text.py` turns it off.
- **Arabic, Indic scripts, Indonesian, German, Turkish, Polish and Dutch** are also matched word by word against the
  text of the facts (so are Latin words of a question of undetermined language, when the store holds facts in one of
  those Latin languages): the SQLite FTS5 index splits Indic words at their vowel signs and cannot see past Arabic and
  Indonesian prefixes, into German compounds, through Turkish consonant changes (`kitap` / `kitabı`) and Dutch plurals
  that change the word (`raam` / `ramen`), or from `szkola` typed without its `ł` to `szkoła`. The rows are read on every such search; a fact is analysed once and again only when its text changes.
  Measured at 5,000 facts: 0.03–0.07 s per search and about the same for the relevance gate; the first such search
  after start analyses every fact, about 1 s at 5,000 facts and 4–6 s at 20,000 facts (later searches 0.2–0.4 s, plus
  0.15–0.2 s for the gate). A store without such facts never scans: an English store answers in about 0.02 s at 5,000
  facts, as before. A Turkish store of 5,000 facts (0.11.0): the first search after start 1.5 s, later search and
  recall together 0.25 s (0.11.0 scans it; 0.10.0 answered in 0.01 s and missed most other word forms). Whether a
  store holds such facts is told by a few common words of those languages; a question itself told to be in one of
  them scans in any case, so a store of short facts without those words is still read word by word. Not covered: a
  one-word question of no told language (`szkola`, typed without its `ł`) against such a store, and a short fact with
  no function word and none of its language's letters (`Ramen gesloten.`), which gets no Dutch, Polish, Turkish or
  Italian stems.

Measured on stores of 20 everyday facts per language (people, school, work, servers, plans). "Found" = the right
fact among the first five search results; "recalled" = injected before the turn; "quiet" = chit-chat, greetings and
off-topic questions recall nothing. No keywords in these stores.

| | Main set (20 questions): found / recalled | Unseen set (8 questions, 15 unrelated facts added): found / recalled | Quiet (all sets) |
|---|---|---|---|
| English | 100 % / 100 % | 100 % / 100 % | 14 / 14 |
| Russian | 100 % / 100 % | 100 % / 88 % ¹ | 14 / 14 |
| Chinese | 100 % / 100 % | 100 % / 100 % | 14 / 14 |
| Spanish | 100 % / 100 % | 100 % / 100 % | 10 / 10 |
| Hindi | 100 % / 100 % | 100 % / 100 % | 9 / 10 |
| Arabic | 100 % / 100 % | 100 % / 100 % | 10 / 10 |
| French | 100 % / 100 % | 100 % / 100 % | 10 / 10 |
| Bengali | 100 % / 100 % | 100 % / 100 % | 10 / 10 |
| Portuguese | 100 % / 100 % | 100 % / 100 % | 10 / 10 |
| Indonesian | 100 % / 100 % | 100 % / 100 % | 10 / 10 |
| Japanese | 95 % / 95 % | 100 % / 100 % | 10 / 10 |
| Korean | 100 % / 100 % | 100 % / 100 % | 10 / 10 |
| Thai | 100 % / 95 % | 100 % / 100 % | 10 / 10 |
| German | 100 % / 100 % | 75 % / 75 % | 10 / 10 |
| Italian ² | 100 % / 100 % | — | 5 / 5 |
| Turkish ² | 100 % / 100 % | — | 5 / 5 |
| Polish ² | 100 % / 100 % | — | 4 / 5 |
| Dutch ² | 100 % / 100 % | — | 5 / 5 |

The right fact is first in 95–100 % of main-set questions and 62–100 % of unseen ones. Recall before the turn is not
only the right fact: about one other fact of the store comes with it per question (0.8–1.2 on the main sets), and in
the unseen sets 0–2 of the 15 unrelated facts per language. English, Russian and Chinese also have a second set (10
questions): found 90 / 90 / 100 %, recalled 80 / 90 / 90 %. ² Italian, Turkish, Polish and Dutch (0.11.0): the main
set was written with the code; their blind sets are below.

How the numbers were obtained, honestly:

- **Main sets are tuned on.** English, Russian and Chinese: the fixes in 0.4.2 were made against them. Spanish, Hindi,
  Arabic, French, Bengali, Portuguese, Indonesian: first run on 0.4.2 (found 85–100 %, recalled 50–100 %, Arabic 50 %,
  French 75 %, Spanish 85 %, chit-chat recalled facts in four languages), fixed in 0.5.0. Japanese, Korean, Thai,
  German: first run on 0.5.0 (found 95–100 %, recalled 95–100 %, but chit-chat recalled facts in Korean 1/5, German
  1/5 and Thai 4/5), fixed in 0.6.0. The fixes are per language and per script, not per question.
- **Unseen sets were written before those fixes were seen to work and run once after them.** English, Russian,
  Chinese: after the run, whole-message greetings were made to recall nothing ("good morning" recalled facts about
  mornings), and the quiet column includes that fix. Spanish to Indonesian: first run before the review fixes below,
  with the same found and recalled numbers. Japanese, Korean, Thai, German: run once; nothing changed after.
- **An independent review (Codex CLI) of 0.4.2 and 0.5.0** found regressions the sets did not: a lone "son" dropped as
  a Spanish function word, `hr-ovh` no longer enforced, `ação` / `ações` apart, "with" / "without" sugar no longer a
  conflict, `Jean` mentioned by `jeans`, and others. Each is fixed with a test, and every set was run again. One of
  those fixes made French `est-ce` a required identifier, which the unseen set showed (French recalled 88 %); the
  remedy (compound words in Romance-language text, a hyphenated word votes for a language with its first part) was
  made after seeing it, so that unseen-set number is no longer blind.
- **The second set was written after the main set of 0.4.2**, first run without changes (Russian recall 90 %, Chinese
  70 %), and two fixes followed. Its numbers published with 0.4.2 (recalled 90 / 100 / 100 %) were partly luck: facts
  of a named person saved in the same second came back in arbitrary order. They now come back newest first, every
  time, and the numbers above are what that gives.
- **Words of the same root and courtesy (0.9.0)** were measured on blind sets written by separate agents that never
  saw the code. Same-root words, two sets each for Russian and English (24 facts, 20 unrelated facts that share
  beginnings or roots with them, 20 questions — 12 asked with another word of the fact's root, 4 with another form, 4
  with a synonym — and 10 traps). The first set was run once before and once after; its misses were read and three
  faults fixed, so it no longer counts. On the second: Russian first 45 → 55 %, found 55 → 60 %, recalled 55 → 55 %;
  English first 60 → 60 %, found 70 → 75 %, recalled 60 → 60 %. Then an independent review (Codex CLI) found
  look-alike joins that these sets had no trap for (`войной` / `вой`, `mothers` / `moth`, Spanish `comer` / `come`);
  since then a root never recalls on its own, roots stay within a determined language, and the second set gives the
  same found and recalled numbers. A second review found padding-only courtesy ("fine?") and question words lost from
  duplicate checks (fixed) and a capital-letter name rule that lost ordinary words ("Baby" / "babies"; removed — on
  every set it changed nothing). Small talk: a fresh set per language (24 facts that use small-talk words, 30
  small-talk messages, 15 questions), run once on the final code: quiet English 50 → 67 %, Russian 63 → 70 %,
  questions found 100 % before and after; what still recalls is a word in another sense or a famous name ("Peter
  Pan", "Лев Толстой", "скрипичный ключ"), which matching words cannot tell apart. Every earlier set gives the 0.8.0
  numbers, except ¹ the Russian unseen set: its one miss ("во сколько ночные бэкапы", nightly / night) is now found,
  not recalled (a root alone); that question is the one the change was written for, so this number is not blind.
  Speed at 5,000 facts is unchanged (0.03 s for a search and a recall).
- **Traditional and Simplified Chinese (0.9.0)**: two blind sets by separate agents (24 facts, half in each script; 20
  unrelated facts; 20 questions — 6 in the fact's script, 6 in the other one, 4 paraphrases, 4 synonyms — and 10 small
  talk). On the first, a question in the other script was found but recalled in 3 of 6; its misses were read and the
  folding written. On the second, run once on the final code: other-script questions found and recalled 33 → 100 %,
  all questions first 55 → 75 %, found 55 → 75 %, recalled 50 → 70 %, small talk quiet 80 % before and after;
  same-script questions 100 % both; synonyms with no shared character 0 % both (keywords cover them). Folding also
  makes a same word in another sense meet across scripts ("圍棋是誰發明的？" recalls the family's Go lessons, written
  `围棋`), as it always did within one script.
- **Russian dictionary (0.10.0)**: a fresh blind set (12 same-root questions, 4 other forms, 4 synonyms, 12
  look-alike traps), run once: same-root questions found 75 % (0.8.0) → 83 % (0.9.0 rules) → 92 % (dictionary),
  recalled 42 → 67 → 67 %; traps quiet 11 of 12 in all three. The choice of the dictionary was made on the earlier,
  already read sets, before this run.
- **0.11.0** was measured on fresh blind sets, one per language, written by separate agents that never saw the code
  and run once on the code of that moment against 0.10.0; every choice (dictionary or rules, which pairs to keep, what
  to leave out) was made before that run, on the sets above or on development sets written with the code. Then an
  independent review (Codex CLI) found false matches the sets had no trap for (below); after its fixes the same sets
  were run again, and those second numbers are no longer blind.
  - English same-root words (Open English Wordnet; 24 facts, 20 look-alike distractors, 12 same-root questions, 4
    other forms, 4 synonyms, 12 look-alike traps): same-root questions found 100 % both, recalled 75 → 83 %; all
    questions recalled 65 → 70 %; traps quiet 12 of 12 both. After the review's filters (commonest sense, no
    unrelated stem) the same-root questions are recalled 75 %, as with 0.10.0's rules: the filter that drops
    `butter` / `butt` and `letter` / `let` also drops `medical` / `medicine`. On the earlier English sets only one
    question moved:
    "When is Oliver's dental checkup?" is still recalled but no longer first (the dictionary has no `dentist` /
    `dental`).
  - Japanese spelling variants (24 facts, 20 distractors that share characters or kana with them, 14 questions with
    the word in another spelling, 6 in the same, 4 synonyms, 8 traps): other-spelling questions found 57 → 71 %,
    recalled 29 → 43 %; all questions first 46 → 54 %, found 71 → 79 %, recalled 46 → 54 %; traps quiet 8 of 8 both.
    After the review's fix (a hiragana variant starts after hiragana only when that is a particle after a kanji or
    katakana word: `かいちょう`, chairperson, held `いちょう`, ginkgo): other-spelling questions found 64 %, recalled
    36 %.
    Still missed: some hiragana spellings (`はいしゃ`, `りんご`), one-kanji words (`ネコ` / `猫`, left out on purpose),
    and questions found but held back by the recall gate.
  - Chinese words (24 facts, 20 distractors, most holding a fact's word across a seam or inside another word; 10
    questions by a shared word, 4 paraphrases, 4 by a one-character word, 4 synonyms, 10 traps): first 59 → 73 %,
    found 64 → 77 %, recalled 41 → 41 %; one-character words found 0 → 50 %, recalled 0 % both; traps quiet 6 of 10
    both. Splitting moves the right fact up and finds one-character words; it does not lift recall, and traps that
    share a real word with a fact (`研究` in both `研究生` and `研究生命`) still recall it. The two Chinese sets of
    0.9.0 (read since): first 65 → 80 % and 75 → 80 %, recalled 70 → 70 % and 70 → 75 %.
  - Italian, Turkish, Polish, Dutch (each: 22 facts, 18 distractors, 10 questions in another word form, 6 in the
    same, 4 synonyms, 5 small talk, 5 traps), recalled / small talk and traps quiet: Italian 80 → 85 % / 50 → 90 %;
    Turkish 35 → 85 % / 70 → 50 %; Polish 55 → 75 % / 60 → 70 %; Dutch 70 → 85 % / 70 → 60 %. Other-form questions
    recalled: Italian 90 → 100 %, Turkish 10 → 100 %, Polish 50 → 90 %, Dutch 60 → 90 %. Turkish recalled more where
    it should and more where it should not: "Günaydın canlarım!" (good morning, my dears) met a boy named Can
    (`canlarım` and `Can` share a stem), and "Tamam, akşam görüşürüz" (see you tonight) was not read as a sign-off;
    terms of endearment and times of day now pad a Turkish courtesy, and the set gives quiet 70 % (not blind).
    Known weak spots of the stemmers: Turkish verbs are not reduced (`okuyor` / `okudu`), and `yazdı` (wrote) meets
    `yaz` (summer); Polish consonant changes (`kotka` / `kotce`) and the moving `e` (`cukier` / `cukru`); Italian
    `chiavi` → `chi`; Dutch irregular plurals (`kind` / `kinderen`).
  - German compounds and verb forms (22 facts, 18 distractors; 6 compound questions, 10 other forms, 6 same, 4
    synonyms, 10 small talk and traps): first 62 → 69 %, found 73 → 81 %, recalled 65 → 77 %; compound questions
    found and recalled 67 → 83 %, other forms recalled 70 → 90 %; quiet 8 → 7 of 10 (`Kopfschuppen`, above).
    Spanish (6 irregular-verb questions among 26): no change at all, 73 % recalled before and after — its irregular
    questions shared another word with their facts and were found by 0.10.0 already, so this set says nothing about
    the forms table either way; the table rests on development sets (first 72 → 84 %).
  - Every earlier set gives the 0.10.0 numbers in every language, except the Chinese ones (above), the English
    dental question, and German, whose synonym set's questions are found 25 → 50 % (88 % with keywords); the Russian
    sets are unchanged.
- **Review of 0.11.0 (Codex CLI)**, each fixed with a test (`tests/test_holographic_v0110_review.py`): Polish stems in
  text of no determined language joined English words (`army` / `arm`; now only with Polish letters); a host name in a
  Turkish, Italian or Dutch question was no longer required (`hr-ovh'de`); a hiragana variant was replaced inside
  another word (`かいちょう`); English links of a rare sense were applied to the common one (`butter` / `butt`, `tower`
  / `tow`, `pear` / `pearl`); a Chinese one-character word recalled an idiom (`猫` / `猫腻`; now it needs a pair of the
  question or a second character); a kanji-only Japanese question missed a fact whose kana folded its spelling (a
  fact keeps its spelling as written too); Turkish, Polish and Dutch stores without the probe words were not scanned;
  "Günaydın canlarım!" recalled facts about Can; a damaged data file broke recall instead of being skipped. A second
  review (of German and Spanish, and of these fixes) found verb forms recalling facts alone (`tuviese` / `contiene`,
  `gelegt` / `liegt`, `Lucía` / `luce`, `fue` / `va`: now 0.6, never alone), a three-letter compound head doing the
  same (`Alltag` / `Tag`), a host name lost in a Dutch tie with Spanish (`De server hr-ovh`: compounds only when every
  tied language writes them), and a malformed compound row raising instead of being skipped; each has a test. The
  German set gives the same numbers after these fixes, quiet 8 of 10 again; every other set is unchanged.
- **Not measured**: other languages (exact words, no stems, no function-word list): among them Vietnamese, Ukrainian
  (compared with Russian stems), Korean beyond its n-grams.

### Synonyms and other roots: keywords

Words are matched, not meanings: "relocating" does not find "move", 宠物 (pet) does not find 猫 (cat); words of one root
are joined only in Russian and English (see Languages). The plugin calls
no model, but the agent that stores a fact is one: `fact_store add` takes `keywords`, 3–8 words someone might ask
with that the fact does not contain (synonyms, other roots, broader terms, the same in the user's other languages).
They are kept with the fact's tags, searched like its text, never shown in the recalled block, and old versions index
them too. This is document expansion, as doc2query does for lexical search: the meaning is added once, when the fact
is written, and recall stays local and deterministic. `enrich` lists the facts that have none, so the agent or a
nightly job can add them with `update`; an update with keywords alone keeps the fact's age (`updated_at`), so old
facts do not turn into recent ones. [hermes-dreaming](https://github.com/itpartypattaya/hermes-dreaming) does this
nightly (`keywords.enabled`).

Measured on a third set per language: 8 questions that share no word with their fact except a name ("Where are we
relocating?", "What breed is our pet?", "Can my office computer handle AI models?"), written after the keywords. The
keywords came from a separate agent that saw only the facts and the description of `keywords`, not the questions; a
first batch was thrown away because the description's example then was the store's own "we move to Chiang Mai".

| | Without keywords: found / recalled | With keywords: found / recalled |
|---|---|---|
| English | 38 % / 38 % | 100 % / 100 % |
| Russian | 25 % / 25 % | 75 % / 62 % |
| Chinese | 38 % / 12 % | 88 % / 75 % |
| Spanish | 38 % / 25 % | 100 % / 100 % |
| French | 38 % / 38 % | 100 % / 100 % |
| Portuguese | 38 % / 38 % | 75 % / 75 % |
| Indonesian | 38 % / 25 % | 88 % / 88 % |
| Arabic | 38 % / 38 % | 75 % / 75 % |
| Hindi | 38 % / 38 % | 88 % / 75 % |
| Bengali | 38 % / 38 % | 75 % / 75 % |
| Japanese | 50 % / 25 % | 88 % / 75 % |
| Korean | 50 % / 12 % | 88 % / 62 % |
| Thai | 38 % / 25 % | 75 % / 62 % |
| German | 25 % / 38 % | 62 % / 75 % |
| **All 112** | **42 / 33** | **94 / 88** |

Without keywords, what is found comes through the name in three of the eight questions (Eva, Anton). With them, the
main sets lose nothing (280 / 280 found and recalled), one more extra fact comes per ten questions, and one chit-chat
line of 70 recalls a fact ("Apa ibu kota Australia?" shares "kota", city, with a keyword "ganti kota"). The misses
left are words the keywords did not include ("перебираемся" against "переселение", "morar" against "moradia"):
keywords are as good as the agent that writes them, and facts stored without them (older ones, `auto_extract`,
`mirror_memory_writes`) are found by their words only until `enrich` gives them some.

What it does not do: find a fact through a synonym or another root that no keyword names ("бэкапы" for "резервные
копии", Spanish `nada` for `natación`, 十月 for 10月). It can surface a fact that shares a word in another sense:
"nightly backups" → "night market"; Hindi `बनाना` is both "to cook" and "to make a plan"; Portuguese `dirigir` is "to
drive" and "to run a company". One shared word is not enough when other facts hold it too: "What day is today?" recalls
none of the facts that happen "every day" (0.17.8), but when one fact alone says "through the week", "How's the
weather this week?" still recalls it, as "wifi password?" recalls the one fact about a password: one word cannot tell
the two apart. In a script without capitals (Hebrew, Hindi, Arabic) a name cannot be told from a common word, so
there one shared word recalls as it did. Stems are search keys, not lemmas: an irregular form can miss (`ponies` / `pony`). A
negation counts for the whole clause up to a "but" ("likes cats but not dogs" is detail, not a conflict), not for a
single word. `auto_extract` recognises English, Russian and Chinese phrasing. The refusal of instruction-like text, of a
credential with its value and of a request to send one covers all eighteen languages (0.15.1; English and Russian
before). The
main sets of all eighteen languages run in CI (`tests/test_holographic_languages.py`).

## Search by meaning (optional, online)

Off by default, and the one part of the plugin that goes online. Words, keywords and HRR stay the recall; this adds
an embedding model you choose for the questions that share no word with their fact ("Which machine copies our data
while we sleep?" against "The nightly backup runs from the OES box at 2 am."). It lives in its own module
(`embeddings.py`) and needs no new dependency (the standard library's HTTP client, numpy as for HRR). With it off
there are no calls and no vectors, and search scores (to six decimal places) and recall are 0.13.0's — a test
pins recall on six English facts and four queries (``test_recall_is_identical_to_main``); what
else 0.14.0 changes is the prompt line (see the CHANGELOG), a `search_by_meaning` field in `stats`, and the empty
`fact_embeddings` table a writable session creates.

```yaml
plugins:
  hermes-memory-store:
    embeddings: on
    embedding_url: https://api.openai.com/v1   # any OpenAI-compatible embeddings API
    embedding_model: text-embedding-3-small
    embedding_key_env: OPENAI_API_KEY           # the NAME of a variable (e.g. in ~/.hermes/.env), never the key
```

| Endpoint | `embedding_url` | `embedding_model` | Key |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-small` | `OPENAI_API_KEY` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-embedding-001` | `GEMINI_API_KEY` |
| A local server with an OpenAI-compatible API (Ollama, Text Embeddings Inference, Infinity) | e.g. `http://127.0.0.1:11434/v1` | the model it serves | none needed |

The first two were measured (below); a local server goes through the same API and was tested against a local test
server only. Other keys: `embedding_dimensions` (shorter vectors — only for a model that can shorten them, like
OpenAI's `text-embedding-3-*`; leave it unset otherwise: a server may answer 400 to the field rather than ignore it,
as bge-m3 behind an aggregator does, and the plugin's status then says so), `embedding_timeout`
(3 s, at most 6: the core gives a provider's recall 8 s), `embedding_min_score` (an absolute floor of similarity; off),
`embedding_prefetch_facts` (1), `embedding_weight` (0.3), `embedding_space` (a label of your own: vectors are kept
per address, model, dimension and this label, so when a server starts serving another model under the same address
and name, change it and every vector is made again), `embedding_catch_up_seconds` (8, from 2 to 60: how long the
fact vectors after a turn may take; below).

A store that had embeddings from another deployment may carry a column of its own vectors in `facts` (e.g. a
`facts.embedding` BLOB). Since 0.14.3 no column the plugin did not create reaches a tool result (0.14.2 dropped
binary values only); nothing reads it either, and `ALTER TABLE facts DROP COLUMN embedding` (SQLite 3.35+) removes
it — unless an index or trigger of the old deployment uses the column: drop those first.

**How it decides**

- *Fact vectors* are made after a turn, on the core's background worker, by a session that may write (the main
  agent, and a cron job unless `cron_writes: false`; never a subagent): requests of up to 32 facts within
  `embedding_catch_up_seconds` (8 s, at most 32 requests), so a store catches up within a few turns (100 facts: 4
  requests to a hosted API, about 0.3 s each) and no reply waits for it in the core's usual path. The first request to
  an endpoint carries 4 facts, and each answer's time sizes the next batch: as many facts as that speed fits into half
  of the time left — on a local server at most into half of `embedding_timeout`, since such a server often answers one
  request at a time and a message's vector before a turn may wait behind the batch — and one fact whenever one is
  expected to take less than the time left. Nothing is sent that would not fit, but for the turn's first request when
  it has most of the budget (a wait for the provider lock can take the rest). A batch that runs out of time is no
  failure — the server may be slow, not down: the next batch is smaller (a quarter, at the same time left); a failure
  is one fact that runs out of the turn's first request, or 3 batches in a row (0.16.3, issue #5: a 4B model on a
  local NPU server took 7.7 s for 32 facts against the 7 s left, and 0.16.2 sent those 32 again after every pause
  without ever storing a vector). With that server's numbers, 119 facts of 570 characters fill in 5 turns, about 25 a
  turn (a fake endpoint at its speed, not the server itself); a reviewer's model puts 2,000 facts at about 72 turns at
  8 s and about 10 at 60 s. Speed is measured from sending, and not from a batch a message was sent during (on a
  one-slot server it waited with it); a fixed cost per request still counts as speed, which only makes batches
  smaller. A longer `embedding_catch_up_seconds` fills a store sooner and delays no reply, but the core runs every
  provider's after-turn work on one worker in turn, so the next turn's extraction waits for it, and at shutdown the
  core waits 5 s for that worker: what comes back after that is not stored (the store is checked before anything is
  written), though the request itself is not called back. `fact_store stats` shows the budget and what the endpoint's last batch did
  (`after_turn`; the endpoint is shared by every store a provider opens). A vector is stored in its own
  table, `fact_embeddings`, with the model, the address (a query string by a short hash of it) and a hash of the
  text it was made from: a changed text, another model, address or `embedding_space` gets a new one, and `forget`, `end` and `remove` delete it with the fact.
- *Before a turn* one request embeds the message — unless it is a greeting or a thank-you, has fewer than two words
  that carry meaning (five characters of Chinese, Japanese, Korean or Thai), looks like it holds a secret (below), or
  no fact has a vector yet. Meaning then reorders the facts that share words with the message (`embedding_weight` of the score),
  and may add **one** fact that shares none (`embedding_prefetch_facts`), after the words' facts and marked
  `(by meaning)`, when it stands far out: its similarity is at least `sqrt(2 ln N) + 1` standard deviations above this
  message's mean over the N facts with a vector (the closest of N unrelated facts reaches about `sqrt(2 ln N)` by
  chance, so the bar rises with the store; it needs 20 vectors, below that only `embedding_min_score` admits a fact),
  and it mentions whoever the message names — by a heuristic: every capitalized word inside a sentence and every
  known entity the message mentions. In meaning "When is Peter's birthday?" is as close to "Lena's birthday is on
  June 3" as a paraphrase is to its fact; only the name tells them apart. The heuristic is off in German (every noun
  is capitalized), sees no names in Chinese, Japanese, Korean or Thai, nor a name that opens a sentence or is written
  in lower case, and turns away a paraphrase that names a place its fact words differently ("Japan" for a fact about
  Tokyo).
- The bar is a heuristic too, not a guarantee. Similarities in one store are not independent: when many facts are
  about as close to a message as each other (a cluster — twenty facts about one child), the spread grows and fewer of
  them clear the bar. Measured: 20 more facts about two people of the set cost 6 of the 52 paraphrases that
  cleared the bar on text-embedding-3-small. A median-based spread let more paraphrases through but also more noise (13
  of the 45 questions that must recall nothing got past the bar alone, against 8), so the mean and standard
  deviation stay.
- `fact_store search` adds the closest facts by meaning after every word match when there are fewer than `limit`,
  with `"match": "meaning"` and `"similarity"` (bar `sqrt(2 ln N) + 0.5`: the agent asked, and judges). And
  even when the words fill the limit (0.14.4), it shows those of the 3 facts closest in meaning that are closer than
  the best fact the words found by more than 0.06 of cosine, stand at least 1 standard deviation above the store's
  mean for the query, and mention whoever the query names (the name check recall uses): words can fill the list
  while the fact asked for is the store's closest in meaning.
  Measured on a family store (58 facts): questions sharing no word with their fact but a name, 13 → 18 of 27; no
  word at all, 3 → 6 of 12; questions that must find nothing still got a fact by meaning in 1 of 30. Taking the 3
  closest by rank alone found as many but answered 20 of those 30 with one. Recall before a turn does not do this.
- Any failure — network, HTTP status, a timeout (`embedding_timeout` for the whole request before a turn; fact
  vectors after a turn share `embedding_catch_up_seconds` with every wait for the provider lock, as no reply waits
  for them, and count as above;
  either way the request, decoding included, runs on a thread of its own and the caller stops waiting then, however
  slowly the server sends), a malformed answer, an HTTP status that comes after the caller stopped waiting — falls
  back to words without a word to the user; the endpoint is then left alone for 60 s, doubling up to 15 min, with one
  warning in the log. One request moves the pause one step at most, however many ways it fails; an answer to a
  request sent before the latest failure does not end the pause. While a request given up on is still running no new
  one starts; the next one waits for it up to half of its own time first (the answer to a batch cut short often comes
  a moment later), so a message's network waits share one `embedding_timeout`. A message's request that runs out of
  time while a fact batch of ours held the endpoint for part of its wait (the batch was running, or ended after a
  quarter of that wait) is no failure: recall uses the words that turn, and the endpoint is not paused. A batch of
  ours counts as holding the endpoint for twice `embedding_timeout` after it was given up on; a request still running
  after that is a stalled server, and fails (the status says so from then on). A redirect is a
  failure too (it would carry the key elsewhere). `fact_store stats` shows `search_by_meaning`: the status, the model,
  and how many facts have a vector, wait for one, or are never sent.

**What leaves the machine.** To the endpoint you name: the text of every fact (once, again when it changes) and of
every message and `fact_store search` query that passes the checks above. Fact texts go out after a cron job's turn
too while `cron_writes` is on: a job that may write also makes missing fact vectors. The key is read from the
variable `embedding_key_env` names — `OPENAI_API_KEY` when it names none — and sent as a Bearer token in the
Authorization header to whatever `embedding_url` is configured, so point the URL only at a provider you mean to give
that key to (a key with a space or a line break inside is refused, and errors are logged as categories, never with
the request in them).
Not sent: a fact recall would not inject (instruction-like text, markup, an image — the same check recall makes), and
a fact or a message that looks like it holds a secret — a credential word with its value, an environment assignment
such as `OPENAI_API_KEY=…` (not a variable named for a file that holds a file name: `SSL_KEY_FILE=server.key`,
`API_KEY_PATH=keys/prod.json`; 0.17.2), an Authorization header, the prefixes of common API keys and tokens (`sk-`, `ghp_`,
`AKIA`, `ASIA`, `AIza`, `xox…-`, `glpat-`), a Telegram bot token, a JWT, a PEM private key, a long encoded blob; of a message, only the first 2,000
characters are checked — and sent. That check is by pattern, so best effort: a secret in a shape
it does not know goes out with the message. `forget` deletes the vector, but the provider has seen the text: what it
keeps is up to its terms, and some free tiers may use what they receive to improve their models. The plugin cannot
tell whether a vector belongs to the text it sent: an endpoint or aggregator that mixes up concurrent answers gives a
search another text's closeness, silently (a user saw one fact by meaning carry the cosine of a parallel query,
issue #3; the plugin's own side is covered by a test). A local server keeps
everything on the machine; plain `http` is accepted only to localhost, a private or link-local address, a host name
without a dot or one ending in `.local`, `.lan`, `.internal` or `.home.arpa` (judged by the name; the address it
resolves to is not checked), and needs no key. A redirect is refused. With `scope: user`, `chat` or `thread` every
database has its own vectors. `embeddings: off` again: the next writable session deletes every stored vector.

**Cost.** One request per turn for the message, one per 32 facts written; at OpenAI's price for
text-embedding-3-small ($0.02 per million tokens) a thousand facts cost well under a cent. One request adds about
0.2 s (OpenAI) to 0.3 s (Gemini) before the reply (median of 12 requests from a VPS). Vectors are kept in
memory: 4 bytes per dimension per fact, 6 KB at 1536 dimensions (60 MB at 10,000 facts); `embedding_dimensions`
shrinks them.

**Measured** (2026-10-14; synthetic, nothing real: 100 facts in English, Russian, Chinese, German and Spanish; 67
questions that share no word with their fact but a name; 45 that must recall nothing; default settings):

| | Words only (off) | + meaning: text-embedding-3-small | + meaning: gemini-embedding-001 |
|---|---|---|---|
| Paraphrases recalled before the turn | 22 / 67 | 54 / 67 | 59 / 67 |
| A wrong fact recalled by meaning | — | 0 | 2 |
| Chit-chat that recalls a fact by meaning | — | 0 / 13 | 0 / 13 |
| Off-topic questions that do | — | 1 / 22 | 2 / 22 |
| Questions about someone or something not stored | — | 2 / 10 | 2 / 10 |
| Facts recalled on meaning alone that were the right ones | — | 29 of 32 (91 %) | 35 of 41 (85 %) |

The off-topic ones sit next to a stored topic ("Сделай план тренировки на неделю" → "Антон занимается в спортзале
три раза в неделю"). The others are names the rule cannot see — Chinese has no capitals (小王 / 小李), German
capitalizes every noun, so the rule is off there — and a related thing in another language ("Какой пароль от роутера?"
→ "Der Router steht im Flur."). With 50 facts text-embedding-3-small recalls 25 of 35 against 12 by words. Keywords
(above) stay worth having: they are local, free, and work with this off.

**On a real store** (2026-10-14, 0.14.1): a family assistant's 58 facts, mostly Russian, each with keywords from
[hermes-dreaming](https://github.com/itpartypattaya/hermes-dreaming)'s nightly job; a copy of the store, questions
written by a model from the facts, only numbers kept, and a run counted only if no request to the endpoint failed.
Natural questions: 57 of 58 recalled, with words alone and with meaning alike. Questions sharing no word with their
fact but the name of the person they ask about: 7 of 27 by words, 8 with meaning. Questions sharing no word at all:
0 of 12 either way. No wrong fact, and nothing from chit-chat, off-topic questions or questions about people not
stored (0 of 30), came in by meaning. On this store keywords and names do most of the work and the default bar added
one recall in these questions; a lower bar on the same questions: margin 0.5 added two with nothing wrong, 0 added
four but brought two wrong facts and three of the 30 other questions. The synthetic table above, whose facts carry no
keywords, is a separate benchmark; one store does not show what another will do.

## Rescoring by a cross-encoder (optional, online)

**Experimental — it does not raise hits.** The gate agreed on in
[#9](https://github.com/itpartypattaya/hermes-plugin-holographic/issues/9) did not pass on either set
(branch A needs +5 points and there are none; branch B allows no lost target and each run loses at
least one — the embeddings-on hold-out loses two),
so the option ships **off by default** and states its trade plainly. Three independent runs, all with
``jina-reranker-v2-base-multilingual`` Q4_K_M, ``rerank_scale: logit``, window 0.05, weight 0 — the
maintainer's held-out set twice, with embeddings off and on (bge-m3 Q4_K_M vectors from
``gpustack/bge-m3-GGUF``, sha256 ``6d39681b…a82fc06``, llama.cpp b11836, CPU — all 199 facts
vectorized per language):

| run | hit@5 off → on | non-target in top 5 | lines/turn | targets lost / gained | confused rows |
|---|---|---|---|---|---|
| this PR's test split (90 queries) | 43 → 43 | 108 → 66 (−39 %) | 1.98 → 1.27 | 1 / 0 | 6 → 4 |
| the maintainer's hold-out (126 queries) | 64 → 64 | 249 → 154 (−38 %) | 2.24 → 1.44 | 1 / 1 | 24 → 18 |
| the maintainer's hold-out, embeddings on (126 queries) | 81 → 81 | 251 → 150 (−40 %) | 2.38 → 1.53 | 2 / 1 | 24 → 18 |

*lines/turn in all three rows is per real question. The hold-out figures 1.84 → 1.21 and 1.97 → 1.30*
*quoted earlier were an average over all 195 rows, chit-chat included (denominator pointed out by the*
*maintainer, 2026-10-24); per question: 2.24 → 1.44 and 2.38 → 1.53.*

With it **on**, a third fewer lines reach the model and ~40 % fewer non-target facts, with fewer
confusions in all runs and noise (chit-chat, no-answer) unchanged; the cost is no gain in hits and
about one target per hundred questions — one in the 126-question emb-off run, two in the emb-on rerun —
lost on questions holding several facts of one kind (``q055``, ``h42``, ``h34``). With it **off**, nothing changes — the off rows match ``main`` (195 of 195 on
the hold-out set), and failed rerank calls were 0 in all three runs.

Off by default (``rerank: off``): words, keywords, HRR and, when on, meaning decide as above, with 0.17.8's scores and
recall — a test pins them on six English facts and four queries (``test_recall_is_identical_to_main``,
golden blocks per mode). Turn it on to add a second stage: a
cross-encoder reranker (Cohere's ``/v1/rerank`` shape; OpenAI has no rerank API) reorders the candidate pool before a
turn, after the blend. It lives in its own module (``rerank.py``), needs no new dependency, and makes one synchronous
HTTP call at recall time; a failure or a slow answer only keeps the current order and pauses the endpoint (60 s
doubling to 900 s). One mode since the 2026-10-22 review: the two designs were compared in
[#9](https://github.com/itpartypattaya/hermes-plugin-holographic/issues/9) and collapsed into one —
``rerank_mode`` and ``rerank_margin`` are gone (a config that still carries them is ignored with a warning):

```yaml
plugins:
  hermes-memory-store:
    rerank: on
    rerank_window: 0.05              # δ: cut the tail below this distance of the best score; 0 = keep all
    rerank_url: http://127.0.0.1:8080/v1/rerank   # llama-server --reranking --embedding --pooling rank
    rerank_model: local-reranker       # the endpoint's --alias
```

The cross-encoder reorders the pool; **membership stays with coverage** (the words' own criterion, exactly as in
off), so the reranker never adds a fact the words would not admit. It can still lose one they did admit: with
``rerank_window`` > 0 the tail below δ of the best score is cut before the top five is taken by rank — if the
cross-encoder is confidently wrong at the top, a fact the words admitted disappears instead of slipping places.
The cut never drops below two scored candidates — that minimum is built into the window itself (review of
2026-10-23), so a confidently wrong top cannot squeeze recall down to a single fact. Meaning- and entity-based
facts compete in the same rank as the lexical pool. A failed call falls back to the pool's blend order.

Other keys: ``rerank_top_k`` (40: how deep a pool the cross-encoder sees; the first stage's limit is widened to
match), ``rerank_weight`` (0 = order-only, recommended; >0 = the min-max-normalised cross-encoder score blended into
the blend score; the window applies to order-only runs), ``rerank_timeout`` (3 s — additionally capped so that ``embedding_timeout + rerank_timeout`` stays under
the 8 s the core gives a provider's prefetch: with ``embedding_timeout`` at its maximum 6 s the rerank
timeout clamps to 1.5 s) and ``rerank_scale`` —
the endpoint's score scale, set explicitly: ``logit`` (raw llama-server logits; the default, no probing) or
``unit`` ([0,1], Cohere-style). Set ``unit`` for an endpoint that already returns [0, 1] (Cohere, TEI): on the
default ``logit`` the sigmoid runs on scores that were normalised once already, the gaps shrink and the window
keeps more than it should. A value outside that pair is a config problem that names the key and turns
memory off for the session, like any other out-of-set value — never a silent fallback.
A query that holds a secret is never sent.
With ``rerank_weight: 0`` the cross-encoder judges text only and ignores trust: a fact whose trust
was lowered (by feedback or by a dreaming plugin) can move back to the top of what is admitted.

Measured on a real llama-server with bge-m3-reranker-v2 (Bge-M3-568M-Q4_K_M), three languages (en/ru/zh),
60 questions and 15 that must recall nothing (``tools/benchmark_rerank.py --endpoint=…``). The first table is
the PR author's CPU box; the same run on a GTX 1050 Ti (2026-10-22) gives p50 32–35 ms / p95 47–51 ms for the
four rerank modes and p50 4 ms off — the same order of cost, and recall/wrong/noise identical:

| mode | recall@5 | wrong-entity | noise | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|
| off | 100% (60/60) | 0 | 0% | 4 | 8 |
| window off, weight 0 | 100% (60/60) | 0 | 0% | 38 | 52 |
| window off, weight 0.5 | 100% (60/60) | 0 | 0% | 40 | 52 |
| window 0.05, weight 0 | 100% (60/60) | 0 | 0% | 38 | 51 |
| window 0.05, weight 0.5 | 100% (60/60) | 0 | 0% | 37 | 51 |

All five rows were measured on the **earlier design**, before the 2026-10-22 collapse into one mode
(`reorder_before_gate` = window off, `gate_reads_score` = window 0.05) — they are not the current result:
the head-commit table is in `results/BENCHMARK.md`, and the labels in it keep the historical names.

This store's recall was already 100% before the turn (its data are calibrated to ≥95%): the reranker keeps it there.
The honest number this table adds is the per-turn cost — 3-4 ms to 40 ms median, ~52 ms p95 on the author's box,
32–35 ms median / 47–51 ms p95 on a GTX 1050 Ti — the price the maintainer's acceptance asked to see next to the
gain. On CPU (`results/bench_cpu_t4.jsonl`, the head): llama-server with the same model, capped at
**threads 4 on a Ryzen 7 2700** — 176–184 ms median / 393–413 ms p95 per call, 0 fallbacks. The earlier
CPU numbers (94–101 ms median / 212–229 ms p95) were measured on a **different processor** — a
Ryzen 7 5800H laptop, `7f1efd6`, before the words-gate fix (`results/bench_p8_cpu.jsonl`): same model
and harness, not comparable across machines — see docs/contrast-benchmark.md. A second reranker of
different quality and a larger public set are run before merging (issue #9).

Choosing a reranker model: ``jina-reranker-v2-base-multilingual`` Q4_K_M is the CPU recommendation — p50 121 /
p95 152 ms at 4 threads (620 calls per model, ``results/bench_cpu_gate_jina.jsonl``) — and it is
**CC-BY-NC-4.0: non-commercial use only** (both upstream ``jinaai`` and the ``gpustack`` GGUF). For work that
must stay commercial, use ``bge-reranker-v2-m3`` (Q4_K_M, **Apache-2.0**): the same CPU run gives p50 343 /
p95 448 ms at 4 threads — it misses the 300 ms bar on CPU and stays the GPU default, but its licence is safe
for commercial use. The plugin ships neither model.

## Data and privacy

- Facts live in one SQLite file (`db_path`), in the Hermes home of the profile, readable by its owner only (POSIX
  `0600`). Nothing is sent anywhere unless `embeddings` is on (see [Search by
  meaning](#search-by-meaning-optional-online) for what is sent, and to whom). A `db_path` outside the Hermes home is
  listed for `hermes backup`.
- Subagents and flush contexts read the store but never write to it (facts, feedback, counters, extraction). A cron
  job writes facts only through its explicit tool calls (`cron_writes`, on by default); the one automatic write it
  makes is a fact's vector when search by meaning is on.
- With `auto_extract` on, user messages matching the preference/decision patterns are stored at session end.
- `retrieval_count` counts how often a fact was handed to the model; nothing else is tracked.
- Each version keeps its source: platform, chat type, chat / thread / session id, the author of the turn (id and
  display name) and, from 0.15.0, its origin (said / confirmed / inferred). Chat names are not kept. Each confirmation
  keeps its time, session id and the person's id, no text. `forget` deletes the source and the confirmations with
  the fact.
- `forget` deletes a fact for good (see [History](#history-sources-and-forgetting-080)); copies outside this database
  (Hermes's session transcripts, backups, other tools' caches) are not the plugin's to rewrite.
- Dependencies: `numpy` (HRR algebra) and `snowballstemmer` (word forms; pure Python; Polish stems need 3.1 or
  newer). Without numpy the provider
  works with keyword search only and says so in its prompt block; without snowballstemmer words compare exactly.

## Database and downgrades

Schema changes are additive: 0.4.0 adds a `meta` table (schema version, HRR dimension), an `entity_aliases` table and
an `entities.name_key` column; nothing older is dropped or rewritten, and the new column is nullable, so earlier
versions (including the copy bundled in Hermes core) keep reading and writing the file. When this version opens a
file an older one wrote to, it reconciles: missing keys are filled in, entities with the same key are merged (links,
aliases and names kept), aliases of entities the older version deleted are dropped, and vectors whose size does not
match `hrr_dim` are re-encoded. Only a writable agent does this; subagents, cron and flush contexts never migrate.

0.8.0 adds `fact_versions` (with its own FTS index) and `forgotten`; `facts` keeps its columns, so older versions
and tools that read `facts` directly keep working, and see only current facts. A file an older version wrote to is
reconciled on open: a fact without a version gets one (start unknown, source unknown), a text changed by an older
`update` becomes a new version with the old one closed, and a fact an older `remove` deleted is forgotten (its history
deleted, a tombstone written). See [docs/history.md](docs/history.md#migration-and-older-versions-on-the-same-file).

What is tested: this release, then 0.1.1, then this release on one database in CI (adds, edits, removals, aliases, a
keyless duplicate entity written by 0.1.1), plus the reconciliation cases above. What a downgrade does not keep:
aliases made after 0.4.0 are visible to 0.1.1 only through the legacy `entities.aliases` column; and with
`scope: user`, `chat` or `thread`, an older version knows nothing of per-user or per-chat files (0.6.x and older do not
know `chat` / `thread` at all and fall back to `profile`), so it reads and writes the shared `db_path` again. Those
files stay untouched (no leak), but new facts would land in the shared store.

0.14.0 adds `fact_embeddings` (search by meaning): created by every writer, empty unless `embeddings` is on, never
read by older versions. A fact an older version deletes leaves its vector until this version next opens the file for
writing (a subagent or another read-only context does not drop it); a text an older version changes no longer matches its vector's hash and is never compared until a
new vector is made.

0.16.0 adds `entity_aliases.source` (with the default `legacy`: an alias written before, or by an older version
since, is of unknown origin) and `entity_tombstones`, both created by a writer, and records the extraction rules in
`meta.entity_rules`. A context that does not migrate still reads names by their keys and aliases once any writer of
0.4.0 or later has migrated the file. An older
version reads the file as before; one that also writes to it may create a name `delete_entity` removed again from a
new fact's text, and this version's next open does not notice: run `rebuild_entities` after going back and forth.

0.17.0 changes no schema. An older version reads and writes the file as before, but does not keep the rule of 0.17.0:
it resolves a declared name through an alias a fact's text gave, as 0.16.x did. After going back and forth, read
`review` (`alias_candidates`) and `entities` (`text_aliases`). 0.17.1 changes no schema either.

## Development

Tests import Hermes core from a checkout, as an installed plugin does:

```bash
HERMES_AGENT_REPO=~/.hermes/hermes-agent PYTHONPATH=~/.hermes/hermes-agent python -m pytest -q
```

CI runs the suite on Linux, macOS and Windows against hermes-agent `main`, `ruff check .` with the rules in
`pyproject.toml`, and `hermes plugins validate`, the check the plugin catalog runs at admission. 0.11.0: 612 tests pass without the core (2 skipped; the auto-extraction tests
need it), 600 with snowballstemmer 2.2.0 (14 skipped: no Polish stemmer).

Original plugin by dusterbloom (hermes-agent PR #2351); its history is in hermes-agent
(`git log -- plugins/memory/holographic`). MIT licence. The bundled language data live in `dictionaries/` (the list
with sources in its `__init__.py`, rebuilt by the scripts in `tools/`); the Traditional → Simplified Chinese table in `cjk_fold.py` is
derived from OpenCC under Apache-2.0 ([LICENSES/OpenCC-Apache-2.0.txt](LICENSES/OpenCC-Apache-2.0.txt)), the Russian
derivation pairs in `ru_derivation.tsv.gz` from DerivBase.Ru (UDer 1.1) under Apache-2.0
([LICENSES/DerivBase.Ru-Apache-2.0.txt](LICENSES/DerivBase.Ru-Apache-2.0.txt)), the Japanese spelling variants in
`ja_variants.tsv.gz` from SudachiDict's synonym source (Works Applications) under Apache-2.0
([LICENSES/SudachiDict-Apache-2.0.txt](LICENSES/SudachiDict-Apache-2.0.txt)), the Chinese words in `zh_words.txt.gz`
from jieba's dictionary (Sun Junyi) under MIT ([LICENSES/jieba-MIT.txt](LICENSES/jieba-MIT.txt)), the English
derivation pairs in `en_derivation.tsv.gz` from Open English Wordnet 2025 (the Open English Wordnet team) under CC BY
4.0, built on Princeton WordNet 3.1 ([LICENSES/OEWN-CC-BY-4.0.md](LICENSES/OEWN-CC-BY-4.0.md),
[LICENSES/OEWN-WNDB-License.txt](LICENSES/OEWN-WNDB-License.txt)), and the German compounds and German and Spanish
verb forms in `de_compounds.tsv.gz` and `de_es_forms.tsv.gz` from Wikidata Lexemes under CC0
([LICENSES/Wikidata-CC0.txt](LICENSES/Wikidata-CC0.txt)).
