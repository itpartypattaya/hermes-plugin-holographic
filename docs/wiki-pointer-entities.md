# Wiki pointer facts

A **wiki pointer line** is how the wiki ingest makes a page a topic of its own in the store. The line is written
through the ordinary `add_fact` path, and it names the page in a fixed shape:

```
[WIKI HINT - consult when relevant] Wiki page: <namespace>/[section/]<slug>.md | type: entity
```

Examples:

```
[WIKI HINT - consult when relevant] Wiki page: ai-agents/entities/llama-cpp.md | type: entity
[WIKI HINT] Wiki page: gaming/openmw/tamriel-rebuilt.md | type: location
```

## What the line links

The path is the only proper noun on such a line, so the entity extractor takes its parts:

- the **namespace** — the first segment (`ai-agents` → `ai agents`);
- the **page slug** — the last segment, minus `.md` (`llama-cpp.md` → `llama cpp`).

Dropped on purpose:

- a section directory in the middle (`gaming/openmw/tamriel-rebuilt.md` names `gaming` and `tamriel rebuilt`, not
  `openmw`);
- structural slugs — `index`, `schema`, `log`, `readme`, `overview`, `changelog`, `notes`, `todo`, `wiki index`,
  `wiki log`;
- a path with no namespace (`llama-cpp.md`): that is not a pointer to a wiki page;
- a name `delete_entity` removed: a path is text, and text never lifts a tombstone.

## What is not a pointer

The line is read only in this shape, so an ordinary note cannot turn words into pages:

- it must start a line, with the `[WIKI HINT …]` marker before `Wiki page:`;
- a value carrying a scheme (`https://…`) is refused: a URL is no page of this wiki;
- Markdown code is blanked out before the line is read — a fact that shows the format in backticks or a fenced block
  is documentation, not a pointer.

```
Wiki page: https://en.wikipedia.org/wiki/Foo is a good intro     # no entities
see Wiki page: gaming/openmw.md for the details                  # no entities (not a line of its own)
`[WIKI HINT] Wiki page: ai-agents/entities/llama-cpp.md`         # no entities (code)
```

## Who writes it

Anyone may: the line is a plain fact body, so bulk wiki ingest, a hand-written `add_fact`, or any other writer can
produce it. Nothing but the shape above is required — no tool call, no metadata, no declared names. Write the marker
and the path as one line at the start of the fact, and the namespace and page slug are linked when the fact is
stored.

## Existing stores

Linkage from a pointer line is part of the entity rules (`_ENTITY_RULES`, currently 5), so a store indexed by an
older release gets the links on its first writer open, without waiting for a re-ingest. Re-adding the same text does
not restore links that another operation dropped: the relinking paths (`_link_entities`, `_detach_alias`,
`rebuild_entity_index`) all read a fact's text through one helper, `MemoryStore._implied_entity_ids`, so a pointer
fact keeps its namespace and page slug through an alias split as well.
