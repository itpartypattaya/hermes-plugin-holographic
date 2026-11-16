# Keywords for issue #95

Frozen keywords for the facts of the probe's sets (`tools/provenance_probe.py --keywords <file>`), one file per set,
shaped `{lang: {fact key: [keywords]}}` (for `contrast_*` the lang is the set's name).

How they were written (2026-11-16, before any run with them): separate agents (Claude Sonnet), one per household
language and one for multi and the three contrast sets, each given only a file of fact texts by key and the
`keywords` field's description from the tool schema, told to read no other file. None saw a question. The
description's example ("Mom's birthday is May 3") appears in none of the sets. The multi + contrast agent first left
out two contrast_ru facts (`sergey_flat`, `sergey_market`) and added them on request under the same rules. Nothing
was edited after: some keywords repeat words of the fact's own text, as the description asks them not to; they stay
as written, since picking them over after a run would tune the measurement.

Every file was checked once on entry: every fact key of the set present, none extra, 3–8 keywords of at most 40
characters each, no comma inside a keyword. The probe stops a run when the plugin does not keep a keyword as written.

## synthetic_zh (issue #95 contribution)

`synthetic_zh.json` (`{synthetic_zh: {fact key "1".."30": [keywords]}}`) accompanies `tools/synthetic_zh.py`, a
Chinese-heavy synthetic set: 30 generic software-project facts, 36 labelled questions (direct / paraphrase /
crosslingual / colloquial) and 12 noise rows (6 negative chit-chat + 6 in-domain no-answer). MIT, copyright
liwangcc; no personal data. 219 keywords, 6–8 per fact.

Written blind by a separate pass that saw only the fact texts and the `keywords` field description — no queries,
results, or eval README. Disclosure, as provided by the contributor: **information-blind, but not a separate
session** (the queries existed calendar-wise before the keywords were written, but the writing pass had no access
to them; a transcript audit found no read of queries/results before the freeze). Frozen before any run with them.
Hash chain: frozen `blind_keywords.json`, sha256
`eef343b0689afd7e356fa5e7839323e8571ee077316ca2630688682338426607` → reshaped to `{lang: {key: [...]}}` with
nothing else changed (same 30 keys, same 219 keywords, same order) → this file, sha256
`0cbae96e2265a597c1d8871b000bd3dc5e495d318238bd043cd76e08f4acf4f7` as committed (LF line endings). A working
copy of the identical content with CRLF endings has sha256
`ce260e160536973e277ec460b98d7940db08d0ffc64954fcecad932a93735612` — that is the hash the original PR
description quoted; the committed LF bytes are what the probe reads and records. Checked on entry against the
same rules: every fact key present, none extra, 3–8 keywords of at most 40 characters, no comma inside a keyword.

The set has no curated confusers, so its confusion figure is "non-target in the top five" — the same measure as
multi's `wrong5`, not household's curated-confuser count. The gate reads it through the `wrong5` branch of
`_confused`; the figure is a precision proxy, identical in both arms.
