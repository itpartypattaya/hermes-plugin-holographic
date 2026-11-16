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
