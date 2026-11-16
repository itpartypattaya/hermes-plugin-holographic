# Issue #95 pilot: keywords in a fact's vector

One model, one coverage, gate G1 fixed before the run (2026-11-16). Rows captured at `b1a3d73`; the gate's report is
`gate_g1_pilot.txt` (`python tools/mcnemar_gate.py --pair embed_text
results/issue95/provenance_{household_test,contrast_en,contrast_ru,contrast_zh,multi}_*_on.jsonl`; the
`synthetic_zh` rows below are a separate run and stay out of this pool).

**Setup.** text-embedding-3-small (1536 dims) through the OpenAI API; rerank off (`--modes off`), `RECALL_MARGIN`
1.0, names_fit on. Sets: household (test split, en/ru/zh), contrast_en/ru/zh, multi (en, ru, zh, de, ja, it, nl,
th). Keywords from `tools/issue95_keywords/` (written blind), on half of each language's facts (seed 95), stored in
`tags` in BOTH arms, so the word side is the same; the arms differ only in what the vectors are made from
(`--embed-text content` vs `content+keywords`), each in stores of its own. Every store fully covered (usable vectors
= expected). Sent to the API over both arms: 2,944 texts in 1,139 requests, 58,312 tokens as the API reported them.

**Power.** With content-only vectors, 36 of the 325 labelled questions miss their target in the top five (22 not
even in the word pool, 14 cut by the coverage filter), so up to 36 could have been rescued.

**Result: no effect.** Pooled hit@5 289 → 289 (of 340 real rows, 325 labelled): b = 1 rescued (household
q056:keys_hook, ru), c = 1 lost (household q067:irina_walk, zh), exact McNemar p = 1.0 — the gate does not pass.
The target anywhere in the output: the same, b = 1, c = 1. Noise rows: added 2, removed 3 (pooled −1; household
alone +2: two chit rows gained output). Confusion: added 0, removed 1. Facts by meaning in the output: 34 → 31.

| set | hit@5 content → content+keywords | b | c | noise added − removed | facts by meaning |
|---|---|---|---|---|---|
| household test | 56/90 → 56/90 | 1 | 1 | 2 − 0 | 21 → 22 |
| contrast_en | 23/30 → 23/30 | 0 | 0 | 0 − 1 | 1 → 1 |
| contrast_ru | 27/30 → 27/30 | 0 | 0 | 0 − 0 | 4 → 4 |
| contrast_zh | 24/30 → 24/30 | 0 | 0 | 0 − 2 | 7 → 3 |
| multi | 159/160 → 159/160 | 0 | 0 | 0 − 0 | 1 → 1 |

**Not measured here.** bge-m3 (the issue's model); full coverage; `fact_store search` on the same questions; the
held-out 126 and the author's set. `ms` in the keywords arm is not comparable: in that mode `Meaning.fits` screens
the joined text of every candidate again. Pairs are counted per (set, lang, qid), so one household question in three
languages is three pairs, as in #17 — it would only make a gain look stronger, and there is none.

## The contributor's set: synthetic_zh on text-embedding-3-small

The set liwangcc contributed in #102 (`tools/synthetic_zh.py`: 30 Chinese facts, 36 labelled questions, 12 noise
rows; keywords `tools/issue95_keywords/synthetic_zh.json`, `0cbae96e2265`, written blind on their side). Rows captured
at `5068883` (2026-11-16), same setup as the pilot: text-embedding-3-small through the OpenAI API, rerank off,
`RECALL_MARGIN` 1.0, keywords in `tags` in both arms, each run in a store of its own, every store fully covered
(30 of 30 vectors). Two coverages: all 30 facts with keywords, and half (seed 95, 15 facts). The four runs sent 4,838
tokens as the API reported them. Gate reports: `gate_g1_synthetic_zh_all.txt` and `gate_g1_synthetic_zh_half.txt`
(`python tools/mcnemar_gate.py --pair embed_text` on the two files of one coverage).

| coverage | hit@5 content → content+keywords | b | c | p | noise added − removed | confusion added − removed | facts by meaning |
|---|---|---|---|---|---|---|---|
| all | 29/36 → 29/36 | 0 | 0 | 1.0 | 0 − 0 | 0 − 0 | 0 → 0 |
| half | 19/36 → 19/36 | 0 | 0 | 1.0 | 0 − 0 | 0 − 0 | 2 → 2 |

**Result: no effect, G1 does not pass at either coverage.** No question changed its hit, noise or confusion status between the arms. At full coverage the
default z-score admission lets no fact in by meaning in either arm, so what the vector is made from cannot matter.
The keywords do work through the word side: from half to full coverage hit@5 rises 19 → 29 in both arms (paraphrase
4 → 7, colloquial 2 → 6). Power is low: at full coverage 7 targets miss the top five (3 not in the word pool, 4 cut by
the coverage filter). Confusion for this set is "non-target in the top five" (`wrong5`), since it has no curated
confusers.

The contributor's own rows on bge-m3 at 0.27.0 and at the #102 commit (in #95) agree: one target rescued, none lost,
p = 1.0. With the pilot, this closes #95: the vector stays content only by default, and `--embed-text` stays in the
probe for measuring other stores.
