# Rerank contrast benchmark — method and final numbers

Context: PR #10 (RFC issue #9) adds layer 3 — a cross-encoder on top of the blend stage. This
document is the method of the contrast sets and the final measurements on the PR head. The full
experiment log (hypotheses H1–H8, threshold sweeps, intermediate tables, deleted probe scripts
`score_probe`/`loss_probe`/`margin_sweep`/…) lives in this file's history:
[`634cfea`](https://github.com/itpartypattaya/hermes-plugin-holographic/blob/634cfea/docs/contrast-benchmark.md).

## 1. Infrastructure

| Component | Value |
|---|---|
| Server | `llama-server` (MSYS2, llama.cpp 0.7.0-dev) or LM Studio |
| Model | `bge-reranker-v2-m3-GGUF`, Q4_K_M; embeddings — bge-m3 Q4 (`:1237`) |
| Flags | `--reranking --embedding --pooling rank`; CPU — `-dev none` |
| Repository | branch `feature/rerank-benchmark` (PR #10) |

The endpoint answers on the scale of **raw logits** (probe: relevant +6.36, garbage −11.01). A
counter ratio on logits is mathematically unreachable at negative values, so a sigmoid
normalization sits before the gate (`rerank.normalized`); it does not change the order.

## 2. Data sets (`tools/contrast_{ru,en,zh}.py`)

One schema: `FACTS` (text, entities) / `PAIRS` / `CHITCHAT` / `self_check()` (exit 0 — no data
errors).

| Set | Facts | Pairs (=30 queries) | chit-chat |
|---|---|---|---|
| ru | 64 | 15 (c1_entity, c2 incl. number, c3) | 45 (15/15/15) |
| en | 64 | 15 (same) | 45 (15/15/15) |
| zh | 69 | 15 (same) | 45 (15/15/15) |

Pair types: **c1_entity** — a name swap (Peter/Lena, 大卫/大伟…); **c2** — a paraphrase bridge:
the target shares no content words with the query, the distractor does (the BM25 trap); the
`number` subtype — a digit in the fact against the spelled-out word in the query; **c3** — a
predicate with one subject. **CHITCHAT** — 45 messages that must recall nothing:
`greeting_entity` (a name from FACTS), `topic_shared_words` (shares words with facts), `vague`.

## 3. Metrics

| Metric | Definition |
|---|---|
| hit | the target fact in the full output (hit@5 — in the first five lines) |
| WRONG5 / wrong-entity | a foreign (partner) target in the output of a query that has its own |
| noise | the share of chit-chat with a non-empty output |
| output | the number of answer lines (avg, max) |
| p50 / p95 | median / 95th percentile of a query run, HTTP call included |

## 4. Final numbers

Every row carries the `git_sha` it ran on (the `provenance_*` rows: `77c9325`, committed later in
the housekeeping `b6cbec5`), default config (`RECALL_MARGIN 1.0` — the 0.6 proposal lives in
PR #17). The full table is `results/BENCHMARK.md`, raw rows are
`results/provenance_*_{off,on}.jsonl`.

Aggregate (4 sets, 450 real / 225 negatives per cell):

| embeddings | mode | hit / 450 | wrong | noise / 225 | output avg (max) |
|---|---|---|---|---|---|
| on | off | 433 | 28 | 73 | 1.80 (7) |
| on | reorder_only (window 0) | 433 | 26 | 73 | 1.72 (5) |
| on | gate_only (window 0.05) | 432 | 6 | 73 | 0.90 (5) |
| off | off | 430 | 25 | 73 | 1.80 (7) |
| off | gate_only (window 0.05) | 429 | 6 | 73 | 0.90 (5) |

Per-cell contrast sets (emb=on; hit/30, wrong, noise/45):

| set | off | reorder_only | gate_only |
|---|---|---|---|
| ru | 27, 8, 25 | 27, 8, 25 | 27, **1**, 25 |
| en | 23, 7, 26 | 23, 5, 26 | 22, **0**, 26 |
| zh | 24, 12, 22 | 24, 12, 22 | 24, **5**, 22 |

Noise is identical in every mode: the gate window runs after recall and does not move noise —
that is an admission (coverage + words) problem, not a rerank one. 73 of 225 (emb=off, mode off):
`greeting_entity` 45/45 — a trap by construction: a message with a person's name must recall
their facts (agreed in the PR thread); `topic_shared_words` 26/45; `vague` 2/45 — the stem
collisions were cured by the words-gate fix in main (PR #13); multilingual negatives 0/90.

CPU latency (bge-q4, 240 calls, 0 fallbacks). Run on head `77c9325`: llama-server `--threads 4`
on a **Ryzen 7 2700** — off p50 3.7 / p95 9.2 ms, rerank modes p50 176–184 / p95 393–413 ms
(`results/bench_cpu_t4.jsonl`). The same numbers, measured earlier on a **different processor**
(Ryzen 7 5800H, msys2 `--device none`, `7f1efd6`, before the words-gate fix): off p50 3 / p95 6
ms, rerank modes p50 94–101 / p95 212–229 ms, pools of 2–5 docs (`results/bench_p8_cpu.jsonl`) —
one model and one harness, but the two machines are not comparable; the 4-thread cap in the new
run is deliberate (small servers). Early measurements on an iGPU box gave ~255–300 ms (iGPU) and
~400 ms (CPU) per call — the same story, commit `634cfea`.

Conclusions that survived the cuts:

1. **Reorder does not fix recall**: 433 == 433, the wrong shift ≤ 2 comes from order — the
   coverage gate and the order do not see each other; only the cut window works (wrong → 6 with
   output 1.80 → 0.90).
2. **Wrong-entity is not fixed by order**: a foreign target is semantically close by the design
   of the pair; off's low wrong@5 is often an artifact of an empty output, not a merit.
3. **Class A (the target is not in the word pool) belongs to embeddings**, not rerank; on zh it
   was the base of the basic MISSes.
4. **`top_k`/width did not fire in these sets** (fewer than 40 candidates) — the RFC's "deep pool"
   was not really exercised on them; the wide runs (~620 facts) moved to the file's history.
5. The maintainer's next step for the gate — a public set of ~200 facts with labels and a
   dev/test split (the recipe is in the PR #10 discussion, point 9).

## 5. Reproducibility

```bash
# runs (from the repository root):
PYTHONPATH=~/.hermes/hermes-agent HERMES_AGENT_REPO=~/.hermes/hermes-agent \
    python -I tools/contrast_ru.py                       # data self-check
CONTRAST_SET=contrast_en python -I tools/contrast_off.py  # EN; likewise zh
python -I tools/provenance_probe.py ...                   # full run (see --help)
```

Tools in the tree: `tools/contrast_{ru,en,zh}.py` (data), `contrast_off.py` (driver),
`provenance_probe.py` (harness with `git_sha` per row), `benchmark_rerank.py` (metrics per
language cell). The probe scripts were deleted at housekeeping (point 10) — they are in the
commit history.