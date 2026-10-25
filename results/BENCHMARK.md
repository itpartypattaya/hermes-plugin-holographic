# Benchmark on the PR head

Every row of the raw files carries `git_sha`, `endpoint` and `scale`; each section names the head it
ran on (the table above: its `provenance_*` rows carry `git_sha 77c9325` — the head they actually ran
on, committed later in the housekeeping `b6cbec5`; default config `RECALL_MARGIN = 1.0` — the 0.6
proposal lives in PR #17; the final merge run: `649276e`). Harness: `tools/provenance_probe.py`; raw rows:
`results/provenance_{multi,contrast_ru,contrast_en,contrast_zh}_{off,on}.jsonl`.

**Sets.** One multilingual set (18 languages — 360 real queries + 90 negatives) and the contrast sets
ru/en/zh (30 real + 45 negatives each): **450 real / 225 negatives per cell**.
**Modes.** `off` (no rerank), `reorder_only` (order only, window 0), `gate_only` (window 0.05 — the
shipped configuration; the pre-0.17.8 labels map as in README, "Benchmark" section).

| embeddings | mode | hit / 450 | wrong-entity | noise / 225 | output lines avg (max) |
|---|---|---|---|---|---|
| on | off | 433 | 28 | 73 | 1.80 (7) |
| on | reorder_only | 433 | 26 | 73 | 1.72 (5) |
| on | gate_only | 432 | 6 | 73 | 0.90 (5) |
| off | off | 430 (428 within the first five) | 25 | 73 | 1.80 (7) |
| off | reorder_only | 430 | 26 | 73 | 1.72 (5) |
| off | gate_only | 429 | 6 | 73 | 0.90 (5) |

- **hit** — the target fact is in the returned output; counting only the first five lines changes
  `off` in 2 queries (emb=off only), the longest output is 7 lines.
- **wrong-entity** — a real query whose output contains the partner fact of a confusable pair:
  27 of 28 (emb=on) are in the contrast sets, one in the multilingual set. Reordering moves the
  count by ±2 (the order changes what the coverage gate keeps); the window cut is what removes
  them — `gate_only` 28 → 6 (emb=on), 25 → 6 (emb=off).
- **noise** — negatives that returned anything. Decomposition (emb=off, `off` mode):
  `greeting_entity` 45/45 — a message that names a person recalls that person, by design per the
  PR thread; `topic_shared_words` 26/45; `vague` 2/45 (the upstream words-gate fix, PR #13,
  removed the stem collisions); the multilingual negatives 0/90.

**CPU latency** (rerank layer only, quality at 100%): `results/bench_cpu_t4.jsonl` —
bge-reranker-v2-m3 Q4_K_M, llama-server `--threads 4`, **Ryzen 7 2700 (this machine)**, 60 queries ×
4 rerank modes = 240 calls, 0 fallbacks, `git_sha 77c9325` (the head):

| mode | p50 ms | p95 ms |
|---|---|---|
| off | 3.7 | 9.2 |
| reorder_only | 178 | 411 |
| reorder_blend | 176 | 393 |
| gate_only | 180 | 398 |
| gate_blend | 184 | 413 |

The earlier CPU table (`results/bench_p8_cpu.jsonl`, `git_sha 7f1efd6`, kept for history) was measured
on a **different processor**: Ryzen 7 5800H laptop, msys2 llama-server `--device none`, before the
upstream words-gate fix — off p50 3 / p95 6 ms, rerank modes p50 94–101 / p95 212–229 ms, pools avg 2–5
docs (max 16). Same model and harness, but the two machines are not comparable; the 4-thread cap on the
new run is deliberate — it is closer to the small servers people run this on.

**CPU gate (2026-10-23, PR #10 point 3):** both rerankers measured in one run — llama-server
`--threads 4` on this machine (Ryzen 7 2700), both models **Q4_K_M**, protocol `--warmup 1
--repeats 5` (warmup passes excluded from every statistic, 620 measured calls per model),
`git_sha 19d7ced`. Raw rows: `results/bench_cpu_gate.jsonl` (bge),
`results/bench_cpu_gate_jina.jsonl` (jina) — each summary already carries its own per-call rows:

| model (CPU, t4) | prefetch p50 ms | prefetch p95 ms | p95 < 300 ms |
|---|---|---|---|
| bge-reranker-v2-m3 Q4_K_M (`:1240`) | 343 | **448** | no |
| jina-reranker-v2-base-multilingual Q4_K_M (`:1239`) | 121 | 152 | yes |

Quality held in every mode on both: recall@5 100%, hit5 60/60, wrong-entity 0, noise 0%.
The 393–413 ms conflict noted above is confirmed at 448 ms p95 on the gate protocol; jina CPU
closes the 300 ms bar with margin and is the CPU candidate, bge stays the GPU default.

**Household sweep (PR #10 points 5/9):** the fixed-label set on heads `a065f9d` (dev + test) and
`649276e` (final test run below), harness `tools/provenance_probe.py`, three modes — `off`,
`reorder_only`, `gate_only` (min-two is built into the window since `a6dcc29`; `gate_named`, `win03`,
`win10` were swept in the older seven-mode run, kept in git history). Cells: bge Q4_K_M `:1235`, jina
Q4_K_M `:1236`, `--emb both`, names-fit on (the default) and `--names-fit off`; per cell 120 real + 90
negatives (dev) / 90 + 45 (test). Raw rows:
`results/provenance_household_{dev,test}{,_nfoff,_jina,_nfoff_jina}_{off,on}.jsonl` — every row stamped
`git_sha a065f9d`. These rows **predate the `rerank_call` stamp** (the field arrived in `649276e`), so
failure absence there rests on the re-run's health checks — sane median ms and non-empty `rank_order`
in rerank rows — not on a per-row status; the final file below carries `rerank_call` per row. The
earlier `dev_nfoff_jina` pair (`79ddd37`, swapped in by `9de6099` — port 1236 was down in that run)
was re-run and replaced.

| cell | hit5 off → gate_only | confused rows off → gate | l5 | lines/turn off → gate |
|---|---|---|---|---|
| dev bge, emb off / on | 69 → 69 / 74 → 75 | 7 → 5 | 0 | 2.08 → 1.27 / 2.15 → 1.35 |
| dev bge names-fit off, off / on | 69 → 69 / 77 → 79 | 7 → 5 | 0 | 2.08 → 1.27 / 2.17 → 1.38 |
| dev jina, off / on | 69 → 69 / 74 → 75 | 7 → 2 | 0 | 2.08 → 1.23 / 2.15 → 1.31 |
| dev jina names-fit off, off / on | 69 → 69 / 77 → 78 | 7 → 2 | 0 | 2.08 → 1.23 / 2.17 → 1.33 |
| test bge, emb off / on | 43 → 43 / 47 → 47 | 6 → 5 | 1 | 1.98 → 1.30 / 2.07 → 1.36 |

Branches per `tools/mcnemar_gate.py` (the guards sit on **both**: paired real+chit rows, confused ≤ off,
noise ≤ off): on **dev, `gate_only` holds branch B in all eight cells** (l5 = 0, unpaired 0, negatives
unchanged); on **test, B is blocked by one target in every configuration** (bge `q057:irina_market[zh]`,
jina `q055:cat_vet[zh]` — class E, cut by the window at the third rank, index 2). Branch A is out of reach
everywhere (+0.0…+1.7 points, p ≥ 0.5).

**Final run (2026-10-23, merge candidate — the run the hold-out is compared against).** Head `649276e`,
one file, one value per field, **0 failed calls** (`rerank_call` per row: answered / skipped_single /
not_called / off — the probe stops a run at the first failure). Model: jina-reranker-v2-base-multilingual
Q4_K_M, sha256 `b2dc8d66b3db2cdfb1818d17e29aba37195bfc75540611b4af5a803e603c6eb7` (gpustack GGUF, the
file confirmed in-thread); server: `C:/msys64/mingw64/bin/llama-server.exe --pooling rank --reranking
--device none --threads 4 --port 1236` (msys2 llama.cpp build); box: Ryzen 7 5800H, Windows, balanced
power scheme. Settings: window 0.05, weight 0, names-fit on, `emb off`, `scale=logit`,
`endpoint http://127.0.0.1:1236/v1/rerank`. Harness: `provenance_probe.py --set household --split test
--modes off,gate_only --emb off --rerank-scale logit --label jina`. Raw rows:
`results/provenance_household_test_jina_off.jsonl` — both modes in one file, 90 real + 45 chit rows
each, no duplicates (checked against the frozen set), same `git_sha` in every row.

| mode | hit@5 / 90 | confused rows | non-target in top 5 | named (full output) | lines/turn | negatives / 45 | l5 vs off |
|---|---|---|---|---|---|---|---|
| off | 43 | 6 | 108 | 91 | 1.98 | 8 | — |
| gate_only | 43 | 4 | 66 | 36 | 1.27 | 8 | **1** |

Both measures, to be explicit: **non-target in top 5** counts every labelled non-target fact that
survives into the first five lines (108 → 66, −39 %); **named (full output)** is the narrower count of
extra facts about a person named in the query, summed over the whole output (91 → 36, −60 %). The
narrower one was the column labelled "non-target in top 5" before this correction — the maintainer's
recount flagged it.

McNemar on the hit5 pairs: b = 0, c = 0, p = 1.0 — A fails on gain; **B is blocked by exactly one lost
target: `q055:cat_vet, zh`** (class E, cut at the third rank — index 2; en/ru keep both), gained none,
unpaired 0.
Per the agreed gate the result is *inconclusive — the feature can stay experimental*; the trade on the
table is −39% non-target facts in the top five (108 → 66), −60% named facts in the full output
(91 → 36) and 1.98 → 1.27 lines per turn, for one lost target.

**CPU timing, `gate_only` alone (the final configuration):** `tools/benchmark_rerank.py --modes
gate_only --endpoint http://127.0.0.1:1236/v1/rerank --rerank-scale logit --warmup 1 --repeats 5`
(`--endpoint` is not optional: without it the harness runs its mock ranker and reports ~0 ms);
raw rows `results/bench_final_jina_gate.jsonl` (summary) and `results/bench_final_jina_gate_calls.jsonl`
(per-call rows), summary stamped `git_sha 649276e`: turn-level
p50 4 / **p95 118 ms**; over 155 warmed calls (31 warm-up excluded, 0 errors) p50 100 / p95 120 /
max 138 ms, pools 2–4 docs, p50 58 / max 83 prompt tokens — the forward pass is the cost, transport
≈ 5 ms. **p95 < 300 ms holds with margin on a laptop CPU at 4 threads.**

**History.** The per-model table (bge/jina, CPU vs GPU, 360 queries, measured before the fixes) and
its raw rows (the `*_[...].jsonl` files) live at commit
[`634cfea`](https://github.com/itpartypattaya/hermes-plugin-holographic/blob/634cfea/results/BENCHMARK.md);
the pre-revert run (RECALL_MARGIN 0.6, `git_sha 7f1efd6`), the wider-store and admission sweeps and
the files removed by the PR housekeeping are in the same history.