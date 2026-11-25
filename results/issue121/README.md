# Issue #121: a lower recall margin whose facts a check confirms

Rows captured at `31de6eb`, the code of the PR that adds this folder; the commit that adds it adds only data.

**Setup.** bge-m3 (1024 dims) behind a loopback `/v1/embeddings`. The check: `Qwen/Qwen3-Reranker-0.6B` behind a
loopback `/v1/rerank`, logit scale, with the instruction in `instruction.txt` (186 bytes, no newline at the end,
sha256 `5336bc93ae7c30960c4f5940de5a98826fe2ecee05c549a846897344d7b73b72`, stamped in every checked row as
`check_instruction_sha256`). Rerank off (`--modes off`), `RECALL_MARGIN` 1.0 in both arms, names_fit on, every store
fully covered. Two arms per set, each in stores of its own: `--recall-check-margin off` (files `_coff`) and
`--recall-check-margin 0.5 --check-model Qwen/Qwen3-Reranker-0.6B --check-instruction instruction.txt` (files
`_c0.5`). Sets: multi (18 languages), contrast_en / ru / zh, household test and dev, synthetic_zh: 1,068 rows per
arm.

| pool | hit@5 | b / c | exact McNemar p | noise | confusion | `--pair check_margin` | report |
|---|---|---|---|---|---|---|---|
| decisive: multi, contrast en / ru / zh, household test: 540 real | 480 → 486 | 6 / 0 | 0.031 | 93 → 93 | 34 → 34 | passes | `gate_check_margin_540.txt` |
| beside: the same with household dev and synthetic_zh: 696 real | 565 → 573 | 8 / 0 | 0.008 | 142 → 142 | 46 → 46 | passes | `gate_check_margin_696.txt` |
| beside: household dev alone: 120 real | 74 → 76 | 2 / 0 | 0.5 | 48 → 48 | 7 → 7 | does not pass (p) | `gate_check_margin_dev.txt` |
| beside: synthetic_zh alone: 36 real | 11 → 11 | 0 / 0 | 1 | 1 → 1 | 5 → 5 | does not pass (p) | `gate_check_margin_synthetic_zh.txt` |

Each report's first line is its command; the decisive one replays without a server:

    python tools/mcnemar_gate.py --pair check_margin results/issue121/provenance_{multi,contrast_en,contrast_ru,contrast_zh,household_test}_{coff,c0.5}_on.jsonl

Every row carries the turn's `check_calls`, `check_outcome`, `check_score` and `check_ms`. On the 696 pool the
checked arm asked 46 times, 22 of them on turns that showed nothing: 11 yes, 35 no, none unanswered.

0.5, the rule (one question a turn, yes above 0) and the instruction were chosen on these same sets, so the p values
are exploratory; a run on another model with everything frozen is a replication.
