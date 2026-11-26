# Issue #121, replicated on text-embedding-3-small

Rows captured at `67229de`, the head of #135, the PR that adds `RECALL_CHECK_MARGIN`; the commit that adds this
folder adds only data and the scripts that produced it. Results posted in #135 (issuecomment-7185845923).

**Setup.** OpenAI `text-embedding-3-small` (1536 dims) read through `scripts/emb_cache_proxy.py`, a loopback
`/v1/embeddings` that caches vectors by text. Everything else as in the folder above: rerank off (`--modes off`),
`RECALL_MARGIN` 1.0, the same seven sets (multi in 18 languages), 1,068 rows per arm. Five arms, each in stores of
its own:

| folder | arm | endpoint stamp |
|---|---|---|
| `off/` | `--recall-check-margin off` | `127.0.0.1:1242/v1/rerank` |
| `margin_0.5/` | `--recall-margin 0.5 --recall-check-margin off`: the lower margin with no check | `127.0.0.1:1242/v1/rerank` |
| `qwen3/` | `--recall-check-margin 0.5` with `Qwen/Qwen3-Reranker-0.6B` and the instruction of the folder above | `127.0.0.1:1242/v1/rerank` |
| `off_decisions/` | `--recall-check-margin off` | `api.openai.com/v1/decisions` |
| `decisions/` | `--recall-check-margin 0.5` with OpenAI's Decisions API as the check | `api.openai.com/v1/decisions` |

The gate pairs a checked arm only with an off arm of the same endpoint stamp, hence two off arms. They come out
identical on all 1,068 rows.

**Why the cache.** OpenAI's embeddings are not deterministic between calls. Two off arms run at different times
without the cache differed by four noise rows, two confusion rows and one hit: the size of the effects the gate
judges. With every arm reading through one cache (same text, same vector), the comparison is paired by
construction.

**The Qwen3 arm** ran on CPU in float32 through `scripts/qwen3_server.py` (the model card's pair format, the
instruction in `../instruction.txt`); the model file matches the sha256 in #135 (`27cd75a4…`). On the two sample
pairs of #135 it gives 5.20 and −1.16 against 5.25 and −0.94 there: the same sign, not the same bits. A call takes
1–2 s on CPU, so the arm ran through `scripts/slow_check.py`, which raises `rerank_timeout` to 10 (the plugin caps it
at 4.5 s). The timeout decides whether an answer arrives, not what it says; none went unanswered.

**The Decisions arm** was pre-registered in #135 (issuecomment-7185238331) before it ran: `gpt-6-luna`, one
`predicate` question with `scripts/decisions_instruction.txt` (sha256 `e28e1830…aefab`, stamped in the rows) on
`scripts/decisions_input_template.txt`, admitted when its probability is above 0.5. `scripts/decisions_check.py`
swaps `recall_check.CHECK` for that call. 62 calls, 13,357 input tokens, none failed.

| pool | arm | hit@5 | b / c | exact McNemar p | noise | confusion | gate | report |
|---|---|---|---|---|---|---|---|---|
| decisive: 540 real | plain 0.5 | 486 → 493 | 7 / 0 | 0.016 | 106 → 124 | 32 → 33 | does not pass (noise, confusion) | `gate_margin_540.txt` |
| | Qwen3 | 486 → 491 | 5 / 0 | 0.0625 | 106 → 106 | 32 → 32 | does not pass (p) | `gate_qwen3_540.txt` |
| | Decisions | 486 → 492 | 6 / 0 | 0.031 | 106 → 106 | 32 → 33 | does not pass (confusion) | `gate_decisions_540.txt` |
| beside: 696 real | plain 0.5 | 575 → 590 | 15 / 0 | 0.0001 | 159 → 182 | 46 → 47 | does not pass (noise, confusion) | `gate_margin_696.txt` |
| | Qwen3 | 575 → 584 | 9 / 0 | 0.004 | 159 → 159 | 46 → 46 | passes | `gate_qwen3_696.txt` |
| | Decisions | 575 → 588 | 13 / 0 | 0.0002 | 159 → 159 | 46 → 47 | does not pass (confusion) | `gate_decisions_696.txt` |

The Decisions arm's extra confusion row is `household ru q053`, admitted at probability 0.51.

Each report's first line is its command; all replay from these rows without a server. `scripts/run.sh` is the run
itself: the servers it needs and the order of the arms.
