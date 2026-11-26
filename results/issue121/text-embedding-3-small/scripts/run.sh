#!/bin/bash
# The five arms of the #121 replication on text-embedding-3-small, as they were run for this folder.
# From the repository root, at the commit the rows carry (git_sha), with the plugin's test dependencies installed:
#
#   OPENAI_API_KEY=... HERMES_AGENT_REPO=<hermes-agent checkout> bash results/issue121/text-embedding-3-small/scripts/run.sh <out dir>
#
# Two servers must already listen on loopback:
#   127.0.0.1:1250  python emb_cache_proxy.py 1250 emb_cache.jsonl     (every arm reads its vectors through it)
#   127.0.0.1:1242  python qwen3_server.py <model dir> results/issue121/instruction.txt 1242 float32 4
# The Decisions arm calls api.openai.com directly; the --endpoint stamps only label the rows, so the gate pairs each
# checked arm with the off arm of the same stamp.
set -eu
OUT=$1
mkdir -p "$OUT"  # the probe makes only the last level of its --out
HERE=results/issue121/text-embedding-3-small/scripts
test -n "$OPENAI_API_KEY" && test -n "$HERMES_AGENT_REPO"
export EMB_URL=http://127.0.0.1:1250/v1 EMB_MODEL=text-embedding-3-small
export PYTHONPATH="tools:$HERMES_AGENT_REPO"
export DECISIONS_INSTRUCTION=$HERE/decisions_instruction.txt DECISIONS_TEMPLATE=$HERE/decisions_input_template.txt
LANGS=ar,bn,de,en,es,fr,hi,id,it,ja,ko,nl,pl,pt,ru,th,tr,zh
QWEN=(--check-model Qwen/Qwen3-Reranker-0.6B --check-instruction results/issue121/instruction.txt)
DEC=(--check-model gpt-6-luna --check-instruction "$DECISIONS_INSTRUCTION")

arm() {  # arm <dir> <endpoint stamp> <script> <probe args...>
  local dir=$1 stamp=$2 script=$3; shift 3
  local common=(--modes off --emb on --endpoint "$stamp" --out "$OUT/$dir")
  python -X utf8 "$script" "${common[@]}" --set multi --langs "$LANGS" "$@"
  for s in contrast_en contrast_ru contrast_zh synthetic_zh; do python -X utf8 "$script" "${common[@]}" --set "$s" "$@"; done
  for split in test dev; do python -X utf8 "$script" "${common[@]}" --set household --split "$split" "$@"; done
}

R=http://127.0.0.1:1242/v1/rerank D=https://api.openai.com/v1/decisions
arm off           $R tools/provenance_probe.py --recall-check-margin off
arm margin_0.5    $R tools/provenance_probe.py --recall-margin 0.5 --recall-check-margin off
arm qwen3         $R $HERE/slow_check.py       --recall-check-margin 0.5 "${QWEN[@]}"
arm off_decisions $D tools/provenance_probe.py --recall-check-margin off
arm decisions     $D $HERE/decisions_check.py  --recall-check-margin 0.5 "${DEC[@]}"
