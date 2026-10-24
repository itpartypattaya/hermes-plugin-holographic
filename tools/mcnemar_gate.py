"""Exact McNemar preview of the PR gate on a probe run (dev or test).

The gate agreed in PR #10 (maintainer, 2026-10-22), to be applied to the test split:
either a real paired gain in hit@5 (>= +5 points and exact McNemar p < 0.05) or fewer
irrelevant facts without losing the right ones — and on BOTH branches wrong-entity
(`confused`) and noise (`neg`) no higher than `off`, with the rows paired. This tool
computes both branches for every mode against `off` on whatever provenance_probe output
files you point it at:

    python tools/mcnemar_gate.py                      # all dev files
    python tools/mcnemar_gate.py results/provenance_household_dev_on.jsonl

A file that cannot be compared honestly aborts instead of printing a pass: duplicated
rows, mixed runs (several git_sha/scale/endpoint/emb/split values) and targets that
disagree with `off` all raise; unpaired real AND chit rows are counted, printed and
block both branches (fix-list 19:07).

Raw rows are paired per (lang, qid); p is the two-sided exact binomial on the
discordant pair counts (b = rescued, c = lost).
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _validate(rows: list[dict], path: str) -> None:
    """Отчёт невозможен — падаем, а не молчим (фикc-лист 19:07, пп. 2-3)."""
    seen = collections.Counter((r.get("mode"), r.get("kind"), r.get("lang"), r.get("qid"))
                               for r in rows)
    dup = sorted(k for k, v in seen.items() if v > 1)
    if dup:
        raise SystemExit(f"{path}: дубликат строк {dup[:5]} — отчёт невозможен")
    for field in ("git_sha", "scale", "endpoint", "emb", "split"):
        vals = sorted({str(r[field]) for r in rows if r.get(field) not in (None, "")})
        if len(vals) > 1:
            raise SystemExit(f"{path}: смешанные прогоны — {field}={vals}")
    off_targets = {(r.get("lang"), r.get("qid")): (r.get("targets") or [])
                   for r in rows
                   if r.get("mode") == "off" and r.get("kind") == "real"}
    for r in rows:
        if r.get("mode") == "off" or r.get("kind") != "real":
            continue
        key = (r.get("lang"), r.get("qid"))
        if key in off_targets and (r.get("targets") or []) != off_targets[key]:
            raise SystemExit(f"{path}: targets расходятся с off у {r.get('mode')} {key}: "
                             f"{r.get('targets')} != {off_targets[key]}")


def report(path: str) -> None:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    _validate(rows, path)
    real = [r for r in rows if r["kind"] == "real"]
    chit = [r for r in rows if r["kind"] == "chit"]
    modes = sorted({r["mode"] for r in rows})
    if "off" not in modes:
        print(f"{path}: no off mode, skipped")
        return
    hits: dict[str, dict[tuple, bool]] = collections.defaultdict(dict)
    top5: dict[str, dict[tuple, set]] = collections.defaultdict(dict)
    confused = collections.Counter()
    lost = collections.Counter()
    neg = collections.Counter()
    # targets берутся только из off: расхождение другого режима — ошибка валидации,
    # а не «какая строка прочиталась последней» (фикc-лист 19:07, п. 3).
    off_targets = {(r["lang"], r["qid"]): set(r.get("targets") or [])
                   for r in real if r["mode"] == "off"}
    for r in real:
        key = (r["lang"], r["qid"])
        hits[r["mode"]][key] = bool(r["hit5"])
        top5[r["mode"]][key] = set((r.get("keys") or [])[:5])  # matched top-5 (п.8)
        if r.get("confused"):
            confused[r["mode"]] += 1
        if key in off_targets:
            lost[r["mode"]] += len(off_targets[key] - set(r.get("keys") or []))
    for r in chit:
        if (r.get("out_n") or 0) > 0:
            neg[r["mode"]] += 1

    base = hits["off"]
    n = len(base)
    base_hits = sum(base.values())
    # Branch B: per-target losses against off IN THE TOP FIVE — a multi-target question that keeps one
    # target and drops the other is a loss. Pairing covers real AND chit rows; unpaired blocks both
    # branches (a mode missing its chit block would otherwise lower `neg` and buy a pass).
    lost5: collections.Counter = collections.Counter()
    unpaired: dict[str, int] = {}
    chit_keys = {(r["lang"], r["qid"]) for r in chit if r["mode"] == "off"}
    for m in modes:
        if m == "off":
            continue
        keys_m, keys_o = set(hits[m]), set(base)
        chit_m = {(r["lang"], r["qid"]) for r in chit if r["mode"] == m}
        unpaired[m] = len(keys_m ^ keys_o) + len(chit_m ^ chit_keys)
        for k in keys_m & keys_o:
            off_targets_in_top5 = off_targets[k] & top5["off"][k]
            lost5[m] += len(off_targets_in_top5 - top5[m][k])
    print(f"\n== {os.path.basename(path)} | real={n} negatives={len(chit)} "
          f"| off hit5 {base_hits}/{n} confused={confused['off']} lost={lost['off']}")
    print(f"{'mode':15s} {'hit5':>8} {'pts':>6} {'b':>3} {'c':>3} {'p':>6} "
          f"{'conf':>5} {'lost':>5} {'l5':>3} {'unp':>4} {'neg':>4}  gate branches")
    for m in modes:
        cur = hits[m]
        b = sum(1 for k in base if not base[k] and cur.get(k))
        c = sum(1 for k in base if base[k] and not cur.get(k))
        h = sum(cur.values())
        pts = 100.0 * (h - base_hits) / n
        p = mcnemar(b, c)
        branches = ""
        if m != "off":
            # Общая охрана ОБОИХ веток (согласованный гейт, 2026-10-22): парность (real И chit),
            # confused и noise не выше off. A добавляет +5 очков при p<0.05, B — l5==0 и
            # строго меньше путаницы (фикc-лист 19:07, п. 4).
            guard = (unpaired.get(m, 0) == 0
                     and confused[m] <= confused["off"] and neg[m] <= neg["off"])
            if guard and pts >= 5.0 and p < 0.05:
                branches += "A"
            if guard and lost5[m] == 0 and confused[m] < confused["off"]:
                branches += "B"
            branches = branches or "-"
        print(f"{m:15s} {h:4d}/{n} {pts:+5.1f} {b:3d} {c:3d} {p:6.3f} "
              f"{confused[m]:5d} {lost[m]:5d} {lost5[m]:3d} {unpaired.get(m, 0):4d} {neg[m]:4d}  {branches}")
    bad = {m: v for m, v in unpaired.items() if v}
    if bad:
        print(f"  UNPAIRED rows vs off (both branches refuse to pass): {bad}")


def main() -> None:
    paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(os.path.dirname(__file__), "..",
                               "results", "provenance_household_dev*.jsonl")))
    for path in paths:
        report(path)


if __name__ == "__main__":
    main()