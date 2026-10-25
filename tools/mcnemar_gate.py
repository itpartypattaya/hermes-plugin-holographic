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
    """The report is impossible — we fail instead of staying silent (fix-list 19:07, items 2-3)."""
    seen = collections.Counter((r.get("mode"), r.get("kind"), r.get("lang"), r.get("qid"))
                               for r in rows)
    dup = sorted(k for k, v in seen.items() if v > 1)
    if dup:
        raise SystemExit(f"{path}: duplicate rows {dup[:5]} — the report is impossible")
    for field in ("git_sha", "scale", "endpoint", "emb", "split"):
        vals = sorted({str(r[field]) for r in rows if r.get(field) not in (None, "")})
        if len(vals) > 1:
            raise SystemExit(f"{path}: mixed runs — {field}={vals}")
    off_targets = {(r.get("lang"), r.get("qid")): (r.get("targets") or [])
                   for r in rows
                   if r.get("mode") == "off" and r.get("kind") == "real"}
    for r in rows:
        if r.get("mode") == "off" or r.get("kind") != "real":
            continue
        key = (r.get("lang"), r.get("qid"))
        if key in off_targets and (r.get("targets") or []) != off_targets[key]:
            raise SystemExit(f"{path}: targets disagree with off for {r.get('mode')} {key}: "
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
    # targets come from off only: a mismatch in another mode is a validation error,
    # not "whichever row was read last" (fix-list 19:07, item 3).
    off_targets = {(r["lang"], r["qid"]): set(r.get("targets") or [])
                   for r in real if r["mode"] == "off"}
    for r in real:
        key = (r["lang"], r["qid"])
        hits[r["mode"]][key] = bool(r["hit5"])
        top5[r["mode"]][key] = set((r.get("keys") or [])[:5])  # matched top-5 (item 8)
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
            # Shared guard for BOTH branches (agreed gate, 2026-10-22): pairing (real AND chit),
            # confused and noise no higher than off. A adds +5 points at p<0.05, B — l5==0 and
            # strictly less confusion (fix-list 19:07, item 4).
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


def _confused(r: dict) -> bool:
    """A confuser in the top five: household stamps `confused`, multi a `wrong5` list, contrast a
    boolean `wrong` that is already computed against the top five."""
    return bool(r.get("confused")) or bool(r.get("wrong5")) or bool(r.get("wrong"))


def _noise(r: dict) -> bool:
    """A row that showed anything it should not: chit with output, or a no-answer real row (a real row
    without targets — the maintainer's PR #17 recipe counts them as noise, unlike the mode report above)."""
    if (r.get("out_n") or 0) <= 0:
        return False
    if r.get("kind") == "chit":
        return True
    return r.get("kind") == "real" and not (r.get("targets") or [])


def report_margins(paths: list[str]) -> None:
    """Two-margin pairing for the PR #17 gate: rows stamped with `margin` (provenance_probe
    --recall-margin) are paired per (set, lang, qid) and compared 0.6 against 1.0 on one head.

    Reports hit@5 with exact McNemar on the pooled rows, the gained and lost top-five targets as
    `set/lang qid -> target`, noise (chit AND no-answer rows with output) and confusion — the four
    guards of the recipe: no target lost anywhere, noise not higher, confusion not higher, and the
    rescue passing p < 0.05 pooled (2026-10-24)."""
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in open(path, encoding="utf-8") if line.strip())
    margins = sorted({float(r["margin"]) for r in rows if r.get("margin") is not None})
    if len(margins) != 2:
        raise SystemExit(f"expected rows stamped with exactly two margins, got {margins} — "
                         f"run provenance_probe --recall-margin twice and pass both files")
    lo, hi = margins  # sorted ascending: lo is the proposal (0.6), hi the current default (1.0)
    for field in ("git_sha", "emb", "mode", "scale"):
        vals = sorted({str(r.get(field)) for r in rows})
        if len(vals) > 1:
            raise SystemExit(f"mixed runs — {field}={vals}: the pairing needs one clean head")
    seen = {m: set() for m in margins}
    for r in rows:
        key = (r.get("set"), r.get("lang"), r.get("qid"))
        if key in seen[float(r["margin"])]:
            raise SystemExit(f"duplicate rows for margin {r['margin']} {key} — the report is impossible")
        seen[float(r["margin"])].add(key)
    # pairing is per (set, lang, qid): the margin is what differs, not part of the pair key
    only = {m: sorted(seen[m] - (seen[margins[0]] & seen[margins[1]])) for m in margins}
    if only[margins[0]] or only[margins[1]]:
        raise SystemExit(f"unpaired rows: {only[margins[0]][:3]} vs {only[margins[1]][:3]}")
    by = {m: {(r["set"], r["lang"], r["qid"]): r for r in rows if float(r["margin"]) == m}
          for m in margins}
    real_keys = [k for k, r in by[lo].items() if r.get("kind") == "real"]

    hit = {m: {k: bool(by[m][k].get("hit5")) for k in real_keys} for m in margins}
    # b/c of McNemar: b = rescued BY the proposal (miss at the default, hit at the proposal), c = lost
    b = sum(1 for k in real_keys if not hit[hi][k] and hit[lo][k])
    c = sum(1 for k in real_keys if hit[hi][k] and not hit[lo][k])
    p = mcnemar(b, c)
    n = len(real_keys)

    def top5(m, k):
        return set((by[m][k].get("keys") or [])[:5])

    gained, lost = [], []
    for k in real_keys:
        targets = set(by[hi][k].get("targets") or set(by[lo][k].get("targets") or []))
        for t in sorted(targets & (top5(lo, k) - top5(hi, k))):
            gained.append(f"{k[0]}/{k[1]} {k[2]} -> {t}")
        for t in sorted(targets & (top5(hi, k) - top5(lo, k))):
            lost.append(f"{k[0]}/{k[1]} {k[2]} -> {t}")
    noise = {m: sum(1 for r in by[m].values() if _noise(r)) for m in margins}
    conf = {m: sum(1 for k in real_keys if _confused(by[m][k])) for m in margins}
    hits = {m: sum(hit[m].values()) for m in margins}

    print(f"\n== margins {hi} (default) vs {lo} (proposal) | rows {n} real | git_sha {rows[0].get('git_sha')}")
    print(f"{'margin':>8} {'hit5':>9} {'noise':>6} {'confused':>9}")
    for m in (lo, hi):
        print(f"{m:>8} {hits[m]:4d}/{n} {noise[m]:6d} {conf[m]:9d}")
    print(f"paired misses: b={b} (rescued at {lo}), c={c} (lost at {lo}), exact McNemar p={p:.4f}")
    print(f"gained targets in the top five ({len(gained)}): " + ("; ".join(gained) or "-"))
    print(f"lost targets in the top five ({len(lost)}): " + ("; ".join(lost) or "-"))
    # the gate judges THE PROPOSAL (lo) against the current default (hi)
    guards = {
        "no target lost": not lost,
        f"noise not higher ({noise[lo]} <= {noise[hi]})": noise[lo] <= noise[hi],
        f"confusion not higher ({conf[lo]} <= {conf[hi]})": conf[lo] <= conf[hi],
        f"pooled McNemar p < 0.05 ({p:.3f})": p < 0.05,
    }
    for name, ok in guards.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    verdict = all(guards.values())
    print(f"  => gate {'PASSES' if verdict else 'DOES NOT PASS'} on these files")


def main() -> None:
    if sys.argv[1:2] == ["--margin-pair"]:
        report_margins(sys.argv[2:])
        return
    paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(os.path.dirname(__file__), "..",
                               "results", "provenance_household_dev*.jsonl")))
    for path in paths:
        report(path)


if __name__ == "__main__":
    main()