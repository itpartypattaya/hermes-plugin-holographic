"""Exact McNemar preview of the PR gate on a probe run (dev or test).

The gate agreed in PR #10 (maintainer, 2026-10-22), to be applied to the test split:
either a real paired gain in hit@5 (>= +5 points and exact McNemar p < 0.05) or fewer
irrelevant facts without losing the right ones — and on BOTH branches wrong-entity
(`confused`) and noise (`neg`) no higher than `off`, with the rows paired. This tool
computes both branches for every mode against `off` on whatever provenance_probe output
files you point it at:

    python tools/mcnemar_gate.py                      # all dev files
    python tools/mcnemar_gate.py results/provenance_household_dev_on.jsonl
    python tools/mcnemar_gate.py --pair margin <two files>       # PR #17 (--margin-pair is the same)
    python tools/mcnemar_gate.py --pair embed_text <two files>   # issue #95, gate G1

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
    """A confuser in the top five, read from the row's OWN field: household stamps `confused`, multi a
    `wrong5` list, contrast a boolean `wrong` already computed against the top five.

    No fallback between the fields: multi's `wrong` covers the whole output, so an empty `wrong5` with a
    foreign key below the fifth position is not confusion — counting it anyway inflates the default side
    and hides a rise (the reviewer's reproduction of PR #31: top-five confusion 0 -> 1, reported 1 -> 1,
    gate passes)."""
    row_set = r.get("set")
    if row_set == "household":
        field = "confused"
    elif row_set == "multi":
        field = "wrong5"
    elif row_set in ("contrast_ru", "contrast_en", "contrast_zh"):
        field = "wrong"
    else:
        raise SystemExit(f"unknown row shape — unsupported set {row_set!r}")
    if field not in r:
        raise SystemExit(f"unknown row shape — {row_set} requires {field}")
    return bool(r[field])


def _noise(r: dict) -> bool:
    """A row that showed anything it should not: chit with output, or a no-answer real row (a real row
    without targets — the maintainer's PR #17 recipe counts them as noise, unlike the mode report above)."""
    if (r.get("out_n") or 0) <= 0:
        return False
    if r.get("kind") == "chit":
        return True
    return r.get("kind") == "real" and not (r.get("targets") or [])


# One clean head for a whole paired comparison: a field that differs between the two runs is a second change.
_HEAD = ("git_sha", "emb", "mode", "scale", "endpoint")
# Issue #95 adds the embedding model and its vector size (two models are not one change of vector text), the margin
# and the keywords in the store (the same file and coverage on both sides: only the vector text may differ).
_EMBED_TEXT_HEAD = _HEAD + ("emb_model", "emb_dim", "margin", "kw_file", "kw_coverage", "names_fit")


def _paired(paths: list[str], field: str, head: tuple, value=lambda v: v,
            same: tuple = ("q", "kind", "split")) -> tuple[list[dict], list, dict, list]:
    """Rows of two runs that differ only in *field*, paired per (set, lang, qid): (rows, the two values sorted, value ->
    key -> row, the keys of real rows). Refuses — never reports — when the field has not exactly two values, a row
    does not carry it, the *head* differs or is missing, a key repeats or has no partner, or a pair differs in its
    query, kind, split or targets."""
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in open(path, encoding="utf-8") if line.strip())
    unstamped = [i for i, r in enumerate(rows) if r.get(field) in (None, "")]
    if unstamped:
        raise SystemExit(f"{len(unstamped)} rows carry no {field} (row {unstamped[0]}) — a row that does "
                         f"not say which run it came from cannot be paired")
    values = sorted({value(r[field]) for r in rows})
    if len(values) != 2:
        raise SystemExit(f"expected rows stamped with exactly two values of {field}, got {values} — "
                         f"run provenance_probe twice and pass both files")
    # One clean head for the WHOLE comparison — the endpoint too: two rerank/embedding servers are not one
    # change. A field no row carries refuses the run instead of passing it (PR #31, item 2: the old
    # filter dropped empty values, so a run stamped with no endpoint at all was a run nobody checked).
    for name in head:
        missing = [i for i, r in enumerate(rows) if r.get(name) in (None, "")]
        if missing:
            raise SystemExit(f"{name} missing from {len(missing)} rows (row {missing[0]}) — the pairing "
                             f"needs one clean head")
        vals = sorted({str(r[name]) for r in rows})
        if len(vals) > 1:
            raise SystemExit(f"mixed runs — {name}={vals}: the pairing needs one clean head")
    seen = {m: set() for m in values}
    for r in rows:
        key = (r.get("set"), r.get("lang"), r.get("qid"))
        if key in seen[value(r[field])]:
            raise SystemExit(f"duplicate rows for {field} {r[field]} {key} — the report is impossible")
        seen[value(r[field])].add(key)
    # pairing is per (set, lang, qid): the field is what differs, not part of the pair key
    both = seen[values[0]] & seen[values[1]]
    only = {m: sorted(seen[m] - both, key=str) for m in values}
    if only[values[0]] or only[values[1]]:
        raise SystemExit(f"unpaired rows: {only[values[0]][:3]} vs {only[values[1]][:3]}")
    by = {m: {(r.get("set"), r.get("lang"), r.get("qid")): r for r in rows if value(r[field]) == m} for m in values}
    # Inside a pair only the field may differ: the query, the row kind, the split and the target set are
    # compared field by field, and a field one side does not carry refuses the pair (PR #31, item 2 —
    # real keys used to come from the proposal alone and the targets from the default with a fallback,
    # so a pair of different questions or different targets was reported as if it were one question).
    for key in sorted(both, key=str):
        a, b = by[values[0]][key], by[values[1]][key]
        for name in same:
            if name not in a or name not in b:
                raise SystemExit(f"unpaired {key}: {name} missing — a pair may differ only in {field}")
            if a[name] != b[name]:
                raise SystemExit(f"unpaired {key}: {name} {a[name]!r} != {b[name]!r}")
        if "targets" not in a or "targets" not in b:
            raise SystemExit(f"unpaired {key}: targets missing — a pair may differ only in {field}")
        if set(a["targets"]) != set(b["targets"]):
            raise SystemExit(f"unpaired {key}: targets {a['targets']} != {b['targets']}")
    # both sides carry the same kind (checked above): neither side alone decides what is real
    real_keys = [k for k, r in by[values[0]].items() if r.get("kind") == "real"]
    return rows, values, by, real_keys


def report_margins(paths: list[str]) -> None:
    """Two-margin pairing for the PR #17 gate: rows stamped with `margin` (provenance_probe
    --recall-margin) are paired per (set, lang, qid) and compared 0.6 against 1.0 on one head.

    Reports hit@5 with exact McNemar on the pooled rows, the gained and lost top-five targets as
    `set/lang qid -> target`, noise (chit AND no-answer rows with output) and confusion — the four
    guards of the recipe: no target lost anywhere, noise not higher, confusion not higher, and the
    rescue passing p < 0.05 pooled (2026-10-24)."""
    rows, margins, by, real_keys = _paired(paths, "margin", _HEAD, value=float)
    lo, hi = margins  # sorted ascending: lo is the proposal (0.6), hi the current default (1.0)

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
        targets = set(by[hi][k]["targets"])  # equal on both sides: checked per pair above
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


def _covered(r: dict) -> bool:
    """Did this row's store have every vector it should (provenance_probe stamps the counts per store)?"""
    return isinstance(r.get("vectors"), int) and r.get("vectors") == r.get("vectors_expected")


def report_embed_text(paths: list[str]) -> bool:
    """Gate G1 of issue #95 (decided 2026-11-16, before any run): two arms that differ only in what fact vectors are
    made from — `embed_text` content against content+keywords — on one head, one embedding model and one keywords
    file and coverage, every store fully covered. A new vector text moves every similarity, so targets and noise rows
    can both come and go; the gate counts both directions:

    - the gain: exact McNemar on top-five hits, b = rescued against c = lost, p < 0.05 with b > c;
    - every lost top-five target is listed as `set qid:target, lang` (listed, not a veto);
    - noise (chit and no-answer rows with output) and confusion as rows added minus rows removed, each <= 0.

    Reported beside the gate: the target anywhere in the output (a fact by meaning comes after up to five word matches),
    facts that came in by meaning, and the top-five count per set and language. Returns whether the gate passes."""
    # each store's keyword count too: a pair whose stores hold different keywords is not one question
    rows, values, by, real_keys = _paired(paths, "embed_text", _EMBED_TEXT_HEAD,
                                          same=("q", "kind", "split", "kw_facts"))
    if any(str(r["git_sha"]).endswith("-dirty") for r in rows):
        raise SystemExit("rows captured on a tree with uncommitted changes (git_sha ...-dirty): measure at a commit")
    if any(r.get("split") == "dev" for r in rows):
        raise SystemExit("dev rows: the decisive pool is the test split and the sets without a split")
    if values != ["content", "content+keywords"]:
        raise SystemExit(f"expected embed_text content and content+keywords, got {values}")
    if any(str(r["emb"]) != "on" for r in rows):
        raise SystemExit("embed_text pairs need embeddings on: off, both arms are the same run")
    short = [i for i, r in enumerate(rows) if not _covered(r)]
    if short:
        r = rows[short[0]]
        raise SystemExit(f"{len(short)} rows from stores with incomplete vector coverage (row {short[0]}: "
                         f"{r.get('vectors')} of {r.get('vectors_expected')}) — the run is discarded")
    base, prop = "content", "content+keywords"
    n = len(real_keys)

    def mcnemar_of(field):
        hit = {m: {k: bool(by[m][k].get(field)) for k in real_keys} for m in (base, prop)}
        b = sum(1 for k in real_keys if not hit[base][k] and hit[prop][k])
        c = sum(1 for k in real_keys if hit[base][k] and not hit[prop][k])
        return {m: sum(hit[m].values()) for m in (base, prop)}, b, c, mcnemar(b, c)

    def moved(flag) -> tuple[int, int]:
        """(rows the proposal added, rows it removed) of a per-row flag."""
        keys = list(by[base])
        added = sum(1 for k in keys if flag(by[prop][k]) and not flag(by[base][k]))
        removed = sum(1 for k in keys if flag(by[base][k]) and not flag(by[prop][k]))
        return added, removed

    hits5, b, c, p = mcnemar_of("hit5")
    hits_all, b_all, c_all, p_all = mcnemar_of("hit")
    lost, gained = [], []
    for k in real_keys:
        targets = set(by[base][k]["targets"])
        top = {m: set((by[m][k].get("keys") or [])[:5]) for m in (base, prop)}
        lost += [f"{k[0]} {k[2]}:{t}, {k[1]}" for t in sorted(targets & (top[base] - top[prop]))]
        gained += [f"{k[0]} {k[2]}:{t}, {k[1]}" for t in sorted(targets & (top[prop] - top[base]))]
    noise_add, noise_rm = moved(_noise)
    conf_add, conf_rm = moved(lambda r: r.get("kind") == "real" and _confused(r))
    by_meaning = {m: sum(1 for r in by[m].values() for tags in (r.get("sources") or {}).values() if "meaning" in tags)
                  for m in (base, prop)}

    head = rows[0]
    print(f"\n== embed_text content vs content+keywords | {n} real, {len(by[base]) - n} other rows | "
          f"git_sha {head['git_sha']} | {head['emb_model']} ({head['emb_dim']} dims) | keywords {head['kw_file']} "
          f"{head['kw_coverage']} | splits {sorted({str(r.get('split')) for r in rows})}")
    print(f"hit@5: {hits5[base]}/{n} -> {hits5[prop]}/{n}; b={b} rescued, c={c} lost, exact McNemar p={p:.4f}")
    print(f"target anywhere in the output: {hits_all[base]}/{n} -> {hits_all[prop]}/{n}; b={b_all}, c={c_all}, "
          f"p={p_all:.4f}")
    print(f"facts by meaning in the output: {by_meaning[base]} -> {by_meaning[prop]}")
    print(f"lost targets in the top five ({len(lost)}): " + ("; ".join(lost) or "-"))
    print(f"gained targets in the top five ({len(gained)}): " + ("; ".join(gained) or "-"))
    slices = sorted({(k[0], k[1]) for k in real_keys}, key=str)
    for s in slices:
        keys = [k for k in real_keys if (k[0], k[1]) == s]
        sb = sum(1 for k in keys if not by[base][k].get("hit5") and by[prop][k].get("hit5"))
        sc = sum(1 for k in keys if by[base][k].get("hit5") and not by[prop][k].get("hit5"))
        print(f"  {s[0]}/{s[1]}: {len(keys)} real, b={sb}, c={sc}")
    guards = {
        f"gain: McNemar p < 0.05 with b > c (b={b}, c={c}, p={p:.3f})": p < 0.05 and b > c,
        f"noise: added {noise_add} - removed {noise_rm} <= 0": noise_add - noise_rm <= 0,
        f"confusion: added {conf_add} - removed {conf_rm} <= 0": conf_add - conf_rm <= 0,
    }
    for name, ok in guards.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    verdict = all(guards.values())
    print(f"  => gate G1 {'PASSES' if verdict else 'DOES NOT PASS'} on these files")
    return verdict


def main() -> None:
    if sys.argv[1:2] == ["--margin-pair"]:
        report_margins(sys.argv[2:])
        return
    if sys.argv[1:2] == ["--pair"]:
        field = sys.argv[2] if len(sys.argv) > 2 else ""
        if field == "margin":
            report_margins(sys.argv[3:])
        elif field == "embed_text":
            report_embed_text(sys.argv[3:])
        else:
            raise SystemExit("--pair margin|embed_text <files>")
        return
    paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(os.path.dirname(__file__), "..",
                               "results", "provenance_household_dev*.jsonl")))
    for path in paths:
        report(path)


if __name__ == "__main__":
    main()