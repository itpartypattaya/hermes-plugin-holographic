# -*- coding: utf-8 -*-
"""Two-hop links experiment (plans/2026-10-28-two-hop.md): the real provider, no model, no network.

Every write goes through fact_store (add / change / end / forget), every recall through prefetch(); fact ids are
read from the block's `- #id` lines. Rerank and embeddings stay off (the defaults).

Run (Git Bash; HERMES_AGENT_REPO = a hermes-agent checkout):
    python -I tools/two_hop_eval.py run --repo <plugin checkout> --mode main --out results/two_hop_main.jsonl
    python -I tools/two_hop_eval.py run --repo . --mode hop --out results/two_hop_hop.jsonl
    python -I tools/two_hop_eval.py report results/two_hop_main.jsonl results/two_hop_hop.jsonl

`--repo` loads the plugin from another checkout (origin/main for the baseline); `--mode main` leaves `two_hop`
out of the config, `--mode hop` sets it to true. Rows: one per question per mode.

The prototype was NOT adopted (it failed the pre-registered threshold), so `two_hop` does not exist on main. To
reproduce the hop rows, apply the prototype to a scratch checkout first and point `--repo` at it:
    git apply results/two_hop_prototype.patch
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))
import two_hop_data as data  # noqa: E402

_LINE = re.compile(r"^- #(\d+) \[")
_VIA = re.compile(r"\(via #(\d+)\)")
REPEATS = 5


def _load(repo: Path):
    agent = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
    if str(agent) not in sys.path:
        sys.path.insert(0, str(agent))
    spec = importlib.util.spec_from_file_location("hermes_plugin_holographic", repo / "__init__.py",
                                                  submodule_search_locations=[str(repo)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["hermes_plugin_holographic"] = module
    spec.loader.exec_module(module)
    return module.HolographicMemoryProvider


def _call(provider, args: dict) -> dict:
    return json.loads(provider.handle_tool_call("fact_store", args))


def build(provider_cls, work: Path, mode: str) -> tuple[dict, dict, dict]:
    os.environ["HERMES_HOME"] = str(work)
    config = {"db_path": str(work / "memory.db"), "scope": "chat"}
    if mode == "hop":
        config["two_hop"] = "true"
    providers, ids, failures = {}, {s: {} for s in data.STORES}, []
    filler = data.filler()
    for store in data.STORES:
        p = provider_cls(config=dict(config))
        p.initialize(session_id="build", chat_id=store, hermes_home=str(work))
        providers[store] = p
        items = [("s", key, text, ents) for key, (st, text, ents) in data.SCENARIO.items() if st == store]
        items += [("f", None, text, ents) for text, ents in filler[store]]
        random.Random(f"order-{store}").shuffle(items)
        for kind, key, text, ents in items:
            args = {"action": "add", "content": text, "origin": "said"}
            if ents:
                args["entities"] = ents
            out = _call(p, args)
            if out.get("status") not in ("added", "exists") or "fact_id" not in out:
                failures.append({"store": store, "key": key, "result": out})
                continue
            if key:
                ids[store][key] = out["fact_id"]
    for action, key, text, ents in data.OPS:
        store = data.SCENARIO[key][0]
        args = {"action": action, "fact_id": ids[store][key]}
        if action == "change":
            args.update(content=text, entities=ents, origin="said")
        out = _call(providers[store], args)
        if "error" in out:
            failures.append({"op": action, "key": key, "result": out})
    counts = {s: _call(providers[s], {"action": "stats"}).get("facts") for s in data.STORES}
    return providers, ids, {"failures": failures, "counts": counts}


def _texts_of(key: str) -> list[str]:
    texts = [data.SCENARIO[key][1]]
    texts += [t for a, k, t, _ in data.OPS if k == key and t]
    return texts


def run(args) -> None:
    provider_cls = _load(Path(args.repo).resolve())
    with tempfile.TemporaryDirectory(prefix="two_hop_") as tmp:
        work = Path(tmp)
        t0 = time.perf_counter()
        providers, ids, meta = build(provider_cls, work, args.mode)
        meta["build_seconds"] = round(time.perf_counter() - t0, 1)
        print(json.dumps({"mode": args.mode, **meta}, ensure_ascii=False), file=sys.stderr)
        rows = []
        for index, (qid, kind, store, lang, question, gold, forbidden) in enumerate(data.QUESTIONS):
            p = providers[store]
            by_id = {v: k for k, v in ids[store].items()}
            times, block = [], ""
            for rep in range(REPEATS):
                start = time.perf_counter()
                out = p.prefetch(question, session_id=f"{args.mode}-{qid}-{rep}")
                times.append((time.perf_counter() - start) * 1000)
                if rep == 0:
                    block = out
            lines = [ln for ln in block.splitlines() if ln.strip()]
            fact_lines = [ln for ln in lines if _LINE.match(ln)]
            other = [ln for ln in lines[1:] if not _LINE.match(ln) and not ln.startswith("Memory attempt:")]
            injected = [int(_LINE.match(ln).group(1)) for ln in fact_lines]
            hop_ids = [int(_LINE.match(ln).group(1)) for ln in fact_lines if _VIA.search(ln)]
            names = [by_id.get(i, f"#{i}") for i in injected]
            forbidden_hits = [k for k in forbidden if any(t in block for t in _texts_of(k))]
            rows.append({
                "qid": qid, "type": kind, "store": store, "lang": lang, "held_out": data.held_out(index),
                "mode": args.mode, "injected": names, "hop": [by_id.get(i, f"#{i}") for i in hop_ids],
                "gold": gold, "gold_all": all(g in names for g in gold),
                "gold_hits": sum(1 for g in gold if g in names),
                "irrelevant": [n for n in names if n not in gold],
                "forbidden_hits": forbidden_hits, "synthesized_lines": len(other),
                "ms": [round(t, 2) for t in times],
            })
        for p in providers.values():
            p.shutdown()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {args.out}", file=sys.stderr)


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _p95(values):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]


CONTROLS = ("one_hop", "namesake", "no_answer")
# Facts that no longer exist in the asked store: ended or forgotten, or kept in another chat's file
REMOVED = {k for action, k, _t, _e in data.OPS if action != "change"} | {k for k, v in data.SCENARIO.items() if v[0] == "work-b"}
SAFETY = ("changed", "forgotten", "cross_scope")


def summarize(rows, subset=lambda r: True) -> dict:
    rows = [r for r in rows if subset(r)]
    out = {}
    for kind in ("two_hop", "one_hop", "namesake", "changed", "forgotten", "cross_scope", "no_answer"):
        part = [r for r in rows if r["type"] == kind]
        if part:
            out[f"recall_{kind}"] = sum(r["gold_all"] for r in part) / len(part)
            out[f"n_{kind}"] = len(part)
    ctrl = [r for r in rows if r["type"] in CONTROLS]
    injected = sum(len(r["injected"]) for r in ctrl)
    irrelevant = sum(len(r["irrelevant"]) for r in ctrl)
    out["controls_n"] = len(ctrl)
    out["controls_injected"] = injected
    out["controls_irrelevant"] = irrelevant
    out["controls_noise_share"] = irrelevant / injected if injected else 0.0
    out["forbidden_hits"] = sum(len(r["forbidden_hits"]) for r in rows)
    out["synthesized_lines"] = sum(r["synthesized_lines"] for r in rows)
    out["hop_facts"] = sum(len(r["hop"]) for r in rows)
    work = [t for r in rows if r["store"] == "work-a" for t in r["ms"]]
    out["p95_ms_work"] = _p95(work) if work else 0.0
    out["median_ms_work"] = statistics.median(work) if work else 0.0
    return out


def report(args) -> None:
    main, hop = _rows(args.main), _rows(args.hop)
    by_q = {r["qid"]: r for r in main}
    result = {"all": {}, "held_out": {}, "dev": {}}
    for name, sub in (("all", lambda r: True), ("held_out", lambda r: r["held_out"]),
                      ("dev", lambda r: not r["held_out"])):
        result[name] = {"main": summarize(main, sub), "hop": summarize(hop, sub)}
    # forbidden facts the hop mode shows and main does not: attributable to the walk
    attributable = [(r["qid"], k) for r in hop for k in r["forbidden_hits"] if k not in by_q[r["qid"]]["forbidden_hits"]]
    extra_ctrl = sum(len(r["irrelevant"]) - len(by_q[r["qid"]]["irrelevant"]) for r in hop if r["type"] in CONTROLS)
    extra_ctrl_held = sum(len(r["irrelevant"]) - len(by_q[r["qid"]]["irrelevant"])
                          for r in hop if r["type"] in CONTROLS and r["held_out"])
    a, h = result["all"]["main"], result["all"]["hop"]
    ha, hh = result["held_out"]["main"], result["held_out"]["hop"]
    checks = {
        "H1_all": (h["recall_two_hop"] - a["recall_two_hop"]) * 100,
        "H1_held_out": (hh["recall_two_hop"] - ha["recall_two_hop"]) * 100,
        "G1_one_hop": (h["recall_one_hop"] - a["recall_one_hop"]) * 100,
        "G2_noise_share_pp": (h["controls_noise_share"] - a["controls_noise_share"]) * 100,
        "G2_extra_irrelevant_per_control": extra_ctrl / max(h["controls_n"], 1),
        "G3_forbidden_hop_strict": h["forbidden_hits"], "G3_forbidden_attributable": len(attributable),
        "G3_removed_in_hop": sum(1 for r in hop for k in r["forbidden_hits"] if k in REMOVED),
        "G3_removed_in_main": sum(1 for r in main for k in r["forbidden_hits"] if k in REMOVED),
        "G3_synthesized": h["synthesized_lines"],
        "G4_p95_delta_ms": h["p95_ms_work"] - a["p95_ms_work"],
    }
    passed = {
        "H1": checks["H1_all"] >= 20 and checks["H1_held_out"] >= 20,
        "G1": checks["G1_one_hop"] >= 0,
        "G2": checks["G2_noise_share_pp"] <= 2 and checks["G2_extra_irrelevant_per_control"] <= 0.10,
        "G3": checks["G3_removed_in_hop"] == 0 and checks["G3_forbidden_attributable"] == 0
        and checks["G3_synthesized"] == 0,
        "G4": checks["G4_p95_delta_ms"] <= 20,
    }
    default_on = all(passed.values()) and checks["H1_all"] >= 30 and extra_ctrl_held == 0
    summary = {"result": result, "checks": checks, "passed": passed, "adopt": all(passed.values()),
               "default_on": default_on, "attributable_forbidden": attributable,
               "extra_irrelevant_controls_held_out": extra_ctrl_held,
               "per_question": [{"qid": r["qid"], "type": r["type"], "held_out": r["held_out"],
                                 "main": by_q[r["qid"]]["injected"], "hop": r["injected"], "hop_facts": r["hop"],
                                 "gold": r["gold"], "main_ok": by_q[r["qid"]]["gold_all"], "hop_ok": r["gold_all"],
                                 "forbidden_main": by_q[r["qid"]]["forbidden_hits"],
                                 "forbidden_hop": r["forbidden_hits"]} for r in hop]}
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", default=str(_TOOLS.parent))
    r.add_argument("--mode", choices=("main", "hop"), required=True)
    r.add_argument("--out", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("main")
    rep.add_argument("hop")
    rep.add_argument("--json")
    args = parser.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
