"""What the presence gate adds (issue #59), measured as a paired delta on synthetic inputs.

The project records no baseline for ``sync_turn`` or prefetch latency (its timing tests pin the screen), so
this harness measures the baseline itself: every scenario runs the gate off and on — same provider, same
inputs, phases interleaved to cancel drift — and reports the added cost next to each phase's p50/p95.

Inputs are synthetic and neutral, shaped the way the plugin and the core emit them: recall blocks wrapped by
``build_memory_context_block``, ``fact_store`` results in both encodings, noise rows without
``<memory-context>``, one 200 KB tool row (the worst case for scanning), neutral facts for recall. No
session data and no paths outside the repository; the output is aggregates only. One machine, fixed inputs:
the numbers are a cost, reproducible in any CI — not the p95 of a live session.

    PYTHONPATH=~/.hermes/hermes-agent python -I tools/presence_delta.py [--runs 30]

Scenarios:
  sync rows=10/100/1000   the snapshot work the gate adds to sync_turn (its cost note: scan only rows whose
                           text holds <memory-context>)
  prefetch no-snapshot     the gate branch alone (a snapshot lookup that misses)
  prefetch snapshot        a present snapshot the gate must check (lines/pairs miss)
  identity                 gate off vs on without a snapshot: byte-identical prefetch output
  state                    bytes of emitted-line text the gate keeps per tracked injection
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

# The package tree loads as hermes_plugin_holographic from the checkout (the CI's directory name rule: a valid
# Python identifier). hermes-agent's checkout goes on sys.path first (its `agent`, `tools`, `hermes_cli`).
_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.memory_context import build_memory_context_block  # noqa: E402
from hermes_plugin_holographic import _PREFETCH_HEADER, HolographicMemoryProvider  # noqa: E402

_FACT = "The nightly backup number {0} runs from box {0}."
_QUERIES = ("nightly backup", "backup runs", "which box runs the backup")
_BIG_ROW = ("cache line " + "x" * 90) * 2300  # ~200 KB: the worst noise row for a scanner


def _provider() -> HolographicMemoryProvider:
    db = Path(tempfile.mkdtemp(prefix="presence-delta-")) / "m.db"
    p = HolographicMemoryProvider(config={"db_path": str(db), "hrr_dim": 512})
    p.initialize(session_id="s")
    return p


def _rows(n: int) -> list:
    """A long session's shape: every 10th row a recall block, fact_store results in both encodings, noise
    without <memory-context>, one 200 KB tool row at the end."""
    block = build_memory_context_block(
        _PREFETCH_HEADER + "\n" + "\n".join(f"- #{i} [0.5] {_FACT.format(i)}" for i in range(1, 6)))
    facts = [{"fact_id": i, "content": _FACT.format(i)} for i in range(1, 6)]
    noise = "The deploy log line repeats: task {0} finished ok; ".format
    rows = []
    for i in range(n):
        if i % 10 == 0:
            rows.append({"role": "user", "api_content": block})
        elif i % 10 == 5:
            rows.append({"role": "tool", "tool_name": "fact_store",
                         "content": json.dumps({"found": facts[:3]}, ensure_ascii=(i % 20) != 5)})
        elif i == n - 1:
            rows.append({"role": "tool", "tool_name": "terminal", "content": _BIG_ROW})
        else:
            rows.append({"role": "assistant", "content": noise(i)})
    return rows


def _paired(run_off, run_on, runs: int, warmup: int = 3):
    """Interleaved off/on on one process; the median of paired differences is the gate's added cost."""
    for _ in range(warmup):
        run_off()
        run_on()
    off, on = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        run_off()
        off.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        run_on()
        on.append((time.perf_counter() - t0) * 1000)
    return off, on


def _pct(xs, q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def _line(fact_id: int) -> str:
    return f"- #{fact_id} [0.5] {_FACT.format(fact_id)}"


def _seed_recall(p: HolographicMemoryProvider) -> str:
    """A store with the facts, all injected long ago (the window expired), so the gate's branch is live."""
    key = "s|" + (p._scope_key or "")
    for i in range(1, 6):
        p.handle_tool_call("fact_store", {"action": "add", "origin": "said", "content": _FACT.format(i)})
        p._remember(key, {"fact_id": i, "content": _FACT.format(i)}, 1, _line(i))
    p._turns[key] = 10  # the next prefetch is turn 11: 11 - 1 > 6, the window is long expired
    return key


def _prefetch(p: HolographicMemoryProvider, query: str):
    return p.prefetch(query, session_id="s")


def main() -> int:
    ap = argparse.ArgumentParser(description="paired gate-off/gate-on cost on synthetic inputs")
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()
    runs = args.runs

    print(f"{'scenario':26} {'off_p50':>8} {'off_p95':>8} {'on_p50':>8} {'on_p95':>8} {'d_p50':>8} {'d_p95':>8}")
    rows_out = []

    # -- sync_turn: the snapshot work the gate adds, per session size -------------------------------
    p = _provider()
    try:
        for n in (10, 100, 1000):
            msgs = _rows(n)

            def run_off(msgs=msgs):
                p._presence_gate = False
                p.sync_turn("u", "a", session_id="s", messages=msgs)

            def run_on(msgs=msgs):
                p._presence_gate = True
                p.sync_turn("u", "a", session_id="s", messages=msgs)

            off, on = _paired(run_off, run_on, runs)
            diff = [b - a for a, b in zip(off, on)]
            rows_out.append((f"sync rows={n}", off, on, diff))

        # -- prefetch: the gate branch, with and without a snapshot to check ----------------------
        key = _seed_recall(p)
        scenarios = (
            ("prefetch no-snapshot", lambda: p._snapshots.pop(key, None)),
            ("prefetch snapshot", lambda: p._publish_snapshot(
                {"key": key, "generation": 0, "turn": 10, "lines": set(), "pairs": set(), "summary": False})),
        )
        for i, (name, arm) in enumerate(scenarios):
            arm()  # arm the scenario once; _paired's warmup keeps it armed

            def run_off(arm=arm, qi=i):
                p._presence_gate = False
                _prefetch(p, _QUERIES[qi % len(_QUERIES)])

            def run_on(arm=arm, qi=i):
                p._presence_gate = True
                arm()
                _prefetch(p, _QUERIES[qi % len(_QUERIES)])

            off, on = _paired(run_off, run_on, runs)
            diff = [b - a for a, b in zip(off, on)]
            rows_out.append((name, off, on, diff))
    finally:
        p.shutdown()

    for name, off, on, diff in rows_out:
        print(f"{name:26} {_pct(off, .5):8.3f} {_pct(off, .95):8.3f} {_pct(on, .5):8.3f} "
              f"{_pct(on, .95):8.3f} {statistics.median(diff):8.3f} {_pct(diff, .95):8.3f}")

    # -- identity: with no snapshot the gate changes nothing byte-wise -----------------------------
    p = _provider()
    try:
        _seed_recall(p)
        p._presence_gate = False
        off_out = _prefetch(p, _QUERIES[0])
        p._injected.clear()
        p._turns.clear()
        p._presence_gate = True
        on_out = _prefetch(p, _QUERIES[0])
        print(f"identity: gate off == gate on (no snapshot, byte-identical): {off_out == on_out}")
        if off_out != on_out:
            return 1
    finally:
        p.shutdown()

    # -- state: what the gate keeps per tracked injection ------------------------------------------
    sample = [_line(i) for i in range(1, 6)]
    print(f"state: emitted lines kept for 5 injections: {sum(len(s) for s in sample)} bytes of text")
    print(f"runs={runs}, interleaved off/on, one machine — a cost on fixed inputs, not a live-session p95")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
