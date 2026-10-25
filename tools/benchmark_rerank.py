"""Benchmark the five rescoring modes (RFC #9) over the multilingual set.

Measures, per mode and per language: recall@5 (the ground-truth fact in the top-5 before the turn), wrong-entity
hits (a fact about another person in the top-5 when the query names one), and the noise on questions that must recall
nothing. A markdown table comes out — the shape itpartypattaya asked for in issue #9.

Quality uses a mock rerank endpoint (the order it answers with is the model's ranking). Latency is measured against a
real llama-server with a real reranker (--rerank-url, see --endpoint): 5 modes x N models, run on a CPU box, not in
the CI.

    PYTHONPATH=~/.hermes/hermes-agent python -I tools/benchmark_rerank.py                # quality, mock
    PYTHONPATH=~/.hermes/hermes-agent python -I tools/benchmark_rerank.py --endpoint=http://127.0.0.1:8080/v1/rerank

Modes: off, reorder_only, reorder_blend, gate_only, gate_blend (the two axes of issue #9: where the cross-encoder's
rank sits — before the coverage gate or as the gate — and whether it blends with the blend score or orders only)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# The package tree loads as hermes_plugin_holographic from the checkout (the CI's directory name rule: a valid
# Python identifier). hermes-agent's checkout goes on sys.path first (its `agent`, `tools`, `hermes_cli`).
import importlib.util
import os

_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "hermes_plugin_holographic", _REPO / "__init__.py", submodule_search_locations=[str(_REPO)])
assert _spec and _spec.loader
if "hermes_plugin_holographic" not in sys.modules:
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["hermes_plugin_holographic"] = _mod
    _spec.loader.exec_module(_mod)

from hermes_plugin_holographic import HolographicMemoryProvider  # noqa: E402
from hermes_plugin_holographic import rerank  # noqa: E402

# The set lives in the test module (its CORPUS / QUERIES / PROBES are the benchmark's data). Load it directly: `tests`
# on sys.path would resolve to hermes-agent's own tests/ first.
assert _spec and _spec.loader
_data_spec = importlib.util.spec_from_file_location(
    "holo_lang_data", _REPO / "tests" / "test_holographic_languages.py")
assert _data_spec and _data_spec.loader
data = importlib.util.module_from_spec(_data_spec)
_data_spec.loader.exec_module(data)

# mode -> (rerank_weight, rerank_window): a single regime (review item 6, 2026-10-22); the labels are kept as
# the historical table nomenclature: gate_* = δ window on (0.05), reorder_* = window off.
MODES = {
    "off": ("0.0", "0.0"),
    "reorder_only": ("0.0", "0.0"),
    "reorder_blend": ("0.5", "0.0"),
    "gate_only": ("0.0", "0.05"),
    "gate_blend": ("0.5", "0.05"),
}


class _Response:
    def __init__(self, body: bytes):
        self.body, self.at = body, 0

    def read(self, n=-1):
        piece = self.body[self.at:] if n is None or n < 0 else self.body[self.at:self.at + n]
        self.at += len(piece)
        return piece

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class MockRerank:
    """Stands in for rerank._OPENER: answers in a (mock) ranker's order. A decent ranker by default — a confident top
    (1.0) over a low tail — so the gate modes stay open on relevant queries; `--reverse` is the worst case. Chit-chat
    noise and honest latency come only from a real endpoint (--endpoint); the mock cannot tell a relevant query from a
    greeting, so its gate-mode noise column is not meaningful."""

    def __init__(self, reverse: bool = False):
        self.requests = 0
        self.reverse = reverse  # a ranker that inverts the order (worst case)

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests += 1
        n = len(body["documents"])
        order = list(reversed(range(n))) if self.reverse else list(range(n))
        top_n = min(body.get("top_n", n), n)
        results = [{"index": i, "relevance_score": 1.0 if rank == 0 else 0.5 - rank * 0.01}
                   for rank, i in enumerate(order[:top_n])]
        return _Response(json.dumps({"results": results}).encode())


class RecordingOpener:
    """Wraps the REAL endpoint opener and records one row per HTTP call (PR #10 point 9, (1)).

    The mode row used to carry ``calls: null`` against a real endpoint — only per-mode p50/p95 of the whole
    prefetch wall time, so forward time could not be separated from transport and cold start. Each row now holds
    the payload the server saw (doc counts, per-doc words/chars, the server's own ``usage.prompt_tokens``),
    ``call_ms`` (this HTTP call), ``prefetch_ms`` (the prefetch that enclosed it, backfilled when it ends), the
    pass index and a warm flag — warm-up passes are marked, not silently averaged in."""

    def __init__(self, real):
        self.real = real
        self.rows: list[dict] = []
        self.lang: str | None = None
        self.pass_no = 0
        self.warm = False
        self._prefetch_open = False

    def begin_prefetch(self) -> None:
        self._prefetch_open = True

    def finish_prefetch(self, ms: float) -> None:
        """Stamp the prefetch wall time on every call it enclosed (trailing rows still unstamped)."""
        for row in reversed(self.rows):
            if row["prefetch_ms"] is None:
                row["prefetch_ms"] = round(ms, 2)
            else:
                break
        self._prefetch_open = False

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        docs = body.get("documents") or []
        row = {"call": len(self.rows), "lang": self.lang, "pass": self.pass_no, "warm": self.warm,
               "docs_n": len(docs), "doc_words": [len(d.split()) for d in docs],
               "doc_chars": [len(d) for d in docs],
               "query_words": len((body.get("query") or "").split()), "top_n": body.get("top_n"),
               "prompt_tokens": None, "call_ms": None, "prefetch_ms": None, "ok": False, "err": None}
        t0 = time.perf_counter()
        try:
            resp = self.real.open(request, timeout=timeout)  # kwargs: OpenerDirector.open(data, timeout)
            raw = resp.read()
        except BaseException as exc:  # noqa: BLE001 — a failed call is a row too, then re-raised
            row["call_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            row["err"] = type(exc).__name__
            self.rows.append(row)
            raise
        row["call_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        try:
            parsed = json.loads(raw)
            usage = parsed.get("usage") or {}
            row["prompt_tokens"] = usage.get("prompt_tokens")
            row["ok"] = "results" in parsed
        except (ValueError, TypeError, AttributeError):
            row["err"] = "malformed"
        self.rows.append(row)
        return _Response(raw)


SCALE = "logit"  # rerank_scale of the run config: llama-server emits raw logits (fix-list 19:07)


def _provider(db_path: str, **config) -> HolographicMemoryProvider:
    p = HolographicMemoryProvider(config={"db_path": db_path, "hrr_dim": 1024, **config})
    p.initialize(session_id="bench")
    return p


def _fill(p, lang: str) -> dict:
    """The language's facts in; returns fact_id -> key."""
    back = {}
    for key, (text, entities) in data.CORPUS[lang].items():
        r = json.loads(p.handle_tool_call("fact_store", {"action": "add", "content": text, "entities": entities}))
        back[r["fact_id"]] = key
    return back


def _inject(p, query: str) -> list[str]:
    """The fact_ids injected before the turn (prefetch), IN OUTPUT ORDER, or [].
    Order matters: recall@5 must count the first five lines (matched top-5, review item 8, 2026-10-22) — off can show
    more than five facts while the unified path cannot, and counting every line mixes the cross-encoder with a
    smaller output."""
    block = p.prefetch(query)
    if not block:
        return []
    # lines after the header are "- #<id> [trust] <text>"
    keys: list[str] = []
    for line in block.splitlines()[1:]:
        try:
            keys.append(line.split()[1][1:])  # the fact_id
        except (IndexError, ValueError):
            pass
    return keys


def _percentile(times: list[float], q: float) -> float:
    if not times:
        return 0.0
    idx = min(len(times) - 1, int(round(q * (len(times) - 1))))
    return sorted(times)[idx]


def _git_sha() -> str:
    """Stem of the branch, so every JSONL row knows at which commit it was captured (item 8: raw JSONL behind the tables)."""
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run(langs: list[str], mode: str, endpoint_url: str | None, reverse: bool,
        warmup: int = 0, repeats: int = 1) -> dict:
    """One mode over the given languages: recall@5, wrong-entity hits, noise, latency p50/p95.

    Latency protocol (PR #10 point 9, (4)): `warmup` passes are executed and marked but excluded from every
    statistic; timing comes from `repeats` warmed passes, so a 90-query set yields 5x more rows and the p95 is
    not decided by a handful of samples. Quality is counted ONCE — on the first warmed pass — so hit5/asked
    keeps its meaning and repeats do not inflate it."""
    weight, window = MODES[mode]
    config = {"rerank": "off" if mode == "off" else "on"}
    if mode != "off":
        config.update({"rerank_weight": weight, "rerank_window": window,
                       "rerank_scale": SCALE, "rerank_model": "bench-reranker"})
        config["rerank_url"] = endpoint_url or "https://rerank.example.net/v1/rerank"

    # Mock: the order the fake ranker answers with is what the modes see.
    mock = MockRerank(reverse=reverse)
    real_opener = rerank._OPENER
    recorder = RecordingOpener(real_opener) if endpoint_url else None
    if not endpoint_url:
        rerank._OPENER = mock
    else:
        rerank._OPENER = recorder  # per-call rows, even against a live endpoint (item 9 (1))

    hit5 = asked = wrong = quiet = negatives = 0
    out_lines = 0  # how many lines recall actually returned (matched top-5 counts the first five)
    queries = 0
    times: list[float] = []
    failures = 0  # ranked is None: fallbacks to the coverage order (counted in rerank.Rescorer)
    try:
        for li, lang in enumerate(langs):
            if recorder is not None:
                recorder.lang = lang
            p = _provider(f"/tmp/bench_{lang}_{mode}.db", **config)
            back = _fill(p, lang)
            for pass_no in range(warmup + repeats):
                warm = pass_no < warmup
                quality = pass_no == warmup  # quality is counted once — on the first warmed pass
                if recorder is not None:
                    recorder.pass_no, recorder.warm = pass_no, warm
                for i, (query, relevant) in enumerate(data.QUERIES[lang]):
                    if quality:
                        queries += 1
                    p.on_session_switch(f"bench{li}_{i}", reset=True)
                    t0 = time.perf_counter()
                    if recorder is not None:
                        recorder.begin_prefetch()
                    injected = _inject(p, query)
                    dt = time.perf_counter() - t0
                    if recorder is not None:
                        recorder.finish_prefetch(dt * 1000.0)
                    if not warm:
                        times.append(dt)
                    if not quality:
                        continue  # out_lines/hit5 only on the warmed pass: repeats do not multiply quality
                    out_lines += len(injected)
                    # matched top-5: only the first five lines are counted — otherwise the score mixes precision with the
                    # output size (off can emit more than five facts, unified cannot)
                    top5 = injected[:5]
                    found_keys = {back[int(k)] for k in top5 if int(k) in back}
                    if relevant:
                        asked += 1
                        hit5 += bool(found_keys & relevant)
                        # wrong-entity: the query names a person (PROBES); a fact about the OTHER person got into the
                        # top-5 (e.g. Eva's school recalled for "what scooter does Anton ride?").
                        persons = dict(data.PROBES[lang])
                        for name, _own in persons.items():
                            if name in query:
                                other = set().union(*(v for n, v in persons.items() if n != name))
                                wrong += len(found_keys & other)
                    else:
                        negatives += 1
                        quiet += not injected
            failures += getattr(getattr(p, "_rescorer", None), "failures", 0)
            p.shutdown()
    finally:
        rerank._OPENER = real_opener

    calls = mock.requests if not endpoint_url else None
    if recorder is not None:
        for row in recorder.rows:
            row["mode"] = mode
        calls = recorder.rows
    return {
        "mode": mode,
        "recall5": (hit5 / asked) if asked else 0.0,
        "hit5": hit5,
        "asked": asked,
        "wrong": wrong,
        "noise": (negatives - quiet) / negatives if negatives else 0.0,
        "quiet": quiet,
        "negatives": negatives,
        "p50_ms": _percentile(times, 0.50) * 1000.0,
        "p95_ms": _percentile(times, 0.95) * 1000.0,
        "timings_n": len(times),
        "warmup": warmup,
        "repeats": repeats,
        "out_lines": out_lines,
        "out_avg": round(out_lines / max(1, queries), 2),
        "calls": calls,
        "failures": failures,
        "git_sha": _git_sha(),
        "langs": langs,
    }


def table(rows: list[dict], endpoint: str | None) -> str:
    head = "| mode | recall@5 (top-5) | hit5/asked | wrong-entity@5 | noise | out avg | p50 (ms) | p95 (ms) |\n"
    head += "|---|---|---|---|---|---|---|---|\n"
    body = []
    for r in rows:
        recall = f"{r['recall5']:.0%}"
        body.append(f"| {r['mode']} | {recall} | {r['hit5']}/{r['asked']} | {r['wrong']} | "
                    f"{r['noise']:.0%} | {r['out_avg']:.1f} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} |")
    note = ("\nEndpoint: " + (endpoint or "mock (order only; latency is the mock's, not a server's)"))
    failed = sum(r.get("failures", 0) for r in rows)
    if failed:
        note += f"\nranked is None (fallbacks to coverage order): {failed}"
    return head + "\n".join(body) + note + "\n"


def main(argv: list[str]) -> None:
    global SCALE
    ap = argparse.ArgumentParser(description="Benchmark the rescoring modes over the multilingual set (RFC #9).")
    ap.add_argument("--langs", default="en,ru,zh", help="comma-separated language keys (default: en,ru,zh)")
    ap.add_argument("--modes", default=",".join(MODES), help="comma-separated modes to run (default: all five)")
    ap.add_argument("--endpoint", default=None,
                    help="a real /v1/rerank URL (llama-server); default: a mock ranker")
    ap.add_argument("--rerank-scale", default="logit", choices=["logit", "unit"],
                    help="rerank_scale for the config (llama-server logits -> logit; [0,1] endpoint -> unit)")
    ap.add_argument("--reverse", action="store_true", help="the mock ranker inverts the order (worst case)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a markdown table")
    ap.add_argument("--label", default=None, help="run label (model x device) stamped into every row")
    ap.add_argument("--out", default=None, help="append rows as JSON-lines to this file (accumulate runs)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="warm-up passes excluded from every statistic (latency protocol, item 9 (4))")
    ap.add_argument("--repeats", type=int, default=1,
                    help="warmed passes to time; quality is still counted once (default: 1)")
    ap.add_argument("--calls-out", default=None,
                    help="write one JSON line per HTTP call (item 9 (1): tokens, doc lengths, call/prefetch ms)")
    args = ap.parse_args(argv)
    SCALE = args.rerank_scale

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    rows = [run(langs, mode, args.endpoint, args.reverse, warmup=args.warmup, repeats=args.repeats)
            for mode in modes]
    if args.label:
        for r in rows:
            r["label"] = args.label
    if args.out:
        # Overwrite: each cell writes its own file fresh, so re-runs never duplicate rows.
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.calls_out:
        with open(args.calls_out, "w", encoding="utf-8") as fh:
            for r in rows:
                for call in (r.get("calls") or []):
                    if isinstance(call, dict):  # a mock run writes a counter, not per-call rows
                        fh.write(json.dumps(call, ensure_ascii=False) + "\n")
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(table(rows, args.endpoint))


if __name__ == "__main__":
    main(sys.argv[1:])