# -*- coding: utf-8 -*-
"""Provenance probe: where each fact in top-5 came from before the turn (lexical / meaning / names),
what the coverage filter cut, whether the rerank_window cut it (class E) and why the target was not found
(classes A/D/C).

Patches only the provider instance in a test run (production code stays unchanged):
  search      -> pool (candidates after blend)
  relevant    -> kept / blocked (+ position in the pool, pool top-5 = class D)
  by_meaning  -> source 'meaning'
  entity_facts-> source 'entity'
  rescorer.scores -> the rerank order (rerank top-5)
  _not_recent / _recallable -> 'recent' / 'unsafe'

Run:
    PYTHONPATH=~/.hermes/hermes-agent python -I tools/provenance_probe.py \
        --set multi --langs en,ru,zh,de,ja,it,nl,th --modes off,reorder_only,gate_only
    PYTHONPATH=~/.hermes/hermes-agent python -I tools/provenance_probe.py \
        --set contrast_ru --modes off,gate_only --emb on
    # --endpoint http://127.0.0.1:1235/v1/rerank   a real rerank (otherwise a mock)
    # --emb on|off|both                              embeddings from :1237 (default off)

Issue #95, keywords in a fact's vector — two arms on the same frozen keywords, each in stores of its own:
    EMB_URL=http://127.0.0.1:1241/v1 EMB_MODEL=bge-m3 python -I tools/provenance_probe.py --set household         --split test --modes off --emb on --keywords tools/issue95_keywords/household.json --kw-coverage half         --embed-text content          # then the same with --embed-text content+keywords
    python tools/mcnemar_gate.py --pair embed_text <the two files>

Rows: results/provenance_<set>_<emb>.jsonl (one row per query; mode is a row field, git_sha is the commit).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if "hermes_plugin_holographic" not in sys.modules:  # in tests the package is already loaded by conftest
    _spec = importlib.util.spec_from_file_location(
        "hermes_plugin_holographic", _REPO / "__init__.py", submodule_search_locations=[str(_REPO)])
    assert _spec and _spec.loader
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["hermes_plugin_holographic"] = _mod
    _spec.loader.exec_module(_mod)

from hermes_plugin_holographic import HolographicMemoryProvider  # noqa: E402
from hermes_plugin_holographic import _sendable as _holographic_sendable  # noqa: E402
from hermes_plugin_holographic import embeddings as _holographic_embeddings  # noqa: E402

# mode -> (rerank_weight, rerank_window): the window (δ) is now a real config key (rerank_window);
# the margin veto and the criterion hooks (_margin_clears / _bottom_cut replacements) are gone (review item 6, 2026-10-22).
MODES = {
    "off": None,
    "reorder_only": ("0.0", "0.0"),
    "reorder_blend": ("0.5", "0.0"),
    "gate_only": ("0.0", "0.05"),
    "gate_blend": ("0.5", "0.05"),
    # window sweep (formerly botwin:δ) — the same numbers in rerank_window:
    "win10": ("0.0", "0.10"),
    "win03": ("0.0", "0.03"),
    "win02": ("0.0", "0.02"),
    # the gate_min2/gate_named modes are gone (review item 1, 13:06): the minimum of two —
    # is built into the window, the named-person guard is gone; gate_only = a ready
    # configuration for the merge (δ=0.05 + this built-in minimum).
}


EMB_URL = os.environ.get("EMB_URL", "http://127.0.0.1:1237/v1")
EMB_MODEL = os.environ.get("EMB_MODEL", "bge-m3")
ENDPOINT = os.environ.get("RERANK_URL", "http://127.0.0.1:1235/v1/rerank")
SCALE = os.environ.get("RERANK_SCALE", "logit")  # endpoint scale (fix-list item 4, 14:01)


# Issue #95 (main sets them for one run and puts them back): the keywords this run stores with the facts, per language
# and fact key (only those the coverage keeps), and the folder this run's stores live in — a store of its own per arm.
KEYWORDS: dict | None = None
DB_DIR: Path | None = None


def _db(name: str) -> Path:
    """A fresh store file of this run: nothing is left of an earlier run's file of that name."""
    db = (DB_DIR or Path(tempfile.gettempdir())) / name
    for s in ("", "-shm", "-wal"):
        try:
            Path(str(db) + s).unlink()
        except OSError:
            pass
    return db


def store_facts(p: HolographicMemoryProvider, lang: str, facts: dict) -> dict[int, str]:
    """Store *facts* (key -> (text, entities)) through the tool, each with the keywords this run gives its key;
    returns fact_id -> key. A keyword the plugin did not keep, or keywords for a key the set does not have, stop the
    run: frozen keywords are stored as written or not measured at all."""
    if KEYWORDS is not None and lang not in KEYWORDS:  # a file of another set: both arms would run without keywords
        raise SystemExit(f"the keywords file has no {lang!r} — it is not this set's file")
    given = (KEYWORDS or {}).get(lang, {})
    unknown = sorted(set(given) - set(facts))
    if unknown:
        raise SystemExit(f"keywords for facts the set does not have ({lang}): {unknown[:5]}")
    back = {}
    for key, (text, entities) in facts.items():
        args = {"action": "add", "content": text, "entities": entities}
        if given.get(key):
            args["keywords"] = given[key]
        r = json.loads(p.handle_tool_call("fact_store", args))
        if "error" in r or r.get("keywords_dropped") or ("keywords" in args and r.get("keywords") != args["keywords"]):
            raise SystemExit(f"{lang}/{key}: not stored as written: {r}")
        back[int(r["fact_id"])] = key
    return back


def _kw_facts(p: HolographicMemoryProvider) -> int:
    """How many facts of this store carry keywords: both arms of a pair must have the same."""
    return len(p._store.read("SELECT 1 FROM facts WHERE tags LIKE 'keywords: %' OR tags LIKE ?", ("%\nkeywords: %",)))


def _git_sha() -> str:
    """Stem of the commit: every JSONL row knows at which code it was captured (review item 8, 2026-10-22). A tree
    with uncommitted changes is "<sha>-dirty": the rows were not captured at that commit (issue #95 review)."""
    try:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                             capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=_REPO,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{sha}-dirty" if dirty and sha != "unknown" else sha
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- set data

def load_multilingual():
    spec = importlib.util.spec_from_file_location(
        "holo_lang_data", _REPO / "tests" / "test_holographic_languages.py")
    assert spec and spec.loader
    data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data)
    return data


def load_contrast(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "tools" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- provider

def make_provider(db: Path, mode: str, emb, extra: dict | None = None) -> HolographicMemoryProvider:
    spec = MODES[mode]
    config = {"db_path": str(db), "hrr_dim": 1024}
    if spec:
        config.update({"rerank": "on", "rerank_weight": spec[0], "rerank_window": spec[1],
                       "rerank_scale": SCALE, "rerank_url": ENDPOINT, "rerank_model": "probe-reranker"})
    if emb == "on":
        config.update({"embeddings": "on", "embedding_url": EMB_URL, "embedding_model": EMB_MODEL})
    if extra:
        config.update(extra)
    p = HolographicMemoryProvider(config=config)
    p.initialize(session_id="prov")
    return p


class Probe:
    """Collects the provenance of a single prefetch. The context resets on every query."""

    def __init__(self, p: HolographicMemoryProvider):
        self.p = p
        self.ctx: dict = {}
        self._orig = {}
        r = p._retriever
        self._wrap(r, "search", self._search)
        self._wrap(r, "relevant", self._relevant)
        self._wrap(r, "by_meaning", self._by_meaning)
        self._wrap(r, "entity_facts", self._entity_facts)
        self._wrap(p, "_sendable_docs", self._sendable_docs)
        self._wrap(p, "_not_recent", self._not_recent)
        self._wrap(p, "_recallable", self._recallable)
        # the runner on main has no _rescorer at all — getattr, no direct access (fix-list item 5)
        rsc = getattr(p, "_rescorer", None)
        if rsc is not None:
            self._wrap(rsc, "scores", self._scores)

    def _wrap(self, obj, name: str, wrapper):
        orig = getattr(obj, name, None)
        if orig is None:
            return
        self._orig[(id(obj), name)] = orig
        setattr(obj, name, wrapper(orig))

    # -- pipeline calls ---------------------------------------------------
    def _search(self, orig):
        def call(*a, **kw):
            out = orig(*a, **kw)
            self.ctx["pool"] = [f["fact_id"] for f in out]
            self.ctx["order"] = list(self.ctx["pool"])
            return out
        return call

    def _sendable_docs(self, orig):
        def call(facts):
            out = orig(facts)
            # ranked indexes DOCUMENTS (the send list), not the pool: in meaning/entity flows
            # union and pool differ, so we remember union and the send positions to translate
            # the response index into fact_id (pool[i] used to be written — foreign ids in JSONL).
            self.ctx["union"] = [f.get("fact_id") for f in facts]
            self.ctx["send_idx"] = list(out[0])
            return out
        return call

    def _scores(self, orig):
        def call(query, docs):
            ranked = orig(query, docs)
            owner = getattr(orig, "__self__", None)  # test stubs replace scores with a plain function without __self__
            status = getattr(owner, "last_status", None) or "not_called"
            if ranked is None and status.startswith("failed"):
                # Fix-list 19:07: a failed call must not look like "the reranker changed nothing" —
                # the run stops at once so the final rows are guaranteed to have 0 failed calls.
                raise SystemExit(
                    f"rerank call failed ({status}) — the run was stopped "
                    "(fix-list 19:07: zero failed-call budget)")
            self.ctx["rerank_call"] = status
            if ranked is None:
                self.ctx["ranked"] = False
                return ranked
            union = self.ctx.get("union") or []
            send_idx = self.ctx.get("send_idx") or []
            order = [union[send_idx[i]] for i, _ in ranked
                     if 0 <= i < len(send_idx) and send_idx[i] < len(union)]
            self.ctx["rank_order"] = order  # the rerank order BEFORE the window and top-5
            self.ctx["rerank_top5"] = order[:5]
            self.ctx["ranked"] = True
            return ranked
        return call

    def _relevant(self, orig):
        def call(query, candidates, min_coverage=0.3):
            cids = [f["fact_id"] for f in candidates]
            self.ctx["gate_called"] = True
            self.ctx["gate_order"] = list(cids)
            kept = orig(query, candidates, min_coverage)
            k = {f["fact_id"] for f in kept}
            for pos, fid in enumerate(cids):
                if fid in k:
                    self.ctx.setdefault("tags", {}).setdefault(fid, set()).add("words")
                else:
                    self.ctx.setdefault("blocked", {})[fid] = pos
            return kept
        return call

    def _by_meaning(self, orig):
        def call(*a, **kw):
            out = orig(*a, **kw)
            for f in out:
                self.ctx.setdefault("tags", {}).setdefault(f["fact_id"], set()).add("meaning")
            return out
        return call

    def _entity_facts(self, orig):
        def call(*a, **kw):
            out = orig(*a, **kw)
            for f in out:
                self.ctx.setdefault("tags", {}).setdefault(f["fact_id"], set()).add("entity")
            return out
        return call

    def _not_recent(self, orig):
        def call(key, fact, turn):
            ok = orig(key, fact, turn)
            if not ok:
                self.ctx.setdefault("tags", {}).setdefault(fact["fact_id"], set()).add("recent")
            return ok
        return call

    def _recallable(self, orig):
        def call(fact):
            ok = orig(fact)
            if not ok:
                self.ctx.setdefault("tags", {}).setdefault(fact["fact_id"], set()).add("unsafe")
            return ok
        return call

    # -- query wrapper -----------------------------------------------------
    def prefetch(self, query: str) -> tuple[str, dict]:
        self.ctx = {"tags": {}, "blocked": {}, "gate_called": False,
                    "pool": [], "rerank_top5": [], "rank_order": [], "ranked": None,
                    "rerank_call": "not_called",
                    "window": 0.0, "weight": 0.0}
        rescorer = getattr(self.p, "_rescorer", None)
        if rescorer is None:
            self.ctx["rerank_call"] = "off"  # rerank is off in the config — no call can happen by construction
        else:
            rescorer.last_status = "not_called"  # the outcome is known only after the call; reset on every query
        if rescorer is not None and getattr(rescorer, "settings", None) is not None:
            # the window cuts only at weight == 0 (see _unified_stream) — recorded for classify
            self.ctx["window"] = float(getattr(rescorer.settings, "window", 0.0) or 0.0)
            self.ctx["weight"] = float(getattr(rescorer.settings, "weight", 0.0) or 0.0)
        block = self.p.prefetch(query)
        ctx = dict(self.ctx)
        ctx["tags"] = {k: sorted(v) for k, v in ctx["tags"].items()}
        ctx["blocked"] = dict(ctx["blocked"])
        return block, ctx


def catch_up_vectors(p: HolographicMemoryProvider) -> dict:
    """Fact vectors are built only in _catch_up_vectors (the core's after-turn hook), which a driver run never calls,
    and one call has a time budget that leaves the tail of a wide store without vectors: it runs while facts wait.

    Returns what the rows of this store are stamped with: the embedding model's label and its one vector size, the
    usable vectors (made from the fact's present vector text, by that model) and the expected ones (every fact whose
    vector text may be sent). A store left short stops the run — it used to report "ready" after 60 tries whatever
    was missing, and a run with part of the vectors missing compares two different stores (issue #95)."""
    if not p._embedding:
        return {}
    tag = p._embedding.tag
    idx = p._retriever.vector_index(tag)
    for _ in range(60):
        if not idx.pending(_holographic_embeddings.BATCH, allowed=_holographic_sendable):
            break
        p._catch_up_vectors()
    counts = idx.counts(_holographic_sendable)
    expected = counts["facts_with_vector"] + counts["waiting"]
    sizes = sorted(r[0] for r in p._store.read(
        "SELECT DISTINCT LENGTH(e.vector) FROM fact_embeddings e JOIN facts f ON f.fact_id = e.fact_id "
        "WHERE e.model = ?", (tag,)))
    if counts["waiting"] or len(sizes) != 1:
        raise SystemExit(f"incomplete vector coverage: {counts['facts_with_vector']} of {expected} usable, vector "
                         f"sizes {sizes} — the run is discarded")
    return {"emb_model": tag, "emb_dim": sizes[0] // 4, "vectors": counts["facts_with_vector"],
            "vectors_expected": expected, "never_sent": counts["never_sent"]}


def parse_ids(block: str) -> list[int]:
    ids = []
    if not block:
        return ids
    for line in block.splitlines()[1:]:
        try:
            ids.append(int(line.split()[1][1:]))
        except (IndexError, ValueError):
            pass
    return ids


def classify(fid: int | None, ctx: dict) -> str:
    """Why the target is not in recall (for real queries).

    E («срезан окном»): the target passed every filter and stood in the top-5 of the rank BEFORE the
    rerank_window cut — the window removed it, not the words and not top-5. Such a case used to
    lie «ниже top-5 (пул#N)»: the pool is the candidate order, not the rerank rank
    (case «What does Andrey do on weekends?», review 2026-10-23).
    """
    if fid is None:
        return "-"
    if fid in ctx["tags"] and any(t in ctx["tags"][fid] for t in ("recent", "unsafe")):
        return "dropped:" + ",".join(ctx["tags"][fid])
    rank = ctx.get("rank_order") or []
    if fid in rank:
        pos = rank.index(fid)
        cut = (ctx.get("window") or 0.0) > 0.0 and (ctx.get("weight") or 0.0) == 0.0
        if pos < 5:
            if cut:
                return f"E: срезан окном (ранг #{pos})"
            # the top-5 of the rank without the cut should have made it into recall — we never get here
        else:
            return f"C: ниже top-5 (ранг #{pos})"
    pool = ctx.get("pool") or []
    if fid not in pool:
        return "A: нет в словесном пуле"
    blocked = ctx.get("blocked") or {}
    if fid in blocked:
        pos = blocked[fid]
        r5 = fid in (ctx.get("rerank_top5") or [])
        return "D: вырезан coverage" + (" (top-5 реранка!)" if r5 else f" (пул#{pos})")
    if not ctx.get("gate_called"):
        # gate_reads_score admits the pool without coverage: the target is either below top-5 or rejected by margin
        order = ctx.get("gate_order") or pool
        if fid in order[:5]:
            return "gate: отвергнут margin (в top-5 порядка!)"
        return "C: ниже top-5"
    # coverage let it through but not into recall: recency / the top-5 limit
    order = ctx.get("gate_order") or pool
    pos = order.index(fid) if fid in order else 999
    return f"C: ниже top-5 (пул#{pos})"


# ---------------------------------------------------------------- runs

def run_multilingual(langs, modes, emb, endpoint, extra: dict | None = None) -> list[dict]:
    data = load_multilingual()
    rows = []
    for mode in modes:
        for lang in langs:
            p = make_provider(_db(f"prov_ml_{lang}_{mode}.db"), mode, emb, extra)
            back = store_facts(p, lang, data.CORPUS[lang])
            probe = Probe(p)
            coverage = {**(catch_up_vectors(p) if emb == "on" else {}), "kw_facts": _kw_facts(p)}
            if "vectors" in coverage:
                print(f"  [{lang}] vectors: {coverage['vectors']} of {coverage['vectors_expected']}")
            first = len(rows)
            persons = dict(data.PROBES[lang])
            for i, (query, relevant) in enumerate(data.QUERIES[lang]):
                p.on_session_switch(f"pv{li(mode, lang)}_{i}", reset=True)
                t0 = time.perf_counter()
                block, ctx = probe.prefetch(query)
                ms = (time.perf_counter() - t0) * 1000
                found = parse_ids(block)
                keys = [back[f] for f in found if f in back]
                kind = "real" if relevant else "chit"
                hit = bool(set(keys) & set(relevant)) if relevant else None
                # matched top-5 (item 8): off can return more than five lines, unified cannot; an honest comparison
                # on the first five, plus the "total lines" count as the output count.
                hit5 = bool(set(keys[:5]) & set(relevant)) if relevant else None
                wrong_keys, wrong_keys5 = [], []
                if relevant:
                    for name in persons:
                        if name in query:
                            other = set().union(*(v for n, v in persons.items() if n != name))
                            wrong_keys = sorted(set(keys) & set(other))
                            wrong_keys5 = sorted(set(keys[:5]) & set(other))
                miss_cls = "-"
                if relevant and not hit5:
                    # target(s) without a hit: the class of the first missing one
                    miss_cls = next((classify(next((f for f, k in back.items() if k == rk), None), ctx)
                                     for rk in relevant if rk not in keys), "-")
                rows.append({"set": "multi", "lang": lang, "mode": mode, "emb": emb,
                             "qid": str(i), "split": "all",
                             "kind": kind, "q": query, "keys": keys,
                             "targets": list(relevant or []),
                             "sources": {str(f): ctx["tags"].get(f, ["pool-only"])
                                         for f in found},
                             "hit": hit, "wrong": wrong_keys,
                             "hit5": hit5, "wrong5": wrong_keys5, "out_n": len(keys),
                             "miss_cls": miss_cls, "ms": round(ms, 1),
                             "pool_n": len(ctx.get("pool") or []),
                             "gate_called": ctx.get("gate_called"),
                         "rerank_call": ctx.get("rerank_call"),
                             "rerank_top5": ctx.get("rerank_top5") or [],
                             "rank_order": ctx.get("rank_order") or []})
            for r in rows[first:]:
                r.update(coverage)
            p.shutdown()
    return rows


def li(mode, lang):  # session prefix — unique per mode+language
    return f"{mode}_{lang}"


def run_contrast(set_name, modes, emb, endpoint, extra: dict | None = None) -> list[dict]:
    contrast = load_contrast(set_name)
    rows = []
    for mode in modes:
        p = make_provider(_db(f"prov_{set_name}_{mode}.db"), mode, emb, extra)
        back = store_facts(p, set_name, contrast.FACTS)
        probe = Probe(p)
        coverage = {**(catch_up_vectors(p) if emb == "on" else {}), "kw_facts": _kw_facts(p)}
        if "vectors" in coverage:
            print(f"  [{set_name}] vectors: {coverage['vectors']} of {coverage['vectors_expected']}")
        first = len(rows)
        for pi, pair in enumerate(contrast.PAIRS):
            partner = [pair["queries"][1 - i][1] for i in range(len(pair["queries"]))]
            for i, (query, target) in enumerate(pair["queries"]):
                p.on_session_switch(f"cp{pi}_{i}", reset=True)
                block, ctx = probe.prefetch(query)
                found = parse_ids(block)
                keys = [back[f] for f in found if f in back]
                own_pos = keys.index(target) + 1 if target in keys else None
                part = partner[i]
                part_pos = None if part == target else (keys.index(part) + 1 if part in keys else None)
                miss_cls = classify(next((f for f, k in back.items() if k == target), None), ctx) \
                    if own_pos is None else "-"
                rows.append({"set": set_name, "lang": set_name, "mode": mode, "emb": emb,
                             "qid": f"p{pi}q{i}", "split": "all",
                             "kind": "real", "q": query, "keys": keys, "target": target,
                             "targets": [target],
                             "own_pos": own_pos, "partner_pos": part_pos,
                             "sources": {str(f): ctx["tags"].get(f, ["pool-only"]) for f in found},
                             "hit": own_pos is not None, "wrong": part_pos is not None and part_pos <= 5,
                             "hit5": own_pos is not None and own_pos <= 5, "out_n": len(keys),
                             "miss_cls": miss_cls, "pool_n": len(ctx.get("pool") or []),
                             "gate_called": ctx.get("gate_called"),
                         "rerank_call": ctx.get("rerank_call"),
                             "rerank_top5": ctx.get("rerank_top5") or [],
                             "rank_order": ctx.get("rank_order") or []})
        for i, (query, cat) in enumerate(contrast.CHITCHAT):
            p.on_session_switch(f"cq{i}", reset=True)
            block, ctx = probe.prefetch(query)
            found = parse_ids(block)
            keys = [back[f] for f in found if f in back]
            rows.append({"set": set_name, "lang": set_name, "mode": mode, "emb": emb,
                         "qid": f"c{i}", "split": "all",
                         "kind": "chit", "cat": cat, "q": query, "keys": keys,
                         "targets": [],
                         "sources": {str(f): ctx["tags"].get(f, ["pool-only"]) for f in found},
                         "hit": None, "wrong": [], "hit5": None, "out_n": len(keys), "miss_cls": "-",
                         "pool_n": len(ctx.get("pool") or []),
                         "gate_called": ctx.get("gate_called"),
                         "rerank_call": ctx.get("rerank_call"),
                         "rerank_top5": ctx.get("rerank_top5") or []})
        for r in rows[first:]:
            r.update(coverage)
        p.shutdown()
    return rows


# ---------------------------------------------------------------- report


def run_household(modes, emb, endpoint, split="all", names_fit="on",
                  extra: dict | None = None) -> list[dict]:
    """Item 9 (the maintainer's recipe, PR #10 2026-10-22): one home, en/ru/zh,
    dev/test frozen, a label on every question.

    Rows carry qid/split/slice/lang — the report counts rescued/lost and the miss
    classes (E cut by the window / A not in the pool / C below top-5) per language.
    names_fit off = the name check is off (as in tests ..._rerank_review).
    """
    hh = load_contrast("household")
    rows = []
    qids = [k for k, q in hh.QUERIES.items() if split == "all" or q["split"] == split]
    hids = [k for k, h in hh.CHITCHAT.items() if split == "all" or h["split"] == split]
    for mode in modes:
        for lang in hh.LANGS:
            p = make_provider(_db(f"prov_hh_{lang}_{mode}_{names_fit}.db"), mode, emb, extra)
            if names_fit == "off":
                p._retriever.names_fit = lambda *a, **k: True
            back = store_facts(p, lang, {key: v[lang] for key, v in hh.FACTS.items()})
            ent_map = hh.entities(lang)  # key -> entity names of this language
            probe = Probe(p)
            coverage = {**(catch_up_vectors(p) if emb == "on" else {}), "kw_facts": _kw_facts(p)}
            if "vectors" in coverage:
                print(f"  [household {lang}] vectors: {coverage['vectors']} of {coverage['vectors_expected']}")
            first = len(rows)
            for qid in qids:
                q = hh.QUERIES[qid]
                query, targets = q[lang], set(q["targets"])
                p.on_session_switch(f"hh{lang}_{mode}_{qid}", reset=True)
                t0 = time.perf_counter()
                block, ctx = probe.prefetch(query)
                ms = (time.perf_counter() - t0) * 1000
                found = parse_ids(block)
                keys = [back[f] for f in found if f in back]
                # the old wrong-lookup searched the fact key in the name dictionary (always empty —
                # a bug from review 10:05). Now there are two columns:
                #   named   — non-targets that mention the person from the query (correct
                #             extras and namesakes; descriptive, weak on ru cases);
                #   confused — labels of wrong answers in the top-5 (a measure, static).
                named = hh.named_extra(lang, query, keys, targets, ent_map)
                conf = hh.confusion(q, keys[:5])
                hit = bool(set(keys) & targets) if targets else None
                hit5 = bool(set(keys[:5]) & targets) if targets else None
                cov = (len(set(keys) & targets) / len(targets)) if targets else None
                miss = []
                for t in q["targets"]:  # every label that missed the top-5
                    if t in keys[:5]:
                        continue
                    if t in keys:
                        miss.append(f"C: ниже top-5 (позиция {keys.index(t) + 1})")
                    else:
                        miss.append(classify(next((f for f, k in back.items() if k == t), None), ctx))
                rows.append({"set": "household", "lang": lang, "mode": mode, "emb": emb,
                             "split": q["split"], "slice": q["slice"], "qid": qid,
                             "kind": "real", "q": query, "keys": keys,
                             "targets": sorted(targets), "hit": hit, "hit5": hit5,
                             "cov": cov, "named": named, "confused": conf, "out_n": len(keys),
                             "miss": miss, "ms": round(ms, 1),
                             "sources": {str(f): ctx["tags"].get(f, ["pool-only"]) for f in found},
                             "pool_n": len(ctx.get("pool") or []),
                             "gate_called": ctx.get("gate_called"),
                         "rerank_call": ctx.get("rerank_call"),
                             "rerank_top5": ctx.get("rerank_top5") or [],
                             "rank_order": ctx.get("rank_order") or []})
            for hid in hids:
                h = hh.CHITCHAT[hid]
                query = h[lang]
                p.on_session_switch(f"hh{lang}_{mode}_{hid}", reset=True)
                block, ctx = probe.prefetch(query)
                found = parse_ids(block)
                keys = [back[f] for f in found if f in back]
                rows.append({"set": "household", "lang": lang, "mode": mode, "emb": emb,
                             "split": h["split"], "slice": "chit", "qid": hid, "kind": "chit",
                             "cat": h["cat"], "q": query, "keys": keys, "targets": [],
                             "hit": None, "hit5": None, "cov": None, "named": [], "confused": [],
                             "out_n": len(keys), "miss": [], "ms": None,
                             "sources": {str(f): ctx["tags"].get(f, ["pool-only"]) for f in found},
                             "pool_n": len(ctx.get("pool") or []),
                             "gate_called": ctx.get("gate_called"),
                         "rerank_call": ctx.get("rerank_call"),
                             "rerank_top5": ctx.get("rerank_top5") or [],
                             "rank_order": ctx.get("rank_order") or []})
            for r in rows[first:]:
                r.update(coverage)
            p.shutdown()
    return rows


def run_synthetic_zh(modes, emb, endpoint, extra: dict | None = None) -> list[dict]:
    """Issue #95 synthetic Chinese-heavy set (tools/synthetic_zh.py): one store, 30 facts,
    36 labelled questions (direct/paraphrase/crosslingual/colloquial) and 12 noise rows
    (6 negative chit-chat + 6 in-domain no-answer). Each query carries its text under its
    own language key (zh/en); crosslingual questions are English text targeting Chinese
    facts. Rows mirror run_household so the gate pairs them by (set, lang, qid)."""
    sz = load_contrast("synthetic_zh")
    sz.self_check()
    rows = []
    lang = sz.LANG  # the keywords file key; both arms store under it
    for mode in modes:
        p = make_provider(_db(f"prov_synthetic_zh_{mode}.db"), mode, emb, extra)
        back = store_facts(p, lang, sz.FACTS)
        probe = Probe(p)
        coverage = {**(catch_up_vectors(p) if emb == "on" else {}), "kw_facts": _kw_facts(p)}
        if "vectors" in coverage:
            print(f"  [synthetic_zh] vectors: {coverage['vectors']} of {coverage['vectors_expected']}")
        first = len(rows)
        for qid, q in sz.QUERIES.items():
            qlang = "en" if "en" in q else "zh"
            query, targets = q[qlang], set(q["targets"])
            p.on_session_switch(f"sz_{mode}_{qid}", reset=True)
            t0 = time.perf_counter()
            block, ctx = probe.prefetch(query)
            ms = (time.perf_counter() - t0) * 1000
            found = parse_ids(block)
            keys = [back[f] for f in found if f in back]
            hit = bool(set(keys) & targets) if targets else None
            hit5 = bool(set(keys[:5]) & targets) if targets else None
            cov = (len(set(keys) & targets) / len(targets)) if targets else None
            conf = sorted(set(keys[:5]) - targets)
            miss, miss_cls = [], "-"
            for t in q["targets"]:
                if t in keys[:5]:
                    continue
                if t in keys:
                    miss.append(f"C: below top-5 (pos {keys.index(t) + 1})")
                else:
                    cls = classify(next((f for f, k in back.items() if k == t), None), ctx)
                    miss.append(cls)
                    if miss_cls == "-":
                        miss_cls = cls
            rows.append({"set": "synthetic_zh", "lang": qlang, "mode": mode, "emb": emb,
                         "split": "all", "slice": q["slice"], "qid": qid,
                         "kind": "real", "q": query, "keys": keys,
                         "targets": sorted(targets), "hit": hit, "hit5": hit5,
                         "cov": cov, "named": [], "confused": conf, "out_n": len(keys),
                         "wrong": conf, "wrong5": conf, "miss": miss, "miss_cls": miss_cls,
                         "ms": round(ms, 1),
                         "sources": {str(f): ctx["tags"].get(f, ["pool-only"]) for f in found},
                         "pool_n": len(ctx.get("pool") or []),
                         "gate_called": ctx.get("gate_called"),
                         "rerank_call": ctx.get("rerank_call"),
                         "rerank_top5": ctx.get("rerank_top5") or [],
                         "rank_order": ctx.get("rank_order") or []})
        for hid, h in sz.CHITCHAT.items():
            qlang = "en" if "en" in h else "zh"
            query = h[qlang]
            p.on_session_switch(f"sz_{mode}_{hid}", reset=True)
            block, ctx = probe.prefetch(query)
            found = parse_ids(block)
            keys = [back[f] for f in found if f in back]
            rows.append({"set": "synthetic_zh", "lang": qlang, "mode": mode, "emb": emb,
                         "split": "all", "slice": "chit", "qid": hid, "kind": "chit",
                         "cat": h["cat"], "q": query, "keys": keys, "targets": [],
                         "hit": None, "hit5": None, "cov": None, "named": [], "confused": [],
                         "out_n": len(keys), "wrong": [], "miss": [], "miss_cls": "-", "ms": None,
                         "sources": {str(f): ctx["tags"].get(f, ["pool-only"]) for f in found},
                         "pool_n": len(ctx.get("pool") or []),
                         "gate_called": ctx.get("gate_called"),
                         "rerank_call": ctx.get("rerank_call"),
                         "rerank_top5": ctx.get("rerank_top5") or [],
                         "rank_order": ctx.get("rank_order") or []})
        for r in rows[first:]:
            r.update(coverage)
        p.shutdown()
    return rows


def run_budget(budgets, split="all", repeats=3) -> list[dict]:
    """prefetch_max_tokens (0.21.0) over tools/budget_set.py: one store per budget, the same facts in the same order
    (so the same ids), the changed facts stored first with their old text and then changed. Per question: hit (every
    label in the block) and hit_any, extra (facts that are not labels), confused (the namesake's fact), stale (the old
    text of a changed label), cut (a line the 300-character line cap ended with "…"), approximate tokens of the
    injected block, facts the budget left out, latency (the best of *repeats* runs, each in a fresh session). No
    rerank, no embeddings: nothing leaves the machine."""
    from hermes_plugin_holographic import approx_tokens
    bs = load_contrast("budget_set")
    bs.self_check()
    qids = [k for k, q in bs.QUERIES.items() if split == "all" or q["split"] == split]
    rows = []
    for budget in budgets:
        db = Path(tempfile.mkdtemp(prefix="prov_budget_")) / f"budget_{budget}.db"
        p = make_provider(db, "off", "off", {"prefetch_max_tokens": budget})
        try:
            back: dict[int, str] = {}
            for key, (text, ents) in bs.FACTS.items():
                first = bs.CHANGED_FROM.get(key, text)
                r = json.loads(p.handle_tool_call(
                    "fact_store", {"action": "add", "content": first, "entities": ents, "origin": "said"}))
                fid = int(r["fact_id"])
                if key in bs.CHANGED_FROM:
                    changed = json.loads(p.handle_tool_call(
                        "fact_store", {"action": "change", "fact_id": fid, "content": text, "origin": "said"}))
                    assert "error" not in changed, (key, changed)
                back[fid] = key
            for qid in qids:
                q = bs.QUERIES[qid]
                targets = set(q["targets"])
                times = []
                for rep in range(repeats):
                    p.on_session_switch(f"budget_{budget}_{qid}_{rep}", reset=True)
                    dropped_before = _budget_dropped(p)
                    t0 = time.perf_counter()
                    block = p.prefetch(q["q"])
                    times.append((time.perf_counter() - t0) * 1000)
                    dropped = _budget_dropped(p) - dropped_before
                keys = [back[f] for f in parse_ids(block) if f in back]
                stale = [t for t in targets if t in bs.CHANGED_FROM and bs.CHANGED_FROM[t] in block]
                rows.append({"set": "budget", "budget": budget, "split": q["split"], "slice": q["slice"],
                             "qid": qid, "keys": keys, "targets": sorted(targets),
                             "hit": bool(targets) and targets <= set(keys),
                             "hit_any": bool(targets & set(keys)) if targets else None,
                             "extra": len([k for k in keys if k not in targets]),
                             "confused": sorted(set(keys) & set(q.get("confused", []))),
                             "stale": stale, "cut": sum(1 for line in block.splitlines() if line.endswith("…")),
                             "tokens": approx_tokens(block), "dropped": dropped,
                             "ms": round(min(times), 2), "ms_runs": [round(t, 2) for t in times]})
        finally:
            p.shutdown()
    return rows


def _budget_dropped(p) -> int:
    return json.loads(p.handle_tool_call("fact_store", {"action": "stats"})).get(
        "prefetch_budget", {}).get("dropped_total", 0)


def summarize_budget(rows: list[dict]) -> list[dict]:
    """Per budget × split: labelled questions with every label back (hit) / any label (any), questions where the
    budget cost a label the unlimited run had (lost), noise (a `none` question that recalled anything), extra facts
    per question, namesake and stale hits, cut lines, approximate tokens avg / p95 / max over non-empty blocks,
    facts left out, latency p50 / p95. Returns the table rows (also printed)."""
    def pct(values, q):
        values = sorted(values)
        return values[min(len(values) - 1, int(q * len(values)))] if values else 0

    base = {r["qid"]: set(r["keys"]) & set(r["targets"]) for r in rows if r["budget"] == 0}
    table = []
    print(f"\n{'budget':>6} {'split':<5} {'lab':>3} {'hit':>3} {'any':>3} {'lost':>4} {'noise':>5} {'extra':>5} "
          f"{'conf':>4} {'stale':>5} {'cut':>3} {'tok_avg':>7} {'tok_p95':>7} {'tok_max':>7} {'dropped':>7} "
          f"{'ms_p50':>6} {'ms_p95':>6}")
    for budget in sorted({r["budget"] for r in rows}):
        for split in ("dev", "test", "all"):
            sel = [r for r in rows if r["budget"] == budget and (split == "all" or r["split"] == split)]
            if not sel:
                continue
            lab = [r for r in sel if r["targets"]]
            none = [r for r in sel if not r["targets"]]
            toks = [r["tokens"] for r in sel if r["keys"]]
            lost = sum(1 for r in lab if r["qid"] in base and base[r["qid"]] - set(r["keys"]))
            ms = [r["ms"] for r in sel]
            row = {"budget": budget, "split": split, "labelled": len(lab), "hit": sum(r["hit"] for r in lab),
                   "hit_any": sum(bool(r["hit_any"]) for r in lab), "lost_vs_unlimited": lost,
                   "noise": sum(1 for r in none if r["keys"]), "none": len(none),
                   "extra_avg": round(sum(r["extra"] for r in sel) / len(sel), 2),
                   "confused": sum(1 for r in sel if r["confused"]), "stale": sum(1 for r in sel if r["stale"]),
                   "cut": sum(r["cut"] for r in sel),
                   "tokens_avg": round(sum(toks) / len(toks), 1) if toks else 0, "tokens_p95": pct(toks, 0.95),
                   "tokens_max": max(toks, default=0), "dropped": sum(r["dropped"] for r in sel),
                   "ms_p50": round(pct(ms, 0.5), 2), "ms_p95": round(pct(ms, 0.95), 2)}
            table.append(row)
            print(f"{budget:>6} {split:<5} {row['labelled']:>3} {row['hit']:>3} {row['hit_any']:>3} {lost:>4} "
                  f"{row['noise']:>2}/{row['none']:<2} {row['extra_avg']:>5} {row['confused']:>4} "
                  f"{row['stale']:>5} {row['cut']:>3} {row['tokens_avg']:>7} {row['tokens_p95']:>7} "
                  f"{row['tokens_max']:>7} {row['dropped']:>7} {row['ms_p50']:>6} {row['ms_p95']:>6}")
    return table


def _cls(misses: list[str], prefix: str) -> int:
    return sum(1 for m in misses if m.startswith(prefix))


def summarize_household(rows: list[dict]) -> None:
    """Table by language × split: hit@5, coverage, named/confused, noise (only
    where the label is empty) and the three miss classes — E cut by the window / A not in the pool /
    C below top-5 (the maintainer's request 2026-10-23, per language). At the end —
    the paired confused count off→mode: the confusion row, when the label of a wrong
    answer lands in the top-5 (his 10:34: count off against on in pairs)."""
    modes = sorted({(r["mode"], r["emb"]) for r in rows})
    for mode, emb in modes:
        sel = [r for r in rows if r["mode"] == mode and r["emb"] == emb]
        print(f"\n=== household mode={mode} emb={emb} ===")
        print(f"{'lang':<4} {'split':<5} {'real':>4} {'hit5':>4} {'cov%':>5} {'named':>5} "
              f"{'conf':>4} {'noise':>4} {'E':>2} {'A':>2} {'C':>2} {'othr':>4}")
        cells = [(lang, sp) for lang in ("en", "ru", "zh") for sp in ("dev", "test")]
        for lang, sp in cells:
            real = [r for r in sel if r["kind"] == "real" and r["lang"] == lang
                    and r["split"] == sp]
            if not real:
                continue
            labelled = [r for r in real if r["targets"]]
            noise_rows = [r for r in sel if r["lang"] == lang and r["split"] == sp
                          and ((r["kind"] == "chit") or (r["kind"] == "real" and not r["targets"]))]
            noise = sum(bool(r["keys"]) for r in noise_rows)
            hits5 = sum(bool(r["hit5"]) for r in labelled)
            cov = sum(r["cov"] or 0 for r in labelled)
            named = sum(1 for r in real if r["named"])
            conf = sum(1 for r in real if r["confused"])
            misses = [m for r in labelled for m in r["miss"]]
            e, a, c = _cls(misses, "E:"), _cls(misses, "A:"), _cls(misses, "C:")
            other = len(misses) - e - a - c
            print(f"{lang:<4} {sp:<5} {len(labelled):>4} {hits5:>4} "
                  f"{100 * cov / max(1, len(labelled)):>5.1f} {named:>5} {conf:>4} "
                  f"{noise:>2}/{len(noise_rows):<2} {e:>2} {a:>2} {c:>2} {other:>4}")
        lab = [r for r in sel if r["kind"] == "real" and r["targets"]]
        hits5 = sum(bool(r["hit5"]) for r in lab)
        named = sum(1 for r in sel if r["kind"] == "real" and r["named"])
        conf = sum(1 for r in sel if r["kind"] == "real" and r["confused"])
        print(f"{'total':<9} {len(lab):>4} {hits5:>4} "
              f"hit5={hits5}/{len(lab)} named={named} confused={conf}")

    # paired count (maintainer 10:34): a confusion row if at least one label
    # of a wrong answer lands in the top-5; off against each mode — removed/added
    for emb in sorted({str(r["emb"]) for r in rows}):
        sel = [r for r in rows if str(r["emb"]) == emb and r["kind"] == "real"]
        base = {(r["lang"], r["qid"]): bool(r["confused"])
                for r in sel if r["mode"] == "off"}
        for m2 in sorted({r["mode"] for r in sel} - {"off"}):
            cur = {(r["lang"], r["qid"]): bool(r["confused"])
                   for r in sel if r["mode"] == m2}
            both = base.keys() & cur.keys()
            cut = sum(1 for k in both if base[k] and not cur[k])
            add = sum(1 for k in both if not base[k] and cur[k])
            print(f"confused paired emb={emb} off→{m2}: "
                  f"removed {cut}, added {add} ({len(both)} rows)")

    # per-target lost/gained (review item 5, 13:06): hit@5 "any target" hides partial
    # losses of multi-target queries. We count top-5 on BOTH sides, paired against off —
    # targets used to come from all of keys while off could have more than five lines, so
    # the count overstated the losses; gained is printed too, by id (qid:target) with the language of each row.
    for emb in sorted({str(r["emb"]) for r in rows}):
        base_top5: dict[tuple, set] = {}
        for r in rows:
            if str(r["emb"]) == emb and r["mode"] == "off" \
                    and r["kind"] == "real" and r["targets"]:
                base_top5[(r["lang"], r["qid"])] = set(r["targets"]) & set(r["keys"][:5])
        for mode in sorted({r["mode"] for r in rows} - {"off"}):
            lost: dict[str, list[str]] = {}
            gained: dict[str, list[str]] = {}
            for r in rows:
                if str(r["emb"]) != emb or r["mode"] != mode \
                        or r["kind"] != "real" or not r["targets"]:
                    continue
                prev = base_top5.get((r["lang"], r["qid"]))
                if prev is None:
                    continue
                cur = set(r["targets"]) & set(r["keys"][:5])
                for t in sorted(prev - cur):
                    lost.setdefault(f"{r['qid']}:{t}", []).append(r["lang"])
                for t in sorted(cur - prev):
                    gained.setdefault(f"{r['qid']}:{t}", []).append(r["lang"])
            if lost or gained:
                def _fmt(d: dict) -> str:
                    return ", ".join(f"{k}[{','.join(v)}]" for k, v in sorted(d.items()))
                print(f"per-target top5 emb={emb} {mode} vs off: "
                      f"lost {sum(map(len, lost.values()))} ({_fmt(lost) or '—'}); "
                      f"gained {sum(map(len, gained.values()))} ({_fmt(gained) or '—'})")
            else:
                print(f"per-target top5 emb={emb} {mode} vs off: lost 0; gained 0")


def summarize(rows: list[dict]) -> None:
    modes = sorted({(r["mode"], r["emb"]) for r in rows})
    print(f"\n{'mode':<14} {'emb':<4} {'real':>5} {'hit':>5} {'hit5':>5} {'noise':>5} "
          f"{'wrong':>5} {'wrong5':>6} {'out':>4} | recall sources (dominant) | miss classes")
    for mode, emb in modes:
        sel = [r for r in rows if r["mode"] == mode and r["emb"] == emb]
        real = [r for r in sel if r["kind"] == "real"]
        chit = [r for r in sel if r["kind"] == "chit"]
        hits = sum(bool(r["hit"]) for r in real)
        hits5 = sum(bool(r.get("hit5")) for r in real)
        noise = sum(bool(r["keys"]) for r in chit)
        wrong = sum(1 for r in real if r["wrong"])
        wrong5 = sum(1 for r in real if (r.get("wrong5") if "wrong5" in r else r.get("wrong")))
        out_avg = sum(len(r["keys"]) for r in sel) / max(1, len(sel))
        # sources of all recalled facts
        src: dict[str, int] = {}
        for r in sel:
            for tags in r["sources"].values():
                key = "+".join(t if t != "pool-only" else "words?" for t in tags) \
                    if "pool-only" not in tags else "words?(no coverage — gate)"
                src[key] = src.get(key, 0) + 1
        src_s = ", ".join(f"{k}:{v}" for k, v in sorted(src.items(), key=lambda x: -x[1]))
        classes: dict[str, int] = {}
        for r in real:
            if r["kind"] == "real" and not r["hit"]:
                classes[r["miss_cls"]] = classes.get(r["miss_cls"], 0) + 1
        cls_s = "; ".join(f"{k}:{v}" for k, v in sorted(classes.items(), key=lambda x: -x[1])) or "—"
        print(f"{mode:<14} {emb:<4} {len(real):>5} {hits:>5} {hits5:>5} {noise:>5}/{len(chit):<3} "
              f"{wrong:>5} {wrong5:>6} {out_avg:>4.1f} | {src_s} | {cls_s}")


def _load_keywords(path: str, coverage: str, seed: int) -> dict:
    """The frozen keywords file ({lang: {fact key: [keywords]}}), cut to the run's coverage: "all", or "half" — per
    language half of the facts, chosen by *seed* over the sorted keys, so both arms of a comparison and any later
    rerun keep the same half."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if coverage == "all":
        return data
    out = {}
    for lang, by_key in data.items():
        keys = sorted(by_key)
        kept = set(random.Random(f"{seed}:{lang}").sample(keys, len(keys) // 2))
        out[lang] = {k: v for k, v in by_key.items() if k in kept}
    return out


def main(argv):
    global ENDPOINT, SCALE, KEYWORDS, DB_DIR
    # Every override below is process-global: a later main([]) in the same process must not run at this
    # run's margin, endpoint or scale under the plain file name and overwrite a baseline (review of PR #31:
    # item 3 for RECALL_MARGIN; ENDPOINT is also --endpoint's default below, so it leaked the same way).
    saved_endpoint, saved_scale = ENDPOINT, SCALE
    saved_margin = _holographic_embeddings.RECALL_MARGIN
    saved_embed_text, saved_keywords, saved_db_dir = _holographic_embeddings.EMBED_TEXT, KEYWORDS, DB_DIR
    ap = argparse.ArgumentParser(description="Provenance probe: the sources of facts in prefetch")
    ap.add_argument("--set", default="multi",
                    help="multi | contrast_ru | contrast_en | contrast_zh | household | budget")
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"],
                    help="for --set household / budget: which split to run (default: all)")
    ap.add_argument("--budgets", default="0,150,100,60",
                    help="for --set budget: prefetch_max_tokens values, 0 = unlimited (the baseline; default "
                         "0,150,100,60)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="for --set budget: runs per question; latency is the best of them (default 3)")
    ap.add_argument("--names-fit", default="on", choices=["on", "off"],
                    help="for --set household: names_fit on/off (the sweep axis, item 9)")
    ap.add_argument("--langs", default="en,ru,zh,de,ja,it,nl,th",
                    help="languages for --set multi (default: the benchmark's key ones)")
    ap.add_argument("--modes", default="off,reorder_only,gate_only")
    ap.add_argument("--emb", default="off", choices=["off", "on", "both"])
    ap.add_argument("--endpoint", default=ENDPOINT, help="/v1/rerank URL (a mock if not set)")
    ap.add_argument("--rerank-scale", default="logit", choices=["logit", "unit"],
                    help="rerank_scale: the endpoint scale; the value is written to every row (item 4, 14:01)")
    ap.add_argument("--out", default=None, help="JSONL directory (default: results/)")
    ap.add_argument("--label", default=None,
                    help="a label in the file name (e.g. jina): does not overwrite the rows of another reranker")
    ap.add_argument("--recall-margin", default=None, type=float,
                    help="override embeddings.RECALL_MARGIN for this run and stamp the value into every row "
                         "and the file name (the 0.6 gate, PR #17); default: the module's own value")
    ap.add_argument("--embed-text", default=None, choices=list(_holographic_embeddings.EMBED_TEXTS),
                    help="what fact vectors are made from for this run (issue #95), stamped into every row and the "
                         "file name; default: the module's own value (content)")
    ap.add_argument("--keywords", default=None,
                    help="a frozen keywords file ({lang: {fact key: [keywords]}}): stored with the facts, the same in "
                         "both arms; its sha256 and the coverage are stamped into every row")
    ap.add_argument("--kw-coverage", default="all", choices=["all", "half"],
                    help="with --keywords: every fact, or half of each language's chosen by --kw-seed")
    ap.add_argument("--kw-seed", type=int, default=95, help="with --kw-coverage half: the seed (default 95)")
    args = ap.parse_args(argv)
    try:
        if args.recall_margin is not None:
            _holographic_embeddings.RECALL_MARGIN = args.recall_margin
        if args.embed_text is not None:
            _holographic_embeddings.EMBED_TEXT = args.embed_text
        kw_file, kw_coverage = None, "none"
        if args.keywords:
            if args.set == "budget":
                raise SystemExit("--keywords: the budget set has no keywords axis")
            kw_file = hashlib.sha256(Path(args.keywords).read_bytes()).hexdigest()[:12]
            kw_coverage = "all" if args.kw_coverage == "all" else f"half:seed={args.kw_seed}"
            KEYWORDS = _load_keywords(args.keywords, args.kw_coverage, args.kw_seed)
        DB_DIR = Path(tempfile.mkdtemp(prefix=f"prov_{args.set}_"))  # this run's stores: an arm of its own

        ENDPOINT = args.endpoint
        SCALE = args.rerank_scale

        out_dir = Path(args.out) if args.out else (_REPO / "results")
        out_dir.mkdir(exist_ok=True)
        embs = ["off", "on"] if args.emb == "both" else [args.emb]
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        all_rows: list[dict] = []

        # a mock runner when no endpoint is set — as in benchmark_rerank
        if not args.endpoint:
            print("warning: no endpoint set — using a mock order (no real scores)")

        if args.set == "budget":  # no rerank, no embeddings: the stand makes no network call
            budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
            if 0 not in budgets:
                budgets.insert(0, 0)  # lost_vs_unlimited is counted against the unlimited run
            rows = run_budget(budgets, split=args.split, repeats=args.repeats)
            sha = _git_sha()
            for r in rows:
                r["git_sha"] = sha
            path = out_dir / f"provenance_budget_{args.split}{f'_{args.label}' if args.label else ''}.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"\n=== budget: {len(rows)} rows -> {path}")
            table = summarize_budget(rows)
            with open(path.with_name(path.stem + "_summary.json"), "w", encoding="utf-8") as fh:
                json.dump({"git_sha": sha, "repeats": args.repeats, "table": table}, fh, indent=1)
                fh.write("\n")
            return rows
        for emb in embs:
            if args.set == "multi":
                rows = run_multilingual([lg.strip() for lg in args.langs.split(",") if lg.strip()],
                                        modes, emb, args.endpoint)
            elif args.set == "household":
                rows = run_household(modes, emb, args.endpoint, split=args.split,
                                     names_fit=args.names_fit)
            elif args.set == "synthetic_zh":
                rows = run_synthetic_zh(modes, emb, args.endpoint)
            else:
                rows = run_contrast(args.set, modes, emb, args.endpoint)
            suffix = "" if args.set != "household" else (
                f"_{args.split}" + ("_nfoff" if args.names_fit == "off" else ""))
            # --label: the second reranker/model must not overwrite the first one's file (item 9: bge vs jina)
            label = f"_{args.label}" if args.label else ""
            # --recall-margin: two margins must not overwrite each other (PR #17 recipe, 2026-10-24)
            margin_sfx = "" if args.recall_margin is None else f"_m{args.recall_margin!r}"
            # --keywords / --embed-text: the two arms of issue #95 and each coverage get files of their own
            kw_sfx = "" if not args.keywords else (
                "_kwall" if args.kw_coverage == "all" else f"_kwhalf{args.kw_seed}")
            et_sfx = "" if args.embed_text is None else "_et" + args.embed_text.replace("+", "-")
            path = out_dir / f"provenance_{args.set}{suffix}{label}{margin_sfx}{kw_sfx}{et_sfx}_{emb}.jsonl"
            sha = _git_sha()
            for r in rows:
                r["git_sha"] = sha
                r["endpoint"] = args.endpoint or "mock"
                r["scale"] = SCALE  # which scale was set for the run
                r["margin"] = _holographic_embeddings.RECALL_MARGIN  # which RECALL_MARGIN the run used
                r["embed_text"] = _holographic_embeddings.EMBED_TEXT  # what the fact vectors were made from
                r["kw_file"], r["kw_coverage"] = kw_file, kw_coverage
                r.setdefault("emb_model", None)  # the store's own stamp when embeddings were on (catch_up_vectors)
                r["names_fit"] = args.names_fit if args.set == "household" else "on"
            with open(path, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"\n=== {args.set} emb={emb}: {len(rows)} queries -> {path}")
            if args.set == "household":
                summarize_household(rows)
            else:
                summarize(rows)
            all_rows.extend(rows)
        return all_rows
    finally:
        ENDPOINT, SCALE = saved_endpoint, saved_scale
        _holographic_embeddings.RECALL_MARGIN = saved_margin
        _holographic_embeddings.EMBED_TEXT = saved_embed_text
        if DB_DIR is not None and DB_DIR != saved_db_dir:  # this run's stores, closed by now: nothing left in Temp
            shutil.rmtree(DB_DIR, ignore_errors=True)
        KEYWORDS, DB_DIR = saved_keywords, saved_db_dir


if __name__ == "__main__":
    main(sys.argv[1:])