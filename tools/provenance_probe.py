# -*- coding: utf-8 -*-
"""Provenance probe: откуда каждый факт в top-5 перед ходом (лексический / смысл / имена),
что фильтр coverage вырезал, срезало ли окно rerank_window (класс E) и почему цель не найдена
(классы A/D/C).

Патчит только экземпляр провайдера в тестовом прогоне (продакшн-код не меняется):
  search      -> пул (кандидаты после blend)
  relevant    -> kept / blocked (+ позиция в пуле, top-5 пула = класс D)
  by_meaning  -> источник 'meaning'
  entity_facts-> источник 'entity'
  rescorer.scores -> порядок реранка (top-5 реранка)
  _not_recent / _recallable -> 'recent' / 'unsafe'

Запуск:
    PYTHONPATH=~/.hermes/hermes-agent python -I tools/provenance_probe.py \
        --set multi --langs en,ru,zh,de,ja,it,nl,th --modes off,reorder_only,gate_only
    PYTHONPATH=~/.hermes/hermes-agent python -I tools/provenance_probe.py \
        --set contrast_ru --modes off,gate_only --emb on
    # --endpoint http://127.0.0.1:1235/v1/rerank   реальный реранк (иначе мок)
    # --emb on|off|both                              embeddings с :1237 (по умолчанию off)

Строки: results/provenance_<set>_<emb>.jsonl (по строке на запрос; mode — поле строки, git_sha — коммит).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if "hermes_plugin_holographic" not in sys.modules:  # в тестах пакет уже загружен conftest
    _spec = importlib.util.spec_from_file_location(
        "hermes_plugin_holographic", _REPO / "__init__.py", submodule_search_locations=[str(_REPO)])
    assert _spec and _spec.loader
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["hermes_plugin_holographic"] = _mod
    _spec.loader.exec_module(_mod)

from hermes_plugin_holographic import HolographicMemoryProvider  # noqa: E402
from hermes_plugin_holographic import _sendable as _holographic_sendable  # noqa: E402
from hermes_plugin_holographic import embeddings as _holographic_embeddings  # noqa: E402

# mode -> (rerank_weight, rerank_window): окно (δ) теперь настоящая клавиша конфига (rerank_window);
# margin-вето и критериальные хуки (_margin_clears / подмены _bottom_cut) удалены (п.6 ревью 2026-10-22).
MODES = {
    "off": None,
    "reorder_only": ("0.0", "0.0"),
    "reorder_blend": ("0.5", "0.0"),
    "gate_only": ("0.0", "0.05"),
    "gate_blend": ("0.5", "0.05"),
    # свип окна (бывшие botwin:δ) — то же числом в rerank_window:
    "win10": ("0.0", "0.10"),
    "win03": ("0.0", "0.03"),
    "win02": ("0.0", "0.02"),
    # режимов gate_min2/gate_named больше (п.1 ревью 13:06): минимум два —
    # встроено в окно, защита названного лица удалена; gate_only = готовая
    # конфигурация для мержа (δ=0.05 + этот встроенный минимум).
}


EMB_URL = os.environ.get("EMB_URL", "http://127.0.0.1:1237/v1")
EMB_MODEL = os.environ.get("EMB_MODEL", "bge-m3")
ENDPOINT = os.environ.get("RERANK_URL", "http://127.0.0.1:1235/v1/rerank")
SCALE = os.environ.get("RERANK_SCALE", "logit")  # шкала эндпоинта (п.4 фикс-листа 14:01)


def _git_sha() -> str:
    """Стем коммита: каждая строка JSONL знает, на каком коде снята (п.8 ревью 2026-10-22)."""
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- данные наборов

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


# ---------------------------------------------------------------- провайдер

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
    """Собирает provenance одного prefetch. Контекст сбрасывается на каждый запрос."""

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
        # runner на main не имеет _rescorer вообще — getattr, не прямой доступ (п.5 фикс-листа)
        rsc = getattr(p, "_rescorer", None)
        if rsc is not None:
            self._wrap(rsc, "scores", self._scores)

    def _wrap(self, obj, name: str, wrapper):
        orig = getattr(obj, name, None)
        if orig is None:
            return
        self._orig[(id(obj), name)] = orig
        setattr(obj, name, wrapper(orig))

    # -- вызовы пайплайна ------------------------------------------------
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
            # ranked индексирует ДОКУМЕНТЫ (send-список), а не пул: при meaning/entity-потоках
            # union и pool не совпадают, поэтому запоминаем union и send-позиции для перевода
            # индекса ответа в fact_id (раньше писалось pool[i] — чужие id в JSONL).
            self.ctx["union"] = [f.get("fact_id") for f in facts]
            self.ctx["send_idx"] = list(out[0])
            return out
        return call

    def _scores(self, orig):
        def call(query, docs):
            ranked = orig(query, docs)
            owner = getattr(orig, "__self__", None)  # тестовые стабы подменяют scores функцией без __self__
            status = getattr(owner, "last_status", None) or "not_called"
            if ranked is None and status.startswith("failed"):
                # Фикс-лист 19:07: упавший вызов не должен выглядеть как «реранкер ничего не изменил» —
                # прогон останавливается сразу, чтобы итоговые строки гарантированно имели 0 failed calls.
                raise SystemExit(
                    f"rerank-вызов не удался ({status}) — прогон остановлен "
                    "(фикс-лист 19:07: нулевой бюджет failed calls)")
            self.ctx["rerank_call"] = status
            if ranked is None:
                self.ctx["ranked"] = False
                return ranked
            union = self.ctx.get("union") or []
            send_idx = self.ctx.get("send_idx") or []
            order = [union[send_idx[i]] for i, _ in ranked
                     if 0 <= i < len(send_idx) and send_idx[i] < len(union)]
            self.ctx["rank_order"] = order  # порядок реранка ДО окна и top-5
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

    # -- обёртка запроса --------------------------------------------------
    def prefetch(self, query: str) -> tuple[str, dict]:
        self.ctx = {"tags": {}, "blocked": {}, "gate_called": False,
                    "pool": [], "rerank_top5": [], "rank_order": [], "ranked": None,
                    "rerank_call": "not_called",
                    "window": 0.0, "weight": 0.0}
        rescorer = getattr(self.p, "_rescorer", None)
        if rescorer is None:
            self.ctx["rerank_call"] = "off"  # rerank выключен конфигом — вызова не будет по построению
        else:
            rescorer.last_status = "not_called"  # исход известен только после вызова; сброс на каждый запрос
        if rescorer is not None and getattr(rescorer, "settings", None) is not None:
            # окно режет только при weight == 0 (см. _unified_stream) — фиксируем для classify
            self.ctx["window"] = float(getattr(rescorer.settings, "window", 0.0) or 0.0)
            self.ctx["weight"] = float(getattr(rescorer.settings, "weight", 0.0) or 0.0)
        block = self.p.prefetch(query)
        ctx = dict(self.ctx)
        ctx["tags"] = {k: sorted(v) for k, v in ctx["tags"].items()}
        ctx["blocked"] = dict(ctx["blocked"])
        return block, ctx


def catch_up_vectors(p: HolographicMemoryProvider) -> bool:
    """Векторы фактов делаются только в _catch_up_vectors (after-turn хук ядра).
    В прогоне драйвера его никто не вызывает -> индекс пуст -> meaning=None.
    Гоняем вручную, пока есть pending: ready() проверяет «хотя бы один вектор»
    (LIMIT 1), а бюджет одного _catch_up_vectors (embedding_catch_up_seconds)
    отсекает хвост — на wide-базе (600+ фактов) это вырезало цели real-запросов
    (fact_id 1..25): «готово» при 25 неэбедженных."""
    if not p._embedding:
        return True
    idx = p._retriever.vector_index(p._embedding.tag)
    for _ in range(60):
        try:
            pending = idx.pending(_holographic_embeddings.BATCH, allowed=_holographic_sendable)
        except Exception:
            pending = None
        if pending is None:  # API неожиданного вида — старая эвристика
            if idx.ready():
                return True
        elif not pending:
            return True
        p._catch_up_vectors()
    return True


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
    """Почему цель не в recall (для real-запросов).

    E («срезан окном»): цель прошла все фильтры и стояла в top-5 ранга ДО среза
    rerank_window — её убрало окно, а не слова и не top-5. Раньше такой случай
    лгал «ниже top-5 (пул#N)»: пул — порядок кандидатов, а не ранг реранка
    (кейс «What does Andrey do on weekends?», ревью 2026-10-23).
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
            # top-5 ранга без среза должен был попасть в recall — сюда не доходим
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
        # gate_reads_score адмитит пул без coverage: цель либо ниже top-5, либо отвергнута margin
        order = ctx.get("gate_order") or pool
        if fid in order[:5]:
            return "gate: отвергнут margin (в top-5 порядка!)"
        return "C: ниже top-5"
    # coverage пустил, но не в recall: свежесть/лимит top-5
    order = ctx.get("gate_order") or pool
    pos = order.index(fid) if fid in order else 999
    return f"C: ниже top-5 (пул#{pos})"


# ---------------------------------------------------------------- прогоны

def run_multilingual(langs, modes, emb, endpoint, extra: dict | None = None) -> list[dict]:
    data = load_multilingual()
    rows = []
    for mode in modes:
        for lang in langs:
            db = Path(tempfile.gettempdir()) / f"prov_ml_{lang}_{mode}.db"
            for s in ("", "-shm", "-wal"):
                try:
                    Path(str(db) + s).unlink()
                except OSError:
                    pass
            p = make_provider(db, mode, emb, extra)
            back = {}
            for key, (text, entities) in data.CORPUS[lang].items():
                r = json.loads(p.handle_tool_call(
                    "fact_store", {"action": "add", "content": text, "entities": entities}))
                back[int(r["fact_id"])] = key
            probe = Probe(p)
            if emb == "on":
                ok = catch_up_vectors(p)
                print(f"  [{lang}] векторы готовы: {ok}")
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
                # matched top-5 (п.8): off умеет вернуть больше пяти строк, unified — нет; честное сравнение
                # по первым пяти, плюс счёт «всего строк» как output count.
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
                    # цель(и) без hit: класс по первой ненайденной
                    miss_cls = next((classify(next((f for f, k in back.items() if k == rk), None), ctx)
                                     for rk in relevant if rk not in keys), "-")
                rows.append({"set": "multi", "lang": lang, "mode": mode, "emb": emb,
                             "kind": kind, "q": query, "keys": keys,
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
            p.shutdown()
    return rows


def li(mode, lang):  # префикс сессии — уникален на режим+язык
    return f"{mode}_{lang}"


def run_contrast(set_name, modes, emb, endpoint, extra: dict | None = None) -> list[dict]:
    contrast = load_contrast(set_name)
    rows = []
    for mode in modes:
        db = Path(tempfile.gettempdir()) / f"prov_{set_name}_{mode}.db"
        for s in ("", "-shm", "-wal"):
            try:
                Path(str(db) + s).unlink()
            except OSError:
                pass
        p = make_provider(db, mode, emb, extra)
        back = {}
        for key, (text, entities) in contrast.FACTS.items():
            r = json.loads(p.handle_tool_call(
                "fact_store", {"action": "add", "content": text, "entities": entities}))
            back[int(r["fact_id"])] = key
        probe = Probe(p)
        if emb == "on":
            ok = catch_up_vectors(p)
            print(f"  [{set_name}] векторы готовы: {ok}")
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
                             "kind": "real", "q": query, "keys": keys, "target": target,
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
                         "kind": "chit", "cat": cat, "q": query, "keys": keys,
                         "sources": {str(f): ctx["tags"].get(f, ["pool-only"]) for f in found},
                         "hit": None, "wrong": [], "hit5": None, "out_n": len(keys), "miss_cls": "-",
                         "pool_n": len(ctx.get("pool") or []),
                         "gate_called": ctx.get("gate_called"),
                         "rerank_call": ctx.get("rerank_call"),
                         "rerank_top5": ctx.get("rerank_top5") or []})
        p.shutdown()
    return rows


# ---------------------------------------------------------------- отчёт


def run_household(modes, emb, endpoint, split="all", names_fit="on",
                  extra: dict | None = None) -> list[dict]:
    """П.9 (рецепт майнтейнера, PR #10 2026-10-22): один дом, en/ru/zh,
    dev/test заморожены, лейблы на каждый вопрос.

    Строки несут qid/split/slice/lang — отчёт считает rescued/lost и классы
    промаха (E срезан окном / A нет в пуле / C ниже top-5) по языкам.
    names_fit off = проверка имени выключена (как в tests ..._rerank_review).
    """
    hh = load_contrast("household")
    rows = []
    qids = [k for k, q in hh.QUERIES.items() if split == "all" or q["split"] == split]
    hids = [k for k, h in hh.CHITCHAT.items() if split == "all" or h["split"] == split]
    for mode in modes:
        for lang in hh.LANGS:
            db = Path(tempfile.gettempdir()) / f"prov_hh_{lang}_{mode}_{names_fit}.db"
            for s in ("", "-shm", "-wal"):
                try:
                    Path(str(db) + s).unlink()
                except OSError:
                    pass
            p = make_provider(db, mode, emb, extra)
            if names_fit == "off":
                p._retriever.names_fit = lambda *a, **k: True
            back: dict[int, str] = {}
            for key, v in hh.FACTS.items():
                text, ents = v[lang]
                r = json.loads(p.handle_tool_call(
                    "fact_store", {"action": "add", "content": text, "entities": ents}))
                back[int(r["fact_id"])] = key
            ent_map = hh.entities(lang)  # key -> имена сущностей этого языка
            probe = Probe(p)
            if emb == "on":
                ok = catch_up_vectors(p)
                print(f"  [household {lang}] векторы готовы: {ok}")
            for qid in qids:
                q = hh.QUERIES[qid]
                query, targets = q[lang], set(q["targets"])
                p.on_session_switch(f"hh{lang}_{mode}_{qid}", reset=True)
                t0 = time.perf_counter()
                block, ctx = probe.prefetch(query)
                ms = (time.perf_counter() - t0) * 1000
                found = parse_ids(block)
                keys = [back[f] for f in found if f in back]
                # старый wrong-лукап искал fact key в словаре имён (всегда пусто —
                # баг из ревью 10:05). Теперь две колонки:
                #   named   — не-цели, упоминающие лицо из запроса (правильные
                #             extra и тёзки; описание, case-слабо на ru);
                #   confused — лейблы неправильных ответов в топ-5 (мера, статика).
                named = hh.named_extra(lang, query, keys, targets, ent_map)
                conf = hh.confusion(q, keys[:5])
                hit = bool(set(keys) & targets) if targets else None
                hit5 = bool(set(keys[:5]) & targets) if targets else None
                cov = (len(set(keys) & targets) / len(targets)) if targets else None
                miss = []
                for t in q["targets"]:  # каждый лейбл, не попавший в top-5
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
            p.shutdown()
    return rows


def _cls(misses: list[str], prefix: str) -> int:
    return sum(1 for m in misses if m.startswith(prefix))


def summarize_household(rows: list[dict]) -> None:
    """Таблица по языкам × сплиту: hit@5, покрытие, named/confused, шум (только
    где лейбл пустой) и три класса промаха — E срезан окном / A нет в пуле /
    C ниже top-5 (запрос майнтейнера 2026-10-23, per language). В конце —
    парный счёт confused off→режим: строка «путаница», когда лейбл неправильного
    ответа попал в топ-5 (его 10:34: считать парно off против on)."""
    modes = sorted({(r["mode"], r["emb"]) for r in rows})
    for mode, emb in modes:
        sel = [r for r in rows if r["mode"] == mode and r["emb"] == emb]
        print(f"\n=== household mode={mode} emb={emb} ===")
        print(f"{'lang':<4} {'split':<5} {'real':>4} {'hit5':>4} {'cov%':>5} {'named':>5} "
              f"{'conf':>4} {'шум':>4} {'E':>2} {'A':>2} {'C':>2} {'проч':>4}")
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
        print(f"{'итого':<9} {len(lab):>4} {hits5:>4} "
              f"hit5={hits5}/{len(lab)} named={named} confused={conf}")

    # парный счёт (майнтейнера 10:34): строка confusion, если хотя бы один лейбл
    # неправильного ответа в топ-5; off против каждого режима — снято/добавлено
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
                  f"снято {cut}, добавлено {add} (строк {len(both)})")

    # per-target lost/gained (п.5 ревью 13:06): hit@5 «любая цель» скрывает частичные
    # потери multi-target запросов. Считаем top-5 с ОБЕИХ сторон парно против off —
    # раньше цели брались из всего keys, а у off могло быть больше пяти строк, так что
    # счёт завышал потери; печатаем и gained, по id (qid:цель) с языком каждой строки.
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
    print(f"\n{'режим':<14} {'emb':<4} {'real':>5} {'hit':>5} {'hit5':>5} {'шум':>5} "
          f"{'wrong':>5} {'wrong5':>6} {'out':>4} | источники recall (доминирует) | классы промахов")
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
        # источники всех recalled-фактов
        src: dict[str, int] = {}
        for r in sel:
            for tags in r["sources"].values():
                key = "+".join(t if t != "pool-only" else "words?" for t in tags) \
                    if "pool-only" not in tags else "words?(без coverage — gate)"
                src[key] = src.get(key, 0) + 1
        src_s = ", ".join(f"{k}:{v}" for k, v in sorted(src.items(), key=lambda x: -x[1]))
        classes: dict[str, int] = {}
        for r in real:
            if r["kind"] == "real" and not r["hit"]:
                classes[r["miss_cls"]] = classes.get(r["miss_cls"], 0) + 1
        cls_s = "; ".join(f"{k}:{v}" for k, v in sorted(classes.items(), key=lambda x: -x[1])) or "—"
        print(f"{mode:<14} {emb:<4} {len(real):>5} {hits:>5} {hits5:>5} {noise:>5}/{len(chit):<3} "
              f"{wrong:>5} {wrong5:>6} {out_avg:>4.1f} | {src_s} | {cls_s}")


def main(argv):
    global ENDPOINT, SCALE
    ap = argparse.ArgumentParser(description="Provenance probe: источники фактов в prefetch")
    ap.add_argument("--set", default="multi",
                    help="multi | contrast_ru | contrast_en | contrast_zh | household")
    ap.add_argument("--split", default="all", choices=["all", "dev", "test"],
                    help="для --set household: какой сплит гонять (по умолчанию все)")
    ap.add_argument("--names-fit", default="on", choices=["on", "off"],
                    help="для --set household: names_fit вкл/выкл (ось свипа п.9)")
    ap.add_argument("--langs", default="en,ru,zh,de,ja,it,nl,th",
                    help="языки для --set multi (по умолчанию: ключевые из бенча)")
    ap.add_argument("--modes", default="off,reorder_only,gate_only")
    ap.add_argument("--emb", default="off", choices=["off", "on", "both"])
    ap.add_argument("--endpoint", default=ENDPOINT, help="/v1/rerank URL (мок, если не задан)")
    ap.add_argument("--rerank-scale", default="logit", choices=["logit", "unit"],
                    help="rerank_scale: шкала эндпоинта; значение пишется в каждую строку (п.4, 14:01)")
    ap.add_argument("--out", default=None, help="каталог JSONL (по умолчанию results/)")
    ap.add_argument("--label", default=None,
                    help="метка в имени файла (например jina): не перезаписывает строки другого реранкера")
    args = ap.parse_args(argv)

    ENDPOINT = args.endpoint
    SCALE = args.rerank_scale

    out_dir = Path(args.out) if args.out else (_REPO / "results")
    out_dir.mkdir(exist_ok=True)
    embs = ["off", "on"] if args.emb == "both" else [args.emb]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_rows: list[dict] = []

    # мок-раннер, если endpoint не задан — как в benchmark_rerank
    if not args.endpoint:
        print("внимание: endpoint не задан — используется мок-порядок (без реальных скоров)")

    for emb in embs:
        if args.set == "multi":
            rows = run_multilingual([lg.strip() for lg in args.langs.split(",") if lg.strip()],
                                    modes, emb, args.endpoint)
        elif args.set == "household":
            rows = run_household(modes, emb, args.endpoint, split=args.split,
                                 names_fit=args.names_fit)
        else:
            rows = run_contrast(args.set, modes, emb, args.endpoint)
        suffix = "" if args.set != "household" else (
            f"_{args.split}" + ("_nfoff" if args.names_fit == "off" else ""))
        # --label: второй реранкер/модель не должен перезаписать файл первого (п.9: bge vs jina)
        label = f"_{args.label}" if args.label else ""
        path = out_dir / f"provenance_{args.set}{suffix}{label}_{emb}.jsonl"
        sha = _git_sha()
        for r in rows:
            r["git_sha"] = sha
            r["endpoint"] = args.endpoint or "mock"
            r["scale"] = SCALE  # какая шкала была выставлена при прогоне
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n=== {args.set} emb={emb}: {len(rows)} запросов -> {path}")
        if args.set == "household":
            summarize_household(rows)
        else:
            summarize(rows)
        all_rows.extend(rows)
    return all_rows


if __name__ == "__main__":
    main(sys.argv[1:])