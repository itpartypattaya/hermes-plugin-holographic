# -*- coding: utf-8 -*-
"""Прогон contrast_ru в режиме off (без эндпоинта): парная метрика по 15 парам,
recall@5, WRONG-ENTITY (чужая цель затеняет свою), тишина chit-chat,
две оси names_fit (on / off) + проба, что подмена names_fit реально подхвачена.

Запуск (Windows / git-bash):

    HERMES_AGENT_REPO="C:/Users/Pavel/AppData/Local/hermes/hermes-agent" \\
        python tools/contrast_off.py

Три ортогональные метрики (как у мейнтейнера, не смешиваются):
  ПАРА    — падает только по MISS: одна из двух целей не найдена (recall-аспект);
  WRONG5  — отдельный счётчик: чужая (partner-)цель в top-5 (wrong-entity);
            по измерениям порядок её НЕ чинит — чужой семантически близок;
  ШУМ     — chit-chat, вернувший не-пустой recall (admission-аспект).
SHADOW (чужая выше своей) — диагностический, по измерениям = 0 во всех режимах.

names_fit OFF эмулируется подменой метода в тестовом прогоне
(provider._retriever.names_fit -> всегда True); продакшн-код не меняется.
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "hermes_plugin_holographic", _REPO / "__init__.py", submodule_search_locations=[str(_REPO)])
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hermes_plugin_holographic"] = _mod
_spec.loader.exec_module(_mod)

from hermes_plugin_holographic import HolographicMemoryProvider  # noqa: E402

# Язык контрастного набора: CONTRAST_SET=contrast_ru (по умолчанию) | contrast_en | contrast_zh
_SET = os.environ.get("CONTRAST_SET", "contrast_ru")
_cspec = importlib.util.spec_from_file_location(_SET, _REPO / "tools" / f"{_SET}.py")
assert _cspec and _cspec.loader
contrast = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(contrast)

# Режимы PR #10 против реального эндпоинта (llama-server, логиты — проверено пробой).
# Второй элемент — rerank_weight, третий — rerank_window (δ; 0 = окно выключено).
# RERANK_URL переопределяется окружением: сравнение iGPU-инстанса и CPU (-dev none).
import os as _os
RERANK_URL = _os.environ.get("RERANK_URL", "http://127.0.0.1:8091/v1/rerank")
MODES = {
    # один режим (п.6 ревью 2026-10-22): (rerank_weight, rerank_window); gate_* = окно δ включено
    "off": None,
    "reorder_0": ("0.0", "0.0"),
    "reorder_05": ("0.5", "0.0"),
    "gate_0": ("0.0", "0.05"),
    "gate_05": ("0.5", "0.05"),
}


def _provider(db_path, **config) -> HolographicMemoryProvider:
    p = HolographicMemoryProvider(config={"db_path": str(db_path), "hrr_dim": 1024, **config})
    p.initialize(session_id="contrast")
    return p


def _fill(p) -> dict[int, str]:
    """FACTS в магазин; возвращает fact_id -> key."""
    back: dict[int, str] = {}
    for key, (text, entities) in contrast.FACTS.items():
        r = json.loads(p.handle_tool_call("fact_store",
                                          {"action": "add", "content": text, "entities": entities}))
        back[int(r["fact_id"])] = key
    return back


def _inject(p, query: str) -> list[int]:
    """fact_id prefetch до поворота, В ПОРЯДКЕ ранжирования (та же расшифровка
    блока, что в benchmark_rerank)."""
    block = p.prefetch(query)
    if not block:
        return []
    ids: list[int] = []
    for line in block.splitlines()[1:]:  # строки "- #<id> [trust] <text>"
        try:
            ids.append(int(line.split()[1][1:]))
        except (IndexError, ValueError):
            pass
    return ids


def _names_fit_probe(p) -> str:
    """Подхватилась ли подмена: запрос с именем из одного факта + чужой факт
    (две несекущиеся сущности; для языков без регистра результат не различим)."""
    items = [(k, v) for k, v in contrast.FACTS.items() if v[1]]
    pair = None
    for i, (_k1, v1) in enumerate(items):
        for _k2, v2 in items[i + 1:]:
            if not (set(v1[1]) & set(v2[1])) and all(e not in v2[0] for e in v1[1]):
                pair = (v1[1][0], {"content": v2[0]})
                break
        if pair:
            break
    if not pair:
        return "нет двух несекущихся сущностей"
    name, fact = pair
    try:
        fits = bool(p._retriever.names_fit(name, fact))
    except Exception as exc:  # noqa: BLE001 — диагностическая проба
        return f"ошибка вызова: {type(exc).__name__}: {exc}"
    return (f"fits={fits} — False: реальный метод отвергает (патч не подхвачен); "
            f"True: патч подхвачен" + ("" if fits else ""))


def run(axis: str, mode: str = "off") -> dict:
    """axis='on' — names_fit как в коде; 'off' — всегда True (тестовая подмена).
    mode — ключ MODES; 'off' = без вызовов, остальные — реальный эндпоинт."""
    spec = MODES[mode]
    config = {}
    if spec:
        config = {"rerank": "on", "rerank_weight": spec[0], "rerank_window": spec[1],
                  "rerank_url": RERANK_URL, "rerank_model": "local-reranker"}
    db = Path(tempfile.gettempdir()) / f"contrast_ru_{axis}_{mode}.db"
    for suffix in ("", "-shm", "-wal"):
        try:
            Path(str(db) + suffix).unlink()
        except OSError:
            pass
    p = _provider(db, **config)
    if axis == "off":
        p._retriever.names_fit = lambda query, fact: True
    probe = _names_fit_probe(p)
    back = _fill(p)

    pairs_out, lat = [], []
    for pair in contrast.PAIRS:
        partner = [pair["queries"][1 - i][1] for i in range(len(pair["queries"]))]
        pair_ok, per = True, []
        for i, (query, target) in enumerate(pair["queries"]):
            p.on_session_switch(f"c_{pair['id']}_{i}", reset=True)
            t0 = time.perf_counter()
            found = _inject(p, query)
            lat.append(time.perf_counter() - t0)
            keys = [back[fid] for fid in found if fid in back]
            own_pos = keys.index(target) + 1 if target in keys else None
            part = partner[i]
            if part == target:
                part_pos = None  # пара-парафразы: оба запроса к одной цели — partner-нет
            else:
                part_pos = keys.index(part) + 1 if part in keys else None
            miss = own_pos is None
            shadow = part_pos is not None and (own_pos is None or part_pos < own_pos)
            wrong5 = part_pos is not None and part_pos <= 5  # чужая в top-5: wrong-entity
            # мейнтейнера; счётчиком, НЕ провалом пары (порядок её не чинит — измерено)
            per.append({"q": query, "own": target, "own_pos": own_pos, "partner": part,
                        "partner_pos": part_pos, "keys": keys, "miss": miss, "shadow": shadow,
                        "wrong5": wrong5})
            pair_ok = pair_ok and not miss
        pairs_out.append({"id": pair["id"], "type": pair["type"], "ok": pair_ok,
                          "note": pair.get("note", ""), "per": per})

    quiet = total = 0
    chit_detail = []
    for i, (query, cat) in enumerate(contrast.CHITCHAT):
        p.on_session_switch(f"q_{i}", reset=True)
        found = _inject(p, query)
        total += 1
        quiet += not found
        if found:
            chit_detail.append((query, [back[fid] for fid in sorted(found)]))
    p.shutdown()

    by_type: dict[str, list[bool]] = {}
    for r in pairs_out:
        by_type.setdefault(r["type"], []).append(r["ok"])
    return {
        "axis": axis, "mode": mode, "probe": probe, "pairs": pairs_out,
        "pair_pass": sum(r["ok"] for r in pairs_out),
        "by_type": {t: (sum(v), len(v)) for t, v in by_type.items()},
        "question_hits": sum(not d["miss"] for r in pairs_out for d in r["per"]),
        "questions": sum(len(r["per"]) for r in pairs_out),
        "miss": sum(d["miss"] for r in pairs_out for d in r["per"]),
        "shadow": sum(d["shadow"] for r in pairs_out for d in r["per"]),
        "partner_top5": sum(d["partner_pos"] is not None and d["partner_pos"] <= 5
                            for r in pairs_out for d in r["per"]),
        "quiet": quiet, "total": total, "chit_leaks": chit_detail,
        "p50_ms": statistics.median(lat) * 1000 if lat else 0.0,
    }


def report(res: dict, verbose: bool = False) -> None:
    print(f"\n=== names_fit {res['axis']} / режим {res['mode']} | проба: {res['probe']} ===")
    print(f"пар пройдено: {res['pair_pass']}/{len(res['pairs'])} | по типам: "
          + ", ".join(f"{t} {a}/{b}" for t, (a, b) in sorted(res["by_type"].items())))
    print(f"вопросов в цель: {res['question_hits']}/{res['questions']} | "
          f"MISS {res['miss']} | SHADOW (чужая выше своей) {res['shadow']} | "
          f"partner в top-5: {res['partner_top5']} | "
          f"тишина chit-chat {res['quiet']}/{res['total']} | p50 {res['p50_ms']:.1f} мс")
    if not verbose:
        return
    for r in res["pairs"]:
        mark = "OK  " if r["ok"] else "FAIL"
        print(f"  {mark} {r['id']:<7} [{r['type']}]")
        if not r["ok"]:
            for d in r["per"]:
                flags = ("MISS" if d["miss"] else "     ") + ("/SHADOW" if d["shadow"] else "") \
                    + ("/WRONG5" if d["wrong5"] else "")
                print(f"        {flags} {d['q']!r}")
                print(f"               своя {d['own']}#{d['own_pos']} | чужая {d['partner']}#{d['partner_pos']}")
                print(f"               recall: {d['keys']}")
    if res["chit_leaks"]:
        print("  утечки chit-chat (должно быть пусто):")
        for q, found in res["chit_leaks"]:
            print(f"        {q!r} -> {found}")


if __name__ == "__main__":
    if not _AGENT.is_dir():
        raise SystemExit(f"нет каталога hermes-agent: {_AGENT} (задай HERMES_AGENT_REPO)")
    rows = [run(axis, mode) for axis in ("on", "off") for mode in MODES]
    print("\n=== СВОДНАЯ ТАБЛИЦА (парная метрика, реальный эндпоинт) ===")
    print(f"{'ось':<4} {'режим':<11} {'пар':>4} {'MISS':>5} {'WRONG5':>7} "
          f"{'тишина':>7} {'шум':>5} {'p50мс':>7}")
    for r in rows:
        noise = (r["total"] - r["quiet"]) / r["total"] if r["total"] else 0.0
        print(f"{r['axis']:<4} {r['mode']:<11} {r['pair_pass']:>4} {r['miss']:>5} "
              f"{r['partner_top5']:>7} {f'{r['quiet']}/{r['total']}':>7} {noise:>5.0%} {r['p50_ms']:>7.1f}")
    # детали — только для оси names_fit on (off-режим на обеих осях идентичен)
    for r in rows:
        if r["axis"] == "on":
            report(r, verbose=True)