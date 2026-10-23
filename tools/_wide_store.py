# -*- coding: utf-8 -*-
"""Wide-база: A/B «размер магазина» на одних и тех же запросах.

  baseline — только свои факты сета (~64);
  wide     — + крауд (факты о тех же персонажах, язык сета; пул запроса разрастается
             к/за top_k=40) + весь многоязычный CORPUS (360 фактов, 18 языков).

Метрики: hit/шум/wrong5, глубина пула (pool_n, доля запросов с pool>=40 = top_k реально
работает), доля hit-целей глубже позиции 5 в пуле (подъём рангом), латентность prefetch
p50/p95, время эмбеддинга базы.

Запуск:
    PYTHONPATH=~/.hermes/hermes-agent HERMES_AGENT_REPO=~/.hermes/hermes-agent \
        python -I tools/_wide_store.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(_REPO / "tools"))

import provenance_probe as pp  # noqa: E402

# крауд-шаблоны: {n} = имя персонажа сета (запросы делят слова с массой этих фактов)
CROWD = {
    "en": [
        "{n} goes jogging along the beach at six in the morning on weekdays",
        "{n} ordered new tyres for the car last month",
        "{n} had dinner with {o} at the Thai place last Friday",
        "{n} keeps a spare key under the flowerpot",
        "{n} is learning to play the guitar this year",
        "{n} booked a table at the Italian restaurant for Saturday",
        "{n} complained about the noise from the neighbours again",
        "{n} started swimming classes in the new pool",
        "{n} works late on Thursdays this month",
        "{n} studies something new every evening",
        "{n} does the shopping at the market on Sundays",
        "{n} has a meeting downtown at nine in the morning",
        "{n} drives the old sedan to work",
        "{n} keeps his schedule in a paper notebook",
        "{n} plays chess in the park after lunch",
        "{n} goes to the gym twice a week",
    ],
    "ru": [
        "{n} бегает по пляжу в шесть утра по будням",
        "{n} заказал новые шины для машины в прошлом месяце",
        "{n} ужинал с {o} в томайском кафе в прошлую пятницу",
        "{n} держит запасной ключ под кашпо",
        "{n} в этом году учится играть на гитаре",
        "{n} забронировал стол в итальянском ресторане на субботу",
        "{n} снова жалуется на шум от соседей",
        "{n} начал плавание в новом бассейне",
        "{n} работает допоздна по четвергам",
        "{n} каждый вечер чему-то учится",
        "{n} ходит за покупками на рынок по воскресеньям",
        "{n} назначил встречу в центре на девять утра",
        "{n} ездит на работу на старом седане",
        "{n} записывает дела в бумажный блокнот",
        "{n} играет в шахматы в парке после обеда",
        "{n} ходит в спортзал дважды в неделю",
    ],
    "zh": [
        "{n}每天早上六点沿着海滩跑步",
        "{n}上个月给车换了新轮胎",
        "{n}上周五和{o}在泰国餐厅吃了晚饭",
        "{n}把备用钥匙放在花盆下面",
        "{n}今年在学吉他",
        "{n}订了周六意大利餐厅的桌子",
        "{n}又抱怨邻居的噪音",
        "{n}开始在新泳池游泳",
        "{n}周四经常加班到很晚",
        "{n}每天晚上都在学新东西",
        "{n}周日去市场买东西",
        "{n}约了九点在市中心见面",
        "{n}开旧轿车上班",
        "{n}把日程记在纸质笔记本上",
        "{n}午饭后在公园下棋",
        "{n}一周去两次健身房",
    ],
}


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def crowd_facts(contrast, lang: str) -> list[tuple[str, str, list[str]]]:
    """(key, content, entities): по 8 фактов на каждого персонажа с именами."""
    names: list[str] = []
    for _text, ents in contrast.FACTS.values():
        for e in ents:
            if e not in names:
                names.append(e)
    out = []
    for idx, n in enumerate(names):
        for j, tpl in enumerate(CROWD[lang]):
            o = names[(idx + 3) % len(names)]
            out.append((f"wide_{lang}_{idx}_{j}", tpl.format(n=n, o=o), [n]))
    return out


def run(set_name: str, wide: bool, mode: str) -> dict:
    contrast = pp.load_contrast(set_name)
    lang = set_name.replace("contrast_", "")
    db = Path(tempfile.gettempdir()) / f"wide_{set_name}_{mode}_{int(wide)}.db"
    for s in ("", "-shm", "-wal"):
        try:
            Path(str(db) + s).unlink()
        except OSError:
            pass
    p = pp.make_provider(db, mode, "on")
    back: dict[int, str] = {}

    def add(text: str, entities: list[str], key: str) -> None:
        r = json.loads(p.handle_tool_call(
            "fact_store", {"action": "add", "content": text, "entities": entities}))
        back[int(r["fact_id"])] = key

    n_facts = 0
    for key, (text, entities) in contrast.FACTS.items():
        add(text, entities, key)
        n_facts += 1
    if wide:
        for key, text, entities in crowd_facts(contrast, lang):
            add(text, entities, key)
            n_facts += 1
        multi = pp.load_multilingual()
        for lg, facts in multi.CORPUS.items():
            for key, (text, entities) in facts.items():
                add(text, entities, f"multi_{lg}_{key}")
                n_facts += 1
    t0 = time.perf_counter()
    assert pp.catch_up_vectors(p), "векторы не готовы"
    embed_s = round(time.perf_counter() - t0, 1)
    probe = pp.Probe(p)

    hit = n = noise = chit = wrong = deep = deep_hits = 0
    misses: list[str] = []
    pools: list[int] = []
    lats: list[float] = []
    for pi, pair in enumerate(contrast.PAIRS):
        partner = [pair["queries"][1 - i][1] for i in range(len(pair["queries"]))]
        for i, (query, target) in enumerate(pair["queries"]):
            p.on_session_switch(f"w{pi}_{i}", reset=True)
            t0 = time.perf_counter()
            block, ctx = probe.prefetch(query)
            lats.append(time.perf_counter() - t0)
            found = pp.parse_ids(block)
            tgt = next((fid for fid, key in back.items() if key == target), None)
            n += 1
            ok = bool(tgt and tgt in found)
            hit += ok
            if not ok and tgt is not None:
                misses.append(pp.classify(tgt, ctx))
            pool = ctx.get("pool") or []
            pools.append(len(pool))
            if tgt and tgt in pool and tgt in found:
                deep_hits += 1
                deep += pool.index(tgt) > 4  # цель не в топ-5 пула: её поднял ранг
            p_partner = next((fid for fid, key in back.items() if key == partner[i]), None)
            wrong += partner[i] != target and bool(p_partner and p_partner in found[:5])
    for i, (query, _cat) in enumerate(contrast.CHITCHAT):
        p.on_session_switch(f"wq{i}", reset=True)
        t0 = time.perf_counter()
        block, ctx = probe.prefetch(query)
        lats.append(time.perf_counter() - t0)
        chit += 1
        noise += bool(pp.parse_ids(block))
        pools.append(len(ctx.get("pool") or []))
    p.shutdown()
    return {
        "facts": n_facts, "embed_s": embed_s,
        "hit": hit, "n": n, "noise": noise, "chit": chit, "wrong": wrong,
        "pool_p50": percentile(pools, 0.5), "pool_max": max(pools),
        "bind40": sum(1 for x in pools if x >= 40),
        "deep": f"{deep}/{deep_hits}", "misses": misses,
        "lat_p50": round(percentile(lats, 0.5) * 1000, 1),
        "lat_p95": round(percentile(lats, 0.95) * 1000, 1),
    }


def main() -> None:
    wanted = sys.argv[1:] or ["contrast_ru", "contrast_en", "contrast_zh"]
    all_misses: list[tuple] = []
    print("| сет | режим | база | facts | hit | шум | wrong5 | pool p50/max | >=40 | deep | lat p50/p95 | embed с |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for set_name in wanted:
        for mode in ("off", "gate_only"):
            for wide in (False, True):
                row = run(set_name, wide, mode)
                all_misses.append((set_name, mode, wide, row["misses"]))
                print(f"| {set_name} | {mode} | {'wide' if wide else 'base'} | {row['facts']} | "
                      f"{row['hit']}/{row['n']} | {row['noise']}/{row['chit']} | {row['wrong']} | "
                      f"{row['pool_p50']:.0f}/{row['pool_max']} | {row['bind40']} | {row['deep']} | "
                      f"{row['lat_p50']}/{row['lat_p95']} | {row['embed_s']} |",
                      flush=True)
    for set_name2, mode2, wide2, ms in all_misses:
        if ms:
            cnt = {}
            for m in ms:
                cnt[m] = cnt.get(m, 0) + 1
            print(f"  miss {set_name2} {mode2} {'wide' if wide2 else 'base'}: {cnt}", flush=True)


if __name__ == "__main__":
    main()