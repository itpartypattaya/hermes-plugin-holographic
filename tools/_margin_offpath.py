# -*- coding: utf-8 -*-
"""Off-path замер RECALL_MARGIN (пункт 3 ревью PR #10): emb=on, rerank off.

Правка порога в рантайме (embeddings.RECALL_MARGIN читается в __init__ динамически),
прогон 3 контраст-сетов на базе mode=off — то есть путь, который ревьюер просит
измерить отдельно от гейта.

    PYTHONPATH=~/.hermes/hermes-agent HERMES_AGENT_REPO=~/.hermes/hermes-agent \
        python -I _margin_offpath.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_REPO / "tools"))

import _wide_store as ws  # noqa: E402

hmod = importlib.import_module("hermes_plugin_holographic")

SETS = ("contrast_ru", "contrast_en", "contrast_zh")
MARGINS = (1.0, 0.6)

rows = []
print("| margin | сет | facts | hit/30 | шум/45 | wrong5 |")
print("|---|---|---|---|---|---|")
for margin in MARGINS:
    hmod.embeddings.RECALL_MARGIN = margin
    for set_name in SETS:
        row = ws.run(set_name, False, "off")
        rows.append({"margin": margin, "set": set_name, **row})
        print(f"| {margin} | {set_name} | {row['facts']} | {row['hit']}/{row['n']} | "
              f"{row['noise']}/{row['chit']} | {row['wrong']} |", flush=True)

out = _REPO / "results" / "margin_offpath.jsonl"
with out.open("w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"артефакт: {out}")
for margin in MARGINS:
    sub = [r for r in rows if r["margin"] == margin]
    print(f"ИТОГО margin={margin}: hit {sum(r['hit'] for r in sub)}/90, "
          f"шум {sum(r['noise'] for r in sub)}/135, wrong5 {sum(r['wrong'] for r in sub)}")