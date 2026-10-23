# -*- coding: utf-8 -*-
"""Свип RECALL_MARGIN (семпоток-барьер): как margin z_bar меняет hit/шум/wrong5
и число meaning-тегов в recall. Подменяет константу в модуле — продакшн не трогает.

Запуск:
    PYTHONPATH=~/.hermes/hermes-agent HERMES_AGENT_REPO=~/.hermes/hermes-agent \
        python -I tools/_sem_stream_sweep.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENT = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
for _p in (str(_AGENT), str(_REPO.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_REPO / "tools"))

import provenance_probe as pp  # noqa: E402
from hermes_plugin_holographic import embeddings  # noqa: E402

SETS = ("contrast_ru", "contrast_en", "contrast_zh")
MARGINS = tuple(float(a) for a in sys.argv[1:]) or (1.0, 0.7, 0.5)
MODES = ("off", "gate_only")


def agg(rows: list[dict]) -> dict:
    real = [r for r in rows if r["kind"] == "real"]
    chit = [r for r in rows if r["kind"] != "real"]
    meaning = sum(1 for r in real for t in (r.get("sources") or {}).values() if "meaning" in t)
    return {
        "hit": sum(1 for r in real if r["hit"]),
        "n": len(real),
        "noise": sum(1 for r in chit if r["keys"]),
        "chit": len(chit),
        "wrong": sum(1 for r in real if r["wrong"]),
        "meaning_tags": meaning,
    }


def main() -> None:
    print("| margin | set | mode | hit | шум | wrong5 | meaning-тегов |")
    print("|---|---|---|---|---|---|---|")
    for margin in MARGINS:
        embeddings.RECALL_MARGIN = margin  # подмена до прогонов: z_bar читает её вызовом
        for set_name in SETS + ("multi",):
            langs = ("ar,bn,de,en,es,fr,hi,id,it,ja,ko,nl,pl,pt,ru,th,tr,zh"
                     if set_name == "multi" else None)
            rows = pp.run_contrast(set_name, list(MODES), "on", pp.ENDPOINT) if langs is None else \
                pp.run_multilingual(langs.split(","), list(MODES), "on", pp.ENDPOINT)
            for mode in MODES:
                part = [r for r in rows if r["mode"] == mode]
                a = agg(part)
                print(f"| {margin} | {set_name} | {mode} | {a['hit']}/{a['n']} | "
                      f"{a['noise']}/{a['chit']} | {a['wrong']} | {a['meaning_tags']} |")


if __name__ == "__main__":
    main()