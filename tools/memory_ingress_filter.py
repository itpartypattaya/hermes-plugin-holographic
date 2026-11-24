#!/usr/bin/env python3
"""Block prompt-extraction text before it reaches Holographic fact_store writes."""
import json
import re
import sys

OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[\s\S]{0,120}\b(?:above|previous|prior|earlier|context|instructions?|rules?)\b",
    re.I,
)
EXFILTRATION = re.compile(
    r"\b(?:print|show|reveal|output|dump|repeat|quote)\b[\s\S]{0,120}\b(?:system\s+prompt|hidden\s+instructions?|developer\s+message|initial\s+instructions?)\b",
    re.I,
)
WRITE_ACTIONS = {"add", "update", "declare"}


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0

    if payload.get("tool_name") != "fact_store":
        return 0

    call = payload.get("tool_input") or payload.get("args") or {}
    extra = payload.get("extra") or {}
    if isinstance(extra, dict) and isinstance(extra.get("tool_input"), dict):
        call = extra["tool_input"]
    if not isinstance(call, dict) or call.get("action") not in WRITE_ACTIONS:
        return 0

    values = [str(value) for value in call.values() if isinstance(value, str)]
    text = " ".join(values)
    if OVERRIDE.search(text) and EXFILTRATION.search(text):
        print(json.dumps({
            "action": "block",
            "message": "Ingress filter: prompt extraction text was not sent to fact_memory.",
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
