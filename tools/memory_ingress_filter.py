#!/usr/bin/env python3
"""Block prompt-extraction text before it reaches Holographic fact_store writes."""
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[\s\S]{0,120}\b(?:above|previous|prior|earlier|context|instructions?|rules?)\b",
    re.I,
)
EXFILTRATION = re.compile(
    r"\b(?:print|show|reveal|output|dump|repeat|quote)\b[\s\S]{0,120}\b(?:system\s+prompt|hidden\s+instructions?|developer\s+message|initial\s+instructions?)\b",
    re.I,
)
WRITE_ACTIONS = {"add", "update", "declare"}
STATE_TTL = 600
STATE_MAX = 12
BLOCK = {"action": "block", "message": "Ingress filter: prompt extraction text was not sent to fact_memory."}


def hostile(text: str) -> bool:
    return bool(OVERRIDE.search(text) and EXFILTRATION.search(text))


def state_file():
    home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    directory = home / "agent-hooks"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return directory / "memory_ingress_state.json"


def screen(text: str, session: str) -> bool:
    path = state_file()
    if path is None:
        return hostile(text)

    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read()
        try:
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}

        now = time.time()
        entries = [entry for entry in data.get(session, [])
                   if isinstance(entry, dict) and now - entry.get("ts", 0) < STATE_TTL]
        entries = entries[-STATE_MAX:]
        combined = " ".join([entry.get("text", "") for entry in entries] + [text])

        if hostile(combined):
            data[session] = []
        else:
            entries.append({"ts": now, "text": text})
            data[session] = entries[-STATE_MAX:]

        handle.seek(0)
        handle.truncate()
        json.dump(data, handle)
        handle.flush()
        return hostile(combined)


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
    session = str(payload.get("session_id") or payload.get("conversation_id") or "default")

    if screen(text, session):
        print(json.dumps(BLOCK))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
