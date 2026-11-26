"""The third arm of #121: recall_check.CHECK swapped for OpenAI's Decisions API, then the probe as usual.

Pre-registered in #135 (issuecomment-7185238331): gpt-6-luna, one predicate question, admitted when probability > 0.5.
A refusal, a malformed answer or a failed call is not answered (the probe stops the run). Prints the call count and
the input tokens the API reported when the run ends.
"""
import atexit
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance_probe as pp  # sets up the plugin's import path
from hermes_plugin_holographic import recall_check

HERE = os.path.dirname(os.path.abspath(__file__))
INSTRUCTION = open(os.environ["DECISIONS_INSTRUCTION"], encoding="utf-8").read()
TEMPLATE = open(os.environ["DECISIONS_TEMPLATE"], encoding="utf-8").read()
URL, MODEL, THRESHOLD = "https://api.openai.com/v1/decisions", "gpt-6-luna", 0.5
STATS = {"calls": 0, "input_tokens": 0, "failed": 0}


def check(message: str, fact: str):
    body = {"model": MODEL,
            "input": [{"role": "user", "content": [{"type": "input_text",
                                                     "text": TEMPLATE.replace("{message}", message).replace("{fact}", fact)}]}],
            "questions": [{"type": "predicate", "name": "answers", "instructions": INSTRUCTION}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"})
    STATS["calls"] += 1
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:  # noqa: BLE001 - reported, then not answered: the probe stops the run
        STATS["failed"] += 1
        print(f"decisions call failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        return recall_check.NOT_ANSWERED, None
    STATS["input_tokens"] += int((data.get("usage") or {}).get("input_tokens") or 0)
    answers = data.get("answers")
    if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], dict):
        return recall_check.NOT_ANSWERED, None
    a = answers[0]
    p = a.get("probability")
    if a.get("type") != "predicate" or a.get("name") != "answers" or not isinstance(p, (int, float)) or not 0 <= p <= 1:
        print(f"decisions answer not usable: {json.dumps(a)[:200]}", file=sys.stderr)
        return recall_check.NOT_ANSWERED, None
    return (recall_check.YES if p > THRESHOLD else recall_check.NO), float(p)


atexit.register(lambda: print(f"decisions: {STATS}", file=sys.stderr))
recall_check.CHECK = check
pp.main(sys.argv[1:])
