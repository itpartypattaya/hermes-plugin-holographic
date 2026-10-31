"""Tool results keep the fact's own words (issue #59, FYI): json.dumps defaults to ensure_ascii=True, so a Cyrillic
fact reached the model as \\uXXXX escapes in every result. All tool handlers serialize through _dumps
(ensure_ascii=False): raw Unicode out, byte-identical output for ASCII-only payloads, network and stored digests
untouched — and a payload that cannot be UTF-8-encoded (a lone surrogate from tool arguments) falls back to the
escaped form, because the tool executor logs the result before anything else reads it.
"""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider

# One fact carrying what the PR measures: Cyrillic, both quote kinds, a newline and an emoji (the screen keeps it).
MIXED = "Кавычки \"двойные\" и 'одинарные'\nвторая строка и 🚀 эмодзи"
KEYWORDS = ["Кавычки", "вторая", "строка"]  # explicit add keywords: the report must carry Cyrillic raw
ESCAPED_EMOJI = "\\ud83d"   # «🚀» as ensure_ascii=True would render its first surrogate half
ESCAPED_CYR = "\\u041a"     # «К» of «Кавычки»


def _texts(value):
    """Every string inside a parsed result, at any nesting."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 512})
    p.initialize(session_id="s")
    yield p
    p.shutdown()


@pytest.fixture
def mixed_fact(provider):
    """The fact added once: its raw add answer and its id feed every measured result."""
    raw = provider.handle_tool_call("fact_store", {"action": "add", "origin": "said", "content": MIXED,
                                                   "keywords": KEYWORDS})
    # enrich lists facts WITHOUT keywords: a second fact, same shape (quotes, newline, emoji), none given
    provider.handle_tool_call("fact_store", {"action": "add", "origin": "said",
                                            "content": "Второй пример: \"кавычками\"\nи 🚀 ещё одна строка"})
    return raw, json.loads(raw)["fact_id"]


def _result(provider, fact_id, action):
    if action == "list":
        return provider.handle_tool_call("fact_store", {"action": "list"})
    if action == "history":
        return provider.handle_tool_call("fact_store", {"action": "history", "fact_id": fact_id})
    if action == "search":
        return provider.handle_tool_call("fact_store", {"action": "search", "query": "Кавычки"})
    if action == "enrich":
        return provider.handle_tool_call("fact_store", {"action": "enrich", "fact_id": fact_id})
    raise AssertionError(action)


@pytest.mark.parametrize("action", ["add", "list", "history", "search", "enrich"])
def test_every_measured_result_keeps_non_ascii_raw(provider, mixed_fact, action):
    """Each result of the token table: parses as JSON, carries the emoji raw (never its \\ud83d escape), and holds
    the fact's own words — the add keywords report carries the Cyrillic words raw."""
    add_raw, fact_id = mixed_fact
    raw = add_raw if action == "add" else _result(provider, fact_id, action)
    parsed = json.loads(raw)                  # every answer stays valid JSON
    assert ESCAPED_CYR not in raw, action
    if action == "add":
        assert "Кавычки" in raw, action        # the keywords report carries Cyrillic raw (no text in an add answer)
    else:
        assert "🚀" in raw, action             # raw emoji: the old default would emit \ud83d\ude80
        assert ESCAPED_EMOJI not in raw, action
        wanted = "ещё одна строка" if action == "enrich" else "вторая строка"  # the newline half of the fact
        assert any(wanted in text for text in _texts(parsed)), action


def test_ascii_payload_is_byte_identical(provider):
    """ASCII-only content serializes the same under ensure_ascii True and False — no wire change for English facts."""
    provider.handle_tool_call("fact_store", {"action": "add", "content": "The cache prefix stays byte-stable",
                                             "origin": "said"})
    out = provider.handle_tool_call("fact_store", {"action": "list"})
    assert out == json.dumps(json.loads(out), ensure_ascii=False)
    assert out == json.dumps(json.loads(out))  # the old default: identical for pure ASCII


def test_lone_surrogate_outcome_still_encodes(tmp_path):
    """A lone surrogate in an experience outcome must leave escaped: the tool executor logs the result first, and a
    strict UTF-8 consumer would fail on a raw code point (the _dumps fallback). A plain outcome stays raw."""
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 512,
                                          "experience": True})
    p.initialize(session_id="s")
    try:
        p.handle_tool_call("fact_store", {"action": "add", "origin": "said",
                                          "content": "The nightly backup runs at 2 am."})
        block = p.prefetch("nightly backup")
        attempt = re.search(r"Memory attempt: ([A-Za-z0-9_-]+)", block)
        assert attempt, block
        aid = attempt.group(1)

        p.handle_tool_call("fact_experience", {"action": "outcome", "attempt_id": aid, "event_id": "evt-1",
                                               "outcome": "useful", "reason": "other",
                                               "correction": "обычное исправление русскими словами"})
        explain_plain = p.handle_tool_call("fact_experience", {"action": "explain", "attempt_id": aid})
        assert "обычное исправление" in explain_plain  # encodable payload: raw Cyrillic, the #62 promise

        # One active outcome per attempt (a second is a revision conflict): the surrogate gets its own attempt.
        p.handle_tool_call("fact_store", {"action": "add", "origin": "said",
                                          "content": "The tape archive runs weekly."})
        second = p.prefetch("tape archive")
        attempt2 = re.search(r"Memory attempt: ([A-Za-z0-9_-]+)", second)
        assert attempt2, second
        aid2 = attempt2.group(1)

        surrogate_correction = "хвост " + chr(0xD800) + " хвост"
        outcome = p.handle_tool_call("fact_experience", {"action": "outcome", "attempt_id": aid2,
                                                         "event_id": "evt-2", "outcome": "corrected",
                                                         "reason": "other", "correction": surrogate_correction})
        outcome.encode("utf-8")                  # must not raise
        assert chr(0xD800) not in outcome        # never a raw code point in the answer
        explain = p.handle_tool_call("fact_experience", {"action": "explain", "attempt_id": aid2})
        explain.encode("utf-8")                  # the maintainer's requirement
        assert chr(0xD800) not in explain        # the surrogate leaves escaped (the fallback escapes the payload)
        assert "\\ud800" in explain              # as its \\ud800 text
        assert any(s == surrogate_correction for s in _texts(json.loads(explain)))  # round-trips
    finally:
        p.shutdown()