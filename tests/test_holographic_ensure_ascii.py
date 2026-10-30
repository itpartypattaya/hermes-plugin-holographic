"""Tool results keep the fact's own words (issue #59, FYI): json.dumps defaults to ensure_ascii=True, so a Cyrillic
fact reached the model as \\uXXXX escapes in every result. All tool handlers now serialize through _dumps
(ensure_ascii=False): raw Unicode out, byte-identical output for ASCII-only payloads, network and stored digests
untouched."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider

CYRILLIC = "Ловушка: change без keywords вычищает их — by design"
ESCAPED_HEAD = "\\u041b"  # «Л» as ensure_ascii=True would render it


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 512})
    p.initialize(session_id="s")
    yield p
    p.shutdown()


def _raw(provider, **args):
    return provider.handle_tool_call("fact_store", args)


def test_cyrillic_fact_is_raw_in_every_result(provider):
    """The case itself: add, list and history must carry «Ловушка», never \\u041b."""
    added = _raw(provider, action="add", content=CYRILLIC, origin="said")
    fact_id = json.loads(added)["fact_id"]
    for call in ({"action": "list"}, {"action": "history", "fact_id": fact_id},
                 {"action": "search", "query": "Ловушка"}):
        out = _raw(provider, **call)
        assert "Ловушка" in out, call
        assert ESCAPED_HEAD not in out, call


def test_ascii_payload_is_byte_identical(provider):
    """ASCII-only content serializes the same under ensure_ascii True and False — no wire change for English facts."""
    _raw(provider, action="add", content="The cache prefix stays byte-stable", origin="said")
    out = _raw(provider, action="list")
    assert out == json.dumps(json.loads(out), ensure_ascii=False)
    assert out == json.dumps(json.loads(out))  # the old default: identical for pure ASCII


def test_results_stay_valid_json(provider):
    """Raw Unicode is an encoding choice, not a format change: every result still parses."""
    _raw(provider, action="add", content=CYRILLIC, origin="said")
    parsed = json.loads(_raw(provider, action="list"))
    assert any("Ловушка" in f.get("content", "") for f in parsed["facts"])