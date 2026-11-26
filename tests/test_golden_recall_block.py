"""0.32.0 changes nothing the model reads while its new switches are off.

The recall block and the system block were recorded on main (e0218ab, 0.31.7) from a store with every kind of
history a fact can have (change, correct, revert, end, merge), facts that share an entity, a relative date in a fact's
text, with ``recall_source`` off and short. With ``recall_history: off`` and ``absolute_dates: off`` the same
sequence must give the same bytes. The recorded file is ``fixtures/golden_recall_block.json``; it is rewritten only by
hand (``python tests/test_golden_recall_block.py <out.json>`` on the release whose output it pins).
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "fixtures" / "golden_recall_block.json"
OFF = {"recall_history": "off", "absolute_dates": "off"}

QUERIES = ["Where does Alice Smith live?", "observatory budget", "Bob Jones night shift",
           "what time is the telescope meeting", "Alice Smith observatory project", "Carol Diaz coffee"]


def _store(p, **args):
    return json.loads(p.handle_tool_call("fact_store", args))


def build(provider_cls, db_path: str, **config) -> dict:
    """The recorded sequence: facts with history, then a recall per query and the system block."""
    p = provider_cls({"db_path": db_path, **config})
    p.initialize(session_id="golden", platform="cli")
    try:
        alice = _store(p, action="add", content="Alice Smith lives in Paris", origin="said")["fact_id"]
        _store(p, action="change", fact_id=alice, content="Alice Smith lives in Berlin", origin="said")
        budget = _store(p, action="add", content="The observatory budget is 500 euros", origin="said")["fact_id"]
        _store(p, action="correct", fact_id=budget, content="The observatory budget is 600 euros", origin="said")
        shift = _store(p, action="add", content="Bob Jones runs the observatory night shift", origin="said")["fact_id"]
        _store(p, action="change", fact_id=shift, content="Bob Jones runs the observatory day shift", origin="said")
        _store(p, action="revert", fact_id=shift)
        _store(p, action="add", content="Alice Smith works on the observatory project", origin="inferred")
        _store(p, action="add", content="The telescope meeting is tomorrow at 10:00", origin="said")
        old = _store(p, action="add", content="Carol Diaz drinks coffee every morning", origin="said")["fact_id"]
        _store(p, action="end", fact_id=old)
        keep = _store(p, action="add", content="Carol Diaz likes tea with milk", origin="said")["fact_id"]
        dup = _store(p, action="add", content="Carol Diaz enjoys milky tea", origin="said")["fact_id"]
        _store(p, action="merge", fact_id=keep, **{"from": [dup]})
        blocks = [p.prefetch(q, session_id="golden") for q in QUERIES]
        return {"blocks": blocks, "system": p.system_prompt_block()}
    finally:
        p.shutdown()


def _neutral(snapshot: dict) -> dict:
    """The day the sequence ran on is the only part that differs between runs."""
    days = {datetime.now(timezone.utc).date().isoformat(), datetime.now().astimezone().date().isoformat()}
    swap = lambda s: re.sub("|".join(map(re.escape, days)), "<DAY>", s)  # noqa: E731
    return {"blocks": [swap(b) for b in snapshot["blocks"]], "system": swap(snapshot["system"])}


@pytest.mark.parametrize("source", ["off", "short"])
def test_with_the_new_switches_off_recall_reads_as_in_0317(tmp_path, source):
    from hermes_plugin_holographic import HolographicMemoryProvider

    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))[source]
    got = _neutral(build(HolographicMemoryProvider, str(tmp_path / "golden.db"), recall_source=source, **OFF))
    assert any(b for b in expected["blocks"]), "the recorded sequence recalls something"
    assert got == expected


if __name__ == "__main__":  # record: run from the repo root with the test environment's Python
    sys.path.insert(0, str(Path(__file__).parent))
    import conftest  # noqa: F401 - loads the package as hermes_plugin_holographic
    from hermes_plugin_holographic import HolographicMemoryProvider
    import tempfile

    out = {}
    for source in ("off", "short"):
        with tempfile.TemporaryDirectory() as d:
            out[source] = _neutral(build(HolographicMemoryProvider, str(Path(d) / "golden.db"), recall_source=source))
    Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
