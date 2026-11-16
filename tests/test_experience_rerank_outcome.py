"""The journal says why the reranker did or did not order a recall (0.28.1).

A measured deployment (Mia, 28.10-16.11.2026): 78 of 109 traced recalls took the rerank-off order, and the journal
could not say why: 52 were messages over RERANK_MAX_QUERY_CHARS, about 23 short ones stayed unexplained, because the
outcome of the call (Rescorer.last_status) lived only in a DEBUG log line. With experience_trace on, each prefetch
receipt now carries one ``rerank`` stage: no facts, and a status without free text (answered, skipped_single,
skipped_long, skipped_secret, skipped_busy, not_called, paused, unavailable, failed).
"""
import json
import urllib.error

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider

QUESTION = "Did Boris Petrov say anything about the crane?"
STATUSES = {"answered", "skipped_single", "skipped_long", "skipped_secret", "skipped_busy", "not_called", "paused",
            "unavailable", "failed"}


class _Answer:
    def __init__(self, body):
        self.body, self.at = body, 0

    def read(self, n=-1):
        piece = self.body[self.at:] if n is None or n < 0 else self.body[self.at:self.at + n]
        self.at += len(piece)
        return piece

    read1 = read

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Endpoint:
    """The cross-encoder: answers in the order sent, or fails with HTTP 500 while ``broken``."""

    def __init__(self):
        self.calls, self.broken = 0, False

    def open(self, request, timeout=None):
        self.calls += 1
        if self.broken:
            raise urllib.error.HTTPError(request.full_url, 500, "server error", {}, None)
        n = len(json.loads(request.data)["documents"])
        return _Answer(json.dumps({"results": [{"index": i, "relevance_score": float(n - i)}
                                               for i in range(n)]}).encode())


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from hermes_plugin_holographic import rerank
    p = HolographicMemoryProvider({"db_path": str(tmp_path / "facts.db"), "experience": "on", "experience_trace": "on",
                                   "rerank": "on", "rerank_url": "https://rerank.example.net/v1/rerank",
                                   "rerank_model": "stub-reranker"})
    p.initialize(session_id="rerank-outcome")
    endpoint = _Endpoint()
    monkeypatch.setattr(rerank, "_OPENER", endpoint)
    yield p, endpoint
    p.shutdown()


def add(p, content):
    return json.loads(p.handle_tool_call("fact_store", {"action": "add", "content": content}))["fact_id"]


def rerank_stage(p, query, session):
    """The ``rerank`` stage of the receipt this prefetch wrote (the turn must recall something to have one)."""
    block = p.prefetch(query, session_id=session)
    assert "Memory attempt:" in block, block
    aid = block.rsplit("Memory attempt: ", 1)[1].strip()
    report = json.loads(p.handle_tool_call("fact_experience", {"action": "explain", "attempt_id": aid}))
    assert report["status"] == "ok", report
    stages = [s for s in report["stages"] if s["stage"] == "rerank"]
    assert len(stages) == 1, report["stages"]
    return stages[0]


def boris(p):
    add(p, "Boris Petrov maintains the harbour crane schedule")
    add(p, "Boris Petrov said the crane needs a new cable")


def test_an_answered_call_is_recorded(setup):
    p, endpoint = setup
    boris(p)
    stage = rerank_stage(p, QUESTION, "s1")
    assert endpoint.calls == 1
    assert stage["status"] == "answered"
    assert stage["facts"] == []


def test_a_long_message_records_that_it_was_not_sent(setup):
    p, endpoint = setup
    boris(p)
    long_question = QUESTION + " " + "Please keep the answer short and plain. " * 15
    assert len(long_question) > 500
    stage = rerank_stage(p, long_question, "s1")
    assert endpoint.calls == 0
    assert stage["status"] == "skipped_long"


def test_one_candidate_records_the_trivial_rank(setup):
    p, endpoint = setup
    add(p, "Boris Petrov maintains the harbour crane schedule")
    stage = rerank_stage(p, QUESTION, "s1")
    assert endpoint.calls == 0
    assert stage["status"] == "skipped_single"


def test_a_failure_then_the_pause_are_recorded_without_the_error_text(setup):
    p, endpoint = setup
    boris(p)
    endpoint.broken = True
    failed = rerank_stage(p, QUESTION, "s1")
    assert endpoint.calls == 1
    assert failed["status"] == "failed"
    # another session: the same facts are not held back by prefetch_repeat_turns
    paused = rerank_stage(p, QUESTION, "s2")
    assert endpoint.calls == 1  # paused: no second request
    assert paused["status"] == "paused"
    for stage in (failed, paused):
        assert set(stage) <= {"stage", "facts", "truncated", "status"}
        assert "500" not in json.dumps(stage) and "HTTP" not in json.dumps(stage)


def test_a_recall_with_nothing_sendable_does_not_repeat_the_previous_outcome(setup):
    p, endpoint = setup
    boris(p)
    assert rerank_stage(p, QUESTION, "s1")["status"] == "answered"
    # recall may show this fact, but its text looks secret: it never goes to the endpoint, so no call is made
    add(p, "The staging gateway header is Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456")
    stage = rerank_stage(p, "What is the staging gateway header?", "s2")
    assert endpoint.calls == 1
    assert stage["status"] == "not_called"


def test_every_status_is_one_of_the_documented_words(setup):
    p, _ = setup
    boris(p)
    assert rerank_stage(p, QUESTION, "s1")["status"] in STATUSES
