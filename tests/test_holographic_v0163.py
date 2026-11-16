"""0.16.3: fact vectors after a turn on a slow endpoint (issue #5 by sprd-tmu).

A 4B embedding model on a local NPU server took 7.7 s for a batch of 32 facts; the after-turn budget left 7 s. Every
batch was cut, every cut paused the endpoint (60 s doubling to 15 min, recall by meaning off with it), and the same 32
facts went out again after each pause: the store never got a vector. Here a fake endpoint takes a fixed time per text,
and the budget is scaled down so a whole batch does not fit."""
from __future__ import annotations

import json
import threading
import time
import urllib.error

import pytest

np = pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider, embeddings  # noqa: E402
from test_holographic_v0140_embeddings import BACKUP, FILLERS, ON, FakeEndpoint  # noqa: E402

import hermes_plugin_holographic as plugin  # noqa: E402


class SlowEndpoint(FakeEndpoint):
    """FakeEndpoint that takes *per_text* seconds for every text it is sent, as a local model does."""

    def __init__(self, per_text: float):
        super().__init__()
        self.per_text = per_text

    def open(self, request, timeout=None):
        time.sleep(self.per_text * len(json.loads(request.data)["input"]))
        return super().open(request, timeout)


def _provider(tmp_path, **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64, **ON, **config})
    p.initialize(session_id="s1")
    return p


def _fill(p) -> int:
    for i, w in enumerate(FILLERS):
        p._store.add_fact(f"The {w} note number {i} is kept on the {w} shelf")
    p._store.add_fact(BACKUP)
    return len(FILLERS) + 1


def _with_vector(p) -> int:
    return p._store.read("SELECT COUNT(*) FROM fact_embeddings")[0][0]


def _stats(p) -> dict:
    return json.loads(p.handle_tool_call("fact_store", {"action": "stats"}))["search_by_meaning"]


def _embedder(**config) -> embeddings.Embedder:
    return embeddings.Embedder(embeddings.Settings({**ON, **config}), environ={"TEST_EMBEDDINGS_KEY": "sk-test-key"})


@pytest.fixture
def slow(monkeypatch):
    fake = SlowEndpoint(per_text=0.06)  # 27 facts: 1.6 s, against 1.1 s left
    monkeypatch.setattr(embeddings, "_OPENER", fake)
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(plugin, "_CATCH_UP_SECONDS", 1.2)
    monkeypatch.setattr(plugin, "_CATCH_UP_RESERVE", 0.1)
    return fake


# -- the report -----------------------------------------------------------------------------------------------------

def test_issue_5_a_slow_server_fills_the_store_and_is_never_paused(tmp_path, slow):
    p = _provider(tmp_path)
    total = _fill(p)
    turns = 0
    while _with_vector(p) < total and turns < 10:
        p.sync_turn("hello", "hi")
        turns += 1
        assert p._embedder.status() == "ready"  # 0.16.2: paused after the first batch, and recall by meaning with it
    assert _with_vector(p) == total and turns <= 4
    sizes = [len(r["body"]["input"]) for r in slow.requests]
    assert sizes[0] == embeddings._PROBE_BATCH  # an endpoint of unknown speed gets a small batch first
    assert "not done" not in p._embedder.last_batch
    assert _stats(p)["waiting"] == 0 and " in " in _stats(p)["after_turn"]["last_batch"]
    p.shutdown()


def test_a_batch_cut_short_shrinks_the_next_one_without_a_pause(tmp_path, slow):
    p = _provider(tmp_path)
    total = _fill(p)
    p._embedder._fact_rate = 1e-6  # measured fast once: the whole batch goes out, and does not fit
    p.sync_turn("hello", "hi")
    assert _with_vector(p) == 0 and p._embedder.status() == "ready"
    assert "not done" in _stats(p)["after_turn"]["last_batch"]
    time.sleep(0.7)  # the cut request finishes on its own thread
    turns = 1
    while _with_vector(p) < total and turns < 10:
        p.sync_turn("hello", "hi")
        turns += 1
        assert p._embedder.status() == "ready"
    sizes = [len(r["body"]["input"]) for r in slow.requests]
    assert _with_vector(p) == total and sizes[0] == total and max(sizes[1:]) < total
    p.shutdown()


def test_recall_by_meaning_goes_on_while_the_store_fills(tmp_path, slow):
    p = _provider(tmp_path)
    _fill(p)
    p.sync_turn("hello", "hi")
    assert _with_vector(p) > 0
    slow.per_text = 0.0
    before = len(slow.requests)
    assert p._meaning_of("which machine copies our data while we sleep?", before_turn=True) is not None
    assert len(slow.requests) == before + 1
    p.shutdown()


# -- what still is a failure ------------------------------------------------------------------------------------------

def test_a_single_fact_that_runs_out_of_time_is_still_a_failure(tmp_path, slow):
    slow.per_text = 2.0
    p = _provider(tmp_path)
    p._store.add_fact(BACKUP)
    p.sync_turn("hello", "hi")
    assert _with_vector(p) == 0
    assert p._embedder.status().startswith("paused after an error (no answer (timeout))")
    p.shutdown()


def test_a_server_that_refuses_is_paused_whatever_the_batch(tmp_path, slow):
    slow.per_text = 0.0
    slow.fail = urllib.error.URLError(ConnectionRefusedError())
    p = _provider(tmp_path)
    _fill(p)
    p.sync_turn("hello", "hi")
    assert p._embedder.status().startswith("paused after an error (cannot reach the endpoint")
    assert len(slow.requests) == 1
    p.shutdown()


def test_batches_cut_short_in_a_row_pause_the_endpoint(tmp_path, slow):
    slow.per_text = 0.0
    slow.fail = TimeoutError()
    p = _provider(tmp_path)
    _fill(p)
    for _ in range(embeddings._MAX_CUTS):  # the batch shrinks to one fact, and one fact cut short is a failure
        p.sync_turn("hello", "hi")
        if p._embedder.status() != "ready":
            break
    assert p._embedder.status().startswith("paused after an error (no answer (timeout")
    p._embedder._recovered()
    for _ in range(embeddings._MAX_CUTS - 1):  # batches that stay large (as if measured fast in between)
        p._embedder._fact_rate = 1e-6
        p.sync_turn("hello", "hi")
        assert p._embedder.status() == "ready"
    p._embedder._fact_rate = 1e-6
    p.sync_turn("hello", "hi")
    assert p._embedder.status().startswith(f"paused after an error (no answer (timeout, {embeddings._MAX_CUTS} batches")
    p.shutdown()


def test_an_error_that_comes_after_a_cut_still_pauses(monkeypatch):
    """Review 0.16.3 (Codex): a 503 that came after the caller stopped waiting was dropped, and the endpoint stayed
    ready while every batch was cut."""
    class LateError(FakeEndpoint):
        def open(self, request, timeout=None):
            time.sleep(0.7)
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)

    monkeypatch.setattr(embeddings, "_OPENER", LateError())
    e = _embedder()
    assert e.embed_facts(["one", "two"], 0.5) is None and e.status() == "ready"
    time.sleep(0.4)  # the 503 arrives
    assert e.embed_facts(["one", "two"], 0.5) is None
    assert e.status().startswith("paused after an error (HTTP 503)")


# -- what runs out of time ------------------------------------------------------------------------------------------

def test_a_read_that_times_out_cuts_the_batch_without_a_pause(tmp_path, slow):
    slow.per_text = 0.0
    slow.fail = TimeoutError()
    p = _provider(tmp_path)
    _fill(p)
    p.sync_turn("hello", "hi")
    assert p._embedder.status() == "ready" and "not done" in p._embedder.last_batch
    p.shutdown()


def test_a_timeout_urllib_wraps_cuts_the_batch_too(tmp_path, slow):
    """Review 0.16.3 (Codex): waiting for the headers times out as URLError(TimeoutError); the draft paused the
    endpoint and kept its batch when urllib's own timeout came first."""
    slow.per_text = 0.0
    slow.fail = urllib.error.URLError(TimeoutError("timed out"))
    p = _provider(tmp_path)
    _fill(p)
    p.sync_turn("hello", "hi")
    assert p._embedder.status() == "ready" and p._embedder._fact_rate is not None
    assert "not done" in p._embedder.last_batch
    p.shutdown()


def test_a_query_waits_for_a_batch_cut_short_a_moment_ago(monkeypatch):
    release = threading.Event()

    class Held(FakeEndpoint):
        def open(self, request, timeout=None):
            if len(json.loads(request.data)["input"]) > 1:
                release.wait(5)
            return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", Held())
    e = _embedder(embedding_timeout=3)
    assert e.embed_facts(["a backup fact", "a birthday fact"], 0.6) is None  # cut, left running
    assert e.status() == "ready" and e._abandoned
    threading.Timer(0.3, release.set).start()
    t = time.monotonic()
    assert e.query("which machine copies our data while we sleep?") is not None  # 0.16.2: failed at once, paused
    assert time.monotonic() - t < 1.5 and e.status() == "ready"


def test_a_query_behind_our_batch_still_running_is_no_failure(monkeypatch):
    release = threading.Event()

    class Held(FakeEndpoint):
        def open(self, request, timeout=None):
            release.wait(5)
            return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", Held())
    e = _embedder(embedding_timeout=1)
    try:
        assert e.embed_facts(["one", "two"], 0.5) is None
        t = time.monotonic()
        assert e.query("which machine copies our data?") is None
        assert 0.4 < time.monotonic() - t < 0.9  # half of embedding_timeout
        assert e.status() == "ready"  # our own batch held the endpoint: not its failure
        assert e.calls == 1  # and nothing more was sent
    finally:
        release.set()


def test_a_message_behind_our_batch_on_a_one_slot_server_is_no_failure(monkeypatch):
    """Review 0.16.3 (Codex): a local server serving one request at a time made a message's vector wait behind a fact
    batch; it ran out of embedding_timeout and paused the endpoint."""
    slot = threading.Lock()

    class OneSlot(FakeEndpoint):
        def open(self, request, timeout=None):
            with slot:
                time.sleep(0.2 * len(json.loads(request.data)["input"]))
                return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", OneSlot())
    e = _embedder(embedding_url="http://127.0.0.1:8731/v1", embedding_timeout=0.5)
    batch = threading.Thread(target=e.embed_facts, args=(["one", "two", "three", "four", "five"], 5.0))
    batch.start()
    time.sleep(0.1)
    assert e.query("which machine copies our data while we sleep?") is None  # waited behind the batch
    assert e.status() == "ready"
    batch.join(5)
    time.sleep(0.3)
    assert e.query("which machine copies our data while we sleep?") is not None


# -- sizing -----------------------------------------------------------------------------------------------------------

def test_a_fast_endpoint_still_gets_whole_batches(tmp_path, monkeypatch):
    fake = FakeEndpoint()
    monkeypatch.setattr(embeddings, "_OPENER", fake)
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    p = _provider(tmp_path)
    total = _fill(p)
    p.sync_turn("hello", "hi")
    assert _with_vector(p) == total
    assert [len(r["body"]["input"]) for r in fake.requests] == [embeddings._PROBE_BATCH, total - embeddings._PROBE_BATCH]
    assert p._embedder.fact_batch([200] * 40, 7.0) == embeddings.BATCH  # measured fast: the batch stays whole
    p.shutdown()


def test_the_batch_follows_the_measured_speed():
    e = _embedder()
    assert e.fact_batch([100] * 40, 7.0) == embeddings._PROBE_BATCH  # nothing measured yet: a small batch
    e._fact_rate = 0.001  # seconds per character
    assert e.fact_batch([100] * 32, 2.0) == 10  # half of 2 s: 1,000 characters
    assert e.fact_batch([5000, 100], 2.0) == 1  # a probe (the turn's first request): one, to try
    assert e.fact_batch([100] * 32, 0.5) == 2
    # review 0.16.3 (Codex): what would not fit is not sent (a fact sent into the last half second paused a working
    # endpoint) ...
    assert e.fact_batch([5000, 100], 2.0, probe=False) == 0
    e._fact_rate = 1.0 / 570  # one fact: 1 s
    assert e.fact_batch([570], 0.5, probe=False) == 0
    # ... but one fact that fits goes out even when half the time left holds none (round 2: 1 fact a turn on a local
    # server whose one fact took longer than half of embedding_timeout)
    assert e.fact_batch([570, 570], 1.5, probe=False) == 1


def test_a_local_server_gets_batches_a_message_can_wait_behind():
    e = _embedder(embedding_url="http://127.0.0.1:8731/v1", embedding_timeout=2)
    e._fact_rate = 0.001
    assert e.fact_batch([100] * 32, 7.0) == 10  # half of embedding_timeout, not of the 7 s left
    hosted = _embedder(embedding_timeout=2)
    hosted._fact_rate = 0.001
    assert hosted.fact_batch([100] * 32, 7.0) == 32
    slow = _embedder(embedding_url="http://127.0.0.1:8731/v1", embedding_timeout=0.5)
    slow._fact_rate = 0.24 / 570
    assert slow.fact_batch([1000] * 5, 5.0, probe=False) == 1  # 0.42 s each, over the 0.25 s cap: still one


def test_a_wait_for_the_lock_makes_the_first_request_no_test(tmp_path, slow, monkeypatch):
    """Review 0.16.3 (round 2): a prefetch held the provider lock, the first request then had 0.6 s for a fact
    measured at 0.8 s, and its cut paused a working endpoint."""
    p = _provider(tmp_path, embedding_catch_up_seconds=2)
    monkeypatch.setattr(plugin, "_CATCH_UP_RESERVE", 1.0)
    p._store.add_fact(BACKUP)
    p._embedder._fact_rate = 0.8 / len(BACKUP)
    taken, release = threading.Event(), threading.Event()

    def hold():
        with p._op_lock:
            taken.set()
            release.wait(0.4)

    holder = threading.Thread(target=hold)
    holder.start()
    assert taken.wait(5)
    p._catch_up_vectors()  # as sync_turn does after the extraction
    holder.join(5)
    assert slow.requests == [] and p._embedder.status() == "ready"
    p.shutdown()


# -- round 2: what holds the endpoint, and for how long --------------------------------------------------------------

class _Never(FakeEndpoint):
    """Answers nothing until released."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def open(self, request, timeout=None):
        self.requests.append(json.loads(request.data))
        self.release.wait(10)
        raise OSError("released")


def test_a_fact_request_that_never_ends_is_ours_only_for_a_while(monkeypatch):
    """Review 0.16.3 (round 2): a fact worker that never ended kept every later request "busy" and the endpoint
    "ready" for good."""
    never = _Never()
    monkeypatch.setattr(embeddings, "_OPENER", never)
    e = _embedder(embedding_timeout=0.5)
    try:
        assert e.embed_facts(["one", "two", "three", "four"], 0.5) is None  # cut, left running
        assert e.query("which machine copies our data?") is None and e.status() == "ready"  # ours, a moment ago
        time.sleep(e._grace())
        assert e.query("which machine copies our data?") is None
        assert "earlier request is still waiting" in e.status()  # a stalled server now
        assert len(never.requests) == 1  # and never a second worker
    finally:
        never.release.set()


def test_a_batch_that_ended_just_before_the_message_gave_up_still_explains_its_wait(monkeypatch):
    """Review 0.16.3 (round 2): on a one-slot server a message waited behind a batch that ended just before the
    message's deadline; nothing of ours was running at that moment, and the endpoint was paused."""
    slot, held, asked = threading.Lock(), threading.Event(), threading.Event()

    class OneSlot(FakeEndpoint):
        # The order is set by events, not by sleeps a slow runner overran (macOS CI: the message took the slot
        # first): the batch holds the slot before the message is sent, and ends 0.4 s after the message reached the
        # server — inside the message's 0.5 s, which its own 0.15 s then overruns.
        def open(self, request, timeout=None):
            if len(json.loads(request.data)["input"]) == 1:
                asked.set()
            with slot:
                if len(json.loads(request.data)["input"]) > 1:
                    held.set()
                    asked.wait(5)
                    time.sleep(0.4)
                else:
                    time.sleep(0.15)
                return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", OneSlot())
    e = _embedder(embedding_url="http://127.0.0.1:8731/v1", embedding_timeout=0.5)
    batch = threading.Thread(target=e.embed_facts, args=(["one", "two"], 5.0))
    batch.start()
    assert held.wait(5)
    assert e.query("which machine copies our data while we sleep?") is None
    batch.join(5)
    assert e.status() == "ready"


def test_a_batch_a_socket_timeout_ended_still_explains_a_message_s_wait(monkeypatch):
    """Review 0.16.3 (round 3, Antigravity): a batch ended by urllib's own timeout (its thread over before the join)
    left no trace, and a message queued behind it on a one-slot server paused the endpoint."""
    slot, held, asked = threading.Lock(), threading.Event(), threading.Event()

    class OneSlot(FakeEndpoint):  # ordered by events, as in the test above
        def open(self, request, timeout=None):
            if len(json.loads(request.data)["input"]) == 1:
                asked.set()
            with slot:
                if len(json.loads(request.data)["input"]) > 1:
                    held.set()
                    asked.wait(5)
                    time.sleep(0.35)
                    raise TimeoutError("read timed out")
                time.sleep(0.2)
                return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", OneSlot())
    e = _embedder(embedding_url="http://127.0.0.1:8731/v1", embedding_timeout=0.5)
    batch = threading.Thread(target=e.embed_facts, args=(["one", "two"], 1.0))
    batch.start()
    assert held.wait(5)
    assert e.query("which machine copies our data while we sleep?") is None  # 0.35 s behind the batch, then 0.2 s
    batch.join(5)
    assert e.status() == "ready" and "not done" in e.last_batch


def test_the_status_tells_a_stalled_server_without_waiting_for_a_request(monkeypatch):
    """Review 0.16.3 (round 3, Antigravity): a request that never ended showed "ready" until the next request."""
    never = _Never()
    monkeypatch.setattr(embeddings, "_OPENER", never)
    e = _embedder(embedding_timeout=0.5)
    try:
        assert e.embed_facts(["one", "two"], 0.5) is None
        assert e.status() == "ready"  # ours, a moment ago
        time.sleep(e._grace() + 0.1)
        assert "earlier request is still waiting" in e.status()
        pause = e._pause
        assert "earlier request is still waiting" in e.status() and e._pause == pause  # counted once
    finally:
        never.release.set()


def test_an_answer_does_not_end_a_pause_that_came_after_it_was_sent(monkeypatch):
    """Review 0.16.3 (round 2): a batch still running ended well and cleared the pause a message's 503 had set
    meanwhile."""
    release = threading.Event()

    class Held(FakeEndpoint):
        def open(self, request, timeout=None):
            release.wait(5)
            return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", Held())
    e = _embedder()
    out = {}
    batch = threading.Thread(target=lambda: out.setdefault("v", e.embed_facts(["one", "two"], 3.0)))
    batch.start()
    time.sleep(0.1)
    e._failed("HTTP 503")  # what a message's request met meanwhile
    release.set()
    batch.join(5)
    assert out["v"] is not None  # the vectors are kept
    assert e.status().startswith("paused after an error (HTTP 503)")


def test_one_request_is_one_step_of_the_pause(monkeypatch):
    """Review 0.16.3 (round 2): a fact cut short paused the endpoint, and the 503 it brought later paused it again —
    120 s for one request."""
    release = threading.Event()

    class LateError(FakeEndpoint):
        def open(self, request, timeout=None):
            release.wait(5)
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)

    monkeypatch.setattr(embeddings, "_OPENER", LateError())
    e = _embedder()
    assert e.embed_facts(["one"], 0.5, decisive=True) is None
    assert e._pause == 60.0
    release.set()
    for given in list(e._abandoned):
        given.worker.join(5)
    e._paused_until = 0.0  # the pause is over
    with e._lock:
        assert e._reap_locked() is None  # already counted
    assert e._pause == 60.0


def test_late_errors_of_several_requests_pause_once_and_say_so(monkeypatch):
    e = _embedder()
    for code in (503, 429):
        box = {"error": urllib.error.HTTPError("https://x", code, "busy", {}, None), "ended": time.monotonic()}
        worker = threading.Thread(target=lambda: None)
        worker.start()
        worker.join()
        e._abandoned.append(embeddings._GivenUp(worker, box, fact=True, busy=False, counted=False))
    with e._lock:
        why, _, pause = e._reap_locked()
    assert why == "HTTP 503 (and 1 more)" and pause == 60.0 and e._abandoned == []


def test_a_batch_a_message_queued_with_is_not_measured(monkeypatch):
    """Review 0.16.3 (round 2): a message queued with a batch on a one-slot server made the batch look twice as slow,
    and the next batch held none."""
    release = threading.Event()

    class Held(FakeEndpoint):
        def open(self, request, timeout=None):
            if len(json.loads(request.data)["input"]) > 1:
                release.wait(5)
            return super().open(request, timeout)

    monkeypatch.setattr(embeddings, "_OPENER", Held())
    e = _embedder()
    e._fact_rate = 1e-4
    batch = threading.Thread(target=e.embed_facts, args=(["one fact", "two facts"], 3.0))
    batch.start()
    time.sleep(0.05)
    assert e.query("which machine copies our data while we sleep?") is not None  # sent meanwhile
    time.sleep(0.3)
    release.set()
    batch.join(5)
    assert e._fact_rate == 1e-4


# -- the setting ------------------------------------------------------------------------------------------------------

def test_the_after_turn_budget_is_a_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "_OPENER", FakeEndpoint())
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    assert embeddings.Settings({}).catch_up is None
    assert embeddings.Settings({"embedding_catch_up_seconds": "30"}).catch_up == 30.0
    assert embeddings.Settings({"embedding_catch_up_seconds": 999}).catch_up == 60.0
    assert embeddings.Settings({"embedding_catch_up_seconds": 0}).catch_up == 2.0
    assert embeddings.Settings({"embedding_catch_up_seconds": "soon"}).catch_up is None
    p = _provider(tmp_path)
    assert _stats(p)["after_turn"] == {"seconds": 8.0, "last_batch": "none yet"}
    p.shutdown()
    q = _provider(tmp_path, embedding_catch_up_seconds=40)
    assert q._catch_up_seconds() == 40.0 and _stats(q)["after_turn"]["seconds"] == 40.0
    q.shutdown()


def test_a_longer_budget_fills_a_slow_store_in_one_turn(tmp_path, slow):
    p = _provider(tmp_path, embedding_catch_up_seconds=4)
    total = _fill(p)
    p.sync_turn("hello", "hi")  # 27 facts: 1.6 s of 3.9
    assert _with_vector(p) == total
    p.shutdown()
