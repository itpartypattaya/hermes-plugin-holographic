"""Review 2026-11-17, H8: one fact whose text the embeddings endpoint refuses as input must not stop vectors for all.

Requirement: a model with a small context refuses a long text (OpenAI answers 400 with {"error": {...}}, Hugging Face
TEI 413). Before the fix every turn sent the same batch with that fact in it, the endpoint was paused, longer each time
(up to 900 s), and neither the other facts nor the person's message got a vector: search by meaning was off for good.
Now the other facts get their vectors, the message's vector is made, the refused fact is set aside (not sent again
every turn) until its text changes, and stats say how many facts are set aside. A bad key (401/403), a limit (429), a
server that is down (5xx) or a request every text gets refused for (a setting the model does not take) still pause
the endpoint for everyone, as before.

A real HTTP server on 127.0.0.1 stands in for the model."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

np = pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider, embeddings  # noqa: E402

from test_holographic_v0140_embeddings import BACKUP, FILLERS, PARAPHRASE, fake_vector  # noqa: E402

LIMIT = 200  # characters this "model" takes per input
LONG = ("The long note about the garden: " + "roses tulips daisies and a lot of other flowers grow there " * 6).strip()
SHORT = "The garden has roses."

# What servers answer to a text too long for the model, word for word in shape
REFUSALS = {
    400: {"error": {"message": "This model's maximum context length is 64 tokens, however you requested 120 tokens "
                               "(120 in your prompt; 0 for the completion). Please reduce your prompt.",
                    "type": "invalid_request_error", "param": None, "code": None}},
    413: {"error": "Input validation error: `inputs` must have less than 64 tokens. Given: 120",
          "error_type": "Validation"},
}


class _Model(BaseHTTPRequestHandler):
    seen: list = []  # the inputs of every request
    refuse_long = 400  # the status a request with a too-long input gets
    every = None  # a status every request gets (a key, a limit, a server down, a setting refused), or None
    lock = threading.Lock()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with type(self).lock:
            type(self).seen.append(list(body["input"]))
        if type(self).every:
            code = type(self).every
            self._answer(code, REFUSALS.get(code) or {"error": {"message": "nope", "type": "error", "code": code}})
        elif any(len(t) > LIMIT for t in body["input"]):
            self._answer(type(self).refuse_long, REFUSALS[type(self).refuse_long])
        else:
            self._answer(200, {"object": "list", "model": body["model"],
                               "data": [{"object": "embedding", "index": i, "embedding": fake_vector(t)}
                                        for i, t in enumerate(body["input"])]})

    def _answer(self, code: int, payload: dict):
        out = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def model():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Model.seen, _Model.refuse_long, _Model.every = [], 400, None
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _provider(tmp_path, url, **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64,
                                          "embeddings": "on", "embedding_url": url, "embedding_model": "small",
                                          "embedding_timeout": 5, **config})
    p.initialize(session_id="s1")
    return p


def _fill(p):
    ids = [p._store.add_fact(f"The {w} note number {i} is kept on the {w} shelf") for i, w in enumerate(FILLERS)]
    backup = p._store.add_fact(BACKUP)
    long_id = p._store.add_fact(LONG)  # the newest: the first batch of every turn carries it
    return ids + [backup], backup, long_id


def _with_vector(p) -> set[int]:
    return {r[0] for r in p._store.read("SELECT fact_id FROM fact_embeddings WHERE LENGTH(vector) > 0")}


def _stats(p) -> dict:
    return json.loads(p.handle_tool_call("fact_store", {"action": "stats"}))["search_by_meaning"]


def _sent(text: str) -> int:
    return sum(text in inputs for inputs in _Model.seen)


@pytest.mark.parametrize("code", [400, 413])
def test_a_fact_the_endpoint_refuses_does_not_stop_vectors_for_the_others(tmp_path, model, code):
    _Model.refuse_long = code
    p = _provider(tmp_path, model)
    good, backup, long_id = _fill(p)
    p.sync_turn("hello", "hi")
    assert p._embedder.status() == "ready"  # not paused: the endpoint works, one text is the problem
    assert _with_vector(p) == set(good)  # every other fact has its vector after one turn
    assert long_id not in _with_vector(p)
    block = p.prefetch(PARAPHRASE)  # the message's vector is made: a paraphrase with no shared word finds the fact
    assert any(f"#{backup} " in line and line.endswith("(by meaning)") for line in block.splitlines())
    stats = _stats(p)
    assert stats["status"] == "ready"
    assert (stats["facts_with_vector"], stats["waiting"], stats["refused_by_endpoint"]) == (len(good), 0, 1)

    sent = _sent(LONG)
    p.sync_turn("and now?", "still here")  # set aside: not sent again with every turn
    assert _sent(LONG) == sent
    assert _stats(p)["refused_by_endpoint"] == 1

    p._store.update_fact(long_id, content=SHORT)  # its text changed: sent again, and taken
    p.sync_turn("one more", "ok")
    assert long_id in _with_vector(p)
    stats = _stats(p)
    assert (stats["facts_with_vector"], stats["waiting"], stats["refused_by_endpoint"]) == (len(good) + 1, 0, 0)
    p.shutdown()


def test_a_refused_fact_stays_aside_after_a_restart(tmp_path, model):
    p = _provider(tmp_path, model)
    good, _, long_id = _fill(p)
    p.sync_turn("hello", "hi")
    assert _with_vector(p) == set(good)
    p.shutdown()
    p = _provider(tmp_path, model)  # a new process: what was refused is kept in the file, not in memory
    sent = _sent(LONG)
    p.sync_turn("hello again", "hi")
    assert _sent(LONG) == sent and p._embedder.status() == "ready"
    assert _stats(p)["refused_by_endpoint"] == 1
    p.shutdown()


@pytest.mark.parametrize("code", [401, 403, 429, 500, 503])
def test_a_key_a_limit_or_a_server_down_still_pauses_for_everyone(tmp_path, model, code):
    _Model.every = code
    p = _provider(tmp_path, model)
    _fill(p)
    p.sync_turn("hello", "hi")
    assert p._embedder.status().startswith(f"paused after an error (HTTP {code})")
    assert len(_Model.seen) == 1  # one request, then the pause: no splitting, no hammering
    assert p._store.read("SELECT COUNT(*) FROM fact_embeddings")[0][0] == 0  # nothing set aside
    assert _stats(p)["refused_by_endpoint"] == 0
    p.shutdown()


def test_every_text_refused_is_the_endpoint_not_the_facts(tmp_path, model):
    # a model that cannot shorten its vectors answers 400 to `dimensions` whatever the text (reported on 0.14.1)
    _Model.every = 400
    p = _provider(tmp_path, model, embedding_dimensions=64)
    _fill(p)
    p.sync_turn("hello", "hi")
    status = p._embedder.status()
    assert status.startswith("paused after an error (HTTP 400") and "embedding_dimensions" in status
    assert p._store.read("SELECT COUNT(*) FROM fact_embeddings")[0][0] == 0  # no fact blamed for it
    stats = _stats(p)
    assert stats["refused_by_endpoint"] == 0 and stats["waiting"] == len(FILLERS) + 2
    p.shutdown()
