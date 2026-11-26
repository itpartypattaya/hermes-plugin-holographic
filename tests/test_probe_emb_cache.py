"""The probe's by-text embedding cache (--emb-cache): two runs of the probe that a paired gate compares must see one
vector for one text.

OpenAI's embeddings are not deterministic between calls: on 2026-11-26 two identical off arms on text-embedding-3-small
differed by 4 noise rows, 2 confusion rows and 1 hit (results/issue121/text-embedding-3-small/), the size of the effects
tools/mcnemar_gate.py judges. The stand-in endpoint here is a loopback server that answers every call with fresh random
vectors, as such an endpoint does at a smaller scale; no real endpoint is called.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import embeddings  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import provenance_probe as pp  # noqa: E402

TEXTS = ["The spare key hangs behind the blue flowerpot.", "Masha feeds the cat at seven.", "The car is grey."]


class _Endpoint(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.asked.append((body["model"], list(body["input"])))
        if self.server.fail:
            self.send_error(self.server.fail)
            return
        raw = json.dumps({"object": "list", "model": body["model"], "data": [
            {"object": "embedding", "index": i,
             "embedding": [0.0] * 8 if self.server.zeros else [random.gauss(0, 1) for _ in range(8)]}
            for i in range(len(body["input"]) + self.server.extra)]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def endpoint(monkeypatch):
    """A loopback /v1/embeddings whose vectors differ on every call; the probe's EMB_URL points at it."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    server.asked, server.fail, server.zeros, server.extra = [], 0, False, 0  # fail: an HTTP code
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # nothing of the machine's goes to the stand-in
    monkeypatch.setattr(pp, "EMB_URL", f"http://127.0.0.1:{server.server_address[1]}/v1")
    monkeypatch.setattr(pp, "EMB_MODEL", "probe-model")
    yield server
    server.shutdown()
    server.server_close()


def _arm(tmp_path, monkeypatch, name, cache=None, texts=TEXTS):
    """One run of the probe as main sets it up: the cache in front of the plugin's opener when given; the vectors the
    plugin gets for *texts*."""
    real = embeddings._OPENER
    if cache is not None:
        monkeypatch.setattr(embeddings, "_OPENER", pp.EmbCache(cache, real))
    p = pp.make_provider(tmp_path / f"{name}.db", "off", "on")
    try:
        return p._embedder.embed(list(texts))
    finally:
        p.shutdown()
        monkeypatch.setattr(embeddings, "_OPENER", real)


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_without_the_cache_two_arms_see_different_vectors(tmp_path, monkeypatch, endpoint):
    """The premise: the stand-in, like OpenAI, gives one text another vector on another call."""
    a = _arm(tmp_path, monkeypatch, "a")
    b = _arm(tmp_path, monkeypatch, "b")
    assert a is not None and b is not None
    assert not any(np.array_equal(x, y) for x, y in zip(a, b))


def test_two_runs_on_one_cache_file_see_one_vector_per_text(tmp_path, monkeypatch, endpoint):
    cache = tmp_path / "vectors.emb-cache.jsonl"
    a = _arm(tmp_path, monkeypatch, "a", cache)
    b = _arm(tmp_path, monkeypatch, "b", cache)
    assert a is not None and b is not None
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert endpoint.asked == [("probe-model", TEXTS)]  # the second run asked nothing


def test_a_rerun_appends_nothing(tmp_path, monkeypatch, endpoint):
    cache = tmp_path / "vectors.emb-cache.jsonl"
    for name in ("a", "b", "c"):
        _arm(tmp_path, monkeypatch, name, cache)
    rows = _lines(cache)
    assert [(r["model"], r["text"]) for r in rows] == [("probe-model", t) for t in TEXTS]
    assert all(len(r["vector"]) == 8 for r in rows)


def test_a_text_asked_twice_in_one_batch_is_one_entry(tmp_path, monkeypatch, endpoint):
    cache = tmp_path / "vectors.emb-cache.jsonl"
    a = _arm(tmp_path, monkeypatch, "a", cache, texts=[TEXTS[0], TEXTS[0]])
    assert np.array_equal(a[0], a[1])
    assert len(_lines(cache)) == 1


def test_another_model_is_another_entry(tmp_path, monkeypatch, endpoint):
    cache = tmp_path / "vectors.emb-cache.jsonl"
    _arm(tmp_path, monkeypatch, "a", cache)
    monkeypatch.setattr(pp, "EMB_MODEL", "another-model")
    _arm(tmp_path, monkeypatch, "b", cache)
    assert [m for m, _ in endpoint.asked] == ["probe-model", "another-model"]
    assert len(_lines(cache)) == 2 * len(TEXTS)


def test_a_torn_last_line_is_skipped_and_its_text_fetched_again(tmp_path, monkeypatch, endpoint):
    """A run that died mid-write leaves half a line: the next run keeps what is whole and asks for the rest."""
    cache = tmp_path / "vectors.emb-cache.jsonl"
    kept = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    whole = json.dumps({"model": "probe-model", "text": TEXTS[0], "vector": kept})
    torn = json.dumps({"model": "probe-model", "text": TEXTS[1], "vector": kept})[:40]
    cache.write_bytes((whole + "\n" + torn).encode("utf-8"))
    got = _arm(tmp_path, monkeypatch, "a", cache)
    assert got is not None
    unit = np.asarray(kept) / np.linalg.norm(kept)
    assert np.allclose(got[0], unit)  # the whole line was used
    assert endpoint.asked == [("probe-model", TEXTS[1:])]
    rows = _lines(cache)  # every line parses again: the torn piece is gone, nothing glued to it
    assert [r["text"] for r in rows] == TEXTS


def test_a_broken_line_inside_the_file_stops_the_run(tmp_path, endpoint):
    """Not what a crash leaves (that is only ever the last line): the file is damaged, and vectors measured on part of
    it would not be the ones the other arm saw."""
    cache = tmp_path / "vectors.emb-cache.jsonl"
    whole = json.dumps({"model": "probe-model", "text": TEXTS[0], "vector": [1.0] * 8})
    cache.write_text("{not json\n" + whole + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="line 1"):
        pp.EmbCache(cache, embeddings._OPENER)


def test_a_failed_request_caches_nothing(tmp_path, monkeypatch, endpoint):
    endpoint.fail = 500
    cache = tmp_path / "vectors.emb-cache.jsonl"
    assert _arm(tmp_path, monkeypatch, "a", cache) is None  # the plugin sees the failure, as without the cache
    assert not cache.exists() or cache.read_text(encoding="utf-8") == ""


def test_an_answer_the_plugin_refuses_is_not_kept(tmp_path, monkeypatch, endpoint):
    """All-zero vectors: the plugin refuses them, and the next run asks again instead of being handed them."""
    endpoint.zeros = True
    cache = tmp_path / "vectors.emb-cache.jsonl"
    assert _arm(tmp_path, monkeypatch, "a", cache) is None
    endpoint.zeros = False
    assert _arm(tmp_path, monkeypatch, "b", cache) is not None
    assert len(endpoint.asked) == 2 and len(_lines(cache)) == len(TEXTS)


def test_an_answer_with_more_vectors_than_texts_sent_fails_the_request(tmp_path, monkeypatch, endpoint):
    """Only the texts the file lacks are sent: an answer with a vector too many must not pass, to the plugin, for the
    whole batch, giving a cached text the endpoint's vector instead of the file's (Codex review)."""
    cache = tmp_path / "vectors.emb-cache.jsonl"
    _arm(tmp_path, monkeypatch, "a", cache, texts=TEXTS[:1])
    endpoint.extra = 1
    assert _arm(tmp_path, monkeypatch, "b", cache, texts=TEXTS[:2]) is None
    assert len(_lines(cache)) == 1


def test_the_plugins_check_of_the_endpoint_is_not_answered_from_the_file(tmp_path, monkeypatch, endpoint):
    """After a fact batch is refused as input the plugin embeds a plain text to tell refused texts from an endpoint
    that refuses everything. Answered from the file, an endpoint refusing everything would look like refused facts
    (Codex review)."""
    cache = tmp_path / "vectors.emb-cache.jsonl"
    _arm(tmp_path, monkeypatch, "a", cache, texts=[embeddings._PLAIN])
    endpoint.fail = 400
    real = embeddings._OPENER
    monkeypatch.setattr(embeddings, "_OPENER", pp.EmbCache(cache, real))
    p = pp.make_provider(tmp_path / "b.db", "off", "on")
    try:
        assert p._embedder.embed_facts(TEXTS[1:2], 10.0) is not embeddings.REFUSED
    finally:
        p.shutdown()
    assert endpoint.asked[-1] == ("probe-model", [embeddings._PLAIN])


def test_the_cache_folder_is_made(tmp_path, monkeypatch, endpoint):
    cache = tmp_path / "new" / "vectors.emb-cache.jsonl"
    assert _arm(tmp_path, monkeypatch, "a", cache) is not None
    assert len(_lines(cache)) == len(TEXTS)


def _main(tmp_path, *extra):
    return pp.main(["--set", "contrast_en", "--modes", "off", "--emb", "on", "--out", str(tmp_path / "out"), *extra])


def test_two_probe_runs_with_the_cache_give_the_same_rows(tmp_path, endpoint):
    """End to end through main: the second arm sends nothing, its rows are the first arm's, and the plugin's opener is
    its own again after each run."""
    real = embeddings._OPENER
    cache = tmp_path / "vectors.emb-cache.jsonl"
    first = _main(tmp_path, "--emb-cache", str(cache))
    assert embeddings._OPENER is real
    asked = len(endpoint.asked)
    assert asked > 0
    second = _main(tmp_path, "--emb-cache", str(cache))
    assert embeddings._OPENER is real
    assert len(endpoint.asked) == asked
    assert first == second


def test_without_the_flag_every_run_asks_the_endpoint(tmp_path, endpoint):
    """The default stays as it was: no cache, every run sends its own requests."""
    real = embeddings._OPENER
    _main(tmp_path)
    asked = len(endpoint.asked)
    _main(tmp_path)
    assert asked > 0 and len(endpoint.asked) == 2 * asked
    assert embeddings._OPENER is real


def test_the_cache_needs_embeddings_on(tmp_path):
    with pytest.raises(SystemExit, match="--emb-cache"):
        pp.main(["--set", "contrast_en", "--modes", "off", "--emb", "off",
                 "--emb-cache", str(tmp_path / "vectors.emb-cache.jsonl")])


def test_a_cache_file_the_repo_would_commit_is_refused(tmp_path, endpoint):
    """The file holds every text the run embedded: inside the repo it must be a name git ignores."""
    path = _REPO / "results" / "vectors-not-ignored.jsonl"
    with pytest.raises(SystemExit, match="--emb-cache"):
        pp.main(["--set", "contrast_en", "--modes", "off", "--emb", "on", "--out", str(tmp_path / "out"),
                 "--emb-cache", str(path)])
    assert not path.exists()


def test_a_cache_file_another_repo_would_commit_is_refused(tmp_path, endpoint):
    """Another checkout or worktree is a repo too (Codex review: the main checkout from a linked worktree)."""
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True, capture_output=True, timeout=30)
    path = other / "results" / "vectors.jsonl"
    with pytest.raises(SystemExit, match="--emb-cache"):
        _main(tmp_path, "--emb-cache", str(path))
    assert not path.exists()


def test_the_repo_ignores_the_cache_file_name():
    ignored = subprocess.run(["git", "check-ignore", "-q", "results/issue121/vectors.emb-cache.jsonl"], cwd=_REPO,
                             capture_output=True, timeout=10)
    assert ignored.returncode == 0
