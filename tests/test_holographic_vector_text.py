"""Issue #95: what a fact's vector is made from (embeddings.EMBED_TEXT, embeddings.vector_text).

By default a vector is made from the fact's content alone, as in every release before: the keywords in its tags never
reach the endpoint and the stored hash is the content's, so an upgrade makes no store's vectors again. A measured arm
of tools/provenance_probe.py switches to "content+keywords": then the keywords — never the person's own tags — are
sent on a line after the content, the hash follows that text everywhere it is checked (the index, the store, the
scores recall keeps), a keywords change makes the vector again, a fact without keywords keeps its vector, and keywords
that may not leave the machine leave the vector to the content alone instead of to no vector at all."""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider, embeddings  # noqa: E402

DIM = 64
_WORDS: dict[str, int] = {}  # word -> its own dimension: no two words collide
DACHA = "Anton's grandmother spends the summer at the dacha near Tver."
JOINED = DACHA + "\nkeywords: country house, summer, granny"


def fake_vector(text: str) -> list[float]:
    """A unit bag of words: texts that share words are close, texts that share none are orthogonal."""
    v = np.zeros(DIM)
    for token in "".join(c if c.isalnum() else " " for c in text.lower()).split():
        v[_WORDS.setdefault(token, len(_WORDS) % DIM)] += 1.0
    return [float(x) for x in v / (np.linalg.norm(v) or 1.0)]


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def read(self, n=-1):
        piece, self.body = (self.body, b"") if n is None or n < 0 else (self.body[:n], self.body[n:])
        return piece

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeEndpoint:
    """Stands in for embeddings._OPENER: keeps every text it was sent and answers with fake_vector."""

    def __init__(self):
        self.inputs: list[str] = []

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.inputs.extend(body["input"])
        data = [{"object": "embedding", "index": i, "embedding": fake_vector(t)} for i, t in enumerate(body["input"])]
        return _Response(json.dumps({"data": data, "model": body["model"]}).encode())


@pytest.fixture
def endpoint(monkeypatch):
    fake = FakeEndpoint()
    monkeypatch.setattr(embeddings, "_OPENER", fake)
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    return fake


@pytest.fixture
def provider(tmp_path, endpoint):
    p = HolographicMemoryProvider(config={
        "db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64, "embeddings": "on",
        "embedding_url": "https://embeddings.example.net/v1", "embedding_model": "fake-1",
        "embedding_key_env": "TEST_EMBEDDINGS_KEY"})
    p.initialize(session_id="s1")
    yield p
    p.shutdown()


@pytest.fixture
def keywords_on(monkeypatch):
    monkeypatch.setattr(embeddings, "EMBED_TEXT", "content+keywords")


def _add(p, content, keywords=None, tags=None) -> int:
    args = {"action": "add", "content": content, "origin": "said"}
    if keywords is not None:
        args["keywords"] = keywords
    if tags is not None:
        args["tags"] = tags
    result = json.loads(p.handle_tool_call("fact_store", args))
    assert "error" not in result, result
    return int(result["fact_id"])


def _catch_up(p) -> None:
    index = p._retriever.vector_index(p._embedding.tag)
    for _ in range(10):
        if not index.pending(embeddings.BATCH, allowed=lambda _t: True):
            return
        p._catch_up_vectors()
    raise AssertionError("vectors never caught up")


def _stored_hash(p, fact_id) -> str | None:
    rows = p._store.read("SELECT text_hash FROM fact_embeddings WHERE fact_id = ?", (fact_id,))
    return rows[0]["text_hash"] if rows else None


def _pending_ids(p) -> list[int]:
    index = p._retriever.vector_index(p._embedding.tag)
    return [fact_id for fact_id, _, _ in index.pending(100, allowed=lambda _t: True)]


def test_the_shipped_default_is_the_content_alone():
    assert embeddings.EMBED_TEXT == "content"
    assert embeddings.vector_text(DACHA, "family\nkeywords: country house, summer") == DACHA


def test_by_default_keywords_and_tags_never_reach_the_endpoint(provider, endpoint):
    """Default unchanged: the text sent is the content, the hash is the content's (an upgraded store keeps every
    vector it has)."""
    fid = _add(provider, DACHA, keywords=["country house", "summer", "granny"], tags="family")
    _catch_up(provider)
    assert endpoint.inputs == [DACHA]
    assert _stored_hash(provider, fid) == embeddings.text_hash(DACHA)


def test_by_default_a_keywords_change_makes_no_new_vector(provider, endpoint):
    fid = _add(provider, DACHA, keywords=["country house"])
    _catch_up(provider)
    provider.handle_tool_call("fact_store", {"action": "update", "fact_id": fid, "keywords": ["granny", "summer"]})
    assert _pending_ids(provider) == []
    _catch_up(provider)
    assert endpoint.inputs == [DACHA]


def test_switched_on_the_keywords_join_the_content_and_the_tags_stay_home(provider, endpoint, keywords_on):
    fid = _add(provider, DACHA, keywords=["country house", "summer", "granny"], tags="family-private")
    _catch_up(provider)
    assert endpoint.inputs == [JOINED]
    assert not any("family-private" in text for text in endpoint.inputs)
    assert _stored_hash(provider, fid) == embeddings.text_hash(JOINED)


def test_switched_on_a_keywords_change_makes_the_vector_again(provider, endpoint, keywords_on):
    fid = _add(provider, DACHA, keywords=["country house"])
    _catch_up(provider)
    provider.handle_tool_call("fact_store", {"action": "update", "fact_id": fid, "keywords": ["granny"]})
    assert _pending_ids(provider) == [fid]
    _catch_up(provider)
    assert endpoint.inputs[-1] == DACHA + "\nkeywords: granny"
    assert _stored_hash(provider, fid) == embeddings.text_hash(DACHA + "\nkeywords: granny")


def test_switching_on_remakes_only_the_facts_with_keywords(provider, endpoint, monkeypatch):
    """The index's cached hashes follow the switch: a fact with keywords is pending after it, one without is not."""
    plain = _add(provider, "The car is parked on level two.")
    with_kw = _add(provider, DACHA, keywords=["granny"])
    _catch_up(provider)
    assert _pending_ids(provider) == []
    monkeypatch.setattr(embeddings, "EMBED_TEXT", "content+keywords")
    assert _pending_ids(provider) == [with_kw]
    _catch_up(provider)
    assert _stored_hash(provider, plain) == embeddings.text_hash("The car is parked on level two.")


def test_keywords_that_look_secret_leave_the_vector_to_the_content(provider, endpoint, keywords_on):
    """Keywords another program wrote (the tool screens its own) that hold a key: the content is sent alone, its
    vector counts as made, and the fact does not stay pending — sent again after every turn."""
    fid = provider._store.add_fact(DACHA, tags="keywords: granny, sk-proj-abcdef1234567890abcd")
    _catch_up(provider)
    assert endpoint.inputs == [DACHA]
    assert not any("sk-proj" in text for text in endpoint.inputs)
    assert _stored_hash(provider, fid) == embeddings.text_hash(DACHA)
    assert _pending_ids(provider) == []
    counts = provider._retriever.vector_index(provider._embedding.tag).counts(embeddings._sendable)
    assert counts == {"facts_with_vector": 1, "waiting": 0, "never_sent": 0}


def test_switched_on_the_store_keeps_a_vector_only_for_the_joined_text(provider, keywords_on):
    """put_vectors skips a vector whose text is no longer the fact's: switched on, that text is the joined one."""
    fid = _add(provider, DACHA, keywords=["country house", "summer", "granny"])
    blob = embeddings.to_blob(fake_vector(DACHA))
    tag = provider._embedding.tag
    assert provider._store.put_vectors(tag, [(fid, embeddings.text_hash(DACHA), blob)]) == 0
    assert provider._store.put_vectors(tag, [(fid, embeddings.text_hash(JOINED), blob)]) == 1


def test_switched_on_recall_keeps_the_score_of_a_fact_with_keywords(provider, keywords_on):
    """The retriever reads a fact again before it uses its score (Meaning.fits): with the switch on it must compare
    the joined text, or every fact with keywords quietly loses its meaning score."""
    fid = _add(provider, DACHA, keywords=["country house", "summer", "granny"])
    other = _add(provider, "The car is parked on level two.", keywords=["garage"])
    _catch_up(provider)
    retriever, tag = provider._retriever, provider._embedding.tag
    query = np.array(fake_vector("granny summer country house dacha"), dtype=np.float32)

    meaning = retriever.meaning(query, tag, 1.0)
    found = retriever.by_meaning(meaning, margin=1.0, min_score=0.1, limit=5)
    assert [f["fact_id"] for f in found] == [fid]

    # the blended score of facts the words found: with weight 1.0 it is the cosine itself, never the neutral median
    scored = {f["fact_id"]: (f["score"], f["trust_score"])
              for f in retriever.search("dacha car", limit=5, meaning=meaning)}
    for fact_id in (fid, other):
        score, trust = scored[fact_id]
        assert score == pytest.approx(meaning.sims[fact_id] * trust, abs=1e-6)

    words = retriever.search("car parked", limit=5)
    beyond = retriever.beyond_words(meaning, "car parked", words, k=3, beat=0.0, z_min=-10.0)
    assert [f["fact_id"] for f in beyond] == [fid]


def test_switched_on_search_beyond_words_measures_the_bar_on_the_joined_text(provider, keywords_on):
    """fact_store search adds a fact by meaning only when it is closer than the best fact the words found. That bar
    is read from the words' own facts: compared on the content alone, they fall to the neutral median and a weaker
    fact gets in."""
    near = _add(provider, "The car is parked on level two.", keywords=["garage"])
    _add(provider, DACHA, keywords=["country house", "summer", "granny"])
    _add(provider, "The kettle is in the cupboard.", keywords=["kitchen"])
    _add(provider, "Lena plays the violin on Sundays.", keywords=["music"])
    _catch_up(provider)
    retriever, tag = provider._retriever, provider._embedding.tag
    meaning = retriever.meaning(np.array(fake_vector("car parked level dacha"), dtype=np.float32), tag, 0.3)
    words = retriever.search("car parked", limit=5)
    assert [f["fact_id"] for f in words] == [near]
    assert retriever.beyond_words(meaning, "car parked", words, k=3, beat=0.0, z_min=-10.0) == []
