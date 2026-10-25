"""Search by meaning: optional, online, off by default. An OpenAI-compatible embeddings endpoint the user names (OpenAI,
Gemini's OpenAI-compatible API, OpenRouter, or a local server such as Ollama or Text Embeddings Inference) turns facts
and messages into vectors, and recall uses them next to the words, never in their place:

- a fact's vector is made after the turn, on the core's background worker (``sync_turn``), never inside a write: a
  network call must not hold the database or the provider lock;
- vectors live in their own table, ``fact_embeddings``: one per current fact, with the model and a hash of the text it
  was made from, so a text an older version changed, or a vector of another model, is never compared;
- every failure falls back to words without a word of complaint to the user; after one the endpoint is left alone for
  a while (60 s, doubling up to 15 min), so a dead server does not cost seconds on every turn; a batch of fact
  vectors that only ran out of the after-turn time is no failure: the next batch is smaller (issue #5);
- what recall must not inject (``_recall_problem``) is never sent, nor a message holding a credential.

Nothing here runs, and nothing leaves the machine, unless ``embeddings: on``. Design after
itpartypattaya/hermes-plugin-holographic#3 by njwangfeng."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import logging
import math
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional, as for HRR: without it the feature stays off
    np = None

from .text import UNSEGMENTED, is_greeting, query_terms

logger = logging.getLogger(__name__)

TEXT_MAX_CHARS = 2000  # a fact holds at most 1000; a message is cut here before it is sent
BATCH = 32  # texts per request
_PAUSE_FIRST, _PAUSE_MAX = 60.0, 900.0  # seconds the endpoint is left alone after a failure, doubling
_QUERY_CACHE = 64  # message vectors kept: prefetch and a fact_store search of the same text make one call
_MAX_RESPONSE = 64 << 20
# How far above this message's usual similarity to the store's facts a fact must stand to be offered on meaning alone,
# in standard deviations over what the closest of N unrelated facts reaches by chance (sqrt(2 ln N)): a paraphrase of a
# stored fact stands out, chit-chat and off-topic questions mostly do not, and the bar rises with the store as the
# chance maximum does. Calibrated 2026-10-14 on 100 synthetic facts in five languages, 67 paraphrases sharing no
# words with their fact and 45 questions that must recall nothing, on text-embedding-3-small and gemini-embedding-001
# (stores of 30, 50, 100): margin 1.0 recalls 79-93 % of paraphrases at 100 facts with no chit-chat and 2 of 22
# off-topic questions (both near a stored topic). Absolute similarity differs by model (a paraphrase scores 0.38-0.66
# on the first, 0.68-0.84 on the second), so it is only an optional floor (embedding_min_score). Below
# MIN_VECTORS_FOR_Z vectors the spread says nothing and only that floor can admit a fact.
MIN_REQUEST_SECONDS = 0.5  # a request is not started with less time than this left
# A batch of fact vectors is sized, once the endpoint's speed was measured, to take this share of the time left:
# a local server may need 7.7 s for 32 facts, longer than the after-turn budget (issue #5).
_FACT_FIT = 0.5
_PROBE_BATCH = 4  # facts in the first request to an endpoint whose speed is not known yet
_MAX_CUTS = 3  # batches of facts cut short in a row before the endpoint counts as failing (a server that never answers)
_CUT = object()  # _request: a request for fact vectors ran out of time (embed_facts decides what that means)
_BUSY = object()  # _request: not sent or not answered in time while our own fact request held the endpoint
RECALL_MARGIN = 0.6  # before a turn: the z the top meaning candidate needs over the store's own
# spread (z_bar). At 1.0 the bar (~3.9 for a 64-fact store) turned away paraphrase targets whose
# z sat at 3.4-3.8. Final gate run (2026-10-25, git_sha 958de48, emb on bge-m3, rerank off, both
# margins, household test + 3 contrast + multilingual, 540 real rows): hit@5 481 -> 486, b=5 c=0,
# targets lost 0, confusion 34 -> 34, noise 94 -> 95 — one ru chit row (contrast_ru c31,
# "light_evening") passes the gate at 0.6. Pooled with the maintainer's held-out 126 (4/0, noise
# 44 -> 44): exact McNemar p = 0.0039, noise 138 -> 139 — three of four conditions pass, noise
# fails by that one row (issuecomment-6381967163). SEARCH_MARGIN (0.5) stays: an asked search is
# deliberately more permissive than a proactive recall.
SEARCH_MARGIN = 0.5  # fact_store search: the agent asked, and judges what it gets
# fact_store search beyond the words (Retriever.beyond_words): of the SEARCH_TOP_K facts closest in meaning, one is
# shown when it is closer than the best the words found by SEARCH_BEAT_WORDS and stands out at least SEARCH_Z_FLOOR.
# Measured 2026-10-14 (issue #3): on Mia's store, questions sharing no word but a name 13/27 -> 18/27 and none
# at all 3/12 -> 6/12 with the same noise on 30 questions that must find nothing (1); synthetic 59 -> 60 of 67
# (text-embedding-3-small) and 63 -> 66 (gemini-embedding-001). Rank alone (top 3 into the pool) found as much but
# put a fact by meaning into 20 of those 30 questions.
SEARCH_TOP_K = 3
SEARCH_BEAT_WORDS = 0.06
SEARCH_Z_FLOOR = 1.0
MIN_VECTORS_FOR_Z = 20


def z_bar(count: int, margin: float) -> float:
    """The z a fact needs among *count* vectors: the chance maximum of that many plus *margin*."""
    return math.sqrt(2.0 * math.log(max(count, 2))) + margin


def _now() -> float:
    return time.monotonic()


def _facts(count: int) -> str:
    """"1 fact", "12 facts" (0.17.4, issue #5: "1 facts in 0.3 s")."""
    return f"{count} fact" if count == 1 else f"{count} facts"


_RE_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_RE_KEY_PREFIX = re.compile(r"(?:sk|pk|rk)[-_]|AIza|gh[pousr]_|github_pat_|xox[abprs]-|glpat-|AKIA[0-9A-Z]{8}")


def text_hash(text: str) -> str:
    """What a vector was made from: a different text (a change, an older version's edit) is a different hash."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]


def to_blob(vector) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def _number(config: dict, key: str, default, low, high, kind=float):
    """config[key] as a number within [low, high]; unset is *default*. A value that is no number is replaced by
    *default*, one outside the range is brought into it, and either is logged with the key's name (not a value that is
    no number: a ${VAR} put in the wrong key would quote a secret): search by meaning must not run quietly on settings
    other than the ones written."""
    value = config.get(key)
    if value is None or str(value).strip() == "":
        return default
    try:
        number = kind(value)
    except (TypeError, ValueError, OverflowError):
        logger.warning("holographic: %s must be %s; using %s", key, "a whole number" if kind is int else "a number",
                       default)
        return default
    clamped = min(high, max(low, number))
    if clamped != number:
        logger.warning("holographic: %s = %s is outside %s..%s; using %s", key, number, low, high, clamped)
    return clamped


class Settings:
    """The plugin config's embedding_* keys, validated (see README, "Search by meaning")."""

    __slots__ = ("url", "model", "key_env", "dimensions", "timeout", "min_score", "recall_facts", "weight", "space",
                 "catch_up")

    def __init__(self, config: dict):
        self.url = str(config.get("embedding_url") or "https://api.openai.com/v1").strip()
        self.model = str(config.get("embedding_model") or "text-embedding-3-small").strip()
        self.key_env = str(config.get("embedding_key_env") or "OPENAI_API_KEY").strip()
        self.dimensions = _number(config, "embedding_dimensions", None, 1, 65536, int)
        # under the core's bound on an external provider's prefetch (8 s), with room for the words
        self.timeout = _number(config, "embedding_timeout", 3.0, 0.5, 6.0)
        self.min_score = _number(config, "embedding_min_score", None, -1.0, 1.0)
        self.recall_facts = _number(config, "embedding_prefetch_facts", 1, 0, 3, int)
        self.weight = _number(config, "embedding_weight", 0.3, 0.0, 0.8)
        # a label of your own, part of the vectors' identity: change it when the server starts serving another model
        # under the same address and name, and every vector is made again
        self.space = str(config.get("embedding_space") or "").strip()
        # seconds the fact vectors after a turn may take (None: the plugin's 8). No reply waits for them, but the core
        # runs every provider's after-turn work on one worker in turn: more here delays the next turn's extraction
        self.catch_up = _number(config, "embedding_catch_up_seconds", None, 2.0, 60.0)

    @property
    def tag(self) -> str:
        """Which vectors may be compared, as stored next to each: the model at this address (one model name may mean
        two models on two servers), with the dimension when one was asked for. Host, port and path only: no user
        name or password in the address; the path keeps its case; a query string (a deployment, or a key) by a short
        hash of it, never as written. A server that starts serving another model under the same address and name
        cannot be told apart: set or change embedding_space then, and every vector is made again."""
        try:
            parts = urllib.parse.urlsplit(self.url)
            where = (f"{parts.scheme.lower()}://{(parts.hostname or '').lower()}"
                     + (f":{parts.port}" if parts.port else "") + parts.path.rstrip("/"))
            if parts.query:
                where += "?" + hashlib.sha256(parts.query.encode("utf-8")).hexdigest()[:8]
        except ValueError:
            where = "?"
        return (f"{where}/{self.model}" + (f"@{self.dimensions}" if self.dimensions else "")
                + (f"#{self.space}" if self.space else ""))


def settings_from(config: dict) -> Settings | None:
    """Settings when ``embeddings`` is on, else None (the default: no vectors, no calls)."""
    value = config.get("embeddings", "off")
    on = value is True or str(value).strip().lower() in {"on", "true", "yes", "1"}
    return Settings(config) if on else None


def _local(host: str) -> bool:
    """A host on this machine or the local network: plain http (and no key) is allowed to it. An address is judged as
    an address first (a public IPv6 one has no dot either); a name by its form, not by what it resolves to."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return (host == "localhost" or "." not in host  # a container or LAN name
                or host.endswith((".localhost", ".local", ".lan", ".internal", ".home.arpa")))
    ip = getattr(ip, "ipv4_mapped", None) or ip  # ::ffff:8.8.8.8 is 8.8.8.8 (private on Python <= 3.12)
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _endpoint(url: str) -> tuple[str | None, bool, str]:
    """(the /embeddings address, whether it is local, "") or (None, False, why it is refused)."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return None, False, "embedding_url is not a valid address"
    if parts.scheme not in ("https", "http") or not host:
        return None, False, "embedding_url is not an http(s) address"
    local = _local(host)
    if parts.scheme == "http" and not local:
        return None, False, "embedding_url sends plain http to a public host: use https"
    path = parts.path.rstrip("/")
    if not path.endswith("/embeddings"):
        path += "/embeddings"
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, "")), local, ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would carry the Authorization header to another address: it is a failure instead."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _vectors(payload, count: int, dimensions: int | None = None) -> list:
    """Unit-length float32 vectors from an OpenAI-style answer, in input order; ValueError when it is not one: another
    number of vectors, indexes that are not exactly 0 … count-1, vectors of different sizes or not of the size asked
    for, anything not a finite number, a zero vector."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or len(data) != count:
        raise ValueError("the answer has a different number of vectors than texts sent")
    indexes = [item.get("index") if isinstance(item, dict) else None for item in data]
    if any(i is not None for i in indexes):
        # A missing index is 0: Gemini's OpenAI-compatible API, like other protobuf-backed servers, leaves a field at
        # its default out (measured 2026-10-14). Two missing ones are two zeros, and refused.
        indexes = [0 if i is None else i for i in indexes]
        if sorted(i if isinstance(i, int) and not isinstance(i, bool) else -1 for i in indexes) != list(range(count)):
            raise ValueError("the answer's vector indexes do not match the texts sent")
        data = [item for _, item in sorted(zip(indexes, data), key=lambda pair: pair[0])]
    out, size = [], None
    for item in data:
        values = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(values, list) or not values:
            raise ValueError("a vector is missing from the answer")
        try:
            vector = np.asarray(values, dtype=np.float64)  # float64: squares of large float32 values overflow
        except (TypeError, ValueError, OverflowError):
            raise ValueError("a vector holds something that is not a number") from None
        if vector.ndim != 1 or (size is not None and vector.shape[0] != size) or not np.all(np.isfinite(vector)):
            raise ValueError("a vector is malformed")
        if dimensions and vector.shape[0] != dimensions:
            raise ValueError("a vector has another size than embedding_dimensions asks for")
        scale = float(np.max(np.abs(vector)))
        if scale == 0.0:
            raise ValueError("a vector is all zeros")
        vector = vector / scale  # then the norm cannot overflow or underflow
        unit = (vector / float(np.linalg.norm(vector))).astype(np.float32)
        if not np.all(np.isfinite(unit)) or abs(float(np.linalg.norm(unit)) - 1.0) > 1e-3:
            raise ValueError("a vector cannot be brought to unit length")
        size = unit.shape[0]
        out.append(unit)
    return out


def _read_until(response, deadline: float) -> bytes:
    """The body, read a piece at a time: the socket timeout bounds each read, not the whole, so a server that sends a
    few bytes before every timeout is cut off at *deadline*, and a body past _MAX_RESPONSE is refused."""
    chunks, total = [], 0
    while True:
        if _now() > deadline:
            raise TimeoutError("the answer took longer than embedding_timeout")
        chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > _MAX_RESPONSE:
            raise ValueError("the answer is too large")
        chunks.append(chunk)


class _GivenUp:
    """A request whose caller stopped waiting: its thread, what it returns, whether it carries fact vectors or was a
    message's request that timed out behind ours (busy), when it was given up on, and whether its failure was already
    counted (a late answer then changes nothing: one request, one step of the pause)."""

    __slots__ = ("worker", "box", "fact", "busy", "at", "counted")

    def __init__(self, worker, box, fact: bool, busy: bool, counted: bool):
        self.worker, self.box, self.fact, self.busy, self.counted = worker, box, fact, busy, counted
        self.at = time.perf_counter()


class Embedder:
    """The endpoint: texts in, unit-length float32 vectors out. Shared by every store a provider opens; thread-safe."""

    def __init__(self, settings: Settings, environ=None):
        import os
        self.settings = settings
        self._environ = os.environ if environ is None else environ
        self._url, self._local, self._refused = _endpoint(settings.url)
        self._lock = threading.Lock()
        self._pause = self._paused_until = 0.0
        self._last_error = ""
        # +1 with every failure: an answer to a request sent before the latest failure does not end its pause
        self._epoch = 0
        self._queries: OrderedDict = OrderedDict()
        # Requests given up on that may still be running (a stalled server): while one is, no new request starts,
        # so a server that never answers holds one thread and connection per caller at most (the catch-up, outside
        # the provider lock, and one caller under it), not one per retry.
        self._abandoned: list[_GivenUp] = []
        self.calls = 0  # requests sent (stats, tests)
        self._messages_sent = 0  # requests other than fact batches: a fact batch they queued behind is not measured
        # Fact vectors (embed_facts): seconds per character the endpoint took, None until measured; fact requests
        # being waited for, and when the last one ended; batches cut short in a row; what the endpoint's last batch
        # did (stats). Shared by every store, as the endpoint is.
        self._fact_rate: float | None = None
        self._facts_out = 0
        self._fact_ended = 0.0
        self._cuts = 0
        self.last_batch = ""

    def _key(self) -> str:
        return str(self._environ.get(self.settings.key_env) or "").strip() if self.settings.key_env else ""

    def unavailable(self) -> str:
        """Why the endpoint cannot be used at all ("" when it can): numpy, the address, the key. Never the key's value:
        a key with a line break inside would make the HTTP library raise an error quoting the header."""
        if np is None:
            return "numpy is not installed"
        if self._refused:
            return self._refused
        name = self.settings.key_env
        if name and (not _RE_ENV_NAME.fullmatch(name) or _RE_KEY_PREFIX.match(name)):
            # a key pasted where its variable's name goes: never echoed (status and log would show it)
            return "embedding_key_env must be the name of an environment variable, not the key itself"
        key = self._key()
        if not key and not self._local:
            return f"{self.settings.key_env} is not set"
        if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in key):
            return f"the value of {self.settings.key_env} has a space or a control character inside"
        return ""

    def status(self) -> str:
        why = self.unavailable()
        if why:
            return f"unavailable: {why}"
        with self._lock:
            noted = self._reap_locked() or self._stalled_locked()
            left = self._paused_until - _now()
            text = (f"paused after an error ({self._last_error}); tried again in {int(left) + 1} s" if left > 0
                    else "ready")
        if noted:
            self._log_failure(*noted)
        return text

    def usable(self) -> bool:
        return self.status() == "ready"

    def embed(self, texts: list[str], budget: float | None = None) -> list | None:
        """Vectors of *texts* (each cut to TEXT_MAX_CHARS), or None: unusable, or a request failed (logged; paused).
        *budget*: seconds the caller can still wait, for every request together (each may use all that is left);
        None: embedding_timeout for each."""
        if not texts or not self.usable():
            return None
        out: list = []
        deadline = None if budget is None else _now() + budget
        for start in range(0, len(texts), BATCH):
            left = None if deadline is None else deadline - _now()
            if left is not None and left < MIN_REQUEST_SECONDS:
                return None
            got = self._request([t[:TEXT_MAX_CHARS] for t in texts[start:start + BATCH]], left)
            if not isinstance(got, list):  # None, or _BUSY: our own fact request held the endpoint
                return None
            out += got
        return out

    def fact_batch(self, sizes: list[int], budget: float, probe: bool = True) -> int:
        """How many of the waiting facts (their lengths, in the order they would be sent) the next request carries with
        *budget* seconds left: _PROBE_BATCH until the endpoint's speed was measured, then as many as that speed fits
        into _FACT_FIT of *budget*, at most BATCH — on a local server into _FACT_FIT of embedding_timeout at most, since
        such a server often answers one request at a time and a message's vector before a turn may wait behind the
        batch; one fact whenever one is expected to take less than *budget*. None otherwise — the end of a budget is
        no test of the server (review 0.16.3: a fact sent into the last half second paused a working endpoint) —
        unless *probe* (the turn's first request, with most of the budget): then one, to try."""
        with self._lock:
            rate = self._fact_rate
        sizes = [max(1, min(size, TEXT_MAX_CHARS)) for size in sizes[:BATCH]]
        if not sizes:
            return 0
        if not rate:
            return min(len(sizes), _PROBE_BATCH)
        room = budget * _FACT_FIT
        if self._local:
            room = min(room, self.settings.timeout * _FACT_FIT)
        count, total = 0, 0
        for size in sizes:
            total += size
            if total * rate > room:
                break
            count += 1
        if count == 0 and (probe or sizes[0] * rate < budget * 0.9):
            count = 1
        return count

    def embed_facts(self, texts: list[str], budget: float, decisive: bool = True) -> list | None:
        """Vectors of up to BATCH fact texts in one request that may take *budget* seconds, or None; how fast the
        endpoint answered sizes the next batch (fact_batch). A request that ran out of time is not the endpoint's
        failure — the server may be slow rather than down (issue #5: 7.7 s for 32 facts against 7 s left): the
        endpoint is not paused and the next batch is smaller. It is a failure when the request was *decisive* (a
        turn's first, with most of its budget) and carried one fact, or after _MAX_CUTS cuts in a row. Nothing is
        sent, and nothing counts, while an earlier request of ours still holds the endpoint."""
        if not texts or len(texts) > BATCH or budget < MIN_REQUEST_SECONDS or not self.usable():
            return None
        batch = [t[:TEXT_MAX_CHARS] for t in texts]
        chars = sum(max(1, len(t)) for t in batch)
        timing: dict = {}
        got = None
        with self._lock:
            self._facts_out += 1
            messages = self._messages_sent
        try:
            got = self._request(batch, budget, fact=True, timing=timing)
        finally:
            with self._lock:
                self._facts_out -= 1
                # sent, and its thread has ended — answered, failed, or timed out inside urllib (review 0.16.3, round
                # 3: a batch cut by a socket timeout left no trace, and a message queued behind it paused the endpoint);
                # one still running is tracked as given up on
                if "took" in timing and "given" not in timing:
                    self._fact_ended = time.perf_counter()
        if got is _BUSY:
            with self._lock:
                self.last_batch = "not sent: an earlier request was still running"
            return None
        if got is _CUT:
            with self._lock:
                self._cuts += 1
                failing = (decisive and len(batch) == 1) or self._cuts >= _MAX_CUTS
                if failing:
                    self._cuts = 0
                    given = timing.get("given")
                    if given is not None:
                        given.counted = True
                    noted = self._fail_locked("no answer (timeout)" if len(batch) == 1
                                              else f"no answer (timeout, {_MAX_CUTS} batches in a row)")
                self._fact_rate = max(self._fact_rate or 0.0, 2.0 * budget / chars)
                self.last_batch = (f"{_facts(len(batch))} {'was' if len(batch) == 1 else 'were'} not done in {budget:.1f} s"
                                   + ("" if failing else "; the next batch is smaller"))
            if failing:
                self._log_failure(*noted)
            else:
                logger.info("holographic: a batch of %d fact vectors took longer than the %.1f s left after the "
                            "turn; the next batch is smaller", len(batch), budget)
            return None
        if got is None:
            return None
        took = timing.get("took", 0.0)  # from sending: a wait for an earlier request is not the server's speed
        with self._lock:
            # a message sent meanwhile may have queued with this batch on the server: its wait is no measure
            if self._fact_rate is None or self._messages_sent == messages:
                self._fact_rate = max(took, 1e-3) / chars
            self._cuts = 0
            self.last_batch = f"{_facts(len(batch))} in {took:.1f} s"
        return got

    def query(self, text: str):
        """The vector of a message, from a small cache when it was asked for a moment ago; None when unavailable."""
        key = (text or "")[:TEXT_MAX_CHARS]
        with self._lock:
            if key in self._queries:
                self._queries.move_to_end(key)
                return self._queries[key]
        got = self.embed([key])
        if got is None:
            return None
        with self._lock:
            self._queries[key] = got[0]
            while len(self._queries) > _QUERY_CACHE:
                self._queries.popitem(last=False)
        return got[0]

    def _grace(self) -> float:
        """How long a request given up on may still hold the endpoint as ours (busy), not as a stalled server."""
        return 2.0 * self.settings.timeout

    def _reap_locked(self):
        """Under _lock: forget the requests given up on that have ended. One whose answer says the endpoint failed — an
        HTTP status or a malformed answer that came after the caller stopped waiting (review 0.16.3: a 503 after every
        cut kept it "ready") — and whose failure was not counted yet pauses it now, once for all of them; returns what
        to log, or None."""
        late, alive = [], []
        for given in self._abandoned:
            if given.worker.is_alive():
                alive.append(given)
                continue
            if given.fact:
                self._fact_ended = max(self._fact_ended, given.box.get("ended", 0.0))
            error = given.box.get("error")
            if given.counted:
                continue
            if isinstance(error, urllib.error.HTTPError):
                late.append(f"HTTP {error.code}")
            elif "malformed" in given.box:
                late.append(f"malformed answer: {given.box['malformed']}")
        self._abandoned = alive
        if not late:
            return None
        return self._fail_locked(late[0] + (f" (and {len(late) - 1} more)" if len(late) > 1 else ""))

    def _stalled_locked(self):
        """Under _lock: a request given up on more than _grace() ago and still running is a stalled server, counted
        now rather than at the next request (review 0.16.3, round 3: the status said "ready" meanwhile); returns what to
        log, or None. Counted once: the next request that finds it still running fails again, as before."""
        now = time.perf_counter()
        stale = [given for given in self._abandoned
                 if not given.counted and given.worker.is_alive() and now - given.at > self._grace()]
        if not stale:
            return None
        for given in stale:
            given.counted = True
        return self._fail_locked("no answer (an earlier request is still waiting)")

    def _ours_busy(self, since: float) -> bool:
        """Under _lock: did a fact request of ours hold the endpoint at some point since *since* — being waited for,
        ended meanwhile, or given up on a moment ago and still running (review 0.16.3: a batch that ended just before
        the message's deadline left the message's wait unexplained)?"""
        if self._facts_out > 0 or self._fact_ended > since:
            return True
        now = time.perf_counter()
        return any(given.fact and (given.box.get("ended", 0.0) > since
                                   or (given.worker.is_alive() and now - given.at <= self._grace()))
                   for given in self._abandoned)

    def _request(self, batch: list[str], budget: float | None = None, fact: bool = False, timing: dict | None = None):
        """One request. It runs on a daemon thread and the caller waits *budget* seconds — embedding_timeout when none
        is given (a message's vector before a turn), the remaining after-turn budget for fact vectors, which no reply
        waits for and come in batches of up to 32 — and no longer: urllib's timeout bounds each socket operation, so a
        server that sends its headers or its chunks a byte at a time could otherwise hold the caller — and the provider
        lock prefetch holds — far longer (review 2026-10-14). A request given up on finishes or fails on its own
        thread; the endpoint is paused. Two exceptions (issue #5): a request for *fact* vectors that runs out of time
        returns _CUT, and embed_facts decides; a message's request that runs out of time while a fact request of ours
        held the endpoint (a local server may answer one request at a time) returns _BUSY, without a pause.

        While a request given up on is still running, no new one starts: this one waits for it up to half of its own
        time (one cut short after the last turn is often about to finish), then returns _BUSY if every request still
        running is ours and was given up on less than _grace() ago, and fails otherwise ("an earlier request is still
        waiting"): a server that never answers is paused, and paused for longer each time, as before 0.16.3.
        *timing* gets "took", the seconds from sending to the answer, and "given", the request if it was given up on.
        An answer ends the pause only if no failure was counted since it was sent."""
        timeout = self.settings.timeout if budget is None else max(0.1, budget)
        with self._lock:
            noted = self._reap_locked()
            waiting = [given.worker for given in self._abandoned]
        if noted:
            self._log_failure(*noted)
            return None
        if waiting:
            started, until = time.perf_counter(), time.perf_counter() + timeout / 2
            for earlier in waiting:
                earlier.join(max(0.0, until - time.perf_counter()))
            with self._lock:
                noted = self._reap_locked()
                now = time.perf_counter()
                running = list(self._abandoned)
                ours = bool(running) and all((given.fact or given.busy) and now - given.at <= self._grace()
                                             for given in running)
                if not noted and running and not ours:
                    for given in running:
                        given.counted = True
                    noted = self._fail_locked("no answer (an earlier request is still waiting)")
            if noted:
                self._log_failure(*noted)
                return None
            if running:
                return _BUSY
            timeout = max(0.1, timeout - (time.perf_counter() - started))
        body = {"model": self.settings.model, "input": batch}
        if self.settings.dimensions:
            body["dimensions"] = self.settings.dimensions
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "hermes-plugin-holographic"}
        key = self._key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(self._url, data=json.dumps(body).encode("utf-8"), headers=headers,
                                         method="POST")
        with self._lock:
            if self._paused_until > _now():  # another caller's failure came while this one waited
                return None
            epoch = self._epoch
            self.calls += 1
            if not fact:
                self._messages_sent += 1
        deadline = _now() + timeout
        box: dict = {}

        def send():  # decoding too, so the caller's wait covers the whole of it
            try:
                try:
                    with _OPENER.open(request, timeout=timeout) as response:
                        raw = _read_until(response, deadline)
                except BaseException as exc:  # noqa: BLE001 - handed to the caller below, which raises or reports it
                    box["error"] = exc
                    return
                try:
                    box["vectors"] = _vectors(json.loads(raw), len(batch), self.settings.dimensions)
                except ValueError as exc:  # our own wording, or the JSON parser's position
                    box["malformed"] = str(exc)[:120]
            finally:
                box["ended"] = time.perf_counter()

        worker = threading.Thread(target=send, name="holographic-embeddings", daemon=True)
        sent = time.perf_counter()
        worker.start()
        worker.join(timeout)
        if timing is not None:
            timing["took"] = time.perf_counter() - sent
        alive = worker.is_alive()
        # urllib's own timeout may end the request at the same moment: a read times out bare, waiting for the headers
        # comes wrapped in URLError (review 0.16.3), as does a connection that timed out (_MAX_CUTS bounds that for
        # facts). A connection refused or reset is a failure.
        error = box.get("error")
        timed_out = alive or isinstance(error, TimeoutError) or (
            isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError))
        if timed_out:
            with self._lock:
                # ours only if a batch of ours held the endpoint for a fair part of the wait, not its last instant
                ours = not fact and self._ours_busy(sent + timeout / 4)
                fails = alive and not fact and not ours
                if alive:
                    given = _GivenUp(worker, box, fact, busy=ours, counted=fails)
                    self._abandoned.append(given)
                    if timing is not None:
                        timing["given"] = given
                noted = self._fail_locked("no answer (timeout)") if fails else None
            if noted:
                self._log_failure(*noted)
                return None
            if fact:
                return _CUT
            if ours:
                return _BUSY
        # What is logged and shown in stats is a category, never the text of an exception raised while sending: it
        # may quote the request (a header with the key in it).
        try:
            if "error" in box:
                raise box["error"]
        except urllib.error.HTTPError as exc:  # the body may echo the request: not read
            # A model that cannot shorten its vectors (bge-m3 behind an aggregator, reported on 0.14.1) answers 400 to
            # the `dimensions` field rather than ignoring it: name the likely cause instead of a bare code.
            hint = (" — embedding_dimensions is set; a model that cannot shorten its vectors rejects it, leave it unset"
                    if exc.code == 400 and self.settings.dimensions else "")
            self._failed(f"HTTP {exc.code}{hint}")
            return None
        except urllib.error.URLError as exc:
            self._failed(f"cannot reach the endpoint ({type(exc.reason).__name__})")
            return None
        except (OSError, ValueError, http.client.HTTPException) as exc:  # timeout, reset, a broken answer
            self._failed(f"no answer ({type(exc).__name__})")
            return None
        except Exception as exc:  # noqa: BLE001 - anything else raised while sending: a category, never its text
            self._failed(f"no answer ({type(exc).__name__})")
            return None
        if "malformed" in box:
            self._failed(f"malformed answer: {box['malformed']}")
            return None
        vectors = box["vectors"]
        self._recovered(epoch)
        return vectors

    def _fail_locked(self, why: str) -> tuple:
        """Under _lock: count a failure — pause the endpoint, longer each time — and return what to log."""
        first = not self._last_error
        self._pause = min(_PAUSE_MAX, self._pause * 2) if self._pause else _PAUSE_FIRST
        self._paused_until = _now() + self._pause
        self._last_error = why
        self._epoch += 1
        return why, first, self._pause

    @staticmethod
    def _log_failure(why: str, first: bool, pause: float) -> None:
        (logger.warning if first else logger.debug)(
            "holographic: embeddings endpoint failed (%s); words only for %d s", why, int(pause))

    def _failed(self, why: str) -> None:
        with self._lock:
            noted = self._fail_locked(why)
        self._log_failure(*noted)

    def _recovered(self, epoch: int | None = None) -> None:
        """An answer came: the endpoint works — unless a failure was counted after its request was sent (review
        0.16.3: a batch that ended well cleared the pause a newer 503 had set)."""
        with self._lock:
            if epoch is not None and epoch != self._epoch:
                return
            was = self._last_error
            self._pause = self._paused_until = 0.0
            self._last_error = ""
        if was:
            logger.info("holographic: embeddings endpoint works again")


def worth_asking(query: str, *, before_turn: bool) -> bool:
    """Is the message worth a call to the endpoint? Never a greeting or a thank-you. Before a turn it needs two words
    that carry meaning, or five characters of Chinese, Japanese, Korean or Thai: short chit-chat sits on the similarity
    noise floor of every store, and only its length tells it apart (#3). An explicit search needs one."""
    text = (query or "").strip()
    if not text or is_greeting(text):
        return False
    glued = sum(len(run) for run in UNSEGMENTED.findall(text))
    words = sum(1 for t in query_terms(UNSEGMENTED.sub(" ", text)) if t.word)
    return words >= (2 if before_turn else 1) or glued >= (5 if before_turn else 2)


class Meaning:
    """How close each current fact is to one message (cosine of unit vectors), and how unusual that closeness is for
    this message over the whole store."""

    __slots__ = ("sims", "neutral", "mean", "std", "weight", "hashes")

    def __init__(self, ids: list[int], sims, weight: float, hashes: list[str] | None = None):
        self.sims = {fact_id: float(s) for fact_id, s in zip(ids, sims)}
        # The text hash each vector was made from: a fact read again later (another writer may have changed it
        # meanwhile) keeps its score only while its text is still that one (fits).
        self.hashes = dict(zip(ids, hashes)) if hashes is not None else None
        self.neutral = float(np.median(sims)) if len(ids) else 0.0  # a fact without a vector neither gains nor loses
        self.mean = float(np.mean(sims)) if len(ids) else 0.0
        self.std = float(np.std(sims)) if len(ids) else 0.0
        self.weight = weight

    def cos(self, fact_id: int) -> float:
        return self.sims.get(fact_id, self.neutral)

    def fits(self, fact_id: int, content: str) -> bool:
        """Is *content* still the text this fact's score was computed from (always, when no hashes were kept)?"""
        return self.hashes is None or self.hashes.get(fact_id) == text_hash(content)

    def z(self, fact_id: int) -> float:
        s = self.sims.get(fact_id)
        return (s - self.mean) / self.std if s is not None and self.std > 1e-9 else 0.0

    def standout(self, fact_id: int, margin: float, min_score: float | None) -> bool:
        """Close enough to be offered on meaning alone: above embedding_min_score when it is set, and, with enough
        vectors to judge, above z_bar(count, margin)."""
        s = self.sims.get(fact_id)
        if s is None or (min_score is not None and s < min_score):
            return False
        if len(self.sims) >= MIN_VECTORS_FOR_Z:
            return self.z(fact_id) >= z_bar(len(self.sims), margin)
        return min_score is not None

    def ranked(self) -> list[tuple[int, float]]:
        return sorted(self.sims.items(), key=lambda item: (-item[1], item[0]))


class VectorIndex:
    """One store's fact vectors of one model, kept between queries: a vector is read from the file again only when its
    fact is new, its text changed or it was made again, and the matrix is rebuilt only when that set changes.
    Thread-safe (the catch-up after a turn runs on the core's background worker)."""

    def __init__(self, store, tag: str):
        self.store, self.tag = store, tag
        self._lock = threading.Lock()
        # fact_id -> (cheap fingerprint of the content, its hash): no plaintext kept in memory (a forgotten fact's
        # text must not outlive it in the process either; forget also calls drop)
        self._hashes: dict[int, tuple[tuple[int, int], str]] = {}
        self._vectors: dict[int, tuple[tuple, object]] = {}  # fact_id -> ((text hash, size, revision), vector)
        self._matrix: tuple | None = None  # (fresh set, dimension, ids, matrix)
        # The size of the last message's vector: a fact vector of another size (the endpoint changed what it returns
        # under the same name) is made again (pending) instead of staying out of recall for good.
        self._dim: int | None = None

    def _state(self) -> list[tuple[int, str, str, str | None, int, str]]:
        """(fact_id, content, hash of the content, hash its vector of this model was made from or None, the vector's
        byte size, its revision — new with every write, so a vector made again is read again) of every current
        fact."""
        try:
            rows = self.store.read("SELECT f.fact_id, f.content, e.text_hash, LENGTH(e.vector), e.revision FROM facts f "
                                   "LEFT JOIN fact_embeddings e ON e.fact_id = f.fact_id AND e.model = ? "
                                   "ORDER BY f.fact_id", (self.tag,))
        except sqlite3.OperationalError:  # no table yet: a store no writer of 0.14.0 has opened
            return []
        out = []
        for fact_id, content, made_from, size, revision in rows:
            hit, seen = self._hashes.get(fact_id), (len(content or ""), hash(content or ""))
            if hit is None or hit[0] != seen:
                hit = self._hashes[fact_id] = (seen, text_hash(content))
            out.append((fact_id, content, hit[1], made_from, size or 0, revision or ""))
        if len(self._hashes) > len(out) + 1024:  # deleted facts
            alive = {row[0] for row in out}
            self._hashes = {k: v for k, v in self._hashes.items() if k in alive}
        return out

    def drop(self, fact_id: int) -> None:
        """Forget what this index holds of a fact (forget)."""
        with self._lock:
            self._hashes.pop(fact_id, None)
            self._vectors.pop(fact_id, None)
            self._matrix = None

    def ready(self) -> bool:
        """Does any current fact have a vector of this model (worth a call for the message)?"""
        try:
            return bool(self.store.read("SELECT 1 FROM fact_embeddings e JOIN facts f ON f.fact_id = e.fact_id "
                                        "WHERE e.model = ? LIMIT 1", (self.tag,)))
        except sqlite3.OperationalError:
            return False

    def matrix(self, dim: int) -> tuple[list[int], object]:
        """(fact ids, their unit vectors as rows) of the current facts whose vector was made from their present text
        and has *dim* dimensions (the message's)."""
        ids, matrix, _ = self.matrix_with_hashes(dim)
        return ids, matrix

    def matrix_with_hashes(self, dim: int) -> tuple[list[int], object, list[str]]:
        """matrix() plus, for each row, the hash of the text its vector was made from."""
        with self._lock:
            fresh = tuple((fact_id, (digest, size, revision)) for fact_id, _, digest, made_from, size, revision
                          in self._state() if made_from == digest and size > 0 and size % 4 == 0)
            need = [fact_id for fact_id, made in fresh if self._vectors.get(fact_id, ((),))[0] != made]
            for start in range(0, len(need), 500):
                ids = need[start:start + 500]
                for fact_id, digest, blob, revision in self.store.read(
                        f"SELECT fact_id, text_hash, vector, revision FROM fact_embeddings WHERE model = ? AND fact_id "
                        f"IN ({','.join('?' * len(ids))})", (self.tag, *ids)):
                    self._vectors[fact_id] = ((digest, len(blob), revision or ""),
                                              np.frombuffer(bytes(blob), dtype="<f4"))
            if self._matrix is None or self._matrix[0] != fresh or self._matrix[1] != dim:
                alive = {fact_id for fact_id, _ in fresh}
                self._vectors = {k: v for k, v in self._vectors.items() if k in alive}
                made = [(fact_id, self._vectors[fact_id][1]) for fact_id, key in fresh
                        if self._vectors.get(fact_id, ((),))[0] == key]
                rows = [(fact_id, v) for fact_id, v in made if v.shape[0] == dim]
                matrix = np.vstack([v for _, v in rows]) if rows else np.zeros((0, dim), dtype=np.float32)
                made_from = dict((fact_id, key[0]) for fact_id, key in fresh)
                self._matrix = (fresh, dim, [fact_id for fact_id, _ in rows], matrix,
                                [made_from[fact_id] for fact_id, _ in rows])
            self._dim = dim
            return self._matrix[2], self._matrix[3], self._matrix[4]

    def _usable(self, digest: str, made_from: str | None, size: int) -> bool:
        """Was this fact's vector made from its present text, and (once a message has been compared) of its size?"""
        return made_from == digest and size > 0 and size % 4 == 0 and (self._dim is None or size == 4 * self._dim)

    def pending(self, limit: int, allowed) -> list[tuple[int, str, str]]:
        """Up to *limit* (fact_id, content, hash) of current facts with no usable vector of this model — none, one
        made from an older text, or one of another size than the last message's — newest first; facts *allowed*
        refuses are never sent."""
        with self._lock:
            waiting = [(fact_id, content, digest) for fact_id, content, digest, made_from, size, _ in self._state()
                       if not self._usable(digest, made_from, size)]
        out = []
        for item in reversed(waiting):
            if allowed(item[1]):
                out.append(item)
                if len(out) >= limit:
                    break
        return out

    def counts(self, allowed) -> dict:
        with self._lock:
            state = self._state()
            usable = [self._usable(digest, made_from, size) for _, _, digest, made_from, size, _ in state]
        done = sum(usable)
        held = sum(1 for ok, (_, content, *_rest) in zip(usable, state) if not ok and not allowed(content))
        return {"facts_with_vector": done, "waiting": len(state) - done - held, "never_sent": held}
