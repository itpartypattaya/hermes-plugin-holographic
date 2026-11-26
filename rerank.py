"""Second-stage rescoring: optional, offline-safe, off by default. A Cohere-compatible `/v1/rerank` endpoint (the
de-facto standard; OpenAI has no rerank API at all) reorders the final candidate pool after the blend. One synchronous
HTTP call at recall time; the only state kept is the endpoint's pause and its last measured speeds, in memory.

Tested against llama-server (`--reranking --embedding --pooling rank`). Any server that answers the Cohere shape
(``{"results": [{"index", "relevance_score"}]}``) without a key should work; the client sends no key, so a cloud API
that requires one (Cohere, Jina) refuses every call, and TEI's own ``/rerank`` answers a bare list, which is refused
as malformed. Either way the endpoint is paused and recall keeps the off order.

Guardrails (from RFC #9):
- off by default (``rerank: false``); behaviour and scores exactly 0.13.4's when off;
- a query holding a secret (``_secret_in``) → no call at all;
- any failure gives exactly the recall of rerank off, one log line of the error category, the endpoint goes to
  backoff 60 s doubling to 900 s (_PAUSE_FIRST / _PAUSE_MAX);
- ``rerank_timeout`` defaults to 3.0 (issue #5: a slow local endpoint is the normal case; this is recall's only
  protection, there is no catch-up);
- a call sends no more than the endpoint has shown it scores in half the timeout (RERANK_TIME_SHARE);
- the reranker judges the pair's text only — no hrr_vector / trust / decay in the request.

Design after itpartypattaya/hermes-plugin-holographic#9 by KostinPavel."""

from __future__ import annotations

import collections
import json
import logging
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import net  # the socket of the in-flight request, for the caller to end the read (issue #26)

logger = logging.getLogger(__name__)

PAUSE_FIRST, PAUSE_MAX = 60.0, 900.0  # seconds the endpoint is left alone after a failure, doubling
MAX_RESPONSE = 64 << 20
# What one query-document pair may hold (2026-10-26): jina-reranker-v2 and bge-reranker-v2-m3 take 1024 tokens per
# pair, and llama-server refuses the whole request when one pair runs over (500 past its batch, 400 past the
# context), which paused the endpoint on a production install's long night-job prompts. Both use XLM-RoBERTa's SentencePiece without
# byte fallback: every token covers at least one character, the word-start mark stands for a space of the text or
# one added at the start, and an unknown character is one token — so a pair of RERANK_PAIR_CHARS characters is at
# most that many tokens plus the start marks and the 4 special tokens, under 1024 for CJK, emoji and dense text
# alike. A message over RERANK_MAX_QUERY_CHARS is not sent (cutting it would rescore by its first lines); each
# document is cut to what the query leaves of the pair.
RERANK_PAIR_CHARS = 1000
RERANK_MAX_QUERY_CHARS = 500
DEFAULT_TOP_K = 40  # depth of the pool the cross-encoder sees; the first stage's limit is widened to match
DEFAULT_TIMEOUT = 3.0  # seconds; 3.0, not 2.0: a slow local endpoint is the normal case (issue #5)
# What one call may take of rerank_timeout (2026-10-29). A cross-encoder on a CPU scores the pairs at a steady speed,
# so its time grows with every character it is sent: a production llama-server (jina-reranker-v2 Q4_K_M, 2 threads, ~485
# tokens/s) answered 5 short facts in 0.4 s, 15 in 1.2 s and 40 in 3.3 s — over the default timeout, so the large
# pools of a big store failed and paused the endpoint. A call sends as many of the first documents as the endpoint,
# at the slowest of its last _SPEED_SAMPLES speeds, scores in this share of the timeout; the rest keep their order
# behind the scored ones, as the tail beyond top_k does. The other half is the margin for a busy server and for text
# with more tokens per character than the text measured. Before the first answer nothing is known and top_k goes out.
RERANK_TIME_SHARE = 0.5
_SPEED_SAMPLES = 5

#: The core's budget for the provider's prefetch: _EXTERNAL_PREFETCH_TIMEOUT_S in agent/memory_manager.py.
#: The embedding call and the rerank run back to back, so the sum of their clamps has to fit below
#: it with margin (maintainer 24:10, point 4: defaults 3 + 3 fit, emb 6 + rerank 10 — no).
_PREFETCH_BUDGET_S = 7.5
_BUSY = object()  # _request: another caller's request is in flight and that caller still waits for it
DEFAULT_WINDOW = 0.05  # rerank_window: the share of the tail below which the pool is cut (0 = cut nothing)


def _number(config: dict, key: str, default, low, high, kind=float):
    """config[key] as a number within [low, high]; unset is *default*. Mirrors embeddings._number (same contract)."""
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
    """The plugin config's rerank_* keys, validated (see README). One mode: the cross-encoder reorders the pool,
    admission stays with coverage; `rerank_window` (δ) trims the tail below δ of the best score, 0 keeps all —
    and never below two scored candidates (built into the window itself, review 13:06: no switch for it).
    `rerank_scale` is the endpoint's own scale, set explicitly: default `logit`, because llama-server
    (the documented setup) returns raw logits."""

    __slots__ = ("url", "model", "top_k", "weight", "timeout", "window", "scale")

    def __init__(self, config: dict):
        # rerank_mode / rerank_margin removed (point 6 of the 2026-10-22 review): one mode, the window is a separate switch.
        for gone, replacement in (("rerank_mode", "one mode; rerank: on is the switch"),
                                  ("rerank_margin", "rerank_window instead")):
            if config.get(gone) is not None and str(config.get(gone)).strip():
                logger.warning("holographic: %s was removed (%s); ignoring it", gone, replacement)
        self.url = str(config.get("rerank_url") or "http://127.0.0.1:8080/v1/rerank").strip()
        self.model = str(config.get("rerank_model") or "local-reranker").strip()
        self.top_k = _number(config, "rerank_top_k", DEFAULT_TOP_K, 1, 1000, int)  # int (kind=int)
        self.weight = _number(config, "rerank_weight", 0.0, 0.0, 1.0)
        self.window = _number(config, "rerank_window", DEFAULT_WINDOW, 0.0, 1.0)
        # the score scale is the ENDPOINT's own property and the user's setting (point 3 of the
        # 13:06 review): explicit, no probing and no latching; default logit — llama-server returns
        # raw logits. The provider's _CHOICES holds the set of values: outside it —
        # a named config problem with memory off, not a silent fallback.
        self.scale = str(config.get("rerank_scale") or "logit").strip().lower()
        # point 2 (19:39 review): the core's budget counts ONLY the enabled embeddings — the stray
        # embedding_timeout with embeddings off must not cut anything. Second problem: _number
        # returns the default WITHOUT a clamp, so the default rerank_timeout bypassed the cap (emb 6 + 3 = 9
        # s against a budget of 7.5) — we cap explicitly: min(user value, budget − emb).
        # This also fixes the dead min(10.0, 7.5 − emb) from the 24:10 checklist: the left branch is now
        # the (live) config value, the right one — the budget; 10 s stays the user's upper clamp.
        emb_on = str(config.get("embeddings", "off")).strip().lower() in {"on", "true", "yes", "1"}
        emb = _number(config, "embedding_timeout", 3.0, 0.5, 6.0) if emb_on else 0.0  # as in embeddings.py
        self.timeout = min(_number(config, "rerank_timeout", DEFAULT_TIMEOUT, 0.5, 10.0),
                           _PREFETCH_BUDGET_S - emb)


def settings_from(config: dict) -> Settings | None:
    """Settings when ``rerank`` is on (truthy), else None (the default: no reranker, no calls)."""
    value = config.get("rerank", False)
    on = value is True or str(value).strip().lower() in {"on", "true", "yes", "1"}
    if not on:
        return None
    return Settings(config)


def _local(host: str) -> bool:
    """A host on this machine or the local network: plain http is allowed to it (mirrors embeddings._local)."""
    import ipaddress
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return (host == "localhost" or "." not in host
                or host.endswith((".localhost", ".local", ".lan", ".internal", ".home.arpa")))
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _endpoint(url: str) -> tuple[str | None, bool, str]:
    """(the /rerank address, whether it is local, "") or (None, False, why it is refused)."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return None, False, "rerank_url is not a valid address"
    if parts.scheme not in ("https", "http") or not host:
        return None, False, "rerank_url is not an http(s) address"
    local = _local(host)
    if parts.scheme == "http" and not local:
        return None, False, "rerank_url sends plain http to a public host: use https"
    path = parts.path.rstrip("/")
    if not path.endswith("/rerank"):
        path += "/rerank"
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, "")), local, ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would carry the request to another address: it is a failure instead (mirrors embeddings)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = net.opener(_NoRedirect)


def _pair_document(query: str, document) -> str:
    """The document as it goes out beside *query*: cut to what the query leaves of RERANK_PAIR_CHARS."""
    return str(document)[:RERANK_PAIR_CHARS - len(query)]


class Rescorer:
    """The endpoint: (query, documents) in, ``(index, score)`` pairs out. Shared by every store a provider opens;
    thread-safe. ``scores`` returns None on any failure (the caller keeps the current order)."""

    def __init__(self, settings: Settings, secret_check=None):
        self.settings = settings
        self._secret = secret_check  # callable(text) -> bool, injected to avoid a circular import
        self._url, self._local, self._refused = _endpoint(settings.url)
        if self._url is None:  # _endpoint refuses: keep "" so Request() never sees None
            self._url = ""
        self._lock = threading.Lock()
        self._pause = self._paused_until = 0.0
        self._last_error = ""
        self._epoch = 0
        self.calls = 0  # requests sent (stats, tests)
        self.failures = 0  # counted fallbacks: scores() returned None on a failure (diagnostics)
        # The outcome of the last scores(): answered / failed: ... / skipped_single / skipped_secret / skipped_long /
        # skipped_busy / not_called.
        # Stamped into the runner lines; the run fails on failed (PR #10, 19:07 fix list).
        self.last_status = "not_called"
        # The live worker of the previous request (point 5 of the 24:10 checklist): after join(timeout) the thread
        # lives while the server is still sending the answer; with the pause expired the next call used to start a
        # SECOND request. The same guard as embeddings._abandoned/_reap_locked.
        self._worker: threading.Thread | None = None
        # True while a caller still waits in join() for that worker: a second call then finds a request in flight, not
        # an abandoned one, and must neither fail it nor pause the endpoint (independent review, 2026-10-25).
        self._awaited = False
        # Seconds per character of the pairs, counted from the moment the connection was ready (net: "connected"):
        # one per answered call, and one per call that timed out after connecting (a lower bound: the server had
        # that long and did not answer). fitting() reads the slowest.
        self._speeds: collections.deque = collections.deque(maxlen=_SPEED_SAMPLES)

    def unavailable(self) -> str:
        """Why the endpoint cannot be used at all (\"\" when it can): a refused address."""
        if self._refused:
            return self._refused
        return ""

    def status(self) -> str:
        why = self.unavailable()
        if why:
            return f"unavailable: {why}"
        with self._lock:
            left = self._paused_until - time.monotonic()
            text = (f"paused after an error ({self._last_error}); tried again in {int(left) + 1} s" if left > 0
                    else "ready")
        return text

    def usable(self) -> bool:
        return self.status() == "ready"

    def outcome(self) -> str:
        """last_status without its free text, for the memory journal (0.28.1): answered / skipped_* / not_called as
        they are; a failure is paused (no request: an earlier error's pause), unavailable (a refused address) or
        failed (this call)."""
        status = self.last_status
        if not status.startswith("failed: "):
            return status
        why = status[len("failed: "):]
        return "paused" if why.startswith("paused") else "unavailable" if why.startswith("unavailable") else "failed"

    def note_failure(self, why: str) -> None:
        """A failure OUTSIDE scores() — an exception in the mapping/window of the rerank stage at the provider (point 3
        of the maintainer's 24:10 checklist): the same result as a failed call — the endpoint is
        paused, failures counted and failed in last_status, so the run's fail-fast sees it."""
        self.last_status = f"failed: {why}"
        self._failed(why)

    def scores(self, query: str, documents: list[str], single: bool = False) -> list[tuple[int, float]] | None:
        """``(index, relevance_score)`` pairs of *documents* in descending relevance, top_k at most; None when the
        call must not be made (a secret in *query*), or failed (logged; paused). The caller keeps the current order
        on None; the score is the cross-encoder's own logit (not in [0, 1]) — order-only unless the caller blends.
        *single*: one document is sent all the same — the caller reads its score, not an order (recall_check.py, the
        check of embeddings.RECALL_CHECK_MARGIN)."""
        if not documents:
            self.last_status = "not_called"
            return None
        query = (query or "").strip()  # what is measured below is what is sent
        # The screen runs the WHOLE query, not the cut: a key starting before the 2000th character in [:2000]
        # landed as a fragment ("pas"), passed the screen, while the first letters went to the endpoint — the send
        # stays truncated while the check is gone (maintainer 24:10, point 2; the same bug in main —
        # their PR #21).
        if self._secret and self._secret(query):
            self.last_status = "skipped_secret"
            return None  # no call at all: what recall must not inject is never sent
        if len(query) > RERANK_MAX_QUERY_CHARS:
            # A message this long (a night job's prompt, a pasted log) would not fit the model with a fact beside it.
            # Not a failure, and not one while the endpoint is paused either: the rerank-off order stays.
            self.last_status = "skipped_long"
            return None
        if not self.usable():
            # paused/unavailable: this call did not happen and will not be retried until the pause ends —
            # a failure for the run's zero-failed-calls budget, not a neutral skip.
            self.last_status = f"failed: {self.status()}"
            return None
        if len(documents) == 1 and not single:
            # One sendable document: nothing to reorder, the window cannot drop the only candidate
            # (a cut judged from the top always keeps the top) and the blend of a single score is
            # the same either way — so no call is made and the trivial rank comes back (PR #10, point 9).
            self.last_status = "skipped_single"
            return [(0, 1.0)]
        limit = int(self.settings.top_k)
        body = {"model": self.settings.model, "query": query,
                "documents": [_pair_document(query, d) for d in documents[:self.fitting(query, documents)]],
                "top_n": limit}
        ranked = self._request(body)
        if ranked is _BUSY:
            self.last_status = "skipped_busy"
            return None
        self.last_status = "answered" if ranked is not None else f"failed: {self._last_error or 'unknown'}"
        return ranked

    def fitting(self, query: str, documents: list[str]) -> int:
        """How many of the first *documents* a call sends: top_k at most and, once the endpoint has answered or timed
        out, as many as it scores in RERANK_TIME_SHARE of rerank_timeout at the slowest of its last speeds — two at
        least, one would not be reordered. The provider's stage trace names the documents by the same count."""
        query = (query or "").strip()
        count = min(int(self.settings.top_k), len(documents))
        with self._lock:
            slowest = max(self._speeds, default=0.0)
        if slowest <= 0.0:
            return count
        budget = RERANK_TIME_SHARE * self.settings.timeout / slowest  # characters of pairs
        used = 0
        for i, document in enumerate(documents[:count]):
            used += len(query) + len(_pair_document(query, document))
            if used > budget:
                return max(i, min(2, count))
        return count

    def _request(self, body: dict) -> list[tuple[int, float]] | None:
        """One request. Runs on a daemon thread; the caller waits ``rerank_timeout`` seconds and no longer."""
        timeout = self.settings.timeout
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "hermes-plugin-holographic"}
        request = urllib.request.Request(self._url or "", data=json.dumps(body).encode("utf-8"), headers=headers,
                                         method="POST")
        held = net.wire(request)  # the request's socket, for the caller to end the read on its deadline (issue #26)
        box: dict = {}
        chars = sum(len(body["query"]) + len(d) for d in body["documents"])  # what the endpoint's time grows with

        def send():
            try:
                # The worker cuts off ITSELF: a body that runs over the cap is aborted here, on ITS OWN thread,
                # by its own deadline; `with` closes the connection on that same thread. A caller-side
                # close() waited for the worker in read() — a really slow server sent 21 bytes in
                # 2.0 s and 101 bytes in 10.1 s at rerank_timeout 0.5, and the lock was held the whole turn
                # (19:39 review, point 1) — so close() from the caller is gone, only join(timeout) remains.
                deadline = time.monotonic() + timeout
                with _OPENER.open(request, timeout=timeout) as response:
                    chunks, size = [], 0
                    while time.monotonic() < deadline:
                        chunk = response.read1(65536)
                        if time.monotonic() >= deadline:
                            raise TimeoutError("the answer came too slowly")
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_RESPONSE:
                            raise ValueError("answer too large")
                        chunks.append(chunk)
                    else:
                        raise TimeoutError("the answer came too slowly")
                box["payload"] = json.loads(b"".join(chunks))
            except BaseException as exc:  # noqa: BLE001 - the worker must record every failure; a silent death would burn the whole join() timeout
                box["error"] = exc
            finally:
                box["ended"] = time.perf_counter()

        worker = threading.Thread(target=send, name="holographic-rerank", daemon=True)
        with self._lock:
            if self._paused_until > time.monotonic():
                return None
            if self._worker is not None and self._worker.is_alive() and self._awaited:
                # Another caller's request is in flight and that caller still waits: not a failure. Failing it here
                # bumped the epoch, so the first caller threw away its valid answer, and paused the endpoint.
                return _BUSY
            if self._worker is not None and self._worker.is_alive():
                # point 5: the earlier worker has not finished — the new request does NOT start, and that counts
                # as a failure (the pause grows as with a timeout) until the thread finishes reading the answer.
                noted = self._fail_locked("no answer (an earlier request is still running)")
            else:
                # smaller (19:39 review): checking whether the earlier one is alive, the assignment and the start of
                # the new one — ONE critical section (there were two before, and an assign without a start would read
                # as a dead worker): Rescorer must be safe under concurrent calls on its own, not only because the
                # provider's inputs happen to be serialised today.
                self._worker = worker  # tracked while it lives (point 5)
                self._awaited = True
                noted = None
                epoch = self._epoch
                self.calls += 1
                worker.start()
        if noted is not None:
            self._log_failure(*noted)
            return None
        worker.join(timeout)
        with self._lock:
            self._awaited = False  # from here on a live worker is an abandoned one
        alive = worker.is_alive()
        error = box.get("error")
        # issue #26: the caller's wait is over, so the read it was waiting for ends with it — the socket is shut
        # down from here and the worker dies with the caller instead of reading on (nothing else bounds the header
        # phase). The snapshot above is taken first: every judgement below must describe the wait, not what the
        # shutdown makes the worker raise a moment later.
        if alive:
            net.cut(held)
        timed_out = alive or isinstance(error, TimeoutError) or (
            isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError))
        if timed_out:
            # We do NOT Close() from outside (19:39 review, point 1): it waits for the worker in read() and itself
            # stretches the turn out to the server's body. The worker closes the connection itself (read1 + with
            # on its own thread), and here only the guard remains — while the worker is alive a
            # second request does not start (point 5).
            with self._lock:
                # Both cases pause the endpoint: join() timed out (alive) OR the worker itself ended with
                # TimeoutError — a socket timeout must set the pause and raise failures, otherwise a slow
                # server is polled every turn (PR #10, point 2).
                noted = self._fail_locked("no answer (timeout)")  # both a live join and the worker's own deadline
                connected = held.get("connected")
                if chars and connected is not None and not isinstance(error, urllib.error.URLError):
                    # Connected and not answered by the end of the wait: at least this slow, so the next call
                    # (fitting) sends about half of this one. A timeout before the connection was ready (DNS, TCP,
                    # TLS, a proxy) and a URLError's (sending the request) say nothing of the scoring speed.
                    end = time.perf_counter() if alive else box.get("ended", time.perf_counter())
                    self._speeds.append(max(end - connected, 0.0) / chars)
            self._log_failure(*noted)
            return None
        try:
            if "error" in box:
                raise box["error"]
        except urllib.error.HTTPError as exc:
            self._failed(f"HTTP {exc.code}")
            return None
        except urllib.error.URLError as exc:
            self._failed(f"cannot reach the endpoint ({type(exc.reason).__name__})")
            return None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._failed(f"no answer ({type(exc).__name__})")
            return None
        except Exception as exc:  # noqa: BLE001 - an unlisted endpoint error still counts as a failure and pauses, instead of escaping the turn
            self._failed(f"no answer ({type(exc).__name__})")
            return None
        payload = box.get("payload")
        try:
            ranked = _ranked(payload, len(body["documents"]))
        except ValueError as exc:
            self._failed(f"malformed answer: {exc}")
            return None
        with self._lock:
            if epoch != self._epoch:
                return None  # a failure was counted while this request was in flight
            self._pause = self._paused_until = 0.0
            self._last_error = ""
            connected = held.get("connected")
            if chars and connected is not None:
                self._speeds.append(max(box["ended"] - connected, 0.0) / chars)
        return ranked

    def _fail_locked(self, why: str) -> tuple:
        """Under _lock: count a failure — pause the endpoint, longer each time — and return what to log."""
        first = not self._last_error
        self.failures += 1
        self._pause = min(PAUSE_MAX, self._pause * 2) if self._pause else PAUSE_FIRST
        self._paused_until = time.monotonic() + self._pause
        self._last_error = why
        self._epoch += 1
        return why, first, self._pause

    @staticmethod
    def _log_failure(why: str, first: bool, pause: float) -> None:
        (logger.warning if first else logger.debug)(
            "holographic: rerank endpoint failed (%s); keeping the current order for %d s", why, int(pause))

    def normalized(self, pairs: list[tuple[int, float]]) -> list[float]:
        """Counts for the window in [0,1] — the scale is set explicitly (`rerank_scale`, default
        "logit" for llama-server). Point 3 of the 13:06 review dropped the probing: it could latch
        wrongly, was decided without the lock (a race with a concurrent request) and added a second
        request of up to 2 × rerank_timeout to the turn.

        A one-candidate pool is not normalised: the window keeps it whatever the scale is.
        """
        if len(pairs) < 2:
            return [s for _, s in pairs]
        return normalized(pairs, self.settings.scale)

    def _failed(self, why: str) -> None:
        with self._lock:
            noted = self._fail_locked(why)
        self._log_failure(*noted)


def _sigmoid(x: float) -> float:
    """σ(x) maps into (0,1), numerically stable on both sides of zero."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normalized(pairs: list[tuple[int, float]], scale: str = "logit") -> list[float]:
    """Counts in [0,1] for the window: sigmoid when the scale is "logit"; "unit" as is.
    The scale is the ENDPOINT's property and the user's setting (point 3 of the 13:06 review): explicit,
    default "logit" (llama-server returns raw logits), probing and latching removed —
    Rescorer passes `settings.scale`, direct calls default to logit too.

    δ of the window is measured in [0,1] units: on raw logits the same δ means a different gap than on the
    [0,1] scale, so the scale is brought over explicitly (`rerank_scale`, points 6/3 of the reviews). sigmoid is monotonic — the order
    does not change, only the gaps are compressed; the margin gate, for which the reachability of the ratio was explained
    here, was removed in point 6.
    """
    scores = [s for _, s in pairs]
    if scale == "logit":
        return [_sigmoid(s) for s in scores]
    return scores


def _ranked(payload, count: int) -> list[tuple[int, float]]:
    """``(index, relevance_score)`` pairs from a Cohere-style ``{"results": [{"index": .., "relevance_score": ..}]}``
    answer, in the order given (already descending by relevance), top_n at most; ValueError when it is not one:
    another number of indexes, an index out of range, a non-integer index, a score that is not a number, or a
    duplicate index."""
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        raise ValueError("the answer has no results")
    ranked = []
    for item in results:
        index = item.get("index") if isinstance(item, dict) else None
        if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < count):
            raise ValueError("an index in the answer is out of range")
        score = item.get("relevance_score") if isinstance(item, dict) else None
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("a relevance_score in the answer is not a finite number")
        # math.isfinite(10**400) raises OverflowError (valid JSON, a giant int) — such an
        # answer would have slipped past except ValueError in _request and never become failed (the
        # maintainer's pre-merge checklist, point 1): catch it as not equal to a finite number before calling isfinite.
        if (isinstance(score, int) and abs(score) > 1e308) or not math.isfinite(score):
            raise ValueError("a relevance_score in the answer is not a finite number")
        ranked.append((index, float(score)))
    # The answer's order is NOT trusted: llama-server and Cohere return it sorted, but nothing
    # stops an implementation from returning it shuffled — sorting by score (stable, ties keep the
    # server's order) closes the case (maintainer 24:10, smaller).
    ranked.sort(key=lambda pair: -pair[1])
    if len({i for i, _ in ranked}) != len(ranked):
        raise ValueError("the answer has duplicate indexes")
    # Counting a partial answer (point 4(3) of the 10:34 review): the docstring already promised an error on "a
    # different number of indexes" — this is that check. Fewer indexes than documents in the request
    # means the server did not answer for all of them: an error, not "a smaller pool" (the second case would not pass
    # anyway — an index out of range, — but a short answer with valid indexes is dropped too).
    if len(ranked) != count:
        raise ValueError(f"the answer covers {len(ranked)} of {count} documents")
    return ranked