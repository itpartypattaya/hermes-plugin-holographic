"""Second-stage rescoring: optional, offline-safe, off by default. A Cohere-compatible `/v1/rerank` endpoint (the
de-facto standard; OpenAI has no rerank API at all) reorders the final candidate pool after the blend. No state of any
kind — one synchronous HTTP call at recall time.

The endpoint is served natively by llama-server (`--reranking --embedding --pooling rank`), TEI, vLLM, Infinity, Jina,
Cohere and LiteLLM — one stdlib client covers the local and the cloud case.

Guardrails (from RFC #9):
- off by default (``rerank: false``); behaviour and scores exactly 0.13.4's when off;
- a query holding a secret (``_secret_in``) → no call at all;
- any failure leaves the list in its current order, one debug line of the error category, the endpoint goes to
  backoff 60 s doubling to 900 s (_PAUSE_FIRST / _PAUSE_MAX);
- ``rerank_timeout`` defaults to 3.0 (issue #5: a slow local endpoint is the normal case; this is recall's only
  protection, there is no catch-up);
- the reranker judges the pair's text only — no hrr_vector / trust / decay in the request.

Design after itpartypattaya/hermes-plugin-holographic#9 by KostinPavel."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

PAUSE_FIRST, PAUSE_MAX = 60.0, 900.0  # seconds the endpoint is left alone after a failure, doubling
MAX_RESPONSE = 64 << 20
DEFAULT_TOP_K = 40  # depth of the pool the cross-encoder sees; the first stage's limit is widened to match
DEFAULT_TIMEOUT = 3.0  # seconds; 3.0, not 2.0: a slow local endpoint is the normal case (issue #5)

#: Бюджет ядра на prefetch провайдера: _EXTERNAL_PREFETCH_TIMEOUT_S в agent/memory_manager.py.
#: Embedding-вызов и реранк идут подряд, поэтому сумма их клампов обязана укладывается меньше
#: него с запасом (майнтейнер 24:10, п.4: дефолты 3 + 3 влезают, emb 6 + rerank 10 — нет).
_PREFETCH_BUDGET_S = 7.5
DEFAULT_WINDOW = 0.05  # rerank_window: доля хвоста ниже которой пул отсекается (0 = ничего не отсекать)


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
        # rerank_mode / rerank_margin удалены (п.6 ревью 2026-10-22): один режим, окно - отдельная клавиша.
        for gone, replacement in (("rerank_mode", "one mode; rerank: on is the switch"),
                                  ("rerank_margin", "rerank_window instead")):
            if config.get(gone) is not None and str(config.get(gone)).strip():
                logger.warning("holographic: %s was removed (%s); ignoring it", gone, replacement)
        self.url = str(config.get("rerank_url") or "http://127.0.0.1:8080/v1/rerank").strip()
        self.model = str(config.get("rerank_model") or "local-reranker").strip()
        self.top_k = _number(config, "rerank_top_k", DEFAULT_TOP_K, 1, 1000, int)  # int (kind=int)
        self.weight = _number(config, "rerank_weight", 0.0, 0.0, 1.0)
        self.window = _number(config, "rerank_window", DEFAULT_WINDOW, 0.0, 1.0)
        # шкала скоров — свойство ЭНДПОИНТА и настройка пользователя (п.3 ревью
        # 13:06): явная, без пробы и латча; дефолт logit — llama-server отдаёт
        # сырые логиты. Набор значений держит _CHOICES у провайдера: вне его —
        # именованная проблема конфига и memory off, а не тихий fallback.
        self.scale = str(config.get("rerank_scale") or "logit").strip().lower()
        emb = _number(config, "embedding_timeout", 3.0, 0.5, 6.0)  # кламп как в embeddings.py
        self.timeout = _number(config, "rerank_timeout", DEFAULT_TIMEOUT, 0.5,
                               min(10.0, _PREFETCH_BUDGET_S - emb))


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


_OPENER = urllib.request.build_opener(_NoRedirect)


class Rescorer:
    """The endpoint: (query, documents) in, indexes in descending relevance order out. Shared by every store a
    provider opens; thread-safe. ``order`` returns None on any failure (the caller keeps the current order)."""

    def __init__(self, settings: Settings, secret_check=None, environ=None):
        import os
        self.settings = settings
        self._secret = secret_check  # callable(text) -> bool, injected to avoid a circular import
        self._environ = os.environ if environ is None else environ
        self._url, self._local, self._refused = _endpoint(settings.url)
        if self._url is None:  # _endpoint refuses: keep "" so Request() never sees None
            self._url = ""
        self._lock = threading.Lock()
        self._pause = self._paused_until = 0.0
        self._last_error = ""
        self._epoch = 0
        self.calls = 0  # requests sent (stats, tests)
        self.failures = 0  # counted fallbacks: scores() returned None on a failure (diagnostics)
        # Исход последнего scores(): answered / failed: ... / skipped_single / skipped_secret / not_called.
        # Стампится в строки раннеров; прогон падает на failed (PR #10, фикс-лист 19:07).
        self.last_status = "not_called"
        # Живой воркер прошлого запроса (п.5 чек-листа 24:10): после join(timeout) нить живёт,
        # пока сервер шлёт ответ; при истёкшей паузе следующий вызов раньше стартовал ВТОРОЙ
        # запрос. Страховка наравне с embeddings._abandoned/_reap_locked.
        self._worker: threading.Thread | None = None

    def unavailable(self) -> str:
        """Why the endpoint cannot be used at all (\"\" when it can): the address, the key."""
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

    def note_failure(self, why: str) -> None:
        """Сбой ВНЕ scores() — исключение в маппинге/окне реранк-стадии у провайдера (п.3
        чек-листа мейнтейнера 24:10): тот же результат, что у неудачного вызова — пауза
        эндпоинта, счётчик failures и failed в last_status, чтобы fail-fast прогона это видел."""
        self.last_status = f"failed: {why}"
        self._failed(why)

    def order(self, query: str, documents: list[str]) -> list[int] | None:
        """Indexes of *documents* in descending relevance to *query*, top_k at most, or None: unusable, a secret in
        *query*, or a request failed (logged; paused). The caller keeps the current order on None."""
        ranked = self.scores(query, documents)
        return [i for i, _ in ranked] if ranked is not None else None

    def scores(self, query: str, documents: list[str]) -> list[tuple[int, float]] | None:
        """``(index, relevance_score)`` pairs of *documents* in descending relevance, top_k at most; None when the
        call must not be made (a secret in *query*), or failed (logged; paused). The caller keeps the current order
        on None; the score is the cross-encoder's own logit (not in [0, 1]) — order-only unless the caller blends."""
        if not documents:
            self.last_status = "not_called"
            return None
        if not self.usable():
            # paused/unavailable: this call did not happen and will not be retried until the pause ends —
            # a failure for the run's zero-failed-calls budget, not a neutral skip.
            self.last_status = f"failed: {self.status()}"
            return None
        # Экран гоняет ВЕСЬ запрос, не срез: ключ, начинающийся перед 2000-м символом, в [:2000]
        # попадал обрывком («pas»), проходил экран, а первые буквы уходили на эндпоинт — отправка
        # остаётся срезанной, а проверка больше нет (майнтейнер 24:10, п.2; тот же баг в main —
        # их PR #21).
        if self._secret and self._secret(query or ""):
            self.last_status = "skipped_secret"
            return None  # no call at all: what recall must not inject is never sent
        if len(documents) == 1:
            # One sendable document: nothing to reorder, the window cannot drop the only candidate
            # (a cut judged from the top always keeps the top) and the blend of a single score is
            # the same either way — so no call is made and the trivial rank comes back (PR #10, point 9).
            self.last_status = "skipped_single"
            return [(0, 1.0)]
        limit = int(self.settings.top_k)
        docs = [documents[i] for i in range(min(limit, len(documents)))]
        body = {"model": self.settings.model, "query": (query or "")[:2000],
                "documents": [str(d)[:2000] for d in docs],
                "top_n": limit}
        ranked = self._request(body)
        self.last_status = "answered" if ranked is not None else f"failed: {self._last_error or 'unknown'}"
        return ranked

    def _request(self, body: dict) -> list[tuple[int, float]] | None:
        """One request. Runs on a daemon thread; the caller waits ``rerank_timeout`` seconds and no longer."""
        timeout = self.settings.timeout
        with self._lock:
            if self._paused_until > time.monotonic():
                return None
            if self._worker is not None and self._worker.is_alive():
                # п.5: прежний воркер ещё не закончил — новый запрос НЕ стартует, и это
                # отказ (пауза растёт, как от таймаута), пока нить не дочитает ответ.
                noted = self._fail_locked("no answer (an earlier request is still running)")
            else:
                self._worker = None  # старый дожил или его не было
                noted = None
                epoch = self._epoch
                self.calls += 1
        if noted is not None:
            self._log_failure(*noted)
            return None
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "hermes-plugin-holographic"}
        request = urllib.request.Request(self._url or "", data=json.dumps(body).encode("utf-8"), headers=headers,
                                         method="POST")
        box: dict = {}

        def send():
            try:
                with _OPENER.open(request, timeout=timeout) as response:
                    box["response"] = response  # main закроет его по таймауту (п.5)
                    raw = response.read(MAX_RESPONSE)
                box["payload"] = json.loads(raw)
            except BaseException as exc:  # noqa: BLE001 - the worker must record every failure; a silent death would burn the whole join() timeout
                box["error"] = exc
            finally:
                box["ended"] = time.perf_counter()

        worker = threading.Thread(target=send, name="holographic-rerank", daemon=True)
        with self._lock:
            self._worker = worker  # отслеживается, пока жив (п.5)
        worker.start()
        worker.join(timeout)
        alive = worker.is_alive()
        error = box.get("error")
        timed_out = alive or isinstance(error, TimeoutError) or (
            isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError))
        if timed_out:
            # п.5: рвём соединение — воркер-читатель просыпается с ошибкой и умирает вместо
            # того, чтобы дочитывать байты до MAX_RESPONSE. Best effort: закрытие из другого
            # потока не гарантирует пробуждение recv (platform-dependent), поэтому ниже остаётся
            # guard — пока воркер жив, второй запрос не стартует.
            closer = getattr(box.get("response"), "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - best effort, платформозависимо
                    pass
            with self._lock:
                # Both cases pause the endpoint: join() вышел по таймауту (alive) ИЛИ сам воркер закончился
                # TimeoutError — сокетный таймаут должен ставить паузу и расти failures, иначе медленный
                # сервер опрашивается каждый turn (PR #10, п.2).
                noted = self._fail_locked("no answer (timeout)" if alive else "no answer (socket timeout)")
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
        """Счётчики для окна в [0,1] — шкала задана явно (`rerank_scale`, дефолт
        "logit" под llama-server). П.3 ревью 13:06 убрал пробу: она могла залатчить
        неверно, решалась без лока (гонка с конкурентным запросом) и добавляла турну
        второй запрос до 2 × rerank_timeout.

        Пул из одного кандидата не нормируется: окно сохранит его при любой шкале.
        """
        if len(pairs) < 2:
            return [s for _, s in pairs]
        return normalized(pairs, self.settings.scale)

    def _failed(self, why: str) -> None:
        with self._lock:
            noted = self._fail_locked(why)
        self._log_failure(*noted)


def _sigmoid(x: float) -> float:
    """σ(x) в (0,1), численно устойчивый на обеих сторонах нуля."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normalized(pairs: list[tuple[int, float]], scale: str = "logit") -> list[float]:
    """Счётчики в [0,1] для окна: sigmoid, когда шкала "logit"; "unit" — как есть.
    Шкала — СВОЙСТВО ЭНДПОИНТА и настройка пользователя (п.3 ревью 13:06): явная,
    дефолт "logit" (llama-server отдаёт сырые логиты), проба и латч удалены —
    Rescorer передаёт `settings.scale`, прямые вызовы по умолчанию тоже logit.

    δ окна измеряется в единицах [0,1]: на сырых логитах тот же δ соответствует другому зазору, чем на
    шкале [0,1], поэтому шкалу приводят явно (`rerank_scale`, п.6/п.3 ревью). sigmoid монотонен — порядок
    не меняется, сжимаются только зазоры; margin-гейт, для которого тут объяснялась достижимость отношения,
    удалён в п.6.
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
        # math.isfinite(10**400) бросает OverflowError (валидный JSON, гигантский int) — такой
        # ответ миновал бы except ValueError в _request и не стал бы failed (pre-merge-чек-лист
        # мейнтейнера, п.1): ловим как неравный конечному числу до вызова isfinite.
        if (isinstance(score, int) and abs(score) > 1e308) or not math.isfinite(score):
            raise ValueError("a relevance_score in the answer is not a finite number")
        ranked.append((index, float(score)))
    # Порядок ответа НЕ доверяем: llama-server, TEI и Cohere отдают отсортированное, но ничего
    # не мешает отдать перепутанное — сортировка по счёту (стабильная, равные — серверный
    # порядок) закрывает кейс (майнтейнер 24:10, smaller).
    ranked.sort(key=lambda pair: -pair[1])
    if len({i for i, _ in ranked}) != len(ranked):
        raise ValueError("the answer has duplicate indexes")
    # Счёт частичного ответа (п.4(3) ревью 10:34): докстринг уже обещал ошибку на «другое
    # число индексов» — теперь это и есть проверка. Меньше индексов, чем документов в запросе,
    # — сервер ответил не на всё: ошибка, а не «пул поменьше» (второе по счёту не пройдёт —
    # индекс out of range, — но короткий ответ валидными индексами тоже отбрасываем).
    if len(ranked) != count:
        raise ValueError(f"the answer covers {len(ranked)} of {count} documents")
    return ranked