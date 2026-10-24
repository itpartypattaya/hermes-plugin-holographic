"""Rescoring by a cross-encoder (rerank.py, RFC #9) — optional, offline-safe, off by default.

A fake endpoint stands in for the /v1/rerank server: it answers with a fixed or reversed order of the documents
sent, or fails as told.

One mode since the 2026-10-22 review (п.6): the cross-encoder reorders the pool, admission stays with coverage.
`rerank_window` (δ) optionally cuts the tail below δ of the best score before the top five (coverage ran
earlier, on the lexical stream);
`rerank_weight` > 0 blends the score into the blend. rerank_mode / rerank_margin are gone — a test covers that
they are ignored with a warning. Any failure keeps the current order and pauses the endpoint."""
from __future__ import annotations

import json
import logging
import re
import time

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, rerank
from hermes_plugin_holographic import _bottom_cut, _window_keep, blend_union


class _Response:
    def __init__(self, body: bytes):
        self.body, self.at = body, 0

    def read(self, n=-1):
        piece = self.body[self.at:] if n is None or n < 0 else self.body[self.at:self.at + n]
        self.at += len(piece)
        return piece

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeRerank:
    """Stands in for rerank._OPENER: records every request, answers with a fixed or reversed order, or fails."""

    def __init__(self):
        self.requests: list[dict] = []
        self.fail = None  # an exception to raise, or bytes to answer with
        self.reverse = False  # answer with the documents reversed (worst ranker)
        self.delay = 0.0  # seconds to sleep in the "network" call
        self.scores = None  # optional relevance_scores to answer with (else n, n-1, ...)

    @property
    def documents(self) -> list[str]:
        return [d for r in self.requests for d in r["body"]["documents"]]

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()), "body": body,
                              "timeout": timeout})
        if self.delay:
            time.sleep(self.delay)
        if isinstance(self.fail, BaseException):
            raise self.fail
        if isinstance(self.fail, bytes):
            return _Response(self.fail)
        n = len(body["documents"])
        order = list(reversed(range(n))) if self.reverse else list(range(n))
        top_n = min(body.get("top_n", n), n)
        if self.scores is not None:
            chosen = [{"index": i, "relevance_score": self.scores[i % len(self.scores)]} for i in order[:top_n]]
        else:
            chosen = [{"index": i, "relevance_score": float(n - i)} for i in order[:top_n]]
        return _Response(json.dumps({"results": chosen, "model": body["model"]}).encode())


@pytest.fixture
def endpoint(monkeypatch):
    fake = FakeRerank()
    monkeypatch.setattr(rerank, "_OPENER", fake)
    return fake


RANK = {"rerank": "on", "rerank_url": "https://rerank.example.net/v1/rerank",
        "rerank_model": "fake-reranker"}


def _provider(tmp_path, name="memory_store.db", **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / name), "hrr_dim": 64, **config})
    p.initialize(session_id="s1")
    return p


FACTS = ["The nightly backup runs from the OES box at 2 am.", "Lena's birthday is on June 3.",
         "Mark plays the violin on Sundays.", "Tom is allergic to peanut satay.",
         # п.9: единственный кандидат вызова не делает — endpoint-тесты нуждаются в пуле >= 2
         "The weekly backup runs from the NAS box at 4 am."]
FILLERS = ["apple note", "brick note", "cloud note", "daisy note", "eagle note"]


def _fill(p):
    for f in FILLERS:
        p._store.add_fact(f"{f} number")
    ids = [p._store.add_fact(f) for f in FACTS]
    p.sync_turn("x", "y")
    return ids


def _block(p, query):
    return p.prefetch(query)


# -- off by default -------------------------------------------------------------------------------------------

def test_off_by_default_no_calls_same_order(tmp_path, endpoint):
    p = _provider(tmp_path)
    _fill(p)
    assert p._rescorer is None and endpoint.requests == []
    assert p.prefetch("when does the backup run?") != ""
    assert endpoint.requests == []
    p.shutdown()


def test_config_schema_defaults(tmp_path):
    p = _provider(tmp_path)
    schema = {f["key"]: f.get("default") for f in p.get_config_schema()
              if str(f["key"]).startswith("rerank")}
    assert schema["rerank"] == "off"
    assert schema["rerank_window"] == "0.05"  # п.6: режим один, окно — клавиша
    # п.1+п.3 (13:06): клавиш window_min/window_named больше — минимум два встроен
    # в окно; шкала явная, дефолт logit (llama-server), без auto и пробы
    assert schema["rerank_scale"] == "logit"
    assert "rerank_window_min" not in schema and "rerank_window_named" not in schema
    assert schema["rerank_weight"] == "0.0"
    p.shutdown()


# -- the call: payload shape, order, failure ----------------------------------------------------------------

def test_order_only_reorders_and_sends_cohere_payload(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    block = _block(p, "when does the backup run?")
    assert block  # facts recalled
    req = endpoint.requests[0]
    assert req["body"]["query"]
    assert req["body"]["model"] == "fake-reranker"
    assert isinstance(req["body"]["documents"], list) and req["body"]["documents"]
    assert req["body"]["top_n"] >= 1
    p.shutdown()


def test_window_cut_uses_ranking(tmp_path, endpoint):
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    _fill(p)
    endpoint.scores = [1.0, 0.2]  # a confident top: the margin admits the pool
    block = _block(p, "when does the backup run?")
    assert block
    assert endpoint.requests  # the gate IS the reranker: it was called
    p.shutdown()


def test_bottom_cut_default_keeps_the_window(tmp_path, endpoint):
    """rerank_window cuts the pool FROM THE BOTTOM: everything within δ of the best score survives, the top
    always survives (a window can't reject the winner); recall still works after the cut."""
    assert _bottom_cut([0.99, 0.96, 0.93, 0.5], 0.05) == [0, 1]  # gap 0.03 in, 0.06 and 0.49 out
    assert _bottom_cut([0.4], 0.05) == [0]  # a lone candidate always survives
    assert _bottom_cut([], 0.05) == []
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    _fill(p)
    endpoint.scores = [1.0, 0.9, 0.89, 0.88, 0.87]  # flat tail — the DEFAULT has no ratio veto
    assert _block(p, "what about the note?") != ""  # unlike the enabled veto in the test above
    p.shutdown()


def test_window_keeps_two_scored_and_refills_missing(tmp_path):
    """Встроенный минимум окна (п.1+п.4 ревью 13:06, без клавиш): срез не опускается
    ниже двух scored-кандидатов; добор идёт по позициям РАНГА, которых не хватает —
    keep не обязан быть префиксом (`_window_keep([1], 3)` -> [0, 1], не [1, 1])."""
    assert _window_keep([0], 4) == [0, 1]        # добирает следующий по рангу
    assert _window_keep([1], 3) == [0, 1]        # п.4(4): его пример — keep не префикс
    assert _window_keep([0, 1], 4) == [0, 1]     # уже не меньше двух
    assert _window_keep([0], 1) == [0]           # единственный scored не воскрешаем
    assert _window_keep(None, 4) is None         # выключено -> не оживляем
    assert _window_keep([], 4) == []             # пустой срез не оживляем
    # клавиш больше: Settings их не читает, шкала дефолтная logit (п.3)
    s = rerank.Settings({"rerank": "on"})
    assert not hasattr(s, "window_min") and not hasattr(s, "window_named")
    assert s.scale == "logit"


def test_window_never_cuts_below_two(tmp_path, endpoint):
    """Интеграция (п.1+п.4(1) ревью 13:06; поправка 14:01): 3+ кандидатов и явный assert
    оставшихся — на двух кандидатах тест прошёл бы, даже если окно не срезало ничего."""
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    _fill(p)
    endpoint.scores = [1.0, 0.0, 0.0, 0.0, 0.0]  # 5 заметок: внутри δ только топ
    block = _block(p, "what about the note?")
    req = endpoint.requests[0]["body"]["documents"]
    assert len(req) >= 3  # иначе окно физически нечего срезать
    lines = [ln for ln in block.splitlines() if ln.startswith("- #")]
    assert len(lines) == 2  # окно срезало 5 -> 1, встроенный минимум вернул второй
    kept = [ln.split("] ", 1)[1] for ln in lines]
    assert kept == req[:2]  # остаются ровно первые два по рангу (id видны в строках)
    p.shutdown()


def test_gate_reads_score_falls_back_to_coverage_on_failure(tmp_path, endpoint):
    """A reranker outage must not empty recall: the pipeline falls back to the coverage gate."""
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    _fill(p)
    endpoint.fail = ConnectionRefusedError("refused")
    block = _block(p, "when does the backup run?")
    assert block  # the coverage gate still recalls the fact that answers the words
    p.shutdown()


def test_gate_reads_score_single_candidate_uses_coverage(tmp_path, endpoint):
    """A single candidate goes to the coverage gate like any other pool."""
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    _fill(p)
    # "apple" matches exactly one filler ("apple note number"); the reranker finds it irrelevant (-10)
    calls = []
    original = p._retriever.relevant

    def spy(query, candidates, min_coverage):
        calls.append(len(candidates))
        return original(query, candidates, min_coverage)

    p._retriever.relevant = spy
    endpoint.scores = [-10.0]
    _block(p, "what about apple?")
    assert calls == [1]  # the single candidate went to the coverage gate, not straight through
    p.shutdown()


def test_failure_keeps_current_order_and_pauses(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = ConnectionRefusedError("refused")
    before = p.prefetch("when does the backup run?")
    # backoff: the endpoint is paused, so no further call is made
    calls = len(endpoint.requests)
    # a different session (the repeat window would drop the same fact) and a fresh query
    after = p.prefetch("what does Tom avoid?")
    assert len(endpoint.requests) == calls  # paused: no second request
    assert before and after  # both prefetches worked (order kept, not empty)
    assert p._rescorer.status().startswith("paused")  # the failure is counted
    p.shutdown()


def test_malformed_answer_keeps_order(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = b'{"results": [{"index": 99, "relevance_score": 1.0}]}'
    block = _block(p, "when does the backup run?")
    assert block  # the malformed answer did not take recall down
    assert p._rescorer.status().startswith("paused")  # counted as a failure
    p.shutdown()


def test_huge_score_overflow_counts_as_failed(tmp_path, endpoint):
    """10**400 — валидный JSON: math.isfinite(int) бросает OverflowError, который минуя
    except ValueError в _request убивал вызов в not_called вместо failed — fail-fast прогона
    такой ответ не видел. Теперь это malformed answer, как и любой другой негодный счёт.
    (п.1 pre-merge-чек-листа мейнтейнера, PR #10)."""
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = ('{"results": [{"index": 0, "relevance_score": %d}]}' % (10 ** 400)).encode()
    block = _block(p, "when does the backup run?")
    assert block  # порядок не потерян
    assert p._rescorer.last_status.startswith("failed: malformed answer")
    assert p._rescorer.status().startswith("paused")  # ошибка посчитана
    p.shutdown()


def test_window_keeps_unscored_tail_in_block(tmp_path, endpoint):
    """Окно не теряет факты, которые реранкер не видел: хвост union (не-sendable-факты и позиции за
    top_k) входит в fresh после keep — докстринг _sendable_docs обещает «состав пула для recall не
    меняется». Майнтейнер 24:10, п.1: rerank_top_k=2 при пуле из трёх — третий факт не отправляется
    и обязан пережить окно в БЛОКЕ (проверяется блок, не payload)."""
    p = _provider(tmp_path, **{**RANK, "rerank_top_k": "2"})
    facts = ["The nightly backup runs from the OES box at 2 am.",
             "The weekly backup runs from the NAS box at 4 am.",
             "The weekly backup runs from the tapes in the garage."]
    for f in facts:
        p._store.add_fact(f)
    p.sync_turn("x", "y")
    block = _block(p, "when does the backup run?")
    assert endpoint.requests, "реранк-вызов должен состояться (пул из трёх)"
    sent = endpoint.requests[-1]["body"]["documents"]
    assert len(sent) == 2, "top_k=2: на эндпоинт уходят только два"
    for f in facts:
        assert f in block, f"факт вне ranked обязан пережить окно в блоке: {f[:30]}"
    p.shutdown()


def test_stage_exception_keeps_order_and_counts(tmp_path, endpoint, monkeypatch):
    """Любой exception в реранк-стадии (вызов, маппинг, окно) = неудачный вызов: off-раскладка в
    блоке, пауза эндпоинта, failures == 1, failed в last_status. Раньше exception уходил в обработчик
    prefetch — турн уходил БЕЗ фактов, failures оставался 0, эндпоинт не ставился на паузу.
    Майнтейнер 24:10, п.3 (Codex воспроизводил Thread.start() -> RuntimeError)."""
    p = _provider(tmp_path, **RANK)
    _fill(p)

    def boom(*args, **kwargs):
        raise RuntimeError("Thread.start() exploded")

    monkeypatch.setattr(p._rescorer, "scores", boom)
    block = _block(p, "when does the backup run?")
    assert block, "исключение в стадии не должно уносить блок турна"
    assert "The nightly backup runs from the OES box at 2 am." in block  # off-раскладка, а не пустота
    assert p._rescorer.failures == 1, "сбой стадии обязан считаться отказом"
    assert p._rescorer.status().startswith("paused"), "эндпоинт должен уйти на паузу"
    assert p._rescorer.last_status.startswith("failed: rerank stage")
    p.shutdown()


def test_no_second_worker_while_one_is_alive(tmp_path, endpoint):
    """п.5: после join(timeout) воркер продолжает жить (сервер шлёт по байту в 100 мс — до
    MAX_RESPONSE), а после паузы следующий вызов стартовал ВТОРОЙ. Теперь прежний воркер
    отслеживается: пока он жив, новый запрос не стартует и это считается отказом — минимальный
    вариант, который санкционировал майнтейнер 24:10 (полный аналог embeddings._abandoned/
    _reap_locked при необходимости можно дотянуть отдельно)."""
    p = _provider(tmp_path, **{**RANK, "rerank_timeout": "0.5"})
    _fill(p)
    endpoint.delay = 1.0  # больше клампа 0.5: воркер завис внутри open()
    block = _block(p, "when does the backup run?")
    assert block  # турн не пострадал
    r = p._rescorer
    assert r.failures == 1, "таймаут посчитан"
    assert r._worker is not None and r._worker.is_alive(), "живой воркер должен быть отслежен"
    with r._lock:  # снимаем паузу, чтобы дойти до нового вызова, как после её истечения
        r._pause = r._paused_until = 0.0
    sent = len(endpoint.requests)
    ranked = r.scores("when does the backup run?",
                      ["The nightly backup runs from the OES box at 2 am.",
                       "The weekly backup runs from the NAS box at 4 am."])
    assert ranked is None, "нового ответа быть не должно"
    assert len(endpoint.requests) == sent, "второй запрос не должен стартовать при живом воркере"
    assert r.failures == 2, "отказ при живом воркере обязан считаться"
    r._worker.join(3)  # тест не оставляет нить
    p.shutdown()


def test_secret_straddling_the_2000_cut_is_screened(tmp_path, endpoint):
    """Секрет-экран гоняет ВЕСЬ запрос: ключ, начинающийся перед 2000-м символом, в срез [:2000]
    попадал обрывком («pas») и проходил экран, а первые буквы уходили на эндпоинт. Майнтейнер 24:10,
    п.2 (тот же баг в main — их PR #21). Прямой вызов scores(): через prefetch straddle-запрос
    дополнительно роняет coverage («TopSecret123» как идентификатор), и баг живёт именно в
    последней линии обороны rerank.py."""
    p = _provider(tmp_path, **RANK)
    _fill(p)
    base = "when does the backup run? "
    query = (base + " backup" * 300)[:1996] + " password: TopSecret123"
    assert len(query) > 2000 and query.index("password") < 2000 < query.index("TopSecret")
    docs = ["The nightly backup runs from the OES box at 2 am.",
            "The weekly backup runs from the NAS box at 4 am."]
    ranked = p._rescorer.scores(query, docs)
    assert ranked is None, "обрезанный ключ не должен проходить экран"
    assert p._rescorer.last_status == "skipped_secret"
    assert endpoint.requests == [], "запрос с обрывком ключа не должен уходить"
    p.shutdown()


def test_secret_in_query_no_call(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    block = _block(p, "password: Qwerty123 what is it?")
    assert endpoint.requests == []  # what recall must not inject is never sent
    assert block is not None
    p.shutdown()


def test_timeout_keeps_order(tmp_path, endpoint, monkeypatch):
    """Майнтейнер 24:10: старая версия не ловила таймаут — rerank_timeout 0.2 клампится до 0.5, а
    delay был ровно 0.5, и в его трёх прогонах статус был answered. Теперь delay задолго выше
    клампа, и тест ловит сам таймаут: пауза + failures == 1, а не только живость блока."""
    p = _provider(tmp_path, **{**RANK, "rerank_timeout": "0.2"})
    _fill(p)
    endpoint.delay = 2.0  # well above the 0.5 clamp (0.2 -> 0.5)
    start = time.perf_counter()
    block = _block(p, "when does the backup run?")
    elapsed = time.perf_counter() - start
    assert block  # recall did not hang
    assert elapsed < 1.0  # the caller waited rerank_timeout (clamped to 0.5), not the delay
    r = p._rescorer
    assert r.failures == 1, "таймаут обязан считаться отказом"
    assert r.status().startswith("paused"), "эндпоинт должен уйти на паузу"
    assert r.last_status.startswith("failed")
    if r._worker is not None:
        r._worker.join(3)  # воркер доспивает свой delay — тест не оставляет нить
    p.shutdown()


# -- blend (weight > 0) --------------------------------------------------------------------------------------

def test_weight_zero_is_order_only(tmp_path, endpoint):
    p = _provider(tmp_path, **{**RANK, "rerank_weight": "0.0"})
    _fill(p)
    _block(p, "when does the backup run?")
    assert endpoint.requests  # still called: order-only is the default
    p.shutdown()


def test_weight_blend_runs_and_keeps_recall(tmp_path, endpoint):
    p = _provider(tmp_path, **{**RANK, "rerank_weight": "0.5"})
    _fill(p)
    block = _block(p, "when does the backup run?")
    assert block and endpoint.requests  # blend recomputes score and sorts; recall still works
    p.shutdown()


# -- the endpoint's own API ---------------------------------------------------------------------------------

def test_settings_validation():
    assert rerank.settings_from({}) is None
    assert rerank.settings_from({"rerank": "off"}) is None
    s = rerank.Settings({"rerank": "on", "rerank_top_k": "15", "rerank_window": "0.1"})
    assert (s.top_k, s.window) == (15, 0.1)
    assert rerank.Settings({"rerank": "on"}).window == rerank.DEFAULT_WINDOW  # дефолт окна

    # п.4 (майнтейнер 24:10): бюджет ядра — prefetch провайдера 8 с, embedding_timeout + 
    # rerank_timeout обязаны укладываться меньше него при любых допустимых значениях
    # (emb клампится 0.5..6, rerank раньше зажимался 0.5..10 независимо).
    assert rerank.Settings({"rerank": "on", "embedding_timeout": "6", "rerank_timeout": "10"}).timeout == 1.5
    assert rerank.Settings({"rerank": "on", "embedding_timeout": "0.5", "rerank_timeout": "10"}).timeout == 7.0
    assert rerank.Settings({"rerank": "on"}).timeout == 3.0  # дефолт 3 + дефолт emb 3 = 6 влезает


def test_removed_keys_are_ignored_with_a_warning(caplog):
    """rerank_mode / rerank_margin удалены (п.6): старая конфигурация не ломает ничего, ключ игнорируется."""
    with caplog.at_level(logging.WARNING):
        s = rerank.Settings({"rerank": "on", "rerank_mode": "gate_reads_score", "rerank_margin": "1.5"})
    assert s.window == rerank.DEFAULT_WINDOW
    assert "rerank_mode was removed" in caplog.text and "rerank_margin was removed" in caplog.text


def test_ranked_shape():
    full = {"results": [{"index": 2, "relevance_score": -0.239},
                        {"index": 1, "relevance_score": -1.5},
                        {"index": 0, "relevance_score": -2.877}]}
    assert rerank._ranked(full, 3) == [(2, -0.239), (1, -1.5), (0, -2.877)]
    # порядок ответа не доверяется: перепутанный список сервера сортируется по счёту
    shuffled = {"results": [{"index": 1, "relevance_score": -1.5},
                            {"index": 2, "relevance_score": -0.239},
                            {"index": 0, "relevance_score": -2.877}]}
    assert rerank._ranked(shuffled, 3) == [(2, -0.239), (1, -1.5), (0, -2.877)]
    # п.4(3): частичный ответ (меньше индексов, чем документов) — ошибка, а не «пул поменьше»:
    # запрос всегда шлётся с top_n >= числом документов, сервер обязан ответить на все.
    with pytest.raises(ValueError):
        rerank._ranked({"results": [{"index": 2, "relevance_score": -0.239},
                                     {"index": 0, "relevance_score": -2.877}]}, 5)  # 2 из 5
    with pytest.raises(ValueError):
        rerank._ranked({"results": [{"index": 9, "relevance_score": 1.0}]}, 5)  # out of range
    with pytest.raises(ValueError):
        rerank._ranked({"results": [{"index": 0, "relevance_score": "x"}]}, 5)  # not a number
    with pytest.raises(ValueError):
        # 10**400 — валидный JSON; math.isfinite(int) бросает OverflowError, минуя except
        # ValueError в _request — до фикса такой ответ не считался failed (п.1 чек-листа).
        rerank._ranked({"results": [{"index": 0, "relevance_score": 10 ** 400}]}, 5)  # переполнение isfinite
    with pytest.raises(ValueError):
        rerank._ranked({"results": []}, 5)  # empty


def test_endpoint_validation():
    local = rerank._endpoint("http://127.0.0.1:8080/v1/rerank")
    assert local[1] is True  # local
    assert rerank._endpoint("http://example.com/v1/rerank")[0] is None  # public http refused
    assert rerank._endpoint("ftp://x")[0] is None  # wrong scheme


# -- фолбэк = off, шкала эндпоинта (п.3 ревью 13:06), blend в одной шкале ------------------------------

def test_scale_is_explicit_and_never_probes(tmp_path, endpoint):
    """П.3 ревью 13:06: шкала явная (дефолт logit), без латча и без пробы — ни одного
    лишнего запроса: проба могла залатчить неверно, решалась без лока (гонка) и
    добавляла турну второй запрос до 2 × rerank_timeout. Ответ в [0,1] шкалу не
    переворачивает — читается той же."""
    p = _provider(tmp_path, **{**RANK, "rerank_window": "0.05"})
    rescorer = p._rescorer
    assert rescorer.settings.scale == "logit"
    first = rescorer.normalized([(0, 5.0), (1, -3.0)])  # логиты -> сигмоида
    assert first == pytest.approx([rerank._sigmoid(5.0), rerank._sigmoid(-3.0)])
    second = rescorer.normalized([(0, 0.9), (1, 0.7)])  # в [0,1] — та же шкала
    assert second == pytest.approx([rerank._sigmoid(0.9), rerank._sigmoid(0.7)])
    assert endpoint.requests == []  # пробы не было и не будет
    p.shutdown()


def test_scale_unit_is_literal_and_bad_value_names_the_key(tmp_path):
    """П.2+п.3 (13:06): unit уважается буквально; значение вне {unit, logit} —
    именованная проблема конфига (memory off), а не тихий fallback на auto."""
    unit = _provider(tmp_path, "unit.db", **{**RANK, "rerank_scale": "unit"})
    assert unit._rescorer.settings.scale == "unit"
    assert unit._rescorer.normalized([(0, -5.0), (1, -3.0)]) == [-5.0, -3.0]  # форс unit: как есть
    unit.shutdown()
    bad = _provider(tmp_path, "bad.db", **{**RANK, "rerank_scale": "banana"})
    assert any("rerank_scale" in problem for problem in bad._config_problems)
    bad.shutdown()


def test_blend_union_scores_every_member_on_one_scale():
    """п.4(4): при rerank_weight > 0 смесь строится для ВСЕХ членов пула: не скорированные
    (за top_k) получают пол min-max-шкалы, а не остаются на голом первом скоре — иначе
    union сортируется по двум шкалам (вес-тест покрывал только полностью скорированный пул)."""
    union = [{"score": 0.4}, {"score": 0.6}, {"score": 0.9}, {"score": 0.8}, {"score": 0.5}]
    out = blend_union(union, [(0, 1.0), (1, 0.0)], 0.5)  # эндпоинт скорировал только два
    # 0: 0.5*0.4+0.5*1.0 = 0.7 (нормированный максимум), 1: 0.5*0.6+0.5*0 = 0.3,
    # 2-4: 0.5*orig + 0 (пол шкалы) -> 0.45 / 0.4 / 0.25
    assert [round(f["score"], 3) for f in out] == [0.7, 0.45, 0.4, 0.3, 0.25]
    # Старое поведение оставляло 2-4 на голых 0.9/0.8/0.5 — они бы встали впереди 0.7.


def test_failed_call_shows_exactly_the_off_block(tmp_path, endpoint):
    """п.4(1): фолбэк упавшего вызова = off-раскладка, включая факт, который entity-поток
    добавляет в off за пределами пяти сохранённых лексических (раньше он в фолбэке терялся:
    дополнения считались от всей лексики)."""
    lena_facts = ["Lena's birthday is on June 3.", "Lena drinks tea with lemon.",
                  "Lena works at the archive on Mondays.", "Lena's brother is Andrey.",
                  "Lena dislikes strong coffee.", "Lena runs the book club in the library."]

    def fill(p):
        for f in FILLERS:
            p._store.add_fact(f"{f} number")
        for text in lena_facts:
            p._store.add_fact_status(text, entities=("Lena",))  # линк задаём явно (харнесс)

    off = _provider(tmp_path, "off.db")
    fill(off)
    down = _provider(tmp_path, "down.db", **RANK)
    fill(down)
    endpoint.fail = ConnectionRefusedError("refused")
    off_block = _block(off, "what about Lena?")
    down_block = _block(down, "what about Lena?")
    endpoint.fail = None
    assert off_block == down_block  # блок при сбое ровно тот, что показал бы выключенный rerank
    assert len(re.findall(r"- #\d+", off_block)) == 6  # пять лексических + шестой из entity-потока
    off.shutdown()
    down.shutdown()
