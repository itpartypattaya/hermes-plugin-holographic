"""PR #10, пункты ревью 2026-10-22 (1, 2, 4, 5) — тесты, которые раньше не проходили бы.

1. Факт, который recall МОЖЕТ показывать, но отправлять нельзя (_sendable < _recallable):
   `OPENAI_API_KEY=*** / `Authorization: Bearer *** — не должен попасть в documents
   ни в одном потоке prefetch, ни в search (раньше search не фильтрововал вообще).
2. Сокет-таймаут воркера (воркер уже закончился, alive=False) обязан ставить паузу
   и расти `failures` — иначе медленный сервер опрашивается каждый turn.
4. Упавший реранк в `_unified_stream` отдаёт в точности раскладку off: lexical[:5],
   затем meaning/entity-дополнения, а не (lexical+…+…)[:5], который их вытесняет.
   Плюс `_search`: хвост limit > top_k сохраняет место ПОСЛЕ ранжированных.
5. `rerank: on` сам по себе включает единственный режим; ключи `rerank_mode`/`rerank_margin` удалены (п.6) — старая конфигурация игнорируется с предупреждением.
"""
from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, rerank, _sendable


# -- fake endpoint -----------------------------------------------------------------------------------------------


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
    """Records every request (documents assertable), answers identity/reversed, or fails as told."""

    def __init__(self):
        self.requests: list[dict] = []
        self.fail = None
        self.reverse = False

    @property
    def documents(self) -> list[str]:
        return [d for r in self.requests for d in r["body"]["documents"]]

    def open(self, request, timeout=None):
        body = json.loads(request.data)
        self.requests.append({"url": request.full_url, "body": body, "timeout": timeout})
        if isinstance(self.fail, BaseException):
            raise self.fail
        n = len(body["documents"])
        order = list(reversed(range(n))) if self.reverse else list(range(n))
        top_n = min(body.get("top_n", n), n)
        chosen = [{"index": i, "relevance_score": float(n - i)} for i in order[:top_n]]
        return _Response(json.dumps({"results": chosen}).encode())


@pytest.fixture
def endpoint(monkeypatch):
    fake = FakeRerank()
    monkeypatch.setattr(rerank, "_OPENER", fake)
    return fake


# -- store --------------------------------------------------------------------------------------------------------

RANK = {"rerank": "on",
        "rerank_url": "https://rerank.example.net/v1/rerank", "rerank_model": "fake-reranker"}
QUERY = "when does the backup run?"
# Нейтральные fillers: без них IDF-вес «backup» почти во всех фактах роняет покрытие relevant()
# ниже порога на этом маленьком сторе (проверено: 0.3 не проходит; с fillers — проходит).
FILLERS = ["apple note", "brick note", "cloud note", "daisy note", "eagle note"]
PLAIN = ["The nightly backup runs from the OES box at 2 am.",
         "The backup rotation keeps seven days of archives.",
         "Mark plays the violin on Sundays.",
         # п.9: единственный кандидат вызова не делает — endpoint-тесты нуждаются в пуле >= 2
         "The weekly backup runs from the NAS box at 4 am."]
# Литералы-«ключи» собираются в рантайме: готовый похожий на ключ текст редактируется при записи файла.
_BEARER = "ey" + "JhbGciOiJIUzI1NiJ9." + "abc123DEF456" * 3
_KEY = "sk-" + "proj-" + "aB3" * 8 + "z9"
assert len(_BEARER) >= 12 and len(_KEY) >= 6  # санити против плейсхолдерной правки
# recallable=True, sendable=False (п.1): recall её показывает, эндпоинт не получает
# (форма Authorization-заголовка — то, что _recall_problem не считает credential-with-value)
_AUTH_HEAD = "The proxy sets Authorization:" + " " + "Bearer" + " "  # фрагменты: литерал не редактируется
SECRET = _AUTH_HEAD + _BEARER + " for backup requests."
# вообще не recallable (credential with its value): search показывает, recall и эндпоинт — нет
KEY_FACT = f"The backup box stores OPENAI_API_KEY={_KEY} in /etc."
VIOLET = "The backup password: violet-quilt-77."


def _provider(tmp_path, name="memory.db", **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / name), "hrr_dim": 64, **config})
    p.initialize(session_id="s1")
    return p


def _fill(p, extra=()):
    for text in FILLERS + list(PLAIN) + list(extra):
        p._store.add_fact(text)
    p.sync_turn("x", "y")


def _without_rerank(tmp_path, query, extra=(), name="off.db") -> str:
    p = _provider(tmp_path, name=name)
    _fill(p, extra)
    try:
        return p.prefetch(query)
    finally:
        p.shutdown()


# -- 1. секрет-экран на обоих путях -------------------------------------------------------------------------------


def test_secret_fact_is_never_in_the_payload(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p, extra=[SECRET, KEY_FACT, VIOLET])
    # посылка п.1: recallable, но не sendable — и на эндпоинт не уходит
    assert _sendable(SECRET) is False
    assert p._recallable({"content": SECRET, "trust_score": 0.9, "fact_id": 1}) is True
    # контраст классов: credential-with-value отсекает уже recall-фильтр (и он тоже не отправляется)
    assert p._recallable({"content": KEY_FACT, "trust_score": 0.9, "fact_id": 2}) is False
    block = p.prefetch(QUERY)
    assert endpoint.requests, "rerank must actually run — the screen must not switch the feature off"
    sent = endpoint.documents
    assert any("nightly backup" in d for d in sent)          # обычные факты уходят
    assert all(_BEARER not in d for d in sent)               # _sendable-факт — нет
    assert all("OPENAI_API_KEY" not in d for d in sent)      # не recallable — тем более нет
    assert all("violet-quilt" not in d for d in sent)
    assert "nightly backup" in block                          # recall работает
    p.shutdown()


def test_search_screens_documents_but_still_shows_the_fact(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p, extra=[SECRET, KEY_FACT, VIOLET])
    res = json.loads(p.handle_tool_call("fact_store", {"action": "search", "query": "backup"}))
    assert endpoint.requests, "search runs the rescorer when it is on"
    assert all(_BEARER not in d for d in endpoint.documents)
    assert all("OPENAI_API_KEY" not in d for d in endpoint.documents)
    assert all("violet-quilt" not in d for d in endpoint.documents)
    # показать — может (search намеренно показывает то, recall не инжектит); отправить — нет
    shown = [r.get("content", "") for r in res.get("results", [])]
    assert any("OPENAI_API_KEY" in c for c in shown)
    p.shutdown()


# -- 2. сокет-таймаут ставит паузу -------------------------------------------------------------------------------


def test_socket_timeout_pauses_the_endpoint(tmp_path, endpoint):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = TimeoutError("timed out")  # воркер закончился сам: alive=False
    block = p.prefetch(QUERY)
    assert block == _without_rerank(tmp_path, QUERY)   # порядок off-пути
    assert len(endpoint.requests) == 1                 # один вызов, дальше пауза
    assert p._rescorer.failures == 1
    assert p._rescorer.status().startswith("paused")
    p.shutdown()


# -- 4a. фолбэк unified = раскладка off --------------------------------------------------------------------------


def test_failed_rerank_appends_extras_like_off(tmp_path, endpoint, monkeypatch):
    p = _provider(tmp_path, **RANK)
    _fill(p)
    endpoint.fail = urllib.error.URLError(ConnectionResetError("reset"))
    # синтетические потоки meaning и entity (embeddings в тесте off): факты лежат в store,
    # возвращаем их реальные dict'ы через search
    m_id = p._store.add_fact("Zebra paraphrase holds no shared word.")
    e_id = p._store.add_fact("Quartz wall fact arrives by entity only.")
    p.sync_turn("x", "y")
    m_fact = p._retriever.search("zebra paraphrase", limit=1)[0]
    e_fact = p._retriever.search("quartz wall", limit=1)[0]
    p._embedding = SimpleNamespace(recall_facts=1, min_score=0.0)
    monkeypatch.setattr(p._retriever, "by_meaning", lambda *a, **k: [m_fact])
    monkeypatch.setattr(p._store, "mentioned_entity_ids", lambda q: {1})  # известные сущности — set
    monkeypatch.setattr(p._retriever, "names_fit", lambda *a, **k: True)
    monkeypatch.setattr(p._retriever, "entity_facts", lambda *a, **k: [e_fact])
    fresh, by_meaning = p._unified_stream(QUERY, object(), list(p._retriever.search(QUERY, limit=10)), "s", 1)
    # раскладка off: lexical[:5], потом extras — extras НЕ вытесняются top-5 (п.4)
    assert [f["fact_id"] for f in fresh[-2:]] == [m_id, e_id]
    lexical_part = [f for f in fresh if f["fact_id"] not in (m_id, e_id)]
    assert len(lexical_part) <= 5
    assert by_meaning == {m_id}
    p.shutdown()


# -- 4b. хвост limit в search ------------------------------------------------------------------------------------


def test_search_keeps_the_unscored_tail(tmp_path, endpoint):
    endpoint.reverse = True          # ранжированные идут задом наперёд — маппинг виден
    facts = [f"note {i} about the backup window" for i in range(45)]

    def build(name, **cfg):
        p = _provider(tmp_path, name=name, **cfg)
        for text in facts:
            p._store.add_fact(text)
        p.sync_turn("x", "y")
        return p

    # window 0: тест изолирует ИНВАРИАНТ хвоста (45 из 45, хвост в исходном порядке), не окно —
    # с sort (24:10) ответ fake по счёту восстанавливает off-порядок, а δ-срез сверху здесь
    # добавил бы хрупкую зависимость от сигмоидных порогов.
    p = build("tail.db", **{**RANK, "rerank_window": "0"})
    res = json.loads(p.handle_tool_call("fact_store", {"action": "search", "query": "backup", "limit": 45}))
    contents = [r.get("content", "") for r in res.get("results", [])]
    p.shutdown()
    # эталон: тот же магазин без реранка — порядок до прикосновения эндпоинта
    off = build("tail_off.db")
    base = [r.get("content", "")
            for r in json.loads(off.handle_tool_call("fact_store",
                                                      {"action": "search", "query": "backup",
                                                       "limit": 45})).get("results", [])]
    off.shutdown()
    # sort (24:10) даёт по счёту: отранжированные 40 возвращаются в off-порядок fake'а,
    # а 5 неоценённых хвостом переживают всё в исходном порядке
    assert len(contents) == 45, f"tail dropped: {len(contents)} of 45"
    assert contents == base  # 40 отранжированных + хвост без потерь


# -- 5. дефолт режима = UI ---------------------------------------------------------------------------------------


def test_rerank_on_alone_is_enough():
    """п.6: один режим — rerank: on включает его без каких-либо режимных ключей."""
    settings = rerank.settings_from({"rerank": "on"})
    assert settings is not None
    assert settings.window == rerank.DEFAULT_WINDOW