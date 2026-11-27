"""Russian preferences and decisions follow the existing opt-in extraction path."""

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider


@pytest.fixture
def provider(tmp_path):
    instance = HolographicMemoryProvider(config={
        "db_path": str(tmp_path / "facts.db"), "hrr_dim": 64, "auto_extract": True,
    })
    instance.initialize(session_id="russian-extraction")
    yield instance
    instance.shutdown()


@pytest.mark.parametrize("sentence,category", [
    ("Мне нравится тёмная тема интерфейса.", "user_pref"),
    ("Мне очень нравится работать в Vim.", "user_pref"),
    ("Мне не нравится шумный интерфейс.", "user_pref"),
    ("Мне больше нравится PostgreSQL.", "user_pref"),
    ("Мне удобнее отвечать короткими сообщениями.", "user_pref"),
    ("Мне так удобнее.", "user_pref"),
    ("Мне лучше писать тесты до кода.", "user_pref"),
    ("Мне так лучше.", "user_pref"),
    ("Мне лучше всего работается утром.", "user_pref"),
    ("Мне нравится, когда тесты зелёные.", "user_pref"),
    ("Мне не нравится, когда интерфейс шумит.", "user_pref"),
    ("Давай сделаем миграцию на выходных.", "project"),
    ("Давай используем SQLite для прототипа.", "project"),
    ("Давай выберем Python для парсера.", "project"),
    ("ДАВАЙ СДЕЛАЕМ МИГРАЦИЮ НА ВЫХОДНЫХ.", "project"),
])
def test_russian_sentence_is_stored_verbatim_with_its_category(provider, sentence, category):
    provider.on_session_end([{"role": "user", "content": sentence}])
    facts = provider._store.list_facts(limit=10)
    assert [(fact["content"], fact["category"]) for fact in facts] == [(sentence, category)]
    assert facts[0]["trust_score"] == pytest.approx(0.4)


@pytest.mark.parametrize("sentence", [
    # A question is not a preference, whatever its subject.
    "Тебе нравится тёмная тема?",
    "А мне нравится этот формат?",
    "Давай сделаем миграцию на выходных?",
    # Third person: somebody else's like, not the person's own.
    "Ему нравится тёмная тема.",
    "Ей нравится пить кофе по утрам.",
    # «мне нравится» with nothing named after it: a comma, or the end of the clause.
    "Мне нравится, спасибо.",
    "Мне нравится.",
    # A reply to the assistant states nothing on its own.
    "Мне нравится это.",
    "Мне нравится этот формат.",
    "Мне нравится то, что ты предложил.",
    "Мне не нравится эта идея.",
    # «мне лучше» as a reply chooses among the assistant's options; a bare «Мне лучше.» is not a reply and stays
    # a preference. Impersonal advice and third person have no «мне».
    "Мне лучше этот вариант.",
    "Мне лучше то, что ты предложил.",
    "Лучше писать тесты до кода.",
    "Ему лучше отдохнуть.",
    "Мне лучше сделать так?",
    # An order to the assistant, or the speaker acting alone, is not the project's decision.
    "Давай сделаешь это сегодня.",
    "Давай я сделаю миграцию сам.",
    # The patterns match after any whitespace, as the existing Russian lines do: «Кстати, мне нравится тёмная
    # тема.» IS extracted, unlike the ^-anchored French ones.
])
def test_russian_question_reaction_or_other_subject_sentence_is_not_stored(provider, sentence):
    provider.on_session_end([{"role": "user", "content": sentence}])
    assert provider._store.list_facts(limit=10) == []


def test_russian_extraction_stays_opt_in(tmp_path):
    instance = HolographicMemoryProvider(config={"db_path": str(tmp_path / "facts.db"), "hrr_dim": 64})
    instance.initialize(session_id="no-auto-extract")
    try:
        instance.on_session_end([{"role": "user", "content": "Мне нравится тёмная тема интерфейса."}])
        assert instance._store.list_facts(limit=10) == []
    finally:
        instance.shutdown()


def test_russian_reply_followed_by_a_preference_keeps_the_sentence(provider):
    """'Мне нравится это, мне удобнее тёмная тема': the reply is taken out and what is left still states a
    preference, as "I like that, I always use tabs" is in English."""
    sentence = "Мне нравится это, мне удобнее тёмная тема."
    provider.on_session_end([{"role": "user", "content": sentence}])
    assert [(f["content"], f["category"]) for f in provider._store.list_facts(limit=10)] == [(sentence, "user_pref")]
