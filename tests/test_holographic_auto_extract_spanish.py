"""Spanish preferences and decisions follow the existing opt-in extraction path.

Spanish drops the subject, so the verb opens the sentence ("prefiero", "me gusta", "uso"): the patterns anchor there
and nowhere else, which keeps a quote ("Paul dice que prefiere …"), a lead-in ("Yo, prefiero …"), a condition ("Si
prefiero …") and a question out. "prefiero que …" is an instruction to the assistant, not a preference, and Spanish
takes the same rule the French patterns do. Replies ("me gusta eso", "me gusta, gracias") are taken out by the
Spanish branch of the shared reaction rule.
"""

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider


@pytest.fixture
def provider(tmp_path):
    instance = HolographicMemoryProvider(config={
        "db_path": str(tmp_path / "facts.db"), "hrr_dim": 64, "auto_extract": True,
    })
    instance.initialize(session_id="spanish-extraction")
    yield instance
    instance.shutdown()


@pytest.mark.parametrize("sentence,category", [
    ("Prefiero las respuestas cortas y directas.", "user_pref"),
    ("prefiero trabajar con Godot para los prototipos.", "user_pref"),
    ("Me gusta el café sin azúcar.", "user_pref"),
    ("Me encanta el té verde.", "user_pref"),
    ("Suelo madrugar los domingos.", "user_pref"),
    ("Uso Linux en casa.", "user_pref"),
    ("Siempre uso vim para editar código.", "user_pref"),
    ("Nunca uso Windows para programar.", "user_pref"),
    ("Por lo general trabajo de noche.", "user_pref"),
    ("Mi editor favorito es Neovim.", "user_pref"),
    ("No me gusta el cilantro.", "user_pref"),
    ("Odio las reuniones largas.", "user_pref"),
    # An adverb between the verb and a real object states what is liked: kept, as the French patterns keep it.
    ("Me gusta mucho el café de aquí.", "user_pref"),
    ("PREFIERO LAS RESPUESTAS CORTAS Y DIRECTAS.", "user_pref"),
    ("Hemos decidido usar PostgreSQL para la persistencia.", "project"),
    ("Ya hemos acordado el reparto de tareas.", "project"),
    ("Hemos elegido Godot para el prototipo.", "project"),
    ("Hemos optado por SQLite en local.", "project"),
    ("Decidimos quedarnos con Godot.", "project"),
    ("El proyecto usa Supabase para el backend.", "project"),
    ("La regla del proyecto es no tocar raw/.", "project"),
])
def test_spanish_sentence_is_stored_verbatim_with_its_category(provider, sentence, category):
    provider.on_session_end([{"role": "user", "content": sentence}])
    stored = [(f["content"], f["category"]) for f in provider._store.list_facts(limit=10)]
    assert stored == [(sentence, category)]


@pytest.mark.parametrize("sentence", [
    # Not the person's: a quote and a third-person subject.
    "Paul dice que prefiere el café sin azúcar.",
    "Ella prefiere las respuestas largas.",
    # An instruction to the assistant, not a lasting preference.
    "Prefiero que me hables de tú.",
    "Prefiero que verifiques el archivo ahora.",
    "Prefiero mucho que lo revises tú.",
    # A question is not a statement.
    "¿Prefieres el café o el té?",
    "¿Qué prefieres para cenar?",
    # A one-off intent, as "I want"/"I need" in English and «хочу» in Russian.
    "Quiero verificar la fecha de la reunión.",
    "Necesito revisar el archivo.",
    # A conditional, and a lead-in that is not looked past.
    "Si prefiero el té, lo compro.",
    "Yo, prefiero el té verde.",
    "Bueno, prefiero las respuestas cortas.",
    # "siempre" with a copula says nothing durable about the person.
    "Siempre es así.",
    "Nunca es tarde.",
    # Spanish "I decided" is a one-off: only "we decided" is a project decision.
    "Decidí preguntarte otra cosa.",
    "Elegí la segunda opción.",
    # "hemos" is the auxiliary of every "we have done …": only a decision verb after it decides anything.
    "Hemos comido pizza en la playa.",
    "Hemos visto la película dos veces.",
    "Ya hemos terminado el informe.",
])
def test_spanish_question_quote_intent_or_conditional_is_not_stored(provider, sentence):
    provider.on_session_end([{"role": "user", "content": sentence}])
    assert provider._store.list_facts(limit=10) == []


@pytest.mark.parametrize("sentence", [
    # A reply to the assistant names nothing on its own.
    "Me gusta eso.",
    "Me gusta esto, gracias.",
    "Me gusta este formato.",
    "Me gusta bastante esa idea.",
    "Me gustan esos.",
    "Me gusta, gracias.",
    "Me gusta mucho.",
    "Prefiero este.",
    "Prefiero esa.",
    "Prefiero más bien esto.",
])
def test_spanish_reply_to_the_assistant_is_not_stored(provider, sentence):
    provider.on_session_end([{"role": "user", "content": sentence}])
    assert provider._store.list_facts(limit=10) == []


def test_spanish_reply_followed_by_a_preference_keeps_the_sentence(provider):
    """'Me gusta eso, prefiero el té verde': the reply is taken out and what is left still states a preference."""
    sentence = "Me gusta eso, prefiero el té verde."
    provider.on_session_end([{"role": "user", "content": sentence}])
    assert [(f["content"], f["category"]) for f in provider._store.list_facts(limit=10)] == [(sentence, "user_pref")]


def test_spanish_extraction_stays_opt_in(tmp_path):
    instance = HolographicMemoryProvider(config={"db_path": str(tmp_path / "facts.db"), "hrr_dim": 64})
    instance.initialize(session_id="no-auto-extract")
    try:
        instance.on_session_end([{"role": "user", "content": "Prefiero las respuestas cortas y directas."}])
        assert instance._store.list_facts(limit=10) == []
    finally:
        instance.shutdown()


def test_spanish_conditional_in_the_middle_is_still_stored(provider):
    """Known limit, as in English and French: only a sentence that opens with "si" is left out."""
    sentence = "Hemos decidido salir si el tiempo lo permite."
    provider.on_session_end([{"role": "user", "content": sentence}])
    assert [(f["content"], f["category"]) for f in provider._store.list_facts(limit=10)] == [(sentence, "project")]
