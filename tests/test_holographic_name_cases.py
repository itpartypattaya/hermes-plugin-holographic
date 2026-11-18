"""A case form of a known name meets its other forms in the coverage gate (0.18.3).

A family store, 0.18.2: «что Яне нужно взять с собой во вторник и какой чай она любит?» brought the fact about
Tuesdays («у Яны балет…») and cut «Яна любит жасминовый чай…» at coverage 0.25. In plain Russian text «Яна», «Яны»
and «Яне» have no key in common: Snowball's stem «ян» is under _MIN_STEM and dropped (as a prefix it would find
«январь»), while «Ксюша» and «Ксюше» share «ксюш». The store knew the name all along: name_keys reads «ян» in an
entity's name. The gate now credits a term that is a form of a name the store knows to a fact holding another form
of it. Nothing else changes: not Word.keys, not the FTS index, not the duplicate and conflict checks at write time,
not entity linking, not a store that does not hold the name, not Latin names. The facts are synthetic, modelled on
the case; none of them is anyone's."""

from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic.text import query_terms

# key -> text; the words of the questions below are spread over the fillers, so that IDF is that of a household store
FACTS = {
    "ballet": "По вторникам у Яны балет; во вторник напоминать взять балетную форму.",
    "tea": "Яна любит жасминовый чай.",
    "anton_coffee": "Антон пьёт чёрный кофе без сахара.",
    "ksu_tea": "Ксюша любит зелёный чай с мятой.",
    "dima_pool": "Дима ходит в бассейн по вторникам и четвергам.",
    "rubbish": "Мусор вывозят во вторник и в пятницу.",
    "umbrella": "В сезон дождей взять с собой зонт и дождевик.",
    "dentist": "Стоматолог принимает по средам после обеда.",
    "lena_books": "Лена любит читать детективы перед сном.",
    "mark_music": "Марк слушает джаз по вечерам.",
    "cat": "Кот Барсик ест только сухой корм.",
    "car": "Машина стоит на парковке у дома.",
    "bike": "Байк заправлять на станции у рынка.",
    "school": "Школа начинается в восемь утра.",
    "market": "Рынок работает по выходным до двух часов.",
    "neighbour": "Сосед сверху громко слушает музыку.",
    "wifi": "Пароль от вайфая записан на холодильнике.",
    "plants": "Орхидеи поливать раз в неделю.",
    "yoga": "Йога по понедельникам в семь вечера.",
    "ksu_work": "Ксюша работает из дома по пятницам.",
    "mama_call": "Маме звонить по воскресеньям.",
    # words that share a two-letter stem with another word: «ель» / «ели» (ate), «уха» / «ухо», «рой» / «рою» (I dig)
    "forest": "В лесу у дома растёт старая ель.",
    "soup": "На ужин сварить уху из форели.",
    "bees": "Пчелиный рой улетел в сад.",
}

TWO_TOPICS = "что Яне нужно взять с собой во вторник и какой чай она любит?"

# How a store comes to hold «Яна»: declared with a fact (entities=[...]); guessed from a two-word name in a text;
# given an alias (the alias action creates the entity, entity_type stays 'unknown'); left by an older version as a
# one-word guessed name no fact is linked to (entity_type 'unknown', the case's store). None: the store never heard
# the name.
KNOWN = ("declared", "phrase", "alias", "legacy")


def _store(tmp_path, how: str | None):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "names.db"), "hrr_dim": 1024})
    provider.initialize(session_id="names")
    back = {}
    for key, text in FACTS.items():
        args = {"action": "add", "content": text}
        if how == "declared" and key == "tea":
            args["entities"] = ["Яна"]
        back[json.loads(provider.handle_tool_call("fact_store", args))["fact_id"]] = key
    if how == "phrase":
        result = provider.handle_tool_call("fact_store", {"action": "add", "content": "Яна Соколова записана в школу балета."})
        back[json.loads(result)["fact_id"]] = "kid_full"
    elif how == "alias":
        assert "error" not in provider.handle_tool_call("fact_store", {"action": "alias", "entity": "Яна", "content": "Яночка"})
    elif how == "legacy":
        conn = sqlite3.connect(str(tmp_path / "names.db"))  # another connection: the index sees the change
        conn.execute("INSERT INTO entities (name, name_key) VALUES ('Яна', 'яна')")
        conn.commit()
        conn.close()
    return provider, back


@pytest.fixture
def known(tmp_path):
    provider, back = _store(tmp_path, "declared")
    yield provider, back
    provider.shutdown()


@pytest.fixture
def unknown(tmp_path):
    provider, back = _store(tmp_path, None)
    yield provider, back
    provider.shutdown()


def _recalled(provider, back, query, turn=0):
    provider.on_session_switch(f"t{turn}", reset=True)
    block = provider.prefetch(query)
    return {back[int(line.split()[1][1:])] for line in block.splitlines()[1:]} if block else set()


def _rows(provider, back, *keys):
    wanted = {key: fact_id for fact_id, key in back.items() if key in keys}
    return [provider._retriever._rows([wanted[key]])[0] for key in keys]


def _gate(provider, query, rows):
    return [f["fact_id"] for f in provider._retriever.relevant(query, rows, provider._min_coverage)]


# -- the bug -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("how", KNOWN)
def test_a_question_of_two_topics_brings_both_facts_about_the_name(tmp_path, how):
    provider, back = _store(tmp_path, how)
    try:
        recalled = _recalled(provider, back, TWO_TOPICS)
    finally:
        provider.shutdown()
    assert {"ballet", "tea"} <= recalled, f"{how}: {sorted(recalled)}"


@pytest.mark.parametrize("query, wanted", [
    ("Какой чай нравится Яне?", "tea"),
    ("Что любит Яне?", "tea"),
    ("Чем угостить Яну: какой чай?", "tea"),
    ("Какой чай купить для Яны?", "tea"),
    ("Яна по вторникам занята?", "ballet"),  # the other way round: the question in the nominative, the fact «у Яны»
])
def test_a_case_form_of_the_name_meets_its_other_forms(known, query, wanted):
    provider, back = known
    recalled = _recalled(provider, back, query)
    assert wanted in recalled, f"{query}: {sorted(recalled)}"


def test_the_gate_credits_the_name_by_the_key_its_forms_share(known):
    provider, back = known
    tea, = _rows(provider, back, "tea")
    assert {"ян", "яна"} <= provider._store.known_name_forms()
    named = provider._retriever._named_terms(query_terms("Какой чай нравится Яне?"))
    assert {t.word: sorted(probe) for t, probe in named.items()} == {"яне": ["ян"]}
    assert _gate(provider, "Какой чай нравится Яне?", [tea]) == [tea["fact_id"]]


def test_an_alias_gives_its_forms_too(known):
    provider, back = known
    provider.handle_tool_call("fact_store", {"action": "alias", "entity": "Яна", "content": "Яночка"})
    assert {"ян", "яночк"} <= provider._store.known_name_forms()


# -- what stays as it was ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("query, key", [
    ("Что вы ели вчера?", "forest"),  # «ели» (ate) and «ель» (a fir) share the stem «ел»
    ("Болит ухо?", "soup"),  # «ухо» (an ear) and «уху» (fish soup) share «ух»
    ("Я рою грядку?", "bees"),  # «рою» (I dig) and «рой» (a swarm) share «ро»
])
def test_words_sharing_a_short_stem_stay_apart_when_neither_is_a_name(known, query, key):
    provider, back = known  # the store knows a name («Яна»), not these words
    row, = _rows(provider, back, key)
    assert _gate(provider, query, [row]) == []


def test_a_name_the_store_does_not_know_is_read_as_before(unknown):
    provider, back = unknown  # «Кот Барсик», guessed from its fact, is the store's only name
    tea, = _rows(provider, back, "tea")
    assert _gate(provider, "Какой чай нравится Яне?", [tea]) == []
    assert "tea" not in _recalled(provider, back, TWO_TOPICS)


def test_latin_names_match_as_written(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "latin.db"), "hrr_dim": 1024})
    provider.initialize(session_id="latin")
    try:
        add = lambda text, *names: json.loads(provider.handle_tool_call(  # noqa: E731
            "fact_store", {"action": "add", "content": text, "entities": list(names)}))["fact_id"]
        back = {add("Jean cooks dinner on Fridays.", "Jean"): "jean", add("Eve likes jasmine tea.", "Eve"): "tea",
                add("The blue jeans are in the wardrobe.", ): "jeans"}
        assert _recalled(provider, back, "What does Eve like?") == {"tea"}
        # "jeans" is no form of Jean: her fact shares that one word with a longer message and is left out, as before
        assert _recalled(provider, back, "Where did I leave my blue jeans yesterday evening?", 1) == {"jeans"}
    finally:
        provider.shutdown()


def test_the_checks_at_write_time_read_a_case_form_as_the_name(known):
    """0.18.3 left the checks at write time as they were: «Яне» named no one there, so «Яне нравится жасминовый чай.»
    met nothing. Since the review of 2026-11-17 (H4) a case form is linked to the name, and the checks see the same
    person a sentence with «Яна» names: a new claim about her tea meets the old one, whichever case names her."""
    provider, _ = known
    add = lambda text: json.loads(provider.handle_tool_call("fact_store", {"action": "add", "content": text}))  # noqa: E731
    conflicts = lambda result: [c["content"] for c in result.get("possible_conflicts", ())]  # noqa: E731
    assert conflicts(add("Яна обожает жасминовый чай.")) == ["Яна любит жасминовый чай."]
    assert "Яна любит жасминовый чай." in conflicts(add("Яне нравится жасминовый чай."))
    assert "Яна любит жасминовый чай." in conflicts(add("Яна не любит жасминовый чай."))


def test_chit_chat_naming_the_person_recalls_what_her_name_recalls(known):
    """A message naming a person brings that person's top facts without a word in common (prefetch, stream 3), and
    «Яна, …» always did. Before the review of 2026-11-17 (H4) «Передай Яне привет» recalled nothing only because
    «Яне» named no one; now it recalls what the name does."""
    provider, back = known
    named = _recalled(provider, back, "Яна, передай привет, скажи, что я задержусь, чай выпью на работе")
    assert named and _recalled(provider, back, "Передай Яне привет, скажи, что я задержусь, чай выпью на работе",
                               1) == named
