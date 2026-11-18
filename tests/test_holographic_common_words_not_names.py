"""A lower-case common word is no mention of a declared one-word name (review 2026-11-17, H9).

The store links a fact to the known entities its words mention (mentioned_entity_ids), and recall brings the facts of
the entities a message mentions. A Russian name was matched by the stems of the words, whatever their case: «Мира»
and «мире» share the stem «мир», «Варя» and «варю» «вар», «Маша» and «машу» «маш», «Гена» and «гены» «ген». So a
fact «В мире много красивых мест» was linked to the girl Мира and came up in probe / reason for her, and a message
about the world brought her facts into the prompt.

What the user needs: a common word written in lower case links nothing to a one-word name and recalls none of its
facts; the name written with a capital, in any case form («Мире», «Маше», «Вари»), still does, and so does the name
in lower case exactly as written («что марк ест на обед?», tests/test_holographic_chit_chat_gate.py). Left as it is:
a common word that opens a sentence («Машу рукой…») is written with a capital and may be the name, and a common word
spelt as the name («до мира») is the name in lower case; nothing in the text tells them apart. The facts are
synthetic.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider


@pytest.fixture
def p(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "words.db"), "hrr_dim": 1024})
    provider.initialize(session_id="words")
    yield provider
    provider.shutdown()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _add(provider, content, *entities):
    args = {"action": "add", "content": content}
    if entities:
        args["entities"] = list(entities)
    result = _call(provider, **args)
    assert "fact_id" in result, result
    return result["fact_id"]


def _probed(provider, entity):
    result = _call(provider, action="probe", entity=entity)
    return {f["fact_id"] for f in result.get("results", result.get("facts", []))}


def _reasoned(provider, entity):
    result = _call(provider, action="reason", entities=[entity])
    return {f["fact_id"] for f in result.get("results", result.get("facts", []))}


# name -> (the fact that declares it, a fact whose common word shares the name's stem, a message with such a word)
NAMES = {
    "Мира": ("Мира любит рисовать акварелью.", "В мире много красивых мест для отпуска.",
             "какие страны в мире самые тёплые зимой?"),
    "Варя": ("Варя ходит на танцы по средам.", "По утрам я варю кофе в турке.",
             "как правильно варю гречку, сколько воды?"),
    "Маша": ("Маша учится во втором классе.", "Уходя из офиса, я машу рукой охраннику.",
             "я машу рукой, а такси не останавливается"),
    "Гена": ("Гена чинит машины в гараже.", "Цвет глаз ребёнку задают гены родителей.",
             "правда ли, что гены определяют рост?"),
}

# a household store around the names, so that recall has something to choose from
FILLERS = (
    "Антон пьёт чёрный кофе без сахара.",
    "Мусор вывозят во вторник и в пятницу.",
    "Стоматолог принимает по средам после обеда.",
    "Пароль от вайфая записан на холодильнике.",
    "Байк заправлять на станции у рынка.",
)


@pytest.mark.parametrize("name", NAMES)
def test_a_lower_case_common_word_does_not_link_a_fact_to_the_name(p, name):
    declared, common, _ = NAMES[name]
    own = _add(p, declared, name)
    other = _add(p, common)
    assert other not in _probed(p, name)
    assert other not in _reasoned(p, name)
    assert own in _probed(p, name)


@pytest.mark.parametrize("name", NAMES)
def test_a_lower_case_common_word_in_a_message_does_not_recall_the_names_facts(p, name):
    declared, _, message = NAMES[name]
    for filler in FILLERS:
        _add(p, filler)
    _add(p, declared, name)
    p.on_session_switch("turn", reset=True)
    assert declared not in p.prefetch(message)


@pytest.mark.parametrize("name, text", [
    ("Мира", "Мире подарили набор красок на день рождения."),
    ("Маша", "Вчера Маше купили новый рюкзак."),
    ("Варя", "У Вари в субботу концерт в музыкальной школе."),
    ("Гена", "Гене нужно вернуть дрель до пятницы."),
])
def test_the_name_written_with_a_capital_still_links_in_any_case(p, name, text):
    _add(p, NAMES[name][0], name)
    other = _add(p, text)
    assert other in _probed(p, name)


@pytest.mark.parametrize("name, message", [
    ("Маша", "что подарить Маше на день рождения?"),
    ("Мира", "во сколько у Миры кружок?"),
])
def test_the_name_written_with_a_capital_in_a_message_still_recalls_its_facts(p, name, message):
    for filler in FILLERS:
        _add(p, filler)
    _add(p, NAMES[name][0], name)
    p.on_session_switch("turn", reset=True)
    assert NAMES[name][0] in p.prefetch(message)
