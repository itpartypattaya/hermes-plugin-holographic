"""A Russian name in another case is the entity the store knows (review 2026-11-17, H4 and H5).

H4. A fact that names a known person in an oblique case was not linked to them unless the agent passed `entities`:
«Подарок для Мии» did not come up in probe / related / reason for «Мия». Entity linking (mentioned_entity_ids)
compared the stems of words, and the stems of short names fall under _MIN_STEM: «Мия» and «Мии» share only «ми»,
which plain text drops. Recall already read such a word as a form of the name (text.name_lemmas); linking did not.
Of 237 forms of common names 60 names found none of their 77 forms: Мия, Анна, Оля, Юля, Ира, Ева, Илья, Вася, Алёна
in every form, Андрей, Сергей, Артём in some.

H5. A full name in an oblique case («Анне Петровой») was taken from the text as a new name next to the known one
(«Анна Петрова»), and a deleted name came back in another case: after delete_entity «Пётр Иванов» the fact «Петру
Иванову…» created «Петру Иванову», and the next fact about Пётр was linked to it.

What must not change: a lower-case word is no name (H9 of the same review, common words linked to short names
through a shared stem: tests/test_holographic_common_words_not_names.py), and a phrase that may be a name in its first form
(«Александра Петрова», a woman, beside a known «Александр Петров») stays a name of its own. The facts are synthetic.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider


@pytest.fixture
def p(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "names.db"), "hrr_dim": 1024})
    provider.initialize(session_id="names")
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


def _names(provider):
    return {e["name"] for e in _call(provider, action="entities", limit=200)["entities"]}


# -- H4: a fact naming a known person in another case is linked to them --------------------------------------------

FORMS = {
    "Мия": ("Мии", "Мию", "Мией"),
    "Анна": ("Анны", "Анне", "Анну", "Анной"),
    "Оля": ("Оли", "Оле", "Олю", "Олей"),
    "Юля": ("Юли", "Юле", "Юлю", "Юлей"),
    "Ира": ("Иры", "Ире", "Иру", "Ирой"),
    "Ева": ("Евы", "Еве", "Еву", "Евой"),
    "Илья": ("Ильи", "Илье", "Илью", "Ильёй"),
    "Вася": ("Васи", "Васе", "Васю", "Васей"),
    "Алёна": ("Алёны", "Алёне", "Алёну", "Алёной"),
    "Андрей": ("Андрея", "Андрею", "Андреем", "Андрее"),
    "Сергей": ("Сергея", "Сергею", "Сергеем", "Сергее"),
    "Артём": ("Артёма", "Артёму", "Артёмом", "Артёме"),
}


@pytest.mark.parametrize("name", FORMS)
def test_a_case_form_of_a_declared_name_links_the_fact_to_it(p, name):
    _add(p, f"{name} любит плавать в бассейне.", name)  # how the store comes to know the name: declared
    facts = {form: _add(p, f"Вчера {form} подарили книгу про динозавров ({i}).") for i, form in enumerate(FORMS[name])}
    probed = _probed(p, name)
    assert {form for form, fact_id in facts.items() if fact_id not in probed} == set(), name


def test_the_review_case_comes_up_in_probe_related_and_reason(p):
    _add(p, "Мия ходит в первый класс.", "Мия")
    gift = _add(p, "Подарок для Мии спрятан в шкафу.")
    first = _add(p, "Мии подарили велосипед.")  # the name opens the sentence
    assert {gift, first} <= _probed(p, "Мия")
    related = _call(p, action="related", entity="Мия")
    assert {gift, first} <= {f["fact_id"] for f in related.get("results", related.get("facts", []))}, related
    reason = _call(p, action="reason", entities=["Мия"])
    assert {gift, first} <= {f["fact_id"] for f in reason.get("results", reason.get("facts", []))}, reason
    assert "Мии" not in _names(p)


@pytest.mark.parametrize("name, text", [
    ("Лев", "В зоопарке видели льва и двух жирафов."),  # «льва» is a case form of «лев», the animal
    ("Мая", "Отпуск запланирован в мае на две недели."),  # «мае»: «в мае» is no «Мая»
    ("Ева", "Вчера мы читали про еву и змея."),  # nonsense on purpose: a lower-case word is never read as a name
])
def test_a_lower_case_word_is_no_case_form_of_a_name(p, name, text):
    _add(p, f"{name} любит плавать в бассейне.", name)
    other = _add(p, text)
    assert other not in _probed(p, name)


# -- H5: a full name in another case is the known entity, and a deleted name stays deleted ------------------------

@pytest.mark.parametrize("known, text", [
    ("Анна Петрова", "Анне Петровой исполнилось тридцать лет."),
    ("Анна Петрова", "Подарок для Анны Петровой лежит в машине."),
    ("Анна Петрова", "С Анной Петровой договорились о встрече."),
    ("Пётр Иванов", "Петру Иванову позвонили из банка."),
])
def test_a_full_name_in_another_case_is_the_known_entity(p, known, text):
    first = _add(p, f"{known} работает в школе учителем.")
    assert known in _names(p)
    other = _add(p, text)
    assert _names(p) == {known}
    assert {first, other} <= _probed(p, known)


def test_a_deleted_name_does_not_come_back_in_another_case(p):
    _add(p, "Пётр Иванов работает в банке.")
    assert "error" not in _call(p, action="delete_entity", entity="Пётр Иванов")
    _add(p, "Петру Иванову позвонили из банка.")
    later = _add(p, "Петру Иванову перевели зарплату.")
    assert _names(p) == set()
    assert p._store.read("SELECT entity_id FROM fact_entities WHERE fact_id = ?", (later,)) == []


def test_a_name_that_may_be_a_first_form_stays_its_own(p):
    _add(p, "Александр Петров работает в школе учителем.")
    _add(p, "Александра Петрова работает в больнице.")  # a woman, not «(у) Александра Петрова»: a name of its own
    assert _names(p) == {"Александр Петров", "Александра Петрова"}
