"""Chit-chat that shares one common word with the store recalls nothing before the turn.

A household store says "every day", "in the evening", "once a week" and "works" in several facts. A message like
"What day is today?" or «Что делать вечером?» shares only that word with them: it is not about any of those facts,
and injecting them is noise the model then has to ignore. A question that says what it is about (a name with the word,
a word only one fact holds, an identifier) still recalls its fact. The facts are this repository's own, not a copy of
anyone's store."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider

# key -> (fact text, declared entities); "day", "evening", "week" and "works" are each in three facts or more, and so
# are Mark and Lena (declared) and Oleg (two facts, not declared)
FACTS = {
"en": {
 "mark_job": ("Mark works night shifts at the city hospital", ["Mark"]),
 "lena_home": ("Lena works from home on Mondays and Fridays", ["Lena"]),
 "printer": ("The old printer only works with the blue cable", []),
 "lena_dog": ("Lena walks the dog every evening after dinner", ["Lena"]),
 "piano": ("The kids practise piano in the evening", []),
 "neighbour": ("The neighbour plays loud music in the evening", []),
 "rubbish": ("The rubbish truck comes twice a week", []),
 "mark_call": ("Mark calls his parents once a week", ["Mark"]),
 "cleaner": ("The cleaner comes once a week on Thursday", []),
 "smoothie": ("Lena drinks a green smoothie every day", ["Lena"]),
 "orchids": ("The orchids on the balcony need water every day", []),
 "birthday": ("Mark's birthday is the day before Christmas", ["Mark"]),
 "allergy": ("Lena is allergic to cats", ["Lena"]),
 "car": ("The family car is a grey Toyota Corolla", []),
 "landlord": ("The landlord Somchai lives on the ground floor", ["Somchai"]),
 "curry": ("Mark's favourite food is green curry", ["Mark"]),
 "nas_backup": ("The home server nas-02 copies the photos every night", []),
 "nas_place": ("nas-02 sits in the hallway cupboard", []),
 "oleg_spanish": ("Oleg is learning Spanish at the community centre", []),
 "oleg_door": ("Oleg fixed the balcony door", []),
},
"ru": {
 "mark_job": ("Марк работает в ночные смены в городской больнице", ["Марк"]),
 "lena_home": ("Лена работает из дома по понедельникам и пятницам", ["Лена"]),
 "printer": ("Старый принтер работает только с синим кабелем", []),
 "lena_dog": ("Лена вечером гуляет с собакой после ужина", ["Лена"]),
 "piano": ("Дети вечером занимаются на пианино", []),
 "neighbour": ("Сосед вечером включает громкую музыку", []),
 "rubbish": ("Мусор вывозят два раза в неделю", []),
 "mark_call": ("Марк звонит родителям раз в неделю", ["Марк"]),
 "cleaner": ("Уборщица приходит раз в неделю по четвергам", []),
 "smoothie": ("Лена каждый день пьёт зелёный смузи", ["Лена"]),
 "orchids": ("Орхидеи на балконе поливают каждый день", []),
 "birthday": ("День рождения Марка — за день до Рождества", ["Марк"]),
 "allergy": ("У Лены аллергия на кошек", ["Лена"]),
 "car": ("Семейная машина — серая Toyota Corolla", []),
 "landlord": ("Хозяин квартиры Сомчай живёт на первом этаже", ["Сомчай"]),
 "curry": ("Любимое блюдо Марка — зелёное карри", ["Марк"]),
 "nas_backup": ("Домашний сервер nas-02 каждую ночь копирует фотографии", []),
 "nas_place": ("nas-02 стоит в шкафу в прихожей", []),
 "oleg_spanish": ("Олег учит испанский в районном центре", []),
 "oleg_door": ("Олег починил дверь на балкон", []),
},
}

CHIT_CHAT = {
 "en": ["What day is today?", "Any plans for this evening?", "Will it rain this week?", "How does a blockchain work?"],
 "ru": ["Какой сегодня день?", "Что делать вечером?", "Какая погода на этой неделе?",
        "Объясни, как работает хеш-таблица."],
}

# (question, facts that must be recalled)
QUESTIONS = {
 "en": [("Where does Mark work?", {"mark_job"}), ("What does Lena drink every day?", {"smoothie"}),
        ("When does the rubbish truck come?", {"rubbish"}), ("landlord?", {"landlord"}), ("Toyota", {"car"}),
        ("what about Lena?", {"lena_home", "lena_dog", "smoothie", "allergy"}),
        ("is nas-02 online?", {"nas_backup", "nas_place"}),
        # a name inside the sentence, no entity declared; a declared entity in lower case (its facts come by it)
        ("What is Oleg studying?", {"oleg_spanish"}), ("what does mark like to eat?", {"curry"})],
 "ru": [("Где работает Марк?", {"mark_job"}), ("Что Лена пьёт каждый день?", {"smoothie"}),
        ("Когда вывозят мусор?", {"rubbish"}), ("хозяин квартиры?", {"landlord"}), ("Toyota", {"car"}),
        ("что там с Леной?", {"lena_home", "lena_dog", "smoothie", "allergy"}),
        ("nas-02 сейчас включён?", {"nas_backup", "nas_place"}),
        ("Что изучает Олег?", {"oleg_spanish"}), ("что марк ест на обед?", {"curry"})],
}


@pytest.fixture(params=list(FACTS))
def household(request, tmp_path):
    lang = request.param
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / f"{lang}.db"), "hrr_dim": 1024})
    provider.initialize(session_id=lang)
    back = {}
    for key, (text, entities) in FACTS[lang].items():
        result = json.loads(provider.handle_tool_call("fact_store", {"action": "add", "content": text, "entities": entities}))
        back[result["fact_id"]] = key
    yield lang, provider, back
    provider.shutdown()


def _recalled(provider, back, query, turn):
    provider.on_session_switch(f"t{turn}", reset=True)
    block = provider.prefetch(query)
    return {back[int(line.split()[1][1:])] for line in block.splitlines()[1:]} if block else set()


def test_chit_chat_sharing_one_common_word_recalls_nothing(household):
    lang, provider, back = household
    noisy = {query: recalled for i, query in enumerate(CHIT_CHAT[lang])
             if (recalled := _recalled(provider, back, query, i))}
    assert not noisy, f"{lang}: chit-chat recalled facts: {noisy}"


def test_a_fact_sharing_only_the_common_word_of_a_question_is_left_out(household):
    # "Where does Mark work?" is about Mark's job: the printer that "works with the blue cable" shares only "work"
    lang, provider, back = household
    recalled = _recalled(provider, back, QUESTIONS[lang][0][0], 0)
    assert recalled & {"lena_home", "printer"} == set(), f"{lang}: {sorted(recalled)}"


def test_questions_that_say_what_they_are_about_still_recall_their_fact(household):
    lang, provider, back = household
    missed = {query: sorted(recalled) for i, (query, wanted) in enumerate(QUESTIONS[lang])
              if not wanted <= (recalled := _recalled(provider, back, query, i))}
    assert not missed, f"{lang}: questions lost their fact: {missed}"


def _small_store(tmp_path, facts):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "small.db"), "hrr_dim": 1024})
    provider.initialize(session_id="small")
    back = {json.loads(provider.handle_tool_call("fact_store", {"action": "add", "content": text}))["fact_id"]: i
            for i, text in enumerate(facts)}
    return provider, back


# (question, facts added without entities, index of the fact that answers it); its one shared word is in both facts,
# or only looks so: "car" begins "carpet", which holds no word "car"
OTHER_FACTS_SHARE_THE_WORD = [
    ("car location?", ["The car is in the garage", "The carpet is blue"], 0),
    ("What is José studying?", ["José takes pottery classes", "José lives in Madrid"], 0),
    ("What is Peter's occupation?", ["Peter repairs watches", "Peter lives in York"], 0),
    ("Wo arbeitet Oleg?", ["Oleg ist Arzt im Krankenhaus", "Oleg wohnt in Berlin"], 0),  # German: every noun capitalized
]
# Hebrew, Hindi, Arabic have no capitals to tell a name by: there a name recalls when the store knows it (declared
# entity), tests/test_non_latin_gate.py. Until the change that file describes, any word counted as a possible name,
# and one shared common word recalled 80-87 % of Hindi and Arabic questions about facts the store does not hold.


@pytest.mark.parametrize("query,facts,answer", OTHER_FACTS_SHARE_THE_WORD)
def test_a_name_in_any_form_or_script_and_a_word_no_other_fact_holds_still_recall(tmp_path, query, facts, answer):
    provider, back = _small_store(tmp_path, facts)
    try:
        assert answer in _recalled(provider, back, query, 0)
    finally:
        provider.shutdown()


def test_german_chit_chat_on_a_common_verb_recalls_nothing(tmp_path):
    # German capitalizes every noun, so a capital proves no name there; a verb in lower case is still a common word
    provider, back = _small_store(tmp_path, ["Der Drucker funktioniert nur mit dem blauen Kabel",
                                             "Die Waschmaschine funktioniert nur mit kaltem Wasser"])
    try:
        assert _recalled(provider, back, "Wie funktioniert eigentlich eine Blockchain?", 0) == set()
    finally:
        provider.shutdown()
