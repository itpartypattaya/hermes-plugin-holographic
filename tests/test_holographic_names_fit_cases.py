"""A fact offered on meaning alone mentions a short Russian name in another case than the message (2026-10-30).

The holographic-memory bridge, 2026-10-30, search by meaning on (text-embedding-3-small): «Где Ева будет учиться в
Чиангмае?» left out «…семья хочет устроить Еву в школу…» (entities ["Ева"]), although it was the store's closest fact
(cosine 0.70, z 3.97 over the bar of 3.67). names_fit asks that a fact offered on meaning alone hold every name the
message names, in any word form; «Ева» and «Еву» share only the stem «ев», which plain Russian text drops as a key
(_MIN_STEM), so the fact was read as being about someone else. The same for «Ира» / «Иру», «Оля» / «Олю», «Яна» / «Яну»,
«Аня» / «Ане»; «Ксюша» / «Ксюшу» share «ксюш» and always met. A name now also meets a word of the fact written with a
capital that may be a case form of the same first form (text.name_lemmas; 0.25.1 compared short stems, which
joined «Аня» and «Анна»); «Евгений», «Ольга» and the month «мае» stay apart. The facts are synthetic, modelled on the
case; none of them is anyone's."""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider, embeddings  # noqa: E402

ON = {"embeddings": "on", "embedding_url": "https://embeddings.example.net/v1", "embedding_model": "m",
      "embedding_key_env": "TEST_EMBEDDINGS_KEY"}

SCHOOL = "После переезда в Чиангмай весной 2027 года семья хочет устроить Еву в школу, организовывать занятия и изучать город."
EVA = ["У Ксении есть дочь Ева; в июне 2026 года Еве было 8 лет.",
       "В июне 2026 года Ева закончила первый класс тайской школы в Паттайе."]
FILLERS = ["Антон пьёт чёрный кофе без сахара.", "Мусор вывозят во вторник и в пятницу.",
           "Стоматолог принимает по средам после обеда.", "Кот Барсик ест только сухой корм.",
           "Машина стоит на парковке у дома.", "Байк заправлять на станции у рынка.",
           "Рынок работает по выходным до двух часов.", "Сосед сверху громко слушает музыку.",
           "Пароль от вайфая записан на холодильнике.", "Орхидеи поливать раз в неделю.",
           "Йога по понедельникам в семь вечера.", "Ксюша работает из дома по пятницам.",
           "Маме звонить по воскресеньям.", "В сезон дождей взять с собой зонт и дождевик.",
           "Аренда квартиры оплачивается первого числа.", "Интернет у провайдера оплачен до декабря.",
           "Кондиционер чистить раз в три месяца.", "Банк закрывается в шесть вечера.",
           "Визу продлевать в иммиграционном офисе.", "Страховка на байк действует до марта.",
           "Холодильник разморозить перед отъездом.", "Лампочку в коридоре заменить на тёплую.",
           "Встреча IT-тусовки по пятницам в баре.", "Сервер бэкапится каждую ночь.",
           "Дима ходит в бассейн по вторникам и четвергам.", "Лена любит читать детективы перед сном.",
           "Марк слушает джаз по вечерам.", "Пляж Джомтьен ближе всего к дому.",
           "Фрукты покупать на рынке у моря.", "Курс бата смотреть утром в обменнике.",
           "Велосипед стоит в подъезде."]


class _NoNetwork:
    def open(self, *a, **k):
        raise AssertionError("no request in these tests: meaning is given")


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "_OPENER", _NoNetwork())
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64, **ON})
    p.initialize(session_id="s1")
    yield p
    p.shutdown()


def _add(p, text, entities=()) -> int:
    args = {"action": "add", "content": text, **({"entities": list(entities)} if entities else {})}
    result = json.loads(p.handle_tool_call("fact_store", args))
    assert "fact_id" in result, result
    return result["fact_id"]


def _fact(p, text, entities=()) -> dict:
    return p._retriever._rows([_add(p, text, entities)], None, 0.0)[0]


# -- the case ---------------------------------------------------------------------------------------------------------

def test_the_closest_fact_about_eva_is_recalled(provider, monkeypatch):
    """Before a turn, the store's closest fact in meaning, far over the bar, is recalled although it names Eva in
    another case («Еву») than the message («Ева»). No fact declares her as an entity, so neither the words (no word
    of the message is in the fact; the store knows no name to read «Еву» as «Ева») nor the facts of an entity bring
    it: only the meaning stream can, as in the bridge."""
    ids = {text: _add(provider, text) for text in [*EVA, SCHOOL, *FILLERS]}
    order = list(ids)
    sims = [0.70 if t == SCHOOL else 0.45 if t in EVA else 0.10 + 0.01 * (i % 12) for i, t in enumerate(order)]
    meaning = embeddings.Meaning([ids[t] for t in order], np.array(sims), 0.3,
                                 [embeddings.text_hash(t) for t in order])
    assert meaning.standout(ids[SCHOOL], embeddings.RECALL_MARGIN, None)  # over the bar: only names_fit can drop it
    monkeypatch.setattr(provider, "_meaning_of", lambda q, before_turn: meaning)
    block = provider.prefetch("Где Ева будет учиться?")
    school = [line for line in block.splitlines() if f"#{ids[SCHOOL]} " in line]
    assert school and school[0].endswith("(by meaning)"), block


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "not-declared"])
def test_the_school_fact_fits_a_question_naming_eva(provider, declared):
    fact = _fact(provider, SCHOOL, ["Ева"] if declared else ())
    assert provider._retriever.names_fit("Где Ева будет учиться в Чиангмае?", fact)


# -- other short names, other cases -------------------------------------------------------------------------------------

@pytest.mark.parametrize("query, text, name", [
    ("Что подарить Еве на день рождения?", "Ева мечтает о велосипеде.", "Ева"),
    ("Когда у Иры отпуск?", "Иру отпускают в отпуск в августе.", "Ира"),
    ("Во сколько забирать Олю из садика?", "Оля в садике до пяти вечера.", "Оля"),
    ("Что нужно взять Яне на балет?", "У Яны балет по вторникам, нужна форма.", "Яна"),
    ("Когда Аню вести к врачу?", "Ане назначили приём у врача на четверг.", "Аня"),
    ("Какой чай любит Аня?", "Ксюша заваривает Ане зелёный чай.", "Аня"),
])
@pytest.mark.parametrize("declared", [True, False], ids=["declared", "not-declared"])
def test_a_case_form_of_a_short_name_fits(provider, query, text, name, declared):
    fact = _fact(provider, text, [name] if declared else ())
    assert provider._retriever.names_fit(query, fact)


# -- a name whose stem changes between its forms ------------------------------------------------------------------------
# «Лев» / «Льва», «Павел» / «Павла» drop a vowel; Snowball cuts «Никите» to another stem than «Никита». None of them met
# on 0.25.1, by any key.

@pytest.mark.parametrize("query, text", [
    ("Что подарить Льву на день рождения?", "Лев собирает модели самолётов."),
    ("Где сейчас Лев?", "Льва записали в секцию плавания."),
    ("Во сколько забрать Льва?", "Со Львом договорились на шесть вечера."),
    ("Когда приедет Павел?", "Павла ждут в пятницу вечером."),
    ("Что передать Павлу?", "Павел просил купить батарейки."),
    ("Где Никита?", "Никите купили новый рюкзак."),
    ("Что купили Никите?", "Никиту записали к стоматологу."),
    ("Что подарить Любови?", "Любовь любит пионы."),
    ("Где сейчас Любовь?", "Любви купили билеты в театр."),
    # an ending read two ways (these held on 0.25.1 too): «Илью» is not «-ью» of «Ил…», «Зою» not «-ою» of «З…»
    ("Где сейчас Илья?", "Илью забрали из школы пораньше."),
    ("Что подарить Зое?", "Зою пригласили на день рождения."),
    # «-ей» after ж, ш, ч, щ, ц is a name in «-а» («Яшей» of «Яша»), not only one in «-я» (Codex review)
    ("Где сейчас Яша?", "С Яшей договорились на субботу."),
    ("Когда забирать Яшу?", "С Яшей договорились на субботу."),
    ("Где сейчас Яша?", "Яшею все гордятся."),
    ("Где сейчас Гоша?", "С Гошей договорились на субботу."),
    ("Где сейчас Даша?", "С Дашей договорились на субботу."),
])
def test_a_name_whose_stem_changes_fits(provider, query, text):
    assert provider._retriever.names_fit(query, _fact(provider, text))


# -- different names stay apart -------------------------------------------------------------------------------------------
# «Ян» and «Яна» share «Яна», «Яну» and «Яне» (each is a form of both), and those still fit either; a form only one of
# them has («Ян», «Яном» / «Яны», «Яной») tells them apart. «Аня» and «Анна» met on 0.25.1 (a short stem of the name
# read in «Анна»), not before.

@pytest.mark.parametrize("query, text", [
    ("Где Ева будет учиться?", "Евгений учится в университете в Бангкоке."),
    ("Где Ева будет учиться?", "Евгению предложили место в школе."),
    ("Во сколько забирать Олю из садика?", "Ольгу забирают из садика в пять."),
    ("Когда у Иры отпуск?", "Ирина берёт отпуск в августе."),
    ("Когда у Мая день рождения?", "День рождения в мае, подарок купить заранее."),  # the name, the month
    ("Когда Ане к врачу?", "Антону назначили приём у врача на четверг."),
    ("Где сейчас Ян?", "У Яны балет по вторникам."),
    ("Где сейчас Ян?", "С Яной договорились на субботу."),
    ("Когда у Яны балет?", "Ян ходит в бассейн по средам."),
    ("Когда у Яны балет?", "С Яном договорились на субботу."),
    ("Что подарить Ане?", "Анна работает в банке."),
    ("Когда Аню вести к врачу?", "Анне назначили приём на четверг."),
    ("Где Анна?", "Аня ушла гулять."),
    ("Где сейчас Лев?", "Лена ушла гулять."),
    ("Где сейчас Олег?", "Ольгу забрали из школы."),
    # a vowel drops only in names that drop it: «Марка» is of «Марк», not «Марек» (Codex review on GitHub)
    ("Где сейчас Марек?", "Марка записали в секцию плавания."),
    ("Где сейчас Марек?", "С Марком договорились на субботу."),
    ("Где сейчас Петер?", "Петра записали в секцию плавания."),
])
def test_another_name_does_not_fit(provider, query, text):
    assert not provider._retriever.names_fit(query, _fact(provider, text))


def test_a_known_name_still_needs_its_entity(provider):
    """The fact names Eva in another case but is linked to no entity of hers: the known-entity part of names_fit still
    asks for the link, as before."""
    _add(provider, EVA[0], ["Ева"])
    fact = _fact(provider, "Иру и Еву отвезти в школу к восьми.", ["Ира"])
    assert not provider._retriever.names_fit("Где Ева будет учиться?", fact)
