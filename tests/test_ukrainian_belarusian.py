"""Ukrainian and Belarusian (unreleased): Cyrillic text is told as Russian, Ukrainian or Belarusian by its letters and
function words, and Ukrainian and Belarusian words get their own stems and function words instead of Russian ones.
Russian text must read exactly as before. Measured on FLORES-200 by tools/african_probe.py."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore
from hermes_plugin_holographic.text import analyze, is_greeting, name_lemmas, query_terms

DIM = 512


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _found(store, query):
    return [f["content"] for f in FactRetriever(store, hrr_dim=DIM).search(query, min_trust=0)]


def _recalled(store, query):
    r = FactRetriever(store, hrr_dim=DIM)
    return [f["content"] for f in r.relevant(query, r.search(query, min_trust=0))]


# -- telling the three apart -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, lang", [
    ("Донька щосереди ходить до музичної школи, і їй там подобається", "uk"),
    ("Що це таке та де воно лежить?", "uk"),  # function words alone, no Ukrainian letter
    ("Навукоўцы абвясцілі ў панядзелак аб адкрыцці", "be"),
    ("Што гэта такое, дзе яно ляжыць?", "be"),
    ("ДОНЬКА ЩОСЕРЕДИ ХОДИТЬ ДО МУЗИЧНОЇ ШКОЛИ, І ЇЙ ТАМ ПОДОБАЄТЬСЯ", "uk"),  # in capitals: no word is a name
    ("Мой бацька сам пабудаваў свой дом і сад", "be"),  # «мой», «свой» are Belarusian words too
])
def test_a_ukrainian_or_belarusian_sentence_is_told_as_such(text, lang):
    assert analyze(text).cyrillic == (lang,)


@pytest.mark.parametrize("text", [
    "Мы поедем в «Київ» в субботу",  # a quoted Ukrainian name
    "Ночные смены у Оксаны в Києві",
    "Ночные поезда в Магілёў отменили",  # a Belarusian name
    "Он сказал «дякую» и ушёл",  # one Ukrainian word in a Russian sentence
    "Яна любит чай",  # Яна is a Russian name, яна Belarusian «she»
    "Согласно труду Бартоломе де лас Касаса испанцы уничтожили много индейцев",  # «де»: Ukrainian «where»
])
def test_a_russian_sentence_with_a_ukrainian_or_belarusian_word_stays_russian(text):
    analysis = analyze(text)
    assert analysis.cyrillic == ("ru",)
    assert all(w.root_lang in ("ru", "") for w in analysis.words)


@pytest.mark.parametrize("text, word, stem", [
    ("Київ посетил турист", "посетил", "посет"),  # a name that begins the sentence
    ("Он написал «дякую ще»", "дякую", "дяк"),  # a short Ukrainian quote
    ("Настроілі почту", "почту", "почт"),  # і typed for и on a Ukrainian layout
])
def test_one_ukrainian_word_leaves_a_russian_message_russian(text, word, stem):
    analysis = analyze(text)
    assert analysis.cyrillic == ("ru",)
    w = next(w for w in analysis.words if w.text == word)
    assert w.keys == {word, stem} and not w.stop


@pytest.mark.parametrize("facts, query, found", [
    (["Закрывали дверь"], "Под’езд закрыт", ["Закрывали дверь"]),  # ’ for ъ: no Ukrainian apostrophe
    (["Агентство КОТРА прислало отчёт"], "КОТРА", ["Агентство КОТРА прислало отчёт"]),  # Ukrainian «котра», which
    # a Russian query keeps
    (["Ночные смены в Києві"], "ночь", []),  # a Ukrainian letter in a name: no Russian roots, as before
])
def test_a_russian_search_finds_what_it_found_before(store, facts, query, found):
    for fact in facts:
        store.add_fact(fact)
    assert _found(store, query) == found


def test_a_russian_one_word_question_recalls_as_before(store):
    store.add_fact("Дикий як пасётся в горах")  # a yak; Ukrainian and Belarusian «як» is «how»
    assert _recalled(store, "Як") == ["Дикий як пасётся в горах"]


def test_a_ukrainian_function_word_is_no_stop_word_for_a_russian_alias(store):
    store.add_fact("Иван Петров aka Ласка починил крышу")  # «ласка»: Ukrainian «будь ласка»
    assert store.read("SELECT source FROM entity_aliases WHERE alias = 'Ласка'")[0][0] == "aka"


# -- function words --------------------------------------------------------------------------------------------------

def test_ukrainian_and_belarusian_function_words_are_dropped_and_content_words_kept():
    uk = {w.text: w.stop for w in analyze("Що це таке та де воно лежить?").words}
    assert uk["що"] and uk["це"] and uk["де"] and uk["воно"] and not uk["лежить"]
    be = {w.text: w.stop for w in analyze("Што гэта такое, дзе яно ляжыць?").words}
    assert be["што"] and be["гэта"] and be["дзе"] and be["яно"] and not be["ляжыць"]


@pytest.mark.parametrize("fact, query", [
    # read as Russian, «де», «це», «все», «було» were search words, and the question recalled the fact
    ("Все це було ще влітку, коли ми жили там", "А де це все було?"),
    ("Гэта ўсё было яшчэ ўлетку, калі мы жылі там", "А дзе гэта ўсё было?"),
])
def test_function_words_alone_recall_nothing(store, fact, query):
    store.add_fact(fact)
    assert _recalled(store, query) == []


# -- stems -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("facts, query", [
    (["Тато щоранку читає газету, бо це йому подобається",
      "Тато щоранку гуляє з собакою, бо це йому подобається"], "Що він читав?"),
    (["Оксана залишила ключі у шухляді", "Оксана купила хліб"], "Де ключ?"),
    (["Аляксей працуе ў бальніцы", "Аляксей гуляе з сабакам"], "Дзе ён працаваў?"),
    (["Бабуля жыве ў горадзе, і яна там шчаслівая", "Бабуля любіць гарбату, і яна там шчаслівая"], "Які горад?"),
])
def test_a_question_finds_the_fact_in_another_form_of_its_word(store, facts, query):
    for fact in facts:
        store.add_fact(fact)
    assert _found(store, query)[:1] == facts[:1]


@pytest.mark.parametrize("word, form", [
    # pairs the Russian stemmer (and its loose beginnings) kept apart; «працює» / «працювала», «лікарня» / «лікарні»
    # met through a shared beginning already
    ("школою", "школі"), ("синів", "сину"), ("люблю", "любити"), ("ранок", "ранку"), ("читає", "читав"),
    ("лікарем", "лікарі"), ("вчителеві", "вчитель"),
])
def test_ukrainian_forms_of_one_word_meet(word, form):
    fact = analyze(f"Оксана каже, що її {word}")  # «що», «її»: Ukrainian
    term = next(t for t in query_terms(f"що її {form}") if t.word == form)
    assert term.match(fact) == 1.0


@pytest.mark.parametrize("word, form", [
    ("працуе", "працаваў"), ("чытае", "чытаць"), ("школай", "школе"), ("білетаў", "білет"),
])
def test_belarusian_forms_of_one_word_meet(word, form):
    fact = analyze(f"Аляксей і {word} ў")
    term = next(t for t in query_terms(f"што {form} ў") if t.word == form)
    assert term.match(fact) == 1.0


def test_a_consonant_the_dative_changes_meets_at_root_weight():
    # «доньці» is «донька» in the dative: an unsure key, weighed as a word of the same root
    fact = analyze("Подарунок доньці від її бабусі")
    term = next(t for t in query_terms("що з її донькою") if t.word == "донькою")
    assert 0.0 < term.match(fact) < 1.0


@pytest.mark.parametrize("fact, query", [
    ("Подарунок доньці від її бабусі", "що з її донькою"),  # к / ц of the dative
    ("Наш кіт спить на дивані, і йому там добре", "Ви бачили її кота?"),  # і / о
    ("Бабуля жыве ў хаце, і ёй там утульна", "Ці вялікая ў яе хата?"),  # Belarusian т / ц
    ("Я купіла кнізе вокладку, і яна мне падабаецца", "Ці ёсць у яе кніга?"),  # Belarusian г / з
])
def test_a_fact_with_the_other_spelling_of_a_word_is_found(store, fact, query):
    # the fact has only the changed consonant or vowel: the index must be searched for it, not only the fact scored
    store.add_fact(fact)
    store.add_fact("Оксана гуляє з собакою в парку, і їй там добре")
    assert fact in _found(store, query)


def test_ukrainian_names_in_cases_with_letters_russian_lacks_share_a_first_form():
    assert "оксана" in name_lemmas("оксані")
    assert "марія" in name_lemmas("марії")
    assert "андрій" in name_lemmas("андрієм")


@pytest.mark.parametrize("message", ["Привіт!", "Дякую, гарного дня!", "Добрий день! Як справи?", "Дзякуй",
                                     "Прывітанне, як справы?"])
def test_a_ukrainian_or_belarusian_greeting_asks_nothing(message):
    assert is_greeting(message)


def test_a_greeting_with_a_question_still_asks():
    # a guard, not a bug: the new courtesy pieces must not swallow the question after them
    assert not is_greeting("Привіт, де ключі?")
    assert not is_greeting("Дзякуй, а дзе ключы?")


def test_belarusian_о_and_а_are_one_word_only_at_root_weight(store):
    # akanye: «горад» / «гарады» are one word, «кот» (cat) / «кат» (executioner) two; written alike they met in full
    store.add_fact("Гэта наш кат, ён жыве ў старым замку")
    assert _recalled(store, "Дзе наш кот? Ён згубіўся ўчора") == []
    fact = analyze("Гэта вялікі горад, і ён вельмі стары")
    term = next(t for t in query_terms("Якія гарады ён бачыў?") if t.word == "гарады")
    assert 0.0 < term.match(fact) < 1.0


def test_a_belarusian_word_whose_о_is_written_а_is_still_searched(store):
    # the FTS index holds «горад» as written: the question's «гарады» must reach it
    store.add_fact("Гэта вялікі горад, і ён вельмі стары")
    assert _found(store, "Якія гарады бачыла бабуля ўчора?") == ["Гэта вялікі горад, і ён вельмі стары"]


def test_a_ukrainian_name_gets_every_first_form_its_ending_allows():
    assert "любов" in name_lemmas("любові")  # «-і» of «Любов», not only «-ові» of a «Люб…»
