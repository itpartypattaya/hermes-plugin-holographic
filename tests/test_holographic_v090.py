"""0.9.0: words of the same root (Russian and English, in text whose language is determined) and a shorter shared
beginning for six-letter Russian stems, matched at a lower weight than the word itself and never recalling a fact on
their own; courtesy messages; Russian question words left out of queries."""

from __future__ import annotations


import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider, text
from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore

DIM = 512


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _r(store):
    return FactRetriever(store, hrr_dim=DIM)


def _ids(store, query, limit=5):
    return [f["fact_id"] for f in _r(store).search(query, min_trust=0, limit=limit)]


def _recalled(store, query):
    candidates = _r(store).search(query, min_trust=0, limit=10)
    return [f["fact_id"] for f in _r(store).relevant(query, candidates, 0.3)]


@pytest.fixture
def no_roots(monkeypatch):
    """0.8.0 matching: no roots, no short beginning."""
    def off():
        monkeypatch.setattr(text, "_ROOTS", False)
        monkeypatch.setattr(text, "_SHORT_LOOSE", False)
        text.analyze.cache_clear()
    yield off
    text.analyze.cache_clear()


# -- the rules ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("word, related", [
    ("ночные", "ноч"), ("ночь", "ночн"), ("городской", "город"), ("школьный", "школ"), ("плавание", "плава"),
    ("учительница", "учител"), ("соседка", "сосед"), ("зимний", "зим"), ("ипотечный", "ипотек"),
    ("дружеский", "друг"),  # an alternation suffix rules cannot reach
])
def test_russian_roots_come_from_the_dictionary(word, related):
    assert related in text._ru_roots(word, text.stem(word, "ru"))


@pytest.mark.parametrize("word, not_related", [
    ("полный", "пол"), ("войной", "вой"), ("речка", "реч"), ("летает", "лет"), ("банк", "бан"), ("гордый", "город"),
    ("задача", "дач"),
])
def test_dictionary_keeps_look_alikes_apart(word, not_related):
    # look-alikes (and DerivBase.Ru's own noise: пол → полк, бан → банк) are filtered by the shape of the pair
    assert not_related not in text._ru_roots(word, text.stem(word, "ru"))


def test_suffix_rules_remain_as_an_option(monkeypatch):
    monkeypatch.setattr(text, "_RU_DERIVATION", "rules")
    text._ru_roots.cache_clear()
    try:
        assert "ноч" in text._ru_roots("ночные", text.stem("ночные", "ru"))
    finally:
        text._ru_roots.cache_clear()


@pytest.mark.parametrize("word, related", [
    ("teacher", "teaches"), ("swimmer", "swims"), ("driver", "drives"), ("photography", "photograph"),
    ("investor", "invest"), ("payment", "pay"), ("delivery", "delivered"), ("decision", "decide"),
    ("explanation", "explain"), ("allergic", "allergy"), ("angry", "anger"), ("contractor", "contract"),
])
def test_english_roots_from_the_dictionary(word, related):
    # 0.11.0: Open English Wordnet's derivation links, as stem pairs; spelling changes too (decis / decid)
    assert text.stem(related, "en") in text._en_roots(word, text.stem(word, "en"))


@pytest.mark.parametrize("word, not_related", [
    ("mother", "moth"), ("mothers", "moth"), ("corners", "corn"), ("summer", "sum"), ("every", "ever"),
    ("party", "part"), ("battery", "bat"), ("factory", "fact"), ("general", "gene"), ("peter", "pet"),
    ("meters", "met"), ("classic", "class"), ("career", "car"), ("planet", "plan"), ("single", "sing"),
    ("heart", "hear"), ("passion", "pass"), ("portion", "port"), ("witness", "wit"),
])
def test_english_roots_do_not_reach_another_word(word, not_related):
    assert text.stem(not_related, "en") not in text._en_roots(word, text.stem(word, "en"))


def test_english_suffix_rules_remain_as_an_option(monkeypatch):
    monkeypatch.setattr(text, "_EN_DERIVATION", "rules")
    text._en_roots.cache_clear()
    try:
        assert "dent" in text._en_roots("dentist", text.stem("dentist", "en"))  # the dictionary has no dentist / dental
        assert "gen" not in text._en_roots("general", text.stem("general", "en"))
    finally:
        text._en_roots.cache_clear()


def test_a_copy_without_the_dictionary_files_has_no_roots(monkeypatch):
    def missing(path, *a, **k):
        raise FileNotFoundError(path)

    monkeypatch.setattr(text, "_PAIRS", {})
    monkeypatch.setattr(text.gzip, "open", missing)
    assert text._related("en", "teach") == frozenset()
    assert text._related("ru", "ноч") == frozenset()


def test_roots_only_in_text_of_a_determined_language():
    assert ("en", "teach") in text.analyze("where is the teacher").roots
    assert not text.analyze("teacher").roots  # one word: English, Spanish, French …? no roots
    assert not text.analyze("le lit est grand").roots  # French text
    assert ("ru", "ноч") in text.analyze("ночные смены").roots
    # Ukrainian text: no Russian roots («головний» is no «гол»); Ukrainian has its own since it is a language here
    assert not [r for lang, r in text.analyze("головний офіс у Києві").roots if lang == "ru"]


# -- search: a root brings in what the word's own forms miss, below the word itself ---------------------------------

@pytest.mark.parametrize("fact, query", [
    ("По ночам у нас работает резервное копирование", "ночные бэкапы"),
    ("Зимой мы ездим в горы", "зимний отдых"),
    ("Тимур ходит на плавание", "кто плавает"),
    ("Grace teaches piano on Tuesdays", "who is the teacher"),
    ("Leo swims at the club every Saturday", "where does the swimmer go"),
    ("The decision about the new school is final", "what did they decide"),
])
def test_root_only_query_finds_the_fact(store, no_roots, fact, query):
    target = store.add_fact(fact)
    store.add_fact("Пицца с ананасами продаётся на углу" if "а" in fact else "Pizza with pineapple is sold on the corner")
    assert target in _ids(store, query)
    no_roots()
    assert target not in _ids(store, query)  # it is the root that found it


def test_exact_word_ranks_above_a_root_either_way(store):
    # the query's root ("ночные" → ноч) and the fact's root ("walker" → walk)
    root_only = store.add_fact("Резервные копии делаются каждую ночь")
    exact = store.add_fact("Ночные смены у Марины по пятницам")
    assert _ids(store, "ночные смены")[0] == exact
    assert root_only in _ids(store, "ночные копии")
    long_exact = store.add_fact("The walk takes an hour and starts by the pier every Tuesday")
    walker = store.add_fact("The walker")
    found = _ids(store, "the walk")
    assert found.index(long_exact) < found.index(walker)


def test_bare_prefix_is_not_a_match(store):
    store.add_fact("The cart has wheels")
    assert _ids(store, "the car") == []


# -- recall: a root alone never recalls ---------------------------------------------------------------------------

@pytest.mark.parametrize("fact, query", [
    ("Вой собаки слышен ночью", "что с войной"),
    ("Пол в квартире покрыт паркетом", "полный"),
    ("Моя речь записана на диктофон", "речка"),
    ("Лето мы проводим в деревне", "летает"),
    ("The moth collection is in the attic", "the mothers"),
    ("The corn delivery is on Tuesday", "the corners"),
    ("The bat lives in the attic", "the battery"),
    ("The fact is in the notebook", "the factory"),
    ("The gene test is booked for Friday", "general"),
    ("Come is the dog training command", "comer"),
    ("Гол забил Иван", "головний"),
    ("The literal translation is in the drawer", "le lit"),
])
def test_look_alikes_are_not_recalled(store, fact, query):
    store.add_fact(fact)
    assert _recalled(store, query) == []


def test_a_root_adds_to_a_word_of_the_message(store):
    target = store.add_fact("Тимур по субботам ходит на плавание")
    store.add_fact("Тимур учится в третьем классе")
    assert target in _recalled(store, "когда Тимур плавает")


def test_root_population_weighs_the_term(store):
    # "ночные" is as common as every night word, not as rare as the one fact that says "ночные"
    store.add_fact("Ночные смены начинаются вечером")
    for i in range(30):
        store.add_fact(f"Ночь номер {i} наступает рано")
    term = text.query_terms("ночные")[0]
    assert _r(store)._doc_freq(term) == 31


def test_short_loose_needs_six_letters(store):
    # "больш(ое)" would leave "боль", the beginning of "больница" and "большинство": not used
    store.add_fact("Дуриан запрещён в большинстве отелей")
    assert _recalled(store, "отлично, большое спасибо") == []


# -- duplicate and conflict checks are 0.8.0's ---------------------------------------------------------------------

def test_write_check_ignores_roots(store, no_roots):
    new = store.add_fact("The teacher takes morning trains to work every Monday")
    store.add_fact("The teacher takes morning trains to work every Tuesday")
    for i in range(11):
        store.add_fact(f"The teacher takes morning trains to work every Monday blue green variant{i}")
    store.add_fact("The teaching trainer")
    with_roots = _r(store).write_check(new)
    no_roots()
    assert _r(store).write_check(new) == with_roots
    assert with_roots["possible_conflicts"]


# -- courtesy, question words ----------------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "Hi, good morning!", "Thanks, that's all for now.", "How's your day going?", "Good morning! How are you today?",
    "ok 👍", "haha nice", "Good night, sleep well", "Fine, thanks, and you?", "Have a great weekend!",
    "спасибо, на этом всё", "Привет! Как дела?", "отлично, большое спасибо", "доброе утро всем", "Как твоя неделя?",
])
def test_courtesy_messages_are_greetings(message):
    assert text.is_greeting(message)


@pytest.mark.parametrize("message", [
    "How is Tim doing at school?", "good morning routine for Leo", "what time is the morning swim",
    "thanks, and when is the dentist?", "как дела у Тимура в школе", "что сегодня на ужин",
    "доброе утро, во сколько у Яны балет?", "какая сегодня погода",
])
def test_a_question_with_courtesy_words_is_not_a_greeting(message):
    assert not text.is_greeting(message)


def test_courtesy_message_recalls_nothing(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": DIM})
    provider.initialize(session_id="s")
    provider.handle_tool_call("fact_store", {"action": "add", "content": "Mark takes his medication every morning"})
    provider.handle_tool_call("fact_store", {"action": "add", "content": "Sophie swims every Saturday morning"})
    assert provider.prefetch("Hi, good morning!") == ""
    provider.on_session_switch("s2", reset=True)
    assert "medication" in provider.prefetch("when does Mark take his medication in the morning?")
    provider.shutdown()


def test_russian_question_words_are_function_words(store):
    # "какая" was not a function word: its stem "как" met "как" in every fact that says "считать как …"
    store.add_fact("Ветку 20 трактовать как рабочие задачи")
    store.add_fact("Ветку 23 трактовать как домашние задачи")
    assert _recalled(store, "какая сегодня погода") == []
    assert not text.query_terms("какая которые сколько почему зачем куда")


@pytest.mark.parametrize("message", ["fine?", "сегодня?", "day?", "night?", "all?", "Day? Night?", "history?",
                                     "thanksgiving plans"])
def test_padding_alone_is_a_question_not_courtesy(message):
    # "fine?" may ask about a library fine, "сегодня?" about today's lesson; "history" is not "hi" + "story"
    assert not text.is_greeting(message)


def test_question_words_stay_in_facts(store):
    # left out of a query's terms only: "Иван знает, почему" and "Иван знает, куда" are different facts
    store.add_fact("Иван знает почему")
    second = store.add_fact("Иван знает куда")
    assert not _r(store).write_check(second)["similar"]


def test_root_matching_stays_within_its_language():
    fact = text.analyze("Come is the dog training command")
    assert text.query_terms("comer")[0].match(fact) == 0.0  # one word: no determined language
    assert text.query_terms("le lit est grand")[0].match(text.analyze("The literal translation is here")) == 0.0
    ukrainian = text.analyze("Иван: покупки в головний офіс")  # Ukrainian letters: no Russian roots or beginnings
    assert [t.match(ukrainian) for t in text.query_terms("Иван покупает")] == [1.0, 0.0]


def test_exact_only_search_does_not_scan_through_roots(store):
    store.add_fact("The swim")
    store.add_fact("Das ist gut")
    assert _r(store).search("in swimmer", min_trust=0, roots=False) == []


def test_capitalised_common_noun_still_finds_its_forms(store):
    babies = store.add_fact("The babies sleep upstairs")
    assert babies in _ids(store, "Where is the Baby?")


# -- Traditional and Simplified Chinese -----------------------------------------------------------------------------

def test_traditional_characters_fold_to_simplified():
    assert text.fold("學校醫院複診補習") == "学校医院复诊补习"
    assert text.analyze("學校").text == text.analyze("学校").text
    assert "们" in text.CJK_FUNCTION and text.fold("們") == "们"  # traditional function characters weigh as such


@pytest.mark.parametrize("fact, query", [
    ("女儿王小雨在实验小学上四年级，班主任姓刘。", "小雨在學校讀幾年級？"),
    ("外婆陳秀琴每三個月回醫院心臟科複診一次。", "外婆多久去一次医院复诊？"),
])
def test_cross_script_question_is_found_and_recalled(store, fact, query):
    target = store.add_fact(fact)
    store.add_fact("台北的捷運週末人很多。")
    assert target in _ids(store, query)
    assert target in _recalled(store, query)


def test_stored_text_keeps_its_script(store):
    fact_id = store.add_fact("我習慣用繁體中文回覆。")
    assert store.read("SELECT content FROM facts WHERE fact_id = ?", (fact_id,))[0][0] == "我習慣用繁體中文回覆。"
