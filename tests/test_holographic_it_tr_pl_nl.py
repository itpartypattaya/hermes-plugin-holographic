"""Italian, Turkish, Polish, Dutch (0.11.0): Snowball stems loaded when the installed snowballstemmer has them,
Turkish İ / ı and apostrophes, Italian elision, Polish word forms and ł, the Dutch 's, detection between neighbouring
languages, function words and courtesy messages that recall nothing."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic import text as _text
from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore
from hermes_plugin_holographic.text import analyze, fold, is_greeting, query_terms, words

DIM = 512


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": DIM})
    p.initialize(session_id="s")
    yield p
    p.shutdown()


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _found(store, query):
    return [f["content"] for f in FactRetriever(store, hrr_dim=DIM).search(query, min_trust=0)]


def _meets(fact: str, query: str) -> bool:
    return any(t.found_in(analyze(fact)) for t in query_terms(query))


def _clear_caches():
    for f in (_text.stem, _text._keys, _text._native_stems, _text._native_keys, analyze):
        f.cache_clear()


# -- stemmers ----------------------------------------------------------------------------------------------------------

class _Snowball:
    """A snowballstemmer that has every algorithm the plugin asks for except *missing*."""

    def __init__(self, missing=()):
        self.missing = set(missing)

    def algorithms(self):
        return [name for _, name in _text._STEMMER_NAMES if name not in self.missing]

    def stemmer(self, name):
        return name


def test_a_stemmer_is_loaded_only_when_the_installed_version_has_it():
    assert set(_text._load_stemmers(_Snowball())) >= {"it", "tr", "pl", "nl"}
    loaded = _text._load_stemmers(_Snowball(missing={"polish"}))  # snowballstemmer 2.x
    assert "pl" not in loaded and {"it", "tr", "nl", "en", "ru"} <= set(loaded)


def test_polish_without_its_stemmer_still_works(monkeypatch):
    """snowballstemmer 2.x has no Polish: words compare as written (folded); function words, greetings and the
    detection of the language are unchanged."""
    _clear_caches()
    monkeypatch.setattr(_text, "_STEMMERS", {k: v for k, v in _text._STEMMERS.items() if k != "pl"})
    try:
        a = analyze("Irma ma alergię na orzeszki")
        assert a.langs == ("pl",)
        assert [w.text for w in a.words if not w.stop] == ["irma", "alergie", "orzeszki"]
        assert _meets("Irma ma alergię na orzeszki", "orzeszki") and _meets("Irma ma alergię na orzeszki", "alergię")
        assert is_greeting("Dzięki, do jutra!")
    finally:
        monkeypatch.undo()
        _clear_caches()


# -- Turkish -----------------------------------------------------------------------------------------------------------

def test_turkish_dotted_and_dotless_i_fold_as_one():
    assert words("İstanbul ISTANBUL istanbul") == ["istanbul"] * 3
    assert fold("ışık") == fold("IŞIK".lower()) == "isik"  # Python lower-cases I to i, Turkish to ı
    assert fold("kızım") == fold("KIZIM".lower()) == fold("kizim")


def test_turkish_words_typed_with_or_without_their_letters_meet(store):
    store.add_fact("Toplantı saat üçte başlıyor")
    store.add_fact("Kitabı masanın üstünde bıraktım")
    store.add_fact("Ailenin kedisi Zelda adında bir British Shorthair")
    for query in ("toplantı ne zaman?", "toplanti ne zaman?", "TOPLANTI NE ZAMAN?"):
        assert _found(store, query)[0].startswith("Toplantı"), query
    assert _found(store, "Kitap nerede?")[0].startswith("Kitabı")  # kitap / kitabı: the scan sees it
    assert _found(store, "Kedi hangi cins?")[0].startswith("Ailenin")  # kedisi: Snowball leaves "kedis"


def test_turkish_apostrophe_takes_the_ending_off_a_name():
    assert words("Ankara'da Ayşe'nin QX24'ün qx-vps'de 03:00'te") == ["ankara", "ayse", "qx24", "qx-vps", "03:00"]
    assert "ankara'da" in analyze("Ankara'da yaşıyor").bag  # the whole stays for containment
    # not in English text, nor for an ending Turkish does not have
    assert words("Irma's dentist at 9 o'clock") == ["irma's", "dentist", "at", "9", "o'clock"]
    assert "rock'n'roll" in words("rock'n'roll")


def test_a_turkish_name_with_an_ending_is_mentioned(provider):
    _call(provider, action="add", content="Ayşe Pattaya'da yaşıyor", entities=["Ayşe"])
    store = provider._store
    ayse = store.entity_ids("Ayşe")
    assert ayse and set(ayse) <= store.mentioned_entity_ids("Ayşe'nin okulu çok uzak")


def test_turkish_identifier_with_an_ending_is_still_required(provider):
    _call(provider, action="add", content="qx-other her gece yedek alıyor")
    _call(provider, action="add", content="QX24 Copilot mülakatları yazıya döküyor")
    assert provider.prefetch("QX24'ün işi ne?")  # "QX24'ün" is QX24
    assert "qx-other" not in provider.prefetch("qx-vps'de ne kadar RAM var?")


# -- Italian -----------------------------------------------------------------------------------------------------------

def test_italian_elision():
    assert words("Dov'è l'appuntamento dell'anno? Quest'anno com'è andato, cos'è successo") == [
        "e", "appuntamento", "anno", "anno", "e", "andato", "e", "successo"]
    assert words("Un'agenzia nell'ufficio c'è") == ["agenzia", "ufficio", "e"]
    assert words("the dev's laptop and Bell's palsy") == ["the", "dev's", "laptop", "and", "bell's", "palsy"]


def test_italian_word_forms_meet():
    for fact, query in [("Irma è allergica alle arachidi", "allergia"),
                        ("Le lezioni di nuoto sono il sabato", "lezione"),
                        ("La riunione con i clienti", "riunioni"), ("Il volo per Milano", "voli")]:
        assert _meets(fact, query), (fact, query)


# -- Polish ------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif("pl" not in _text._STEMMERS, reason="snowballstemmer before 3.1 has no Polish stemmer")
@pytest.mark.parametrize("fact,query", [
    ("Dom babci jest duży", "co jest w domu?"), ("Kupili nowy dom", "domem"), ("Córka Anny ma pięć lat", "córki"),
    ("Córka Anny ma pięć lat", "córką"), ("Lekarz przyjmuje w środę", "lekarza"),
    ("Irma ma alergię na orzeszki", "alergia"), ("Szkoła Zosi jest blisko", "szkole"),
    ("Szkoła Zosi jest blisko", "szkola"), ("Rodzina planuje przeprowadzkę", "kiedy się przeprowadzamy?")])
def test_polish_word_forms_meet(fact, query):
    assert _meets(fact, query)


def test_polish_l_with_stroke_is_searched_as_written_and_as_l(store):
    store.add_fact("Szkoła Zosi jest niedaleko domu")
    assert _found(store, "Gdzie jest szkoła Zosi?") and _found(store, "gdzie jest szkola?")
    if "pl" in _text._STEMMERS:  # the Polish stem; snowballstemmer before 3.1 has none
        assert "szkoł" in analyze("szkoła").keys  # the form FTS5 indexed (unicode61 keeps ł)


# -- Dutch -------------------------------------------------------------------------------------------------------------

def test_dutch_s_is_the_word():
    assert words("Irma is allergisch voor pinda's") == ["irma", "is", "allergisch", "voor", "pinda"]
    assert _meets("Irma is allergisch voor pinda's", "pinda")
    assert words("Irma's dentist is on Friday")[0] == "irma's"  # English text keeps it


def test_dutch_compounds_are_not_taken_apart_by_default():
    assert not analyze("De reservesleutels liggen in de keukenla").dutch  # _NL_COMPOUNDS: measured unsafe
    assert not _meets("Opa ligt in het ziekenhuis", "Hoeveel slaapkamers heeft ons huis?")


@pytest.mark.parametrize("fact,query", [
    ("Het gezin gaat naar Hua Hin", "Heb je zin in pizza?"),
    ("Dit gerecht is erg pittig", "Heeft hij recht op vakantie?"),
    ("Zijn gezicht was rood", "Wat een mooi zicht op zee"), ("We hebben het hotel geboekt", "Welk boek lees je?")])
def test_dutch_ge_is_part_of_a_noun(fact, query):
    """Snowball's Dutch takes ge- off ("gezin", family, would be "zin", sense); it is put back."""
    assert not _meets(fact, query)


def test_dutch_plural_and_diminutive_meet_their_noun():
    if _text._STEMMERS["nl"].stemWord("katje") != "kat":
        pytest.skip("this snowballstemmer's Dutch is Porter's (2.x): it keeps diminutives")
    assert _meets("Lotte heeft een katje", "Waar is de kat van Lotte?")
    assert _meets("Drie gezinnen wonen hier", "Hoe groot is het gezin?")  # ge- put back on both


# -- detection between neighbours --------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,langs", [
    ("Dov'è la gatta di Ana?", ("it",)), ("¿Dónde está la gata de Ana?", ("es",)),
    ("Onde está a gata da Ana?", ("pt",)),
    ("Waar is de kat van Ana?", ("nl",)), ("Wo ist die Katze von Ana?", ("de",)), ("Where is the cat?", ("en",)),
    ("Ana'nın kedisi nerede?", ("tr",)), ("Di mana kucing yang hilang itu?", ("id",)), ("Gdzie jest kot Ani?", ("pl",)),
    ("Kim jest Leon?", ("pl",)), ("Leon kimi arıyor?", ("tr",))])
def test_each_language_by_its_function_words(text, langs):
    assert analyze(text).langs == langs


@pytest.mark.parametrize("text,kept", [
    ("Ana come carne todos los días", ["ana", "come", "carne", "todos", "dias"]),  # Spanish "come", not Italian "how"
    ("Ana no come carne", ["ana", "come", "carne"]),
    ("Leon met Irma at the door", ["leon", "met", "irma", "door"]),  # English, not Dutch "met", "door"
    ("Mia Moss lives in Rome", ["mia", "moss", "lives", "rome"]),  # a name, not Italian "mia"
    ("Kim Lane lives in Oslo", ["kim", "lane", "lives", "oslo"]),  # not Turkish "kim" (who)
    ("Bo Lin lives in Hue", ["bo", "lin", "lives", "hue"]),  # not Polish "bo" (because)
    ("Berapa besar anggaran pemasaran per bulan?", ["besar", "anggaran", "pemasaran", "per", "bulan"]),
    ("Schafft der Bürocomputer KI-Modelle?", ["schafft", "burocomputer", "ki-modelle"]),
    ("We decided to ship on Fridays", ["decided", "ship", "fridays"])])
def test_a_new_language_does_not_take_words_from_its_neighbours(text, kept):
    assert [t.word for t in query_terms(text)] == kept


def test_weak_words_are_function_words_of_their_language():
    for lang, weak in _text._WEAK.items():
        assert weak <= _text.STOP[lang], lang
    # dropped where Italian alone is best supported, kept where the other words fit French as well ("il")
    terms = [t.word for t in query_terms("Come sono le lezioni di Irma e dove si tengono?")]
    assert terms == ["lezioni", "irma", "tengono"]
    assert [t.word for t in query_terms("Come prende il caffè Leon?")] == ["come", "prende", "caffe", "leon"]



# Before 0.33.1: Turkish "ve" (and), its commonest word, voted for Italian too ("ve", there), and Turkish
# postpositions and determiners were no function words, so a fifth of Turkish sentences tied with Italian or were of no
# language. Polish "w" (in) and "z" (with), its commonest words, did not vote, being initials elsewhere.

@pytest.mark.parametrize("text", [
    "Proje Ahmet ve Mehmet tarafından hazırlandı", "Annem ve babam yarın geliyor",
    "Bazı arkadaşlar toplantıya katılmadı",
])
def test_turkish_by_ve_and_its_postpositions(text):
    assert analyze(text).langs == ("tr",)


@pytest.mark.parametrize("text, lang, word", [
    ("Proje Ahmet ve Mehmet tarafından hazırlandı", "tr", "tarafindan"),
    ("Podczas spotkania omówiliśmy budżet", "pl", "podczas"),
    ("De grot bevindt zich op een berg", "nl", "zich"),
])
def test_the_new_voters_stay_search_words(text, lang, word):
    """They tell the language but are not dropped: as function words they cost retrieval (Dutch 0.7 points on
    FLORES-200)."""
    assert analyze(text).langs == (lang,)
    assert word in [t.word for t in query_terms(text)]


@pytest.mark.parametrize("text, lang", [
    ("Ve lo dico io", "it"),  # Italian "ve" before a clitic; alone it is Turkish "and"
    ("Esa poza azul", "es"),  # Spanish "poza" (pool), no Polish "poza" (beyond)
])
def test_old_languages_keep_their_words(text, lang):
    assert analyze(text).langs == (lang,)


@pytest.mark.parametrize("text", [
    "Spotkanie w piątek z Anną", "Mieszkam w Warszawie od trzech lat", "Podczas spotkania omówiliśmy budżet",
    "W środę idę do kina",  # a capital "W" that starts a sentence
])
def test_polish_by_w_z_and_more_function_words(text):
    assert analyze(text).langs == ("pl",)


@pytest.mark.parametrize("text", ["George W. Bush visited Warsaw", "Gen Z shoppers like TikTok", "from a to z",
                                  "gen z trends", "w !", "George W Bush", "W Edwards Deming", "W hotels are nice",
                                  "Book a room at W hotel"])
def test_w_and_z_as_initials_or_letters_are_no_polish(text):
    assert "pl" not in analyze(text).langs


def test_a_place_name_is_no_turkish_word():
    """Hatta (a town in Dubai, an Indonesian name) is no voter, though Turkish "hatta" (even) is a function word."""
    assert "tr" not in analyze("Hatta hiking").langs

# -- courtesy and function words recall nothing ------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "Buongiorno a tutti!", "Grazie mille, a domani!", "Ciao ragazzi, come state?", "Va bene, a presto",
    "Günaydın herkese!", "Çok teşekkürler, görüşürüz!", "Merhaba arkadaşlar, nasılsınız?", "Tamam, sağ ol!",
    "Dzień dobry wszystkim!", "Wielkie dzięki, do jutra!", "Cześć, co słychać?", "Okej, dzięki!",
    "Goedemorgen allemaal!", "Heel erg bedankt, tot morgen!", "Hoi, hoe gaat het?", "Oké, dank je wel!"])
def test_courtesy_in_the_four_languages(message):
    assert is_greeting(message)


@pytest.mark.parametrize("message", ["Ciao Irma, quando hai nuoto?", "Merhaba, toplantı ne zaman?", "oggi?",
                                     "Cześć, gdzie są klucze?", "Hoi, waar is de sleutel?"])
def test_a_question_after_a_greeting_is_a_question(message):
    assert not is_greeting(message)


@pytest.mark.parametrize("facts,message", [
    (["Il giorno della riunione è il lunedì alle nove", "La sera Leon va in palestra"], "Buongiorno, buona serata!"),
    (["Toplantı her gün sabah dokuzda", "Akşam yemeği saat yedide"], "Günaydın, iyi akşamlar!"),
    (["Spotkanie jest w każdy dzień roboczy", "Wieczór filmowy w piątek"], "Dzień dobry, dobry wieczór!"),
    (["De vergadering is elke dag om negen uur", "Morgen gaat Irma naar de tandarts"], "Goedemorgen, tot morgen!"),
    (["La riunione è il lunedì alle nove"], "Qual è la capitale dell'Australia?"),
    (["Toplantı her pazartesi saat dokuzda"], "Avustralya'nın başkenti neresi?"),
    (["Spotkanie jest w każdy poniedziałek"], "Jaka jest stolica Australii?"),
    (["De vergadering is elke maandag om negen uur"], "Wat is de hoofdstad van Australië?")])
def test_courtesy_and_function_words_recall_nothing(provider, facts, message):
    for fact in facts:
        _call(provider, action="add", content=fact)
    assert provider.prefetch(message) == ""


def test_a_store_of_a_scanned_language_is_scanned(store, monkeypatch):
    store.add_fact("Kitabı masanın üstünde bıraktım, çünkü bir toplantı vardı")
    retriever = FactRetriever(store, hrr_dim=DIM)
    assert retriever._scan_terms(query_terms("kitap nerede"))  # Turkish facts: the scan runs
