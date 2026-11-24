"""Fixes from the review of 0.11.0's language layer: a damaged Chinese dictionary, names with ı or ł in text of any
language, stems of undetermined text that are function words of another language ("media" / "dia"), lexicalised
German compounds, Japanese kanji kept apart from Traditional Chinese characters, Spanish "va" telling the language."""

from __future__ import annotations


import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import text
from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore
from hermes_plugin_holographic.text import analyze, de_parts, fold, gram_coverage, query_grams, query_terms

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


def _clear_caches():
    for f in (text.stem, text._keys, text._undotted, text._native_keys, text.fold, text._fold_ja, text.zh_words,
              text.de_parts, analyze):
        f.cache_clear()


# -- a damaged Chinese dictionary is a missing one ----------------------------------------------------------------

def test_a_damaged_chinese_dictionary_gives_no_words(monkeypatch, tmp_path):
    """A gzip header followed by garbage raises zlib.error (not EOFError, which a truncated file raises): the loader
    must treat it as missing, or every Chinese text raises and the load is retried on each of them."""
    damaged = tmp_path / "zh_words.txt.gz"
    damaged.write_bytes(b"\x1f\x8b\x08\x00garbage, not a deflate stream")
    _clear_caches()
    monkeypatch.setattr(text, "_ZH_FILE", damaged)
    monkeypatch.setattr(text, "_ZH_TABLE", None)
    try:
        assert text.zh_words("我和服务员聊天") == tuple("我和服务员聊天")  # pairs only, as without the file
        assert text.zh_seamed("我和服务员聊天") == "我和服务员聊天"
        assert text._ZH_TABLE is not None and not text._ZH_TABLE.words  # loaded once, empty
    finally:
        monkeypatch.undo()
        _clear_caches()


# -- ı and ł in text of any language --------------------------------------------------------------------------------

def test_a_name_with_dotless_i_or_l_is_searched_as_written_in_any_language():
    """FTS5's unicode61 keeps ı and ł; the comparison form writes them as i and l. A Turkish or Polish text got the
    as-written key through _native_keys; an English one ("Who is Işık?") did not, and the prefix query "isik"* could
    not reach "ışık" in the index."""
    _clear_caches()
    a = analyze("Who is Işık?")
    assert a.langs == ("en",)
    isik = next(w for w in a.words if w.text == "isik")
    assert "isık" in isik.keys  # as unicode61 indexes "Işık": I lower-cased to i, ş without its mark, ı kept
    lukasz = next(w for w in analyze("Łukasz teaches Irma").words if w.text == "lukasz")
    assert "łukasz" in lukasz.keys
    assert "łukasz" in next(w for w in analyze("Где живёт Łukasz").words if w.text == "lukasz").keys


def test_a_name_with_dotless_i_or_l_is_found_from_english_text(store):
    for fact in ("Işık is Irma's teacher", "Isik is Irma's teacher", "Łukasz is Irma's teacher"):
        store.add_fact(fact)
    assert set(_found(store, "Who is Işık?")) == {"Işık is Irma's teacher", "Isik is Irma's teacher"}
    assert _found(store, "Who is Łukasz?") == ["Łukasz is Irma's teacher"]
    # typed without the stroke it is still out of the index's reach from English text (Polish text scans the facts)
    assert _found(store, "Gdzie mieszka Lukasz?") == ["Łukasz is Irma's teacher"]


# -- stems of text with no determined language -------------------------------------------------------------------

def test_undetermined_text_keeps_only_stems_that_could_be_words():
    """Latin text of no determined language is stemmed in six languages; the Indonesian stemmer read "media" as
    me- + "dia" (he, she), a function word every Indonesian fact has, and "beer" as be- + "er"."""
    _clear_caches()
    media = next(w for w in analyze("social media").words if w.text == "media")
    assert "dia" not in media.keys and "media" in media.keys
    assert "be" not in analyze("beer").keys
    # words of one language still meet their forms
    assert "cat" in analyze("cats").keys and "walk" in analyze("walking").keys
    assert "ночн" in analyze("ночные").keys and "escuel" in analyze("escuelas").keys
    # in text of a determined language nothing changes
    assert "dia" in next(w for w in analyze("Dia suka media sosial").words if w.text == "media").keys


def test_an_english_word_does_not_meet_an_indonesian_function_word(store):
    for fact in ("Dia tinggal di Jakarta", "Dia suka kopi", "Media sosial penting untuk QX24"):
        store.add_fact(fact)
    assert _found(store, "social media") == ["Media sosial penting untuk QX24"]
    assert _recalled(store, "social media") == ["Media sosial penting untuk QX24"]


# -- lexicalised German compounds ----------------------------------------------------------------------------------

@pytest.mark.parametrize("word", ["hochzeit", "hochzeiten", "fruhstuck", "weihnachten", "augenblick", "verstand",
                                  "tablette", "tabletten", "kartoffel", "reisende", "reisenden", "wochenende",
                                  "flugzeug"])
def test_a_lexicalised_compound_stays_whole(word):
    assert de_parts(word) == ()


def test_other_compounds_are_still_split():
    assert de_parts("zahnarzttermin") == ("zahn", "arzt", "termin")
    assert de_parts("familienkatze") == ("familien", "katze")


def test_the_end_of_a_lexicalised_compound_does_not_recall_it(store):
    for fact in ("Die Hochzeit ist im Juni", "Irma hat Freizeit am Samstag", "Weihnachten feiern wir in Pattaya",
                 "Das Frühstück gibt es um acht", "Irma nimmt morgens eine Tablette"):
        store.add_fact(fact)
    assert _recalled(store, "Wann hast du Zeit?") == ["Irma hat Freizeit am Samstag"]
    assert _recalled(store, "Welche Nacht ist frei?") == []
    assert _recalled(store, "Gibt es ein Stück?") == []
    assert _recalled(store, "Wo ist der Tab?") == []
    assert _recalled(store, "Wann ist das Frühstück?") == ["Das Frühstück gibt es um acht"]


# -- Japanese kanji are not Traditional Chinese characters --------------------------------------------------------

def test_japanese_text_keeps_its_kanji():
    """The Traditional -> Simplified table rewrote Japanese kanji that share a code point with a Traditional
    character (書 -> 书, 紙 -> 纸, 達 -> 达): the comparison form of a Japanese text was Simplified Chinese, and a
    Chinese fact met a Japanese question through a kanji it had rewritten (手紙, a letter, as 手纸, toilet paper)."""
    _clear_caches()
    assert text._fold_text("手紙を書いた") == "手紙を書いた"
    assert analyze("申込書の締切").text.split("\n")[0] == "申し込み書の締め切り"  # the variants still apply
    assert fold("子ども") == "子供" and text._fold_ja("いっしょ") == "一緒"  # the table's representatives keep theirs
    assert fold("學校") == "学校"  # Chinese text (no kana) folds as before
    assert gram_coverage(query_grams("手紙を"), analyze("手纸很贵")) == 0.0


def test_a_question_in_kanji_only_still_meets_a_japanese_fact():
    """A question without kana cannot be told from Chinese and folds its kanji (締切 -> 缔切); a Japanese fact keeps
    that form of its written text too."""
    assert gram_coverage(query_grams("締切"), analyze("締切は明日")) == 1.0
    assert gram_coverage(query_grams("東京"), analyze("東京に住んでいます")) == 1.0
    assert gram_coverage(query_grams("學校"), analyze("学校在北京")) == 1.0  # Traditional / Simplified, as before


def test_a_japanese_question_does_not_meet_a_chinese_fact_through_rewritten_kanji(store):
    store.add_fact("手紙を書いた")
    store.add_fact("手纸很贵")
    assert _found(store, "手紙を書きましたか") == ["手紙を書いた"]
    store.add_fact("東京に住んでいます")
    assert _found(store, "東京") == ["東京に住んでいます"]


# -- Spanish "va" tells the language ------------------------------------------------------------------------------

def test_spanish_forms_of_ir_tell_the_language_and_stay_words():
    """"va" is a function word of French and Italian here, not of Spanish: "Irma va a la escuela" was French or
    Italian, with no Spanish stems and no verb forms. The forms vote for Spanish but are not dropped, so a question
    "¿Cómo va Javier?" still meets "fue" through the verb's forms (test_holographic_de_es)."""
    _clear_caches()
    a = analyze("Irma va a la escuela")
    assert a.determined and "es" in a.langs
    assert next(w for w in a.words if w.text == "escuela").form_lang == "es"
    assert analyze("Ana va al mercado").langs == ("es", "it")  # "al" is Spanish and Italian
    assert [t.word for t in query_terms("¿Cómo va Javier a Bilbao?")] == ["va", "javier", "bilbao"]
    assert "va" not in text.STOP["es"]
