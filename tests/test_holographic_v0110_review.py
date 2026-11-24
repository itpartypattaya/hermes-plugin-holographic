"""0.11.0: the cases an independent review (Codex CLI) found before release, each as it was reported: a fact, a message,
and what must (not) come back."""

from __future__ import annotations

import gzip

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import text
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


# -- the newer languages' stems stay out of text of no determined language -------------------------------------------

@pytest.mark.parametrize("fact, query", [
    ("The arm is broken.", "army"), ("The apple is green.", "apply"), ("The police station is nearby.", "policy"),
])
def test_polish_stems_do_not_join_english_words(store, fact, query):
    store.add_fact(fact)
    assert _recalled(store, query) == []


def test_a_polish_stem_makes_no_english_duplicate(store):
    store.add_fact("The apple is green.")
    assert _r(store).near_duplicate("apply") is None


def test_polish_and_turkish_letters_still_bring_their_stems():
    assert "pl" in text.analyze("szkoły").langs  # Polish by its letters alone (ł)
    assert "tr" in text.analyze("kitabı").langs  # Turkish by ı
    assert "pl" not in text.analyze("army").langs and "tr" not in text.analyze("army").langs


# -- a host name stays required in the newer languages ----------------------------------------------------------------

@pytest.mark.parametrize("fact, query", [
    ("qx-other sunucusunda kitaplar var.", "qx-vps'de kitap nerede?"),
    ("Il server qx-other è nel garage.", "dov'è il server qx-vps?"),
    ("De server qx-other staat in de garage.", "waar staat de server qx-vps?"),
])
def test_a_host_name_stays_required(store, fact, query):
    store.add_fact(fact)
    assert _recalled(store, query) == []


# -- Japanese: no variant inside another hiragana word; a kanji-only question meets a folded fact ----------------------

def test_a_hiragana_variant_is_not_replaced_after_a_particle_letter(store):
    store.add_fact("銀杏の木")
    assert _recalled(store, "かいちょうは誰？") == []


def test_a_kanji_only_question_meets_a_fact_folded_by_its_kana(store):
    target = store.add_fact("締切は明日")
    assert target in _ids(store, "締切")


def test_a_spelling_variant_alone_finds_the_fact(store, monkeypatch):
    target = store.add_fact("きょうは雨です。")
    store.add_fact("車は二台です。")
    assert _ids(store, "今日の天気は？")[:1] == [target]
    monkeypatch.setattr(text, "_JA_VARIANTS", False)
    text.analyze.cache_clear()
    try:
        assert target not in _ids(store, "今日の天気は？")  # only the variant joins them
    finally:
        text.analyze.cache_clear()


# -- English: a link of a rare sense is not applied to the common one --------------------------------------------------

@pytest.mark.parametrize("word, other", [
    ("butter", "butt"), ("tower", "tow"), ("pearl", "pear"), ("letter", "let"), ("format", "form"), ("liner", "line"),
])
def test_english_pairs_of_a_rare_sense_are_left_out(word, other):
    assert text.stem(other, "en") not in text._en_roots(word, text.stem(word, "en"))


def test_a_shared_name_and_a_rare_sense_recall_nothing(store):
    store.add_fact("The butter belongs to Alice.")
    assert _recalled(store, "Is Alice allergic to butt pain?") == []


# -- Chinese: a one-character word does not recall a fact alone -------------------------------------------------------

def test_a_one_character_word_does_not_recall_an_idiom(store):
    store.add_fact("合同里有猫腻。")
    assert _recalled(store, "猫几岁了？") == []


def test_a_one_character_word_with_a_pair_still_recalls(store):
    target = store.add_fact("家里的橘猫今年三岁了。")
    assert target in _recalled(store, "猫几岁了？")


# -- the word scan does not depend on the probe words ------------------------------------------------------------------

@pytest.mark.parametrize("fact, query", [
    ("Kitabı masada bıraktım.", "Kitap nerede?"),
    ("Szkoła zamknięta.", "Czy szkola jest otwarta?"),
])
def test_a_question_in_a_scanned_language_scans_a_store_without_probe_words(store, fact, query):
    target = store.add_fact(fact)
    assert target in _ids(store, query)


# -- Turkish greetings ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("query", ["Günaydın canlarım!", "Tamam, akşam görüşürüz"])
def test_turkish_greetings_recall_nothing(store, query):
    store.add_fact("Can lives in Ankara.")
    store.add_fact("Can akşamları yüzmeye gider.")
    assert _recalled(store, query) == []


# -- a damaged data file is a missing one --------------------------------------------------------------------------

def test_a_truncated_dictionary_gives_no_roots(monkeypatch):
    def truncated(path, *a, **k):
        raise EOFError("Compressed file ended before the end-of-stream marker was reached")

    monkeypatch.setattr(text, "_PAIRS", {})
    monkeypatch.setattr(text.gzip, "open", truncated)
    assert text._related("en", "teach") == frozenset()


def test_a_truncated_japanese_table_gives_no_variants(monkeypatch, tmp_path):
    damaged = tmp_path / "ja_variants.tsv.gz"
    whole = gzip.compress("".join(f"子ども{i}\t子供{i}\n" for i in range(2000)).encode("utf-8"))
    damaged.write_bytes(whole[:len(whole) // 2])
    monkeypatch.setattr(text, "_JA_FILE", damaged)
    monkeypatch.setattr(text, "_JA_TABLE", None)
    assert text._ja_table() == ({}, {})


# -- second review: German and Spanish ---------------------------------------------------------------------------

@pytest.mark.parametrize("fact, query", [
    ("Si Ana tuviese tiempo, viajaría.", "¿Qué contiene la caja?"),  # "tuviese" under "contener" in the source
    ("Paul hat das Buch auf den Tisch gelegt.", "Wo liegt die Katze?"),  # "gelegt" under "liegen" in the source
    ("Lucía vive en Madrid.", "¿Dónde luce?"),  # a name spelled like a form of "lucir"
    ("La contraseña fue cambiada ayer.", "¿Adónde va?"),  # "fue" is "ser" here, not "ir"
])
def test_a_verb_form_alone_never_recalls(store, fact, query):
    store.add_fact(fact)
    assert _recalled(store, query) == []


def test_a_verb_form_counts_as_a_root(monkeypatch):
    fact = text.analyze("Lena ging am Montag mit Oma ins Kino.")
    geht = next(t for t in text.query_terms("Wohin geht Lena mit Oma?") if t.word == "geht")
    assert geht.match(fact) == text._DERIV_WEIGHT
    monkeypatch.setattr(text, "_LEMMA_FORMS", False)
    text.analyze.cache_clear()
    try:
        fact = text.analyze("Lena ging am Montag mit Oma ins Kino.")
        assert next(t for t in text.query_terms("Wohin geht Lena mit Oma?") if t.word == "geht").match(fact) == 0.0
    finally:
        text.analyze.cache_clear()


def test_a_three_letter_head_alone_never_recalls(store):
    store.add_fact("Der Alltag ist stressig.")
    assert _recalled(store, "Welcher Tag ist frei?") == []


def test_a_host_name_stays_required_in_a_tie_with_dutch(store):
    store.add_fact("De server qx-other staat in de garage.")
    assert _recalled(store, "De server qx-vps") == []


def test_a_malformed_compound_row_is_a_damaged_file(monkeypatch, tmp_path):
    damaged = tmp_path / "de_compounds.tsv.gz"
    damaged.write_bytes(gzip.compress(b"haus\tN\n"))
    monkeypatch.setattr(text, "_DE_FILE", damaged)
    monkeypatch.setattr(text, "_DE_TABLE", [])
    text.de_parts.cache_clear()
    text.analyze.cache_clear()
    try:
        assert text.analyze("Das Haus ist rot.").words
    finally:
        text.de_parts.cache_clear()
        text.analyze.cache_clear()


def test_a_question_compound_needs_the_splitter(store, monkeypatch):
    target = store.add_fact("Opa hat am Freitag einen Termin beim Zahnarzt in Altona.")
    store.add_fact("Die Hundeleine hängt an der Garderobe neben der Haustür.")
    assert target in _recalled(store, "Wann ist Opas Zahnarzttermin?")
    monkeypatch.setattr(text, "_DE_COMPOUNDS", False)
    text.analyze.cache_clear()
    text.de_parts.cache_clear()
    try:
        assert target not in _recalled(store, "Wann ist Opas Zahnarzttermin?")
    finally:
        text.analyze.cache_clear()
        text.de_parts.cache_clear()
