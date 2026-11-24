"""0.11.0: German compounds split by a Wikidata dictionary, German and Spanish verb forms Snowball keeps apart from
their lemma (Wikidata Lexemes, CC0); both only in text of that language, and a copy without the data still works."""

from __future__ import annotations

import pytest

pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import text
from hermes_plugin_holographic.store import MemoryStore
from hermes_plugin_holographic.retrieval import FactRetriever

DIM = 512


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _search(store, query):
    return [r["content"] for r in FactRetriever(store, hrr_dim=DIM).search(query, limit=5)]


def _recalled(store, query):
    retriever = FactRetriever(store, hrr_dim=DIM)
    return [r["content"] for r in retriever.relevant(query, retriever.search(query, limit=10))]


@pytest.mark.parametrize("word, parts", [
    ("zahnarzttermin", ("zahn", "arzt", "termin")),
    ("familienkatze", ("familien", "katze")),
    ("kinderarzt", ("kinder", "arzt")),
    ("kinderarzte", ("kinder", "arzte")),  # Kinderärzte
    ("blumentopferde", ("blumen", "topf", "erde")),  # not Blumento-Pferde
])
def test_compounds_split(word, parts):
    assert text.de_parts(word) == parts


@pytest.mark.parametrize("word", ["staubecken", "wachstube", "kartoffel", "urlaub", "samstag", "lehrerin", "ausland"])
def test_ambiguous_words_and_simple_words_stay_whole(word):
    assert not text.de_parts(word)  # Stau-Becken / Staub-Ecken, Wach-Stube / Wachs-Tube: left whole


def test_parts_only_in_german_text():
    assert text.analyze("Lena hat einen Zahnarzttermin am Dienstag").heads >= {"termin"}
    assert not text.analyze("Lena has a Zahnarzttermin on Tuesday").heads


def test_a_compound_in_a_fact_is_found_by_its_last_part(store):
    store.add_fact("Lena hat am Dienstag einen Zahnarzttermin in der Innenstadt.")
    store.add_fact("The passport is in the drawer")
    assert _search(store, "Wann ist der Termin?")[0].startswith("Lena hat")


def test_a_compound_in_the_question_finds_its_parts(store):
    target = "Opa hat am Freitag einen Termin beim Zahnarzt in Altona."
    store.add_fact(target)
    store.add_fact("Die Hundeleine hängt an der Garderobe neben der Haustür.")
    assert _search(store, "Wann ist Opas Zahnarzttermin?")[0] == target
    assert target in _recalled(store, "Wann ist Opas Zahnarzttermin?")


def test_a_modifier_alone_never_recalls(store):
    store.add_fact("Das Geburtstagsgeschenk ist ein Fotobuch vom Sommer.")
    fact = text.analyze("Das Geburtstagsgeschenk ist ein Fotobuch vom Sommer.")
    term = text.query_terms("Wo ist das Foto?")[0]
    assert 0.0 < term.match(fact) < 1.0  # Foto is a part before the last of Fotobuch
    assert _recalled(store, "Wo ist das Foto?") == []


def test_question_head_alone_counts_as_a_root():
    fact = text.analyze("Am Abend geht Irma schwimmen und danach essen")
    term = text.query_terms("Wann ist der Elternabend?")[0]
    assert term.match(fact) == text._DERIV_WEIGHT


@pytest.mark.parametrize("fact, query", [
    ("Javier fue a Bilbao en tren para la reunión.", "¿Cómo va Javier a Bilbao?"),  # fue: ir (and ser)
    ("Carlos puso las llaves en el cajón de la cocina.", "¿Dónde pone Carlos las llaves?"),
    ("Javier pidió una pizza de jamón para cenar.", "¿Qué pide Javier para cenar?"),
    ("Lena ging am Montag mit Oma ins Kino.", "Wohin geht Lena mit Oma?"),
    ("Paul las in den Ferien drei Bücher über Dinosaurier.", "Was hat Paul in den Ferien gelesen?"),
])
def test_irregular_forms_meet_their_lemma(fact, query):
    fact_ = text.analyze(fact)
    # 0.6 of a match, like a root (0.11.0 review: a form may belong to several verbs, a name may look like one)
    assert any(t.match(fact_) == text._DERIV_WEIGHT and t.keys.isdisjoint(fact_.keys) for t in text.query_terms(query))


def test_homographs_are_not_forms():
    assert not text.word_lemmas("vino", "es")[0]  # wine, not venir
    assert not text.word_lemmas("macht", "de")[0]  # Macht, power
    assert "ir" in text.word_lemmas("fue", "es")[1] and "ser" in text.word_lemmas("fue", "es")[1]


def test_forms_only_in_text_of_that_language():
    assert not text.analyze("la casa").lemma_keys or all(lang == "es" for lang, _ in text.analyze("la casa").lemma_keys)
    assert not text.analyze("he went home").lemma_keys  # "he" is English and Spanish: a tie with English
    assert not text.analyze("ging").lemma_keys  # one word: no language


def test_other_languages_untouched():
    a = text.analyze("The family cat is a British shorthair")
    assert not a.heads and not a.mods and not a.lemma_keys and a.suffix_words is None


def test_switches_turn_both_off(monkeypatch):
    monkeypatch.setattr(text, "_DE_COMPOUNDS", False)
    monkeypatch.setattr(text, "_LEMMA_FORMS", False)
    text.de_parts.cache_clear()
    text.word_lemmas.cache_clear()
    text.analyze.cache_clear()
    try:
        a = text.analyze("Lena hat einen Zahnarzttermin und ging ins Kino")
        assert not a.heads and not a.lemma_keys and a.suffix_words is None
    finally:
        text.de_parts.cache_clear()
        text.word_lemmas.cache_clear()
        text.analyze.cache_clear()


def test_a_copy_without_the_data_files_still_works(monkeypatch, tmp_path, store):
    monkeypatch.setattr(text, "_DE_FILE", tmp_path / "missing.tsv.gz")
    monkeypatch.setattr(text, "_FORMS_FILE", tmp_path / "missing.tsv.gz")
    monkeypatch.setattr(text, "_DE_TABLE", [])
    monkeypatch.setattr(text, "_FORMS", {})
    text.de_parts.cache_clear()
    text.word_lemmas.cache_clear()
    text.analyze.cache_clear()
    try:
        assert text.de_parts("zahnarzttermin") is None
        assert text.word_lemmas("fue", "es") == ((), ())
        store.add_fact("Die Familienkatze ist eine Britisch Kurzhaar namens Zelda")
        assert _search(store, "Welche Rasse ist die Katze?")  # the suffix test of 0.6.0 still finds it
    finally:
        text.de_parts.cache_clear()
        text.word_lemmas.cache_clear()
        text.analyze.cache_clear()
