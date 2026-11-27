"""African languages (0.33.0): Hausa hooked letters typed as plain ones; Swahili, Hausa, Yoruba, Somali, Zulu and
Afrikaans told by their function words; Amharic and Tigrinya in Ethiopic script, Amharic with a light stemmer.
Measured on FLORES-200 and MasakhaNER by tools/african_probe.py."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore
from hermes_plugin_holographic.text import analyze, content_keys, fold, query_grams, query_terms, soft_overlap, words

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


# -- Hausa ɓ ɗ ƙ ƴ ---------------------------------------------------------------------------------------------------
# Hausa writes four hooked letters that phone and laptop keyboards mostly lack; people type b, d, k, y instead
# ("kasa" for "ƙasa", "dan" for "ɗan"). A question typed so must meet a fact written with the hooks, and the other way.

def test_hausa_hooked_letters_compare_as_plain_ones():
    assert fold("ɓera") == "bera"
    assert fold("ɗana") == "dana"
    assert fold("ƙwallon") == "kwallon"
    assert fold("ƴaƴa") == "yaya"
    assert fold("Ɗana".lower()) == "dana"


HAUSA = ("Ɗanmu Musa yana son ƙwallon ƙafa sosai",  # our son Musa loves football
         "Matarsa tana aiki a asibitin Kano")  # his wife works at the Kano hospital


@pytest.mark.parametrize("query", ["kwallon kafa", "Wane ne yake son kwallon kafa?", "ƙwallon ƙafa"])
def test_a_hausa_question_finds_a_fact_with_or_without_the_hooks(store, query):
    for fact in HAUSA:
        store.add_fact(fact)
    assert _found(store, query)[:1] == [HAUSA[0]]
    assert _recalled(store, query) == [HAUSA[0]]


@pytest.mark.xfail(strict=True, reason="known limit: one Hausa function word does not decide a short message "
                   "(_TWO_WORDS), and Spanish/French 'son' makes it a language whose words the index serves, so "
                   "the plain spelling cannot reach the hooked one; the same as Polish ł from English text")
def test_a_hausa_question_with_one_hausa_function_word_finds_the_hooked_fact(store):
    for fact in HAUSA:
        store.add_fact(fact)
    assert _found(store, "Wa yake son kwallon kafa?")[:1] == [HAUSA[0]]


@pytest.mark.parametrize("query", ["ƙwallon ƙafa", "Wane ne yake son ƙwallon ƙafa?"])
def test_a_hausa_question_with_the_hooks_finds_a_fact_typed_without_them(store, query):
    plain = "Danmu Musa yana son kwallon kafa sosai"
    store.add_fact(plain)
    store.add_fact(HAUSA[1])
    assert _found(store, query)[:1] == [plain]
    assert _recalled(store, query) == [plain]


@pytest.mark.parametrize("fact, query", [("Ɓera ta shiga gida", "ɓera"), ("Ɗanmu Musa likita ne", "ɗanmu"),
                                         ("Ɗanmu Musa likita ne", "Ɗanmu")])
def test_a_word_that_begins_with_a_hook_is_found_as_written(store, fact, query):
    """ɓ and ɗ are IPA-extension letters: read as not Latin, the word lost its as-written key and the index missed it."""
    store.add_fact(fact)
    assert _found(store, query) == [fact]


def test_a_hooked_letter_in_the_tags_turns_the_scan_on(store):
    store.add_fact("Training schedule for the boys", tags="ƙwallon ƙafa")
    assert _found(store, "kwallon kafa") == ["Training schedule for the boys"]


def test_a_store_without_hooks_or_scan_languages_pays_for_no_scan(store):
    """The probe must not turn the word scan on for an English store: a question of undetermined language there is
    served by the index."""
    store.add_fact("Leon rides a scooter around the bay every day")
    r = FactRetriever(store, hrr_dim=DIM)
    terms = query_terms("kwallon kafa")
    assert any(t.scan for t in terms)
    assert r._scan_terms(terms, query="kwallon kafa") == []
    store.add_fact("Ɗanmu yana son ƙwallon ƙafa")
    assert FactRetriever(store, hrr_dim=DIM)._scan_terms(terms, query="kwallon kafa")


def test_hooked_and_plain_spellings_have_the_same_content_keys():
    """The duplicate and conflict checks of the provider compare content_keys: the same sentence typed without the
    hooks is the same claim (the store itself refuses only the exact text again, in every language)."""
    hooked, plain = content_keys(HAUSA[0]), content_keys("Danmu Musa yana son kwallon kafa sosai")
    assert soft_overlap(hooked, plain) == (len(hooked), len(plain))


def test_plain_letters_get_no_hooked_keys():
    """Only a word written with a hook is also searched as written; "kid", "baby" stay as they are."""
    keys = analyze("The kid plays with the baby daily").keys
    assert not any(c in "ɓɗƙƴ" for k in keys for c in k)
    assert "ƙafa" in analyze("ƙafa").keys and "kafa" in analyze("ƙafa").keys


# -- function words: which language a text is in --------------------------------------------------------------------
# Before (0.31.7): none of these languages was known; Swahili and Hausa text was read as Spanish or French by a word
# or two it shares with them ("la", "son"), Afrikaans as Polish ("nie"), Yoruba as Spanish ("ni", "si"), and their own
# grammar words stayed search words.

def _stops(text):
    return {w.text for w in analyze(text).words if w.stop}


@pytest.mark.parametrize("text, lang, grammar, content", [
    ("Mtoto wangu anasoma katika shule ya msingi huko Nairobi", "sw", {"wangu", "katika", "ya", "huko"},
     {"mtoto", "anasoma", "shule", "msingi", "nairobi"}),
    ("Musa yana aiki a cikin asibitin Kano kuma yana son wannan aiki", "ha", {"yana", "cikin", "kuma", "wannan"},
     {"musa", "aiki", "asibitin", "kano", "son"}),
    ("Awon omo naa ti lo si ile iwe ni ilu Eko", "yo", {"awon", "naa", "ti", "si", "ni"},
     {"omo", "lo", "ile", "iwe", "ilu", "eko"}),
    ("Wiilkaygu wuxuu dhigtaa iskuul ku yaal Muqdisho waana arday fiican", "so", {"wuxuu", "ku"},
     {"wiilkaygu", "dhigtaa", "iskuul", "muqdisho", "arday", "fiican"}),
    ("Ingane lami lifunda esikoleni futhi lithanda kakhulu ibhola", "zu", {"futhi", "kakhulu"},
     {"ingane", "lami", "lifunda", "esikoleni", "lithanda", "ibhola"}),
    ("My seun gaan nie vandag skool toe nie want hy is baie siek", "af", {"nie", "want", "hy", "is", "baie"},
     {"seun", "vandag", "skool", "siek"}),
])
def test_an_african_text_is_told_by_its_function_words(text, lang, grammar, content):
    a = analyze(text)
    assert a.determined and lang in a.langs
    assert grammar <= _stops(text)
    assert not content & _stops(text)


@pytest.mark.parametrize("text, word", [
    ("Ninataka kutoka hapa", "kutoka"),  # Swahili "to leave" (also "from")
    ("Musa yana godiya kuma ya tafi", "godiya"),  # Hausa "gratitude"
    ("Yaushe ne sabon wata zai fito?", "wata"),  # Hausa "moon, month"
    ("Mo ra eyin mejila ni oja", "eyin"),  # Yoruba "eggs"
])
def test_content_words_of_the_new_languages_stay_search_words(text, word):
    assert word not in _stops(text)


@pytest.mark.parametrize("text, lang", [
    ("The lake was pale and the game was fun, so we can take a walk", "en"),
    ("Die Katze ist schwarz und sehr alt", "de"),
    ("Nie wiem, gdzie jest moja szkoła", "pl"),
    ("Ana come carne todos los días con su hermano", "es"),
    ("Het huis van mijn ouders is heel oud", "nl"),
    ("Il lago è molto bello in estate", "it"),
])
def test_old_languages_stay_theirs(text, lang):
    a = analyze(text)
    assert a.langs == (lang,)


@pytest.mark.parametrize("text", [
    "Kanye albums", "PE classes", "Mo meetings", "Wale beobachten", "Baie vitrée", "Sana güveniyorum",
    "Kati kommt morgen", "Ankara ili", "Kuma oturdum", "Kaas kopen", "Lakin gelmedi", "Noma reservation",
    "Wees voorzichtig", "Quer vir amanhã?", "Pé direito", "Tabi canım", "Comida sana", "Shi Tzu puppy",
])
def test_one_word_of_a_new_language_does_not_decide_a_short_message(text):
    """Review (Fable, Codex): with one new-language word these became Zulu, Swahili, Somali... — a language with no
    stemmer — and lost the stems and the word scan they had as text of undetermined language."""
    assert not analyze(text).determined


@pytest.mark.parametrize("text, stop", [
    ("Saya suka nasi goreng", "saya"), ("Book Noma for Friday", "for"), ("PE class at school", "at"),
    ("Mo is late", "is"), ("Laga final malam ini", "ini"),
])
def test_a_new_language_word_does_not_unmake_an_old_function_word(text, stop):
    assert stop in _stops(text)


@pytest.mark.parametrize("text, word", [
    ("The lake was pale", "lake"), ("The lake was pale", "pale"), ("We can take the game", "game"),
    ("We can take the game", "take"), ("Mimi lives in Lagos", "mimi"), ("Yao called Ana", "yao"),
    ("Yao called Ana", "ana"), ("Amma and Musa are here", "amma"), ("Wale called Wangu", "wale"),
    ("Wale called Wangu", "wangu"),
])
def test_english_words_and_names_stay_search_words(text, word):
    assert word not in _stops(text)


def test_a_latin_word_in_chinese_text_is_not_taken_for_a_hausa_function_word():
    """The language-blind list (STOPWORDS) holds of the new languages only words that are no word elsewhere."""
    assert "game" in dict(query_grams("这是game吗"))


# -- Ethiopic script: Amharic and Tigrinya ---------------------------------------------------------------------------

AMHARIC = ("ልጄ በአዲስ አበባ ትምህርት ቤት ይማራል።",  # my child studies at a school in Addis Ababa
           "ባለቤቴ በሆስፒታል ውስጥ ትሰራለች።")  # my wife works in a hospital


def test_amharic_sure_stems_undo_prefixes_plurals_and_plural_possessives():
    from hermes_plugin_holographic.text import _amharic_stem
    assert _amharic_stem("ቤቶች") == "ቤት"  # houses
    assert _amharic_stem("ልጆች") == "ልጅ"  # children
    assert _amharic_stem("ቤታችን") == "ቤት"  # our house
    assert _amharic_stem("ቤቶቻችን") == "ቤት"  # our houses
    assert _amharic_stem("በአዲስ") == "አዲስ"  # in Addis
    assert _amharic_stem("ሐሙስ") == _amharic_stem("ሀሙስ")  # Thursday, written with either h
    assert _amharic_stem("ቤት") == "ቤት" and _amharic_stem("በር") == "በር"


def test_amharic_unsure_forms_are_roots_not_keys():
    """A one-letter suffix cannot be told from the word's own last letter: ox (በሬ) is no door (በር), the world (ዓለም)
    no "there is" (አለ). Such forms are compared as same-root words, which never recall a fact alone."""
    for word, other in (("በሬ", "በር"), ("ዓለም", "አለ"), ("ወንድም", "ወንድ"), ("ሰላም", "ሰላ")):
        w = analyze(word).words[0]
        assert other not in w.keys
    w = analyze("ልጄ").words[0]
    assert "ልጅ" in w.roots and "ልጅ" not in w.keys
    w = analyze("በሬዎች").words[0]  # oxen: the noun begins with በ, the plural still meets በሬ
    assert "በሬ" in w.keys


@pytest.mark.parametrize("query", ["ልጆች የት ይማራሉ?", "ልጆቼ የት ይማራሉ?", "ስለ ልጆቻችን"])
def test_an_amharic_question_finds_the_fact_through_its_forms(store, query):
    """children / my children / about our children: no word of the question is written as in the fact."""
    for fact in AMHARIC:
        store.add_fact(fact)
    assert _found(store, query)[:1] == [AMHARIC[0]]


@pytest.mark.parametrize("fact, query", [
    ("ጓደኛዬ አበበ ዶክተር ነው", "ጓደኛ"),  # my friend Abebe is a doctor / friend
    ("ውሻው ቡችላ ወለደ", "ውሻ"),  # the dog had a puppy / dog
    ("ውሾች ከቤት ውጭ ይተኛሉ", "ውሻ"),  # dogs sleep outside / dog
    ("ልጆቹ ትምህርት ቤት ናቸው", "ልጆች መጡ ወይ?"),  # the children are at school / did the children come?
])
def test_amharic_nouns_that_end_in_a_vowel_meet_their_forms(store, fact, query):
    store.add_fact(fact)
    store.add_fact(AMHARIC[1])
    assert _found(store, query)[:1] == [fact]


def test_amharic_spelling_variants_meet(store):
    store.add_fact("ሐሙስ ስብሰባ አለኝ")  # I have a meeting on Thursday
    store.add_fact(AMHARIC[1])
    assert _found(store, "ሀሙስ ምን አለ?")[:1] == ["ሐሙስ ስብሰባ አለኝ"]
    store.add_fact("ፀሐይ ጠዋት ትወጣለች")  # the sun rises in the morning; ፀ and ጸ are one sound in both languages
    assert _found(store, "ጸሐይ")[:1] == ["ፀሐይ ጠዋት ትወጣለች"]


def test_amharic_christmas_and_questions_stay_amharic(store):
    """ገና is Christmas in Amharic (Tigrinya "yet"), ወይ an Amharic question particle: neither makes text Tigrinya."""
    assert "ገና" not in _stops("ገና መቼ ነው?")
    store.add_fact("የገና በዓል ጥር 7 ነው")  # Christmas is on January 7
    store.add_fact("ቡና ጥሩ ነው")  # coffee is good: shares only ነው (is), a function word
    assert _found(store, "ገና መቼ ነው?") == ["የገና በዓል ጥር 7 ነው"]


def test_amharic_and_tigrinya_are_told_apart():
    from hermes_plugin_holographic.text import _ethiopic_langs
    assert _ethiopic_langs(["ልጄ", "ትምህርት", "ቤት", "ነው"]) == ("am",)
    assert _ethiopic_langs(["ቡና", "ትፈልጋለህ", "ወይ"]) != ("ti",)  # ወይ is Amharic too: not Tigrinya by it
    assert _ethiopic_langs(["ወዲ", "ኣብ", "ኣስመራ", "እዩ"]) == ("ti",)
    assert _ethiopic_langs(["ኣስመራ"]) == ("am", "ti")


def test_tigrinya_keeps_its_two_h_letters_apart():
    """Tigrinya writes ħ (ሐ) and h (ሀ) as two sounds: only Amharic text merges them."""
    ti = analyze("ሓደ ሰብ ኣብ ገዛ ሓሙስ እዩ").keys  # Tigrinya, by ኣብ and እዩ
    assert "ሓሙስ" in ti and "ሀሙስ" not in ti and "ሃሙስ" not in ti
    am = analyze("ሐሙስ ስብሰባ ነው").keys
    assert "ሀሙስ" in am


def test_ethiopic_numerals_marks_and_word_space():
    from hermes_plugin_holographic.text import _amharic_roots, _amharic_stem
    for numeral in ("፲፩", "፴", "፹"):  # 11, 30, 80: numbers, not syllables
        assert _amharic_stem(numeral) == numeral and not _amharic_roots(numeral)
    assert words("፟") == []  # a lone gemination mark is no word
    assert words("ልጄ፡በአዲስ፡አበባ") == ["ልጄ", "በአዲስ", "አበባ"]  # ፡, the old word space, separates words
    assert {"ውስጥ", "ነው", "ወደ"} <= _stops("ልጄ ወደ ትምህርት ቤት ውስጥ ነው")


# -- 0.33.1: Hausa, Yoruba and Amharic lists, measured on FLORES-200 -------------------------------------------------
# In 0.33.0 Hausa was told in 69 % of FLORES devtest sentences and Yoruba in 84 %: their commonest words ("a", "ko",
# "mai"; "o", "lo", "se") are also Portuguese, Spanish or Italian ones and did not vote, so those languages won.

@pytest.mark.parametrize("text, lang", [
    ("Mai gidan ya ce za a yi taro a kasuwa ko a makaranta", "ha"),  # the landlord said the meeting is at the market
    ("Wannan mota ce mai kyau kuma ana iya gyara ta", "ha"),  # this is a good car and it can be repaired
    ("Bàbá mi lọ sí ọjà láti ra ẹja", "yo"),  # my father went to the market to buy fish
    ("Ó ṣe iṣẹ́ náà dáadáa", "yo"),  # he did the work well
])
def test_hausa_and_yoruba_short_sentences_are_told(text, lang):
    assert lang in analyze(text).langs


def test_yoruba_common_verbs_vote_but_stay_search_words():
    """lọ (go), ṣe (do), jẹ (be, eat), fi (use) tell the language and are not dropped in Yoruba text. (In a tie with
    Spanish a word is dropped when it is a function word of either, as "lo" is in Spanish.)"""
    text = "Bàbá mi lọ sí ọjà láti ra ẹja àti ẹran, ó sì ṣe iṣẹ́ náà dáadáa"
    assert analyze(text).langs == ("yo",)
    for word in ("lo", "se"):
        assert word not in _stops(text)


@pytest.mark.parametrize("query", ["Mai ko?", "Ko Mai?"])
def test_a_name_that_is_a_hausa_word_still_finds_its_fact(store, query):
    """Mai is a name (Vietnamese, Burmese) and a Hausa word (one who has, "mai kyau", good); "ko" is Hausa "or".
    The two together make the message Hausa, and the name must stay a word to search by: «Mai ko?» found the fact
    about Mai before Hausa "mai" and "ko" voted (Codex review of 0.33.1, 2026-11-27)."""
    store.add_fact("Mai lives in Hanoi")
    store.add_fact("The office is in Bangkok")
    assert "mai" in [t.word for t in query_terms(query)]
    assert _recalled(store, query) == ["Mai lives in Hanoi"]


@pytest.mark.parametrize("text, word", [
    ("Musa yana iya karatu sosai kuma yana son makaranta", "iya"),
    ("Mutanen masu kudi suna zuwa kasuwa a cikin gari", "masu"),
])
def test_new_hausa_function_words_are_dropped(text, word):
    assert word in _stops(text)


def test_amharic_free_prepositions_and_modals_are_function_words():
    assert {"በ", "ይችላል", "ደግሞ"} <= _stops("ልጁ በ ትምህርት ቤት ደግሞ መጫወት ይችላል")


@pytest.mark.parametrize("text", [
    "O presidente recebeu bem o anúncio do governo",  # FLORES devtest sentences that tied with Yoruba by "o" ...
    "Orada bulunan insan sayısı o kadar fazlaydı ki herkes geldi",
    "W poniedziałek o godzinie siódmej",
    "Kim ya da ne olduğu belli değil",  # ... or with Hausa by Turkish "ya"
])
def test_old_languages_get_no_new_language_by_o_or_ya(text):
    """Yoruba "o" and Hausa "ya" are Portuguese, Polish and Turkish words too: they do not vote. (The ties these
    sentences have between older languages, "o" and "do" in Portuguese and Polish, were there before.)"""
    from hermes_plugin_holographic.text import _TWO_WORDS
    assert not set(analyze(text).langs) & set(_TWO_WORDS)


def test_a_turkish_either_or_keeps_turkish():
    """Known: "ya ... ya" (either ... or) is two Swahili "ya" too, a tie that keeps Turkish and its stems; making Swahili
    "ya" not vote would lose 9 points of Swahili sentences on FLORES."""
    assert "tr" in analyze("Ya geldi ya gelmedi, bilmiyorum").langs


# -- Afrikaans and Dutch, Zulu prefixes (unreleased) -----------------------------------------------------------------
# Before: Afrikaans was told in 3 % of FLORES-200 sentences. Its commonest words (die, is, het, wat, om, dat) are Dutch
# too and voted for Dutch only, so Afrikaans text was read as Dutch (Dutch stems, Afrikaans grammar words kept as search
# words). Zulu glues its grammar to the word and has few free function words: a third of its sentences were of no
# language.

@pytest.mark.parametrize("text", [
    "Ek het 'n nuwe motor gekoop",  # the article 'n
    "Ek het ’n nuwe motor gekoop",  # typed with a curly apostrophe
    "Ek het 'n motor", "Ek het ('n motor)",  # a short one; the article after a bracket
    pytest.param("Die vergadering word môre in die stadsaal gehou", marks=pytest.mark.xfail(
        strict=True, reason="known: 'word' (is, passive) is Dutch 'ik word' too, so it does not decide; the other "
        "words are shared, and the text stays Dutch")),
    "Ons gaan nie vandag werk toe nie",
    "Dit is nie wat ons wil nie",
])
def test_afrikaans_is_told_apart_from_dutch(text):
    assert analyze(text).langs == ("af",)


def test_afrikaans_grammar_words_drop_and_its_content_words_stay():
    text = "Ek het 'n nuwe motor gekoop"
    assert {"ek", "het", "n"} <= _stops(text)
    assert not {"nuwe", "motor", "gekoop"} & _stops(text)


@pytest.mark.parametrize("text", [
    "Dat is het", "Het is wat het is", "Wat is dat voor een boek?", "Zo'n auto wil ik ook",
    "Iedereen die in Vaticaanstad woont, is rooms-katholiek.",  # FLORES-200: only words Afrikaans shares
])
def test_dutch_made_of_words_afrikaans_shares_stays_dutch(text):
    """Afrikaans decides only with a word or article Dutch does not have."""
    assert analyze(text).langs == ("nl",)


@pytest.mark.parametrize("text", [
    "Ukudla kwakumnandi ngesikhathi somcimbi",  # the food was tasty during the ceremony
    "Izingane ezincane zidlala ngaphandle",  # the small children play outside
])
def test_zulu_is_told_by_its_prefixes(text):
    assert analyze(text).langs == ("zu",)


@pytest.mark.parametrize("text", [
    "Ngorongoro safari booking",  # one word with a Zulu prefix does not decide
    "Ukubwa wa ukurasa huu ni mdogo",  # Swahili has uku- nouns too
])
def test_a_zulu_prefix_does_not_make_other_text_zulu(text):
    assert "zu" not in analyze(text).langs



# Review (Codex, of the change above): words that decide Afrikaans must be no Dutch or English words, the article must
# not decide English or Dutch text alone, Zulu prefixes must not outvote English or Swahili.

@pytest.mark.parametrize("text, lang, word", [
    ("Wat is die deur?", "nl", "deur"),  # Dutch "deur" (door), not Afrikaans "deur" (through)
    ("Is Word open?", "en", "word"),
    ("Het EK begint morgen", "nl", "ek"),  # the European championship, not Afrikaans "ek" (I)
])
def test_a_dutch_or_english_word_does_not_make_text_afrikaans(text, lang, word):
    a = analyze(text)
    assert "af" not in a.langs and lang in a.langs
    assert word not in _stops(text)


@pytest.mark.parametrize("text", [
    "Doe de deur toe", "Dat is wat ik word", "Amerika om te gaan",  # FLORES-200: four words of a Dutch sentence
    "Rock 'n Roll Is Fun", "Fish 'n chips in London",
])
def test_no_afrikaans_from_homographs_or_an_english_n(text):
    assert "af" not in analyze(text).langs


@pytest.mark.parametrize("text", [
    "Ukulele tabs and ezines", "Ukulele lessons in Ngorongoro",  # English
    "Kuna ukungu ukumbini",  # Swahili: there is fog in the hall
])
def test_zulu_prefixes_do_not_outvote_english_or_swahili(text):
    assert "zu" not in analyze(text).langs
