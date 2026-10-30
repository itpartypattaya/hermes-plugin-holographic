"""Text normalisation shared by search, the prefetch gate, entity matching and duplicate checks.

Every word gets comparison keys: its folded form plus its stem in the language of the text, and two words match when
their keys meet, so "Маши" finds "Маша", "kucingnya" finds "kucing" and "لإيرما" finds "إيرما". The language comes from
the script (Cyrillic: Russian; Arabic script: Arabic; Devanagari: Hindi; Bengali script: Bengali) and, for Latin
script, from the function words the text uses ("the", "el", "le", "o", "yang", "der", "il", "bir", "się", "het":
English, Spanish, French, Portuguese, Indonesian, German, Italian, Turkish, Polish, Dutch); Latin text with none of them
gets the stems of all ten. Stems come from Snowball (English, Russian, Spanish, French, Portuguese, Indonesian, German,
Hindi, Italian, Turkish, Polish, Dutch; the last four stemmed as written and folded, see _NATIVE), Larkey's light10 for
Arabic (as in Lucene) next to Snowball's, and a light suffix stripper for Bengali. Folding drops Latin accents (and
writes Turkish ı and Polish ł as i and l), Arabic vowel marks and letter variants and the Indic nukta, and turns
native digits into ASCII. A stem is a search key, not a dictionary lemma. Russian and English
words also get root keys (word_roots: "ночные" / "ночь", "teacher" / "teaches"), matched at _DERIV_WEIGHT in search and
recall, never in duplicate checks. Identifiers (anything with
a digit, or a dotted / pathed token) are never stemmed and never stopwords. Chinese, Japanese, Korean and the scripts
of Southeast Asia are compared by character n-grams instead (see UNSEGMENTED); Traditional Chinese characters compare
as Simplified ones (not Japanese kanji: text with kana keeps them), in text with kana a Japanese spelling variant
compares as the word's representative spelling
(子ども / 子供, 引越し / 引っ越し; see _ja_fold), and Chinese text is also split into words, so a pair of characters
across two words is not that pair (和服 in 我和服务员) and a one-character word is searched alone (see zh_words)."""

from __future__ import annotations

import gzip
import io
import logging
import math
import re
import sys
import threading
import unicodedata
import zlib
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import NamedTuple

from .dictionaries.cjk_fold import SIMPLIFIED, TRADITIONAL

# the bundled language data (tables and their licences: dictionaries/__init__.py, LICENSES/)
_DICTIONARIES = Path(__file__).with_name("dictionaries")
logger = logging.getLogger(__name__)


def _unreadable(path: Path, exc: BaseException) -> None:
    """A dictionary that is missing or damaged: its language is matched without it, which is said (0.17.8; it was
    silent). Each table loads once per process, so this is said once."""
    logger.warning("holographic: dictionary %s could not be read (%s); its words are matched without it", path.name,
                   type(exc).__name__)

_STEMMER_NAMES = (("en", "english"), ("ru", "russian"), ("es", "spanish"), ("fr", "french"), ("pt", "portuguese"),
                  ("id", "indonesian"), ("de", "german"), ("hi", "hindi"), ("ar", "arabic"), ("it", "italian"),
                  ("tr", "turkish"), ("pl", "polish"), ("nl", "dutch"))


def _load_stemmers(snowball) -> dict:
    """Stemmers by language code, each only if this snowballstemmer has its algorithm: Polish is not in 2.x (in 3.1.1,
    the version Hermes pins, it is), and without it Polish words compare as written (folded) while the rest of Polish
    (function words, greetings) still works. "dutch" is Porter's algorithm in 2.x and Snowball's newer one in 3.x,
    which also takes off diminutives ("katje" / "kat")."""
    available = frozenset(snowball.algorithms())
    return {code: snowball.stemmer(name) for code, name in _STEMMER_NAMES if name in available}


try:  # pure Python, BSD; optional: without it words compare exactly (Arabic and Bengali keep their light stemmers)
    import snowballstemmer
    _STEMMERS = _load_stemmers(snowballstemmer)
except ImportError:  # pragma: no cover - exercised only without the dependency
    _STEMMERS = {}

PUNCT = ".,;:!?\"'()[]{}#@<>«»“”„…"
_MIN_STEM = 3  # shorter stems match too much as prefixes ("ма*")
_LOOSE_FROM, _LOOSE_MIN = 7, 5  # stems this long also match on their first len-2 (at least 5) letters
_SHORT_LOOSE = True  # Russian stems of _SHORT_FROM-6 letters also match on their first len-1 letters ("покупа" / "покупк")
_SHORT_FROM = 6  # not five: "больш(ой)" would leave "боль", which begins "больница"
_ROOTS = True  # same-root keys: "ночь" / "ночной", "город" / "городской", "teach" / "teacher"
_DERIV_WEIGHT = 0.6  # a same-root word or a shorter shared beginning counts this much of an exact match

# -- scripts and folding -----------------------------------------------------------------------------------------

def _script(ch: str) -> str:
    o = ord(ch)
    if o < 0x250 or 0x1E00 <= o <= 0x1EFF:
        return "latin"
    if 0x400 <= o <= 0x52F:
        return "cyrillic"
    if 0x600 <= o <= 0x6FF or 0x750 <= o <= 0x77F or 0x8A0 <= o <= 0x8FF or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
        return "arabic"
    if 0x900 <= o <= 0x97F:
        return "devanagari"
    if 0x980 <= o <= 0x9FF:
        return "bengali"
    if 0xA00 <= o <= 0xDFF:
        return "indic"
    return "other"


def word_script(word: str) -> str:
    """Script of the first letter; "common" for a token without letters (numbers, symbols)."""
    return next((_script(c) for c in word if c.isalpha()), "common")


# Scripts the FTS5 unicode61 index cannot serve: it splits Indic words at their vowel signs (marks are not token
# characters), and Arabic attaches conjunctions, prepositions and the article to the front of a word, where a prefix
# query cannot see past them. Words in these scripts are matched against the text of every fact instead.
_SCAN_SCRIPTS = frozenset({"arabic", "devanagari", "bengali", "indic"})
_SCRIPT_LANGS = {"cyrillic": ("ru",), "arabic": ("ar",), "devanagari": ("hi",), "bengali": ("bn",)}
LATIN = ("en", "es", "fr", "pt", "id", "de", "it", "tr", "pl", "nl")
# Latin text of no determined language is stemmed in every language of LATIN but these, which need their own letters
# first: Polish and Turkish stems of English words join unrelated ones ("army" / "arm", "apply" / "apple", "policy" /
# "police" are one stem in Polish), and Italian has no letters of its own
_HYPHEN_COMPOUNDS = frozenset({"es", "fr", "pt", "id", "de"})
_EVIDENCE = {"it": None, "tr": re.compile("[ığş\u0307]"), "pl": re.compile("[ąęłńśźżć]"), "nl": re.compile("ij")}
# Latin languages whose words are also matched word by word against the facts' text (see Word.scan): Indonesian
# prefixes, German compounds, Dutch plurals that change the word ("raam" / "ramen"), Turkish consonant changes ("kitap"
# / "kitabı") and Turkish and Polish letters typed without their marks ("toplanti", "szkola") hide a word from an FTS5
# prefix query.
_SCAN_LATIN = frozenset({"id", "de", "tr", "pl", "nl"})
# Languages stemmed as written as well as folded: Snowball reads the letters folding drops (Turkish "çocuğun" is
# "çocuk", folded "cocugun" stays "cocugu"; Polish "domów" is "dom", "domow" stays), and a word typed without its marks
# still meets one typed with them through the folded stem.
_NATIVE = frozenset({"it", "tr", "pl", "nl"})
# Turkish dotless ı and Polish ł are letters of their own to Unicode (no accent to drop), but are typed as i and l on
# keyboards without them, and Turkish upper-case I lower-cases to i in Python: they compare as i and l. FTS5's
# unicode61 keeps them, so a word that has them is also searched as written (see _native_keys).
_DOTLESS = str.maketrans({"ı": "i", "ł": "l"})
# Dutch writes compounds as one word, as German does ("kinderarts", "reservesleutels"). True: a word matches the end of
# a compound in Dutch text too, as in German. Off: measured on 10 questions that ask by the end of a compound and 7
# that use that end in its own sense, it found 2 more right facts but recalled 4 more wrong ones ("huis" in
# "ziekenhuis", "weer" in "brandweer", "kast" in "koelkast").
_NL_COMPOUNDS = False

_FOLD = str.maketrans({
    "œ": "oe", "æ": "ae",  # Latin ligatures ("cœur" = "coeur")
    **{c: "ا" for c in "أإآٱ"}, "ى": "ي", "ة": "ه", "ـ": None,  # Arabic letter variants, tatweel
    **{chr(c): None for c in range(0x64B, 0x660)}, "ٰ": None,  # Arabic vowel marks
    "़": None, "ँ": "ं",  # Devanagari nukta; candrabindu written as anusvara
    "়": None, "ঁ": None,  # Bengali nukta and candrabindu
    **{chr(base + d): str(d) for base in (0x660, 0x6F0, 0x966, 0x9E6) for d in range(10)},  # native digits
})
# Traditional Chinese characters compare as Simplified ones (學校 = 学校, 醫院 = 医院): a Taiwan user's fact and a mainland
# user's question, or the other way round, share their characters. OpenCC's table (cjk_fold.py); stored text unchanged.
# Not in Japanese text (it has kana): its kanji share code points with Traditional characters but are not them (手紙,
# a letter, is no 手纸; 書 no 书), so a Japanese question no longer meets a Chinese fact through a kanji it rewrote. A
# question in kanji only cannot be told from Chinese and is folded; a Japanese fact also keeps that form (see analyze).
_HAN_FOLD = True
_FOLD_HAN = {**_FOLD, **str.maketrans(TRADITIONAL, SIMPLIFIED)}
# Japanese spelling variants (表記揺れ) compare as the word's representative spelling, in text that has kana (so Chinese
# text is never touched): 子ども = 子供, 引越し = 引越 = 引っ越し, スマホ = スマフォ. SudachiDict's synonym source, spellings of
# one word only (ja_variants.tsv.gz, tools/build_ja_variants.py); stored text unchanged. See _ja_fold.
_JA_VARIANTS = True
_ja_ready = False  # set at the end of this module: the word lists below are folded before the table may load
_KANA = re.compile(r"[ぁ-ゖァ-ヺ]")  # hiragana and katakana letters (not ー or ・)


def _fold_text(text: str, ja: bool | None = None, dotless: bool = True, han: bool | None = None) -> str:
    """The comparison form of a text (see fold). *ja*: apply the Japanese spelling variants; None decides by the text
    itself (it has kana). analyze decides once for a whole text and passes it on to each of its words. *dotless*=False
    keeps ı and ł, as FTS5's unicode61 tokenizer does. *han*: write Traditional Chinese characters as Simplified;
    None does so unless the text is Japanese (it has kana, or *ja* says so): Japanese kanji share their code points
    with Traditional characters, and 手紙 (a letter) is no 手纸 (toilet paper), 書 no 书."""
    if text.isascii():
        return text
    out, latin = [], False
    for c in unicodedata.normalize("NFD", text):
        if "̀" <= c <= "ͯ":
            if latin:
                continue  # an accent on a Latin letter; the breve of Cyrillic й stays
        else:
            latin = c.isascii() or _script(c) == "latin"
        out.append(c)
    kana = _KANA.search(text) is not None
    if han is None:
        han = _HAN_FOLD and not (kana or ja)
    folded = unicodedata.normalize("NFC", "".join(out)).translate(_FOLD_HAN if han else _FOLD)
    if dotless:
        folded = folded.translate(_DOTLESS)
    if _JA_VARIANTS and _ja_ready and (kana if ja is None else ja):
        return _ja_fold(folded)
    return folded


@lru_cache(maxsize=65536)
def fold(word: str) -> str:
    """Comparison form of a lower-cased word: Latin accents dropped ("está" = "esta"), Turkish ı and Polish ł as i
    and l, Arabic vowel marks, tatweel and letter variants normalised (أ إ آ -> ا, ى -> ي, ة -> ه), the Indic nukta
    dropped, native digits as ASCII, Traditional Chinese characters as Simplified (not Japanese kanji: a word with
    kana keeps them), a Japanese spelling variant (in a word with kana) as the representative one."""
    return _fold_text(word)


@lru_cache(maxsize=65536)
def _fold_ja(word: str) -> str:
    """fold() for a word of a text that has kana: the word itself may have none ("申込" in "申込 締切 いつ？")."""
    return _fold_text(word, True)


def _folded(words: str) -> frozenset:
    return frozenset(fold(w) for w in words.split())


# -- function words ------------------------------------------------------------------------------------------------

STOP = {
"en": _folded("""
    a about above after again all am an and any are as at be because been before being between both but by can could
    did do does doing don down during each few for from further had has have having he her here hers herself him himself
    his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our
    ours ourselves out over own same she should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves ok okay yes yeah thanks thank please hi hello hey"""),
"ru": _folded("""
    а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже для до его ее её если
    есть еще ещё же за здесь и из или им их к как ко когда кто ли либо мне может мы на над надо наш не него нее неё нет
    ни них но ну о об однако он она они оно от очень по под при про с со так также такой там те тебе тем то того тоже
    той только том ты у уже хотя чего чей чем что чтобы чье чья эта эти это этот я ага ок окей спасибо привет пожалуйста
    да нет ладно хорошо понял поняла давай конечно просто вообще кстати понятно ясно угу кого кому чему чем ведь вон
    чём всё моё твоё своё нём"""),
"es": _folded("""
    a al algo alguna algunas alguno algunos algún ante antes como cómo con contra cual cuál cuales cuando cuándo cuanto
    cuánto cuanta cuánta cuantos cuántos cuantas cuántas de del desde donde dónde adónde durante e el él ella ellas ellos
    en entre era eran eres es esa esas ese eso esos esta está estaba estaban estamos están estar estas estás este esto
    estos estoy fue fueron fui ha había han has hasta hay he hemos la las le les lo los más me mi mí mis mucho muchos
    muy nada ni no nos nosotros nuestra nuestro o os otra otras otro otros para pero poco por porque que qué quien
    quién quienes quiénes se sea ser si sí sin sobre sois somos son soy su sus también tanto te tenemos tener tengo
    tiene tienen ti tu tú tus un una unas uno unos usted ustedes vosotros y ya yo cada hacer hace hacen hago puedo
    puede pueden puedes podemos debo debe hola gracias vale ok bueno claro dale vamos"""),
"fr": _folded("""
    à a ai as au aux avec avez avoir avons c ça ce ceci cela celle celles celui ces cet cette chez comme comment d dans
    de des donc dont du elle elles en encore est et étaient était été être eu eux fait faire fais font il ils j je l la
    le les leur leurs lui m ma mais me même mes moi mon n ne ni nos notre nous on ont ou où par pas peu peut peux
    pouvez pour pourquoi qu quand que quel quelle quelles quels qui quoi s sa sans se ses si son sont sous suis sur t
    ta te tes toi ton tous tout toute toutes très tu un une vers vos votre vous y combien quelque lequel laquelle oui
    non merci salut bonjour accord ok va vais vas allons allez vont aller faut"""),
"pt": _folded("""
    a à ao aos as às o os um uma uns umas de do da dos das em no na nos nas num numa por pelo pela pelos pelas para pra
    com sem e ou mas que se não mais como quando onde quem qual quais quanto quanta quantos quantas porque porquê eu tu
    você vocês ele ela eles elas nós me te lhe lhes meu minha meus minhas teu tua seu sua seus suas nosso nossa este
    esta estes estas esse essa esses essas isto isso aquilo aquele aquela já também só muito é são era eram foi ser sou
    somos está estão estou estava estar tem têm tenho tinha ter vai vou vamos há pode posso podem poder devo deve devem
    algum alguma alguns algumas nenhum nenhuma cada outro outra ok obrigado obrigada oi olá sim valeu"""),
"id": _folded("""
    yang dan di ke dari ini itu dengan untuk tidak tak ada akan juga atau pada adalah ialah saya aku kamu engkau dia ia
    mereka kita kami anda sudah telah belum bisa dapat harus sedang masih lebih sangat apa siapa kapan mana bagaimana
    mengapa kenapa berapa apakah yaitu yakni oleh karena jika kalau agar supaya tetapi tapi namun hanya saja pun lagi
    para sebuah seorang setiap tiap semua banyak sedikit nya ya oke ok makasih halo hai sip seperti
    sebaiknya punya mau ingin boleh perlu nih dong kok sih deh"""),
"de": _folded("""
    der die das den dem des ein eine einer eines einem einen und oder aber nicht kein keine ist sind war waren bin
    bist sein hat haben habe hast hatte wird werden wurde ich du er sie es wir ihr mich mir dich dir ihn ihm uns euch
    sich mein meine dein deine seine ihre unser zu im in am an auf aus bei mit nach von vor für über unter um durch
    gegen ohne bis als wie wo was wer wen wem wann warum welche welcher welches viel viele noch schon nur auch so ja
    nein doch mal da dort hier dann denn dass weil wenn ob kann können soll sollen muss müssen will wollen darf gibt
    etwas ok okay danke bitte hallo"""),
"ar": _folded("""
    في من إلى الى على عن عمن مع هذا هذه ذلك تلك هؤلاء هو هي هم هن هما أنا نحن أنت أنتم كان كانت يكون تكون كانوا ما
    ماذا متى أين كيف لماذا لم لن لا ليس ليست قد كل بعض غير بين عند حتى إذا إن أن أو ثم بل لكن التي الذي الذين يتم تم
    لدى لديه لديها لدي لديك له لها لهم لك لي به بها فيه فيها منه منها عليه عليها و ف ب ل ك يا أيضا فقط جدا شكرا
    مرحبا حسنا تمام نعم أهلا هل أي كم يمكن يجب يفعل تفعل يقوم تقوم"""),
"hi": _folded("""
    का की के को में से पर है हैं था थी थे हो होता होती होते होगा होगी होंगे और या भी तो ही न नहीं यह वह ये वे वो इस उस
    इन उन इसे उसे मैं हम तुम आप मेरा मेरी मेरे हमारा हमारी हमारे तुम्हारा तेरा आपका आपकी आपके उसका उसकी उसके उनका
    उनकी उनके इसका इसकी इसके कौन क्या कब कहाँ कहां कैसे क्यों कितना कितनी कितने किस किसे किसी कोई कुछ एक लिए साथ बाद
    पहले जो जब तक अब यहाँ वहाँ सकता सकती सकते रहा रही रहे गया गई गए कर करता करती करते करना करें किया हाँ ठीक
    धन्यवाद नमस्ते शुक्रिया अच्छा ओके तरह वाला वाली वाले बहुत"""),
"bn": _folded("""
    এবং ও আর বা কিন্তু যে যা যিনি যারা এই ওই সেই এটা ওটা সেটা এটি ওটি সেটি সে তারা আমি আমরা তুমি তোমরা আপনি আপনারা
    তিনি তাঁরা আমার আমাদের তোমার তোমাদের আপনার আপনাদের তার তাদের তাঁর তাঁদের কে কী কি কেন কোন কোথায় কখন কবে কীভাবে
    কিভাবে কেমন কত কতটা কাকে কার একটি একটা এক জন্য সাথে সঙ্গে থেকে দিয়ে নিয়ে হয় হয়ে হলে হবে হচ্ছে ছিল ছিলেন আছে
    আছেন নেই না নয় হ্যাঁ ঠিক আচ্ছা ধন্যবাদ হ্যালো নমস্কার প্রতি করে করেন করা করছে যায় খুব এর এ তে র"""),
# Italian, Turkish, Polish, Dutch (unreleased). Left out: everyday English words ("go", "one", "ten", "pod", Turkish
# "size") and the name Mia (Italian "mia", my). Words that are also content words, names or abbreviations elsewhere are
# listed but do not tell the language (_WEAK).
"it": _folded("""
    il lo la i gli le un uno una di a da in con su per tra fra del dello della dei degli delle al allo alla ai agli
    alle dal dallo dalla dai dagli dalle nel nello nella nei negli nelle sul sullo sulla sui sugli sulle col coi e ed o
    ma anche che chi cui non né se come dove quando quanto quanta quanti quante quale quali perché cosa io tu lui lei
    noi voi loro mi ti ci vi si ne me te ce ve mio miei mie tuo tua tuoi tue suo sua suoi sue nostro nostra nostri
    nostre vostro vostra vostri vostre questo questa questi queste quello quella quelli quelle quel è sono sei siamo
    siete era erano ero essere stato stata stati ha ho hai abbiamo avete hanno aveva avevo avere può posso puoi
    possiamo possono devo deve dobbiamo devono devi vuole voglio vuoi più molto poco tutto tutti tutta tutte ogni già
    ancora sempre così poi allora qui qua lì là ok okay sì no grazie ciao prego bene va"""),
"tr": _folded("""
    ve veya ya ile ama fakat ancak çünkü için gibi kadar daha en çok az bir bu şu o bunlar şunlar onlar ben sen biz siz
    onu bunu şunu ona buna şuna bana sana bize beni seni bizi sizi benim senin onun bizim sizin onların bunun şunun de
    da ki mi mı mu mü ne neden niye nasıl nerede nereye nereden zaman kim kimi kime kimin hangi hangisi kaç kaçta var
    yok değil evet hayır tamam olan olarak oldu olur olacak ise her hiç şey sonra önce şimdi zaten artık hep bile
    sadece yani teşekkürler teşekkür sağol merhaba selam lütfen tabii peki"""),
"pl": _folded("""
    i a o u w we z ze na do od po za przy przez dla bez nad przed między się nie tak jak co kto gdzie kiedy dlaczego
    czemu ile jaki jaka jakie jakiego jakiej jakim jakimi który która które którego której którym czy to ta te tego tej
    tym tych jest są był była było byli będzie być mam ma masz mamy macie mają miał miała może można musi trzeba już
    jeszcze też także tylko bardzo ja ty on ona ono my wy oni mnie mi ciebie cię ci tobie mu jego jej ich im nas nam
    was wam mój moja moje mojego twój twoja twoje swój swoja swoje nasz nasza nasze wasz wasza ale lub albo że żeby
    więc bo oraz jakiś coś ktoś ok okej dzięki dziękuję cześć proszę no"""),
"nl": _folded("""
    de het een en of maar niet geen wel ook nog al dan als dat die dit deze wat wie waar wanneer waarom hoe welke welk
    hoeveel ik jij je u hij zij ze wij we jullie hen hun mij me mijn jouw jou zijn haar ons onze is ben bent was waren
    wordt worden werd heeft heb hebt hebben had hadden kan kun kunt kunnen moet moeten moest wil wilt willen zal zullen
    zou in op aan bij met van voor naar uit over onder tot om door te tegen zonder na er hier daar nu zo heel veel meer
    ja nee oké ok bedankt dank hallo hoi doei graag z'n zo'n"""),
}
STOPWORDS = frozenset().union(*STOP.values())
# Function words of a new language that are also content words, names or abbreviations of another ("come" and "dove"
# in English, Spanish "come"; Turkish "kim", "ben"; Polish "bo", single "u", "w", "z"; Dutch "door", "met", "van", the
# Thai "wat"): they do not vote for their language, and are dropped only in text that language alone is best
# supported in, so "Ana come carne" stays Spanish and "Anton met Irma at the door" English.
_WEAK = {
    "it": _folded("per come dove ci vi io lei ed chi fra sei col"),
    "tr": _folded("kim ben ki var bile sen beni yani daha az sana ama"),
    "pl": _folded("u w z bo ci ale maja nam mu coś ile po od"),
    "nl": _folded("door met van wat heel ben dank"),
}
# The other way round: words that tell the language but stay words. The Spanish forms of "ir" ("va", "van", "vas",
# "voy") are function words of French and Italian here, and without them "Irma va a la escuela" was French or Italian
# (no Spanish stems, no verb forms); as function words they would be dropped from a question, and "¿Cómo va Javier?"
# meets "fue" through them (see word_lemmas).
_VOTE_ONLY = {"es": _folded("va van vas voy")}
_VOTERS = {lang: (words | _VOTE_ONLY.get(lang, frozenset())) - _WEAK.get(lang, frozenset())
           for lang, words in STOP.items()}

# Question words that say nothing about the subject of a QUESTION ("какая сегодня погода": "какая" met "как" in every
# fact that says "считать как …"), but are part of a statement ("Иван знает, почему" is not "Иван знает, куда"): left
# out of a query's terms only, never out of a fact's words, duplicate or conflict checks.
QUERY_ONLY_STOP = _folded("""какой какая какое какие какого какому каким каком какую каких какими каков который которая
    которое которые которого которой которому которым котором которую которых сколько куда откуда почему зачем""")

NEGATORS = _folded("""nicht kein keine keinen keinem keiner nie niemals nichts
    not no never nobody nothing cannot don't doesn't didn't isn't aren't wasn't weren't won't can't
    couldn't shouldn't haven't hasn't dont doesnt didnt isnt cant wont не нет никогда ни nunca jamás tampoco nadie
    ningún ninguna ninguno ne pas jamais aucun aucune non não nem nenhum nenhuma tidak tak bukan belum jangan لا لم لن
    ليس ليست नहीं न मत না নয় নেই নি nessuno nessuna niente nulla değil yok hiç asla hayır nigdy nic żaden żadna żadne
    nikt niet geen nooit niets niemand""")  # not Italian "mai" (never): French "mai" (May)


# Function words a claim still needs ("with sugar" / "without sugar"), and conjunctions after which a negation belongs
# to another clause ("likes cats but not dogs").
CLAIM_WORDS = _folded("with without avec sans con sin com sem dengan tanpa mit ohne مع بدون साथ बिना সাথে ছাড়া с без "
                      "senza ile olmadan z ze bez met zonder")
CONTRASTIVE = _folded("but however pero sino mais mas porém tapi tetapi namun aber sondern jedoch لكن बल्कि लेकिन किंतु "
                      "কিন্তু но però tuttavia invece bensì ama fakat ancak lakin ale lecz jednak maar echter")


def _greeting_key(text: str) -> str:
    folded = _fold_text(unicodedata.normalize("NFC", text or "").lower())
    return "".join(c for c in folded if c.isalnum() or unicodedata.category(c)[0] == "M")


# Whole-message greetings and courtesies: they name a time of day ("good morning") but ask nothing, so they must not
# recall facts that mention mornings. Compared without case, accents, punctuation and spaces.
GREETINGS = frozenset(_greeting_key(g) for g in """
    goodmorning|goodafternoon|goodevening|goodnight|morning|gm|gn|hello|hi|hey|hiall|hellothere|hithere|howareyou|
    thankyou|thanksalot|thankssomuch|
    доброеутро|добрыйдень|добрыйвечер|спокойнойночи|добройночи|утро|привет|приветик|всемпривет|здравствуй|здравствуйте|
    какдела|спасибо|спасибобольшое|благодарю|
    早上好|早安|上午好|中午好|下午好|晚上好|晚安|你好|您好|大家好|嗨|谢谢|多谢|非常感谢|
    hola|buenos días|buen día|buenas|buenas tardes|buenas noches|hola a todos|qué tal|gracias|muchas gracias|mil gracias|
    bonjour|bonsoir|salut|coucou|bonne nuit|bonne journée|bonne soirée|bonjour à tous|salut à tous|ça va|merci|
    merci beaucoup|
    olá|oi|bom dia|boa tarde|boa noite|olá a todos|oi pessoal|tudo bem|obrigado|obrigada|muito obrigado|muito obrigada|
    halo|hai|selamat pagi|selamat siang|selamat sore|selamat malam|halo semua|apa kabar|terima kasih|makasih|
    terima kasih banyak|
    مرحبا|أهلا|أهلا وسهلا|السلام عليكم|صباح الخير|مساء الخير|تصبح على خير|كيف حالك|شكرا|شكرا جزيلا|
    नमस्ते|नमस्कार|हैलो|हेलो|सुप्रभात|शुभ प्रभात|शुभ रात्रि|आप कैसे हैं|धन्यवाद|बहुत धन्यवाद|शुक्रिया|
    নমস্কার|হ্যালো|শুভ সকাল|সুপ্রভাত|শুভ রাত্রি|আসসালামু আলাইকুম|কেমন আছেন|কেমন আছো|ধন্যবাদ|অনেক ধন্যবাদ|
    guten morgen|guten tag|guten abend|gute nacht|hallo zusammen|servus|moin|danke schön|dankeschön|vielen dank|
    danke sehr|wie geht's|
    おはよう|おはようございます|こんにちは|こんばんは|おやすみ|おやすみなさい|ありがとう|ありがとうございます|
    どうもありがとう|どうも|よろしく|よろしくお願いします|はじめまして|
    안녕|안녕하세요|안녕하십니까|좋은 아침|좋은 아침이에요|고마워|고마워요|고맙습니다|감사합니다|감사해요|잘 자|잘 자요|
    สวัสดี|สวัสดีครับ|สวัสดีค่ะ|อรุณสวัสดิ์|อรุณสวัสดิ์ครับ|อรุณสวัสดิ์ค่ะ|ขอบคุณ|ขอบคุณครับ|ขอบคุณค่ะ|ขอบคุณมาก|
    ขอบคุณมากครับ|ขอบคุณมากค่ะ|ราตรีสวัสดิ์|
    buongiorno|buon giorno|buonasera|buona sera|buonanotte|buona notte|buon pomeriggio|ciao|ciao a tutti|salve|grazie|
    grazie mille|mille grazie|molte grazie|grazie tante|come stai|come va|tutto bene|arrivederci|a dopo|a domani|
    merhaba|merhabalar|selam|selamlar|herkese merhaba|günaydın|tünaydın|iyi günler|iyi akşamlar|iyi geceler|nasılsın|
    nasılsınız|naber|teşekkürler|teşekkür ederim|çok teşekkürler|çok teşekkür ederim|sağ ol|sağ olun|eyvallah|
    görüşürüz|hoşça kal|
    cześć|dzień dobry|dobry wieczór|dobranoc|witam|witaj|witajcie|siema|hej|cześć wszystkim|jak się masz|co słychać|
    dziękuję|dzięki|dziękuję bardzo|bardzo dziękuję|wielkie dzięki|do zobaczenia|na razie|
    hallo|hoi|goedemorgen|goedemiddag|goedenavond|goedenacht|hallo allemaal|hoi allemaal|hoe gaat het|hoe is het|
    dank je|dank je wel|dankjewel|dank u|dank u wel|bedankt|heel erg bedankt|tot ziens|tot morgen|doei|
    fijne dag""".split("|"))


# Pieces a courtesy message is made of in English and Russian (and, unreleased, Italian, Turkish, Polish and Dutch),
# beyond the whole greetings above: thanks, sign-offs,
# reactions, "how are you"-questions and the words that pad them ("today", "there", "so much"). A message whose every
# clause is a run of these pieces ("Hi, good morning!", "Thanks a lot, that's all for now", "Good morning! How are you
# today?", "спасибо, на этом всё") asks nothing. A clause with any other word ("how is Tim doing at school") is a
# question. Compared like GREETINGS (no case, accents, spaces or punctuation inside a clause).
COURTESY = frozenset(_greeting_key(g) for g in """
    good|morning|afternoon|evening|night|day|hi|hello|hey|heya|hiya|yo|sup|whatsup|howdy|there|all|everyone|guys|
    team|folks|mate|buddy|dear|again|today|tonight|so|much|a lot|very|too|really|
    thanks|thank you|thx|ty|cheers|appreciate it|much appreciated|you're welcome|no problem|np|no worries|
    that's all|that's it|for now|all good|all set|ok|okay|k|kk|great|cool|nice|awesome|perfect|excellent|super|
    sounds good|got it|understood|noted|sure|yes|yeah|yep|no|nope|lol|haha|hahaha|hehe|wow|oh|ah|hm|hmm|
    see you|see ya|see you later|see you tomorrow|bye|goodbye|bye bye|take care|talk later|later|
    have a nice day|have a good day|have a great day|have a good one|have a good evening|have a good night|
    have a good weekend|have a nice weekend|happy friday|happy monday|happy weekend|happy holidays|
    how are you|how are you doing|how are things|how's it going|how is it going|how have you been|what's new|
    how's your day|how's your day going|how is your day going|how was your day|how's your week|how's your week going|
    how was your weekend|how's everything|you good|well done|good job|nice one|
    sleep well|sweet dreams|fine|i'm fine|i'm good|and you|you too|same to you|have a great weekend|have a great week|
    have a nice week|have a good week|happy weekend|happy birthday|congrats|congratulations|good luck|
    добрый|доброе|доброго|утро|утра|день|дня|вечер|вечера|ночи|привет|приветик|приветствую|здравствуй|здравствуйте|
    хай|всем|всех|ребята|друзья|еще раз|ещё раз|снова|сегодня|
    спасибо|спс|благодарю|большое|огромное|пожалуйста|не за что|отлично|супер|класс|круто|здорово|прекрасно|понятно|
    ясно|хорошо|ок|окей|ладно|договорились|принято|понял|поняла|ага|угу|да|нет|ха|хаха|ахах|ахаха|лол|ого|ух ты|
    на этом всё|на этом все|это всё|это все|пока всё|пока все|всё|все|пока|до встречи|до завтра|увидимся|до связи|
    хорошего дня|хорошего вечера|хороших выходных|спокойной ночи|удачи|всего доброго|
    как дела|как ты|как вы|как у тебя дела|как у вас дела|как поживаешь|как поживаете|как прошёл день|как прошел день|
    как твой день|как настроение|что нового|как сам|как сама|как твоя неделя|как прошла неделя|как выходные|
    как прошли выходные|с пятницей|с праздником|с выходными|с днём рождения|с днем рождения|поздравляю|
    сладких снов|спи спокойно|и тебе|и вам|взаимно|у меня всё хорошо|у меня все хорошо|нормально|
    buongiorno|buonasera|buonanotte|buon pomeriggio|ciao|salve|ehi|grazie|grazie mille|mille grazie|grazie tante|
    molte grazie|prego|di niente|figurati|perfetto|ottimo|benissimo|bene|va bene|d'accordo|capito|certo|sì|tutto bene|
    tutto ok|come stai|come va|come state|sto bene|anche tu|anche a te|altrettanto|arrivederci|a dopo|a domani|a presto|
    ci vediamo|alla prossima|buona giornata|buona serata|buon weekend|buon fine settimana|sogni d'oro|auguri|
    complimenti|bravo|brava|è tutto|per ora è tutto|per oggi è tutto|ahah|
    a tutti|tutti|ragazzi|di nuovo|ancora|oggi|stamattina|molto|tante|davvero|per ora|per oggi|
    merhaba|merhabalar|selam|selamlar|günaydın|tünaydın|iyi günler|iyi akşamlar|iyi geceler|iyi sabahlar|teşekkürler|
    teşekkür ederim|teşekkür ederiz|çok teşekkürler|çok teşekkür ederim|sağ ol|sağol|sağ olun|eyvallah|rica ederim|
    bir şey değil|tamam|tamamdır|peki|tabii|tabi|olur|anladım|anlaşıldı|harika|süper|mükemmel|güzel|çok güzel|evet|
    hayır|nasılsın|nasılsınız|naber|ne haber|nasıl gidiyor|iyiyim|ben de iyiyim|sen nasılsın|siz nasılsınız|sen de|
    siz de|sana da|size de|görüşürüz|görüşmek üzere|hoşça kal|hoşça kalın|kendine iyi bak|iyi hafta sonları|
    iyi şanslar|kolay gelsin|tebrikler|bu kadar|şimdilik bu kadar|hepsi bu|vay|
    herkese|herkes|arkadaşlar|millet|bugün|tekrar|çok|şimdilik|canım|canlarım|canlar|tatlım|dostlar|akşam|sabah|
    yarın|
    cześć|dzień dobry|dobry wieczór|dobranoc|dobrej nocy|witam|witaj|witajcie|siema|siemka|hej|hejka|dziękuję|dzięki|
    dziękuję bardzo|bardzo dziękuję|wielkie dzięki|stokrotne dzięki|dziękuję ci|dziękuję wam|nie ma za co|proszę|
    świetnie|ekstra|dobrze|okej|jasne|rozumiem|zrozumiałem|zrozumiałam|spoko|tak|nie|jak się masz|jak się macie|
    jak się czujesz|co słychać|co u ciebie|co u was|jak leci|wszystko dobrze|wszystko w porządku|w porządku|
    u mnie dobrze|u mnie wszystko dobrze|nawzajem|wzajemnie|ty też|do zobaczenia|do jutra|na razie|pa|papa|
    trzymaj się|miłego dnia|miłego wieczoru|udanego weekendu|miłego weekendu|kolorowych snów|śpij dobrze|gratulacje|
    powodzenia|wszystkiego najlepszego|to wszystko|na dziś to wszystko|na razie to wszystko|
    wszystkim|wszystkich|kochani|dzisiaj|dziś|jeszcze raz|bardzo|na dziś|ponownie|
    hallo|hoi|hé|hee|goedemorgen|goedemiddag|goedenavond|goedenacht|dag|bedankt|dank je|dank je wel|dankjewel|dank u|
    dank u wel|heel erg bedankt|alvast bedankt|graag gedaan|geen probleem|geen dank|top|prima|mooi|fijn|oké|helder|
    duidelijk|begrepen|komt goed|ja|nee|goed|hoe gaat het|hoe gaat het met je|hoe gaat het met jou|hoe is het|
    hoe is het met je|alles goed|alles prima|met mij gaat het goed|tot ziens|tot morgen|tot straks|tot later|tot zo|
    doei|fijne dag|fijne avond|fijn weekend|prettig weekend|slaap lekker|welterusten|gefeliciteerd|succes|jij ook|
    u ook|insgelijks|dat is alles|dat was het|voor nu|
    allemaal|iedereen|jongens|vandaag|nogmaals|heel|erg|""".split("|"))
_CLAUSE = re.compile(r"[.,;:!?…\n\r]+")
# Pieces that only pad a courtesy ("good", "day", "today", "fine", "сегодня", "всё"): a clause of these alone is a
# question or an answer ("fine?" about a fine, "сегодня?" about a lesson), not small talk.
_COURTESY_PADDING = frozenset(_greeting_key(g) for g in """
    good|afternoon|evening|night|day|there|all|everyone|guys|team|folks|mate|buddy|dear|again|today|tonight|so|much|
    a lot|very|too|really|fine|later|
    добрый|доброе|доброго|утра|день|дня|вечер|вечера|ночи|всем|всех|ребята|друзья|еще раз|ещё раз|снова|сегодня|
    большое|огромное|всё|все|пока|нормально|
    a tutti|tutti|ragazzi|di nuovo|ancora|oggi|stamattina|molto|tante|davvero|per ora|per oggi|
    herkese|herkes|arkadaşlar|millet|bugün|tekrar|çok|şimdilik|canım|canlarım|canlar|tatlım|dostlar|akşam|sabah|
    yarın|
    wszystkim|wszystkich|kochani|dzisiaj|dziś|jeszcze raz|bardzo|na dziś|ponownie|
    allemaal|iedereen|jongens|vandaag|nogmaals|heel|erg""".split("|"))


def _clause_cuts(clause: str) -> tuple[str, frozenset]:
    """The clause folded as _greeting_key folds it, and the positions a piece may start or end at: word boundaries,
    and any position next to a character of a script written without spaces (Chinese, Japanese, Thai)."""
    folded = _fold_text(unicodedata.normalize("NFC", clause).lower())
    key, cuts, gap = [], {0}, True
    for c in folded:
        if c.isalnum() or unicodedata.category(c)[0] == "M":
            if key and (gap or UNSEGMENTED.match(c) or UNSEGMENTED.match(key[-1])):
                cuts.add(len(key))
            key.append(c)
            gap = False
        else:
            gap = True
    cuts.add(len(key))
    return "".join(key), frozenset(cuts)


@lru_cache(maxsize=4096)
def _courtesy_clause(clause: str) -> int:
    """0: the clause is not made of greeting and courtesy pieces alone (pieces start and end at word boundaries, so
    "history" is not "hi" + "story"); 1: it is, of padding only; 2: it is, with a greeting, thanks, sign-off, reaction
    or "how are you"."""
    key, cuts = _clause_cuts(clause)
    pieces = COURTESY | GREETINGS
    longest = max(len(p) for p in pieces)
    best = [-1] * (len(key) + 1)  # best[i]: -1 unreachable, 1 padding only, 2 with a real courtesy piece
    best[0] = 1
    for end in sorted(cuts):
        for start in sorted(c for c in cuts if end - longest <= c < end):
            piece = key[start:end]
            if best[start] > 0 and piece in pieces:
                best[end] = max(best[end], best[start], 1 if piece in _COURTESY_PADDING else 2)
    return max(best[len(key)], 0) if key else 0


def is_greeting(text: str) -> bool:
    """A whole-message greeting, or a message every clause of which is made only of greeting and courtesy pieces, with
    at least one that is more than padding ("Hi, good morning!", "Thanks, that's all for now", "спасибо, на этом всё";
    not "fine?" or "сегодня?")."""
    if _greeting_key(text) in GREETINGS:
        return True
    clauses = [c for c in _CLAUSE.split(text or "") if _greeting_key(c)]
    kinds = [_courtesy_clause(c) for c in clauses]
    return bool(kinds) and all(kinds) and max(kinds) == 2


def is_identifier(token: str) -> bool:
    """'pve-01', 'v0.21.6', 'gpt-4o', 'host:22', '/etc/hosts', 'hr-vps': kept exact, never stemmed or dropped, and a
    message naming one is about that one. A hyphenated word with a letter outside ASCII is a word ("кто-то",
    "peut-être"), and so is any letters-only hyphenated word in Spanish, French, Portuguese or Indonesian text (see
    analyze); French inversion, Portuguese enclisis and reduplication are taken apart before this is asked."""
    return any(c.isdigit() for c in token) or (len(token) > 2 and (
        any(c in token for c in "._:/") or ("-" in token and token.isascii())))


# -- stemming ------------------------------------------------------------------------------------------------------

# Arabic light10 (Larkey, Ballesteros and Connell; the stemmer Lucene ships): one prefix, then each suffix in turn,
# leaving at least two letters. Input is folded, so ة is already ه.
_AR_PREFIXES = ("ال", "وال", "بال", "كال", "فال", "لل", "و")
_AR_SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "يه", "ه", "ي")


def _arabic_light(word: str) -> str:
    for prefix in _AR_PREFIXES:
        if word.startswith(prefix) and len(word) >= (4 if prefix == "و" else len(prefix) + 2):
            word = word[len(prefix):]
            break
    for suffix in _AR_SUFFIXES:
        if word.endswith(suffix) and len(word) >= len(suffix) + 2:
            word = word[:-len(suffix)]
    return word


# Bengali: Snowball has no stemmer; one inflectional suffix (case, plural, classifier, common verb endings) is
# stripped, longest first, leaving at least three characters (so "মাটি", soil, does not become "মা", mother). A bare
# word that loses a letter it does not inflect with ("শুক্রবার" -> "শুক্রবা") still meets its inflected forms through
# its unstripped form.
_BN_SUFFIXES = sorted({fold(unicodedata.normalize("NFC", s)) for s in """
    গুলোকে গুলোতে গুলোর গুলো গুলিকে গুলিতে গুলির গুলি দেরকে দের েরা রা টাকে টিকে টাতে টিতে টার টির টা টি খানা খানি
    য়েরা য়ের য়ে য় ের কে তে েতে র ে এর এ চ্ছিলাম চ্ছিলেন চ্ছিল চ্ছেন চ্ছিস চ্ছি চ্ছে ছিলাম ছিলেন ছিল ছেন ছিস ছি ছে
    েছিলাম েছিলেন েছেন েছি েছে লাম লেন বেন বে বো তাম তেন ওয়ার ওয়া ানো""".split()}, key=len, reverse=True)


def _bengali_stem(word: str) -> str:
    for suffix in _BN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


_DEFAULT_LANG = {"latin": "en", "cyrillic": "ru", "arabic": "ar", "devanagari": "hi", "bengali": "bn"}


@lru_cache(maxsize=65536)
def stem(word: str, lang: str | None = None) -> str:
    """Stem of a folded, lower-cased word in *lang* (by its script when None); the word itself for identifiers, scripts
    without a stemmer, or without snowballstemmer."""
    if is_identifier(word):
        return word
    lang = lang or _DEFAULT_LANG.get(word_script(word))
    if lang == "ar":
        return _arabic_light(word)
    if lang == "bn":
        return _bengali_stem(word)
    stemmer = _STEMMERS.get(lang)
    return fold(_snowball(stemmer, word, lang)) if stemmer else word


def _snowball(stemmer, word: str, lang: str) -> str:
    """Snowball's stem. Dutch: a leading ge- that Snowball 3.x takes off is put back. It is taken off so that a past
    participle meets its verb ("gemaakt" / "maak"), and that turns nouns into other words ("gezin", family, into "zin",
    sense; "gerecht", dish, into "recht", right; "gezicht", face, into "zicht", view; "geboekt" into "boek", book)."""
    stem_ = stemmer.stemWord(word)
    if lang == "nl" and word.startswith("ge") and not stem_.startswith("ge") and len(word) > 4:
        return "ge" + stem_
    return stem_


# A conjunction (و ف) and/or a preposition (ب ك ل) glued to a word that starts with alef, as names and the article
# do: "لإيرما" is "for Irma", "ولإيرما" "and for Irma". A lexical first letter ("كتاب") is left alone.
_AR_CLITICS = re.compile(r"^[وف]?[بكل]?(?=ا)")


_TR_VOWELS = frozenset("aeıioöuü")


def _tr_possessive(word: str, stem_: str) -> str:
    """Snowball's Turkish leaves the buffer s of a possessive after a vowel ("kedisi" is "kedis", "annesi" "annes",
    "toplantısı" "toplantıs"): the noun without it ("kedi", "anne", "toplantı"); "" when the stem is not that."""
    if len(stem_) >= 4 and stem_[-1] == "s" and stem_[-2] in _TR_VOWELS and word.startswith(stem_) \
            and word[len(stem_):len(stem_) + 1] in ("ı", "i", "u", "ü"):
        return stem_[:-1]
    return ""


def _written(raw: str) -> str:
    """A lower-cased word as Snowball reads it: Turkish İ lower-cases to i with a combining dot, which is just i."""
    return unicodedata.normalize("NFC", raw.replace("i\u0307", "i"))


@lru_cache(maxsize=65536)
def _native_stems(written: str, lang: str) -> tuple[str, ...]:
    """Stems of a word of a _NATIVE language as written (not folded), unfolded; () without its stemmer."""
    stemmer = _STEMMERS.get(lang)
    if stemmer is None or is_identifier(written):
        return ()
    stem_ = _snowball(stemmer, written, lang)
    possessive = _tr_possessive(written, stem_) if lang == "tr" else ""
    return tuple(s for s in (stem_, possessive) if len(s) >= _MIN_STEM)


@lru_cache(maxsize=65536)
def _native_keys(written: str, lang: str) -> frozenset:
    """Keys of a word of a _NATIVE language stemmed as written, folded afterwards; and, when it has ı or ł, the word
    and its stems as FTS5 indexed them (unicode61 drops accents but keeps those letters), so the prefix query reaches
    "toplantı" and "szkoła" in the index although the comparison form is "toplanti", "szkola"."""
    stems_ = _native_stems(written, lang)
    keys = {fold(s) for s in stems_}
    if "ı" in written or "ł" in written:
        keys |= {_fold_text(s, dotless=False) for s in (written, *stems_)}
    return frozenset(keys)


@lru_cache(maxsize=65536)
def _undotted(written: str, langs: tuple[str, ...]) -> frozenset:
    """A word with ı or ł and its stems in *langs* as FTS5 indexed them (unicode61 drops accents but keeps those
    letters), for text of any language: "Işık" in an English question has to reach "ışık" in the index, as a Turkish
    one does through _native_keys; the comparison form stays "isik"."""
    stems_ = () if is_identifier(written) else tuple(
        _snowball(_STEMMERS[lang], written, lang) for lang in langs if lang in _STEMMERS and lang not in _NATIVE)
    return frozenset(_fold_text(s, dotless=False) for s in (written, *stems_) if s == written or len(s) >= _MIN_STEM)


@lru_cache(maxsize=65536)
def _keys(word: str, langs: tuple[str, ...], written: str = "", determined: bool = True) -> frozenset:
    """The word and its stems in *langs*. *written*: the word before folding, for languages Snowball reads better
    with its letters (_NATIVE) and for a word with ı or ł; "" when it is the same. *determined*=False: the text's
    language is not known, and a stem is kept only when it could be a word of its language (at least _MIN_STEM
    letters, not a function word): the Indonesian stemmer reads "media" as me- + "dia" (he, she), and a key "dia"
    would meet every Indonesian fact that says "dia"."""
    keys = {word}
    for lang in langs:
        stem_ = stem(word, lang)
        if lang not in _NATIVE:
            if determined or (len(stem_) >= _MIN_STEM and stem_ not in STOP[lang]):
                keys.add(stem_)
            continue
        if len(stem_) >= _MIN_STEM or stem_ == word:  # Snowball's Turkish leaves "ç" of "cuma", "ai" of "aile"
            keys.add(stem_)
        if lang == "tr" and (possessive := _tr_possessive(word, stem_)):
            keys.add(possessive)
        if written:
            keys |= _native_keys(written, lang)
    if written and ("ı" in written or "ł" in written) and _NATIVE.isdisjoint(langs):
        keys |= _undotted(written, langs)  # a _NATIVE language in langs added these through _native_keys
    if "ar" in langs:
        if "ar" in _STEMMERS:
            keys.add(fold(_STEMMERS["ar"].stemWord(word)))
        match = _AR_CLITICS.match(word)
        cut = match.end() if match else 0
        if cut and len(word) - cut >= 3:
            keys |= {word[cut:], _arabic_light(word[cut:])}
    if "pt" in langs and word.endswith("oes") and len(word) > 4:  # ação / ações: Snowball keeps -ão and -ões apart
        keys.add(stem(word[:-3] + "ao", "pt"))
    if "es" in langs and word.endswith("ces") and len(word) > 4:  # lápiz / lápices, vez / veces
        keys.add(stem(word[:-3] + "z", "es"))
    # A Latin spelling joins its group only in English or in text of no known language: German "über" folds to
    # "uber", French "chat" is a cat, Spanish "botes" are boats (review 2026-10-15).
    loan = _loan_keys(word) if word_script(word) != "latin" or "en" in langs else frozenset()
    return frozenset(keys | loan)


# -- one word in Latin and Cyrillic --------------------------------------------------------------------------------
# Russian text writes a borrowed word either way: "бэкап" and "backup", "телеграм" and "Telegram", "впн" and "VPN".
# Without search by meaning the two spellings never met (0.14.5). Each group below is one word; a word of a group gets
# the group's mark (a key with ":", which never reaches an FTS5 query) and the prefixes that find the other spellings
# in the index. A word belongs to a group by its exact form — the spelling, an English plural, a Russian case ending —
# never by its stem: Snowball cuts "токен" to "ток" (electric current) and "виза" to "виз" (as in "визит").
_LOANWORDS = """
telegram телеграм телеграмм | whatsapp вотсап вотсапп ватсап ватсапп вацап | youtube ютуб
instagram инстаграм инстаграмм | facebook фейсбук | google гугл | gmail гмейл | iphone айфон | ipad айпад
android андроид | windows виндовс | linux линукс | macbook макбук | backup бэкап бекап | server сервер | vpn впн
wifi вайфай | bluetooth блютуз | email имейл емейл эмейл | zoom зум | skype скайп | excel эксель | notion ношн
github гитхаб | docker докер | python питон | deploy деплой | commit коммит | token токен | prompt промпт
chatgpt чатгпт | dashboard дашборд | bitcoin биткоин биткойн | hotel отель | condo кондо | visa виза
netflix нетфликс | spotify спотифай | tiktok тикток | twitter твиттер | samsung самсунг | xiaomi сяоми
booking букинг | laptop лэптоп ноутбук | smartphone смартфон | router роутер | chat чат | online онлайн | site сайт
bot бот | marketplace маркетплейс | startup стартап | meetup митап | coworking коворкинг | freelance фриланс
"""
# Russian noun endings after the stem a spelling leaves: "сервер" -> "сервера", "серверов"; "виза" -> "визы", "визой";
# "отель" -> "отеля", "отелей". A spelling ending in о, и, у, э or ю is a word Russian does not decline ("кондо").
# A spelling whose forms are also a common Russian word is not listed: "убер" gives "уберу", "убери", "уберем" of
# "убрать" (to tidy up), so Uber is out (review 2026-10-15).
_RU_ENDINGS = ("а", "я", "у", "ю", "е", "и", "ы", "ой", "ей", "ою", "ею", "ом", "ем", "ам", "ям", "ами", "ями", "ах",
               "ях", "ов", "ев")
_LOAN: dict | None = None
_LOAN_LOCK = threading.Lock()


def _loan_forms(spelling: str) -> tuple[frozenset, frozenset, bool]:
    """(the forms of a spelling that make a word one of its group, the FTS5 terms that find those forms, whether those
    terms are whole words rather than prefixes). A short spelling is searched by its forms as whole words: "бот"* would
    bring in "ботинки" and "ботаника", "bot"* every "bottle" (review 2026-10-15)."""
    if word_script(spelling) != "cyrillic":
        plural = spelling + ("es" if spelling.endswith(("s", "x", "z", "ch", "sh")) else "s")  # not "botes"
        forms = frozenset({spelling, plural})
        return (forms, forms, True) if len(spelling) < 5 else (forms, frozenset({spelling}), False)
    if spelling[-1] in "оиуэю":
        return frozenset({spelling}), frozenset({spelling}), len(spelling) < 4
    base = spelling[:-1] if spelling[-1] in "аяьй" else spelling
    forms = frozenset({spelling} | {base + e for e in _RU_ENDINGS})
    # the base as a prefix when it is long enough to say little else ("отел", "сервер"); each form when it is not
    # ("виз" would bring in every "визит")
    return (forms, frozenset({base}), False) if len(base) >= 4 else (forms, forms, True)


def _loan_table() -> dict:
    """form -> the group's keys: its mark "loan:<first spelling>", which is all a comparison may meet (two words of a
    group share it), and its FTS5 terms as "fts=:<word>" (a whole word) or "fts*:<prefix>", which only Term.fts reads.
    The spellings themselves are not keys: a query's "chat" would meet a French fact's "chat" (a cat) that the
    language check kept out of the group, and the prefix "отел" a fact's stem of "отелиться" (review 2026-10-15)."""
    global _LOAN
    if _LOAN is None:
        with _LOAN_LOCK:
            if _LOAN is None:
                table = {}
                for group in _LOANWORDS.replace("\n", " | ").split("|"):
                    spellings = [fold(w) for w in group.split()]
                    if not spellings:
                        continue
                    parts = [_loan_forms(s) for s in spellings]
                    keys = frozenset({"loan:" + spellings[0]} | {("fts=:" if whole else "fts*:") + t
                                                                 for _, terms, whole in parts for t in terms})
                    for forms, _, _ in parts:
                        for form in forms:
                            table[form] = keys
                _LOAN = table
    return _LOAN


@lru_cache(maxsize=65536)
def _loan_keys(word: str) -> frozenset:
    """The group keys of a borrowed word written in Latin or Cyrillic (_LOANWORDS), or nothing."""
    return _loan_table().get(word, frozenset())


# -- case forms of Russian nouns that share no key (an experiment, off by default) ------------------------------------
# Snowball cuts «чай» to «ча», «мае» to «ма», «рта» to «рт»; a stem under _MIN_STEM is no key, so «чая» never meets
# «чай». A dropped vowel does the same with longer stems: «подарок» / «подарк(а)», «отец» / «отц(а)». A shared key for
# every two-letter stem would join different words («ель» / «ели» of «есть», «рой» / «рою» of «рыть»), so each line
# below is one noun's own case forms, written out and checked: on 0.22.1 most of them do not meet the first form; no
# other Russian lexeme of Wikidata (CC0) has a form spelled like them, or the form is left out; forms that are also a
# verb form are left out by hand (Wikidata has few Russian verbs): «ели», «шей», «рою», «вою», «лая», «матери»
# (материть); «потолок» (потолочь) is the noun's first form, and without it the rest of its forms already share the
# stem «потолк», so the noun is not listed (Codex review 2026-10-29). «маю» is left out too: in Ukrainian it is "I
# have". Kept although also another word, judged rare in what people write: «чая», «чаю», «чаем» of «чаять»; «рот» of
# «рота»; «звонка» of «звонкий». A word of a group gets the group's mark ("ru=:" + its first form), which only another
# form of the same group meets, and its forms as whole FTS5 words (as a borrowed word does). These keys are for search
# and recall only (Word.forms): the write-time checks (content_keys, the claim of a conflict) compare a word by its own
# keys, so «В мае я предпочитаю…» is no duplicate of «Мая предпочитает…». And only in Russian text (_ru_text): Cyrillic
# alone is not Russian («Я маю новий план» is Ukrainian). «день» is not listed: measured, «дни рождения» and «в какие
# дни» then met every «День рождения …» of the household set (two more questions with another person's birthday in the
# top five); a word that light needs the question's other words, which the gate weighs against it. Measured in
# plans/2026-10-29-two-topic-and-short-nouns-experiment.md (outside the repository).
_RU_FORM_GROUPS = False
_RU_FORMS = """
чай чая чаю чаем чае чаи чаёв чаям чаями чаях
сон сна сну сном сне сны снов снам снами снах | рот рта рту ртом рте рты ртов ртам ртами ртах
лоб лба лбу лбом лбе лбы лбов лбам лбами лбах | пёс пса псу псом псе псы псов псам псами псах
лёд льда льду льдом льде льды льдов льдам льдами льдах | май мая маем мае | рай раю раем рае
бой боя бою боем бое бои боёв боям боями боях | ум ума уму умом уме умы умов умам умах
имя имени именем имена имён именам именами именах | шея шеи шее шею шеей шеею шеям шеями шеях
ухо ухом уши ушей ушам ушами ушах | уха ухи ухой ухою | ель елью елям елями елях
яйцо яйца яйцу яйцом яйце яиц яйцам яйцами яйцах | отец отца отцу отцом отце отцы отцов отцам отцами отцах
мать матерью матерей матерям матерями матерях | дочь дочери дочерью дочерей дочерям дочерями дочерьми дочерях
ребёнок ребёнка ребёнку ребёнком ребёнке
подарок подарка подарку подарком подарке подарки подарков подаркам подарками подарках
рынок рынка рынку рынком рынке рынки рынков рынкам рынками рынках
звонок звонка звонку звонком звонке звонки звонков звонкам звонками звонках
кусок куска куску куском куске куски кусков кускам кусками кусках
палец пальца пальцу пальцем пальце пальцы пальцев пальцам пальцами пальцах
перец перца перцу перцем перце перцы перцев перцам перцами перцах
огурец огурца огурцу огурцом огурце огурцы огурцов огурцам огурцами огурцах
ветер ветра ветру ветром ветре ветры ветров ветрам ветрами ветрах
угол угла углу углом углы углов углам углами углах
"""
_RU_FORM_TABLE: dict | None = None
_RU_FORM_LOCK = threading.Lock()


def _ru_form_table() -> dict:
    """form -> the keys of its group: the mark "ru=:<first form>" and every form as "fts=:<form>" (a whole word for
    Term.fts). A form with «ё» is listed with «е» too, as people type it."""
    global _RU_FORM_TABLE
    if _RU_FORM_TABLE is None:
        with _RU_FORM_LOCK:
            if _RU_FORM_TABLE is None:
                table = {}
                for group in _RU_FORMS.replace("\n", " | ").split("|"):
                    forms = [fold(w) for w in group.split()]
                    if not forms:
                        continue
                    spellings = list(dict.fromkeys(s for f in forms for s in (f, f.replace("ё", "е"))))
                    keys = frozenset({"ru=:" + forms[0]} | {"fts=:" + s for s in spellings})
                    for s in spellings:
                        table[s] = keys
                _RU_FORM_TABLE = table
    return _RU_FORM_TABLE


@lru_cache(maxsize=65536)
def _ru_form_keys(word: str) -> frozenset:
    """The group keys of a case form of a listed Russian noun (_RU_FORMS), or nothing."""
    return _ru_form_table().get(word, frozenset())


# Common Ukrainian and Belarusian words spelled with letters Russian has too (і, ї, є, ґ, ў are _NOT_RUSSIAN): one of
# them in a text says it is not Russian, and the Russian case forms stay out of it. None of them is a Russian word.
_NOT_RUSSIAN_WORDS = _folded("""що це цей ця цю цих який яка яке якого якому якою якщо вона воно вони його ми ви
    вже ще або дуже треба можна був була було були будуть хоче можу можемо маю мають тобто отже проте але аби хто
    чому куди щось хтось десь вчора трохи мене також щоб чи тепер мого твого нашого вашого кожен кожна кожного багато
    чого поки спочатку завжди
    што гэта гэты гэтая ён яны таксама ёсць няма сёння учора твая яго яе трэба дзе чаму куды адкуль будзе каб""")


def _ru_text(lowered: str, words) -> bool:
    """Is a Cyrillic text Russian, as far as it tells: no letter of Ukrainian or Belarusian (_NOT_RUSSIAN) and none
    of their common words (_NOT_RUSSIAN_WORDS). A text of a few words may say nothing either way and counts as Russian;
    «Я маю новий план» does not."""
    return _NOT_RUSSIAN.search(lowered) is None and _NOT_RUSSIAN_WORDS.isdisjoint(words)


# -- same-root words ---------------------------------------------------------------------------------------------
# Snowball removes inflection and some derivation, not all: "ночь" is "ноч" but "ночные" is "ночн"; "teacher",
# "swimmer" and "decision" stay apart from "teach", "swim" and "decide". Root keys are the stems a dictionary links to
# the word's stem by a derivation (below), so the noun, the adjective, the verb and the agent noun of one root meet; a
# match through one counts _DERIV_WEIGHT of an exact one and never recalls a fact alone.
_NOT_RUSSIAN = re.compile(r"[іїєґў]")  # Ukrainian and Belarusian letters
_RU_ADJ = re.compile(r"(?:ый|ий|ой|ая|яя|ое|ее|ые|ие|ого|его|ому|ему|ым|им|ых|их|ую|юю|ыми|ими)$")
_RU_VERB = re.compile(r"(?:ть|ться|тся|ет|ют|ут|ит|ат|ят|ем|ешь|ишь|ете|ите|ется|ится|ются|утся|ятся|ла|ло|ли|лся|"
                      r"лась|лись)$")
_RU_VERBAL_NOUN = re.compile(r"ни(?:е|я|ю|ем|и|й|ям|ями|ях)$")
_RU_ADJ_SUFFIXES = ("ическ", "еск", "ск", "енн", "онн", "ичн", "ов", "ев", "ив", "н")
_EN_SUFFIXES = ("ist", "ian", "er", "or", "al", "ic")
# -er words that are not "one who …": "mother" is not one who moths, "summer" not one who sums
_EN_NOT_AGENT = frozenset("""mother father brother sister daughter water number summer winter corner paper letter order
    member other under after never over ever together weather power matter answer finger dinner butter tower flower
    river silver center centre monster master chapter cancer danger hunger anger manner banner border shoulder
    ladder filter shower sweater copper hammer wonder rubber pepper oyster lobster liver offer cover enter remember
    consider deliver discover gather former latter outer inner upper lower proper super tiger spider timber lumber
    minister register semester character temper litter corridor mirror horror error color colour honor honour
    favor favour labor labour humor humour mayor motor tractor doctor actor sector factor vendor""".split())


def word_roots(word: str, stem_: str, lang: str) -> frozenset:
    """Root keys of a folded word whose stem in *lang* is *stem_* (Russian and English; empty for other languages)."""
    if not _ROOTS or not stem_ or is_identifier(word):
        return frozenset()
    if lang == "ru":
        return _ru_roots(word, stem_)
    if lang == "en":
        return _en_roots(word, stem_)
    return frozenset()


# Russian words of one root come from a dictionary, not from suffix rules: DerivBase.Ru (UDer 1.1, Apache-2.0), its
# direct parent-child links as pairs of Snowball stems, kept when their shape is a derivation (tools/
# build_ru_derivation.py). Suffix rules cannot tell "ночные" / "ночь" from "полный" / "пол"; the dictionary can, and it
# also has what rules cannot reach ("друг" / "дружеский"). "rules" keeps 0.9.0's first rules.
_RU_DERIVATION = "dict"
# English the same way from Open English Wordnet 2025 (CC BY 4.0, built on Princeton WordNet 3.1): its derivation and
# pertainym links between single words, as Snowball stem pairs of the shape of a derivation (tools/
# build_en_derivation.py): "pay" / "payment", "decide" / "decision", "angry" / "anger"; rules gave "Peter" / "pet" and
# "meters" / "met". "rules" keeps the one-step suffix rules of 0.9.0.
_EN_DERIVATION = "dict"
_PAIRS: dict[str, dict] = {}
_PAIRS_LOCK = threading.Lock()


def _related(lang: str, stem_: str) -> frozenset:
    """Stems linked to *stem_* by a derivation in the *lang* dictionary (loaded on the first word of that language)."""
    pairs = _PAIRS.get(lang)
    if pairs is None:
        with _PAIRS_LOCK:
            pairs = _PAIRS.get(lang)
            if pairs is None:
                pairs = {}
                try:
                    with gzip.open(_DICTIONARIES / f"{lang}_derivation.tsv.gz", "rt", encoding="utf-8") as fh:
                        for line in fh:
                            a, tab, b = line.rstrip("\n").partition("\t")
                            if tab and not line.startswith("#"):
                                a, b = sys.intern(a), sys.intern(b)  # each stem kept once
                                pairs[a] = pairs.get(a, ()) + (b,)
                                pairs[b] = pairs.get(b, ()) + (a,)
                except (OSError, EOFError, ValueError, zlib.error) as exc:  # missing or damaged data file: no roots
                    _unreadable(_DICTIONARIES / f"{lang}_derivation.tsv.gz", exc)
                    pairs = {}
                _PAIRS[lang] = pairs
    return frozenset(pairs.get(stem_, ()))


def _ru_related(stem_: str) -> frozenset:
    return _related("ru", stem_)


@lru_cache(maxsize=65536)
def _ru_roots(word: str, stem_: str) -> frozenset:
    if _RU_DERIVATION == "dict":
        return _ru_related(stem_)
    return _ru_rule_roots(word, stem_)


def _ru_rule_roots(word: str, stem_: str) -> frozenset:
    out = set()

    def keep(s: str) -> str:
        s = s[:-1] if s.endswith("ь") else s
        if len(s) >= _MIN_STEM and s != stem_:
            out.add(s)
        return s

    if _RU_VERBAL_NOUN.search(word) and stem_[-2:] in ("ен", "ан") and len(stem_) >= 5:  # плаван(ие) → плав
        keep(stem_[:-2])
    elif _RU_ADJ.search(word):  # ночн(ые) → ноч, городск(ой) → город, банковск(ий) → банков → банк, школьн(ый) → школ
        s = stem_
        for _ in range(2):
            cut = next((x for x in _RU_ADJ_SUFFIXES if s.endswith(x) and len(s) - len(x) >= _MIN_STEM), "")
            if not cut:
                break
            s = keep(s[:-len(cut)])
    elif _RU_VERB.search(word) and stem_[-1:] in ("а", "я", "и", "е") and len(stem_) >= 4:  # плава(ет) → плав
        keep(stem_[:-3] if stem_[-3:] in ("ова", "ева", "ыва", "ива") and len(stem_) >= 6 else stem_[:-1])
    elif stem_.endswith("ост") and len(stem_) >= 6:  # сладост(и) → слад
        keep(stem_[:-3])
    elif stem_.endswith("ниц"):  # учительниц(а) → учитель
        keep(stem_[:-3])
    elif stem_.endswith(("ник", "чик")) and len(stem_) >= 6:  # праздник → праздн, как праздничн(ый)
        keep(stem_[:-2])
    elif stem_.endswith("к") and (len(stem_) >= 5 or stem_.endswith("чк")):  # соседк(а) → сосед, дочк(а) → доч; банк — нет
        keep(stem_[:-1])
    return frozenset(out)


@lru_cache(maxsize=65536)
def _en_roots(word: str, stem_: str) -> frozenset:
    if _EN_DERIVATION == "dict":
        return _related("en", stem_)
    return _en_rule_roots(word, stem_)


def _en_rule_roots(word: str, stem_: str) -> frozenset:
    """One step off the Porter2 stem, never two: teacher → teach, swimmer → swim, driver → drive, dentist / dental →
    dent, electrician → electric, investor → invest, photographi → photograph (battery stops at batter, not bat;
    general at gener, not gene). Words and stems in _EN_NOT_AGENT get none ("mothers" is not "moths")."""
    if word in _EN_NOT_AGENT or stem_ in _EN_NOT_AGENT:
        return frozenset()
    out = set()
    if stem_.endswith("i") and len(stem_) >= 7:  # Porter2 writes a final -y as -i: photographi, allergi, veterinari;
        out.add(stem_[:-1])                      # not every → ever, party → part, Sophie → soph
    else:
        cut = next((x for x in _EN_SUFFIXES if stem_.endswith(x) and len(stem_) - len(x) >= 3), "")
        agent = cut in ("er", "or")
        if cut:
            s = stem_[:-len(cut)]
            doubled = agent and len(s) >= 4 and s[-1] == s[-2] and s[-1] not in "aeiouls"
            s = s[:-1] if doubled else s  # swimm → swim, runn → run
            if len(s) >= 4 or (agent and len(s) >= 3):
                out.add(s)
                if agent and not doubled and s[-1] not in "aeiouy" and s[-2:-1] in tuple("aeiou"):
                    out.add(s + "e")  # driv → drive, bak → bake, writ → write (a doubled letter says no -e: swimmer)
    return frozenset(r for r in out if len(r) >= 3 and r != stem_)


# -- German compounds --------------------------------------------------------------------------------------------
# German writes a compound as one word: "Zahnarzttermin" is Zahn + Arzt + Termin, and no stem of it meets "Termin" or
# "Zahnarzt". In text whose language is determined as German, a word is split by a dictionary of German lexemes from
# Wikidata (CC0; de_compounds.tsv.gz, tools/build_de_es_lexicon.py), the way Koehn & Knight split compounds: parts
# of three letters or more joined by the linking elements German uses (-s-, -es-, -n-, -en-, -er-, -e-, a dropped -e),
# the analysis with the fewest parts, parts that are rarely parts of other words only when nothing else fits, two
# analyses that cut differently left whole unless one is clearly more likely ("Staubecken": Stau-Becken or
# Staub-Ecken). A word the dictionary knows as one word stays whole ("Kartoffel"), and so do its forms, and so does a
# compound whose meaning is not its parts' (_DE_ONE_WORD: "Hochzeit", a wedding, is no Zeit); a
# compound of the dictionary is cut where the dictionary (or Wikidata's own analysis, "combines lexemes") says. The
# last part names what the word is ("Familienkatze" is a Katze) and matches as the word itself does; the parts before
# it (modifiers) count _DERIV_WEIGHT and never recall a fact alone. Both ways: a compound in a fact is found by its
# parts, a compound in a question finds a fact that writes its parts apart ("Termin beim Zahnarzt").
_DE_COMPOUNDS = True
_DE_QUESTION_HEAD = _DERIV_WEIGHT  # a compound of the question whose last part alone is in a fact: "Elternabend" / "Abend"
_DE_FILE = _DICTIONARIES / "de_compounds.tsv.gz"
_DE_TABLE: list = []  # [splitter] once loaded; [None] for a copy without the data file
_DE_LOCK = threading.Lock()
# separable verb particles, prepositions and prefixes: never a part ("Ausland" is not aus + Land)
_DE_PREFIXES = frozenset("""ab an auf aus bei da dar ein emp ent er fort ge her hin miss mit nach ob um un ur ver vor
    weg zer zu zur zum uber unter durch wider wieder gegen hinter neben ohne seit statt trotz entlang zuruck zusammen
    dran drin drum los voll""".split())
# derivational suffixes that are also words: never the last part ("dauerhaft" is no Haft, "Gefährdung" no Dung)
_DE_NOT_HEADS = frozenset("""bar haft lein chen ion sam tum ling heit keit schaft ung nis lich isch los mal weise tion
    ismus ist ent ant ei erei in innen iker ik lei ich dung""".split())
_DE_LINKS = {"_": "", "s": "s", "z": "es", "n": "n", "m": "en", "r": "er", "e": "e", "x": "ens", "y": "nen"}
_DE_NOUN_END = ("", "e", "en", "n", "er", "s", "es", "ern", "ens", "nen", "se", "sen")
_DE_FEM_END = ("in", "innen")  # Lehrerin, Ärztinnen: the female form of a noun, not a compound with -in
_DE_ADJ_END = ("", "e", "en", "er", "es", "em")
_DE_VERB_END = ("e", "st", "t", "en", "et", "te", "ten", "test", "tet", "tes", "end", "ende", "enden")
_DE_CLASS = (0, 2, 9, 40, 200)  # how often a word is a part of other words, by class: the likelier analysis wins
_DE_TWO_LETTER_HEADS = frozenset({"ol"})  # Öl: Olivenöl, Heizöl
# Compounds whose meaning is not their parts', never split (nor their forms): "Hochzeit" (a wedding) is no Zeit,
# "Frühstück" no Stück, "Weihnachten" no Nacht, "Augenblick" (a moment) no Blick, "Verstand" (the mind) no Tand,
# "Wochenende" no Ende, "Flugzeug" no Zeug, "Reisende" (travellers) no Ende and no Reis; and words the dictionary lacks
# or cut wrongly ("Tablette" is no Lette). Folded, as the words are. A question "Wann hast du Zeit?" must not recall
# "Die Hochzeit ist im Juni" at full weight for the Zeit at its end.
_DE_ONE_WORD = frozenset("""hochzeit fruhstuck weihnachten augenblick verstand tablette kartoffel reisend wochenende
    jahreszeit mahlzeit handschuh flugzeug fahrzeug spielzeug werkzeug feierabend burgermeister""".split())


def _de_vstem(verb: str) -> str:
    """The stem a verb lends a compound: schwimmen → schwimm (Schwimmbad), wandern → wander (Wanderweg)."""
    if verb.endswith(("eln", "ern")) and len(verb) >= 6:
        return verb[:-1]
    return verb[:-2] if verb.endswith("en") and len(verb) >= 5 else ""


class _DeSplitter:
    """The splitter over a table of words (word -> (kinds, class, links)) and dictionary cuts (word -> cut positions).
    kinds: N noun, A adjective, V verb; class 0-4: how often the word is a part of other words; links: the linking
    elements seen after it as a modifier (codes of _DE_LINKS, "d" a dropped -e), None for any. *need*: the least class
    of a part of 3, 4 and 5+ letters (*head_need* for the last part, *need_verb* for a verb's stem). The build
    (tools/build_de_es_lexicon.py) judges the dictionary's own words with this class and stricter limits."""

    def __init__(self, words: dict, cuts: dict, need=(2, 1, 1), head_need=None, need_verb=3, margin=1.0,
                 min_part=3, min_word=6, max_parts=4):
        self.words, self.cuts_of = words, cuts
        self.need, self.head_need, self.need_verb = need, head_need or need, need_verb
        self.margin, self.min_part, self.min_word, self.max_parts = margin, min_part, min_word, max_parts

    @classmethod
    def load(cls, path) -> "_DeSplitter":
        words, cuts = {}, {}
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                word, tab, code = line.rstrip("\n").partition("\t")
                if not tab or line.startswith("#"):
                    continue
                if code.startswith("="):
                    cuts[word] = tuple(int(c) for c in code[1:].split(",") if c)
                else:
                    i = next(i for i, c in enumerate(code) if c.isdigit())
                    words[word] = (sys.intern(code[:i]), int(code[i]), code[i + 1:] or None)
        return cls(words, cuts)

    def _ok(self, part: str, cls: int, need) -> bool:
        return len(part) >= self.min_part and part not in _DE_PREFIXES and cls >= need[min(len(part), 5) - 3]

    def _kinds(self, word: str) -> str:
        entry = self.words.get(word)
        return entry[0] if entry else ""

    def _modifiers(self, s: str) -> list:
        """(lemma, class, kind) readings of s before another part: a noun with a linking element, an adjective, a
        verb's stem."""
        out, words = [], self.words
        if len(s) < self.min_part or s in _DE_PREFIXES:
            return out
        for code, link in _DE_LINKS.items():
            base = s[:len(s) - len(link)] if link else s
            entry = words.get(base) if s.endswith(link) else None
            if entry and "N" in entry[0] and (entry[2] is None or code in entry[2]) and self._ok(base, entry[1], self.need):
                out.append((base, entry[1], "N"))
        entry = words.get(s + "e")  # Schule → Schulbus
        if entry and "N" in entry[0] and (entry[2] is None or "d" in entry[2]) and self._ok(s, entry[1], self.need):
            out.append((s + "e", entry[1], "N"))
        entry = words.get(s)
        if entry and "A" in entry[0] and self._ok(s, entry[1], self.need):
            out.append((s, entry[1], "A"))
        for stem_ in (s, s[:-1] if s.endswith("e") else ""):  # Schwimmbad, Wartezimmer
            if len(stem_) >= self.min_part and stem_ not in _DE_PREFIXES:
                for verb in (stem_ + "en", stem_ + "n"):
                    entry = words.get(verb)
                    if entry and "V" in entry[0] and entry[1] >= self.need_verb and _de_vstem(verb) == stem_:
                        out.append((verb, entry[1], "V"))
        return out

    def _heads(self, s: str) -> list:
        """(lemma, class, kind) readings of s as the last part: a noun in any of its forms, a verb's infinitive used
        as a noun (Abendessen)."""
        out, words = [], self.words
        for end in _DE_NOUN_END:
            base = s[:len(s) - len(end)] if end else s
            entry = words.get(base) if s.endswith(end) else None
            if entry and "N" in entry[0] and base not in _DE_NOT_HEADS and \
                    (self._ok(base, entry[1], self.head_need) or base in _DE_TWO_LETTER_HEADS):
                out.append((base, entry[1], "N"))
        for end in ("", "s"):
            base = s[:len(s) - len(end)] if end else s
            entry = words.get(base) if s.endswith(end) else None
            if entry and entry[0] == "V" and base.endswith("en") and entry[1] >= self.need_verb:
                out.append((base, entry[1], "V"))
        return out

    def known(self, word: str) -> bool:
        """A word of the table that is not a compound, or one of its forms: never split."""
        kinds = self._kinds
        if any(word.endswith(e) and "N" in kinds(word[:len(word) - len(e)] if e else word) for e in _DE_NOUN_END):
            return True
        if any(word.endswith(e) and ("N" in kinds(word[:-len(e)]) or "N" in kinds(word[:-len(e)] + "e"))
               for e in _DE_FEM_END):
            return True
        if any(word.endswith(e) and "A" in kinds(word[:len(word) - len(e)] if e else word) for e in _DE_ADJ_END):
            return True
        for end in _DE_VERB_END:
            base = word[:-len(end)]
            if word.endswith(end) and ("V" in kinds(base + "en") or "V" in kinds(base + "n")):
                return True
        if word.startswith("ge") and word.endswith(("t", "en")):  # participles: gemacht, gefahren
            base = word[2:-1] if word.endswith("t") else word[2:-2]
            if "V" in kinds(base + "en") or "V" in kinds(base + "n"):
                return True
        return bool(kinds(word))

    def segmentations(self, word: str) -> list:
        """Every reading of word as modifiers + a last part, each part (surface, lemma, class, kind)."""
        memo: dict = {}

        def segs(i: int, depth: int) -> list:
            if (i, depth) not in memo:
                out = [((word[i:], *head),) for head in self._heads(word[i:])] if i else []
                if depth < self.max_parts - 1:
                    for j in range(i + self.min_part, min(len(word) - 2, i + 24) + 1):
                        readings = self._modifiers(word[i:j])
                        if readings:
                            tails = segs(j, depth + 1)
                            out += [((word[i:j], *m), *t) for m in readings for t in tails]
                memo[(i, depth)] = out
            return memo[(i, depth)]

        # "-ende" after a verb's stem is a present participle (Reisende), not Ende
        return [s for s in segs(0, 0) if not (s[-1][1] == "ende" and s[-2][3] == "V")]

    @staticmethod
    def cuts(parts) -> frozenset:
        """Positions of the cuts between parts (surfaces, or (surface, ...) tuples)."""
        out, pos = [], 0
        for part in parts[:-1]:
            pos += len(part[0] if isinstance(part, tuple) else part)
            out.append(pos)
        return frozenset(out)

    @staticmethod
    def _score(seg) -> float:
        return sum(math.log(1 + _DE_CLASS[p[2]]) for p in seg)

    def choose(self, word: str):
        """The analysis with the fewest parts; analyses that cut differently are joined when one segmentation has
        all their cuts (Zahnarzt|termin, Zahn|arzttermin: Zahn|arzt|termin), else the likelier wins by *margin*, else
        None. The word's own entry is not looked at (the build judges the dictionary's words with this)."""
        by_cuts: dict = {}
        for seg in self.segmentations(word):
            by_cuts.setdefault(self.cuts(seg), []).append(seg)
        if not by_cuts:
            return None
        fewest = min(len(c) for c in by_cuts)
        top = [c for c in by_cuts if len(c) == fewest]

        def best(c):
            return max(by_cuts[c], key=self._score)

        if len(top) > 1:
            union = frozenset().union(*top)
            if union in by_cuts:
                top = [union]
            else:
                top.sort(key=lambda c: self._score(best(c)), reverse=True)
                if self._score(best(top[0])) - self._score(best(top[1])) < self.margin:
                    return None
        return best(top[0])

    def analyse(self, word: str) -> tuple | None:
        """Surfaces of the parts of a folded word ("zahn", "arzt", "termin"); () for a word of the table that is no
        compound (or one of its forms); None for a word the table knows nothing of."""
        if len(word) < self.min_word:
            return () if self.known(word) else None
        if any(word.endswith(end) and (word[:len(word) - len(end)] if end else word) in _DE_ONE_WORD
               for end in _DE_NOUN_END):
            return ()  # a lexicalised compound, before the dictionary's own cuts ("Reisende" has one)
        for end in _DE_NOUN_END + _DE_FEM_END:
            base = word[:len(word) - len(end)] if end else word
            if word.endswith(end) and base in self.cuts_of:
                cut = list(self.cuts_of[base])
                return tuple(word[a:b] for a, b in zip([0, *cut], [*cut, len(word)])) if cut else ()
        if self.known(word):
            return ()
        seg = self.choose(word)
        return tuple(p[0] for p in seg) if seg else None

    def split(self, word: str) -> tuple:
        """Surfaces of the parts of a folded word; () when it is not split."""
        return self.analyse(word) or ()


def _de_splitter():
    """The splitter, loaded on the first German word that might be a compound; None without the data file."""
    if not _DE_TABLE:
        with _DE_LOCK:
            if not _DE_TABLE:
                try:
                    _DE_TABLE.append(_DeSplitter.load(_DE_FILE))
                except (OSError, EOFError, ValueError, zlib.error, StopIteration, IndexError, KeyError) as exc:
                    _unreadable(_DE_FILE, exc)  # missing or damaged
                    _DE_TABLE.append(None)
    return _DE_TABLE[0]


@lru_cache(maxsize=65536)
def de_parts(word: str) -> tuple | None:
    """Parts of a folded German word; () when the dictionary knows it as one word, None when it knows nothing of it
    (or is missing)."""
    splitter = _de_splitter() if _DE_COMPOUNDS and word.isalpha() else None
    return splitter.analyse(word) if splitter else None


def _compound_keys(parts: tuple) -> tuple:
    """(head keys, modifier keys, keys of each modifier) of a compound's parts. Heads: the last part and every run of
    parts that ends with it ("termin", "arzttermin": a Zahnarzttermin is an Arzttermin); modifiers: every run before
    it ("zahn", "arzt", "zahnarzt"). A part keeps its linking element, whose stem drops it ("familien" / "famili").
    Each modifier's keys come with the keys of all the modifiers together, which stand in for any of them."""
    n = len(parts)
    head = frozenset().union(*(_keys("".join(parts[i:]), ("de",)) for i in range(1, n)))
    mods = frozenset().union(*(_keys("".join(parts[i:j]), ("de",)) for i in range(n - 1) for j in range(i + 1, n)))
    joined = _keys("".join(parts[:-1]), ("de",)) if n > 2 else frozenset()
    return head, mods, tuple(_keys(p, ("de",)) | joined for p in parts[:-1])


# -- German and Spanish verb forms -------------------------------------------------------------------------------
# Snowball leaves the forms of a strong or irregular verb apart from its lemma: "ging" is not "gehen", "fuhr" not
# "fahren", "fue" not "ir" (or "ser"), "hizo" not "hacer". A table of such forms from Wikidata's German and Spanish
# lexemes (CC0; de_es_forms.tsv.gz, tools/build_de_es_lexicon.py) gives the word the stems of its lemmas, at full
# weight like a stem, in text of that language only (Snowball's own stem already joins the regular forms). Forms that
# are also another word ("Macht", "vino", "cuenta") or a form of another verb are not in the table.
_LEMMA_FORMS = True
_FORMS_FILE = _DICTIONARIES / "de_es_forms.tsv.gz"
_FORMS: dict[str, tuple] = {}  # language -> (form -> lemmas, lemma -> forms)
_FORMS_LOCK = threading.Lock()


def _forms(lang: str) -> tuple:
    """(form -> lemmas, lemma -> forms) of *lang*, loaded on its first word; empty without the data file."""
    table = _FORMS.get(lang)
    if table is None:
        with _FORMS_LOCK:
            table = _FORMS.get(lang)
            if table is None:
                forms: dict = {}
                lemmas: dict = {}
                try:
                    with gzip.open(_FORMS_FILE, "rt", encoding="utf-8") as fh:
                        for line in fh:
                            code, _, rest = line.partition("\t")
                            if code != lang:
                                continue
                            lemma, _, written = rest.rstrip("\n").partition("\t")
                            lemma = sys.intern(lemma)
                            lemmas[lemma] = tuple(written.split())
                            for form in lemmas[lemma]:
                                forms[form] = forms.get(form, ()) + (lemma,)
                    # a form under more than three verbs is an error of the source ("tuviese" under twelve)
                    forms = {form: under for form, under in forms.items() if len(under) <= 3}
                except (OSError, EOFError, ValueError, zlib.error, IndexError, KeyError) as exc:  # missing or damaged
                    _unreadable(_FORMS_FILE, exc)
                    forms, lemmas = {}, {}
                table = _FORMS[lang] = (forms, lemmas)
    return table


@lru_cache(maxsize=65536)
def word_lemmas(word: str, lang: str) -> tuple:
    """(lemma stems, lemmas) of a folded German or Spanish word: the lemmas whose form it is ("ging": gehen), and
    the word itself when it is a lemma of the table (its forms are searched for it); ((), ()) for other words."""
    if not _LEMMA_FORMS or lang not in ("de", "es") or not word.isalpha():
        return (), ()
    forms, lemmas = _forms(lang)
    found = forms.get(word, ())
    own = (word,) if word in lemmas else ()
    return tuple(dict.fromkeys(stem(lemma, lang) for lemma in found)), found + own


# -- tokenisation --------------------------------------------------------------------------------------------------

# French/Italian elision: "d'Irma", "l'offre", "qu'on" are Irma, offre, on; Italian "quest'anno", "dov'è", "com'è",
# "cos'è", "anch'io", "nessun'altra" are anno, è, è, è, io, altra (not before a lone s: English "dev's", "Bell's").
_ELISION = re.compile(r"^(?:l|d|j|m|n|s|t|c|qu|jusqu|lorsqu|puisqu|quoiqu|dell|all|nell|sull|dall|un)'(?=[^\W\d_])"
                      r"|^(?:quest|quell|dov|com|cos|anch|nessun|tutt|sant|senz|bell|buon|quant|quand|qual|dev)'"
                      r"(?=[^\W\d_])(?!s$)")
# Turkish writes the case ending of a name after an apostrophe ("Ankara'da", "Ayşe'nin", "HR24'ün", "hr-vps'de"): in
# text that may be Turkish the name is the word, and the whole stays in the bag. These endings only (folded), so
# "Irma's", "o'clock" and "rock'n'roll" are left alone.
_TR_ENDINGS = _folded("""a e ı i u ü ya ye yı yi yu yü na ne nı ni nu nü da de ta te nda nde dan den tan ten ndan nden
    ın in un ün nın nin nun nün yın yin yun yün la le yla yle lar ler lara lere larda lerde lardan lerden ların lerin
    ları leri sı si su sü sına sine sında sinde sından sinden sının sinin daki deki taki teki ndaki ndeki dır dir dur
    dür tır tir tur tür dı di du dü tı ti tu tü ydı ydi ca ce ça çe""")
# Pronouns after the verb, taken off only in text of that language: French inversion ("prend-il", "a-t-elle",
# "est-ce", "vas-y"; not the noun "rendez-vous") and Portuguese enclisis ("chama-se", "lembra-me").
_FR_CLITICS = frozenset("je tu il elle on nous vous ils elles ce moi toi lui leur le la les y en".split())
_FR_COMPOUNDS = frozenset({"rendez-vous"})
_PT_CLITICS = frozenset("se me te lhe lhes nos vos o a os as lo la los las no na nas".split())
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _edge(c: str) -> bool:
    return c in PUNCT or (not c.isascii() and unicodedata.category(c)[0] == "P")


def _strip_edges(token: str) -> str:
    i, j = 0, len(token)
    while i < j and _edge(token[i]):
        i += 1
    while j > i and _edge(token[j - 1]):
        j -= 1
    return token[i:j]


def _script_split(token: str) -> list[str]:
    """A hyphenated token split where the script changes: "hr-vps-এর" is hr-vps + এর, "IT-тусовка" is it +
    тусовка, "03:00-এ" is 03:00 + এ (digits and symbols go with Latin)."""
    if "-" not in token or UNSEGMENTED.search(token):
        return [token]
    groups: list[list] = []
    for part in token.split("-"):
        script = word_script(part)
        script = "latin" if script == "common" else script
        if groups and groups[-1][0] == script:
            groups[-1][1].append(part)
        else:
            groups.append([script, [part]])
    return ["-".join(parts) for _, parts in groups]


def _letters_only(group: str) -> bool:
    return not any(c.isdigit() or c in "._:/" for c in group)


def _resolve_hyphen(group: str, langs: tuple, determined: bool) -> tuple[str, list[str]]:
    """(the word, extra bag entries) of a hyphenated word: reduplication is the word ("anak-anak"); a French or
    Portuguese verb loses its pronoun in text of that language; the whole and its parts stay available to
    containment checks ("sextas-feiras" contains "sextas")."""
    parts = group.split("-")
    if len(parts) < 2 or not all(parts) or not _letters_only(group):
        return group, []
    kept = parts
    if len(set(parts)) == 1:
        kept = parts[:1]
    elif determined and "fr" in langs and group not in _FR_COMPOUNDS:
        while len(kept) > 1 and kept[-1] in _FR_CLITICS:
            kept = kept[:-1]
        if len(kept) > 1 and kept[-1] == "t":
            kept = kept[:-1]
    elif determined and "pt" in langs:
        while len(kept) > 1 and kept[-1] in _PT_CLITICS:
            kept = kept[:-1]
    return "-".join(kept), [group] * (len(kept) != len(parts)) + parts


def _apostrophe(group: str, langs: tuple, determined: bool) -> tuple[str, list[str]]:
    """(the word, extra bag entries) of a word with an apostrophe inside: a Turkish ending off a name ("ankara'da" is
    ankara) in text that may be Turkish; the Dutch plural and possessive 's ("pinda's", "irma's") in Dutch text."""
    head, _, tail = group.partition("'")
    if not head or not tail:
        return group, []
    if "tr" in langs and fold(tail) in _TR_ENDINGS:
        return head, [group]
    if determined and "nl" in langs and "en" not in langs and tail == "s" and len(head) > 1:
        return head, [group]
    return group, []


@dataclass(frozen=True)
class Word:
    text: str  # folded, lower-cased
    stem: str  # primary stem (the first language's)
    keys: frozenset  # the word and its stems
    stop: bool
    identifier: bool
    scan: bool  # matched against the facts' text: the FTS5 index cannot find its other forms
    fts: bool = True  # the FTS5 index can find the word itself (not Arabic or Indic script)
    roots: frozenset = frozenset()  # same-root keys (word_roots), compared at _DERIV_WEIGHT
    root_lang: str = ""  # "en" / "ru" when the word's language is determined: roots are built and compared only then
    head: frozenset = frozenset()  # a German compound: keys of its last part (and the runs ending with it)
    mods: frozenset = frozenset()  # ... keys of the parts before it (and their runs)
    mod_parts: tuple = ()  # ... keys of each part before it
    lemmas: frozenset = frozenset()  # German / Spanish: stems of the lemmas whose form the word is ("ging": geh)
    lemma_of: tuple = ()  # those lemmas, and the word when it is one: a search for it also looks for their forms
    form_lang: str = ""  # "de" / "es" when the word's language is determined: lemma stems are compared only then
    # the keys of its Russian case-form group (_RU_FORMS, an experiment) that are in `keys`: search and recall only, the
    # write-time checks leave them out (content_keys, FactRetriever._claim)
    forms: frozenset = frozenset()


@dataclass(frozen=True)
class Analysis:
    words: tuple  # Word, in order
    bag: frozenset  # folded words, plus the parts of hyphenated ones
    keys: frozenset  # every key of every word in the bag
    text: str  # folded, lower-cased text, for boundary matches
    langs: tuple  # languages of the Latin words (all of LATIN when undetermined)
    german: bool = False  # determined German: its compounds end in the word they name ("Familienkatze" is a Katze)
    roots: frozenset = frozenset()  # (language, root key) of every word (search and recall only, not duplicate checks)
    lang_keys: frozenset = frozenset()  # (language, key) of the words whose language is determined
    determined: bool = False  # the Latin language was told by function words
    seamed: str = ""  # Chinese: the folded text with SEAM at the seams between its words (zh_seamed); "" otherwise
    dutch: bool = False  # determined Dutch: compounds as in German ("kinderarts" is an arts), see _NL_COMPOUNDS
    heads: frozenset = frozenset()  # keys of the last parts of its German compounds (search and recall only)
    mods: frozenset = frozenset()  # keys of the parts before them
    lemma_keys: frozenset = frozenset()  # (language, key or lemma stem) of the words of determined German / Spanish
    suffix_words: frozenset | None = None  # German words the compound dictionary knows nothing of (None: all)


def _stop(word: str, script: str, votes: dict, sentence: bool) -> bool:
    """A function word of the text's language. For Latin script that language is told by the other words: "son" in
    "where does my son study" is a word, in "il boit son café" French "his". A word that is the only evidence decides
    only in a sentence ("Ana come carne todos los días"), not in a short message ("son", "bisa"): there, with no other
    evidence, the English list applies, as it did to all Latin text before 0.5.0."""
    if script != "latin":
        return any(word in STOP[lang] for lang in _SCRIPT_LANGS.get(script, ()))
    others = {lang: votes[lang] - (word in _VOTERS[lang]) for lang in LATIN}
    best = max(others.values())
    if best:
        top = [lang for lang in LATIN if others[lang] == best]
        # a _WEAK word only where its language alone is best supported
        return any(word in STOP[lang] and (len(top) == 1 or word not in _WEAK.get(lang, ())) for lang in top)
    if sentence:
        return any(word in STOP[lang] for lang in LATIN if votes[lang])
    return word in STOP["en"]


@lru_cache(maxsize=1024)
def analyze(text: str) -> Analysis:
    lowered = _CONTROL.sub(" ", unicodedata.normalize("NFC", text or "")).lower().replace("’", "'")
    groups = []
    for token in lowered.split():
        token = _ELISION.sub("", _strip_edges(token))
        groups += [g for g in (_strip_edges(g) for g in _script_split(token)) if g] if token else []
    # Latin language by function words (ties keep every tied language; none found: all of LATIN; _WEAK words do not
    # vote). A hyphenated word votes with its first part only: "est-ce" is French by "est", "carry-on" is not French
    # by "on".
    latin = [fold(g.split("-")[0]) for g in groups if word_script(g) == "latin" and _letters_only(g) and g.split("-")[0]]
    votes = {lang: sum(w in _VOTERS[lang] for w in latin) for lang in LATIN}
    best = max(votes.values())
    if best:
        langs, determined = tuple(lang for lang in LATIN if votes[lang] == best), True
    else:
        langs, determined = tuple(lang for lang in LATIN if lang not in _EVIDENCE or (
            _EVIDENCE[lang] is not None and _EVIDENCE[lang].search(lowered))), False
    # Spanish, French, Portuguese, Indonesian and German write compound words with a hyphen; English text, text of the
    # newer languages and text of no determined language keep letters-only hyphenated tokens as identifiers ("hr-vps",
    # also in "hr-vps'de")
    compounds = determined and _HYPHEN_COMPOUNDS.issuperset(langs)  # every tied language ("de" is es/fr/pt/nl)
    english = determined and "en" in langs  # a tie ("on" is English and French) still includes English
    # Cyrillic is Russian for roots unless the text has letters Russian does not (Ukrainian, Belarusian): the Russian
    # suffix rules would join "головний" (main) with "гол" (goal)
    russian = _NOT_RUSSIAN.search(lowered) is None
    # Russian case forms (_RU_FORMS, an experiment, off by default): only in a text that is Russian as far as it tells
    ru_forms = _RU_FORM_GROUPS and russian and _ru_text(lowered, {fold(g) for g in groups})
    # German compounds are split, and German or Spanish verb forms get their lemmas, only in text of that language
    german = determined and "de" in langs and "en" not in langs
    forms_lang = "" if not _LEMMA_FORMS else "de" if german else         "es" if determined and "es" in langs and "en" not in langs else ""  # "he went": English as much as Spanish
    heads, compound_mods, lemma_keys = set(), set(), set()
    # Japanese spelling variants, and kanji kept as written: decided for the whole text, so every word of it folds as
    # the text does ("申込" in "申込 締切 いつ？" has no kana of its own)
    fold_word = _fold_ja if _KANA.search(lowered) else fold
    lang_keys = set()
    words, bag, keys, roots = [], set(), set(), set()
    for group in groups:
        group, extra = _apostrophe(group, langs, determined) if "'" in group else (group, [])
        raw, parts = _resolve_hyphen(group, langs, determined) if "-" in group else (group, [])
        extra += parts
        word, script, ident = fold_word(raw), word_script(raw), is_identifier(raw)
        if ident and compounds and "-" in raw and _letters_only(raw):
            ident = False  # "week-end", "sexta-feira", "rendez-vous": a compound word, not a host name
        unsegmented = UNSEGMENTED.search(raw) is not None
        word_langs = () if ident or unsegmented else langs if script == "latin" else _SCRIPT_LANGS.get(script, ())
        is_stop = not ident and not unsegmented and _stop(word, script, votes, len(latin) >= 4)
        written = _written(raw) if word_langs and (not _NATIVE.isdisjoint(word_langs) or "ı" in raw or "ł" in raw) \
            else word
        written = "" if written == word else written  # only a word whose letters folding changed is stemmed twice
        word_keys = _keys(word, word_langs, written, determined) if word_langs else frozenset({word})
        forms = _ru_form_keys(word) - word_keys if ru_forms and script == "cyrillic" and not ident else frozenset()
        if forms:
            word_keys |= forms
        bag.add(word)
        keys |= word_keys
        for piece in (fold(e) for e in extra):  # the whole and the parts of a hyphenated word, for containment
            bag.add(piece)
            keys |= _keys(piece, word_langs, "", determined) if word_langs and _letters_only(piece) else {piece}
        root_lang = "" if is_stop or ident or unsegmented or "-" in word else \
            "en" if script == "latin" and english else "ru" if script == "cyrillic" and russian else ""
        word_roots_ = word_roots(word, stem(word, root_lang), root_lang) - word_keys if root_lang else frozenset()
        if root_lang:
            roots |= {(root_lang, r) for r in word_roots_}
            lang_keys |= {(root_lang, k) for k in word_keys}
        plain = script == "latin" and not (ident or "-" in word)
        parts = (de_parts(word) or ()) if german and plain and not is_stop else ()
        head, mods, mod_parts = _compound_keys(parts) if parts else (frozenset(), frozenset(), ())
        heads |= head
        compound_mods |= mods
        form_lang = forms_lang if plain else ""  # function words too: "fue" (ser, ir) in a fact meets "va" in a question
        lemma_stems, lemma_of = word_lemmas(word, form_lang) if form_lang else ((), ())
        lemma_stems = frozenset(lemma_stems) - word_keys
        if form_lang:
            lemma_keys |= {(form_lang, k) for k in word_keys | lemma_stems}
        words.append(Word(
            text=word, stem=_primary_stem(word, word_langs[0], written) if word_langs else word, keys=word_keys,
            stop=is_stop, identifier=ident,
            # Indonesian prefixes (me-, di-, ber-), compounds and Turkish consonant changes hide a word from a prefix
            # query (see _SCAN_LATIN); text of undetermined language might be in any of them
            scan=script in _SCAN_SCRIPTS or (script == "latin" and not _SCAN_LATIN.isdisjoint(word_langs)),
            fts=script not in _SCAN_SCRIPTS, roots=word_roots_, root_lang=root_lang, head=head, mods=mods,
            mod_parts=mod_parts, lemmas=lemma_stems, lemma_of=lemma_of, form_lang=form_lang, forms=forms))
    # the suffix test of 0.6.0 ("katze" ends "familienkatze") stays for the words the dictionary knows nothing of
    unknown = frozenset(w for w in bag if de_parts(w) is None) if german and _DE_COMPOUNDS and _de_splitter() else None
    folded = _fold_text(lowered)
    if fold_word is _fold_ja:
        # A Japanese text also keeps its spelling as written, so a question in kanji only (not folded: it has no kana)
        # still meets it (締切 / 締切は明日), and, since such a question folds its kanji as Traditional Chinese
        # characters (締切 -> 缔切), that form of the written text too
        forms = (folded, _fold_text(lowered, False), _fold_text(lowered, False, han=True))
        folded = "\n".join(dict.fromkeys(forms))
    return Analysis(tuple(words), frozenset(bag), frozenset(keys), folded, langs,
                    german=german, determined=determined, roots=frozenset(roots), lang_keys=frozenset(lang_keys), seamed=zh_seamed(folded),
                    dutch=_NL_COMPOUNDS and determined and "nl" in langs and "en" not in langs, heads=frozenset(heads),
                    mods=frozenset(compound_mods), lemma_keys=frozenset(lemma_keys), suffix_words=unknown)


def _primary_stem(word: str, lang: str, written: str) -> str:
    """The stem a word is ranked by (the first language's): for _NATIVE languages the one of the word as written, and
    never a stem shorter than _MIN_STEM ("cuma" is not "c")."""
    if lang not in _NATIVE:
        return stem(word, lang)
    native = _native_stems(written, lang) if written else ()
    if native:
        return fold(native[0])
    stem_ = stem(word, lang)
    return stem_ if len(stem_) >= _MIN_STEM else word


def name_forms(word: str) -> frozenset:
    """The keys of a folded word read as a name: the word and every stem, the short ones too. «Яна», «Яны» and «Яне»
    share «ян», which the words of a plain Russian text do not carry: _keys keeps a stem under _MIN_STEM only in text
    whose language is determined, and that is told by Latin function words ("the Яна" has «ян», «Яна любит чай»
    has not). frozenset() for a word of a script whose names are not inflected (a Latin name matches as written)
    or an unsegmented run; the callers leave identifiers out."""
    script = word_script(word)
    if script not in _SCRIPT_LANGS or UNSEGMENTED.search(word):
        return frozenset()
    return _keys(word, _SCRIPT_LANGS[script])


def name_keys(name: str) -> tuple[tuple[frozenset, bool], ...]:
    """(forms, exact?) of each word of an entity name. Latin-script names are not inflected, so they match only
    themselves (and the English possessive): "Jean" is not mentioned by "jeans", "Rose" not by "roses". Names in
    other scripts match by their keys, which carry their inflections ("Маши" mentions "Маша", "لإيرما" "إيرما")."""
    out = []
    for word in analyze(name).words:
        forms = name_forms(word.text) if not word.identifier else frozenset()
        out.append((forms, False) if forms else (frozenset({word.text, word.text + "'s"}), True))
    return tuple(out)


# A Russian name's case ending -> the endings its first form may have after the stem; "~" marks a stem that may have
# lost a vowel, looked up in _RU_NAME_DROPPED («Льв-а» of «Лев»). Every ending that fits is read: «Зою» is «-ою» of «За…» and
# «-ю» of «Зоя», «Илью» «-ью» and «-ю» of «Илья». «-ой», «-ы» are only «-а» names («Яной»,
# «Яны» of «Яна»), «-ом» only names with no ending («Яном» of «Ян»); «-а», «-у», «-е» are either («Яну»: «Ян», «Яна»).
# «-ей», «-ею» after ж, ш, ч, щ, ц are «-а» names too («Яшей» of «Яша»; Codex review): "^" is «-а» after those only.
_RU_NAME_CASES = (("ью", ("ь",)), ("ою", ("а",)), ("ею", ("я", "^")), ("ой", ("а",)), ("ей", ("я", "^")), ("ом", ("", "~")),
                  ("ем", ("ь", "й")), ("ы", ("а",)), ("и", ("я", "ь", "й", "а", "~")), ("а", ("", "~")),
                  ("я", ("ь", "й")), ("у", ("", "а", "~")), ("ю", ("ь", "й", "я")),
                  ("е", ("", "а", "я", "ь", "~")))
# The names whose stem drops a vowel in its other forms, by that stem. Only these: a rule for every final pair of
# consonants made «Марек» of «Марка» and «Петер» of «Петра» (Codex review).
_RU_NAME_DROPPED = {"льв": "лев", "павл": "павел", "любв": "любовь"}


@lru_cache(maxsize=65536)
def name_lemmas(word: str) -> frozenset:
    """The first forms a lower-cased Russian name may have, read from its case ending: «Льва» -> «лев», «льва» (a
    name in «-а», as written), «Никите» -> «никита», «Яной» -> «яна», «Яном» -> «ян». Two forms of one name share one;
    «Яны» and «Ян» do not, nor «Аня» and «Анна», while «Яну» (of «Ян» or of «Яна») shares with both. Unlike a stem,
    this keeps what the ending says. frozenset() for a word not in Cyrillic (a Latin name matches as written)."""
    word = word.replace("ё", "е")
    if word_script(word) != "cyrillic" or len(word) < 2:
        return frozenset()
    out = {word}
    for ending, firsts in _RU_NAME_CASES:
        stem_ = word[:-len(ending)]
        if not word.endswith(ending) or not stem_:
            continue
        for first in firsts:
            if first == "^":
                if stem_[-1] in "жшчщц":
                    out.add(stem_ + "а")
            elif first != "~":
                out.add(stem_ + first)
            elif stem_ in _RU_NAME_DROPPED:
                out.add(_RU_NAME_DROPPED[stem_])
    return frozenset(out)


def words(text: str) -> list[str]:
    """Folded, lower-cased words with surrounding punctuation stripped."""
    return [w.text for w in analyze(text).words]


def stems(text: str) -> set[str]:
    """Every comparison key of the text's words."""
    return set(analyze(text).keys)


def content_keys(text: str) -> list[frozenset]:
    """Key sets of the informative words (stopwords dropped), distinct; unsegmented runs give their content n-grams.
    Compared with soft_overlap, so two texts whose language was told apart differently (one has a function word, the
    other none) still meet on a shared stem."""
    out = []
    for w in analyze(text).words:
        if UNSEGMENTED.search(w.text):
            out += [frozenset({g}) for g, weight in run_grams(w.text) if weight]
        elif w.identifier or (not w.stop and len(w.text) > 1):
            out.append(w.keys - w.forms if w.forms else w.keys)  # a case-form group is for recall, not for this
    return list(dict.fromkeys(out))


def soft_overlap(a: list[frozenset], b: list[frozenset]) -> tuple[int, int]:
    """(words of a found in b, words of b found in a); two words match when their keys meet."""
    return (sum(1 for x in a if any(not x.isdisjoint(y) for y in b)),
            sum(1 for y in b if any(not y.isdisjoint(x) for x in a)))


def soft_jaccard(a: list[frozenset], b: list[frozenset]) -> float:
    if not a or not b:
        return 0.0
    shared = min(soft_overlap(a, b))
    return shared / (len(a) + len(b) - shared)


def content_stems(text: str) -> set[str]:
    """Stems of the informative words only (stopwords dropped); runs of unsegmented scripts contribute their
    content n-grams, so Chinese or Thai text compares by fragments rather than as one huge "word" (Chinese also by
    its one-character words, its pairs across a seam marked as such: see run_grams)."""
    out = set()
    analysis = analyze(text)
    chinese = is_chinese(analysis.text)
    for w in analysis.words:
        if UNSEGMENTED.search(w.text):
            out |= {g for g, weight in run_grams(w.text, chinese, False) if weight}
        elif w.identifier or (not w.stop and len(w.text) > 1):
            out.add(w.stem)
    return out


# Scripts written without spaces between words (Thai, Lao, Khmer, Myanmar, Chinese, Japanese) and Korean, whose
# particles attach to the word: the FTS5 unicode61 tokenizer indexes a whole run as one token, so these are matched
# by character n-grams against the text (bigrams for CJK and Hangul, trigrams for Southeast Asian scripts).
UNSEGMENTED = re.compile(r"[฀-໿က-႟ក-៿]+|[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]+")
_SEA = re.compile(r"[฀-໿က-႟ក-៿]")
# Chinese function characters (particles, pronouns, question words, auxiliaries). A gram made only of them (可以, 什么,
# 怎么, 我们) carries no topic; a gram with one of them carries half; others carry full weight.
CJK_FUNCTION = frozenset(_fold_text("的是了吗呢吧啊呀哦嗯么怎什哪谁几多少在有和与及或就都也还要会能可以把被给让对为从到这那你我他她它们个一不没很太该应些吗嘛"
                         "はがをにのでともへやかねよなだ"))  # + Japanese particles and copula
# Thai function words: a trigram inside them carries no topic, one touching them half (Thai letters, unlike Chinese
# characters, mean nothing alone, so Thai is weighted by words found in the run, longest first).
_TH_FUNCTION = sorted(set("""ของ ที่ และ ใน กับ ไม่ ได้ จะ แล้ว ไหม อะไร ยังไง อย่างไร คือ เป็น มี ให้ ครับ ค่ะ คะ นะ เลย ด้วย แบบ
    ไหน เท่าไหร่ กี่ ว่า ก็ หรือ แต่ จาก ถึง เรา ฉัน ผม คุณ เขา นี้ นั้น หน่อย ควร""".split()), key=len, reverse=True)
# Whole runs that are chat fillers, as "ok", "yes", "thanks" are in the English list: they recall nothing.
FILLER_RUNS = frozenset(_fold_text(w) for w in """はい うん ええ いいえ 了解 了解です りょうかい オーケー ありがとう ありがとうございます お願い お願いします
    よろしく そう そうです わかった わかりました 네 예 응 아니 아니요 좋아 좋아요 그래 그래요 알겠어 알겠어요 알겠습니다 오케이
    고마워 고마워요 감사 감사합니다 부탁해 부탁해요 ครับ ค่ะ คะ นะ จ้ะ จ้า โอเค ได้ ได้เลย ใช่ ขอบคุณ ขอบคุณครับ ขอบคุณค่ะ
    ขอบคุณมาก""".split())
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CJK_NEGATORS = frozenset(_fold_text("不没无别非未"))
_MAX_GRAMS = 32
# Bengali, Arabic-Indic, Persian and Devanagari digits: the FTS5 index keeps them as written, so a number is also
# searched in each of them ("2027" finds "২০২৭"); folding makes the comparison itself digit-agnostic.
_NATIVE_DIGITS = tuple(str.maketrans("0123456789", d) for d in ("০১২৩৪৫৬৭৮৯", "٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹", "०१२३४५६७८९"))
_LATIN_IN_CJK = re.compile(r"[A-Za-z][A-Za-z0-9._:/-]*[A-Za-z0-9]|[A-Za-z0-9]*\d[A-Za-z0-9._:/-]*")

# -- Japanese spelling variants --------------------------------------------------------------------------------------
# Japanese writes one word in several ways: kanji or kana (子供 / 子ども, 一緒 / いっしょ), with more or less okurigana
# (引越し / 引越 / 引っ越し, 申込 / 申し込み), katakana loanwords with or without the long vowel or small kana
# (コンピュータ / コンピューター, ウイルス / ウィルス). Character n-grams cannot join these. SudachiDict's synonym source lists
# them as spellings of one lexeme; ja_variants.tsv.gz keeps those spellings only, never synonyms (tools/
# build_ja_variants.py). In text that has kana, each variant is replaced by its word's representative spelling, longest
# first, left to right; a representative spelled as it is stays (so コンピュータ in コンピューター is not touched). There
# are no spaces between Japanese words, so a variant is replaced only where its edges can be word edges:
# - a variant that starts with hiragana not after a kanji (okurigana: "白い草" is not い草) or after other hiragana,
#   unless that is a particle after a kanji or katakana word ("子供はきょう"; "べんきょう" holds no きょう, "かいちょう"
#   no いちょう); one written in hiragana only not before hiragana other than a particle or a suffix (たち, さん,
#   だけ …: "きょうだい" holds no きょう). A variant with a kanji in it may run into hiragana: that is its inflection
#   or a suffix ("引越しします", "子どもたち");
# - a katakana edge not against more katakana ("トラック" holds no トラ, "エラー" no エラ); a variant of six or more
#   letters may sit inside a longer katakana word (セキュリティポリシー), but not before ー or a small kana.
# Kanji edges are free: 申込書 is 申し込み書. The text is folded again until nothing changes (子どもたち -> 子供たち ->
# 子供達), so folding folded text changes nothing. Variants inside the filler and greeting lists are left out, so those
# still compare as written. Hiragana variants of two letters (いす, かに, おば) are not in the table. Loaded on the first
# text with kana.
_JA_FILE = _DICTIONARIES / "ja_variants.tsv.gz"
_JA_TABLE: tuple | None = None
_JA_LOCK = threading.Lock()
_JA_PARTICLES = frozenset("のはがをにでともへやかねよ")  # hiragana that may stand before or after a word
_JA_AFTER = (*_JA_PARTICLES, "たち", "さん", "さま", "ちゃん", "くん", "だけ", "まで", "より", "など", "って", "だった")
_JA_KATA_CONT = frozenset("ーァィゥェォッャュョヮヵヶヽヾ")  # katakana that cannot begin a word
_JA_EMBED = 6  # katakana variants this long may sit inside a longer katakana word
_JA_PASSES = 4  # the fold runs again on its own output, until nothing changes or this many times


def _ja_kind(c: str) -> str:
    """'h' hiragana, 'k' katakana (with ー, not ・), 'c' kanji, '' anything else."""
    if "ぁ" <= c <= "ゟ":
        return "h"
    if "ァ" <= c <= "ヿ" and c != "・" or "ㇰ" <= c <= "ㇿ":
        return "k"
    if "㐀" <= c <= "鿿" or "豈" <= c <= "﫿" or c in "々〆〇":
        return "c"
    return ""


def _ja_table() -> tuple:
    """({spelling: (representative, edges or None)}, {first character: lengths, longest first}), loaded once. A
    representative maps to itself with no edge rules: it is consumed as it stands, so no shorter variant matches in it.
    Spellings are folded as Japanese text is (kanji kept, see _fold_text); one that folds onto two words is dropped."""
    global _JA_TABLE
    if _JA_TABLE is None:
        with _JA_LOCK:
            if _JA_TABLE is None:
                pairs: dict[str, str] = {}
                try:
                    with gzip.open(_JA_FILE, "rt", encoding="utf-8") as fh:
                        for line in fh:
                            variant, tab, target = line.rstrip("\n").partition("\t")
                            if tab and not line.startswith("#"):
                                pairs[variant] = target
                except (OSError, EOFError, ValueError, zlib.error) as exc:  # missing or damaged data file: no variants
                    _unreadable(_JA_FILE, exc)
                    pairs = {}
                # what _fold_text does to Japanese text (no Traditional -> Simplified), without its loop
                base = {s: unicodedata.normalize("NFC", s).lower().translate(_FOLD) for s in {*pairs, *pairs.values()}}
                targets = {base[t] for t in pairs.values()}
                protected = "\0".join(w for w in (*FILLER_RUNS, *GREETINGS) if _KANA.search(w))
                folded: dict[str, set] = {}
                for variant, target in pairs.items():
                    v, t = base[variant], base[target]
                    if v != t and v not in targets and v not in protected:
                        folded.setdefault(v, set()).add(t)
                table: dict[str, tuple] = {t: (t, None) for t in targets}
                for v, ts in folded.items():
                    if len(ts) == 1:
                        kinds = [_ja_kind(c) for c in v]
                        last = "" if kinds[-1] == "h" and set(kinds) != {"h"} else kinds[-1]  # see above: kanji in it
                        table[v] = (next(iter(ts)), (kinds[0], last, len(v) >= _JA_EMBED))
                lengths: dict[str, set] = {}
                for s in table:
                    lengths.setdefault(s[0], set()).add(len(s))
                _JA_TABLE = (table, {c: tuple(sorted(ls, reverse=True)) for c, ls in lengths.items()})
    return _JA_TABLE


def _ja_edges_fit(text: str, start: int, end: int, edges: tuple) -> bool:
    """May text[start:end] be a word: its hiragana and katakana edges against the characters around it (see above)."""
    first, last, embed = edges
    before = _ja_kind(text[start - 1]) if start else ""
    after = _ja_kind(text[end]) if end < len(text) else ""
    if first == "h" and (before == "c" or before == "h" and not (
            text[start - 1] in _JA_PARTICLES and start >= 2 and _ja_kind(text[start - 2]) in ("c", "k"))):
        return False  # after hiragana only when that is a particle after a kanji or katakana word (今日はきょう-like)
    if first == "k" and before == "k" and not embed:
        return False
    if last == "h" and after == "h" and not text.startswith(_JA_AFTER, end):
        return False
    return not (last == "k" and after == "k" and (not embed or text[end] in _JA_KATA_CONT))


def _ja_fold(text: str) -> str:
    """Japanese spelling variants of folded *text* replaced by their representative spellings (see above), again
    until nothing changes."""
    table, lengths = _ja_table()
    for _ in range(_JA_PASSES if table else 0):
        out, i, done, n = [], 0, 0, len(text)
        while i < n:
            for size in lengths.get(text[i], ()):
                hit = table.get(text[i:i + size])
                if hit is not None and (hit[1] is None or _ja_edges_fit(text, i, i + size, hit[1])):
                    if hit[0] != text[i:i + size]:
                        out += (text[done:i], hit[0])
                        done = i + size
                    i += size
                    break
            else:
                i += 1
        if not out:
            break
        text = "".join(out) + text[done:]
    return text


# -- Chinese words ---------------------------------------------------------------------------------------------------
# Chinese has no spaces, and character pairs cannot tell a word from the halves of two words: 和服 (kimono) is in
# 我和服务员 ("I and the waiter"), 研究生 (graduate student) in 研究生命 ("to research life"). Text whose Han is Chinese
# (no kana, no Hangul) is also split into words with jieba's dictionary (zh_words.txt.gz, tools/build_zh_words.py):
# the most probable split by word frequency, as jieba cuts without its HMM; a character that begins no word stands
# alone. The words add to the n-grams, they do not replace them (a name the dictionary lacks is still found by its
# pairs, a partial overlap still counts):
# - a seam is where two dictionary words of two or more characters meet, or one meets a function character standing
#   alone (和 | 服务员). A pair across a seam is read as such (和 SEAM 服): a Chinese query's pair inside a word is not
#   found in a fact that holds it only across a seam (研究 | 生命 holds no 究生, 学校 | 长期 no 校长), and its
#   pair across a seam only meets the same seam (几岁 | 了 meets 三岁 | 了). Every pair keeps its weight, so a query's
#   coverage is counted as before. Next to a content character standing alone a split is not sure enough (看 | 牙医
#   still holds 看牙, 续 | 保险 still 续保): no seam there;
# - a one-character word is also searched alone, wherever it stands in a fact (猫 in 猫粮, 牙 in 牙医), unless it is a
#   function character, a negator, a numeral or a unit of the calendar, one of the commonest words (过, 看, 去), or a
#   character jieba tags as a person name (王, 李, 陈: most surnames). In a query, two such characters side by side are
#   more likely a name the dictionary lacks (怡 君, 思 琪): their pair stands for them, as before.
# Folded text is split (Traditional -> Simplified first; the table is folded the same way). Loaded on the first
# Chinese text: about 0.2 s and 30 MB. Without the data file, Chinese is compared as before (pairs only).
class _ZhTable(NamedTuple):
    words: dict  # word -> log probability (its frequency over the total)
    longest: dict  # first character -> length of the longest word it begins
    floor: float  # log probability of a character that begins no word
    log_total: float
    names: frozenset  # single characters jieba tags as a person name


_ZH_WORDS = True
_ZH_FILE = _DICTIONARIES / "zh_words.txt.gz"
_ZH_TABLE: _ZhTable | None = None
_ZH_LOCK = threading.Lock()
_ZH_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"  # round(4 * ln f), one character
_ZH_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_HANGUL = re.compile(r"[\uac00-\ud7af]")
_ZH_NOT_ALONE = frozenset(_fold_text("零〇一二三四五六七八九十百千万亿两半几年月日号天周点分秒"))
_ZH_COMMON = 43  # round(4 * ln f) from which a one-character word is too common to search alone (f >= about 45,000)
_ZH_CHAR_WEIGHT = 1.0  # a one-character word weighs as a pair of content characters
SEAM = "\x1f"  # between the two characters of a pair read across a seam: "和" SEAM "服" is not 和服


def _zh_table() -> _ZhTable:
    """The dictionary, loaded once (tools/build_zh_words.py: "=" ln of the total, "!" the name characters, then a line
    per first character: "研 Q 究g 究生U"); empty without the data file."""
    global _ZH_TABLE
    if _ZH_TABLE is None:
        with _ZH_LOCK:
            if _ZH_TABLE is None:
                words: dict[str, float] = {}
                longest: dict[str, int] = {}
                total, logp, names = 0.0, {}, frozenset()
                try:
                    with gzip.open(_ZH_FILE, "rb") as fh:
                        data = fh.read().decode("utf-8")
                    for line in io.StringIO(data):
                        line = line.rstrip("\n")
                        if not line or line[0] == "#":
                            continue
                        if line[0] == "=":
                            total = float(line[1:])
                            logp = {d: i / 4 - total for i, d in enumerate(_ZH_DIGITS)}  # one float per step, shared
                        elif line[0] == "!":
                            names = frozenset(line[1:])
                        else:
                            first, *rests = line.split(" ")
                            for rest in rests:
                                words[first + rest[:-1]] = logp[rest[-1]]
                            longest[first] = max(map(len, rests))
                except (OSError, EOFError, ValueError, KeyError, IndexError, UnicodeDecodeError, zlib.error) as exc:
                    _unreadable(_ZH_FILE, exc)
                    words, longest, total, names = {}, {}, 0.0, frozenset()  # no data, or damaged: no words
                _ZH_TABLE = _ZhTable(words, longest, -total, total, names)
    return _ZH_TABLE


@lru_cache(maxsize=4096)
def zh_words(run: str) -> tuple[str, ...]:
    """A folded run of Chinese characters split into words: the split whose words are most probable together (jieba's
    word graph and route, without its HMM); a character that begins no word of the dictionary stands alone."""
    table = _zh_table()
    words, longest = table.words, table.longest
    if not words:
        return tuple(run)
    n = len(run)
    score, end = [0.0] * (n + 1), list(range(1, n + 2))
    for i in range(n - 1, -1, -1):
        alone = words.get(run[i])
        best = None if alone is None else alone + score[i + 1]
        for size in range(2, min(longest.get(run[i], 1), n - i) + 1):
            p = words.get(run[i:i + size])
            if p is not None and (best is None or p + score[i + size] >= best):  # a tie: the longer word, as jieba
                best, end[i] = p + score[i + size], i + size
        score[i] = table.floor + score[i + 1] if best is None else best
    out, i = [], 0
    while i < n:
        out.append(run[i:end[i]])
        i = end[i]
    return tuple(out)


def _zh_seams(tokens: tuple) -> frozenset:
    """Offsets in the run where a pair of characters is read across a seam (see above)."""
    seams, pos = set(), 0
    for a, b in zip(tokens, tokens[1:]):
        pos += len(a)
        if (len(a) > 1 or len(b) > 1) and (len(a) > 1 or a in CJK_FUNCTION) and (len(b) > 1 or b in CJK_FUNCTION):
            seams.add(pos)
    return frozenset(seams)


def _zh_alone(token: str) -> int:
    """2: a one-character word searched on its own (猫, 车, 药; see above); 1: a character of a name (王, 轩), which
    a one-character word next to it may belong to; 0: neither."""
    if len(token) != 1 or token in CJK_FUNCTION or token in CJK_NEGATORS or token in _ZH_NOT_ALONE:
        return 0
    table = _zh_table()
    if token in table.names:
        return 1
    p = table.words.get(token)
    return 2 if p is not None and p + table.log_total < (_ZH_COMMON - 0.5) / 4 else 0


def is_chinese(text: str) -> bool:
    """Text whose Han characters are Chinese: it has some, and no kana (Japanese) or Hangul (Korean)."""
    return _ZH_WORDS and _ZH_HAN.search(text) is not None and _KANA.search(text) is None and \
        _HANGUL.search(text) is None


# Chinese function words, as words of their own: pronouns, particles, aspect and modal markers, a few adverbs and
# prepositions (0.16.0). Words, not characters: 别克 (Buick), 网吧, 了解 and 他山之石 hold these characters inside one
# word of the dictionary. Names hold one or two of them as words too — 有赞, 去哪儿, 和记黄埔, 平凡的世界, 我和我的祖国,
# 这个杀手不太冷 — so a quoted span is a phrase someone said only with three or more different ones besides 的
# ("先去搜索再告诉我答案": 先 去 再 我), or when its last word is a sentence-final particle ("知道了", "明天再说吧").
ZH_PHRASE_WORDS = frozenset("""
    我 你 您 他 她 它 我们 你们 他们 她们 咱们 的 了 着 过 吗 呢 吧 啊 呀 把 被 让 给 请 别 就 都 也 再 还 又 很 太 不 没
    没有 要 会 能 可以 这 那 哪 这个 那个 什么 怎么 为什么 先 到 去 来 在 是 有 和 跟 向 从 对""".split())
ZH_FINAL_PARTICLES = frozenset("了 吗 呢 吧 啊 呀 啦 哦 嘛 呗".split())  # 么 is not one: 饿了么, 什么
_ZH_PHRASE_MIN_WORDS = 3


def zh_phrase(span: str) -> bool:
    """A Chinese span that reads as something said, not a name (see ZH_PHRASE_WORDS). False for text that is not
    Chinese, and without the word list (zh_words then gives single characters, and the words would turn into a list of
    characters)."""
    if not is_chinese(span) or not _zh_table().words:
        return False
    words = [w for m in _ZH_HAN.finditer(fold(span)) for w in zh_words(m.group())]
    if not words:
        return False
    return words[-1] in ZH_FINAL_PARTICLES or len(
        {w for w in words if w in ZH_PHRASE_WORDS} - {"的"}) >= _ZH_PHRASE_MIN_WORDS



# A count in Chinese (0.16.2, issue #4): a number in Arabic digits followed only by units and measure words and the
# words that qualify a count — "3个月", "15分钟", "2天后", "100多人", "5号楼", "1 小时" — is no name, as "14 days" is none.
# Anything else that begins with a number is one: 360安全卫士, 115网盘, 36氪, 1905电影网, 101次求婚 (次 is a unit, 求婚
# is not). Measured on 26,308 Chinese labels of films, series, games, books, companies, brands, apps and sites in
# Wikidata: 5 of them read as a count (100号, 21克拉, 361度, 76天, 15分鐘), and every one of 176 counts did. Arabic
# digits only: with Chinese numerals the same rule took 五月天 (a band), 三人行 and 半次元 for counts.
ZH_UNITS = frozenset("""
    个 元 块 角 分 次 岁 克 度 米 斤 两 吨 升 秒 年 月 日 号 天 周 点 页 本 张 条 位 名 人 只 件 台 辆 层 楼 期 届 集 章 节 部
    首 场 遍 倍 成 折 帧 行 列 字 篇 份 套 组 项 款 种 家 间 座 户 头 匹 根 片 瓶 杯 碗 盒 包 箱 袋 分钟 小时 公里 千米 厘米 毫米
    公斤 千克 毫升 公升 星期 礼拜 个月 周年 克拉 英寸 英尺 英里 美元 港元 欧元 日元 块钱 毛钱 平米 平方米 毫秒 微秒 年代 世纪
    季度 天后 年前""".split())
ZH_COUNT_WORDS = frozenset("前 后 以上 以下 左右 内 之内 以内 以后 以前 多 余 来 半 整 左 右 后的 前的 里 间 中 起".split())
_ZH_COUNT_PIECES = sorted(ZH_UNITS | ZH_COUNT_WORDS, key=len, reverse=True)  # the longest first: 分钟 before 分
_LEADING_NUMBER = re.compile(r"[0-9]+(?:[.,:/][0-9]+)*")
_ZH_MULTIPLIERS = frozenset("十百千万亿")


def zh_count(span: str) -> bool:
    """A Chinese span that is a count, not a name (see ZH_UNITS): one or more parts of a number in Arabic digits, the
    multipliers 十 百 千 万 亿 after it ("3万年", "1亿人"), then units and the words that qualify a count, each part
    with a unit ("2小时30分钟", review 0.16.2) unless a multiplier joins it to the next ("1万2千人"), or the number alone
    ("1000万"). Read in NFKC ("３个月") and folded (15分鐘), spaces left out ("1 小时")."""
    span = unicodedata.normalize("NFKC", span or "")
    if not is_chinese(span):
        return False
    s = fold("".join(span.split()))
    i = 0
    while i < len(s):
        m = _LEADING_NUMBER.match(s, i)
        if m is None:
            return False
        i = start = m.end()
        while i < len(s) and s[i] in _ZH_MULTIPLIERS:
            i += 1
        if i == len(s) and i > start:  # a number alone: "1000万", "3亿"
            return True
        if i > start and s[i].isdigit():  # the number goes on: "1万2千人" (review 0.16.2)
            continue
        unit = False
        while i < len(s) and not s[i].isdigit():
            piece = next((p for p in _ZH_COUNT_PIECES if s.startswith(p, i)), None)
            if piece is None:
                return False
            unit = unit or piece in ZH_UNITS
            i += len(piece)
        if not unit:
            return False
    return True



# How a Japanese or Korean sentence ends (0.16.1): polite and plain predicates, several kana or syllables long, so a
# name that ends in one of their letters (ゆか, あやね, 캐나다, 혼다) is not taken for a sentence.
JA_PREDICATE_ENDINGS = ("です", "ます", "ません", "ました", "でした", "ましょう", "でしょう", "ください", "だった",
                        "ない", "なかった", "たい", "かもしれない")
KO_PREDICATE_ENDINGS = ("니다", "세요", "어요", "아요", "해요", "에요", "예요", "까요", "네요", "군요")


def ja_ko_phrase(span: str) -> bool:
    """A quoted Japanese or Korean span that ends as a sentence does ("私は学生です", "明日は雨が降るかもしれません",
    "좋아요"), not as a name."""
    words = span.strip().rstrip("。.!?！？、,").split()
    last = words[-1] if words else ""
    if _KANA.search(span) and last.endswith(JA_PREDICATE_ENDINGS):
        return True
    return _HANGUL.search(span) is not None and last.endswith(KO_PREDICATE_ENDINGS)

def zh_word_cuts(folded: str) -> frozenset:
    """Offsets in *folded* text where a Chinese word ends and the next begins, as zh_words splits each run of Han
    characters, the edges of every run included (0.16.1): a known name is found in Chinese text only from cut to cut,
    so 和服 is not in 我和服务员 (和 | 服务员) while 小米科技 is in 我在小米科技工作 (小米 | 科技)."""
    cuts = set()
    for m in _ZH_HAN.finditer(folded):
        pos = m.start()
        cuts.add(pos)
        for word in zh_words(m.group()):
            pos += len(word)
            cuts.add(pos)
    return frozenset(cuts)

def _zh_run_grams(run: str, query: bool) -> list[tuple[str, float]]:
    """run_grams of Chinese text with its words: a pair across a seam is written with SEAM between its characters, a
    one-character word is a gram of its own. *run* may hold other characters too (a word of analyze: "10月23日（周五）
    下午"); its runs of Han characters are split. *query*: two one-character words side by side are not searched
    alone (a name the dictionary lacks)."""
    seams, alone = set(), []
    for m in _ZH_HAN.finditer(run):
        tokens = zh_words(m.group())
        seams |= {m.start() + k for k in _zh_seams(tokens)}
        kind = [0, *map(_zh_alone, tokens), 0]
        alone += [t for k, t in enumerate(tokens, 1) if kind[k] == 2 and not (query and (kind[k - 1] or kind[k + 1]))]
    grams = [(g[0] + SEAM + g[1] if len(run) > 2 and k + 1 in seams else g, w)
             for k, (g, w) in enumerate(_plain_grams(run))]
    return grams + [(t, _ZH_CHAR_WEIGHT) for t in dict.fromkeys(alone)]


def zh_seamed(text: str) -> str:
    """Folded Chinese *text* with SEAM at each seam between its words ("研究" SEAM "生命科学"), so that a pair is found in
    it only as it stands: 究生 inside a word, 究 SEAM 生 across a seam. "" for text that is not Chinese."""
    if not is_chinese(text):
        return ""
    out, done = [], 0
    for m in _ZH_HAN.finditer(text):
        if len(m.group()) > 2:
            for k in sorted(_zh_seams(zh_words(m.group()))):
                out += (text[done:m.start() + k], SEAM)
                done = m.start() + k
    return "".join(out) + text[done:]


def _thai_mask(run: str) -> list[bool]:
    """Which characters of a Thai run belong to a function word (greedy, longest word first)."""
    mask, i = [False] * len(run), 0
    while i < len(run):
        word = next((w for w in _TH_FUNCTION if run.startswith(w, i)), "")
        for k in range(i, i + len(word)):
            mask[k] = True
        i += len(word) or 1
    return mask


def is_sea(run: str) -> bool:
    """Thai, Lao, Khmer or Myanmar: scripts whose characters are letters, compared by trigrams."""
    return _SEA.search(run) is not None


def run_grams(run: str, chinese: bool = False, query: bool = True) -> list[tuple[str, float]]:
    """(n-gram, weight) of one unsegmented run; a run no longer than n is one gram; a filler run has none. *chinese*:
    the run is of a Chinese text (is_chinese), so its words are used too: a pair across a seam is written with SEAM,
    a one-character word is a gram of its own (see "Chinese words"); *query*: the grams are a query's."""
    if run in FILLER_RUNS:
        return []
    if chinese and _ZH_HAN.search(run) and not is_sea(run):
        return _zh_run_grams(run, query)
    return _plain_grams(run)


def _plain_grams(run: str) -> list[tuple[str, float]]:
    """run_grams without words: character pairs (trigrams for Thai), as for every script but Chinese."""
    sea = is_sea(run)
    n = 3 if sea else 2
    function = _thai_mask(run) if sea else [c in CJK_FUNCTION for c in run]
    spans = [(0, len(run))] if len(run) <= n else [(i, i + n) for i in range(len(run) - n + 1)]
    out = []
    for i, j in spans:
        hits = sum(function[i:j])
        out.append((run[i:j], 0.0 if hits == j - i else 0.5 if hits else 1.0))
    return out


def query_grams(text: str) -> list[tuple[str, float]]:
    """Weighted substrings to look for when *text* contains unsegmented script: its content n-grams, plus the Latin
    words and identifiers written into it ("HR24", "hr-vps", "Python"), which the tokenizer glues to the
    neighbouring characters. Lower-cased; weight-0 grams dropped; [] for text without unsegmented script."""
    lowered = _fold_text((text or "").lower())
    if not UNSEGMENTED.search(lowered):
        return []
    grams: dict[str, float] = {}
    chinese = is_chinese(lowered)
    for run in UNSEGMENTED.findall(lowered):
        for g, weight in run_grams(run, chinese):
            if weight:
                grams[g] = max(grams.get(g, 0.0), weight)
    for m in _LATIN_IN_CJK.finditer(UNSEGMENTED.sub(" ", lowered)):
        token = m.group(0).strip("._:/-")
        if len(token) >= 3 and token not in STOPWORDS:
            grams[token] = 1.0
    return list(grams.items())[:_MAX_GRAMS]


def content_chars(text: str) -> list[str]:
    """Distinct Chinese characters (also in Japanese text) that carry meaning: no function characters or negators.
    Kana, Hangul and Thai are left out: their characters are sounds, not words."""
    chars = []
    for run in UNSEGMENTED.findall(_fold_text((text or "").lower())):
        if run not in FILLER_RUNS:
            chars += [c for c in run if _HAN.match(c) and c not in CJK_FUNCTION and c not in CJK_NEGATORS]
    return list(dict.fromkeys(chars))


def gram_coverage(grams: list[tuple[str, float]], text, words: bool = False, total: float | None = None) -> float:
    """Weighted share of the grams present in *text* (a str: case-insensitive, folded as the grams are; or its
    Analysis). *words*: the grams are a Chinese query's (is_chinese), so a Chinese text holds a pair only as it
    stands: 究生 inside a word, 究 SEAM 生 across a seam between two (研究 | 生命 holds no 究生, 我和服务员 no 和服).
    Otherwise, and in text that is not Chinese, a pair is a substring. *total*: the grams' weight, if known."""
    total = sum(w for _, w in grams) if total is None else total
    if not total:
        return 0.0
    if not isinstance(text, Analysis):
        folded = _fold_text((text or "").lower())
        text = Analysis((), frozenset(), frozenset(), folded, (), seamed=zh_seamed(folded) if words else "")
    if words and text.seamed:
        return sum(w for g, w in grams if g in text.seamed) / total
    return sum(w for g, w in grams if (g.replace(SEAM, "") if SEAM in g else g) in text.text) / total


def zh_char_text(grams: list[tuple[str, float]], analysis: Analysis) -> str:
    """The folded text a Chinese query's characters are looked for in, one by one (see content_chars): without a
    character it holds only as one half of a pair of the query (*grams*) split there by a seam, and not inside a pair
    of the query it holds as it stands. 研究 | 生命 holds the 生 of 研究生 only so (its 究 also in 研究, which stays);
    驾照 still holds 驾 and 照 of 驾驶执照 (no pair of it is split). The text itself when it is not Chinese."""
    seamed = analysis.seamed
    splits = [g[0] + SEAM + g[1] for g, _ in grams if len(g) == 2]
    if not seamed or not any(g in seamed for g in splits):
        return analysis.text
    holds = [g for g, _ in grams if len(g) == 2 or (len(g) == 3 and g[1] == SEAM)]  # as they stand, seams included
    split, held = set(), set()
    for patterns, marks in ((splits, split), (holds, held)):
        for pattern in patterns:
            start = seamed.find(pattern)
            while start >= 0:
                marks.update((start, start + len(pattern) - 1))
                start = seamed.find(pattern, start + 1)
    return "".join(" " if k in split and k not in held else c for k, c in enumerate(seamed) if c != SEAM)


# FTS5 syntax characters (quotes, grouping, prefix, column filter, NOT/required). Never interpolated into a MATCH
# expression: inside a token they split it into a phrase of the tokens unicode61 indexes ("pve-01" -> "pve 01").
FTS_OPERATORS = str.maketrans({c: " " for c in '"()*^:-+'})


@dataclass(frozen=True)
class Term:
    """One informative query term: the folded word, its primary stem, whether it is an identifier, the FTS-safe
    tokens it is searched as (operator characters removed), its comparison keys, and whether it is matched by
    scanning the facts' text instead of through the FTS5 index."""
    word: str
    stem: str
    identifier: bool
    tokens: tuple[str, ...] = ()
    keys: frozenset = frozenset()
    scan: bool = False
    fts_ok: bool = True
    roots: frozenset = frozenset()  # same-root keys, matched at _DERIV_WEIGHT
    root_lang: str = ""  # the language its roots (and the fact's) are compared in; "": no root matching
    head: frozenset = frozenset()  # a German compound: keys of its last part (Word.head)
    mod_parts: tuple = ()  # ... keys of each part before it
    lemmas: frozenset = frozenset()  # German / Spanish: stems of its lemmas (Word.lemmas)
    lemma_of: tuple = ()  # ... the lemmas, whose forms its search looks for too
    form_lang: str = ""  # ... the language they are compared in

    @cached_property
    def _lemma_keys(self) -> frozenset:
        return frozenset((self.form_lang, k) for k in self.keys | self.lemmas) if self.form_lang else frozenset()

    @cached_property
    def lemma_forms(self) -> tuple:
        """The other forms of its lemmas in the table ("va": fue, iba, vamos …), searched as words."""
        if not self.lemma_of:
            return ()
        lemmas = _forms(self.form_lang)[1]
        forms = dict.fromkeys(f for lemma in self.lemma_of for f in (lemma, *lemmas.get(lemma, ())))
        return tuple(f for f in forms if f != self.word and f.translate(FTS_OPERATORS) == f)[:64]

    def compound_match(self, fact: Analysis) -> float:
        """A German compound of the question in a fact that writes its parts apart ("Zahnarzttermin": "Termin beim
        Zahnarzt"): 1.0 with the last part and every part before it, _DERIV_WEIGHT with only some of them."""
        pool = fact.keys | fact.heads
        head = not self.head.isdisjoint(pool)
        found = [not keys.isdisjoint(pool) or not keys.isdisjoint(fact.mods) for keys in self.mod_parts]
        if head and all(found):
            return 1.0
        return _DE_QUESTION_HEAD if head else _DERIV_WEIGHT if any(found) else 0.0

    @property
    def loose(self) -> str:
        """A shorter shared beginning for long Russian words, so derivations Snowball keeps apart still meet
        ("переезжаем" / "переезд" share "переез"); "" for other words (in English "contra" would join "contract" and
        "contrast"), short words and identifiers."""
        if self.identifier or len(self.tokens) != 1 or len(self.stem) < _LOOSE_FROM or word_script(self.word) != "cyrillic":
            return ""
        return self.stem[:max(_LOOSE_MIN, len(self.stem) - 2)]

    @property
    def short_loose(self) -> str:
        """The same for Russian stems of six letters, one letter shorter ("покупа" / "покупк" share "покуп",
        "работа" / "рабоч" share "рабо"); matched at _DERIV_WEIGHT, since a four-letter beginning says less."""
        if not _SHORT_LOOSE or self.identifier or len(self.tokens) != 1 or not _SHORT_FROM <= len(self.stem) < _LOOSE_FROM \
                or self.root_lang != "ru":
            return ""
        return self.stem[:-1]

    def fts(self, operators: dict | None = None) -> str:
        """FTS5 expression built only from operator-free tokens: the exact word (a phrase when it splits into
        several tokens), OR each of its keys as a prefix when the word is one plain token."""
        tokens = self.tokens or tuple(self.word.translate(FTS_OPERATORS).split())
        exact = '"' + " ".join(tokens) + '"'
        if self.identifier or len(tokens) != 1 or tokens[0] != self.word:
            if not any("0" <= c <= "9" for t in tokens for c in t):
                return exact
            native = ['"' + " ".join(t.translate(d) for t in tokens) + '"' for d in _NATIVE_DIGITS]
            return "(" + " OR ".join([exact, *native]) + ")"
        prefixes = sorted(k for k in (self.keys or {self.stem}) | ({self.loose} if self.loose else set())
                          if len(k) >= _MIN_STEM and k.translate(FTS_OPERATORS) == k and k.split() == [k])
        parts = [f'"{p}"*' for p in prefixes] + ([] if self.word in prefixes else [exact])
        # a borrowed word's other spellings (_loan_table): its short forms as whole words ("бота", not every "ботаника")
        parts += sorted({f'"{k[5:]}"' + ("*" if k[3] == "*" else "") for k in self.keys
                         if k[:5] in ("fts=:", "fts*:") and f'"{k[5:]}"*' not in parts})
        parts += [f'"{f}"' for f in self.lemma_forms if not any(f.startswith(p) for p in prefixes)]
        return parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"

    def fts_extra(self) -> str:
        """FTS5 expression of the root keys and the short beginning only, "" when there are none. A separate query:
        OR-ed into fts() it would score a fact that has the word itself twice (the root prefix matches it too) and
        push it above facts that answer better; here it only brings in facts the word's own forms miss."""
        if self.identifier or len(self.tokens) != 1 or self.tokens[0] != self.word:
            return ""
        prefixes = sorted(k for k in self.roots | ({self.short_loose} if self.short_loose else set())
                          if len(k) >= _MIN_STEM and k.translate(FTS_OPERATORS) == k and k.split() == [k])
        parts = [f'"{p}"*' for p in prefixes]
        return "" if not parts else parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"

    @cached_property
    def _bounded(self) -> re.Pattern:
        # An ASCII identifier is bounded by ASCII word characters only: in Chinese text "hr-vps" is glued to the
        # characters around it ("服务器hr-vps运行").
        edge = r"A-Za-z0-9_\-" if self.word.isascii() else r"\w\-"
        return re.compile(rf"(?<![{edge}])" + re.escape(self.word) + rf"(?![{edge}])")

    @cached_property
    def _prefix_keys(self) -> tuple:
        """Latin: keys of six letters or more that a word of the fact may extend ("allergi" reaches "allergic", as
        the FTS prefix query does); a whole stem, so "contract" does not reach "contrast"."""
        if self.identifier or len(self.tokens) != 1 or word_script(self.word) != "latin":
            return ()
        return tuple(k for k in self.keys if len(k) >= 6)

    @cached_property
    def _head_keys(self) -> tuple:
        """Latin: keys of four letters or more that may be the last part of a German or Dutch compound ("katze" in
        "familienkatze", "laptop" in "arbeitslaptop", "arts" in "kinderarts"). Checked only against facts written in
        German or Dutch."""
        if self.identifier or len(self.tokens) != 1 or word_script(self.word) != "latin":
            return ()
        return tuple(k for k in self.keys if len(k) >= 4)

    def found_in(self, fact: Analysis) -> bool:
        """Does a fact contain this term, exactly or through a word of the same root (see match)."""
        return self.match(fact) > 0.0

    def match(self, fact: Analysis) -> float:
        """How a fact contains this term: 1.0 for the word itself, a shared key, the loose beginning or the same
        identifier; _DERIV_WEIGHT for a word of the same root ("ночные" / "ночь", "teacher" / "teaches") or the
        short beginning of a middle-length Russian stem; 0.0 for none."""
        if self._exact(fact):
            return 1.0
        if self.identifier or len(self.tokens) != 1:
            return 0.0
        if fact.mods and not self.keys.isdisjoint(fact.mods):  # a part before the last of a German compound
            return _DERIV_WEIGHT
        if fact.heads and not self.keys.isdisjoint(fact.heads):  # the last part, of three letters ("Tag" / "Alltag")
            return _DERIV_WEIGHT
        if self._lemma_keys and fact.lemma_keys and not self._lemma_keys.isdisjoint(fact.lemma_keys):
            # a form of the same German or Spanish verb ("ging" / "gehen", "fue" / "va"): like a root, since a form
            # may belong to several verbs ("fue": ir, ser) and a name may be spelled like one ("Lucía" / lucir)
            return _DERIV_WEIGHT
        if self.head:  # a German compound of the question whose parts the fact has only some of
            strength = self.compound_match(fact)
            if strength:
                return strength
        lang = self.root_lang
        if lang and fact.roots | fact.lang_keys:  # only roots and keys of the same, determined language meet
            fact_roots = {r for lng, r in fact.roots if lng == lang}
            if (self.roots and not self.roots.isdisjoint(k for lng, k in fact.lang_keys if lng == lang)) \
                    or (fact_roots and not (self.keys | self.roots).isdisjoint(fact_roots)):
                return _DERIV_WEIGHT
        short = self.short_loose  # Russian only, so against the fact's Russian words only (not "покупки" in Ukrainian)
        if short and short in fact.text and any(lng == "ru" and k.startswith(short) for lng, k in fact.lang_keys):
            return _DERIV_WEIGHT
        return 0.0

    def _exact(self, fact: Analysis) -> bool:
        """The same word, a shared key (word or stem), the loose beginning, or (for identifiers and multi-token
        words) the same token run with word boundaries, so "pve-01" does not match inside "pve-010"."""
        if self.word in fact.bag:
            return True
        if not self.identifier and len(self.tokens) <= 1 and not self.keys.isdisjoint(fact.keys):
            return True
        if self.loose and self.loose in fact.text and any(w.startswith(self.loose) for w in fact.bag):
            return True
        if self._prefix_keys and any(k in fact.text for k in self._prefix_keys) and \
                any(f.startswith(k) for k in self._prefix_keys for f in fact.keys):
            return True
        if fact.german and self._head_keys and any(k in fact.text for k in self._head_keys) and \
                any(len(f) >= len(k) + 3 and f.endswith(k) for k in self._head_keys
                    for f in (fact.bag if fact.suffix_words is None else fact.suffix_words)):
            return True
        # Dutch the same, also against the compound's stem: "sleutel" ends "reservesleutels" as "reservesleutel"
        if fact.dutch and self._head_keys and any(k in fact.text for k in self._head_keys) and \
                any(len(f) >= len(k) + 3 and f.endswith(k) for k in self._head_keys for f in fact.bag | fact.keys):
            return True
        if not self.identifier and len(self.tokens) <= 1:
            if fact.heads and any(len(k) >= 4 for k in self.keys & fact.heads):  # the last part of a compound
                return True
            if self.head and self.compound_match(fact) == 1.0:
                return True
        return self.word in fact.text and self._bounded.search(fact.text) is not None


def _term_key(w: Word) -> str | None:
    """The key by which a word of a query is one of its terms (Term.stem), or None for a word that is no term."""
    if w.stop or w.text in QUERY_ONLY_STOP or (not w.identifier and len(w.text) < 2):
        return None
    tokens = w.text.translate(FTS_OPERATORS).split()
    if not tokens:
        return None
    return w.stem if len(tokens) == 1 else w.text


def query_terms(text: str) -> list[Term]:
    """Distinct informative terms of a natural-language query, in order of appearance."""
    seen, terms = set(), []
    for w in analyze(text).words:
        key = _term_key(w)
        if key is None:
            continue
        tokens = tuple(w.text.translate(FTS_OPERATORS).split())
        single = len(tokens) == 1
        if key not in seen:
            seen.add(key)
            terms.append(Term(w.text, key, w.identifier, tokens, w.keys if single else frozenset({w.text}), w.scan, w.fts,
                              w.roots if single else frozenset(), w.root_lang if single else "",
                              *((w.head, w.mod_parts, w.lemmas, w.lemma_of, w.form_lang) if single else ())))
    return terms


# -- two questions in one message (prefetch_clause_coverage, an experiment, off by default) -------------------------
# «что Злате нужно взять с собой в четверг и какой чай она любит?» asks two things: each answer holds only its half of
# the words, and the coverage gate, which weighs a fact against the whole message, can leave it out. Only Russian and
# English questions are split; a question starts with one of these words, after a preposition or not.
_QUESTION_WORDS = """что чем чего чему кто кого кому кем где куда откуда когда как какой какая какое какие какого какому
    каким каком какую каких какими каков какова каковы который которая которое которые которого которой которому
    которым котором которую которых сколько почему зачем чей чья чьё чье чьи чьего чьей чьих
    what when where which who whom whose why how""".split()
_QUESTION_PREPOSITIONS = """во в на с со у для по из о об при за от до к ко про через после перед
    at on in for with to from by of about""".split()
_WH = frozenset(_QUESTION_WORDS)
_PREP = frozenset(_QUESTION_PREPOSITIONS)
# A question word that asks which one, or how many, of something: the word after it may be all it asks about, and the
# question's verb may stand in the next one («какой чай и какой кофе она любит?», "which tea and which coffee does she
# like?" are one question about two things). Its question stands alone only with a word of a clause of its own.
_DETERMINERS = frozenset("""какой какая какое какие какого какому каким каком какую каких какими каков какова каковы
    который которая которое которые которого которой которому которым котором которую которых чей чья чьё чье чьи
    чьего чьей чьих сколько what which whose how""".split())
# words of a clause: a personal pronoun, a predicative, a form of "be", an auxiliary or a modal verb
_CLAUSE_WORDS = frozenset("""я ты он она оно мы вы они меня мне тебя тебе его её ее ему ей нас нам вас вам их им него
    неё нее нему ней них ним нужно надо можно нельзя должен должна должно должны есть будет будут был была было были
    i you he she it we they me him us them do does did is are was were be been am should shall can could will would
    must may might have has had need needs""".split())
# One pass over the message: a word (its hyphenated parts together: «что-то» is no «что») or one mark. A regular
# expression that looks ahead over the spaces from every position takes quadratic time on a long run of them.
_SPLIT_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*|[?;,«»„“”\"]")
_QUOTES = {"«": "»", "„": "“", "“": "”", '"': '"'}  # an opening quote -> the one that closes it
_JOINS = frozenset({"и", "а", "and"})
# a person the other question names: «и какой чай она любит?»
PRONOUNS_3P = _folded("""он она оно они его её ее него неё нее ему ей нему ней им ими ним ними их них нём нем ею нею
    he she they him her them his hers their theirs""")


def _stands_alone(tokens: list, at: int, end: int) -> bool:
    """Do the words after the question word tokens[at], up to *end*, make a question of their own: at least one word,
    and, after a word that asks which one (_DETERMINERS), a word of a clause (a pronoun, an auxiliary, a predicative or,
    past the word it asks about, a Russian verb form)."""
    after = [t for t, _, _ in tokens[at + 1:end] if t[0].isalnum()]
    if not after:
        return False
    if tokens[at][0] not in _DETERMINERS:
        return True
    return any(w in _CLAUSE_WORDS for w in after) or any(
        len(w) >= 4 and _RU_VERB.search(w.replace("ё", "е")) for w in after[1:])


def two_questions(text: str) -> tuple[str, str] | None:
    """The two questions of a message that asks exactly two, or None. The second begins at a question word (after a
    preposition or not) that follows "?" or ";" (and an «и» / «а» / "and" or not), or «и» / «а» / «а также» / "and"
    (with a comma or not); a boundary inside quotes does not count (a quoted line is what is asked about). The first
    has a question word too, and each stands alone (_stands_alone). When in doubt — one question, three or more, «и»
    between two words of one question («какой чай и кофе», "which tea and which coffee does she like"), an unclosed
    quote — None: the message is not split. Linear in the length of the message."""
    text = (text or "").strip()
    tokens = [(m.group(0).lower(), m.start(), m.end()) for m in _SPLIT_TOKEN.finditer(text)]
    quoted, stack = [], []
    for t, _, _ in tokens:  # a token inside quotes (an unclosed quote runs to the end)
        quoted.append(bool(stack))
        if stack and t == _QUOTES[stack[-1]]:
            stack.pop()
        elif t in _QUOTES:
            stack.append(t)
    n = len(tokens)

    def wh_at(k: int) -> int:
        """The index of the question word a question starting at token k opens with, or -1."""
        if k < n and not quoted[k] and tokens[k][0] in _WH:
            return k
        if k + 1 < n and not quoted[k] and not quoted[k + 1] and tokens[k][0] in _PREP and tokens[k + 1][0] in _WH:
            return k + 1
        return -1

    joins, i = [], 0  # (where the first question ends, the token the second one starts at)
    while i < n:
        t = tokens[i][0]
        if not quoted[i] and t in ("?", ";"):
            k = i
            while k + 1 < n and not quoted[k + 1] and tokens[k + 1][0] in ("?", ";"):
                k += 1
            start = k + 1
            if start < n and not quoted[start] and tokens[start][0] in _JOINS:
                start += 1
            if wh_at(start) >= 0:
                joins.append((tokens[i][1], start))
                i = start  # past the «и» after the mark: it is this boundary, not another one
            else:
                i = k + 1
            continue
        if not quoted[i] and t in _JOINS and i > 0:
            start = i + 2 if t == "а" and i + 1 < n and tokens[i + 1][0] == "также" else i + 1
            if wh_at(start) >= 0:
                joins.append((tokens[i - 1][1] if tokens[i - 1][0] == "," else tokens[i][1], start))
        i += 1
    if len(joins) != 1:
        return None
    cut, start = joins[0]
    end_first = next(k for k in range(n) if tokens[k][1] >= cut)
    first_wh = next((k for k in range(end_first) if not quoted[k] and tokens[k][0] in _WH), -1)
    second_wh = wh_at(start)
    if first_wh < 0 or not _stands_alone(tokens, first_wh, end_first) or not _stands_alone(tokens, second_wh, n):
        return None
    first, second = text[:cut].strip(" ,;?"), text[tokens[start][1]:].strip()
    if not first or not second:
        return None
    return first, second


def question_parts(text: str) -> tuple[tuple[frozenset, bool, str], tuple[frozenset, bool, str]] | None:
    """For a message of two questions (two_questions): each question's term keys (Term.stem), whether it has a
    pronoun of the third person, and its text; the words read in the whole message, as the gate reads it (the language,
    the function words): the first question is the message's first len(analyze(first).words) words."""
    parts = two_questions(text)
    if parts is None:
        return None
    words = analyze(text).words
    n = len(analyze(parts[0]).words)

    def part(ws, said):
        return (frozenset(k for w in ws if (k := _term_key(w)) is not None), any(w.text in PRONOUNS_3P for w in ws),
                said)

    return part(words[:n], parts[0]), part(words[n:], parts[1])


_ja_ready = True  # every word list above is built: Japanese spelling variants may load from here on
