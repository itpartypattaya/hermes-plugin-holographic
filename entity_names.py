"""Entity names: what counts as one, the identity key two spellings of a name share (entity_key), and the
rules that guess names from a fact's text. Stateless; the entity half of MemoryStore (entities.py) applies them.
store.py re-exports the names its tests read from it."""

import bisect
import re
import unicodedata

from .screen import _fact_problem
from .text import UNSEGMENTED, _script, is_chinese, is_greeting, is_identifier, ja_ko_phrase, zh_count, zh_phrase

# Entity extraction, applied in order: capitalized multi-word phrases in any cased script ("John Doe", "Мария
# Иванова"), short quoted terms (straight, curly, guillemets; straight quotes only at word boundaries, so
# contractions like don't / it's never pair up), then both sides of "X aka Y".
# Combining marks are part of a word (0.16.1): \w leaves out the vowel signs of Indic and Thai script and Arabic
# harakat, so "श्री राम aka Ram" had no side. Ranges: combining diacritics, Arabic marks, the Indic blocks, Thai, Lao,
# Myanmar and Khmer signs.
_MARKS = "".join(f"{chr(a)}-{chr(b)}" for a, b in (
    (0x0300, 0x036F), (0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06ED), (0x0900, 0x0DFF),
    (0x0E31, 0x0E31), (0x0E34, 0x0E3A), (0x0E47, 0x0E4E), (0x0EB1, 0x0EB1), (0x0EB4, 0x0EBC), (0x0EC8, 0x0ECD),
    (0x102B, 0x103E), (0x17B4, 0x17D3)))
_RE_WORD = re.compile(r"(?:[^\W\d_]|[" + _MARKS + r"])+(?:-(?:[^\W\d_]|[" + _MARKS + r"])+)*")  # hyphenated: one word
_RE_TOKEN = re.compile(r"(?:\w|[" + _MARKS + r"])+(?:-(?:\w|[" + _MARKS + r"])+)*")  # aka sides: handles with digits
# Straight double quotes are found by _straight_quoted (0.16.1); single quotes keep \w beside them (don't, it's).
# Bounded by _QUOTED_MAX_CHARS (a longer span is no name anyway): an opening mark without a closing one is then
# read once, not to the end of the text from every opening mark (review 0.16.1: 5,000 "«" took 0.4 s).
_RE_QUOTED = tuple(re.compile(p.replace("+", "{1,60}")) for p in (
    r"(?<!\w)'([^'\n]+)'(?!\w)", r"«([^»\n]+)»", r"“([^”\n]+)”", r"„([^“”\n]+)[“”]",
    # 0.16.1: the corner and title marks of Chinese, Japanese and Korean (「华为」, 《平凡的世界》, 『東京大学』): in scripts
    # without case, quotes are the only way a name is found
    r"「([^」\n]+)」", r"『([^』\n]+)』", r"《([^》\n]+)》", r"〈([^〉\n]+)〉"))
# Spaces on one line (no line break of any kind: the str.splitlines boundaries) — also around "aka" (0.16.1)
_INLINE_SPACE_CHAR = "[^\\S\n\r\x0b\x0c\x1c\x1d\x1e\x85" + chr(0x2028) + chr(0x2029) + "]"
_INLINE_SPACE = _INLINE_SPACE_CHAR + "+"
# The word alone, with one inline space on each side as lookarounds: a leading run of spaces in the pattern made a
# text of 10,000 spaces quadratic (review 0.16.1); _aka_pairs checks the spaces around it.
_RE_AKA = re.compile(f"(?<={_INLINE_SPACE_CHAR})(?:aka|a\\.k\\.a\\.|also{_INLINE_SPACE}known{_INLINE_SPACE}as)"
                     f"(?={_INLINE_SPACE_CHAR})", re.IGNORECASE)
# A comma too (0.16.0): the legacy aliases column is comma-separated, see _legacy_aliases.
_RE_NOT_A_NAME = re.compile(r"[=/\\@<>{}\[\]|`,]|://|\b\d{1,3}(?:\.\d{1,3}){3}\b")
# A quoted span longer than this is a quotation ("she said: 'see you on Friday'"), not a name. Chinese and Japanese
# write a quotation without spaces: more Han characters than _QUOTED_MAX_HAN is one too (中国人民解放军总医院 has 10).
_QUOTED_MAX_WORDS, _QUOTED_MAX_CHARS, _QUOTED_MAX_HAN = 3, 60, 16
_RE_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_RE_VERSION = re.compile(r"[vV]?\d+(?:\.\d+)+")  # 0.17.0, v2.1: a release, not a name (a whole word; gpt-3.5-turbo stays)
_PHRASE_MAX_WORDS, _ENTITY_MAX_CHARS = 5, 60
_AKA_MAX, _AKA_WINDOW = 16, 4 * _ENTITY_MAX_CHARS  # "X aka Y" read per text, and how far from the "aka" (0.16.0)
# Capitalized only because they open a sentence: dropped from the front of a capitalized run ("So Apple").
_SENTENCE_STARTERS = frozenset("""
    a an and as at but by for from here if in it its my no not now of on or our so that the then there these this
    those to today tomorrow we when where yes yesterday you your
    а в во вот вчера да если завтра и или когда мой моя мои на наш наша наши не но однако по при сегодня так там
    тут у это этот эта эти""".split())


def fold(text):
    """Comparison key for names: Unicode casefold with whitespace collapsed. SQLite's lower()/LIKE fold ASCII only,
    so this is registered on the connection as hfold() for entity lookups."""
    return " ".join(text.casefold().split()) if isinstance(text, str) else text


def entity_key(name: str) -> str:
    """Identity key of an entity name: NFKC, casefold, hyphens/underscores as spaces, whitespace collapsed, edge
    punctuation stripped. "Hermes Agent", "hermes-agent" and "HERMES  AGENT" are one entity; "coop" and "co-op" are
    not merged into each other (a hyphen becomes a space, it is not deleted)."""
    text = " ".join(unicodedata.normalize("NFKC", name or "").casefold().replace("-", " ").replace("_", " ")
                    .replace("ё", "е").split())  # Семён is Семен (0.16.1)
    text = text.strip(_KEY_EDGES).rstrip(_KEY_EDGES + ".")
    # A leading dot before a letter or a digit is part of the name (0.16.1): ".NET" is not "net", whose key it had,
    # so the word "net" in "a fishing net" mentioned .NET
    return text if text[:1] == "." and text[1:2].isalnum() else text.lstrip(_KEY_EDGES + ".")


_KEY_EDGES = " ,;:!?'\"«»“”„‘’"  # ‘ ’ since 0.16.1


def _entity_key_0160(name: str) -> str:
    """entity_key as 0.16.0 and earlier computed it: the key a tombstone of then was written under, and that older
    rows hold until _rekey_entities moves them."""
    text = unicodedata.normalize("NFKC", name or "").casefold().replace("-", " ").replace("_", " ")
    return " ".join(text.split()).strip(" .,;:!?'\"«»“”„")


def _is_capitalized(word: str) -> bool:
    """'John', 'Мария', 'Jean-Luc': each hyphen part an upper-case letter followed by lower-case ones (so not
    'NASA', 'I' or 'iPhone')."""
    return all(len(part) > 1 and part[0].isupper() and part[1:].islower() for part in word.split("-"))


class _Piece:
    """A part of a regex match, with the match's start / end / group interface."""
    __slots__ = ("_start", "_end", "_text", "word")

    def __init__(self, start: int, end: int, text: str, word: tuple[int, int]):
        self._start, self._end, self._text = start, end, text
        self.word = word  # (start, end) of the whole match the piece came from

    def start(self) -> int:
        return self._start

    def end(self) -> int:
        return self._end

    def group(self, _=0) -> str:
        return self._text


def _script_pieces(text: str, pattern: re.Pattern) -> list[_Piece]:
    """The matches of *pattern* in *text*, each split where its letters change script or letter case (Latin, Cyrillic,
    Greek; Han, kana, Hangul, Thai...): "Hub直连可用" is two pieces, "Hub" and "直连可用", and "Hubе" two (0.16.1, issue #4:
    "Docker Hub直连可用" was taken as one name). Hyphens and digits go with the letters after them; two pieces of one word
    have no gap between them, so a run of names never crosses the seam."""
    out = []
    for m in pattern.finditer(text):
        word, base = m.group(0), m.start()
        start, key, last = 0, None, -1
        for i, c in enumerate(word):
            if not c.isalpha() or unicodedata.category(c) == "Lm":  # ʻ ʼ ー 々 hold a word together (0.16.1)
                continue
            cased = c.lower() != c.upper()
            k = (_script(c), cased, "" if cased else _uncased_block(c))
            if key is not None and k != key:
                out.append(_Piece(base + start, base + last + 1, word[start:last + 1], (base, m.end())))
                start = last + 1
            key, last = k, i
        out.append(_Piece(base + start, base + len(word), word[start:], (base, m.end())))
    return out


def _hugs(text: str, i: int) -> bool:
    """Whether text[i] is a digit or a letter with case: a straight double quote beside one is no quote (a 5" screen,
    an inch mark inside a word). A letter of a script without case may touch it: Chinese writes 用"Docker Hub"直连."""
    if not 0 <= i < len(text):
        return False
    c = text[i]
    return c.isdigit() or (c.isalpha() and c.lower() != c.upper())


def _straight_quoted(text: str) -> list[str]:
    """Spans between straight double quotes on one line, as a regex with lookarounds would pair them: an opening quote
    not right after a digit or a letter with case, the next quote closing it unless a digit or such a letter follows
    (0.16.1: the old word-character lookarounds missed 用"Docker Hub"直连, and character ranges missed full-width letters)."""
    return [text[i + 1:j - 1] for i, j in _straight_quote_spans(text)]


def _straight_quote_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each span _straight_quoted finds, its quotes included."""
    quotes = [i for i, c in enumerate(text) if c == '"']
    out, k = [], 0
    while k < len(quotes) - 1:
        i, j = quotes[k], quotes[k + 1]
        if not _hugs(text, i - 1) and i + 1 < j <= i + 1 + _QUOTED_MAX_CHARS and "\n" not in text[i + 1:j] \
                and not _hugs(text, j + 1):
            out.append((i, j + 1))
            k += 2
        else:
            k += 1
    return out


# Quotes that hold what someone said, a label or a word in passing as often as a name — the title marks 《》〈〉 hold
# titles and are not among them (0.16.2). An opening mark and its closing ones.
_PLAIN_QUOTE_PAIRS = {"«": "»", "“": "”", "„": "“”", "「": "」", "『": "』", '"': '"', "'": "'", "‘": "’"}
_APOSTROPHES = "'’"  # quotes, and apostrophes inside a word: don't, O'Brien, don’t
_RE_LINE_BREAK = re.compile("[\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")


def _word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _cased_word_in(text: str, word: str) -> bool:
    """Whether *word* (letters with case) stands in *text* as a word: no digit or letter with case right beside it, so
    a word glued to Chinese counts ("新的huawei手机": the seam of _script_pieces), "huaweis" does not."""
    def joins(c: str) -> bool:
        return c.isdigit() or c == "_" or (c.isalpha() and c.lower() != c.upper())
    i = text.find(word) if word else -1
    while i >= 0:
        j = i + len(word)
        if not (i > 0 and joins(text[i - 1])) and not (j < len(text) and joins(text[j])):
            return True
        i = text.find(word, i + 1)
    return False


def _uncased(name: str) -> bool:
    """A name with letters and none of them with case: Chinese, Japanese, Thai, Arabic... (0.16.2). Text finds such a
    name only in quotes (or in "X aka Y"): it has no capitals to stand out by. One Latin letter gives it case ("变量x"):
    such a name keeps the two-fact rule of the names with capitals."""
    letters = [c for c in name if c.isalpha()]
    return bool(letters) and all(c.lower() == c.upper() for c in letters)


def _outside_plain_quotes(text: str) -> str:
    """*text* with code and everything in plain quotes blanked (line breaks, so no name runs across), read in NFKC as
    mentions are ("＂" is a quote too); the title marks stay (0.16.2). Stricter than the extractor on purpose — a quote
    of any length, quotes nested in one of their own kind («a «b» c»), a quote never closed (to the end of its line),
    single quotes (but not an apostrophe inside a word: don't, O'Brien) — since what is blanked here only keeps a name
    provisional. Each character is looked at once or twice. Not cached: the store keeps what it needs per fact
    (_used_outside_quotes), and a process-wide cache would keep forgotten text in memory (review 0.16.2)."""
    text = unicodedata.normalize("NFKC", _without_code(unicodedata.normalize("NFC", text or "")))
    chars, i, n = list(text), 0, len(text)
    ends = [m.start() for m in _RE_LINE_BREAK.finditer(text)] + [n]  # any line break (str.splitlines)
    while i < n:
        opener = text[i]
        closers = _PLAIN_QUOTE_PAIRS.get(opener)
        if closers is None or (opener in _APOSTROPHES and i > 0 and _word_char(text[i - 1])):
            i += 1
            continue
        eol = ends[bisect.bisect_right(ends, i)]  # the end of its line
        depth, close, k = 1, eol - 1, i + 1
        while k < eol:
            c = text[k]
            if c in closers and not (c in _APOSTROPHES and k + 1 < n and _word_char(text[k + 1])):
                depth -= 1
                if depth == 0:
                    close = k
                    break
            elif c == opener and opener not in closers:
                depth += 1
            k += 1
        chars[i:close + 1] = "\n" * (close + 1 - i)
        i = close + 1
    return "".join(chars)


_RE_FENCE = re.compile(r"[ ]{0,3}(`{3,}|~{3,})")
_RE_TICKS = re.compile(r"`+")


def _without_code(text: str) -> str:
    """*text* with Markdown code blanked out, offsets kept: code is not a name (0.16.1: `Alice Smith` in a code span
    was a person, and "True aka False" a pair). Fenced blocks — ``` with no backtick after it on the line (else it is
    an inline span), to the closing fence or the end; ~~~ only with a closing fence, since a line of tildes is as often
    a rule — and inline spans of any run of backticks closed by a run of the same length; one pass. Indented lines are
    not taken for code: in notes they are nested lists ("Family:⏎    - Alice Smith", review 0.16.1). Blanked with NUL,
    neither a letter nor a space, so no name runs across it."""
    if "`" not in text and "~~~" not in text:
        return text
    chars = list(text)
    lines = text.splitlines(keepends=True)
    fences = [_RE_FENCE.match(line) for line in lines]
    # the last line that is a ~~~ fence: one opens a block only if another comes after it (review 0.16.2: looking
    # at every later line from every fence took 23 s on 80,000 lines of "~~~")
    last_tilde = max((n for n, f in enumerate(fences) if f is not None and f.group(1)[0] == "~"), default=-1)
    pos, fence = 0, None
    for n, (line, m) in enumerate(zip(lines, fences)):
        if m is not None and fence is None:
            mark = m.group(1)[0]
            opens = "`" not in line[m.end():] if mark == "`" else last_tilde > n
            if opens:
                fence = mark * 3
                chars[pos:pos + len(line)] = "\0" * len(line)
        elif fence is not None:
            chars[pos:pos + len(line)] = "\0" * len(line)
            if m is not None and m.group(1).startswith(fence):
                fence = None
        pos += len(line)
    text = "".join(chars)
    opened = None
    for m in _RE_TICKS.finditer(text):
        if opened is None:
            opened = m
        elif len(m.group(0)) == len(opened.group(0)):
            chars[opened.start():m.end()] = "\0" * (m.end() - opened.start())
            opened = None
    return "".join(chars)


def _mixed(name: str) -> bool:
    """A word of two scripts or cases glued together ("华为mate60"): read as strictly as a name in quotes."""
    return any(p.word != (p.start(), p.end()) for p in _script_pieces(name, _RE_TOKEN))


_ZERO_WIDTH_JOINERS = chr(0x200C) + chr(0x200D)
_MENTION_KEEPS = frozenset("'’&-‐‑_@#*%+")  # inside a word or a name: o'brien, at&t, jean-luc, c++


def _mention_stretches(text: str) -> list[str]:
    """*text* cut at everything the words of one name never stand across (0.16.1): line breaks and control characters,
    invisible format characters, brackets, quotes and other punctuation but the few that hold words together
    (_MENTION_KEEPS), and emoji. The analysis keeps the words and drops what stood between them, so before this
    "Alice\nSmith", "Alice, Smith" and two items of a list mentioned Alice Smith."""
    out, start = [], 0
    for i, c in enumerate(text):
        if c == "\t":
            continue
        cat = unicodedata.category(c)
        if cat in ("Cc", "Zl", "Zp", "So", "Sk") or (cat == "Cf" and c not in _ZERO_WIDTH_JOINERS) \
                or (cat[0] == "P" and c not in _MENTION_KEEPS and not _holds_word(text, i)):
            if i > start:
                out.append(text[start:i])
            start = i + 1
    if start < len(text):
        out.append(text[start:])
    return [_split_seams(stretch) for stretch in out]


def _holds_word(text: str, i: int) -> bool:
    """Whether the punctuation at *i* is part of a name, not a cut (review 0.16.1): a dot, colon or slash between two
    letters or digits (Node.js, Яндекс.Такси, 1С:Предприятие, app.example.com), or the dot after an initial or a short
    abbreviation (J. R. R. Tolkien, А. С. Пушкин, Dr. Alice Smith, U.S. Army). "Alice. Smith went" stays cut."""
    c = text[i]
    if c in ".:/" and 0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum():
        return True
    if c != ".":
        return False
    j = i
    while j > 0 and text[j - 1].isalpha():
        j -= 1
    return 1 <= i - j <= 2 and (j == 0 or not text[j - 1].isalnum())


def _split_seams(stretch: str) -> str:
    """*stretch* with a space where a word changes script, as the extractor reads it (0.16.1): "用Docker Hub直连" and
    "на Docker Hubе" mention Docker Hub."""
    cuts = [piece.start() for piece in _script_pieces(stretch, _RE_TOKEN) if piece.start() != piece.word[0]]
    if not cuts:
        return stretch
    out, last = [], 0
    for cut in cuts:
        out += (stretch[last:cut], " ")
        last = cut
    return "".join(out) + stretch[last:]


def _found_between_cuts(folded: str, name: str, cuts: frozenset) -> bool:
    """Whether *name* occurs in *folded* Chinese text beginning and ending at word cuts (or beside a character that is
    not Han, where a Han word ends anyway)."""
    def cut(i: int) -> bool:
        return i in cuts or i == 0 or i == len(folded) or not (
            _RE_HAN.match(folded[i - 1]) and _RE_HAN.match(folded[i]))
    start = folded.find(name) if name else -1
    while start >= 0:
        if cut(start) and cut(start + len(name)):
            return True
        start = folded.find(name, start + 1)
    return False


def _uncased_block(c: str) -> str:
    """Which writing a letter without case belongs to, for _script_pieces: Hangul, Thai, Lao, Khmer, Myanmar apart, and
    Han with kana as one (a Japanese word mixes them: 山田さん). Review 0.16.1: "测试삼성" was one piece."""
    name = unicodedata.name(c, "")
    for block in ("HANGUL", "THAI", "LAO", "KHMER", "MYANMAR"):
        if name.startswith(block):
            return block
    return "CJK" if UNSEGMENTED.match(c) else _script(c)


def _inline_gap(gap: str) -> bool:
    """Spaces between two words of one name: some, and no line break of any kind — str.splitlines' boundaries: LF, CR,
    VT, FF, NEL, the Unicode line and paragraph separators (0.16.1: a heading and a name on the next line were taken
    as one name)."""
    return bool(gap) and gap.isspace() and len(("x" + gap + "x").splitlines()) == 1 \
        and not any(unicodedata.category(c) == "Cc" for c in gap)  # a tab too: cells of a table (review 0.16.1)


def _capitalized_phrases(text: str) -> list[str]:
    """Names of two to five capitalized words in any cased script, minus a sentence-opening function word."""
    phrases = []
    words_in_order = _script_pieces(text, _RE_WORD)
    run: list[_Piece] = []

    def flush():
        words = run[1:] if run and fold(run[0].group(0)) in _SENTENCE_STARTERS else run
        if 2 <= len(words) <= _PHRASE_MAX_WORDS:
            phrases.append(text[words[0].start():words[-1].end()])

    for m in words_in_order:
        if _is_capitalized(m.group(0)) and (not run or _inline_gap(text[run[-1].end():m.start()])):
            run.append(m)
            continue
        flush()
        run = [m] if _is_capitalized(m.group(0)) else []
    flush()
    return phrases


def _aka_side(segment: str, *, from_end: bool) -> str | None:
    """The name right before (from_end) or right after an 'aka': a capitalized run, else a single token."""
    tokens = _script_pieces(segment, _RE_TOKEN)
    if not tokens:
        return None
    edge = tokens[-1] if from_end else tokens[0]
    if (edge.end() != len(segment)) if from_end else (edge.start() != 0):
        return None  # punctuation between the name and 'aka'
    if edge.word != (edge.start(), edge.end()):
        # a word of two scripts ("华为mate60 aka Huawei"): the whole word is the side, alone, and _aka_pairs judges it
        # as a name in quotes is judged (0.16.1: cutting it at the seam lost a model name and its alias)
        return segment[edge.word[0]:edge.word[1]]
    picked = [edge]
    if _is_capitalized(edge.group(0)):
        neighbours = reversed(tokens[:-1]) if from_end else tokens[1:]
        for tok in neighbours:
            gap = segment[tok.end():picked[-1].start()] if from_end else segment[picked[-1].end():tok.start()]
            if len(picked) >= _PHRASE_MAX_WORDS or not _inline_gap(gap) or not _is_capitalized(tok.group(0)):
                break
            picked.append(tok)
    if from_end:
        picked.reverse()
    return segment[picked[0].start():picked[-1].end()]


def _clean_given(name) -> str:
    """A name someone gives, without what they cannot have meant: edge punctuation (a leading dot stays: ".NET") and
    invisible format characters such as the bidi marks a browser leaves in an Arabic or Hebrew name (ZWJ / ZWNJ stay
    where the script writes them). Review 0.16.1: such a name was refused, silently."""
    if not isinstance(name, str):
        return ""
    name = "".join(c for i, c in enumerate(name) if unicodedata.category(c) != "Cf" or _joiner_ok(name, i))
    return name.strip(" ,;:!?").rstrip(".")


def _legacy_aliases(names) -> str:
    """The legacy comma-separated entities.aliases value, which older versions resolve aliases from and every writer
    open imports again by splitting it on commas: a name with a comma is left out of it (entity_aliases keeps it whole),
    or "Smith, John" would come back as the aliases "Smith" and "John" (review 0.16.0)."""
    return ",".join(dict.fromkeys(n.strip() for n in names if isinstance(n, str) and n.strip() and "," not in n))


def _valid_entity(name: str, *, given: bool = False) -> bool:
    """Reject what is not a name: too short or long, not beginning with a letter or a digit (".NET" may: a dot and a
    capital), key=value, paths, URLs, IPs, mostly non-letters. The
    check for names someone gave (entities=[...], alias, rename); names guessed from text pass _name_shaped too."""
    if not 2 <= len(name) <= _ENTITY_MAX_CHARS or _RE_NOT_A_NAME.search(name):
        return False
    # The same checks on the form names are compared in (0.16.1): "https：／／example.com" is a URL once NFKC folds its
    # full-width colon and slashes, as entity_key does; and no invisible format character (a zero-width space makes a
    # copy of a deleted name that its tombstone does not know) — but ZWJ / ZWNJ, which some scripts write words with.
    folded = unicodedata.normalize("NFKC", name)
    if _RE_NOT_A_NAME.search(folded) or len(name.splitlines()) > 1 or any(
            unicodedata.category(c) == "Cc" or (unicodedata.category(c) == "Cf" and not _joiner_ok(name, i))
            for i, c in enumerate(name)):
        return False
    if not (name[0].isalnum() or (name[0] == "." and name[1:2].isupper())  # 0.16.1: "+.siliconflow.cn", "-flag"
            or (given and name[0] in ".'’$" and name[1:2].isalnum())):  # given: ".net", "'s-Hertogenbosch", "$GME"
        return False
    visible = [c for c in name if not c.isspace()]
    if given:  # someone named it: "3M", "O2", "C++" are names; a letter is enough, but not a credential or an
        # instruction, which would be stored and shown as a name (review 2026-11-17, H2: a token in entities was)
        return any(c.isalpha() for c in visible) and not _fact_problem(name)
    if name[0].isdigit() and is_chinese(name):
        # 0.16.2 (issue #4): a Han character is a word, so the share of letters below dropped half the Chinese names
        # that begin with a number (360安全卫士, 115网盘, 36氪) and kept some counts ("3个月" yes, "15分钟" no). A
        # number and units is a count (text.zh_count); anything else after the number makes a name.
        return not zh_count(name) and any(c.isalpha() for c in visible)
    # vowel signs and other marks count with their letters (review 0.16.1: "श्री" fell under the bar)
    return sum(c.isalpha() or unicodedata.category(c)[0] == "M" for c in visible) >= 0.6 * len(visible)


_JOINER_SCRIPTS = frozenset({"arabic", "devanagari", "bengali", "indic"})


def _joiner_ok(name: str, i: int) -> bool:
    """A zero-width joiner or non-joiner between two letters (or a letter and a mark) of a script that writes words
    with them; anywhere else it only makes a copy of a name that its key and tombstone do not know (review 0.16.1:
    "Alice\u200d Smith" came back after "Alice Smith" was deleted)."""
    if name[i] not in "\u200c\u200d" or not 0 < i < len(name) - 1:
        return False
    return all(_script(c) in _JOINER_SCRIPTS for c in (name[i - 1], name[i + 1]) if c.isalpha()) and \
        any(c.isalpha() or unicodedata.category(c)[0] == "M" for c in (name[i - 1], name[i + 1]))


_NOT_A_NAME = (f"not a name (2-{_ENTITY_MAX_CHARS} characters, mostly letters; no paths, URLs, key=value, commas, "
               "line breaks or flags)")


def unusable_names(names) -> list[dict]:
    """The names in *names* that a declaration (entities=[...]) cannot link, by position (0-based) with why: what
    _link_declared skips. A fact is stored all the same; the agent is told which names did not link, rather than
    believing they did. Not the name itself: a refused one may be `KEY=secret` or an instruction, which the
    answer would carry back past the screen."""
    out = []
    for index, name in enumerate(names or ()):
        if not isinstance(name, str):
            out.append({"index": index, "reason": "not a string"})
        elif not _valid_entity(_clean_given(name), given=True):
            out.append({"index": index, "reason": _NOT_A_NAME})
    return out


def _name_shaped(name: str, *, quoted: bool) -> bool:
    """_valid_entity, and for a name guessed from a fact's text (0.16.0) also not: a version number with nothing that
    names a thing beside it ("0.17.0 installed", «версия 0.17.0»; "Python 3.12" and «1С:Предприятие 8.3» stay, for a
    word with a capital or in an uncased script names one); in quotes, a span in a cased script all in lower case with
    no identifier in it ('count', 'run fast', «быстро беги»; 'kanban.db', 'gpt-4o' and 'qx-vps' stay), a greeting or
    courtesy ("Good morning", «Спасибо большое», “谢谢”: text.is_greeting), a Chinese span that reads as something said
    (text.zh_phrase: “先去搜索再告诉我答案”, “知道了”; 平凡的世界, 有赞科技有限公司 and 饿了么 stay), or more Han characters
    than a name has (a quotation). A name in lower case, or a phrase that only reads as one, is what someone said;
    the agent can still name it with entities, and what passes and is no name stays provisional."""
    if not _valid_entity(name) or any(unicodedata.category(c) in ("So", "Sk") for c in name):
        return False  # an emoji or another picture is no part of a name guessed from text ("Alice🙂Smith")
    tokens = [t.strip(".,;:!?()'\"«»“”„") for t in name.split()]
    if any(_RE_VERSION.fullmatch(t) for t in tokens) and not any(
            any(c.isupper() or (c.isalpha() and c.lower() == c.upper()) for c in t)
            for t in tokens if not _RE_VERSION.fullmatch(t)):
        return False
    if not quoted:
        return True
    # Judged on the letters that have case, whatever else the span holds ("hrr_dim太大" too), and an identifier must
    # have a letter in it: "14 days" is a count, "gpt-4o" a name (0.16.1, issue #4)
    cased = [c for c in name if c.isalpha() and c.lower() != c.upper()]
    if cased and not any(c.isupper() for c in cased) \
            and not any(is_identifier(t) and any(c.isalpha() for c in t) for t in tokens) \
            and not (len(tokens) == 1 and _mixed(name)):  # one word glued to Chinese: 微信pay, 小米pad (review 0.16.1)
        return False
    # Every letter of a script written without spaces counts (Han, kana, Hangul, Thai...: review 0.16.1, a Japanese or
    # Thai sentence in quotes has no space and few Han), and Japanese and Korean predicates end what is said
    return sum(len(run) for run in UNSEGMENTED.findall(name)) <= _QUOTED_MAX_HAN and not is_greeting(name) \
        and not zh_phrase(name) and not ja_ko_phrase(name)
