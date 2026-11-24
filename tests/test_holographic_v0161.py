"""0.16.1: what the 0.16.0 rebuild showed on a Chinese store (issue #4) — a capitalized name glued to Chinese text was
taken with its tail, a name could begin with "+", a count in quotes passed — and the same kinds of seam elsewhere:
other scripts glued to a name, a name across a line break, straight quotes beside Chinese characters, "aka" beside a
glued word. Every rule in Chinese, English and Russian."""
from __future__ import annotations

import pytest

from hermes_plugin_holographic import store as store_module
from hermes_plugin_holographic.store import MemoryStore, _valid_entity

DIM = 64


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


@pytest.mark.parametrize("lang,text,expected", [
    # a capitalized run ends where the script changes (issue #4: "Docker Hub直连可用" was one name)
    ("zh", "Docker Hub直连可用", ["Docker Hub"]),
    ("zh", "Docker Hub直连通有新版", ["Docker Hub"]),
    ("zh", "测试Docker Hub直连", ["Docker Hub"]),  # the name glued after Chinese is found too
    ("ru", "на Docker Hubе лежит образ", ["Docker Hub"]),  # a Russian ending glued to a Latin name
    ("ru", "звонила Мария ИвановаBank", ["Мария Иванова"]),
    ("en", "met Alice Smith和Bob", ["Alice Smith"]),
    # a run never crosses a line break
    ("en", "Meeting Notes\nAlice Smith called", ["Meeting Notes", "Alice Smith"]),
    ("ru", "Список Покупок\nМария Иванова звонила", ["Список Покупок", "Мария Иванова"]),
    ("zh", "Weekly Report\nDocker Hub直连可用", ["Weekly Report", "Docker Hub"]),
    # what stays as it was
    ("en", "Jean-Luc Picard came", ["Jean-Luc Picard"]),
    ("en", "Alice Smith works at Bank Of America", ["Alice Smith", "Bank Of America"]),
    ("ru", "Вчера Мария Иванова звонила", ["Мария Иванова"]),
    ("zh", "他在“华为”工作", ["华为"]),
])
def test_a_capitalized_name_ends_at_a_script_seam_and_a_line_break(store, lang, text, expected):
    assert store._extract_entities(text) == expected


@pytest.mark.parametrize("lang,text,expected", [
    ("zh", '用"Docker Hub"直连', ["Docker Hub"]),  # straight quotes beside Chinese characters are quotes
    ("zh", '他在"华为"工作', ["华为"]),
    ("ru", 'Счёт в "Сбер Бизнес" открыт', ["Сбер Бизнес"]),
    ("en", 'a 5" screen and a 6" one', []),  # beside a digit they are inches
    ("en", "don't stop it's fine", []),  # single quotes beside a letter are apostrophes
])
def test_straight_double_quotes_beside_chinese_are_quotes(store, lang, text, expected):
    assert store._extract_entities(text) == expected


@pytest.mark.parametrize("lang,text", [
    ("zh", "代理“+.siliconflow.cn”"),  # issue #4: a leading "+" passed
    ("en", 'proxy "+.siliconflow.cn"'),
    ("en", 'tagged "#Phoenix"'),
    ("ru", "флаг «*Важно»"),
    ("en", 'waited "14 days"'),  # issue #4: a count is no identifier
    ("ru", "ждали «14 дней»"),
    ("en", 'bought "华为 mate"'),  # lower-case Latin beside Chinese: judged on its letters with case
])
def test_what_begins_with_a_symbol_or_is_a_count_is_no_name(store, lang, text):
    assert store._extract_entities(text) == []


@pytest.mark.parametrize("lang,text,name", [
    ("zh", "买了“华为mate60”", "华为mate60"),  # glued and with a digit: a model name
    ("en", "use 'gpt-4o' for drafts", "gpt-4o"),
    ("en", 'He bought "iPhone 15 Pro"', "iPhone 15 Pro"),
    ("en", 'we use "Notepad++"', "Notepad++"),
    ("ru", "работаем в «1С:Предприятие 8.3»", "1С:Предприятие 8.3"),
    ("en", 'shop at "7-Eleven"', "7-Eleven"),
])
def test_names_that_begin_with_a_digit_or_hold_one_stay(store, lang, text, name):
    assert name in store._extract_entities(text)


@pytest.mark.parametrize("name,ok", [
    ("+.siliconflow.cn", False), ("-flag", False), ("—Тест", False), ("#Phoenix", False), ("*Важно", False),
    (".NET", True), (".net", False), ("7-Eleven", True), ("北京大学", True), ("Мария", True),
])
def test_a_given_name_begins_with_a_letter_or_a_digit(name, ok):
    assert _valid_entity(name) is ok


@pytest.mark.parametrize("text,expected", [
    # a side that is one word of two scripts is that whole word, judged as a name in quotes (review 0.16.1: cutting
    # it lost "华为mate60 aka Huawei"); a glued phrase then stays provisional
    ("华为mate60 aka Huawei", ["华为mate60", "Huawei"]),
    ("Docker Hub直连 aka DH", ["Docker Hub", "Hub直连", "DH"]),
    ("DH aka Docker Hub直连", ["Docker Hub", "DH"]),  # the right side stops at the seam
    ("Мария Иванова aka Маша звонила", ["Мария Иванова", "Маша"]),
    ("北京大学 aka 北大 很美", ["北京大学", "北大"]),
    ("user42 aka bob", ["user42", "bob"]),
    ("Meeting Notes\nAlice Smith aka Ally", ["Meeting Notes", "Alice Smith", "Ally"]),
])
def test_aka_sides_stop_at_a_seam_and_a_line_break(store, text, expected):
    assert store._extract_entities(text) == expected


def test_a_store_indexed_under_the_0160_rules_is_rebuilt(tmp_path):
    """The rules changed again: a store 0.16.0 indexed (rules 2) is rebuilt on open, and the names of issue #4 go."""
    path = tmp_path / "facts.db"
    s = MemoryStore(str(path), hrr_dim=DIM)
    ids = [s.add_fact(t) for t in ("Docker Hub直连可用", "代理“+.siliconflow.cn”", 'waited "14 days"', "Code name “Phoenix”")]
    for fact_id, name in zip(ids[:3], ("Docker Hub直连可用", "+.siliconflow.cn", "14 days")):
        cur = s._write("INSERT INTO entities (name, name_key) VALUES (?, ?)", (name, store_module.entity_key(name)))
        s._write("INSERT INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (fact_id, cur.lastrowid))
    s._write("UPDATE meta SET value = '2' WHERE key = 'entity_rules'")
    s.close()
    s = MemoryStore(str(path), hrr_dim=DIM)
    assert s.entity_rules() == store_module._ENTITY_RULES >= 3
    assert sorted(r["name"] for r in s.list_entities()) == ["Docker Hub", "Phoenix"]
    s.close()


# --- review of 0.16.1 (Codex) ------------------------------------------------------------------------------------

def test_a_model_name_and_its_alias_survive_the_rebuild_from_0160(tmp_path):
    """Review 0.16.1, P1: cutting "华为mate60 aka Huawei" at the seam lost the pair, and the rebuild then dropped the
    alias and the entity. The whole word is the side again."""
    path = tmp_path / "facts.db"
    s = MemoryStore(str(path), hrr_dim=DIM)
    s.add_fact("华为mate60 aka Huawei 很好用")
    s._write("UPDATE meta SET value = '2' WHERE key = 'entity_rules'")
    s.close()
    s = MemoryStore(str(path), hrr_dim=DIM)
    assert s.entity_rules() == store_module._ENTITY_RULES and s.entity_ids("Huawei") == s.entity_ids("华为mate60") != []
    s.close()


@pytest.mark.parametrize("name,text,mentioned", [
    ("Alice Smith", "met Alice Smith today", True),
    ("Alice Smith", "Alice\nSmith", False),
    ("Alice Smith", "Alice, Smith", False),
    ("Alice Smith", "Alice [ ] Smith", False),
    ("Alice Smith", "Alice" + chr(0x2028) + "Smith", False),
    ("Мария Иванова", "звонила Мария Иванова", True),
    ("Мария Иванова", "Мария;\nИванова", False),
])
def test_a_known_name_is_mentioned_only_as_written_on_one_line(store, name, text, mentioned):
    """Review 0.16.1, P1: the analysis keeps words and drops what stood between them, so items of a list mentioned a
    person whose first and last names they held."""
    fact, _ = store.add_fact_status("A fact about them", entities=[name])
    entity_id = store.entity_ids(name)[0]
    assert (entity_id in store.mentioned_entity_ids(text)) is mentioned


@pytest.mark.parametrize("name,text,mentioned", [
    ("和服", "我和服务员说话", False),  # 和 | 服务员: across a seam between words
    ("和服", "她穿了一件和服", True),
    ("北京大学", "我在北京大学读书", True),
    ("臺灣大學", "他在台湾大学", True),  # Traditional and Simplified as one
])
def test_a_chinese_name_is_not_found_across_a_word_seam(store, name, text, mentioned):
    """Review 0.16.1, P1: a declared Chinese name was found as a substring across two words."""
    store.add_fact_status("A fact about it", entities=[name])
    entity_id = store.entity_ids(name)[0]
    assert (entity_id in store.mentioned_entity_ids(text)) is mentioned


@pytest.mark.parametrize("sep", ["\n", "\r", "\x0b", "\x0c", "\x85", chr(0x2028), chr(0x2029)])
def test_no_line_break_of_any_kind_joins_two_names(store, sep):
    """Review 0.16.1, P1: only LF and CR ended a name; the Unicode line and paragraph separators, NEL, VT and FF did
    not."""
    assert store._extract_entities(f"Alice Smith{sep}Bob Jones") == ["Alice Smith", "Bob Jones"]
    assert store._aka_pairs(f"Alice Smith{sep}aka{sep}Ally") == []


@pytest.mark.parametrize("text", [
    'see "https：／／example.com"',  # a URL once NFKC folds its full-width colon and slashes
    "“Alice" + chr(0x200B) + " Smith”",  # a zero-width space
    'called "Bob' + chr(0x2060) + 'Brown"',  # a word joiner
])
def test_full_width_and_invisible_forms_do_not_pass_the_gate(store, text):
    """Review 0.16.1, P2: the checks ran on the text as written, the identity on its NFKC form."""
    assert store._extract_entities(text) == []


@pytest.mark.parametrize("name", ["3M", "O2", "C++", "AT&T"])
def test_a_short_name_someone_gives_is_accepted(store, name):
    """Review 0.16.1, P2: the 60 % letter rule refused short brands even when the agent named them; it applies to
    names guessed from text only."""
    fact, _ = store.add_fact_status("They make tape", entities=[name])
    assert store.entity_ids(name) and _valid_entity(name, given=True)
    store.add_alias(name, name + " Company")


@pytest.mark.parametrize("text,name", [
    ("Jose" + chr(0x301) + " Alvarez called", "José Alvarez"),  # a combining accent: one letter in NFC
    ('We use ".NET" daily', ".NET"),  # the leading dot stays
])
def test_text_is_read_in_nfc_and_a_leading_dot_stays(store, text, name):
    assert name in store._extract_entities(text)


@pytest.mark.parametrize("text", [
    "run `Alice Smith` in the shell",
    "```\nAlice Smith aka Ally\n```",
    "code: `return True aka False`",
    "use `'kanban.db'` there",
])
def test_code_spans_and_blocks_give_no_names(store, text):
    """Review 0.16.1: a name in a code span became a person, and "True aka False" a pair."""
    assert store._extract_entities(text) == []


@pytest.mark.parametrize("text,expected", [
    ('Ａ"Phoenix"Ｂ', []),  # full-width letters with case beside the quotes: not quotes
    ('5" and 6"', []),
    ('他用"Docker Hub"', ["Docker Hub"]),
    ('ок "Сбер Бизнес" ок', ["Сбер Бизнес"]),
])
def test_straight_quotes_are_judged_by_the_letters_beside_them(store, text, expected):
    """Review 0.16.1, P3: character ranges missed full-width letters with case."""
    assert store._extract_entities(text) == expected


# --- second review of 0.16.1 (Codex gpt-6-astra) ------------------------------------------------------------------

@pytest.mark.parametrize("sep", ["\r", "\x0b", "\x0c", "\x85", "\x00", chr(0x2028), chr(0x2029)])
@pytest.mark.parametrize("left,right,quote", [("Alice", "Smith", '"{}"'), ("Мария", "Иванова", "«{}»"),
                                              ("Alice", "Smith", "“{}”")])
def test_a_quoted_name_holds_no_line_break_or_control_character(store, sep, left, right, quote):
    """A line break or a control character inside quotes made "Alice\rSmith", which entity_key folds into Alice Smith."""
    assert store._extract_entities(quote.format(left + sep + right)) == []


@pytest.mark.parametrize("text", ["Alice" + c + "Smith" for c in ("\x0b", "\x0c", "\x85", "\x00")]
                         + ["Alice, Smith; malice smithson", "Alice (Smith)", "Alice / Smith", "Alice — Smith"])
def test_a_known_name_is_not_mentioned_across_a_control_or_a_separator(store, text):
    store.add_fact_status("x", entities=["Alice Smith"])
    assert store.entity_ids("Alice Smith")[0] not in store.mentioned_entity_ids(text)


@pytest.mark.parametrize("text", ["met Alice Smith", "met Ａｌｉｃｅ Ｓｍｉｔｈ", "Alice Smith's car", "@Alice Smith"])
def test_a_known_name_is_mentioned_in_nfkc_and_beside_kept_marks(store, text):
    store.add_fact_status("x", entities=["Alice Smith"])
    assert store.entity_ids("Alice Smith")[0] in store.mentioned_entity_ids(text)


@pytest.mark.parametrize("name,text,mentioned", [
    ("小米科技", "我在小米科技工作", True),  # a name the segmenter splits in two (小米 | 科技) is still one name
    ("山田花子", "我认识山田花子", True),
    ("和服", "我和服务员说话", False),
    ("北京", "我在北京大学读书", False),  # 北京大学 is one word: 北京 inside it is not a mention
    ("北京", "我住在北京", True),
])
def test_a_chinese_name_is_found_from_word_cut_to_word_cut(store, name, text, mentioned):
    store.add_fact_status("x", entities=[name])
    assert (store.entity_ids(name)[0] in store.mentioned_entity_ids(text)) is mentioned


@pytest.mark.parametrize("text", [
    "``Alice Smith aka Mallory Jones``",
    "~~~\nAlice Smith aka Ally\n~~~",
    "```\nAlice Smith aka Ally",  # an unclosed fence runs to the end
    "```python\nAlice Smith aka Ally\n```",  # an indented line is no code (review 0.16.1): see the nested lists below
])
def test_markdown_code_of_every_kind_gives_no_names(store, text):
    assert store._extract_entities(text) == []


def test_code_does_not_give_an_established_name_an_alias(store):
    store.add_fact_status("x", entities=["Alice Smith"])
    store.add_fact("``Alice Smith aka Mallory Jones``")
    assert store.entity_ids("Mallory Jones") == []


@pytest.mark.parametrize("name", ["Alice" + chr(0x200D) + " Smith", "Alice" + chr(0x200C) + "Smith",
                                  "Alice" + chr(0x00AD) + "Smith"])
def test_a_joiner_outside_the_scripts_that_write_them_is_refused(store, name):
    """A zero-width joiner made a copy of a deleted Latin name that its tombstone did not know."""
    assert not _valid_entity(name)


@pytest.mark.parametrize("name", ["क्ष" + chr(0x200D) + "त्र", "می" + chr(0x200C) + "خواهم", "श्री"])
def test_a_joiner_and_marks_in_the_scripts_that_write_them_stay(name):
    assert _valid_entity(name)


@pytest.mark.parametrize("deleted,again", [("臺灣大學", "台湾大学"), ("José Alvarez", "Jose Alvarez")])
def test_a_tombstone_holds_the_folded_form_too(store, deleted, again):
    """Search compares Traditional and Simplified Chinese, and accents, as one; deletion now does too."""
    store.add_fact_status("x", entities=[deleted])
    store.delete_entity(deleted)
    assert store._tombstoned(again)
    store.add_fact(f"“{again}”")
    assert store.entity_ids(again) == []


def test_a_tombstone_written_by_0160_is_still_read(store):
    store.add_fact_status("x", entities=["Phoenix Labs"])
    store._write("INSERT INTO entity_tombstones (digest) VALUES (?)", (store._name_digest("Old Name"),))
    assert store._tombstoned("Old Name")


@pytest.mark.parametrize("text", [" " * 10000, "«" * 10000, "“" * 10000, "'" * 10000, '"' * 10000, "`" * 10000,
                                  "a aka " * 1600, ("Alice\t" * 2000)],
                         ids=["spaces", "guillemets", "curly", "single", "straight", "backticks", "aka", "tabs"])
def test_long_adversarial_text_is_read_in_linear_time(store, text):
    """Review 0.16.1: a leading run of spaces in the "aka" pattern and unbounded quoted spans were quadratic (10,000
    spaces: 1.1 s per fact; three such facts held the rebuild's write lock for 6.5 s).

    Measured against the same text in pieces rather than a fixed limit, which a slow runner crossed in linear time
    (macOS CI: 0.65 s for the tabs): reading the text once is as much work as reading a sixteenth of it sixteen times
    when the text is read in linear time, and sixteen times more when quadratic. The two take about as long, so a busy
    machine slows both alike; taken in turns, the fastest of three each, so a pause in one does not count."""
    import time

    def took(sample, reads):
        started = time.perf_counter()
        for _ in range(reads):
            store._extract_entities(sample)
            store.mentioned_entity_ids(sample)
        return time.perf_counter() - started

    whole, pieces = zip(*[(took(text, 1), took(text[:len(text) // 16], 16)) for _ in range(3)])
    assert min(whole) < 4 * min(pieces)


def test_a_declared_name_keeps_its_leading_dot(store):
    store.add_fact_status("runtime", entities=[".NET"])
    assert [r["name"] for r in store.list_entities()] == [".NET"]


# --- review of 0.16.1 (Antigravity) ---------------------------------------------------------------------------------

def test_dotnet_is_not_the_word_net(store):
    """".NET" had the key "net", so "a fishing net" mentioned it and the two were one entity."""
    from hermes_plugin_holographic.store import entity_key
    assert entity_key(".NET") == ".net" and entity_key("net") == "net" and entity_key(".hidden.") == ".hidden"
    store.add_fact_status("runtime", entities=[".NET"])
    dotnet = store.entity_ids(".NET")[0]
    assert dotnet not in store.mentioned_entity_ids("We caught fish with a net")
    assert dotnet in store.mentioned_entity_ids("We build it on .NET 8")
    store.add_fact_status("a fishing trip", entities=["net"])
    assert store.entity_ids("net") != [dotnet]


def test_keys_written_before_are_moved_to_today_s_on_open(tmp_path):
    from hermes_plugin_holographic.store import _entity_key_0160
    path = tmp_path / "facts.db"
    s = MemoryStore(str(path), hrr_dim=DIM)
    s.add_fact_status("runtime", entities=[".NET"])
    s.add_alias(".NET", "dotnet framework")
    s._write("UPDATE entities SET name_key = ? WHERE name = '.NET'", (_entity_key_0160(".NET"),))
    s.close()
    s = MemoryStore(str(path), hrr_dim=DIM)
    assert s.read("SELECT name_key FROM entities WHERE name = '.NET'")[0][0] == ".net"
    assert s.entity_ids(".NET") == s.entity_ids("dotnet framework") != []
    s.close()


def test_a_tombstone_of_0160_holds_for_a_rekeyed_name(store):
    from hermes_plugin_holographic.store import _entity_key_0160
    store.add_fact("x")
    store._write("INSERT INTO entity_tombstones (digest) VALUES (?)", (store.digest("entity " + _entity_key_0160(".NET")),))
    assert store._tombstoned(".NET")


@pytest.mark.parametrize("name,key", [("‘Alice’", "alice"), ("’Alice’", "alice"), ("«Alice»", "alice")])
def test_curly_single_quotes_are_edges_of_a_key(name, key):
    from hermes_plugin_holographic.store import entity_key
    assert entity_key(name) == key


@pytest.mark.parametrize("text", [
    '彼は"明日は雨が降るかもしれません"と言った',
    "「私は学生です」",
    "“私は学生です”",
    "“좋아요”",
    "“감사합니다”",
    "“พรุ่งนี้ฝนอาจจะตก”",  # 17 Thai letters and marks: over the bar for a script written without spaces
])
def test_japanese_korean_and_thai_sentences_in_quotes_are_no_names(store, text):
    assert store._extract_entities(text) == []


@pytest.mark.parametrize("text,name", [
    ("“となりのトトロ”を見た", "となりのトトロ"),
    ("“東京ディズニーランド”に行った", "東京ディズニーランド"),
    ("“ゆか”さん", "ゆか"),  # a name that ends in か is no sentence
    ("“캐나다”", "캐나다"),  # 다 alone is no predicate ending
    ("“삼성전자”", "삼성전자"),
    ("“กรุงเทพมหานคร”", "กรุงเทพมหานคร"),
])
def test_japanese_korean_and_thai_names_in_quotes_stay(store, text, name):
    assert name in store._extract_entities(text)


def test_a_seam_between_han_and_hangul_splits_a_word(store):
    from hermes_plugin_holographic.store import _RE_TOKEN, _script_pieces
    assert [p.group(0) for p in _script_pieces("测试삼성", _RE_TOKEN)] == ["测试", "삼성"]
    assert [p.group(0) for p in _script_pieces("山田さん", _RE_TOKEN)] == ["山田さん"]  # Han with kana is one


# --- review of 0.16.1 (Claude Fable 5.1) ----------------------------------------------------------------------------

@pytest.mark.parametrize("name,text", [
    ("Node.js", "I like Node.js a lot"),
    ("Vue.js", "rewrote it in Vue.js"),
    ("kanban.db", "the board lives in kanban.db"),
    ("app.qx24copilot.online", "deployed to app.qx24copilot.online today"),
    ("Mail.ru Group", "works at Mail.ru Group"),
    ("Яндекс.Такси", "вызвал Яндекс.Такси"),
    ("1С:Предприятие", "работаем в 1С:Предприятие"),
    ("J. R. R. Tolkien", "reading J. R. R. Tolkien again"),
    ("А. С. Пушкин", "стихи А. С. Пушкин писал"),
    ("U.S. Army", "served in the U.S. Army"),
    ("Dr. Alice Smith", "met Dr. Alice Smith"),
])
def test_a_name_with_a_dot_colon_or_initials_is_still_mentioned(store, name, text):
    """Review 0.16.1, P1 (a regression of the first draft): the text was cut at every dot, colon and slash, so names
    with one inside, and initials, were no longer mentioned."""
    store.add_fact_status("about " + name, entities=[name])
    assert store.entity_ids(name)[0] in store.mentioned_entity_ids(text)


def test_a_full_stop_after_a_word_still_cuts(store):
    store.add_fact_status("x", entities=["Alice Smith"])
    assert store.entity_ids("Alice Smith")[0] not in store.mentioned_entity_ids("I saw Alice. Smith went home")


@pytest.mark.parametrize("text,expected", [
    ("Family:\n    - Alice Smith (wife)\n    - Bob Smith (son)", ["Alice Smith", "Bob Smith"]),
    ("Семья:\n    - Мария Иванова (жена)\n    - Пётр Иванов (сын)", ["Мария Иванова", "Пётр Иванов"]),
    ("- Alice Smith\n    - Bob Jones", ["Alice Smith", "Bob Jones"]),
    ("Family:\n\t- Alice Smith (wife)", ["Alice Smith"]),
    ("```pip install x``` then call Alice Smith", ["Alice Smith"]),  # a fence with a backtick after it is a span
    ("Alice Smith\n~~~~~~~~\nBob Jones", ["Alice Smith", "Bob Jones"]),  # a line of tildes is a rule, not a fence
])
def test_indented_lists_and_rules_are_no_code(store, text, expected):
    """Review 0.16.1, P1 (a regression of the first draft): an indented line was taken for code, so nested lists — the
    usual shape of notes — gave no names."""
    assert store._extract_entities(text) == expected


@pytest.mark.parametrize("via", ["update", "change"])
def test_the_rebuild_does_not_give_a_name_two_facts_hold_to_an_edited_claim(tmp_path, via):
    """Review 0.16.1, P1: fact ids stay through update and change, so a fact edited into "Mallory Jones aka Alice Smith"
    sorted before the facts about Alice Smith, and the rebuild merged her into Mallory."""
    path = tmp_path / "facts.db"
    s = MemoryStore(str(path), hrr_dim=DIM)
    first = s.add_fact("The weather is nice today")
    s.add_fact("Alice Smith lives in Paris")
    s.add_fact("Alice Smith works at Acme Corp")
    if via == "update":
        s.update_fact(first, content="Mallory Jones aka Alice Smith is a friend")
    else:
        s.change_fact(first, "Mallory Jones aka Alice Smith is a friend")
    s.rebuild_entity_index()
    assert s.entity_ids("Alice Smith") != s.entity_ids("Mallory Jones")
    assert next(r for r in s.list_entities() if r["name"] == "Alice Smith")["facts"] == 3
    s.close()


@pytest.mark.parametrize("text,expected", [
    ("Alice Smith\tBob Jones\t42", ["Alice Smith", "Bob Jones"]),
    ("Name\tCity\nAlice Smith\tParis", ["Alice Smith"]),
    ("Мария Иванова\tМосква", ["Мария Иванова"]),
])
def test_a_tab_ends_a_name(store, text, expected):
    """Review 0.16.1, P2: a tab joined the cells of a table into one run, which the gate then refused whole."""
    assert store._extract_entities(text) == expected


@pytest.mark.parametrize("given,stored", [
    ("محمد" + chr(0x200F) + " علي", "محمد علي"),  # a bidi mark from the browser goes, the name stays
    ("Alice" + chr(0x200E) + " Smith", "Alice Smith"),
    (".net", ".net"),
    ("'s-Hertogenbosch", "'s-Hertogenbosch"),
    ("$GME", "$GME"),
])
def test_a_name_someone_gives_is_cleaned_not_refused(store, given, stored):
    """Review 0.16.1, P2: such names were dropped without a word."""
    store.add_fact_status("about it", entities=[given])
    assert [r["name"] for r in store.list_entities()] == [stored]


@pytest.mark.parametrize("text,name", [
    ("We flew to Hawaiʻi Island", "Hawaiʻi Island"),
    ("Swam at Oʻahu Beach", "Oʻahu Beach"),
    ("вчера звонил Вʼячеслав Петренко", "Вʼячеслав Петренко"),
])
def test_modifier_letters_hold_a_word_together(store, text, name):
    """Review 0.16.1, P2 (a regression of the first draft): ʻ and ʼ cut a word at a "script seam"."""
    assert name in store._extract_entities(text)


def test_yo_and_ye_are_one_name(store):
    """Review 0.16.1, P2: Семён and Семен were two people, and deleting one left the other."""
    store.add_fact("Семён Петров пришёл")
    store.add_fact("Семен Петров ушёл")
    assert len(store.list_entities()) == 1
    store.delete_entity("Семён Петров")
    assert store._tombstoned("Семен Петров")


@pytest.mark.parametrize("text,name", [
    ("他在「华为」工作", "华为"),
    ("我读了《平凡的世界》", "平凡的世界"),
    ("彼は『東京大学』にいる", "東京大学"),
    ("「山田花子」に会った", "山田花子"),
    ("〈小王子〉很好看", "小王子"),
])
def test_the_quotes_of_chinese_japanese_and_korean_are_quotes(store, text, name):
    assert name in store._extract_entities(text)


@pytest.mark.parametrize("text,expected", [
    ("श्री राम aka Ram", ["राम", "Ram"]),  # one word: a script without case builds no run
    ("กิตติ aka Kit", ["กิตติ", "Kit"]),
])
def test_vowel_signs_belong_to_the_side_of_aka(store, text, expected):
    """Review 0.16.1, P2: the word class leaves out vowel signs, so a Hindi or Thai name ending in one had no side."""
    assert sorted(store._extract_entities(text)) == sorted(expected)


@pytest.mark.parametrize("text", ["Bruce Banner aka the Hulk", "Alice Smith aka a friend"])
def test_a_function_word_is_no_alias(store, text):
    assert all(name.lower() not in ("the", "a") for pair in store._aka_pairs(text) for name in pair)


def test_kinship_aliases_stay(store):
    assert store._aka_pairs("Пётр Иванов aka папа") == [("Пётр Иванов", "папа")]


@pytest.mark.parametrize("text,name", [("用“微信pay”付款", "微信pay"), ("买了“小米pad”", "小米pad")])
def test_a_lower_case_word_glued_to_chinese_is_a_name(store, text, name):
    assert name in store._extract_entities(text)


@pytest.mark.parametrize("name,text", [
    ("Docker Hub", "用Docker Hub直连"),
    ("Docker Hub", "на Docker Hubе лежит"),
    ("Docker", "用Docker直连"),
])
def test_a_known_name_glued_to_another_script_is_mentioned(store, name, text):
    """Review 0.16.1, P3: the seam was split when extracting but not when looking for a known name."""
    store.add_fact_status("about " + name, entities=[name])
    assert store.entity_ids(name)[0] in store.mentioned_entity_ids(text)
