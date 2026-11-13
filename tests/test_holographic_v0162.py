"""0.16.2: what 0.16.1 showed on a Chinese store (issue #4) — once quotes beside Chinese were read, buttons, menu items
and steps in quotes became names, and two facts quoting one made it established; names that begin with a number were
refused while some counts passed; cleaning up took one call per name. Every rule in Chinese, English and Russian."""
from __future__ import annotations

import json

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic import store as store_module
from hermes_plugin_holographic import text as text_module
from hermes_plugin_holographic.store import MemoryStore, _valid_entity

DIM = 64


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


@pytest.fixture
def p(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": DIM})
    provider.initialize(session_id="s")
    yield provider
    provider.shutdown()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _status(store, name):
    rows = [r for r in store.list_entities(limit=1000) if r["name"] == name]
    assert rows, f"{name} is no entity: {[r['name'] for r in store.list_entities(limit=1000)]}"
    return rows[0]["status"]


# --- A. a name without capitals that the facts only ever quote stays provisional ---------------------------------

@pytest.mark.parametrize("lang,facts,name", [
    # issue #4: an operation in quotes, in two facts, became established
    ("zh", ["面板里点“自动添加任务并解析元数据”", "又勾选了“自动添加任务并解析元数据”"], "自动添加任务并解析元数据"),
    ("zh", ['watch已加"矮子将军预警"', '"矮子将军预警"又响了'], "矮子将军预警"),
    ("zh", ["面板「变量」页打不开", "「变量」页又空了"], "变量"),
    ("zh", ["他在“华为”工作", "“华为”发布了新手机"], "华为"),  # one word: no mention outside quotes links it
    ("ja", ["「東京大学」に行った", "「東京大学」の入試"], "東京大学"),
])
def test_a_name_without_capitals_only_ever_quoted_stays_provisional(store, lang, facts, name):
    for fact in facts:
        store.add_fact(fact)
    rows = [r for r in store.list_entities(limit=1000) if r["name"] == name]
    assert rows and rows[0]["facts"] == 2 and rows[0]["status"] == "provisional"


@pytest.mark.parametrize("lang,facts,name", [
    # a title in title marks is a name: 《》 count as using it
    ("zh", ["读完了《平凡的世界》", "《平凡的世界》很长"], "平凡的世界"),
    # "X aka Y" names it outside quotes (a one-word Chinese name nobody declared is not found in running text: a
    # quoted label such as 变量 would attach itself to every fact that says the word)
    ("zh", ["他去了“小米科技”", "小米科技 aka Xiaomi 很大"], "小米科技"),
])
def test_a_name_without_capitals_used_as_a_name_becomes_established(store, lang, facts, name):
    for fact in facts:
        store.add_fact(fact)
    assert _status(store, name) == "established"


@pytest.mark.parametrize("lang,facts,name", [
    # names with capitals keep the 0.16.0 rule: two facts are enough
    ("en", ["Code name 'Phoenix' is set", "We shipped 'Phoenix' today"], "Phoenix"),
    ("ru", ["Счёт в «Сбер Бизнес» открыт", "Платёж через «Сбер Бизнес» прошёл"], "Сбер Бизнес"),
    ("zh", ["买了“华为mate60”", "“华为mate60”到了"], "华为mate60"),  # lower-case Latin has case
])
def test_names_with_capitals_are_established_by_two_facts_as_before(store, lang, facts, name):
    for fact in facts:
        store.add_fact(fact)
    assert _status(store, name) == "established"


@pytest.mark.parametrize("lang,fact,name", [
    ("zh", "他在“华为”工作", "华为"),
    ("zh", "面板里点“自动添加任务并解析元数据”", "自动添加任务并解析元数据"),
    ("ja", "「東京大学」に行った", "東京大学"),
])
def test_a_person_naming_it_establishes_it(store, lang, fact, name):
    store.add_fact(fact)
    store.add_fact_status(fact + "。", entities=[name])
    assert _status(store, name) == "established"


def test_an_alias_someone_gives_establishes_a_quoted_name(store):
    store.add_fact("他在“华为”工作")
    store.add_fact("“华为”发布了新手机")
    store.add_alias("华为", "Huawei")
    assert _status(store, "华为") == "established"


def test_quoted_only_names_are_counted_as_provisional_and_listed_for_review(store):
    for fact in ("面板里点“自动添加任务并解析元数据”", "又勾选了“自动添加任务并解析元数据”"):
        store.add_fact(fact)
    assert store.entity_counts()["provisional"] == 1
    store._write("UPDATE entities SET created_at = datetime('now', '-90 days')")
    [row] = store.stale_provisional_entities()
    assert row["entity"] == "自动添加任务并解析元数据" and "only in quotes" in row["reason"]


@pytest.mark.parametrize("text,kept", [
    ("我在小米科技工作", "小米科技"),
    ("读完了《小米科技》", "小米科技"),  # title marks stay
    ("he went to Phoenix", "Phoenix"),
    ("Счёт в Сбер Бизнес", "Сбер Бизнес"),
])
def test_outside_plain_quotes_keeps_running_text_and_titles(text, kept):
    assert kept in store_module._outside_plain_quotes(text)


@pytest.mark.parametrize("text", [
    "他去了“小米科技”", '他去了"小米科技"', "他去了「小米科技」", "他去了『小米科技』", "he went to 'Phoenix'",
    "Счёт в «Сбер Бизнес»", "`小米科技` 是代码",
    # review 0.16.2: a quote of any length, nested, never closed, full-width
    "他说“" + "测试" * 35 + "小米科技的按钮失效”", "「他说“小米科技”好」", "他说“小米科技的按钮", "他说＂小米科技＂",
])
def test_outside_plain_quotes_blanks_every_plain_quote(text):
    assert "小米科技" not in store_module._outside_plain_quotes(text) and "Phoenix" not in \
        store_module._outside_plain_quotes(text) and "Сбер" not in store_module._outside_plain_quotes(text)


def test_outside_plain_quotes_is_linear():
    import time
    text = "“a" * 20_000 + "”" + '"' * 20_000
    t = time.perf_counter()
    store_module._outside_plain_quotes(text)
    assert time.perf_counter() - t < 1.0


def test_tilde_fences_are_read_in_linear_time():
    """Review 0.16.2 (round 2): every ~~~ fence looked at every later line; 80,000 lines took 23 s."""
    import time
    t = time.perf_counter()
    store_module._without_code("~~~\n" * 80_000)
    assert time.perf_counter() - t < 2.0
    assert store_module._without_code("~~~\nAlice Smith\n~~~\nBob Brown").endswith("Bob Brown")
    assert "Alice Smith" in store_module._without_code("~~~\nAlice Smith")  # a lone rule is no block


@pytest.mark.parametrize("text", [
    # review 0.16.2 (round 2): a quote nested in one of its own kind, a single quote never closed, curly single quotes
    "زار «كلام «نص» أحمد علي»", "他说“甲“乙”小米科技”", "زار 'أحمد علي اليوم", "زار ‘أحمد علي’", "he said 'Phoenix",
])
def test_nested_and_unclosed_quotes_are_blanked(text):
    bare = store_module._outside_plain_quotes(text)
    assert "أحمد" not in bare and "小米科技" not in bare and "Phoenix" not in bare


@pytest.mark.parametrize("text,kept", [
    ("don't tell Alice Smith", "Alice Smith"), ("O'Brien met D'Angelo and Alice Smith", "Alice Smith"),
    ("I don’t know Alice Smith", "Alice Smith"), ("'Phoenix' and Alice Smith", "Alice Smith"),
])
def test_apostrophes_inside_words_are_no_quotes(text, kept):
    assert kept in store_module._outside_plain_quotes(text)


@pytest.mark.parametrize("lang,facts,name", [
    # review 0.16.2: a long, nested or unclosed quote of the name, or a full-width quote, is no use as a name
    ("zh", ["点“小米科技”", "再点“小米科技”，他说“" + "测试" * 35 + "小米科技的按钮失效”"], "小米科技"),
    ("zh", ["点“小米科技”", "「他说“小米科技”好」"], "小米科技"),
    ("ar", ["زار «أحمد علي»", "زار ＂أحمد علي＂ اليوم"], "أحمد علي"),
    ("ar", ["زار «نور علي»", "«نور علي»; نور علياء; 测试"], "نور علي"),  # Han elsewhere in the fact proves nothing
    ("zh", ["读完了《平凡的世界》"], "平凡的世界"),  # one fact is one fact, title or not
    ("ar", ["زار «أحمد علي»", "زار «كلام «نص» أحمد علي»"], "أحمد علي"),  # round 2
    ("zh", ["点“小米科技”", "再点“小米科技”，他说“甲“乙”《小米科技》”"], "小米科技"),
    ("ar", ["زار «أحمد علي»", "زار 'أحمد علي اليوم"], "أحمد علي"),
    # review 0.16.2 (round 2, Antigravity): an apostrophe inside single quotes, curly single quotes
    ("ar", ["زار «أحمد علي»", "he said 'it's fine: أحمد علي' today"], "أحمد علي"),
    ("ar", ["زار «أحمد علي»", "زار ‘أحمد علي’"], "أحمد علي"),
    # a one-word Chinese name nobody declared is not looked up in running text, even in a fact that quotes it too:
    # a quoted label is written bare as often (面板「变量」页 … 变量很多)
    ("zh", ["他去了“小米科技”", "“小米科技”发布了新手机，大家都在小米科技买到了"], "小米科技"),
])
def test_quotes_never_establish_a_name_without_capitals(store, lang, facts, name):
    for fact in facts:
        store.add_fact(fact)
    assert _status(store, name) == "provisional"


@pytest.mark.parametrize("lang,facts,name", [
    # the same readers as linking: an Arabic name with a prefix is mentioned (review 0.16.2)
    ("ar", ["زار «أحمد علي»", "تحدث لأحمد علي"], "أحمد علي"),
    ("zh", ["他去了“小米科技”", "小米科技 aka Xiaomi 很大"], "小米科技"),
    # an alias with case outside quotes, compared as keys compare (review 0.16.2, round 2)
    ("ar", ["“أحمد aka Straße”", "“أحمد” STRASSE"], "أحمد"),
    ("ar", ["“أحمد aka Friedrich Straße”", "“أحمد” FRIEDRICH STRASSE"], "أحمد"),
    ("zh", ["“华为 aka Huawei”", "“华为”发布了新的HUAWEI手机"], "华为"),
])
def test_a_use_outside_quotes_that_links_establishes_it(store, lang, facts, name):
    for fact in facts:
        store.add_fact(fact)
    assert _status(store, name) == "established"


def test_the_standing_keeps_no_text_of_a_forgotten_fact(store):
    """Review 0.16.2 (round 2): the cache of what each text names kept every text it had read, forgotten ones too."""
    store.add_fact("زار «أحمد علي»")
    store.add_fact("زار «أحمد علي» أمس")
    store.list_entities()
    gone = [store.add_fact(f"خبر {i} «أحمد علي»") for i in range(20)]
    store.list_entities()
    for fact_id in gone:
        store.forget_fact(fact_id)
    store.list_entities()
    kept = [content for content, _, _ in store._entry["outside"][1].values()]
    assert len(kept) == 2 and not any(c.startswith("خبر") for c in kept)


def test_the_standing_follows_an_edited_fact(store):
    """Review 0.16.2 (round 2, Antigravity): an edit in the same connection bumps neither the generation nor
    PRAGMA data_version; what a text names is kept by fact and content, so the edit is read."""
    store.add_fact("زار «أحمد علي»")
    second = store.add_fact("زار «أحمد علي» أمس")
    assert _status(store, "أحمد علي") == "provisional"
    store.update_fact(second, content="تحدث لأحمد علي أمس")
    assert _status(store, "أحمد علي") == "established"
    store.update_fact(second, content="زار «أحمد علي» أمس")
    assert _status(store, "أحمد علي") == "provisional"


def test_standing_reads_each_fact_once(store):
    import time
    long = "很长的说明文字" * 3000
    with store._tx():
        a = store.add_fact("点了“按钮甲”" + long)
        b = store.add_fact("又点了“按钮甲”" + long)
        for i in range(1000):
            entity = store._write("INSERT INTO entities (name, name_key) VALUES (?, ?)",
                                  (f"按钮{i}号", store_module.entity_key(f"按钮{i}号"))).lastrowid
            for fact_id in (a, b):
                store._write("INSERT INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (fact_id, entity))
    t = time.perf_counter()
    assert store.entity_counts()["provisional"] == 1001
    assert time.perf_counter() - t < 3.0
    t = time.perf_counter()
    store.list_entities()
    assert time.perf_counter() - t < 1.0  # what the texts name is kept until an entity changes


# --- B. a Chinese name that begins with a number; a count is none ------------------------------------------------

@pytest.mark.parametrize("name", ["360安全卫士", "115网盘", "36氪", "1905电影网", "2345好压", "101次求婚",
                                  "1980年代的爱情", "17直播", "12.1槍殺大案"])
def test_a_chinese_name_that_begins_with_a_number_is_a_name(name):
    # measured on 26,308 Chinese labels of real names in Wikidata: the share of letters refused 66 of the 134 that
    # begin with a number (CHANGELOG 0.16.2)
    assert store_module._name_shaped(name, quoted=True)


@pytest.mark.parametrize("lang,span", [
    ("zh", "3个月"), ("zh", "2分钟"), ("zh", "15分钟"), ("zh", "15分鐘"), ("zh", "2小时"), ("zh", "2天后"),
    ("zh", "3年前"), ("zh", "100多人"), ("zh", "10块钱"), ("zh", "5号楼"), ("zh", "1 小时"), ("zh", "３个月"),
    ("zh", "2.5小时"), ("zh", "3个小时"), ("zh", "2小时30分钟"), ("zh", "3万年"), ("zh", "1亿人"),
    ("en", "14 days"), ("en", "3 months"),  # as 0.16.1
    ("ru", "14 дней"), ("ru", "3 месяца"),
])
def test_a_count_is_no_name(lang, span):
    assert not store_module._name_shaped(span, quoted=True)


@pytest.mark.parametrize("lang,text", [
    ("zh", "等了“3个月”"), ("zh", '等了"2小时"'), ("zh", "过了「15分钟」"),
    ("en", 'waited "14 days"'), ("ru", "ждали «14 дней»"),
])
def test_a_count_in_quotes_gives_no_entity(store, lang, text):
    assert store._extract_entities(text) == []


@pytest.mark.parametrize("lang,text,name", [
    ("zh", "装了“360安全卫士”", "360安全卫士"),
    ("zh", '文件在"115网盘"', "115网盘"),
    ("en", 'shop at "7-Eleven"', "7-Eleven"),  # as 0.16.1
    ("ru", "работаем в «1С:Предприятие 8.3»", "1С:Предприятие 8.3"),
])
def test_a_quoted_name_that_begins_with_a_number_gives_an_entity(store, lang, text, name):
    assert name in store._extract_entities(text)


@pytest.mark.parametrize("span,count", [
    ("3个月", True), ("15分鐘", True), ("1 小时", True), ("100多人", True),
    ("101次求婚", False), ("360安全卫士", False), ("三个月", False),  # Chinese numerals: not by this rule (五月天)
    ("3", False), ("个月", False), ("3 months", False), ("五月天", False),
    # review 0.16.2: counts of several parts and with multipliers
    ("2小时30分钟", True), ("3万年", True), ("1亿人", True), ("2万元", True), ("15万分钟", True), ("1000万", True),
    ("360百科", False), ("3万", True), ("1万2千人", True), ("3千5百元", True),  # round 2
])
def test_zh_count(span, count):
    assert text_module.zh_count(span) is count


def test_a_count_someone_gives_is_still_accepted():
    assert _valid_entity("3个月", given=True)


def test_a_store_indexed_under_the_0161_rules_is_rebuilt(tmp_path):
    """The rules changed: a store 0.16.1 indexed (rules 3) is rebuilt on open; a count goes, a name that begins with a
    number comes."""
    path = tmp_path / "facts.db"
    s = MemoryStore(str(path), hrr_dim=DIM)
    s.add_fact("装了“360安全卫士”")
    count_fact = s.add_fact("等了“3个月”")
    cur = s._write("INSERT INTO entities (name, name_key) VALUES (?, ?)", ("3个月", store_module.entity_key("3个月")))
    s._write("INSERT INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (count_fact, cur.lastrowid))
    s._write("DELETE FROM fact_entities WHERE entity_id IN (SELECT entity_id FROM entities WHERE name = '360安全卫士')")
    s._write("DELETE FROM entities WHERE name = '360安全卫士'")
    s._write("UPDATE meta SET value = '3' WHERE key = 'entity_rules'")
    s.close()
    s = MemoryStore(str(path), hrr_dim=DIM)
    assert s.entity_rules() == store_module._ENTITY_RULES == 5
    assert [r["name"] for r in s.list_entities()] == ["360安全卫士"]
    s.close()


# --- C. several names in one delete_entity call ------------------------------------------------------------------

def test_delete_entity_takes_several_names(p):
    for fact in ("面板「变量」页打不开", "勾选了“自动添加任务并解析元数据”", "Code name 'Phoenix' is set",
                 "Счёт в «Сбер Бизнес» открыт"):
        _call(p, action="add", content=fact)
    out = _call(p, action="delete_entity", entities=["变量", "自动添加任务并解析元数据", "Сбер Бизнес", "nobody"])
    assert [d["entity"] for d in out["deleted"]] == ["变量", "自动添加任务并解析元数据", "Сбер Бизнес"]
    assert out["not_found"] == ["nobody"] and out["scrub"] == "done"
    assert all(d["facts_deleted"] == 0 for d in out["deleted"])
    assert [r["name"] for r in p._store.list_entities()] == ["Phoenix"]
    assert p._store.count_facts() == 4
    _call(p, action="add", content="又打开了「变量」页")  # the tombstones hold
    assert [r["name"] for r in p._store.list_entities()] == ["Phoenix"]


def test_delete_entity_with_one_name_answers_as_before(p):
    _call(p, action="add", content="Code name 'Phoenix' is set")
    out = _call(p, action="delete_entity", entity="Phoenix")
    assert out["deleted"] is True and out["entity"] == "Phoenix"


def test_delete_entity_with_no_known_name_changes_nothing(p):
    _call(p, action="add", content="Code name 'Phoenix' is set")
    out = _call(p, action="delete_entity", entities=["nobody", "никто", "没有人"])
    assert "Nothing was changed" in out["error"]
    assert [r["name"] for r in p._store.list_entities()] == ["Phoenix"]


def test_delete_entity_takes_at_most_a_hundred_names(p):
    _call(p, action="add", content="Code name 'Phoenix' is set")
    out = _call(p, action="delete_entity", entities=["Phoenix"] + [f"n{i}" for i in range(100)])
    assert "at most 100" in out["error"] and "Nothing was changed" in out["error"]
    assert [r["name"] for r in p._store.list_entities()] == ["Phoenix"]


def test_delete_entity_with_a_name_and_its_alias(p):
    _call(p, action="add", content="他在“华为”工作")
    _call(p, action="alias", entity="华为", content="Huawei")
    out = _call(p, action="delete_entity", entities=["华为", "Huawei"])
    assert [d["entity"] for d in out["deleted"]] == ["华为"] and out["covered"] == ["Huawei"] and out["not_found"] == []
    assert p._store.list_entities() == []


@pytest.mark.parametrize("entities", [
    ["Phoenix", 42], ["Phoenix", None], ["Phoenix", ""], ["Phoenix", "  "], {"Phoenix": False}, 42,
    ["Phoenix"] * 101,  # the length is checked before duplicates go (review 0.16.2)
])
def test_delete_entity_refuses_a_bad_list_whole(p, entities):
    _call(p, action="add", content="Code name 'Phoenix' is set")
    _call(p, action="add", content="'Phoenix' ships today")
    out = _call(p, action="delete_entity", entities=entities)
    assert "Nothing was changed" in out["error"]
    assert [r["name"] for r in p._store.list_entities()] == ["Phoenix"]


def test_delete_entity_refuses_a_name_of_two_entities_on_both_paths(p):
    """Review 0.16.2 (round 2): the batch refused a name of two entities and advised the single call, which deleted
    both."""
    _call(p, action="add", content="Alice Smith met Bob Brown")
    store = p._store
    bob = store.entity_ids("Bob Brown")[0]
    store._write("INSERT INTO entity_aliases (alias_key, entity_id, alias) VALUES (?, ?, ?)",
                 (store_module.entity_key("Alice Smith"), bob, "Alice Smith"))
    if len(store.entity_ids("Alice Smith")) < 2:
        pytest.skip("this store resolves the name to one entity")
    for args in ({"entity": "Alice Smith"}, {"entities": ["Alice Smith"]}):
        out = _call(p, action="delete_entity", **args)
        assert "more than one entity" in out["error"] and "one at a time" not in out["error"]
        assert len(store.list_entities()) == 2


def test_delete_entity_refuses_a_name_of_two_entities(store):
    """Review 0.16.2: in a store an older version wrote, a name can be one entity's name and another's alias; a list
    deleted both and named one."""
    store.add_fact("Alice Smith met Bob Brown")
    bob = store.entity_ids("Bob Brown")[0]
    store._write("INSERT INTO entity_aliases (alias_key, entity_id, alias) VALUES (?, ?, ?)",
                 (store_module.entity_key("Alice Smith"), bob, "Alice Smith"))
    if len(store.entity_ids("Alice Smith")) < 2:
        pytest.skip("this store resolves the name to one entity")
    with pytest.raises(ValueError, match="more than one entity"):
        store.delete_entities(["Alice Smith"])
    assert len(store.list_entities()) == 2


def test_delete_entity_is_a_write_action():
    from hermes_plugin_holographic import _WRITE_ACTIONS
    assert "delete_entity" in _WRITE_ACTIONS
