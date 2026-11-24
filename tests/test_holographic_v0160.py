"""0.16.0: which names are entities (issue #4). A name guessed from a fact's text must be shaped like one; a guessed
name nobody confirmed is provisional; entities can be deleted (with a tombstone) and renamed; a store indexed under
older rules is rebuilt. Every rule is checked in Chinese, English and Russian."""
from __future__ import annotations

import json
import sqlite3

import pytest

from hermes_plugin_holographic import _WRITE_ACTIONS, HolographicMemoryProvider
from hermes_plugin_holographic import store as store_module
from hermes_plugin_holographic import text as text_module
from hermes_plugin_holographic.store import MemoryStore

DIM = 64


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "facts.db"), hrr_dim=DIM)
    yield s
    s.close()


def _provider(tmp_path, **kwargs):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": DIM})
    p.initialize(session_id="s", **kwargs)
    return p


@pytest.fixture
def p(tmp_path):
    provider = _provider(tmp_path)
    yield provider
    provider.shutdown()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _names(store, fact_id):
    return sorted(r[0] for r in store.read("SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
                                           "WHERE fe.fact_id = ?", (fact_id,)))


def _status(store, name):
    rows = [r for r in store.list_entities(limit=1000) if r["name"] == name]
    return rows[0]["status"] if rows else None


def _assert_consistent(store):
    """No link or alias points at a missing fact or entity, and no two entities share a key."""
    assert not store.read("SELECT 1 FROM fact_entities WHERE entity_id NOT IN (SELECT entity_id FROM entities)")
    assert not store.read("SELECT 1 FROM fact_entities WHERE fact_id NOT IN (SELECT fact_id FROM facts)")
    assert not store.read("SELECT 1 FROM entity_aliases WHERE entity_id NOT IN (SELECT entity_id FROM entities)")
    assert not store.read("SELECT name_key FROM entities WHERE name_key IS NOT NULL GROUP BY name_key HAVING COUNT(*) > 1")


# --- the gate: what text gives as a name -------------------------------------------------------------------------

DROPPED = [
    # zh: three or more function words, a sentence-final particle, a courtesy, a version, a quotation
    ("zh", '用户说“先去搜索再告诉我答案”'),
    ("zh", "他回“知道了”"),
    ("zh", "她说“好的呀”"),
    ("zh", "回复“不用了”"),
    ("zh", "他说“明天再说吧”"),
    ("zh", "她说“你先去看看吧”"),
    ("zh", "他说“谢谢”"),
    ("zh", "升级到 “0.18.0”"),
    ("zh", "他说“今天下午三点我们在公司门口集合然后一起出发”"),
    # en: lower case without an identifier, a version
    ("en", 'We said "count" and "max" and "sum"'),
    ("en", "she said 'run fast'"),
    ("en", "it printed 'uv trampoline failed'"),
    ("en", 'Hermes "0.17.0 installed"'),
    ("en", 'She said "Good morning"'),
    ("en", 'He said "Thank you"'),
    # ru
    ("ru", "Он сказал «быстро беги»"),
    ("ru", "Слово «сумма» в отчёте"),
    ("ru", "Поставили «версия 0.17.0»"),
    ("ru", "Поставили «0.17.0»"),
    ("ru", "Написала «Доброе утро»"),
    ("ru", "Ответил «Спасибо большое»"),
]

KEPT = [
    ("zh", "他在“华为”工作", "华为"),
    ("zh", "他考上了“北京大学”", "北京大学"),
    ("zh", "“海底捞”很好吃", "海底捞"),
    ("zh", "“张伟”是同事", "张伟"),
    ("zh", "他开“别克”", "别克"),  # 别 is a function word, but 别克 is one word
    ("zh", "楼下有个“网吧”", "网吧"),
    ("zh", "这叫“他山之石”", "他山之石"),
    ("zh", "她在“中国人民解放军总医院”上班", "中国人民解放军总医院"),
    ("zh", "他去过“臺灣大學”", "臺灣大學"),
    ("zh", "买了“华为 P60”", "华为 P60"),
    # short names with a function word as a word of their own (review 0.16.0): under five Han characters it stays
    ("zh", "用“有道”查词", "有道"),
    ("zh", "在“去哪儿”订票", "去哪儿"),
    ("zh", "在“去哪儿网”订票", "去哪儿网"),
    ("zh", "叫了“饿了么”", "饿了么"),
    ("zh", "采访了“向太”", "向太"),
    ("zh", "“我的世界”很好玩", "我的世界"),
    # titles and companies with one or two function words (review 0.16.0, Antigravity)
    ("zh", "看了“平凡的世界”", "平凡的世界"),
    ("zh", "看了“人民的名义”", "人民的名义"),
    ("zh", "看了“后来的我们”", "后来的我们"),
    ("zh", "看了“不能说的秘密”", "不能说的秘密"),
    ("zh", "看了“我和我的祖国”", "我和我的祖国"),
    ("zh", "看了“这个杀手不太冷”", "这个杀手不太冷"),
    ("zh", "看了“被嫌弃的松子的一生”", "被嫌弃的松子的一生"),
    ("zh", "入职“去哪儿网科技有限公司”", "去哪儿网科技有限公司"),
    ("zh", "入职“有赞科技有限公司”", "有赞科技有限公司"),
    ("zh", "入职“和记黄埔有限公司”", "和记黄埔有限公司"),
    ("zh", "找了“我爱我家房地产”", "我爱我家房地产"),
    ("en", 'Code name "Phoenix"', "Phoenix"),
    ("en", "the board in 'kanban.db'", "kanban.db"),
    ("en", "use 'gpt-4o' for drafts", "gpt-4o"),
    ("en", "the model 'gpt-3.5-turbo'", "gpt-3.5-turbo"),
    ("en", "ssh into 'qx-vps'", "qx-vps"),
    ("en", 'He bought "iPhone 15 Pro"', "iPhone 15 Pro"),
    ("en", 'It runs on "Windows 11"', "Windows 11"),
    # a version beside a word that names a thing (review 0.16.0, Antigravity)
    ("en", 'We moved to "Ubuntu 24.04 LTS"', "Ubuntu 24.04 LTS"),
    ("en", '"Python 3.12" installed', "Python 3.12"),
    ("en", 'Watched "Good Morning America"', "Good Morning America"),
    ("en", 'Bought a "Hello Kitty" mug', "Hello Kitty"),
    ("en", "Alice Smith works at Bank Of America", "Bank Of America"),
    ("ru", "Счёт в «Сбер Бизнес»", "Сбер Бизнес"),
    ("ru", "Вчера Мария Иванова звонила", "Мария Иванова"),
    ("ru", "Сервер «qx-vps» перезагружен", "qx-vps"),
    ("ru", "Кафе «Дом у Дороги» закрыто", "Дом у Дороги"),
    ("ru", "Бухгалтерия в «1С:Предприятие 8.3»", "1С:Предприятие 8.3"),
]


@pytest.mark.parametrize("lang,text", DROPPED)
def test_the_gate_drops_what_is_not_a_name(store, lang, text):
    assert store._extract_entities(text) == []


@pytest.mark.parametrize("lang,text,name", KEPT)
def test_the_gate_keeps_names(store, lang, text, name):
    assert name in store._extract_entities(text)


@pytest.mark.parametrize("lang,text", [
    ("en", 'we run "nginx" and "redis"'),
    ("ru", "купили на «авито» через «яндекс»"),
])
def test_a_lower_case_name_in_quotes_is_a_documented_loss(store, lang, text):
    """Written in lower case, a name looks like any word ('count'); the agent names it with entities=[...]."""
    assert store._extract_entities(text) == []


@pytest.mark.parametrize("lang,text,name", [
    ("zh", "他说“手动巡检”", "手动巡检"),  # a phrase in form no rule tells from a name: provisional, not dropped
    ("zh", "用户说“以后再弄”", "以后再弄"),  # one function word: as likely a brand (饿了么)
    ("zh", "他说“我们以后再弄”", "我们以后再弄"),  # two: as likely a title (后来的我们)
    ("zh", "他说“没问题”", "没问题"),
    ("en", 'he said "Call Me"', "Call Me"),
    ("ru", "написал «Давай Потом»", "Давай Потом"),
    ("en", "a 'Quick Check' later", "Quick Check"),
    ("ru", "назвали «Быстрая Проверка»", "Быстрая Проверка"),
])
def test_what_only_reads_as_a_name_passes_the_gate_and_is_provisional(store, lang, text, name):
    assert store._extract_entities(text) == [name]
    store.add_fact(text)
    assert _status(store, name) == "provisional"


def test_a_given_name_skips_the_gate(store):
    """entities=[...] is someone naming it: no rule of the gate applies (only the check for paths, flags, URLs)."""
    fact, _ = store.add_fact_status("Played it all weekend", entities=["我的世界", "nginx", "сумма"])
    assert _names(store, fact) == ["nginx", "сумма", "我的世界"]


def test_without_the_chinese_word_list_the_function_word_rule_is_off(store, monkeypatch):
    """zh_words gives single characters without its dictionary: a list of words would become a list of characters
    and drop 别克 and 网吧, so the rule does not apply (the length rule still does)."""
    empty = text_module._ZhTable({}, {}, 0.0, 0.0, frozenset())
    monkeypatch.setattr(text_module, "_zh_table", lambda: empty)
    assert text_module.zh_phrase("先去搜索再告诉我答案") is False
    assert store._extract_entities("他开“别克”") == ["别克"]
    assert store._extract_entities("他说“今天下午三点我们在公司门口集合然后一起出发”") == []


def test_zh_phrase_is_about_chinese_only():
    assert text_module.zh_phrase("先去搜索再告诉我答案")
    assert not text_module.zh_phrase("北京大学")
    assert not text_module.zh_phrase("私の名前")  # Japanese (kana): not judged by Chinese words
    assert not text_module.zh_phrase("быстро беги")


@pytest.mark.parametrize("text,expected", [
    ("user42 aka bob", ["user42", "bob"]),  # aka sides are not quoted: lower case stays
    ("Alice Smith aka Bob Brown works here", ["Alice Smith", "Bob Brown"]),
    ("Hermes aka 0.17.0", []),
    ("Мария aka Маша", ["Мария", "Маша"]),
])
def test_aka_sides_pass_the_gate_without_the_quoted_rules(store, text, expected):
    assert store._extract_entities(text) == expected


# --- standing: established and provisional -----------------------------------------------------------------------

@pytest.mark.parametrize("first,second,name", [
    # 0.16.2: a name without capitals in plain quotes stays provisional however many facts quote it; a title in
    # title marks is used as a name (test_holographic_v0162)
    ("他读了《平凡的世界》", "《平凡的世界》很长", "平凡的世界"),
    ('Code name "Phoenix"', '"Phoenix" ships in May', "Phoenix"),
    ("Счёт в «Сбер Бизнес»", "«Сбер Бизнес» закрыли", "Сбер Бизнес"),
])
def test_a_second_fact_makes_a_name_established_and_losing_it_makes_it_provisional_again(store, first, second, name):
    store.add_fact(first)
    assert _status(store, name) == "provisional"
    other = store.add_fact(second)
    assert _status(store, name) == "established"
    store.forget_fact(other)
    assert _status(store, name) == "provisional"
    _assert_consistent(store)


@pytest.mark.parametrize("text,name", [("他在“华为”工作", "华为"), ('Code name "Phoenix"', "Phoenix"),
                                       ("Счёт в «Сбер Бизнес»", "Сбер Бизнес")])
def test_declaring_or_aliasing_a_name_makes_it_established(tmp_path, text, name):
    declared = MemoryStore(str(tmp_path / "declared.db"), hrr_dim=DIM)
    fact = declared.add_fact(text)
    declared.update_fact(fact, entities=[name])
    assert _status(declared, name) == "established"
    declared.close()
    aliased = MemoryStore(str(tmp_path / "aliased.db"), hrr_dim=DIM)
    aliased.add_fact(text)
    aliased.add_alias(name, name + " 2")
    assert _status(aliased, name) == "established"
    aliased.close()


def test_an_alias_found_in_text_does_not_make_a_name_established(store):
    store.add_fact("Robert Brown aka Bobby fixed the roof")
    assert _status(store, "Robert Brown") == "provisional"
    assert store.read("SELECT source FROM entity_aliases WHERE alias = 'Bobby'")[0][0] == "aka"


def test_an_alias_given_in_text_after_one_given_by_hand_stays_given(store):
    store.add_alias("Robert Brown", "Bobby")
    store.add_fact("Robert Brown aka Bobby fixed the roof")
    assert store.read("SELECT source FROM entity_aliases WHERE alias = 'Bobby'")[0][0] == "user"


def test_an_aka_entity_goes_with_its_last_fact_and_a_given_alias_keeps_it(store):
    fact = store.add_fact("Robert Brown aka Bobby fixed the roof")
    store.update_fact(fact, content="The roof was fixed")
    assert _status(store, "Robert Brown") is None
    store.add_alias("Anna Lee", "Annie")
    assert _status(store, "Anna Lee") == "established"  # an alias someone gave, no fact yet: kept


def test_entities_lists_status_and_counts_provisional_names_across_the_whole_store(p):
    _call(p, action="add", content="他在“华为”工作")
    _call(p, action="add", content='Code name "Phoenix"')
    _call(p, action="add", content="Счёт в «Сбер Бизнес»")
    _call(p, action="add", content="«Сбер Бизнес» закрыли")
    listed = _call(p, action="entities", limit=1)
    assert listed["count"] == 1 and listed["provisional_entities"] == 2 and "delete_entity" in listed["hint"]
    assert {r["name"] for r in _call(p, action="entities", status="provisional")["entities"]} == {"华为", "Phoenix"}
    assert [r["name"] for r in _call(p, action="entities", status="established")["entities"]] == ["Сбер Бизнес"]
    assert "error" in _call(p, action="entities", status="maybe")


def test_review_lists_stale_provisional_names_apart_from_facts(p):
    _call(p, action="add", content="他说“手动巡检”")
    _call(p, action="add", content='Code name "Phoenix"')
    assert _call(p, action="review")["entity_candidates"] == []
    p._store._write("UPDATE entities SET created_at = datetime('now', '-90 days') WHERE name = '手动巡检'")
    review = _call(p, action="review")
    assert [c["entity"] for c in review["entity_candidates"]] == ["手动巡检"]
    assert all("fact_id" in c for c in review["candidates"])
    assert "delete_entity" in review["hint"]


def test_stats_reports_provisional_names_and_the_index_state(p):
    _call(p, action="add", content='Code name "Phoenix"')
    stats = _call(p, action="stats")
    assert stats["entities"] == 1 and stats["provisional_entities"] == 1 and stats["entity_index"] == "current"


# --- delete_entity -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,name,again", [
    ("用户说“手动巡检”", "手动巡检", "今天又说“手动巡检”"),
    ("we said 'Quick Check' twice", "Quick Check", 'then "Quick Check" again'),
    ("назвали «Быстрая Проверка»", "Быстрая Проверка", "опять «Быстрая Проверка»"),
])
def test_delete_entity_keeps_every_fact_and_text_never_brings_the_name_back(p, text, name, again):
    fact = _call(p, action="add", content=text)["fact_id"]
    before = p._store.read("SELECT content, trust_score, hrr_vector FROM facts WHERE fact_id = ?", (fact,))[0]
    versions = p._store.read("SELECT COUNT(*) FROM fact_versions")[0][0]
    result = _call(p, action="delete_entity", entity=name)
    assert result == {"entity": name, "deleted": True, "facts_unlinked": 1, "aliases_dropped": 0, "facts_deleted": 0,
                      "scrub": "done"}
    after = p._store.read("SELECT content, trust_score FROM facts WHERE fact_id = ?", (fact,))[0]
    assert tuple(after) == tuple(before)[:2]
    assert p._store.read("SELECT COUNT(*) FROM fact_versions")[0][0] == versions
    assert [r["content"] for r in _call(p, action="probe", entity=name)["results"]] == [text]  # falls back to search
    _call(p, action="add", content=again)
    assert _status(p._store, name) is None
    assert name not in json.dumps([tuple(r) for r in p._store.read("SELECT * FROM entity_tombstones")], ensure_ascii=False)
    assert p._store.read("SELECT COUNT(*) FROM entity_tombstones")[0][0] == 1
    _assert_consistent(p._store)


def test_delete_entity_by_an_alias_takes_the_name_and_every_alias_and_aka_cannot_bring_them_back(p):
    _call(p, action="add", content="Robert Brown fixed the roof")
    _call(p, action="alias", entity="Robert Brown", content="Bobby")
    assert _call(p, action="delete_entity", entity="bobby")["aliases_dropped"] == 1
    fact = _call(p, action="add", content="Robert Brown aka Bobby painted the fence")["fact_id"]
    assert _names(p._store, fact) == []


def test_a_name_given_again_lifts_the_tombstone(p):
    _call(p, action="add", content='Code name "Phoenix"')
    _call(p, action="delete_entity", entity="Phoenix")
    fact = _call(p, action="add", content='"Phoenix" ships in May')["fact_id"]
    assert _names(p._store, fact) == []
    _call(p, action="update", fact_id=fact, entities=["Phoenix"])
    assert _names(p._store, fact) == ["Phoenix"]
    later = _call(p, action="add", content='Then "Phoenix" slipped to June')["fact_id"]
    assert _names(p._store, later) == ["Phoenix"]  # text names it again now


def test_alias_and_rename_lift_the_tombstone_too(p):
    _call(p, action="add", content="Счёт в «Сбер Бизнес»")
    _call(p, action="delete_entity", entity="Сбер Бизнес")
    _call(p, action="alias", entity="Сбербанк", content="Сбер Бизнес")
    fact = _call(p, action="add", content="«Сбер Бизнес» закрыли")["fact_id"]
    assert _names(p._store, fact) == ["Сбербанк"]
    _call(p, action="add", content='Code name "Phoenix"')
    _call(p, action="delete_entity", entity="Phoenix")
    _call(p, action="add", content='Code name "Firebird"')
    _call(p, action="rename", entity="Firebird", content="Phoenix")
    fact = _call(p, action="add", content='"Phoenix" ships in May')["fact_id"]
    assert _names(p._store, fact) == ["Phoenix"]


def test_delete_entity_answers_for_an_unknown_name_and_a_missing_argument(p):
    assert "no entity is named" in _call(p, action="delete_entity", entity="Nobody Here")["error"]
    assert "Missing required argument" in _call(p, action="delete_entity")["error"]


def test_entity_actions_are_writes(tmp_path):
    assert {"rename", "delete_entity", "rebuild_entities"} <= _WRITE_ACTIONS
    writer = _provider(tmp_path)
    _call(writer, action="add", content='Code name "Phoenix"')
    reader = _provider(tmp_path, agent_context="subagent")
    for args in ({"action": "delete_entity", "entity": "Phoenix"},
                 {"action": "rename", "entity": "Phoenix", "content": "Firebird"},
                 {"action": "rebuild_entities"}):
        assert "read-only" in _call(reader, **args)["error"]
    assert _status(writer._store, "Phoenix") == "provisional"
    reader.shutdown()
    writer.shutdown()


def _raw_bytes(path) -> bytes:
    data = b""
    for suffix in ("", "-wal"):
        try:
            with open(f"{path}{suffix}", "rb") as fh:
                data += fh.read()
        except FileNotFoundError:
            pass
    return data


@pytest.mark.parametrize("name,alias", [("Robert Brown", "Xylophone Zymurgy"), ("Мария Иванова", "Ксилофон Зимургия"),
                                        ("北京大学", "燕园秘名")])
def test_delete_entity_leaves_no_copy_of_a_name_no_fact_says_in_the_file(tmp_path, name, alias):
    """Review 0.16.0: the alias was gone from the tables but its bytes stayed in freed pages until a rewrite."""
    path = tmp_path / "facts.db"
    s = MemoryStore(str(path), hrr_dim=DIM)
    s.add_fact(f'"{name}" fixed the roof')
    s.add_alias(name, alias)
    s._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert alias.encode() in _raw_bytes(path)
    result = s.delete_entity(name)
    assert result["scrub"] == "done" and alias.encode() not in _raw_bytes(path)
    s.close()


def test_forget_does_not_leave_a_cached_name_that_links_new_facts_to_a_deleted_entity(tmp_path):
    """Review 0.16.0: _erase deleted an entity with a user alias without invalidating the cached index of names, so
    the next fact mentioning it was linked to an id that no longer existed — through any instance on the file."""
    path = str(tmp_path / "facts.db")
    one, two = MemoryStore(path, hrr_dim=DIM), MemoryStore(path, hrr_dim=DIM)
    fact = one.add_fact("Robert Brown fixed the roof")
    one.add_alias("Robert Brown", "Bob Brown")
    two.mentioned_entity_ids("Robert Brown")  # warm the shared cache
    one.forget_fact(fact)
    two.add_fact("met robert brown yesterday")
    _assert_consistent(one)
    one.close()
    two.close()


@pytest.mark.parametrize("said,other,alias", [
    ("Robert Brown aka Secret Alias fixed the roof", "Robert Brown likes tea", "Secret Alias"),
    ("Мария Иванова aka Тайное Имя звонила", "Мария Иванова любит чай", "Тайное Имя"),
    ("北京大学 aka 燕园秘名 很美", "“北京大学”在海淀", "燕园秘名"),
])
def test_an_alias_from_x_aka_y_goes_with_the_fact_that_said_it(tmp_path, said, other, alias):
    """Review 0.16.0: forgetting the only fact that said "X aka Y" left Y as X's alias (in the table, the legacy
    column and the list), and a rebuild kept it. A forgotten text must not live on as a name."""
    path = tmp_path / "facts.db"
    s = MemoryStore(str(path), hrr_dim=DIM)
    gone = s.add_fact(said)
    s.add_fact(other)
    assert alias in s.list_entities()[0]["aliases"]
    s.forget_fact(gone)
    rows = s.list_entities()
    assert [r["aliases"] for r in rows] == [[]] and alias not in (s.read("SELECT aliases FROM entities")[0][0] or "")
    assert s.entity_ids(alias) == [] and alias.encode() not in _raw_bytes(path)
    s.rebuild_entity_index()
    assert s.entity_ids(alias) == []
    s.close()


def test_an_aka_alias_goes_with_a_new_text_and_a_given_one_stays(store):
    fact = store.add_fact("Robert Brown aka Bobby fixed the roof")
    store.add_fact("Robert Brown likes tea")
    store.add_alias("Robert Brown", "Rob")
    store.update_fact(fact, content="Robert Brown fixed the roof")
    assert store.list_entities()[0]["aliases"] == ["Rob"]
    store.add_fact("Robert Brown aka Bobby painted the fence")
    assert store.list_entities()[0]["aliases"] == ["Bobby", "Rob"]


@pytest.mark.parametrize("claim,other,x,y", [
    ("Xylophone Zymurgy aka Bob Brown fixed the roof", '"Bob Brown" likes tea', "Xylophone Zymurgy", "Bob Brown"),
    ("Ксилофон Зимургия aka Мария Иванова звонила", "«Мария Иванова» любит чай", "Ксилофон Зимургия", "Мария Иванова"),
    ("燕园秘名 aka 张伟 发了消息", "“张伟”喜欢茶", "燕园秘名", "张伟"),
])
def test_a_name_only_a_forgotten_fact_gave_does_not_stay_on_the_facts_it_named(tmp_path, claim, other, x, y):
    """Review 0.16.0: "X aka Y" stored first makes Y an alias of X, and a later fact about Y links to X. Forgetting the
    claim dropped the alias but left X — a name from the forgotten text — on the other fact and in the file."""
    path = tmp_path / "facts.db"
    s = MemoryStore(str(path), hrr_dim=DIM)
    gone = s.add_fact(claim)
    kept = s.add_fact(other)
    assert _names(s, kept) == [x]
    assert s.forget_fact(gone)["scrub"] == "done"
    assert _names(s, kept) == [y] and s.entity_ids(x) == [] and x.encode() not in _raw_bytes(path)
    _assert_consistent(s)
    s.close()


def test_a_merge_without_numpy_empties_the_vectors_it_changed(store):
    """Review 0.16.0: _merge_into recomputed vectors only with numpy, so without it a vector kept the old name's role
    at the right size, and no later writer could tell."""
    fact = store.add_fact('"Alice Smith" likes tea')
    assert store.read("SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact,))[0][0] is not None
    store._hrr_available = False
    store.add_alias("Mallory Jones", "Alice Smith")
    assert store.read("SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact,))[0][0] is None


def test_a_name_with_a_comma_never_splits_into_aliases(tmp_path):
    """Review 0.16.0: "Smith, John" in the comma-separated legacy column came back on the next open as the aliases
    "Smith" and "John". A comma is no longer part of a name; an older one is left out of the legacy column."""
    path = tmp_path / "facts.db"
    s = MemoryStore(str(path), hrr_dim=DIM)
    with pytest.raises(ValueError, match="cannot be a name"):
        s.add_alias("John Smith", "Smith, John")
    assert s._extract_entities('Signed "Smith, John"') == []
    s.add_fact("John Smith fixed the roof")
    s._write("INSERT INTO entities (name, name_key) VALUES ('Smith, John', 'smith, john')")  # as an older version could
    s.add_alias("John Smith", "Johnny Smith")
    s.rename_entity("Smith, John", "J Smith")
    s.close()
    s = MemoryStore(str(path), hrr_dim=DIM)
    assert s.entity_ids("Smith") == [] and s.entity_ids("John") == [] and s.entity_ids("Smith, John")
    s.close()


@pytest.mark.parametrize("name,alias", [("x", "Bob"), ("Bob Brown", "/private/secret.txt"), ("Bob Brown", "a" * 200),
                                        ("Bob Brown", "key=value"), ("Bob Brown", None)])
def test_alias_refuses_what_is_not_a_name(p, name, alias):
    """Review 0.16.0: alias was the one way to give a name that skipped even the check every given name passes."""
    args = {"action": "alias", "entity": name, **({"content": alias} if alias is not None else {})}
    assert "error" in _call(p, **args)
    assert _call(p, action="entities")["count"] == 0


@pytest.mark.parametrize("name,alias", [("Robert Smith", "Bob"), ("Мария Иванова", "Маша"), ("北京大学", "北大")])
def test_a_context_that_does_not_migrate_finds_an_entity_by_its_alias(tmp_path, name, alias):
    """Review 0.16.0 (Antigravity): a store opened without migrating (a subagent, a cron job) matched names only,
    though the file had keys and aliases, so probe on an alias fell back to keyword search."""
    path = str(tmp_path / "facts.db")
    writer = MemoryStore(path, hrr_dim=DIM)
    writer.add_fact(f'"{name}" fixed the roof')
    writer.add_alias(name, alias)
    entity_id = writer.entity_ids(name)
    writer.close()
    reader = MemoryStore(path, hrr_dim=DIM, migrate=False)
    assert reader.entity_ids(alias) == entity_id and not reader._entry.get("migrated")
    reader.close()
    writer = MemoryStore(path, hrr_dim=DIM)  # a writer after a reader in one process still migrates
    assert writer._entry.get("migrated") and writer.entity_ids(alias) == entity_id
    writer.close()


def test_an_alias_said_with_extra_spaces_is_backed_by_its_fact(store):
    """Review 0.16.0 (Antigravity): "also  known as" with two spaces matched the pattern but not the SQL that finds the
    facts backing an alias, so the next clean-up dropped a live alias."""
    store.add_fact("Robert Brown also  known as Bobby fixed the roof")
    other = store.add_fact("Robert Brown likes tea")
    store.update_fact(other, content="Robert Brown likes green tea")  # a rewrite runs the clean-up
    assert store.list_entities()[0]["aliases"] == ["Bobby"]


def test_an_entity_is_never_its_own_alias(store):
    """Review 0.16.0 (Antigravity): alias("Bob", "Robert Smith") with Bob already Robert Smith's alias stored the
    entity's own name as an alias of it."""
    store.add_fact("Robert Smith lives in London")
    store.add_alias("Robert Smith", "Bob")
    store.add_alias("Bob", "Robert Smith")
    assert [(r["name"], r["aliases"]) for r in store.list_entities()] == [("Robert Smith", ["Bob"])]
    assert store.read("SELECT aliases FROM entities")[0][0] == "Bob"


def test_entity_changes_need_a_migrated_store(tmp_path):
    """Review 0.16.0 (Antigravity): delete_entity on a store opened without migrating wrote no tombstone and said
    "scrub": "done" for a file it had not rewritten."""
    path = str(tmp_path / "facts.db")
    writer = MemoryStore(path, hrr_dim=DIM)
    writer.add_fact('Code name "Phoenix"')
    writer.close()
    reader = MemoryStore(path, hrr_dim=DIM, migrate=False)
    for call in (lambda: reader.delete_entity("Phoenix"), lambda: reader.rename_entity("Phoenix", "Firebird"),
                 reader.rebuild_entity_index):
        with pytest.raises(ValueError, match="read-only"):
            call()
    assert reader.entity_ids("Phoenix")
    reader.close()


def test_a_refused_entity_change_says_why_and_that_nothing_changed(p):
    _call(p, action="add", content='Code name "Phoenix"')
    _call(p, action="add", content='Code name "Firebird"')
    taken = _call(p, action="rename", entity="Phoenix", content="Firebird")["error"]
    assert "already names" in taken and "Nothing was changed" in taken
    assert "Nothing was changed" in _call(p, action="alias", entity="Phoenix", content="/etc/hosts")["error"]


def test_a_chain_of_aka_is_read_in_linear_time(store):
    """Review 0.16.0: every "aka" read the whole text before and after it; 'Alice aka Bob aka …' of 10,000 characters
    took a second per fact, and the rebuild of three such facts twelve seconds under the write lock."""
    import time
    chain = ("Alice aka Bob aka " * 600)[:10000]
    started = time.perf_counter()
    store._aka_pairs(chain)
    store._extract_entities(chain)
    assert time.perf_counter() - started < 0.5
    with store._tx():
        for i in range(3):
            store._write("INSERT INTO facts (content) VALUES (?)", (f"{chain} note{i}",))
    started = time.perf_counter()
    store.rebuild_entity_index()
    assert time.perf_counter() - started < 3


def test_the_rebuild_on_open_is_bounded_by_the_text_as_well(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    _old_store(path, ["Played it all weekend", "We met at noon"], junk=[(0, "count")])
    monkeypatch.setattr(store_module, "_AUTO_REBUILD_MAX_CHARS", 20)
    s = MemoryStore(str(path), hrr_dim=DIM)
    assert s.entity_rules() == 1 and "rebuild_entities" in s.entity_index_state()
    s.close()


def test_without_numpy_the_vectors_of_unlinked_facts_are_emptied(store, monkeypatch):
    fact = store.add_fact('Code name "Phoenix"')
    monkeypatch.setattr(store, "_hrr_available", False)
    store.delete_entity("Phoenix")
    assert store.read("SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact,))[0][0] is None


# --- rename -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,old,new", [
    ("他在“华为”工作", "华为", "华为技术"),
    ('Code name "Optimized Hermes Studio"', "Optimized Hermes Studio", "Hermes Studio"),
    ("Счёт в «Сбер Бизнес»", "Сбер Бизнес", "СберБизнес Онлайн"),
])
def test_rename_keeps_the_entity_and_the_old_name_finds_it(p, text, old, new):
    fact = _call(p, action="add", content=text)["fact_id"]
    entity_id = p._store.entity_ids(old)[0]
    result = _call(p, action="rename", entity=old, content=new)
    assert result["entity_id"] == entity_id and result["name"] == new and result["previous"] == old
    assert old in result["aliases"]
    for name in (new, old):
        assert [r["fact_id"] for r in _call(p, action="probe", entity=name)["results"]] == [fact]
    row = p._store.read("SELECT entity_type, aliases FROM entities WHERE entity_id = ?", (entity_id,))[0]
    assert row["entity_type"] == "declared" and old in row["aliases"].split(",")
    assert _status(p._store, new) == "established"
    _assert_consistent(p._store)


def test_rename_to_another_spelling_changes_the_name_only(store):
    store.add_fact('Code name "phoenix-x"')
    store.rename_entity("phoenix-x", "Phoenix X")
    assert [(r["name"], r["aliases"]) for r in store.list_entities()] == [("Phoenix X", [])]


def test_rename_to_its_own_alias_swaps_them(store):
    store.add_fact("Robert Brown fixed the roof")
    store.add_alias("Robert Brown", "Bob Brown")
    store.rename_entity("Robert Brown", "Bob Brown")
    assert [(r["name"], r["aliases"]) for r in store.list_entities()] == [("Bob Brown", ["Robert Brown"])]
    assert store.read("SELECT aliases FROM entities")[0][0] == "Robert Brown"


@pytest.mark.parametrize("taken", ["Bob Brown", "Bobby"])
def test_rename_to_a_name_another_entity_has_is_refused_with_the_way_to_merge(store, taken):
    store.add_fact("Robert Brown fixed the roof")
    store.add_fact("Bob Brown painted the fence")
    store.add_alias("Bob Brown", "Bobby")
    with pytest.raises(ValueError, match="alias"):
        store.rename_entity("Robert Brown", taken)
    assert {r["name"] for r in store.list_entities()} == {"Robert Brown", "Bob Brown"}


@pytest.mark.parametrize("new", ["", "  ", "/etc/hosts", "--flag", "x"])
def test_rename_refuses_a_new_name_that_is_not_one(store, new):
    store.add_fact('Code name "Phoenix"')
    with pytest.raises(ValueError, match="cannot be a name"):
        store.rename_entity("Phoenix", new)


def test_rename_of_an_unknown_name_is_an_answer(p):
    assert "no entity is named" in _call(p, action="rename", entity="Nobody Here", content="Somebody")["error"]


def test_rename_recomputes_the_vectors_of_its_facts(store):
    pytest.importorskip("numpy")
    fact = store.add_fact('Code name "Phoenix"')
    before = store.read("SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact,))[0][0]
    store.rename_entity("Phoenix", "Firebird")
    after = store.read("SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact,))[0][0]
    assert after is not None and after != before


def test_alias_keeps_the_entity_as_the_name_and_rename_changes_it(store):
    """Issue #4's third request: which name survives a merge. alias keeps `entity`; rename picks another."""
    store.add_fact('We use "Optimized Hermes Studio"')
    store.add_fact('We use "Hermes Studio" daily')
    store.add_alias("Optimized Hermes Studio", "Hermes Studio")
    store.rename_entity("Optimized Hermes Studio", "Hermes Studio")
    assert [(r["name"], r["aliases"]) for r in store.list_entities()] == [("Hermes Studio", ["Optimized Hermes Studio"])]


# --- rebuilding an index built under older rules -----------------------------------------------------------------

def _old_store(path, facts, junk, declared=(), aliased=()):
    """A store as 0.15.x and the core left it: the facts, then entities and links of the old rules, raw, and the
    schema of 0.15.x (no entity_aliases.source, no entity_tombstones, no meta.entity_rules)."""
    s = MemoryStore(str(path), hrr_dim=DIM)
    ids = [s.add_fact(f) for f in facts]
    for fact_index, name in junk:
        cur = s._write("INSERT INTO entities (name) VALUES (?)", (name,))
        s._write("INSERT INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (ids[fact_index], cur.lastrowid))
    for fact_index, name in declared:
        s.update_fact(ids[fact_index], entities=[name])
    for name, alias in aliased:
        s.add_alias(name, alias)
    s._write("DELETE FROM meta WHERE key = 'entity_rules'")
    s._write("DROP TABLE entity_tombstones")
    s._write("ALTER TABLE entity_aliases DROP COLUMN source")
    s.close()
    return ids


def _schema(path):
    conn = sqlite3.connect(str(path))
    try:
        tables = sorted(tuple(r) for r in conn.execute("SELECT type, name, sql FROM sqlite_master"))
        columns = [tuple(r) for r in conn.execute("PRAGMA table_info(entity_aliases)")]
        meta = sorted(tuple(r) for r in conn.execute("SELECT key, value FROM meta"))
    finally:
        conn.close()
    return tables, columns, meta


def test_the_old_schema_fixture_is_the_old_schema(tmp_path):
    _old_store(tmp_path / "old.db", ["Played it all weekend"], junk=[])
    tables, columns, meta = _schema(tmp_path / "old.db")
    assert "entity_tombstones" not in {t[1] for t in tables}
    assert "source" not in {c[1] for c in columns} and "entity_rules" not in dict(meta)


def test_an_old_index_is_rebuilt_on_open(tmp_path):
    path = tmp_path / "old.db"
    ids = _old_store(path, [
        "met alice smith at the market",               # no name of its own; mentions the next fact's name
        "Alice Smith lives in Paris",
        "用户说“先去搜索再告诉我答案”",
        "Слово «сумма» в отчёте",
        "Played it all weekend",
    ], junk=[(2, "先去搜索再告诉我答案"), (3, "сумма"), (4, "count"), (4, r"^\s*\d+\|"), (4, "0.17.0 installed")],
        declared=[(4, "我的世界")], aliased=[("Anna Lee", "Annie")])
    s = MemoryStore(str(path), hrr_dim=DIM)
    try:
        assert sorted(r["name"] for r in s.list_entities(limit=100)) == ["Alice Smith", "Anna Lee", "我的世界"]
        assert _names(s, ids[0]) == ["Alice Smith"]  # the second pass: mentions, whatever the order of the facts
        assert _names(s, ids[4]) == ["我的世界"]
        assert s.entity_rules() == store_module._ENTITY_RULES
        assert s.count_facts() == 5
        _assert_consistent(s)
    finally:
        s.close()


def test_a_rebuild_keeps_the_id_and_age_of_names_that_stay_and_a_second_one_changes_nothing(tmp_path):
    path = tmp_path / "old.db"
    _old_store(path, ['Code name "Phoenix"', '"Phoenix" ships in May'], junk=[(0, "count")])
    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE entities SET created_at = '2026-01-01 00:00:00' WHERE name = 'Phoenix'")
    conn.commit()
    entity_id = conn.execute("SELECT entity_id FROM entities WHERE name = 'Phoenix'").fetchone()[0]
    conn.close()
    s = MemoryStore(str(path), hrr_dim=DIM)  # the rebuild on open
    assert s.entity_ids("Phoenix") == [entity_id] and s.entity_ids("count") == []
    assert s.read("SELECT created_at FROM entities WHERE entity_id = ?", (entity_id,))[0][0] == "2026-01-01 00:00:00"
    again = s.rebuild_entity_index()
    assert (again["links_removed"], again["links_added"], again["vectors"]) == (0, 0, 0)
    s.close()


def test_a_rebuild_keeps_a_declared_entity_without_facts(tmp_path):
    path = tmp_path / "old.db"
    _old_store(path, ["Played it all weekend"], junk=[])
    conn = sqlite3.connect(str(path))
    conn.execute("INSERT INTO entities (name, name_key, entity_type) VALUES ('Masha', 'masha', 'declared')")
    conn.commit()
    conn.close()
    s = MemoryStore(str(path), hrr_dim=DIM)
    assert [r["name"] for r in s.list_entities()] == ["Masha"]
    s.close()


AKA_PAIRS = [
    ['"Bob Brown" likes tea', "Robert Brown aka Bob Brown fixed the roof"],
    ["“华为”在深圳", "Huawei Technologies aka 华为 makes phones"],
    ["«Сбер Бизнес» закрыли", "Сбербанк aka Сбер Бизнес открыл счёт"],
]


def _inventory(store):
    return sorted((r["name"], tuple(r["aliases"]), r["facts"]) for r in store.list_entities(limit=100))


@pytest.mark.parametrize("texts", AKA_PAIRS)
@pytest.mark.parametrize("reverse", [False, True])
def test_a_rebuild_reads_the_facts_as_adding_them_in_order_would(tmp_path, texts, reverse):
    """The rebuild replays the facts in the order they were stored, whatever the old index held: the result is what
    adding them to an empty store gives (the mention pass may only add links)."""
    order = list(reversed(texts)) if reverse else texts
    fresh = MemoryStore(str(tmp_path / "fresh.db"), hrr_dim=DIM)
    for text in order:
        fresh.add_fact(text)
    expected = _inventory(fresh)
    fresh.close()
    _old_store(tmp_path / "old.db", order, junk=[(0, "count"), (1, "count")])
    s = MemoryStore(str(tmp_path / "old.db"), hrr_dim=DIM)
    s.rebuild_entity_index()
    assert _inventory(s) == expected
    s.close()


@pytest.mark.parametrize("y_facts,claim,y,x", [
    (['"Alice Smith" likes tea', '"Alice Smith" lives in London'], "Mallory Jones aka Alice Smith sent a message",
     "Alice Smith", "Mallory Jones"),
    (["“张伟”喜欢茶", "“张伟”住在北京"], "李强 aka 张伟 发了消息", "张伟", "李强"),
    (["«Мария Иванова» любит чай", "«Мария Иванова» живёт в Москве"], "Анна Петрова aka Мария Иванова звонила",
     "Мария Иванова", "Анна Петрова"),
])
def test_a_stored_x_aka_y_never_takes_over_a_name_already_in_use(tmp_path, y_facts, claim, y, x):
    """Review 0.16.0: the rebuild merged an established Y into X because of one stored sentence; adding the
    sentence never did. Neither may now: Y keeps its facts."""
    path = tmp_path / "old.db"
    _old_store(path, [*y_facts, claim], junk=[])
    s = MemoryStore(str(path), hrr_dim=DIM)
    for _ in range(2):  # on open, and on request
        assert s.entity_ids(y) and s.entity_ids(x) and s.entity_ids(y) != s.entity_ids(x)
        assert next(r for r in s.list_entities() if r["name"] == y)["facts"] == 3  # its two facts and the claim
        s.rebuild_entity_index()
    s.close()


def test_a_rebuild_keeps_tombstones(tmp_path):
    path = tmp_path / "old.db"
    _old_store(path, ['Code name "Phoenix"'], junk=[])
    s = MemoryStore(str(path), hrr_dim=DIM)
    s.delete_entity("Phoenix")
    s.rebuild_entity_index()
    assert s.list_entities() == []
    s.close()


def test_a_large_old_store_is_rebuilt_on_request(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    _old_store(path, ["Played it all weekend", "We met at noon"], junk=[(0, "count")])
    monkeypatch.setattr(store_module, "_AUTO_REBUILD_MAX_FACTS", 1)
    p = HolographicMemoryProvider(config={"db_path": str(path), "hrr_dim": DIM})
    p.initialize(session_id="s")
    try:
        assert "rebuild_entities" in _call(p, action="stats")["entity_index"]
        assert _call(p, action="entities")["count"] == 1
        result = _call(p, action="rebuild_entities")
        assert result["entities_before"] == 1 and result["entities_after"] == 0
        assert _call(p, action="stats")["entity_index"] == "current"
    finally:
        p.shutdown()


def test_a_newer_release_s_index_is_left_alone(tmp_path):
    path = tmp_path / "new.db"
    s = MemoryStore(str(path), hrr_dim=DIM)
    s.add_fact('Code name "Phoenix"')
    s._write("UPDATE meta SET value = ? WHERE key = 'entity_rules'", (str(store_module._ENTITY_RULES + 1),))
    with pytest.raises(ValueError, match="newer release"):
        s.rebuild_entity_index()
    s.close()
    s = MemoryStore(str(path), hrr_dim=DIM)
    assert s.entity_rules() == store_module._ENTITY_RULES + 1 and _status(s, "Phoenix") == "provisional"
    s.close()


def test_a_failed_rebuild_leaves_the_old_index_and_opens(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    _old_store(path, ["Played it all weekend"], junk=[(0, "count")])

    def boom(self, content):
        raise RuntimeError("boom")
    monkeypatch.setattr(MemoryStore, "mentioned_entity_ids", boom)
    s = MemoryStore(str(path), hrr_dim=DIM)
    assert [r["name"] for r in s.list_entities()] == ["count"] and s.entity_rules() == 1
    s.close()


def test_a_read_only_open_of_an_old_store_changes_nothing(tmp_path):
    path = tmp_path / "old.db"
    _old_store(path, ["Played it all weekend"], junk=[(0, "count")], aliased=[("Anna Lee", "Annie")])
    before = _schema(path)
    s = MemoryStore(str(path), hrr_dim=DIM, migrate=False)
    assert sorted(r["name"] for r in s.list_entities()) == ["Anna Lee", "count"] and s.entity_rules() == 1
    assert _status(s, "Anna Lee") == "established"
    assert s.entity_counts() == {"entities": 2, "provisional": 1, "text_aliases": 0}
    assert s.stale_provisional_entities(days=0) and s._extract_entities('Code name "Phoenix"') == ["Phoenix"]
    s.close()
    assert _schema(path) == before


def test_old_aliases_are_of_unknown_origin_kept_like_given_ones_and_go_only_with_a_forgotten_fact(tmp_path):
    """Review 0.16.0: guessing at migration that an old alias a fact's "X aka Y" repeats was the text's dropped
    aliases someone had given. Nothing tells them apart, so they are 'legacy': kept and protected like given ones,
    kept when the fact is reworded, and dropped only when a forgotten fact said them."""
    path = tmp_path / "old.db"
    ids = _old_store(path, ["Robert Brown aka Bobby fixed the roof", "Robert Brown likes tea",
                            "Robert Brown aka Rob painted the fence", "Anna Lee likes tea"],
                     junk=[], aliased=[("Anna Lee", "Annie"), ("Robert Brown", "Bobby")])
    s = MemoryStore(str(path), hrr_dim=DIM)
    assert set(s.read("SELECT source FROM entity_aliases")[i][0] for i in range(3)) == {"legacy"}
    assert _status(s, "Anna Lee") == "established"
    s.update_fact(ids[2], content="Robert Brown painted the fence")  # reworded: an alias of unknown origin stays
    assert "Rob" in next(r for r in s.list_entities() if r["name"] == "Robert Brown")["aliases"]
    s.forget_fact(ids[0])  # forgotten: what it said goes, and so does the bytes' copy
    robert = next(r for r in s.list_entities() if r["name"] == "Robert Brown")
    assert "Bobby" not in robert["aliases"] and "Rob" in robert["aliases"] and b"Bobby" not in _raw_bytes(path)
    assert "Annie" in next(r for r in s.list_entities() if r["name"] == "Anna Lee")["aliases"]
    s.close()


def test_a_rebuild_does_not_touch_history_confirmations_or_trust(tmp_path):
    path = tmp_path / "old.db"
    ids = _old_store(path, ['Code name "Phoenix"', "Played it all weekend"], junk=[(1, "count")])
    conn = sqlite3.connect(str(path))
    before = [conn.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in ("fact_versions", "fact_confirmations")]
    trust = conn.execute("SELECT fact_id, trust_score, retrieval_count FROM facts ORDER BY 1").fetchall()
    conn.close()
    s = MemoryStore(str(path), hrr_dim=DIM)
    s.close()
    conn = sqlite3.connect(str(path))
    assert [conn.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in ("fact_versions", "fact_confirmations")] == before
    assert conn.execute("SELECT fact_id, trust_score, retrieval_count FROM facts ORDER BY 1").fetchall() == trust
    assert conn.execute("SELECT COUNT(*) FROM fact_entities WHERE fact_id = ?", (ids[1],)).fetchone()[0] == 0
    conn.close()


def test_the_rebuild_takes_the_same_order_whatever_order_the_facts_came_in(tmp_path):
    texts = ["met alice smith at the market", "Alice Smith lives in Paris", "alice smith likes tea",
             'Code name "Phoenix"', "the phoenix project is late"]
    links = []
    for order, name in ((texts, "a.db"), (list(reversed(texts)), "b.db")):
        _old_store(tmp_path / name, order, junk=[])
        s = MemoryStore(str(tmp_path / name), hrr_dim=DIM)
        s.rebuild_entity_index()
        links.append(sorted((c, n) for c in texts for n in _names(s, s.read("SELECT fact_id FROM facts WHERE content = ?",
                                                                             (c,))[0][0])))
        s.close()
    assert links[0] == links[1]
    assert ("met alice smith at the market", "Alice Smith") in links[0]
