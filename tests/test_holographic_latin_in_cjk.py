"""NousResearch/hermes-agent#83593: a Latin word or identifier written into Chinese text with no space around it
("用户用keyring保存密钥") was not found by that word. FTS5 indexes the whole run, characters and letters, as one token,
and the substring path ran only for a query with Chinese in it, so `keyring` alone found nothing while `密钥` did. Such a
word is now found in search and in the recall before a turn, and only as a whole word: `key` is not `keyring`."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider

# the issue's fact, as its author wrote it
WATCHDOG = "桌面后端秒退根因:web_server的HERMES_PARENT_PID父进程watchdog单层ppid比较会误杀健康后端。"
KEYRING = "用户用keyring保存密钥"


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 512})
    p.initialize(session_id="s")
    yield p
    p.shutdown()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _add(provider, text):
    result = _call(provider, action="add", content=text, origin="said")
    assert result["status"] == "added", result
    return result["fact_id"]


def _found(provider, query):
    return [r["fact_id"] for r in _call(provider, action="search", query=query)["results"]]


def _recalled(provider, message):
    block = provider.prefetch(message, session_id="s")
    return [int(line.split()[1][1:]) for line in block.splitlines()[1:]] if block else []


@pytest.mark.parametrize("query", ["watchdog", "ppid", "HERMES_PARENT_PID", "web_server", "Watchdog"])
def test_a_word_glued_to_chinese_is_found_by_itself(provider, query):
    fact = _add(provider, WATCHDOG)
    assert _found(provider, query) == [fact]


def test_the_issue_keyring_example(provider):
    fact = _add(provider, KEYRING)
    _add(provider, "用户的猫叫小白")
    assert _found(provider, "keyring") == [fact]
    assert _found(provider, "keyring 密钥") == [fact]  # unchanged: a query with Chinese found it before


@pytest.mark.parametrize("query", ["key", "ring", "dog", "pid", "server", "web"])
def test_only_the_whole_word(provider, query):
    """Part of a glued word is not the word: `key` is not in `keyring`, `pid` not in `ppid` or `PARENT_PID`."""
    _add(provider, KEYRING)
    _add(provider, WATCHDOG)
    assert _found(provider, query) == []


def test_an_english_question_recalls_the_chinese_fact(provider):
    """The recall before a turn counts the glued word as held by that fact; before, the word counted as held by no
    fact, so the gate weighed the message against nothing and recalled nothing."""
    fact = _add(provider, WATCHDOG)
    _add(provider, "The production server hr-vps runs Debian 12")
    assert _recalled(provider, "How does the watchdog compare ppid?") == [fact]


def test_both_spellings_are_found(provider):
    """A store with the word written apart in one fact and glued in another finds both."""
    glued = _add(provider, KEYRING)
    apart = _add(provider, "The keyring backend on the laptop is KWallet")
    assert sorted(_found(provider, "keyring")) == sorted([glued, apart])


def test_a_latin_store_is_unchanged(provider):
    """No fact has Chinese in it: the same results as a word search alone gives, chit-chat still recalls nothing."""
    ids = [_add(provider, t) for t in ("The keyring backend on the laptop is KWallet",
                                         "The watchdog restarts the backend after 30 seconds")]
    assert _found(provider, "keyring") == [ids[0]]
    assert _found(provider, "watchdog") == [ids[1]]
    assert _found(provider, "key") == []
    assert _recalled(provider, "ok, thanks") == []


def test_a_fact_written_after_a_search_is_found(provider):
    """The facts holding such words are listed once and kept until a fact's text may have changed: a fact added,
    changed in place or forgotten after a search counts at the next one."""
    other = _add(provider, "用户的猫叫小白")
    assert _found(provider, "keyring") == []
    fact = _add(provider, KEYRING)
    assert _found(provider, "keyring") == [fact]
    changed = _call(provider, action="change", fact_id=other, content="用户的watchdog每分钟检查一次")
    assert changed.get("changed") or changed.get("status") == "changed", changed
    assert _found(provider, "watchdog") == [other]
    _call(provider, action="forget", fact_id=fact)
    assert _found(provider, "keyring") == []


def test_a_fact_written_by_another_process_is_found(provider, tmp_path):
    import sqlite3
    assert _found(provider, "keyring") == []
    with sqlite3.connect(tmp_path / "memory_store.db") as conn:  # as another process writes it
        conn.execute("INSERT INTO facts (content) VALUES (?)", (KEYRING,))
    assert len(_found(provider, "keyring")) == 1


def test_recall_counts_do_not_hide_a_new_fact(provider):
    """The recall counts are written every few searches and change no text: they keep the list, a new fact does not."""
    fact = _add(provider, KEYRING)
    for _ in range(30):
        assert _found(provider, "keyring") == [fact]
    second = _add(provider, "运维用keyring管理服务器密码")
    assert sorted(_found(provider, "keyring")) == sorted([fact, second])


@pytest.mark.parametrize("text", ["パスワードはkeyringに保存している", "비밀번호는 keyring에 저장한다", "เก็บรหัสผ่านไว้ในkeyringของระบบ"])
def test_other_unsegmented_scripts(provider, text):
    """Japanese, a Korean particle and Thai glue to the word the same way."""
    fact = _add(provider, text)
    assert _found(provider, "keyring") == [fact]
    assert _found(provider, "key") == []


def test_a_letter_of_another_script_is_no_edge(provider):
    """Codex review: the word must stand whole on both sides; a Cyrillic or accented letter glued to it makes it part
    of another word (`яkeyring保存`), as an ASCII letter does."""
    _add(provider, "яkeyring保存密钥")
    _add(provider, "用户用keyringé保存")
    assert _found(provider, "keyring") == []


def test_a_fact_is_counted_once_in_the_words_frequency(provider):
    """Codex review: the recall gate weighs a word by how many facts hold it. A word at the start of a glued run
    (`keyring保存密钥`) is found by the index as a prefix already, and a fact that has the word both apart and glued is
    one fact: each counts once."""
    from hermes_plugin_holographic.text import query_terms
    for text in ("keyring保存密钥", "用户用keyring保存密钥", "The keyring backend is KWallet; 系统keyring默认开启"):
        _add(provider, text)
    _add(provider, "用户的猫叫小白")
    term = query_terms("keyring")[0]
    assert provider._retriever._doc_freqs([term]) == {term: 3}


def test_the_gate_keeps_a_fact_counted_once(provider):
    """Codex review on GitHub, its example: counted twice, `keyring保存密钥` (an FTS prefix match too) weighed less in a
    two-fact store and fell below the gate for "keyring watchdog backend"."""
    keyring = _add(provider, "keyring保存密钥")
    watchdog = _add(provider, "The watchdog monitors the backend")
    retriever = provider._retriever
    candidates = [dict(f) for f in provider._store.list_facts(min_trust=0.0, limit=10)]
    kept = {f["fact_id"] for f in retriever.relevant("keyring watchdog backend", candidates)}
    assert kept == {keyring, watchdog}


def test_a_long_run_without_latin_is_read_in_linear_time():
    """Codex review on GitHub: the pattern that finds a letter against the script took a whole run of it at each
    position, so a run of n characters with no letter after it took n² steps (9 ms for 1,000 characters)."""
    import time
    from hermes_plugin_holographic import retrieval
    run = "用户保存密钥" * 5000  # 30,000 characters, no Latin letter
    start = time.perf_counter()
    assert retrieval._GLUED.search(run) is None
    assert time.perf_counter() - start < 0.5
