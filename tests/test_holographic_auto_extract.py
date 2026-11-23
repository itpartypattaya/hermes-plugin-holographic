"""Regression tests for #57682 — holographic auto_extract harvested
context-compaction handoff summaries into fact_store, and ran even when
configured off.

Two compounding defects:

1. Gate: the plugin's config schema declares ``auto_extract`` as a string enum
   (``"false"``/``"true"``), and the ``on_session_end`` gate used plain
   truthiness — ``not "false"`` is ``False`` — so extraction ran despite being
   explicitly configured off.

2. Eligibility: ``_auto_extract_facts`` scanned every ``role == "user"``
   message. Compaction handoff summaries can be inserted as ``role="user"``
   messages, and their prose reliably matches the decision patterns
   (``we decided/agreed/chose``, ``the project uses/needs/requires``), so the
   compactor's own output was stored as a durable ``project`` fact on every
   session rollover that followed a compaction.
"""

import sys
import types

import pytest

try:  # Hermes v0.23.3 moved the compaction helpers here; agent.context_compressor re-exports them for now
    from session_rules import turns as _compaction
except ImportError:  # Hermes up to v0.23.2
    from agent import context_compressor as _compaction
from hermes_plugin_holographic import HolographicMemoryProvider

COMPRESSED_SUMMARY_METADATA_KEY = _compaction.COMPRESSED_SUMMARY_METADATA_KEY
SUMMARY_PREFIX = _compaction.SUMMARY_PREFIX
_MERGED_PRIOR_CONTEXT_HEADER = _compaction._MERGED_PRIOR_CONTEXT_HEADER
_MERGED_SUMMARY_DELIMITER = _compaction._MERGED_SUMMARY_DELIMITER
is_compaction_summary_message = _compaction.is_compaction_summary_message


def _make_provider(tmp_path, **config):
    base = {"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    base.update(config)
    provider = HolographicMemoryProvider(config=base)
    provider.initialize(session_id="test-session")
    return provider


def _fact_contents(provider):
    return [f["content"] for f in provider._store.list_facts(limit=100)]


def _user(content, **extra):
    msg = {"role": "user", "content": content}
    msg.update(extra)
    return msg


DECISION_MSG = "we decided to use PostgreSQL for the persistence layer"
SUMMARY_MSG = (
    f"{SUMMARY_PREFIX}\n## Historical Task Snapshot\n"
    "The project uses a kanban board for all dispatch. "
    "We agreed to route reviews through the fan-in consumer."
)


# ---------------------------------------------------------------------------
# Defect 1 — string-boolean gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("off_value", [False, "false", "False", "no", "off", "0", None, ""])
def test_auto_extract_off_values_disable_extraction(tmp_path, off_value):
    provider = _make_provider(tmp_path, auto_extract=off_value)
    provider.on_session_end([_user(DECISION_MSG)])
    assert _fact_contents(provider) == []
    provider.shutdown()


# ---------------------------------------------------------------------------
# Defect 2 — compaction summaries harvested as facts
# ---------------------------------------------------------------------------


def test_compaction_summary_not_harvested(tmp_path):
    """The exact failure mode from #57682: summary prose matches the decision
    patterns but must not become a fact."""
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(SUMMARY_MSG)])
    assert _fact_contents(provider) == []
    provider.shutdown()


def test_metadata_marked_summary_not_harvested(tmp_path):
    """In-process summaries carry COMPRESSED_SUMMARY_METADATA_KEY even if a
    future prefix rewrite changes the content sentinel."""
    provider = _make_provider(tmp_path, auto_extract=True)
    marked = _user(DECISION_MSG, **{COMPRESSED_SUMMARY_METADATA_KEY: True})
    provider.on_session_end([marked])
    assert _fact_contents(provider) == []
    provider.shutdown()


def test_merged_into_tail_summary_suffix_not_harvested_prefix_content_ignored(tmp_path):
    """Merge-into-tail summaries embed the handoff prefix after the delimiter,
    not at the start of the message. The wrapped pre-delimiter segment here has
    no fact-pattern match, so nothing is harvested from either side."""
    provider = _make_provider(tmp_path, auto_extract=True)
    merged = _user(
        f"{_MERGED_PRIOR_CONTEXT_HEADER}\nplease fix the login bug\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_MSG}"
    )
    provider.on_session_end([merged])
    assert _fact_contents(provider) == []
    provider.shutdown()


def test_real_user_messages_still_extracted_alongside_summary(tmp_path):
    """The guard must skip only the summary, not suppress extraction for the
    genuine user turns around it."""
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end(
        [
            _user(SUMMARY_MSG),
            _user("I prefer tabs over spaces for indentation"),
        ]
    )
    facts = _fact_contents(provider)
    assert len(facts) == 1
    assert "tabs over spaces" in facts[0]
    provider.shutdown()


# ---------------------------------------------------------------------------
# is_compaction_summary_message — public helper contract
# ---------------------------------------------------------------------------


def test_helper_detects_prefix_metadata_and_merged_forms():
    assert is_compaction_summary_message(_user(SUMMARY_MSG))
    assert is_compaction_summary_message(
        _user("anything", **{COMPRESSED_SUMMARY_METADATA_KEY: True})
    )
    assert is_compaction_summary_message(
        _user(f"prior tail\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_MSG}")
    )
    assert not is_compaction_summary_message(_user(DECISION_MSG))
    assert not is_compaction_summary_message(_user(""))


# ---------------------------------------------------------------------------
# Where the core keeps the compaction helpers. auto_extract needs the merged-summary header and delimiter and
# is_compaction_summary_message: Hermes up to v0.23.2 has them only in agent.context_compressor, v0.23.3 moved them to
# session_rules.turns and keeps agent.context_compressor re-exporting them, a later core may drop that re-export.
# ---------------------------------------------------------------------------

_HELPERS = ("_MERGED_PRIOR_CONTEXT_HEADER", "_MERGED_SUMMARY_DELIMITER", "is_compaction_summary_message")


@pytest.fixture
def core_without_compressor_reexport(monkeypatch):
    """A core that keeps the helpers only in session_rules.turns. The helpers are the running core's own; only their old
    address goes (on a core before v0.23.3 they are also given the new one, as v0.23.3 has it)."""
    try:
        import session_rules.turns  # noqa: F401 - v0.23.3 and later: the new address is real
    except ImportError:
        package, turns = types.ModuleType("session_rules"), types.ModuleType("session_rules.turns")
        package.__path__ = []
        for name in _HELPERS:
            setattr(turns, name, getattr(_compaction, name))
        package.turns = turns
        monkeypatch.setitem(sys.modules, "session_rules", package)
        monkeypatch.setitem(sys.modules, "session_rules.turns", turns)
    monkeypatch.setitem(sys.modules, "agent.context_compressor", types.ModuleType("agent.context_compressor"))


@pytest.fixture
def core_before_session_rules(monkeypatch):
    """Hermes up to v0.23.2 (what Mia ran when this was written): no session_rules, the helpers only in
    agent.context_compressor. The helpers are the running core's own, given the old address whether or not the running
    core still re-exports them there."""
    compressor = types.ModuleType("agent.context_compressor")
    for name in _HELPERS:
        setattr(compressor, name, getattr(_compaction, name))
    monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor)
    monkeypatch.setitem(sys.modules, "session_rules", None)
    monkeypatch.setitem(sys.modules, "session_rules.turns", None)


@pytest.mark.parametrize("core", ["core_without_compressor_reexport", "core_before_session_rules"])
def test_auto_extract_finds_the_compaction_helpers_where_the_core_keeps_them(tmp_path, request, core):
    """At session end a person's own words are stored and the compactor's are not, wherever the core keeps the
    helpers that tell them apart: the user text before a merged summary's delimiter is kept, the summary after it and
    a summary message are not."""
    request.getfixturevalue(core)
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([
        _user(f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{DECISION_MSG}\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_MSG}"),
        _user(SUMMARY_MSG),
        _user("I prefer tabs over spaces for indentation"),
    ])
    facts = _fact_contents(provider)
    provider.shutdown()
    assert len(facts) == 2, facts
    assert any("PostgreSQL" in f for f in facts) and any("tabs over spaces" in f for f in facts), facts


# ---------------------------------------------------------------------------
# Chinese extraction, questions, restatements (0.9.0; the Chinese patterns, the full-width terminators and dropping a
# restatement come from itpartypattaya/hermes-plugin-holographic#2 by au61987ai-code, adapted to 0.8.0's forgetting)
# ---------------------------------------------------------------------------

ZH_PREFERENCE = "我偏好簡潔且可驗證的解法，不要過度工程。"
ZH_HABITUAL = "我一律使用繁體中文回覆，不會用簡體。"
ZH_DECISION = "我們決定專案以 Julia 作為唯一的模擬核心。"
ZH_DECLARATIVE = "命名規則是先日期後主題，再加版本號。"


@pytest.mark.parametrize("message,expected_category", [
    (ZH_PREFERENCE, "user_pref"),
    (ZH_HABITUAL, "user_pref"),
    (ZH_DECISION, "project"),
    (ZH_DECLARATIVE, "project"),
    ("我习惯用简体中文回复所有的消息。", "user_pref"),
    ("我们决定项目统一使用 Python 3.12。", "project"),
    ("项目使用 PostgreSQL 作为主数据库。", "project"),
])
def test_zh_extraction_stores_the_sentence_under_the_right_category(tmp_path, message, expected_category):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(message)])
    facts = provider._store.list_facts(limit=100)
    assert [f["content"] for f in facts] == [message]
    assert facts[0]["category"] == expected_category
    provider.shutdown()


def test_zh_full_width_terminators_split_a_paragraph_into_one_fact_per_clause(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user("我偏好繁體中文的回覆內容。我也偏好簡單直接的解法！我們決定使用 Julia 語言。")])
    assert _fact_contents(provider) == ["我偏好繁體中文的回覆內容。", "我也偏好簡單直接的解法！",
                                        "我們決定使用 Julia 語言。"]
    provider.shutdown()


def test_zh_sentence_is_stored_under_one_category_only(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user("我們決定命名規則是 YYMMDDNN-topic.md，這是專案的固定規則。")])
    assert len(provider._store.list_facts(limit=100)) == 1
    provider.shutdown()


@pytest.mark.parametrize("message", [
    "我不太確定這個能不能用，先看看再說吧。",        # uncertainty
    "你知道為什麼會這樣嗎？",                       # a question
    "你必須先備份資料庫再動手，不得跳過。",           # an order to the assistant
    "你好，今天天氣不錯，出去走走吧。",               # chit-chat
    "請幫我檢查一下這個檔案的內容。",                 # a request
    "你喜歡什麼樣的音樂和電影呢？",                  # someone else's preference, and a question
    "他喜歡吃辣的食物，每天都要。",                   # someone else's preference
    "我不用了，謝謝你的幫忙喔。",                    # "no need, thanks"
    "以我來看這個方案其實還不錯。",                   # a passing opinion
    "這個文件是我昨天晚上寫的。",                     # "I wrote this document", not a project rule
    "我們決定好要去哪裡吃飯了嗎",                     # a question without a question mark
])
def test_zh_utterances_that_are_not_durable_facts_are_not_stored(tmp_path, message):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(message)])
    assert _fact_contents(provider) == []
    provider.shutdown()


def test_a_question_is_not_extracted_in_any_language(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user("Do I always need a visa for Japan? Мы решили ехать в июле?")])
    assert _fact_contents(provider) == []
    provider.shutdown()


def test_a_restatement_is_not_stored_again_and_leaves_no_tombstone(tmp_path):
    # Checked before storing: dropping a stored copy would be a forget in 0.8.0 (erase, tombstone, scrubs downstream)
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user("I prefer green tea without sugar in the morning.")])
    provider.on_session_end([_user("I prefer green tea without sugar in the mornings.")])
    assert len(_fact_contents(provider)) == 1
    assert provider._store.read("SELECT COUNT(*) FROM forgotten")[0][0] == 0
    provider.shutdown()


def test_a_change_is_not_a_restatement(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user("We decided the monthly budget is 500 dollars.")])
    provider.on_session_end([_user("We decided the monthly budget is 600 dollars.")])
    assert len(_fact_contents(provider)) == 2
    provider.shutdown()


def test_a_dot_inside_a_word_does_not_end_the_sentence(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user("We decided to pin Python 3.12 and keep notes in topic.md files. I prefer tea.")])
    assert sorted(_fact_contents(provider)) == ["I prefer tea.",
                                                "We decided to pin Python 3.12 and keep notes in topic.md files."]
    provider.shutdown()


def test_zh_restatement_is_not_stored_again(tmp_path):
    # Both sentences are extracted on their own; the second says nearly what the first does
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user("我偏好所有回覆一律使用繁體中文與台灣用語。")])
    provider.on_session_end([_user("我偏好所有的回覆都一律使用繁體中文與台灣用語。")])
    assert _fact_contents(provider) == ["我偏好所有回覆一律使用繁體中文與台灣用語。"]
    provider.shutdown()
