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

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    is_compaction_summary_message,
)
from hermes_plugin_holographic import HolographicMemoryProvider


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
# zh-TW / zh-CN extraction
# ---------------------------------------------------------------------------

ZH_PREFERENCE = "我偏好簡潔且可驗證的解法，不要過度工程。"
ZH_HABITUAL = "我一律使用繁體中文回覆，不會用簡體。"
ZH_DECISION = "我們決定專案以 Julia 作為唯一的模擬核心。"
ZH_DECLARATIVE = "命名規則是先日期後主題。"


@pytest.mark.parametrize("message,expected_category", [
    (ZH_PREFERENCE, "user_pref"),
    (ZH_HABITUAL, "user_pref"),
    (ZH_DECISION, "project"),
    (ZH_DECLARATIVE, "project"),
])
def test_zh_extraction_stores_the_sentence_under_the_right_category(tmp_path, message, expected_category):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(message)])
    facts = provider._store.list_facts(limit=100)
    assert [f["content"] for f in facts] == [message]
    assert facts[0]["category"] == expected_category
    provider.shutdown()


def test_zh_full_width_terminators_split_a_paragraph_into_one_fact_per_clause(tmp_path):
    """A Chinese paragraph is not one fact: 。and ；are sentence terminators too, so a
    message holding three separate statements yields three facts, not one long one."""
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user("我偏好繁體中文回覆。我也偏好簡單的解法。我們決定用 Julia。")])
    assert _fact_contents(provider) == ["我偏好繁體中文回覆。", "我也偏好簡單的解法。", "我們決定用 Julia。"]
    provider.shutdown()


def test_zh_restatement_is_dropped_rather_than_stored_again(tmp_path):
    """An unattended extractor has nobody to read write_check's advisory hint, so a
    near-duplicate is dropped instead of becoming one more copy of the same fact."""
    provider = _make_provider(tmp_path, auto_extract=True)
    for message in ("我偏好所有回覆一律使用繁體中文與台灣用語。",
                    "Frank 要求所有回覆只使用繁體中文與台灣用語。",
                    "所有回覆只用繁體中文和台灣用語，不用簡體。"):
        provider.on_session_end([_user(message)])
    assert len(_fact_contents(provider)) == 1
    provider.shutdown()


def test_zh_sentence_is_stored_under_one_category_only(tmp_path):
    """「我們決定命名規則是 …」matches both a decision pattern and a declarative one; the
    sentence is stored once, not once per matching category."""
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user("我們決定命名規則是 YYMMDDNN-topic.md，這是專案的固定規則。")])
    facts = provider._store.list_facts(limit=100)
    assert len(facts) == 1
    provider.shutdown()


@pytest.mark.parametrize("message", [
    "我不太確定這個能不能用，先看看再說吧。",      # uncertainty, not a preference
    "你知道為什麼會這樣嗎？",                     # a question
    "你必須先備份資料庫再動手，不得跳過。",         # an imperative addressed to the assistant
    "你好，今天天氣不錯。",                       # chit-chat
    "請幫我檢查一下這個檔案。",                   # a request
])
def test_zh_utterances_that_are_not_durable_facts_are_not_stored(tmp_path, message):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(message)])
    assert _fact_contents(provider) == []
    provider.shutdown()
