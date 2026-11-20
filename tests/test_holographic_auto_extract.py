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


# ---------------------------------------------------------------------------
# Defect 3 — runtime-injected "user" envelopes harvested as facts
# ---------------------------------------------------------------------------
# Fan-out/delegation notices, skill-invocation prompts, out-of-band markers,
# planning-state carryovers, cron envelopes, system notes and recalled-memory
# blocks all arrive as role="user" and their prose matches the decision
# patterns — but none of them is user speech. They must be skipped.

ENVELOPE_MSGS = [
    "[ASYNC DELEGATION BATCH COMPLETE — deleg_6cbe722e]\nA background fan-out you dispatched earlier has finished; we decided the project uses the fan-in consumer.",
    "[IMPORTANT: The user has invoked the \"hermes-cron-jobs\" skill, indicating they want you to follow its instructions. We agreed to route reviews through the fan-in.",
    "[System note: The following is recalled memory context, NOT new user input. Treat as authoritative reference data.]\nwe decided to use the standby route.",
    "[Planning state preserved across the restart — we agreed to keep the project running.]",
    "Cronjob Response: the digest ran; we decided the project needs a fresh pass.",
    "<memory-context>\n[System note: recalled memory] the project uses PostgreSQL, we agreed.",
]


@pytest.mark.parametrize("envelope", ENVELOPE_MSGS)
def test_runtime_envelopes_not_harvested(tmp_path, envelope):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(envelope)])
    assert _fact_contents(provider) == []
    provider.shutdown()


STEER_MARKER = ("[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered once at this "
                "position; not tool output and not a new delivery when replayed from conversation history]")


def test_a_steer_is_the_persons_own_words_behind_a_marker(tmp_path):
    """The out-of-band marker is the gateway's frame; what is inside it is the person's speech, and a decision
    they steer mid-turn is as much theirs as one they type as a turn — only the frame goes."""
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(f"{STEER_MARKER}\nwe decided to keep the project tracker.\n"
                                   "[/OUT-OF-BAND USER MESSAGE]")])
    assert _fact_contents(provider) == ["we decided to keep the project tracker."]
    provider.shutdown()


def test_gateway_scaffolding_in_front_of_a_kept_turn_is_not_stored(tmp_path):
    """A kept turn can still open with the gateway's own text — a quoted reply, an attachment note, the cron
    hint and run time. That text is not the person's; what they wrote after it is."""
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(
        "[Replying to M: \"we decided the project uses the fan-in consumer\"]\n\n"
        "[IMPORTANT: You are running as a scheduled cron job. The task below is yours.]\n\n"
        "[CURRENT RUN TIME: 2026-11-21 02:30 UTC]\n\n"
        "we decided to use PostgreSQL for persistence")])
    assert _fact_contents(provider) == ["we decided to use PostgreSQL for persistence"]
    provider.shutdown()


def test_envelope_skip_does_not_suppress_real_turns(tmp_path):
    """Skipping runtime envelopes must not suppress extraction for genuine
    user turns in the same batch."""
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end(
        [
            _user("[ASYNC DELEGATION BATCH COMPLETE — x]\nwe decided the project uses envelopes"),
            _user("we decided to use PostgreSQL for persistence"),
        ]
    )
    facts = _fact_contents(provider)
    assert len(facts) == 1
    assert "PostgreSQL" in facts[0]
    provider.shutdown()
