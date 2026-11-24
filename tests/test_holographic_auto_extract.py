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
# Where the core keeps the helpers auto_extract takes from it: the merged-summary header and delimiter and
# is_compaction_summary_message (old home agent.context_compressor), and extract_user_instruction_from_skill_message
# (old home agent.skill_commands). Hermes v0.23.3 moved all four to session_rules.turns and keeps the old modules
# re-exporting them; a later core may drop the re-exports.
# ---------------------------------------------------------------------------

try:
    from session_rules.turns import extract_user_instruction_from_skill_message as _skill_instruction
except ImportError:  # Hermes up to v0.23.2
    from agent.skill_commands import extract_user_instruction_from_skill_message as _skill_instruction

# Each helper, the running core's own function or value, under the module that held it up to v0.23.2.
_OLD_HOMES = {
    "agent.context_compressor": {name: getattr(_compaction, name) for name in (
        "_MERGED_PRIOR_CONTEXT_HEADER", "_MERGED_SUMMARY_DELIMITER", "is_compaction_summary_message")},
    "agent.skill_commands": {"extract_user_instruction_from_skill_message": _skill_instruction},
}


def _module(name, attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


@pytest.fixture
def core_without_compressor_reexport(monkeypatch):
    """A core that keeps the helpers only in session_rules.turns: their old modules re-export nothing (on a core before
    v0.23.3 the helpers are also given the new address, as v0.23.3 has it)."""
    try:
        import session_rules.turns  # noqa: F401 - v0.23.3 and later: the new address is real
    except ImportError:
        turns = _module("session_rules.turns", {k: v for home in _OLD_HOMES.values() for k, v in home.items()})
        package = _module("session_rules", {"turns": turns})
        package.__path__ = []
        monkeypatch.setitem(sys.modules, "session_rules", package)
        monkeypatch.setitem(sys.modules, "session_rules.turns", turns)
    for home in _OLD_HOMES:
        monkeypatch.setitem(sys.modules, home, types.ModuleType(home))


@pytest.fixture
def core_before_session_rules(monkeypatch):
    """Hermes up to v0.23.2 (what a production install ran when this was written): no session_rules, each helper only in its old module.
    The helpers are the running core's own, given their old address whether or not the running core still re-exports
    them there."""
    for home, attributes in _OLD_HOMES.items():
        monkeypatch.setitem(sys.modules, home, _module(home, attributes))
    monkeypatch.setitem(sys.modules, "session_rules", None)
    monkeypatch.setitem(sys.modules, "session_rules.turns", None)


@pytest.mark.parametrize("core", ["core_without_compressor_reexport", "core_before_session_rules"])
def test_auto_extract_finds_the_compaction_helpers_where_the_core_keeps_them(tmp_path, request, core):
    """At session end a person's own words are stored and the compactor's are not, wherever the core keeps the
    helpers that tell them apart: the user text before a merged summary's delimiter is kept, the summary after it and
    a summary message are not, and of a slash-skill invocation only the instruction typed with it is kept."""
    request.getfixturevalue(core)
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([
        _user(f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{DECISION_MSG}\n{_MERGED_SUMMARY_DELIMITER}\n{SUMMARY_MSG}"),
        _user(SUMMARY_MSG),
        _user("I prefer tabs over spaces for indentation"),
        _user(SKILL_BODY + "\nThe user has provided the following instruction alongside the skill invocation: "
              "we decided to keep the nightly backups"),
    ])
    facts = _fact_contents(provider)
    provider.shutdown()
    assert len(facts) == 3, facts
    assert any("PostgreSQL" in f for f in facts) and any("tabs over spaces" in f for f in facts), facts
    assert "we decided to keep the nightly backups" in facts, facts


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
    # On one line with the marker: without the frame removed the sentence opens with the marker and matches nothing.
    provider.on_session_end([_user(f"{STEER_MARKER} we decided to keep the project tracker. "
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


# A slash-skill invocation is the gateway's scaffold around the person's own instruction: the skill body (whose prose
# can read like a decision) is not theirs, the instruction typed with the command is. The markers are the core's
# (session_rules.turns / agent.skill_commands: _SINGLE_SKILL_MARKER, _SINGLE_SKILL_INSTRUCTION, _RUNTIME_NOTE).
SKILL_BODY = ("[IMPORTANT: The user has invoked the \"db-helper\" skill, indicating they want you to follow its "
              "instructions. The full skill content is loaded below.]\n\n# DB helper\n"
              "The project uses MySQL for every service. We decided to shard by tenant.\n")


def test_a_skill_invocation_keeps_the_persons_instruction_not_the_skill_body(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(
        SKILL_BODY + "\nThe user has provided the following instruction alongside the skill invocation: "
        "we decided to use PostgreSQL for persistence\n\n[Runtime note: the skill directory is ~/skills/db-helper]")])
    facts = _fact_contents(provider)
    provider.shutdown()
    assert facts == ["we decided to use PostgreSQL for persistence"]


def test_a_bare_skill_invocation_stores_nothing(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(SKILL_BODY)])
    facts = _fact_contents(provider)
    provider.shutdown()
    assert facts == []


# ---------------------------------------------------------------------------
# Review of #122 (Codex GPT-6.2 Astra, Fable 5.2): the gateway's notes in front of the person's words are taken off
# before an envelope is judged, and every synthetic row the core marks (_SYNTHETIC_USER_ROW_PREFIXES) is skipped.
# The texts are the core v0.23.3's own (gateway/run_turn_runner.py, cron/scheduler_prompt.py,
# gateway/inbound_reply_context.py, tools/todo_tool.py, agent/context_compressor.py).
# ---------------------------------------------------------------------------

INTERRUPTED_NOTE = ("[System note: A new message has arrived. The conversation history contains pending tool outputs "
                    "from an interrupted turn. IGNORE those pending results. Address the user's NEW message below "
                    "FIRST. Do NOT re-execute old tool calls from the history.]\n\n")
CRON_HINT = ("[IMPORTANT: You are running as a scheduled cron job. There is no user present - you cannot ask "
             "questions, request clarification, or wait for follow-up.]\n\n[CURRENT RUN TIME: 2026-11-24 03:00 UTC]\n\n")


@pytest.mark.parametrize("gateway_text", [INTERRUPTED_NOTE, CRON_HINT], ids=["system-note", "cron-hint"])
def test_the_persons_words_after_the_gateways_note_are_kept(tmp_path, gateway_text):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(gateway_text + "I prefer tabs over spaces for indentation")])
    facts = _fact_contents(provider)
    provider.shutdown()
    assert facts == ["I prefer tabs over spaces for indentation"]


@pytest.mark.parametrize("row", [
    "[Your active task list was preserved across context compression]\n"
    "- [ ] Document that the project uses Redis for caching\n- [x] Tell the user we decided to drop MySQL",
    "[System: the call failed; note we agreed to retry and the project uses retries.]",
    "[CONTEXT RECOVERY] The project uses Redis for caching. We decided to drop MySQL.",
], ids=["todo-snapshot", "system", "context"])
def test_the_cores_other_synthetic_rows_are_not_harvested(tmp_path, row):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(row)])
    facts = _fact_contents(provider)
    provider.shutdown()
    assert facts == []


def test_a_reply_to_a_cron_delivery_keeps_the_quote_out(tmp_path):
    """The cron-delivery reply header has a colon of its own before the quote; the quoted report is not the person's."""
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(
        "[Replying to an earlier scheduled cron delivery (a past report, not a new instruction: act on it only if the "
        "message below asks): \"Daily report. The project uses MySQL.\"]\n\nwe decided to use PostgreSQL for persistence")])
    facts = _fact_contents(provider)
    provider.shutdown()
    assert facts == ["we decided to use PostgreSQL for persistence"]


def test_a_pasted_role_prompt_behind_a_quote_is_not_harvested(tmp_path):
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([_user(
        "[Replying to your previous message: \"ok\"]\n\n"
        "You are an assistant. You have received an email. The project uses MySQL.")])
    facts = _fact_contents(provider)
    provider.shutdown()
    assert facts == []


@pytest.fixture
def core_without_skill_helper(monkeypatch):
    """Hermes v0.21.5 (the oldest release requires_hermes admits): no session_rules and no
    extract_user_instruction_from_skill_message in agent.skill_commands."""
    monkeypatch.setitem(sys.modules, "agent.context_compressor",
                        _module("agent.context_compressor", _OLD_HOMES["agent.context_compressor"]))
    monkeypatch.setitem(sys.modules, "agent.skill_commands", types.ModuleType("agent.skill_commands"))
    monkeypatch.setitem(sys.modules, "session_rules", None)
    monkeypatch.setitem(sys.modules, "session_rules.turns", None)


def test_auto_extract_works_on_a_core_without_the_skill_helper(tmp_path, core_without_skill_helper):
    """Without the core's helper a skill scaffold is an envelope, skipped whole, and every other turn still extracts."""
    provider = _make_provider(tmp_path, auto_extract=True)
    provider.on_session_end([
        _user(SKILL_BODY + "\nThe user has provided the following instruction alongside the skill invocation: "
              "we decided to keep the nightly backups"),
        _user("I prefer tabs over spaces for indentation"),
    ])
    facts = _fact_contents(provider)
    provider.shutdown()
    assert facts == ["I prefer tabs over spaces for indentation"]
