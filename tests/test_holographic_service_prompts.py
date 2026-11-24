"""Recall before a turn reads the person's own words, not the text Hermes wraps around them.

Measured on a live deployment (18 days, 107 recorded recalls, plans/2026-11-15-mia-service-prompts.md): the core hands
``prefetch`` the turn's message as the gateway built it. A Telegram reply carries the whole quoted message in front of
what the person wrote (``[Replying to: "..."]``), an attachment carries a note with its saved path and instructions for
the model (``[The user sent a document: ... saved at: /home/.../.hermes/cache/...]``), a cron run starts with the fixed
``[IMPORTANT: You are running as a scheduled cron job. ...]`` hint, and a finished delegation or background process
arrives as a turn of its own (``[ASYNC DELEGATION BATCH COMPLETE — ...]``, ``[IMPORTANT: Background process ...]``).
Replies were 5 % of the person's messages but 43 of 74 Telegram turns that recalled anything; in 32 of them a fact
came only from the quote, most often a spelling list of common English words that a quoted English message matched.

Requirement: what selects facts is what the person wrote. A quote, a gateway note or the cron hint does not; a turn
that is only a notice recalls nothing. The facts are this repository's own, not a copy of anyone's store; the wrapper
formats are copied from hermes-agent (gateway/run_inbound.py, gateway/run.py, cron/scheduler_prompt.py,
tools/process_registry_notifications.py)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider

# "spelling" has the shape of the fact that came most often from wrapper text on the measured deployment: a school word
# list of common English words (the public Dolch sight words), which any English sentence of the gateway or cron matches.
FACTS = {
    "spelling": ("Sight words for Mark's reading test: cold, does, fast, first, five, found, gave, goes, green, its, "
                 "made, many, off, or, pull, read, right, sing, sit, tell, their, these, those, upon, us, use, very, "
                 "wash, which, why, wish, work, would, write, your", ["Mark"]),
    "dentist": ("Lena's dentist appointment is on Tuesday at ten at the Pratumnak clinic", ["Lena"]),
    "maintainer": ("Leon maintains the holographic memory provider for Hermes Agent", ["Leon", "Hermes Agent"]),
    "cron_cache": ("The weekly cron job that cleans the photo cache is scheduled every Monday", []),
    "weather": ("The weather forecast for Pattaya comes from the Thai Meteorological Department", []),
}

# The fixed hint cron puts in front of every job prompt, byte for byte (cron/scheduler_prompt.py _CRON_HINT), and the
# run-time line after it.
CRON_HINT = (
    "[IMPORTANT: You are running as a scheduled cron job. "
    "DELIVERY: Your final response will be automatically delivered "
    "to the user — do NOT use send_message or try to deliver "
    "the output yourself. Just produce your report/output as your "
    "final response and the system handles the rest. "
    "SILENT: If there is genuinely nothing new to report, respond "
    "with exactly \"[SILENT]\" (nothing else) to suppress delivery. "
    "[SILENT] is a literal ASCII control token — never translate or "
    "rephrase it, whatever language the rest of your answer uses. "
    "Never combine [SILENT] with content — either report your "
    "findings normally, or say [SILENT] and nothing more. "
    "FAILURE: If a delegated child fails and this cron run must be "
    "recorded as failed, put [CRON_FAILURE] on the first line by itself, "
    "then explain the child failure on following lines. "
    "RECURSION: This is a run of an EXISTING scheduled job — execute "
    "the task now. NEVER create or update a cron job because of "
    "recurring or future-schedule language in the task prompt below; "
    "treat phrasing like \"each Monday\" or \"every day at 9\" as "
    "context for this run, not as a request to schedule another job.]\n\n"
    "[CURRENT RUN TIME: Saturday, 2026-11-14 07:00 +07 (UTC+07:00)]\n\n"
)


@pytest.fixture
def store(tmp_path):
    provider = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": 1024})
    provider.initialize(session_id="s0")
    back = {}
    for key, (text, entities) in FACTS.items():
        result = json.loads(provider.handle_tool_call("fact_store", {"action": "add", "content": text,
                                                                     "entities": entities}))
        back[result["fact_id"]] = key
    yield provider, back
    provider.shutdown()


_turn = iter(range(10**6))


def _recalled(store, message):
    provider, back = store
    provider.on_session_switch(f"t{next(_turn)}", reset=True)  # a fresh session: nothing held back as a repeat
    block = provider.prefetch(message)
    return {back[int(line.split()[1][1:])] for line in block.splitlines()[1:]} if block else set()


# -- guards: the person's own words keep recalling (pass before and after the fix) --------------------------------

def test_a_plain_question_recalls_its_fact(store):
    assert "dentist" in _recalled(store, "When is Lena's dentist appointment?")


def test_the_quoted_words_alone_still_recall_when_the_person_writes_them(store):
    """Without the wrapper the same words do select the list: the fix must not hide the fact itself."""
    assert "spelling" in _recalled(store, "I found the first draft, made the changes and read it again")


def test_a_reply_recalls_by_what_the_person_wrote(store):
    message = ('[Replying to: "I found the first draft, made the changes and read it again"]\n\n'
               "When is Lena's dentist appointment?")
    assert "dentist" in _recalled(store, message)


def test_an_inlined_document_still_recalls_by_its_content(store):
    """The person sent the document: its text is theirs, only the gateway's note in front of it is not."""
    message = ("[The user sent a text document: 'notes.txt'. Its content has been included below. "
               "The file is also saved at: /home/debian/.hermes/cache/documents/doc_1a2b_notes.txt]\n\n"
               "Move Lena's dentist appointment to Thursday")
    assert "dentist" in _recalled(store, message)


# -- the wrapper selects nothing and hides nothing (fail before the fix) -------------------------------------------

def test_a_cron_job_recalls_by_its_own_prompt(store):
    """~1500 characters of hint words no fact holds counted against the job's own words: the fact it needs was lost
    (on the measured deployment 13 facts came back in cron runs once the hint was left out)."""
    assert _recalled(store, CRON_HINT + "Check the weather forecast for Pattaya and summarise it") == {"weather"}

def test_a_quoted_message_does_not_select_facts(store):
    """The agent's own English message, quoted by the gateway, matched the spelling list on its common words."""
    message = ('[Replying to your previous message: "I found the first draft, made the changes and read it again"]'
               "\n\nWhen is Lena's dentist appointment?")
    assert "spelling" not in _recalled(store, message)


def test_a_short_reply_to_a_quote_recalls_nothing_from_the_quote(store):
    """The price of the rule: a bare "ok" under a quote recalls nothing (the quote is in the model's view anyway)."""
    message = '[Replying to: "I found the first draft, made the changes and read it again"]\n\nok, thanks'
    assert "spelling" in _recalled(store, "I found the first draft, made the changes and read it again")
    assert _recalled(store, message) == set()


AUDIO_NOTE = ("[The user sent an audio file attachment: 'voice.ogg'. It is saved at: "
              "/home/debian/.hermes/cache/audio/aud_3c4d.ogg. Its content is not inlined here. If the user's request "
              "involves what the audio contains, transcribe or process it yourself — for example by passing the path "
              "to a transcription or media tool — instead of asking the user to describe it. Only ask what to do "
              "with it if their intent is genuinely unclear.]\n\n")


# Guards: these wrappers did not pull a synthetic fact on the code before the fix (on the measured deployment they did,
# with that store's words); they pin that the note goes and the person's words stay.

def test_an_attachment_note_leaves_only_the_persons_question(store):
    message = ("[The user sent a document: 'invoice.pdf'. It is saved at: "
               "/home/debian/.hermes/cache/documents/doc_9f8e_invoice.pdf. Its text is not inlined here (it's a "
               "binary format such as PDF or DOCX). To read it, extract the document's text yourself — for example "
               "with the terminal tool or the ocr-and-documents skill — before answering, instead of asking the "
               "user to paste the contents.]\n\nWhen is Lena's dentist appointment?")
    assert _recalled(store, message) == {"dentist"}


def test_reply_and_attachment_notes_together_leave_only_the_persons_question(store):
    """The reply pointer is outermost, the attachment note under it (gateway/run_inbound.py order)."""
    message = ('[Replying to: "I found the first draft, made the changes and read it again"]\n\n'
               + AUDIO_NOTE + "When is Lena's dentist appointment?")
    assert _recalled(store, message) == {"dentist"}


@pytest.mark.parametrize("notice", [
    "[ASYNC DELEGATION BATCH COMPLETE — deleg_7c1]\n\nTask 1/1 (completed): Lena's dentist appointment at the "
    "Pratumnak clinic was moved to Tuesday at ten.",
    "[IMPORTANT: Background process proc_42 completed (exit code 0).\nCommand: python check.py\n"
    "Output: Lena's dentist appointment at the Pratumnak clinic is confirmed]",
    "[IMPORTANT: 2 background processes completed. Treat these results as one batch and give one consolidated "
    "response; preserve failures and actionable results.]\n\n[IMPORTANT: Background process proc_1 completed "
    "(exit code 0).\nOutput: Lena's dentist appointment is on Tuesday]",
], ids=["delegation", "background", "background-batch"])
def test_a_notice_turn_recalls_nothing(store, notice):
    """Nobody wrote this turn: a finished task reports back, the model answers the person from it."""
    body = notice.rsplit("Output: ", 1)[-1].rsplit("): ", 1)[-1].rstrip("]")
    assert "dentist" in _recalled(store, body)  # the same words, written by the person, do recall
    assert _recalled(store, notice) == set()


def test_the_authors_name_the_gateway_puts_in_front_does_not_select_their_facts(store):
    """In a shared chat every message of Leon's started with "[Leon] ": his maintainer fact came with each one."""
    provider, _ = store
    provider.on_turn_start(1, "", author_id="101", author_name="Leon")
    message = ('[Replying to: "Weekly notes about the garden and the pool"]\n\n'
               "[Leon] When is Lena's dentist appointment?")
    assert _recalled(store, message) == {"dentist"}


def test_a_name_that_is_not_the_authors_stays_in_the_words(store):
    """Only the gateway's own prefix goes: a bracketed name someone else writes is what they asked about."""
    provider, _ = store
    provider.on_turn_start(1, "", author_id="102", author_name="Lena")
    assert "maintainer" in _recalled(store, "[Leon] maintains the holographic provider for Hermes Agent, right?")


def test_a_long_quote_does_not_push_the_persons_words_out_of_the_window(store):
    """Recall reads the first 4000 characters: a long quoted message used them all and the ask was never read."""
    quote = "Weekly notes about the garden and the pool. " * 120  # ~5300 characters, no stored fact in it
    message = f'[Replying to: "{quote}"]\n\nWhen is Lena\'s dentist appointment?'
    assert "dentist" in _recalled(store, message)
