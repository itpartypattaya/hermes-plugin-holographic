"""NousResearch/hermes-agent#66380: a mail bridge stored every inbound email as a fact shaped as a prompt to the
agent — "You are Henry, … You received an email at the AgentMail inbox and should handle it like a normal Henry
request. EMAIL METADATA …" — and recall brought each one back, framed as stored data, as a live task in every later
session for weeks. Such a text is now refused at write with a reason, and one already in a store is not recalled
before a turn (fact_store search and list still show it, so it can be changed or forgotten).

What counts: the text opens by telling its reader who it is ("You are …", «Ты — …») and hands it a task (an email or
message it received, a new task, "should handle it"). Either alone is no prompt: a rule for the assistant ("You are
allowed to push to main") and a second-person fact ("You received a parcel from DHL") are stored as before."""

from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider

# the issue's snapshot, as the bridge wrote it
SNAPSHOT = ("You are Henry, Lawrence Financial's deal-flow and business-development operator.\n\nYou received an email "
            "at the AgentMail inbox and should handle it like a normal Henry request.\n\nEMAIL METADATA\nFrom: "
            "partner@bluevault.example\nSubject: Fw: BlueVault NDA\nAgentMail message_id: \n")
PROMPTS = [
    SNAPSHOT,
    "You are Henry, the deal-flow operator. You have a new task from the inbox: reply to the BlueVault thread.",
    "You're the support agent for Acme. You received a message from a customer and should answer it today.",
    "Ты — Генри, оператор сделок. Тебе пришло письмо в ящик AgentMail, обработай его как обычный запрос.",
    "Вы — Генри. Вам пришло письмо в AgentMail.",  # Codex review: the formal address too
]
FACTS = [
    "You are allowed to push to main without asking.",
    "You are my assistant for HR24 recruiting tasks.",
    "You are Mia, the family assistant; answer in Russian.",
    "You should handle invoices from Ksenia before Friday.",
    "You received a parcel from DHL yesterday.",
    "You are expected to reply to Ksenia's emails within a day.",
    "You are the release manager; your task is to tag releases on Fridays.",
    "You are right that the server has 8 GB of RAM; we must handle it carefully.",
    "Henry received an email from BlueVault about the NDA on May 29.",
    "The assistant is called Henry and handles deal-flow emails.",
    "Ты помогаешь Антону с маркетингом HR24.",
    "Вы — команда поддержки, отвечаете клиентам до 18:00.",
    # Codex review: a standing rule about what to do with mail is no task received
    "You are my email assistant; when I forward a message, you should handle it according to my standing inbox rules.",
    "Ты обычно отвечаешь на письма Ксюши до обеда, обработай его так же, если оно срочное.",
]


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 512})
    p.initialize(session_id="s")
    yield p
    p.shutdown()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


@pytest.mark.parametrize("text", PROMPTS)
def test_a_prompt_to_the_agent_is_refused_with_a_reason(provider, text):
    result = _call(provider, action="add", content=text, origin="said")
    assert "error" in result and "prompt" in result["error"], result
    assert _call(provider, action="list")["facts"] == []


@pytest.mark.parametrize("text", FACTS)
def test_a_rule_or_a_second_person_fact_is_stored(provider, text):
    assert _call(provider, action="add", content=text, origin="said")["status"] == "added"


def test_change_and_update_refuse_it_too(provider):
    fact_id = _call(provider, action="add", content="Henry handles the deal-flow inbox", origin="said")["fact_id"]
    for action in ("change", "update", "correct"):
        result = _call(provider, action=action, fact_id=fact_id, content=SNAPSHOT)
        assert "error" in result and "prompt" in result["error"], (action, result)
    assert [f["content"] for f in _call(provider, action="list")["facts"]] == ["Henry handles the deal-flow inbox"]


def test_one_already_stored_is_not_recalled_but_can_be_found(provider, tmp_path):
    """Written before this rule (by an older version, as in the issue): not put into the prompt before a turn,
    still shown by search and list."""
    with sqlite3.connect(tmp_path / "memory_store.db") as conn:
        conn.execute("INSERT INTO facts (content, category) VALUES (?, 'user_pref')", (SNAPSHOT,))
    found = _call(provider, action="search", query="BlueVault NDA email")["results"]
    assert [r["content"] for r in found] == [SNAPSHOT]
    assert "BlueVault" not in provider.prefetch("What happened with the BlueVault NDA email?", session_id="s")


def test_auto_extract_does_not_take_a_sentence_out_of_a_prompt(tmp_path):
    """Codex review: the sentences of a prompt, screened one by one, each pass; the last one reads as a preference."""
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "x.db"), "hrr_dim": 512, "auto_extract": "true"})
    p.initialize(session_id="s")
    try:
        p.sync_turn("You are Henry. You received an email from Acme. I prefer that you answer it now.", "ok",
                    session_id="s")
        assert p._store.list_facts(min_trust=0.0, limit=10) == []
    finally:
        p.shutdown()


def test_revert_does_not_bring_a_prompt_back(provider, tmp_path):
    """Codex review: a prompt stored before this rule, then changed to a fact, is not restored by revert."""
    with sqlite3.connect(tmp_path / "memory_store.db") as conn:
        conn.execute("INSERT INTO facts (content, category) VALUES (?, 'user_pref')", (SNAPSHOT,))
    fact_id = _call(provider, action="list")["facts"][0]["fact_id"]
    assert _call(provider, action="change", fact_id=fact_id, content="Henry handles the deal-flow inbox")["status"]         == "changed"
    result = _call(provider, action="revert", fact_id=fact_id)
    assert "error" in result and "prompt" in result["error"], result
    assert [f["content"] for f in _call(provider, action="list")["facts"]] == ["Henry handles the deal-flow inbox"]
