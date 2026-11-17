"""Recall before a turn (prefetch) with a number, a time or a date in the message (review 2026-11-17, H1).

A bare count or year ("3", "2026"), a time ("15:30", "9.30", "5pm") or a date ("17.11", "2026-11-17") says when or how
many, not what the message is about. It counts as one of the message's words, but it is neither required of a fact
(the message "Сегодня 17.11, на каком байке ездит Антон?" is about Anton's bike, and recalled nothing) nor enough on
its own (the message "Wake me up at 15:30" is not about a ballet class at 15:30). Identifiers with letters or of
another shape ("pve-01", "v0.21.6", an IP address) stay what they were: required and decisive."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("snowballstemmer")

from hermes_plugin_holographic import HolographicMemoryProvider

FACTS = [
    "The backup server is located in Pattaya",
    "Anton rides a Honda Forza scooter",
    "Антон ездит на байке Honda Forza",
    "Резервный сервер находится в Паттайе",
    "Mia's ballet class starts at 15:30 on Thursdays",
    "The office printer has 3 paper trays",
    "The rent contract was signed in 2026",
    "pve-01 runs the family photo backup",
    "pve-010 is the old test box",
    "10.0.0.5 is the address of the NAS",
    "10.0.0.50 is the address of the printer",
]


@pytest.fixture(scope="module")
def provider(tmp_path_factory):
    home = tmp_path_factory.mktemp("numbers")
    # prefetch_repeat_turns 0: every message is judged on its own (a fact recalled once is otherwise held back)
    p = HolographicMemoryProvider(config={"db_path": str(home / "memory_store.db"), "auto_extract": "false",
                                          "prefetch_repeat_turns": 0})
    p.initialize(session_id="s", hermes_home=str(home))
    for content in FACTS:
        assert "fact_id" in json.loads(p.handle_tool_call("fact_store", {"action": "add", "content": content}))
    yield p
    p.shutdown()


def recalled(provider, message: str) -> list[str]:
    block = provider.prefetch(message)
    return [fact for fact in FACTS if fact in block]


# (message with a number, the same message without it, the fact both should recall)
WITH_NUMBERS = [
    ("Where is the backup server located in 2026?", "Where is the backup server located?",
     "The backup server is located in Pattaya"),
    ("It is 15:30 now. Which scooter does Anton ride?", "Which scooter does Anton ride?",
     "Anton rides a Honda Forza scooter"),
    ("Which scooter does Anton ride, 2 words please", "Which scooter does Anton ride, in short please",
     "Anton rides a Honda Forza scooter"),
    ("Meeting at 5pm: which scooter does Anton ride?", "Meeting soon: which scooter does Anton ride?",
     "Anton rides a Honda Forza scooter"),
    ("Сегодня 17.11, на каком байке ездит Антон?", "Сегодня, на каком байке ездит Антон?",
     "Антон ездит на байке Honda Forza"),
    ("Встреча 2026-11-20: на каком байке ездит Антон?", "Встреча скоро: на каком байке ездит Антон?",
     "Антон ездит на байке Honda Forza"),
    ("Где находится резервный сервер? Нужно до 18:00", "Где находится резервный сервер? Нужно срочно",
     "Резервный сервер находится в Паттайе"),
    ("Созвон в 9.30, где находится резервный сервер?", "Созвон утром, где находится резервный сервер?",
     "Резервный сервер находится в Паттайе"),
    ("До 20/06 узнай, где находится резервный сервер", "Сначала узнай, где находится резервный сервер",
     "Резервный сервер находится в Паттайе"),
]


@pytest.mark.parametrize("message, plain, fact", WITH_NUMBERS, ids=[m for m, _, _ in WITH_NUMBERS])
def test_a_number_in_the_message_does_not_stop_recall(provider, message, plain, fact):
    assert fact in recalled(provider, plain)  # the control: without the number the fact is recalled
    assert fact in recalled(provider, message)


# (message, a fact that shares nothing with it but the number)
NUMBER_ONLY = [
    ("Give me 3 reasons to learn Thai", "The office printer has 3 paper trays"),
    ("Wake me up at 15:30 tomorrow please", "Mia's ballet class starts at 15:30 on Thursdays"),
    ("What are the best phones of 2026?", "The rent contract was signed in 2026"),
]


@pytest.mark.parametrize("message, fact", NUMBER_ONLY, ids=[m for m, _ in NUMBER_ONLY])
def test_a_shared_number_alone_does_not_recall_a_fact(provider, message, fact):
    assert fact not in recalled(provider, message)


# Identifiers stay required and decisive, as before
def test_an_identifier_with_letters_is_still_required_and_exact(provider):
    got = recalled(provider, "What does pve-01 do?")
    assert "pve-01 runs the family photo backup" in got
    assert "pve-010 is the old test box" not in got


def test_an_address_of_digits_is_still_an_exact_identifier(provider):
    got = recalled(provider, "What is at 10.0.0.5?")
    assert "10.0.0.5 is the address of the NAS" in got
    assert "10.0.0.50 is the address of the printer" not in got
