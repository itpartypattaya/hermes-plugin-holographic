"""NousResearch/hermes-agent#22907: auto_extract stored replies to the assistant as preferences. "I like that, sounds
good" and "ok, I love it" match "I like / love …", but what they like is whatever the assistant just said: stored, the
fact says nothing on its own and comes back in later sessions as a preference. A sentence whose preference has only
a pronoun or "this/that" + one word as its object ("I prefer this one", "I like this approach better") is now skipped;
a preference that names its object is stored as before, and so is a reaction followed by a real one."""

from __future__ import annotations

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64,
                                          "auto_extract": "true"})
    p.initialize(session_id="s")
    yield p
    p.shutdown()


def _stored(provider):
    return [f["content"] for f in provider._store.list_facts(min_trust=0.0, limit=100)]


# the issue's examples, and the same reply in other words
REACTIONS = [
    "I like that, sounds good",
    "ok, I love it",
    "I love it!",
    "Great, I like this approach better.",
    "I prefer this one, let's go with it",
    "Yes, I like that idea",
    "I like that 👍",  # Codex review: an emoji or a dash ends the reply as a comma does
    "I love it — sounds good",
    "I like those a lot, thanks",
    "Ок, я люблю это",
]


@pytest.mark.parametrize("message", REACTIONS)
def test_a_reply_to_the_assistant_is_not_a_preference(provider, message):
    provider.sync_turn(message, "Done.", session_id="s")
    assert _stored(provider) == []


# preferences that say what they are about: stored (each in its own message)
PREFERENCES = [
    "I prefer dark mode in every editor.",
    "I always check git status before committing",  # a habit, said as one: kept (see CHANGELOG)
    "I like it when answers are short and have sources",  # "it" that the sentence itself explains
    "I like that you ask before deleting files",  # "that" as a conjunction: a clause follows
    "I use this laptop for work and the desktop for games",
    "I like Node.js a lot",
    "I use this Node.js library for all my projects.",  # Codex review: a dot inside a name ends nothing
    "I use this Node.js package for every backend.",
    "I prefer this format: JSON with two-space indentation.",  # Codex review on GitHub: a colon explains
    "Я люблю это кафе на Джомтьене за тишину",
    "I love them, my kids are my everything",  # "them", «его»: people more often than a proposal
    "Я люблю его, это мой брат",
]


@pytest.mark.parametrize("message", PREFERENCES)
def test_a_preference_that_names_its_object_is_stored(provider, message):
    provider.sync_turn(message, "Noted.", session_id="s")
    assert _stored(provider) == [message]


def test_a_reaction_followed_by_a_preference_in_one_sentence_is_stored(provider):
    """The reaction does not take a real preference of the same sentence with it."""
    provider.sync_turn("I like that, I always use tabs for indentation", "Noted.", session_id="s")
    assert _stored(provider) == ["I like that, I always use tabs for indentation"]


def test_a_reaction_and_a_preference_in_two_sentences(provider):
    provider.sync_turn("I love it! I prefer green tea in the morning.", "Noted.", session_id="s")
    assert _stored(provider) == ["I prefer green tea in the morning."]
