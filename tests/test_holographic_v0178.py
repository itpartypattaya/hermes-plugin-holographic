"""0.17.8: a wrong setting does not run quietly (second review against "10 principles of good code for AI agents",
2026-10-23).

- A ``scope`` that is none of profile / user / chat / thread used to become ``profile``: with ``scope: users`` every
  person and chat shared one store. Now memory is off with the reason, as for a setting that is no number. The same
  for the other settings with a fixed set of values (recall_source, embeddings, the on/off switches).
- A config.yaml the core cannot parse: core 0.22 serves its last good copy or, without one, the defaults, and raises
  nothing; the plugin took the defaults (scope: profile) without a word. With no copy that holds this plugin's
  settings memory is off now; with one, it is used and that is logged.
- A KeyError from a bug inside an action was answered as "no such fact / name" or "missing argument", and any other
  failure's text went to the agent, where SQLite may quote a fact. Now only NotFound is "not found", only a missing
  schema argument is "missing", and the rest is named by its kind and logged.
- Failures logged at debug only (stored vectors not deleted, search by meaning and fact vectors stopped, the memory
  mirror, a damaged dictionary, the timezone) are warnings."""
from __future__ import annotations

import json
import logging
import sqlite3

import pytest

import hermes_plugin_holographic as holo
from hermes_plugin_holographic import HolographicMemoryProvider, embeddings, text
from hermes_plugin_holographic.store import MemoryStore

DIM = 64
SECRET = "Masha's PIN is 4821"


def _provider(tmp_path, **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": DIM, **config})
    p.initialize(session_id="s1", platform="telegram", user_id="111", chat_id="-100500")
    return p


def _call(p, **args):
    return json.loads(p.handle_tool_call("fact_store", args))


def _no_store_files(tmp_path) -> bool:
    return not [f for f in tmp_path.rglob("*.db")]


def _quoting(exc_type=sqlite3.OperationalError):
    exc = exc_type(f"Could not decode to UTF-8 column 'content' with text '{SECRET}'")
    exc.sqlite_errorname = "SQLITE_ERROR"
    return exc


# -- settings with a fixed set of values -------------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["users", "per-user", "chats", "profiles", "1", "user chat"])
def test_a_scope_that_is_no_scope_turns_memory_off_rather_than_sharing_one_store(tmp_path, value):
    p = _provider(tmp_path, scope=value)
    block = p.system_prompt_block()
    assert "plugins.hermes-memory-store.scope must be one of profile, user, chat, thread" in block
    assert "scope" in _call(p, action="add", content="Masha likes green tea")["error"]
    p.sync_turn("I prefer green tea in the morning.", "Noted.")
    assert p.prefetch("what tea does Masha like") == ""
    assert _no_store_files(tmp_path)  # no shared store was opened, nothing was written
    p.shutdown()


@pytest.mark.parametrize("value", ["user", "User", " chat ", "THREAD", "profile", None])
def test_every_scope_as_written_still_works(tmp_path, value):
    p = _provider(tmp_path, scope=value)
    assert "must be one of" not in p.system_prompt_block()
    assert _call(p, action="add", content="Masha likes green tea")["status"] == "added"
    p.shutdown()


@pytest.mark.parametrize("key,value", [("recall_source", "full"), ("recall_source", "yes"), ("embeddings", "maybe"),
                                       ("embeddings", "enabled"), ("auto_extract", "sure"), ("recall_indicator", "yse"),
                                       ("cron_writes", "nope"), ("mirror_memory_writes", 2)])
def test_another_setting_with_fixed_values_is_named_and_memory_stays_off(tmp_path, key, value):
    p = _provider(tmp_path, **{key: value})
    assert f"plugins.hermes-memory-store.{key} must be" in p.system_prompt_block()
    assert _no_store_files(tmp_path)
    p.shutdown()


@pytest.mark.parametrize("key,value", [("recall_source", "short"), ("recall_source", "Off"), ("embeddings", "off"),
                                       ("embeddings", False), ("auto_extract", "true"), ("auto_extract", "no"),
                                       ("recall_indicator", False), ("cron_writes", "1"), ("cron_writes", 0),
                                       ("mirror_memory_writes", "On"), ("auto_extract", None), ("recall_source", "")])
def test_those_settings_as_written_still_work(tmp_path, key, value):
    p = _provider(tmp_path, **{key: value})
    assert "must be" not in p.system_prompt_block()
    p.shutdown()


def test_a_wrong_value_is_named_not_quoted(tmp_path, caplog):
    key = "sk-proj-" + "a1B2c3D4" * 6  # a ${VAR} put in the wrong key is substituted before the plugin reads it
    with caplog.at_level(logging.WARNING):
        p = _provider(tmp_path, scope=key, recall_source=key)
    said = p.system_prompt_block() + _call(p, action="search", query="tea")["error"] + caplog.text
    assert "scope" in said and "recall_source" in said and "sk-proj" not in said and "a1B2" not in said
    p.shutdown()


# -- a config.yaml the core cannot parse ------------------------------------------------------------------------------

BROKEN = "plugins:\n  hermes-memory-store:\n    scope: user\n  - this line breaks the YAML: [\n"


def test_an_unparsable_config_with_no_good_copy_turns_memory_off(_isolated_hermes_home, caplog):
    """A fresh process, a config.yaml broken before it was ever read: the core hands out its defaults, which say
    nothing about this plugin, so nobody can tell whether scope was user. Before: memory ran on scope: profile."""
    (_isolated_hermes_home / "config.yaml").write_text(BROKEN, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        p = HolographicMemoryProvider(config=holo._load_plugin_config())
        p.initialize(session_id="s1", platform="telegram", user_id="111")
    block = p.system_prompt_block()
    assert "config.yaml could not be read" in block
    assert "config.yaml could not be read" in _call(p, action="search", query="tea")["error"]
    assert not list(_isolated_hermes_home.rglob("*.db"))
    assert "could not read plugins.hermes-memory-store" in caplog.text
    p.shutdown()


def test_an_unparsable_config_with_a_good_copy_keeps_its_settings(_isolated_hermes_home, caplog):
    """The core serves the copy it last read whole: this plugin's settings are in it, scope included."""
    config = _isolated_hermes_home / "config.yaml"
    config.write_text("plugins:\n  hermes-memory-store:\n    scope: user\n    hrr_dim: 64\n", encoding="utf-8")
    assert holo._load_plugin_config()["scope"] == "user"  # read once whole
    config.write_text(BROKEN + "# edited\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        loaded = holo._load_plugin_config()
    assert loaded["scope"] == "user" and "using its last good copy" in caplog.text
    p = HolographicMemoryProvider(config=loaded)
    p.initialize(session_id="s1", platform="telegram", user_id="111", hermes_home=str(_isolated_hermes_home))
    assert p._scope == "user" and "could not be read" not in p.system_prompt_block()
    assert _call(p, action="add", content="Masha likes green tea")["status"] == "added"
    assert list((_isolated_hermes_home / "holographic-users").glob("*.db"))  # the person's own store
    p.shutdown()


def test_a_core_that_raises_on_config_yaml_turns_memory_off(monkeypatch, tmp_path):
    """Older cores raise instead of serving a fallback; that too leaves the scope unknown."""
    import hermes_cli.config as core_config

    def broken():
        raise ValueError("line 3: OPENAI_API_KEY: sk-proj-not-a-real-key")

    monkeypatch.setattr(core_config, "load_config_readonly", broken)
    p = HolographicMemoryProvider(config=holo._load_plugin_config())
    p.initialize(session_id="s1")
    block = p.system_prompt_block()
    assert "config.yaml could not be read" in block and "sk-proj" not in block
    p.shutdown()


# -- not found, missing, and everything else ---------------------------------------------------------------------------

def test_a_bug_inside_a_fact_action_is_not_called_a_missing_fact(tmp_path, monkeypatch, caplog):
    p = _provider(tmp_path)
    fact_id = _call(p, action="add", content="Masha likes green tea")["fact_id"]

    def buggy(fid):
        return {}["hrr_vector"]

    monkeypatch.setattr(p._store, "forget_fact", buggy)
    with caplog.at_level(logging.WARNING):
        error = _call(p, action="forget", fact_id=fact_id)["error"]
    assert "does not exist" not in error and "Missing required argument" not in error
    assert "KeyError" in error and "KeyError" in caplog.text
    monkeypatch.undo()
    assert "does not exist" in _call(p, action="forget", fact_id=fact_id + 100)["error"]  # a real miss still is one
    p.shutdown()


def test_a_bug_inside_an_entity_action_is_not_called_a_missing_name(tmp_path, monkeypatch):
    p = _provider(tmp_path)
    _call(p, action="add", content="Masha Petrova likes green tea", entities=["Masha Petrova"])

    def buggy(name):
        return {}["entity_id"]

    monkeypatch.setattr(p._store, "delete_entity", buggy)
    error = _call(p, action="delete_entity", entity="Masha Petrova")["error"]
    assert "lists the names" not in error and "no entity" not in error and "KeyError" in error
    p.shutdown()


def test_a_key_error_that_is_no_argument_is_not_called_missing(tmp_path, monkeypatch):
    p = _provider(tmp_path)

    def buggy(*a, **k):
        return {}["hrr_vector"]

    monkeypatch.setattr(p._retriever, "search", buggy)
    error = _call(p, action="search", query="tea")["error"]
    assert "Missing required argument" not in error and "KeyError" in error
    assert "Missing required argument" in _call(p, action="search")["error"]  # a real one still is
    p.shutdown()


def test_a_failure_answers_with_its_kind_not_its_text(tmp_path, monkeypatch, caplog):
    p = _provider(tmp_path)

    def quoting(*a, **k):
        raise _quoting()

    monkeypatch.setattr(p._retriever, "search", quoting)
    with caplog.at_level(logging.WARNING):
        error = _call(p, action="search", query="tea")["error"]
    assert "OperationalError (SQLITE_ERROR)" in error and "4821" not in error
    assert "OperationalError (SQLITE_ERROR)" in caplog.text and "4821" not in caplog.text
    p.shutdown()


def test_a_refusal_still_says_why(tmp_path):
    """ValueError and the history errors are the plugin's own answers: their text is what the agent needs."""
    p = _provider(tmp_path)
    assert "must be a date as YYYY-MM-DD" in _call(p, action="search", query="tea", as_of="last week")["error"]
    p.shutdown()


# -- debug-only failures are warnings ---------------------------------------------------------------------------------

def test_stored_vectors_that_cannot_be_deleted_are_warned(tmp_path, monkeypatch, caplog):
    def locked(self):
        raise _quoting()

    monkeypatch.setattr(MemoryStore, "drop_vectors", locked)
    with caplog.at_level(logging.WARNING):
        p = _provider(tmp_path)
        p.system_prompt_block()
    assert "could not delete stored vectors (OperationalError (SQLITE_ERROR))" in caplog.text
    assert "4821" not in caplog.text
    p.shutdown()


ON = {"embeddings": "on", "embedding_url": "https://embeddings.example.net/v1", "embedding_model": "fake-1",
      "embedding_key_env": "TEST_EMBEDDINGS_KEY"}


class _Broken:
    def ready(self):
        raise _quoting()

    def pending(self, *a, **k):
        raise _quoting()


@pytest.fixture
def no_network(monkeypatch):
    class Opener:
        def open(self, request, timeout=None):
            raise AssertionError("no request is expected")
    monkeypatch.setattr(embeddings, "_OPENER", Opener())
    monkeypatch.setenv("TEST_EMBEDDINGS_KEY", "sk-test-not-a-real-key")


def test_search_by_meaning_that_fails_here_is_warned(tmp_path, monkeypatch, caplog, no_network):
    p = _provider(tmp_path, **ON)
    monkeypatch.setattr(p._retriever, "vector_index", lambda tag: _Broken())
    with caplog.at_level(logging.WARNING):
        assert p._meaning_of("what tea does Masha like in the morning", before_turn=False) is None
    assert "search by meaning skipped (OperationalError (SQLITE_ERROR))" in caplog.text and "4821" not in caplog.text
    p.shutdown()


def test_fact_vectors_that_stop_here_are_warned(tmp_path, monkeypatch, caplog, no_network):
    p = _provider(tmp_path, **ON)
    monkeypatch.setattr(p._retriever, "vector_index", lambda tag: _Broken())
    with caplog.at_level(logging.WARNING):
        p._catch_up_vectors()
    assert "making fact vectors stopped (OperationalError (SQLITE_ERROR))" in caplog.text and "4821" not in caplog.text
    p.shutdown()


def test_a_mirror_write_that_fails_is_warned(tmp_path, monkeypatch, caplog):
    p = _provider(tmp_path, mirror_memory_writes="true")

    def broken(*a, **k):
        raise _quoting()

    monkeypatch.setattr(p._store, "add_fact_status", broken)
    with caplog.at_level(logging.WARNING):
        p.on_memory_write("add", "memory", "Masha likes green tea")
    assert "memory mirror failed (OperationalError (SQLITE_ERROR))" in caplog.text and "4821" not in caplog.text
    p.shutdown()


def test_a_damaged_dictionary_is_warned(tmp_path, monkeypatch, caplog):
    bad = tmp_path / "ja_variants.tsv.gz"
    bad.write_bytes(b"not gzip at all")
    monkeypatch.setattr(text, "_JA_FILE", bad)
    monkeypatch.setattr(text, "_JA_TABLE", None)
    with caplog.at_level(logging.WARNING):
        text._ja_table()
    assert "ja_variants.tsv.gz could not be read" in caplog.text
    monkeypatch.setattr(text, "_JA_TABLE", None)  # the next test loads the real one again


def test_a_timezone_that_fails_is_warned(monkeypatch, caplog):
    import hermes_time

    def broken():
        raise ValueError("unknown timezone 'Asia/Bangok'")

    monkeypatch.setattr(hermes_time, "get_timezone", broken)
    with caplog.at_level(logging.WARNING):
        assert holo._hermes_timezone() is None
    assert "timezone" in caplog.text and "ValueError" in caplog.text


# -- after the Codex review of the commit ------------------------------------------------------------------------------

def test_a_bare_off_in_config_yaml_is_off_not_a_problem(_isolated_hermes_home):
    """Hermes reads config.yaml as YAML 1.1: a bare off is the bool False, a bare on True (Codex: recall_source: off
    turned memory off)."""
    (_isolated_hermes_home / "config.yaml").write_text(
        "plugins:\n  hermes-memory-store:\n    recall_source: off\n    embeddings: off\n    auto_extract: on\n"
        "    scope: profile\n    hrr_dim: 64\n", encoding="utf-8")
    loaded = holo._load_plugin_config()
    assert loaded["recall_source"] is False and loaded["auto_extract"] is True  # what the core's YAML gives
    p = HolographicMemoryProvider(config={**loaded, "db_path": str(_isolated_hermes_home / "m.db")})
    p.initialize(session_id="s1")
    assert "must be" not in p.system_prompt_block() and not p._recall_source
    p.shutdown()


def test_a_bare_on_for_recall_source_is_still_a_problem(tmp_path):
    p = _provider(tmp_path, recall_source=True)  # recall_source: on — neither off nor short
    assert "recall_source must be off or short" in p.system_prompt_block()
    p.shutdown()


def test_a_mistyped_rerank_switch_is_a_problem_not_a_silent_off(tmp_path):
    """``rerank: alwasy`` used to read as off without a word, unlike every other fixed-value switch
    (review 2026-10-23): the same key as in the schema's choices, the same route as ``embeddings``."""
    p = _provider(tmp_path, rerank="alwasy")
    assert "plugins.hermes-memory-store.rerank must be on or off" in p.system_prompt_block()
    p.shutdown()


def test_a_unicode_error_inside_an_entity_action_is_not_quoted(tmp_path, monkeypatch, caplog):
    p = _provider(tmp_path)
    _call(p, action="add", content="Masha Petrova likes green tea", entities=["Masha Petrova"])

    def quoting(name):
        raise UnicodeError(SECRET)

    monkeypatch.setattr(p._store, "delete_entity", quoting)
    with caplog.at_level(logging.WARNING):
        error = _call(p, action="delete_entity", entity="Masha Petrova")["error"]
    assert "4821" not in error and "UnicodeError" in error and "4821" not in caplog.text
    p.shutdown()


@pytest.mark.parametrize("key", [[], {}, ("entity",)])
def test_a_key_error_with_an_odd_key_is_answered(tmp_path, monkeypatch, key):
    p = _provider(tmp_path)

    def buggy(*a, **k):
        raise KeyError(key)

    monkeypatch.setattr(p._retriever, "search", buggy)
    assert "failed (KeyError)" in _call(p, action="search", query="tea")["error"]
    p.shutdown()


def test_a_new_failure_is_said_however_many_were_said_before(monkeypatch, caplog):
    from hermes_plugin_holographic import retrieval
    monkeypatch.setattr(retrieval, "_WARNED", set())
    for n in range(75):  # 25 actions x 3 kinds of failure: more than the old cap of 64
        retrieval._warn_once(f"fact_store action{n} failed", KeyError("x"))
    with caplog.at_level(logging.WARNING):
        retrieval._warn_once("memory mirror failed", _quoting())
    assert "memory mirror failed (OperationalError (SQLITE_ERROR))" in caplog.text


def test_a_bug_inside_remove_is_not_called_a_missing_fact(tmp_path, monkeypatch):
    p = _provider(tmp_path)
    fact_id = _call(p, action="add", content="Masha likes green tea")["fact_id"]

    def buggy(fid):
        return {}["hrr_vector"]

    monkeypatch.setattr(p._store, "forget_fact", buggy)
    answer = _call(p, action="remove", fact_id=fact_id)
    assert answer.get("removed") is not False and "failed (KeyError)" in answer["error"]
    monkeypatch.undo()
    assert _call(p, action="remove", fact_id=fact_id + 100) == {"removed": False}  # a real miss still is one
    p.shutdown()


# -- after the Codex review on GitHub (PR #12) -------------------------------------------------------------------------

def test_a_key_error_naming_a_given_argument_is_not_called_missing(tmp_path, monkeypatch):
    """query was given: a KeyError('query') from inside the search is a fault, not a missing argument."""
    p = _provider(tmp_path)

    def buggy(*a, **k):
        return {}["query"]

    monkeypatch.setattr(p._retriever, "search", buggy)
    error = _call(p, action="search", query="tea")["error"]
    assert "Missing required argument" not in error and "failed (KeyError)" in error
    p.shutdown()


@pytest.mark.parametrize("value", ["", "  ", None])
def test_an_empty_switch_is_its_default_not_its_opposite(tmp_path, value):
    """recall_indicator and cron_writes default to true: empty in config.yaml is that default (it was false)."""
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "m.db"), "hrr_dim": DIM,
                                          "recall_indicator": value, "cron_writes": value})
    p.initialize(session_id="s1", agent_context="cron")
    assert p._recall_indicator is True
    assert _call(p, action="add", content="Masha likes green tea")["status"] == "added"  # cron_writes stays on
    p.shutdown()


def test_an_empty_setting_does_not_change_the_cores_cached_config(_isolated_hermes_home):
    """The section comes from the core's read-only config cache: the plugin must not edit it in place."""
    (_isolated_hermes_home / "config.yaml").write_text(
        "plugins:\n  hermes-memory-store:\n    recall_indicator: ''\n    hrr_dim: 64\n", encoding="utf-8")
    loaded = holo._load_plugin_config()
    HolographicMemoryProvider(config=loaded)
    assert holo._load_plugin_config() == {"recall_indicator": "", "hrr_dim": 64}
