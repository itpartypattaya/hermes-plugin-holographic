"""Review 2026-11-17, H12–H17: answers the agent and the person get in corner cases.

Requirements, one per test group:
- H12: merge with a 'from' id that does not exist names that id, not the fact kept.
- H13: in user scope one person's damaged database turns memory off for that person only: the others keep their
  memory, and their prompt carries no "Unavailable" line with a path to somebody else's file.
- H14: with mirror_memory_writes on, a memory entry replaced by a text the screen keeps out leaves no copy of the old
  entry behind as a current fact (the entry it mirrored is gone).
- H15: a sync the core left running after shutdown() does not reopen the database and write to it.
- H16: a search query longer than what recall reads (4000 characters) is refused at once, with a hint; a long query
  held the provider's lock for seconds (48 000 characters: 17 s).
- H17: action given as a list is an error answer, not an exception; correct with keywords only points to update;
  update of an ended fact says it ended, of a missing one that it does not exist (was a bare {"updated": false});
  memory_map with an unknown --tz says so instead of a traceback, and names a confirming person by the name they had
  last.
The facts are synthetic.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic.memory_map import export_map


def _provider(tmp_path, **config):
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 256, **config})
    p.initialize(session_id="s1", hermes_home=str(tmp_path), platform="telegram")
    return p


def _call(p, **args):
    return json.loads(p.handle_tool_call("fact_store", args))


def _add(p, content):
    out = _call(p, action="add", content=content)
    assert out.get("status") == "added", out
    return out["fact_id"]


# -- H12 -----------------------------------------------------------------------------------------------------------

def test_merge_names_the_missing_copy_not_the_fact_kept(tmp_path):
    p = _provider(tmp_path)
    keep = _add(p, "Leon rides a Honda Vario.")
    missing_copy = _call(p, action="merge", fact_id=keep, **{"from": [999]})
    assert "#999" in missing_copy["error"] and f"#{keep} " not in missing_copy["error"], missing_copy
    missing_keep = _call(p, action="merge", fact_id=998, **{"from": [keep]})
    assert "#998" in missing_keep["error"], missing_keep
    p.shutdown()


# -- H13 -----------------------------------------------------------------------------------------------------------

def test_one_persons_damaged_store_leaves_the_others_memory_alone(tmp_path):
    users = tmp_path / "holographic-users"
    users.mkdir()
    (users / "telegram-222.db").write_bytes(b"this is not a database, " * 200)  # a damaged file
    p = HolographicMemoryProvider(config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 256,
                                          "scope": "user"})
    p.initialize(session_id="s1", hermes_home=str(tmp_path), platform="telegram", user_id="111")
    assert _call(p, action="add", content="Alice Smith is allergic to nuts.")["status"] == "added"
    p.on_turn_start(2, "hi", author_id="222", author_name="Bob")
    assert "Unavailable this session" in p.system_prompt_block()  # Bob's own store: off, and said
    assert "error" in _call(p, action="search", query="nuts")
    p.on_turn_start(3, "hi", author_id="111", author_name="Alice")
    prompt = p.system_prompt_block()
    assert "Unavailable" not in prompt and "telegram-222" not in prompt, prompt
    assert _call(p, action="search", query="allergic nuts")["count"] == 1
    p.shutdown()


# -- H14 -----------------------------------------------------------------------------------------------------------

def test_a_mirrored_entry_replaced_by_a_refused_text_leaves_no_old_copy(tmp_path):
    p = _provider(tmp_path, mirror_memory_writes=True)
    p.on_memory_write("add", "memory", "The office Wi-Fi is called Sokol.")
    assert [f["content"] for f in p._store.list_facts()] == ["The office Wi-Fi is called Sokol."]
    p.on_memory_write("replace", "memory", "The office Wi-Fi password: Sokol-2026-xyz",
                      {"previous_content": "The office Wi-Fi is called Sokol."})
    assert p._store.list_facts() == []
    p.shutdown()


def test_a_mirrored_entry_replaced_by_an_ordinary_text_still_changes_its_fact(tmp_path):
    p = _provider(tmp_path, mirror_memory_writes=True)  # the control: the usual replace is untouched
    p.on_memory_write("add", "memory", "The office Wi-Fi is called Sokol.")
    p.on_memory_write("replace", "memory", "The office Wi-Fi is called Berkut.",
                      {"previous_content": "The office Wi-Fi is called Sokol."})
    assert [f["content"] for f in p._store.list_facts()] == ["The office Wi-Fi is called Berkut."]
    p.shutdown()


# -- H15 -----------------------------------------------------------------------------------------------------------

EXTRACTED = "I use Dark mode in every editor on my machines."


def _facts(db: Path) -> list:
    with sqlite3.connect(db) as conn:
        return [r[0] for r in conn.execute("SELECT content FROM facts")]


def test_a_sync_after_shutdown_does_not_reopen_the_store(tmp_path):
    live = _provider(tmp_path / "live", auto_extract=True)  # the control: this sync does write
    live.sync_turn(EXTRACTED, "Noted.")
    live.shutdown()
    assert _facts(tmp_path / "live" / "memory_store.db"), "the control extracted nothing: the test proves nothing"

    p = _provider(tmp_path, auto_extract=True)
    p.shutdown()
    p.sync_turn(EXTRACTED, "Noted.")  # a worker the core did not stop
    assert p._stores == {} and _facts(tmp_path / "memory_store.db") == []


# -- H16 -----------------------------------------------------------------------------------------------------------

def test_a_search_query_longer_than_recall_reads_is_refused_with_a_hint(tmp_path):
    p = _provider(tmp_path)
    _add(p, "Leon rides a Honda Vario.")
    long = _call(p, action="search", query="Honda " * 700)  # 4200 characters
    assert "error" in long and "4200" in long["error"] and "4000" in long["error"], long
    fits = _call(p, action="search", query=("Honda " * 666)[:4000])
    assert "error" not in fits and fits["count"] == 1, fits
    p.shutdown()


# -- H17 -----------------------------------------------------------------------------------------------------------

def test_an_action_given_as_a_list_is_an_error_answer(tmp_path):
    p = _provider(tmp_path)
    out = _call(p, action=["add"], content="Leon rides a Honda Vario.")
    assert "action must be a string" in out["error"], out
    assert p._store.list_facts() == []
    p.shutdown()


def test_correct_with_keywords_only_points_to_update(tmp_path):
    p = _provider(tmp_path)
    fact_id = _add(p, "Leon rides a Honda Vario.")
    out = _call(p, action="correct", fact_id=fact_id, keywords=["bike"])
    assert "update" in out["error"] and f"fact_id: {fact_id}" in out["error"], out
    p.shutdown()


def test_update_of_an_ended_or_missing_fact_says_why(tmp_path):
    p = _provider(tmp_path)
    fact_id = _add(p, "Leon rides a Honda Vario.")
    assert "error" not in _call(p, action="end", fact_id=fact_id)
    ended = _call(p, action="update", fact_id=fact_id, tags="bike")
    assert "has ended" in ended.get("error", "") and "revert" in ended["error"], ended
    missing = _call(p, action="update", fact_id=999, tags="bike")
    assert "#999 does not exist" in missing.get("error", ""), missing
    p.shutdown()


def test_memory_map_says_an_unknown_zone_is_unknown(tmp_path):
    p = _provider(tmp_path)
    _add(p, "Leon rides a Honda Vario.")
    p.shutdown()
    with pytest.raises(ValueError, match="unknown time zone"):
        export_map(tmp_path / "memory_store.db", tz="Nope/Zone")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # as tests/test_memory_map.py runs the CLI
    env["PYTHONIOENCODING"] = "utf-8"
    cli = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "memory_map.py"), "--store",
                          str(tmp_path / "memory_store.db"), "--tz", "Nope/Zone"],
                         capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
    assert cli.returncode == 2 and "unknown time zone" in cli.stderr and "Traceback" not in cli.stderr, cli.stderr


def test_memory_map_names_a_confirming_person_by_their_last_name(tmp_path):
    home = tmp_path
    first = HolographicMemoryProvider(config={"db_path": str(home / "memory_store.db"), "hrr_dim": 256})
    first.initialize(session_id="old", hermes_home=str(home), platform="telegram", chat_type="dm", chat_id="301",
                     user_id="301", user_name="Дмитрий")
    early, late = _add(first, "Дима ходит в бассейн по вторникам."), _add(first, "Дима пьёт чёрный кофе.")
    first.shutdown()
    later = HolographicMemoryProvider(config={"db_path": str(home / "memory_store.db"), "hrr_dim": 256})
    later.initialize(session_id="new", hermes_home=str(home), platform="telegram", chat_type="dm", chat_id="301",
                     user_id="301", user_name="Дима")
    assert "error" not in _call(later, action="change", fact_id=early, content="Дима ходит в бассейн по средам.")
    assert "error" not in json.loads(later.handle_tool_call("fact_feedback", {"action": "confirmed",
                                                                              "fact_id": late}))
    later.shutdown()
    with sqlite3.connect(home / "memory_store.db") as conn:  # the sessions were days apart, as in production
        conn.execute("UPDATE fact_versions SET recorded_at = '2026-11-01 09:00:00' WHERE src_user_name = 'Дмитрий'")
        conn.execute("UPDATE fact_versions SET recorded_at = '2026-11-15 09:00:00' WHERE src_user_name = 'Дима'")
    snap = export_map(home / "memory_store.db")
    by = [c["by"] for f in snap["FACTS"] if f["id"] == late for c in f["conf"]]
    assert by == ["Дима"], by
