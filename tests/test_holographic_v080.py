"""0.8.0: fact history (change / end / correct / revert / forget, as_of), provenance, tombstones. docs/history.md
is the contract these tests check."""

from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic.history import HistoryMixin, HistoryUnavailable, version_state
from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore

DIM = 512


class Clock:
    """The store's "now": dates in these tests are fixed, whatever day the suite runs."""

    def __init__(self, day: str):
        self.day = day


@pytest.fixture
def clock(monkeypatch):
    c = Clock("2026-10-12")
    monkeypatch.setattr(HistoryMixin, "_now", staticmethod(lambda: f"{c.day} 12:00:00"))
    monkeypatch.setattr(HistoryMixin, "_today", lambda self: c.day)
    return c


def _provider(tmp_path, config=None, **init):
    cfg = {"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": DIM, **(config or {})}
    p = HolographicMemoryProvider(config=cfg)
    p.initialize(session_id=init.pop("session_id", "s1"), **init)
    return p


@pytest.fixture
def provider(tmp_path, clock):
    p = _provider(tmp_path)
    yield p
    p.shutdown()


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _found(provider, query, **kw):
    return [r["content"] for r in _call(provider, action="search", query=query, **kw)["results"]]


def _versions(provider, fact_id):
    return _call(provider, action="history", fact_id=fact_id)["versions"]


# --- change -------------------------------------------------------------------------------------------------------


def test_change_keeps_the_id_and_hides_the_old_text_from_ordinary_recall(provider, clock):
    clock.day = "2026-01-05"
    fid = _call(provider, action="add", content="Alice Smith lives in Paris near the Louvre", valid_from="2026-01-01")["fact_id"]
    clock.day = "2026-10-12"
    result = _call(provider, action="change", fact_id=fid, content="Alice Smith lives in Berlin",
                   valid_from="2026-09-01")
    assert result["fact_id"] == fid and result["status"] == "changed" and result["version"] == 2
    versions = _versions(provider, fid)
    assert [(v["version"], v["status"]) for v in versions] == [(1, "changed"), (2, "current")]
    assert versions[0]["valid_to"] == "2026-09-01" and versions[0]["replaced_by_version"] == 2
    assert versions[1]["valid_from"] == "2026-09-01" and versions[1]["replaces_version"] == 1
    # the old text's own words find nothing now: not FTS, not the scan, not related / probe
    assert _found(provider, "Louvre") == [] and _found(provider, "Paris") == []
    assert [f["content"] for f in _call(provider, action="probe", entity="Alice Smith")["results"]] == [
        "Alice Smith lives in Berlin"]
    assert all("Paris" not in f["content"] for f in _call(provider, action="related", entity="Alice Smith")["results"])
    assert provider.prefetch("Does Alice Smith still live near the Louvre in Paris?") == "" or \
        "Paris" not in provider.prefetch("Does Alice Smith still live near the Louvre in Paris?")


def test_as_of_returns_the_version_that_held_on_the_day(provider, clock):
    clock.day = "2026-01-05"
    fid = _call(provider, action="add", content="Alice Smith lives in Paris", valid_from="2026-01-01")["fact_id"]
    clock.day = "2026-10-12"
    _call(provider, action="change", fact_id=fid, content="Alice Smith lives in Berlin", valid_from="2026-09-01")
    before = _call(provider, action="search", query="where does Alice Smith live", as_of="2026-08-31")
    assert [r["content"] for r in before["results"]] == ["Alice Smith lives in Paris"]
    on_the_day = _call(provider, action="search", query="where does Alice Smith live", as_of="2026-09-01")
    assert [r["content"] for r in on_the_day["results"]] == ["Alice Smith lives in Berlin"]  # [from, to)
    # limit=1 ranks only what held then: today's text does not qualify for May, the old one does
    may = _call(provider, action="search", query="Alice Smith lives in Berlin", as_of="2026-05-01", limit=1)
    assert [r["content"] for r in may["results"]] == ["Alice Smith lives in Paris"]
    assert _call(provider, action="search", query="Alice", as_of="2025-12-31")["results"] == []
    listed = _call(provider, action="list", as_of="2026-08-31")
    assert [f["content"] for f in listed["facts"]] == ["Alice Smith lives in Paris"]


def test_unknown_start_and_undated_change_are_reported_as_undetermined(provider, clock):
    clock.day = "2026-03-10"
    fid = _call(provider, action="add", content="Bob Brown works at Acme Corp")["fact_id"]
    held = _call(provider, action="search", query="Bob Brown work", as_of="2026-03-10")
    assert [r["content"] for r in held["results"]] == ["Bob Brown works at Acme Corp"]
    earlier = _call(provider, action="search", query="Bob Brown work", as_of="2026-03-01")
    assert earlier["results"] == [] and [u["content"] for u in earlier["undetermined"]] == ["Bob Brown works at Acme Corp"]
    assert "start unknown" in earlier["undetermined"][0]["reason"] and "note" in earlier
    clock.day = "2026-04-10"
    _call(provider, action="change", fact_id=fid, content="Bob Brown works at Globex")  # no date
    old = _versions(provider, fid)[0]
    assert old["valid_to"] == "before 2026-04-10"
    between = _call(provider, action="search", query="Bob Brown work", as_of="2026-03-20")
    assert between["results"] == []
    assert {u["fact_id"] for u in between["undetermined"]} == {fid}  # may have held: Acme or Globex, not known
    after = _call(provider, action="search", query="Bob Brown work", as_of="2026-04-10")
    assert [r["content"] for r in after["results"]] == ["Bob Brown works at Globex"]


def test_end_closes_the_fact_without_a_successor(provider, clock):
    clock.day = "2026-02-01"
    fid = _call(provider, action="add", content="Carol White works at Initech", valid_from="2026-01-01")["fact_id"]
    clock.day = "2026-10-12"
    ended = _call(provider, action="end", fact_id=fid, valid_to="2026-06-01")
    assert ended == {"fact_id": fid, "status": "ended", "valid_to": "2026-06-01", "exact": True}
    assert _found(provider, "Carol Initech") == []
    assert _call(provider, action="list")["facts"] == []
    assert [r["content"] for r in _call(provider, action="search", query="Carol Initech", as_of="2026-05-31")["results"]] \
        == ["Carol White works at Initech"]
    assert _call(provider, action="search", query="Carol Initech", as_of="2026-06-01")["results"] == []
    assert "error" in _call(provider, action="end", fact_id=fid)  # already ended
    # revert undoes the end; change resumes an ended fact under the same id
    assert _call(provider, action="revert", fact_id=fid)["status"] == "reverted"
    assert _found(provider, "Carol Initech") == ["Carol White works at Initech"]
    _call(provider, action="end", fact_id=fid, valid_to="2026-06-01")
    resumed = _call(provider, action="change", fact_id=fid, content="Carol White works at Initech again",
                    valid_from="2026-09-01")
    assert resumed["status"] == "resumed" and resumed["fact_id"] == fid
    assert _found(provider, "Carol Initech") == ["Carol White works at Initech again"]
    assert _call(provider, action="search", query="Carol Initech", as_of="2026-07-01")["results"] == []


# --- correct / revert ---------------------------------------------------------------------------------------------


def test_a_corrected_version_is_never_returned_as_the_past(provider, clock):
    clock.day = "2026-02-01"
    fid = _call(provider, action="add", content="Dave Black lives in Rmoe", valid_from="2026-01-01")["fact_id"]
    fixed = _call(provider, action="correct", fact_id=fid, content="Dave Black lives in Rome")
    assert fixed["status"] == "corrected" and fixed["fact_id"] == fid
    assert _found(provider, "Rmoe") == []
    assert _call(provider, action="search", query="Rmoe", as_of="2026-01-15")["results"] == []
    assert _call(provider, action="search", query="Rmoe", as_of="2026-01-15")["undetermined"] == []
    versions = _versions(provider, fid)
    assert versions[0]["status"] == "corrected" and versions[1]["valid_from"] == "2026-01-01"  # the start is inherited
    # update content= (0.7.0 prompts) is a correction too
    _call(provider, action="update", fact_id=fid, content="Dave Black lives in Rome, Italy")
    assert [v["status"] for v in _versions(provider, fid)] == ["corrected", "corrected", "current"]


def test_correcting_only_the_date_moves_the_end_of_the_previous_version(provider, clock):
    clock.day = "2026-02-01"
    fid = _call(provider, action="add", content="Eve Green lives in Oslo", valid_from="2026-01-01")["fact_id"]
    clock.day = "2026-10-12"
    _call(provider, action="change", fact_id=fid, content="Eve Green lives in Bergen", valid_from="2026-09-01")
    _call(provider, action="correct", fact_id=fid, valid_from="2026-08-15")
    versions = _versions(provider, fid)
    assert versions[0]["valid_to"] == "2026-08-15" and versions[-1]["valid_from"] == "2026-08-15"
    assert [r["content"] for r in _call(provider, action="search", query="Eve Green", as_of="2026-08-20")["results"]] \
        == ["Eve Green lives in Bergen"]
    _call(provider, action="revert", fact_id=fid)  # undo the date correction: the old boundary comes back
    assert _versions(provider, fid)[0]["valid_to"] == "2026-09-01"


def test_revert_undoes_a_change(provider, clock):
    fid = _call(provider, action="add", content="Frank Hill prefers green tea")["fact_id"]
    _call(provider, action="change", fact_id=fid, content="Frank Hill prefers black coffee")
    assert _call(provider, action="revert", fact_id=fid) == {"fact_id": fid, "status": "reverted", "current_version": 1}
    assert _found(provider, "Frank Hill prefers") == ["Frank Hill prefers green tea"]
    assert [v["status"] for v in _versions(provider, fid)] == ["current", "corrected"]
    assert "error" in _call(provider, action="revert", fact_id=fid)  # a single live version: nothing to undo


# --- forget -------------------------------------------------------------------------------------------------------


def _raw_bytes(db_path) -> bytes:
    data = b""
    for suffix in ("", "-wal"):
        try:
            with open(f"{db_path}{suffix}", "rb") as fh:
                data += fh.read()
        except FileNotFoundError:
            pass
    return data.lower()


def test_forget_leaves_no_trace_in_tables_indexes_or_file_bytes(tmp_path, clock):
    provider = _provider(tmp_path, config={"auto_extract": "true"})
    db = tmp_path / "memory_store.db"
    keep = _call(provider, action="add", content="Grace Lee keeps a pet otter")["fact_id"]
    fid = _call(provider, action="add", content="Mister Zanzibarqux keeps a pet otter named Quokkaflint")["fact_id"]
    _call(provider, action="change", fact_id=fid, content="Mister Zanzibarqux keeps a pet otter named Plumwhistle")
    _call(provider, action="correct", fact_id=fid, content="Mister Zanzibarqux keeps two otters named Plumwhistle")
    assert b"quokkaflint" in _raw_bytes(db)
    result = _call(provider, action="forget", fact_id=fid)
    assert result["forgotten"] is True and result["versions"] == 3 and result["scrub"] == "done"
    store = provider._store
    for table in ("facts", "fact_versions", "fact_entities"):
        assert store.read(f"SELECT COUNT(*) FROM {table} WHERE fact_id = ?", (fid,))[0][0] == 0
    for fts in ("facts_fts", "fact_versions_fts"):
        for word in ("zanzibarqux", "quokkaflint", "plumwhistle"):
            assert store.read(f"SELECT COUNT(*) FROM {fts} WHERE {fts} MATCH ?", (word,))[0][0] == 0
    assert store.read("SELECT COUNT(*) FROM entities WHERE name LIKE '%Zanzibarqux%'")[0][0] == 0  # orphan gone
    raw = _raw_bytes(db)
    for word in (b"zanzibarqux", b"quokkaflint", b"plumwhistle"):
        assert word not in raw, word
    tomb = store.read("SELECT * FROM forgotten WHERE fact_id = ?", (fid,))[0]
    assert len(json.loads(tomb["digests"])) == 3 and b"otter" not in tomb["digests"].encode()
    assert _found(provider, "otter") == ["Grace Lee keeps a pet otter"]
    assert _call(provider, action="history", fact_id=fid)["forgotten"] is True
    # old messages do not bring a forgotten sentence back
    said = _call(provider, action="add", content="I prefer Zanzibarqux tea in the morning.")["fact_id"]
    _call(provider, action="forget", fact_id=said)
    provider.on_session_end([{"role": "user", "content": "Hi. I prefer  zanzibarqux tea in the morning!"}])
    provider.sync_turn("I prefer Zanzibarqux tea in the morning.", "ok")
    assert store.count_facts() == 1 and _found(provider, "Zanzibarqux") == []
    # a manual add of the same text is stored, with a hint that the user asked to forget it
    again = _call(provider, action="add", content="Mister Zanzibarqux keeps a pet otter named Quokkaflint")
    assert again["status"] == "added" and again["forgotten_before"]["fact_id"] == fid and "forget" in again["hint"]
    assert keep != again["fact_id"] != fid  # ids are never reused
    provider.shutdown()


def test_auto_extraction_does_not_bring_back_a_past_version(tmp_path, clock):
    provider = _provider(tmp_path, config={"auto_extract": "true"})
    fid = _call(provider, action="add", content="I prefer tea in the morning.")["fact_id"]
    _call(provider, action="change", fact_id=fid, content="I prefer coffee in the morning.")
    provider.on_session_end([{"role": "user", "content": "I prefer tea in the morning."}])
    assert provider._store.count_facts() == 1
    provider.shutdown()


def test_remove_is_forget(provider, clock):
    fid = _call(provider, action="add", content="Henry Ford likes jazz")["fact_id"]
    assert _call(provider, action="remove", fact_id=fid) == {"removed": True}
    assert provider._store.read("SELECT COUNT(*) FROM forgotten WHERE fact_id = ?", (fid,))[0][0] == 1
    assert _call(provider, action="remove", fact_id=fid) == {"removed": False}
    assert "error" in _call(provider, action="change", fact_id=fid, content="Henry Ford likes blues")


# --- a fact that comes back ---------------------------------------------------------------------------------------


def test_a_fact_that_comes_back_is_a_version_of_the_same_fact(provider, clock):
    clock.day = "2026-03-01"
    fid = _call(provider, action="add", content="Ivy Stone lives in Lisbon")["fact_id"]
    clock.day = "2026-05-01"
    _call(provider, action="change", fact_id=fid, content="Ivy Stone lives in Porto")
    again = _call(provider, action="add", content="Ivy Stone lives in Lisbon")
    assert again["status"] == "past_version" and again["fact_id"] == fid and again["stored"] is False
    assert f"change {{fact_id: {fid}" in again["hint"]
    clock.day = "2026-08-01"
    back = _call(provider, action="change", fact_id=fid, content="Ivy Stone lives in Lisbon")
    assert back["fact_id"] == fid and back["version"] == 3
    near = _call(provider, action="add", content="Ivy Stone lives in Porto now")
    assert [p["fact_id"] for p in near.get("past_versions", [])] == [fid] and "hint" in near


def test_supersedes_from_older_prompts_is_a_change(provider, clock):
    old = _call(provider, action="add", content="Jack Frost lives in Oslo")["fact_id"]
    new = _call(provider, action="add", content="Jack Frost lives in Tromso", supersedes=old)
    assert new["fact_id"] == old and new["superseded"] == old and new["status"] == "changed"
    other = _call(provider, action="add", content="Kim Lane lives in Oslo")["fact_id"]
    # what it changes into is already a fact: the old one ends and that one is returned
    merged = _call(provider, action="change", fact_id=old, content="Kim Lane lives in Oslo")
    assert merged == {"fact_id": other, "status": "exists", "ended": old}


# --- validation and atomicity -------------------------------------------------------------------------------------


def test_dates_are_validated(provider, clock):
    fid = _call(provider, action="add", content="Liam Moss lives in Rome", valid_from="2026-02-01")["fact_id"]
    assert "future" in _call(provider, action="add", content="Mia Moss lives in Rome", valid_from="2026-12-01")["error"]
    assert "YYYY-MM-DD" in _call(provider, action="add", content="Mia Moss lives in Rome", valid_from="01.02.2026")["error"]
    assert "not after" in _call(provider, action="change", fact_id=fid, content="Liam Moss lives in Milan",
                                valid_from="2026-01-01")["error"]
    assert "already what the fact says" in _call(provider, action="change", fact_id=fid, content="Liam Moss lives in Rome")["error"]
    assert "YYYY-MM-DD" in _call(provider, action="search", query="Liam", as_of="yesterday")["error"]
    for action in ("probe", "related"):
        assert "as_of" in _call(provider, action=action, entity="Liam Moss", as_of="2026-03-01")["error"]
    assert "as_of" in _call(provider, action="reason", entities=["Liam Moss"], as_of="2026-03-01")["error"]
    assert "error" in _call(provider, action="correct", fact_id=fid)


def test_a_failure_midway_changes_nothing(provider, clock, monkeypatch):
    fid = _call(provider, action="add", content="Nora Fox lives in Kyiv")["fact_id"]
    before = provider._store.read("SELECT * FROM fact_versions ORDER BY version_id")

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(MemoryStore, "_rewrite_fact", boom)
    assert "disk full" in _call(provider, action="change", fact_id=fid, content="Nora Fox lives in Lviv")["error"]
    assert [tuple(r) for r in provider._store.read("SELECT * FROM fact_versions ORDER BY version_id")] == \
        [tuple(r) for r in before]
    assert _found(provider, "Nora Fox") == ["Nora Fox lives in Kyiv"]


# --- provenance ---------------------------------------------------------------------------------------------------


def test_source_is_recorded_shown_in_list_and_kept_out_of_recall_by_default(tmp_path, clock):
    ctx = dict(platform="telegram", chat_id="-1001", chat_type="group", thread_id="7", user_id="111",
               user_name="Leon", chat_name="Secret Club Chat")
    provider = _provider(tmp_path, **ctx)
    provider.on_turn_start(1, "hi", author_id="222", author_name="Kostya")
    fid = _call(provider, action="add", content="Olga Park runs the Pattaya meetup")["fact_id"]
    src = _call(provider, action="list")["facts"][0]["source"]
    assert src == {"via": "tool", "platform": "telegram", "chat_type": "group", "chat_id": "-1001", "thread_id": "7",
                   "session_id": "s1", "user_id": "222", "user_name": "Kostya", "recorded_on": "2026-10-12"}
    assert _versions(provider, fid)[0]["source"]["user_name"] == "Kostya"
    assert "Kostya" in _call(provider, action="export")["markdown"]
    recalled = provider.prefetch("who runs the Pattaya meetup?")
    assert "Olga Park" in recalled and "Kostya" not in recalled and "recorded" not in recalled
    provider.shutdown()
    assert b"secret club chat" not in _raw_bytes(tmp_path / "memory_store.db")  # chat names are not stored

    short = _provider(tmp_path, config={"recall_source": "short"}, **{**ctx, "chat_id": "-1002"})
    line = short.prefetch("who runs the Pattaya meetup?")
    assert "(recorded 2026-10-12 · Kostya · another chat)" in line and "-1001" not in line
    short.shutdown()


def test_an_anonymous_turn_in_a_shared_chat_is_nobody(tmp_path, clock):
    provider = _provider(tmp_path, platform="telegram", chat_id="-1001", user_id="111", user_name="Leon")
    provider.on_turn_start(1, "hi", author_id="222", author_name="Kostya")
    provider.on_turn_start(2, "hi")  # no author: not the session's first user
    _call(provider, action="add", content="Pavel Gray plays chess on Fridays")
    src = _call(provider, action="list")["facts"][0]["source"]
    assert "user_id" not in src and src["chat_id"] == "-1001"
    provider.shutdown()


def test_facts_from_before_0_8_0_have_an_unknown_source(tmp_path, clock):
    db = tmp_path / "memory_store.db"
    legacy = MemoryStore(str(db), hrr_dim=DIM, migrate=False)  # no history yet: as an older version wrote it
    legacy._conn.execute("DELETE FROM meta WHERE key = 'history_schema'")
    legacy.add_fact("Quinn Reed lives in Lima")
    legacy.close()
    provider = _provider(tmp_path)
    fact = _call(provider, action="list")["facts"][0]
    assert fact["source"] == "unknown"
    assert _versions(provider, fact["fact_id"])[0]["valid_from"] == "unknown"
    assert "source unknown" in _call(provider, action="export")["markdown"]
    provider.shutdown()


# --- read-only contexts, mirror, scopes, stats --------------------------------------------------------------------


def test_read_only_contexts_do_not_migrate(tmp_path, clock):
    db = tmp_path / "memory_store.db"
    raw = sqlite3.connect(db)
    raw.executescript("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL UNIQUE, "
                      "category TEXT DEFAULT 'general', tags TEXT DEFAULT '', trust_score REAL DEFAULT 0.5, "
                      "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0, "
                      "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                      "hrr_vector BLOB);"
                      "INSERT INTO facts (content) VALUES ('Rita Vale lives in Quito');")
    raw.close()
    reader = _provider(tmp_path, agent_context="subagent")
    assert "set up" in _call(reader, action="search", query="Rita", as_of="2026-10-01")["error"]
    assert reader._store.read("SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'fact_versions%' "
                              "OR name = 'forgotten'")[0][0] == 0  # a reader creates no history tables
    reader.shutdown()
    writer = _provider(tmp_path)
    assert writer._store.read("SELECT COUNT(*) FROM fact_versions")[0][0] == 1
    writer.shutdown()


def test_mirrored_memory_writes_change_and_forget(tmp_path, clock):
    provider = _provider(tmp_path, config={"mirror_memory_writes": "true"})
    provider.on_memory_write("add", "memory", "Deploys go through the release script")
    fid = _call(provider, action="list")["facts"][0]["fact_id"]
    provider.on_memory_write("replace", "memory", "Deploys go through GitHub Actions",
                             metadata={"previous_content": "Deploys go through the release script"})
    assert [v["status"] for v in _versions(provider, fid)] == ["changed", "current"]
    assert _versions(provider, fid)[1]["source"]["via"] == "mirror"
    provider.on_memory_write("remove", "memory", "", metadata={"previous_content": "Deploys go through GitHub Actions"})
    assert _call(provider, action="history", fact_id=fid)["forgotten"] is True
    provider.shutdown()


def test_each_chat_store_has_its_own_history_and_tombstones(tmp_path, clock):
    a = _provider(tmp_path, config={"scope": "chat"}, platform="telegram", chat_id="-1001")
    b = _provider(tmp_path, config={"scope": "chat"}, platform="telegram", chat_id="-1002")
    fa = _call(a, action="add", content="Sam Hart hosts the quiz night")["fact_id"]
    fb = _call(b, action="add", content="Sam Hart hosts the quiz night")["fact_id"]
    _call(a, action="forget", fact_id=fa)
    assert a._store.read("SELECT COUNT(*) FROM forgotten")[0][0] == 1
    assert b._store.read("SELECT COUNT(*) FROM forgotten")[0][0] == 0
    assert _found(b, "quiz night") == ["Sam Hart hosts the quiz night"] and fb
    a.shutdown()
    b.shutdown()


def test_stats_count_the_history(provider, clock):
    fid = _call(provider, action="add", content="Tom Ash lives in Riga")["fact_id"]
    _call(provider, action="change", fact_id=fid, content="Tom Ash lives in Tallinn")
    gone = _call(provider, action="add", content="Uma Bell lives in Vilnius")["fact_id"]
    _call(provider, action="forget", fact_id=gone)
    stats = _call(provider, action="stats")
    assert stats["history"] == {"past_versions": 1, "corrected": 0, "ended_facts": 0, "merged_facts": 0, "forgotten": 1}


def test_version_state_rules():
    row = {"status": "changed", "valid_from": None, "valid_to": "2026-04-10", "valid_to_exact": 0,
           "recorded_on": "2026-03-10"}
    assert version_state(row, "2026-03-09")[0] == "undetermined"  # before it was recorded
    assert version_state(row, "2026-03-10")[0] == "holds"  # the day it was recorded
    assert version_state(row, "2026-03-11")[0] == "undetermined"  # changed at some point before the bound
    assert version_state(row, "2026-04-10")[0] == "no"
    assert version_state({**row, "status": "corrected"}, "2026-03-10")[0] == "no"


def test_history_unavailable_is_explicit(tmp_path, clock):
    store = MemoryStore(str(tmp_path / "x.db"), hrr_dim=DIM, migrate=False)
    store._conn.execute("DELETE FROM meta WHERE key = 'history_schema'")
    with pytest.raises(HistoryUnavailable):
        store.history(1)
    with pytest.raises(HistoryUnavailable):
        store.versions_on("2026-01-01")
    store.close()


def test_retriever_search_without_history_tables_ready_is_unchanged(tmp_path, clock):
    store = MemoryStore(str(tmp_path / "y.db"), hrr_dim=DIM)
    retriever = FactRetriever(store, hrr_dim=DIM)
    store.add_fact("Vera Lin lives in Minsk")
    assert [f["content"] for f in retriever.search("Vera Minsk")] == ["Vera Lin lives in Minsk"]
    store.close()


def test_a_cron_job_writes_through_its_explicit_tool_calls_only(tmp_path, clock):
    writer = _provider(tmp_path)
    fid = _call(writer, action="add", content="Wendy Park lives in Seoul")["fact_id"]
    writer.shutdown()
    cron = _provider(tmp_path, agent_context="cron", platform="cron", session_id="cron_job_1",
                     config={"auto_extract": "true", "mirror_memory_writes": "true"})
    added = _call(cron, action="add", content="Xavi Ruiz teaches salsa on Thursdays", keywords=["dance class"])
    assert added["status"] == "added"
    assert _call(cron, action="update", fact_id=fid, keywords=["Korea", "capital"])["updated"] is True
    assert _call(cron, action="change", fact_id=fid, content="Wendy Park lives in Busan")["status"] == "changed"
    assert "fact_id" in json.loads(cron.handle_tool_call("fact_feedback", {"action": "helpful", "fact_id": fid}))
    assert _versions(cron, added["fact_id"])[0]["source"]["via"] == "tool"
    cron.on_session_end([{"role": "user", "content": "I prefer tea over coffee in the morning."}])  # nothing automatic
    cron.on_memory_write("add", "user", "Prefers short answers")
    _call(cron, action="search", query="salsa")
    cron.shutdown()
    store = MemoryStore(str(tmp_path / "memory_store.db"), hrr_dim=DIM)
    assert store.count_facts() == 2 and store.read("SELECT SUM(retrieval_count) FROM facts")[0][0] == 0
    store.close()
    # a forgotten sentence found again in old messages is not stored by a scheduled job
    main = _provider(tmp_path)
    gone = _call(main, action="add", content="Zoe Quinn owes Leon 500 baht")["fact_id"]
    _call(main, action="forget", fact_id=gone)
    main.shutdown()
    cron = _provider(tmp_path, agent_context="cron", platform="cron")
    again = _call(cron, action="add", content="Zoe Quinn owes Leon 500 baht")
    assert again["status"] == "forgotten" and again["stored"] is False and again["fact_id"] == gone
    cron.shutdown()
    off = _provider(tmp_path, agent_context="cron", config={"cron_writes": "false"})
    assert "read-only" in _call(off, action="add", content="Yuki Sato plays go")["error"]
    off.shutdown()


# --- review fixes (Codex, 2026-10-12) ---------------------------------------------------------------------------


def test_revert_of_a_resume_leaves_the_fact_ended_and_resumable(provider, clock):
    fid = _call(provider, action="add", content="Ada Hale works at Umbrella", valid_from="2026-01-01")["fact_id"]
    _call(provider, action="end", fact_id=fid, valid_to="2026-03-01")
    _call(provider, action="change", fact_id=fid, content="Ada Hale works at Umbrella again", valid_from="2026-05-01")
    assert _call(provider, action="revert", fact_id=fid)["ended"] == "2026-03-01"
    assert _found(provider, "Ada Hale") == []
    assert _call(provider, action="revert", fact_id=fid)["status"] == "reverted"  # the end itself can be undone
    assert _found(provider, "Ada Hale") == ["Ada Hale works at Umbrella"]
    _call(provider, action="end", fact_id=fid, valid_to="2026-03-01")
    resumed = _call(provider, action="change", fact_id=fid, content="Ada Hale works at Umbrella again",
                    valid_from="2026-06-01")
    assert resumed["status"] == "resumed"
    # change -> revert -> end -> revert
    other = _call(provider, action="add", content="Bo Lin lives in Hanoi")["fact_id"]
    _call(provider, action="change", fact_id=other, content="Bo Lin lives in Hue")
    _call(provider, action="revert", fact_id=other)
    _call(provider, action="end", fact_id=other)
    assert _call(provider, action="revert", fact_id=other)["status"] == "reverted"
    assert _found(provider, "Bo Lin") == ["Bo Lin lives in Hanoi"]


def test_forgetting_again_retries_the_file_rewrite(provider, clock, monkeypatch):
    fid = _call(provider, action="add", content="Cy Moor hides the key under the mat")["fact_id"]
    store = provider._store
    real_scrub = type(store).scrub
    monkeypatch.setattr(type(store), "scrub", lambda self: False)  # another process holds the file
    assert _call(provider, action="forget", fact_id=fid)["scrub"] == "pending"
    monkeypatch.setattr(type(store), "scrub", real_scrub)
    again = _call(provider, action="forget", fact_id=fid)
    assert again["already"] is True and again["scrub"] == "done"
    assert store.read("SELECT COUNT(*) FROM meta WHERE key = 'scrub_pending'")[0][0] == 0


def test_as_of_ranking_ignores_versions_outside_the_date(provider, clock):
    clock.day = "2026-02-01"
    apple = _call(provider, action="add", content="Dee keeps apple pear jam", valid_from="2026-01-01")["fact_id"]
    plain = _call(provider, action="add", content="Dee keeps pear cider", valid_from="2026-01-01")["fact_id"]
    before = _call(provider, action="search", query="apple pear", as_of="2026-01-15")["results"]
    # many corrected versions mentioning "apple" in the versions index, none of them held on the date
    noisy = _call(provider, action="add", content="Eli grows apple 0", valid_from="2026-01-20")["fact_id"]
    for i in range(1, 30):
        _call(provider, action="correct", fact_id=noisy, content=f"Eli grows apple {i}")
    after = _call(provider, action="search", query="apple pear", as_of="2026-01-15")["results"]
    assert [(r["fact_id"], r["score"]) for r in after] == [(r["fact_id"], r["score"]) for r in before]
    assert after[0]["fact_id"] == apple and plain in [r["fact_id"] for r in after]


def test_as_of_rank_counts_words_no_version_contains(provider, clock):
    clock.day = "2026-02-01"
    _call(provider, action="add", content="Fay grows apple trees", valid_from="2026-01-01")
    _call(provider, action="add", content="Fay sells pear jam", valid_from="2026-01-01")
    plain = _call(provider, action="search", query="apple", as_of="2026-01-15")["results"][0]["score"]
    vague = _call(provider, action="search", query="apple zzqxunknown", as_of="2026-01-15")["results"][0]["score"]
    assert vague < plain  # the unknown word is part of what was asked
