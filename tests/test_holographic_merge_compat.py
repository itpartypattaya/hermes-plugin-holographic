"""0.19.0 merge and the releases before it on one file: the compat matrix of plans/2026-10-25-fact-merge.md (v3).

- 0.18.2 (the last writer of history schema 1), loaded as a package from its git tag: a writer opened on a file after
  a merge is refused (HistoryTooNew) and does not stamp the file back; a reader of it sees the keep only, the copy in
  `as_of` under its own id without the clip (the limit of the old code, documented), `history` without `merged_into`;
  a writer that was open BEFORE the merge is not stopped by the mark: its forget of the keep leaves the copy's text,
  and 0.19.0 erases the copy on its next open, a tombstone for it; its forget of the copy leaves the merge row, which
  0.19.0 drops without a word.
- v0.7.0 (no history at all) reads `facts` and its FTS index as before; what it adds of a copy's text is a second fact
  that 0.19.0 does not fold back (the boundary "not promised", documented).

The older releases are read from their git tags (CI checks out with fetch-depth 0); skipped locally without them."""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic import HolographicMemoryProvider
from hermes_plugin_holographic.history import HistoryTooNew
from hermes_plugin_holographic.store import MemoryStore

pytestmark = pytest.mark.usefixtures("old_release_hermes")  # old releases import hermes_state_wal

REPO = Path(__file__).resolve().parent.parent
DIM = 256
KEEP_TEXT, COPY_TEXT = "Alice Smith lives in Paris near the Louvre", "Alice Smith is living in Paris by the Louvre zorblat"


def _unavailable(reason):  # release CI must run these tests, not skip them
    (pytest.fail if os.environ.get("CI") else pytest.skip)(reason)


def _load_package(tag: str, tmp_path_factory):
    """The whole plugin at *tag* as the package holo_<tag> (its modules import each other relatively)."""
    name = "holo_" + tag.replace(".", "")
    if name in sys.modules:
        return sys.modules[name]
    if shutil.which("git") is None:
        _unavailable("git not available")
    shown = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", tag], capture_output=True)
    if shown.returncode != 0:
        _unavailable(f"tag {tag} not available (fetch tags to run this test)")
    pkg = tmp_path_factory.mktemp(name) / name
    pkg.mkdir()
    with tarfile.open(fileobj=io.BytesIO(shown.stdout)) as tar:
        tar.extractall(pkg, filter="data")
    spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_legacy(tag: str, tmp_path_factory):
    """A release without history (v0.7.0): its store and retriever, as test_holographic_compat_080 loads them."""
    name = "holo_" + tag.replace(".", "")
    if name in sys.modules:
        return sys.modules[name].store, sys.modules[name].retrieval
    if shutil.which("git") is None:
        _unavailable("git not available")
    pkg = tmp_path_factory.mktemp(name) / name
    pkg.mkdir()
    for file in ("store.py", "retrieval.py", "holographic.py", "text.py"):
        shown = subprocess.run(["git", "-C", str(REPO), "show", f"{tag}:{file}"], capture_output=True)
        if shown.returncode != 0:
            _unavailable(f"tag {tag} not available (fetch tags to run this test)")
        (pkg / file).write_bytes(shown.stdout)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.store = importlib.import_module(f"{name}.store")
    module.retrieval = importlib.import_module(f"{name}.retrieval")
    return module.store, module.retrieval


@pytest.fixture(scope="module")
def old(tmp_path_factory):
    """0.18.2 as a package: its MemoryStore has a registry of its own, so its connections are another process's."""
    return _load_package("v0.18.2", tmp_path_factory)


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("fact_store", args))


def _merged_file(tmp_path) -> tuple[Path, int, int]:
    """A file 0.19.0 wrote: two facts, the second merged into the first, a confirmation on the copy."""
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    keep = store.add_fact_status(KEEP_TEXT, valid_from="2026-01-01", source={"via": "tool", "origin": "said",
                                                                               "session_id": "s1"})[0]
    copy = store.add_fact_status(COPY_TEXT, valid_from="2026-03-01", source={"via": "tool", "origin": "said",
                                                                               "session_id": "s1"})[0]
    store.record_confirmation(copy, {"session_id": "s9", "user_id": "u2", "via": "feedback"})
    store.merge_facts(keep, [copy])
    store.close()
    MemoryStore._shared.clear()
    return db, keep, copy


def _raw_bytes(db_path) -> bytes:
    data = b""
    for suffix in ("", "-wal"):
        try:
            with open(f"{db_path}{suffix}", "rb") as fh:
                data += fh.read()
        except FileNotFoundError:
            pass
    return data.lower()


def _mark(db) -> str:
    return sqlite3.connect(str(db)).execute("SELECT value FROM meta WHERE key = 'history_schema'").fetchone()[0]


# --- 0.18.2 on a file after a merge ----------------------------------------------------------------------------------


def test_an_old_writer_is_refused_and_does_not_stamp_the_file_back(tmp_path, old):
    db, keep, copy = _merged_file(tmp_path)
    assert _mark(db) == "2"
    old.store.MemoryStore._shared.clear()
    with pytest.raises(old.history.HistoryTooNew, match="update the plugin"):
        old.store.MemoryStore(str(db), hrr_dim=DIM)
    assert _mark(db) == "2"
    old.store.MemoryStore._shared.clear()
    provider = old.HolographicMemoryProvider(config={"db_path": str(db), "hrr_dim": DIM})
    provider.initialize(session_id="s")
    assert "update the plugin" in provider.system_prompt_block()  # memory is off for the old main agent
    assert "unavailable" in _call(provider, action="search", query="Alice")["error"]
    provider.shutdown()
    old.store.MemoryStore._shared.clear()
    assert _mark(db) == "2"
    reader = MemoryStore(str(db), hrr_dim=DIM, migrate=False)  # the facts are intact
    assert reader.count_facts() == 1 and reader.read("SELECT COUNT(*) FROM fact_versions")[0][0] == 2
    reader.close()
    MemoryStore._shared.clear()


def test_an_old_reader_sees_the_keep_only_and_the_copy_in_as_of_without_the_clip(tmp_path, old):
    db, keep, copy = _merged_file(tmp_path)
    old.store.MemoryStore._shared.clear()
    reader = old.store.MemoryStore(str(db), hrr_dim=DIM, migrate=False)
    retriever = old.retrieval.FactRetriever(reader, hrr_dim=DIM)
    assert [f["fact_id"] for f in reader.list_facts()] == [keep]
    assert [f["fact_id"] for f in retriever.search("Alice Louvre", min_trust=0)] == [keep]
    assert retriever.search("zorblat", min_trust=0) == []
    before = reader.versions_on("2026-06-01")["facts"]
    assert {(f["fact_id"], f["status"]) for f in before} == {(keep, "current"), (copy, "merged")}
    assert all("merged_into" not in f for f in before)
    # the limit of the old code: it does not clip the copy at the merge date, so it still "holds" today
    today = reader.versions_on(reader.today())["facts"]
    assert {(f["fact_id"], f["status"]) for f in today} == {(keep, "current"), (copy, "merged")}
    history = reader.history(copy)
    assert history["current"] is False and "merged_into" not in history
    assert history["versions"][0]["status"] == "merged"
    assert reader.past_version(COPY_TEXT) is None  # CLOSED knows no 'merged'
    assert "merged_facts" not in reader.history_counts()
    reader.close()
    old.store.MemoryStore._shared.clear()


def test_an_old_writer_open_before_the_merge_forgets_the_keep_and_the_next_open_of_this_version_erases_the_copy(
        tmp_path, old):
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    keep = store.add_fact(KEEP_TEXT)
    copy = store.add_fact(COPY_TEXT)
    store.close()
    MemoryStore._shared.clear()
    old.store.MemoryStore._shared.clear()
    writer = old.store.MemoryStore(str(db), hrr_dim=DIM)  # open before the mark rises: another process
    assert _mark(db) == "1"
    store = MemoryStore(str(db), hrr_dim=DIM)
    store.merge_facts(keep, [copy])
    assert _mark(db) == "2"
    store.close()
    MemoryStore._shared.clear()
    result = writer.forget_fact(keep)  # the mark does not stop a live process
    assert result["forgotten"] is True and "merged_copies" not in result
    assert writer.read("SELECT COUNT(*) FROM fact_versions WHERE fact_id = ?", (copy,))[0][0] == 1
    assert writer.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 1  # it knows nothing of the table
    writer.close()
    old.store.MemoryStore._shared.clear()
    assert b"zorblat" in _raw_bytes(db)
    assert _mark(db) == "2"
    store = MemoryStore(str(db), hrr_dim=DIM)  # the next open of 0.19.0: the cascade
    assert {r[0] for r in store.read("SELECT fact_id FROM forgotten")} == {keep, copy}
    assert store.read("SELECT COUNT(*) FROM fact_versions")[0][0] == 0
    assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
    assert store.forgotten_match(COPY_TEXT) is not None
    assert store.read("SELECT COUNT(*) FROM meta WHERE key = 'scrub_pending'")[0][0] == 0  # scrubbed on open
    assert b"zorblat" not in _raw_bytes(db)
    store.close()
    MemoryStore._shared.clear()


def _old_writer_forgets_the_keep(tmp_path, old, *, confirm_copy=False):
    """0.18.2 open before the merge (another process); 0.19.0 merges; the old writer then forgets the keep."""
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    keep = store.add_fact(KEEP_TEXT)
    copy = store.add_fact(COPY_TEXT)
    if confirm_copy:
        store.record_confirmation(copy, {"session_id": "s9", "user_id": "u2", "via": "feedback"})
    store.close()
    MemoryStore._shared.clear()
    old.store.MemoryStore._shared.clear()
    writer = old.store.MemoryStore(str(db), hrr_dim=DIM)
    store = MemoryStore(str(db), hrr_dim=DIM)
    store.merge_facts(keep, [copy])
    store.close()
    MemoryStore._shared.clear()
    assert writer.forget_fact(keep)["forgotten"] is True
    return db, writer, keep, copy


def test_the_cascade_on_open_also_drops_the_confirmations_of_the_erased_copy(tmp_path, old):
    """Codex review 2 of #36 (5): the copy erased on the next open of 0.19.0 takes its confirmations with it
    (who confirmed a forgotten fact is part of it)."""
    db, writer, keep, copy = _old_writer_forgets_the_keep(tmp_path, old, confirm_copy=True)
    assert writer.read("SELECT COUNT(*) FROM fact_confirmations WHERE fact_id = ?", (copy,))[0][0] == 1
    writer.close()
    old.store.MemoryStore._shared.clear()
    store = MemoryStore(str(db), hrr_dim=DIM)
    assert store.read("SELECT COUNT(*) FROM fact_versions")[0][0] == 0
    assert store.read("SELECT COUNT(*) FROM fact_confirmations")[0][0] == 0
    store.close()
    MemoryStore._shared.clear()


def test_the_cascade_on_open_runs_also_when_an_older_release_removed_another_fact(tmp_path, old):
    """Codex review 2 of #36 (2): one open finds both — a fact whose row an older release removed (its versions are
    left) and a keep the old writer forgot. The copy of that keep is erased on this open too, a tombstone for it."""
    db, writer, keep, copy = _old_writer_forgets_the_keep(tmp_path, old)
    other = writer.add_fact("Bob Brown lives in Rome")
    writer.close()
    old.store.MemoryStore._shared.clear()
    raw = sqlite3.connect(str(db))  # a release without history removes the row only (its remove)
    raw.execute("DELETE FROM fact_entities WHERE fact_id = ?", (other,))
    raw.execute("DELETE FROM facts WHERE fact_id = ?", (other,))
    raw.commit()
    raw.close()
    store = MemoryStore(str(db), hrr_dim=DIM)
    assert {r[0] for r in store.read("SELECT fact_id FROM forgotten")} == {keep, copy, other}
    assert store.read("SELECT COUNT(*) FROM fact_versions")[0][0] == 0
    assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
    assert store.forgotten_match(COPY_TEXT) is not None
    assert b"zorblat" not in _raw_bytes(db)
    store.close()
    MemoryStore._shared.clear()


def test_a_merge_row_whose_copy_an_old_writer_forgot_is_dropped_without_a_word(tmp_path, old):
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    keep = store.add_fact(KEEP_TEXT)
    copy = store.add_fact(COPY_TEXT)
    store.close()
    MemoryStore._shared.clear()
    old.store.MemoryStore._shared.clear()
    writer = old.store.MemoryStore(str(db), hrr_dim=DIM)
    store = MemoryStore(str(db), hrr_dim=DIM)
    store.merge_facts(keep, [copy])
    store.close()
    MemoryStore._shared.clear()
    assert writer.forget_fact(copy)["forgotten"] is True
    assert writer.read("SELECT COUNT(*) FROM fact_merges WHERE from_fact_id = ?", (copy,))[0][0] == 1
    writer.close()
    old.store.MemoryStore._shared.clear()
    store = MemoryStore(str(db), hrr_dim=DIM)
    assert store.read("SELECT COUNT(*) FROM fact_merges")[0][0] == 0
    assert store.count_facts() == 1 and store.history(keep)["current"] is True
    assert "merged_from" not in store.history(keep)
    assert store.merge_facts(keep, [store.add_fact("Alice Smith resides in Paris quxxle")])["merged"]  # it works on
    store.close()
    MemoryStore._shared.clear()


def test_a_file_without_merges_is_still_written_by_the_old_release(tmp_path, old):
    """The mark rises with the first merge, not on open: a downgrade before any merge costs nothing."""
    db = tmp_path / "memory_store.db"
    store = MemoryStore(str(db), hrr_dim=DIM)
    keep = store.add_fact(KEEP_TEXT)
    store.close()
    MemoryStore._shared.clear()
    assert _mark(db) == "1"
    old.store.MemoryStore._shared.clear()
    writer = old.store.MemoryStore(str(db), hrr_dim=DIM)
    assert writer.change_fact(keep, "Alice Smith lives in Lyon")["status"] == "changed"
    writer.close()
    old.store.MemoryStore._shared.clear()
    store = MemoryStore(str(db), hrr_dim=DIM)
    assert [v["status"] for v in store.history(keep)["versions"]] == ["changed", "current"]
    store.close()
    MemoryStore._shared.clear()


# --- a release without history -------------------------------------------------------------------------------------------


def test_a_release_without_history_reads_the_file_and_may_add_the_copys_text_as_a_second_fact(tmp_path, tmp_path_factory):
    legacy_store, legacy_retrieval = _load_legacy("v0.7.0", tmp_path_factory)
    db, keep, copy = _merged_file(tmp_path)
    legacy_store.MemoryStore._shared.clear()
    legacy = legacy_store.MemoryStore(str(db), hrr_dim=DIM)
    legacy._conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('integrity-check')")
    assert [f["fact_id"] for f in legacy.list_facts()] == [keep]
    retriever = legacy_retrieval.FactRetriever(legacy, hrr_dim=DIM)
    assert [f["fact_id"] for f in retriever.search("Alice Louvre", min_trust=0)] == [keep]
    twin = legacy.add_fact(COPY_TEXT)  # it knows no merge: a second fact
    assert twin > copy
    legacy.close()
    legacy_store.MemoryStore._shared.clear()
    provider = HolographicMemoryProvider(config={"db_path": str(db), "hrr_dim": DIM})
    provider.initialize(session_id="s")
    assert provider._store.count_facts() == 2  # the boundary: not folded back, said in the README
    assert [v["status"] for v in _call(provider, action="history", fact_id=twin)["versions"]] == ["current"]
    result = _call(provider, action="add", content=COPY_TEXT)
    assert result["status"] == "exists" and result["fact_id"] == twin  # the current fact wins over the merged copy
    assert _call(provider, action="history", fact_id=copy)["merged_into"] == keep
    provider.shutdown()
    MemoryStore._shared.clear()


def test_this_version_still_opens_a_file_the_old_release_marked(tmp_path, old):
    """The ordinary way round: 0.18.2 wrote the file (mark 1), 0.19.0 opens it, merges, and marks it 2."""
    db = tmp_path / "memory_store.db"
    old.store.MemoryStore._shared.clear()
    writer = old.store.MemoryStore(str(db), hrr_dim=DIM)
    keep = writer.add_fact(KEEP_TEXT)
    copy = writer.add_fact(COPY_TEXT)
    writer.close()
    old.store.MemoryStore._shared.clear()
    assert _mark(db) == "1"
    provider = HolographicMemoryProvider(config={"db_path": str(db), "hrr_dim": DIM})
    provider.initialize(session_id="s")
    assert _mark(db) == "1"
    assert _call(provider, action="merge", fact_id=keep, **{"from": [copy]})["merged"] == [copy]
    assert _mark(db) == "2"
    assert _call(provider, action="stats")["history"]["merged_facts"] == 1
    provider.shutdown()
    MemoryStore._shared.clear()
    with pytest.raises(old.history.HistoryTooNew):  # the old package's own class, not this one's
        old.store.MemoryStore(str(db), hrr_dim=DIM)
    old.store.MemoryStore._shared.clear()
    assert HistoryTooNew is not old.history.HistoryTooNew
