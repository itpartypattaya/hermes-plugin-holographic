"""One database, 0.8.0 -> an older release -> 0.8.0, for v0.1.1 (the oldest a downgrade can land on) and v0.7.0 (the
last release without history). The older release must read what 0.8.0 left (ended and forgotten facts are gone for
it, the FTS index is intact), and what it writes in between must be reconciled into the history on the next open:
an `update content=` becomes a changed version, a `remove` becomes a forget with a tombstone, a new fact gets a legacy
version, an ended fact stays ended.

The older releases are read from their git tags (CI checks out with fetch-depth 0); skipped locally without them."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from hermes_plugin_holographic.retrieval import FactRetriever
from hermes_plugin_holographic.store import MemoryStore

pytestmark = pytest.mark.usefixtures("old_release_hermes")  # old releases import hermes_state_wal

REPO = Path(__file__).resolve().parent.parent
DIM = 256


def _load(tag: str, tmp_path_factory):
    def unavailable(reason):  # release CI must run this test, not skip it
        (pytest.fail if os.environ.get("CI") else pytest.skip)(reason)

    if shutil.which("git") is None:
        unavailable("git not available")
    name = "holo_" + tag.replace(".", "")
    if name in sys.modules:
        return sys.modules[name].store, sys.modules[name].retrieval
    pkg = tmp_path_factory.mktemp(name) / name
    pkg.mkdir()
    for file in ("store.py", "retrieval.py", "holographic.py", "text.py"):
        shown = subprocess.run(["git", "-C", str(REPO), "show", f"{tag}:{file}"], capture_output=True)
        if shown.returncode != 0:
            if file == "text.py":
                continue  # v0.1.1 has no text module
            unavailable(f"tag {tag} not available (fetch tags to run this test)")
        (pkg / file).write_bytes(shown.stdout)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    store = importlib.import_module(f"{name}.store")
    retrieval = importlib.import_module(f"{name}.retrieval")
    module.store, module.retrieval = store, retrieval
    return store, retrieval


def _fts_ok(conn) -> None:
    conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('integrity-check')")


@pytest.mark.parametrize("tag", ["v0.1.1", "v0.7.0"])
def test_history_survives_an_older_release_in_between(tmp_path, tmp_path_factory, tag):
    old_store, old_retrieval = _load(tag, tmp_path_factory)
    db = tmp_path / "memory_store.db"

    new = MemoryStore(str(db), hrr_dim=DIM)
    a = new.add_fact_status("Alice Smith lives in Paris", valid_from="2026-01-01")[0]
    new.change_fact(a, "Alice Smith lives in Berlin", valid_from="2026-09-01")
    b = new.add_fact("Bob Brown teaches yoga", category="project")
    c = new.add_fact_status("Carol White works at Initech", valid_from="2026-01-01")[0]
    new.end_fact(c, valid_to="2026-06-01")
    d = new.add_fact("Dan Gold collects stamps")
    new.forget_fact(d)
    e = new.add_fact("Erin Cole plays the cello")
    new.close()

    legacy = old_store.MemoryStore(str(db), hrr_dim=DIM)
    legacy_retriever = old_retrieval.FactRetriever(legacy, hrr_dim=DIM)
    assert {f["content"] for f in legacy.list_facts()} == {"Alice Smith lives in Berlin", "Bob Brown teaches yoga",
                                                            "Erin Cole plays the cello"}
    assert legacy_retriever.search("Paris", min_trust=0) == []  # history is invisible to an older release
    assert legacy_retriever.search("Initech", min_trust=0) == []
    legacy.update_fact(b, content="Bob Brown teaches yoga on Mondays")
    legacy.remove_fact(e)
    f = legacy.add_fact("Fay Moon bakes bread", category="project")
    _fts_ok(legacy._conn)
    legacy.close()

    new = MemoryStore(str(db), hrr_dim=DIM)
    retriever = FactRetriever(new, hrr_dim=DIM)
    history_b = new.history(b)["versions"]
    assert [(v["content"], v["status"]) for v in history_b] == [("Bob Brown teaches yoga", "changed"),
                                                                ("Bob Brown teaches yoga on Mondays", "current")]
    assert history_b[0]["valid_to"].startswith("before ") and history_b[1]["source"] == "unknown"
    assert new.history(e) == {"fact_id": e, "forgotten": True, "forgotten_at": new.history(e)["forgotten_at"],
                              "versions": []}
    assert new.read("SELECT COUNT(*) FROM fact_versions WHERE fact_id = ?", (e,))[0][0] == 0
    assert [v["status"] for v in new.history(c)["versions"]] == ["ended"]  # legitimately absent, left alone
    assert [v["status"] for v in new.history(a)["versions"]] == ["changed", "current"]
    assert new.history(f)["versions"][0]["source"] == "unknown"
    assert [r["content"] for r in retriever.search_as_of("where does Alice Smith live", "2026-05-01")["results"]] == [
        "Alice Smith lives in Paris"]
    assert [r["content"] for r in retriever.search_as_of("Carol Initech", "2026-05-01")["results"]] == [
        "Carol White works at Initech"]
    _fts_ok(new._conn)
    new._conn.execute("INSERT INTO fact_versions_fts(fact_versions_fts) VALUES ('integrity-check')")
    assert new.read("SELECT COUNT(*) FROM forgotten")[0][0] == 2  # d by 0.8.0, e by the older release
    new.close()

    legacy = old_store.MemoryStore(str(db), hrr_dim=DIM)
    assert len(legacy.list_facts()) == 3
    _fts_ok(legacy._conn)
    legacy.close()
