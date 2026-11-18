"""One database, three versions in turn: this release, then v0.1.1 (what a downgrade or the bundled core copy runs),
then this release again. Facts, links and the FTS index must survive every hand-over in both directions.

v0.1.1 is read from the git tag, so the test needs the tags (CI checks out with fetch-depth 0); it is skipped when
git or the tag is unavailable."""

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
OLD_TAG = "v0.1.1"
DIM = 256


@pytest.fixture(scope="module")
def old(tmp_path_factory):
    """The v0.1.1 store and retriever modules, loaded as their own package."""
    def unavailable(reason):  # release CI must run this test, not skip it
        (pytest.fail if os.environ.get("CI") else pytest.skip)(reason)

    if shutil.which("git") is None:
        unavailable("git not available")
    pkg = tmp_path_factory.mktemp("old") / "holo_v011"
    pkg.mkdir()
    for name in ("store.py", "retrieval.py", "holographic.py"):
        shown = subprocess.run(["git", "-C", str(REPO), "show", f"{OLD_TAG}:{name}"], capture_output=True)
        if shown.returncode != 0:
            unavailable(f"tag {OLD_TAG} not available (fetch tags to run this test)")
        (pkg / name).write_bytes(shown.stdout)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("holo_v011", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["holo_v011"] = module
    spec.loader.exec_module(module)
    from holo_v011 import retrieval as old_retrieval, store as old_store
    return old_store, old_retrieval


def _fts_ok(conn) -> None:
    conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('integrity-check')")


def _contents(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT content FROM facts")}


def test_new_old_new_on_one_database(tmp_path, old):
    old_store, old_retrieval = old
    db = tmp_path / "memory_store.db"

    new = MemoryStore(str(db), hrr_dim=DIM)
    a = new.add_fact("Alice Smith lives in Paris", category="user_pref")
    new.add_fact("Alice Smith and Bob Brown run a coffee shop", category="project")
    hermes = new.add_fact("Hermes Agent runs on the server", category="project")
    new.add_alias("Bob Brown", "Bobby Brown")
    new.update_fact(a, content="Alice Smith lives in London")
    new.record_retrievals([a])
    new.record_feedback(a, helpful=True)
    new.close()

    legacy = old_store.MemoryStore(str(db), hrr_dim=DIM)
    legacy_retriever = old_retrieval.FactRetriever(legacy, hrr_dim=DIM)
    assert {f["content"] for f in legacy.list_facts()} == {"Alice Smith lives in London",
                                                            "Alice Smith and Bob Brown run a coffee shop",
                                                            "Hermes Agent runs on the server"}
    assert [f["content"] for f in legacy_retriever.search("London", min_trust=0)] == ["Alice Smith lives in London"]
    legacy_retriever.probe("Alice Smith", category="user_pref")  # the bank path must not fail on what we left
    b = legacy.add_fact("Bob Brown teaches yoga", category="project")
    repo = legacy.add_fact('The "hermes-agent" repo moved', category="project")  # a second, keyless entity
    assert legacy_retriever.probe("Bobby Brown")  # old alias lookup reads the legacy column we keep in step
    legacy.update_fact(b, content="Bob Brown teaches yoga on Mondays")
    _fts_ok(legacy._conn)
    legacy.close()

    new = MemoryStore(str(db), hrr_dim=DIM)
    retriever = FactRetriever(new, hrr_dim=DIM)
    assert _contents(new._conn) == {"Alice Smith lives in London", "Alice Smith and Bob Brown run a coffee shop",
                                    "Bob Brown teaches yoga on Mondays", "Hermes Agent runs on the server",
                                    'The "hermes-agent" repo moved'}
    assert {f["fact_id"] for f in retriever.probe("hermes-agent", min_trust=0)} == {hermes, repo}  # merged on open
    assert new.read("SELECT COUNT(*) FROM entities WHERE name_key = 'hermes agent'")[0][0] == 1
    assert [f["content"] for f in retriever.search("Mondays", min_trust=0)] == ["Bob Brown teaches yoga on Mondays"]
    assert {f["content"] for f in retriever.probe("bob brown", min_trust=0)} == {
        "Alice Smith and Bob Brown run a coffee shop", "Bob Brown teaches yoga on Mondays"}
    assert tuple(new.read("SELECT retrieval_count, helpful_count FROM facts WHERE fact_id = ?", (a,))[0]) == (1, 1)
    new.remove_fact(b)
    _fts_ok(new._conn)
    new.close()

    legacy = old_store.MemoryStore(str(db), hrr_dim=DIM)
    assert len(legacy.list_facts()) == 4
    _fts_ok(legacy._conn)
    legacy.close()
