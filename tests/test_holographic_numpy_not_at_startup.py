"""Starting the provider must not import numpy (NousResearch/hermes-agent#145156).

#45 moved the module-level imports to first use, but the first use still came at start: opening the store asked
has_numpy() (which imports), the retriever asked it for its weights, and system_prompt_block() asked it for the
"numpy is not installed" line. On Windows a Kanban worker hangs on a late numpy load, so the provider's start
should not load it; the first search that scores by structure does. Each case runs in a fresh process, against a
store an earlier process wrote, the way a worker meets the profile's store. A store whose vectors need repair (made
without numpy, another hrr_dim) still loads numpy at open, to repair them: that is real work, not a check."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_LOAD = '''
import importlib.util, json, os, sys
{block}
sys.path.insert(0, os.environ["HERMES_AGENT_REPO"])
spec = importlib.util.spec_from_file_location("hermes_plugin_holographic", os.path.join(sys.argv[1], "__init__.py"),
                                              submodule_search_locations=[sys.argv[1]])
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
p = mod.HolographicMemoryProvider(config={{"db_path": sys.argv[2]{extra}}})
'''

WRITE = textwrap.dedent(_LOAD.format(block="", extra="") + '''
p.initialize(session_id="writer")
for text in ("Ksenia looks for a sales manager in Dubai.", "The nightly backup runs from the OES box at 2 am.",
             "Anton rides a Honda Forza around Jomtien."):
    p.handle_tool_call("fact_store", {"action": "add", "content": text, "origin": "said"})
p.shutdown()
print(json.dumps({"numpy": "numpy" in sys.modules}))
''')

START = textwrap.dedent(_LOAD.format(block="", extra="") + '''
p.initialize(session_id="worker")
block = p.system_prompt_block()
at_start = "numpy" in sys.modules
recalled = p.prefetch("who is looking for a sales manager in Dubai?")
after_search = "numpy" in sys.modules
weights = [p._retriever.fts_weight, p._retriever.jaccard_weight, p._retriever.hrr_weight]
print(json.dumps({"numpy_at_start": at_start, "numpy_after_search": after_search, "weights": weights,
                  "says_missing": "numpy is not installed" in block, "recalled": "sales manager" in recalled}))
p.shutdown()
''')

WITHOUT = textwrap.dedent(_LOAD.format(block='sys.modules["numpy"] = None  # any "import numpy" raises ImportError',
                                       extra="") + '''
p.initialize(session_id="worker")
block = p.system_prompt_block()
recalled = p.prefetch("who is looking for a sales manager in Dubai?")
print(json.dumps({"says_missing": "numpy is not installed" in block, "recalled": "sales manager" in recalled}))
p.shutdown()
''')

# numpy that Python can find but that fails to import (a broken install): find_spec says yes, the import says no.
_BROKEN = '''
import importlib.abc, importlib.machinery
class _Broken(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        return importlib.machinery.ModuleSpec(name, self) if name == "numpy" else None
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        raise ImportError("numpy is broken here")
sys.meta_path.insert(0, _Broken())
'''

RESIZE = textwrap.dedent(_LOAD.format(block=_BROKEN, extra=', "hrr_dim": 256') + '''
p.initialize(session_id="worker")
p.shutdown()
import sqlite3
con = sqlite3.connect(sys.argv[2])
dim = con.execute("SELECT value FROM meta WHERE key = 'hrr_dim'").fetchone()[0]
sizes = sorted({r[0] for r in con.execute("SELECT LENGTH(hrr_vector) FROM facts")})
con.close()
print(json.dumps({"dim": dim, "sizes": sizes}))
''')


def _run(script: str, db: Path, home: Path) -> dict:
    agent = os.environ.get("HERMES_AGENT_REPO") or str(Path.home() / ".hermes" / "hermes-agent")
    env = dict(os.environ, HERMES_HOME=str(home), HERMES_AGENT_REPO=agent)
    out = subprocess.run([sys.executable, "-c", script, str(REPO), str(db)], env=env, capture_output=True, text=True,
                         timeout=180)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture
def written_store(tmp_path):
    if importlib.util.find_spec("numpy") is None:  # pragma: no cover - the CI images install numpy
        pytest.skip("numpy is not installed here")
    home = tmp_path / "home"
    home.mkdir()
    db = tmp_path / "memory_store.db"
    assert _run(WRITE, db, home) == {"numpy": True}  # adding facts encodes their vectors: numpy is used there
    return db, home


def test_starting_on_an_existing_store_does_not_import_numpy(written_store):
    """initialize() and system_prompt_block() leave numpy unloaded; the first search loads it and works as before."""
    db, home = written_store
    state = _run(START, db, home)
    assert state["numpy_at_start"] is False, "numpy was imported while the provider started"
    assert state["numpy_after_search"] is True, "the first search did not load numpy for structural scoring"
    assert state["weights"] == [0.4, 0.3, 0.3]  # the configured blend, HRR share kept (not the no-numpy 0.6/0.4)
    assert state["says_missing"] is False  # numpy is installed: the prompt must not claim otherwise
    assert state["recalled"] is True


def test_without_numpy_the_prompt_still_says_so(written_store):
    """The no-import check must not hide a missing numpy: the line and keyword recall stay as they were."""
    db, home = written_store
    assert _run(WITHOUT, db, home) == {"says_missing": True, "recalled": True}


def test_a_broken_numpy_does_not_rewrite_the_vector_size(written_store):
    """numpy found but failing to import: a changed hrr_dim must leave the store as it was (the old size recorded,
    the vectors untouched) instead of recording the new size over vectors that were never re-encoded."""
    db, home = written_store
    state = _run(RESIZE, db, home)
    assert state["sizes"] == [4 + 4 * 1024], "the vectors changed"  # the writer's default hrr_dim, as written
    assert state["dim"] == "1024", "the new hrr_dim was recorded over vectors of the old size"
