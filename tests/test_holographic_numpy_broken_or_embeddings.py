"""Two holes #64 (NousResearch/hermes-agent#145156) left, in a fresh process each (an import by another test would hide
one):

- with search by meaning on, the provider's constructor asked the Embedder whether it could run, which imported
  numpy: starting a session loaded it after all;
- numpy that is installed but will not load raised past the loader when the import failed with anything but
  ImportError: numpy's own import raises RuntimeError when its BLAS sanity check fails (a wrong OpenBLAS, mixed
  package managers), and ``os.add_dll_directory`` in its Windows wheel can raise OSError. A DLL that does not load
  comes out as ImportError, which was taken for "not installed" and never said. It is now treated as missing: nothing crashes, recall goes
  on by words, the log says it once and ``stats`` says why."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# argv: repo, db path, provider config (JSON), steps (JSON list). Prints one JSON line: where numpy was loaded.
SCRIPT = textwrap.dedent('''
    import importlib.util, json, logging, os, sys
    if os.environ.get("BLOCK_NUMPY"):
        sys.modules["numpy"] = None  # any "import numpy" raises ImportError from here on: not installed
    if os.environ.get("BROKEN_NUMPY"):
        sys.path.insert(0, os.environ["BROKEN_NUMPY"])  # a numpy that find_spec finds and that fails to import
    sys.path.insert(0, os.environ["HERMES_AGENT_REPO"])
    warnings = []
    class Catch(logging.Handler):
        def emit(self, record):
            if "numpy" in record.getMessage():
                warnings.append(record.getMessage())
    logging.getLogger().addHandler(Catch(logging.WARNING))
    spec = importlib.util.spec_from_file_location("hermes_plugin_holographic", os.path.join(sys.argv[1], "__init__.py"),
                                                  submodule_search_locations=[sys.argv[1]])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    out = {"loaded": sys.modules.get("numpy") is not None}
    config = {"db_path": sys.argv[2], **json.loads(sys.argv[3])}
    p = mod.HolographicMemoryProvider(config=config)
    for step in json.loads(sys.argv[4]):
        if step == "initialize":
            p.initialize(session_id="s")
        elif step == "block":
            out["block"] = p.system_prompt_block()
        elif step == "prefetch":
            out["prefetch"] = p.prefetch("what does Masha like for breakfast?")
        elif step == "add":
            out["add"] = json.loads(p.handle_tool_call("fact_store", {
                "action": "add", "content": "Masha likes oatmeal for breakfast.", "entities": ["Masha"]}))
        elif step == "related":
            out["related"] = json.loads(p.handle_tool_call("fact_store", {"action": "related", "entity": "Masha"}))
        elif step == "stats":
            out["stats"] = json.loads(p.handle_tool_call("fact_store", {"action": "stats"}))
        elif step == "second":  # another provider in the same process (a gateway runs several)
            q = mod.HolographicMemoryProvider(config=config)
            q.initialize(session_id="s2")
            out["second"] = q.prefetch("what does Masha like for breakfast?")
            q.shutdown()
        out[step + "_numpy"] = sys.modules.get("numpy") is not None
    p.shutdown()
    out["warnings"] = warnings
    print(json.dumps(out))
''')

ON_START = ["initialize", "block"]
EMBEDDINGS_ON = {"embeddings": "on", "embedding_url": "http://127.0.0.1:9/v1"}


def _run(tmp_path, config: dict, steps: list, **env_extra) -> dict:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    agent = os.environ.get("HERMES_AGENT_REPO") or str(Path.home() / ".hermes" / "hermes-agent")
    env = {k: v for k, v in os.environ.items() if k not in ("BLOCK_NUMPY", "BROKEN_NUMPY")}
    env.update(HERMES_HOME=str(home), HERMES_AGENT_REPO=agent, **env_extra)
    out = subprocess.run([sys.executable, "-c", SCRIPT, str(REPO), str(tmp_path / "memory_store.db"),
                          json.dumps(config), json.dumps(steps)],
                         env=env, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-3000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["loaded"] is False, "numpy was imported while the plugin loaded (issue #43)"
    return result


def _broken_numpy(tmp_path, error: str) -> str:
    """A numpy package that is installed (find_spec finds it) and raises on import, like a wheel whose DLL is gone."""
    site = tmp_path / "broken_site"
    (site / "numpy").mkdir(parents=True)
    (site / "numpy" / "__init__.py").write_text(f"raise {error}\n", encoding="utf-8")
    return str(site)


def test_starting_with_search_by_meaning_on_does_not_import_numpy(tmp_path):
    """The constructor asks whether search by meaning can run, initialize() opens the store; neither needs numpy,
    nor does a recall on a store that holds no vector of the model."""
    result = _run(tmp_path, EMBEDDINGS_ON, ON_START + ["prefetch"])
    assert result["initialize_numpy"] is False, "starting with search by meaning on imported numpy"
    assert result["block_numpy"] is False and result["prefetch_numpy"] is False
    assert "numpy is not installed" not in result["block"]


def test_without_numpy_the_prompt_and_stats_still_say_so(tmp_path):
    """Not installed: found without an import, and said exactly as before (keyword recall only)."""
    result = _run(tmp_path, {}, ON_START + ["add", "prefetch", "stats"], BLOCK_NUMPY="1")
    assert "- numpy is not installed: keyword recall only (no structural 'related')." in result["block"]
    assert result["stats"]["structural_recall"] == "off (numpy missing)"
    assert "oatmeal" in result["prefetch"]
    assert result["warnings"] == []  # not installed is a setup, not a fault: nothing in the log, as before


@pytest.mark.parametrize("error", ['ImportError("DLL load failed while importing _multiarray_umath: '
                                   'The specified module could not be found.")',
                                   # numpy's own import-time BLAS check (numpy/__init__.py, _sanity_check)
                                   'RuntimeError("The current Numpy installation fails to pass simple sanity checks. '
                                   'This can be caused for example by incorrect BLAS library being linked in")',
                                   'OSError(22, "[WinError 193] %1 is not a valid Win32 application")'],
                         ids=["ImportError", "RuntimeError", "OSError"])
def test_numpy_installed_but_broken_is_treated_as_missing(tmp_path, error):
    """find_spec says yes and the import fails: no crash anywhere; the fact is kept and recalled by its words,
    'related' falls back to keyword search, stats says why, and the log says it once (two providers, both modules)."""
    site = _broken_numpy(tmp_path, error)
    result = _run(tmp_path, EMBEDDINGS_ON, ON_START + ["add", "prefetch", "related", "stats", "second"],
                  BROKEN_NUMPY=site)
    assert result["add"].get("fact_id"), result["add"]
    assert "oatmeal" in result["prefetch"] and "oatmeal" in result["second"]
    assert any("oatmeal" in f["content"] for f in result["related"]["results"]), result["related"]
    assert result["stats"]["structural_recall"].startswith("off (numpy did not load: ")
    assert result["stats"]["search_by_meaning"]["status"].startswith("unavailable: numpy did not load")
    # the loader says it once; each provider with search by meaning on also says it cannot run, as without numpy
    loader = [w for w in result["warnings"] if w.startswith("holographic: numpy is installed but did not load (")]
    assert len(loader) == 1, result["warnings"]
