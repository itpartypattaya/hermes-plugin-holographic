"""On Windows, numpy loaded by the plugin runs its OpenBLAS on one thread (2026-11-10).

numpy's OpenBLAS commits a buffer of ~32 MB per CPU thread when the DLL loads. Measured with the venv of a local
Hermes on an 8-thread laptop (numpy 2.4.6, OpenBLAS 0.3.31): 242 MB of commit charge after ``import numpy``, 17 MB
with OPENBLAS_NUM_THREADS=1; the working set is the same. Hermes core does not import numpy, the plugin does on the
first recall, so every Hermes process that recalls paid ~225 MB. The plugin's math (a matrix-vector product, FFTs)
needs one thread. On Linux the same buffers cost no resident memory (hr-ovh: RSS 46.5 vs 46.6 MB), so nothing changes
there.

The requirements:
- on Windows, when the plugin is what loads numpy, its OpenBLAS runs one thread;
- the variable lives only for the import: Hermes runs the person's commands in child processes, which must not
  inherit a one-thread numpy from the plugin;
- a thread count the person set (any of the variables OpenBLAS reads) stays theirs;
- numpy that was loaded before the plugin is left as it is, and an import that fails leaves the environment as it was;
- elsewhere the plugin changes nothing.

Each case runs in a fresh process (an import by another test would hide it) and asks the OpenBLAS DLL that numpy
loaded how many threads it runs, next to a control process with the same environment that loads numpy without the
plugin.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

pytest.importorskip("numpy")

_VARS = ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_DEFAULT_NUM_THREADS")

# argv: repo, mode. mode "plugin": the plugin's module loads numpy as on a first recall (has_numpy, then encode);
# "control": plain "import numpy"; "numpy_first": numpy is imported before the plugin asks for it. Prints one JSON line.
CHILD = textwrap.dedent("""
    import ctypes, glob, importlib, importlib.util, json, os, subprocess, sys
    root, mode = sys.argv[1], sys.argv[2]
    if os.environ.get("BROKEN_NUMPY"):
        sys.path.insert(0, os.environ["BROKEN_NUMPY"])  # a numpy that find_spec finds and that fails to import
    VARS = %r
    seen_before = {v: os.environ.get(v) for v in VARS}
    if mode == "numpy_first":
        import numpy
    if mode in ("plugin", "numpy_first"):
        spec = importlib.util.spec_from_file_location("blas_probe", os.path.join(root, "__init__.py"),
                                                      submodule_search_locations=[root])
        sys.modules["blas_probe"] = importlib.util.module_from_spec(spec)
        hrr = importlib.import_module("blas_probe.holographic")
        loaded = hrr.has_numpy()
        if loaded:
            hrr.encode_atom("probe", 64)
    else:
        import numpy
        loaded = True
    out = {"loaded": loaded, "env_after": {v: os.environ.get(v) for v in VARS}, "env_before": seen_before}
    child = subprocess.run([sys.executable, "-c", "import os, json; print(json.dumps({v: os.environ.get(v) for v in %r}))"],
                           capture_output=True, text=True)
    out["child_env"] = json.loads(child.stdout)
    if loaded:
        import numpy
        out["blas"] = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"].get("name", "")
        libs = [p for p in glob.glob(os.path.join(os.path.dirname(numpy.__file__) + ".libs", "*openblas*"))
                if p.endswith((".dll", ".so", ".dylib")) or ".so." in p]
        out["threads"] = None
        if libs:
            lib = ctypes.CDLL(libs[0])
            get = getattr(lib, "scipy_openblas_get_num_threads64_", None) or getattr(lib, "openblas_get_num_threads")
            out["threads"] = get()
    print(json.dumps(out))
""") % (_VARS, _VARS)


def _run(mode: str, broken: Path | None = None, **env: str) -> dict:
    child_env = {k: v for k, v in os.environ.items() if k not in _VARS and k != "BROKEN_NUMPY"}
    child_env.update(env)
    if broken is not None:
        child_env["BROKEN_NUMPY"] = str(broken)
    out = subprocess.run([sys.executable, "-I", "-c", CHILD, str(REPO), mode], capture_output=True, text=True,
                         timeout=120, env=child_env)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def _measured(mode: str, **env: str) -> dict:
    found = _run(mode, **env)
    assert found["loaded"], "numpy did not load"
    if "openblas" not in found["blas"].lower():
        pytest.skip(f"this numpy is built against {found['blas'] or 'an unnamed BLAS'}, not OpenBLAS: nothing to limit")
    assert found["threads"] is not None, f"numpy says it uses {found['blas']} but its library is not in numpy.libs"
    return found


windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the one-thread load is for Windows only: on "
                                  "Linux the buffers cost no resident memory, so the plugin leaves numpy as it is")


@windows_only
def test_numpy_loaded_by_the_plugin_runs_openblas_on_one_thread():
    control = _measured("control")
    if control["threads"] == 1:
        pytest.skip("OpenBLAS already runs one thread here (one CPU): nothing to limit, the test would prove nothing")
    assert _measured("plugin")["threads"] == 1, f"OpenBLAS runs {control['threads']} threads as without the plugin"


@windows_only
def test_the_variable_does_not_outlive_the_import():
    found = _measured("plugin")
    assert found["env_after"]["OPENBLAS_NUM_THREADS"] is None
    assert found["child_env"]["OPENBLAS_NUM_THREADS"] is None  # a command Hermes runs gets the person's numpy


@pytest.mark.parametrize("var", _VARS)
def test_a_count_the_person_set_is_kept(var):
    found = _measured("plugin", **{var: "2"})
    assert found["env_after"] == {**dict.fromkeys(_VARS), var: "2"}  # not replaced, not removed
    assert found["threads"] == _measured("control", **{var: "2"})["threads"]  # what OpenBLAS makes of their setting


def test_numpy_loaded_before_the_plugin_is_left_as_it_is():
    found = _measured("numpy_first")
    assert found["env_after"] == dict.fromkeys(_VARS)
    assert found["threads"] == _measured("control")["threads"]


def test_a_failed_import_leaves_the_environment_as_it_was(tmp_path):
    pkg = tmp_path / "numpy"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise OSError('a wheel without its DLL')\n", encoding="utf-8")
    found = _run("plugin", broken=tmp_path)
    assert found["loaded"] is False
    assert found["env_after"] == dict.fromkeys(_VARS)
    assert found["child_env"] == dict.fromkeys(_VARS)


@pytest.mark.skipif(sys.platform == "win32", reason="checks the platforms where the plugin leaves numpy as it is")
def test_elsewhere_the_plugin_changes_nothing():
    found = _measured("plugin")
    assert found["env_after"] == dict.fromkeys(_VARS)
    assert found["threads"] == _measured("control")["threads"]
