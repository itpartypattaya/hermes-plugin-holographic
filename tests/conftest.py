"""Test bootstrap.

The plugin imports Hermes core (``agent``, ``tools``, ``hermes_cli``) from a checkout: set
``HERMES_AGENT_REPO`` to your hermes-agent clone (default ``~/.hermes/hermes-agent``), or
``pip install -e`` it. This repo is loaded as the package ``hermes_plugin_holographic`` — the same tree
``hermes plugins install`` copies to ``$HERMES_HOME/plugins/holographic/``.
"""
import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

import pytest

HERMES_AGENT_REPO = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(HERMES_AGENT_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT_REPO))

PACKAGE = "hermes_plugin_holographic"
if PACKAGE not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        PACKAGE, REPO_ROOT / "__init__.py", submodule_search_locations=[str(REPO_ROOT)]
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[PACKAGE] = _module
    _spec.loader.exec_module(_module)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    """Every test gets its own HERMES_HOME; nothing touches the real ~/.hermes."""
    home = tmp_path / "hermes_test"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home



@pytest.fixture
def old_release_hermes(monkeypatch):
    """The Hermes an older release of this plugin ran on. Releases before 0.30.0 import hermes_state_wal, which Hermes main
    removed on 2026-11-17 (be3d6affc) for sqlite_support.apply_wal_with_fallback, the same function; the current
    store imports the new name first. A test that runs an old release on the same database gives it the module under
    its old name, for that test only, as the Hermes of its day had it."""
    try:
        import hermes_state_wal  # noqa: F401 - a Hermes release: nothing to give
    except ImportError:
        import sqlite_support
        monkeypatch.setitem(sys.modules, "hermes_state_wal", sqlite_support)

# PRAGMA synchronous=OFF on every connection the suite opens: no test checks what survives a power cut. CI's Python
# links SQLite 3.50.4, which core holds to the WAL-reset bug, so every store runs in DELETE journal mode and each
# commit waits on two fsyncs; on the Windows runner's disk that made the suite 28 minutes against 2 on Linux. The
# journal mode, locking and every write stay as they are; only the wait for the disk goes. On macOS core sets
# synchronous=FULL after WAL turns on, and that stands. Patched at import, not in a fixture, so that stores the module
# fixtures open get it too.
_real_connect = sqlite3.connect


def _connect_without_fsync(*args, **kwargs):
    conn = _real_connect(*args, **kwargs)
    try:
        conn.execute("PRAGMA synchronous=OFF")
    except sqlite3.Error:
        pass  # a file a test made unreadable on purpose fails where that test expects it to, not here
    return conn


sqlite3.connect = _connect_without_fsync


def _symlinks_supported(tmp_dir: Path) -> bool:
    target, link = tmp_dir / "target", tmp_dir / "link"
    target.write_text("x", encoding="utf-8")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        return False
    return True


def pytest_configure(config):
    config.addinivalue_line("markers", "require_symlinks: skip when symbolic links cannot be created here")


def pytest_runtest_setup(item):
    if item.get_closest_marker("require_symlinks"):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            if not _symlinks_supported(Path(tmp)):
                pytest.skip("symbolic links are not supported here (Windows needs developer mode)")
