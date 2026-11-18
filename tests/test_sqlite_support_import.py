"""The ``sqlite_support`` import fallback (issue #114).

hermes-agent removed ``hermes_state_wal``; the helpers now live in the public
``sqlite_support`` API. These tests pin both directions of that compatibility import
and, above all, that a broken core is not silently downgraded to the legacy helpers:
falling back on an unrelated missing dependency would turn an installation problem
into a store that quietly runs different code.
"""
import sys
import types

import pytest

from hermes_plugin_holographic.store import MemoryStore


def _real_helpers():
    """The real helpers, taken from whichever module this core actually provides.

    The suite runs against more than one core layout, so the test must not assume
    ``sqlite_support`` exists: on an older core the same callables live in
    ``hermes_state_wal`` / ``hermes_cli.sqlite_util``.
    """
    try:
        from sqlite_support import add_column_if_missing, apply_wal_with_fallback

        return apply_wal_with_fallback, add_column_if_missing
    except ImportError:
        from hermes_cli.sqlite_util import add_column_if_missing
        from hermes_state_wal import apply_wal_with_fallback

        return apply_wal_with_fallback, add_column_if_missing


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _import_hook(monkeypatch, name, error, fail_from_call=1):
    """Make ``import name`` raise ``error`` from the Nth call onwards."""
    real_import = __import__
    calls = {"n": 0}

    def fake_import(module, *args, **kwargs):
        if module == name:
            calls["n"] += 1
            if calls["n"] >= fail_from_call:
                raise error
        return real_import(module, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    return calls


def test_store_opens_with_public_helpers(tmp_path):
    """The public helpers build a store, including the added ``hrr_vector`` column."""
    store = MemoryStore(db_path=str(tmp_path / "facts.db"))
    assert store.count_facts() == 0
    store.close()


def test_missing_module_falls_back_to_legacy(tmp_path, monkeypatch):
    """A core without ``sqlite_support`` still opens, and really calls the legacy helper."""
    real_wal, real_add = _real_helpers()
    calls = []

    def spy_wal(conn, **kwargs):
        calls.append(kwargs)
        return real_wal(conn, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "hermes_state_wal",
        _module("hermes_state_wal", apply_wal_with_fallback=spy_wal),
    )
    _import_hook(monkeypatch, "sqlite_support", ModuleNotFoundError("No module named 'sqlite_support'", name="sqlite_support"))

    store = MemoryStore(db_path=str(tmp_path / "facts.db"))
    assert store.count_facts() == 0
    store.close()
    # The legacy helper must actually be used, with the same arguments the public one gets:
    # recording only the call would also pass if the store never applied its result.
    assert calls and calls[0].get("db_label") == "memory_store.db (holographic)", calls


def test_missing_name_falls_back_to_legacy(tmp_path, monkeypatch):
    """A core whose ``sqlite_support`` predates the helpers (name absent) also falls back."""
    real_wal, real_add = _real_helpers()

    monkeypatch.setitem(sys.modules, "sqlite_support", _module("sqlite_support"))  # no attributes
    monkeypatch.setitem(
        sys.modules, "hermes_state_wal", _module("hermes_state_wal", apply_wal_with_fallback=real_wal)
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.sqlite_util",
        _module("hermes_cli.sqlite_util", add_column_if_missing=real_add),
    )

    store = MemoryStore(db_path=str(tmp_path / "facts.db"))
    assert store.count_facts() == 0
    store.close()


def test_broken_dependency_is_not_silently_downgraded(tmp_path, monkeypatch):
    """A missing dependency *inside* ``sqlite_support`` must raise, not fall back.

    Falling back here would report the unrelated "hermes_state_wal is missing" on a new
    core and hide the real installation problem.
    """
    _import_hook(monkeypatch, "sqlite_support", ModuleNotFoundError("No module named 'psutil'", name="psutil"))
    monkeypatch.setitem(
        sys.modules, "hermes_state_wal", _module("hermes_state_wal", apply_wal_with_fallback=lambda *a, **k: "wal")
    )

    with pytest.raises(ModuleNotFoundError) as excinfo:
        MemoryStore(db_path=str(tmp_path / "facts.db"))
    assert excinfo.value.name == "psutil"


def test_broken_dependency_on_add_column_is_not_downgraded(tmp_path, monkeypatch):
    """The second import (the column helper) is guarded too, on a pre-HRR database.

    Current ``_SCHEMA`` already carries ``hrr_vector``, so a fresh database never reaches
    this branch; the column has to be missing, as it is on a database written by an older
    version. The first import (WAL) succeeds, the second one fails.
    """
    import sqlite3

    db = tmp_path / "facts.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE facts ("
        "fact_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL UNIQUE,"
        "category TEXT DEFAULT 'general', tags TEXT DEFAULT '', trust_score REAL DEFAULT 0.5,"
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0,"
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )  # the pre-HRR shape: every current column except hrr_vector
    con.commit()
    con.close()

    real_wal, real_add = _real_helpers()
    _import_hook(
        monkeypatch,
        "sqlite_support",
        ModuleNotFoundError("No module named 'psutil'", name="psutil"),
        fail_from_call=2,
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.sqlite_util",
        _module("hermes_cli.sqlite_util", add_column_if_missing=lambda *a, **k: True),
    )

    with pytest.raises(ModuleNotFoundError) as excinfo:
        MemoryStore(db_path=str(db))
    assert excinfo.value.name == "psutil"


def test_circular_import_is_not_silently_downgraded(tmp_path, monkeypatch):
    """A circular import is a core bug, not an absent helper: it must not fall back."""
    _import_hook(
        monkeypatch,
        "sqlite_support",
        ImportError(
            "cannot import name 'apply_wal_with_fallback' from partially initialized module "
            "'sqlite_support' (most likely due to a circular import)",
            name="sqlite_support",
        ),
    )

    with pytest.raises(ImportError) as excinfo:
        MemoryStore(db_path=str(tmp_path / "facts.db"))
    assert "partially initialized" in str(excinfo.value)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ModuleNotFoundError("No module named 'sqlite_support'", name="sqlite_support"), True),
        (ModuleNotFoundError("No module named 'psutil'", name="psutil"), False),
        (
            ImportError("cannot import name 'apply_wal_with_fallback' from 'sqlite_support'", name="sqlite_support"),
            True,
        ),
        (
            ImportError(
                "cannot import name 'apply_wal_with_fallback' from partially initialized module "
                "'sqlite_support' (most likely due to a circular import)",
                name="sqlite_support",
            ),
            False,
        ),
        (ImportError("cannot import name 'other' from 'sqlite_support'", name="sqlite_support"), True),
        (ImportError("sqlite_support is broken"), False),
        # An error raised *inside* the module, carrying its name but not the "No module named"
        # text: that is a broken core, not an absent one, and must not fall back.
        (ModuleNotFoundError("inner_dep", name="sqlite_support"), False),
        (ModuleNotFoundError("No module named 'sqlite_support.inner'", name="sqlite_support.inner"), False),
    ],
)
def test_guard_admits_only_an_absent_helper(error, expected):
    """The guard itself, so both call sites stay covered even where a fresh DB cannot reach one.

    The two imports share this predicate: exercising it directly covers the column-helper
    branch on every core, not just on a database old enough to need the column added.
    """
    from hermes_plugin_holographic.store import _is_absent_core_helper

    assert _is_absent_core_helper(error) is expected


def test_guard_matches_real_interpreter_errors():
    """The guard against exceptions the interpreter itself raises, not only hand-built ones.

    Uses the same ``from module import name`` form as the store, since that is the shape
    whose ``name``/message the guard reads.
    """
    from hermes_plugin_holographic.store import _is_absent_core_helper

    # A real "module does not exist" failure, through from-import as the store does it.
    with pytest.raises(ImportError) as missing_module:
        from sqlite_support_definitely_absent import anything  # noqa: F401
    assert _is_absent_core_helper(missing_module.value) is False  # another module -> propagate

    # A real "name absent from an existing module" failure (transitional core shape).
    with pytest.raises(ImportError) as missing_name:
        from sqlite_support import this_helper_does_not_exist  # noqa: F401
    assert _is_absent_core_helper(missing_name.value) is True


def test_add_column_falls_back_on_a_pre_hrr_database(tmp_path, monkeypatch):
    """The column helper's own fallback, reached on a database that lacks ``hrr_vector``.

    A fresh database never reaches this branch (the current schema already has the column),
    so the WAL-only fallback test does not cover it.
    """
    import sqlite3

    db = tmp_path / "facts.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE facts ("
        "fact_id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL UNIQUE,"
        "category TEXT DEFAULT 'general', tags TEXT DEFAULT '', trust_score REAL DEFAULT 0.5,"
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0,"
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    con.commit()
    con.close()

    real_wal, real_add = _real_helpers()
    calls = []

    def spy_add(conn, table, column, ddl):
        calls.append((table, column, ddl))
        return real_add(conn, table, column, ddl)

    # This core has the WAL helper but not the column one (transitional layout).
    monkeypatch.setitem(
        sys.modules,
        "sqlite_support",
        _module("sqlite_support", apply_wal_with_fallback=real_wal),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.sqlite_util",
        _module("hermes_cli.sqlite_util", add_column_if_missing=spy_add),
    )

    store = MemoryStore(db_path=str(db))
    assert store.count_facts() == 0
    store.close()
    assert calls == [("facts", "hrr_vector", "hrr_vector BLOB")], calls
