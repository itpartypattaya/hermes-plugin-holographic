"""What a module may import, checked by reading its source (no import runs).

entity_names.py holds the rules of what a name is; store, entities and the provider read them. It leans on text.py
and three standard modules only, so it never pulls in the store, the core or a network client, and it can be read
(or copied) on its own. A new import there is a decision: make it here, in ALLOWED.

.screen (review 2026-11-17, H2): a name someone gives must not be a credential or an instruction, which the store
would keep and show as a name; the screen decides that. screen.py imports standard modules only (it is vendored
alone by hermes-dreaming), so the rule above still holds."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# module -> what it may import: top-level standard modules by name, the plugin's own modules as ".name"
ALLOWED = {
    "entity_names.py": {"bisect", "re", "unicodedata", ".text", ".screen"},
}


def _imports(path: Path) -> set[str]:
    """Every import in the file, at any depth (one inside a function counts as well), and every call that imports
    by name (__import__("x"), importlib.import_module("x")): those are imports too, which no list can allow."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if not node.level:
                found.add((node.module or "").split(".")[0])
            elif node.module:
                found.add("." * node.level + node.module)
            else:  # from . import text
                found |= {"." * node.level + alias.name for alias in node.names}
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name in ("__import__", "import_module"):
                found.add(f"{name}()")
    return found


@pytest.mark.parametrize("module", sorted(ALLOWED))
def test_a_module_imports_only_what_it_may(module):
    extra = _imports(ROOT / module) - ALLOWED[module]
    assert not extra, f"{module} imports {sorted(extra)}; allowed: {sorted(ALLOWED[module])}"
