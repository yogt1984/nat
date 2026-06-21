"""Guards the CLI-modularization oracle gap.

The `nat --json commands` snapshot pins the *parser tree* but cannot catch a
handler that references a module-level name that was dropped during extraction
(a runtime NameError, e.g. the _LIFECYCLE_STATES regression). This AST scan flags
any name Load-ed in a scripts/cli/*.py module that is neither imported, defined,
nor a builtin — catching orphaned constants/helpers before they ship.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

_CLI = Path(__file__).resolve().parents[1] / "cli"
_BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__all__"}


def _undefined_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    bound: set[str] = set()
    used: set[str] = set()
    star = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (bound if isinstance(n.ctx, ast.Store) else used).add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name == "*":
                    star = True
                else:
                    bound.add(a.asname or a.name)
        elif isinstance(n, ast.Global):
            bound.update(n.names)
    if star:  # `import *` makes module-level resolution opaque — skip
        return []
    return sorted(u for u in used - bound - _BUILTINS if not u.startswith("__"))


@pytest.mark.parametrize("path", sorted(_CLI.glob("*.py")), ids=lambda p: p.name)
def test_cli_module_has_no_undefined_names(path):
    undef = _undefined_names(path)
    assert not undef, f"{path.name} references undefined module-level name(s): {undef}"
