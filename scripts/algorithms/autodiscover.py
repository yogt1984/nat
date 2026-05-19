"""
Auto-discovery of algorithm modules.

Scans scripts/algorithms/*.py and imports each module to trigger @register decorators.
Replaces manual import lists in evaluate.py and the nat CLI.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

_SKIP = {"base", "registry", "runner", "evaluate", "autodiscover", "__init__"}
_discovered = False


def discover_all() -> int:
    """Import all algorithm modules in this package. Returns count loaded."""
    global _discovered
    if _discovered:
        from .registry import list_algorithms
        return len(list_algorithms())

    pkg_dir = Path(__file__).parent
    count = 0

    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.name in _SKIP or info.ispkg:
            continue
        try:
            importlib.import_module(f".{info.name}", package="algorithms")
            count += 1
        except Exception:
            pass  # Skip broken modules silently

    _discovered = True
    return count
