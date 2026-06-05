"""Integration test: validate all registered algorithms against ML constraints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.autodiscover import discover_all
from algorithms.registry import _REGISTRY
from utils.ml_constraints import validate_algorithm_class


def test_all_registered_algorithms_pass_constraints():
    """Import all algorithms, run validate_algorithm_class on each.
    Assert zero violations for every registered algorithm."""
    discover_all()
    assert len(_REGISTRY) > 0, "No algorithms discovered"

    failures = {}
    for name in sorted(_REGISTRY):
        cls = _REGISTRY[name]
        violations = validate_algorithm_class(cls)
        if violations:
            failures[name] = violations

    if failures:
        lines = []
        for name, viols in failures.items():
            lines.append(f"  {name}:")
            for v in viols:
                lines.append(f"    - {v}")
        msg = f"{len(failures)} algorithm(s) failed constraint validation:\n" + "\n".join(lines)
        raise AssertionError(msg)
