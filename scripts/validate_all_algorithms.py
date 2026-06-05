#!/usr/bin/env python3
"""
Validate all registered algorithms against ML constraints.

Discovers every algorithm via autodiscover, runs validate_algorithm_class()
on each, and prints PASS/FAIL per algorithm. Exits 1 if any fail.

Usage:
    python scripts/validate_all_algorithms.py
"""

import sys
from pathlib import Path

# Ensure scripts/ is on path
sys.path.insert(0, str(Path(__file__).parent))

from algorithms.autodiscover import discover_all
from algorithms.registry import _REGISTRY
from utils.ml_constraints import validate_algorithm_class


def main():
    count = discover_all()
    print(f"Discovered {count} algorithm modules\n")

    failures = 0
    for name in sorted(_REGISTRY):
        cls = _REGISTRY[name]
        violations = validate_algorithm_class(cls)
        if violations:
            print(f"  FAIL  {name}")
            for v in violations:
                print(f"        - {v}")
            failures += 1
        else:
            print(f"  PASS  {name}")

    print(f"\n{len(_REGISTRY)} algorithms checked, {failures} failed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
