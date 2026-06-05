"""
ML algorithm constraint validation utilities.

Checks that algorithm classes conform to the MicrostructureAlgorithm contract
before they are deployed. Catches the common gotchas from ml_implementation_plan.txt
Section 0 (registry no-arg constructor, alg_ prefix, bar suffixes, step() keys).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Type

import numpy as np


VALID_BAR_SUFFIXES = (
    "_mean", "_std", "_last", "_sum", "_slope",
    "_close", "_open", "_high", "_low",
)


def check_bar_suffix(col: str) -> bool:
    """Return True if column name ends with a valid bar-aggregation suffix."""
    return any(col.endswith(s) for s in VALID_BAR_SUFFIXES)


def validate_model_path(model_path: str) -> bool:
    """Check model_path parent exists and is writable."""
    parent = Path(model_path).parent
    return parent.exists() and os.access(str(parent), os.W_OK)


def validate_algorithm_class(cls) -> list[str]:
    """Return list of constraint violations for a MicrostructureAlgorithm subclass.

    Checks:
    1. cls() callable with no arguments (simulates @register)
    2. All alg_features() names start with 'alg_'
    3. required_columns() returns list of strings
    4. step() returns dict with exactly the keys from alg_features()
    5. If bar_level is True, required_columns use aggregated suffixes
    """
    violations = []

    # 1. No-arg constructor
    try:
        instance = cls()
    except TypeError as e:
        violations.append(f"constructor: cls() failed — {e}")
        return violations  # Can't check further without an instance

    # 2. alg_features() names start with 'alg_'
    try:
        features = instance.alg_features()
        for f in features:
            if not f.name.startswith("alg_"):
                violations.append(f"feature_prefix: '{f.name}' does not start with 'alg_'")
    except Exception as e:
        violations.append(f"alg_features: raised {e}")
        return violations

    # 3. required_columns() returns list of strings
    try:
        cols = instance.required_columns()
        if not isinstance(cols, list):
            violations.append(f"required_columns: returned {type(cols).__name__}, expected list")
        else:
            for col in cols:
                if not isinstance(col, str):
                    violations.append(f"required_columns: item {col!r} is not a string")
    except Exception as e:
        violations.append(f"required_columns: raised {e}")
        return violations

    # 4. step() returns dict with exactly the keys from alg_features()
    expected_keys = {f.name for f in features}
    try:
        tick = {c: 0.5 for c in cols}
        result = instance.step(tick)
        if not isinstance(result, dict):
            violations.append(f"step: returned {type(result).__name__}, expected dict")
        else:
            result_keys = set(result.keys())
            missing = expected_keys - result_keys
            extra = result_keys - expected_keys
            if missing:
                violations.append(f"step_keys: missing {missing}")
            if extra:
                violations.append(f"step_keys: extra {extra}")
    except Exception as e:
        violations.append(f"step: raised {e}")

    # 5. If bar_level, required_columns must use aggregated suffixes
    if getattr(instance, "bar_level", False):
        for col in cols:
            if not check_bar_suffix(col):
                violations.append(f"bar_suffix: '{col}' missing bar-aggregation suffix")

    return violations
