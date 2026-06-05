"""Unit tests for ML constraint validation utilities."""

import math
import os
import pytest

from utils.ml_constraints import (
    check_bar_suffix,
    validate_algorithm_class,
    validate_model_path,
)
from algorithms.base import AlgorithmFeature, MicrostructureAlgorithm


# --- Stub classes for testing ---

class GoodAlgorithm(MicrostructureAlgorithm):
    """Valid algorithm with all-default __init__."""

    def name(self):
        return "good_test"

    def alg_features(self):
        return [AlgorithmFeature("alg_good_signal")]

    def required_columns(self):
        return ["col_a_mean"]

    def step(self, tick):
        return {"alg_good_signal": 0.5}

    def reset(self):
        pass


class NeedsArgAlgorithm(MicrostructureAlgorithm):
    """Algorithm that requires a positional arg — should fail."""

    def __init__(self, required_param):
        self.required_param = required_param

    def name(self):
        return "needs_arg"

    def alg_features(self):
        return [AlgorithmFeature("alg_needs")]

    def required_columns(self):
        return []

    def step(self, tick):
        return {"alg_needs": 0.0}

    def reset(self):
        pass


class BadPrefixAlgorithm(MicrostructureAlgorithm):
    """Algorithm with features that don't start with 'alg_'."""

    def name(self):
        return "bad_prefix"

    def alg_features(self):
        return [
            AlgorithmFeature("alg_ok_feature"),
            AlgorithmFeature("wrong_prefix"),
        ]

    def required_columns(self):
        return []

    def step(self, tick):
        return {"alg_ok_feature": 0.1, "wrong_prefix": 0.2}

    def reset(self):
        pass


class ExtraKeyAlgorithm(MicrostructureAlgorithm):
    """Algorithm whose step() returns extra keys."""

    def name(self):
        return "extra_key"

    def alg_features(self):
        return [AlgorithmFeature("alg_expected")]

    def required_columns(self):
        return []

    def step(self, tick):
        return {"alg_expected": 0.1, "alg_surprise": 0.2}

    def reset(self):
        pass


class BarLevelGoodAlgorithm(MicrostructureAlgorithm):
    """Bar-level algorithm with correct suffixes."""
    bar_level = True

    def name(self):
        return "bar_good"

    def alg_features(self):
        return [AlgorithmFeature("alg_bar_signal")]

    def required_columns(self):
        return ["ent_tick_1m_mean", "vol_returns_5m_last"]

    def step(self, tick):
        return {"alg_bar_signal": 0.0}

    def reset(self):
        pass


class BarLevelBadSuffixAlgorithm(MicrostructureAlgorithm):
    """Bar-level algorithm with a column missing suffix."""
    bar_level = True

    def name(self):
        return "bar_bad"

    def alg_features(self):
        return [AlgorithmFeature("alg_bar_signal")]

    def required_columns(self):
        return ["ent_tick_1m_mean", "ent_tick_1m"]

    def step(self, tick):
        return {"alg_bar_signal": 0.0}

    def reset(self):
        pass


# --- Tests ---

def test_no_arg_constructor_passes():
    """A class with all-default __init__ returns empty violations."""
    violations = validate_algorithm_class(GoodAlgorithm)
    assert violations == []


def test_no_arg_constructor_fails():
    """A class requiring a positional arg returns 'constructor' violation."""
    violations = validate_algorithm_class(NeedsArgAlgorithm)
    assert len(violations) == 1
    assert "constructor" in violations[0]


def test_feature_prefix_check():
    """Features not starting with 'alg_' are flagged."""
    violations = validate_algorithm_class(BadPrefixAlgorithm)
    prefix_violations = [v for v in violations if "feature_prefix" in v]
    assert len(prefix_violations) == 1
    assert "wrong_prefix" in prefix_violations[0]


def test_bar_suffix_valid():
    """'ent_tick_1m_mean' passes, 'ent_tick_1m' fails."""
    assert check_bar_suffix("ent_tick_1m_mean") is True
    assert check_bar_suffix("vol_returns_5m_last") is True
    assert check_bar_suffix("ent_tick_1m") is False
    assert check_bar_suffix("raw_price") is False


def test_bar_suffix_validation_on_bar_level():
    """Bar-level algorithm with missing suffix gets flagged."""
    good_violations = validate_algorithm_class(BarLevelGoodAlgorithm)
    assert good_violations == []

    bad_violations = validate_algorithm_class(BarLevelBadSuffixAlgorithm)
    suffix_violations = [v for v in bad_violations if "bar_suffix" in v]
    assert len(suffix_violations) == 1
    assert "ent_tick_1m" in suffix_violations[0]


def test_step_key_mismatch():
    """step() returning extra keys is flagged."""
    violations = validate_algorithm_class(ExtraKeyAlgorithm)
    key_violations = [v for v in violations if "step_keys" in v]
    assert len(key_violations) == 1
    assert "extra" in key_violations[0]


def test_validate_model_path_existing(tmp_path):
    """Existing writable directory returns True."""
    model_path = str(tmp_path / "model.pkl")
    assert validate_model_path(model_path) is True


def test_validate_model_path_missing():
    """Non-existent parent directory returns False."""
    assert validate_model_path("/nonexistent/path/model.pkl") is False
