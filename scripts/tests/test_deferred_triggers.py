"""Tests for check_deferred_triggers.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_deferred_triggers import (
    DATA_DAYS_THRESHOLD,
    DEPLOYED_ML_THRESHOLD,
    SHARPE_DEGRADATION_THRESHOLD,
    check_data_volume,
    check_deployed_ml_count,
    check_sharpe_degradation,
    evaluate_triggers,
)


def test_data_trigger_not_met(tmp_path):
    """Below threshold date dirs → not triggered."""
    # Create 10 date directories (well below 60)
    for i in range(10):
        (tmp_path / f"2026-01-{i+1:02d}").mkdir()

    triggered, n_days = check_data_volume(str(tmp_path))
    assert not triggered
    assert n_days == 10


def test_data_trigger_met(tmp_path):
    """At or above threshold date dirs → triggered."""
    for i in range(DATA_DAYS_THRESHOLD):
        month = (i // 28) + 1
        day = (i % 28) + 1
        (tmp_path / f"2026-{month:02d}-{day:02d}").mkdir()

    triggered, n_days = check_data_volume(str(tmp_path))
    assert triggered
    assert n_days == DATA_DAYS_THRESHOLD


def test_deployed_count_trigger():
    """Trigger fires when enough ML algos are deployed."""
    # Below threshold
    few = ["change_point_detector", "momentum_continuation"]
    triggered, count = check_deployed_ml_count(few)
    assert not triggered
    assert count == 2

    # At threshold
    enough = [
        "change_point_detector",
        "momentum_continuation",
        "regime_state_machine",
        "mean_reversion_detector",
    ]
    triggered, count = check_deployed_ml_count(enough)
    assert triggered
    assert count == DEPLOYED_ML_THRESHOLD


def test_sharpe_degradation_trigger():
    """Trigger fires when any algo's Sharpe drops > 30% from peak."""
    peak = {"algo_a": 2.0, "algo_b": 1.5}
    # algo_a: 2.0 → 1.0 = 50% drop (> 30%)
    current = {"algo_a": 1.0, "algo_b": 1.4}

    triggered, degraded = check_sharpe_degradation(peak, current)
    assert triggered
    assert "algo_a" in degraded
    assert "algo_b" not in degraded


def test_sharpe_stable():
    """No trigger when Sharpe within tolerance."""
    peak = {"algo_a": 2.0, "algo_b": 1.5}
    # algo_a: 2.0 → 1.8 = 10% drop (< 30%)
    current = {"algo_a": 1.8, "algo_b": 1.5}

    triggered, degraded = check_sharpe_degradation(peak, current)
    assert not triggered
    assert degraded == []
