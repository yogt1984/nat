"""Tests for ml_health_check.py."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_health_check import (
    CRITICAL,
    OK,
    WARN,
    check_model_age,
    check_nan_rate,
    check_sharpe_7d,
    evaluate_health,
)


NOW = datetime(2026, 6, 5, tzinfo=timezone.utc)


def test_model_age_ok():
    """training_date 3 days ago -> OK."""
    status, age = check_model_age("2026-06-02T00:00:00", now=NOW)
    assert status == OK
    assert age == 3


def test_model_age_warn():
    """training_date 20 days ago -> WARN."""
    status, age = check_model_age("2026-05-16T00:00:00", now=NOW)
    assert status == WARN
    assert age == 20


def test_model_age_critical():
    """training_date 35 days ago -> CRITICAL."""
    status, age = check_model_age("2026-05-01T00:00:00", now=NOW)
    assert status == CRITICAL
    assert age == 35


def test_nan_rate_warn():
    """25% NaN signals -> WARN."""
    assert check_nan_rate(0.25) == WARN


def test_nan_rate_ok():
    """5% NaN signals -> OK."""
    assert check_nan_rate(0.05) == OK


def test_sharpe_critical():
    """7d Sharpe = -0.7 -> CRITICAL."""
    assert check_sharpe_7d(-0.7) == CRITICAL


def test_all_ok(tmp_path):
    """All metrics healthy -> overall OK."""
    result = evaluate_health(
        "test_algo",
        models_dir=tmp_path,
        training_date="2026-06-03T00:00:00",
        nan_fraction=0.05,
        sharpe_7d=0.8,
        now=NOW,
    )
    assert result["overall"] == OK
    assert result["age_days"] == 2
    assert result["nan_status"] == OK
    assert result["sharpe_status"] == OK
