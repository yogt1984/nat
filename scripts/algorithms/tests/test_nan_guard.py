"""Tests for NaN availability guard in AlgorithmRunner.

Verifies that run_on_dataframe() emits warnings when required columns
are >95% NaN, without blocking execution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from algorithms.base import AlgorithmFeature, MicrostructureAlgorithm  # noqa: E402
from algorithms.runner import AlgorithmRunner  # noqa: E402


# --- Stub algorithm ---

class StubAlgorithm(MicrostructureAlgorithm):
    """Minimal algorithm for testing runner behavior."""

    def name(self) -> str:
        return "stub_algo"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [AlgorithmFeature(name="alg_stub_out", warmup=0)]

    def required_columns(self) -> list[str]:
        return ["col_a", "col_b"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        return {"alg_stub_out": tick.get("col_a", 0.0) + tick.get("col_b", 0.0)}

    def reset(self) -> None:
        pass


def _make_df(n: int = 200, col_a_nan_rate: float = 0.0,
             col_b_nan_rate: float = 0.0) -> pd.DataFrame:
    """Create a DataFrame with controllable NaN rates per column."""
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)

    if col_a_nan_rate > 0:
        mask = rng.random(n) < col_a_nan_rate
        a[mask] = np.nan
    if col_b_nan_rate > 0:
        mask = rng.random(n) < col_b_nan_rate
        b[mask] = np.nan

    return pd.DataFrame({"col_a": a, "col_b": b})


# --- Tests ---

def test_no_warning_when_data_clean(caplog):
    """No warning when all columns have valid data."""
    runner = AlgorithmRunner(StubAlgorithm())
    df = _make_df(200, col_a_nan_rate=0.0, col_b_nan_rate=0.0)

    with caplog.at_level(logging.WARNING, logger="algorithms.runner"):
        result = runner.run_on_dataframe(df)

    assert result.n_ticks == 200
    assert not any("NaN" in r.message for r in caplog.records)


def test_no_warning_at_moderate_nan(caplog):
    """No warning when NaN rate is moderate (50%)."""
    runner = AlgorithmRunner(StubAlgorithm())
    df = _make_df(200, col_a_nan_rate=0.5)

    with caplog.at_level(logging.WARNING, logger="algorithms.runner"):
        runner.run_on_dataframe(df)

    assert not any("NaN" in r.message for r in caplog.records)


def test_no_warning_at_95pct_nan(caplog):
    """No warning when NaN rate is exactly at threshold (95%)."""
    runner = AlgorithmRunner(StubAlgorithm())
    n = 1000
    df = pd.DataFrame({
        "col_a": [np.nan] * 950 + [1.0] * 50,
        "col_b": np.random.default_rng(42).normal(0, 1, n),
    })

    with caplog.at_level(logging.WARNING, logger="algorithms.runner"):
        runner.run_on_dataframe(df)

    assert not any("NaN" in r.message for r in caplog.records)


def test_warning_above_95pct_nan(caplog):
    """Warning emitted when a column exceeds 95% NaN."""
    runner = AlgorithmRunner(StubAlgorithm())
    n = 1000
    df = pd.DataFrame({
        "col_a": [np.nan] * 960 + [1.0] * 40,  # 96% NaN
        "col_b": np.random.default_rng(42).normal(0, 1, n),
    })

    with caplog.at_level(logging.WARNING, logger="algorithms.runner"):
        runner.run_on_dataframe(df)

    warnings = [r for r in caplog.records if "NaN" in r.message]
    assert len(warnings) == 1
    assert "col_a" in warnings[0].message
    assert "stub_algo" in warnings[0].message


def test_warning_100pct_nan(caplog):
    """Warning emitted when a column is entirely NaN."""
    runner = AlgorithmRunner(StubAlgorithm())
    n = 200
    df = pd.DataFrame({
        "col_a": [np.nan] * n,
        "col_b": [np.nan] * n,
    })

    with caplog.at_level(logging.WARNING, logger="algorithms.runner"):
        runner.run_on_dataframe(df)

    warnings = [r for r in caplog.records if "NaN" in r.message]
    assert len(warnings) == 2  # one per column


def test_warning_does_not_block_execution(caplog):
    """Execution proceeds and returns valid result even with high NaN."""
    runner = AlgorithmRunner(StubAlgorithm())
    n = 200
    df = pd.DataFrame({
        "col_a": [np.nan] * n,
        "col_b": np.random.default_rng(42).normal(0, 1, n),
    })

    with caplog.at_level(logging.WARNING, logger="algorithms.runner"):
        result = runner.run_on_dataframe(df)

    assert result.n_ticks == n
    assert result.algorithm_name == "stub_algo"
    assert "alg_stub_out" in result.features_df.columns


def test_multiple_bad_columns_emit_separate_warnings(caplog):
    """Each column above threshold gets its own warning."""
    runner = AlgorithmRunner(StubAlgorithm())
    n = 1000
    df = pd.DataFrame({
        "col_a": [np.nan] * 970 + [1.0] * 30,  # 97% NaN
        "col_b": [np.nan] * 980 + [1.0] * 20,  # 98% NaN
    })

    with caplog.at_level(logging.WARNING, logger="algorithms.runner"):
        runner.run_on_dataframe(df)

    warnings = [r for r in caplog.records if "NaN" in r.message]
    assert len(warnings) == 2
    warned_cols = {w.message.split("'")[1] for w in warnings}
    assert warned_cols == {"col_a", "col_b"}
