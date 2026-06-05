"""Integration tests for ChangePointDetector on bar DataFrames."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from algorithms.change_point_detector import ChangePointDetector
from algorithms.tests.conftest import make_bar_df


@pytest.fixture
def bars():
    return make_bar_df(n_bars=400)


@pytest.fixture
def cpd():
    return ChangePointDetector()


def test_run_batch_on_bar_df(cpd, bars):
    """Run change_point_detector.run_batch() on synthetic bar DataFrame.
    Output has 4 columns. After warmup (100 bars), >50% of rows finite."""
    result = cpd.run_batch(bars)

    # Correct number of columns
    assert result.shape[1] == 4
    expected_cols = {"alg_cpd_cusum_signal", "alg_cpd_run_length",
                     "alg_cpd_change_prob", "alg_cpd_regime_age"}
    assert set(result.columns) == expected_cols

    # Same length as input
    assert len(result) == len(bars)

    # After warmup, most values should be finite
    post_warmup = result.iloc[100:]
    for col in expected_cols:
        finite_rate = post_warmup[col].notna().mean()
        assert finite_rate > 0.5, f"{col}: only {finite_rate:.1%} finite after warmup"


def test_step_batch_consistency(cpd, bars):
    """step() iterated row-by-row produces same results as run_batch()
    for alg_cpd_cusum_signal (correlation > 0.95)."""
    # run_batch path
    batch_result = cpd.run_batch(bars)

    # Manual step path
    cpd2 = ChangePointDetector()
    step_signals = []
    cols = cpd2.required_columns()
    for _, row in bars.iterrows():
        tick = {c: float(row[c]) for c in cols if c in bars.columns}
        r = cpd2.step(tick)
        step_signals.append(r["alg_cpd_cusum_signal"])

    step_arr = np.array(step_signals)
    batch_arr = batch_result["alg_cpd_cusum_signal"].values

    # Compare post-warmup values
    mask = np.isfinite(step_arr) & np.isfinite(batch_arr)
    assert mask.sum() > 100, "Not enough finite values to compare"

    corr = np.corrcoef(step_arr[mask], batch_arr[mask])[0, 1]
    assert corr > 0.95, f"Step vs batch correlation: {corr:.3f}"
