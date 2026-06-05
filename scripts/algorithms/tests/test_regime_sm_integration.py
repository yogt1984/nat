"""Integration tests for RegimeStateMachine on bar DataFrames."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from algorithms.regime_state_machine import RegimeStateMachine
from algorithms.tests.conftest import make_bar_df


@pytest.fixture
def bars():
    return make_bar_df(n_bars=400)


@pytest.fixture
def rsm():
    return RegimeStateMachine()


def test_run_batch_state_distribution(rsm, bars):
    """No state occupancy > 95% on 400-bar synthetic data."""
    result = rsm.run_batch(bars)
    post_warmup = result.iloc[rsm.warmup:]
    regimes = post_warmup["alg_rsm_regime"].dropna()

    assert len(regimes) > 200

    # Check no single state dominates > 95%
    counts = regimes.value_counts(normalize=True)
    max_occupancy = counts.max()
    assert max_occupancy < 0.95, f"Single state occupies {max_occupancy:.1%}"


def test_step_batch_consistency(rsm, bars):
    """step() row-by-row and run_batch() produce same regime labels."""
    batch_result = rsm.run_batch(bars)

    rsm2 = RegimeStateMachine()
    cols = rsm2.required_columns()
    step_regimes = []
    for _, row in bars.iterrows():
        tick = {c: float(row[c]) for c in cols if c in bars.columns}
        r = rsm2.step(tick)
        step_regimes.append(r["alg_rsm_regime"])

    step_arr = np.array(step_regimes)
    batch_arr = batch_result["alg_rsm_regime"].values

    # Compare post-warmup
    warmup = rsm.warmup
    mask = np.isfinite(step_arr[warmup:]) & np.isfinite(batch_arr[warmup:])
    np.testing.assert_array_equal(
        step_arr[warmup:][mask], batch_arr[warmup:][mask],
        err_msg="Step-by-step and batch regime labels differ"
    )
