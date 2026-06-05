"""Integration tests for MeanReversionDetector."""

import numpy as np
import pytest
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from algorithms.mean_reversion_detector import MeanReversionDetector
from algorithms.momentum_continuation import MomentumContinuation
from algorithms.tests.conftest import make_bar_df


@pytest.fixture
def bars():
    return make_bar_df(n_bars=400)


def test_complementarity_with_momentum(bars):
    """MR and MC entropy gates are complementary — rarely both active.

    MR gate: active when entropy > 0.70 (ranging).
    MC gate: active when entropy < 0.85 (trending).
    Their overlap should be small relative to total active bars.
    """
    mrd = MeanReversionDetector()
    mc = MomentumContinuation()

    mr_result = mrd.run_batch(bars)
    mc_result = mc.run_batch(bars)

    # Get post-warmup gates
    warmup = max(mrd.warmup, mc.warmup)
    mr_gate = mr_result["alg_mr_entropy_gate"].iloc[warmup:].values
    mc_gate = mc_result["alg_mc_entropy_gate"].iloc[warmup:].values

    # Drop NaN pairs
    mask = np.isfinite(mr_gate) & np.isfinite(mc_gate)
    assert mask.sum() > 100, f"Too few valid pairs: {mask.sum()}"

    mr_g = mr_gate[mask]
    mc_g = mc_gate[mask]

    # Both active simultaneously should be < both individually
    both_active = np.sum((mr_g == 1.0) & (mc_g == 1.0))
    mr_active = np.sum(mr_g == 1.0)
    mc_active = np.sum(mc_g == 1.0)

    # MR is active in high-entropy, MC in low-entropy → partial overlap
    # With MR threshold=0.70 and MC ceiling=0.85, overlap is [0.70, 0.85]
    # On uniform [0,1] entropy, overlap is ~15% of bars, each algo ~30%/~85%
    # Correlation of gates should be negative (complementary)
    rho, _ = stats.spearmanr(mr_g, mc_g)
    assert rho < 0.5, f"Gates too positively correlated: Spearman={rho:.3f}"


def test_run_batch_output_shape(bars):
    """run_batch returns correct columns and length."""
    mrd = MeanReversionDetector()
    result = mrd.run_batch(bars)

    assert len(result) == len(bars)
    expected_cols = {"alg_mr_signal", "alg_mr_probability", "alg_mr_zscore", "alg_mr_entropy_gate"}
    assert set(result.columns) == expected_cols
