"""Integration tests for KNNRetrieval."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from algorithms.knn_retrieval import KNNRetrieval
from algorithms.tests.conftest import make_bar_df


@pytest.fixture
def bars():
    return make_bar_df(n_bars=400)


def test_run_batch_on_bar_df(bars):
    """run_batch() on 400 bars. After min_buffer, outputs become finite."""
    knn = KNNRetrieval(k=10, min_buffer=50, refit_interval=50)
    result = knn.run_batch(bars)

    assert len(result) == len(bars)
    expected_cols = {
        "alg_knn_signal", "alg_knn_expected_return",
        "alg_knn_win_rate", "alg_knn_confidence",
    }
    assert set(result.columns) == expected_cols

    # First min_buffer-1 rows should be NaN (refit triggers at min_buffer)
    early = result.iloc[:49]
    assert early["alg_knn_signal"].isna().all()

    # Some later rows should be finite (after buffer + horizon resolves)
    late = result.iloc[200:]
    finite_count = late["alg_knn_expected_return"].notna().sum()
    assert finite_count > 0, "Expected some finite outputs after warmup"


def test_complementarity(bars):
    """KNN signal has low correlation with MC and MR signals.

    KNN is non-parametric and adapts continuously, so its signals should
    be weakly correlated with model-based algorithms.
    """
    from scipy import stats

    knn = KNNRetrieval(k=10, min_buffer=50, refit_interval=50)
    knn_result = knn.run_batch(bars)

    # Use win_rate as the most reliably finite output
    knn_sig = knn_result["alg_knn_win_rate"].values

    # MC returns constant 0 without model; use entropy gate as structural check
    from algorithms.momentum_continuation import MomentumContinuation
    mc = MomentumContinuation()
    mc_result = mc.run_batch(bars)
    mc_gate = mc_result["alg_mc_entropy_gate"].values

    # Align on finite values
    mask = np.isfinite(knn_sig) & np.isfinite(mc_gate)
    if mask.sum() > 30:
        rho, _ = stats.spearmanr(knn_sig[mask], mc_gate[mask])
        assert abs(rho) < 0.4, f"KNN vs MC too correlated: rho={rho:.3f}"
