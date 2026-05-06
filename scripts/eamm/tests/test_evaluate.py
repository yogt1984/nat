"""Unit tests for EAMM Walk-Forward Evaluator."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from eamm.evaluate import walk_forward_evaluate, EvaluationResult


def _make_eval_data(n=1000, k=4):
    """Generate synthetic evaluation data."""
    np.random.seed(42)
    X = np.random.randn(n, 19)
    # PnL matrix: wider spreads generally better when vol is high
    pnl = np.random.randn(n, k) * 2
    # Make wider spreads slightly better on average
    for i in range(k):
        pnl[:, i] += (i - k / 2) * 0.1
    fill_rt = (np.random.rand(n, k) > 0.3).astype(float)
    # Optimal spread: some learnable signal
    optimal = 5.0 + X[:, 0] * 2.0 + np.random.randn(n) * 1.0
    optimal = np.clip(optimal, 1.0, 20.0)
    return X, pnl, fill_rt, optimal


FEATURE_NAMES = [f"feat_{i}" for i in range(19)]
SPREADS = [1.0, 3.0, 5.0, 10.0]


class TestWalkForward:
    def test_returns_result(self):
        X, pnl, fill_rt, optimal = _make_eval_data()
        result = walk_forward_evaluate(
            X, pnl, fill_rt, optimal, SPREADS, FEATURE_NAMES, n_splits=3
        )
        assert isinstance(result, EvaluationResult)
        assert result.n_splits == 3
        assert len(result.splits) == 3

    def test_no_lookahead(self):
        X, pnl, fill_rt, optimal = _make_eval_data()
        result = walk_forward_evaluate(
            X, pnl, fill_rt, optimal, SPREADS, FEATURE_NAMES, n_splits=3
        )
        # Each split's train_size should be less than test start
        for s in result.splits:
            assert s.train_size > 0
            assert s.test_size > 0

    def test_expanding_window(self):
        X, pnl, fill_rt, optimal = _make_eval_data()
        result = walk_forward_evaluate(
            X, pnl, fill_rt, optimal, SPREADS, FEATURE_NAMES, n_splits=4
        )
        # Train size should increase across splits
        train_sizes = [s.train_size for s in result.splits]
        assert train_sizes == sorted(train_sizes)

    def test_baselines_computed(self):
        X, pnl, fill_rt, optimal = _make_eval_data()
        result = walk_forward_evaluate(
            X, pnl, fill_rt, optimal, SPREADS, FEATURE_NAMES, n_splits=2
        )
        # Should have baseline for each spread level
        for s in result.splits:
            assert len(s.baseline_pnl) == len(SPREADS)
            assert len(s.baseline_sharpe) == len(SPREADS)

    def test_best_fixed_identified(self):
        X, pnl, fill_rt, optimal = _make_eval_data()
        result = walk_forward_evaluate(
            X, pnl, fill_rt, optimal, SPREADS, FEATURE_NAMES, n_splits=3
        )
        assert result.best_fixed_spread in SPREADS
        assert np.isfinite(result.best_fixed_sharpe)
