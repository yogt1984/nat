"""Tests for alpha.screener — IC computation, turnover, BH correction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from alpha.screener import (
    compute_forward_returns,
    compute_rolling_ic,
    compute_turnover,
    compute_breakeven_bps,
    benjamini_hochberg,
    FeatureAlpha,
    FORWARD_HORIZONS,
)


# ---------------------------------------------------------------------------
# compute_forward_returns
# ---------------------------------------------------------------------------


class TestComputeForwardReturns:
    def test_basic(self):
        prices = np.array([100, 102, 104, 106, 108], dtype=float)
        fwd = compute_forward_returns(prices, horizon=2)
        assert len(fwd) == 5
        assert abs(fwd[0] - 0.04) < 1e-10
        assert abs(fwd[1] - (106 / 102 - 1)) < 1e-10
        assert abs(fwd[2] - (108 / 104 - 1)) < 1e-10
        assert np.isnan(fwd[3])
        assert np.isnan(fwd[4])

    def test_horizon_exceeds_length(self):
        prices = np.array([100, 102, 104], dtype=float)
        fwd = compute_forward_returns(prices, horizon=5)
        assert np.all(np.isnan(fwd))

    def test_single_element(self):
        prices = np.array([100.0])
        fwd = compute_forward_returns(prices, horizon=1)
        assert len(fwd) == 1
        assert np.isnan(fwd[0])

    def test_horizon_one(self):
        prices = np.array([100, 110, 121], dtype=float)
        fwd = compute_forward_returns(prices, horizon=1)
        assert abs(fwd[0] - 0.1) < 1e-10
        assert abs(fwd[1] - 0.1) < 1e-10
        assert np.isnan(fwd[2])


# ---------------------------------------------------------------------------
# compute_rolling_ic
# ---------------------------------------------------------------------------


class TestComputeRollingIC:
    def test_perfect_positive_correlation(self):
        np.random.seed(42)
        n = 200
        feature = np.arange(n, dtype=float)
        fwd = feature * 0.01 + np.random.normal(0, 0.001, n)
        ic = compute_rolling_ic(feature, fwd, window=50, min_obs=20)
        valid = ic[~np.isnan(ic)]
        assert len(valid) > 0
        assert np.mean(valid) > 0.9

    def test_uncorrelated(self):
        np.random.seed(42)
        n = 500
        feature = np.random.randn(n)
        fwd = np.random.randn(n)
        ic = compute_rolling_ic(feature, fwd, window=100, min_obs=30)
        valid = ic[~np.isnan(ic)]
        if len(valid) > 0:
            assert abs(np.mean(valid)) < 0.3

    def test_constant_feature_returns_zero(self):
        feature = np.ones(200)
        fwd = np.random.randn(200)
        ic = compute_rolling_ic(feature, fwd, window=50)
        valid = ic[~np.isnan(ic)]
        for v in valid:
            assert abs(v) < 1e-10

    def test_nan_handling(self):
        n = 200
        feature = np.arange(n, dtype=float)
        fwd = np.arange(n, dtype=float)
        feature[10:20] = np.nan
        ic = compute_rolling_ic(feature, fwd, window=50)
        assert len(ic) > 0

    def test_insufficient_data(self):
        feature = np.arange(10, dtype=float)
        fwd = np.arange(10, dtype=float)
        ic = compute_rolling_ic(feature, fwd, window=50, min_obs=30)
        assert len(ic) == 0


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------


class TestComputeTurnover:
    def test_constant_signal(self):
        assert compute_turnover(np.ones(100)) == 0.0

    def test_positive_turnover(self):
        np.random.seed(42)
        signal = np.random.randn(1000)
        turnover = compute_turnover(signal)
        assert turnover > 0
        assert np.isfinite(turnover)

    def test_single_element(self):
        assert np.isnan(compute_turnover(np.array([1.0])))

    def test_all_nan(self):
        assert np.isnan(compute_turnover(np.full(10, np.nan)))

    def test_binary_signal(self):
        signal = np.array([1, -1, 1, -1, 1, -1] * 50, dtype=float)
        turnover = compute_turnover(signal)
        assert turnover > 1.0  # high turnover for alternating signal


# ---------------------------------------------------------------------------
# compute_breakeven_bps
# ---------------------------------------------------------------------------


class TestComputeBreakevenBps:
    def test_normal_case(self):
        be = compute_breakeven_bps(0.03, 0.01, 0.5)
        assert abs(be - 6.0) < 1e-10

    def test_zero_turnover(self):
        assert compute_breakeven_bps(0.03, 0.01, 0.0) == np.inf

    def test_nan_turnover(self):
        assert compute_breakeven_bps(0.03, 0.01, np.nan) == np.inf

    def test_negative_ic_uses_abs(self):
        be = compute_breakeven_bps(-0.05, 0.02, 1.0)
        assert abs(be - 10.0) < 1e-10


# ---------------------------------------------------------------------------
# benjamini_hochberg
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_all_significant(self):
        p = np.array([0.001, 0.002, 0.003])
        adj = benjamini_hochberg(p, alpha=0.05)
        assert len(adj) == 3
        assert all(a < 0.05 for a in adj)

    def test_none_significant(self):
        p = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        adj = benjamini_hochberg(p, alpha=0.05)
        assert all(a >= 0.05 for a in adj)

    def test_monotonicity_when_sorted(self):
        np.random.seed(42)
        p = np.random.uniform(0, 1, 20)
        adj = benjamini_hochberg(p, alpha=0.05)
        order = np.argsort(p)
        sorted_adj = adj[order]
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1] - 1e-10

    def test_empty_array(self):
        adj = benjamini_hochberg(np.array([]))
        assert len(adj) == 0

    def test_nan_handling(self):
        p = np.array([0.01, np.nan, 0.05])
        adj = benjamini_hochberg(p)
        assert not np.isnan(adj[0])
        assert np.isnan(adj[1])
        assert not np.isnan(adj[2])

    def test_clipped_to_one(self):
        p = np.array([0.9, 0.95, 0.99])
        adj = benjamini_hochberg(p)
        assert all(a <= 1.0 for a in adj)

    def test_single_value(self):
        p = np.array([0.03])
        adj = benjamini_hochberg(p, alpha=0.05)
        assert len(adj) == 1
        assert abs(adj[0] - 0.03) < 1e-10


# ---------------------------------------------------------------------------
# FORWARD_HORIZONS sanity
# ---------------------------------------------------------------------------


def test_forward_horizons_keys():
    assert "15min" in FORWARD_HORIZONS
    assert "1h" in FORWARD_HORIZONS
    for tf, horizons in FORWARD_HORIZONS.items():
        for name, bars in horizons.items():
            assert bars > 0
