"""Tests for scalping_profiler.py — core profiling functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from scalping_profiler import (
    ScalpingProfiler,
    _safe_spearman,
    _DEFAULT_CONFIG,
    FeatureProfile,
    ConditionalIC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profiler(**overrides) -> ScalpingProfiler:
    cfg = dict(_DEFAULT_CONFIG)
    cfg.update(overrides)
    return ScalpingProfiler(cfg)


def _make_bars(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic bar DataFrame with price and a few feature columns."""
    rng = np.random.RandomState(seed)
    price = 100.0 + np.cumsum(rng.randn(n) * 0.01)
    return pd.DataFrame({
        "raw_midprice_close": price,
        "feat_alpha": rng.randn(n),
        "feat_noise": rng.randn(n),
        "feat_const": np.ones(n) * 5.0,
    })


# ---------------------------------------------------------------------------
# _safe_spearman
# ---------------------------------------------------------------------------

class TestSafeSpearman:
    def test_correlated_arrays(self):
        a = np.arange(100, dtype=float)
        b = a * 2 + 1
        assert _safe_spearman(a, b) == pytest.approx(1.0)

    def test_anticorrelated(self):
        a = np.arange(100, dtype=float)
        b = -a
        assert _safe_spearman(a, b) == pytest.approx(-1.0)

    def test_too_few_elements(self):
        assert _safe_spearman(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0

    def test_constant_array_returns_zero(self):
        a = np.ones(20)
        b = np.arange(20, dtype=float)
        assert _safe_spearman(a, b) == 0.0

    def test_random_near_zero(self):
        rng = np.random.RandomState(0)
        a = rng.randn(200)
        b = rng.randn(200)
        result = _safe_spearman(a, b)
        assert -1.0 <= result <= 1.0
        assert abs(result) < 0.3  # uncorrelated


# ---------------------------------------------------------------------------
# compute_forward_returns
# ---------------------------------------------------------------------------

class TestComputeForwardReturns:
    def test_shapes(self):
        profiler = _make_profiler(horizons=[1, 5])
        bars = _make_bars(200)
        fwd = profiler.compute_forward_returns(bars)

        assert set(fwd.keys()) == {1, 5}
        for h, arr in fwd.items():
            assert len(arr) == 200
            # Last h values must be NaN
            assert np.all(np.isnan(arr[-h:]))
            # Non-trailing values should be finite
            assert np.all(np.isfinite(arr[:-h]))

    def test_values_are_log_returns(self):
        profiler = _make_profiler(horizons=[1])
        prices = np.array([100.0, 110.0, 105.0, 120.0])
        bars = pd.DataFrame({"raw_midprice_close": prices})
        fwd = profiler.compute_forward_returns(bars)

        expected_0 = np.log(110.0 / 100.0)
        assert fwd[1][0] == pytest.approx(expected_0, rel=1e-10)
        assert np.isnan(fwd[1][-1])

    def test_multi_horizon(self):
        profiler = _make_profiler(horizons=[1, 2])
        prices = np.array([100.0, 102.0, 105.0, 110.0, 108.0])
        bars = pd.DataFrame({"raw_midprice_close": prices})
        fwd = profiler.compute_forward_returns(bars)

        # h=2: log(105/100) at t=0
        assert fwd[2][0] == pytest.approx(np.log(105.0 / 100.0), rel=1e-10)
        # Last 2 are NaN
        assert np.isnan(fwd[2][-1])
        assert np.isnan(fwd[2][-2])


# ---------------------------------------------------------------------------
# _rolling_ic_ir
# ---------------------------------------------------------------------------

class TestRollingIcIr:
    def test_returns_finite(self):
        profiler = _make_profiler(rolling_ic_window=20)
        rng = np.random.RandomState(1)
        values = rng.randn(200)
        fwd = rng.randn(200)
        ir = profiler._rolling_ic_ir(values, fwd)
        assert np.isfinite(ir)

    def test_too_few_bars_returns_zero(self):
        profiler = _make_profiler(rolling_ic_window=100)
        values = np.arange(20, dtype=float)
        fwd = np.arange(20, dtype=float)
        ir = profiler._rolling_ic_ir(values, fwd)
        assert ir == 0.0


# ---------------------------------------------------------------------------
# _quintile_analysis
# ---------------------------------------------------------------------------

class TestQuintileAnalysis:
    def test_monotonic_signal(self):
        profiler = _make_profiler()
        rng = np.random.RandomState(2)
        # Signal perfectly predicts forward return
        n = 500
        signal = np.linspace(-1, 1, n)
        fwd = signal + rng.randn(n) * 0.01
        spread, mono = profiler._quintile_analysis(signal, fwd)
        # Spread should be positive (Q5 > Q1)
        assert spread > 0
        assert mono is True

    def test_noise_signal_finite(self):
        profiler = _make_profiler()
        rng = np.random.RandomState(3)
        n = 500
        signal = rng.randn(n)
        fwd = rng.randn(n)
        spread, _ = profiler._quintile_analysis(signal, fwd)
        # Should produce a finite spread (may be large for random data)
        assert np.isfinite(spread)


# ---------------------------------------------------------------------------
# _autocorr_1
# ---------------------------------------------------------------------------

class TestAutocorr1:
    def test_constant_returns_zero(self):
        profiler = _make_profiler()
        vals = np.ones(100)
        assert profiler._autocorr_1(vals) == 0.0

    def test_alternating_signal(self):
        profiler = _make_profiler()
        vals = np.array([1.0, -1.0] * 100)
        ac = profiler._autocorr_1(vals)
        # Alternating → strong negative autocorrelation
        assert ac < -0.9

    def test_trending_signal(self):
        profiler = _make_profiler()
        vals = np.cumsum(np.ones(200))
        ac = profiler._autocorr_1(vals)
        assert ac > 0.9


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_directional(self):
        profiler = _make_profiler(min_ic=0.02, min_hit_rate=0.51)
        role = profiler._classify(
            best_ic=0.05, hit_rate=0.55, autocorr=0.3,
            conditional=[], nan_rate=0.01,
        )
        assert role == "directional"

    def test_noise(self):
        profiler = _make_profiler(min_ic=0.02, min_hit_rate=0.51)
        role = profiler._classify(
            best_ic=0.001, hit_rate=0.50, autocorr=0.0,
            conditional=[], nan_rate=0.5,
        )
        assert role == "noise"


# ---------------------------------------------------------------------------
# _compute_score
# ---------------------------------------------------------------------------

class TestComputeScore:
    def test_range_0_to_1(self):
        profiler = _make_profiler()
        score = profiler._compute_score(
            best_ic=0.05, ic_ir=1.5, hit_rate=0.55,
            q_spread=5.0, autocorr=0.8, net_edge=2.0,
            role="directional",
        )
        assert 0.0 <= score <= 1.0

    def test_noise_gets_low_score(self):
        profiler = _make_profiler()
        score = profiler._compute_score(
            best_ic=0.001, ic_ir=0.1, hit_rate=0.50,
            q_spread=0.5, autocorr=0.0, net_edge=-1.0,
            role="noise",
        )
        assert score < 0.3


# ---------------------------------------------------------------------------
# profile_feature (integration)
# ---------------------------------------------------------------------------

class TestProfileFeature:
    def test_returns_none_for_insufficient_obs(self):
        profiler = _make_profiler(min_observations=1000)
        bars = _make_bars(100)
        fwd = profiler.compute_forward_returns(bars)
        result = profiler.profile_feature(
            bars["feat_alpha"].values, fwd, "feat_alpha", "test", bars
        )
        assert result is None

    def test_returns_profile_for_valid_data(self):
        profiler = _make_profiler(min_observations=50, horizons=[1, 2])
        bars = _make_bars(200)
        fwd = profiler.compute_forward_returns(bars)
        result = profiler.profile_feature(
            bars["feat_alpha"].values, fwd, "feat_alpha", "test", bars
        )
        assert result is not None
        assert isinstance(result, FeatureProfile)
        assert result.name == "feat_alpha"
        assert 0.0 <= result.scalp_score <= 1.0
        assert result.role in ("directional", "gate", "regime", "noise")
        assert 1 in result.ic or 2 in result.ic

    def test_constant_feature_returns_none_or_noise(self):
        profiler = _make_profiler(min_observations=50, horizons=[1])
        bars = _make_bars(200)
        fwd = profiler.compute_forward_returns(bars)
        result = profiler.profile_feature(
            bars["feat_const"].values, fwd, "feat_const", "test", bars
        )
        # Constant feature: either None (all ICs zero → below threshold)
        # or noise role
        if result is not None:
            assert result.role == "noise"
