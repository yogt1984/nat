"""Unit tests for MomentumContinuation algorithm."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from algorithms.momentum_continuation import MomentumContinuation


@pytest.fixture
def algo():
    return MomentumContinuation()


@pytest.fixture
def algo_with_model():
    """Create algorithm with an injected minimal LogisticRegression."""
    algo = MomentumContinuation()
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 7))
    y = (X[:, 0] > 0).astype(float)  # Simple rule based on first feature
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = LogisticRegression(C=1.0, max_iter=200)
    model.fit(X_s, y)
    algo._model = model
    algo._scaler = scaler
    return algo


def _make_tick(ent: float = 0.5, **overrides) -> dict:
    """Create a tick dict with all 7 required features."""
    tick = {
        "ent_tick_1m_mean": ent,
        "ent_permutation_returns_16_mean": 0.5,
        "trend_hurst_300_mean": 0.5,
        "toxic_vpin_50_mean": 0.3,
        "whale_net_flow_4h_sum": 100.0,
        "regime_accumulation_score_mean": 0.6,
        "vol_returns_5m_last": 0.001,
    }
    tick.update(overrides)
    return tick


def test_entropy_gate_blocks_high_entropy(algo_with_model):
    """tick with ent_tick_1m_mean=0.90 (> 0.85 ceiling) -> signal=0.0, gate=0.0"""
    r = algo_with_model.step(_make_tick(ent=0.90))
    assert r["alg_mc_entropy_gate"] == 0.0
    assert r["alg_mc_signal"] == 0.0


def test_entropy_gate_passes_low_entropy(algo_with_model):
    """tick with ent_tick_1m_mean=0.50 -> gate=1.0"""
    r = algo_with_model.step(_make_tick(ent=0.50))
    assert r["alg_mc_entropy_gate"] == 1.0


def test_dead_zone_zeroes_signal():
    """When p_short <= P(continuation) <= p_long, signal must be 0.0.

    Directly inject a model that produces exactly 0.5 on the tick features
    by using no scaler and forcing the model intercept/coefs to zero.
    """
    from sklearn.linear_model import LogisticRegression

    algo = MomentumContinuation(p_long=0.6, p_short=0.4)

    # Train a dummy model then zero out coefficients -> always predicts 0.5
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 7))
    y = rng.integers(0, 2, 50).astype(float)
    model = LogisticRegression(C=1.0, max_iter=200)
    model.fit(X, y)
    model.coef_[:] = 0.0
    model.intercept_[:] = 0.0

    algo._model = model
    algo._scaler = None  # No scaler -> raw features passed directly

    r = algo.step(_make_tick(ent=0.3))
    assert r["alg_mc_confidence"] == pytest.approx(0.5, abs=1e-6)
    assert r["alg_mc_signal"] == 0.0, "Signal must be 0 in dead zone"
    assert r["alg_mc_entropy_gate"] == 1.0


def test_no_model_returns_neutral(algo):
    """With no model loaded: signal=0.0, confidence=0.5, gate computed normally."""
    r = algo.step(_make_tick(ent=0.50))
    assert r["alg_mc_signal"] == 0.0
    assert r["alg_mc_confidence"] == 0.5
    assert r["alg_mc_entropy_gate"] == 1.0


def test_signal_range_bounded(algo_with_model):
    """Across 500 random ticks, alg_mc_signal stays in [-1, 1]."""
    rng = np.random.default_rng(42)
    for _ in range(500):
        tick = {
            "ent_tick_1m_mean": rng.uniform(0, 1),
            "ent_permutation_returns_16_mean": rng.uniform(0, 1),
            "trend_hurst_300_mean": rng.uniform(0, 1),
            "toxic_vpin_50_mean": rng.uniform(0, 1),
            "whale_net_flow_4h_sum": rng.normal(0, 1000),
            "regime_accumulation_score_mean": rng.uniform(0, 1),
            "vol_returns_5m_last": rng.normal(0, 0.01),
        }
        r = algo_with_model.step(tick)
        assert -1.0 <= r["alg_mc_signal"] <= 1.0, f"Signal out of range: {r['alg_mc_signal']}"


def test_confidence_range(algo_with_model):
    """alg_mc_confidence always in [0, 1] when model loaded."""
    rng = np.random.default_rng(42)
    for _ in range(200):
        tick = {
            "ent_tick_1m_mean": rng.uniform(0, 1),
            "ent_permutation_returns_16_mean": rng.uniform(0, 1),
            "trend_hurst_300_mean": rng.uniform(0, 1),
            "toxic_vpin_50_mean": rng.uniform(0, 1),
            "whale_net_flow_4h_sum": rng.normal(0, 1000),
            "regime_accumulation_score_mean": rng.uniform(0, 1),
            "vol_returns_5m_last": rng.normal(0, 0.01),
        }
        r = algo_with_model.step(tick)
        assert 0.0 <= r["alg_mc_confidence"] <= 1.0


def test_nan_input_propagation(algo_with_model):
    """If any FEATURE_COL is NaN, all outputs are NaN."""
    tick = _make_tick()
    tick["trend_hurst_300_mean"] = float("nan")
    r = algo_with_model.step(tick)
    assert np.isnan(r["alg_mc_signal"])
    assert np.isnan(r["alg_mc_confidence"])
    assert np.isnan(r["alg_mc_entropy_gate"])
