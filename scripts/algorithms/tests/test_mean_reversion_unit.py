"""Unit tests for MeanReversionDetector algorithm."""

import numpy as np
import pytest

from algorithms.mean_reversion_detector import MeanReversionDetector


@pytest.fixture
def mrd():
    return MeanReversionDetector()


def _make_tick(**overrides) -> dict:
    """Create a neutral tick dict with all required columns."""
    tick = {
        "vol_returns_5m_last": 0.0015,
        "ent_tick_1m_mean": 0.50,
        "trend_hurst_300_mean": 0.50,
        "imbalance_qty_l1_mean": 0.0,
        "toxic_vpin_50_mean": 0.40,
        "raw_midprice_mean": 67500.0,
        "mf_ema_15m_last": 67400.0,
    }
    tick.update(overrides)
    return tick


def test_entropy_gate_inverted(mrd):
    """ent=0.50 (< 0.70) -> gate=0 (blocked). ent=0.80 -> gate=1 (active)."""
    r_low = mrd.step(_make_tick(ent_tick_1m_mean=0.50))
    assert r_low["alg_mr_entropy_gate"] == 0.0

    r_high = mrd.step(_make_tick(ent_tick_1m_mean=0.80))
    assert r_high["alg_mr_entropy_gate"] == 1.0


def test_zscore_computation(mrd):
    """midprice=67500, ema=67400, vol=0.0015 -> zscore ≈ +0.99."""
    r = mrd.step(_make_tick(
        raw_midprice_mean=67500.0,
        mf_ema_15m_last=67400.0,
        vol_returns_5m_last=0.0015,
    ))
    expected = (67500.0 - 67400.0) / (0.0015 * 67500.0)
    assert abs(r["alg_mr_zscore"] - expected) < 0.01


def test_signal_is_contrarian():
    """Positive zscore + high P(reversion) -> negative signal (short).

    Uses a mock model to test signal direction.
    """
    mrd = MeanReversionDetector()

    # Inject a mock model that always returns high reversion probability
    class MockModel:
        def predict(self, X):
            return np.array([0.85] * len(X))

    mrd._model = MockModel()

    # Positive zscore (price above EMA) + high entropy (gate active)
    r = mrd.step(_make_tick(
        raw_midprice_mean=68000.0,
        mf_ema_15m_last=67000.0,
        vol_returns_5m_last=0.002,
        ent_tick_1m_mean=0.80,
    ))

    # Contrarian: positive zscore -> negative signal
    assert r["alg_mr_signal"] < 0, f"Expected negative signal, got {r['alg_mr_signal']}"

    # Negative zscore (price below EMA) + high entropy
    r2 = mrd.step(_make_tick(
        raw_midprice_mean=66000.0,
        mf_ema_15m_last=67000.0,
        vol_returns_5m_last=0.002,
        ent_tick_1m_mean=0.80,
    ))

    # Contrarian: negative zscore -> positive signal
    assert r2["alg_mr_signal"] > 0, f"Expected positive signal, got {r2['alg_mr_signal']}"


def test_signal_range():
    """alg_mr_signal always in [-1, 1]."""
    mrd = MeanReversionDetector()

    class MockModel:
        def predict(self, X):
            return np.array([0.95] * len(X))

    mrd._model = MockModel()

    rng = np.random.default_rng(42)
    for _ in range(300):
        tick = _make_tick(
            vol_returns_5m_last=max(rng.normal(0.001, 0.002), 1e-6),
            ent_tick_1m_mean=rng.uniform(0, 1),
            raw_midprice_mean=67000 + rng.normal(0, 500),
            mf_ema_15m_last=67000 + rng.normal(0, 200),
        )
        r = mrd.step(tick)
        sig = r["alg_mr_signal"]
        if np.isfinite(sig):
            assert -1.0 <= sig <= 1.0, f"Signal out of range: {sig}"


def test_no_model_returns_neutral_for_signal():
    """No model: signal=0.0, probability=0.5 (neutral), zscore and gate computed.

    A trained model now ships in models/, so force the no-model path explicitly
    rather than relying on the (model-loading) `mrd` fixture."""
    mrd = MeanReversionDetector(model_path="/nonexistent/no_model_here")
    r = mrd.step(_make_tick(ent_tick_1m_mean=0.80))

    assert r["alg_mr_signal"] == 0.0
    assert r["alg_mr_probability"] == 0.5
    assert np.isfinite(r["alg_mr_zscore"])
    assert r["alg_mr_entropy_gate"] == 1.0


def test_nan_input_propagation(mrd):
    """NaN in required column -> all outputs NaN."""
    r = mrd.step(_make_tick(vol_returns_5m_last=float("nan")))
    assert np.isnan(r["alg_mr_signal"])
    assert np.isnan(r["alg_mr_probability"])
    assert np.isnan(r["alg_mr_zscore"])
    assert np.isnan(r["alg_mr_entropy_gate"])
