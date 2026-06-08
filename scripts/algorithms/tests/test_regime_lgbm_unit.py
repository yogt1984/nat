"""Unit tests for RegimeConditionedLGBM algorithm."""

import numpy as np
import pytest

from algorithms.regime_conditioned_lgbm import (
    RegimeConditionedLGBM,
    TRENDING_FEATURES, RANGING_FEATURES, VOLATILE_FEATURES, GLOBAL_FEATURES,
    REGIME_TO_GROUP, GROUP_FEATURES,
)


@pytest.fixture
def rlgbm():
    return RegimeConditionedLGBM()


def _make_tick(**overrides) -> dict:
    """Create a neutral tick dict with all required columns."""
    tick = {
        "trend_hurst_300_mean": 0.50,
        "whale_net_flow_4h_sum": 0.0,
        "regime_accumulation_score_mean": 0.50,
        "trend_momentum_300_mean": 0.0,
        "ent_tick_1m_mean": 0.50,
        "vol_returns_5m_last": 0.001,
        "imbalance_qty_l1_mean": 0.0,
        "mf_bb_pctb_5m_last": 0.50,
        "toxic_vpin_50_mean": 0.40,
        "alg_rsm_regime_last": 2.0,      # TRENDING_UP
        "alg_rsm_confidence_last": 0.75,  # above threshold
    }
    tick.update(overrides)
    return tick


class MockBooster:
    """Mock LightGBM Booster that returns a configurable prediction."""
    def __init__(self, return_val=0.001):
        self._val = return_val

    def predict(self, X):
        return np.full(len(X), self._val)


def _inject_models(rlgbm, trending_val=0.002, ranging_val=-0.001,
                    volatile_val=0.0, global_val=0.0005):
    """Inject mock models into the algorithm."""
    rlgbm._models = {
        "trending": MockBooster(trending_val),
        "ranging": MockBooster(ranging_val),
        "volatile": MockBooster(volatile_val),
        "global": MockBooster(global_val),
    }


def test_regime_dispatch():
    """Regime=2 (TRENDING_UP) uses trending model, regime=4 uses volatile model."""
    rlgbm = RegimeConditionedLGBM()
    _inject_models(rlgbm, trending_val=0.005, volatile_val=-0.003)

    # Regime 2 = TRENDING_UP -> trending model (positive prediction)
    r_trend = rlgbm.step(_make_tick(alg_rsm_regime_last=2.0, alg_rsm_confidence_last=0.8))
    assert r_trend["alg_rlgbm_regime_used"] == 2.0
    assert r_trend["alg_rlgbm_predicted_return"] == pytest.approx(0.005)

    # Regime 4 = RANGING -> volatile model (negative prediction)
    r_vol = rlgbm.step(_make_tick(alg_rsm_regime_last=4.0, alg_rsm_confidence_last=0.8))
    assert r_vol["alg_rlgbm_regime_used"] == 4.0
    assert r_vol["alg_rlgbm_predicted_return"] == pytest.approx(-0.003)


def test_global_fallback():
    """Low confidence (0.3) routes to global model."""
    rlgbm = RegimeConditionedLGBM(confidence_threshold=0.60)
    _inject_models(rlgbm, trending_val=0.01, global_val=0.0001)

    r = rlgbm.step(_make_tick(
        alg_rsm_regime_last=2.0,
        alg_rsm_confidence_last=0.3,  # below threshold
    ))
    assert r["alg_rlgbm_regime_used"] == 5.0  # global
    assert r["alg_rlgbm_predicted_return"] == pytest.approx(0.0001)


def test_missing_regime_label():
    """NaN regime label uses global model, regime_used=5."""
    rlgbm = RegimeConditionedLGBM()
    _inject_models(rlgbm, global_val=0.0002)

    r = rlgbm.step(_make_tick(alg_rsm_regime_last=float("nan")))
    assert r["alg_rlgbm_regime_used"] == 5.0
    assert np.isfinite(r["alg_rlgbm_predicted_return"])


def test_no_models_returns_neutral(rlgbm):
    """No models loaded: neutral output (0.0 signal/return), regime computed."""
    r = rlgbm.step(_make_tick())
    assert r["alg_rlgbm_signal"] == 0.0
    assert r["alg_rlgbm_predicted_return"] == 0.0
    assert np.isfinite(r["alg_rlgbm_regime_used"])


def test_signal_range():
    """alg_rlgbm_signal in [-1, 1] across varied predictions."""
    rlgbm = RegimeConditionedLGBM()

    rng = np.random.default_rng(42)
    for _ in range(200):
        val = rng.normal(0, 0.01)
        _inject_models(rlgbm, trending_val=val, global_val=val)
        r = rlgbm.step(_make_tick(alg_rsm_regime_last=2.0, alg_rsm_confidence_last=0.8))
        sig = r["alg_rlgbm_signal"]
        if np.isfinite(sig):
            assert -1.0 <= sig <= 1.0, f"Signal out of range: {sig}"


def test_per_regime_features_differ():
    """Trending and ranging models use different feature subsets."""
    assert set(TRENDING_FEATURES) != set(RANGING_FEATURES)
    assert set(TRENDING_FEATURES) != set(VOLATILE_FEATURES)
    # All subsets are subsets of global
    assert set(TRENDING_FEATURES).issubset(set(GLOBAL_FEATURES))
    assert set(RANGING_FEATURES).issubset(set(GLOBAL_FEATURES))
    assert set(VOLATILE_FEATURES).issubset(set(GLOBAL_FEATURES))


def test_rare_regime_uses_global():
    """When per-regime model is missing, global model is used."""
    rlgbm = RegimeConditionedLGBM()
    # Only load global model, no per-regime models
    rlgbm._models = {"global": MockBooster(0.0003)}

    r = rlgbm.step(_make_tick(alg_rsm_regime_last=2.0, alg_rsm_confidence_last=0.8))
    # No trending model -> should fall back to global
    assert r["alg_rlgbm_regime_used"] == 5.0
    assert r["alg_rlgbm_predicted_return"] == pytest.approx(0.0003)
