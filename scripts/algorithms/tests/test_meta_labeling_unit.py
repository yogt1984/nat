"""Unit tests for MetaLabeling algorithm and triple-barrier labeling."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from algorithms.meta_labeling import MetaLabeling
from build_meta_training_data import compute_triple_barrier_labels
from train_meta_labeling import purged_kfold_split


@pytest.fixture
def ml():
    return MetaLabeling()


def _make_tick(**overrides) -> dict:
    """Create a neutral tick dict with all required meta state columns."""
    tick = {
        "ent_tick_1m_mean": 0.50,
        "ent_rate_of_change_5s_mean": 0.0,
        "toxic_vpin_10_mean": 0.40,
        "toxic_index_mean": 0.30,
        "conc_hhi_last": 0.10,
        "whale_directional_agreement_last": 0.0,
        "vol_returns_5m_mean": 0.001,
        "vol_ratio_short_long_last": 1.0,
        "regime_clarity_last": 0.50,
        "raw_spread_bps_mean": 0.50,
    }
    tick.update(overrides)
    return tick


def test_probability_range():
    """alg_meta_probability always in [0, 1] with mock model."""
    ml = MetaLabeling()

    class MockModel:
        def predict_proba(self, X):
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    ml._model = MockModel()
    ml._scaler = None

    rng = np.random.default_rng(42)
    for _ in range(200):
        tick = _make_tick(
            ent_tick_1m_mean=rng.uniform(0, 1),
            vol_returns_5m_mean=rng.normal(0, 0.01),
        )
        r = ml.step(tick)
        prob = r["alg_meta_probability"]
        if np.isfinite(prob):
            assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"


def test_no_model_returns_neutral(ml):
    """No model: probability=0.5, side=0, size=0."""
    r = ml.step(_make_tick())
    assert r["alg_meta_probability"] == 0.5
    assert r["alg_meta_side"] == 0.0
    assert r["alg_meta_size"] == 0.0


def test_triple_barrier_upper_hit():
    """Price path hitting upper barrier first -> label=1."""
    # Price starts at 100, immediately rises to 101 (1% = 100 bps)
    prices = np.array([100.0, 100.03, 100.05, 100.06, 100.10, 100.15])
    labels = compute_triple_barrier_labels(
        prices, profit_target_bps=5.0, stop_loss_bps=10.0, max_holding_bars=10
    )
    assert labels[0] == 1.0  # 0.05% = 5 bps profit target hit


def test_triple_barrier_lower_hit():
    """Price path hitting lower barrier first -> label=0."""
    # Price starts at 100, drops to 99.89 (11 bps down)
    prices = np.array([100.0, 99.95, 99.92, 99.90, 99.88, 99.85])
    labels = compute_triple_barrier_labels(
        prices, profit_target_bps=5.0, stop_loss_bps=10.0, max_holding_bars=10
    )
    assert labels[0] == 0.0  # 10 bps stop loss hit


def test_triple_barrier_time_expiry():
    """Neither barrier hit -> label = sign(exit return)."""
    # Price stays flat, barely moves
    prices = np.array([100.0, 100.01, 100.01, 99.99, 100.00, 100.02])
    labels = compute_triple_barrier_labels(
        prices, profit_target_bps=50.0, stop_loss_bps=50.0, max_holding_bars=3
    )
    # After 3 bars from entry at bar 0, exit at bar 3 (99.99) < entry -> label=0
    assert labels[0] == 0.0


def test_purged_no_leakage():
    """For K=5 fold split with embargo=10, no test index within 10 of any train index."""
    n = 500
    splits = purged_kfold_split(n, k=5, embargo=10)

    assert len(splits) == 5

    for train_idx, test_idx in splits:
        train_set = set(train_idx)
        for t in test_idx:
            # No train index should be within embargo distance of any test index
            nearby = set(range(max(0, t - 10), min(n, t + 11)))
            overlap = train_set & nearby
            assert len(overlap) == 0, (
                f"Leakage: test idx {t} has train indices {overlap} within embargo"
            )


def test_nan_input_propagation(ml):
    """NaN in required column -> all outputs NaN."""
    r = ml.step(_make_tick(ent_tick_1m_mean=float("nan")))
    assert np.isnan(r["alg_meta_probability"])
    assert np.isnan(r["alg_meta_side"])
    assert np.isnan(r["alg_meta_size"])
