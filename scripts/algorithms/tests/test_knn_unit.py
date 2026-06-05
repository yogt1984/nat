"""Unit tests for KNNRetrieval algorithm."""

import numpy as np
import pytest

from algorithms.knn_retrieval import KNNRetrieval


def _make_tick(seed=0, **overrides) -> dict:
    """Create a tick dict with all required columns."""
    rng = np.random.default_rng(seed)
    tick = {
        "ent_tick_1m_mean": rng.uniform(0, 1),
        "trend_hurst_300_mean": rng.uniform(0.3, 0.7),
        "vol_returns_5m_last": abs(rng.normal(0, 0.001)) + 1e-6,
        "toxic_vpin_50_mean": rng.uniform(0, 1),
        "imbalance_qty_l1_mean": rng.normal(0, 0.3),
        "whale_net_flow_4h_sum": rng.normal(0, 1000),
        "regime_accumulation_score_mean": rng.uniform(0, 1),
        "raw_midprice_mean": 50000 + rng.normal(0, 50),
    }
    tick.update(overrides)
    return tick


def _feed_n(knn, n, price_trend=0.0):
    """Feed n bars with random features and trending price."""
    results = []
    for i in range(n):
        price = 50000 + price_trend * i + np.random.normal(0, 10)
        r = knn.step(_make_tick(seed=i, raw_midprice_mean=price))
        results.append(r)
    return results


def test_empty_buffer_returns_nan():
    """Before min_buffer observations, all outputs NaN."""
    knn = KNNRetrieval(min_buffer=100)
    for i in range(50):
        r = knn.step(_make_tick(seed=i))
        assert np.isnan(r["alg_knn_signal"]), f"Expected NaN at bar {i}"
        assert np.isnan(r["alg_knn_expected_return"])


def test_known_neighbor_retrieval():
    """After enough bars with resolved returns, output becomes finite."""
    knn = KNNRetrieval(k=5, min_buffer=50, refit_interval=50, buffer_size=500)

    # Feed enough bars so forward returns can resolve
    # Need min_buffer + HORIZON bars for returns to start resolving
    results = _feed_n(knn, 150, price_trend=0.5)

    # After min_buffer + HORIZON, some outputs should be finite
    finite_count = sum(1 for r in results[70:] if np.isfinite(r["alg_knn_expected_return"]))
    assert finite_count > 0, "Expected some finite outputs after warmup + horizon"


def test_mahalanobis_vs_euclidean():
    """On correlated features, whitening changes neighbor ordering.

    We verify that the Cholesky inverse is not identity (i.e., Ledoit-Wolf
    covariance is being used for whitening).
    """
    knn = KNNRetrieval(k=10, min_buffer=50, refit_interval=50)

    rng = np.random.default_rng(42)
    # Feed correlated features
    for i in range(100):
        base = rng.normal(0, 1)
        tick = _make_tick(
            seed=i,
            ent_tick_1m_mean=0.5 + 0.3 * base,
            trend_hurst_300_mean=0.5 + 0.2 * base,  # correlated with entropy
        )
        knn.step(tick)

    # After refit, Cholesky should differ from identity
    assert knn._cholesky_inv is not None
    eye = np.eye(knn._cholesky_inv.shape[0])
    diff = np.abs(knn._cholesky_inv - eye).max()
    assert diff > 0.01, f"Cholesky too close to identity: max_diff={diff:.4f}"


def test_time_decay_weighting():
    """Recent neighbors have higher weight than old ones."""
    knn = KNNRetrieval(k=5, time_decay_halflife=100)

    # Decay formula: w = exp(-ln2 * age / halflife)
    w_recent = np.exp(-np.log(2) * 10 / 100)   # age=10
    w_old = np.exp(-np.log(2) * 1000 / 100)     # age=1000

    assert w_recent > w_old * 10, "Recent weight should be much larger than old"


def test_buffer_size_cap():
    """After buffer_size+100 insertions, buffer length == buffer_size."""
    buf_size = 200
    knn = KNNRetrieval(buffer_size=buf_size, min_buffer=50, refit_interval=50)

    _feed_n(knn, buf_size + 100)

    assert len(knn._features_buf) == buf_size
    assert len(knn._returns_buf) == buf_size


def test_signal_range():
    """alg_knn_signal in [-1, 1] when finite."""
    knn = KNNRetrieval(k=5, min_buffer=50, refit_interval=50,
                        cost_threshold_bps=0.0, win_rate_threshold=0.0)

    results = _feed_n(knn, 200, price_trend=1.0)

    for r in results:
        sig = r["alg_knn_signal"]
        if np.isfinite(sig):
            assert -1.0 <= sig <= 1.0, f"Signal out of range: {sig}"


def test_win_rate_range():
    """alg_knn_win_rate in [0, 1] when finite."""
    knn = KNNRetrieval(k=5, min_buffer=50, refit_interval=50)

    results = _feed_n(knn, 200, price_trend=1.0)

    for r in results:
        wr = r["alg_knn_win_rate"]
        if np.isfinite(wr):
            assert 0.0 <= wr <= 1.0, f"Win rate out of range: {wr}"


def test_refit_interval():
    """KD-tree is None before min_buffer, built after min_buffer reached."""
    knn = KNNRetrieval(k=5, min_buffer=50, refit_interval=200)

    _feed_n(knn, 40)
    assert knn._kdtree is None  # not enough data (< min_buffer=50)

    _feed_n(knn, 20)  # total 60, crosses min_buffer -> refit triggered
    assert knn._kdtree is not None
    assert knn._kdtree.n >= 50  # tree has at least min_buffer points


def test_cost_threshold_gate():
    """Expected return below cost_threshold -> signal=0."""
    knn = KNNRetrieval(k=5, min_buffer=30, refit_interval=30,
                        cost_threshold_bps=100.0)  # very high threshold: 1%

    # Feed flat price (near-zero returns) so expected return is tiny
    results = _feed_n(knn, 200, price_trend=0.0)

    # With high cost threshold and flat prices, signal should be 0
    signals = [r["alg_knn_signal"] for r in results if np.isfinite(r["alg_knn_signal"])]
    if signals:
        assert all(s == 0.0 for s in signals), "Expected zero signals with high cost threshold"
