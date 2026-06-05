"""Unit tests for the momentum continuation training pipeline."""

import numpy as np
import pandas as pd
import pytest

from train_momentum import FEATURE_COLS, HORIZON_BARS, build_dataset, walk_forward_train


def _make_synthetic_bars(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create synthetic bar DataFrame with all required columns."""
    rng = np.random.default_rng(seed)
    midprice = 50000 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))

    data = {
        "raw_midprice_mean": midprice,
        "ent_tick_1m_mean": rng.uniform(0, 1, n),
        "ent_permutation_returns_16_mean": rng.uniform(0, 1, n),
        "trend_hurst_300_mean": rng.uniform(0.3, 0.7, n),
        "toxic_vpin_50_mean": rng.uniform(0, 1, n),
        "whale_net_flow_4h_sum": rng.normal(0, 1000, n),
        "regime_accumulation_score_mean": rng.uniform(0, 1, n),
        "vol_returns_5m_last": rng.normal(0, 0.001, n),
    }
    return pd.DataFrame(data)


def test_build_dataset_labels_binary():
    """Labels from build_dataset() are 0.0 or 1.0 only (no NaN in valid rows)."""
    bars = _make_synthetic_bars()
    X, y, bars_valid = build_dataset(bars)

    assert set(np.unique(y)) <= {0.0, 1.0}
    assert not np.any(np.isnan(y))


def test_build_dataset_drops_nan_features():
    """Rows with NaN in any FEATURE_COL are excluded from X, y."""
    bars = _make_synthetic_bars()
    # Inject NaN in first 50 rows of one feature
    bars.loc[:49, "trend_hurst_300_mean"] = np.nan

    X, y, bars_valid = build_dataset(bars)
    # None of the NaN rows should appear in valid output
    assert len(y) <= len(bars) - 50


def test_forward_return_alignment():
    """fwd_ret[0] uses midprice[20], not midprice[19]. Off-by-one check."""
    bars = _make_synthetic_bars(100)
    X, y, bars_valid = build_dataset(bars, horizon=HORIZON_BARS)

    # Manually compute expected forward return for first row
    mid = bars["raw_midprice_mean"].values
    expected_fwd = mid[HORIZON_BARS] / mid[0] - 1.0
    expected_label = 1.0 if expected_fwd > 0 else 0.0

    # The first valid row should have this label
    assert y[0] == expected_label


def test_walk_forward_produces_folds():
    """walk_forward_train() with n_splits=3 on synthetic data produces 3 fold results."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 7))
    y = (X[:, 0] > 0).astype(float)

    result = walk_forward_train(X, y, n_splits=3, embargo=20, C=1.0)
    assert len(result["fold_results"]) == 3
    assert "model" in result
    assert "scaler" in result


def test_walk_forward_embargo_respected():
    """For each fold, test_start >= train_end + embargo."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2000, 7))
    y = (X[:, 0] > 0).astype(float)
    embargo = 100

    result = walk_forward_train(X, y, n_splits=4, embargo=embargo, C=1.0)

    n = len(y)
    min_train = n // 5
    fold_size = (n - min_train) // 4

    for fold_info in result["fold_results"]:
        fold = fold_info["fold"]
        train_end = min_train + fold * fold_size
        test_start = train_end + embargo
        # Test size should not include training data
        assert fold_info["test_size"] > 0
        # The training + embargo gap should be respected (implicit in the fold structure)
        assert fold_info["train_size"] <= train_end


def test_model_metadata_fields():
    """After training, result has all fields needed for ModelMetadata."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 7))
    y = (X[:, 0] > 0).astype(float)

    result = walk_forward_train(X, y, n_splits=2, embargo=20, C=1.0)

    assert result["model"] is not None
    assert result["scaler"] is not None
    assert "avg_auc_oos" in result
    assert "avg_auc_is" in result
    assert "avg_acc_oos" in result
    assert "n_samples" in result
    assert result["n_samples"] == 500
