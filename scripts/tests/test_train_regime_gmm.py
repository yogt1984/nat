"""Planted test for the GMM 5-D regime classifier training (BUG-3, part 1).

Two contracts:
  1. train_gmm recovers well-separated planted regimes (a GMM sanity/estimator check).
  2. load_features resolves the 5 regime features to REAL schema columns — the blocker-1
     regression: the old FEATURE_COLUMNS (`ill_kyle_lambda_300`, `tox_vpin_50`, ...) matched
     no emitted column, so training raised "Missing required features".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ (top-level module)
from train_regime_gmm import (  # noqa: E402
    FEATURE_COLUMNS,
    load_features,
    select_n_components,
    train_gmm,
)


# The real emitted schema names for the 5-D regime space (100%-finite, fast-warmup).
REAL_NAMES = [
    "illiq_kyle_500",
    "toxic_vpin_50",
    "derived_regime_type_score",
    "trend_hurst_300",
    "vol_returns_5m",
]


def _planted_regimes(seed: int = 0, k: int = 4, per: int = 250):
    """K well-separated 5-D Gaussians with known labels."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(k, 5))
    X = np.vstack([centers[c] + 0.4 * rng.standard_normal((per, 5)) for c in range(k)])
    y = np.concatenate([np.full(per, c) for c in range(k)])
    return X, y


class TestGmmRecovery:
    def test_recovers_planted_regimes(self):
        X, y = _planted_regimes(seed=1, k=4)
        gmm, scaler = train_gmm(X, n_components=4)
        pred = gmm.predict(scaler.transform(X))
        assert adjusted_rand_score(y, pred) > 0.95
        assert gmm.converged_

    def test_bic_auto_selects_near_k(self):
        X, _ = _planted_regimes(seed=2, k=4)
        from sklearn.preprocessing import StandardScaler
        n = select_n_components(StandardScaler().fit_transform(X), max_components=10)
        assert 3 <= n <= 6  # BIC should land near the 4 planted clusters


class TestFeatureColumnResolution:
    """Blocker-1 regression: the 5 features must resolve to real emitted columns."""

    def _write_parquet(self, tmp_path, cols):
        n = 400
        rng = np.random.default_rng(0)
        data = {c: rng.standard_normal(n) for c in cols}
        data["some_other_col"] = rng.standard_normal(n)  # decoy
        pl.DataFrame(data).write_parquet(tmp_path / "day.parquet")

    def test_loads_with_real_primary_names(self, tmp_path):
        # A parquet with the *real* schema names must load. With the old (wrong)
        # FEATURE_COLUMNS this raised "Missing required features" — the blocker-1 bug.
        self._write_parquet(tmp_path, REAL_NAMES)
        X, _ = load_features(tmp_path)
        assert X.shape[1] == 5 and X.shape[0] == 400

    def test_all_feature_columns_are_real_schema_names(self):
        # Guard against regressing to nonexistent names like `ill_kyle_lambda_300`.
        assert FEATURE_COLUMNS == REAL_NAMES

    def test_missing_feature_raises(self, tmp_path):
        self._write_parquet(tmp_path, REAL_NAMES[:-1])  # drop one required feature
        with pytest.raises(ValueError):
            load_features(tmp_path)

    def test_drops_non_finite_rows(self, tmp_path):
        # The Rust ingestor NaN-pads warmup; those rows must be dropped, not fed to sklearn.
        n = 300
        rng = np.random.default_rng(0)
        data = {c: rng.standard_normal(n) for c in REAL_NAMES}
        data[REAL_NAMES[0]][:50] = np.nan  # 50 warmup-style NaN rows in one feature
        pl.DataFrame(data).write_parquet(tmp_path / "day.parquet")
        X, _ = load_features(tmp_path)
        assert np.isfinite(X).all()
        assert X.shape[0] == 250  # the 50 NaN rows dropped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
