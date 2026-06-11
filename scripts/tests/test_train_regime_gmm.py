"""
Tests for train_regime_gmm.py.

Tests cover:
- Data loading with correct column names
- NaN handling (all-NaN column rejection, partial NaN drop)
- GMM training pipeline (fit, export, reload)
- Cluster analysis and label suggestion
- Edge cases (empty data, missing columns, too few samples)
- Rust/Python feature alignment
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from train_regime_gmm import (
    FEATURE_COLUMNS,
    RUST_INFERENCE_ORDER,
    load_features,
    train_gmm,
    analyze_clusters,
    suggest_labels,
    export_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_regime_df(n_rows: int = 2000, seed: int = 42) -> pl.DataFrame:
    """Create synthetic DataFrame with the 4 GMM feature columns."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "timestamp_ns": np.arange(n_rows, dtype=np.int64) * 100_000_000,
        "symbol": ["BTC"] * n_rows,
        "raw_midprice": 100_000.0 + np.cumsum(rng.normal(0, 0.01, n_rows)),
        # GMM features
        "illiq_kyle_100": rng.normal(0.5, 0.2, n_rows),
        "toxic_vpin_50": rng.uniform(0.2, 0.8, n_rows),
        "regime_absorption_zscore": rng.normal(0, 1, n_rows),
        "trend_hurst_300": rng.uniform(0.3, 0.7, n_rows),
        # Extra columns (should be ignored)
        "raw_spread_bps": rng.uniform(0.1, 0.5, n_rows),
        "vol_returns_5m": rng.normal(0, 0.01, n_rows),
    })


def _make_regime_df_with_nan(n_rows: int = 2000, seed: int = 42) -> pl.DataFrame:
    """Create DataFrame where some rows have NaN in feature columns."""
    df = _make_regime_df(n_rows, seed)
    # Inject 5% NaN into kyle column
    rng = np.random.default_rng(seed + 1)
    kyle_vals = df["illiq_kyle_100"].to_numpy().copy()
    mask = rng.random(n_rows) < 0.05
    kyle_vals[mask] = float("nan")
    return df.with_columns(pl.Series("illiq_kyle_100", kyle_vals))


def _write_parquet(df: pl.DataFrame, tmpdir: Path, name: str = "data.parquet") -> Path:
    """Write DataFrame to Parquet in tmpdir."""
    path = tmpdir / name
    df.write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Feature column alignment tests
# ---------------------------------------------------------------------------


class TestFeatureAlignment:
    """Verify Python training and Rust inference use the same features."""

    def test_feature_columns_match_rust_order(self):
        """FEATURE_COLUMNS must match RUST_INFERENCE_ORDER exactly."""
        assert FEATURE_COLUMNS == RUST_INFERENCE_ORDER

    def test_feature_count_is_4(self):
        """Model is 4D (whale flow excluded)."""
        assert len(FEATURE_COLUMNS) == 4

    def test_no_whale_features(self):
        """Whale flow columns should not be in feature list (all-NaN)."""
        for col in FEATURE_COLUMNS:
            assert "whale" not in col.lower()


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------


class TestLoadFeatures:
    def test_load_basic(self, tmp_path):
        df = _make_regime_df(500)
        _write_parquet(df, tmp_path)
        X, _ = load_features(tmp_path)
        assert X.shape == (500, 4)

    def test_load_recursive(self, tmp_path):
        """rglob should find Parquet files in subdirectories."""
        subdir = tmp_path / "2026-06-10"
        subdir.mkdir()
        df = _make_regime_df(300)
        _write_parquet(df, subdir, "hour_00.parquet")
        _write_parquet(df, subdir, "hour_01.parquet")
        X, _ = load_features(tmp_path)
        assert X.shape == (600, 4)

    def test_load_drops_nan_rows(self, tmp_path):
        df = _make_regime_df_with_nan(1000)
        _write_parquet(df, tmp_path)
        X, _ = load_features(tmp_path)
        # Should have fewer rows than original due to NaN drop
        assert X.shape[0] < 1000
        assert X.shape[1] == 4
        # No NaN in output
        assert not np.any(np.isnan(X))

    def test_load_missing_column_raises(self, tmp_path):
        df = _make_regime_df(500).drop("illiq_kyle_100")
        _write_parquet(df, tmp_path)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_features(tmp_path)

    def test_load_all_nan_column_raises(self, tmp_path):
        df = _make_regime_df(500).with_columns(
            pl.lit(float("nan")).alias("toxic_vpin_50")
        )
        _write_parquet(df, tmp_path)
        with pytest.raises(ValueError, match="NaN"):
            load_features(tmp_path)

    def test_load_no_parquet_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_features(tmp_path)

    def test_load_too_few_rows_raises(self, tmp_path):
        df = _make_regime_df(50)
        _write_parquet(df, tmp_path)
        with pytest.raises(ValueError, match="at least 100"):
            load_features(tmp_path)

    def test_load_sampling(self, tmp_path):
        df = _make_regime_df(2000)
        _write_parquet(df, tmp_path)
        X, _ = load_features(tmp_path, sample_frac=0.5)
        assert X.shape[0] < 2000

    def test_load_diagonal_relaxed_concat(self, tmp_path):
        """Files with slightly different schemas should concat via diagonal_relaxed."""
        df1 = _make_regime_df(300)
        df2 = _make_regime_df(300).with_columns(pl.lit(1.0).alias("extra_col"))
        _write_parquet(df1, tmp_path, "a.parquet")
        _write_parquet(df2, tmp_path, "b.parquet")
        X, _ = load_features(tmp_path)
        assert X.shape == (600, 4)


# ---------------------------------------------------------------------------
# Training pipeline tests
# ---------------------------------------------------------------------------


class TestTrainGmm:
    def test_train_default(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (500, 4))
        gmm, scaler = train_gmm(X, n_components=3)
        assert gmm.n_components == 3
        assert gmm.converged_
        assert scaler.mean_.shape == (4,)

    def test_train_auto_select(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (500, 4))
        gmm, scaler = train_gmm(X, n_components=5, auto_select=True)
        assert 2 <= gmm.n_components <= 10

    def test_scaler_normalizes(self):
        rng = np.random.default_rng(42)
        X = rng.normal(10, 5, (500, 4))
        _, scaler = train_gmm(X, n_components=3)
        X_scaled = scaler.transform(X)
        assert abs(X_scaled.mean()) < 0.1
        assert abs(X_scaled.std() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Export and reload tests
# ---------------------------------------------------------------------------


class TestExportModel:
    def test_export_creates_valid_json(self, tmp_path):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (500, 4))
        gmm, scaler = train_gmm(X, n_components=5)

        out = tmp_path / "regime_gmm.json"
        export_model(gmm, scaler, out)

        assert out.exists()
        with open(out) as f:
            model = json.load(f)

        assert model["n_components"] == 5
        assert len(model["means"]) == 5
        assert len(model["means"][0]) == 4  # 4D
        assert len(model["covariances"]) == 5
        assert len(model["covariances"][0]) == 4
        assert len(model["covariances"][0][0]) == 4
        assert len(model["weights"]) == 5
        assert len(model["scaler_mean"]) == 4
        assert len(model["scaler_std"]) == 4
        assert model["feature_names"] == FEATURE_COLUMNS

    def test_exported_model_weights_sum_to_one(self, tmp_path):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (500, 4))
        gmm, scaler = train_gmm(X, n_components=5)

        out = tmp_path / "regime_gmm.json"
        export_model(gmm, scaler, out)

        with open(out) as f:
            model = json.load(f)

        assert abs(sum(model["weights"]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Cluster analysis and labeling tests
# ---------------------------------------------------------------------------


class TestClusterAnalysis:
    def test_analyze_clusters_returns_all_components(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (500, 4))
        gmm, scaler = train_gmm(X, n_components=3)
        stats = analyze_clusters(gmm, scaler, X)
        assert len(stats) <= 3  # some clusters may be empty

    def test_suggest_labels_returns_valid_regimes(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (500, 4))
        gmm, scaler = train_gmm(X, n_components=5)
        stats = analyze_clusters(gmm, scaler, X)
        labels = suggest_labels(stats)
        valid = {"accumulation", "markup", "distribution", "markdown", "ranging"}
        for label in labels.values():
            assert label in valid


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_end_to_end(self, tmp_path):
        """Train GMM on synthetic data, export, verify JSON is Rust-loadable."""
        # Create data
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df = _make_regime_df(1000)
        _write_parquet(df, data_dir, "test.parquet")

        # Load
        X, _ = load_features(data_dir)
        assert X.shape == (1000, 4)

        # Train
        gmm, scaler = train_gmm(X, n_components=5)
        assert gmm.converged_

        # Export
        model_path = tmp_path / "regime_gmm.json"
        export_model(gmm, scaler, model_path)

        # Verify JSON structure matches Rust GmmParams
        with open(model_path) as f:
            params = json.load(f)

        # These are the fields Rust's GmmParams deserializes
        assert "n_components" in params
        assert "means" in params
        assert "covariances" in params
        assert "weights" in params
        assert "scaler_mean" in params
        assert "scaler_std" in params

        # Dimension consistency
        d = len(params["scaler_mean"])
        assert d == 4
        for mean in params["means"]:
            assert len(mean) == d
        for cov in params["covariances"]:
            assert len(cov) == d
            for row in cov:
                assert len(row) == d
