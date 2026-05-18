"""
Comprehensive tests for train_baseline.py.

Tests cover:
- Data loading (snapshot + data-dir paths)
- Feature auto-detection
- Feature/label preparation (NaN handling, forward returns, column filtering)
- ElasticNet training pipeline
- LightGBM training pipeline
- CLI argument parsing
- Edge cases (empty data, all-NaN columns, too few samples)
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import polars as pl
import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_baseline import (
    load_snapshot_data,
    load_data_dir,
    auto_detect_features,
    prepare_features_labels,
    train_elasticnet,
    train_lightgbm,
    _NON_FEATURE_PREFIXES,
    _NON_FEATURE_EXACT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n_rows: int = 5000, n_features: int = 10, seed: int = 42) -> pl.DataFrame:
    """Create a synthetic DataFrame matching real NAT data schema."""
    rng = np.random.default_rng(seed)
    # Metadata columns
    data = {
        "timestamp_ns": np.arange(n_rows, dtype=np.int64) * 100_000_000,
        "symbol": ["BTC"] * n_rows,
        "sequence_id": np.arange(n_rows, dtype=np.int64),
        "raw_midprice": 100_000.0 + np.cumsum(rng.normal(0, 0.01, n_rows)),
        "raw_spread": rng.uniform(0.01, 0.05, n_rows),
        "raw_spread_bps": rng.uniform(0.1, 0.5, n_rows),
        "raw_microprice": 100_000.0 + np.cumsum(rng.normal(0, 0.01, n_rows)),
        "raw_bid_depth_5": rng.uniform(100, 500, n_rows),
        "raw_ask_depth_5": rng.uniform(100, 500, n_rows),
        "raw_bid_depth_10": rng.uniform(200, 1000, n_rows),
        "raw_ask_depth_10": rng.uniform(200, 1000, n_rows),
        "raw_bid_orders_5": rng.integers(10, 50, n_rows).astype(float),
        "raw_ask_orders_5": rng.integers(10, 50, n_rows).astype(float),
    }
    # Feature columns
    for i in range(n_features):
        data[f"feature_{i}"] = rng.normal(0, 1, n_rows)

    return pl.DataFrame(data)


def _make_df_with_nan(n_rows: int = 5000, n_features: int = 10, seed: int = 42) -> pl.DataFrame:
    """Create DataFrame with some all-NaN columns (mimicking optional features)."""
    df = _make_df(n_rows, n_features, seed)
    # Add columns that are 100% NaN (like whale_flow)
    for name in ["whale_net_flow_1h", "whale_flow_intensity", "liquidation_risk_above_1pct"]:
        df = df.with_columns(pl.lit(float("nan")).alias(name))
    # Add a column with 30% NaN (should survive)
    rng = np.random.default_rng(seed + 1)
    vals = rng.normal(0, 1, n_rows)
    mask = rng.random(n_rows) < 0.3
    vals[mask] = float("nan")
    df = df.with_columns(pl.Series("partial_nan_feature", vals))
    return df


@pytest.fixture
def tmp_dir():
    """Create a temporary directory and clean up after."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_df():
    return _make_df()


@pytest.fixture
def snapshot_dir(tmp_dir):
    """Write sample data as snapshot (flat directory of parquet files)."""
    df = _make_df()
    # Split into two files
    n = len(df)
    df[:n // 2].write_parquet(tmp_dir / "part_0.parquet")
    df[n // 2:].write_parquet(tmp_dir / "part_1.parquet")
    return tmp_dir


@pytest.fixture
def data_dir_with_dates(tmp_dir):
    """Write sample data as date-partitioned directory."""
    df = _make_df()
    n = len(df)
    d1 = tmp_dir / "2026-05-10"
    d2 = tmp_dir / "2026-05-11"
    d1.mkdir()
    d2.mkdir()
    df[:n // 2].write_parquet(d1 / "BTC_00.parquet")
    df[n // 2:].write_parquet(d2 / "BTC_01.parquet")
    return tmp_dir


# ---------------------------------------------------------------------------
# Tests: Data Loading
# ---------------------------------------------------------------------------


class TestLoadSnapshotData:
    def test_loads_all_rows(self, snapshot_dir):
        df = load_snapshot_data(snapshot_dir)
        assert len(df) == 5000

    def test_concatenates_multiple_files(self, snapshot_dir):
        # Should have loaded from 2 files
        df = load_snapshot_data(snapshot_dir)
        assert df.shape[0] == 5000
        assert "raw_midprice" in df.columns

    def test_raises_on_empty_dir(self, tmp_dir):
        with pytest.raises(ValueError, match="No Parquet files"):
            load_snapshot_data(tmp_dir)

    def test_raises_on_nonexistent_dir(self, tmp_dir):
        bad = tmp_dir / "nonexistent"
        with pytest.raises(Exception):
            load_snapshot_data(bad)

    def test_deterministic_ordering(self, snapshot_dir):
        """Loading same snapshot twice should give same result."""
        df1 = load_snapshot_data(snapshot_dir)
        df2 = load_snapshot_data(snapshot_dir)
        assert df1.equals(df2)


class TestLoadDataDir:
    def test_loads_recursively(self, data_dir_with_dates):
        df = load_data_dir(data_dir_with_dates)
        assert len(df) == 5000

    def test_raises_on_empty_dir(self, tmp_dir):
        sub = tmp_dir / "empty_subdir"
        sub.mkdir()
        with pytest.raises(ValueError, match="No Parquet files"):
            load_data_dir(sub)

    def test_finds_nested_parquet(self, data_dir_with_dates):
        df = load_data_dir(data_dir_with_dates)
        assert "raw_midprice" in df.columns


# ---------------------------------------------------------------------------
# Tests: Feature Auto-Detection
# ---------------------------------------------------------------------------


class TestAutoDetectFeatures:
    def test_excludes_metadata_columns(self, sample_df):
        features = auto_detect_features(sample_df)
        for col in features:
            assert not col.startswith("timestamp")
            assert not col.startswith("symbol")
            assert not col.startswith("raw_")
            assert not col.startswith("target_")
            assert col != "sequence_id"

    def test_includes_feature_columns(self, sample_df):
        features = auto_detect_features(sample_df)
        for i in range(10):
            assert f"feature_{i}" in features

    def test_correct_count(self, sample_df):
        features = auto_detect_features(sample_df)
        # 10 feature_ columns, nothing else should be detected
        assert len(features) == 10

    def test_handles_empty_df(self):
        df = pl.DataFrame({"timestamp_ns": [], "symbol": [], "raw_midprice": []})
        features = auto_detect_features(df)
        assert features == []

    def test_excludes_target_prefix(self):
        df = pl.DataFrame({
            "feature_a": [1.0],
            "target_up": [1],
            "target_return": [0.01],
        })
        features = auto_detect_features(df)
        assert features == ["feature_a"]

    def test_non_feature_prefixes_are_complete(self):
        """Verify the exclusion lists cover all known metadata prefixes."""
        assert "timestamp" in _NON_FEATURE_PREFIXES
        assert "raw_" in _NON_FEATURE_PREFIXES
        assert "symbol" in _NON_FEATURE_PREFIXES
        assert "target_" in _NON_FEATURE_PREFIXES
        assert "sequence_id" in _NON_FEATURE_EXACT

    def test_with_real_like_columns(self):
        """Test with column names matching real NAT features."""
        cols = {
            "timestamp_ns": [1],
            "symbol": ["BTC"],
            "sequence_id": [1],
            "raw_midprice": [100.0],
            "raw_spread": [0.01],
            "imbalance_qty_l1": [0.5],
            "flow_count_1s": [10.0],
            "vol_midprice_std_1m": [0.01],
            "regime_entropy": [0.8],
        }
        df = pl.DataFrame(cols)
        features = auto_detect_features(df)
        assert set(features) == {
            "imbalance_qty_l1", "flow_count_1s",
            "vol_midprice_std_1m", "regime_entropy",
        }


# ---------------------------------------------------------------------------
# Tests: Feature/Label Preparation
# ---------------------------------------------------------------------------


class TestPrepareFeaturesLabels:
    def test_returns_correct_shapes(self, sample_df):
        feature_cols = [f"feature_{i}" for i in range(10)]
        X, y, used = prepare_features_labels(sample_df, feature_cols, horizon=100)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 10
        # Should have removed `horizon` rows at the end
        assert X.shape[0] <= len(sample_df) - 100

    def test_returns_used_feature_cols(self, sample_df):
        feature_cols = [f"feature_{i}" for i in range(10)]
        _, _, used = prepare_features_labels(sample_df, feature_cols, horizon=100)
        assert used == feature_cols

    def test_forward_return_calculation(self):
        """Forward return should be (future - current) / current."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        df = pl.DataFrame({
            "raw_midprice": prices,
            "feature_0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })
        X, y, _ = prepare_features_labels(df, ["feature_0"], horizon=2)
        # y[0] = (prices[2] - prices[0]) / prices[0] = 2/100 = 0.02
        assert abs(y[0] - 0.02) < 1e-10
        # y[1] = (prices[3] - prices[1]) / prices[1] = 2/101
        assert abs(y[1] - 2.0 / 101.0) < 1e-10

    def test_drops_high_nan_columns(self):
        df = _make_df_with_nan()
        all_cols = auto_detect_features(df)
        _, _, used = prepare_features_labels(df, all_cols, horizon=100)
        # 100% NaN columns should be dropped
        assert "whale_net_flow_1h" not in used
        assert "whale_flow_intensity" not in used
        assert "liquidation_risk_above_1pct" not in used

    def test_keeps_partial_nan_columns(self):
        df = _make_df_with_nan()
        all_cols = auto_detect_features(df)
        _, _, used = prepare_features_labels(df, all_cols, horizon=100, max_nan_frac=0.5)
        # 30% NaN column should survive
        assert "partial_nan_feature" in used

    def test_removes_nan_rows(self, sample_df):
        """After NaN column filtering, remaining NaN rows should be removed."""
        feature_cols = [f"feature_{i}" for i in range(10)]
        X, y, _ = prepare_features_labels(sample_df, feature_cols, horizon=100)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_fallback_to_midprice(self):
        """Should fall back to 'midprice' if 'raw_midprice' doesn't exist."""
        df = pl.DataFrame({
            "midprice": [100.0, 101.0, 102.0, 103.0, 104.0],
            "feature_0": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        X, y, _ = prepare_features_labels(df, ["feature_0"], horizon=1)
        assert len(X) > 0

    def test_raises_on_all_nan_features(self):
        """If all features are >50% NaN, should raise ValueError."""
        df = pl.DataFrame({
            "raw_midprice": [100.0, 101.0, 102.0],
            "bad_feature": [float("nan")] * 3,
        })
        with pytest.raises(ValueError, match="No usable feature columns"):
            prepare_features_labels(df, ["bad_feature"], horizon=1)

    def test_horizon_larger_than_data(self):
        df = pl.DataFrame({
            "raw_midprice": [100.0, 101.0],
            "feature_0": [0.1, 0.2],
        })
        X, y, _ = prepare_features_labels(df, ["feature_0"], horizon=10)
        assert len(X) == 0

    def test_max_nan_frac_threshold(self):
        """Test that max_nan_frac parameter controls the cutoff."""
        n = 100
        rng = np.random.default_rng(99)
        vals = rng.normal(0, 1, n)
        # 40% NaN
        vals[:40] = float("nan")
        df = pl.DataFrame({
            "raw_midprice": np.linspace(100, 110, n),
            "feat_40pct_nan": vals,
            "feat_clean": rng.normal(0, 1, n),
        })
        # With max_nan_frac=0.3 the 40%-NaN column should be dropped
        _, _, used = prepare_features_labels(
            df, ["feat_40pct_nan", "feat_clean"], horizon=5, max_nan_frac=0.3,
        )
        assert "feat_40pct_nan" not in used
        assert "feat_clean" in used

        # With max_nan_frac=0.5 it should survive
        _, _, used2 = prepare_features_labels(
            df, ["feat_40pct_nan", "feat_clean"], horizon=5, max_nan_frac=0.5,
        )
        assert "feat_40pct_nan" in used2


# ---------------------------------------------------------------------------
# Tests: Model Training
# ---------------------------------------------------------------------------


def _training_data(n: int = 2000, d: int = 5, seed: int = 42):
    """Generate synthetic training and test data with a weak signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    # y has a weak linear relationship with first feature
    y = 0.001 * X[:, 0] + rng.normal(0, 0.01, n)
    split = int(n * 0.7)
    return X[:split], y[:split], X[split:], y[split:]


class TestTrainElasticNet:
    def test_returns_four_values(self):
        X_tr, y_tr, X_te, y_te = _training_data()
        result = train_elasticnet(X_tr, y_tr, X_te, y_te)
        assert len(result) == 4
        model, scaler, hyperparams, metrics = result

    def test_model_can_predict(self):
        X_tr, y_tr, X_te, y_te = _training_data()
        model, scaler, _, _ = train_elasticnet(X_tr, y_tr, X_te, y_te)
        X_scaled = scaler.transform(X_te)
        preds = model.predict(X_scaled)
        assert preds.shape == (len(X_te),)

    def test_hyperparameters_populated(self):
        X_tr, y_tr, X_te, y_te = _training_data()
        _, _, hyperparams, _ = train_elasticnet(X_tr, y_tr, X_te, y_te)
        assert "alpha" in hyperparams
        assert "l1_ratio" in hyperparams
        assert hyperparams["alpha"] > 0

    def test_performance_metrics_populated(self):
        X_tr, y_tr, X_te, y_te = _training_data()
        _, _, _, metrics = train_elasticnet(X_tr, y_tr, X_te, y_te)
        assert "train_r2" in metrics
        assert "test_r2" in metrics
        assert "train_rmse" in metrics
        assert "test_rmse" in metrics
        assert metrics["train_samples"] == len(_training_data()[0])
        assert metrics["test_samples"] == len(_training_data()[2])

    def test_scaler_fitted(self):
        X_tr, y_tr, X_te, y_te = _training_data()
        _, scaler, _, _ = train_elasticnet(X_tr, y_tr, X_te, y_te)
        assert hasattr(scaler, "mean_")
        assert scaler.mean_ is not None

    def test_rmse_non_negative(self):
        X_tr, y_tr, X_te, y_te = _training_data()
        _, _, _, metrics = train_elasticnet(X_tr, y_tr, X_te, y_te)
        assert metrics["train_rmse"] >= 0
        assert metrics["test_rmse"] >= 0


class TestTrainLightGBM:
    def test_returns_three_values(self):
        X_tr, y_tr, X_te, y_te = _training_data()
        result = train_lightgbm(X_tr, y_tr, X_te, y_te)
        assert len(result) == 3

    def test_model_can_predict(self):
        X_tr, y_tr, X_te, y_te = _training_data()
        model, _, _ = train_lightgbm(X_tr, y_tr, X_te, y_te)
        preds = model.predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_hyperparameters_populated(self):
        X_tr, y_tr, X_te, y_te = _training_data()
        _, hyperparams, _ = train_lightgbm(X_tr, y_tr, X_te, y_te)
        assert "objective" in hyperparams
        assert hyperparams["objective"] == "regression"
        assert "best_iteration" in hyperparams
        assert hyperparams["best_iteration"] >= 1

    def test_performance_metrics_populated(self):
        X_tr, y_tr, X_te, y_te = _training_data()
        _, _, metrics = train_lightgbm(X_tr, y_tr, X_te, y_te)
        assert "train_r2" in metrics
        assert "test_r2" in metrics
        assert "train_rmse" in metrics
        assert "test_rmse" in metrics

    def test_early_stopping_respects_rounds(self):
        """Model should stop before 500 rounds on tiny noisy data."""
        X_tr, y_tr, X_te, y_te = _training_data(n=500, d=3)
        _, hyperparams, _ = train_lightgbm(X_tr, y_tr, X_te, y_te)
        assert hyperparams["best_iteration"] < 500

    def test_rmse_non_negative(self):
        X_tr, y_tr, X_te, y_te = _training_data()
        _, _, metrics = train_lightgbm(X_tr, y_tr, X_te, y_te)
        assert metrics["train_rmse"] >= 0
        assert metrics["test_rmse"] >= 0


# ---------------------------------------------------------------------------
# Tests: CLI Argument Parsing
# ---------------------------------------------------------------------------


class TestCLI:
    def test_data_dir_and_snapshot_are_mutually_exclusive(self):
        """Can't pass both --data-dir and --snapshot."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/train_baseline.py",
             "--data-dir", "/tmp/foo", "--snapshot", "bar",
             "--model", "lightgbm"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_one_of_data_dir_or_snapshot_required(self):
        """Must pass either --data-dir or --snapshot."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/train_baseline.py", "--model", "lightgbm"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_model_is_required(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/train_baseline.py", "--data-dir", "/tmp/foo"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_invalid_model_rejected(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/train_baseline.py",
             "--data-dir", "/tmp/foo", "--model", "xgboost"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Tests: End-to-End (small synthetic data)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_lightgbm_with_data_dir(self, data_dir_with_dates, tmp_dir):
        """Full pipeline: load → prepare → train → save with --data-dir."""
        output_dir = tmp_dir / "models"
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/train_baseline.py",
             "--data-dir", str(data_dir_with_dates),
             "--model", "lightgbm",
             "--horizon", "50",
             "--output-dir", str(output_dir),
             "--no-tracking"],
            capture_output=True, text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Model file should exist
        model_files = list(output_dir.glob("lightgbm_*.txt"))
        assert len(model_files) == 1
        # Metadata file should exist
        meta_files = list(output_dir.glob("*_metadata.json"))
        assert len(meta_files) == 1
        # Metadata should be valid JSON
        with open(meta_files[0]) as f:
            meta = json.load(f)
        assert meta["model_type"] == "lightgbm"
        assert len(meta["feature_names"]) == 10

    def test_elasticnet_with_snapshot(self, snapshot_dir, tmp_dir):
        """Full pipeline with --snapshot (requires experiments/snapshots/ prefix)."""
        # We need to set up the expected directory structure
        snapshots_base = tmp_dir / "experiments" / "snapshots" / "test_snap"
        snapshots_base.mkdir(parents=True)
        # Copy parquet files
        for f in snapshot_dir.glob("*.parquet"):
            shutil.copy(f, snapshots_base / f.name)

        output_dir = tmp_dir / "models"
        import subprocess
        # Run from tmp_dir so the relative path works
        result = subprocess.run(
            [sys.executable, str(Path("scripts/train_baseline.py").resolve()),
             "--snapshot", "test_snap",
             "--model", "elasticnet",
             "--horizon", "50",
             "--output-dir", str(output_dir),
             "--no-tracking"],
            capture_output=True, text=True,
            timeout=120,
            cwd=str(tmp_dir),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        model_files = list(output_dir.glob("elasticnet_*.pkl"))
        assert len(model_files) == 1

    def test_symbol_filter(self, tmp_dir):
        """--symbol should filter data to only that symbol."""
        # Create data with two symbols
        rng = np.random.default_rng(77)
        n = 2000
        data = {
            "timestamp_ns": np.arange(n, dtype=np.int64),
            "symbol": ["BTC"] * (n // 2) + ["ETH"] * (n // 2),
            "sequence_id": np.arange(n),
            "raw_midprice": 100.0 + np.cumsum(rng.normal(0, 0.01, n)),
            "feature_0": rng.normal(0, 1, n),
            "feature_1": rng.normal(0, 1, n),
        }
        df = pl.DataFrame(data)
        d = tmp_dir / "2026-05-10"
        d.mkdir()
        df.write_parquet(d / "data.parquet")

        output_dir = tmp_dir / "models"
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/train_baseline.py",
             "--data-dir", str(tmp_dir),
             "--model", "lightgbm",
             "--symbol", "BTC",
             "--horizon", "20",
             "--output-dir", str(output_dir),
             "--no-tracking"],
            capture_output=True, text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Filtered to BTC" in result.stdout

    def test_explicit_features(self, data_dir_with_dates, tmp_dir):
        """--features should override auto-detection."""
        output_dir = tmp_dir / "models"
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/train_baseline.py",
             "--data-dir", str(data_dir_with_dates),
             "--model", "lightgbm",
             "--horizon", "50",
             "--features", "feature_0", "feature_1",
             "--output-dir", str(output_dir),
             "--no-tracking"],
            capture_output=True, text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        meta_files = list(output_dir.glob("*_metadata.json"))
        with open(meta_files[0]) as f:
            meta = json.load(f)
        assert len(meta["feature_names"]) == 2

    def test_missing_data_dir_exits_gracefully(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/train_baseline.py",
             "--data-dir", "/tmp/nonexistent_xyz_456",
             "--model", "lightgbm",
             "--no-tracking"],
            capture_output=True, text=True,
        )
        assert "Error: Data directory not found" in result.stdout

    def test_too_few_samples_exits(self, tmp_dir):
        """With horizon > data size, should exit with error."""
        d = tmp_dir / "tiny"
        d.mkdir()
        df = _make_df(n_rows=50)
        df.write_parquet(d / "data.parquet")

        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/train_baseline.py",
             "--data-dir", str(d),
             "--model", "lightgbm",
             "--horizon", "1000",
             "--no-tracking"],
            capture_output=True, text=True,
            timeout=30,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Tests: Metadata and Model Saving
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_json_roundtrip(self, data_dir_with_dates, tmp_dir):
        """Metadata JSON should be valid and round-trippable."""
        output_dir = tmp_dir / "models"
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/train_baseline.py",
             "--data-dir", str(data_dir_with_dates),
             "--model", "lightgbm",
             "--horizon", "50",
             "--output-dir", str(output_dir),
             "--no-tracking"],
            capture_output=True, text=True,
            timeout=120,
        )
        meta_files = list(output_dir.glob("*_metadata.json"))
        assert len(meta_files) == 1
        with open(meta_files[0]) as f:
            meta = json.load(f)
        # Required fields
        for key in ["model_type", "model_name", "feature_names", "hyperparameters",
                     "performance_metrics", "training_date", "n_features"]:
            assert key in meta, f"Missing key: {key}"
        assert meta["n_features"] == len(meta["feature_names"])

    def test_model_name_has_no_slashes(self, data_dir_with_dates, tmp_dir):
        """Model name should be safe for filenames (no path separators)."""
        output_dir = tmp_dir / "models"
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/train_baseline.py",
             "--data-dir", str(data_dir_with_dates),
             "--model", "lightgbm",
             "--horizon", "50",
             "--output-dir", str(output_dir),
             "--no-tracking"],
            capture_output=True, text=True,
            timeout=120,
        )
        model_files = list(output_dir.glob("lightgbm_*.txt"))
        for f in model_files:
            assert "/" not in f.name
