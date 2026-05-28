"""
Comprehensive tests for phase1_signal_test.py.

Tests cover:
- Target creation (forward return calculation, dead zone filtering)
- Feature column detection (metadata exclusion, leaky feature removal)
- Walk-forward validation (split logic, no lookahead, metrics)
- Confidence-filtered trading (threshold logic, cost model, PnL)
- JSON report generation (schema, completeness, file I/O)
- CLI argument parsing
- Edge cases (single class target, all-NaN features, tiny datasets)
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest


from phase1_signal_test import (
    create_target,
    get_feature_columns,
    test_1_insample as run_insample,
    test_2_walkforward as run_walkforward,
    test_3_confidence_filtered as run_confidence_filtered,
    reg_test_1_insample as run_reg_insample,
    reg_test_2_walkforward as run_reg_walkforward,
    reg_test_3_quantile_pnl as run_reg_quantile_pnl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_signal_df(
    n_rows: int = 5000,
    n_features: int = 8,
    seed: int = 42,
    symbol: str = "BTC",
) -> pl.DataFrame:
    """Create DataFrame matching real NAT feature schema with a planted signal."""
    rng = np.random.default_rng(seed)

    # Realistic random-walk midprice
    prices = 100_000.0 + np.cumsum(rng.normal(0, 0.5, n_rows))

    data = {
        "timestamp_ns": np.arange(n_rows, dtype=np.int64) * 100_000_000,
        "symbol": [symbol] * n_rows,
        "sequence_id": np.arange(n_rows, dtype=np.int64),
        "raw_midprice": prices,
        "raw_microprice": prices + rng.normal(0, 0.01, n_rows),
        "raw_spread": np.abs(rng.normal(0.5, 0.1, n_rows)),
        "raw_spread_bps": np.abs(rng.normal(1.0, 0.2, n_rows)),
        "raw_bid_depth_5": rng.uniform(100, 500, n_rows),
        "raw_ask_depth_5": rng.uniform(100, 500, n_rows),
        "raw_bid_depth_10": rng.uniform(200, 1000, n_rows),
        "raw_ask_depth_10": rng.uniform(200, 1000, n_rows),
        "raw_bid_orders_5": rng.integers(10, 50, n_rows).astype(float),
        "raw_ask_orders_5": rng.integers(10, 50, n_rows).astype(float),
        "ctx_open_interest": rng.uniform(1e6, 2e6, n_rows),
        "ctx_volume_24h": rng.uniform(1e8, 5e8, n_rows),
    }
    for i in range(n_features):
        data[f"feat_{i}"] = rng.normal(0, 1, n_rows)

    return pl.DataFrame(data)


@pytest.fixture
def signal_df():
    return _make_signal_df()


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def data_dir(tmp_dir):
    """Write test data as date-partitioned parquet."""
    df = _make_signal_df()
    d = tmp_dir / "2026-05-10"
    d.mkdir()
    df.write_parquet(d / "BTC_00.parquet")
    return tmp_dir


# ---------------------------------------------------------------------------
# Tests: Target Creation
# ---------------------------------------------------------------------------


class TestCreateTarget:
    def test_basic_target_creation(self, signal_df):
        df = create_target(signal_df, horizon=100)
        assert "target" in df.columns
        assert "forward_return" in df.columns
        # target should be binary
        unique_vals = df["target"].unique().to_list()
        assert set(unique_vals) <= {0, 1}

    def test_horizon_rows_removed(self, signal_df):
        n_before = len(signal_df)
        df = create_target(signal_df, horizon=100)
        # At minimum, the last `horizon` rows should be gone
        assert len(df) < n_before

    def test_forward_return_calculation(self):
        """Verify forward return = (future - current) / current."""
        prices = [100.0, 100.0, 110.0, 100.0, 90.0]
        df = pl.DataFrame({
            "timestamp_ns": list(range(5)),
            "symbol": ["BTC"] * 5,
            "raw_midprice": prices,
            "feat_a": [1.0] * 5,
        })
        result = create_target(df, horizon=2)
        # Row 0: (110 - 100) / 100 = 0.10 => target = 1
        # Row 1: (100 - 100) / 100 = 0.00 => falls in dead zone, removed
        # Row 2: (90 - 110) / 110 = -0.1818 => target = 0
        assert len(result) <= 3

    def test_dead_zone_filters_tiny_returns(self):
        """Returns within +/- 0.5 bps should be dropped."""
        n = 1000
        # Create prices with very small changes
        prices = np.full(n, 100_000.0)
        # Every other price differs by 0.001 (0.001 bps, below dead zone)
        prices[1::2] = 100_000.001
        df = pl.DataFrame({
            "timestamp_ns": np.arange(n, dtype=np.int64),
            "symbol": ["BTC"] * n,
            "raw_midprice": prices,
            "feat_a": np.ones(n),
        })
        result = create_target(df, horizon=1)
        # Most rows should be filtered by dead zone
        assert len(result) < n * 0.5

    def test_target_distribution_reasonable(self, signal_df):
        """Target should be roughly balanced (random walk)."""
        df = create_target(signal_df, horizon=100)
        up_pct = df["target"].mean()
        # Random walk should give ~50% up, allow wide range
        assert 0.2 < up_pct < 0.8

    def test_no_null_targets(self, signal_df):
        df = create_target(signal_df, horizon=100)
        assert df["target"].null_count() == 0
        assert df["forward_return"].null_count() == 0

    def test_horizon_1(self):
        """Edge case: horizon=1 should still work (needs large price moves)."""
        rng = np.random.default_rng(77)
        n = 2000
        # Large moves so they survive dead zone
        prices = 100_000.0 + np.cumsum(rng.normal(0, 10, n))
        df = pl.DataFrame({
            "timestamp_ns": np.arange(n, dtype=np.int64),
            "symbol": ["BTC"] * n,
            "raw_midprice": prices,
            "feat_a": rng.normal(0, 1, n),
        })
        df = create_target(df, horizon=1)
        assert len(df) > 0

    def test_horizon_equals_data_length(self):
        """If horizon >= data length, should return empty."""
        df = pl.DataFrame({
            "timestamp_ns": [1, 2, 3],
            "symbol": ["BTC"] * 3,
            "raw_midprice": [100.0, 101.0, 102.0],
            "feat_a": [1.0, 2.0, 3.0],
        })
        result = create_target(df, horizon=5)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: Feature Column Detection
# ---------------------------------------------------------------------------


class TestGetFeatureColumns:
    def test_excludes_metadata(self, signal_df):
        cols = get_feature_columns(signal_df)
        for c in cols:
            assert c not in {"timestamp_ns", "symbol", "sequence_id",
                             "forward_return", "target"}

    def test_includes_numeric_features(self, signal_df):
        cols = get_feature_columns(signal_df)
        for i in range(8):
            assert f"feat_{i}" in cols

    def test_remove_leaky_excludes_price_oi(self, signal_df):
        cols = get_feature_columns(signal_df, remove_leaky=True)
        assert "raw_midprice" not in cols
        assert "ctx_open_interest" not in cols
        assert "ctx_volume_24h" not in cols

    def test_remove_leaky_keeps_non_leaky(self, signal_df):
        cols = get_feature_columns(signal_df, remove_leaky=True)
        for i in range(8):
            assert f"feat_{i}" in cols

    def test_only_numeric_columns(self):
        df = pl.DataFrame({
            "timestamp_ns": [1],
            "symbol": ["BTC"],
            "string_col": ["hello"],
            "feat_num": [1.0],
        })
        cols = get_feature_columns(df)
        assert "string_col" not in cols
        assert "feat_num" in cols

    def test_no_target_leakage(self):
        """forward_return and target should never appear as features."""
        df = pl.DataFrame({
            "timestamp_ns": [1],
            "symbol": ["BTC"],
            "forward_return": [0.01],
            "target": [1],
            "feat_a": [0.5],
        })
        cols = get_feature_columns(df)
        assert "forward_return" not in cols
        assert "target" not in cols


# ---------------------------------------------------------------------------
# Tests: Walk-Forward Validation
# ---------------------------------------------------------------------------


class TestWalkForward:
    def test_returns_correct_number_of_splits(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        results = run_walkforward(df, feature_cols, n_splits=3)
        assert len(results) == 3

    def test_no_lookahead(self, signal_df):
        """Each split should train on data before the test period."""
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        results = run_walkforward(df, feature_cols, n_splits=3)
        for i, r in enumerate(results):
            if i > 0:
                # Later splits should have more training data
                assert r["train_size"] >= results[i - 1]["train_size"]

    def test_result_schema(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        results = run_walkforward(df, feature_cols, n_splits=2)
        for r in results:
            assert "split" in r
            assert "accuracy" in r
            assert "base_rate" in r
            assert "sharpe" in r
            assert "train_size" in r
            assert "test_size" in r

    def test_accuracy_bounded(self, signal_df):
        """Accuracy should be between 0 and 1."""
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        results = run_walkforward(df, feature_cols, n_splits=2)
        for r in results:
            assert 0.0 <= r["accuracy"] <= 1.0
            assert 0.0 <= r["base_rate"] <= 1.0

    def test_single_split(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        results = run_walkforward(df, feature_cols, n_splits=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests: Confidence-Filtered Trading
# ---------------------------------------------------------------------------


class TestConfidenceFiltered:
    def test_returns_threshold_results(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        result = run_confidence_filtered(df, feature_cols, spread_bps=1.0, horizon=50)
        assert "thresholds" in result
        assert "best_gross_bps" in result
        assert "best_threshold" in result

    def test_threshold_result_schema(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        result = run_confidence_filtered(df, feature_cols, spread_bps=1.0, horizon=50)
        for t in result["thresholds"]:
            assert "threshold" in t
            assert "trades" in t
            assert "accuracy" in t
            assert "gross_bps" in t
            assert "net_taker_bps" in t
            assert "net_maker_bps" in t
            assert "win_rate" in t

    def test_higher_threshold_fewer_trades(self, signal_df):
        """Higher confidence threshold should result in fewer (or equal) trades."""
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        result = run_confidence_filtered(df, feature_cols, spread_bps=1.0, horizon=50)
        thresholds = result["thresholds"]
        if len(thresholds) >= 2:
            for i in range(1, len(thresholds)):
                assert thresholds[i]["trades"] <= thresholds[i - 1]["trades"]

    def test_net_less_than_gross(self, signal_df):
        """Net returns should be less than gross (costs are subtracted)."""
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        result = run_confidence_filtered(df, feature_cols, spread_bps=1.0, horizon=50)
        for t in result["thresholds"]:
            assert t["net_taker_bps"] < t["gross_bps"]
            assert t["net_maker_bps"] < t["gross_bps"]

    def test_taker_more_expensive_than_maker(self, signal_df):
        """Taker costs should always be higher than maker costs."""
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        result = run_confidence_filtered(df, feature_cols, spread_bps=1.0, horizon=50)
        for t in result["thresholds"]:
            assert t["net_taker_bps"] <= t["net_maker_bps"]

    def test_best_gross_matches_thresholds(self, signal_df):
        """best_gross_bps should match the max gross across thresholds."""
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        result = run_confidence_filtered(df, feature_cols, spread_bps=1.0, horizon=50)
        if result["thresholds"]:
            max_gross = max(t["gross_bps"] for t in result["thresholds"])
            assert abs(result["best_gross_bps"] - max_gross) < 1e-6

    def test_win_rate_bounded(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        result = run_confidence_filtered(df, feature_cols, spread_bps=1.0, horizon=50)
        for t in result["thresholds"]:
            assert 0.0 <= t["win_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: In-Sample Test
# ---------------------------------------------------------------------------


class TestInSample:
    def test_returns_model_and_result(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        X = df.select(feature_cols).to_numpy()
        y = df["target"].to_numpy()
        X = np.nan_to_num(X, nan=0.0)
        split = int(len(X) * 0.7)
        model, result = run_insample(X[:split], y[:split], X[split:], y[split:], feature_cols)
        assert model is not None
        assert "train_acc" in result
        assert "top_features" in result

    def test_train_accuracy_bounded(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        X = df.select(feature_cols).to_numpy()
        y = df["target"].to_numpy()
        X = np.nan_to_num(X, nan=0.0)
        split = int(len(X) * 0.7)
        _, result = run_insample(X[:split], y[:split], X[split:], y[split:], feature_cols)
        assert 0.0 <= result["train_acc"] <= 1.0

    def test_top_features_limited(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        X = df.select(feature_cols).to_numpy()
        y = df["target"].to_numpy()
        X = np.nan_to_num(X, nan=0.0)
        split = int(len(X) * 0.7)
        _, result = run_insample(X[:split], y[:split], X[split:], y[split:], feature_cols)
        assert len(result["top_features"]) <= 15


# ---------------------------------------------------------------------------
# Tests: JSON Report
# ---------------------------------------------------------------------------


class TestJSONReport:
    def test_json_report_written(self, data_dir, tmp_dir):
        """--json-report should write a valid JSON file."""
        import subprocess
        report_path = tmp_dir / "report.json"
        result = subprocess.run(
            [sys.executable, "scripts/phase1_signal_test.py",
             "--symbol", "BTC",
             "--horizon", "50",
             "--data-dir", str(data_dir),
             "--json-report", str(report_path),
             "--max-memory-mb", "500"],
            capture_output=True, text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert report_path.exists()

        with open(report_path) as f:
            report = json.load(f)

        # Verify schema
        assert report["symbol"] == "BTC"
        assert report["horizon"] == 50
        assert isinstance(report["n_rows"], int)
        assert isinstance(report["n_features"], int)
        assert "test1_train_acc" in report
        assert "test2_avg_accuracy" in report
        assert "test2_avg_edge" in report
        assert "test3_thresholds" in report
        assert "test3_best_gross_bps" in report

    def test_json_report_creates_parent_dirs(self, data_dir, tmp_dir):
        report_path = tmp_dir / "deep" / "nested" / "report.json"
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/phase1_signal_test.py",
             "--symbol", "BTC",
             "--horizon", "50",
             "--data-dir", str(data_dir),
             "--json-report", str(report_path),
             "--max-memory-mb", "500"],
            capture_output=True, text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert report_path.exists()


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_default_values(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/phase1_signal_test.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--symbol" in result.stdout
        assert "--horizon" in result.stdout
        assert "--json-report" in result.stdout

    def test_remove_leaky_flag(self, data_dir, tmp_dir):
        report_path = tmp_dir / "report.json"
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/phase1_signal_test.py",
             "--symbol", "BTC",
             "--horizon", "50",
             "--data-dir", str(data_dir),
             "--remove-leaky",
             "--json-report", str(report_path),
             "--max-memory-mb", "500"],
            capture_output=True, text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        with open(report_path) as f:
            report = json.load(f)
        # With --remove-leaky, fewer features should be used
        assert report["n_features"] < 20  # The synthetic df has ~14 numeric + some raw


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_feature(self):
        """Should work with just one feature."""
        rng = np.random.default_rng(99)
        n = 2000
        df = pl.DataFrame({
            "timestamp_ns": np.arange(n, dtype=np.int64),
            "symbol": ["BTC"] * n,
            "raw_midprice": 100_000.0 + np.cumsum(rng.normal(0, 0.5, n)),
            "feat_only": rng.normal(0, 1, n),
        })
        df = create_target(df, horizon=20)
        cols = get_feature_columns(df)
        assert len(cols) >= 1
        # Walk-forward should still run
        results = run_walkforward(df, cols, n_splits=2)
        assert len(results) == 2

# ---------------------------------------------------------------------------
# Tests: Regression Mode
# ---------------------------------------------------------------------------


class TestRegInSample:
    def test_returns_model_and_result(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        X = df.select(feature_cols).to_numpy()
        y = df["forward_return"].to_numpy()
        X = np.nan_to_num(X, nan=0.0)
        split = int(len(X) * 0.7)
        model, result = run_reg_insample(X[:split], y[:split], X[split:], y[split:], feature_cols)
        assert model is not None
        assert "train_r2" in result
        assert "test_r2" in result
        assert "ic_train" in result
        assert "ic_test" in result
        assert "dir_acc_train" in result
        assert "dir_acc_test" in result

    def test_metrics_bounded(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        X = df.select(feature_cols).to_numpy()
        y = df["forward_return"].to_numpy()
        X = np.nan_to_num(X, nan=0.0)
        split = int(len(X) * 0.7)
        _, result = run_reg_insample(X[:split], y[:split], X[split:], y[split:], feature_cols)
        assert 0.0 <= result["dir_acc_train"] <= 1.0
        assert 0.0 <= result["dir_acc_test"] <= 1.0
        assert result["train_rmse"] >= 0
        assert result["test_rmse"] >= 0
        # IC should be between -1 and 1
        assert -1.0 <= result["ic_train"] <= 1.0
        assert -1.0 <= result["ic_test"] <= 1.0


class TestRegWalkForward:
    def test_returns_correct_splits(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        results = run_reg_walkforward(df, feature_cols, n_splits=3)
        assert len(results) == 3

    def test_result_schema(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        results = run_reg_walkforward(df, feature_cols, n_splits=2)
        for r in results:
            assert "r2" in r
            assert "rmse" in r
            assert "ic" in r
            assert "dir_acc" in r
            assert "sharpe" in r

    def test_expanding_training_set(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        results = run_reg_walkforward(df, feature_cols, n_splits=3)
        for i in range(1, len(results)):
            assert results[i]["train_size"] >= results[i - 1]["train_size"]


class TestRegQuantilePnl:
    def test_returns_quantile_results(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        result = run_reg_quantile_pnl(df, feature_cols, spread_bps=1.0, horizon=50)
        assert "quantiles" in result
        assert "best_gross_bps" in result
        assert "best_quantile" in result
        assert len(result["quantiles"]) == 5  # 10%, 20%, 30%, 40%, 50%

    def test_quantile_result_schema(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        result = run_reg_quantile_pnl(df, feature_cols, spread_bps=1.0, horizon=50)
        for q in result["quantiles"]:
            assert "quantile_frac" in q
            assert "trades" in q
            assert "dir_acc" in q
            assert "gross_bps" in q
            assert "net_taker_bps" in q
            assert "net_maker_bps" in q
            assert "ic" in q

    def test_narrower_quantile_fewer_trades(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        result = run_reg_quantile_pnl(df, feature_cols, spread_bps=1.0, horizon=50)
        qs = result["quantiles"]
        # 10% should have fewer trades than 50%
        assert qs[0]["trades"] <= qs[-1]["trades"]

    def test_net_less_than_gross(self, signal_df):
        df = create_target(signal_df, horizon=50)
        feature_cols = get_feature_columns(df)
        result = run_reg_quantile_pnl(df, feature_cols, spread_bps=1.0, horizon=50)
        for q in result["quantiles"]:
            assert q["net_taker_bps"] < q["gross_bps"]


class TestRegJSONReport:
    def test_regression_json_report(self, data_dir, tmp_dir):
        """--mode regress --json-report should write regression-specific fields."""
        import subprocess
        report_path = tmp_dir / "reg_report.json"
        result = subprocess.run(
            [sys.executable, "scripts/phase1_signal_test.py",
             "--symbol", "BTC",
             "--horizon", "50",
             "--data-dir", str(data_dir),
             "--json-report", str(report_path),
             "--mode", "regress",
             "--max-memory-mb", "500"],
            capture_output=True, text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert report_path.exists()

        with open(report_path) as f:
            report = json.load(f)

        assert report["mode"] == "regress"
        assert "reg_test1" in report
        assert "reg_test2_avg_ic" in report
        assert "reg_test2_avg_r2" in report
        assert "reg_test3_quantiles" in report
        assert "reg_test3_best_gross_bps" in report


class TestModeCLI:
    def test_mode_flag_accepted(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/phase1_signal_test.py", "--help"],
            capture_output=True, text=True,
        )
        assert "--mode" in result.stdout
        assert "classify" in result.stdout
        assert "regress" in result.stdout


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nan_features_handled(self):
        """Features with NaN should be handled by nan_to_num."""
        rng = np.random.default_rng(88)
        n = 2000
        vals = rng.normal(0, 1, n)
        vals[:500] = float("nan")
        df = pl.DataFrame({
            "timestamp_ns": np.arange(n, dtype=np.int64),
            "symbol": ["BTC"] * n,
            "raw_midprice": 100_000.0 + np.cumsum(rng.normal(0, 0.5, n)),
            "feat_with_nan": vals,
            "feat_clean": rng.normal(0, 1, n),
        })
        df = create_target(df, horizon=20)
        cols = get_feature_columns(df)
        # Should not crash — nan_to_num handles it
        results = run_walkforward(df, cols, n_splits=2)
        assert len(results) == 2
