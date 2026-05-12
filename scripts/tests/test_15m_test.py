"""
Skeptical tests for 15m_test.py — 15-minute smoke test pipeline.

Test philosophy:
  - Each check function returns CheckResult with correct pass/fail
  - Gate logic: critical failures prevent phases 3-4
  - Report generation: valid JSON + markdown output
  - Edge cases: empty data, single symbol, all-NaN features, constant features
  - Config: defaults work when TOML section missing
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import sys
import importlib.util

# Import 15m_test.py (numeric prefix requires spec-based import)
_spec = importlib.util.spec_from_file_location(
    "smoke_test", str(Path(__file__).parent.parent / "15m_test.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["smoke_test"] = _mod
_spec.loader.exec_module(_mod)

CheckResult = _mod.CheckResult
PhaseResult = _mod.PhaseResult
SmokeTestReport = _mod.SmokeTestReport
_core_feature_cols = _mod._core_feature_cols
_numeric_feature_cols = _mod._numeric_feature_cols
_optional_feature_cols = _mod._optional_feature_cols
_vector_columns = _mod._vector_columns
check_bar_aggregation = _mod.check_bar_aggregation
check_continuity = _mod.check_continuity
check_correlation_collapse = _mod.check_correlation_collapse
check_cross_symbol_consistency = _mod.check_cross_symbol_consistency
check_emission_rate = _mod.check_emission_rate
check_feature_completeness = _mod.check_feature_completeness
check_file_integrity = _mod.check_file_integrity
check_kmeans_cluster = _mod.check_kmeans_cluster
check_nan_core = _mod.check_nan_core
check_nan_optional = _mod.check_nan_optional
check_outlier_fraction = _mod.check_outlier_fraction
check_range = _mod.check_range
check_schema = _mod.check_schema
check_sequence_monotonicity = _mod.check_sequence_monotonicity
check_zero_variance = _mod.check_zero_variance
load_config = _mod.load_config
phase_validate = _mod.phase_validate
phase_profile = _mod.phase_profile
phase_cluster = _mod.phase_cluster
DEFAULT_THRESHOLDS = _mod.DEFAULT_THRESHOLDS
_format_markdown = _mod._format_markdown


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_healthy_df(n_rows: int = 9000, n_symbols: int = 3) -> pd.DataFrame:
    """Create a synthetic healthy dataframe matching ingestor output."""
    rng = np.random.RandomState(42)
    rows_per_sym = n_rows // n_symbols
    symbols = ["BTC", "ETH", "SOL"][:n_symbols]

    frames = []
    for sym in symbols:
        # 100ms intervals = 10/sec
        ts_start = 1_700_000_000_000_000_000  # some base ns
        ts = ts_start + np.arange(rows_per_sym) * 100_000_000  # 100ms gaps
        df = pd.DataFrame({
            "timestamp_ns": ts,
            "symbol": sym,
            "sequence_id": np.arange(rows_per_sym),
            # Entropy features
            "ent_permutation_returns_8": rng.randn(rows_per_sym),
            "ent_permutation_returns_16": rng.randn(rows_per_sym),
            "ent_tick_1s": rng.randn(rows_per_sym),
            "ent_tick_5s": rng.randn(rows_per_sym),
            "ent_book_shape": rng.randn(rows_per_sym),
            # Flow features
            "flow_count_1s": rng.exponential(5, rows_per_sym),
            "flow_volume_1s": rng.exponential(100, rows_per_sym),
            # Orderflow features
            "imbalance_qty_l1": rng.randn(rows_per_sym),
            "imbalance_qty_l5": rng.randn(rows_per_sym),
            # Trend features
            "trend_momentum_60": rng.randn(rows_per_sym),
            "trend_hurst_300": rng.uniform(0.3, 0.7, rows_per_sym),
            # Volatility features
            "vol_returns_1m": rng.exponential(0.01, rows_per_sym),
            "vol_zscore": rng.randn(rows_per_sym),
            # Illiquidity
            "illiq_kyle_100": rng.exponential(1, rows_per_sym),
            # Toxicity
            "toxic_vpin_10": rng.uniform(0, 1, rows_per_sym),
            # Context
            "ctx_funding_rate": rng.randn(rows_per_sym) * 0.001,
            # Derived
            "derived_entropy_trend_zscore": rng.randn(rows_per_sym),
            # Regime
            "regime_absorption_1h": rng.randn(rows_per_sym),
            # Raw
            "raw_midprice": 50000 + rng.randn(rows_per_sym) * 100,
            "raw_spread_bps": rng.exponential(0.5, rows_per_sym),
            # Optional (all NaN)
            "whale_net_flow_1h": np.full(rows_per_sym, np.nan),
            "liquidation_risk_above_1pct": np.full(rows_per_sym, np.nan),
            "top5_concentration": np.full(rows_per_sym, np.nan),
        })
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def healthy_df():
    return _make_healthy_df()


@pytest.fixture
def cfg():
    return dict(DEFAULT_THRESHOLDS)


@pytest.fixture
def tmp_parquet_dir(healthy_df):
    """Write healthy_df to a temp dir as parquet."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.parquet"
        healthy_df.to_parquet(path)
        yield Path(d)


# ===========================================================================
# Config tests
# ===========================================================================


def test_load_config_missing_file():
    cfg = load_config(Path("/nonexistent/path.toml"))
    assert cfg == DEFAULT_THRESHOLDS


def test_load_config_defaults():
    cfg = dict(DEFAULT_THRESHOLDS)
    assert cfg["expected_rate_per_sec"] == 10
    assert cfg["emission_rate_tolerance"] == 0.20
    assert "whale" in cfg["optional_vectors"]


# ===========================================================================
# Phase 2: Validation checks
# ===========================================================================


class TestFileIntegrity:
    def test_pass_valid_files(self, tmp_parquet_dir):
        result = check_file_integrity(tmp_parquet_dir)
        assert result.passed
        assert result.critical
        assert "1/1 files readable" in result.message

    def test_fail_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            result = check_file_integrity(Path(d))
            assert not result.passed
            assert "No parquet files" in result.message

    def test_fail_corrupt_file(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bad.parquet").write_text("not a parquet file")
            result = check_file_integrity(Path(d))
            assert not result.passed
            assert "corrupt" in result.message.lower() or "1" in result.message


class TestSchema:
    def test_pass_healthy(self, healthy_df):
        result = check_schema(healthy_df)
        assert result.passed
        assert result.critical

    def test_fail_missing_timestamp(self, healthy_df):
        df = healthy_df.drop(columns=["timestamp_ns"])
        result = check_schema(df)
        assert not result.passed
        assert "timestamp_ns" in result.message

    def test_fail_missing_symbol(self, healthy_df):
        df = healthy_df.drop(columns=["symbol"])
        result = check_schema(df)
        assert not result.passed
        assert "symbol" in result.message


class TestEmissionRate:
    def test_pass_healthy(self, healthy_df, cfg):
        result = check_emission_rate(healthy_df, cfg)
        assert result.passed
        assert result.critical

    def test_fail_too_slow(self, healthy_df, cfg):
        # Stretch timestamps to make rate ~5/sec (below 8/sec threshold)
        df = healthy_df.copy()
        for sym in df["symbol"].unique():
            mask = df["symbol"] == sym
            ts = df.loc[mask, "timestamp_ns"].values
            ts_start = ts[0]
            df.loc[mask, "timestamp_ns"] = ts_start + np.arange(mask.sum()) * 200_000_000
        result = check_emission_rate(df, cfg)
        assert not result.passed


class TestCrossSymbol:
    def test_pass_equal(self, healthy_df, cfg):
        result = check_cross_symbol_consistency(healthy_df, cfg)
        assert result.passed

    def test_fail_unbalanced(self, cfg):
        df = _make_healthy_df(9000, 3)
        # Drop half of SOL
        sol_idx = df[df["symbol"] == "SOL"].index[:1500]
        df = df.drop(sol_idx)
        result = check_cross_symbol_consistency(df, cfg)
        assert not result.passed


class TestNanCore:
    def test_pass_healthy(self, healthy_df, cfg):
        result = check_nan_core(healthy_df, cfg)
        assert result.passed
        assert result.critical

    def test_fail_high_nan(self, healthy_df, cfg):
        df = healthy_df.copy()
        # Set 50% of entropy features to NaN
        mask = np.random.RandomState(1).random(len(df)) < 0.5
        df.loc[mask, "ent_permutation_returns_8"] = np.nan
        result = check_nan_core(df, cfg)
        assert not result.passed


class TestNanOptional:
    def test_always_passes(self, healthy_df, cfg):
        result = check_nan_optional(healthy_df, cfg)
        assert result.passed  # optional is informational only
        assert not result.critical


class TestContinuity:
    def test_pass_no_gaps(self, healthy_df, cfg):
        result = check_continuity(healthy_df, cfg)
        assert result.passed

    def test_fail_with_gap(self, healthy_df, cfg):
        df = healthy_df.copy()
        # Insert a 30s gap for BTC
        btc = df[df["symbol"] == "BTC"].copy()
        mid = len(btc) // 2
        btc.iloc[mid:, btc.columns.get_loc("timestamp_ns")] += 30_000_000_000
        df.loc[df["symbol"] == "BTC"] = btc.values
        result = check_continuity(df, cfg)
        assert not result.passed


class TestSequenceMonotonicity:
    def test_pass_monotonic(self, healthy_df):
        result = check_sequence_monotonicity(healthy_df)
        assert result.passed

    def test_fail_non_monotonic(self, healthy_df):
        df = healthy_df.copy()
        # Swap two timestamps for BTC
        btc_idx = df[df["symbol"] == "BTC"].index
        ts = df.loc[btc_idx, "timestamp_ns"].values.copy()
        ts[10], ts[11] = ts[11], ts[10]
        df.loc[btc_idx, "timestamp_ns"] = ts
        result = check_sequence_monotonicity(df)
        assert not result.passed


# ===========================================================================
# Phase 3: Profile checks
# ===========================================================================


class TestZeroVariance:
    def test_pass_healthy(self, healthy_df):
        result = check_zero_variance(healthy_df)
        assert result.passed

    def test_fail_many_constant(self, healthy_df):
        df = healthy_df.copy()
        # Make 50% of numeric features constant
        num_cols = _numeric_feature_cols(df)
        for c in num_cols[:len(num_cols) // 2]:
            df[c] = 1.0
        result = check_zero_variance(df)
        assert not result.passed


class TestRange:
    def test_pass_healthy(self, healthy_df):
        result = check_range(healthy_df)
        assert result.passed

    def test_fail_overflow(self, healthy_df):
        df = healthy_df.copy()
        df.loc[0, "ent_tick_1s"] = 1e16
        result = check_range(df)
        assert not result.passed


class TestCorrelationCollapse:
    def test_pass_healthy(self, healthy_df):
        result = check_correlation_collapse(healthy_df)
        assert result.passed

    def test_fail_all_correlated(self):
        rng = np.random.RandomState(42)
        n = 1000
        base = rng.randn(n)
        df = pd.DataFrame({
            "timestamp_ns": np.arange(n),
            "symbol": "BTC",
            **{f"feat_{i}": base + rng.randn(n) * 0.001 for i in range(25)}
        })
        result = check_correlation_collapse(df)
        assert not result.passed


class TestBarAggregation:
    def test_pass_healthy(self, healthy_df, cfg):
        result = check_bar_aggregation(healthy_df, cfg)
        assert result.passed
        assert "bars at 1min" in result.message


class TestFeatureCompleteness:
    def test_always_passes(self, healthy_df):
        result = check_feature_completeness(healthy_df)
        assert result.passed  # informational


# ===========================================================================
# Phase 4: Cluster
# ===========================================================================


class TestKMeansCluster:
    def test_pass_healthy(self, healthy_df, cfg):
        result = check_kmeans_cluster(healthy_df, cfg)
        assert result.passed
        assert "Silhouette" in result.message
        assert result.details["active_clusters"] >= 2

    def test_fail_too_few_rows(self, cfg):
        df = _make_healthy_df(30, 1)  # only 30 rows
        result = check_kmeans_cluster(df, cfg)
        assert not result.passed


# ===========================================================================
# Phase orchestration
# ===========================================================================


class TestPhaseValidate:
    def test_gate_on_critical(self, healthy_df, cfg):
        # Remove timestamp_ns to trigger critical schema failure
        df = healthy_df.drop(columns=["timestamp_ns"])
        with tempfile.TemporaryDirectory() as d:
            healthy_df.to_parquet(Path(d) / "test.parquet")
            result = phase_validate(df, Path(d), cfg)
            assert result.gated
            assert not result.passed

    def test_no_gate_on_noncritical(self, healthy_df, tmp_parquet_dir, cfg):
        # Insert a gap (non-critical failure)
        df = healthy_df.copy()
        btc = df[df["symbol"] == "BTC"].copy()
        mid = len(btc) // 2
        btc.iloc[mid:, btc.columns.get_loc("timestamp_ns")] += 30_000_000_000
        df.loc[df["symbol"] == "BTC"] = btc.values
        result = phase_validate(df, tmp_parquet_dir, cfg)
        assert not result.gated  # continuity is non-critical
        assert not result.passed  # but phase still fails


class TestPhaseProfile:
    def test_pass_healthy(self, healthy_df, cfg):
        result = phase_profile(healthy_df, cfg)
        assert result.passed


class TestPhaseCluster:
    def test_pass_healthy(self, healthy_df, cfg):
        result = phase_cluster(healthy_df, cfg)
        assert result.passed


# ===========================================================================
# Report
# ===========================================================================


class TestReport:
    def test_markdown_format(self, healthy_df, cfg, tmp_parquet_dir):
        report = SmokeTestReport(
            timestamp="2026-05-12T12:00:00",
            data_dir=str(tmp_parquet_dir),
            total_rows=len(healthy_df),
            symbols=["BTC", "ETH", "SOL"],
            phases=[phase_validate(healthy_df, tmp_parquet_dir, cfg)],
            overall_passed=True,
            critical_passed=True,
        )
        md = _format_markdown(report)
        assert "# 15-Minute Smoke Test Report" in md
        assert "PASS" in md or "FAIL" in md
        assert "Validate" in md

    def test_json_serializable(self):
        from dataclasses import asdict
        report = SmokeTestReport(
            timestamp="2026-05-12T12:00:00",
            data_dir="/tmp/test",
            total_rows=100,
            symbols=["BTC"],
            phases=[PhaseResult("test", [
                CheckResult("check1", True, False, "ok", {"val": 1.0}, 10.0)
            ], True, False, 1.0)],
            overall_passed=True,
            critical_passed=True,
        )
        serialized = json.dumps(asdict(report))
        parsed = json.loads(serialized)
        assert parsed["overall_passed"] is True
        assert len(parsed["phases"]) == 1


# ===========================================================================
# Helper functions
# ===========================================================================


class TestHelpers:
    def test_core_feature_cols(self, healthy_df):
        cols = _core_feature_cols(healthy_df)
        assert len(cols) > 0
        # Should include entropy, flow, trend etc
        assert any("ent_" in c for c in cols)
        assert any("flow_" in c for c in cols)
        # Should NOT include optional
        assert not any("whale_" in c for c in cols)

    def test_optional_feature_cols(self, healthy_df, cfg):
        cols = _optional_feature_cols(healthy_df, cfg)
        assert any("whale_" in c for c in cols)
        assert any("liquidation_" in c for c in cols)

    def test_vector_columns(self, healthy_df):
        ent_cols = _vector_columns(healthy_df, "entropy")
        assert len(ent_cols) > 0
        assert all("ent_" in c for c in ent_cols)
