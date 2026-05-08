"""Tests for alpha.combiner — feature selection, dedup, signal combination."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import pytest
from alpha.combiner import (
    FeatureSpec,
    CombineResult,
    select_top_features,
    deduplicate_by_correlation,
    combine_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_screen_results(n=10, symbol="BTC", horizon="4h"):
    """Create synthetic screen result dicts."""
    results = []
    for i in range(n):
        results.append({
            "feature": f"feat_{i}",
            "symbol": symbol,
            "horizon": horizon,
            "horizon_bars": 16,
            "ic_mean": 0.05 - i * 0.003,
            "ic_std": 0.02,
            "turnover": 0.5 + i * 0.05,
            "significant": i < 5,
        })
    return results


def _make_polars_df(n_rows=500, feature_names=None):
    """Create synthetic polars DataFrame with feature columns."""
    np.random.seed(42)
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(5)]
    data = {name: np.random.randn(n_rows) for name in feature_names}
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# select_top_features
# ---------------------------------------------------------------------------


class TestSelectTopFeatures:
    def test_returns_top_n(self):
        results = _make_screen_results(20)
        specs = select_top_features(results, symbol="BTC", top_n=5)
        assert len(specs) == 5

    def test_sorted_by_abs_ic(self):
        results = _make_screen_results(10)
        specs = select_top_features(results, symbol="BTC", top_n=10)
        ics = [abs(s.ic_mean) for s in specs]
        assert ics == sorted(ics, reverse=True)

    def test_filter_by_symbol(self):
        results = _make_screen_results(5, symbol="BTC")
        results += _make_screen_results(5, symbol="ETH")
        specs = select_top_features(results, symbol="ETH", top_n=10)
        assert all(s.symbol == "ETH" for s in specs)

    def test_filter_by_horizon(self):
        results = _make_screen_results(5, horizon="4h")
        results += _make_screen_results(5, horizon="1d")
        specs = select_top_features(results, symbol="BTC", horizon="1d", top_n=10)
        assert all(s.horizon == "1d" for s in specs)

    def test_require_significant(self):
        results = _make_screen_results(10)
        specs = select_top_features(results, symbol="BTC", require_significant=True, top_n=20)
        assert len(specs) == 5  # only first 5 are significant

    def test_deduplicates_features(self):
        """Same feature at multiple horizons — keep best."""
        results = [
            {"feature": "feat_0", "symbol": "BTC", "horizon": "4h", "horizon_bars": 16,
             "ic_mean": 0.05, "ic_std": 0.02, "turnover": 0.5, "significant": True},
            {"feature": "feat_0", "symbol": "BTC", "horizon": "1d", "horizon_bars": 96,
             "ic_mean": 0.03, "ic_std": 0.02, "turnover": 0.5, "significant": True},
        ]
        specs = select_top_features(results, symbol="BTC", top_n=5)
        assert len(specs) == 1
        assert specs[0].ic_mean == 0.05  # kept the higher IC


# ---------------------------------------------------------------------------
# deduplicate_by_correlation
# ---------------------------------------------------------------------------


class TestDeduplicateByCorrelation:
    def test_keeps_uncorrelated(self):
        np.random.seed(42)
        df = pl.DataFrame({
            "feat_0": np.random.randn(500),
            "feat_1": np.random.randn(500),
            "feat_2": np.random.randn(500),
        })
        specs = [
            FeatureSpec("feat_0", "BTC", "4h", 16, 0.05, 0.5),
            FeatureSpec("feat_1", "BTC", "4h", 16, 0.04, 0.5),
            FeatureSpec("feat_2", "BTC", "4h", 16, 0.03, 0.5),
        ]
        result = deduplicate_by_correlation(df, specs, max_corr=0.8)
        assert len(result) == 3  # all uncorrelated → all kept

    def test_drops_highly_correlated(self):
        np.random.seed(42)
        base = np.random.randn(500)
        df = pl.DataFrame({
            "feat_0": base,
            "feat_1": base + np.random.normal(0, 0.01, 500),  # highly correlated
            "feat_2": np.random.randn(500),
        })
        specs = [
            FeatureSpec("feat_0", "BTC", "4h", 16, 0.05, 0.5),
            FeatureSpec("feat_1", "BTC", "4h", 16, 0.03, 0.5),
            FeatureSpec("feat_2", "BTC", "4h", 16, 0.04, 0.5),
        ]
        result = deduplicate_by_correlation(df, specs, max_corr=0.8)
        names = [s.name for s in result]
        # feat_1 should be dropped (lower IC than correlated feat_0)
        assert "feat_0" in names
        assert "feat_1" not in names
        assert "feat_2" in names

    def test_single_feature(self):
        df = pl.DataFrame({"feat_0": np.random.randn(100)})
        specs = [FeatureSpec("feat_0", "BTC", "4h", 16, 0.05, 0.5)]
        result = deduplicate_by_correlation(df, specs, max_corr=0.8)
        assert len(result) == 1

    def test_missing_column_skipped(self):
        df = pl.DataFrame({"feat_0": np.random.randn(100)})
        specs = [
            FeatureSpec("feat_0", "BTC", "4h", 16, 0.05, 0.5),
            FeatureSpec("feat_missing", "BTC", "4h", 16, 0.03, 0.5),
        ]
        result = deduplicate_by_correlation(df, specs, max_corr=0.8)
        assert len(result) == 1
        assert result[0].name == "feat_0"


# ---------------------------------------------------------------------------
# combine_signals
# ---------------------------------------------------------------------------


class TestCombineSignals:
    def test_equal_method(self):
        np.random.seed(42)
        df = pl.DataFrame({
            "feat_0": np.arange(100, dtype=float),
            "feat_1": np.arange(100, dtype=float) * 2,
        })
        specs = [
            FeatureSpec("feat_0", "BTC", "4h", 16, 0.05, 0.5),
            FeatureSpec("feat_1", "BTC", "4h", 16, 0.03, 0.5),
        ]
        signal = combine_signals(df, specs, method="equal")
        assert len(signal) == 100
        # Rank-normalized to [-1, +1]
        valid = signal[np.isfinite(signal)]
        assert len(valid) > 0
        assert valid.min() >= -1.0 - 1e-10
        assert valid.max() <= 1.0 + 1e-10

    def test_ic_weighted_method(self):
        df = pl.DataFrame({
            "feat_0": np.arange(100, dtype=float),
            "feat_1": np.arange(100, dtype=float),
        })
        specs = [
            FeatureSpec("feat_0", "BTC", "4h", 16, 0.05, 0.5),
            FeatureSpec("feat_1", "BTC", "4h", 16, 0.03, 0.5),
        ]
        signal = combine_signals(df, specs, method="ic_weighted")
        valid = signal[np.isfinite(signal)]
        assert len(valid) > 0
        assert valid.min() >= -1.0 - 1e-10
        assert valid.max() <= 1.0 + 1e-10

    def test_empty_specs(self):
        df = pl.DataFrame({"x": np.random.randn(50)})
        signal = combine_signals(df, [], method="equal")
        assert np.all(signal == 0)

    def test_missing_columns_handled(self):
        df = pl.DataFrame({"feat_0": np.arange(50, dtype=float)})
        specs = [
            FeatureSpec("feat_0", "BTC", "4h", 16, 0.05, 0.5),
            FeatureSpec("feat_missing", "BTC", "4h", 16, 0.03, 0.5),
        ]
        signal = combine_signals(df, specs, method="equal")
        assert len(signal) == 50
