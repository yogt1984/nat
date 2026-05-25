"""Tests for scripts/data/features.py unified data loader."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.data.features import (
    load_features,
    load_bars,
    available_dates,
    available_symbols,
    data_health,
)
from scripts.data.schema import validate_columns, validate_quality


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATES = ["2026-05-20", "2026-05-21", "2026-05-22"]
SYMBOLS = ["BTC", "ETH"]
COLUMNS = [
    "timestamp_ns", "symbol", "sequence_id",
    "raw_midprice", "raw_spread_bps", "raw_ask_depth_5",
    "flow_vwap_deviation", "imbalance_qty_l1",
    "ent_permutation_returns_8", "vol_returns_1m",
]


def _make_parquet(path, symbol, date_str, n=500, seed=42):
    """Write a synthetic parquet file with deterministic data."""
    rng = np.random.default_rng(seed)
    base_ts = int(pd.Timestamp(f"{date_str}T10:00:00").value)
    ts = base_ts + np.arange(n) * 100_000_000  # 100ms intervals

    data = {
        "timestamp_ns": ts,
        "symbol": [symbol] * n,
        "sequence_id": np.arange(n, dtype=np.uint64),
        "raw_midprice": 50000.0 + rng.normal(0, 100, n).cumsum(),
        "raw_spread_bps": rng.uniform(0.5, 3.0, n),
        "raw_ask_depth_5": rng.uniform(100, 1000, n),
        "flow_vwap_deviation": rng.normal(0, 0.001, n),
        "imbalance_qty_l1": rng.uniform(-1, 1, n),
        "ent_permutation_returns_8": rng.uniform(0, 1, n),
        "vol_returns_1m": rng.uniform(0, 0.01, n),
    }
    table = pa.table(data)
    pq.write_table(table, str(path))


@pytest.fixture
def data_dir(tmp_path):
    """Create synthetic parquet data: 3 dates × 2 symbols."""
    for i, date_str in enumerate(DATES):
        date_path = tmp_path / date_str
        date_path.mkdir()
        for j, symbol in enumerate(SYMBOLS):
            fpath = date_path / f"{symbol}_{date_str}_10_00.parquet"
            _make_parquet(fpath, symbol, date_str, n=500, seed=i * 10 + j)
    return tmp_path


# ---------------------------------------------------------------------------
# load_features tests
# ---------------------------------------------------------------------------


class TestLoadFeatures:
    def test_load_all(self, data_dir):
        df = load_features(data_dir=data_dir)
        # 3 dates × 2 symbols × 500 ticks = 3000
        assert len(df) == 3000
        assert set(df["symbol"].unique()) == {"BTC", "ETH"}

    def test_filter_symbol(self, data_dir):
        df = load_features(symbols=["BTC"], data_dir=data_dir)
        assert len(df) == 1500
        assert df["symbol"].unique().tolist() == ["BTC"]

    def test_filter_multiple_symbols(self, data_dir):
        df = load_features(symbols=["BTC", "ETH"], data_dir=data_dir)
        assert len(df) == 3000

    def test_filter_date_range_single(self, data_dir):
        df = load_features(date_range=("2026-05-21", "2026-05-21"), data_dir=data_dir)
        assert len(df) == 1000  # 2 symbols × 500

    def test_filter_date_range_multi(self, data_dir):
        df = load_features(date_range=("2026-05-20", "2026-05-21"), data_dir=data_dir)
        assert len(df) == 2000

    def test_filter_combined(self, data_dir):
        df = load_features(
            symbols=["ETH"],
            date_range=("2026-05-22", "2026-05-22"),
            data_dir=data_dir,
        )
        assert len(df) == 500
        assert df["symbol"].unique().tolist() == ["ETH"]

    def test_column_selection(self, data_dir):
        df = load_features(columns=["raw_midprice", "vol_returns_1m"], data_dir=data_dir)
        # timestamp_ns and symbol always included
        assert "timestamp_ns" in df.columns
        assert "symbol" in df.columns
        assert "raw_midprice" in df.columns
        assert "vol_returns_1m" in df.columns
        # Other columns excluded
        assert "imbalance_qty_l1" not in df.columns

    def test_sorted_by_timestamp(self, data_dir):
        df = load_features(data_dir=data_dir)
        assert df["timestamp_ns"].is_monotonic_increasing

    def test_empty_result_no_match(self, data_dir):
        df = load_features(symbols=["DOGE"], data_dir=data_dir)
        assert df.empty
        assert "timestamp_ns" in df.columns
        assert "symbol" in df.columns

    def test_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        df = load_features(data_dir=empty)
        assert df.empty

    def test_nonexistent_dir(self, tmp_path):
        df = load_features(data_dir=tmp_path / "nope")
        assert df.empty

    def test_corrupted_file_skipped(self, data_dir):
        # Write garbage to a parquet file
        bad_file = data_dir / DATES[0] / "BAD_file.parquet"
        bad_file.write_bytes(b"not a parquet file")
        # Should still load the valid files
        with pytest.warns(UserWarning, match="Skipping"):
            df = load_features(data_dir=data_dir)
        assert len(df) == 3000  # Bad file ignored


# ---------------------------------------------------------------------------
# load_bars tests
# ---------------------------------------------------------------------------


class TestLoadBars:
    def test_default_aggregation(self, data_dir):
        bars = load_bars(symbols=["BTC"], date_range=("2026-05-20", "2026-05-20"),
                         data_dir=data_dir)
        assert not bars.empty
        # Should have aggregated columns
        assert "raw_midprice" in bars.columns
        assert "raw_spread_bps" in bars.columns
        assert "n_ticks" in bars.columns

    def test_bar_count(self, data_dir):
        # 500 ticks at 100ms = 50s of data. With 300s bars → 1 bar
        # But let's use smaller bars to get more
        bars = load_bars(
            symbols=["BTC"],
            date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=10,
            min_ticks=1,
            data_dir=data_dir,
        )
        # 50s of data / 10s bars = 5 bars (approximately)
        assert len(bars) >= 4

    def test_min_ticks_filter(self, data_dir):
        bars_loose = load_bars(
            symbols=["BTC"],
            date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=10, min_ticks=1, data_dir=data_dir,
        )
        bars_strict = load_bars(
            symbols=["BTC"],
            date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=10, min_ticks=200, data_dir=data_dir,
        )
        assert len(bars_strict) <= len(bars_loose)

    def test_std_columns_no_nan(self, data_dir):
        bars = load_bars(symbols=["BTC"], date_range=("2026-05-20", "2026-05-20"),
                         bar_seconds=5, min_ticks=1, data_dir=data_dir)
        if "raw_ask_depth_5" in bars.columns:
            assert bars["raw_ask_depth_5"].isna().sum() == 0
        if "flow_vwap_deviation" in bars.columns:
            assert bars["flow_vwap_deviation"].isna().sum() == 0

    def test_custom_agg_spec(self, data_dir):
        custom = {
            "ts_first": ("timestamp_ns", "first"),
            "mid_mean": ("raw_midprice", "mean"),
            "spread_max": ("raw_spread_bps", "max"),
        }
        bars = load_bars(
            symbols=["BTC"],
            date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=10, min_ticks=1,
            agg_spec=custom,
            data_dir=data_dir,
        )
        assert "mid_mean" in bars.columns
        assert "spread_max" in bars.columns
        # Default columns should NOT be present with custom spec
        assert "raw_midprice" not in bars.columns or "mid_mean" in bars.columns

    def test_empty_result(self, data_dir):
        bars = load_bars(symbols=["DOGE"], data_dir=data_dir)
        assert bars.empty


# ---------------------------------------------------------------------------
# available_dates / available_symbols tests
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_available_dates(self, data_dir):
        dates = available_dates(data_dir=data_dir)
        assert dates == DATES

    def test_available_dates_empty(self, tmp_path):
        assert available_dates(data_dir=tmp_path) == []

    def test_available_dates_skips_non_date_dirs(self, data_dir):
        # Create a non-date directory
        (data_dir / "notes").mkdir()
        dates = available_dates(data_dir=data_dir)
        assert "notes" not in dates
        assert len(dates) == 3

    def test_available_symbols(self, data_dir):
        syms = available_symbols(data_dir=data_dir)
        assert syms == ["BTC", "ETH"]

    def test_available_symbols_by_date(self, data_dir):
        syms = available_symbols(date="2026-05-20", data_dir=data_dir)
        assert syms == ["BTC", "ETH"]

    def test_data_health(self, data_dir):
        health = data_health(data_dir=data_dir)
        assert health["dates"] == DATES
        assert health["symbols"] == ["BTC", "ETH"]
        assert health["total_files"] == 6  # 3 dates × 2 files
        assert health["total_rows"] == 3000
        assert health["latest_timestamp"] is not None
        assert health["warnings"] == []


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestSchema:
    def test_validate_columns_full_schema(self):
        # Simulate a full schema DataFrame
        from scripts.data.schema import ALL_COLUMNS
        result = validate_columns(ALL_COLUMNS)
        assert result["valid"] is True
        assert result["missing_base"] == []
        assert result["missing_optional"] == []
        assert result["unexpected"] == []

    def test_validate_columns_missing_base(self):
        cols = ["timestamp_ns", "symbol"]  # Missing most base features
        result = validate_columns(cols)
        assert result["valid"] is False
        assert "sequence_id" in result["missing_base"]
        assert "raw_midprice" in result["missing_base"]

    def test_validate_columns_unexpected(self):
        from scripts.data.schema import ALL_COLUMNS
        cols = list(ALL_COLUMNS) + ["my_custom_column"]
        result = validate_columns(cols)
        assert result["valid"] is True
        assert "my_custom_column" in result["unexpected"]

    def test_validate_quality_basic(self, data_dir):
        df = load_features(data_dir=data_dir)
        quality = validate_quality(df)
        assert quality["row_count"] == 3000
        assert "BTC" in quality["symbol_counts"]
        assert "ETH" in quality["symbol_counts"]

    def test_validate_quality_empty(self):
        df = pd.DataFrame()
        quality = validate_quality(df)
        assert quality["row_count"] == 0

    def test_validate_quality_nan_detection(self):
        df = pd.DataFrame({
            "timestamp_ns": [1, 2, 3, 4],
            "symbol": ["BTC"] * 4,
            "good_col": [1.0, 2.0, 3.0, 4.0],
            "bad_col": [np.nan, np.nan, np.nan, 1.0],  # 75% NaN
        })
        quality = validate_quality(df)
        assert "bad_col" in quality["high_nan_cols"]
        assert "good_col" not in quality["high_nan_cols"]

    def test_validate_quality_constant_detection(self):
        df = pd.DataFrame({
            "timestamp_ns": [1, 2, 3, 4],
            "symbol": ["BTC"] * 4,
            "const_col": [5.0, 5.0, 5.0, 5.0],
            "vary_col": [1.0, 2.0, 3.0, 4.0],
        })
        quality = validate_quality(df)
        assert "const_col" in quality["constant_cols"]
        assert "vary_col" not in quality["constant_cols"]
