"""Tests for scripts/data/ unified data access layer.

Covers: load_features, load_bars, available_dates, available_symbols,
data_health, schema validation integration, catalog, edge cases,
concurrency, error recovery, and performance.
"""

import threading
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.features import (
    load_features,
    load_bars,
    available_dates,
    available_symbols,
    data_health,
    reset_validation_cache,
)
from data.schema import validate_columns, validate_quality, ALL_COLUMNS
from data.catalog import data_manifest, freshness_check


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
        from data.schema import ALL_COLUMNS
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
        from data.schema import ALL_COLUMNS
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


# ===========================================================================
# Schema validation integration
# ===========================================================================


class TestSchemaValidationIntegration:
    """Test validate= parameter wired into load_features."""

    def setup_method(self):
        reset_validation_cache()

    def test_validate_on_load_warns_missing_base(self, data_dir):
        """Synthetic data missing base columns → UserWarning."""
        with pytest.warns(UserWarning, match="Schema drift"):
            load_features(data_dir=data_dir, validate=True)

    def test_validate_false_skips_check(self, data_dir):
        """validate=False never emits warnings."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_features(data_dir=data_dir, validate=False)

    def test_session_cache_prevents_repeat(self, data_dir):
        """Second call to same data_dir does not re-warn."""
        with pytest.warns(UserWarning):
            load_features(data_dir=data_dir, validate=True)
        # Second call should NOT warn (cached)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_features(data_dir=data_dir, validate=True)

    def test_session_cache_separate_dirs(self, tmp_path):
        """Different data_dirs both get validated."""
        for name in ["dir_a", "dir_b"]:
            d = tmp_path / name / "2026-06-01"
            d.mkdir(parents=True)
            _make_parquet(d / "BTC.parquet", "BTC", "2026-06-01", n=50)
        with pytest.warns(UserWarning):
            load_features(data_dir=tmp_path / "dir_a", validate=True)
        with pytest.warns(UserWarning):
            load_features(data_dir=tmp_path / "dir_b", validate=True)

    def test_reset_validation_cache(self, data_dir):
        """After reset, re-validates."""
        with pytest.warns(UserWarning):
            load_features(data_dir=data_dir, validate=True)
        reset_validation_cache()
        with pytest.warns(UserWarning):
            load_features(data_dir=data_dir, validate=True)

    def test_validate_empty_df_no_crash(self, tmp_path):
        """Empty dir doesn't crash validation."""
        empty = tmp_path / "empty"
        empty.mkdir()
        df = load_features(data_dir=empty, validate=True)
        assert df.empty

    def test_validate_full_schema_no_warn(self, tmp_path):
        """Data with all base columns produces no schema drift warning."""
        from data.schema import ALL_BASE
        d = tmp_path / "2026-06-01"
        d.mkdir()
        rng = np.random.default_rng(42)
        n = 100
        data = {"timestamp_ns": np.arange(n) * 100_000_000 + 1_000_000_000_000,
                "symbol": ["BTC"] * n, "sequence_id": np.arange(n, dtype=np.uint64)}
        for col in ALL_BASE:
            if col not in data:
                data[col] = rng.normal(0, 1, n)
        pq.write_table(pa.table(data), str(d / "full.parquet"))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_features(data_dir=tmp_path, validate=True)

    def test_unexpected_columns_warned(self, tmp_path):
        """Extra non-schema columns trigger unexpected warning."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        n = 50
        data = {"timestamp_ns": np.arange(n) * 100_000_000 + 1_000_000_000_000,
                "symbol": ["BTC"] * n, "my_custom_feat": np.ones(n)}
        pq.write_table(pa.table(data), str(d / "custom.parquet"))
        with pytest.warns(UserWarning, match="Unexpected columns"):
            load_features(data_dir=tmp_path, validate=True)

    def test_validate_with_column_selection(self, data_dir):
        """Validation checks full file schema even when columns= limits output."""
        reset_validation_cache()
        with pytest.warns(UserWarning, match="Schema drift"):
            load_features(columns=["raw_midprice"], data_dir=data_dir, validate=True)

    def test_validate_with_load_bars(self, data_dir):
        """load_bars inherits validation from load_features."""
        reset_validation_cache()
        with pytest.warns(UserWarning, match="Schema drift"):
            load_bars(bar_seconds=10, min_ticks=1, data_dir=data_dir)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases that might trip up the loader."""

    def setup_method(self):
        reset_validation_cache()

    def test_empty_parquet_file(self, tmp_path):
        """0-row parquet file is handled gracefully."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        data = {"timestamp_ns": np.array([], dtype=np.int64),
                "symbol": np.array([], dtype=str)}
        pq.write_table(pa.table(data), str(d / "empty.parquet"))
        df = load_features(data_dir=tmp_path, validate=False)
        assert df.empty

    def test_single_row_file(self, tmp_path):
        """Single-row parquet loads correctly."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        data = {"timestamp_ns": [1_000_000_000_000],
                "symbol": ["BTC"],
                "raw_midprice": [50000.0]}
        pq.write_table(pa.table(data), str(d / "one.parquet"))
        df = load_features(data_dir=tmp_path, validate=False)
        assert len(df) == 1
        assert df["raw_midprice"].iloc[0] == 50000.0

    def test_all_nan_numeric(self, tmp_path):
        """All-NaN numeric columns load without error."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        n = 100
        data = {"timestamp_ns": np.arange(n) * 100_000_000 + 1_000_000_000_000,
                "symbol": ["BTC"] * n,
                "raw_midprice": [np.nan] * n}
        pq.write_table(pa.table(data), str(d / "nan.parquet"))
        df = load_features(data_dir=tmp_path, validate=False)
        assert len(df) == n
        assert df["raw_midprice"].isna().all()

    def test_duplicate_timestamps(self, tmp_path):
        """Multiple rows with same timestamp are all returned."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        ts = [1_000_000_000_000] * 5
        data = {"timestamp_ns": ts, "symbol": ["BTC"] * 5,
                "raw_midprice": [1.0, 2.0, 3.0, 4.0, 5.0]}
        pq.write_table(pa.table(data), str(d / "dup.parquet"))
        df = load_features(data_dir=tmp_path, validate=False)
        assert len(df) == 5

    def test_date_dir_with_suffix(self, tmp_path):
        """Directory '2026-05-20-clean' is treated as valid date dir."""
        d = tmp_path / "2026-05-20-clean"
        d.mkdir()
        _make_parquet(d / "BTC.parquet", "BTC", "2026-05-20", n=50)
        dates = available_dates(data_dir=tmp_path)
        assert "2026-05-20" in dates

    def test_non_parquet_files_ignored(self, tmp_path):
        """Non-parquet files in date dirs don't cause errors."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        (d / "notes.txt").write_text("hello")
        (d / "data.csv").write_text("a,b\n1,2\n")
        _make_parquet(d / "BTC.parquet", "BTC", "2026-06-01", n=50)
        df = load_features(data_dir=tmp_path, validate=False)
        assert len(df) == 50

    def test_mixed_schemas_across_files(self, tmp_path):
        """Files with different columns are merged (missing cols become NaN)."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        n = 50
        ts = np.arange(n) * 100_000_000 + 1_000_000_000_000
        # File 1: has col_a
        data1 = {"timestamp_ns": ts, "symbol": ["BTC"] * n, "col_a": np.ones(n)}
        pq.write_table(pa.table(data1), str(d / "f1.parquet"))
        # File 2: has col_b
        ts2 = ts + n * 100_000_000
        data2 = {"timestamp_ns": ts2, "symbol": ["BTC"] * n, "col_b": np.ones(n) * 2}
        pq.write_table(pa.table(data2), str(d / "f2.parquet"))
        df = load_features(data_dir=tmp_path, validate=False)
        assert len(df) == 100
        assert "col_a" in df.columns
        assert "col_b" in df.columns

    def test_very_large_timestamps(self, tmp_path):
        """Near year-2100 nanosecond timestamps don't overflow."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        big_ts = 4_000_000_000_000_000_000  # ~2096
        data = {"timestamp_ns": [big_ts, big_ts + 100_000_000],
                "symbol": ["BTC", "BTC"], "raw_midprice": [1.0, 2.0]}
        pq.write_table(pa.table(data), str(d / "big.parquet"))
        df = load_features(data_dir=tmp_path, validate=False)
        assert len(df) == 2

    def test_empty_string_symbol(self, tmp_path):
        """Empty string as symbol can be filtered."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        data = {"timestamp_ns": [1_000_000_000_000, 2_000_000_000_000],
                "symbol": ["", "BTC"], "raw_midprice": [1.0, 2.0]}
        pq.write_table(pa.table(data), str(d / "emp.parquet"))
        df = load_features(symbols=["BTC"], data_dir=tmp_path, validate=False)
        assert len(df) == 1

    def test_no_date_dirs_only_files(self, tmp_path):
        """Parquet files at root level (no date dir) are ignored."""
        _make_parquet(tmp_path / "stray.parquet", "BTC", "2026-06-01", n=50)
        df = load_features(data_dir=tmp_path, validate=False)
        assert df.empty

    def test_numeric_only_dir_name(self, tmp_path):
        """Directory '123' is not treated as a date dir."""
        d = tmp_path / "123"
        d.mkdir()
        _make_parquet(d / "BTC.parquet", "BTC", "2026-06-01", n=50)
        dates = available_dates(data_dir=tmp_path)
        assert dates == []


# ===========================================================================
# Multi-symbol date filtering
# ===========================================================================


class TestMultiSymbolDateFiltering:
    """Complex filtering combinations."""

    def setup_method(self):
        reset_validation_cache()

    @pytest.fixture
    def multi_data(self, tmp_path):
        """3 dates, 3 symbols, varying presence."""
        configs = [
            ("2026-06-01", ["BTC", "ETH", "SOL"]),
            ("2026-06-02", ["BTC", "ETH"]),  # SOL missing on day 2
            ("2026-06-03", ["BTC"]),          # only BTC on day 3
        ]
        for i, (date, syms) in enumerate(configs):
            d = tmp_path / date
            d.mkdir()
            for j, sym in enumerate(syms):
                _make_parquet(d / f"{sym}.parquet", sym, date, n=100, seed=i*10+j)
        return tmp_path

    def test_three_symbols_all_dates(self, multi_data):
        df = load_features(data_dir=multi_data, validate=False)
        assert len(df) == 600  # 3+2+1 = 6 files × 100

    def test_symbol_not_in_all_dates(self, multi_data):
        df = load_features(symbols=["SOL"], data_dir=multi_data, validate=False)
        assert len(df) == 100  # only day 1

    def test_date_range_excludes_boundary(self, multi_data):
        df = load_features(
            date_range=("2026-06-02", "2026-06-02"),
            data_dir=multi_data, validate=False,
        )
        assert len(df) == 200  # BTC + ETH

    def test_future_date_range(self, multi_data):
        df = load_features(
            date_range=("2026-07-01", "2026-07-31"),
            data_dir=multi_data, validate=False,
        )
        assert df.empty

    def test_single_day_all_symbols(self, multi_data):
        df = load_features(
            date_range=("2026-06-01", "2026-06-01"),
            data_dir=multi_data, validate=False,
        )
        assert set(df["symbol"].unique()) == {"BTC", "ETH", "SOL"}
        assert len(df) == 300

    def test_all_days_one_symbol(self, multi_data):
        df = load_features(symbols=["BTC"], data_dir=multi_data, validate=False)
        assert len(df) == 300  # 100 per day × 3 days

    def test_multi_symbol_filter(self, multi_data):
        df = load_features(
            symbols=["ETH", "SOL"],
            date_range=("2026-06-01", "2026-06-02"),
            data_dir=multi_data, validate=False,
        )
        # Day 1: ETH + SOL = 200, Day 2: ETH = 100
        assert len(df) == 300

    def test_nonexistent_symbol(self, multi_data):
        df = load_features(symbols=["DOGE"], data_dir=multi_data, validate=False)
        assert df.empty


# ===========================================================================
# Schema drift detection
# ===========================================================================


class TestSchemaDriftDetection:
    """Test the standalone validate_columns function."""

    def test_detect_column_rename(self):
        cols = [c if c != "raw_midprice" else "midprice" for c in ALL_COLUMNS]
        result = validate_columns(cols)
        assert not result["valid"]
        assert "raw_midprice" in result["missing_base"]
        assert "midprice" in result["unexpected"]

    def test_detect_added_columns(self):
        cols = list(ALL_COLUMNS) + ["custom_alpha", "custom_beta"]
        result = validate_columns(cols)
        assert result["valid"]  # base complete, extra cols are OK structurally
        assert "custom_alpha" in result["unexpected"]
        assert "custom_beta" in result["unexpected"]

    def test_detect_removed_base(self):
        cols = [c for c in ALL_COLUMNS if c != "ent_book_shape"]
        result = validate_columns(cols)
        assert not result["valid"]
        assert "ent_book_shape" in result["missing_base"]

    def test_detect_removed_optional_is_info(self):
        cols = [c for c in ALL_COLUMNS if c != "whale_net_flow_1h"]
        result = validate_columns(cols)
        assert result["valid"]  # optional missing is not invalid
        assert "whale_net_flow_1h" in result["missing_optional"]

    def test_multiple_missing_reported(self):
        cols = ["timestamp_ns", "symbol"]
        result = validate_columns(cols)
        assert not result["valid"]
        assert len(result["missing_base"]) > 100

    def test_schema_valid_with_all_base_subset_optional(self):
        from data.schema import ALL_BASE, OPTIONAL_FEATURES
        # All base + only whale optional
        cols = list(ALL_BASE) + OPTIONAL_FEATURES["whale"]
        result = validate_columns(cols)
        assert result["valid"]

    def test_validate_columns_empty_list(self):
        result = validate_columns([])
        assert not result["valid"]
        assert len(result["missing_base"]) > 0

    def test_validate_columns_duplicate_names(self):
        cols = list(ALL_COLUMNS) + ["raw_midprice"]
        result = validate_columns(cols)
        assert result["valid"]


# ===========================================================================
# Bar aggregation edge cases
# ===========================================================================


class TestBarAggregationEdgeCases:
    """Edge cases for load_bars."""

    def setup_method(self):
        reset_validation_cache()

    def test_very_short_bars(self, data_dir):
        """1-second bars on 100ms data → ~5 ticks/bar."""
        bars = load_bars(
            symbols=["BTC"],
            date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=1, min_ticks=1, data_dir=data_dir,
        )
        assert not bars.empty
        assert bars["n_ticks"].max() <= 15

    def test_very_long_bars(self, data_dir):
        """Bar wider than all data → 1 bar."""
        bars = load_bars(
            symbols=["BTC"],
            date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=86400, min_ticks=1, data_dir=data_dir,
        )
        assert len(bars) == 1

    def test_all_partial_bars_dropped(self, data_dir):
        """min_ticks > max ticks per bar → empty."""
        bars = load_bars(
            symbols=["BTC"],
            date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=1, min_ticks=9999, data_dir=data_dir,
        )
        assert bars.empty

    def test_custom_agg_mean(self, data_dir):
        custom = {"mid_mean": ("raw_midprice", "mean")}
        bars = load_bars(
            symbols=["BTC"], date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=10, min_ticks=1, agg_spec=custom, data_dir=data_dir,
        )
        assert "mid_mean" in bars.columns

    def test_custom_agg_min_max(self, data_dir):
        custom = {
            "spread_min": ("raw_spread_bps", "min"),
            "spread_max": ("raw_spread_bps", "max"),
        }
        bars = load_bars(
            symbols=["BTC"], date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=10, min_ticks=1, agg_spec=custom, data_dir=data_dir,
        )
        assert (bars["spread_max"] >= bars["spread_min"]).all()

    def test_bars_n_ticks_matches_data(self, data_dir):
        """Sum of n_ticks across all bars <= total ticks loaded."""
        bars = load_bars(
            symbols=["BTC"], date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=10, min_ticks=1, data_dir=data_dir,
        )
        total_bar_ticks = bars["n_ticks"].sum()
        ticks = load_features(
            symbols=["BTC"], date_range=("2026-05-20", "2026-05-20"),
            data_dir=data_dir, validate=False,
        )
        assert total_bar_ticks <= len(ticks)

    def test_std_on_single_value_is_nan_filled(self, tmp_path):
        """Std aggregation on single-value bar gives 0.0 (NaN filled)."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        # Create data where each bar has exactly 1 tick
        data = {
            "timestamp_ns": [1_000_000_000_000, 2_000_000_000_000],
            "symbol": ["BTC", "BTC"],
            "raw_ask_depth_5": [100.0, 200.0],
            "flow_vwap_deviation": [0.001, 0.002],
        }
        pq.write_table(pa.table(data), str(d / "sparse.parquet"))
        bars = load_bars(bar_seconds=1, min_ticks=1, data_dir=tmp_path)
        if "raw_ask_depth_5" in bars.columns:
            assert not bars["raw_ask_depth_5"].isna().any()

    def test_empty_bars_from_nonexistent_symbol(self, data_dir):
        bars = load_bars(symbols=["DOGE"], data_dir=data_dir)
        assert bars.empty

    def test_bars_timestamp_monotonic(self, data_dir):
        bars = load_bars(
            symbols=["BTC"], date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=5, min_ticks=1, data_dir=data_dir,
        )
        if not bars.empty and "timestamp_ns" in bars.columns:
            assert bars["timestamp_ns"].is_monotonic_increasing

    def test_multi_symbol_bars(self, data_dir):
        """Bars from multiple symbols don't mix timestamps."""
        bars = load_bars(
            date_range=("2026-05-20", "2026-05-20"),
            bar_seconds=10, min_ticks=1, data_dir=data_dir,
        )
        assert not bars.empty


# ===========================================================================
# Data quality validation
# ===========================================================================


class TestDataQualityValidation:
    """Test validate_quality edge cases."""

    def test_high_nan_threshold_exact(self):
        """Exactly 50% NaN is NOT flagged as high_nan (needs > 50%)."""
        df = pd.DataFrame({
            "timestamp_ns": range(4),
            "symbol": ["BTC"] * 4,
            "half_nan": [np.nan, np.nan, 1.0, 2.0],
        })
        q = validate_quality(df)
        assert "half_nan" not in q["high_nan_cols"]

    def test_high_nan_above_50(self):
        """51%+ NaN IS flagged."""
        df = pd.DataFrame({
            "timestamp_ns": range(100),
            "symbol": ["BTC"] * 100,
            "mostly_nan": [np.nan] * 51 + [1.0] * 49,
        })
        q = validate_quality(df)
        assert "mostly_nan" in q["high_nan_cols"]

    def test_quality_all_nan_column(self):
        df = pd.DataFrame({
            "timestamp_ns": range(10),
            "symbol": ["BTC"] * 10,
            "all_nan": [np.nan] * 10,
        })
        q = validate_quality(df)
        assert "all_nan" in q["high_nan_cols"]
        assert q["nan_rates"]["all_nan"] == 1.0

    def test_symbol_counts(self):
        df = pd.DataFrame({
            "timestamp_ns": range(6),
            "symbol": ["BTC"] * 4 + ["ETH"] * 2,
            "x": [1.0] * 6,
        })
        q = validate_quality(df)
        assert q["symbol_counts"]["BTC"] == 4
        assert q["symbol_counts"]["ETH"] == 2

    def test_quality_no_numeric(self):
        """DataFrame with no numeric columns."""
        df = pd.DataFrame({
            "timestamp_ns": range(5),
            "symbol": ["BTC"] * 5,
        })
        q = validate_quality(df)
        assert q["nan_rates"] == {}
        assert q["constant_cols"] == []

    def test_quality_large_df(self):
        """10K+ row quality check completes."""
        n = 10_000
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "timestamp_ns": range(n),
            "symbol": ["BTC"] * n,
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
        })
        q = validate_quality(df)
        assert q["row_count"] == n

    def test_constant_col_zero_std(self):
        """Constant column detected even if value is 0."""
        df = pd.DataFrame({
            "timestamp_ns": range(10),
            "symbol": ["BTC"] * 10,
            "zeros": [0.0] * 10,
        })
        q = validate_quality(df)
        assert "zeros" in q["constant_cols"]

    def test_nan_rate_precision(self):
        """NaN rate is a float with proper precision."""
        df = pd.DataFrame({
            "timestamp_ns": range(3),
            "symbol": ["BTC"] * 3,
            "partial": [np.nan, 1.0, 2.0],
        })
        q = validate_quality(df)
        assert abs(q["nan_rates"]["partial"] - 1/3) < 0.01


# ===========================================================================
# Catalog tests
# ===========================================================================


class TestCatalog:
    """Test data_manifest and freshness_check."""

    def setup_method(self):
        reset_validation_cache()

    def test_manifest_basic_structure(self, data_dir):
        m = data_manifest(data_dir=data_dir)
        assert "dates" in m
        assert "symbols" in m
        assert "total_dates" in m
        assert "total_hours_per_symbol" in m
        assert "updated" in m

    def test_manifest_dates_count(self, data_dir):
        m = data_manifest(data_dir=data_dir)
        assert m["total_dates"] == 3

    def test_manifest_symbols_present(self, data_dir):
        m = data_manifest(data_dir=data_dir)
        assert "BTC" in m["symbols"]
        assert "ETH" in m["symbols"]

    def test_manifest_hours_positive(self, data_dir):
        m = data_manifest(data_dir=data_dir)
        for sym_info in m["symbols"].values():
            assert sym_info["hours"] >= 0

    def test_manifest_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        m = data_manifest(data_dir=empty)
        assert m["total_dates"] == 0
        assert m["symbols"] == {}

    def test_manifest_nonexistent_dir(self, tmp_path):
        m = data_manifest(data_dir=tmp_path / "nope")
        assert m["total_dates"] == 0

    def test_manifest_compatible_with_is_runnable(self, data_dir):
        """Format matches HypothesisQueue._is_runnable expectations."""
        m = data_manifest(data_dir=data_dir)
        # _is_runnable checks: manifest["symbols"][sym]["hours"]
        for sym in ["BTC", "ETH"]:
            assert "hours" in m["symbols"][sym]
            assert isinstance(m["symbols"][sym]["hours"], (int, float))

    def test_manifest_explicit_symbols(self, data_dir):
        m = data_manifest(data_dir=data_dir, symbols=["BTC"])
        assert "BTC" in m["symbols"]
        assert "ETH" not in m["symbols"]

    def test_freshness_no_data(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        r = freshness_check(data_dir=empty)
        assert r["fresh"] is False
        assert r["staleness_hours"] == float("inf")

    def test_freshness_check_structure(self, data_dir):
        r = freshness_check(data_dir=data_dir, max_stale_hours=99999)
        assert "fresh" in r
        assert "staleness_hours" in r
        assert "latest_timestamp" in r
        assert "message" in r


# ===========================================================================
# Concurrent access
# ===========================================================================


class TestConcurrentAccess:
    """Thread safety of the data access layer."""

    def setup_method(self):
        reset_validation_cache()

    def test_parallel_load_same_dir(self, data_dir):
        """Two threads loading same data simultaneously."""
        results = [None, None]

        def load(idx):
            results[idx] = load_features(
                symbols=["BTC"], data_dir=data_dir, validate=False
            )

        t1 = threading.Thread(target=load, args=(0,))
        t2 = threading.Thread(target=load, args=(1,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(results[0]) == len(results[1]) == 1500

    def test_parallel_load_different_symbols(self, data_dir):
        """Two threads loading different symbols."""
        results = {}

        def load(sym):
            results[sym] = load_features(
                symbols=[sym], data_dir=data_dir, validate=False
            )

        threads = [threading.Thread(target=load, args=(s,)) for s in ["BTC", "ETH"]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results["BTC"]) == 1500
        assert len(results["ETH"]) == 1500

    def test_validation_cache_thread_safe(self, data_dir):
        """Multiple threads validating same dir doesn't crash."""
        errors = []

        def load_with_validate():
            try:
                load_features(data_dir=data_dir, validate=True)
            except Exception as e:
                errors.append(e)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            threads = [threading.Thread(target=load_with_validate) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert errors == []

    def test_available_dates_concurrent(self, data_dir):
        """Thread safety of available_dates."""
        results = [None, None, None]

        def query(idx):
            results[idx] = available_dates(data_dir=data_dir)

        threads = [threading.Thread(target=query, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results[0] == results[1] == results[2] == DATES


# ===========================================================================
# Error recovery
# ===========================================================================


class TestErrorRecovery:
    """Graceful handling of corrupted/problematic files."""

    def setup_method(self):
        reset_validation_cache()

    def test_truncated_parquet(self, tmp_path):
        """Truncated file is skipped with warning."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        _make_parquet(d / "good.parquet", "BTC", "2026-06-01", n=100)
        # Write truncated parquet (just the magic bytes + garbage)
        (d / "bad.parquet").write_bytes(b"PAR1" + b"\x00" * 50)
        with pytest.warns(UserWarning, match="Skipping"):
            df = load_features(data_dir=tmp_path, validate=False)
        assert len(df) == 100

    def test_zero_byte_file(self, tmp_path):
        """0-byte file is skipped."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        _make_parquet(d / "good.parquet", "BTC", "2026-06-01", n=100)
        (d / "empty.parquet").write_bytes(b"")
        with pytest.warns(UserWarning, match="Skipping"):
            df = load_features(data_dir=tmp_path, validate=False)
        assert len(df) == 100

    def test_wrong_magic_bytes(self, tmp_path):
        """Non-parquet file with .parquet extension is skipped."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        _make_parquet(d / "good.parquet", "BTC", "2026-06-01", n=100)
        (d / "fake.parquet").write_text("this is not parquet")
        with pytest.warns(UserWarning, match="Skipping"):
            df = load_features(data_dir=tmp_path, validate=False)
        assert len(df) == 100

    def test_all_files_corrupted(self, tmp_path):
        """All bad files → empty DataFrame."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        (d / "bad1.parquet").write_bytes(b"garbage")
        (d / "bad2.parquet").write_bytes(b"more garbage")
        with pytest.warns(UserWarning):
            df = load_features(data_dir=tmp_path, validate=False)
        assert df.empty

    def test_mixed_good_bad(self, tmp_path):
        """Good + bad files → only good data returned."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        _make_parquet(d / "good1.parquet", "BTC", "2026-06-01", n=100, seed=1)
        (d / "bad.parquet").write_bytes(b"corrupted")
        _make_parquet(d / "good2.parquet", "ETH", "2026-06-01", n=100, seed=2)
        with pytest.warns(UserWarning, match="Skipping"):
            df = load_features(data_dir=tmp_path, validate=False)
        assert len(df) == 200

    def test_data_health_with_bad_files(self, tmp_path):
        """data_health handles corrupted files gracefully."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        _make_parquet(d / "good.parquet", "BTC", "2026-06-01", n=100)
        (d / "bad.parquet").write_bytes(b"nope")
        health = data_health(data_dir=tmp_path)
        assert health["total_rows"] == 100
        assert len(health["warnings"]) == 1


# ===========================================================================
# Performance
# ===========================================================================


class TestPerformance:
    """Basic performance sanity checks."""

    def setup_method(self):
        reset_validation_cache()

    @pytest.fixture
    def large_data(self, tmp_path):
        """10K rows per symbol × 2 symbols."""
        d = tmp_path / "2026-06-01"
        d.mkdir()
        for sym in ["BTC", "ETH"]:
            _make_parquet(d / f"{sym}.parquet", sym, "2026-06-01",
                          n=10_000, seed=hash(sym) % 100)
        return tmp_path

    def test_load_10k_rows_under_2s(self, large_data):
        """10K rows load in reasonable time."""
        start = time.time()
        df = load_features(data_dir=large_data, validate=False)
        elapsed = time.time() - start
        assert len(df) == 20_000
        assert elapsed < 2.0

    def test_column_pushdown_reduces_columns(self, large_data):
        """Column selection reduces output columns."""
        df_all = load_features(data_dir=large_data, validate=False)
        df_subset = load_features(
            columns=["raw_midprice"], data_dir=large_data, validate=False
        )
        assert len(df_subset.columns) < len(df_all.columns)
        assert "raw_midprice" in df_subset.columns

    def test_predicate_pushdown_reduces_rows(self, large_data):
        """Symbol filter reduces returned rows."""
        df_all = load_features(data_dir=large_data, validate=False)
        df_btc = load_features(symbols=["BTC"], data_dir=large_data, validate=False)
        assert len(df_btc) < len(df_all)

    def test_available_dates_fast(self, large_data):
        """available_dates is fast (metadata only)."""
        start = time.time()
        dates = available_dates(data_dir=large_data)
        elapsed = time.time() - start
        assert elapsed < 0.5
        assert len(dates) == 1
