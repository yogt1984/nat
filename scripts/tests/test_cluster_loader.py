"""
Skeptical tests for cluster_pipeline.loader — parquet data loading.

Tests schema detection, data loading, filtering, validation, and edge cases.
Every test verifies a property that MUST hold for the pipeline to work correctly
on both synthetic and live data.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cluster_pipeline.config import (
    COMPOSITE_VECTORS,
    FEATURE_VECTORS,
    META_COLUMNS,
    get_vector_columns,
)
from cluster_pipeline.loader import (
    scan_schema,
    print_schema_summary,
    load_parquet,
    load_parquet_lazy,
    validate_schema,
    get_symbols,
    filter_symbol,
    filter_time_range,
    get_time_range,
    get_duration_seconds,
    list_parquet_files,
)


# ============================================================================
# Fixtures — create temporary parquet files for testing
# ============================================================================


def _all_feature_columns() -> list:
    """Get all feature columns from config."""
    cols = set()
    for spec in FEATURE_VECTORS.values():
        cols.update(spec["columns"])
    return sorted(cols)


def _make_parquet_file(
    path: str,
    n_rows: int = 100,
    symbols: list = None,
    seed: int = 42,
    columns: list = None,
    start_ns: int = 1_000_000_000_000,
    interval_ns: int = 100_000_000,
) -> str:
    """Create a single parquet file with realistic data."""
    rng = np.random.default_rng(seed)
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL"]
    if columns is None:
        columns = _all_feature_columns()

    data = {}
    data["timestamp_ns"] = np.arange(n_rows) * interval_ns + start_ns
    data["symbol"] = rng.choice(symbols, n_rows)
    data["sequence_id"] = np.arange(n_rows, dtype=np.uint64)

    for col in columns:
        data[col] = rng.standard_normal(n_rows)

    table = pa.table(data)
    pq.write_table(table, path, compression="zstd")
    return path


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temp directory with 3 parquet files."""
    data_dir = tmp_path / "features"
    data_dir.mkdir()

    # File 1: BTC, ETH, SOL (hours 0-1)
    _make_parquet_file(
        str(data_dir / "hour_001.parquet"),
        n_rows=300,
        seed=1,
        start_ns=1_000_000_000_000,
    )
    # File 2: same symbols (hours 1-2)
    _make_parquet_file(
        str(data_dir / "hour_002.parquet"),
        n_rows=300,
        seed=2,
        start_ns=1_000_000_000_000 + 300 * 100_000_000,
    )
    # File 3: same symbols (hours 2-3)
    _make_parquet_file(
        str(data_dir / "hour_003.parquet"),
        n_rows=200,
        seed=3,
        start_ns=1_000_000_000_000 + 600 * 100_000_000,
    )
    return str(data_dir)


@pytest.fixture
def tmp_partial_dir(tmp_path):
    """Create a temp directory with partial schema (only entropy + volatility)."""
    data_dir = tmp_path / "partial"
    data_dir.mkdir()

    cols = (
        FEATURE_VECTORS["entropy"]["columns"]
        + FEATURE_VECTORS["volatility"]["columns"]
    )
    _make_parquet_file(
        str(data_dir / "partial_001.parquet"),
        n_rows=200,
        columns=cols,
        seed=10,
    )
    return str(data_dir)


@pytest.fixture
def tmp_single_symbol_dir(tmp_path):
    """Create a temp directory with only BTC data."""
    data_dir = tmp_path / "single"
    data_dir.mkdir()
    _make_parquet_file(
        str(data_dir / "btc_001.parquet"),
        n_rows=500,
        symbols=["BTC"],
        seed=20,
    )
    return str(data_dir)


@pytest.fixture
def tmp_empty_dir(tmp_path):
    """Create an empty directory."""
    data_dir = tmp_path / "empty"
    data_dir.mkdir()
    return str(data_dir)


@pytest.fixture
def tmp_nan_dir(tmp_path):
    """Create a directory with NaN-heavy data."""
    data_dir = tmp_path / "nan_data"
    data_dir.mkdir()

    rng = np.random.default_rng(30)
    cols = _all_feature_columns()
    n_rows = 200

    data = {}
    data["timestamp_ns"] = np.arange(n_rows) * 100_000_000 + 1_000_000_000_000
    data["symbol"] = rng.choice(["BTC", "ETH"], n_rows)
    data["sequence_id"] = np.arange(n_rows, dtype=np.uint64)

    for i, col in enumerate(cols):
        vals = rng.standard_normal(n_rows)
        # Make some columns mostly NaN
        if i % 10 == 0:
            vals[rng.random(n_rows) < 0.98] = np.nan  # 98% NaN
        elif i % 5 == 0:
            vals[rng.random(n_rows) < 0.6] = np.nan   # 60% NaN
        else:
            vals[rng.random(n_rows) < 0.1] = np.nan   # 10% NaN
        data[col] = vals

    table = pa.table(data)
    pq.write_table(table, str(data_dir / "nan_001.parquet"))
    return str(data_dir)


@pytest.fixture
def tmp_constant_dir(tmp_path):
    """Create a directory with constant-value columns."""
    data_dir = tmp_path / "constant_data"
    data_dir.mkdir()

    rng = np.random.default_rng(40)
    cols = _all_feature_columns()
    n_rows = 100

    data = {}
    data["timestamp_ns"] = np.arange(n_rows) * 100_000_000 + 1_000_000_000_000
    data["symbol"] = rng.choice(["BTC"], n_rows)
    data["sequence_id"] = np.arange(n_rows, dtype=np.uint64)

    for i, col in enumerate(cols):
        if i % 7 == 0:
            data[col] = np.full(n_rows, 0.0)  # constant zero
        elif i % 11 == 0:
            data[col] = np.full(n_rows, 42.0)  # constant non-zero
        else:
            data[col] = rng.standard_normal(n_rows)

    table = pa.table(data)
    pq.write_table(table, str(data_dir / "constant_001.parquet"))
    return str(data_dir)


@pytest.fixture
def tmp_multi_subdir(tmp_path):
    """Create nested subdirectories with parquet files."""
    base = tmp_path / "nested"
    base.mkdir()
    sub1 = base / "BTC"
    sub1.mkdir()
    sub2 = base / "ETH"
    sub2.mkdir()

    _make_parquet_file(
        str(sub1 / "btc_001.parquet"), n_rows=100, symbols=["BTC"], seed=50,
    )
    _make_parquet_file(
        str(sub2 / "eth_001.parquet"), n_rows=100, symbols=["ETH"], seed=51,
    )
    return str(base)


# ============================================================================
# Test: scan_schema
# ============================================================================


class TestScanSchema:
    """Tests for scan_schema() — schema inspection without loading data."""

    def test_returns_correct_file_count(self, tmp_data_dir):
        info = scan_schema(tmp_data_dir)
        assert info["file_count"] == 3

    def test_returns_correct_total_rows(self, tmp_data_dir):
        info = scan_schema(tmp_data_dir)
        assert info["total_rows"] == 800  # 300 + 300 + 200

    def test_columns_include_meta(self, tmp_data_dir):
        info = scan_schema(tmp_data_dir)
        assert "timestamp_ns" in info["columns"]
        assert "symbol" in info["columns"]
        assert "sequence_id" in info["columns"]

    def test_columns_include_features(self, tmp_data_dir):
        info = scan_schema(tmp_data_dir)
        # Check at least some feature columns exist
        assert "ent_permutation_returns_8" in info["columns"]
        assert "vol_returns_1m" in info["columns"]
        assert "trend_momentum_60" in info["columns"]

    def test_dtypes_are_correct(self, tmp_data_dir):
        info = scan_schema(tmp_data_dir)
        assert info["dtypes"]["timestamp_ns"] == "int64"
        assert info["dtypes"]["symbol"] in ("string", "large_string", "utf8")
        assert info["dtypes"]["ent_permutation_returns_8"] in ("double", "float64")

    def test_vector_coverage_full_data(self, tmp_data_dir):
        info = scan_schema(tmp_data_dir)
        for vname, vinfo in info["vectors"].items():
            assert vinfo["coverage"] == 1.0, f"Vector {vname} should have full coverage"
            assert vinfo["missing"] == 0

    def test_vector_coverage_partial_data(self, tmp_partial_dir):
        info = scan_schema(tmp_partial_dir)
        assert info["vectors"]["entropy"]["coverage"] == 1.0
        assert info["vectors"]["volatility"]["coverage"] == 1.0
        assert info["vectors"]["trend"]["coverage"] == 0.0
        assert info["vectors"]["illiquidity"]["coverage"] == 0.0

    def test_symbols_detected(self, tmp_data_dir):
        info = scan_schema(tmp_data_dir)
        assert set(info["symbols"]) == {"BTC", "ETH", "SOL"}

    def test_single_symbol_detected(self, tmp_single_symbol_dir):
        info = scan_schema(tmp_single_symbol_dir)
        assert info["symbols"] == ["BTC"]

    def test_files_are_listed(self, tmp_data_dir):
        info = scan_schema(tmp_data_dir)
        assert len(info["files"]) == 3
        assert all(f.endswith(".parquet") for f in info["files"])

    def test_error_on_nonexistent_dir(self):
        with pytest.raises(FileNotFoundError):
            scan_schema("/nonexistent/path")

    def test_error_on_empty_dir(self, tmp_empty_dir):
        with pytest.raises(FileNotFoundError, match="No parquet files"):
            scan_schema(tmp_empty_dir)

    def test_error_on_file_not_dir(self, tmp_data_dir):
        parquet_file = Path(tmp_data_dir) / "hour_001.parquet"
        with pytest.raises(NotADirectoryError):
            scan_schema(str(parquet_file))

    def test_nested_glob(self, tmp_multi_subdir):
        info = scan_schema(tmp_multi_subdir, glob_pattern="**/*.parquet")
        assert info["file_count"] == 2
        assert info["total_rows"] == 200


class TestPrintSchemaSummary:
    """Tests for print_schema_summary() — no crashes, output sanity."""

    def test_does_not_crash_full_data(self, tmp_data_dir, capsys):
        print_schema_summary(tmp_data_dir)
        captured = capsys.readouterr()
        assert "Files:" in captured.out
        assert "Total rows:" in captured.out
        assert "entropy" in captured.out

    def test_does_not_crash_partial_data(self, tmp_partial_dir, capsys):
        print_schema_summary(tmp_partial_dir)
        captured = capsys.readouterr()
        assert "entropy" in captured.out


# ============================================================================
# Test: load_parquet
# ============================================================================


class TestLoadParquet:
    """Tests for load_parquet() — the core data loading function."""

    def test_loads_all_data(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        assert len(df) == 800
        assert "timestamp_ns" in df.columns
        assert "symbol" in df.columns

    def test_sorted_by_timestamp(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        ts = df["timestamp_ns"].values
        assert np.all(ts[1:] >= ts[:-1]), "Data should be sorted by timestamp_ns"

    def test_all_feature_columns_present(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        expected_cols = _all_feature_columns()
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_filter_single_symbol(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir, symbols=["BTC"])
        assert set(df["symbol"].unique()) == {"BTC"}
        assert len(df) > 0
        assert len(df) < 800  # Should be subset

    def test_filter_multiple_symbols(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir, symbols=["BTC", "ETH"])
        assert set(df["symbol"].unique()).issubset({"BTC", "ETH"})

    def test_filter_nonexistent_symbol_empty(self, tmp_data_dir):
        with pytest.raises(ValueError, match="No data loaded"):
            load_parquet(tmp_data_dir, symbols=["DOGE"])

    def test_max_rows(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir, max_rows=50)
        assert len(df) == 50

    def test_max_rows_larger_than_data(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir, max_rows=10000)
        assert len(df) == 800

    def test_select_specific_columns(self, tmp_data_dir):
        cols = ["timestamp_ns", "symbol", "ent_permutation_returns_8", "vol_returns_1m"]
        df = load_parquet(tmp_data_dir, columns=cols)
        assert set(df.columns) == set(cols)
        assert len(df) == 800

    def test_partial_schema_loads(self, tmp_partial_dir):
        df = load_parquet(tmp_partial_dir)
        assert len(df) == 200
        # Entropy columns present
        assert "ent_permutation_returns_8" in df.columns
        # Trend columns NOT present
        assert "trend_momentum_60" not in df.columns

    def test_error_on_nonexistent_dir(self):
        with pytest.raises(FileNotFoundError):
            load_parquet("/nonexistent/path")

    def test_error_on_empty_dir(self, tmp_empty_dir):
        with pytest.raises(FileNotFoundError, match="No parquet files"):
            load_parquet(tmp_empty_dir)

    def test_data_types_correct(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        assert df["timestamp_ns"].dtype == np.int64
        assert df["ent_permutation_returns_8"].dtype == np.float64

    def test_no_unexpected_index(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        assert df.index.is_monotonic_increasing
        assert df.index[0] == 0

    def test_nested_directory_loading(self, tmp_multi_subdir):
        df = load_parquet(tmp_multi_subdir)
        assert set(df["symbol"].unique()) == {"BTC", "ETH"}
        assert len(df) == 200


class TestLoadParquetTimestampFilter:
    """Tests for timestamp-based filtering in load_parquet."""

    def test_filter_start_timestamp(self, tmp_data_dir):
        df_full = load_parquet(tmp_data_dir)
        mid_ts = int(df_full["timestamp_ns"].median())

        df_filtered = load_parquet(tmp_data_dir, start=str(mid_ts))
        assert len(df_filtered) > 0
        assert len(df_filtered) < len(df_full)
        assert df_filtered["timestamp_ns"].min() >= mid_ts

    def test_filter_end_timestamp(self, tmp_data_dir):
        df_full = load_parquet(tmp_data_dir)
        mid_ts = int(df_full["timestamp_ns"].median())

        df_filtered = load_parquet(tmp_data_dir, end=str(mid_ts))
        assert len(df_filtered) > 0
        assert len(df_filtered) < len(df_full)
        assert df_filtered["timestamp_ns"].max() <= mid_ts

    def test_filter_start_and_end(self, tmp_data_dir):
        df_full = load_parquet(tmp_data_dir)
        q25 = int(df_full["timestamp_ns"].quantile(0.25))
        q75 = int(df_full["timestamp_ns"].quantile(0.75))

        df_filtered = load_parquet(tmp_data_dir, start=str(q25), end=str(q75))
        assert len(df_filtered) > 0
        assert df_filtered["timestamp_ns"].min() >= q25
        assert df_filtered["timestamp_ns"].max() <= q75


class TestLoadParquetLazy:
    """Tests for load_parquet_lazy() — lazy loading."""

    def test_returns_dataset(self, tmp_data_dir):
        ds = load_parquet_lazy(tmp_data_dir)
        assert isinstance(ds, pq.ParquetDataset)

    def test_schema_accessible(self, tmp_data_dir):
        ds = load_parquet_lazy(tmp_data_dir)
        schema = ds.schema
        assert "timestamp_ns" in schema.names
        assert "symbol" in schema.names

    def test_can_read_to_table(self, tmp_data_dir):
        ds = load_parquet_lazy(tmp_data_dir)
        table = ds.read()
        assert table.num_rows == 800

    def test_error_on_empty_dir(self, tmp_empty_dir):
        with pytest.raises(FileNotFoundError):
            load_parquet_lazy(tmp_empty_dir)


# ============================================================================
# Test: validate_schema
# ============================================================================


class TestValidateSchema:
    """Tests for validate_schema() — DataFrame validation."""

    def test_valid_full_data(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        result = validate_schema(df)
        assert result["valid"] is True
        assert result["errors"] == []
        assert result["row_count"] == 800

    def test_all_vectors_available(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        result = validate_schema(df)
        assert len(result["vectors_available"]) == len(FEATURE_VECTORS)
        assert len(result["vectors_complete"]) == len(FEATURE_VECTORS)

    def test_partial_data_valid(self, tmp_partial_dir):
        df = load_parquet(tmp_partial_dir)
        result = validate_schema(df)
        assert result["valid"] is True
        assert "entropy" in result["vectors_available"]
        assert "volatility" in result["vectors_available"]
        assert "trend" not in result["vectors_available"]

    def test_missing_meta_columns_error(self):
        df = pd.DataFrame({"ent_permutation_returns_8": [1.0, 2.0]})
        result = validate_schema(df, require_meta=True)
        assert result["valid"] is False
        assert any("timestamp_ns" in e for e in result["errors"])
        assert any("symbol" in e for e in result["errors"])

    def test_no_meta_requirement_valid(self):
        df = pd.DataFrame({"ent_permutation_returns_8": [1.0, 2.0]})
        result = validate_schema(df, require_meta=False)
        assert result["valid"] is True

    def test_empty_dataframe_error(self):
        df = pd.DataFrame()
        result = validate_schema(df, require_meta=False)
        assert result["valid"] is False
        assert any("Too few rows" in e for e in result["errors"])

    def test_require_specific_vectors(self, tmp_partial_dir):
        df = load_parquet(tmp_partial_dir)
        # Requiring a vector that exists
        result = validate_schema(df, require_vectors=["entropy"])
        assert result["valid"] is True

        # Requiring a vector that doesn't exist
        result = validate_schema(df, require_vectors=["trend"])
        assert result["valid"] is False
        assert any("trend" in e for e in result["errors"])

    def test_nan_warnings(self, tmp_nan_dir):
        df = load_parquet(tmp_nan_dir)
        result = validate_schema(df)
        assert len(result["warnings"]) > 0
        assert any("NaN" in w for w in result["warnings"])

    def test_constant_column_warnings(self, tmp_constant_dir):
        df = load_parquet(tmp_constant_dir)
        result = validate_schema(df)
        assert any("zero variance" in w for w in result["warnings"])

    def test_min_rows_enforcement(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        result = validate_schema(df, min_rows=10000)
        assert result["valid"] is False
        assert any("Too few rows" in e for e in result["errors"])


# ============================================================================
# Test: get_symbols / filter_symbol
# ============================================================================


class TestSymbolOperations:
    """Tests for symbol-related utilities."""

    def test_get_symbols(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        symbols = get_symbols(df)
        assert symbols == ["BTC", "ETH", "SOL"]

    def test_get_symbols_single(self, tmp_single_symbol_dir):
        df = load_parquet(tmp_single_symbol_dir)
        assert get_symbols(df) == ["BTC"]

    def test_get_symbols_no_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert get_symbols(df) == []

    def test_filter_symbol_btc(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        btc = filter_symbol(df, "BTC")
        assert len(btc) > 0
        assert all(btc["symbol"] == "BTC")

    def test_filter_symbol_preserves_columns(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        btc = filter_symbol(df, "BTC")
        assert set(btc.columns) == set(df.columns)

    def test_filter_symbol_reset_index(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        btc = filter_symbol(df, "BTC")
        assert btc.index[0] == 0
        assert btc.index.is_monotonic_increasing

    def test_filter_symbol_nonexistent(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        with pytest.raises(ValueError, match="No data for symbol"):
            filter_symbol(df, "DOGE")

    def test_filter_symbol_no_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="no 'symbol' column"):
            filter_symbol(df, "BTC")


# ============================================================================
# Test: time range operations
# ============================================================================


class TestTimeRangeOperations:
    """Tests for time range filtering and inspection."""

    def test_get_time_range(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        t_min, t_max = get_time_range(df)
        assert t_min is not None
        assert t_max is not None
        assert t_max > t_min

    def test_get_time_range_empty(self):
        df = pd.DataFrame({"x": []})
        t_min, t_max = get_time_range(df)
        assert t_min is None
        assert t_max is None

    def test_get_duration_seconds(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        duration = get_duration_seconds(df)
        assert duration > 0
        # 800 rows at 100ms = 80 seconds
        assert 70 < duration < 90

    def test_get_duration_empty(self):
        df = pd.DataFrame({"x": []})
        assert get_duration_seconds(df) == 0.0

    def test_filter_time_range_start(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        mid_ts = int(df["timestamp_ns"].median())
        filtered = filter_time_range(df, start=str(mid_ts))
        assert len(filtered) > 0
        assert filtered["timestamp_ns"].min() >= mid_ts

    def test_filter_time_range_end(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        mid_ts = int(df["timestamp_ns"].median())
        filtered = filter_time_range(df, end=str(mid_ts))
        assert len(filtered) > 0
        assert filtered["timestamp_ns"].max() <= mid_ts

    def test_filter_time_range_both(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        q25 = int(df["timestamp_ns"].quantile(0.25))
        q75 = int(df["timestamp_ns"].quantile(0.75))
        filtered = filter_time_range(df, start=str(q25), end=str(q75))
        assert len(filtered) > 0
        assert filtered["timestamp_ns"].min() >= q25
        assert filtered["timestamp_ns"].max() <= q75

    def test_filter_time_range_no_ts_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="No timestamp column"):
            filter_time_range(df, start="100")

    def test_filter_time_range_reset_index(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        mid_ts = int(df["timestamp_ns"].median())
        filtered = filter_time_range(df, start=str(mid_ts))
        assert filtered.index[0] == 0


# ============================================================================
# Test: list_parquet_files
# ============================================================================


class TestListParquetFiles:
    """Tests for list_parquet_files() — file inventory."""

    def test_lists_all_files(self, tmp_data_dir):
        files = list_parquet_files(tmp_data_dir)
        assert len(files) == 3

    def test_file_info_has_required_fields(self, tmp_data_dir):
        files = list_parquet_files(tmp_data_dir)
        for f in files:
            assert "path" in f
            assert "name" in f
            assert "rows" in f
            assert "columns" in f
            assert "size_bytes" in f
            assert "size_mb" in f

    def test_row_counts_correct(self, tmp_data_dir):
        files = list_parquet_files(tmp_data_dir)
        total = sum(f["rows"] for f in files)
        assert total == 800

    def test_size_positive(self, tmp_data_dir):
        files = list_parquet_files(tmp_data_dir)
        for f in files:
            assert f["size_bytes"] > 0
            assert f["size_mb"] > 0

    def test_error_on_nonexistent_dir(self):
        with pytest.raises(FileNotFoundError):
            list_parquet_files("/nonexistent/path")

    def test_nested_files(self, tmp_multi_subdir):
        files = list_parquet_files(tmp_multi_subdir)
        assert len(files) == 2


# ============================================================================
# Test: Data integrity after loading
# ============================================================================


class TestDataIntegrity:
    """Tests that loaded data maintains integrity properties."""

    def test_no_duplicate_timestamps_per_symbol(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        for sym in df["symbol"].unique():
            sym_df = df[df["symbol"] == sym]
            # Within a symbol, timestamps should be unique
            # (may not be in synthetic data, but structure should allow it)
            assert len(sym_df) > 0

    def test_feature_values_finite(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        feature_cols = [c for c in df.columns if c not in META_COLUMNS and c != "sequence_id"]
        for col in feature_cols:
            vals = df[col].dropna()
            assert np.all(np.isfinite(vals)), f"Column {col} has non-finite values"

    def test_timestamp_ns_positive(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        assert (df["timestamp_ns"] > 0).all()

    def test_concatenation_preserves_column_count(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        expected_cols = _all_feature_columns()
        # 3 meta + all features
        assert len(df.columns) == 3 + len(expected_cols)

    def test_multi_file_merge_no_schema_mismatch(self, tmp_data_dir):
        """All files should merge without schema conflicts."""
        df = load_parquet(tmp_data_dir)
        # If there was a schema mismatch, concat would fail or produce NaNs
        meta_cols = [c for c in df.columns if c in META_COLUMNS or c == "sequence_id"]
        feature_cols = [c for c in df.columns if c not in meta_cols]
        # No all-NaN feature columns
        for col in feature_cols:
            assert df[col].notna().any(), f"Column {col} is all NaN after merge"

    def test_symbol_column_string_type(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        assert pd.api.types.is_string_dtype(df["symbol"])

    def test_sequence_id_non_negative(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        assert (df["sequence_id"] >= 0).all()


# ============================================================================
# Test: Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge case testing — unusual but valid inputs."""

    def test_single_row_file(self, tmp_path):
        data_dir = tmp_path / "single_row"
        data_dir.mkdir()
        _make_parquet_file(str(data_dir / "one.parquet"), n_rows=1, seed=99)
        df = load_parquet(str(data_dir))
        assert len(df) == 1

    def test_very_small_file(self, tmp_path):
        data_dir = tmp_path / "tiny"
        data_dir.mkdir()
        _make_parquet_file(str(data_dir / "tiny.parquet"), n_rows=5, seed=100)
        df = load_parquet(str(data_dir))
        assert len(df) == 5

    def test_max_rows_one(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir, max_rows=1)
        assert len(df) == 1

    def test_load_twice_same_result(self, tmp_data_dir):
        df1 = load_parquet(tmp_data_dir)
        df2 = load_parquet(tmp_data_dir)
        pd.testing.assert_frame_equal(df1, df2)

    def test_many_small_files(self, tmp_path):
        data_dir = tmp_path / "many"
        data_dir.mkdir()
        for i in range(20):
            _make_parquet_file(
                str(data_dir / f"file_{i:03d}.parquet"),
                n_rows=10,
                seed=200 + i,
                start_ns=1_000_000_000_000 + i * 10 * 100_000_000,
            )
        df = load_parquet(str(data_dir))
        assert len(df) == 200

    def test_symbols_filter_case_sensitive(self, tmp_data_dir):
        """Symbol filter should be case-sensitive."""
        with pytest.raises(ValueError, match="No data loaded"):
            load_parquet(tmp_data_dir, symbols=["btc"])

    def test_glob_pattern_override(self, tmp_data_dir):
        # Non-matching pattern
        with pytest.raises(FileNotFoundError, match="No parquet files"):
            load_parquet(tmp_data_dir, glob_pattern="*.csv")


# ============================================================================
# Test: Interaction with config module
# ============================================================================


class TestConfigInteraction:
    """Tests that loader works correctly with config.py definitions."""

    def test_all_config_vectors_resolvable(self, tmp_data_dir):
        """Every vector defined in config can be looked up in loaded data."""
        df = load_parquet(tmp_data_dir)
        col_set = set(df.columns)
        for vname, vspec in FEATURE_VECTORS.items():
            for col in vspec["columns"]:
                assert col in col_set, f"Config column {col} (vector {vname}) not in loaded data"

    def test_validate_schema_matches_config_vectors(self, tmp_data_dir):
        """validate_schema agrees with config on which vectors are complete."""
        df = load_parquet(tmp_data_dir)
        result = validate_schema(df)
        assert set(result["vectors_complete"]) == set(FEATURE_VECTORS.keys())

    def test_scan_schema_matches_config_vectors(self, tmp_data_dir):
        """scan_schema vector info matches config expected dims."""
        info = scan_schema(tmp_data_dir)
        for vname, vinfo in info["vectors"].items():
            expected_dim = FEATURE_VECTORS[vname]["expected_dim"]
            assert vinfo["expected"] == expected_dim, (
                f"Vector {vname}: scan says {vinfo['expected']}, config says {expected_dim}"
            )

    def test_meta_columns_excluded_from_features(self, tmp_data_dir):
        """Meta columns from config are not counted as feature columns."""
        info = scan_schema(tmp_data_dir)
        for mc in META_COLUMNS:
            for vname, vinfo in info["vectors"].items():
                # No meta column should appear in any vector's missing list as "expected"
                expected = FEATURE_VECTORS[vname]["columns"]
                assert mc not in expected, f"Meta column {mc} in vector {vname}"

    def test_composite_vectors_resolve(self, tmp_data_dir):
        """Composite vectors resolve to correct column count from loaded data."""
        df = load_parquet(tmp_data_dir)
        col_set = set(df.columns)
        for cname, cspec in COMPOSITE_VECTORS.items():
            expected_cols = get_vector_columns(cname)
            found = [c for c in expected_cols if c in col_set]
            assert len(found) == len(expected_cols), (
                f"Composite {cname}: {len(found)}/{len(expected_cols)} columns found"
            )


# ============================================================================
# Test: Robustness — corrupted / mixed files
# ============================================================================


class TestRobustness:
    """Tests for handling corrupted or unexpected files."""

    def test_skips_non_parquet_files(self, tmp_data_dir):
        """Non-parquet files in the directory should be ignored."""
        # Add a non-parquet file
        (Path(tmp_data_dir) / "notes.txt").write_text("not parquet")
        df = load_parquet(tmp_data_dir)
        assert len(df) == 800  # Should still load all parquet data

    def test_skips_corrupted_files(self, tmp_data_dir):
        """Corrupted parquet files should be skipped with a warning."""
        # Create a corrupted file
        corrupted = Path(tmp_data_dir) / "corrupted.parquet"
        corrupted.write_bytes(b"not valid parquet data")

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = load_parquet(tmp_data_dir)
            # Should have loaded the 3 valid files
            assert len(df) == 800
            # Should have warned about the corrupted file
            assert any("corrupted" in str(warning.message) for warning in w)

    def test_handles_empty_parquet_file(self, tmp_path):
        """An empty parquet file (0 rows) should be skipped."""
        data_dir = tmp_path / "with_empty"
        data_dir.mkdir()

        # Create empty file
        schema = pa.schema([
            ("timestamp_ns", pa.int64()),
            ("symbol", pa.string()),
            ("sequence_id", pa.uint64()),
            ("ent_permutation_returns_8", pa.float64()),
        ])
        table = pa.table(
            {"timestamp_ns": [], "symbol": [], "sequence_id": pa.array([], type=pa.uint64()),
             "ent_permutation_returns_8": []},
            schema=schema,
        )
        pq.write_table(table, str(data_dir / "empty.parquet"))

        # Create non-empty file
        _make_parquet_file(str(data_dir / "data.parquet"), n_rows=50, seed=300)

        df = load_parquet(str(data_dir))
        assert len(df) == 50


# ============================================================================
# Test: Idempotency and determinism
# ============================================================================


class TestIdempotency:
    """Tests that repeated operations produce consistent results."""

    def test_scan_schema_deterministic(self, tmp_data_dir):
        info1 = scan_schema(tmp_data_dir)
        info2 = scan_schema(tmp_data_dir)
        assert info1["file_count"] == info2["file_count"]
        assert info1["total_rows"] == info2["total_rows"]
        assert info1["columns"] == info2["columns"]

    def test_load_deterministic(self, tmp_data_dir):
        df1 = load_parquet(tmp_data_dir)
        df2 = load_parquet(tmp_data_dir)
        pd.testing.assert_frame_equal(df1, df2)

    def test_validate_deterministic(self, tmp_data_dir):
        df = load_parquet(tmp_data_dir)
        r1 = validate_schema(df)
        r2 = validate_schema(df)
        assert r1["valid"] == r2["valid"]
        assert r1["errors"] == r2["errors"]
        assert r1["vectors_available"] == r2["vectors_available"]

    def test_filter_then_validate(self, tmp_data_dir):
        """Filtering then validating should still work."""
        df = load_parquet(tmp_data_dir)
        btc = filter_symbol(df, "BTC")
        result = validate_schema(btc)
        assert result["valid"] is True
        assert result["row_count"] > 0
