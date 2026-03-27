"""
Skeptical Tests for Data Loader

These tests ensure data is loaded correctly and validates inputs.
"""

import pytest
import polars as pl
import tempfile
from pathlib import Path
from backtest.data_loader import load_features, validate_features_for_strategy, FeatureDataset


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_parquet(temp_data_dir):
    """Create a sample Parquet file."""
    df = pl.DataFrame({
        "timestamp_ms": [1000, 2000, 3000, 4000, 5000],
        "symbol": ["BTC"] * 5,
        "raw_midprice": [100.0, 101.0, 102.0, 103.0, 104.0],
        "whale_flow_zscore_1h": [0.5, 1.0, 1.5, 2.0, 2.5],
        "accumulation_score": [0.3, 0.4, 0.5, 0.6, 0.7],
    })

    filepath = temp_data_dir / "BTC_features.parquet"
    df.write_parquet(filepath)
    return temp_data_dir


@pytest.fixture
def multi_symbol_parquet(temp_data_dir):
    """Create Parquet files with multiple symbols."""
    # BTC file
    btc_df = pl.DataFrame({
        "timestamp_ms": [1000, 2000, 3000],
        "symbol": ["BTC"] * 3,
        "raw_midprice": [50000.0, 50100.0, 50200.0],
    })
    btc_df.write_parquet(temp_data_dir / "BTC_features.parquet")

    # ETH file
    eth_df = pl.DataFrame({
        "timestamp_ms": [1000, 2000, 3000],
        "symbol": ["ETH"] * 3,
        "raw_midprice": [3000.0, 3010.0, 3020.0],
    })
    eth_df.write_parquet(temp_data_dir / "ETH_features.parquet")

    return temp_data_dir


# =============================================================================
# BASIC LOADING TESTS
# =============================================================================


class TestBasicLoading:
    """Basic data loading tests."""

    def test_load_returns_dataset(self, sample_parquet):
        """Loading should return FeatureDataset."""
        dataset = load_features(sample_parquet, "BTC")
        assert isinstance(dataset, FeatureDataset)

    def test_load_contains_dataframe(self, sample_parquet):
        """Dataset should contain a DataFrame."""
        dataset = load_features(sample_parquet, "BTC")
        assert isinstance(dataset.df, pl.DataFrame)

    def test_load_correct_row_count(self, sample_parquet):
        """Should load correct number of rows."""
        dataset = load_features(sample_parquet, "BTC")
        assert dataset.n_rows == 5

    def test_load_extracts_symbol(self, sample_parquet):
        """Should extract symbol correctly."""
        dataset = load_features(sample_parquet, "BTC")
        assert dataset.symbol == "BTC"

    def test_load_extracts_time_range(self, sample_parquet):
        """Should extract time range correctly."""
        dataset = load_features(sample_parquet, "BTC")
        assert dataset.start_time == 1000
        assert dataset.end_time == 5000


# =============================================================================
# COLUMN HANDLING TESTS
# =============================================================================


class TestColumnHandling:
    """Test column detection and renaming."""

    def test_standardizes_timestamp_column(self, temp_data_dir):
        """Should rename timestamp variants to timestamp_ms."""
        df = pl.DataFrame({
            "time": [1, 2, 3],  # Non-standard name
            "price": [100.0, 101.0, 102.0],  # Non-standard name
        })
        df.write_parquet(temp_data_dir / "BTC_features.parquet")

        dataset = load_features(temp_data_dir, "BTC")
        assert "timestamp_ms" in dataset.df.columns

    def test_standardizes_price_column(self, temp_data_dir):
        """Should rename price variants to raw_midprice."""
        df = pl.DataFrame({
            "timestamp_ms": [1, 2, 3],
            "close": [100.0, 101.0, 102.0],  # Non-standard name
        })
        df.write_parquet(temp_data_dir / "BTC_features.parquet")

        dataset = load_features(temp_data_dir, "BTC")
        assert "raw_midprice" in dataset.df.columns

    def test_identifies_feature_columns(self, sample_parquet):
        """Should identify feature columns (excluding metadata)."""
        dataset = load_features(sample_parquet, "BTC")

        # Should not include metadata columns
        assert "timestamp_ms" not in dataset.feature_columns
        assert "symbol" not in dataset.feature_columns

        # Should include actual features
        assert "whale_flow_zscore_1h" in dataset.feature_columns


# =============================================================================
# FILTERING TESTS
# =============================================================================


class TestFiltering:
    """Test data filtering functionality."""

    def test_filter_by_symbol(self, multi_symbol_parquet):
        """Should filter data by symbol."""
        btc_dataset = load_features(multi_symbol_parquet, "BTC")
        eth_dataset = load_features(multi_symbol_parquet, "ETH")

        assert btc_dataset.df["raw_midprice"].mean() > 40000  # BTC prices
        assert eth_dataset.df["raw_midprice"].mean() < 4000  # ETH prices

    def test_filter_by_start_time(self, sample_parquet):
        """Should filter data after start_time_ms."""
        dataset = load_features(sample_parquet, "BTC", start_time_ms=3000)

        # Should only have rows with timestamp >= 3000
        assert dataset.df["timestamp_ms"].min() >= 3000
        assert dataset.n_rows == 3  # 3000, 4000, 5000

    def test_filter_by_end_time(self, sample_parquet):
        """Should filter data before end_time_ms."""
        dataset = load_features(sample_parquet, "BTC", end_time_ms=3000)

        # Should only have rows with timestamp <= 3000
        assert dataset.df["timestamp_ms"].max() <= 3000
        assert dataset.n_rows == 3  # 1000, 2000, 3000

    def test_filter_by_both_times(self, sample_parquet):
        """Should filter data between start and end times."""
        dataset = load_features(sample_parquet, "BTC", start_time_ms=2000, end_time_ms=4000)

        assert dataset.df["timestamp_ms"].min() >= 2000
        assert dataset.df["timestamp_ms"].max() <= 4000
        assert dataset.n_rows == 3  # 2000, 3000, 4000


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Test error handling in data loader."""

    def test_no_files_raises_error(self, temp_data_dir):
        """Should raise error if no parquet files found."""
        with pytest.raises(ValueError) as exc_info:
            load_features(temp_data_dir, "BTC")

        assert "No parquet files found" in str(exc_info.value)

    def test_wrong_symbol_raises_error(self, sample_parquet):
        """Should raise error if symbol not found."""
        with pytest.raises(ValueError) as exc_info:
            load_features(sample_parquet, "NONEXISTENT")

        assert "No parquet files found" in str(exc_info.value) or "No data found" in str(exc_info.value)

    def test_no_timestamp_column_raises_error(self, temp_data_dir):
        """Should raise error if no timestamp column."""
        df = pl.DataFrame({
            "price": [100.0, 101.0, 102.0],
        })
        df.write_parquet(temp_data_dir / "BTC_features.parquet")

        with pytest.raises(ValueError) as exc_info:
            load_features(temp_data_dir, "BTC")

        assert "timestamp" in str(exc_info.value).lower()

    def test_no_price_column_raises_error(self, temp_data_dir):
        """Should raise error if no price column."""
        df = pl.DataFrame({
            "timestamp_ms": [1, 2, 3],
        })
        df.write_parquet(temp_data_dir / "BTC_features.parquet")

        with pytest.raises(ValueError) as exc_info:
            load_features(temp_data_dir, "BTC")

        assert "price" in str(exc_info.value).lower()

    def test_empty_after_filter_raises_error(self, sample_parquet):
        """Should raise error if no data after filtering."""
        with pytest.raises(ValueError) as exc_info:
            # Request data outside available range
            load_features(sample_parquet, "BTC", start_time_ms=100000)

        assert "No data remaining" in str(exc_info.value)


# =============================================================================
# DATA QUALITY TESTS
# =============================================================================


class TestDataQuality:
    """Test data quality handling."""

    def test_sorts_by_timestamp(self, temp_data_dir):
        """Should sort data by timestamp."""
        # Create unsorted data
        df = pl.DataFrame({
            "timestamp_ms": [3000, 1000, 2000, 5000, 4000],
            "raw_midprice": [102.0, 100.0, 101.0, 104.0, 103.0],
        })
        df.write_parquet(temp_data_dir / "BTC_features.parquet")

        dataset = load_features(temp_data_dir, "BTC")

        # Should be sorted
        timestamps = dataset.df["timestamp_ms"].to_list()
        assert timestamps == sorted(timestamps)

    def test_removes_duplicates(self, temp_data_dir):
        """Should remove duplicate timestamps."""
        df = pl.DataFrame({
            "timestamp_ms": [1000, 1000, 2000, 2000, 3000],
            "raw_midprice": [100.0, 100.1, 101.0, 101.1, 102.0],
        })
        df.write_parquet(temp_data_dir / "BTC_features.parquet")

        dataset = load_features(temp_data_dir, "BTC")

        # Should keep only unique timestamps
        assert dataset.n_rows == 3

    def test_concatenates_multiple_files(self, temp_data_dir):
        """Should concatenate data from multiple files."""
        # Create multiple files
        df1 = pl.DataFrame({
            "timestamp_ms": [1000, 2000],
            "raw_midprice": [100.0, 101.0],
        })
        df2 = pl.DataFrame({
            "timestamp_ms": [3000, 4000],
            "raw_midprice": [102.0, 103.0],
        })

        df1.write_parquet(temp_data_dir / "BTC_part1.parquet")
        df2.write_parquet(temp_data_dir / "BTC_part2.parquet")

        dataset = load_features(temp_data_dir, "BTC")

        # Should have data from both files
        assert dataset.n_rows == 4


# =============================================================================
# FEATURE VALIDATION TESTS
# =============================================================================


class TestFeatureValidation:
    """Test feature validation for strategies."""

    def test_validate_features_returns_missing(self, sample_parquet):
        """Should return list of missing features."""
        dataset = load_features(sample_parquet, "BTC")

        missing = validate_features_for_strategy(
            dataset,
            ["whale_flow_zscore_1h", "nonexistent_feature"],
        )

        assert "nonexistent_feature" in missing
        assert "whale_flow_zscore_1h" not in missing

    def test_validate_features_empty_when_all_present(self, sample_parquet):
        """Should return empty list when all features present."""
        dataset = load_features(sample_parquet, "BTC")

        missing = validate_features_for_strategy(
            dataset,
            ["whale_flow_zscore_1h", "accumulation_score"],
        )

        assert missing == []


# =============================================================================
# DATASET REPRESENTATION TESTS
# =============================================================================


class TestDatasetRepr:
    """Test dataset string representation."""

    def test_repr_contains_info(self, sample_parquet):
        """Repr should contain key information."""
        dataset = load_features(sample_parquet, "BTC")
        s = repr(dataset)

        assert "BTC" in s
        assert str(dataset.n_rows) in s or "5" in s
