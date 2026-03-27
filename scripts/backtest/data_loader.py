"""
Data Loader for NAT Backtester

Loads Parquet feature files and prepares them for backtesting.
"""

import polars as pl
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class FeatureDataset:
    """Container for loaded feature data with metadata."""
    df: pl.DataFrame
    symbol: str
    start_time: int
    end_time: int
    n_rows: int
    feature_columns: List[str]

    def __repr__(self) -> str:
        return (
            f"FeatureDataset(symbol={self.symbol}, "
            f"rows={self.n_rows:,}, "
            f"features={len(self.feature_columns)})"
        )


def load_features(
    data_dir: Path,
    symbol: str = "BTC",
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> FeatureDataset:
    """
    Load all feature files for a symbol, sorted by time.

    Parameters
    ----------
    data_dir : Path
        Directory containing Parquet files (can have subdirectories)
    symbol : str
        Symbol to load (e.g., "BTC", "ETH")
    start_time_ms : int, optional
        Filter to data after this timestamp
    end_time_ms : int, optional
        Filter to data before this timestamp

    Returns
    -------
    FeatureDataset
        Loaded and prepared dataset

    Raises
    ------
    ValueError
        If no parquet files found for symbol
    """
    data_dir = Path(data_dir)

    # Find all parquet files for this symbol
    patterns = [
        f"**/{symbol}*.parquet",
        f"**/{symbol.lower()}*.parquet",
        f"**/features_{symbol}*.parquet",
    ]

    files = []
    for pattern in patterns:
        files.extend(data_dir.glob(pattern))

    # Also check for generic feature files
    if not files:
        files = list(data_dir.glob("**/*.parquet"))

    files = sorted(set(files))

    if not files:
        raise ValueError(f"No parquet files found for {symbol} in {data_dir}")

    # Load and concatenate
    dfs = []
    for f in files:
        try:
            df = pl.read_parquet(f)
            # Filter by symbol if column exists
            if "symbol" in df.columns:
                df = df.filter(pl.col("symbol") == symbol)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    if not dfs:
        raise ValueError(f"No data found for {symbol}")

    df = pl.concat(dfs)

    # Ensure timestamp column exists
    timestamp_col = None
    for col in ["timestamp_ms", "timestamp", "time", "ts"]:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        raise ValueError("No timestamp column found in data")

    # Rename to standard name if needed
    if timestamp_col != "timestamp_ms":
        df = df.rename({timestamp_col: "timestamp_ms"})

    # Sort by time
    df = df.sort("timestamp_ms")

    # Remove duplicates
    df = df.unique(subset=["timestamp_ms"], keep="first")

    # Apply time filters
    if start_time_ms is not None:
        df = df.filter(pl.col("timestamp_ms") >= start_time_ms)
    if end_time_ms is not None:
        df = df.filter(pl.col("timestamp_ms") <= end_time_ms)

    if len(df) == 0:
        raise ValueError(f"No data remaining after time filters")

    # Ensure price column exists
    price_col = None
    for col in ["raw_midprice", "midprice", "price", "close"]:
        if col in df.columns:
            price_col = col
            break

    if price_col is None:
        raise ValueError("No price column found in data")

    if price_col != "raw_midprice":
        df = df.rename({price_col: "raw_midprice"})

    # Get feature columns (exclude metadata)
    metadata_cols = {"timestamp_ms", "symbol", "raw_midprice", "sequence_id"}
    feature_columns = [c for c in df.columns if c not in metadata_cols]

    return FeatureDataset(
        df=df,
        symbol=symbol,
        start_time=df["timestamp_ms"].min(),
        end_time=df["timestamp_ms"].max(),
        n_rows=len(df),
        feature_columns=feature_columns,
    )


def validate_features_for_strategy(
    dataset: FeatureDataset,
    required_features: List[str],
) -> List[str]:
    """
    Check if required features exist in dataset.

    Returns list of missing features.
    """
    available = set(dataset.df.columns)
    missing = [f for f in required_features if f not in available]
    return missing
