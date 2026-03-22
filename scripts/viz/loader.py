"""
Data loading utilities for visualization.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd
import pyarrow.parquet as pq


def load_data(
    data_dir: Union[str, Path] = './data/features',
    symbols: Optional[List[str]] = None,
    hours: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load feature data from Parquet files.

    Args:
        data_dir: Directory containing Parquet files
        symbols: Filter to specific symbols (e.g., ['BTC', 'ETH'])
        hours: Only load last N hours of data
        columns: Only load specific columns

    Returns:
        DataFrame with all features
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No Parquet files found in {data_dir}")

    # Filter by date if hours specified
    if hours:
        cutoff = datetime.now() - timedelta(hours=hours)
        filtered = []
        for f in files:
            try:
                date_str = f.parent.name
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date >= cutoff.replace(hour=0, minute=0, second=0):
                    filtered.append(f)
            except ValueError:
                filtered.append(f)
        files = filtered

    # Load files
    dfs = []
    for f in files:
        try:
            table = pq.read_table(f, columns=columns)
            df = table.to_pandas()
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Convert timestamp
    ts_col = 'timestamp_ns' if 'timestamp_ns' in df.columns else 'timestamp'
    if ts_col in df.columns:
        df['datetime'] = pd.to_datetime(df[ts_col], unit='ns')
        df = df.sort_values('datetime').reset_index(drop=True)

    # Filter symbols
    if symbols and 'symbol' in df.columns:
        df = df[df['symbol'].isin(symbols)]

    return df


def load_recent(hours: int = 24, **kwargs) -> pd.DataFrame:
    """Load last N hours of data."""
    return load_data(hours=hours, **kwargs)


def get_symbols(df: pd.DataFrame) -> List[str]:
    """Get list of symbols in DataFrame."""
    if 'symbol' in df.columns:
        return sorted(df['symbol'].unique().tolist())
    return []


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (excluding metadata)."""
    meta_cols = ['timestamp_ns', 'timestamp', 'symbol', 'sequence_id', 'datetime']
    return [c for c in df.columns if c not in meta_cols]


def summarize(df: pd.DataFrame) -> None:
    """Print summary of loaded data."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"  Rows:        {len(df):,}")
    print(f"  Columns:     {len(df.columns)}")

    if 'symbol' in df.columns:
        symbols = get_symbols(df)
        print(f"  Symbols:     {', '.join(symbols)}")

    if 'datetime' in df.columns:
        print(f"  Date range:  {df['datetime'].min()} to {df['datetime'].max()}")
        duration = df['datetime'].max() - df['datetime'].min()
        print(f"  Duration:    {duration}")

    # Feature columns
    feature_cols = get_feature_columns(df)
    print(f"  Features:    {len(feature_cols)}")

    # NaN summary
    nan_counts = df[feature_cols].isna().sum()
    high_nan = nan_counts[nan_counts > 0]
    if len(high_nan) > 0:
        print(f"  Cols w/ NaN: {len(high_nan)}")

    print("=" * 60 + "\n")
