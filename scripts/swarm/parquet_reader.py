"""
Parquet reader for swarm evaluators.

Thin wrapper around cluster_pipeline.loader that handles the specific needs
of swarm evaluation: symbol filtering, time-window selection, and column
pruning to keep memory low (~500 MB per worker).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _find_latest_date(data_dir: str) -> Optional[str]:
    """Find the most recent date directory in data_dir."""
    import re
    data_path = Path(data_dir)
    if not data_path.exists():
        return None
    dirs = sorted(
        [d.name for d in data_path.iterdir()
         if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)],
        reverse=True,
    )
    return dirs[0] if dirs else None


def read_evaluation_data(
    data_dir: str,
    *,
    symbol: str = "BTC",
    hours: int = 24,
    columns: Optional[list[str]] = None,
    max_memory_mb: float = 500.0,
) -> pd.DataFrame:
    """Read Parquet data for a swarm evaluation window.

    Uses the most recent N hours of available data. If real-time data
    is stale (ingestor not running), falls back to the latest data
    on disk rather than returning empty.

    Args:
        data_dir: Path to features directory (e.g. data/features).
        symbol: Symbol to load (BTC, ETH, SOL).
        hours: Number of hours of history to load.
        columns: Specific columns to load (None = all).
        max_memory_mb: Memory cap per worker.

    Returns:
        DataFrame sorted by timestamp_ns, single symbol.
    """
    from cluster_pipeline.loader import load_parquet

    # Find the latest available date and compute range from there
    latest = _find_latest_date(data_dir)
    if latest is None:
        raise FileNotFoundError(f"No date directories in {data_dir}")

    end_date = latest
    days_back = max(1, (hours + 23) // 24)
    start_dt = datetime.fromisoformat(latest) - timedelta(days=days_back)
    start_date = start_dt.strftime("%Y-%m-%d")

    # Always include meta columns for downstream processing
    meta = ["timestamp_ns", "symbol", "raw_midprice", "raw_spread"]
    if columns is not None:
        load_cols = list(set(meta + columns))
    else:
        load_cols = None  # load all

    df = load_parquet(
        data_dir,
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        columns=load_cols,
        max_memory_mb=max_memory_mb,
    )

    # Trim to requested hours from the end of available data
    if "timestamp_ns" in df.columns and len(df) > 0:
        max_ts = df["timestamp_ns"].max()
        cutoff_ns = max_ts - int(hours * 3600 * 1e9)
        df = df[df["timestamp_ns"] >= cutoff_ns].reset_index(drop=True)

    logger.info(
        "Loaded %d rows for %s (%d hours, %.1f MB est)",
        len(df), symbol, hours, len(df) * df.shape[1] * 8 / 1e6,
    )
    return df


def list_available_data(data_dir: str) -> dict:
    """Quick scan of available Parquet data for swarm planning.

    Returns dict with date range, symbols, row count estimate.
    """
    from cluster_pipeline.loader import scan_schema

    info = scan_schema(data_dir)
    return {
        "file_count": info["file_count"],
        "total_rows": info["total_rows"],
        "symbols": info["symbols"],
        "columns": len(info["columns"]),
    }
