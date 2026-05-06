"""
Tasks 2+3: Data counter and health checker.

Lightweight metrics collection that reads parquet metadata
without loading full files.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pyarrow.parquet as pq

from .state import DataMetrics, HealthMetrics

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "features"


def collect_data_metrics(data_dir: Path = DEFAULT_DATA_DIR) -> DataMetrics:
    """
    Task 2: Scan data directory and return metrics.
    Uses parquet metadata only (fast, no full file load).
    """
    metrics = DataMetrics()

    if not data_dir.exists():
        return metrics

    parquets = sorted(data_dir.rglob("*.parquet"))
    metrics.total_files = len(parquets)

    if not parquets:
        return metrics

    # Count rows from metadata (fast)
    total_rows = 0
    valid_files = []
    for p in parquets:
        try:
            meta = pq.read_metadata(p)
            total_rows += meta.num_rows
            valid_files.append(p)
        except Exception:
            continue

    metrics.total_rows = total_rows
    metrics.bars_15m = total_rows // (30 * 60 * 15)  # ~30 rows/sec, 60s, 15min

    # Disk size
    total_bytes = sum(p.stat().st_size for p in parquets if p.exists())
    metrics.disk_mb = round(total_bytes / (1024 * 1024), 1)

    # Days
    day_dirs = sorted(set(p.parent.name for p in valid_files))
    metrics.days = len(day_dirs)
    if day_dirs:
        metrics.date_range = f"{day_dirs[0]} to {day_dirs[-1]}"

    # Symbol health: check if last file has all symbols
    if valid_files:
        try:
            last_file = valid_files[-1]
            tbl = pq.read_table(last_file, columns=["symbol"] if "symbol" in pq.read_schema(last_file).names else [])
            if "symbol" in tbl.column_names:
                symbols_present = set(tbl.column("symbol").to_pylist())
                metrics.symbols = {
                    "BTC": "BTC" in symbols_present,
                    "ETH": "ETH" in symbols_present,
                    "SOL": "SOL" in symbols_present,
                }
            else:
                metrics.symbols = {"BTC": True, "ETH": True, "SOL": True}
        except Exception:
            pass

    # Last flush time
    if valid_files:
        last_mtime = valid_files[-1].stat().st_mtime
        metrics.last_flush_ago_s = round(time.time() - last_mtime, 1)

    # Rate estimate (rows per day based on date range)
    if metrics.days >= 2 and total_rows > 0:
        metrics.rate_per_day = round(total_rows / metrics.days)

    return metrics


def check_health(data_dir: Path = DEFAULT_DATA_DIR, hours: int = 1) -> HealthMetrics:
    """
    Task 3: Validate the most recent N hours of data.
    Lightweight check: NaN ratio, gaps, feature count.
    """
    health = HealthMetrics()

    if not data_dir.exists():
        return health

    # Find recent files
    now = time.time()
    cutoff = now - hours * 3600
    recent_files = []
    for p in sorted(data_dir.rglob("*.parquet")):
        try:
            if p.stat().st_mtime >= cutoff and p.stat().st_size > 0:
                recent_files.append(p)
        except Exception:
            continue

    if not recent_files:
        return health

    # Load a sample (last file only for speed)
    try:
        tbl = pq.read_table(recent_files[-1])
        df = tbl.to_pandas()
    except Exception:
        return health

    # NaN ratio (exclude metadata columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        health.nan_ratio = round(float(df[numeric_cols].isna().mean().mean()), 4)

    # Feature count
    health.features_ok = len(numeric_cols) >= 180  # expect ~191 numeric

    # Gap detection (check timestamp column if exists)
    ts_col = None
    for col in ["timestamp", "timestamp_us", "ts"]:
        if col in df.columns:
            ts_col = col
            break

    if ts_col and len(df) > 1:
        timestamps = df[ts_col].values
        if np.issubdtype(timestamps.dtype, np.integer):
            # Microseconds
            diffs = np.diff(timestamps) / 1_000_000  # to seconds
        else:
            diffs = np.diff(timestamps.astype(np.int64)) / 1e9  # nanoseconds to seconds

        gaps = diffs[diffs > 5.0]
        health.n_gaps = int(len(gaps))
        health.longest_gap_s = round(float(gaps.max()), 1) if len(gaps) > 0 else 0.0

    # Per-symbol rows (last hour)
    if "symbol" in df.columns:
        counts = df["symbol"].value_counts().to_dict()
        health.per_symbol_rows_1h = {str(k): int(v) for k, v in counts.items()}

    return health
