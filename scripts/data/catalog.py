"""Data catalog — discovery and manifest for the NAT feature store.

Provides data_manifest() which returns availability info matching
the format expected by agent.hypothesis_queue._is_runnable():
    manifest["symbols"][sym]["hours"] >= required_hours
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from .features import DATA_DIR, available_dates, available_symbols

log = logging.getLogger(__name__)

# Approximate rows per hour at 10 Hz emission × 3600s
ROWS_PER_HOUR = 36_000


def data_manifest(
    data_dir: Optional[Path] = None,
    symbols: Optional[list[str]] = None,
) -> dict:
    """Build a data availability manifest.

    Returns:
        {
            "dates": {
                "2026-05-20": {"n_files": 3, "total_rows": 5400, "hours_per_symbol": 0.05},
                ...
            },
            "symbols": {
                "BTC": {"hours": 50.5, "latest": "2026-05-22T18:30:00"},
                ...
            },
            "total_dates": int,
            "total_hours_per_symbol": float,
            "updated": str (ISO timestamp),
        }

    Compatible with HypothesisQueue._is_runnable() which checks:
        manifest["symbols"][sym]["hours"] >= min_hours
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    dates_list = available_dates(data_dir=root)

    if not dates_list:
        return {
            "dates": {},
            "symbols": {},
            "total_dates": 0,
            "total_hours_per_symbol": 0.0,
            "updated": datetime.now(timezone.utc).isoformat(),
        }

    if symbols is None:
        symbols = available_symbols(data_dir=root)

    # Per-date stats
    dates_info: dict[str, dict] = {}
    symbol_hours: dict[str, float] = {sym: 0.0 for sym in symbols}

    n_symbols = max(len(symbols), 1)

    for date_str in dates_list:
        date_dir = root / date_str
        files = sorted(date_dir.glob("*.parquet"))

        total_rows = 0
        for f in files:
            try:
                meta = pq.read_metadata(str(f))
                total_rows += meta.num_rows
            except Exception:
                continue

        rows_per_sym = total_rows / n_symbols
        hours = rows_per_sym / ROWS_PER_HOUR

        dates_info[date_str] = {
            "n_files": len(files),
            "total_rows": total_rows,
            "hours_per_symbol": round(hours, 2),
        }

        for sym in symbols:
            symbol_hours[sym] += hours

    # Build symbol summary with latest timestamp from most recent date
    symbols_info: dict[str, dict] = {}
    for sym in symbols:
        symbols_info[sym] = {
            "hours": round(symbol_hours[sym], 1),
            "latest": None,
        }

    # Get latest timestamp per symbol from most recent date
    if dates_list:
        import pandas as pd
        from .features import load_features

        latest_date = dates_list[-1]
        for sym in symbols:
            try:
                df = load_features(
                    symbols=[sym],
                    date_range=(latest_date, latest_date),
                    columns=["timestamp_ns"],
                    data_dir=root,
                    validate=False,
                )
                if not df.empty:
                    ts_val = df["timestamp_ns"].max()
                    symbols_info[sym]["latest"] = pd.Timestamp(
                        ts_val, unit="ns"
                    ).isoformat()
            except Exception:
                pass

    total_hours = sum(symbol_hours.values()) / n_symbols

    return {
        "dates": dates_info,
        "symbols": symbols_info,
        "total_dates": len(dates_list),
        "total_hours_per_symbol": round(total_hours, 1),
        "updated": datetime.now(timezone.utc).isoformat(),
    }


def freshness_check(
    data_dir: Optional[Path] = None,
    max_stale_hours: float = 2.0,
) -> dict:
    """Check if data is fresh enough for live trading.

    Returns:
        {
            "fresh": True/False,
            "staleness_hours": float,
            "latest_timestamp": str or None,
            "message": str,
        }
    """
    from .features import data_health

    health = data_health(data_dir=data_dir)
    if health["freshness_seconds"] is None:
        return {
            "fresh": False,
            "staleness_hours": float("inf"),
            "latest_timestamp": None,
            "message": "No data found",
        }

    stale_h = health["freshness_seconds"] / 3600
    fresh = stale_h <= max_stale_hours

    return {
        "fresh": fresh,
        "staleness_hours": round(stale_h, 2),
        "latest_timestamp": health["latest_timestamp"],
        "message": "Data fresh" if fresh else f"Data stale ({stale_h:.1f}h > {max_stale_hours}h)",
    }
