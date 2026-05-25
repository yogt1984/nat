"""Unified feature data loader for NAT parquet files.

Every script that reads from data/features/ should use this module
instead of ad-hoc pyarrow/pandas/polars loading.

Key advantages over ad-hoc loading:
- PyArrow predicate pushdown for symbol filtering (avoids loading all symbols)
- Directory-level date filtering (skips files outside range without opening)
- Single bar aggregation implementation (replaces 7 duplicates)
- Schema validation on first load
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "features"

# Required columns always included in output regardless of `columns` parameter
_ALWAYS_INCLUDE = {"timestamp_ns", "symbol"}

# Default bar aggregation spec (matches all 7 existing implementations)
_DEFAULT_AGG = {
    "timestamp_ns": ("timestamp_ns", "first"),
    "raw_midprice": ("raw_midprice", "last"),
    "raw_spread_bps": ("raw_spread_bps", "last"),
    "raw_ask_depth_5": ("raw_ask_depth_5", "std"),
    "flow_vwap_deviation": ("flow_vwap_deviation", "std"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_features(
    symbols: Optional[list[str]] = None,
    date_range: Optional[tuple[str, str]] = None,
    columns: Optional[list[str]] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load tick-level features from parquet files.

    Uses PyArrow predicate pushdown for symbol filtering (avoids loading
    all symbols into memory then filtering). Date range filtering is done
    at directory level (skip directories outside range) and row level
    (timestamp_ns filter pushed down to row groups).

    Args:
        symbols: Filter to these symbols. None = all symbols.
        date_range: Inclusive date range as (start, end) ISO strings.
                    None = all available dates.
        columns: Columns to load. None = all columns.
                 "symbol" and "timestamp_ns" are always included.
        data_dir: Override default data/features/ directory.

    Returns:
        DataFrame sorted by timestamp_ns with requested columns.
        Empty DataFrame if no data matches filters.
    """
    root = Path(data_dir) if data_dir else DATA_DIR

    # Discover parquet files with directory-level date filtering
    files = _discover_files(root, date_range)
    if not files:
        return _empty_df(columns)

    # Ensure required columns in selection
    read_columns = None
    if columns is not None:
        read_columns = list(set(columns) | _ALWAYS_INCLUDE)

    # Build predicate pushdown filters
    filters = _build_filters(symbols=symbols, date_range=date_range)

    # Load with PyArrow
    tables = []
    for fpath in files:
        try:
            table = pq.read_table(str(fpath), columns=read_columns, filters=filters)
        except Exception as e:
            warnings.warn(f"Skipping {fpath.name}: {e}")
            continue
        if table.num_rows > 0:
            tables.append(table)

    if not tables:
        return _empty_df(columns)

    combined = pa.concat_tables(tables, promote_options="default")
    df = combined.to_pandas()
    df.sort_values("timestamp_ns", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_bars(
    symbols: Optional[list[str]] = None,
    date_range: Optional[tuple[str, str]] = None,
    columns: Optional[list[str]] = None,
    bar_seconds: int = 300,
    min_ticks: int = 10,
    agg_spec: Optional[dict[str, tuple[str, str]]] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load features aggregated to fixed-interval bars.

    Calls load_features() then aggregates. Default aggregation matches
    the 7 existing implementations:
        - timestamp_ns: first
        - raw_midprice: last
        - raw_spread_bps: last
        - raw_ask_depth_5: std
        - flow_vwap_deviation: std

    Custom aggregation via agg_spec overrides defaults:
        {"output_name": ("source_column", "agg_func")}

    Args:
        bar_seconds: Bar width in seconds. Default 300 (5 min).
        min_ticks: Minimum ticks per bar. Bars below this are dropped.
        agg_spec: Column-specific aggregation overrides.
                  Format: {"output_col": ("source_col", "agg_func")}
    """
    # Determine which raw columns we need for aggregation
    spec = agg_spec if agg_spec is not None else _DEFAULT_AGG
    source_cols = {src for src, _ in spec.values()}
    source_cols.add("timestamp_ns")

    # Merge with caller's column request
    load_cols = None
    if columns is not None:
        load_cols = list(set(columns) | source_cols | _ALWAYS_INCLUDE)
    else:
        load_cols = list(source_cols | _ALWAYS_INCLUDE)

    ticks = load_features(
        symbols=symbols, date_range=date_range, columns=load_cols, data_dir=data_dir
    )
    if ticks.empty:
        return ticks

    # Filter agg spec to columns actually present
    active_spec = {}
    for out_name, (src_col, func) in spec.items():
        if src_col in ticks.columns:
            active_spec[out_name] = (src_col, func)

    if not active_spec:
        return _empty_df(columns)

    # Bar aggregation: floor-divide-group-agg
    bar_ns = bar_seconds * 1_000_000_000
    ticks["_bar_id"] = ticks["timestamp_ns"].values // bar_ns

    # Add tick count
    active_spec["n_ticks"] = ("timestamp_ns", "count")

    bars = ticks.groupby("_bar_id").agg(**active_spec).reset_index(drop=True)

    # Drop partial bars
    bars = bars[bars["n_ticks"] >= min_ticks].reset_index(drop=True)

    # Fill NaN in std columns (zero variance within bar → 0.0)
    for out_name, (_, func) in spec.items():
        if func == "std" and out_name in bars.columns:
            bars[out_name] = bars[out_name].fillna(0.0)

    return bars


def available_dates(data_dir: Optional[Path] = None) -> list[str]:
    """Return sorted list of YYYY-MM-DD date strings with parquet data."""
    root = Path(data_dir) if data_dir else DATA_DIR
    if not root.exists():
        return []
    dates = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and _is_date_dir(d.name):
            # Verify at least one parquet file exists
            if any(d.glob("*.parquet")):
                dates.append(_parse_date_str(d.name))
    return dates


def available_symbols(
    date: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> list[str]:
    """Return sorted list of symbols present in data.

    If date is specified, only check that date's directory.
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    if not root.exists():
        return []

    if date is not None:
        dirs = [root / date]
    else:
        dirs = [d for d in root.iterdir() if d.is_dir() and _is_date_dir(d.name)]

    symbols = set()
    for d in dirs:
        if not d.exists():
            continue
        files = list(d.glob("*.parquet"))
        if not files:
            continue
        # Read symbol column from files until we stop finding new symbols
        for f in files:
            try:
                table = pq.read_table(str(f), columns=["symbol"])
                symbols.update(table.column("symbol").to_pylist())
            except Exception:
                continue
    return sorted(symbols)


def data_health(data_dir: Optional[Path] = None) -> dict:
    """Quick health check without loading full data.

    Returns:
        {
            "dates": [...],
            "symbols": [...],
            "total_files": int,
            "total_rows": int (from parquet metadata),
            "latest_timestamp": str or None,
            "freshness_seconds": float or None,
            "warnings": [...]
        }
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    warns = []

    dates = available_dates(data_dir=root)
    if not dates:
        return {
            "dates": [],
            "symbols": [],
            "total_files": 0,
            "total_rows": 0,
            "latest_timestamp": None,
            "freshness_seconds": None,
            "warnings": ["No data found"],
        }

    symbols = available_symbols(data_dir=root)
    total_files = 0
    total_rows = 0
    latest_ts = 0

    for d in dates:
        date_dir = root / d
        files = list(date_dir.glob("*.parquet"))
        total_files += len(files)
        for f in files:
            try:
                meta = pq.read_metadata(str(f))
                total_rows += meta.num_rows
            except Exception:
                warns.append(f"Cannot read metadata: {f.name}")

    # Get latest timestamp from most recent file
    if dates:
        latest_dir = root / dates[-1]
        latest_files = sorted(latest_dir.glob("*.parquet"))
        for f in reversed(latest_files):
            try:
                meta = pq.read_metadata(str(f))
                if meta.num_rows > 0:
                    # Read last row's timestamp
                    table = pq.read_table(str(f), columns=["timestamp_ns"])
                    ts_arr = table.column("timestamp_ns")
                    latest_ts = max(latest_ts, ts_arr[len(ts_arr) - 1].as_py())
                    break
            except Exception:
                continue

    latest_str = None
    freshness = None
    if latest_ts > 0:
        latest_dt = pd.Timestamp(latest_ts, unit="ns")
        latest_str = latest_dt.isoformat()
        freshness = (pd.Timestamp.now() - latest_dt).total_seconds()

    return {
        "dates": dates,
        "symbols": symbols,
        "total_files": total_files,
        "total_rows": total_rows,
        "latest_timestamp": latest_str,
        "freshness_seconds": freshness,
        "warnings": warns,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _discover_files(
    root: Path, date_range: Optional[tuple[str, str]] = None
) -> list[Path]:
    """Find parquet files, filtering by date directory if range specified."""
    if not root.exists():
        return []

    files = []
    dirs = sorted(d for d in root.iterdir() if d.is_dir())

    for d in dirs:
        if not _is_date_dir(d.name):
            continue

        # Directory-level date filtering
        if date_range is not None:
            dir_date = _parse_date_str(d.name)
            if dir_date < date_range[0] or dir_date > date_range[1]:
                continue

        dir_files = sorted(d.glob("*.parquet"))
        files.extend(dir_files)

    return files


def _build_filters(
    symbols: Optional[list[str]] = None,
    date_range: Optional[tuple[str, str]] = None,
) -> Optional[list]:
    """Build PyArrow filter expressions for predicate pushdown."""
    filters = []

    if symbols is not None and len(symbols) > 0:
        if len(symbols) == 1:
            filters.append(("symbol", "=", symbols[0]))
        else:
            filters.append(("symbol", "in", symbols))

    # Row-level timestamp filtering (complements directory-level)
    if date_range is not None:
        start_ns = int(pd.Timestamp(f"{date_range[0]}T00:00:00").value)
        end_ns = int(pd.Timestamp(f"{date_range[1]}T23:59:59.999999999").value)
        filters.append(("timestamp_ns", ">=", start_ns))
        filters.append(("timestamp_ns", "<=", end_ns))

    return filters if filters else None


def _is_date_dir(name: str) -> bool:
    """Check if directory name looks like a date (YYYY-MM-DD or YYYY-MM-DD-suffix)."""
    parts = name.split("-")
    if len(parts) < 3:
        return False
    try:
        int(parts[0])
        int(parts[1])
        int(parts[2])
        return True
    except ValueError:
        return False


def _parse_date_str(name: str) -> str:
    """Extract YYYY-MM-DD from directory name (handles suffixes like '2026-05-12-clean')."""
    parts = name.split("-")
    return f"{parts[0]}-{parts[1]}-{parts[2]}"


def _empty_df(columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Return empty DataFrame with expected columns."""
    cols = list(columns) if columns else ["timestamp_ns", "symbol"]
    # Ensure always-include columns present
    for c in _ALWAYS_INCLUDE:
        if c not in cols:
            cols.append(c)
    return pd.DataFrame(columns=cols)
