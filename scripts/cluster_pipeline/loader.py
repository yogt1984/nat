"""
Parquet data loader for NAT cluster analysis pipeline.

Scans a directory of parquet files (written by the Rust ingestor), auto-detects
the schema, validates against expected feature vectors, and returns a unified
DataFrame ready for preprocessing.

Usage:
    from cluster_pipeline.loader import load_parquet, scan_schema, print_schema_summary

    df = load_parquet("./data/features")
    scan_schema("./data/features")
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import logging
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import FEATURE_VECTORS, COMPOSITE_VECTORS, META_COLUMNS, get_vector_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

# Current expected schema version — must match SCHEMA_VERSION in
# rust/ing/src/output/schema.rs.
CURRENT_SCHEMA_VERSION = 1


def read_schema_version(path: Union[str, Path]) -> Optional[int]:
    """Read schema_version from a Parquet file's metadata.

    Returns None for legacy files that pre-date schema versioning.
    """
    try:
        meta = pq.read_schema(str(path))
        ver = meta.metadata.get(b"schema_version")
        return int(ver) if ver is not None else None
    except Exception:
        return None


def normalize_schema(
    df: pd.DataFrame,
    files: List[Path],
) -> pd.DataFrame:
    """Pad missing expected columns with NaN and warn on version mismatch.

    Reads schema_version from the first file's metadata. If files pre-date
    versioning (legacy) or have a different version than CURRENT_SCHEMA_VERSION,
    a warning is logged and missing base columns are padded with NaN.

    Unknown columns (e.g. alg_* algorithm outputs or new features from a
    newer schema) are kept — never dropped.
    """
    from data.schema import ALL_BASE, ALL_OPTIONAL

    # Read version from first file
    version = read_schema_version(files[0]) if files else None

    if version is None:
        logger.warning(
            "Parquet file %s has no schema_version metadata (legacy file). "
            "Expected version %d. Missing columns will be padded with NaN.",
            files[0].name, CURRENT_SCHEMA_VERSION,
        )
    elif version < CURRENT_SCHEMA_VERSION:
        logger.warning(
            "Parquet schema version %d < current %d. "
            "Missing columns will be padded with NaN.",
            version, CURRENT_SCHEMA_VERSION,
        )
    elif version > CURRENT_SCHEMA_VERSION:
        logger.warning(
            "Parquet schema version %d > current %d. "
            "Loader may not recognize all columns. Consider updating.",
            version, CURRENT_SCHEMA_VERSION,
        )

    # Pad missing base columns with NaN (optional columns too, for completeness)
    expected = ALL_BASE + ALL_OPTIONAL
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.info(
            "Padding %d missing columns with NaN: %s%s",
            len(missing),
            ", ".join(missing[:5]),
            f" (+{len(missing)-5} more)" if len(missing) > 5 else "",
        )
        pad = pd.DataFrame(
            np.nan, index=df.index, columns=missing, dtype=np.float64,
        )
        df = pd.concat([df, pad], axis=1)

    return df


# ---------------------------------------------------------------------------
# Schema inspection
# ---------------------------------------------------------------------------


def scan_schema(
    data_dir: Union[str, Path],
    *,
    glob_pattern: str = "**/*.parquet",
) -> Dict:
    """
    Scan parquet files and return schema summary without loading data.

    Returns a dict with:
        - file_count: number of parquet files found
        - total_rows: sum of row counts across files
        - columns: list of column names from the first file
        - dtypes: dict mapping column name -> arrow type string
        - vectors: dict mapping vector name -> {expected, found, missing, coverage}
        - meta_columns: list of meta columns found
        - symbols: unique symbols found (sampled from first file)
        - files: list of file paths
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_dir}")

    files = sorted(data_path.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {data_dir} "
            f"(pattern: {glob_pattern})"
        )

    # Read schema from first valid file
    schema = None
    for f in files:
        try:
            schema = pq.read_schema(str(f))
            break
        except Exception:
            continue
    if schema is None:
        raise ValueError(f"No readable parquet files in {data_dir}")
    columns = schema.names
    dtypes = {name: str(schema.field(name).type) for name in columns}

    # Count total rows (skip corrupted files)
    total_rows = 0
    valid_files = []
    for f in files:
        try:
            meta = pq.read_metadata(str(f))
            total_rows += meta.num_rows
            valid_files.append(f)
        except Exception:
            import warnings
            warnings.warn(f"Skipping corrupted file in schema scan: {f.name}")
    files = valid_files

    # Check vector coverage
    col_set = set(columns)
    vectors = {}
    for vname, vspec in FEATURE_VECTORS.items():
        expected = vspec["columns"]
        found = [c for c in expected if c in col_set]
        missing = [c for c in expected if c not in col_set]
        vectors[vname] = {
            "expected": len(expected),
            "found": len(found),
            "missing": len(missing),
            "missing_columns": missing,
            "coverage": len(found) / len(expected) if expected else 0.0,
        }

    # Meta columns
    meta_found = [c for c in columns if c in META_COLUMNS]

    # Sample symbols from first file
    symbols = []
    try:
        table = pq.read_table(str(files[0]), columns=["symbol"])
        symbols = sorted(table.column("symbol").to_pylist())
        symbols = sorted(set(symbols))
    except Exception:
        pass

    # Schema version (from first readable file)
    schema_version = read_schema_version(files[0]) if files else None

    return {
        "file_count": len(files),
        "total_rows": total_rows,
        "columns": columns,
        "dtypes": dtypes,
        "vectors": vectors,
        "meta_columns": meta_found,
        "symbols": symbols,
        "files": [str(f) for f in files],
        "schema_version": schema_version,
    }


def print_schema_summary(data_dir: Union[str, Path], **kwargs) -> None:
    """Print a human-readable schema summary."""
    info = scan_schema(data_dir, **kwargs)

    print(f"Data directory: {data_dir}")
    print(f"Files:          {info['file_count']}")
    print(f"Total rows:     {info['total_rows']:,}")
    print(f"Columns:        {len(info['columns'])}")
    print(f"Symbols:        {info['symbols']}")
    print(f"Meta columns:   {info['meta_columns']}")
    print()

    print(f"{'Vector':<16} {'Found':>5} / {'Expected':>8}  {'Coverage':>8}  Missing")
    print("-" * 72)
    for vname, vinfo in info["vectors"].items():
        cov = f"{vinfo['coverage']:.0%}"
        missing_str = ", ".join(vinfo["missing_columns"][:3])
        if len(vinfo["missing_columns"]) > 3:
            missing_str += f" (+{len(vinfo['missing_columns'])-3} more)"
        print(
            f"{vname:<16} {vinfo['found']:>5} / {vinfo['expected']:>8}  {cov:>8}  {missing_str}"
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_parquet(
    data_dir: Union[str, Path],
    *,
    symbols: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    columns: Optional[List[str]] = None,
    glob_pattern: str = "**/*.parquet",
    max_rows: Optional[int] = None,
    max_memory_mb: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load parquet files from a directory into a pandas DataFrame.

    Args:
        data_dir: path to directory containing parquet files
        symbols: optional list of symbols to filter (e.g. ["BTC", "ETH"])
        start: optional start timestamp (ISO format or nanosecond int)
        end: optional end timestamp (ISO format or nanosecond int)
        start_date: optional start date for directory-level filtering
            (e.g. "2026-05-10"). Files in directories before this date
            are skipped without being read.
        end_date: optional end date for directory-level filtering
            (e.g. "2026-05-15"). Files in directories after this date
            are skipped without being read.
        columns: optional list of specific columns to load (reduces memory)
        glob_pattern: glob pattern for finding parquet files
        max_rows: optional maximum number of rows to load (for sampling)
        max_memory_mb: optional memory limit in MB. If estimated memory
            exceeds this, only files up to the limit are loaded (most
            recent first). Default: None (no limit).

    Returns:
        pd.DataFrame with all loaded data, sorted by timestamp_ns
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(data_path.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {data_dir} (pattern: {glob_pattern})"
        )

    # Directory-level date filtering: skip files whose parent directory
    # falls outside [start_date, end_date]. Directories named like
    # "2026-05-12" or "2026-05-12-clean" are parsed; others pass through.
    if start_date is not None or end_date is not None:
        files = _filter_files_by_date(files, start_date=start_date, end_date=end_date)
        if not files:
            raise FileNotFoundError(
                f"No parquet files in date range [{start_date}, {end_date}] "
                f"under {data_dir}"
            )

    # Memory guard: estimate total size from file metadata and cap if needed
    if max_memory_mb is not None:
        files = _apply_memory_limit(files, max_memory_mb=max_memory_mb, columns=columns)

    # Build pyarrow filters
    filters = _build_filters(symbols=symbols, start=start, end=end)

    # Load
    tables = []
    rows_loaded = 0
    for fpath in files:
        try:
            table = pq.read_table(
                str(fpath),
                columns=columns,
                filters=filters if filters else None,
            )
        except (pa.ArrowInvalid, pa.ArrowIOError, OSError) as e:
            # Skip corrupted files but warn
            import warnings
            warnings.warn(f"Skipping {fpath.name}: {e}")
            continue

        if table.num_rows == 0:
            continue

        tables.append(table)
        rows_loaded += table.num_rows

        if max_rows is not None and rows_loaded >= max_rows:
            break

    if not tables:
        raise ValueError(
            f"No data loaded from {data_dir} after filtering "
            f"(symbols={symbols}, start={start}, end={end}, "
            f"start_date={start_date}, end_date={end_date})"
        )

    # Concatenate
    combined = pa.concat_tables(tables, promote_options="default")
    df = combined.to_pandas()

    # Schema compat: warn on version mismatch, pad missing columns
    df = normalize_schema(df, files)

    # Apply max_rows after concatenation if needed
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    # Sort by timestamp
    ts_col = _detect_timestamp_column(df)
    if ts_col is not None:
        df = df.sort_values(ts_col).reset_index(drop=True)

    return df


def load_parquet_lazy(
    data_dir: Union[str, Path],
    *,
    glob_pattern: str = "**/*.parquet",
) -> pq.ParquetDataset:
    """
    Return a PyArrow ParquetDataset for lazy/chunked reading.

    Useful when data is too large to fit in memory.
    Caller can iterate with .to_batches() or read specific columns.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(data_path.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {data_dir} (pattern: {glob_pattern})"
        )

    return pq.ParquetDataset([str(f) for f in files])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_schema(
    df: pd.DataFrame,
    *,
    require_meta: bool = True,
    require_vectors: Optional[List[str]] = None,
    min_rows: int = 1,
) -> Dict:
    """
    Validate a loaded DataFrame against expected schema.

    Returns a dict with:
        - valid: bool
        - errors: list of error strings
        - warnings: list of warning strings
        - row_count: int
        - column_count: int
        - vectors_available: list of vector names with > 0 columns found
        - vectors_complete: list of vector names with 100% coverage
    """
    errors = []
    warnings = []

    # Row count
    if len(df) < min_rows:
        errors.append(f"Too few rows: {len(df)} < {min_rows}")

    # Meta columns
    if require_meta:
        for mc in ["timestamp_ns", "symbol"]:
            if mc not in df.columns:
                errors.append(f"Missing required meta column: {mc}")

    # Vector coverage
    col_set = set(df.columns)
    vectors_available = []
    vectors_complete = []

    check_vectors = require_vectors or list(FEATURE_VECTORS.keys())
    for vname in check_vectors:
        if vname not in FEATURE_VECTORS:
            errors.append(f"Unknown vector: {vname}")
            continue
        expected = FEATURE_VECTORS[vname]["columns"]
        found = [c for c in expected if c in col_set]
        if found:
            vectors_available.append(vname)
        if len(found) == len(expected):
            vectors_complete.append(vname)
        elif require_vectors and vname in require_vectors and not found:
            errors.append(f"Required vector '{vname}' has no columns in data")
        elif found and len(found) < len(expected):
            missing = [c for c in expected if c not in col_set]
            warnings.append(
                f"Vector '{vname}': {len(found)}/{len(expected)} columns "
                f"(missing: {missing[:3]}{'...' if len(missing) > 3 else ''})"
            )

    # Check for NaN-heavy columns
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    for col in feature_cols:
        if df[col].dtype in (np.float64, np.float32, float):
            nan_rate = df[col].isna().mean()
            if nan_rate > 0.95:
                warnings.append(f"Column '{col}' is >95% NaN ({nan_rate:.1%})")
            elif nan_rate > 0.5:
                warnings.append(f"Column '{col}' is >50% NaN ({nan_rate:.1%})")

    # Check for constant columns (zero variance)
    for col in feature_cols:
        if df[col].dtype in (np.float64, np.float32, float):
            if df[col].notna().sum() > 10:
                std = df[col].std()
                if std == 0.0 or (std is not None and np.isclose(std, 0.0)):
                    warnings.append(f"Column '{col}' has zero variance (constant)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "row_count": len(df),
        "column_count": len(df.columns),
        "vectors_available": vectors_available,
        "vectors_complete": vectors_complete,
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_symbols(df: pd.DataFrame) -> List[str]:
    """Return sorted list of unique symbols in the DataFrame."""
    if "symbol" not in df.columns:
        return []
    return sorted(df["symbol"].unique().tolist())


def filter_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return rows for a specific symbol."""
    if "symbol" not in df.columns:
        raise ValueError("DataFrame has no 'symbol' column")
    result = df[df["symbol"] == symbol].copy()
    if result.empty:
        available = get_symbols(df)
        raise ValueError(
            f"No data for symbol '{symbol}'. Available: {available}"
        )
    return result.reset_index(drop=True)


def filter_time_range(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Filter DataFrame by time range using timestamp_ns."""
    ts_col = _detect_timestamp_column(df)
    if ts_col is None:
        raise ValueError("No timestamp column found in DataFrame")

    result = df.copy()

    if start is not None:
        start_ns = _parse_timestamp(start, ts_col, df)
        result = result[result[ts_col] >= start_ns]

    if end is not None:
        end_ns = _parse_timestamp(end, ts_col, df)
        result = result[result[ts_col] <= end_ns]

    return result.reset_index(drop=True)


def get_time_range(df: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
    """Return (min_timestamp_ns, max_timestamp_ns) from the DataFrame."""
    ts_col = _detect_timestamp_column(df)
    if ts_col is None or df.empty:
        return None, None
    return int(df[ts_col].min()), int(df[ts_col].max())


def get_duration_seconds(df: pd.DataFrame) -> float:
    """Return the time span of the data in seconds."""
    t_min, t_max = get_time_range(df)
    if t_min is None or t_max is None:
        return 0.0
    return (t_max - t_min) / 1e9


def list_parquet_files(
    data_dir: Union[str, Path],
    glob_pattern: str = "**/*.parquet",
) -> List[Dict]:
    """
    List parquet files with metadata (row count, size, date range).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(data_path.glob(glob_pattern))
    result = []
    for f in files:
        try:
            meta = pq.read_metadata(str(f))
            result.append({
                "path": str(f),
                "name": f.name,
                "rows": meta.num_rows,
                "columns": meta.num_columns,
                "size_bytes": f.stat().st_size,
                "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
            })
        except Exception as e:
            result.append({
                "path": str(f),
                "name": f.name,
                "error": str(e),
            })
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _parse_dir_date(path: Path) -> Optional[date]:
    """Extract a date from a directory name like '2026-05-12' or '2026-05-12-clean'."""
    m = _DATE_RE.search(path.name)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    # Also check parent (file might be directly under a date dir)
    if path.parent != path:
        m = _DATE_RE.search(path.parent.name)
        if m:
            try:
                return date.fromisoformat(m.group(1))
            except ValueError:
                pass
    return None


def _filter_files_by_date(
    files: List[Path],
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Path]:
    """Filter file list by the date in their parent directory name."""
    sd = date.fromisoformat(start_date) if start_date else None
    ed = date.fromisoformat(end_date) if end_date else None

    result = []
    for f in files:
        d = _parse_dir_date(f)
        if d is None:
            # Can't determine date — include by default
            result.append(f)
            continue
        if sd is not None and d < sd:
            continue
        if ed is not None and d > ed:
            continue
        result.append(f)
    return result


def _apply_memory_limit(
    files: List[Path],
    *,
    max_memory_mb: float,
    columns: Optional[List[str]] = None,
) -> List[Path]:
    """
    Estimate in-memory size from parquet metadata and drop files if over limit.

    Keeps files in order (oldest first), dropping from the front if over budget
    so the most recent data is preserved.
    """
    max_bytes = max_memory_mb * 1024 * 1024
    # Estimate bytes per row: 8 bytes per float64 column
    BYTES_PER_CELL = 8

    file_info = []  # (path, estimated_bytes)
    for f in files:
        try:
            meta = pq.read_metadata(str(f))
            n_cols = len(columns) if columns else meta.num_columns
            est = meta.num_rows * n_cols * BYTES_PER_CELL
            file_info.append((f, est))
        except Exception:
            # Corrupt — will be skipped later, include with 0 estimate
            file_info.append((f, 0))

    total_est = sum(est for _, est in file_info)
    if total_est <= max_bytes:
        return files

    import warnings
    warnings.warn(
        f"Estimated memory {total_est / 1024**2:.0f} MB exceeds "
        f"limit {max_memory_mb:.0f} MB — loading most recent files only"
    )

    # Keep most recent files (end of sorted list) that fit within budget
    kept = []
    budget = max_bytes
    for fpath, est in reversed(file_info):
        if budget - est < 0 and kept:
            break
        kept.append(fpath)
        budget -= est
    kept.reverse()
    return kept


def _detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Detect which timestamp column is present."""
    for candidate in ["timestamp_ns", "timestamp", "datetime", "time"]:
        if candidate in df.columns:
            return candidate
    return None


def _parse_timestamp(value: str, ts_col: str, df: pd.DataFrame) -> int:
    """Parse a timestamp string to nanoseconds (for filtering)."""
    # If it looks like a nanosecond integer
    try:
        return int(value)
    except (ValueError, TypeError):
        pass

    # Parse ISO string to nanoseconds
    ts = pd.Timestamp(value)
    return int(ts.value)  # pandas Timestamp.value is nanoseconds


def _build_filters(
    *,
    symbols: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[List]:
    """Build pyarrow filter expressions for predicate pushdown."""
    filters = []

    if symbols is not None and len(symbols) > 0:
        if len(symbols) == 1:
            filters.append(("symbol", "=", symbols[0]))
        else:
            filters.append(("symbol", "in", symbols))

    if start is not None:
        try:
            start_ns = int(start)
        except (ValueError, TypeError):
            start_ns = int(pd.Timestamp(start).value)
        filters.append(("timestamp_ns", ">=", start_ns))

    if end is not None:
        try:
            end_ns = int(end)
        except (ValueError, TypeError):
            end_ns = int(pd.Timestamp(end).value)
        filters.append(("timestamp_ns", "<=", end_ns))

    return filters if filters else None
