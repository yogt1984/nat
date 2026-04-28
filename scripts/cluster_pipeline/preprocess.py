"""
Bar aggregation and preprocessing for NAT cluster analysis pipeline.

Converts raw 100ms tick-level feature data into time bars at multiple
timeframes (5m, 15m, 1h, 4h), then applies NaN handling, scaling, and
feature filtering to produce matrices ready for clustering.

Usage:
    from cluster_pipeline.preprocess import aggregate_bars, preprocess, TIMEFRAMES

    bars = aggregate_bars(df, timeframe="15min")
    X, columns, meta = preprocess(bars, vector="entropy", scaler="zscore")
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import FEATURE_VECTORS, COMPOSITE_VECTORS, META_COLUMNS, get_vector_columns


# ---------------------------------------------------------------------------
# Timeframe definitions
# ---------------------------------------------------------------------------

TIMEFRAMES = {
    "5min": {"label": "5 min", "freq": "5min", "use": "Fine-grained intraday"},
    "15min": {"label": "15 min", "freq": "15min", "use": "Primary analysis horizon"},
    "1h": {"label": "1 hour", "freq": "1h", "use": "Swing-level regimes"},
    "4h": {"label": "4 hour", "freq": "4h", "use": "Macro regime detection"},
}

# Category-specific aggregation overrides.
# Keys are column prefixes. Values are lists of (suffix, agg_func) pairs
# that replace the default aggregations for columns matching that prefix.
#
# Default for most features: mean, std, last
# Overrides below capture domain semantics from the spec:
#   - Price columns: OHLC
#   - Volume / flow count: sum
#   - Entropy: mean + slope (linear regression slope over the bar)
#   - Whale flow: sum (cumulative net flow)

_PRICE_COLUMNS = {"raw_midprice", "raw_microprice", "raw_spread"}

_SUM_PREFIXES = {"flow_volume_", "flow_count_"}

_SUM_COLUMNS = {
    "whale_net_flow_1h",
    "whale_net_flow_4h",
    "whale_net_flow_24h",
    "whale_total_activity",
    "active_whale_count",
    "illiq_trade_count",
    "toxic_trade_count",
    "positions_at_risk_count",
}

_ENTROPY_PREFIX = "ent_"


# ---------------------------------------------------------------------------
# Bar aggregation
# ---------------------------------------------------------------------------


def aggregate_bars(
    df: pd.DataFrame,
    timeframe: str = "15min",
    *,
    timestamp_col: str = "timestamp_ns",
    symbol_col: str = "symbol",
    custom_aggs: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Aggregate 100ms tick data into time bars.

    For each bar window, produces per-feature aggregations:
      - Default features: mean, std, last
      - Price columns (raw_midprice, raw_microprice): open, high, low, close, mean
      - Volume/count columns: sum
      - Entropy columns: mean, std, slope (OLS over bar window)
      - Whale flow: sum

    Output columns follow the pattern: {original_col}_{agg_suffix}
    e.g., ent_tick_1m_mean, ent_tick_1m_slope, raw_midprice_open

    Args:
        df: DataFrame with raw tick data (must have timestamp_ns and feature columns)
        timeframe: one of "5min", "15min", "1h", "4h" (or any pandas freq string)
        timestamp_col: name of the nanosecond timestamp column
        symbol_col: name of the symbol column
        custom_aggs: optional dict mapping column name -> agg function name,
                     overriding the automatic category-based rules

    Returns:
        DataFrame with one row per (symbol, bar_start) and aggregated feature columns.
        Includes meta columns: bar_start, bar_end, symbol, tick_count.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not in DataFrame")

    if df.empty:
        raise ValueError("Cannot aggregate empty DataFrame")

    freq = _resolve_freq(timeframe)

    # Convert nanoseconds to datetime for resampling
    working = df.copy()
    working["_datetime"] = pd.to_datetime(working[timestamp_col], unit="ns")

    # Identify feature columns (everything except meta)
    feature_cols = [c for c in working.columns if c not in META_COLUMNS and c != "_datetime"]

    # Build aggregation dict
    agg_plan = _build_agg_plan(feature_cols, custom_aggs)

    # Group by symbol if present, then resample
    has_symbol = symbol_col in working.columns
    if has_symbol:
        groups = working.set_index("_datetime").groupby(symbol_col)
    else:
        # Single-symbol mode: add dummy grouper
        working["_symbol"] = "UNKNOWN"
        groups = working.set_index("_datetime").groupby("_symbol")

    bar_frames = []
    for sym, group in groups:
        resampled = _resample_group(group, freq, agg_plan, feature_cols)
        resampled["symbol"] = sym
        bar_frames.append(resampled)

    if not bar_frames:
        raise ValueError("No bars produced after aggregation")

    bars = pd.concat(bar_frames, ignore_index=True)

    # Sort by symbol, bar_start
    bars = bars.sort_values(["symbol", "bar_start"]).reset_index(drop=True)

    return bars


def aggregate_multi_timeframe(
    df: pd.DataFrame,
    timeframes: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate to multiple timeframes at once.

    Returns dict mapping timeframe label -> aggregated DataFrame.
    """
    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())

    result = {}
    for tf in timeframes:
        result[tf] = aggregate_bars(df, timeframe=tf, **kwargs)

    return result


# ---------------------------------------------------------------------------
# Preprocessing (NaN handling, scaling, feature filtering)
# ---------------------------------------------------------------------------


def preprocess(
    bars: pd.DataFrame,
    *,
    vector: Optional[str] = None,
    columns: Optional[List[str]] = None,
    scaler: Literal["zscore", "minmax", "robust", "none"] = "zscore",
    nan_threshold: float = 0.5,
    variance_floor: float = 1e-10,
    clip_sigma: Optional[float] = 5.0,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Preprocess aggregated bars into a feature matrix ready for clustering.

    Steps:
      1. Select columns (by vector name or explicit list)
      2. Drop columns with NaN fraction > nan_threshold
      3. Drop columns with near-zero variance
      4. Fill remaining NaN with column median
      5. Optionally clip outliers
      6. Scale features

    Args:
        bars: aggregated bar DataFrame (from aggregate_bars)
        vector: vector name from config (uses bar-aggregated column names with fuzzy matching)
        columns: explicit list of columns to use (overrides vector)
        scaler: scaling method
        nan_threshold: max fraction of NaN allowed per column (0-1)
        variance_floor: minimum variance to keep a column
        clip_sigma: clip values beyond this many std deviations (None to disable)

    Returns:
        (X, column_names, meta_df) where:
          X: numpy array (n_bars, n_features)
          column_names: list of feature column names used
          meta_df: DataFrame with bar_start, bar_end, symbol, tick_count
    """
    meta_cols = [c for c in ["bar_start", "bar_end", "symbol", "tick_count"] if c in bars.columns]

    # Select feature columns
    if columns is not None:
        feat_cols = [c for c in columns if c in bars.columns]
        if not feat_cols:
            raise ValueError(f"None of the specified columns found in bars: {columns[:5]}...")
    elif vector is not None:
        feat_cols = _match_vector_columns(vector, bars.columns.tolist())
        if not feat_cols:
            raise ValueError(
                f"No columns matching vector '{vector}' found in aggregated bars. "
                f"Available columns: {[c for c in bars.columns if c not in meta_cols][:10]}..."
            )
    else:
        # Use all non-meta numeric columns
        feat_cols = [
            c for c in bars.columns
            if c not in meta_cols and bars[c].dtype in (np.float64, np.float32, np.int64, float, int)
        ]

    if not feat_cols:
        raise ValueError("No feature columns found for preprocessing")

    X = bars[feat_cols].copy()

    # Step 1: Drop high-NaN columns
    nan_fracs = X.isna().mean()
    keep_mask = nan_fracs <= nan_threshold
    dropped_nan = [c for c, keep in zip(feat_cols, keep_mask) if not keep]
    X = X.loc[:, keep_mask]
    feat_cols = X.columns.tolist()

    if not feat_cols:
        raise ValueError(
            f"All columns dropped due to NaN threshold ({nan_threshold}). "
            f"Dropped: {dropped_nan[:5]}..."
        )

    # Step 2: Drop near-zero variance columns
    variances = X.var(skipna=True)
    var_mask = variances > variance_floor
    dropped_var = [c for c, keep in zip(feat_cols, var_mask) if not keep]
    X = X.loc[:, var_mask]
    feat_cols = X.columns.tolist()

    if not feat_cols:
        raise ValueError(
            f"All columns dropped due to zero variance. "
            f"Dropped: {dropped_var[:5]}..."
        )

    # Step 3: Fill remaining NaN with column median
    for col in feat_cols:
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X[col] = X[col].fillna(median_val)

    # Step 4: Clip outliers
    if clip_sigma is not None and clip_sigma > 0:
        for col in feat_cols:
            mu = X[col].mean()
            sigma = X[col].std()
            if sigma > variance_floor:
                lower = mu - clip_sigma * sigma
                upper = mu + clip_sigma * sigma
                X[col] = X[col].clip(lower, upper)

    # Step 5: Scale
    X_arr = X.values.astype(np.float64)
    X_arr = _scale(X_arr, method=scaler)

    # Build meta
    meta_df = bars[meta_cols].copy() if meta_cols else pd.DataFrame(index=bars.index)

    return X_arr, feat_cols, meta_df


# ---------------------------------------------------------------------------
# Scaling functions
# ---------------------------------------------------------------------------


def _scale(X: np.ndarray, method: str) -> np.ndarray:
    """Apply column-wise scaling."""
    if method == "none":
        return X

    if method == "zscore":
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)
        stds[stds < 1e-10] = 1.0  # avoid division by zero
        return (X - means) / stds

    if method == "minmax":
        mins = np.nanmin(X, axis=0)
        maxs = np.nanmax(X, axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-10] = 1.0
        return (X - mins) / ranges

    if method == "robust":
        medians = np.nanmedian(X, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-10] = 1.0
        return (X - medians) / iqr

    raise ValueError(f"Unknown scaler: {method}. Use 'zscore', 'minmax', 'robust', or 'none'.")


# ---------------------------------------------------------------------------
# Summary / inspection
# ---------------------------------------------------------------------------


def bar_summary(bars: pd.DataFrame) -> Dict:
    """
    Return summary statistics for an aggregated bar DataFrame.
    """
    meta_cols = {"bar_start", "bar_end", "symbol", "tick_count"}
    feat_cols = [c for c in bars.columns if c not in meta_cols]

    result = {
        "n_bars": len(bars),
        "n_features": len(feat_cols),
        "symbols": sorted(bars["symbol"].unique().tolist()) if "symbol" in bars.columns else [],
        "tick_count_stats": {},
        "nan_fraction": {},
        "time_range": {},
    }

    if "tick_count" in bars.columns:
        result["tick_count_stats"] = {
            "mean": float(bars["tick_count"].mean()),
            "min": int(bars["tick_count"].min()),
            "max": int(bars["tick_count"].max()),
            "median": float(bars["tick_count"].median()),
        }

    if "bar_start" in bars.columns:
        result["time_range"] = {
            "start": str(bars["bar_start"].min()),
            "end": str(bars["bar_end"].max()) if "bar_end" in bars.columns else str(bars["bar_start"].max()),
        }

    # NaN fraction per feature
    for col in feat_cols:
        nan_frac = bars[col].isna().mean()
        if nan_frac > 0:
            result["nan_fraction"][col] = round(float(nan_frac), 4)

    return result


def list_bar_columns(bars: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize bar columns by their aggregation suffix.

    Returns dict: {"mean": [...], "std": [...], "last": [...], ...}
    """
    meta_cols = {"bar_start", "bar_end", "symbol", "tick_count"}
    result: Dict[str, List[str]] = {}

    for col in bars.columns:
        if col in meta_cols:
            continue
        # Find the last underscore — suffix is the agg type
        parts = col.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("mean", "std", "last", "sum", "slope", "open", "high", "low", "close", "min", "max"):
            suffix = parts[1]
        else:
            suffix = "other"
        result.setdefault(suffix, []).append(col)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_freq(timeframe: str) -> str:
    """Resolve a timeframe label to a pandas frequency string."""
    if timeframe in TIMEFRAMES:
        return TIMEFRAMES[timeframe]["freq"]
    # Allow raw pandas freq strings
    try:
        pd.tseries.frequencies.to_offset(timeframe)
        return timeframe
    except ValueError:
        valid = list(TIMEFRAMES.keys())
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. Valid: {valid} or any pandas freq string."
        )


def _build_agg_plan(
    feature_cols: List[str],
    custom_aggs: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build per-column aggregation plan.

    Returns dict: column_name -> [(output_suffix, agg_func_name), ...]
    """
    plan: Dict[str, List[Tuple[str, str]]] = {}
    custom = custom_aggs or {}

    for col in feature_cols:
        if col in custom:
            plan[col] = [("custom", custom[col])]
            continue

        if col in _PRICE_COLUMNS:
            plan[col] = [
                ("open", "first"),
                ("high", "max"),
                ("low", "min"),
                ("close", "last"),
                ("mean", "mean"),
            ]
        elif col in _SUM_COLUMNS or any(col.startswith(p) for p in _SUM_PREFIXES):
            plan[col] = [("sum", "sum")]
        elif col.startswith(_ENTROPY_PREFIX):
            plan[col] = [
                ("mean", "mean"),
                ("std", "std"),
                ("slope", "_slope"),
            ]
        else:
            plan[col] = [
                ("mean", "mean"),
                ("std", "std"),
                ("last", "last"),
            ]

    return plan


def _resample_group(
    group: pd.DataFrame,
    freq: str,
    agg_plan: Dict[str, List[Tuple[str, str]]],
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Resample a single-symbol group into bars.
    """
    resampler = group.resample(freq)

    result_cols = {}

    for col in feature_cols:
        if col not in group.columns:
            continue

        aggs = agg_plan.get(col, [("mean", "mean")])
        series = group[col]

        for suffix, func in aggs:
            out_name = f"{col}_{suffix}"
            if func == "_slope":
                result_cols[out_name] = resampler[col].apply(_linear_slope)
            elif func == "custom":
                # custom_aggs values are valid pandas agg strings
                result_cols[out_name] = resampler[col].agg(func)
            else:
                result_cols[out_name] = resampler[col].agg(func)

    # Tick count per bar
    # Use the first feature column as proxy for tick counting
    if feature_cols and feature_cols[0] in group.columns:
        result_cols["tick_count"] = resampler[feature_cols[0]].count()

    if not result_cols:
        return pd.DataFrame()

    result = pd.DataFrame(result_cols)

    # Drop bars with zero ticks (outside data range)
    if "tick_count" in result.columns:
        result = result[result["tick_count"] > 0]

    # Add bar metadata
    result["bar_start"] = result.index
    result["bar_end"] = result.index + pd.tseries.frequencies.to_offset(freq)

    result = result.reset_index(drop=True)

    return result


def _linear_slope(series: pd.Series) -> float:
    """
    Compute the OLS slope of a series within a bar window.

    Returns 0.0 if the series has fewer than 2 non-NaN values.
    """
    vals = series.dropna()
    n = len(vals)
    if n < 2:
        return 0.0

    x = np.arange(n, dtype=np.float64)
    y = vals.values.astype(np.float64)

    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)

    if denom < 1e-15:
        return 0.0

    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _match_vector_columns(vector_name: str, bar_columns: List[str]) -> List[str]:
    """
    Match config vector columns to aggregated bar column names.

    Bar columns have suffixes like _mean, _std, _last, _slope, etc.
    We match by checking if a bar column starts with any config vector column + "_".
    Uses longest-prefix matching to avoid duplicates when one base column
    name is a prefix of another (e.g. trend_ma_crossover vs trend_ma_crossover_norm).
    """
    base_cols = get_vector_columns(vector_name)
    bar_set = set(bar_columns)
    matched = set()

    for bar_col in bar_columns:
        if bar_col not in bar_set:
            continue
        # Find the longest matching base column
        best_base = None
        for base in base_cols:
            if bar_col.startswith(base + "_"):
                if best_base is None or len(base) > len(best_base):
                    best_base = base
        if best_base is not None:
            matched.add(bar_col)

    # Preserve original column order
    return [c for c in bar_columns if c in matched]
