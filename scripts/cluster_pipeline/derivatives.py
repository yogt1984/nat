"""
Derivative generation engine for NAT profiling pipeline.

Selects the most informative base features, then generates temporal derivatives
(velocity, acceleration, z-score, rolling volatility) and cross-feature
derivatives (ratios, correlations, divergences) to capture dynamics that raw
feature levels miss.

This addresses the Q3 failure mode: raw features find structural separation
but not predictive states. Derivatives capture *how* features are changing
and *how features relate to each other*, which distinguishes actionable regimes.

Usage:
    from cluster_pipeline.derivatives import (
        select_top_features, temporal_derivatives, cross_feature_derivatives,
    )

    top_cols = select_top_features(bars, vector="entropy", max_features=15)
    td = temporal_derivatives(bars, columns=top_cols, windows=[5, 15, 30])
    cd = cross_feature_derivatives(bars, pairs=DEFAULT_CROSS_PAIRS, windows=[5, 15])
"""

from __future__ import annotations

import fnmatch
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FEATURE_VECTORS, COMPOSITE_VECTORS, get_vector_columns
from .preprocess import _match_vector_columns


def select_top_features(
    bars: pd.DataFrame,
    vector: str,
    max_features: int = 15,
    method: str = "variance",
    variance_floor: float = 1e-10,
) -> List[str]:
    """
    Select the most informative features from a vector in aggregated bars.

    Prevents derivative explosion by selecting only the top-N features
    before generating derivatives. 15 features × 5 derivative types × 3
    windows = 225 derivatives (manageable) vs 191 × 5 × 3 = 2,865 (noise).

    Args:
        bars: aggregated bar DataFrame (from aggregate_bars)
        vector: vector name from config (e.g. "entropy", "orderflow")
        max_features: maximum number of features to return
        method: selection method
            - "variance": top N by variance (most variable = most informative)
            - "autocorrelation_range": top N by range of autocorrelation
              across lags 1-30 (features whose persistence varies most
              across time scales carry regime information)
        variance_floor: minimum variance to consider a feature non-constant

    Returns:
        List of column names, ordered by informativeness (most informative first).
        Length is min(max_features, number of non-constant features).

    Raises:
        ValueError: if vector has no matching columns in bars, or if method is unknown.
    """
    if max_features < 1:
        raise ValueError(f"max_features must be >= 1, got {max_features}")

    # Get candidate columns from this vector in bar-aggregated form
    candidates = _match_vector_columns(vector, bars.columns.tolist())
    if not candidates:
        raise ValueError(
            f"No columns matching vector '{vector}' in bars. "
            f"Available: {[c for c in bars.columns[:10]]}..."
        )

    # Filter to numeric columns only
    candidates = [
        c for c in candidates
        if bars[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
    ]
    if not candidates:
        raise ValueError(f"No numeric columns for vector '{vector}'")

    # Drop constant / near-constant columns
    variances = {c: bars[c].var(skipna=True) for c in candidates}
    candidates = [c for c in candidates if variances[c] > variance_floor]

    if not candidates:
        return []

    if method == "variance":
        return _select_by_variance(candidates, variances, max_features)
    elif method == "autocorrelation_range":
        return _select_by_autocorrelation_range(bars, candidates, max_features)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'variance' or 'autocorrelation_range'."
        )


def _select_by_variance(
    candidates: List[str],
    variances: dict,
    max_features: int,
) -> List[str]:
    """Select top features by variance (descending)."""
    ranked = sorted(candidates, key=lambda c: variances[c], reverse=True)
    return ranked[:max_features]


def _select_by_autocorrelation_range(
    bars: pd.DataFrame,
    candidates: List[str],
    max_features: int,
    max_lag: int = 30,
) -> List[str]:
    """
    Select features by autocorrelation range across lags 1..max_lag.

    Features whose autocorrelation varies most across time scales carry
    the most regime information: a feature with AC(1)=0.95 and AC(30)=0.1
    is more interesting than one with AC(1)=0.5 and AC(30)=0.45.
    """
    ac_ranges = {}
    for col in candidates:
        series = bars[col].dropna()
        if len(series) < max_lag + 10:
            ac_ranges[col] = 0.0
            continue

        values = series.values.astype(np.float64)
        mean = np.mean(values)
        var = np.var(values)
        if var < 1e-15:
            ac_ranges[col] = 0.0
            continue

        # Compute autocorrelation at lags 1..max_lag
        acs = []
        centered = values - mean
        for lag in range(1, min(max_lag + 1, len(values))):
            ac = np.sum(centered[:-lag] * centered[lag:]) / (var * (len(values) - lag))
            acs.append(ac)

        ac_ranges[col] = max(acs) - min(acs) if acs else 0.0

    ranked = sorted(candidates, key=lambda c: ac_ranges[c], reverse=True)
    return ranked[:max_features]


# ---------------------------------------------------------------------------
# Temporal derivative generation
# ---------------------------------------------------------------------------


def temporal_derivatives(
    bars: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [5, 15, 30],
) -> pd.DataFrame:
    """
    Generate temporal derivatives for selected feature columns.

    For each column, computes:
      - Velocity (1st difference): {col}_vel
      - Acceleration (2nd difference): {col}_accel
      - Rolling z-score per window: {col}_zscore_{w}
      - Rolling slope per window: {col}_slope_{w}
      - Rolling volatility per window: {col}_rvol_{w}

    Total output columns = len(columns) * (2 + 3 * len(windows)).

    Args:
        bars: aggregated bar DataFrame
        columns: feature columns to derive (typically from select_top_features)
        windows: rolling window sizes for z-score, slope, and rvol

    Returns:
        DataFrame with derivative columns only (same index/length as bars).
        First max(windows) rows will have NaN for rolling derivatives.
        No rows are dropped — the caller decides how to handle NaN.

    Raises:
        ValueError: if columns is empty or contains columns not in bars.
    """
    if not columns:
        raise ValueError("columns must be non-empty")

    missing = [c for c in columns if c not in bars.columns]
    if missing:
        raise ValueError(f"Columns not in bars: {missing[:5]}")

    if not windows:
        raise ValueError("windows must be non-empty")

    result = {}

    for col in columns:
        series = bars[col].astype(np.float64)

        # Velocity: f(t) - f(t-1)
        vel = series.diff()
        result[f"{col}_vel"] = vel

        # Acceleration: vel(t) - vel(t-1)
        result[f"{col}_accel"] = vel.diff()

        # Window-dependent derivatives
        for w in windows:
            # Rolling z-score: (f(t) - rolling_mean(w)) / rolling_std(w)
            roll_mean = series.rolling(window=w, min_periods=w).mean()
            roll_std = series.rolling(window=w, min_periods=w).std()
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_zscore = (series.values - roll_mean.values) / roll_std.values
            zscore = np.where(roll_std.values < 1e-10, 0.0, raw_zscore)
            # Preserve NaN where rolling stats are NaN
            zscore = np.where(np.isnan(roll_mean.values), np.nan, zscore)
            result[f"{col}_zscore_{w}"] = zscore

            # Rolling slope: OLS slope over window
            result[f"{col}_slope_{w}"] = _rolling_slope(series, w)

            # Rolling volatility: rolling std
            result[f"{col}_rvol_{w}"] = roll_std.values

    out = pd.DataFrame(result, index=bars.index)

    # Replace any inf with 0.0
    out.replace([np.inf, -np.inf], 0.0, inplace=True)

    return out


def _rolling_slope(series: pd.Series, window: int) -> np.ndarray:
    """
    Compute rolling OLS slope over a fixed window.

    Uses the closed-form formula for simple linear regression slope:
        slope = (n * sum(x*y) - sum(x)*sum(y)) / (n * sum(x^2) - sum(x)^2)

    where x = [0, 1, ..., w-1] is fixed for every window position.
    This avoids calling np.polyfit per window (slow).
    """
    n = len(series)
    w = window
    values = series.values.astype(np.float64)
    slopes = np.full(n, np.nan)

    if n < w or w < 2:
        return slopes

    # Precompute constants for x = [0, 1, ..., w-1]
    x = np.arange(w, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = w * sum_x2 - sum_x * sum_x  # constant for all windows

    if abs(denom) < 1e-15:
        return slopes

    # Sliding window via cumsum for sum(y) and sum(x*y)
    # sum(x*y) for window ending at position i:
    #   sum_{j=0}^{w-1} j * values[i-w+1+j]
    # We can compute this efficiently with a weighted rolling sum.
    for i in range(w - 1, n):
        y_window = values[i - w + 1: i + 1]
        if np.any(np.isnan(y_window)):
            continue
        sum_y = y_window.sum()
        sum_xy = np.dot(x, y_window)
        slopes[i] = (w * sum_xy - sum_x * sum_y) / denom

    return slopes


# ---------------------------------------------------------------------------
# Spectral derivative generation
# ---------------------------------------------------------------------------


def spectral_derivatives(
    bars: pd.DataFrame,
    columns: List[str],
    window: int = 30,
) -> pd.DataFrame:
    """
    Generate spectral (frequency-domain) derivatives for selected columns.

    For each column, computes over a rolling window:
      - Low-frequency power (first 1/5 of spectrum)
      - High-frequency power (last 1/5 of spectrum)
      - Spectral ratio (low/high) — high = trending, low = oscillating
      - Dominant period — bars per cycle of strongest frequency

    Args:
        bars: aggregated bar DataFrame.
        columns: feature columns to derive.
        window: FFT window size (must be >= 10).

    Returns:
        DataFrame with spectral columns. First (window-1) rows are NaN.

    Raises:
        ValueError: if columns is empty or window < 10.
    """
    if not columns:
        raise ValueError("columns must be non-empty")
    if window < 10:
        raise ValueError(f"spectral window must be >= 10, got {window}")

    n = len(bars)
    result = {}

    for col in columns:
        if col not in bars.columns:
            continue

        series = bars[col].values.astype(np.float64)

        spec_low = np.full(n, np.nan)
        spec_high = np.full(n, np.nan)
        spec_ratio = np.full(n, np.nan)
        spec_period = np.full(n, np.nan)

        low_cutoff = max(1, window // 5)
        high_start = max(1, 4 * window // 5)
        n_freq = window // 2 + 1

        for t in range(window - 1, n):
            seg = series[t - window + 1: t + 1]
            if np.any(np.isnan(seg)):
                continue

            # Detrend (remove mean) before FFT
            seg = seg - seg.mean()

            # One-sided power spectrum
            X = np.fft.rfft(seg)
            power = np.abs(X) ** 2

            # Low-frequency power (excluding DC at index 0)
            p_low = power[1:low_cutoff].sum() if low_cutoff > 1 else 0.0
            # High-frequency power
            p_high = power[high_start:].sum() if high_start < n_freq else 0.0

            spec_low[t] = p_low
            spec_high[t] = p_high
            spec_ratio[t] = p_low / (p_high + 1e-10)

            # Dominant period: only if there's a clear peak
            if len(power) > 1:
                power_no_dc = power[1:]
                if power_no_dc.max() > 2.0 * power_no_dc.mean():
                    peak_idx = int(np.argmax(power_no_dc)) + 1  # +1 for DC offset
                    spec_period[t] = window / peak_idx
                # else: no dominant frequency → NaN

        result[f"{col}_spec_low_{window}"] = spec_low
        result[f"{col}_spec_high_{window}"] = spec_high
        result[f"{col}_spec_ratio_{window}"] = spec_ratio
        result[f"{col}_spec_period_{window}"] = spec_period

    out = pd.DataFrame(result, index=bars.index)
    out.replace([np.inf, -np.inf], 0.0, inplace=True)
    return out


# ---------------------------------------------------------------------------
# Cross-feature derivative generation
# ---------------------------------------------------------------------------

# Default cross-feature pairs encoding economic hypotheses about feature
# relationships. Glob patterns match against bar-aggregated column names.
DEFAULT_CROSS_PAIRS: List[Dict] = [
    {"a": "ent_*_mean",           "b": "vol_*_mean",              "ops": ["ratio", "corr"]},
    {"a": "imbalance_*_mean",     "b": "raw_spread_*",            "ops": ["ratio"]},
    {"a": "whale_*_sum",          "b": "flow_volume_*_sum",       "ops": ["ratio"]},
    {"a": "toxic_*_mean",         "b": "illiq_*_mean",            "ops": ["ratio", "corr"]},
    {"a": "ent_*_mean",           "b": "trend_*_mean",            "ops": ["corr", "divergence"]},
]


def cross_feature_derivatives(
    bars: pd.DataFrame,
    pairs: List[Dict],
    windows: List[int] = [5, 15, 30],
    ratio_clip: float = 100.0,
    ratio_eps: float = 1e-10,
) -> pd.DataFrame:
    """
    Compute cross-feature derivatives between economically meaningful pairs.

    For each pair, resolves glob patterns to actual columns, then computes:
      - Ratio:       a / (b + eps), clipped to [-clip, clip]
      - Correlation:  rolling Pearson correlation over each window
      - Divergence:   zscore(a, w) - zscore(b, w) for each window

    Pairs that cannot be resolved (no matching columns) are skipped with a
    warning, not an error.

    Args:
        bars: aggregated bar DataFrame
        pairs: list of pair dicts, each with keys "a", "b", "ops".
            "a"/"b" are glob patterns or exact column names.
            "ops" is a list from {"ratio", "corr", "divergence"}.
        windows: rolling window sizes for correlation and divergence
        ratio_clip: absolute max for ratio values
        ratio_eps: epsilon added to denominator in ratio

    Returns:
        DataFrame with cross-feature derivative columns (same length as bars).
        Empty DataFrame (0 columns) if no pairs resolve.
    """
    if not windows:
        raise ValueError("windows must be non-empty")

    result = {}
    bar_cols = bars.columns.tolist()

    for pair in pairs:
        a_pattern = pair.get("a", "")
        b_pattern = pair.get("b", "")
        ops = pair.get("ops", [])

        if not ops:
            continue

        # Resolve glob patterns to actual columns
        a_cols = _resolve_glob(a_pattern, bar_cols)
        b_cols = _resolve_glob(b_pattern, bar_cols)

        if not a_cols:
            warnings.warn(f"Cross-feature pair: no columns match pattern '{a_pattern}'")
            continue
        if not b_cols:
            warnings.warn(f"Cross-feature pair: no columns match pattern '{b_pattern}'")
            continue

        # Use first matching column from each side
        a_col = a_cols[0]
        b_col = b_cols[0]

        a_series = bars[a_col].astype(np.float64)
        b_series = bars[b_col].astype(np.float64)

        # Short names for output columns
        a_short = _shorten_col(a_col)
        b_short = _shorten_col(b_col)
        prefix = f"cross_{a_short}_{b_short}"

        if "ratio" in ops:
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_ratio = a_series.values / (b_series.values + ratio_eps)
            # Where b is NaN, ratio should be NaN
            raw_ratio = np.where(np.isnan(b_series.values) | np.isnan(a_series.values),
                                 np.nan, raw_ratio)
            clipped = np.clip(raw_ratio, -ratio_clip, ratio_clip)
            # Replace any remaining inf
            clipped = np.where(np.isinf(clipped), 0.0, clipped)
            result[f"{prefix}_ratio"] = clipped

        if "corr" in ops:
            for w in windows:
                corr = a_series.rolling(window=w, min_periods=w).corr(b_series)
                result[f"{prefix}_corr_{w}"] = corr.values

        if "divergence" in ops:
            for w in windows:
                a_mean = a_series.rolling(window=w, min_periods=w).mean()
                a_std = a_series.rolling(window=w, min_periods=w).std()
                b_mean = b_series.rolling(window=w, min_periods=w).mean()
                b_std = b_series.rolling(window=w, min_periods=w).std()

                with np.errstate(divide="ignore", invalid="ignore"):
                    a_z = (a_series.values - a_mean.values) / a_std.values
                    b_z = (b_series.values - b_mean.values) / b_std.values

                # Guard zero-std: zscore=0 when std<eps
                a_z = np.where(a_std.values < 1e-10, 0.0, a_z)
                b_z = np.where(b_std.values < 1e-10, 0.0, b_z)

                # Preserve NaN during warmup
                a_z = np.where(np.isnan(a_mean.values), np.nan, a_z)
                b_z = np.where(np.isnan(b_mean.values), np.nan, b_z)

                result[f"{prefix}_div_{w}"] = a_z - b_z

    out = pd.DataFrame(result, index=bars.index)

    # Final safety: replace any inf
    out.replace([np.inf, -np.inf], 0.0, inplace=True)

    return out


def _resolve_glob(pattern: str, columns: List[str]) -> List[str]:
    """
    Resolve a glob pattern against column names.

    If the pattern is an exact match, returns [pattern].
    Otherwise uses fnmatch to find all matching columns.
    """
    if pattern in columns:
        return [pattern]
    return [c for c in columns if fnmatch.fnmatch(c, pattern)]


def _shorten_col(col: str) -> str:
    """
    Shorten a bar column name for use in cross-derivative naming.

    Strips common suffixes (_mean, _std, _last, _sum, _slope) to keep
    output column names readable. E.g. "ent_tick_1s_mean" → "ent_tick_1s".
    """
    for suffix in ("_mean", "_std", "_last", "_sum", "_slope",
                    "_open", "_high", "_low", "_close"):
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


# ---------------------------------------------------------------------------
# Derivative orchestrator
# ---------------------------------------------------------------------------


@dataclass
class DerivativeResult:
    """Output of the generate_derivatives orchestrator."""

    derivatives: pd.DataFrame
    n_base_features: int
    base_features: List[str]
    n_temporal: int
    n_cross: int
    n_total: int
    warmup_rows: int
    metadata: Dict = field(default_factory=dict)


def generate_derivatives(
    bars: pd.DataFrame,
    vector: str,
    max_base_features: int = 15,
    selection_method: str = "variance",
    temporal_windows: List[int] = [5, 15, 30],
    cross_pairs: Optional[List[Dict]] = None,
    cross_windows: Optional[List[int]] = None,
    include_spectral: bool = True,
    spectral_window: int = 30,
) -> DerivativeResult:
    """
    Single entry point for the full derivative generation pipeline.

    Runs:
      1. select_top_features() — pick the most informative base features
      2. temporal_derivatives() — velocity, acceleration, z-score, slope, rvol
      3. cross_feature_derivatives() — ratios, correlations, divergences
      4. spectral_derivatives() — frequency-domain features (if include_spectral)
      5. Concatenate into a single DataFrame

    Args:
        bars: aggregated bar DataFrame (from aggregate_bars)
        vector: feature vector name (e.g. "entropy", "orderflow")
        max_base_features: max features to select before derivation
        selection_method: "variance" or "autocorrelation_range"
        temporal_windows: window sizes for temporal derivatives
        cross_pairs: cross-feature pair definitions (default: DEFAULT_CROSS_PAIRS)
        cross_windows: window sizes for cross derivatives (default: temporal_windows)
        include_spectral: if True, include spectral (FFT) derivatives
        spectral_window: window size for spectral computation

    Returns:
        DerivativeResult with combined derivative DataFrame and metadata.

    Raises:
        ValueError: if vector has no matching columns in bars.
    """
    if cross_pairs is None:
        cross_pairs = DEFAULT_CROSS_PAIRS
    if cross_windows is None:
        cross_windows = temporal_windows

    # Step 1: Feature selection
    base_features = select_top_features(
        bars, vector=vector, max_features=max_base_features, method=selection_method,
    )

    if not base_features:
        return DerivativeResult(
            derivatives=pd.DataFrame(index=bars.index),
            n_base_features=0,
            base_features=[],
            n_temporal=0,
            n_cross=0,
            n_total=0,
            warmup_rows=0,
        )

    # Step 2: Temporal derivatives
    td = temporal_derivatives(bars, columns=base_features, windows=temporal_windows)

    # Step 3: Cross-feature derivatives (warnings suppressed for unresolvable pairs)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        cd = cross_feature_derivatives(
            bars, pairs=cross_pairs, windows=cross_windows,
        )

    # Step 4: Spectral derivatives (optional)
    n_spectral = 0
    if include_spectral and len(bars) >= spectral_window:
        sd = spectral_derivatives(bars, columns=base_features, window=spectral_window)
        n_spectral = sd.shape[1]
    else:
        sd = pd.DataFrame(index=bars.index)

    # Step 5: Concatenate
    parts = [td]
    if cd.shape[1] > 0:
        parts.append(cd)
    if sd.shape[1] > 0:
        parts.append(sd)
    combined = pd.concat(parts, axis=1)

    # Compute warmup: largest temporal window determines when all derivatives are valid
    warmup = max(temporal_windows) if temporal_windows else 0
    if include_spectral:
        warmup = max(warmup, spectral_window)

    return DerivativeResult(
        derivatives=combined,
        n_base_features=len(base_features),
        base_features=base_features,
        n_temporal=td.shape[1],
        n_cross=cd.shape[1],
        n_total=combined.shape[1],
        warmup_rows=warmup,
        metadata={
            "vector": vector,
            "selection_method": selection_method,
            "temporal_windows": temporal_windows,
            "cross_windows": cross_windows,
            "n_cross_pairs_attempted": len(cross_pairs),
            "n_cross_pairs_resolved": cd.shape[1] > 0,
            "include_spectral": include_spectral,
            "n_spectral": n_spectral,
        },
    )
