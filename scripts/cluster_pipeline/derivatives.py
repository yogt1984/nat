"""
Derivative generation engine for NAT profiling pipeline.

Selects the most informative base features, then generates temporal derivatives
(velocity, acceleration, z-score, rolling volatility) to capture dynamics
that raw feature levels miss.

This addresses the Q3 failure mode: raw features find structural separation
but not predictive states. Derivatives capture *how* features are changing,
which is what distinguishes actionable regimes.

Usage:
    from cluster_pipeline.derivatives import select_top_features

    top_cols = select_top_features(bars, vector="entropy", max_features=15)
"""

from __future__ import annotations

from typing import List, Optional

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
