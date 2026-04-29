"""
Dimensionality reduction pipeline for NAT profiling system.

Pre-PCA filtering: removes near-constant and highly correlated derivative
columns to prevent PCA from wasting components on noise or redundancy.

This module sits between the derivative engine (Phase 1) and PCA (Task 2.2).
The derivative engine produces ~150-200 columns; this module reduces that to
a cleaner set before eigendecomposition.

Usage:
    from cluster_pipeline.reduction import filter_derivatives

    filtered_df, report = filter_derivatives(derivatives_df)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def filter_derivatives(
    X: pd.DataFrame,
    variance_percentile: float = 10.0,
    correlation_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove near-constant and redundant derivative columns before PCA.

    Steps:
      1. Drop columns whose variance is below the variance_percentile-th
         percentile of all column variances (low-information features).
      2. Greedy correlation deduplication: for each pair with |corr| > threshold,
         drop the column with lower variance (keep the more informative one).

    Args:
        X: DataFrame of derivative columns (output of generate_derivatives).
            Must contain only numeric columns. NaN values are filled with 0
            for variance/correlation computation.
        variance_percentile: percentile threshold (0-100). Columns with variance
            below this percentile are dropped. E.g. 10.0 drops the bottom 10%.
        correlation_threshold: absolute correlation above which one column in
            a correlated pair is dropped. Range (0, 1].

    Returns:
        (filtered_df, report) where:
          - filtered_df: DataFrame with surviving columns (same row count as input)
          - report: dict with keys:
              - n_input: original number of columns
              - n_after_variance: columns after variance filtering
              - n_after_correlation: final column count
              - dropped_variance: list of columns dropped for low variance
              - dropped_correlation: list of columns dropped for high correlation
              - variance_threshold_value: the actual variance cutoff used

    Raises:
        ValueError: if X is empty or has no columns, or if parameters are invalid.
    """
    if X.empty or X.shape[1] == 0:
        raise ValueError("Input DataFrame must have at least one column")

    if not (0 <= variance_percentile <= 100):
        raise ValueError(
            f"variance_percentile must be in [0, 100], got {variance_percentile}"
        )

    if not (0 < correlation_threshold <= 1.0):
        raise ValueError(
            f"correlation_threshold must be in (0, 1], got {correlation_threshold}"
        )

    n_input = X.shape[1]

    # Work on a copy, fill NaN for computation
    work = X.fillna(0.0)

    # ----- Step 1: Variance filtering -----
    variances = work.var()
    threshold_value = np.percentile(variances.values, variance_percentile)

    # Keep columns above the threshold.
    # Always drop effectively-zero-variance columns (< 1e-20) regardless of percentile.
    VARIANCE_FLOOR = 1e-20
    if threshold_value < VARIANCE_FLOOR:
        keep_mask = variances > VARIANCE_FLOOR
    else:
        keep_mask = (variances >= threshold_value) & (variances > VARIANCE_FLOOR)

    dropped_variance = variances.index[~keep_mask].tolist()
    surviving_cols = variances.index[keep_mask].tolist()

    if not surviving_cols:
        # All columns dropped — return empty DataFrame
        return pd.DataFrame(index=X.index), {
            "n_input": n_input,
            "n_after_variance": 0,
            "n_after_correlation": 0,
            "dropped_variance": dropped_variance,
            "dropped_correlation": [],
            "variance_threshold_value": float(threshold_value),
        }

    n_after_variance = len(surviving_cols)

    # ----- Step 2: Greedy correlation deduplication -----
    dropped_correlation = _greedy_correlation_filter(
        work[surviving_cols], variances[surviving_cols], correlation_threshold
    )

    final_cols = [c for c in surviving_cols if c not in set(dropped_correlation)]

    # Build output
    filtered_df = X[final_cols].copy()

    report = {
        "n_input": n_input,
        "n_after_variance": n_after_variance,
        "n_after_correlation": len(final_cols),
        "dropped_variance": dropped_variance,
        "dropped_correlation": dropped_correlation,
        "variance_threshold_value": float(threshold_value),
    }

    return filtered_df, report


def _greedy_correlation_filter(
    df: pd.DataFrame,
    variances: pd.Series,
    threshold: float,
) -> List[str]:
    """
    Greedy correlation-based column removal.

    For each pair with |corr| > threshold, drop the column with lower variance.
    Process pairs in descending order of |corr| to remove the most redundant first.

    Returns list of dropped column names.
    """
    if df.shape[1] <= 1:
        return []

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Find all pairs above threshold
    # Use upper triangle to avoid duplicates
    cols = corr_matrix.columns.tolist()
    n = len(cols)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            abs_corr = abs(corr_matrix.iloc[i, j])
            if abs_corr > threshold:
                pairs.append((abs_corr, cols[i], cols[j]))

    if not pairs:
        return []

    # Sort by |corr| descending — remove most redundant first
    pairs.sort(key=lambda x: x[0], reverse=True)

    dropped = set()
    for _, col_a, col_b in pairs:
        # Skip if either already dropped
        if col_a in dropped or col_b in dropped:
            continue

        # Drop the one with lower variance
        if variances[col_a] >= variances[col_b]:
            dropped.add(col_b)
        else:
            dropped.add(col_a)

    return list(dropped)
