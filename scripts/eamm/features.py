"""
EAMM Module 3: Context Feature Extractor

Extracts the 19-dimensional information state vector c(t) from NAT parquet data.
These features capture the entropy regime, toxicity, volatility, order flow,
and trend state needed for optimal spread prediction.

Reference: EAMM_SPEC.md §1.5
"""

import numpy as np
import polars as pl
from typing import List, Tuple
import warnings


# The 19 context features, in order
CONTEXT_FEATURES: List[Tuple[str, str, str]] = [
    # (output_name, parquet_column, category)
    ("H_tick_1s",       "ent_tick_1s",                  "entropy"),
    ("H_tick_5s",       "ent_tick_5s",                  "entropy"),
    ("H_tick_30s",      "ent_tick_30s",                 "entropy"),
    ("H_tick_1m",       "ent_tick_1m",                  "entropy"),
    ("H_perm_8",        "ent_permutation_returns_8",    "entropy"),
    ("H_perm_16",       "ent_permutation_returns_16",   "entropy"),
    ("H_perm_32",       "ent_permutation_returns_32",   "entropy"),
    ("VPIN_50",         "toxic_vpin_50",                "toxicity"),
    ("toxic_index",     "toxic_index",                  "toxicity"),
    ("adverse_sel",     "toxic_adverse_selection",       "toxicity"),
    ("sigma_1m",        "vol_returns_1m",               "volatility"),
    ("sigma_5m",        "vol_returns_5m",               "volatility"),
    ("lambda_flow",     "flow_intensity",               "flow"),
    ("aggressor_5s",    "flow_aggressor_ratio_5s",      "flow"),
    ("I_l1",            "imbalance_qty_l1",             "imbalance"),
    ("I_l5",            "imbalance_qty_l5",             "imbalance"),
    ("mom_60",          "trend_momentum_60",            "trend"),
    ("hurst_300",       "trend_hurst_300",              "trend"),
    ("S_bps",           "raw_spread_bps",               "raw"),
]

CONTEXT_FEATURE_NAMES = [name for name, _, _ in CONTEXT_FEATURES]
CONTEXT_PARQUET_COLS = [col for _, col, _ in CONTEXT_FEATURES]
CONTEXT_FEATURE_COUNT = len(CONTEXT_FEATURES)

# Entropy theoretical maximum for tick entropy (3 categories: up/down/neutral)
LN3 = np.log(3.0)


def extract_context(df: pl.DataFrame) -> pl.DataFrame:
    """Extract the 19-dim context vector from NAT parquet data.

    Parameters
    ----------
    df : pl.DataFrame
        Raw NAT parquet data. Must contain timestamp_ns and all 19 source columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: timestamp_ns + 19 context feature columns.
        NaN values are replaced with 0.0 (with warning).
    """
    # Validate all required columns exist
    missing = [col for col in CONTEXT_PARQUET_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {df.columns}"
        )

    # Select and rename
    exprs = [pl.col("timestamp_ns")]
    for out_name, parquet_col, _ in CONTEXT_FEATURES:
        exprs.append(pl.col(parquet_col).alias(out_name))

    result = df.select(exprs)

    # Check and report NaN counts
    nan_counts = {}
    for name in CONTEXT_FEATURE_NAMES:
        n_nan = result[name].is_nan().sum()
        n_null = result[name].is_null().sum()
        total_bad = n_nan + n_null
        if total_bad > 0:
            nan_counts[name] = total_bad

    if nan_counts:
        total_rows = len(result)
        warn_parts = [
            f"{name}: {count} ({count/total_rows*100:.1f}%)"
            for name, count in nan_counts.items()
        ]
        warnings.warn(
            f"NaN/null values found in context features "
            f"(replacing with 0.0): {', '.join(warn_parts)}"
        )
        # Replace NaN and null with 0.0
        fill_exprs = [pl.col("timestamp_ns")]
        for name in CONTEXT_FEATURE_NAMES:
            fill_exprs.append(
                pl.col(name).fill_nan(0.0).fill_null(0.0).alias(name)
            )
        result = result.select(fill_exprs)

    return result


def context_to_numpy(context_df: pl.DataFrame) -> np.ndarray:
    """Convert context DataFrame to numpy array (N, 19).

    Parameters
    ----------
    context_df : pl.DataFrame
        Output of extract_context().

    Returns
    -------
    np.ndarray of shape (N, 19)
    """
    return context_df.select(CONTEXT_FEATURE_NAMES).to_numpy().astype(np.float64)


def validate_context(context_df: pl.DataFrame) -> dict:
    """Validate context features are within expected ranges.

    Returns dict of {feature_name: {issue: str, count: int}} for any violations.
    """
    issues = {}

    for name, _, category in CONTEXT_FEATURES:
        col = context_df[name]
        arr = col.to_numpy()
        valid = ~np.isnan(arr)

        if category == "entropy":
            # Tick entropy should be in [0, ln(3)], permutation in [0, 1]
            if "perm" in name:
                oob = np.sum((arr[valid] < -0.01) | (arr[valid] > 1.01))
                if oob > 0:
                    issues[name] = {
                        "issue": f"{oob} values outside [0, 1]",
                        "count": int(oob),
                    }
            else:
                oob = np.sum((arr[valid] < -0.01) | (arr[valid] > LN3 + 0.01))
                if oob > 0:
                    issues[name] = {
                        "issue": f"{oob} values outside [0, ln(3)]",
                        "count": int(oob),
                    }

        n_nan = np.sum(np.isnan(arr))
        if n_nan > 0:
            issues[name] = {"issue": f"{n_nan} NaN values", "count": int(n_nan)}

    return issues
