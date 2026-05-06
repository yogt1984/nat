"""
EAMM Module 2: Optimal Spread Label Generator

Given the PnL matrix from the simulator, computes the optimal spread
label for each timestamp — i.e., which spread level maximized realized PnL.

Reference: EAMM_SPEC.md §1.7
"""

import numpy as np
import polars as pl
from typing import List, Optional
from .simulator import SimulationResult, pnl_to_bps


def compute_labels(result: SimulationResult) -> pl.DataFrame:
    """Compute optimal spread labels from simulation results.

    For each row t:
      y(t) = argmax_k PnL(t, delta_k)

    Parameters
    ----------
    result : SimulationResult
        Output from simulate_mm().

    Returns
    -------
    pl.DataFrame with columns:
        - timestamp_ns
        - optimal_spread_class : int (index into spread_levels_bps)
        - optimal_spread_bps : float (the spread value at that index)
        - pnl_at_optimal_bps : float (PnL in bps at optimal spread)
        - pnl_level_0 .. pnl_level_K-1 : float (PnL in bps at each level)
    """
    pnl_bps = pnl_to_bps(result)
    N, K = pnl_bps.shape
    spreads = result.spread_levels_bps

    # Find optimal class per row (argmax, ignoring NaN)
    # For rows that are all NaN (beyond valid horizon), set to -1
    optimal_class = np.full(N, -1, dtype=np.int32)
    optimal_spread = np.full(N, np.nan)
    pnl_at_optimal = np.full(N, np.nan)

    valid_mask = ~np.isnan(pnl_bps[:, 0])

    # For valid rows, find argmax
    valid_pnl = pnl_bps[valid_mask]
    optimal_class[valid_mask] = np.argmax(valid_pnl, axis=1).astype(np.int32)
    optimal_spread[valid_mask] = np.array(spreads)[optimal_class[valid_mask]]
    pnl_at_optimal[valid_mask] = np.take_along_axis(
        valid_pnl, optimal_class[valid_mask, np.newaxis], axis=1
    ).ravel()

    # Build output DataFrame
    data = {
        "timestamp_ns": result.timestamps,
        "optimal_spread_class": optimal_class,
        "optimal_spread_bps": optimal_spread,
        "pnl_at_optimal_bps": pnl_at_optimal,
    }
    for k, s in enumerate(spreads):
        data[f"pnl_level_{k}"] = pnl_bps[:, k]

    df = pl.DataFrame(data)
    # Drop invalid rows (beyond horizon)
    df = df.filter(pl.col("optimal_spread_class") >= 0)
    return df


def compute_continuous_optimal(result: SimulationResult) -> np.ndarray:
    """Estimate continuous optimal spread via quadratic interpolation.

    For each row, fits a quadratic to (spread, PnL) pairs and finds the
    analytic maximum. Falls back to discrete argmax if fit is degenerate.

    Returns
    -------
    np.ndarray of shape (N_valid,) with optimal spread in bps.
    """
    pnl_bps = pnl_to_bps(result)
    spreads = np.array(result.spread_levels_bps)
    N, K = pnl_bps.shape

    valid_mask = ~np.isnan(pnl_bps[:, 0])
    valid_pnl = pnl_bps[valid_mask]
    N_valid = valid_pnl.shape[0]

    continuous_optimal = np.zeros(N_valid)

    for i in range(N_valid):
        pnl_row = valid_pnl[i]
        # Fit quadratic: PnL = a*delta^2 + b*delta + c
        try:
            coeffs = np.polyfit(spreads, pnl_row, deg=2)
            a, b, c = coeffs
            if a < 0:
                # Concave — maximum at -b/(2a)
                opt = -b / (2.0 * a)
                # Clamp to [min_spread, max_spread]
                opt = np.clip(opt, spreads[0], spreads[-1])
                continuous_optimal[i] = opt
            else:
                # Convex or flat — use discrete argmax
                continuous_optimal[i] = spreads[np.argmax(pnl_row)]
        except (np.linalg.LinAlgError, ValueError):
            continuous_optimal[i] = spreads[np.argmax(pnl_row)]

    return continuous_optimal


def label_distribution(labels_df: pl.DataFrame, n_classes: int) -> dict:
    """Compute distribution statistics of optimal spread labels.

    Returns dict with:
        - counts: list of count per class
        - fractions: list of fraction per class
        - entropy: Shannon entropy of the distribution
        - is_degenerate: True if >90% in one class
    """
    classes = labels_df["optimal_spread_class"].to_numpy()
    counts = np.bincount(classes, minlength=n_classes)
    total = counts.sum()
    fractions = counts / total if total > 0 else counts.astype(float)

    # Shannon entropy
    nonzero = fractions[fractions > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    max_entropy = np.log(n_classes)

    return {
        "counts": counts.tolist(),
        "fractions": fractions.tolist(),
        "entropy": float(entropy),
        "normalized_entropy": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
        "is_degenerate": bool(np.max(fractions) > 0.9),
    }
