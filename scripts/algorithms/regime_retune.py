"""Regime-conditional parameter re-estimation utilities.

When the GMM regime detector signals a transition, algorithms should
re-estimate their parameters on the new regime's data window rather
than using stale parameters from the previous regime.

The `regime` column in parquet data is a float (0-5) corresponding to:
  0=Accumulation, 1=Markup, 2=Distribution, 3=Markdown, 4=Ranging, 5=Unknown

Usage in run_batch():
    segments = segment_by_regime(df["regime"].values, min_segment=200)
    for start, end, regime_id in segments:
        params = estimate_on_window(data[start:end])
        run_with_params(data[start:end], params)
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("nat.regime_retune")

# Column name for regime label in parquet data
REGIME_COL = "regime"

# Minimum segment length for re-estimation (too short = noisy estimates)
MIN_SEGMENT_TICKS = 200


def segment_by_regime(
    regime: np.ndarray,
    min_segment: int = MIN_SEGMENT_TICKS,
) -> list[tuple[int, int, int]]:
    """Split a regime label array into contiguous segments.

    Args:
        regime: 1D array of regime labels (float, 0-5). NaN = unknown.
        min_segment: Minimum ticks for a segment to trigger re-estimation.
            Shorter segments inherit the previous segment's parameters.

    Returns:
        List of (start_idx, end_idx, regime_id) tuples.
        Segments shorter than min_segment are merged into the previous one.
    """
    n = len(regime)
    if n == 0:
        return []

    # Quantize to int, NaN → -1
    finite_mask = np.isfinite(regime)
    safe = np.zeros_like(regime)
    safe[finite_mask] = regime[finite_mask]
    labels = np.where(finite_mask, safe.astype(int), -1)

    # Find transition points
    segments: list[tuple[int, int, int]] = []
    seg_start = 0
    current_label = labels[0]

    for i in range(1, n):
        if labels[i] != current_label:
            segments.append((seg_start, i, int(current_label)))
            seg_start = i
            current_label = labels[i]
    segments.append((seg_start, n, int(current_label)))

    # Merge short segments into previous
    if len(segments) <= 1:
        return segments

    merged: list[tuple[int, int, int]] = [segments[0]]
    for start, end, label in segments[1:]:
        length = end - start
        if length < min_segment:
            # Extend previous segment to cover this short one
            prev_start, _, prev_label = merged[-1]
            merged[-1] = (prev_start, end, prev_label)
        else:
            merged.append((start, end, label))

    return merged


def has_regime_column(df) -> bool:
    """Check if DataFrame has a usable regime column."""
    if REGIME_COL not in df.columns:
        return False
    regime = df[REGIME_COL].values
    finite = np.isfinite(regime)
    return finite.sum() > MIN_SEGMENT_TICKS
