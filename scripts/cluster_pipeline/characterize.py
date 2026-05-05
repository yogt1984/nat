"""
State characterization for NAT profiling pipeline.

Phase 5: Centroid profiling and entry/exit signatures.

Usage:
    from cluster_pipeline.characterize import characterize_states

    profiles = characterize_states(derivatives, hierarchy, transition_model)
    for state_id, profile in profiles.items():
        print(f"State {state_id}: {profile.top_elevated[:3]}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cluster_pipeline.hierarchy import HierarchicalLabels
from cluster_pipeline.transitions import TransitionModel

logger = logging.getLogger(__name__)


@dataclass
class StateProfile:
    """Complete characterization of a single hierarchical state."""

    state_id: int  # global micro state ID
    regime_id: int  # parent macro regime
    local_state_id: int  # local micro state within regime
    n_bars: int  # number of bars in this state
    centroid: Dict[str, float]  # mean derivative value per column
    top_elevated: List[Tuple[str, float]]  # (column, z-score) most above global mean
    top_suppressed: List[Tuple[str, float]]  # (column, z-score) most below global mean
    duration_mean: float
    duration_median: float
    duration_p90: float
    successor_probs: Dict[int, float]  # state_id → transition probability


def characterize_states(
    derivatives: pd.DataFrame,
    hierarchy: HierarchicalLabels,
    transition_model: TransitionModel,
    top_n: int = 10,
) -> Dict[int, StateProfile]:
    """
    Characterize each hierarchical state by its centroid, defining features,
    duration statistics, and transition probabilities.

    For each global micro state:
      1. Compute centroid (mean of derivatives where label == state)
      2. Compute z-score of centroid vs global mean/std
      3. Rank features by z-score: top elevated (positive) and suppressed (negative)
      4. Extract duration statistics from the transition model
      5. Extract successor probabilities from transition matrix row

    Args:
        derivatives: DataFrame of derivative columns (same rows as labels).
        hierarchy: HierarchicalLabels from assemble_hierarchy().
        transition_model: TransitionModel from empirical_transitions() on micro_labels.
        top_n: Number of top features to include in elevated/suppressed lists.

    Returns:
        Dict mapping global micro state ID → StateProfile.

    Raises:
        ValueError: if derivatives length != label length.
    """
    n_bars = len(derivatives)
    if n_bars != len(hierarchy.micro_labels):
        raise ValueError(
            f"derivatives rows ({n_bars}) != "
            f"micro_labels length ({len(hierarchy.micro_labels)})"
        )

    if n_bars == 0:
        raise ValueError("derivatives is empty")

    micro_labels = hierarchy.micro_labels
    columns = derivatives.columns.tolist()

    # Global statistics for z-score computation
    global_mean = derivatives.mean()
    global_std = derivatives.std()
    # Avoid division by zero/near-zero for constant columns
    # Use a relative threshold: if std < 1e-10 * |mean| or std < 1e-12
    threshold = np.maximum(np.abs(global_mean) * 1e-10, 1e-12)
    global_std[global_std < threshold] = 1.0

    # Build profiles for each state
    profiles: Dict[int, StateProfile] = {}

    for state_id in range(hierarchy.n_micro_total):
        mask = micro_labels == state_id
        n_state = int(np.sum(mask))

        regime_id, local_state_id = hierarchy.label_map[state_id]

        # ----- Centroid -----
        state_data = derivatives.loc[mask]
        centroid_series = state_data.mean()
        centroid = {col: float(centroid_series[col]) for col in columns}

        # ----- Z-scores (centroid vs global) -----
        z_scores = (centroid_series - global_mean) / global_std

        # ----- Top elevated (highest z-score) -----
        # Use threshold to exclude near-zero z-scores from floating point noise
        z_threshold = 1e-10
        sorted_z = z_scores.sort_values(ascending=False)
        top_elevated = [
            (col, float(val)) for col, val in sorted_z.head(top_n).items()
            if val > z_threshold
        ]

        # ----- Top suppressed (lowest z-score) -----
        sorted_z_asc = z_scores.sort_values(ascending=True)
        top_suppressed = [
            (col, float(val)) for col, val in sorted_z_asc.head(top_n).items()
            if val < -z_threshold
        ]

        # ----- Duration statistics -----
        if state_id in transition_model.duration_distributions:
            durs = transition_model.duration_distributions[state_id]
            if len(durs) > 0:
                duration_mean = float(np.mean(durs))
                duration_median = float(np.median(durs))
                duration_p90 = float(np.percentile(durs, 90))
            else:
                duration_mean = 0.0
                duration_median = 0.0
                duration_p90 = 0.0
        else:
            duration_mean = float(n_state) if n_state > 0 else 0.0
            duration_median = float(n_state) if n_state > 0 else 0.0
            duration_p90 = float(n_state) if n_state > 0 else 0.0

        # ----- Successor probabilities -----
        successor_probs: Dict[int, float] = {}
        # Map state_id to matrix index in transition model
        unique_states = sorted(transition_model.duration_distributions.keys())
        if state_id in unique_states:
            idx = unique_states.index(state_id)
            row = transition_model.matrix[idx, :]
            for j, target_state in enumerate(unique_states):
                if target_state != state_id and row[j] > 0:
                    successor_probs[target_state] = float(row[j])

        profiles[state_id] = StateProfile(
            state_id=state_id,
            regime_id=regime_id,
            local_state_id=local_state_id,
            n_bars=n_state,
            centroid=centroid,
            top_elevated=top_elevated,
            top_suppressed=top_suppressed,
            duration_mean=duration_mean,
            duration_median=duration_median,
            duration_p90=duration_p90,
            successor_probs=successor_probs,
        )

    return profiles
