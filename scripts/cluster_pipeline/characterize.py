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


# ---------------------------------------------------------------------------
# Task 5.2: Entry and Exit Signatures
# ---------------------------------------------------------------------------


@dataclass
class TransitionSignature:
    """Average derivative trajectory around state entry/exit."""

    state_id: int
    entry_trajectory: pd.DataFrame  # shape (lookback, n_columns) — mean trajectory
    exit_trajectory: pd.DataFrame  # shape (lookback, n_columns) — mean trajectory
    entry_count: int  # number of entry events used
    exit_count: int  # number of exit events used
    entry_std: pd.DataFrame  # shape (lookback, n_columns) — std across events
    exit_std: pd.DataFrame  # shape (lookback, n_columns) — std across events


def compute_signatures(
    derivatives: pd.DataFrame,
    labels: np.ndarray,
    state_id: int,
    lookback: int = 5,
    min_events: int = 5,
) -> Optional[TransitionSignature]:
    """
    Compute average derivative trajectory before state entry and after exit.

    Entry signature: for each time t where label[t]==state_id and
    label[t-1]!=state_id, collect derivatives[t-lookback:t] (the approach).

    Exit signature: for each time t where label[t]==state_id and
    label[t+1]!=state_id, collect derivatives[t+1:t+1+lookback] (the departure).

    Args:
        derivatives: DataFrame of derivative columns.
        labels: 1-D array of state labels (same length as derivatives).
        state_id: Which state to compute signatures for.
        lookback: Number of bars before entry / after exit to capture.
        min_events: Minimum number of entry/exit events required.
            Returns None if both entry_count and exit_count < min_events.

    Returns:
        TransitionSignature if sufficient events, None otherwise.

    Raises:
        ValueError: if derivatives and labels have different lengths,
            or if lookback < 1.
    """
    labels = np.asarray(labels)

    if len(derivatives) != len(labels):
        raise ValueError(
            f"derivatives rows ({len(derivatives)}) != "
            f"labels length ({len(labels)})"
        )

    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")

    n = len(labels)
    columns = derivatives.columns.tolist()

    # ----- Find entry points -----
    # Entry at t: labels[t] == state_id AND (t==0 OR labels[t-1] != state_id)
    entry_indices = []
    for t in range(n):
        if labels[t] == state_id:
            if t == 0 or labels[t - 1] != state_id:
                entry_indices.append(t)

    # ----- Find exit points -----
    # Exit at t: labels[t] == state_id AND (t==n-1 OR labels[t+1] != state_id)
    exit_indices = []
    for t in range(n):
        if labels[t] == state_id:
            if t == n - 1 or labels[t + 1] != state_id:
                exit_indices.append(t)

    # ----- Collect entry trajectories (lookback bars BEFORE entry) -----
    entry_windows = []
    for t in entry_indices:
        start = t - lookback
        if start >= 0:
            window = derivatives.iloc[start:t].values
            if window.shape[0] == lookback:
                entry_windows.append(window)

    # ----- Collect exit trajectories (lookback bars AFTER exit) -----
    exit_windows = []
    for t in exit_indices:
        end = t + 1 + lookback
        if end <= n:
            window = derivatives.iloc[t + 1:end].values
            if window.shape[0] == lookback:
                exit_windows.append(window)

    entry_count = len(entry_windows)
    exit_count = len(exit_windows)

    # Check minimum events
    if entry_count < min_events and exit_count < min_events:
        return None

    # ----- Compute mean and std trajectories -----
    if entry_count > 0:
        entry_stack = np.stack(entry_windows, axis=0)  # (n_events, lookback, n_cols)
        entry_mean = np.mean(entry_stack, axis=0)
        entry_std = np.std(entry_stack, axis=0)
    else:
        entry_mean = np.full((lookback, len(columns)), np.nan)
        entry_std = np.full((lookback, len(columns)), np.nan)

    if exit_count > 0:
        exit_stack = np.stack(exit_windows, axis=0)
        exit_mean = np.mean(exit_stack, axis=0)
        exit_std = np.std(exit_stack, axis=0)
    else:
        exit_mean = np.full((lookback, len(columns)), np.nan)
        exit_std = np.full((lookback, len(columns)), np.nan)

    # Build DataFrames with relative time index
    entry_trajectory = pd.DataFrame(
        entry_mean, columns=columns,
        index=range(-lookback, 0),
    )
    exit_trajectory = pd.DataFrame(
        exit_mean, columns=columns,
        index=range(1, lookback + 1),
    )
    entry_std_df = pd.DataFrame(
        entry_std, columns=columns,
        index=range(-lookback, 0),
    )
    exit_std_df = pd.DataFrame(
        exit_std, columns=columns,
        index=range(1, lookback + 1),
    )

    return TransitionSignature(
        state_id=state_id,
        entry_trajectory=entry_trajectory,
        exit_trajectory=exit_trajectory,
        entry_count=entry_count,
        exit_count=exit_count,
        entry_std=entry_std_df,
        exit_std=exit_std_df,
    )
