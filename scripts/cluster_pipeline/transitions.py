"""
Transition modeling for NAT profiling pipeline.

Phase 4: Empirical transition matrices and duration analysis.

Usage:
    from cluster_pipeline.transitions import empirical_transitions

    model = empirical_transitions(labels)
    print(model.matrix)  # row-stochastic transition matrix
    print(model.most_likely_successor)  # next state prediction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TransitionModel:
    """Empirical transition model from observed label sequences."""

    matrix: np.ndarray  # (k, k) row-stochastic transition matrix
    state_names: List[str]
    self_transition_rates: Dict[int, float]  # state_id → P(stay)
    row_entropy: Dict[int, float]  # state_id → entropy of transition row
    most_likely_successor: Dict[int, int]  # state_id → most likely NEXT state (off-diagonal)
    mean_durations: Dict[int, float]  # state_id → mean run length
    duration_distributions: Dict[int, np.ndarray]  # state_id → array of run lengths


def empirical_transitions(
    labels: np.ndarray,
    state_names: Optional[List[str]] = None,
) -> TransitionModel:
    """
    Compute empirical transition probabilities from a label sequence.

    Steps:
      1. Count transitions: T[i,j] = count(label[t]=i AND label[t+1]=j)
      2. Normalize rows to get row-stochastic matrix
      3. Self-transition rate: diagonal of T
      4. Row entropy: -sum(T[i,:] * log(T[i,:] + eps))
      5. Most likely successor: argmax of off-diagonal elements per row
      6. Duration distributions: collect run lengths per state

    Args:
        labels: 1-D array of integer state labels (length n_bars).
        state_names: Optional list of state names. If None, uses "S0", "S1", ...

    Returns:
        TransitionModel with matrix, statistics, and durations.

    Raises:
        ValueError: if labels is empty or not 1-D.
    """
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")

    if len(labels) == 0:
        raise ValueError("labels is empty")

    unique_states = sorted(np.unique(labels).tolist())
    k = len(unique_states)

    # Build state index mapping (handles non-contiguous labels like [0, 2, 5])
    state_to_idx = {s: i for i, s in enumerate(unique_states)}

    if state_names is None:
        state_names = [f"S{s}" for s in unique_states]
    else:
        if len(state_names) != k:
            raise ValueError(
                f"state_names length ({len(state_names)}) != "
                f"number of unique states ({k})"
            )

    # ----- Step 1: Count transitions -----
    counts = np.zeros((k, k), dtype=float)
    for t in range(len(labels) - 1):
        i = state_to_idx[int(labels[t])]
        j = state_to_idx[int(labels[t + 1])]
        counts[i, j] += 1

    # ----- Step 2: Normalize rows -----
    row_sums = counts.sum(axis=1, keepdims=True)
    # For states with no outgoing transitions (only appear at end),
    # assign uniform distribution to maintain row-stochastic property
    zero_rows = (row_sums == 0).flatten()
    if np.any(zero_rows):
        counts[zero_rows, :] = 1.0 / k
        row_sums = counts.sum(axis=1, keepdims=True)
    matrix = counts / row_sums

    # ----- Step 3: Self-transition rates -----
    self_transition_rates = {}
    for s in unique_states:
        idx = state_to_idx[s]
        self_transition_rates[s] = float(matrix[idx, idx])

    # ----- Step 4: Row entropy -----
    row_entropy = {}
    eps = 1e-15
    for s in unique_states:
        idx = state_to_idx[s]
        row = matrix[idx, :]
        # Shannon entropy: -sum(p * log(p))
        entropy = -float(np.sum(row * np.log(row + eps)))
        row_entropy[s] = entropy

    # ----- Step 5: Most likely successor (off-diagonal) -----
    most_likely_successor = {}
    for s in unique_states:
        idx = state_to_idx[s]
        row = matrix[idx, :].copy()
        if k == 1:
            # Single state — successor is itself
            most_likely_successor[s] = s
        else:
            # Zero out the diagonal to find off-diagonal max
            row[idx] = -1.0
            best_idx = int(np.argmax(row))
            most_likely_successor[s] = unique_states[best_idx]

    # ----- Step 6: Duration distributions -----
    duration_distributions = _compute_duration_distributions(labels, unique_states)
    mean_durations = {}
    for s in unique_states:
        durs = duration_distributions[s]
        if len(durs) > 0:
            mean_durations[s] = float(np.mean(durs))
        else:
            mean_durations[s] = 0.0

    return TransitionModel(
        matrix=matrix,
        state_names=state_names,
        self_transition_rates=self_transition_rates,
        row_entropy=row_entropy,
        most_likely_successor=most_likely_successor,
        mean_durations=mean_durations,
        duration_distributions=duration_distributions,
    )


def _compute_duration_distributions(
    labels: np.ndarray,
    unique_states: List[int],
) -> Dict[int, np.ndarray]:
    """
    Compute run-length distributions per state.

    E.g. labels=[0,0,0,1,1,0,0] → {0: array([3, 2]), 1: array([2])}
    """
    durations: Dict[int, List[int]] = {s: [] for s in unique_states}

    if len(labels) == 0:
        return {s: np.array([], dtype=int) for s in unique_states}

    current_state = int(labels[0])
    current_run = 1

    for t in range(1, len(labels)):
        if int(labels[t]) == current_state:
            current_run += 1
        else:
            durations[current_state].append(current_run)
            current_state = int(labels[t])
            current_run = 1

    # Last run
    durations[current_state].append(current_run)

    return {s: np.array(d, dtype=int) for s, d in durations.items()}
