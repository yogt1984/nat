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


# ---------------------------------------------------------------------------
# Task 5.3: Forward Return Profiling (Multi-Horizon)
# ---------------------------------------------------------------------------


@dataclass
class ReturnProfile:
    """Forward return distribution at multiple horizons for a state."""

    state_id: int
    horizons: Dict[int, Dict]  # horizon_bars → {mean, median, std, skew, kurtosis, p5, p95, sharpe, n}


def return_profile(
    labels: np.ndarray,
    prices: np.ndarray,
    state_id: int,
    horizons: Optional[List[int]] = None,
    mean_duration: Optional[int] = None,
) -> ReturnProfile:
    """
    Compute forward log-return distribution at multiple horizons for a state.

    For each bar t where labels[t] == state_id, computes:
        return_h = log(prices[t+h] / prices[t])

    Then aggregates statistics across all such bars.

    Args:
        labels: 1-D array of state labels.
        prices: 1-D array of prices (same length as labels).
        state_id: Which state to profile.
        horizons: List of forward horizons in bars. Default [1, 5, 10, 20].
        mean_duration: If provided, automatically added to horizons
            (ensures evaluation at the state's natural timescale).

    Returns:
        ReturnProfile with per-horizon statistics.

    Raises:
        ValueError: if labels and prices have different lengths,
            or if prices contain non-positive values.
    """
    labels = np.asarray(labels)
    prices = np.asarray(prices, dtype=float)

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")

    if prices.ndim != 1:
        raise ValueError(f"prices must be 1-D, got shape {prices.shape}")

    if len(labels) != len(prices):
        raise ValueError(
            f"labels length ({len(labels)}) != prices length ({len(prices)})"
        )

    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive for log returns")

    if horizons is None:
        horizons = [1, 5, 10, 20]

    # Auto-add mean_duration to horizons
    if mean_duration is not None and mean_duration > 0:
        if mean_duration not in horizons:
            horizons = sorted(set(horizons) | {mean_duration})

    n = len(labels)
    state_mask = labels == state_id
    state_indices = np.where(state_mask)[0]

    log_prices = np.log(prices)

    horizon_stats: Dict[int, Dict] = {}

    for h in horizons:
        # Collect forward returns for bars in this state with enough forward data
        valid_indices = state_indices[state_indices + h < n]
        if len(valid_indices) == 0:
            horizon_stats[h] = {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "skew": np.nan,
                "kurtosis": np.nan,
                "p5": np.nan,
                "p95": np.nan,
                "sharpe": np.nan,
                "n": 0,
            }
            continue

        returns = log_prices[valid_indices + h] - log_prices[valid_indices]

        n_obs = len(returns)
        mean = float(np.mean(returns))
        median = float(np.median(returns))
        std = float(np.std(returns, ddof=1)) if n_obs > 1 else 0.0
        p5 = float(np.percentile(returns, 5))
        p95 = float(np.percentile(returns, 95))

        # Skewness
        if n_obs > 2 and std > 1e-15:
            skew = float(
                np.mean(((returns - mean) / std) ** 3) * n_obs / ((n_obs - 1) * (n_obs - 2) / n_obs)
            )
        else:
            skew = 0.0

        # Excess kurtosis
        if n_obs > 3 and std > 1e-15:
            kurt = float(np.mean(((returns - mean) / std) ** 4) - 3.0)
        else:
            kurt = 0.0

        # Sharpe (annualized is not meaningful here — use per-bar Sharpe)
        sharpe = mean / std if std > 1e-15 else 0.0

        horizon_stats[h] = {
            "mean": mean,
            "median": median,
            "std": std,
            "skew": skew,
            "kurtosis": kurt,
            "p5": p5,
            "p95": p95,
            "sharpe": sharpe,
            "n": n_obs,
        }

    return ReturnProfile(state_id=state_id, horizons=horizon_stats)


# ---------------------------------------------------------------------------
# Cross-Asset Lead/Lag Analysis (FR-4.5.4)
# ---------------------------------------------------------------------------


@dataclass
class StateLeadLag:
    """Lead/lag statistics for a single state across symbols."""

    state_id: int
    symbol_fractions: Dict[str, float]
    entry_order: List[Tuple[str, float]]  # [(symbol, mean_offset)] sorted
    leader: Optional[str]
    mean_lead_bars: float
    return_correlations: Optional[pd.DataFrame]
    n_episodes: int


@dataclass
class LeadLagResult:
    """Cross-asset lead/lag analysis across all states (FR-4.5.4)."""

    per_state: Dict[int, StateLeadLag]
    symbols: List[str]
    n_bars: int


def cross_asset_lead_lag(
    per_symbol_labels: Dict[str, np.ndarray],
    per_symbol_prices: Optional[Dict[str, np.ndarray]] = None,
    min_episodes: int = 3,
    max_entry_gap: int = 10,
) -> LeadLagResult:
    """
    Compute cross-asset lead/lag for each state.

    For each state, identifies which symbol enters first on average
    ("the leader") and by how many bars.

    Args:
        per_symbol_labels: {symbol: 1-D label array}. All same length.
        per_symbol_prices: Optional {symbol: 1-D price array} for return correlations.
        min_episodes: Minimum synchronized episodes to declare a leader.
        max_entry_gap: Maximum bars between first and last symbol entry
                       to consider them the same episode.

    Returns:
        LeadLagResult with per-state lead/lag statistics.

    Raises:
        ValueError: If fewer than 2 symbols or arrays differ in length.
    """
    symbols = sorted(per_symbol_labels.keys())
    if len(symbols) < 2:
        raise ValueError(f"Need >= 2 symbols, got {len(symbols)}")

    n_bars = len(per_symbol_labels[symbols[0]])
    for sym in symbols:
        if len(per_symbol_labels[sym]) != n_bars:
            raise ValueError(
                f"Label length mismatch: {symbols[0]}={n_bars}, {sym}={len(per_symbol_labels[sym])}"
            )
    if per_symbol_prices is not None:
        for sym in symbols:
            if sym not in per_symbol_prices:
                raise ValueError(f"Missing prices for symbol {sym}")
            if len(per_symbol_prices[sym]) != n_bars:
                raise ValueError(f"Price length mismatch for {sym}")

    # Collect all unique states
    all_states = set()
    for labels in per_symbol_labels.values():
        all_states.update(np.unique(labels).tolist())

    # Detect entry events: bars where label changes from previous bar
    def _find_entries(labels: np.ndarray) -> Dict[int, List[int]]:
        entries: Dict[int, List[int]] = {}
        entries.setdefault(int(labels[0]), []).append(0)
        for t in range(1, len(labels)):
            if labels[t] != labels[t - 1]:
                entries.setdefault(int(labels[t]), []).append(t)
        return entries

    per_sym_entries = {sym: _find_entries(per_symbol_labels[sym]) for sym in symbols}

    per_state: Dict[int, StateLeadLag] = {}

    for state_id in sorted(all_states):
        # Symbol fractions
        sym_fracs = {}
        for sym in symbols:
            sym_fracs[sym] = float(np.mean(per_symbol_labels[sym] == state_id))

        # Gather all entry events for this state across symbols
        events: List[Tuple[int, str]] = []  # (time, symbol)
        for sym in symbols:
            for t in per_sym_entries[sym].get(state_id, []):
                events.append((t, sym))
        events.sort(key=lambda x: x[0])

        # Group into episodes: consecutive entries within max_entry_gap
        episodes: List[Dict[str, int]] = []  # [{symbol: entry_time}]
        i = 0
        while i < len(events):
            episode: Dict[str, int] = {events[i][1]: events[i][0]}
            anchor = events[i][0]
            j = i + 1
            while j < len(events) and events[j][0] - anchor <= max_entry_gap:
                sym = events[j][1]
                if sym not in episode:  # first entry per symbol in episode
                    episode[sym] = events[j][0]
                j += 1
            # Only keep episodes with >= 2 symbols
            if len(episode) >= 2:
                episodes.append(episode)
            i = j if j > i + 1 else i + 1

        # Compute entry order from episodes
        if len(episodes) >= min_episodes:
            # For each episode, compute offset relative to earliest entry
            offsets: Dict[str, List[float]] = {sym: [] for sym in symbols}
            for episode in episodes:
                earliest = min(episode.values())
                for sym, t in episode.items():
                    offsets[sym].append(float(t - earliest))

            mean_offsets = {}
            for sym in symbols:
                if offsets[sym]:
                    mean_offsets[sym] = float(np.mean(offsets[sym]))

            entry_order = sorted(mean_offsets.items(), key=lambda x: x[1])

            # Leader: symbol with lowest mean offset
            # Only declare if gap to next is >= 0.5 bars
            if len(entry_order) >= 2 and entry_order[1][1] - entry_order[0][1] >= 0.5:
                leader = entry_order[0][0]
                mean_lead = float(entry_order[-1][1] - entry_order[0][1])
            else:
                leader = None
                mean_lead = 0.0
        else:
            entry_order = []
            leader = None
            mean_lead = 0.0

        # Return correlations within simultaneous state occupancy
        ret_corr = None
        if per_symbol_prices is not None:
            # Find bars where ALL symbols are in this state
            mask = np.ones(n_bars, dtype=bool)
            for sym in symbols:
                mask &= per_symbol_labels[sym] == state_id
            simultaneous = np.where(mask)[0]

            if len(simultaneous) > 10:
                # Compute log-returns at these bars
                returns = {}
                for sym in symbols:
                    prices = per_symbol_prices[sym]
                    # Use returns between consecutive simultaneous bars
                    valid = simultaneous[simultaneous > 0]
                    if len(valid) > 1:
                        r = np.log(prices[valid] / prices[valid - 1])
                        returns[sym] = r

                if len(returns) >= 2 and all(len(r) > 1 for r in returns.values()):
                    min_len = min(len(r) for r in returns.values())
                    mat = np.column_stack([returns[sym][:min_len] for sym in symbols])
                    corr = np.corrcoef(mat.T)
                    ret_corr = pd.DataFrame(corr, index=symbols, columns=symbols)

        per_state[state_id] = StateLeadLag(
            state_id=state_id,
            symbol_fractions=sym_fracs,
            entry_order=entry_order,
            leader=leader,
            mean_lead_bars=mean_lead,
            return_correlations=ret_corr,
            n_episodes=len(episodes),
        )

    return LeadLagResult(per_state=per_state, symbols=symbols, n_bars=n_bars)
