"""
Transition modeling for NAT profiling pipeline.

Phase 4: Empirical transition matrices, duration analysis, and HMM fitting.

Usage:
    from cluster_pipeline.transitions import empirical_transitions, fit_hmm

    model = empirical_transitions(labels)
    print(model.matrix)  # row-stochastic transition matrix

    hmm = fit_hmm(X, n_states=3)
    print(hmm.smoothed_labels)  # Viterbi-decoded state sequence
    print(hmm.transition_matrix)  # learned transition probabilities
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


# ===========================================================================
# HMM Fitting (Phase 4, Task 4.2)
# ===========================================================================


@dataclass
class HMMResult:
    """Result of HMM fitting via Baum-Welch (EM)."""

    model: Any  # Fitted GaussianHMM object
    smoothed_labels: np.ndarray  # Viterbi-decoded hidden state sequence
    transition_matrix: np.ndarray  # (n_states, n_states) learned transitions
    stationary_distribution: np.ndarray  # Ergodic distribution over states
    log_likelihood: float  # Training data log-likelihood
    bic: float  # Bayesian Information Criterion
    convergence: bool  # Whether EM converged


def fit_hmm(
    X: np.ndarray,
    n_states: int,
    n_iter: int = 100,
    random_state: int = 42,
    covariance_type: str = "full",
    init_transmat: Optional[np.ndarray] = None,
    init_means: Optional[np.ndarray] = None,
    min_samples: int = 200,
) -> Optional[HMMResult]:
    """
    Fit a Gaussian HMM via Baum-Welch (EM) and decode with Viterbi.

    Args:
        X: Observed feature vectors, shape (n_samples, n_features).
        n_states: Number of hidden states.
        n_iter: Maximum EM iterations (default 100).
        random_state: Seed for reproducibility.
        covariance_type: "full", "diag", "spherical", or "tied".
        init_transmat: Optional initial transition matrix from empirical_transitions.
        init_means: Optional initial state means (e.g., from GMM centroids).
        min_samples: Minimum samples required; returns None if insufficient.

    Returns:
        HMMResult on success, None if data is insufficient or fitting fails.
    """
    from hmmlearn.hmm import GaussianHMM

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    # Gate: minimum sample check
    if n_samples < min_samples:
        logger.warning(
            f"Skipping HMM: {n_samples} samples < {min_samples} minimum. "
            f"Need more data for reliable estimation."
        )
        return None

    if n_samples < n_states * 50:
        logger.warning(
            f"Low sample count ({n_samples}) for {n_states} states. "
            f"Results may be unreliable."
        )

    # Build model
    init_params = "stmc"  # start, transition, means, covariance
    if init_transmat is not None:
        init_params = init_params.replace("t", "")
    if init_means is not None:
        init_params = init_params.replace("m", "")

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        init_params=init_params,
        tol=1e-4,
    )

    # Apply initializations
    if init_transmat is not None:
        transmat = np.array(init_transmat, dtype=np.float64)
        # Ensure row-stochastic and no zeros (hmmlearn requirement)
        transmat = np.clip(transmat, 1e-6, None)
        transmat /= transmat.sum(axis=1, keepdims=True)
        hmm.transmat_ = transmat

    if init_means is not None:
        hmm.means_ = np.array(init_means, dtype=np.float64)

    # Fit
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            hmm.fit(X)
    except Exception as e:
        logger.error(f"HMM fitting failed: {e}")
        return None

    converged = hmm.monitor_.converged

    # Viterbi decoding
    try:
        smoothed_labels = hmm.predict(X)
    except Exception as e:
        logger.error(f"Viterbi decoding failed: {e}")
        return None

    # Log-likelihood
    log_likelihood = float(hmm.score(X))

    # BIC = -2 * LL + k * ln(n)
    # Number of free parameters for GaussianHMM:
    #   start: (n_states - 1)
    #   transitions: n_states * (n_states - 1)
    #   means: n_states * n_features
    #   covariances: depends on type
    n_params = (n_states - 1) + n_states * (n_states - 1) + n_states * n_features
    if covariance_type == "full":
        n_params += n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        n_params += n_states * n_features
    elif covariance_type == "spherical":
        n_params += n_states
    elif covariance_type == "tied":
        n_params += n_features * (n_features + 1) // 2

    bic = -2.0 * log_likelihood * n_samples + n_params * np.log(n_samples)

    # Transition matrix
    transition_matrix = hmm.transmat_.copy()

    # Stationary distribution: left eigenvector of transition matrix
    stationary = _compute_stationary(transition_matrix)

    logger.info(
        f"HMM fit: {n_states} states, {n_samples} samples, "
        f"LL={log_likelihood:.4f}, BIC={bic:.1f}, "
        f"converged={converged}"
    )

    return HMMResult(
        model=hmm,
        smoothed_labels=smoothed_labels,
        transition_matrix=transition_matrix,
        stationary_distribution=stationary,
        log_likelihood=log_likelihood,
        bic=bic,
        convergence=converged,
    )


def compare_hmm_gmm(
    hmm_labels: np.ndarray,
    gmm_labels: np.ndarray,
) -> dict:
    """
    Compare HMM smoothed labels against GMM hard assignments.

    Returns:
        Dict with ARI, agreement_rate, and transition_smoothness metrics.
    """
    from sklearn.metrics import adjusted_rand_score

    ari = adjusted_rand_score(gmm_labels, hmm_labels)

    # Agreement rate: fraction of bars with same label
    # (after optimal permutation via ARI — just raw overlap)
    agreement = float(np.mean(gmm_labels == hmm_labels))

    # Transition smoothness: HMM should have fewer state changes
    gmm_transitions = int(np.sum(np.diff(gmm_labels) != 0))
    hmm_transitions = int(np.sum(np.diff(hmm_labels) != 0))
    smoothness_ratio = hmm_transitions / max(gmm_transitions, 1)

    return {
        "ari": float(ari),
        "agreement_rate": agreement,
        "gmm_transitions": gmm_transitions,
        "hmm_transitions": hmm_transitions,
        "smoothness_ratio": smoothness_ratio,
    }


def _compute_stationary(T: np.ndarray) -> np.ndarray:
    """Compute stationary distribution via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    # Find eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    total = pi.sum()
    if total > 0:
        pi /= total
    else:
        pi = np.ones(len(pi)) / len(pi)
    return pi
