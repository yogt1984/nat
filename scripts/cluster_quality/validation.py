"""
External Validation Metrics

Validate clusters against external outcomes (returns, volatility).
Tests whether clusters have predictive value.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from scipy import stats


@dataclass
class ReturnDifferentiationResult:
    """Results of testing if clusters have different returns."""
    horizon: int
    anova_f: float
    anova_p: float
    kruskal_h: float
    kruskal_p: float
    eta_squared: float  # Effect size
    cluster_means: Dict[int, float]
    cluster_stds: Dict[int, float]
    significant: bool

    def effect_size_interpretation(self) -> str:
        """Interpret effect size."""
        if self.eta_squared >= 0.14:
            return "Large effect"
        elif self.eta_squared >= 0.06:
            return "Medium effect"
        elif self.eta_squared >= 0.01:
            return "Small effect"
        return "Negligible effect"


@dataclass
class VolatilityDifferentiationResult:
    """Results of testing if clusters have different volatility."""
    levene_stat: float
    levene_p: float  # Tests variance equality
    kruskal_h: float
    kruskal_p: float  # Tests median differences
    cluster_volatilities: Dict[int, float]
    significant: bool


@dataclass
class TransitionMatrixResult:
    """Transition matrix analysis results."""
    transition_matrix: List[List[float]]
    self_transition_rate: float  # Regime persistence
    transition_entropy: float  # Predictability
    avg_regime_duration: float  # Average bars in regime
    cluster_labels: List[int]


@dataclass
class ExternalValidationResults:
    """Combined external validation results."""
    return_differentiation: Dict[int, ReturnDifferentiationResult]
    volatility_differentiation: Optional[VolatilityDifferentiationResult]
    transitions: Optional[TransitionMatrixResult]

    def is_predictive(self) -> bool:
        """Check if clusters have predictive value."""
        # Need at least one significant return horizon
        any_significant = any(
            r.significant for r in self.return_differentiation.values()
        )
        return any_significant


def compute_return_differentiation(
    labels: np.ndarray,
    forward_returns: Dict[int, np.ndarray],
) -> Dict[int, ReturnDifferentiationResult]:
    """
    Test if clusters have statistically different forward returns.

    Args:
        labels: Cluster assignments (n_samples,)
        forward_returns: Dict of horizon (seconds) -> returns array

    Returns:
        Dict of horizon -> ReturnDifferentiationResult
    """
    results = {}

    for horizon, returns in forward_returns.items():
        if len(returns) != len(labels):
            continue

        # Group returns by cluster
        unique_labels = [l for l in np.unique(labels) if l != -1]
        if len(unique_labels) < 2:
            continue

        groups = []
        cluster_means = {}
        cluster_stds = {}

        for label in unique_labels:
            mask = labels == label
            group_returns = returns[mask]
            if len(group_returns) > 0:
                groups.append(group_returns)
                cluster_means[int(label)] = float(group_returns.mean())
                cluster_stds[int(label)] = float(group_returns.std())

        if len(groups) < 2:
            continue

        # ANOVA (assumes normality)
        try:
            f_stat, anova_p = stats.f_oneway(*groups)
        except Exception:
            f_stat, anova_p = 0.0, 1.0

        # Kruskal-Wallis (non-parametric)
        try:
            h_stat, kw_p = stats.kruskal(*groups)
        except Exception:
            h_stat, kw_p = 0.0, 1.0

        # Effect size (eta-squared)
        all_returns = returns[labels != -1]
        grand_mean = all_returns.mean()

        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2
            for g in groups
        )
        ss_total = ((all_returns - grand_mean) ** 2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

        results[horizon] = ReturnDifferentiationResult(
            horizon=horizon,
            anova_f=float(f_stat),
            anova_p=float(anova_p),
            kruskal_h=float(h_stat),
            kruskal_p=float(kw_p),
            eta_squared=float(eta_squared),
            cluster_means=cluster_means,
            cluster_stds=cluster_stds,
            significant=kw_p < 0.05,
        )

    return results


def compute_volatility_differentiation(
    labels: np.ndarray,
    forward_volatility: np.ndarray,
) -> VolatilityDifferentiationResult:
    """
    Test if clusters correspond to different volatility regimes.

    Args:
        labels: Cluster assignments
        forward_volatility: Forward-looking volatility measure

    Returns:
        VolatilityDifferentiationResult with test statistics
    """
    unique_labels = [l for l in np.unique(labels) if l != -1]

    if len(unique_labels) < 2:
        return VolatilityDifferentiationResult(
            levene_stat=0.0,
            levene_p=1.0,
            kruskal_h=0.0,
            kruskal_p=1.0,
            cluster_volatilities={},
            significant=False,
        )

    groups = []
    cluster_vols = {}

    for label in unique_labels:
        mask = labels == label
        group_vol = forward_volatility[mask]
        if len(group_vol) > 0:
            groups.append(group_vol)
            cluster_vols[int(label)] = float(group_vol.mean())

    if len(groups) < 2:
        return VolatilityDifferentiationResult(
            levene_stat=0.0,
            levene_p=1.0,
            kruskal_h=0.0,
            kruskal_p=1.0,
            cluster_volatilities=cluster_vols,
            significant=False,
        )

    # Levene's test for equality of variances
    try:
        levene_stat, levene_p = stats.levene(*groups)
    except Exception:
        levene_stat, levene_p = 0.0, 1.0

    # Kruskal-Wallis for median differences
    try:
        h_stat, kw_p = stats.kruskal(*groups)
    except Exception:
        h_stat, kw_p = 0.0, 1.0

    return VolatilityDifferentiationResult(
        levene_stat=float(levene_stat),
        levene_p=float(levene_p),
        kruskal_h=float(h_stat),
        kruskal_p=float(kw_p),
        cluster_volatilities=cluster_vols,
        significant=kw_p < 0.05 or levene_p < 0.05,
    )


def compute_transition_matrix(
    labels: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> TransitionMatrixResult:
    """
    Analyze cluster transition patterns.

    Args:
        labels: Cluster assignments (n_samples,)
        timestamps: Optional timestamps for ordering

    Returns:
        TransitionMatrixResult with transition analysis
    """
    if timestamps is not None:
        sort_idx = np.argsort(timestamps)
        sorted_labels = labels[sort_idx]
    else:
        sorted_labels = labels

    # Filter out noise
    valid_mask = sorted_labels != -1
    sorted_labels = sorted_labels[valid_mask]

    unique_labels = sorted(np.unique(sorted_labels))
    n_clusters = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    if n_clusters < 2:
        return TransitionMatrixResult(
            transition_matrix=[],
            self_transition_rate=1.0,
            transition_entropy=0.0,
            avg_regime_duration=float('inf'),
            cluster_labels=[],
        )

    # Count transitions
    transition_counts = np.zeros((n_clusters, n_clusters))

    for i in range(len(sorted_labels) - 1):
        from_idx = label_to_idx[sorted_labels[i]]
        to_idx = label_to_idx[sorted_labels[i + 1]]
        transition_counts[from_idx, to_idx] += 1

    # Normalize to probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.divide(
        transition_counts, row_sums,
        where=row_sums > 0,
        out=np.zeros_like(transition_counts, dtype=float)
    )

    # Self-transition rate (regime persistence)
    self_transition = np.diag(transition_probs).mean()

    # Entropy of transitions (predictability)
    def row_entropy(row):
        row = row[row > 0]
        return -np.sum(row * np.log(row + 1e-10))

    transition_entropy = np.mean([row_entropy(row) for row in transition_probs])

    # Average regime duration
    avg_duration = 1.0 / (1.0 - self_transition + 1e-10)

    return TransitionMatrixResult(
        transition_matrix=transition_probs.tolist(),
        self_transition_rate=float(self_transition),
        transition_entropy=float(transition_entropy),
        avg_regime_duration=float(avg_duration),
        cluster_labels=[int(l) for l in unique_labels],
    )


def compute_all_external_validation(
    labels: np.ndarray,
    forward_returns: Dict[int, np.ndarray],
    forward_volatility: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
) -> ExternalValidationResults:
    """
    Compute all external validation metrics.

    Args:
        labels: Cluster assignments
        forward_returns: Dict of horizon -> returns
        forward_volatility: Optional volatility array
        timestamps: Optional timestamps for transition analysis

    Returns:
        ExternalValidationResults with all metrics
    """
    return_diff = compute_return_differentiation(labels, forward_returns)

    vol_diff = None
    if forward_volatility is not None:
        vol_diff = compute_volatility_differentiation(labels, forward_volatility)

    transitions = None
    if timestamps is not None:
        transitions = compute_transition_matrix(labels, timestamps)

    return ExternalValidationResults(
        return_differentiation=return_diff,
        volatility_differentiation=vol_diff,
        transitions=transitions,
    )
