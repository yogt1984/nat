"""
Hierarchical state discovery for NAT profiling system.

Phase 3: Structure existence tests and macro regime discovery.

Usage:
    from cluster_pipeline.hierarchy import test_structure_existence
    from cluster_pipeline.hierarchy import discover_macro_regimes

    result = test_structure_existence(X_reduced)
    if result.has_structure:
        regime = discover_macro_regimes(derivatives_df)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import diptest
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from cluster_pipeline.reduction import PCAResult, reduce

logger = logging.getLogger(__name__)


@dataclass
class StructureTest:
    """Result of structure existence test."""

    hopkins_statistic: float  # > 0.7 suggests clusters exist
    dip_test_p: float  # < 0.05 suggests multimodality on PC1
    has_structure: bool  # True if either test passes
    recommendation: str  # "proceed", "weak_structure", "no_structure"


def test_structure_existence(
    X_reduced: np.ndarray,
    significance: float = 0.05,
    hopkins_threshold: float = 0.7,
    hopkins_sample_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> StructureTest:
    """
    Test whether non-trivial structure exists in PCA-reduced data.

    Runs two complementary statistical tests:
      1. Hopkins statistic — measures clustering tendency across all dimensions.
         Uniform data scores ~0.5; clustered data scores >0.7.
      2. Hartigan dip test — tests unimodality on the first principal component.
         p < significance suggests multimodality (i.e. at least 2 modes).

    Decision logic:
      - Hopkins > threshold AND dip p < significance → "proceed"
      - Hopkins > threshold OR dip p < significance → "weak_structure"
      - Both fail → "no_structure"

    Args:
        X_reduced: PCA-reduced data, shape (n_samples, n_components).
            Must have at least 10 samples and 1 dimension.
        significance: p-value threshold for the dip test.
        hopkins_threshold: Hopkins statistic threshold for clustering tendency.
        hopkins_sample_ratio: fraction of data to sample for Hopkins test.
            Clamped to yield at least 5 and at most n//2 sample points.
        seed: random seed for reproducibility.

    Returns:
        StructureTest with statistics and recommendation.

    Raises:
        ValueError: if X_reduced is invalid.
    """
    if X_reduced.ndim != 2:
        raise ValueError(f"X_reduced must be 2-D, got shape {X_reduced.shape}")

    n_samples, n_dims = X_reduced.shape

    if n_samples < 10:
        raise ValueError(f"Need at least 10 samples, got {n_samples}")

    if n_dims < 1:
        raise ValueError("X_reduced has no dimensions")

    if np.any(np.isnan(X_reduced)):
        raise ValueError("X_reduced contains NaN")

    if not (0 < significance < 1):
        raise ValueError(f"significance must be in (0, 1), got {significance}")

    if not (0 < hopkins_threshold < 1):
        raise ValueError(
            f"hopkins_threshold must be in (0, 1), got {hopkins_threshold}"
        )

    # ----- Hopkins statistic -----
    hopkins = _hopkins_statistic(
        X_reduced, sample_ratio=hopkins_sample_ratio, seed=seed
    )

    # ----- Dip test on PC1 -----
    pc1 = X_reduced[:, 0]
    dip_stat, dip_p = diptest.diptest(pc1)

    # ----- Decision -----
    hopkins_pass = hopkins > hopkins_threshold
    dip_pass = dip_p < significance

    if hopkins_pass and dip_pass:
        recommendation = "proceed"
    elif hopkins_pass or dip_pass:
        recommendation = "weak_structure"
    else:
        recommendation = "no_structure"

    has_structure = hopkins_pass or dip_pass

    return StructureTest(
        hopkins_statistic=hopkins,
        dip_test_p=dip_p,
        has_structure=has_structure,
        recommendation=recommendation,
    )


# Prevent pytest from collecting test_structure_existence as a test
test_structure_existence.__test__ = False  # type: ignore[attr-defined]


def _hopkins_statistic(
    X: np.ndarray,
    sample_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> float:
    """
    Compute the Hopkins statistic for clustering tendency.

    Compares nearest-neighbor distances of random points in the data space
    to nearest-neighbor distances of actual data points. Values near 0.5
    indicate uniform randomness; values near 1.0 indicate strong clustering.

    Args:
        X: data array, shape (n_samples, n_dims).
        sample_ratio: fraction of n_samples to use as test points.
        seed: random seed.

    Returns:
        Hopkins statistic in [0, 1].
    """
    n, d = X.shape
    # Offset seed to avoid collisions when data was generated with the same
    # seed value — otherwise random reference points can coincide with data.
    rng = np.random.RandomState(None if seed is None else seed + 2_147_483_647)

    # Determine sample size: at least 5, at most n//2
    m = max(5, min(int(n * sample_ratio), n // 2))

    # Fit nearest-neighbor model on the full dataset
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn.fit(X)

    # --- u_i: distances from random points to their nearest data neighbor ---
    # Generate random points uniformly in the data bounding box
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    # Add tiny buffer to avoid zero-range dimensions
    ranges = maxs - mins
    ranges[ranges < 1e-20] = 1.0

    random_points = rng.uniform(size=(m, d)) * ranges + mins
    nn_random = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn_random.fit(X)
    u_distances, _ = nn_random.kneighbors(random_points)
    u = u_distances[:, 0]

    # --- w_i: distances from sampled data points to their nearest OTHER neighbor ---
    sample_idx = rng.choice(n, size=m, replace=False)
    sample_points = X[sample_idx]
    # k=2 because nearest neighbor of a data point in the dataset is itself
    w_distances, _ = nn.kneighbors(sample_points)
    w = w_distances[:, 1]  # second nearest (first is the point itself)

    # Hopkins statistic — use raw distances (no d-th power) for numerical
    # stability in high dimensions.  The d-th-power formulation is
    # theoretically correct but amplifies tiny distance differences
    # in d > 5, making the statistic degenerate toward 0 or 1.
    u_sum = np.sum(u)
    w_sum = np.sum(w)

    denom = u_sum + w_sum
    if denom < 1e-30:
        return 0.5  # degenerate case

    return float(u_sum / denom)


# ---------------------------------------------------------------------------
# Task 3.1: Macro Regime Discovery
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Result of GMM k-sweep."""

    k_range: List[int]
    bic_scores: List[float]
    best_k: int
    best_bic: float


@dataclass
class QualityReport:
    """Cluster quality metrics."""

    silhouette: float  # mean silhouette score (-1 to 1)
    min_cluster_fraction: float  # fraction of bars in smallest cluster
    n_per_cluster: Dict[int, int]  # count per cluster label


@dataclass
class StabilityReport:
    """Block bootstrap stability metrics."""

    mean_ari: float
    std_ari: float
    n_bootstrap: int
    block_size: int


@dataclass
class RegimeResult:
    """Result of macro regime discovery."""

    labels: np.ndarray  # (n_samples,) cluster assignments
    k: int
    pca_result: PCAResult
    gmm_params: Dict  # means, covariances, weights
    quality: QualityReport
    stability: StabilityReport
    sweep: SweepResult
    centroid_profiles: pd.DataFrame  # mean derivative values per regime
    self_transition_rate: float  # fraction of t→t+1 same-label pairs
    durations: Dict[int, List[int]]  # run lengths per regime
    structure_test: StructureTest
    slow_columns: List[str]  # columns selected as "slow"
    filter_report: Dict  # from reduce()
    early_exit: bool = False  # True if no structure found
    early_exit_reason: str = ""


def discover_macro_regimes(
    derivatives: pd.DataFrame,
    autocorrelation_threshold: float = 0.7,
    k_range: range = range(2, 6),
    pca_variance: float = 0.95,
    n_bootstrap: int = 30,
    block_size: int = 15,
    random_state: int = 42,
) -> RegimeResult:
    """
    Discover 2-4 broad market regimes using slow-moving derivatives.

    Selects persistent features (autocorrelation > threshold at lag=5),
    reduces dimensionality, checks for structure, then fits GMM with
    k-sweep and block bootstrap stability validation.

    Args:
        derivatives: DataFrame of derivative columns.
        autocorrelation_threshold: lag-5 autocorrelation cutoff for "slow"
            features. Columns above this are used for regime detection.
        k_range: range of cluster counts to sweep.
        pca_variance: cumulative variance threshold for PCA.
        n_bootstrap: number of block bootstrap resamples for stability.
        block_size: contiguous block size for block bootstrap.
        random_state: random seed for reproducibility.

    Returns:
        RegimeResult with labels, quality, stability, and diagnostics.
        If no structure is found, returns with early_exit=True and
        labels set to all zeros.

    Raises:
        ValueError: if derivatives is empty or has < 30 rows.
    """
    if derivatives.empty or derivatives.shape[1] == 0:
        raise ValueError("derivatives DataFrame is empty")

    n_samples = len(derivatives)
    if n_samples < 30:
        raise ValueError(
            f"Need at least 30 rows for regime discovery, got {n_samples}"
        )

    # ----- Step 1: Autocorrelation split -----
    slow_cols = _autocorrelation_split(
        derivatives, lag=5, threshold=autocorrelation_threshold
    )

    if len(slow_cols) < 2:
        # Not enough slow features — use all columns with warning
        warnings.warn(
            f"Only {len(slow_cols)} slow columns found "
            f"(threshold={autocorrelation_threshold}). Using all columns."
        )
        slow_cols = derivatives.columns.tolist()

    slow_df = derivatives[slow_cols]

    # ----- Step 2: Reduce dimensionality -----
    try:
        X_reduced, pca_result, filter_report = reduce(
            slow_df, pca_variance=pca_variance
        )
    except ValueError as e:
        # All columns filtered out
        return _early_exit_result(
            n_samples=n_samples,
            reason=f"Reduction failed: {e}",
            slow_columns=slow_cols,
            derivatives=derivatives,
        )

    # ----- Step 3: Structure existence test -----
    structure = test_structure_existence(X_reduced, seed=random_state)

    if not structure.has_structure:
        return _early_exit_result(
            n_samples=n_samples,
            reason=f"No structure found (Hopkins={structure.hopkins_statistic:.3f}, "
            f"dip_p={structure.dip_test_p:.3f})",
            slow_columns=slow_cols,
            derivatives=derivatives,
            structure_test=structure,
            pca_result=pca_result,
            filter_report=filter_report,
        )

    # ----- Step 4: k-sweep -----
    sweep = _k_sweep_gmm(X_reduced, k_range=k_range, random_state=random_state)

    # ----- Step 5: Fit final GMM at best k -----
    gmm = GaussianMixture(
        n_components=sweep.best_k,
        covariance_type="full",
        n_init=5,
        random_state=random_state,
    )
    labels = gmm.fit_predict(X_reduced)

    gmm_params = {
        "means": gmm.means_.tolist(),
        "weights": gmm.weights_.tolist(),
        "covariance_type": gmm.covariance_type,
    }

    # ----- Step 6: Quality -----
    quality = _compute_quality(X_reduced, labels)

    # ----- Step 7: Block bootstrap stability -----
    stability = _block_bootstrap_stability(
        X_reduced,
        labels,
        n_components=sweep.best_k,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        random_state=random_state,
    )

    # ----- Step 8: Durations and self-transition rate -----
    durations = _compute_durations(labels)
    str_val = _self_transition_rate(labels)

    # ----- Step 9: Centroid profiles -----
    centroid_profiles = _centroid_profiles(slow_df, labels)

    return RegimeResult(
        labels=labels,
        k=sweep.best_k,
        pca_result=pca_result,
        gmm_params=gmm_params,
        quality=quality,
        stability=stability,
        sweep=sweep,
        centroid_profiles=centroid_profiles,
        self_transition_rate=str_val,
        durations=durations,
        structure_test=structure,
        slow_columns=slow_cols,
        filter_report=filter_report,
    )


# ---------------------------------------------------------------------------
# Autocorrelation split
# ---------------------------------------------------------------------------


def _autocorrelation_split(
    df: pd.DataFrame,
    lag: int = 5,
    threshold: float = 0.7,
) -> List[str]:
    """
    Select columns with autocorrelation > threshold at given lag.

    Returns list of "slow" column names sorted by autocorrelation descending.
    """
    slow = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) <= lag:
            continue
        ac = _lag_autocorrelation(series.values, lag)
        if ac > threshold:
            slow.append((col, ac))

    # Sort by autocorrelation descending
    slow.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in slow]


def _lag_autocorrelation(x: np.ndarray, lag: int) -> float:
    """Compute autocorrelation of x at given lag."""
    n = len(x)
    if n <= lag:
        return 0.0
    mean = np.mean(x)
    var = np.var(x)
    if var < 1e-20:
        return 0.0
    x_centered = x - mean
    ac = np.sum(x_centered[: n - lag] * x_centered[lag:]) / ((n - lag) * var)
    return float(ac)


# ---------------------------------------------------------------------------
# GMM k-sweep
# ---------------------------------------------------------------------------


def _k_sweep_gmm(
    X: np.ndarray,
    k_range: range = range(2, 6),
    random_state: int = 42,
) -> SweepResult:
    """Sweep over k values, fit GMM, return BIC scores."""
    k_list = list(k_range)
    bic_scores = []

    for k in k_list:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=3,
            random_state=random_state,
        )
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))

    best_idx = int(np.argmin(bic_scores))

    return SweepResult(
        k_range=k_list,
        bic_scores=bic_scores,
        best_k=k_list[best_idx],
        best_bic=bic_scores[best_idx],
    )


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def _compute_quality(X: np.ndarray, labels: np.ndarray) -> QualityReport:
    """Compute cluster quality metrics."""
    unique_labels = np.unique(labels)
    n = len(labels)

    if len(unique_labels) < 2:
        sil = 0.0
    else:
        sil = float(silhouette_score(X, labels))

    n_per_cluster = {int(k): int(np.sum(labels == k)) for k in unique_labels}
    min_frac = min(n_per_cluster.values()) / n if n > 0 else 0.0

    return QualityReport(
        silhouette=sil,
        min_cluster_fraction=min_frac,
        n_per_cluster=n_per_cluster,
    )


# ---------------------------------------------------------------------------
# Block bootstrap stability
# ---------------------------------------------------------------------------


def _block_bootstrap_stability(
    X: np.ndarray,
    reference_labels: np.ndarray,
    n_components: int,
    n_bootstrap: int = 30,
    block_size: int = 15,
    random_state: int = 42,
) -> StabilityReport:
    """
    Block bootstrap: resample contiguous blocks, re-fit GMM, measure ARI.

    Preserves temporal autocorrelation within blocks for honest stability
    estimates (unlike random bootstrap which destroys temporal structure).
    """
    n = len(X)
    rng = np.random.RandomState(random_state)
    ari_scores = []

    n_blocks = max(1, n // block_size)

    for _ in range(n_bootstrap):
        # Sample block start indices with replacement
        block_starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        indices = []
        for start in block_starts:
            indices.extend(range(start, min(start + block_size, n)))

        # Trim or pad to original size
        indices = indices[:n]
        if len(indices) < n_components + 1:
            continue

        X_boot = X[indices]

        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                n_init=1,
                random_state=rng.randint(0, 2**31),
            )
            boot_labels = gmm.fit_predict(X_boot)
            # Compare against reference labels at the same indices
            ref_subset = reference_labels[indices]
            ari = adjusted_rand_score(ref_subset, boot_labels)
            ari_scores.append(ari)
        except Exception:
            # GMM convergence failure on bootstrap sample — skip
            continue

    if not ari_scores:
        return StabilityReport(
            mean_ari=0.0, std_ari=0.0,
            n_bootstrap=n_bootstrap, block_size=block_size,
        )

    return StabilityReport(
        mean_ari=float(np.mean(ari_scores)),
        std_ari=float(np.std(ari_scores)),
        n_bootstrap=len(ari_scores),
        block_size=block_size,
    )


# ---------------------------------------------------------------------------
# Duration and self-transition rate
# ---------------------------------------------------------------------------


def _compute_durations(labels: np.ndarray) -> Dict[int, List[int]]:
    """
    Compute run lengths per regime.

    E.g. labels=[0,0,0,1,1,0,0,0,0,1] → {0: [3, 4], 1: [2, 1]}
    """
    if len(labels) == 0:
        return {}

    durations: Dict[int, List[int]] = {}
    current_label = labels[0]
    current_run = 1

    for i in range(1, len(labels)):
        if labels[i] == current_label:
            current_run += 1
        else:
            durations.setdefault(int(current_label), []).append(current_run)
            current_label = labels[i]
            current_run = 1

    # Don't forget the last run
    durations.setdefault(int(current_label), []).append(current_run)

    return durations


def _self_transition_rate(labels: np.ndarray) -> float:
    """Fraction of consecutive pairs where label[t] == label[t+1]."""
    if len(labels) < 2:
        return 1.0
    same = np.sum(labels[:-1] == labels[1:])
    return float(same / (len(labels) - 1))


# ---------------------------------------------------------------------------
# Centroid profiles
# ---------------------------------------------------------------------------


def _centroid_profiles(
    derivatives: pd.DataFrame, labels: np.ndarray
) -> pd.DataFrame:
    """Mean derivative values per regime label."""
    df = derivatives.copy()
    df["_regime_"] = labels
    profiles = df.groupby("_regime_").mean()
    return profiles


# ---------------------------------------------------------------------------
# Early exit helper
# ---------------------------------------------------------------------------


def _early_exit_result(
    n_samples: int,
    reason: str,
    slow_columns: List[str],
    derivatives: pd.DataFrame,
    structure_test: Optional[StructureTest] = None,
    pca_result: Optional[PCAResult] = None,
    filter_report: Optional[Dict] = None,
) -> RegimeResult:
    """Build a RegimeResult for early exit (no structure found)."""
    if structure_test is None:
        structure_test = StructureTest(
            hopkins_statistic=0.0, dip_test_p=1.0,
            has_structure=False, recommendation="no_structure",
        )

    if pca_result is None:
        # Dummy PCA result
        pca_result = PCAResult(
            X_reduced=np.zeros((n_samples, 1)),
            n_components=0,
            explained_variance_ratio=np.array([]),
            cumulative_variance=np.array([]),
            components=np.empty((0, 0)),
            mean=np.array([]),
            std=np.array([]),
            column_names=[],
            loadings={},
            regularized=False,
        )

    if filter_report is None:
        filter_report = {}

    labels = np.zeros(n_samples, dtype=int)

    return RegimeResult(
        labels=labels,
        k=0,
        pca_result=pca_result,
        gmm_params={},
        quality=QualityReport(silhouette=0.0, min_cluster_fraction=1.0, n_per_cluster={0: n_samples}),
        stability=StabilityReport(mean_ari=0.0, std_ari=0.0, n_bootstrap=0, block_size=0),
        sweep=SweepResult(k_range=[], bic_scores=[], best_k=0, best_bic=0.0),
        centroid_profiles=pd.DataFrame(),
        self_transition_rate=1.0,
        durations={0: [n_samples]},
        structure_test=structure_test,
        slow_columns=slow_columns,
        filter_report=filter_report,
        early_exit=True,
        early_exit_reason=reason,
    )


# ---------------------------------------------------------------------------
# Task 3.2: Micro State Discovery (Per Regime)
# ---------------------------------------------------------------------------


@dataclass
class MicroStateResult:
    """Result of micro-state discovery within one macro regime."""

    regime_id: int
    labels: np.ndarray  # (n_regime_bars,) local cluster assignments
    k: int
    pca_result: PCAResult  # separate PCA fitted on this regime's subset
    gmm_params: Dict
    quality: QualityReport
    stability: StabilityReport
    sweep: SweepResult
    centroid_profiles: pd.DataFrame
    structure_test: StructureTest
    filter_report: Dict
    n_bars: int  # number of bars in this regime


def discover_micro_states(
    derivatives: pd.DataFrame,
    macro_labels: np.ndarray,
    regime_id: int,
    k_range: range = range(2, 6),
    pca_variance: float = 0.95,
    n_bootstrap: int = 30,
    block_size: int = 10,
    min_bars: int = 100,
    random_state: int = 42,
) -> Optional[MicroStateResult]:
    """
    Discover 2-5 micro-states within a single macro regime.

    Uses ALL derivative columns (not just slow), since micro-states capture
    fast dynamics within a regime. PCA is fitted per-regime and will
    automatically use Ledoit-Wolf regularization when the regime subset
    is small relative to feature count.

    Args:
        derivatives: Full DataFrame of derivative columns (all bars).
        macro_labels: Macro regime labels for all bars (from discover_macro_regimes).
        regime_id: Which macro regime to analyze.
        k_range: Range of cluster counts to sweep.
        pca_variance: Cumulative variance threshold for per-regime PCA.
        n_bootstrap: Number of block bootstrap resamples.
        block_size: Block size for block bootstrap.
        min_bars: Minimum bars required in the regime to attempt clustering.
        random_state: Random seed for reproducibility.

    Returns:
        MicroStateResult if clustering succeeds, None if regime is too small
        or has no discoverable structure.
    """
    if len(derivatives) != len(macro_labels):
        raise ValueError(
            f"derivatives rows ({len(derivatives)}) != "
            f"macro_labels length ({len(macro_labels)})"
        )

    # ----- Step 1: Subset to this regime -----
    mask = macro_labels == regime_id
    n_regime = int(np.sum(mask))

    if n_regime < min_bars:
        warnings.warn(
            f"Regime {regime_id} has only {n_regime} bars "
            f"(< {min_bars} minimum). Skipping micro-state discovery."
        )
        return None

    regime_df = derivatives.loc[mask].reset_index(drop=True)

    # ----- Step 2: Reduce dimensionality (all columns, per-regime PCA) -----
    try:
        X_reduced, pca_result, filter_report = reduce(
            regime_df, pca_variance=pca_variance
        )
    except ValueError as e:
        warnings.warn(
            f"Regime {regime_id}: reduction failed ({e}). "
            "Skipping micro-state discovery."
        )
        return None

    # ----- Step 3: Structure existence test -----
    if n_regime < 10:
        # Too few samples even for structure test
        return None

    structure = test_structure_existence(X_reduced, seed=random_state)

    if not structure.has_structure:
        warnings.warn(
            f"Regime {regime_id}: no structure found "
            f"(Hopkins={structure.hopkins_statistic:.3f}, "
            f"dip_p={structure.dip_test_p:.3f}). "
            "Skipping micro-state discovery."
        )
        return None

    # ----- Step 4: k-sweep -----
    # Cap k_range so max k < n_regime (can't have more clusters than data)
    effective_k_range = [k for k in k_range if k < n_regime]
    if len(effective_k_range) < 1:
        warnings.warn(
            f"Regime {regime_id}: not enough bars ({n_regime}) "
            f"for any k in {list(k_range)}."
        )
        return None

    sweep = _k_sweep_gmm(
        X_reduced, k_range=range(effective_k_range[0], effective_k_range[-1] + 1),
        random_state=random_state,
    )

    # ----- Step 5: Fit final GMM -----
    gmm = GaussianMixture(
        n_components=sweep.best_k,
        covariance_type="full",
        n_init=5,
        random_state=random_state,
    )
    labels = gmm.fit_predict(X_reduced)

    gmm_params = {
        "means": gmm.means_.tolist(),
        "weights": gmm.weights_.tolist(),
        "covariance_type": gmm.covariance_type,
    }

    # ----- Step 6: Quality + stability -----
    quality = _compute_quality(X_reduced, labels)

    stability = _block_bootstrap_stability(
        X_reduced,
        labels,
        n_components=sweep.best_k,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        random_state=random_state,
    )

    # ----- Step 7: Centroid profiles -----
    centroid_profiles = _centroid_profiles(regime_df, labels)

    return MicroStateResult(
        regime_id=regime_id,
        labels=labels,
        k=sweep.best_k,
        pca_result=pca_result,
        gmm_params=gmm_params,
        quality=quality,
        stability=stability,
        sweep=sweep,
        centroid_profiles=centroid_profiles,
        structure_test=structure,
        filter_report=filter_report,
        n_bars=n_regime,
    )
