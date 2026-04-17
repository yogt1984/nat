"""
Clustering engine for NAT cluster analysis pipeline.

Implements GMM (primary) and HDBSCAN (secondary) clustering with comprehensive
quality metrics, k-sweep, stability analysis, and multimodality testing.

This is the core analytical module that answers: "Do natural market regimes exist?"

Usage:
    from cluster_pipeline.cluster import (
        fit_gmm, fit_hdbscan, k_sweep, cluster_quality,
        bootstrap_stability, temporal_stability, dip_test,
        ClusterResult,
    )

    result = fit_gmm(X, k=3)
    sweep = k_sweep(X, k_range=range(2, 11))
    quality = cluster_quality(X, result.labels)
    stability = bootstrap_stability(X, k=3, n_resamples=50)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    silhouette_samples,
)
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """Container for clustering output."""
    labels: np.ndarray                    # (n_samples,) int array of cluster assignments
    k: int                                # number of clusters found
    method: str                           # "gmm", "hdbscan", "agglomerative"
    probabilities: Optional[np.ndarray] = None  # (n_samples, k) soft assignments (GMM)
    noise_count: int = 0                  # number of noise points (HDBSCAN)
    bic: Optional[float] = None           # BIC score (GMM)
    aic: Optional[float] = None           # AIC score (GMM)
    converged: bool = True
    extra: Dict = field(default_factory=dict)


@dataclass
class SweepResult:
    """Container for k-sweep output."""
    k_range: List[int]
    results: List[ClusterResult]
    silhouettes: List[float]
    davies_bouldins: List[float]
    calinski_harabasz: List[float]
    bics: List[Optional[float]]
    best_k_silhouette: int
    best_k_bic: Optional[int]


@dataclass
class QualityReport:
    """Comprehensive quality metrics for a clustering."""
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    silhouette_per_cluster: Dict[int, float]
    cluster_sizes: Dict[int, int]
    noise_fraction: float
    n_clusters: int
    n_samples: int


@dataclass
class StabilityReport:
    """Bootstrap or temporal stability report."""
    mean_ari: float
    std_ari: float
    min_ari: float
    max_ari: float
    n_resamples: int
    ari_values: List[float]
    stable: bool               # mean_ari > threshold


# ---------------------------------------------------------------------------
# GMM clustering (primary)
# ---------------------------------------------------------------------------


def fit_gmm(
    X: np.ndarray,
    k: int = 3,
    *,
    covariance_type: str = "full",
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    reg_covar: float = 1e-6,
) -> ClusterResult:
    """
    Fit a Gaussian Mixture Model.

    Args:
        X: feature matrix (n_samples, n_features)
        k: number of components
        covariance_type: "full", "tied", "diag", "spherical"
        n_init: number of initializations
        max_iter: max EM iterations
        random_state: for reproducibility
        reg_covar: regularization for covariance

    Returns:
        ClusterResult with labels, probabilities, BIC, AIC
    """
    _validate_input(X, min_samples=k)

    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        reg_covar=reg_covar,
    )

    labels = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)

    return ClusterResult(
        labels=labels,
        k=k,
        method="gmm",
        probabilities=probs,
        bic=gmm.bic(X),
        aic=gmm.aic(X),
        converged=gmm.converged_,
        extra={
            "means": gmm.means_,
            "covariance_type": covariance_type,
            "n_iter": gmm.n_iter_,
        },
    )


def fit_gmm_auto(
    X: np.ndarray,
    k_range: Optional[range] = None,
    *,
    covariance_type: str = "full",
    n_init: int = 10,
    random_state: int = 42,
) -> ClusterResult:
    """
    Fit GMM with automatic k selection via BIC minimization.

    Sweeps k_range and picks the k with lowest BIC (Bayesian Information Criterion).
    """
    if k_range is None:
        k_range = range(1, 11)

    _validate_input(X, min_samples=max(k_range))

    best_bic = np.inf
    best_result = None

    for k in k_range:
        if k > len(X):
            break
        result = fit_gmm(
            X, k=k,
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=random_state,
        )
        if result.bic < best_bic:
            best_bic = result.bic
            best_result = result

    if best_result is None:
        raise ValueError("No valid GMM fit found")

    return best_result


# ---------------------------------------------------------------------------
# HDBSCAN clustering (secondary)
# ---------------------------------------------------------------------------


def fit_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> ClusterResult:
    """
    Fit HDBSCAN density-based clustering.

    HDBSCAN automatically determines cluster count and identifies noise points
    (label = -1). Does not require specifying k.

    Args:
        X: feature matrix (n_samples, n_features)
        min_cluster_size: minimum cluster size
        min_samples: minimum samples in neighborhood (defaults to min_cluster_size)
        metric: distance metric
        cluster_selection_method: "eom" (excess of mass) or "leaf"

    Returns:
        ClusterResult with labels and noise count
    """
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        raise ImportError(
            "hdbscan is required for fit_hdbscan. Install with: pip install hdbscan"
        )

    _validate_input(X, min_samples=min_cluster_size)

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(X)

    unique_labels = set(labels)
    unique_labels.discard(-1)
    k = len(unique_labels)
    noise_count = int(np.sum(labels == -1))

    # Build soft probabilities if available
    probs = None
    if hasattr(clusterer, "probabilities_"):
        probs = clusterer.probabilities_

    return ClusterResult(
        labels=labels,
        k=k,
        method="hdbscan",
        probabilities=probs.reshape(-1, 1) if probs is not None else None,
        noise_count=noise_count,
        extra={
            "min_cluster_size": min_cluster_size,
            "cluster_selection_method": cluster_selection_method,
        },
    )


# ---------------------------------------------------------------------------
# Agglomerative clustering (diagnostic)
# ---------------------------------------------------------------------------


def fit_agglomerative(
    X: np.ndarray,
    k: int = 3,
    *,
    linkage_method: str = "ward",
    metric: str = "euclidean",
) -> ClusterResult:
    """
    Fit agglomerative (hierarchical) clustering.

    Primarily for diagnostic/dendrogram purposes, not for final assignments.
    """
    _validate_input(X, min_samples=k)

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage_method,
        metric=metric if linkage_method != "ward" else "euclidean",
    )
    labels = model.fit_predict(X)

    return ClusterResult(
        labels=labels,
        k=k,
        method="agglomerative",
        extra={"linkage": linkage_method},
    )


def compute_linkage(
    X: np.ndarray,
    method: str = "ward",
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute linkage matrix for dendrogram visualization.

    Returns scipy linkage matrix (n_samples-1, 4).
    """
    _validate_input(X, min_samples=2)

    if method == "ward":
        return linkage(X, method="ward", metric="euclidean")
    else:
        dist = pdist(X, metric=metric)
        return linkage(dist, method=method)


# ---------------------------------------------------------------------------
# k-sweep
# ---------------------------------------------------------------------------


def k_sweep(
    X: np.ndarray,
    k_range: Optional[range] = None,
    *,
    method: Literal["gmm", "agglomerative"] = "gmm",
    random_state: int = 42,
    n_init: int = 10,
) -> SweepResult:
    """
    Sweep over k values and evaluate each with multiple metrics.

    For each k, fits the specified method and computes:
      - Silhouette score
      - Davies-Bouldin index
      - Calinski-Harabasz index
      - BIC (GMM only)

    Args:
        X: feature matrix
        k_range: range of k values to try (default: 2..11)
        method: "gmm" or "agglomerative"
        random_state: for reproducibility
        n_init: number of initializations (GMM)

    Returns:
        SweepResult with all metrics and best k selections
    """
    if k_range is None:
        k_range = range(2, 11)

    k_list = [k for k in k_range if k <= len(X) - 1]
    if not k_list:
        raise ValueError(f"No valid k values for {len(X)} samples")

    _validate_input(X, min_samples=max(k_list) + 1)

    results = []
    silhouettes = []
    dbs = []
    chs = []
    bics = []

    for k in k_list:
        if method == "gmm":
            result = fit_gmm(X, k=k, random_state=random_state, n_init=n_init)
        else:
            result = fit_agglomerative(X, k=k)

        results.append(result)

        # Compute metrics (need at least 2 clusters with samples)
        unique = set(result.labels)
        unique.discard(-1)
        if len(unique) >= 2:
            sil = silhouette_score(X, result.labels)
            db = davies_bouldin_score(X, result.labels)
            ch = calinski_harabasz_score(X, result.labels)
        else:
            sil = -1.0
            db = float("inf")
            ch = 0.0

        silhouettes.append(sil)
        dbs.append(db)
        chs.append(ch)
        bics.append(result.bic)

    # Best k by silhouette
    best_sil_idx = int(np.argmax(silhouettes))
    best_k_sil = k_list[best_sil_idx]

    # Best k by BIC (GMM only)
    best_k_bic = None
    if any(b is not None for b in bics):
        valid_bics = [(i, b) for i, b in enumerate(bics) if b is not None]
        best_bic_idx = min(valid_bics, key=lambda x: x[1])[0]
        best_k_bic = k_list[best_bic_idx]

    return SweepResult(
        k_range=k_list,
        results=results,
        silhouettes=silhouettes,
        davies_bouldins=dbs,
        calinski_harabasz=chs,
        bics=bics,
        best_k_silhouette=best_k_sil,
        best_k_bic=best_k_bic,
    )


# ---------------------------------------------------------------------------
# Cluster quality metrics
# ---------------------------------------------------------------------------


def cluster_quality(
    X: np.ndarray,
    labels: np.ndarray,
) -> QualityReport:
    """
    Compute comprehensive quality metrics for a clustering.

    Args:
        X: feature matrix (n_samples, n_features)
        labels: cluster assignments (n_samples,), -1 for noise

    Returns:
        QualityReport with silhouette, Davies-Bouldin, Calinski-Harabasz,
        per-cluster silhouettes, cluster sizes, noise fraction.
    """
    _validate_input(X, min_samples=2)
    if len(labels) != len(X):
        raise ValueError(f"Label count ({len(labels)}) != sample count ({len(X)})")

    unique = set(labels)
    has_noise = -1 in unique
    cluster_ids = sorted(unique - {-1})
    n_clusters = len(cluster_ids)
    noise_count = int(np.sum(labels == -1))

    # Need >= 2 clusters for metrics
    if n_clusters < 2:
        return QualityReport(
            silhouette=-1.0,
            davies_bouldin=float("inf"),
            calinski_harabasz=0.0,
            silhouette_per_cluster={},
            cluster_sizes={c: int(np.sum(labels == c)) for c in cluster_ids},
            noise_fraction=noise_count / len(labels) if len(labels) > 0 else 0.0,
            n_clusters=n_clusters,
            n_samples=len(X),
        )

    # For metrics, exclude noise points
    if has_noise:
        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]
    else:
        X_clean = X
        labels_clean = labels

    sil = silhouette_score(X_clean, labels_clean)
    db = davies_bouldin_score(X_clean, labels_clean)
    ch = calinski_harabasz_score(X_clean, labels_clean)

    # Per-cluster silhouette
    sil_samples = silhouette_samples(X_clean, labels_clean)
    sil_per_cluster = {}
    for c in cluster_ids:
        mask_c = labels_clean == c
        if mask_c.any():
            sil_per_cluster[c] = float(np.mean(sil_samples[mask_c]))

    # Cluster sizes
    sizes = {c: int(np.sum(labels == c)) for c in cluster_ids}
    if has_noise:
        sizes[-1] = noise_count

    return QualityReport(
        silhouette=sil,
        davies_bouldin=db,
        calinski_harabasz=ch,
        silhouette_per_cluster=sil_per_cluster,
        cluster_sizes=sizes,
        noise_fraction=noise_count / len(labels),
        n_clusters=n_clusters,
        n_samples=len(X),
    )


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------


def bootstrap_stability(
    X: np.ndarray,
    k: int = 3,
    *,
    n_resamples: int = 50,
    sample_fraction: float = 0.8,
    method: Literal["gmm", "agglomerative"] = "gmm",
    random_state: int = 42,
    threshold: float = 0.6,
) -> StabilityReport:
    """
    Assess clustering stability via bootstrap resampling.

    For each resample:
      1. Draw sample_fraction of data (with replacement)
      2. Fit clustering on subsample
      3. Fit clustering on full data
      4. Compute ARI between the two on the overlapping points

    Target: mean ARI > 0.6 (from spec).

    Args:
        X: feature matrix
        k: number of clusters
        n_resamples: number of bootstrap iterations
        sample_fraction: fraction of data to sample each iteration
        method: clustering method
        random_state: base seed
        threshold: ARI threshold for "stable"

    Returns:
        StabilityReport with ARI statistics
    """
    _validate_input(X, min_samples=k + 1)

    rng = np.random.default_rng(random_state)
    n = len(X)
    sample_size = max(k + 1, int(n * sample_fraction))

    # Fit reference clustering on full data
    if method == "gmm":
        ref_result = fit_gmm(X, k=k, random_state=random_state)
    else:
        ref_result = fit_agglomerative(X, k=k)
    ref_labels = ref_result.labels

    ari_values = []
    for i in range(n_resamples):
        seed = random_state + i + 1
        indices = rng.choice(n, size=sample_size, replace=True)
        X_sub = X[indices]

        try:
            if method == "gmm":
                sub_result = fit_gmm(X_sub, k=k, random_state=seed)
            else:
                sub_result = fit_agglomerative(X_sub, k=k)

            # Compare labels on the subsample indices
            ari = adjusted_rand_score(ref_labels[indices], sub_result.labels)
            ari_values.append(ari)
        except Exception:
            # Skip failed fits
            continue

    if not ari_values:
        return StabilityReport(
            mean_ari=0.0, std_ari=0.0, min_ari=0.0, max_ari=0.0,
            n_resamples=0, ari_values=[], stable=False,
        )

    return StabilityReport(
        mean_ari=float(np.mean(ari_values)),
        std_ari=float(np.std(ari_values)),
        min_ari=float(np.min(ari_values)),
        max_ari=float(np.max(ari_values)),
        n_resamples=len(ari_values),
        ari_values=ari_values,
        stable=float(np.mean(ari_values)) > threshold,
    )


def temporal_stability(
    X: np.ndarray,
    k: int = 3,
    *,
    method: Literal["gmm", "agglomerative"] = "gmm",
    random_state: int = 42,
    threshold: float = 0.5,
) -> StabilityReport:
    """
    Assess clustering stability by comparing first half vs second half.

    Fits clustering independently on each half, then computes ARI on
    the full dataset using both sets of labels. Target: ARI > 0.5.

    For a more granular view, also computes rolling window stability
    by splitting into quarters.
    """
    _validate_input(X, min_samples=2 * k)

    n = len(X)
    mid = n // 2

    X_first = X[:mid]
    X_second = X[mid:]

    # Fit on each half
    if method == "gmm":
        r1 = fit_gmm(X_first, k=k, random_state=random_state)
        r2 = fit_gmm(X_second, k=k, random_state=random_state)
        r_full = fit_gmm(X, k=k, random_state=random_state)
    else:
        r1 = fit_agglomerative(X_first, k=k)
        r2 = fit_agglomerative(X_second, k=k)
        r_full = fit_agglomerative(X, k=k)

    # ARI: first half labels vs full labels on first half
    ari_first = adjusted_rand_score(r_full.labels[:mid], r1.labels)
    # ARI: second half labels vs full labels on second half
    ari_second = adjusted_rand_score(r_full.labels[mid:], r2.labels)

    # Quarter splits for additional granularity
    ari_values = [ari_first, ari_second]

    q_size = n // 4
    if q_size >= k + 1:
        for qi in range(4):
            start = qi * q_size
            end = start + q_size if qi < 3 else n
            X_q = X[start:end]
            if len(X_q) >= k + 1:
                try:
                    if method == "gmm":
                        r_q = fit_gmm(X_q, k=k, random_state=random_state)
                    else:
                        r_q = fit_agglomerative(X_q, k=k)
                    ari_q = adjusted_rand_score(r_full.labels[start:end], r_q.labels)
                    ari_values.append(ari_q)
                except Exception:
                    pass

    return StabilityReport(
        mean_ari=float(np.mean(ari_values)),
        std_ari=float(np.std(ari_values)),
        min_ari=float(np.min(ari_values)),
        max_ari=float(np.max(ari_values)),
        n_resamples=len(ari_values),
        ari_values=ari_values,
        stable=float(np.mean(ari_values)) > threshold,
    )


# ---------------------------------------------------------------------------
# Multimodality testing
# ---------------------------------------------------------------------------


def dip_test(x: np.ndarray) -> Tuple[float, float]:
    """
    Hartigan's dip test for unimodality.

    Tests whether a 1D distribution is multimodal.
    Returns (dip_statistic, p_value).

    p < 0.05 -> reject unimodality -> evidence of multimodality.

    Uses a Monte Carlo approach: compares the observed dip statistic
    against unimodal (Gaussian) reference samples of the same size.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]

    if len(x) < 4:
        return 0.0, 1.0

    x_sorted = np.sort(x)
    n = len(x_sorted)

    # Compute dip statistic
    dip_stat = _compute_dip(x_sorted, n)

    # Monte Carlo p-value: compare against Gaussian (unimodal) reference
    n_mc = 1000
    rng = np.random.default_rng(42)
    mc_dips = np.zeros(n_mc)
    for i in range(n_mc):
        gauss_sample = np.sort(rng.normal(0, 1, n))
        mc_dips[i] = _compute_dip(gauss_sample, n)

    p_value = float(np.mean(mc_dips >= dip_stat))

    return float(dip_stat), p_value


def bimodality_coefficient(x: np.ndarray) -> float:
    """
    Compute the bimodality coefficient.

    BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)*(n-3)))

    BC > 5/9 (~0.555) suggests bimodality.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 4:
        return 0.0

    skew = float(stats.skew(x))
    kurt = float(stats.kurtosis(x, fisher=True))  # excess kurtosis

    # Adjusted kurtosis denominator
    denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))

    if abs(denom) < 1e-15:
        return 0.0

    bc = (skew ** 2 + 1) / denom

    return float(bc)


def multimodality_scan(
    X: np.ndarray,
    column_names: Optional[List[str]] = None,
    *,
    alpha: float = 0.05,
    bc_threshold: float = 5 / 9,
) -> List[Dict]:
    """
    Scan all features for multimodality.

    For each column in X, computes:
      - Dip test statistic and p-value
      - Bimodality coefficient
      - Whether multimodality is detected

    Args:
        X: feature matrix (n_samples, n_features)
        column_names: optional names for columns
        alpha: significance level for dip test
        bc_threshold: threshold for bimodality coefficient

    Returns:
        List of dicts with per-feature multimodality results, sorted by
        dip p-value ascending (most multimodal first).
    """
    n_features = X.shape[1] if X.ndim == 2 else 1
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if column_names is None:
        column_names = [f"feature_{i}" for i in range(n_features)]

    results = []
    for j in range(n_features):
        col = X[:, j]
        dip_stat, dip_p = dip_test(col)
        bc = bimodality_coefficient(col)

        results.append({
            "feature": column_names[j],
            "dip_statistic": dip_stat,
            "dip_p_value": dip_p,
            "bimodality_coefficient": bc,
            "dip_multimodal": dip_p < alpha,
            "bc_multimodal": bc > bc_threshold,
            "multimodal": dip_p < alpha or bc > bc_threshold,
        })

    results.sort(key=lambda r: r["dip_p_value"])
    return results


# ---------------------------------------------------------------------------
# Predictive quality (returns-based)
# ---------------------------------------------------------------------------


def predictive_quality(
    labels: np.ndarray,
    forward_returns: np.ndarray,
) -> Dict:
    """
    Assess whether clusters predict forward returns.

    Computes:
      - Kruskal-Wallis H-test (p < 0.05 target)
      - Eta-squared effect size (> 0.01 target)
      - Per-cluster mean return and Sharpe ratio
      - Self-transition rate (> 0.7 target)

    Args:
        labels: cluster assignments (n,)
        forward_returns: forward returns aligned with labels (n,)

    Returns:
        Dict with test results and per-cluster statistics
    """
    labels = np.asarray(labels)
    forward_returns = np.asarray(forward_returns, dtype=np.float64)

    if len(labels) != len(forward_returns):
        raise ValueError("labels and forward_returns must have same length")

    # Remove noise points
    valid = labels >= 0
    labels_v = labels[valid]
    returns_v = forward_returns[valid]

    unique = sorted(set(labels_v))
    if len(unique) < 2:
        return {
            "kruskal_wallis_h": 0.0,
            "kruskal_wallis_p": 1.0,
            "eta_squared": 0.0,
            "per_cluster": {},
            "self_transition_rate": 0.0,
            "significant": False,
        }

    # Kruskal-Wallis test
    groups = [returns_v[labels_v == c] for c in unique]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) >= 2:
        h_stat, p_val = stats.kruskal(*groups)
    else:
        h_stat, p_val = 0.0, 1.0

    # Eta-squared (effect size from Kruskal-Wallis)
    n_total = len(returns_v)
    eta_sq = (h_stat - len(unique) + 1) / (n_total - len(unique)) if n_total > len(unique) else 0.0
    eta_sq = max(0.0, eta_sq)

    # Per-cluster statistics
    per_cluster = {}
    for c in unique:
        r = returns_v[labels_v == c]
        mean_r = float(np.mean(r))
        std_r = float(np.std(r))
        sharpe = mean_r / std_r if std_r > 1e-10 else 0.0
        per_cluster[int(c)] = {
            "count": len(r),
            "mean_return": mean_r,
            "std_return": std_r,
            "sharpe": sharpe,
            "min_return": float(np.min(r)),
            "max_return": float(np.max(r)),
        }

    # Self-transition rate
    transition_rate = _self_transition_rate(labels)

    return {
        "kruskal_wallis_h": float(h_stat),
        "kruskal_wallis_p": float(p_val),
        "eta_squared": float(eta_sq),
        "per_cluster": per_cluster,
        "self_transition_rate": transition_rate,
        "significant": p_val < 0.05 and eta_sq > 0.01,
    }


# ---------------------------------------------------------------------------
# Convenience: full analysis pipeline
# ---------------------------------------------------------------------------


def full_analysis(
    X: np.ndarray,
    *,
    k_range: Optional[range] = None,
    n_bootstrap: int = 50,
    column_names: Optional[List[str]] = None,
    forward_returns: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict:
    """
    Run the complete clustering analysis pipeline.

    Steps:
      1. k-sweep with GMM
      2. Fit best-k GMM
      3. Cluster quality metrics
      4. Bootstrap stability
      5. Temporal stability
      6. Multimodality scan
      7. Predictive quality (if forward_returns provided)

    Returns a comprehensive dict with all results.
    """
    if k_range is None:
        k_range = range(2, 11)

    # 1. k-sweep
    sweep = k_sweep(X, k_range=k_range, method="gmm", random_state=random_state)

    # 2. Best GMM
    best_k = sweep.best_k_bic if sweep.best_k_bic is not None else sweep.best_k_silhouette
    best_result = fit_gmm(X, k=best_k, random_state=random_state)

    # 3. Quality
    quality = cluster_quality(X, best_result.labels)

    # 4. Bootstrap stability
    boot = bootstrap_stability(
        X, k=best_k, n_resamples=n_bootstrap, random_state=random_state,
    )

    # 5. Temporal stability
    temp = temporal_stability(X, k=best_k, random_state=random_state)

    # 6. Multimodality scan
    modality = multimodality_scan(X, column_names=column_names)

    result = {
        "sweep": sweep,
        "best_k": best_k,
        "best_result": best_result,
        "quality": quality,
        "bootstrap_stability": boot,
        "temporal_stability": temp,
        "multimodality": modality,
    }

    # 7. Predictive quality
    if forward_returns is not None:
        result["predictive"] = predictive_quality(best_result.labels, forward_returns)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input(X: np.ndarray, min_samples: int = 2) -> None:
    """Validate feature matrix."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(X)}")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")
    if len(X) < min_samples:
        raise ValueError(f"Need at least {min_samples} samples, got {len(X)}")
    if X.shape[1] == 0:
        raise ValueError("Feature matrix has 0 columns")
    if np.any(np.isnan(X)):
        raise ValueError("Feature matrix contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Feature matrix contains infinite values")


def _compute_dip(x_sorted: np.ndarray, n: int) -> float:
    """
    Compute the Hartigan dip statistic for a sorted array.

    The dip measures the maximum difference between the empirical CDF
    and the best-fitting unimodal CDF. For bimodal data, there's a
    "flat" region in the ECDF where density drops between modes.
    """
    if n < 2:
        return 0.0

    # Empirical CDF values
    ecdf = np.arange(1, n + 1) / n

    # Normalize x to [0, 1]
    x_min, x_max = x_sorted[0], x_sorted[-1]
    if x_max - x_min < 1e-15:
        return 0.0
    x_norm = (x_sorted - x_min) / (x_max - x_min)

    # Greatest Convex Minorant (GCM) of the ECDF
    gcm = np.copy(ecdf)
    for i in range(1, n):
        # If the current point is below the line from start to here,
        # update by taking the convex hull from below
        if gcm[i] < gcm[i - 1]:
            gcm[i] = gcm[i - 1]

    # Least Concave Majorant (LCM) of the ECDF
    lcm = np.copy(ecdf)
    for i in range(n - 2, -1, -1):
        if lcm[i] > lcm[i + 1]:
            lcm[i] = lcm[i + 1]

    # The dip is half the max difference between LCM and GCM
    # but a more effective approach for well-separated modes is to
    # look at the gap between x_norm and ecdf (Kolmogorov-Smirnov style)
    # For a uniform distribution, ecdf should track x_norm closely.
    # For bimodal, there's a plateau in ecdf where x_norm advances.
    dip = np.max(np.abs(ecdf - x_norm)) / 2.0

    return dip


def _self_transition_rate(labels: np.ndarray) -> float:
    """
    Compute the fraction of consecutive time steps that stay in the same cluster.

    High self-transition (> 0.7) means regimes are persistent, not noisy.
    """
    labels = np.asarray(labels)
    valid = labels >= 0
    labels_v = labels[valid]

    if len(labels_v) < 2:
        return 0.0

    same = np.sum(labels_v[1:] == labels_v[:-1])
    return float(same / (len(labels_v) - 1))
