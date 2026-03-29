"""
Core Cluster Quality Metrics

Internal validation metrics for measuring cluster structure quality.
These metrics use only feature data, no external labels.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.cluster import KMeans


@dataclass
class SilhouetteResult:
    """Silhouette analysis results."""
    overall: float
    per_cluster: Dict[int, float]
    std: float
    pct_negative: float  # Fraction of misclassified points

    def is_acceptable(self) -> bool:
        """Check if silhouette indicates meaningful clusters."""
        return self.overall >= 0.25 and self.pct_negative < 0.3


@dataclass
class GapStatisticResult:
    """Gap statistic results."""
    gaps: List[float]
    gap_stds: List[float]
    optimal_k: int
    gap_at_optimal: float


@dataclass
class QualityMetrics:
    """Combined internal quality metrics."""
    silhouette: SilhouetteResult
    davies_bouldin: float
    calinski_harabasz: float
    gap_statistic: Optional[GapStatisticResult]
    n_clusters: int
    n_samples: int

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "CLUSTER QUALITY METRICS",
            "=" * 60,
            f"Samples: {self.n_samples}, Clusters: {self.n_clusters}",
            "",
            "Internal Validation:",
            f"  Silhouette Score:     {self.silhouette.overall:.3f}",
            f"  Davies-Bouldin Index: {self.davies_bouldin:.3f}",
            f"  Calinski-Harabasz:    {self.calinski_harabasz:.1f}",
            "",
            "Silhouette Details:",
            f"  Std Dev:              {self.silhouette.std:.3f}",
            f"  % Negative:           {self.silhouette.pct_negative:.1%}",
        ]

        for label, score in self.silhouette.per_cluster.items():
            lines.append(f"  Cluster {label}:           {score:.3f}")

        if self.gap_statistic:
            lines.extend([
                "",
                f"Gap Statistic:",
                f"  Optimal k:            {self.gap_statistic.optimal_k}",
                f"  Gap at optimal:       {self.gap_statistic.gap_at_optimal:.3f}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> SilhouetteResult:
    """
    Compute silhouette metrics with per-cluster breakdown.

    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster assignments (n_samples,)

    Returns:
        SilhouetteResult with overall score, per-cluster scores, and diagnostics
    """
    # Filter out noise labels (-1)
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    if len(np.unique(labels_clean)) < 2:
        return SilhouetteResult(
            overall=0.0,
            per_cluster={},
            std=0.0,
            pct_negative=1.0,
        )

    overall = silhouette_score(X_clean, labels_clean)
    samples = silhouette_samples(X_clean, labels_clean)

    per_cluster = {}
    for label in np.unique(labels_clean):
        cluster_mask = labels_clean == label
        per_cluster[int(label)] = float(samples[cluster_mask].mean())

    return SilhouetteResult(
        overall=float(overall),
        per_cluster=per_cluster,
        std=float(samples.std()),
        pct_negative=float((samples < 0).mean()),
    )


def compute_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Davies-Bouldin index. Lower is better.

    Args:
        X: Feature matrix
        labels: Cluster assignments

    Returns:
        Davies-Bouldin index (0 = perfect, higher = worse)
    """
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    if len(np.unique(labels_clean)) < 2:
        return float('inf')

    return float(davies_bouldin_score(X_clean, labels_clean))


def compute_calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Calinski-Harabasz index (variance ratio). Higher is better.

    Args:
        X: Feature matrix
        labels: Cluster assignments

    Returns:
        Calinski-Harabasz score (higher = better separation)
    """
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    if len(np.unique(labels_clean)) < 2:
        return 0.0

    return float(calinski_harabasz_score(X_clean, labels_clean))


def compute_gap_statistic(
    X: np.ndarray,
    max_clusters: int = 10,
    n_refs: int = 20,
    random_state: int = 42,
) -> GapStatisticResult:
    """
    Compute gap statistic for optimal cluster number selection.

    Compares within-cluster dispersion to expected dispersion under
    uniform random null distribution.

    Args:
        X: Feature matrix
        max_clusters: Maximum number of clusters to try
        n_refs: Number of reference datasets to generate
        random_state: Random seed for reproducibility

    Returns:
        GapStatisticResult with gaps, stds, and optimal k
    """
    rng = np.random.RandomState(random_state)

    def compute_Wk(X: np.ndarray, labels: np.ndarray) -> float:
        """Within-cluster sum of squares."""
        W = 0.0
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_points = X[labels == label]
            centroid = cluster_points.mean(axis=0)
            W += ((cluster_points - centroid) ** 2).sum()
        return W

    # Compute for real data
    Wks = []
    for k in range(1, min(max_clusters + 1, len(X) // 10)):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        wk = compute_Wk(X, labels)
        Wks.append(np.log(wk + 1e-10))

    # Compute for reference (uniform random in bounding box)
    Wks_ref = []
    X_min, X_max = X.min(axis=0), X.max(axis=0)

    for k in range(1, min(max_clusters + 1, len(X) // 10)):
        ref_Wks = []
        for _ in range(n_refs):
            X_ref = rng.uniform(X_min, X_max, size=X.shape)
            kmeans = KMeans(n_clusters=k, random_state=None, n_init=3)
            labels = kmeans.fit_predict(X_ref)
            ref_Wks.append(np.log(compute_Wk(X_ref, labels) + 1e-10))
        Wks_ref.append((np.mean(ref_Wks), np.std(ref_Wks)))

    # Gap = E[log(W_ref)] - log(W)
    gaps = [ref[0] - wk for ref, wk in zip(Wks_ref, Wks)]
    gap_stds = [ref[1] for ref in Wks_ref]

    optimal_k = int(np.argmax(gaps) + 1)

    return GapStatisticResult(
        gaps=gaps,
        gap_stds=gap_stds,
        optimal_k=optimal_k,
        gap_at_optimal=gaps[optimal_k - 1] if gaps else 0.0,
    )


def compute_all_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    compute_gap: bool = True,
) -> QualityMetrics:
    """
    Compute all internal quality metrics.

    Args:
        X: Feature matrix
        labels: Cluster assignments
        compute_gap: Whether to compute gap statistic (slower)

    Returns:
        QualityMetrics with all internal validation metrics
    """
    silhouette = compute_silhouette(X, labels)
    db = compute_davies_bouldin(X, labels)
    ch = compute_calinski_harabasz(X, labels)

    gap = compute_gap_statistic(X) if compute_gap else None

    return QualityMetrics(
        silhouette=silhouette,
        davies_bouldin=db,
        calinski_harabasz=ch,
        gap_statistic=gap,
        n_clusters=len(np.unique(labels[labels != -1])),
        n_samples=len(X),
    )
