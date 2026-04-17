"""
Dimensionality reduction for NAT cluster analysis pipeline.

Wraps PCA (always first), UMAP (primary visualization), and t-SNE (confirmation)
with consistent APIs, multi-seed support, and projection metadata.

Usage:
    from cluster_pipeline.reduce import fit_pca, fit_umap, fit_tsne, reduce_all

    pca = fit_pca(X, n_components=2)
    umap_2d = fit_umap(X, n_components=2)
    all_projections = reduce_all(X)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ProjectionResult:
    """Container for a dimensionality reduction result."""
    embedding: np.ndarray          # (n_samples, n_components)
    method: str                    # "pca", "umap", "tsne"
    n_components: int
    input_dim: int
    n_samples: int
    explained_variance: Optional[np.ndarray] = None     # PCA only
    explained_variance_ratio: Optional[np.ndarray] = None  # PCA only
    cumulative_variance: Optional[np.ndarray] = None     # PCA only
    seed: Optional[int] = None
    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PCA — always first
# ---------------------------------------------------------------------------


def fit_pca(
    X: np.ndarray,
    n_components: int = 2,
    *,
    scale: bool = False,
) -> ProjectionResult:
    """
    Fit PCA projection.

    PCA preserves global variance structure and is deterministic.
    Always run PCA first before non-linear methods.

    Args:
        X: feature matrix (n_samples, n_features)
        n_components: target dimensionality (2 or 3 typical)
        scale: whether to standardize columns before PCA

    Returns:
        ProjectionResult with embedding and explained variance info
    """
    _validate_input(X)
    n_components = min(n_components, X.shape[1], X.shape[0])

    data = X
    if scale:
        data = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(data)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    return ProjectionResult(
        embedding=embedding,
        method="pca",
        n_components=n_components,
        input_dim=X.shape[1],
        n_samples=len(X),
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        cumulative_variance=cumvar,
        extra={
            "components": pca.components_,
            "mean": pca.mean_,
            "singular_values": pca.singular_values_,
        },
    )


def pca_optimal_components(
    X: np.ndarray,
    variance_threshold: float = 0.95,
    *,
    scale: bool = False,
) -> int:
    """
    Find the minimum number of PCA components to explain variance_threshold.

    Useful for deciding how much to compress before UMAP/t-SNE.
    """
    _validate_input(X)

    data = X
    if scale:
        data = StandardScaler().fit_transform(X)

    max_k = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=max_k)
    pca.fit(data)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    return min(n_components, max_k)


# ---------------------------------------------------------------------------
# UMAP — primary visualization
# ---------------------------------------------------------------------------


def fit_umap(
    X: np.ndarray,
    n_components: int = 2,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    pca_preprocess: Optional[int] = 50,
) -> ProjectionResult:
    """
    Fit UMAP projection.

    UMAP preserves both local and global topology. It's the primary
    visualization method per the spec.

    Spec defaults: n_neighbors=15, min_dist=0.1, metric="euclidean"

    Args:
        X: feature matrix (n_samples, n_features)
        n_components: target dimensionality (2 or 3)
        n_neighbors: UMAP neighborhood size
        min_dist: minimum distance in embedding space
        metric: distance metric
        random_state: for reproducibility
        pca_preprocess: if set and X has more columns, reduce with PCA first

    Returns:
        ProjectionResult with UMAP embedding
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for fit_umap. Install with: pip install umap-learn"
        )

    _validate_input(X)

    data = X
    pca_applied = False
    if pca_preprocess is not None and X.shape[1] > pca_preprocess:
        pca = PCA(n_components=pca_preprocess)
        data = pca.fit_transform(X)
        pca_applied = True

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, len(X) - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(data)

    return ProjectionResult(
        embedding=embedding,
        method="umap",
        n_components=n_components,
        input_dim=X.shape[1],
        n_samples=len(X),
        seed=random_state,
        extra={
            "n_neighbors": min(n_neighbors, len(X) - 1),
            "min_dist": min_dist,
            "metric": metric,
            "pca_preprocess": pca_preprocess if pca_applied else None,
        },
    )


def fit_umap_multi_seed(
    X: np.ndarray,
    n_components: int = 2,
    *,
    seeds: Optional[List[int]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    pca_preprocess: Optional[int] = 50,
) -> List[ProjectionResult]:
    """
    Fit UMAP with multiple random seeds.

    Per the spec: "Run multiple seeds" because UMAP is stochastic.
    Comparing projections across seeds helps assess stability.

    Returns list of ProjectionResults, one per seed.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    results = []
    for seed in seeds:
        result = fit_umap(
            X,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
            pca_preprocess=pca_preprocess,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# t-SNE — confirmation
# ---------------------------------------------------------------------------


def fit_tsne(
    X: np.ndarray,
    n_components: int = 2,
    *,
    perplexity: float = 30.0,
    learning_rate: Union[float, str] = "auto",
    max_iter: int = 1000,
    random_state: int = 42,
    pca_preprocess: Optional[int] = 50,
    init: str = "pca",
) -> ProjectionResult:
    """
    Fit t-SNE projection.

    t-SNE preserves local neighborhood structure. Used for confirmation
    after PCA and UMAP. Slower than UMAP for large datasets.

    Args:
        X: feature matrix (n_samples, n_features)
        n_components: target dimensionality (2 or 3)
        perplexity: t-SNE perplexity (roughly: expected number of neighbors)
        learning_rate: learning rate ("auto" recommended)
        max_iter: number of optimization iterations
        random_state: for reproducibility
        pca_preprocess: if set and X has more columns, reduce with PCA first
        init: initialization method ("pca" recommended for stability)

    Returns:
        ProjectionResult with t-SNE embedding
    """
    _validate_input(X)

    data = X
    pca_applied = False
    if pca_preprocess is not None and X.shape[1] > pca_preprocess:
        pca = PCA(n_components=pca_preprocess)
        data = pca.fit_transform(X)
        pca_applied = True

    # Perplexity must be less than n_samples
    effective_perplexity = min(perplexity, len(X) - 1)

    # Use PCA init only when n_components <= input dims
    effective_init = init
    if init == "pca" and n_components > data.shape[1]:
        effective_init = "random"

    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
        init=effective_init,
    )
    embedding = tsne.fit_transform(data)

    return ProjectionResult(
        embedding=embedding,
        method="tsne",
        n_components=n_components,
        input_dim=X.shape[1],
        n_samples=len(X),
        seed=random_state,
        extra={
            "perplexity": effective_perplexity,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "kl_divergence": tsne.kl_divergence_,
            "pca_preprocess": pca_preprocess if pca_applied else None,
        },
    )


# ---------------------------------------------------------------------------
# Multi-method convenience
# ---------------------------------------------------------------------------


def reduce_all(
    X: np.ndarray,
    n_components: int = 2,
    *,
    methods: Optional[List[str]] = None,
    random_state: int = 42,
    pca_preprocess: Optional[int] = 50,
) -> Dict[str, ProjectionResult]:
    """
    Run all reduction methods and return a dict of results.

    Default methods: ["pca", "tsne"]. UMAP is included if installed.

    Args:
        X: feature matrix
        n_components: target dimensionality
        methods: list of methods to run (default: auto-detect available)
        random_state: for reproducibility
        pca_preprocess: PCA pre-reduction for UMAP/t-SNE

    Returns:
        Dict mapping method name -> ProjectionResult
    """
    if methods is None:
        methods = ["pca", "tsne"]
        try:
            import umap  # noqa: F401
            methods.insert(1, "umap")
        except ImportError:
            pass

    results = {}

    for method in methods:
        if method == "pca":
            results["pca"] = fit_pca(X, n_components=n_components)
        elif method == "umap":
            results["umap"] = fit_umap(
                X, n_components=n_components,
                random_state=random_state,
                pca_preprocess=pca_preprocess,
            )
        elif method == "tsne":
            results["tsne"] = fit_tsne(
                X, n_components=n_components,
                random_state=random_state,
                pca_preprocess=pca_preprocess,
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca', 'umap', or 'tsne'.")

    return results


# ---------------------------------------------------------------------------
# Projection utilities
# ---------------------------------------------------------------------------


def projection_summary(result: ProjectionResult) -> Dict:
    """
    Return a human-readable summary of a projection result.
    """
    summary = {
        "method": result.method,
        "n_components": result.n_components,
        "input_dim": result.input_dim,
        "n_samples": result.n_samples,
        "embedding_range": {
            f"dim_{i}": {
                "min": float(result.embedding[:, i].min()),
                "max": float(result.embedding[:, i].max()),
                "mean": float(result.embedding[:, i].mean()),
                "std": float(result.embedding[:, i].std()),
            }
            for i in range(result.n_components)
        },
    }

    if result.explained_variance_ratio is not None:
        summary["explained_variance_ratio"] = result.explained_variance_ratio.tolist()
        summary["cumulative_variance"] = result.cumulative_variance.tolist()

    if result.seed is not None:
        summary["seed"] = result.seed

    return summary


def compare_projections(
    projections: List[ProjectionResult],
    labels: np.ndarray,
) -> Dict:
    """
    Compare multiple projections using silhouette score in the projected space.

    Helps assess whether cluster structure is preserved after reduction.
    """
    from sklearn.metrics import silhouette_score

    unique = set(labels)
    unique.discard(-1)
    if len(unique) < 2:
        return {p.method: {"silhouette_2d": -1.0, "seed": p.seed} for p in projections}

    result = {}
    for proj in projections:
        mask = labels >= 0
        sil = silhouette_score(proj.embedding[mask], labels[mask])
        key = f"{proj.method}" if proj.seed is None else f"{proj.method}_seed{proj.seed}"
        result[key] = {
            "silhouette_2d": float(sil),
            "seed": proj.seed,
            "method": proj.method,
        }

    return result


def top_pca_features(
    pca_result: ProjectionResult,
    feature_names: List[str],
    n_top: int = 5,
) -> List[Dict]:
    """
    Return the top contributing features for each PCA component.

    Useful for interpreting what the principal components represent.
    """
    if pca_result.method != "pca":
        raise ValueError("top_pca_features requires a PCA ProjectionResult")

    components = pca_result.extra.get("components")
    if components is None:
        raise ValueError("PCA result missing 'components' in extra")

    results = []
    for i in range(pca_result.n_components):
        loadings = components[i]
        abs_loadings = np.abs(loadings)
        top_idx = np.argsort(abs_loadings)[::-1][:n_top]
        results.append({
            "component": i,
            "explained_variance_ratio": float(pca_result.explained_variance_ratio[i]),
            "top_features": [
                {
                    "feature": feature_names[idx],
                    "loading": float(loadings[idx]),
                    "abs_loading": float(abs_loadings[idx]),
                }
                for idx in top_idx
            ],
        })

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_input(X: np.ndarray) -> None:
    """Validate feature matrix for reduction."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(X)}")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")
    if len(X) < 2:
        raise ValueError(f"Need at least 2 samples, got {len(X)}")
    if X.shape[1] == 0:
        raise ValueError("Feature matrix has 0 columns")
    if np.any(np.isnan(X)):
        raise ValueError("Feature matrix contains NaN values")
    if np.any(np.isinf(X)):
        raise ValueError("Feature matrix contains infinite values")
