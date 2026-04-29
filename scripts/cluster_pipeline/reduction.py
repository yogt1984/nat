"""
Dimensionality reduction pipeline for NAT profiling system.

Pre-PCA filtering and regularized PCA for the derivative feature space.
Sits between the derivative engine (Phase 1) and clustering (Phase 3).

Usage:
    from cluster_pipeline.reduction import filter_derivatives, pca_reduce
    from cluster_pipeline.reduction import save_pca_basis, load_pca_basis

    filtered_df, report = filter_derivatives(derivatives_df)
    pca_result = pca_reduce(filtered_df.values, filtered_df.columns.tolist())
    save_pca_basis(pca_result, Path("basis.npz"))
    loaded = load_pca_basis(Path("basis.npz"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def filter_derivatives(
    X: pd.DataFrame,
    variance_percentile: float = 10.0,
    correlation_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove near-constant and redundant derivative columns before PCA.

    Steps:
      1. Drop columns whose variance is below the variance_percentile-th
         percentile of all column variances (low-information features).
      2. Greedy correlation deduplication: for each pair with |corr| > threshold,
         drop the column with lower variance (keep the more informative one).

    Args:
        X: DataFrame of derivative columns (output of generate_derivatives).
            Must contain only numeric columns. NaN values are filled with 0
            for variance/correlation computation.
        variance_percentile: percentile threshold (0-100). Columns with variance
            below this percentile are dropped. E.g. 10.0 drops the bottom 10%.
        correlation_threshold: absolute correlation above which one column in
            a correlated pair is dropped. Range (0, 1].

    Returns:
        (filtered_df, report) where:
          - filtered_df: DataFrame with surviving columns (same row count as input)
          - report: dict with keys:
              - n_input: original number of columns
              - n_after_variance: columns after variance filtering
              - n_after_correlation: final column count
              - dropped_variance: list of columns dropped for low variance
              - dropped_correlation: list of columns dropped for high correlation
              - variance_threshold_value: the actual variance cutoff used

    Raises:
        ValueError: if X is empty or has no columns, or if parameters are invalid.
    """
    if X.empty or X.shape[1] == 0:
        raise ValueError("Input DataFrame must have at least one column")

    if not (0 <= variance_percentile <= 100):
        raise ValueError(
            f"variance_percentile must be in [0, 100], got {variance_percentile}"
        )

    if not (0 < correlation_threshold <= 1.0):
        raise ValueError(
            f"correlation_threshold must be in (0, 1], got {correlation_threshold}"
        )

    n_input = X.shape[1]

    # Work on a copy, fill NaN for computation
    work = X.fillna(0.0)

    # ----- Step 1: Variance filtering -----
    variances = work.var()
    threshold_value = np.percentile(variances.values, variance_percentile)

    # Keep columns above the threshold.
    # Always drop effectively-zero-variance columns (< 1e-20) regardless of percentile.
    VARIANCE_FLOOR = 1e-20
    if threshold_value < VARIANCE_FLOOR:
        keep_mask = variances > VARIANCE_FLOOR
    else:
        keep_mask = (variances >= threshold_value) & (variances > VARIANCE_FLOOR)

    dropped_variance = variances.index[~keep_mask].tolist()
    surviving_cols = variances.index[keep_mask].tolist()

    if not surviving_cols:
        # All columns dropped — return empty DataFrame
        return pd.DataFrame(index=X.index), {
            "n_input": n_input,
            "n_after_variance": 0,
            "n_after_correlation": 0,
            "dropped_variance": dropped_variance,
            "dropped_correlation": [],
            "variance_threshold_value": float(threshold_value),
        }

    n_after_variance = len(surviving_cols)

    # ----- Step 2: Greedy correlation deduplication -----
    dropped_correlation = _greedy_correlation_filter(
        work[surviving_cols], variances[surviving_cols], correlation_threshold
    )

    final_cols = [c for c in surviving_cols if c not in set(dropped_correlation)]

    # Build output
    filtered_df = X[final_cols].copy()

    report = {
        "n_input": n_input,
        "n_after_variance": n_after_variance,
        "n_after_correlation": len(final_cols),
        "dropped_variance": dropped_variance,
        "dropped_correlation": dropped_correlation,
        "variance_threshold_value": float(threshold_value),
    }

    return filtered_df, report


def _greedy_correlation_filter(
    df: pd.DataFrame,
    variances: pd.Series,
    threshold: float,
) -> List[str]:
    """
    Greedy correlation-based column removal.

    For each pair with |corr| > threshold, drop the column with lower variance.
    Process pairs in descending order of |corr| to remove the most redundant first.

    Returns list of dropped column names.
    """
    if df.shape[1] <= 1:
        return []

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Find all pairs above threshold
    # Use upper triangle to avoid duplicates
    cols = corr_matrix.columns.tolist()
    n = len(cols)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            abs_corr = abs(corr_matrix.iloc[i, j])
            if abs_corr > threshold:
                pairs.append((abs_corr, cols[i], cols[j]))

    if not pairs:
        return []

    # Sort by |corr| descending — remove most redundant first
    pairs.sort(key=lambda x: x[0], reverse=True)

    dropped = set()
    for _, col_a, col_b in pairs:
        # Skip if either already dropped
        if col_a in dropped or col_b in dropped:
            continue

        # Drop the one with lower variance
        if variances[col_a] >= variances[col_b]:
            dropped.add(col_b)
        else:
            dropped.add(col_a)

    return list(dropped)


# ---------------------------------------------------------------------------
# Task 2.2: PCA with Ledoit-Wolf regularization
# ---------------------------------------------------------------------------


@dataclass
class PCAResult:
    """Result of regularized PCA reduction."""

    X_reduced: np.ndarray  # (n_samples, n_components)
    n_components: int
    explained_variance_ratio: np.ndarray  # (n_components,)
    cumulative_variance: np.ndarray  # (n_components,)
    components: np.ndarray  # (n_components, n_features)
    mean: np.ndarray  # (n_features,)
    std: np.ndarray  # (n_features,)
    column_names: List[str]
    loadings: Dict[int, List[Tuple[str, float]]]  # top 10 loadings per PC
    regularized: bool  # whether Ledoit-Wolf was used


def pca_reduce(
    X: np.ndarray,
    column_names: List[str],
    variance_threshold: float = 0.95,
    max_components: int = 50,
) -> PCAResult:
    """
    PCA reduction with optional Ledoit-Wolf regularization.

    When n_samples < 2 * n_features, uses Ledoit-Wolf shrinkage to estimate
    the covariance matrix, preventing unstable components when data is scarce.

    Args:
        X: array of shape (n_samples, n_features). Must not contain NaN.
        column_names: feature names matching X's columns.
        variance_threshold: cumulative explained variance target in (0, 1].
            Selects the smallest k components reaching this threshold.
        max_components: hard cap on number of components.

    Returns:
        PCAResult with reduced data, saved basis, and per-PC loadings.

    Raises:
        ValueError: if inputs are invalid or contain NaN.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    n_samples, n_features = X.shape

    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples, got {n_samples}")

    if n_features == 0:
        raise ValueError("X has no features (0 columns)")

    if len(column_names) != n_features:
        raise ValueError(
            f"column_names length ({len(column_names)}) != n_features ({n_features})"
        )

    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values — filter before PCA")

    if not (0 < variance_threshold <= 1.0):
        raise ValueError(
            f"variance_threshold must be in (0, 1], got {variance_threshold}"
        )

    if max_components < 1:
        raise ValueError(f"max_components must be >= 1, got {max_components}")

    # ----- Step 1: Standardize -----
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)

    # Replace zero-std columns with 1.0 to avoid division by zero
    # (these columns are constant and will produce zero-variance PCs)
    std_safe = std.copy()
    std_safe[std_safe < 1e-20] = 1.0

    Z = (X - mean) / std_safe

    # ----- Step 2: Covariance estimation -----
    regularized = n_samples < 2 * n_features

    if regularized:
        lw = LedoitWolf()
        lw.fit(Z)
        cov_matrix = lw.covariance_
    else:
        cov_matrix = np.cov(Z, rowvar=False)

    # np.cov returns a scalar for 1 feature — reshape to 2D
    cov_matrix = np.atleast_2d(cov_matrix)

    # ----- Step 3: Eigendecomposition -----
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # eigh returns ascending order — reverse to descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp negative eigenvalues to zero (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    total_variance = eigenvalues.sum()
    if total_variance < 1e-20:
        # All data is constant — return 1 component of zeros
        return PCAResult(
            X_reduced=np.zeros((n_samples, 1)),
            n_components=1,
            explained_variance_ratio=np.array([1.0]),
            cumulative_variance=np.array([1.0]),
            components=eigenvectors[:, :1].T,
            mean=mean,
            std=std_safe,
            column_names=column_names,
            loadings={0: []},
            regularized=regularized,
        )

    explained_ratio = eigenvalues / total_variance
    cumulative = np.cumsum(explained_ratio)

    # ----- Step 4: Select n_components -----
    # Smallest k where cumulative >= threshold, capped at max_components
    max_possible = min(n_samples, n_features)
    candidates = min(max_possible, max_components)

    k = 1
    for i in range(candidates):
        if cumulative[i] >= variance_threshold:
            k = i + 1
            break
    else:
        # Threshold not reached within candidates — use all candidates
        k = candidates

    # ----- Step 5: Project -----
    components = eigenvectors[:, :k].T  # (k, n_features)
    X_reduced = Z @ components.T  # (n_samples, k)

    # ----- Step 6: Compute loadings -----
    loadings: Dict[int, List[Tuple[str, float]]] = {}
    for pc_idx in range(k):
        weights = components[pc_idx]
        # Sort by |weight| descending, take top 10
        sorted_indices = np.argsort(np.abs(weights))[::-1][:10]
        loadings[pc_idx] = [
            (column_names[j], float(weights[j])) for j in sorted_indices
        ]

    return PCAResult(
        X_reduced=X_reduced,
        n_components=k,
        explained_variance_ratio=explained_ratio[:k],
        cumulative_variance=cumulative[:k],
        components=components,
        mean=mean,
        std=std_safe,
        column_names=column_names,
        loadings=loadings,
        regularized=regularized,
    )


# ---------------------------------------------------------------------------
# Task 2.3: Save/Load PCA Basis
# ---------------------------------------------------------------------------


def save_pca_basis(result: PCAResult, path: Path) -> None:
    """
    Serialize a PCAResult to disk.

    Creates two files:
      - <path>.npz  — numpy arrays (X_reduced, components, mean, std,
        explained_variance_ratio, cumulative_variance)
      - <path>.json — metadata (n_components, column_names, loadings,
        regularized)

    The .npz/.json extensions are appended automatically if *path* has
    neither.  If *path* already ends in .npz the JSON sidecar is placed
    next to it with .json instead.

    Args:
        result: PCAResult from pca_reduce().
        path: Base path (e.g. Path("models/pca_basis")).

    Raises:
        ValueError: if result is not a PCAResult.
    """
    if not isinstance(result, PCAResult):
        raise ValueError(f"Expected PCAResult, got {type(result).__name__}")

    path = Path(path)

    # Resolve file paths
    if path.suffix == ".npz":
        npz_path = path
        json_path = path.with_suffix(".json")
    elif path.suffix == ".json":
        json_path = path
        npz_path = path.with_suffix(".npz")
    else:
        npz_path = path.with_suffix(".npz")
        json_path = path.with_suffix(".json")

    # Ensure parent directory exists
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.savez_compressed(
        npz_path,
        X_reduced=result.X_reduced,
        components=result.components,
        mean=result.mean,
        std=result.std,
        explained_variance_ratio=result.explained_variance_ratio,
        cumulative_variance=result.cumulative_variance,
    )

    # Save metadata as JSON
    # Convert loadings: Dict[int, List[Tuple[str, float]]] → JSON-safe form
    # JSON keys must be strings
    loadings_json = {
        str(k): [[name, weight] for name, weight in v]
        for k, v in result.loadings.items()
    }

    metadata = {
        "n_components": result.n_components,
        "column_names": result.column_names,
        "loadings": loadings_json,
        "regularized": result.regularized,
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_pca_basis(path: Path) -> PCAResult:
    """
    Deserialize a PCAResult from disk.

    Reads the .npz and .json files written by save_pca_basis().

    Args:
        path: Base path used during save (e.g. Path("models/pca_basis")).
            Accepts paths ending in .npz, .json, or no extension.

    Returns:
        Reconstructed PCAResult.

    Raises:
        FileNotFoundError: if either the .npz or .json file is missing.
        ValueError: if the files are corrupt or inconsistent.
    """
    path = Path(path)

    # Resolve file paths
    if path.suffix == ".npz":
        npz_path = path
        json_path = path.with_suffix(".json")
    elif path.suffix == ".json":
        json_path = path
        npz_path = path.with_suffix(".npz")
    else:
        npz_path = path.with_suffix(".npz")
        json_path = path.with_suffix(".json")

    if not npz_path.exists():
        raise FileNotFoundError(f"Array file not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    # Load arrays
    with np.load(npz_path) as data:
        X_reduced = data["X_reduced"]
        components = data["components"]
        mean = data["mean"]
        std = data["std"]
        explained_variance_ratio = data["explained_variance_ratio"]
        cumulative_variance = data["cumulative_variance"]

    # Load metadata
    with open(json_path, "r") as f:
        metadata = json.load(f)

    n_components = metadata["n_components"]
    column_names = metadata["column_names"]
    regularized = metadata["regularized"]

    # Reconstruct loadings: JSON keys are strings → convert back to int
    loadings: Dict[int, List[Tuple[str, float]]] = {
        int(k): [(name, float(weight)) for name, weight in v]
        for k, v in metadata["loadings"].items()
    }

    # Consistency checks
    if components.shape[0] != n_components:
        raise ValueError(
            f"components rows ({components.shape[0]}) != n_components ({n_components})"
        )
    if len(column_names) != components.shape[1]:
        raise ValueError(
            f"column_names length ({len(column_names)}) != "
            f"components cols ({components.shape[1]})"
        )

    return PCAResult(
        X_reduced=X_reduced,
        n_components=n_components,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_variance=cumulative_variance,
        components=components,
        mean=mean,
        std=std,
        column_names=column_names,
        loadings=loadings,
        regularized=regularized,
    )
