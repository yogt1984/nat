"""
Hierarchical state discovery for NAT profiling system.

Phase 3: Before clustering, verify structure exists in PCA-reduced data.
Prevents wasting compute fitting GMM to uniform noise.

Usage:
    from cluster_pipeline.hierarchy import test_structure_existence

    result = test_structure_existence(X_reduced)
    if result.has_structure:
        # proceed to GMM clustering
    else:
        # stop — no meaningful clusters in this data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import diptest
import numpy as np
from sklearn.neighbors import NearestNeighbors


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
