"""Tests for explore_clusters.py — feature selection, PCA, clustering, analysis."""

import sys
from pathlib import Path

import numpy as np
import pytest


try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

pytestmark = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

from explore_clusters import (
    FEATURE_SUBSETS,
    get_feature_columns,
    prepare_features,
    compute_pca,
    compute_tsne,
    find_clusters,
    analyze_clusters,
    plot_embedding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 500, seed: int = 42) -> pl.DataFrame:
    """Synthetic polars DataFrame with feature columns."""
    rng = np.random.RandomState(seed)
    return pl.DataFrame({
        "timestamp_ns": np.arange(n),
        "ent_permutation_8": rng.randn(n),
        "ent_permutation_16": rng.randn(n),
        "ent_book_shape": rng.randn(n),
        "flow_volume_5s": rng.exponential(100, n),
        "flow_trade_count_5s": rng.poisson(10, n).astype(np.float64),
        "imb_l5": rng.randn(n),
        "vol_realized_20": rng.exponential(0.01, n),
        "vol_realized_100": rng.exponential(0.01, n),
        "raw_spread_bps": rng.exponential(2.0, n),
    })


# ---------------------------------------------------------------------------
# FEATURE_SUBSETS
# ---------------------------------------------------------------------------

class TestFeatureSubsets:
    def test_has_required_subsets(self):
        for key in ("entropy", "orderbook", "flow", "volatility", "all"):
            assert key in FEATURE_SUBSETS

    def test_each_has_description(self):
        for key, val in FEATURE_SUBSETS.items():
            assert "description" in val
            assert isinstance(val["description"], str)


# ---------------------------------------------------------------------------
# get_feature_columns
# ---------------------------------------------------------------------------

class TestGetFeatureColumns:
    def test_entropy_subset(self):
        df = _make_df()
        cols = get_feature_columns(df, "entropy")
        assert "ent_permutation_8" in cols
        assert "ent_permutation_16" in cols
        assert "flow_volume_5s" not in cols

    def test_flow_subset(self):
        df = _make_df()
        cols = get_feature_columns(df, "flow")
        assert "flow_volume_5s" in cols
        assert "ent_permutation_8" not in cols

    def test_all_subset(self):
        df = _make_df()
        cols = get_feature_columns(df, "all")
        # Should include all numeric cols except timestamp_ns
        assert "ent_permutation_8" in cols
        assert "flow_volume_5s" in cols
        assert "timestamp_ns" not in cols

    def test_unknown_subset_raises(self):
        df = _make_df()
        with pytest.raises(ValueError):
            get_feature_columns(df, "nonexistent")


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def test_output_shape(self):
        df = _make_df(200)
        cols = ["ent_permutation_8", "ent_permutation_16", "ent_book_shape"]
        X, names = prepare_features(df, cols)
        assert X.shape[1] == 3
        assert X.shape[0] <= 200
        assert names == cols

    def test_scaled_output(self):
        df = _make_df(200)
        cols = ["ent_permutation_8", "ent_permutation_16"]
        X, _ = prepare_features(df, cols)
        # StandardScaler → mean ~ 0, std ~ 1
        assert abs(X[:, 0].mean()) < 0.1
        assert abs(X[:, 0].std() - 1.0) < 0.1

    def test_missing_columns_warns(self, capsys):
        df = _make_df(200)
        cols = ["ent_permutation_8", "nonexistent_col"]
        X, names = prepare_features(df, cols)
        assert X.shape[1] == 1
        assert "nonexistent_col" not in names

    def test_too_few_samples_raises(self):
        df = _make_df(10)
        # Add NaN to reduce valid rows below 100
        data = {"col1": [np.nan] * 10, "col2": [np.nan] * 10}
        df2 = pl.DataFrame(data)
        with pytest.raises(ValueError, match="Too few"):
            prepare_features(df2, ["col1", "col2"])


# ---------------------------------------------------------------------------
# compute_pca
# ---------------------------------------------------------------------------

class TestComputePca:
    def test_output_dimensions(self):
        X = np.random.randn(100, 5)
        emb, var = compute_pca(X, n_components=2)
        assert emb.shape == (100, 2)
        assert len(var) == 2
        assert all(0 <= v <= 1 for v in var)

    def test_variance_sums_to_leq_1(self):
        X = np.random.randn(100, 5)
        _, var = compute_pca(X, n_components=5)
        assert sum(var) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# find_clusters
# ---------------------------------------------------------------------------

class TestFindClusters:
    def test_gmm_returns_labels(self):
        rng = np.random.RandomState(42)
        # Two clear clusters
        X = np.vstack([rng.randn(100, 3) + [3, 0, 0], rng.randn(100, 3) - [3, 0, 0]])
        labels, info = find_clusters(X, method="gmm", max_clusters=5)
        assert len(labels) == 200
        assert info["method"] == "gmm"
        assert info["n_clusters"] >= 2

    def test_kmeans_returns_labels(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(100, 3) + [5, 0, 0], rng.randn(100, 3) - [5, 0, 0]])
        labels, info = find_clusters(X, method="kmeans", max_clusters=5)
        assert len(labels) == 200
        assert info["method"] == "kmeans"
        assert info["n_clusters"] >= 2

    def test_silhouette_computed(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(100, 3) + [5, 0, 0], rng.randn(100, 3) - [5, 0, 0]])
        _, info = find_clusters(X, method="gmm", max_clusters=5)
        assert "silhouette" in info
        assert -1.0 <= info["silhouette"] <= 1.0

    def test_unknown_method_raises(self):
        X = np.random.randn(50, 3)
        with pytest.raises(ValueError):
            find_clusters(X, method="invalid")


# ---------------------------------------------------------------------------
# analyze_clusters
# ---------------------------------------------------------------------------

class TestAnalyzeClusters:
    def test_returns_per_cluster_info(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        labels = np.array([0] * 50 + [1] * 50)
        names = ["f1", "f2", "f3"]
        analysis = analyze_clusters(X, labels, names)

        assert 0 in analysis
        assert 1 in analysis
        assert analysis[0]["size"] == 50
        assert analysis[1]["size"] == 50
        assert len(analysis[0]["top_features"]) > 0

    def test_skips_noise_label(self):
        X = np.random.randn(100, 2)
        labels = np.array([-1] * 30 + [0] * 40 + [1] * 30)
        analysis = analyze_clusters(X, labels, ["a", "b"])
        assert -1 not in analysis
        assert 0 in analysis
