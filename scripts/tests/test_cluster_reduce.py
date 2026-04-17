"""
Skeptical tests for cluster_pipeline.reduce — dimensionality reduction.

Tests cover:
  - PCA correctness (variance, determinism, component count)
  - PCA optimal component selection
  - t-SNE output properties
  - UMAP output properties (if installed)
  - UMAP multi-seed
  - reduce_all convenience
  - Projection summary and comparison utilities
  - Top PCA features
  - Input validation
  - Edge cases (small data, high-dim, 3D projections)
  - Determinism and reproducibility
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_pipeline.reduce import (
    ProjectionResult,
    compare_projections,
    fit_pca,
    fit_tsne,
    pca_optimal_components,
    projection_summary,
    reduce_all,
    top_pca_features,
    _validate_input,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_data(
    n: int = 200,
    n_features: int = 10,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, n_features))


def _make_clustered(
    n_per_cluster: int = 100,
    k: int = 3,
    n_features: int = 8,
    separation: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    parts = []
    labels = []
    for i in range(k):
        center = np.zeros(n_features)
        center[0] = i * separation
        center[1] = (i % 2) * separation
        parts.append(rng.normal(center, 0.5, (n_per_cluster, n_features)))
        labels.extend([i] * n_per_cluster)
    return np.vstack(parts), np.array(labels)


@pytest.fixture
def random_data():
    return _make_data(n=200, n_features=10)


@pytest.fixture
def clustered_data():
    return _make_clustered(n_per_cluster=100, k=3)


@pytest.fixture
def high_dim_data():
    return _make_data(n=150, n_features=50)


@pytest.fixture
def small_data():
    return _make_data(n=20, n_features=5)


# ---------------------------------------------------------------------------
# TestFitPCA
# ---------------------------------------------------------------------------


class TestFitPCA:

    def test_returns_projection_result(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert isinstance(result, ProjectionResult)

    def test_method_label(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert result.method == "pca"

    def test_embedding_shape_2d(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert result.embedding.shape == (200, 2)

    def test_embedding_shape_3d(self, random_data):
        result = fit_pca(random_data, n_components=3)
        assert result.embedding.shape == (200, 3)

    def test_n_components_stored(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert result.n_components == 2

    def test_input_dim_stored(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert result.input_dim == 10

    def test_n_samples_stored(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert result.n_samples == 200

    def test_explained_variance_present(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert result.explained_variance is not None
        assert len(result.explained_variance) == 2

    def test_explained_variance_ratio_sums_to_less_than_1(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert result.explained_variance_ratio.sum() <= 1.0 + 1e-10

    def test_explained_variance_ratio_non_negative(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert (result.explained_variance_ratio >= 0).all()

    def test_explained_variance_ratio_decreasing(self, random_data):
        result = fit_pca(random_data, n_components=5)
        ratios = result.explained_variance_ratio
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1] - 1e-10

    def test_cumulative_variance_monotonic(self, random_data):
        result = fit_pca(random_data, n_components=5)
        cumvar = result.cumulative_variance
        for i in range(len(cumvar) - 1):
            assert cumvar[i] <= cumvar[i + 1] + 1e-10

    def test_full_components_explain_all_variance(self, random_data):
        result = fit_pca(random_data, n_components=10)
        assert result.cumulative_variance[-1] == pytest.approx(1.0, abs=1e-6)

    def test_deterministic(self, random_data):
        r1 = fit_pca(random_data, n_components=2)
        r2 = fit_pca(random_data, n_components=2)
        np.testing.assert_array_equal(r1.embedding, r2.embedding)

    def test_no_nan_in_embedding(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert not np.any(np.isnan(result.embedding))

    def test_no_inf_in_embedding(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert not np.any(np.isinf(result.embedding))

    def test_components_in_extra(self, random_data):
        result = fit_pca(random_data, n_components=2)
        assert "components" in result.extra
        assert result.extra["components"].shape == (2, 10)

    def test_scale_option(self, random_data):
        r_unscaled = fit_pca(random_data, n_components=2, scale=False)
        r_scaled = fit_pca(random_data, n_components=2, scale=True)
        # Results should differ when scaling is applied
        assert not np.allclose(r_unscaled.embedding, r_scaled.embedding)

    def test_n_components_capped_at_features(self):
        X = _make_data(n=100, n_features=3)
        result = fit_pca(X, n_components=10)
        assert result.n_components == 3

    def test_n_components_capped_at_samples(self):
        X = _make_data(n=5, n_features=20)
        result = fit_pca(X, n_components=10)
        assert result.n_components == 5


# ---------------------------------------------------------------------------
# TestPCAOptimalComponents
# ---------------------------------------------------------------------------


class TestPCAOptimalComponents:

    def test_returns_int(self, random_data):
        n = pca_optimal_components(random_data, variance_threshold=0.95)
        assert isinstance(n, int)

    def test_at_least_1(self, random_data):
        n = pca_optimal_components(random_data, variance_threshold=0.01)
        assert n >= 1

    def test_at_most_input_dim(self, random_data):
        n = pca_optimal_components(random_data, variance_threshold=0.9999)
        assert n <= 10

    def test_higher_threshold_more_components(self, random_data):
        n_low = pca_optimal_components(random_data, variance_threshold=0.5)
        n_high = pca_optimal_components(random_data, variance_threshold=0.99)
        assert n_low <= n_high

    def test_full_variance_returns_all(self, random_data):
        n = pca_optimal_components(random_data, variance_threshold=1.0)
        assert n == 10


# ---------------------------------------------------------------------------
# TestFitTSNE
# ---------------------------------------------------------------------------


class TestFitTSNE:

    def test_returns_projection_result(self, random_data):
        result = fit_tsne(random_data, n_components=2)
        assert isinstance(result, ProjectionResult)

    def test_method_label(self, random_data):
        result = fit_tsne(random_data, n_components=2)
        assert result.method == "tsne"

    def test_embedding_shape(self, random_data):
        result = fit_tsne(random_data, n_components=2)
        assert result.embedding.shape == (200, 2)

    def test_no_nan(self, random_data):
        result = fit_tsne(random_data, n_components=2)
        assert not np.any(np.isnan(result.embedding))

    def test_no_inf(self, random_data):
        result = fit_tsne(random_data, n_components=2)
        assert not np.any(np.isinf(result.embedding))

    def test_seed_stored(self, random_data):
        result = fit_tsne(random_data, random_state=123)
        assert result.seed == 123

    def test_kl_divergence_in_extra(self, random_data):
        result = fit_tsne(random_data, n_components=2)
        assert "kl_divergence" in result.extra
        assert result.extra["kl_divergence"] >= 0

    def test_deterministic_with_same_seed(self, random_data):
        r1 = fit_tsne(random_data, random_state=42)
        r2 = fit_tsne(random_data, random_state=42)
        np.testing.assert_allclose(r1.embedding, r2.embedding, atol=1e-6)

    def test_perplexity_capped_for_small_data(self, small_data):
        result = fit_tsne(small_data, n_components=2, perplexity=50)
        assert result.extra["perplexity"] < len(small_data)

    def test_pca_preprocess(self, high_dim_data):
        result = fit_tsne(high_dim_data, pca_preprocess=10)
        assert result.embedding.shape == (150, 2)
        assert result.extra.get("pca_preprocess") == 10

    def test_no_pca_preprocess_when_low_dim(self, random_data):
        result = fit_tsne(random_data, pca_preprocess=50)
        # 10 features < 50, so PCA not applied
        assert result.extra.get("pca_preprocess") is None

    def test_3d_embedding(self, random_data):
        result = fit_tsne(random_data, n_components=3, init="random")
        assert result.embedding.shape == (200, 3)


# ---------------------------------------------------------------------------
# TestFitUMAP — only if umap-learn installed
# ---------------------------------------------------------------------------


class TestFitUMAP:

    @pytest.fixture(autouse=True)
    def _check_umap(self):
        pytest.importorskip("umap")

    def test_returns_projection_result(self, random_data):
        from cluster_pipeline.reduce import fit_umap
        result = fit_umap(random_data, n_components=2)
        assert isinstance(result, ProjectionResult)

    def test_method_label(self, random_data):
        from cluster_pipeline.reduce import fit_umap
        result = fit_umap(random_data, n_components=2)
        assert result.method == "umap"

    def test_embedding_shape(self, random_data):
        from cluster_pipeline.reduce import fit_umap
        result = fit_umap(random_data, n_components=2)
        assert result.embedding.shape == (200, 2)

    def test_no_nan(self, random_data):
        from cluster_pipeline.reduce import fit_umap
        result = fit_umap(random_data, n_components=2)
        assert not np.any(np.isnan(result.embedding))

    def test_seed_stored(self, random_data):
        from cluster_pipeline.reduce import fit_umap
        result = fit_umap(random_data, random_state=123)
        assert result.seed == 123

    def test_spec_default_params(self, random_data):
        from cluster_pipeline.reduce import fit_umap
        result = fit_umap(random_data)
        assert result.extra["n_neighbors"] == 15
        assert result.extra["min_dist"] == 0.1
        assert result.extra["metric"] == "euclidean"

    def test_3d_embedding(self, random_data):
        from cluster_pipeline.reduce import fit_umap
        result = fit_umap(random_data, n_components=3)
        assert result.embedding.shape == (200, 3)

    def test_deterministic_same_seed(self, random_data):
        from cluster_pipeline.reduce import fit_umap
        r1 = fit_umap(random_data, random_state=42)
        r2 = fit_umap(random_data, random_state=42)
        np.testing.assert_allclose(r1.embedding, r2.embedding, atol=1e-6)

    def test_pca_preprocess(self, high_dim_data):
        from cluster_pipeline.reduce import fit_umap
        result = fit_umap(high_dim_data, pca_preprocess=10)
        assert result.embedding.shape == (150, 2)


class TestFitUMAPMultiSeed:

    @pytest.fixture(autouse=True)
    def _check_umap(self):
        pytest.importorskip("umap")

    def test_returns_list(self, random_data):
        from cluster_pipeline.reduce import fit_umap_multi_seed
        results = fit_umap_multi_seed(random_data, seeds=[42, 123])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_different_seeds_different_embeddings(self, random_data):
        from cluster_pipeline.reduce import fit_umap_multi_seed
        results = fit_umap_multi_seed(random_data, seeds=[42, 123])
        # Different seeds should produce different embeddings
        assert not np.allclose(results[0].embedding, results[1].embedding)

    def test_each_result_valid(self, random_data):
        from cluster_pipeline.reduce import fit_umap_multi_seed
        results = fit_umap_multi_seed(random_data, seeds=[42, 123])
        for r in results:
            assert isinstance(r, ProjectionResult)
            assert r.method == "umap"
            assert r.embedding.shape == (200, 2)

    def test_default_seeds(self, random_data):
        from cluster_pipeline.reduce import fit_umap_multi_seed
        results = fit_umap_multi_seed(random_data)
        assert len(results) == 5  # default 5 seeds


# ---------------------------------------------------------------------------
# TestReduceAll
# ---------------------------------------------------------------------------


class TestReduceAll:

    def test_returns_dict(self, random_data):
        result = reduce_all(random_data, methods=["pca", "tsne"])
        assert isinstance(result, dict)

    def test_has_requested_methods(self, random_data):
        result = reduce_all(random_data, methods=["pca", "tsne"])
        assert "pca" in result
        assert "tsne" in result

    def test_pca_only(self, random_data):
        result = reduce_all(random_data, methods=["pca"])
        assert len(result) == 1
        assert "pca" in result

    def test_all_results_valid(self, random_data):
        result = reduce_all(random_data, methods=["pca", "tsne"])
        for name, proj in result.items():
            assert isinstance(proj, ProjectionResult)
            assert proj.embedding.shape[0] == 200
            assert proj.embedding.shape[1] == 2

    def test_3d_projection(self, random_data):
        result = reduce_all(random_data, n_components=3, methods=["pca"])
        assert result["pca"].embedding.shape == (200, 3)

    def test_unknown_method_raises(self, random_data):
        with pytest.raises(ValueError, match="Unknown method"):
            reduce_all(random_data, methods=["pca", "invalid"])

    def test_includes_umap_when_available(self, random_data):
        result = reduce_all(random_data)
        # Should always have pca and tsne
        assert "pca" in result
        assert "tsne" in result


# ---------------------------------------------------------------------------
# TestProjectionSummary
# ---------------------------------------------------------------------------


class TestProjectionSummary:

    def test_returns_dict(self, random_data):
        proj = fit_pca(random_data, n_components=2)
        summary = projection_summary(proj)
        assert isinstance(summary, dict)

    def test_has_method(self, random_data):
        proj = fit_pca(random_data, n_components=2)
        summary = projection_summary(proj)
        assert summary["method"] == "pca"

    def test_has_embedding_range(self, random_data):
        proj = fit_pca(random_data, n_components=2)
        summary = projection_summary(proj)
        assert "embedding_range" in summary
        assert "dim_0" in summary["embedding_range"]
        assert "dim_1" in summary["embedding_range"]

    def test_pca_has_variance_info(self, random_data):
        proj = fit_pca(random_data, n_components=2)
        summary = projection_summary(proj)
        assert "explained_variance_ratio" in summary
        assert "cumulative_variance" in summary

    def test_tsne_no_variance_info(self, random_data):
        proj = fit_tsne(random_data, n_components=2)
        summary = projection_summary(proj)
        assert "explained_variance_ratio" not in summary

    def test_seed_in_summary(self, random_data):
        proj = fit_tsne(random_data, random_state=99)
        summary = projection_summary(proj)
        assert summary["seed"] == 99


# ---------------------------------------------------------------------------
# TestCompareProjections
# ---------------------------------------------------------------------------


class TestCompareProjections:

    def test_returns_dict(self, clustered_data):
        X, labels = clustered_data
        pca = fit_pca(X, n_components=2)
        tsne = fit_tsne(X, n_components=2)
        result = compare_projections([pca, tsne], labels)
        assert isinstance(result, dict)

    def test_has_silhouette(self, clustered_data):
        X, labels = clustered_data
        pca = fit_pca(X, n_components=2)
        result = compare_projections([pca], labels)
        key = list(result.keys())[0]
        assert "silhouette_2d" in result[key]

    def test_silhouette_in_range(self, clustered_data):
        X, labels = clustered_data
        pca = fit_pca(X, n_components=2)
        result = compare_projections([pca], labels)
        key = list(result.keys())[0]
        sil = result[key]["silhouette_2d"]
        assert -1.0 <= sil <= 1.0

    def test_well_separated_high_silhouette(self, clustered_data):
        X, labels = clustered_data
        pca = fit_pca(X, n_components=2)
        result = compare_projections([pca], labels)
        key = list(result.keys())[0]
        assert result[key]["silhouette_2d"] > 0.3

    def test_single_cluster_returns_degenerate(self, random_data):
        labels = np.zeros(200, dtype=int)
        pca = fit_pca(random_data, n_components=2)
        result = compare_projections([pca], labels)
        key = list(result.keys())[0]
        assert result[key]["silhouette_2d"] == -1.0


# ---------------------------------------------------------------------------
# TestTopPCAFeatures
# ---------------------------------------------------------------------------


class TestTopPCAFeatures:

    def test_returns_list(self, random_data):
        pca = fit_pca(random_data, n_components=2)
        names = [f"f{i}" for i in range(10)]
        result = top_pca_features(pca, names, n_top=3)
        assert isinstance(result, list)
        assert len(result) == 2  # 2 components

    def test_per_component_structure(self, random_data):
        pca = fit_pca(random_data, n_components=2)
        names = [f"f{i}" for i in range(10)]
        result = top_pca_features(pca, names, n_top=3)
        for comp in result:
            assert "component" in comp
            assert "explained_variance_ratio" in comp
            assert "top_features" in comp
            assert len(comp["top_features"]) == 3

    def test_features_have_names(self, random_data):
        pca = fit_pca(random_data, n_components=2)
        names = [f"feat_{i}" for i in range(10)]
        result = top_pca_features(pca, names, n_top=2)
        for comp in result:
            for feat in comp["top_features"]:
                assert feat["feature"].startswith("feat_")

    def test_loadings_non_zero(self, random_data):
        pca = fit_pca(random_data, n_components=2)
        names = [f"f{i}" for i in range(10)]
        result = top_pca_features(pca, names, n_top=3)
        for comp in result:
            assert comp["top_features"][0]["abs_loading"] > 0

    def test_sorted_by_abs_loading(self, random_data):
        pca = fit_pca(random_data, n_components=2)
        names = [f"f{i}" for i in range(10)]
        result = top_pca_features(pca, names, n_top=5)
        for comp in result:
            abs_loadings = [f["abs_loading"] for f in comp["top_features"]]
            assert abs_loadings == sorted(abs_loadings, reverse=True)

    def test_non_pca_raises(self, random_data):
        tsne = fit_tsne(random_data, n_components=2)
        with pytest.raises(ValueError, match="PCA"):
            top_pca_features(tsne, ["a", "b"])


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:

    def test_non_array_raises(self):
        with pytest.raises(TypeError):
            _validate_input([[1, 2], [3, 4]])

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            _validate_input(np.array([1, 2, 3]))

    def test_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            _validate_input(np.array([[1, 2]]))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            _validate_input(np.array([[1, 2], [np.nan, 4]]))

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="infinite"):
            _validate_input(np.array([[1, 2], [np.inf, 4]]))

    def test_zero_columns_raises(self):
        with pytest.raises(ValueError, match="0 columns"):
            _validate_input(np.empty((5, 0)))

    def test_fit_pca_validates(self):
        with pytest.raises(ValueError):
            fit_pca(np.array([[1, np.nan]]), n_components=1)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_two_samples(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = fit_pca(X, n_components=1)
        assert result.embedding.shape == (2, 1)

    def test_more_features_than_samples(self):
        X = _make_data(n=10, n_features=50)
        result = fit_pca(X, n_components=2)
        assert result.embedding.shape == (10, 2)

    def test_tsne_small_data(self):
        X = _make_data(n=10, n_features=5)
        result = fit_tsne(X, n_components=2, perplexity=3)
        assert result.embedding.shape == (10, 2)

    def test_single_feature(self):
        X = _make_data(n=50, n_features=1)
        result = fit_pca(X, n_components=1)
        assert result.embedding.shape == (50, 1)

    def test_constant_column_pca(self):
        """PCA on data with a constant column should still work."""
        X = _make_data(n=100, n_features=5)
        X[:, 2] = 0.0  # constant
        result = fit_pca(X, n_components=2)
        assert not np.any(np.isnan(result.embedding))


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_pca_always_deterministic(self, random_data):
        r1 = fit_pca(random_data, n_components=2)
        r2 = fit_pca(random_data, n_components=2)
        np.testing.assert_array_equal(r1.embedding, r2.embedding)

    def test_tsne_deterministic_same_seed(self, random_data):
        r1 = fit_tsne(random_data, random_state=42)
        r2 = fit_tsne(random_data, random_state=42)
        np.testing.assert_allclose(r1.embedding, r2.embedding, atol=1e-6)

    def test_reduce_all_deterministic(self, random_data):
        a1 = reduce_all(random_data, methods=["pca", "tsne"], random_state=42)
        a2 = reduce_all(random_data, methods=["pca", "tsne"], random_state=42)
        np.testing.assert_array_equal(a1["pca"].embedding, a2["pca"].embedding)
        np.testing.assert_allclose(a1["tsne"].embedding, a2["tsne"].embedding, atol=1e-6)

    def test_pca_optimal_deterministic(self, random_data):
        n1 = pca_optimal_components(random_data, variance_threshold=0.9)
        n2 = pca_optimal_components(random_data, variance_threshold=0.9)
        assert n1 == n2
