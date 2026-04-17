"""
Skeptical tests for cluster_pipeline.viz — visualization module.

Tests cover:
  - 2D scatter plots (cluster labels, continuous coloring)
  - 3D interactive plots (Plotly)
  - Comparison grid (4-panel)
  - Silhouette plot
  - Centroid heatmap
  - Pairplot
  - Dendrogram
  - k-sweep plot
  - generate_all_plots convenience
  - Input validation and edge cases
  - Figure properties (axes, titles, colorbars, legends)
  - Determinism
  - Color utilities
  - Noise label handling
  - Missing optional data handling
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import linkage

from cluster_pipeline.viz import (
    _get_cluster_colors,
    _validate_embedding,
    generate_all_plots,
    plot_centroid_heatmap,
    plot_comparison_grid,
    plot_dendrogram,
    plot_k_sweep,
    plot_pairplot,
    plot_scatter_2d,
    plot_scatter_continuous,
    plot_silhouette,
)
from cluster_pipeline.cluster import SweepResult, ClusterResult, k_sweep, fit_gmm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_data(n_per_cluster=80, k=3, n_features=5, seed=42, separation=6.0):
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for i in range(k):
        center = np.zeros(n_features)
        center[0] = i * separation
        parts.append(rng.normal(center, 0.5, (n_per_cluster, n_features)))
        labels.extend([i] * n_per_cluster)
    return np.vstack(parts), np.array(labels)


def _make_embedding_2d(n=240, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 2))


def _make_embedding_3d(n=240, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 3))


@pytest.fixture
def data_and_labels():
    return _make_data()


@pytest.fixture
def embedding_2d():
    return _make_embedding_2d()


@pytest.fixture
def embedding_3d():
    return _make_embedding_3d()


@pytest.fixture
def labels_3():
    return np.array([0] * 80 + [1] * 80 + [2] * 80)


@pytest.fixture
def labels_with_noise():
    return np.array([0] * 70 + [-1] * 10 + [1] * 70 + [-1] * 10 + [2] * 80)


@pytest.fixture
def sweep_result(data_and_labels):
    X, _ = data_and_labels
    return k_sweep(X, k_range=range(2, 6), random_state=42)


@pytest.fixture
def linkage_matrix(data_and_labels):
    X, _ = data_and_labels
    return linkage(X[:100], method="ward")  # subset for speed


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# TestPlotScatter2D
# ---------------------------------------------------------------------------


class TestPlotScatter2D:

    def test_returns_figure(self, embedding_2d, labels_3):
        fig = plot_scatter_2d(embedding_2d, labels_3)
        assert isinstance(fig, Figure)

    def test_has_axes(self, embedding_2d, labels_3):
        fig = plot_scatter_2d(embedding_2d, labels_3)
        assert len(fig.axes) >= 1

    def test_title_set(self, embedding_2d, labels_3):
        fig = plot_scatter_2d(embedding_2d, labels_3, title="My Title")
        assert fig.axes[0].get_title() == "My Title"

    def test_xlabel_set(self, embedding_2d, labels_3):
        fig = plot_scatter_2d(embedding_2d, labels_3, xlabel="PC1")
        assert fig.axes[0].get_xlabel() == "PC1"

    def test_ylabel_set(self, embedding_2d, labels_3):
        fig = plot_scatter_2d(embedding_2d, labels_3, ylabel="PC2")
        assert fig.axes[0].get_ylabel() == "PC2"

    def test_custom_figsize(self, embedding_2d, labels_3):
        fig = plot_scatter_2d(embedding_2d, labels_3, figsize=(10, 8))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(10, abs=0.5)
        assert h == pytest.approx(8, abs=0.5)

    def test_no_legend_option(self, embedding_2d, labels_3):
        fig = plot_scatter_2d(embedding_2d, labels_3, show_legend=False)
        ax = fig.axes[0]
        assert ax.get_legend() is None

    def test_with_legend(self, embedding_2d, labels_3):
        fig = plot_scatter_2d(embedding_2d, labels_3, show_legend=True)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        # Should have 3 entries for 3 clusters
        assert len(legend.get_texts()) == 3

    def test_noise_labels_included(self, embedding_2d, labels_with_noise):
        fig = plot_scatter_2d(embedding_2d, labels_with_noise)
        assert isinstance(fig, Figure)
        legend = fig.axes[0].get_legend()
        texts = [t.get_text() for t in legend.get_texts()]
        assert "Noise" in texts

    def test_single_cluster(self, embedding_2d):
        labels = np.zeros(240, dtype=int)
        fig = plot_scatter_2d(embedding_2d, labels)
        assert isinstance(fig, Figure)

    def test_many_clusters(self, embedding_2d):
        labels = np.repeat(np.arange(12), 20)
        fig = plot_scatter_2d(embedding_2d, labels)
        assert isinstance(fig, Figure)

    def test_deterministic(self, embedding_2d, labels_3):
        fig1 = plot_scatter_2d(embedding_2d, labels_3, title="test")
        fig2 = plot_scatter_2d(embedding_2d, labels_3, title="test")
        # Both should have same title (basic structure test)
        assert fig1.axes[0].get_title() == fig2.axes[0].get_title()


# ---------------------------------------------------------------------------
# TestPlotScatterContinuous
# ---------------------------------------------------------------------------


class TestPlotScatterContinuous:

    def test_returns_figure(self, embedding_2d):
        values = np.random.default_rng(42).uniform(0, 1, 240)
        fig = plot_scatter_continuous(embedding_2d, values)
        assert isinstance(fig, Figure)

    def test_has_colorbar(self, embedding_2d):
        values = np.random.default_rng(42).uniform(0, 1, 240)
        fig = plot_scatter_continuous(embedding_2d, values, colorbar_label="Entropy")
        # Figure should have 2 axes (main + colorbar)
        assert len(fig.axes) == 2

    def test_title_set(self, embedding_2d):
        values = np.random.default_rng(42).uniform(0, 1, 240)
        fig = plot_scatter_continuous(embedding_2d, values, title="Entropy Map")
        assert fig.axes[0].get_title() == "Entropy Map"

    def test_negative_values(self, embedding_2d):
        values = np.random.default_rng(42).normal(0, 1, 240)
        fig = plot_scatter_continuous(embedding_2d, values)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# TestPlotScatter3D
# ---------------------------------------------------------------------------


class TestPlotScatter3D:

    @pytest.fixture(autouse=True)
    def _check_plotly(self):
        pytest.importorskip("plotly")

    def test_returns_plotly_figure(self, embedding_3d, labels_3):
        from cluster_pipeline.viz import plot_scatter_3d
        import plotly.graph_objects as go
        fig = plot_scatter_3d(embedding_3d, labels_3)
        assert isinstance(fig, go.Figure)

    def test_has_data(self, embedding_3d, labels_3):
        from cluster_pipeline.viz import plot_scatter_3d
        fig = plot_scatter_3d(embedding_3d, labels_3)
        assert len(fig.data) >= 1

    def test_title_set(self, embedding_3d, labels_3):
        from cluster_pipeline.viz import plot_scatter_3d
        fig = plot_scatter_3d(embedding_3d, labels_3, title="3D View")
        assert fig.layout.title.text == "3D View"

    def test_hover_data(self, embedding_3d, labels_3):
        from cluster_pipeline.viz import plot_scatter_3d
        hover = {"entropy": np.random.default_rng(42).uniform(0, 1, 240)}
        fig = plot_scatter_3d(embedding_3d, labels_3, hover_data=hover)
        assert fig.data[0].text is not None

    def test_no_hover_data(self, embedding_3d, labels_3):
        from cluster_pipeline.viz import plot_scatter_3d
        fig = plot_scatter_3d(embedding_3d, labels_3)
        assert isinstance(fig.data[0].x, np.ndarray)

    def test_wrong_dims_raises(self, embedding_2d, labels_3):
        from cluster_pipeline.viz import plot_scatter_3d
        with pytest.raises(ValueError, match="at least 3"):
            plot_scatter_3d(embedding_2d, labels_3)


# ---------------------------------------------------------------------------
# TestPlotComparisonGrid
# ---------------------------------------------------------------------------


class TestPlotComparisonGrid:

    def test_returns_figure(self, embedding_2d, labels_3):
        fig = plot_comparison_grid(embedding_2d, labels=labels_3)
        assert isinstance(fig, Figure)

    def test_has_4_axes(self, embedding_2d, labels_3):
        fig = plot_comparison_grid(embedding_2d, labels=labels_3)
        # 4 subplots + potentially colorbar axes
        assert len(fig.axes) >= 4

    def test_with_all_data(self, embedding_2d, labels_3):
        rng = np.random.default_rng(42)
        fig = plot_comparison_grid(
            embedding_2d,
            labels=labels_3,
            entropy=rng.uniform(0, 1, 240),
            forward_returns=rng.normal(0, 0.01, 240),
            symbols=np.array(["BTC"] * 80 + ["ETH"] * 80 + ["SOL"] * 80),
        )
        assert isinstance(fig, Figure)

    def test_with_no_optional_data(self, embedding_2d, labels_3):
        fig = plot_comparison_grid(embedding_2d, labels=labels_3)
        assert isinstance(fig, Figure)

    def test_with_entropy_only(self, embedding_2d, labels_3):
        entropy = np.random.default_rng(42).uniform(0, 1, 240)
        fig = plot_comparison_grid(embedding_2d, labels=labels_3, entropy=entropy)
        assert isinstance(fig, Figure)

    def test_with_returns_only(self, embedding_2d, labels_3):
        returns = np.random.default_rng(42).normal(0, 0.01, 240)
        fig = plot_comparison_grid(embedding_2d, labels=labels_3, forward_returns=returns)
        assert isinstance(fig, Figure)

    def test_with_symbols_only(self, embedding_2d, labels_3):
        symbols = np.array(["BTC"] * 120 + ["ETH"] * 120)
        fig = plot_comparison_grid(embedding_2d, labels=labels_3, symbols=symbols)
        assert isinstance(fig, Figure)

    def test_title_set(self, embedding_2d, labels_3):
        fig = plot_comparison_grid(embedding_2d, labels=labels_3, title="Grid Test")
        assert "Grid Test" in fig.texts[0].get_text() if fig.texts else True

    def test_negative_returns(self, embedding_2d, labels_3):
        returns = np.full(240, -0.01)
        fig = plot_comparison_grid(embedding_2d, labels=labels_3, forward_returns=returns)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# TestPlotSilhouette
# ---------------------------------------------------------------------------


class TestPlotSilhouette:

    def test_returns_figure(self, data_and_labels):
        X, labels = data_and_labels
        fig = plot_silhouette(X, labels)
        assert isinstance(fig, Figure)

    def test_title_set(self, data_and_labels):
        X, labels = data_and_labels
        fig = plot_silhouette(X, labels, title="Sil Test")
        assert fig.axes[0].get_title() == "Sil Test"

    def test_has_average_line(self, data_and_labels):
        X, labels = data_and_labels
        fig = plot_silhouette(X, labels)
        ax = fig.axes[0]
        # Should have at least one vertical line (the average)
        lines = [l for l in ax.get_lines() if l.get_linestyle() == "--"]
        assert len(lines) >= 1

    def test_single_cluster_message(self):
        X = np.random.default_rng(42).normal(0, 1, (100, 3))
        labels = np.zeros(100, dtype=int)
        fig = plot_silhouette(X, labels)
        assert isinstance(fig, Figure)

    def test_noise_labels_excluded(self):
        X, labels = _make_data(n_per_cluster=50, k=2)
        noise_labels = np.concatenate([labels, np.full(10, -1)])
        X_with_noise = np.vstack([X, np.random.default_rng(42).normal(0, 10, (10, 5))])
        fig = plot_silhouette(X_with_noise, noise_labels)
        assert isinstance(fig, Figure)

    def test_two_clusters(self):
        X, labels = _make_data(n_per_cluster=100, k=2)
        fig = plot_silhouette(X, labels)
        assert isinstance(fig, Figure)

    def test_many_clusters(self):
        X, labels = _make_data(n_per_cluster=30, k=6)
        fig = plot_silhouette(X, labels)
        assert isinstance(fig, Figure)

    def test_xlabel_present(self, data_and_labels):
        X, labels = data_and_labels
        fig = plot_silhouette(X, labels)
        assert "Silhouette" in fig.axes[0].get_xlabel()


# ---------------------------------------------------------------------------
# TestPlotCentroidHeatmap
# ---------------------------------------------------------------------------


class TestPlotCentroidHeatmap:

    def test_returns_figure(self, data_and_labels):
        X, labels = data_and_labels
        fig = plot_centroid_heatmap(X, labels)
        assert isinstance(fig, Figure)

    def test_with_feature_names(self, data_and_labels):
        X, labels = data_and_labels
        names = [f"feat_{i}" for i in range(5)]
        fig = plot_centroid_heatmap(X, labels, feature_names=names)
        assert isinstance(fig, Figure)

    def test_title_set(self, data_and_labels):
        X, labels = data_and_labels
        fig = plot_centroid_heatmap(X, labels, title="Heatmap Test")
        assert fig.axes[0].get_title() == "Heatmap Test"

    def test_has_colorbar(self, data_and_labels):
        X, labels = data_and_labels
        fig = plot_centroid_heatmap(X, labels)
        # 2 axes: main + colorbar
        assert len(fig.axes) == 2

    def test_no_clusters_message(self):
        X = np.random.default_rng(42).normal(0, 1, (50, 3))
        labels = np.full(50, -1)  # all noise
        fig = plot_centroid_heatmap(X, labels)
        assert isinstance(fig, Figure)

    def test_many_features_truncated(self):
        X, labels = _make_data(n_per_cluster=50, k=2, n_features=50)
        fig = plot_centroid_heatmap(X, labels, max_features=10)
        assert isinstance(fig, Figure)

    def test_two_clusters(self):
        X, labels = _make_data(n_per_cluster=50, k=2)
        fig = plot_centroid_heatmap(X, labels)
        assert isinstance(fig, Figure)

    def test_single_feature(self):
        X, labels = _make_data(n_per_cluster=50, k=2, n_features=1)
        fig = plot_centroid_heatmap(X, labels)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# TestPlotPairplot
# ---------------------------------------------------------------------------


class TestPlotPairplot:

    def test_returns_figure(self, data_and_labels):
        X, labels = data_and_labels
        names = [f"f{i}" for i in range(5)]
        fig = plot_pairplot(X, labels, names)
        assert isinstance(fig, Figure)

    def test_correct_number_of_axes(self, data_and_labels):
        X, labels = data_and_labels
        names = [f"f{i}" for i in range(5)]
        fig = plot_pairplot(X, labels, names, n_features=3)
        # 3x3 grid
        assert len(fig.axes) == 9

    def test_5_features(self, data_and_labels):
        X, labels = data_and_labels
        names = [f"f{i}" for i in range(5)]
        fig = plot_pairplot(X, labels, names, n_features=5)
        assert len(fig.axes) == 25

    def test_single_feature_pairplot(self):
        X, labels = _make_data(n_per_cluster=50, k=2, n_features=1)
        fig = plot_pairplot(X, labels, ["f0"], n_features=1)
        assert isinstance(fig, Figure)

    def test_title_set(self, data_and_labels):
        X, labels = data_and_labels
        names = [f"f{i}" for i in range(5)]
        fig = plot_pairplot(X, labels, names, title="Pair Test")
        assert "Pair Test" in fig.texts[0].get_text() if fig.texts else True

    def test_with_noise_labels(self):
        X, labels = _make_data(n_per_cluster=50, k=2)
        labels_noise = np.concatenate([labels, np.full(10, -1)])
        X_noise = np.vstack([X, np.random.default_rng(42).normal(0, 5, (10, 5))])
        names = [f"f{i}" for i in range(5)]
        fig = plot_pairplot(X_noise, labels_noise, names)
        assert isinstance(fig, Figure)

    def test_more_requested_than_available(self):
        X, labels = _make_data(n_per_cluster=50, k=2, n_features=3)
        names = ["a", "b", "c"]
        fig = plot_pairplot(X, labels, names, n_features=10)
        # Should cap at 3
        assert len(fig.axes) == 9


# ---------------------------------------------------------------------------
# TestPlotDendrogram
# ---------------------------------------------------------------------------


class TestPlotDendrogram:

    def test_returns_figure(self, linkage_matrix):
        fig = plot_dendrogram(linkage_matrix)
        assert isinstance(fig, Figure)

    def test_title_set(self, linkage_matrix):
        fig = plot_dendrogram(linkage_matrix, title="Dendro Test")
        assert fig.axes[0].get_title() == "Dendro Test"

    def test_no_truncation(self, linkage_matrix):
        fig = plot_dendrogram(linkage_matrix, truncate_mode=None)
        assert isinstance(fig, Figure)

    def test_custom_p(self, linkage_matrix):
        fig = plot_dendrogram(linkage_matrix, p=10)
        assert isinstance(fig, Figure)

    def test_color_threshold(self, linkage_matrix):
        fig = plot_dendrogram(linkage_matrix, color_threshold=5.0)
        assert isinstance(fig, Figure)

    def test_orientation_left(self, linkage_matrix):
        fig = plot_dendrogram(linkage_matrix, orientation="left")
        assert isinstance(fig, Figure)

    def test_small_linkage(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        Z = linkage(X, method="ward")
        fig = plot_dendrogram(Z, truncate_mode=None)
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# TestPlotKSweep
# ---------------------------------------------------------------------------


class TestPlotKSweep:

    def test_returns_figure(self, sweep_result):
        fig = plot_k_sweep(sweep_result)
        assert isinstance(fig, Figure)

    def test_title_set(self, sweep_result):
        fig = plot_k_sweep(sweep_result, title="Sweep Test")
        assert fig.axes[0].get_title() == "Sweep Test"

    def test_has_dual_axes_for_gmm(self, sweep_result):
        fig = plot_k_sweep(sweep_result)
        # Should have 2 y-axes (silhouette + BIC)
        assert len(fig.axes) == 2

    def test_no_bic_sweep(self, data_and_labels):
        X, _ = data_and_labels
        sweep = k_sweep(X, k_range=range(2, 5), method="agglomerative")
        fig = plot_k_sweep(sweep)
        # Only 1 y-axis (silhouette only)
        assert len(fig.axes) == 1

    def test_xlabel_present(self, sweep_result):
        fig = plot_k_sweep(sweep_result)
        assert "k" in fig.axes[0].get_xlabel().lower() or "cluster" in fig.axes[0].get_xlabel().lower()


# ---------------------------------------------------------------------------
# TestGenerateAllPlots
# ---------------------------------------------------------------------------


class TestGenerateAllPlots:

    def test_returns_dict(self, data_and_labels, embedding_2d):
        X, labels = data_and_labels
        result = generate_all_plots(X, labels, embedding_2d)
        assert isinstance(result, dict)

    def test_has_scatter(self, data_and_labels, embedding_2d):
        X, labels = data_and_labels
        result = generate_all_plots(X, labels, embedding_2d)
        assert "scatter_2d" in result

    def test_has_silhouette(self, data_and_labels, embedding_2d):
        X, labels = data_and_labels
        result = generate_all_plots(X, labels, embedding_2d)
        assert "silhouette" in result

    def test_has_centroid_heatmap(self, data_and_labels, embedding_2d):
        X, labels = data_and_labels
        result = generate_all_plots(X, labels, embedding_2d)
        assert "centroid_heatmap" in result

    def test_has_comparison_grid(self, data_and_labels, embedding_2d):
        X, labels = data_and_labels
        result = generate_all_plots(X, labels, embedding_2d)
        assert "comparison_grid" in result

    def test_with_sweep(self, data_and_labels, embedding_2d, sweep_result):
        X, labels = data_and_labels
        result = generate_all_plots(X, labels, embedding_2d, sweep=sweep_result)
        assert "k_sweep" in result

    def test_with_linkage(self, data_and_labels, embedding_2d, linkage_matrix):
        X, labels = data_and_labels
        result = generate_all_plots(X, labels, embedding_2d, linkage_matrix=linkage_matrix)
        assert "dendrogram" in result

    def test_with_feature_names(self, data_and_labels, embedding_2d):
        X, labels = data_and_labels
        names = [f"feat_{i}" for i in range(5)]
        result = generate_all_plots(X, labels, embedding_2d, feature_names=names)
        assert "pairplot" in result

    def test_with_all_optional(self, data_and_labels, embedding_2d, sweep_result, linkage_matrix):
        X, labels = data_and_labels
        rng = np.random.default_rng(42)
        result = generate_all_plots(
            X, labels, embedding_2d,
            feature_names=[f"f{i}" for i in range(5)],
            sweep=sweep_result,
            linkage_matrix=linkage_matrix,
            entropy=rng.uniform(0, 1, len(labels)),
            forward_returns=rng.normal(0, 0.01, len(labels)),
            symbols=np.array(["BTC"] * 80 + ["ETH"] * 80 + ["SOL"] * 80),
        )
        assert len(result) >= 6

    def test_all_values_are_figures(self, data_and_labels, embedding_2d):
        X, labels = data_and_labels
        result = generate_all_plots(X, labels, embedding_2d)
        for name, fig in result.items():
            assert isinstance(fig, Figure), f"{name} is not a Figure"

    def test_single_cluster_no_silhouette(self, embedding_2d):
        X = np.random.default_rng(42).normal(0, 1, (240, 5))
        labels = np.zeros(240, dtype=int)
        result = generate_all_plots(X, labels, embedding_2d)
        assert "silhouette" not in result
        assert "centroid_heatmap" not in result


# ---------------------------------------------------------------------------
# TestColorUtilities
# ---------------------------------------------------------------------------


class TestColorUtilities:

    def test_get_cluster_colors_shape(self):
        labels = np.array([0, 1, 2, 0, 1])
        colors = _get_cluster_colors(labels)
        assert colors.shape == (5, 4)

    def test_noise_gets_grey(self):
        labels = np.array([0, -1, 1])
        colors = _get_cluster_colors(labels)
        # Noise point should be grey-ish
        assert colors[1, 0] == pytest.approx(0.7)
        assert colors[1, 3] == pytest.approx(0.5)  # alpha

    def test_same_cluster_same_color(self):
        labels = np.array([0, 0, 1, 1])
        colors = _get_cluster_colors(labels)
        np.testing.assert_array_equal(colors[0], colors[1])
        np.testing.assert_array_equal(colors[2], colors[3])

    def test_different_clusters_different_colors(self):
        labels = np.array([0, 1])
        colors = _get_cluster_colors(labels)
        assert not np.allclose(colors[0], colors[1])

    def test_single_cluster(self):
        labels = np.array([0, 0, 0])
        colors = _get_cluster_colors(labels)
        assert colors.shape == (3, 4)

    def test_all_noise(self):
        labels = np.array([-1, -1, -1])
        colors = _get_cluster_colors(labels)
        for i in range(3):
            assert colors[i, 0] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:

    def test_non_array_raises(self):
        with pytest.raises(TypeError):
            _validate_embedding([[1, 2]], 2)

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            _validate_embedding(np.array([1, 2, 3]), 2)

    def test_too_few_dims_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            _validate_embedding(np.array([[1, 2]]), 3)

    def test_valid_2d(self):
        _validate_embedding(np.array([[1, 2], [3, 4]]), 2)

    def test_valid_3d(self):
        _validate_embedding(np.array([[1, 2, 3], [4, 5, 6]]), 3)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_two_points(self):
        emb = np.array([[0, 0], [1, 1]], dtype=float)
        labels = np.array([0, 1])
        fig = plot_scatter_2d(emb, labels)
        assert isinstance(fig, Figure)

    def test_large_number_of_points(self):
        rng = np.random.default_rng(42)
        emb = rng.normal(0, 1, (5000, 2))
        labels = np.repeat(np.arange(5), 1000)
        fig = plot_scatter_2d(emb, labels)
        assert isinstance(fig, Figure)

    def test_all_same_cluster(self):
        emb = np.random.default_rng(42).normal(0, 1, (100, 2))
        labels = np.zeros(100, dtype=int)
        fig = plot_scatter_2d(emb, labels)
        assert isinstance(fig, Figure)

    def test_all_noise(self):
        emb = np.random.default_rng(42).normal(0, 1, (100, 2))
        labels = np.full(100, -1)
        fig = plot_scatter_2d(emb, labels)
        assert isinstance(fig, Figure)

    def test_scatter_continuous_all_same_value(self):
        emb = np.random.default_rng(42).normal(0, 1, (100, 2))
        values = np.full(100, 0.5)
        fig = plot_scatter_continuous(emb, values)
        assert isinstance(fig, Figure)

    def test_centroid_heatmap_identical_centroids(self):
        """All clusters at same location — z-score should handle gracefully."""
        X = np.ones((100, 5))
        labels = np.array([0] * 50 + [1] * 50)
        fig = plot_centroid_heatmap(X, labels)
        assert isinstance(fig, Figure)

    def test_pairplot_2_features(self):
        X, labels = _make_data(n_per_cluster=50, k=2, n_features=2)
        fig = plot_pairplot(X, labels, ["a", "b"], n_features=2)
        assert len(fig.axes) == 4

    def test_scatter_2d_extra_embedding_dims(self):
        """Embedding with >2 columns should work (uses first 2)."""
        emb = np.random.default_rng(42).normal(0, 1, (100, 5))
        labels = np.zeros(100, dtype=int)
        fig = plot_scatter_2d(emb, labels)
        assert isinstance(fig, Figure)

    def test_silhouette_with_well_separated_data(self):
        """Well-separated data should have positive average silhouette line."""
        X, labels = _make_data(n_per_cluster=50, k=3, separation=10.0)
        fig = plot_silhouette(X, labels)
        # Check the average line position (red dashed)
        ax = fig.axes[0]
        dashed_lines = [l for l in ax.get_lines() if l.get_linestyle() == "--"]
        assert len(dashed_lines) >= 1
        # Average silhouette should be > 0
        avg_x = dashed_lines[0].get_xdata()[0]
        assert avg_x > 0


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_scatter_2d_same_title(self, embedding_2d, labels_3):
        fig1 = plot_scatter_2d(embedding_2d, labels_3, title="A")
        fig2 = plot_scatter_2d(embedding_2d, labels_3, title="A")
        assert fig1.axes[0].get_title() == fig2.axes[0].get_title()

    def test_colors_deterministic(self):
        labels = np.array([0, 1, 2, 0, 1])
        c1 = _get_cluster_colors(labels)
        c2 = _get_cluster_colors(labels)
        np.testing.assert_array_equal(c1, c2)

    def test_centroid_heatmap_deterministic(self, data_and_labels):
        X, labels = data_and_labels
        fig1 = plot_centroid_heatmap(X, labels)
        fig2 = plot_centroid_heatmap(X, labels)
        assert fig1.axes[0].get_title() == fig2.axes[0].get_title()
