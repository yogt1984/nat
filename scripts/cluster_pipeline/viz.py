"""
Visualization module for NAT cluster analysis pipeline.

Generates all diagnostic plots required by the decision gate:
scatter projections, silhouette plots, centroid heatmaps, dendrograms,
k-sweep plots, comparison grids, and pairplots.

All functions return matplotlib Figure objects (or Plotly figures for 3D)
so callers can save, display, or compose them freely.

Usage:
    from cluster_pipeline.viz import (
        plot_scatter_2d, plot_scatter_3d, plot_comparison_grid,
        plot_silhouette, plot_centroid_heatmap, plot_pairplot,
        plot_dendrogram, plot_k_sweep,
    )

    fig = plot_scatter_2d(embedding, labels, title="PCA — 3 clusters")
    fig.savefig("pca_clusters.png", dpi=150)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

from .cluster import SweepResult


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

# Default colormap for discrete cluster labels
_CLUSTER_CMAP = "tab10"
# Continuous colormap for heatmaps / entropy / returns
_CONTINUOUS_CMAP = "RdYlBu_r"
_DIVERGING_CMAP = "RdBu_r"


def _get_cluster_colors(labels: np.ndarray, cmap_name: str = _CLUSTER_CMAP) -> np.ndarray:
    """Map integer labels to RGBA colors. Noise (-1) gets grey."""
    cmap = plt.get_cmap(cmap_name)
    unique = sorted(set(labels) - {-1})
    n_clusters = max(len(unique), 1)

    colors = np.zeros((len(labels), 4))
    for i, lab in enumerate(labels):
        if lab == -1:
            colors[i] = [0.7, 0.7, 0.7, 0.5]  # grey for noise
        else:
            idx = unique.index(lab) if lab in unique else 0
            colors[i] = cmap(idx / max(n_clusters - 1, 1))

    return colors


# ---------------------------------------------------------------------------
# 1. 2D Scatter (PCA, UMAP, t-SNE)
# ---------------------------------------------------------------------------


def plot_scatter_2d(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    title: str = "2D Projection",
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    point_size: float = 10,
    alpha: float = 0.6,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = _CLUSTER_CMAP,
    show_legend: bool = True,
) -> Figure:
    """
    2D scatter plot of projected data colored by cluster label.

    Args:
        embedding: (n_samples, 2) projected coordinates
        labels: (n_samples,) cluster assignments (-1 = noise)
        title: plot title
        xlabel/ylabel: axis labels
        point_size: marker size
        alpha: marker transparency
        figsize: figure size in inches
        cmap: colormap name for cluster colors
        show_legend: whether to show cluster legend

    Returns:
        matplotlib Figure
    """
    _validate_embedding(embedding, 2)

    fig, ax = plt.subplots(figsize=figsize)
    colors = _get_cluster_colors(labels, cmap)

    ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=colors, s=point_size, alpha=alpha, edgecolors="none",
    )

    if show_legend:
        _add_cluster_legend(ax, labels, cmap)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_scatter_continuous(
    embedding: np.ndarray,
    values: np.ndarray,
    *,
    title: str = "2D Projection",
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    colorbar_label: str = "Value",
    point_size: float = 10,
    alpha: float = 0.6,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = _CONTINUOUS_CMAP,
) -> Figure:
    """
    2D scatter colored by a continuous variable (entropy, returns, etc.).
    """
    _validate_embedding(embedding, 2)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=values, cmap=cmap, s=point_size, alpha=alpha, edgecolors="none",
    )
    fig.colorbar(sc, ax=ax, label=colorbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# 2. 3D Interactive (Plotly)
# ---------------------------------------------------------------------------


def plot_scatter_3d(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    title: str = "3D Projection",
    hover_data: Optional[Dict[str, np.ndarray]] = None,
    point_size: float = 3,
    opacity: float = 0.7,
):
    """
    3D interactive scatter plot using Plotly.

    Args:
        embedding: (n_samples, 3) projected coordinates
        labels: cluster assignments
        hover_data: optional dict of arrays to show on hover
        point_size: marker size
        opacity: marker opacity

    Returns:
        plotly.graph_objects.Figure (or None if plotly not installed)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for 3D plots. Install with: pip install plotly")

    _validate_embedding(embedding, 3)

    hover_text = None
    if hover_data:
        hover_text = []
        for i in range(len(labels)):
            parts = [f"Cluster: {labels[i]}"]
            for key, vals in hover_data.items():
                parts.append(f"{key}: {vals[i]:.4f}" if isinstance(vals[i], float) else f"{key}: {vals[i]}")
            hover_text.append("<br>".join(parts))

    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            color=labels,
            colorscale="Viridis",
            opacity=opacity,
            colorbar=dict(title="Cluster"),
        ),
        text=hover_text,
        hoverinfo="text" if hover_text else "x+y+z",
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3",
        ),
        width=800,
        height=600,
    )

    return fig


# ---------------------------------------------------------------------------
# 3. Comparison Grid (same projection, 4 colorings)
# ---------------------------------------------------------------------------


def plot_comparison_grid(
    embedding: np.ndarray,
    *,
    labels: np.ndarray,
    entropy: Optional[np.ndarray] = None,
    forward_returns: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None,
    title: str = "Comparison Grid",
    point_size: float = 8,
    figsize: Tuple[float, float] = (16, 12),
) -> Figure:
    """
    2x2 grid: same projection colored 4 different ways.

    Per the spec, this is the key visual test: if structure appears
    regardless of coloring, the clusters are real.

    Panels:
      - Top-left: Cluster label
      - Top-right: Entropy level (continuous)
      - Bottom-left: Forward return sign (green/red)
      - Bottom-right: Symbol (BTC/ETH/SOL)
    """
    _validate_embedding(embedding, 2)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, y=0.98)

    # Panel 1: Cluster labels
    ax = axes[0, 0]
    colors = _get_cluster_colors(labels)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=point_size, alpha=0.6, edgecolors="none")
    _add_cluster_legend(ax, labels)
    ax.set_title("Cluster Label")
    ax.grid(True, alpha=0.3)

    # Panel 2: Entropy (continuous)
    ax = axes[0, 1]
    if entropy is not None:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=entropy, cmap=_CONTINUOUS_CMAP,
                        s=point_size, alpha=0.6, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="Entropy")
        ax.set_title("Entropy Level")
    else:
        ax.text(0.5, 0.5, "No entropy data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Entropy Level (N/A)")
    ax.grid(True, alpha=0.3)

    # Panel 3: Forward return sign
    ax = axes[1, 0]
    if forward_returns is not None:
        ret_colors = np.where(forward_returns >= 0, "green", "red")
        ax.scatter(embedding[:, 0], embedding[:, 1], c=ret_colors, s=point_size, alpha=0.6, edgecolors="none")
        ax.set_title("Forward Return Sign")
        # Manual legend
        ax.scatter([], [], c="green", label="Positive")
        ax.scatter([], [], c="red", label="Negative")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No return data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Forward Return Sign (N/A)")
    ax.grid(True, alpha=0.3)

    # Panel 4: Symbol
    ax = axes[1, 1]
    if symbols is not None:
        unique_syms = sorted(set(symbols))
        sym_cmap = plt.get_cmap("Set2")
        for i, sym in enumerate(unique_syms):
            mask = symbols == sym
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[sym_cmap(i / max(len(unique_syms) - 1, 1))],
                       s=point_size, alpha=0.6, label=sym, edgecolors="none")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Symbol")
    else:
        ax.text(0.5, 0.5, "No symbol data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Symbol (N/A)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Silhouette Plot
# ---------------------------------------------------------------------------


def plot_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    title: str = "Silhouette Plot",
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    """
    Per-cluster silhouette distribution plot.

    Each cluster is a horizontal band. Wider bands with values > 0
    indicate well-separated clusters. Negative values indicate misclassified points.
    """
    from sklearn.metrics import silhouette_samples, silhouette_score

    unique = sorted(set(labels) - {-1})
    if len(unique) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Need >= 2 clusters for silhouette plot",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    # Exclude noise
    mask = labels >= 0
    X_clean = X[mask]
    labels_clean = labels[mask]

    sil_samples = silhouette_samples(X_clean, labels_clean)
    sil_avg = silhouette_score(X_clean, labels_clean)

    fig, ax = plt.subplots(figsize=figsize)
    y_lower = 0
    cmap = plt.get_cmap(_CLUSTER_CMAP)

    for i, cluster_id in enumerate(unique):
        cluster_sil = sil_samples[labels_clean == cluster_id]
        cluster_sil = np.sort(cluster_sil)
        n = len(cluster_sil)

        y_upper = y_lower + n
        color = cmap(i / max(len(unique) - 1, 1))
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, cluster_sil,
            facecolor=color, edgecolor=color, alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * n, str(cluster_id),
                fontsize=10, fontweight="bold")
        y_lower = y_upper + 2  # gap between clusters

    ax.axvline(x=sil_avg, color="red", linestyle="--", linewidth=1.5,
               label=f"Avg: {sil_avg:.3f}")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(title)
    ax.set_yticks([])
    ax.legend(loc="upper right")
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# 5. Centroid Heatmap
# ---------------------------------------------------------------------------


def plot_centroid_heatmap(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    *,
    title: str = "Cluster Centroids (z-scored)",
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = _DIVERGING_CMAP,
    max_features: int = 30,
) -> Figure:
    """
    Heatmap of cluster centroids (z-scored per feature).

    Rows = clusters, columns = features. Red = high relative value,
    blue = low. Shows what each cluster "looks like" in feature space.
    """
    unique = sorted(set(labels) - {-1})
    if not unique:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No clusters", ha="center", va="center", transform=ax.transAxes)
        return fig

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    # Truncate if too many features
    if n_features > max_features:
        feature_names = feature_names[:max_features]
        X = X[:, :max_features]
        n_features = max_features

    # Compute centroids
    centroids = np.zeros((len(unique), n_features))
    for i, c in enumerate(unique):
        centroids[i] = X[labels == c].mean(axis=0)

    # Z-score per feature (across clusters)
    means = centroids.mean(axis=0)
    stds = centroids.std(axis=0)
    stds[stds < 1e-10] = 1.0
    z_centroids = (centroids - means) / stds

    if figsize is None:
        figsize = (max(8, n_features * 0.4), max(3, len(unique) * 0.8))

    fig, ax = plt.subplots(figsize=figsize)

    vmax = max(abs(z_centroids.min()), abs(z_centroids.max()), 1.0)
    im = ax.imshow(z_centroids, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels([f"Cluster {c}" for c in unique])
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="Z-score")
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# 6. Pairplot (top features)
# ---------------------------------------------------------------------------


def plot_pairplot(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    *,
    n_features: int = 5,
    title: str = "Feature Pairplot",
    point_size: float = 5,
    alpha: float = 0.4,
    figsize_per_cell: float = 2.5,
) -> Figure:
    """
    Pairplot of top N features colored by cluster.

    Shows scatter plots of all pairs of the first n_features columns.
    Diagonal shows 1D histograms per cluster.
    """
    n = min(n_features, X.shape[1])
    fig, axes = plt.subplots(n, n, figsize=(figsize_per_cell * n, figsize_per_cell * n))
    if n == 1:
        axes = np.array([[axes]])

    colors = _get_cluster_colors(labels)
    unique = sorted(set(labels) - {-1})
    cmap = plt.get_cmap(_CLUSTER_CMAP)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histogram per cluster
                for k, c in enumerate(unique):
                    mask = labels == c
                    color = cmap(k / max(len(unique) - 1, 1))
                    ax.hist(X[mask, i], bins=20, alpha=0.5, color=color, density=True)
            else:
                ax.scatter(X[:, j], X[:, i], c=colors, s=point_size, alpha=alpha, edgecolors="none")

            if j == 0:
                ax.set_ylabel(feature_names[i] if i < len(feature_names) else f"f{i}", fontsize=7)
            if i == n - 1:
                ax.set_xlabel(feature_names[j] if j < len(feature_names) else f"f{j}", fontsize=7)

            ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# 7. Dendrogram
# ---------------------------------------------------------------------------


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    *,
    title: str = "Hierarchical Clustering Dendrogram",
    figsize: Tuple[float, float] = (12, 6),
    truncate_mode: Optional[str] = "lastp",
    p: int = 30,
    color_threshold: Optional[float] = None,
    orientation: str = "top",
) -> Figure:
    """
    Dendrogram from a scipy linkage matrix.

    Use with cluster.compute_linkage() output.

    Args:
        linkage_matrix: (n-1, 4) array from scipy.cluster.hierarchy.linkage
        title: plot title
        truncate_mode: "lastp" shows only last p merged clusters
        p: number of leaf nodes to show when truncating
        color_threshold: distance threshold for coloring branches
        orientation: "top", "bottom", "left", "right"
    """
    fig, ax = plt.subplots(figsize=figsize)

    kwargs = dict(
        Z=linkage_matrix,
        ax=ax,
        orientation=orientation,
        leaf_rotation=90 if orientation in ("top", "bottom") else 0,
        leaf_font_size=8,
    )
    if truncate_mode:
        kwargs["truncate_mode"] = truncate_mode
        kwargs["p"] = p
    if color_threshold is not None:
        kwargs["color_threshold"] = color_threshold

    scipy_dendrogram(**kwargs)

    ax.set_title(title)
    ax.set_ylabel("Distance")
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# 8. k-Sweep Plot
# ---------------------------------------------------------------------------


def plot_k_sweep(
    sweep: SweepResult,
    *,
    title: str = "k-Sweep: Cluster Count Selection",
    figsize: Tuple[float, float] = (12, 5),
) -> Figure:
    """
    Plot silhouette score and BIC across k values from a k-sweep.

    Dual y-axis: left = silhouette (higher is better), right = BIC (lower is better).
    Vertical lines mark best k by each metric.
    """
    k_vals = sweep.k_range

    fig, ax1 = plt.subplots(figsize=figsize)

    # Silhouette (left axis)
    color_sil = "tab:blue"
    ax1.plot(k_vals, sweep.silhouettes, "o-", color=color_sil, linewidth=2, label="Silhouette")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score", color=color_sil)
    ax1.tick_params(axis="y", labelcolor=color_sil)
    ax1.axvline(sweep.best_k_silhouette, color=color_sil, linestyle="--", alpha=0.5,
                label=f"Best k (sil) = {sweep.best_k_silhouette}")

    # BIC (right axis) if available
    has_bic = any(b is not None for b in sweep.bics)
    if has_bic:
        ax2 = ax1.twinx()
        color_bic = "tab:red"
        bic_vals = [b if b is not None else np.nan for b in sweep.bics]
        ax2.plot(k_vals, bic_vals, "s--", color=color_bic, linewidth=2, label="BIC")
        ax2.set_ylabel("BIC", color=color_bic)
        ax2.tick_params(axis="y", labelcolor=color_bic)
        if sweep.best_k_bic is not None:
            ax2.axvline(sweep.best_k_bic, color=color_bic, linestyle=":", alpha=0.5,
                        label=f"Best k (BIC) = {sweep.best_k_bic}")
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    else:
        ax1.legend(loc="upper right")

    ax1.set_xticks(k_vals)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all standard plots
# ---------------------------------------------------------------------------


def generate_all_plots(
    X: np.ndarray,
    labels: np.ndarray,
    embedding_2d: np.ndarray,
    *,
    feature_names: Optional[List[str]] = None,
    sweep: Optional[SweepResult] = None,
    linkage_matrix: Optional[np.ndarray] = None,
    entropy: Optional[np.ndarray] = None,
    forward_returns: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None,
    method_name: str = "PCA",
) -> Dict[str, Figure]:
    """
    Generate all standard diagnostic plots.

    Returns dict mapping plot name -> matplotlib Figure.
    """
    plots = {}

    # 1. 2D scatter
    plots["scatter_2d"] = plot_scatter_2d(
        embedding_2d, labels, title=f"{method_name} — Cluster Labels",
    )

    # 2. Silhouette
    unique = set(labels) - {-1}
    if len(unique) >= 2:
        plots["silhouette"] = plot_silhouette(X, labels)

    # 3. Centroid heatmap
    if len(unique) >= 2:
        plots["centroid_heatmap"] = plot_centroid_heatmap(
            X, labels, feature_names=feature_names,
        )

    # 4. Comparison grid
    plots["comparison_grid"] = plot_comparison_grid(
        embedding_2d,
        labels=labels,
        entropy=entropy,
        forward_returns=forward_returns,
        symbols=symbols,
    )

    # 5. Pairplot
    if feature_names and X.shape[1] >= 2:
        plots["pairplot"] = plot_pairplot(
            X, labels, feature_names,
            n_features=min(5, X.shape[1]),
        )

    # 6. Dendrogram
    if linkage_matrix is not None:
        plots["dendrogram"] = plot_dendrogram(linkage_matrix)

    # 7. k-sweep
    if sweep is not None:
        plots["k_sweep"] = plot_k_sweep(sweep)

    return plots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_embedding(embedding: np.ndarray, expected_dims: int) -> None:
    if not isinstance(embedding, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(embedding)}")
    if embedding.ndim != 2:
        raise ValueError(f"Expected 2D array, got {embedding.ndim}D")
    if embedding.shape[1] < expected_dims:
        raise ValueError(
            f"Expected at least {expected_dims} columns, got {embedding.shape[1]}"
        )


def _add_cluster_legend(ax, labels: np.ndarray, cmap_name: str = _CLUSTER_CMAP) -> None:
    """Add a legend with one entry per cluster."""
    cmap = plt.get_cmap(cmap_name)
    unique = sorted(set(labels))
    n_clusters = len([u for u in unique if u != -1])

    for i, lab in enumerate(unique):
        if lab == -1:
            color = [0.7, 0.7, 0.7, 0.5]
            name = "Noise"
        else:
            idx = [u for u in unique if u != -1].index(lab)
            color = cmap(idx / max(n_clusters - 1, 1))
            name = f"Cluster {lab}"
        ax.scatter([], [], c=[color], label=name, s=30)

    ax.legend(loc="upper right", fontsize=8, markerscale=1.5)
