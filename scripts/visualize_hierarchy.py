"""
Hierarchical profiling visualizations (Phase 8, Task 8.1).

Generates 7 plot types from ProfilingResult:
  1. hierarchy_overview  — macro regime + micro state stacked timelines
  2. derivative_pca      — PCA scatter colored by hierarchical state
  3. signatures          — entry/exit derivative heatmaps per state
  4. return_violins      — multi-horizon forward return distributions
  5. transition_graph    — directed graph of state transitions
  6. drift_dashboard     — rolling LL + drift flags over time
  7. structure_test      — Hopkins + dip test gauge visualization

Usage:
    from visualize_hierarchy import visualize_all
    visualize_all(profiling_result, prices=prices, output_dir="reports/figures")

    # Or via CLI:
    nat visualize hierarchy --data data/features --output reports/figures
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from cluster_pipeline.hierarchy import ProfilingResult, HierarchicalLabels
from cluster_pipeline.transitions import empirical_transitions
from cluster_pipeline.characterize import compute_signatures, return_profile

# ---------------------------------------------------------------------------
# Global palette — consistent across all plots
# ---------------------------------------------------------------------------

REGIME_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
STATE_COLORS = [
    "#1565C0", "#42A5F5", "#90CAF9",  # regime 0 shades
    "#BF360C", "#FF7043", "#FFAB91",  # regime 1 shades
    "#2E7D32", "#66BB6A", "#A5D6A7",  # regime 2 shades
    "#6A1B9A", "#AB47BC", "#CE93D8",  # regime 3 shades
    "#E65100", "#FFA726", "#FFCC80",  # regime 4 shades
    "#00838F", "#26C6DA", "#80DEEA",  # regime 5 shades
]


def _state_color(state_id: int, label_map: Dict) -> str:
    """Get color for a global micro state, grouped by parent regime."""
    regime_id, local_id = label_map.get(state_id, (0, 0))
    idx = regime_id * 3 + min(local_id, 2)
    return STATE_COLORS[idx % len(STATE_COLORS)]


def _regime_color(regime_id: int) -> str:
    return REGIME_COLORS[regime_id % len(REGIME_COLORS)]


# ---------------------------------------------------------------------------
# 1. Hierarchy Overview
# ---------------------------------------------------------------------------


def plot_hierarchy_overview(
    result: ProfilingResult,
    output_dir: Path,
    figsize=(20, 8),
) -> Path:
    """Macro regime timeline (top) + micro state timeline (bottom)."""
    hierarchy = result.hierarchy
    n = len(hierarchy.macro_labels)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # --- Macro timeline ---
    ax = axes[0]
    for i in range(n):
        r = int(hierarchy.macro_labels[i])
        ax.axvspan(i - 0.5, i + 0.5, color=_regime_color(r), alpha=0.7)
    ax.step(range(n), hierarchy.macro_labels, color="black", linewidth=0.6, where="mid")
    ax.set_ylabel("Macro Regime")
    ax.set_ylim(-0.5, hierarchy.n_macro - 0.5)
    ax.set_yticks(range(hierarchy.n_macro))
    ax.set_yticklabels([f"R{i}" for i in range(hierarchy.n_macro)])
    ax.set_title(
        f"Hierarchy Overview — {hierarchy.n_macro} regimes, "
        f"{hierarchy.n_micro_total} micro states, {n} bars"
    )
    # Legend
    patches = [mpatches.Patch(color=_regime_color(i), label=f"R{i}")
               for i in range(hierarchy.n_macro)]
    ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=hierarchy.n_macro)

    # --- Micro timeline ---
    ax2 = axes[1]
    for i in range(n):
        s = int(hierarchy.micro_labels[i])
        ax2.axvspan(i - 0.5, i + 0.5, color=_state_color(s, hierarchy.label_map), alpha=0.7)
    ax2.step(range(n), hierarchy.micro_labels, color="black", linewidth=0.6, where="mid")
    ax2.set_ylabel("Micro State")
    ax2.set_ylim(-0.5, hierarchy.n_micro_total - 0.5)
    ax2.set_yticks(range(hierarchy.n_micro_total))
    labels = []
    for s in range(hierarchy.n_micro_total):
        r, l = hierarchy.label_map[s]
        labels.append(f"R{r}_S{l}")
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Bar Index")

    # Micro legend
    patches2 = [mpatches.Patch(color=_state_color(s, hierarchy.label_map), label=labels[s])
                for s in range(hierarchy.n_micro_total)]
    ax2.legend(handles=patches2, loc="upper right", fontsize=7, ncol=min(hierarchy.n_micro_total, 6))

    plt.tight_layout()
    path = output_dir / "hierarchy_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Derivative PCA
# ---------------------------------------------------------------------------


def plot_derivative_pca(
    result: ProfilingResult,
    output_dir: Path,
    figsize=(16, 7),
) -> Path:
    """PCA scatter of macro-reduced space, colored by hierarchical micro labels."""
    macro_pca = result.macro.pca_result
    hierarchy = result.hierarchy

    X_reduced = macro_pca.X_reduced
    if X_reduced.shape[1] < 2:
        # Skip if only 1 component
        return output_dir / "derivative_pca.png"

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left: colored by micro state ---
    ax = axes[0]
    for s in range(hierarchy.n_micro_total):
        mask = hierarchy.micro_labels == s
        r, l = hierarchy.label_map[s]
        ax.scatter(
            X_reduced[mask, 0], X_reduced[mask, 1],
            c=_state_color(s, hierarchy.label_map),
            label=f"R{r}_S{l} (n={mask.sum()})",
            alpha=0.6, s=40, edgecolors="white", linewidth=0.3,
        )
    ev = macro_pca.explained_variance_ratio
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.set_title("PCA — Hierarchical States")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.2)

    # --- Right: loading arrows ---
    ax2 = axes[1]
    for s in range(hierarchy.n_micro_total):
        mask = hierarchy.micro_labels == s
        ax2.scatter(
            X_reduced[mask, 0], X_reduced[mask, 1],
            c=_state_color(s, hierarchy.label_map), alpha=0.25, s=25,
        )

    loadings = macro_pca.components[:2].T  # (n_features, 2)
    col_names = macro_pca.column_names
    loading_mag = np.sqrt(loadings[:, 0] ** 2 + loadings[:, 1] ** 2)
    top_idx = np.argsort(loading_mag)[-10:]

    scale = np.max(np.abs(X_reduced[:, :2])) / np.max(np.abs(loadings[top_idx]) + 1e-12)
    for idx in top_idx:
        ax2.annotate(
            "", xy=(loadings[idx, 0] * scale, loadings[idx, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )
        name = col_names[idx] if idx < len(col_names) else f"f{idx}"
        # Shorten long names
        if len(name) > 28:
            name = name[:25] + "..."
        ax2.annotate(
            name,
            xy=(loadings[idx, 0] * scale * 1.08, loadings[idx, 1] * scale * 1.08),
            fontsize=6, color="red", fontweight="bold",
        )

    ax2.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax2.set_title("PCA — Top 10 Feature Loadings")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    path = output_dir / "derivative_pca.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Entry/Exit Signatures
# ---------------------------------------------------------------------------


def plot_signatures(
    result: ProfilingResult,
    derivatives: pd.DataFrame,
    output_dir: Path,
    lookback: int = 5,
    top_n_features: int = 12,
    figsize_per_state=(10, 5),
) -> List[Path]:
    """Entry/exit signature heatmaps for each micro state."""
    hierarchy = result.hierarchy
    paths = []

    # Align labels to derivatives length (derivatives may be shorter due to warmup)
    n_deriv = len(derivatives)
    n_labels = len(hierarchy.micro_labels)
    if n_deriv < n_labels:
        labels = hierarchy.micro_labels[n_labels - n_deriv:]
    else:
        labels = hierarchy.micro_labels

    for s in range(hierarchy.n_micro_total):
        sig = compute_signatures(
            derivatives, labels, state_id=s,
            lookback=lookback, min_events=3,
        )
        if sig is None:
            continue

        r, l = hierarchy.label_map[s]

        # Select top features by entry trajectory range
        entry_range = sig.entry_trajectory.max() - sig.entry_trajectory.min()
        exit_range = sig.exit_trajectory.max() - sig.exit_trajectory.min()
        combined_range = entry_range + exit_range
        top_cols = combined_range.nlargest(top_n_features).index.tolist()

        # Concatenate entry and exit for a single heatmap
        entry_sub = sig.entry_trajectory[top_cols]
        exit_sub = sig.exit_trajectory[top_cols]
        combined = pd.concat([entry_sub, exit_sub])

        # Shorten column names
        short = [c.split("_mean")[0][-30:] for c in top_cols]

        fig, ax = plt.subplots(figsize=figsize_per_state)
        im = ax.imshow(
            combined.values.T, cmap="RdBu_r", aspect="auto",
            vmin=-np.percentile(np.abs(combined.values), 95),
            vmax=np.percentile(np.abs(combined.values), 95),
        )
        ax.set_xticks(range(len(combined)))
        x_labels = [str(i) for i in combined.index]
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=7)

        # Draw vertical line at entry boundary
        boundary = lookback - 0.5
        ax.axvline(boundary, color="black", linestyle="--", linewidth=2, label="entry/exit")

        ax.set_xlabel("Relative bar (negative=before entry, positive=after exit)")
        ax.set_title(
            f"R{r}_S{l} — Entry/Exit Signature "
            f"(entries={sig.entry_count}, exits={sig.exit_count})"
        )
        plt.colorbar(im, ax=ax, label="Mean derivative value", shrink=0.8)
        ax.legend(fontsize=8)

        plt.tight_layout()
        path = output_dir / f"signature_R{r}_S{l}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# 4. Return Violins
# ---------------------------------------------------------------------------


def plot_return_violins(
    result: ProfilingResult,
    prices: np.ndarray,
    output_dir: Path,
    horizons: Optional[List[int]] = None,
    figsize=(18, 10),
) -> Path:
    """Multi-horizon forward return violin plots, one column per horizon."""
    if horizons is None:
        horizons = [1, 5, 10, 20]

    hierarchy = result.hierarchy
    n_states = hierarchy.n_micro_total
    prices = np.asarray(prices, dtype=float)
    log_prices = np.log(prices)

    fig, axes = plt.subplots(1, len(horizons), figsize=figsize, sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    state_labels = []
    for s in range(n_states):
        r, l = hierarchy.label_map[s]
        state_labels.append(f"R{r}_S{l}")

    for h_idx, h in enumerate(horizons):
        ax = axes[h_idx]
        all_returns = []
        positions = []
        colors = []

        for s in range(n_states):
            mask = hierarchy.micro_labels == s
            indices = np.where(mask)[0]
            valid = indices[indices + h < len(prices)]
            if len(valid) < 3:
                all_returns.append(np.array([0.0]))
                positions.append(s)
                colors.append(_state_color(s, hierarchy.label_map))
                continue

            rets = log_prices[valid + h] - log_prices[valid]
            all_returns.append(rets)
            positions.append(s)
            colors.append(_state_color(s, hierarchy.label_map))

        parts = ax.violinplot(
            all_returns, positions=positions,
            showmeans=True, showmedians=True, showextrema=False,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax.axhline(0, color="black", linestyle="--", alpha=0.4)
        ax.set_xticks(range(n_states))
        ax.set_xticklabels(state_labels, fontsize=7, rotation=45, ha="right")
        ax.set_title(f"h={h} bars")
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("Forward log-return")
    fig.suptitle("Forward Returns by State and Horizon", fontsize=14, y=1.02)
    plt.tight_layout()
    path = output_dir / "return_violins.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. Transition Graph
# ---------------------------------------------------------------------------


def plot_transition_graph(
    result: ProfilingResult,
    output_dir: Path,
    min_prob: float = 0.05,
    figsize=(12, 10),
) -> Path:
    """Directed graph: nodes = micro states, edges = transition probabilities."""
    hierarchy = result.hierarchy
    trans = empirical_transitions(hierarchy.micro_labels)
    T = trans.matrix
    n_states = hierarchy.n_micro_total

    fig, ax = plt.subplots(figsize=figsize)

    # Layout nodes in a circle
    angles = np.linspace(0, 2 * np.pi, n_states, endpoint=False)
    radius = 3.0
    node_x = radius * np.cos(angles)
    node_y = radius * np.sin(angles)

    # Node sizes proportional to bar count
    bar_fractions = np.array([
        np.sum(hierarchy.micro_labels == s) / len(hierarchy.micro_labels)
        for s in range(n_states)
    ])
    node_sizes = 800 + bar_fractions * 5000

    # Draw edges first (behind nodes)
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            prob = T[i, j]
            if prob < min_prob:
                continue
            # Arrow from i to j
            dx = node_x[j] - node_x[i]
            dy = node_y[j] - node_y[i]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            # Shorten to not overlap nodes
            shrink = 0.35
            sx = node_x[i] + dx * shrink
            sy = node_y[i] + dy * shrink
            ex = node_x[j] - dx * shrink
            ey = node_y[j] - dy * shrink

            ax.annotate(
                "", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="gray",
                    lw=1 + prob * 4,
                    alpha=0.3 + prob * 0.5,
                    connectionstyle="arc3,rad=0.15",
                ),
            )
            # Label with probability
            mx = (sx + ex) / 2 + (dy / dist) * 0.25
            my = (sy + ey) / 2 - (dx / dist) * 0.25
            ax.text(mx, my, f"{prob:.2f}", fontsize=7, color="gray",
                    ha="center", va="center", fontweight="bold")

    # Draw nodes
    for s in range(n_states):
        r, l = hierarchy.label_map[s]
        ax.scatter(
            node_x[s], node_y[s], s=node_sizes[s],
            c=_state_color(s, hierarchy.label_map),
            edgecolors="black", linewidth=1.5, zorder=5,
        )
        ax.text(
            node_x[s], node_y[s], f"R{r}_S{l}\n{bar_fractions[s]:.0%}",
            ha="center", va="center", fontsize=9, fontweight="bold", zorder=6,
        )

    # Self-loops as text annotations
    for s in range(n_states):
        self_p = T[s, s]
        if self_p > min_prob:
            r, l = hierarchy.label_map[s]
            offset = 0.6
            tx = node_x[s] + offset * np.cos(angles[s])
            ty = node_y[s] + offset * np.sin(angles[s])
            ax.text(tx, ty, f"self={self_p:.2f}", fontsize=7,
                    color=_state_color(s, hierarchy.label_map),
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xlim(-radius * 1.6, radius * 1.6)
    ax.set_ylim(-radius * 1.6, radius * 1.6)
    ax.set_aspect("equal")
    ax.set_title(
        f"State Transition Graph — {n_states} states "
        f"(edges with P > {min_prob})",
        fontsize=13,
    )
    ax.axis("off")

    plt.tight_layout()
    path = output_dir / "transition_graph.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. Drift Dashboard
# ---------------------------------------------------------------------------


def plot_drift_dashboard(
    ll_scores: np.ndarray,
    training_p10: float,
    training_p50: float,
    output_dir: Path,
    window: int = 50,
    figsize=(18, 8),
    drift_flags: Optional[np.ndarray] = None,
) -> Path:
    """
    3-panel drift monitoring dashboard.

    Panel 1: Per-bar log-likelihood
    Panel 2: Rolling mean LL vs training percentile bands
    Panel 3: Drift flag timeline

    Args:
        ll_scores: per-bar log-likelihood scores from GMM.score_samples().
        training_p10: 10th percentile LL from training data.
        training_p50: 50th percentile LL from training data.
        output_dir: where to save the figure.
        window: rolling window size for smoothing.
        drift_flags: optional boolean array of per-bar drift flags.
    """
    n = len(ll_scores)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Panel 1: raw LL
    ax = axes[0]
    ax.plot(ll_scores, color="#1565C0", linewidth=0.5, alpha=0.6)
    ax.axhline(training_p10, color="red", linestyle="--", alpha=0.7, label=f"p10={training_p10:.1f}")
    ax.axhline(training_p50, color="green", linestyle="--", alpha=0.7, label=f"p50={training_p50:.1f}")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("Per-Bar Log-Likelihood vs Training Baseline")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.2)

    # Panel 2: rolling mean
    ax2 = axes[1]
    if n >= window:
        rolling_ll = pd.Series(ll_scores).rolling(window, min_periods=1).mean().values
    else:
        rolling_ll = ll_scores
    ax2.plot(rolling_ll, color="#1565C0", linewidth=1.5, label=f"Rolling mean (w={window})")
    ax2.axhline(training_p10, color="red", linestyle="--", alpha=0.7)
    ax2.axhline(training_p50, color="green", linestyle="--", alpha=0.7)
    # Shade drift zone
    ax2.fill_between(
        range(n),
        np.full(n, training_p10 - abs(training_p50 - training_p10)),
        np.full(n, training_p10),
        color="red", alpha=0.08, label="Drift zone",
    )
    ax2.set_ylabel("Rolling Mean LL")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.2)

    # Panel 3: drift flags
    ax3 = axes[2]
    if drift_flags is not None:
        flags = np.asarray(drift_flags, dtype=bool)
    else:
        # Infer: drift when rolling LL < training p10
        flags = rolling_ll < training_p10

    for i in range(n):
        if flags[i]:
            ax3.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.4)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.5])
    ax3.set_yticklabels(["Drift?"])
    ax3.set_xlabel("Bar Index")
    ax3.set_title(f"Drift Flags — {int(np.sum(flags))} / {n} bars flagged ({np.mean(flags):.1%})")
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    path = output_dir / "drift_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 7. Structure Test Visualization
# ---------------------------------------------------------------------------


def plot_structure_test(
    result: ProfilingResult,
    output_dir: Path,
    figsize=(12, 5),
) -> Path:
    """Hopkins + dip test gauge visualization."""
    st = result.structure_test
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Hopkins gauge ---
    ax = axes[0]
    val = st.hopkins_statistic
    threshold = 0.7
    color = "#4CAF50" if val > threshold else ("#FF9800" if val > 0.55 else "#F44336")

    ax.barh(0, val, height=0.5, color=color, edgecolor="black", linewidth=1.5)
    ax.barh(0, 1.0, height=0.5, fill=False, edgecolor="gray", linewidth=0.5)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"threshold={threshold}")
    ax.text(val + 0.02, 0, f"{val:.3f}", va="center", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Hopkins Statistic")
    ax.set_title("Clustering Tendency (Hopkins)")
    ax.legend(fontsize=9, loc="upper left")

    # Color bands
    ax.axvspan(0, 0.55, color="#F44336", alpha=0.06)
    ax.axvspan(0.55, threshold, color="#FF9800", alpha=0.06)
    ax.axvspan(threshold, 1.05, color="#4CAF50", alpha=0.06)
    ax.text(0.27, -0.35, "random", ha="center", fontsize=8, color="gray")
    ax.text(0.625, -0.35, "weak", ha="center", fontsize=8, color="gray")
    ax.text(0.85, -0.35, "clustered", ha="center", fontsize=8, color="gray")

    # --- Dip test gauge ---
    ax2 = axes[1]
    p_val = st.dip_test_p
    threshold_dip = 0.05
    color2 = "#4CAF50" if p_val < threshold_dip else ("#FF9800" if p_val < 0.10 else "#F44336")

    # Plot on log scale for better visibility of small p-values
    ax2.barh(0, max(p_val, 0.001), height=0.5, color=color2, edgecolor="black", linewidth=1.5)
    ax2.barh(0, 1.0, height=0.5, fill=False, edgecolor="gray", linewidth=0.5)
    ax2.axvline(threshold_dip, color="black", linestyle="--", linewidth=2, label=f"alpha={threshold_dip}")
    ax2.text(max(p_val, 0.001) + 0.02, 0, f"p={p_val:.4f}", va="center", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Dip Test p-value")
    ax2.set_title("Multimodality (Hartigan Dip)")
    ax2.legend(fontsize=9, loc="upper right")

    # Color bands (reversed — small p is good)
    ax2.axvspan(0, threshold_dip, color="#4CAF50", alpha=0.06)
    ax2.axvspan(threshold_dip, 0.10, color="#FF9800", alpha=0.06)
    ax2.axvspan(0.10, 1.05, color="#F44336", alpha=0.06)
    ax2.text(0.025, -0.35, "multimodal", ha="center", fontsize=8, color="gray")
    ax2.text(0.075, -0.35, "weak", ha="center", fontsize=8, color="gray")
    ax2.text(0.55, -0.35, "unimodal", ha="center", fontsize=8, color="gray")

    # Overall verdict
    verdict = st.recommendation
    verdict_color = {
        "proceed": "#4CAF50", "weak_structure": "#FF9800", "no_structure": "#F44336"
    }.get(verdict, "gray")
    fig.text(
        0.5, 0.01,
        f"Verdict: {verdict.upper()} (has_structure={st.has_structure})",
        ha="center", fontsize=13, fontweight="bold", color=verdict_color,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = output_dir / "structure_test.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def visualize_all(
    result: ProfilingResult,
    output_dir: str = "reports/figures",
    prices: Optional[np.ndarray] = None,
    derivatives: Optional[pd.DataFrame] = None,
    ll_scores: Optional[np.ndarray] = None,
) -> List[Path]:
    """
    Generate all Phase 8 visualizations from a ProfilingResult.

    Args:
        result: Complete profiling output.
        output_dir: Directory for PNG files.
        prices: Optional price array (enables return violins).
        derivatives: Optional derivative DataFrame (enables signatures).
        ll_scores: Optional per-bar LL scores (enables drift dashboard).

    Returns:
        List of saved file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    print("Generating Phase 8 visualizations...", flush=True)

    # 1. Hierarchy overview
    p = plot_hierarchy_overview(result, out)
    saved.append(p)
    print(f"  [saved] {p.name}", flush=True)

    # 2. Derivative PCA
    p = plot_derivative_pca(result, out)
    saved.append(p)
    print(f"  [saved] {p.name}", flush=True)

    # 3. Signatures (needs derivatives)
    if derivatives is not None:
        paths = plot_signatures(result, derivatives, out)
        saved.extend(paths)
        for pp in paths:
            print(f"  [saved] {pp.name}", flush=True)
        if not paths:
            print("  [skip] signatures — insufficient entry/exit events", flush=True)
    else:
        print("  [skip] signatures — no derivatives provided", flush=True)

    # 4. Return violins (needs prices)
    if prices is not None:
        p = plot_return_violins(result, prices, out)
        saved.append(p)
        print(f"  [saved] {p.name}", flush=True)
    else:
        print("  [skip] return_violins — no prices provided", flush=True)

    # 5. Transition graph
    p = plot_transition_graph(result, out)
    saved.append(p)
    print(f"  [saved] {p.name}", flush=True)

    # 6. Drift dashboard (needs LL scores)
    if ll_scores is not None and result.training_stats:
        p10 = result.training_stats.get("log_likelihood_p10", np.percentile(ll_scores, 10))
        p50 = result.training_stats.get("log_likelihood_p50", np.percentile(ll_scores, 50))
        p = plot_drift_dashboard(ll_scores, p10, p50, out)
        saved.append(p)
        print(f"  [saved] {p.name}", flush=True)
    else:
        print("  [skip] drift_dashboard — no LL scores or training stats", flush=True)

    # 7. Structure test
    p = plot_structure_test(result, out)
    saved.append(p)
    print(f"  [saved] {p.name}", flush=True)

    print(f"\nDone. {len(saved)} figures saved to {out}/", flush=True)
    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Run visualizations from collected data (standalone mode)."""
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical profiling visualizations")
    parser.add_argument("--data", default="data/features", help="Data directory")
    parser.add_argument("--output", default="reports/figures", help="Output directory")
    parser.add_argument("--vector", default="entropy", help="Feature vector")
    parser.add_argument("--timeframe", default="15min", help="Bar aggregation timeframe")
    args = parser.parse_args()

    from cluster_pipeline.loader import load_parquet
    from cluster_pipeline.preprocess import aggregate_bars
    from cluster_pipeline.hierarchy import profile

    print("Loading data...", flush=True)
    df = load_parquet(args.data)
    print(f"Loaded {len(df):,} rows", flush=True)

    if "symbol" in df.columns:
        df = df[df["symbol"] == "BTC"].reset_index(drop=True)

    bars = aggregate_bars(df, timeframe=args.timeframe)
    bars_pd = bars.to_pandas() if hasattr(bars, "to_pandas") else bars
    print(f"Aggregated to {len(bars_pd)} bars", flush=True)

    result = profile(
        bars_pd, vector=args.vector, skip_aggregation=True,
        macro_k_range=range(2, 6), micro_k_range=range(2, 6),
    )

    # Generate derivatives for signature plots
    from cluster_pipeline.derivatives import generate_derivatives

    deriv = generate_derivatives(result.bars, vector=args.vector)
    derivatives = deriv.derivatives
    if deriv.warmup_rows > 0:
        derivatives = derivatives.iloc[deriv.warmup_rows:].reset_index(drop=True)

    # Generate prices for return plots
    price_cols = [c for c in result.bars.columns if "midprice" in c.lower() or "close" in c.lower()]
    prices = None
    if price_cols:
        prices = result.bars[price_cols[0]].values

    # Generate LL scores for drift dashboard
    ll_scores = None
    if not result.macro.early_exit:
        from sklearn.mixture import GaussianMixture

        pca = result.macro.pca_result
        cols = [c for c in pca.column_names if c in derivatives.columns]
        if cols:
            X = derivatives[cols].values
            X_std = (X - pca.mean) / np.where(pca.std > 1e-12, pca.std, 1.0)
            X_red = X_std @ pca.components.T
            gmm = GaussianMixture(
                n_components=result.macro.k, covariance_type="full",
                n_init=5, random_state=42,
            ).fit(X_red)
            ll_scores = gmm.score_samples(X_red)

    visualize_all(result, args.output, prices=prices, derivatives=derivatives, ll_scores=ll_scores)


if __name__ == "__main__":
    main()
