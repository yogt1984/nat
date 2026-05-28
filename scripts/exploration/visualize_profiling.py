"""
Profiling Visualizations — Market State Discovery & Intuition Building

Generates comprehensive visualizations for the top clustering configurations
to help develop intuition about market regime structure.

Usage:
    PYTHONPATH=. python scripts/visualize_profiling.py
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from cluster_pipeline.preprocess import preprocess, aggregate_bars
from cluster_pipeline.loader import load_parquet
from cluster_pipeline.cluster import (
    k_sweep, cluster_quality, bootstrap_stability, temporal_stability
)

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Color palette for states
STATE_COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4', '#E91E63']
STATE_CMAP = ListedColormap(STATE_COLORS[:7])

# Top configurations to visualize
CONFIGS = [
    ('5min',  'entropy',    2),
    ('15min', 'orderflow',  3),
    ('15min', 'derived',    2),
    ('2h',    'orderflow',  4),
    ('5min',  'orderflow',  4),
    ('15min', 'illiquidity', 4),  # highest separation, interesting temporal dynamics
]


def load_and_cluster(df, tf, vector, k):
    """Aggregate, preprocess, cluster, return everything needed for plotting."""
    bars = aggregate_bars(df, timeframe=tf)
    X, cols, meta = preprocess(bars, vector=vector, scaler='zscore', clip_sigma=5.0)

    sweep = k_sweep(X, k_range=range(k, k+1))
    labels = sweep.results[0].labels
    q = cluster_quality(X, labels)

    # PCA
    pca = PCA(n_components=min(X.shape[1], 10))
    X_pca = pca.fit_transform(X)

    # t-SNE on PCA-reduced (faster, more stable)
    n_tsne = min(3, X_pca.shape[1])
    perp = min(30, len(X) // 4)
    tsne = TSNE(n_components=2, perplexity=max(5, perp), random_state=42)
    X_tsne = tsne.fit_transform(X_pca[:, :n_tsne])

    return {
        'bars': bars, 'X': X, 'cols': cols, 'meta': meta,
        'labels': labels, 'quality': q, 'pca': pca, 'X_pca': X_pca,
        'X_tsne': X_tsne, 'k': k,
    }


# ─── PLOT 1: PCA Scatter with Feature Loadings ─────────────────────────────

def plot_pca_with_loadings(data, tf, vector, k):
    """PCA projection colored by cluster, with top feature loading arrows."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    X_pca = data['X_pca']
    labels = data['labels']
    pca = data['pca']
    cols = data['cols']

    # Left: PCA scatter
    ax = axes[0]
    for i in range(k):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=STATE_COLORS[i], label=f'State {i} (n={mask.sum()})',
                   alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title(f'{vector} @ {tf} — PCA Projection (k={k})')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Right: Feature loading biplot
    ax2 = axes[1]
    for i in range(k):
        mask = labels == i
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=STATE_COLORS[i], alpha=0.3, s=30)

    # Top 10 loading arrows
    loadings = pca.components_[:2].T  # (n_features, 2)
    loading_mag = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_idx = np.argsort(loading_mag)[-10:]

    scale = np.max(np.abs(X_pca[:, :2])) / np.max(np.abs(loadings[top_idx]))
    for idx in top_idx:
        ax2.annotate('', xy=(loadings[idx, 0]*scale, loadings[idx, 1]*scale),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        # Shorten column name for readability
        name = cols[idx].replace(f'{vector}_', '').replace('_mean', '').replace('_std', '(s)')
        if len(name) > 25:
            name = name[:22] + '...'
        ax2.annotate(name, xy=(loadings[idx, 0]*scale*1.1, loadings[idx, 1]*scale*1.1),
                     fontsize=7, color='red', fontweight='bold')

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax2.set_title(f'{vector} @ {tf} — Feature Loadings (top 10)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / f'pca_{vector}_{tf}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] pca_{vector}_{tf}.png', flush=True)


# ─── PLOT 2: Centroid Heatmap — What Each State Means ───────────────────────

def plot_centroid_heatmap(data, tf, vector, k):
    """Z-scored cluster centroids — reveals what makes each state distinct."""
    X = data['X']
    labels = data['labels']
    cols = data['cols']

    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i] = X[labels == i].mean(axis=0)

    # Shorten column names
    short_names = []
    for c in cols:
        name = c.replace(f'{vector}_', '').replace('_mean', '(m)').replace('_std', '(s)')
        name = name.replace('_last', '(L)').replace('_slope', '(sl)')
        if len(name) > 30:
            name = name[:27] + '...'
        short_names.append(name)

    fig, ax = plt.subplots(figsize=(max(8, k*2.5), max(8, len(cols)*0.3)))
    im = ax.imshow(centroids.T, cmap='RdBu_r', aspect='auto',
                   vmin=-2, vmax=2)

    ax.set_xticks(range(k))
    ax.set_xticklabels([f'State {i}' for i in range(k)], fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_title(f'{vector} @ {tf} — Cluster Centroids (z-scored)', fontsize=14)

    plt.colorbar(im, ax=ax, label='Z-score', shrink=0.6)
    plt.tight_layout()
    fig.savefig(OUT / f'centroids_{vector}_{tf}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] centroids_{vector}_{tf}.png', flush=True)


# ─── PLOT 3: State Timeline with Price ──────────────────────────────────────

def plot_state_timeline(data, tf, vector, k, df_raw):
    """State labels as colored bands with price overlay, per symbol."""
    bars = data['bars']
    labels = data['labels']

    # Get bar timestamps
    if 'bar_start' in bars.columns:
        times = pd.to_datetime(bars['bar_start'])
    else:
        times = pd.RangeIndex(len(bars))

    symbols = bars['symbol'].unique() if 'symbol' in bars.columns else ['ALL']

    fig, axes = plt.subplots(len(symbols) + 1, 1,
                             figsize=(20, 3 * (len(symbols) + 1)),
                             sharex=True)
    if len(symbols) + 1 == 1:
        axes = [axes]

    # Top plot: full state timeline
    ax = axes[0]
    for i in range(len(labels)):
        ax.axvspan(i - 0.5, i + 0.5, color=STATE_COLORS[labels[i]], alpha=0.6)
    ax.set_ylabel('State')
    ax.set_yticks(range(k))
    ax.set_title(f'{vector} @ {tf} — State Timeline (all symbols)', fontsize=13)

    # Overlay state as a step function
    ax.step(range(len(labels)), labels, color='black', linewidth=0.8, where='mid')
    ax.set_ylim(-0.5, k - 0.5)

    # Per-symbol: state bands + price approximation using midprice if available
    for s_idx, sym in enumerate(symbols):
        ax = axes[s_idx + 1]
        if 'symbol' in bars.columns:
            sym_mask = bars['symbol'] == sym
            sym_labels = labels[sym_mask.values]
            sym_indices = np.where(sym_mask.values)[0]
        else:
            sym_labels = labels
            sym_indices = np.arange(len(labels))

        # Background state colors
        for i, idx in enumerate(sym_indices):
            ax.axvspan(idx - 0.5, idx + 0.5,
                       color=STATE_COLORS[sym_labels[i]], alpha=0.3)

        # Try to plot a price-like feature
        price_candidates = [c for c in bars.columns
                            if any(p in c.lower() for p in ['midprice', 'mid_price', 'close', 'last'])]
        if price_candidates:
            price_col = price_candidates[0]
            if 'symbol' in bars.columns:
                prices = bars.loc[sym_mask, price_col].values
            else:
                prices = bars[price_col].values
            ax.plot(sym_indices, prices, color='black', linewidth=1.2, label=price_col)
            ax.legend(loc='upper left', fontsize=8)

        ax.set_ylabel(sym, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Bar Index')
    plt.tight_layout()
    fig.savefig(OUT / f'timeline_{vector}_{tf}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] timeline_{vector}_{tf}.png', flush=True)


# ─── PLOT 4: Return Distribution Per State ──────────────────────────────────

def plot_return_distributions(data, tf, vector, k):
    """Forward return distributions per state — the money plot."""
    bars = data['bars']
    labels = data['labels']

    # Compute forward returns from any price-like column
    price_candidates = [c for c in bars.columns
                        if any(p in c.lower() for p in ['midprice', 'mid_price', 'close'])]
    if not price_candidates:
        # Use first numeric column as proxy
        price_candidates = [c for c in bars.columns
                            if bars[c].dtype in (np.float64, np.float32)
                            and c not in ['bar_start', 'bar_end', 'tick_count']]

    if not price_candidates:
        print(f'  [skip] return distributions — no price column found', flush=True)
        return

    price_col = price_candidates[0]
    prices = bars[price_col].values
    fwd_returns = np.zeros(len(prices))
    fwd_returns[:-1] = (prices[1:] - prices[:-1]) / (np.abs(prices[:-1]) + 1e-10)
    fwd_returns[-1] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Overlaid histograms
    ax = axes[0]
    for i in range(k):
        mask = labels == i
        rets = fwd_returns[mask]
        rets = rets[~np.isnan(rets)]
        if len(rets) > 2:
            ax.hist(rets, bins=30, alpha=0.5, color=STATE_COLORS[i],
                    label=f'State {i} (n={len(rets)}, μ={np.mean(rets):.4f})',
                    density=True, edgecolor='white')
    ax.set_xlabel('Forward 1-bar Return')
    ax.set_ylabel('Density')
    ax.set_title(f'{vector} @ {tf} — Return Distribution per State')
    ax.legend(fontsize=8)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    box_data = []
    box_labels_list = []
    for i in range(k):
        mask = labels == i
        rets = fwd_returns[mask]
        rets = rets[~np.isnan(rets)]
        if len(rets) > 0:
            box_data.append(rets)
            box_labels_list.append(f'State {i}')
    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels_list, patch_artist=True)
        for patch, color in zip(bp['boxes'], STATE_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax2.set_ylabel('Forward Return')
    ax2.set_title('Return Box Plot per State')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # Cumulative returns per state (equity curve)
    ax3 = axes[2]
    for i in range(k):
        mask = labels == i
        rets = fwd_returns.copy()
        rets[~mask] = 0
        rets[np.isnan(rets)] = 0
        cum = np.cumsum(rets)
        ax3.plot(cum, color=STATE_COLORS[i], label=f'State {i}', linewidth=1.5)
    ax3.set_xlabel('Bar Index')
    ax3.set_ylabel('Cumulative Return')
    ax3.set_title('Cumulative Return (long only in each state)')
    ax3.legend(fontsize=9)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / f'returns_{vector}_{tf}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] returns_{vector}_{tf}.png', flush=True)


# ─── PLOT 5: Transition Matrix Heatmap ──────────────────────────────────────

def plot_transition_matrix(data, tf, vector, k):
    """Transition probability matrix — reveals regime dynamics."""
    labels = data['labels']

    # Count transitions
    trans = np.zeros((k, k))
    for i in range(len(labels) - 1):
        trans[labels[i], labels[i + 1]] += 1

    # Normalize to probabilities
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Probability matrix
    ax = axes[0]
    im = ax.imshow(trans_prob, cmap='YlOrRd', vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            color = 'white' if trans_prob[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{trans_prob[i, j]:.2f}',
                    ha='center', va='center', color=color, fontsize=14, fontweight='bold')
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([f'To {i}' for i in range(k)])
    ax.set_yticklabels([f'From {i}' for i in range(k)])
    ax.set_title(f'{vector} @ {tf} — Transition Probabilities')
    plt.colorbar(im, ax=ax, label='P(next | current)', shrink=0.8)

    # Self-transition (persistence) bar chart
    ax2 = axes[1]
    self_trans = np.diag(trans_prob)
    bars = ax2.bar(range(k), self_trans, color=STATE_COLORS[:k], edgecolor='black')
    ax2.axhline(0.7, color='red', linestyle='--', alpha=0.7, label='Persistence threshold (0.7)')
    ax2.set_xticks(range(k))
    ax2.set_xticklabels([f'State {i}' for i in range(k)])
    ax2.set_ylabel('Self-transition Rate')
    ax2.set_title('State Persistence')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for bar, val in zip(bars, self_trans):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f'{val:.2f}', ha='center', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUT / f'transitions_{vector}_{tf}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] transitions_{vector}_{tf}.png', flush=True)


# ─── PLOT 6: State Duration Distribution ────────────────────────────────────

def plot_state_durations(data, tf, vector, k):
    """How long does each state last before transitioning?"""
    labels = data['labels']

    durations = {i: [] for i in range(k)}
    current_state = labels[0]
    current_len = 1

    for i in range(1, len(labels)):
        if labels[i] == current_state:
            current_len += 1
        else:
            durations[current_state].append(current_len)
            current_state = labels[i]
            current_len = 1
    durations[current_state].append(current_len)

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = []
    for i in range(k):
        if durations[i]:
            pos = i
            positions.append(pos)
            parts = ax.violinplot([durations[i]], positions=[pos], showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(STATE_COLORS[i])
                pc.set_alpha(0.7)

            mean_d = np.mean(durations[i])
            median_d = np.median(durations[i])
            max_d = np.max(durations[i])
            ax.text(pos, max_d + 0.5,
                    f'μ={mean_d:.1f}, med={median_d:.0f}, max={max_d}',
                    ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(range(k))
    ax.set_xticklabels([f'State {i}' for i in range(k)])
    ax.set_ylabel(f'Duration (number of {tf} bars)')
    ax.set_title(f'{vector} @ {tf} — State Duration Distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / f'durations_{vector}_{tf}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] durations_{vector}_{tf}.png', flush=True)


# ─── PLOT 7: t-SNE with Multiple Overlays ──────────────────────────────────

def plot_tsne_overlays(data, tf, vector, k):
    """t-SNE colored by: cluster, symbol, time, top feature."""
    X_tsne = data['X_tsne']
    labels = data['labels']
    bars = data['bars']
    X = data['X']
    cols = data['cols']

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top-left: cluster labels
    ax = axes[0, 0]
    for i in range(k):
        mask = labels == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=STATE_COLORS[i], label=f'State {i}', alpha=0.7, s=40)
    ax.set_title('Colored by Cluster State')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Top-right: symbol
    ax = axes[0, 1]
    sym_colors = {'BTC': '#F7931A', 'ETH': '#627EEA', 'SOL': '#9945FF'}
    if 'symbol' in bars.columns:
        for sym in bars['symbol'].unique():
            mask = (bars['symbol'] == sym).values
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=sym_colors.get(sym, 'gray'), label=sym, alpha=0.7, s=40)
        ax.legend(fontsize=9)
    ax.set_title('Colored by Symbol')
    ax.grid(True, alpha=0.2)

    # Bottom-left: temporal position (early → late)
    ax = axes[1, 0]
    time_color = np.linspace(0, 1, len(X_tsne))
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=time_color,
                    cmap='viridis', alpha=0.7, s=40)
    plt.colorbar(sc, ax=ax, label='Time (early → late)')
    ax.set_title('Colored by Time')
    ax.grid(True, alpha=0.2)

    # Bottom-right: most discriminative feature (highest variance ratio across clusters)
    ax = axes[1, 1]
    # Find feature with highest between-cluster variance
    best_feat_idx = 0
    best_f_ratio = 0
    for f_idx in range(X.shape[1]):
        groups = [X[labels == i, f_idx] for i in range(k)]
        grand_mean = X[:, f_idx].mean()
        between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        within = sum(np.sum((g - g.mean())**2) for g in groups)
        if within > 0:
            f_ratio = between / within
            if f_ratio > best_f_ratio:
                best_f_ratio = f_ratio
                best_feat_idx = f_idx

    feat_vals = X[:, best_feat_idx]
    feat_name = cols[best_feat_idx].replace(f'{vector}_', '')
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=feat_vals,
                    cmap='RdYlBu_r', alpha=0.7, s=40)
    plt.colorbar(sc, ax=ax, label=feat_name)
    ax.set_title(f'Most Discriminative Feature: {feat_name}')
    ax.grid(True, alpha=0.2)

    fig.suptitle(f'{vector} @ {tf} — t-SNE Manifold (4 views)', fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT / f'tsne_{vector}_{tf}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] tsne_{vector}_{tf}.png', flush=True)


# ─── PLOT 8: Cross-Asset State Alignment ────────────────────────────────────

def plot_cross_asset_alignment(data, tf, vector, k):
    """Do BTC, ETH, SOL enter the same state simultaneously?"""
    bars = data['bars']
    labels = data['labels']

    if 'symbol' not in bars.columns:
        return

    symbols = sorted(bars['symbol'].unique())
    if len(symbols) < 2:
        return

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    # Top: state timelines stacked by symbol
    ax = axes[0]
    for s_idx, sym in enumerate(symbols):
        mask = (bars['symbol'] == sym).values
        sym_labels = labels[mask]
        for i in range(len(sym_labels)):
            ax.barh(s_idx, 1, left=i, color=STATE_COLORS[sym_labels[i]],
                    edgecolor='none', height=0.8)

    ax.set_yticks(range(len(symbols)))
    ax.set_yticklabels(symbols, fontsize=12, fontweight='bold')
    ax.set_xlabel('Bar Index (within symbol)')
    ax.set_title(f'{vector} @ {tf} — Cross-Asset State Alignment')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=STATE_COLORS[i], label=f'State {i}') for i in range(k)]
    ax.legend(handles=legend_elements, loc='upper right')

    # Bottom: state agreement heatmap over time
    ax2 = axes[1]
    sym_labels_dict = {}
    for sym in symbols:
        mask = (bars['symbol'] == sym).values
        sym_labels_dict[sym] = labels[mask]

    min_len = min(len(v) for v in sym_labels_dict.values())

    # Count how many symbols agree on state at each time step
    agreement = np.zeros(min_len)
    for t in range(min_len):
        states_at_t = [sym_labels_dict[sym][t] for sym in symbols]
        # Fraction that match the mode
        from collections import Counter
        counts = Counter(states_at_t)
        agreement[t] = counts.most_common(1)[0][1] / len(symbols)

    ax2.fill_between(range(min_len), agreement, alpha=0.6, color='#2196F3')
    ax2.plot(range(min_len), agreement, color='#1565C0', linewidth=1)
    ax2.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Full agreement')
    ax2.axhline(1/len(symbols), color='red', linestyle='--', alpha=0.5, label='Random chance')
    ax2.set_ylabel('Cross-Asset Agreement')
    ax2.set_xlabel('Bar Index')
    ax2.set_title('Fraction of Symbols in Same State')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / f'crossasset_{vector}_{tf}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] crossasset_{vector}_{tf}.png', flush=True)


# ─── PLOT 9: Feature Evolution by State ─────────────────────────────────────

def plot_feature_evolution(data, tf, vector, k):
    """Top 6 features over time, colored by state — builds raw intuition."""
    X = data['X']
    labels = data['labels']
    cols = data['cols']

    # Pick top 6 by between-cluster F-ratio
    f_ratios = []
    for f_idx in range(X.shape[1]):
        groups = [X[labels == i, f_idx] for i in range(k)]
        grand_mean = X[:, f_idx].mean()
        between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        within = sum(np.sum((g - g.mean())**2) for g in groups)
        f_ratios.append(between / within if within > 0 else 0)

    top6 = np.argsort(f_ratios)[-6:][::-1]

    fig, axes = plt.subplots(6, 1, figsize=(20, 18), sharex=True)

    for plot_idx, feat_idx in enumerate(top6):
        ax = axes[plot_idx]
        feat_vals = X[:, feat_idx]

        # Plot with state-colored segments
        for i in range(len(feat_vals) - 1):
            ax.plot([i, i+1], [feat_vals[i], feat_vals[i+1]],
                    color=STATE_COLORS[labels[i]], linewidth=1.5, alpha=0.8)

        # State background
        for i in range(len(labels)):
            ax.axvspan(i - 0.5, i + 0.5, color=STATE_COLORS[labels[i]], alpha=0.08)

        feat_name = cols[feat_idx].replace(f'{vector}_', '')
        ax.set_ylabel(feat_name, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)

        # Mark state transitions
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                ax.axvline(i, color='black', linestyle=':', alpha=0.4, linewidth=0.8)

    axes[-1].set_xlabel('Bar Index')
    axes[0].set_title(f'{vector} @ {tf} — Top 6 Discriminative Features Over Time', fontsize=13)

    # Add state legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=STATE_COLORS[i], label=f'State {i}') for i in range(k)]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=k)

    plt.tight_layout()
    fig.savefig(OUT / f'features_{vector}_{tf}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] features_{vector}_{tf}.png', flush=True)


# ─── PLOT 10: Grand Summary Dashboard ──────────────────────────────────────

def plot_summary_dashboard(all_results):
    """Single overview comparing all configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    configs = list(all_results.keys())
    n = len(configs)

    # Top-left: Silhouette comparison
    ax = axes[0, 0]
    sils = [all_results[c]['quality'].silhouette for c in configs]
    labels_list = [f'{c[1]}\n{c[0]}' for c in configs]
    bars_plot = ax.bar(range(n), sils, color=[STATE_COLORS[i % len(STATE_COLORS)] for i in range(n)])
    ax.axhline(0.25, color='red', linestyle='--', label='Q1 threshold')
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels_list, fontsize=9)
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Cluster Separation (Q1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: PCA explained variance
    ax2 = axes[0, 1]
    for i, c in enumerate(configs):
        pca = all_results[c]['pca']
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumvar)+1), cumvar,
                 color=STATE_COLORS[i % len(STATE_COLORS)],
                 label=f'{c[1]}@{c[0]}', linewidth=2)
    ax2.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='90% var')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('PCA Dimensionality')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Bottom-left: k vs silhouette bubble chart
    ax3 = axes[1, 0]
    for i, c in enumerate(configs):
        k = c[2]
        sil = all_results[c]['quality'].silhouette
        n_bars = len(all_results[c]['labels'])
        ax3.scatter(k, sil, s=n_bars, color=STATE_COLORS[i % len(STATE_COLORS)],
                    alpha=0.7, edgecolors='black', linewidth=1.5)
        ax3.annotate(f'{c[1]}@{c[0]}', (k + 0.1, sil), fontsize=8)
    ax3.axhline(0.25, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Silhouette Score')
    ax3.set_title('k vs Separation (bubble size = n_bars)')
    ax3.grid(True, alpha=0.3)

    # Bottom-right: text summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = "PROFILING SUMMARY\n" + "=" * 40 + "\n\n"
    for c in configs:
        d = all_results[c]
        q = d['quality']
        summary_text += f"{c[1]} @ {c[0]} (k={c[2]})\n"
        summary_text += f"  Silhouette: {q.silhouette:.3f}\n"
        summary_text += f"  Sizes: {list(q.cluster_sizes.values())}\n"
        summary_text += f"  PCA 90% at {np.searchsorted(np.cumsum(d['pca'].explained_variance_ratio_), 0.9)+1} components\n\n"

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Market Regime Profiling — Configuration Comparison', fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] summary_dashboard.png', flush=True)


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    print('Loading data...', flush=True)
    df = load_parquet('./data/features')
    print(f'Loaded {len(df):,} rows\n', flush=True)

    all_results = {}

    for tf, vector, k in CONFIGS:
        tag = f'{vector} @ {tf} (k={k})'
        print(f'=== {tag} ===', flush=True)

        try:
            data = load_and_cluster(df, tf, vector, k)
            all_results[(tf, vector, k)] = data

            plot_pca_with_loadings(data, tf, vector, k)
            plot_centroid_heatmap(data, tf, vector, k)
            plot_state_timeline(data, tf, vector, k, df)
            plot_return_distributions(data, tf, vector, k)
            plot_transition_matrix(data, tf, vector, k)
            plot_state_durations(data, tf, vector, k)
            plot_tsne_overlays(data, tf, vector, k)
            plot_cross_asset_alignment(data, tf, vector, k)
            plot_feature_evolution(data, tf, vector, k)

        except Exception as e:
            print(f'  ERROR: {e}', flush=True)

        print(flush=True)

    if all_results:
        print('=== Summary Dashboard ===', flush=True)
        plot_summary_dashboard(all_results)

    print(f'\nDone. {len(list(OUT.glob("*.png")))} figures saved to {OUT}/', flush=True)


if __name__ == '__main__':
    main()
