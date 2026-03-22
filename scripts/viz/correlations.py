"""
Correlation analysis and visualization.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Custom diverging colormap
CORR_CMAP = LinearSegmentedColormap.from_list(
    'corr', ['#f85149', '#161b22', '#3fb950']
)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns (excluding metadata)."""
    meta_cols = ['timestamp_ns', 'timestamp', 'symbol', 'sequence_id', 'datetime']
    return [c for c in df.columns if c not in meta_cols]


def plot_correlation_matrix(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    symbol: Optional[str] = None,
    method: str = 'pearson',
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = False,
) -> plt.Figure:
    """
    Plot correlation matrix heatmap.

    Args:
        df: DataFrame with features
        features: List of features to include (default: all)
        symbol: Filter to specific symbol
        method: Correlation method ('pearson', 'spearman', 'kendall')
        figsize: Figure size
        annot: Show correlation values in cells

    Returns:
        matplotlib Figure
    """
    from .features import apply_style
    apply_style()

    if symbol and 'symbol' in df.columns:
        df = df[df['symbol'] == symbol]

    if features is None:
        features = get_feature_columns(df)

    # Filter to existing columns
    features = [f for f in features if f in df.columns]

    if len(features) < 2:
        raise ValueError("Need at least 2 features for correlation matrix")

    # Calculate correlation matrix
    corr = df[features].corr(method=method)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', rotation=270, labelpad=20)

    # Set ticks
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(features, fontsize=8)

    # Annotate if requested
    if annot:
        for i in range(len(features)):
            for j in range(len(features)):
                val = corr.iloc[i, j]
                color = 'white' if abs(val) > 0.5 else '#c9d1d9'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=7)

    title = f'Correlation Matrix ({method})'
    if symbol:
        title = f'{symbol} - {title}'
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_scatter_matrix(
    df: pd.DataFrame,
    features: List[str],
    symbol: Optional[str] = None,
    sample_size: int = 5000,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot scatter matrix (pairplot) of features.

    Args:
        df: DataFrame with features
        features: List of features to include (max 6 recommended)
        symbol: Filter to specific symbol
        sample_size: Downsample for performance
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from .features import apply_style, COLORS
    apply_style()

    if symbol and 'symbol' in df.columns:
        df = df[df['symbol'] == symbol]

    # Filter to existing columns
    features = [f for f in features if f in df.columns]
    n = len(features)

    if n < 2 or n > 8:
        raise ValueError("Need 2-8 features for scatter matrix")

    # Downsample for performance
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    if figsize is None:
        figsize = (3 * n, 3 * n)

    fig, axes = plt.subplots(n, n, figsize=figsize)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                ax.hist(df[features[i]].dropna(), bins=50, color=COLORS[0],
                       alpha=0.7, edgecolor='none')
            else:
                # Off-diagonal: scatter
                ax.scatter(df[features[j]], df[features[i]],
                          alpha=0.3, s=1, color=COLORS[0])

            # Labels
            if i == n - 1:
                ax.set_xlabel(features[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(features[i], fontsize=8)

            ax.tick_params(labelsize=6)

    title = 'Feature Scatter Matrix'
    if symbol:
        title = f'{symbol} - {title}'
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def plot_pca(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_components: int = 2,
    symbol: Optional[str] = None,
    color_by: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot PCA projection of features.

    Args:
        df: DataFrame with features
        features: Features to include in PCA
        n_components: Number of PCA components (2 or 3)
        symbol: Filter to specific symbol
        color_by: Column to use for coloring points
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from .features import apply_style, COLORS
    apply_style()

    if symbol and 'symbol' in df.columns:
        df = df[df['symbol'] == symbol].copy()

    if features is None:
        features = get_feature_columns(df)

    features = [f for f in features if f in df.columns]

    # Prepare data
    X = df[features].dropna()
    if len(X) < 10:
        raise ValueError("Not enough non-NaN data for PCA")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Plot
    if n_components == 2:
        fig, ax = plt.subplots(figsize=figsize)

        if color_by and color_by in df.columns:
            colors = df.loc[X.index, color_by]
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors,
                                cmap='viridis', alpha=0.5, s=5)
            plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=5, color=COLORS[0])

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax.grid(True, alpha=0.3)

    else:  # 3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                  alpha=0.3, s=5, color=COLORS[0])

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')

    # Title
    total_var = sum(pca.explained_variance_ratio_[:n_components])
    title = f'PCA Projection ({total_var:.1%} variance explained)'
    if symbol:
        title = f'{symbol} - {title}'
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def find_correlated_pairs(
    df: pd.DataFrame,
    threshold: float = 0.8,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Find highly correlated feature pairs.

    Args:
        df: DataFrame with features
        threshold: Minimum absolute correlation
        features: Features to analyze

    Returns:
        DataFrame with correlated pairs
    """
    if features is None:
        features = get_feature_columns(df)

    features = [f for f in features if f in df.columns]
    corr = df[features].corr()

    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append({
                    'feature_1': features[i],
                    'feature_2': features[j],
                    'correlation': r,
                })

    return pd.DataFrame(pairs).sort_values('correlation', key=abs, ascending=False)
