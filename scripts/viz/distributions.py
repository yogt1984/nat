"""
Distribution analysis and visualization.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns (excluding metadata)."""
    meta_cols = ['timestamp_ns', 'timestamp', 'symbol', 'sequence_id', 'datetime']
    return [c for c in df.columns if c not in meta_cols]


def plot_distributions(
    df: pd.DataFrame,
    features: List[str],
    symbol: Optional[str] = None,
    bins: int = 100,
    figsize: Optional[Tuple[int, int]] = None,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot histograms of feature distributions.

    Args:
        df: DataFrame with features
        features: List of features to plot
        symbol: Filter to specific symbol
        bins: Number of histogram bins
        figsize: Figure size
        log_scale: Use log scale for y-axis

    Returns:
        matplotlib Figure
    """
    from .features import apply_style, COLORS
    apply_style()

    if symbol and 'symbol' in df.columns:
        df = df[df['symbol'] == symbol]

    features = [f for f in features if f in df.columns]
    n = len(features)

    if n == 0:
        raise ValueError("No valid features to plot")

    # Calculate grid dimensions
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, feature in enumerate(features):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        data = df[feature].dropna()
        color = COLORS[idx % len(COLORS)]

        ax.hist(data, bins=bins, color=color, alpha=0.7, edgecolor='none')

        if log_scale:
            ax.set_yscale('log')

        # Add statistics
        mean = data.mean()
        std = data.std()
        skew = data.skew()
        kurt = data.kurtosis()

        stats_text = f'μ={mean:.3g}\nσ={std:.3g}\nskew={skew:.2f}\nkurt={kurt:.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))

        ax.axvline(mean, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)

    title = 'Feature Distributions'
    if symbol:
        title = f'{symbol} - {title}'
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def plot_qq(
    df: pd.DataFrame,
    features: List[str],
    symbol: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot Q-Q plots to check normality of features.

    Args:
        df: DataFrame with features
        features: List of features to plot
        symbol: Filter to specific symbol
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from .features import apply_style, COLORS
    apply_style()

    if symbol and 'symbol' in df.columns:
        df = df[df['symbol'] == symbol]

    features = [f for f in features if f in df.columns]
    n = len(features)

    cols = min(3, n)
    rows = (n + cols - 1) // cols

    if figsize is None:
        figsize = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, feature in enumerate(features):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        data = df[feature].dropna()
        color = COLORS[idx % len(COLORS)]

        # Q-Q plot
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
        ax.scatter(osm, osr, alpha=0.5, s=3, color=color)
        ax.plot(osm, slope * osm + intercept, color='white', linestyle='--', linewidth=1)

        # Shapiro-Wilk test (on sample if too large)
        sample = data.sample(min(5000, len(data)), random_state=42)
        _, p_value = stats.shapiro(sample)

        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.set_title(f'{feature}\n(R²={r**2:.3f}, p={p_value:.3g})', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)

    title = 'Q-Q Plots (Normality Check)'
    if symbol:
        title = f'{symbol} - {title}'
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def plot_outliers(
    df: pd.DataFrame,
    features: List[str],
    symbol: Optional[str] = None,
    method: str = 'zscore',
    threshold: float = 3.0,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Visualize outliers in features.

    Args:
        df: DataFrame with features
        features: List of features to analyze
        symbol: Filter to specific symbol
        method: 'zscore' or 'iqr'
        threshold: Threshold for outlier detection
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from .features import apply_style, COLORS
    apply_style()

    if symbol and 'symbol' in df.columns:
        df = df[df['symbol'] == symbol]

    features = [f for f in features if f in df.columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Box plots
    data_to_plot = [df[f].dropna() for f in features]
    bp = ax1.boxplot(data_to_plot, labels=features, patch_artist=True)

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.7)

    ax1.set_ylabel('Value')
    ax1.set_title('Box Plots', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: Outlier counts
    outlier_counts = []
    for feature in features:
        data = df[feature].dropna()

        if method == 'zscore':
            z = np.abs((data - data.mean()) / data.std())
            n_outliers = (z > threshold).sum()
        else:  # IQR
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            n_outliers = ((data < q1 - threshold * iqr) |
                         (data > q3 + threshold * iqr)).sum()

        outlier_counts.append(n_outliers)

    colors = [COLORS[i % len(COLORS)] for i in range(len(features))]
    bars = ax2.barh(features, outlier_counts, color=colors, alpha=0.7)

    # Add percentage labels
    total_rows = len(df)
    for bar, count in zip(bars, outlier_counts):
        pct = count / total_rows * 100
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{count} ({pct:.1f}%)', va='center', fontsize=8)

    ax2.set_xlabel('Number of Outliers')
    ax2.set_title(f'Outlier Counts ({method}, threshold={threshold})',
                  fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    title = 'Outlier Analysis'
    if symbol:
        title = f'{symbol} - {title}'
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def distribution_summary(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate summary statistics for feature distributions.

    Args:
        df: DataFrame with features
        features: Features to summarize (default: all)

    Returns:
        DataFrame with summary statistics
    """
    if features is None:
        features = get_feature_columns(df)

    features = [f for f in features if f in df.columns]

    summary = []
    for feature in features:
        data = df[feature].dropna()

        row = {
            'feature': feature,
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'q25': data.quantile(0.25),
            'median': data.median(),
            'q75': data.quantile(0.75),
            'max': data.max(),
            'skew': data.skew(),
            'kurtosis': data.kurtosis(),
            'nan_pct': df[feature].isna().mean() * 100,
        }
        summary.append(row)

    return pd.DataFrame(summary)
