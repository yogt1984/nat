"""
Feature time series visualization.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Style configuration
STYLE = {
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
}

COLORS = ['#58a6ff', '#3fb950', '#f85149', '#d29922', '#a371f7', '#79c0ff']


def apply_style():
    """Apply dark theme style."""
    plt.rcParams.update(STYLE)


def plot_features(
    df: pd.DataFrame,
    features: List[str],
    symbol: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    title: Optional[str] = None,
    normalize: bool = False,
) -> plt.Figure:
    """
    Plot multiple features on the same time axis.

    Args:
        df: DataFrame with datetime index or column
        features: List of feature column names to plot
        symbol: Filter to specific symbol
        figsize: Figure size
        title: Plot title
        normalize: If True, z-score normalize features for comparison

    Returns:
        matplotlib Figure
    """
    apply_style()

    # Filter by symbol
    if symbol and 'symbol' in df.columns:
        df = df[df['symbol'] == symbol].copy()

    if df.empty:
        raise ValueError(f"No data for symbol {symbol}")

    # Ensure datetime
    if 'datetime' not in df.columns:
        ts_col = 'timestamp_ns' if 'timestamp_ns' in df.columns else 'timestamp'
        df['datetime'] = pd.to_datetime(df[ts_col], unit='ns')

    fig, ax = plt.subplots(figsize=figsize)

    for i, feature in enumerate(features):
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found, skipping")
            continue

        y = df[feature].values
        x = df['datetime'].values

        if normalize:
            y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-10)

        color = COLORS[i % len(COLORS)]
        ax.plot(x, y, color=color, label=feature, linewidth=0.8, alpha=0.9)

    ax.set_xlabel('Time')
    ax.set_ylabel('Value' if not normalize else 'Z-Score')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    elif symbol:
        ax.set_title(f'{symbol} Features', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_feature_panel(
    df: pd.DataFrame,
    features: List[str],
    symbol: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    share_x: bool = True,
) -> plt.Figure:
    """
    Plot each feature in a separate subplot panel.

    Args:
        df: DataFrame with features
        features: List of feature names
        symbol: Filter to specific symbol
        figsize: Figure size (auto-calculated if None)
        share_x: Share x-axis across panels

    Returns:
        matplotlib Figure
    """
    apply_style()

    # Filter by symbol
    if symbol and 'symbol' in df.columns:
        df = df[df['symbol'] == symbol].copy()

    # Ensure datetime
    if 'datetime' not in df.columns:
        ts_col = 'timestamp_ns' if 'timestamp_ns' in df.columns else 'timestamp'
        df['datetime'] = pd.to_datetime(df[ts_col], unit='ns')

    n_features = len(features)
    if figsize is None:
        figsize = (14, 2.5 * n_features)

    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=share_x)
    if n_features == 1:
        axes = [axes]

    x = df['datetime'].values

    for i, (ax, feature) in enumerate(zip(axes, features)):
        if feature not in df.columns:
            ax.text(0.5, 0.5, f"'{feature}' not found",
                   ha='center', va='center', transform=ax.transAxes)
            continue

        y = df[feature].values
        color = COLORS[i % len(COLORS)]

        ax.plot(x, y, color=color, linewidth=0.8)
        ax.fill_between(x, y, alpha=0.2, color=color)
        ax.set_ylabel(feature, fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add mean line
        mean_val = np.nanmean(y)
        ax.axhline(mean_val, color='white', linestyle='--', alpha=0.3, linewidth=0.5)

    # Format x-axis on bottom panel
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel('Time')
    plt.xticks(rotation=45)

    if symbol:
        fig.suptitle(f'{symbol} Feature Panel', fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def plot_rolling_stats(
    df: pd.DataFrame,
    feature: str,
    windows: List[int] = [10, 50, 200],
    symbol: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot feature with rolling mean and standard deviation bands.

    Args:
        df: DataFrame with features
        feature: Feature column name
        windows: List of rolling window sizes
        symbol: Filter to specific symbol
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    apply_style()

    if symbol and 'symbol' in df.columns:
        df = df[df['symbol'] == symbol].copy()

    if 'datetime' not in df.columns:
        ts_col = 'timestamp_ns' if 'timestamp_ns' in df.columns else 'timestamp'
        df['datetime'] = pd.to_datetime(df[ts_col], unit='ns')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    x = df['datetime'].values
    y = df[feature].values

    # Top panel: feature with rolling means
    ax1.plot(x, y, color='#8b949e', linewidth=0.5, alpha=0.5, label='Raw')

    for i, window in enumerate(windows):
        rolling_mean = pd.Series(y).rolling(window).mean()
        color = COLORS[i % len(COLORS)]
        ax1.plot(x, rolling_mean, color=color, linewidth=1.2, label=f'MA({window})')

    ax1.set_ylabel(feature)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: rolling volatility
    for i, window in enumerate(windows):
        rolling_std = pd.Series(y).rolling(window).std()
        color = COLORS[i % len(COLORS)]
        ax2.plot(x, rolling_std, color=color, linewidth=1.0, label=f'Std({window})')

    ax2.set_ylabel(f'{feature} Volatility')
    ax2.set_xlabel('Time')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    title = f'{feature} Rolling Statistics'
    if symbol:
        title = f'{symbol} - {title}'
    fig.suptitle(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig
