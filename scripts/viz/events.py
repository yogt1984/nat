"""
Event detection and visualization.

Detects and visualizes significant events like:
- Whale flow spikes
- Volatility regime changes
- VPIN spikes (toxicity)
- Large price moves
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


@dataclass
class Event:
    """Represents a detected event."""
    timestamp: pd.Timestamp
    event_type: str
    value: float
    zscore: float
    duration_idx: int = 1


def detect_events(
    df: pd.DataFrame,
    feature: str,
    threshold_std: float = 2.0,
    min_gap: int = 10,
) -> List[Event]:
    """
    Detect events where feature exceeds threshold standard deviations.

    Args:
        df: DataFrame with feature data
        feature: Feature column to analyze
        threshold_std: Number of std devs to trigger event
        min_gap: Minimum samples between events

    Returns:
        List of detected events
    """
    if feature not in df.columns:
        return []

    values = df[feature].values
    mean = np.nanmean(values)
    std = np.nanstd(values)

    if std == 0:
        return []

    zscores = (values - mean) / std

    events = []
    last_event_idx = -min_gap

    for i, (z, v) in enumerate(zip(zscores, values)):
        if abs(z) > threshold_std and (i - last_event_idx) >= min_gap:
            if 'datetime' in df.columns:
                ts = df['datetime'].iloc[i]
            else:
                ts = pd.Timestamp(df.iloc[i].get('timestamp_ns', i), unit='ns')

            events.append(Event(
                timestamp=ts,
                event_type=f"{feature}_{'high' if z > 0 else 'low'}",
                value=v,
                zscore=z,
            ))
            last_event_idx = i

    return events


def detect_all_events(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    threshold_std: float = 2.5,
) -> Dict[str, List[Event]]:
    """
    Detect events across multiple features.

    Args:
        df: DataFrame with features
        features: List of features to analyze (default: key features)
        threshold_std: Threshold in standard deviations

    Returns:
        Dict mapping feature names to event lists
    """
    if features is None:
        # Default key features to monitor
        features = [
            'vpin_10', 'vpin_50',           # Toxicity
            'qty_l1', 'qty_l5',             # Imbalance
            'returns_1m',                    # Volatility
            'kyle_lambda_100',               # Illiquidity
            'whale_net_flow_1h',             # Whale activity
        ]

    events = {}
    for feature in features:
        if feature in df.columns:
            events[feature] = detect_events(df, feature, threshold_std)

    return events


def plot_events(
    df: pd.DataFrame,
    price_col: str = 'midprice',
    event_features: Optional[List[str]] = None,
    symbol: Optional[str] = None,
    threshold_std: float = 2.5,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot price with event markers overlaid.

    Args:
        df: DataFrame with features
        price_col: Column name for price (y-axis)
        event_features: Features to detect events in
        symbol: Filter to specific symbol
        threshold_std: Event detection threshold
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from .features import apply_style, COLORS
    apply_style()

    if symbol and 'symbol' in df.columns:
        df = df[df['symbol'] == symbol].copy()

    if 'datetime' not in df.columns:
        ts_col = 'timestamp_ns' if 'timestamp_ns' in df.columns else 'timestamp'
        df['datetime'] = pd.to_datetime(df[ts_col], unit='ns')

    # Find price column
    if price_col not in df.columns:
        # Try alternatives
        for alt in ['midprice', 'raw_mid_price', 'price', 'close']:
            if alt in df.columns:
                price_col = alt
                break

    # Detect events
    all_events = detect_all_events(df, event_features, threshold_std)

    # Create figure
    n_panels = 1 + len([f for f in all_events if all_events[f]])
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [2] + [1] * (n_panels - 1)})
    if n_panels == 1:
        axes = [axes]

    x = df['datetime'].values

    # Top panel: price with event markers
    ax_price = axes[0]
    if price_col in df.columns:
        ax_price.plot(x, df[price_col], color='#c9d1d9', linewidth=0.8)
    ax_price.set_ylabel('Price')
    ax_price.grid(True, alpha=0.3)

    # Mark events on price chart
    event_colors = {
        'high': '#3fb950',  # Green for high
        'low': '#f85149',   # Red for low
    }

    for feature, events in all_events.items():
        for event in events:
            direction = 'high' if event.zscore > 0 else 'low'
            color = event_colors[direction]
            ax_price.axvline(event.timestamp, color=color, alpha=0.3, linewidth=1)

    # Additional panels: event features
    panel_idx = 1
    for i, (feature, events) in enumerate(all_events.items()):
        if not events or feature not in df.columns:
            continue

        ax = axes[panel_idx]
        color = COLORS[i % len(COLORS)]

        ax.plot(x, df[feature], color=color, linewidth=0.6, alpha=0.8)
        ax.set_ylabel(feature, fontsize=8)
        ax.grid(True, alpha=0.3)

        # Mark events
        for event in events:
            marker_color = '#3fb950' if event.zscore > 0 else '#f85149'
            ax.axvline(event.timestamp, color=marker_color, alpha=0.5, linewidth=1)

        # Add threshold lines
        mean = df[feature].mean()
        std = df[feature].std()
        ax.axhline(mean + threshold_std * std, color='#3fb950', linestyle='--', alpha=0.4)
        ax.axhline(mean - threshold_std * std, color='#f85149', linestyle='--', alpha=0.4)

        panel_idx += 1

    # Format x-axis
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel('Time')
    plt.xticks(rotation=45)

    # Title
    title = 'Event Detection'
    if symbol:
        title = f'{symbol} - {title}'
    n_events = sum(len(e) for e in all_events.values())
    title += f' ({n_events} events detected)'
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def plot_event_study(
    df: pd.DataFrame,
    events: List[Event],
    feature: str = 'midprice',
    window_before: int = 50,
    window_after: int = 100,
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot average feature behavior around events (event study).

    Args:
        df: DataFrame with features
        events: List of events to study
        feature: Feature to analyze around events
        window_before: Samples before event
        window_after: Samples after event
        normalize: Normalize to event time value
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from .features import apply_style, COLORS
    apply_style()

    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found")

    if 'datetime' not in df.columns:
        ts_col = 'timestamp_ns' if 'timestamp_ns' in df.columns else 'timestamp'
        df['datetime'] = pd.to_datetime(df[ts_col], unit='ns')

    # Extract windows around each event
    windows = []
    for event in events:
        # Find index of event
        idx = df['datetime'].searchsorted(event.timestamp)

        if idx < window_before or idx + window_after >= len(df):
            continue

        window = df[feature].iloc[idx - window_before:idx + window_after + 1].values

        if normalize:
            event_val = window[window_before]
            if event_val != 0:
                window = (window / event_val - 1) * 100  # Percent change

        windows.append(window)

    if not windows:
        raise ValueError("No valid event windows found")

    windows = np.array(windows)
    x = np.arange(-window_before, window_after + 1)

    # Calculate statistics
    mean = np.nanmean(windows, axis=0)
    std = np.nanstd(windows, axis=0)
    median = np.nanmedian(windows, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Individual traces (faint)
    for w in windows:
        ax.plot(x, w, color='#8b949e', alpha=0.1, linewidth=0.5)

    # Mean with confidence band
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, color='#58a6ff')
    ax.plot(x, mean, color='#58a6ff', linewidth=2, label='Mean')
    ax.plot(x, median, color='#3fb950', linewidth=1.5, linestyle='--', label='Median')

    # Event line
    ax.axvline(0, color='#f85149', linewidth=2, linestyle='-', label='Event')
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('Samples from Event')
    ax.set_ylabel(f'{feature} (% change)' if normalize else feature)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.set_title(f'Event Study: {feature} ({len(windows)} events)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig
