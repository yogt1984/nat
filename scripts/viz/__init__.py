"""
ING Visualization Module

Tools for exploring and visualizing feature data from the ingestor.

Usage:
    from viz import load_data, plot_features, plot_events, plot_correlations

    df = load_data('./data/features')
    plot_features(df, symbol='BTC', features=['midprice', 'vpin_10', 'qty_l1'])
    plot_events(df, symbol='BTC')
    plot_correlations(df)
"""

from .loader import load_data, load_recent, get_symbols, summarize
from .features import plot_features, plot_feature_panel, plot_rolling_stats
from .events import plot_events, detect_events, plot_event_study
from .correlations import plot_correlation_matrix, plot_scatter_matrix, plot_pca
from .distributions import plot_distributions, plot_qq, plot_outliers

__all__ = [
    # Loader
    'load_data', 'load_recent', 'get_symbols', 'summarize',
    # Features
    'plot_features', 'plot_feature_panel', 'plot_rolling_stats',
    # Events
    'plot_events', 'detect_events', 'plot_event_study',
    # Correlations
    'plot_correlation_matrix', 'plot_scatter_matrix', 'plot_pca',
    # Distributions
    'plot_distributions', 'plot_qq', 'plot_outliers',
]
