"""
NAT Backtesting Infrastructure

A minimal, skeptical backtester for validating trading strategies
on collected feature data.
"""

from .data_loader import load_features, FeatureDataset
from .strategy import Strategy, accumulation_long, distribution_short, entropy_breakout
from .costs import CostModel
from .engine import run_backtest, BacktestResult, Trade
from .walk_forward import walk_forward_validation, WalkForwardResult

__all__ = [
    "load_features",
    "FeatureDataset",
    "Strategy",
    "accumulation_long",
    "distribution_short",
    "entropy_breakout",
    "CostModel",
    "run_backtest",
    "BacktestResult",
    "Trade",
    "walk_forward_validation",
    "WalkForwardResult",
]
