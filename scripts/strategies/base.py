"""
Strategy base class — the interface all strategies must implement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class StrategyMeta:
    name: str
    description: str
    paper: str  # paper reference or "empirical"
    horizon: str  # "1min", "5min", "15min", "1h", "4h"
    required_columns: List[str]
    parameters: Dict


class Strategy:
    """Base class for all trading strategies."""

    meta: StrategyMeta

    def compute_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute strategy-specific features from aggregated bars."""
        raise NotImplementedError

    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate trading signal from features.
        Returns Series in [-1.0, +1.0]. NaN = no signal.
        """
        raise NotImplementedError

    def warmup_bars(self) -> int:
        """Number of initial bars needed before signal is valid."""
        raise NotImplementedError
