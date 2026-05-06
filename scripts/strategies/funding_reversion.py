"""
Strategy B3: Funding Rate Mean Reversion.

Thesis: When Hyperliquid funding rate is extreme (high z-score),
price tends to revert toward the direction that reduces the imbalance.

- Positive funding = longs pay shorts → too many longs → price likely falls
- Negative funding = shorts pay longs → too many shorts → price likely rises

The 8-hour settlement cycle on Hyperliquid creates a structural mean-reversion
dynamic: extreme funding attracts arbitrageurs who push price back.

Signal: short when funding_zscore > threshold, long when < -threshold.
Position size proportional to z-score magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, StrategyMeta


class FundingReversion(Strategy):
    """
    Mean-reversion strategy based on funding rate extremes.

    Parameters:
        zscore_entry: z-score threshold to enter (default: 2.0)
        zscore_exit: z-score threshold to exit (default: 0.5)
        max_position: maximum signal magnitude (default: 1.0)
        lookback: bars to compute rolling stats (default: 96 = 24h at 15min bars)
        use_raw_zscore: if True, use pre-computed ctx_funding_zscore;
                        if False, compute our own rolling z-score
    """

    def __init__(
        self,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        max_position: float = 1.0,
        lookback: int = 96,
        use_raw_zscore: bool = True,
    ):
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.max_position = max_position
        self.lookback = lookback
        self.use_raw_zscore = use_raw_zscore

        self.meta = StrategyMeta(
            name="funding_reversion",
            description="Short when funding is extremely positive, long when extremely negative",
            paper="empirical — Hyperliquid 8h funding settlement mechanic",
            horizon="4h",
            required_columns=["ctx_funding_rate", "ctx_funding_zscore"],
            parameters={
                "zscore_entry": zscore_entry,
                "zscore_exit": zscore_exit,
                "max_position": max_position,
                "lookback": lookback,
            },
        )

    def warmup_bars(self) -> int:
        return self.lookback if not self.use_raw_zscore else 1

    def compute_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute funding z-score and signal features."""
        result = pd.DataFrame(index=bars.index)

        # Get funding rate (use mean aggregation from bars)
        fr_col = None
        for col in ["ctx_funding_rate_mean", "ctx_funding_rate"]:
            if col in bars.columns:
                fr_col = col
                break

        if fr_col is None:
            result["fr_zscore"] = np.nan
            return result

        funding = bars[fr_col].values.astype(np.float64)

        if self.use_raw_zscore:
            # Use the pre-computed z-score from the ingestor
            # Prefer _last (point-in-time) over _mean (smoothed within bar)
            zs_col = None
            for col in ["ctx_funding_zscore_last", "ctx_funding_zscore_mean", "ctx_funding_zscore"]:
                if col in bars.columns:
                    zs_col = col
                    break
            if zs_col:
                result["fr_zscore"] = bars[zs_col].values
            else:
                # Fallback: compute our own
                result["fr_zscore"] = self._rolling_zscore(funding)
        else:
            result["fr_zscore"] = self._rolling_zscore(funding)

        # Absolute z-score for position sizing
        result["fr_zscore_abs"] = np.abs(result["fr_zscore"])

        # Funding rate direction
        result["fr_sign"] = np.sign(funding)

        return result

    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate signal based on funding z-score thresholds.

        Logic:
          - |zscore| > entry_threshold: enter counter-funding position
          - |zscore| < exit_threshold: flatten
          - Position size scales linearly with |zscore| between entry and 2*entry
        """
        zscore = features["fr_zscore"].values
        signal = np.full(len(zscore), 0.0)

        for i in range(len(zscore)):
            z = zscore[i]
            if np.isnan(z):
                signal[i] = np.nan
                continue

            abs_z = abs(z)

            if abs_z >= self.zscore_entry:
                # Position size: linear scale from 0 at entry to max at 2*entry
                intensity = min(1.0, (abs_z - self.zscore_entry) / self.zscore_entry)
                size = intensity * self.max_position

                # Counter-funding: short if funding positive, long if negative
                if z > 0:
                    signal[i] = -size  # funding positive → short
                else:
                    signal[i] = size  # funding negative → long
            elif abs_z < self.zscore_exit:
                signal[i] = 0.0
            else:
                # Between exit and entry: hold previous (use 0 for simplicity)
                signal[i] = 0.0

        return pd.Series(signal, index=features.index, name="signal")

    def _rolling_zscore(self, arr: np.ndarray) -> np.ndarray:
        """Compute rolling z-score over lookback window."""
        n = len(arr)
        zscore = np.full(n, np.nan)
        for i in range(self.lookback, n):
            window = arr[i - self.lookback: i]
            valid = window[~np.isnan(window)]
            if len(valid) < 10:
                continue
            mu = valid.mean()
            sigma = valid.std()
            if sigma > 1e-12:
                zscore[i] = (arr[i] - mu) / sigma
        return zscore
