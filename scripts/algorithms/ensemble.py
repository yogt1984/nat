"""
Cross-algorithm ensemble — combine signals from multiple algorithms.

Three combination methods:
  - equal_weight: mean of z-scored signal features
  - ic_weight: weight each algorithm by its trailing IC
  - regime_switch: select algorithm based on regime state

Usage:
    ensemble = Ensemble(["jump_detector", "optimal_entry", "funding_reversion"])
    combined = ensemble.combine(results_dict)

Config: [ensemble] section in config/algorithms.toml
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default signal features per algorithm — the primary directional signal.
# These are the features that get z-scored and combined.
DEFAULT_SIGNAL_FEATURES = {
    "jump_detector": "alg_post_jump_reversion",
    "optimal_entry": "alg_entry_signal",
    "funding_reversion": "alg_funding_signal",
    "surprise_signal": "alg_entropy_surprise",
    "weighted_ofi": "alg_weighted_ofi",
}


class Ensemble:
    """Combine signals from multiple algorithms into a unified output.

    Parameters
    ----------
    algorithms : list[str]
        Algorithm names to combine.
    method : str
        Combination method: "equal_weight", "ic_weight", or "regime_switch".
    signal_features : dict[str, str] | None
        Map algorithm name → signal column to use. Defaults to
        DEFAULT_SIGNAL_FEATURES for known algorithms.
    ic_lookback : int
        Number of ticks for trailing IC calculation (ic_weight method).
    regime_column : str
        Column name for regime state (regime_switch method).
    """

    def __init__(
        self,
        algorithms: list[str],
        method: str = "equal_weight",
        signal_features: Optional[dict[str, str]] = None,
        ic_lookback: int = 5000,
        regime_column: str = "ent_book_shape",
    ):
        self.algorithms = algorithms
        self.method = method
        self.signal_features = signal_features or {}
        self.ic_lookback = ic_lookback
        self.regime_column = regime_column

    def _resolve_signal_col(self, algo_name: str) -> str:
        """Get the signal column for an algorithm."""
        if algo_name in self.signal_features:
            return self.signal_features[algo_name]
        if algo_name in DEFAULT_SIGNAL_FEATURES:
            return DEFAULT_SIGNAL_FEATURES[algo_name]
        raise ValueError(
            f"No signal feature defined for '{algo_name}'. "
            f"Pass signal_features={{'{algo_name}': 'col_name'}}"
        )

    def combine(
        self,
        results: dict[str, pd.DataFrame],
        base_df: Optional[pd.DataFrame] = None,
        forward_returns: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Combine algorithm outputs into ensemble signals.

        Parameters
        ----------
        results : dict[str, pd.DataFrame]
            Map algorithm name → features_df from AlgorithmResult.
        base_df : pd.DataFrame | None
            Original data (needed for regime_switch method).
        forward_returns : np.ndarray | None
            Forward returns (needed for ic_weight method).

        Returns
        -------
        pd.DataFrame with columns:
            - ens_signal: combined signal
            - ens_weight_{name}: per-algorithm weight (for ic_weight/regime_switch)
        """
        if self.method == "equal_weight":
            return self._equal_weight(results)
        elif self.method == "ic_weight":
            if forward_returns is None:
                raise ValueError("ic_weight method requires forward_returns")
            return self._ic_weight(results, forward_returns)
        elif self.method == "regime_switch":
            if base_df is None:
                raise ValueError("regime_switch method requires base_df")
            if forward_returns is None:
                raise ValueError("regime_switch method requires forward_returns")
            return self._regime_switch(results, base_df, forward_returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _extract_signals(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract and z-score signal columns from each algorithm."""
        signals = {}
        for name in self.algorithms:
            if name not in results:
                logger.warning("Algorithm '%s' missing from results, skipping", name)
                continue
            col = self._resolve_signal_col(name)
            if col not in results[name].columns:
                logger.warning("Signal column '%s' not in %s output", col, name)
                continue
            raw = results[name][col].values.astype(np.float64)
            # Z-score: (x - mean) / std, with NaN propagation
            mean = np.nanmean(raw)
            std = np.nanstd(raw)
            if std > 1e-12:
                signals[name] = (raw - mean) / std
            else:
                signals[name] = np.zeros_like(raw)

        return pd.DataFrame(signals)

    def _equal_weight(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Mean of z-scored signals."""
        signals = self._extract_signals(results)

        if signals.empty:
            raise ValueError("No valid signals to combine")

        combined = signals.mean(axis=1)

        out = pd.DataFrame({"ens_signal": combined})
        for name in signals.columns:
            out[f"ens_weight_{name}"] = 1.0 / len(signals.columns)
        return out

    def _ic_weight(
        self,
        results: dict[str, pd.DataFrame],
        forward_returns: np.ndarray,
    ) -> pd.DataFrame:
        """Weight by trailing IC (rank correlation with forward returns)."""
        signals = self._extract_signals(results)

        if signals.empty:
            raise ValueError("No valid signals to combine")

        n = len(forward_returns)
        weights = {}

        for name in signals.columns:
            sig = signals[name].values
            # Trailing IC: rank correlation over lookback window
            ic_values = np.full(n, np.nan)
            for i in range(self.ic_lookback, n):
                window_sig = sig[i - self.ic_lookback:i]
                window_ret = forward_returns[i - self.ic_lookback:i]
                mask = np.isfinite(window_sig) & np.isfinite(window_ret)
                if mask.sum() > 30:
                    from scipy.stats import spearmanr
                    ic_values[i] = spearmanr(window_sig[mask], window_ret[mask])[0]
            weights[name] = np.abs(ic_values)  # abs IC as weight

        # Normalize weights
        weight_df = pd.DataFrame(weights)
        weight_sum = weight_df.sum(axis=1).replace(0, np.nan)
        for name in weight_df.columns:
            weight_df[name] = weight_df[name] / weight_sum

        # Weighted combination
        combined = np.zeros(n)
        for name in signals.columns:
            w = weight_df[name].fillna(0).values
            combined += w * signals[name].values

        out = pd.DataFrame({"ens_signal": combined})
        for name in weight_df.columns:
            out[f"ens_weight_{name}"] = weight_df[name]
        return out

    def _regime_switch(
        self,
        results: dict[str, pd.DataFrame],
        base_df: pd.DataFrame,
        forward_returns: np.ndarray,
    ) -> pd.DataFrame:
        """Select algorithm based on regime state.

        Low-entropy regime → jump_detector/optimal_entry (trending)
        High-entropy regime → funding_reversion/weighted_ofi (mean-reverting)
        """
        signals = self._extract_signals(results)

        if signals.empty:
            raise ValueError("No valid signals to combine")

        n = len(signals)

        if self.regime_column not in base_df.columns:
            logger.warning(
                "Regime column '%s' not found, falling back to equal_weight",
                self.regime_column,
            )
            return self._equal_weight(results)

        regime_vals = base_df[self.regime_column].values.astype(np.float64)
        median_regime = np.nanmedian(regime_vals)

        # Low entropy → trending signals; high entropy → mean-reverting signals
        trending_algos = [a for a in ["jump_detector", "optimal_entry", "surprise_signal"]
                          if a in signals.columns]
        reverting_algos = [a for a in ["funding_reversion", "weighted_ofi"]
                           if a in signals.columns]

        # Fall back to all algorithms if categories are empty
        if not trending_algos:
            trending_algos = list(signals.columns)
        if not reverting_algos:
            reverting_algos = list(signals.columns)

        low_mask = regime_vals < median_regime
        combined = np.zeros(n)
        weights = {name: np.zeros(n) for name in signals.columns}

        for i in range(n):
            if np.isnan(regime_vals[i]):
                # Unknown regime: equal weight
                active = list(signals.columns)
            elif low_mask[i]:
                active = trending_algos
            else:
                active = reverting_algos

            w = 1.0 / len(active)
            for name in active:
                val = signals[name].iloc[i]
                if np.isfinite(val):
                    combined[i] += w * val
                weights[name][i] = w

        out = pd.DataFrame({"ens_signal": combined})
        for name in signals.columns:
            out[f"ens_weight_{name}"] = weights[name]
        return out
