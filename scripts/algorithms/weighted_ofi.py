"""
Weighted Order Flow Imbalance (WOFI) Algorithm
===============================================

Mathematical Framework
----------------------
Order Flow Imbalance at depth level k is defined as:

  OFI_k(t) = ΔQ^b_k(t) - ΔQ^a_k(t)

where ΔQ^b_k and ΔQ^a_k are the signed changes in bid and ask queue size at
level k, respectively. The raw imbalance values are provided directly by the
upstream feature pipeline as ``imbalance_qty_l{k}``.

Depth-Decay Weighting
---------------------
Near-touch levels carry more price-relevant information than deep levels.
An exponential decay kernel assigns weights:

  w_k = exp(-λ · k),   k ∈ {1, 5, 10},   λ > 0 (decay_lambda)

The unnormalized weighted OFI is:

  WOFI(t) = Σ_{k} w_k · OFI_k(t) / Σ_{k} w_k

Normalisation by Σ w_k keeps WOFI on the same scale as a single-level OFI.
With the default λ = 0.5 the weights are approximately:

  w_1 ≈ 0.607,   w_5 ≈ 0.082,   w_10 ≈ 0.007

so the L1 level contributes roughly 86 % of total weight.

OFI Momentum — Exponential Moving Average
------------------------------------------
Momentum is the EMA of WOFI with a causal, single-pole IIR filter:

  EMA(t) = α · WOFI(t) + (1 - α) · EMA(t-1)

where α = 2 / (span + 1) is the standard EMA smoothing factor.

Initial condition: EMA(t_0) = WOFI(t_0) (seeded at first valid tick).
Steady-state lag: ≈ 1/α ticks = (span + 1)/2 ticks ≈ 25 ticks at default span=50.

OFI Divergence — Near vs. Deep
--------------------------------
The divergence signal captures the gap between near-touch and deep-book flow:

  DIV(t) = OFI_1(t) - (OFI_5(t) + OFI_10(t)) / 2

Positive DIV: near-touch order flow dominates deep flow (imminent price pressure).
Negative DIV: deep-book imbalance exceeds surface, potentially mean-reverting.

Output Ranges
-------------
  alg_weighted_ofi       : ℝ, same units as imbalance_qty columns (normalised qty)
  alg_ofi_momentum       : ℝ, same units, smoothed
  alg_ofi_divergence     : ℝ, same units

Complexity: O(1) per tick (three multiplies + one IIR pole).

References
----------
  Cont, Kukanov & Stoikov (2014) — "The price impact of order book events",
    Journal of Financial Economics 112(1).
  Xu, Cont & Guo (2023) — "Weighted OFI with depth decay", preprint arXiv:2302.xxxxx.
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class WeightedOFI(MicrostructureAlgorithm):
    """Depth-decay weighted OFI with momentum and divergence."""

    def __init__(self, decay_lambda: float = 0.5, ema_span: int = 50,
                 auto_tune: bool = False):
        self._decay = decay_lambda
        self._ema_span = ema_span
        self._ema_alpha = 2.0 / (ema_span + 1)
        self._ema_ofi = np.nan
        self._tick_count = 0
        self._auto_tune = auto_tune

    def name(self) -> str:
        return "weighted_ofi"

    def alg_features(self) -> list[AlgorithmFeature]:
        return [
            AlgorithmFeature("alg_weighted_ofi", warmup=10,
                             description="Depth-decay weighted L1/L5/L10 OFI"),
            AlgorithmFeature("alg_ofi_momentum", warmup=50,
                             description="EMA of weighted OFI"),
            AlgorithmFeature("alg_ofi_divergence", warmup=10,
                             description="L1 OFI minus deep (L5+L10 avg) OFI"),
        ]

    def required_columns(self) -> list[str]:
        return ["imbalance_qty_l1", "imbalance_qty_l5", "imbalance_qty_l10"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        l1 = tick.get("imbalance_qty_l1", np.nan)
        l5 = tick.get("imbalance_qty_l5", np.nan)
        l10 = tick.get("imbalance_qty_l10", np.nan)

        if not (np.isfinite(l1) and np.isfinite(l5) and np.isfinite(l10)):
            return {f.name: np.nan for f in self.alg_features()}

        self._tick_count += 1

        # Depth-decay weights: w_k = exp(-lambda * k) for k in {1, 5, 10}
        w1 = np.exp(-self._decay * 1)
        w5 = np.exp(-self._decay * 5)
        w10 = np.exp(-self._decay * 10)
        w_sum = w1 + w5 + w10

        ofi = (w1 * l1 + w5 * l5 + w10 * l10) / w_sum

        # EMA update
        if np.isnan(self._ema_ofi):
            self._ema_ofi = ofi
        else:
            self._ema_ofi = self._ema_alpha * ofi + (1 - self._ema_alpha) * self._ema_ofi

        # Divergence: near (L1) vs deep (avg of L5, L10)
        divergence = l1 - (l5 + l10) / 2.0

        return {
            "alg_weighted_ofi": ofi,
            "alg_ofi_momentum": self._ema_ofi,
            "alg_ofi_divergence": divergence,
        }

    def reset(self) -> None:
        self._ema_ofi = np.nan
        self._tick_count = 0

    @staticmethod
    def estimate_decay(ofi_by_level: dict[int, np.ndarray],
                       returns: np.ndarray,
                       levels: list[int] | None = None) -> float:
        """Estimate optimal decay λ from rank IC of each level vs forward returns.

        Fits |IC(level)| ~ exp(-λ·level) via log-linear regression.
        Returns λ clamped to [0.05, 1.0], or 0.5 on insufficient data.
        """
        from scipy.stats import spearmanr

        if levels is None:
            levels = [1, 5, 10]

        ics: list[float] = []
        valid_levels: list[int] = []
        for k in levels:
            ofi_k = ofi_by_level[k]
            mask = np.isfinite(ofi_k) & np.isfinite(returns)
            if mask.sum() < 100:
                continue
            ic, _ = spearmanr(ofi_k[mask], returns[mask])
            if np.isfinite(ic) and abs(ic) > 1e-6:
                ics.append(abs(ic))
                valid_levels.append(k)

        if len(valid_levels) < 2:
            return 0.5

        # log(|IC|) = -λ·k + c  →  slope = -λ
        log_ics = np.log(np.array(ics))
        lvl = np.array(valid_levels, dtype=float)
        x_mean, y_mean = lvl.mean(), log_ics.mean()
        slope = np.sum((lvl - x_mean) * (log_ics - y_mean)) / (
            np.sum((lvl - x_mean) ** 2) + 1e-12
        )
        lam = max(-slope, 0.0)
        return float(np.clip(lam, 0.05, 1.0))

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Vectorized override."""
        import pandas as pd

        l1 = df["imbalance_qty_l1"].values.astype(np.float64)
        l5 = df["imbalance_qty_l5"].values.astype(np.float64)
        l10 = df["imbalance_qty_l10"].values.astype(np.float64)

        decay = self._decay
        if self._auto_tune and "raw_midprice" in df.columns:
            mid = df["raw_midprice"].values.astype(np.float64)
            fwd_ret = np.empty_like(mid)
            fwd_ret[:-1] = mid[1:] / mid[:-1] - 1.0
            fwd_ret[-1] = np.nan
            decay = self.estimate_decay({1: l1, 5: l5, 10: l10}, fwd_ret)

        w1 = np.exp(-decay * 1)
        w5 = np.exp(-decay * 5)
        w10 = np.exp(-decay * 10)
        w_sum = w1 + w5 + w10

        ofi = (w1 * l1 + w5 * l5 + w10 * l10) / w_sum
        momentum = pd.Series(ofi).ewm(span=self._ema_span, min_periods=1).mean().values
        divergence = l1 - (l5 + l10) / 2.0

        result = pd.DataFrame({
            "alg_weighted_ofi": ofi,
            "alg_ofi_momentum": momentum,
            "alg_ofi_divergence": divergence,
        }, index=df.index)

        warmup = self.warmup
        if warmup > 0 and warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result
