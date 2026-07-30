"""HF1: microprice — the Stoikov fair-value anchor as a registered algorithm.

The micro-price (Stoikov 2018, "The Micro-Price: A High Frequency Estimator of
Future Prices") is the expected future mid conditional on order-book state. Its
deviation from mid, in bps, is the *calibrated* expected short-horizon mid move —
exactly the ``center`` adjustment a maker quote engine prices against (the GAP-04
``microprice_maker_sim`` primitive, and the fair-value input to A4/HF5). This
directly attacks the platform's binding constraint: the fill-conditional IC
collapse — a maker centered on the micro-price leans quotes away from adverse
flow instead of paying taker fees.

Two modes, per tick:

    fitted   : if ``mp_micro_adj_bps`` is present (the Markov-chain g* from
               scripts/features/microprice.py), use it verbatim — the calibrated
               expected mid move for the current (imbalance, spread) state.
    analytic : otherwise the closed-form weighted-mid deviation
               (I − 0.5) · spread_bps — the naive micro-price (equivalent to the
               live ``raw_microprice`` column), Stoikov's g* first-order term.
               Conservative, always available.

Outputs:

    alg_mp_dev_bps   the deviation itself (expected mid move, bps) — maker center
    alg_mp_dev_ema   EMA-smoothed deviation — a denoised quoting center
    alg_mp_signal    EMA z-score of the deviation — the IC-comparable signal

EMA mean/variance recurrences are used (not windowed stats) so ``run_batch`` can
replay ``step()``'s arithmetic *exactly* — backtest/live parity at rtol 1e-9 is a
birth requirement here (the BUG-5 lesson). NaN inputs freeze all state.

Parameters via ``config/algorithms.toml [microprice]``.
References: Stoikov (2018) SSRN 2970694; Gatheral & Oomen (2010).
"""

from __future__ import annotations

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register

_INPUTS = ("raw_midprice", "imbalance_qty_l1", "raw_spread_bps")


@register
class Microprice(MicrostructureAlgorithm):
    """Micro-price fair-value deviation (bps), EMA center and EMA z-signal."""

    def __init__(self, ema_alpha: float = 0.02, dev_ema_alpha: float = 0.1,
                 warmup: int = 50):
        self._ema_alpha = float(ema_alpha)          # z-score mean/var recurrence
        self._dev_ema_alpha = float(dev_ema_alpha)  # quoting-center smoothing
        self._warmup = int(warmup)
        self.reset()

    def name(self) -> str:
        return "microprice"

    def alg_features(self) -> list[AlgorithmFeature]:
        w = self._warmup
        return [
            AlgorithmFeature("alg_mp_dev_bps", warmup=w,
                             description="micro-price − mid in bps (expected mid move; "
                                         "fitted g* if provided, else (I−0.5)·spread)"),
            AlgorithmFeature("alg_mp_dev_ema", warmup=w,
                             description="EMA-smoothed deviation — denoised maker center"),
            AlgorithmFeature("alg_mp_signal", warmup=w,
                             description="EMA z-score of the deviation"),
        ]

    def required_columns(self) -> list[str]:
        return list(_INPUTS)

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._var = 0.0
        self._dev_ema = 0.0

    # ── core recurrence (shared verbatim by step and run_batch) ──────────────
    def _update(self, dev: float) -> tuple[float, float]:
        """Advance EMA state with one deviation; returns (dev_ema, z)."""
        a, da = self._ema_alpha, self._dev_ema_alpha
        if self._n == 0:
            self._mean = dev
            self._var = 0.0
            self._dev_ema = dev
        else:
            innov = dev - self._mean
            self._mean = self._mean + a * innov
            self._var = (1.0 - a) * (self._var + a * innov * innov)
            self._dev_ema = self._dev_ema + da * (dev - self._dev_ema)
        self._n += 1
        sd = np.sqrt(self._var)
        z = (dev - self._mean) / sd if sd > 1e-12 else 0.0
        return self._dev_ema, z

    @staticmethod
    def _deviation(tick: dict[str, float]) -> float:
        mp = tick.get("mp_micro_adj_bps")
        if mp is not None and np.isfinite(mp):
            return float(mp)                         # fitted Stoikov g*
        return (tick["imbalance_qty_l1"] - 0.5) * tick["raw_spread_bps"]

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        vals = [tick.get(c, np.nan) for c in _INPUTS]
        if not all(np.isfinite(v) for v in vals):
            return {f.name: np.nan for f in self.alg_features()}
        dev = self._deviation(tick)
        dev_ema, z = self._update(dev)
        return {
            "alg_mp_dev_bps": dev,
            "alg_mp_dev_ema": dev_ema,
            "alg_mp_signal": z,
        }

    def run_batch(self, df) -> "pd.DataFrame":  # noqa: F821
        """Vectorized deviation + exact sequential replay of the EMA recurrence.

        The recurrence loop runs only over VALID ticks (NaN rows freeze state,
        as in step) using the identical arithmetic — exact parity by construction.
        """
        import pandas as pd

        n = len(df)
        imb = df["imbalance_qty_l1"].to_numpy(dtype=np.float64, na_value=np.nan)
        spread = df["raw_spread_bps"].to_numpy(dtype=np.float64, na_value=np.nan)
        mid = df["raw_midprice"].to_numpy(dtype=np.float64, na_value=np.nan)

        dev = (imb - 0.5) * spread
        if "mp_micro_adj_bps" in df.columns:
            fitted = df["mp_micro_adj_bps"].to_numpy(dtype=np.float64, na_value=np.nan)
            dev = np.where(np.isfinite(fitted), fitted, dev)
        valid = np.isfinite(imb) & np.isfinite(spread) & np.isfinite(mid)
        dev = np.where(valid, dev, np.nan)

        dev_ema = np.full(n, np.nan)
        z = np.full(n, np.nan)
        self.reset()
        for i in np.flatnonzero(valid):
            dev_ema[i], z[i] = self._update(dev[i])

        out = pd.DataFrame({
            "alg_mp_dev_bps": dev,
            "alg_mp_dev_ema": dev_ema,
            "alg_mp_signal": z,
        }, index=df.index)
        return out
