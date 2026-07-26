"""
Lee-Mykland Jump Detection v2 — EVT threshold, staggered bipower,
directional routing, adaptive reversion horizon
=================================================================

Maturity: PRELIM (planted 17/17 + real-parquet smoke green; no OOS backtest
yet — A/B vs frozen v1 baseline scheduled on the next >=30-clean-day window,
task Q3). Do not deploy ahead of that verdict.

Upgrades over the frozen v1 baseline (``jump_detector.py``, untouched):

1. EVT (Gumbel) Detection Threshold
-----------------------------------
v1 declares a jump when L(t) = |r_t| / σ̂ exceeds a flat constant c = 3.0.
Under the continuous-record asymptotics of Lee & Mykland (2008), the correct
critical value for the *maximum* of n test statistics follows a Gumbel law:
rejecting when

  L(t) > C_n + S_n · β*,       β* = -ln(-ln(1 - α))

controls the family-wise false-jump rate at level α over a block of n
observations, where with c = √(2/π):

  C_n = √(2 ln n)/c - (ln π + ln ln n) / (2 c √(2 ln n))
  S_n = 1 / (c √(2 ln n))

At n = 864,000 (one day at 100 ms) and α = 0.01 the threshold is ≈ 7.2 —
materially stricter than 3.0, which at tick frequency fires thousands of
times per day on pure diffusion. ``threshold_mode="fixed"`` preserves the
legacy behaviour for A/B comparison.

2. Staggered (Skip-One) Bipower Variation
-----------------------------------------
The local volatility estimate at tick t is built from products of absolute
returns. The v1 (adjacent) estimator includes the term |r_t|·|r_{t-1}|:
when r_t itself is the jump under test, that product inflates σ̂ and
*masks* the jump (self-masking) — worst when two jumps arrive on
consecutive ticks, since |r_{t}|·|r_{t-1}| then contains both. The
staggered estimator pairs each return with the one two ticks back
(Huang & Tauchen 2005; Andersen, Dobrev & Schaumburg 2012):

  adjacent : σ̂²_BV(t) = (π/2) · mean_k |r_k|·|r_{k-1}|,  k ≤ t
  staggered: σ̂²_BV(t) = (π/2) · mean_k |r_k|·|r_{k-2}|,  k ≤ t

so the current return is never multiplied with its immediate predecessor,
removing the self-masking term (the π/2 = 1/μ₁² correction is unchanged
because non-adjacent returns are independent under H₀). Staggering also
reduces bias from bid-ask-bounce autocorrelation at 100 ms cadence.

Post-shock volatility floor: immediately after a large return, a
near-zero-volatility window can drive σ̂ → 0 and flag *any* subsequent
move as a jump. σ̂ is therefore floored at

  σ̂(t) ≥ vol_floor_frac · |r_{t-1}|

so a tick following a shock of size J is only flagged if it exceeds
vol_floor_frac⁻¹ standard-shock-units (default 10× guard, i.e. a genuine
second jump of comparable size still yields L = 1/vol_floor_frac = 10,
well above every threshold in use).

3. Directional Routing
----------------------
Post-jump reversion after an UP jump (buy-side liquidation of shorts,
breakout exhaustion) and a DOWN jump (long-liquidation cascades) are
different economic events in perpetual futures. The single reversion
signal is therefore also routed by the sign of the triggering jump into
``alg_jd2_rev_up`` / ``alg_jd2_rev_down`` so downstream consumers can
weight the two sides independently.

4. Magnitude-Adaptive Reversion Horizon
---------------------------------------
v1 tracks reversion for a fixed 50 ticks regardless of jump size. Larger
displacements take longer to mean-revert, so with
``adaptive_horizon=True`` the tracking window scales with the detection
statistic:

  H = min(horizon_max, round(reversion_horizon + horizon_per_l · L_det))

with floor ``reversion_horizon`` and cap ``horizon_max``.

Reversion definition (identical sign convention to v1): for the last
detected jump at tick t_J with log-return r_J and price p_J,

  REV(t) = -ln(p_t / p_J) / r_J,   0 < t - t_J ≤ H,   else 0.0

REV > 0 ⇔ price is moving back against the jump direction.

Outputs
-------
  alg_jd2_statistic : L(t) = |r_t| / σ̂(t)                      [0, ∞)
  alg_jd2_detected  : 1.0 if L(t) > active threshold            {0, 1}
  alg_jd2_magnitude : signed log-return at detected jump, else 0
  alg_jd2_reversion : REV(t) as above (primary signal)
  alg_jd2_rev_up    : REV(t) if last jump was upward, else 0
  alg_jd2_rev_down  : REV(t) if last jump was downward, else 0

Parameters (config/algorithms.toml → [jump_detector_v2])
--------------------------------------------------------
  window            (K)  : trailing ticks for the bipower pool (default 100)
  threshold_mode         : "evt" | "fixed"                (default "evt")
  significance      (c)  : fixed-mode threshold           (default 3.0)
  evt_alpha         (α)  : Gumbel tail probability        (default 0.01)
  evt_block         (n)  : observations per EVT family    (default 864000)
  staggered_bv           : skip-one bipower               (default True)
  vol_floor_frac         : post-shock σ̂ floor fraction    (default 0.1)
  adaptive_horizon       : magnitude-scaled horizon       (default True)
  reversion_horizon (H₀) : fixed horizon / adaptive floor (default 50)
  horizon_per_l          : adaptive ticks per unit L      (default 10.0)
  horizon_max            : adaptive cap                   (default 500)

References
----------
  Lee, S.S. & Mykland, P.A. (2008), Review of Financial Studies 21(6).
  Huang, X. & Tauchen, G. (2005), Journal of Financial Econometrics 3(4).
  Andersen, T., Dobrev, D. & Schaumburg, E. (2012), Journal of
    Econometrics 169(1) — jump-robust volatility estimation.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register


@register
class JumpDetectorV2(MicrostructureAlgorithm):
    """Lee-Mykland jump test v2: EVT threshold, staggered BV, directional
    reversion routing, magnitude-adaptive horizon."""

    def __init__(self, window: int = 100, threshold_mode: str = "evt",
                 significance: float = 3.0, evt_alpha: float = 0.01,
                 evt_block: int = 864000, staggered_bv: bool = True,
                 vol_floor_frac: float = 0.1, adaptive_horizon: bool = True,
                 reversion_horizon: int = 50, horizon_per_l: float = 10.0,
                 horizon_max: int = 500):
        if threshold_mode not in ("evt", "fixed"):
            raise ValueError(f"threshold_mode must be 'evt' or 'fixed', got {threshold_mode!r}")
        self._window = window
        self._threshold_mode = threshold_mode
        self._significance = significance
        self._evt_alpha = evt_alpha
        self._evt_block = evt_block
        self._staggered = staggered_bv
        self._vol_floor_frac = vol_floor_frac
        self._adaptive = adaptive_horizon
        self._reversion_horizon = reversion_horizon
        self._horizon_per_l = horizon_per_l
        self._horizon_max = horizon_max

        # Bipower product pools (append-only ring buffers, finite values only).
        # Adjacent pool holds |r_k||r_{k-1}| for k ≤ t (window-1 most recent);
        # staggered pool holds |r_k||r_{k-2}| for k ≤ t (window-2 most recent).
        self._pool_adj: deque[float] = deque(maxlen=window - 1)
        self._pool_stag: deque[float] = deque(maxlen=window - 2)
        self._prev_mid: float = np.nan
        self._prev_abs1: float = np.nan   # |r_{t-1}|
        self._prev_abs2: float = np.nan   # |r_{t-2}|
        # Last-jump tracking
        self._tick_count: int = 0
        self._last_jump_tick: int = -9999
        self._last_jump_mag: float = 0.0
        self._last_jump_price: float = np.nan
        self._last_jump_horizon: int = reversion_horizon

    # ------------------------------------------------------------------ #
    #  Contract surface                                                   #
    # ------------------------------------------------------------------ #

    def name(self) -> str:
        return "jump_detector_v2"

    def alg_features(self) -> list[AlgorithmFeature]:
        w = self._window
        return [
            AlgorithmFeature("alg_jd2_statistic", warmup=w,
                             description="Lee-Mykland L(t) = |return| / staggered-BV local vol"),
            AlgorithmFeature("alg_jd2_detected", warmup=w,
                             description="1.0 if L(t) > EVT (Gumbel) or fixed threshold"),
            AlgorithmFeature("alg_jd2_magnitude", warmup=w,
                             description="Signed return at detected jump (0 if no jump)"),
            AlgorithmFeature("alg_jd2_reversion", warmup=w,
                             description="Post-jump reversion within adaptive horizon (primary)"),
            AlgorithmFeature("alg_jd2_rev_up", warmup=w,
                             description="Reversion routed to upward jumps only"),
            AlgorithmFeature("alg_jd2_rev_down", warmup=w,
                             description="Reversion routed to downward jumps only"),
        ]

    def required_columns(self) -> list[str]:
        return ["raw_midprice"]

    # ------------------------------------------------------------------ #
    #  Thresholds                                                         #
    # ------------------------------------------------------------------ #

    def evt_threshold(self) -> float:
        """Lee-Mykland (2008) Gumbel-asymptotic critical value for a block
        of ``evt_block`` observations at tail probability ``evt_alpha``."""
        n = float(self._evt_block)
        alpha = self._evt_alpha
        c = np.sqrt(2.0 / np.pi)
        ln_n = np.log(n)
        root = np.sqrt(2.0 * ln_n)
        c_n = root / c - (np.log(np.pi) + np.log(ln_n)) / (2.0 * c * root)
        s_n = 1.0 / (c * root)
        beta_star = -np.log(-np.log(1.0 - alpha))
        return float(c_n + s_n * beta_star)

    def _active_threshold(self) -> float:
        if self._threshold_mode == "evt":
            return self.evt_threshold()
        return self._significance

    # Minimum product counts before σ̂ is considered estimable (mirrors the
    # run_batch min_periods so step()/run_batch() agree tick-for-tick).
    _MIN_PRODUCTS_ADJ = 19
    _MIN_PRODUCTS_STAG = 18

    # ------------------------------------------------------------------ #
    #  Streaming path                                                     #
    # ------------------------------------------------------------------ #

    def step(self, tick: dict[str, float]) -> dict[str, float]:
        mid = tick.get("raw_midprice", np.nan)
        if not np.isfinite(mid):
            return {f.name: np.nan for f in self.alg_features()}

        if not (np.isfinite(self._prev_mid) and self._prev_mid > 0):
            self._prev_mid = mid
            return {f.name: np.nan for f in self.alg_features()}

        ret = float(np.log(mid / self._prev_mid))
        self._prev_mid = mid
        self._tick_count += 1
        abs_ret = abs(ret)

        # Append this tick's products FIRST: both pools include k = t, so the
        # adjacent pool carries the self-masking |r_t||r_{t-1}| term and the
        # staggered pool replaces it with |r_t||r_{t-2}| (see module docstring).
        if np.isfinite(self._prev_abs1):
            self._pool_adj.append(abs_ret * self._prev_abs1)
        if np.isfinite(self._prev_abs2):
            self._pool_stag.append(abs_ret * self._prev_abs2)

        pool = self._pool_stag if self._staggered else self._pool_adj
        min_products = self._MIN_PRODUCTS_STAG if self._staggered else self._MIN_PRODUCTS_ADJ

        if len(pool) < min_products:
            # σ̂ not yet estimable — warmup region.
            self._prev_abs2 = self._prev_abs1
            self._prev_abs1 = abs_ret
            return {f.name: np.nan for f in self.alg_features()}

        bv = float(np.sqrt(np.pi / 2.0 * np.mean(pool)))
        floor = self._vol_floor_frac * self._prev_abs1 if np.isfinite(self._prev_abs1) else 0.0
        bv = max(bv, floor, 1e-20)

        self._prev_abs2 = self._prev_abs1
        self._prev_abs1 = abs_ret

        L = abs_ret / bv
        threshold = self._active_threshold()
        detected = 1.0 if L > threshold else 0.0
        magnitude = ret if detected else 0.0

        if detected:
            self._last_jump_tick = self._tick_count
            self._last_jump_mag = ret
            self._last_jump_price = mid
            self._last_jump_horizon = self._horizon_for(L)

        reversion = self._reversion(mid)
        rev_up = reversion if self._last_jump_mag > 0 else 0.0
        rev_down = reversion if self._last_jump_mag < 0 else 0.0

        return {
            "alg_jd2_statistic": L,
            "alg_jd2_detected": detected,
            "alg_jd2_magnitude": magnitude,
            "alg_jd2_reversion": reversion,
            "alg_jd2_rev_up": rev_up,
            "alg_jd2_rev_down": rev_down,
        }

    def _horizon_for(self, L: float) -> int:
        if not self._adaptive:
            return self._reversion_horizon
        return int(min(self._horizon_max,
                       round(self._reversion_horizon + self._horizon_per_l * L)))

    def _reversion(self, mid: float) -> float:
        ticks_since = self._tick_count - self._last_jump_tick
        if (0 < ticks_since <= self._last_jump_horizon
                and np.isfinite(self._last_jump_price)
                and abs(self._last_jump_mag) > 1e-20):
            return float(-np.log(mid / self._last_jump_price) / self._last_jump_mag)
        return 0.0

    def reset(self) -> None:
        self._pool_adj.clear()
        self._pool_stag.clear()
        self._prev_mid = np.nan
        self._prev_abs1 = np.nan
        self._prev_abs2 = np.nan
        self._tick_count = 0
        self._last_jump_tick = -9999
        self._last_jump_mag = 0.0
        self._last_jump_price = np.nan
        self._last_jump_horizon = self._reversion_horizon

    # ------------------------------------------------------------------ #
    #  Vectorized path (must agree with step() at rtol 1e-9)              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _windowed_nanmean(values: np.ndarray, window: int,
                          min_periods: int) -> np.ndarray:
        """Trailing-window mean over finite values via cumulative sums.

        Chosen over pandas' rolling mean because the sliding add/subtract
        accumulator drifts (~1e-6 relative on 1e-10-scale products), which
        breaks the step()/run_batch() parity contract; cumsum differencing
        keeps the two paths within ~1e-12 relative at test scale.
        """
        finite = np.isfinite(values)
        vals = np.where(finite, values, 0.0)
        csum = np.concatenate([[0.0], np.cumsum(vals)])
        ccnt = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
        n = len(values)
        idx = np.arange(n) + 1
        lo = np.maximum(idx - window, 0)
        sums = csum[idx] - csum[lo]
        counts = ccnt[idx] - ccnt[lo]
        out = np.full(n, np.nan)
        ok = counts >= min_periods
        out[ok] = sums[ok] / counts[ok]
        return out

    def run_batch(self, df: "pd.DataFrame") -> "pd.DataFrame":
        import pandas as pd

        mid = df["raw_midprice"].values.astype(np.float64)
        n = len(df)
        # log(mid[i]/mid[i-1]), not diff(log(mid)): must round identically
        # to step()'s np.log(mid / prev_mid) for the parity contract.
        log_ret = np.full(n, np.nan)
        if n > 1:
            log_ret[1:] = np.log(mid[1:] / mid[:-1])
        abs_ret = np.abs(log_ret)

        # Product streams (k = index of the later return in the pair).
        prod_adj = abs_ret * np.concatenate([[np.nan], abs_ret[:-1]])
        prod_stag = abs_ret * np.concatenate([[np.nan, np.nan], abs_ret[:-2]])

        if self._staggered:
            pool_mean = self._windowed_nanmean(
                prod_stag, self._window - 2, self._MIN_PRODUCTS_STAG)
        else:
            pool_mean = self._windowed_nanmean(
                prod_adj, self._window - 1, self._MIN_PRODUCTS_ADJ)

        bv = np.sqrt(np.pi / 2.0 * pool_mean)
        prev_abs = np.concatenate([[np.nan], abs_ret[:-1]])
        floor = np.where(np.isfinite(prev_abs),
                         self._vol_floor_frac * prev_abs, 0.0)
        bv = np.maximum(np.maximum(bv, floor), 1e-20)

        L = abs_ret / bv
        threshold = self._active_threshold()
        detected = (L > threshold).astype(np.float64)
        magnitude = np.where(detected > 0, log_ret, 0.0)

        reversion = np.zeros(n)
        rev_up = np.zeros(n)
        rev_down = np.zeros(n)
        last_jump_idx = -9999
        last_jump_mag = 0.0
        last_jump_price = np.nan
        last_horizon = self._reversion_horizon
        for i in range(n):
            if detected[i]:
                last_jump_idx = i
                last_jump_mag = log_ret[i]
                last_jump_price = mid[i]
                last_horizon = self._horizon_for(L[i])
            ticks_since = i - last_jump_idx
            if (0 < ticks_since <= last_horizon
                    and np.isfinite(last_jump_price)
                    and abs(last_jump_mag) > 1e-20):
                rev = -np.log(mid[i] / last_jump_price) / last_jump_mag
                reversion[i] = rev
                if last_jump_mag > 0:
                    rev_up[i] = rev
                else:
                    rev_down[i] = rev

        L[0] = np.nan
        detected[0] = np.nan

        result = pd.DataFrame({
            "alg_jd2_statistic": L,
            "alg_jd2_detected": detected,
            "alg_jd2_magnitude": magnitude,
            "alg_jd2_reversion": reversion,
            "alg_jd2_rev_up": rev_up,
            "alg_jd2_rev_down": rev_down,
        }, index=df.index)

        warmup = self.warmup
        if 0 < warmup < len(df):
            result.iloc[:warmup] = np.nan
        return result
