#!/usr/bin/env python3
"""
Scalping Signal Profiler — Feature-level alpha analysis for scalping strategies.

Mathematical Framework
======================
This module implements a complete feature-evaluation pipeline for short-horizon
(scalping) alpha signals.  Every feature column is scored against forward log
returns on bars of a configurable timeframe.  The pipeline produces ten
statistics per feature and combines them into a single composite viability score.

Notation used throughout this file
-----------------------------------
  x[t]      : feature value at bar index t  (real-valued, mean-centred where noted)
  r_h[t]    : forward log return at horizon h  (see compute_forward_returns)
  p[t]      : price (mid or micro) at bar index t
  N         : total number of bars in the sample
  n         : number of valid (non-NaN) observations used in a given computation
  rank(·)   : ordinal rank mapping, 1-indexed, ties averaged  (scipy default)
  E[·]      : sample expectation (arithmetic mean over valid observations)
  Var(·)    : sample variance  (denominator n, not n-1, unless stated otherwise)
  bps       : basis points  (1 bps = 1e-4 in return units)
  h         : forward horizon in bars  (element of self.horizons, e.g. {1,2,5,10})
  W         : rolling window size for IC IR computation  (self.rolling_ic_window)

Pipeline stages
---------------
  1. compute_forward_returns  →  r_h[t] for each h in horizons
  2. profile_feature          →  all statistics for one feature column
     a. IC(x, r_h)            Spearman rank correlation at each horizon
     b. IC IR                 rolling IC mean / std (stability)
     c. Hit rate              P(sign(x[t]) = sign(r_1[t]))
     d. Quintile spread       E[r | Q5] - E[r | Q1] in bps
     e. Autocorrelation       lag-1 Pearson r of mean-centred x
     f. Cost-adjusted edge    gross_edge - cost * turnover_factor
     g. Conditional IC        IC within regime-conditioned subsets
     h. Classification        rule-based role assignment
     i. Composite score       weighted sum into [0, 1]
  3. profile_all              →  iterate over all feature columns, rank by score
  4. forward_test (optional)  →  validation:
       - default:        walk-forward k-fold expanding window
                         (forward_test_walkforward, see method docstring)
       - --legacy-split: single 70/30 temporal split (forward_test)

Profiles every feature signal for scalping viability by computing:
  - Information Coefficient (IC) at multiple forward horizons
  - IC Information Ratio (stability across rolling windows)
  - Directional hit rate
  - Quintile return spreads (Q5 - Q1)
  - Signal autocorrelation (turnover cost proxy)
  - Regime-conditional effectiveness (how IC shifts under VPIN/entropy/spread)
  - Cost-adjusted edge estimation
  - Feature classification: directional / gate / regime / noise

Forward test: temporal split, profile on in-sample, validate on out-of-sample.

Usage:
    # Profile all features for BTC at 5min bars
    python scripts/scalping_profiler.py profile --symbol BTC --timeframe 5min

    # Include forward-test validation (walk-forward k-fold, default 5 folds)
    python scripts/scalping_profiler.py profile --symbol BTC --forward-test

    # Use the legacy 70/30 single split (not recommended at small N)
    python scripts/scalping_profiler.py profile --symbol BTC --forward-test --legacy-split

    # Show top-N features only
    python scripts/scalping_profiler.py profile --symbol BTC --top 20

    # Check process status
    python scripts/scalping_profiler.py status
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


from cluster_pipeline.config import FEATURE_VECTORS, META_COLUMNS
from cluster_pipeline.loader import load_parquet, filter_symbol
from cluster_pipeline.preprocess import aggregate_bars

log = logging.getLogger("scalping_profiler")


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation between arrays a and b, returning 0.0 for
    degenerate inputs.

    Mathematical definition
    -----------------------
    Given n paired observations (a_i, b_i), let R_i = rank(a_i) and S_i = rank(b_i)
    with ties averaged.  The Spearman correlation is:

        rho = Pearson_r(R, S)
            = [ sum_{i=1}^{n} (R_i - R_bar)(S_i - S_bar) ]
              / sqrt[ sum_i (R_i - R_bar)^2 * sum_i (S_i - S_bar)^2 ]

    When there are no ties this simplifies to the classic d-squared formula:

        rho = 1 - 6 * sum_{i=1}^{n} d_i^2 / (n * (n^2 - 1))

    where  d_i = R_i - S_i.

    Degeneracy conditions that return 0.0
    --------------------------------------
      - n < 10          : too few points for a stable estimate
      - ptp(a) == 0     : constant array → all ranks identical → rho undefined
      - ptp(b) == 0     : same reasoning for b
      - corr is NaN     : numerical failure in scipy (e.g. all identical ranks)

    Parameters
    ----------
    a, b : np.ndarray, shape (n,)
        Paired observations.  Must not contain NaN (caller is responsible for
        masking before calling this function).

    Returns
    -------
    float in [-1, 1], or 0.0 for degenerate inputs.
    """
    if len(a) < 10 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = sp_stats.spearmanr(a, b)
    return float(corr) if np.isfinite(corr) else 0.0

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "timeframe": "5min",
    # h in {1, 2, 5, 10} bars — defines the set of forward horizons evaluated
    "horizons": [1, 2, 5, 10],
    # c_rt in basis points: assumed round-trip cost per trade (entry + exit)
    "cost_bps": 3.5,
    # IC threshold below which a feature is not considered directionally useful
    "min_ic": 0.02,
    # Minimum hit rate (> 0.50 is directionally informative)
    "min_hit_rate": 0.51,
    # Fraction of data used as in-sample in legacy single-split forward_test
    "forward_test_split": 0.70,
    # Walk-forward k-fold settings (replaces single-split as default validation)
    # Number of expanding-window folds
    "n_folds": 5,
    # Minimum number of bars in the first fold's training window
    "min_train_bars": 200,
    "top_n": 30,
    # W: rolling window size (bars) for IC IR computation
    "rolling_ic_window": 50,
    # Minimum non-NaN observations required to profile a feature
    "min_observations": 100,
    "symbols": ["BTC"],
}


def load_profiler_config(path: str = "config/pipeline.toml") -> Dict[str, Any]:
    """Load [profiler] section from pipeline config, with defaults."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    config = dict(_DEFAULT_CONFIG)
    p = Path(path)
    if p.exists():
        with open(p, "rb") as f:
            raw = tomllib.load(f)
        if "profiler" in raw:
            config.update(raw["profiler"])
    return config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ConditionalIC:
    """IC of a feature under a specific regime condition.

    Fields
    ------
    condition : str
        Label identifying the regime subset, e.g. "vpin_high" or "entropy_low".
        Constructed as  "{regime_variable}_{high|low}"  where the split is at
        the unconditional median of the conditioning variable.
    ic : float
        Spearman rank correlation of the feature with 1-bar forward returns,
        restricted to observations in this regime.  In [-1, 1].
    n_obs : int
        Number of valid (non-NaN, in-regime) observations used.
    """
    condition: str        # e.g. "vpin_high", "entropy_low"
    ic: float
    n_obs: int


@dataclass
class FeatureProfile:
    """Complete profile of a single feature's scalping viability.

    Every numeric field has a precise mathematical meaning; see the docstrings on
    the methods that compute each field for closed-form definitions.
    """
    name: str
    vector: str

    # IC(x, r_h) at each horizon h  →  Dict[h, rho_h]
    ic: Dict[int, float]
    # h* = argmax_h |IC(x, r_h)|
    ic_best_horizon: int
    # IC* = IC(x, r_{h*})  in [-1, 1]
    ic_best: float
    # IC IR = E[IC_window] / std[IC_window]  (dimensionless, sign-preserving)
    ic_ir: float

    # hit_rate = #{t : sign(x[t]) = sign(r_1[t])} / n  in [0, 1]
    hit_rate: float

    # Q5-Q1 spread  =  (E[r_1 | Q5] - E[r_1 | Q1]) * 1e4  in bps
    quintile_spread_bps: float
    # True if quintile means are monotonically non-decreasing or non-increasing
    quintile_monotonic: bool

    # rho_1 = Pearson autocorrelation of (x - x_bar) at lag 1  in (-1, 1)
    autocorr_1: float
    # #{t : x[t] is NaN} / N  in [0, 1]
    nan_rate: float

    # gross_edge = |quintile_spread_bps| / 2  in bps
    gross_edge_bps: float
    # net_edge = gross_edge - c_rt * turnover_factor  in bps
    net_edge_bps: float

    # IC conditioned on regime subsets (see _conditional_ic)
    conditional: List[ConditionalIC]

    # Rule-based role: "directional" | "gate" | "regime" | "noise"
    role: str
    # Composite viability score  in [0, 1]
    scalp_score: float


@dataclass
class ProfileReport:
    """Full profiling results."""
    symbol: str
    timeframe: str
    n_bars: int
    n_features: int
    timestamp: str
    profiles: List[FeatureProfile]
    config: Dict[str, Any]

    # Summary counts by role
    n_directional: int = 0
    n_gate: int = 0
    n_regime: int = 0
    n_noise: int = 0


@dataclass
class ForwardTestResult:
    """Validation of profiled features on out-of-sample data.

    Fields
    ------
    ic_correlation : float
        Spearman rank correlation of best-horizon IC values across features,
        comparing in-sample to out-of-sample rankings.  Measures whether the
        ordinal ranking of features by predictive power is preserved OOS.
        Range [-1, 1]; values > 0.3 are considered acceptable stability.
    stable_count : int
        Number of features for which |IC_OOS| >= min_ic.
    degraded_count : int
        Number of features where |IS IC| > min_ic and
        |ic_change_pct| > 50%, indicating substantial OOS degradation.
    """
    symbol: str
    timeframe: str
    in_sample_bars: int
    out_of_sample_bars: int
    split_ratio: float

    # Per-feature: in-sample vs out-of-sample comparison
    comparisons: List[Dict[str, Any]]

    # Spearman r of IS IC vs OOS IC vectors (see above)
    ic_correlation: float
    # Features with |OOS IC| >= min_ic
    stable_count: int
    # Features with |IS IC| > min_ic and |ic_change_pct| > 50%
    degraded_count: int


@dataclass
class WalkForwardFeature:
    """Per-feature aggregate from walk-forward k-fold validation.

    Fields
    ------
    name : str
        Feature column name.
    vector : str
        Feature vector group (entropy, trend, volatility, ...).
    horizon : int
        Forward-return horizon (in bars) at which this feature was evaluated.
        Selected as argmax_h |E_i[IS_IC(f, h, i)]|, i.e. the horizon whose mean
        in-sample IC across folds has the largest magnitude.  Selection uses
        only IS data so it does not leak OOS information.
    n_folds : int
        Number of expanding-window folds k.
    is_ic_per_fold, oos_ic_per_fold : List[float], length k
        IS_IC[i] = Spearman( x[t in IS_i],  r_h[t in IS_i] )
        OOS_IC[i] = Spearman( x[t in OOS_i], r_h[t in OOS_i] )
        IS_i, OOS_i are defined in forward_test_walkforward.
    is_ic_mean, oos_ic_mean : float
        Arithmetic means over the k folds.
    oos_ic_std : float
        Population standard deviation (ddof=0) of OOS_IC across folds.  Lower
        is better — measures fold-to-fold OOS instability.
    sign_consistency : float in [0, 1]
        Fraction of folds where sign(IS_IC[i]) == sign(OOS_IC[i]), restricted
        to folds with |IS_IC[i]| > 1e-6 to avoid degenerate sign comparisons.
        High consistency means the feature's directional read survives OOS.
    decision : str
        keep    : sign_consistency >= 0.6  AND  |oos_ic_mean| >= min_ic
        monitor : sign_consistency >= 0.6  AND  |oos_ic_mean| <  min_ic
        drop    : sign_consistency <  0.6
    """
    name: str
    vector: str
    horizon: int
    n_folds: int
    is_ic_per_fold: List[float]
    oos_ic_per_fold: List[float]
    is_ic_mean: float
    oos_ic_mean: float
    oos_ic_std: float
    sign_consistency: float
    decision: str


@dataclass
class WalkForwardResult:
    """Aggregate result of walk-forward k-fold validation across all features.

    Fold layout (expanding window)
    ------------------------------
    Let N = total bars, m = min_train_bars, k = n_folds, L = floor((N-m)/k).
    For i = 0, ..., k-1:
        IS_i  = bars[0 : m + i*L]
        OOS_i = bars[m + i*L : m + (i+1)*L]      for i < k-1
        OOS_{k-1} = bars[m + (k-1)*L : N]        (last fold absorbs remainder)

    Leakage guard
    -------------
    For horizon h, IC pairs (x[t], r_h[t]) require r_h[t] = log(p[t+h]/p[t]),
    which uses bar t+h.  To prevent peeking past a fold boundary at index B:
        IS pairs:  t in [0, B - h)
        OOS pairs: t in [B, t_end - h)
    """
    symbol: str
    timeframe: str
    n_folds: int
    total_bars: int
    min_train_bars: int
    fold_len: int
    horizons: List[int]
    features: List[WalkForwardFeature]
    keep_count: int
    monitor_count: int
    drop_count: int


# ---------------------------------------------------------------------------
# Core profiler
# ---------------------------------------------------------------------------


class ScalpingProfiler:
    """Profiles feature signals for scalping viability.

    Parameters
    ----------
    config : dict
        Configuration dictionary.  Relevant keys:

        timeframe : str
            Bar aggregation period (e.g. "5min").
        horizons : list[int]
            Forward return horizons H = {h_1, h_2, ...} in bars.
        cost_bps : float
            c_rt/2, the one-way transaction cost in basis points.
            Round-trip cost = 2 * cost_bps.
        min_ic : float
            Minimum |IC| threshold for directional utility.
        min_hit_rate : float
            Minimum hit-rate threshold for directional classification.
        rolling_ic_window : int
            W, the window size in bars for rolling IC IR computation.
        min_observations : int
            Minimum valid observations n_min required to profile a feature.
    """

    def __init__(self, config: Dict[str, Any]):
        self.timeframe = config["timeframe"]
        self.horizons = config["horizons"]
        self.cost_bps = config["cost_bps"]
        self.min_ic = config["min_ic"]
        self.min_hit_rate = config["min_hit_rate"]
        self.rolling_ic_window = config["rolling_ic_window"]
        self.min_obs = config["min_observations"]
        self.top_n = config["top_n"]
        self.config = config

    # ── Data loading ──────────────────────────────────────────────────

    def load_and_aggregate(
        self, data_dir: str, symbol: str
    ) -> pd.DataFrame:
        """Load parquet data, filter symbol, aggregate to bars."""
        log.info("Loading data from %s for %s", data_dir, symbol)
        df = load_parquet(data_dir)
        df = filter_symbol(df, symbol)
        log.info("Raw ticks: %d", len(df))

        log.info("Aggregating to %s bars", self.timeframe)
        bars = aggregate_bars(df, self.timeframe)
        log.info("Bars: %d", len(bars))
        return bars

    # ── Forward returns ───────────────────────────────────────────────

    def compute_forward_returns(
        self, bars: pd.DataFrame
    ) -> Dict[int, np.ndarray]:
        """Compute forward log returns at each horizon h in self.horizons.

        Mathematical definition
        -----------------------
        Let p[t] be the price at bar index t (0-indexed).  The h-step-ahead
        forward log return is:

            r_h[t] = log( p[t+h] / p[t] )
                   = log(p[t+h]) - log(p[t])        for t = 0, 1, ..., N-h-1

        Boundary handling
        -----------------
        The last h elements cannot have a valid forward return because
        p[t+h] does not exist.  These are set to NaN:

            r_h[t] = NaN    for t = N-h, ..., N-1

        The result is therefore a length-N array with exactly h trailing NaNs.

        Price column selection
        ----------------------
        _find_price_col() searches for mid-price or micro-price columns in a
        priority order (close > last > mean, mid before micro).  The selected
        column defines p[t].

        Parameters
        ----------
        bars : pd.DataFrame
            Aggregated OHLC-style bars, index 0..N-1.

        Returns
        -------
        Dict[int, np.ndarray]
            Mapping  h -> r_h  where r_h has dtype float64 and length N.
            Values are in natural units (not bps); multiply by 1e4 for bps.

        Computational complexity
        ------------------------
        O(N * |H|) time and space where |H| = len(self.horizons).

        Numerical note
        --------------
        Using log(p[t+h] / p[t]) rather than log(p[t+h]) - log(p[t]) avoids
        catastrophic cancellation when prices are nearly equal.  The ratio form
        is numerically equivalent but the subtraction form can lose precision
        for large prices with small relative changes.

        Gap-aware NaN-out
        -----------------
        Bars are indexed 0..N-1 contiguously even when the underlying tick
        stream had time gaps (e.g. OS-suspend or WS reconnect windows).  In
        that case `prices[t+h]` and `prices[t]` may straddle a wall-clock gap
        much larger than h * timeframe, and the resulting log-return is
        spurious.  Whenever bar timestamps are available we therefore set
        r_h[t] = NaN if  bar_start[t+h] - bar_start[t] > 2 * h * timeframe.
        The factor of 2 tolerates normal jitter (late bars, brief
        reconnects) without flagging them; only true gaps trip it.  IC
        computation already masks NaN, so this propagates correctly.
        """
        price_col = self._find_price_col(bars)
        prices = bars[price_col].values.astype(np.float64)

        # Bar-start timestamps for gap detection.  Falls back to "no gap
        # check" if no timestamp column is present (older test fixtures).
        ts_arr = None
        for ts_col in ("bar_start", "bar_end", "timestamp_ns"):
            if ts_col in bars.columns:
                ts_arr = pd.to_datetime(bars[ts_col], utc=True, errors="coerce")
                ts_arr = ts_arr.values.astype("datetime64[ns]")
                break
        try:
            tf_seconds = pd.Timedelta(self.timeframe).total_seconds()
        except Exception:
            tf_seconds = None

        fwd = {}
        for h in self.horizons:
            ret = np.full(len(prices), np.nan)
            if h < len(prices):
                # r_h[t] = log(p[t+h] / p[t]),  t in [0, N-h)
                ret[:-h] = np.log(prices[h:] / prices[:-h])

                # Gap-aware NaN-out: drop pairs whose wall-clock spacing
                # exceeds 2 * h * timeframe (handles OS-suspend windows and
                # long WS reconnects).
                if ts_arr is not None and tf_seconds is not None:
                    dt_ns = (ts_arr[h:] - ts_arr[:-h]).astype("timedelta64[ns]")
                    dt_s = dt_ns.astype(np.int64) / 1e9
                    expected = h * tf_seconds
                    spans_gap = dt_s > 2.0 * expected
                    ret[:-h][spans_gap] = np.nan
            fwd[h] = ret
        return fwd

    # ── Single feature profiling ──────────────────────────────────────

    def profile_feature(
        self,
        values: np.ndarray,
        fwd_returns: Dict[int, np.ndarray],
        name: str,
        vector: str,
        bars: pd.DataFrame,
    ) -> Optional[FeatureProfile]:
        """Compute all statistics for a single feature column.

        The method gates on minimum observation count, then sequentially
        computes IC, IC IR, hit rate, quintile analysis, autocorrelation,
        cost-adjusted edge, conditional IC, classification, and composite score.

        Parameters
        ----------
        values : np.ndarray, shape (N,)
            Feature values for all bars.  May contain NaN.
        fwd_returns : Dict[int, np.ndarray]
            Output of compute_forward_returns.  Each value is shape (N,) with
            trailing NaNs.
        name : str
            Column name, used as the profile identifier.
        vector : str
            Feature vector group label (from _detect_vector).
        bars : pd.DataFrame
            Full bar DataFrame, needed for conditional IC regime columns.

        Returns
        -------
        FeatureProfile or None
            None if the number of valid observations is below self.min_obs.

        IC at each horizon
        ------------------
        For horizon h, the valid mask is:

            mask_h = { t : x[t] is not NaN AND r_h[t] is not NaN }

        The IC is:

            IC_h = Spearman_rho( x[mask_h], r_h[mask_h] )

        The best horizon is:

            h* = argmax_{h in H}  |IC_h|

        IC IR
        -----
        Computed by _rolling_ic_ir using x and r_{h*}  (see that method).

        Hit rate
        --------
        Uses the shortest horizon h_1 = self.horizons[0] (1-bar lookahead):

            hit_mask = { t : x[t] is not NaN AND r_{h1}[t] is not NaN }

            hit_rate = #{t in hit_mask : sign(x[t]) = sign(r_{h1}[t])} / |hit_mask|

        Note: sign(0) = 0, so ties contribute 0 to the numerator.

        Gross edge and net edge
        -----------------------
        See _quintile_analysis for the spread definition.

            gross_edge = |quintile_spread_bps| / 2       [bps]

        This halves the Q5-Q1 spread as a per-trade estimate: a long-only
        strategy on the top quintile earns half the full spread on average.

            turnover_factor = max(0.1, 1 - autocorr_1)   if autocorr_1 > 0
                            = 1.0                         if autocorr_1 <= 0

        High positive autocorrelation means the signal rarely changes sign, so
        the strategy trades less frequently, reducing effective cost burden.

            round_trip_cost = 2 * cost_bps               [bps]

            net_edge = gross_edge - round_trip_cost * turnover_factor   [bps]

        A negative net_edge means the signal cannot overcome transaction costs.
        """
        valid = ~np.isnan(values)
        nan_rate = 1.0 - valid.mean()
        if valid.sum() < self.min_obs:
            return None

        # ── IC at each horizon ──
        # IC_h = Spearman_rho(x[mask_h], r_h[mask_h])
        ic_map = {}
        for h, fwd in fwd_returns.items():
            mask = valid & ~np.isnan(fwd)
            if mask.sum() < self.min_obs:
                ic_map[h] = 0.0
                continue
            ic_map[h] = _safe_spearman(values[mask], fwd[mask])

        # h* = argmax_h |IC_h|
        best_h = max(ic_map, key=lambda k: abs(ic_map[k]))
        best_ic = ic_map[best_h]

        # ── Rolling IC stability (IC IR) ──
        # IC IR = E[{IC_w}] / std[{IC_w}] over overlapping windows of size W
        ic_ir = self._rolling_ic_ir(values, fwd_returns.get(best_h, fwd_returns[self.horizons[0]]))

        # ── Hit rate (1-bar horizon) ──
        # hit_rate = #{sign(x[t]) == sign(r_1[t])} / n  over valid 1-bar pairs
        fwd_1 = fwd_returns[self.horizons[0]]
        hit_mask = valid & ~np.isnan(fwd_1)
        if hit_mask.sum() > 0:
            agree = np.sign(values[hit_mask]) == np.sign(fwd_1[hit_mask])
            hit_rate = float(agree.mean())
        else:
            hit_rate = 0.5

        # ── Quintile analysis ──
        # q_spread = (E[r_1 | Q5] - E[r_1 | Q1]) * 1e4  in bps
        q_spread, q_mono = self._quintile_analysis(
            values, fwd_returns[self.horizons[0]]
        )

        # ── Autocorrelation ──
        # rho_1 = Pearson autocorrelation of (x - x_bar) at lag 1
        autocorr = self._autocorr_1(values[valid])

        # ── Edge estimation ──
        # gross_edge = |q_spread| / 2  (per-trade estimate from quintile half-spread)
        gross_edge = abs(q_spread) / 2.0  # half-spread is per-trade edge
        round_trip_cost = self.cost_bps * 2.0  # entry + exit
        # Adjust cost by turnover: high autocorr = lower effective turnover
        # turnover_factor in [0.1, 1.0]: smaller when signal is persistent
        turnover_factor = max(0.1, 1.0 - autocorr) if autocorr > 0 else 1.0
        net_edge = gross_edge - round_trip_cost * turnover_factor

        # ── Regime-conditional IC ──
        # IC restricted to high/low regimes of VPIN, entropy, spread
        conditional = self._conditional_ic(values, fwd_returns[self.horizons[0]], bars)

        # ── Classification ──
        # Rule-based assignment into {directional, gate, regime, noise}
        role = self._classify(
            best_ic, hit_rate, autocorr, conditional, nan_rate
        )

        # ── Composite score ──
        # Weighted linear combination of normalised sub-scores; see _compute_score
        scalp_score = self._compute_score(
            best_ic, ic_ir, hit_rate, q_spread, autocorr, net_edge, role
        )

        return FeatureProfile(
            name=name,
            vector=vector,
            ic=ic_map,
            ic_best_horizon=best_h,
            ic_best=best_ic,
            ic_ir=ic_ir,
            hit_rate=hit_rate,
            quintile_spread_bps=q_spread,
            quintile_monotonic=q_mono,
            autocorr_1=autocorr,
            nan_rate=nan_rate,
            gross_edge_bps=gross_edge,
            net_edge_bps=net_edge,
            conditional=conditional,
            role=role,
            scalp_score=scalp_score,
        )

    # ── Profile all features ──────────────────────────────────────────

    def profile_all(self, bars: pd.DataFrame) -> ProfileReport:
        """Profile every numeric feature column in bars."""
        fwd_returns = self.compute_forward_returns(bars)

        # Identify feature columns (exclude meta)
        meta = {"bar_start", "bar_end", "symbol", "tick_count"} | META_COLUMNS
        feature_cols = [
            c for c in bars.columns
            if c not in meta and bars[c].dtype in (np.float64, np.float32, float)
        ]
        log.info("Profiling %d feature columns", len(feature_cols))

        profiles = []
        for col in feature_cols:
            vector = self._detect_vector(col)
            values = bars[col].values.astype(np.float64)
            prof = self.profile_feature(values, fwd_returns, col, vector, bars)
            if prof is not None:
                profiles.append(prof)

        # Sort by scalp_score descending
        profiles.sort(key=lambda p: p.scalp_score, reverse=True)

        report = ProfileReport(
            symbol=self.config.get("symbol", "BTC"),
            timeframe=self.timeframe,
            n_bars=len(bars),
            n_features=len(profiles),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
            profiles=profiles,
            config=self.config,
        )
        report.n_directional = sum(1 for p in profiles if p.role == "directional")
        report.n_gate = sum(1 for p in profiles if p.role == "gate")
        report.n_regime = sum(1 for p in profiles if p.role == "regime")
        report.n_noise = sum(1 for p in profiles if p.role == "noise")

        return report

    # ── Forward test ──────────────────────────────────────────────────

    def forward_test(
        self, bars: pd.DataFrame, split: float = 0.7
    ) -> ForwardTestResult:
        """Temporal split validation of feature alpha stability.

        Methodology
        -----------
        The bar sequence is split at a deterministic index with no overlap,
        preserving time order to avoid look-ahead bias:

            split_idx = floor(N * split)

            IS  = bars[0 : split_idx]          (in-sample,      length = split_idx)
            OOS = bars[split_idx : N]           (out-of-sample,  length = N - split_idx)

        Both splits must contain at least self.min_obs bars; otherwise a
        ValueError is raised.

        In-sample profiling
        -------------------
        profile_all(IS) is run in full, producing a ranked list of features
        with IC, score, role, etc.  Only the top self.top_n features (by
        scalp_score) are carried forward to the OOS comparison.

        Out-of-sample profiling
        -----------------------
        For each feature in the IS top-N, profile_feature is called on the OOS
        split independently.  Forward returns are recomputed from OOS prices.

        Per-feature comparison
        ----------------------
        For each matched feature, the following quantities are recorded:

            ic_change_pct = (IC_OOS - IC_IS) / max(|IC_IS|, 1e-8) * 100   [%]

        IC rank correlation
        -------------------
        Let  IS_ic = [IC_IS^1, ..., IC_IS^K]  and  OOS_ic = [IC_OOS^1, ..., IC_OOS^K]
        be the best-horizon IC values across the K matched features.  The
        aggregate stability measure is:

            ic_correlation = Spearman_rho(IS_ic, OOS_ic)

        This tests whether the ordinal ranking of features by predictive power
        is preserved in the OOS period.  A value near +1 means the ranking is
        stable; near 0 or negative means the IS ranking is uninformative OOS.

        Stability classification
        ------------------------
            stable    : |IC_OOS| >= min_ic
            degraded  : |IC_IS| > min_ic  AND  |ic_change_pct| > 50

        Verdict threshold (in print_forward_test)
        ------------------------------------------
            PASS  if  ic_correlation > 0.3  AND  degraded_count < K/2
            FAIL  otherwise

        Parameters
        ----------
        bars : pd.DataFrame
            Full bar sequence, time-ordered.
        split : float in (0, 1)
            Fraction of bars used as in-sample (default 0.7).

        Returns
        -------
        ForwardTestResult
        """
        n = len(bars)
        split_idx = int(n * split)
        # OOS can run with fewer bars than full profiling (validation, not discovery)
        ft_min = min(self.min_obs, n // 4)
        if split_idx < ft_min or (n - split_idx) < ft_min:
            raise ValueError(
                f"Not enough bars for forward test: {n} total, "
                f"need ≥{ft_min} in each split"
            )

        is_bars = bars.iloc[:split_idx].copy().reset_index(drop=True)
        oos_bars = bars.iloc[split_idx:].copy().reset_index(drop=True)

        log.info("Forward test: IS=%d bars, OOS=%d bars", len(is_bars), len(oos_bars))

        # Profile in-sample
        is_report = self.profile_all(is_bars)

        # Profile out-of-sample — relax min_obs since we're validating, not discovering
        saved_min = self.min_obs
        self.min_obs = max(30, n - split_idx)
        oos_fwd = self.compute_forward_returns(oos_bars)
        meta = {"bar_start", "bar_end", "symbol", "tick_count"} | META_COLUMNS

        comparisons = []
        is_ics = []
        oos_ics = []

        for is_prof in is_report.profiles[:self.top_n]:
            col = is_prof.name
            if col not in oos_bars.columns:
                continue

            oos_values = oos_bars[col].values.astype(np.float64)
            oos_prof = self.profile_feature(
                oos_values, oos_fwd, col, is_prof.vector, oos_bars
            )
            if oos_prof is None:
                continue

            is_ics.append(is_prof.ic_best)
            oos_ics.append(oos_prof.ic_best)

            # ic_change_pct = (IC_OOS - IC_IS) / max(|IC_IS|, 1e-8) * 100
            comparisons.append({
                "name": col,
                "vector": is_prof.vector,
                "role": is_prof.role,
                "is_ic": is_prof.ic_best,
                "oos_ic": oos_prof.ic_best,
                "ic_change_pct": (
                    (oos_prof.ic_best - is_prof.ic_best) / max(abs(is_prof.ic_best), 1e-8) * 100
                ),
                "is_hit_rate": is_prof.hit_rate,
                "oos_hit_rate": oos_prof.hit_rate,
                "is_edge_bps": is_prof.net_edge_bps,
                "oos_edge_bps": oos_prof.net_edge_bps,
                "is_score": is_prof.scalp_score,
                "oos_score": oos_prof.scalp_score,
            })

        # Restore original threshold
        self.min_obs = saved_min

        # Aggregate validation
        # ic_correlation = Spearman_rho(IS_ic, OOS_ic)  across K matched features
        if len(is_ics) >= 3:
            ic_corr = _safe_spearman(np.array(is_ics), np.array(oos_ics))
        else:
            ic_corr = 0.0

        stable = sum(1 for c in comparisons if abs(c["oos_ic"]) >= self.min_ic)
        degraded = sum(
            1 for c in comparisons
            if abs(c["is_ic"]) > self.min_ic and abs(c["ic_change_pct"]) > 50
        )

        return ForwardTestResult(
            symbol=self.config.get("symbol", "BTC"),
            timeframe=self.timeframe,
            in_sample_bars=len(is_bars),
            out_of_sample_bars=len(oos_bars),
            split_ratio=split,
            comparisons=comparisons,
            ic_correlation=ic_corr,
            stable_count=stable,
            degraded_count=degraded,
        )

    # ── Walk-forward k-fold validation ────────────────────────────────

    def forward_test_walkforward(
        self,
        bars: pd.DataFrame,
        n_folds: int = 5,
        min_train_bars: int = 200,
        min_fold_obs: int = 30,
    ) -> WalkForwardResult:
        """Walk-forward k-fold validation with expanding training window.

        Why this replaces the single 70/30 split
        ----------------------------------------
        The single-split forward_test produces a single point estimate of OOS
        IC per feature.  When the OOS chunk happens to be a contiguous trending
        regime — common at small sample sizes — every feature collapses to a
        degenerate (hit_rate=1, IC=0) result, which is a methodological
        artifact rather than evidence of feature instability.  k-fold gives
        a distribution of OOS ICs across non-overlapping windows, and sign
        consistency across folds is a much more robust stability signal.

        Procedure
        ---------
        1. fold_len = floor((N - min_train_bars) / n_folds)
        2. For each feature column f and each horizon h in self.horizons:
             For each fold i in [0, n_folds):
                 t_split = min_train_bars + i*fold_len
                 t_end   = t_split + fold_len    (or N for the last fold)
                 IS_IC[f,h,i]  = Spearman(x[t in [0, t_split-h)],  r_h[same])
                 OOS_IC[f,h,i] = Spearman(x[t in [t_split, t_end-h)], r_h[same])
        3. For each feature, pick best horizon h*:
                 h* = argmax_h |mean_i IS_IC[f,h,i]|
           Selection uses IS-only data, so no leakage.
        4. Aggregate at h*:
                 is_ic_mean, oos_ic_mean, oos_ic_std (across folds)
                 sign_consistency = mean_i 1{ sign(IS_IC[i]) == sign(OOS_IC[i]) }
                                    over folds where |IS_IC[i]| > 1e-6
        5. Decision:
                 keep    if sign_consistency >= 0.6 AND |oos_ic_mean| >= min_ic
                 monitor if sign_consistency >= 0.6 AND |oos_ic_mean| <  min_ic
                 drop    otherwise

        Complexity
        ----------
        O(F * H * K * N log N) where F = features, H = |horizons|,
        K = n_folds, N = bars.  Dominated by Spearman ranking.
        For the current corpus (F~140, H=4, K=5, N~3000) this is well under
        a minute.

        Parameters
        ----------
        bars : pd.DataFrame
            Time-ordered bar sequence.
        n_folds : int, default 5
            Number of expanding-window folds.
        min_train_bars : int, default 200
            Bars in fold 0's training window.  Subsequent folds expand by
            fold_len each.
        min_fold_obs : int, default 30
            Minimum non-NaN paired observations required in a fold to
            compute an IC; otherwise IC is recorded as 0.0.

        Returns
        -------
        WalkForwardResult
            Per-feature aggregates plus keep/monitor/drop counts.

        Raises
        ------
        ValueError
            If the bar sequence is too short to fit n_folds folds with at
            least min_fold_obs OOS bars each.
        """
        n = len(bars)
        if n < min_train_bars + n_folds * min_fold_obs:
            raise ValueError(
                f"Not enough bars for walk-forward: have {n}, need "
                f">= {min_train_bars + n_folds * min_fold_obs} "
                f"(min_train_bars={min_train_bars}, n_folds={n_folds}, "
                f"min_fold_obs={min_fold_obs})"
            )
        fold_len = (n - min_train_bars) // n_folds
        if fold_len < min_fold_obs:
            raise ValueError(
                f"fold_len={fold_len} < min_fold_obs={min_fold_obs}; "
                f"reduce n_folds or min_train_bars"
            )

        log.info(
            "Walk-forward: N=%d, k=%d, min_train=%d, fold_len=%d",
            n, n_folds, min_train_bars, fold_len,
        )

        # Forward returns r_h[t] = log(p[t+h]/p[t]) computed once on full sample
        fwd = self.compute_forward_returns(bars)

        # Identify feature columns (same gate as profile_all)
        meta = {"bar_start", "bar_end", "symbol", "tick_count"} | META_COLUMNS
        feature_cols = [
            c for c in bars.columns
            if c not in meta and bars[c].dtype in (np.float64, np.float32, float)
        ]
        log.info("Walk-forward over %d feature columns", len(feature_cols))

        # Pre-compute fold boundaries to avoid recomputing per feature
        # Each entry: (t_split, t_end) where IS = [0, t_split), OOS = [t_split, t_end)
        boundaries = []
        for i in range(n_folds):
            t_split = min_train_bars + i * fold_len
            t_end = n if i == n_folds - 1 else t_split + fold_len
            boundaries.append((t_split, t_end))

        features: List[WalkForwardFeature] = []
        for col in feature_cols:
            x = bars[col].values.astype(np.float64)

            # For each horizon, collect IS and OOS IC across folds
            per_horizon: Dict[int, Dict[str, List[float]]] = {}
            for h in self.horizons:
                r = fwd[h]
                is_ics: List[float] = []
                oos_ics: List[float] = []
                for (t_split, t_end) in boundaries:
                    # IS pairs: t in [0, t_split - h) so r[t] uses bar t+h <= t_split-1
                    is_hi = max(t_split - h, 0)
                    is_x = x[:is_hi]
                    is_r = r[:is_hi]
                    is_mask = np.isfinite(is_x) & np.isfinite(is_r)
                    is_ic = (
                        _safe_spearman(is_x[is_mask], is_r[is_mask])
                        if is_mask.sum() >= min_fold_obs else 0.0
                    )
                    is_ics.append(is_ic)

                    # OOS pairs: t in [t_split, t_end - h)
                    oos_hi = max(t_end - h, t_split)
                    oos_x = x[t_split:oos_hi]
                    oos_r = r[t_split:oos_hi]
                    oos_mask = np.isfinite(oos_x) & np.isfinite(oos_r)
                    oos_ic = (
                        _safe_spearman(oos_x[oos_mask], oos_r[oos_mask])
                        if oos_mask.sum() >= min_fold_obs else 0.0
                    )
                    oos_ics.append(oos_ic)
                per_horizon[h] = {"is": is_ics, "oos": oos_ics}

            # Pick horizon with max |mean(IS_IC)| — IS-only selection, no leakage
            best_h = max(
                self.horizons,
                key=lambda h: abs(float(np.mean(per_horizon[h]["is"]))),
            )
            is_arr = np.array(per_horizon[best_h]["is"])
            oos_arr = np.array(per_horizon[best_h]["oos"])

            is_mean = float(np.mean(is_arr))
            oos_mean = float(np.mean(oos_arr))
            oos_std = float(np.std(oos_arr, ddof=0))

            # Sign consistency over folds with non-degenerate IS_IC
            nonzero = np.abs(is_arr) > 1e-6
            if nonzero.any():
                sign_match = (np.sign(is_arr[nonzero]) == np.sign(oos_arr[nonzero]))
                sign_consistency = float(sign_match.mean())
            else:
                sign_consistency = 0.0

            # Decision rule
            if sign_consistency >= 0.6 and abs(oos_mean) >= self.min_ic:
                decision = "keep"
            elif sign_consistency >= 0.6:
                decision = "monitor"
            else:
                decision = "drop"

            # Skip features that produced all-zero IS ICs across all horizons
            # (constant features, all-NaN, etc.) — they pollute output without info
            all_zero = all(
                all(abs(v) < 1e-9 for v in per_horizon[h]["is"])
                for h in self.horizons
            )
            if all_zero:
                continue

            features.append(WalkForwardFeature(
                name=col,
                vector=self._detect_vector(col),
                horizon=int(best_h),
                n_folds=n_folds,
                is_ic_per_fold=[float(v) for v in is_arr],
                oos_ic_per_fold=[float(v) for v in oos_arr],
                is_ic_mean=is_mean,
                oos_ic_mean=oos_mean,
                oos_ic_std=oos_std,
                sign_consistency=sign_consistency,
                decision=decision,
            ))

        # Sort by |oos_ic_mean| descending — most predictive OOS first
        features.sort(key=lambda f: abs(f.oos_ic_mean), reverse=True)

        keep = sum(1 for f in features if f.decision == "keep")
        monitor = sum(1 for f in features if f.decision == "monitor")
        drop = sum(1 for f in features if f.decision == "drop")

        return WalkForwardResult(
            symbol=self.config.get("symbol", "BTC"),
            timeframe=self.timeframe,
            n_folds=n_folds,
            total_bars=n,
            min_train_bars=min_train_bars,
            fold_len=fold_len,
            horizons=list(self.horizons),
            features=features,
            keep_count=keep,
            monitor_count=monitor,
            drop_count=drop,
        )

    # ── Internal helpers ──────────────────────────────────────────────

    def _find_price_col(self, bars: pd.DataFrame) -> str:
        for c in [
            "raw_midprice_close", "raw_midprice_last", "raw_midprice_mean",
            "raw_microprice_close", "raw_microprice_last", "raw_microprice_mean",
        ]:
            if c in bars.columns:
                return c
        mid_cols = [c for c in bars.columns if "midprice" in c]
        if mid_cols:
            return mid_cols[0]
        raise ValueError(f"No price column found in bars. Columns: {list(bars.columns)[:20]}")

    def _rolling_ic_ir(self, values: np.ndarray, fwd: np.ndarray) -> float:
        """IC Information Ratio: stability of the IC signal over time.

        Mathematical definition
        -----------------------
        Partition the time axis into overlapping windows of size W, stepping
        by W/2 (50 % overlap).  For each window starting at index s:

            window_s = [s, s + W)
            v_s = values[window_s],  f_s = fwd[window_s]
            mask_s = {i : v_s[i] not NaN  AND  f_s[i] not NaN}

            IC_s = Spearman_rho(v_s[mask_s], f_s[mask_s])
                   if |mask_s| >= 20,  else window is skipped

        Windows where _safe_spearman returns exactly 0.0 (degenerate) are also
        excluded.

        Let M be the number of valid windows and  {IC_s}_{s=1}^{M}  be their ICs.
        The IC IR is:

            IC_IR = E_M[IC_s] / std_M[IC_s]
                  = ( (1/M) sum_{s=1}^{M} IC_s )
                    / sqrt( (1/M) sum_{s=1}^{M} (IC_s - E_M[IC_s])^2 )

        where std is the population standard deviation (numpy default, ddof=0).

        Interpretation
        --------------
        IC_IR is analogous to the Sharpe ratio of the IC time series.  A value
        of |IC_IR| >= 0.5 is considered acceptable stability; >= 1.0 is strong.
        Sign tracks the IC sign (positive = feature predicts direction).

        Boundary and degeneracy conditions
        -----------------------------------
          - Returns 0.0 if N < 2W  (fewer than two non-overlapping windows)
          - Returns 0.0 if M < 3   (not enough windows for a stable estimate)
          - Returns 0.0 if std < 1e-10  (all window ICs identical — perfectly
            stable IC but variance undefined)

        Parameters
        ----------
        values : np.ndarray, shape (N,)
            Feature values (may contain NaN).
        fwd : np.ndarray, shape (N,)
            Forward returns for the best horizon (may contain NaN).

        Returns
        -------
        float
            IC IR in (-inf, +inf), clipped to 0.0 for degenerate cases.

        Computational complexity
        ------------------------
        O(N * W) in the worst case due to per-window Spearman calls; in
        practice O(N log W) because each window is length W and Spearman
        is O(W log W).  Number of windows is approximately 2N/W.
        """
        w = self.rolling_ic_window
        n = len(values)
        if n < w * 2:
            return 0.0

        ics = []
        # Overlapping windows: step = W/2, so consecutive windows share W/2 bars.
        # Window indices: s in {0, W/2, W, 3W/2, ...} while s + W <= N
        for start in range(0, n - w, w // 2):
            end = start + w
            v = values[start:end]
            f = fwd[start:end]
            mask = ~np.isnan(v) & ~np.isnan(f)
            if mask.sum() < 20:
                continue
            corr = _safe_spearman(v[mask], f[mask])
            if corr != 0.0:
                ics.append(corr)

        if len(ics) < 3:
            return 0.0
        arr = np.array(ics)
        std = arr.std()  # population std (ddof=0)
        if std < 1e-10:
            return 0.0
        # IC_IR = E[IC_s] / std[IC_s]
        return float(arr.mean() / std)

    def _quintile_analysis(
        self, values: np.ndarray, fwd: np.ndarray
    ) -> Tuple[float, bool]:
        """Quintile return spread and monotonicity check.

        Mathematical definition
        -----------------------
        Let (x, r) be the joint valid sample after removing NaN pairs:

            valid_mask = {t : x[t] not NaN  AND  r[t] not NaN},   |valid_mask| = n

        Partition x into 5 equal-count buckets (quintiles) Q_1, ..., Q_5 by
        their rank order:

            Q_k = { t in valid_mask : (k-1)/5 <= F_n(x[t]) < k/5 }

        where F_n is the empirical CDF.  Implementation uses pd.qcut with
        duplicates="drop", which may merge bins if x has repeated values,
        reducing the number of distinct quintiles (handled gracefully below).

        For each occupied quintile Q_k, compute the mean forward return:

            mu_k = E[r | t in Q_k] = (1/|Q_k|) * sum_{t in Q_k} r[t]

        Quintile spread (in bps)
        -------------------------
            spread = (mu_{K} - mu_{1}) * 1e4

        where K is the highest observed quintile index (normally 5, but may be
        fewer if pd.qcut merges bins).  This is the return difference between
        the top and bottom feature quintiles, expressed in basis points.

        Monotonicity
        ------------
        Let delta_k = mu_{k+1} - mu_k  for k = 1, ..., K-1.

            monotonic = True   if  all(delta_k >= 0)  OR  all(delta_k <= 0)
                      = False  otherwise

        Monotonicity is a stronger condition than a non-zero spread: it
        requires a consistent ordering across all intermediate quintiles, not
        just the extremes.

        Edge cases
        ----------
          - Returns (0.0, False) if n < 50  (insufficient data)
          - Returns (0.0, False) if pd.qcut raises ValueError (constant x)
          - Returns (0.0, False) if fewer than 2 quintiles are populated

        Parameters
        ----------
        values : np.ndarray, shape (N,)
            Feature values (may contain NaN).
        fwd : np.ndarray, shape (N,)
            1-bar forward returns (may contain NaN).

        Returns
        -------
        spread : float
            (mu_K - mu_1) * 1e4 in basis points.  Positive means high feature
            values predict positive returns; negative means the opposite.
        monotonic : bool
            True if quintile means are globally non-decreasing or non-increasing.

        Computational complexity
        ------------------------
        O(n log n) due to pd.qcut (sorting) and per-quintile mean: O(n).
        """
        mask = ~np.isnan(values) & ~np.isnan(fwd)
        if mask.sum() < 50:
            return 0.0, False

        v = values[mask]
        f = fwd[mask]

        try:
            # pd.qcut assigns each observation to one of 5 equal-frequency bins.
            # labels=False produces integer bin indices 0..4 (0 = lowest quantile).
            quintiles = pd.qcut(v, 5, labels=False, duplicates="drop")
        except ValueError:
            return 0.0, False

        # Compute per-quintile mean of forward returns
        q_means = []
        for q in sorted(np.unique(quintiles)):
            bucket = f[quintiles == q]
            if len(bucket) == 0:
                continue
            q_means.append(np.nanmean(bucket))

        if len(q_means) < 2:
            return 0.0, False

        # spread = (mu_K - mu_1) * 1e4  in bps
        spread = (q_means[-1] - q_means[0]) * 1e4  # convert to bps
        # Check monotonicity: are quintile means increasing or decreasing?
        # delta_k = mu_{k+1} - mu_k  for k = 1..K-1
        diffs = np.diff(q_means)
        monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))

        return float(spread), monotonic

    def _autocorr_1(self, x: np.ndarray) -> float:
        """Lag-1 Pearson autocorrelation of the mean-centred feature signal.

        Mathematical definition
        -----------------------
        Given a mean-centred sequence  z[t] = x[t] - x_bar  for t = 0..n-1,
        the lag-1 (unnormalised) autocorrelation estimator is:

            rho_1 = C(1) / C(0)

        where the autocovariance at lag k is:

            C(k) = sum_{t=0}^{n-k-1} z[t] * z[t+k]

        and the total variance term is:

            C(0) = sum_{t=0}^{n-1} z[t]^2

        Substituting k = 1:

            rho_1 = [ sum_{t=0}^{n-2} z[t] * z[t+1] ]
                    / [ sum_{t=0}^{n-1} z[t]^2 ]

        This is the biased estimator (both numerator and denominator use the
        same denominator n rather than n-1 and n-1 respectively), which is
        standard in signal processing contexts.

        Note: C(0) in the denominator uses ALL n lags, not n-1, while the
        numerator uses only n-1 paired products.  This gives a slight downward
        bias compared to the standard Pearson sample correlation, but ensures
        |rho_1| <= 1 always holds.

        Interpretation for turnover estimation
        ---------------------------------------
        Autocorrelation proxies signal persistence:

            rho_1 near +1  →  signal rarely changes sign  →  low turnover
            rho_1 near  0  →  signal is white-noise-like  →  high turnover
            rho_1 near -1  →  signal alternates sign every bar  →  very high turnover

        For scalping, moderate autocorrelation (0.3 to 0.8) is desirable:
        persistent enough to hold a position, but responsive enough to generate
        trade opportunities.  rho_1 > 0.95 typically indicates a slowly-varying
        regime indicator rather than a tradeable signal.

        Parameters
        ----------
        x : np.ndarray, shape (n,)
            Feature values with NaN already removed (caller passes values[valid]).

        Returns
        -------
        float in approximately (-1, 1), or 0.0 for degenerate cases.
            Exactly 0.0 is returned if n < 10 or if sum(z^2) < 1e-15
            (constant or near-constant signal).

        Computational complexity
        ------------------------
        O(n) time and O(n) space (for the centred copy z).
        """
        if len(x) < 10:
            return 0.0
        # z[t] = x[t] - x_bar  (mean-centre to remove DC offset)
        x = x - x.mean()
        # C(0) = sum_{t} z[t]^2
        denom = np.sum(x ** 2)
        if denom < 1e-15:
            return 0.0
        # rho_1 = sum_{t=0}^{n-2} z[t]*z[t+1]  /  sum_{t=0}^{n-1} z[t]^2
        return float(np.sum(x[:-1] * x[1:]) / denom)

    def _conditional_ic(
        self,
        values: np.ndarray,
        fwd: np.ndarray,
        bars: pd.DataFrame,
    ) -> List[ConditionalIC]:
        """Compute IC of the feature restricted to high/low regime subsets.

        Mathematical definition
        -----------------------
        For each conditioning variable C (VPIN, entropy, or spread) with
        observed values c[t]:

            median_C = median( c[t] : c[t] not NaN )

        Two regime subsets are defined:

            R_low  = { t : c[t] < median_C }
            R_high = { t : c[t] >= median_C }

        For each subset R in {R_low, R_high}, the conditional IC is:

            IC(R) = Spearman_rho(
                        x[t]   for t in R ∩ valid_mask,
                        r_1[t] for t in R ∩ valid_mask
                    )

        where  valid_mask = { t : x[t] not NaN  AND  r_1[t] not NaN  AND
                                    c[t] not NaN }.

        Regimes evaluated
        -----------------
        Three conditioning variables are attempted, each with a prioritised
        list of candidate column names (first found in bars is used):

          1. VPIN (Volume-Synchronized Probability of Informed Trading)
             Candidates: "toxic_vpin_10_mean", "toxic_vpin_10_last", "toxic_vpin_10"
             Interpretation: high VPIN = toxic order flow; IC may differ
             between informed and uninformed flow regimes.

          2. Order-flow entropy
             Candidates: "ent_tick_1m_mean", "ent_tick_1m_last", "ent_tick_1m"
             Interpretation: low entropy = concentrated, directional flow;
             high entropy = dispersed, noisy flow.

          3. Spread (in bps)
             Candidates: "raw_spread_bps_mean", "raw_spread_bps_last", "raw_spread_bps"
             Interpretation: wide spread = low liquidity, higher adverse
             selection; IC in spread regimes reveals liquidity sensitivity.

        Conditions with fewer than 30 valid observations are skipped.

        Utility for classification
        --------------------------
        If a feature shows IC(R_high) >> IC(R_low) (or vice versa), it is
        better used as a gate — signal only when the conditioning regime is
        active — than as a direct directional signal.  The IC range:

            ic_range = max_R IC(R) - min_R IC(R)

        is used in _classify to detect gate-type features.

        Parameters
        ----------
        values : np.ndarray, shape (N,)
            Feature values (may contain NaN).
        fwd : np.ndarray, shape (N,)
            1-bar forward returns (may contain NaN).
        bars : pd.DataFrame
            Bar DataFrame used to look up conditioning columns.

        Returns
        -------
        List[ConditionalIC]
            One entry per (variable, regime) pair for which a valid column was
            found and n_obs >= 30.  May be empty.
        """
        results = []
        conditions = {
            "vpin": [
                "toxic_vpin_10_mean", "toxic_vpin_10_last", "toxic_vpin_10",
            ],
            "entropy": [
                "ent_tick_1m_mean", "ent_tick_1m_last", "ent_tick_1m",
            ],
            "spread": [
                "raw_spread_bps_mean", "raw_spread_bps_last", "raw_spread_bps",
            ],
        }

        for cond_name, candidates in conditions.items():
            cond_col = None
            for c in candidates:
                if c in bars.columns:
                    cond_col = c
                    break
            if cond_col is None:
                continue

            cond_vals = bars[cond_col].values.astype(np.float64)
            valid_cond = ~np.isnan(cond_vals)
            if valid_cond.sum() < self.min_obs:
                continue

            # median_C = median of conditioning variable over all valid observations
            median_val = np.nanmedian(cond_vals)

            for label, mask_fn in [
                (f"{cond_name}_low", lambda cv: cv < median_val),
                (f"{cond_name}_high", lambda cv: cv >= median_val),
            ]:
                # regime_mask = R_low or R_high, intersected with valid feature/return mask
                regime_mask = valid_cond & mask_fn(cond_vals) & ~np.isnan(values) & ~np.isnan(fwd)
                n_obs = int(regime_mask.sum())
                if n_obs < 30:
                    continue
                # IC(R) = Spearman_rho(x[regime_mask], r_1[regime_mask])
                ic_val = _safe_spearman(values[regime_mask], fwd[regime_mask])
                results.append(ConditionalIC(condition=label, ic=ic_val, n_obs=n_obs))

        return results

    def _classify(
        self,
        best_ic: float,
        hit_rate: float,
        autocorr: float,
        conditional: List[ConditionalIC],
        nan_rate: float,
    ) -> str:
        """Rule-based classification of a feature into one of four roles.

        Decision tree
        -------------
        The following rules are applied in order; the first match is returned.

        Rule 0 — Missing data gate
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
            if nan_rate > 0.8:
                return "noise"
            # Feature is mostly missing; unreliable regardless of other stats.

        Rule 1 — Regime-gate detection (strong regime variability, weak IC)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ic_range = max({IC(R)}) - min({IC(R)})  over all conditional regimes R

            if ic_range > 0.05  AND  |IC*| < 2 * min_ic:
                return "gate"
            # Feature's predictive power is regime-dependent but its overall IC
            # is too weak to be directional.  Best used as a filter/gate that
            # activates other signals when the regime is favourable.

        Rule 2 — Directional or regime indicator (strong overall IC)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if |IC*| >= min_ic  AND  hit_rate >= min_hit_rate:
                if autocorr_1 > 0.95:
                    return "regime"
                    # Very slow-moving signal; carries macro state information
                    # but is not a fast scalping signal.
                return "directional"
                # Strong, fast, directionally informative signal.

        Rule 3 — Conditional gate (regime-dependent but overall IC weak)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if any |IC(R)| >= min_ic  AND  |IC*| < min_ic:
                return "gate"
            # At least one regime shows meaningful IC, but the unconditional
            # IC averages out.  Feature is useful as a conditional gate.

        Rule 4 — Default
        ~~~~~~~~~~~~~~~~
            return "noise"
            # No meaningful signal detected under any condition.

        Parameters
        ----------
        best_ic : float
            IC* = IC at the best horizon.
        hit_rate : float
            Directional hit rate (fraction of sign agreements with 1-bar return).
        autocorr : float
            Lag-1 autocorrelation of the feature signal.
        conditional : List[ConditionalIC]
            Regime-conditional IC results from _conditional_ic.
        nan_rate : float
            Fraction of NaN values in the feature column.

        Returns
        -------
        str : one of {"directional", "gate", "regime", "noise"}
        """
        if nan_rate > 0.8:
            return "noise"

        abs_ic = abs(best_ic)

        # Check if IC varies significantly across regimes → gate
        # ic_range = max_R IC(R) - min_R IC(R)
        if conditional:
            ics = [c.ic for c in conditional]
            ic_range = max(ics) - min(ics) if len(ics) > 1 else 0
            if ic_range > 0.05 and abs_ic < self.min_ic * 2:
                return "gate"

        # Strong IC + reasonable hit rate → directional
        if abs_ic >= self.min_ic and hit_rate >= self.min_hit_rate:
            # Slow-changing signal = regime indicator (autocorr > 0.95 threshold)
            if autocorr > 0.95:
                return "regime"
            return "directional"

        # Weak IC but regime-dependent → gate
        if conditional:
            high_conds = [c for c in conditional if abs(c.ic) >= self.min_ic]
            if len(high_conds) >= 1 and abs_ic < self.min_ic:
                return "gate"

        return "noise"

    def _compute_score(
        self,
        best_ic: float,
        ic_ir: float,
        hit_rate: float,
        q_spread: float,
        autocorr: float,
        net_edge: float,
        role: str,
    ) -> float:
        """Composite scalping viability score.

        Mathematical definition
        -----------------------
        The score is a weighted sum of six normalised sub-scores, each mapped
        to [0, 1] before weighting.  The total is then clipped to [0, 1] and
        modified by a role-dependent penalty multiplier.

        Sub-scores and weights
        ----------------------

        1. IC magnitude  (weight = 0.25)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           Reference range: |IC*| in [0, 0.15].  Normalisation:

               s_IC = min( |IC*| / 0.15,  1.0 )

           Contribution: s_IC * 0.25

        2. IC stability  (weight = 0.15)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           Reference range: |IC_IR| in [0, 1.0].  Normalisation:

               s_IR = min( |IC_IR| / 1.0,  1.0 )

           Contribution: s_IR * 0.15

        3. Hit rate  (weight = 0.15)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           Reference range: hit_rate in [0.50, 0.60] (excess over 0.5 chance level).
           Normalisation:

               s_HR = min( max(hit_rate - 0.5, 0) / 0.10,  1.0 )

           Contribution: s_HR * 0.15

        4. Quintile spread  (weight = 0.15)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           Reference range: |spread| in [0, 20] bps.  Normalisation:

               s_QS = min( |q_spread| / 20.0,  1.0 )

           Contribution: s_QS * 0.15

        5. Autocorrelation penalty  (weight = 0.10)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           The penalty factor ac_penalty depends on the autocorrelation regime:

               ac_penalty = 0.5   if  autocorr < 0.1     (noise: too random)
               ac_penalty = 0.6   if  autocorr > 0.95    (regime: too slow)
               ac_penalty = 1.0   otherwise               (ideal range [0.1, 0.95])

           Contribution: ac_penalty * 0.10

           Note: This sub-score is not normalised from 0; it provides a fixed
           weight of 0.05–0.10 depending on the autocorrelation regime.

        6. Cost-adjusted edge  (weight = 0.20)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           Reference range: net_edge in [0, 5] bps.  Normalisation:

               s_edge = min( max(net_edge, 0) / 5.0,  1.0 )

           Contribution: s_edge * 0.20
           Negative net_edge contributes zero (not penalised beyond zero).

        Total before role penalty
        -------------------------
            total = s_IC*0.25 + s_IR*0.15 + s_HR*0.15 + s_QS*0.15
                    + ac_penalty*0.10 + s_edge*0.20

        Maximum achievable without role penalty:
            0.25 + 0.15 + 0.15 + 0.15 + 0.10 + 0.20 = 1.00

        Role penalty
        ------------
            role = "noise"        → total *= 0.20
            role = "gate"         → total *= 0.70
            role = "directional"  → no penalty (× 1.0)
            role = "regime"       → no penalty (× 1.0)

        Final output
        ------------
            scalp_score = round( min(total, 1.0), 4 )

        Output domain: [0, 1], with 4 decimal places of precision.

        Parameters
        ----------
        best_ic : float
            IC* = IC at the best horizon.
        ic_ir : float
            IC Information Ratio (mean / std of rolling IC).
        hit_rate : float
            Directional hit rate in [0, 1].
        q_spread : float
            Quintile return spread in bps.
        autocorr : float
            Lag-1 autocorrelation of the feature signal.
        net_edge : float
            Cost-adjusted per-trade edge in bps.
        role : str
            Feature role from _classify: "directional", "gate", "regime", or "noise".

        Returns
        -------
        float in [0.0, 1.0] (4 decimal places).
        """
        # s_IC = min(|IC*| / 0.15, 1.0) * 0.25
        ic_score = min(abs(best_ic) / 0.15, 1.0) * 0.25

        # s_IR = min(|IC_IR| / 1.0, 1.0) * 0.15
        ir_score = min(abs(ic_ir) / 1.0, 1.0) * 0.15

        # s_HR = min(max(hit_rate - 0.5, 0) / 0.10, 1.0) * 0.15
        hr_score = min(max(hit_rate - 0.5, 0) / 0.1, 1.0) * 0.15

        # s_QS = min(|q_spread| / 20.0, 1.0) * 0.15
        qs_score = min(abs(q_spread) / 20.0, 1.0) * 0.15

        # Autocorrelation: moderate is best (0.3-0.8)
        # Too low = noise, too high = slow (regime)
        ac_penalty = 1.0
        if autocorr < 0.1:
            ac_penalty = 0.5
        elif autocorr > 0.95:
            ac_penalty = 0.6
        ac_score = ac_penalty * 0.1

        # s_edge = min(max(net_edge, 0) / 5.0, 1.0) * 0.20
        edge_score = min(max(net_edge, 0) / 5.0, 1.0) * 0.2

        total = ic_score + ir_score + hr_score + qs_score + ac_score + edge_score

        # Role penalty: noise gets crushed
        if role == "noise":
            total *= 0.2
        elif role == "gate":
            total *= 0.7

        return round(min(total, 1.0), 4)

    def _detect_vector(self, col_name: str) -> str:
        """Determine which feature vector a column belongs to."""
        # Strip aggregation suffixes
        base = col_name
        for suffix in ["_mean", "_std", "_last", "_slope", "_open", "_high", "_low", "_close", "_sum"]:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        for vname, vspec in FEATURE_VECTORS.items():
            if base in vspec["columns"]:
                return vname
            for prefix in vspec.get("prefixes", []):
                if base.startswith(prefix):
                    return vname
        return "unknown"


# ---------------------------------------------------------------------------
# Process state machine
# ---------------------------------------------------------------------------


class ProfilerState(str, Enum):
    IDLE = "IDLE"
    LOADING = "LOADING"
    AGGREGATING = "AGGREGATING"
    PROFILING = "PROFILING"
    RANKING = "RANKING"
    FORWARD_TESTING = "FORWARD_TESTING"
    DONE = "DONE"
    ERROR = "ERROR"


class ScalpingProfilerProcess:
    """
    State machine orchestrating the profiling pipeline.

    IDLE → LOADING → AGGREGATING → PROFILING → RANKING → FORWARD_TESTING → DONE

    State is persisted to disk for resume-on-interrupt.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_dir: str = "data/features",
        state_file: str = "data/profiler_state.json",
        report_dir: str = "reports/profiler",
    ):
        self.config = config
        self.data_dir = data_dir
        self.state_file = Path(state_file)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.profiler = ScalpingProfiler(config)
        self._state_data = self._load_state()

    @property
    def state(self) -> ProfilerState:
        return ProfilerState(self._state_data.get("state", ProfilerState.IDLE.value))

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"state": ProfilerState.IDLE.value}

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self._state_data, f, indent=2)

    def _transition(self, new_state: ProfilerState, **extra):
        old = self.state
        self._state_data["state"] = new_state.value
        self._state_data["last_transition"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._state_data.update(extra)
        self._save_state()
        log.info("%s → %s", old.value, new_state.value)

    def run(
        self,
        symbol: str = "BTC",
        forward_test: bool = False,
        top_n: Optional[int] = None,
        legacy_split: bool = False,
        n_folds: Optional[int] = None,
        min_train_bars: Optional[int] = None,
    ) -> Tuple[ProfileReport, Optional[Any]]:
        """Execute the full profiling pipeline.

        When forward_test=True:
            legacy_split=False (default) → walk-forward k-fold validation.
            legacy_split=True            → legacy single 70/30 split.
        """
        if top_n:
            self.profiler.top_n = top_n

        self.config["symbol"] = symbol
        self.profiler.config["symbol"] = symbol
        t0 = time.time()

        try:
            # ── LOADING ──
            self._transition(ProfilerState.LOADING, symbol=symbol)
            df = load_parquet(self.data_dir)
            df = filter_symbol(df, symbol)
            self._transition(
                ProfilerState.AGGREGATING,
                raw_ticks=len(df),
            )

            # ── AGGREGATING ──
            bars = aggregate_bars(df, self.profiler.timeframe)
            del df  # free memory
            self._transition(
                ProfilerState.PROFILING,
                n_bars=len(bars),
                n_columns=len(bars.columns),
            )

            # ── PROFILING ──
            report = self.profiler.profile_all(bars)
            self._transition(
                ProfilerState.RANKING,
                n_profiled=report.n_features,
                n_directional=report.n_directional,
                n_gate=report.n_gate,
                n_regime=report.n_regime,
                n_noise=report.n_noise,
            )

            # ── RANKING → save report ──
            report_path = self._save_report(report)

            # ── FORWARD TESTING (optional) ──
            ft_result = None
            if forward_test:
                self._transition(ProfilerState.FORWARD_TESTING)
                if legacy_split:
                    split = self.config.get("forward_test_split", 0.7)
                    ft_result = self.profiler.forward_test(bars, split)
                    ft_path = self._save_forward_test(ft_result)
                    self._transition(
                        ProfilerState.DONE,
                        elapsed_sec=round(time.time() - t0, 1),
                        report_path=str(report_path),
                        forward_test_path=str(ft_path),
                        ic_correlation=ft_result.ic_correlation,
                        stable_count=ft_result.stable_count,
                        degraded_count=ft_result.degraded_count,
                    )
                else:
                    nf = n_folds if n_folds is not None else int(
                        self.config.get("n_folds", 5)
                    )
                    mtb = min_train_bars if min_train_bars is not None else int(
                        self.config.get("min_train_bars", 200)
                    )
                    ft_result = self.profiler.forward_test_walkforward(
                        bars, n_folds=nf, min_train_bars=mtb
                    )
                    ft_path = self._save_walk_forward(ft_result)
                    self._transition(
                        ProfilerState.DONE,
                        elapsed_sec=round(time.time() - t0, 1),
                        report_path=str(report_path),
                        walk_forward_path=str(ft_path),
                        keep_count=ft_result.keep_count,
                        monitor_count=ft_result.monitor_count,
                        drop_count=ft_result.drop_count,
                    )
            else:
                self._transition(
                    ProfilerState.DONE,
                    elapsed_sec=round(time.time() - t0, 1),
                    report_path=str(report_path),
                )

            return report, ft_result

        except Exception as e:
            self._transition(ProfilerState.ERROR, error=str(e))
            raise

    def status(self) -> Dict:
        """Return current process state."""
        return dict(self._state_data)

    # ── Report I/O ────────────────────────────────────────────────────

    def _save_report(self, report: ProfileReport) -> Path:
        """Save profile report as JSON."""
        path = self.report_dir / f"profile_{report.symbol}_{report.timeframe}.json"

        data = {
            "symbol": report.symbol,
            "timeframe": report.timeframe,
            "n_bars": report.n_bars,
            "n_features": report.n_features,
            "timestamp": report.timestamp,
            "summary": {
                "directional": report.n_directional,
                "gate": report.n_gate,
                "regime": report.n_regime,
                "noise": report.n_noise,
            },
            "config": _json_safe(report.config),
            "profiles": [_profile_to_dict(p) for p in report.profiles],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)
        log.info("Report saved: %s", path)
        return path

    def _save_forward_test(self, result: ForwardTestResult) -> Path:
        """Save forward test results as JSON."""
        path = self.report_dir / f"forward_test_{result.symbol}_{result.timeframe}.json"

        data = {
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "in_sample_bars": result.in_sample_bars,
            "out_of_sample_bars": result.out_of_sample_bars,
            "split_ratio": result.split_ratio,
            "ic_correlation": result.ic_correlation,
            "stable_count": result.stable_count,
            "degraded_count": result.degraded_count,
            "comparisons": result.comparisons,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)
        log.info("Forward test saved: %s", path)
        return path

    def _save_walk_forward(self, result: WalkForwardResult) -> Path:
        """Save walk-forward results as JSON."""
        path = self.report_dir / f"walk_forward_{result.symbol}_{result.timeframe}.json"

        data = {
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "n_folds": result.n_folds,
            "total_bars": result.total_bars,
            "min_train_bars": result.min_train_bars,
            "fold_len": result.fold_len,
            "horizons": result.horizons,
            "keep_count": result.keep_count,
            "monitor_count": result.monitor_count,
            "drop_count": result.drop_count,
            "features": [asdict(f) for f in result.features],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)
        log.info("Walk-forward saved: %s", path)
        return path


# ---------------------------------------------------------------------------
# Printing / formatting
# ---------------------------------------------------------------------------


def print_profile_report(report: ProfileReport, top_n: int = 30):
    """Print a human-readable profile summary."""
    print(f"\n{'='*80}")
    print(f"  SCALPING SIGNAL PROFILER — {report.symbol} @ {report.timeframe}")
    print(f"{'='*80}")
    print(f"  Bars: {report.n_bars:,}  |  Features profiled: {report.n_features}")
    print(f"  Directional: {report.n_directional}  |  Gate: {report.n_gate}  "
          f"|  Regime: {report.n_regime}  |  Noise: {report.n_noise}")
    print(f"  Generated: {report.timestamp}")

    # Top features table
    print(f"\n  {'Rank':>4}  {'Feature':<40} {'Vector':<12} {'Role':<12} "
          f"{'IC':>7} {'H':>2} {'IR':>6} {'Hit%':>5} {'Q-Spr':>7} {'Edge':>6} {'Score':>6}")
    print(f"  {'-'*4}  {'-'*40} {'-'*12} {'-'*12} "
          f"{'-'*7} {'-'*2} {'-'*6} {'-'*5} {'-'*7} {'-'*6} {'-'*6}")

    for i, p in enumerate(report.profiles[:top_n]):
        ic_str = f"{p.ic_best:+.3f}"
        ir_str = f"{p.ic_ir:+.2f}"
        hr_str = f"{p.hit_rate:.1%}"
        qs_str = f"{p.quintile_spread_bps:+.1f}"
        edge_str = f"{p.net_edge_bps:+.1f}"
        score_str = f"{p.scalp_score:.3f}"
        name_trunc = p.name[:40]
        print(f"  {i+1:>4}  {name_trunc:<40} {p.vector:<12} {p.role:<12} "
              f"{ic_str:>7} {p.ic_best_horizon:>2} {ir_str:>6} {hr_str:>5} "
              f"{qs_str:>7} {edge_str:>6} {score_str:>6}")

    # Regime-conditional highlights
    print(f"\n  REGIME-CONDITIONAL HIGHLIGHTS")
    print(f"  {'Feature':<40} {'Condition':<16} {'IC':>7} {'N':>6}")
    print(f"  {'-'*40} {'-'*16} {'-'*7} {'-'*6}")

    shown = 0
    for p in report.profiles[:top_n]:
        for c in p.conditional:
            if abs(c.ic) >= 0.04:
                print(f"  {p.name[:40]:<40} {c.condition:<16} {c.ic:+.3f} {c.n_obs:>6}")
                shown += 1
                if shown >= 20:
                    break
        if shown >= 20:
            break

    if shown == 0:
        print("  (no strong regime-conditional effects found)")


def print_forward_test(result: ForwardTestResult):
    """Print forward test results."""
    print(f"\n{'='*80}")
    print(f"  FORWARD TEST — {result.symbol} @ {result.timeframe}")
    print(f"{'='*80}")
    print(f"  In-sample: {result.in_sample_bars:,} bars  |  "
          f"Out-of-sample: {result.out_of_sample_bars:,} bars  |  "
          f"Split: {result.split_ratio:.0%}/{1-result.split_ratio:.0%}")
    print(f"  IC rank correlation (IS vs OOS): {result.ic_correlation:+.3f}")
    print(f"  Stable features (OOS IC > min): {result.stable_count}")
    print(f"  Degraded features (IC drop > 50%): {result.degraded_count}")

    verdict = "PASS" if result.ic_correlation > 0.3 and result.degraded_count < len(result.comparisons) / 2 else "FAIL"
    print(f"\n  Verdict: [{verdict}] — ", end="")
    if verdict == "PASS":
        print("Feature rankings are stable out-of-sample. Signals likely genuine.")
    else:
        print("Significant IC degradation. Possible overfitting or regime shift.")

    print(f"\n  {'Feature':<36} {'Role':<10} {'IS IC':>7} {'OOS IC':>7} "
          f"{'Chg%':>6} {'IS Hit':>6} {'OOS Hit':>7} {'IS Edge':>8} {'OOS Edge':>8}")
    print(f"  {'-'*36} {'-'*10} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*8}")

    for c in result.comparisons:
        name = c["name"][:36]
        print(f"  {name:<36} {c['role']:<10} {c['is_ic']:+.3f} {c['oos_ic']:+.3f} "
              f"{c['ic_change_pct']:+.0f}% {c['is_hit_rate']:.1%} {c['oos_hit_rate']:.1%} "
              f"{c['is_edge_bps']:+.1f} {c['oos_edge_bps']:+.1f}")


def print_walk_forward(result: WalkForwardResult, top_n: int = 30):
    """Print walk-forward k-fold validation results."""
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD VALIDATION — {result.symbol} @ {result.timeframe}")
    print(f"{'='*80}")
    print(f"  N={result.total_bars:,}  k={result.n_folds}  "
          f"min_train={result.min_train_bars}  fold_len={result.fold_len}")
    print(f"  Horizons: {result.horizons}")
    total = result.keep_count + result.monitor_count + result.drop_count
    pct = lambda x: f"{x / total:.0%}" if total else "0%"
    print(f"  KEEP:    {result.keep_count:>4}  ({pct(result.keep_count)})  "
          f"sign_consistency >= 0.6 AND |OOS IC| >= min_ic")
    print(f"  MONITOR: {result.monitor_count:>4}  ({pct(result.monitor_count)})  "
          f"consistent direction but weak edge")
    print(f"  DROP:    {result.drop_count:>4}  ({pct(result.drop_count)})  "
          f"sign flips across folds")

    keepers = [f for f in result.features if f.decision == "keep"][:top_n]
    if keepers:
        print(f"\n  TOP KEEPERS (by |OOS IC|)")
        print(f"  {'Feature':<40} {'Vector':<12} {'h':>2} "
              f"{'IS IC':>7} {'OOS IC':>7} {'OOS std':>8} {'Sign':>5}")
        print(f"  {'-'*40} {'-'*12} {'-'*2} "
              f"{'-'*7} {'-'*7} {'-'*8} {'-'*5}")
        for f in keepers:
            print(f"  {f.name[:40]:<40} {f.vector:<12} {f.horizon:>2} "
                  f"{f.is_ic_mean:+.3f} {f.oos_ic_mean:+.3f} "
                  f"{f.oos_ic_std:>8.3f} {f.sign_consistency:>5.0%}")
    else:
        print("\n  (no features passed the keep threshold)")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _json_safe(d: Dict) -> Dict:
    """Convert config dict values to JSON-safe types."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            out[k] = [int(x) if isinstance(x, (np.integer,)) else x for x in v]
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _profile_to_dict(p: FeatureProfile) -> Dict:
    """Convert FeatureProfile to JSON-serializable dict."""
    return {
        "name": p.name,
        "vector": p.vector,
        "ic": {str(k): v for k, v in p.ic.items()},
        "ic_best_horizon": p.ic_best_horizon,
        "ic_best": p.ic_best,
        "ic_ir": p.ic_ir,
        "hit_rate": p.hit_rate,
        "quintile_spread_bps": p.quintile_spread_bps,
        "quintile_monotonic": p.quintile_monotonic,
        "autocorr_1": p.autocorr_1,
        "nan_rate": p.nan_rate,
        "gross_edge_bps": p.gross_edge_bps,
        "net_edge_bps": p.net_edge_bps,
        "conditional": [
            {"condition": c.condition, "ic": c.ic, "n_obs": c.n_obs}
            for c in p.conditional
        ],
        "role": p.role,
        "scalp_score": p.scalp_score,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Scalping Signal Profiler — feature-level alpha analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── profile ──
    p_profile = sub.add_parser("profile", help="Profile all features for scalping viability")
    p_profile.add_argument("--symbol", default="BTC", help="Symbol (default: BTC)")
    p_profile.add_argument("--timeframe", default=None, help="Bar timeframe (default: from config)")
    p_profile.add_argument("--data-dir", default="data/features", help="Parquet data directory")
    p_profile.add_argument("--top", type=int, default=None, help="Show top N features")
    p_profile.add_argument(
        "--forward-test", action="store_true",
        help="Run forward-test validation (walk-forward k-fold by default)"
    )
    p_profile.add_argument(
        "--legacy-split", action="store_true",
        help="Use the legacy single 70/30 split instead of walk-forward k-fold "
             "(only meaningful with --forward-test)"
    )
    p_profile.add_argument(
        "--n-folds", type=int, default=None,
        help="Number of walk-forward folds (default: from config, typ. 5)"
    )
    p_profile.add_argument(
        "--min-train-bars", type=int, default=None,
        help="Bars in fold-0 training window (default: from config, typ. 200)"
    )
    p_profile.add_argument("--config", default="config/pipeline.toml", help="Config file path")
    p_profile.add_argument("-v", "--verbose", action="store_true")

    # ── status ──
    sub.add_parser("status", help="Show profiler process status")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "status":
        state_file = Path("data/profiler_state.json")
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            print(json.dumps(state, indent=2))
        else:
            print("No profiler state found. Run 'profile' first.")
        return

    # Load config
    config = load_profiler_config(args.config)
    if args.timeframe:
        config["timeframe"] = args.timeframe
    if args.top:
        config["top_n"] = args.top

    # Run
    process = ScalpingProfilerProcess(
        config=config,
        data_dir=args.data_dir,
    )

    report, ft_result = process.run(
        symbol=args.symbol,
        forward_test=args.forward_test,
        top_n=args.top,
        legacy_split=args.legacy_split,
        n_folds=args.n_folds,
        min_train_bars=args.min_train_bars,
    )

    print_profile_report(report, top_n=config["top_n"])

    if ft_result is not None:
        if isinstance(ft_result, WalkForwardResult):
            print_walk_forward(ft_result, top_n=config["top_n"])
        else:
            print_forward_test(ft_result)

    print(f"\nReport saved to: {process.report_dir}/")


if __name__ == "__main__":
    main()
