#!/usr/bin/env python3
"""
Scalp Edge Scanner — find rare-but-repeatable feature tail configurations
that precede profitable 5-min moves.

Five analyses, each building on the prior:
  1. Tail return profiling     — per-feature distributional tail edge
  2. Conjunction scan          — pairwise tail combinations
  3. Temporal characterization — holding period, clustering, cooldown
  4. Stability assessment      — first-half vs second-half validation
  5. Archetype classification  — map setups to strategy classes

Usage:
    python scripts/scalp_edge_scanner.py scan --symbol BTC
    python scripts/scalp_edge_scanner.py scan --symbol BTC --data-dir ./data/features
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

from cluster_pipeline.config import FEATURE_VECTORS
from cluster_pipeline.loader import load_parquet, filter_symbol
from cluster_pipeline.preprocess import aggregate_bars

from alpha.screener import benjamini_hochberg


# ---------------------------------------------------------------------------
# JSON encoder (handles numpy types)
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "timeframe": "5min",
    "cost_bps": 3.5,
    "tail_percentiles": [1, 5, 10, 90, 95, 99],
    "tail_threshold": 5,
    "top_features": 40,
    "conjunction_features": 15,
    "min_observations": 100,
    "min_tail_obs": 10,
    "significance_alpha": 0.05,
    "forward_horizon_bars": 1,
    "symbols": ["BTC"],
}


def load_scanner_config(toml_path: Optional[str] = None) -> dict:
    """Load [scanner] section from pipeline.toml, merged with defaults."""
    cfg = dict(_DEFAULT_CONFIG)
    if toml_path is None:
        toml_path = str(ROOT / "config" / "pipeline.toml")
    path = Path(toml_path)
    if not path.exists():
        return cfg
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return cfg
    with open(path, "rb") as f:
        data = tomllib.load(f)
    scanner = data.get("scanner", {})
    for k, v in scanner.items():
        cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TailStats:
    mean: float
    median: float
    std: float
    skew: float
    sharpe: float
    win_rate: float
    frequency: float
    n_obs: int
    t_stat: float
    p_value: float
    direction: str  # "bullish" / "bearish" / "neutral"


@dataclass
class FeatureTailProfile:
    name: str
    vector: str
    percentile_thresholds: Dict[int, float]
    lower_tail: Optional[TailStats]
    upper_tail: Optional[TailStats]
    is_significant: bool
    edge_bps: float
    p_adjusted: float


@dataclass
class ConjunctionSetup:
    feature_a: str
    feature_b: str
    tail_a: str  # "lower" / "upper"
    tail_b: str  # "lower" / "upper"
    edge_bps: float
    win_rate: float
    sharpe: float
    n_occurrences: int
    p_value: float
    p_adjusted: float
    weighted_sharpe: float
    confirming: bool


@dataclass
class TemporalProfile:
    setup_id: str
    optimal_holding_bars: int
    clustering_coeff: float
    cooldown_bars: float
    regime_edges: Dict[str, float]


@dataclass
class StabilityResult:
    setup_id: str
    edge_first_half: float
    edge_second_half: float
    is_stable: Optional[bool]  # None = insufficient data
    status: str  # "stable" / "fragile" / "insufficient_data"


@dataclass
class StrategyArchetype:
    name: str
    description: str
    entry_logic: str
    exit_logic: str
    matching_setups: List[str]
    confidence: float


@dataclass
class ScanReport:
    symbol: str
    timeframe: str
    n_bars: int
    timestamp: str
    tail_profiles: List[FeatureTailProfile]
    conjunctions: List[ConjunctionSetup]
    temporal: List[TemporalProfile]
    stability: List[StabilityResult]
    archetypes: List[StrategyArchetype]
    config: dict
    warnings: List[str]


# ---------------------------------------------------------------------------
# Archetype definitions
# ---------------------------------------------------------------------------

ARCHETYPES = {
    "imbalance_reversion": {
        "name": "Imbalance Reversion",
        "description": "Fade extreme order book imbalance expecting mean-reversion",
        "entry_logic": "Enter counter-imbalance when OB imbalance > 95th pctile",
        "exit_logic": "Exit on imbalance normalization or 5min timeout",
        "vectors": {"orderflow", "imbalance"},
    },
    "flow_momentum": {
        "name": "Flow Momentum",
        "description": "Follow heavy one-sided aggressor flow",
        "entry_logic": "Enter in aggressor direction when flow intensity > 95th",
        "exit_logic": "Exit on flow reversal or 10min timeout",
        "vectors": {"flow"},
    },
    "toxicity_alert": {
        "name": "Toxicity Alert",
        "description": "Follow informed flow when VPIN spikes",
        "entry_logic": "Enter in informed-flow direction when VPIN > 95th",
        "exit_logic": "Exit on VPIN decay or 5min timeout",
        "vectors": {"toxicity"},
    },
    "entropy_breakout": {
        "name": "Entropy Breakout",
        "description": "Ride directional continuation when entropy collapses with trend",
        "entry_logic": "Enter in trend direction when entropy < 5th and trend aligned",
        "exit_logic": "Exit on entropy rise or 10min timeout",
        "vectors": {"entropy"},
    },
    "liquidity_drain": {
        "name": "Liquidity Drain",
        "description": "Anticipate impact event when Kyle lambda spikes",
        "entry_logic": "Enter in expected impact direction when Kyle lambda > 95th",
        "exit_logic": "Exit when lambda normalizes",
        "vectors": {"illiquidity"},
    },
    "funding_squeeze": {
        "name": "Funding Squeeze",
        "description": "Trade liquidation cascades from extreme funding + OI",
        "entry_logic": "Enter counter-funding when funding extreme + OI diverging",
        "exit_logic": "Exit on funding normalization",
        "vectors": {"context"},
    },
    "trend_persistence": {
        "name": "Trend Persistence",
        "description": "Continue with strong trend momentum signals",
        "entry_logic": "Enter in trend direction when momentum + monotonicity extreme",
        "exit_logic": "Exit on trend reversal signal or 10min timeout",
        "vectors": {"trend"},
    },
    "volatility_expansion": {
        "name": "Volatility Expansion",
        "description": "Trade directional moves during volatility expansion",
        "entry_logic": "Enter when realized vol spikes with directional bias",
        "exit_logic": "Exit on vol contraction or 5min timeout",
        "vectors": {"volatility"},
    },
}


# ---------------------------------------------------------------------------
# Core Scanner
# ---------------------------------------------------------------------------

class ScalpEdgeScanner:
    """Scans feature tails for exploitable 5-min scalping edges."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_scanner_config()
        self.timeframe = cfg["timeframe"]
        self.cost_bps = cfg["cost_bps"]
        self.tail_percentiles = cfg["tail_percentiles"]
        self.tail_threshold = cfg["tail_threshold"]
        self.top_features = cfg["top_features"]
        self.conjunction_features = cfg["conjunction_features"]
        self.min_observations = cfg["min_observations"]
        self.min_tail_obs = cfg["min_tail_obs"]
        self.significance_alpha = cfg["significance_alpha"]
        self.forward_horizon = cfg["forward_horizon_bars"]
        self.config = cfg

    # ── Data Loading ─────────────────────────────────────────────────

    def load_and_aggregate(
        self, data_dir: str, symbol: str
    ) -> pd.DataFrame:
        """Load parquet data and aggregate to bars."""
        df = load_parquet(data_dir)
        df = filter_symbol(df, symbol)
        if len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} rows")
        bars = aggregate_bars(df, timeframe=self.timeframe)
        return bars

    def compute_forward_returns(
        self, bars: pd.DataFrame, horizon: Optional[int] = None
    ) -> np.ndarray:
        """Compute log forward returns at the given horizon (in bars)."""
        h = horizon or self.forward_horizon
        price_col = self._find_price_col(bars)
        prices = bars[price_col].values.astype(float)
        n = len(prices)
        fwd = np.full(n, np.nan)
        valid = (prices[:-h] > 0) & (prices[h:] > 0)
        fwd[:n - h][valid] = np.log(prices[h:][valid] / prices[:n - h][valid])
        return fwd

    def select_features(self, bars: pd.DataFrame, n: int = 40) -> List[str]:
        """Select top-N informative numeric features by variance."""
        exclude = {"timestamp", "symbol", "date", "hour"}
        cols = [
            c for c in bars.columns
            if c not in exclude
            and bars[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
            and not c.startswith("raw_midprice")
            and not c.startswith("raw_microprice")
        ]
        variances = {}
        for c in cols:
            v = bars[c].var()
            nan_rate = bars[c].isna().mean()
            if np.isfinite(v) and v > 1e-12 and nan_rate < 0.5:
                variances[c] = v
        ranked = sorted(variances, key=variances.get, reverse=True)
        return ranked[:n]

    # ── Analysis 1: Tail Return Profiling ────────────────────────────

    def tail_return_profile(
        self,
        bars: pd.DataFrame,
        fwd_returns: np.ndarray,
        feature_cols: List[str],
    ) -> List[FeatureTailProfile]:
        """Profile each feature's tail regions for directional edge."""
        profiles = []
        all_p_values = []
        all_indices = []  # (profile_index, "lower"/"upper")

        for col in feature_cols:
            values = bars[col].values.astype(float)
            valid = np.isfinite(values) & np.isfinite(fwd_returns)
            if valid.sum() < self.min_observations:
                continue

            pctiles = {}
            for p in self.tail_percentiles:
                pctiles[p] = float(np.nanpercentile(values, p))

            lo_thresh = pctiles.get(self.tail_threshold, pctiles.get(5))
            hi_thresh = pctiles.get(100 - self.tail_threshold, pctiles.get(95))

            lo_mask = valid & (values <= lo_thresh)
            hi_mask = valid & (values >= hi_thresh)

            lo_stats = self._compute_tail_stats(fwd_returns, lo_mask)
            hi_stats = self._compute_tail_stats(fwd_returns, hi_mask)

            best_edge = 0.0
            best_p = 1.0
            for ts, label in [(lo_stats, "lower"), (hi_stats, "upper")]:
                if ts is not None:
                    all_p_values.append(ts.p_value)
                    all_indices.append((len(profiles), label))
                    edge = abs(ts.mean) * 1e4  # to bps
                    if edge > best_edge:
                        best_edge = edge
                        best_p = ts.p_value

            profiles.append(FeatureTailProfile(
                name=col,
                vector=self._detect_vector(col),
                percentile_thresholds=pctiles,
                lower_tail=lo_stats,
                upper_tail=hi_stats,
                is_significant=False,
                edge_bps=best_edge,
                p_adjusted=1.0,
            ))

        # BH-FDR correction
        if all_p_values:
            p_arr = np.array(all_p_values)
            p_adj = benjamini_hochberg(p_arr, alpha=self.significance_alpha)

            for i, (prof_idx, tail_label) in enumerate(all_indices):
                adj = p_adj[i] if np.isfinite(p_adj[i]) else 1.0
                prof = profiles[prof_idx]
                ts = prof.lower_tail if tail_label == "lower" else prof.upper_tail
                if ts is not None and adj < self.significance_alpha and ts.n_obs >= self.min_tail_obs:
                    prof.is_significant = True
                    if adj < prof.p_adjusted:
                        prof.p_adjusted = adj

        # Sort by edge descending
        profiles.sort(key=lambda p: p.edge_bps, reverse=True)
        return profiles

    def _compute_tail_stats(
        self, fwd_returns: np.ndarray, mask: np.ndarray
    ) -> Optional[TailStats]:
        """Compute return statistics for bars matching the tail mask."""
        n_obs = int(mask.sum())
        if n_obs < 3:
            return None

        rets = fwd_returns[mask]
        rets = rets[np.isfinite(rets)]
        n_obs = len(rets)
        if n_obs < 3:
            return None

        mean = float(np.mean(rets))
        median = float(np.median(rets))
        std = float(np.std(rets, ddof=1)) if n_obs > 1 else 1e-10
        skew_val = float(stats.skew(rets)) if n_obs > 2 else 0.0
        sharpe = mean / max(std, 1e-10)

        cost_threshold = self.cost_bps * 1e-4
        if mean > 0:
            wins = np.sum(rets > cost_threshold)
        elif mean < 0:
            wins = np.sum(rets < -cost_threshold)
        else:
            wins = 0
        win_rate = float(wins / n_obs)

        frequency = float(mask.sum() / len(mask))

        t_result = stats.ttest_1samp(rets, 0.0)
        t_stat = float(t_result.statistic) if np.isfinite(t_result.statistic) else 0.0
        p_value = float(t_result.pvalue) if np.isfinite(t_result.pvalue) else 1.0

        if mean > 0 and p_value < 0.1:
            direction = "bullish"
        elif mean < 0 and p_value < 0.1:
            direction = "bearish"
        else:
            direction = "neutral"

        return TailStats(
            mean=mean, median=median, std=std, skew=skew_val,
            sharpe=sharpe, win_rate=win_rate, frequency=frequency,
            n_obs=n_obs, t_stat=t_stat, p_value=p_value,
            direction=direction,
        )

    # ── Analysis 2: Conjunction Scan ─────────────────────────────────

    def conjunction_scan(
        self,
        bars: pd.DataFrame,
        fwd_returns: np.ndarray,
        significant_profiles: List[FeatureTailProfile],
    ) -> List[ConjunctionSetup]:
        """Test pairwise tail conjunctions for combined edge."""
        top = significant_profiles[:self.conjunction_features]
        if len(top) < 2:
            return []

        # Pre-compute tail masks
        tail_masks = {}
        tail_dirs = {}
        for prof in top:
            values = bars[prof.name].values.astype(float)
            lo = prof.percentile_thresholds.get(self.tail_threshold, np.nan)
            hi = prof.percentile_thresholds.get(100 - self.tail_threshold, np.nan)
            valid = np.isfinite(values) & np.isfinite(fwd_returns)
            tail_masks[(prof.name, "lower")] = valid & (values <= lo)
            tail_masks[(prof.name, "upper")] = valid & (values >= hi)
            # Direction from Analysis 1
            for label, ts in [("lower", prof.lower_tail), ("upper", prof.upper_tail)]:
                tail_dirs[(prof.name, label)] = ts.direction if ts else "neutral"

        setups = []
        all_p = []

        for (a, b) in combinations(top, 2):
            for tail_a in ["lower", "upper"]:
                for tail_b in ["lower", "upper"]:
                    mask = tail_masks[(a.name, tail_a)] & tail_masks[(b.name, tail_b)]
                    n_occ = int(mask.sum())
                    if n_occ < self.min_tail_obs:
                        continue

                    rets = fwd_returns[mask]
                    rets = rets[np.isfinite(rets)]
                    if len(rets) < 3:
                        continue

                    mean = float(np.mean(rets))
                    std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 1e-10
                    sharpe = mean / max(std, 1e-10)
                    edge = abs(mean) * 1e4

                    cost_thresh = self.cost_bps * 1e-4
                    if mean > 0:
                        wr = float(np.sum(rets > cost_thresh) / len(rets))
                    elif mean < 0:
                        wr = float(np.sum(rets < -cost_thresh) / len(rets))
                    else:
                        wr = 0.0

                    t_res = stats.ttest_1samp(rets, 0.0)
                    p_val = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else 1.0

                    dir_a = tail_dirs.get((a.name, tail_a), "neutral")
                    dir_b = tail_dirs.get((b.name, tail_b), "neutral")
                    confirming = (
                        dir_a == dir_b
                        and dir_a != "neutral"
                    )

                    ws = sharpe * np.sqrt(n_occ)

                    setup = ConjunctionSetup(
                        feature_a=a.name, feature_b=b.name,
                        tail_a=tail_a, tail_b=tail_b,
                        edge_bps=edge, win_rate=wr, sharpe=sharpe,
                        n_occurrences=n_occ, p_value=p_val, p_adjusted=1.0,
                        weighted_sharpe=float(ws), confirming=confirming,
                    )
                    setups.append(setup)
                    all_p.append(p_val)

        # FDR correction
        if all_p:
            p_adj = benjamini_hochberg(np.array(all_p), alpha=self.significance_alpha)
            for i, s in enumerate(setups):
                s.p_adjusted = float(p_adj[i]) if np.isfinite(p_adj[i]) else 1.0

        # Filter significant and sort by weighted Sharpe
        sig = [s for s in setups if s.p_adjusted < self.significance_alpha]
        sig.sort(key=lambda s: abs(s.weighted_sharpe), reverse=True)
        return sig

    # ── Analysis 3: Temporal Characterization ────────────────────────

    def temporal_characterize(
        self,
        bars: pd.DataFrame,
        fwd_returns: np.ndarray,
        setups: List[Tuple[str, np.ndarray]],
    ) -> List[TemporalProfile]:
        """Characterize temporal patterns of significant setups."""
        price_col = self._find_price_col(bars)
        prices = bars[price_col].values.astype(float)
        n = len(prices)
        results = []

        for setup_id, trigger_mask in setups:
            trigger_idx = np.where(trigger_mask)[0]
            if len(trigger_idx) < 3:
                results.append(TemporalProfile(
                    setup_id=setup_id, optimal_holding_bars=1,
                    clustering_coeff=0.0, cooldown_bars=0.0,
                    regime_edges={},
                ))
                continue

            # Optimal holding period
            horizons = [1, 2, 3, 5, 10]
            horizon_edges = {}
            for h in horizons:
                valid_idx = trigger_idx[trigger_idx + h < n]
                if len(valid_idx) < 2:
                    continue
                rets = np.log(prices[valid_idx + h] / prices[valid_idx])
                rets = rets[np.isfinite(rets)]
                if len(rets) > 0:
                    horizon_edges[h] = float(np.mean(rets))
            optimal_h = max(horizon_edges, key=lambda h: abs(horizon_edges[h])) if horizon_edges else 1

            # Clustering coefficient (CV of inter-trigger intervals)
            intervals = np.diff(trigger_idx).astype(float)
            if len(intervals) > 1:
                cv = float(np.std(intervals) / max(np.mean(intervals), 1e-10))
            else:
                cv = 0.0

            # Cooldown (median interval)
            cooldown = float(np.median(intervals)) if len(intervals) > 0 else 0.0

            # Regime dependency (quartile split)
            q_size = n // 4
            regime_edges = {}
            for qi in range(4):
                start, end = qi * q_size, (qi + 1) * q_size if qi < 3 else n
                q_mask = trigger_mask.copy()
                q_mask[:start] = False
                q_mask[end:] = False
                q_idx = np.where(q_mask)[0]
                if len(q_idx) < 2:
                    regime_edges[f"Q{qi + 1}"] = 0.0
                    continue
                h = min(optimal_h, n - 1 - q_idx.max()) if len(q_idx) > 0 else 1
                h = max(h, 1)
                valid_qi = q_idx[q_idx + h < n]
                if len(valid_qi) > 0:
                    rets = np.log(prices[valid_qi + h] / prices[valid_qi])
                    rets = rets[np.isfinite(rets)]
                    regime_edges[f"Q{qi + 1}"] = float(np.mean(rets)) * 1e4 if len(rets) > 0 else 0.0
                else:
                    regime_edges[f"Q{qi + 1}"] = 0.0

            results.append(TemporalProfile(
                setup_id=setup_id,
                optimal_holding_bars=optimal_h,
                clustering_coeff=cv,
                cooldown_bars=cooldown,
                regime_edges=regime_edges,
            ))

        return results

    # ── Analysis 4: Stability Assessment ─────────────────────────────

    def stability_assess(
        self,
        fwd_returns: np.ndarray,
        setups: List[Tuple[str, np.ndarray]],
    ) -> List[StabilityResult]:
        """Assess stability by comparing edge in first vs second half."""
        n = len(fwd_returns)
        mid = n // 2
        results = []

        for setup_id, mask in setups:
            mask_1h = mask.copy()
            mask_1h[mid:] = False
            mask_2h = mask.copy()
            mask_2h[:mid] = False

            n1 = int(mask_1h.sum())
            n2 = int(mask_2h.sum())

            if n1 < self.min_tail_obs or n2 < self.min_tail_obs:
                results.append(StabilityResult(
                    setup_id=setup_id,
                    edge_first_half=0.0, edge_second_half=0.0,
                    is_stable=None, status="insufficient_data",
                ))
                continue

            rets_1 = fwd_returns[mask_1h]
            rets_1 = rets_1[np.isfinite(rets_1)]
            rets_2 = fwd_returns[mask_2h]
            rets_2 = rets_2[np.isfinite(rets_2)]

            e1 = float(np.mean(rets_1)) * 1e4 if len(rets_1) > 0 else 0.0
            e2 = float(np.mean(rets_2)) * 1e4 if len(rets_2) > 0 else 0.0

            same_sign = (e1 > 0 and e2 > 0) or (e1 < 0 and e2 < 0)
            both_material = abs(e1) > 0.5 and abs(e2) > 0.5
            stable = same_sign and both_material

            results.append(StabilityResult(
                setup_id=setup_id,
                edge_first_half=e1, edge_second_half=e2,
                is_stable=stable,
                status="stable" if stable else "fragile",
            ))

        return results

    # ── Analysis 5: Archetype Classification ─────────────────────────

    def classify_archetypes(
        self,
        profiles: List[FeatureTailProfile],
        conjunctions: List[ConjunctionSetup],
        stability: List[StabilityResult],
    ) -> List[StrategyArchetype]:
        """Map significant setups to strategy archetypes."""
        # Collect all significant setup features and their vectors
        setup_vectors = {}  # setup_id -> list of vectors

        for p in profiles:
            if p.is_significant:
                sid = f"tail:{p.name}"
                setup_vectors[sid] = [p.vector]

        for c in conjunctions:
            sid = f"conj:{c.feature_a}+{c.feature_b}"
            v_a = self._detect_vector(c.feature_a)
            v_b = self._detect_vector(c.feature_b)
            setup_vectors[sid] = [v_a, v_b]

        # Stability lookup
        stable_ids = {s.setup_id for s in stability if s.is_stable}

        # Match setups to archetypes
        archetype_results = []
        for arch_key, arch_def in ARCHETYPES.items():
            matching = []
            for sid, vectors in setup_vectors.items():
                if any(v in arch_def["vectors"] for v in vectors):
                    matching.append(sid)

            if not matching:
                continue

            stable_count = sum(1 for s in matching if s in stable_ids)
            confidence = stable_count / max(len(matching), 1)

            archetype_results.append(StrategyArchetype(
                name=arch_def["name"],
                description=arch_def["description"],
                entry_logic=arch_def["entry_logic"],
                exit_logic=arch_def["exit_logic"],
                matching_setups=matching,
                confidence=confidence,
            ))

        archetype_results.sort(key=lambda a: (len(a.matching_setups), a.confidence), reverse=True)
        return archetype_results

    # ── Full Scan Pipeline ───────────────────────────────────────────

    def scan(
        self, data_dir: str, symbol: str
    ) -> ScanReport:
        """Run the full 5-analysis scan pipeline."""
        warnings = []

        # Load and aggregate
        bars = self.load_and_aggregate(data_dir, symbol)
        n_bars = len(bars)
        if n_bars < self.min_observations:
            raise ValueError(f"Only {n_bars} bars, need >= {self.min_observations}")
        if n_bars < 200:
            warnings.append(f"LOW_DATA: only {n_bars} bars. Results have limited statistical power.")

        # Forward returns
        fwd = self.compute_forward_returns(bars)

        # Feature selection
        feature_cols = self.select_features(bars, n=self.top_features)
        if len(feature_cols) < 5:
            warnings.append(f"Only {len(feature_cols)} features passed variance filter")

        # Analysis 1: Tail return profiling
        tail_profiles = self.tail_return_profile(bars, fwd, feature_cols)
        sig_profiles = [p for p in tail_profiles if p.is_significant]
        if not sig_profiles:
            warnings.append("No features passed FDR significance. Try a lower tail_threshold or collect more data.")

        # Analysis 2: Conjunction scan
        conjunctions = self.conjunction_scan(bars, fwd, sig_profiles)

        # Build trigger masks for temporal + stability analysis
        setup_masks = []
        for p in sig_profiles[:15]:
            values = bars[p.name].values.astype(float)
            lo = p.percentile_thresholds.get(self.tail_threshold, np.nan)
            hi = p.percentile_thresholds.get(100 - self.tail_threshold, np.nan)
            best_tail = None
            if p.upper_tail and p.lower_tail:
                if abs(p.upper_tail.mean) >= abs(p.lower_tail.mean):
                    best_tail = "upper"
                else:
                    best_tail = "lower"
            elif p.upper_tail:
                best_tail = "upper"
            elif p.lower_tail:
                best_tail = "lower"
            if best_tail == "upper":
                mask = np.isfinite(values) & (values >= hi)
            elif best_tail == "lower":
                mask = np.isfinite(values) & (values <= lo)
            else:
                continue
            setup_masks.append((f"tail:{p.name}", mask))

        for c in conjunctions[:10]:
            v_a = bars[c.feature_a].values.astype(float)
            v_b = bars[c.feature_b].values.astype(float)
            pa = [p for p in sig_profiles if p.name == c.feature_a]
            pb = [p for p in sig_profiles if p.name == c.feature_b]
            if not pa or not pb:
                continue
            lo_a = pa[0].percentile_thresholds.get(self.tail_threshold, np.nan)
            hi_a = pa[0].percentile_thresholds.get(100 - self.tail_threshold, np.nan)
            lo_b = pb[0].percentile_thresholds.get(self.tail_threshold, np.nan)
            hi_b = pb[0].percentile_thresholds.get(100 - self.tail_threshold, np.nan)
            mask_a = (v_a <= lo_a) if c.tail_a == "lower" else (v_a >= hi_a)
            mask_b = (v_b <= lo_b) if c.tail_b == "lower" else (v_b >= hi_b)
            mask = np.isfinite(v_a) & np.isfinite(v_b) & mask_a & mask_b
            sid = f"conj:{c.feature_a}+{c.feature_b}"
            setup_masks.append((sid, mask))

        # Analysis 3: Temporal characterization
        temporal = self.temporal_characterize(bars, fwd, setup_masks)

        # Analysis 4: Stability assessment
        stability = self.stability_assess(fwd, setup_masks)

        # Analysis 5: Archetype classification
        archetypes = self.classify_archetypes(sig_profiles, conjunctions, stability)

        return ScanReport(
            symbol=symbol,
            timeframe=self.timeframe,
            n_bars=n_bars,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tail_profiles=tail_profiles,
            conjunctions=conjunctions,
            temporal=temporal,
            stability=stability,
            archetypes=archetypes,
            config=self.config,
            warnings=warnings,
        )

    # ── Helpers ──────────────────────────────────────────────────────

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
        raise ValueError(f"No price column found. Columns: {list(bars.columns)[:20]}")

    def _detect_vector(self, col_name: str) -> str:
        base = col_name
        for suffix in ["_mean", "_std", "_last", "_slope", "_open", "_high",
                        "_low", "_close", "_sum"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        for vname, vspec in FEATURE_VECTORS.items():
            if base in vspec.get("columns", []):
                return vname
            for prefix in vspec.get("prefixes", []):
                if base.startswith(prefix):
                    return vname
        return "unknown"


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def save_json_report(report: ScanReport, output_dir: str) -> str:
    """Save full scan report as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    path = os.path.join(output_dir, f"scan_{report.symbol}_{date_str}.json")
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2, cls=_NumpyEncoder)
    return path


def save_md_report(report: ScanReport, output_dir: str) -> str:
    """Save human-readable Markdown summary."""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    path = os.path.join(output_dir, f"scan_{report.symbol}_{date_str}.md")

    sig_tails = [p for p in report.tail_profiles if p.is_significant]
    stable_count = sum(1 for s in report.stability if s.status == "stable")

    lines = [
        f"# Scalp Edge Scan: {report.symbol} ({report.timeframe})",
        f"",
        f"**Generated:** {report.timestamp}",
        f"**Bars:** {report.n_bars} | **Significant features:** {len(sig_tails)} "
        f"| **Conjunctions:** {len(report.conjunctions)} "
        f"| **Stable setups:** {stable_count}",
        f"",
    ]

    # Warnings
    if report.warnings:
        lines.append("## Warnings")
        for w in report.warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Top single-feature tail setups
    lines.append("## Top Feature Tail Setups")
    lines.append("")
    lines.append("| Rank | Feature | Vector | Edge (bps) | Win Rate | Direction | Sharpe | Freq | p_adj | Stable |")
    lines.append("|------|---------|--------|-----------|----------|-----------|--------|------|-------|--------|")

    for i, p in enumerate(sig_tails[:15]):
        best = p.upper_tail if (p.upper_tail and (not p.lower_tail or abs(p.upper_tail.mean) >= abs(p.lower_tail.mean))) else p.lower_tail
        if not best:
            continue
        sid = f"tail:{p.name}"
        stab = next((s for s in report.stability if s.setup_id == sid), None)
        stab_str = stab.status if stab else "?"
        lines.append(
            f"| {i + 1} | `{p.name}` | {p.vector} | {p.edge_bps:.1f} | "
            f"{best.win_rate:.1%} | {best.direction} | {best.sharpe:.2f} | "
            f"{best.frequency:.1%} | {p.p_adjusted:.3f} | {stab_str} |"
        )
    lines.append("")

    # Conjunctions
    if report.conjunctions:
        lines.append("## Top Conjunction Setups")
        lines.append("")
        lines.append("| Rank | Features | Tails | Edge (bps) | Win Rate | Sharpe | N | Confirming | p_adj |")
        lines.append("|------|----------|-------|-----------|----------|--------|---|------------|-------|")
        for i, c in enumerate(report.conjunctions[:10]):
            lines.append(
                f"| {i + 1} | `{c.feature_a}` + `{c.feature_b}` | "
                f"{c.tail_a}/{c.tail_b} | {c.edge_bps:.1f} | {c.win_rate:.1%} | "
                f"{c.sharpe:.2f} | {c.n_occurrences} | {'yes' if c.confirming else 'no'} | "
                f"{c.p_adjusted:.3f} |"
            )
        lines.append("")

    # Temporal profiles
    if report.temporal:
        lines.append("## Temporal Characteristics")
        lines.append("")
        lines.append("| Setup | Optimal Hold | Clustering CV | Cooldown (bars) | Regime Dep. |")
        lines.append("|-------|-------------|---------------|-----------------|-------------|")
        for t in report.temporal[:15]:
            edges = [f"{k}={v:.1f}" for k, v in t.regime_edges.items()]
            regime_str = ", ".join(edges) if edges else "-"
            lines.append(
                f"| `{t.setup_id}` | {t.optimal_holding_bars} bars | "
                f"{t.clustering_coeff:.2f} | {t.cooldown_bars:.0f} | {regime_str} |"
            )
        lines.append("")

    # Archetypes
    if report.archetypes:
        lines.append("## Strategy Archetypes")
        lines.append("")
        for a in report.archetypes:
            lines.append(f"### {a.name} (confidence: {a.confidence:.0%}, {len(a.matching_setups)} setups)")
            lines.append(f"")
            lines.append(f"*{a.description}*")
            lines.append(f"")
            lines.append(f"- **Entry:** {a.entry_logic}")
            lines.append(f"- **Exit:** {a.exit_logic}")
            lines.append(f"- **Matching setups:** {', '.join(a.matching_setups[:5])}")
            lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="scalp_edge_scanner",
        description="Scan feature tails for exploitable 5-min scalping edges",
    )
    sub = parser.add_subparsers(dest="command")

    scan_p = sub.add_parser("scan", help="Run the full scan pipeline")
    scan_p.add_argument("--symbol", default="BTC")
    scan_p.add_argument("--data-dir", default=str(ROOT / "data" / "features"))
    scan_p.add_argument("--tail", type=int, default=None, help="Tail percentile threshold")
    scan_p.add_argument("--config", default=None, help="Path to pipeline.toml")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    if args.command == "scan":
        cfg = load_scanner_config(args.config)
        if args.tail is not None:
            cfg["tail_threshold"] = args.tail
        scanner = ScalpEdgeScanner(cfg)

        print(f"\n  Scanning {args.symbol} for scalp edges...\n")
        report = scanner.scan(args.data_dir, args.symbol)

        out_dir = str(ROOT / "reports" / "scalp_scanner")
        json_path = save_json_report(report, out_dir)
        md_path = save_md_report(report, out_dir)

        sig = [p for p in report.tail_profiles if p.is_significant]
        print(f"  Bars:           {report.n_bars}")
        print(f"  Significant:    {len(sig)} features, {len(report.conjunctions)} conjunctions")
        print(f"  Archetypes:     {len(report.archetypes)}")
        print(f"  Stable setups:  {sum(1 for s in report.stability if s.status == 'stable')}")
        print(f"\n  Reports:")
        print(f"    {json_path}")
        print(f"    {md_path}")
        for w in report.warnings:
            print(f"\n  WARNING: {w}")
        print()


if __name__ == "__main__":
    main()
