#!/usr/bin/env python3
"""
Spannung Regime Screener — systematic search for profitable trading conditions.

Tests 17 microstructure features as regime conditions, measures conditional IC
of L1 book imbalance (raw and ultra-low bandpass-filtered) within each regime,
and finds Pareto-optimal multi-factor combinations (highest IC × enough trades).

Usage:
    python scripts/spannung_regime_screener.py --data-dir data/features/2026-05-12
    python scripts/spannung_regime_screener.py --data-dir data/features/2026-05-12 --symbol BTC
    nat spannung regime --data data/features/2026-05-12

Output:
    reports/spannung/regime_screen_{SYM}.json
    printed: single-factor ranking, Pareto frontier, persistence stats
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
warnings.filterwarnings("ignore")

from cluster_pipeline.loader import load_parquet
from spannung_spectral import bandpass_filter

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("regime_screener")

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS = ["BTC", "ETH", "SOL"]

REGIME_FEATURES = {
    "entropy": [
        "ent_tick_1m", "ent_tick_5s", "ent_tick_30s",
        "ent_permutation_returns_16", "ent_spread_dispersion", "ent_book_shape",
    ],
    "illiquidity": ["illiq_kyle_100", "illiq_composite", "illiq_amihud_100"],
    "toxicity": ["toxic_vpin_50", "toxic_adverse_selection", "toxic_index"],
    "volatility": ["vol_returns_1m", "vol_returns_5m", "vol_ratio_short_long"],
    "derived": [
        "derived_regime_type_score", "derived_regime_confidence",
        "derived_informed_trend_score",
    ],
}

ALL_REGIME_FEATURES = [c for cols in REGIME_FEATURES.values() for c in cols]
FEATURE_CATEGORY = {c: cat for cat, cols in REGIME_FEATURES.items() for c in cols}

NEEDED = (
    ["timestamp_ns", "symbol", "raw_midprice", "imbalance_qty_l1"]
    + ALL_REGIME_FEATURES
)

FS = 10.0
PERCENTILES = [20, 40, 60, 80]
FWD_HORIZONS = {"1s": 10, "5s": 50}
ULTRALOW_BAND = (0.005, 0.1)

# IC params
IC_WINDOW = 3000
IC_MIN_OBS = 100
MIN_REGIME_OBS = 500

# Phase 2 params
TOP_K_SINGLE = 10
CORRELATION_THRESHOLD = 0.7


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SingleFactorResult:
    feature: str
    category: str
    percentile: int
    direction: str         # "above" or "below"
    threshold_value: float
    n_obs: int
    coverage: float
    ic_raw_1s: float
    ic_raw_5s: float
    ic_ir_raw_5s: float
    ic_filt_1s: float
    ic_filt_5s: float
    ic_ir_filt_5s: float
    ic_improve_raw_5s: float
    ic_improve_filt_5s: float
    label: str             # e.g. "ent_tick_1m<P20"


@dataclass
class MultiFactorResult:
    labels: List[str]
    n_factors: int
    n_obs: int
    coverage: float
    ic_raw_5s: float
    ic_ir_raw_5s: float
    ic_filt_5s: float
    ic_ir_filt_5s: float
    ic_improve_raw_5s: float
    ic_improve_filt_5s: float
    is_pareto: bool


@dataclass
class PersistenceResult:
    label: str
    mean_duration_s: float
    median_duration_s: float
    longest_s: float
    n_entries: int
    entries_per_min: float
    frac_above_5s: float    # fraction of episodes > 5s
    frac_above_30s: float   # fraction of episodes > 30s


@dataclass
class ScreenerResult:
    timestamp: str
    data_dir: str
    symbol: str
    n_rows: int
    duration_hours: float
    baseline_ic_raw_1s: float
    baseline_ic_raw_5s: float
    baseline_ic_filt_1s: float
    baseline_ic_filt_5s: float
    single_factors: List[SingleFactorResult]
    multi_factors: List[MultiFactorResult]
    persistence: List[PersistenceResult]
    assessment: str


# ── Core functions ───────────────────────────────────────────────────────────

def compute_forward_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Forward log return: log(p(t+h) / p(t)), NaN-padded at tail."""
    n = len(prices)
    fwd = np.full(n, np.nan)
    if horizon >= n:
        return fwd
    fwd[:n - horizon] = np.log(prices[horizon:] / np.clip(prices[:n - horizon], 1e-15, None))
    return fwd


def compute_ic(signal: np.ndarray, returns: np.ndarray) -> Tuple[float, float, float, int]:
    """Rolling non-overlapping Spearman IC. Returns (mean, std, ir, n_windows)."""
    n = min(len(signal), len(returns))
    valid = np.isfinite(signal[:n]) & np.isfinite(returns[:n])
    ic_vals = []
    start = 0
    while start + IC_WINDOW <= n:
        end = start + IC_WINDOW
        m = valid[start:end]
        if m.sum() >= IC_MIN_OBS:
            s = signal[start:end][m]
            r = returns[start:end][m]
            if np.ptp(s) > 1e-15 and np.ptp(r) > 1e-15:
                rho, _ = stats.spearmanr(s, r)
                if np.isfinite(rho):
                    ic_vals.append(float(rho))
        start = end
    if len(ic_vals) < 2:
        return 0.0, 0.0, 0.0, len(ic_vals)
    arr = np.array(ic_vals)
    m, s = float(np.mean(arr)), float(np.std(arr))
    return m, s, m / s if s > 1e-8 else 0.0, len(ic_vals)


def compute_conditional_ic(
    signal: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float, float, int]:
    """Spearman IC on masked subset. Uses rolling windows on contiguous blocks."""
    combined = mask & np.isfinite(signal) & np.isfinite(returns)
    n_valid = combined.sum()
    if n_valid < IC_MIN_OBS:
        return 0.0, 0.0, 0.0, 0

    s = signal[combined]
    r = returns[combined]
    if np.ptp(s) < 1e-15 or np.ptp(r) < 1e-15:
        return 0.0, 0.0, 0.0, 0

    # If enough data for rolling windows, use them for IC_IR
    window = min(IC_WINDOW, max(n_valid // 4, IC_MIN_OBS))
    ic_vals = []
    start = 0
    while start + window <= n_valid:
        end = start + window
        sw = s[start:end]
        rw = r[start:end]
        if np.ptp(sw) > 1e-15 and np.ptp(rw) > 1e-15:
            rho, _ = stats.spearmanr(sw, rw)
            if np.isfinite(rho):
                ic_vals.append(float(rho))
        start = end

    if len(ic_vals) >= 2:
        arr = np.array(ic_vals)
        m, std = float(np.mean(arr)), float(np.std(arr))
        return m, std, m / std if std > 1e-8 else 0.0, len(ic_vals)

    # Fallback: single global IC
    rho, _ = stats.spearmanr(s, r)
    return float(rho) if np.isfinite(rho) else 0.0, 0.0, 0.0, 1


def screen_single_factor(
    df: pd.DataFrame,
    feature: str,
    signal_raw: np.ndarray,
    signal_filt: Optional[np.ndarray],
    fwd_ret: Dict[str, np.ndarray],
    baseline: Dict[str, float],
    n_total: int,
) -> Tuple[List[SingleFactorResult], Dict[str, np.ndarray]]:
    """Screen one feature at all quintile thresholds. Returns results + masks."""
    results = []
    masks = {}
    category = FEATURE_CATEGORY[feature]

    vals = df[feature].values.astype(np.float64)
    valid_vals = vals[np.isfinite(vals)]
    if len(valid_vals) < MIN_REGIME_OBS:
        return results, masks

    thresholds = {p: float(np.percentile(valid_vals, p)) for p in PERCENTILES}

    for pct, thresh in thresholds.items():
        for direction in ("below", "above"):
            if direction == "below":
                mask = np.isfinite(vals) & (vals <= thresh)
                label = f"{feature}<P{pct}"
            else:
                mask = np.isfinite(vals) & (vals >= thresh)
                label = f"{feature}>P{pct}"

            n_obs = int(mask.sum())
            if n_obs < MIN_REGIME_OBS:
                continue

            coverage = n_obs / n_total

            # Raw IC
            ic_r1, _, _, _ = compute_conditional_ic(signal_raw, fwd_ret["1s"], mask)
            ic_r5, ic_r5_std, ir_r5, _ = compute_conditional_ic(signal_raw, fwd_ret["5s"], mask)

            # Filtered IC
            if signal_filt is not None:
                ic_f1, _, _, _ = compute_conditional_ic(signal_filt, fwd_ret["1s"], mask)
                ic_f5, ic_f5_std, ir_f5, _ = compute_conditional_ic(signal_filt, fwd_ret["5s"], mask)
            else:
                ic_f1, ic_f5, ir_f5 = 0.0, 0.0, 0.0

            results.append(SingleFactorResult(
                feature=feature,
                category=category,
                percentile=pct,
                direction=direction,
                threshold_value=round(thresh, 6),
                n_obs=n_obs,
                coverage=round(coverage, 4),
                ic_raw_1s=round(ic_r1, 4),
                ic_raw_5s=round(ic_r5, 4),
                ic_ir_raw_5s=round(ir_r5, 2),
                ic_filt_1s=round(ic_f1, 4),
                ic_filt_5s=round(ic_f5, 4),
                ic_ir_filt_5s=round(ir_f5, 2),
                ic_improve_raw_5s=round(ic_r5 - baseline["ic_raw_5s"], 4),
                ic_improve_filt_5s=round(ic_f5 - baseline["ic_filt_5s"], 4),
                label=label,
            ))
            masks[label] = mask

    return results, masks


def screen_multi_factor(
    top_labels: List[str],
    masks: Dict[str, np.ndarray],
    signal_raw: np.ndarray,
    signal_filt: Optional[np.ndarray],
    fwd_ret: Dict[str, np.ndarray],
    baseline: Dict[str, float],
    n_total: int,
    regime_features_df: pd.DataFrame,
) -> List[MultiFactorResult]:
    """Test 2-way and 3-way combinations of top single factors."""
    results = []

    # Correlation guard: compute pairwise correlation of underlying features
    # Extract feature name from label (e.g. "ent_tick_1m<P20" -> "ent_tick_1m")
    def feat_from_label(lbl):
        return lbl.split("<")[0].split(">")[0]

    # 2-way combinations
    for a, b in combinations(top_labels, 2):
        if a not in masks or b not in masks:
            continue
        mask = masks[a] & masks[b]
        n_obs = int(mask.sum())
        if n_obs < MIN_REGIME_OBS:
            continue

        coverage = n_obs / n_total
        ic_r5, _, ir_r5, _ = compute_conditional_ic(signal_raw, fwd_ret["5s"], mask)

        if signal_filt is not None:
            ic_f5, _, ir_f5, _ = compute_conditional_ic(signal_filt, fwd_ret["5s"], mask)
        else:
            ic_f5, ir_f5 = 0.0, 0.0

        results.append(MultiFactorResult(
            labels=[a, b],
            n_factors=2,
            n_obs=n_obs,
            coverage=round(coverage, 4),
            ic_raw_5s=round(ic_r5, 4),
            ic_ir_raw_5s=round(ir_r5, 2),
            ic_filt_5s=round(ic_f5, 4),
            ic_ir_filt_5s=round(ir_f5, 2),
            ic_improve_raw_5s=round(ic_r5 - baseline["ic_raw_5s"], 4),
            ic_improve_filt_5s=round(ic_f5 - baseline["ic_filt_5s"], 4),
            is_pareto=False,
        ))

    # 3-way: top 5 two-way combos × remaining single factors
    results_2way = sorted(results, key=lambda r: r.ic_filt_5s, reverse=True)[:5]
    for r2 in results_2way:
        for lbl in top_labels:
            if lbl in r2.labels or lbl not in masks:
                continue
            # Correlation guard: skip if underlying feature is redundant
            feat_c = feat_from_label(lbl)
            feats_ab = [feat_from_label(l) for l in r2.labels]
            skip = False
            for f_existing in feats_ab:
                if f_existing in regime_features_df.columns and feat_c in regime_features_df.columns:
                    corr = regime_features_df[[f_existing, feat_c]].corr().iloc[0, 1]
                    if abs(corr) > CORRELATION_THRESHOLD:
                        skip = True
                        break
            if skip:
                continue

            mask = masks[r2.labels[0]] & masks[r2.labels[1]] & masks[lbl]
            n_obs = int(mask.sum())
            if n_obs < MIN_REGIME_OBS:
                continue

            coverage = n_obs / n_total
            ic_r5, _, ir_r5, _ = compute_conditional_ic(signal_raw, fwd_ret["5s"], mask)
            if signal_filt is not None:
                ic_f5, _, ir_f5, _ = compute_conditional_ic(signal_filt, fwd_ret["5s"], mask)
            else:
                ic_f5, ir_f5 = 0.0, 0.0

            results.append(MultiFactorResult(
                labels=r2.labels + [lbl],
                n_factors=3,
                n_obs=n_obs,
                coverage=round(coverage, 4),
                ic_raw_5s=round(ic_r5, 4),
                ic_ir_raw_5s=round(ir_r5, 2),
                ic_filt_5s=round(ic_f5, 4),
                ic_ir_filt_5s=round(ir_f5, 2),
                ic_improve_raw_5s=round(ic_r5 - baseline["ic_raw_5s"], 4),
                ic_improve_filt_5s=round(ic_f5 - baseline["ic_filt_5s"], 4),
                is_pareto=False,
            ))

    return results


def pareto_filter(results: List[MultiFactorResult]) -> List[MultiFactorResult]:
    """Mark Pareto-optimal points: no other has BOTH higher IC AND higher coverage."""
    for i, ri in enumerate(results):
        dominated = False
        for j, rj in enumerate(results):
            if i == j:
                continue
            if rj.ic_filt_5s >= ri.ic_filt_5s and rj.coverage >= ri.coverage:
                if rj.ic_filt_5s > ri.ic_filt_5s or rj.coverage > ri.coverage:
                    dominated = True
                    break
        ri.is_pareto = not dominated
    return results


def measure_persistence(mask: np.ndarray, label: str, duration_hours: float) -> PersistenceResult:
    """Run-length analysis of regime mask."""
    changes = np.diff(mask.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    n_min = min(len(starts), len(ends))
    if n_min == 0:
        return PersistenceResult(
            label=label, mean_duration_s=0, median_duration_s=0,
            longest_s=0, n_entries=0, entries_per_min=0,
            frac_above_5s=0, frac_above_30s=0,
        )

    durations = (ends[:n_min] - starts[:n_min]).astype(float)
    dur_s = durations * 0.1  # ticks to seconds

    total_minutes = duration_hours * 60
    return PersistenceResult(
        label=label,
        mean_duration_s=round(float(np.mean(dur_s)), 2),
        median_duration_s=round(float(np.median(dur_s)), 2),
        longest_s=round(float(np.max(dur_s)), 1),
        n_entries=int(n_min),
        entries_per_min=round(n_min / total_minutes, 2) if total_minutes > 0 else 0,
        frac_above_5s=round(float(np.mean(dur_s > 5)), 3),
        frac_above_30s=round(float(np.mean(dur_s > 30)), 3),
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_screener(df: pd.DataFrame, symbol: str, data_dir: str) -> ScreenerResult:
    """Run full regime screening pipeline."""
    n = len(df)
    ts = df["timestamp_ns"].values
    duration_h = (ts[-1] - ts[0]) / 1e9 / 3600

    signal_raw = pd.Series(df["imbalance_qty_l1"].values.astype(np.float64)).ffill().bfill().values
    prices = pd.Series(df["raw_midprice"].values.astype(np.float64)).ffill().bfill().values

    # Ultra-low bandpass
    log.info("    Bandpass filter (ultra-low) ...")
    signal_filt = bandpass_filter(signal_raw, *ULTRALOW_BAND)

    # Forward returns
    log.info("    Forward returns ...")
    fwd_ret = {label: compute_forward_returns(prices, h) for label, h in FWD_HORIZONS.items()}

    # Unconditional baselines
    log.info("    Unconditional IC baselines ...")
    bl_r1, _, _, _ = compute_ic(signal_raw, fwd_ret["1s"])
    bl_r5, _, _, _ = compute_ic(signal_raw, fwd_ret["5s"])
    if signal_filt is not None:
        bl_f1, _, _, _ = compute_ic(signal_filt, fwd_ret["1s"])
        bl_f5, _, _, _ = compute_ic(signal_filt, fwd_ret["5s"])
    else:
        bl_f1, bl_f5 = 0.0, 0.0

    baseline = {
        "ic_raw_1s": bl_r1, "ic_raw_5s": bl_r5,
        "ic_filt_1s": bl_f1, "ic_filt_5s": bl_f5,
    }

    # ── Phase 1: Single-factor screening ──
    log.info(f"    Phase 1: screening {len(ALL_REGIME_FEATURES)} features × {len(PERCENTILES)} percentiles ...")
    all_single = []
    all_masks: Dict[str, np.ndarray] = {}

    for feat in ALL_REGIME_FEATURES:
        if feat not in df.columns:
            continue
        results, masks = screen_single_factor(
            df, feat, signal_raw, signal_filt, fwd_ret, baseline, n,
        )
        all_single.extend(results)
        all_masks.update(masks)

    # Sort by best IC improvement (filtered 5s, then raw 5s)
    all_single.sort(key=lambda r: r.ic_improve_filt_5s, reverse=True)
    log.info(f"    {len(all_single)} single-factor conditions evaluated")

    # ── Phase 2: Multi-factor combinations ──
    top_labels = []
    seen_features = set()
    for sf in all_single:
        if sf.feature not in seen_features and sf.ic_improve_filt_5s > 0:
            top_labels.append(sf.label)
            seen_features.add(sf.feature)
        if len(top_labels) >= TOP_K_SINGLE:
            break

    # Also include top by raw IC improvement (may differ)
    all_single_by_raw = sorted(all_single, key=lambda r: r.ic_improve_raw_5s, reverse=True)
    for sf in all_single_by_raw:
        if sf.feature not in seen_features and sf.ic_improve_raw_5s > 0:
            top_labels.append(sf.label)
            seen_features.add(sf.feature)
        if len(top_labels) >= TOP_K_SINGLE + 5:
            break

    log.info(f"    Phase 2: combining top {len(top_labels)} factors ...")
    regime_feat_df = df[ALL_REGIME_FEATURES].copy() if len(top_labels) > 0 else pd.DataFrame()
    multi_results = screen_multi_factor(
        top_labels, all_masks, signal_raw, signal_filt, fwd_ret, baseline, n, regime_feat_df,
    )
    multi_results.sort(key=lambda r: r.ic_filt_5s, reverse=True)
    pareto_filter(multi_results)

    n_pareto = sum(1 for r in multi_results if r.is_pareto)
    log.info(f"    {len(multi_results)} combos tested, {n_pareto} Pareto-optimal")

    # ── Phase 3: Persistence ──
    log.info(f"    Phase 3: persistence analysis ...")
    pareto_combos = [r for r in multi_results if r.is_pareto][:15]
    persistence = []
    for r in pareto_combos:
        # Reconstruct mask
        mask = np.ones(n, dtype=bool)
        for lbl in r.labels:
            if lbl in all_masks:
                mask &= all_masks[lbl]
        persistence.append(measure_persistence(mask, " & ".join(r.labels), duration_h))

    # Also measure persistence for top single factors
    for sf in all_single[:5]:
        if sf.label in all_masks:
            persistence.append(measure_persistence(all_masks[sf.label], sf.label, duration_h))

    # Assessment
    best_single = all_single[0] if all_single else None
    best_pareto = next((r for r in multi_results if r.is_pareto), None)

    parts = []
    if best_single:
        parts.append(f"Best single: {best_single.label} (IC_filt_5s={best_single.ic_filt_5s:+.3f},"
                     f" +{best_single.ic_improve_filt_5s:+.3f} vs baseline, coverage={best_single.coverage:.0%})")
    if best_pareto:
        parts.append(f"Best combo: {' & '.join(best_pareto.labels)} (IC_filt_5s={best_pareto.ic_filt_5s:+.3f},"
                     f" +{best_pareto.ic_improve_filt_5s:+.3f} vs baseline, coverage={best_pareto.coverage:.0%})")

    assessment = " | ".join(parts) if parts else "No regime conditions improve IC"

    return ScreenerResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_dir=data_dir,
        symbol=symbol,
        n_rows=n,
        duration_hours=round(duration_h, 1),
        baseline_ic_raw_1s=round(bl_r1, 4),
        baseline_ic_raw_5s=round(bl_r5, 4),
        baseline_ic_filt_1s=round(bl_f1, 4),
        baseline_ic_filt_5s=round(bl_f5, 4),
        single_factors=all_single,
        multi_factors=multi_results,
        persistence=persistence,
        assessment=assessment,
    )


# ── Display ──────────────────────────────────────────────────────────────────

def print_screener(result: ScreenerResult):
    W, BOLD = "\033[0m", "\033[1m"
    G, Y, R = "\033[32m", "\033[33m", "\033[31m"

    w = 110
    print(f"\n{'=' * w}")
    print(f"  {BOLD}REGIME SCREENER — {result.symbol}{W}"
          f"  ({result.n_rows:,} rows, {result.duration_hours:.1f}h)")
    print(f"  Baselines — raw: IC_1s={result.baseline_ic_raw_1s:+.3f}  IC_5s={result.baseline_ic_raw_5s:+.3f}"
          f"  |  filtered: IC_1s={result.baseline_ic_filt_1s:+.3f}  IC_5s={result.baseline_ic_filt_5s:+.3f}")
    print(f"{'=' * w}")

    # ── Phase 1: Single factors ──
    print(f"\n  {BOLD}1. Single-Factor Screening (top 20 by filtered IC improvement){W}")
    print(f"  {'─' * (w - 4)}")
    hdr = (f"  {'condition':<35} {'category':>10} {'cover':>6} {'n_obs':>7}"
           f" {'IC_r5s':>7} {'IC_f5s':>7} {'IR_f5s':>7} {'dIC_f5s':>8}")
    print(hdr)
    print(f"  {'─'*35} {'─'*10} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")

    for sf in result.single_factors[:20]:
        color = G if sf.ic_improve_filt_5s > 0.01 else Y if sf.ic_improve_filt_5s > 0 else W
        print(f"  {sf.label:<35} {sf.category:>10} {sf.coverage:>5.0%} {sf.n_obs:>7,}"
              f" {sf.ic_raw_5s:>+7.3f} {sf.ic_filt_5s:>+7.3f} {sf.ic_ir_filt_5s:>7.1f}"
              f" {color}{sf.ic_improve_filt_5s:>+8.4f}{W}")

    # ── Phase 2: Multi-factor Pareto ──
    pareto = [r for r in result.multi_factors if r.is_pareto]
    non_pareto_top = [r for r in result.multi_factors if not r.is_pareto][:10]

    print(f"\n  {BOLD}2. Multi-Factor Pareto Frontier ({len(pareto)} combos){W}")
    print(f"  {'─' * (w - 4)}")
    hdr2 = (f"  {'P':>1} {'condition':<50} {'#f':>2} {'cover':>6} {'n_obs':>7}"
            f" {'IC_r5s':>7} {'IC_f5s':>7} {'dIC_f5s':>8}")
    print(hdr2)
    print(f"  {'─'*1} {'─'*50} {'─'*2} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")

    for r in pareto[:20]:
        label = " & ".join(r.labels)
        if len(label) > 50:
            label = label[:47] + "..."
        print(f"  {G}*{W} {label:<50} {r.n_factors:>2} {r.coverage:>5.0%} {r.n_obs:>7,}"
              f" {r.ic_raw_5s:>+7.3f} {G}{r.ic_filt_5s:>+7.3f}{W} {r.ic_improve_filt_5s:>+8.4f}")

    for r in non_pareto_top:
        label = " & ".join(r.labels)
        if len(label) > 50:
            label = label[:47] + "..."
        print(f"    {label:<50} {r.n_factors:>2} {r.coverage:>5.0%} {r.n_obs:>7,}"
              f" {r.ic_raw_5s:>+7.3f} {r.ic_filt_5s:>+7.3f} {r.ic_improve_filt_5s:>+8.4f}")

    # ── Phase 3: Persistence ──
    print(f"\n  {BOLD}3. Regime Persistence{W}")
    print(f"  {'─' * (w - 4)}")
    hdr3 = (f"  {'condition':<50} {'mean_s':>7} {'med_s':>7} {'long_s':>7}"
            f" {'entries':>7} {'/min':>6} {'>5s':>5} {'>30s':>5}")
    print(hdr3)
    print(f"  {'─'*50} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*6} {'─'*5} {'─'*5}")

    for p in result.persistence:
        label = p.label
        if len(label) > 50:
            label = label[:47] + "..."
        gt5_color = G if p.frac_above_5s > 0.5 else Y if p.frac_above_5s > 0.2 else R
        print(f"  {label:<50} {p.mean_duration_s:>7.1f} {p.median_duration_s:>7.1f}"
              f" {p.longest_s:>7.0f} {p.n_entries:>7,} {p.entries_per_min:>6.1f}"
              f" {gt5_color}{p.frac_above_5s:>5.0%}{W} {p.frac_above_30s:>5.0%}")

    # ── Assessment ──
    print(f"\n  {BOLD}Assessment:{W} {result.assessment}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Spannung regime screener")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to parquet data")
    parser.add_argument("--symbol", type=str, default="all", help='Symbol or "all"')
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = Path(args.output) if args.output else ROOT / "reports" / "spannung"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = SYMBOLS if args.symbol == "all" else [args.symbol.upper()]

    for sym in symbols:
        log.info(f"  Loading {sym} from {data_dir} ...")
        df = load_parquet(str(data_dir), symbols=[sym], columns=NEEDED)
        if df.empty:
            log.warning(f"  No data for {sym}, skipping")
            continue
        df = df.sort_values("timestamp_ns").reset_index(drop=True)
        log.info(f"  {len(df):,} rows")

        t0 = time.time()
        result = run_screener(df, sym, data_dir)
        elapsed = time.time() - t0
        log.info(f"    Completed in {elapsed:.1f}s")

        result_path = out_dir / f"regime_screen_{sym}.json"
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        log.info(f"  Saved: {result_path}")

        print_screener(result)
        del df


if __name__ == "__main__":
    main()
