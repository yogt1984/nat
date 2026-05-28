#!/usr/bin/env python3
"""
Scanner & Data Quality Visualizations for NAT.

Generates 10 diagnostic plots from Scalp Edge Scanner results and raw data:
  Scanner (1-7): conditional KDE, edge-frequency, conjunction heatmap,
                 trigger raster, cumulative PnL, holding curves, stability bars
  Data (8-10):   feature coverage, tick activity, return distribution

Usage:
    python scripts/visualize_scanner.py scan --symbol BTC
    python scripts/visualize_scanner.py data --symbol BTC
    python scripts/visualize_scanner.py all  --symbol BTC
    python scripts/visualize_scanner.py scan --report reports/scalp_scanner/scan_BTC_20260508.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

from scalp_edge_scanner import (
    ConjunctionSetup,
    FeatureTailProfile,
    ScalpEdgeScanner,
    ScanReport,
    StabilityResult,
    StrategyArchetype,
    TailStats,
    TemporalProfile,
    load_scanner_config,
    save_json_report,
)
from cluster_pipeline.loader import load_parquet, filter_symbol
from cluster_pipeline.preprocess import aggregate_bars
from cluster_pipeline.config import FEATURE_VECTORS

from viz.features import STYLE, COLORS, apply_style


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

STABILITY_COLORS = {"stable": "#3fb950", "fragile": "#f85149", "insufficient_data": "#8b949e"}


def _apply():
    apply_style()
    plt.rcParams.update({"figure.dpi": 100})


# ---------------------------------------------------------------------------
# Report loading (JSON -> dataclasses)
# ---------------------------------------------------------------------------

def load_scan_report(path: str) -> ScanReport:
    """Deserialize a scanner JSON report back into a ScanReport."""
    with open(path) as f:
        data = json.load(f)

    def _tail(d):
        if d is None:
            return None
        return TailStats(**d)

    profiles = []
    for p in data.get("tail_profiles", []):
        pctiles = {int(k): v for k, v in p.get("percentile_thresholds", {}).items()}
        profiles.append(FeatureTailProfile(
            name=p["name"], vector=p["vector"],
            percentile_thresholds=pctiles,
            lower_tail=_tail(p.get("lower_tail")),
            upper_tail=_tail(p.get("upper_tail")),
            is_significant=p["is_significant"],
            edge_bps=p["edge_bps"], p_adjusted=p["p_adjusted"],
        ))

    conjunctions = [ConjunctionSetup(**c) for c in data.get("conjunctions", [])]

    temporal = []
    for t in data.get("temporal", []):
        temporal.append(TemporalProfile(
            setup_id=t["setup_id"],
            optimal_holding_bars=t["optimal_holding_bars"],
            clustering_coeff=t["clustering_coeff"],
            cooldown_bars=t["cooldown_bars"],
            regime_edges=t.get("regime_edges", {}),
        ))

    stability = [StabilityResult(**s) for s in data.get("stability", [])]

    archetypes = []
    for a in data.get("archetypes", []):
        archetypes.append(StrategyArchetype(
            name=a["name"], description=a["description"],
            entry_logic=a["entry_logic"], exit_logic=a["exit_logic"],
            matching_setups=a.get("matching_setups", []),
            confidence=a["confidence"],
        ))

    return ScanReport(
        symbol=data["symbol"], timeframe=data["timeframe"],
        n_bars=data["n_bars"], timestamp=data["timestamp"],
        tail_profiles=profiles, conjunctions=conjunctions,
        temporal=temporal, stability=stability,
        archetypes=archetypes, config=data.get("config", {}),
        warnings=data.get("warnings", []),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_trigger_mask(
    profile: FeatureTailProfile,
    bars: pd.DataFrame,
    tail_threshold: int,
) -> Tuple[np.ndarray, str]:
    """Build boolean trigger mask for the best tail of a feature profile.

    Returns (mask, tail_label) where tail_label is "upper" or "lower".
    """
    values = bars[profile.name].values.astype(float)
    lo = profile.percentile_thresholds.get(tail_threshold, np.nan)
    hi = profile.percentile_thresholds.get(100 - tail_threshold, np.nan)

    if profile.upper_tail and profile.lower_tail:
        best = "upper" if abs(profile.upper_tail.mean) >= abs(profile.lower_tail.mean) else "lower"
    elif profile.upper_tail:
        best = "upper"
    elif profile.lower_tail:
        best = "lower"
    else:
        return np.zeros(len(bars), dtype=bool), "upper"

    if best == "upper":
        mask = np.isfinite(values) & (values >= hi)
    else:
        mask = np.isfinite(values) & (values <= lo)
    return mask, best


def _short_name(name: str, max_len: int = 22) -> str:
    """Shorten feature name for plot labels."""
    for suffix in ("_mean", "_std", "_last", "_close", "_open"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name[:max_len] if len(name) > max_len else name


def _sig_profiles(report: ScanReport) -> List[FeatureTailProfile]:
    return [p for p in report.tail_profiles if p.is_significant]


def _best_tail(p: FeatureTailProfile) -> Optional[TailStats]:
    if p.upper_tail and p.lower_tail:
        return p.upper_tail if abs(p.upper_tail.mean) >= abs(p.lower_tail.mean) else p.lower_tail
    return p.upper_tail or p.lower_tail


# ---------------------------------------------------------------------------
# Plot 1: Conditional Return KDE
# ---------------------------------------------------------------------------

def plot_conditional_return_kde(
    report: ScanReport,
    bars: pd.DataFrame,
    fwd_returns: np.ndarray,
) -> plt.Figure:
    """Overlay KDE of fwd returns in tail vs middle for top significant features."""
    _apply()
    sig = _sig_profiles(report)[:6]
    tail_thresh = report.config.get("tail_threshold", 5)

    n_plots = max(len(sig), 1)
    ncols = min(n_plots, 3)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    finite = np.isfinite(fwd_returns)

    for i, prof in enumerate(sig):
        ax = axes[i]
        values = bars[prof.name].values.astype(float)
        lo = prof.percentile_thresholds.get(tail_thresh, np.nan)
        hi = prof.percentile_thresholds.get(100 - tail_thresh, np.nan)

        tail_mask, tail_label = _build_trigger_mask(prof, bars, tail_thresh)
        mid_mask = finite & np.isfinite(values) & (values > lo) & (values < hi)

        rets_tail = fwd_returns[tail_mask & finite]
        rets_mid = fwd_returns[mid_mask]

        if len(rets_tail) > 2:
            kde_t = stats.gaussian_kde(rets_tail)
            x = np.linspace(
                min(rets_tail.min(), rets_mid.min() if len(rets_mid) else rets_tail.min()),
                max(rets_tail.max(), rets_mid.max() if len(rets_mid) else rets_tail.max()),
                200,
            )
            ax.fill_between(x, kde_t(x), alpha=0.4, color=COLORS[0],
                            label=f"{tail_label} tail (n={len(rets_tail)})")
        if len(rets_mid) > 2:
            kde_m = stats.gaussian_kde(rets_mid)
            x = np.linspace(rets_mid.min(), rets_mid.max(), 200)
            ax.fill_between(x, kde_m(x), alpha=0.3, color=COLORS[3],
                            label=f"middle (n={len(rets_mid)})")

        bt = _best_tail(prof)
        edge_str = f"{prof.edge_bps:.1f}bps" if bt else ""
        ax.set_title(f"{_short_name(prof.name)} [{edge_str}]", fontsize=10)
        ax.axvline(0, color="#8b949e", ls="--", lw=0.8, alpha=0.6)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("fwd return", fontsize=8)
        ax.set_ylabel("density", fontsize=8)

    for j in range(len(sig), len(axes)):
        axes[j].set_visible(False)

    if not sig:
        axes[0].text(0.5, 0.5, "No significant features", transform=axes[0].transAxes,
                     ha="center", va="center", fontsize=14, color="#8b949e")

    fig.suptitle(f"Conditional Return KDE — {report.symbol} ({report.timeframe})",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Edge vs Frequency Scatter
# ---------------------------------------------------------------------------

def plot_edge_vs_frequency(report: ScanReport) -> plt.Figure:
    """Scatter: frequency vs edge, sized by n_obs, colored by stability."""
    _apply()
    sig = _sig_profiles(report)
    stab_map = {s.setup_id: s.status for s in report.stability}
    cost = report.config.get("cost_bps", 3.5)

    fig, ax = plt.subplots(figsize=(10, 7))

    if not sig:
        ax.text(0.5, 0.5, "No significant features", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#8b949e")
        return fig

    for prof in sig:
        bt = _best_tail(prof)
        if not bt:
            continue
        status = stab_map.get(f"tail:{prof.name}", "insufficient_data")
        color = STABILITY_COLORS.get(status, "#8b949e")
        size = max(bt.n_obs * 0.5, 20)
        ax.scatter(bt.frequency * 100, prof.edge_bps, s=size, c=color,
                   alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(_short_name(prof.name, 18),
                    (bt.frequency * 100, prof.edge_bps),
                    fontsize=6.5, color="#c9d1d9", alpha=0.9,
                    xytext=(4, 4), textcoords="offset points")

    ax.axhline(cost, color="#f85149", ls="--", lw=1, alpha=0.6, label=f"break-even ({cost} bps)")
    ax.set_xlabel("Tail Frequency (%)", fontsize=11)
    ax.set_ylabel("Edge (bps)", fontsize=11)
    ax.set_title(f"Edge vs Frequency — {report.symbol}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Legend for stability colors
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for l, c in STABILITY_COLORS.items()]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], fontsize=8, loc="upper right")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Conjunction Heatmap
# ---------------------------------------------------------------------------

def plot_conjunction_heatmap(report: ScanReport) -> plt.Figure:
    """Symmetric heatmap of pairwise conjunction edges."""
    _apply()
    conj = report.conjunctions
    fig, ax = plt.subplots(figsize=(10, 8))

    if not conj:
        ax.text(0.5, 0.5, "No significant conjunctions", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#8b949e")
        ax.set_title(f"Conjunction Heatmap — {report.symbol}", fontsize=13)
        return fig

    # Collect unique features, cap at 20
    feat_counts: Dict[str, int] = {}
    for c in conj:
        feat_counts[c.feature_a] = feat_counts.get(c.feature_a, 0) + 1
        feat_counts[c.feature_b] = feat_counts.get(c.feature_b, 0) + 1
    top_feats = sorted(feat_counts, key=feat_counts.get, reverse=True)[:20]
    feat_idx = {f: i for i, f in enumerate(top_feats)}
    n = len(top_feats)

    matrix = np.full((n, n), np.nan)
    for c in conj:
        if c.feature_a in feat_idx and c.feature_b in feat_idx:
            i, j = feat_idx[c.feature_a], feat_idx[c.feature_b]
            val = c.edge_bps
            if np.isnan(matrix[i, j]) or abs(val) > abs(matrix[i, j]):
                matrix[i, j] = val
                matrix[j, i] = val

    labels = [_short_name(f, 16) for f in top_feats]
    mask = np.isnan(matrix)

    sns.heatmap(matrix, ax=ax, mask=mask, cmap="RdYlGn", center=0,
                annot=True, fmt=".1f", linewidths=0.5, linecolor="#30363d",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Edge (bps)"})
    ax.set_title(f"Conjunction Heatmap — {report.symbol}", fontsize=13)
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7, rotation=0)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Trigger Raster
# ---------------------------------------------------------------------------

def plot_trigger_raster(
    report: ScanReport,
    bars: pd.DataFrame,
    fwd_returns: np.ndarray,
) -> plt.Figure:
    """Raster plot of tail triggers across time for top features."""
    _apply()
    sig = _sig_profiles(report)[:15]
    tail_thresh = report.config.get("tail_threshold", 5)

    fig, ax = plt.subplots(figsize=(16, max(4, len(sig) * 0.4 + 1)))

    if not sig:
        ax.text(0.5, 0.5, "No significant features", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#8b949e")
        ax.set_title(f"Trigger Raster — {report.symbol}", fontsize=13)
        return fig

    # Background shade by forward return sign
    finite = np.isfinite(fwd_returns)
    for i in range(len(fwd_returns)):
        if finite[i]:
            color = "#3fb95008" if fwd_returns[i] > 0 else "#f8514908"
            ax.axvspan(i - 0.5, i + 0.5, color=color, lw=0)

    for y_idx, prof in enumerate(sig):
        if prof.name not in bars.columns:
            continue
        mask, _ = _build_trigger_mask(prof, bars, tail_thresh)
        trigger_idx = np.where(mask)[0]
        ax.scatter(trigger_idx, [y_idx] * len(trigger_idx),
                   s=8, c=COLORS[y_idx % len(COLORS)], alpha=0.7,
                   marker="|", linewidths=1.5)

    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels([_short_name(p.name, 20) for p in sig], fontsize=7)
    ax.set_xlabel("Bar index", fontsize=10)
    ax.set_title(f"Trigger Raster — {report.symbol} ({report.timeframe})", fontsize=13)
    ax.set_ylim(-0.5, len(sig) - 0.5)
    ax.grid(True, axis="x", alpha=0.15)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 5: Cumulative Triggered PnL
# ---------------------------------------------------------------------------

def plot_cumulative_triggered_pnl(
    report: ScanReport,
    bars: pd.DataFrame,
    fwd_returns: np.ndarray,
) -> plt.Figure:
    """Cumulative PnL at trigger points for top setups (gross + cost-adjusted)."""
    _apply()
    sig = _sig_profiles(report)[:6]
    tail_thresh = report.config.get("tail_threshold", 5)
    cost = report.config.get("cost_bps", 3.5) * 1e-4 * 2  # round-trip

    fig, ax = plt.subplots(figsize=(14, 6))

    if not sig:
        ax.text(0.5, 0.5, "No significant features", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#8b949e")
        ax.set_title(f"Cumulative Triggered PnL — {report.symbol}", fontsize=13)
        return fig

    finite = np.isfinite(fwd_returns)
    for i, prof in enumerate(sig):
        if prof.name not in bars.columns:
            continue
        mask, _ = _build_trigger_mask(prof, bars, tail_thresh)
        combined = mask & finite
        rets = fwd_returns[combined]

        bt = _best_tail(prof)
        if bt and bt.direction == "bearish":
            rets = -rets

        gross = np.cumsum(rets) * 1e4
        net = np.cumsum(rets - cost) * 1e4

        color = COLORS[i % len(COLORS)]
        label = _short_name(prof.name, 18)
        ax.plot(gross, color=color, lw=1.5, label=f"{label} (gross)")
        ax.plot(net, color=color, lw=1, ls="--", alpha=0.6)

    ax.axhline(0, color="#8b949e", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Trigger #", fontsize=10)
    ax.set_ylabel("Cumulative PnL (bps)", fontsize=10)
    ax.set_title(f"Cumulative Triggered PnL — {report.symbol} (solid=gross, dashed=net)",
                 fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 6: Holding Period Curves
# ---------------------------------------------------------------------------

def plot_holding_period_curves(
    report: ScanReport,
    bars: pd.DataFrame,
    scanner: ScalpEdgeScanner,
) -> plt.Figure:
    """Mean return at multiple horizons for top setups."""
    _apply()
    sig = _sig_profiles(report)[:6]
    tail_thresh = report.config.get("tail_threshold", 5)
    horizons = [1, 2, 3, 5, 10]

    fig, ax = plt.subplots(figsize=(10, 6))

    if not sig:
        ax.text(0.5, 0.5, "No significant features", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#8b949e")
        ax.set_title(f"Holding Period Curves — {report.symbol}", fontsize=13)
        return fig

    # Pre-compute forward returns at each horizon
    fwd_by_h = {}
    for h in horizons:
        fwd_by_h[h] = scanner.compute_forward_returns(bars, horizon=h)

    for i, prof in enumerate(sig):
        if prof.name not in bars.columns:
            continue
        mask, _ = _build_trigger_mask(prof, bars, tail_thresh)

        bt = _best_tail(prof)
        flip = -1 if (bt and bt.direction == "bearish") else 1

        means, sems = [], []
        for h in horizons:
            fwd = fwd_by_h[h]
            combined = mask & np.isfinite(fwd)
            rets = fwd[combined] * flip
            if len(rets) > 1:
                means.append(float(np.mean(rets)) * 1e4)
                sems.append(float(np.std(rets, ddof=1) / np.sqrt(len(rets))) * 1e4)
            else:
                means.append(0.0)
                sems.append(0.0)

        color = COLORS[i % len(COLORS)]
        ax.errorbar(horizons, means, yerr=sems, color=color, marker="o",
                    lw=1.5, capsize=3, label=_short_name(prof.name, 18))

    ax.axhline(0, color="#8b949e", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Holding period (bars)", fontsize=10)
    ax.set_ylabel("Mean return (bps)", fontsize=10)
    ax.set_title(f"Holding Period Curves — {report.symbol}", fontsize=13)
    ax.set_xticks(horizons)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 7: Stability Half-Split Bars
# ---------------------------------------------------------------------------

def plot_stability_half_split(report: ScanReport) -> plt.Figure:
    """Grouped bar chart: first-half vs second-half edge per setup."""
    _apply()
    stab = [s for s in report.stability if s.status != "insufficient_data"]

    fig, ax = plt.subplots(figsize=(max(8, len(stab) * 0.8 + 2), 6))

    if not stab:
        all_stab = report.stability
        if all_stab:
            ax.text(0.5, 0.5, f"All {len(all_stab)} setups have insufficient data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="#8b949e")
        else:
            ax.text(0.5, 0.5, "No stability data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="#8b949e")
        ax.set_title(f"Stability Half-Split — {report.symbol}", fontsize=13)
        return fig

    x = np.arange(len(stab))
    width = 0.35

    colors_1h = [STABILITY_COLORS[s.status] for s in stab]
    colors_2h = colors_1h  # same color per setup

    ax.bar(x - width / 2, [s.edge_first_half for s in stab], width,
           color=colors_1h, alpha=0.8, edgecolor="white", linewidth=0.5, label="1st half")
    ax.bar(x + width / 2, [s.edge_second_half for s in stab], width,
           color=colors_2h, alpha=0.5, edgecolor="white", linewidth=0.5, label="2nd half")

    ax.axhline(0.5, color="#3fb950", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(-0.5, color="#3fb950", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(0, color="#8b949e", ls="--", lw=0.8, alpha=0.5)

    labels = [_short_name(s.setup_id.replace("tail:", "").replace("conj:", ""), 16)
              for s in stab]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Edge (bps)", fontsize=10)
    ax.set_title(f"Stability Half-Split — {report.symbol}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 8: Feature Coverage Heatmap
# ---------------------------------------------------------------------------

def plot_feature_coverage_heatmap(bars: pd.DataFrame) -> plt.Figure:
    """NaN rate per feature vector category per data day."""
    _apply()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Detect date from index
    if hasattr(bars.index, "date"):
        bars = bars.copy()
        bars["_date"] = bars.index.date
    elif "bar_start" in bars.columns:
        bars = bars.copy()
        bars["_date"] = pd.to_datetime(bars["bar_start"]).dt.date
    else:
        ax.text(0.5, 0.5, "Cannot determine dates from bars",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color="#8b949e")
        return fig

    dates = sorted(bars["_date"].unique())
    if len(dates) < 1:
        ax.text(0.5, 0.5, "No data dates found", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#8b949e")
        return fig

    # Group columns by vector
    vector_cols: Dict[str, List[str]] = {}
    for col in bars.columns:
        if col.startswith("_") or col in ("symbol", "bar_start", "bar_end", "tick_count"):
            continue
        for vname, vspec in FEATURE_VECTORS.items():
            matched = False
            for prefix in vspec.get("prefixes", []):
                if col.startswith(prefix):
                    vector_cols.setdefault(vname, []).append(col)
                    matched = True
                    break
            if matched:
                break

    if not vector_cols:
        ax.text(0.5, 0.5, "No feature vectors matched", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#8b949e")
        return fig

    vectors = sorted(vector_cols.keys())
    matrix = np.zeros((len(dates), len(vectors)))
    for d_idx, date in enumerate(dates):
        day_mask = bars["_date"] == date
        day = bars.loc[day_mask]
        for v_idx, vec in enumerate(vectors):
            cols = vector_cols[vec]
            existing = [c for c in cols if c in day.columns]
            if existing:
                matrix[d_idx, v_idx] = day[existing].isna().mean().mean()

    date_labels = [str(d) for d in dates]
    sns.heatmap(matrix, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                xticklabels=vectors, yticklabels=date_labels,
                annot=len(dates) <= 14, fmt=".0%",
                linewidths=0.5, linecolor="#30363d",
                cbar_kws={"label": "NaN rate"})
    ax.set_title("Feature Coverage by Vector & Day", fontsize=13)
    plt.xticks(fontsize=9, rotation=30, ha="right")
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 9: Tick Activity Heatmap
# ---------------------------------------------------------------------------

def plot_tick_activity_heatmap(bars: pd.DataFrame) -> plt.Figure:
    """Hour-of-day vs date heatmap of tick counts."""
    _apply()
    fig, ax = plt.subplots(figsize=(14, 6))

    if hasattr(bars.index, "hour"):
        hours = bars.index.hour
        dates = bars.index.date
    elif "bar_start" in bars.columns:
        ts = pd.to_datetime(bars["bar_start"])
        hours = ts.dt.hour
        dates = ts.dt.date
    else:
        ax.text(0.5, 0.5, "Cannot determine timestamps from bars",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color="#8b949e")
        return fig

    tmp = pd.DataFrame({"date": dates, "hour": hours})
    if "tick_count" in bars.columns:
        tmp["count"] = bars["tick_count"].values
        pivot = tmp.groupby(["hour", "date"])["count"].sum().unstack(fill_value=0)
    else:
        pivot = tmp.groupby(["hour", "date"]).size().unstack(fill_value=0)

    date_labels = [str(d) for d in pivot.columns]
    sns.heatmap(pivot.values, ax=ax, cmap="viridis",
                xticklabels=date_labels, yticklabels=pivot.index,
                linewidths=0.3, linecolor="#30363d",
                cbar_kws={"label": "Tick count"})
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Hour (UTC)", fontsize=10)
    ax.set_title("Tick Activity by Hour & Day", fontsize=13)
    plt.xticks(fontsize=8, rotation=30, ha="right")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 10: Return Distribution Panel
# ---------------------------------------------------------------------------

def plot_return_distribution_panel(
    bars: pd.DataFrame,
    fwd_returns: np.ndarray,
    cost_bps: float = 3.5,
) -> plt.Figure:
    """Histogram + QQ plot of forward returns."""
    _apply()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    rets = fwd_returns[np.isfinite(fwd_returns)]
    cost = cost_bps * 1e-4

    # Histogram
    ax = axes[0]
    if len(rets) > 0:
        ax.hist(rets, bins=80, density=True, color=COLORS[0], alpha=0.7, edgecolor="none")
        ax.axvline(cost, color="#f85149", ls="--", lw=1, alpha=0.7, label=f"+{cost_bps}bps")
        ax.axvline(-cost, color="#f85149", ls="--", lw=1, alpha=0.7, label=f"-{cost_bps}bps")
        ax.axvline(0, color="#8b949e", ls="--", lw=0.8, alpha=0.5)

        sk = float(stats.skew(rets))
        ku = float(stats.kurtosis(rets))
        ax.text(0.02, 0.95, f"skew={sk:.2f}\nkurtosis={ku:.1f}\nn={len(rets):,}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="#161b22", alpha=0.8, edgecolor="#30363d"))
    ax.set_xlabel("Forward return", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Return Distribution", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # QQ plot
    ax2 = axes[1]
    if len(rets) > 10:
        stats.probplot(rets, dist="norm", plot=ax2)
        ax2.get_lines()[0].set(color=COLORS[0], alpha=0.6, markersize=2)
        ax2.get_lines()[1].set(color="#f85149", lw=1.5)
    ax2.set_title("QQ Plot (vs Normal)", fontsize=12)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(f"Return Distribution ({len(rets):,} observations)", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_scanner_plots(
    report: ScanReport,
    bars: pd.DataFrame,
    fwd_returns: np.ndarray,
    scanner: ScalpEdgeScanner,
    output_dir: str,
    dpi: int = 150,
) -> List[str]:
    """Generate all 7 scanner plots, save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    sym = report.symbol
    saved = []

    plots = [
        ("conditional_kde", lambda: plot_conditional_return_kde(report, bars, fwd_returns)),
        ("edge_frequency", lambda: plot_edge_vs_frequency(report)),
        ("conjunction_heatmap", lambda: plot_conjunction_heatmap(report)),
        ("trigger_raster", lambda: plot_trigger_raster(report, bars, fwd_returns)),
        ("cumulative_pnl", lambda: plot_cumulative_triggered_pnl(report, bars, fwd_returns)),
        ("holding_curves", lambda: plot_holding_period_curves(report, bars, scanner)),
        ("stability_bars", lambda: plot_stability_half_split(report)),
    ]

    for name, fn in plots:
        try:
            fig = fn()
            path = os.path.join(output_dir, f"{name}_{sym}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            saved.append(path)
            print(f"  [saved] {name}_{sym}.png", flush=True)
        except Exception as e:
            print(f"  [skip] {name}: {e}", flush=True)

    return saved


def run_data_plots(
    bars: pd.DataFrame,
    fwd_returns: np.ndarray,
    cost_bps: float,
    output_dir: str,
    symbol: str = "BTC",
    dpi: int = 150,
) -> List[str]:
    """Generate 3 data quality plots."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    plots = [
        ("data_coverage", lambda: plot_feature_coverage_heatmap(bars)),
        ("data_activity", lambda: plot_tick_activity_heatmap(bars)),
        ("data_returns", lambda: plot_return_distribution_panel(bars, fwd_returns, cost_bps)),
    ]

    for name, fn in plots:
        try:
            fig = fn()
            path = os.path.join(output_dir, f"{name}_{symbol}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            saved.append(path)
            print(f"  [saved] {name}_{symbol}.png", flush=True)
        except Exception as e:
            print(f"  [skip] {name}: {e}", flush=True)

    return saved


def run_all(
    data_dir: str,
    symbol: str,
    output_dir: str,
    report_path: Optional[str] = None,
    dpi: int = 150,
    plots: str = "all",
) -> List[str]:
    """Full visualization suite: load data, run scanner if needed, generate plots.

    Args:
        plots: "scanner", "data", or "all"
    """
    cfg = load_scanner_config()
    scanner = ScalpEdgeScanner(cfg)
    cost_bps = cfg.get("cost_bps", 3.5)

    # Load report or run scanner
    report = None
    if report_path:
        print(f"  Loading report: {report_path}", flush=True)
        report = load_scan_report(report_path)

    # Load raw data for plots that need bars
    need_bars = plots in ("all", "data") or (plots == "scanner" and report_path is None)
    bars = None
    fwd_returns = None

    if need_bars or plots in ("all", "scanner"):
        print(f"  Loading data from {data_dir}...", flush=True)
        bars = scanner.load_and_aggregate(data_dir, symbol)
        fwd_returns = scanner.compute_forward_returns(bars)
        print(f"  {len(bars)} bars loaded", flush=True)

    if report is None and plots in ("all", "scanner"):
        print(f"  Running scanner on {symbol}...", flush=True)
        report = scanner.scan(data_dir, symbol)

    saved = []

    if plots in ("all", "scanner") and report is not None and bars is not None:
        print(f"\n  --- Scanner Plots ---", flush=True)
        saved += run_scanner_plots(report, bars, fwd_returns, scanner, output_dir, dpi)

    if plots in ("all", "data") and bars is not None:
        print(f"\n  --- Data Quality Plots ---", flush=True)
        saved += run_data_plots(bars, fwd_returns, cost_bps, output_dir, symbol, dpi)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="visualize_scanner",
        description="Scanner & data quality visualizations for NAT",
    )
    sub = parser.add_subparsers(dest="command")

    for cmd in ("scan", "data", "all"):
        p = sub.add_parser(cmd, help=f"Generate {cmd} plots")
        p.add_argument("--symbol", default="BTC")
        p.add_argument("--data-dir", default=str(ROOT / "data" / "features"))
        p.add_argument("--report", default=None, help="Scanner JSON report path")
        p.add_argument("--output", default=str(ROOT / "reports" / "figures" / "scanner"))
        p.add_argument("--dpi", type=int, default=150)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    paths = run_all(
        data_dir=args.data_dir, symbol=args.symbol,
        output_dir=args.output, report_path=args.report,
        dpi=args.dpi, plots=args.command,
    )
    print(f"\n  Done. {len(paths)} figures saved to {args.output}/\n")


if __name__ == "__main__":
    main()
