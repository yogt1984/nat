"""15-Minute Visual Health Check — market microstructure snapshot.

Companion to 15m_test.py. Produces two pages of 6-panel PNGs per symbol:
  Page 1: Price, Depth, Flow, Microstructure, Entropy, Heatmap
  Page 2: Toxicity, Trend Regime, Funding/OI, Illiquidity, Multi-scale Entropy, Interactions

Usage:
    python3 scripts/15m_visualize.py --latest --symbol BTC
    python3 scripts/15m_visualize.py --latest --symbol all --page 2
    python3 scripts/15m_visualize.py --data-dir data/features/2026-05-12 --symbol BTC --window 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# Allow imports from scripts/
from cluster_pipeline.config import META_COLUMNS
from cluster_pipeline.loader import load_parquet, get_symbols
from cluster_pipeline.preprocess import aggregate_bars
from viz.features import STYLE, COLORS, apply_style

log = logging.getLogger("15m_viz")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "smoke_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in df.columns:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["timestamp_ns"], unit="ns")
    return df


def _zscore(s: pd.Series) -> pd.Series:
    m, sd = np.nanmean(s), np.nanstd(s)
    return (s - m) / (sd + 1e-10)


def _safe_col(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    s = df[col]
    if s.isna().mean() > 0.5:
        return None
    return s


def _bar_col(bars: pd.DataFrame, base: str, preferred_suffix: str = "sum") -> Optional[pd.Series]:
    """Find an aggregated bar column, trying preferred suffix first."""
    for suffix in [preferred_suffix, "mean", "last"]:
        col = f"{base}_{suffix}"
        if col in bars.columns:
            return bars[col]
    return None


def _no_data(ax: plt.Axes, msg: str, title: str) -> None:
    """Show placeholder when data is missing."""
    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="#8b949e")
    ax.set_title(title, fontsize=10, color="#c9d1d9")


def _detect_anomalies(df: pd.DataFrame, symbol: str) -> list[str]:
    warnings = []
    ts = df["timestamp_ns"].values
    diffs = np.diff(ts) / 1e9  # seconds
    gaps = np.sum(diffs > 5.0)
    if gaps > 0:
        warnings.append(f"{symbol}: {gaps} gaps >5s detected")

    mp = _safe_col(df, "raw_midprice")
    if mp is not None:
        # Check 30s rolling windows for frozen midprice
        window = 300  # 30s at 10/sec
        if len(mp) > window:
            rolling_std = mp.rolling(window).std()
            frozen = (rolling_std < 1e-10).sum()
            if frozen > 0:
                warnings.append(f"{symbol}: midprice frozen for {frozen} ticks")

    # NaN spike check — exclude optional features (whale, liquidation, concentration, gmm, regime_prob)
    optional_prefixes = ("whale_", "liquidation_", "top5_", "gmm_", "regime_prob_")
    num_cols = [c for c in df.columns if c not in META_COLUMNS and c != "datetime"
                and df[c].dtype in [np.float64, np.float32]
                and not c.startswith(optional_prefixes)]
    if num_cols:
        nan_rate = df[num_cols].isna().mean(axis=1)
        spike = (nan_rate > 0.2).sum()
        if spike > 0:
            warnings.append(f"{symbol}: {spike} ticks with >20% NaN")

    return warnings


# ---------------------------------------------------------------------------
# Derived features (computed at viz time)
# ---------------------------------------------------------------------------


def _compute_derived_viz(df: pd.DataFrame) -> pd.DataFrame:
    """Add exploratory derived columns for page 2 panels."""
    df = df.copy()

    # A. Flow momentum: short/long ratio smoothed
    f5 = _safe_col(df, "flow_count_5s")
    f30 = _safe_col(df, "flow_count_30s")
    if f5 is not None and f30 is not None:
        ratio = f5.ewm(span=30, min_periods=5).mean() / (f30.ewm(span=30, min_periods=5).mean() + 1e-8)
        df["viz_flow_momentum"] = ratio
        df["viz_flow_acceleration"] = ratio.diff(10)

    # B. Book pressure gradient: L1 vs L5 imbalance
    l1 = _safe_col(df, "imbalance_qty_l1")
    l5 = _safe_col(df, "imbalance_qty_l5")
    if l1 is not None and l5 is not None:
        df["viz_book_pressure_gradient"] = l1 / (l5.abs() + 1e-8)

    # C. VPIN regime z-score
    vpin = _safe_col(df, "toxic_vpin_50")
    if vpin is not None:
        mu = vpin.rolling(300, min_periods=30).mean()
        sd = vpin.rolling(300, min_periods=30).std()
        df["viz_vpin_zscore"] = (vpin - mu) / (sd + 1e-8)

    # D. Entropy slope across timescales (per row)
    ent_cols = ["ent_tick_1s", "ent_tick_5s", "ent_tick_10s", "ent_tick_15s",
                "ent_tick_30s", "ent_tick_1m", "ent_tick_15m"]
    log_windows = np.log([1, 5, 10, 15, 30, 60, 900])
    available = [(c, lw) for c, lw in zip(ent_cols, log_windows) if _safe_col(df, c) is not None]
    if len(available) >= 3:
        ent_matrix = np.column_stack([df[c].values for c, _ in available])
        lw = np.array([w for _, w in available])
        # Vectorized linear regression slope per row
        lw_centered = lw - lw.mean()
        denom = np.sum(lw_centered ** 2)
        ent_centered = ent_matrix - ent_matrix.mean(axis=1, keepdims=True)
        df["viz_entropy_slope"] = ent_centered @ lw_centered / (denom + 1e-10)

    # E. Rolling correlation: flow aggressor vs tick entropy
    aggr = _safe_col(df, "flow_aggressor_ratio_5s")
    ent1m = _safe_col(df, "ent_tick_1m")
    if aggr is not None and ent1m is not None:
        df["viz_corr_flow_entropy"] = aggr.rolling(600, min_periods=60).corr(ent1m)

    # F. Rolling correlation: VPIN vs Kyle lambda
    kyle = _safe_col(df, "illiq_kyle_100")
    if vpin is not None and kyle is not None:
        df["viz_corr_vpin_kyle"] = vpin.rolling(600, min_periods=60).corr(kyle)

    return df


# ---------------------------------------------------------------------------
# Page 1 panels (existing)
# ---------------------------------------------------------------------------


def panel_price_spread(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 1: Midprice line + spread area fill."""
    mid = _safe_col(df, "raw_midprice")
    if mid is None:
        _no_data(ax, "No midprice data", "Price + Spread")
        return

    x = df["datetime"]
    ax.plot(x, mid, color=COLORS[0], linewidth=0.8, label="midprice")
    ax.set_ylabel("Midprice", color=COLORS[0], fontsize=8)
    ax.tick_params(axis="y", labelcolor=COLORS[0], labelsize=7)

    spread = _safe_col(df, "raw_spread_bps")
    if spread is not None:
        ax_r = ax.twinx()
        ax_r.fill_between(x, spread, alpha=0.3, color=COLORS[3], label="spread (bps)")
        ax_r.set_ylabel("Spread (bps)", color=COLORS[3], fontsize=8)
        ax_r.tick_params(axis="y", labelcolor=COLORS[3], labelsize=7)
        ax_r.invert_yaxis()
        ax_r.set_facecolor("none")

    ax.set_title("Price + Spread", fontsize=10, color="#c9d1d9")
    ax.grid(True, alpha=0.3)


def panel_book_depth(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 2: Bid/ask depth mirrored + imbalance overlay."""
    bid = _safe_col(df, "raw_bid_depth_5")
    ask = _safe_col(df, "raw_ask_depth_5")
    if bid is None and ask is None:
        _no_data(ax, "No depth data", "Book Depth + Imbalance")
        return

    x = df["datetime"]
    if bid is not None:
        ax.fill_between(x, 0, bid, alpha=0.4, color=COLORS[1], label="bid depth L5")
    if ask is not None:
        ax.fill_between(x, 0, -ask, alpha=0.4, color=COLORS[2], label="ask depth L5")

    ax.axhline(0, color="#8b949e", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Depth (mirrored)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)

    imb = _safe_col(df, "imbalance_qty_l1")
    if imb is not None:
        ax_r = ax.twinx()
        ax_r.plot(x, imb, color=COLORS[4], linewidth=0.6, alpha=0.8, label="imbalance L1")
        ax_r.set_ylabel("Imbalance L1", color=COLORS[4], fontsize=8)
        ax_r.tick_params(axis="y", labelcolor=COLORS[4], labelsize=7)
        ax_r.set_facecolor("none")

    ax.set_title("Book Depth + Imbalance", fontsize=10, color="#c9d1d9")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.3)


def panel_trade_flow(ax: plt.Axes, bars: pd.DataFrame) -> None:
    """Panel 3: Volume bars colored by aggressor + trade count step."""
    vol = _bar_col(bars, "flow_volume_1s", "sum")
    if vol is None:
        _no_data(ax, "No flow data", "Trade Flow")
        return

    x = bars["bar_start"]
    aggr = _bar_col(bars, "flow_aggressor_ratio_5s", "mean")

    # Color by aggressor ratio: green = buyer, red = seller
    if aggr is not None:
        colors = [COLORS[1] if v > 0.5 else COLORS[2] if v < 0.5 else COLORS[3]
                  for v in aggr.fillna(0.5)]
    else:
        colors = COLORS[0]

    width = pd.Timedelta("50s")  # slightly less than 1min for gaps
    ax.bar(x, vol, width=width, color=colors, alpha=0.7, label="volume")
    ax.set_ylabel("Volume (sum)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)

    cnt = _bar_col(bars, "flow_count_1s", "sum")
    if cnt is not None:
        ax_r = ax.twinx()
        ax_r.step(x, cnt, color=COLORS[5], linewidth=0.8, where="mid", label="trade count")
        ax_r.set_ylabel("Count", color=COLORS[5], fontsize=8)
        ax_r.tick_params(axis="y", labelcolor=COLORS[5], labelsize=7)
        ax_r.set_facecolor("none")

    ax.set_title("Trade Flow", fontsize=10, color="#c9d1d9")
    ax.grid(True, alpha=0.3)


def panel_microstructure(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 4: Kyle lambda, spread vol, VPIN — z-scored."""
    features = [
        ("illiq_kyle_100", "Kyle \u03bb", COLORS[0]),
        ("vol_spread_std_1m", "Spread Vol", COLORS[1]),
        ("toxic_vpin_10", "VPIN", COLORS[2]),
    ]
    plotted = False
    for col, label, color in features:
        s = _safe_col(df, col)
        if s is not None:
            ax.plot(df["datetime"], _zscore(s), color=color, linewidth=0.6,
                    alpha=0.8, label=label)
            plotted = True

    if not plotted:
        _no_data(ax, "No microstructure data", "Microstructure Quality")

    ax.set_title("Microstructure Quality (z-scored)", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("Z-score", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.axhline(0, color="#8b949e", linewidth=0.5, linestyle="--")
    ax.legend(loc="upper right", fontsize=6, framealpha=0.3)
    ax.grid(True, alpha=0.3)


def panel_entropy_regime(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 5: Tick entropy + permutation entropy, regime bg fill."""
    x = df["datetime"]
    plotted = False

    tick_ent = _safe_col(df, "ent_tick_1s")
    if tick_ent is not None:
        ax.plot(x, _zscore(tick_ent), color=COLORS[0], linewidth=0.6,
                label="tick entropy 1s")
        plotted = True

    perm_ent = _safe_col(df, "ent_permutation_returns_8")
    if perm_ent is not None:
        ax.plot(x, _zscore(perm_ent), color=COLORS[1], linewidth=0.6,
                label="perm entropy 8")
        plotted = True

    regime = _safe_col(df, "derived_entropy_trend_zscore")
    if regime is not None:
        r = regime.values
        pos = np.where(r > 0, r, 0)
        neg = np.where(r < 0, r, 0)
        ax.fill_between(x, 0, _zscore(pd.Series(pos)), alpha=0.15, color=COLORS[1])
        ax.fill_between(x, 0, _zscore(pd.Series(neg)), alpha=0.15, color=COLORS[2])

    if not plotted:
        _no_data(ax, "No entropy data", "Entropy + Regime")

    ax.set_title("Entropy + Regime", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("Z-score", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.axhline(0, color="#8b949e", linewidth=0.5, linestyle="--")
    ax.legend(loc="upper right", fontsize=6, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def panel_feature_heatmap(ax: plt.Axes, bars: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Panel 6: Top-20 features by variance, normalized heatmap."""
    # Pick numeric feature columns from bars
    skip = {"bar_start", "bar_end", "symbol", "tick_count", "datetime"}
    feat_cols = [c for c in bars.columns if c not in skip
                 and bars[c].dtype in [np.float64, np.float32]
                 and bars[c].notna().mean() > 0.5]

    if len(feat_cols) < 2:
        _no_data(ax, "Insufficient features for heatmap", "Feature Heatmap")
        return

    # Rank by variance, take top 20
    variances = bars[feat_cols].var()
    top_cols = variances.nlargest(20).index.tolist()

    data = bars[top_cols].values.T  # (features, bars)

    # Normalize each feature to [0, 1]
    mins = np.nanmin(data, axis=1, keepdims=True)
    maxs = np.nanmax(data, axis=1, keepdims=True)
    rng = maxs - mins
    rng[rng < 1e-10] = 1.0
    normed = (data - mins) / rng

    # Replace NaN with 0.5 for display
    normed = np.nan_to_num(normed, nan=0.5)

    # Time axis
    if "bar_start" in bars.columns:
        time_labels = pd.to_datetime(bars["bar_start"]).dt.strftime("%H:%M")
    else:
        time_labels = [str(i) for i in range(normed.shape[1])]

    im = ax.pcolormesh(normed, cmap="inferno", vmin=0, vmax=1)

    # Y labels = feature names (shortened)
    short_names = [c.replace("_mean", "").replace("_std", "\u03c3").replace("_sum", "\u03a3")
                   .replace("_last", "\u2020").replace("_slope", "\u2207")
                   for c in top_cols]
    ax.set_yticks(np.arange(len(short_names)) + 0.5)
    ax.set_yticklabels(short_names, fontsize=6)

    # X labels = time
    n_bars = normed.shape[1]
    step = max(1, n_bars // 10)
    ax.set_xticks(np.arange(0, n_bars, step) + 0.5)
    ax.set_xticklabels(time_labels.iloc[::step] if hasattr(time_labels, "iloc")
                       else time_labels[::step], fontsize=7, rotation=45)

    ax.set_title("Feature Heatmap (top-20 by variance)", fontsize=10, color="#c9d1d9")


# ---------------------------------------------------------------------------
# Page 2 panels (new — advanced analytics)
# ---------------------------------------------------------------------------


def panel_toxicity(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 7: VPIN + adverse selection + flow imbalance + toxic index bg."""
    x = df["datetime"]
    plotted = False

    vpin = _safe_col(df, "toxic_vpin_50")
    if vpin is not None:
        ax.plot(x, vpin, color=COLORS[0], linewidth=0.7, label="VPIN(50)")
        ax.axhline(0.5, color=COLORS[0], linewidth=0.4, linestyle=":", alpha=0.5)
        plotted = True

    adv = _safe_col(df, "toxic_adverse_selection")
    if adv is not None:
        ax.plot(x, _zscore(adv), color=COLORS[4], linewidth=0.6, alpha=0.7,
                label="adverse sel (z)")
        plotted = True

    # Toxic index as background shading
    tidx = _safe_col(df, "toxic_index")
    if tidx is not None:
        ax.fill_between(x, 0, tidx, alpha=0.12, color=COLORS[2], label="toxic index")

    # VPIN z-score from derived
    vz = _safe_col(df, "viz_vpin_zscore")
    if vz is not None:
        ax_r = ax.twinx()
        ax_r.plot(x, vz, color=COLORS[3], linewidth=0.5, alpha=0.7, label="VPIN z-score")
        ax_r.set_ylabel("VPIN z-score", color=COLORS[3], fontsize=8)
        ax_r.tick_params(axis="y", labelcolor=COLORS[3], labelsize=7)
        ax_r.axhline(2, color=COLORS[2], linewidth=0.4, linestyle="--", alpha=0.4)
        ax_r.axhline(-2, color=COLORS[1], linewidth=0.4, linestyle="--", alpha=0.4)
        ax_r.set_facecolor("none")

    if not plotted:
        _no_data(ax, "No toxicity data", "Toxicity & Informed Trading")

    ax.set_title("Toxicity & Informed Trading", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("VPIN / z-score", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.3)
    ax.grid(True, alpha=0.3)


def panel_trend_regime(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 8: Hurst exponent + momentum R2 + regime indicator background."""
    x = df["datetime"]
    plotted = False

    hurst = _safe_col(df, "trend_hurst_300")
    if hurst is not None:
        ax.plot(x, hurst, color=COLORS[0], linewidth=0.7, label="Hurst(300)")
        ax.axhline(0.5, color="#8b949e", linewidth=0.5, linestyle="--", alpha=0.6)
        plotted = True

    r2 = _safe_col(df, "trend_momentum_r2_300")
    if r2 is not None:
        ax.plot(x, r2, color=COLORS[4], linewidth=0.6, alpha=0.7, label="R\u00b2(300)")
        plotted = True

    # Regime indicator as background
    regime = _safe_col(df, "derived_regime_indicator")
    if regime is not None:
        r = regime.values
        trending = np.where(r < 0, 1, 0)
        reverting = np.where(r > 0, 1, 0)
        ax.fill_between(x, 0, 1, where=trending.astype(bool),
                        alpha=0.08, color=COLORS[1], transform=ax.get_xaxis_transform(),
                        label="trending")
        ax.fill_between(x, 0, 1, where=reverting.astype(bool),
                        alpha=0.08, color=COLORS[2], transform=ax.get_xaxis_transform(),
                        label="reverting")

    # Momentum as bars on secondary axis
    mom = _safe_col(df, "trend_momentum_300")
    if mom is not None:
        ax_r = ax.twinx()
        ax_r.fill_between(x, 0, mom, alpha=0.2, color=COLORS[3])
        ax_r.set_ylabel("Momentum", color=COLORS[3], fontsize=8)
        ax_r.tick_params(axis="y", labelcolor=COLORS[3], labelsize=7)
        ax_r.set_facecolor("none")

    if not plotted:
        _no_data(ax, "No trend data", "Trend Regime & Persistence")

    ax.set_title("Trend Regime (H>0.5=trend, H<0.5=revert)", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("Hurst / R\u00b2", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.3)
    ax.grid(True, alpha=0.3)


def panel_funding_oi(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 9: Funding rate + open interest + OI change + premium."""
    x = df["datetime"]
    plotted = False

    funding = _safe_col(df, "ctx_funding_rate")
    if funding is not None:
        ax.plot(x, funding * 100, color=COLORS[0], linewidth=0.7, label="funding (%)")
        ax.axhline(0, color="#8b949e", linewidth=0.4, linestyle="--")
        plotted = True

    premium = _safe_col(df, "ctx_premium_bps")
    if premium is not None:
        ax.plot(x, premium, color=COLORS[4], linewidth=0.5, alpha=0.6, label="premium (bps)")
        plotted = True

    ax.set_ylabel("Funding % / Premium bps", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)

    # OI on secondary axis
    oi = _safe_col(df, "ctx_open_interest")
    if oi is not None:
        ax_r = ax.twinx()
        ax_r.fill_between(x, oi, alpha=0.15, color=COLORS[1])
        ax_r.plot(x, oi, color=COLORS[1], linewidth=0.5, alpha=0.5, label="OI")
        ax_r.set_ylabel("Open Interest", color=COLORS[1], fontsize=8)
        ax_r.tick_params(axis="y", labelcolor=COLORS[1], labelsize=7)
        ax_r.set_facecolor("none")
        plotted = True

    if not plotted:
        _no_data(ax, "No context data", "Funding & Open Interest")

    ax.set_title("Funding & Open Interest", fontsize=10, color="#c9d1d9")
    ax.legend(loc="upper left", fontsize=6, framealpha=0.3)
    ax.grid(True, alpha=0.3)


def panel_illiquidity(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 10: Kyle lambda short/long + ratio + composite."""
    x = df["datetime"]
    plotted = False

    kyle100 = _safe_col(df, "illiq_kyle_100")
    kyle500 = _safe_col(df, "illiq_kyle_500")
    if kyle100 is not None:
        ax.plot(x, _zscore(kyle100), color=COLORS[0], linewidth=0.7,
                label="Kyle \u03bb(100) z")
        plotted = True
    if kyle500 is not None:
        ax.plot(x, _zscore(kyle500), color=COLORS[5], linewidth=0.6, alpha=0.7,
                label="Kyle \u03bb(500) z")
        plotted = True

    # Composite as shaded area
    comp = _safe_col(df, "illiq_composite")
    if comp is not None:
        ax.fill_between(x, 0, _zscore(comp), alpha=0.1, color=COLORS[3])

    # Kyle ratio on secondary axis
    ratio = _safe_col(df, "illiq_kyle_ratio")
    if ratio is not None:
        ax_r = ax.twinx()
        ax_r.plot(x, ratio, color=COLORS[2], linewidth=0.5, alpha=0.6,
                  label="kyle ratio (100/500)")
        ax_r.axhline(1.0, color=COLORS[2], linewidth=0.4, linestyle=":", alpha=0.4)
        ax_r.set_ylabel("Kyle Ratio", color=COLORS[2], fontsize=8)
        ax_r.tick_params(axis="y", labelcolor=COLORS[2], labelsize=7)
        ax_r.set_facecolor("none")
        plotted = True

    if not plotted:
        _no_data(ax, "No illiquidity data", "Illiquidity Dynamics")

    ax.set_title("Illiquidity Dynamics (ratio>1 = fragile)", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("Z-score", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.axhline(0, color="#8b949e", linewidth=0.4, linestyle="--")
    ax.legend(loc="upper left", fontsize=6, framealpha=0.3)
    ax.grid(True, alpha=0.3)


def panel_multiscale_entropy(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 11: Tick entropy at multiple timescales + entropy slope."""
    x = df["datetime"]
    plotted = False

    ent_specs = [
        ("ent_tick_1s", "1s", COLORS[0]),
        ("ent_tick_5s", "5s", COLORS[1]),
        ("ent_tick_30s", "30s", COLORS[3]),
        ("ent_tick_1m", "1m", COLORS[4]),
        ("ent_tick_15m", "15m", COLORS[5]),
    ]
    for col, label, color in ent_specs:
        s = _safe_col(df, col)
        if s is not None:
            ax.plot(x, s, color=color, linewidth=0.5, alpha=0.7, label=label)
            plotted = True

    ax.set_ylabel("Tick Entropy", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)

    # Entropy slope on secondary axis
    slope = _safe_col(df, "viz_entropy_slope")
    if slope is not None:
        ax_r = ax.twinx()
        ax_r.plot(x, slope, color=COLORS[2], linewidth=0.6, alpha=0.7,
                  label="entropy slope")
        ax_r.axhline(0, color=COLORS[2], linewidth=0.3, linestyle=":", alpha=0.4)
        ax_r.set_ylabel("Entropy Slope", color=COLORS[2], fontsize=8)
        ax_r.tick_params(axis="y", labelcolor=COLORS[2], labelsize=7)
        ax_r.set_facecolor("none")

    if not plotted:
        _no_data(ax, "No entropy data", "Multi-scale Entropy")

    ax.set_title("Multi-scale Entropy (1s \u2192 15m)", fontsize=10, color="#c9d1d9")
    ax.legend(loc="upper left", fontsize=6, framealpha=0.3, ncol=3)
    ax.grid(True, alpha=0.3)


def panel_cross_interactions(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 12: Rolling correlations + derived interaction scores."""
    x = df["datetime"]
    plotted = False

    # Rolling correlations
    corr_fe = _safe_col(df, "viz_corr_flow_entropy")
    if corr_fe is not None:
        ax.plot(x, corr_fe, color=COLORS[0], linewidth=0.6, alpha=0.8,
                label="corr(flow, entropy)")
        plotted = True

    corr_vk = _safe_col(df, "viz_corr_vpin_kyle")
    if corr_vk is not None:
        ax.plot(x, corr_vk, color=COLORS[4], linewidth=0.6, alpha=0.8,
                label="corr(VPIN, Kyle)")
        plotted = True

    ax.axhline(0, color="#8b949e", linewidth=0.4, linestyle="--")
    ax.set_ylabel("Rolling Correlation", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(-1.05, 1.05)

    # Derived scores on secondary axis
    ax_r = ax.twinx()
    informed = _safe_col(df, "derived_informed_trend_score")
    if informed is not None:
        ax_r.plot(x, _zscore(informed), color=COLORS[1], linewidth=0.5, alpha=0.6,
                  label="informed trend (z)")
        plotted = True

    chop = _safe_col(df, "derived_toxic_chop_score")
    if chop is not None:
        ax_r.plot(x, _zscore(chop), color=COLORS[2], linewidth=0.5, alpha=0.6,
                  label="toxic chop (z)")
        plotted = True

    ax_r.set_ylabel("Interaction Z-score", fontsize=8)
    ax_r.tick_params(axis="y", labelsize=7)
    ax_r.set_facecolor("none")
    ax_r.legend(loc="lower right", fontsize=6, framealpha=0.3)

    if not plotted:
        _no_data(ax, "No interaction data", "Cross-Feature Interactions")

    ax.set_title("Cross-Feature Interactions", fontsize=10, color="#c9d1d9")
    ax.legend(loc="upper left", fontsize=6, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_page(
    panels: list[tuple],
    df_sym: pd.DataFrame,
    bars: pd.DataFrame,
    symbol: str,
    title_suffix: str,
    output_dir: Path,
    filename: str,
    page_label: str,
) -> Path:
    """Render a multi-panel figure page."""
    apply_style()

    ds = df_sym.iloc[::10].copy()
    n = len(panels)

    fig = plt.figure(figsize=(16, 4 * n))
    ratios = [1] * (n - 1) + [1.2]
    gs = gridspec.GridSpec(n, 1, height_ratios=ratios, hspace=0.25)

    axes = [fig.add_subplot(gs[i]) for i in range(n)]

    # Hide x labels on upper panels
    for a in axes[:max(1, n - 2)]:
        plt.setp(a.get_xticklabels(), visible=False)

    for i, (func, data_key) in enumerate(panels):
        ax = axes[i]
        try:
            if data_key == "ds":
                func(ax, ds)
            elif data_key == "bars":
                func(ax, bars)
            elif data_key == "bars+raw":
                func(ax, bars, df_sym)
            else:
                func(ax, ds)
        except Exception as e:
            log.warning("Panel %s failed: %s", func.__name__, e)
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color=COLORS[2])

    # Format time axes on panels that use bar data
    for ax in axes:
        if ax.get_xlabel() == "" and len(ax.get_xticks()) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.suptitle(f"{symbol} — {title_suffix} [{page_label}]",
                 fontsize=14, color="#c9d1d9", y=0.995)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


PAGE1_PANELS = [
    (panel_price_spread, "ds"),
    (panel_book_depth, "ds"),
    (panel_trade_flow, "bars"),
    (panel_microstructure, "ds"),
    (panel_entropy_regime, "ds"),
    (panel_feature_heatmap, "bars+raw"),
]

PAGE2_PANELS = [
    (panel_price_spread, "ds"),
    (panel_toxicity, "ds"),
    (panel_trend_regime, "ds"),
    (panel_funding_oi, "ds"),
    (panel_illiquidity, "ds"),
    (panel_multiscale_entropy, "ds"),
    (panel_cross_interactions, "ds"),
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_visualization(
    df: pd.DataFrame,
    symbol: str,
    output_dir: Path,
    timeframe: str = "1min",
    window_minutes: Optional[int] = None,
    page: str = "all",
) -> list[Path]:
    """Generate microstructure snapshot(s) for one symbol.

    page: "1" = existing panels, "2" = new panels, "all" = both pages.
    """
    df_sym = df[df["symbol"] == symbol].copy()
    if df_sym.empty:
        raise ValueError(f"No data for symbol {symbol}")

    df_sym = df_sym.sort_values("timestamp_ns").reset_index(drop=True)
    df_sym = _ensure_datetime(df_sym)

    # Compute derived features for page 1 (advanced)
    if page in ("1", "all"):
        df_sym = _compute_derived_viz(df_sym)

    if window_minutes is None:
        return _render_single(df_sym, symbol, output_dir, timeframe, page,
                              "Microstructure Snapshot")

    return _render_windowed(df_sym, symbol, output_dir, timeframe, page,
                            window_minutes)


def _render_single(
    df_sym: pd.DataFrame,
    symbol: str,
    output_dir: Path,
    timeframe: str,
    page: str,
    title_suffix: str,
) -> list[Path]:
    """Render one or two pages for the full data range."""
    bars = aggregate_bars(df_sym, timeframe)
    bars["bar_start"] = pd.to_datetime(bars["bar_start"])
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outputs = []

    if page in ("0", "all"):
        out = _render_page(PAGE1_PANELS, df_sym, bars, symbol,
                           title_suffix, output_dir,
                           f"15m_viz_{symbol}__{ts_str}__0.png", "Page 0")
        outputs.append(out)

    if page in ("1", "all"):
        out = _render_page(PAGE2_PANELS, df_sym, bars, symbol,
                           title_suffix, output_dir,
                           f"15m_viz_{symbol}__{ts_str}__1.png", "Page 1")
        outputs.append(out)

    return outputs


def _render_windowed(
    df_sym: pd.DataFrame,
    symbol: str,
    output_dir: Path,
    timeframe: str,
    page: str,
    window_minutes: int,
) -> list[Path]:
    """Split data into consecutive windows and render each."""
    ts_min = df_sym["timestamp_ns"].min()
    ts_max = df_sym["timestamp_ns"].max()
    window_ns = int(window_minutes * 60 * 1e9)

    edges = list(range(int(ts_min), int(ts_max) + 1, window_ns))
    if edges[-1] < ts_max:
        edges.append(int(ts_max) + 1)

    outputs = []
    for i in range(len(edges) - 1):
        t0, t1 = edges[i], edges[i + 1]
        chunk = df_sym[(df_sym["timestamp_ns"] >= t0) & (df_sym["timestamp_ns"] < t1)]
        if len(chunk) < 100:
            log.warning("Window %d: only %d rows, skipping", i + 1, len(chunk))
            continue

        chunk = chunk.reset_index(drop=True)
        try:
            bars = aggregate_bars(chunk, timeframe)
        except ValueError:
            log.warning("Window %d: bar aggregation failed, skipping", i + 1)
            continue
        bars["bar_start"] = pd.to_datetime(bars["bar_start"])

        win_start = pd.to_datetime(t0, unit="ns").strftime("%H:%M")
        win_end = pd.to_datetime(t1, unit="ns").strftime("%H:%M")
        title = f"{window_minutes}min Window {i + 1} ({win_start}\u2013{win_end})"
        base = f"15m_viz_{symbol}_w{i + 1:02d}_{win_start.replace(':', '')}_{win_end.replace(':', '')}__{{pg}}.png"

        if page in ("0", "all"):
            out = _render_page(PAGE1_PANELS, chunk, bars, symbol, title,
                               output_dir, base.format(pg="0"), "Page 0")
            outputs.append(out)
        if page in ("1", "all"):
            out = _render_page(PAGE2_PANELS, chunk, bars, symbol, title,
                               output_dir, base.format(pg="1"), "Page 1")
            outputs.append(out)

    return outputs


def print_summary(df: pd.DataFrame, symbols: list[str], data_dir: str) -> None:
    """Print stdout summary with anomaly detection."""
    ts_min = pd.to_datetime(df["timestamp_ns"].min(), unit="ns")
    ts_max = pd.to_datetime(df["timestamp_ns"].max(), unit="ns")
    dur = (df["timestamp_ns"].max() - df["timestamp_ns"].min()) / 1e9

    print()
    print("=" * 60)
    print("  15-Minute Visual Health Check")
    print("=" * 60)
    print(f"  Data:     {data_dir}")
    print(f"  Window:   {ts_min.strftime('%H:%M:%S')} \u2014 {ts_max.strftime('%H:%M:%S')} UTC")
    print(f"  Duration: {dur:.0f}s ({dur / 60:.1f}min)")
    print(f"  Rows:     {len(df):,}")
    print(f"  Symbols:  {', '.join(symbols)}")
    print()

    for sym in symbols:
        df_sym = df[df["symbol"] == sym]
        print(f"  {sym}: {len(df_sym):,} rows")

    # Anomaly detection
    all_warnings = []
    for sym in symbols:
        df_sym = df[df["symbol"] == sym]
        all_warnings.extend(_detect_anomalies(df_sym, sym))

    if all_warnings:
        print()
        print("  Anomalies:")
        for w in all_warnings:
            print(f"    [-] {w}")
    else:
        print()
        print("  [+] No anomalies detected")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        prog="15m_visualize",
        description="15-Minute Visual Health Check — microstructure snapshot",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to parquet data directory")
    parser.add_argument("--latest", action="store_true",
                        help="Use data from latest 15m experiment (reports/smoke_test/latest)")
    parser.add_argument("--symbol", type=str, default="BTC",
                        help="Symbol to visualize, or 'all' (default: BTC)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for PNG files")
    parser.add_argument("--timeframe", type=str, default="1min",
                        help="Bar aggregation timeframe (default: 1min)")
    parser.add_argument("--window", type=int, default=None, metavar="MINUTES",
                        help="Split data into N-minute windows (e.g. --window 15)")
    parser.add_argument("--page", type=str, default="all", choices=["0", "1", "all"],
                        help="Which page(s) to generate: 0=basic, 1=advanced, all=both")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Resolve data source
    data_file = None
    if args.data_dir is None or args.latest:
        ref_path = DEFAULT_OUTPUT / "latest" / "data_ref.json"
        if ref_path.exists():
            ref = json.loads(ref_path.read_text())
            # Warn if experiment is stale (> 1 hour old)
            if "created" in ref:
                created = datetime.fromisoformat(ref["created"])
                age = datetime.now(timezone.utc) - created
                age_hours = age.total_seconds() / 3600
                if age_hours > 1:
                    days = int(age.days)
                    hours = int((age.total_seconds() % 86400) / 3600)
                    age_str = f"{days}d {hours}h" if days else f"{hours}h"
                    log.warning(
                        "Experiment is %s old (created %s). "
                        "Run 'nat 15m' to collect fresh data.",
                        age_str, created.strftime("%Y-%m-%d %H:%M"),
                    )
            # Prefer 15m__ data file from experiment dir (check ref or glob)
            if "data_file" in ref and Path(ref["data_file"]).exists():
                data_file = Path(ref["data_file"])
            else:
                args.data_dir = Path(ref.get("source_dir", ref.get("data_dir", "")))
            if args.output is None:
                args.output = DEFAULT_OUTPUT / "latest"
            log.info("Using latest experiment: %s (%d rows)",
                     data_file or args.data_dir, ref["rows"])
        else:
            parser.error("--data-dir required (no latest experiment found; run 'nat 15m' first)")

    if args.output is None:
        args.output = DEFAULT_OUTPUT

    if data_file is not None:
        log.info("Loading 15m data from %s", data_file)
        df = pd.read_parquet(data_file)
    else:
        log.info("Loading data from %s", args.data_dir)
        df = load_parquet(args.data_dir)
    log.info("Loaded %d rows", len(df))

    symbols = get_symbols(df)
    if args.symbol.lower() == "all":
        targets = symbols
    else:
        if args.symbol not in symbols:
            log.error("Symbol %s not in data (available: %s)", args.symbol, symbols)
            sys.exit(1)
        targets = [args.symbol]

    print_summary(df, symbols, str(args.data_dir))

    for sym in targets:
        log.info("Generating visualization for %s (page=%s)...", sym, args.page)
        outputs = generate_visualization(
            df, sym, args.output, args.timeframe, args.window, args.page,
        )
        for out in outputs:
            print(f"  Saved: {out}")

    print()


if __name__ == "__main__":
    main()
