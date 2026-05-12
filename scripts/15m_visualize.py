"""15-Minute Visual Health Check — market microstructure snapshot.

Companion to 15m_test.py. Produces a 6-panel PNG per symbol showing
price, depth, flow, microstructure quality, entropy regime, and a
feature heatmap for quick visual QA.

Usage:
    python3 scripts/15m_visualize.py --data-dir data/features/2026-05-12-clean --symbol BTC
    python3 scripts/15m_visualize.py --data-dir data/features/2026-05-12-clean --symbol all
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(Path(__file__).parent))
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
# Panel functions
# ---------------------------------------------------------------------------


def panel_price_spread(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 1: Midprice line + spread area fill."""
    mid = _safe_col(df, "raw_midprice")
    if mid is None:
        ax.text(0.5, 0.5, "No midprice data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#8b949e")
        ax.set_title("Price + Spread", fontsize=10, color="#c9d1d9")
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
        ax.text(0.5, 0.5, "No depth data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#8b949e")
        ax.set_title("Book Depth + Imbalance", fontsize=10, color="#c9d1d9")
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
        ax.text(0.5, 0.5, "No flow data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#8b949e")
        ax.set_title("Trade Flow", fontsize=10, color="#c9d1d9")
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
        ("illiq_kyle_100", "Kyle λ", COLORS[0]),
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
        ax.text(0.5, 0.5, "No microstructure data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#8b949e")

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
        ax.text(0.5, 0.5, "No entropy data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#8b949e")

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
        ax.text(0.5, 0.5, "Insufficient features for heatmap", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#8b949e")
        ax.set_title("Feature Heatmap", fontsize=10, color="#c9d1d9")
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
    short_names = [c.replace("_mean", "").replace("_std", "σ").replace("_sum", "Σ")
                   .replace("_last", "†").replace("_slope", "∇")
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
# Orchestrator
# ---------------------------------------------------------------------------


def generate_visualization(
    df: pd.DataFrame,
    symbol: str,
    output_dir: Path,
    timeframe: str = "1min",
) -> Path:
    """Generate 6-panel microstructure snapshot for one symbol."""
    apply_style()

    # Filter to symbol
    df_sym = df[df["symbol"] == symbol].copy()
    if df_sym.empty:
        raise ValueError(f"No data for symbol {symbol}")

    df_sym = df_sym.sort_values("timestamp_ns").reset_index(drop=True)
    df_sym = _ensure_datetime(df_sym)

    # Aggregate bars for panels 3 and 6
    bars = aggregate_bars(df_sym, timeframe)
    bars["bar_start"] = pd.to_datetime(bars["bar_start"])

    # Downsample tick data for plotting (every 10th row ~ 1/sec)
    ds = df_sym.iloc[::10].copy()

    # Layout: 6 panels, shared x on 0-4, independent heatmap
    fig = plt.figure(figsize=(16, 24))
    gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1.2],
                           hspace=0.25)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2])  # bars have different x
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    ax5 = fig.add_subplot(gs[5])  # heatmap has different x

    # Hide x labels on shared panels except the last shared one
    for a in [ax0, ax1, ax3]:
        plt.setp(a.get_xticklabels(), visible=False)

    # Draw panels — each wrapped to handle missing data gracefully
    panels = [
        (panel_price_spread, ax0, [ds]),
        (panel_book_depth, ax1, [ds]),
        (panel_trade_flow, ax2, [bars]),
        (panel_microstructure, ax3, [ds]),
        (panel_entropy_regime, ax4, [ds]),
        (panel_feature_heatmap, ax5, [bars, df_sym]),
    ]

    for func, ax, args in panels:
        try:
            func(ax, *args)
        except Exception as e:
            log.warning("Panel %s failed: %s", func.__name__, e)
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color=COLORS[2])

    # Format bar panel x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fig.suptitle(f"{symbol} — 15-Minute Microstructure Snapshot",
                 fontsize=14, color="#c9d1d9", y=0.995)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"15m_viz_{symbol}_{ts_str}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


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
    print(f"  Window:   {ts_min.strftime('%H:%M:%S')} — {ts_max.strftime('%H:%M:%S')} UTC")
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
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to parquet data directory")
    parser.add_argument("--symbol", type=str, default="BTC",
                        help="Symbol to visualize, or 'all' (default: BTC)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory for PNG files")
    parser.add_argument("--timeframe", type=str, default="1min",
                        help="Bar aggregation timeframe (default: 1min)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

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
        log.info("Generating visualization for %s ...", sym)
        out = generate_visualization(df, sym, args.output, args.timeframe)
        print(f"  Saved: {out}")

    print()


if __name__ == "__main__":
    main()
