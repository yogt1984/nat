"""Paper Trade Snapshot — visual trade summary per symbol per day.

Produces a single-page 5-panel PNG per symbol showing trade intuition:
  Panel 1: Price + trade entry/exit markers
  Panel 2: Cumulative PnL curve
  Panel 3: Per-trade waterfall
  Panel 4: Signal vs PnL scatter
  Panel 5: Market context at trade times

Usage:
    python3 scripts/trade_visualize.py --latest --symbol BTC
    python3 scripts/trade_visualize.py --date 2026-05-23 --symbol all
    python3 scripts/trade_visualize.py --date-range 2026-05-20 2026-05-23 --symbol ETH
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).parent))
from viz.features import STYLE, COLORS, apply_style

log = logging.getLogger("trade_viz")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "trade_viz"
TRADE_DIR = ROOT / "data" / "paper_trades"
FEATURES_DIR = ROOT / "data" / "features"

BAR_SECONDS = 300
HORIZON_BARS = 20
FEE_BPS = 1.61

LOAD_COLUMNS = [
    "timestamp_ns", "symbol", "raw_midprice",
    "raw_spread_bps", "raw_ask_depth_5", "flow_vwap_deviation",
]

# Marker colors
WIN_COLOR = "#3fb950"
LOSS_COLOR = "#f85149"
LONG_COLOR = COLORS[0]   # blue
SHORT_COLOR = COLORS[3]  # orange


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def discover_trade_dates(trades_dir: Path, symbol: str | None = None) -> list[str]:
    """Find available trade dates, optionally filtered by symbol."""
    dates = set()
    for f in trades_dir.iterdir():
        if f.suffix != ".json" or f.name == "batch_report.json":
            continue
        parts = f.stem.split("_", 1)
        if len(parts) == 2:
            date_str, sym = parts
            if symbol is None or sym == symbol:
                dates.add(date_str)
    return sorted(dates)


def load_trades(trades_dir: Path, date_str: str, symbol: str) -> list[dict]:
    path = trades_dir / f"{date_str}_{symbol}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_batch_summary(trades_dir: Path, date_str: str, symbol: str) -> dict | None:
    path = trades_dir / "batch_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        report = json.load(f)
    results = report.get("results", {}).get(symbol, {})
    if not results:
        return None
    for day in results.get("daily", []):
        if day.get("date") == date_str:
            return day
    return None


def load_price_bars(
    features_dir: Path, date_str: str, symbol: str,
) -> pd.DataFrame | None:
    """Load feature parquet for a date and aggregate to 5-min bars."""
    date_path = features_dir / date_str
    if not date_path.is_dir():
        return None
    files = sorted(f for f in date_path.iterdir() if f.suffix == ".parquet")
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            tbl = pq.read_table(str(f))
            df = tbl.to_pandas()
            cols = [c for c in LOAD_COLUMNS if c in df.columns]
            df = df[cols]
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol].copy()
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    ticks = pd.concat(dfs, ignore_index=True).sort_values("timestamp_ns").reset_index(drop=True)
    return _aggregate_to_bars(ticks)


def _aggregate_to_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticks to 5-min bars (mirrors paper_trader.aggregate_to_bars)."""
    bar_ns = BAR_SECONDS * 1_000_000_000
    ticks = ticks.copy()
    ticks["bar_id"] = ticks["timestamp_ns"].values // bar_ns
    agg = {
        "timestamp_ns": ("timestamp_ns", "first"),
        "midprice_open": ("raw_midprice", "first"),
        "midprice_high": ("raw_midprice", "max"),
        "midprice_low": ("raw_midprice", "min"),
        "midprice_last": ("raw_midprice", "last"),
        "spread_bps_last": ("raw_spread_bps", "last"),
        "depth_5_std": ("raw_ask_depth_5", "std"),
        "n_ticks": ("raw_midprice", "count"),
    }
    if "flow_vwap_deviation" in ticks.columns:
        agg["vwap_deviation_std"] = ("flow_vwap_deviation", "std")
    bars = ticks.groupby("bar_id").agg(**agg).reset_index(drop=True)
    bars = bars[bars["n_ticks"] >= 10].reset_index(drop=True)
    bars["depth_5_std"] = bars["depth_5_std"].fillna(0.0)
    if "vwap_deviation_std" in bars.columns:
        bars["vwap_deviation_std"] = bars["vwap_deviation_std"].fillna(0.0)
    bars["datetime"] = pd.to_datetime(bars["timestamp_ns"], unit="ns")
    return bars


def _no_data(ax: plt.Axes, msg: str, title: str) -> None:
    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="#8b949e")
    ax.set_title(title, fontsize=10, color="#c9d1d9")


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def panel_price_trades(ax: plt.Axes, trades: list[dict], bars: pd.DataFrame | None) -> None:
    """Panel 1: Price line with trade entry/exit markers."""
    if bars is None or bars.empty:
        # Trade-only mode: plot entry prices
        if not trades:
            _no_data(ax, "No trade or price data", "Price + Trades")
            return
        idxs = [t["bar_idx"] for t in trades]
        prices = [t["entry_price"] for t in trades]
        ax.plot(idxs, prices, color="#8b949e", linewidth=0.5, alpha=0.5)
    else:
        ax.plot(bars.index, bars["midprice_last"], color=COLORS[0],
                linewidth=0.8, label="midprice", zorder=1)
        # Spread on secondary axis
        if "spread_bps_last" in bars.columns:
            ax_r = ax.twinx()
            ax_r.fill_between(bars.index, 0, bars["spread_bps_last"],
                              alpha=0.15, color=COLORS[3])
            ax_r.set_ylabel("spread (bps)", color=COLORS[3], fontsize=8)
            ax_r.tick_params(axis="y", labelcolor=COLORS[3], labelsize=7)
            ax_r.set_facecolor("none")

    # Trade markers
    for t in trades:
        bi = t["bar_idx"]
        ei = t.get("exit_bar_idx", bi + HORIZON_BARS)
        win = t.get("net_bps", 0) > 0
        is_long = t["direction"] == 1
        color = WIN_COLOR if win else LOSS_COLOR
        marker = "^" if is_long else "v"
        fill = color if win else "none"

        entry_price = t["entry_price"]
        exit_price = t.get("exit_price", entry_price)

        ax.scatter(bi, entry_price, marker=marker, s=30, c=fill,
                   edgecolors=color, linewidths=0.8, zorder=3)
        ax.plot([bi, ei], [entry_price, exit_price],
                color=color, linewidth=0.4, alpha=0.4, zorder=2)

    ax.set_title("Price + Trade Markers", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("Price", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.3)


def panel_cumulative_pnl(ax: plt.Axes, trades: list[dict]) -> None:
    """Panel 2: Cumulative net PnL (bps) with gross overlay."""
    if not trades:
        _no_data(ax, "No trades", "Cumulative PnL")
        return

    sorted_trades = sorted(trades, key=lambda t: t["bar_idx"])
    bar_idxs = [t["bar_idx"] for t in sorted_trades]
    net_cum = np.cumsum([t.get("net_bps", 0) for t in sorted_trades])
    gross_cum = np.cumsum([t.get("gross_bps", 0) for t in sorted_trades])

    ax.step(bar_idxs, net_cum, where="post", color=COLORS[0], linewidth=1.2,
            label="net", zorder=2)
    ax.fill_between(bar_idxs, 0, net_cum, step="post",
                    where=net_cum >= 0, color=WIN_COLOR, alpha=0.15, interpolate=True)
    ax.fill_between(bar_idxs, 0, net_cum, step="post",
                    where=net_cum < 0, color=LOSS_COLOR, alpha=0.15, interpolate=True)
    ax.step(bar_idxs, gross_cum, where="post", color=COLORS[3],
            linewidth=0.7, linestyle="--", alpha=0.7, label="gross")
    ax.axhline(0, color="#8b949e", linewidth=0.4, linestyle="--")

    # Annotate final value
    total = net_cum[-1]
    n = len(trades)
    ax.text(0.98, 0.95, f"net {total:+.1f} bps | {n} trades",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#c9d1d9",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", alpha=0.8))

    ax.set_title("Cumulative PnL", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("cumul. bps", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.3)
    ax.grid(True, alpha=0.3)


def panel_trade_waterfall(ax: plt.Axes, trades: list[dict]) -> None:
    """Panel 3: Per-trade PnL bars (waterfall)."""
    if not trades:
        _no_data(ax, "No trades", "Per-Trade PnL")
        return

    sorted_trades = sorted(trades, key=lambda t: t["bar_idx"])
    bar_idxs = [t["bar_idx"] for t in sorted_trades]
    pnls = [t.get("net_bps", 0) for t in sorted_trades]
    colors = [WIN_COLOR if p > 0 else LOSS_COLOR for p in pnls]

    ax.bar(range(len(pnls)), pnls, color=colors, width=0.8, alpha=0.8)
    ax.axhline(0, color="#8b949e", linewidth=0.4)
    ax.axhline(-FEE_BPS, color=COLORS[3], linewidth=0.6, linestyle=":",
               alpha=0.6, label=f"fee hurdle ({FEE_BPS} bps)")

    # Win rate annotation
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / len(pnls) if pnls else 0
    mean_pnl = np.mean(pnls)
    ax.text(0.98, 0.95, f"WR {wr:.0%} | mean {mean_pnl:+.1f} bps",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#c9d1d9",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", alpha=0.8))

    # Sparse x-ticks
    n = len(pnls)
    if n > 40:
        step = max(1, n // 20)
        ticks = list(range(0, n, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(bar_idxs[i]) for i in ticks], fontsize=6)
    else:
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(bi) for bi in bar_idxs], fontsize=6, rotation=45)

    ax.set_title("Per-Trade PnL", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("net bps", fontsize=8)
    ax.set_xlabel("bar idx", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")


def panel_signal_scatter(ax: plt.Axes, trades: list[dict]) -> None:
    """Panel 4: Signal value vs net PnL scatter with regression."""
    if not trades:
        _no_data(ax, "No trades", "Signal vs PnL")
        return

    signals = np.array([t["signal_value"] for t in trades])
    pnls = np.array([t.get("net_bps", 0) for t in trades])
    dirs = np.array([t["direction"] for t in trades])

    longs = dirs == 1
    shorts = dirs == -1

    if longs.any():
        ax.scatter(signals[longs], pnls[longs], c=LONG_COLOR, s=20, alpha=0.6,
                   label="long", zorder=2)
    if shorts.any():
        ax.scatter(signals[shorts], pnls[shorts], c=SHORT_COLOR, s=20, alpha=0.6,
                   label="short", zorder=2)

    # Regression line
    if len(signals) > 2:
        valid = np.isfinite(signals) & np.isfinite(pnls)
        if valid.sum() > 2:
            m, b = np.polyfit(signals[valid], pnls[valid], 1)
            x_fit = np.linspace(signals[valid].min(), signals[valid].max(), 50)
            ax.plot(x_fit, m * x_fit + b, color="#8b949e", linewidth=1,
                    linestyle="--", alpha=0.7)
            r = np.corrcoef(signals[valid], pnls[valid])[0, 1]
            ax.text(0.98, 0.95, f"r = {r:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color="#c9d1d9",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22", alpha=0.8))

    ax.axhline(0, color="#8b949e", linewidth=0.4, linestyle="--")
    ax.axvline(0, color="#8b949e", linewidth=0.4, linestyle="--")

    ax.set_title("Signal vs PnL", fontsize=10, color="#c9d1d9")
    ax.set_xlabel("signal value", fontsize=8)
    ax.set_ylabel("net bps", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.3)
    ax.grid(True, alpha=0.3)


def panel_market_context(ax: plt.Axes, trades: list[dict], bars: pd.DataFrame | None) -> None:
    """Panel 5: Spread/depth/vwap_dev with trade-entry points highlighted."""
    if bars is None or bars.empty:
        _no_data(ax, "No feature data (parquet missing)", "Market Context")
        return

    x = bars.index

    # Spread
    if "spread_bps_last" in bars.columns:
        s = bars["spread_bps_last"]
        ax.plot(x, s, color=COLORS[0], linewidth=0.7, alpha=0.7, label="spread (bps)")

    # Depth volatility
    ax2 = ax.twinx()
    if "depth_5_std" in bars.columns:
        d = bars["depth_5_std"]
        ax2.plot(x, d, color=COLORS[1], linewidth=0.7, alpha=0.7, label="depth vol")
        ax2.set_ylabel("depth vol", color=COLORS[1], fontsize=8)
        ax2.tick_params(axis="y", labelcolor=COLORS[1], labelsize=7)
        ax2.set_facecolor("none")

    # VWAP deviation
    if "vwap_deviation_std" in bars.columns:
        v = bars["vwap_deviation_std"]
        ax.plot(x, v * 100, color=COLORS[4], linewidth=0.6, alpha=0.5,
                label="vwap dev (x100)")

    # Highlight trade entry bars
    max_bar = len(bars) - 1
    for t in trades:
        bi = t["bar_idx"]
        if bi > max_bar:
            continue
        win = t.get("net_bps", 0) > 0
        color = WIN_COLOR if win else LOSS_COLOR
        ax.axvline(bi, color=color, linewidth=0.3, alpha=0.3)

    ax.set_title("Market Context at Trade Times", fontsize=10, color="#c9d1d9")
    ax.set_ylabel("spread / vwap dev", fontsize=8)
    ax.set_xlabel("bar idx", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.3)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_trade_snapshot(
    trades: list[dict],
    bars: pd.DataFrame | None,
    symbol: str,
    date_str: str,
    summary: dict | None,
    output_dir: Path,
) -> Path:
    """Render 5-panel trade snapshot PNG."""
    apply_style()

    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(5, 1, height_ratios=[1.5, 1, 1, 1, 1.2], hspace=0.28)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    # Hide x labels on upper panels
    for a in axes[:3]:
        plt.setp(a.get_xticklabels(), visible=False)

    # Render panels
    panel_funcs = [
        lambda ax: panel_price_trades(ax, trades, bars),
        lambda ax: panel_cumulative_pnl(ax, trades),
        lambda ax: panel_trade_waterfall(ax, trades),
        lambda ax: panel_signal_scatter(ax, trades),
        lambda ax: panel_market_context(ax, trades, bars),
    ]
    for i, func in enumerate(panel_funcs):
        try:
            func(axes[i])
        except Exception as e:
            log.warning("Panel %d failed: %s", i, e)
            axes[i].text(0.5, 0.5, f"Error: {e}", transform=axes[i].transAxes,
                         ha="center", va="center", fontsize=10, color=COLORS[2])

    # Suptitle
    n = len(trades)
    total = sum(t.get("net_bps", 0) for t in trades)
    wins = sum(1 for t in trades if t.get("net_bps", 0) > 0)
    wr = wins / n if n else 0
    fig.suptitle(
        f"{symbol} \u2014 Paper Trades {date_str}  [{n} trades | net {total:+.0f} bps | WR {wr:.0%}]",
        fontsize=13, color="#c9d1d9", y=0.995,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"trade_viz_{symbol}_{date_str}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def print_summary(trades: list[dict], symbol: str, date_str: str) -> None:
    """Print compact trade summary to stdout."""
    n = len(trades)
    if n == 0:
        print(f"  {symbol} {date_str}: no trades")
        return
    total = sum(t.get("net_bps", 0) for t in trades)
    wins = sum(1 for t in trades if t.get("net_bps", 0) > 0)
    n_long = sum(1 for t in trades if t["direction"] == 1)
    n_short = sum(1 for t in trades if t["direction"] == -1)
    wr = wins / n
    mean_pnl = total / n
    print(f"  {symbol} {date_str}: {n} trades ({n_long}L/{n_short}S) | "
          f"net {total:+.1f} bps | WR {wr:.0%} | mean {mean_pnl:+.1f} bps")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        prog="trade_visualize",
        description="Paper Trade Snapshot \u2014 visual trade summary per symbol per day",
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Date to visualize (YYYY-MM-DD)")
    parser.add_argument("--latest", action="store_true",
                        help="Use the most recent trade date")
    parser.add_argument("--date-range", nargs=2, metavar=("START", "END"),
                        help="Date range to visualize (inclusive)")
    parser.add_argument("--symbol", type=str, default="BTC",
                        help="Symbol to visualize, or 'all' (default: BTC)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: reports/trade_viz)")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR,
                        help="Feature data directory")
    parser.add_argument("--trades-dir", type=Path, default=TRADE_DIR,
                        help="Paper trades directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Resolve dates
    if args.date:
        dates = [args.date]
    elif args.latest:
        all_dates = discover_trade_dates(args.trades_dir)
        if not all_dates:
            parser.error("No trade files found")
        dates = [all_dates[-1]]
    elif args.date_range:
        start, end = args.date_range
        all_dates = discover_trade_dates(args.trades_dir)
        dates = [d for d in all_dates if start <= d <= end]
        if not dates:
            parser.error(f"No trade dates in range {start}..{end}")
    else:
        parser.error("One of --date, --latest, or --date-range required")

    # Resolve symbols
    if args.symbol.lower() == "all":
        symbols = ["BTC", "ETH", "SOL"]
    else:
        symbols = [args.symbol]

    output_dir = args.output or DEFAULT_OUTPUT

    print()
    print("=" * 60)
    print("  Paper Trade Snapshot")
    print("=" * 60)
    print(f"  Dates:   {', '.join(dates)}")
    print(f"  Symbols: {', '.join(symbols)}")
    print()

    for date_str in dates:
        for symbol in symbols:
            trades = load_trades(args.trades_dir, date_str, symbol)
            if not trades:
                log.warning("No trades for %s %s, skipping", symbol, date_str)
                continue

            print_summary(trades, symbol, date_str)

            bars = load_price_bars(args.features_dir, date_str, symbol)
            if bars is None:
                log.info("No feature parquet for %s, trade-only mode", date_str)

            summary = load_batch_summary(args.trades_dir, date_str, symbol)
            out = render_trade_snapshot(trades, bars, symbol, date_str, summary, output_dir)
            print(f"    Saved: {out}")

    print()


if __name__ == "__main__":
    main()
