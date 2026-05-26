"""
Paper Trading — MF 3-Feature Liquidity Signal

Two modes:
  batch:  replay all dates, log paper trades, compare with backtest
  watch:  continuously monitor data/features/ for new bars (daemon)

Signal: 3f composite = (zscore(spread) + zscore(depth) + zscore(vwap_dev)) / 3
Entry:  long when composite > P80, short when composite < P20
Exit:   fixed 100min horizon (20 bars)
Train:  z-score params from prior 3 dates (walk-forward)

Quality Gate G8:
  - Paper Sharpe within 2x of backtest Sharpe
  - No single day > 2% loss
  - IC decay < 50% vs backtest IC
  - Infrastructure runs error-free for 14 days

Usage:
  python scripts/alpha/paper_trader.py batch --save
  python scripts/alpha/paper_trader.py watch --symbol BTC
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
TRADE_DIR = ROOT / "data" / "paper_trades"

# ── Config ────────────────────────────────────────────────────────────────

BAR_SECONDS = 300  # 5min
HORIZON_BARS = 20  # 100min
TRAIN_WINDOW = 3
MIN_BARS_PER_DATE = 12
P_LONG = 80
P_SHORT = 20

FEE_BPS = 1.61  # Binance VIP9 taker RT

LOAD_COLUMNS = [
    "timestamp_ns", "symbol", "raw_midprice",
    "raw_spread_bps", "raw_ask_depth_5", "flow_vwap_deviation",
]

BACKTEST_REFERENCE = {
    "BTC": {"sharpe": 11.8, "net_bps": 6.69, "ic": 0.52},
    "ETH": {"sharpe": 6.8, "net_bps": 6.80, "ic": 0.40},
    "SOL": {"sharpe": 9.2, "net_bps": 7.16, "ic": 0.35},
}


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class PaperTrade:
    date: str
    bar_idx: int
    symbol: str
    direction: int  # +1 long, -1 short
    signal_value: float
    entry_price: float
    exit_price: float | None = None
    exit_bar_idx: int | None = None
    gross_bps: float | None = None
    net_bps: float | None = None


@dataclass
class DailySummary:
    date: str
    symbol: str
    n_trades: int
    n_long: int
    n_short: int
    gross_bps: float
    net_bps: float
    total_net_bps: float
    win_rate: float
    max_loss_bps: float


# ── Data loading (shared with backtest) ──────────────────────────────────

def load_date(data_dir: Path, date_str: str, symbol: str) -> pd.DataFrame | None:
    from data.features import load_features
    df = load_features(
        symbols=[symbol],
        date_range=(date_str, date_str),
        columns=LOAD_COLUMNS,
        data_dir=data_dir,
        validate=False,
    )
    return df if not df.empty else None


def aggregate_to_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    bar_ns = BAR_SECONDS * 1_000_000_000
    ticks = ticks.copy()
    ticks["bar_id"] = ticks["timestamp_ns"].values // bar_ns
    agg = {
        "timestamp_ns": ("timestamp_ns", "first"),
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
    return bars


def discover_dates(data_dir: Path) -> list[str]:
    from data.features import available_dates
    return [d for d in available_dates(data_dir=data_dir) if "clean" not in d]


# ── Signal (3-feature composite) ────────────────────────────────────────

def compute_zscore_params_3f(train_bars_list: list[pd.DataFrame]) -> dict | None:
    spread = np.concatenate([b["spread_bps_last"].values for b in train_bars_list])
    depth = np.concatenate([b["depth_5_std"].values for b in train_bars_list])
    vwap_arrs = [b["vwap_deviation_std"].values for b in train_bars_list
                 if "vwap_deviation_std" in b.columns]
    if not vwap_arrs:
        return None
    vwap = np.concatenate(vwap_arrs)
    n = min(len(spread), len(depth), len(vwap))
    spread, depth, vwap = spread[:n], depth[:n], vwap[:n]
    mask = np.isfinite(spread) & np.isfinite(depth) & np.isfinite(vwap)
    spread, depth, vwap = spread[mask], depth[mask], vwap[mask]
    if len(spread) < 20:
        return None
    params = {
        "spread_mean": np.mean(spread), "spread_std": max(np.std(spread), 1e-10),
        "depth_mean": np.mean(depth), "depth_std": max(np.std(depth), 1e-10),
        "vwap_mean": np.mean(vwap), "vwap_std": max(np.std(vwap), 1e-10),
    }
    z_s = (spread - params["spread_mean"]) / params["spread_std"]
    z_d = (depth - params["depth_mean"]) / params["depth_std"]
    z_v = (vwap - params["vwap_mean"]) / params["vwap_std"]
    composite = (z_s + z_d + z_v) / 3.0
    params["p_long"] = np.percentile(composite, P_LONG)
    params["p_short"] = np.percentile(composite, P_SHORT)
    return params


def apply_signal_3f(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    bars = bars.copy()
    z_s = (bars["spread_bps_last"] - params["spread_mean"]) / params["spread_std"]
    z_d = (bars["depth_5_std"] - params["depth_mean"]) / params["depth_std"]
    z_v = (bars["vwap_deviation_std"] - params["vwap_mean"]) / params["vwap_std"]
    bars["composite"] = (z_s + z_d + z_v) / 3.0
    bars["direction"] = 0
    bars.loc[bars["composite"] >= params["p_long"], "direction"] = 1
    bars.loc[bars["composite"] <= params["p_short"], "direction"] = -1
    return bars


# ── Trade generation ─────────────────────────────────────────────────────

def generate_trades(
    bars: pd.DataFrame, date_str: str, symbol: str,
) -> list[PaperTrade]:
    """Generate paper trades from signal-scored bars with fixed horizon exit."""
    prices = bars["midprice_last"].values
    directions = bars["direction"].values
    composites = bars["composite"].values
    n = len(prices)
    trades = []

    for i in range(n - HORIZON_BARS):
        d = directions[i]
        if d == 0:
            continue
        entry_p = prices[i]
        exit_p = prices[i + HORIZON_BARS]
        if entry_p <= 0 or not np.isfinite(entry_p) or not np.isfinite(exit_p):
            continue
        ret_bps = (exit_p - entry_p) / entry_p * 1e4
        gross = d * ret_bps
        net = gross - FEE_BPS

        trades.append(PaperTrade(
            date=date_str,
            bar_idx=i,
            symbol=symbol,
            direction=d,
            signal_value=float(composites[i]),
            entry_price=float(entry_p),
            exit_price=float(exit_p),
            exit_bar_idx=i + HORIZON_BARS,
            gross_bps=round(gross, 4),
            net_bps=round(net, 4),
        ))

    return trades


def summarize_day(trades: list[PaperTrade], date_str: str, symbol: str) -> DailySummary:
    if not trades:
        return DailySummary(
            date=date_str, symbol=symbol,
            n_trades=0, n_long=0, n_short=0,
            gross_bps=0.0, net_bps=0.0, total_net_bps=0.0,
            win_rate=0.0, max_loss_bps=0.0,
        )
    gross = np.array([t.gross_bps for t in trades])
    net = np.array([t.net_bps for t in trades])
    return DailySummary(
        date=date_str,
        symbol=symbol,
        n_trades=len(trades),
        n_long=sum(1 for t in trades if t.direction == 1),
        n_short=sum(1 for t in trades if t.direction == -1),
        gross_bps=round(float(np.mean(gross)), 3),
        net_bps=round(float(np.mean(net)), 3),
        total_net_bps=round(float(np.sum(net)), 2),
        win_rate=round(float(np.mean(net > 0)), 3),
        max_loss_bps=round(float(np.min(net)), 3),
    )


# ── Trade persistence ────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_trades(trades: list[PaperTrade], date_str: str, symbol: str):
    TRADE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRADE_DIR / f"{date_str}_{symbol}.json"
    with open(path, "w") as f:
        json.dump([asdict(t) for t in trades], f, indent=2, default=_json_default)
    return path


def load_trades(date_str: str, symbol: str) -> list[PaperTrade]:
    path = TRADE_DIR / f"{date_str}_{symbol}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return [PaperTrade(**d) for d in json.load(f)]


# ── Batch mode ───────────────────────────────────────────────────────────

def run_batch(
    data_dir: Path,
    symbols: list[str],
    save: bool = False,
):
    """Replay all dates with walk-forward paper trading. Compare to backtest."""
    all_dates = discover_dates(data_dir)
    print(f"Found {len(all_dates)} dates: {all_dates[0]} to {all_dates[-1]}")
    print(f"Signal: 3f composite | Horizon: {HORIZON_BARS * BAR_SECONDS // 60}min | "
          f"Fee: {FEE_BPS} bps RT\n")

    all_results = {}

    for symbol in symbols:
        print(f"═══ {symbol} ═══")

        # Load all dates
        date_bars: list[tuple[str, pd.DataFrame]] = []
        for date_str in all_dates:
            ticks = load_date(data_dir, date_str, symbol)
            if ticks is None or len(ticks) < 100:
                continue
            bars = aggregate_to_bars(ticks)
            if len(bars) >= MIN_BARS_PER_DATE:
                date_bars.append((date_str, bars))

        print(f"  {len(date_bars)} usable dates")

        if len(date_bars) < TRAIN_WINDOW + 1:
            print(f"  Not enough dates (need {TRAIN_WINDOW + 1})\n")
            continue

        # Walk-forward paper trading
        daily_summaries = []
        all_trades_flat = []

        for i in range(TRAIN_WINDOW, len(date_bars)):
            train_bar_list = [b for _, b in date_bars[i - TRAIN_WINDOW:i]]
            test_date_str, test_bars = date_bars[i]

            params = compute_zscore_params_3f(train_bar_list)
            if params is None:
                continue

            scored = apply_signal_3f(test_bars, params)
            trades = generate_trades(scored, test_date_str, symbol)
            summary = summarize_day(trades, test_date_str, symbol)
            daily_summaries.append(summary)
            all_trades_flat.extend(trades)

            if save:
                save_trades(trades, test_date_str, symbol)

            tag = "+" if summary.total_net_bps > 0 else " "
            print(f"  {test_date_str}: {summary.n_trades:3d} trades | "
                  f"net {summary.net_bps:+.2f} bps/trade | "
                  f"total {tag}{summary.total_net_bps:+.1f} bps | "
                  f"WR {summary.win_rate:.0%} | "
                  f"L:{summary.n_long} S:{summary.n_short}")

        # Aggregate stats
        if daily_summaries:
            daily_pnl = np.array([s.total_net_bps for s in daily_summaries])
            total_trades = sum(s.n_trades for s in daily_summaries)
            total_pnl = float(np.sum(daily_pnl))
            daily_std = np.std(daily_pnl)
            sharpe = float(np.mean(daily_pnl) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
            n_positive = int(np.sum(daily_pnl > 0))

            ref = BACKTEST_REFERENCE.get(symbol, {})
            bt_sharpe = ref.get("sharpe", 0)
            bt_net = ref.get("net_bps", 0)

            print(f"\n  Paper:    {len(daily_summaries)} OOS days | {total_trades} trades | "
                  f"Sharpe {sharpe:+.1f} | net total {total_pnl:+.0f} bps | "
                  f"WR {n_positive}/{len(daily_summaries)}")
            print(f"  Backtest: Sharpe {bt_sharpe:+.1f} | net {bt_net:+.2f} bps/trade")

            # G8 gates
            max_daily_loss = float(np.min(daily_pnl))
            sharpe_ratio = sharpe / bt_sharpe if bt_sharpe > 0 else 0
            gate_sharpe = sharpe_ratio > 0.5
            gate_loss = max_daily_loss > -200  # -2% in bps at typical position

            print(f"\n  G8 Gates:")
            print(f"    Sharpe ratio (paper/backtest): {sharpe_ratio:.2f} "
                  f"{'PASS' if gate_sharpe else 'FAIL'} (>0.5)")
            print(f"    Max daily loss: {max_daily_loss:+.1f} bps "
                  f"{'PASS' if gate_loss else 'FAIL'}")

            all_results[symbol] = {
                "n_oos_dates": len(daily_summaries),
                "n_trades": total_trades,
                "sharpe": round(sharpe, 2),
                "total_pnl_bps": round(total_pnl, 1),
                "daily_win_rate": round(n_positive / len(daily_summaries), 2),
                "max_daily_loss_bps": round(max_daily_loss, 1),
                "backtest_sharpe": bt_sharpe,
                "sharpe_ratio": round(sharpe_ratio, 2),
                "daily": [asdict(s) for s in daily_summaries],
            }

        print()

    if save and all_results:
        report = {
            "title": "Paper Trading — 3f/100min Walk-Forward",
            "generated": datetime.now(timezone.utc).isoformat(),
            "signal": "3f composite (spread + depth + vwap_deviation)",
            "horizon_min": HORIZON_BARS * BAR_SECONDS // 60,
            "fee_bps_rt": FEE_BPS,
            "results": all_results,
        }
        out = ROOT / "data" / "paper_trades" / "batch_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved: {out}")

    return all_results


# ── Watch mode (daemon) ──────────────────────────────────────────────────

def run_watch(
    data_dir: Path,
    symbol: str,
    poll_seconds: int = 300,
):
    """Continuously watch for new data and generate paper trades."""
    print(f"Paper trader watching {data_dir} for {symbol}")
    print(f"Signal: 3f/100min | Poll: {poll_seconds}s | Fee: {FEE_BPS} bps\n")

    processed_dates: set[str] = set()

    while True:
        try:
            all_dates = discover_dates(data_dir)
            if len(all_dates) < TRAIN_WINDOW + 1:
                log.info(f"Waiting for data ({len(all_dates)} dates, need {TRAIN_WINDOW + 1})")
                time.sleep(poll_seconds)
                continue

            # Load bars for all dates
            date_bars: list[tuple[str, pd.DataFrame]] = []
            for date_str in all_dates:
                ticks = load_date(data_dir, date_str, symbol)
                if ticks is None or len(ticks) < 100:
                    continue
                bars = aggregate_to_bars(ticks)
                if len(bars) >= MIN_BARS_PER_DATE:
                    date_bars.append((date_str, bars))

            # Process unprocessed test dates
            for i in range(TRAIN_WINDOW, len(date_bars)):
                test_date = date_bars[i][0]
                if test_date in processed_dates:
                    continue

                train_bar_list = [b for _, b in date_bars[i - TRAIN_WINDOW:i]]
                test_bars = date_bars[i][1]

                params = compute_zscore_params_3f(train_bar_list)
                if params is None:
                    continue

                scored = apply_signal_3f(test_bars, params)
                trades = generate_trades(scored, test_date, symbol)
                summary = summarize_day(trades, test_date, symbol)

                save_trades(trades, test_date, symbol)
                processed_dates.add(test_date)

                now = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(f"[{now}] {test_date} {symbol}: "
                      f"{summary.n_trades} trades | "
                      f"net {summary.net_bps:+.2f} bps | "
                      f"total {summary.total_net_bps:+.1f} bps | "
                      f"WR {summary.win_rate:.0%}")

        except KeyboardInterrupt:
            print("\nStopping paper trader.")
            break
        except Exception as e:
            log.error(f"Watch loop error: {e}")

        time.sleep(poll_seconds)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper Trading — MF 3f Signal")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Batch mode
    p_batch = sub.add_parser("batch", help="Replay all dates, compare with backtest")
    p_batch.add_argument("--data-dir", default="data/features")
    p_batch.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    p_batch.add_argument("--save", action="store_true", help="Save trade logs + report")
    p_batch.add_argument("--json-output", type=str, default=None,
                         help="Write structured results JSON to this path")

    # Watch mode
    p_watch = sub.add_parser("watch", help="Continuously monitor for new data")
    p_watch.add_argument("--data-dir", default="data/features")
    p_watch.add_argument("--symbol", default="BTC")
    p_watch.add_argument("--poll", type=int, default=300, help="Poll interval seconds")

    args = parser.parse_args()

    from logging_config import setup_logging
    setup_logging("nat.paper_trader")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    if args.mode == "batch":
        all_results = run_batch(data_dir, args.symbols, save=args.save)
        if args.json_output and all_results:
            out = Path(args.json_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(all_results, f, indent=2, default=_json_default)
            print(f"JSON output saved: {out}")
    elif args.mode == "watch":
        run_watch(data_dir, args.symbol, poll_seconds=args.poll)


if __name__ == "__main__":
    main()
