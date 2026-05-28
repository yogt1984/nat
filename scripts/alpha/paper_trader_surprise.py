"""
Paper Trading — Surprise Signal (Entropy Regime Transition)

Runs the SurpriseSignal algorithm on tick-level data, aggregates to 5min bars,
and generates paper trades using the entropy surprise z-score as the signal.

Trading thesis:
  Negative surprise (entropy dropping → market ordering) → momentum continuation → long
  Positive surprise (entropy spiking → market disordering) → reversal pressure → short

Walk-forward: z-score thresholds calibrated on prior 3 dates.

Usage:
  python scripts/alpha/paper_trader_surprise.py batch --save
  python scripts/alpha/paper_trader_surprise.py watch --symbol BTC
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
TRADE_DIR = ROOT / "data" / "paper_trades_surprise"

# Add algorithms to path

from algorithms.surprise_signal import SurpriseSignal

# ── Config ────────────────────────────────────────────────────────────────

BAR_SECONDS = 300  # 5min
HORIZON_BARS = 20  # 100min
TRAIN_WINDOW = 3
MIN_BARS_PER_DATE = 12
P_LONG = 20   # Long when surprise < P20 (ordering = negative surprise)
P_SHORT = 80  # Short when surprise > P80 (disordering = positive surprise)

FEE_BPS = 1.61  # Binance VIP9 taker RT

LOAD_COLUMNS = [
    "timestamp_ns", "symbol", "raw_midprice",
    "ent_book_shape", "ent_tick_5s",
]


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class PaperTrade:
    date: str
    bar_idx: int
    symbol: str
    direction: int
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


# ── Data loading ────────────────────────────────────────────────────────

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


def discover_dates(data_dir: Path) -> list[str]:
    from data.features import available_dates
    return [d for d in available_dates(data_dir=data_dir) if "clean" not in d]


# ── Surprise signal computation ────────────────────────────────────────

def compute_surprise_features(ticks: pd.DataFrame) -> pd.DataFrame:
    """Run SurpriseSignal algorithm on tick data, return features aligned to ticks."""
    algo = SurpriseSignal(roc_window=50, transition_threshold=2.0)
    features = algo.run_batch(ticks)
    return features


def aggregate_to_bars(ticks: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tick data + surprise features to 5min bars."""
    bar_ns = BAR_SECONDS * 1_000_000_000
    ticks = ticks.copy()
    ticks["bar_id"] = ticks["timestamp_ns"].values // bar_ns

    # Attach surprise features
    ticks["surprise"] = features["alg_entropy_surprise"].values
    ticks["entropy_roc"] = features["alg_entropy_roc"].values
    ticks["transition_prob"] = features["alg_regime_transition_prob"].values

    bars = ticks.groupby("bar_id").agg(
        timestamp_ns=("timestamp_ns", "first"),
        midprice_last=("raw_midprice", "last"),
        n_ticks=("raw_midprice", "count"),
        surprise_mean=("surprise", "mean"),
        surprise_last=("surprise", "last"),
        surprise_std=("surprise", "std"),
        entropy_roc_mean=("entropy_roc", "mean"),
        transition_prob_max=("transition_prob", "max"),
    ).reset_index(drop=True)

    bars = bars[bars["n_ticks"] >= 10].reset_index(drop=True)
    bars["surprise_std"] = bars["surprise_std"].fillna(0.0)
    return bars


# ── Signal (surprise-based) ──────────────────────────────────────────────

def compute_signal_params(train_bars_list: list[pd.DataFrame]) -> dict | None:
    """Compute z-score params and percentile thresholds from training bars."""
    surprise_vals = np.concatenate([b["surprise_mean"].values for b in train_bars_list])
    mask = np.isfinite(surprise_vals)
    surprise_vals = surprise_vals[mask]

    if len(surprise_vals) < 20:
        return None

    params = {
        "surprise_mean": float(np.mean(surprise_vals)),
        "surprise_std": float(max(np.std(surprise_vals), 1e-10)),
    }

    z = (surprise_vals - params["surprise_mean"]) / params["surprise_std"]
    # Negative surprise → long, so p_long is a low percentile
    params["p_long"] = float(np.percentile(z, P_LONG))
    params["p_short"] = float(np.percentile(z, P_SHORT))

    return params


def apply_signal(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Score bars: long when surprise z-score < p_long, short when > p_short."""
    bars = bars.copy()
    z = (bars["surprise_mean"] - params["surprise_mean"]) / params["surprise_std"]
    bars["composite"] = z

    bars["direction"] = 0
    # Negative surprise = ordering = momentum → long
    bars.loc[bars["composite"] <= params["p_long"], "direction"] = 1
    # Positive surprise = disordering = reversal → short
    bars.loc[bars["composite"] >= params["p_short"], "direction"] = -1

    return bars


# ── Trade generation ─────────────────────────────────────────────────────

def generate_trades(
    bars: pd.DataFrame, date_str: str, symbol: str,
) -> list[PaperTrade]:
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


# ── Batch mode ───────────────────────────────────────────────────────────

def run_batch(
    data_dir: Path,
    symbols: list[str],
    save: bool = False,
):
    all_dates = discover_dates(data_dir)
    print(f"Found {len(all_dates)} dates: {all_dates[0]} to {all_dates[-1]}")
    print(f"Signal: entropy surprise | Horizon: {HORIZON_BARS * BAR_SECONDS // 60}min | "
          f"Fee: {FEE_BPS} bps RT\n")

    all_results = {}

    for symbol in symbols:
        print(f"═══ {symbol} ═══")

        # Load all dates and compute surprise features
        date_bars: list[tuple[str, pd.DataFrame]] = []
        for date_str in all_dates:
            ticks = load_date(data_dir, date_str, symbol)
            if ticks is None or len(ticks) < 200:
                continue
            features = compute_surprise_features(ticks)
            bars = aggregate_to_bars(ticks, features)
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

            params = compute_signal_params(train_bar_list)
            if params is None:
                continue

            scored = apply_signal(test_bars, params)
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

            print(f"\n  Summary: {len(daily_summaries)} OOS days | {total_trades} trades | "
                  f"Sharpe {sharpe:+.1f} | net total {total_pnl:+.0f} bps | "
                  f"WR {n_positive}/{len(daily_summaries)}")

            all_results[symbol] = {
                "n_oos_dates": len(daily_summaries),
                "n_trades": total_trades,
                "sharpe": round(sharpe, 2),
                "total_pnl_bps": round(total_pnl, 1),
                "daily_win_rate": round(n_positive / len(daily_summaries), 2),
                "max_daily_loss_bps": round(float(np.min(daily_pnl)), 1),
                "daily": [asdict(s) for s in daily_summaries],
            }

        print()

    if save and all_results:
        report = {
            "title": "Paper Trading — Surprise Signal Walk-Forward",
            "generated": datetime.now(timezone.utc).isoformat(),
            "signal": "entropy surprise (ordering → long, disordering → short)",
            "horizon_min": HORIZON_BARS * BAR_SECONDS // 60,
            "fee_bps_rt": FEE_BPS,
            "results": all_results,
        }
        out = ROOT / "reports" / "surprise_paper_trade.json"
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
    print(f"Surprise paper trader watching {data_dir} for {symbol}")
    print(f"Signal: entropy surprise/100min | Poll: {poll_seconds}s | Fee: {FEE_BPS} bps\n")

    processed_dates: set[str] = set()

    while True:
        try:
            all_dates = discover_dates(data_dir)
            if len(all_dates) < TRAIN_WINDOW + 1:
                log.info(f"Waiting for data ({len(all_dates)} dates, need {TRAIN_WINDOW + 1})")
                time.sleep(poll_seconds)
                continue

            date_bars: list[tuple[str, pd.DataFrame]] = []
            for date_str in all_dates:
                ticks = load_date(data_dir, date_str, symbol)
                if ticks is None or len(ticks) < 200:
                    continue
                features = compute_surprise_features(ticks)
                bars = aggregate_to_bars(ticks, features)
                if len(bars) >= MIN_BARS_PER_DATE:
                    date_bars.append((date_str, bars))

            for i in range(TRAIN_WINDOW, len(date_bars)):
                test_date = date_bars[i][0]
                if test_date in processed_dates:
                    continue

                train_bar_list = [b for _, b in date_bars[i - TRAIN_WINDOW:i]]
                test_bars = date_bars[i][1]

                params = compute_signal_params(train_bar_list)
                if params is None:
                    continue

                scored = apply_signal(test_bars, params)
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
            print("\nStopping surprise paper trader.")
            break
        except Exception as e:
            log.error(f"Watch loop error: {e}")

        time.sleep(poll_seconds)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paper Trading — Surprise Signal")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_batch = sub.add_parser("batch", help="Replay all dates with walk-forward")
    p_batch.add_argument("--data-dir", default="data/features")
    p_batch.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    p_batch.add_argument("--save", action="store_true", help="Save trade logs + report")

    p_watch = sub.add_parser("watch", help="Continuously monitor for new data")
    p_watch.add_argument("--data-dir", default="data/features")
    p_watch.add_argument("--symbol", default="BTC")
    p_watch.add_argument("--poll", type=int, default=300, help="Poll interval seconds")

    args = parser.parse_args()

    from logging_config import setup_logging
    setup_logging("nat.paper_trader_surprise")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    if args.mode == "batch":
        run_batch(data_dir, args.symbols, save=args.save)
    elif args.mode == "watch":
        run_watch(data_dir, args.symbol, poll_seconds=args.poll)


if __name__ == "__main__":
    main()
