#!/usr/bin/env python3
"""
Generic Paper Trader — runs any MicrostructureAlgorithm through walk-forward.

For each algorithm:
  1. Run run_batch() on tick data to produce features
  2. Aggregate to 5min bars (mean of primary feature per bar)
  3. Walk-forward z-score calibration on prior 3 dates
  4. Entry: long when z-score below P20 or above P80 (configurable per algorithm)
  5. Exit: fixed 100min horizon

Usage:
  python scripts/alpha/paper_trader_generic.py --algorithms entropy_momentum hawkes_intensity
  python scripts/alpha/paper_trader_generic.py --all --save
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from algorithms import get_algorithm, list_algorithms, discover_all
from backtest.costs import CostModel, hyperliquid_taker, hyperliquid_maker, conservative
from config_utils import load_cost_config

# ── Config ────────────────────────────────────────────────────────────────

BAR_SECONDS = 300
HORIZON_BARS = 20  # 100min
TRAIN_WINDOW = 3
MIN_BARS_PER_DATE = 12

# Named cost presets
COST_PRESETS: dict[str, CostModel] = {
    "taker": hyperliquid_taker(),
    "maker": hyperliquid_maker(),
    "conservative": conservative(),
}

# Default: load from config/agent.toml [defaults.costs]
_cost_cfg = load_cost_config()
COST_PRESETS["config"] = CostModel(fee_bps=_cost_cfg["fee_bps"], slippage_bps=_cost_cfg["slippage_bps"])
COST_MODEL = COST_PRESETS["config"]

# Per-algorithm: which feature to use as primary signal, and signal polarity.
# "high_long" = high z-score → long (e.g., momentum).
# "low_long"  = low z-score → long (e.g., entropy surprise, mean-reversion).
ALGO_CONFIG = {
    "entropy_momentum": {
        "primary": "alg_entropy_gated_momentum",
        "polarity": "high_long",
        "bar_agg": "mean",
    },
    "hawkes_intensity": {
        "primary": "alg_bid_ask_hawkes_imbalance",
        "polarity": "high_long",
        "bar_agg": "mean",
    },
    "weighted_ofi": {
        "primary": "alg_weighted_ofi",
        "polarity": "high_long",
        "bar_agg": "mean",
    },
    "trade_through": {
        "primary": "alg_trade_through_imbalance",
        "polarity": "high_long",
        "bar_agg": "mean",
    },
    "propagator": {
        "primary": "alg_transient_impact",
        "polarity": "low_long",  # negative transient impact → reversal → long
        "bar_agg": "mean",
    },
    "regime_gated": {
        "primary": "alg_regime_gated_imbalance",
        "polarity": "high_long",
        "bar_agg": "mean",
    },
    "bipower_jump": {
        "primary": "alg_jump_ratio",
        "polarity": "low_long",  # low jump fraction → stable → momentum
        "bar_agg": "mean",
    },
    "spread_decomp": {
        "primary": "alg_adverse_component",
        "polarity": "low_long",  # low adverse selection → less informed trading → safe
        "bar_agg": "mean",
    },
    "vpin_regime": {
        "primary": "alg_vpin_gated_imbalance",
        "polarity": "high_long",
        "bar_agg": "mean",
    },
    "switching_ou": {
        "primary": "alg_switching_ou_state",
        "polarity": "high_long",
        "bar_agg": "last",
    },
    "kalman_imbalance": {
        "primary": "alg_kalman_signal_strength",
        "polarity": "high_long",
        "bar_agg": "last",
    },
    "optimal_entry": {
        "primary": "alg_entry_signal",
        "polarity": "high_long",
        "bar_agg": "last",
    },
    "jump_detector": {
        "primary": "alg_post_jump_reversion",
        "polarity": "low_long",  # negative post-jump → expecting reversion up
        "bar_agg": "mean",
    },
    "multi_level_imb": {
        "primary": "alg_composite_imbalance",
        "polarity": "high_long",
        "bar_agg": "mean",
    },
    "funding_reversion": {
        "primary": "alg_funding_signal",
        "polarity": "high_long",
        "bar_agg": "last",
    },
    "oi_divergence": {
        "primary": "alg_oi_price_divergence",
        "polarity": "low_long",  # divergence = contrarian
        "bar_agg": "mean",
    },
}

P_LONG_HIGH = 80  # for high_long polarity
P_SHORT_HIGH = 20
P_LONG_LOW = 20   # for low_long polarity
P_SHORT_LOW = 80


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

def discover_dates(data_dir: Path) -> list[str]:
    from data.features import available_dates
    return [d for d in available_dates(data_dir=data_dir) if "clean" not in d]


def load_date_ticks(data_dir: Path, date_str: str, symbol: str,
                    columns: list[str]) -> pd.DataFrame | None:
    from data.features import load_features
    base_cols = ["timestamp_ns", "symbol", "raw_midprice"]
    load_cols = list(set(base_cols + columns))
    df = load_features(
        symbols=[symbol],
        date_range=(date_str, date_str),
        columns=load_cols,
        data_dir=data_dir,
        validate=False,
    )
    return df if not df.empty else None


def aggregate_to_bars(ticks: pd.DataFrame, features: pd.DataFrame,
                      primary_col: str, agg_method: str) -> pd.DataFrame:
    bar_ns = BAR_SECONDS * 1_000_000_000
    ticks = ticks.copy()
    ticks["bar_id"] = ticks["timestamp_ns"].values // bar_ns
    ticks["_signal"] = features[primary_col].values

    agg_dict = {
        "timestamp_ns": ("timestamp_ns", "first"),
        "midprice_last": ("raw_midprice", "last"),
        "n_ticks": ("raw_midprice", "count"),
    }
    if agg_method == "last":
        agg_dict["signal"] = ("_signal", "last")
    else:
        agg_dict["signal_mean"] = ("_signal", "mean")
        agg_dict["signal_std"] = ("_signal", "std")

    bars = ticks.groupby("bar_id").agg(**agg_dict).reset_index(drop=True)
    bars = bars[bars["n_ticks"] >= 10].reset_index(drop=True)

    if agg_method == "last":
        bars["signal"] = bars["signal"].astype(float)
    else:
        bars["signal"] = bars["signal_mean"].astype(float)

    return bars


# ── Signal ──────────────────────────────────────────────────────────────

def compute_params(train_bars_list: list[pd.DataFrame], polarity: str) -> dict | None:
    vals = np.concatenate([b["signal"].values for b in train_bars_list])
    mask = np.isfinite(vals)
    vals = vals[mask]
    if len(vals) < 20:
        return None

    params = {
        "mean": float(np.mean(vals)),
        "std": float(max(np.std(vals), 1e-10)),
    }
    z = (vals - params["mean"]) / params["std"]

    if polarity == "high_long":
        params["p_long"] = float(np.percentile(z, P_LONG_HIGH))
        params["p_short"] = float(np.percentile(z, P_SHORT_HIGH))
    else:  # low_long
        params["p_long"] = float(np.percentile(z, P_LONG_LOW))
        params["p_short"] = float(np.percentile(z, P_SHORT_LOW))

    params["polarity"] = polarity
    return params


def apply_signal(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    bars = bars.copy()
    z = (bars["signal"] - params["mean"]) / params["std"]
    bars["composite"] = z
    bars["direction"] = 0

    if params["polarity"] == "high_long":
        bars.loc[bars["composite"] >= params["p_long"], "direction"] = 1
        bars.loc[bars["composite"] <= params["p_short"], "direction"] = -1
    else:  # low_long
        bars.loc[bars["composite"] <= params["p_long"], "direction"] = 1
        bars.loc[bars["composite"] >= params["p_short"], "direction"] = -1

    return bars


# ── Trade generation ─────────────────────────────────────────────────────

def generate_trades(bars: pd.DataFrame, date_str: str, symbol: str,
                    cost_model: CostModel = COST_MODEL) -> list[PaperTrade]:
    prices = bars["midprice_last"].values
    directions = bars["direction"].values
    composites = bars["composite"].values
    n = len(prices)
    rt_cost = cost_model.round_trip_cost_bps
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
        net = gross - rt_cost

        trades.append(PaperTrade(
            date=date_str, bar_idx=i, symbol=symbol, direction=d,
            signal_value=float(composites[i]),
            entry_price=float(entry_p), exit_price=float(exit_p),
            exit_bar_idx=i + HORIZON_BARS,
            gross_bps=round(gross, 4), net_bps=round(net, 4),
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
        date=date_str, symbol=symbol,
        n_trades=len(trades),
        n_long=sum(1 for t in trades if t.direction == 1),
        n_short=sum(1 for t in trades if t.direction == -1),
        gross_bps=round(float(np.mean(gross)), 3),
        net_bps=round(float(np.mean(net)), 3),
        total_net_bps=round(float(np.sum(net)), 2),
        win_rate=round(float(np.mean(net > 0)), 3),
        max_loss_bps=round(float(np.min(net)), 3),
    )


# ── Main runner ─────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run_algorithm(algo_name: str, data_dir: Path, symbols: list[str],
                  save: bool = False, return_trades: bool = False,
                  cost_model: CostModel = COST_MODEL) -> dict:
    """Run walk-forward paper trading for one algorithm.

    If return_trades=True, result[symbol] includes a "trades" key with the
    flat list of PaperTrade dicts (for continuous testing / visualization).
    """
    config = ALGO_CONFIG.get(algo_name)
    if config is None:
        print(f"  No config for '{algo_name}', skipping")
        return {}

    algo = get_algorithm(algo_name)
    required = algo.required_columns()
    primary = config["primary"]
    polarity = config["polarity"]
    agg_method = config["bar_agg"]

    all_dates = discover_dates(data_dir)

    print(f"\n{'═' * 60}")
    print(f"  Algorithm: {algo_name}")
    print(f"  Primary feature: {primary} | Polarity: {polarity}")
    print(f"  Horizon: {HORIZON_BARS * BAR_SECONDS // 60}min | Cost: {cost_model.round_trip_cost_bps} bps RT "
          f"(fee={cost_model.fee_bps}+slip={cost_model.slippage_bps} one-way)")
    print(f"  Dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")
    print(f"{'═' * 60}")

    all_results = {}

    for symbol in symbols:
        print(f"\n  ─── {symbol} ───")

        # Load all dates
        date_bars: list[tuple[str, pd.DataFrame]] = []
        for date_str in all_dates:
            ticks = load_date_ticks(data_dir, date_str, symbol, required)
            if ticks is None or len(ticks) < 200:
                continue
            # Check required columns exist
            missing = [c for c in required if c not in ticks.columns]
            if missing:
                continue
            try:
                features = algo.run_batch(ticks)
                algo.reset()
            except Exception as e:
                print(f"    {date_str}: algo error: {e}")
                continue
            if primary not in features.columns:
                continue
            bars = aggregate_to_bars(ticks, features, primary, agg_method)
            if len(bars) >= MIN_BARS_PER_DATE:
                date_bars.append((date_str, bars))

        print(f"    {len(date_bars)} usable dates")

        if len(date_bars) < TRAIN_WINDOW + 1:
            print(f"    Not enough dates (need {TRAIN_WINDOW + 1})")
            continue

        daily_summaries = []
        all_trades = [] if return_trades else None

        for i in range(TRAIN_WINDOW, len(date_bars)):
            train_bar_list = [b for _, b in date_bars[i - TRAIN_WINDOW:i]]
            test_date_str, test_bars = date_bars[i]

            params = compute_params(train_bar_list, polarity)
            if params is None:
                continue

            scored = apply_signal(test_bars, params)
            trades = generate_trades(scored, test_date_str, symbol, cost_model=cost_model)
            summary = summarize_day(trades, test_date_str, symbol)
            daily_summaries.append(summary)
            if return_trades:
                all_trades.extend(trades)

            tag = "+" if summary.total_net_bps > 0 else " "
            print(f"    {test_date_str}: {summary.n_trades:3d} trades | "
                  f"net {summary.net_bps:+7.2f} bps | "
                  f"total {tag}{summary.total_net_bps:+8.1f} bps | "
                  f"WR {summary.win_rate:.0%}")

        if daily_summaries:
            daily_pnl = np.array([s.total_net_bps for s in daily_summaries])
            total_trades = sum(s.n_trades for s in daily_summaries)
            total_pnl = float(np.sum(daily_pnl))
            daily_std = np.std(daily_pnl)
            sharpe = float(np.mean(daily_pnl) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
            n_positive = int(np.sum(daily_pnl > 0))

            print(f"\n    Summary: {len(daily_summaries)} days | {total_trades} trades | "
                  f"Sharpe {sharpe:+.1f} | total {total_pnl:+.0f} bps | "
                  f"WR {n_positive}/{len(daily_summaries)}")

            result = {
                "n_oos_dates": len(daily_summaries),
                "n_trades": total_trades,
                "net_bps_per_trade": round(total_pnl / total_trades, 3) if total_trades else 0,
                "sharpe": round(sharpe, 2),
                "total_pnl_bps": round(total_pnl, 1),
                "daily_win_rate": round(n_positive / len(daily_summaries), 2),
                "max_daily_loss_bps": round(float(np.min(daily_pnl)), 1),
                "daily": [asdict(s) for s in daily_summaries],
            }
            if return_trades:
                result["trades"] = [asdict(t) for t in all_trades]
            all_results[symbol] = result

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Generic Paper Trader")
    parser.add_argument("--algorithms", nargs="+", help="Algorithm names to test")
    parser.add_argument("--all", action="store_true", help="Test all configured algorithms")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--save", action="store_true", help="Save report JSON")
    parser.add_argument("--json-output", type=str, default=None,
                        help="Write structured results JSON to this path")
    parser.add_argument("--cost-mode", choices=list(COST_PRESETS.keys()),
                        default="config",
                        help="Cost model preset: taker (11bps RT), maker (1.4bps RT), "
                             "conservative (25bps RT), config (from agent.toml)")
    args = parser.parse_args()

    cost_model = COST_PRESETS[args.cost_mode]

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    if args.all:
        algo_names = sorted(ALGO_CONFIG.keys())
    elif args.algorithms:
        algo_names = args.algorithms
    else:
        print("Specify --algorithms or --all")
        sys.exit(1)

    print(f"Testing {len(algo_names)} algorithms: {', '.join(algo_names)}")
    print(f"Cost mode: {args.cost_mode} → {cost_model}")

    master_results = {}
    for algo_name in algo_names:
        t0 = time.time()
        results = run_algorithm(algo_name, data_dir, args.symbols, save=args.save,
                               cost_model=cost_model)
        elapsed = time.time() - t0
        if results:
            master_results[algo_name] = {
                "elapsed_s": round(elapsed, 1),
                "results": results,
            }

    # Print comparison table
    print(f"\n\n{'=' * 80}")
    print(f"  COMPARISON TABLE — Net BPS and Sharpe by Algorithm × Symbol")
    print(f"{'=' * 80}")
    print(f"  {'Algorithm':<25s} {'BTC':>18s} {'ETH':>18s} {'SOL':>18s}")
    print(f"  {'':25s} {'bps    Sharpe':>18s} {'bps    Sharpe':>18s} {'bps    Sharpe':>18s}")
    print(f"  {'-' * 79}")

    for algo_name in algo_names:
        if algo_name not in master_results:
            continue
        res = master_results[algo_name]["results"]
        cols = []
        for sym in ["BTC", "ETH", "SOL"]:
            if sym in res:
                cols.append(f"{res[sym]['total_pnl_bps']:+7.0f} {res[sym]['sharpe']:+5.1f}")
            else:
                cols.append(f"{'N/A':>13s}")
        print(f"  {algo_name:<25s} {cols[0]:>18s} {cols[1]:>18s} {cols[2]:>18s}")

    print()

    if args.save and master_results:
        report = {
            "title": "Generic Paper Trade — Algorithm Comparison",
            "generated": datetime.now(timezone.utc).isoformat(),
            "horizon_min": HORIZON_BARS * BAR_SECONDS // 60,
            "cost_mode": args.cost_mode,
            "fee_bps_rt": cost_model.round_trip_cost_bps,
            "algorithms": master_results,
        }
        out = ROOT / "reports" / "algo_paper_trade_comparison.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=_json_default)
        print(f"Report saved: {out}")

    if args.json_output and master_results:
        out2 = Path(args.json_output)
        out2.parent.mkdir(parents=True, exist_ok=True)
        with open(out2, "w") as f:
            json.dump(master_results, f, indent=2, default=_json_default)
        print(f"JSON output saved: {out2}")


if __name__ == "__main__":
    main()
