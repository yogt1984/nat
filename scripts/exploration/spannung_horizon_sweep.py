#!/usr/bin/env python3
"""
Spannung Horizon Sweep — test aggregated imbalance signals at longer timescales.

Aggregates tick-level imbalance into bars (30s–15min), computes signal variants,
and measures IC + cost-aware P&L at multiple forward horizons.

Usage:
    python scripts/spannung_horizon_sweep.py --data-dir data/features/2026-05-12
    python scripts/spannung_horizon_sweep.py --data-dir data/features/2026-05-12 --symbol BTC
    nat spannung horizon --data data/features/2026-05-12

Output:
    reports/spannung/horizon_sweep_{SYM}.json
    printed matrix of (timeframe × signal × forward horizon) → IC, net Sharpe
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
warnings.filterwarnings("ignore")

from cluster_pipeline.loader import load_parquet

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("spannung_hz")

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS = ["BTC", "ETH", "SOL"]

NEEDED = [
    "timestamp_ns", "symbol", "raw_midprice",
    "imbalance_qty_l1", "illiq_composite", "illiq_kyle_100",
]

# Bar timeframes (pandas freq strings)
TIMEFRAMES = ["30s", "1min", "2min", "5min", "10min", "15min"]

# Forward horizon multipliers (in bars)
FWD_BARS = [1, 2, 4, 8]

# Cost assumptions
TAKER_BPS = 3.5
ROUND_TRIP_TAKER = 2 * TAKER_BPS / 10_000

# Spannung EWM params (in ticks, applied before aggregation)
ALPHA_HL = 5    # 0.5s
BETA_HL = 600   # 60s
EPS = 1e-10


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SweepPoint:
    timeframe: str
    signal: str
    fwd_bars: int
    fwd_seconds: float
    ic_mean: float
    ic_std: float
    ic_ir: float
    n_windows: int
    n_bars: int
    gross_sharpe: float
    net_sharpe_taker: float
    gross_bps: float
    net_bps_taker: float
    avg_return_per_trade_bps: float
    hit_rate: float
    breakeven_bps: float    # fee level where net Sharpe = 0


@dataclass
class SweepResult:
    timestamp: str
    data_dir: str
    symbol: str
    n_rows: int
    duration_hours: float
    points: List[SweepPoint]


# ── Signal computation ────────────────────────────────────────────────────────

def aggregate_to_bars(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aggregate tick data to bars with custom imbalance metrics.

    Returns DataFrame with one row per bar, columns:
        bar_start, close_price, imbalance_mean, imbalance_last,
        imbalance_trend, imbalance_persistence, spannung_mean,
        imbalance_x_illiq
    """
    work = df.copy()
    work["_dt"] = pd.to_datetime(work["timestamp_ns"], unit="ns")
    work = work.set_index("_dt")

    # Pre-compute tick-level Spannung
    flow = work["imbalance_qty_l1"].values.astype(np.float64)
    illiq = work["illiq_composite"].values.astype(np.float64)
    flow_ewm = pd.Series(flow, index=work.index).ewm(halflife=ALPHA_HL, min_periods=3).mean()
    illiq_ewm = pd.Series(illiq, index=work.index).ewm(halflife=BETA_HL, min_periods=100).mean()
    work["_spannung"] = flow_ewm / (illiq_ewm.abs() + EPS)

    # Resample
    resampler = work.resample(freq)

    bars = pd.DataFrame()
    bars["close_price"] = resampler["raw_midprice"].last()
    bars["imbalance_mean"] = resampler["imbalance_qty_l1"].mean()
    bars["imbalance_last"] = resampler["imbalance_qty_l1"].last()
    bars["illiq_mean"] = resampler["illiq_composite"].mean()
    bars["spannung_mean"] = resampler["_spannung"].mean()
    bars["tick_count"] = resampler["raw_midprice"].count()

    # Imbalance trend (slope over bar) — use rank of tick within bar as x
    def _slope(group):
        vals = group.values
        n = len(vals)
        if n < 3 or np.all(np.isnan(vals)):
            return np.nan
        x = np.arange(n, dtype=np.float64)
        valid = ~np.isnan(vals)
        if valid.sum() < 3:
            return np.nan
        # Normalize x to [0,1] so slope is comparable across timeframes
        x_v = x[valid] / max(n - 1, 1)
        y_v = vals[valid]
        if np.ptp(y_v) < 1e-15:
            return 0.0
        slope = np.polyfit(x_v, y_v, 1)[0]
        return slope

    bars["imbalance_trend"] = resampler["imbalance_qty_l1"].apply(_slope)

    # Imbalance persistence: fraction of ticks where sign matches bar mean
    def _persistence(group):
        vals = group.values
        valid = vals[~np.isnan(vals)]
        if len(valid) < 3:
            return np.nan
        mean_sign = np.sign(np.mean(valid))
        if mean_sign == 0:
            return 0.5
        return float(np.mean(np.sign(valid) == mean_sign))

    bars["imbalance_persistence"] = resampler["imbalance_qty_l1"].apply(_persistence)

    # Imbalance × illiquidity interaction
    bars["imbalance_x_illiq"] = bars["imbalance_mean"] * bars["illiq_mean"]

    # Drop bars with too few ticks (< 50% expected)
    bars = bars.dropna(subset=["close_price"]).copy()
    bars = bars[bars["tick_count"] > 0]

    return bars.reset_index().rename(columns={"_dt": "bar_start"})


# ── IC and backtest ───────────────────────────────────────────────────────────

def measure_signal(
    signal: np.ndarray,
    prices: np.ndarray,
    fwd_bars: int,
    timeframe_seconds: float,
) -> Optional[SweepPoint]:
    """Compute IC and simple backtest for one signal at one forward horizon."""
    n = len(signal)
    if fwd_bars >= n:
        return None

    # Forward log returns
    fwd_ret = np.full(n, np.nan)
    fwd_ret[:n - fwd_bars] = np.log(prices[fwd_bars:] / np.clip(prices[:n - fwd_bars], 1e-15, None))

    # IC (non-overlapping windows)
    valid = ~(np.isnan(signal) | np.isnan(fwd_ret))
    # Window size: ~20 bars minimum, ~50 if available
    window = max(20, min(50, n // 10))
    ics = []
    start = 0
    while start + window <= n:
        end = start + window
        mask = valid[start:end]
        if mask.sum() >= 15:
            s, r = signal[start:end][mask], fwd_ret[start:end][mask]
            if np.ptp(s) > 1e-15 and np.ptp(r) > 1e-15:
                rho, _ = stats.spearmanr(s, r)
                ics.append(float(rho) if np.isfinite(rho) else 0.0)
            else:
                ics.append(0.0)
        else:
            ics.append(np.nan)
        start = end

    ic_arr = np.array(ics)
    ic_valid = ic_arr[~np.isnan(ic_arr)]
    if len(ic_valid) < 3:
        return None

    ic_mean = float(np.mean(ic_valid))
    ic_std = float(np.std(ic_valid))
    ic_ir = ic_mean / (ic_std + 1e-10)

    # Simple backtest: position = sign(signal)
    position = np.sign(signal)
    position[np.isnan(signal)] = 0

    gross_pnl = position * fwd_ret
    gross_pnl[np.isnan(gross_pnl)] = 0

    # Costs: each position change incurs one-side fee
    changes = np.zeros(n)
    changes[0] = abs(position[0])
    changes[1:] = np.abs(np.diff(position))
    cost = changes * TAKER_BPS / 10_000
    net_pnl = gross_pnl - cost

    # Annualization
    bar_seconds = timeframe_seconds * fwd_bars  # holding period
    bars_per_year = 365.25 * 24 * 3600 / timeframe_seconds

    def sharpe(pnl):
        if np.std(pnl) < 1e-15:
            return 0.0
        return float(np.mean(pnl) / np.std(pnl) * np.sqrt(bars_per_year))

    # Hit rate
    trading = position != 0
    n_trades = int(trading.sum())
    if n_trades > 0:
        hit = float(np.mean(np.sign(position[trading]) == np.sign(fwd_ret[trading])))
        avg_ret = float(np.mean(gross_pnl[trading]) * 10_000)
    else:
        hit = 0.0
        avg_ret = 0.0

    # Breakeven: find fee level where gross - fee*changes = 0
    total_gross = np.sum(gross_pnl)
    total_changes = np.sum(changes)
    breakeven = (total_gross / total_changes * 10_000) if total_changes > 0 else 0.0

    return SweepPoint(
        timeframe="",  # filled by caller
        signal="",     # filled by caller
        fwd_bars=fwd_bars,
        fwd_seconds=round(timeframe_seconds * fwd_bars, 1),
        ic_mean=round(ic_mean, 4),
        ic_std=round(ic_std, 4),
        ic_ir=round(ic_ir, 3),
        n_windows=len(ic_valid),
        n_bars=n,
        gross_sharpe=round(sharpe(gross_pnl), 1),
        net_sharpe_taker=round(sharpe(net_pnl), 1),
        gross_bps=round(float(np.sum(gross_pnl) * 10_000), 1),
        net_bps_taker=round(float(np.sum(net_pnl) * 10_000), 1),
        avg_return_per_trade_bps=round(avg_ret, 2),
        hit_rate=round(hit, 3),
        breakeven_bps=round(breakeven, 2),
    )


# ── Sweep engine ──────────────────────────────────────────────────────────────

def _tf_seconds(tf: str) -> float:
    """Convert timeframe string to seconds."""
    mapping = {"30s": 30, "1min": 60, "2min": 120, "5min": 300, "10min": 600, "15min": 900}
    return mapping.get(tf, 60)


SIGNAL_NAMES = [
    "imbalance_mean", "imbalance_last", "imbalance_trend",
    "imbalance_persistence", "spannung_mean", "imbalance_x_illiq",
]


def run_sweep(df: pd.DataFrame, symbol: str) -> SweepResult:
    """Run the full horizon sweep for one symbol."""
    t0 = time.time()
    ts = df["timestamp_ns"].values
    duration_h = (ts[-1] - ts[0]) / 1e9 / 3600

    all_points: List[SweepPoint] = []

    for tf in TIMEFRAMES:
        log.info(f"    {tf} bars ...")
        bars = aggregate_to_bars(df, tf)
        if len(bars) < 30:
            log.info(f"      too few bars ({len(bars)}), skipping")
            continue

        prices = bars["close_price"].values
        tf_sec = _tf_seconds(tf)

        for sig_name in SIGNAL_NAMES:
            if sig_name not in bars.columns:
                continue
            signal = bars[sig_name].values.astype(np.float64)

            for fwd in FWD_BARS:
                pt = measure_signal(signal, prices, fwd, tf_sec)
                if pt is None:
                    continue
                pt.timeframe = tf
                pt.signal = sig_name
                all_points.append(pt)

    elapsed = time.time() - t0
    log.info(f"    {len(all_points)} points in {elapsed:.1f}s")

    return SweepResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_dir="",
        symbol=symbol,
        n_rows=len(df),
        duration_hours=round(duration_h, 1),
        points=all_points,
    )


# ── Display ───────────────────────────────────────────────────────────────────

def print_sweep(result: SweepResult):
    """Print sweep results as a summary matrix."""
    print(f"\n{'=' * 115}")
    print(f"  SPANNUNG HORIZON SWEEP — {result.symbol} "
          f"({result.n_rows:,} rows, {result.duration_hours:.1f}h)")
    print(f"  Taker: {TAKER_BPS} bps/side round-trip")
    print(f"{'=' * 115}\n")

    if not result.points:
        print("  No valid results.\n")
        return

    # Group by signal, show best timeframe/horizon for each
    print(f"  {'signal':<25} {'timeframe':>9} {'fwd':>8} {'IC_mean':>8} {'IC_IR':>6} "
          f"{'hit%':>5} {'gross_bps':>9} {'net_bps':>8} {'Sh_grs':>6} {'Sh_net':>6} "
          f"{'bps/trd':>7} {'brkevn':>6} {'bars':>5}")
    print(f"  {'─'*25} {'─'*9} {'─'*8} {'─'*8} {'─'*6} "
          f"{'─'*5} {'─'*9} {'─'*8} {'─'*6} {'─'*6} "
          f"{'─'*7} {'─'*6} {'─'*5}")

    # Sort by net sharpe descending
    sorted_pts = sorted(result.points, key=lambda p: p.net_sharpe_taker, reverse=True)

    for pt in sorted_pts[:40]:
        profitable = " $" if pt.net_sharpe_taker > 0 else "  "
        print(f"{profitable}{pt.signal:<24} {pt.timeframe:>9} {pt.fwd_seconds:>7.0f}s "
              f"{pt.ic_mean:>8.4f} {pt.ic_ir:>6.2f} "
              f"{pt.hit_rate:>5.1%} {pt.gross_bps:>9.1f} {pt.net_bps_taker:>8.1f} "
              f"{pt.gross_sharpe:>6.1f} {pt.net_sharpe_taker:>6.1f} "
              f"{pt.avg_return_per_trade_bps:>7.2f} {pt.breakeven_bps:>6.1f} "
              f"{pt.n_bars:>5}")

    # Highlight profitable combinations
    profitable = [p for p in result.points if p.net_sharpe_taker > 0]
    if profitable:
        print(f"\n  PROFITABLE COMBINATIONS (net Sharpe > 0 after taker fees):")
        for p in sorted(profitable, key=lambda x: x.net_sharpe_taker, reverse=True):
            print(f"    {p.signal:<24} {p.timeframe:>6} fwd={p.fwd_seconds:.0f}s  "
                  f"IC={p.ic_mean:.4f}  Sharpe={p.net_sharpe_taker:.1f}  "
                  f"net={p.net_bps_taker:.0f}bps  brkevn={p.breakeven_bps:.1f}bps")
    else:
        print(f"\n  No combinations profitable after taker fees ({TAKER_BPS} bps/side).")

        # Check at what fee level they'd break even
        best_gross = max(result.points, key=lambda p: p.breakeven_bps)
        print(f"  Best breakeven: {best_gross.breakeven_bps:.1f} bps/side "
              f"({best_gross.signal}, {best_gross.timeframe}, fwd={best_gross.fwd_seconds:.0f}s)")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Spannung longer-horizon sweep")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to parquet data")
    parser.add_argument("--symbol", type=str, default="all", help='Symbol or "all"')
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    symbols = SYMBOLS if args.symbol.lower() == "all" else [args.symbol.upper()]
    out_dir = Path(args.output) if args.output else ROOT / "reports" / "spannung"
    out_dir.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        log.info(f"\n  Loading {sym} from {data_dir} ...")
        try:
            df = load_parquet(str(data_dir), symbols=[sym], columns=NEEDED)
        except Exception as e:
            log.warning(f"  Failed: {e}")
            continue
        if df.empty:
            log.warning(f"  No data for {sym}")
            continue

        df = df.sort_values("timestamp_ns").reset_index(drop=True)
        log.info(f"  {len(df):,} rows, sweeping {len(TIMEFRAMES)} timeframes × "
                 f"{len(SIGNAL_NAMES)} signals × {len(FWD_BARS)} horizons")

        result = run_sweep(df, sym)
        result.data_dir = str(data_dir)
        print_sweep(result)

        out_path = out_dir / f"horizon_sweep_{sym}.json"
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        log.info(f"  Saved: {out_path}")

    log.info("")


if __name__ == "__main__":
    main()
