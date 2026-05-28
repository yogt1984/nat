#!/usr/bin/env python3
"""
Spannung Backtest — cost-aware P&L simulation for L1 imbalance signal.

Strategy:
  At each h-tick interval, observe imbalance_qty_l1.
  If |imbalance| > threshold: take position in sign(imbalance).
  If |imbalance| <= threshold: stay flat.
  P&L per interval = position * forward_log_return - cost_if_position_changed.

Tests multiple threshold levels and cost assumptions (taker vs maker).

Usage:
    python scripts/spannung_backtest.py --data-dir data/features/2026-05-12
    python scripts/spannung_backtest.py --data-dir data/features/2026-05-12 --symbol BTC --horizon 50
    nat spannung backtest --data data/features/2026-05-12

Output:
    reports/spannung/backtest_{SYM}.json
    printed summary with gross/net Sharpe, hit rate, trade count
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
warnings.filterwarnings("ignore")

from cluster_pipeline.loader import load_parquet

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("spannung_bt")

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS = ["BTC", "ETH", "SOL"]
DEFAULT_HORIZONS = {"BTC": 50, "ETH": 50, "SOL": 10}  # ticks (5s, 5s, 1s)

# Hyperliquid fees
TAKER_BPS = 3.5      # per side
MAKER_BPS = 1.0      # per side (if you can post limit orders)
ROUND_TRIP_TAKER = 2 * TAKER_BPS / 10_000  # 7 bps
ROUND_TRIP_MAKER = 2 * MAKER_BPS / 10_000  # 2 bps

# Threshold grid to test
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

NEEDED = ["timestamp_ns", "symbol", "raw_midprice", "imbalance_qty_l1"]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ThresholdResult:
    threshold: float
    n_intervals: int
    n_trades: int            # intervals where position != 0
    n_flips: int             # position changes (each incurs cost)
    trade_fraction: float    # n_trades / n_intervals
    hit_rate: float          # P(sign(position) == sign(return)) when trading
    gross_return_bps: float  # total gross return in bps
    net_return_taker_bps: float
    net_return_maker_bps: float
    gross_sharpe: float      # annualized Sharpe (gross)
    net_sharpe_taker: float
    net_sharpe_maker: float
    avg_return_per_trade_bps: float  # gross return per trade in bps
    max_drawdown_bps: float


@dataclass
class BacktestResult:
    timestamp: str
    data_dir: str
    symbol: str
    horizon_ticks: int
    horizon_seconds: float
    n_rows: int
    duration_hours: float
    thresholds: List[ThresholdResult]


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    horizon: int,
    thresholds: List[float] = THRESHOLDS,
) -> BacktestResult:
    """Run threshold-sweep backtest for one symbol."""

    prices = df["raw_midprice"].values
    imbalance = df["imbalance_qty_l1"].values.astype(np.float64)
    n = len(prices)

    # Build non-overlapping h-tick intervals
    # At each interval start, observe imbalance, compute forward return
    interval_starts = np.arange(0, n - horizon, horizon)
    n_intervals = len(interval_starts)

    # Pre-compute forward log returns at interval boundaries
    fwd_ret = np.log(prices[interval_starts + horizon] / prices[interval_starts])
    imb_at_start = imbalance[interval_starts]

    # Data duration
    ts = df["timestamp_ns"].values
    duration_h = (ts[-1] - ts[0]) / 1e9 / 3600

    # Annualization factor: intervals per year
    interval_sec = horizon * 0.1  # ticks to seconds
    intervals_per_year = 365.25 * 24 * 3600 / interval_sec

    results = []
    for thresh in thresholds:
        # Position: +1 if imbalance > thresh, -1 if < -thresh, 0 otherwise
        position = np.where(imb_at_start > thresh, 1.0,
                   np.where(imb_at_start < -thresh, -1.0, 0.0))

        # Per-interval P&L (gross)
        gross_pnl = position * fwd_ret

        # Count position changes (flips) for cost calculation
        # First interval: if position != 0, that's an entry
        pos_changes = np.zeros(n_intervals)
        pos_changes[0] = abs(position[0])  # entry cost
        pos_changes[1:] = np.abs(np.diff(position))
        # Each change of 2 (long→short or short→long) = close + open = 2 round trips
        # Each change of 1 (flat→long or long→flat) = 1 side = 0.5 round trip
        # Simplify: cost = |position_change| * one_side_fee
        cost_taker = pos_changes * TAKER_BPS / 10_000
        cost_maker = pos_changes * MAKER_BPS / 10_000

        net_pnl_taker = gross_pnl - cost_taker
        net_pnl_maker = gross_pnl - cost_maker

        # Stats
        trading_mask = position != 0
        n_trades = int(trading_mask.sum())
        n_flips = int((pos_changes > 0).sum())

        if n_trades > 0:
            trade_returns = fwd_ret[trading_mask]
            trade_positions = position[trading_mask]
            hit_rate = float(np.mean(np.sign(trade_positions) == np.sign(trade_returns)))
            avg_ret = float(np.mean(gross_pnl[trading_mask]) * 10_000)
        else:
            hit_rate = 0.0
            avg_ret = 0.0

        # Sharpe
        def sharpe(pnl_arr):
            if np.std(pnl_arr) < 1e-15:
                return 0.0
            return float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(intervals_per_year))

        # Max drawdown
        def max_dd(pnl_arr):
            cumulative = np.cumsum(pnl_arr)
            peak = np.maximum.accumulate(cumulative)
            dd = peak - cumulative
            return float(np.max(dd) * 10_000) if len(dd) > 0 else 0.0

        results.append(ThresholdResult(
            threshold=thresh,
            n_intervals=n_intervals,
            n_trades=n_trades,
            n_flips=n_flips,
            trade_fraction=round(n_trades / n_intervals, 4),
            hit_rate=round(hit_rate, 4),
            gross_return_bps=round(float(np.sum(gross_pnl) * 10_000), 2),
            net_return_taker_bps=round(float(np.sum(net_pnl_taker) * 10_000), 2),
            net_return_maker_bps=round(float(np.sum(net_pnl_maker) * 10_000), 2),
            gross_sharpe=round(sharpe(gross_pnl), 2),
            net_sharpe_taker=round(sharpe(net_pnl_taker), 2),
            net_sharpe_maker=round(sharpe(net_pnl_maker), 2),
            avg_return_per_trade_bps=round(avg_ret, 3),
            max_drawdown_bps=round(max_dd(gross_pnl), 2),
        ))

    return BacktestResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_dir="",
        symbol=symbol,
        horizon_ticks=horizon,
        horizon_seconds=horizon * 0.1,
        n_rows=n,
        duration_hours=round(duration_h, 1),
        thresholds=results,
    )


# ── Display ───────────────────────────────────────────────────────────────────

def print_backtest(result: BacktestResult):
    """Print backtest results."""
    print(f"\n{'=' * 105}")
    print(f"  SPANNUNG BACKTEST — {result.symbol} "
          f"(h={result.horizon_ticks} ticks = {result.horizon_seconds}s, "
          f"{result.n_rows:,} rows, {result.duration_hours}h)")
    print(f"  Taker: {TAKER_BPS} bps/side, Maker: {MAKER_BPS} bps/side")
    print(f"{'=' * 105}\n")

    print(f"  {'thresh':>6}  {'trades':>6}  {'trade%':>6}  {'flips':>5}  "
          f"{'hit%':>5}  {'gross_bps':>9}  {'net_tkr':>8}  {'net_mkr':>8}  "
          f"{'Sh_grs':>6}  {'Sh_tkr':>6}  {'Sh_mkr':>6}  {'bps/trd':>7}  {'maxDD':>7}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}  "
          f"{'─'*5}  {'─'*9}  {'─'*8}  {'─'*8}  "
          f"{'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")

    for t in result.thresholds:
        marker = " *" if t.net_sharpe_taker > 0 else "  "
        print(f"{marker}{t.threshold:>5.1f}  {t.n_trades:>6}  {t.trade_fraction:>6.1%}  "
              f"{t.n_flips:>5}  {t.hit_rate:>5.1%}  "
              f"{t.gross_return_bps:>9.1f}  {t.net_return_taker_bps:>8.1f}  "
              f"{t.net_return_maker_bps:>8.1f}  "
              f"{t.gross_sharpe:>6.1f}  {t.net_sharpe_taker:>6.1f}  "
              f"{t.net_sharpe_maker:>6.1f}  {t.avg_return_per_trade_bps:>7.2f}  "
              f"{t.max_drawdown_bps:>7.1f}")

    # Find best threshold for each cost regime
    best_taker = max(result.thresholds, key=lambda t: t.net_sharpe_taker)
    best_maker = max(result.thresholds, key=lambda t: t.net_sharpe_maker)
    print(f"\n  Best (taker): thresh={best_taker.threshold:.1f}, "
          f"Sharpe={best_taker.net_sharpe_taker:.1f}, "
          f"net={best_taker.net_return_taker_bps:.1f} bps, "
          f"trades={best_taker.n_trades}")
    print(f"  Best (maker): thresh={best_maker.threshold:.1f}, "
          f"Sharpe={best_maker.net_sharpe_maker:.1f}, "
          f"net={best_maker.net_return_maker_bps:.1f} bps, "
          f"trades={best_maker.n_trades}")
    print()


# ── Regime gating ─────────────────────────────────────────────────────────────

GATE_NEEDED = NEEDED + [
    "ent_tick_1m", "toxic_vpin_50", "vol_returns_1m", "illiq_kyle_100",
]


@dataclass
class RegimeResult:
    regime_name: str
    condition: str
    n_intervals: int
    n_trades: int
    hit_rate: float
    gross_return_bps: float
    net_return_taker_bps: float
    gross_sharpe: float
    net_sharpe_taker: float
    ic_in_regime: float


def run_regime_gating(
    df: pd.DataFrame,
    symbol: str,
    horizon: int,
    threshold: float = 0.3,
) -> List[RegimeResult]:
    """Test imbalance signal conditioned on entropy/toxicity/vol/illiq regimes."""

    prices = df["raw_midprice"].values
    imbalance = df["imbalance_qty_l1"].values.astype(np.float64)
    n = len(prices)

    # Extract regime features (with safe fallbacks)
    def safe_col(name):
        if name in df.columns:
            v = df[name].values.astype(np.float64)
            return v if np.isnan(v).mean() < 0.5 else np.full(n, np.nan)
        return np.full(n, np.nan)

    entropy = safe_col("ent_tick_1m")
    vpin = safe_col("toxic_vpin_50")
    vol = safe_col("vol_returns_1m")
    kyle = safe_col("illiq_kyle_100")

    # Non-overlapping intervals
    interval_starts = np.arange(0, n - horizon, horizon)
    n_intervals = len(interval_starts)
    fwd_ret = np.log(prices[interval_starts + horizon] / prices[interval_starts])
    imb = imbalance[interval_starts]

    # Regime features at interval starts
    ent_iv = entropy[interval_starts]
    vpin_iv = vpin[interval_starts]
    vol_iv = vol[interval_starts]
    kyle_iv = kyle[interval_starts]

    # Compute medians for splitting
    ent_med = np.nanmedian(ent_iv)
    vpin_med = np.nanmedian(vpin_iv)
    vol_med = np.nanmedian(vol_iv)
    kyle_med = np.nanmedian(kyle_iv)

    interval_sec = horizon * 0.1
    intervals_per_year = 365.25 * 24 * 3600 / interval_sec

    def evaluate_regime(name, condition_str, mask):
        """Evaluate the imbalance signal within a regime subset."""
        if mask.sum() < 100:
            return RegimeResult(name, condition_str, int(mask.sum()), 0, 0, 0, 0, 0, 0, 0)

        imb_r = imb[mask]
        ret_r = fwd_ret[mask]
        pos = np.where(imb_r > threshold, 1.0,
              np.where(imb_r < -threshold, -1.0, 0.0))

        gross = pos * ret_r
        changes = np.zeros(len(pos))
        changes[0] = abs(pos[0])
        changes[1:] = np.abs(np.diff(pos))
        cost = changes * TAKER_BPS / 10_000
        net = gross - cost

        trading = pos != 0
        n_trades = int(trading.sum())
        hit = float(np.mean(np.sign(pos[trading]) == np.sign(ret_r[trading]))) if n_trades > 0 else 0

        def sharpe(a):
            return float(np.mean(a) / np.std(a) * np.sqrt(intervals_per_year)) if np.std(a) > 1e-15 else 0

        # IC in this regime
        from scipy import stats as sp_stats
        valid = ~(np.isnan(imb_r) | np.isnan(ret_r))
        if valid.sum() > 30:
            rho, _ = sp_stats.spearmanr(imb_r[valid], ret_r[valid])
            ic = float(rho) if np.isfinite(rho) else 0
        else:
            ic = 0

        return RegimeResult(
            regime_name=name,
            condition=condition_str,
            n_intervals=int(mask.sum()),
            n_trades=n_trades,
            hit_rate=round(hit, 4),
            gross_return_bps=round(float(np.sum(gross) * 10_000), 2),
            net_return_taker_bps=round(float(np.sum(net) * 10_000), 2),
            gross_sharpe=round(sharpe(gross), 2),
            net_sharpe_taker=round(sharpe(net), 2),
            ic_in_regime=round(ic, 4),
        )

    # Define regime splits
    regimes = [
        ("ALL (baseline)", "no filter", np.ones(n_intervals, dtype=bool)),
        # Entropy regimes
        ("Low entropy", f"ent < {ent_med:.2f}", ent_iv < ent_med),
        ("High entropy", f"ent >= {ent_med:.2f}", ent_iv >= ent_med),
        # Toxicity regimes
        ("Low VPIN", f"vpin < {vpin_med:.2f}", vpin_iv < vpin_med),
        ("High VPIN", f"vpin >= {vpin_med:.2f}", vpin_iv >= vpin_med),
        # Volatility regimes
        ("Low vol", f"vol < {vol_med:.4f}", vol_iv < vol_med),
        ("High vol", f"vol >= {vol_med:.4f}", vol_iv >= vol_med),
        # Illiquidity regimes
        ("Low illiq", f"kyle < {kyle_med:.2f}", kyle_iv < kyle_med),
        ("High illiq", f"kyle >= {kyle_med:.2f}", kyle_iv >= kyle_med),
        # Combinations (the money shots)
        ("Low ent + High illiq", "informed + fragile",
         (ent_iv < ent_med) & (kyle_iv >= kyle_med)),
        ("High ent + Low illiq", "noisy + deep",
         (ent_iv >= ent_med) & (kyle_iv < kyle_med)),
        ("Low ent + High VPIN", "informed + toxic",
         (ent_iv < ent_med) & (vpin_iv >= vpin_med)),
        ("High ent + Low VPIN", "noisy + clean",
         (ent_iv >= ent_med) & (vpin_iv < vpin_med)),
        ("Low ent + Low vol", "structured + calm",
         (ent_iv < ent_med) & (vol_iv < vol_med)),
        ("High vol + High illiq", "volatile + fragile",
         (vol_iv >= vol_med) & (kyle_iv >= kyle_med)),
    ]

    return [evaluate_regime(name, cond, mask) for name, cond, mask in regimes]


def print_regime_results(symbol: str, results: List[RegimeResult], horizon: int):
    """Print regime gating results."""
    print(f"\n{'=' * 105}")
    print(f"  REGIME GATING — {symbol} (h={horizon} ticks = {horizon*0.1}s, threshold=0.3)")
    print(f"{'=' * 105}\n")

    print(f"  {'Regime':<25} {'Condition':<22} {'N':>5}  {'trades':>6}  "
          f"{'hit%':>5}  {'IC':>6}  {'gross_bps':>9}  {'net_tkr':>8}  "
          f"{'Sh_grs':>6}  {'Sh_tkr':>6}")
    print(f"  {'─'*25} {'─'*22} {'─'*5}  {'─'*6}  "
          f"{'─'*5}  {'─'*6}  {'─'*9}  {'─'*8}  "
          f"{'─'*6}  {'─'*6}")

    for r in results:
        marker = " *" if r.net_sharpe_taker > results[0].net_sharpe_taker else "  "
        print(f"{marker}{r.regime_name:<24} {r.condition:<22} {r.n_intervals:>5}  "
              f"{r.n_trades:>6}  {r.hit_rate:>5.1%}  {r.ic_in_regime:>6.3f}  "
              f"{r.gross_return_bps:>9.1f}  {r.net_return_taker_bps:>8.1f}  "
              f"{r.gross_sharpe:>6.1f}  {r.net_sharpe_taker:>6.1f}")

    # Highlight best regime
    best = max(results[1:], key=lambda r: r.net_sharpe_taker)  # skip baseline
    baseline = results[0]
    print(f"\n  Best regime: {best.regime_name} — "
          f"Sharpe {baseline.net_sharpe_taker:.1f} → {best.net_sharpe_taker:.1f} "
          f"(IC {baseline.ic_in_regime:.3f} → {best.ic_in_regime:.3f})")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Spannung cost-aware backtest + regime gating")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to parquet data")
    parser.add_argument("--symbol", type=str, default="all", help='Symbol or "all"')
    parser.add_argument("--horizon", type=int, default=None,
                        help="Horizon in ticks (default: 50 for BTC/ETH, 10 for SOL)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    symbols = SYMBOLS if args.symbol.lower() == "all" else [args.symbol.upper()]
    out_dir = Path(args.output) if args.output else ROOT / "reports" / "spannung"
    out_dir.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        horizon = args.horizon or DEFAULT_HORIZONS.get(sym, 50)

        log.info(f"\n  Loading {sym} from {data_dir} ...")
        try:
            df = load_parquet(str(data_dir), symbols=[sym], columns=GATE_NEEDED)
        except Exception as e:
            log.warning(f"  Failed: {e}")
            continue
        if df.empty:
            log.warning(f"  No data for {sym}")
            continue

        df = df.sort_values("timestamp_ns").reset_index(drop=True)
        log.info(f"  {len(df):,} rows")

        # Part 1: Cost-aware backtest
        bt = run_backtest(df, sym, horizon)
        bt.data_dir = str(data_dir)
        print_backtest(bt)

        bt_path = out_dir / f"backtest_{sym}.json"
        with open(bt_path, "w") as f:
            json.dump(asdict(bt), f, indent=2)

        # Part 2: Regime gating
        regime_results = run_regime_gating(df, sym, horizon, threshold=0.3)
        print_regime_results(sym, regime_results, horizon)

        regime_path = out_dir / f"regime_{sym}.json"
        with open(regime_path, "w") as f:
            json.dump([asdict(r) for r in regime_results], f, indent=2)

    log.info(f"  Results saved to {out_dir}/\n")


if __name__ == "__main__":
    main()
