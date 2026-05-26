#!/usr/bin/env python3
"""
Spannung Offline Grid Search — parameter landscape exploration.

Spannung(t) = EWM_alpha[signed_flow(t)] / (EWM_beta[illiq(t)] + eps)

Scans a grid of (alpha, beta, flow_feature, illiq_feature, horizon) and
measures predictive power via Spearman IC against forward returns.

Usage:
    python scripts/spannung_grid.py --data-dir data/features/2026-05-12
    python scripts/spannung_grid.py --data-dir data/features/2026-05-12 --symbol BTC --horizons 10 50 100
    nat spannung --data data/features/2026-05-12

Output:
    reports/spannung_grid.json  — full grid results
    printed summary             — top parameter combos ranked by IC
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from cluster_pipeline.loader import load_parquet

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("spannung")

# ── Configuration ─────────────────────────────────────────────────────────────

# Flow features (numerator candidates) — directional order pressure
FLOW_FEATURES = [
    "flow_aggressor_ratio_5s",   # directional conviction, range ~[0,1]
    "toxic_flow_imbalance",      # signed directional pressure
    "imbalance_qty_l1",          # level-1 order book imbalance
]

# Illiquidity features (denominator candidates) — absorption capacity
ILLIQ_FEATURES = [
    "illiq_kyle_100",            # price impact per unit flow (100-tick)
    "illiq_kyle_500",            # price impact per unit flow (500-tick)
    "illiq_composite",           # multi-measure illiquidity composite
]

# EWM halflife grid (in ticks = 100ms each)
# 0.5s=5, 1s=10, 2s=20, 5s=50, 10s=100, 30s=300
ALPHA_HALFLIFES = [5, 10, 20, 50, 100, 300]    # flow decay
BETA_HALFLIFES = [10, 50, 100, 300, 600]        # illiq decay

# Forward return horizons (in ticks = 100ms each)
# 1s=10, 5s=50, 10s=100, 30s=300, 60s=600
HORIZON_TICKS = [10, 50, 100, 300, 600]

# IC computation
IC_WINDOW = 3000       # ticks per IC window (~5 min at 10/sec)
IC_MIN_OBS = 100       # minimum valid observations per window
EPS = 1e-10            # regularization for illiq denominator


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class GridPoint:
    """Result for one parameter combination."""
    flow_feature: str
    illiq_feature: str
    alpha_halflife: int      # ticks
    beta_halflife: int       # ticks
    horizon: int             # ticks
    ic_mean: float
    ic_std: float
    ic_ir: float             # ic_mean / ic_std
    ic_median: float
    ic_hit_rate: float       # fraction of windows with same sign as mean
    n_windows: int
    turnover: float
    alpha_seconds: float     # alpha in seconds for readability
    beta_seconds: float      # beta in seconds for readability
    horizon_seconds: float   # horizon in seconds for readability


@dataclass
class GridResult:
    """Complete grid search result."""
    timestamp: str
    data_dir: str
    symbol: str
    n_rows: int
    duration_seconds: float
    n_combinations: int
    n_evaluated: int
    elapsed_seconds: float
    grid: List[GridPoint]


# ── Core computation ──────────────────────────────────────────────────────────

def compute_spannung(
    flow: np.ndarray,
    illiq: np.ndarray,
    alpha_halflife: int,
    beta_halflife: int,
) -> np.ndarray:
    """
    Compute Spannung = EWM_alpha(flow) / (EWM_beta(illiq) + eps).

    Parameters
    ----------
    flow : array of signed flow values (numerator)
    illiq : array of illiquidity values (denominator)
    alpha_halflife : EWM halflife for flow in ticks
    beta_halflife : EWM halflife for illiquidity in ticks

    Returns
    -------
    Spannung time series (same length as input)
    """
    flow_s = pd.Series(flow)
    illiq_s = pd.Series(illiq)

    flow_ewm = flow_s.ewm(halflife=alpha_halflife, min_periods=max(alpha_halflife // 2, 1)).mean().values
    illiq_ewm = illiq_s.ewm(halflife=beta_halflife, min_periods=max(beta_halflife // 2, 1)).mean().values

    # Illiquidity should be positive — take abs to handle sign conventions
    illiq_abs = np.abs(illiq_ewm) + EPS

    return flow_ewm / illiq_abs


def compute_forward_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Forward log return: log(p(t+h) / p(t))."""
    n = len(prices)
    fwd = np.full(n, np.nan)
    if horizon >= n:
        return fwd
    fwd[:n - horizon] = np.log(prices[horizon:] / np.clip(prices[:n - horizon], 1e-15, None))
    return fwd


def compute_rolling_ic(
    signal: np.ndarray,
    returns: np.ndarray,
    window: int,
    min_obs: int,
) -> np.ndarray:
    """Non-overlapping rolling Spearman IC."""
    n = len(signal)
    valid = ~(np.isnan(signal) | np.isnan(returns))
    ic_values = []

    start = 0
    while start + window <= n:
        end = start + window
        mask = valid[start:end]
        n_valid = mask.sum()

        if n_valid >= min_obs:
            s = signal[start:end][mask]
            r = returns[start:end][mask]
            if np.ptp(s) < 1e-15 or np.ptp(r) < 1e-15:
                ic_values.append(0.0)
            else:
                rho, _ = stats.spearmanr(s, r)
                ic_values.append(float(rho) if np.isfinite(rho) else 0.0)
        else:
            ic_values.append(np.nan)

        start = end

    return np.array(ic_values)


def compute_turnover(signal: np.ndarray) -> float:
    """Signal turnover: mean|delta(signal)| / std(signal)."""
    valid = signal[~np.isnan(signal)]
    if len(valid) < 2:
        return np.nan
    sigma = np.std(valid)
    if sigma < 1e-15:
        return 0.0
    return float(np.mean(np.abs(np.diff(valid))) / sigma)


# ── Grid search ───────────────────────────────────────────────────────────────

def run_grid(
    df: pd.DataFrame,
    symbol: str,
    alpha_halflifes: List[int] = ALPHA_HALFLIFES,
    beta_halflifes: List[int] = BETA_HALFLIFES,
    horizons: List[int] = HORIZON_TICKS,
    flow_features: List[str] = FLOW_FEATURES,
    illiq_features: List[str] = ILLIQ_FEATURES,
) -> GridResult:
    """Run the full grid search for one symbol."""

    t0 = time.time()
    prices = df["raw_midprice"].values

    # Pre-extract feature columns, skip missing
    flow_data = {}
    for f in flow_features:
        if f in df.columns:
            vals = df[f].values.astype(np.float64)
            if np.isnan(vals).mean() < 0.5:  # skip if >50% NaN
                flow_data[f] = vals
    illiq_data = {}
    for f in illiq_features:
        if f in df.columns:
            vals = df[f].values.astype(np.float64)
            if np.isnan(vals).mean() < 0.5:
                illiq_data[f] = vals

    if not flow_data:
        log.error("No usable flow features found")
        sys.exit(1)
    if not illiq_data:
        log.error("No usable illiquidity features found")
        sys.exit(1)

    log.info(f"  Flow features:  {list(flow_data.keys())}")
    log.info(f"  Illiq features: {list(illiq_data.keys())}")

    # Pre-compute forward returns for all horizons
    fwd_returns = {}
    for h in horizons:
        fwd_returns[h] = compute_forward_returns(prices, h)

    # Count total combinations
    combos = list(product(
        flow_data.keys(), illiq_data.keys(),
        alpha_halflifes, beta_halflifes, horizons,
    ))
    n_total = len(combos)
    log.info(f"  Grid: {n_total} combinations "
             f"({len(flow_data)} flow × {len(illiq_data)} illiq × "
             f"{len(alpha_halflifes)} α × {len(beta_halflifes)} β × "
             f"{len(horizons)} horizons)")

    results: List[GridPoint] = []
    evaluated = 0

    for i, (ff, ilf, alpha, beta, h) in enumerate(combos):
        spannung = compute_spannung(flow_data[ff], illiq_data[ilf], alpha, beta)
        fwd = fwd_returns[h]

        ic_arr = compute_rolling_ic(spannung, fwd, IC_WINDOW, IC_MIN_OBS)
        ic_valid = ic_arr[~np.isnan(ic_arr)]

        if len(ic_valid) < 3:
            continue

        ic_mean = float(np.mean(ic_valid))
        ic_std = float(np.std(ic_valid))
        ic_ir = ic_mean / (ic_std + 1e-10)
        ic_median = float(np.median(ic_valid))
        ic_hit = float(np.mean(np.sign(ic_valid) == np.sign(ic_mean))) if ic_mean != 0 else 0.0

        results.append(GridPoint(
            flow_feature=ff,
            illiq_feature=ilf,
            alpha_halflife=alpha,
            beta_halflife=beta,
            horizon=h,
            ic_mean=round(ic_mean, 6),
            ic_std=round(ic_std, 6),
            ic_ir=round(ic_ir, 4),
            ic_median=round(ic_median, 6),
            ic_hit_rate=round(ic_hit, 4),
            n_windows=len(ic_valid),
            turnover=round(compute_turnover(spannung), 4),
            alpha_seconds=alpha * 0.1,
            beta_seconds=beta * 0.1,
            horizon_seconds=h * 0.1,
        ))
        evaluated += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            log.info(f"  ... {i + 1}/{n_total} ({elapsed:.0f}s)")

    # Sort by |IC mean| descending
    results.sort(key=lambda g: abs(g.ic_mean), reverse=True)

    # Data duration
    ts = df["timestamp_ns"].values
    duration_s = (ts[-1] - ts[0]) / 1e9

    elapsed = time.time() - t0
    log.info(f"  Done: {evaluated}/{n_total} evaluated in {elapsed:.1f}s")

    return GridResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_dir="",  # filled by caller
        symbol=symbol,
        n_rows=len(df),
        duration_seconds=round(duration_s, 1),
        n_combinations=n_total,
        n_evaluated=evaluated,
        elapsed_seconds=round(elapsed, 1),
        grid=results,
    )


# ── Display ───────────────────────────────────────────────────────────────────

def print_summary(result: GridResult, top_n: int = 20):
    """Print top parameter combinations."""
    print(f"\n{'=' * 100}")
    print(f"  SPANNUNG GRID SEARCH — {result.symbol}")
    print(f"  {result.n_rows:,} rows, {result.duration_seconds:.0f}s of data, "
          f"{result.n_evaluated}/{result.n_combinations} combos evaluated")
    print(f"{'=' * 100}\n")

    if not result.grid:
        print("  No valid results.\n")
        return

    # Header
    print(f"  {'#':>3}  {'IC_mean':>8}  {'IC_IR':>7}  {'IC_hit':>6}  "
          f"{'α(s)':>6}  {'β(s)':>6}  {'h(s)':>6}  "
          f"{'flow':>28}  {'illiq':>18}  {'windows':>7}  {'turnover':>8}")
    print(f"  {'─' * 3}  {'─' * 8}  {'─' * 7}  {'─' * 6}  "
          f"{'─' * 6}  {'─' * 6}  {'─' * 6}  "
          f"{'─' * 28}  {'─' * 18}  {'─' * 7}  {'─' * 8}")

    for i, g in enumerate(result.grid[:top_n]):
        marker = "**" if abs(g.ic_mean) >= 0.05 else "  "
        print(f"{marker}{i + 1:>3}  {g.ic_mean:>8.4f}  {g.ic_ir:>7.3f}  {g.ic_hit_rate:>6.2f}  "
              f"{g.alpha_seconds:>6.1f}  {g.beta_seconds:>6.1f}  {g.horizon_seconds:>6.1f}  "
              f"{g.flow_feature:>28}  {g.illiq_feature:>18}  {g.n_windows:>7}  {g.turnover:>8.3f}")

    # Summary statistics
    top10 = result.grid[:10]
    if top10:
        best = top10[0]
        print(f"\n  Best: IC={best.ic_mean:.4f}, IR={best.ic_ir:.3f} "
              f"(α={best.alpha_seconds}s, β={best.beta_seconds}s, h={best.horizon_seconds}s, "
              f"{best.flow_feature} / {best.illiq_feature})")

        ics = [abs(g.ic_mean) for g in result.grid]
        above_005 = sum(1 for ic in ics if ic >= 0.05)
        above_003 = sum(1 for ic in ics if ic >= 0.03)
        print(f"  |IC| >= 0.05: {above_005}/{len(ics)}   |IC| >= 0.03: {above_003}/{len(ics)}")

    # Feature breakdown — which flow/illiq features dominate the top 20
    if len(result.grid) >= 10:
        top20 = result.grid[:20]
        flow_counts = {}
        illiq_counts = {}
        for g in top20:
            flow_counts[g.flow_feature] = flow_counts.get(g.flow_feature, 0) + 1
            illiq_counts[g.illiq_feature] = illiq_counts.get(g.illiq_feature, 0) + 1
        print(f"\n  Top-20 flow dominance:  {flow_counts}")
        print(f"  Top-20 illiq dominance: {illiq_counts}")

        # Horizon breakdown
        h_counts = {}
        for g in top20:
            h_counts[g.horizon_seconds] = h_counts.get(g.horizon_seconds, 0) + 1
        print(f"  Top-20 horizon dist:   {h_counts}")

    print()


# ── Data directory resolution ─────────────────────────────────────────────────

SYMBOLS = ["BTC", "ETH", "SOL"]


def resolve_data_dir(data_dir: Path) -> Path:
    """
    If data_dir points to the top-level features/ dir (contains date subdirs),
    pick the date subdir with the most data (largest total parquet size).
    If it already points to a date subdir, use it directly.
    """
    # Check if it contains date-named subdirs (YYYY-MM-DD)
    subdirs = [
        d for d in data_dir.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name[4] == '-'
    ]
    if subdirs:
        # Pick the subdir with the largest total parquet size
        best, best_size = None, 0
        for d in subdirs:
            total = sum(f.stat().st_size for f in d.glob("*.parquet") if f.stat().st_size > 0)
            if total > best_size:
                best, best_size = d, total
        if best:
            return best
    # Already a date subdir or flat dir with parquets
    return data_dir


def run_all_symbols(
    data_dir: Path,
    symbols: List[str],
    top_n: int,
    output_dir: Optional[Path],
    alpha_halflifes: List[int],
    beta_halflifes: List[int],
    horizons: List[int],
):
    """Run grid search for multiple symbols, save per-symbol + combined results."""

    needed = (
        ["timestamp_ns", "symbol", "raw_midprice"]
        + FLOW_FEATURES + ILLIQ_FEATURES
    )

    # Output directory
    out_dir = output_dir or ROOT / "reports" / "spannung"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for sym in symbols:
        log.info(f"\n  Loading {sym} from {data_dir} ...")
        try:
            df_sym = load_parquet(str(data_dir), symbols=[sym], columns=needed)
        except Exception as e:
            log.warning(f"  Failed to load {sym}: {e}")
            continue

        if df_sym.empty:
            log.warning(f"  No data for {sym}, skipping")
            continue

        df_sym = df_sym.sort_values("timestamp_ns").reset_index(drop=True)
        log.info(f"  {sym}: {len(df_sym):,} rows")

        result = run_grid(
            df_sym, sym,
            alpha_halflifes=alpha_halflifes,
            beta_halflifes=beta_halflifes,
            horizons=horizons,
        )
        result.data_dir = str(data_dir)

        print_summary(result, top_n=top_n)

        # Save per-symbol JSON
        sym_path = out_dir / f"spannung_{sym}.json"
        with open(sym_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        log.info(f"  Saved: {sym_path}")

        all_results.append(result)
        del df_sym  # free memory before next symbol

    # Print cross-symbol comparison
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print(f"  CROSS-SYMBOL COMPARISON")
        print(f"{'=' * 80}\n")
        print(f"  {'Symbol':>6}  {'Best IC':>8}  {'Best IR':>8}  {'Best α(s)':>9}  "
              f"{'Best β(s)':>9}  {'Best h(s)':>9}  {'IC>0.05':>8}  {'flow':>20}")
        print(f"  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 9}  {'─' * 9}  "
              f"{'─' * 9}  {'─' * 8}  {'─' * 20}")
        for r in all_results:
            b = r.grid[0] if r.grid else None
            n05 = sum(1 for g in r.grid if abs(g.ic_mean) >= 0.05)
            if b:
                print(f"  {r.symbol:>6}  {b.ic_mean:>8.4f}  {b.ic_ir:>8.3f}  "
                      f"{b.alpha_seconds:>9.1f}  {b.beta_seconds:>9.1f}  "
                      f"{b.horizon_seconds:>9.1f}  {n05:>4}/{len(r.grid):<3}  "
                      f"{b.flow_feature:>20}")
        print()

    # Save combined summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "symbols": [r.symbol for r in all_results],
        "per_symbol": {
            r.symbol: {
                "n_rows": r.n_rows,
                "duration_seconds": r.duration_seconds,
                "best_ic": r.grid[0].ic_mean if r.grid else None,
                "best_ir": r.grid[0].ic_ir if r.grid else None,
                "best_params": asdict(r.grid[0]) if r.grid else None,
                "n_above_005": sum(1 for g in r.grid if abs(g.ic_mean) >= 0.05),
                "n_total": len(r.grid),
            }
            for r in all_results
        },
    }
    combined_path = out_dir / "spannung_summary.json"
    with open(combined_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"  Combined summary: {combined_path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Spannung offline grid search")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to data dir (date subdir or top-level features/)")
    parser.add_argument("--symbol", type=str, default="all",
                        help='Symbol: BTC, ETH, SOL, or "all" (default: all)')
    parser.add_argument("--alphas", type=int, nargs="+", default=ALPHA_HALFLIFES,
                        help="Flow EWM halflife grid (ticks, 1 tick=100ms)")
    parser.add_argument("--betas", type=int, nargs="+", default=BETA_HALFLIFES,
                        help="Illiq EWM halflife grid (ticks)")
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZON_TICKS,
                        help="Forward return horizons (ticks)")
    parser.add_argument("--top", type=int, default=20, help="Number of top results to display")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    data_dir = resolve_data_dir(Path(args.data_dir))
    if not data_dir.exists():
        log.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    log.info(f"  Data dir resolved: {data_dir}")

    symbols = SYMBOLS if args.symbol.lower() == "all" else [args.symbol.upper()]
    output_dir = Path(args.output) if args.output else None

    run_all_symbols(
        data_dir, symbols,
        top_n=args.top,
        output_dir=output_dir,
        alpha_halflifes=args.alphas,
        beta_halflifes=args.betas,
        horizons=args.horizons,
    )


if __name__ == "__main__":
    main()
