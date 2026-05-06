#!/usr/bin/env python3
"""
EAMM CLI — Entropy-Adaptive Market Making Pipeline

Usage:
    python -m eamm.cli run --symbol BTC --horizon 3000
    python -m eamm.cli regime --symbol BTC --horizon 3000
    python -m eamm.cli backtest --symbol BTC --horizon 3000
"""

import argparse
import glob
import sys
import numpy as np
import polars as pl
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eamm.simulator import simulate_mm, pnl_to_bps, DEFAULT_SPREAD_LEVELS_BPS
from eamm.features import extract_context, context_to_numpy, CONTEXT_FEATURE_NAMES
from eamm.labels import compute_labels, compute_continuous_optimal, label_distribution
from eamm.train import train_eamm, predict_spread
from eamm.evaluate import walk_forward_evaluate, print_evaluation_report
from eamm.regime_analysis import analyze_regimes, print_regime_report
from eamm.backtest import run_backtest, print_backtest_report


def load_data(data_dir: str, symbol: str) -> pl.DataFrame:
    """Load and concatenate all parquet files for a symbol."""
    files = sorted(glob.glob(f"{data_dir}/**/*.parquet", recursive=True))
    dfs = []
    for f in files:
        try:
            df = pl.read_parquet(f)
            dfs.append(df.filter(pl.col("symbol") == symbol))
        except Exception as e:
            print(f"  SKIP {f}: {e}")
    if not dfs:
        print(f"ERROR: no data found for {symbol} in {data_dir}")
        sys.exit(1)
    df = pl.concat(dfs).sort("timestamp_ns")
    df = df.unique(subset=["timestamp_ns"], keep="first")
    print(f"Loaded {df.shape[0]:,} rows for {symbol}")
    return df


def cmd_run(args):
    """Full pipeline: simulate → label → features → train → evaluate."""
    print("=" * 70)
    print("EAMM FULL PIPELINE")
    print("=" * 70)

    df = load_data(args.data_dir, args.symbol)
    spreads = DEFAULT_SPREAD_LEVELS_BPS

    # Step 1: Simulate
    print(f"\n[1/5] Simulating MM at {len(spreads)} spread levels, horizon={args.horizon}...")
    sim = simulate_mm(df, spread_levels_bps=spreads, horizon=args.horizon)
    pnl_bps = pnl_to_bps(sim)

    # Step 2: Labels
    print("[2/5] Computing optimal spread labels...")
    labels_df = compute_labels(sim)
    cont_optimal = compute_continuous_optimal(sim)
    dist = label_distribution(labels_df, len(spreads))
    print(f"  Label distribution entropy: {dist['normalized_entropy']:.3f} "
          f"(1.0=uniform, degenerate={dist['is_degenerate']})")
    print(f"  Counts: {dist['counts']}")

    # Step 3: Features
    print("[3/5] Extracting 19-dim context features...")
    ctx_df = extract_context(df)
    X = context_to_numpy(ctx_df)

    # Align: trim to valid rows (same length as labels)
    valid_end = len(df) - args.horizon
    X = X[:valid_end]
    pnl_valid = pnl_bps[:valid_end]
    fill_rt_valid = sim.fill_round_trip[:valid_end]

    # Step 4: Train + Evaluate
    print("[4/5] Walk-forward evaluation (5 splits)...")
    eval_result = walk_forward_evaluate(
        context_matrix=X,
        pnl_matrix_bps=pnl_valid,
        fill_rt_matrix=fill_rt_valid,
        optimal_spread_bps=cont_optimal[:len(X)],
        spread_levels_bps=spreads,
        feature_names=CONTEXT_FEATURE_NAMES,
        n_splits=5,
        mode=args.mode,
    )
    print_evaluation_report(eval_result)

    # Step 5: Regime analysis
    print("\n[5/5] Entropy regime analysis...")
    entropy_col = ctx_df["H_tick_30s"].to_numpy()[:valid_end]
    regime_result = analyze_regimes(
        entropy_values=entropy_col,
        pnl_matrix=pnl_valid,
        fill_bid_matrix=sim.fill_bid[:valid_end],
        fill_ask_matrix=sim.fill_ask[:valid_end],
        fill_rt_matrix=fill_rt_valid,
        spread_levels_bps=spreads,
        optimal_spread_bps=cont_optimal[:len(X)],
    )
    print_regime_report(regime_result, spreads)


def cmd_regime(args):
    """Run only the regime analysis (thesis test)."""
    df = load_data(args.data_dir, args.symbol)
    spreads = DEFAULT_SPREAD_LEVELS_BPS

    print(f"\nSimulating MM, horizon={args.horizon}...")
    sim = simulate_mm(df, spread_levels_bps=spreads, horizon=args.horizon)
    pnl_bps = pnl_to_bps(sim)

    cont_optimal = compute_continuous_optimal(sim)
    ctx_df = extract_context(df)

    valid_end = len(df) - args.horizon
    entropy_col = ctx_df["H_tick_30s"].to_numpy()[:valid_end]

    result = analyze_regimes(
        entropy_values=entropy_col,
        pnl_matrix=pnl_bps[:valid_end],
        fill_bid_matrix=sim.fill_bid[:valid_end],
        fill_ask_matrix=sim.fill_ask[:valid_end],
        fill_rt_matrix=sim.fill_round_trip[:valid_end],
        spread_levels_bps=spreads,
        optimal_spread_bps=cont_optimal[:valid_end],
    )
    print_regime_report(result, spreads)


def cmd_backtest(args):
    """Run full stateful backtest."""
    df = load_data(args.data_dir, args.symbol)
    spreads = DEFAULT_SPREAD_LEVELS_BPS

    print(f"\nSimulating + training model...")
    sim = simulate_mm(df, spread_levels_bps=spreads, horizon=args.horizon)
    cont_optimal = compute_continuous_optimal(sim)
    ctx_df = extract_context(df)
    X = context_to_numpy(ctx_df)

    valid_end = len(df) - args.horizon

    # Train on first 60%, backtest on last 40%
    split = int(valid_end * 0.6)
    X_tr = X[:split]
    y_tr = cont_optimal[:split]

    result = train_eamm(X_tr, y_tr, CONTEXT_FEATURE_NAMES,
                        mode="regression", save_dir=None)
    predicted = predict_spread(result, X[split:valid_end])

    # Run backtest
    midprices = df["raw_midprice"].to_numpy()[split:valid_end]
    timestamps = df["timestamp_ns"].to_numpy()[split:valid_end]
    volatility = df["vol_returns_1m"].to_numpy()[split:valid_end]

    bt = run_backtest(
        midprices=midprices,
        timestamps=timestamps,
        predicted_spreads_bps=predicted,
        volatility=volatility,
        gamma=args.gamma,
        q_max=args.q_max,
        horizon=args.horizon,
    )
    print_backtest_report(bt)

    # Also report top features
    print(f"\n  Top 5 features:")
    for name, imp in list(result.feature_importances.items())[:5]:
        print(f"    {name:20s} {imp:6d}")


def main():
    parser = argparse.ArgumentParser(description="EAMM — Entropy-Adaptive Market Making")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--horizon", type=int, default=3000)

    subparsers = parser.add_subparsers(dest="command")

    # run
    p_run = subparsers.add_parser("run", help="Full pipeline")
    p_run.add_argument("--mode", default="regression", choices=["regression", "classification"])

    # regime
    p_regime = subparsers.add_parser("regime", help="Regime analysis only (thesis test)")

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Stateful backtest")
    p_bt.add_argument("--gamma", type=float, default=0.1, help="Risk aversion")
    p_bt.add_argument("--q-max", type=float, default=1.0, help="Max inventory")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "regime":
        cmd_regime(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
