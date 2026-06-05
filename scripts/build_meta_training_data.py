#!/usr/bin/env python3
"""
Build meta-labeling training data.

Runs the 5 winner base algorithms on tick data, aggregates to 5-min bars,
computes triple-barrier labels (De Prado method), and outputs a training
dataset for the MetaLabeling classifier.

Usage:
    python scripts/build_meta_training_data.py --symbol BTC --data-dir data/features
    python scripts/build_meta_training_data.py --symbol BTC --data-dir data/features --output data/meta_training/

See docs/research/new/ml_algorithms.txt Section 3 for full specification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_pipeline.preprocess import aggregate_bars
from cluster_pipeline.loader import load_parquet
from algorithms.runner import AlgorithmRunner
from algorithms.registry import get_algorithm

# Base algorithms whose signals trigger meta-labeling
BASE_ALGOS = [
    "jump_detector",
    "3f_liquidity",
    "optimal_entry",
    "funding_reversion",
    "surprise_signal",
]

# Primary signal column for each base algorithm
BASE_SIGNAL_COLS = {
    "jump_detector": "alg_jd_signal",
    "3f_liquidity": "alg_3f_signal",
    "optimal_entry": "alg_oe_signal",
    "funding_reversion": "alg_fr_signal",
    "surprise_signal": "alg_ss_signal",
}

# Meta-labeling state features (non-directional market condition)
META_FEATURE_COLS = [
    "ent_tick_1m_mean",
    "ent_rate_of_change_5s_mean",
    "toxic_vpin_10_mean",
    "toxic_index_mean",
    "conc_hhi_last",
    "whale_directional_agreement_last",
    "vol_returns_5m_mean",
    "vol_ratio_short_long_last",
    "regime_clarity_last",
    "raw_spread_bps_mean",
]


def compute_triple_barrier_labels(
    prices: np.ndarray,
    profit_target_bps: float = 5.0,
    stop_loss_bps: float = 10.0,
    max_holding_bars: int = 100,
) -> np.ndarray:
    """Compute triple-barrier labels (De Prado Ch. 3).

    For each bar t:
      - Upper barrier: price[t] * (1 + profit_target_bps/10000)
      - Lower barrier: price[t] * (1 - stop_loss_bps/10000)
      - Time barrier: t + max_holding_bars
      - Label: 1 if upper hit first, 0 if lower hit first
      - If neither hit by time barrier: 1 if exit_price > entry, else 0

    Returns array of labels (NaN for last max_holding_bars rows).
    """
    n = len(prices)
    labels = np.full(n, np.nan)

    for t in range(n - 1):
        entry = prices[t]
        if not np.isfinite(entry) or entry <= 0:
            continue

        upper = entry * (1 + profit_target_bps / 10000)
        lower = entry * (1 - stop_loss_bps / 10000)
        end = min(t + max_holding_bars, n - 1)

        label = np.nan
        for j in range(t + 1, end + 1):
            p = prices[j]
            if not np.isfinite(p):
                continue
            if p >= upper:
                label = 1.0
                break
            if p <= lower:
                label = 0.0
                break
        else:
            # Time barrier hit — label by sign of return
            exit_price = prices[end]
            if np.isfinite(exit_price):
                label = 1.0 if exit_price > entry else 0.0

        labels[t] = label

    return labels


def build_meta_training_data(
    data_dir: str,
    symbol: str,
    base_algos: list[str] | None = None,
    profit_target_bps: float = 5.0,
    stop_loss_bps: float = 10.0,
    max_holding_bars: int = 100,
    signal_threshold: float = 0.01,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build meta-labeling training data.

    Returns:
        (bars_df, labels, meta_features) where:
        - bars_df: 5-min bars with all columns
        - labels: triple-barrier binary labels
        - meta_features: feature matrix for meta-labeling model
    """
    if base_algos is None:
        base_algos = BASE_ALGOS

    # Load raw ticks
    df = load_parquet(data_dir, symbols=[symbol], max_memory_mb=4000)
    print(f"Loaded {len(df):,} ticks for {symbol}")

    if len(df) < 1000:
        print(f"ERROR: Only {len(df)} ticks, need at least 1000")
        sys.exit(1)

    # Run base algorithms and merge outputs
    for algo_name in base_algos:
        try:
            algo = get_algorithm(algo_name)
            runner = AlgorithmRunner(algo)
            result = runner.run_on_dataframe(df)
            for col in result.features_df.columns:
                df[col] = result.features_df[col].values
            print(f"  Ran {algo_name}: {len(result.features_df.columns)} features")
        except Exception as e:
            print(f"  WARNING: Could not run {algo_name}: {e}")

    # Aggregate to 5-min bars
    bars = aggregate_bars(df, timeframe="5min")
    print(f"Aggregated to {len(bars):,} bars")

    # Filter bars where at least one base signal > threshold
    signal_cols = []
    for algo_name in base_algos:
        sig_col = BASE_SIGNAL_COLS.get(algo_name)
        if sig_col:
            # Check both _mean and _last suffixes
            for suffix in ["_mean", "_last"]:
                candidate = f"{sig_col}{suffix}"
                if candidate in bars.columns:
                    signal_cols.append(candidate)
                    break

    if signal_cols:
        has_signal = bars[signal_cols].abs().max(axis=1) > signal_threshold
        n_before = len(bars)
        bars = bars[has_signal].copy()
        print(f"Filtered to {len(bars):,} bars with active signals (from {n_before:,})")

    # Compute triple-barrier labels
    if "raw_midprice_mean" in bars.columns:
        prices = bars["raw_midprice_mean"].values
    else:
        print("ERROR: raw_midprice_mean not found in bars")
        sys.exit(1)

    labels = compute_triple_barrier_labels(
        prices, profit_target_bps, stop_loss_bps, max_holding_bars
    )

    # Build meta-feature matrix
    available_meta = [c for c in META_FEATURE_COLS if c in bars.columns]
    missing_meta = [c for c in META_FEATURE_COLS if c not in bars.columns]
    if missing_meta:
        print(f"WARNING: Missing meta features: {missing_meta}")

    meta_features = bars[available_meta].values if available_meta else np.zeros((len(bars), 0))

    # Drop rows with NaN labels or features
    valid = np.isfinite(labels)
    if available_meta:
        valid &= np.all(np.isfinite(meta_features), axis=1)

    bars_valid = bars[valid].copy()
    labels_valid = labels[valid]
    meta_valid = meta_features[valid]

    print(f"Valid samples: {len(labels_valid):,}")
    if len(labels_valid) > 0:
        print(f"Label balance: {labels_valid.mean():.3f} (win rate)")

    return bars_valid, labels_valid, meta_valid


def main():
    parser = argparse.ArgumentParser(description="Build meta-labeling training data")
    parser.add_argument("--symbol", default="BTC", help="Symbol")
    parser.add_argument("--data-dir", default="data/features", help="Parquet data directory")
    parser.add_argument("--output", default="data/meta_training", help="Output directory")
    parser.add_argument("--profit-target-bps", type=float, default=5.0)
    parser.add_argument("--stop-loss-bps", type=float, default=10.0)
    parser.add_argument("--max-holding-bars", type=int, default=100)
    args = parser.parse_args()

    print(f"=== Building Meta-Labeling Training Data: {args.symbol} ===")

    bars, labels, meta_features = build_meta_training_data(
        args.data_dir, args.symbol,
        profit_target_bps=args.profit_target_bps,
        stop_loss_bps=args.stop_loss_bps,
        max_holding_bars=args.max_holding_bars,
    )

    # Save output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bars.to_parquet(output_dir / f"{args.symbol}_bars.parquet")
    np.save(output_dir / f"{args.symbol}_labels.npy", labels)
    np.save(output_dir / f"{args.symbol}_meta_features.npy", meta_features)

    print(f"Saved to {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
