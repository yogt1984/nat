#!/usr/bin/env python3
"""
Phase 1: Does signal exist in the feature vector?

The simplest possible test: can a LightGBM model predict the SIGN of
the next-30s return better than 50%? If not, nothing else matters.

Three tests, increasing difficulty:
  1. In-sample accuracy (sanity check — should be high, means nothing alone)
  2. Walk-forward accuracy (the real test — train on past, predict future)
  3. After transaction costs (the final test — is signal > costs?)

Usage:
    python scripts/phase1_signal_test.py
    python scripts/phase1_signal_test.py --symbol BTC --horizon 300
    python scripts/phase1_signal_test.py --symbol BTC --horizon 300 --spread-bps 1.0
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report


def load_all_data(data_dir: str, symbol: str) -> pl.DataFrame:
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
        print(f"ERROR: no data found for {symbol}")
        sys.exit(1)
    df = pl.concat(dfs).sort("timestamp_ns")
    # deduplicate by timestamp
    df = df.unique(subset=["timestamp_ns"], keep="first")
    print(f"Loaded {df.shape[0]:,} rows for {symbol}")
    return df


def create_target(df: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """
    Create binary target: will midprice be higher in `horizon` rows?

    horizon=300 at 100ms emission = 30 seconds forward.
    Uses a dead zone around 0 to avoid noise: returns within ±0.5 bps are dropped.
    """
    future_price = df["raw_midprice"].shift(-horizon)
    current_price = df["raw_midprice"]
    ret = (future_price - current_price) / current_price
    df = df.with_columns([
        ret.alias("forward_return"),
        (ret > 0).cast(pl.Int32).alias("target"),
    ])
    # drop rows where target is null (last `horizon` rows)
    df = df.drop_nulls(subset=["target"])
    # drop tiny returns (dead zone) — these are noise, not signal
    dead_zone = 0.00005  # 0.5 bps
    df = df.filter(pl.col("forward_return").abs() > dead_zone)
    return df


def get_feature_columns(df: pl.DataFrame, remove_leaky: bool = False) -> list:
    """Get only the base feature columns (no metadata, no target).

    If remove_leaky=True, also removes absolute-value features that the model
    can memorize (price level, OI level, 24h volume) — these change over time
    and don't generalize across regimes.
    """
    exclude = {
        "timestamp_ns", "symbol", "sequence_id",
        "forward_return", "target",
    }
    # Absolute-value features that leak regime identity
    leaky = {
        "raw_midprice",
        "ctx_open_interest",
        "ctx_volume_24h",
        "raw_bid_depth_5", "raw_ask_depth_5",
        "raw_bid_depth_10", "raw_ask_depth_10",
        "raw_bid_orders_5", "raw_ask_orders_5",
    }
    if remove_leaky:
        exclude |= leaky
    # only numeric columns
    features = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            features.append(col)
    return features


def test_1_insample(X_train, y_train, X_test, y_test, feature_names):
    """Test 1: In-sample fit (sanity check)."""
    print("\n" + "=" * 60)
    print("TEST 1: In-Sample Accuracy (sanity check)")
    print("=" * 60)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  (Should be >0.55 to show model can learn patterns)")
    if train_acc < 0.52:
        print("  WARNING: Model can barely fit training data. Features may lack signal.")

    # feature importance (top 15)
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"\n  Top 15 features by importance:")
    for name, imp in importances[:15]:
        print(f"    {name:40s} {imp:6d}")

    return model


def test_2_walkforward(df: pl.DataFrame, feature_cols: list, n_splits: int = 5):
    """
    Test 2: Walk-forward validation.
    Train on past, predict future. No lookahead.
    """
    print("\n" + "=" * 60)
    print(f"TEST 2: Walk-Forward Validation ({n_splits} splits)")
    print("=" * 60)

    n = len(df)
    # minimum 20% for first training set
    min_train = int(n * 0.2)
    test_size = int(n * 0.8 / n_splits)

    X = df.select(feature_cols).to_numpy()
    y = df["target"].to_numpy()
    returns = df["forward_return"].to_numpy()

    # Replace NaN/inf with 0 for LightGBM
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    results = []
    for i in range(n_splits):
        test_start = min_train + i * test_size
        test_end = min(test_start + test_size, n)
        if test_start >= n:
            break

        X_tr, y_tr = X[:test_start], y[:test_start]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]
        ret_te = returns[test_start:test_end]

        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, preds)

        # simple PnL: go long if predict up, short if predict down
        positions = np.where(preds == 1, 1.0, -1.0)
        pnl = positions * ret_te
        sharpe = np.mean(pnl) / (np.std(pnl) + 1e-10) * np.sqrt(len(pnl))

        results.append({
            "split": i + 1,
            "train_size": len(y_tr),
            "test_size": len(y_te),
            "accuracy": acc,
            "mean_return": np.mean(pnl),
            "sharpe": sharpe,
            "base_rate": np.mean(y_te),
        })

        print(f"  Split {i+1}: acc={acc:.4f} | base_rate={np.mean(y_te):.4f} | "
              f"edge={acc - np.mean(y_te):+.4f} | sharpe={sharpe:.2f}")

    avg_acc = np.mean([r["accuracy"] for r in results])
    avg_base = np.mean([r["base_rate"] for r in results])
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    print(f"\n  AVERAGE: acc={avg_acc:.4f} | base_rate={avg_base:.4f} | "
          f"edge={avg_acc - avg_base:+.4f} | sharpe={avg_sharpe:.2f}")

    if avg_acc > avg_base + 0.01:
        print(f"  RESULT: Signal detected. Edge of {(avg_acc - avg_base)*100:.2f}% over base rate.")
    elif avg_acc > avg_base:
        print(f"  RESULT: Marginal signal. Edge of {(avg_acc - avg_base)*100:.2f}% — may not survive costs.")
    else:
        print(f"  RESULT: No signal detected out of sample.")

    return results


def test_3_confidence_filtered(df: pl.DataFrame, feature_cols: list, spread_bps: float, horizon: int):
    """
    Test 3: Trade only on high-confidence predictions.

    This is the real test. Instead of trading every bar:
    - Train on first 60%, test on last 40%
    - Only trade when model probability > threshold or < (1-threshold)
    - Hold for exactly `horizon` bars (one entry, one exit)
    - Pay round-trip cost once per trade
    """
    print("\n" + "=" * 60)
    print("TEST 3: Confidence-Filtered Trading (the real test)")
    print("=" * 60)

    taker_fee_bps = 3.5
    half_spread_bps = spread_bps / 2
    round_trip_cost = 2 * (taker_fee_bps + half_spread_bps) / 10000

    # Also test maker orders (0 fee on Hyperliquid)
    maker_round_trip = 2 * half_spread_bps / 10000

    print(f"  Round-trip cost (taker): {round_trip_cost*10000:.1f} bps")
    print(f"  Round-trip cost (maker): {maker_round_trip*10000:.1f} bps")
    print(f"  Horizon: {horizon} bars ({horizon * 0.1:.0f}s)")
    print()

    n = len(df)
    split = int(n * 0.6)

    X = df.select(feature_cols).to_numpy()
    y = df["target"].to_numpy()
    returns = df["forward_return"].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]
    ret_te = returns[split:]

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    print(f"  {'Threshold':>10} | {'Trades':>7} | {'Accuracy':>8} | {'Avg Ret':>10} | "
          f"{'Net(taker)':>10} | {'Net(maker)':>10} | {'Win Rate':>8}")
    print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for thresh in thresholds:
        # long when prob > thresh, short when prob < (1-thresh)
        long_mask = probs > thresh
        short_mask = probs < (1 - thresh)
        trade_mask = long_mask | short_mask

        n_trades = trade_mask.sum()
        if n_trades < 10:
            print(f"  {thresh:>10.2f} | {n_trades:>7d} | {'(too few)':>8}")
            continue

        # direction: +1 for long, -1 for short
        directions = np.where(long_mask[trade_mask], 1.0, -1.0)
        trade_returns = ret_te[trade_mask]
        trade_actuals = y_te[trade_mask]
        trade_probs = probs[trade_mask]

        # gross PnL per trade
        gross_pnl = directions * trade_returns
        # accuracy
        trade_preds = np.where(directions > 0, 1, 0)
        acc = accuracy_score(trade_actuals, trade_preds)
        # win rate
        win_rate = (gross_pnl > 0).mean()

        avg_gross = gross_pnl.mean()
        avg_net_taker = avg_gross - round_trip_cost
        avg_net_maker = avg_gross - maker_round_trip

        # trades per hour (assuming 100ms bars, horizon = hold time)
        hours_in_test = len(ret_te) * 0.1 / 3600
        trades_per_hour = n_trades / hours_in_test

        taker_tag = "+" if avg_net_taker > 0 else ""
        maker_tag = "+" if avg_net_maker > 0 else ""

        print(f"  {thresh:>10.2f} | {n_trades:>7d} | {acc:>8.4f} | "
              f"{avg_gross*10000:>+9.2f}bp | "
              f"{avg_net_taker*10000:>+9.2f}bp | "
              f"{avg_net_maker*10000:>+9.2f}bp | "
              f"{win_rate:>8.1%}")

    # Summary
    print(f"\n  Trades/hour (approx): {trades_per_hour:.0f} at lowest threshold")
    print()
    print("  INTERPRETATION:")
    print("    - Look for rows where Net(taker) > 0 — that's real profit")
    print("    - If only Net(maker) > 0 — need to use limit orders")
    print("    - Higher threshold = fewer but better trades")
    print("    - Win rate > 55% with positive net = tradeable signal")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Signal existence test")
    parser.add_argument("--symbol", default="BTC", help="Symbol to test (default: BTC)")
    parser.add_argument("--horizon", type=int, default=300,
                        help="Prediction horizon in rows (300 = 30s at 100ms)")
    parser.add_argument("--spread-bps", type=float, default=1.0,
                        help="Assumed spread in basis points (default: 1.0)")
    parser.add_argument("--data-dir", default="data/features",
                        help="Data directory (default: data/features)")
    parser.add_argument("--remove-leaky", action="store_true",
                        help="Remove leaky absolute-value features (midprice, OI, volume_24h)")
    args = parser.parse_args()

    print(f"Phase 1: Signal Test for {args.symbol}")
    print(f"  Horizon: {args.horizon} rows ({args.horizon * 0.1:.0f}s)")
    print(f"  Spread assumption: {args.spread_bps:.1f} bps")
    print()

    # Load data
    df = load_all_data(args.data_dir, args.symbol)
    feature_cols = get_feature_columns(df, remove_leaky=args.remove_leaky)
    if args.remove_leaky:
        print(f"Feature columns: {len(feature_cols)} (leaky features REMOVED)")
    else:
        print(f"Feature columns: {len(feature_cols)}")

    # Create target
    df = create_target(df, args.horizon)
    up_pct = df["target"].mean() * 100
    print(f"Target distribution: {up_pct:.1f}% up / {100-up_pct:.1f}% down")
    print(f"Mean forward return: {df['forward_return'].mean()*10000:.2f} bps")

    # Prepare data
    X = df.select(feature_cols).to_numpy()
    y = df["target"].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Run tests
    model = test_1_insample(X_train, y_train, X_test, y_test, feature_cols)
    results = test_2_walkforward(df, feature_cols)
    test_3_confidence_filtered(df, feature_cols, args.spread_bps, args.horizon)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("  If signal exists:")
    print("    1. Try different horizons: --horizon 100 (10s), 600 (60s), 3000 (5m)")
    print("    2. Try other symbols: --symbol ETH, --symbol SOL")
    print("    3. Check if signal is stable across all walk-forward splits")
    print("    4. If profitable after costs → paper trade for 1 week")
    print()
    print("  If no signal:")
    print("    1. Longer horizon (3000 = 5min, 18000 = 30min)")
    print("    2. More data (keep ingestor running for 2+ weeks)")
    print("    3. Feature engineering (lag features, rolling stats)")
    print()


if __name__ == "__main__":
    main()
