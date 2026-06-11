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
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

# Add scripts to path for cluster_pipeline imports

from cluster_pipeline.loader import load_parquet
from utils.metrics import annualized_sharpe


def load_all_data(
    data_dir: str,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    max_memory_mb: float | None = None,
) -> pl.DataFrame:
    """Load and concatenate all parquet files for a symbol.

    Uses the hardened loader with date-range filtering and memory guard
    to avoid OOM when data_dir contains many days of data.
    """
    df_pd = load_parquet(
        data_dir,
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        max_memory_mb=max_memory_mb,
    )
    df = pl.from_pandas(df_pd)
    df = df.sort("timestamp_ns")
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

    test1_result = {
        "train_acc": float(train_acc),
        "top_features": [{"name": n, "importance": int(i)} for n, i in importances[:15]],
    }
    return model, test1_result


def test_2_walkforward(df: pl.DataFrame, feature_cols: list, n_splits: int = 5, horizon: int = 300):
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
        # Non-overlapping return periods per year (crypto 365d, 100ms ticks)
        periods_per_year = 365 * 24 * 3600 * 10 / horizon
        sharpe = annualized_sharpe(pnl, periods_per_year=periods_per_year)

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
    threshold_results = []

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

        threshold_results.append({
            "threshold": float(thresh),
            "trades": int(n_trades),
            "accuracy": float(acc),
            "gross_bps": float(avg_gross * 10000),
            "net_taker_bps": float(avg_net_taker * 10000),
            "net_maker_bps": float(avg_net_maker * 10000),
            "win_rate": float(win_rate),
        })

    # Summary
    print(f"\n  Trades/hour (approx): {trades_per_hour:.0f} at lowest threshold")
    print()
    print("  INTERPRETATION:")
    print("    - Look for rows where Net(taker) > 0 — that's real profit")
    print("    - If only Net(maker) > 0 — need to use limit orders")
    print("    - Higher threshold = fewer but better trades")
    print("    - Win rate > 55% with positive net = tradeable signal")

    best = max(threshold_results, key=lambda r: r["gross_bps"]) if threshold_results else {}
    return {
        "thresholds": threshold_results,
        "best_gross_bps": best.get("gross_bps", 0.0),
        "best_threshold": best.get("threshold", 0.0),
    }


# ---------------------------------------------------------------------------
# Regression mode: predict magnitude, not just direction
# ---------------------------------------------------------------------------


def reg_test_1_insample(X_train, y_train, X_test, y_test, feature_names):
    """Regression Test 1: In-sample fit (R², RMSE) — sanity check."""
    from sklearn.metrics import r2_score, mean_squared_error

    print("\n" + "=" * 60)
    print("REG TEST 1: In-Sample Fit (sanity check)")
    print("=" * 60)

    model = lgb.LGBMRegressor(
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

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Directional accuracy: does the sign of prediction match?
    dir_acc_train = np.mean(np.sign(y_pred_train) == np.sign(y_train))
    dir_acc_test = np.mean(np.sign(y_pred_test) == np.sign(y_test))

    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Test RMSE:  {test_rmse:.6f}")
    print(f"  Directional acc (train): {dir_acc_train:.4f}")
    print(f"  Directional acc (test):  {dir_acc_test:.4f}")

    # IC (information coefficient = correlation)
    ic_train = np.corrcoef(y_train, y_pred_train)[0, 1]
    ic_test = np.corrcoef(y_test, y_pred_test)[0, 1]
    print(f"  IC (train): {ic_train:.4f}")
    print(f"  IC (test):  {ic_test:.4f}")

    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print(f"\n  Top 15 features:")
    for name, imp in importances[:15]:
        print(f"    {name:40s} {imp:6d}")

    result = {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "dir_acc_train": float(dir_acc_train),
        "dir_acc_test": float(dir_acc_test),
        "ic_train": float(ic_train),
        "ic_test": float(ic_test),
        "top_features": [{"name": n, "importance": int(i)} for n, i in importances[:15]],
    }
    return model, result


def reg_test_2_walkforward(df: pl.DataFrame, feature_cols: list, n_splits: int = 5, horizon: int = 300):
    """
    Regression Test 2: Walk-forward R² and IC.
    Train on past, predict future returns (continuous).
    """
    from sklearn.metrics import r2_score

    print("\n" + "=" * 60)
    print(f"REG TEST 2: Walk-Forward Regression ({n_splits} splits)")
    print("=" * 60)

    n = len(df)
    min_train = int(n * 0.2)
    test_size = int(n * 0.8 / n_splits)

    X = df.select(feature_cols).to_numpy()
    y = df["forward_return"].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    results = []
    for i in range(n_splits):
        test_start = min_train + i * test_size
        test_end = min(test_start + test_size, n)
        if test_start >= n:
            break

        X_tr, y_tr = X[:test_start], y[:test_start]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        model = lgb.LGBMRegressor(
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

        y_pred = model.predict(X_te)
        r2 = r2_score(y_te, y_pred)
        rmse = np.sqrt(np.mean((y_te - y_pred) ** 2))
        ic = np.corrcoef(y_te, y_pred)[0, 1] if len(y_te) > 1 else 0.0

        # Directional accuracy from regression predictions
        dir_acc = np.mean(np.sign(y_pred) == np.sign(y_te))

        # PnL: position proportional to predicted return
        positions = y_pred  # continuous sizing
        pnl = positions * y_te
        # Non-overlapping return periods per year (crypto 365d, 100ms ticks)
        periods_per_year = 365 * 24 * 3600 * 10 / horizon
        sharpe = annualized_sharpe(pnl, periods_per_year=periods_per_year)

        results.append({
            "split": i + 1,
            "train_size": len(y_tr),
            "test_size": len(y_te),
            "r2": float(r2),
            "rmse": float(rmse),
            "ic": float(ic),
            "dir_acc": float(dir_acc),
            "sharpe": float(sharpe),
        })

        print(f"  Split {i+1}: R²={r2:.4f} | IC={ic:.4f} | "
              f"dir_acc={dir_acc:.4f} | sharpe={sharpe:.2f}")

    avg_r2 = np.mean([r["r2"] for r in results])
    avg_ic = np.mean([r["ic"] for r in results])
    avg_dir = np.mean([r["dir_acc"] for r in results])
    avg_sharpe = np.mean([r["sharpe"] for r in results])

    print(f"\n  AVERAGE: R²={avg_r2:.4f} | IC={avg_ic:.4f} | "
          f"dir_acc={avg_dir:.4f} | sharpe={avg_sharpe:.2f}")

    if avg_ic > 0.02:
        print(f"  RESULT: Predictive signal detected (IC={avg_ic:.4f}).")
    elif avg_ic > 0.005:
        print(f"  RESULT: Weak signal (IC={avg_ic:.4f}) — may not survive costs.")
    else:
        print(f"  RESULT: No regression signal detected (IC={avg_ic:.4f}).")

    return results


def reg_test_3_quantile_pnl(df: pl.DataFrame, feature_cols: list, spread_bps: float, horizon: int):
    """
    Regression Test 3: Quantile-based PnL.

    Instead of confidence thresholds, sort predictions into quantiles.
    Go long the top quantile, short the bottom quantile, flat in the middle.
    This tests whether the regression model ranks returns correctly.
    """
    print("\n" + "=" * 60)
    print("REG TEST 3: Quantile-Based Trading (the real test)")
    print("=" * 60)

    taker_fee_bps = 3.5
    half_spread_bps = spread_bps / 2
    round_trip_cost = 2 * (taker_fee_bps + half_spread_bps) / 10000
    maker_round_trip = 2 * half_spread_bps / 10000

    print(f"  Round-trip cost (taker): {round_trip_cost*10000:.1f} bps")
    print(f"  Round-trip cost (maker): {maker_round_trip*10000:.1f} bps")
    print()

    n = len(df)
    split = int(n * 0.6)

    X = df.select(feature_cols).to_numpy()
    y = df["forward_return"].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    model = lgb.LGBMRegressor(
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
    y_pred = model.predict(X_te)

    quantile_configs = [
        (0.10, "Top/Bottom 10%"),
        (0.20, "Top/Bottom 20%"),
        (0.30, "Top/Bottom 30%"),
        (0.40, "Top/Bottom 40%"),
        (0.50, "All (sign-based)"),
    ]

    quantile_results = []

    print(f"  {'Quantile':>20} | {'Trades':>7} | {'Dir Acc':>8} | {'Avg Ret':>10} | "
          f"{'Net(taker)':>10} | {'Net(maker)':>10} | {'IC':>6}")
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")

    for frac, label in quantile_configs:
        if frac >= 0.5:
            # Sign-based: trade everything
            trade_mask = np.ones(len(y_pred), dtype=bool)
            positions = np.sign(y_pred)
        else:
            top_thresh = np.quantile(y_pred, 1 - frac)
            bot_thresh = np.quantile(y_pred, frac)
            long_mask = y_pred >= top_thresh
            short_mask = y_pred <= bot_thresh
            trade_mask = long_mask | short_mask
            positions = np.where(long_mask, 1.0, np.where(short_mask, -1.0, 0.0))
            positions = positions[trade_mask]

        n_trades = trade_mask.sum()
        trade_returns = y_te[trade_mask]
        trade_preds = y_pred[trade_mask]

        gross_pnl = positions * trade_returns
        avg_gross = gross_pnl.mean()
        avg_net_taker = avg_gross - round_trip_cost
        avg_net_maker = avg_gross - maker_round_trip

        # Direction accuracy
        dir_acc = np.mean(np.sign(positions) == np.sign(trade_returns))

        # IC on the traded subset
        ic = np.corrcoef(trade_preds, trade_returns)[0, 1] if n_trades > 10 else 0.0

        print(f"  {label:>20} | {n_trades:>7d} | {dir_acc:>8.4f} | "
              f"{avg_gross*10000:>+9.2f}bp | "
              f"{avg_net_taker*10000:>+9.2f}bp | "
              f"{avg_net_maker*10000:>+9.2f}bp | "
              f"{ic:>+.4f}")

        quantile_results.append({
            "quantile_frac": float(frac),
            "label": label,
            "trades": int(n_trades),
            "dir_acc": float(dir_acc),
            "gross_bps": float(avg_gross * 10000),
            "net_taker_bps": float(avg_net_taker * 10000),
            "net_maker_bps": float(avg_net_maker * 10000),
            "ic": float(ic),
        })

    print()
    print("  INTERPRETATION:")
    print("    - Top quantiles should have higher returns than bottom")
    print("    - Monotonic IC across quantiles = strong ranking ability")
    print("    - Positive net at narrow quantile = tradeable regression signal")

    best = max(quantile_results, key=lambda r: r["gross_bps"]) if quantile_results else {}
    return {
        "quantiles": quantile_results,
        "best_gross_bps": best.get("gross_bps", 0.0),
        "best_quantile": best.get("quantile_frac", 0.0),
    }


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
    parser.add_argument("--start-date", default=None,
                        help="Start date for directory filtering (e.g. 2026-05-10)")
    parser.add_argument("--end-date", default=None,
                        help="End date for directory filtering (e.g. 2026-05-15)")
    parser.add_argument("--max-memory-mb", type=float, default=2000.0,
                        help="Max memory in MB for data loading (default: 2000)")
    parser.add_argument("--json-report", default=None,
                        help="Write structured JSON report to this path")
    parser.add_argument("--mode", choices=["classify", "regress"], default="classify",
                        help="Test mode: classify (binary direction) or regress (return magnitude)")
    args = parser.parse_args()

    mode_label = "Classification" if args.mode == "classify" else "Regression"
    print(f"Phase 1: Signal Test for {args.symbol} ({mode_label})")
    print(f"  Horizon: {args.horizon} rows ({args.horizon * 0.1:.0f}s)")
    print(f"  Spread assumption: {args.spread_bps:.1f} bps")
    if args.start_date or args.end_date:
        print(f"  Date range: [{args.start_date or '...'}, {args.end_date or '...'}]")
    print()

    # Load data
    df = load_all_data(
        args.data_dir, args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        max_memory_mb=args.max_memory_mb,
    )
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
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    split = int(len(X) * 0.7)

    if args.mode == "classify":
        y = df["target"].to_numpy()
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model, test1_result = test_1_insample(X_train, y_train, X_test, y_test, feature_cols)
        wf_results = test_2_walkforward(df, feature_cols, horizon=args.horizon)
        test3_result = test_3_confidence_filtered(df, feature_cols, args.spread_bps, args.horizon)

        if args.json_report:
            import json
            avg_acc = np.mean([r["accuracy"] for r in wf_results])
            avg_base = np.mean([r["base_rate"] for r in wf_results])
            avg_sharpe = np.mean([r["sharpe"] for r in wf_results])
            report = {
                "mode": "classify",
                "symbol": args.symbol,
                "horizon": args.horizon,
                "spread_bps": args.spread_bps,
                "n_rows": int(len(df)),
                "n_features": len(feature_cols),
                "target_up_pct": float(up_pct),
                "mean_forward_return_bps": float(df["forward_return"].mean() * 10000),
                "test1_train_acc": test1_result["train_acc"],
                "test1_top_features": test1_result["top_features"],
                "test2_avg_accuracy": float(avg_acc),
                "test2_avg_base_rate": float(avg_base),
                "test2_avg_edge": float(avg_acc - avg_base),
                "test2_avg_sharpe": float(avg_sharpe),
                "test2_splits": wf_results,
                "test3_thresholds": test3_result["thresholds"],
                "test3_best_gross_bps": test3_result["best_gross_bps"],
                "test3_best_threshold": test3_result["best_threshold"],
            }
            Path(args.json_report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.json_report, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nJSON report written to {args.json_report}")

    else:  # regress
        y = df["forward_return"].to_numpy()
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model, test1_result = reg_test_1_insample(X_train, y_train, X_test, y_test, feature_cols)
        wf_results = reg_test_2_walkforward(df, feature_cols, horizon=args.horizon)
        test3_result = reg_test_3_quantile_pnl(df, feature_cols, args.spread_bps, args.horizon)

        if args.json_report:
            import json
            avg_r2 = np.mean([r["r2"] for r in wf_results])
            avg_ic = np.mean([r["ic"] for r in wf_results])
            avg_dir = np.mean([r["dir_acc"] for r in wf_results])
            avg_sharpe = np.mean([r["sharpe"] for r in wf_results])
            report = {
                "mode": "regress",
                "symbol": args.symbol,
                "horizon": args.horizon,
                "spread_bps": args.spread_bps,
                "n_rows": int(len(df)),
                "n_features": len(feature_cols),
                "target_up_pct": float(up_pct),
                "mean_forward_return_bps": float(df["forward_return"].mean() * 10000),
                "reg_test1": test1_result,
                "reg_test2_avg_r2": float(avg_r2),
                "reg_test2_avg_ic": float(avg_ic),
                "reg_test2_avg_dir_acc": float(avg_dir),
                "reg_test2_avg_sharpe": float(avg_sharpe),
                "reg_test2_splits": wf_results,
                "reg_test3_quantiles": test3_result["quantiles"],
                "reg_test3_best_gross_bps": test3_result["best_gross_bps"],
                "reg_test3_best_quantile": test3_result["best_quantile"],
            }
            Path(args.json_report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.json_report, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nJSON report written to {args.json_report}")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    if args.mode == "classify":
        print("  If signal exists:")
        print("    1. Try regression mode: --mode regress")
        print("    2. Try different horizons: --horizon 100 (10s), 600 (60s), 3000 (5m)")
        print("    3. Try other symbols: --symbol ETH, --symbol SOL")
        print("    4. If profitable after costs -> paper trade for 1 week")
    else:
        print("  If regression signal exists:")
        print("    1. Compare with classification: --mode classify")
        print("    2. Top-quantile positive net = tradeable with proportional sizing")
        print("    3. IC > 0.02 across walk-forward = robust signal")
        print("    4. Use regression predictions for continuous position sizing")
    print()


if __name__ == "__main__":
    main()
