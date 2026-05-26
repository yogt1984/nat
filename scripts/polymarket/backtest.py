"""
Polymarket Probability Model Backtest

Before going live, validate the model on historical data:
  1. Does our conditional probability model outperform naive (unconditional)?
  2. What is the Brier score improvement from feature conditioning?
  3. What would the P&L be if we traded every divergence > threshold?

Uses existing NAT data — no Polymarket connection needed.

Usage:
    cd scripts && python -m polymarket.backtest
    cd scripts && python -m polymarket.backtest --horizon 5 --strike-offset 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cluster_pipeline.loader import load_parquet
from polymarket.probability_model import ProbabilityModel, EmpiricalModel
from polymarket.edge_detector import EdgeDetector, polymarket_fee


def load_btc_data(data_dir: str, max_days: int | None = None) -> pd.DataFrame:
    """Load BTC data at 1-second resolution."""
    cols = [
        "timestamp_ns", "symbol", "raw_midprice",
        "vol_parkinson_5m", "vol_returns_1m", "vol_zscore",
        "trend_momentum_300", "trend_hurst_300",
        "ent_book_shape",
        "ctx_funding_rate", "ctx_funding_zscore", "ctx_premium_bps",
    ]

    kwargs = {"symbols": ["BTC"]}
    if max_days:
        dates = sorted(Path(data_dir).iterdir())
        dates = [d for d in dates if d.is_dir() and not d.name.endswith("-clean")]
        if len(dates) > max_days:
            kwargs["start_date"] = dates[-max_days].name

    df = load_parquet(data_dir, columns=cols, **kwargs)
    print(f"Loaded {len(df):,} BTC rows")

    df["ts_s"] = df["timestamp_ns"] // 1_000_000_000
    df = df.groupby("ts_s").last().reset_index()
    df = df.set_index("ts_s").sort_index()
    print(f"After 1s bucketing: {len(df):,} rows")
    return df


def simulate_polymarket_trading(
    df: pd.DataFrame,
    horizon_minutes: int = 5,
    strike_offset_pct: float = 0.5,
    edge_threshold_cents: float = 2.0,
    train_fraction: float = 0.5,
):
    """
    Walk-forward backtest of Polymarket probability trading.

    At each 5-minute interval:
      1. Take current BTC price S
      2. Create synthetic market: "Will BTC > S × (1 + offset%) in H minutes?"
      3. Set market_price = naive probability (unconditional model)
      4. Compute model probability using features
      5. If divergence > threshold, simulate trade
      6. Check outcome at horizon and compute P&L
    """
    horizon_rows = horizon_minutes * 60

    n = len(df)
    train_end = int(n * train_fraction)

    # Train model on first half
    print(f"\n--- Training on first {train_fraction:.0%} ({train_end:,} rows) ---")
    train_df = df.iloc[:train_end]

    model = ProbabilityModel()
    model.fit(train_df, horizon_minutes=horizon_minutes)

    emp_model = EmpiricalModel()
    try:
        emp_model.fit(train_df, horizon_minutes=horizon_minutes)
        has_empirical = True
    except Exception as e:
        print(f"  Empirical model failed: {e}")
        has_empirical = False

    # Test on second half
    test_df = df.iloc[train_end:]
    print(f"\n--- Testing on remaining {1-train_fraction:.0%} ({len(test_df):,} rows) ---")

    # Sample every 5 minutes for decisions
    step = horizon_minutes * 60
    mid = test_df["raw_midprice"].values

    results = []
    n_test = len(test_df)

    for i in range(0, n_test - horizon_rows, step):
        S = mid[i]
        S_future = mid[i + horizon_rows]

        if not (np.isfinite(S) and np.isfinite(S_future)):
            continue

        # Strike: current price + offset
        K = S * (1 + strike_offset_pct / 100)
        actual_outcome = 1.0 if S_future > K else 0.0

        # Current features
        row = test_df.iloc[i]
        features = {col: float(row[col]) for col in test_df.columns
                    if col != "symbol" and isinstance(row[col], (int, float, np.floating))}

        # Naive probability (unconditional — what a simple market would price)
        naive_vol = model._base_vol
        naive_drift = model._base_drift
        log_m = np.log(S / K)
        d_naive = (log_m + naive_drift) / (naive_vol + 1e-12)
        naive_prob = float(sp_stats.norm.cdf(d_naive))

        # Model probability (conditional)
        model_prob, confidence = model.predict_with_confidence(
            features, S, K, horizon_minutes
        )

        # Empirical probability
        emp_prob = None
        if has_empirical:
            emp_prob = emp_model.predict(features, S, K, horizon_minutes)

        # Simulate trade decision
        edge = model_prob - naive_prob
        trade_side = None
        pnl = 0.0

        if abs(edge) * 100 > edge_threshold_cents:
            if edge > 0:
                # Model says YES more likely → BUY YES at naive_prob
                entry_price = naive_prob
                fee = polymarket_fee(entry_price)
                pnl = (actual_outcome - entry_price) - fee
                trade_side = "BUY_YES"
            else:
                # Model says NO more likely → BUY NO at (1 - naive_prob)
                entry_price = 1 - naive_prob
                fee = polymarket_fee(entry_price)
                pnl = ((1 - actual_outcome) - entry_price) - fee
                trade_side = "BUY_NO"

        results.append({
            "i": i,
            "S": S,
            "K": K,
            "S_future": S_future,
            "actual": actual_outcome,
            "naive_prob": naive_prob,
            "model_prob": model_prob,
            "emp_prob": emp_prob,
            "confidence": confidence,
            "edge": edge,
            "trade_side": trade_side,
            "pnl": pnl,
        })

    return pd.DataFrame(results)


def analyze_results(results: pd.DataFrame, horizon_minutes: int, strike_offset: float):
    """Print backtest analysis."""
    n = len(results)
    print(f"\n{'='*70}")
    print(f"POLYMARKET BACKTEST RESULTS")
    print(f"Horizon: {horizon_minutes}min | Strike offset: {strike_offset}% above current")
    print(f"{'='*70}")

    # Brier scores
    actual = results["actual"].values
    naive = results["naive_prob"].values
    model = results["model_prob"].values

    brier_naive = np.mean((naive - actual) ** 2)
    brier_model = np.mean((model - actual) ** 2)
    improvement = (brier_naive - brier_model) / brier_naive * 100

    print(f"\n  Probability Calibration ({n} decisions):")
    print(f"    Naive Brier score:  {brier_naive:.4f}")
    print(f"    Model Brier score:  {brier_model:.4f}")
    print(f"    Improvement:        {improvement:+.1f}%")

    # Empirical model
    emp = results["emp_prob"].dropna()
    if len(emp) > 100:
        mask_emp = results["emp_prob"].notna()
        brier_emp = np.mean((results.loc[mask_emp, "emp_prob"].values -
                             results.loc[mask_emp, "actual"].values) ** 2)
        print(f"    Empirical Brier:    {brier_emp:.4f}")

    # Calibration by probability bin
    print(f"\n  Calibration Table:")
    print(f"    {'Model P':>8} {'Actual %':>9} {'Count':>6} {'Error':>7}")
    print(f"    {'-'*35}")
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (model >= lo) & (model < hi)
        if mask.sum() > 0:
            actual_rate = actual[mask].mean()
            midpoint = (lo + hi) / 2
            error = actual_rate - midpoint
            print(f"    {lo:.1f}-{hi:.1f}   {actual_rate:>8.1%}  {mask.sum():>5}  {error:>+6.1%}")

    # Trading P&L
    trades = results[results["trade_side"].notna()]
    n_trades = len(trades)

    print(f"\n  Trading Results ({n_trades} trades out of {n} opportunities):")
    if n_trades > 0:
        total_pnl = trades["pnl"].sum()
        mean_pnl = trades["pnl"].mean()
        win_rate = (trades["pnl"] > 0).mean()
        sharpe = trades["pnl"].mean() / (trades["pnl"].std() + 1e-12)

        # By side
        yes_trades = trades[trades["trade_side"] == "BUY_YES"]
        no_trades = trades[trades["trade_side"] == "BUY_NO"]

        print(f"    Total P&L:     {total_pnl:+.2f} per $1 notional")
        print(f"    Mean P&L:      {mean_pnl:+.4f} per trade")
        print(f"    Win rate:      {win_rate:.1%}")
        print(f"    Sharpe (per-trade): {sharpe:.3f}")
        print(f"    BUY YES:       {len(yes_trades)} trades, "
              f"mean={yes_trades['pnl'].mean():+.4f}" if len(yes_trades) > 0 else "")
        print(f"    BUY NO:        {len(no_trades)} trades, "
              f"mean={no_trades['pnl'].mean():+.4f}" if len(no_trades) > 0 else "")

        # By confidence level
        print(f"\n  P&L by Confidence Level:")
        for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.0)]:
            mask_c = (trades["confidence"] >= lo) & (trades["confidence"] < hi)
            if mask_c.sum() > 0:
                sub = trades[mask_c]
                print(f"    conf {lo:.2f}-{hi:.2f}: n={len(sub):>4}, "
                      f"pnl={sub['pnl'].mean():+.4f}, "
                      f"win={sub['pnl'].gt(0).mean():.1%}")

        # By edge magnitude
        print(f"\n  P&L by Edge Size:")
        for lo, hi in [(2, 3), (3, 5), (5, 10), (10, 100)]:
            mask_e = (trades["edge"].abs() * 100 >= lo) & (trades["edge"].abs() * 100 < hi)
            if mask_e.sum() > 0:
                sub = trades[mask_e]
                print(f"    edge {lo:>2}-{hi:>3}¢: n={len(sub):>4}, "
                      f"pnl={sub['pnl'].mean():+.4f}, "
                      f"win={sub['pnl'].gt(0).mean():.1%}")

    # Annualized estimate
    if n_trades > 0 and n > 0:
        total_seconds = len(results) * horizon_minutes * 60  # approx
        days = total_seconds / 86400
        daily_pnl = total_pnl / max(days, 1)
        annual_pnl = daily_pnl * 365

        print(f"\n  Annualized Estimate:")
        print(f"    Test period: ~{days:.1f} days")
        print(f"    Daily P&L per $1: {daily_pnl:+.4f}")
        print(f"    Annual P&L per $1: {annual_pnl:+.2f}")
        print(f"    Annual P&L per $1000 bankroll: ${annual_pnl * 1000:+,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Polymarket probability model backtest")
    parser.add_argument("--data-dir", default="../data/features")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=5, help="Horizon in minutes")
    parser.add_argument("--strike-offset", type=float, default=0.5,
                        help="Strike offset %% above current price")
    parser.add_argument("--edge-threshold", type=float, default=2.0,
                        help="Min edge in cents to trade")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    args = parser.parse_args()

    df = load_btc_data(args.data_dir, max_days=args.days)

    results = simulate_polymarket_trading(
        df,
        horizon_minutes=args.horizon,
        strike_offset_pct=args.strike_offset,
        edge_threshold_cents=args.edge_threshold,
        train_fraction=args.train_fraction,
    )

    analyze_results(results, args.horizon, args.strike_offset)

    # Also test with different strike offsets
    print("\n\n" + "=" * 70)
    print("SENSITIVITY: VARYING STRIKE OFFSET")
    print("=" * 70)

    for offset in [0.1, 0.25, 0.5, 1.0, 2.0]:
        print(f"\n--- Strike offset: {offset}% ---")
        r = simulate_polymarket_trading(
            df, horizon_minutes=args.horizon,
            strike_offset_pct=offset,
            edge_threshold_cents=args.edge_threshold,
            train_fraction=args.train_fraction,
        )
        if len(r) > 0:
            trades = r[r["trade_side"].notna()]
            if len(trades) > 0:
                brier_naive = np.mean((r["naive_prob"].values - r["actual"].values) ** 2)
                brier_model = np.mean((r["model_prob"].values - r["actual"].values) ** 2)
                print(f"  Brier: naive={brier_naive:.4f}, model={brier_model:.4f}, "
                      f"improvement={((brier_naive-brier_model)/brier_naive*100):+.1f}%")
                print(f"  Trades: {len(trades)}, mean P&L={trades['pnl'].mean():+.4f}, "
                      f"win={trades['pnl'].gt(0).mean():.1%}")
            else:
                print(f"  No trades at this offset")

    print("\nDone.")


if __name__ == "__main__":
    main()
