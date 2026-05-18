#!/usr/bin/env python3
"""
Skeptical Regression Signal Test Battery

10 systematic tests to validate whether a regression signal is real
or an artifact of overlapping predictions, overfitting, or single-feature dependence.

Tests:
  T1:  Permutation test (shuffle y)         — spurious correlation
  T2:  Effective N (overlap correction)      — overlap inflation
  T3:  Block bootstrap PnL                   — lucky streak
  T4:  Feature ablation (remove funding)     — single-feature dependence
  T5:  Per-day IC stability                  — temporal decay
  T6:  Symbol replication (ETH, SOL)         — generalization
  T7:  Non-overlapping trades only           — real trade count
  T8:  Regime split (funding sign)           — one-sided signal
  T9:  Cost sensitivity                      — fragility
  T10: Walk-forward with full embargo        — train/test leakage

Usage:
    python scripts/skeptical_regression_test.py --symbol BTC --horizon 18000
    python scripts/skeptical_regression_test.py --symbol BTC --horizon 18000 --json-report /tmp/skeptical.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl
import lightgbm as lgb
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from phase1_signal_test import load_all_data, create_target, get_feature_columns


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    test_id: str
    name: str
    passed: bool
    statistic: float
    threshold: float
    p_value: Optional[float] = None
    detail: str = ""
    verdict: str = ""  # PASS / FAIL / WARN

    def __post_init__(self):
        if not self.verdict:
            self.verdict = "PASS" if self.passed else "FAIL"


# ---------------------------------------------------------------------------
# Shared model helper
# ---------------------------------------------------------------------------


def _train_regressor(X_train, y_train, seed=42, fast=False):
    n_estimators = 50 if fast else 200
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        verbose=-1,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def _subsample(X, y, max_rows=10000, seed=42):
    """Subsample data if too large — overlapping rows are redundant."""
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), max_rows, replace=False))
    return X[idx], y[idx]


def _walkforward_ic(X, y, n_splits=5, embargo=0, seed=42, fast=False):
    """Run walk-forward regression and return per-split IC values."""
    n = len(X)
    min_train = int(n * 0.2)
    test_size = int(n * 0.8 / n_splits)
    ics = []
    for i in range(n_splits):
        train_end = min_train + i * test_size
        test_start = train_end + embargo
        test_end = min(test_start + test_size, n)
        if test_start >= n or test_end - test_start < 100:
            break
        model = _train_regressor(X[:train_end], y[:train_end], seed=seed, fast=fast)
        y_pred = model.predict(X[test_start:test_end])
        y_true = y[test_start:test_end]
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            ic = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            ic = 0.0
        ics.append(ic)
    return ics


# ---------------------------------------------------------------------------
# T1: Permutation Test
# ---------------------------------------------------------------------------


def test_t1_permutation(X, y, n_splits=5, n_perms=200, seed=42, max_rows=10000):
    """Shuffle y, rerun walk-forward, check if real IC beats null."""
    print("\n  T1: Permutation Test (shuffle y)")
    rng = np.random.default_rng(seed)

    # Subsample for permutation — overlapping rows are redundant
    Xs, ys = _subsample(X, y, max_rows=max_rows, seed=seed)
    if len(Xs) < len(X):
        print(f"      Subsampled: {len(X):,} → {len(Xs):,} rows")

    real_ics = _walkforward_ic(Xs, ys, n_splits=n_splits, seed=seed, fast=True)
    real_ic = np.mean(real_ics)
    print(f"      Real IC: {real_ic:.4f}")

    null_ics = []
    for p in range(n_perms):
        y_shuf = rng.permutation(ys)
        perm_ic = np.mean(_walkforward_ic(Xs, y_shuf, n_splits=n_splits, seed=seed, fast=True))
        null_ics.append(perm_ic)
        if (p + 1) % 50 == 0:
            print(f"      ... {p + 1}/{n_perms} permutations done")

    null_ics = np.array(null_ics)
    p_value = float(np.mean(null_ics >= real_ic))
    passed = p_value < 0.05

    print(f"      Null IC 95th pctile: {np.percentile(null_ics, 95):.4f}")
    print(f"      p-value: {p_value:.4f} {'PASS' if passed else 'FAIL'}")

    return TestResult(
        "T1", "Permutation Test", passed,
        statistic=real_ic, threshold=float(np.percentile(null_ics, 95)),
        p_value=p_value,
        detail=f"Real IC={real_ic:.4f} vs null 95th={np.percentile(null_ics, 95):.4f}",
    )


# ---------------------------------------------------------------------------
# T2: Effective N
# ---------------------------------------------------------------------------


def test_t2_effective_n(observed_ic, n_trades, horizon, dt=1):
    """Check if IC is significant after overlap correction."""
    print("\n  T2: Effective N (Overlap Correction)")

    effective_n = n_trades * dt / horizon
    se_ic = 1.0 / np.sqrt(max(effective_n, 1))
    z = observed_ic / se_ic
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    passed = p_value < 0.05

    print(f"      Total trades: {n_trades}")
    print(f"      Effective N: {effective_n:.1f}")
    print(f"      IC: {observed_ic:.4f}, SE: {se_ic:.4f}, z: {z:.3f}")
    print(f"      p-value: {p_value:.4f} {'PASS' if passed else 'FAIL'}")

    return TestResult(
        "T2", "Effective N (Overlap)", passed,
        statistic=effective_n, threshold=0.05,
        p_value=p_value,
        detail=f"effective_n={effective_n:.1f}, z={z:.3f}",
    )


# ---------------------------------------------------------------------------
# T3: Block Bootstrap PnL
# ---------------------------------------------------------------------------


def test_t3_block_bootstrap(pnl_per_block, n_bootstrap=1000, seed=42):
    """Bootstrap non-overlapping block PnLs, check if Sharpe > 0."""
    print("\n  T3: Block Bootstrap PnL")
    rng = np.random.default_rng(seed)
    n_blocks = len(pnl_per_block)

    if n_blocks < 3:
        print(f"      Only {n_blocks} blocks — too few for bootstrap")
        return TestResult(
            "T3", "Block Bootstrap PnL", False,
            statistic=0.0, threshold=0.0,
            detail=f"Only {n_blocks} non-overlapping blocks",
        )

    boot_sharpes = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_blocks, n_blocks, replace=True)
        sample = pnl_per_block[idx]
        s = np.mean(sample) / (np.std(sample) + 1e-12) * np.sqrt(n_blocks)
        boot_sharpes.append(s)

    boot_sharpes = np.array(boot_sharpes)
    pct5 = float(np.percentile(boot_sharpes, 5))
    passed = pct5 > 0

    print(f"      Blocks: {n_blocks}")
    print(f"      Bootstrap Sharpe median: {np.median(boot_sharpes):.2f}")
    print(f"      Bootstrap Sharpe 5th pctile: {pct5:.2f} {'PASS' if passed else 'FAIL'}")

    return TestResult(
        "T3", "Block Bootstrap PnL", passed,
        statistic=pct5, threshold=0.0,
        detail=f"n_blocks={n_blocks}, sharpe_5th={pct5:.3f}",
    )


# ---------------------------------------------------------------------------
# T4: Feature Ablation
# ---------------------------------------------------------------------------


def test_t4_feature_ablation(X, y, feature_cols, n_splits=5, seed=42, max_rows=20000):
    """Remove funding rate features, check if signal persists."""
    print("\n  T4: Feature Ablation (Remove Funding Rate)")

    funding_keywords = ["ctx_funding"]
    ablated_idx = [
        i for i, c in enumerate(feature_cols)
        if not any(k in c for k in funding_keywords)
    ]
    removed = [c for c in feature_cols if any(k in c for k in funding_keywords)]
    print(f"      Removed features: {removed}")

    if not ablated_idx:
        return TestResult(
            "T4", "Feature Ablation", False,
            statistic=0.0, threshold=0.05,
            detail="No features left after ablation",
        )

    # Subsample for speed — ablation comparison is relative, not absolute
    Xs, ys = _subsample(X, y, max_rows=max_rows, seed=seed)
    if len(Xs) < len(X):
        print(f"      Subsampled: {len(X):,} → {len(Xs):,} rows")

    X_ablated = Xs[:, ablated_idx]
    full_ics = _walkforward_ic(Xs, ys, n_splits=n_splits, seed=seed)
    ablated_ics = _walkforward_ic(X_ablated, ys, n_splits=n_splits, seed=seed)

    full_ic = np.mean(full_ics)
    ablated_ic = np.mean(ablated_ics)
    ic_drop_pct = (full_ic - ablated_ic) / (abs(full_ic) + 1e-10) * 100

    passed = ablated_ic > 0.05
    print(f"      Full IC: {full_ic:.4f}")
    print(f"      Ablated IC: {ablated_ic:.4f} (drop: {ic_drop_pct:.0f}%)")
    print(f"      {'PASS' if passed else 'FAIL'}")

    return TestResult(
        "T4", "Feature Ablation (Funding)", passed,
        statistic=ablated_ic, threshold=0.05,
        detail=f"full_ic={full_ic:.4f}, ablated_ic={ablated_ic:.4f}, drop={ic_drop_pct:.0f}%",
    )


# ---------------------------------------------------------------------------
# T5: Per-Day IC Stability
# ---------------------------------------------------------------------------


def test_t5_temporal_stability(df, feature_cols, horizon, seed=42):
    """Compute IC per day, check for temporal decay."""
    print("\n  T5: Per-Day IC Stability")

    X_all = df.select(feature_cols).to_numpy()
    y_all = df["forward_return"].to_numpy()
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    ts = df["timestamp_ns"].to_numpy()
    # Convert to day boundaries
    ns_per_day = 86400 * 1_000_000_000
    days = (ts - ts.min()) // ns_per_day
    unique_days = np.unique(days)

    day_ics = []
    for d in unique_days:
        mask = days == d
        n_day = mask.sum()
        if n_day < 500:
            continue

        # Train on all data before this day, predict on this day
        train_mask = days < d
        if train_mask.sum() < 1000:
            continue

        # Subsample training data for speed (keep all test data)
        X_tr, y_tr = _subsample(X_all[train_mask], y_all[train_mask], max_rows=20000, seed=seed)
        model = _train_regressor(X_tr, y_tr, seed=seed)
        y_pred = model.predict(X_all[mask])
        y_true = y_all[mask]
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            ic = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            ic = 0.0
        day_ics.append({"day_idx": int(d), "ic": ic, "n_rows": int(n_day)})
        print(f"      Day {int(d)}: IC={ic:+.4f} ({n_day} rows)")

    n_days = len(day_ics)
    if n_days == 0:
        return TestResult(
            "T5", "Per-Day IC Stability", False,
            statistic=0.0, threshold=0.0,
            detail="No days with enough data",
        )

    positive_days = sum(1 for d in day_ics if d["ic"] > 0)
    last_3_ics = [d["ic"] for d in day_ics[-3:]]
    last_3_all_neg = all(ic <= 0 for ic in last_3_ics)

    # Trend: Spearman correlation of day index vs IC
    if n_days >= 3:
        trend_corr, _ = stats.spearmanr(
            [d["day_idx"] for d in day_ics],
            [d["ic"] for d in day_ics],
        )
    else:
        trend_corr = 0.0

    passed = positive_days >= n_days * 0.7 and not last_3_all_neg
    avg_ic = np.mean([d["ic"] for d in day_ics])

    print(f"      Days with IC>0: {positive_days}/{n_days}")
    print(f"      Last 3 days IC: {last_3_ics}")
    print(f"      Trend (Spearman): {trend_corr:+.3f}")
    print(f"      {'PASS' if passed else 'FAIL'}")

    return TestResult(
        "T5", "Per-Day IC Stability", passed,
        statistic=avg_ic, threshold=0.0,
        detail=f"positive_days={positive_days}/{n_days}, trend={trend_corr:+.3f}, last3={last_3_ics}",
    )


# ---------------------------------------------------------------------------
# T6: Symbol Replication
# ---------------------------------------------------------------------------


def test_t6_symbol_replication(
    data_dir, symbols, horizon, seed=42,
    start_date=None, end_date=None, max_memory_mb=2000.0,
):
    """Run same regression on other symbols, check if signal replicates."""
    print("\n  T6: Symbol Replication")

    symbol_ics = []
    for sym in symbols:
        try:
            df = load_all_data(data_dir, sym, start_date=start_date,
                               end_date=end_date, max_memory_mb=max_memory_mb)
            feature_cols = get_feature_columns(df, remove_leaky=True)
            df = create_target(df, horizon)
            if len(df) < 5000:
                print(f"      {sym}: too few rows ({len(df)})")
                continue

            X = df.select(feature_cols).to_numpy()
            y = df["forward_return"].to_numpy()
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            ics = _walkforward_ic(X, y, n_splits=5, seed=seed)
            avg_ic = float(np.mean(ics))
            symbol_ics.append({"symbol": sym, "ic": avg_ic})
            print(f"      {sym}: IC={avg_ic:+.4f}")
        except Exception as e:
            print(f"      {sym}: FAILED ({e})")

    n_replicated = sum(1 for s in symbol_ics if s["ic"] > 0.05)
    passed = n_replicated >= 1

    print(f"      Replicated (IC>0.05): {n_replicated}/{len(symbol_ics)} {'PASS' if passed else 'FAIL'}")

    return TestResult(
        "T6", "Symbol Replication", passed,
        statistic=n_replicated, threshold=1.0,
        detail=str(symbol_ics),
    )


# ---------------------------------------------------------------------------
# T7: Non-Overlapping Trades
# ---------------------------------------------------------------------------


def test_t7_nonoverlapping(X, y, horizon, seed=42):
    """Trade only every H bars — real independent trade count."""
    print("\n  T7: Non-Overlapping Trades Only")

    split = int(len(X) * 0.6)
    model = _train_regressor(X[:split], y[:split], seed=seed)
    y_pred = model.predict(X[split:])
    y_true = y[split:]

    # Take every horizon-th prediction
    indices = np.arange(0, len(y_pred), horizon)
    pred_sampled = y_pred[indices]
    true_sampled = y_true[indices]
    n_trades = len(indices)

    # PnL: go long if predicted positive, short if negative
    positions = np.sign(pred_sampled)
    gross_pnl = positions * true_sampled
    avg_gross_bps = float(np.mean(gross_pnl) * 10000)

    taker_cost_bps = 8.0
    avg_net_bps = avg_gross_bps - taker_cost_bps
    maker_cost_bps = 1.0
    avg_net_maker_bps = avg_gross_bps - maker_cost_bps

    win_rate = float(np.mean(gross_pnl > 0)) if n_trades > 0 else 0.0
    sharpe = float(np.mean(gross_pnl) / (np.std(gross_pnl) + 1e-12) * np.sqrt(n_trades))

    passed = avg_net_bps > 0 and n_trades >= 10

    print(f"      Independent trades: {n_trades}")
    print(f"      Gross: {avg_gross_bps:+.2f} bp")
    print(f"      Net (taker): {avg_net_bps:+.2f} bp")
    print(f"      Net (maker): {avg_net_maker_bps:+.2f} bp")
    print(f"      Win rate: {win_rate:.1%}")
    print(f"      Sharpe: {sharpe:.2f}")
    print(f"      {'PASS' if passed else 'FAIL'}")

    return TestResult(
        "T7", "Non-Overlapping Trades", passed,
        statistic=avg_net_bps, threshold=0.0,
        detail=f"n_trades={n_trades}, gross={avg_gross_bps:+.2f}bp, net_taker={avg_net_bps:+.2f}bp, win_rate={win_rate:.1%}",
    )


# ---------------------------------------------------------------------------
# T8: Regime Split
# ---------------------------------------------------------------------------


def test_t8_regime_split(df, feature_cols, X, y, seed=42):
    """Split by funding rate sign, check IC in both regimes."""
    print("\n  T8: Regime Split (Funding Rate Sign)")

    if "ctx_funding_rate" not in df.columns:
        return TestResult(
            "T8", "Regime Split", False,
            statistic=0.0, threshold=0.0,
            detail="ctx_funding_rate not in data",
        )

    funding = df["ctx_funding_rate"].to_numpy()

    split = int(len(X) * 0.6)
    model = _train_regressor(X[:split], y[:split], seed=seed)
    y_pred = model.predict(X[split:])
    y_true = y[split:]
    funding_test = funding[split:]

    pos_mask = funding_test > 0
    neg_mask = funding_test <= 0

    results = {}
    for label, mask in [("positive_funding", pos_mask), ("negative_funding", neg_mask)]:
        n = mask.sum()
        if n < 100:
            results[label] = {"ic": 0.0, "n": int(n)}
            continue
        p, t = y_pred[mask], y_true[mask]
        ic = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 and np.std(t) > 0 else 0.0
        results[label] = {"ic": ic, "n": int(n)}
        print(f"      {label}: IC={ic:+.4f} (n={n})")

    both_positive = all(v["ic"] > 0 for v in results.values() if v["n"] >= 100)
    one_strong = any(v["ic"] > 0.1 for v in results.values() if v["n"] >= 100)
    passed = both_positive or one_strong

    print(f"      {'PASS' if passed else 'FAIL'}")

    return TestResult(
        "T8", "Regime Split (Funding)", passed,
        statistic=min(v["ic"] for v in results.values() if v["n"] >= 100) if results else 0.0,
        threshold=0.0,
        detail=str(results),
    )


# ---------------------------------------------------------------------------
# T9: Cost Sensitivity
# ---------------------------------------------------------------------------


def test_t9_cost_sensitivity(gross_bps_per_trade):
    """Find breakeven cost, check buffer above taker."""
    print("\n  T9: Cost Sensitivity")

    taker_cost = 8.0
    costs = np.arange(0, 16, 0.5)
    net_at_cost = [(c, gross_bps_per_trade - c) for c in costs]

    # Breakeven = gross PnL per trade (that's the cost at which net = 0)
    breakeven = gross_bps_per_trade
    buffer = breakeven - taker_cost
    passed = breakeven > 10.0  # at least 2bp buffer above taker

    print(f"      Gross per trade: {gross_bps_per_trade:+.2f} bp")
    print(f"      Breakeven cost: {breakeven:.2f} bp")
    print(f"      Taker cost: {taker_cost:.1f} bp")
    print(f"      Buffer: {buffer:+.2f} bp")
    print(f"      {'PASS' if passed else 'FAIL'}")

    return TestResult(
        "T9", "Cost Sensitivity", passed,
        statistic=breakeven, threshold=10.0,
        detail=f"breakeven={breakeven:.2f}bp, buffer={buffer:+.2f}bp",
    )


# ---------------------------------------------------------------------------
# T10: Walk-Forward with Embargo
# ---------------------------------------------------------------------------


def test_t10_embargo_walkforward(X, y, horizon, n_splits=5, seed=42, max_rows=20000):
    """Walk-forward with embargo = full horizon between train and test."""
    print("\n  T10: Walk-Forward with Full Embargo")

    # Subsample for speed — scale embargo proportionally
    Xs, ys = _subsample(X, y, max_rows=max_rows, seed=seed)
    if len(Xs) < len(X):
        scale = len(Xs) / len(X)
        embargo_scaled = max(int(horizon * scale), 100)
        print(f"      Subsampled: {len(X):,} → {len(Xs):,} rows (embargo: {horizon} → {embargo_scaled})")
    else:
        embargo_scaled = horizon

    ics = _walkforward_ic(Xs, ys, n_splits=n_splits, embargo=embargo_scaled, seed=seed)
    if not ics:
        return TestResult(
            "T10", "Embargo Walk-Forward", False,
            statistic=0.0, threshold=0.05,
            detail="No valid splits after embargo",
        )

    avg_ic = float(np.mean(ics))
    passed = avg_ic > 0.05

    for i, ic in enumerate(ics):
        print(f"      Split {i + 1}: IC={ic:+.4f}")
    print(f"      Average IC: {avg_ic:+.4f} {'PASS' if passed else 'FAIL'}")

    return TestResult(
        "T10", "Embargo Walk-Forward", passed,
        statistic=avg_ic, threshold=0.05,
        detail=f"avg_ic={avg_ic:.4f}, splits={[round(ic, 4) for ic in ics]}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Skeptical Regression Signal Test Battery")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--horizon", type=int, default=18000)
    parser.add_argument("--json-report", default=None)
    parser.add_argument("--max-memory-mb", type=float, default=2000.0)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("SKEPTICAL REGRESSION SIGNAL TEST BATTERY")
    print("=" * 70)
    print(f"  Symbol: {args.symbol}")
    print(f"  Horizon: {args.horizon} rows ({args.horizon * 0.1:.0f}s)")
    print()

    # Load data
    df = load_all_data(
        args.data_dir, args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        max_memory_mb=args.max_memory_mb,
    )
    feature_cols = get_feature_columns(df, remove_leaky=True)
    df = create_target(df, args.horizon)
    n_rows = len(df)
    print(f"  Data: {n_rows:,} rows, {len(feature_cols)} features")
    print()

    X = df.select(feature_cols).to_numpy()
    y = df["forward_return"].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Baseline metrics (needed by several tests) ---
    print("  Computing baseline walk-forward IC...")
    Xb, yb = _subsample(X, y, max_rows=20000, seed=args.seed)
    if len(Xb) < len(X):
        print(f"  Subsampled for baseline: {len(X):,} → {len(Xb):,} rows")
    baseline_ics = _walkforward_ic(Xb, yb, n_splits=5, seed=args.seed)
    baseline_ic = float(np.mean(baseline_ics))
    print(f"  Baseline IC: {baseline_ic:.4f}")

    # Baseline quantile PnL (top 10%) — use subsampled training for speed
    split = int(len(X) * 0.6)
    X_train_sub, y_train_sub = _subsample(X[:split], y[:split], max_rows=20000, seed=args.seed)
    model_base = _train_regressor(X_train_sub, y_train_sub, seed=args.seed)
    y_pred_base = model_base.predict(X[split:])
    y_true_base = y[split:]
    top_thresh = np.quantile(y_pred_base, 0.90)
    bot_thresh = np.quantile(y_pred_base, 0.10)
    trade_mask = (y_pred_base >= top_thresh) | (y_pred_base <= bot_thresh)
    positions = np.where(y_pred_base[trade_mask] >= top_thresh, 1.0, -1.0)
    trade_returns = y_true_base[trade_mask]
    gross_pnl = positions * trade_returns
    avg_gross_bps = float(np.mean(gross_pnl) * 10000) if len(gross_pnl) > 0 else 0.0
    n_trades_baseline = int(trade_mask.sum())

    # Block PnL for T3
    n_test = len(y_true_base)
    block_pnls = []
    for start in range(0, n_test, args.horizon):
        end = min(start + args.horizon, n_test)
        if end - start < args.horizon // 2:
            break
        block_pred = y_pred_base[start:end]
        block_true = y_true_base[start:end]
        # Top/bottom 10% within block
        if len(block_pred) < 100:
            continue
        bt = np.quantile(block_pred, 0.90)
        bb = np.quantile(block_pred, 0.10)
        bm = (block_pred >= bt) | (block_pred <= bb)
        if bm.sum() < 5:
            continue
        bp = np.where(block_pred[bm] >= bt, 1.0, -1.0)
        block_pnls.append(float(np.mean(bp * block_true[bm]) * 10000))
    block_pnls = np.array(block_pnls)

    # --- Run tests (cheapest first) ---
    results: List[TestResult] = []

    t0 = time.time()

    # T2: Effective N (pure math)
    results.append(test_t2_effective_n(baseline_ic, n_trades_baseline, args.horizon))

    # T9: Cost sensitivity (no training)
    results.append(test_t9_cost_sensitivity(avg_gross_bps))

    # T8: Regime split (one model already trained)
    results.append(test_t8_regime_split(df, feature_cols, X, y, seed=args.seed))

    # T5: Per-day IC stability
    results.append(test_t5_temporal_stability(df, feature_cols, args.horizon, seed=args.seed))

    # T7: Non-overlapping trades
    results.append(test_t7_nonoverlapping(X, y, args.horizon, seed=args.seed))

    # T3: Block bootstrap PnL
    results.append(test_t3_block_bootstrap(block_pnls, n_bootstrap=1000, seed=args.seed))

    # T10: Embargo walk-forward
    results.append(test_t10_embargo_walkforward(X, y, args.horizon, n_splits=5, seed=args.seed))

    # T4: Feature ablation (expensive)
    results.append(test_t4_feature_ablation(X, y, feature_cols, n_splits=5, seed=args.seed))

    # T1: Permutation (most expensive)
    results.append(test_t1_permutation(X, y, n_splits=5, n_perms=args.n_permutations, seed=args.seed))

    # T6: Symbol replication (loads new data)
    replication_symbols = [s for s in ["ETH", "SOL"] if s != args.symbol]
    results.append(test_t6_symbol_replication(
        args.data_dir, replication_symbols, args.horizon,
        seed=args.seed, start_date=args.start_date,
        end_date=args.end_date, max_memory_mb=args.max_memory_mb,
    ))

    elapsed = time.time() - t0

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Test':<35} {'Verdict':>8}  {'Statistic':>12}  Detail")
    print(f"  {'-'*35} {'-'*8}  {'-'*12}  {'-'*30}")
    for r in results:
        print(f"  {r.test_id + ': ' + r.name:<35} {r.verdict:>8}  {r.statistic:>+12.4f}  {r.detail[:50]}")

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)

    # Hard kill: T2 + T7 both fail
    t2 = next((r for r in results if r.test_id == "T2"), None)
    t7 = next((r for r in results if r.test_id == "T7"), None)
    hard_kill = (t2 and not t2.passed) and (t7 and not t7.passed)

    if hard_kill or n_fail >= 4:
        overall = "REJECT"
    elif n_fail >= 2:
        overall = "INVESTIGATE"
    else:
        overall = "PROCEED"

    print(f"\n  Passed: {n_pass}/10  Failed: {n_fail}/10")
    if hard_kill:
        print(f"  HARD KILL: T2 (Effective N) + T7 (Non-Overlapping) both failed")
    print(f"  OVERALL VERDICT: {overall}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print()

    # --- JSON Report ---
    if args.json_report:
        report = {
            "symbol": args.symbol,
            "horizon": args.horizon,
            "n_rows": n_rows,
            "n_features": len(feature_cols),
            "baseline_ic": baseline_ic,
            "baseline_gross_bps": avg_gross_bps,
            "n_trades_baseline": n_trades_baseline,
            "tests": [asdict(r) for r in results],
            "overall_verdict": overall,
            "summary": {"passed": n_pass, "failed": n_fail},
            "elapsed_s": round(elapsed, 1),
        }
        Path(args.json_report).parent.mkdir(parents=True, exist_ok=True)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(args.json_report, "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        print(f"  JSON report: {args.json_report}")


if __name__ == "__main__":
    main()
