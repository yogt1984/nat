#!/usr/bin/env python3
"""
Q3 Predictive Quality Test for Q1+Q2-passing cluster configurations.

Tests whether discovered clusters predict forward returns using:
  - Kruskal-Wallis H-test (p < 0.05)
  - Eta-squared effect size (> 0.01)
  - Self-transition rate (regime persistence)
  - Per-cluster mean return and Sharpe ratio
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.cluster_pipeline.loader import load_parquet
from scripts.cluster_pipeline.preprocess import aggregate_bars, preprocess
from scripts.cluster_pipeline.cluster import fit_gmm, predictive_quality

# Q1+Q2 winners from sweep
CONFIGS = [
    {"vector": "entropy",   "timeframe": "5min",  "k": 2},
    {"vector": "orderflow", "timeframe": "5min",  "k": 4},
    {"vector": "orderflow", "timeframe": "15min", "k": 3},
    {"vector": "derived",   "timeframe": "15min", "k": 2},
    {"vector": "volatility","timeframe": "2h",    "k": 4},
    {"vector": "orderflow", "timeframe": "2h",    "k": 4},
]

Q3_THRESHOLDS = {
    "kruskal_wallis_p": 0.05,
    "eta_squared": 0.01,
    "self_transition_rate": 0.7,
}


def compute_forward_returns(bars: pd.DataFrame, timeframe: str) -> np.ndarray:
    """Compute 1-bar-ahead log returns from bar midprice close."""
    # Find price column in bars
    price_col = None
    for candidate in ["raw_midprice_close", "raw_midprice_last", "raw_midprice_mean",
                       "raw_microprice_close", "raw_microprice_last", "raw_microprice_mean"]:
        if candidate in bars.columns:
            price_col = candidate
            break

    if price_col is None:
        # Fallback: look for anything with midprice
        mid_cols = [c for c in bars.columns if "midprice" in c]
        if mid_cols:
            price_col = mid_cols[0]
        else:
            raise ValueError(f"No price column found. Available: {list(bars.columns)[:20]}")

    prices = bars[price_col].values.astype(np.float64)
    # Log returns: log(p_{t+1} / p_t), last bar gets NaN
    fwd = np.full(len(prices), np.nan)
    fwd[:-1] = np.log(prices[1:] / prices[:-1])
    return fwd


def run_q3():
    print("Loading data...")
    df = load_parquet("data/features")
    print(f"  {len(df):,} rows loaded")

    results = []

    for cfg in CONFIGS:
        vec = cfg["vector"]
        tf = cfg["timeframe"]
        k = cfg["k"]
        label = f"{vec}@{tf} (k={k})"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        # Aggregate bars
        bars = aggregate_bars(df, tf)
        print(f"  Bars: {len(bars)}")

        # Compute forward returns before preprocessing (so indices align)
        fwd_returns = compute_forward_returns(bars, tf)

        # Preprocess features
        X, cols, meta = preprocess(bars, vector=vec, scaler="zscore")
        print(f"  Features: {X.shape[1]}")

        # Drop bars where forward return is NaN (last bar)
        valid = ~np.isnan(fwd_returns)
        X_valid = X[valid]
        fwd_valid = fwd_returns[valid]

        # Fit GMM
        cr = fit_gmm(X_valid, k=k, random_state=42)
        labels = cr.labels

        # Predictive quality
        pq_result = predictive_quality(labels, fwd_valid)

        # Q3 gate
        q3_pass = pq_result["significant"]  # p < 0.05 AND eta² > 0.01
        str_pass = pq_result["self_transition_rate"] >= Q3_THRESHOLDS["self_transition_rate"]

        print(f"\n  Kruskal-Wallis H:      {pq_result['kruskal_wallis_h']:.2f}")
        print(f"  Kruskal-Wallis p:      {pq_result['kruskal_wallis_p']:.4e}")
        print(f"  Eta-squared:           {pq_result['eta_squared']:.4f}")
        print(f"  Self-transition rate:  {pq_result['self_transition_rate']:.3f}")
        print(f"  Q3 (significant):      {'PASS' if q3_pass else 'FAIL'}")
        print(f"  Self-transition gate:  {'PASS' if str_pass else 'FAIL'}")

        print(f"\n  Per-cluster statistics:")
        for c, stats in sorted(pq_result["per_cluster"].items()):
            print(f"    State {c}: n={stats['count']:4d}, "
                  f"mean={stats['mean_return']:+.6f}, "
                  f"std={stats['std_return']:.6f}, "
                  f"sharpe={stats['sharpe']:+.3f}")

        entry = {
            **cfg,
            "label": label,
            "n_bars": int(X_valid.shape[0]),
            "kruskal_h": pq_result["kruskal_wallis_h"],
            "kruskal_p": pq_result["kruskal_wallis_p"],
            "eta_squared": pq_result["eta_squared"],
            "self_transition_rate": pq_result["self_transition_rate"],
            "q3_significant": q3_pass,
            "q3_str_pass": str_pass,
            "q3_full_pass": q3_pass and str_pass,
            "per_cluster": pq_result["per_cluster"],
        }
        results.append(entry)

    # Summary
    print("\n" + "="*60)
    print("  Q3 SUMMARY")
    print("="*60)
    print(f"  {'Config':<28} {'K-W p':>10} {'eta²':>8} {'STR':>6} {'Q3':>6}")
    print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")
    for r in results:
        status = "PASS" if r["q3_full_pass"] else ("sig" if r["q3_significant"] else "FAIL")
        print(f"  {r['label']:<28} {r['kruskal_p']:>10.4e} {r['eta_squared']:>8.4f} "
              f"{r['self_transition_rate']:>6.3f} {status:>6}")

    # Full pass count
    full_pass = [r for r in results if r["q3_full_pass"]]
    sig_only = [r for r in results if r["q3_significant"] and not r["q3_full_pass"]]
    print(f"\n  Full Q3 pass (sig + STR >= 0.7): {len(full_pass)}/6")
    print(f"  Significant but low STR:         {len(sig_only)}/6")

    # Save results (convert numpy types)
    out_path = "reports/q3_predictive_quality.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run_q3()
