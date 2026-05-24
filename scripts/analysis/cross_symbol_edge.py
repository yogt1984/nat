"""
Cross-Symbol Edge Quantification

Tests the three most promising signals found in the microstructure research:
1. BTC→SOL lead-lag (IC=0.30 at 10 ticks = 1s)
2. BTC→ETH lead-lag (IC=0.16 at 10 ticks = 1s)
3. BTC-ETH spread convergence (IC=0.13 at 100 ticks)
4. Premium reversion (IC=0.26 at 16.7min)

For each signal: measures edge in bps, compares to SOL/ETH taker fees,
estimates net PnL per trade, and runs quintile analysis.

Usage:
    cd scripts && python -m analysis.cross_symbol_edge
    cd scripts && python -m analysis.cross_symbol_edge --days 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add scripts to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cluster_pipeline.loader import load_parquet


# Hyperliquid fee structure (bps, round-trip)
TAKER_FEE_RT = {
    "BTC": 7.0,   # 3.5 bps each way
    "ETH": 7.0,
    "SOL": 7.0,
}
MAKER_FEE_RT = {
    "BTC": 0.4,   # 0.2 bps each way (no rebate currently)
    "ETH": 0.4,
    "SOL": 0.4,
}


def load_aligned(data_dir: str, max_days: int | None = None) -> pd.DataFrame:
    """Load all symbols and time-align to 1-second buckets."""
    cols = [
        "timestamp_ns", "symbol", "raw_midprice",
        "imbalance_qty_l1", "flow_volume_1s",
        "ctx_funding_rate", "ctx_funding_zscore",
        "ctx_premium_bps", "ent_book_shape",
    ]

    kwargs = {}
    if max_days:
        # Load most recent N days
        dates = sorted(Path(data_dir).iterdir())
        dates = [d for d in dates if d.is_dir() and not d.name.endswith("-clean")]
        if len(dates) > max_days:
            start_date = dates[-max_days].name
            kwargs["start_date"] = start_date

    df = load_parquet(data_dir, columns=cols, **kwargs)
    print(f"Loaded {len(df):,} rows, symbols: {df['symbol'].unique()}")

    # Create 1-second time buckets
    df["ts_s"] = df["timestamp_ns"] // 1_000_000_000

    # Pivot: one row per second, columns per symbol
    aligned = {}
    for sym in ["BTC", "ETH", "SOL"]:
        sdf = df[df["symbol"] == sym].copy()
        # Take last tick per second (most recent state)
        sdf = sdf.groupby("ts_s").last().reset_index()
        sdf = sdf.set_index("ts_s")
        # Rename columns
        for c in sdf.columns:
            if c != "symbol":
                aligned[f"{sym}_{c}"] = sdf[c]

    result = pd.DataFrame(aligned)
    result = result.dropna(subset=["BTC_raw_midprice", "SOL_raw_midprice", "ETH_raw_midprice"])
    result = result.sort_index()
    print(f"Aligned: {len(result):,} seconds with all 3 symbols")
    return result


def compute_returns(df: pd.DataFrame, sym: str, horizons: list[int]) -> pd.DataFrame:
    """Compute forward returns in bps at multiple horizons (in seconds)."""
    mid = df[f"{sym}_raw_midprice"]
    for h in horizons:
        df[f"{sym}_fret_{h}s"] = (mid.shift(-h) / mid - 1) * 10000  # bps
    return df


def signal_edge_analysis(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str,
    signal_name: str,
    n_quantiles: int = 5,
):
    """Analyze a signal's edge: IC, quintile spreads, conditional returns."""
    mask = df[signal_col].notna() & df[return_col].notna()
    sig = df.loc[mask, signal_col].values
    ret = df.loc[mask, return_col].values
    n = len(sig)

    if n < 1000:
        print(f"  {signal_name}: insufficient data ({n} obs)")
        return None

    # IC (rank correlation)
    ic = stats.spearmanr(sig, ret).statistic

    # Quintile analysis
    try:
        quantile_labels = pd.qcut(sig, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        print(f"  {signal_name}: can't form quantiles (degenerate signal)")
        return None

    quintile_returns = pd.Series(ret).groupby(quantile_labels).agg(["mean", "std", "count"])
    q_top = quintile_returns.iloc[-1]["mean"]  # Top quintile mean return
    q_bot = quintile_returns.iloc[0]["mean"]   # Bottom quintile mean return
    spread = q_top - q_bot  # Long-short spread in bps

    # Directional edge: if signal > 0, go long; < 0, go short
    long_mask = sig > np.median(sig)
    short_mask = sig < np.median(sig)
    long_ret = ret[long_mask].mean() if long_mask.sum() > 0 else 0
    short_ret = ret[short_mask].mean() if short_mask.sum() > 0 else 0
    directional_edge = long_ret - short_ret  # bps

    # Conditional edge: top decile
    p90 = np.percentile(sig, 90)
    p10 = np.percentile(sig, 10)
    strong_long = ret[sig > p90].mean() if (sig > p90).sum() > 100 else np.nan
    strong_short = ret[sig < p10].mean() if (sig < p10).sum() > 100 else np.nan

    # Hit rate
    correct_long = (ret[long_mask] > 0).mean() if long_mask.sum() > 0 else 0
    correct_short = (ret[short_mask] < 0).mean() if short_mask.sum() > 0 else 0

    result = {
        "signal": signal_name,
        "n_obs": n,
        "ic": ic,
        "spread_bps": spread,
        "directional_edge_bps": directional_edge,
        "strong_long_bps": strong_long,
        "strong_short_bps": strong_short,
        "hit_rate_long": correct_long,
        "hit_rate_short": correct_short,
        "quintile_returns": quintile_returns,
    }

    print(f"\n  === {signal_name} ===")
    print(f"  N={n:,} | IC={ic:+.4f}")
    print(f"  Q5-Q1 spread: {spread:+.2f} bps")
    print(f"  Directional edge (above/below median): {directional_edge:+.2f} bps")
    print(f"  Strong long (P90): {strong_long:+.2f} bps | Strong short (P10): {strong_short:+.2f} bps")
    print(f"  Hit rate: long={correct_long:.1%} | short={correct_short:.1%}")
    print(f"  Quintile breakdown (bps):")
    for qi in range(len(quintile_returns)):
        qr = quintile_returns.iloc[qi]
        print(f"    Q{qi+1}: mean={qr['mean']:+.3f} std={qr['std']:.3f} n={int(qr['count'])}")

    return result


def cost_viability(results: list[dict], target_symbol: str):
    """Compare signal edges against actual fee costs."""
    taker_rt = TAKER_FEE_RT[target_symbol]

    print(f"\n{'='*70}")
    print(f"COST VIABILITY — {target_symbol} taker RT = {taker_rt:.1f} bps")
    print(f"{'='*70}")
    print(f"{'Signal':<35} {'Edge':>8} {'Fee':>8} {'Net':>8} {'Ratio':>8}")
    print("-" * 70)

    for r in results:
        if r is None:
            continue
        # Use strong-signal edge (P90/P10) as the tradeable edge
        edge = r.get("strong_long_bps", 0)
        if np.isnan(edge):
            edge = r["directional_edge_bps"]
        net = abs(edge) - taker_rt
        ratio = abs(edge) / taker_rt if taker_rt > 0 else 0
        viable = "OK" if net > 0 else "NO"
        print(f"  {r['signal']:<33} {abs(edge):>7.2f} {taker_rt:>7.1f} {net:>+7.2f} {ratio:>7.2f}x  {viable}")


def taker_simulation(
    df: pd.DataFrame,
    signal_col: str,
    target_mid_col: str,
    target_symbol: str,
    signal_name: str,
    threshold_pct: float = 90,
    horizon_s: int = 10,
):
    """
    Simulate taker strategy: enter when signal > P(threshold), exit at horizon.
    Measures realized PnL after fees.
    """
    fret_col = f"{target_symbol}_fret_{horizon_s}s"
    if fret_col not in df.columns:
        return None

    mask = df[signal_col].notna() & df[fret_col].notna()
    sig = df.loc[mask, signal_col].values
    ret = df.loc[mask, fret_col].values

    taker_rt = TAKER_FEE_RT[target_symbol]

    # Long when signal > P(threshold)
    p_hi = np.percentile(sig, threshold_pct)
    p_lo = np.percentile(sig, 100 - threshold_pct)

    long_mask = sig > p_hi
    short_mask = sig < p_lo
    n_long = long_mask.sum()
    n_short = short_mask.sum()

    if n_long < 100 or n_short < 100:
        return None

    long_pnl = ret[long_mask] - taker_rt
    short_pnl = -ret[short_mask] - taker_rt

    combined = np.concatenate([long_pnl, short_pnl])

    total_seconds = len(df)
    trades_per_day = (n_long + n_short) / total_seconds * 86400
    daily_pnl_bps = combined.mean() * trades_per_day

    print(f"\n  --- Taker Simulation: {signal_name} → {target_symbol} ---")
    print(f"  Threshold: P{threshold_pct:.0f}/P{100-threshold_pct:.0f} | Horizon: {horizon_s}s")
    print(f"  Longs: {n_long:,} | Shorts: {n_short:,} | Total: {n_long+n_short:,}")
    print(f"  Long PnL (after fees): {long_pnl.mean():+.2f} bps (std={long_pnl.std():.2f})")
    print(f"  Short PnL (after fees): {short_pnl.mean():+.2f} bps (std={short_pnl.std():.2f})")
    print(f"  Combined mean: {combined.mean():+.2f} bps")
    print(f"  Win rate: {(combined > 0).mean():.1%}")
    print(f"  Trades/day: {trades_per_day:.0f}")
    print(f"  Daily edge: {daily_pnl_bps:+.1f} bps ({daily_pnl_bps/10000*100:+.4f}%)")
    print(f"  Sharpe (annualized, rough): {combined.mean() / (combined.std() + 1e-12) * np.sqrt(trades_per_day * 365):.2f}")

    return {
        "signal": signal_name,
        "target": target_symbol,
        "n_trades": n_long + n_short,
        "mean_pnl_bps": combined.mean(),
        "win_rate": (combined > 0).mean(),
        "trades_per_day": trades_per_day,
        "daily_edge_bps": daily_pnl_bps,
        "sharpe": combined.mean() / (combined.std() + 1e-12) * np.sqrt(trades_per_day * 365),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-symbol edge quantification")
    parser.add_argument("--data-dir", default="../data/features", help="Data directory")
    parser.add_argument("--days", type=int, default=None, help="Max days to load")
    args = parser.parse_args()

    print("=" * 70)
    print("CROSS-SYMBOL EDGE QUANTIFICATION")
    print("=" * 70)

    # Load and align data
    df = load_aligned(args.data_dir, max_days=args.days)

    # Compute returns at multiple horizons for all symbols
    horizons = [1, 5, 10, 30, 60, 300]
    for sym in ["BTC", "ETH", "SOL"]:
        df = compute_returns(df, sym, horizons)

    # ============================================================
    # Signal 1: BTC→SOL lead-lag
    # BTC return over last N seconds predicts SOL forward return
    # ============================================================
    print("\n" + "=" * 70)
    print("SIGNAL 1: BTC → SOL LEAD-LAG")
    print("=" * 70)

    for lookback in [1, 5, 10]:
        btc_mid = df["BTC_raw_midprice"]
        df[f"btc_ret_{lookback}s"] = (btc_mid / btc_mid.shift(lookback) - 1) * 10000

    results_sol = []
    for lb in [1, 5, 10]:
        for fh in [1, 5, 10, 30]:
            r = signal_edge_analysis(
                df,
                signal_col=f"btc_ret_{lb}s",
                return_col=f"SOL_fret_{fh}s",
                signal_name=f"BTC_ret({lb}s)→SOL@{fh}s",
            )
            results_sol.append(r)

    # ============================================================
    # Signal 2: BTC→ETH lead-lag
    # ============================================================
    print("\n" + "=" * 70)
    print("SIGNAL 2: BTC → ETH LEAD-LAG")
    print("=" * 70)

    results_eth = []
    for lb in [1, 5]:
        for fh in [1, 5, 10]:
            r = signal_edge_analysis(
                df,
                signal_col=f"btc_ret_{lb}s",
                return_col=f"ETH_fret_{fh}s",
                signal_name=f"BTC_ret({lb}s)→ETH@{fh}s",
            )
            results_eth.append(r)

    # ============================================================
    # Signal 3: BTC-ETH spread convergence
    # ============================================================
    print("\n" + "=" * 70)
    print("SIGNAL 3: BTC-ETH SPREAD CONVERGENCE")
    print("=" * 70)

    # Compute log-price spread and z-score
    btc_log = np.log(df["BTC_raw_midprice"])
    eth_log = np.log(df["ETH_raw_midprice"])
    spread = btc_log - eth_log
    for window in [60, 300, 600]:
        spread_mean = spread.rolling(window, min_periods=window // 2).mean()
        spread_std = spread.rolling(window, min_periods=window // 2).std()
        df[f"spread_z_{window}s"] = (spread - spread_mean) / (spread_std + 1e-12)

    # When spread_z > 0: BTC expensive relative to ETH → short BTC, long ETH
    # So signal = -spread_z should predict ETH forward return positively
    results_pairs = []
    for w in [60, 300, 600]:
        df[f"neg_spread_z_{w}s"] = -df[f"spread_z_{w}s"]
        for fh in [10, 30, 60, 300]:
            r = signal_edge_analysis(
                df,
                signal_col=f"neg_spread_z_{w}s",
                return_col=f"ETH_fret_{fh}s",
                signal_name=f"spread_z({w}s)→ETH@{fh}s",
            )
            results_pairs.append(r)

    # ============================================================
    # Signal 4: Premium reversion
    # ============================================================
    print("\n" + "=" * 70)
    print("SIGNAL 4: PREMIUM REVERSION")
    print("=" * 70)

    # -premium_bps → BTC forward return (mean reversion)
    if "BTC_ctx_premium_bps" in df.columns:
        df["neg_premium"] = -df["BTC_ctx_premium_bps"]
        results_prem = []
        for fh in [10, 30, 60, 300]:
            r = signal_edge_analysis(
                df,
                signal_col="neg_premium",
                return_col=f"BTC_fret_{fh}s",
                signal_name=f"-premium→BTC@{fh}s",
            )
            results_prem.append(r)
    else:
        print("  ctx_premium_bps not available")
        results_prem = []

    # ============================================================
    # COST VIABILITY SUMMARY
    # ============================================================
    all_sol = [r for r in results_sol if r is not None]
    all_eth = [r for r in results_eth + results_pairs if r is not None]
    all_btc = [r for r in results_prem if r is not None]

    if all_sol:
        cost_viability(all_sol, "SOL")
    if all_eth:
        cost_viability(all_eth, "ETH")
    if all_btc:
        cost_viability(all_btc, "BTC")

    # ============================================================
    # TAKER SIMULATIONS — best signals
    # ============================================================
    print("\n" + "=" * 70)
    print("TAKER STRATEGY SIMULATIONS")
    print("=" * 70)

    sim_results = []

    # Best lead-lag signals
    for lb, fh, target in [(1, 5, "SOL"), (1, 10, "SOL"), (5, 10, "SOL"),
                            (1, 5, "ETH"), (1, 10, "ETH")]:
        sig_col = f"btc_ret_{lb}s"
        for thresh in [80, 90, 95]:
            r = taker_simulation(
                df, sig_col, f"{target}_raw_midprice", target,
                f"BTC_ret({lb}s)→{target}@{fh}s[P{thresh}]",
                threshold_pct=thresh, horizon_s=fh,
            )
            if r:
                sim_results.append(r)

    # Print summary table
    if sim_results:
        print(f"\n{'='*80}")
        print("SIMULATION SUMMARY")
        print(f"{'='*80}")
        print(f"{'Signal':<40} {'PnL':>7} {'Win%':>6} {'#/day':>6} {'Daily':>8} {'Sharpe':>7}")
        print("-" * 80)
        for r in sorted(sim_results, key=lambda x: x["daily_edge_bps"], reverse=True):
            print(
                f"  {r['signal']:<38} "
                f"{r['mean_pnl_bps']:>+6.2f} "
                f"{r['win_rate']:>5.1%} "
                f"{r['trades_per_day']:>5.0f} "
                f"{r['daily_edge_bps']:>+7.1f} "
                f"{r['sharpe']:>6.2f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
