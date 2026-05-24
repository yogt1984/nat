"""
Cross-Symbol Edge v2: Extended horizon + volatility scaling analysis.

Key question: BTC→SOL IC=0.27 produces only 0.78 bps edge at 10s.
Can we find horizons where return variance is large enough that
IC × σ > fee (7 bps)?

Required: IC × σ_return > 7 bps
At 10s: σ=3 bps, need IC > 2.33 (impossible, max is 1.0)
At 60s: σ≈8 bps, need IC > 0.875 (still impossible)
At 300s: σ≈18 bps, need IC > 0.39

So the question: does BTC lead-lag IC persist at 5-minute horizons?

Also tests:
- Regime conditioning (low entropy = cleaner signal)
- Volume conditioning (high flow = faster propagation)
- Combined BTC+microstructure signals

Usage:
    cd scripts && python -m analysis.cross_symbol_edge_v2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cluster_pipeline.loader import load_parquet

TAKER_FEE_RT = 7.0  # bps


def load_aligned(data_dir: str, max_days: int | None = None) -> pd.DataFrame:
    """Load all symbols, align to 1-second buckets."""
    cols = [
        "timestamp_ns", "symbol", "raw_midprice",
        "imbalance_qty_l1", "flow_volume_1s", "flow_volume_5s",
        "ent_book_shape", "vol_returns_1m",
        "ctx_funding_rate", "ctx_premium_bps",
    ]

    kwargs = {}
    if max_days:
        dates = sorted(Path(data_dir).iterdir())
        dates = [d for d in dates if d.is_dir() and not d.name.endswith("-clean")]
        if len(dates) > max_days:
            kwargs["start_date"] = dates[-max_days].name

    df = load_parquet(data_dir, columns=cols, **kwargs)
    print(f"Loaded {len(df):,} rows")

    df["ts_s"] = df["timestamp_ns"] // 1_000_000_000

    aligned = {}
    for sym in ["BTC", "ETH", "SOL"]:
        sdf = df[df["symbol"] == sym].copy()
        sdf = sdf.groupby("ts_s").last().reset_index().set_index("ts_s")
        for c in sdf.columns:
            if c != "symbol":
                aligned[f"{sym}_{c}"] = sdf[c]

    result = pd.DataFrame(aligned)
    result = result.dropna(subset=["BTC_raw_midprice", "SOL_raw_midprice"])
    result = result.sort_index()
    print(f"Aligned: {len(result):,} seconds")
    return result


def main():
    data_dir = "../data/features"

    print("=" * 70)
    print("CROSS-SYMBOL EDGE V2: HORIZON SCALING + REGIME CONDITIONING")
    print("=" * 70)

    df = load_aligned(data_dir, max_days=8)

    # ================================================================
    # Part 1: IC decay across horizons
    # Does BTC→SOL predictability persist at longer horizons?
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 1: IC DECAY ACROSS HORIZONS")
    print("=" * 70)

    btc_mid = df["BTC_raw_midprice"]
    sol_mid = df["SOL_raw_midprice"]
    eth_mid = df["ETH_raw_midprice"]

    # BTC trailing returns
    for lb in [1, 5, 10, 30]:
        df[f"btc_ret_{lb}s"] = (btc_mid / btc_mid.shift(lb) - 1) * 10000

    # Forward returns at extended horizons
    horizons = [1, 5, 10, 30, 60, 120, 300, 600]
    for h in horizons:
        df[f"SOL_fret_{h}s"] = (sol_mid.shift(-h) / sol_mid - 1) * 10000
        df[f"ETH_fret_{h}s"] = (eth_mid.shift(-h) / eth_mid - 1) * 10000

    print(f"\n{'Lookback':>10} {'Target':>6} {'Horizon':>8} {'IC':>8} {'ReturnSD':>10} "
          f"{'Edge_P90':>10} {'Edge_P10':>10} {'NetEdge':>10} {'Viable':>8}")
    print("-" * 95)

    viable_signals = []

    for lb in [1, 5, 10]:
        sig_col = f"btc_ret_{lb}s"
        for target in ["SOL", "ETH"]:
            for h in horizons:
                fret_col = f"{target}_fret_{h}s"
                mask = df[sig_col].notna() & df[fret_col].notna()
                sig = df.loc[mask, sig_col].values
                ret = df.loc[mask, fret_col].values

                if len(sig) < 1000:
                    continue

                ic = stats.spearmanr(sig, ret).statistic
                ret_sd = np.std(ret)

                # Edge at P90/P10
                p90 = np.percentile(sig, 90)
                p10 = np.percentile(sig, 10)
                edge_long = ret[sig > p90].mean() if (sig > p90).sum() > 100 else np.nan
                edge_short = -ret[sig < p10].mean() if (sig < p10).sum() > 100 else np.nan

                # Signed net edge (positive = profitable direction)
                best_edge = max(edge_long, edge_short) if np.isfinite(edge_long) and np.isfinite(edge_short) else 0
                net = best_edge - TAKER_FEE_RT
                viable = "YES" if net > 0 else ""

                if net > -2:  # Show anything remotely close
                    viable_signals.append({
                        "lb": lb, "target": target, "h": h,
                        "ic": ic, "edge": best_edge, "net": net,
                    })

                print(f"{lb:>8}s {target:>6} {h:>6}s {ic:>+8.4f} {ret_sd:>9.2f} "
                      f"{edge_long:>+9.2f} {edge_short:>+9.2f} {net:>+9.2f} {viable:>8}")

    # ================================================================
    # Part 2: Regime-conditioned lead-lag
    # Does entropy gating improve the cross-symbol signal?
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 2: REGIME-CONDITIONED LEAD-LAG")
    print("=" * 70)

    # Use SOL's book entropy for regime gating
    sol_ent = df["SOL_ent_book_shape"]
    btc_ent = df["BTC_ent_book_shape"]

    for ent_col, ent_name in [(sol_ent, "SOL_ent"), (btc_ent, "BTC_ent")]:
        valid = ent_col.dropna()
        if len(valid) < 1000:
            continue

        p30 = valid.quantile(0.30)
        p70 = valid.quantile(0.70)

        low_ent = ent_col < p30
        high_ent = ent_col > p70

        print(f"\n  Gating on {ent_name} (P30={p30:.4f}, P70={p70:.4f})")
        print(f"  {'Signal':>25} {'Full IC':>8} {'Low-Ent IC':>11} {'High-Ent IC':>12} {'Boost':>7}")
        print("  " + "-" * 70)

        for lb, target, h in [(1, "SOL", 10), (1, "SOL", 30), (1, "SOL", 60),
                               (1, "ETH", 10), (5, "SOL", 30), (5, "SOL", 60)]:
            sig_col = f"btc_ret_{lb}s"
            fret_col = f"{target}_fret_{h}s"

            mask_full = df[sig_col].notna() & df[fret_col].notna()
            mask_low = mask_full & low_ent
            mask_high = mask_full & high_ent

            for m, name in [(mask_full, "full"), (mask_low, "low"), (mask_high, "high")]:
                if m.sum() < 500:
                    continue

            ic_full = stats.spearmanr(
                df.loc[mask_full, sig_col].values,
                df.loc[mask_full, fret_col].values
            ).statistic

            ic_low = stats.spearmanr(
                df.loc[mask_low, sig_col].values,
                df.loc[mask_low, fret_col].values
            ).statistic if mask_low.sum() > 500 else np.nan

            ic_high = stats.spearmanr(
                df.loc[mask_high, sig_col].values,
                df.loc[mask_high, fret_col].values
            ).statistic if mask_high.sum() > 500 else np.nan

            boost = ic_low / ic_full if ic_full != 0 and np.isfinite(ic_low) else np.nan

            sig_name = f"BTC_ret({lb}s)→{target}@{h}s"
            print(f"  {sig_name:>25} {ic_full:>+8.4f} {ic_low:>+10.4f} {ic_high:>+11.4f} {boost:>6.2f}x")

    # ================================================================
    # Part 3: Volume-conditioned lead-lag
    # Does high volume accelerate cross-symbol propagation?
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 3: VOLUME-CONDITIONED LEAD-LAG")
    print("=" * 70)

    btc_vol = df["BTC_flow_volume_5s"]
    valid_vol = btc_vol.dropna()
    if len(valid_vol) > 1000:
        p75_vol = valid_vol.quantile(0.75)
        p25_vol = valid_vol.quantile(0.25)
        high_vol = btc_vol > p75_vol
        low_vol = btc_vol < p25_vol

        print(f"  BTC flow_volume_5s: P25={p25_vol:.0f}, P75={p75_vol:.0f}")
        print(f"  {'Signal':>25} {'Full IC':>8} {'HighVol IC':>11} {'LowVol IC':>11} {'Edge(HV)':>10}")
        print("  " + "-" * 70)

        for lb, target, h in [(1, "SOL", 5), (1, "SOL", 10), (1, "SOL", 30),
                               (1, "SOL", 60), (5, "SOL", 30), (5, "SOL", 60)]:
            sig_col = f"btc_ret_{lb}s"
            fret_col = f"{target}_fret_{h}s"

            mask = df[sig_col].notna() & df[fret_col].notna()
            mask_hv = mask & high_vol
            mask_lv = mask & low_vol

            ic_full = stats.spearmanr(
                df.loc[mask, sig_col].values, df.loc[mask, fret_col].values
            ).statistic

            ic_hv = stats.spearmanr(
                df.loc[mask_hv, sig_col].values, df.loc[mask_hv, fret_col].values
            ).statistic if mask_hv.sum() > 500 else np.nan

            ic_lv = stats.spearmanr(
                df.loc[mask_lv, sig_col].values, df.loc[mask_lv, fret_col].values
            ).statistic if mask_lv.sum() > 500 else np.nan

            # Edge at P90 in high-vol regime
            if mask_hv.sum() > 500:
                s = df.loc[mask_hv, sig_col].values
                r = df.loc[mask_hv, fret_col].values
                p90 = np.percentile(s, 90)
                edge_hv = r[s > p90].mean() if (s > p90).sum() > 50 else np.nan
            else:
                edge_hv = np.nan

            name = f"BTC_ret({lb}s)→{target}@{h}s"
            print(f"  {name:>25} {ic_full:>+8.4f} {ic_hv:>+10.4f} {ic_lv:>+10.4f} {edge_hv:>+9.2f}")

    # ================================================================
    # Part 4: The arithmetic of tradeability
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 4: TRADEABILITY ARITHMETIC")
    print("=" * 70)

    print("\n  Return volatility (σ) by horizon:")
    for h in horizons:
        for target in ["SOL", "ETH", "BTC"]:
            col = f"{target}_fret_{h}s"
            if col in df.columns:
                vals = df[col].dropna().values
                sigma = np.std(vals)
                # Required IC to break even: fee / sigma
                required_ic = TAKER_FEE_RT / sigma if sigma > 0 else np.inf
                print(f"    {target}@{h:>4}s: σ={sigma:>6.2f} bps | required IC > {required_ic:.3f}")

    print(f"\n  Conclusion: at what horizon can IC=0.27 break even?")
    for h in horizons:
        col = f"SOL_fret_{h}s"
        if col in df.columns:
            sigma = df[col].dropna().std()
            edge = 0.27 * sigma  # IC × σ approximation
            net = edge - TAKER_FEE_RT
            print(f"    SOL@{h:>4}s: σ={sigma:.2f}, edge≈{edge:.2f}, net={net:+.2f} bps {'← BREAK EVEN' if net > 0 else ''}")

    # ================================================================
    # Part 5: Cross-venue arbitrage sizing
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 5: WHAT FEE LEVEL WOULD MAKE THIS WORK?")
    print("=" * 70)

    # Best signal: BTC_ret(1s) → SOL@10s
    sig_col = "btc_ret_1s"
    fret_col = "SOL_fret_10s"
    mask = df[sig_col].notna() & df[fret_col].notna()
    sig = df.loc[mask, sig_col].values
    ret = df.loc[mask, fret_col].values

    p90 = np.percentile(sig, 90)
    p10 = np.percentile(sig, 10)
    edge_long = ret[sig > p90].mean()
    edge_short = -ret[sig < p10].mean()
    best_edge = max(edge_long, edge_short)

    print(f"\n  Best signal: BTC_ret(1s)→SOL@10s")
    print(f"  P90 long edge: {edge_long:+.3f} bps")
    print(f"  P10 short edge: {edge_short:+.3f} bps")
    print(f"  Best directional edge: {best_edge:.3f} bps")
    print(f"")
    print(f"  Required RT fee for profitability: < {best_edge:.2f} bps")
    print(f"  Current Hyperliquid taker RT: 7.00 bps")
    print(f"  Gap: {TAKER_FEE_RT / best_edge:.1f}x")
    print(f"")
    print(f"  Venues with lower fees:")
    print(f"    Binance VIP9 maker: -0.09 bps (rebate)")
    print(f"    Binance VIP9 taker: 1.7 bps")
    print(f"    Binance VIP9 RT:    1.61 bps")
    print(f"    dYdX maker:         0.0 bps (free)")
    print(f"    dYdX taker:         2.0 bps")
    print(f"    dYdX RT:            2.0 bps")
    print(f"")
    bin_net = best_edge - 1.61
    dydx_net = best_edge - 2.0
    print(f"  At Binance VIP9: net = {bin_net:+.2f} bps {'← MARGINAL' if abs(bin_net) < 0.5 else ''}")
    print(f"  At dYdX:         net = {dydx_net:+.2f} bps")
    print(f"")

    # What about regime-conditioned edge on a low-fee venue?
    if "SOL_ent_book_shape" in df.columns:
        sol_ent = df["SOL_ent_book_shape"]
        valid = sol_ent.dropna()
        if len(valid) > 1000:
            p30 = valid.quantile(0.30)
            low_ent = sol_ent < p30

            mask_rg = mask & low_ent
            if mask_rg.sum() > 500:
                sig_rg = df.loc[mask_rg, sig_col].values
                ret_rg = df.loc[mask_rg, fret_col].values
                p90_rg = np.percentile(sig_rg, 90)
                edge_rg = ret_rg[sig_rg > p90_rg].mean()
                ic_rg = stats.spearmanr(sig_rg, ret_rg).statistic

                print(f"  Regime-gated (SOL ent < P30, n={mask_rg.sum():,}):")
                print(f"    IC: {ic_rg:+.4f}")
                print(f"    P90 long edge: {edge_rg:+.3f} bps")
                print(f"    At Binance VIP9: net = {edge_rg - 1.61:+.2f} bps")
                print(f"    At dYdX:         net = {edge_rg - 2.0:+.2f} bps")

    print("\nDone.")


if __name__ == "__main__":
    main()
