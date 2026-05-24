"""
Rare Event Edge Analysis

Identifies events where price dislocations are large enough (>7 bps)
to overcome Hyperliquid taker fees. Tests each event type's:
  - Frequency (how often it fires)
  - Dislocation magnitude (how many bps of move)
  - Post-event predictability (reversion or continuation)
  - Net edge after 7 bps RT fees

Event types tested:
  1. Jump detection (Lee-Mykland) — flash crashes, large ticks
  2. VPIN spikes — sudden toxicity increase
  3. Funding rate extremes — mean-reversion at settlement
  4. Entropy collapse — order book regime transition
  5. Cross-symbol dislocation — BTC moves, alts lag
  6. Volume explosion — sudden activity spike
  7. Premium/basis blowout — perp vs spot divergence

Usage:
    cd scripts && python -m analysis.rare_events
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cluster_pipeline.loader import load_parquet


def load_data(data_dir: str, max_days: int | None = None) -> pd.DataFrame:
    cols = [
        "timestamp_ns", "symbol", "raw_midprice", "raw_spread_bps",
        "imbalance_qty_l1", "flow_volume_1s", "flow_volume_5s",
        "flow_count_5s", "flow_intensity",
        "vol_returns_1m", "vol_zscore", "vol_parkinson_5m",
        "ent_book_shape", "ent_tick_5s",
        "toxic_vpin_50", "toxic_adverse_selection",
        "ctx_funding_rate", "ctx_funding_zscore", "ctx_premium_bps",
        "trend_momentum_300",
    ]

    kwargs = {}
    if max_days:
        dates = sorted(Path(data_dir).iterdir())
        dates = [d for d in dates if d.is_dir() and not d.name.endswith("-clean")]
        if len(dates) > max_days:
            kwargs["start_date"] = dates[-max_days].name

    df = load_parquet(data_dir, columns=cols, **kwargs)
    return df


def analyze_event(
    df: pd.DataFrame,
    event_mask: pd.Series,
    event_name: str,
    mid_col: str = "raw_midprice",
    horizons_s: list[int] = [1, 5, 10, 30, 60, 300],
):
    """
    Analyze post-event returns at multiple horizons.
    Returns a summary dict.
    """
    n_events = event_mask.sum()
    n_total = len(df)

    if n_events < 10:
        print(f"  {event_name}: only {n_events} events, skipping")
        return None

    mid = df[mid_col].values
    freq_pct = n_events / n_total * 100

    print(f"\n  === {event_name} ===")
    print(f"  Events: {n_events:,} / {n_total:,} ({freq_pct:.2f}%)")

    results = {"name": event_name, "n_events": n_events, "freq_pct": freq_pct}
    horizons_data = []

    event_indices = np.where(event_mask.values)[0]

    for h in horizons_s:
        # Forward return in bps
        fwd = np.full(len(mid), np.nan)
        valid = np.arange(len(mid) - h)
        fwd[valid] = (mid[valid + h] / mid[valid] - 1) * 10000

        event_ret = fwd[event_indices]
        event_ret = event_ret[np.isfinite(event_ret)]

        if len(event_ret) < 10:
            continue

        # Non-event return for comparison
        non_event = np.where(~event_mask.values)[0]
        non_event = non_event[non_event < len(mid) - h]
        non_event_ret = fwd[non_event]
        non_event_ret = non_event_ret[np.isfinite(non_event_ret)]

        # Stats
        mean_ret = np.mean(event_ret)
        abs_mean = np.mean(np.abs(event_ret))
        std_ret = np.std(event_ret)
        baseline_abs = np.mean(np.abs(non_event_ret))

        # Is the event directionally predictable?
        # Check: do events with positive imbalance lead to positive returns?
        up_frac = np.mean(event_ret > 0)
        down_frac = np.mean(event_ret < 0)
        directional_bias = abs(up_frac - 0.5)

        # Reversion: does the direction at event reverse?
        # Use trailing 1s return as "event direction"
        if h > 1:
            trail = np.full(len(mid), np.nan)
            valid_t = np.arange(1, len(mid))
            trail[valid_t] = (mid[valid_t] / mid[valid_t - 1] - 1) * 10000
            trail_at_event = trail[event_indices]
            ml = min(len(trail_at_event), len(event_ret))
            t_ev = trail_at_event[:ml]
            e_ev = event_ret[:ml]
            valid_both = np.isfinite(t_ev) & np.isfinite(e_ev)
            if valid_both.sum() > 10:
                reversion_frac = np.mean(
                    np.sign(t_ev[valid_both]) != np.sign(e_ev[valid_both])
                )
            else:
                reversion_frac = 0.5
        else:
            reversion_frac = 0.5

        # Edge: if we trade the reversion at this horizon
        # Assume: fade the trailing move direction
        if h > 1:
            min_len = min(len(trail_at_event), len(event_ret))
            trail_at_event = trail_at_event[:min_len]
            event_ret_trimmed = event_ret[:min_len]
            valid_mask = np.isfinite(trail_at_event) & np.isfinite(event_ret_trimmed)
            if valid_mask.sum() > 10:
                # Reversion PnL: -sign(trailing_move) × forward_return
                reversion_pnl = -np.sign(trail_at_event[valid_mask]) * event_ret_trimmed[valid_mask]
                reversion_edge = np.mean(reversion_pnl)
                reversion_edge_net = reversion_edge - 7.0
            else:
                reversion_edge = 0
                reversion_edge_net = -7.0
        else:
            reversion_edge = abs_mean / 2  # Rough approximation
            reversion_edge_net = reversion_edge - 7.0

        viable = "YES" if reversion_edge_net > 0 else ""

        horizons_data.append({
            "horizon_s": h,
            "mean_ret": mean_ret,
            "abs_mean": abs_mean,
            "std": std_ret,
            "baseline_abs": baseline_abs,
            "amplification": abs_mean / baseline_abs if baseline_abs > 0 else 0,
            "reversion_frac": reversion_frac,
            "reversion_edge": reversion_edge,
            "net_edge": reversion_edge_net,
            "viable": reversion_edge_net > 0,
        })

        print(f"    @{h:>4}s: |ret|={abs_mean:>6.2f}bps (vs {baseline_abs:.2f} baseline, "
              f"{abs_mean/baseline_abs:.1f}x) | revert={reversion_frac:.0%} | "
              f"edge={reversion_edge:>+.2f} | net={reversion_edge_net:>+.2f} {viable}")

    results["horizons"] = horizons_data
    return results


def main():
    data_dir = "../data/features"

    print("=" * 80)
    print("RARE EVENT EDGE ANALYSIS")
    print("=" * 80)

    # Load all symbols at 1-second resolution
    df_raw = load_data(data_dir, max_days=8)
    print(f"Loaded {len(df_raw):,} rows")

    all_results = []

    for sym in ["BTC", "ETH", "SOL"]:
        df = df_raw[df_raw["symbol"] == sym].copy()
        df["ts_s"] = df["timestamp_ns"] // 1_000_000_000
        df = df.groupby("ts_s").last().reset_index()
        df = df.sort_values("ts_s").reset_index(drop=True)

        print(f"\n{'='*80}")
        print(f"SYMBOL: {sym} ({len(df):,} seconds)")
        print(f"{'='*80}")

        mid = df["raw_midprice"].values

        # Precompute trailing returns
        df["ret_1s"] = pd.Series(mid).pct_change() * 10000
        df["ret_5s"] = (pd.Series(mid) / pd.Series(mid).shift(5) - 1) * 10000
        df["ret_30s"] = (pd.Series(mid) / pd.Series(mid).shift(30) - 1) * 10000

        # ============================================================
        # Event 1: Large tick moves (proxy for jumps)
        # ============================================================
        ret_1s = df["ret_1s"].dropna()
        p99 = ret_1s.abs().quantile(0.99)
        p999 = ret_1s.abs().quantile(0.999)
        p9999 = ret_1s.abs().quantile(0.9999)

        print(f"\n  1s return distribution: P99={p99:.2f}, P99.9={p999:.2f}, P99.99={p9999:.2f} bps")

        for thresh_name, thresh in [("P99 (1s)", p99), ("P99.9 (1s)", p999), ("P99.99 (1s)", p9999)]:
            mask = df["ret_1s"].abs() > thresh
            r = analyze_event(df, mask, f"Jump {thresh_name} |ret|>{thresh:.1f}bps")
            if r:
                all_results.append(r)

        # ============================================================
        # Event 2: VPIN spike (toxicity surge)
        # ============================================================
        vpin = df["toxic_vpin_50"]
        valid_vpin = vpin.dropna()
        if len(valid_vpin) > 1000:
            p90_vpin = valid_vpin.quantile(0.90)
            p95_vpin = valid_vpin.quantile(0.95)
            p99_vpin = valid_vpin.quantile(0.99)

            for pname, pval in [("P90", p90_vpin), ("P95", p95_vpin), ("P99", p99_vpin)]:
                mask = vpin > pval
                r = analyze_event(df, mask, f"VPIN spike {pname} >{pval:.3f}")
                if r:
                    all_results.append(r)

        # ============================================================
        # Event 3: Entropy collapse (book structure transition)
        # ============================================================
        ent = df["ent_book_shape"]
        valid_ent = ent.dropna()
        if len(valid_ent) > 1000:
            p5_ent = valid_ent.quantile(0.05)
            p10_ent = valid_ent.quantile(0.10)

            # Entropy drop: current ent below P5 AND was above P30 recently
            ent_ma = ent.rolling(60, min_periods=10).mean()
            p30_ent = valid_ent.quantile(0.30)

            for pname, pval in [("P5", p5_ent), ("P10", p10_ent)]:
                mask = (ent < pval) & (ent_ma > p30_ent)
                r = analyze_event(df, mask, f"Entropy collapse {pname} <{pval:.3f}")
                if r:
                    all_results.append(r)

        # ============================================================
        # Event 4: Funding rate extreme
        # ============================================================
        fz = df["ctx_funding_zscore"]
        valid_fz = fz.dropna()
        if len(valid_fz) > 1000:
            for thresh in [2.0, 3.0]:
                mask_pos = fz > thresh
                mask_neg = fz < -thresh
                mask_either = fz.abs() > thresh
                r = analyze_event(df, mask_either, f"Funding |z|>{thresh}")
                if r:
                    all_results.append(r)

        # ============================================================
        # Event 5: Volume explosion
        # ============================================================
        vol = df["flow_volume_5s"]
        valid_vol = vol.dropna()
        if len(valid_vol) > 1000:
            p95_vol = valid_vol.quantile(0.95)
            p99_vol = valid_vol.quantile(0.99)

            for pname, pval in [("P95", p95_vol), ("P99", p99_vol)]:
                mask = vol > pval
                r = analyze_event(df, mask, f"Volume explosion {pname}")
                if r:
                    all_results.append(r)

        # ============================================================
        # Event 6: Premium/basis blowout
        # ============================================================
        prem = df["ctx_premium_bps"]
        valid_prem = prem.dropna()
        if len(valid_prem) > 1000:
            p95_prem = valid_prem.abs().quantile(0.95)
            p99_prem = valid_prem.abs().quantile(0.99)

            for pname, pval in [("P95", p95_prem), ("P99", p99_prem)]:
                mask = prem.abs() > pval
                r = analyze_event(df, mask, f"Premium |bps|>{pval:.1f} ({pname})")
                if r:
                    all_results.append(r)

        # ============================================================
        # Event 7: Volatility spike (vol_zscore extreme)
        # ============================================================
        vz = df["vol_zscore"]
        valid_vz = vz.dropna()
        if len(valid_vz) > 1000:
            for thresh in [2.0, 3.0]:
                mask = vz > thresh
                r = analyze_event(df, mask, f"Vol zscore > {thresh}")
                if r:
                    all_results.append(r)

        # ============================================================
        # Event 8: Combined: jump + low entropy (structured liquidation)
        # ============================================================
        if len(valid_ent) > 1000:
            jump_mask = df["ret_1s"].abs() > p99
            low_ent = ent < p10_ent
            combined = jump_mask & low_ent
            r = analyze_event(df, combined, "Jump P99 + Low Entropy P10")
            if r:
                all_results.append(r)

        # ============================================================
        # Event 9: VPIN spike + large move (informed flow event)
        # ============================================================
        if len(valid_vpin) > 1000:
            vpin_high = vpin > p90_vpin
            big_move = df["ret_5s"].abs() > df["ret_5s"].abs().quantile(0.95)
            combined2 = vpin_high & big_move
            r = analyze_event(df, combined2, "VPIN P90 + Large 5s move P95")
            if r:
                all_results.append(r)

    # ============================================================
    # CROSS-SYMBOL DISLOCATION (needs multi-symbol alignment)
    # ============================================================
    print(f"\n{'='*80}")
    print("CROSS-SYMBOL DISLOCATION EVENTS")
    print(f"{'='*80}")

    # Align symbols
    aligned = {}
    for sym in ["BTC", "ETH", "SOL"]:
        sdf = df_raw[df_raw["symbol"] == sym].copy()
        sdf["ts_s"] = sdf["timestamp_ns"] // 1_000_000_000
        sdf = sdf.groupby("ts_s").last().reset_index().set_index("ts_s")
        aligned[sym] = sdf["raw_midprice"]

    adf = pd.DataFrame(aligned).dropna().sort_index()
    print(f"  Aligned: {len(adf):,} seconds")

    if len(adf) > 1000:
        for sym in ["BTC", "ETH", "SOL"]:
            adf[f"{sym}_ret_1s"] = adf[sym].pct_change() * 10000
            adf[f"{sym}_ret_5s"] = (adf[sym] / adf[sym].shift(5) - 1) * 10000

        # Event: BTC large move but SOL hasn't followed
        btc_big = adf["BTC_ret_1s"].abs() > adf["BTC_ret_1s"].abs().quantile(0.99)
        sol_small = adf["SOL_ret_1s"].abs() < adf["SOL_ret_1s"].abs().quantile(0.50)

        dislocation_mask = btc_big & sol_small
        print(f"\n  BTC jump P99 + SOL flat: {dislocation_mask.sum():,} events")

        if dislocation_mask.sum() > 10:
            # SOL forward returns at these events
            for h in [1, 5, 10, 30, 60]:
                adf[f"SOL_fwd_{h}s"] = (adf["SOL"].shift(-h) / adf["SOL"] - 1) * 10000

            event_idx = np.where(dislocation_mask.values)[0]
            btc_dir = np.sign(adf["BTC_ret_1s"].values[event_idx])

            print(f"  {'Horizon':>8} {'SOL follow':>10} {'SOL_abs':>8} {'Follow_edge':>12} {'Net':>8} {'Viable':>7}")
            print(f"  {'-'*60}")

            for h in [1, 5, 10, 30, 60]:
                col = f"SOL_fwd_{h}s"
                sol_fwd = adf[col].values[event_idx]
                valid = np.isfinite(sol_fwd) & np.isfinite(btc_dir)

                if valid.sum() < 10:
                    continue

                # Follow trade: go in BTC's direction on SOL
                follow_pnl = btc_dir[valid] * sol_fwd[valid]
                follow_edge = np.mean(follow_pnl)
                sol_abs = np.mean(np.abs(sol_fwd[valid]))
                net = follow_edge - 7.0
                viable = "YES" if net > 0 else ""

                print(f"  {h:>6}s {follow_edge:>+9.2f}bps {sol_abs:>7.2f} {follow_edge:>+11.2f} {net:>+7.2f} {viable:>7}")

        # Same for ETH
        eth_small = adf["ETH_ret_1s"].abs() < adf["ETH_ret_1s"].abs().quantile(0.50)
        disloc_eth = btc_big & eth_small
        print(f"\n  BTC jump P99 + ETH flat: {disloc_eth.sum():,} events")

        if disloc_eth.sum() > 10:
            for h in [1, 5, 10, 30, 60]:
                adf[f"ETH_fwd_{h}s"] = (adf["ETH"].shift(-h) / adf["ETH"] - 1) * 10000

            event_idx = np.where(disloc_eth.values)[0]
            btc_dir = np.sign(adf["BTC_ret_1s"].values[event_idx])

            print(f"  {'Horizon':>8} {'ETH follow':>10} {'ETH_abs':>8} {'Follow_edge':>12} {'Net':>8} {'Viable':>7}")
            print(f"  {'-'*60}")

            for h in [1, 5, 10, 30, 60]:
                col = f"ETH_fwd_{h}s"
                eth_fwd = adf[col].values[event_idx]
                valid = np.isfinite(eth_fwd) & np.isfinite(btc_dir)

                if valid.sum() < 10:
                    continue

                follow_pnl = btc_dir[valid] * eth_fwd[valid]
                follow_edge = np.mean(follow_pnl)
                eth_abs = np.mean(np.abs(eth_fwd[valid]))
                net = follow_edge - 7.0
                viable = "YES" if net > 0 else ""

                print(f"  {h:>6}s {follow_edge:>+9.2f}bps {eth_abs:>7.2f} {follow_edge:>+11.2f} {net:>+7.2f} {viable:>7}")

    # ============================================================
    # SUMMARY: Which events are viable?
    # ============================================================
    print(f"\n{'='*80}")
    print("VIABLE EVENTS SUMMARY (net edge > 0 after 7 bps RT fees)")
    print(f"{'='*80}")
    print(f"{'Event':<45} {'Freq':>6} {'Horizon':>8} {'Edge':>8} {'Net':>8}")
    print("-" * 80)

    for r in all_results:
        for h in r.get("horizons", []):
            if h["viable"]:
                print(f"  {r['name'][:43]:<43} {r['freq_pct']:>5.2f}% {h['horizon_s']:>6}s "
                      f"{h['reversion_edge']:>+7.2f} {h['net_edge']:>+7.2f}")

    # If none viable, show closest
    if not any(h["viable"] for r in all_results for h in r.get("horizons", [])):
        print("  NONE — showing top 10 closest:")
        all_horizons = []
        for r in all_results:
            for h in r.get("horizons", []):
                all_horizons.append((r["name"], r["freq_pct"], h))
        all_horizons.sort(key=lambda x: x[2]["net_edge"], reverse=True)
        for name, freq, h in all_horizons[:10]:
            print(f"  {name[:43]:<43} {freq:>5.2f}% {h['horizon_s']:>6}s "
                  f"{h['reversion_edge']:>+7.2f} {h['net_edge']:>+7.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
