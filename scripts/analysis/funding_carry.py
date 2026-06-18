"""
Funding Carry Hypothesis Test

Strategy: When funding rate is extreme, take the opposite side (collect funding).
  - Positive funding + z > threshold → SHORT perp (collect from longs)
  - Negative funding + z < -threshold → LONG perp (collect from shorts)
  - Exit when funding normalizes (|z| < exit_threshold)

Key questions this answers:
  1. Funding persistence: how long do extreme episodes last?
  2. Adverse price drift: does price move against you during extreme funding?
  3. Net P&L: funding collected - price drift - fees
  4. Feature conditioning: which NAT features predict persistence?

Funding mechanics (Hyperliquid):
  - Funding settled every 1 hour
  - Rate = hourly rate applied to position notional
  - Positive rate: longs pay shorts. Negative: shorts pay longs.
"""

import glob
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load_symbol(symbol: str = "BTC") -> pd.DataFrame:
    """Load all data for a symbol, sorted by time."""
    files = sorted(glob.glob("data/features/**/*.parquet", recursive=True))
    files = [f for f in files if os.path.getsize(f) > 0]

    dfs = []
    for f in files:
        try:
            t = pq.read_table(f)
            d = t.to_pandas()
            dfs.append(d[d["symbol"] == symbol])
        except Exception:
            pass

    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp_ns").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["timestamp_ns"], unit="ns")
    return df


def analyze_funding_persistence(df: pd.DataFrame):
    """How long do extreme funding episodes last?"""
    print("=" * 80)
    print("1. FUNDING PERSISTENCE ANALYSIS")
    print("=" * 80)

    # Resample to 1-second to avoid overcounting (data is ~100ms)
    df_1s = df.set_index("ts").resample("1s").last().dropna(subset=["ctx_funding_rate"])

    fz = df_1s["ctx_funding_zscore"].values
    fr = df_1s["ctx_funding_rate"].values
    ts = df_1s.index

    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        # Find episodes where |z| > threshold
        extreme = np.abs(fz) > threshold
        # Find episode starts (transition from non-extreme to extreme)
        starts = np.where(~extreme[:-1] & extreme[1:])[0] + 1
        # Find episode ends
        ends = np.where(extreme[:-1] & ~extreme[1:])[0] + 1

        if len(starts) == 0:
            print(f"\n  |z| > {threshold}: no episodes found")
            continue

        # Match starts to ends
        durations = []
        for s in starts:
            matching_ends = ends[ends > s]
            if len(matching_ends) > 0:
                e = matching_ends[0]
                dur_seconds = (ts[e] - ts[s]).total_seconds()
                durations.append(dur_seconds)

        if not durations:
            print(f"\n  |z| > {threshold}: episodes found but none completed")
            continue

        durs = np.array(durations)
        print(f"\n  |z| > {threshold}:")
        print(f"    Episodes: {len(durs)}")
        print(f"    Duration: mean={durs.mean()/60:.1f}min, median={np.median(durs)/60:.1f}min")
        print(f"    P25={np.percentile(durs, 25)/60:.1f}min, P75={np.percentile(durs, 75)/60:.1f}min")
        print(f"    P90={np.percentile(durs, 90)/60:.1f}min, P95={np.percentile(durs, 95)/60:.1f}min")
        print(f"    Max={durs.max()/3600:.1f}h")
        print(f"    % lasting > 1h: {100 * np.mean(durs > 3600):.1f}%")
        print(f"    % lasting > 4h: {100 * np.mean(durs > 14400):.1f}%")
        print(f"    % lasting > 8h: {100 * np.mean(durs > 28800):.1f}%")


def analyze_adverse_drift(df: pd.DataFrame):
    """When funding is extreme positive (longs pay), does price keep going UP?
    That's adverse selection — you short to collect funding but price moves against you."""
    print("\n" + "=" * 80)
    print("2. ADVERSE PRICE DRIFT DURING EXTREME FUNDING")
    print("=" * 80)

    df_1s = df.set_index("ts").resample("1s").last().dropna(subset=["ctx_funding_rate"])

    mid = df_1s["raw_midprice"].values
    fz = df_1s["ctx_funding_zscore"].values
    fr = df_1s["ctx_funding_rate"].values

    horizons_sec = [60, 300, 600, 1800, 3600]  # 1m, 5m, 10m, 30m, 1h
    horizon_names = ["1m", "5m", "10m", "30m", "1h"]

    for threshold in [1.5, 2.0, 2.5, 3.0]:
        print(f"\n  Funding z > +{threshold} (longs pay → strategy SHORTs):")
        extreme_pos = fz > threshold
        n_pos = extreme_pos.sum()
        if n_pos < 100:
            print(f"    Only {n_pos} samples, skipping")
            continue

        for h, name in zip(horizons_sec, horizon_names):
            if len(mid) <= h:
                continue
            fwd_ret_bps = (mid[h:] / mid[:-h] - 1) * 10000
            fwd = fwd_ret_bps[:len(extreme_pos)]
            mask = extreme_pos[:len(fwd)]

            cond_ret = fwd[mask]
            base_ret = fwd[~mask]
            valid = np.isfinite(cond_ret)

            # Positive fwd_ret = price went UP = BAD for short
            mean_drift = np.nanmean(cond_ret)
            std_drift = np.nanstd(cond_ret)
            base_drift = np.nanmean(base_ret)
            n_valid = valid.sum()

            # t-test significance
            if n_valid > 30:
                se = std_drift / np.sqrt(n_valid)
                t_stat = mean_drift / se if se > 0 else 0
                print(f"    @{name:>4s}: drift={mean_drift:+.2f}bps (base={base_drift:+.2f}), "
                      f"std={std_drift:.1f}bps, t={t_stat:.2f}, n={n_valid:,}")

        print(f"\n  Funding z < -{threshold} (shorts pay → strategy LONGs):")
        extreme_neg = fz < -threshold
        n_neg = extreme_neg.sum()
        if n_neg < 100:
            print(f"    Only {n_neg} samples, skipping")
            continue

        for h, name in zip(horizons_sec, horizon_names):
            if len(mid) <= h:
                continue
            fwd_ret_bps = (mid[h:] / mid[:-h] - 1) * 10000
            fwd = fwd_ret_bps[:len(extreme_neg)]
            mask = extreme_neg[:len(fwd)]

            cond_ret = fwd[mask]
            # Negative fwd_ret = price went DOWN = BAD for long
            mean_drift = np.nanmean(cond_ret)
            std_drift = np.nanstd(cond_ret)
            n_valid = np.isfinite(cond_ret).sum()

            if n_valid > 30:
                se = std_drift / np.sqrt(n_valid)
                t_stat = mean_drift / se if se > 0 else 0
                print(f"    @{name:>4s}: drift={mean_drift:+.2f}bps (adverse if negative), "
                      f"std={std_drift:.1f}bps, t={t_stat:.2f}, n={n_valid:,}")


def simulate_carry_trades(df: pd.DataFrame, fee_bps: float = 7.0):
    """
    Simulate the actual carry trade with explicit entry/exit.

    Rules:
      - ENTER SHORT when funding z > entry_z (positive funding, collect from longs)
      - ENTER LONG when funding z < -entry_z (negative funding, collect from shorts)
      - EXIT when |z| < exit_z (funding normalized)
      - Funding collected = hourly rate × hours held
      - P&L = funding_collected - price_change - RT_fees
    """
    print("\n" + "=" * 80)
    print("3. CARRY TRADE SIMULATION")
    print("=" * 80)

    # Resample to 1-minute bars for simulation (faster, still granular)
    df_1m = df.set_index("ts").resample("1min").last().dropna(subset=["ctx_funding_rate"])

    mid = df_1m["raw_midprice"].values
    fr = df_1m["ctx_funding_rate"].values
    fz = df_1m["ctx_funding_zscore"].values
    ts = df_1m.index

    configs = [
        {"entry_z": 1.5, "exit_z": 0.5},
        {"entry_z": 2.0, "exit_z": 0.5},
        {"entry_z": 2.0, "exit_z": 1.0},
        {"entry_z": 2.5, "exit_z": 0.5},
        {"entry_z": 2.5, "exit_z": 1.0},
        {"entry_z": 3.0, "exit_z": 0.5},
        {"entry_z": 3.0, "exit_z": 1.0},
    ]

    for cfg in configs:
        entry_z = cfg["entry_z"]
        exit_z = cfg["exit_z"]

        trades = []
        position = 0  # +1 = long, -1 = short, 0 = flat
        entry_price = 0.0
        entry_idx = 0
        funding_collected = 0.0

        for i in range(1, len(mid)):
            if not np.isfinite(fz[i]) or not np.isfinite(fr[i]) or not np.isfinite(mid[i]):
                continue

            # Accumulate funding while in position (every minute, rate is hourly)
            if position != 0:
                # funding_rate is hourly. Per minute = rate / 60
                # If SHORT and rate > 0: we COLLECT (longs pay us)
                # If LONG and rate < 0: we COLLECT (shorts pay us)
                minute_funding = fr[i] / 60.0  # Rate per minute
                if position == -1:  # Short
                    funding_collected += minute_funding  # Positive rate = we collect
                elif position == 1:  # Long
                    funding_collected -= minute_funding  # Negative rate = we collect

            # Entry logic
            if position == 0:
                if fz[i] > entry_z:
                    position = -1  # Short (collect positive funding)
                    entry_price = mid[i]
                    entry_idx = i
                    funding_collected = 0.0
                elif fz[i] < -entry_z:
                    position = 1  # Long (collect negative funding)
                    entry_price = mid[i]
                    entry_idx = i
                    funding_collected = 0.0

            # Exit logic
            elif abs(fz[i]) < exit_z:
                exit_price = mid[i]
                hold_minutes = i - entry_idx
                hold_hours = hold_minutes / 60.0

                # Price P&L
                if position == -1:
                    price_pnl_bps = -(exit_price / entry_price - 1) * 10000  # Short: profit when price drops
                else:
                    price_pnl_bps = (exit_price / entry_price - 1) * 10000  # Long: profit when price rises

                # Funding P&L in bps
                funding_pnl_bps = funding_collected * 10000

                # Net
                net_pnl = price_pnl_bps + funding_pnl_bps - fee_bps

                trades.append({
                    "entry_time": ts[entry_idx],
                    "exit_time": ts[i],
                    "side": "SHORT" if position == -1 else "LONG",
                    "hold_hours": hold_hours,
                    "price_pnl_bps": price_pnl_bps,
                    "funding_pnl_bps": funding_pnl_bps,
                    "fee_bps": fee_bps,
                    "net_pnl_bps": net_pnl,
                    "entry_z": fz[entry_idx],
                })

                position = 0

        if not trades:
            print(f"\n  Entry |z|>{entry_z}, Exit |z|<{exit_z}: NO TRADES")
            continue

        tdf = pd.DataFrame(trades)
        n = len(tdf)
        win_rate = (tdf["net_pnl_bps"] > 0).mean()
        shorts = tdf[tdf["side"] == "SHORT"]
        longs = tdf[tdf["side"] == "LONG"]

        print(f"\n  Entry |z|>{entry_z}, Exit |z|<{exit_z}  (fee={fee_bps}bps RT):")
        print(f"    Trades: {n} ({len(shorts)} short, {len(longs)} long)")
        print(f"    Win rate: {100*win_rate:.1f}%")
        print(f"    Hold time: mean={tdf['hold_hours'].mean():.1f}h, median={tdf['hold_hours'].median():.1f}h")
        print(f"    --- P&L Breakdown (bps per trade) ---")
        print(f"    Price drift:   mean={tdf['price_pnl_bps'].mean():+.2f}, median={tdf['price_pnl_bps'].median():+.2f}")
        print(f"    Funding:       mean={tdf['funding_pnl_bps'].mean():+.2f}, median={tdf['funding_pnl_bps'].median():+.2f}")
        print(f"    Fees:          -{fee_bps:.1f}")
        print(f"    NET:           mean={tdf['net_pnl_bps'].mean():+.2f}, median={tdf['net_pnl_bps'].median():+.2f}")
        print(f"    Total P&L:     {tdf['net_pnl_bps'].sum():+.1f} bps over {n} trades")
        if len(shorts) > 0:
            print(f"    Shorts only:   mean={shorts['net_pnl_bps'].mean():+.2f}, n={len(shorts)}")
        if len(longs) > 0:
            print(f"    Longs only:    mean={longs['net_pnl_bps'].mean():+.2f}, n={len(longs)}")

        # Cumulative P&L
        cum = tdf["net_pnl_bps"].cumsum()
        max_dd = (cum - cum.cummax()).min()
        print(f"    Max drawdown:  {max_dd:.1f} bps")
        print(f"    Sharpe (per trade): {tdf['net_pnl_bps'].mean() / (tdf['net_pnl_bps'].std() + 1e-12):.3f}")


def feature_conditioned_carry(df: pd.DataFrame, fee_bps: float = 7.0):
    """
    Test whether NAT features can improve carry trade timing.

    Hypothesis: enter carry only when features predict funding will PERSIST.
    Features that might help:
      - ent_book_shape: low entropy → stable regime → funding persists
      - trend_hurst_300: high Hurst → trending → funding persists
      - vol_zscore: low vol → calm market → funding persists
      - trend_momentum_300: strong momentum → trend driving funding → persists
    """
    print("\n" + "=" * 80)
    print("4. FEATURE-CONDITIONED CARRY (does timing help?)")
    print("=" * 80)

    df_1m = df.set_index("ts").resample("1min").last().dropna(subset=["ctx_funding_rate"])

    mid = df_1m["raw_midprice"].values
    fr = df_1m["ctx_funding_rate"].values
    fz = df_1m["ctx_funding_zscore"].values

    # Conditioning features
    cond_features = {
        "ent_book_shape": {"col": None, "filter": "low", "desc": "Low entropy (stable book)"},
        "trend_hurst_300": {"col": None, "filter": "high", "desc": "High Hurst (trending)"},
        "vol_zscore": {"col": None, "filter": "low", "desc": "Low vol zscore (calm)"},
        "trend_momentum_300": {"col": None, "filter": "high_abs", "desc": "Strong momentum"},
    }

    # Find columns (may have BTC_ prefix)
    for feat in list(cond_features.keys()):
        for prefix in ["", "BTC_"]:
            col = f"{prefix}{feat}"
            if col in df_1m.columns:
                cond_features[feat]["col"] = col
                break

    entry_z = 2.0
    exit_z = 0.5

    def run_carry(mask_name, entry_mask):
        """Run carry sim with additional entry mask."""
        trades = []
        position = 0
        entry_price = 0.0
        entry_idx = 0
        funding_collected = 0.0

        for i in range(1, len(mid)):
            if not np.isfinite(fz[i]) or not np.isfinite(fr[i]) or not np.isfinite(mid[i]):
                continue

            if position != 0:
                minute_funding = fr[i] / 60.0
                if position == -1:
                    funding_collected += minute_funding
                elif position == 1:
                    funding_collected -= minute_funding

            if position == 0:
                if fz[i] > entry_z and entry_mask[i]:
                    position = -1
                    entry_price = mid[i]
                    entry_idx = i
                    funding_collected = 0.0
                elif fz[i] < -entry_z and entry_mask[i]:
                    position = 1
                    entry_price = mid[i]
                    entry_idx = i
                    funding_collected = 0.0

            elif abs(fz[i]) < exit_z:
                exit_price = mid[i]
                hold_minutes = i - entry_idx

                if position == -1:
                    price_pnl_bps = -(exit_price / entry_price - 1) * 10000
                else:
                    price_pnl_bps = (exit_price / entry_price - 1) * 10000

                funding_pnl_bps = funding_collected * 10000
                net_pnl = price_pnl_bps + funding_pnl_bps - fee_bps

                trades.append({
                    "hold_hours": hold_minutes / 60.0,
                    "price_pnl_bps": price_pnl_bps,
                    "funding_pnl_bps": funding_pnl_bps,
                    "net_pnl_bps": net_pnl,
                })
                position = 0

        return trades

    # Baseline: no conditioning
    base_mask = np.ones(len(mid), dtype=bool)
    base_trades = run_carry("baseline", base_mask)
    if base_trades:
        bt = pd.DataFrame(base_trades)
        print(f"\n  Baseline (entry |z|>{entry_z}, exit |z|<{exit_z}):")
        print(f"    Trades: {len(bt)}, win={100*(bt['net_pnl_bps']>0).mean():.0f}%, "
              f"mean={bt['net_pnl_bps'].mean():+.2f}, total={bt['net_pnl_bps'].sum():+.1f}")

    # Conditioned entries
    for feat, info in cond_features.items():
        col = info["col"]
        if col is None:
            print(f"\n  {feat}: column not found, skipping")
            continue

        vals = df_1m[col].values
        valid = np.isfinite(vals)

        if info["filter"] == "low":
            # Enter only when feature is below P30
            thresh = np.nanpercentile(vals[valid], 30)
            mask = valid & (vals < thresh)
            label = f"< P30 ({thresh:.3f})"
        elif info["filter"] == "high":
            thresh = np.nanpercentile(vals[valid], 70)
            mask = valid & (vals > thresh)
            label = f"> P70 ({thresh:.3f})"
        elif info["filter"] == "high_abs":
            thresh = np.nanpercentile(np.abs(vals[valid]), 70)
            mask = valid & (np.abs(vals) > thresh)
            label = f"|val| > P70 ({thresh:.3f})"
        else:
            continue

        trades = run_carry(feat, mask)
        if trades:
            tdf = pd.DataFrame(trades)
            improvement = tdf["net_pnl_bps"].mean() - bt["net_pnl_bps"].mean() if base_trades else 0
            print(f"\n  + {feat} {label}  ({info['desc']}):")
            print(f"    Trades: {len(tdf)}, win={100*(tdf['net_pnl_bps']>0).mean():.0f}%, "
                  f"mean={tdf['net_pnl_bps'].mean():+.2f}, total={tdf['net_pnl_bps'].sum():+.1f}, "
                  f"Δ={improvement:+.2f}")
        else:
            print(f"\n  + {feat}: no trades with this filter")


def funding_rate_mechanics(df: pd.DataFrame):
    """Understand the actual funding rate dynamics on Hyperliquid."""
    print("\n" + "=" * 80)
    print("5. FUNDING RATE MECHANICS")
    print("=" * 80)

    df_1m = df.set_index("ts").resample("1min").last().dropna(subset=["ctx_funding_rate"])
    fr = df_1m["ctx_funding_rate"]
    fz = df_1m["ctx_funding_zscore"]
    prem = df_1m["ctx_premium_bps"]
    ts = df_1m.index

    # How often does funding rate change?
    fr_diff = fr.diff()
    changes = (fr_diff != 0) & fr_diff.notna()
    print(f"\n  Funding rate changes: {changes.sum()} times in {len(fr)} minutes")
    print(f"  Change frequency: every {len(fr) / max(changes.sum(), 1):.0f} minutes on average")

    # Autocorrelation of funding rate
    print("\n  Funding rate autocorrelation:")
    for lag_min in [1, 5, 15, 30, 60, 120, 240, 480]:
        if lag_min < len(fr):
            ac = fr.autocorr(lag=lag_min)
            print(f"    Lag {lag_min:>4d}min ({lag_min/60:.1f}h): {ac:.4f}")

    # Funding rate autocorrelation of ZSCORE (more informative)
    print("\n  Funding zscore autocorrelation:")
    for lag_min in [1, 5, 15, 30, 60, 120, 240, 480]:
        if lag_min < len(fz):
            ac = fz.autocorr(lag=lag_min)
            print(f"    Lag {lag_min:>4d}min ({lag_min/60:.1f}h): {ac:.4f}")

    # Cumulative funding over rolling windows
    print("\n  Cumulative funding (bps) over rolling windows:")
    for window_h in [1, 4, 8, 24]:
        window_min = window_h * 60
        # Each minute contributes rate/60 to total funding
        cum_funding = (fr / 60).rolling(window_min).sum() * 10000  # In bps
        valid = cum_funding.dropna()
        if len(valid) > 0:
            print(f"    {window_h:>2d}h window: mean={valid.mean():.2f}bps, "
                  f"P5={valid.quantile(0.05):.2f}, P95={valid.quantile(0.95):.2f}, "
                  f"max={valid.max():.2f}")

    # Premium → funding relationship
    print("\n  Premium → Funding correlation:")
    valid = prem.notna() & fr.notna()
    if valid.sum() > 100:
        corr = prem[valid].corr(fr[valid])
        print(f"    Contemporaneous: {corr:.4f}")
        # Does premium predict future funding changes?
        for lag in [60, 120, 240]:
            fz_future = fz.shift(-lag)
            valid_lag = prem.notna() & fz_future.notna()
            if valid_lag.sum() > 100:
                corr_lag = prem[valid_lag].corr(fz_future[valid_lag])
                print(f"    Premium → funding z (t+{lag}min): {corr_lag:.4f}")


def multi_symbol_carry(fee_bps: float = 7.0):
    """Compare carry dynamics across BTC, ETH, SOL."""
    print("\n" + "=" * 80)
    print("6. MULTI-SYMBOL CARRY COMPARISON")
    print("=" * 80)

    for symbol in ["BTC", "ETH", "SOL"]:
        df = load_symbol(symbol)
        df_1m = df.set_index("ts").resample("1min").last().dropna(subset=["ctx_funding_rate"])

        fr = df_1m["ctx_funding_rate"]
        fz = df_1m["ctx_funding_zscore"]
        mid = df_1m["raw_midprice"].values

        # Quick carry sim
        entry_z, exit_z = 2.0, 0.5
        trades = []
        position = 0
        entry_price = 0.0
        entry_idx = 0
        funding_collected = 0.0

        fr_vals = fr.values
        fz_vals = fz.values

        for i in range(1, len(mid)):
            if not np.isfinite(fz_vals[i]) or not np.isfinite(fr_vals[i]) or not np.isfinite(mid[i]):
                continue

            if position != 0:
                minute_funding = fr_vals[i] / 60.0
                if position == -1:
                    funding_collected += minute_funding
                elif position == 1:
                    funding_collected -= minute_funding

            if position == 0:
                if fz_vals[i] > entry_z:
                    position = -1
                    entry_price = mid[i]
                    entry_idx = i
                    funding_collected = 0.0
                elif fz_vals[i] < -entry_z:
                    position = 1
                    entry_price = mid[i]
                    entry_idx = i
                    funding_collected = 0.0
            elif abs(fz_vals[i]) < exit_z:
                exit_price = mid[i]
                hold_min = i - entry_idx
                if position == -1:
                    price_pnl = -(exit_price / entry_price - 1) * 10000
                else:
                    price_pnl = (exit_price / entry_price - 1) * 10000
                funding_pnl = funding_collected * 10000
                trades.append({
                    "hold_h": hold_min / 60,
                    "price": price_pnl,
                    "funding": funding_pnl,
                    "net": price_pnl + funding_pnl - fee_bps,
                })
                position = 0

        if trades:
            tdf = pd.DataFrame(trades)
            print(f"\n  {symbol}: {len(tdf)} trades, "
                  f"win={100*(tdf['net']>0).mean():.0f}%, "
                  f"mean_net={tdf['net'].mean():+.2f}bps, "
                  f"funding_mean={tdf['funding'].mean():+.2f}bps, "
                  f"price_mean={tdf['price'].mean():+.2f}bps, "
                  f"hold={tdf['hold_h'].mean():.1f}h, "
                  f"total={tdf['net'].sum():+.1f}bps")
        else:
            print(f"\n  {symbol}: no trades")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])

    print("Loading BTC data...")
    df = load_symbol("BTC")
    print(f"Loaded {len(df):,} rows, {df['ts'].min()} to {df['ts'].max()}")

    # Run all analyses
    funding_rate_mechanics(df)
    analyze_funding_persistence(df)
    analyze_adverse_drift(df)
    simulate_carry_trades(df, fee_bps=7.0)
    feature_conditioned_carry(df, fee_bps=7.0)
    multi_symbol_carry(fee_bps=7.0)

    # Also test with maker fees (lower)
    print("\n" + "=" * 80)
    print("7. SENSITIVITY: MAKER FEES (2 bps RT instead of 7)")
    print("=" * 80)
    simulate_carry_trades(df, fee_bps=2.0)

    print("\nDone.")
