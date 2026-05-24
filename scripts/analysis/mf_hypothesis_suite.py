#!/usr/bin/env python3
"""
MF Liquidity Signal — Hypothesis Suite

Six experiments probing the spread+depth composite signal:
  H1: Directional vs volatility (signed IC vs |return| IC)
  H2: Long/short leg decomposition (walk-forward by direction)
  H3: Time-of-day proxy test (residualize spread on hour dummies)
  H4: Signal decay profile (IC at 10/25/50/100/200 min)
  H5: 3-feature composite vs 2-feature (add flow_vwap_deviation_std)
  H6: Maker execution simulation (limit fill model at reduced fees)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr


# ── Config ────────────────────────────────────────────────────────────────

BAR_SECONDS = 300
MIN_BARS_PER_DATE = 12
TRAIN_WINDOW = 3
P_LONG = 80
P_SHORT = 20

FEE_MODELS = {
    "binance_vip9": {"taker": 1.61, "maker": 0.30},
    "binance_vip0": {"taker": 3.50, "maker": 1.00},
    "hyperliquid": {"taker": 7.00, "maker": 1.00},
}

LOAD_COLUMNS = [
    "timestamp_ns", "symbol", "raw_midprice",
    "raw_spread_bps", "raw_ask_depth_5", "flow_vwap_deviation",
]


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class HypothesisResult:
    id: str
    name: str
    verdict: str  # CONFIRMED / REJECTED / INCONCLUSIVE
    metrics: dict = field(default_factory=dict)
    criteria: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)


# ── Data loading ─────────────────────────────────────────────────────────

def load_date(data_dir: Path, date_str: str, symbol: str) -> pd.DataFrame | None:
    date_path = data_dir / date_str
    if not date_path.is_dir():
        return None
    files = sorted(f for f in date_path.iterdir() if f.suffix == ".parquet")
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            # Load available columns (flow_vwap_deviation may be missing)
            tbl = pq.read_table(str(f))
            df = tbl.to_pandas()
            cols_present = [c for c in LOAD_COLUMNS if c in df.columns]
            df = df[cols_present]
            df = df[df["symbol"] == symbol].copy() if "symbol" in df.columns else df
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True).sort_values("timestamp_ns").reset_index(drop=True)


def aggregate_to_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    bar_ns = BAR_SECONDS * 1_000_000_000
    ticks = ticks.copy()
    ticks["bar_id"] = ticks["timestamp_ns"].values // bar_ns

    agg = {
        "timestamp_ns": "first",
        "raw_midprice": "last",
        "raw_spread_bps": "last",
        "raw_ask_depth_5": "std",
    }
    if "flow_vwap_deviation" in ticks.columns:
        agg["flow_vwap_deviation"] = "std"

    bars = ticks.groupby("bar_id").agg(agg).reset_index(drop=True)
    bars = bars.rename(columns={
        "raw_midprice": "midprice_last",
        "raw_spread_bps": "spread_bps_last",
        "raw_ask_depth_5": "depth_5_std",
    })
    if "flow_vwap_deviation" in bars.columns:
        bars = bars.rename(columns={"flow_vwap_deviation": "vwap_deviation_std"})

    # Bar hour from timestamp_ns
    bars["bar_hour"] = pd.to_datetime(bars["timestamp_ns"], unit="ns", utc=True).dt.hour

    # Count ticks per bar for filtering
    tick_counts = ticks.groupby("bar_id").size()
    bars["n_ticks"] = tick_counts.values
    bars = bars[bars["n_ticks"] >= 10].reset_index(drop=True)
    bars["depth_5_std"] = bars["depth_5_std"].fillna(0.0)
    if "vwap_deviation_std" in bars.columns:
        bars["vwap_deviation_std"] = bars["vwap_deviation_std"].fillna(0.0)
    return bars


def load_all_data(
    data_dir: Path, symbols: list[str],
) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    all_dates = sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("2026-") and "clean" not in d and (data_dir / d).is_dir()
    )
    result = {}
    for symbol in symbols:
        date_bars = []
        for date_str in all_dates:
            ticks = load_date(data_dir, date_str, symbol)
            if ticks is None or len(ticks) < 100:
                continue
            bars = aggregate_to_bars(ticks)
            if len(bars) >= MIN_BARS_PER_DATE:
                date_bars.append((date_str, bars))
        result[symbol] = date_bars
        print(f"  {symbol}: {len(date_bars)} dates loaded")
    return result


# ── Shared helpers ───────────────────────────────────────────────────────

def forward_returns(prices: np.ndarray, horizon: int) -> np.ndarray:
    n = len(prices)
    fwd = np.full(n, np.nan)
    for i in range(n - horizon):
        if prices[i] > 0 and np.isfinite(prices[i]) and np.isfinite(prices[i + horizon]):
            fwd[i] = (prices[i + horizon] - prices[i]) / prices[i] * 1e4
    return fwd


def date_ic(bars: pd.DataFrame, feat_col: str, horizon: int,
            use_abs_return: bool = False) -> float | None:
    prices = bars["midprice_last"].values
    fwd = forward_returns(prices, horizon)
    if use_abs_return:
        fwd = np.abs(fwd)
    valid = np.isfinite(fwd)
    if feat_col not in bars.columns:
        return None
    x = bars[feat_col].values
    both = valid & np.isfinite(x)
    if both.sum() < 15 or np.std(x[both]) < 1e-12:
        return None
    ic, _ = spearmanr(x[both], fwd[both])
    return float(ic) if np.isfinite(ic) else None


def compute_zscore_params_2f(train_bars_list: list[pd.DataFrame]) -> dict:
    spread = np.concatenate([b["spread_bps_last"].values for b in train_bars_list])
    depth = np.concatenate([b["depth_5_std"].values for b in train_bars_list])
    mask = np.isfinite(spread) & np.isfinite(depth)
    spread, depth = spread[mask], depth[mask]
    params = {
        "spread_mean": np.mean(spread), "spread_std": max(np.std(spread), 1e-10),
        "depth_mean": np.mean(depth), "depth_std": max(np.std(depth), 1e-10),
    }
    z_s = (spread - params["spread_mean"]) / params["spread_std"]
    z_d = (depth - params["depth_mean"]) / params["depth_std"]
    composite = (z_s + z_d) / 2.0
    params["p_long"] = np.percentile(composite, P_LONG)
    params["p_short"] = np.percentile(composite, P_SHORT)
    return params


def compute_zscore_params_3f(train_bars_list: list[pd.DataFrame]) -> dict | None:
    spread = np.concatenate([b["spread_bps_last"].values for b in train_bars_list])
    depth = np.concatenate([b["depth_5_std"].values for b in train_bars_list])
    vwap_arrs = []
    for b in train_bars_list:
        if "vwap_deviation_std" in b.columns:
            vwap_arrs.append(b["vwap_deviation_std"].values)
    if not vwap_arrs:
        return None
    vwap = np.concatenate(vwap_arrs)
    n = min(len(spread), len(depth), len(vwap))
    spread, depth, vwap = spread[:n], depth[:n], vwap[:n]
    mask = np.isfinite(spread) & np.isfinite(depth) & np.isfinite(vwap)
    spread, depth, vwap = spread[mask], depth[mask], vwap[mask]
    if len(spread) < 20:
        return None
    params = {
        "spread_mean": np.mean(spread), "spread_std": max(np.std(spread), 1e-10),
        "depth_mean": np.mean(depth), "depth_std": max(np.std(depth), 1e-10),
        "vwap_mean": np.mean(vwap), "vwap_std": max(np.std(vwap), 1e-10),
    }
    z_s = (spread - params["spread_mean"]) / params["spread_std"]
    z_d = (depth - params["depth_mean"]) / params["depth_std"]
    z_v = (vwap - params["vwap_mean"]) / params["vwap_std"]
    composite = (z_s + z_d + z_v) / 3.0
    params["p_long"] = np.percentile(composite, P_LONG)
    params["p_short"] = np.percentile(composite, P_SHORT)
    return params


def apply_signal_2f(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    bars = bars.copy()
    z_s = (bars["spread_bps_last"] - params["spread_mean"]) / params["spread_std"]
    z_d = (bars["depth_5_std"] - params["depth_mean"]) / params["depth_std"]
    bars["composite"] = (z_s + z_d) / 2.0
    bars["direction"] = 0
    bars.loc[bars["composite"] >= params["p_long"], "direction"] = 1
    bars.loc[bars["composite"] <= params["p_short"], "direction"] = -1
    return bars


def apply_signal_3f(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    bars = bars.copy()
    z_s = (bars["spread_bps_last"] - params["spread_mean"]) / params["spread_std"]
    z_d = (bars["depth_5_std"] - params["depth_mean"]) / params["depth_std"]
    z_v = (bars["vwap_deviation_std"] - params["vwap_mean"]) / params["vwap_std"]
    bars["composite"] = (z_s + z_d + z_v) / 3.0
    bars["direction"] = 0
    bars.loc[bars["composite"] >= params["p_long"], "direction"] = 1
    bars.loc[bars["composite"] <= params["p_short"], "direction"] = -1
    return bars


# ── H1: Directional vs Volatility ───────────────────────────────────────

def run_h1(data: dict, horizon: int = 10) -> HypothesisResult:
    feat = "spread_bps_last"
    per_symbol = {}

    for symbol, date_bars in data.items():
        signed_ics, abs_ics = [], []
        for d, bars in date_bars:
            s_ic = date_ic(bars, feat, horizon, use_abs_return=False)
            a_ic = date_ic(bars, feat, horizon, use_abs_return=True)
            if s_ic is not None:
                signed_ics.append(s_ic)
            if a_ic is not None:
                abs_ics.append(a_ic)

        if signed_ics and abs_ics:
            s_arr = np.array(signed_ics)
            a_arr = np.array(abs_ics)
            s_mean = float(np.mean(s_arr))
            a_mean = float(np.mean(a_arr))
            s_t = s_mean / (np.std(s_arr) / np.sqrt(len(s_arr))) if np.std(s_arr) > 0 else 0
            per_symbol[symbol] = {
                "signed_ic": round(s_mean, 4),
                "abs_ic": round(a_mean, 4),
                "signed_t": round(s_t, 2),
                "ratio": round(abs(s_mean) / max(abs(a_mean), 1e-6), 2),
                "n_dates": len(signed_ics),
            }

    # Verdict: check BTC primarily
    btc = per_symbol.get("BTC", {})
    ratio = btc.get("ratio", 0)
    t_stat = btc.get("signed_t", 0)

    if ratio > 2.0 and t_stat > 2.0:
        verdict = "CONFIRMED"
    elif btc.get("abs_ic", 0) and abs(btc.get("abs_ic", 0)) >= abs(btc.get("signed_ic", 0)):
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    return HypothesisResult(
        id="H1", name="Spread is directional, not just volatility",
        verdict=verdict,
        metrics=per_symbol,
        criteria={"confirmed": "signed IC > 2x |return| IC and t>2",
                  "rejected": "|return| IC >= signed IC"},
    )


# ── H2: Long/Short Decomposition ────────────────────────────────────────

def compute_trades_by_side(bars: pd.DataFrame, horizon: int, fee_bps: float) -> dict:
    prices = bars["midprice_last"].values
    directions = bars["direction"].values
    n = len(prices)
    long_pnls, short_pnls = [], []

    for i in range(n - horizon):
        d = directions[i]
        if d == 0 or prices[i] <= 0 or np.isnan(prices[i]) or np.isnan(prices[i + horizon]):
            continue
        ret_bps = (prices[i + horizon] - prices[i]) / prices[i] * 1e4
        net = d * ret_bps - fee_bps
        if d == 1:
            long_pnls.append(net)
        else:
            short_pnls.append(net)

    def stats(pnls):
        if not pnls:
            return {"n": 0, "net_bps": 0.0, "win_rate": 0.0}
        arr = np.array(pnls)
        return {
            "n": len(arr),
            "net_bps": round(float(np.mean(arr)), 3),
            "total_net_bps": round(float(np.sum(arr)), 1),
            "win_rate": round(float(np.mean(arr > 0)), 2),
        }

    return {"long": stats(long_pnls), "short": stats(short_pnls)}


def run_h2(data: dict, fee_bps: float = 1.61, horizon: int = 10) -> HypothesisResult:
    per_symbol = {}

    for symbol, date_bars in data.items():
        if len(date_bars) < TRAIN_WINDOW + 1:
            continue

        all_long, all_short = [], []
        for i in range(TRAIN_WINDOW, len(date_bars)):
            train = [b for _, b in date_bars[i - TRAIN_WINDOW:i]]
            params = compute_zscore_params_2f(train)
            _, test_bars = date_bars[i]
            test_bars = apply_signal_2f(test_bars, params)
            sides = compute_trades_by_side(test_bars, horizon, fee_bps)
            all_long.append(sides["long"])
            all_short.append(sides["short"])

        long_n = sum(s["n"] for s in all_long)
        short_n = sum(s["n"] for s in all_short)
        long_net = (sum(s["net_bps"] * s["n"] for s in all_long) / long_n) if long_n > 0 else 0
        short_net = (sum(s["net_bps"] * s["n"] for s in all_short) / short_n) if short_n > 0 else 0

        per_symbol[symbol] = {
            "long_n": long_n, "long_net_bps": round(long_net, 3),
            "short_n": short_n, "short_net_bps": round(short_net, 3),
            "total_n": long_n + short_n,
        }

    # Verdict: asymmetric if one side <= 0 across 2+ symbols
    n_asym = sum(1 for s in per_symbol.values()
                 if (s["long_net_bps"] > 0) != (s["short_net_bps"] > 0))
    n_both_pos = sum(1 for s in per_symbol.values()
                     if s["long_net_bps"] > 0 and s["short_net_bps"] > 0)

    if n_asym >= 2:
        verdict = "CONFIRMED"
    elif n_both_pos >= 2:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    return HypothesisResult(
        id="H2", name="Long leg carries the edge, not short leg",
        verdict=verdict,
        metrics=per_symbol,
        criteria={"confirmed": "one side >0 other <=0 on 2+ symbols",
                  "rejected": "both sides >0 on 2+ symbols (symmetric)"},
    )


# ── H3: Time-of-Day Proxy ───────────────────────────────────────────────

def run_h3(data: dict, horizon: int = 10) -> HypothesisResult:
    per_symbol = {}

    for symbol, date_bars in data.items():
        raw_ics, resid_ics = [], []
        for d, bars in date_bars:
            hours = bars["bar_hour"].values
            unique_hours = np.unique(hours[np.isfinite(hours)])
            if len(unique_hours) < 4:
                continue

            # Raw IC
            r_ic = date_ic(bars, "spread_bps_last", horizon)
            if r_ic is None:
                continue
            raw_ics.append(r_ic)

            # Residualize spread on hour dummies
            spread = bars["spread_bps_last"].values.copy()
            valid = np.isfinite(spread) & np.isfinite(hours)
            if valid.sum() < 20:
                continue

            # Build hour dummy matrix
            h = hours[valid].astype(int)
            unique_h = np.unique(h)
            X = np.zeros((valid.sum(), len(unique_h)))
            for j, uh in enumerate(unique_h):
                X[:, j] = (h == uh).astype(float)

            y = spread[valid]
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ beta

            # Compute IC of residuals vs forward return
            prices = bars["midprice_last"].values
            fwd = forward_returns(prices, horizon)
            fwd_v = fwd[valid]
            both_ok = np.isfinite(fwd_v) & np.isfinite(residuals)
            if both_ok.sum() < 15 or np.std(residuals[both_ok]) < 1e-12:
                continue
            ic, _ = spearmanr(residuals[both_ok], fwd_v[both_ok])
            if np.isfinite(ic):
                resid_ics.append(float(ic))

        if raw_ics and resid_ics:
            raw_mean = float(np.mean(raw_ics))
            resid_mean = float(np.mean(resid_ics))
            retention = abs(resid_mean) / max(abs(raw_mean), 1e-6)
            per_symbol[symbol] = {
                "raw_ic": round(raw_mean, 4),
                "residual_ic": round(resid_mean, 4),
                "retention_pct": round(retention * 100, 1),
                "n_dates": len(resid_ics),
            }

    btc = per_symbol.get("BTC", {})
    ret = btc.get("retention_pct", 0)

    if ret >= 70:
        verdict = "CONFIRMED"
    elif ret < 30:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    return HypothesisResult(
        id="H3", name="Spread is not just a time-of-day proxy",
        verdict=verdict,
        metrics=per_symbol,
        criteria={"confirmed": "residual IC >= 70% of raw IC",
                  "rejected": "residual IC < 30% of raw IC"},
    )


# ── H4: Signal Decay Profile ────────────────────────────────────────────

def run_h4(data: dict) -> HypothesisResult:
    horizon_bars = [2, 5, 10, 20, 40]  # 10, 25, 50, 100, 200 min
    horizon_names = ["10min", "25min", "50min", "100min", "200min"]
    feat = "spread_bps_last"
    per_symbol = {}

    for symbol, date_bars in data.items():
        decay = {}
        for hb, hn in zip(horizon_bars, horizon_names):
            ics = []
            for d, bars in date_bars:
                ic = date_ic(bars, feat, hb)
                if ic is not None:
                    ics.append(ic)
            if ics:
                arr = np.array(ics)
                decay[hn] = {
                    "mean_ic": round(float(np.mean(arr)), 4),
                    "std_ic": round(float(np.std(arr)), 4),
                    "n_dates": len(arr),
                    "n_positive": int(np.sum(arr > 0)),
                }
        per_symbol[symbol] = decay

    # Verdict: check if 50min is near the peak for BTC
    btc = per_symbol.get("BTC", {})
    ics = {k: v["mean_ic"] for k, v in btc.items()}
    if ics:
        peak_h = max(ics, key=lambda k: abs(ics[k]))
        peak_idx = horizon_names.index(peak_h) if peak_h in horizon_names else -1
        ic_50 = ics.get("50min", 0)

        # Check monotonic decay after peak
        ic_vals = [ics.get(h, 0) for h in horizon_names]
        peak_val_idx = ic_vals.index(max(ic_vals, key=abs))
        decays_after = all(
            abs(ic_vals[j]) >= abs(ic_vals[j + 1])
            for j in range(peak_val_idx, len(ic_vals) - 1)
            if ic_vals[j + 1] != 0
        )

        if peak_idx >= 2 and decays_after:  # Peak at 50min or later
            verdict = "CONFIRMED"
        elif peak_idx < 2:  # Peak before 50min
            verdict = "REJECTED"
        else:
            verdict = "INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE"

    return HypothesisResult(
        id="H4", name="Signal decay profile across horizons",
        verdict=verdict,
        metrics=per_symbol,
        criteria={"confirmed": "IC peaks near 50min, decays after",
                  "rejected": "IC peaks at 10-25min"},
    )


# ── H5: 3-Feature vs 2-Feature ──────────────────────────────────────────

def walk_forward_backtest(date_bars, horizon, fee_bps, n_features=2):
    """Run walk-forward, return aggregate stats."""
    if len(date_bars) < TRAIN_WINDOW + 1:
        return None

    daily = []
    for i in range(TRAIN_WINDOW, len(date_bars)):
        train = [b for _, b in date_bars[i - TRAIN_WINDOW:i]]
        test_date, test_bars = date_bars[i]

        if n_features == 3:
            params = compute_zscore_params_3f(train)
            if params is None:
                continue
            test_bars = apply_signal_3f(test_bars, params)
        else:
            params = compute_zscore_params_2f(train)
            test_bars = apply_signal_2f(test_bars, params)

        prices = test_bars["midprice_last"].values
        directions = test_bars["direction"].values
        n = len(prices)
        pnls = []
        for j in range(n - horizon):
            d = directions[j]
            if d == 0 or prices[j] <= 0 or np.isnan(prices[j]) or np.isnan(prices[j + horizon]):
                continue
            ret = (prices[j + horizon] - prices[j]) / prices[j] * 1e4
            pnls.append(d * ret - fee_bps)

        daily.append({"date": test_date, "n": len(pnls),
                       "total_net": sum(pnls) if pnls else 0,
                       "mean_net": np.mean(pnls) if pnls else 0})

    total_trades = sum(d["n"] for d in daily)
    if total_trades == 0:
        return None

    net_bps = sum(d["mean_net"] * d["n"] for d in daily) / total_trades
    total_pnl = sum(d["total_net"] for d in daily)
    daily_pnl = np.array([d["total_net"] for d in daily])
    sharpe = (np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)) if np.std(daily_pnl) > 0 else 0

    return {
        "n_oos": len(daily),
        "n_trades": total_trades,
        "net_bps": round(float(net_bps), 3),
        "total_pnl_bps": round(float(total_pnl), 1),
        "sharpe": round(float(sharpe), 2),
        "win_rate": round(float(np.mean(daily_pnl > 0)), 2),
    }


def run_h5(data: dict, fee_bps: float = 1.61, horizon: int = 10) -> HypothesisResult:
    per_symbol = {}

    for symbol, date_bars in data.items():
        r2 = walk_forward_backtest(date_bars, horizon, fee_bps, n_features=2)
        r3 = walk_forward_backtest(date_bars, horizon, fee_bps, n_features=3)
        if r2 and r3:
            per_symbol[symbol] = {
                "2f": r2, "3f": r3,
                "sharpe_delta": round(r3["sharpe"] - r2["sharpe"], 2),
                "net_delta": round(r3["net_bps"] - r2["net_bps"], 3),
            }

    n_improved = sum(1 for s in per_symbol.values()
                     if s["sharpe_delta"] > 0 and s["net_delta"] > 0)

    if n_improved >= 2:
        verdict = "CONFIRMED"
    elif all(s["sharpe_delta"] < 0 for s in per_symbol.values() if "sharpe_delta" in s):
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    return HypothesisResult(
        id="H5", name="3-feature composite outperforms 2-feature",
        verdict=verdict,
        metrics=per_symbol,
        criteria={"confirmed": "3f Sharpe > 2f AND net > 2f on 2+ symbols",
                  "rejected": "3f underperforms on all symbols"},
    )


# ── H6: Maker Execution Simulation ──────────────────────────────────────

def run_h6(data: dict, taker_fee: float = 1.61, maker_fee: float = 0.30,
           horizon: int = 10) -> HypothesisResult:
    fill_rates = [0.50, 0.70, 0.90]
    per_symbol = {}

    for symbol, date_bars in data.items():
        if len(date_bars) < TRAIN_WINDOW + 1:
            continue

        # Collect all trade-level data
        all_trades = []  # (direction, ret_bps, spread_at_entry)
        for i in range(TRAIN_WINDOW, len(date_bars)):
            train = [b for _, b in date_bars[i - TRAIN_WINDOW:i]]
            params = compute_zscore_params_2f(train)
            _, test_bars = date_bars[i]
            test_bars = apply_signal_2f(test_bars, params)

            prices = test_bars["midprice_last"].values
            directions = test_bars["direction"].values
            spreads = test_bars["spread_bps_last"].values
            n = len(prices)
            for j in range(n - horizon):
                d = directions[j]
                if d == 0 or prices[j] <= 0 or np.isnan(prices[j]) or np.isnan(prices[j + horizon]):
                    continue
                ret = (prices[j + horizon] - prices[j]) / prices[j] * 1e4
                sp = spreads[j] if np.isfinite(spreads[j]) else 0
                all_trades.append((d, ret, sp))

        if not all_trades:
            continue

        # Taker baseline
        taker_pnls = [d * r - taker_fee for d, r, _ in all_trades]
        taker_net = float(np.mean(taker_pnls))

        # Maker at each fill rate
        fr_results = {}
        for fr in fill_rates:
            rng = np.random.default_rng(42)
            maker_pnls = []
            for d, ret, sp in all_trades:
                if rng.random() > fr:
                    continue  # Not filled
                # Maker entry is better by half-spread
                improved_ret = d * ret + sp / 2.0
                maker_pnls.append(improved_ret - maker_fee)

            if maker_pnls:
                fr_results[f"{int(fr*100)}%"] = {
                    "n_filled": len(maker_pnls),
                    "net_bps": round(float(np.mean(maker_pnls)), 3),
                    "total_pnl": round(float(np.sum(maker_pnls)), 1),
                }

        # Find breakeven fill rate
        breakeven_fr = None
        for test_fr in np.arange(0.10, 1.01, 0.05):
            rng = np.random.default_rng(42)
            m_pnls = []
            for d, ret, sp in all_trades:
                if rng.random() > test_fr:
                    continue
                m_pnls.append(d * ret + sp / 2.0 - maker_fee)
            if m_pnls and np.mean(m_pnls) >= taker_net:
                breakeven_fr = round(float(test_fr), 2)
                break

        per_symbol[symbol] = {
            "taker_net_bps": round(taker_net, 3),
            "n_trades": len(all_trades),
            "fill_rate_results": fr_results,
            "breakeven_fill_rate": breakeven_fr,
        }

    # Verdict
    btc = per_symbol.get("BTC", {})
    be = btc.get("breakeven_fill_rate")
    fr50 = btc.get("fill_rate_results", {}).get("50%", {})

    if be is not None and be <= 0.50:
        verdict = "CONFIRMED"
    elif be is not None and be > 0.90:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    return HypothesisResult(
        id="H6", name="Maker execution captures spread + directional edge",
        verdict=verdict,
        metrics=per_symbol,
        criteria={"confirmed": "maker > taker even at 50% fill",
                  "rejected": "breakeven fill rate > 90%"},
    )


# ── Report / CLI ─────────────────────────────────────────────────────────

HYPOTHESIS_RUNNERS = {
    "H1": run_h1,
    "H2": run_h2,
    "H3": run_h3,
    "H4": run_h4,
    "H5": run_h5,
    "H6": run_h6,
}


def print_result(r: HypothesisResult):
    tag = {"CONFIRMED": "+", "REJECTED": "x", "INCONCLUSIVE": "?"}[r.verdict]
    print(f"\n  [{tag}] {r.id}: {r.name} — {r.verdict}")
    for sym, m in r.metrics.items():
        if isinstance(m, dict):
            parts = [f"{k}={v}" for k, v in m.items()
                     if not isinstance(v, dict) and k != "per_date"]
            print(f"      {sym}: {', '.join(parts[:6])}")


def main():
    parser = argparse.ArgumentParser(description="MF Liquidity Signal: Hypothesis Suite")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--hypotheses", type=str, default="H1,H2,H3,H4,H5,H6")
    parser.add_argument("--fee-model", choices=list(FEE_MODELS.keys()), default="binance_vip9")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    fees = FEE_MODELS[args.fee_model]
    selected = [h.strip().upper() for h in args.hypotheses.split(",")]

    print(f"Loading data from {data_dir}...")
    data = load_all_data(data_dir, args.symbols)
    print()

    results = []
    for h_id in selected:
        if h_id not in HYPOTHESIS_RUNNERS:
            print(f"Unknown hypothesis: {h_id}")
            continue
        print(f"{'=' * 60}")
        print(f"  Running {h_id}...")
        runner = HYPOTHESIS_RUNNERS[h_id]

        if h_id in ("H2", "H5"):
            r = runner(data, fee_bps=fees["taker"])
        elif h_id == "H6":
            r = runner(data, taker_fee=fees["taker"], maker_fee=fees["maker"])
        else:
            r = runner(data)

        print_result(r)
        results.append(r)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        tag = {"CONFIRMED": "+", "REJECTED": "x", "INCONCLUSIVE": "?"}[r.verdict]
        print(f"  [{tag}] {r.id}: {r.verdict}")

    confirmed = [r.id for r in results if r.verdict == "CONFIRMED"]
    rejected = [r.id for r in results if r.verdict == "REJECTED"]
    inconclusive = [r.id for r in results if r.verdict == "INCONCLUSIVE"]
    print(f"\n  Confirmed: {', '.join(confirmed) or 'none'}")
    print(f"  Rejected:  {', '.join(rejected) or 'none'}")
    print(f"  Inconclusive: {', '.join(inconclusive) or 'none'}")

    if args.save:
        report = {
            "title": "MF Liquidity Signal: Hypothesis Suite",
            "generated": datetime.now(timezone.utc).isoformat(),
            "data": {
                "n_dates": {s: len(db) for s, db in data.items()},
                "symbols": args.symbols,
                "fee_model": args.fee_model,
            },
            "summary": {
                "confirmed": confirmed,
                "rejected": rejected,
                "inconclusive": inconclusive,
            },
            "hypotheses": {r.id: asdict(r) for r in results},
        }
        out = Path("reports/mf_hypothesis_suite.json")
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved: {out}")


if __name__ == "__main__":
    main()
