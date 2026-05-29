"""
Multi-Frequency Integration (Alpha Roadmap Step 6).

Combines micro signal (15min, Steps 1-5) with macro signal (daily OHLCV).
Macro filter gates positions via SMA(50)/SMA(200) crossover.

Quality Gate G6:
  - Composite Sharpe > max(macro_sharpe, micro_sharpe)
  - Composite max DD < min(macro_max_dd, micro_max_dd)

Usage:
    python -m alpha.multi_freq --data data/features --symbol BTC
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = ROOT / "reports"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MacroFilter:
    """Macro regime filter state per bar."""
    long_allowed: bool
    short_allowed: bool
    trend_strength: float  # 0-1, distance from SMA crossover


@dataclass
class MultiFreqResult:
    """Output of multi-frequency integration."""
    n_bars: int
    n_bars_long_allowed: int
    n_bars_short_allowed: int
    n_bars_flat_only: int
    micro_sharpe: float
    macro_sharpe: float
    composite_sharpe: float
    micro_max_dd: float
    macro_max_dd: float
    composite_max_dd: float
    gate_sharpe_improves: bool
    gate_dd_improves: bool
    gate_pass: bool


# ---------------------------------------------------------------------------
# Macro signal
# ---------------------------------------------------------------------------


def compute_macro_filter(
    daily_df: pd.DataFrame,
    fast_period: int = 50,
    slow_period: int = 200,
) -> pd.DataFrame:
    """
    Compute macro trend filter from daily OHLCV.

    Rules:
      - SMA(50) > SMA(200): long_allowed=True, short_allowed=False
      - SMA(50) < SMA(200): long_allowed=False, short_allowed=True
      - Within 1% of crossover: both allowed (transition zone)

    Returns DataFrame with columns: timestamp, long_allowed, short_allowed, trend_strength
    """
    close = daily_df["close"].values.astype(float)

    # SMAs
    sma_fast = pd.Series(close).rolling(fast_period, min_periods=fast_period).mean().values
    sma_slow = pd.Series(close).rolling(slow_period, min_periods=slow_period).mean().values

    n = len(close)
    long_allowed = np.zeros(n, dtype=bool)
    short_allowed = np.zeros(n, dtype=bool)
    trend_strength = np.zeros(n, dtype=float)

    for i in range(n):
        if np.isnan(sma_fast[i]) or np.isnan(sma_slow[i]):
            # Before warmup: allow both directions
            long_allowed[i] = True
            short_allowed[i] = True
            continue

        # Distance from crossover as fraction
        dist = (sma_fast[i] - sma_slow[i]) / sma_slow[i]
        trend_strength[i] = min(abs(dist) / 0.05, 1.0)  # normalize to [0,1]

        if dist > 0.01:  # fast above slow by >1%
            long_allowed[i] = True
            short_allowed[i] = False
        elif dist < -0.01:  # fast below slow by >1%
            long_allowed[i] = False
            short_allowed[i] = True
        else:  # transition zone
            long_allowed[i] = True
            short_allowed[i] = True

    result_df = pd.DataFrame({
        "long_allowed": long_allowed,
        "short_allowed": short_allowed,
        "trend_strength": trend_strength,
    })

    if "timestamp" in daily_df.columns:
        result_df["timestamp"] = daily_df["timestamp"]

    return result_df


def align_macro_to_micro(
    macro_filter: pd.DataFrame,
    micro_df: pd.DataFrame,
    micro_timeframe: str = "15min",
) -> np.ndarray:
    """
    Broadcast daily macro filter to micro-frequency bars.

    Returns arrays (long_allowed, short_allowed, trend_strength) aligned
    to micro_df rows. If no timestamp alignment is possible, uses
    sequential daily blocks.

    Returns:
        (n_micro, 3) array: [long_allowed, short_allowed, trend_strength]
    """
    n_micro = len(micro_df)
    n_macro = len(macro_filter)

    from utils.metrics import bars_per_day_for_timeframe
    bars_per_day = bars_per_day_for_timeframe(micro_timeframe)

    result = np.zeros((n_micro, 3))

    for i in range(n_micro):
        day_idx = min(i // bars_per_day, n_macro - 1)
        result[i, 0] = macro_filter["long_allowed"].iloc[day_idx]
        result[i, 1] = macro_filter["short_allowed"].iloc[day_idx]
        result[i, 2] = macro_filter["trend_strength"].iloc[day_idx]

    return result


# ---------------------------------------------------------------------------
# Signal composition
# ---------------------------------------------------------------------------


def apply_macro_gate(
    micro_signal: np.ndarray,
    macro_state: np.ndarray,
) -> np.ndarray:
    """
    Apply macro gate to micro signal.

    macro_state shape: (n, 3) = [long_allowed, short_allowed, trend_strength]

    Rules:
      - If signal > 0 and not long_allowed: set to 0
      - If signal < 0 and not short_allowed: set to 0
      - Scale remaining by trend_strength
    """
    gated = micro_signal.copy()

    long_ok = macro_state[:, 0].astype(bool)
    short_ok = macro_state[:, 1].astype(bool)
    strength = macro_state[:, 2]

    # Gate
    gated[(gated > 0) & ~long_ok] = 0.0
    gated[(gated < 0) & ~short_ok] = 0.0

    # Scale by trend strength
    gated *= (0.5 + 0.5 * strength)  # scale from 50% to 100%

    return gated


def profit_sensitive_exit(
    signal: np.ndarray,
    prices: np.ndarray,
    entry_threshold: float = 0.3,
    tighten_factor: float = 0.5,
) -> np.ndarray:
    """
    Tighten exit threshold when unrealized PnL is positive.

    When in a position with positive PnL, reduces the exit threshold
    so we lock in profits faster.
    """
    adjusted = signal.copy()
    position = 0.0
    entry_price = 0.0

    for i in range(len(signal)):
        if abs(position) < 0.01:
            # No position — check for entry
            if abs(signal[i]) > entry_threshold:
                position = signal[i]
                entry_price = prices[i]
        else:
            # In position — check unrealized PnL
            if entry_price > 0:
                pnl = (prices[i] - entry_price) / entry_price * np.sign(position)
            else:
                pnl = 0.0

            if pnl > 0:
                # Positive PnL — tighten exit
                exit_thresh = entry_threshold * tighten_factor
                if abs(signal[i]) < exit_thresh:
                    adjusted[i] = 0.0
                    position = 0.0
            else:
                # Negative PnL — use normal threshold
                if abs(signal[i]) < entry_threshold * 0.5:
                    adjusted[i] = 0.0
                    position = 0.0

    return adjusted


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------


def _compute_signal_pnl(signal: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Simple signal-weighted PnL: position[t] * return[t+1]."""
    returns = np.diff(prices) / prices[:-1]
    position = signal[:-1]
    pnl = position * returns
    return pnl


def _sharpe(pnl: np.ndarray, bars_per_day: float = 96) -> float:
    """Annualized Sharpe — aggregates intraday bars to daily first."""
    from utils.metrics import sharpe_daily
    bpd = int(bars_per_day)
    n_full_days = len(pnl) // bpd
    if n_full_days < 2:
        return 0.0
    trimmed = pnl[:n_full_days * bpd]
    daily_pnl = trimmed.reshape(n_full_days, bpd).sum(axis=1)
    return sharpe_daily(daily_pnl)


def _max_dd(pnl: np.ndarray) -> float:
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(np.max(dd)) if len(dd) > 0 else 0.0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_multi_freq(
    data_dir: str = "data/features",
    symbol: str = "BTC",
    timeframe: str = "15min",
    signal_path: Optional[str] = None,
    output: str = "reports/alpha_multi_freq.json",
) -> MultiFreqResult:
    """
    Full multi-frequency integration pipeline.

    1. Load micro data + compute proxy signal
    2. Fetch daily macro candles
    3. Compute macro filter
    4. Apply macro gate to micro signal
    5. Apply profit-sensitive exit
    6. Evaluate G6 quality gate
    """
    from data.macro import fetch_candles, add_indicators
    from cluster_pipeline.loader import load_parquet
    from cluster_pipeline.preprocess import aggregate_bars
    from utils.metrics import bars_per_day_for_timeframe

    bpd = bars_per_day_for_timeframe(timeframe)

    # Load micro data
    df = load_parquet(data_dir)
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].reset_index(drop=True)

    bars = aggregate_bars(df, timeframe=timeframe)
    bars_pd = bars.to_pandas() if hasattr(bars, "to_pandas") else bars

    # Price column
    price_col = None
    for c in ["midprice_mean", "close", "mid_price"]:
        if c in bars_pd.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError("No price column found")

    prices = bars_pd[price_col].values

    # Micro signal: use pre-computed or generate proxy
    if signal_path and Path(signal_path).exists():
        with open(signal_path) as f:
            sig_data = json.load(f)
        micro_signal = np.array(sig_data.get("signal", []))
        if len(micro_signal) != len(bars_pd):
            micro_signal = np.zeros(len(bars_pd))
    else:
        # Proxy: momentum signal from returns
        ret = np.zeros(len(prices))
        ret[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        micro_signal = pd.Series(ret).rolling(20, min_periods=1).mean().values
        # Normalize to [-1, 1]
        std = np.std(micro_signal)
        if std > 1e-10:
            micro_signal = np.clip(micro_signal / (3 * std), -1, 1)

    # Fetch macro candles
    daily = fetch_candles(symbol, interval="1d", days=365)
    if daily is None or len(daily) == 0:
        log.warning("No macro data — returning micro-only result")
        pnl = _compute_signal_pnl(micro_signal, prices)
        return MultiFreqResult(
            n_bars=len(bars_pd),
            n_bars_long_allowed=len(bars_pd),
            n_bars_short_allowed=len(bars_pd),
            n_bars_flat_only=0,
            micro_sharpe=_sharpe(pnl, bpd), macro_sharpe=0.0,
            composite_sharpe=_sharpe(pnl, bpd),
            micro_max_dd=_max_dd(pnl), macro_max_dd=0.0,
            composite_max_dd=_max_dd(pnl),
            gate_sharpe_improves=False, gate_dd_improves=False,
            gate_pass=False,
        )

    daily = add_indicators(daily)

    # Compute macro filter
    macro_filter = compute_macro_filter(daily)
    macro_state = align_macro_to_micro(macro_filter, bars_pd, timeframe)

    # Micro-only PnL
    micro_pnl = _compute_signal_pnl(micro_signal, prices)
    micro_s = _sharpe(micro_pnl, bpd)
    micro_dd = _max_dd(micro_pnl)

    # Macro-only PnL (buy-and-hold when long_allowed)
    macro_signal = np.where(macro_state[:, 0], 1.0, -1.0)
    macro_pnl = _compute_signal_pnl(macro_signal, prices)
    macro_s = _sharpe(macro_pnl, bpd)
    macro_dd = _max_dd(macro_pnl)

    # Composite: macro-gated micro + profit-sensitive exit
    gated = apply_macro_gate(micro_signal, macro_state)
    composite = profit_sensitive_exit(gated, prices)
    comp_pnl = _compute_signal_pnl(composite, prices)
    comp_s = _sharpe(comp_pnl, bpd)
    comp_dd = _max_dd(comp_pnl)

    # Count bars
    n_long = int(np.sum(macro_state[:, 0]))
    n_short = int(np.sum(macro_state[:, 1]))
    n_flat = int(np.sum(~macro_state[:, 0].astype(bool) & ~macro_state[:, 1].astype(bool)))

    # G6 quality gate
    gate_sharpe = comp_s > max(micro_s, macro_s)
    gate_dd = comp_dd < min(micro_dd, macro_dd) if micro_dd > 0 and macro_dd > 0 else False

    result = MultiFreqResult(
        n_bars=len(bars_pd),
        n_bars_long_allowed=n_long,
        n_bars_short_allowed=n_short,
        n_bars_flat_only=n_flat,
        micro_sharpe=micro_s,
        macro_sharpe=macro_s,
        composite_sharpe=comp_s,
        micro_max_dd=micro_dd,
        macro_max_dd=macro_dd,
        composite_max_dd=comp_dd,
        gate_sharpe_improves=gate_sharpe,
        gate_dd_improves=gate_dd,
        gate_pass=gate_sharpe and gate_dd,
    )

    # Save
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    log.info(f"Multi-freq: micro_S={micro_s:.2f}, macro_S={macro_s:.2f}, "
             f"composite_S={comp_s:.2f}, G6={'PASS' if result.gate_pass else 'FAIL'}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-frequency integration (Step 6)")
    parser.add_argument("--data", default="data/features")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--timeframe", default="15min")
    parser.add_argument("--signal", default=None, help="Pre-computed signal JSON")
    parser.add_argument("--output", default="reports/alpha_multi_freq.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_multi_freq(
        data_dir=args.data, symbol=args.symbol, timeframe=args.timeframe,
        signal_path=args.signal, output=args.output,
    )
    gate = "PASS" if result.gate_pass else "FAIL"
    print(f"\nG6 Quality Gate: {gate}")
    print(f"  Micro Sharpe:     {result.micro_sharpe:.3f}")
    print(f"  Macro Sharpe:     {result.macro_sharpe:.3f}")
    print(f"  Composite Sharpe: {result.composite_sharpe:.3f}")
    print(f"  Micro MaxDD:      {result.micro_max_dd:.4f}")
    print(f"  Composite MaxDD:  {result.composite_max_dd:.4f}")


if __name__ == "__main__":
    main()
