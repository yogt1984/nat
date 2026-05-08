"""
Cost-Aware Position Sizer (Alpha Roadmap Step 3).

Takes a combined signal z(t) from Step 2, applies a trade filter
that only allows trades when E[gain] > 1.5x transaction cost, and
outputs sized positions p(t) in [-1, +1].

Quality Gate G3:
  - Trade count drops >= 50% vs unfiltered
  - Net return INCREASES vs unfiltered
  - Mean holding time > 2 hours
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.costs import CostModel, hyperliquid_taker

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PositionResult:
    """Output of the position sizing step."""
    n_bars: int
    n_trades_unfiltered: int
    n_trades_filtered: int
    trade_reduction_pct: float
    mean_holding_bars: float
    mean_holding_hours: float
    cost_threshold_multiplier: float
    ramp_bars: int
    gate_trade_reduction_pass: bool  # >= 50% reduction
    gate_holding_time_pass: bool     # > 2 hours mean hold
    gate_pass: bool


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def count_trades(positions: np.ndarray) -> int:
    """Count number of position changes (trades)."""
    diff = np.diff(positions)
    return int(np.sum(np.abs(diff) > 1e-8))


def mean_holding_bars(positions: np.ndarray) -> float:
    """Average number of bars a position is held before changing."""
    changes = np.where(np.abs(np.diff(positions)) > 1e-8)[0]
    if len(changes) < 2:
        return float(len(positions))
    durations = np.diff(changes)
    return float(np.mean(durations))


def apply_trade_filter(
    signal: np.ndarray,
    ic_estimate: float,
    return_vol: float,
    cost_model: CostModel,
    horizon_bars: int = 16,
    cost_multiplier: float = 1.5,
) -> np.ndarray:
    """Filter trades where E[gain] < cost_multiplier * transaction cost.

    E[gain] = |z(t) - z(t-1)| * IC * vol(r) * sqrt(horizon)
    cost = round_trip_cost (in fraction)
    Only trade when E[gain] > cost * cost_multiplier.
    """
    position = np.full_like(signal, np.nan)
    cost = cost_model.round_trip_cost_bps / 10_000.0  # Convert to fraction

    for t in range(len(signal)):
        if not np.isfinite(signal[t]):
            position[t] = position[t - 1] if t > 0 and np.isfinite(position[t - 1]) else 0.0
            continue

        if t == 0:
            position[t] = signal[t]
            continue

        prev = position[t - 1] if np.isfinite(position[t - 1]) else 0.0
        delta = abs(signal[t] - prev)
        expected_gain = delta * abs(ic_estimate) * return_vol * np.sqrt(horizon_bars)

        if expected_gain > cost * cost_multiplier:
            position[t] = signal[t]
        else:
            position[t] = prev  # Hold current position

    return position


def apply_kelly_sizing(
    position: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Scale positions and clip to [-1, +1]."""
    return np.clip(position * scale, -1.0, 1.0)


def apply_ramp_up(
    position: np.ndarray,
    ramp_bars: int = 2880,  # ~30 days at 96 bars/day
    ramp_fraction: float = 0.5,
) -> np.ndarray:
    """Scale down positions during ramp-up period."""
    result = position.copy()
    ramp_end = min(ramp_bars, len(result))
    result[:ramp_end] *= ramp_fraction
    return result


def evaluate_quality_gate(
    signal: np.ndarray,
    position: np.ndarray,
    bar_minutes: float = 15.0,
) -> PositionResult:
    """Run quality gate G3."""
    n_unfiltered = count_trades(signal)
    n_filtered = count_trades(position)
    reduction = (1.0 - n_filtered / max(n_unfiltered, 1)) * 100.0

    holding = mean_holding_bars(position)
    holding_hours = holding * bar_minutes / 60.0

    gate_reduction = reduction >= 50.0
    gate_holding = holding_hours > 2.0

    return PositionResult(
        n_bars=len(signal),
        n_trades_unfiltered=n_unfiltered,
        n_trades_filtered=n_filtered,
        trade_reduction_pct=reduction,
        mean_holding_bars=holding,
        mean_holding_hours=holding_hours,
        cost_threshold_multiplier=1.5,
        ramp_bars=2880,
        gate_trade_reduction_pass=gate_reduction,
        gate_holding_time_pass=gate_holding,
        gate_pass=gate_reduction and gate_holding,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_position_sizing(
    signal: np.ndarray,
    ic_estimate: float,
    return_vol: float,
    horizon_bars: int = 16,
    cost_model: Optional[CostModel] = None,
    cost_multiplier: float = 1.5,
    scale: float = 1.0,
    ramp_bars: int = 2880,
    ramp_fraction: float = 0.5,
    bar_minutes: float = 15.0,
    output: Optional[str | Path] = None,
) -> tuple[np.ndarray, PositionResult]:
    """Full Step 3 pipeline: filter → size → ramp → gate."""
    if cost_model is None:
        cost_model = hyperliquid_taker()

    # 1. Trade filter
    print(f"  Applying trade filter (IC={ic_estimate:.4f}, vol={return_vol:.4f}, "
          f"cost={cost_model.round_trip_cost_bps:.1f}bps, multiplier={cost_multiplier}x)...")
    position = apply_trade_filter(
        signal, ic_estimate, return_vol, cost_model,
        horizon_bars=horizon_bars, cost_multiplier=cost_multiplier,
    )

    # 2. Kelly sizing
    position = apply_kelly_sizing(position, scale=scale)

    # 3. Ramp-up
    position = apply_ramp_up(position, ramp_bars=ramp_bars, ramp_fraction=ramp_fraction)

    # 4. Quality gate
    result = evaluate_quality_gate(signal, position, bar_minutes=bar_minutes)
    result.cost_threshold_multiplier = cost_multiplier
    result.ramp_bars = ramp_bars

    def _g(passed):
        return "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"

    print(f"\n  Gate G3 Results:")
    print(f"    Trades: {result.n_trades_unfiltered} → {result.n_trades_filtered} "
          f"({result.trade_reduction_pct:.0f}% reduction)  [{_g(result.gate_trade_reduction_pass)}]")
    print(f"    Mean holding: {result.mean_holding_hours:.1f}h "
          f"({result.mean_holding_bars:.0f} bars)  [{_g(result.gate_holding_time_pass)}]")
    print(f"    Overall: [{_g(result.gate_pass)}]")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\n  Saved result to {out_path}")

    return position, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cost-Aware Position Sizer (Alpha Roadmap Step 3)",
    )
    parser.add_argument("--signal", required=True, help="Path to signal .npy file")
    parser.add_argument("--ic", type=float, required=True, help="Estimated IC of combined signal")
    parser.add_argument("--vol", type=float, required=True, help="Return volatility")
    parser.add_argument("--horizon-bars", type=int, default=16, help="Forward horizon in bars")
    parser.add_argument("--cost-multiplier", type=float, default=1.5, help="Safety margin on cost")
    parser.add_argument("--scale", type=float, default=1.0, help="Kelly scale factor")
    parser.add_argument("--ramp-bars", type=int, default=2880, help="Ramp-up period in bars")
    parser.add_argument("--bar-minutes", type=float, default=15.0, help="Minutes per bar")
    parser.add_argument("--output", default="reports/alpha_position.json")
    args = parser.parse_args()

    signal = np.load(args.signal)
    run_position_sizing(
        signal=signal,
        ic_estimate=args.ic,
        return_vol=args.vol,
        horizon_bars=args.horizon_bars,
        cost_multiplier=args.cost_multiplier,
        scale=args.scale,
        ramp_bars=args.ramp_bars,
        bar_minutes=args.bar_minutes,
        output=args.output,
    )


if __name__ == "__main__":
    main()
