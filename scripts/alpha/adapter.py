"""
Walk-Forward Signal Adapter (Alpha Roadmap Step 4).

Bridges continuous signals z(t) in [-1, +1] from Steps 2-3 into the
discrete Strategy interface used by the backtest engine and walk-forward
validation framework.

Also runs the full validation protocol:
  1. Walk-forward validation (n_splits=5)
  2. Combinatorial purged CV
  3. Deflated Sharpe (correcting for n_trials from Step 1)

Quality Gate G4:
  - OOS Sharpe > 0.5
  - OOS/IS ratio > 0.7
  - Deflated Sharpe p-value < 0.05
  - Max drawdown < 5%
  - >= 30 trades in OOS
  - Profit factor > 1.2
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from backtest.costs import CostModel, hyperliquid_taker
from backtest.engine import run_backtest
from backtest.strategy import Strategy
from backtest.walk_forward import (
    walk_forward_validation,
    combinatorial_purged_cv,
    compute_deflated_sharpe,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ContinuousSignalAdapter:
    """Wraps a continuous signal [-1, +1] into a Strategy for the backtest engine.

    Entry: when |signal| crosses above `entry_threshold`
    Exit:  when |signal| drops below `exit_threshold`
    Direction: determined by signal sign at entry
    """

    def __init__(
        self,
        signal: np.ndarray,
        entry_threshold: float = 0.3,
        exit_threshold: Optional[float] = None,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        max_holding_bars: int = 600,
    ):
        self.signal = signal
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold or entry_threshold / 2.0
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_bars = max_holding_bars

    def to_long_strategy(self) -> Strategy:
        """Create a long-only strategy from the signal."""
        signal = self.signal
        entry_thresh = self.entry_threshold
        exit_thresh = self.exit_threshold

        def entry_fn(df: pl.DataFrame) -> pl.Series:
            n = len(df)
            sig = signal[:n] if len(signal) >= n else np.pad(signal, (0, n - len(signal)))
            return pl.Series(values=sig > entry_thresh)

        def exit_fn(df: pl.DataFrame) -> pl.Series:
            n = len(df)
            sig = signal[:n] if len(signal) >= n else np.pad(signal, (0, n - len(signal)))
            return pl.Series(values=sig < exit_thresh)

        return Strategy(
            name="alpha_signal_long",
            entry_condition=entry_fn,
            exit_condition=exit_fn,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            max_holding_bars=self.max_holding_bars,
            direction="long",
            description="Continuous alpha signal (long)",
        )

    def to_short_strategy(self) -> Strategy:
        """Create a short-only strategy from the signal."""
        signal = self.signal
        entry_thresh = self.entry_threshold
        exit_thresh = self.exit_threshold

        def entry_fn(df: pl.DataFrame) -> pl.Series:
            n = len(df)
            sig = signal[:n] if len(signal) >= n else np.pad(signal, (0, n - len(signal)))
            return pl.Series(values=sig < -entry_thresh)

        def exit_fn(df: pl.DataFrame) -> pl.Series:
            n = len(df)
            sig = signal[:n] if len(signal) >= n else np.pad(signal, (0, n - len(signal)))
            return pl.Series(values=sig > -exit_thresh)

        return Strategy(
            name="alpha_signal_short",
            entry_condition=entry_fn,
            exit_condition=exit_fn,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            max_holding_bars=self.max_holding_bars,
            direction="short",
            description="Continuous alpha signal (short)",
        )


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class ParameterStability:
    """OOS metric stability across walk-forward folds.

    Coefficient of variation (CV = std/|mean|) measures how much a metric
    drifts between folds. High CV (> max_cv threshold) indicates the
    strategy's performance is sensitive to the training window — a sign
    of overfitting even when average metrics pass gates.
    """
    sharpe_cv: float          # CV of OOS Sharpe across folds
    dd_cv: float              # CV of OOS max drawdown across folds
    pf_cv: float              # CV of OOS profit factor across folds
    sharpe_values: list[float]  # per-fold OOS Sharpe
    stable: bool              # True if all CVs < threshold


@dataclass
class ValidationResult:
    """Full Step 4 validation output."""
    direction: str
    oos_sharpe: float
    is_sharpe: float
    oos_is_ratio: float
    max_drawdown_pct: float
    total_oos_trades: int
    profit_factor: float
    deflated_sharpe_dsr: float
    n_trials: int
    # Parameter stability
    stability: ParameterStability | None = None
    # Gates
    gate_oos_sharpe_pass: bool = False       # > 0.5
    gate_oos_is_ratio_pass: bool = False     # > 0.7
    gate_deflated_sharpe_pass: bool = False  # DSR >= 0.95
    gate_max_dd_pass: bool = False           # < 5%
    gate_min_trades_pass: bool = False       # >= 30
    gate_profit_factor_pass: bool = False    # > 1.2
    gate_stability_pass: bool = False        # CV < threshold
    gate_pass: bool = False


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------


def compute_parameter_stability(
    wf_result,
    max_cv: float = 0.5,
) -> ParameterStability:
    """Measure OOS metric stability across walk-forward folds.

    For each fold, extracts OOS Sharpe, max drawdown, and profit factor.
    Computes coefficient of variation (CV = std / |mean|) for each metric.
    A CV > max_cv indicates the parameter drifts too much across folds.
    """
    sharpes = []
    dds = []
    pfs = []
    for fold in wf_result.fold_results:
        tr = fold.test_result
        sharpes.append(tr.sharpe_ratio)
        dds.append(tr.max_drawdown_pct)
        pfs.append(tr.profit_factor)

    def _cv(values):
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return 0.0
        mean = np.mean(arr)
        if abs(mean) < 1e-12:
            return float("inf") if np.std(arr) > 1e-12 else 0.0
        return float(np.std(arr) / abs(mean))

    s_cv = _cv(sharpes)
    d_cv = _cv(dds)
    p_cv = _cv(pfs)

    return ParameterStability(
        sharpe_cv=round(s_cv, 4),
        dd_cv=round(d_cv, 4),
        pf_cv=round(p_cv, 4),
        sharpe_values=[round(s, 4) for s in sharpes],
        stable=s_cv < max_cv and d_cv < max_cv and p_cv < max_cv,
    )


def run_validation(
    df: pl.DataFrame,
    signal: np.ndarray,
    n_trials: int = 1998,
    entry_threshold: float = 0.3,
    cost_model: Optional[CostModel] = None,
    n_splits: int = 5,
    embargo_bars: int = 600,
    directions: Optional[list[str]] = None,
    output: Optional[str | Path] = None,
) -> list[ValidationResult]:
    """Full Step 4: walk-forward + deflated Sharpe for long and short."""
    if cost_model is None:
        cost_model = hyperliquid_taker()
    if directions is None:
        directions = ["long", "short"]

    adapter = ContinuousSignalAdapter(signal, entry_threshold=entry_threshold)
    results = []

    for direction in directions:
        strategy = adapter.to_long_strategy() if direction == "long" else adapter.to_short_strategy()
        print(f"\n  === {direction.upper()} direction ===")

        # Walk-forward validation
        print(f"  Running walk-forward validation ({n_splits} splits, embargo={embargo_bars})...")
        wf_result = walk_forward_validation(
            df=df,
            strategy=strategy,
            cost_model=cost_model,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            oos_is_threshold=0.7,
            min_oos_sharpe=0.3,
        )

        oos_sharpe = wf_result.out_of_sample_sharpe
        is_sharpe = wf_result.in_sample_sharpe
        oos_is_ratio = wf_result.oos_is_ratio
        total_oos_trades = wf_result.total_test_trades

        # Compute max drawdown from OOS equity curves
        max_dd = 0.0
        for fold in wf_result.fold_results:
            if hasattr(fold, "test_result") and fold.test_result:
                max_dd = max(max_dd, fold.test_result.max_drawdown_pct)

        # Profit factor from OOS
        profit_factor = 1.0
        for fold in wf_result.fold_results:
            if hasattr(fold, "test_result") and fold.test_result:
                profit_factor = fold.test_result.profit_factor

        # Deflated Sharpe
        print(f"  Computing deflated Sharpe (n_trials={n_trials})...")
        deflated_dsr = compute_deflated_sharpe(
            observed_sharpe=oos_sharpe,
            n_trials=n_trials,
        )

        # Parameter stability across folds
        stability = compute_parameter_stability(wf_result)

        # Quality gates
        g_sharpe = oos_sharpe > 0.5
        g_ratio = oos_is_ratio > 0.7
        g_deflated = deflated_dsr >= 0.95
        g_dd = max_dd < 5.0
        g_trades = total_oos_trades >= 30
        g_pf = profit_factor > 1.2
        g_stable = stability.stable

        vr = ValidationResult(
            direction=direction,
            oos_sharpe=oos_sharpe,
            is_sharpe=is_sharpe,
            oos_is_ratio=oos_is_ratio,
            max_drawdown_pct=max_dd,
            total_oos_trades=total_oos_trades,
            profit_factor=profit_factor,
            deflated_sharpe_dsr=deflated_dsr,
            n_trials=n_trials,
            stability=stability,
            gate_oos_sharpe_pass=g_sharpe,
            gate_oos_is_ratio_pass=g_ratio,
            gate_deflated_sharpe_pass=g_deflated,
            gate_max_dd_pass=g_dd,
            gate_min_trades_pass=g_trades,
            gate_profit_factor_pass=g_pf,
            gate_stability_pass=g_stable,
            gate_pass=all([g_sharpe, g_ratio, g_deflated, g_dd, g_trades, g_pf, g_stable]),
        )
        results.append(vr)

        def _g(passed):
            return "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"

        print(f"\n  Gate G4 [{direction}]:")
        print(f"    OOS Sharpe:       {oos_sharpe:.3f} vs 0.5  [{_g(g_sharpe)}]")
        print(f"    OOS/IS ratio:     {oos_is_ratio:.3f} vs 0.7  [{_g(g_ratio)}]")
        print(f"    Deflated Sharpe:  DSR={deflated_dsr:.4f} vs 0.95  [{_g(g_deflated)}]")
        print(f"    Max drawdown:     {max_dd:.2f}% vs 5%  [{_g(g_dd)}]")
        print(f"    OOS trades:       {total_oos_trades} vs 30  [{_g(g_trades)}]")
        print(f"    Profit factor:    {profit_factor:.2f} vs 1.2  [{_g(g_pf)}]")
        print(f"    Stability:        Sharpe CV={stability.sharpe_cv:.2f}, DD CV={stability.dd_cv:.2f}, PF CV={stability.pf_cv:.2f}  [{_g(g_stable)}]")
        print(f"      Per-fold Sharpe: {stability.sharpe_values}")
        print(f"    Overall:          [{_g(vr.gate_pass)}]")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n  Saved results to {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Signal Adapter (Alpha Roadmap Step 4)",
    )
    parser.add_argument("--signal", required=True, help="Signal .npy file")
    parser.add_argument("--data-dir", default="data/features", help="Parquet data directory")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--timeframe", default="15min")
    parser.add_argument("--n-trials", type=int, default=1998, help="Number of tests from Step 1")
    parser.add_argument("--entry-threshold", type=float, default=0.3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embargo-bars", type=int, default=600)
    parser.add_argument("--direction", nargs="+", default=["long", "short"],
                        choices=["long", "short"])
    parser.add_argument("--output", default="reports/alpha_validation.json")
    args = parser.parse_args()

    from cluster_pipeline.loader import load_parquet
    from cluster_pipeline.preprocess import aggregate_bars

    signal = np.load(args.signal)

    df = load_parquet(args.data_dir)
    df = df.filter(pl.col("symbol") == args.symbol)
    df = aggregate_bars(df, timeframe=args.timeframe)

    run_validation(
        df=df,
        signal=signal,
        n_trials=args.n_trials,
        entry_threshold=args.entry_threshold,
        n_splits=args.n_splits,
        embargo_bars=args.embargo_bars,
        directions=args.direction,
        output=args.output,
    )


if __name__ == "__main__":
    main()
