#!/usr/bin/env python3
"""
Wave 1 Decision Gate — evaluate whether to proceed to Wave 2.

Collects metrics from Wave 1 algorithms (#5 Change-Point Detector, #1 Momentum
Continuation) and applies a 4-case decision matrix.

Usage:
    python scripts/evaluate_wave1_gate.py --data-dir data/features --symbols BTC,ETH,SOL
    python scripts/evaluate_wave1_gate.py --data-dir data/features --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# --- Decision matrix cases ---

CASE_A = "CASE_A"  # Proceed full (Wave 2 all algorithms)
CASE_B = "CASE_B"  # Proceed cautious (skip meta-labeling)
CASE_C = "CASE_C"  # Investigate (tune features, check data)
CASE_D = "CASE_D"  # Stop ML work

CASE_DESCRIPTIONS = {
    CASE_A: "PROCEED TO WAVE 2 (full)",
    CASE_B: "PROCEED TO WAVE 2 (cautious — skip meta-labeling)",
    CASE_C: "INVESTIGATE (tune features, check data quality)",
    CASE_D: "STOP ML WORK (dataset does not support ML alpha)",
}


def evaluate_momentum_gate(oos_sharpe: float, symbols_positive: int) -> str:
    """Apply the 4-case decision matrix for momentum continuation.

    Args:
        oos_sharpe: Out-of-sample Sharpe ratio from walk-forward validation.
        symbols_positive: Number of symbols (out of 3) with positive PnL.

    Returns:
        One of CASE_A, CASE_B, CASE_C, CASE_D.
    """
    if oos_sharpe < 0.0:
        return CASE_D
    if oos_sharpe < 0.5:
        return CASE_C
    if symbols_positive >= 2:
        return CASE_A
    return CASE_B


def evaluate_cpd(cpd_variance: float, cpd_vol_corr: float) -> str:
    """Evaluate CPD independently of the ML gate.

    Args:
        cpd_variance: Variance of alg_cpd_cusum_signal over evaluation period.
        cpd_vol_corr: Spearman correlation of CPD signal with forward volatility.

    Returns:
        "KEEP" or "RETIRE" recommendation.
    """
    if cpd_variance > 0.01 and cpd_vol_corr > 0.10:
        return "KEEP"
    return "RETIRE"


def load_momentum_metadata(model_dir: str = "models/momentum_continuation") -> dict:
    """Load the latest momentum model metadata JSON."""
    model_path = Path(model_dir)
    if not model_path.exists():
        return {}

    json_files = sorted(model_path.glob("*_metadata.json"))
    if not json_files:
        return {}

    with open(json_files[-1]) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Wave 1 Decision Gate")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--symbols", default="BTC,ETH,SOL")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--model-dir", default="models/momentum_continuation")
    # Manual overrides for testing without real data
    parser.add_argument("--oos-sharpe", type=float, default=None)
    parser.add_argument("--symbols-positive", type=int, default=None)
    parser.add_argument("--cpd-variance", type=float, default=None)
    parser.add_argument("--cpd-vol-corr", type=float, default=None)
    args = parser.parse_args()

    symbols = args.symbols.split(",")

    # Collect metrics
    meta = load_momentum_metadata(args.model_dir)
    perf = meta.get("performance_metrics", {})

    oos_sharpe = args.oos_sharpe if args.oos_sharpe is not None else perf.get("avg_sharpe_oos", 0.0)
    oos_auc = perf.get("avg_auc_oos", 0.0)
    oos_is_ratio = perf.get("oos_is_ratio", 0.0)
    symbols_positive = args.symbols_positive if args.symbols_positive is not None else 0
    cpd_variance = args.cpd_variance if args.cpd_variance is not None else 0.0
    cpd_vol_corr = args.cpd_vol_corr if args.cpd_vol_corr is not None else 0.0

    # Evaluate
    momentum_case = evaluate_momentum_gate(oos_sharpe, symbols_positive)
    cpd_decision = evaluate_cpd(cpd_variance, cpd_vol_corr)

    result = {
        "momentum": {
            "oos_sharpe": oos_sharpe,
            "oos_auc": oos_auc,
            "oos_is_ratio": oos_is_ratio,
            "symbols_positive": symbols_positive,
            "symbols_total": len(symbols),
            "case": momentum_case,
            "decision": CASE_DESCRIPTIONS[momentum_case],
        },
        "cpd": {
            "signal_variance": cpd_variance,
            "vol_correlation": cpd_vol_corr,
            "decision": cpd_decision,
        },
        "overall_decision": momentum_case,
    }

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print("WAVE 1 DECISION GATE")
        print("=" * 50)
        print(f"  #1 Momentum OOS AUC:       {oos_auc:.4f}")
        print(f"  #1 Momentum OOS Sharpe:     {oos_sharpe:.2f}")
        print(f"  #1 Symbols positive:        {symbols_positive}/{len(symbols)}")
        print(f"  #5 CPD signal variance:     {cpd_variance:.4f}")
        print(f"  #5 CPD vol correlation:     {cpd_vol_corr:.2f}")
        print("=" * 50)
        print(f"  MOMENTUM: {momentum_case} — {CASE_DESCRIPTIONS[momentum_case]}")
        print(f"  CPD:      {cpd_decision} as gating signal")
        print("=" * 50)

    # Save results
    output_path = Path("data/research/wave1_gate_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    if not args.json_output:
        print(f"\n  Results saved to {output_path}")

    # Exit code: 0 for A/B (proceed), 1 for C/D (hold/stop)
    sys.exit(0 if momentum_case in (CASE_A, CASE_B) else 1)


if __name__ == "__main__":
    main()
