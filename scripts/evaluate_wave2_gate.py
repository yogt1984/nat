#!/usr/bin/env python3
"""
Wave 2 Decision Gate — evaluate whether to proceed to Wave 3.

Collects OOS metrics from all Wave 2 algorithms (#4 RSM, #2 MR, #3 Meta)
plus Wave 1 (#5 CPD, #1 MC), computes pairwise signal correlations, and
applies a 4-case decision matrix.

Usage:
    python scripts/evaluate_wave2_gate.py --data-dir data/features
    python scripts/evaluate_wave2_gate.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# --- Decision matrix cases ---

CASE_A = "CASE_A"  # Proceed full (Wave 3: #7 + #6)
CASE_B = "CASE_B"  # Proceed partial (Wave 3: #7 only)
CASE_C = "CASE_C"  # Stop, deploy only the positive algo
CASE_D = "CASE_D"  # Stop, ML adds no alpha

CASE_DESCRIPTIONS = {
    CASE_A: "PROCEED TO WAVE 3 (full — #7 regime LightGBM + #6 KNN)",
    CASE_B: "PROCEED TO WAVE 3 (partial — #7 only)",
    CASE_C: "STOP — deploy only positive algorithm(s)",
    CASE_D: "STOP — ML adds no alpha",
}

# ML algorithms evaluated at Wave 2 gate
ML_ALGOS = [
    "change_point_detector",
    "momentum_continuation",
    "regime_state_machine",
    "mean_reversion_detector",
    "meta_labeling",
]

CORRELATION_THRESHOLD = 0.5


def evaluate_wave2(
    n_positive: int,
    max_rho: float,
    n_correlated_pairs: int = 0,
) -> str:
    """Apply the 4-case decision matrix for Wave 2.

    Args:
        n_positive: Number of ML algorithms with positive OOS Sharpe.
        max_rho: Maximum absolute pairwise Spearman correlation of primary signals.
        n_correlated_pairs: Number of pairs with |rho| > CORRELATION_THRESHOLD.

    Returns:
        One of CASE_A, CASE_B, CASE_C, CASE_D.
    """
    if n_positive == 0:
        return CASE_D
    if n_positive == 1:
        return CASE_C
    if n_positive == 2:
        return CASE_B
    # n_positive >= 3
    if n_correlated_pairs == 0:
        return CASE_A
    return CASE_B


def compute_pairwise_correlations(
    signals: dict[str, np.ndarray],
) -> tuple[dict[tuple[str, str], float], float, list[tuple[str, str]]]:
    """Compute pairwise Spearman correlations of primary signals.

    Args:
        signals: Dict mapping algo name -> signal array (same length).

    Returns:
        (rho_matrix, max_abs_rho, correlated_pairs)
    """
    from scipy import stats

    names = sorted(signals.keys())
    rho_matrix = {}
    max_rho = 0.0
    correlated = []

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            sa, sb = signals[a], signals[b]
            # Align on finite values
            mask = np.isfinite(sa) & np.isfinite(sb)
            if mask.sum() < 30:
                rho_matrix[(a, b)] = np.nan
                continue

            rho, _ = stats.spearmanr(sa[mask], sb[mask])
            if np.isnan(rho):
                rho_matrix[(a, b)] = np.nan
                continue

            rho_matrix[(a, b)] = rho
            abs_rho = abs(rho)
            if abs_rho > max_rho:
                max_rho = abs_rho
            if abs_rho > CORRELATION_THRESHOLD:
                correlated.append((a, b))

    return rho_matrix, max_rho, correlated


def load_algo_metadata(algo_name: str, model_dir: str = "models") -> dict:
    """Load the latest metadata JSON for an algorithm."""
    algo_path = Path(model_dir) / algo_name
    if not algo_path.exists():
        return {}

    json_files = sorted(algo_path.glob("*_metadata.json"))
    if not json_files:
        return {}

    with open(json_files[-1]) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Wave 2 Decision Gate")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--json", action="store_true", dest="json_output")
    # Manual overrides for testing without real data
    parser.add_argument("--n-positive", type=int, default=None)
    parser.add_argument("--max-rho", type=float, default=None)
    parser.add_argument("--n-correlated", type=int, default=None)
    args = parser.parse_args()

    algo_results = []
    n_positive = 0

    for algo_name in ML_ALGOS:
        meta = load_algo_metadata(algo_name, args.model_dir)
        perf = meta.get("performance_metrics", {})
        oos_sharpe = perf.get("avg_sharpe_oos", 0.0)
        oos_auc = perf.get("avg_auc_oos", 0.0)

        is_positive = oos_sharpe > 0.0 or oos_auc > 0.52
        if is_positive:
            n_positive += 1

        algo_results.append({
            "name": algo_name,
            "oos_sharpe": oos_sharpe,
            "oos_auc": oos_auc,
            "positive": is_positive,
            "has_model": bool(meta),
        })

    # Use manual overrides if provided
    if args.n_positive is not None:
        n_positive = args.n_positive
    max_rho = args.max_rho if args.max_rho is not None else 0.0
    n_correlated = args.n_correlated if args.n_correlated is not None else 0

    decision = evaluate_wave2(n_positive, max_rho, n_correlated)

    result = {
        "algorithms": algo_results,
        "n_positive": n_positive,
        "n_total": len(ML_ALGOS),
        "max_pairwise_rho": max_rho,
        "n_correlated_pairs": n_correlated,
        "case": decision,
        "decision": CASE_DESCRIPTIONS[decision],
    }

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print("WAVE 2 DECISION GATE")
        print("=" * 55)
        for ar in algo_results:
            status = "+" if ar["positive"] else "-"
            model = "model" if ar["has_model"] else "no model"
            print(f"  [{status}] {ar['name']:30s} AUC={ar['oos_auc']:.4f} ({model})")
        print("-" * 55)
        print(f"  Algorithms with positive OOS: {n_positive}/{len(ML_ALGOS)}")
        print(f"  Max pairwise |rho|:           {max_rho:.3f}")
        print(f"  Correlated pairs (>{CORRELATION_THRESHOLD}):   {n_correlated}")
        print("=" * 55)
        print(f"  DECISION: {decision} — {CASE_DESCRIPTIONS[decision]}")
        print("=" * 55)

    # Save results
    output_path = Path("data/research/wave2_gate_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    if not args.json_output:
        print(f"\n  Results saved to {output_path}")

    sys.exit(0 if decision in (CASE_A, CASE_B) else 1)


if __name__ == "__main__":
    main()
