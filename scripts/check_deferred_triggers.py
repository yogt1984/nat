#!/usr/bin/env python3
"""
Check trigger conditions for deferred ML algorithms (#8, #9, #10).

Usage:
    python scripts/check_deferred_triggers.py --data-dir data/features
    python scripts/check_deferred_triggers.py --json

Checks whether conditions for re-evaluating deferred algorithms are met.
Run monthly or after each wave completion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Known ML algorithm names (registered in DAILY_ALGOS)
ML_ALGO_NAMES = [
    "change_point_detector",
    "momentum_continuation",
    "regime_state_machine",
    "mean_reversion_detector",
    "meta_labeling",
    "regime_conditioned_lgbm",
    "knn_retrieval",
]

# Trigger thresholds
DATA_DAYS_THRESHOLD = 60
DEPLOYED_ML_THRESHOLD = 4
SHARPE_DEGRADATION_THRESHOLD = 0.30  # 30% drop from peak


def check_data_volume(data_dir: str) -> tuple[bool, int]:
    """Check if data volume exceeds threshold.

    Returns (triggered, n_days).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return False, 0

    # Count date directories (YYYY-MM-DD format)
    date_dirs = [d for d in data_path.iterdir()
                 if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"]
    n_days = len(date_dirs)
    return n_days >= DATA_DAYS_THRESHOLD, n_days


def check_deployed_ml_count(daily_algos: list[str] | None = None) -> tuple[bool, int]:
    """Check if enough ML algorithms are deployed.

    Returns (triggered, count).
    """
    if daily_algos is None:
        try:
            from alpha.paper_trader_daily import DAILY_ALGOS
            daily_algos = DAILY_ALGOS
        except ImportError:
            return False, 0

    ml_deployed = [a for a in daily_algos if a in ML_ALGO_NAMES]
    count = len(ml_deployed)
    return count >= DEPLOYED_ML_THRESHOLD, count


def check_sharpe_degradation(
    peak_sharpes: dict[str, float] | None = None,
    current_sharpes: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    """Check if any deployed ML model has Sharpe degradation > threshold.

    Returns (triggered, list of degraded algo names).
    """
    if not peak_sharpes or not current_sharpes:
        return False, []

    degraded = []
    for algo, peak in peak_sharpes.items():
        current = current_sharpes.get(algo, 0.0)
        if peak > 0 and (peak - current) / peak > SHARPE_DEGRADATION_THRESHOLD:
            degraded.append(algo)

    return len(degraded) > 0, degraded


def evaluate_triggers(
    data_dir: str = "data/features",
    daily_algos: list[str] | None = None,
    peak_sharpes: dict[str, float] | None = None,
    current_sharpes: dict[str, float] | None = None,
) -> dict:
    """Evaluate all deferred algorithm triggers.

    Returns dict with per-algorithm trigger status.
    """
    data_triggered, n_days = check_data_volume(data_dir)
    ml_triggered, ml_count = check_deployed_ml_count(daily_algos)
    sharpe_triggered, degraded = check_sharpe_degradation(peak_sharpes, current_sharpes)

    return {
        "hmm_emissions": {
            "triggered": data_triggered,
            "reason": f"data={n_days} days, need {DATA_DAYS_THRESHOLD}",
        },
        "stacking_ensemble": {
            "triggered": ml_triggered,
            "reason": f"deployed ML={ml_count}, need {DEPLOYED_ML_THRESHOLD}",
        },
        "online_learner": {
            "triggered": sharpe_triggered,
            "reason": f"degraded={degraded}" if degraded else "no Sharpe degradation detected",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Check deferred algorithm triggers")
    parser.add_argument("--data-dir", default="data/features")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    result = evaluate_triggers(data_dir=args.data_dir)

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        print("DEFERRED ALGORITHM TRIGGERS")
        print("=" * 55)
        labels = {
            "hmm_emissions": "#8 HMM Emissions",
            "stacking_ensemble": "#9 Stacking",
            "online_learner": "#10 Online Learning",
        }
        for key, label in labels.items():
            status = "TRIGGERED" if result[key]["triggered"] else "NOT TRIGGERED"
            reason = result[key]["reason"]
            print(f"  {label:25s} {status:15s} ({reason})")
        print("=" * 55)


if __name__ == "__main__":
    main()
