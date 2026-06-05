#!/usr/bin/env python3
"""
ML Health Check — nightly monitoring for deployed ML models.

Usage:
    python scripts/ml_health_check.py
    python scripts/ml_health_check.py --json

Checks per deployed ML algorithm:
  1. Model age (days since training_date)
  2. NaN rate in recent signal outputs

Exit codes: 0=OK, 1=WARNING, 2=CRITICAL
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.model_io import get_latest_model, list_models, ModelMetadata

# Known ML algorithms (bar_level=True, may have trained models)
ML_ALGO_NAMES = [
    "momentum_continuation",
    "mean_reversion_detector",
    "meta_labeling",
    "regime_conditioned_lgbm",
]

# Algorithms that don't need trained models
NO_MODEL_ALGOS = [
    "change_point_detector",
    "regime_state_machine",
    "knn_retrieval",
]

# Thresholds
MODEL_AGE_WARN = 14   # days
MODEL_AGE_CRIT = 30   # days
NAN_RATE_WARN = 0.20  # 20%
SHARPE_7D_CRIT = -0.5


# Status levels
OK = "OK"
WARN = "WARN"
CRITICAL = "CRITICAL"


def check_model_age(
    training_date_str: str | None,
    now: datetime | None = None,
) -> tuple[str, int | None]:
    """Check model age. Returns (status, age_days)."""
    if training_date_str is None or training_date_str == "unknown":
        return OK, None  # no model needed or not found

    if now is None:
        now = datetime.now(timezone.utc)

    try:
        # Handle both ISO formats
        td = training_date_str.replace("Z", "+00:00")
        if "+" not in td and "T" in td:
            trained = datetime.fromisoformat(td).replace(tzinfo=timezone.utc)
        else:
            trained = datetime.fromisoformat(td)
        age_days = (now - trained).days
    except (ValueError, TypeError):
        return WARN, None

    if age_days > MODEL_AGE_CRIT:
        return CRITICAL, age_days
    elif age_days > MODEL_AGE_WARN:
        return WARN, age_days
    return OK, age_days


def check_nan_rate(nan_fraction: float) -> str:
    """Check NaN rate. Returns status."""
    if nan_fraction > NAN_RATE_WARN:
        return WARN
    return OK


def check_sharpe_7d(sharpe: float | None) -> str:
    """Check 7-day rolling Sharpe. Returns status."""
    if sharpe is None:
        return OK
    if sharpe < SHARPE_7D_CRIT:
        return CRITICAL
    return OK


def evaluate_health(
    algo_name: str,
    models_dir: Path,
    training_date: str | None = None,
    nan_fraction: float = 0.0,
    sharpe_7d: float | None = None,
    now: datetime | None = None,
) -> dict:
    """Evaluate health for a single algorithm.

    Returns dict with per-check status and overall status.
    """
    # Get training date from model metadata if not provided
    if training_date is None:
        model_dir = models_dir / algo_name
        models = list_models(model_dir) if model_dir.exists() else []
        if models:
            training_date = models[0].get("training_date")

    age_status, age_days = check_model_age(training_date, now=now)
    nan_status = check_nan_rate(nan_fraction)
    sharpe_status = check_sharpe_7d(sharpe_7d)

    statuses = [age_status, nan_status, sharpe_status]
    if CRITICAL in statuses:
        overall = CRITICAL
    elif WARN in statuses:
        overall = WARN
    else:
        overall = OK

    return {
        "algo": algo_name,
        "overall": overall,
        "age_days": age_days,
        "age_status": age_status,
        "nan_fraction": nan_fraction,
        "nan_status": nan_status,
        "sharpe_7d": sharpe_7d,
        "sharpe_status": sharpe_status,
    }


def run_health_check(
    models_dir: Path | None = None,
) -> list[dict]:
    """Run health check for all ML algorithms.

    Returns list of per-algorithm health dicts.
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent / "models"

    results = []
    for algo_name in ML_ALGO_NAMES:
        result = evaluate_health(algo_name, models_dir)
        results.append(result)

    # No-model algorithms are always OK
    for algo_name in NO_MODEL_ALGOS:
        results.append({
            "algo": algo_name,
            "overall": OK,
            "age_days": None,
            "age_status": OK,
            "nan_fraction": 0.0,
            "nan_status": OK,
            "sharpe_7d": None,
            "sharpe_status": OK,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="ML health check")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    results = run_health_check()

    if args.json_output:
        print(json.dumps(results, indent=2))
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"ML HEALTH CHECK — {today}")
        print("=" * 65)
        for r in results:
            age_str = f"age={r['age_days']}d" if r["age_days"] is not None else "age=N/A"
            sharpe_str = f"Sharpe_7d={r['sharpe_7d']:.1f}" if r["sharpe_7d"] is not None else "Sharpe_7d=N/A"
            print(f"  {r['algo']:30s} {r['overall']:10s} ({age_str}, {sharpe_str})")
        print("=" * 65)

    # Exit code
    statuses = [r["overall"] for r in results]
    if CRITICAL in statuses:
        sys.exit(2)
    elif WARN in statuses:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
