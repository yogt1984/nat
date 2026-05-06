"""
Task 4: Quick profiler.

Runs a lightweight profiling pass on available data.
Returns structured result for the dashboard.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import numpy as np

from .state import ProfilingSnapshot

logger = logging.getLogger(__name__)

# Add scripts to path for cluster_pipeline imports
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "features"


def quick_profile(data_dir: Path = DEFAULT_DATA_DIR) -> ProfilingSnapshot:
    """
    Run a fast profiling pass on whatever data exists.

    - bars < 50:   returns status="insufficient"
    - bars 50-100: Hopkins test only (status="partial")
    - bars > 100:  full profile + quality gates (status="complete")
    """
    snapshot = ProfilingSnapshot()
    snapshot.last_run = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        from cluster_pipeline.loader import load_parquet
        from cluster_pipeline.preprocess import aggregate_bars
        from cluster_pipeline.hierarchy import profile
        from cluster_pipeline.validate import validate
    except ImportError as e:
        logger.error(f"Cannot import cluster_pipeline: {e}")
        snapshot.status = "error"
        return snapshot

    # Load data
    try:
        df = load_parquet(str(data_dir))
    except Exception as e:
        logger.error(f"Cannot load data: {e}")
        snapshot.status = "error"
        return snapshot

    if len(df) == 0:
        snapshot.status = "insufficient"
        return snapshot

    # Aggregate to 15-min bars
    try:
        bars = aggregate_bars(df, timeframe="15min")
    except Exception as e:
        logger.error(f"Cannot aggregate bars: {e}")
        snapshot.status = "error"
        return snapshot

    n_bars = len(bars)
    snapshot.n_bars_used = n_bars

    if n_bars < 50:
        snapshot.status = "insufficient"
        return snapshot

    # Run profiling (handles derivatives internally)
    try:
        result = profile(df, vector="entropy", include_spectral=True)
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        snapshot.status = "error"
        return snapshot

    macro = result.macro

    # Structure test
    if hasattr(macro, 'structure_test') and macro.structure_test:
        snapshot.hopkins = round(macro.structure_test.hopkins_statistic, 3)

    if macro.early_exit:
        snapshot.status = "partial" if n_bars < 100 else "complete"
        snapshot.k = 0
        snapshot.current_verdict = "DROP"
        return snapshot

    # GMM results
    snapshot.k = macro.k
    snapshot.silhouette = round(macro.quality.silhouette, 3)
    snapshot.bootstrap_ari = round(macro.stability.mean_ari, 3)

    # State summaries
    hierarchy = result.hierarchy
    states = []
    for i in range(hierarchy.n_micro_total):
        mask = hierarchy.micro_labels == i
        n_state = int(mask.sum())
        if n_state == 0:
            continue
        states.append({
            "id": i,
            "n_bars": n_state,
            "pct": round(n_state / n_bars, 2),
            "mean_duration": 0.0,  # computed from transitions
            "label": f"state_{i}",
        })
    snapshot.states = states

    # Transition matrix
    if hasattr(macro, 'transitions') and macro.transitions is not None:
        tm = macro.transitions
        if hasattr(tm, 'matrix'):
            snapshot.transition_matrix = [[round(float(x), 3) for x in row] for row in tm.matrix]
            # Self-transition rate = mean of diagonal
            diag = [tm.matrix[i][i] for i in range(len(tm.matrix))]
            snapshot.self_transition_rate = round(float(np.mean(diag)), 3)

        # Mean duration from transitions
        if hasattr(tm, 'durations') and tm.durations:
            all_durs = []
            for durs in tm.durations.values():
                all_durs.extend(durs)
            if all_durs:
                snapshot.mean_duration = round(float(np.mean(all_durs)), 1)
                # Update per-state durations
                for state_info in snapshot.states:
                    sid = state_info["id"]
                    if sid in tm.durations and tm.durations[sid]:
                        state_info["mean_duration"] = round(float(np.mean(tm.durations[sid])), 1)

    # Quality gates
    try:
        prices = result.bars["raw_midprice_mean"].values if "raw_midprice_mean" in result.bars.columns else np.ones(len(result.bars))
        verdict = validate(result, prices)
        snapshot.q1_pass = verdict.q1_structural["pass"]
        snapshot.q2_pass = verdict.q2_predictive["pass"]
        snapshot.q3_pass = verdict.q3_operational["pass"]
        snapshot.current_verdict = verdict.overall

        # Best Q2 p-value
        kruskal = verdict.q2_predictive.get("kruskal_results", {})
        p_values = [v.get("p_value", 1.0) for v in kruskal.values() if isinstance(v, dict)]
        if p_values:
            snapshot.q2_best_p = round(min(p_values), 4)
    except Exception as e:
        logger.warning(f"Validation failed: {e}")
        snapshot.current_verdict = "ERROR"

    snapshot.status = "complete"
    return snapshot
