"""
Validation framework for NAT profiling pipeline.

Phase 6: Quality gates (Q1 structural, Q2 predictive, Q3 operational)
and decision logic.

Usage:
    from cluster_pipeline.validate import validate

    verdict = validate(profiling_result, prices)
    print(verdict.overall)  # "GO", "PIVOT", "COLLECT", "DROP"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from cluster_pipeline.hierarchy import ProfilingResult
from cluster_pipeline.characterize import return_profile

logger = logging.getLogger(__name__)


# Default thresholds
Q1_DEFAULTS = {
    "silhouette": 0.25,
    "block_bootstrap_ari": 0.6,
}

Q2_DEFAULTS = {
    "kruskal_p": 0.05,
    "eta_squared": 0.01,
    "min_sharpe": 0.3,
    "any_horizon": True,
}

Q3_DEFAULTS = {
    "macro_str": 0.8,
    "micro_str": 0.5,
    "min_duration": 3,
    "entry_lead": 1,
}


@dataclass
class ValidationVerdict:
    """Complete validation result with per-gate details."""

    q1_structural: Dict[str, Any]
    q2_predictive: Dict[str, Any]
    q3_operational: Dict[str, Any]
    overall: str  # "GO", "PIVOT", "COLLECT", "DROP"
    per_state_verdicts: Dict[int, str]
    summary: str


def validate(
    profiling_result: ProfilingResult,
    prices: np.ndarray,
    q1_thresholds: Optional[Dict] = None,
    q2_thresholds: Optional[Dict] = None,
    q3_thresholds: Optional[Dict] = None,
) -> ValidationVerdict:
    """
    Run Q1/Q2/Q3 quality gates and produce a verdict.

    Decision logic:
      - Q1 fail → "DROP" (clusters aren't real)
      - Q1 pass + Q2 fail at all horizons → "COLLECT" (need more data)
      - Q1 pass + Q2 pass + Q3 fail → "PIVOT" (real but not tradeable yet)
      - All pass → "GO" (deploy)

    Args:
        profiling_result: Output from profile().
        prices: 1-D price array aligned with profiling_result.bars.
        q1_thresholds: Override Q1 defaults.
        q2_thresholds: Override Q2 defaults.
        q3_thresholds: Override Q3 defaults.

    Returns:
        ValidationVerdict with gate details and overall decision.

    Raises:
        ValueError: if prices length doesn't match bars.
    """
    prices = np.asarray(prices, dtype=float)

    n_bars = len(profiling_result.bars)
    if len(prices) != n_bars:
        raise ValueError(
            f"prices length ({len(prices)}) != bars length ({n_bars})"
        )

    q1_t = {**Q1_DEFAULTS, **(q1_thresholds or {})}
    q2_t = {**Q2_DEFAULTS, **(q2_thresholds or {})}
    q3_t = {**Q3_DEFAULTS, **(q3_thresholds or {})}

    macro = profiling_result.macro
    hierarchy = profiling_result.hierarchy

    # ===== Q1: Structural Quality =====
    q1 = _evaluate_q1(macro, q1_t)

    # ===== Q2: Predictive Quality =====
    q2 = _evaluate_q2(hierarchy, prices, q2_t)

    # ===== Q3: Operational Quality =====
    q3 = _evaluate_q3(macro, hierarchy, q3_t)

    # ===== Decision Logic =====
    if not q1["pass"]:
        overall = "DROP"
    elif not q2["pass"]:
        overall = "COLLECT"
    elif not q3["pass"]:
        overall = "PIVOT"
    else:
        overall = "GO"

    # Per-state verdicts (simplified: GO if state has Sharpe > threshold at any horizon)
    per_state_verdicts = _per_state_verdicts(hierarchy, prices, q2_t)

    summary = (
        f"Q1({'PASS' if q1['pass'] else 'FAIL'}): "
        f"sil={q1.get('silhouette', 0):.3f}, "
        f"ari={q1.get('bootstrap_ari', 0):.3f} | "
        f"Q2({'PASS' if q2['pass'] else 'FAIL'}): "
        f"kruskal_pass_horizons={q2.get('pass_horizons', [])} | "
        f"Q3({'PASS' if q3['pass'] else 'FAIL'}): "
        f"str={q3.get('macro_str', 0):.3f} | "
        f"→ {overall}"
    )

    return ValidationVerdict(
        q1_structural=q1,
        q2_predictive=q2,
        q3_operational=q3,
        overall=overall,
        per_state_verdicts=per_state_verdicts,
        summary=summary,
    )


def _evaluate_q1(macro, thresholds: Dict) -> Dict[str, Any]:
    """
    Q1: Structural quality — are the clusters real?

    Checks:
      - Silhouette score > threshold
      - Block bootstrap ARI > threshold
    """
    if macro.early_exit:
        return {
            "pass": False,
            "silhouette": 0.0,
            "bootstrap_ari": 0.0,
            "reason": "early_exit: no structure found",
        }

    silhouette = macro.quality.silhouette
    bootstrap_ari = macro.stability.mean_ari

    sil_pass = silhouette >= thresholds["silhouette"]
    ari_pass = bootstrap_ari >= thresholds["block_bootstrap_ari"]

    passed = sil_pass and ari_pass

    result = {
        "pass": passed,
        "silhouette": silhouette,
        "silhouette_threshold": thresholds["silhouette"],
        "silhouette_pass": sil_pass,
        "bootstrap_ari": bootstrap_ari,
        "bootstrap_ari_threshold": thresholds["block_bootstrap_ari"],
        "bootstrap_ari_pass": ari_pass,
    }

    if not passed:
        reasons = []
        if not sil_pass:
            reasons.append(f"silhouette {silhouette:.3f} < {thresholds['silhouette']}")
        if not ari_pass:
            reasons.append(f"bootstrap_ari {bootstrap_ari:.3f} < {thresholds['block_bootstrap_ari']}")
        result["reason"] = "; ".join(reasons)

    return result


def _evaluate_q2(hierarchy, prices: np.ndarray, thresholds: Dict) -> Dict[str, Any]:
    """
    Q2: Predictive quality — do states predict returns?

    Uses Kruskal-Wallis test across states at multiple horizons.
    Passes if significant at ANY horizon (any_horizon=True).
    """
    micro_labels = hierarchy.micro_labels
    unique_states = np.unique(micro_labels)
    log_prices = np.log(prices)
    n = len(prices)

    horizons = [1, 5, 10, 20]
    kruskal_results = {}
    pass_horizons = []

    for h in horizons:
        # Collect returns per state
        groups = []
        for s in unique_states:
            state_mask = micro_labels == s
            state_indices = np.where(state_mask)[0]
            valid = state_indices[state_indices + h < n]
            if len(valid) >= 5:
                returns = log_prices[valid + h] - log_prices[valid]
                groups.append(returns)

        if len(groups) < 2:
            kruskal_results[h] = {"p_value": 1.0, "statistic": 0.0, "pass": False}
            continue

        try:
            stat, p_value = sp_stats.kruskal(*groups)
        except Exception:
            kruskal_results[h] = {"p_value": 1.0, "statistic": 0.0, "pass": False}
            continue

        # Eta-squared (effect size): H / (n-1)
        n_total = sum(len(g) for g in groups)
        eta_sq = float(stat / (n_total - 1)) if n_total > 1 else 0.0

        h_pass = (p_value < thresholds["kruskal_p"] and
                  eta_sq >= thresholds["eta_squared"])

        kruskal_results[h] = {
            "p_value": float(p_value),
            "statistic": float(stat),
            "eta_squared": eta_sq,
            "pass": h_pass,
        }

        if h_pass:
            pass_horizons.append(h)

    if thresholds.get("any_horizon", True):
        q2_pass = len(pass_horizons) > 0
    else:
        q2_pass = len(pass_horizons) == len(horizons)

    return {
        "pass": q2_pass,
        "kruskal_results": kruskal_results,
        "pass_horizons": pass_horizons,
        "any_horizon": thresholds.get("any_horizon", True),
    }


def _evaluate_q3(macro, hierarchy, thresholds: Dict) -> Dict[str, Any]:
    """
    Q3: Operational quality — are states tradeable?

    Checks:
      - Macro self-transition rate > threshold (regimes are persistent)
      - Mean duration > min_duration bars
    """
    if macro.early_exit:
        return {
            "pass": False,
            "macro_str": 0.0,
            "mean_duration": 0.0,
            "reason": "early_exit",
        }

    macro_str = macro.self_transition_rate
    str_pass = macro_str >= thresholds["macro_str"]

    # Mean duration across all regimes
    all_durations = []
    for durs in macro.durations.values():
        all_durations.extend(durs)
    mean_duration = float(np.mean(all_durations)) if all_durations else 0.0
    dur_pass = mean_duration >= thresholds["min_duration"]

    passed = str_pass and dur_pass

    result = {
        "pass": passed,
        "macro_str": macro_str,
        "macro_str_threshold": thresholds["macro_str"],
        "macro_str_pass": str_pass,
        "mean_duration": mean_duration,
        "min_duration_threshold": thresholds["min_duration"],
        "duration_pass": dur_pass,
    }

    if not passed:
        reasons = []
        if not str_pass:
            reasons.append(f"macro_str {macro_str:.3f} < {thresholds['macro_str']}")
        if not dur_pass:
            reasons.append(f"mean_duration {mean_duration:.1f} < {thresholds['min_duration']}")
        result["reason"] = "; ".join(reasons)

    return result


def _per_state_verdicts(
    hierarchy, prices: np.ndarray, q2_thresholds: Dict
) -> Dict[int, str]:
    """Compute per-state verdict based on return Sharpe at any horizon."""
    micro_labels = hierarchy.micro_labels
    verdicts = {}
    min_sharpe = q2_thresholds.get("min_sharpe", 0.3)

    for state_id in range(hierarchy.n_micro_total):
        rp = return_profile(
            micro_labels, prices, state_id=state_id,
            horizons=[1, 5, 10, 20],
        )
        # Check if any horizon has sufficient Sharpe
        has_signal = False
        for h, stats in rp.horizons.items():
            if stats["n"] >= 10 and abs(stats["sharpe"]) >= min_sharpe:
                has_signal = True
                break

        verdicts[state_id] = "GO" if has_signal else "COLLECT"

    return verdicts
