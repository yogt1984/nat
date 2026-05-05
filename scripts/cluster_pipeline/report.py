"""
Automated profiling report generator.

Phase 8, Task 8.2: Generates a structured markdown report summarizing
all profiling pipeline outputs.

Usage:
    from cluster_pipeline.report import generate_report

    report = generate_report(profiling_result, prices, vector="orderflow", timeframe="15min")
    print(report)  # or write to file
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from cluster_pipeline.hierarchy import ProfilingResult
from cluster_pipeline.validate import validate, ValidationVerdict, cross_symbol_consistency, CrossSymbolResult

logger = logging.getLogger(__name__)


def generate_report(
    profiling_result: ProfilingResult,
    prices: np.ndarray,
    vector: str = "unknown",
    timeframe: str = "unknown",
    cross_symbol_result: Optional[CrossSymbolResult] = None,
    training_ll_p10: Optional[float] = None,
    training_ll_p50: Optional[float] = None,
) -> str:
    """
    Generate a structured markdown profiling report.

    Args:
        profiling_result: Output from profile().
        prices: 1-D price array aligned with profiling_result.bars.
        vector: Feature vector name (e.g. "orderflow", "entropy").
        timeframe: Aggregation timeframe (e.g. "15min", "1h").
        cross_symbol_result: Optional cross-symbol consistency result.
        training_ll_p10: Optional training log-likelihood 10th percentile.
        training_ll_p50: Optional training log-likelihood 50th percentile.

    Returns:
        Markdown string with full profiling report.
    """
    prices = np.asarray(prices, dtype=float)
    sections = []

    # Header
    sections.append(f"# Profiling Report — {vector}@{timeframe}\n")

    # Data Summary
    sections.append(_section_data_summary(profiling_result))

    # Structural Breaks
    sections.append(_section_breaks(profiling_result))

    # Structure Test
    sections.append(_section_structure_test(profiling_result))

    # Macro Regimes
    sections.append(_section_macro_regimes(profiling_result))

    # Micro States
    sections.append(_section_micro_states(profiling_result))

    # Transition Structure
    sections.append(_section_transitions(profiling_result))

    # Predictive Quality + Validation
    verdict = validate(profiling_result, prices)
    sections.append(_section_validation(verdict))

    # Cross-Symbol Consistency
    if cross_symbol_result is not None:
        sections.append(_section_cross_symbol(cross_symbol_result))

    # Drift Baseline
    if training_ll_p10 is not None:
        sections.append(_section_drift_baseline(training_ll_p10, training_ll_p50))

    # Recommendations
    sections.append(_section_recommendations(verdict, profiling_result))

    return "\n".join(sections)


def _section_data_summary(pr: ProfilingResult) -> str:
    lines = ["## Data Summary\n"]
    n_bars = len(pr.bars) if pr.bars is not None else 0
    lines.append(f"- **Total bars:** {n_bars}")
    lines.append(f"- **Macro regimes (k):** {pr.macro.k}")
    lines.append(f"- **Total micro states:** {pr.hierarchy.n_micro_total}")
    if pr.macro.early_exit:
        lines.append("- **Early exit:** Yes (no structure found)")
    lines.append("")
    return "\n".join(lines)


def _section_breaks(pr: ProfilingResult) -> str:
    lines = ["## Structural Breaks\n"]
    if hasattr(pr, 'break_indices') and pr.break_indices:
        lines.append(f"- Breaks detected at indices: {pr.break_indices}")
        lines.append(f"- Longest segment used for profiling")
    else:
        lines.append("- No structural breaks detected (or not applicable)")
    lines.append("")
    return "\n".join(lines)


def _section_structure_test(pr: ProfilingResult) -> str:
    lines = ["## Structure Test\n"]
    st = pr.macro.structure_test
    lines.append(f"- **Has structure:** {st.has_structure}")
    lines.append(f"- **Hopkins statistic:** {st.hopkins:.4f}")
    lines.append(f"- **Dip test p-value:** {st.dip_p:.4f}")
    if hasattr(st, 'recommendation'):
        lines.append(f"- **Recommendation:** {st.recommendation}")
    lines.append("")
    return "\n".join(lines)


def _section_macro_regimes(pr: ProfilingResult) -> str:
    lines = ["## Macro Regimes\n"]
    macro = pr.macro
    if macro.early_exit:
        lines.append("- Early exit — no meaningful regimes found.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"- **k:** {macro.k}")
    lines.append(f"- **Silhouette:** {macro.quality.silhouette:.4f}")
    lines.append(f"- **Bootstrap ARI:** {macro.stability.mean_ari:.4f}")
    lines.append(f"- **Self-transition rate:** {macro.self_transition_rate:.4f}")

    # Duration summary per regime
    lines.append("\n### Regime Durations\n")
    lines.append("| Regime | Mean Duration | Count |")
    lines.append("|--------|--------------|-------|")
    for regime_id, durations in sorted(macro.durations.items()):
        mean_d = np.mean(durations) if durations else 0
        lines.append(f"| {regime_id} | {mean_d:.1f} bars | {len(durations)} |")

    lines.append("")
    return "\n".join(lines)


def _section_micro_states(pr: ProfilingResult) -> str:
    lines = ["## Micro States\n"]

    for regime_id, micro in sorted(pr.micros.items()):
        if micro is None:
            lines.append(f"### Regime {regime_id}: no micro structure\n")
            continue

        lines.append(f"### Regime {regime_id}\n")
        lines.append(f"- **Micro k:** {micro.k}")
        lines.append(f"- **Bars:** {micro.n_bars}")
        lines.append(f"- **Silhouette:** {micro.quality.silhouette:.4f}")
        lines.append(f"- **Bootstrap ARI:** {micro.stability.mean_ari:.4f}")
        lines.append("")

    return "\n".join(lines)


def _section_transitions(pr: ProfilingResult) -> str:
    lines = ["## Transition Structure\n"]
    hierarchy = pr.hierarchy
    lines.append(f"- **Global micro states:** {hierarchy.n_micro_total}")

    # Label map summary
    lines.append("\n### State Map\n")
    lines.append("| Global ID | Regime | Local ID |")
    lines.append("|-----------|--------|----------|")
    for gid, (regime, local) in sorted(hierarchy.label_map.items()):
        lines.append(f"| {gid} | {regime} | {local} |")

    lines.append("")
    return "\n".join(lines)


def _section_validation(verdict: ValidationVerdict) -> str:
    lines = ["## Validation Verdict\n"]
    lines.append(f"- **Overall:** `{verdict.overall}`")
    lines.append(f"- **Summary:** {verdict.summary}")

    # Q1
    q1 = verdict.q1_structural
    lines.append(f"\n### Q1: Structural ({'PASS' if q1['pass'] else 'FAIL'})\n")
    lines.append(f"- Silhouette: {q1.get('silhouette', 0):.4f} (threshold: {q1.get('silhouette_threshold', 0.25)})")
    lines.append(f"- Bootstrap ARI: {q1.get('bootstrap_ari', 0):.4f} (threshold: {q1.get('bootstrap_ari_threshold', 0.6)})")

    # Q2
    q2 = verdict.q2_predictive
    lines.append(f"\n### Q2: Predictive ({'PASS' if q2['pass'] else 'FAIL'})\n")
    lines.append(f"- Pass horizons: {q2.get('pass_horizons', [])}")
    lines.append(f"- Any horizon mode: {q2.get('any_horizon', True)}")

    # Q3
    q3 = verdict.q3_operational
    lines.append(f"\n### Q3: Operational ({'PASS' if q3['pass'] else 'FAIL'})\n")
    lines.append(f"- Macro STR: {q3.get('macro_str', 0):.4f} (threshold: {q3.get('macro_str_threshold', 0.8)})")
    lines.append(f"- Mean duration: {q3.get('mean_duration', 0):.1f} (threshold: {q3.get('min_duration_threshold', 3)})")

    # Per-state verdicts
    lines.append("\n### Per-State Verdicts\n")
    lines.append("| State | Verdict |")
    lines.append("|-------|---------|")
    for state_id, v in sorted(verdict.per_state_verdicts.items()):
        lines.append(f"| {state_id} | {v} |")

    lines.append("")
    return "\n".join(lines)


def _section_cross_symbol(cs: CrossSymbolResult) -> str:
    lines = ["## Cross-Symbol Consistency\n"]
    lines.append(f"- **Mean agreement (ARI):** {cs.mean_agreement:.4f}")
    lines.append(f"- **Above random:** {cs.above_random}")
    lines.append(f"- **Disagreement rate:** {cs.disagreement_rate:.4f}")
    lines.append(f"- **Symbols:** {cs.symbol_names}")
    lines.append("")
    return "\n".join(lines)


def _section_drift_baseline(ll_p10: float, ll_p50: Optional[float]) -> str:
    lines = ["## Drift Baseline\n"]
    lines.append(f"- **Training log-likelihood p10:** {ll_p10:.4f}")
    if ll_p50 is not None:
        lines.append(f"- **Training log-likelihood p50:** {ll_p50:.4f}")
    lines.append("- Drift fires when rolling LL < p10 for 20+ consecutive bars")
    lines.append("")
    return "\n".join(lines)


def _section_recommendations(verdict: ValidationVerdict, pr: ProfilingResult) -> str:
    lines = ["## Recommendations\n"]

    overall = verdict.overall
    if overall == "GO":
        lines.append("- **Deploy:** All quality gates passed. Deploy online classifier.")
        lines.append("- Monitor drift dashboard for distribution shift.")
    elif overall == "PIVOT":
        lines.append("- **Pivot:** Clusters are real and predictive but not operationally tradeable.")
        lines.append("- Consider: longer timeframe, different entry logic, or combining with other signals.")
        q3 = verdict.q3_operational
        if not q3.get("macro_str_pass", True):
            lines.append("- Issue: Low self-transition rate — regimes switch too frequently.")
        if not q3.get("duration_pass", True):
            lines.append("- Issue: Short durations — not enough time to enter/exit positions.")
    elif overall == "COLLECT":
        lines.append("- **Collect more data:** Clusters exist but do not predict returns yet.")
        lines.append("- Wait for more data or try different feature vector / timeframe.")
    elif overall == "DROP":
        lines.append("- **Drop:** No meaningful cluster structure found.")
        lines.append("- Try: different vector, longer timeframe, or more data.")
        if pr.macro.early_exit:
            lines.append("- Root cause: Structure test failed (data appears uniform).")

    lines.append("")
    return "\n".join(lines)
