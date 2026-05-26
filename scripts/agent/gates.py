"""Gate protocol — structured, composable gate checks for hypothesis testing.

Each Gate wraps a stateless check function and produces a GateResult with
uniform fields: name, passed, metric, threshold, p_value, message.

Free functions (check_ic_gate, apply_fdr, etc.) provide the underlying
stateless logic, used both by the Gate classes and by BaseRunner directly.

Usage in runners:
    gates = [ICGate(min_ic=0.10), DeltaICGate(min_dIC=0.05)]
    for gate in gates:
        result = gate.evaluate(report, thresholds)
        if not result.passed:
            return False, result.message
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Protocol

log = logging.getLogger("nat.agent")


@dataclass
class GateResult:
    """Structured result from a gate evaluation."""
    name: str
    passed: bool
    metric: Optional[float]
    threshold: Optional[float]
    p_value: Optional[float]
    message: str


class Gate(Protocol):
    """Protocol for hypothesis gate checks."""
    name: str

    def evaluate(self, report: dict, thresholds: dict) -> GateResult: ...


# ---------------------------------------------------------------------------
# Stateless gate check functions
# ---------------------------------------------------------------------------

def _extract_pvalue(gate_msg: str) -> Optional[float]:
    """Extract p=... from a gate result message."""
    if "p=" not in gate_msg:
        return None
    try:
        return float(gate_msg.split("p=")[1].split()[0])
    except (IndexError, ValueError):
        return None


def apply_fdr(hypotheses: list, q: float = 0.05) -> list[str]:
    """Benjamini-Hochberg FDR control across a batch of tested hypotheses.

    Collects the IC p-value from each hypothesis that passed discovery,
    applies BH at level q, and returns IDs of hypotheses that should be
    rejected (marked fdr_rejected).

    Args:
        hypotheses: list of Hypothesis objects (or dicts) from this cycle
        q: false discovery rate threshold (default 0.05)

    Returns:
        List of hypothesis IDs that fail FDR correction.
    """
    # Collect (id, pvalue) for hypotheses that passed discovery
    pvals = []
    for h in hypotheses:
        results = h.get("results") if isinstance(h, dict) else getattr(h, "results", None)
        hyp_id = h.get("id") if isinstance(h, dict) else getattr(h, "id", None)
        status = h.get("status") if isinstance(h, dict) else getattr(h, "status", None)
        if results is None or status == "queued":
            continue
        # Look for p-value in gate_results
        for gr in (results.get("gate_results") or []):
            p = _extract_pvalue(gr.get("msg", ""))
            if p is not None:
                pvals.append((hyp_id, p))
                break  # one p-value per hypothesis (first IC gate)

    if len(pvals) < 2:
        return []  # nothing to correct with fewer than 2 tests

    # Sort by p-value (ascending)
    pvals.sort(key=lambda x: x[1])
    m = len(pvals)

    # BH: find largest k where p(k) <= k/m * q
    bh_threshold = 0.0
    for k, (hyp_id, p) in enumerate(pvals, 1):
        if p <= (k / m) * q:
            bh_threshold = p

    # Reject hypotheses with p > bh_threshold (they don't survive FDR)
    rejected = []
    for hyp_id, p in pvals:
        if p > bh_threshold and bh_threshold > 0:
            rejected.append(hyp_id)

    return rejected


def _find_gate_entry(report: dict, regime_gate: str):
    """Find the single_factors entry matching a regime_gate label."""
    for entry in report.get("single_factors", []):
        if entry.get("label") == regime_gate:
            return entry
    return None


def _ic_pvalue(ic: float, n_obs: int) -> float:
    """Two-sided p-value for Spearman IC under H0: no predictive power."""
    from math import erfc, sqrt
    if n_obs < 2:
        return 1.0
    z = abs(ic) * sqrt(n_obs)
    return erfc(z / sqrt(2))


def check_ic_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check if IC exceeds minimum threshold.

    Computes p-value for FDR control.
    """
    min_ic = thresholds.get("min_ic", 0.10)
    regime_gate = thresholds.get("regime_gate")
    ic = None
    n_obs = report.get("n_rows", 0)

    if regime_gate and "single_factors" in report:
        entry = _find_gate_entry(report, regime_gate)
        if entry:
            ic = entry.get("ic_filt_5s")
            n_obs = entry.get("n_obs", n_obs)

    if ic is None:
        if "baseline_ic_filt_5s" in report:
            ic = report["baseline_ic_filt_5s"]
        elif "best_ic" in report:
            ic = report["best_ic"]
        elif "profiles" in report and len(report["profiles"]) > 0:
            ic = max(abs(p.get("ic_best", 0)) for p in report["profiles"])

    if ic is None:
        return False, "could not extract IC from report"
    pval = _ic_pvalue(ic, n_obs)
    passed = abs(ic) >= min_ic
    label = f"gated({regime_gate})" if regime_gate else "aggregate"
    return passed, (f"IC={ic:.4f} [{label}] vs min={min_ic} p={pval:.2e}"
                    + (" PASS" if passed else " FAIL"))


def check_dIC_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check that the regime gate improves IC over the ungated baseline."""
    min_dIC = thresholds.get("min_dIC", 0.05)
    regime_gate = thresholds.get("regime_gate")

    if not regime_gate or "single_factors" not in report:
        return True, "no regime gate, dIC check skipped"

    baseline = report.get("baseline_ic_filt_5s", 0.0)
    entry = _find_gate_entry(report, regime_gate)
    if entry is None:
        return False, f"gate {regime_gate} not found in report FAIL"

    gated_ic = entry.get("ic_filt_5s", 0.0)
    dIC = gated_ic - baseline
    passed = dIC >= min_dIC
    return passed, (f"dIC={dIC:+.4f} (gated={gated_ic:.4f} - base={baseline:.4f}) "
                    f"vs min={min_dIC}" + (" PASS" if passed else " FAIL"))


def check_cost_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check if signal has sufficient per-trade edge."""
    min_avg_bps = thresholds.get("min_avg_return_bps", 0.1)
    entries = report.get("thresholds", [])
    if not entries:
        return True, "no backtest thresholds, cost check skipped"

    best = max(entries, key=lambda t: t.get("avg_return_per_trade_bps", 0))
    avg_ret = best.get("avg_return_per_trade_bps", 0)
    maker_sharpe = best.get("net_sharpe_maker", 0)
    thresh = best.get("threshold", 0)
    passed = avg_ret >= min_avg_bps
    return passed, (f"avg_ret={avg_ret:.3f}bps (maker_sharpe={maker_sharpe:.1f}) "
                    f"at thresh={thresh:.1f} vs min={min_avg_bps}bps"
                    + (" PASS" if passed else " FAIL"))


def check_coverage_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check if regime coverage exceeds minimum."""
    min_coverage = thresholds.get("min_coverage", 0.20)
    pareto = report.get("pareto_optimal", [])
    for p in pareto:
        if p.get("coverage", 0) >= min_coverage:
            return True, f"coverage={p['coverage']:.0%} >= {min_coverage:.0%} PASS"
    return False, f"no Pareto combo with coverage >= {min_coverage:.0%} FAIL"


def check_walkforward_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check walk-forward KEEP verdict."""
    keep_count = report.get("keep_count", 0)
    total = keep_count + report.get("monitor_count", 0) + report.get("drop_count", 0)
    if total == 0:
        return False, "no walk-forward results"
    keep_frac = keep_count / total
    passed = keep_frac >= thresholds.get("min_keep_frac", 0.3)
    return passed, f"KEEP={keep_count}/{total} ({keep_frac:.0%})" + (" PASS" if passed else " FAIL")


def _parse_gate_spec(gate_str: str):
    """Parse 'ent_book_shape<P40' into ('ent_book_shape', '<', 'P40')."""
    m = re.match(r'^([a-z_0-9]+)([<>])(P\d+)$', gate_str)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def check_correlation_gate(
    candidate_feature: str,
    candidate_gate,
    registry: list[dict],
    data_dir: str,
    symbol: str = "BTC",
    max_corr: float = 0.70,
) -> tuple[bool, str]:
    """Check if a candidate signal is redundant with existing registry signals."""
    import numpy as np
    import pandas as pd

    if not registry:
        return True, "empty registry, no dedup needed PASS"

    from .runner import _load_feature_data, _extract_gated_signal

    df = _load_feature_data(data_dir, symbol)
    if df is None or len(df) == 0:
        log.warning("  Correlation check: could not load data from %s", data_dir)
        return True, "no data for correlation check, skipped"

    cand_vals = _extract_gated_signal(df, candidate_feature, candidate_gate)
    if cand_vals is None:
        return True, f"feature {candidate_feature} not in data, skipped"

    worst_corr = 0.0
    worst_name = ""
    for sig in registry:
        sig_features = sig.get("features", [])
        sig_gate = sig.get("regime_gate")
        for sf in sig_features:
            ref_vals = _extract_gated_signal(df, sf, sig_gate)
            if ref_vals is None:
                continue
            valid = ~(np.isnan(cand_vals) | np.isnan(ref_vals))
            if valid.sum() < 100:
                continue
            corr = pd.Series(cand_vals[valid]).corr(
                pd.Series(ref_vals[valid]), method="spearman"
            )
            if abs(corr) > abs(worst_corr):
                worst_corr = corr
                worst_name = f"{sf}|{sig_gate or 'ungated'}"

    passed = abs(worst_corr) <= max_corr
    if worst_name:
        msg = (f"max_corr={worst_corr:+.3f} vs {worst_name} "
               f"(threshold={max_corr})" + (" PASS" if passed else " REDUNDANT"))
    else:
        msg = "no comparable registry signals PASS"
    return passed, msg


# ---------------------------------------------------------------------------
# Concrete Gate classes — wrap the stateless check functions above
# ---------------------------------------------------------------------------

class ICGate:
    """Check if IC exceeds minimum threshold."""
    name = "IC"

    def evaluate(self, report: dict, thresholds: dict) -> GateResult:
        min_ic = thresholds.get("min_ic", 0.10)
        regime_gate = thresholds.get("regime_gate")
        ic = None
        n_obs = report.get("n_rows", 0)

        if regime_gate and "single_factors" in report:
            entry = _find_gate_entry(report, regime_gate)
            if entry:
                ic = entry.get("ic_filt_5s")
                n_obs = entry.get("n_obs", n_obs)

        if ic is None:
            if "baseline_ic_filt_5s" in report:
                ic = report["baseline_ic_filt_5s"]
            elif "best_ic" in report:
                ic = report["best_ic"]
            elif "profiles" in report and len(report["profiles"]) > 0:
                ic = max(abs(p.get("ic_best", 0)) for p in report["profiles"])

        if ic is None:
            return GateResult(
                name=self.name, passed=False, metric=None,
                threshold=min_ic, p_value=None,
                message="could not extract IC from report",
            )

        pval = _ic_pvalue(ic, n_obs)
        passed = abs(ic) >= min_ic
        label = f"gated({regime_gate})" if regime_gate else "aggregate"
        msg = (f"IC={ic:.4f} [{label}] vs min={min_ic} p={pval:.2e}"
               + (" PASS" if passed else " FAIL"))
        return GateResult(
            name=self.name, passed=passed, metric=ic,
            threshold=min_ic, p_value=pval, message=msg,
        )


class DeltaICGate:
    """Check that the regime gate improves IC over ungated baseline."""
    name = "dIC"

    def evaluate(self, report: dict, thresholds: dict) -> GateResult:
        min_dIC = thresholds.get("min_dIC", 0.05)
        regime_gate = thresholds.get("regime_gate")

        if not regime_gate or "single_factors" not in report:
            return GateResult(
                name=self.name, passed=True, metric=None,
                threshold=min_dIC, p_value=None,
                message="no regime gate, dIC check skipped",
            )

        baseline = report.get("baseline_ic_filt_5s", 0.0)
        entry = _find_gate_entry(report, regime_gate)
        if entry is None:
            return GateResult(
                name=self.name, passed=False, metric=None,
                threshold=min_dIC, p_value=None,
                message=f"gate {regime_gate} not found in report FAIL",
            )

        gated_ic = entry.get("ic_filt_5s", 0.0)
        dIC = gated_ic - baseline
        passed = dIC >= min_dIC
        msg = (f"dIC={dIC:+.4f} (gated={gated_ic:.4f} - base={baseline:.4f}) "
               f"vs min={min_dIC}" + (" PASS" if passed else " FAIL"))
        return GateResult(
            name=self.name, passed=passed, metric=dIC,
            threshold=min_dIC, p_value=None, message=msg,
        )


class CostGate:
    """Check if signal has sufficient per-trade edge."""
    name = "cost"

    def evaluate(self, report: dict, thresholds: dict) -> GateResult:
        min_avg_bps = thresholds.get("min_avg_return_bps", 0.1)
        entries = report.get("thresholds", [])
        if not entries:
            return GateResult(
                name=self.name, passed=True, metric=None,
                threshold=min_avg_bps, p_value=None,
                message="no backtest thresholds, cost check skipped",
            )

        best = max(entries, key=lambda t: t.get("avg_return_per_trade_bps", 0))
        avg_ret = best.get("avg_return_per_trade_bps", 0)
        maker_sharpe = best.get("net_sharpe_maker", 0)
        thresh = best.get("threshold", 0)
        passed = avg_ret >= min_avg_bps
        msg = (f"avg_ret={avg_ret:.3f}bps (maker_sharpe={maker_sharpe:.1f}) "
               f"at thresh={thresh:.1f} vs min={min_avg_bps}bps"
               + (" PASS" if passed else " FAIL"))
        return GateResult(
            name=self.name, passed=passed, metric=avg_ret,
            threshold=min_avg_bps, p_value=None, message=msg,
        )


class CoverageGate:
    """Check if regime coverage exceeds minimum."""
    name = "coverage"

    def evaluate(self, report: dict, thresholds: dict) -> GateResult:
        min_coverage = thresholds.get("min_coverage", 0.20)
        pareto = report.get("pareto_optimal", [])
        for p in pareto:
            cov = p.get("coverage", 0)
            if cov >= min_coverage:
                return GateResult(
                    name=self.name, passed=True, metric=cov,
                    threshold=min_coverage, p_value=None,
                    message=f"coverage={cov:.0%} >= {min_coverage:.0%} PASS",
                )
        return GateResult(
            name=self.name, passed=False, metric=0.0,
            threshold=min_coverage, p_value=None,
            message=f"no Pareto combo with coverage >= {min_coverage:.0%} FAIL",
        )


class WalkforwardGate:
    """Check walk-forward KEEP verdict."""
    name = "walkforward"

    def evaluate(self, report: dict, thresholds: dict) -> GateResult:
        keep_count = report.get("keep_count", 0)
        total = keep_count + report.get("monitor_count", 0) + report.get("drop_count", 0)
        min_keep_frac = thresholds.get("min_keep_frac", 0.3)
        if total == 0:
            return GateResult(
                name=self.name, passed=False, metric=0.0,
                threshold=min_keep_frac, p_value=None,
                message="no walk-forward results",
            )
        keep_frac = keep_count / total
        passed = keep_frac >= min_keep_frac
        msg = f"KEEP={keep_count}/{total} ({keep_frac:.0%})" + (" PASS" if passed else " FAIL")
        return GateResult(
            name=self.name, passed=passed, metric=keep_frac,
            threshold=min_keep_frac, p_value=None, message=msg,
        )


# ---------------------------------------------------------------------------
# Default gate sets
# ---------------------------------------------------------------------------

# Discovery gates — run inside run_discovery() against each report
DISCOVERY_GATES: list[Gate] = [ICGate(), DeltaICGate()]
