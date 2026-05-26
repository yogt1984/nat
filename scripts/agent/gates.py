"""Gate protocol — structured, composable gate checks for hypothesis testing.

Each Gate wraps a stateless check function and produces a GateResult with
uniform fields: name, passed, metric, threshold, p_value, message.

Usage in runners:
    gates = [ICGate(min_ic=0.10), DeltaICGate(min_dIC=0.05)]
    for gate in gates:
        result = gate.evaluate(report, thresholds)
        if not result.passed:
            return False, result.message
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


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
# Concrete gates — wrap the stateless check functions from base.py
# ---------------------------------------------------------------------------

class ICGate:
    """Check if IC exceeds minimum threshold."""
    name = "IC"

    def evaluate(self, report: dict, thresholds: dict) -> GateResult:
        from .base import check_ic_gate, _ic_pvalue, _find_gate_entry

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
        from .base import _find_gate_entry

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
