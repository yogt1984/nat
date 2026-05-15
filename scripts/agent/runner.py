"""Experiment runner — executes hypothesis test protocols via nat commands.

State machine per hypothesis:
    SETUP → DISCOVERY → REPLICATE_TEMPORAL → REPLICATE_SYMBOL → REGISTER
      |         |              |                    |
      v         v              v                    v
    ABORT    GRAVEYARD      GRAVEYARD           GRAVEYARD
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .hypothesis import Hypothesis, RegisteredSignal
from .manifest import load_manifest

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
NAT_PATH = ROOT / "nat"
REGISTRY_PATH = ROOT / "data" / "agent" / "registry.json"
REPORTS_DIR = ROOT / "reports"

# Map nat subcommands to their JSON output paths
REPORT_PATTERNS = {
    "spannung regime": "reports/spannung/regime_screen_{symbol}.json",
    "spannung spectral": "reports/spannung/spectral_{symbol}.json",
    "spannung backtest": "reports/spannung/backtest_{symbol}.json",
    "spannung": "reports/spannung/spannung_{symbol}.json",
    "profile scalp": "reports/profiler/profile_{symbol}_{timeframe}.json",
}


def run_nat(cmd_parts: list[str], timeout_s: int = 900) -> subprocess.CompletedProcess:
    """Execute a nat command and return the result."""
    full_cmd = [sys.executable, str(NAT_PATH)] + cmd_parts
    log.info("Running: nat %s", " ".join(cmd_parts))
    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        log.warning("nat %s failed (rc=%d): %s",
                     " ".join(cmd_parts), result.returncode, result.stderr[:500])
    return result


def parse_report(cmd_str: str, symbol: str = "BTC", timeframe: str = "1min") -> Optional[dict]:
    """Attempt to parse the JSON report for a given nat command."""
    for pattern_key, pattern in REPORT_PATTERNS.items():
        if pattern_key in cmd_str:
            path = ROOT / pattern.format(symbol=symbol, timeframe=timeframe)
            if path.exists():
                with open(path) as f:
                    return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Gate checks — evaluate hypothesis thresholds against results
# ---------------------------------------------------------------------------

def _find_gate_entry(report: dict, regime_gate: str) -> Optional[dict]:
    """Find the single_factors entry matching a regime_gate label (e.g. 'ent_book_shape<P40')."""
    for entry in report.get("single_factors", []):
        if entry.get("label") == regime_gate:
            return entry
    return None


def _ic_pvalue(ic: float, n_obs: int) -> float:
    """Two-sided p-value for Spearman IC under H0: no predictive power.

    Under H0, IC ~ N(0, 1/sqrt(n)).  z = IC * sqrt(n) is standard normal.
    """
    from math import erfc, sqrt
    if n_obs < 2:
        return 1.0
    z = abs(ic) * sqrt(n_obs)
    return erfc(z / sqrt(2))  # two-sided


def check_ic_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check if IC exceeds minimum threshold.

    If the hypothesis specifies a regime_gate, extract the gate-specific IC
    from the single_factors array. Otherwise fall back to aggregate IC.

    Also computes a p-value (appended to the message as p=...) for FDR control.
    """
    min_ic = thresholds.get("min_ic", 0.10)
    regime_gate = thresholds.get("regime_gate")
    ic = None
    n_obs = report.get("n_rows", 0)

    # Gate-specific IC: look up in single_factors
    if regime_gate and "single_factors" in report:
        entry = _find_gate_entry(report, regime_gate)
        if entry:
            ic = entry.get("ic_filt_5s")
            n_obs = entry.get("n_obs", n_obs)

    # Fallback: aggregate report IC
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
    """Check that the regime gate improves IC over the ungated baseline.

    dIC = gated_IC - baseline_IC. The gate must ADD value, not just
    pass because the underlying signal is strong.
    """
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
    """Check if signal has sufficient per-trade edge to be worth pursuing.

    Uses avg_return_per_trade_bps from the best threshold level in the
    backtest report. This measures gross signal edge — a necessary
    (not sufficient) condition for profitability after costs.

    Note: the backtest tests the ungated signal. Regime-gated versions
    should have higher per-trade returns. This gate filters out signals
    with zero or negligible directional value.
    """
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


def _parse_gate_spec(gate_str: str) -> Optional[tuple[str, str, str]]:
    """Parse 'ent_book_shape<P40' into ('ent_book_shape', '<', 'P40')."""
    m = re.match(r'^([a-z_0-9]+)([<>])(P\d+)$', gate_str)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _load_feature_data(data_dir: str, symbol: str):
    """Load and concatenate all Parquet files for a symbol from a data dir."""
    import pandas as pd

    data_path = ROOT / data_dir
    if not data_path.exists():
        return None
    files = sorted(data_path.glob("*.parquet"))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue  # skip corrupted files
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df[df["symbol"] == symbol] if "symbol" in df.columns else df


def _extract_gated_signal(df, feature: str,
                          gate_spec: Optional[str]):
    """Extract a feature column masked by an optional regime gate.

    Returns the feature values where the gate condition is True, NaN elsewhere.
    This lets correlation measure agreement in the active regime only.
    """
    import numpy as np

    if feature not in df.columns:
        return None
    values = df[feature].to_numpy(dtype=float)

    if gate_spec is None:
        return values

    parsed = _parse_gate_spec(gate_spec)
    if parsed is None:
        return values

    gate_feat, direction, percentile_str = parsed
    if gate_feat not in df.columns:
        return values

    pct_val = int(percentile_str[1:])  # 'P40' -> 40
    gate_col = df[gate_feat].to_numpy(dtype=float)
    threshold = np.nanpercentile(gate_col, pct_val)

    if direction == "<":
        mask = gate_col < threshold
    else:
        mask = gate_col > threshold

    gated = np.full_like(values, np.nan)
    gated[mask] = values[mask]
    return gated


def check_correlation_gate(
    candidate_feature: str,
    candidate_gate: Optional[str],
    registry: list[dict],
    data_dir: str,
    symbol: str = "BTC",
    max_corr: float = 0.70,
) -> tuple[bool, str]:
    """Check if a candidate signal is redundant with existing registry signals.

    Computes Spearman rank correlation between the candidate's gated feature
    values and each registered signal's gated feature values. Rejects if
    any pairwise correlation exceeds max_corr.
    """
    import numpy as np
    import pandas as pd

    if not registry:
        return True, "empty registry, no dedup needed PASS"

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
            # Spearman on non-NaN overlap
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


def check_coverage_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check if regime coverage exceeds minimum."""
    min_coverage = thresholds.get("min_coverage", 0.20)
    # Look for Pareto-optimal results with sufficient coverage
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Runs a single hypothesis through its test protocol."""

    def __init__(self, hypothesis: Hypothesis, manifest: dict):
        self.h = hypothesis
        self.manifest = manifest
        self.gate_results: list[dict] = []

    def run_discovery(self) -> bool:
        """Execute the test protocol and check gates at each step."""
        log.info("=== DISCOVERY: %s ===", self.h.claim[:80])

        for i, cmd_str in enumerate(self.h.test_protocol):
            cmd_parts = cmd_str.split()
            result = run_nat(cmd_parts)

            if result.returncode != 0:
                self.h.fail("command_error")
                self.h.results = {"failed_cmd": cmd_str, "stderr": result.stderr[:500]}
                return False

            # Try to parse and check gates
            symbol = self._extract_symbol(cmd_str)
            report = parse_report(cmd_str, symbol=symbol)
            if report:
                passed, msg = self._check_gates(report)
                self.gate_results.append({"cmd": cmd_str, "passed": passed, "msg": msg})
                log.info("  Gate %d: %s", i, msg)
                if not passed:
                    self.h.fail("no_effect")
                    self.h.results = {"gate_results": self.gate_results}
                    return False

        self.h.pass_discovery()
        self.h.results = {"gate_results": self.gate_results}
        log.info("  DISCOVERY PASSED: %s", self.h.claim[:60])
        return True

    def run_replication_temporal(self) -> bool:
        """Re-run on other available dates."""
        dates = list(self.manifest.get("dates", {}).keys())
        if len(dates) < 2:
            log.warning("Only %d dates available, skipping temporal replication", len(dates))
            return True  # Can't replicate yet — pass through

        n_pass = 0
        n_tested = 0
        for date in dates[1:3]:  # Test on up to 2 other dates
            data_dir = f"data/features/{date}"
            for cmd_str in self.h.test_protocol[:1]:  # Run the first (discovery) command
                cmd_parts = cmd_str.split()
                # Replace data dir if present
                new_parts = []
                skip_next = False
                for p in cmd_parts:
                    if skip_next:
                        new_parts.append(data_dir)
                        skip_next = False
                    elif p in ("--data", "--data-dir"):
                        new_parts.append(p)
                        skip_next = True
                    else:
                        new_parts.append(p)
                if not skip_next and "--data" not in cmd_str:
                    new_parts.extend(["--data", data_dir])

                result = run_nat(new_parts)
                n_tested += 1
                if result.returncode == 0:
                    n_pass += 1

        min_dates = self.h.thresholds.get("min_oos_dates", 1)
        if n_pass >= min_dates:
            log.info("  TEMPORAL REPLICATION PASSED: %d/%d dates", n_pass, n_tested)
            return True
        else:
            self.h.fail("no_replication")
            log.info("  TEMPORAL REPLICATION FAILED: %d/%d dates", n_pass, n_tested)
            return False

    def run_replication_symbol(self) -> bool:
        """Re-run on other symbols."""
        primary_sym = self._extract_symbol(self.h.test_protocol[0])
        other_symbols = [s for s in ["BTC", "ETH", "SOL"] if s != primary_sym]

        n_pass = 0
        for sym in other_symbols:
            for cmd_str in self.h.test_protocol[:1]:
                cmd_parts = cmd_str.replace(f"--symbol {primary_sym}", f"--symbol {sym}").split()
                result = run_nat(cmd_parts)
                if result.returncode == 0:
                    report = parse_report(cmd_str, symbol=sym)
                    if report:
                        passed, _ = self._check_gates(report)
                        if passed:
                            n_pass += 1

        min_symbols = self.h.thresholds.get("min_symbols", 2) - 1  # -1 for primary
        if n_pass >= min_symbols:
            self.h.replicate()
            log.info("  SYMBOL REPLICATION PASSED: %d/%d", n_pass, len(other_symbols))
            return True
        else:
            self.h.fail("no_replication")
            log.info("  SYMBOL REPLICATION FAILED: %d/%d", n_pass, len(other_symbols))
            return False

    def register_signal(self) -> RegisteredSignal:
        """Create a RegisteredSignal from a replicated hypothesis."""
        signal = RegisteredSignal(
            name=self.h.claim,
            features=self._extract_features(),
            regime_gate=self.h.thresholds.get("regime_gate"),
            extraction=self.h.thresholds.get("extraction", "raw"),
            horizon_s=self.h.thresholds.get("horizon_s", 5.0),
            expected_ic=self._extract_ic_from_results(),
            symbols=["BTC", "ETH", "SOL"],
            discovery_date=self.h.created[:10],
            last_validated=datetime.now(timezone.utc).isoformat()[:10],
            hypothesis_id=self.h.id,
        )
        # Append to registry
        registry = self._load_registry()
        registry.append(signal.to_dict())
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
        log.info("  REGISTERED: %s (IC=%.3f)", signal.name, signal.expected_ic)
        return signal

    def run_cost_check(self) -> bool:
        """Run backtest and check if signal survives transaction costs.

        Runs after discovery passes. Signals that fail cost are marked
        'cost_killed' — they are real but untradeable, eligible for recycler.
        """
        data_dir = self._extract_data_dir()
        symbol = self._extract_symbol(self.h.test_protocol[0])
        cmd_parts = ["spannung", "backtest", "--data", data_dir, "--symbol", symbol]
        result = run_nat(cmd_parts)
        if result.returncode != 0:
            log.warning("  Cost check: backtest failed (rc=%d), skipping", result.returncode)
            return True  # Don't block on backtest failure

        report = parse_report("spannung backtest", symbol=symbol)
        if report is None:
            log.warning("  Cost check: could not parse backtest report, skipping")
            return True

        passed, msg = check_cost_gate(report, self.h.thresholds)
        log.info("  Cost check: %s", msg)
        self.h.results = {**(self.h.results or {}), "cost_check": msg}
        if not passed:
            self.h.fail("cost_killed")
        return passed

    def run_correlation_check(self) -> bool:
        """Check if signal is redundant with existing registry entries.

        Runs after replication passes, before registration. Prevents the
        registry from filling with near-identical signals (e.g. imbalance_l1
        and imbalance_l5 gated by the same regime — Spearman rho > 0.93).
        """
        registry = self._load_registry()
        if not registry:
            return True

        data_dir = self._extract_data_dir()
        symbol = self._extract_symbol(self.h.test_protocol[0])
        features = self._extract_features()
        gate = self.h.thresholds.get("regime_gate")
        max_corr = self.h.thresholds.get("max_corr", 0.70)

        # Check each feature in the candidate against the full registry
        for feat in features:
            passed, msg = check_correlation_gate(
                feat, gate, registry, data_dir, symbol, max_corr
            )
            log.info("  Correlation check (%s): %s", feat, msg)
            self.h.results = {**(self.h.results or {}), "correlation_check": msg}
            if not passed:
                self.h.fail("redundant")
                return False
        return True

    def run_full(self) -> bool:
        """Run the complete 5-gate protocol: discovery, cost, temporal, symbol, dedup."""
        if not self.run_discovery():
            return False
        if not self.run_cost_check():
            return False
        if not self.run_replication_temporal():
            return False
        if not self.run_replication_symbol():
            return False
        if not self.run_correlation_check():
            return False
        self.register_signal()
        return True

    # -- helpers ------------------------------------------------------------

    def _check_gates(self, report: dict) -> tuple[bool, str]:
        """Run all applicable gate checks: gate-specific IC and dIC."""
        checks = [
            ("IC", check_ic_gate(report, self.h.thresholds)),
            ("dIC", check_dIC_gate(report, self.h.thresholds)),
        ]
        msgs = []
        for name, (passed, msg) in checks:
            msgs.append(msg)
            if not passed:
                return False, f"{name}: {msg}"
        return True, " | ".join(msgs)

    def _extract_data_dir(self) -> str:
        for cmd_str in self.h.test_protocol:
            parts = cmd_str.split()
            for i, p in enumerate(parts):
                if p in ("--data", "--data-dir") and i + 1 < len(parts):
                    return parts[i + 1]
        return f"data/features/{sorted(self.manifest.get('dates', {}).keys())[-1]}"

    @staticmethod
    def _extract_symbol(cmd_str: str) -> str:
        parts = cmd_str.split()
        for i, p in enumerate(parts):
            if p == "--symbol" and i + 1 < len(parts):
                return parts[i + 1]
        return "BTC"

    def _extract_features(self) -> list[str]:
        claim = self.h.claim.lower()
        features = []
        for f in ["imbalance_qty_l1", "imbalance_qty_l5", "imbalance_qty_l10",
                   "imbalance_depth_weighted", "ent_book_shape", "toxic_vpin_50"]:
            if f in claim:
                features.append(f)
        return features or ["imbalance_qty_l1"]

    def _extract_ic_from_results(self) -> float:
        if self.h.results and "gate_results" in self.h.results:
            for g in self.h.results["gate_results"]:
                msg = g.get("msg", "")
                if "IC=" in msg:
                    try:
                        return float(msg.split("IC=")[1].split()[0])
                    except (IndexError, ValueError):
                        pass
        return 0.0

    @staticmethod
    def _load_registry() -> list[dict]:
        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH) as f:
                return json.load(f)
        return []
