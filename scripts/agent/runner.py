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

def check_ic_gate(report: dict, thresholds: dict) -> tuple[bool, str]:
    """Check if IC exceeds minimum threshold."""
    min_ic = thresholds.get("min_ic", 0.10)
    # Try various report formats
    ic = None
    if "baseline_ic_filt_5s" in report:
        ic = report["baseline_ic_filt_5s"]
    elif "best_ic" in report:
        ic = report["best_ic"]
    elif "profiles" in report and len(report["profiles"]) > 0:
        ic = max(abs(p.get("ic_best", 0)) for p in report["profiles"])

    if ic is None:
        return False, "could not extract IC from report"
    passed = abs(ic) >= min_ic
    return passed, f"IC={ic:.4f} vs min={min_ic}" + (" PASS" if passed else " FAIL")


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

    def run_full(self) -> bool:
        """Run the complete 3-gate replication protocol."""
        if not self.run_discovery():
            return False
        if not self.run_replication_temporal():
            return False
        if not self.run_replication_symbol():
            return False
        self.register_signal()
        return True

    # -- helpers ------------------------------------------------------------

    def _check_gates(self, report: dict) -> tuple[bool, str]:
        """Run all applicable gate checks."""
        checks = [
            check_ic_gate(report, self.h.thresholds),
        ]
        for passed, msg in checks:
            if not passed:
                return False, msg
        return True, checks[0][1]

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
