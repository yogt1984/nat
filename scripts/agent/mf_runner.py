"""Medium-frequency experiment runner — 4-gate protocol at 5min bar resolution.

Gate protocol:
    DISCOVERY → TEMPORAL REPLICATION → SYMBOL REPLICATION → CORRELATION DEDUP → REGISTER

Reuses gate check functions from runner.py (check_ic_gate, check_dIC_gate,
check_correlation_gate) with medium-frequency-appropriate thresholds.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .base import BaseRunner
from .hypothesis import Hypothesis, RegisteredSignal
from .runner import (
    run_nat_cached,
    parse_report,
    check_ic_gate,
    check_dIC_gate,
    check_correlation_gate,
    _load_feature_data,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
MF_REGISTRY_PATH = ROOT / "data" / "agent_mf" / "registry.json"

# Medium-frequency signal features that may appear in claims
MF_SIGNAL_FEATURES = [
    "trend_momentum_300", "trend_momentum_r2_300", "trend_hurst_300",
    "trend_ma_crossover_norm", "vol_ratio_short_long", "vol_zscore",
    "imbalance_qty_l5", "flow_aggressor_ratio_5s", "flow_volume_5s",
]


class MediumFrequencyRunner(BaseRunner):
    """Runs a medium-frequency hypothesis through the 4-gate protocol."""

    TIMEFRAME = "5min"

    def steps(self) -> list:
        """4-gate protocol: discovery -> temporal -> symbol -> dedup."""
        return [
            self.run_discovery,
            self.run_replication_temporal,
            self.run_replication_symbol,
            self.run_correlation_check,
        ]

    def run_discovery(self) -> bool:
        """Execute the test protocol and check IC/dIC gates."""
        log.info("=== MF DISCOVERY: %s ===", self.h.claim[:80])

        for i, cmd_str in enumerate(self.h.test_protocol):
            cmd_parts = cmd_str.split()
            symbol = self._extract_symbol(cmd_str)
            result, report = run_nat_cached(
                cmd_parts, symbol=symbol,
            )

            if result.returncode != 0:
                self.h.fail("command_error")
                self.h.results = {"failed_cmd": cmd_str, "stderr": result.stderr[:500]}
                return False

            if report is None:
                # Try parsing with timeframe
                report = parse_report(
                    cmd_str, symbol=symbol, timeframe=self.TIMEFRAME,
                )

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
        log.info("  MF DISCOVERY PASSED: %s", self.h.claim[:60])
        return True

    def run_replication_temporal(self) -> bool:
        """Re-run on other available dates with --timeframe propagated."""
        dates = list(self.manifest.get("dates", {}).keys())
        if len(dates) < 2:
            log.warning("Only %d dates available, skipping temporal replication", len(dates))
            return True

        n_pass = 0
        n_tested = 0
        for date in dates[1:3]:
            data_dir = f"data/features/{date}"
            for cmd_str in self.h.test_protocol[:1]:
                cmd_parts = cmd_str.split()
                # Replace data dir
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
                # Ensure --timeframe is present
                if "--timeframe" not in cmd_str:
                    new_parts.extend(["--timeframe", self.TIMEFRAME])

                symbol = self._extract_symbol(cmd_str)
                result, _ = run_nat_cached(new_parts, symbol=symbol)
                n_tested += 1
                if result.returncode == 0:
                    n_pass += 1

        min_dates = self.h.thresholds.get("min_oos_dates", 2)
        if n_pass >= min_dates:
            log.info("  MF TEMPORAL REPLICATION PASSED: %d/%d dates", n_pass, n_tested)
            return True
        else:
            self.h.fail("no_replication")
            log.info("  MF TEMPORAL REPLICATION FAILED: %d/%d dates", n_pass, n_tested)
            return False

    def run_replication_symbol(self) -> bool:
        """Re-run on other symbols."""
        primary_sym = self._extract_symbol(self.h.test_protocol[0])
        other_symbols = [s for s in ["BTC", "ETH", "SOL"] if s != primary_sym]

        n_pass = 0
        passed_symbols = [primary_sym]
        failed_symbols = []
        for sym in other_symbols:
            for cmd_str in self.h.test_protocol[:1]:
                new_cmd = cmd_str.replace(f"--symbol {primary_sym}", f"--symbol {sym}")
                cmd_parts = new_cmd.split()
                result, report = run_nat_cached(cmd_parts, symbol=sym)
                if result.returncode == 0 and report:
                    passed, _ = self._check_gates(report)
                    if passed:
                        n_pass += 1
                        passed_symbols.append(sym)
                    else:
                        failed_symbols.append(sym)
                else:
                    failed_symbols.append(sym)

        self.h.results = {
            **(self.h.results or {}),
            "symbol_replication": {
                "passed": passed_symbols,
                "failed": failed_symbols,
                "n_pass": n_pass,
                "n_total": len(other_symbols),
            },
        }

        min_symbols = self.h.thresholds.get("min_symbols", 2) - 1
        if n_pass >= min_symbols:
            self.h.replicate()
            log.info("  MF SYMBOL REPLICATION PASSED: %d/%d", n_pass, len(other_symbols))
            return True
        else:
            self.h.fail("no_replication")
            log.info("  MF SYMBOL REPLICATION FAILED: %d/%d", n_pass, len(other_symbols))
            return False

    def run_correlation_check(self) -> bool:
        """Check if signal is redundant with existing MF registry entries."""
        registry = self._load_registry()
        if not registry:
            return True

        data_dir = self._extract_data_dir()
        symbol = self._extract_symbol(self.h.test_protocol[0])
        features = self._extract_features()
        gate = self.h.thresholds.get("regime_gate")
        max_corr = self.h.thresholds.get("max_corr", 0.70)

        for feat in features:
            passed, msg = check_correlation_gate(
                feat, gate, registry, data_dir, symbol, max_corr,
            )
            log.info("  MF Correlation check (%s): %s", feat, msg)
            self.h.results = {**(self.h.results or {}), "correlation_check": msg}
            if not passed:
                self.h.fail("redundant")
                return False
        return True

    def register_signal(self) -> RegisteredSignal:
        """Create a RegisteredSignal and write to MF registry."""
        horizon_s = self.h.thresholds.get("horizon_s", 300.0)
        signal = RegisteredSignal(
            name=self.h.claim,
            features=self._extract_features(),
            regime_gate=self.h.thresholds.get("regime_gate"),
            extraction=self.h.thresholds.get("extraction", "raw"),
            horizon_s=horizon_s,
            expected_ic=self._extract_ic_from_results(),
            symbols=["BTC", "ETH", "SOL"],
            discovery_date=self.h.created[:10],
            last_validated=datetime.now(timezone.utc).isoformat()[:10],
            hypothesis_id=self.h.id,
        )
        registry = self._load_registry()
        registry.append(signal.to_dict())
        MF_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MF_REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
        log.info("  MF REGISTERED: %s (IC=%.3f, horizon=%.0fs)",
                 signal.name, signal.expected_ic, horizon_s)
        return signal

    # -- helpers --------------------------------------------------------------

    def _check_gates(self, report: dict) -> tuple[bool, str]:
        """Run IC and dIC gate checks."""
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

    def _extract_features(self) -> list[str]:
        """Extract signal feature names from the hypothesis claim."""
        claim = self.h.claim.lower()
        features = []
        for f in MF_SIGNAL_FEATURES:
            if f in claim:
                features.append(f)
        return features or ["trend_momentum_300"]

    def _extract_ic_from_results(self) -> float:
        """Extract the IC value from gate results."""
        if self.h.results and "gate_results" in self.h.results:
            for g in self.h.results["gate_results"]:
                msg = g.get("msg", "")
                if "IC=" in msg and "dIC=" not in msg:
                    try:
                        return float(msg.split("IC=")[1].split()[0])
                    except (IndexError, ValueError):
                        pass
        return 0.0

    @staticmethod
    def _load_registry() -> list[dict]:
        """Load MF-specific registry (separate from microstructure)."""
        if MF_REGISTRY_PATH.exists():
            with open(MF_REGISTRY_PATH) as f:
                return json.load(f)
        return []
