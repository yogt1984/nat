"""Base classes for autonomous research agents.

Provides:
- ResearchAgent ABC — cycle loop, state machine, generator dispatch, FDR
- BaseRunner ABC — sequential gate protocol execution
- AgentPhase, AgentState — shared infrastructure
"""

from __future__ import annotations

import logging
import re
import signal as signal_mod
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from .hypothesis import Hypothesis, GeneratorStats
from .hypothesis_queue import HypothesisQueue

log = logging.getLogger("nat.agent")


# ---------------------------------------------------------------------------
# Agent state (persistent)
# ---------------------------------------------------------------------------

class AgentPhase(str, Enum):
    IDLE = "IDLE"
    MANIFEST = "MANIFEST"
    GENERATE = "GENERATE"
    EXECUTE = "EXECUTE"
    MONITOR = "MONITOR"
    SLEEPING = "SLEEPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class AgentState:
    """Persistent agent state backed by SQLite (via StateStore)."""

    _DEFAULTS = {
        "phase": AgentPhase.IDLE.value,
        "cycle_count": 0,
        "total_hypotheses_tested": 0,
        "total_signals_registered": 0,
        "current_hypothesis": None,
        "started_at": None,
        "last_cycle_at": None,
    }

    def __init__(self, *, store, agent: str = "agent"):
        self._store = store
        self._agent = agent
        self._data = self._load()

    def _load(self) -> dict:
        loaded = self._store.load_state(self._agent)
        if loaded:
            loaded["history"] = self._store.load_history(
                self._agent, limit=200)
            return loaded
        return {**self._DEFAULTS, "history": []}

    def save(self) -> None:
        self._store.save_state(self._agent, self._data)

    def transition(self, phase: AgentPhase, msg: str = "") -> None:
        old = self._data["phase"]
        self._data["phase"] = phase.value
        entry = {
            "from": old, "to": phase.value,
            "at": datetime.now(timezone.utc).isoformat(),
            "msg": msg,
        }
        self._data.setdefault("history", []).append(entry)
        # Keep in-memory history bounded
        if len(self._data["history"]) > 500:
            self._data["history"] = self._data["history"][-200:]
        self._store.save_state(self._agent, self._data)
        self._store.append_history(self._agent, entry)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self.save()

    @property
    def phase(self) -> AgentPhase:
        return AgentPhase(self._data["phase"])


# ---------------------------------------------------------------------------
# FDR control (generic statistical method)
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


# ---------------------------------------------------------------------------
# Config inheritance
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (neither dict is mutated)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# Keys recognised at the top level and in common subsections.
# Unknown keys trigger a warning (not an error) so legacy configs keep working.
_KNOWN_TOP_KEYS = {
    "cycle_interval_s", "max_experiments_per_cycle", "max_cycle_runtime_s",
    "timeframe", "generators_enabled",
    "gates", "cost", "decay", "promotion", "symbols", "paths",
}
_KNOWN_GATE_KEYS = {
    "min_ic", "min_dIC", "min_coverage", "fdr_q",
    "min_walkforward_sign_consistency", "min_oos_dates", "min_symbols",
}


_REQUIRED_KEYS = {"cycle_interval_s", "max_experiments_per_cycle", "generators_enabled"}


def validate_config(config: dict, section: str) -> list[str]:
    """Validate agent config after defaults merge.

    Returns warnings for unknown keys.
    Raises ValueError if required keys are missing.
    """
    # Required-key check (hard error)
    missing = _REQUIRED_KEYS - config.keys()
    if missing:
        raise ValueError(
            f"[{section}] missing required keys: {sorted(missing)}"
        )

    # Unknown-key check (soft warning)
    warnings = []
    for key in config:
        if key not in _KNOWN_TOP_KEYS:
            warnings.append(f"[{section}] unknown key: {key!r}")
    gates = config.get("gates", {})
    for key in gates:
        if key not in _KNOWN_GATE_KEYS:
            warnings.append(f"[{section}.gates] unknown key: {key!r}")
    return warnings


def load_agent_config(config_path: Path, section: str,
                      base_config: dict) -> dict:
    """Load agent config with inheritance: base_config → [defaults] → [section].

    Deep-merges nested subsections (gates, decay, symbols, paths) so that
    a section only needs to override the keys that differ from [defaults].
    Injects symbols.primary from config/symbols.toml if not set.
    """
    if not config_path.exists():
        return dict(base_config)
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    defaults = raw.get("defaults", {})
    section_cfg = raw.get(section, {})
    merged = _deep_merge(dict(base_config), defaults)
    merged = _deep_merge(merged, section_cfg)

    # Inject canonical symbols if not provided by TOML
    if "symbols" not in merged or "primary" not in merged.get("symbols", {}):
        try:
            from scripts.config_utils import load_symbols
        except ImportError:
            from config_utils import load_symbols
        merged.setdefault("symbols", {})["primary"] = load_symbols()

    for w in validate_config(merged, section):
        log.warning("Config: %s", w)
    return merged


# ---------------------------------------------------------------------------
# Gate check functions — shared across all runners
# ---------------------------------------------------------------------------

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
# BaseRunner ABC
# ---------------------------------------------------------------------------

class BaseRunner(ABC):
    """Base for running a hypothesis through a gate protocol.

    Provides default 4-gate protocol (discovery → temporal → symbol → dedup)
    and shared registration logic. Subclasses configure via class attributes
    and override ``steps()`` for different gate counts.

    Class attributes for subclass configuration:
        TIMEFRAME: None for tick-level (micro), "5min" (MF), "1h" (macro)
        SIGNAL_FEATURES: feature names to match in hypothesis claims
        DEFAULT_FEATURE: fallback when no feature matched in claim
        DEFAULT_HORIZON_S: default signal horizon in seconds
        REGISTRY_PATH: Path to the signal registry JSON file
    """

    TIMEFRAME: str | None = None
    SIGNAL_FEATURES: list[str] = []
    DEFAULT_FEATURE: str = "unknown"
    DEFAULT_HORIZON_S: float = 5.0
    REGISTRY_PATH: Path | None = None

    def __init__(self, hypothesis: Hypothesis, manifest: dict, *,
                 store=None, agent: str | None = None):
        self.h = hypothesis
        self.manifest = manifest
        self.gate_results: list[dict] = []
        self._store = store
        self._agent = agent

    def run_full(self) -> bool:
        """Run all gate steps sequentially. Register signal on full success."""
        for step in self.steps():
            if not step():
                return False
        self.register_signal()
        return True

    def steps(self) -> list:
        """Default 4-gate protocol: discovery → temporal → symbol → dedup."""
        return [
            self.run_discovery,
            self.run_replication_temporal,
            self.run_replication_symbol,
            self.run_correlation_check,
        ]

    def _publish_event(self, event_type: str, payload: dict) -> None:
        """Publish a research event to Redis (best-effort)."""
        try:
            from .research_output import publish_research_event
            publish_research_event(event_type, payload)
        except Exception:
            pass

    # --- Gate methods (shared across all runners) -------------------------

    def run_discovery(self) -> bool:
        """Execute the test protocol and check IC/dIC gates."""
        from .runner import run_nat_cached, parse_report

        log.info("=== DISCOVERY: %s ===", self.h.claim[:80])

        for i, cmd_str in enumerate(self.h.test_protocol):
            cmd_parts = cmd_str.split()
            symbol = self._extract_symbol(cmd_str)
            result, report = run_nat_cached(cmd_parts, symbol=symbol)

            if result.returncode != 0:
                self.h.fail("command_error")
                self.h.results = {"failed_cmd": cmd_str, "stderr": result.stderr[:500]}
                return False

            # Bar-resampled runners try timeframe-aware parse as fallback
            if report is None and self.TIMEFRAME is not None:
                report = parse_report(
                    cmd_str, symbol=symbol, timeframe=self.TIMEFRAME,
                )

            if report:
                passed, msg = self._check_gates(report)
                self.gate_results.append({"cmd": cmd_str, "passed": passed, "msg": msg})
                gate_name = f"G{i + 1}_discovery"
                if passed:
                    self._publish_event("gate_passed", {
                        "id": self.h.id, "gate": gate_name, "msg": msg,
                    })
                else:
                    self._publish_event("gate_failed", {
                        "id": self.h.id, "gate": gate_name, "reason": msg,
                    })
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
        from .runner import run_nat_cached

        dates = list(self.manifest.get("dates", {}).keys())
        if len(dates) < 2:
            log.warning("Only %d dates available, skipping temporal replication",
                        len(dates))
            return True

        n_pass = 0
        n_tested = 0
        for date in dates[1:3]:
            data_dir = f"data/features/{date}"
            for cmd_str in self.h.test_protocol[:1]:
                cmd_parts = cmd_str.split()
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
                # Append --timeframe for bar-resampled agents
                if self.TIMEFRAME and "--timeframe" not in cmd_str:
                    new_parts.extend(["--timeframe", self.TIMEFRAME])

                symbol = self._extract_symbol(cmd_str)
                result, _ = run_nat_cached(new_parts, symbol=symbol)
                n_tested += 1
                if result.returncode == 0:
                    n_pass += 1

        # Tick-level agents default to min_oos_dates=1, bar-resampled to 2
        default_min = 1 if self.TIMEFRAME is None else 2
        min_dates = self.h.thresholds.get("min_oos_dates", default_min)
        if n_pass >= min_dates:
            self._publish_event("gate_passed", {
                "id": self.h.id, "gate": "G2_temporal",
                "msg": f"{n_pass}/{n_tested} dates passed",
            })
            log.info("  TEMPORAL REPLICATION PASSED: %d/%d dates", n_pass, n_tested)
            return True
        else:
            self._publish_event("gate_failed", {
                "id": self.h.id, "gate": "G2_temporal",
                "reason": f"only {n_pass}/{n_tested} dates passed",
            })
            self.h.fail("no_replication")
            log.info("  TEMPORAL REPLICATION FAILED: %d/%d dates", n_pass, n_tested)
            return False

    def run_replication_symbol(self) -> bool:
        """Re-run on other symbols."""
        from .runner import run_nat_cached

        primary_sym = self._extract_symbol(self.h.test_protocol[0])
        try:
            from scripts.config_utils import load_symbols
        except ImportError:
            from config_utils import load_symbols
        other_symbols = [s for s in load_symbols() if s != primary_sym]

        n_pass = 0
        passed_symbols = [primary_sym]
        failed_symbols = []
        for sym in other_symbols:
            for cmd_str in self.h.test_protocol[:1]:
                new_cmd = cmd_str.replace(
                    f"--symbol {primary_sym}", f"--symbol {sym}")
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
            self._publish_event("gate_passed", {
                "id": self.h.id, "gate": "G3_symbol",
                "msg": f"{n_pass}/{len(other_symbols)} symbols passed",
            })
            log.info("  SYMBOL REPLICATION PASSED: %d/%d", n_pass, len(other_symbols))
            return True
        else:
            self._publish_event("gate_failed", {
                "id": self.h.id, "gate": "G3_symbol",
                "reason": f"only {n_pass}/{len(other_symbols)} symbols passed",
            })
            self.h.fail("no_replication")
            log.info("  SYMBOL REPLICATION FAILED: %d/%d", n_pass, len(other_symbols))
            return False

    def run_correlation_check(self) -> bool:
        """Check if signal is redundant with existing registry entries."""
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
            log.info("  Correlation check (%s): %s", feat, msg)
            self.h.results = {**(self.h.results or {}), "correlation_check": msg}
            if not passed:
                self._publish_event("gate_failed", {
                    "id": self.h.id, "gate": "G4_correlation",
                    "reason": msg,
                })
                self.h.fail("redundant")
                return False
        self._publish_event("gate_passed", {
            "id": self.h.id, "gate": "G4_correlation",
            "msg": "not redundant",
        })
        return True

    # --- Registration (shared) --------------------------------------------

    def register_signal(self):
        """Register a validated signal into the registry."""
        from .hypothesis import RegisteredSignal

        horizon_s = self.h.thresholds.get("horizon_s", self.DEFAULT_HORIZON_S)
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
        self._store.append_signal(self._agent, signal.to_dict())
        log.info("  REGISTERED: %s (IC=%.3f, horizon=%.0fs)",
                 signal.name, signal.expected_ic, horizon_s)
        return signal

    # --- Helpers ----------------------------------------------------------

    def _check_gates(self, report: dict) -> tuple[bool, str]:
        """Run IC and dIC gate checks against hypothesis thresholds."""
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
        features = [f for f in self.SIGNAL_FEATURES if f in claim]
        return features or [self.DEFAULT_FEATURE]

    def _extract_ic_from_results(self) -> float:
        """Extract the IC value from gate results (excludes dIC messages)."""
        if self.h.results and "gate_results" in self.h.results:
            for g in self.h.results["gate_results"]:
                msg = g.get("msg", "")
                if "IC=" in msg and "dIC=" not in msg:
                    try:
                        return float(msg.split("IC=")[1].split()[0])
                    except (IndexError, ValueError):
                        pass
        return 0.0

    def _load_registry(self) -> list[dict]:
        """Load the signal registry from SQLite."""
        return self._store.load_registry(self._agent)

    @staticmethod
    def _extract_symbol(cmd_str: str) -> str:
        """Parse --symbol from a command string."""
        parts = cmd_str.split()
        for i, p in enumerate(parts):
            if p == "--symbol" and i + 1 < len(parts):
                return parts[i + 1]
        return "BTC"

    def _extract_data_dir(self) -> str:
        """Parse --data/--data-dir from test protocol commands."""
        for cmd_str in self.h.test_protocol:
            parts = cmd_str.split()
            for i, p in enumerate(parts):
                if p in ("--data", "--data-dir") and i + 1 < len(parts):
                    return parts[i + 1]
        dates = sorted(self.manifest.get("dates", {}).keys())
        return f"data/features/{dates[-1]}" if dates else "data/features"


# ---------------------------------------------------------------------------
# ResearchAgent ABC
# ---------------------------------------------------------------------------

class ResearchAgent(ABC):
    """Abstract base for autonomous hypothesis-driven research agents.

    Subclasses provide domain-specific hooks: generators, runner factory,
    monitoring logic. The base class owns the cycle loop, state machine,
    generator dispatch, FDR control, and stats tracking.

    Lifecycle:
        MANIFEST → GENERATE → EXECUTE → MONITOR → SLEEP → repeat
    """

    agent_type: str = "base"
    config_section: str = "agent"
    default_generators: list[str] = []
    generator_module_prefix: str = "agent.generators"  # importlib path prefix

    BASE_CONFIG = {
        "cycle_interval_s": 3600,
        "max_experiments_per_cycle": 10,
        "max_cycle_runtime_s": 5400,
    }

    # Rolling IC configuration — subclasses override these
    _rolling_ic_bar_period: str | None = None  # None = tick-level, "5min", "1h"
    _rolling_ic_horizon_default: float = 5.0   # seconds
    _rolling_ic_min_valid: int = 200
    _rolling_ic_feature_suffixes: tuple[str, ...] = ("",)
    _rolling_ic_use_gated_signal: bool = False

    def __init__(self, config: dict | None = None, *, store=None):
        self.config = config or self.load_config()
        self.config.setdefault("generators_enabled", list(self.default_generators))
        # SQLite state store — auto-created or injected for testing
        self._store = store or self._create_store()
        self.state = AgentState(store=self._store, agent=self.agent_type)
        self.queue = HypothesisQueue(store=self._store, agent=self.agent_type)
        self.gen_stats = self._load_gen_stats()
        self._shutdown = False
        signal_mod.signal(signal_mod.SIGTERM, self._handle_signal)
        signal_mod.signal(signal_mod.SIGINT, self._handle_signal)

    def _create_store(self):
        from data.state import StateStore
        return StateStore(self.root / "data" / "nat.db")

    # --- Paths (subclass overrides for testability / multi-agent) ----------

    @property
    def root(self) -> Path:
        """Project root directory."""
        return Path(__file__).resolve().parent.parent.parent

    @property
    def agent_dir(self) -> str:
        """Data subdirectory name. Override for non-default agents."""
        return "agent"

    @property
    def state_path(self) -> Path:
        return self.root / "data" / self.agent_dir / "agent_state.json"

    @property
    def queue_path(self) -> Path:
        return self.root / "data" / self.agent_dir / "hypotheses.json"

    @property
    def stats_path(self) -> Path:
        return self.root / "data" / self.agent_dir / "generator_stats.json"

    @property
    def registry_path(self) -> Path:
        return self.root / "data" / self.agent_dir / "registry.json"

    @property
    def research_output_root(self) -> Path | None:
        """Root for structured research output JSON files.

        Returns data/research/ by default. Set to None to disable output.
        """
        return self.root / "data" / "research"

    # --- Config -----------------------------------------------------------

    def load_config(self) -> dict:
        """Load config with inheritance: BASE_CONFIG → [defaults] → [section]."""
        return load_agent_config(
            self.root / "config" / "agent.toml",
            self.config_section,
            self.BASE_CONFIG,
        )

    # --- Generator stats --------------------------------------------------

    def _load_gen_stats(self) -> dict[str, GeneratorStats]:
        raw = self._store.load_gen_stats(self.agent_type)
        if raw:
            return {k: GeneratorStats(**v) for k, v in raw.items()}
        return {g: GeneratorStats()
                for g in self.config.get("generators_enabled", [])}

    def _save_gen_stats(self) -> None:
        data = {k: {"attempts": v.attempts, "successes": v.successes}
                for k, v in self.gen_stats.items()}
        self._store.save_gen_stats(self.agent_type, data)

    # --- Signal handling --------------------------------------------------

    def _handle_signal(self, signum, frame):
        log.info("Received signal %d, will stop after current experiment", signum)
        self._shutdown = True

    # --- Main loop --------------------------------------------------------

    def run(self) -> None:
        """Run the agent loop until shutdown."""
        self.state.set("started_at", datetime.now(timezone.utc).isoformat())
        self.state.transition(AgentPhase.IDLE, "agent started")
        log.info("%s agent started (cycle_interval=%ds)",
                 self.agent_type, self.config["cycle_interval_s"])

        while not self._shutdown:
            try:
                self.run_cycle()
            except Exception as e:
                log.error("Cycle error: %s", e, exc_info=True)
                self.state.transition(AgentPhase.ERROR, str(e))

            if self._shutdown:
                break

            self.state.transition(AgentPhase.SLEEPING, "waiting for next cycle")
            sleep_s = self.config["cycle_interval_s"]
            for _ in range(sleep_s):
                if self._shutdown:
                    break
                time.sleep(1)

        self.state.transition(AgentPhase.STOPPED, "graceful shutdown")
        log.info("%s agent stopped", self.agent_type)

    def run_cycle(self) -> None:
        """Execute one complete research cycle."""
        from logging_config import set_context, clear_context
        import uuid
        cycle_id = f"CYC-{uuid.uuid4().hex[:8]}"
        set_context(cycle_id=cycle_id, agent=self.agent_type)
        cycle_start = time.monotonic()

        # 1. Update manifest
        self.state.transition(AgentPhase.MANIFEST, "scanning data")
        manifest = self.build_manifest()

        # 2. Generate hypotheses
        self.state.transition(AgentPhase.GENERATE, "generating hypotheses")
        self._run_generators(manifest)

        # 3. Execute experiments (budget-limited)
        self.state.transition(AgentPhase.EXECUTE, "running experiments")
        n_run = 0
        n_registered = 0
        max_run = self.config["max_experiments_per_cycle"]
        max_time = self.config["max_cycle_runtime_s"]
        cycle_hypotheses = []

        while n_run < max_run and not self._shutdown:
            elapsed = time.monotonic() - cycle_start
            if elapsed > max_time:
                log.info("Cycle time budget exhausted (%.0fs)", elapsed)
                break

            hypothesis = self.queue.pop(manifest)
            if hypothesis is None:
                log.info("Queue empty, ending execution phase")
                break

            self.state.set("current_hypothesis", hypothesis.id)
            set_context(hypothesis_id=hypothesis.id)
            self._publish_event("hypothesis_started", {
                "id": hypothesis.id, "agent": self.agent_type,
                "claim": hypothesis.claim[:200],
                "generator": hypothesis.generator,
            })
            self.pre_execute(hypothesis)
            runner = self.create_runner(hypothesis, manifest)
            success = runner.run_full()

            # Update stats
            gen_name = hypothesis.generator
            if gen_name in self.gen_stats:
                self.gen_stats[gen_name].record(success)

            self.queue.update(hypothesis)
            cycle_hypotheses.append(hypothesis)
            self._emit_hypothesis_record(hypothesis)
            n_run += 1
            self.state.set("total_hypotheses_tested",
                          self.state.get("total_hypotheses_tested", 0) + 1)
            if success:
                n_registered += 1
                self.state.set("total_signals_registered",
                              self.state.get("total_signals_registered", 0) + 1)
                ic = None
                if hypothesis.results and hypothesis.results.get("gate_results"):
                    for gr in hypothesis.results["gate_results"]:
                        if "IC=" in gr.get("msg", ""):
                            import re as _re
                            m = _re.search(r"IC=([+-]?\d+\.?\d*)", gr["msg"])
                            if m:
                                ic = float(m.group(1))
                                break
                self._publish_event("hypothesis_registered", {
                    "id": hypothesis.id, "agent": self.agent_type,
                    "ic": ic,
                })

        # 3b. FDR control — retroactively reject hypotheses that don't survive
        #     Benjamini-Hochberg correction across this cycle's tests.
        fdr_q = self.config.get("gates", {}).get("fdr_q", 0.05)
        rejected_ids = apply_fdr(
            [h.to_dict() for h in cycle_hypotheses], q=fdr_q
        )
        if rejected_ids:
            for h in cycle_hypotheses:
                if h.id in rejected_ids and h.status == "replicated":
                    log.info("  FDR rejected: %s (was replicated)", h.claim[:60])
                    h.fail("fdr_rejected")
                    self.queue.update(h)
                    self.on_fdr_reject(h)
                    n_registered -= 1
                    self.state.set("total_signals_registered",
                                  max(0, self.state.get("total_signals_registered", 0) - 1))
            log.info("FDR control (q=%.2f): %d/%d rejected",
                     fdr_q, len(rejected_ids), len(cycle_hypotheses))

        # 3c. Post-cycle hooks (chaining, etc.)
        n_chained = self.post_cycle(cycle_hypotheses)
        if n_chained:
            log.info("Post-cycle: spawned %d follow-up hypotheses", n_chained)

        self._save_gen_stats()
        self.state.set("current_hypothesis", None)

        # 3d. Emit structured cycle summary
        self._emit_cycle_summary(
            cycle_id, cycle_start, cycle_hypotheses,
            n_registered, len(rejected_ids) if rejected_ids else 0,
            n_chained or 0, fdr_q,
        )

        # 4. Monitor registered signals
        self.state.transition(AgentPhase.MONITOR, "checking registered signals")
        self.run_monitor()

        # 5. Update cycle counter
        cycle_num = self.state.get("cycle_count", 0) + 1
        self.state.set("cycle_count", cycle_num)
        self.state.set("last_cycle_at", datetime.now(timezone.utc).isoformat())
        self._publish_event("cycle_completed", {
            "agent": self.agent_type, "cycle": cycle_num,
            "tested": n_run, "passed": n_registered,
        })
        log.info("Cycle complete: %d experiments, queue depth=%d",
                 n_run, self.queue.depth)
        clear_context()

    def _publish_event(self, event_type: str, payload: dict) -> None:
        """Publish a research event to Redis (best-effort, never throws)."""
        try:
            from .research_output import publish_research_event
            publish_research_event(event_type, payload)
        except Exception:
            pass

    def _emit_hypothesis_record(self, hypothesis) -> None:
        """Emit structured JSON record for a completed hypothesis."""
        try:
            from .research_output import build_hypothesis_record
            build_hypothesis_record(
                hypothesis,
                agent_type=self.agent_type,
                output_root=self.research_output_root,
            )
        except Exception as e:
            log.debug("Failed to emit hypothesis record: %s", e)

    def _emit_cycle_summary(
        self, cycle_id: str, cycle_start: float,
        hypotheses: list, n_registered: int, n_fdr_rejected: int,
        n_chained: int, fdr_q: float,
    ) -> None:
        """Emit structured JSON cycle summary."""
        try:
            from .research_output import build_cycle_summary
            started = datetime.fromtimestamp(
                time.time() - (time.monotonic() - cycle_start),
                tz=timezone.utc,
            ).isoformat()
            duration_s = time.monotonic() - cycle_start
            build_cycle_summary(
                cycle_id=cycle_id,
                agent_type=self.agent_type,
                started=started,
                duration_s=duration_s,
                hypotheses=hypotheses,
                n_registered=n_registered,
                n_fdr_rejected=n_fdr_rejected,
                n_chained=n_chained,
                fdr_q=fdr_q,
                generator_stats=self.gen_stats,
                output_root=self.research_output_root,
            )
        except Exception as e:
            log.debug("Failed to emit cycle summary: %s", e)

    def _run_generators(self, manifest: dict) -> int:
        """Run all enabled generators and push hypotheses into the queue."""
        total = 0
        for gen_name in self.config.get("generators_enabled",
                                        self.default_generators):
            try:
                gen_func = self.get_generator(gen_name)
                if gen_func is None:
                    continue
                hypotheses = gen_func(manifest, self.queue,
                                      self.gen_stats.get(gen_name))
                for h in hypotheses:
                    self.queue.push(h)
                    total += 1
            except Exception as e:
                log.warning("Generator %s failed: %s", gen_name, e)
        log.info("Generated %d new hypotheses", total)
        return total

    def print_status(self) -> None:
        """Print current agent status."""
        s = self.state
        print(f"Phase:       {s.phase.value}")
        print(f"Cycles:      {s.get('cycle_count', 0)}")
        print(f"Tested:      {s.get('total_hypotheses_tested', 0)} hypotheses")
        print(f"Registered:  {s.get('total_signals_registered', 0)} signals")
        print(f"Queue depth: {self.queue.depth}")
        print(f"Graveyard:   {len(self.queue.graveyard)}")
        print(f"Started:     {s.get('started_at', 'never')}")
        print(f"Last cycle:  {s.get('last_cycle_at', 'never')}")
        current = s.get("current_hypothesis")
        if current:
            print(f"Running:     {current}")

        # Generator stats
        print("\nGenerator performance:")
        for name, gs in self.gen_stats.items():
            print(f"  {name:15s}  attempts={gs.attempts:3d}  "
                  f"successes={gs.successes:3d}  "
                  f"hit_rate={gs.hit_rate:.0%}  "
                  f"weight={gs.weight:.3f}")

    # --- Hooks (subclass may override) --------------------------------------

    def get_generator(self, name: str):
        """Return the generate() function for a named generator, or None.

        Default implementation imports from ``{generator_module_prefix}.{name}``.
        Subclasses can override for custom import paths.
        """
        try:
            mod = __import__(f"{self.generator_module_prefix}.{name}",
                            fromlist=["generate"])
            return mod.generate
        except (ImportError, AttributeError) as e:
            log.debug("Generator %s not available: %s", name, e)
        return None

    @abstractmethod
    def create_runner(self, hypothesis: Hypothesis,
                      manifest: dict) -> BaseRunner:
        """Create an experiment runner for the given hypothesis."""
        ...

    def run_monitor(self) -> None:
        """Check registered signals for IC decay, auto-retire, and promotion.

        Combines IC decay monitoring with paper trading promotion checks.
        """
        registry = self._store.load_registry(self.agent_type)
        if not registry:
            return

        decay_cfg = self.config.get("decay", {})
        decay_ratio = decay_cfg.get("ic_decay_ratio", 0.5)
        decay_days_limit = decay_cfg.get("consecutive_days_limit", 14)

        n_retired = 0
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        modified = False
        label = self.agent_type.upper()

        for sig in registry:
            if sig.get("status") == "retired":
                continue
            rolling_ic = self._compute_rolling_ic(sig)
            if rolling_ic is not None:
                expected_ic = sig.get("expected_ic", 0)
                threshold = expected_ic * decay_ratio
                sig.setdefault("ic_history", [])
                sig["ic_history"].append({"date": today, "ic": rolling_ic})
                sig["ic_history"] = sig["ic_history"][-30:]
                sig["latest_ic"] = rolling_ic
                modified = True

                if rolling_ic < threshold:
                    days = sig.get("decay_days", 0) + 1
                    sig["decay_days"] = days
                    log.warning("  %s IC DECAY: %s IC=%.3f < %.3f, day %d/%d",
                                label, sig["name"][:40], rolling_ic, threshold,
                                days, decay_days_limit)
                    if days >= decay_days_limit:
                        sig["status"] = "retired"
                        sig["retired_reason"] = "ic_decay"
                        sig["retired_date"] = today
                        n_retired += 1
                        log.info("  %s AUTO-RETIRED: %s", label, sig["name"][:40])
                else:
                    if sig.get("decay_days", 0) > 0:
                        log.info("  %s IC recovered: %s IC=%.3f, resetting decay",
                                 label, sig["name"][:40], rolling_ic)
                    sig["decay_days"] = 0

        if modified or n_retired > 0:
            for sig in registry:
                hyp_id = sig.get("hypothesis_id")
                if hyp_id:
                    self._store.update_signal(self.agent_type, hyp_id, sig)

        if n_retired > 0:
            self.state.set("total_signals_registered",
                          max(0, self.state.get("total_signals_registered", 0) - n_retired))

        n_active = sum(1 for s in registry if s.get("status") != "retired")
        log.info("%s Monitor: %d active signals, %d retired this cycle",
                 label, n_active, n_retired)

        # Promotion check: paper → live
        self._check_promotions(registry)

    def _check_promotions(self, registry: list[dict]) -> None:
        """Check paper trading signals for promotion eligibility."""
        promotion = self.config.get("promotion", {})
        paper_sharpe_min = promotion.get("paper_sharpe_min", 1.5)
        paper_days = promotion.get("paper_days", 7)
        realized_ic_ratio_min = promotion.get("realized_ic_ratio_min", 0.8)
        max_drawdown_pct = promotion.get("max_drawdown_pct", 2.0)

        n_validated = n_paper = n_promotable = 0
        for sig in registry:
            status = sig.get("status", "validated")
            if status == "validated":
                n_validated += 1
            elif status == "paper":
                n_paper += 1
                paper_sharpe = sig.get("paper_sharpe")
                paper_days_actual = sig.get("paper_days_elapsed", 0)
                realized_ic = sig.get("realized_ic", 0)
                expected_ic = sig.get("expected_ic", 1)
                max_dd = sig.get("max_drawdown_pct", 100)

                if (paper_sharpe is not None
                        and paper_sharpe >= paper_sharpe_min
                        and paper_days_actual >= paper_days
                        and (realized_ic / expected_ic if expected_ic else 0) >= realized_ic_ratio_min
                        and max_dd <= max_drawdown_pct):
                    n_promotable += 1
                    log.info("  PROMOTION ELIGIBLE: %s (Sharpe=%.2f, %dd, IC ratio=%.2f)",
                             sig["name"][:50], paper_sharpe, paper_days_actual,
                             realized_ic / expected_ic if expected_ic else 0)

        if n_paper > 0:
            log.info("Promotion check: %d validated, %d paper, %d promotable",
                     n_validated, n_paper, n_promotable)

    # --- Optional hooks (subclass can override) ---------------------------

    def pre_execute(self, hypothesis: Hypothesis) -> None:
        """Inject adaptive IC threshold — raises the bar as registry grows."""
        adaptive_min_ic = self._compute_adaptive_ic()
        floor_ic = self.config.get("gates", {}).get("min_ic", 0.10)
        hypothesis.thresholds.setdefault("min_ic", floor_ic)
        if adaptive_min_ic > hypothesis.thresholds["min_ic"]:
            hypothesis.thresholds["min_ic"] = adaptive_min_ic

    def post_cycle(self, cycle_hypotheses: list) -> int:
        """Hypothesis chaining: spawn follow-ups from near-misses and strong results."""
        return self._chain_hypotheses(cycle_hypotheses)

    def _chain_hypotheses(self, cycle_hypotheses: list) -> int:
        """Spawn follow-up hypotheses from near-misses and strong results."""
        n_spawned = 0
        min_dIC = self.config.get("gates", {}).get("min_dIC", 0.05)

        for h in cycle_hypotheses:
            # Rule 1: symbol-specific variant when only 1 symbol fails
            if (h.status == "failed"
                    and h.failure_reason == "no_replication"
                    and h.results
                    and "symbol_replication" in h.results):
                sr = h.results["symbol_replication"]
                passed_syms = sr.get("passed", [])
                failed_syms = sr.get("failed", [])

                if len(failed_syms) == 1 and len(passed_syms) >= 2:
                    claim = f"{h.claim} [symbol-specific: {','.join(passed_syms)}]"
                    existing = {x.claim for x in self.queue._all}
                    if claim not in existing:
                        variant = Hypothesis.create(
                            claim=claim,
                            generator=h.generator,
                            test_protocol=h.test_protocol,
                            priority=h.priority * 0.7,
                            thresholds={
                                **h.thresholds,
                                "min_symbols": len(passed_syms),
                                "symbols_override": passed_syms,
                            },
                            parent_id=h.id,
                        )
                        self.queue.push(variant)
                        n_spawned += 1
                        log.info("  Chained symbol-specific: %s (%s)",
                                 claim[:60], ",".join(passed_syms))

            # Rule 2: strong dIC → flag for ensemble pairing
            if h.status in ("replicated", "passed") and h.results:
                dIC = self._extract_dIC_from_results(h)
                if dIC >= min_dIC * 2:
                    log.info("  Strong gate (dIC=%.3f): %s — ensemble candidate",
                             dIC, h.claim[:60])

        return n_spawned

    @staticmethod
    def _extract_dIC_from_results(h) -> float:
        """Extract dIC value from hypothesis gate results."""
        if not h.results or "gate_results" not in h.results:
            return 0.0
        for gr in h.results["gate_results"]:
            msg = gr.get("msg", "")
            if "dIC=" in msg:
                try:
                    return float(msg.split("dIC=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
        return 0.0

    def on_fdr_reject(self, hypothesis: Hypothesis) -> None:
        """Remove FDR-rejected signal from registry."""
        self._remove_from_registry(hypothesis.id)

    def build_manifest(self) -> dict:
        """Build the data manifest. Override for custom data sources."""
        from .manifest import build_manifest
        return build_manifest()

    # --- Shared monitoring methods ----------------------------------------

    def _remove_from_registry(self, hypothesis_id: str) -> None:
        """Remove a signal from the registry by its hypothesis_id."""
        self._store.remove_signal(self.agent_type, hypothesis_id)

    def _compute_adaptive_ic(self) -> float:
        """Compute adaptive IC threshold: max(floor, median(registry_ic) * 0.8)."""
        floor_ic = self.config.get("gates", {}).get("min_ic", 0.10)
        registry = self._store.load_registry(self.agent_type)
        ics = [s.get("expected_ic", 0) for s in registry
               if s.get("status") != "retired"]
        if not ics:
            return floor_ic
        ics.sort()
        median_ic = ics[len(ics) // 2]
        adaptive = max(floor_ic, median_ic * 0.8)
        log.info("%s Adaptive IC: median=%.3f -> threshold=%.3f (floor=%.2f)",
                 self.agent_type.capitalize(), median_ic, adaptive, floor_ic)
        return adaptive

    def _compute_rolling_ic(self, sig: dict) -> float | None:
        """Compute rolling IC for a registered signal on the latest data.

        Behaviour is configured by class-level _rolling_ic_* attributes:
        - _rolling_ic_bar_period: None for tick-level, "5min"/"1h" for resampled
        - _rolling_ic_horizon_default: default forward horizon in seconds
        - _rolling_ic_min_valid: minimum valid data points
        - _rolling_ic_feature_suffixes: suffixes to try for feature column
        - _rolling_ic_use_gated_signal: whether to apply regime gating
        """
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            return None

        from agent.runner import _load_feature_data

        data_root = self.root / "data" / "features"
        if not data_root.exists():
            return None
        dates = sorted(d.name for d in data_root.iterdir() if d.is_dir())
        if not dates:
            return None

        features = sig.get("features", [])
        if not features:
            return None
        feature = features[0]
        gate = sig.get("regime_gate")
        symbol = sig.get("symbols", ["BTC"])[0]
        horizon_s = sig.get("horizon_s", self._rolling_ic_horizon_default)

        for date in reversed(dates[-2:]):
            data_dir = f"data/features/{date}"
            df = _load_feature_data(data_dir, symbol)
            if df is None or len(df) < 500:
                continue

            # Tick-level path (microstructure)
            if self._rolling_ic_bar_period is None:
                if self._rolling_ic_use_gated_signal:
                    from agent.runner import _extract_gated_signal
                    signal_vals = _extract_gated_signal(df, feature, gate)
                    if signal_vals is None:
                        continue
                elif feature in df.columns:
                    signal_vals = df[feature].to_numpy(dtype=float)
                else:
                    continue

                if "raw_midprice" not in df.columns:
                    continue
                mid = df["raw_midprice"].to_numpy(dtype=float)
                horizon_rows = max(1, int(horizon_s / 0.1))  # 100ms ticks
                fwd = np.full_like(mid, np.nan)
                if len(mid) > horizon_rows:
                    fwd[:-horizon_rows] = (mid[horizon_rows:] - mid[:-horizon_rows]) / mid[:-horizon_rows]
            else:
                # Bar-resampled path (MF / macro)
                try:
                    from cluster_pipeline.preprocess import aggregate_bars
                    bars = aggregate_bars(df, self._rolling_ic_bar_period)
                except (ImportError, Exception):
                    continue

                # Find feature column with suffix matching
                feat_col = None
                for suffix in self._rolling_ic_feature_suffixes:
                    candidate = f"{feature}{suffix}" if suffix else feature
                    if candidate in bars.columns:
                        feat_col = candidate
                        break
                if feat_col is None:
                    continue

                signal_vals = bars[feat_col].to_numpy(dtype=float)

                # Forward returns at bar horizon
                bar_seconds = {"5min": 300, "1h": 3600}.get(self._rolling_ic_bar_period, 300)
                horizon_bars = max(1, int(horizon_s / bar_seconds))
                if "raw_midprice_mean" in bars.columns:
                    mid_col = "raw_midprice_mean"
                elif "raw_midprice_last" in bars.columns:
                    mid_col = "raw_midprice_last"
                else:
                    continue
                mid = bars[mid_col].to_numpy(dtype=float)
                fwd = np.full_like(mid, np.nan)
                if len(mid) > horizon_bars:
                    fwd[:-horizon_bars] = (mid[horizon_bars:] - mid[:-horizon_bars]) / mid[:-horizon_bars]

            valid = ~(np.isnan(signal_vals) | np.isnan(fwd))
            if valid.sum() < self._rolling_ic_min_valid:
                continue

            ic = pd.Series(signal_vals[valid]).corr(
                pd.Series(fwd[valid]), method="spearman"
            )
            return float(ic) if not np.isnan(ic) else None

        return None

    def print_report(self) -> None:
        """Print a full summary: registry + graveyard + generator stats."""
        s = self.state
        title = f"NAT {self.agent_type.replace('_', ' ').title()} Agent Report"

        print("=" * 60)
        print(f"  {title}")
        print("=" * 60)

        print(f"\nPhase:       {s.phase.value}")
        print(f"Cycles:      {s.get('cycle_count', 0)}")
        print(f"Tested:      {s.get('total_hypotheses_tested', 0)} hypotheses")
        print(f"Registered:  {s.get('total_signals_registered', 0)} signals")
        print(f"Queue depth: {self.queue.depth}")
        print(f"Graveyard:   {len(self.queue.graveyard)}")

        print(f"\n{'─' * 60}")
        print("REGISTRY")
        print(f"{'─' * 60}")
        registry = self._store.load_registry(self.agent_type)
        if registry:
            for sig in registry:
                print(f"  IC={sig['expected_ic']:.3f}  gate={sig.get('regime_gate', '-'):30s}  "
                      f"{sig['name'][:50]}")
        else:
            print("  (empty)")

        print(f"\n{'─' * 60}")
        print("GRAVEYARD BREAKDOWN")
        print(f"{'─' * 60}")
        reasons = {}
        for h in self.queue.graveyard:
            r = h.failure_reason or "unknown"
            reasons[r] = reasons.get(r, 0) + 1
        if reasons:
            for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"  {r:20s}  {count}")
        else:
            print("  (no failures)")

        print(f"\n{'─' * 60}")
        print("GENERATOR PERFORMANCE")
        print(f"{'─' * 60}")
        for name, gs in self.gen_stats.items():
            print(f"  {name:18s}  attempts={gs.attempts:3d}  "
                  f"successes={gs.successes:3d}  "
                  f"hit_rate={gs.hit_rate:.0%}  "
                  f"weight={gs.weight:.3f}")

        print()


# ---------------------------------------------------------------------------
# CLI helper — shared across all daemon subclasses
# ---------------------------------------------------------------------------

def cli_main(agent_class: type, description: str = "NAT Agent") -> None:
    """Standard CLI entry point for agent daemons."""
    import argparse
    from logging_config import setup_logging
    setup_logging("nat.agent")

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("action", choices=["start", "status", "once", "queue",
                                            "registry", "graveyard", "report"],
                        help="Action to perform")
    args = parser.parse_args()

    agent = agent_class()

    if args.action == "start":
        agent.run()
    elif args.action == "once":
        agent.run_cycle()
    elif args.action == "status":
        agent.print_status()
    elif args.action == "queue":
        for h in agent.queue.peek(20):
            print(f"  [{h.priority:6.3f}] {h.id}  {h.generator:12s}  {h.claim[:60]}")
    elif args.action == "registry":
        registry = agent._store.load_registry(agent.agent_type)
        if registry:
            for sig in registry:
                print(f"  IC={sig['expected_ic']:.3f}  {sig['status']:10s}  "
                      f"{','.join(sig['symbols']):12s}  {sig['name']}")
        else:
            print("  (empty)")
    elif args.action == "graveyard":
        for h in agent.queue.graveyard[-20:]:
            print(f"  {h.id}  {h.failure_reason or '??':20s}  {h.claim[:50]}")
    elif args.action == "report":
        agent.print_report()
