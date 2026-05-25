"""Base classes for autonomous research agents.

Provides:
- ResearchAgent ABC — cycle loop, state machine, generator dispatch, FDR
- BaseRunner ABC — sequential gate protocol execution
- AgentPhase, AgentState — shared infrastructure
"""

from __future__ import annotations

import json
import logging
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
    """Persistent agent state — survives restarts."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {
            "phase": AgentPhase.IDLE.value,
            "cycle_count": 0,
            "total_hypotheses_tested": 0,
            "total_signals_registered": 0,
            "current_hypothesis": None,
            "started_at": None,
            "last_cycle_at": None,
            "history": [],
        }

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def transition(self, phase: AgentPhase, msg: str = "") -> None:
        old = self._data["phase"]
        self._data["phase"] = phase.value
        self._data["history"].append({
            "from": old, "to": phase.value,
            "at": datetime.now(timezone.utc).isoformat(),
            "msg": msg,
        })
        # Keep history bounded
        if len(self._data["history"]) > 500:
            self._data["history"] = self._data["history"][-200:]
        self.save()

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

    def __init__(self, hypothesis: Hypothesis, manifest: dict):
        self.h = hypothesis
        self.manifest = manifest
        self.gate_results: list[dict] = []

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
            log.info("  TEMPORAL REPLICATION PASSED: %d/%d dates", n_pass, n_tested)
            return True
        else:
            self.h.fail("no_replication")
            log.info("  TEMPORAL REPLICATION FAILED: %d/%d dates", n_pass, n_tested)
            return False

    def run_replication_symbol(self) -> bool:
        """Re-run on other symbols."""
        from .runner import run_nat_cached

        primary_sym = self._extract_symbol(self.h.test_protocol[0])
        other_symbols = [s for s in ["BTC", "ETH", "SOL"] if s != primary_sym]

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
            log.info("  SYMBOL REPLICATION PASSED: %d/%d", n_pass, len(other_symbols))
            return True
        else:
            self.h.fail("no_replication")
            log.info("  SYMBOL REPLICATION FAILED: %d/%d", n_pass, len(other_symbols))
            return False

    def run_correlation_check(self) -> bool:
        """Check if signal is redundant with existing registry entries."""
        from .runner import check_correlation_gate

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
                self.h.fail("redundant")
                return False
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
        registry = self._load_registry()
        registry.append(signal.to_dict())
        path = self.REGISTRY_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(registry, f, indent=2)
        log.info("  REGISTERED: %s (IC=%.3f, horizon=%.0fs)",
                 signal.name, signal.expected_ic, horizon_s)
        return signal

    # --- Helpers ----------------------------------------------------------

    def _check_gates(self, report: dict) -> tuple[bool, str]:
        """Run IC and dIC gate checks against hypothesis thresholds."""
        from .runner import check_ic_gate, check_dIC_gate

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

    @classmethod
    def _load_registry(cls) -> list[dict]:
        """Load the signal registry from REGISTRY_PATH."""
        path = cls.REGISTRY_PATH
        if path and path.exists():
            with open(path) as f:
                return json.load(f)
        return []

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

    def __init__(self, config: dict | None = None):
        self.config = config or self.load_config()
        self.config.setdefault("generators_enabled", list(self.default_generators))
        self.state = AgentState(self.state_path)
        self.queue = HypothesisQueue(path=self.queue_path)
        self.gen_stats = self._load_gen_stats()
        self._shutdown = False
        signal_mod.signal(signal_mod.SIGTERM, self._handle_signal)
        signal_mod.signal(signal_mod.SIGINT, self._handle_signal)

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

    # --- Config -----------------------------------------------------------

    def load_config(self) -> dict:
        """Load config from TOML [config_section] merged over BASE_CONFIG."""
        config_path = self.root / "config" / "agent.toml"
        if config_path.exists():
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            with open(config_path, "rb") as f:
                return {**self.BASE_CONFIG, **tomllib.load(f).get(self.config_section, {})}
        return dict(self.BASE_CONFIG)

    # --- Generator stats --------------------------------------------------

    def _load_gen_stats(self) -> dict[str, GeneratorStats]:
        if self.stats_path.exists():
            with open(self.stats_path) as f:
                raw = json.load(f)
            return {k: GeneratorStats(**v) for k, v in raw.items()}
        return {g: GeneratorStats()
                for g in self.config.get("generators_enabled", [])}

    def _save_gen_stats(self) -> None:
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, "w") as f:
            json.dump({k: {"attempts": v.attempts, "successes": v.successes}
                        for k, v in self.gen_stats.items()}, f, indent=2)

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
            self.pre_execute(hypothesis)
            runner = self.create_runner(hypothesis, manifest)
            success = runner.run_full()

            # Update stats
            gen_name = hypothesis.generator
            if gen_name in self.gen_stats:
                self.gen_stats[gen_name].record(success)

            self.queue.update(hypothesis)
            cycle_hypotheses.append(hypothesis)
            n_run += 1
            self.state.set("total_hypotheses_tested",
                          self.state.get("total_hypotheses_tested", 0) + 1)
            if success:
                n_registered += 1
                self.state.set("total_signals_registered",
                              self.state.get("total_signals_registered", 0) + 1)

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

        # 4. Monitor registered signals
        self.state.transition(AgentPhase.MONITOR, "checking registered signals")
        self.run_monitor()

        # 5. Update cycle counter
        self.state.set("cycle_count", self.state.get("cycle_count", 0) + 1)
        self.state.set("last_cycle_at", datetime.now(timezone.utc).isoformat())
        log.info("Cycle complete: %d experiments, queue depth=%d",
                 n_run, self.queue.depth)

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

    # --- Abstract hooks (subclass must implement) -------------------------

    @abstractmethod
    def get_generator(self, name: str):
        """Return the generate() function for a named generator, or None."""
        ...

    @abstractmethod
    def create_runner(self, hypothesis: Hypothesis,
                      manifest: dict) -> BaseRunner:
        """Create an experiment runner for the given hypothesis."""
        ...

    def run_monitor(self) -> None:
        """Check registered signals for IC decay. Auto-retire after N days.

        Subclasses can override to add promotion logic (e.g., microstructure).
        """
        if not self.registry_path.exists():
            return
        with open(self.registry_path) as f:
            registry = json.load(f)
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
            with open(self.registry_path, "w") as f:
                json.dump(registry, f, indent=2)

        if n_retired > 0:
            self.state.set("total_signals_registered",
                          max(0, self.state.get("total_signals_registered", 0) - n_retired))

        n_active = sum(1 for s in registry if s.get("status") != "retired")
        log.info("%s Monitor: %d active signals, %d retired this cycle",
                 label, n_active, n_retired)

    # --- Optional hooks (subclass can override) ---------------------------

    def pre_execute(self, hypothesis: Hypothesis) -> None:
        """Inject adaptive IC threshold — raises the bar as registry grows."""
        adaptive_min_ic = self._compute_adaptive_ic()
        floor_ic = self.config.get("gates", {}).get("min_ic", 0.10)
        hypothesis.thresholds.setdefault("min_ic", floor_ic)
        if adaptive_min_ic > hypothesis.thresholds["min_ic"]:
            hypothesis.thresholds["min_ic"] = adaptive_min_ic

    def post_cycle(self, cycle_hypotheses: list) -> int:
        """Called after execution phase. Return number of spawned follow-ups."""
        return 0

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
        if not self.registry_path.exists():
            return
        with open(self.registry_path) as f:
            registry = json.load(f)
        registry = [s for s in registry if s.get("hypothesis_id") != hypothesis_id]
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def _compute_adaptive_ic(self) -> float:
        """Compute adaptive IC threshold: max(floor, median(registry_ic) * 0.8)."""
        floor_ic = self.config.get("gates", {}).get("min_ic", 0.10)
        if not self.registry_path.exists():
            return floor_ic
        with open(self.registry_path) as f:
            registry = json.load(f)
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
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                registry = json.load(f)
            if registry:
                for sig in registry:
                    print(f"  IC={sig['expected_ic']:.3f}  gate={sig.get('regime_gate', '-'):30s}  "
                          f"{sig['name'][:50]}")
            else:
                print("  (empty)")
        else:
            print("  (no registry file)")

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

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
        if agent.registry_path.exists():
            with open(agent.registry_path) as f:
                for sig in json.load(f):
                    print(f"  IC={sig['expected_ic']:.3f}  {sig['status']:10s}  "
                          f"{','.join(sig['symbols']):12s}  {sig['name']}")
        else:
            print("  (empty)")
    elif args.action == "graveyard":
        for h in agent.queue.graveyard[-20:]:
            print(f"  {h.id}  {h.failure_reason or '??':20s}  {h.claim[:50]}")
    elif args.action == "report":
        agent.print_report()
