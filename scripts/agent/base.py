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
    """Abstract base for running a hypothesis through a gate protocol.

    Subclasses define their gate steps via ``steps()`` and signal
    registration via ``register_signal()``. The base class provides
    the sequential execution logic and common argument parsing.
    """

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

    @abstractmethod
    def steps(self) -> list:
        """Return ordered list of gate step methods. Each returns bool."""
        ...

    @abstractmethod
    def register_signal(self):
        """Register a validated signal after all gates pass."""
        ...

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
    default_generators: list[str] = []

    BASE_CONFIG = {
        "cycle_interval_s": 3600,
        "max_experiments_per_cycle": 10,
        "max_cycle_runtime_s": 5400,
    }

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
    def state_path(self) -> Path:
        return self.root / "data" / "agent" / "agent_state.json"

    @property
    def queue_path(self) -> Path:
        return self.root / "data" / "agent" / "hypotheses.json"

    @property
    def stats_path(self) -> Path:
        return self.root / "data" / "agent" / "generator_stats.json"

    # --- Config -----------------------------------------------------------

    def load_config(self) -> dict:
        """Load configuration. Override for custom config sources."""
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

    @abstractmethod
    def run_monitor(self) -> None:
        """Check registered signals for health, decay, promotion."""
        ...

    # --- Optional hooks (subclass can override) ---------------------------

    def pre_execute(self, hypothesis: Hypothesis) -> None:
        """Called before each hypothesis execution. Default: no-op."""
        pass

    def post_cycle(self, cycle_hypotheses: list) -> int:
        """Called after execution phase. Return number of spawned follow-ups."""
        return 0

    def on_fdr_reject(self, hypothesis: Hypothesis) -> None:
        """Called when a hypothesis is FDR-rejected. Default: no-op."""
        pass

    def build_manifest(self) -> dict:
        """Build the data manifest. Override for custom data sources."""
        from .manifest import build_manifest
        return build_manifest()
