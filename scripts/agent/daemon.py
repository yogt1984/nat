#!/usr/bin/env python3
"""NAT Agent Daemon — Autonomous hypothesis-driven alpha discovery.

Main loop:
    1. UPDATE MANIFEST  — scan data/features/, write manifest.json
    2. GENERATE         — each generator emits hypotheses into queue
    3. EXECUTE          — pop highest-priority, run 3-gate protocol
    4. MONITOR          — check paper trading metrics for registered signals
    5. SLEEP            — wait until next cycle

Follows the pipeline_runner.py pattern: persistent state, SIGTERM handling,
tmux-based launch, cron watchdog.

Usage:
    python scripts/agent/daemon.py start     # run main loop
    python scripts/agent/daemon.py status    # print current state
    python scripts/agent/daemon.py once      # single cycle (for testing)
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from agent.hypothesis import Hypothesis, GeneratorStats
from agent.hypothesis_queue import HypothesisQueue
from agent.manifest import build_manifest, load_manifest
from agent.runner import ExperimentRunner

log = logging.getLogger("nat.agent")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "cycle_interval_s": 3600,          # 1 hour between cycles
    "max_experiments_per_cycle": 10,
    "max_cycle_runtime_s": 5400,       # 90 minutes
    "generators_enabled": [
        "systematic",
        "spectral",
        "regime",
        "cross_asset",
        "recycler",
    ],
}

STATE_PATH = ROOT / "data" / "agent" / "agent_state.json"
STATS_PATH = ROOT / "data" / "agent" / "generator_stats.json"


def load_config() -> dict:
    """Load agent config from TOML or return defaults."""
    config_path = ROOT / "config" / "agent.toml"
    if config_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(config_path, "rb") as f:
            return {**DEFAULT_CONFIG, **tomllib.load(f).get("agent", {})}
    return DEFAULT_CONFIG


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

    def __init__(self, path: Path = STATE_PATH):
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
# Generator stats persistence
# ---------------------------------------------------------------------------

def load_gen_stats() -> dict[str, GeneratorStats]:
    if STATS_PATH.exists():
        with open(STATS_PATH) as f:
            raw = json.load(f)
        return {k: GeneratorStats(**v) for k, v in raw.items()}
    return {g: GeneratorStats() for g in DEFAULT_CONFIG["generators_enabled"]}


def save_gen_stats(stats: dict[str, GeneratorStats]) -> None:
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump({k: {"attempts": v.attempts, "successes": v.successes}
                    for k, v in stats.items()}, f, indent=2)


# ---------------------------------------------------------------------------
# Generator dispatch
# ---------------------------------------------------------------------------

def run_generators(queue: HypothesisQueue, manifest: dict,
                   config: dict, gen_stats: dict[str, GeneratorStats]) -> int:
    """Run all enabled generators and push hypotheses into the queue."""
    total = 0
    for gen_name in config["generators_enabled"]:
        try:
            gen_func = _get_generator(gen_name)
            if gen_func is None:
                continue
            hypotheses = gen_func(manifest, queue, gen_stats.get(gen_name))
            for h in hypotheses:
                queue.push(h)
                total += 1
        except Exception as e:
            log.warning("Generator %s failed: %s", gen_name, e)
    log.info("Generated %d new hypotheses", total)
    return total


def _get_generator(name: str):
    """Lazy-import generator functions."""
    try:
        if name == "systematic":
            from agent.generators.systematic import generate
            return generate
        elif name == "spectral":
            from agent.generators.spectral import generate
            return generate
        elif name == "regime":
            from agent.generators.regime import generate
            return generate
        elif name == "cross_asset":
            from agent.generators.cross_asset import generate
            return generate
        elif name == "recycler":
            from agent.generators.recycler import generate
            return generate
    except ImportError as e:
        log.debug("Generator %s not yet implemented: %s", name, e)
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class AgentDaemon:
    """The autonomous research agent."""

    def __init__(self):
        self.config = load_config()
        self.state = AgentState()
        self.queue = HypothesisQueue()
        self.gen_stats = load_gen_stats()
        self._shutdown = False

        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        log.info("Received signal %d, will stop after current experiment", signum)
        self._shutdown = True

    @staticmethod
    def _remove_from_registry(hypothesis_id: str) -> None:
        """Remove a signal from the registry by its hypothesis_id."""
        from agent.runner import REGISTRY_PATH
        if not REGISTRY_PATH.exists():
            return
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        registry = [s for s in registry if s.get("hypothesis_id") != hypothesis_id]
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)

    def run_cycle(self) -> None:
        """Execute one complete research cycle."""
        cycle_start = time.monotonic()

        # 1. Update manifest
        self.state.transition(AgentPhase.MANIFEST, "scanning data")
        manifest = build_manifest()

        # 2. Generate hypotheses
        self.state.transition(AgentPhase.GENERATE, "generating hypotheses")
        run_generators(self.queue, manifest, self.config, self.gen_stats)

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
            runner = ExperimentRunner(hypothesis, manifest)
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
        from agent.runner import apply_fdr
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
                    # Remove from registry
                    self._remove_from_registry(h.id)
                    n_registered -= 1
                    self.state.set("total_signals_registered",
                                  max(0, self.state.get("total_signals_registered", 0) - 1))
            log.info("FDR control (q=%.2f): %d/%d rejected",
                     fdr_q, len(rejected_ids), len(cycle_hypotheses))

        save_gen_stats(self.gen_stats)
        self.state.set("current_hypothesis", None)

        # Log cache stats
        from agent.runner import get_cache
        cache_stats = get_cache().stats
        log.info("Cache stats: %d hits, %d misses (%.0f%% hit rate, %d entries)",
                 cache_stats["hits"], cache_stats["misses"],
                 cache_stats["hit_rate"] * 100, cache_stats["entries"])

        # 4. Monitor registered signals
        self.state.transition(AgentPhase.MONITOR, "checking registered signals")
        self.run_monitor()

        # 5. Update cycle counter
        self.state.set("cycle_count", self.state.get("cycle_count", 0) + 1)
        self.state.set("last_cycle_at", datetime.now(timezone.utc).isoformat())
        log.info("Cycle complete: %d experiments, queue depth=%d", n_run, self.queue.depth)

    def run_monitor(self) -> None:
        """Check registered signals for health/promotion.

        Reads the registry and promotion config. Logs warnings for signals
        that may have degraded. Flags promotion-eligible signals.
        """
        from agent.runner import REGISTRY_PATH
        if not REGISTRY_PATH.exists():
            return
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        if not registry:
            return

        promotion = self.config.get("promotion", {})
        paper_sharpe_min = promotion.get("paper_sharpe_min", 1.5)
        paper_days = promotion.get("paper_days", 7)
        realized_ic_ratio_min = promotion.get("realized_ic_ratio_min", 0.8)
        max_drawdown_pct = promotion.get("max_drawdown_pct", 2.0)

        n_validated = 0
        n_paper = 0
        n_promotable = 0
        for sig in registry:
            status = sig.get("status", "validated")
            if status == "validated":
                n_validated += 1
            elif status == "paper":
                n_paper += 1
                # Check promotion criteria (when paper trading data is available)
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

        log.info("Monitor: %d validated, %d paper, %d promotable",
                 n_validated, n_paper, n_promotable)

    def print_report(self) -> None:
        """Print a full summary: registry + queue + graveyard + generator stats."""
        from agent.runner import REGISTRY_PATH
        s = self.state

        print("=" * 60)
        print("  NAT Agent Report")
        print("=" * 60)

        # Status
        print(f"\nPhase:       {s.phase.value}")
        print(f"Cycles:      {s.get('cycle_count', 0)}")
        print(f"Tested:      {s.get('total_hypotheses_tested', 0)} hypotheses")
        print(f"Registered:  {s.get('total_signals_registered', 0)} signals")
        print(f"Queue depth: {self.queue.depth}")
        print(f"Graveyard:   {len(self.queue.graveyard)}")

        # Registry
        print(f"\n{'─' * 60}")
        print("REGISTRY")
        print(f"{'─' * 60}")
        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH) as f:
                registry = json.load(f)
            if registry:
                for sig in registry:
                    print(f"  IC={sig['expected_ic']:.3f}  gate={sig.get('regime_gate', '-'):30s}  "
                          f"{sig['name'][:50]}")
            else:
                print("  (empty)")
        else:
            print("  (no registry file)")

        # Graveyard breakdown
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

        # Generator stats
        print(f"\n{'─' * 60}")
        print("GENERATOR PERFORMANCE")
        print(f"{'─' * 60}")
        for name, gs in self.gen_stats.items():
            print(f"  {name:15s}  attempts={gs.attempts:3d}  "
                  f"successes={gs.successes:3d}  "
                  f"hit_rate={gs.hit_rate:.0%}  "
                  f"weight={gs.weight:.3f}")

        # Cache
        cache_dir = ROOT / "data" / "agent" / "cache"
        if cache_dir.exists():
            entries = len(list(cache_dir.glob("*.meta.json")))
            size_kb = sum(f.stat().st_size for f in cache_dir.glob("*.json")) / 1024
            print(f"\n{'─' * 60}")
            print("CACHE")
            print(f"{'─' * 60}")
            print(f"  Entries: {entries}  Size: {size_kb:.1f} KB")

        print()

    def run(self) -> None:
        """Run the agent loop until shutdown."""
        self.state.set("started_at", datetime.now(timezone.utc).isoformat())
        self.state.transition(AgentPhase.IDLE, "agent started")
        log.info("NAT Agent started (cycle_interval=%ds)", self.config["cycle_interval_s"])

        while not self._shutdown:
            try:
                self.run_cycle()
            except Exception as e:
                log.error("Cycle error: %s", e, exc_info=True)
                self.state.transition(AgentPhase.ERROR, str(e))

            if self._shutdown:
                break

            self.state.transition(AgentPhase.SLEEPING, "waiting for next cycle")
            # Sleep in small increments so we can respond to signals
            sleep_s = self.config["cycle_interval_s"]
            for _ in range(sleep_s):
                if self._shutdown:
                    break
                time.sleep(1)

        self.state.transition(AgentPhase.STOPPED, "graceful shutdown")
        log.info("NAT Agent stopped")

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="NAT Agent Daemon")
    parser.add_argument("action", choices=["start", "status", "once", "queue",
                                            "registry", "graveyard", "report"],
                        help="Action to perform")
    args = parser.parse_args()

    agent = AgentDaemon()

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
        reg_path = ROOT / "data" / "agent" / "registry.json"
        if reg_path.exists():
            with open(reg_path) as f:
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


if __name__ == "__main__":
    main()
