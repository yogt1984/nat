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
        "ensemble",
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
        elif name == "ensemble":
            from agent.generators.ensemble import generate
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
        self.queue = HypothesisQueue(path=ROOT / "data" / "agent" / "hypotheses.json")
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

        # 2b. Adaptive IC threshold — raise the bar as registry quality grows
        adaptive_min_ic = self._compute_adaptive_ic()

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
            # Inject adaptive IC threshold (raises bar as registry grows)
            hypothesis.thresholds.setdefault("min_ic", 0.10)
            if adaptive_min_ic > hypothesis.thresholds["min_ic"]:
                hypothesis.thresholds["min_ic"] = adaptive_min_ic
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

        # 3c. Hypothesis chaining — spawn follow-up hypotheses from near-misses
        n_chained = self._chain_hypotheses(cycle_hypotheses)
        if n_chained:
            log.info("Chaining: spawned %d follow-up hypotheses", n_chained)

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

    def _chain_hypotheses(self, cycle_hypotheses: list) -> int:
        """Spawn follow-up hypotheses from near-misses and strong results.

        Rule 1 — Symbol-specific variant: if a hypothesis passed discovery,
        cost, and temporal replication but failed symbol replication on
        exactly 1 symbol, spawn a symbol-specific variant (primary + passing
        symbol only) with lower priority.

        Rule 2 — Ensemble promotion: if a hypothesis passed with dIC >> min_dIC
        (at least 2x), it's a strong single gate. Trigger the ensemble generator
        to pair it with other strong gates. (Ensemble generator picks these up
        automatically on the next cycle; here we just log the signal.)

        Returns the number of chained hypotheses spawned.
        """
        n_spawned = 0
        min_dIC = self.config.get("gates", {}).get("min_dIC", 0.05)

        for h in cycle_hypotheses:
            # Rule 1: symbol-specific variant
            if (h.status == "failed"
                    and h.failure_reason == "no_replication"
                    and h.results
                    and "symbol_replication" in h.results):
                sr = h.results["symbol_replication"]
                passed_syms = sr.get("passed", [])
                failed_syms = sr.get("failed", [])

                # Failed on exactly 1 symbol → spawn variant for passing symbols
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
                    # Ensemble generator will pick this up next cycle

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

    def _compute_adaptive_ic(self) -> float:
        """Compute adaptive IC threshold: max(0.10, median(registry_ic) * 0.8).

        As the registry grows with high-quality signals, the bar rises so
        marginal signals are rejected. Floor is always the config min_ic.
        """
        from agent.runner import REGISTRY_PATH
        floor_ic = self.config.get("gates", {}).get("min_ic", 0.10)
        if not REGISTRY_PATH.exists():
            log.info("Adaptive IC: no registry, using floor %.2f", floor_ic)
            return floor_ic
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        ics = [s.get("expected_ic", 0) for s in registry
               if s.get("status") != "retired"]
        if not ics:
            log.info("Adaptive IC: empty registry, using floor %.2f", floor_ic)
            return floor_ic
        ics.sort()
        median_ic = ics[len(ics) // 2]
        adaptive = max(floor_ic, median_ic * 0.8)
        log.info("Adaptive IC: median=%.3f -> threshold=%.3f (floor=%.2f, %d signals)",
                 median_ic, adaptive, floor_ic, len(ics))
        return adaptive

    def run_monitor(self) -> None:
        """Check registered signals for IC decay and promotion.

        For each registered signal:
        1. Compute IC on the latest data window
        2. If IC < 50% of discovery IC, increment decay counter
        3. Auto-retire after 14 consecutive decay days
        4. Check paper trading signals for promotion eligibility
        """
        from agent.runner import REGISTRY_PATH
        if not REGISTRY_PATH.exists():
            return
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        if not registry:
            return

        decay_cfg = self.config.get("decay", {})
        decay_ratio = decay_cfg.get("ic_decay_ratio", 0.5)
        decay_days_limit = decay_cfg.get("consecutive_days_limit", 14)

        promotion = self.config.get("promotion", {})
        paper_sharpe_min = promotion.get("paper_sharpe_min", 1.5)
        paper_days = promotion.get("paper_days", 7)
        realized_ic_ratio_min = promotion.get("realized_ic_ratio_min", 0.8)
        max_drawdown_pct = promotion.get("max_drawdown_pct", 2.0)

        # IC decay monitoring
        n_retired = 0
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        modified = False
        for sig in registry:
            if sig.get("status") == "retired":
                continue
            rolling_ic = self._compute_rolling_ic(sig)
            if rolling_ic is not None:
                expected_ic = sig.get("expected_ic", 0)
                threshold = expected_ic * decay_ratio
                sig.setdefault("ic_history", [])
                sig["ic_history"].append({"date": today, "ic": rolling_ic})
                # Keep last 30 entries
                sig["ic_history"] = sig["ic_history"][-30:]
                sig["latest_ic"] = rolling_ic
                modified = True

                if rolling_ic < threshold:
                    days = sig.get("decay_days", 0) + 1
                    sig["decay_days"] = days
                    log.warning("  IC DECAY: %s IC=%.3f < %.3f (%.0f%% of %.3f), day %d/%d",
                                sig["name"][:40], rolling_ic, threshold,
                                decay_ratio * 100, expected_ic, days, decay_days_limit)
                    if days >= decay_days_limit:
                        sig["status"] = "retired"
                        sig["retired_reason"] = "ic_decay"
                        sig["retired_date"] = today
                        n_retired += 1
                        log.info("  AUTO-RETIRED: %s (IC decayed for %d days)",
                                 sig["name"][:40], days)
                else:
                    # Reset decay counter on recovery
                    if sig.get("decay_days", 0) > 0:
                        log.info("  IC recovered: %s IC=%.3f >= %.3f, resetting decay counter",
                                 sig["name"][:40], rolling_ic, threshold)
                    sig["decay_days"] = 0

        # Promotion check
        n_validated = 0
        n_paper = 0
        n_promotable = 0
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

        # Save registry if modified
        if modified or n_retired > 0:
            with open(REGISTRY_PATH, "w") as f:
                json.dump(registry, f, indent=2)

        if n_retired > 0:
            self.state.set("total_signals_registered",
                          max(0, self.state.get("total_signals_registered", 0) - n_retired))

        log.info("Monitor: %d validated, %d paper, %d promotable, %d retired this cycle",
                 n_validated, n_paper, n_promotable, n_retired)

    def _compute_rolling_ic(self, sig: dict) -> float | None:
        """Compute rolling IC for a registered signal on the latest available data.

        Loads the most recent date's Parquet data, applies the regime gate,
        and computes Spearman correlation between the signal feature and
        5-second forward returns.

        Returns None if data is unavailable or insufficient.
        """
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            return None

        from agent.runner import _load_feature_data, _extract_gated_signal

        # Find the latest data directory
        data_root = ROOT / "data" / "features"
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

        # Try latest date, fall back to second latest
        for date in reversed(dates[-2:]):
            data_dir = f"data/features/{date}"
            df = _load_feature_data(data_dir, symbol)
            if df is None or len(df) < 500:
                continue

            signal_vals = _extract_gated_signal(df, feature, gate)
            if signal_vals is None:
                continue

            # Compute 5s forward returns (50 rows at 100ms)
            if "raw_midprice" not in df.columns:
                continue
            mid = df["raw_midprice"].to_numpy(dtype=float)
            fwd = np.full_like(mid, np.nan)
            fwd[:-50] = (mid[50:] - mid[:-50]) / mid[:-50]

            # Spearman on valid overlap
            valid = ~(np.isnan(signal_vals) | np.isnan(fwd))
            if valid.sum() < 200:
                continue

            ic = pd.Series(signal_vals[valid]).corr(
                pd.Series(fwd[valid]), method="spearman"
            )
            return float(ic) if not np.isnan(ic) else None

        return None

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
