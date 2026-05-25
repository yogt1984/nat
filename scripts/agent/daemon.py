#!/usr/bin/env python3
"""NAT Microstructure Agent — Autonomous hypothesis-driven alpha discovery.

Discovers scalping signals (5s horizon) from Hyperliquid perpetual futures
using 6 generators and a 5-gate replication protocol.

Usage:
    python scripts/agent/daemon.py start     # run main loop
    python scripts/agent/daemon.py status    # print current state
    python scripts/agent/daemon.py once      # single cycle (for testing)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from agent.base import (  # noqa: E402
    ResearchAgent, BaseRunner, AgentPhase, AgentState,  # re-export for compat
    cli_main,
)
from agent.hypothesis import Hypothesis, GeneratorStats  # noqa: E402
from agent.hypothesis_queue import HypothesisQueue  # noqa: E402

log = logging.getLogger("nat.agent")


class MicrostructureAgent(ResearchAgent):
    """Microstructure research agent for HFT alpha discovery.

    Discovers scalping signals (5s horizon) from Hyperliquid perpetual
    futures data. Uses 6 generators and a 5-gate replication protocol.
    """

    agent_type = "microstructure"
    config_section = "agent"
    default_generators = [
        "systematic", "spectral", "regime",
        "cross_asset", "recycler", "ensemble",
    ]

    @property
    def root(self) -> Path:
        return ROOT

    # Tick-level rolling IC (no bar resampling)
    _rolling_ic_bar_period = None
    _rolling_ic_horizon_default = 5.0  # 5 seconds (50 rows at 100ms)
    _rolling_ic_min_valid = 200
    _rolling_ic_feature_suffixes = ("",)
    _rolling_ic_use_gated_signal = True

    def get_generator(self, name: str):
        """Lazy-import microstructure generator functions."""
        try:
            mod = __import__(f"agent.generators.{name}", fromlist=["generate"])
            return mod.generate
        except (ImportError, AttributeError) as e:
            log.debug("Generator %s not yet implemented: %s", name, e)
        return None

    def create_runner(self, hypothesis: Hypothesis, manifest: dict) -> BaseRunner:
        from agent.runner import MicrostructureRunner
        return MicrostructureRunner(hypothesis, manifest)

    def run_monitor(self) -> None:
        """IC decay monitoring + microstructure-specific promotion check."""
        super().run_monitor()
        self._check_promotions()

    def _check_promotions(self) -> None:
        """Check paper trading signals for promotion eligibility."""
        if not self.registry_path.exists():
            return
        with open(self.registry_path) as f:
            registry = json.load(f)

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

        log.info("Promotion check: %d validated, %d paper, %d promotable",
                 n_validated, n_paper, n_promotable)

    def post_cycle(self, cycle_hypotheses: list) -> int:
        """Hypothesis chaining + cache stats logging."""
        n = self._chain_hypotheses(cycle_hypotheses)
        from agent.runner import get_cache
        cache_stats = get_cache().stats
        log.info("Cache stats: %d hits, %d misses (%.0f%% hit rate, %d entries)",
                 cache_stats["hits"], cache_stats["misses"],
                 cache_stats["hit_rate"] * 100, cache_stats["entries"])
        return n

    def _chain_hypotheses(self, cycle_hypotheses: list) -> int:
        """Spawn follow-up hypotheses from near-misses and strong results."""
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

    def print_report(self) -> None:
        """Report with cache stats appended."""
        super().print_report()
        cache_dir = self.root / "data" / "agent" / "cache"
        if cache_dir.exists():
            entries = len(list(cache_dir.glob("*.meta.json")))
            size_kb = sum(f.stat().st_size for f in cache_dir.glob("*.json")) / 1024
            print(f"{'─' * 60}")
            print("CACHE")
            print(f"{'─' * 60}")
            print(f"  Entries: {entries}  Size: {size_kb:.1f} KB")
            print()


# Backward compatibility aliases and module-level constants
AgentDaemon = MicrostructureAgent
STATE_PATH = ROOT / "data" / "agent" / "agent_state.json"
STATS_PATH = ROOT / "data" / "agent" / "generator_stats.json"


# Backward compatibility — module-level functions
def load_config() -> dict:
    """Load agent config from TOML or return defaults."""
    config_path = ROOT / "config" / "agent.toml"
    DEFAULT_CONFIG = {
        "cycle_interval_s": 3600,
        "max_experiments_per_cycle": 10,
        "max_cycle_runtime_s": 5400,
        "generators_enabled": [
            "systematic", "spectral", "regime",
            "cross_asset", "recycler", "ensemble",
        ],
    }
    if config_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(config_path, "rb") as f:
            return {**DEFAULT_CONFIG, **tomllib.load(f).get("agent", {})}
    return DEFAULT_CONFIG


def load_gen_stats() -> dict[str, GeneratorStats]:
    STATS_PATH = ROOT / "data" / "agent" / "generator_stats.json"
    if STATS_PATH.exists():
        with open(STATS_PATH) as f:
            raw = json.load(f)
        return {k: GeneratorStats(**v) for k, v in raw.items()}
    return {g: GeneratorStats() for g in ["systematic", "spectral", "regime",
                                            "cross_asset", "recycler", "ensemble"]}


def save_gen_stats(stats: dict[str, GeneratorStats]) -> None:
    STATS_PATH = ROOT / "data" / "agent" / "generator_stats.json"
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump({k: {"attempts": v.attempts, "successes": v.successes}
                    for k, v in stats.items()}, f, indent=2)


def main():
    cli_main(MicrostructureAgent, "NAT Microstructure Agent")


if __name__ == "__main__":
    main()
