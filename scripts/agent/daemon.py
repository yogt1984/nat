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

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from agent.base import (  # noqa: E402
    ResearchAgent, BaseRunner, AgentPhase, AgentState,  # re-export for compat
    cli_main, load_agent_config,
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
    generator_module_prefix = "agent.generators"

    @property
    def root(self) -> Path:
        return ROOT

    # Tick-level rolling IC (no bar resampling)
    _rolling_ic_bar_period = None
    _rolling_ic_horizon_default = 5.0  # 5 seconds (50 rows at 100ms)
    _rolling_ic_min_valid = 200
    _rolling_ic_feature_suffixes = ("",)
    _rolling_ic_use_gated_signal = True

    def create_runner(self, hypothesis: Hypothesis, manifest: dict) -> BaseRunner:
        from agent.runner import MicrostructureRunner
        return MicrostructureRunner(hypothesis, manifest,
                                    store=self._store, agent=self.agent_type)

    def post_cycle(self, cycle_hypotheses: list) -> int:
        """Hypothesis chaining + cache stats logging."""
        n = super().post_cycle(cycle_hypotheses)
        from agent.runner import get_cache
        cache_stats = get_cache().stats
        log.info("Cache stats: %d hits, %d misses (%.0f%% hit rate, %d entries)",
                 cache_stats["hits"], cache_stats["misses"],
                 cache_stats["hit_rate"] * 100, cache_stats["entries"])
        return n

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


def load_config() -> dict:
    """Load agent config from TOML or return defaults."""
    return load_agent_config(
        ROOT / "config" / "agent.toml",
        "agent",
        MicrostructureAgent.BASE_CONFIG,
    )


def main():
    cli_main(MicrostructureAgent, "NAT Microstructure Agent")


if __name__ == "__main__":
    main()
