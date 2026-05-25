#!/usr/bin/env python3
"""NAT Macro Agent — 1h-24h alpha discovery.

Discovers signals from 1h-resampled bars using funding rate mean-reversion,
OI divergence, and whale flow momentum. 4-gate replication protocol.

Usage:
    python scripts/agent/macro_daemon.py start
    python scripts/agent/macro_daemon.py status
    python scripts/agent/macro_daemon.py once
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from agent.base import ResearchAgent, BaseRunner, AgentPhase, cli_main  # noqa: E402
from agent.hypothesis import Hypothesis, GeneratorStats  # noqa: E402

log = logging.getLogger("nat.agent_macro")


class MacroAgent(ResearchAgent):
    """Macro research agent for 1h-24h alpha discovery."""

    agent_type = "macro"
    config_section = "agent_macro"
    default_generators = ["funding_meanrev", "oi_divergence", "whale_momentum"]

    BASE_CONFIG = {
        "cycle_interval_s": 14400,
        "max_experiments_per_cycle": 6,
        "max_cycle_runtime_s": 7200,
    }

    @property
    def root(self) -> Path:
        return ROOT

    # 1h bar rolling IC
    _rolling_ic_bar_period = "1h"
    _rolling_ic_horizon_default = 3600.0  # 1 hour
    _rolling_ic_min_valid = 20
    _rolling_ic_feature_suffixes = ("_last", "_mean", "_sum", "")
    _rolling_ic_use_gated_signal = False

    @property
    def agent_dir(self) -> str:
        return "agent_macro"

    def get_generator(self, name: str):
        """Lazy-import macro generator functions."""
        try:
            mod = __import__(f"agent.generators.macro.{name}", fromlist=["generate"])
            return mod.generate
        except (ImportError, AttributeError) as e:
            log.debug("Macro generator %s not yet implemented: %s", name, e)
        return None

    def create_runner(self, hypothesis: Hypothesis, manifest: dict) -> BaseRunner:
        from agent.macro_runner import MacroRunner
        return MacroRunner(hypothesis, manifest,
                           store=self._store, agent=self.agent_type)


# Backward compatibility — module-level path constants
MACRO_STATE_PATH = ROOT / "data" / "agent_macro" / "agent_state.json"
MACRO_STATS_PATH = ROOT / "data" / "agent_macro" / "generator_stats.json"
MACRO_REGISTRY_PATH = ROOT / "data" / "agent_macro" / "registry.json"


def load_config() -> dict:
    """Load macro agent config from TOML or return defaults.

    Uses [defaults] → [agent_macro] inheritance via load_agent_config.
    """
    from agent.base import load_agent_config
    return load_agent_config(
        ROOT / "config" / "agent.toml",
        "agent_macro",
        MacroAgent.BASE_CONFIG,
    )


def main():
    cli_main(MacroAgent, "NAT Macro Agent")


if __name__ == "__main__":
    main()
