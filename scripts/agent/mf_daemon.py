#!/usr/bin/env python3
"""NAT Medium-Frequency Agent — 1min-1h alpha discovery.

Discovers signals from 5min-resampled bars using momentum, vol breakout,
and flow clustering generators. 4-gate replication protocol.

Usage:
    python scripts/agent/mf_daemon.py start
    python scripts/agent/mf_daemon.py status
    python scripts/agent/mf_daemon.py once
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from agent.base import ResearchAgent, BaseRunner, AgentPhase, cli_main  # noqa: E402
from agent.hypothesis import Hypothesis, GeneratorStats  # noqa: E402

log = logging.getLogger("nat.agent_mf")


class MediumFrequencyAgent(ResearchAgent):
    """Medium-frequency research agent for 1min-1h alpha discovery."""

    agent_type = "medium_freq"
    config_section = "agent_mf"
    default_generators = ["momentum", "vol_breakout", "flow_cluster"]

    BASE_CONFIG = {
        "cycle_interval_s": 7200,
        "max_experiments_per_cycle": 8,
        "max_cycle_runtime_s": 5400,
    }

    @property
    def root(self) -> Path:
        return ROOT

    # 5min bar rolling IC
    _rolling_ic_bar_period = "5min"
    _rolling_ic_horizon_default = 300.0  # 5 minutes
    _rolling_ic_min_valid = 50
    _rolling_ic_feature_suffixes = ("_last", "_mean", "")
    _rolling_ic_use_gated_signal = False

    @property
    def agent_dir(self) -> str:
        return "agent_mf"

    def get_generator(self, name: str):
        """Lazy-import medium-frequency generator functions."""
        try:
            mod = __import__(f"agent.generators.medium_freq.{name}", fromlist=["generate"])
            return mod.generate
        except (ImportError, AttributeError) as e:
            log.debug("MF generator %s not yet implemented: %s", name, e)
        return None

    def create_runner(self, hypothesis: Hypothesis, manifest: dict) -> BaseRunner:
        from agent.mf_runner import MediumFrequencyRunner
        return MediumFrequencyRunner(hypothesis, manifest)


# Backward compatibility — module-level path constants
MF_STATE_PATH = ROOT / "data" / "agent_mf" / "agent_state.json"
MF_STATS_PATH = ROOT / "data" / "agent_mf" / "generator_stats.json"
MF_REGISTRY_PATH = ROOT / "data" / "agent_mf" / "registry.json"


def load_config() -> dict:
    """Load MF agent config from TOML or return defaults.

    Uses [defaults] → [agent_mf] inheritance via load_agent_config.
    """
    from agent.base import load_agent_config
    return load_agent_config(
        ROOT / "config" / "agent.toml",
        "agent_mf",
        MediumFrequencyAgent.BASE_CONFIG,
    )


def main():
    cli_main(MediumFrequencyAgent, "NAT Medium-Frequency Agent")


if __name__ == "__main__":
    main()
