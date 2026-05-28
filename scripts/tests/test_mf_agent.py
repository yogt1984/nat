"""Tests for the Medium-Frequency Agent (4.2).

Tests cover:
- Generator contracts: momentum, vol_breakout, flow_cluster
- MediumFrequencyRunner: 4-gate protocol, registration, helpers
- MediumFrequencyAgent: ABC compliance, state isolation, lifecycle
- Backward compatibility: microstructure agent unaffected
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


import pytest
from agent.base import ResearchAgent, BaseRunner
from agent.hypothesis import Hypothesis, GeneratorStats
from agent.hypothesis_queue import HypothesisQueue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manifest():
    return {
        "dates": {"2026-05-12": {}, "2026-05-13": {}, "2026-05-14": {}},
        "symbols": {"BTC": {"hours": 8}, "ETH": {"hours": 8}, "SOL": {"hours": 8}},
        "total_hours_per_symbol": 24,
    }


@pytest.fixture
def queue(tmp_path):
    return HypothesisQueue(path=tmp_path / "hypotheses.json")


@pytest.fixture
def mf_agent(tmp_path):
    """Create a MediumFrequencyAgent with tmp_path-isolated SQLite store."""
    from agent.mf_daemon import MediumFrequencyAgent
    from data.state import StateStore

    store = StateStore(tmp_path / "nat.db")
    agent = MediumFrequencyAgent(store=store)
    return agent


# ===========================================================================
# Generator tests
# ===========================================================================

class TestMomentumGenerator:
    def test_generates_valid_hypotheses(self, manifest, queue):
        from agent.generators.medium_freq.momentum import generate
        hyps = generate(manifest, queue)
        assert len(hyps) > 0
        for h in hyps:
            assert h.generator == "momentum"
            assert h.id.startswith("HYP-MOM-")

    def test_deduplicates_against_queue(self, manifest, queue):
        from agent.generators.medium_freq.momentum import generate
        hyps1 = generate(manifest, queue)
        for h in hyps1:
            queue.push(h)
        hyps2 = generate(manifest, queue)
        assert len(hyps2) == 0

    def test_empty_manifest_returns_empty(self, queue):
        from agent.generators.medium_freq.momentum import generate
        assert generate({}, queue) == []
        assert generate({"dates": {}}, queue) == []

    def test_includes_timeframe_in_protocol(self, manifest, queue):
        from agent.generators.medium_freq.momentum import generate
        hyps = generate(manifest, queue)
        for h in hyps:
            assert any("--timeframe 5min" in cmd for cmd in h.test_protocol)

    def test_priority_boost_for_r2(self, manifest, queue):
        from agent.generators.medium_freq.momentum import generate
        hyps = generate(manifest, queue)
        r2_hyps = [h for h in hyps if "r2" in h.claim]
        non_r2 = [h for h in hyps if "r2" not in h.claim and "hurst" not in h.claim]
        if r2_hyps and non_r2:
            # R2 hypotheses should have higher base priority
            assert max(h.priority for h in r2_hyps) > min(h.priority for h in non_r2)


class TestVolBreakoutGenerator:
    def test_generates_two_classes(self, manifest, queue):
        from agent.generators.medium_freq.vol_breakout import generate
        hyps = generate(manifest, queue)
        assert len(hyps) > 0
        claims = [h.claim for h in hyps]
        assert any("continuation" in c for c in claims)
        assert any("reversion" in c for c in claims)

    def test_uses_correct_prefix(self, manifest, queue):
        from agent.generators.medium_freq.vol_breakout import generate
        hyps = generate(manifest, queue)
        for h in hyps:
            assert h.id.startswith("HYP-VBK-")
            assert h.generator == "vol_breakout"

    def test_empty_manifest_returns_empty(self, queue):
        from agent.generators.medium_freq.vol_breakout import generate
        assert generate({}, queue) == []


class TestFlowClusterGenerator:
    def test_generates_bar_aggregated(self, manifest, queue):
        from agent.generators.medium_freq.flow_cluster import generate
        hyps = generate(manifest, queue)
        assert len(hyps) > 0
        # Claims should mention bar-flow
        assert all("bar-flow" in h.claim for h in hyps)

    def test_uses_correct_prefix(self, manifest, queue):
        from agent.generators.medium_freq.flow_cluster import generate
        hyps = generate(manifest, queue)
        for h in hyps:
            assert h.id.startswith("HYP-FCL-")
            assert h.generator == "flow_cluster"

    def test_includes_timeframe_in_protocol(self, manifest, queue):
        from agent.generators.medium_freq.flow_cluster import generate
        hyps = generate(manifest, queue)
        for h in hyps:
            assert any("--timeframe 5min" in cmd for cmd in h.test_protocol)


# ===========================================================================
# Runner tests
# ===========================================================================

class TestMFRunner:
    def test_steps_returns_4_gates(self):
        from agent.mf_runner import MediumFrequencyRunner
        h = Hypothesis.create("test", "momentum", ["cmd --data d --symbol BTC"], 1.0,
                              thresholds={"min_ic": 0.08})
        runner = MediumFrequencyRunner(h, {})
        steps = runner.steps()
        assert len(steps) == 4
        assert steps[0].__name__ == "run_discovery"
        assert steps[1].__name__ == "run_replication_temporal"
        assert steps[2].__name__ == "run_replication_symbol"
        assert steps[3].__name__ == "run_correlation_check"

    def test_is_base_runner(self):
        from agent.mf_runner import MediumFrequencyRunner
        assert issubclass(MediumFrequencyRunner, BaseRunner)

    def test_extract_features_from_claim(self):
        from agent.mf_runner import MediumFrequencyRunner
        h = Hypothesis.create(
            "trend_momentum_300 gated by ent_tick_1m<P40 predicts 5min returns",
            "momentum", ["cmd"], 1.0)
        runner = MediumFrequencyRunner(h, {})
        features = runner._extract_features()
        assert "trend_momentum_300" in features

    def test_extract_features_default(self):
        from agent.mf_runner import MediumFrequencyRunner
        h = Hypothesis.create("some unknown claim", "momentum", ["cmd"], 1.0)
        runner = MediumFrequencyRunner(h, {})
        features = runner._extract_features()
        assert features == ["trend_momentum_300"]

    def test_extract_ic_from_results(self):
        from agent.mf_runner import MediumFrequencyRunner
        h = Hypothesis.create("test", "momentum", ["cmd"], 1.0)
        h.results = {"gate_results": [{"msg": "IC=0.120 [gated] vs min=0.08 p=0.001 PASS"}]}
        runner = MediumFrequencyRunner(h, {})
        assert abs(runner._extract_ic_from_results() - 0.120) < 1e-6

    def test_extract_ic_ignores_dIC(self):
        from agent.mf_runner import MediumFrequencyRunner
        h = Hypothesis.create("test", "momentum", ["cmd"], 1.0)
        h.results = {"gate_results": [
            {"msg": "IC=0.120 PASS"},
            {"msg": "dIC=+0.050 PASS"},
        ]}
        runner = MediumFrequencyRunner(h, {})
        assert abs(runner._extract_ic_from_results() - 0.120) < 1e-6

    def test_register_signal_writes_to_mf_registry(self, tmp_path):
        from agent.mf_runner import MediumFrequencyRunner

        # Point registry to tmp via class attr
        orig = MediumFrequencyRunner.REGISTRY_PATH
        MediumFrequencyRunner.REGISTRY_PATH = tmp_path / "registry.json"

        h = Hypothesis.create(
            "trend_momentum_300 predicts 5min returns",
            "momentum", ["cmd --data d --symbol BTC"], 1.0,
            thresholds={"horizon_s": 300.0})
        h.status = "replicated"
        h.results = {"gate_results": [{"msg": "IC=0.150 PASS"}]}

        runner = MediumFrequencyRunner(h, {})
        sig = runner.register_signal()

        assert sig.hypothesis_id == h.id
        assert sig.horizon_s == 300.0

        # Check file was written
        with open(tmp_path / "registry.json") as f:
            registry = json.load(f)
        assert len(registry) == 1
        assert registry[0]["hypothesis_id"] == h.id

        MediumFrequencyRunner.REGISTRY_PATH = orig

    def test_load_registry_separate_from_micro(self, tmp_path):
        from agent.mf_runner import MediumFrequencyRunner
        from agent.hypothesis import Hypothesis

        h = Hypothesis(id="HYP-TEST-001", claim="test", generator="test",
                       priority=1.0, test_protocol=["echo ok"])
        runner = MediumFrequencyRunner(h, {})

        orig = MediumFrequencyRunner.REGISTRY_PATH
        MediumFrequencyRunner.REGISTRY_PATH = tmp_path / "mf_registry.json"

        # MF registry should be empty (no file)
        assert runner._load_registry() == []

        # Write something to it
        with open(tmp_path / "mf_registry.json", "w") as f:
            json.dump([{"name": "mf_signal"}], f)
        assert len(runner._load_registry()) == 1

        MediumFrequencyRunner.REGISTRY_PATH = orig


# ===========================================================================
# Agent tests
# ===========================================================================

class TestMFAgent:
    def test_is_research_agent_subclass(self):
        from agent.mf_daemon import MediumFrequencyAgent
        assert issubclass(MediumFrequencyAgent, ResearchAgent)

    def test_agent_type_is_medium_freq(self):
        from agent.mf_daemon import MediumFrequencyAgent
        assert MediumFrequencyAgent.agent_type == "medium_freq"

    def test_default_generators(self):
        from agent.mf_daemon import MediumFrequencyAgent
        assert "momentum" in MediumFrequencyAgent.default_generators
        assert "vol_breakout" in MediumFrequencyAgent.default_generators
        assert "flow_cluster" in MediumFrequencyAgent.default_generators

    def test_state_paths_separate_from_microstructure(self):
        from agent.mf_daemon import MF_STATE_PATH, MF_STATS_PATH, MF_REGISTRY_PATH
        from agent.daemon import STATE_PATH, STATS_PATH
        from agent.runner import REGISTRY_PATH

        assert MF_STATE_PATH != STATE_PATH
        assert MF_STATS_PATH != STATS_PATH
        assert MF_REGISTRY_PATH != REGISTRY_PATH
        assert "agent_mf" in str(MF_STATE_PATH)
        assert "agent_mf" in str(MF_STATS_PATH)
        assert "agent_mf" in str(MF_REGISTRY_PATH)

    def test_get_generator_returns_callables(self, tmp_path):
        from agent.mf_daemon import MediumFrequencyAgent
        import agent.mf_daemon as mf_mod

        state_dir = tmp_path / "data" / "agent_mf"
        state_dir.mkdir(parents=True, exist_ok=True)

        orig_state = mf_mod.MF_STATE_PATH
        orig_stats = mf_mod.MF_STATS_PATH
        mf_mod.MF_STATE_PATH = state_dir / "agent_state.json"
        mf_mod.MF_STATS_PATH = state_dir / "generator_stats.json"

        agent = MediumFrequencyAgent()
        for name in ["momentum", "vol_breakout", "flow_cluster"]:
            gen = agent.get_generator(name)
            assert callable(gen), f"Generator {name} not callable"

        mf_mod.MF_STATE_PATH = orig_state
        mf_mod.MF_STATS_PATH = orig_stats

    def test_get_generator_unknown_returns_none(self, tmp_path):
        from agent.mf_daemon import MediumFrequencyAgent
        import agent.mf_daemon as mf_mod

        state_dir = tmp_path / "data" / "agent_mf"
        state_dir.mkdir(parents=True, exist_ok=True)

        orig_state = mf_mod.MF_STATE_PATH
        orig_stats = mf_mod.MF_STATS_PATH
        mf_mod.MF_STATE_PATH = state_dir / "agent_state.json"
        mf_mod.MF_STATS_PATH = state_dir / "generator_stats.json"

        agent = MediumFrequencyAgent()
        assert agent.get_generator("nonexistent") is None

        mf_mod.MF_STATE_PATH = orig_state
        mf_mod.MF_STATS_PATH = orig_stats

    def test_create_runner_returns_mf_runner(self, tmp_path):
        from agent.mf_daemon import MediumFrequencyAgent
        from agent.mf_runner import MediumFrequencyRunner
        import agent.mf_daemon as mf_mod

        state_dir = tmp_path / "data" / "agent_mf"
        state_dir.mkdir(parents=True, exist_ok=True)

        orig_state = mf_mod.MF_STATE_PATH
        orig_stats = mf_mod.MF_STATS_PATH
        mf_mod.MF_STATE_PATH = state_dir / "agent_state.json"
        mf_mod.MF_STATS_PATH = state_dir / "generator_stats.json"

        agent = MediumFrequencyAgent()
        h = Hypothesis.create("test", "momentum", ["cmd"], 1.0)
        runner = agent.create_runner(h, {})
        assert isinstance(runner, MediumFrequencyRunner)

        mf_mod.MF_STATE_PATH = orig_state
        mf_mod.MF_STATS_PATH = orig_stats

    def test_config_loads_agent_mf_section(self):
        from agent.mf_daemon import load_config
        config = load_config()
        # Should have MF defaults even if TOML doesn't have [agent_mf]
        assert config["cycle_interval_s"] in (7200, 3600)  # Either MF default or from TOML
        assert "momentum" in config["generators_enabled"]

    def test_run_cycle_calls_hooks(self, tmp_path):
        from agent.mf_daemon import MediumFrequencyAgent
        import agent.mf_daemon as mf_mod

        state_dir = tmp_path / "data" / "agent_mf"
        state_dir.mkdir(parents=True, exist_ok=True)

        orig_state = mf_mod.MF_STATE_PATH
        orig_stats = mf_mod.MF_STATS_PATH
        mf_mod.MF_STATE_PATH = state_dir / "agent_state.json"
        mf_mod.MF_STATS_PATH = state_dir / "generator_stats.json"

        agent = MediumFrequencyAgent()
        calls = []
        agent.build_manifest = lambda: (calls.append("manifest"),
                                         {"dates": {"2026-05-12": {}},
                                          "symbols": {"BTC": {"hours": 8}}})[1]
        agent._run_generators = lambda m: calls.append("generators")
        agent.run_monitor = lambda: calls.append("monitor")
        agent.run_cycle()
        assert calls == ["manifest", "generators", "monitor"]

        mf_mod.MF_STATE_PATH = orig_state
        mf_mod.MF_STATS_PATH = orig_stats


# ===========================================================================
# Backward compatibility
# ===========================================================================

class TestMFBackcompat:
    def test_microstructure_agent_unaffected(self):
        from agent.daemon import MicrostructureAgent, AgentDaemon
        assert AgentDaemon is MicrostructureAgent
        assert MicrostructureAgent.agent_type == "microstructure"

    def test_agent_daemon_alias_still_works(self):
        from agent.daemon import AgentDaemon
        assert AgentDaemon is not None

    def test_experiment_runner_alias_still_works(self):
        from agent.runner import ExperimentRunner, MicrostructureRunner
        assert ExperimentRunner is MicrostructureRunner

    def test_hypothesis_prefixes_dont_collide(self):
        # Register MF prefixes
        Hypothesis.register_prefix("momentum", "MOM")
        Hypothesis.register_prefix("vol_breakout", "VBK")
        Hypothesis.register_prefix("flow_cluster", "FCL")

        # Micro prefixes still intact
        assert Hypothesis.GEN_PREFIX["systematic"] == "SYS"
        assert Hypothesis.GEN_PREFIX["spectral"] == "SPE"
        assert Hypothesis.GEN_PREFIX["regime"] == "REG"

        # MF prefixes work
        h = Hypothesis.create("test", "momentum", [], 1.0)
        assert h.id.startswith("HYP-MOM-")
        h2 = Hypothesis.create("test", "vol_breakout", [], 1.0)
        assert h2.id.startswith("HYP-VBK-")

    def test_two_agents_different_types(self):
        from agent.daemon import MicrostructureAgent
        from agent.mf_daemon import MediumFrequencyAgent
        assert MicrostructureAgent.agent_type != MediumFrequencyAgent.agent_type
