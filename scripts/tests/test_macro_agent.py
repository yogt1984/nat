"""Tests for the Macro Agent (4.3).

Tests cover:
- Generator contracts: funding_meanrev, oi_divergence, whale_momentum
- MacroRunner: 4-gate protocol, registration, helpers
- MacroAgent: ABC compliance, state isolation, lifecycle
- Backward compatibility: micro and MF agents unaffected
"""

import json
from pathlib import Path


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
def macro_agent(tmp_path):
    from agent.macro_daemon import MacroAgent
    from data.state import StateStore

    store = StateStore(tmp_path / "nat.db")
    agent = MacroAgent(store=store)
    return agent


# ===========================================================================
# Generator tests
# ===========================================================================

class TestFundingMeanrevGenerator:
    def test_generates_valid_hypotheses(self, manifest, queue):
        from agent.generators.macro.funding_meanrev import generate
        hyps = generate(manifest, queue)
        assert len(hyps) > 0
        for h in hyps:
            assert h.generator == "funding_meanrev"
            assert h.id.startswith("HYP-FRM-")

    def test_deduplicates_against_queue(self, manifest, queue):
        from agent.generators.macro.funding_meanrev import generate
        hyps1 = generate(manifest, queue)
        for h in hyps1:
            queue.push(h)
        hyps2 = generate(manifest, queue)
        assert len(hyps2) == 0

    def test_empty_manifest_returns_empty(self, queue):
        from agent.generators.macro.funding_meanrev import generate
        assert generate({}, queue) == []

    def test_includes_timeframe_in_protocol(self, manifest, queue):
        from agent.generators.macro.funding_meanrev import generate
        hyps = generate(manifest, queue)
        for h in hyps:
            assert any("--timeframe 1h" in cmd for cmd in h.test_protocol)


class TestOIDivergenceGenerator:
    def test_generates_valid_hypotheses(self, manifest, queue):
        from agent.generators.macro.oi_divergence import generate
        hyps = generate(manifest, queue)
        assert len(hyps) > 0
        assert all(h.generator == "oi_divergence" for h in hyps)

    def test_uses_correct_prefix(self, manifest, queue):
        from agent.generators.macro.oi_divergence import generate
        hyps = generate(manifest, queue)
        for h in hyps:
            assert h.id.startswith("HYP-OID-")

    def test_includes_timeframe_in_protocol(self, manifest, queue):
        from agent.generators.macro.oi_divergence import generate
        hyps = generate(manifest, queue)
        for h in hyps:
            assert any("--timeframe 1h" in cmd for cmd in h.test_protocol)


class TestWhaleMomentumGenerator:
    def test_generates_valid_hypotheses(self, manifest, queue):
        from agent.generators.macro.whale_momentum import generate
        hyps = generate(manifest, queue)
        assert len(hyps) > 0
        assert all(h.generator == "whale_momentum" for h in hyps)

    def test_uses_correct_prefix(self, manifest, queue):
        from agent.generators.macro.whale_momentum import generate
        hyps = generate(manifest, queue)
        for h in hyps:
            assert h.id.startswith("HYP-WHM-")

    def test_includes_timeframe_in_protocol(self, manifest, queue):
        from agent.generators.macro.whale_momentum import generate
        hyps = generate(manifest, queue)
        for h in hyps:
            assert any("--timeframe 1h" in cmd for cmd in h.test_protocol)


# ===========================================================================
# Runner tests
# ===========================================================================

class TestMacroRunner:
    def test_steps_returns_4_gates(self):
        from agent.macro_runner import MacroRunner
        h = Hypothesis.create("test", "funding_meanrev",
                              ["cmd --data d --symbol BTC"], 1.0,
                              thresholds={"min_ic": 0.07})
        runner = MacroRunner(h, {})
        steps = runner.steps()
        assert len(steps) == 4
        assert steps[0].__name__ == "run_discovery"
        assert steps[1].__name__ == "run_replication_temporal"
        assert steps[2].__name__ == "run_replication_symbol"
        assert steps[3].__name__ == "run_correlation_check"

    def test_is_base_runner(self):
        from agent.macro_runner import MacroRunner
        assert issubclass(MacroRunner, BaseRunner)

    def test_extract_features_from_claim(self):
        from agent.macro_runner import MacroRunner
        h = Hypothesis.create(
            "ctx_funding_zscore meanrev gated by vol<P40 predicts 1h returns",
            "funding_meanrev", ["cmd"], 1.0)
        runner = MacroRunner(h, {})
        features = runner._extract_features()
        assert "ctx_funding_zscore" in features

    def test_extract_features_default(self):
        from agent.macro_runner import MacroRunner
        h = Hypothesis.create("some unknown claim", "funding_meanrev", ["cmd"], 1.0)
        runner = MacroRunner(h, {})
        features = runner._extract_features()
        assert features == ["ctx_funding_zscore"]

    def test_extract_ic_from_results(self):
        from agent.macro_runner import MacroRunner
        h = Hypothesis.create("test", "funding_meanrev", ["cmd"], 1.0)
        h.results = {"gate_results": [{"msg": "IC=0.095 PASS"}]}
        runner = MacroRunner(h, {})
        assert abs(runner._extract_ic_from_results() - 0.095) < 1e-6

    def test_register_signal_writes_to_macro_registry(self, tmp_path):
        from agent.macro_runner import MacroRunner

        orig = MacroRunner.REGISTRY_PATH
        MacroRunner.REGISTRY_PATH = tmp_path / "registry.json"

        h = Hypothesis.create(
            "ctx_funding_zscore predicts 1h returns",
            "funding_meanrev", ["cmd --data d --symbol BTC"], 1.0,
            thresholds={"horizon_s": 3600.0})
        h.status = "replicated"
        h.results = {"gate_results": [{"msg": "IC=0.095 PASS"}]}

        runner = MacroRunner(h, {})
        sig = runner.register_signal()

        assert sig.hypothesis_id == h.id
        assert sig.horizon_s == 3600.0

        with open(tmp_path / "registry.json") as f:
            registry = json.load(f)
        assert len(registry) == 1
        assert registry[0]["hypothesis_id"] == h.id

        MacroRunner.REGISTRY_PATH = orig

    def test_register_signal_stores_1h_horizon(self, tmp_path):
        from agent.macro_runner import MacroRunner

        orig = MacroRunner.REGISTRY_PATH
        MacroRunner.REGISTRY_PATH = tmp_path / "registry.json"

        h = Hypothesis.create("test", "funding_meanrev",
                              ["cmd --data d --symbol BTC"], 1.0,
                              thresholds={"horizon_s": 3600.0})
        h.status = "replicated"
        h.results = {"gate_results": [{"msg": "IC=0.08 PASS"}]}

        runner = MacroRunner(h, {})
        sig = runner.register_signal()
        assert sig.horizon_s == 3600.0

        MacroRunner.REGISTRY_PATH = orig

    def test_load_registry_separate_from_micro_and_mf(self, tmp_path):
        from agent.macro_runner import MacroRunner
        from agent.hypothesis import Hypothesis

        h = Hypothesis(id="HYP-TEST-001", claim="test", generator="test",
                       priority=1.0, test_protocol=["echo ok"])
        runner = MacroRunner(h, {})

        orig = MacroRunner.REGISTRY_PATH
        MacroRunner.REGISTRY_PATH = tmp_path / "macro_registry.json"

        assert runner._load_registry() == []

        with open(tmp_path / "macro_registry.json", "w") as f:
            json.dump([{"name": "macro_signal"}], f)
        assert len(runner._load_registry()) == 1

        MacroRunner.REGISTRY_PATH = orig

    def test_timeframe_is_1h(self):
        from agent.macro_runner import MacroRunner
        assert MacroRunner.TIMEFRAME == "1h"


# ===========================================================================
# Agent tests
# ===========================================================================

class TestMacroAgent:
    def test_is_research_agent_subclass(self):
        from agent.macro_daemon import MacroAgent
        assert issubclass(MacroAgent, ResearchAgent)

    def test_agent_type_is_macro(self):
        from agent.macro_daemon import MacroAgent
        assert MacroAgent.agent_type == "macro"

    def test_default_generators(self):
        from agent.macro_daemon import MacroAgent
        assert "funding_meanrev" in MacroAgent.default_generators
        assert "oi_divergence" in MacroAgent.default_generators
        assert "whale_momentum" in MacroAgent.default_generators

    def test_state_paths_separate_from_micro_and_mf(self):
        from agent.macro_daemon import MACRO_STATE_PATH, MACRO_STATS_PATH, MACRO_REGISTRY_PATH
        from agent.daemon import STATE_PATH, STATS_PATH
        from agent.runner import REGISTRY_PATH
        from agent.mf_daemon import MF_STATE_PATH, MF_STATS_PATH, MF_REGISTRY_PATH

        # Separate from microstructure
        assert MACRO_STATE_PATH != STATE_PATH
        assert MACRO_REGISTRY_PATH != REGISTRY_PATH
        # Separate from MF
        assert MACRO_STATE_PATH != MF_STATE_PATH
        assert MACRO_REGISTRY_PATH != MF_REGISTRY_PATH
        # In correct directory
        assert "agent_macro" in str(MACRO_STATE_PATH)

    def test_get_generator_returns_callables(self, tmp_path):
        from agent.macro_daemon import MacroAgent
        import agent.macro_daemon as macro_mod

        state_dir = tmp_path / "data" / "agent_macro"
        state_dir.mkdir(parents=True, exist_ok=True)

        orig_state = macro_mod.MACRO_STATE_PATH
        orig_stats = macro_mod.MACRO_STATS_PATH
        macro_mod.MACRO_STATE_PATH = state_dir / "agent_state.json"
        macro_mod.MACRO_STATS_PATH = state_dir / "generator_stats.json"

        agent = MacroAgent()
        for name in ["funding_meanrev", "oi_divergence", "whale_momentum"]:
            gen = agent.get_generator(name)
            assert callable(gen), f"Generator {name} not callable"

        macro_mod.MACRO_STATE_PATH = orig_state
        macro_mod.MACRO_STATS_PATH = orig_stats

    def test_get_generator_unknown_returns_none(self, tmp_path):
        from agent.macro_daemon import MacroAgent
        import agent.macro_daemon as macro_mod

        state_dir = tmp_path / "data" / "agent_macro"
        state_dir.mkdir(parents=True, exist_ok=True)

        orig_state = macro_mod.MACRO_STATE_PATH
        orig_stats = macro_mod.MACRO_STATS_PATH
        macro_mod.MACRO_STATE_PATH = state_dir / "agent_state.json"
        macro_mod.MACRO_STATS_PATH = state_dir / "generator_stats.json"

        agent = MacroAgent()
        assert agent.get_generator("nonexistent") is None

        macro_mod.MACRO_STATE_PATH = orig_state
        macro_mod.MACRO_STATS_PATH = orig_stats

    def test_create_runner_returns_macro_runner(self, tmp_path):
        from agent.macro_daemon import MacroAgent
        from agent.macro_runner import MacroRunner
        import agent.macro_daemon as macro_mod

        state_dir = tmp_path / "data" / "agent_macro"
        state_dir.mkdir(parents=True, exist_ok=True)

        orig_state = macro_mod.MACRO_STATE_PATH
        orig_stats = macro_mod.MACRO_STATS_PATH
        macro_mod.MACRO_STATE_PATH = state_dir / "agent_state.json"
        macro_mod.MACRO_STATS_PATH = state_dir / "generator_stats.json"

        agent = MacroAgent()
        h = Hypothesis.create("test", "funding_meanrev", ["cmd"], 1.0)
        runner = agent.create_runner(h, {})
        assert isinstance(runner, MacroRunner)

        macro_mod.MACRO_STATE_PATH = orig_state
        macro_mod.MACRO_STATS_PATH = orig_stats

    def test_config_loads_agent_macro_section(self):
        from agent.macro_daemon import load_config
        config = load_config()
        assert "funding_meanrev" in config["generators_enabled"]

    def test_run_cycle_calls_hooks(self, tmp_path):
        from agent.macro_daemon import MacroAgent
        import agent.macro_daemon as macro_mod

        state_dir = tmp_path / "data" / "agent_macro"
        state_dir.mkdir(parents=True, exist_ok=True)

        orig_state = macro_mod.MACRO_STATE_PATH
        orig_stats = macro_mod.MACRO_STATS_PATH
        macro_mod.MACRO_STATE_PATH = state_dir / "agent_state.json"
        macro_mod.MACRO_STATS_PATH = state_dir / "generator_stats.json"

        agent = MacroAgent()
        calls = []
        agent.build_manifest = lambda: (calls.append("manifest"),
                                         {"dates": {"2026-05-12": {}},
                                          "symbols": {"BTC": {"hours": 8}}})[1]
        agent._run_generators = lambda m: calls.append("generators")
        agent.run_monitor = lambda: calls.append("monitor")
        agent.run_cycle()
        assert calls == ["manifest", "generators", "monitor"]

        macro_mod.MACRO_STATE_PATH = orig_state
        macro_mod.MACRO_STATS_PATH = orig_stats


# ===========================================================================
# Backward compatibility
# ===========================================================================

class TestMacroBackcompat:
    def test_microstructure_agent_unaffected(self):
        from agent.daemon import MicrostructureAgent, AgentDaemon
        assert AgentDaemon is MicrostructureAgent
        assert MicrostructureAgent.agent_type == "microstructure"

    def test_mf_agent_unaffected(self):
        from agent.mf_daemon import MediumFrequencyAgent
        assert MediumFrequencyAgent.agent_type == "medium_freq"

    def test_hypothesis_prefixes_dont_collide(self):
        # Ensure MF generators are imported (registers prefixes)
        import agent.generators.medium_freq.momentum  # noqa: F401

        # Macro prefixes
        assert Hypothesis.GEN_PREFIX["funding_meanrev"] == "FRM"
        assert Hypothesis.GEN_PREFIX["oi_divergence"] == "OID"
        assert Hypothesis.GEN_PREFIX["whale_momentum"] == "WHM"
        # Micro prefixes still intact
        assert Hypothesis.GEN_PREFIX["systematic"] == "SYS"
        # MF prefixes still intact
        assert Hypothesis.GEN_PREFIX["momentum"] == "MOM"

    def test_three_agents_different_types(self):
        from agent.daemon import MicrostructureAgent
        from agent.mf_daemon import MediumFrequencyAgent
        from agent.macro_daemon import MacroAgent
        types = {MicrostructureAgent.agent_type,
                 MediumFrequencyAgent.agent_type,
                 MacroAgent.agent_type}
        assert len(types) == 3
