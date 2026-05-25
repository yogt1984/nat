"""Tests for agent base classes (ResearchAgent ABC, BaseRunner ABC).

Tests cover:
- ABC enforcement: cannot instantiate without implementing abstract methods
- ResearchAgent: cycle orchestration, hooks, FDR, budget/time limits
- BaseRunner: sequential gate execution, register on success, stop on failure
- Backward compatibility: AgentDaemon alias, ExperimentRunner alias
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from agent.base import ResearchAgent, BaseRunner, AgentPhase, AgentState, apply_fdr
from agent.hypothesis import Hypothesis


# ---------------------------------------------------------------------------
# Fixtures — minimal concrete subclasses for testing
# ---------------------------------------------------------------------------

class StubRunner(BaseRunner):
    """Minimal runner for testing BaseRunner contract."""

    def __init__(self, hypothesis, manifest, step_results=None):
        super().__init__(hypothesis, manifest)
        self._step_results = step_results or [True]
        self._registered = False

    def steps(self):
        return [lambda r=r: r for r in self._step_results]

    def register_signal(self):
        self._registered = True


class StubAgent(ResearchAgent):
    """Minimal agent for testing ResearchAgent contract."""

    agent_type = "stub"
    default_generators = ["gen_a"]

    def __init__(self, tmp_path, **kwargs):
        self._tmp_path = tmp_path
        (tmp_path / "data" / "agent").mkdir(parents=True, exist_ok=True)
        super().__init__(**kwargs)

    @property
    def root(self):
        return self._tmp_path

    @property
    def state_path(self):
        return self._tmp_path / "data" / "agent" / "agent_state.json"

    @property
    def queue_path(self):
        return self._tmp_path / "data" / "agent" / "hypotheses.json"

    @property
    def stats_path(self):
        return self._tmp_path / "data" / "agent" / "generator_stats.json"

    def create_runner(self, hypothesis, manifest):
        return StubRunner(hypothesis, manifest, step_results=[True])


@pytest.fixture
def tmp_agent(tmp_path):
    return StubAgent(tmp_path)


# ===========================================================================
# ResearchAgent ABC enforcement
# ===========================================================================

class TestResearchAgentABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError, match="abstract"):
            ResearchAgent()

    def test_get_generator_has_default_impl(self, tmp_path):
        """get_generator uses generator_module_prefix, returns None for missing."""
        from unittest.mock import MagicMock
        class Minimal(ResearchAgent):
            agent_type = "minimal"
            generator_module_prefix = "agent.generators.nonexistent"
            @property
            def root(self):
                return tmp_path
            def create_runner(self, h, m):
                pass
        agent = Minimal(store=MagicMock())
        assert agent.get_generator("no_such_gen") is None

    def test_must_implement_create_runner(self, tmp_path):
        class Bad(ResearchAgent):
            agent_type = "bad"
            @property
            def root(self):
                return tmp_path
        with pytest.raises(TypeError, match="create_runner"):
            Bad()

    def test_run_monitor_has_default_impl(self, tmp_path):
        """run_monitor is concrete — it has shared IC decay + promotion logic."""
        class Minimal(ResearchAgent):
            agent_type = "minimal"
            @property
            def root(self):
                return tmp_path
            def create_runner(self, h, m):
                pass
        agent = Minimal()
        # Should not raise — base run_monitor handles missing registry gracefully
        agent.run_monitor()


# ===========================================================================
# ResearchAgent lifecycle
# ===========================================================================

class TestResearchAgentLifecycle:
    def test_initial_state(self, tmp_agent):
        assert tmp_agent.state.phase == AgentPhase.IDLE
        assert tmp_agent.queue.depth == 0
        assert tmp_agent._shutdown is False

    def test_run_cycle_calls_hooks_in_order(self, tmp_agent):
        """run_cycle must call: build_manifest → generators → execute → monitor."""
        calls = []
        tmp_agent.build_manifest = lambda: (calls.append("manifest"), {"dates": {}})[1]
        tmp_agent._run_generators = lambda m: calls.append("generators")
        tmp_agent.run_monitor = lambda: calls.append("monitor")
        tmp_agent.run_cycle()
        assert calls == ["manifest", "generators", "monitor"]

    def test_pre_execute_called_per_hypothesis(self, tmp_agent):
        """pre_execute hook is called before each hypothesis execution."""
        pre_calls = []
        h = Hypothesis.create("test claim", "gen_a", ["cmd --data d --symbol BTC"], 1.0)
        tmp_agent.queue.push(h)
        tmp_agent.pre_execute = lambda hyp: pre_calls.append(hyp.id)
        tmp_agent.build_manifest = lambda: {"dates": {"2026-05-12": {}}, "symbols": {"BTC": {"hours": 8}}}
        tmp_agent.run_cycle()
        assert len(pre_calls) == 1

    def test_post_cycle_called_with_results(self, tmp_agent):
        """post_cycle receives the list of tested hypotheses."""
        post_args = []
        h = Hypothesis.create("test claim", "gen_a", ["cmd --data d --symbol BTC"], 1.0)
        tmp_agent.queue.push(h)
        original_post = tmp_agent.post_cycle
        tmp_agent.post_cycle = lambda ch: (post_args.append(ch), 0)[1]
        tmp_agent.build_manifest = lambda: {"dates": {"2026-05-12": {}}, "symbols": {"BTC": {"hours": 8}}}
        tmp_agent.run_cycle()
        assert len(post_args) == 1
        assert len(post_args[0]) == 1  # one hypothesis tested

    def test_shutdown_flag_stops_cycle(self, tmp_agent):
        """Setting _shutdown=True stops the execution loop."""
        for i in range(5):
            h = Hypothesis.create(f"claim {i}", "gen_a", ["cmd --data d --symbol BTC"], 1.0)
            tmp_agent.queue.push(h)
        tmp_agent._shutdown = True
        tmp_agent.build_manifest = lambda: {"dates": {"2026-05-12": {}}, "symbols": {"BTC": {"hours": 8}}}
        tmp_agent.run_cycle()
        # No hypotheses should be tested since shutdown is set
        assert tmp_agent.state.get("total_hypotheses_tested", 0) == 0

    def test_budget_limit_stops_execution(self, tmp_agent):
        """max_experiments_per_cycle limits how many hypotheses run."""
        tmp_agent.config["max_experiments_per_cycle"] = 2
        for i in range(5):
            h = Hypothesis.create(f"claim {i}", "gen_a", ["cmd --data d --symbol BTC"],
                                  priority=5 - i)
            tmp_agent.queue.push(h)
        tmp_agent.build_manifest = lambda: {"dates": {"2026-05-12": {}}, "symbols": {"BTC": {"hours": 8}}}
        tmp_agent.run_cycle()
        assert tmp_agent.state.get("total_hypotheses_tested") == 2

    def test_failed_runner_not_registered(self, tmp_agent):
        """A hypothesis whose runner fails should not increment registered count."""
        h = Hypothesis.create("failing", "gen_a", ["cmd --data d --symbol BTC"], 1.0)
        tmp_agent.queue.push(h)
        tmp_agent.create_runner = lambda hyp, m: StubRunner(hyp, m, step_results=[False])
        tmp_agent.build_manifest = lambda: {"dates": {"2026-05-12": {}}, "symbols": {"BTC": {"hours": 8}}}
        tmp_agent.run_cycle()
        assert tmp_agent.state.get("total_signals_registered", 0) == 0

    def test_successful_runner_increments_registered(self, tmp_agent):
        """A hypothesis whose runner passes all gates should increment registered."""
        h = Hypothesis.create("passing", "gen_a", ["cmd --data d --symbol BTC"], 1.0)
        tmp_agent.queue.push(h)
        tmp_agent.build_manifest = lambda: {"dates": {"2026-05-12": {}}, "symbols": {"BTC": {"hours": 8}}}
        tmp_agent.run_cycle()
        assert tmp_agent.state.get("total_signals_registered") == 1

    def test_generator_stats_updated(self, tmp_agent):
        """Generator stats should be updated after execution."""
        h = Hypothesis.create("claim", "gen_a", ["cmd --data d --symbol BTC"], 1.0)
        tmp_agent.queue.push(h)
        tmp_agent.build_manifest = lambda: {"dates": {"2026-05-12": {}}, "symbols": {"BTC": {"hours": 8}}}
        tmp_agent.run_cycle()
        assert tmp_agent.gen_stats["gen_a"].attempts == 1
        assert tmp_agent.gen_stats["gen_a"].successes == 1


# ===========================================================================
# BaseRunner ABC enforcement
# ===========================================================================

class TestBaseRunnerABC:
    def test_instantiable_with_defaults(self):
        """BaseRunner can be instantiated — has concrete steps/register."""
        h = Hypothesis.create("test", "gen", ["cmd --data d --symbol BTC"], 1.0)
        runner = BaseRunner(h, {})
        assert runner.h is h
        assert runner.gate_results == []

    def test_default_steps_returns_4_gates(self):
        """Base steps() provides default 4-gate protocol."""
        h = Hypothesis.create("test", "gen", ["cmd --data d --symbol BTC"], 1.0)
        runner = StubRunner(h, {})
        base_steps = BaseRunner.steps(runner)
        assert len(base_steps) == 4
        names = [s.__name__ for s in base_steps]
        assert names == ["run_discovery", "run_replication_temporal",
                         "run_replication_symbol", "run_correlation_check"]

    def test_default_register_signal_uses_registry_path(self, tmp_path):
        """Base register_signal() writes to REGISTRY_PATH class attr."""
        reg_path = tmp_path / "registry.json"

        class TestRunner(BaseRunner):
            REGISTRY_PATH = reg_path
            DEFAULT_HORIZON_S = 42.0
            DEFAULT_FEATURE = "test_feat"

        h = Hypothesis.create("test claim", "gen",
                              ["cmd --data d --symbol BTC"], 1.0)
        h.status = "replicated"
        h.results = {"gate_results": [{"msg": "IC=0.15 PASS"}]}
        runner = TestRunner(h, {})
        sig = runner.register_signal()
        assert sig.horizon_s == 42.0
        assert reg_path.exists()
        with open(reg_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["hypothesis_id"] == h.id


# ===========================================================================
# BaseRunner behavior
# ===========================================================================

class TestBaseRunnerBehavior:
    def test_run_full_calls_steps_sequentially(self):
        h = Hypothesis.create("test", "gen", ["cmd"], 1.0)
        calls = []
        runner = StubRunner(h, {}, step_results=[True, True, True])
        # Replace steps with tracking versions
        runner.steps = lambda: [
            lambda: (calls.append(1), True)[1],
            lambda: (calls.append(2), True)[1],
            lambda: (calls.append(3), True)[1],
        ]
        runner.run_full()
        assert calls == [1, 2, 3]

    def test_run_full_stops_on_first_failure(self):
        h = Hypothesis.create("test", "gen", ["cmd"], 1.0)
        calls = []
        runner = StubRunner(h, {})
        runner.steps = lambda: [
            lambda: (calls.append(1), True)[1],
            lambda: (calls.append(2), False)[1],
            lambda: (calls.append(3), True)[1],
        ]
        result = runner.run_full()
        assert result is False
        assert calls == [1, 2]  # Step 3 never called

    def test_run_full_calls_register_on_success(self):
        h = Hypothesis.create("test", "gen", ["cmd"], 1.0)
        runner = StubRunner(h, {}, step_results=[True, True])
        result = runner.run_full()
        assert result is True
        assert runner._registered is True

    def test_run_full_no_register_on_failure(self):
        h = Hypothesis.create("test", "gen", ["cmd"], 1.0)
        runner = StubRunner(h, {}, step_results=[True, False])
        result = runner.run_full()
        assert result is False
        assert runner._registered is False

    def test_extract_symbol_from_cmd(self):
        assert BaseRunner._extract_symbol("regime --data d --symbol ETH") == "ETH"
        assert BaseRunner._extract_symbol("regime --data d") == "BTC"  # default

    def test_extract_data_dir_from_cmd(self):
        h = Hypothesis.create("test", "gen", ["cmd --data data/features/2026-05-12 --symbol BTC"], 1.0)
        runner = StubRunner(h, {})
        assert runner._extract_data_dir() == "data/features/2026-05-12"

    def test_extract_data_dir_fallback(self):
        h = Hypothesis.create("test", "gen", ["cmd --symbol BTC"], 1.0)
        runner = StubRunner(h, {"dates": {"2026-05-12": {}, "2026-05-13": {}}})
        assert runner._extract_data_dir() == "data/features/2026-05-13"


# ===========================================================================
# FDR (moved to base)
# ===========================================================================

class TestApplyFDR:
    def test_empty_returns_empty(self):
        assert apply_fdr([]) == []

    def test_single_hypothesis_returns_empty(self):
        h = {"id": "H1", "status": "passed",
             "results": {"gate_results": [{"msg": "IC=0.5 p=0.01 PASS"}]}}
        assert apply_fdr([h]) == []

    def test_rejects_high_pvalue(self):
        h1 = {"id": "H1", "status": "passed",
               "results": {"gate_results": [{"msg": "IC=0.5 p=0.001 PASS"}]}}
        h2 = {"id": "H2", "status": "passed",
               "results": {"gate_results": [{"msg": "IC=0.1 p=0.80 PASS"}]}}
        rejected = apply_fdr([h1, h2], q=0.05)
        assert "H2" in rejected
        assert "H1" not in rejected


# ===========================================================================
# Backward compatibility
# ===========================================================================

class TestBackwardCompatibility:
    def test_agent_daemon_alias(self):
        from agent.daemon import AgentDaemon, MicrostructureAgent
        assert AgentDaemon is MicrostructureAgent

    def test_experiment_runner_alias(self):
        from agent.runner import ExperimentRunner, MicrostructureRunner
        assert ExperimentRunner is MicrostructureRunner

    def test_agent_phase_importable_from_daemon(self):
        from agent.daemon import AgentPhase
        assert AgentPhase.IDLE.value == "IDLE"

    def test_agent_state_importable_from_daemon(self):
        from agent.daemon import AgentState
        assert AgentState is not None

    def test_apply_fdr_importable_from_runner(self):
        from agent.runner import apply_fdr
        assert callable(apply_fdr)

    def test_base_runner_importable_from_base(self):
        from agent.base import BaseRunner
        assert BaseRunner is not None

    def test_microstructure_agent_is_research_agent(self):
        from agent.daemon import MicrostructureAgent
        assert issubclass(MicrostructureAgent, ResearchAgent)

    def test_microstructure_runner_is_base_runner(self):
        from agent.runner import MicrostructureRunner
        assert issubclass(MicrostructureRunner, BaseRunner)


# ===========================================================================
# Hypothesis.register_prefix
# ===========================================================================

class TestRegisterPrefix:
    def test_register_new_prefix(self):
        Hypothesis.register_prefix("momentum", "MOM")
        assert Hypothesis.GEN_PREFIX["momentum"] == "MOM"
        h = Hypothesis.create("test", "momentum", [], 1.0)
        assert h.id.startswith("HYP-MOM-")
        # Clean up
        del Hypothesis.GEN_PREFIX["momentum"]

    def test_override_existing_prefix(self):
        old = Hypothesis.GEN_PREFIX.get("systematic")
        Hypothesis.register_prefix("systematic", "NEW")
        assert Hypothesis.GEN_PREFIX["systematic"] == "NEW"
        # Restore
        Hypothesis.GEN_PREFIX["systematic"] = old
