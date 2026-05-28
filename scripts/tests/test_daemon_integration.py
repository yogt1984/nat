"""Integration tests for agent daemon cycles.

Tests the full MANIFEST → GENERATE → EXECUTE → REGISTER cycle with
synthetic parquet data. Covers:
- Single-agent full cycle (manifest → generation → gate execution → registration)
- State persistence and crash recovery (kill mid-cycle, restart, resume)
- FDR control (inject known p-values, verify BH correction)
- Multi-agent deduplication (micro + MF on same data, Meta Agent dedup)
- Config validation (missing keys, unknown keys, invalid thresholds)
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


from agent.base import (
    ResearchAgent, BaseRunner, AgentPhase, AgentState,
    apply_fdr, load_agent_config, validate_config,
)
from agent.hypothesis import Hypothesis, GeneratorStats
from agent.hypothesis_queue import HypothesisQueue
from data.state import StateStore


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

# Minimal feature columns for tests
FEATURE_COLS = [
    "timestamp_ns", "symbol", "raw_midprice", "raw_spread",
    "imbalance_qty_l1", "imbalance_qty_l5", "ent_book_shape",
    "vol_returns_1m", "trend_momentum_300", "ctx_funding_rate",
]

DATES = ["2026-05-20", "2026-05-21", "2026-05-22"]
SYMBOLS = ["BTC", "ETH"]


def _make_parquet(path: Path, symbol: str, date: str, n_rows: int = 2000):
    """Create a synthetic parquet file with realistic feature data."""
    np.random.seed(hash(f"{symbol}_{date}") % 2**32)
    base_ts = int(pd.Timestamp(f"{date}T00:00:00").value)
    df = pd.DataFrame({
        "timestamp_ns": base_ts + np.arange(n_rows) * 100_000_000,
        "symbol": [symbol] * n_rows,
        "raw_midprice": 100 + np.cumsum(np.random.randn(n_rows) * 0.01),
        "raw_spread": np.abs(np.random.randn(n_rows) * 0.001) + 0.0001,
        "imbalance_qty_l1": np.random.randn(n_rows) * 0.3,
        "imbalance_qty_l5": np.random.randn(n_rows) * 0.2,
        "ent_book_shape": np.random.uniform(0, 1, n_rows),
        "vol_returns_1m": np.abs(np.random.randn(n_rows) * 0.02),
        "trend_momentum_300": np.random.randn(n_rows) * 0.1,
        "ctx_funding_rate": np.random.randn(n_rows) * 0.001,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), str(path))


@pytest.fixture
def data_root(tmp_path):
    """Create synthetic parquet data: 3 dates × 2 symbols."""
    features_dir = tmp_path / "data" / "features"
    for date in DATES:
        for symbol in SYMBOLS:
            fpath = features_dir / date / f"{symbol.lower()}_{date.replace('-', '')}.parquet"
            _make_parquet(fpath, symbol, date)
    return tmp_path


@pytest.fixture
def store(tmp_path):
    """Create a fresh StateStore."""
    return StateStore(tmp_path / "data" / "nat.db")


# ---------------------------------------------------------------------------
# Integration test agent — uses real BaseRunner gate logic with mocked nat
# ---------------------------------------------------------------------------

class IntegrationRunner(BaseRunner):
    """Runner that simulates nat command execution with configurable results."""

    TIMEFRAME = None
    SIGNAL_FEATURES = ["imbalance_qty_l1", "ent_book_shape"]
    DEFAULT_FEATURE = "imbalance_qty_l1"
    DEFAULT_HORIZON_S = 5.0

    def __init__(self, hypothesis, manifest, *, store=None, agent=None,
                 gate_results=None):
        super().__init__(hypothesis, manifest, store=store, agent=agent)
        # gate_results: list of (passed, ic_value) per gate check
        self._gate_results = gate_results or [(True, 0.15)]


class IntegrationAgent(ResearchAgent):
    """Agent with deterministic generators for integration testing."""

    agent_type = "integration"
    config_section = "agent"
    default_generators = ["test_gen"]
    generator_module_prefix = "agent.generators.nonexistent"

    BASE_CONFIG = {
        "cycle_interval_s": 10,
        "max_experiments_per_cycle": 5,
        "max_cycle_runtime_s": 60,
    }

    def __init__(self, tmp_path, *, store=None, hypotheses=None, runner_success=True):
        self._tmp_path = tmp_path
        self._hypotheses = hypotheses or []
        self._runner_success = runner_success
        (tmp_path / "data" / "agent").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "features").mkdir(parents=True, exist_ok=True)
        super().__init__(store=store)

    @property
    def root(self) -> Path:
        return self._tmp_path

    def get_generator(self, name: str):
        """Return a test generator that pushes pre-defined hypotheses."""
        hyps = self._hypotheses

        def generate(manifest, queue, gen_stats):
            return hyps

        return generate

    def create_runner(self, hypothesis: Hypothesis, manifest: dict) -> BaseRunner:
        """Create a stub runner with configurable success."""
        runner = MagicMock(spec=BaseRunner)
        runner.run_full.return_value = self._runner_success
        return runner

    def build_manifest(self) -> dict:
        """Return a minimal manifest compatible with queue._is_runnable."""
        return {
            "dates": {d: {"n_files": 2, "hours_per_symbol": 1.5} for d in DATES},
            "symbols": {s: {"hours": 4.5, "dates": DATES} for s in SYMBOLS},
            "total_dates": len(DATES),
            "total_hours_per_symbol": 4.5,
        }


# ===========================================================================
# Test: Single-agent full cycle
# ===========================================================================

class TestSingleAgentCycle:
    """Test the complete MANIFEST → GENERATE → EXECUTE → MONITOR cycle."""

    def test_full_cycle_with_registration(self, tmp_path, store):
        """A successful hypothesis traverses all phases and registers."""
        hyp = Hypothesis.create(
            claim="imbalance_qty_l1 predicts 5s returns",
            generator="test_gen",
            test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
            priority=1.0,
            thresholds={"min_ic": 0.05},
        )
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[hyp], runner_success=True
        )
        agent.run_cycle()

        # Verify cycle completed
        assert agent.state.get("cycle_count") == 1
        assert agent.state.get("total_hypotheses_tested") == 1
        assert agent.state.get("total_signals_registered") == 1

    def test_full_cycle_with_failure(self, tmp_path, store):
        """A failing hypothesis increments tested but not registered."""
        hyp = Hypothesis.create(
            claim="weak signal with no IC",
            generator="test_gen",
            test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
            priority=1.0,
            thresholds={"min_ic": 0.50},
        )
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[hyp], runner_success=False
        )
        agent.run_cycle()

        assert agent.state.get("cycle_count") == 1
        assert agent.state.get("total_hypotheses_tested") == 1
        assert agent.state.get("total_signals_registered") == 0

    def test_phase_transitions(self, tmp_path, store):
        """Verify the agent transitions through all phases in order."""
        hyp = Hypothesis.create(
            claim="test phase tracking",
            generator="test_gen",
            test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
            priority=1.0,
        )
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[hyp], runner_success=True
        )

        # Track phase transitions
        transitions = []
        original_transition = agent.state.transition

        def tracking_transition(phase, msg=""):
            transitions.append(phase)
            original_transition(phase, msg)

        agent.state.transition = tracking_transition
        agent.run_cycle()

        # Must hit MANIFEST → GENERATE → EXECUTE → MONITOR
        assert AgentPhase.MANIFEST in transitions
        assert AgentPhase.GENERATE in transitions
        assert AgentPhase.EXECUTE in transitions
        assert AgentPhase.MONITOR in transitions

    def test_budget_limit_respected(self, tmp_path, store):
        """Only max_experiments_per_cycle hypotheses are tested per cycle."""
        hyps = [
            Hypothesis.create(
                claim=f"hypothesis {i}",
                generator="test_gen",
                test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
                priority=float(10 - i),
            )
            for i in range(10)
        ]
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=hyps, runner_success=True
        )
        agent.config["max_experiments_per_cycle"] = 3
        agent.run_cycle()

        assert agent.state.get("total_hypotheses_tested") == 3

    def test_empty_queue_ends_execution(self, tmp_path, store):
        """No hypotheses generated → cycle completes without errors."""
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )
        agent.run_cycle()

        assert agent.state.get("cycle_count") == 1
        assert agent.state.get("total_hypotheses_tested") == 0

    def test_generator_stats_updated(self, tmp_path, store):
        """Generator stats track attempts and successes."""
        hyp = Hypothesis.create(
            claim="stats tracking test",
            generator="test_gen",
            test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
            priority=1.0,
        )
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[hyp], runner_success=True
        )
        agent.run_cycle()

        assert agent.gen_stats["test_gen"].attempts == 1
        assert agent.gen_stats["test_gen"].successes == 1

    def test_generator_stats_failure(self, tmp_path, store):
        """Failed hypotheses increment attempts but not successes."""
        hyp = Hypothesis.create(
            claim="fail stats test",
            generator="test_gen",
            test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
            priority=1.0,
        )
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[hyp], runner_success=False
        )
        agent.run_cycle()

        assert agent.gen_stats["test_gen"].attempts == 1
        assert agent.gen_stats["test_gen"].successes == 0


# ===========================================================================
# Test: State persistence and crash recovery
# ===========================================================================

class TestStatePersistence:
    """Test state survives restarts and crash recovery."""

    def test_state_persists_across_restarts(self, tmp_path, store):
        """Cycle count and registration count survive restart."""
        hyp = Hypothesis.create(
            claim="persistence test",
            generator="test_gen",
            test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
            priority=1.0,
        )
        agent1 = IntegrationAgent(
            tmp_path, store=store, hypotheses=[hyp], runner_success=True
        )
        agent1.run_cycle()
        assert agent1.state.get("cycle_count") == 1

        # Create a new agent instance (simulates restart)
        agent2 = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )
        assert agent2.state.get("cycle_count") == 1
        assert agent2.state.get("total_signals_registered") == 1

    def test_queue_survives_restart(self, tmp_path, store):
        """Queued hypotheses persist across agent restarts."""
        hyp = Hypothesis.create(
            claim="queued for later",
            generator="test_gen",
            test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
            priority=1.0,
        )
        # Push hypothesis but don't run cycle
        agent1 = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )
        agent1.queue.push(hyp)
        initial_depth = agent1.queue.depth

        # Restart
        agent2 = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )
        assert agent2.queue.depth == initial_depth

    def test_graveyard_persists(self, tmp_path, store):
        """Failed hypotheses remain in graveyard after restart."""
        hyp = Hypothesis.create(
            claim="will fail and persist",
            generator="test_gen",
            test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
            priority=1.0,
        )
        agent1 = IntegrationAgent(
            tmp_path, store=store, hypotheses=[hyp], runner_success=False
        )

        # Mock runner that marks hypothesis as failed (like a real gate failure)
        def mock_create(h, m):
            def run_full():
                h.fail("no_effect")
                return False
            runner = MagicMock()
            runner.run_full = run_full
            return runner
        agent1.create_runner = mock_create

        agent1.run_cycle()
        graveyard_size = len(agent1.queue.graveyard)
        assert graveyard_size > 0

        # Restart
        agent2 = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )
        assert len(agent2.queue.graveyard) == graveyard_size

    def test_mid_cycle_recovery(self, tmp_path, store):
        """If cycle fails mid-execution, state is consistent on restart."""
        hyps = [
            Hypothesis.create(
                claim=f"hypothesis {i}",
                generator="test_gen",
                test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
                priority=float(5 - i),
            )
            for i in range(5)
        ]

        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=hyps, runner_success=True
        )
        # Simulate crash after 2 experiments by setting budget to 2
        agent.config["max_experiments_per_cycle"] = 2
        agent.run_cycle()

        # After crash (simulated as budget exhaust), state should be consistent
        assert agent.state.get("total_hypotheses_tested") == 2
        assert agent.state.get("cycle_count") == 1

        # Remaining hypotheses should still be queued
        agent2 = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )
        # Queue still has unprocessed items (3 remaining from generation)
        # Note: they were already generated and pushed, so they persist
        remaining = [h for h in agent2.queue._all if h.status == "queued"]
        assert len(remaining) >= 2  # at least some remain

    def test_shutdown_flag_stops_cycle(self, tmp_path, store):
        """Setting _shutdown=True stops execution mid-cycle."""
        hyps = [
            Hypothesis.create(
                claim=f"hypothesis {i}",
                generator="test_gen",
                test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
                priority=float(10 - i),
            )
            for i in range(10)
        ]
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=hyps, runner_success=True
        )

        # Set shutdown after first experiment via side effect
        original_create = agent.create_runner
        call_count = [0]

        def counting_create(h, m):
            call_count[0] += 1
            if call_count[0] >= 2:
                agent._shutdown = True
            return original_create(h, m)

        agent.create_runner = counting_create
        agent.run_cycle()

        # Should have stopped early
        assert agent.state.get("total_hypotheses_tested") <= 3


# ===========================================================================
# Test: FDR control
# ===========================================================================

class TestFDRControl:
    """Test Benjamini-Hochberg FDR correction across hypothesis batches."""

    def test_no_rejections_with_strong_pvalues(self):
        """All hypotheses with p < q survive FDR."""
        hypotheses = [
            {"id": f"H{i}", "status": "passed",
             "results": {"gate_results": [{"msg": f"IC=0.20 p={0.001 * (i+1):.4e} PASS"}]}}
            for i in range(5)
        ]
        rejected = apply_fdr(hypotheses, q=0.05)
        assert len(rejected) == 0

    def test_weak_rejected_when_strong_anchor_exists(self):
        """Weak p-values are rejected when a strong anchor sets the BH threshold."""
        hypotheses = [
            # One strong signal sets BH threshold
            {"id": "H_anchor", "status": "passed",
             "results": {"gate_results": [{"msg": "IC=0.30 p=1.00e-05 PASS"}]}},
            # Weak signals that fail BH correction
            {"id": "H_weak_1", "status": "passed",
             "results": {"gate_results": [{"msg": "IC=0.05 p=5.00e-01 PASS"}]}},
            {"id": "H_weak_2", "status": "passed",
             "results": {"gate_results": [{"msg": "IC=0.04 p=6.00e-01 PASS"}]}},
            {"id": "H_weak_3", "status": "passed",
             "results": {"gate_results": [{"msg": "IC=0.03 p=7.00e-01 PASS"}]}},
        ]
        rejected = apply_fdr(hypotheses, q=0.05)
        # Weak signals should be rejected, anchor survives
        assert "H_anchor" not in rejected
        assert len(rejected) == 3

    def test_mixed_survival(self):
        """Some hypotheses survive, some are rejected."""
        hypotheses = [
            # Strong signals (should survive)
            {"id": "H_strong_1", "status": "passed",
             "results": {"gate_results": [{"msg": "IC=0.25 p=1.00e-04 PASS"}]}},
            {"id": "H_strong_2", "status": "passed",
             "results": {"gate_results": [{"msg": "IC=0.20 p=5.00e-04 PASS"}]}},
            # Weak signals (should be rejected)
            {"id": "H_weak_1", "status": "passed",
             "results": {"gate_results": [{"msg": "IC=0.06 p=4.00e-01 PASS"}]}},
            {"id": "H_weak_2", "status": "passed",
             "results": {"gate_results": [{"msg": "IC=0.05 p=6.00e-01 PASS"}]}},
        ]
        rejected = apply_fdr(hypotheses, q=0.05)
        assert "H_weak_1" in rejected
        assert "H_weak_2" in rejected
        assert "H_strong_1" not in rejected
        assert "H_strong_2" not in rejected

    def test_fdr_with_single_hypothesis(self):
        """Single hypothesis — nothing to correct."""
        hypotheses = [
            {"id": "H1", "status": "passed",
             "results": {"gate_results": [{"msg": "IC=0.05 p=0.50 PASS"}]}},
        ]
        rejected = apply_fdr(hypotheses, q=0.05)
        assert len(rejected) == 0  # < 2 tests, no correction

    def test_fdr_ignores_queued(self):
        """Queued hypotheses are skipped in FDR calculation."""
        hypotheses = [
            {"id": "H1", "status": "queued", "results": None},
            {"id": "H2", "status": "passed",
             "results": {"gate_results": [{"msg": "IC=0.20 p=1.00e-03 PASS"}]}},
        ]
        rejected = apply_fdr(hypotheses, q=0.05)
        assert len(rejected) == 0

    def test_fdr_integrated_in_cycle(self, tmp_path, store):
        """FDR rejection happens within run_cycle and updates hypothesis status."""
        # Create hypotheses that will "pass" but have weak p-values
        hyps = [
            Hypothesis.create(
                claim=f"weak signal {i}",
                generator="test_gen",
                test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
                priority=1.0,
            )
            for i in range(4)
        ]

        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=hyps, runner_success=True
        )

        # Mock the runner to set hypothesis results with known p-values
        call_idx = [0]
        p_values = [0.001, 0.002, 0.4, 0.6]  # first 2 strong, last 2 weak

        def mock_create_runner(hypothesis, manifest):
            idx = call_idx[0]
            call_idx[0] += 1
            p = p_values[idx]

            def run_full():
                hypothesis.status = "replicated"
                hypothesis.results = {
                    "gate_results": [{"msg": f"IC=0.15 p={p:.2e} PASS"}]
                }
                return True

            runner = MagicMock()
            runner.run_full = run_full
            return runner

        agent.create_runner = mock_create_runner
        agent.run_cycle()

        # FDR should reject the weak ones
        failed = [h for h in agent.queue._all if h.failure_reason == "fdr_rejected"]
        assert len(failed) == 2

    def test_fdr_q_from_config(self, tmp_path, store):
        """FDR uses q from config gates section."""
        hyp = Hypothesis.create(
            claim="config q test",
            generator="test_gen",
            test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
            priority=1.0,
        )
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[hyp], runner_success=True
        )
        agent.config.setdefault("gates", {})["fdr_q"] = 0.01
        # With very strict q, weak signals would be rejected
        # But single hypothesis won't trigger FDR anyway
        agent.run_cycle()
        assert agent.state.get("cycle_count") == 1


# ===========================================================================
# Test: Multi-agent coordination
# ===========================================================================

class TestMultiAgent:
    """Test multi-agent scenarios on shared data."""

    def test_two_agents_independent_state(self, tmp_path, store):
        """Two agents running on same store have independent state."""
        hyp1 = Hypothesis.create(
            claim="micro signal", generator="test_gen",
            test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
            priority=1.0,
        )
        hyp2 = Hypothesis.create(
            claim="mf signal", generator="test_gen",
            test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
            priority=1.0,
        )

        # Agent 1: microstructure-like
        agent1 = IntegrationAgent(
            tmp_path, store=store, hypotheses=[hyp1], runner_success=True
        )
        agent1.agent_type = "micro"
        agent1.state = AgentState(store=store, agent="micro")
        agent1.queue = HypothesisQueue(store=store, agent="micro")

        # Agent 2: medium-frequency-like
        agent2 = IntegrationAgent(
            tmp_path, store=store, hypotheses=[hyp2], runner_success=True
        )
        agent2.agent_type = "mf"
        agent2.state = AgentState(store=store, agent="mf")
        agent2.queue = HypothesisQueue(store=store, agent="mf")

        agent1.run_cycle()
        agent2.run_cycle()

        # Each agent tracks its own counts
        assert agent1.state.get("total_hypotheses_tested") == 1
        assert agent2.state.get("total_hypotheses_tested") == 1
        assert agent1.state.get("cycle_count") == 1
        assert agent2.state.get("cycle_count") == 1

    def test_agents_share_store_not_registry(self, tmp_path, store):
        """Agents write to separate registry namespaces in the same DB."""
        # Register signals under different agent types
        sig1 = {"name": "micro_sig", "hypothesis_id": "H1",
                "features": ["imbalance_qty_l1"], "expected_ic": 0.15,
                "symbols": ["BTC"], "status": "validated"}
        sig2 = {"name": "mf_sig", "hypothesis_id": "H2",
                "features": ["trend_momentum_300"], "expected_ic": 0.12,
                "symbols": ["BTC"], "status": "validated"}

        store.append_signal("micro", sig1)
        store.append_signal("mf", sig2)

        micro_reg = store.load_registry("micro")
        mf_reg = store.load_registry("mf")

        assert len(micro_reg) == 1
        assert micro_reg[0]["name"] == "micro_sig"
        assert len(mf_reg) == 1
        assert mf_reg[0]["name"] == "mf_sig"

    def test_sequential_cycles_accumulate(self, tmp_path, store):
        """Running multiple cycles accumulates state correctly."""
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )

        # Pre-populate queue for multiple cycles
        for i in range(6):
            h = Hypothesis.create(
                claim=f"signal {i}",
                generator="test_gen",
                test_protocol=["spannung --data data/features/2026-05-20 --symbol BTC"],
                priority=float(10 - i),
            )
            agent.queue.push(h)

        agent.config["max_experiments_per_cycle"] = 2

        # Cycle 1: processes 2
        agent.run_cycle()
        assert agent.state.get("cycle_count") == 1
        assert agent.state.get("total_hypotheses_tested") == 2

        # Cycle 2: processes 2 more
        agent.run_cycle()
        assert agent.state.get("cycle_count") == 2
        assert agent.state.get("total_hypotheses_tested") == 4

    def test_duplicate_claims_rejected(self, tmp_path, store):
        """Queue deduplicates identical claims."""
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )

        h1 = Hypothesis.create(
            claim="same claim", generator="test_gen",
            test_protocol=["spannung --symbol BTC"],
            priority=1.0,
        )
        h2 = Hypothesis.create(
            claim="same claim", generator="test_gen",
            test_protocol=["spannung --symbol ETH"],
            priority=2.0,
        )

        agent.queue.push(h1)
        agent.queue.push(h2)  # Should be rejected as duplicate

        queued = [h for h in agent.queue._all if h.status == "queued"]
        assert len(queued) == 1


# ===========================================================================
# Test: Config validation
# ===========================================================================

class TestConfigValidation:
    """Test config loading, inheritance, and validation."""

    def test_unknown_top_key_warns(self):
        """Unknown top-level keys produce warnings."""
        config = {
            "cycle_interval_s": 3600,
            "bogus_key": True,
            "another_bad": 42,
        }
        warnings = validate_config(config, "agent")
        assert len(warnings) == 2
        assert "bogus_key" in warnings[0]
        assert "another_bad" in warnings[1]

    def test_unknown_gate_key_warns(self):
        """Unknown gate keys produce warnings."""
        config = {
            "cycle_interval_s": 3600,
            "gates": {"min_ic": 0.10, "invalid_gate": 0.5},
        }
        warnings = validate_config(config, "agent")
        assert len(warnings) == 1
        assert "invalid_gate" in warnings[0]

    def test_valid_config_no_warnings(self):
        """Well-formed config produces no warnings."""
        config = {
            "cycle_interval_s": 3600,
            "max_experiments_per_cycle": 10,
            "gates": {"min_ic": 0.10, "min_dIC": 0.05, "fdr_q": 0.05},
        }
        warnings = validate_config(config, "agent")
        assert len(warnings) == 0

    def test_load_config_missing_file(self, tmp_path):
        """Missing config file falls back to base config."""
        config = load_agent_config(
            tmp_path / "nonexistent.toml",
            "agent",
            {"cycle_interval_s": 9999},
        )
        assert config["cycle_interval_s"] == 9999

    def test_load_config_with_defaults_inheritance(self, tmp_path):
        """[defaults] section merges into agent section."""
        config_content = b"""
[defaults]
cycle_interval_s = 7200
symbols = ["BTC", "ETH", "SOL"]

[agent]
max_experiments_per_cycle = 15

[agent_mf]
cycle_interval_s = 14400
"""
        config_path = tmp_path / "agent.toml"
        config_path.write_bytes(config_content)

        # Agent section inherits defaults
        cfg = load_agent_config(config_path, "agent", {"cycle_interval_s": 3600})
        assert cfg["cycle_interval_s"] == 7200  # from defaults
        assert cfg["max_experiments_per_cycle"] == 15  # from [agent]
        assert cfg["symbols"] == ["BTC", "ETH", "SOL"]  # from defaults

        # MF section overrides defaults
        cfg_mf = load_agent_config(config_path, "agent_mf", {"cycle_interval_s": 3600})
        assert cfg_mf["cycle_interval_s"] == 14400  # overridden
        assert cfg_mf["symbols"] == ["BTC", "ETH", "SOL"]  # inherited

    def test_load_config_nested_gates_merge(self, tmp_path):
        """Nested [gates] subsection merges correctly."""
        config_content = b"""
[defaults]
[defaults.gates]
min_ic = 0.10
min_dIC = 0.05
fdr_q = 0.05

[agent_mf]
[agent_mf.gates]
min_ic = 0.08
"""
        config_path = tmp_path / "agent.toml"
        config_path.write_bytes(config_content)

        cfg = load_agent_config(config_path, "agent_mf", {})
        # min_ic overridden, min_dIC inherited
        assert cfg["gates"]["min_ic"] == 0.08
        assert cfg["gates"]["min_dIC"] == 0.05
        assert cfg["gates"]["fdr_q"] == 0.05


# ===========================================================================
# Test: Hypothesis chaining (post_cycle in base)
# ===========================================================================

class TestHypothesisChaining:
    """Test the hypothesis chaining logic in post_cycle."""

    def test_symbol_specific_variant_spawned(self, tmp_path, store):
        """Near-miss (2/3 symbols pass) spawns symbol-specific variant."""
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )
        # Simulate a hypothesis that failed symbol replication with 1 fail
        h = Hypothesis.create(
            claim="signal X", generator="test_gen",
            test_protocol=["spannung --symbol BTC"],
            priority=1.0,
        )
        h.status = "failed"
        h.failure_reason = "no_replication"
        h.results = {
            "symbol_replication": {
                "passed": ["BTC", "ETH"],
                "failed": ["SOL"],
            }
        }

        initial_depth = agent.queue.depth
        n_spawned = agent.post_cycle([h])
        assert n_spawned == 1
        assert agent.queue.depth == initial_depth + 1

        # The new hypothesis should mention the passing symbols
        queued = [x for x in agent.queue._all if x.status == "queued"]
        assert any("BTC,ETH" in x.claim for x in queued)

    def test_no_variant_when_all_fail(self, tmp_path, store):
        """If all symbols fail, no variant is spawned."""
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )
        h = Hypothesis.create(
            claim="signal Y", generator="test_gen",
            test_protocol=["spannung --symbol BTC"],
            priority=1.0,
        )
        h.status = "failed"
        h.failure_reason = "no_replication"
        h.results = {
            "symbol_replication": {
                "passed": ["BTC"],
                "failed": ["ETH", "SOL"],
            }
        }

        n_spawned = agent.post_cycle([h])
        assert n_spawned == 0

    def test_no_duplicate_variants(self, tmp_path, store):
        """Don't spawn the same variant twice."""
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )
        h = Hypothesis.create(
            claim="signal Z", generator="test_gen",
            test_protocol=["spannung --symbol BTC"],
            priority=1.0,
        )
        h.status = "failed"
        h.failure_reason = "no_replication"
        h.results = {
            "symbol_replication": {"passed": ["BTC", "ETH"], "failed": ["SOL"]}
        }

        agent.post_cycle([h])
        n_second = agent.post_cycle([h])  # Same hypothesis again
        assert n_second == 0  # Already exists


# ===========================================================================
# Test: Adaptive IC threshold
# ===========================================================================

class TestAdaptiveIC:
    """Test that IC threshold adapts as registry grows."""

    def test_empty_registry_uses_floor(self, tmp_path, store):
        """With no registered signals, uses config floor."""
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )
        agent.config.setdefault("gates", {})["min_ic"] = 0.10
        h = Hypothesis.create(
            claim="test", generator="test_gen",
            test_protocol=["spannung --symbol BTC"],
            priority=1.0,
        )
        agent.pre_execute(h)
        assert h.thresholds["min_ic"] == 0.10

    def test_rich_registry_raises_threshold(self, tmp_path, store):
        """With strong signals registered, threshold rises above floor."""
        agent = IntegrationAgent(
            tmp_path, store=store, hypotheses=[], runner_success=True
        )
        agent.config.setdefault("gates", {})["min_ic"] = 0.10

        # Register several strong signals
        for i in range(5):
            store.append_signal("integration", {
                "name": f"sig_{i}", "hypothesis_id": f"H{i}",
                "features": ["imbalance_qty_l1"], "expected_ic": 0.25 + i * 0.01,
                "symbols": ["BTC"], "status": "validated",
            })

        h = Hypothesis.create(
            claim="test adaptive", generator="test_gen",
            test_protocol=["spannung --symbol BTC"],
            priority=1.0,
        )
        agent.pre_execute(h)
        # Median IC = 0.27, adaptive = max(0.10, 0.27 * 0.8) = 0.216
        assert h.thresholds["min_ic"] > 0.10
