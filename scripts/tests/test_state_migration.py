"""Tests for P1-4: SQLite state migration of meta, cascade, pipeline, alpha_pipeline.

Verifies:
- MetaAgent uses StateStore for state/stats/registry reads
- CascadeState uses StateStore for state + gate history
- PipelineState dual-mode: SQLite when store provided, JSON fallback
- AlphaPipelineState dual-mode: same pattern
- CLI status/migrate commands
- Cross-agent queries via store.all_registries()
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from data.state import StateStore


# ===========================================================================
# MetaAgent StateStore integration
# ===========================================================================

class TestMetaStateStore:
    @pytest.fixture
    def store(self, tmp_path):
        return StateStore(tmp_path / "test.db")

    def test_meta_agent_uses_store(self, store, monkeypatch, tmp_path):
        import agent.meta_daemon as meta_mod
        monkeypatch.setattr(meta_mod, "DB_PATH", tmp_path / "test.db")
        monkeypatch.setattr(meta_mod, "META_STATE_PATH", tmp_path / "x.json")
        monkeypatch.setattr(meta_mod, "CORRELATION_PATH", tmp_path / "corr.json")
        monkeypatch.setattr(meta_mod, "PORTFOLIO_PATH", tmp_path / "port.json")

        from agent.meta_daemon import MetaAgent
        agent = MetaAgent(store=store)
        assert agent._state["phase"] == "IDLE"
        assert agent._state["cycle_count"] == 0

    def test_meta_saves_and_reloads(self, store, monkeypatch, tmp_path):
        import agent.meta_daemon as meta_mod
        monkeypatch.setattr(meta_mod, "DB_PATH", tmp_path / "test.db")
        monkeypatch.setattr(meta_mod, "META_STATE_PATH", tmp_path / "x.json")
        monkeypatch.setattr(meta_mod, "CORRELATION_PATH", tmp_path / "corr.json")
        monkeypatch.setattr(meta_mod, "PORTFOLIO_PATH", tmp_path / "port.json")

        from agent.meta_daemon import MetaAgent
        agent = MetaAgent(store=store)
        agent._state["cycle_count"] = 5
        agent._state["phase"] = "RUNNING"
        agent._save_state()

        # Reload
        agent2 = MetaAgent(store=store)
        assert agent2._state["cycle_count"] == 5
        assert agent2._state["phase"] == "RUNNING"

    def test_meta_reads_registries_from_store(self, store):
        # Seed registries for two agents
        store.append_signal("microstructure", {
            "name": "sig_micro", "features": ["f1"], "status": "validated",
        })
        store.append_signal("macro", {
            "name": "sig_macro", "features": ["f2"], "status": "validated",
        })

        all_regs = store.all_registries()
        assert "microstructure" in all_regs
        assert "macro" in all_regs
        assert all_regs["microstructure"][0]["name"] == "sig_micro"

    def test_meta_reads_gen_stats_from_store(self, store):
        store.save_gen_stats("medium_freq", {
            "momentum": {"attempts": 10, "successes": 3},
        })
        loaded = store.load_gen_stats("medium_freq")
        assert loaded["momentum"]["attempts"] == 10


# ===========================================================================
# CascadeState StateStore integration
# ===========================================================================

class TestCascadeStateStore:
    @pytest.fixture
    def store(self, tmp_path):
        return StateStore(tmp_path / "test.db")

    def test_cascade_state_roundtrip(self, store):
        from agent.cascade_daemon import CascadeState
        state = CascadeState(store)
        assert state.data["phase"] == "idle"
        assert state.data["cycle_count"] == 0

        state.data["phase"] = "validating"
        state.data["cycle_count"] = 3
        state.save()

        # Reload
        state2 = CascadeState(store)
        assert state2.data["phase"] == "validating"
        assert state2.data["cycle_count"] == 3

    def test_cascade_gate_history_persists(self, store):
        from agent.cascade_daemon import CascadeState
        state = CascadeState(store)
        results = {
            "timestamp": "2026-05-25T10:00:00",
            "gates": {"G1": {"passed": True}, "G2": {"passed": False}},
            "overall": False,
        }
        state.append_gate_result(results)

        # Reload and check history
        state2 = CascadeState(store)
        history = state2.data["gate_history"]
        assert len(history) >= 1


# ===========================================================================
# PipelineState dual-mode
# ===========================================================================

class TestPipelineStateDualMode:
    def test_json_fallback(self, tmp_path):
        from pipeline_runner import PipelineState, State
        sf = str(tmp_path / "state.json")
        ps = PipelineState(sf)
        ps.transition(State.BUILDING, "test")
        assert ps.current == State.BUILDING

        # Verify JSON file exists
        assert (tmp_path / "state.json").exists()

    def test_sqlite_mode(self, tmp_path):
        from pipeline_runner import PipelineState, State
        store = StateStore(tmp_path / "test.db")
        sf = str(tmp_path / "state.json")
        ps = PipelineState(sf, store=store)
        ps.transition(State.INGESTING, "via store")
        assert ps.current == State.INGESTING

        # Verify state in SQLite
        saved = store.load_state("pipeline")
        assert saved["state"] == "INGESTING"

    def test_json_migration_to_sqlite(self, tmp_path):
        from pipeline_runner import PipelineState, State
        # Create legacy JSON state
        legacy = tmp_path / "state.json"
        legacy.write_text(json.dumps({
            "state": "ANALYZING",
            "started_at": "2026-01-01T00:00:00",
            "history": [{"from": "IDLE", "to": "ANALYZING", "at": "t1"}],
        }))

        store = StateStore(tmp_path / "test.db")
        ps = PipelineState(str(legacy), store=store)
        assert ps.current == State.ANALYZING


# ===========================================================================
# AlphaPipelineState dual-mode
# ===========================================================================

class TestAlphaPipelineStateDualMode:
    def test_json_fallback(self, tmp_path):
        from alpha.alpha_pipeline import AlphaPipelineState, Phase
        sf = str(tmp_path / "alpha_state.json")
        ps = AlphaPipelineState(sf)
        ps.transition(Phase.SCREENING, "test")
        assert ps.current == Phase.SCREENING
        assert (tmp_path / "alpha_state.json").exists()

    def test_sqlite_mode(self, tmp_path):
        from alpha.alpha_pipeline import AlphaPipelineState, Phase
        store = StateStore(tmp_path / "test.db")
        sf = str(tmp_path / "alpha_state.json")
        ps = AlphaPipelineState(sf, store=store)
        ps.transition(Phase.COMBINING, "via store")
        assert ps.current == Phase.COMBINING

        saved = store.load_state("alpha_pipeline")
        assert saved["phase"] == "COMBINING"

    def test_record_gate_persists(self, tmp_path):
        from alpha.alpha_pipeline import AlphaPipelineState, Phase
        store = StateStore(tmp_path / "test.db")
        sf = str(tmp_path / "state.json")
        ps = AlphaPipelineState(sf, store=store)
        ps.record_gate("G1_SCREENING", "PASS", {"ic": 0.15}, "good")
        ps2 = AlphaPipelineState(sf, store=store)
        assert ps2.get("gates")["G1_SCREENING"]["verdict"] == "PASS"


# ===========================================================================
# CLI
# ===========================================================================

class TestCLI:
    def test_status_empty_db(self, tmp_path, capsys):
        from data.state import _cli_status
        db = tmp_path / "empty.db"
        _cli_status(db)
        out = capsys.readouterr().out
        assert "no agents found" in out

    def test_status_with_data(self, tmp_path, capsys):
        from data.state import _cli_status
        store = StateStore(tmp_path / "test.db")
        store.save_state("test_agent", {"phase": "IDLE", "cycle_count": 5})
        store.close()

        _cli_status(tmp_path / "test.db")
        out = capsys.readouterr().out
        assert "test_agent" in out
        assert "IDLE" in out

    def test_migrate_idempotent(self, tmp_path, capsys):
        from data.state import _cli_migrate
        # Create a fake agent dir with state
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "agent_state.json").write_text(json.dumps({
            "phase": "IDLE", "cycle_count": 2, "history": [],
        }))

        _cli_migrate(tmp_path / "test.db", tmp_path)
        out1 = capsys.readouterr().out
        assert "migrated" in out1

        # Second run should say "already done"
        _cli_migrate(tmp_path / "test.db", tmp_path)
        out2 = capsys.readouterr().out
        assert "already done" in out2


# ===========================================================================
# Cross-agent queries
# ===========================================================================

class TestCrossAgentQueries:
    def test_all_registries_returns_all_agents(self, tmp_path):
        store = StateStore(tmp_path / "test.db")
        store.append_signal("microstructure", {
            "name": "s1", "features": ["f1"], "status": "validated",
        })
        store.append_signal("medium_freq", {
            "name": "s2", "features": ["f2"], "status": "validated",
        })
        store.append_signal("macro", {
            "name": "s3", "features": ["f3"], "status": "validated",
        })

        all_regs = store.all_registries()
        assert len(all_regs) == 3
        assert all_regs["microstructure"][0]["name"] == "s1"
        assert all_regs["medium_freq"][0]["name"] == "s2"
        assert all_regs["macro"][0]["name"] == "s3"

    def test_all_states_returns_all_agents(self, tmp_path):
        store = StateStore(tmp_path / "test.db")
        store.save_state("microstructure", {"phase": "IDLE", "cycle_count": 10})
        store.save_state("meta", {"phase": "RUNNING", "cycle_count": 2})

        all_st = store.all_states()
        assert all_st["microstructure"]["phase"] == "IDLE"
        assert all_st["meta"]["phase"] == "RUNNING"
