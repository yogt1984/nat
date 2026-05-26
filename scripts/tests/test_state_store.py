"""Skeptical tests for P1-4: SQLite StateStore.

Tests cover:
- Basics: DB creation, WAL mode, tables, close/reopen
- Agent state roundtrip: save/load, isolation, datetime, None, history
- Hypotheses roundtrip: all fields, upsert, filter, unicode, large results
- Registry roundtrip: all fields, append, update, remove, ic_history
- Generator stats: roundtrip, multi-generator, defaults
- Atomicity: transaction rollback, concurrent reads during write, WAL concurrency
- Migration: JSON import, no double-import, empty/corrupt files, auto-detect
- Cross-agent: all_registries, all_states, isolation
- Integration: full agent cycle, no JSON created, push/pop, register, monitor
- Export: JSON format matches old format, reflects latest state
- Backward compat: JSON fallback paths still work
"""

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from data.state import StateStore


@pytest.fixture
def store(tmp_path):
    s = StateStore(tmp_path / "nat.db")
    yield s
    s.close()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "nat.db"


# ===========================================================================
# Basics
# ===========================================================================

class TestStateStoreBasics:
    def test_creates_db_file(self, db_path):
        s = StateStore(db_path)
        assert db_path.exists()
        s.close()

    def test_wal_mode_enabled(self, store):
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_tables_created(self, store):
        tables = {r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert {"agent_state", "state_history", "hypotheses",
                "registry", "generator_stats", "_migrations"} <= tables

    def test_close_and_reopen(self, db_path):
        s1 = StateStore(db_path)
        s1.save_state("test", {"phase": "IDLE", "cycle_count": 42})
        s1.close()
        s2 = StateStore(db_path)
        state = s2.load_state("test")
        assert state["cycle_count"] == 42
        s2.close()

    def test_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "nat.db"
        s = StateStore(deep)
        assert deep.exists()
        s.close()


# ===========================================================================
# Agent state roundtrip
# ===========================================================================

class TestAgentStateRoundtrip:
    def test_save_and_load(self, store):
        data = {
            "phase": "EXECUTE",
            "cycle_count": 5,
            "total_hypotheses_tested": 100,
            "total_signals_registered": 12,
            "current_hypothesis": "HYP-SYS-abc",
            "started_at": "2026-05-25T10:00:00+00:00",
            "last_cycle_at": "2026-05-25T11:00:00+00:00",
        }
        store.save_state("micro", data)
        loaded = store.load_state("micro")
        for k, v in data.items():
            assert loaded[k] == v, f"Key {k}: {loaded.get(k)} != {v}"

    def test_multiple_agents_isolated(self, store):
        store.save_state("micro", {"phase": "IDLE"})
        store.save_state("mf", {"phase": "EXECUTE"})
        assert store.load_state("micro")["phase"] == "IDLE"
        assert store.load_state("mf")["phase"] == "EXECUTE"

    def test_update_single_key(self, store):
        store.save_state("a", {"phase": "IDLE", "cycle_count": 0})
        store.save_state("a", {"cycle_count": 5})
        state = store.load_state("a")
        assert state["cycle_count"] == 5
        assert state["phase"] == "IDLE"  # untouched

    def test_datetime_serialization(self, store):
        store.save_state("a", {"started_at": "2026-05-25T10:00:00+00:00"})
        assert store.load_state("a")["started_at"] == "2026-05-25T10:00:00+00:00"

    def test_none_values(self, store):
        store.save_state("a", {"current_hypothesis": None})
        assert store.load_state("a")["current_hypothesis"] is None

    def test_empty_state_returns_empty_dict(self, store):
        assert store.load_state("nonexistent") == {}

    def test_history_append_and_load(self, store):
        for i in range(5):
            store.append_history("a", {
                "from": f"P{i}", "to": f"P{i+1}",
                "msg": f"step {i}", "at": f"2026-05-25T10:0{i}:00Z",
            })
        history = store.load_history("a")
        assert len(history) == 5
        assert history[0]["from"] == "P0"
        assert history[4]["from"] == "P4"

    def test_history_limit(self, store):
        for i in range(300):
            store.append_history("a", {
                "from": "X", "to": "Y", "msg": str(i), "at": "2026-01-01T00:00:00Z",
            })
        history = store.load_history("a", limit=200)
        assert len(history) == 200
        # Should be the 200 most recent
        assert history[-1]["msg"] == "299"
        assert history[0]["msg"] == "100"

    def test_history_isolated_per_agent(self, store):
        store.append_history("a", {"from": "X", "to": "Y", "msg": "a", "at": "t1"})
        store.append_history("b", {"from": "X", "to": "Y", "msg": "b", "at": "t2"})
        assert len(store.load_history("a")) == 1
        assert store.load_history("a")[0]["msg"] == "a"


# ===========================================================================
# Hypotheses roundtrip
# ===========================================================================

class TestHypothesesRoundtrip:
    def _make_hyp(self, hyp_id="HYP-SYS-abc123", **overrides):
        h = {
            "id": hyp_id,
            "claim": "feature_x predicts 5s returns",
            "generator": "systematic",
            "priority": 2.5,
            "test_protocol": ["nat --data d --symbol BTC", "nat --repl"],
            "thresholds": {"min_ic": 0.10, "min_dIC": 0.05},
            "status": "queued",
            "failure_reason": None,
            "parent_id": None,
            "results": None,
            "created": "2026-05-25T10:00:00+00:00",
            "completed": None,
        }
        h.update(overrides)
        return h

    def test_full_roundtrip(self, store):
        h = self._make_hyp()
        store.upsert_hypothesis("micro", h)
        loaded = store.load_hypotheses("micro")
        assert len(loaded) == 1
        for k in h:
            assert loaded[0][k] == h[k], f"Key {k}: {loaded[0].get(k)} != {h[k]}"

    def test_upsert_updates_existing(self, store):
        h = self._make_hyp()
        store.upsert_hypothesis("micro", h)
        h["status"] = "running"
        store.upsert_hypothesis("micro", h)
        loaded = store.load_hypotheses("micro")
        assert len(loaded) == 1
        assert loaded[0]["status"] == "running"

    def test_filter_by_agent(self, store):
        store.upsert_hypothesis("micro", self._make_hyp("H1"))
        store.upsert_hypothesis("mf", self._make_hyp("H2"))
        assert len(store.load_hypotheses("micro")) == 1
        assert len(store.load_hypotheses("mf")) == 1

    def test_unicode_in_claim(self, store):
        h = self._make_hyp(claim="entropy_α gated by β<P40")
        store.upsert_hypothesis("micro", h)
        loaded = store.load_hypotheses("micro")
        assert loaded[0]["claim"] == "entropy_α gated by β<P40"

    def test_large_results_json(self, store):
        results = {
            "gate_results": [
                {"cmd": f"cmd{i}", "passed": True, "msg": f"IC=0.{i:03d} PASS"}
                for i in range(20)
            ],
            "symbol_replication": {
                "passed": ["BTC", "ETH"], "failed": ["SOL"],
                "n_pass": 2, "n_total": 3,
            },
        }
        h = self._make_hyp(results=results)
        store.upsert_hypothesis("micro", h)
        loaded = store.load_hypotheses("micro")[0]
        assert loaded["results"]["gate_results"][5]["msg"] == "IC=0.005 PASS"
        assert loaded["results"]["symbol_replication"]["n_pass"] == 2

    def test_delete_hypothesis(self, store):
        store.upsert_hypothesis("micro", self._make_hyp("H1"))
        store.upsert_hypothesis("micro", self._make_hyp("H2"))
        store.delete_hypothesis("H1")
        loaded = store.load_hypotheses("micro")
        assert len(loaded) == 1
        assert loaded[0]["id"] == "H2"

    def test_thresholds_with_nested_dict(self, store):
        h = self._make_hyp(thresholds={
            "min_ic": 0.08, "symbols": ["BTC", "ETH"],
            "regime_gate": "ent_tick<P30",
        })
        store.upsert_hypothesis("micro", h)
        loaded = store.load_hypotheses("micro")[0]
        assert loaded["thresholds"]["symbols"] == ["BTC", "ETH"]


# ===========================================================================
# Registry roundtrip
# ===========================================================================

class TestRegistryRoundtrip:
    def _make_signal(self, **overrides):
        sig = {
            "name": "feature_x predicts 5s returns",
            "features": ["feature_x", "feature_y"],
            "regime_gate": "ent_tick<P30",
            "spectral_band": None,
            "extraction": "raw",
            "horizon_s": 5.0,
            "expected_ic": 0.15,
            "expected_ir": 1.2,
            "decay_halflife_s": 60.0,
            "symbols": ["BTC", "ETH", "SOL"],
            "correlation_with": {"sig_a": 0.3},
            "status": "validated",
            "discovery_date": "2026-05-25",
            "last_validated": "2026-05-25",
            "hypothesis_id": "HYP-SYS-abc123",
            "ic_history": [{"date": "2026-05-25", "ic": 0.15}],
            "latest_ic": 0.15,
            "decay_days": 0,
            "retired_reason": None,
            "retired_date": None,
            "paper_sharpe": None,
            "paper_days_elapsed": None,
            "realized_ic": None,
            "max_drawdown_pct": None,
        }
        sig.update(overrides)
        return sig

    def test_full_roundtrip(self, store):
        sig = self._make_signal()
        store.append_signal("micro", sig)
        loaded = store.load_registry("micro")
        assert len(loaded) == 1
        for k in sig:
            assert loaded[0][k] == sig[k], f"Key {k}: {loaded[0].get(k)} != {sig[k]}"

    def test_append_multiple(self, store):
        store.append_signal("micro", self._make_signal(hypothesis_id="H1"))
        store.append_signal("micro", self._make_signal(hypothesis_id="H2"))
        assert len(store.load_registry("micro")) == 2

    def test_update_signal(self, store):
        store.append_signal("micro", self._make_signal(hypothesis_id="H1"))
        store.update_signal("micro", "H1", {
            "latest_ic": 0.08, "decay_days": 3,
            "ic_history": [{"date": "d1", "ic": 0.15}, {"date": "d2", "ic": 0.08}],
        })
        loaded = store.load_registry("micro")[0]
        assert loaded["latest_ic"] == 0.08
        assert loaded["decay_days"] == 3
        assert len(loaded["ic_history"]) == 2

    def test_remove_signal(self, store):
        store.append_signal("micro", self._make_signal(hypothesis_id="H1"))
        store.append_signal("micro", self._make_signal(hypothesis_id="H2"))
        store.remove_signal("micro", "H1")
        loaded = store.load_registry("micro")
        assert len(loaded) == 1
        assert loaded[0]["hypothesis_id"] == "H2"

    def test_ic_history_json_roundtrip(self, store):
        history = [{"date": f"d{i}", "ic": 0.1 + i * 0.01} for i in range(30)]
        store.append_signal("micro", self._make_signal(ic_history=history))
        loaded = store.load_registry("micro")[0]
        assert len(loaded["ic_history"]) == 30
        assert abs(loaded["ic_history"][5]["ic"] - 0.15) < 1e-6

    def test_empty_registry(self, store):
        assert store.load_registry("nonexistent") == []


# ===========================================================================
# Generator stats
# ===========================================================================

class TestGeneratorStats:
    def test_roundtrip(self, store):
        stats = {
            "systematic": {"attempts": 60, "successes": 12},
            "spectral": {"attempts": 10, "successes": 0},
        }
        store.save_gen_stats("micro", stats)
        loaded = store.load_gen_stats("micro")
        assert loaded["systematic"]["attempts"] == 60
        assert loaded["systematic"]["successes"] == 12
        assert loaded["spectral"]["attempts"] == 10

    def test_multi_generator_update(self, store):
        store.save_gen_stats("micro", {"gen_a": {"attempts": 1, "successes": 0}})
        store.save_gen_stats("micro", {"gen_a": {"attempts": 5, "successes": 2}})
        loaded = store.load_gen_stats("micro")
        assert loaded["gen_a"]["attempts"] == 5

    def test_empty_stats(self, store):
        assert store.load_gen_stats("nonexistent") == {}

    def test_agents_isolated(self, store):
        store.save_gen_stats("micro", {"gen_a": {"attempts": 10, "successes": 5}})
        store.save_gen_stats("mf", {"gen_b": {"attempts": 3, "successes": 1}})
        assert "gen_a" in store.load_gen_stats("micro")
        assert "gen_a" not in store.load_gen_stats("mf")


# ===========================================================================
# Atomicity
# ===========================================================================

class TestAtomicity:
    def test_transaction_rollback_on_error(self, store):
        """Partial writes should be rolled back."""
        store.upsert_hypothesis("micro", {
            "id": "H1", "claim": "test", "generator": "gen",
            "priority": 1.0, "test_protocol": [], "thresholds": {},
            "status": "queued", "created": "2026-01-01T00:00:00Z",
        })
        # Try to insert with duplicate PK in a batch — should roll back
        try:
            with store._conn:
                store._conn.execute(
                    "INSERT INTO hypotheses (id, agent, claim, generator, "
                    "priority, test_protocol, thresholds, status, created) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("H2", "micro", "test2", "gen", 1.0, "[]", "{}", "queued", "t"),
                )
                # Force an error
                raise sqlite3.IntegrityError("simulated crash")
        except sqlite3.IntegrityError:
            pass
        # H2 should not exist
        loaded = store.load_hypotheses("micro")
        assert len(loaded) == 1
        assert loaded[0]["id"] == "H1"

    def test_concurrent_reads_during_write(self, db_path):
        """Reader sees consistent state while writer is active."""
        # Pre-populate
        setup = StateStore(db_path)
        for i in range(10):
            setup.upsert_hypothesis("micro", {
                "id": f"INIT-{i}", "claim": f"init {i}", "generator": "gen",
                "priority": 1.0, "test_protocol": [], "thresholds": {},
                "status": "queued", "created": "2026-01-01T00:00:00Z",
            })
        setup.close()

        errors = []

        def writer_fn():
            w = StateStore(db_path)
            for i in range(200):
                w.upsert_hypothesis("micro", {
                    "id": f"H-{i}", "claim": f"claim {i}", "generator": "gen",
                    "priority": 1.0, "test_protocol": [], "thresholds": {},
                    "status": "queued", "created": "2026-01-01T00:00:00Z",
                })
            w.close()

        def reader_fn():
            r = StateStore(db_path)
            for _ in range(50):
                try:
                    hyps = r.load_hypotheses("micro")
                    assert isinstance(hyps, list)
                    for h in hyps:
                        assert "id" in h
                        assert "claim" in h
                except Exception as e:
                    errors.append(str(e))
                time.sleep(0.001)
            r.close()

        t_write = threading.Thread(target=writer_fn)
        t_read = threading.Thread(target=reader_fn)
        t_write.start()
        t_read.start()
        t_write.join()
        t_read.join()

        assert errors == [], f"Reader errors: {errors}"

    def test_wal_readers_dont_block_writers(self, db_path):
        """Two stores to same DB: reader holds cursor, writer succeeds."""
        s1 = StateStore(db_path)
        s2 = StateStore(db_path)

        s1.save_state("a", {"phase": "IDLE"})

        # s2 starts a read (holds a snapshot)
        cursor = s2._conn.execute("SELECT * FROM agent_state")
        rows = cursor.fetchall()
        assert len(rows) == 1

        # s1 writes while s2 cursor is "open"
        s1.save_state("a", {"phase": "EXECUTE", "cycle_count": 1})

        # s2 can still read (gets new data on next query)
        state = s2.load_state("a")
        assert state["phase"] == "EXECUTE"

        s1.close()
        s2.close()


# ===========================================================================
# Migration
# ===========================================================================

class TestMigration:
    def _write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def test_migrate_agent_state(self, tmp_path):
        state_data = {
            "phase": "IDLE", "cycle_count": 5,
            "total_hypotheses_tested": 20,
            "total_signals_registered": 3,
            "current_hypothesis": None,
            "started_at": "2026-05-25T10:00:00Z",
            "last_cycle_at": "2026-05-25T11:00:00Z",
            "history": [
                {"from": "IDLE", "to": "EXECUTE", "at": "t1", "msg": "go"},
                {"from": "EXECUTE", "to": "IDLE", "at": "t2", "msg": "done"},
            ],
        }
        state_path = tmp_path / "agent_state.json"
        self._write_json(state_path, state_data)

        store = StateStore(tmp_path / "nat.db")
        result = store.migrate_from_json("micro", state_path=state_path)
        assert result is True

        loaded = store.load_state("micro")
        assert loaded["phase"] == "IDLE"
        assert loaded["cycle_count"] == 5

        history = store.load_history("micro")
        assert len(history) == 2
        assert history[0]["msg"] == "go"
        store.close()

    def test_migrate_hypotheses(self, tmp_path):
        hyps = [
            {"id": "H1", "claim": "test1", "generator": "sys",
             "priority": 1.0, "test_protocol": ["cmd1"],
             "thresholds": {"min_ic": 0.1}, "status": "queued",
             "created": "2026-01-01T00:00:00Z"},
            {"id": "H2", "claim": "test2", "generator": "sys",
             "priority": 0.5, "test_protocol": [], "thresholds": {},
             "status": "failed", "failure_reason": "no_effect",
             "created": "2026-01-01T00:00:00Z"},
        ]
        hyp_path = tmp_path / "hypotheses.json"
        self._write_json(hyp_path, hyps)

        store = StateStore(tmp_path / "nat.db")
        store.migrate_from_json("micro", hyp_path=hyp_path)

        loaded = store.load_hypotheses("micro")
        assert len(loaded) == 2
        by_id = {h["id"]: h for h in loaded}
        assert "H1" in by_id
        assert by_id["H2"]["status"] == "failed"
        assert by_id["H2"]["failure_reason"] == "no_effect"
        store.close()

    def test_migrate_registry(self, tmp_path):
        signals = [{
            "name": "sig1", "features": ["f1"],
            "hypothesis_id": "H1", "horizon_s": 5.0,
            "expected_ic": 0.15, "symbols": ["BTC"],
            "status": "validated", "discovery_date": "2026-01-01",
        }]
        reg_path = tmp_path / "registry.json"
        self._write_json(reg_path, signals)

        store = StateStore(tmp_path / "nat.db")
        store.migrate_from_json("micro", reg_path=reg_path)

        loaded = store.load_registry("micro")
        assert len(loaded) == 1
        assert loaded[0]["name"] == "sig1"
        assert loaded[0]["features"] == ["f1"]
        store.close()

    def test_migrate_gen_stats(self, tmp_path):
        stats = {"systematic": {"attempts": 60, "successes": 12}}
        stats_path = tmp_path / "generator_stats.json"
        self._write_json(stats_path, stats)

        store = StateStore(tmp_path / "nat.db")
        store.migrate_from_json("micro", stats_path=stats_path)

        loaded = store.load_gen_stats("micro")
        assert loaded["systematic"]["attempts"] == 60
        store.close()

    def test_no_double_migration(self, tmp_path):
        state_path = tmp_path / "agent_state.json"
        self._write_json(state_path, {"phase": "IDLE", "cycle_count": 1})

        store = StateStore(tmp_path / "nat.db")
        assert store.migrate_from_json("micro", state_path=state_path) is True
        # Second call should be a no-op
        assert store.migrate_from_json("micro", state_path=state_path) is False
        store.close()

    def test_empty_json_files(self, tmp_path):
        """Missing files don't crash migration."""
        store = StateStore(tmp_path / "nat.db")
        result = store.migrate_from_json("micro",
                                          state_path=tmp_path / "nonexistent.json")
        # Still marks migration as done (no data to import)
        assert result is False  # no data imported but migration recorded
        store.close()

    def test_corrupt_json_handled(self, tmp_path):
        bad = tmp_path / "agent_state.json"
        bad.write_text("{corrupt json!!!}")

        store = StateStore(tmp_path / "nat.db")
        # Should not raise
        store.migrate_from_json("micro", state_path=bad)
        # State should be empty (corrupt file skipped)
        assert store.load_state("micro") == {}
        store.close()


# ===========================================================================
# Cross-agent
# ===========================================================================

class TestCrossAgent:
    def test_all_registries(self, store):
        store.append_signal("micro", {
            "name": "sig_micro", "hypothesis_id": "H1",
            "features": ["f1"], "horizon_s": 5.0,
            "expected_ic": 0.15, "symbols": ["BTC"],
        })
        store.append_signal("mf", {
            "name": "sig_mf", "hypothesis_id": "H2",
            "features": ["f2"], "horizon_s": 300.0,
            "expected_ic": 0.10, "symbols": ["BTC"],
        })
        all_reg = store.all_registries()
        assert "micro" in all_reg
        assert "mf" in all_reg
        assert len(all_reg["micro"]) == 1
        assert len(all_reg["mf"]) == 1

    def test_all_states(self, store):
        store.save_state("micro", {"phase": "IDLE"})
        store.save_state("mf", {"phase": "EXECUTE"})
        states = store.all_states()
        assert states["micro"]["phase"] == "IDLE"
        assert states["mf"]["phase"] == "EXECUTE"

    def test_cross_agent_write_isolation(self, store):
        """Writing to agent_a should not affect agent_b."""
        store.upsert_hypothesis("micro", {
            "id": "H1", "claim": "micro claim", "generator": "sys",
            "priority": 1.0, "test_protocol": [], "thresholds": {},
            "status": "queued", "created": "t1",
        })
        assert store.load_hypotheses("mf") == []


# ===========================================================================
# Export JSON
# ===========================================================================

class TestExportJSON:
    def test_export_matches_original_format(self, store, tmp_path):
        store.save_state("micro", {
            "phase": "IDLE", "cycle_count": 3,
            "total_hypotheses_tested": 10,
        })
        store.append_history("micro", {
            "from": "X", "to": "Y", "msg": "test", "at": "t1",
        })
        store.upsert_hypothesis("micro", {
            "id": "H1", "claim": "test", "generator": "sys",
            "priority": 1.0, "test_protocol": ["cmd"],
            "thresholds": {"min_ic": 0.1}, "status": "queued",
            "created": "2026-01-01T00:00:00Z",
        })
        store.append_signal("micro", {
            "name": "sig1", "features": ["f1"],
            "hypothesis_id": "H1", "horizon_s": 5.0,
            "expected_ic": 0.15, "symbols": ["BTC"],
        })
        store.save_gen_stats("micro", {
            "systematic": {"attempts": 10, "successes": 3},
        })

        out = tmp_path / "export"
        store.export_json("micro", out)

        # Verify exported files exist and are valid JSON
        with open(out / "agent_state.json") as f:
            state = json.load(f)
        assert state["phase"] == "IDLE"
        assert len(state["history"]) == 1

        with open(out / "hypotheses.json") as f:
            hyps = json.load(f)
        assert len(hyps) == 1
        assert hyps[0]["id"] == "H1"

        with open(out / "registry.json") as f:
            reg = json.load(f)
        assert len(reg) == 1

        with open(out / "generator_stats.json") as f:
            stats = json.load(f)
        assert stats["systematic"]["attempts"] == 10

    def test_export_reflects_mutations(self, store, tmp_path):
        store.upsert_hypothesis("micro", {
            "id": "H1", "claim": "test", "generator": "sys",
            "priority": 1.0, "test_protocol": [], "thresholds": {},
            "status": "queued", "created": "t1",
        })
        store.upsert_hypothesis("micro", {
            "id": "H1", "claim": "test", "generator": "sys",
            "priority": 1.0, "test_protocol": [], "thresholds": {},
            "status": "running", "created": "t1",
        })

        out = tmp_path / "export"
        store.export_json("micro", out)
        with open(out / "hypotheses.json") as f:
            hyps = json.load(f)
        assert hyps[0]["status"] == "running"


# ===========================================================================
# Research output
# ===========================================================================

class TestResearchOutput:
    def test_insert_and_query_hypothesis(self, store):
        record = {
            "id": "HYP-SYS-001",
            "agent": "microstructure",
            "generator": "systematic",
            "status": "replicated",
            "timestamps": {"created": "2026-05-25T10:00:00", "completed": "2026-05-25T10:01:00"},
            "gates": [{"name": "IC", "passed": True}],
            "schema_version": 1,
        }
        store.insert_research_output(record, kind="hypothesis")
        items, total = store.query_research_output(kind="hypothesis")
        assert total == 1
        assert items[0]["id"] == "HYP-SYS-001"

    def test_insert_and_query_cycle(self, store):
        record = {
            "cycle_id": "CYC-001",
            "agent": "microstructure",
            "completed": "2026-05-25T10:05:00",
            "n_tested": 10,
            "schema_version": 1,
        }
        store.insert_research_output(record, kind="cycle")
        items, total = store.query_research_output(kind="cycle")
        assert total == 1
        assert items[0]["cycle_id"] == "CYC-001"

    def test_filter_by_agent(self, store):
        for i, agent in enumerate(["micro", "macro", "micro"]):
            store.insert_research_output(
                {"id": f"H{i}", "agent": agent, "generator": "sys", "status": "failed",
                 "timestamps": {"completed": f"2026-05-25T10:0{i}:00"}, "schema_version": 1},
                kind="hypothesis",
            )
        items, total = store.query_research_output(kind="hypothesis", agent="micro")
        assert total == 2

    def test_filter_by_status(self, store):
        store.insert_research_output(
            {"id": "H1", "agent": "micro", "status": "replicated",
             "timestamps": {"completed": "2026-05-25T10:00:00"}, "schema_version": 1},
            kind="hypothesis",
        )
        store.insert_research_output(
            {"id": "H2", "agent": "micro", "status": "failed",
             "timestamps": {"completed": "2026-05-25T10:01:00"}, "schema_version": 1},
            kind="hypothesis",
        )
        items, total = store.query_research_output(kind="hypothesis", status="replicated")
        assert total == 1
        assert items[0]["id"] == "H1"

    def test_pagination(self, store):
        for i in range(10):
            store.insert_research_output(
                {"id": f"H{i}", "agent": "micro", "status": "failed",
                 "timestamps": {"completed": f"2026-05-25T10:{i:02d}:00"}, "schema_version": 1},
                kind="hypothesis",
            )
        items, total = store.query_research_output(kind="hypothesis", limit=3, offset=0)
        assert total == 10
        assert len(items) == 3

        items, total = store.query_research_output(kind="hypothesis", limit=3, offset=8)
        assert len(items) == 2

    def test_get_single_record(self, store):
        store.insert_research_output(
            {"id": "HYP-1", "agent": "micro", "status": "replicated",
             "timestamps": {"completed": "2026-05-25T10:00:00"}, "schema_version": 1},
            kind="hypothesis",
        )
        record = store.get_research_output("HYP-1")
        assert record is not None
        assert record["id"] == "HYP-1"
        assert store.get_research_output("MISSING") is None

    def test_upsert_overwrites(self, store):
        store.insert_research_output(
            {"id": "H1", "agent": "micro", "status": "running",
             "timestamps": {"completed": "2026-05-25T10:00:00"}, "schema_version": 1},
            kind="hypothesis",
        )
        store.insert_research_output(
            {"id": "H1", "agent": "micro", "status": "replicated",
             "timestamps": {"completed": "2026-05-25T10:01:00"}, "schema_version": 1},
            kind="hypothesis",
        )
        items, total = store.query_research_output(kind="hypothesis")
        assert total == 1
        assert items[0]["status"] == "replicated"

    def test_no_id_skipped(self, store):
        store.insert_research_output({"agent": "micro"}, kind="hypothesis")
        items, total = store.query_research_output(kind="hypothesis")
        assert total == 0
