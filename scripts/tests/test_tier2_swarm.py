"""
Tier 2 swarm unit tests — config generator, evaluator, orchestrator.

These tests use synthetic data and temp directories. No Docker or live
Parquet data required. Integration tests with real data are in
test_tier2_swarm_integration.py.
"""

import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import toml

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from swarm.config_generator import ConfigGenerator, config_hash, _linspace
from swarm.evaluator import Evaluator, SCHEMA_SQL


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def base_config(tmp_path):
    """Write a minimal algorithms.toml for testing."""
    cfg = {
        "evaluation": {"horizons": [1, 5], "regime_feature": "ent_book_shape"},
        "jump_detector": {"window": 100, "significance": 3.0},
        "optimal_entry": {"theta": 0.1, "sigma_process": 0.01},
        "ensemble": {
            "algorithms": ["jump_detector", "optimal_entry"],
            "method": "equal_weight",
            "ic_lookback": 5000,
        },
    }
    path = tmp_path / "base.toml"
    with open(path, "w") as f:
        toml.dump(cfg, f)
    return str(path)


@pytest.fixture
def ranges_config(tmp_path):
    """Write a minimal swarm_ranges.toml for testing."""
    cfg = {
        "jump_detector": {
            "window": {"min": 50, "max": 500, "type": "int"},
            "significance": {"min": 2.0, "max": 5.0, "type": "float"},
        },
        "optimal_entry": {
            "sigma_process": {"min": 0.001, "max": 0.1, "type": "float", "log": True},
        },
        "ensemble": {
            "method": {"choices": ["equal_weight", "ic_weight"], "type": "categorical"},
        },
        "feature_selection": {
            "whale_flow": {"default": True, "type": "bool"},
        },
    }
    path = tmp_path / "ranges.toml"
    with open(path, "w") as f:
        toml.dump(cfg, f)
    return str(path)


@pytest.fixture
def real_configs():
    """Return paths to real project config files (skip if missing)."""
    base = ROOT / "config" / "algorithms.toml"
    ranges = ROOT / "config" / "swarm_ranges.toml"
    if not base.exists() or not ranges.exists():
        pytest.skip("Real config files not found")
    return str(base), str(ranges)


@pytest.fixture
def synthetic_df():
    """Generate a synthetic DataFrame matching ingestor output."""
    rng = np.random.default_rng(42)
    n = 5000
    mid = 50000 * np.exp(np.cumsum(rng.normal(0, 0.0001, n)))
    return pd.DataFrame({
        "timestamp_ns": np.arange(n) * int(1e8),  # 100ms ticks
        "symbol": "BTC",
        "raw_midprice": mid,
        "raw_spread": np.abs(rng.normal(0.5, 0.2, n)),
        "imbalance_qty_l1": rng.normal(0, 0.3, n),
        "ent_book_shape": rng.uniform(0, 1, n),
        "vol_returns_1m": np.abs(rng.normal(0.001, 0.0005, n)),
        "flow_count_1m": rng.poisson(5, n).astype(float),
        "depth_5_bid": rng.exponential(100, n),
        "depth_5_ask": rng.exponential(100, n),
    })


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "results.db")


# ── ConfigGenerator ─────────────────────────────────────────────────────────


class TestConfigGenerator:
    def test_generate_random_count(self, base_config, ranges_config):
        gen = ConfigGenerator(base_config, ranges_config)
        configs = gen.generate_random(10, seed=42)
        assert len(configs) == 10

    def test_generate_random_reproducible(self, base_config, ranges_config):
        gen = ConfigGenerator(base_config, ranges_config)
        a = gen.generate_random(5, seed=42)
        b = gen.generate_random(5, seed=42)
        for ca, cb in zip(a, b):
            assert ca == cb, "Same seed should produce identical configs"

    def test_generate_random_different_seeds(self, base_config, ranges_config):
        gen = ConfigGenerator(base_config, ranges_config)
        a = gen.generate_random(5, seed=42)
        b = gen.generate_random(5, seed=99)
        assert a != b, "Different seeds should produce different configs"

    def test_parameter_ranges_respected(self, base_config, ranges_config):
        gen = ConfigGenerator(base_config, ranges_config)
        configs = gen.generate_random(100, seed=42)
        for cfg in configs:
            w = cfg["jump_detector"]["window"]
            assert 50 <= w <= 500, f"window {w} out of range"
            assert isinstance(w, int), f"window should be int, got {type(w)}"

            s = cfg["jump_detector"]["significance"]
            assert 2.0 <= s <= 5.0, f"significance {s} out of range"

            sp = cfg["optimal_entry"]["sigma_process"]
            assert 0.001 <= sp <= 0.1, f"sigma_process {sp} out of range"

            m = cfg["ensemble"]["method"]
            assert m in ("equal_weight", "ic_weight"), f"method '{m}' invalid"

    def test_log_scale_sampling(self, base_config, ranges_config):
        gen = ConfigGenerator(base_config, ranges_config)
        configs = gen.generate_random(1000, seed=42)
        sigma_vals = [c["optimal_entry"]["sigma_process"] for c in configs]
        # Log-scale: most values should be in lower half of [0.001, 0.1]
        median = np.median(sigma_vals)
        assert median < 0.05, \
            f"Log-scale median {median:.4f} should be < 0.05 (biased toward lower values)"

    def test_bool_sampling(self, base_config, ranges_config):
        gen = ConfigGenerator(base_config, ranges_config)
        configs = gen.generate_random(200, seed=42)
        wf_vals = [c.get("feature_selection", {}).get("whale_flow") for c in configs]
        # Should have a mix of True/False
        assert any(v is True for v in wf_vals)
        assert any(v is False for v in wf_vals)

    def test_base_config_preserved(self, base_config, ranges_config):
        """Parameters NOT in ranges should keep their base values."""
        gen = ConfigGenerator(base_config, ranges_config)
        configs = gen.generate_random(5, seed=42)
        for cfg in configs:
            # theta is not in ranges, should be base value
            assert cfg["optimal_entry"]["theta"] == 0.1

    def test_write_configs(self, base_config, ranges_config, tmp_path):
        gen = ConfigGenerator(base_config, ranges_config)
        configs = gen.generate_random(3, seed=42)
        out_dir = str(tmp_path / "configs")
        paths = gen.write_configs(configs, out_dir)

        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            cfg = toml.load(str(p))
            assert "_meta" in cfg
            assert "config_hash" in cfg["_meta"]

    def test_write_configs_unique_hashes(self, base_config, ranges_config, tmp_path):
        gen = ConfigGenerator(base_config, ranges_config)
        configs = gen.generate_random(10, seed=42)
        out_dir = str(tmp_path / "configs")
        gen.write_configs(configs, out_dir)

        hashes = set()
        for p in Path(out_dir).glob("*.toml"):
            cfg = toml.load(str(p))
            hashes.add(cfg["_meta"]["config_hash"])
        assert len(hashes) == 10, "All configs should have unique hashes"

    def test_generate_grid(self, base_config, ranges_config):
        gen = ConfigGenerator(base_config, ranges_config)
        configs = gen.generate_grid(resolution=2)
        # 2 (window) * 2 (significance) * 2 (sigma) * 2 (method) * 2 (bool) = 32
        assert len(configs) == 32

    def test_generate_from_optuna_stub(self, base_config, ranges_config):
        """Test Optuna integration with a mock trial."""
        gen = ConfigGenerator(base_config, ranges_config)
        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value = 200
        mock_trial.suggest_float.return_value = 3.5
        mock_trial.suggest_categorical.return_value = "ic_weight"

        cfg = gen.generate_from_optuna(mock_trial)
        assert cfg["jump_detector"]["window"] == 200
        assert cfg["ensemble"]["method"] == "ic_weight"


class TestConfigGeneratorRealFiles:
    def test_real_configs_parse(self, real_configs):
        base, ranges = real_configs
        gen = ConfigGenerator(base, ranges)
        configs = gen.generate_random(5, seed=42)
        assert len(configs) == 5
        for cfg in configs:
            assert "ensemble" in cfg
            assert "jump_detector" in cfg


class TestConfigHash:
    def test_deterministic(self):
        cfg = {"a": 1, "b": {"c": 2}}
        assert config_hash(cfg) == config_hash(cfg)

    def test_ignores_meta(self):
        cfg1 = {"a": 1, "_meta": {"hash": "x"}}
        cfg2 = {"a": 1, "_meta": {"hash": "y"}}
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_different_configs(self):
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_hash_length(self):
        h = config_hash({"x": 42})
        assert len(h) == 16  # truncated to 16 hex chars


class TestLinspace:
    def test_single_point(self):
        assert _linspace(0, 10, 1) == [5.0]

    def test_two_points(self):
        pts = _linspace(0, 10, 2)
        assert pts == [0.0, 10.0]

    def test_five_points(self):
        pts = _linspace(0, 1, 5)
        assert len(pts) == 5
        assert abs(pts[0] - 0.0) < 1e-10
        assert abs(pts[-1] - 1.0) < 1e-10
        # Evenly spaced
        diffs = [pts[i+1] - pts[i] for i in range(4)]
        assert all(abs(d - 0.25) < 1e-10 for d in diffs)


# ── Evaluator ───────────────────────────────────────────────────────────────


class TestEvaluatorDB:
    def test_db_creation(self, db_path, tmp_path):
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text('[ensemble]\nalgorithms = ["jump_detector"]\n')
        ev = Evaluator(str(cfg_path), "data/features", db_path)
        assert Path(db_path).exists()

        with sqlite3.connect(db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]
            assert "trials" in table_names

    def test_schema_columns(self, db_path, tmp_path):
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text('[ensemble]\nalgorithms = []\n')
        ev = Evaluator(str(cfg_path), "data/features", db_path)

        with sqlite3.connect(db_path) as conn:
            info = conn.execute("PRAGMA table_info(trials)").fetchall()
            col_names = {row[1] for row in info}
            expected = {"trial_id", "config_hash", "config_json", "symbol",
                        "eval_hours", "sharpe", "mean_ic", "max_drawdown",
                        "signal_count", "turnover", "n_rows", "eval_time_s",
                        "created_at"}
            missing = expected - col_names
            assert not missing, f"Missing columns: {missing}"

    def test_store_result(self, db_path, tmp_path):
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text('[ensemble]\nalgorithms = []\n[_meta]\nconfig_hash = "abc123"\n')
        ev = Evaluator(str(cfg_path), "data/features", db_path)
        fitness = {
            "sharpe": 2.5, "mean_ic": 0.03, "max_drawdown": 0.08,
            "signal_count": 100.0, "turnover": 50.0,
        }
        ev.store_result(fitness, "BTC", 24, 10000, 5.0)

        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT * FROM trials").fetchone()
            assert row is not None
            # trial_id, hash, json, symbol, hours, sharpe, ic, dd, sig, turn, nrows, time, created
            assert row[1] == "abc123"
            assert row[3] == "BTC"
            assert abs(row[5] - 2.5) < 1e-6  # sharpe


class TestEvaluatorFitness:
    def test_fitness_computation(self, tmp_path):
        """Test fitness on a synthetic ensemble signal."""
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text('[ensemble]\nalgorithms = []\n')
        ev = Evaluator(str(cfg_path), "data/features", str(tmp_path / "db.sqlite"))

        n = 5000
        rng = np.random.default_rng(42)
        mid = 50000 * np.exp(np.cumsum(rng.normal(0, 0.0001, n)))

        base_df = pd.DataFrame({
            "raw_midprice": mid,
            "ent_book_shape": rng.uniform(0, 1, n),
        })
        # Signal that correlates with forward returns (should have positive IC)
        fwd = np.roll(mid, -1) / mid - 1
        fwd[-1] = 0
        ensemble_df = pd.DataFrame({
            "ens_signal": fwd + rng.normal(0, 0.0001, n),  # noisy oracle
        })

        fitness = ev.compute_fitness(ensemble_df, base_df)
        assert "sharpe" in fitness
        assert "mean_ic" in fitness
        assert "max_drawdown" in fitness
        assert "signal_count" in fitness
        assert "turnover" in fitness
        assert fitness["mean_ic"] > 0, "Oracle signal should have positive IC"

    def test_nan_fitness_on_insufficient_data(self, tmp_path):
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text('[ensemble]\nalgorithms = []\n')
        ev = Evaluator(str(cfg_path), "data/features", str(tmp_path / "db.sqlite"))

        # Only 10 rows — insufficient
        base_df = pd.DataFrame({
            "raw_midprice": [50000] * 10,
        })
        ensemble_df = pd.DataFrame({
            "ens_signal": [0.0] * 10,
        })
        fitness = ev.compute_fitness(ensemble_df, base_df)
        assert math.isnan(fitness["sharpe"])

    def test_nan_fitness_no_midprice(self, tmp_path):
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text('[ensemble]\nalgorithms = []\n')
        ev = Evaluator(str(cfg_path), "data/features", str(tmp_path / "db.sqlite"))

        base_df = pd.DataFrame({"other_col": [1.0] * 100})
        ensemble_df = pd.DataFrame({"ens_signal": [0.0] * 100})
        fitness = ev.compute_fitness(ensemble_df, base_df)
        assert math.isnan(fitness["sharpe"])


# ── Orchestrator ────────────────────────────────────────────────────────────


class TestOrchestratorStatus:
    def test_status_no_db(self, tmp_path):
        from swarm.orchestrator import SwarmOrchestrator
        orch = SwarmOrchestrator(db_path=str(tmp_path / "nonexistent.db"))
        status = orch.status()
        assert status["status"] == "no_runs"
        assert status["total_trials"] == 0

    def test_status_with_results(self, db_path, tmp_path):
        from swarm.orchestrator import SwarmOrchestrator
        # Seed the DB with some results
        with sqlite3.connect(db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute(
                "INSERT INTO trials (config_hash, config_json, symbol, eval_hours, sharpe, mean_ic, max_drawdown, signal_count, turnover, n_rows, eval_time_s) "
                "VALUES ('hash1', '{}', 'BTC', 24, 2.5, 0.03, 0.08, 100, 50, 10000, 5.0)")
            conn.execute(
                "INSERT INTO trials (config_hash, config_json, symbol, eval_hours, sharpe, mean_ic, max_drawdown, signal_count, turnover, n_rows, eval_time_s) "
                "VALUES ('hash2', '{}', 'BTC', 24, 1.5, 0.02, 0.12, 80, 40, 10000, 4.0)")

        orch = SwarmOrchestrator(db_path=db_path)
        status = orch.status()
        assert status["total_trials"] == 2
        assert status["best_sharpe"] == 2.5


class TestOrchestratorResults:
    def test_results_empty(self, tmp_path):
        from swarm.orchestrator import SwarmOrchestrator
        orch = SwarmOrchestrator(db_path=str(tmp_path / "nonexistent.db"))
        assert orch.results() == []

    def test_results_ordered_by_sharpe(self, db_path):
        from swarm.orchestrator import SwarmOrchestrator
        with sqlite3.connect(db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            for sharpe in [1.0, 3.0, 2.0]:
                conn.execute(
                    "INSERT INTO trials (config_hash, config_json, symbol, eval_hours, sharpe) "
                    f"VALUES ('h', '{{}}', 'BTC', 24, {sharpe})")

        orch = SwarmOrchestrator(db_path=db_path)
        rows = orch.results(top_n=3)
        sharpes = [r["sharpe"] for r in rows]
        assert sharpes == [3.0, 2.0, 1.0], "Should be sorted descending"

    def test_results_top_n_limit(self, db_path):
        from swarm.orchestrator import SwarmOrchestrator
        with sqlite3.connect(db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            for i in range(20):
                conn.execute(
                    "INSERT INTO trials (config_hash, config_json, symbol, eval_hours, sharpe) "
                    f"VALUES ('h{i}', '{{}}', 'BTC', 24, {i})")

        orch = SwarmOrchestrator(db_path=db_path)
        assert len(orch.results(top_n=5)) == 5


class TestOrchestratorExport:
    def test_export_best(self, db_path, tmp_path):
        from swarm.orchestrator import SwarmOrchestrator
        best_cfg = {"jump_detector": {"window": 200}, "ensemble": {"method": "ic_weight"}}
        with sqlite3.connect(db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute(
                "INSERT INTO trials (config_hash, config_json, symbol, eval_hours, sharpe) "
                "VALUES ('best', ?, 'BTC', 24, 5.0)",
                (json.dumps(best_cfg),))

        orch = SwarmOrchestrator(db_path=db_path)
        out = str(tmp_path / "best.toml")
        orch.export_best(out)

        exported = toml.load(out)
        assert exported["jump_detector"]["window"] == 200
        assert exported["ensemble"]["method"] == "ic_weight"

    def test_export_no_results(self, tmp_path):
        from swarm.orchestrator import SwarmOrchestrator
        orch = SwarmOrchestrator(db_path=str(tmp_path / "empty.db"))
        result = orch.export_best(str(tmp_path / "out.toml"))
        assert result is None


# ── ParquetReader ───────────────────────────────────────────────────────────


class TestParquetReader:
    def test_find_latest_date(self, tmp_path):
        from swarm.parquet_reader import _find_latest_date
        (tmp_path / "2026-05-01").mkdir()
        (tmp_path / "2026-06-04").mkdir()
        (tmp_path / "2026-05-15").mkdir()
        assert _find_latest_date(str(tmp_path)) == "2026-06-04"

    def test_find_latest_date_empty(self, tmp_path):
        from swarm.parquet_reader import _find_latest_date
        assert _find_latest_date(str(tmp_path)) is None

    def test_find_latest_date_ignores_non_dates(self, tmp_path):
        from swarm.parquet_reader import _find_latest_date
        (tmp_path / "2026-05-01").mkdir()
        (tmp_path / "archive").mkdir()
        (tmp_path / "README.md").touch()
        assert _find_latest_date(str(tmp_path)) == "2026-05-01"


# ── End-to-end ConfigGenerator → write → reload ────────────────────────────


class TestRoundTrip:
    def test_config_roundtrip(self, base_config, ranges_config, tmp_path):
        """Generate → write → reload → verify all params in range."""
        gen = ConfigGenerator(base_config, ranges_config)
        configs = gen.generate_random(5, seed=42)
        paths = gen.write_configs(configs, str(tmp_path / "configs"))

        for p in paths:
            cfg = toml.load(str(p))
            # Verify range compliance
            w = cfg["jump_detector"]["window"]
            assert 50 <= w <= 500
            s = cfg["jump_detector"]["significance"]
            assert 2.0 <= s <= 5.0
            sp = cfg["optimal_entry"]["sigma_process"]
            assert 0.001 <= sp <= 0.1
