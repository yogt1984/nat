"""
Tier 2 integration tests — run swarm components against real Parquet data.

Skipped automatically if data/features/ doesn't have recent data.
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import toml

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

DATA_DIR = ROOT / "data" / "features"
BASE_CONFIG = ROOT / "config" / "algorithms.toml"
RANGES_CONFIG = ROOT / "config" / "swarm_ranges.toml"

HAS_DATA = DATA_DIR.exists() and any(DATA_DIR.iterdir()) if DATA_DIR.exists() else False
HAS_CONFIGS = BASE_CONFIG.exists() and RANGES_CONFIG.exists()

skip_no_data = pytest.mark.skipif(
    not HAS_DATA, reason="No Parquet data in data/features/"
)
skip_no_configs = pytest.mark.skipif(
    not HAS_CONFIGS, reason="Config files missing"
)


# ── Parquet Reader Integration ──────────────────────────────────────────────


@skip_no_data
class TestParquetReaderIntegration:
    def test_read_evaluation_data_btc(self):
        from swarm.parquet_reader import read_evaluation_data
        df = read_evaluation_data(str(DATA_DIR), symbol="BTC", hours=1)
        assert len(df) > 0, "Should load at least some BTC data"
        assert "timestamp_ns" in df.columns
        assert "symbol" in df.columns
        assert "raw_midprice" in df.columns
        assert (df["symbol"] == "BTC").all()

    def test_read_evaluation_data_sorted(self):
        from swarm.parquet_reader import read_evaluation_data
        df = read_evaluation_data(str(DATA_DIR), symbol="BTC", hours=1)
        ts = df["timestamp_ns"].values
        assert np.all(ts[1:] >= ts[:-1]), "Data should be sorted by timestamp"

    def test_read_evaluation_data_memory_cap(self):
        from swarm.parquet_reader import read_evaluation_data
        df = read_evaluation_data(
            str(DATA_DIR), symbol="BTC", hours=1, max_memory_mb=100.0
        )
        est_mb = len(df) * df.shape[1] * 8 / 1e6
        # Allow 2x slack for overhead
        assert est_mb < 200, f"Loaded {est_mb:.0f} MB, should stay near 100 MB cap"

    def test_list_available_data(self):
        from swarm.parquet_reader import list_available_data
        info = list_available_data(str(DATA_DIR))
        assert info["file_count"] > 0
        assert info["total_rows"] > 0
        assert len(info["symbols"]) > 0


# ── Config Generator → Evaluator Round-Trip ─────────────────────────────────


@skip_no_data
@skip_no_configs
class TestSwarmIntegration:
    def test_generate_and_evaluate_single(self, tmp_path):
        """Generate 1 config, evaluate on 1 hour of BTC data."""
        from swarm.config_generator import ConfigGenerator
        from swarm.evaluator import Evaluator

        gen = ConfigGenerator(str(BASE_CONFIG), str(RANGES_CONFIG))
        configs = gen.generate_random(1, seed=42)
        paths = gen.write_configs(configs, str(tmp_path / "configs"))

        db_path = str(tmp_path / "results.db")
        ev = Evaluator(str(paths[0]), str(DATA_DIR), db_path)
        fitness = ev.evaluate(symbol="BTC", hours=1)

        # Fitness should have all expected keys
        assert set(fitness.keys()) >= {"sharpe", "mean_ic", "max_drawdown",
                                        "signal_count", "turnover"}
        # At least some algorithms should have produced output
        assert not (np.isnan(fitness["sharpe"]) and fitness["signal_count"] == 0), \
            "Evaluator produced no output — likely all algorithms failed"

        # Check SQLite was written
        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
            assert count == 1

    def test_evaluator_multiple_configs(self, tmp_path):
        """Evaluate 3 different configs, verify they produce different results."""
        from swarm.config_generator import ConfigGenerator
        from swarm.evaluator import Evaluator

        gen = ConfigGenerator(str(BASE_CONFIG), str(RANGES_CONFIG))
        configs = gen.generate_random(3, seed=42)
        paths = gen.write_configs(configs, str(tmp_path / "configs"))

        db_path = str(tmp_path / "results.db")
        sharpes = []
        for p in paths:
            ev = Evaluator(str(p), str(DATA_DIR), db_path)
            fitness = ev.evaluate(symbol="BTC", hours=1)
            sharpes.append(fitness.get("sharpe", float("nan")))

        # At least 2 out of 3 should produce non-NaN results
        valid = [s for s in sharpes if not np.isnan(s)]
        assert len(valid) >= 2, f"Only {len(valid)}/3 evals produced results"

        # Results should be stored in SQLite
        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
            assert count == 3

    def test_orchestrator_status_after_eval(self, tmp_path):
        """Run 1 eval, then check status."""
        from swarm.config_generator import ConfigGenerator
        from swarm.evaluator import Evaluator
        from swarm.orchestrator import SwarmOrchestrator

        gen = ConfigGenerator(str(BASE_CONFIG), str(RANGES_CONFIG))
        configs = gen.generate_random(1, seed=42)
        paths = gen.write_configs(configs, str(tmp_path / "configs"))

        db_path = str(tmp_path / "results.db")
        ev = Evaluator(str(paths[0]), str(DATA_DIR), db_path)
        ev.evaluate(symbol="BTC", hours=1)

        orch = SwarmOrchestrator(db_path=db_path)
        status = orch.status()
        assert status["total_trials"] == 1
        assert status["best_sharpe"] is not None

        results = orch.results(top_n=1)
        assert len(results) == 1
        assert results[0]["symbol"] == "BTC"

    def test_export_best_roundtrip(self, tmp_path):
        """Evaluate → export best → verify exported TOML is valid."""
        from swarm.config_generator import ConfigGenerator
        from swarm.evaluator import Evaluator
        from swarm.orchestrator import SwarmOrchestrator

        gen = ConfigGenerator(str(BASE_CONFIG), str(RANGES_CONFIG))
        configs = gen.generate_random(2, seed=42)
        paths = gen.write_configs(configs, str(tmp_path / "configs"))

        db_path = str(tmp_path / "results.db")
        for p in paths:
            ev = Evaluator(str(p), str(DATA_DIR), db_path)
            ev.evaluate(symbol="BTC", hours=1)

        orch = SwarmOrchestrator(db_path=db_path)
        export_path = str(tmp_path / "best.toml")
        orch.export_best(export_path)

        # Verify exported TOML is valid and has algorithm sections
        exported = toml.load(export_path)
        assert "ensemble" in exported
        # Should have at least some algorithm configs
        algo_sections = [k for k in exported if k not in ("_meta", "evaluation")]
        assert len(algo_sections) >= 3, \
            f"Exported config missing algorithm sections: {list(exported.keys())}"
