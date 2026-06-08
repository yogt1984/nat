"""
Tier 3 tests — verify Optuna optimizer, study lifecycle, guard rails,
deflated Sharpe, and CLI integration.

These are offline tests that mock the evaluator to avoid needing real data.
"""

import copy
import json
import math
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import toml

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from swarm.optuna_optimizer import (
    DEFAULT_STORAGE,
    MAX_TURNOVER_PER_DAY,
    MIN_SIGNAL_COUNT_PER_DAY,
    NATOptimizer,
    OVERFIT_REJECT_RATIO,
    OVERFIT_WARN_RATIO,
    SAMPLERS,
    deflated_sharpe,
)

BASE_CONFIG = ROOT / "config" / "algorithms.toml"
RANGES_CONFIG = ROOT / "config" / "swarm_ranges.toml"

HAS_CONFIGS = BASE_CONFIG.exists() and RANGES_CONFIG.exists()
skip_no_configs = pytest.mark.skipif(
    not HAS_CONFIGS, reason="Config files missing",
)


# ── NATOptimizer construction ──────────────────────────────────────────────


class TestNATOptimizerInit:
    def test_create_single_objective(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(
            study_name="test_single",
            storage=storage,
            sampler="cma",
        )
        assert opt.multi_objective is False
        assert opt.study.study_name == "test_single"
        assert len(opt.study.directions) == 1

    def test_create_multi_objective(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(
            study_name="test_multi",
            storage=storage,
            sampler="nsga2",
        )
        assert opt.multi_objective is True
        assert len(opt.study.directions) == 3

    def test_create_tpe(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(
            study_name="test_tpe",
            storage=storage,
            sampler="tpe",
        )
        assert opt.multi_objective is False
        assert "TPE" in type(opt.study.sampler).__name__

    def test_invalid_sampler(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        with pytest.raises(ValueError, match="Unknown sampler"):
            NATOptimizer(study_name="bad", storage=storage, sampler="pso")

    def test_load_if_exists(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt1 = NATOptimizer(study_name="persist", storage=storage, sampler="cma")
        opt2 = NATOptimizer(study_name="persist", storage=storage, sampler="cma")
        assert opt2.study.study_name == "persist"

    def test_storage_dir_created(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        storage = f"sqlite:///{nested / 'test.db'}"
        NATOptimizer(study_name="nested", storage=storage, sampler="cma")
        assert nested.exists()

    def test_default_params(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(study_name="defaults", storage=storage)
        assert opt.eval_hours == 720
        assert abs(opt.train_frac - 2 / 3) < 0.01
        assert opt.symbol == "BTC"
        assert opt.guard_rails is True


# ── from_study classmethod ─────────────────────────────────────────────────


class TestFromStudy:
    def test_load_existing_study(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        NATOptimizer(study_name="load_me", storage=storage, sampler="cma")
        loaded = NATOptimizer.from_study("load_me", storage)
        assert loaded.study.study_name == "load_me"
        assert loaded.multi_objective is False

    def test_load_multi_objective(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        NATOptimizer(study_name="multi", storage=storage, sampler="nsga2")
        loaded = NATOptimizer.from_study("multi", storage)
        assert loaded.multi_objective is True

    def test_load_nonexistent_study(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        with pytest.raises(KeyError):
            NATOptimizer.from_study("nonexistent", storage)


# ── Status with no trials ─────────────────────────────────────────────────


class TestStatusEmpty:
    def test_empty_study(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(study_name="empty", storage=storage)
        s = opt.status()
        assert s["total"] == 0
        assert s["study_name"] == "empty"


# ── Objective function with mocks ──────────────────────────────────────────


@skip_no_configs
class TestObjective:
    @pytest.fixture
    def optimizer(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        return NATOptimizer(
            study_name="obj_test",
            storage=storage,
            sampler="cma",
            base_config=str(BASE_CONFIG),
            ranges_config=str(RANGES_CONFIG),
            data_dir=str(tmp_path / "data"),
            eval_hours=24,
            guard_rails=True,
        )

    def _make_mock_fitness(self, sharpe=2.0, ic=0.05, dd=0.02,
                           sig_count=200.0, turnover=50.0):
        return {
            "sharpe": sharpe,
            "mean_ic": ic,
            "max_drawdown": dd,
            "signal_count": sig_count,
            "turnover": turnover,
        }

    def _make_mock_df(self, n_rows=10000):
        df = MagicMock()
        df.__len__ = MagicMock(return_value=n_rows)
        df.iloc.__getitem__ = MagicMock(return_value=df)
        df.reset_index = MagicMock(return_value=df)
        return df

    @patch("swarm.evaluator.Evaluator")
    @patch("swarm.config_generator.ConfigGenerator")
    def test_successful_trial(self, MockGen, MockEval, optimizer, tmp_path):
        """A trial with good fitness should return positive Sharpe."""
        import optuna

        mock_gen = MockGen.return_value
        mock_gen.generate_from_optuna.return_value = {"ensemble": {}, "_test": True}

        mock_eval = MockEval.return_value
        df = self._make_mock_df()
        mock_eval.load_data.return_value = df
        mock_eval.run_algorithms.return_value = {"algo1": df}
        mock_eval.run_ensemble.return_value = df
        mock_eval.compute_fitness.return_value = self._make_mock_fitness(
            sharpe=3.0, ic=0.05, dd=0.01, sig_count=200.0, turnover=30.0,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: optimizer._objective(trial),
            n_trials=1,
        )

        assert len(study.trials) == 1
        trial = study.trials[0]
        assert trial.state == optuna.trial.TrialState.COMPLETE

    @patch("swarm.evaluator.Evaluator")
    @patch("swarm.config_generator.ConfigGenerator")
    def test_low_signal_count_rejected(self, MockGen, MockEval, optimizer):
        """Trial with signal_count < 50/day should return worst."""
        import optuna

        mock_gen = MockGen.return_value
        mock_gen.generate_from_optuna.return_value = {"ensemble": {}}

        mock_eval = MockEval.return_value
        df = self._make_mock_df()
        mock_eval.load_data.return_value = df
        mock_eval.run_algorithms.return_value = {"algo1": df}
        mock_eval.run_ensemble.return_value = df
        mock_eval.compute_fitness.return_value = self._make_mock_fitness(
            sharpe=5.0, sig_count=10.0,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optimizer._objective(trial), n_trials=1)

        trial = study.trials[0]
        assert trial.value == 0.0
        assert trial.user_attrs.get("reject_reason") == "low_signal_count"

    @patch("swarm.evaluator.Evaluator")
    @patch("swarm.config_generator.ConfigGenerator")
    def test_high_turnover_rejected(self, MockGen, MockEval, optimizer):
        """Trial with turnover > 100/day should return worst."""
        import optuna

        mock_gen = MockGen.return_value
        mock_gen.generate_from_optuna.return_value = {"ensemble": {}}

        mock_eval = MockEval.return_value
        df = self._make_mock_df()
        mock_eval.load_data.return_value = df
        mock_eval.run_algorithms.return_value = {"algo1": df}
        mock_eval.run_ensemble.return_value = df
        mock_eval.compute_fitness.return_value = self._make_mock_fitness(
            sharpe=5.0, turnover=200.0,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optimizer._objective(trial), n_trials=1)

        trial = study.trials[0]
        assert trial.value == 0.0
        assert trial.user_attrs.get("reject_reason") == "high_turnover"

    @patch("swarm.evaluator.Evaluator")
    @patch("swarm.config_generator.ConfigGenerator")
    def test_negative_sharpe_rejected(self, MockGen, MockEval, optimizer):
        """Trial with OOS Sharpe <= 0 should return worst."""
        import optuna

        mock_gen = MockGen.return_value
        mock_gen.generate_from_optuna.return_value = {"ensemble": {}}

        mock_eval = MockEval.return_value
        df = self._make_mock_df()
        mock_eval.load_data.return_value = df
        mock_eval.run_algorithms.return_value = {"algo1": df}
        mock_eval.run_ensemble.return_value = df
        mock_eval.compute_fitness.return_value = self._make_mock_fitness(
            sharpe=-1.0, sig_count=200.0,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optimizer._objective(trial), n_trials=1)

        trial = study.trials[0]
        assert trial.value == 0.0
        assert trial.user_attrs.get("reject_reason") == "non_positive_sharpe"

    @patch("swarm.evaluator.Evaluator")
    @patch("swarm.config_generator.ConfigGenerator")
    def test_data_load_failure_returns_worst(self, MockGen, MockEval, optimizer):
        """Trial where data load fails should return worst gracefully."""
        import optuna

        mock_gen = MockGen.return_value
        mock_gen.generate_from_optuna.return_value = {"ensemble": {}}

        mock_eval = MockEval.return_value
        mock_eval.load_data.side_effect = FileNotFoundError("No data")

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optimizer._objective(trial), n_trials=1)

        assert study.trials[0].value == 0.0

    @patch("swarm.evaluator.Evaluator")
    @patch("swarm.config_generator.ConfigGenerator")
    def test_no_algo_output_returns_worst(self, MockGen, MockEval, optimizer):
        """Trial where no algorithms produce output returns worst."""
        import optuna

        mock_gen = MockGen.return_value
        mock_gen.generate_from_optuna.return_value = {"ensemble": {}}

        mock_eval = MockEval.return_value
        df = self._make_mock_df()
        mock_eval.load_data.return_value = df
        mock_eval.run_algorithms.return_value = {}  # empty

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optimizer._objective(trial), n_trials=1)

        assert study.trials[0].value == 0.0


# ── Guard rails: overfit detection ─────────────────────────────────────────


@skip_no_configs
class TestGuardRails:
    @patch("swarm.evaluator.Evaluator")
    @patch("swarm.config_generator.ConfigGenerator")
    def test_overfit_flagged(self, MockGen, MockEval, tmp_path):
        """IS/OOS ratio > 3.0 should flag overfit and penalize Sharpe."""
        import optuna

        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(
            study_name="overfit_test",
            storage=storage,
            sampler="cma",
            base_config=str(BASE_CONFIG),
            ranges_config=str(RANGES_CONFIG),
            data_dir=str(tmp_path / "data"),
            eval_hours=24,
            guard_rails=True,
        )

        mock_gen = MockGen.return_value
        mock_gen.generate_from_optuna.return_value = {"ensemble": {}}

        mock_eval = MockEval.return_value
        df = MagicMock()
        df.__len__ = MagicMock(return_value=10000)
        df.iloc.__getitem__ = MagicMock(return_value=df)
        df.reset_index = MagicMock(return_value=df)
        mock_eval.load_data.return_value = df
        mock_eval.run_algorithms.return_value = {"algo1": df}
        mock_eval.run_ensemble.return_value = df

        # IS Sharpe=12.0, OOS Sharpe=2.0 → ratio=6.0 (overfit)
        call_count = [0]
        def fitness_side_effect(ens, base):
            call_count[0] += 1
            # First call is OOS, second is IS (guard rails)
            if call_count[0] == 1:
                return {"sharpe": 2.0, "mean_ic": 0.03, "max_drawdown": 0.02,
                        "signal_count": 200.0, "turnover": 50.0}
            else:
                return {"sharpe": 12.0, "mean_ic": 0.08, "max_drawdown": 0.01,
                        "signal_count": 300.0, "turnover": 30.0}

        mock_eval.compute_fitness.side_effect = fitness_side_effect

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: opt._objective(trial), n_trials=1)

        trial = study.trials[0]
        assert trial.user_attrs.get("overfit_flag") is True
        assert trial.user_attrs["overfit_ratio"] == 6.0
        # Sharpe should be penalized (< original 2.0)
        assert trial.value < 2.0
        assert "penalized_sharpe" in trial.user_attrs

    @patch("swarm.evaluator.Evaluator")
    @patch("swarm.config_generator.ConfigGenerator")
    def test_guard_rails_disabled(self, MockGen, MockEval, tmp_path):
        """With guard_rails=False, overfit detection should be skipped."""
        import optuna

        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(
            study_name="no_guard",
            storage=storage,
            sampler="cma",
            base_config=str(BASE_CONFIG),
            ranges_config=str(RANGES_CONFIG),
            data_dir=str(tmp_path / "data"),
            eval_hours=24,
            guard_rails=False,
        )

        mock_gen = MockGen.return_value
        mock_gen.generate_from_optuna.return_value = {"ensemble": {}}

        mock_eval = MockEval.return_value
        df = MagicMock()
        df.__len__ = MagicMock(return_value=10000)
        df.iloc.__getitem__ = MagicMock(return_value=df)
        df.reset_index = MagicMock(return_value=df)
        mock_eval.load_data.return_value = df
        mock_eval.run_algorithms.return_value = {"algo1": df}
        mock_eval.run_ensemble.return_value = df
        mock_eval.compute_fitness.return_value = {
            "sharpe": 5.0, "mean_ic": 0.05, "max_drawdown": 0.01,
            "signal_count": 200.0, "turnover": 30.0,
        }

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: opt._objective(trial), n_trials=1)

        trial = study.trials[0]
        assert "overfit_flag" not in trial.user_attrs
        # compute_fitness should only be called once (OOS), not twice
        assert mock_eval.compute_fitness.call_count == 1


# ── Multi-objective (NSGA-II) ──────────────────────────────────────────────


@skip_no_configs
class TestMultiObjective:
    @patch("swarm.evaluator.Evaluator")
    @patch("swarm.config_generator.ConfigGenerator")
    def test_nsga2_returns_triple(self, MockGen, MockEval, tmp_path):
        """NSGA-II objective should return (sharpe, dd, ic) tuple."""
        import optuna

        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(
            study_name="nsga2_test",
            storage=storage,
            sampler="nsga2",
            base_config=str(BASE_CONFIG),
            ranges_config=str(RANGES_CONFIG),
            data_dir=str(tmp_path / "data"),
            eval_hours=24,
            guard_rails=False,
        )

        mock_gen = MockGen.return_value
        mock_gen.generate_from_optuna.return_value = {"ensemble": {}}

        mock_eval = MockEval.return_value
        df = MagicMock()
        df.__len__ = MagicMock(return_value=10000)
        df.iloc.__getitem__ = MagicMock(return_value=df)
        df.reset_index = MagicMock(return_value=df)
        mock_eval.load_data.return_value = df
        mock_eval.run_algorithms.return_value = {"algo1": df}
        mock_eval.run_ensemble.return_value = df
        mock_eval.compute_fitness.return_value = {
            "sharpe": 3.0, "mean_ic": 0.04, "max_drawdown": 0.015,
            "signal_count": 200.0, "turnover": 40.0,
        }

        opt.study.optimize(lambda trial: opt._objective(trial), n_trials=1)

        trial = opt.study.trials[0]
        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert len(trial.values) == 3
        assert trial.values[0] == 3.0   # sharpe
        assert trial.values[1] == 0.015  # dd
        assert trial.values[2] == 0.04   # ic


# ── Status and best_configs ────────────────────────────────────────────────


class TestStatusAndBest:
    def _populate_study(self, storage, study_name, n=5):
        """Create a study with n synthetic completed trials."""
        import optuna

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        for i in range(n):
            trial = study.ask()
            trial.suggest_float("x", 0.0, 10.0)
            study.tell(trial, float(i + 1))
            study.trials[i].set_user_attr("oos_sharpe", float(i + 1))
        return study

    def test_status_with_trials(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        self._populate_study(storage, "status_test", n=5)
        opt = NATOptimizer.from_study("status_test", storage)
        s = opt.status()
        assert s["complete"] == 5
        assert s["best_sharpe"] == 5.0
        assert s["best_trial"] == 4

    def test_best_configs_ordering(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        self._populate_study(storage, "best_test", n=10)
        opt = NATOptimizer.from_study("best_test", storage)
        configs = opt.best_configs(top_n=3)
        assert len(configs) == 3
        # Should be ordered by Sharpe descending
        assert configs[0]["sharpe"] >= configs[1]["sharpe"]
        assert configs[1]["sharpe"] >= configs[2]["sharpe"]

    def test_best_configs_returns_params(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        self._populate_study(storage, "params_test", n=3)
        opt = NATOptimizer.from_study("params_test", storage)
        configs = opt.best_configs(top_n=1)
        assert "params" in configs[0]
        assert "x" in configs[0]["params"]


# ── Pareto front ───────────────────────────────────────────────────────────


class TestParetoFront:
    def test_pareto_requires_nsga2(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(
            study_name="single_obj",
            storage=storage,
            sampler="cma",
        )
        with pytest.raises(ValueError, match="NSGA-II"):
            opt.pareto_front()

    def test_pareto_front_structure(self, tmp_path):
        import optuna

        storage = f"sqlite:///{tmp_path / 'test.db'}"
        study = optuna.create_study(
            study_name="pareto_test",
            storage=storage,
            directions=["maximize", "minimize", "maximize"],
        )
        # Add some trials manually
        for i in range(5):
            trial = study.ask()
            trial.suggest_float("x", 0.0, 10.0)
            study.tell(trial, [float(i + 1), 0.1 * (5 - i), 0.01 * (i + 1)])

        opt = NATOptimizer.from_study("pareto_test", storage)
        front = opt.pareto_front()
        assert len(front) > 0
        for entry in front:
            assert "trial" in entry
            assert "sharpe" in entry
            assert "drawdown" in entry
            assert "ic" in entry


# ── Export best ────────────────────────────────────────────────────────────


@skip_no_configs
class TestExportBest:
    def test_export_creates_toml(self, tmp_path):
        import optuna

        storage = f"sqlite:///{tmp_path / 'test.db'}"
        study = optuna.create_study(
            study_name="export_test",
            storage=storage,
            direction="maximize",
        )
        trial = study.ask()
        trial.suggest_float("jump_detector.z_threshold", 2.0, 5.0)
        trial.suggest_int("jump_detector.window", 50, 500)
        study.tell(trial, 5.0)

        opt = NATOptimizer.from_study(
            "export_test", storage,
            base_config=str(BASE_CONFIG),
            ranges_config=str(RANGES_CONFIG),
        )
        out = tmp_path / "exported.toml"
        opt.export_best(str(out))

        assert out.exists()
        cfg = toml.load(out)
        assert "_meta" in cfg
        assert cfg["_meta"]["source"] == "optuna"
        assert cfg["_meta"]["sharpe"] == 5.0
        # Should have the suggested params merged into base config
        assert "jump_detector" in cfg

    def test_export_no_trials_raises(self, tmp_path):
        import optuna

        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(
            study_name="empty_export",
            storage=storage,
            sampler="cma",
            base_config=str(BASE_CONFIG),
            ranges_config=str(RANGES_CONFIG),
        )
        with pytest.raises(Exception):
            opt.export_best(str(tmp_path / "out.toml"))


# ── params_to_config ───────────────────────────────────────────────────────


@skip_no_configs
class TestParamsToConfig:
    def test_flat_to_nested(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(
            study_name="params_config",
            storage=storage,
            base_config=str(BASE_CONFIG),
            ranges_config=str(RANGES_CONFIG),
        )
        params = {
            "jump_detector.z_threshold": 3.5,
            "jump_detector.window": 200,
            "ensemble.method": "ic_weight",
        }
        config = opt._params_to_config(params)

        assert config["jump_detector"]["z_threshold"] == 3.5
        assert config["jump_detector"]["window"] == 200
        assert config["ensemble"]["method"] == "ic_weight"

    def test_preserves_base_config(self, tmp_path):
        storage = f"sqlite:///{tmp_path / 'test.db'}"
        opt = NATOptimizer(
            study_name="base_preserve",
            storage=storage,
            base_config=str(BASE_CONFIG),
            ranges_config=str(RANGES_CONFIG),
        )
        base = toml.load(str(BASE_CONFIG))
        config = opt._params_to_config({"jump_detector.z_threshold": 99.0})

        # Overridden param should change
        assert config["jump_detector"]["z_threshold"] == 99.0
        # Other sections should be preserved from base
        sections_in_base = set(base.keys())
        sections_in_config = set(config.keys())
        assert sections_in_base.issubset(sections_in_config)


# ── Deflated Sharpe ────────────────────────────────────────────────────────


class TestDeflatedSharpe:
    def test_basic_computation(self):
        dsr = deflated_sharpe(sharpe=3.0, n_trials=100)
        assert 0.0 <= dsr <= 1.0

    def test_higher_sharpe_higher_dsr(self):
        dsr_low = deflated_sharpe(sharpe=1.0, n_trials=100)
        dsr_high = deflated_sharpe(sharpe=5.0, n_trials=100)
        assert dsr_high > dsr_low

    def test_more_trials_lower_dsr(self):
        dsr_few = deflated_sharpe(sharpe=2.0, n_trials=10)
        dsr_many = deflated_sharpe(sharpe=2.0, n_trials=10000)
        assert dsr_few > dsr_many

    def test_edge_case_n_trials_1(self):
        dsr = deflated_sharpe(sharpe=5.0, n_trials=1)
        assert dsr == 0.0

    def test_edge_case_zero_std(self):
        dsr = deflated_sharpe(sharpe=5.0, n_trials=100, sharpe_std=0.0)
        assert dsr == 0.0

    def test_marginal_sharpe_low_dsr(self):
        """A barely-positive Sharpe with many trials should have low DSR."""
        dsr = deflated_sharpe(sharpe=0.5, n_trials=5000)
        assert dsr < 0.5


# ── Constants and samplers ─────────────────────────────────────────────────


class TestConstants:
    def test_all_samplers_defined(self):
        assert set(SAMPLERS.keys()) == {"cma", "tpe", "nsga2"}

    def test_hard_constraint_values(self):
        assert MIN_SIGNAL_COUNT_PER_DAY == 50
        assert MAX_TURNOVER_PER_DAY == 100

    def test_overfit_thresholds(self):
        assert OVERFIT_WARN_RATIO == 2.0
        assert OVERFIT_REJECT_RATIO == 3.0
        assert OVERFIT_REJECT_RATIO > OVERFIT_WARN_RATIO
