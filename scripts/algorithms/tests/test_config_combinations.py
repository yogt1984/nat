"""
Config combination tests — verify algorithms work with varied parameters.

Tests that each algorithm:
  1. Works with the default parameters from config/algorithms.toml
  2. Works with halved/doubled parameter values (stress test)
  3. Works with boundary parameter values (minimum reasonable)
  4. Produces finite, non-degenerate output across configurations

This catches regressions where a code change implicitly assumes a specific
parameter value (e.g., hardcoded buffer size matching the default window).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from algorithms.autodiscover import discover_all
from algorithms.registry import get_algorithm
from algorithms.tests.conftest import make_synthetic_ticks

discover_all()

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "config" / "algorithms.toml"

# Load the config once
with open(CONFIG_PATH, "rb") as f:
    ALGO_CONFIG = tomllib.load(f)


# ---------------------------------------------------------------------------
# Winner algorithms with their tunable parameters
# ---------------------------------------------------------------------------

WINNER_PARAMS = {
    "jump_detector": {
        "default": {"window": 100, "significance": 3.0, "reversion_horizon": 50},
        "halved": {"window": 50, "significance": 1.5, "reversion_horizon": 25},
        "doubled": {"window": 200, "significance": 6.0, "reversion_horizon": 100},
        "minimal": {"window": 20, "significance": 1.0, "reversion_horizon": 5},
    },
    "optimal_entry": {
        "default": {"theta": 0.1, "sigma_process": 0.01, "sigma_obs": 0.1,
                     "dt": 0.1, "sprt_drift": 0.001, "alpha_error": 0.05,
                     "beta_error": 0.20},
        "halved": {"theta": 0.05, "sigma_process": 0.005, "sigma_obs": 0.05,
                   "dt": 0.05, "sprt_drift": 0.0005, "alpha_error": 0.025,
                   "beta_error": 0.10},
        "doubled": {"theta": 0.2, "sigma_process": 0.02, "sigma_obs": 0.2,
                    "dt": 0.2, "sprt_drift": 0.002, "alpha_error": 0.10,
                    "beta_error": 0.40},
        "minimal": {"theta": 0.01, "sigma_process": 0.001, "sigma_obs": 0.01,
                    "dt": 0.01, "sprt_drift": 0.0001, "alpha_error": 0.01,
                    "beta_error": 0.05},
    },
    "funding_reversion": {
        "default": {"zscore_entry": 2.0, "momentum_span": 100, "premium_weight": 0.3},
        "halved": {"zscore_entry": 1.0, "momentum_span": 50, "premium_weight": 0.15},
        "doubled": {"zscore_entry": 4.0, "momentum_span": 200, "premium_weight": 0.6},
        "minimal": {"zscore_entry": 0.5, "momentum_span": 10, "premium_weight": 0.0},
    },
    "surprise_signal": {
        "default": {"roc_window": 50},
        "halved": {"roc_window": 25},
        "doubled": {"roc_window": 100},
        "minimal": {"roc_window": 5},
    },
    "weighted_ofi": {
        "default": {"decay_lambda": 0.5, "ema_span": 50, "auto_tune": False},
        "halved": {"decay_lambda": 0.25, "ema_span": 25, "auto_tune": False},
        "doubled": {"decay_lambda": 1.0, "ema_span": 100, "auto_tune": False},
        "minimal": {"decay_lambda": 0.1, "ema_span": 5, "auto_tune": False},
    },
}

# Non-winner algorithms with config sections
OTHER_PARAMS = {
    "hawkes_intensity": {
        "default": {"baseline_window": 300, "decay_beta": 0.1, "alpha_fraction": 0.5,
                     "auto_tune": False},
        "halved": {"baseline_window": 150, "decay_beta": 0.05, "alpha_fraction": 0.25,
                   "auto_tune": False},
        "doubled": {"baseline_window": 600, "decay_beta": 0.2, "alpha_fraction": 0.8,
                    "auto_tune": False},
    },
    "propagator": {
        "default": {"decay_exponent": 0.5, "impact_window": 100,
                     "permanent_ema_span": 500, "auto_tune": False},
        "halved": {"decay_exponent": 0.25, "impact_window": 50,
                   "permanent_ema_span": 250, "auto_tune": False},
        "doubled": {"decay_exponent": 1.0, "impact_window": 200,
                    "permanent_ema_span": 1000, "auto_tune": False},
    },
    "vpin_regime": {
        "default": {"vpin_threshold_pct": 80, "momentum_span": 50, "gate_window": 300},
        "halved": {"vpin_threshold_pct": 40, "momentum_span": 25, "gate_window": 150},
        "doubled": {"vpin_threshold_pct": 95, "momentum_span": 100, "gate_window": 600},
    },
}

ALL_PARAMS = {**WINNER_PARAMS, **OTHER_PARAMS}


def _run_algorithm(name: str, kwargs: dict, n_ticks: int = 500) -> pd.DataFrame:
    """Instantiate algorithm with kwargs, run on synthetic data, return output."""
    alg = get_algorithm(name, **kwargs)
    df = make_synthetic_ticks(n_ticks, alg.required_columns())
    return alg.run_batch(df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Build parametrize list: (algo_name, variant_name, kwargs)
_PARAM_CASES = []
for algo, variants in ALL_PARAMS.items():
    for variant, kwargs in variants.items():
        _PARAM_CASES.append(
            pytest.param(algo, variant, kwargs, id=f"{algo}_{variant}")
        )


class TestConfigCombinations:
    """Verify algorithms work across parameter configurations."""

    @pytest.mark.parametrize("algo_name,variant,kwargs", _PARAM_CASES)
    def test_no_crash(self, algo_name, variant, kwargs):
        """Algorithm runs without error for this config."""
        result = _run_algorithm(algo_name, kwargs)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.parametrize("algo_name,variant,kwargs", _PARAM_CASES)
    def test_correct_columns(self, algo_name, variant, kwargs):
        """Output columns match declared features regardless of config."""
        alg = get_algorithm(algo_name, **kwargs)
        df = make_synthetic_ticks(500, alg.required_columns())
        result = alg.run_batch(df)
        expected = [f.name for f in alg.alg_features()]
        assert list(result.columns) == expected

    @pytest.mark.parametrize("algo_name,variant,kwargs", _PARAM_CASES)
    def test_finite_post_warmup(self, algo_name, variant, kwargs):
        """At least some values are finite after warmup."""
        alg = get_algorithm(algo_name, **kwargs)
        warmup = alg.warmup
        n = max(warmup * 3, 500)
        df = make_synthetic_ticks(n, alg.required_columns())
        result = alg.run_batch(df)
        post = result.iloc[warmup + 50:]
        if len(post) < 10:
            pytest.skip("Not enough post-warmup data")
        finite_rate = post.apply(lambda c: np.isfinite(c).mean()).mean()
        assert finite_rate > 0.1, (
            f"{algo_name}[{variant}] produced <10% finite values post-warmup"
        )


class TestConfigFromToml:
    """Verify algorithms work with exact parameters from algorithms.toml."""

    @pytest.mark.parametrize("algo_name", list(WINNER_PARAMS.keys()))
    def test_winner_with_toml_config(self, algo_name):
        """Winner algorithm runs with parameters from config/algorithms.toml."""
        toml_section = ALGO_CONFIG.get(algo_name, {})
        # Filter to only params the constructor accepts
        alg_default = get_algorithm(algo_name)
        import inspect
        sig = inspect.signature(type(alg_default).__init__)
        valid_params = {k for k in sig.parameters if k != "self"}
        kwargs = {k: v for k, v in toml_section.items() if k in valid_params}

        result = _run_algorithm(algo_name, kwargs)
        assert len(result) > 0
        # Verify output has correct columns
        alg = get_algorithm(algo_name, **kwargs)
        expected = [f.name for f in alg.alg_features()]
        assert list(result.columns) == expected


class TestParameterEdgeCases:
    """Test specific edge-case parameters that might break algorithms."""

    def test_jump_detector_window_equals_warmup(self):
        """Window matching warmup shouldn't cause issues."""
        result = _run_algorithm("jump_detector", {"window": 100, "significance": 3.0})
        assert len(result) > 0

    def test_funding_reversion_zero_premium_weight(self):
        """Zero premium weight disables the premium term."""
        result = _run_algorithm("funding_reversion",
                                {"zscore_entry": 2.0, "premium_weight": 0.0})
        assert len(result) > 0

    def test_weighted_ofi_high_decay(self):
        """Very high decay concentrates on top-of-book."""
        result = _run_algorithm("weighted_ofi",
                                {"decay_lambda": 5.0, "ema_span": 50, "auto_tune": False})
        assert len(result) > 0

    def test_surprise_signal_short_roc_window(self):
        """Very short ROC window should still produce valid output."""
        result = _run_algorithm("surprise_signal", {"roc_window": 5})
        alg = get_algorithm("surprise_signal", roc_window=5)
        warmup = alg.warmup
        post = result.iloc[warmup + 50:]
        if len(post) > 10:
            finite_rate = post.apply(lambda c: np.isfinite(c).mean()).mean()
            assert finite_rate > 0.1

    def test_optimal_entry_high_alpha_error(self):
        """High alpha error should lower SPRT boundaries, producing more signals."""
        result = _run_algorithm("optimal_entry", {
            "theta": 0.1, "sigma_process": 0.01, "sigma_obs": 0.1,
            "dt": 0.1, "sprt_drift": 0.001, "alpha_error": 0.30,
            "beta_error": 0.30,
        })
        assert len(result) > 0
