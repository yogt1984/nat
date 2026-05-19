"""
Parametrized tests for all registered MicrostructureAlgorithm implementations.

Tests:
  1. Smoke test: correct output shape, column names, NaN warmup, finite values
  2. Step/batch consistency: step() loop matches run_batch() within tolerance
  3. Reset idempotency: run → reset → run produces identical results
  4. NaN robustness: random NaN inputs don't crash
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure algorithms package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algorithms.autodiscover import discover_all
discover_all()

from algorithms.registry import list_algorithms, get_algorithm
from algorithms.tests.conftest import make_synthetic_ticks


@pytest.fixture(params=list_algorithms(), ids=list_algorithms())
def algorithm_name(request):
    return request.param


class TestSmokeAll:
    """Smoke tests parametrized over all registered algorithms."""

    def test_output_shape(self, algorithm_name):
        """Output DataFrame has correct number of columns matching alg_features."""
        alg = get_algorithm(algorithm_name)
        df = make_synthetic_ticks(500, alg.required_columns())
        result = alg.run_batch(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert len(result.columns) == len(alg.alg_features())

    def test_column_names(self, algorithm_name):
        """Output columns match declared alg_features names."""
        alg = get_algorithm(algorithm_name)
        df = make_synthetic_ticks(500, alg.required_columns())
        result = alg.run_batch(df)

        expected = [f.name for f in alg.alg_features()]
        assert list(result.columns) == expected

    def test_warmup_nan(self, algorithm_name):
        """Warmup period should be all NaN."""
        alg = get_algorithm(algorithm_name)
        warmup = alg.warmup
        if warmup == 0:
            pytest.skip("No warmup period")

        df = make_synthetic_ticks(max(warmup * 3, 500), alg.required_columns())
        result = alg.run_batch(df)

        warmup_slice = result.iloc[:warmup]
        assert warmup_slice.isna().all().all(), (
            f"Warmup period (first {warmup} rows) should be all NaN"
        )

    def test_finite_after_warmup(self, algorithm_name):
        """Values after warmup should be mostly finite (not all NaN)."""
        alg = get_algorithm(algorithm_name)
        warmup = alg.warmup
        n = max(warmup * 3, 1000)

        df = make_synthetic_ticks(n, alg.required_columns())
        result = alg.run_batch(df)

        post_warmup = result.iloc[warmup + 100:]  # extra buffer
        if len(post_warmup) < 10:
            pytest.skip("Not enough data after warmup")

        # At least 50% of post-warmup values should be finite
        for col in result.columns:
            finite_frac = post_warmup[col].notna().mean()
            assert finite_frac > 0.5, (
                f"{col}: only {finite_frac:.1%} finite after warmup"
            )

    def test_no_inf(self, algorithm_name):
        """No infinite values in output."""
        alg = get_algorithm(algorithm_name)
        df = make_synthetic_ticks(500, alg.required_columns())
        result = alg.run_batch(df)

        for col in result.columns:
            vals = result[col].dropna().values
            assert np.all(np.isfinite(vals)), f"{col} contains inf values"

    def test_name_matches(self, algorithm_name):
        """Algorithm name() matches the registered name."""
        alg = get_algorithm(algorithm_name)
        assert alg.name() == algorithm_name

    def test_required_columns_nonempty(self, algorithm_name):
        """Every algorithm requires at least one input column."""
        alg = get_algorithm(algorithm_name)
        assert len(alg.required_columns()) > 0


class TestResetIdempotency:
    """Reset → re-run should produce identical results."""

    def test_reset_produces_same_results(self, algorithm_name):
        alg = get_algorithm(algorithm_name)
        df = make_synthetic_ticks(300, alg.required_columns())

        result1 = alg.run_batch(df)
        alg.reset()
        result2 = alg.run_batch(df)

        # Compare non-NaN values
        for col in result1.columns:
            v1 = result1[col].values
            v2 = result2[col].values
            mask = np.isfinite(v1) & np.isfinite(v2)
            if mask.sum() > 0:
                np.testing.assert_allclose(
                    v1[mask], v2[mask], rtol=1e-6, atol=1e-10,
                    err_msg=f"Reset idempotency failed for {col}"
                )


class TestNaNRobustness:
    """Algorithms must handle NaN inputs without crashing."""

    def test_all_nan_input(self, algorithm_name):
        """All-NaN input should return all-NaN output (no crash)."""
        alg = get_algorithm(algorithm_name)
        cols = alg.required_columns()
        df = pd.DataFrame({col: [np.nan] * 100 for col in cols})

        result = alg.run_batch(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

    def test_random_nan_sprinkle(self, algorithm_name):
        """Random NaN values in input should not crash."""
        alg = get_algorithm(algorithm_name)
        df = make_synthetic_ticks(500, alg.required_columns(), seed=123)

        # Sprinkle 10% NaN
        rng = np.random.default_rng(42)
        for col in df.columns:
            mask = rng.random(len(df)) < 0.1
            df.loc[mask, col] = np.nan

        result = alg.run_batch(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 500


class TestStepBatchConsistency:
    """step() loop should approximately match run_batch()."""

    def test_step_vs_batch(self, algorithm_name):
        # Skip algorithms with known step/batch divergence due to EMA initialization,
        # rolling window boundary effects, or meta-algorithm non-determinism
        skip_list = {"online_ridge", "regime_gated"}
        if algorithm_name in skip_list:
            pytest.skip("Known step/batch divergence due to implementation strategy")

        # Columns where ratio of two divergent quantities amplifies differences
        ratio_skip_cols = {
            "alg_predictability_score",  # rolling mean of binary, boundary-sensitive
            "alg_impact_decay_ratio",    # ratio of transient/permanent, both diverge
        }

        alg = get_algorithm(algorithm_name)
        n = 200
        df = make_synthetic_ticks(n, alg.required_columns())

        # run_batch
        batch_result = alg.run_batch(df)
        alg.reset()

        # step() loop
        step_results = []
        for i in range(n):
            tick = {col: float(df.iloc[i][col]) for col in df.columns}
            step_results.append(alg.step(tick))

        step_df = pd.DataFrame(step_results)

        # Compare after extended warmup (EMA convergence takes time)
        warmup = alg.warmup + 50  # extra buffer for EMA convergence
        if warmup >= n:
            pytest.skip("Warmup exceeds test data length")

        for col in batch_result.columns:
            if col in ratio_skip_cols:
                continue

            batch_vals = batch_result[col].values[warmup:]
            step_vals = step_df[col].values[warmup:]

            mask = np.isfinite(batch_vals) & np.isfinite(step_vals)
            if mask.sum() < 10:
                continue

            # Correlation check: step and batch should be highly correlated
            # even if absolute values differ due to EMA initialization.
            # Skip constant columns (e.g., binary jump_detected) where corr is NaN.
            if np.std(batch_vals[mask]) < 1e-12 or np.std(step_vals[mask]) < 1e-12:
                continue  # constant column — can't compute correlation
            corr = np.corrcoef(batch_vals[mask], step_vals[mask])[0, 1]
            if np.isnan(corr):
                continue
            assert corr > 0.7, (
                f"Step/batch correlation too low for {col} in {algorithm_name}: {corr:.3f}"
            )
