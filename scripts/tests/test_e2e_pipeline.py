"""
E2E pipeline test — synthetic ticks → AlgorithmRunner → AlgorithmEvaluator → JSON report.

Validates that all 5 winner algorithms chain through the full evaluation
pipeline without crashing, produce structurally valid reports, and complete
within a reasonable time budget.

Uses synthetic data (no real fixture dependency).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from algorithms.autodiscover import discover_all
from algorithms.evaluate import AlgorithmEvaluator
from algorithms.registry import get_algorithm
from algorithms.runner import AlgorithmRunner, run_chain
from algorithms.tests.conftest import make_synthetic_ticks

discover_all()

WINNERS = ["jump_detector", "optimal_entry", "funding_reversion",
           "surprise_signal", "weighted_ofi"]


@pytest.fixture(scope="module")
def synthetic_df():
    """Synthetic tick DataFrame covering all 5 winners' requirements."""
    all_cols = set()
    for name in WINNERS:
        alg = get_algorithm(name)
        all_cols.update(alg.required_columns())
    # Extra columns needed by AlgorithmEvaluator
    all_cols.update(["raw_midprice", "raw_spread", "ent_book_shape"])
    return make_synthetic_ticks(5000, sorted(all_cols), seed=42)


@pytest.fixture(scope="module")
def runner_results(synthetic_df):
    """Pre-computed AlgorithmResult for each winner."""
    results = {}
    for name in WINNERS:
        alg = get_algorithm(name)
        runner = AlgorithmRunner(alg)
        results[name] = runner.run_on_dataframe(synthetic_df)
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWinnerOutput:
    """Each winner runs and produces correct output."""

    @pytest.mark.parametrize("name", WINNERS)
    def test_produces_output(self, name, runner_results):
        result = runner_results[name]
        assert result.algorithm_name == name
        assert len(result.features_df) > 0
        assert result.n_ticks == 5000

    @pytest.mark.parametrize("name", WINNERS)
    def test_correct_columns(self, name, runner_results):
        result = runner_results[name]
        alg = get_algorithm(name)
        expected = [f.name for f in alg.alg_features()]
        assert list(result.features_df.columns) == expected

    @pytest.mark.parametrize("name", WINNERS)
    def test_no_100pct_nan_post_warmup(self, name, runner_results):
        result = runner_results[name]
        warmup = result.warmup_ticks
        post = result.features_df.iloc[warmup + 50:]
        for col in post.columns:
            nan_frac = post[col].isna().mean()
            assert nan_frac < 1.0, f"{name}/{col}: 100% NaN post-warmup"


class TestChain:
    """run_chain() works with all 5 winners."""

    def test_no_crash(self, synthetic_df):
        algos = [get_algorithm(n) for n in WINNERS]
        results = run_chain(synthetic_df, algos)
        assert len(results) == 5
        names = [r.algorithm_name for r in results]
        for w in WINNERS:
            assert w in names


class TestEvaluation:
    """AlgorithmEvaluator produces valid reports."""

    @pytest.mark.parametrize("name", WINNERS)
    def test_ic_report_structure(self, name, runner_results):
        evaluator = AlgorithmEvaluator(runner_results[name])
        ic = evaluator.ic_analysis()

        assert "horizons" in ic
        assert "features" in ic
        assert set(ic["horizons"]) == {1, 5, 10, 50, 100}

        alg = get_algorithm(name)
        expected_features = [f.name for f in alg.alg_features()]
        assert set(ic["features"].keys()) == set(expected_features)

        for feat, ics in ic["features"].items():
            for h_key, ic_val in ics.items():
                assert isinstance(ic_val, float), f"{feat}/{h_key} not float"
                if np.isfinite(ic_val):
                    assert -1.0 <= ic_val <= 1.0, f"{feat}/{h_key}={ic_val} out of range"

    @pytest.mark.parametrize("name", WINNERS)
    def test_full_report_structure(self, name, runner_results):
        evaluator = AlgorithmEvaluator(runner_results[name])
        report = evaluator.full_report()

        assert report["algorithm"] == name
        assert report["n_ticks"] == 5000
        assert isinstance(report["warmup_ticks"], int)
        assert isinstance(report["elapsed_s"], float)
        assert "ic" in report
        assert "ic_regime" in report
        # drift present because raw_spread is in base_df
        assert "drift" in report

    @pytest.mark.parametrize("name", WINNERS)
    def test_report_json_serializable(self, name, runner_results):
        evaluator = AlgorithmEvaluator(runner_results[name])
        report = evaluator.full_report()
        serialized = json.dumps(report, default=str)
        assert len(serialized) > 0
        parsed = json.loads(serialized)
        assert parsed["algorithm"] == name


class TestPerformance:
    """Pipeline completes within time budget."""

    def test_evaluation_completes_fast(self, synthetic_df):
        t0 = time.time()
        for name in WINNERS:
            alg = get_algorithm(name)
            runner = AlgorithmRunner(alg)
            result = runner.run_on_dataframe(synthetic_df)
            evaluator = AlgorithmEvaluator(result)
            evaluator.full_report()
        elapsed = time.time() - t0
        assert elapsed < 30.0, f"Pipeline took {elapsed:.1f}s (budget: 30s)"
