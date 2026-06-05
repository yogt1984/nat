"""End-to-end pipeline test: load Parquet -> run algorithms -> evaluate -> JSON report.

Uses the 1-hour BTC fixture from scripts/algorithms/tests/fixtures/btc_1h_real.parquet.
Tests the full chain that matters for production: data loading, algorithm execution,
IC evaluation, and report generation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from algorithms.autodiscover import discover_all  # noqa: E402
from algorithms.registry import get_algorithm  # noqa: E402
from algorithms.runner import AlgorithmRunner  # noqa: E402
from algorithms.evaluate import AlgorithmEvaluator  # noqa: E402

FIXTURE = ROOT / "scripts" / "algorithms" / "tests" / "fixtures" / "btc_1h_real.parquet"

WINNERS = ["jump_detector", "optimal_entry", "funding_reversion",
           "surprise_signal", "weighted_ofi"]


@pytest.fixture(scope="module")
def fixture_df():
    if not FIXTURE.exists():
        pytest.skip(
            f"Fixture not found: {FIXTURE}\n"
            "Run: python scripts/algorithms/tests/extract_fixture.py "
            "--date 2026-06-04 --hour 8"
        )
    discover_all()
    return pd.read_parquet(FIXTURE)


# --- e2e: load -> run -> correct output shape ---

@pytest.mark.parametrize("name", WINNERS)
def test_e2e_run_produces_correct_columns(fixture_df, name):
    """Algorithm output has exactly the expected column names."""
    alg = get_algorithm(name)
    runner = AlgorithmRunner(alg)
    result = runner.run_on_dataframe(fixture_df)

    expected = set(alg.feature_names)
    actual = set(result.features_df.columns)
    assert actual == expected, f"Column mismatch: extra={actual - expected}, missing={expected - actual}"


@pytest.mark.parametrize("name", WINNERS)
def test_e2e_no_100pct_nan_post_warmup(fixture_df, name):
    """No feature column is 100% NaN after warmup."""
    alg = get_algorithm(name)
    runner = AlgorithmRunner(alg)
    result = runner.run_on_dataframe(fixture_df)

    post_warmup = result.features_df.iloc[alg.warmup:]
    for col in post_warmup.columns:
        nan_rate = post_warmup[col].isna().mean()
        assert nan_rate < 1.0, f"{name}.{col} is 100% NaN post-warmup"


# --- e2e: load -> run -> evaluate -> report ---

@pytest.mark.parametrize("name", WINNERS)
def test_e2e_evaluation_report(fixture_df, name):
    """Full evaluation produces a valid JSON report with expected keys."""
    alg = get_algorithm(name)
    runner = AlgorithmRunner(alg)
    result = runner.run_on_dataframe(fixture_df)

    evaluator = AlgorithmEvaluator(result)
    report = evaluator.full_report()

    # Report structure
    assert isinstance(report, dict)
    assert report["algorithm"] == name
    assert report["n_ticks"] == len(fixture_df)
    assert report["warmup_ticks"] == alg.warmup
    assert report["elapsed_s"] >= 0

    # IC section
    assert "ic" in report
    ic = report["ic"]
    assert "horizons" in ic
    assert "features" in ic
    assert len(ic["features"]) == len(alg.alg_features())
    for feat_name, ic_dict in ic["features"].items():
        assert feat_name.startswith("alg_"), f"Feature {feat_name} missing alg_ prefix"
        # Each feature has IC at each horizon
        for h in ic["horizons"]:
            key = f"{h}t"
            assert key in ic_dict, f"Missing IC at horizon {h} for {feat_name}"

    # IC regime section
    assert "ic_regime" in report

    # Report should be JSON-serializable
    json_str = json.dumps(report, default=str)
    assert len(json_str) > 100


def test_e2e_all_winners_complete_under_60s(fixture_df):
    """Running all 5 winners + evaluation should complete in < 60s."""
    start = time.time()

    for name in WINNERS:
        alg = get_algorithm(name)
        runner = AlgorithmRunner(alg)
        result = runner.run_on_dataframe(fixture_df)
        evaluator = AlgorithmEvaluator(result)
        evaluator.full_report()

    elapsed = time.time() - start
    assert elapsed < 60, f"All 5 winners took {elapsed:.1f}s (limit: 60s)"


# --- e2e: IC sanity ---

def test_e2e_ic_values_bounded(fixture_df):
    """IC values should be in [-1, 1] (Spearman correlation)."""
    for name in WINNERS:
        alg = get_algorithm(name)
        runner = AlgorithmRunner(alg)
        result = runner.run_on_dataframe(fixture_df)
        evaluator = AlgorithmEvaluator(result)
        report = evaluator.full_report()

        for feat_name, ic_dict in report["ic"]["features"].items():
            for h_key, ic_val in ic_dict.items():
                if ic_val is not None and not np.isnan(ic_val):
                    assert -1.0 <= ic_val <= 1.0, (
                        f"{name}.{feat_name} IC at {h_key} = {ic_val} (out of [-1,1])"
                    )
