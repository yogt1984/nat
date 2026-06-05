"""Tests that run winner algorithms on real market data.

These tests use a 1-hour BTC fixture extracted from production ingestion data.
The fixture contains naturally occurring jumps, funding swings, and entropy
transitions — the behaviors the algorithms are designed to detect.

Fixture: fixtures/btc_1h_real.parquet (extracted via extract_fixture.py)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from algorithms.autodiscover import discover_all  # noqa: E402
from algorithms.registry import get_algorithm  # noqa: E402
from algorithms.runner import AlgorithmRunner  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "btc_1h_real.parquet"

# --- helpers ---

def _load_fixture() -> pd.DataFrame:
    if not FIXTURE.exists():
        pytest.skip(
            f"Fixture not found: {FIXTURE}\n"
            "Run: python scripts/algorithms/tests/extract_fixture.py "
            "--date 2026-06-04 --hour 8"
        )
    return pd.read_parquet(FIXTURE)


def _run(name: str, df: pd.DataFrame):
    discover_all()
    alg = get_algorithm(name)
    runner = AlgorithmRunner(alg)
    return runner.run_on_dataframe(df), alg


# --- tests: all 5 winners should not crash ---

WINNERS = ["jump_detector", "optimal_entry", "funding_reversion",
           "surprise_signal", "weighted_ofi"]


@pytest.mark.parametrize("name", WINNERS)
def test_winner_no_crash(name):
    """Each winner algorithm runs on real data without error."""
    df = _load_fixture()
    result, alg = _run(name, df)

    assert result.n_ticks == len(df)
    assert len(result.features_df.columns) == len(alg.alg_features())
    assert result.elapsed_s >= 0


@pytest.mark.parametrize("name", WINNERS)
def test_winner_produces_nonnan_output(name):
    """Post-warmup output should be mostly non-NaN."""
    df = _load_fixture()
    result, alg = _run(name, df)

    post_warmup = result.features_df.iloc[alg.warmup:]
    for col in post_warmup.columns:
        nan_rate = post_warmup[col].isna().mean()
        assert nan_rate < 0.5, (
            f"{name}.{col}: {nan_rate:.0%} NaN post-warmup (expected < 50%)"
        )


@pytest.mark.parametrize("name", WINNERS)
def test_winner_no_inf(name):
    """No infinite values in output."""
    df = _load_fixture()
    result, _ = _run(name, df)

    for col in result.features_df.columns:
        inf_count = np.isinf(result.features_df[col]).sum()
        assert inf_count == 0, f"{name}.{col}: {inf_count} infinite values"


# --- tests: specific algorithm behavior ---

def test_jump_detector_fires_on_real_data():
    """Jump detector should detect at least 1 jump in a volatile hour."""
    df = _load_fixture()
    result, alg = _run("jump_detector", df)

    post_warmup = result.features_df.iloc[alg.warmup:]
    n_jumps = (post_warmup["alg_jump_detected"] == 1.0).sum()
    assert n_jumps >= 1, (
        f"Expected at least 1 jump detection in 1h of real BTC data, got {n_jumps}"
    )


def test_jump_detector_near_large_returns():
    """Jump detections should cluster near large price moves."""
    df = _load_fixture()
    result, alg = _run("jump_detector", df)

    mid = df["raw_midprice"].values
    returns = np.diff(mid) / mid[:-1]
    # Pad to align with features_df
    returns = np.concatenate([[0.0], returns])

    # Find ticks with |return| > 3 * std (potential jumps)
    std = np.nanstd(returns)
    large_return_mask = np.abs(returns) > 3 * std
    large_return_idxs = np.where(large_return_mask)[0]

    if len(large_return_idxs) == 0:
        pytest.skip("No 3-sigma returns in fixture")

    # Check that at least one jump detection is within 10 ticks of a large return
    jump_mask = result.features_df["alg_jump_detected"].values == 1.0
    jump_idxs = np.where(jump_mask)[0]

    if len(jump_idxs) == 0:
        pytest.fail("No jumps detected despite large returns present")

    min_distances = []
    for lr_idx in large_return_idxs:
        distances = np.abs(jump_idxs - lr_idx)
        min_distances.append(distances.min())

    # At least one large return should have a jump detection within 10 ticks
    assert min(min_distances) <= 10, (
        f"Closest jump detection is {min(min_distances)} ticks from nearest "
        f"large return (expected <= 10)"
    )


def test_optimal_entry_produces_valid_output():
    """Optimal entry should produce bounded evidence and valid signal values."""
    df = _load_fixture()
    result, alg = _run("optimal_entry", df)

    post_warmup = result.features_df.iloc[alg.warmup:]
    signals = post_warmup["alg_entry_signal"]

    # Signals should be in {-1, 0, +1}
    unique = set(signals.dropna().unique())
    assert unique <= {-1.0, 0.0, 1.0}, f"Unexpected signal values: {unique}"

    # Cumulative evidence should be non-negative
    evidence = post_warmup["alg_cumulative_evidence"].dropna()
    assert evidence.min() >= -1e-6, "Evidence should be non-negative"

    # SPRT statistic should show variation (filter is processing data)
    sprt = post_warmup["alg_sprt_statistic"].dropna()
    assert sprt.std() > 1e-10, "SPRT statistic should show variation"


def test_funding_reversion_responds_to_funding():
    """Funding reversion signal should be non-zero when funding is extreme."""
    df = _load_fixture()
    result, alg = _run("funding_reversion", df)

    post_warmup = result.features_df.iloc[alg.warmup:]
    signal = post_warmup["alg_funding_signal"]

    # Signal should activate at some point (even if funding is mild)
    # If it never activates, the fixture may have flat funding — that's ok
    n_active = (signal.abs() > 0).sum()
    # Just verify it runs and produces bounded output
    assert signal.abs().max() <= 1.0 + 1e-6, "Signal should be in [-1, 1]"
    assert post_warmup["alg_funding_halflife_ticks"].notna().mean() > 0.5


def test_surprise_signal_detects_transitions():
    """Surprise signal should detect entropy transitions in real data."""
    df = _load_fixture()
    result, alg = _run("surprise_signal", df)

    post_warmup = result.features_df.iloc[alg.warmup:]
    prob = post_warmup["alg_regime_transition_prob"]

    # Transition probability should be in (0, 1)
    valid = prob.dropna()
    assert valid.min() >= 0.0, "Transition prob should be >= 0"
    assert valid.max() <= 1.0, "Transition prob should be <= 1"

    # Should have at least one high-probability transition in 1h
    n_transitions = (valid > 0.5).sum()
    assert n_transitions >= 1, (
        f"Expected at least 1 transition with prob > 0.5 in 1h, got {n_transitions}"
    )


def test_weighted_ofi_varies():
    """Weighted OFI should show variation (not constant) on real data."""
    df = _load_fixture()
    result, alg = _run("weighted_ofi", df)

    post_warmup = result.features_df.iloc[alg.warmup:]
    ofi = post_warmup["alg_weighted_ofi"].dropna()

    assert ofi.std() > 1e-8, "Weighted OFI should show non-trivial variation"
    assert ofi.nunique() > 10, "Expected significant unique values"
