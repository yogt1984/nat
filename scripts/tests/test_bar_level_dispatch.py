"""Regression tests for bar_level dispatch in AlgorithmRunner."""

import numpy as np
import pandas as pd
import pytest

from algorithms.base import AlgorithmFeature, MicrostructureAlgorithm
from algorithms.runner import AlgorithmRunner


# --- Stub algorithms ---

class StubTickAlgo(MicrostructureAlgorithm):
    """Tick-level stub (bar_level=False, the default)."""

    def name(self):
        return "stub_tick"

    def alg_features(self):
        return [AlgorithmFeature("alg_stub_tick_signal")]

    def required_columns(self):
        return ["col_a"]

    def step(self, tick):
        return {"alg_stub_tick_signal": tick.get("col_a", 0.0) * 2}

    def reset(self):
        pass


class StubBarAlgo(MicrostructureAlgorithm):
    """Bar-level stub (bar_level=True)."""
    bar_level = True
    bar_timeframe = "5min"

    def name(self):
        return "stub_bar"

    def alg_features(self):
        return [AlgorithmFeature("alg_stub_bar_signal")]

    def required_columns(self):
        return ["col_a_mean"]

    def step(self, tick):
        return {"alg_stub_bar_signal": tick.get("col_a_mean", 0.0) + 1}

    def reset(self):
        pass


def _make_tick_df(n: int = 1000) -> pd.DataFrame:
    """Create a minimal tick DataFrame with timestamp_ns."""
    rng = np.random.default_rng(42)
    t0 = 1_700_000_000_000_000_000  # nanosecond timestamp
    return pd.DataFrame({
        "timestamp_ns": t0 + np.arange(n) * 100_000_000,  # 100ms apart
        "symbol": "BTC",
        "col_a": rng.normal(0, 1, n),
    })


# --- Tests ---

def test_tick_algo_receives_raw_df():
    """A tick-level algorithm (bar_level=False) receives the original DataFrame unchanged."""
    df = _make_tick_df(500)
    runner = AlgorithmRunner(StubTickAlgo())
    result = runner.run_on_dataframe(df)
    # Output should have same length as input
    assert len(result.features_df) == len(df)
    # Values should be 2x col_a (from our stub)
    expected = df["col_a"].values * 2
    # Skip warmup (0 here) — values should match
    np.testing.assert_allclose(result.features_df["alg_stub_tick_signal"].values, expected)


def test_bar_algo_receives_aggregated_bars():
    """A bar-level algorithm (bar_level=True) receives aggregated bars with _mean suffixes."""
    df = _make_tick_df(3000)  # enough ticks for several 5-min bars
    runner = AlgorithmRunner(StubBarAlgo())
    result = runner.run_on_dataframe(df)
    # Output should be forward-filled to tick-level length
    assert len(result.features_df) == len(df)
    # Should have our output column
    assert "alg_stub_bar_signal" in result.features_df.columns


def test_bar_algo_output_ffilled_to_tick_index():
    """Bar-level algorithm output is forward-filled to match tick-level index length."""
    df = _make_tick_df(3000)
    runner = AlgorithmRunner(StubBarAlgo())
    result = runner.run_on_dataframe(df)
    signal = result.features_df["alg_stub_bar_signal"]
    # Forward-filled means consecutive ticks within same bar have same value
    # Check that we don't have all-unique values (would mean no ffill happened)
    unique_count = signal.dropna().nunique()
    assert unique_count < len(df), "Expected forward-filled (repeated) values"


def test_existing_tick_algos_unaffected():
    """Run existing tick-level algorithms — they should work through the runner without error."""
    from algorithms.autodiscover import discover_all
    from algorithms.registry import get_algorithm

    discover_all()

    # Pick 3 well-known tick-level algorithms
    for name in ["weighted_ofi", "jump_detector", "hawkes_intensity"]:
        algo = get_algorithm(name)
        assert algo.bar_level is False, f"{name} should be tick-level"

        # Create synthetic data with required columns
        required = algo.required_columns()
        rng = np.random.default_rng(42)
        n = 500
        data = {col: rng.normal(0, 1, n) for col in required}
        # Add midprice for algorithms that need it
        if "raw_midprice" not in data:
            data["raw_midprice"] = 50000 + rng.normal(0, 10, n).cumsum()
        df = pd.DataFrame(data)

        runner = AlgorithmRunner(algo)
        result = runner.run_on_dataframe(df)
        assert len(result.features_df) == n, f"{name} output length mismatch"
        assert result.features_df.shape[1] == len(algo.alg_features()), f"{name} column count mismatch"
