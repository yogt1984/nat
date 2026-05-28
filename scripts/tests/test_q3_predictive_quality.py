"""Tests for q3_predictive_quality.py — forward return computation and Q3 config."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from q3_predictive_quality import compute_forward_returns, CONFIGS, Q3_THRESHOLDS


# ---------------------------------------------------------------------------
# compute_forward_returns
# ---------------------------------------------------------------------------

class TestComputeForwardReturns:
    def test_basic_log_returns(self):
        prices = pd.Series([100.0, 110.0, 105.0, 120.0, 115.0])
        df = pd.DataFrame({"raw_midprice_mean": prices})
        result = compute_forward_returns(df, "5min")

        # Returns np.ndarray of log returns
        expected = np.log(110.0 / 100.0)
        assert result[0] == pytest.approx(expected, rel=1e-10)

    def test_last_row_is_nan(self):
        df = pd.DataFrame({"raw_midprice_mean": [100.0, 105.0, 110.0]})
        result = compute_forward_returns(df, "5min")
        assert np.isnan(result[-1])

    def test_length_preserved(self):
        df = pd.DataFrame({"raw_midprice_mean": np.arange(1.0, 51.0)})
        result = compute_forward_returns(df, "15min")
        assert len(result) == 50

    def test_negative_returns(self):
        df = pd.DataFrame({"raw_midprice_mean": [100.0, 90.0, 80.0]})
        result = compute_forward_returns(df, "5min")
        assert result[0] < 0  # price dropped

    def test_constant_price_zero_return(self):
        df = pd.DataFrame({"raw_midprice_mean": [100.0, 100.0, 100.0]})
        result = compute_forward_returns(df, "5min")
        assert result[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CONFIGS structure
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_configs_is_list(self):
        assert isinstance(CONFIGS, list)
        assert len(CONFIGS) > 0

    def test_each_config_has_required_keys(self):
        required = {"k", "vector", "timeframe"}
        for cfg in CONFIGS:
            assert required.issubset(cfg.keys()), f"Config missing keys: {cfg}"

    def test_k_values_are_positive_ints(self):
        for cfg in CONFIGS:
            assert isinstance(cfg["k"], int)
            assert cfg["k"] >= 2

    def test_vector_values_are_strings(self):
        for cfg in CONFIGS:
            assert isinstance(cfg["vector"], str)
            assert len(cfg["vector"]) > 0

    def test_timeframe_values_are_strings(self):
        for cfg in CONFIGS:
            assert isinstance(cfg["timeframe"], str)
            assert len(cfg["timeframe"]) > 0


# ---------------------------------------------------------------------------
# Q3_THRESHOLDS
# ---------------------------------------------------------------------------

class TestQ3Thresholds:
    def test_has_required_keys(self):
        assert "kruskal_wallis_p" in Q3_THRESHOLDS
        assert "eta_squared" in Q3_THRESHOLDS
        assert "self_transition_rate" in Q3_THRESHOLDS

    def test_values_in_range(self):
        assert 0 < Q3_THRESHOLDS["kruskal_wallis_p"] < 1
        assert 0 < Q3_THRESHOLDS["eta_squared"] < 1
        assert 0 < Q3_THRESHOLDS["self_transition_rate"] <= 1
