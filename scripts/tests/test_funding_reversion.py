"""
Tests for FundingReversion strategy.

Covers:
  - Signal direction (short on positive funding, long on negative)
  - Z-score thresholds (entry/exit behavior)
  - Position sizing (linear scaling)
  - Edge cases (NaN handling, missing columns, fallback z-score)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.base import Strategy, StrategyMeta
from strategies.funding_reversion import FundingReversion


class TestFundingReversionBasic:
    """Basic interface compliance."""

    def test_implements_strategy_interface(self):
        s = FundingReversion()
        assert isinstance(s, Strategy)
        assert hasattr(s, "meta")
        assert isinstance(s.meta, StrategyMeta)

    def test_warmup_bars_raw_zscore(self):
        s = FundingReversion(use_raw_zscore=True)
        assert s.warmup_bars() == 1

    def test_warmup_bars_rolling(self):
        s = FundingReversion(use_raw_zscore=False, lookback=96)
        assert s.warmup_bars() == 96

    def test_meta_fields(self):
        s = FundingReversion()
        assert s.meta.name == "funding_reversion"
        assert "ctx_funding_rate" in s.meta.required_columns


class TestSignalDirection:
    """Core thesis: short when funding positive, long when negative."""

    def _make_bars(self, funding_rates, zscores):
        return pd.DataFrame({
            "ctx_funding_rate": funding_rates,
            "ctx_funding_zscore": zscores,
        })

    def test_short_on_high_positive_funding(self):
        """Positive funding = too many longs = price should fall = short."""
        s = FundingReversion(zscore_entry=2.0)
        bars = self._make_bars([0.001] * 5, [3.0] * 5)
        features = s.compute_features(bars)
        signal = s.generate_signal(features)
        assert all(signal < 0), f"Expected negative signal, got {signal.values}"

    def test_long_on_high_negative_funding(self):
        """Negative funding = too many shorts = price should rise = long."""
        s = FundingReversion(zscore_entry=2.0)
        bars = self._make_bars([-0.001] * 5, [-3.0] * 5)
        features = s.compute_features(bars)
        signal = s.generate_signal(features)
        assert all(signal > 0), f"Expected positive signal, got {signal.values}"

    def test_flat_below_exit(self):
        """Z-score below exit threshold = no position."""
        s = FundingReversion(zscore_entry=2.0, zscore_exit=0.5)
        bars = self._make_bars([0.0001] * 5, [0.2] * 5)
        features = s.compute_features(bars)
        signal = s.generate_signal(features)
        assert all(signal == 0.0)

    def test_flat_between_exit_and_entry(self):
        """Z-score between exit and entry = hold (0 for simplicity)."""
        s = FundingReversion(zscore_entry=2.0, zscore_exit=0.5)
        bars = self._make_bars([0.0005] * 5, [1.0] * 5)
        features = s.compute_features(bars)
        signal = s.generate_signal(features)
        assert all(signal == 0.0)


class TestPositionSizing:
    """Position scales linearly from 0 at entry to max at 2*entry."""

    def test_at_entry_threshold_size_is_zero(self):
        """Exactly at entry: intensity = 0."""
        s = FundingReversion(zscore_entry=2.0, max_position=1.0)
        bars = pd.DataFrame({
            "ctx_funding_rate": [0.001],
            "ctx_funding_zscore": [2.0],
        })
        features = s.compute_features(bars)
        signal = s.generate_signal(features)
        # At exactly entry, (2.0 - 2.0) / 2.0 = 0
        assert signal.iloc[0] == 0.0

    def test_at_double_entry_size_is_max(self):
        """At 2*entry: intensity = 1.0, size = max_position."""
        s = FundingReversion(zscore_entry=2.0, max_position=0.8)
        bars = pd.DataFrame({
            "ctx_funding_rate": [0.001],
            "ctx_funding_zscore": [4.0],
        })
        features = s.compute_features(bars)
        signal = s.generate_signal(features)
        assert signal.iloc[0] == pytest.approx(-0.8)

    def test_midpoint_scaling(self):
        """At 1.5*entry: intensity = 0.5."""
        s = FundingReversion(zscore_entry=2.0, max_position=1.0)
        bars = pd.DataFrame({
            "ctx_funding_rate": [0.001],
            "ctx_funding_zscore": [3.0],  # 1.5 * entry
        })
        features = s.compute_features(bars)
        signal = s.generate_signal(features)
        # (3.0 - 2.0) / 2.0 = 0.5, size = 0.5, direction = -1
        assert signal.iloc[0] == pytest.approx(-0.5)

    def test_beyond_double_entry_caps_at_max(self):
        """Beyond 2*entry: capped at max_position."""
        s = FundingReversion(zscore_entry=2.0, max_position=1.0)
        bars = pd.DataFrame({
            "ctx_funding_rate": [-0.01],
            "ctx_funding_zscore": [-10.0],
        })
        features = s.compute_features(bars)
        signal = s.generate_signal(features)
        assert signal.iloc[0] == pytest.approx(1.0)


class TestEdgeCases:
    """NaN handling, missing columns, rolling fallback."""

    def test_nan_zscore_produces_nan_signal(self):
        s = FundingReversion()
        bars = pd.DataFrame({
            "ctx_funding_rate": [0.001, np.nan, 0.001],
            "ctx_funding_zscore": [3.0, np.nan, 3.0],
        })
        features = s.compute_features(bars)
        signal = s.generate_signal(features)
        assert np.isnan(signal.iloc[1])
        assert not np.isnan(signal.iloc[0])

    def test_missing_funding_column(self):
        """No funding columns → all NaN features."""
        s = FundingReversion()
        bars = pd.DataFrame({"close": [100, 101, 102]})
        features = s.compute_features(bars)
        assert "fr_zscore" in features.columns
        assert all(features["fr_zscore"].isna())

    def test_rolling_zscore_fallback(self):
        """When use_raw_zscore=False, computes rolling z-score."""
        n = 200
        np.random.seed(42)
        funding = np.random.normal(0, 0.001, n)
        # Inject extreme value at the end
        funding[-1] = 0.01  # very high

        s = FundingReversion(use_raw_zscore=False, lookback=50, zscore_entry=2.0)
        bars = pd.DataFrame({"ctx_funding_rate": funding})
        features = s.compute_features(bars)

        # Should have computed z-score (not NaN after warmup)
        assert not np.isnan(features["fr_zscore"].iloc[-1])
        # Last value should be high positive z-score
        assert features["fr_zscore"].iloc[-1] > 2.0

    def test_uses_mean_column_variant(self):
        """Handles ctx_funding_rate_mean from aggregated bars."""
        s = FundingReversion()
        bars = pd.DataFrame({
            "ctx_funding_rate_mean": [0.001] * 5,
            "ctx_funding_zscore_mean": [3.0] * 5,
        })
        features = s.compute_features(bars)
        signal = s.generate_signal(features)
        assert all(signal < 0)


class TestSignalBounds:
    """Signal always within [-1, 1]."""

    def test_signal_bounded(self):
        n = 100
        np.random.seed(123)
        zscores = np.random.uniform(-20, 20, n)
        rates = np.sign(zscores) * 0.001

        s = FundingReversion(zscore_entry=2.0, max_position=1.0)
        bars = pd.DataFrame({
            "ctx_funding_rate": rates,
            "ctx_funding_zscore": zscores,
        })
        features = s.compute_features(bars)
        signal = s.generate_signal(features)

        valid = signal.dropna()
        assert all(valid >= -1.0)
        assert all(valid <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
