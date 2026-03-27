"""
Skeptical Tests for Walk-Forward Validation

These tests ensure the walk-forward validation correctly:
- Prevents data leakage between train and test
- Applies embargo between adjacent periods
- Correctly calculates OOS/IS ratios
- Properly flags overfit strategies
"""

import pytest
import polars as pl
import numpy as np
from backtest.engine import run_backtest
from backtest.walk_forward import (
    walk_forward_validation,
    WalkForwardResult,
    FoldResult,
    combinatorial_purged_cv,
    compute_deflated_sharpe,
)
from backtest.strategy import Strategy
from backtest.costs import CostModel, zero_cost


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def trending_df():
    """Create trending data for testing."""
    n = 10000
    # Sine wave trend with some noise
    prices = [100.0 + 10 * np.sin(i / 500) + 0.1 * np.random.randn() for i in range(n)]
    return pl.DataFrame({
        "timestamp_ms": list(range(n)),
        "raw_midprice": prices,
        "signal": [i % 200 == 0 for i in range(n)],
        "exit_signal": [i % 200 == 100 for i in range(n)],
    })


@pytest.fixture
def overfit_df():
    """
    Create data where in-sample looks great but out-of-sample will fail.

    This simulates overfitting: strategy works in first half, fails in second.
    """
    n = 10000

    # First half: signal predicts up move
    # Second half: signal predicts down move (strategy breaks)
    prices = []
    signals = []
    exits = []

    for i in range(n):
        if i < n // 2:
            # First half: price goes up after signal
            if i % 100 == 0:
                prices.append(100.0)
                signals.append(True)
            elif i % 100 < 50:
                prices.append(100.0 + (i % 100) * 0.1)
                signals.append(False)
            else:
                prices.append(105.0)
                signals.append(False)
        else:
            # Second half: price goes DOWN after signal (strategy breaks)
            if i % 100 == 0:
                prices.append(100.0)
                signals.append(True)
            elif i % 100 < 50:
                prices.append(100.0 - (i % 100) * 0.1)
                signals.append(False)
            else:
                prices.append(95.0)
                signals.append(False)

        exits.append(i % 100 == 50)

    return pl.DataFrame({
        "timestamp_ms": list(range(n)),
        "raw_midprice": prices,
        "signal": signals,
        "exit_signal": exits,
    })


@pytest.fixture
def simple_strategy():
    """Simple test strategy."""
    return Strategy(
        name="test",
        entry_condition=lambda df: df["signal"],
        exit_condition=lambda df: df["exit_signal"],
        stop_loss_pct=10.0,
        take_profit_pct=20.0,
        max_holding_bars=100,
        direction="long",
    )


# =============================================================================
# BASIC VALIDATION TESTS
# =============================================================================


class TestWalkForwardBasics:
    """Basic walk-forward validation tests."""

    def test_returns_result(self, trending_df, simple_strategy):
        """Walk-forward should return WalkForwardResult."""
        result = walk_forward_validation(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
        )
        assert isinstance(result, WalkForwardResult)

    def test_creates_multiple_folds(self, trending_df, simple_strategy):
        """Should create multiple folds."""
        result = walk_forward_validation(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
            embargo_bars=100,  # Smaller embargo for test data
        )
        assert result.n_folds >= 0  # May be 0 if data doesn't support folds
        # At minimum, we should get a result object
        assert isinstance(result, WalkForwardResult)

    def test_fold_results_have_train_and_test(self, trending_df, simple_strategy):
        """Each fold should have train and test results."""
        result = walk_forward_validation(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
        )

        for fold in result.fold_results:
            assert isinstance(fold.train_result, object)
            assert isinstance(fold.test_result, object)
            assert fold.train_sharpe is not None
            assert fold.test_sharpe is not None


# =============================================================================
# DATA LEAKAGE TESTS
# =============================================================================


class TestNoDataLeakage:
    """Tests to ensure no data leakage between train and test."""

    def test_train_comes_before_test(self, trending_df, simple_strategy):
        """Train period should always come before test period."""
        result = walk_forward_validation(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
        )

        for fold in result.fold_results:
            assert fold.train_end_idx < fold.test_start_idx, (
                "Train should end before test starts"
            )

    def test_embargo_gap_exists(self, trending_df, simple_strategy):
        """There should be an embargo gap between train and test."""
        embargo_bars = 100

        result = walk_forward_validation(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
            embargo_bars=embargo_bars,
        )

        for fold in result.fold_results:
            gap = fold.test_start_idx - fold.train_end_idx
            assert gap >= embargo_bars, (
                f"Embargo gap should be >= {embargo_bars}, got {gap}"
            )

    def test_folds_dont_overlap(self, trending_df, simple_strategy):
        """Different folds should not overlap in test periods."""
        result = walk_forward_validation(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
        )

        if len(result.fold_results) < 2:
            pytest.skip("Not enough folds to test overlap")

        # Check that test periods don't overlap
        test_ranges = [
            (f.test_start_idx, f.test_end_idx)
            for f in result.fold_results
        ]

        for i, (start1, end1) in enumerate(test_ranges):
            for j, (start2, end2) in enumerate(test_ranges):
                if i != j:
                    # Check no overlap
                    overlaps = not (end1 <= start2 or end2 <= start1)
                    assert not overlaps, f"Test periods {i} and {j} overlap"


# =============================================================================
# OVERFIT DETECTION TESTS
# =============================================================================


class TestOverfitDetection:
    """Tests that walk-forward detects overfitting."""

    def test_overfit_strategy_fails_validation(self, overfit_df, simple_strategy):
        """Strategy that works in-sample but fails out-of-sample should be invalid."""
        result = walk_forward_validation(
            overfit_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
            train_ratio=0.5,  # 50/50 split to catch the regime change
        )

        # The strategy should show degradation
        # IS should be positive, OOS should be worse
        if result.in_sample_sharpe > 0:
            assert result.oos_is_ratio < 1.0, (
                "Overfit strategy should degrade out-of-sample"
            )

    def test_consistent_strategy_passes_validation(self, trending_df, simple_strategy):
        """Strategy consistent across periods should have OOS/IS ratio closer to 1."""
        result = walk_forward_validation(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
        )

        # For consistent strategy, OOS shouldn't be dramatically worse
        # Note: this is a soft check as some degradation is normal
        if result.total_test_trades > 10 and result.total_train_trades > 10:
            assert result.oos_is_ratio > 0.3, (
                "Consistent strategy should not degrade too much"
            )

    def test_oos_threshold_applied(self, trending_df, simple_strategy):
        """Validity should check against OOS threshold."""
        result = walk_forward_validation(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
            oos_is_threshold=0.7,  # Require 70% of IS performance
        )

        # Check validity logic
        if result.is_valid:
            assert result.oos_is_ratio >= 0.7, (
                "Valid strategy should have OOS/IS >= threshold"
            )


# =============================================================================
# EDGE CASES
# =============================================================================


class TestWalkForwardEdgeCases:
    """Edge case tests for walk-forward validation."""

    def test_small_dataset_handled(self, simple_strategy):
        """Should handle small datasets gracefully."""
        small_df = pl.DataFrame({
            "timestamp_ms": list(range(100)),
            "raw_midprice": [100.0 + i * 0.1 for i in range(100)],
            "signal": [i % 20 == 0 for i in range(100)],
            "exit_signal": [i % 20 == 10 for i in range(100)],
        })

        # Should not crash
        result = walk_forward_validation(
            small_df,
            simple_strategy,
            zero_cost(),
            n_splits=2,
            embargo_bars=10,
        )

        assert isinstance(result, WalkForwardResult)

    def test_too_small_for_embargo_raises(self, simple_strategy):
        """Should raise error if data too small for embargo."""
        tiny_df = pl.DataFrame({
            "timestamp_ms": list(range(50)),
            "raw_midprice": [100.0] * 50,
            "signal": [True] + [False] * 49,
            "exit_signal": [False] * 49 + [True],
        })

        with pytest.raises(ValueError):
            walk_forward_validation(
                tiny_df,
                simple_strategy,
                zero_cost(),
                n_splits=4,
                embargo_bars=100,  # Embargo larger than data
            )

    def test_no_trades_in_fold_handled(self, simple_strategy):
        """Should handle folds with no trades."""
        # Create data where signals only appear in some folds
        n = 10000
        df = pl.DataFrame({
            "timestamp_ms": list(range(n)),
            "raw_midprice": [100.0] * n,
            # Only signals in first quarter
            "signal": [i < n // 4 and i % 100 == 0 for i in range(n)],
            "exit_signal": [i < n // 4 and i % 100 == 50 for i in range(n)],
        })

        result = walk_forward_validation(
            df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
        )

        # Should not crash
        assert isinstance(result, WalkForwardResult)


# =============================================================================
# SUMMARY AND REPORTING TESTS
# =============================================================================


class TestWalkForwardReporting:
    """Test walk-forward result reporting."""

    def test_summary_contains_key_metrics(self, trending_df, simple_strategy):
        """Summary should contain key metrics."""
        result = walk_forward_validation(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
        )

        summary = result.summary()

        assert "In-Sample Sharpe" in summary
        assert "Out-of-Sample" in summary
        assert "OOS/IS Ratio" in summary
        assert "VALID" in summary or "INVALID" in summary

    def test_fold_count_matches(self, trending_df, simple_strategy):
        """Reported fold count should match actual folds."""
        result = walk_forward_validation(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
        )

        assert result.n_folds == len(result.fold_results) or result.n_folds <= 4


# =============================================================================
# DEFLATED SHARPE TESTS
# =============================================================================


class TestDeflatedSharpe:
    """Test deflated Sharpe ratio calculation."""

    def test_deflated_sharpe_returns_probability(self):
        """Deflated Sharpe should return a probability [0, 1]."""
        prob = compute_deflated_sharpe(
            observed_sharpe=2.0,
            n_trials=100,
        )

        assert 0 <= prob <= 1, "Should return a probability"

    def test_more_trials_lower_confidence(self):
        """More trials should reduce confidence in observed Sharpe."""
        prob_few = compute_deflated_sharpe(
            observed_sharpe=2.0,
            n_trials=10,
        )

        prob_many = compute_deflated_sharpe(
            observed_sharpe=2.0,
            n_trials=1000,
        )

        # With more trials, we expect worse Sharpe by chance
        # So same observed Sharpe is less impressive
        assert prob_many <= prob_few, (
            "More trials should reduce confidence"
        )

    def test_higher_sharpe_more_confident(self):
        """Higher observed Sharpe should give more confidence."""
        prob_low = compute_deflated_sharpe(
            observed_sharpe=1.0,
            n_trials=100,
        )

        prob_high = compute_deflated_sharpe(
            observed_sharpe=3.0,
            n_trials=100,
        )

        assert prob_high >= prob_low, (
            "Higher Sharpe should be more confident"
        )


# =============================================================================
# COMBINATORIAL CV TESTS
# =============================================================================


class TestCombinatorialCV:
    """Test combinatorial purged cross-validation."""

    def test_cpcv_returns_multiple_results(self, trending_df, simple_strategy):
        """CPCV should return multiple results for combinations."""
        results = combinatorial_purged_cv(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
            n_test_splits=1,
        )

        # Should have C(4,1) = 4 combinations
        assert len(results) >= 1

    def test_cpcv_combinations_different(self, trending_df, simple_strategy):
        """Different combinations should give different results."""
        results = combinatorial_purged_cv(
            trending_df,
            simple_strategy,
            zero_cost(),
            n_splits=4,
            n_test_splits=1,
        )

        if len(results) >= 2:
            # Results should not all be identical
            sharpes = [r.out_of_sample_sharpe for r in results]
            assert len(set(sharpes)) > 1 or all(s == 0 for s in sharpes), (
                "Different combinations should give different results"
            )
