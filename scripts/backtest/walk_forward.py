"""
Walk-Forward Validation for NAT Backtester

Implements purged walk-forward cross-validation to prevent overfitting.
"""

import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import List, Tuple
from .engine import run_backtest, BacktestResult
from .strategy import Strategy
from .costs import CostModel


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_idx: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_result: BacktestResult
    test_result: BacktestResult

    @property
    def train_sharpe(self) -> float:
        return self.train_result.sharpe_ratio

    @property
    def test_sharpe(self) -> float:
        return self.test_result.sharpe_ratio

    @property
    def degradation(self) -> float:
        """How much worse is test vs train. Lower is better."""
        if self.train_sharpe <= 0:
            return float("inf")
        return 1 - (self.test_sharpe / self.train_sharpe)


@dataclass
class WalkForwardResult:
    """
    Complete walk-forward validation results.

    Attributes
    ----------
    strategy_name : str
        Name of strategy tested
    fold_results : List[FoldResult]
        Results from each fold
    in_sample_sharpe : float
        Average Sharpe across all training sets
    out_of_sample_sharpe : float
        Average Sharpe across all test sets
    oos_is_ratio : float
        OOS Sharpe / IS Sharpe - should be > 0.7 for valid strategy
    is_valid : bool
        Whether strategy passes validation criteria
    n_folds : int
        Number of folds used
    embargo_bars : int
        Gap between train and test to prevent leakage
    """

    strategy_name: str
    fold_results: List[FoldResult]
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    oos_is_ratio: float
    is_valid: bool
    n_folds: int
    embargo_bars: int
    total_train_trades: int = 0
    total_test_trades: int = 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"{'='*60}",
            f"WALK-FORWARD VALIDATION: {self.strategy_name}",
            f"{'='*60}",
            f"Status:            {status}",
            f"Folds:             {self.n_folds}",
            f"Embargo:           {self.embargo_bars} bars",
            f"",
            f"In-Sample Sharpe:  {self.in_sample_sharpe:.2f}",
            f"Out-of-Sample:     {self.out_of_sample_sharpe:.2f}",
            f"OOS/IS Ratio:      {self.oos_is_ratio:.2%}",
            f"",
            f"Total Train Trades: {self.total_train_trades}",
            f"Total Test Trades:  {self.total_test_trades}",
            f"{'='*60}",
        ]

        # Add per-fold breakdown
        lines.append("\nPer-Fold Results:")
        lines.append("-" * 50)
        for f in self.fold_results:
            lines.append(
                f"  Fold {f.fold_idx}: "
                f"Train={f.train_sharpe:.2f} ({f.train_result.total_trades} trades), "
                f"Test={f.test_sharpe:.2f} ({f.test_result.total_trades} trades)"
            )

        return "\n".join(lines)


def walk_forward_validation(
    df: pl.DataFrame,
    strategy: Strategy,
    cost_model: CostModel,
    n_splits: int = 4,
    train_ratio: float = 0.75,
    embargo_bars: int = 600,
    min_trades_per_fold: int = 5,
    oos_is_threshold: float = 0.7,
    min_oos_sharpe: float = 0.3,
) -> WalkForwardResult:
    """
    Purged walk-forward cross-validation.

    Splits data into n_splits folds. For each fold:
    - Train (evaluate) on train_ratio of the fold
    - Test on remaining portion (after embargo gap)
    - No data leakage between folds

    Parameters
    ----------
    df : pl.DataFrame
        Feature data
    strategy : Strategy
        Strategy to validate
    cost_model : CostModel
        Transaction costs
    n_splits : int
        Number of folds
    train_ratio : float
        Fraction of each fold for training
    embargo_bars : int
        Gap between train and test to prevent leakage
    min_trades_per_fold : int
        Minimum trades required for fold to count
    oos_is_threshold : float
        Required OOS/IS ratio for validity
    min_oos_sharpe : float
        Minimum OOS Sharpe for validity

    Returns
    -------
    WalkForwardResult
        Validation results with pass/fail determination
    """
    n = len(df)
    fold_size = n // n_splits

    if fold_size < embargo_bars * 2:
        raise ValueError(
            f"Fold size ({fold_size}) too small for embargo ({embargo_bars})"
        )

    fold_results = []
    total_train_trades = 0
    total_test_trades = 0

    for i in range(n_splits):
        fold_start = i * fold_size
        fold_end = min((i + 1) * fold_size, n)

        fold_df = df.slice(fold_start, fold_end - fold_start)
        fold_n = len(fold_df)

        train_end = int(fold_n * train_ratio)
        test_start = train_end + embargo_bars

        if test_start >= fold_n:
            # Not enough data for test set
            continue

        train_df = fold_df.slice(0, train_end)
        test_df = fold_df.slice(test_start, fold_n - test_start)

        if len(train_df) < 100 or len(test_df) < 100:
            continue

        # Run backtest on train (in-sample)
        try:
            train_result = run_backtest(train_df, strategy, cost_model)
        except Exception as e:
            print(f"Warning: Fold {i} train failed: {e}")
            continue

        # Run backtest on test (out-of-sample)
        try:
            test_result = run_backtest(test_df, strategy, cost_model)
        except Exception as e:
            print(f"Warning: Fold {i} test failed: {e}")
            continue

        fold_results.append(
            FoldResult(
                fold_idx=i,
                train_start_idx=fold_start,
                train_end_idx=fold_start + train_end,
                test_start_idx=fold_start + test_start,
                test_end_idx=fold_end,
                train_result=train_result,
                test_result=test_result,
            )
        )

        total_train_trades += train_result.total_trades
        total_test_trades += test_result.total_trades

    # Aggregate metrics
    if not fold_results:
        return WalkForwardResult(
            strategy_name=strategy.name,
            fold_results=[],
            in_sample_sharpe=0.0,
            out_of_sample_sharpe=0.0,
            oos_is_ratio=0.0,
            is_valid=False,
            n_folds=0,
            embargo_bars=embargo_bars,
        )

    # Filter folds with enough trades
    valid_folds = [
        f
        for f in fold_results
        if f.train_result.total_trades >= min_trades_per_fold
        and f.test_result.total_trades >= min_trades_per_fold
    ]

    if not valid_folds:
        # Use all folds even if trade count is low
        valid_folds = fold_results

    # Compute average Sharpe ratios
    train_sharpes = [f.train_sharpe for f in valid_folds]
    test_sharpes = [f.test_sharpe for f in valid_folds]

    in_sample_sharpe = np.mean(train_sharpes) if train_sharpes else 0
    out_of_sample_sharpe = np.mean(test_sharpes) if test_sharpes else 0

    # OOS/IS ratio
    if in_sample_sharpe > 0:
        oos_is_ratio = out_of_sample_sharpe / in_sample_sharpe
    else:
        oos_is_ratio = 0.0

    # Validity check
    is_valid = (
        oos_is_ratio >= oos_is_threshold
        and out_of_sample_sharpe >= min_oos_sharpe
        and total_test_trades >= n_splits * min_trades_per_fold
    )

    return WalkForwardResult(
        strategy_name=strategy.name,
        fold_results=fold_results,
        in_sample_sharpe=in_sample_sharpe,
        out_of_sample_sharpe=out_of_sample_sharpe,
        oos_is_ratio=oos_is_ratio,
        is_valid=is_valid,
        n_folds=len(valid_folds),
        embargo_bars=embargo_bars,
        total_train_trades=total_train_trades,
        total_test_trades=total_test_trades,
    )


def combinatorial_purged_cv(
    df: pl.DataFrame,
    strategy: Strategy,
    cost_model: CostModel,
    n_splits: int = 5,
    n_test_splits: int = 2,
    embargo_bars: int = 600,
) -> List[WalkForwardResult]:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    More rigorous than simple walk-forward. Tests all combinations
    of train/test splits.

    Parameters
    ----------
    df : pl.DataFrame
        Feature data
    strategy : Strategy
        Strategy to validate
    cost_model : CostModel
        Transaction costs
    n_splits : int
        Total number of time splits
    n_test_splits : int
        Number of splits to use for test in each combination
    embargo_bars : int
        Gap between adjacent train/test splits

    Returns
    -------
    List[WalkForwardResult]
        Results for each combination
    """
    from itertools import combinations

    n = len(df)
    split_size = n // n_splits

    # Create split boundaries
    splits = []
    for i in range(n_splits):
        start = i * split_size
        end = min((i + 1) * split_size, n)
        splits.append((start, end))

    results = []

    # Test all combinations of test splits
    for test_indices in combinations(range(n_splits), n_test_splits):
        train_indices = [i for i in range(n_splits) if i not in test_indices]

        # Build train and test dataframes
        train_dfs = []
        test_dfs = []

        for idx in train_indices:
            start, end = splits[idx]
            # Apply embargo at boundaries
            if idx + 1 in test_indices:
                end = max(start, end - embargo_bars)
            if idx - 1 in test_indices:
                start = min(end, start + embargo_bars)
            if end > start:
                train_dfs.append(df.slice(start, end - start))

        for idx in test_indices:
            start, end = splits[idx]
            test_dfs.append(df.slice(start, end - start))

        if not train_dfs or not test_dfs:
            continue

        train_df = pl.concat(train_dfs)
        test_df = pl.concat(test_dfs)

        try:
            train_result = run_backtest(train_df, strategy, cost_model)
            test_result = run_backtest(test_df, strategy, cost_model)

            # Create a WalkForwardResult for this combination
            fold_result = FoldResult(
                fold_idx=0,
                train_start_idx=0,
                train_end_idx=len(train_df),
                test_start_idx=0,
                test_end_idx=len(test_df),
                train_result=train_result,
                test_result=test_result,
            )

            is_sharpe = train_result.sharpe_ratio
            oos_sharpe = test_result.sharpe_ratio
            oos_is_ratio = oos_sharpe / is_sharpe if is_sharpe > 0 else 0

            results.append(
                WalkForwardResult(
                    strategy_name=strategy.name,
                    fold_results=[fold_result],
                    in_sample_sharpe=is_sharpe,
                    out_of_sample_sharpe=oos_sharpe,
                    oos_is_ratio=oos_is_ratio,
                    is_valid=oos_is_ratio >= 0.7 and oos_sharpe > 0.3,
                    n_folds=1,
                    embargo_bars=embargo_bars,
                    total_train_trades=train_result.total_trades,
                    total_test_trades=test_result.total_trades,
                )
            )
        except Exception as e:
            print(f"Warning: CPCV combination {test_indices} failed: {e}")

    return results


def compute_deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    variance_of_trials: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Compute deflated Sharpe ratio to account for multiple testing.

    Based on Bailey & Lopez de Prado (2014).

    Parameters
    ----------
    observed_sharpe : float
        Sharpe ratio from backtest
    n_trials : int
        Number of strategy variants tested
    variance_of_trials : float
        Estimated variance of Sharpe across trials
    skewness : float
        Skewness of returns
    kurtosis : float
        Kurtosis of returns

    Returns
    -------
    float
        Probability that observed Sharpe is a false positive
    """
    from scipy import stats

    # Expected maximum Sharpe under null hypothesis
    e_max_sharpe = variance_of_trials * (
        (1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / n_trials)
        + np.euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )

    # Standard deviation of maximum
    std_max_sharpe = variance_of_trials * (
        np.pi / np.sqrt(6) * stats.norm.ppf(1 - 1 / n_trials)
    )

    # Deflated Sharpe
    if std_max_sharpe > 0:
        deflated = (observed_sharpe - e_max_sharpe) / std_max_sharpe
        # Probability of false positive
        prob_false_positive = stats.norm.cdf(deflated)
        return prob_false_positive
    else:
        return 0.5
