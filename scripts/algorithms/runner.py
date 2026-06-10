"""
Algorithm execution engine — batch and parquet modes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import MicrostructureAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmResult:
    """Output of running an algorithm over data."""
    algorithm_name: str
    features_df: pd.DataFrame   # alg_features columns
    base_df: pd.DataFrame       # original data (midprice, signal, etc.)
    n_ticks: int
    warmup_ticks: int
    elapsed_s: float


def run_chain(
    df: pd.DataFrame,
    algorithms: list[MicrostructureAlgorithm],
) -> list[AlgorithmResult]:
    """Run algorithms in dependency order: tick-level first, then bar-level.

    Tick-level algorithm outputs are appended to the DataFrame so that
    bar-level algorithms (which consume aggregated bars) can see them.
    This enables chaining, e.g. convolver (tick) → momentum_continuation (bar).

    Args:
        df: Raw tick-level DataFrame.
        algorithms: List of algorithm instances to run.

    Returns:
        List of AlgorithmResult, one per algorithm.
    """
    tick_algos = [a for a in algorithms if not a.bar_level]
    bar_algos = [a for a in algorithms if a.bar_level]

    enriched = df.copy()
    results: list[AlgorithmResult] = []

    # Phase 1: tick-level algorithms — append outputs to enriched df
    for algo in tick_algos:
        runner = AlgorithmRunner(algo)
        result = runner.run_on_dataframe(enriched)
        results.append(result)
        for col in result.features_df.columns:
            enriched[col] = result.features_df[col]

    # Phase 2: bar-level algorithms — see tick algo outputs via aggregate_bars
    for algo in bar_algos:
        runner = AlgorithmRunner(algo)
        result = runner.run_on_dataframe(enriched)
        results.append(result)

    return results


def enrich_with_convolver(
    df: pd.DataFrame,
    symbol: str = "BTC",
    kernel_dir: str = "models",
) -> pd.DataFrame:
    """Run convolver on tick data and append its outputs to the DataFrame.

    Use before bar aggregation in training scripts so that convolver features
    (e.g. alg_conv_best_score) appear in aggregated bars as alg_conv_best_score_max.

    Args:
        df: Tick-level DataFrame with raw_midprice and flow_volume_1s.
        symbol: Symbol name — loads models/convolver_kernels_{symbol}.
        kernel_dir: Directory containing kernel .npz/.json files.

    Returns:
        DataFrame with convolver output columns appended.
    """
    from .convolver import Convolver

    kernel_path = f"{kernel_dir}/convolver_kernels_{symbol}"
    try:
        conv = Convolver(kernel_path=kernel_path)
    except FileNotFoundError:
        # Try default (no symbol suffix)
        try:
            conv = Convolver(kernel_path=f"{kernel_dir}/convolver_kernels")
        except FileNotFoundError:
            logger.warning("No convolver kernel library found, skipping enrichment")
            return df

    logger.info("Enriching with convolver features (kernel=%s)", kernel_path)
    features_df = conv.run_batch(df)

    enriched = df.copy()
    for col in features_df.columns:
        enriched[col] = features_df[col].values
    logger.info("Added %d convolver columns, %.1f%% non-NaN",
                len(features_df.columns),
                features_df.notna().all(axis=1).mean() * 100)
    return enriched


class AlgorithmRunner:
    """Run a MicrostructureAlgorithm over data."""

    def __init__(self, algorithm: MicrostructureAlgorithm):
        self.algorithm = algorithm

    def run_on_dataframe(self, df: pd.DataFrame) -> AlgorithmResult:
        """Run algorithm over a pre-loaded DataFrame.

        For bar-level algorithms (bar_level=True), tick data is aggregated
        to bars before calling run_batch(). Results are forward-filled back
        to tick-level index for downstream compatibility.
        """
        algo = self.algorithm

        # Bar-level algorithms operate on aggregated bars, not raw ticks
        if algo.bar_level and "timestamp_ns" in df.columns:
            from cluster_pipeline.preprocess import aggregate_bars

            bars = aggregate_bars(df, timeframe=algo.bar_timeframe)

            # Validate required columns against bar-aggregated names
            missing = [c for c in algo.required_columns() if c not in bars.columns]
            if missing:
                raise ValueError(
                    f"Algorithm '{algo.name()}' requires bar columns {missing} "
                    f"not found after aggregation. Available: {list(bars.columns[:20])}..."
                )

            # NaN availability guard
            for col in algo.required_columns():
                if col in bars.columns:
                    nan_rate = bars[col].isna().mean()
                    if nan_rate > 0.95:
                        logger.warning(
                            "%s: required column '%s' is %.0f%% NaN",
                            algo.name(), col, nan_rate * 100,
                        )

            t0 = time.time()
            features_df = algo.run_batch(bars)
            elapsed = time.time() - t0

            # Forward-fill bar results to tick-level index
            features_df = features_df.reindex(df.index, method="ffill")

            return AlgorithmResult(
                algorithm_name=algo.name(),
                features_df=features_df,
                base_df=df,
                n_ticks=len(df),
                warmup_ticks=algo.warmup,
                elapsed_s=round(elapsed, 2),
            )

        # Tick-level algorithms (default path)
        missing = [c for c in algo.required_columns() if c not in df.columns]
        if missing:
            raise ValueError(
                f"Algorithm '{algo.name()}' requires columns {missing} "
                f"not found in data. Available: {list(df.columns[:20])}..."
            )

        # NaN availability guard
        for col in algo.required_columns():
            nan_rate = df[col].isna().mean()
            if nan_rate > 0.95:
                logger.warning(
                    "%s: required column '%s' is %.0f%% NaN",
                    algo.name(), col, nan_rate * 100,
                )

        t0 = time.time()
        features_df = algo.run_batch(df)
        elapsed = time.time() - t0

        return AlgorithmResult(
            algorithm_name=algo.name(),
            features_df=features_df,
            base_df=df,
            n_ticks=len(df),
            warmup_ticks=algo.warmup,
            elapsed_s=round(elapsed, 2),
        )

    def run_on_parquet(
        self,
        data_dir: str,
        symbol: str,
        max_memory_mb: float = 2000.0,
        columns: Optional[list[str]] = None,
    ) -> AlgorithmResult:
        """Load parquet data and run algorithm."""
        # Import loader here to avoid circular dependency
        from cluster_pipeline.loader import load_parquet

        # Determine columns to load
        required = self.algorithm.required_columns()
        base_cols = ["timestamp_ns", "symbol", "raw_midprice", "raw_spread"]
        load_cols = list(set(base_cols + required + (columns or [])))

        df = load_parquet(
            data_dir,
            symbols=[symbol],
            columns=load_cols,
            max_memory_mb=max_memory_mb,
        )

        if len(df) < 100:
            raise ValueError(f"Only {len(df)} rows loaded for {symbol}, need at least 100")

        return self.run_on_dataframe(df)
