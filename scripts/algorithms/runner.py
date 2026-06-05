"""
Algorithm execution engine — batch and parquet modes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import MicrostructureAlgorithm


@dataclass
class AlgorithmResult:
    """Output of running an algorithm over data."""
    algorithm_name: str
    features_df: pd.DataFrame   # alg_features columns
    base_df: pd.DataFrame       # original data (midprice, signal, etc.)
    n_ticks: int
    warmup_ticks: int
    elapsed_s: float


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
