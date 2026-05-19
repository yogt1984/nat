"""
Core abstractions for the microstructure algorithm framework.

AlgorithmFeature: declarative descriptor for one derived feature.
MicrostructureAlgorithm: ABC for tick-by-tick algorithms.

Mirrors the Rust trait in `rust/ing/src/algorithms/mod.rs`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlgorithmFeature:
    """Descriptor for a single algorithm-derived feature."""
    name: str
    warmup: int = 0
    description: str = ""


class MicrostructureAlgorithm(ABC):
    """Base class for microstructure algorithms.

    Algorithms process one tick at a time via step(), producing
    a dict of algorithm-specific derived features (alg_features).

    step() uses dict[str, float] (not pd.Series) to:
    - Avoid pandas overhead per tick
    - Match the Rust trait's step(&Features) interface
    - Enable direct porting to Rust later
    """

    @abstractmethod
    def name(self) -> str:
        """Unique algorithm name."""
        ...

    @abstractmethod
    def alg_features(self) -> list[AlgorithmFeature]:
        """Descriptors for each output feature."""
        ...

    @abstractmethod
    def required_columns(self) -> list[str]:
        """Base feature columns needed as input."""
        ...

    @abstractmethod
    def step(self, tick: dict[str, float]) -> dict[str, float]:
        """Process one tick, return algorithm feature values."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
        ...

    @property
    def warmup(self) -> int:
        """Max warmup across all features."""
        feats = self.alg_features()
        return max((f.warmup for f in feats), default=0) if feats else 0

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.alg_features()]

    def run_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run step() over every row of df.

        Default implementation iterates rows. Override for vectorized.
        Returns DataFrame of alg_features with same index as df.
        """
        self.reset()
        required = self.required_columns()
        names = self.feature_names
        n = len(df)

        # Pre-allocate arrays
        arrays = {name: np.empty(n) for name in names}

        for i in range(n):
            row = df.iloc[i]
            tick = {}
            for col in required:
                if col in df.columns:
                    tick[col] = float(row[col])
            result = self.step(tick)
            for name in names:
                arrays[name][i] = result.get(name, np.nan)

        result_df = pd.DataFrame(arrays, index=df.index)

        # NaN-out warmup period
        warmup = self.warmup
        if warmup > 0 and warmup < n:
            result_df.iloc[:warmup] = np.nan

        return result_df
