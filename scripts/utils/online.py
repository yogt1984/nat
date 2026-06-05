"""Online statistical utilities for streaming algorithms."""

from __future__ import annotations

import numpy as np


class WelfordNormalizer:
    """Online mean/variance tracker (Welford's algorithm)."""

    def __init__(self, d: int):
        self.n = 0
        self.mean = np.zeros(d)
        self.M2 = np.zeros(d)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.n < 2:
            return x - self.mean
        std = np.sqrt(self.M2 / (self.n - 1))
        std = np.where(std < 1e-10, 1.0, std)
        return (x - self.mean) / std
