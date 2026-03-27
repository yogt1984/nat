"""
Pytest Configuration for Backtest Tests

Shared fixtures and configuration.
"""

import pytest
import numpy as np


# Seed random number generator for reproducible tests
@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield


@pytest.fixture
def large_dataset_size():
    """Standard size for large dataset tests."""
    return 10000


@pytest.fixture
def small_dataset_size():
    """Standard size for small dataset tests."""
    return 100
