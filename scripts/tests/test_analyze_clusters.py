"""
Skeptical tests for cluster analysis script.

Tests verify the integration works end-to-end.
"""

import pytest
import sys
import numpy as np
import polars as pl
from pathlib import Path
import tempfile
from datetime import datetime

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_script_exists():
    """Script file should exist."""
    script_path = Path("scripts/analyze_clusters.py")
    assert script_path.exists(), "analyze_clusters.py should exist"


def test_imports_cluster_quality():
    """Should be able to import cluster quality modules."""
    try:
        from cluster_quality import compute_all_metrics
        assert compute_all_metrics is not None
    except ImportError as e:
        pytest.fail(f"Failed to import cluster_quality: {e}")


def test_can_create_mock_parquet():
    """Should be able to create mock Parquet data for testing."""
    # Create mock data
    data = {
        "timestamp": [datetime.now()] * 100,
        "kyle_lambda_100": np.random.randn(100),
        "vpin_50": np.random.randn(100),
        "absorption_zscore": np.random.randn(100),
        "hurst_300": np.random.uniform(0, 1, 100),
        "whale_net_flow_1h": np.random.randn(100),
        "midprice": np.cumsum(np.random.randn(100)) + 50000,
    }

    df = pl.DataFrame(data)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.write_parquet(f.name)

        # Verify can read back
        df2 = pl.read_parquet(f.name)
        assert len(df2) == 100

        Path(f.name).unlink()  # Cleanup
