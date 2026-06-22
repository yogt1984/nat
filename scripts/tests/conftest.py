"""Shared test configuration.

With `pip install -e scripts/`, Python packaging handles imports.
This file is kept as a fallback for running tests without the editable install.
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

EXPLORATION_DIR = SCRIPTS_DIR / "exploration"
if EXPLORATION_DIR.exists() and str(EXPLORATION_DIR) not in sys.path:
    sys.path.insert(0, str(EXPLORATION_DIR))


def pytest_configure(config):
    """Register custom markers.

    `slow` flags the heavy clustering/profiling suites (test_hierarchy,
    test_cluster_engine, test_pipeline_runner) that each take 1–2 min of genuine
    GMM/Hopkins/block-bootstrap compute. They PASS — they're just slow — so a
    constrained/fast run can `pytest -m "not slow"` to skip them while full runs
    (CI/nightly) still cover them. Avoids them being misread as "hung" under a
    short per-file timeout.
    """
    config.addinivalue_line(
        "markers",
        "slow: heavy compute (>60s); deselect with -m 'not slow' for fast runs",
    )
