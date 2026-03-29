"""
Tests for experiment governance script.

Verifies snapshot and manifest creation works.
"""

import pytest
import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_governance import ExperimentGovernance


def test_script_exists():
    """Script file should exist."""
    script_path = Path("scripts/experiment_governance.py")
    assert script_path.exists(), "experiment_governance.py should exist"


def test_can_create_governance():
    """Should be able to create governance instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gov = ExperimentGovernance(base_dir=Path(tmpdir))
        assert gov.snapshots_dir.exists()
        assert gov.manifests_dir.exists()


def test_can_list_empty():
    """Should handle empty snapshot/experiment lists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gov = ExperimentGovernance(base_dir=Path(tmpdir))
        snapshots = gov.list_snapshots()
        experiments = gov.list_experiments()

        assert isinstance(snapshots, list)
        assert isinstance(experiments, list)
        assert len(snapshots) == 0
        assert len(experiments) == 0
