"""Tests for ml_rollback.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_rollback import (
    _read_daily_algos,
    _write_daily_algos,
    disable,
    enable,
    list_models,
    rollback_model,
)


MOCK_DAILY = '''\
DAILY_ALGOS = [
    "jump_detector", "optimal_entry",
    "momentum_continuation",
]
SYMBOLS = ["BTC"]
'''


def _write_mock_daily(tmp_path):
    f = tmp_path / "paper_trader_daily.py"
    f.write_text(MOCK_DAILY)
    return f


def test_disable_removes_from_daily_algos(tmp_path):
    """Disable removes algo from DAILY_ALGOS list."""
    f = _write_mock_daily(tmp_path)
    result = disable("momentum_continuation", filepath=f)
    assert result is True

    algos = _read_daily_algos(f)
    assert "momentum_continuation" not in algos
    assert "jump_detector" in algos


def test_enable_adds_back(tmp_path):
    """After disable + enable, algo is back in DAILY_ALGOS."""
    f = _write_mock_daily(tmp_path)
    disable("momentum_continuation", filepath=f)
    enable("momentum_continuation", filepath=f)

    algos = _read_daily_algos(f)
    assert "momentum_continuation" in algos


def test_rollback_model_archives_newest(tmp_path):
    """Create 2 model files. Rollback moves newest to archived/."""
    model_dir = tmp_path / "test_algo"
    model_dir.mkdir()

    # Create 2 fake models with metadata
    for i, date in enumerate(["2026-01-01T00:00:00", "2026-06-01T00:00:00"]):
        name = f"v{i+1}"
        (model_dir / f"{name}.pkl").write_text("model_data")
        (model_dir / f"{name}_metadata.json").write_text(json.dumps({
            "model_type": "sklearn",
            "model_name": "test",
            "feature_names": ["f1"],
            "hyperparameters": {},
            "performance_metrics": {},
            "training_date": date,
        }))

    result = rollback_model("test_algo", models_dir=tmp_path)
    assert result is True
    assert (model_dir / "archived" / "v2.pkl").exists()
    assert not (model_dir / "v2.pkl").exists()
    assert (model_dir / "v1.pkl").exists()


def test_list_models_empty_dir(tmp_path):
    """Empty model dir -> returns empty list."""
    model_dir = tmp_path / "empty_algo"
    model_dir.mkdir()

    result = list_models("empty_algo", models_dir=tmp_path)
    assert result == []


def test_rollback_no_previous(tmp_path):
    """Only 1 model file -> rollback returns False."""
    model_dir = tmp_path / "solo_algo"
    model_dir.mkdir()

    (model_dir / "v1.pkl").write_text("model_data")
    (model_dir / "v1_metadata.json").write_text(json.dumps({
        "model_type": "sklearn",
        "model_name": "test",
        "feature_names": ["f1"],
        "hyperparameters": {},
        "performance_metrics": {},
        "training_date": "2026-01-01T00:00:00",
    }))

    result = rollback_model("solo_algo", models_dir=tmp_path)
    assert result is False
    assert (model_dir / "v1.pkl").exists()  # not deleted
