"""
Task 1: Experiment state schema.

Single JSON file that represents the entire dashboard state.
All components write to it; the web server reads from it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "experiment_state.json"


@dataclass
class ExperimentInfo:
    name: str = "EXP-0"
    started: str = ""  # ISO 8601
    duration_days: int = 7
    status: str = "IDLE"  # IDLE, COLLECTING, ANALYZING, DONE


@dataclass
class DataMetrics:
    total_rows: int = 0
    bars_15m: int = 0
    total_files: int = 0
    disk_mb: float = 0.0
    rate_per_day: float = 0.0
    symbols: Dict[str, bool] = field(default_factory=lambda: {"BTC": False, "ETH": False, "SOL": False})
    days: int = 0
    date_range: str = ""
    last_flush_ago_s: float = -1


@dataclass
class HealthMetrics:
    nan_ratio: float = 0.0
    n_gaps: int = 0
    longest_gap_s: float = 0.0
    features_ok: bool = True
    per_symbol_rows_1h: Dict[str, int] = field(default_factory=dict)


@dataclass
class StateInfo:
    id: int = 0
    n_bars: int = 0
    pct: float = 0.0
    mean_duration: float = 0.0
    label: str = ""
    top_features: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProfilingSnapshot:
    last_run: str = ""  # ISO 8601
    n_bars_used: int = 0
    status: str = "pending"  # pending, insufficient, partial, complete
    hopkins: float = 0.0
    k: int = 0
    silhouette: float = 0.0
    bootstrap_ari: float = 0.0
    states: List[Dict[str, Any]] = field(default_factory=list)
    transition_matrix: List[List[float]] = field(default_factory=list)
    self_transition_rate: float = 0.0
    q1_pass: Optional[bool] = None
    q2_pass: Optional[bool] = None
    q2_best_p: float = 1.0
    q3_pass: Optional[bool] = None
    current_verdict: str = ""
    mean_duration: float = 0.0


@dataclass
class ExperimentState:
    experiment: ExperimentInfo = field(default_factory=ExperimentInfo)
    data: DataMetrics = field(default_factory=DataMetrics)
    health: HealthMetrics = field(default_factory=HealthMetrics)
    profiling: ProfilingSnapshot = field(default_factory=ProfilingSnapshot)
    events: List[Dict[str, str]] = field(default_factory=list)
    updated: str = ""  # ISO 8601


def save_state(state: ExperimentState, path: Path = DEFAULT_STATE_PATH) -> None:
    """Write state to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2)


def load_state(path: Path = DEFAULT_STATE_PATH) -> ExperimentState:
    """Read state from JSON file. Returns default state if file doesn't exist."""
    if not path.exists():
        return ExperimentState()
    with open(path) as f:
        data = json.load(f)

    state = ExperimentState()
    if "experiment" in data:
        state.experiment = ExperimentInfo(**{k: v for k, v in data["experiment"].items() if k in ExperimentInfo.__dataclass_fields__})
    if "data" in data:
        state.data = DataMetrics(**{k: v for k, v in data["data"].items() if k in DataMetrics.__dataclass_fields__})
    if "health" in data:
        state.health = HealthMetrics(**{k: v for k, v in data["health"].items() if k in HealthMetrics.__dataclass_fields__})
    if "profiling" in data:
        state.profiling = ProfilingSnapshot(**{k: v for k, v in data["profiling"].items() if k in ProfilingSnapshot.__dataclass_fields__})
    if "events" in data:
        state.events = data["events"]
    if "updated" in data:
        state.updated = data["updated"]
    return state
