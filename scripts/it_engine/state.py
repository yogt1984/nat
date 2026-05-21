"""
IT Engine state — persistence and serialization.

State is stored per-symbol at data/it_engine/state_{symbol}.json
and published to Redis at nat:it:{symbol} for dashboard/agent consumption.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ITState:
    """Snapshot of information-theoretic analysis for one symbol."""

    # MI matrix: {feature_name: {horizon_label: MI_bits}}
    mi_matrix: dict[str, dict[str, float]] = field(default_factory=dict)

    # Conditional MI: {feature_name: {horizon_label: CMI_bits}}
    cmi_matrix: dict[str, dict[str, float]] = field(default_factory=dict)

    # Interaction information: {feature_name: II_bits}
    interaction: dict[str, float] = field(default_factory=dict)

    # Transfer entropy: {source_feature: {target: TE_bits}}
    transfer_entropy: dict[str, dict[str, float]] = field(default_factory=dict)

    # Greedy selection output
    selected_features: list[str] = field(default_factory=list)
    cumulative_mi: list[float] = field(default_factory=list)

    # Cost viability per feature: {feature_name: bool}
    cost_viable: dict[str, bool] = field(default_factory=dict)

    # Metadata
    symbol: str = ""
    n_samples: int = 0
    last_updated: str = ""
    cycle_count: int = 0

    def save(self, data_dir: str = "data/it_engine") -> str:
        """Persist state to JSON file. Returns file path."""
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, f"state_{self.symbol}.json")
        self.last_updated = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=_json_default)
        return path

    @classmethod
    def load(cls, symbol: str, data_dir: str = "data/it_engine") -> "ITState":
        """Load state from JSON file, or return empty state."""
        path = os.path.join(data_dir, f"state_{symbol}.json")
        if not os.path.exists(path):
            return cls(symbol=symbol)
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_redis_dict(self) -> dict:
        """Flatten state for Redis HSET publication."""
        return {
            "symbol": self.symbol,
            "n_samples": str(self.n_samples),
            "cycle_count": str(self.cycle_count),
            "last_updated": self.last_updated,
            "selected_features": json.dumps(self.selected_features),
            "cumulative_mi": json.dumps(self.cumulative_mi),
            "mi_matrix": json.dumps(self.mi_matrix),
            "cmi_matrix": json.dumps(self.cmi_matrix),
            "interaction": json.dumps(self.interaction),
            "transfer_entropy": json.dumps(self.transfer_entropy),
            "cost_viable": json.dumps(self.cost_viable),
        }

    def top_features(self, n: int = 10, by: str = "mi",
                     horizon: Optional[str] = None) -> list[tuple[str, float]]:
        """Return top-N features sorted by MI or CMI at given horizon.

        Parameters
        ----------
        n       : number of features to return
        by      : "mi" or "cmi"
        horizon : horizon label (e.g. "50min"). If None, uses max across horizons.

        Returns
        -------
        List of (feature_name, value) sorted descending.
        """
        matrix = self.mi_matrix if by == "mi" else self.cmi_matrix
        scores = {}
        for feat, horizons in matrix.items():
            if horizon and horizon in horizons:
                scores[feat] = horizons[horizon]
            else:
                vals = [v for v in horizons.values() if v is not None]
                scores[feat] = max(vals) if vals else 0.0
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:n]


def _json_default(obj):
    """Handle non-serializable types."""
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)
