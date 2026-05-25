"""Medium-frequency experiment runner — 4-gate protocol at 5min bar resolution.

Gate protocol:
    DISCOVERY → TEMPORAL REPLICATION → SYMBOL REPLICATION → CORRELATION DEDUP → REGISTER

Inherits all gate logic from BaseRunner. Configures via class attributes
for 5min timeframe, MF-specific features, and separate registry.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseRunner

ROOT = Path(__file__).resolve().parent.parent.parent
MF_REGISTRY_PATH = ROOT / "data" / "agent_mf" / "registry.json"

# Medium-frequency signal features that may appear in claims
MF_SIGNAL_FEATURES = [
    "trend_momentum_300", "trend_momentum_r2_300", "trend_hurst_300",
    "trend_ma_crossover_norm", "vol_ratio_short_long", "vol_zscore",
    "imbalance_qty_l5", "flow_aggressor_ratio_5s", "flow_volume_5s",
]


class MediumFrequencyRunner(BaseRunner):
    """Runs a medium-frequency hypothesis through the 4-gate protocol."""

    TIMEFRAME = "5min"
    SIGNAL_FEATURES = MF_SIGNAL_FEATURES
    DEFAULT_FEATURE = "trend_momentum_300"
    DEFAULT_HORIZON_S = 300.0
    REGISTRY_PATH = MF_REGISTRY_PATH
