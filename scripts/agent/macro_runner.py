"""Macro experiment runner — 4-gate protocol at 1h bar resolution.

Gate protocol:
    DISCOVERY → TEMPORAL REPLICATION → SYMBOL REPLICATION → CORRELATION DEDUP → REGISTER

Inherits all gate logic from BaseRunner. Configures via class attributes
for 1h timeframe, macro-specific features, and separate registry.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseRunner

ROOT = Path(__file__).resolve().parent.parent.parent
MACRO_REGISTRY_PATH = ROOT / "data" / "agent_macro" / "registry.json"

# Macro signal features that may appear in claims
MACRO_SIGNAL_FEATURES = [
    "ctx_funding_rate", "ctx_funding_zscore",
    "ctx_oi_change_pct_5m", "regime_divergence_1h", "regime_divergence_4h",
    "whale_flow_momentum", "whale_net_flow_1h", "whale_directional_agreement",
]


class MacroRunner(BaseRunner):
    """Runs a macro hypothesis through the 4-gate protocol."""

    TIMEFRAME = "1h"
    SIGNAL_FEATURES = MACRO_SIGNAL_FEATURES
    DEFAULT_FEATURE = "ctx_funding_zscore"
    DEFAULT_HORIZON_S = 3600.0
    REGISTRY_PATH = MACRO_REGISTRY_PATH
