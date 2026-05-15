"""Momentum Generator — trend features at 5min bar resolution.

Tests whether trend slope, R2, Hurst exponent, and MA crossover features
(resampled to 5min bars) predict multi-bar forward returns.

Gate conditions: low tick entropy, stable vol, trending regime.
"""

from __future__ import annotations

import logging
from typing import Optional

from ...hypothesis import Hypothesis, GeneratorStats
from ...hypothesis_queue import HypothesisQueue

log = logging.getLogger(__name__)

# Register prefix at import time so IDs are correct even without the agent
Hypothesis.register_prefix("momentum", "MOM")

# Trend features to test as directional signals (at 5min aggregation)
SIGNAL_FEATURES = [
    "trend_momentum_300",
    "trend_momentum_r2_300",
    "trend_hurst_300",
    "trend_ma_crossover_norm",
]

# Conditioning gates for trend signals
GATE_FEATURES = [
    "ent_tick_1m",
    "vol_ratio_short_long",
    "derived_regime_type_score",
]

THRESHOLDS = ["P20", "P40", "P60", "P80"]
DIRECTIONS = {"ent_tick_1m": "<", "vol_ratio_short_long": "<",
              "derived_regime_type_score": ">"}


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Generate hypotheses for trend features at 5min bar resolution."""
    existing_claims = {h.claim for h in queue._all}
    hypotheses = []

    dates = sorted(manifest.get("dates", {}).keys())
    if not dates:
        return []
    latest_date = dates[-1]
    data_dir = f"data/features/{latest_date}"

    for signal_feat in SIGNAL_FEATURES:
        for gate_feat in GATE_FEATURES:
            direction = DIRECTIONS[gate_feat]
            for thresh in THRESHOLDS:
                claim = (f"{signal_feat} gated by {gate_feat}{direction}{thresh} "
                         f"predicts 5min returns")

                if claim in existing_claims:
                    continue

                test_protocol = [
                    f"profile scalp --symbol BTC --data {data_dir} "
                    f"--timeframe 5min --forward-test",
                ]

                priority = 0.5
                if "r2" in signal_feat:
                    priority += 0.2
                if "hurst" in signal_feat:
                    priority += 0.15
                if direction == "<" and thresh in ("P20", "P40"):
                    priority += 0.1

                h = Hypothesis.create(
                    claim=claim,
                    generator="momentum",
                    test_protocol=test_protocol,
                    priority=priority,
                    thresholds={
                        "min_ic": 0.08,
                        "min_dIC": 0.03,
                        "min_coverage": 0.10,
                        "min_hours": 4,
                        "symbols": ["BTC"],
                        "min_oos_dates": 2,
                        "min_symbols": 2,
                        "regime_gate": f"{gate_feat}{direction}{thresh}",
                        "horizon_s": 300.0,
                        "timeframe": "5min",
                    },
                )
                hypotheses.append(h)

    log.info("Momentum generator: %d new hypotheses", len(hypotheses))
    return hypotheses
