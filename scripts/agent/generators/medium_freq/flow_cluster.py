"""Flow Clustering Generator — aggregated order flow at 5min bar level.

Tests whether bar-level flow imbalance and aggressor ratio predict multi-bar
returns. The claim: sustained directional pressure over minutes is a different
signal from instantaneous tick-level microstructure imbalance.

Gate conditions: high illiquidity (flow has more impact), high VPIN (informed).
"""

from __future__ import annotations

import logging
from typing import Optional

from ...hypothesis import Hypothesis, GeneratorStats
from ...hypothesis_queue import HypothesisQueue

log = logging.getLogger(__name__)

Hypothesis.register_prefix("flow_cluster", "FCL")

# Order flow features (aggregated to 5min bars)
SIGNAL_FEATURES = [
    "imbalance_qty_l5",
    "flow_aggressor_ratio_5s",
    "flow_volume_5s",
]

# Gate features for flow quality
GATE_FEATURES = [
    "illiq_composite",
    "toxic_vpin_50",
    "illiq_kyle_100",
]

THRESHOLDS = ["P40", "P60", "P80"]
DIRECTIONS = {"illiq_composite": ">", "toxic_vpin_50": ">", "illiq_kyle_100": ">"}


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Generate flow clustering hypotheses at bar resolution."""
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
                claim = (f"{signal_feat} bar-flow gated by "
                         f"{gate_feat}{direction}{thresh} predicts 5min returns")

                if claim in existing_claims:
                    continue

                test_protocol = [
                    f"profile scalp --symbol BTC --data {data_dir} "
                    f"--timeframe 5min --forward-test",
                ]

                priority = 0.45
                if "imbalance" in signal_feat:
                    priority += 0.15  # Direct directional signal
                if gate_feat == "toxic_vpin_50":
                    priority += 0.1  # Informed flow gate is high value

                h = Hypothesis.create(
                    claim=claim,
                    generator="flow_cluster",
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

    log.info("Flow cluster generator: %d new hypotheses", len(hypotheses))
    return hypotheses
