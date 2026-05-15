"""Systematic Screening Generator — exhaustive feature x condition x threshold search.

Wraps `nat spannung regime` and `nat profile scalp`. For each untested
(feature x gate x threshold) triple, emits a hypothesis.

Schedule: nightly. Priority boost for features uncorrelated with registry.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..hypothesis import Hypothesis, GeneratorStats
from ..queue import HypothesisQueue

log = logging.getLogger(__name__)

# Features to test as directional signals
DIRECTIONAL_FEATURES = [
    "imbalance_qty_l1", "imbalance_qty_l5", "imbalance_qty_l10",
    "imbalance_depth_weighted", "imbalance_notional_l5",
    "flow_aggressor_ratio_5s", "toxic_flow_imbalance",
]

# Features to test as regime gates
GATE_FEATURES = [
    "ent_book_shape", "ent_tick_5s", "ent_tick_30s", "ent_tick_1m",
    "ent_permutation_returns_16", "ent_spread_dispersion",
    "illiq_kyle_100", "illiq_composite", "illiq_amihud_100",
    "toxic_vpin_50", "toxic_adverse_selection", "toxic_index",
    "vol_returns_1m", "vol_returns_5m", "vol_ratio_short_long",
    "derived_regime_type_score", "derived_regime_confidence",
]

THRESHOLDS = ["P20", "P40", "P60", "P80"]
DIRECTIONS = ["<", ">"]
HORIZONS_S = [1, 5]


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Generate hypotheses for untested feature x gate combinations."""
    # Get existing claims to avoid duplicates
    existing_claims = {h.claim for h in queue._all}
    hypotheses = []

    # Get the latest date directory
    dates = sorted(manifest.get("dates", {}).keys())
    if not dates:
        return []
    latest_date = dates[-1]
    data_dir = f"data/features/{latest_date}"

    for signal_feat in DIRECTIONAL_FEATURES[:3]:  # Start with top 3
        for gate_feat in GATE_FEATURES:
            for thresh in THRESHOLDS:
                for direction in DIRECTIONS:
                    claim = (f"{signal_feat} gated by {gate_feat}{direction}{thresh} "
                             f"predicts 5s returns")

                    if claim in existing_claims:
                        continue

                    # Build test protocol
                    test_protocol = [
                        f"spannung regime --data {data_dir} --symbol BTC",
                        f"profile scalp --symbol BTC --data {data_dir} --forward-test",
                    ]

                    # Priority: ent_book_shape conditions get a boost (proven winner)
                    priority = 0.5
                    if "ent_book_shape" in gate_feat:
                        priority += 0.3
                    if direction == "<" and thresh in ("P20", "P40"):
                        priority += 0.1  # Low-threshold conditions tend to work better

                    h = Hypothesis.create(
                        claim=claim,
                        generator="systematic",
                        test_protocol=test_protocol,
                        priority=priority,
                        thresholds={
                            "min_ic": 0.10,
                            "min_dIC": 0.05,
                            "min_coverage": 0.10,
                            "min_hours": 4,
                            "symbols": ["BTC"],
                            "min_oos_dates": 1,
                            "min_symbols": 2,
                            "regime_gate": f"{gate_feat}{direction}{thresh}",
                            "horizon_s": 5.0,
                        },
                    )
                    hypotheses.append(h)

    log.info("Systematic generator: %d new hypotheses", len(hypotheses))
    return hypotheses
