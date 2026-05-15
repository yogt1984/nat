"""Cross-Asset Prober — tests lead-lag relationships between symbols.

Schedule: weekly (computationally expensive).
"""

from __future__ import annotations

import logging
from typing import Optional

from ..hypothesis import Hypothesis, GeneratorStats
from ..queue import HypothesisQueue

log = logging.getLogger(__name__)

SYMBOL_PAIRS = [("BTC", "ETH"), ("BTC", "SOL"), ("ETH", "SOL")]


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Generate cross-asset lead-lag hypotheses."""
    hypotheses = []
    dates = sorted(manifest.get("dates", {}).keys())
    if not dates:
        return []

    latest = dates[-1]
    data_dir = f"data/features/{latest}"
    existing_claims = {h.claim for h in queue._all}

    for leader, follower in SYMBOL_PAIRS:
        claim = f"{leader} imbalance leads {follower} returns at 68s coherence frequency"
        if claim not in existing_claims:
            h = Hypothesis.create(
                claim=claim,
                generator="cross_asset",
                test_protocol=[
                    f"spannung spectral --data {data_dir} --symbol {leader}",
                    f"spannung spectral --data {data_dir} --symbol {follower}",
                ],
                priority=0.6,  # High novelty — cross-asset signals are capacity-additive
                thresholds={
                    "min_ic": 0.05,
                    "min_hours": 4,
                    "symbols": [leader, follower],
                },
            )
            hypotheses.append(h)

    log.info("Cross-asset generator: %d hypotheses", len(hypotheses))
    return hypotheses
