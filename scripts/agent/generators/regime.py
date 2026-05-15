"""Regime Transition Detector — monitors HMM state transitions for signal quality changes.

Schedule: daily.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..hypothesis import Hypothesis, GeneratorStats
from ..queue import HypothesisQueue

log = logging.getLogger(__name__)


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Detect regime transitions and test transition-gated trading."""
    hypotheses = []
    dates = sorted(manifest.get("dates", {}).keys())
    if not dates:
        return []

    latest = dates[-1]
    data_dir = f"data/features/{latest}"

    # Generate hypothesis to test if regime transitions improve IC
    existing_claims = {h.claim for h in queue._all}
    claim = f"IC improves in first 5min after HMM state transition on {latest}"
    if claim not in existing_claims:
        h = Hypothesis.create(
            claim=claim,
            generator="regime",
            test_protocol=[
                f"cluster hmm-fit --data {data_dir} --symbol BTC --n-states 3",
                f"spannung regime --data {data_dir} --symbol BTC",
            ],
            priority=0.4,
            thresholds={"min_ic": 0.10, "min_hours": 4, "symbols": ["BTC"]},
        )
        hypotheses.append(h)

    log.info("Regime generator: %d hypotheses", len(hypotheses))
    return hypotheses
