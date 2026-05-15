"""Failure Recycler — re-examines failed hypotheses when conditions change.

Scans the graveyard for hypotheses whose failure condition may no longer apply:
  - insufficient_data → re-queue when new data accumulates
  - no_replication → re-queue when new date available
  - cost_killed → re-queue when complementary signal found

Schedule: weekly.
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
    """Re-queue failed hypotheses whose failure conditions have changed."""
    hypotheses = []
    graveyard = queue.graveyard

    n_dates = len(manifest.get("dates", {}))
    total_hours = manifest.get("total_hours_per_symbol", 0)

    for h in graveyard:
        reason = h.failure_reason

        if reason == "insufficient_data":
            # Check if we now have enough data
            req_hours = h.thresholds.get("min_hours", 4)
            if total_hours >= req_hours * 1.5:  # 50% margin
                recycled = Hypothesis.create(
                    claim=h.claim,
                    generator="recycler",
                    test_protocol=h.test_protocol,
                    priority=h.priority * 0.8,  # Slightly lower priority
                    thresholds=h.thresholds,
                    parent_id=h.id,
                )
                hypotheses.append(recycled)

        elif reason == "no_replication":
            # Check if we have new dates to test on
            req_dates = h.thresholds.get("min_oos_dates", 2)
            if n_dates >= req_dates + 2:  # 2 extra dates beyond requirement
                recycled = Hypothesis.create(
                    claim=h.claim,
                    generator="recycler",
                    test_protocol=h.test_protocol,
                    priority=h.priority * 0.7,
                    thresholds=h.thresholds,
                    parent_id=h.id,
                )
                hypotheses.append(recycled)

    log.info("Recycler generator: %d hypotheses from %d graveyard entries",
             len(hypotheses), len(graveyard))
    return hypotheses
