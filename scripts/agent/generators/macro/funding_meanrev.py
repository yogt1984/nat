"""Funding Mean-Reversion Generator — funding rate extremes at 1h bars.

Tests whether extreme funding rates (via z-score) predict mean-reversion
within 1–12 hours. Bidirectional: high funding → short, low funding → long.

Uses ctx_funding_zscore as workaround until 5.6 adds funding momentum.
"""

from __future__ import annotations

import logging
from typing import Optional

from ...hypothesis import Hypothesis, GeneratorStats
from ...hypothesis_queue import HypothesisQueue

log = logging.getLogger(__name__)

Hypothesis.register_prefix("funding_meanrev", "FRM")

# Funding features
SIGNAL_FEATURES = [
    "ctx_funding_rate",
    "ctx_funding_zscore",
]

# Gate conditions for mean-reversion quality
GATE_FEATURES = [
    "vol_ratio_short_long",
    "regime_range_pos_24h",
    "ctx_premium_bps",
]

THRESHOLDS = ["P20", "P40", "P60", "P80"]
# Funding mean-reverts from extremes, so test both directions on gates
DIRECTIONS = {"vol_ratio_short_long": "<", "regime_range_pos_24h": ">",
              "ctx_premium_bps": ">"}


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Generate funding mean-reversion hypotheses at 1h resolution."""
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
                claim = (f"{signal_feat} meanrev gated by "
                         f"{gate_feat}{direction}{thresh} predicts 1h returns")

                if claim in existing_claims:
                    continue

                test_protocol = [
                    f"profile scalp --symbol BTC --data {data_dir} "
                    f"--timeframe 1h --forward-test",
                ]

                priority = 0.5
                if "zscore" in signal_feat:
                    priority += 0.2  # Z-score is cleaner signal
                if gate_feat == "regime_range_pos_24h":
                    priority += 0.1  # Range extremes amplify mean-reversion

                h = Hypothesis.create(
                    claim=claim,
                    generator="funding_meanrev",
                    test_protocol=test_protocol,
                    priority=priority,
                    thresholds={
                        "min_ic": 0.07,
                        "min_dIC": 0.02,
                        "min_coverage": 0.10,
                        "min_hours": 4,
                        "symbols": ["BTC"],
                        "min_oos_dates": 2,
                        "min_symbols": 2,
                        "regime_gate": f"{gate_feat}{direction}{thresh}",
                        "horizon_s": 3600.0,
                        "timeframe": "1h",
                    },
                )
                hypotheses.append(h)

    log.info("Funding meanrev generator: %d new hypotheses", len(hypotheses))
    return hypotheses
