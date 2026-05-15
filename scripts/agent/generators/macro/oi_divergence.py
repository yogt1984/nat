"""OI Divergence Generator — open interest divergence at 1h bars.

Tests whether OI divergence from price predicts continuation or reversal.
Uses regime_divergence features and OI change aggregated to 1h bars.
"""

from __future__ import annotations

import logging
from typing import Optional

from ...hypothesis import Hypothesis, GeneratorStats
from ...hypothesis_queue import HypothesisQueue

log = logging.getLogger(__name__)

Hypothesis.register_prefix("oi_divergence", "OID")

# OI and divergence features
SIGNAL_FEATURES = [
    "ctx_oi_change_pct_5m",
    "regime_divergence_1h",
    "regime_divergence_4h",
]

# Gate conditions
GATE_FEATURES = [
    "ctx_funding_zscore",
    "regime_absorption_1h",
    "vol_returns_5m",
]

THRESHOLDS = ["P20", "P40", "P60", "P80"]
DIRECTIONS = {"ctx_funding_zscore": ">", "regime_absorption_1h": ">",
              "vol_returns_5m": ">"}


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Generate OI divergence hypotheses at 1h resolution."""
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
                claim = (f"{signal_feat} divergence gated by "
                         f"{gate_feat}{direction}{thresh} predicts 1h returns")

                if claim in existing_claims:
                    continue

                test_protocol = [
                    f"profile scalp --symbol BTC --data {data_dir} "
                    f"--timeframe 1h --forward-test",
                ]

                priority = 0.45
                if "divergence" in signal_feat:
                    priority += 0.15  # Direct divergence signal
                if gate_feat == "regime_absorption_1h":
                    priority += 0.1  # Absorption confirms accumulation

                h = Hypothesis.create(
                    claim=claim,
                    generator="oi_divergence",
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

    log.info("OI divergence generator: %d new hypotheses", len(hypotheses))
    return hypotheses
