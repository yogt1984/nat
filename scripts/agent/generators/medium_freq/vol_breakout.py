"""Vol Breakout Generator — volatility ratio breakout events at 5min bars.

Tests two classes of hypotheses:
  1. Continuation: high vol_ratio + rising OI -> breakout follows through
  2. Mean-reversion: high vol_ratio + stable/falling OI -> vol spike reverts

Gate condition: low tick entropy (directional, not noise).
"""

from __future__ import annotations

import logging
from typing import Optional

from ...hypothesis import Hypothesis, GeneratorStats
from ...hypothesis_queue import HypothesisQueue

log = logging.getLogger(__name__)

Hypothesis.register_prefix("vol_breakout", "VBK")

# Primary breakout features
SIGNAL_FEATURES = [
    "vol_ratio_short_long",
    "vol_zscore",
]

# Entropy gate for quality filter
GATE_FEATURES = [
    "ent_tick_1m",
    "ent_tick_5s",
]

THRESHOLDS = ["P20", "P40"]


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Generate vol breakout hypotheses (continuation + mean-reversion)."""
    existing_claims = {h.claim for h in queue._all}
    hypotheses = []

    dates = sorted(manifest.get("dates", {}).keys())
    if not dates:
        return []
    latest_date = dates[-1]
    data_dir = f"data/features/{latest_date}"

    for signal_feat in SIGNAL_FEATURES:
        for gate_feat in GATE_FEATURES:
            for thresh in THRESHOLDS:
                for regime_type in ("continuation", "reversion"):
                    claim = (f"{signal_feat} {regime_type} gated by "
                             f"{gate_feat}<{thresh} predicts 5min returns")

                    if claim in existing_claims:
                        continue

                    test_protocol = [
                        f"profile scalp --symbol BTC --data {data_dir} "
                        f"--timeframe 5min --forward-test",
                    ]

                    priority = 0.4
                    if regime_type == "continuation":
                        priority += 0.1  # Continuation is more actionable
                    if "vol_ratio" in signal_feat:
                        priority += 0.1

                    h = Hypothesis.create(
                        claim=claim,
                        generator="vol_breakout",
                        test_protocol=test_protocol,
                        priority=priority,
                        thresholds={
                            "min_ic": 0.08,
                            "min_dIC": 0.03,
                            "min_coverage": 0.05,
                            "min_hours": 4,
                            "symbols": ["BTC"],
                            "min_oos_dates": 2,
                            "min_symbols": 2,
                            "regime_gate": f"{gate_feat}<{thresh}",
                            "horizon_s": 300.0,
                            "timeframe": "5min",
                            "regime_type": regime_type,
                        },
                    )
                    hypotheses.append(h)

    log.info("Vol breakout generator: %d new hypotheses", len(hypotheses))
    return hypotheses
