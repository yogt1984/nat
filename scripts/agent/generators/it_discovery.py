"""IT Discovery Generator — hypotheses from information-theoretic analysis.

Reads IT engine state and generates hypotheses for features that:
  1. Have cost-viable MI (MI > I_min at some horizon)
  2. Show positive interaction information (synergy with entropy gating)

Priority = CMI × (1 + II), rewarding features that benefit from entropy gating.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from ..hypothesis import Hypothesis, GeneratorStats
from ..hypothesis_queue import HypothesisQueue

log = logging.getLogger(__name__)

Hypothesis.register_prefix("it_discovery", "ITD")

# Horizons to test (bar-level)
HORIZONS = ["5min", "25min", "50min"]

# Default entropy gates to test
GATES = [
    ("ent_tick_5s", "<", "P30"),
    ("ent_book_shape_std", "<", "P30"),
]


def _load_it_state(symbol: str, state_dir: str = "data/it_engine") -> Optional[dict]:
    """Load IT engine state for a symbol."""
    path = os.path.join(state_dir, f"state_{symbol}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Generate hypotheses from IT engine output.

    Reads the IT engine state file and creates hypotheses for
    cost-viable features with positive interaction information.
    """
    existing_claims = {h.claim for h in queue._all}
    hypotheses = []

    dates = sorted(manifest.get("dates", {}).keys())
    if not dates:
        return []
    latest_date = dates[-1]
    data_dir = f"data/features/{latest_date}"

    symbols = manifest.get("symbols", ["BTC"])
    if isinstance(symbols, str):
        symbols = [symbols]

    for symbol in symbols:
        state = _load_it_state(symbol)
        if state is None:
            log.info("No IT state for %s, skipping", symbol)
            continue

        cost_viable = state.get("cost_viable", {})
        mi_matrix = state.get("mi_matrix", {})
        cmi_matrix = state.get("cmi_matrix", {})
        interaction = state.get("interaction", {})

        # Filter to cost-viable features
        viable_features = [f for f, v in cost_viable.items() if v]
        if not viable_features:
            log.info("No cost-viable features for %s", symbol)
            continue

        for feat in viable_features:
            mi_vals = mi_matrix.get(feat, {})
            cmi_vals = cmi_matrix.get(feat, {})
            ii = interaction.get(feat, 0.0)

            # Best MI across horizons
            best_mi = max(mi_vals.values()) if mi_vals else 0.0
            best_cmi = max(cmi_vals.values()) if cmi_vals else 0.0

            for horizon in HORIZONS:
                # Without gate
                claim_nogat = (
                    f"IT:{feat} predicts {symbol} {horizon} returns "
                    f"(MI={best_mi:.4f} bits)"
                )
                if claim_nogat not in existing_claims:
                    priority = float(best_mi)
                    h = Hypothesis.create(
                        claim=claim_nogat,
                        generator="it_discovery",
                        test_protocol=[
                            f"profile scalp --symbol {symbol} --data {data_dir} "
                            f"--timeframe {horizon} --forward-test",
                        ],
                        priority=priority,
                        thresholds={
                            "min_ic": 0.08,
                            "min_dIC": 0.03,
                            "min_coverage": 0.10,
                            "min_hours": 4,
                            "symbols": [symbol],
                            "min_oos_dates": 2,
                            "min_symbols": 2,
                            "horizon_s": _horizon_to_seconds(horizon),
                            "timeframe": horizon,
                            "it_mi": best_mi,
                            "it_ii": ii,
                        },
                    )
                    hypotheses.append(h)

                # With entropy gate (only if synergistic: II > 0)
                if ii > 0:
                    for gate_feat, direction, thresh in GATES:
                        claim_gated = (
                            f"IT:{feat} gated by {gate_feat}{direction}{thresh} "
                            f"predicts {symbol} {horizon} returns "
                            f"(CMI={best_cmi:.4f}, II={ii:+.4f})"
                        )
                        if claim_gated in existing_claims:
                            continue

                        # Priority: CMI × (1 + II) — rewards synergy
                        priority = float(best_cmi * (1.0 + max(ii, 0.0)))

                        h = Hypothesis.create(
                            claim=claim_gated,
                            generator="it_discovery",
                            test_protocol=[
                                f"profile scalp --symbol {symbol} "
                                f"--data {data_dir} --timeframe {horizon} "
                                f"--forward-test",
                            ],
                            priority=priority,
                            thresholds={
                                "min_ic": 0.08,
                                "min_dIC": 0.03,
                                "min_coverage": 0.10,
                                "min_hours": 4,
                                "symbols": [symbol],
                                "min_oos_dates": 2,
                                "min_symbols": 2,
                                "regime_gate": f"{gate_feat}{direction}{thresh}",
                                "horizon_s": _horizon_to_seconds(horizon),
                                "timeframe": horizon,
                                "it_mi": best_mi,
                                "it_cmi": best_cmi,
                                "it_ii": ii,
                            },
                        )
                        hypotheses.append(h)

    log.info("IT discovery generator: %d new hypotheses", len(hypotheses))
    return hypotheses


def _horizon_to_seconds(horizon: str) -> float:
    """Convert horizon string to seconds."""
    if horizon.endswith("min"):
        return float(horizon[:-3]) * 60
    if horizon.endswith("h"):
        return float(horizon[:-1]) * 3600
    return 300.0  # default 5min
