"""Ensemble Gate Generator — tests intersection of two regime gates.

For each pair of single gates that individually passed discovery,
emits a hypothesis testing gate_A AND gate_B. The intersection is
expected to produce higher IC at lower coverage.

Priority boost for uncorrelated gate pairs (e.g. entropy x illiquidity)
since correlated gates (e.g. two entropy features) add little information.

Schedule: runs after systematic generator has populated the graveyard
with enough single-gate results.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional

from ..hypothesis import Hypothesis, GeneratorStats
from ..hypothesis_queue import HypothesisQueue

log = logging.getLogger(__name__)

# Gate categories — pairs across categories get a priority boost
GATE_CATEGORIES = {
    "entropy": {"ent_book_shape", "ent_tick_5s", "ent_tick_30s", "ent_tick_1m",
                "ent_permutation_returns_16", "ent_spread_dispersion"},
    "illiquidity": {"illiq_kyle_100", "illiq_composite", "illiq_amihud_100"},
    "toxicity": {"toxic_vpin_50", "toxic_adverse_selection", "toxic_index"},
    "volatility": {"vol_returns_1m", "vol_returns_5m", "vol_ratio_short_long"},
    "regime": {"derived_regime_type_score", "derived_regime_confidence"},
}


def _gate_category(gate_str: str) -> str:
    """Return the category of a gate feature (e.g. 'ent_book_shape<P20' -> 'entropy')."""
    # Extract the feature name before the operator
    for op in ("<", ">"):
        if op in gate_str:
            feat = gate_str.split(op)[0]
            for cat, members in GATE_CATEGORIES.items():
                if feat in members:
                    return cat
    return "unknown"


def _extract_passing_gates(queue: HypothesisQueue) -> list[dict]:
    """Find single-gate hypotheses that passed discovery (any terminal status beyond queued).

    Returns list of dicts with signal, gate, ic, hypothesis_id.
    """
    passing = []
    seen = set()
    for h in queue._all:
        if h.status in ("queued", "running"):
            continue
        gate = h.thresholds.get("regime_gate")
        if not gate:
            continue
        # Extract IC from results
        ic = 0.0
        if h.results and "gate_results" in h.results:
            for gr in h.results["gate_results"]:
                msg = gr.get("msg", "")
                if "IC=" in msg and "PASS" in msg and "dIC=" not in msg:
                    try:
                        ic = float(msg.split("IC=")[1].split()[0])
                    except (IndexError, ValueError):
                        pass
        if ic < 0.10:
            continue
        # Extract signal feature from claim
        signal = _extract_signal_from_claim(h.claim)
        key = (signal, gate)
        if key in seen:
            continue
        seen.add(key)
        passing.append({
            "signal": signal,
            "gate": gate,
            "ic": ic,
            "hypothesis_id": h.id,
            "dIC": _extract_dIC(h),
        })
    return passing


def _extract_signal_from_claim(claim: str) -> str:
    """Extract the signal feature name from a hypothesis claim."""
    # Claims have form: "imbalance_qty_l1 gated by ent_book_shape<P20 predicts 5s returns"
    return claim.split(" gated by")[0].strip() if " gated by" in claim else claim.split()[0]


def _extract_dIC(h: Hypothesis) -> float:
    """Extract dIC value from hypothesis results."""
    if not h.results or "gate_results" not in h.results:
        return 0.0
    for gr in h.results["gate_results"]:
        msg = gr.get("msg", "")
        if "dIC=" in msg:
            try:
                return float(msg.split("dIC=")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return 0.0


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Generate ensemble gate hypotheses from pairs of passing single gates."""
    passing = _extract_passing_gates(queue)
    if len(passing) < 2:
        log.info("Ensemble generator: need >= 2 passing gates, have %d", len(passing))
        return []

    existing_claims = {h.claim for h in queue._all}
    hypotheses = []

    dates = sorted(manifest.get("dates", {}).keys())
    if not dates:
        return []
    latest_date = dates[-1]
    data_dir = f"data/features/{latest_date}"

    # Group passing gates by signal feature
    by_signal: dict[str, list[dict]] = {}
    for p in passing:
        by_signal.setdefault(p["signal"], []).append(p)

    for signal, gates in by_signal.items():
        if len(gates) < 2:
            continue

        for g_a, g_b in combinations(gates, 2):
            gate_a = g_a["gate"]
            gate_b = g_b["gate"]

            # Skip if same gate feature (e.g. ent_book_shape<P20 and ent_book_shape<P40)
            feat_a = gate_a.split("<")[0].split(">")[0]
            feat_b = gate_b.split("<")[0].split(">")[0]
            if feat_a == feat_b:
                continue

            claim = (f"{signal} gated by {gate_a} AND {gate_b} "
                     f"predicts 5s returns")
            if claim in existing_claims:
                continue

            test_protocol = [
                f"spannung regime --data {data_dir} --symbol BTC",
                f"profile scalp --symbol BTC --data {data_dir} --forward-test",
            ]

            # Priority: cross-category pairs get a boost
            cat_a = _gate_category(gate_a)
            cat_b = _gate_category(gate_b)
            cross_category = cat_a != cat_b and cat_a != "unknown" and cat_b != "unknown"

            # Base priority from constituent ICs
            avg_ic = (g_a["ic"] + g_b["ic"]) / 2
            priority = avg_ic * 0.8  # Slightly below single-gate priority
            if cross_category:
                priority += 0.15  # Uncorrelated gates are more valuable

            h = Hypothesis.create(
                claim=claim,
                generator="ensemble",
                test_protocol=test_protocol,
                priority=priority,
                thresholds={
                    "min_ic": 0.10,
                    "min_dIC": 0.05,
                    "min_coverage": 0.05,  # Lower coverage acceptable for ensemble
                    "min_hours": 4,
                    "symbols": ["BTC"],
                    "min_oos_dates": 1,
                    "min_symbols": 2,
                    "regime_gate": gate_a,       # Primary gate (used by IC check)
                    "regime_gate_b": gate_b,     # Secondary gate (for ensemble)
                    "ensemble": True,
                    "horizon_s": 5.0,
                },
                parent_id=f"{g_a['hypothesis_id']}+{g_b['hypothesis_id']}",
            )
            hypotheses.append(h)

    log.info("Ensemble generator: %d hypotheses from %d passing gates across %d signals",
             len(hypotheses), len(passing), len(by_signal))
    return hypotheses
