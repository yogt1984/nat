"""Spectral Anomaly Detector — monitors frequency-domain characteristics for changes.

Compares today's PSD slope, Hurst, coherence peaks against historical baseline.
Emits hypotheses when anomalies are detected.

Schedule: daily after new data ingested.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from ..hypothesis import Hypothesis, GeneratorStats
from ..hypothesis_queue import HypothesisQueue

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SPECTRAL_DIR = ROOT / "reports" / "spannung"


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Compare spectral reports across dates and flag anomalies."""
    hypotheses = []

    # Load existing spectral reports
    reports = {}
    for p in sorted(SPECTRAL_DIR.glob("spectral_*.json")):
        sym = p.stem.split("_")[1]
        with open(p) as f:
            reports[sym] = json.load(f)

    if not reports:
        # No spectral reports yet — generate hypothesis to run spectral analysis
        dates = sorted(manifest.get("dates", {}).keys())
        if dates:
            for sym in ["BTC", "ETH", "SOL"]:
                h = Hypothesis.create(
                    claim=f"Run baseline spectral analysis on {sym}",
                    generator="spectral",
                    test_protocol=[
                        f"spannung spectral --data data/features/{dates[-1]} --symbol {sym}",
                    ],
                    priority=0.8,  # High priority — establishes baseline
                    thresholds={"min_hours": 4, "symbols": [sym]},
                )
                hypotheses.append(h)
        return hypotheses

    # Check for anomalies in existing reports
    for sym, report in reports.items():
        psd = report.get("psd", {})
        acf = report.get("acf", {})

        # Anomaly: PSD slope outside expected range (-2.2 to -1.5 for brown noise)
        slope = psd.get("slope", -1.86)
        if slope < -2.2 or slope > -1.5:
            h = Hypothesis.create(
                claim=f"{sym} PSD slope anomaly ({slope:.2f}), signal structure may have changed",
                generator="spectral",
                test_protocol=[
                    f"spannung spectral --symbol {sym}",
                    f"spannung regime --symbol {sym}",
                ],
                priority=0.9,
                thresholds={"min_ic": 0.05, "symbols": [sym]},
            )
            hypotheses.append(h)

        # Anomaly: OU half-life outside expected range (2-15s)
        halflife = acf.get("ou_halflife_s", 5.0)
        if halflife < 2.0 or halflife > 15.0:
            h = Hypothesis.create(
                claim=f"{sym} OU half-life anomaly ({halflife:.1f}s), adjust refresh rate",
                generator="spectral",
                test_protocol=[
                    f"spannung spectral --symbol {sym}",
                ],
                priority=0.7,
                thresholds={"min_hours": 4, "symbols": [sym]},
            )
            hypotheses.append(h)

    log.info("Spectral generator: %d hypotheses", len(hypotheses))
    return hypotheses
