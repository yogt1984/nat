"""
Composite Cluster Quality Scoring

Combines internal, stability, and external metrics into a single
actionable score for HMM readiness assessment.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

from .metrics import QualityMetrics
from .stability import StabilityMetrics
from .validation import ExternalValidationResults


@dataclass
class ClusterQualityScore:
    """
    Composite cluster quality assessment.

    Combines multiple metrics into a single score [0, 1].
    """
    # Internal metrics (0-1 scale)
    silhouette: float = 0.0
    davies_bouldin_normalized: float = 0.0  # Inverted, capped

    # Stability metrics (0-1 scale)
    bootstrap_stability: float = 0.0
    temporal_stability: float = 0.0

    # External metrics (0-1 scale)
    return_significance: float = 0.0
    volatility_significance: float = 0.0

    # Optional: raw metrics for reference
    raw_metrics: Optional[QualityMetrics] = None
    raw_stability: Optional[StabilityMetrics] = None
    raw_validation: Optional[ExternalValidationResults] = None

    def compute_composite(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted composite score.

        Default weights emphasize stability and predictive power
        over internal metrics alone.
        """
        if weights is None:
            weights = {
                "silhouette": 0.15,
                "davies_bouldin": 0.10,
                "bootstrap_stability": 0.20,
                "temporal_stability": 0.20,
                "return_significance": 0.20,
                "volatility_significance": 0.15,
            }

        score = (
            weights.get("silhouette", 0) * max(0, self.silhouette) +
            weights.get("davies_bouldin", 0) * self.davies_bouldin_normalized +
            weights.get("bootstrap_stability", 0) * self.bootstrap_stability +
            weights.get("temporal_stability", 0) * self.temporal_stability +
            weights.get("return_significance", 0) * self.return_significance +
            weights.get("volatility_significance", 0) * self.volatility_significance
        )

        return min(1.0, max(0.0, score))

    def get_grade(self) -> str:
        """Human-readable quality grade."""
        score = self.compute_composite()
        if score >= 0.8:
            return "A - Excellent: Ready for HMM"
        elif score >= 0.6:
            return "B - Good: Minor refinements needed"
        elif score >= 0.4:
            return "C - Fair: Significant refinements needed"
        elif score >= 0.2:
            return "D - Poor: Consider different features"
        else:
            return "F - Failed: No meaningful structure"

    def is_hmm_ready(self) -> bool:
        """Check if quality meets HMM readiness thresholds."""
        return (
            self.silhouette >= 0.3 and
            self.davies_bouldin_normalized >= 0.25 and  # DB < 1.5
            self.bootstrap_stability >= 0.6 and
            self.temporal_stability >= 0.5 and
            (self.return_significance >= 0.5 or self.volatility_significance >= 0.5)
        )

    def get_weaknesses(self) -> list:
        """Identify weak areas for targeted improvement."""
        weaknesses = []

        if self.silhouette < 0.3:
            weaknesses.append(("silhouette", "Clusters overlap significantly"))
        if self.davies_bouldin_normalized < 0.25:
            weaknesses.append(("davies_bouldin", "Poor cluster separation"))
        if self.bootstrap_stability < 0.6:
            weaknesses.append(("bootstrap_stability", "Clusters unstable to resampling"))
        if self.temporal_stability < 0.5:
            weaknesses.append(("temporal_stability", "Clusters drift over time"))
        if self.return_significance < 0.5:
            weaknesses.append(("return_significance", "No return predictive power"))
        if self.volatility_significance < 0.5:
            weaknesses.append(("volatility_significance", "No volatility differentiation"))

        return weaknesses

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "CLUSTER QUALITY ASSESSMENT",
            "=" * 60,
            "",
            f"Overall Grade: {self.get_grade()}",
            f"Composite Score: {self.compute_composite():.3f}",
            f"HMM Ready: {'Yes' if self.is_hmm_ready() else 'No'}",
            "",
            "Component Scores:",
            f"  Silhouette:           {self.silhouette:.3f}",
            f"  Davies-Bouldin (inv): {self.davies_bouldin_normalized:.3f}",
            f"  Bootstrap Stability:  {self.bootstrap_stability:.3f}",
            f"  Temporal Stability:   {self.temporal_stability:.3f}",
            f"  Return Significance:  {self.return_significance:.3f}",
            f"  Volatility Signif.:   {self.volatility_significance:.3f}",
        ]

        weaknesses = self.get_weaknesses()
        if weaknesses:
            lines.extend(["", "Weaknesses:"])
            for metric, desc in weaknesses:
                lines.append(f"  - {metric}: {desc}")

        lines.append("=" * 60)
        return "\n".join(lines)


def compute_quality_score(
    metrics: QualityMetrics,
    stability: StabilityMetrics,
    validation: ExternalValidationResults,
) -> ClusterQualityScore:
    """
    Compute composite quality score from component metrics.

    Args:
        metrics: Internal quality metrics
        stability: Stability metrics
        validation: External validation results

    Returns:
        ClusterQualityScore with all components
    """
    # Normalize Davies-Bouldin (invert, cap at 2)
    db_normalized = 1.0 - min(metrics.davies_bouldin, 2.0) / 2.0

    # Convert return significance from p-values
    return_sig = 0.0
    for horizon_result in validation.return_differentiation.values():
        if horizon_result.significant:
            # Scale by effect size
            return_sig = max(return_sig, min(1.0, horizon_result.eta_squared * 10))

    # Volatility significance
    vol_sig = 0.0
    if validation.volatility_differentiation:
        if validation.volatility_differentiation.significant:
            vol_sig = 1.0

    # Temporal stability (may be None)
    temporal = 0.5  # Default
    if stability.temporal:
        temporal = stability.temporal.temporal_ari

    return ClusterQualityScore(
        silhouette=metrics.silhouette.overall,
        davies_bouldin_normalized=db_normalized,
        bootstrap_stability=stability.bootstrap.mean_ari,
        temporal_stability=temporal,
        return_significance=return_sig,
        volatility_significance=vol_sig,
        raw_metrics=metrics,
        raw_stability=stability,
        raw_validation=validation,
    )
