"""
Cluster Quality Measurement Framework

Tools for validating GMM regime classification quality through:
- Internal validation metrics (silhouette, Davies-Bouldin, etc.)
- Stability analysis (bootstrap, temporal)
- External validation (return differentiation, volatility regimes)
- Composite scoring and agentic feature refinement
"""

from .metrics import (
    compute_silhouette,
    compute_davies_bouldin,
    compute_calinski_harabasz,
    compute_gap_statistic,
    compute_all_metrics,
    SilhouetteResult,
    GapStatisticResult,
    QualityMetrics,
)

from .stability import (
    compute_bootstrap_stability,
    compute_temporal_stability,
    compute_cross_symbol_stability,
    BootstrapStabilityResult,
    TemporalStabilityResult,
    StabilityMetrics,
)

__version__ = "0.1.0"

__all__ = [
    # Metrics
    "compute_silhouette",
    "compute_davies_bouldin",
    "compute_calinski_harabasz",
    "compute_gap_statistic",
    "compute_all_metrics",
    "SilhouetteResult",
    "GapStatisticResult",
    "QualityMetrics",
    # Stability
    "compute_bootstrap_stability",
    "compute_temporal_stability",
    "compute_cross_symbol_stability",
    "BootstrapStabilityResult",
    "TemporalStabilityResult",
    "StabilityMetrics",
]
