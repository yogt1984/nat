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

from .validation import (
    compute_return_differentiation,
    compute_volatility_differentiation,
    compute_transition_matrix,
    compute_all_external_validation,
    ReturnDifferentiationResult,
    VolatilityDifferentiationResult,
    TransitionMatrixResult,
    ExternalValidationResults,
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
    # Validation
    "compute_return_differentiation",
    "compute_volatility_differentiation",
    "compute_transition_matrix",
    "compute_all_external_validation",
    "ReturnDifferentiationResult",
    "VolatilityDifferentiationResult",
    "TransitionMatrixResult",
    "ExternalValidationResults",
]
