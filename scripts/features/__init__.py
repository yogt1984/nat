"""Python-side derived features.

(The relative-value pairs strategy lives in scripts/strategies/
relative_value_pairs.py — it is pair-level, not a per-symbol feature.)

Features here are computed from existing parquet columns — no ingestor or
schema changes. Per the design rule in
docs/tasks_assigned_12_6_26/feature_algorithm_gaps.md, new features are born
in Python, IC-validated on real data, and promoted into the Rust ingestor
only if they earn a place in the feature-vector contract.
"""

from features.har_rv import (  # noqa: F401
    HAR_RV_FEATURES,
    HarRvEstimator,
    compute_har_rv,
    compute_rv_components,
)
from features.microprice import (  # noqa: F401
    MICROPRICE_FEATURES,
    MicropriceEstimator,
    compute_microprice,
    fit_per_symbol,
)
from features.multilevel_ofi import (  # noqa: F401
    MULTILEVEL_OFI_FEATURES,
    OFIEstimator,
    compute_multilevel_ofi,
)
from features.realized_moments import (  # noqa: F401
    REALIZED_MOMENTS_FEATURES,
    compute_realized_moments,
)
from features.settlement_clock import (  # noqa: F401
    SETTLEMENT_CLOCK_FEATURES,
    compute_settlement_clock,
)
