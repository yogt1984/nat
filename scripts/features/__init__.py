"""Python-side derived features.

Features here are computed from existing parquet columns — no ingestor or
schema changes. Per the design rule in
docs/tasks_assigned_12_6_26/feature_algorithm_gaps.md, new features are born
in Python, IC-validated on real data, and promoted into the Rust ingestor
only if they earn a place in the feature-vector contract.
"""

from features.settlement_clock import (  # noqa: F401
    SETTLEMENT_CLOCK_FEATURES,
    compute_settlement_clock,
)
