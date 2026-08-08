# Analytical Process Framework
#
# Third first-class citizen of NAT: feature = what is computed,
# algorithm = how it trades, process = whether/where information about
# price action exists. See docs/tasks_assigned_12_6_26/process_concept.md.

from .base import (
    EvaluationProcess,
    Finding,
    Process,
    ProcessContext,
    ProcessResult,
    TransformProcess,
    partition_usable_columns,
)
from .registry import get_process, list_processes, list_processes_by_kind, register
from .fdr import (
    FdrReport,
    apply_process_fdr,
    default_ledger_path,
    read_ledger,
    record_sweep,
)
from .standing import (
    StandingEval,
    audit_standing_evals,
    get_standing_eval,
    list_standing_evals,
    run_standing_eval,
)
from .surface import (
    SURFACE_COLUMNS,
    aggregate_from_index,
    build_surface,
    load_surface,
    render_surface,
    save_surface,
)

# Import process modules so @register decorators fire
from . import ic_horizon  # noqa: F401,E402
from . import info_theory  # noqa: F401,E402
from . import spectral  # noqa: F401,E402
from . import ml_importance  # noqa: F401,E402
from . import labeling  # noqa: F401,E402
from . import mi_combiner  # noqa: F401,E402
from . import mi_stability  # noqa: F401,E402
from . import pca_combo  # noqa: F401,E402
from . import persistence_stats  # noqa: F401,E402
from . import residualize  # noqa: F401,E402
from . import agreement_gate_eval  # noqa: F401,E402
from . import conditional_predictability  # noqa: F401,E402
from . import horizon_label_scan  # noqa: F401,E402
# XS-10: without these imports the @register decorators never fire, so the units are
# invisible to `get_process` and every run must bypass the runner — which is exactly
# why the PROC-13 ledger was empty despite 13 trials being spent.
from . import xs_rank_predictability  # noqa: F401,E402
from . import xs_persistence  # noqa: F401,E402

__all__ = [
    "Process", "EvaluationProcess", "TransformProcess",
    "ProcessContext", "ProcessResult", "Finding",
    "partition_usable_columns",
    "register", "get_process", "list_processes", "list_processes_by_kind",
    "apply_process_fdr", "FdrReport", "record_sweep", "read_ledger", "default_ledger_path",
    "StandingEval", "list_standing_evals", "get_standing_eval", "audit_standing_evals",
    "run_standing_eval",
    "SURFACE_COLUMNS", "build_surface", "save_surface", "load_surface",
    "aggregate_from_index", "render_surface",
]
