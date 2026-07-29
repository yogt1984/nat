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

# Import process modules so @register decorators fire
from . import ic_horizon  # noqa: F401,E402
from . import info_theory  # noqa: F401,E402
from . import spectral  # noqa: F401,E402
from . import ml_importance  # noqa: F401,E402
from . import labeling  # noqa: F401,E402
from . import pca_combo  # noqa: F401,E402
from . import conditional_predictability  # noqa: F401,E402

__all__ = [
    "Process", "EvaluationProcess", "TransformProcess",
    "ProcessContext", "ProcessResult", "Finding",
    "partition_usable_columns",
    "register", "get_process", "list_processes", "list_processes_by_kind",
    "apply_process_fdr", "FdrReport", "record_sweep", "read_ledger", "default_ledger_path",
]
