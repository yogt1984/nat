"""
Core abstractions for the analytical process framework.

A Process is the third first-class citizen of NAT, alongside features and
algorithms:

  - feature   : what is computed (ingestor columns, derived Python features)
  - algorithm : how it trades (MicrostructureAlgorithm)
  - process   : whether/where information about price action exists

A Process is an analytical description — statistical, signal-processing, or
ML — that identifies whether a feature, or a derivative/combination of
features, carries information about future price action.

Two kinds share one registry/CLI/persistence surface:

  - EvaluationProcess : scores existing columns        -> ProcessResult
  - TransformProcess  : produces NEW derived series    -> (DataFrame, ProcessResult)
                        (PCA components, triple-barrier labels, ...) which are
                        themselves evaluable by any EvaluationProcess.

Mirrors the algorithm framework in `scripts/algorithms/base.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

# Meta columns never treated as features (bar-level and tick-level)
META_COLUMNS = {
    "timestamp", "timestamp_ns", "symbol", "sequence_id", "datetime",
    "bar_start", "bar_end", "tick_count",
}

# Minimum valid observations for a column to be usable by default
DEFAULT_MIN_OBS = 50


@dataclass(frozen=True)
class ProcessContext:
    """Immutable run context handed to every process.

    `extra_sources` is the extension point for exogenous feature sources
    (e.g. the prism narrative/hype series): DataFrames merged onto the bars
    by the runner before evaluation. Processes operate on columns and never
    know where a column came from.
    """
    symbol: str
    timeframe: str
    price_col: str
    horizons: dict[str, int]          # horizon name -> bars (or ticks)
    costs: dict                       # config/costs.toml via utils.costs.load_costs()
    data_dir: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sample_rate_hz: float = 10.0      # tick-level sampling rate (100ms ticks)
    target_col: Optional[str] = None  # alternative target (e.g. tb_label)
    extra_sources: dict[str, Any] = field(default_factory=dict)


@dataclass
class Finding:
    """One (feature, horizon, metric) verdict."""
    feature: str
    horizon: Optional[str]
    metric: str                       # "ic_mean", "mi_bits", "te_bits", ...
    value: float
    threshold: Optional[float] = None
    p_value: Optional[float] = None
    p_adjusted: Optional[float] = None
    informative: bool = False
    extras: dict = field(default_factory=dict)


@dataclass
class ProcessResult:
    """Complete record of one process run (JSON schema_version 1)."""
    run_id: str
    process: str
    kind: str
    symbol: str
    timeframe: str
    params: dict
    schema_version: int = 1
    data: dict = field(default_factory=dict)        # dir, dates, n_rows, n_bars, fingerprint
    provenance: dict = field(default_factory=dict)  # git_sha, dirty, generated_at
    features_tested: list[str] = field(default_factory=list)
    features_skipped: list[dict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    derived: Optional[dict] = None                  # columns, parquet, scored_by
    summary: dict = field(default_factory=dict)     # n_tested, n_informative, top, runtime_s, error

    def finalize(self, runtime_s: float = 0.0, error: Optional[str] = None) -> "ProcessResult":
        """Fill the summary block from findings."""
        informative = [f for f in self.findings if f.informative]
        ranked = sorted(self.findings, key=lambda f: abs(f.value), reverse=True)
        self.summary = {
            "n_tested": len(self.features_tested),
            "n_informative": len(informative),
            "top": [
                {"feature": f.feature, "horizon": f.horizon,
                 "metric": f.metric, "value": f.value}
                for f in ranked[:5]
            ],
            "runtime_s": round(runtime_s, 1),
            "error": error,
        }
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class Process(ABC):
    """Base class for analytical processes.

    Subclasses declare tunable parameters in PARAMS as
    ``{name: (default, doc)}``; constructor kwargs override defaults and
    unknown kwargs raise. Section names in ``config/processes.toml`` match
    ``name()`` and feed the constructor (the algorithms.toml convention).
    """

    kind: str = "evaluation"          # set by intermediate ABCs
    data_level: str = "bars"          # "bars" | "ticks" — what the runner feeds
    PARAMS: dict[str, tuple] = {}

    def __init__(self, **kwargs):
        unknown = set(kwargs) - set(self.PARAMS)
        if unknown:
            raise TypeError(
                f"{type(self).__name__}: unknown params {sorted(unknown)}. "
                f"Known: {sorted(self.PARAMS)}"
            )
        self.params = {k: kwargs.get(k, spec[0]) for k, spec in self.PARAMS.items()}

    @abstractmethod
    def name(self) -> str:
        """Unique process name (== registry key == config section)."""
        ...

    def required_columns(self, available: list[str]) -> list[str]:
        """Columns to load, chosen from the available schema.

        Default: every non-meta column, optionally narrowed by a
        ``features`` param (list of name prefixes). Receives the available
        column names so processes can pattern-select without loading data.
        """
        cols = [c for c in available if c not in META_COLUMNS]
        patterns = self.params.get("features")
        if patterns:
            cols = [c for c in cols if any(c.startswith(p) for p in patterns)]
        return cols

    def describe(self) -> dict:
        """Machine-readable spec — enough for an agent generator to target."""
        doc = (self.__doc__ or "").strip().split("\n\n")[0]
        return {
            "name": self.name(),
            "kind": self.kind,
            "data_level": self.data_level,
            "doc": doc,
            "params": {
                k: {"default": spec[0], "doc": spec[1] if len(spec) > 1 else ""}
                for k, spec in self.PARAMS.items()
            },
        }


class EvaluationProcess(Process):
    """Process that scores existing columns for price-action information."""

    kind = "evaluation"

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, ctx: ProcessContext) -> ProcessResult:
        """Evaluate single-symbol data (bars or ticks per ``data_level``)."""
        ...


class TransformProcess(Process):
    """Process that derives NEW series from existing columns.

    The derived DataFrame shares the input index, so the runner can chain
    it straight into any EvaluationProcess (``--score-with ic_horizon``).
    """

    kind = "transform"

    @abstractmethod
    def transform(
        self, df: pd.DataFrame, ctx: ProcessContext,
    ) -> tuple[pd.DataFrame, ProcessResult]:
        ...


def partition_usable_columns(
    df: pd.DataFrame,
    columns: list[str],
    min_obs: int = DEFAULT_MIN_OBS,
) -> tuple[list[str], list[dict]]:
    """Split columns into (usable, skipped-with-reason).

    The K2 guard: 82/236 ingestor columns may be all-NaN — processes skip
    them with a recorded reason instead of crashing.

    Reasons: "missing", "non_numeric", "all_nan", "constant", "n_valid=K<min".
    """
    usable: list[str] = []
    skipped: list[dict] = []
    for col in columns:
        if col not in df.columns:
            skipped.append({"feature": col, "reason": "missing"})
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            skipped.append({"feature": col, "reason": "non_numeric"})
            continue
        vals = series.to_numpy(dtype=np.float64, na_value=np.nan)
        finite = np.isfinite(vals)
        n_valid = int(finite.sum())
        if n_valid == 0:
            skipped.append({"feature": col, "reason": "all_nan"})
            continue
        if n_valid < min_obs:
            skipped.append({"feature": col, "reason": f"n_valid={n_valid}<{min_obs}"})
            continue
        if np.nanstd(vals) < 1e-15:
            skipped.append({"feature": col, "reason": "constant"})
            continue
        usable.append(col)
    return usable, skipped


def make_run_id(process: str, symbol: str, now: Optional[datetime] = None) -> str:
    """Deterministic-format run id: proc_<name>_<SYM>_<UTC stamp>."""
    now = now or datetime.now(timezone.utc)
    return f"proc_{process}_{symbol}_{now.strftime('%Y%m%dT%H%M%SZ')}"


def get_provenance() -> dict:
    """Git SHA + dirty flag; degrades to nulls until scripts/provenance.py lands (plan T2)."""
    try:
        from provenance import get_provenance as _gp  # type: ignore
        prov = _gp()
        prov.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        return prov
    except Exception:
        pass
    try:
        import subprocess
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent.parent
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
        dirty_out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return {
            "git_sha": sha,
            "dirty": bool(dirty_out),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return {
            "git_sha": None,
            "dirty": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
