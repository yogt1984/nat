"""PROC-13: FDR/DSR on the process layer + the cross-run sweep ledger.

A process that sweeps a grid of ``(combo × horizon × label × regime)`` cells *guarantees*
a great-looking argmax by chance — with a thousand cells at α=0.05 you expect ~50 spurious
"discoveries". FDR (Benjamini–Hochberg) is already enforced on features
(``alpha/screener.py``) but not on the process-layer sweeps. This module is the shared step:

  1. collect every cell's p-value (produced by the PROC-12 null-calibration),
  2. apply Benjamini–Hochberg at ``q = alpha``,
  3. annotate every ``Finding`` with its BH q-value (``p_adjusted``) and compose the
     informative flag as ``informative_in AND (q ≤ alpha)`` — FDR can only *tighten*,
  4. surface the argmax **only** as "argmax, BH-q = …" (never bare),
  5. record the sweep in an append-only program-level ledger
     ``(process, target, n_tested, git_sha, alpha, n_discoveries)`` so multiple-testing is
     accounted for *across* runs, not just within one sweep.

Reuses the battle-tested ``benjamini_hochberg`` from ``alpha/screener.py`` (the spec's
"reuse the existing FDR machinery"). Spec: docs/specs/process_layer.md §13.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np

from alpha.screener import benjamini_hochberg

from .base import Finding, ProcessResult

DEFAULT_FDR_ALPHA = 0.05


@dataclass
class FdrReport:
    """Outcome of one Benjamini–Hochberg pass over a sweep's cells."""
    alpha: float
    n_cells: int                       # cells considered (incl. p-less ones)
    n_pvalued: int                     # cells that carried a p-value (entered BH)
    n_discoveries: int                 # cells with q ≤ alpha
    argmax: Optional[dict] = None      # the headline cell, always annotated with q_value
    discoveries: list[dict] = field(default_factory=list)  # every surviving cell + q


def _as_findings(results) -> list[Finding]:
    """Accept a Finding list, a ProcessResult, or a list of either; return the flat list."""
    if isinstance(results, ProcessResult):
        return results.findings
    if isinstance(results, Finding):
        return [results]
    out: list[Finding] = []
    for r in results:
        if isinstance(r, ProcessResult):
            out.extend(r.findings)
        elif isinstance(r, Finding):
            out.append(r)
        else:
            raise TypeError(f"apply_process_fdr: unexpected item {type(r).__name__}")
    return out


def _cell(f: Finding, q: Optional[float]) -> dict:
    return {
        "feature": f.feature,
        "horizon": f.horizon,
        "metric": f.metric,
        "value": f.value,
        "p_value": f.p_value,
        "q_value": q,
        "extras": dict(f.extras),
    }


def apply_process_fdr(
    results: Union[ProcessResult, Finding, list],
    *,
    alpha: float = DEFAULT_FDR_ALPHA,
) -> FdrReport:
    """Benjamini–Hochberg over all p-valued cells of a sweep; annotate & report in place.

    Every ``Finding`` that carries a ``p_value`` gets its BH q-value written to
    ``p_adjusted`` and its ``informative`` flag recomposed as ``informative AND q ≤ alpha``
    (so FDR only ever removes false positives, never invents a discovery). Cells without a
    p-value are left untouched — they were never part of this multiple-testing family.

    The argmax (largest |effect|) is always returned WITH its q-value, so no headline cell
    is ever reported without its correction.
    """
    findings = _as_findings(results)
    n_cells = len(findings)

    pvalued = [f for f in findings if f.p_value is not None]
    q_by_id: dict[int, float] = {}
    if pvalued:
        pvals = np.array([f.p_value for f in pvalued], dtype=np.float64)
        qvals = benjamini_hochberg(pvals, alpha=alpha)
        for f, q in zip(pvalued, qvals):
            qf = None if (q is None or np.isnan(q)) else float(q)
            f.p_adjusted = qf
            f.informative = bool(f.informative and qf is not None and qf <= alpha)
            if qf is not None:
                q_by_id[id(f)] = qf

    discoveries = [
        _cell(f, q_by_id.get(id(f))) for f in pvalued
        if q_by_id.get(id(f)) is not None and q_by_id[id(f)] <= alpha
    ]

    argmax = None
    if findings:
        top = max(findings, key=lambda f: abs(f.value) if f.value is not None else -np.inf)
        argmax = _cell(top, q_by_id.get(id(top)))

    return FdrReport(
        alpha=alpha,
        n_cells=n_cells,
        n_pvalued=len(pvalued),
        n_discoveries=len(discoveries),
        argmax=argmax,
        discoveries=discoveries,
    )


# --------------------------------------------------------------------------- #
# Cross-run ledger (=B3): program-level multiple-testing accounting.          #
# Append-only JSONL so every sweep across the whole program is on the record. #
# --------------------------------------------------------------------------- #

def default_ledger_path() -> Path:
    """Standard location for the program-level sweep ledger."""
    try:
        import nat_paths
        return nat_paths.state_dir("processes") / "fdr_ledger.jsonl"
    except Exception:
        return Path(__file__).resolve().parents[2] / "data" / "processes" / "fdr_ledger.jsonl"


def record_sweep(
    path: Union[str, Path],
    *,
    process: str,
    target: str,
    n_tested: int,
    git_sha: Optional[str] = None,
    alpha: float = DEFAULT_FDR_ALPHA,
    n_discoveries: Optional[int] = None,
    **extra,
) -> dict:
    """Append one sweep's accounting row to the ledger; returns the row written.

    The tuple ``(process, target, n_tested, git_sha)`` is the program-level record the spec
    calls for — enough to reconstruct how many tests the whole program has run against a
    target, so a later meta-correction (or audit) is possible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "process": process,
        "target": target,
        "n_tested": int(n_tested),
        "n_discoveries": None if n_discoveries is None else int(n_discoveries),
        "alpha": float(alpha),
        "git_sha": git_sha,
        **extra,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def read_ledger(path: Union[str, Path]) -> list[dict]:
    """Read every sweep row from the ledger (empty list if it does not exist yet)."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
