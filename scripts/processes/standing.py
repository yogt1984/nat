"""PROC-5: standing evaluations — the process-layer's recurring, scheduled evals.

An ad-hoc `nat process run` is a one-off; a *standing* evaluation is a named, recurring
chain that the discovery/agent cadence is expected to run and that we can audit ("has it
ever actually run?"). The founding entry is the 3-bar triple-barrier classifier:

    triple_barrier  --score-with mi_ksg  (target = tb_label)

i.e. derive the López de Prado 3-barrier label, then measure how much information each
feature carries about it — MI(feature; tb_label), gated by the PROC-12 null-calibration
(a label is not a tradeable return, so the fee-based i_min gate does not apply).

`audit_standing_evals()` answers the task's "audit whether ever run" by scanning the
persisted process-results index and confirming a matching label-mode run exists.
Spec: docs/specs/process_layer.md §5 (schedule the 3-bar classifier as standing eval).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import persistence


@dataclass(frozen=True)
class StandingEval:
    """One recurring evaluation: an optional transform chained into a scorer on a target."""
    name: str
    scorer: str                              # evaluation process name (e.g. mi_ksg)
    target: Optional[str] = None             # target column the scorer must use
    transform: Optional[str] = None          # transform run first (None = score directly)
    symbols: tuple[str, ...] = ("BTC", "ETH", "SOL")
    timeframe: str = "15min"
    note: str = ""


# The registry. Add recurring evals here; each becomes auditable + runnable by name.
STANDING_EVALS: dict[str, StandingEval] = {
    "barrier_3bar_mi": StandingEval(
        name="barrier_3bar_mi",
        transform="triple_barrier",
        scorer="mi_ksg",
        target="tb_label",
        symbols=("BTC", "ETH", "SOL"),
        timeframe="15min",
        note="3-bar triple-barrier label vs feature MI, PROC-12 null-gated (PROC-5).",
    ),
    "agreement_gate": StandingEval(
        name="agreement_gate",
        transform=None,
        scorer="agreement_gate_eval",
        target=None,
        symbols=("BTC", "ETH", "SOL"),
        timeframe="5min",
        note="A-1: conditional IC given fast/slow agreement vs a size-preserving gate "
             "permutation null. Promotes §5's pilot to a monitored fact, or kills it.",
    ),
}


def list_standing_evals() -> list[StandingEval]:
    return list(STANDING_EVALS.values())


def get_standing_eval(name: str) -> StandingEval:
    if name not in STANDING_EVALS:
        raise KeyError(
            f"unknown standing eval {name!r}; known: {sorted(STANDING_EVALS)}"
        )
    return STANDING_EVALS[name]


def _record_matches(record: dict, ev: StandingEval) -> bool:
    """True iff a persisted JSON record is a run of this standing eval.

    Identified structurally: the scorer process produced label-mode findings
    (horizon == "label") whose ``extras.target`` equals the eval's target. This
    distinguishes it from a plain forward-return run of the same scorer.
    """
    if record.get("process") != ev.scorer:
        return False
    for f in record.get("findings", []) or []:
        extras = f.get("extras") or {}
        if f.get("horizon") == "label" and extras.get("target") == ev.target:
            return True
    return False


def audit_standing_evals(
    db_path: Optional[Path | str] = None,
    limit: int = 500,
) -> list[dict]:
    """For each standing eval, report whether/when it has ever run.

    Reads the process-results index (newest first), loads each candidate run's JSON via
    its stored ``json_path``, and counts the records that match the eval. Returns one dict
    per eval: name/transform/scorer/target, ever_run, n_runs, last_run, symbols_seen.
    """
    db = Path(db_path) if db_path is not None else persistence.DEFAULT_DB_PATH
    out: list[dict] = []
    for ev in list_standing_evals():
        rows = persistence.list_results(process=ev.scorer, limit=limit, db_path=db)
        matches: list[dict] = []
        for row in rows:                       # newest-first
            jp = row.get("json_path")
            if not jp or not Path(jp).exists():
                continue
            try:
                rec = json.loads(Path(jp).read_text())
            except Exception:
                continue
            if _record_matches(rec, ev):
                matches.append(row)
        out.append({
            "name": ev.name,
            "transform": ev.transform,
            "scorer": ev.scorer,
            "target": ev.target,
            "ever_run": bool(matches),
            "n_runs": len(matches),
            "last_run": matches[0].get("created_at") if matches else None,
            "symbols_seen": sorted({m.get("symbol") for m in matches if m.get("symbol")}),
        })
    return out


def run_standing_eval(name: str, symbol: Optional[str] = None, *, save: bool = True, **kw):
    """Run one standing eval now (transform -> scorer on its target). Returns the result."""
    ev = get_standing_eval(name)
    from processes.runner import run_process

    sym = symbol or ev.symbols[0]
    if ev.transform:
        return run_process(
            ev.transform, symbol=sym, timeframe=ev.timeframe,
            score_with=ev.scorer, score_target=ev.target, save=save, **kw,
        )
    params = dict(kw.pop("params", {}) or {})
    if ev.target:
        params.setdefault("target_col", ev.target)
    return run_process(
        ev.scorer, symbol=sym, timeframe=ev.timeframe, params=params, save=save, **kw,
    )
