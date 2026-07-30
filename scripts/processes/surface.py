"""PROC-8: the predictability surface — NAT's central research artifact.

Findings were scattered across process runs; this module aggregates them into ONE
first-class, queryable object with axes (combo × horizon × label-definition × regime)
and value = null-calibrated (PROC-12), FDR-corrected (PROC-13) MI:

    surface_cell{combo_id, horizon, label_def, regime_bin, mi_bits, z_null, bh_q, ...}

Sources: `conditional_predictability` (PROC-6), `horizon_label_scan` (PROC-7), and
label-mode `mi_ksg` runs (PROC-5). It is the output aggregator of the process layer,
the input to the PROC-1 compiler, and the data source for `nat viz predictability`
(the api crate's /api/research/* already serves data/research/).

Maturity per cell: FDR-passed → [PRELIM], else [SPEC]. Never [PROVEN] from statistics
alone — that tag requires downstream validation (walk-forward/paper), i.e. the
lifecycle ladder, not a p-value. Spec: docs/specs/process_layer.md §8.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from . import persistence

# The processes whose findings enter the surface.
SOURCE_PROCESSES = ("conditional_predictability", "horizon_label_scan", "mi_ksg")

SURFACE_COLUMNS = [
    "combo_id", "horizon", "label_def", "regime_var", "regime_bin",
    "mi_bits", "raw_bits", "z_null", "p_value", "bh_q",
    "informative", "maturity", "n",
    "symbol", "timeframe", "process", "run_id", "git_sha", "generated_at",
]

_FLOAT_COLS = ["mi_bits", "raw_bits", "z_null", "p_value", "bh_q", "n"]

DEFAULT_SURFACE_PATH = persistence.DEFAULT_OUT_DIR / "surface.parquet"


def _empty_surface() -> pd.DataFrame:
    return pd.DataFrame(columns=SURFACE_COLUMNS)


def surface_rows_from_record(record: dict) -> list[dict]:
    """Map one persisted ProcessResult record's findings to surface rows.

    mi_ksg contributes ONLY its label-mode findings (horizon == "label") — its
    forward-return findings belong to a different family (fee-gated, no null z)
    and would poison the surface's honesty semantics.
    """
    process = record.get("process")
    if process not in SOURCE_PROCESSES:
        return []
    prov = record.get("provenance") or {}
    rows: list[dict] = []
    for f in record.get("findings", []) or []:
        extras = f.get("extras") or {}
        if process == "mi_ksg":
            if f.get("horizon") != "label":
                continue
            label_def = extras.get("target")
            regime_var, regime_bin = "", "all"
            n = extras.get("n_samples")
        else:
            if f.get("metric") != "cond_mi_bits":
                continue
            if process == "horizon_label_scan":
                label_def = f"tb({extras.get('pt_mult')},{extras.get('sl_mult')})"
            else:
                label_def = "fwd_ret"
            bucket = extras.get("bucket")
            regime_var = extras.get("conditioning") or ""
            regime_bin = "all" if bucket is None else f"b{bucket}"
            n = extras.get("n")
        informative = bool(f.get("informative"))
        rows.append({
            "combo_id": f.get("feature"),
            "horizon": f.get("horizon"),
            "label_def": label_def,
            "regime_var": regime_var,
            "regime_bin": regime_bin,
            "mi_bits": f.get("value"),
            "raw_bits": extras.get("raw_bits", extras.get("bits_above_null")),
            "z_null": extras.get("z"),
            "p_value": f.get("p_value"),
            "bh_q": f.get("p_adjusted"),
            "informative": informative,
            "maturity": "PRELIM" if informative else "SPEC",
            "n": n,
            "symbol": record.get("symbol"),
            "timeframe": record.get("timeframe"),
            "process": process,
            "run_id": record.get("run_id"),
            "git_sha": prov.get("git_sha"),
            "generated_at": prov.get("generated_at"),
        })
    return rows


def build_surface(records: Iterable[dict]) -> pd.DataFrame:
    """Aggregate records into the surface frame. Deterministic: input order irrelevant."""
    rows: list[dict] = []
    for rec in records:
        rows.extend(surface_rows_from_record(rec))
    if not rows:
        return _empty_surface()
    df = pd.DataFrame(rows, columns=SURFACE_COLUMNS)
    for c in _FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
    df["informative"] = df["informative"].astype(bool)
    df = df.sort_values(
        ["symbol", "process", "combo_id", "label_def", "horizon",
         "regime_var", "regime_bin", "run_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def save_surface(df: pd.DataFrame, path: Optional[Union[str, Path]] = None) -> Path:
    path = Path(path) if path is not None else DEFAULT_SURFACE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_surface(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    path = Path(path) if path is not None else DEFAULT_SURFACE_PATH
    if not path.exists():
        return _empty_surface()
    return pd.read_parquet(path)


def aggregate_from_index(
    db_path: Optional[Union[str, Path]] = None,
    out_path: Optional[Union[str, Path]] = None,
    limit: int = 1000,
) -> tuple[pd.DataFrame, Path]:
    """Build + persist the surface from persisted process runs.

    Newest run per (process, symbol, timeframe) wins — the surface reflects the
    latest measurement of each cell, never a stale duplicate; older runs remain
    queryable via their JSON records.
    """
    db = Path(db_path) if db_path is not None else persistence.DEFAULT_DB_PATH
    chosen: dict[tuple, dict] = {}
    for proc_name in SOURCE_PROCESSES:
        for row in persistence.list_results(process=proc_name, limit=limit, db_path=db):
            key = (row.get("process"), row.get("symbol"), row.get("timeframe"))
            if key not in chosen:            # rows arrive newest-first
                chosen[key] = row
    records = []
    for row in chosen.values():
        jp = row.get("json_path")
        if not jp or not Path(jp).exists():
            continue
        try:
            records.append(json.loads(Path(jp).read_text()))
        except Exception:
            continue
    df = build_surface(records)
    path = save_surface(df, path=out_path)
    return df, path


def render_surface(df: pd.DataFrame, top: int = 15) -> str:
    """Terminal render: argmax (always WITH its BH q), ranked cells, regime × horizon
    grid for the argmax combo. '*' marks FDR-passed (informative) cells."""
    if df.empty:
        return "  predictability surface: empty — no process runs aggregated yet\n" \
               "  (run horizon_label_scan / conditional_predictability, then rebuild)"
    lines: list[str] = []
    n_info = int(df["informative"].sum())
    lines.append(f"  predictability surface — {len(df)} cells, {n_info} FDR-passed (*)")

    best = df.loc[df["mi_bits"].idxmax()]
    q = best["bh_q"]
    q_s = f"q={q:g}" if np.isfinite(q) else "q=n/a"
    lines.append(
        f"  argmax: {best['combo_id']}  {best['label_def']}  {best['horizon']}  "
        f"{best['regime_bin']}  bits={best['mi_bits']:.4f}  z={best['z_null']:.1f}  "
        f"{q_s}  [{best['maturity']}]"
    )

    lines.append("")
    lines.append(f"  {'combo':<34} {'label':<14} {'hzn':>6} {'regime':>7} "
                 f"{'bits':>8} {'z':>6} {'q':>7} {'mat':<7}")
    ranked = df.sort_values("mi_bits", ascending=False).head(top)
    for _, r in ranked.iterrows():
        mark = "*" if r["informative"] else " "
        qv = f"{r['bh_q']:.3f}" if np.isfinite(r["bh_q"]) else "-"
        lines.append(
            f"  {str(r['combo_id']):<34} {str(r['label_def']):<14} {str(r['horizon']):>6} "
            f"{str(r['regime_bin']):>7} {r['mi_bits']:>8.4f} {r['z_null']:>6.1f} {qv:>7} "
            f"[{r['maturity']}]{mark}"
        )

    # regime × horizon mini-grid for the argmax combo (the spec's heatmap, terminal-first)
    combo = df[(df["combo_id"] == best["combo_id"]) & (df["label_def"] == best["label_def"])]
    horizons = sorted(combo["horizon"].unique())
    regimes = sorted(combo["regime_bin"].unique())
    if len(horizons) > 1 or len(regimes) > 1:
        lines.append("")
        lines.append(f"  {best['combo_id']} × {best['label_def']} (bits, * = FDR pass)")
        lines.append("  " + f"{'regime':>8} " + " ".join(f"{h:>10}" for h in horizons))
        for rb in regimes:
            cells = []
            for h in horizons:
                sub = combo[(combo["horizon"] == h) & (combo["regime_bin"] == rb)]
                if sub.empty:
                    cells.append(f"{'-':>10}")
                else:
                    r = sub.iloc[0]
                    mark = "*" if r["informative"] else " "
                    cells.append(f"{r['mi_bits']:>9.4f}{mark}")
            lines.append("  " + f"{rb:>8} " + " ".join(cells))
    return "\n".join(lines)
