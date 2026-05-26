#!/usr/bin/env python3
"""Aggregate walk-forward JSONs from `audit_sweep.py` into a single decision table.

Inputs
------
    audit_dir/walk_forward_BTC_1min.json
    audit_dir/walk_forward_BTC_2min.json
    ...
    audit_dir/walk_forward_SOL_5min.json

Outputs
-------
    audit_dir/feature_audit.csv     one row per feature, with per-cell columns,
                                    aggregate stats, and final_decision in
                                    {keep, monitor, drop}.
    audit_dir/feature_audit.md      human-readable summary: top keepers,
                                    per-vector breakdown, dropped count.

Decision rule
-------------
    Let n_cells = number of (symbol, timeframe) cells where the feature was
    evaluated.  Per cell, the walk-forward emits a {keep, monitor, drop}
    decision (see WalkForwardFeature.decision in scalping_profiler.py).

    Aggregate per feature:
        keep_frac      = keep_cells   / n_cells
        active_frac    = (keep + monitor) / n_cells   (= "non-drop fraction")
        sign_consistency_mean = mean of per-cell sign_consistency
        oos_ic_mean    = mean of per-cell oos_ic_mean

    Final decision:
        keep    if  keep_frac >= 5/9   AND  sign_consistency_mean >= 0.6
        monitor if  active_frac >= 5/9 (the rest of the time)
        drop    otherwise

    The 5/9 threshold requires the feature to survive in a majority of
    cells, which catches cross-asset AND cross-timeframe stability in one
    rule.

Usage
-----
    python scripts/audit_aggregate.py reports/profiler/audit_2026-05-11
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cells(audit_dir: Path) -> List[dict]:
    """Read every walk_forward_*.json in the audit dir."""
    cells = []
    for fp in sorted(audit_dir.glob("walk_forward_*.json")):
        with open(fp) as f:
            cells.append(json.load(f))
    if not cells:
        raise FileNotFoundError(
            f"No walk_forward_*.json files found in {audit_dir}. "
            f"Run `python scripts/audit_sweep.py --out-dir {audit_dir}` first."
        )
    return cells


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(cells: List[dict]) -> pd.DataFrame:
    """Build one row per feature from all (symbol, timeframe) cells."""
    n_cells = len(cells)

    # per-feature accumulator
    accum: Dict[str, dict] = defaultdict(lambda: {
        "vector": None,
        "decisions": [],
        "sign_consistency": [],
        "oos_ic_mean": [],
        "oos_ic_std": [],
        "horizon": [],
        "cell_keys": [],
    })

    for c in cells:
        cell_key = f"{c['symbol']}_{c['timeframe']}"
        for f in c["features"]:
            a = accum[f["name"]]
            a["vector"] = f["vector"]
            a["decisions"].append(f["decision"])
            a["sign_consistency"].append(f["sign_consistency"])
            a["oos_ic_mean"].append(f["oos_ic_mean"])
            a["oos_ic_std"].append(f["oos_ic_std"])
            a["horizon"].append(f["horizon"])
            a["cell_keys"].append(cell_key)

    rows = []
    for name, a in accum.items():
        decisions = a["decisions"]
        n = len(decisions)  # cells where this feature was profiled
        n_keep = decisions.count("keep")
        n_monitor = decisions.count("monitor")
        n_drop = decisions.count("drop")

        keep_frac = n_keep / n_cells if n_cells else 0.0
        active_frac = (n_keep + n_monitor) / n_cells if n_cells else 0.0
        sc_mean = sum(a["sign_consistency"]) / n if n else 0.0
        oos_ic_mean = sum(a["oos_ic_mean"]) / n if n else 0.0
        oos_ic_std_max = max(a["oos_ic_std"]) if a["oos_ic_std"] else 0.0

        if keep_frac >= 5 / 9 and sc_mean >= 0.6:
            final = "keep"
        elif active_frac >= 5 / 9:
            final = "monitor"
        else:
            final = "drop"

        rows.append({
            "feature": name,
            "vector": a["vector"],
            "n_cells_seen": n,
            "keep": n_keep,
            "monitor": n_monitor,
            "drop": n_drop,
            "keep_frac": round(keep_frac, 3),
            "active_frac": round(active_frac, 3),
            "sign_consistency_mean": round(sc_mean, 3),
            "oos_ic_mean": round(oos_ic_mean, 4),
            "oos_ic_std_max": round(oos_ic_std_max, 4),
            "final_decision": final,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["final_decision", "keep_frac", "sign_consistency_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_markdown(df: pd.DataFrame, cells: List[dict], out_path: Path) -> None:
    """Emit a human-readable summary alongside the CSV."""
    n_cells = len(cells)
    cell_labels = [f"{c['symbol']}_{c['timeframe']}" for c in cells]

    counts = df["final_decision"].value_counts().reindex(
        ["keep", "monitor", "drop"], fill_value=0
    )

    lines = []
    lines.append("# Feature Audit — Cross-symbol / Cross-timeframe Walk-Forward")
    lines.append("")
    lines.append(f"- **Cells evaluated:** {n_cells} ({', '.join(cell_labels)})")
    lines.append(f"- **Features ranked:** {len(df)}")
    lines.append(f"- **Decision tally:** keep={counts['keep']}, "
                 f"monitor={counts['monitor']}, drop={counts['drop']}")
    lines.append("")
    lines.append("Decision rule: `keep` requires `keep_frac >= 5/9` AND "
                 "`sign_consistency_mean >= 0.6`. `monitor` requires "
                 "`(keep+monitor)_frac >= 5/9`. Otherwise `drop`.")
    lines.append("")

    # Top keepers
    keepers = df[df["final_decision"] == "keep"].head(30)
    if len(keepers) > 0:
        lines.append("## Top keepers")
        lines.append("")
        lines.append("| rank | feature | vector | keep | mon | drop | keep_frac | sign_cons | OOS IC |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
        for i, r in enumerate(keepers.itertuples(index=False), 1):
            lines.append(
                f"| {i} | `{r.feature}` | {r.vector} | "
                f"{r.keep} | {r.monitor} | {r.drop} | "
                f"{r.keep_frac:.2f} | {r.sign_consistency_mean:.2f} | "
                f"{r.oos_ic_mean:+.4f} |"
            )
        lines.append("")
    else:
        lines.append("## Top keepers")
        lines.append("")
        lines.append("_None — no feature passed the keep threshold._")
        lines.append("")

    # Monitor list (shorter)
    monitors = df[df["final_decision"] == "monitor"].head(15)
    if len(monitors) > 0:
        lines.append("## Monitor (consistent direction but weak edge)")
        lines.append("")
        lines.append("| feature | vector | keep | mon | sign_cons | OOS IC |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for r in monitors.itertuples(index=False):
            lines.append(
                f"| `{r.feature}` | {r.vector} | {r.keep} | {r.monitor} | "
                f"{r.sign_consistency_mean:.2f} | {r.oos_ic_mean:+.4f} |"
            )
        lines.append("")

    # Per-vector breakdown
    lines.append("## Per-vector summary")
    lines.append("")
    vec = df.groupby("vector")["final_decision"].value_counts().unstack(fill_value=0)
    for col in ("keep", "monitor", "drop"):
        if col not in vec.columns:
            vec[col] = 0
    vec = vec[["keep", "monitor", "drop"]]
    lines.append("| vector | keep | monitor | drop |")
    lines.append("|---|---:|---:|---:|")
    for vname, row in vec.iterrows():
        lines.append(f"| {vname} | {row['keep']} | {row['monitor']} | {row['drop']} |")
    lines.append("")

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audit_dir", help="directory containing walk_forward_*.json files")
    args = ap.parse_args()

    audit_dir = Path(args.audit_dir)
    if not audit_dir.exists():
        sys.exit(f"audit_dir not found: {audit_dir}")

    cells = load_cells(audit_dir)
    df = aggregate(cells)

    csv_path = audit_dir / "feature_audit.csv"
    md_path = audit_dir / "feature_audit.md"
    df.to_csv(csv_path, index=False)
    write_markdown(df, cells, md_path)

    counts = df["final_decision"].value_counts().reindex(
        ["keep", "monitor", "drop"], fill_value=0
    )
    print(f"Aggregated {len(cells)} cells -> {len(df)} features")
    print(f"  keep    = {counts['keep']:>4}")
    print(f"  monitor = {counts['monitor']:>4}")
    print(f"  drop    = {counts['drop']:>4}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
