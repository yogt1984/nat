"""Does the "eight independent axes" claim survive a holdout? (FINDINGS §5 follow-up)

§1 classifies the feature set into eight independent signal axes, and
`specs/maker_system.md` §2 turns that into a contract — one representative per axis.
Both rest on correlation dedup computed on **full-sample** statistics. PROC-15's first
datum (BTC 2026-08-04) suggested that is a forward-looking claim being made from a
backward-looking measurement: residualizing `imbalance_qty_l5` against
`imbalance_qty_l1` left ~0.19 correlation on the holdout because the OLS beta drifted
~10 % within the day.

This sweep replicates that across every available day × symbol × pair, using the
PROC-15 process unchanged. Per episode it records:

    prefix |corr(res, Z)|   — must be ~0; it is OLS arithmetic and only a sanity check
    holdout |corr(res, Z)|  — the number that can fail
    beta_prefix, beta_holdout, and the relative drift between them

`flow_vwap_deviation` is carried as a **control**: §1 calls it a distinct
(mean-reverting) axis, so if the method is sound it should stay near-zero everywhere
while the imbalance cousins do not. A sweep where everything looks non-orthogonal is a
broken method; a sweep where the control stays clean and the cousins drift is a finding.

Artifact: `reports/orthogonality_sweep.json`. Read-only, no capital path.

Usage:  python -m exploration.orthogonality_sweep [--days N] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
REPORT = Path(__file__).resolve().parents[2] / "reports" / "orthogonality_sweep.json"

#: The conditioning variable: §1's dominant axis.
CONDITION_ON = "imbalance_qty_l1"

#: Targets — the rest of the fast-directional block plus one representative per other axis.
TARGETS = [
    # same block as the conditioner (expected to be redundant, but by how much, and stably?)
    "imbalance_qty_l5", "imbalance_qty_l10", "imbalance_notional_l5",
    "imbalance_orders_l5", "imbalance_depth_weighted",
    "raw_bid_depth_5", "raw_ask_depth_5",
    # other axes per §1 — these are the ones the contract treats as separable
    "cross_obi_mean",                 # (3) cross-symbol imbalance
    "micro_queue_position_bid",       # (4) queue dynamics
    "flow_vwap_deviation",            # (5) VWAP deviation — the CONTROL
    "micro_obi_velocity",             # (6) OBI velocity
    "ent_permutation_imbalance_16",   # (7) imbalance entropy
    "flow_aggressor_ratio_5s",        # (8) aggressor flow
]

CONTROL = "flow_vwap_deviation"
MIN_TICKS = 50_000
FIT_FRAC = 0.7


def _episode(task) -> dict:
    """One (day, symbol): residualize every target against the conditioner."""
    day, sym = task
    from cluster_pipeline.loader import load_parquet
    from processes.base import ProcessContext
    from processes.residualize import ResidualizeProcess

    cols = ["timestamp_ns", "symbol", "raw_midprice", CONDITION_ON] + TARGETS
    try:
        df = load_parquet(str(DATA_DIR), symbols=[sym], start_date=day, end_date=day,
                          columns=cols, max_memory_mb=2000)
    except Exception as exc:
        return {"day": day, "symbol": sym, "error": str(exc)[:120], "pairs": {}}
    if len(df) < MIN_TICKS:
        return {"day": day, "symbol": sym, "skipped": f"{len(df)} ticks", "pairs": {}}

    ctx = ProcessContext(symbol=sym, timeframe="tick", price_col="raw_midprice",
                         horizons={"1m": 600}, costs={})
    present = [t for t in TARGETS if t in df.columns]
    _, res = ResidualizeProcess(features=present, conditioning=[CONDITION_ON],
                                fit_frac=FIT_FRAC).transform(df, ctx)

    z = df[CONDITION_ON].to_numpy(dtype=np.float64, na_value=np.nan)
    cut = int(len(df) * FIT_FRAC)
    pairs = {}
    for f in res.findings:
        src = f.extras["source_feature"]
        col = df[src].to_numpy(dtype=np.float64, na_value=np.nan)
        b_pre = _ols_beta(z[:cut], col[:cut])
        b_hold = _ols_beta(z[cut:], col[cut:])
        drift = (abs(b_hold - b_pre) / abs(b_pre)) if (b_pre and np.isfinite(b_pre)) else None
        pairs[src] = {
            "holdout_abs_corr": f.value,
            "r2_fit": f.extras["r2_fit"],
            "beta_prefix": None if b_pre is None else round(b_pre, 6),
            "beta_holdout": None if b_hold is None else round(b_hold, 6),
            "beta_drift_rel": None if drift is None else round(float(drift), 4),
            "n_holdout": f.extras["n_holdout_rows"],
        }
    return {"day": day, "symbol": sym, "n_ticks": len(df), "pairs": pairs}


def _ols_beta(z: np.ndarray, f: np.ndarray):
    m = np.isfinite(z) & np.isfinite(f)
    if m.sum() < 100 or np.std(z[m]) <= 0:
        return None
    return float(np.polyfit(z[m], f[m], 1)[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    days = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    if args.days:
        days = days[-args.days:]
    tasks = [(d, s) for d in days for s in args.symbols]
    print(f"orthogonality sweep · {len(days)} days × {args.symbols} = {len(tasks)} episodes\n"
          f"conditioning on {CONDITION_ON}, fit_frac={FIT_FRAC}, control={CONTROL}", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    episodes = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, ep in enumerate(pool.map(_episode, tasks, chunksize=1), 1):
            episodes.append(ep)
            if i % 15 == 0:
                print(f"  [{i}/{len(tasks)}] {ep['day']} {ep['symbol']}", flush=True)

    # aggregate per (symbol, pair) and pooled
    agg = {}
    for target in TARGETS:
        rows = [(e["symbol"], e["pairs"][target]) for e in episodes
                if target in e.get("pairs", {})]
        if not rows:
            continue
        hc = np.array([r[1]["holdout_abs_corr"] for r in rows], dtype=float)
        dr = np.array([r[1]["beta_drift_rel"] for r in rows
                       if r[1]["beta_drift_rel"] is not None], dtype=float)
        r2 = np.array([r[1]["r2_fit"] for r in rows], dtype=float)
        entry = {
            "n_episodes": len(rows),
            "median_holdout_abs_corr": round(float(np.median(hc)), 4),
            "p90_holdout_abs_corr": round(float(np.quantile(hc, 0.9)), 4),
            "frac_above_0.10": round(float((hc > 0.10).mean()), 3),
            "median_r2_fit": round(float(np.median(r2)), 4),
            "median_beta_drift_rel": (round(float(np.median(dr)), 4) if dr.size else None),
            "by_symbol": {},
        }
        for sym in args.symbols:
            s = np.array([r[1]["holdout_abs_corr"] for r in rows if r[0] == sym], dtype=float)
            if s.size:
                entry["by_symbol"][sym] = {
                    "n": int(s.size), "median": round(float(np.median(s)), 4),
                    "frac_above_0.10": round(float((s > 0.10).mean()), 3)}
        agg[target] = entry

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {"conditioning": CONDITION_ON, "control": CONTROL, "fit_frac": FIT_FRAC,
         "days": days, "symbols": args.symbols, "aggregate": agg, "episodes": episodes},
        indent=1, default=float))
    print(f"\nartifact: {REPORT}")

    print(f"\n{'target':<32}{'n':>4}{'med|corr|':>11}{'p90':>8}{'>0.10':>8}"
          f"{'R2fit':>8}{'βdrift':>9}")
    for t, e in sorted(agg.items(), key=lambda kv: -kv[1]["median_holdout_abs_corr"]):
        tag = "  <- CONTROL" if t == CONTROL else ""
        print(f"{t:<32}{e['n_episodes']:>4}{e['median_holdout_abs_corr']:>11.4f}"
              f"{e['p90_holdout_abs_corr']:>8.3f}{e['frac_above_0.10']:>8.2f}"
              f"{e['median_r2_fit']:>8.3f}"
              f"{(e['median_beta_drift_rel'] or float('nan')):>9.3f}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
