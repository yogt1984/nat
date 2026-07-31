"""Touch-maker experiment driver (FINDINGS §4.9) — pre-registered, multi-day.

Runs the 8-cell grid over every available day × {BTC, ETH, SOL}, per-day episodes
(no state across days), and evaluates the pre-registered verdict criteria (P6):

    cells   : V1 base (touch both sides) · V2 +HF1 side-selection ·
              V3 +inventory skew · V4 side+skew+HF4 gate  — each × EV gate off/on
    survive : (a) pooled per-fill EV > 0
              (b) positive-day share >= 0.55
              (c) no single day > 30% of total PnL (concentration guard)
              (d) sign stable at l1_fraction 0.2 / 0.8 (proxy sensitivity)

Criteria are declared HERE, before any result is seen; nothing is re-tuned after.
Artifact: reports/touch_maker_experiment.json. Proxy caveats of §4.7 apply
(flow-window volume split, depth-fraction queue join). Sim-only.

Usage: python -m execution.touch_maker_experiment [--symbols BTC ETH SOL]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from execution.touch_maker import TouchMakerSim, TouchParams  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
REPORT = Path(__file__).resolve().parents[2] / "reports" / "touch_maker_experiment.json"

COLUMNS = ["timestamp_ns", "symbol", "raw_midprice", "raw_spread", "raw_spread_bps",
           "imbalance_qty_l1", "toxic_vpin_50", "raw_bid_depth_5", "raw_ask_depth_5",
           "flow_volume_1s", "flow_aggressor_ratio_5s"]

CELLS = {
    "V1_base": {},
    "V2_side": {"use_hf1_side": True},
    "V3_skew": {"use_inv_skew": True},
    "V4_all": {"use_hf1_side": True, "use_inv_skew": True, "use_hf4_gate": True},
}

MIN_TICKS = 50_000          # skip stub days
POS_SHARE_MIN = 0.55        # (b)
MAX_DAY_SHARE = 0.30        # (c)
SENSITIVITY_L1 = (0.2, 0.8) # (d)


def _episode_inputs(df):
    """Per-day input series (causal throughout)."""
    from algorithms.microprice import Microprice
    n = len(df)
    mid = df["raw_midprice"].to_numpy(dtype=np.float64, na_value=np.nan)
    spread = df["raw_spread"].to_numpy(dtype=np.float64, na_value=np.nan)
    bb, ba = mid - spread / 2.0, mid + spread / 2.0

    mp = Microprice().run_batch(df[["raw_midprice", "imbalance_qty_l1", "raw_spread_bps"]])
    dev = np.nan_to_num(mp["alg_mp_dev_ema"].to_numpy(dtype=np.float64), nan=0.0)

    vol1s = np.nan_to_num(df["flow_volume_1s"].to_numpy(np.float64), nan=0.0)
    aggr = np.nan_to_num(df["flow_aggressor_ratio_5s"].to_numpy(np.float64), nan=0.5)
    sell_exec = vol1s / 10.0 * (1.0 - aggr)
    buy_exec = vol1s / 10.0 * aggr
    db = np.nan_to_num(df["raw_bid_depth_5"].to_numpy(np.float64), nan=0.0)
    da = np.nan_to_num(df["raw_ask_depth_5"].to_numpy(np.float64), nan=0.0)

    # HF4 gate: VPIN below its EXPANDING past-only 70th percentile (block-updated)
    vpin = df["toxic_vpin_50"].to_numpy(dtype=np.float64, na_value=np.nan)
    gate = np.ones(n, dtype=bool)
    thr = np.inf
    for start in range(0, n, 1000):
        block = slice(start, min(start + 1000, n))
        gate[block] = ~np.isfinite(vpin[block]) | (vpin[block] < thr)
        past = vpin[: block.stop]
        past = past[np.isfinite(past)]
        if len(past) >= 500:
            thr = float(np.quantile(past, 0.70))

    return dict(mid=mid, best_bid=bb, best_ask=ba, sell_exec=sell_exec,
                buy_exec=buy_exec, depth_bid=db, depth_ask=da,
                fair_dev_bps=dev, gate_open=gate)


def _run_grid(days, symbols, l1_fraction, cells=CELLS):
    """One full pass; returns {cell: {symbol: [per-day records]}}."""
    from cluster_pipeline.loader import load_parquet
    out = {f"{c}{s}": {sym: [] for sym in symbols}
           for c in cells for s in ("", "_ev")}
    for day in days:
        for sym in symbols:
            try:
                df = load_parquet(str(DATA_DIR), symbols=[sym], start_date=day,
                                  end_date=day, columns=COLUMNS, max_memory_mb=2500)
            except Exception:
                continue
            if len(df) < MIN_TICKS:
                continue
            inputs = _episode_inputs(df)
            for cname, flags in cells.items():
                for suffix, ev in (("", False), ("_ev", True)):
                    p = TouchParams(l1_fraction=l1_fraction, requote_every=10,
                                    use_ev_gate=ev, **flags)
                    r = TouchMakerSim(p).run(**inputs)
                    out[f"{cname}{suffix}"][sym].append(
                        {"day": day, "pnl_bps": r["pnl_bps"], "n_fills": r["n_fills"],
                         "n_postings": r["n_postings"],
                         "max_q": r["max_abs_inventory"]})
            print(f"  {day} {sym}: {len(df):,} ticks done", flush=True)
    return out


def _cell_stats(records):
    """Pool a cell's per-day records across symbols → P6 (a)-(c) verdict inputs."""
    days = [r for sym_recs in records.values() for r in sym_recs]
    if not days:
        return None
    pnl = np.array([r["pnl_bps"] for r in days])
    fills = np.array([r["n_fills"] for r in days])
    total_pnl, total_fills = float(pnl.sum()), int(fills.sum())
    per_fill = total_pnl / total_fills if total_fills else 0.0
    pos_share = float((pnl > 0).mean())
    max_day_share = float(pnl.max() / total_pnl) if total_pnl > 0 else 1.0
    return {
        "n_days": len(days), "total_pnl_bps": round(total_pnl, 1),
        "total_fills": total_fills, "per_fill_bps": round(per_fill, 4),
        "pos_day_share": round(pos_share, 3),
        "max_day_share": round(max_day_share, 3), "worst_day_bps": round(float(pnl.min()), 1),
        "pass_a": per_fill > 0, "pass_b": pos_share >= POS_SHARE_MIN,
        "pass_c": max_day_share <= MAX_DAY_SHARE,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    args = ap.parse_args(argv)

    days = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    print(f"days available: {len(days)} ({days[0]}..{days[-1]})", flush=True)

    grid = _run_grid(days, args.symbols, l1_fraction=0.4)
    stats = {cell: _cell_stats(recs) for cell, recs in grid.items()}

    # (d) sensitivity: only cells passing (a)-(c) are re-run at the l1 extremes
    survivors = [c for c, s in stats.items()
                 if s and s["pass_a"] and s["pass_b"] and s["pass_c"]]
    sensitivity = {}
    for l1 in (SENSITIVITY_L1 if survivors else ()):
        sub = {k.replace("_ev", ""): v for k, v in
               ((c, CELLS[c.replace("_ev", "")]) for c in survivors)}
        g = _run_grid(days, args.symbols, l1_fraction=l1, cells=sub)
        for cell in survivors:
            key = cell.replace("_ev", "") + ("_ev" if cell.endswith("_ev") else "")
            s = _cell_stats(g.get(key, {}))
            sensitivity.setdefault(cell, {})[str(l1)] = s and s["per_fill_bps"]

    verdicts = {}
    for cell, s in stats.items():
        if not s:
            verdicts[cell] = "NO DATA"
            continue
        ok = s["pass_a"] and s["pass_b"] and s["pass_c"]
        if ok and cell in sensitivity:
            ok = all(v is not None and v > 0 for v in sensitivity[cell].values())
            verdicts[cell] = "SURVIVES" if ok else "FAIL(d: proxy-sensitive)"
        else:
            verdicts[cell] = "SURVIVES(a-c; d pending)" if ok else \
                "FAIL(" + ",".join(k for k in ("a", "b", "c") if not s[f"pass_{k}"]) + ")"

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {"criteria": {"per_fill>0": True, "pos_share>=": POS_SHARE_MIN,
                      "max_day_share<=": MAX_DAY_SHARE, "sensitivity_l1": SENSITIVITY_L1},
         "stats": stats, "sensitivity": sensitivity, "verdicts": verdicts,
         "grid": grid}, indent=1))
    print(f"\nartifact: {REPORT}")

    print(f"\n{'cell':<12} {'days':>4} {'fills':>7} {'pnl':>10} {'per-fill':>9} "
          f"{'pos%':>6} {'maxday%':>8}  verdict")
    for cell, s in stats.items():
        if not s:
            continue
        print(f"{cell:<12} {s['n_days']:>4} {s['total_fills']:>7} "
              f"{s['total_pnl_bps']:>10.1f} {s['per_fill_bps']:>9.4f} "
              f"{s['pos_day_share']:>6.2f} {s['max_day_share']:>8.2f}  {verdicts[cell]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
