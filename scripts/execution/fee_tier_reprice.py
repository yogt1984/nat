"""X-1: reprice the §4.7 / §4.9 maker grids at the HYPE-staking fee tiers.

Question (spec `docs/specs/maker_system.md` §9.1): **does any maker cell flip at the
discounted tier?** The pre-registered §4.9 criteria are IMPORTED UNCHANGED — this study
changes prices, never the bar.

Two parts:

A. §4.7 queue-value EV/posting. The replay itself (fills, adverse selection) is
   cost-INDEPENDENT, so it runs once and the EV arithmetic is priced at every tier:
       EV/posting = P(fill) · (half_spread + maker_rebate − E[adverse | fill])

B. §4.9 touch-maker 8-cell grid. Exact repricing, not re-simulation, for the
   rebate-intact ladder: the staking discount touches the taker fee, and in
   `TouchMakerSim` the taker fee enters at exactly one place — the terminal
   liquidation of leftover inventory. The fill path is therefore bit-identical
   across those tiers and
       pnl_bps(tier) = pnl_bps(base) + liq_cost_bps(base) · (1 − taker(tier)/taker(base))
   is exact. The one configuration that DOES change the fill path is the pessimistic
   sensitivity (`rebate_discount_applies = true`): the rebate enters both the per-fill
   cash and the A4 EV gate, so that variant is genuinely re-simulated.

Costs only via `load_costs()`; every result stamped with `tier_summary()`.
Sim-only — no live path. Artifact: `reports/fee_tier_reprice.json`.

Usage:
    python -m execution.fee_tier_reprice                 # full study
    python -m execution.fee_tier_reprice --days 5        # smoke
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

from execution.touch_maker import TouchMakerSim, TouchParams          # noqa: E402
from execution.touch_maker_experiment import (                        # noqa: E402
    CELLS, COLUMNS, DATA_DIR, MIN_TICKS, MAX_DAY_SHARE, POS_SHARE_MIN,
    _cell_stats, _episode_inputs,
)
from utils.costs import (                                             # noqa: E402
    fee_tier_override, maker_bps, staking_discount, taker_bps, tier_summary,
)

REPORT = Path(__file__).resolve().parents[2] / "reports" / "fee_tier_reprice.json"
LADDER = ["none", "wood", "bronze", "silver", "gold", "platinum", "diamond"]

# §4.7 replay window and parameters — identical to the 2026-07-30 run.
QV_SYMBOL = "BTC"
QV_DAYS = ("2026-07-29", "2026-07-30")
QV_POST_EVERY, QV_HORIZON, QV_L1_FRACTION, QV_MARKOUT = 200, 300, 0.4, 50


# ── Part A: §4.7 queue-value EV ──────────────────────────────────────────────────
def queue_value_reprice(verbose: bool = True) -> dict:
    """Re-run the A4 replay once (cost-free), then price EV/posting at every tier."""
    from cluster_pipeline.loader import load_parquet
    from execution.queue_value import ADAPTER_COLUMNS, replay_from_frame

    df = load_parquet(str(DATA_DIR), symbols=[QV_SYMBOL], start_date=QV_DAYS[0],
                      end_date=QV_DAYS[1], columns=ADAPTER_COLUMNS + ["timestamp_ns", "symbol"],
                      max_memory_mb=4000)
    mid = df["raw_midprice"].to_numpy(dtype=np.float64, na_value=np.nan)
    spread = df["raw_spread"].to_numpy(dtype=np.float64, na_value=np.nan)

    sides = {}
    for side in ("bid", "ask"):
        recs = replay_from_frame(df, side=side, post_every=QV_POST_EVERY,
                                 horizon=QV_HORIZON, l1_fraction=QV_L1_FRACTION,
                                 markout_ticks=QV_MARKOUT)
        ticks = np.array([r["tick"] for r in recs])
        filled = np.array([r["filled"] for r in recs])
        adverse = np.array([r["adverse_bps"] for r in recs], dtype=np.float64)
        adverse = adverse[np.isfinite(adverse)]
        # half-spread at the posting ticks (bps) — the capture side of the EV rule
        hs = (spread[ticks] / 2.0) / mid[ticks] * 1e4
        hs = float(np.nanmean(hs))
        sides[side] = {
            "n_postings": len(recs),
            "fill_rate": round(float(filled.mean()), 4),
            "half_spread_bps": round(hs, 4),
            "adverse_bps_given_fill": round(float(adverse.mean()), 4),
        }
        if verbose:
            print(f"  {side}: postings={len(recs):,} fill={filled.mean():.3f} "
                  f"half_spread={hs:.4f} adverse={adverse.mean():.4f}", flush=True)

    priced = {}
    for tier in LADDER:
        for reb in (False, True):
            with fee_tier_override(tier, rebate_discount=reb):
                rebate = maker_bps()
                key = f"{tier}" + ("+rebate_discounted" if reb else "")
                priced[key] = {
                    "taker_bps": round(taker_bps(), 4),
                    "maker_rebate_bps": round(rebate, 4),
                    **{side: round(s["fill_rate"] * (s["half_spread_bps"] + rebate
                                                     - s["adverse_bps_given_fill"]), 5)
                       for side, s in sides.items()},
                }
    return {"window": {"symbol": QV_SYMBOL, "days": list(QV_DAYS), "n_ticks": len(df),
                       "post_every": QV_POST_EVERY, "horizon": QV_HORIZON,
                       "l1_fraction": QV_L1_FRACTION, "markout_ticks": QV_MARKOUT},
            "measured": sides, "ev_per_posting_bps": priced}


# ── Part B: §4.9 touch-maker grid ────────────────────────────────────────────────
def _episode_cells(task) -> tuple:
    """One (day, symbol) episode × all 8 cells. Worker body — must stay picklable.

    The tier is applied INSIDE the worker (`fee_tier_override`), so a parallel pass
    prices exactly the tier it was handed, never the parent's ambient environment.
    """
    day, sym, l1_fraction, tier, rebate_discount = task
    from cluster_pipeline.loader import load_parquet
    try:
        df = load_parquet(str(DATA_DIR), symbols=[sym], start_date=day, end_date=day,
                          columns=COLUMNS, max_memory_mb=2500)
    except Exception:
        return day, sym, 0, {}
    if len(df) < MIN_TICKS:
        return day, sym, len(df), {}
    inputs = _episode_inputs(df)
    recs = {}
    with fee_tier_override(tier, rebate_discount=rebate_discount):
        for cname, flags in CELLS.items():
            for suffix, ev in (("", False), ("_ev", True)):
                p = TouchParams(l1_fraction=l1_fraction, requote_every=10,
                                use_ev_gate=ev, **flags)
                r = TouchMakerSim(p).run(**inputs)
                recs[f"{cname}{suffix}"] = {
                    "day": day, "pnl_bps": r["pnl_bps"], "n_fills": r["n_fills"],
                    "n_postings": r["n_postings"], "max_q": r["max_abs_inventory"],
                    "liq_cost_bps": r["liquidation_cost_bps"],
                    "terminal_q": r["terminal_inventory"],
                    "taker_bps_used": r["taker_bps_used"], "maker_bps_used": r["maker_bps_used"],
                }
    return day, sym, len(df), recs


def run_grid(days, symbols, l1_fraction=0.4, verbose=True, tier="none",
             rebate_discount=False, workers=6) -> dict:
    """One full 8-cell pass, recording the liquidation component needed for repricing.

    Episodes are independent by construction (no state across days — §4.9 protocol), so
    they fan out across processes; results are re-sorted into deterministic day order.
    """
    from concurrent.futures import ProcessPoolExecutor
    out = {f"{c}{s}": {sym: [] for sym in symbols} for c in CELLS for s in ("", "_ev")}
    tasks = [(day, sym, l1_fraction, tier, rebate_discount)
             for day in days for sym in symbols]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for day, sym, n_ticks, recs in pool.map(_episode_cells, tasks, chunksize=1):
            done += 1
            if verbose:
                status = f"{n_ticks:,} ticks" if recs else f"skipped ({n_ticks:,} ticks)"
                print(f"  [{done}/{len(tasks)}] {day} {sym}: {status}", flush=True)
            for cell, rec in recs.items():
                out[cell][sym].append(rec)
    for cell in out:
        for sym in out[cell]:
            out[cell][sym].sort(key=lambda r: r["day"])
    return out


def reprice_grid(grid: dict, base_taker: float, tier: str) -> dict:
    """Exact taker-fee repricing of a base-tier grid (fill path is tier-invariant)."""
    with fee_tier_override(tier):
        scale = taker_bps() / base_taker
    out = {}
    for cell, per_sym in grid.items():
        out[cell] = {sym: [{**r, "pnl_bps": r["pnl_bps"] + r["liq_cost_bps"] * (1.0 - scale),
                            "liq_cost_bps": r["liq_cost_bps"] * scale}
                           for r in recs]
                     for sym, recs in per_sym.items()}
    return out


def verdict(stats: dict) -> str:
    if not stats:
        return "NO DATA"
    fails = [k for k in ("a", "b", "c") if not stats[f"pass_{k}"]]
    return "SURVIVES(a-c; d pending)" if not fails else "FAIL(" + ",".join(fails) + ")"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--days", type=int, default=0, help="limit to the last N days (smoke)")
    ap.add_argument("--skip-queue-value", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="parallel episode workers")
    args = ap.parse_args(argv)

    days = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    if args.days:
        days = days[-args.days:]
    print(f"days: {len(days)} ({days[0]}..{days[-1]})  symbols: {args.symbols}", flush=True)

    report = {"study": "X-1 fee-tier repricing",
              "criteria_imported_from": "FINDINGS §4.9 (pre-registered; unchanged)",
              "criteria": {"per_fill>0": True, "pos_share>=": POS_SHARE_MIN,
                           "max_day_share<=": MAX_DAY_SHARE},
              "base_tier_stamp": tier_summary(),
              "ladder": {t: staking_discount(t) for t in LADDER}}

    if not args.skip_queue_value:
        print("\n[A] §4.7 queue-value replay (cost-free) → EV priced per tier", flush=True)
        report["queue_value"] = queue_value_reprice()

    print("\n[B] §4.9 touch-maker grid — base tier pass", flush=True)
    base_taker = taker_bps()
    base_grid = run_grid(days, args.symbols, workers=args.workers)

    tiers = {}
    for tier in LADDER:
        g = base_grid if tier == "none" else reprice_grid(base_grid, base_taker, tier)
        with fee_tier_override(tier):
            stamp = tier_summary()
        stats = {cell: _cell_stats(recs) for cell, recs in g.items()}
        tiers[tier] = {"stamp": stamp, "stats": stats,
                       "verdicts": {c: verdict(s) for c, s in stats.items()}}

    print("\n[B2] pessimistic sensitivity — diamond WITH the rebate discounted "
          "(re-simulated: the rebate moves the fill path)", flush=True)
    with fee_tier_override("diamond", rebate_discount=True):
        stamp = tier_summary()
    sens_grid = run_grid(days, args.symbols, tier="diamond", rebate_discount=True,
                         workers=args.workers)
    stats = {cell: _cell_stats(recs) for cell, recs in sens_grid.items()}
    tiers["diamond+rebate_discounted"] = {
        "stamp": stamp, "stats": stats, "verdicts": {c: verdict(s) for c, s in stats.items()}}

    report["touch_maker"] = tiers
    report["any_cell_flips"] = sorted(
        {f"{t}:{c}" for t, d in tiers.items()
         for c, v in d["verdicts"].items() if v.startswith("SURVIVES")})

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nartifact: {REPORT}")

    # ── console summary ──────────────────────────────────────────────────────────
    if "queue_value" in report:
        print("\n§4.7 EV per posting (bps)")
        print(f"{'tier':<28} {'taker':>6} {'rebate':>7} {'BID':>9} {'ASK':>9}")
        for k, v in report["queue_value"]["ev_per_posting_bps"].items():
            print(f"{k:<28} {v['taker_bps']:>6.2f} {v['maker_rebate_bps']:>7.3f} "
                  f"{v['bid']:>9.4f} {v['ask']:>9.4f}")

    print("\n§4.9 cells — per-fill bps by tier")
    cells = sorted(next(iter(tiers.values()))["stats"])
    print(f"{'cell':<12} " + " ".join(f"{t[:9]:>10}" for t in tiers))
    for cell in cells:
        row = []
        for t in tiers:
            s = tiers[t]["stats"].get(cell)
            row.append(f"{s['per_fill_bps']:>10.4f}" if s else f"{'-':>10}")
        print(f"{cell:<12} " + " ".join(row))
    print("\nverdict flips (SURVIVES at any tier): "
          f"{report['any_cell_flips'] or 'NONE — every cell still FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
