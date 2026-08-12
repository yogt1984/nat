"""LF8 — funding carry as a held position (family 4). **Pre-registered.**

`funding_reversion` was refuted as a *directional* signal (FINDINGS §4.6). This is a
different mechanism with a different counterparty: the crowded side pays funding every
hour (venue-verified, COST-9), and a dollar-neutral book — short the most-positive-
funding coins, long the most-negative — collects the spread while aggregate price
exposure is hedged. The position is in the *funding*, not the price.

Decidable now: 90 d of hourly funding (data/funding, LF8 fetcher) matches the candle
archive; no accrual wait. Depends on COST-9 — funding accrues via `CostModel`
(SSOT interval), because measuring carry with funding mispriced is circular.

═══════════════════════════════════════════════════════════════════════════════
GATE 0 — COST RATIO, DECLARED BEFORE THE BACKTEST RUNS
═══════════════════════════════════════════════════════════════════════════════
The transferable lesson (TASKS §0): expected-move ÷ round-trip-cost, computed BEFORE
building on results. Here: expected gross carry per day (the trailing signal itself is
the forecast — no forward data enters gate 0) divided by expected daily rebalance cost
(measured membership churn × per-pair RT cost from the SSOT).

    ratio >= 3 for at least one declared configuration, else STOP.

A stop is a *verdict* (family 4 dies by arithmetic, the §2 death), recorded in
FINDINGS with the same care as a positive. `--force` exists for diagnosis only and
prints that it is being used; a forced run can never promote.

═══════════════════════════════════════════════════════════════════════════════
ACCEPTANCE CRITERIA — DECLARED BEFORE THE RUN, IMPORTED NOT INVENTED
═══════════════════════════════════════════════════════════════════════════════
Identical set to XS-6 (G4 + §4.9), evaluated on daily net returns:

  (a) net Sharpe > 0.5 after SSOT costs (funding accrued via CostModel)   [G4]
  (b) deflated Sharpe p < 0.05, penalised for trials tested               [G4, B&LdP]
  (c) positive-period share >= 0.55                                       [§4.9 (b)]
  (d) no single day contributes > 30% of total P&L                        [§4.9 (c)]
  (e) OOS/IS Sharpe ratio > 0.7 (walk-forward 60/40, no fitting either side) [G4]
  (f) sign stable under a 2x cost stress                                  [§4.9 (d)]

Trials for (b): k ∈ {5, 10, 20} × signal window ∈ {24 h, 72 h} × 2 cost modes = 12,
declared here so the deflation cannot be chosen after seeing results (XS-6 convention:
stress runs count as trials).

Construction, fixed in advance: equal-weight ±1/(2k), dollar-neutral, daily rebalance
on the hourly grid, trailing-mean funding as the score. NOT beta-optimised — XS-9
showed beta-neutralisation sharpens a real signal, but it was designed post-hoc there;
here the plain construction is the registered one and any refinement is a NEW trial.

SCEPTICAL PRIORS, DECLARED: the crowd is usually crowded for a reason — positive
funding may be fair compensation for drift that the hedge leg then eats (the price leg
of this book is short-the-crowd, so a trending crowd shows up as gross-price loss, not
as a hidden cost); trailing funding may not persist (churn shows up in gate 0's cost
side); and 90 daily observations bound the power — *undecidable* is an admissible
outcome and is not *refuted* (REV-2 rule).

NOT tested here, and not to be claimed: live fills and queue position (taker pricing
throughout), venue funding-formula changes, capacity beyond XS-5's per-pair depth
measurements, and any cross-venue leg (family 3's business, not ours).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backtest.costs import CostModel  # noqa: E402
from exploration.xs_rotation_study import (  # noqa: E402 — reuse, don't re-derive
    _deflated_sharpe_p,
    _round_trip_cost_bps,
    _sharpe,
)
from processes.candles import available_candle_symbols, load_candles  # noqa: E402
from utils.costs import load_costs  # noqa: E402
from xs.capacity import admit, aggregate_l2, load_l2_snapshots  # noqa: E402

FUNDING_DIR = Path(__file__).resolve().parents[2] / "data" / "funding"
REBALANCE_H = 24                 # daily, on the hourly settlement grid
IS_FRAC = 0.6                    # walk-forward split — same convention as XS-6
SPREAD_CEILING = 2.0             # XS-5 operating point, imported unchanged
K_VALUES = (5, 10, 20)
WINDOWS_H = (24, 72)
N_TRIALS = 12                    # 3 k x 2 windows x 2 cost modes — declared, fixed
GATE0_MIN_RATIO = 3.0            # TASKS §0: "Ratio >= 3 or do not start"


# ── pure construction (planted tests exercise these, no I/O) ─────────────────

def carry_scores(funding_wide: pd.DataFrame, t_idx: int, window_h: int) -> pd.Series:
    """Trailing-mean hourly funding per coin at row `t_idx` (inclusive, backward).

    The score uses only settlements already paid at the rebalance instant —
    the look-ahead discipline is structural, not a parameter.
    """
    lo = max(0, t_idx - window_h + 1)
    hist = funding_wide.iloc[lo:t_idx + 1]
    scores = hist.mean(axis=0, skipna=True)
    n_ok = hist.notna().sum(axis=0)
    return scores[n_ok >= max(2, int(window_h * 0.8))]


def carry_weights(scores: pd.Series, k: int) -> pd.Series | None:
    """±1/(2k) dollar-neutral book: long most-negative funding, short most-positive."""
    if len(scores) < 2 * k:
        return None
    ranked = scores.sort_values()                  # ascending: most negative first
    wts = pd.Series(0.0, index=scores.index)
    wts[ranked.index[:k]] = +1.0 / (2 * k)
    wts[ranked.index[-k:]] = -1.0 / (2 * k)
    return wts


def funding_leg_fraction(weights: pd.Series, held_funding: pd.DataFrame,
                         cost_model: CostModel) -> float:
    """Funding P&L (fraction of gross book) for one held period, via the SSOT model.

    Accrual per coin routes through `CostModel.compute_funding_cost` — the COST-9
    dependency: the interval is the venue-verified SSOT value, never a literal here.
    Positive funding debits the long side and credits the short side.
    """
    held_hours = float(len(held_funding))
    pnl = 0.0
    for coin, w in weights.items():
        if w == 0.0 or coin not in held_funding.columns:
            continue
        avg_rate = float(np.nanmean(held_funding[coin].to_numpy(float)))
        if not np.isfinite(avg_rate):
            continue
        accr_pct = cost_model.compute_funding_cost(held_hours, avg_rate * 1e4)
        pnl += -float(w) * accr_pct / 100.0        # w>0 long pays positive funding
    return pnl


def gate0(funding_wide: pd.DataFrame, half_spread_bps: pd.Series, costs: dict,
          k: int, window_h: int) -> dict:
    """Expected carry/day ÷ expected cost/day, from the signal side only.

    The carry forecast is the trailing signal itself (no forward returns enter);
    the cost side is the *measured* daily membership churn of that signal times
    the per-pair SSOT round-trip. Both are knowable before any backtest exists.
    """
    cost_bps = _round_trip_cost_bps(half_spread_bps, costs)
    carry_days, turn_costs = [], []
    prev = None
    for i in range(window_h, len(funding_wide), REBALANCE_H):
        scores = carry_scores(funding_wide, i, window_h)
        wts = carry_weights(scores, k)
        if wts is None:
            continue
        # expected gross carry over the next day if the trailing rate persists
        exp = float((-wts * scores.reindex(wts.index)).sum()) * REBALANCE_H
        carry_days.append(exp)
        prev_w = prev.reindex(wts.index).fillna(0.0) if prev is not None \
            else pd.Series(0.0, index=wts.index)
        turnover = (wts - prev_w).abs()
        turn_costs.append(float((turnover * cost_bps.reindex(turnover.index)
                                 .fillna(cost_bps.median())).sum()) * 1e-4)
        prev = wts
    if not carry_days:
        return {"k": k, "window_h": window_h, "n_days": 0, "ratio": 0.0}
    carry_bps_day = float(np.mean(carry_days)) * 1e4
    cost_bps_day = float(np.mean(turn_costs[1:]) if len(turn_costs) > 1
                         else np.mean(turn_costs)) * 1e4  # steady-state churn, not day-1 entry
    # 0/0 must FAIL, not promote: with no churn the ratio is only meaningful
    # when there is carry to collect (the planted zero-dispersion null caught
    # an earlier version returning inf here).
    if carry_bps_day <= 0:
        ratio = 0.0
    elif cost_bps_day <= 0:
        ratio = float("inf")
    else:
        ratio = carry_bps_day / cost_bps_day
    return {"k": k, "window_h": window_h, "n_days": len(carry_days),
            "carry_bps_day": round(carry_bps_day, 3),
            "cost_bps_day": round(cost_bps_day, 3),
            "ratio": round(ratio, 2)}


def run_config(k: int, window_h: int, stress: float, funding_wide: pd.DataFrame,
               price_wide: pd.DataFrame, half_spread_bps: pd.Series, costs: dict,
               cost_model: CostModel) -> dict:
    """One declared configuration: daily-rebalanced carry book, three P&L legs."""
    cost_bps = _round_trip_cost_bps(half_spread_bps, costs, stress)
    cols = [c for c in funding_wide.columns if c in price_wide.columns]
    fw, pw = funding_wide[cols], price_wide[cols]

    prev_wts = None
    rows = []
    for i in range(window_h, len(fw) - REBALANCE_H, REBALANCE_H):
        scores = carry_scores(fw, i, window_h)
        wts = carry_weights(scores, k)
        if wts is None:
            continue
        p0 = pw.iloc[i].reindex(wts.index)
        p1 = pw.iloc[i + REBALANCE_H].reindex(wts.index)
        fwd = (p1 / p0 - 1.0).replace([np.inf, -np.inf], np.nan)
        gross = float((wts * fwd.fillna(0.0)).sum())
        funding_pnl = funding_leg_fraction(
            wts, fw.iloc[i + 1:i + 1 + REBALANCE_H], cost_model)
        prev_w = prev_wts.reindex(wts.index).fillna(0.0) if prev_wts is not None \
            else pd.Series(0.0, index=wts.index)
        turnover = (wts - prev_w).abs()
        cost = float((turnover * cost_bps.reindex(turnover.index)
                      .fillna(cost_bps.median())).sum()) * 1e-4
        rows.append({"t": fw.index[i], "gross": gross, "funding": funding_pnl,
                     "cost": cost, "net": gross + funding_pnl - cost,
                     "turnover": float(turnover.sum())})
        prev_wts = wts

    if not rows:
        return {"k": k, "window_h": window_h, "stress": stress, "n": 0}
    df = pd.DataFrame(rows)
    split = int(len(df) * IS_FRAC)
    net = df.net.to_numpy()
    is_r, oos_r = net[:split], net[split:]
    total = float(df.net.sum())
    max_day = float(df.net.abs().max() / abs(total)) if total else float("inf")
    return {
        "k": k, "window_h": window_h, "stress": stress, "n": len(df),
        "net_total_pct": round(total * 100, 3),
        "gross_price_pct": round(float(df.gross.sum()) * 100, 3),
        "funding_pct": round(float(df.funding.sum()) * 100, 3),
        "cost_total_pct": round(float(df.cost.sum()) * 100, 3),
        "sharpe_net": round(_sharpe(net), 3),
        "sharpe_is": round(_sharpe(is_r), 3),
        "sharpe_oos": round(_sharpe(oos_r), 3),
        "oos_is_ratio": round(_sharpe(oos_r) / _sharpe(is_r), 3) if _sharpe(is_r) else None,
        "dsr_p": round(_deflated_sharpe_p(_sharpe(net) / np.sqrt(365), len(df),
                                          N_TRIALS), 4),
        "positive_share": round(float((df.net > 0).mean()), 3),
        "max_day_share": round(max_day, 3),
        "mean_turnover": round(float(df.turnover.mean()), 3),
    }


def evaluate_criteria(results: list[dict]) -> list[str]:
    """The six pre-registered checks; returns surviving config tags."""
    survivors = []
    for r in results:
        if not r.get("n") or r["stress"] != 1.0:
            continue
        stressed = next((s for s in results if s.get("n") and s["k"] == r["k"]
                         and s["window_h"] == r["window_h"] and s["stress"] == 2.0),
                        None)
        checks = {
            "a": r["sharpe_net"] > 0.5,
            "b": r["dsr_p"] < 0.05,
            "c": r["positive_share"] >= 0.55,
            "d": r["max_day_share"] <= 0.30,
            "e": (r["oos_is_ratio"] or 0) > 0.7,
            "f": bool(stressed and np.sign(stressed["net_total_pct"])
                      == np.sign(r["net_total_pct"])),
        }
        failed = [c for c, ok in checks.items() if not ok]
        tag = f"k={r['k']} w={r['window_h']}h"
        r["verdict"] = "SURVIVES" if not failed else f"FAILS({','.join(failed)})"
        if not failed:
            survivors.append(tag)
    return survivors


# ── I/O + orchestration ──────────────────────────────────────────────────────

def load_funding_panel(data_dir: Path | str = FUNDING_DIR) -> pd.DataFrame:
    """Hourly time × symbol funding-rate panel from the LF8 fetcher's parquet."""
    data_dir = Path(data_dir)
    frames = {}
    for f in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["time", "funding_rate"])
        if len(df):
            # settlements land ~on the hour with ms jitter; snap to the hour grid
            hours = (df["time"] // 3_600_000) * 3_600_000
            frames[f.stem] = pd.Series(df["funding_rate"].to_numpy(),
                                       index=hours).groupby(level=0).mean()
    if not frames:
        raise SystemExit(f"no funding parquet in {data_dir} — run "
                         "scripts/data/fetch_funding.py first")
    panel = pd.DataFrame(frames).sort_index()
    panel.index = pd.to_datetime(panel.index, unit="ms", utc=True)
    return panel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="run the study even if gate 0 fails (diagnosis only; "
                         "a forced run can never promote)")
    args = ap.parse_args(argv)

    costs = load_costs()
    agg = aggregate_l2(load_l2_snapshots(), min_snapshots=10)
    admitted, _ = admit(agg, max_half_spread_bps=SPREAD_CEILING)
    funding = load_funding_panel()
    cols = [c for c in funding.columns if c in admitted]
    funding = funding[cols]
    hs = agg.loc[[c for c in cols if c in agg.index], "half_spread_bps"]

    from utils.costs import tier_summary
    print(f"universe: {len(cols)} pairs (admitted at <= {SPREAD_CEILING} bps "
          f"half-spread with funding history)")
    print(f"funding span: {funding.index[0]} -> {funding.index[-1]} "
          f"({len(funding)} hourly rows)")
    print(f"costs (SSOT): {tier_summary()}\n")

    # ── GATE 0 ──
    print(f"GATE 0 — carry/cost ratio (declared threshold >= {GATE0_MIN_RATIO}):")
    gates = []
    for k in K_VALUES:
        for w in WINDOWS_H:
            g = gate0(funding, hs, costs, k, w)
            gates.append(g)
            print(f"  k={g['k']:<3} w={g['window_h']:<3}h  "
                  f"carry {g.get('carry_bps_day', 0):>7.2f} bps/d  "
                  f"cost {g.get('cost_bps_day', 0):>7.2f} bps/d  "
                  f"ratio {g['ratio']:>6.2f}")
    best = max((g["ratio"] for g in gates), default=0.0)
    gate_pass = best >= GATE0_MIN_RATIO

    out = Path("reports/lf8_funding_carry_study.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "criteria": {"a_sharpe_net": "> 0.5", "b_dsr_p": "< 0.05",
                     "c_positive_share": ">= 0.55", "d_max_day_share": "<= 0.30",
                     "e_oos_is_ratio": "> 0.7", "f_sign_stable_at_2x_cost": True,
                     "n_trials_declared": N_TRIALS,
                     "gate0_min_ratio": GATE0_MIN_RATIO},
        "tier_summary": tier_summary(), "n_universe": len(cols),
        "gate0": gates, "gate0_best_ratio": best, "gate0_pass": gate_pass,
    }

    if not gate_pass and not args.force:
        print(f"\nGATE 0 FAILS — best ratio {best:.2f} < {GATE0_MIN_RATIO}. "
              "The study does not run; family 4's carry branch dies by arithmetic. "
              "Record in FINDINGS; use --force only to diagnose.")
        payload["verdict"] = "GATE0_FAIL"
        out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"-> {out}")
        return 1
    if not gate_pass:
        print("\n*** --force: gate 0 failed; this run is diagnostic and cannot promote ***")

    prices = load_candles(available_candle_symbols(interval="1h"), "1h")
    price_wide = prices.pivot_table(index="timestamp", columns="symbol",
                                    values="close", aggfunc="last").sort_index()
    price_wide = price_wide.reindex(funding.index).ffill(limit=2)
    cost_model = CostModel(fee_bps=0.0, slippage_bps=0.0)  # funding leg only; trading
    # costs are charged per-pair above (half-spread + SSOT taker + slippage)

    print("\nSTUDY — 12 declared trials:")
    results = []
    for k in K_VALUES:
        for w in WINDOWS_H:
            for stress in (1.0, 2.0):
                r = run_config(k, w, stress, funding, price_wide, hs, costs,
                               cost_model)
                results.append(r)
                if r.get("n"):
                    print(f"  k={k:<3} w={w:<3}h stress={stress:<4} "
                          f"net {r['net_total_pct']:>8.2f}%  "
                          f"(price {r['gross_price_pct']:>7.2f} "
                          f"+ funding {r['funding_pct']:>6.2f} "
                          f"- cost {r['cost_total_pct']:>6.2f})  "
                          f"SR {r['sharpe_net']:>6.2f}  pos {r['positive_share']:.2f}")

    survivors = evaluate_criteria(results)
    print("\nVERDICT (criteria declared before the run):")
    for r in results:
        if r.get("verdict"):
            print(f"  k={r['k']} w={r['window_h']}h  {r['verdict']}")
    print(f"\n  survivors: {survivors or 'NONE'}")
    if survivors and not gate_pass:
        print("  (forced run — survivors CANNOT promote; rerun honestly)")

    payload["results"] = results
    payload["survivors"] = survivors if gate_pass else []
    payload["verdict"] = ("SURVIVES" if (survivors and gate_pass) else
                          "GATE0_FAIL_FORCED" if not gate_pass else "ALL_FAIL")
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
