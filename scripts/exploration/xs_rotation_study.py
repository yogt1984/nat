"""XS-6 — cross-sectional rotation OOS study. **Pre-registered.**

The question every earlier XS row was building toward: after SSOT costs, does ranking the
universe by `vol` (negative polarity — long low-vol) make money out of sample?

  §7.4  vol rank-IC -0.0690, z -8.37, BH q 0.007, 83 non-overlapping daily rebalances
  §7.5  vol rank half-life >=30 days (rho 0.691 at the first disjoint lag)
  §7.6  ~117 pairs support $1k/pair/day at 1% of ADV with half-spread <= 2 bps

Everything above is signal-level. This file is where cost enters, and the record says what
usually happens then: §2's taker arithmetic, and five "winners" that died at §4.6 precisely
because their economics were never run honestly.

═══════════════════════════════════════════════════════════════════════════════
ACCEPTANCE CRITERIA — DECLARED BEFORE THE RUN, IMPORTED NOT INVENTED
═══════════════════════════════════════════════════════════════════════════════
Imported from G4 (`GLOSSARY.md`) and the §4.9 pre-registration set. A configuration
SURVIVES only if it clears ALL of:

  (a) net Sharpe > 0.5 after SSOT costs                        [G4]
  (b) deflated Sharpe p < 0.05, penalised for trials tested    [G4, Bailey & LdP]
  (c) positive-period share >= 0.55                            [§4.9 criterion (b)]
  (d) no single day contributes > 30% of total P&L             [§4.9 criterion (c)]
  (e) OOS/IS Sharpe ratio > 0.7                                [G4]
  (f) sign stable under a 2x cost stress                       [§4.9 criterion (d)]

Trials counted for (b): 2 constructions (long-only, long-short) x 3 k values x 2 cost
modes = 12. Declared here so the deflation cannot be chosen after seeing results.

A configuration that clears all six promotes to lifecycle DISCOVERED. Anything else is
recorded in FINDINGS as a negative with the same care as a positive — the §4.6 lesson is
that unrecorded negatives return as false positives.

NOT tested here, and not to be claimed: live fills (this crosses the spread at the
prevailing quote, it does not model queue or impact), funding accrual on held inventory
(the gap `DOCS_IMPROVEMENT_PLAN` §D.1 flags), and regime breadth — one 90-day window.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from processes.candles import available_candle_symbols, load_candles  # noqa: E402
from utils.costs import load_costs  # noqa: E402
from xs.capacity import admit, aggregate_l2, load_l2_snapshots  # noqa: E402
from xs.features import realized_vol  # noqa: E402

LOOKBACK = 168          # bars used to score (7 days) — as in §7.4/§7.5
REBALANCE = 24          # daily, matching the measured IC cadence
IS_FRAC = 0.6           # walk-forward: first 60% in-sample, last 40% out
SPREAD_CEILING = 2.0    # XS-5 operating point: 117 pairs at $1k/pair
K_VALUES = (10, 20, 40)


def _round_trip_cost_bps(half_spread_bps: pd.Series, costs: dict,
                         stress: float = 1.0) -> pd.Series:
    """Per-unit-turnover cost of crossing, per pair, from the SSOT.

    A rotation is a TAKER strategy: it crosses the spread to rebalance. Cost per side is
    the pair's own measured half-spread plus the SSOT taker fee and slippage.

    **No default literals.** The SSOT is nested (`costs["hyperliquid"]["taker_bps"]`) and
    the first version of this function used `.get(..., 4.5)` fallbacks — which silently
    supplied hardcoded fees when the key lookup missed, exactly the guardrail violation
    (`all costs via load_costs()`) behind §4.6's wrong-venue pricing. A missing key now
    raises rather than inventing a number.
    """
    from utils.costs import taker_bps as ssot_taker_bps

    hl = costs["hyperliquid"]                       # KeyError if the SSOT changes shape
    taker = float(ssot_taker_bps())                 # accessor applies the staking tier
    slip = float(hl["slippage_bps"])
    return (half_spread_bps + taker + slip) * stress


def _sharpe(r: np.ndarray, periods_per_year: float = 365.0) -> float:
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def _deflated_sharpe_p(sr: float, n: int, n_trials: int, skew=0.0, kurt=3.0) -> float:
    """Bailey & Lopez de Prado deflated Sharpe: p-value of SR given `n_trials` searched."""
    from scipy import stats
    if n < 3:
        return 1.0
    # Expected max SR under the null across n_trials independent trials.
    e = 0.5772156649
    sr0 = np.sqrt(2 * np.log(max(n_trials, 2))) * (1 - e) + e * np.sqrt(
        2 * np.log(max(n_trials, 2) * np.e ** 2))
    sr0 *= 1.0 / np.sqrt(n)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    if denom <= 0:
        return 1.0
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    return float(1 - stats.norm.cdf(z))


def run(k: int, long_short: bool, stress: float, wide, agg, costs, admitted) -> dict:
    """One configuration: daily top-k rotation on `vol` rank (low vol = long)."""
    cols = [c for c in wide.columns if c in admitted]
    w = wide[cols]
    hs = agg.loc[cols, "half_spread_bps"]
    cost_bps = _round_trip_cost_bps(hs, costs, stress)

    idx = w.index
    prev_wts = pd.Series(0.0, index=cols)
    rows = []

    for i in range(LOOKBACK, len(idx) - REBALANCE, REBALANCE):
        hist = w.iloc[i - LOOKBACK:i + 1]
        fwd = w.iloc[i + REBALANCE] / w.iloc[i] - 1.0

        scores = {}
        for c in cols:
            v = hist[c].dropna().to_numpy(float)
            if len(v) >= LOOKBACK * 0.8 and np.isfinite(fwd[c]):
                s = realized_vol(np.diff(np.log(v)))
                if np.isfinite(s):
                    scores[c] = s
        if len(scores) < 2 * k:
            continue
        sc = pd.Series(scores).sort_values()          # ascending: lowest vol first

        wts = pd.Series(0.0, index=cols)
        wts[sc.index[:k]] = 1.0 / k                   # long the low-vol tail
        if long_short:
            wts[sc.index[-k:]] = -1.0 / k             # short the high-vol tail

        turnover = (wts - prev_wts).abs()
        cost = float((turnover * cost_bps.reindex(turnover.index).fillna(
            cost_bps.median())).sum()) * 1e-4
        gross = float((wts * fwd.reindex(wts.index).fillna(0.0)).sum())
        rows.append({"t": idx[i], "gross": gross, "cost": cost, "net": gross - cost,
                     "turnover": float(turnover.sum())})
        prev_wts = wts

    if not rows:
        return {"k": k, "long_short": long_short, "stress": stress, "n": 0}

    df = pd.DataFrame(rows)
    split = int(len(df) * IS_FRAC)
    is_r, oos_r = df.net.to_numpy()[:split], df.net.to_numpy()[split:]
    net = df.net.to_numpy()

    total = float(df.net.sum())
    max_day = float(df.net.abs().max() / abs(total)) if total else float("inf")
    return {
        "k": k, "long_short": long_short, "stress": stress, "n": len(df),
        "net_total_pct": round(total * 100, 3),
        "gross_total_pct": round(float(df.gross.sum()) * 100, 3),
        "cost_total_pct": round(float(df.cost.sum()) * 100, 3),
        "sharpe_net": round(_sharpe(net), 3),
        "sharpe_is": round(_sharpe(is_r), 3),
        "sharpe_oos": round(_sharpe(oos_r), 3),
        "oos_is_ratio": round(_sharpe(oos_r) / _sharpe(is_r), 3) if _sharpe(is_r) else None,
        "dsr_p": round(_deflated_sharpe_p(_sharpe(net) / np.sqrt(365), len(df), 12), 4),
        "positive_share": round(float((df.net > 0).mean()), 3),
        "max_day_share": round(max_day, 3),
        "mean_turnover": round(float(df.turnover.mean()), 3),
    }


def main() -> int:
    costs = load_costs()
    agg = aggregate_l2(load_l2_snapshots(), min_snapshots=10)
    admitted, _ = admit(agg, max_half_spread_bps=SPREAD_CEILING)
    frame = load_candles(available_candle_symbols(interval="1h"), "1h")
    wide = frame.pivot_table(index="timestamp", columns="symbol",
                             values="close", aggfunc="last").sort_index()

    print(f"universe admitted at <= {SPREAD_CEILING} bps: {len(admitted)} pairs")
    from utils.costs import taker_bps as ssot_taker_bps, tier_summary
    print(f"costs (SSOT): taker {ssot_taker_bps()} bps, "
          f"slippage {costs['hyperliquid']['slippage_bps']} bps, tier {tier_summary()}\n")

    results = []
    for k in K_VALUES:
        for ls in (False, True):
            for stress in (1.0, 2.0):
                r = run(k, ls, stress, wide, agg, costs, admitted)
                results.append(r)
                if r.get("n"):
                    tag = "L/S" if ls else "L  "
                    print(f"k={k:<3} {tag} stress={stress:<4} "
                          f"net {r['net_total_pct']:>8.2f}%  "
                          f"(gross {r['gross_total_pct']:>7.2f} - cost {r['cost_total_pct']:>6.2f})  "
                          f"SR {r['sharpe_net']:>6.2f}  OOS {r['sharpe_oos']:>6.2f}  "
                          f"pos {r['positive_share']:.2f}  maxday {r['max_day_share']:.2f}  "
                          f"turn {r['mean_turnover']:.2f}")

    out = Path("reports/xs_rotation_study.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"criteria": {
        "a_sharpe_net": "> 0.5", "b_dsr_p": "< 0.05", "c_positive_share": ">= 0.55",
        "d_max_day_share": "<= 0.30", "e_oos_is_ratio": "> 0.7",
        "f_sign_stable_at_2x_cost": True, "n_trials_declared": 12},
        "spread_ceiling_bps": SPREAD_CEILING, "n_admitted": len(admitted),
        "results": results}, indent=2))
    print(f"\n-> {out}")

    # Verdict against the PRE-REGISTERED criteria only.
    print("\nVERDICT (criteria declared before the run):")
    survivors = []
    for r in results:
        if not r.get("n") or r["stress"] != 1.0:
            continue
        stressed = next((s for s in results if s.get("n") and s["k"] == r["k"]
                         and s["long_short"] == r["long_short"] and s["stress"] == 2.0), None)
        checks = {
            "a": r["sharpe_net"] > 0.5,
            "b": r["dsr_p"] < 0.05,
            "c": r["positive_share"] >= 0.55,
            "d": r["max_day_share"] <= 0.30,
            "e": (r["oos_is_ratio"] or 0) > 0.7,
            "f": bool(stressed and np.sign(stressed["net_total_pct"]) == np.sign(r["net_total_pct"])),
        }
        failed = [c for c, ok in checks.items() if not ok]
        tag = f"k={r['k']} {'L/S' if r['long_short'] else 'long-only'}"
        print(f"  {tag:<18} {'SURVIVES' if not failed else 'FAILS(' + ','.join(failed) + ')'}")
        if not failed:
            survivors.append(tag)
    print(f"\n  survivors: {survivors or 'NONE'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
