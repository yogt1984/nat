"""XS-10 — the beta-neutral cross-sectional rotation, as a callable unit.

**Maturity: PRELIM.** Nothing here is promoted: FINDINGS §7.8 records 4 of 6 pre-registered
criteria passing, and §7.7 records 0 of 6 for the construction this replaced.

This is the strategy §7.8 measured, extracted from the one-off script that produced it so
the trajectory tracker can re-run it as the `XS-7` archive grows. That extraction is the
point — a result computed by a script nobody re-runs is a snapshot, and §7.8's only
conclusion is that the question needs ~325 rebalances against the 83 it had.

**The construction, and why each piece is there** (FINDINGS §7.8):

* Score = realized vol over the trailing window, **residualised on beta**. Vol correlates
  with beta at 0.556, so ranking on raw vol implicitly ranks on beta.
* Weights ∝ **−z(score)**: negative polarity, long low residual-vol. The sign is the finding
  (§7.4), not a convention.
* The beta dimension is then **projected out**, so net portfolio beta is ~0. §7.7's
  equal-weight top-k carried a −0.33 tilt that produced 80 % of P&L variance while earning
  nothing — beta does not predict relative returns (IC −0.026, t −1.01).
* Costs are each pair's own measured half-spread plus the SSOT taker and slippage, via
  `load_costs()`. No literals: the first version of the §7.7 driver used `.get(..., 4.5)`
  fallbacks that silently supplied hardcoded fees.

The honest caveat travels with the code: this construction was chosen *after* seeing §7.7
fail on the same 83 days. The mechanism is theory rather than search, but the measured
magnitude is an upper bound until the window grows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["run_rotation", "DEFAULTS"]

DEFAULTS = dict(lookback=168, rebalance=24, spread_ceiling_bps=2.0, is_frac=0.6,
                periods_per_year=365.0)


def _sharpe(r: np.ndarray, periods_per_year: float) -> float:
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def run_rotation(wide: pd.DataFrame, cost_bps: pd.Series, *, lookback: int = 168,
                 rebalance: int = 24, is_frac: float = 0.6,
                 periods_per_year: float = 365.0, cost_stress: float = 1.0,
                 band: float = 0.0, band_mode: str = "edge",
                 funding_wide: pd.DataFrame | None = None) -> dict:
    """Run the beta-neutral rotation over a (timestamp x symbol) close matrix.

    Args:
        wide: close prices, NaN where a pair had not listed (PROC-19 keeps the holes).
        cost_bps: per-pair round-trip cost in bps for one unit of turnover.
        cost_stress: multiplier for the §4.9 criterion-(f) sensitivity run.
        funding_wide: per-bar funding RATE (a fraction, like `fwd` — not bps), aligned to
            `wide`. Optional: `None` reproduces the pre-COST-9 result exactly, so any
            re-pricing delta is attributable to funding and nothing else.

    Returns a metrics dict shaped for `xs.trajectory.evaluate_criteria`.

    **Funding (COST-9).** This priced only turnover until 2026-08-11; inventory held between
    rebalances paid nothing, which the XS-6 study docstring declared but never closed. Since
    Hyperliquid settles hourly and these bars are hourly, the charge over one holding period
    is `Σ_i held_i × Σ_t f[t, i]` — signed, so a long pays a positive rate and a short
    receives it, matching `backtest.costs.CostModel.compute_pnl`.

    On a beta-neutral book with a *uniform* rate this nets to ~0: the weights sum to zero,
    so what survives is exposure to the cross-sectional **dispersion** of funding rates, not
    their level. That is why the headline "21 bps/week" is an upper bound belonging to
    long-only configurations, and why both cases are pinned in
    `tests/test_xs_rotation_funding.py`.
    """
    cols = list(wide.columns)
    prev = pd.Series(0.0, index=cols)
    rows = []

    for i in range(lookback, len(wide) - rebalance, rebalance):
        hist = wide.iloc[i - lookback:i + 1]
        live = [c for c in cols
                if hist[c].notna().sum() >= lookback * 0.8
                and np.isfinite(wide.iloc[i + rebalance][c]) and np.isfinite(wide.iloc[i][c])]
        if len(live) < 40:
            continue

        r = np.log(hist[live]).diff().dropna(how="all").fillna(0.0)
        mkt = r.mean(axis=1)
        beta = np.array([np.polyfit(mkt, r[c], 1)[0] for c in live])
        vol = r.std().to_numpy()
        fwd = (wide.iloc[i + rebalance][live] / wide.iloc[i][live] - 1.0).to_numpy(float)

        # score = what vol knows that beta doesn't, standardised, negative polarity
        resid = vol - np.polyval(np.polyfit(beta, vol, 1), beta)
        sd = resid.std() or 1.0
        wts = pd.Series(-(resid - resid.mean()) / sd, index=live)

        # project out beta so the net exposure is zero, then normalise to unit gross
        bser = pd.Series(beta, index=live)
        wts = wts - (wts * bser).sum() / float((bser * bser).sum()) * bser
        gross_abs = wts.abs().sum()
        if gross_abs <= 0:
            continue
        wts = wts / gross_abs

        full = wts.reindex(cols).fillna(0.0)
        if band > 0:
            # A5 hysteresis: skip drifts too small to pay for themselves. "edge" is the
            # proportional-cost optimum (trade to the boundary, not the target).
            from execution.rebalance import no_trade_band, trade_to_edge
            fn = trade_to_edge if band_mode == "edge" else no_trade_band
            full = fn(full, prev, band)
        turn = (full - prev).abs()
        cost = float((turn * cost_bps.reindex(turn.index).fillna(
            cost_bps.median())).sum()) * 1e-4 * cost_stress
        # Gross must be priced on the weights actually HELD, not the target. With a
        # hysteresis band these differ, and using `wts` here reported falling cost with
        # unchanged gross — free money, and therefore a bug. Identical when band == 0,
        # which is why it stayed latent until A5.
        held = full.reindex(live).fillna(0.0)
        gross = float((held * pd.Series(fwd, index=live)).sum())

        # Funding on the inventory actually HELD over [i, i+rebalance), on the same
        # weights `gross` is priced on. Missing history is 0 for that cell, never NaN for
        # the period: a pair listed mid-window would otherwise delete a whole rebalance
        # from a study whose binding constraint is already n (§7.8 wanted ~325, had 83).
        funding = 0.0
        if funding_wide is not None:
            fw = funding_wide.iloc[i:i + rebalance]
            rate_sum = fw.reindex(columns=live).sum(axis=0, skipna=True).fillna(0.0)
            funding = float((held * rate_sum).sum())

        rows.append({"gross": gross, "cost": cost, "funding": funding,
                     "net": gross - cost - funding,
                     "turnover": float(turn.sum()),
                     "net_beta": float((wts * bser).sum())})
        prev = full

    if not rows:
        return {"n_periods": 0, "reason": "no rebalance reached the minimum universe"}

    d = pd.DataFrame(rows)
    net = d.net.to_numpy()
    split = int(len(d) * is_frac)
    sr_is = _sharpe(net[:split], periods_per_year)
    sr_oos = _sharpe(net[split:], periods_per_year)
    total = float(d.net.sum())

    return {
        "n_periods": len(d),
        "gross_total_pct": round(float(d.gross.sum()) * 100, 3),
        "cost_total_pct": round(float(d.cost.sum()) * 100, 3),
        # COST-9: reported separately from turnover cost, because the two answer different
        # questions — turnover is a choice, funding is a toll on holding at all.
        "funding_total_pct": round(float(d.funding.sum()) * 100, 3),
        "net_total_pct": round(total * 100, 3),
        "sharpe_net": round(_sharpe(net, periods_per_year), 3),
        "sharpe_is": round(sr_is, 3),
        "sharpe_oos": round(sr_oos, 3),
        "oos_is_ratio": round(sr_oos / sr_is, 3) if sr_is else None,
        "positive_share": round(float((d.net > 0).mean()), 3),
        "max_day_share": round(float(d.net.abs().max() / abs(total)), 3) if total else None,
        "mean_turnover": round(float(d.turnover.mean()), 3),
        "mean_abs_net_beta": round(float(d.net_beta.abs().mean()), 6),
    }
