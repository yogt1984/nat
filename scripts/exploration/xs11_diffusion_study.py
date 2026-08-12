"""XS-11 — gradual information diffusion x liquidity (family 6). **Pre-registered.**

The mechanism, stated so it can fail (MECHANISM_FAMILIES.md): information reaches
holders of thin, low-attention perps more slowly than holders of BTC, so their
prices adjust over days rather than seconds. The counterparty is the slow-to-update
holder; the crypto analogue of Hong-Lim-Stein's "low analyst coverage" is a thin,
rarely-quoted pair.

Why the record's reversion results do NOT already answer this (and pooling would):
PROC-20/TC-1/XS-3 measured minutes-to-days, this lives at weeks; and every prior
test either pooled all 177 pairs (diluting a liquidity-conditional effect toward
zero) or excluded the illiquid tail outright. A pooled IC of -0.039 is consistent
with +0.10 in the illiquid tercile and -0.10 in the liquid one. Conditioning is
the entire study.

═══════════════════════════════════════════════════════════════════════════════
PRE-REGISTERED DESIGN — DECLARED BEFORE THE RUN
═══════════════════════════════════════════════════════════════════════════════
Grid (12 cells, declared for FDR): signal window ∈ {1w, 4w} trailing return
(skipping the most recent 24 h — short-horizon reversal is a *different*,
already-measured effect and must not contaminate the diffusion signal) ×
horizon ∈ {1w, 4w} forward return × liquidity tercile ∈ {tight, mid, wide} by
measured median half-spread (XS-8 L2 aggregate).

Per cell, at non-overlapping rebalances: cross-sectional Spearman rank IC
between signal and forward return, within tercile. Directional prior: family 6
predicts POSITIVE IC concentrated in the WIDE tercile; tested two-sided anyway.

VERDICT RULE, fixed now (one per cell — never a single pooled number):
  PRESENT      BH-FDR q < 0.05 over the declared 12 cells (PROC-13 convention)
  ABSENT       not significant AND 95% CI for mean IC inside (-0.10, +0.10)
               — a *powered* null, not an unpowered shrug
  UNDECIDABLE  otherwise — reported as such, never as refutation
Tradability, computed for every PRESENT cell (and reported for all): expected
per-period move = |mean IC| x cross-sectional dispersion of forward returns;
cost = tercile median round trip (own half-spread + SSOT taker + slippage) x
measured rebalance churn. TRADEABLE iff move/cost >= 3 (the m/c rule at the
design horizon). Final labels per cell: real-and-tradeable / real-and-
untradeable / absent / undecidable.

POWER, PRE-RECORDED: 90 d of 1 h candles = 12 non-overlapping weeks and only 2
non-overlapping months. The 4w-horizon cells are undecidable BY DESIGN and run
only so the point estimate is on record; *undecidable* is the expected verdict
for most of the grid (TASKS §0 queue row 5 says exactly this). At n=12, ABSENT
requires an IC standard error below ~0.05 — attainable only if the
cross-sectional IC is very stable.

DECLARED BIASES, all working against a clean read: (i) survivorship — 55
delisted perps are absent, and delisting concentrates in the wide tercile where
the effect is predicted; (ii) cost is worst exactly where the edge is predicted;
(iii) tercile membership is measured from the August L2 sample, i.e. at the END
of the price window — spread ranks are highly persistent, but this is
end-of-window conditioning and is recorded as a limitation, not hidden.

NOT tested here: attention/listing events (family 7, n=2), anything at the
minute scale (already settled), and any live-fill claim.
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

from processes.candles import available_candle_symbols, load_candles  # noqa: E402
from utils.costs import load_costs, slippage_bps, taker_bps  # noqa: E402
from xs.capacity import aggregate_l2, load_l2_snapshots  # noqa: E402

HOURS_WEEK = 168
SKIP_H = 24                       # reversal guard, declared above
SIGNAL_WINDOWS_H = (HOURS_WEEK, 4 * HOURS_WEEK)
HORIZONS_H = (HOURS_WEEK, 4 * HOURS_WEEK)
TERCILES = ("tight", "mid", "wide")
N_CELLS = 12                      # declared FDR family
FDR_Q = 0.05
ABSENT_IC_BAND = 0.10
MC_MIN_RATIO = 3.0                # the m/c rule, imported from TASKS §0


# ── pure construction (planted tests exercise these) ─────────────────────────

def liquidity_terciles(half_spread_bps: pd.Series) -> pd.Series:
    """Static tercile label per symbol by measured median half-spread."""
    ranks = half_spread_bps.rank(pct=True)
    return pd.Series(np.where(ranks <= 1 / 3, "tight",
                              np.where(ranks <= 2 / 3, "mid", "wide")),
                     index=half_spread_bps.index)


def momentum_signal(prices: pd.DataFrame, t_idx: int, window_h: int,
                    skip_h: int = SKIP_H) -> pd.Series:
    """Trailing return over [t-skip-window, t-skip], strictly backward-looking."""
    hi = t_idx - skip_h
    lo = hi - window_h
    if lo < 0:
        return pd.Series(dtype=float)
    p0, p1 = prices.iloc[lo], prices.iloc[hi]
    sig = (p1 / p0 - 1.0).replace([np.inf, -np.inf], np.nan)
    return sig.dropna()


def forward_return(prices: pd.DataFrame, t_idx: int, horizon_h: int) -> pd.Series:
    p0, p1 = prices.iloc[t_idx], prices.iloc[t_idx + horizon_h]
    fwd = (p1 / p0 - 1.0).replace([np.inf, -np.inf], np.nan)
    return fwd.dropna()


def cell_ics(prices: pd.DataFrame, members: list[str], window_h: int,
             horizon_h: int, min_names: int = 10) -> list[dict]:
    """Rank IC at each NON-OVERLAPPING rebalance (spacing = horizon)."""
    out = []
    start = max(SKIP_H + window_h, 0)
    for i in range(start, len(prices) - horizon_h, horizon_h):
        sig = momentum_signal(prices, i, window_h)
        fwd = forward_return(prices, i, horizon_h)
        common = [c for c in members if c in sig.index and c in fwd.index]
        if len(common) < min_names:
            continue
        s, f = sig[common], fwd[common]
        ic = s.rank().corr(f.rank())
        if np.isfinite(ic):
            out.append({"t_idx": i, "ic": float(ic), "n_names": len(common),
                        "fwd_dispersion": float(f.std())})
    return out


def summarize_cell(ics: list[dict]) -> dict:
    vals = np.array([r["ic"] for r in ics])
    n = len(vals)
    if n < 2:
        return {"n_periods": n, "mean_ic": None, "t_stat": None, "p_value": 1.0}
    mean, se = float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(n))
    # se == 0 with a nonzero mean is a *perfectly stable* IC (the planted-signal
    # case), not a zero t — collapsing it to t=0 would file a perfect signal as
    # undecidable.
    if se > 0:
        t = mean / se
    else:
        t = float(np.inf * np.sign(mean)) if mean != 0 else 0.0
    from scipy import stats
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))) if np.isfinite(t) \
        else (0.0 if mean != 0 else 1.0)
    ci = stats.t.ppf(0.975, df=n - 1) * se
    return {"n_periods": n, "mean_ic": round(mean, 4), "se": round(se, 4),
            "t_stat": round(t, 2), "p_value": round(p, 4),
            "ci_lo": round(mean - ci, 4), "ci_hi": round(mean + ci, 4),
            "mean_fwd_dispersion": round(float(np.mean(
                [r["fwd_dispersion"] for r in ics])), 5)}


def bh_fdr(pvals: list[float], q: float = FDR_Q) -> list[bool]:
    """Benjamini-Hochberg over the declared family; returns pass flags."""
    n = len(pvals)
    order = np.argsort(pvals)
    passed = [False] * n
    thresh = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / n:
            thresh = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= thresh:
            passed[idx] = True
    return passed


def verdict(cell: dict, fdr_pass: bool) -> str:
    if cell.get("mean_ic") is None:
        return "undecidable"
    if fdr_pass:
        return "present"
    if cell["ci_lo"] > -ABSENT_IC_BAND and cell["ci_hi"] < ABSENT_IC_BAND:
        return "absent"
    return "undecidable"


def tradability(cell: dict, tercile_rt_cost_bps: float) -> dict:
    """m/c at the design horizon: |IC| x dispersion vs the round trip."""
    if cell.get("mean_ic") is None:
        return {"mc_ratio": None}
    move_bps = abs(cell["mean_ic"]) * cell["mean_fwd_dispersion"] * 1e4
    ratio = move_bps / tercile_rt_cost_bps if tercile_rt_cost_bps > 0 else np.inf
    return {"expected_move_bps": round(move_bps, 2),
            "rt_cost_bps": round(tercile_rt_cost_bps, 2),
            "mc_ratio": round(float(ratio), 2),
            "tradeable": bool(ratio >= MC_MIN_RATIO)}


# ── orchestration ────────────────────────────────────────────────────────────

def run_study(prices: pd.DataFrame, half_spread: pd.Series) -> dict:
    terciles = liquidity_terciles(half_spread)
    rt_cost = 2.0 * (half_spread + taker_bps() + slippage_bps())
    cells = []
    for w in SIGNAL_WINDOWS_H:
        for h in HORIZONS_H:
            for terc in TERCILES:
                members = terciles[terciles == terc].index.tolist()
                ics = cell_ics(prices, members, w, h)
                cell = {"window_h": w, "horizon_h": h, "tercile": terc,
                        **summarize_cell(ics)}
                cell.update(tradability(
                    cell, float(rt_cost[terciles == terc].median())))
                cells.append(cell)
    assert len(cells) == N_CELLS
    flags = bh_fdr([c["p_value"] for c in cells])
    for c, f in zip(cells, flags):
        c["fdr_pass"] = bool(f)
        v = verdict(c, f)
        if v == "present":
            v = "real-and-tradeable" if c.get("tradeable") else "real-and-untradeable"
        c["verdict"] = v
    return {"cells": cells,
            "tercile_sizes": terciles.value_counts().to_dict(),
            "criteria": {"fdr_q": FDR_Q, "n_cells_declared": N_CELLS,
                         "absent_ic_band": ABSENT_IC_BAND,
                         "mc_min_ratio": MC_MIN_RATIO, "skip_h": SKIP_H}}


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    agg = aggregate_l2(load_l2_snapshots(), min_snapshots=10)
    syms = [s for s in available_candle_symbols(interval="1h") if s in agg.index]
    frame = load_candles(syms, "1h")
    prices = frame.pivot_table(index="timestamp", columns="symbol",
                               values="close", aggfunc="last").sort_index()
    hs = agg.loc[[s for s in prices.columns if s in agg.index], "half_spread_bps"]

    print(f"universe: {len(hs)} pairs with candles + L2 measurement; "
          f"span {prices.index[0]} -> {prices.index[-1]} ({len(prices)} hourly rows)")
    _ = load_costs()  # fail fast if the SSOT is unreadable

    result = run_study(prices, hs)
    print(f"terciles: {result['tercile_sizes']}\n")
    print(f"{'wind':>5} {'hori':>5} {'terc':>6} {'n':>3} {'meanIC':>8} "
          f"{'t':>6} {'p':>7} {'m/c':>6}  verdict")
    for c in result["cells"]:
        print(f"{c['window_h'] // HOURS_WEEK:>4}w {c['horizon_h'] // HOURS_WEEK:>4}w "
              f"{c['tercile']:>6} {c['n_periods']:>3} "
              f"{c['mean_ic'] if c['mean_ic'] is not None else '—':>8} "
              f"{c['t_stat'] if c['t_stat'] is not None else '—':>6} "
              f"{c['p_value']:>7} "
              f"{c['mc_ratio'] if c['mc_ratio'] is not None else '—':>6}  "
              f"{c['verdict']}")

    out = Path("reports/xs11_diffusion_study.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    verdicts = [c["verdict"] for c in result["cells"]]
    print(f"\nsummary: { {v: verdicts.count(v) for v in sorted(set(verdicts))} }")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
