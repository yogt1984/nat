"""WP-3 — cohort construction from the WP-2 position panel.

Step 3 of `docs/specs/wallet_positioning.md`, and tier-0 item #4 of `FINAL_PLAN.md`. It carries
the **early kill**: if cohort membership does not persist, family 5's positioning branch is dead
permanently at ~30 days, without waiting for WP-5 on 2026-11-07.

**The causality rule is the whole unit.** Cohorts are ranked on a window ending **strictly
before** `as_of`, re-ranked walk-forward. A ranking window that touches the evaluation period is
the A-2 error in new clothing — the combiner's weights were fitted three days *after* the window
they were scored on, and that alone produced its result (§5.1). `tests/test_cohorts.py` pins it
with a leakage test *and* an in-sample control, because a leakage test that passes against a
broken ranker is indistinguishable from one that passes against a causal one.

**Realised P&L, and why flows are read rather than inferred.** From snapshots alone

    Δaccount_value − Δ uPnL  =  realised P&L + flows + fees

with no way to split the terms. Measured on 20,065 real WP-2 intervals the residual has a 99th
percentile of 0.43 of account value and **6.6 % of intervals move account value by >2 % net of
uPnL** — so the spec's original "flag as unattributable" would file 6.6 % of the panel as
unknown. `data/fetch_ledger.py` reads the flows explicitly instead, leaving

    realised P&L  =  Δaccount_value − Δ uPnL − net_perp_flow

A window containing an **unknown** ledger delta type is reported `contaminated=True` rather than
returned as a clean number: unknown is not zero, and a caller must be able to tell.

**Normalisation.** Cohort positioning is divided by cohort account value, so one whale does not
become the cohort.

**Rank stability is measured and reported, never assumed** — "the cohort is not a cohort" is
failure mode 1 of the spec, and the number is emitted even when it is bad, because the early kill
needs the measurement rather than a refusal to produce one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from data.fetch_ledger import net_perp_flow
    from data.wallet_roster import is_valid_address
except ImportError:                        # pragma: no cover - path bootstrap
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.fetch_ledger import net_perp_flow
    from data.wallet_roster import is_valid_address

__all__ = ["account_panel", "realised_pnl", "rank_cohorts", "cohort_net_positioning",
           "rank_stability", "load_position_panel", "load_ledger"]

#: A cohort of one is a wallet, not a cohort — every statistic below would be that wallet's.
MIN_COHORT = 2


def account_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the (ts, wallet, coin) position panel to one row per (ts, wallet).

    `account_value` is a wallet-level field repeated on every coin row, so it is taken once;
    `unrealized_pnl` is per position and summed.
    """
    ok = panel[panel.get("status", "ok").astype(str) == "ok"] if "status" in panel else panel
    return (ok.groupby(["ts_ms", "wallet"], as_index=False)
              .agg(account_value=("account_value", "first"),
                   upnl=("unrealized_pnl", "sum"))
              .sort_values(["wallet", "ts_ms"])
              .reset_index(drop=True))


def realised_pnl(panel: pd.DataFrame, ledger: dict, wallet: str,
                 t0_ms: int, t1_ms: int) -> tuple[float, dict]:
    """Realised P&L for `wallet` over `(t0_ms, t1_ms]`, net of flows.

    Returns `(pnl, report)`. `report["contaminated"]` is True when the window contains a ledger
    delta type with no handler — the figure is then not trustworthy, and callers must not treat
    it as merely approximate.
    """
    acct = account_panel(panel)
    s = acct[(acct.wallet == wallet) & (acct.ts_ms >= t0_ms) & (acct.ts_ms <= t1_ms)]
    if len(s) < 2:
        return float("nan"), {"n_obs": len(s), "net_flow": 0.0, "contaminated": False,
                              "unknown_count": 0, "reason": "fewer than two snapshots"}

    d_av = float(s.account_value.iloc[-1] - s.account_value.iloc[0])
    d_upnl = float(s.upnl.iloc[-1] - s.upnl.iloc[0])
    flow, frep = net_perp_flow(ledger.get(wallet, []), wallet, t0_ms, t1_ms)

    pnl = d_av - d_upnl - flow
    return pnl, {"n_obs": len(s), "d_account_value": d_av, "d_upnl": d_upnl,
                 "net_flow": flow, "unknown_count": frep["unknown_count"],
                 "unknown_types": frep["unknown_types"],
                 "contaminated": frep["unknown_count"] > 0}


def rank_cohorts(panel: pd.DataFrame, as_of: int, lookback_ms: int, k: int = 20,
                 ledger: dict | None = None) -> dict:
    """Rank wallets on realised P&L over `[as_of - lookback, as_of)`, strictly before `as_of`.

    The window is **half-open on the right**: no observation at or after `as_of` may influence
    membership. That single line is what the leakage test defends.
    """
    ledger = ledger or {}
    start = int(as_of - lookback_ms)
    end = int(as_of) - 1                     # strictly before as_of
    if end <= start:
        raise ValueError(f"empty ranking window [{start}, {end}]")

    acct = account_panel(panel)
    scores, reports = {}, {}
    for w in sorted(acct.wallet.unique()):
        pnl, rep = realised_pnl(panel, ledger, w, start, end)
        if np.isfinite(pnl):
            scores[w] = pnl
            reports[w] = rep

    if len(scores) < max(2 * k, MIN_COHORT):
        raise ValueError(
            f"only {len(scores)} wallets have usable history in [{start}, {end}] — "
            f"need at least {max(2 * k, MIN_COHORT)} for top/bottom cohorts of {k}")

    ordered = sorted(scores, key=lambda w: (-scores[w], w))     # ties break on address
    return {
        "as_of": int(as_of),
        "window": (start, end),
        "top": ordered[:k],
        "bottom": ordered[-k:],
        "scores": {w: float(scores[w]) for w in ordered},
        "n_ranked": len(ordered),
        "n_contaminated": sum(1 for r in reports.values() if r["contaminated"]),
    }


def cohort_net_positioning(panel: pd.DataFrame, cohort, coin: str, as_of: int) -> float:
    """Signed notional the cohort holds in `coin` at `as_of`, per unit of cohort account value.

    Normalised so one whale is not the cohort: doubling a member's size **and** its account
    value leaves the signal unchanged.
    """
    members = [w for w in (cohort or []) if is_valid_address(w)]
    if len(members) < MIN_COHORT:
        raise ValueError(f"cohort has {len(members)} valid members; need >= {MIN_COHORT}")

    snap = panel[(panel.ts_ms <= as_of) & (panel.wallet.isin(members))]
    if snap.empty:
        return float("nan")
    latest = snap.ts_ms.max()
    snap = snap[snap.ts_ms == latest]

    pos = snap[snap.coin == coin]
    signed = float((np.sign(pos["size"].to_numpy(float))
                    * pos["position_value"].to_numpy(float)).sum())
    # account_value is wallet-level; take it once per wallet, not once per coin row
    denom = float(snap.groupby("wallet").account_value.first().sum())
    return signed / denom if denom else float("nan")


def rank_stability(memberships) -> dict:
    """Cohort-membership persistence across consecutive rebalances.

    `mean_overlap` is the mean Jaccard index between successive cohorts. Reported even when it
    is poor — a low number *is* the early-kill result, not a reason to withhold it.
    """
    groups = [set(g) for g in (memberships or [])]
    if len(groups) < 2:
        return {"n_rebalances": len(groups), "mean_overlap": float("nan"), "overlaps": []}

    overlaps = []
    for a, b in zip(groups, groups[1:]):
        union = a | b
        overlaps.append(len(a & b) / len(union) if union else float("nan"))
    finite = [o for o in overlaps if np.isfinite(o)]
    return {"n_rebalances": len(groups),
            "mean_overlap": float(np.mean(finite)) if finite else float("nan"),
            "min_overlap": float(np.min(finite)) if finite else float("nan"),
            "overlaps": [float(o) for o in overlaps]}


# ── loaders ──────────────────────────────────────────────────────────────────────

def load_position_panel(data_dir: Path | str | None = None, days: int | None = None
                        ) -> pd.DataFrame:
    """Read WP-2 sweeps into one frame.

    Days written before 2026-08-13 lack `total_raw_usd`; `pd.concat` fills it with NaN, which is
    the intended behaviour (spec §B3 — readers tolerate the column's absence).
    """
    from data.fetch_positions import DATA_DIR
    root = Path(data_dir) if data_dir else DATA_DIR
    day_dirs = sorted(p for p in root.glob("*-*-*") if p.is_dir())
    if days:
        day_dirs = day_dirs[-days:]
    files = [f for d in day_dirs for f in sorted(d.glob("positions_*.parquet"))]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_ledger(data_dir: Path | str | None = None) -> dict:
    """Read the Part B backfill into `{wallet: [entry, ...]}` in `net_perp_flow`'s shape."""
    import json as _json
    from data.fetch_ledger import DATA_DIR
    root = Path(data_dir) if data_dir else DATA_DIR
    out: dict[str, list] = {}
    if not root.exists():
        return out
    for p in sorted(root.glob("*.parquet")):
        df = pd.read_parquet(p)
        out[p.stem] = [{"time": int(r.time), "hash": r.hash,
                        "delta": _json.loads(r.raw)} for r in df.itertuples()]
    return out
