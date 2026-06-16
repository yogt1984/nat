"""T15 — approval-evidence reductions (NAT6/NAT7).

Pure functions that turn the OOS-validation state (data/oos_validation/state.json)
into the series the `nat viz paper` / `nat viz portfolio` commands render:
cumulative P&L, IC-decay (7-day rolling-sharpe proxy), per-signal risk, the G8
PASS/FAIL scorecard (reusing the canonical build_paper_report), and the
cross-signal correlation matrix (target < 0.35).

All reductions take a plain state dict so they are unit-testable without the
filesystem. They render real data once the OOS/paper data accrues, and degrade
gracefully (empty) for unknown signals or missing data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

# Lifecycle signal ids sometimes differ from the OOS algo keys.
_ALIASES = {"3f_liquidity": "3f"}


def load_oos_state(path: Path | str | None = None) -> dict:
    """Load the OOS-validation state; {} if absent/unreadable."""
    p = Path(path) if path else ROOT / "data" / "oos_validation" / "state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def resolve_name(state: dict, name: str) -> str | None:
    """Map a lifecycle signal id to its key in state['algos'] (or None)."""
    algos = state.get("algos", {})
    if name in algos:
        return name
    alias = _ALIASES.get(name)
    if alias and alias in algos:
        return alias
    # generic: strip a trailing _<word> qualifier (e.g. *_liquidity)
    if "_" in name and name.rsplit("_", 1)[0] in algos:
        return name.rsplit("_", 1)[0]
    return None


def _metrics(state: dict, name: str, symbol: str) -> dict | None:
    key = resolve_name(state, name)
    if key is None:
        return None
    sym = state["algos"][key].get("symbols", {}).get(symbol)
    return sym.get("metrics") if sym else None


def _daily(state: dict, name: str, symbol: str) -> list[dict]:
    key = resolve_name(state, name)
    if key is None:
        return []
    sym = state["algos"][key].get("symbols", {}).get(symbol)
    return sym.get("daily", []) if sym else []


# ── paper P&L + IC decay ────────────────────────────────────────────────────


def paper_pnl_series(state: dict, name: str, symbol: str = "BTC") -> list[dict]:
    """Cumulative-P&L trace [{date, cum_bps}]."""
    m = _metrics(state, name, symbol)
    return list(m.get("cumulative_pnl_series", [])) if m else []


def ic_decay_series(state: dict, name: str, symbol: str = "BTC") -> list[dict]:
    """7-day rolling-sharpe series [{date, sharpe}] — the IC-decay proxy."""
    m = _metrics(state, name, symbol)
    return list(m.get("rolling_sharpe_7d", [])) if m else []


# ── per-signal risk ─────────────────────────────────────────────────────────


def per_signal_risk(state: dict, name: str, symbol: str = "BTC") -> dict:
    """{sharpe, max_dd_bps, n_days, cum_pnl_bps, profit_factor} or {}."""
    m = _metrics(state, name, symbol)
    if not m:
        return {}
    daily = _daily(state, name, symbol)
    pnls = [float(d.get("total_net_bps", 0.0)) for d in daily]
    wins = sum(v for v in pnls if v > 0)
    losses = abs(sum(v for v in pnls if v < 0))
    pf = (wins / losses) if losses > 0 else float("inf")
    return {
        "sharpe": float(m.get("current_sharpe", 0.0)),
        "max_dd_bps": float(m.get("max_drawdown_bps", 0.0)),
        "n_days": int(m.get("n_days", len(daily))),
        "cum_pnl_bps": float(m.get("cumulative_pnl_bps", 0.0)),
        "profit_factor": pf,
    }


# ── G8 scorecard (provisional, from OOS — reuses the canonical mapping) ──────


def g8_scorecard(state: dict, name: str, gates: dict, symbol: str = "BTC") -> dict:
    """The 5 G8 PASS/FAIL criteria for *name*, derived from OOS metrics.

    Provisional preview: uses OOS current_sharpe / degradation as proxies for the
    live paper metrics until a real paper window accrues. Reuses
    promotion_daemon.build_paper_report so the gate logic is single-sourced.
    """
    key = resolve_name(state, name)
    m = _metrics(state, name, symbol)
    if key is None or not m:
        return {}
    from promotion_daemon import build_paper_report

    daily = _daily(state, name, symbol)
    max_loss_bps = max((abs(float(d.get("max_loss_bps", 0.0))) for d in daily), default=0.0)
    # baseline_sharpe may be a scalar or a per-symbol dict.
    bs = state["algos"][key].get("baseline_sharpe", 1.0)
    if isinstance(bs, dict):
        bs = bs.get(symbol, next(iter(bs.values()), 1.0))
    metrics = {
        "paper_sharpe": float(m.get("current_sharpe", 0.0)),
        "baseline_sharpe": float(bs),
        "max_daily_loss_pct": max_loss_bps / 100.0,  # bps → pct
        "ic_decay_pct": float(m.get("degradation", 0.0)),
        "infra_stable": True,
        "n_days": int(m.get("n_days", len(daily))),
    }
    return build_paper_report(metrics, gates)


# ── cross-signal correlation matrix ─────────────────────────────────────────


def signal_correlation_matrix(state: dict, names: list[str], symbol: str = "BTC") -> dict:
    """Cross-signal Spearman correlation of daily P&L over common dates.

    Returns {"signals": [resolvable names], "matrix": NxN}. Diagonal = 1.0.
    """
    series: dict[str, dict] = {}
    valid: list[str] = []
    for nm in names:
        daily = _daily(state, nm, symbol)
        if not daily:
            continue
        valid.append(nm)
        series[nm] = {d["date"]: float(d.get("total_net_bps", 0.0)) for d in daily}

    n = len(valid)
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dates = sorted(set(series[valid[i]]) & set(series[valid[j]]))
            if len(dates) < 3:
                rho = float("nan")
            else:
                a = np.array([series[valid[i]][d] for d in dates])
                b = np.array([series[valid[j]][d] for d in dates])
                if np.std(a) < 1e-15 or np.std(b) < 1e-15:
                    rho = 0.0
                else:
                    rho, _ = stats.spearmanr(a, b)
            matrix[i][j] = matrix[j][i] = float(rho)
    return {"signals": valid, "matrix": matrix}
