"""XS-10 — the standing rotation tracker: making an 8-month wait self-executing.

**Maturity: PRELIM** (planted + smoke pass, merged; `contracts/README.md` ladder). The
rotation it tracks is *not* promoted — FINDINGS §7.8 records 4 of 6 pre-registered criteria
passing, so nothing is DISCOVERED.

§7.8's only actionable conclusion is arithmetic: at the measured Sharpe the strategy needs
**~325 daily rebalances** to reach t = 2, and it has 83. A conclusion shaped like that decays
into nothing unless something re-runs it as the `XS-7` candle archive grows — so this turns
the wait into a measured trajectory.

What it exists to catch is not the good case. If the edge is real, t climbs as √n and the
answer arrives on schedule. If §7.8's in-sample design choice was optimistic — XS-9's
construction was chosen *after* seeing XS-6 fail on the same 83 days — then the Sharpe
**decays as the sample grows**, and the sequence of rows makes that visible early rather
than at the end of the eight months. That is why the trajectory is append-only: a single
overwritten number always looks like the present.

The six acceptance criteria are **imported verbatim** from the XS-6 driver, never
recomputed here. A tracker that quietly relaxed a threshold as data accrued would be
precisely the failure pre-registration exists to prevent.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["CRITERIA", "power_status", "evaluate_criteria",
           "append_trajectory", "read_trajectory", "default_trajectory_path"]

#: The pre-registered set, declared in `exploration/xs_rotation_study.py` before its run
#: and committed in f3eea78. Text form kept so the tracker cannot drift from the record.
CRITERIA = {
    "a": "net Sharpe > 0.5 after SSOT costs",
    "b": "deflated Sharpe p < 0.05 (penalised for declared trials)",
    "c": "positive-period share >= 0.55",
    "d": "no single period > 0.30 of total P&L",
    "e": "OOS/IS Sharpe ratio > 0.7",
    "f": "sign stable under 2x cost stress",
}


def default_trajectory_path() -> Path:
    try:
        import nat_paths
        return nat_paths.state_dir("xs") / "rotation_trajectory.jsonl"
    except Exception:
        return Path(__file__).resolve().parents[2] / "data" / "xs" / "rotation_trajectory.jsonl"


def power_status(sharpe_annual: float, n_periods: int,
                 periods_per_year: float = 365.0, t_target: float = 2.0) -> dict:
    """How far the series is from resolving, in periods.

    `t = SR_period * sqrt(n)`, so `n_required = (t_target / SR_period)^2`. Because that is
    quadratic in the Sharpe, doubling SR **quarters** the requirement — the relation that
    turned §7.7's 2.55-year wait into §7.8's 0.89.

    A non-positive Sharpe is never resolvable: there is no n at which a zero-or-negative
    mean becomes significantly positive, and reporting a finite one would invent a schedule
    for a dead strategy.
    """
    sr_period = float(sharpe_annual) / math.sqrt(periods_per_year)
    n = int(n_periods)
    t_stat = sr_period * math.sqrt(n) if n > 0 else 0.0

    if sr_period <= 0:
        return {"sharpe_annual": float(sharpe_annual), "sharpe_period": sr_period,
                "n_periods": n, "t_stat": t_stat, "t_target": t_target,
                "n_required_t2": float("inf"), "n_remaining": float("inf"),
                "resolved": False}

    n_req = (t_target / sr_period) ** 2
    return {
        "sharpe_annual": float(sharpe_annual), "sharpe_period": sr_period,
        "n_periods": n, "t_stat": t_stat, "t_target": t_target,
        "n_required_t2": n_req,
        "n_remaining": max(0.0, n_req - n),
        "resolved": bool(n >= n_req),
    }


def evaluate_criteria(metrics: dict) -> tuple[list[str], list[str]]:
    """Apply the six pre-registered criteria. Returns (passed, failed).

    A **missing** metric fails its criterion rather than passing silently: an absent number
    is not a satisfied condition, and the opposite convention is how a partial run gets
    reported as a survivor.
    """
    def _get(key):
        v = metrics.get(key)
        return v if isinstance(v, (int, float, bool)) and not (
            isinstance(v, float) and math.isnan(v)) else None

    checks = {
        "a": (lambda: (_get("sharpe_net") or -math.inf) > 0.5),
        "b": (lambda: (_get("dsr_p") if _get("dsr_p") is not None else math.inf) < 0.05),
        "c": (lambda: (_get("positive_share") or -math.inf) >= 0.55),
        "d": (lambda: (_get("max_day_share") if _get("max_day_share") is not None
                       else math.inf) <= 0.30),
        "e": (lambda: (_get("oos_is_ratio") or -math.inf) > 0.7),
        "f": (lambda: bool(_get("sign_stable_2x"))),
    }
    passed, failed = [], []
    for key, fn in checks.items():
        (passed if fn() else failed).append(key)
    return passed, failed


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[2]).stdout.strip() or None
    except Exception:
        return None


def append_trajectory(path: Path | str, record: dict) -> dict:
    """Append one measurement. **Never rewrites** — the sequence is the product.

    Whether t is climbing as √n or the Sharpe is decaying as the sample grows is only
    visible across rows, and the decay case is the one worth catching early.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), "git_sha": _git_sha(), **record}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def read_trajectory(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
