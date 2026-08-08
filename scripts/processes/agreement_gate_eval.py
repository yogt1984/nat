"""A-1: `agreement_gate_eval` — is conditioning on agreement better than an arbitrary split?

§5 makes two claims about the hierarchical combiner. A-2 refuted the first: the composite
scores walk-forward IC 0.06/0.10/−0.02 against `trend_ema_short` alone at ~0.20, so the stack
destroys information rather than adding it. This process tests the second and more
interesting claim — *"L2 conditional-on-agreement IC exceeds unconditional — the first
architecture structurally addressing §2"* — which is the only structure in the record that
claims to attack the adverse-selection collapse head-on rather than route around it.

**The failure mode is selection, not estimation.** Split any sample on any condition and
report the better half, and you will find a lift: the agreement subset is smaller and
differently distributed, so the maximum of (agree, disagree) exceeds the pooled figure by
construction. A check of the form "agreement IC > unconditional IC" passes on pure noise, and
§5's pilot is exactly that shape.

So the null permutes **the gate**, not the outcome. It reshuffles which observations count as
agreeing while holding the fast signal, the target, and — critically — the **subset size**
fixed, which asks the only question worth asking: *is this partition better than an arbitrary
partition of the same shape?* Holding size fixed matters because IC's sampling variance
depends on n; a null that resized the subset would measure sample size instead of structure.

Reported per (fast, slow, horizon): `ic_agree`, `ic_disagree`, `ic_unconditional`, the raw
lift, its null z, and — per §4.9, where day-consistency was the binding failure rather than
the pooled mean — a per-day series and a `durable | non_durable` verdict.

Registered as a standing evaluation (`processes/standing.py`) so the §5 pilot becomes a
monitored fact rather than a citation, which is what the A-1 row asks for.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from alpha.screener import compute_forward_returns
from it_engine.null_calibration import load_null_config

from .base import EvaluationProcess, Finding, ProcessContext, ProcessResult, make_run_id
from .mi_stability import MIStabilityProcess
from .registry import register


@register
class AgreementGateEval(EvaluationProcess):
    """Conditional IC given fast/slow agreement, calibrated against a size-preserving null."""

    data_level = "bars"

    PARAMS = {
        "fast": ("alg_mp_dev_ema", "the fast directional signal"),
        "slow": ("regime_divergence_1h", "the slow bias whose agreement gates it"),
        "n_shuffles": (None, "gate permutations (default: it_engine.toml)"),
        "day_shuffles": (60, "gate permutations per day fold"),
        "min_obs": (100, "minimum jointly-valid rows"),
        "min_side": (30, "minimum rows on EACH side of the partition"),
        "min_days": (3, "minimum usable days before a durability verdict"),
        "min_frac_informative": (0.6, "day fraction required for 'durable'"),
        "seed": (0, "RNG seed"),
    }

    def name(self) -> str:
        return "agreement_gate_eval"

    # ── the process ──────────────────────────────────────────────────────────────
    def evaluate(self, bars: pd.DataFrame, ctx: ProcessContext) -> ProcessResult:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        p = self.params
        fast_col, slow_col = p["fast"], p["slow"]

        missing = [c for c in (fast_col, slow_col, ctx.price_col) if c not in bars.columns]
        if missing:
            return result.finalize(time.time() - t0, error=f"missing columns: {missing}")

        days = MIStabilityProcess._day_key(bars)          # PROC-4's calendar folds
        cfg = load_null_config()
        z_thr = float(cfg["null_z_threshold"])
        n_shuffles = int(p["n_shuffles"] or cfg["n_shuffles"])

        fast = bars[fast_col].to_numpy(dtype=np.float64)
        slow = bars[slow_col].to_numpy(dtype=np.float64)
        price = bars[ctx.price_col].to_numpy(dtype=np.float64)
        agree_all = np.sign(fast) == np.sign(slow)

        for h_name, h_bars in (ctx.horizons or {}).items():
            fwd = compute_forward_returns(price, int(h_bars))
            valid = np.isfinite(fast) & np.isfinite(slow) & np.isfinite(fwd)
            if int(valid.sum()) < int(p["min_obs"]):
                continue
            f, y, ag = fast[valid], fwd[valid], agree_all[valid]
            n_a, n_d = int(ag.sum()), int((~ag).sum())
            if min(n_a, n_d) < int(p["min_side"]):
                result.findings.append(Finding(
                    feature=f"{fast_col}|{slow_col}", horizon=h_name,
                    metric="agreement_ic_lift", value=0.0, informative=False,
                    extras={"verdict": "degenerate_partition", "n_agree": n_a,
                            "n_disagree": n_d, "z": None, "raw_lift": None,
                            "null_preserves_subset_size": True, "per_day": []}))
                continue

            ic_a, ic_d, ic_u = _ic(f[ag], y[ag]), _ic(f[~ag], y[~ag]), _ic(f, y)
            raw_lift = ic_a - ic_u
            rng = np.random.default_rng(int(p["seed"]))
            z, p_val, null_mean = _score(raw_lift, _gate_null(f, y, n_a, ic_u,
                                                              n_shuffles, rng))

            per_day = self._per_day(days, valid, f, y, ag, int(p["day_shuffles"]),
                                    int(p["seed"]), z_thr, int(p["min_side"]))
            frac = float(np.mean([d["informative"] for d in per_day])) if per_day else 0.0
            verdict = ("insufficient_days" if len(per_day) < int(p["min_days"])
                       else "durable" if frac >= float(p["min_frac_informative"])
                       else "non_durable")

            result.findings.append(Finding(
                feature=f"{fast_col}|{slow_col}", horizon=h_name,
                metric="agreement_ic_lift", value=round(float(raw_lift), 6),
                p_value=None if p_val is None else round(p_val, 6), threshold=z_thr,
                # A lift is only real if it beats a same-shape arbitrary partition AND
                # holds across days. Either alone is how §5's pilot happened.
                informative=bool(z is not None and z >= z_thr and verdict == "durable"),
                extras={"ic_agree": round(ic_a, 6), "ic_disagree": round(ic_d, 6),
                        "ic_unconditional": round(ic_u, 6),
                        "raw_lift": round(float(raw_lift), 6),
                        "z": None if z is None else round(z, 3),
                        "null_mean": None if null_mean is None else round(null_mean, 6),
                        "n_agree": n_a, "n_disagree": n_d,
                        "null_preserves_subset_size": True,
                        "frac_days_informative": round(frac, 4),
                        "n_days": len(per_day), "verdict": verdict, "per_day": per_day}))

        return result.finalize(time.time() - t0)

    # ── per-day folds ────────────────────────────────────────────────────────────
    def _per_day(self, days, valid, f, y, ag, day_shuffles, seed, z_thr, min_side):
        if days is None:
            return []
        d_valid = days[valid]
        out = []
        for d in np.unique(d_valid):
            m = d_valid == d
            if min(int((m & ag).sum()), int((m & ~ag).sum())) < min_side:
                continue
            fd, yd, agd = f[m], y[m], ag[m]
            lift = _ic(fd[agd], yd[agd]) - _ic(fd, yd)
            rng = np.random.default_rng(seed + int(d))
            z, _, _ = _score(lift, _gate_null(fd, yd, int(agd.sum()), _ic(fd, yd),
                                              day_shuffles, rng))
            out.append({"day": int(d), "n": int(m.sum()), "lift": round(float(lift), 6),
                        "z": None if z is None else round(z, 3),
                        "informative": bool(z is not None and z >= z_thr)})
        return out


# ── helpers ──────────────────────────────────────────────────────────────────────
def _ic(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else 0.0


def _gate_null(f: np.ndarray, y: np.ndarray, n_agree: int, ic_uncond: float,
               n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Lifts obtained by partitioning at random into a subset of the SAME size.

    Permuting the gate rather than the outcome is the whole point: it asks whether this
    partition beats an arbitrary one, which is the question §5's pilot never posed.
    """
    n = len(f)
    out = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        idx = rng.choice(n, size=n_agree, replace=False)
        out[i] = _ic(f[idx], y[idx]) - ic_uncond
    return out


def _score(stat: float, draws: np.ndarray):
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return None, None, None
    mu, sd = float(d.mean()), float(d.std(ddof=1))
    z = float((stat - mu) / sd) if sd > 0 else 0.0
    p = float((d >= stat).sum() + 1) / (d.size + 1)
    return z, p, mu
