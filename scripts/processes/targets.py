"""PROC-17: the target as a first-class node.

Before this module a target was a string (`target_col`) threaded through
`ProcessContext` and re-interpreted by every consumer. Each of `ic_horizon`,
`ml_importance` and `info_theory` independently answered the same five questions:

    1. which target did the caller mean (param? context? neither)?
    2. does the column exist / is it usable?
    3. how is it materialised into an array?
    4. **which columns leak it and must never be scored as features?**
    5. which gate applies — the fee-based `i_min` floor, or the PROC-12 null z?

Question 4 is why this module exists. A barrier label ships with siblings (`tb_ret`,
`tb_hit_bars`) that encode the answer; scoring one of them against the label prints a
large, entirely circular MI. Three copies of that exclusion rule is three chances to
forget it — and the FDR machinery downstream cannot tell a circular finding from a real
one, because both are "significant".

A `Target` therefore owns its own materialisation, its leakage set, its gate and its
provenance. Its `signed` flag is the precondition for PROC-1: mutual information is
unsigned, so a *direction* can only come from a target that has one, and an unsigned
target (a magnitude, a 0/1 flag) can never produce a trading polarity — `polarity_of`
refuses rather than inventing one.

Spec: `docs/TASKS.md` PROC-17 (substrate for PROC-5/6/7). Consumers keep their public
parameter names; this module only removes the duplication behind them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from alpha.screener import compute_forward_returns

from .base import ProcessContext

#: Columns trivially correlated with a forward return (it is built from them).
PRICE_PREFIXES = ("raw_midprice", "raw_microprice")

#: Prefix whose members jointly encode a triple-barrier outcome.
BARRIER_PREFIX = "tb_"

#: Provenance name used by `surface.py` when the target is a forward return.
FORWARD_RETURN_LABEL = "fwd_ret"


class TargetNotFound(ValueError):
    """The requested target cannot be used — missing, all-NaN, or constant.

    Raised rather than degraded: falling back to forward returns would emit findings
    stamped with a target the process never actually scored.
    """


@dataclass(frozen=True)
class Target:
    """One prediction target: what a process is trying to predict, and its rules."""

    name: str                       # column name, or "fwd_ret_<horizon>"
    kind: str                       # "label" | "forward_return"
    horizon_name: str               # "label" for label mode, else the horizon key
    horizon_bars: int               # 0 for label mode
    column: Optional[str] = None    # source column (label mode only)
    signed: bool = True             # does the realised target carry a direction?
    price_col: str = "raw_midprice"
    extra_leakage: tuple[str, ...] = field(default_factory=tuple)

    # ── materialisation ──────────────────────────────────────────────────────────
    def values(self, bars: pd.DataFrame) -> np.ndarray:
        """Materialise the target. Never mutates `bars`."""
        if self.kind == "label":
            return bars[self.column].to_numpy(dtype=np.float64, na_value=np.nan)
        prices = bars[self.price_col].to_numpy(dtype=np.float64, na_value=np.nan)
        return compute_forward_returns(prices, self.horizon_bars)

    # ── rules the target owns ────────────────────────────────────────────────────
    @property
    def cost_gated(self) -> bool:
        """Only a tradeable return can be compared against a fee floor.

        A label is a classification outcome, not a P&L, so `i_min(fee, σ)` is
        meaningless against it (see `info_theory._evaluate_label`).
        """
        return self.kind == "forward_return"

    @property
    def gate(self) -> str:
        return "fee" if self.cost_gated else "null_z"

    @property
    def label_def(self) -> str:
        """Provenance string, matching `surface.py`'s convention."""
        return self.column if self.kind == "label" else FORWARD_RETURN_LABEL

    def leakage_columns(self, bars: pd.DataFrame) -> set[str]:
        """Columns that must never be scored as features against this target."""
        cols = set(self.extra_leakage)
        if self.kind == "label":
            cols.add(self.column)
            if self.column and self.column.startswith(BARRIER_PREFIX):
                cols |= {c for c in bars.columns if c.startswith(BARRIER_PREFIX)}
        else:
            cols.add(self.price_col)
            cols |= {c for c in bars.columns if c.startswith(PRICE_PREFIXES)}
        return cols

    # ── direction (the PROC-1 precondition) ──────────────────────────────────────
    def polarity_of(self, feature: np.ndarray,
                    bars: Optional[pd.DataFrame] = None) -> int:
        """Sign of the feature→target relation: +1 or -1.

        Refuses on an unsigned target. MI says information exists; only a signed target
        can say which way to trade, and PROC-1 will not compile a rule without one.
        """
        if not self.signed:
            raise ValueError(
                f"target {self.name!r} is unsigned — no trading direction is definable "
                "from it (PROC-1 requires an explicit polarity)")
        if bars is None:
            raise ValueError("polarity_of needs the frame to materialise the target")
        from scipy.stats import spearmanr
        y = self.values(bars)
        x = np.asarray(feature, dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            raise ValueError("not enough jointly-valid observations for a polarity")
        rho = spearmanr(x[m], y[m]).statistic
        if not np.isfinite(rho) or rho == 0.0:
            raise ValueError("relation has no sign (rho == 0)")
        return 1 if rho > 0 else -1

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "horizon_name": self.horizon_name,
                "horizon_bars": self.horizon_bars, "column": self.column,
                "signed": self.signed, "label_def": self.label_def, "gate": self.gate,
                "cost_gated": self.cost_gated}


# ── resolution ───────────────────────────────────────────────────────────────────
def resolve_target_col(params: dict, ctx: ProcessContext) -> Optional[str]:
    """The single precedence rule: explicit parameter > context > forward returns."""
    return (params or {}).get("target_col") or getattr(ctx, "target_col", None) or None


def _is_signed(values: np.ndarray) -> bool:
    """A target carries direction iff it realises both signs."""
    v = values[np.isfinite(values)]
    return bool(v.size and (v > 0).any() and (v < 0).any())


def resolve_targets(bars: pd.DataFrame, ctx: ProcessContext,
                    params: Optional[dict] = None) -> list[Target]:
    """Every target a process should evaluate, in the order it should report them.

    Label mode returns exactly one `Target` (horizon name "label", matching the existing
    finding convention); return mode returns one per `ctx.horizons` entry.
    """
    target_col = resolve_target_col(params or {}, ctx)
    if target_col:
        if target_col not in bars.columns:
            raise TargetNotFound(f"target_col '{target_col}' not in data")
        v = bars[target_col].to_numpy(dtype=np.float64, na_value=np.nan)
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            raise TargetNotFound(f"target_col '{target_col}' has no finite values")
        if np.unique(finite).size < 2:
            raise TargetNotFound(
                f"target_col '{target_col}' is constant — zero entropy, so any "
                "information measure against it is 0 by construction")
        return [Target(name=target_col, kind="label", horizon_name="label",
                       horizon_bars=0, column=target_col, signed=_is_signed(v),
                       price_col=ctx.price_col)]

    prices = bars[ctx.price_col].to_numpy(dtype=np.float64, na_value=np.nan)
    out: list[Target] = []
    for h_name, h_bars in (ctx.horizons or {}).items():
        fwd = compute_forward_returns(prices, int(h_bars))
        out.append(Target(name=f"{FORWARD_RETURN_LABEL}_{h_name}", kind="forward_return",
                          horizon_name=h_name, horizon_bars=int(h_bars), column=None,
                          signed=_is_signed(fwd), price_col=ctx.price_col))
    return out


def feature_columns(bars: pd.DataFrame, candidates: list[str], target: Target) -> list[str]:
    """`candidates` minus everything that leaks `target`. Order preserved."""
    leak = target.leakage_columns(bars)
    return [c for c in candidates if c not in leak]
