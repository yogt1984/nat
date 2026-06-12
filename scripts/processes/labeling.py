"""
Triple-barrier labeling — the López de Prado event-based target transform.

For each bar t, three barriers are placed (Advances in Financial Machine
Learning, Ch. 3):

    upper   p_t * exp(+pt_mult * vol_t)     (profit-take)
    lower   p_t * exp(-sl_mult * vol_t)     (stop-loss)
    vertical t + max_holding_bars           (time-out)

where vol_t is the rolling realized volatility of 1-bar log returns over
`vol_window` bars computed from PAST data only. The first barrier touched
yields:

    tb_label    +1 (upper) / -1 (lower) / 0 (vertical)
    tb_ret      log(p_hit / p_t) — the return actually realized at the touch
    tb_hit_bars bars until the touch

Bars whose horizon extends past the end of the data, or whose volatility
estimate is not yet formed, are NaN — never silently labeled.

The output is a first-class evaluable target: chain it into ic_horizon or
ml_importance via `target_col="tb_label"` (`--score-with` in the runner) to
ask "which features predict barrier outcomes?" instead of raw returns —
labels carry the path dependence (stop-outs) that fixed-horizon returns
ignore.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .base import Finding, ProcessContext, ProcessResult, TransformProcess, make_run_id
from .registry import register


@register
class TripleBarrierProcess(TransformProcess):
    """Volatility-scaled triple-barrier labels (tb_label, tb_ret, tb_hit_bars)."""

    PARAMS = {
        "pt_mult": (2.0, "profit-take barrier in units of rolling vol"),
        "sl_mult": (1.0, "stop-loss barrier in units of rolling vol"),
        "max_holding_bars": (16, "vertical barrier (bars)"),
        "vol_window": (96, "rolling window (bars) for realized vol, past-only"),
    }

    def name(self) -> str:
        return "triple_barrier"

    def required_columns(self, available: list[str]) -> list[str]:
        return []  # consumes only the price column from the context

    def transform(
        self, bars: pd.DataFrame, ctx: ProcessContext,
    ) -> tuple[pd.DataFrame, ProcessResult]:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        pt = float(self.params["pt_mult"])
        sl = float(self.params["sl_mult"])
        hold = int(self.params["max_holding_bars"])
        vol_win = int(self.params["vol_window"])

        p = bars[ctx.price_col].to_numpy(dtype=np.float64, na_value=np.nan)
        n = len(p)

        # Past-only rolling vol of 1-bar log returns: vol[t] uses r[1..t]
        log_ret = pd.Series(np.concatenate([[np.nan], np.diff(np.log(p))]))
        vol = log_ret.rolling(vol_win, min_periods=max(vol_win // 2, 2)).std().to_numpy()

        label = np.full(n, np.nan)
        ret = np.full(n, np.nan)
        hit = np.full(n, np.nan)
        done = np.zeros(n, dtype=bool)

        upper = p * np.exp(pt * vol)
        lower = p * np.exp(-sl * vol)
        # Only bars with a formed vol estimate AND a full horizon inside the
        # data get labels — the tail stays NaN, never silently labeled
        eligible = np.isfinite(vol) & np.isfinite(p)
        eligible[max(n - hold, 0):] = False

        for k in range(1, hold + 1):
            m = n - k
            if m <= 0:
                break
            future = p[k:]
            up_hit = future >= upper[:m]
            dn_hit = future <= lower[:m]
            newly = eligible[:m] & ~done[:m] & (up_hit | dn_hit)
            if newly.any():
                # Upper wins bar-level ties (both barriers inside one bar is
                # unresolvable without intrabar data)
                lab_k = np.where(up_hit[:m], 1.0, -1.0)
                label[:m][newly] = lab_k[newly]
                ret[:m][newly] = np.log(future[newly] / p[:m][newly])
                hit[:m][newly] = k
                done[:m][newly] = True

        # Vertical barrier: eligible, untouched -> label 0 at t + hold
        vertical = eligible & ~done
        idx = np.flatnonzero(vertical)
        label[idx] = 0.0
        ret[idx] = np.log(p[idx + hold] / p[idx])
        hit[idx] = hold

        derived = pd.DataFrame({"tb_label": label, "tb_ret": ret, "tb_hit_bars": hit},
                               index=bars.index)
        for meta in ("bar_start", "symbol"):
            if meta in bars.columns:
                derived[meta] = bars[meta]

        labeled = label[np.isfinite(label)]
        n_lab = len(labeled)
        counts = {
            "+1": int((labeled == 1).sum()),
            "-1": int((labeled == -1).sum()),
            "0": int((labeled == 0).sum()),
        }
        mean_abs_ret_bps = {
            cls: round(float(np.nanmean(np.abs(ret[label == v]))) * 1e4, 2)
            if (label == v).any() else None
            for cls, v in (("+1", 1.0), ("-1", -1.0), ("0", 0.0))
        }
        result.features_tested = ["tb_label", "tb_ret", "tb_hit_bars"]
        result.findings = [Finding(
            feature="tb_label", horizon=f"{hold}bar", metric="barrier_touch_rate",
            value=round((counts["+1"] + counts["-1"]) / n_lab, 4) if n_lab else 0.0,
            extras={
                "counts": counts,
                "n_labeled": n_lab,
                "n_unlabeled_tail": int(n - n_lab),
                "mean_abs_ret_bps": mean_abs_ret_bps,
                "mean_hit_bars": round(float(np.nanmean(hit)), 2) if n_lab else None,
                "vol_window": vol_win, "pt_mult": pt, "sl_mult": sl,
            },
        )]
        return derived, result.finalize(time.time() - t0)
