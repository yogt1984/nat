"""PROC-4: `mi_stability` — is an edge durable, or a within-window mirage?

The IT daemon recomputes on a 600 s rolling buffer (`buffer_size = 6000` @ 100 ms), so it
answers *"is there MI right now"* and can never answer *"does `MI(f; r)` hold across days"*.
Every unit built on top of the process layer eventually needs the second answer: PROC-3 can
find a synergistic combo and PROC-1 can compile it, but neither knows whether it survives to
next week.

**One fold per calendar day, never pooled.** This is the load-bearing decision. Estimating MI
on a concatenated multi-day frame lets *between-day* structure masquerade as prediction: if a
feature's daily level and the daily return level both drift, their joint distribution shows
dependence that no single day contains. Days are therefore split on the calendar (from
`timestamp_ns` / `datetime` / `bar_start`, not on row counts, so unequal-length days stay
whole), each day is estimated independently, and `tests/test_mi_stability.py` plants exactly
that trap and asserts the process does not fall into it.

Per (feature, horizon) it reports the series `MI_d` — null-calibrated per day (PROC-12), so
"informative" means *above its own shuffle null on that day*, not above a fixed number — plus
`mean`, `cv`, `slope_per_day`, `frac_days_informative`, and a verdict:

    durable       informative on >= `min_frac_informative` of days AND |slope| small
    non_durable   scattered across days, or trending toward zero
    insufficient_days   fewer than `min_days` usable folds — said out loud, not averaged over

An edge alive on half the days averages to something publishable; the point of this unit is
that the average is the wrong statistic and the per-day series is the finding.

Targets come from the PROC-17 node (forward returns per horizon, or label mode with its own
leakage set). Real runs want >= 10 clean days (§7's binding constraint); the planted path runs
today.

Spec: `docs/specs/process_layer.md` §4.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from it_engine.estimators import ksg_mi
from it_engine.null_calibration import load_null_config, null_calibrate

from .base import (
    EvaluationProcess, Finding, ProcessContext, ProcessResult, make_run_id,
    partition_usable_columns,
)
from .registry import register
from .targets import TargetNotFound, feature_columns, resolve_targets

_PRICE_PREFIXES = ("raw_midprice", "raw_microprice")

#: Columns a day key can be derived from, in preference order.
_TIME_COLUMNS = ("date", "timestamp_ns", "timestamp", "datetime", "bar_start")


@register
class MIStabilityProcess(EvaluationProcess):
    """Per-day null-calibrated MI, reported as a series rather than an average."""

    PARAMS = {
        "features": (None, "feature name prefixes to score; None = all non-meta numeric"),
        "ksg_k": (5, "k for the KSG MI estimator"),
        "n_shuffles": (None, "permutation draws per day (default: it_engine.toml)"),
        "min_fold_obs": (200, "minimum jointly-valid rows for a day to be estimated"),
        "min_days": (3, "minimum usable days before a verdict is issued"),
        "min_frac_informative": (0.6, "fraction of days above null required for 'durable'"),
        "max_abs_slope": (0.05, "|bits/day| below which an edge counts as trendless"),
        "max_samples": (4000, "subsample cap per day for the estimator"),
        "seed": (0, "RNG seed for reproducible shuffles"),
        "target_col": (None, "label column replacing forward returns (PROC-17)"),
    }

    def name(self) -> str:
        return "mi_stability"

    # ── fold construction ────────────────────────────────────────────────────────
    @staticmethod
    def _day_key(bars: pd.DataFrame) -> Optional[np.ndarray]:
        """Calendar day per row, or None if the frame carries no usable time column."""
        for col in _TIME_COLUMNS:
            if col not in bars.columns:
                continue
            s = bars[col]
            if col == "timestamp_ns":
                return (s.to_numpy(dtype=np.int64) // 86_400_000_000_000).astype(np.int64)
            if col == "timestamp":
                v = s.to_numpy(dtype=np.float64)
                unit = 1e9 if np.nanmax(v) > 1e17 else 1.0     # ns vs seconds
                return (v / unit // 86_400).astype(np.int64)
            try:
                return pd.to_datetime(s).dt.floor("D").astype("int64").to_numpy()
            except Exception:                                  # pragma: no cover - defensive
                continue
        return None

    # ── the process ──────────────────────────────────────────────────────────────
    def evaluate(self, bars: pd.DataFrame, ctx: ProcessContext) -> ProcessResult:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        p = self.params

        days = self._day_key(bars)
        if days is None:
            return result.finalize(
                time.time() - t0,
                error="no time column to split days on — stability across days cannot be "
                      "measured from a frame that does not say which day a row is from")

        try:
            targets = resolve_targets(bars, ctx, p)
        except TargetNotFound as exc:
            return result.finalize(time.time() - t0, error=str(exc))

        cfg = load_null_config()
        n_shuffles = int(p["n_shuffles"] or cfg["n_shuffles"])
        z_thr, i_min = float(cfg["null_z_threshold"]), float(cfg["i_min"])
        k = int(p["ksg_k"])
        min_fold = int(p["min_fold_obs"])

        cand = feature_columns(bars, [
            c for c in self.required_columns(list(bars.columns))
            if not c.startswith(_PRICE_PREFIXES) and c != ctx.price_col
        ], targets[0])
        usable, skipped = partition_usable_columns(bars, cand, min_obs=min_fold)
        result.features_tested = usable
        result.features_skipped = skipped
        if not usable:
            return result.finalize(time.time() - t0, error="no usable features")

        uniq = np.unique(days)
        folds, folds_skipped = [], []
        for d in uniq:
            m = days == d
            if int(m.sum()) < min_fold:
                folds_skipped.append({"day": int(d), "n_rows": int(m.sum()),
                                      "reason": f"fold rows {int(m.sum())} < {min_fold}"})
                continue
            folds.append((int(d), m))

        for target in targets:
            y_full = target.values(bars)
            for feat in usable:
                x_full = bars[feat].to_numpy(dtype=np.float64, na_value=np.nan)
                per_day = []
                for d, m in folds:
                    rng = np.random.default_rng(int(p["seed"]) + int(d))   # per-day, seeded
                    valid = m & np.isfinite(x_full) & np.isfinite(y_full)
                    if int(valid.sum()) < min_fold:
                        continue
                    idx = np.flatnonzero(valid)
                    cap = int(p["max_samples"])
                    if len(idx) > cap:
                        idx = np.sort(rng.choice(idx, size=cap, replace=False))
                    nr = null_calibrate(lambda a, b: ksg_mi(a, b, k=k),
                                        _rank01(x_full[idx]), _rank01(y_full[idx]),
                                        n_shuffles=n_shuffles, rng=rng)
                    per_day.append({
                        "day": int(d), "n": int(len(idx)),
                        "bits_above_null": round(nr.bits_above_null, 6),
                        "z": round(nr.z, 3), "p": round(nr.p, 6),
                        "informative": bool(nr.informative(i_min=i_min, z_threshold=z_thr)),
                    })

                if not per_day:
                    continue
                result.findings.append(
                    self._summarise(feat, target, per_day, z_thr, len(folds)))

        result.finalize(time.time() - t0)
        result.summary.update({
            "n_days_total": int(len(uniq)),
            "n_days_used": len(folds),
            "folds_skipped": folds_skipped,
            "min_days": int(p["min_days"]),
            "target": targets[0].as_dict(),
        })
        return result

    # ── verdict ──────────────────────────────────────────────────────────────────
    def _summarise(self, feat: str, target, per_day: list[dict],
                   z_thr: float, n_folds: int) -> Finding:
        bits = np.array([d["bits_above_null"] for d in per_day], dtype=np.float64)
        flags = np.array([d["informative"] for d in per_day], dtype=bool)
        frac = float(flags.mean())
        mean = float(bits.mean())
        std = float(bits.std(ddof=1)) if len(bits) > 1 else 0.0
        cv = float(std / abs(mean)) if abs(mean) > 1e-12 else float("inf")
        slope = 0.0
        if len(bits) > 1:
            slope = float(np.polyfit(np.arange(len(bits), dtype=np.float64), bits, 1)[0])

        p = self.params
        if len(per_day) < int(p["min_days"]):
            verdict = "insufficient_days"
        elif frac >= float(p["min_frac_informative"]) and abs(slope) <= float(p["max_abs_slope"]):
            verdict = "durable"
        else:
            verdict = "non_durable"

        return Finding(
            feature=feat, horizon=target.horizon_name, metric="mi_stability_bits",
            value=round(mean, 6), threshold=z_thr,
            informative=bool(verdict == "durable"),
            extras={
                "verdict": verdict,
                "n_days": len(per_day),
                "frac_days_informative": round(frac, 4),
                "mean_bits_above_null": round(mean, 6),
                "std_bits": round(std, 6),
                "cv": round(cv, 4) if np.isfinite(cv) else None,
                "slope_per_day": round(slope, 6),
                "target": target.label_def,
                "gate": target.gate,
                "per_day": per_day,
            },
        )


def _rank01(v: np.ndarray) -> np.ndarray:
    """Copula (rank) transform — KSG's noise floor is smaller on ranks."""
    from scipy.stats import rankdata
    x = np.asarray(v, dtype=np.float64)
    return rankdata(x) / (len(x) + 1.0)
