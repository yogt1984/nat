"""PROC-6: conditional_predictability — MI(feature; label | Z=z) as a function of z.

The genuinely-absent concept in NAT's IT layer. Not the z-AVERAGED CMI (a single number),
but conditional predictability as a FUNCTION of the conditioning variable: partition Z into
quantile buckets and measure MI(feature; label) WITHIN each bucket. The bucket where MI
spikes (argmax) is the tradeable regime — where the edge actually lives.

Every per-bucket MI is null-calibrated (PROC-12): reported as bits-above-null / z / p, so a
regime counts as real only if it clears the shuffle-null z-threshold, never raw bits.

Reuses it_engine.estimators.ksg_mi with the copula/rank transform (KSG has a ~0.07-bit
floor otherwise) and it_engine.null_calibration (the PROC-12 gate).
Spec: docs/specs/process_layer.md §6.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import rankdata

from it_engine.estimators import ksg_mi
from it_engine.null_calibration import (
    DEFAULT_I_MIN,
    DEFAULT_NULL_Z_THRESHOLD,
    null_calibrate,
)

from alpha.screener import compute_forward_returns

from .base import (
    EvaluationProcess,
    Finding,
    ProcessContext,
    ProcessResult,
    make_run_id,
    partition_usable_columns,
)
from .registry import register

_PRICE_PREFIXES = ("raw_midprice", "raw_microprice")


def _rank01(a: np.ndarray) -> np.ndarray:
    """Copula transform: ranks scaled to (0, 1] (KSG floor fix — see info_theory.py)."""
    return rankdata(a) / len(a)


def conditional_predictability(
    feature: np.ndarray,
    label: np.ndarray,
    z: np.ndarray,
    *,
    n_buckets: int = 4,
    k: int = 5,
    n_shuffles: int = 100,
    min_bucket_obs: int = 50,
    rng=None,
):
    """MI(feature; label | Z=z) per Z-quantile bucket, each null-calibrated (PROC-12).

    Returns (buckets, argmax_idx):
      buckets     — list of {bucket, n, z_range, result: NullResult|None} per Z-bucket,
      argmax_idx  — bucket with the largest bits-above-null (the candidate tradeable
                    regime), or None if no bucket had enough data.
    """
    feature = np.asarray(feature, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(feature) & np.isfinite(label) & np.isfinite(z)
    f, lab, zz = feature[finite], label[finite], z[finite]

    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    edges = None
    if len(zz) >= n_buckets * min_bucket_obs and len(np.unique(zz)) >= n_buckets:
        edges = np.quantile(zz, np.linspace(0.0, 1.0, n_buckets + 1))
        edges[-1] = np.nextafter(edges[-1], np.inf)  # include the max value

    buckets = []
    for b in range(n_buckets):
        if edges is None:
            buckets.append({"bucket": b, "n": 0, "z_range": None, "result": None})
            continue
        mask = (zz >= edges[b]) & (zz < edges[b + 1])
        nb = int(mask.sum())
        z_range = (float(edges[b]), float(edges[b + 1]))
        if nb < min_bucket_obs:
            buckets.append({"bucket": b, "n": nb, "z_range": z_range, "result": None})
            continue
        fb, lb = _rank01(f[mask]), _rank01(lab[mask])
        nr = null_calibrate(
            lambda a, c: ksg_mi(a, c, k=k), fb, lb, n_shuffles=n_shuffles, rng=gen
        )
        buckets.append({"bucket": b, "n": nb, "z_range": z_range, "result": nr})

    scored = [(bb["bucket"], bb["result"].bits_above_null) for bb in buckets if bb["result"]]
    argmax = max(scored, key=lambda t: t[1])[0] if scored else None
    return buckets, argmax


@register
class ConditionalPredictabilityProcess(EvaluationProcess):
    """Conditional predictability MI(f; label | Z=z) as a function of z (PROC-6)."""

    PARAMS = {
        "features": (None, "feature name prefixes to score; None = all non-meta numeric"),
        "conditioning": ([], "column names Z to condition on (the regime variables)"),
        "n_buckets": (4, "quantile buckets per conditioning variable"),
        "ksg_k": (5, "k for the KSG MI estimator"),
        "n_shuffles": (100, "permutation-null draws per bucket (PROC-12)"),
        "null_z_threshold": (DEFAULT_NULL_Z_THRESHOLD, "z >= this to call a bucket informative"),
        "i_min": (DEFAULT_I_MIN, "minimum bits-above-null (effect-size gate)"),
        "min_bucket_obs": (200, "minimum rows per bucket to estimate"),
        "seed": (0, "RNG seed for reproducible shuffles"),
    }

    def name(self) -> str:
        return "conditional_predictability"

    def evaluate(self, bars, ctx: ProcessContext) -> ProcessResult:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        p = self.params
        cond_cols = [c for c in (p["conditioning"] or []) if c in bars.columns]
        if not cond_cols:
            return result.finalize(time.time() - t0, error="no conditioning columns present")

        cols = [
            c for c in self.required_columns(list(bars.columns))
            if not c.startswith(_PRICE_PREFIXES) and c != ctx.price_col and c not in cond_cols
        ]
        usable, skipped = partition_usable_columns(bars, cols, min_obs=int(p["min_bucket_obs"]))
        result.features_tested = usable
        result.features_skipped = skipped

        prices = bars[ctx.price_col].to_numpy(dtype=np.float64, na_value=np.nan)
        gen = np.random.default_rng(int(p["seed"]))

        for h_name, h_bars in ctx.horizons.items():
            fr = compute_forward_returns(prices, h_bars)
            for zcol in cond_cols:
                zvals = bars[zcol].to_numpy(dtype=np.float64, na_value=np.nan)
                for feat in usable:
                    x = bars[feat].to_numpy(dtype=np.float64, na_value=np.nan)
                    buckets, argmax = conditional_predictability(
                        x, fr, zvals,
                        n_buckets=int(p["n_buckets"]), k=int(p["ksg_k"]),
                        n_shuffles=int(p["n_shuffles"]),
                        min_bucket_obs=int(p["min_bucket_obs"]), rng=gen,
                    )
                    for bb in buckets:
                        nr = bb["result"]
                        if nr is None:
                            continue
                        info = nr.informative(
                            i_min=float(p["i_min"]),
                            z_threshold=float(p["null_z_threshold"]),
                        )
                        result.findings.append(Finding(
                            feature=feat, horizon=h_name, metric="cond_mi_bits",
                            value=round(nr.bits_above_null, 6),
                            p_value=round(nr.p, 6),
                            threshold=float(p["null_z_threshold"]),
                            informative=bool(info),
                            extras={
                                "conditioning": zcol,
                                "bucket": bb["bucket"],
                                "n_buckets": int(p["n_buckets"]),
                                "z_range": bb["z_range"],
                                "n": bb["n"],
                                "raw_bits": round(nr.raw_bits, 6),
                                "z": round(nr.z, 3),
                                "is_argmax_regime": bb["bucket"] == argmax,
                            },
                        ))

        return result.finalize(time.time() - t0)
