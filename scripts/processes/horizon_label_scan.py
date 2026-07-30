"""PROC-7: horizon_label_scan — the horizon/label MI-surface meta-process.

Nobody had assembled the pieces into a scan over *what to predict × how far × in which
regime*. This meta-process runs the grid:

    for h in horizons, g = (pt_mult, sl_mult) in geometries:
        triple_barrier(h, g)                        -> tb_label        (PROC-5 substrate)
        conditional_predictability(f, tb_label, Z)  -> per-regime MI   (PROC-6, null-gated
                                                                        per PROC-12)

and assembles the MI **surface** over (feature, h, g, regime bucket). The whole surface is
then Benjamini–Hochberg corrected (PROC-13), so the argmax — the best (target, horizon,
regime) triple — is only ever surfaced WITH its BH q-value. The per-horizon profile answers
"which horizons are predictable at all" (the measured MF-vs-macro split).

Grid size is bounded by `max_cells`; anything truncated is logged and recorded in the
summary — no silent caps. Spec: docs/specs/process_layer.md §7. Deps: PROC-5/6/12/13.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from it_engine.estimators import ksg_mi
from it_engine.null_calibration import DEFAULT_NULL_Z_THRESHOLD, null_calibrate

from .base import (
    EvaluationProcess,
    Finding,
    ProcessContext,
    ProcessResult,
    make_run_id,
    partition_usable_columns,
)
from .conditional_predictability import _rank01, conditional_predictability
from .fdr import DEFAULT_FDR_ALPHA, apply_process_fdr
from .labeling import TripleBarrierProcess
from .registry import register

log = logging.getLogger(__name__)

_PRICE_PREFIXES = ("raw_midprice", "raw_microprice")


@register
class HorizonLabelScanProcess(EvaluationProcess):
    """MI surface over (horizon × barrier geometry × regime): what to predict, how far,
    and in which regime (PROC-7)."""

    PARAMS = {
        "features": (None, "feature name prefixes to score; None = all non-meta numeric"),
        "conditioning": ([], "regime columns Z; empty = unconditional (one cell per (h,g))"),
        "horizons": ([4, 16, 48], "vertical-barrier horizons in bars (max_holding_bars)"),
        "geometries": ([[2.0, 1.0], [1.0, 1.0]], "barrier shapes [pt_mult, sl_mult]"),
        "vol_window": (96, "rolling vol window (bars) for the barrier scale, past-only"),
        "n_buckets": (4, "regime quantile buckets per conditioning variable"),
        "ksg_k": (5, "k for the KSG MI estimator"),
        "n_shuffles": (100, "permutation-null draws per cell (PROC-12)"),
        "null_z_threshold": (DEFAULT_NULL_Z_THRESHOLD, "z >= this to call a cell informative"),
        "min_bucket_obs": (200, "minimum labelled rows per cell to estimate"),
        "fdr_alpha": (DEFAULT_FDR_ALPHA, "BH q over the whole surface (PROC-13)"),
        "max_cells": (5000, "hard cap on surface cells; excess features dropped + LOGGED"),
        "seed": (0, "RNG seed for reproducible shuffles"),
    }

    def name(self) -> str:
        return "horizon_label_scan"

    def evaluate(self, bars, ctx: ProcessContext) -> ProcessResult:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        p = self.params
        horizons = [int(h) for h in p["horizons"]]
        geometries = [tuple(map(float, g)) for g in p["geometries"]]
        if not horizons or any(h < 1 for h in horizons):
            raise ValueError(f"horizons must be positive ints, got {p['horizons']}")
        if not geometries or any(len(g) != 2 for g in geometries):
            raise ValueError(f"geometries must be [pt_mult, sl_mult] pairs, got {p['geometries']}")
        n_buckets = int(p["n_buckets"])
        min_obs = int(p["min_bucket_obs"])
        n_shuffles = int(p["n_shuffles"])
        k = int(p["ksg_k"])
        z_thr = float(p["null_z_threshold"])

        declared = list(p["conditioning"] or [])
        cond_cols = [c for c in declared if c in bars.columns]
        if declared and not cond_cols:
            return result.finalize(
                time.time() - t0,
                error=f"no conditioning columns present: {declared}",
            )

        cols = [
            c for c in self.required_columns(list(bars.columns))
            if not c.startswith(_PRICE_PREFIXES) and c != ctx.price_col
            and c not in cond_cols and not c.startswith("tb_")
        ]
        usable, skipped = partition_usable_columns(bars, cols, min_obs=min_obs)

        # Grid bound: drop excess FEATURES (the largest multiplier), never silently.
        buckets_per = n_buckets * len(cond_cols) if cond_cols else 1
        cells_per_feature = len(horizons) * len(geometries) * buckets_per
        max_features = max(1, int(p["max_cells"]) // cells_per_feature)
        truncated: list[str] = []
        if len(usable) > max_features:
            truncated = usable[max_features:]
            usable = usable[:max_features]
            log.warning(
                "horizon_label_scan: grid cap max_cells=%s -> dropping %d/%d features: %s",
                p["max_cells"], len(truncated), len(truncated) + len(usable), truncated,
            )
        result.features_tested = usable
        result.features_skipped = skipped

        rng = np.random.default_rng(int(p["seed"]))

        for h in horizons:
            for pt, sl in geometries:
                tb = TripleBarrierProcess(
                    pt_mult=pt, sl_mult=sl, max_holding_bars=h,
                    vol_window=int(p["vol_window"]),
                )
                derived, _ = tb.transform(bars, ctx)
                label = derived["tb_label"].to_numpy(dtype=np.float64, na_value=np.nan)
                h_name = f"{h}bar"

                for feat in usable:
                    x = bars[feat].to_numpy(dtype=np.float64, na_value=np.nan)
                    cells = []  # (zcol, bucket, z_range, n, NullResult)
                    if cond_cols:
                        for zcol in cond_cols:
                            zvals = bars[zcol].to_numpy(dtype=np.float64, na_value=np.nan)
                            buckets, _am = conditional_predictability(
                                x, label, zvals,
                                n_buckets=n_buckets, k=k, n_shuffles=n_shuffles,
                                min_bucket_obs=min_obs, rng=rng,
                            )
                            cells += [
                                (zcol, b["bucket"], b["z_range"], b["n"], b["result"])
                                for b in buckets if b["result"] is not None
                            ]
                    else:
                        finite = np.isfinite(x) & np.isfinite(label)
                        xf, lf = x[finite], label[finite]
                        if len(xf) >= min_obs and len(np.unique(lf)) >= 2:
                            nr = null_calibrate(
                                lambda a, b: ksg_mi(a, b, k=k),
                                _rank01(xf), _rank01(lf),
                                n_shuffles=n_shuffles, rng=rng,
                            )
                            cells.append((None, None, None, len(xf), nr))

                    for zcol, bucket, z_range, n_cell, nr in cells:
                        result.findings.append(Finding(
                            feature=feat, horizon=h_name, metric="cond_mi_bits",
                            value=round(nr.bits_above_null, 6),
                            p_value=round(nr.p, 6),
                            threshold=z_thr,
                            informative=bool(nr.informative(z_threshold=z_thr)),
                            extras={
                                "pt_mult": pt, "sl_mult": sl,
                                "conditioning": zcol,
                                "bucket": bucket, "z_range": z_range,
                                "n": n_cell,
                                "raw_bits": round(nr.raw_bits, 6),
                                "z": round(nr.z, 3),
                            },
                        ))

        result.finalize(time.time() - t0)
        # PROC-13 over the whole surface: annotate q, tighten informative, and build the
        # surface summary so the argmax is never reported without its correction. (The
        # runner re-applies BH on save — idempotent: same family, same p-values.)
        rep = apply_process_fdr(result, alpha=float(p["fdr_alpha"]))
        profile: dict[str, float] = {}
        for f in result.findings:
            profile[f.horizon] = max(profile.get(f.horizon, -np.inf), f.value)
        result.summary["n_informative"] = sum(1 for f in result.findings if f.informative)
        result.summary["surface"] = {
            "argmax": rep.argmax,
            "n_cells": rep.n_cells,
            "n_discoveries": rep.n_discoveries,
            "fdr_alpha": rep.alpha,
            "horizon_profile": profile,
            "features_truncated": truncated,
            "grid": {
                "horizons": horizons,
                "geometries": [list(g) for g in geometries],
                "n_buckets": n_buckets if cond_cols else None,
                "conditioning": cond_cols,
            },
        }
        return result
