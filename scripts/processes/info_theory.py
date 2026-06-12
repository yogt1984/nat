"""
Information-theoretic processes — model-free dependence measures.

MutualInfoProcess ("mi_ksg")
    KSG k-NN mutual information I(feature; forward return) in bits
    (Kraskov et al. 2004), per feature x horizon, with the cost-viability
    gate from the IT engine: informative iff

        MI >= I_min = -0.5 * log2(1 - (fee_RT / sigma_r)^2) * (kurtosis/3)

    i.e. the signal carries at least the information needed to overcome
    round-trip costs at that horizon (fat-tail corrected). Optional
    `conditioning` columns switch on CMI I(f; r | Z) and interaction
    information (synergy/redundancy) in extras.

TransferEntropyProcess ("transfer_entropy")
    Directed information flow TE(feature -> 1-bar return) via linear
    (Gaussian AR) or KSG estimator, with the reverse direction as a
    directionality control: informative iff TE_fwd >= I_min AND
    TE_fwd > TE_rev.

Reuses ONLY `it_engine.estimators` (numpy/scipy-pure) — never the IT-engine
daemon/state/config, which drag in Redis and symbol config.

All inputs to the KSG estimators are rank (copula) transformed first:
MI/TE are invariant under monotone marginal transforms, but the raw KSG
joint Chebyshev metric is NOT scale-invariant — features at scale ~1 vs
returns at scale ~1e-3 let the feature axis dominate the k-NN radius,
producing a spurious ~0.07-bit floor for pure noise while masking real
dependence (verified empirically on planted data). Ranking fixes the
geometry and handles fat tails for free. The linear TE path stays raw
(OLS is scale-invariant and ranking would change its Gaussian semantics).

KSG estimators build kd-trees with per-point ball queries — the slow path.
`max_samples` stride-subsamples larger inputs (the IT engine's own guard).
"""

from __future__ import annotations

import time

import numpy as np
from scipy import stats
from scipy.stats import rankdata

from it_engine.estimators import cmi, interaction_info, ksg_mi, ksg_te, linear_te, min_info_bits

from alpha.screener import compute_forward_returns

from .base import EvaluationProcess, Finding, ProcessContext, ProcessResult, make_run_id, partition_usable_columns
from .registry import register

_PRICE_PREFIXES = ("raw_midprice", "raw_microprice")


def _rank01(a: np.ndarray) -> np.ndarray:
    """Copula transform: ranks scaled to (0, 1]. See module docstring."""
    return rankdata(a) / len(a)


def _fee_rt_bps(ctx: ProcessContext, override) -> float:
    if override is not None:
        return float(override)
    return float(ctx.costs.get("hyperliquid", {}).get("round_trip_taker_bps", 7.0))


def _subsample(*arrays: np.ndarray, max_samples: int) -> tuple[np.ndarray, ...]:
    """Joint finite-mask then stride-subsample to <= max_samples points.

    Valid for order-independent estimators (MI/CMI) ONLY — striding destroys
    the lag structure that transfer entropy measures (use _joint_tail there).
    """
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    out = [a[mask] for a in arrays]
    n = len(out[0])
    if n > max_samples:
        stride = int(np.ceil(n / max_samples))
        out = [a[::stride] for a in out]
    return tuple(out)


def _joint_tail(*arrays: np.ndarray, max_samples: int) -> tuple[np.ndarray, ...]:
    """Joint finite-mask then take the most recent contiguous window.

    Preserves the lag-1 temporal coupling that TE estimates (assumes NaN
    gaps are sparse — bar-level data after partition_usable_columns).
    """
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    out = [a[mask] for a in arrays]
    if len(out[0]) > max_samples:
        out = [a[-max_samples:] for a in out]
    return tuple(out)


@register
class MutualInfoProcess(EvaluationProcess):
    """KSG mutual information vs forward returns with cost-viability gate."""

    PARAMS = {
        "features": (None, "list of name prefixes to score; None = all non-meta numeric"),
        "ksg_k": (5, "k nearest neighbors for the KSG estimator"),
        "max_samples": (20000, "stride-subsample guard — KSG is O(N log N) with big constants"),
        "min_obs": (200, "minimum joint-valid observations per (feature, horizon)"),
        "conditioning": ([], "column names Z for CMI I(f;r|Z) + interaction info"),
        "kurtosis_correction": (True, "scale I_min by realized kurtosis / 3 (fat tails)"),
        "fee_rt_bps": (None, "round-trip fee override; None = ctx.costs hyperliquid"),
    }

    def name(self) -> str:
        return "mi_ksg"

    def evaluate(self, bars, ctx: ProcessContext) -> ProcessResult:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        min_obs = int(self.params["min_obs"])
        max_samples = int(self.params["max_samples"])
        k = int(self.params["ksg_k"])
        fee_rt = _fee_rt_bps(ctx, self.params["fee_rt_bps"])
        cond_cols = [c for c in (self.params["conditioning"] or []) if c in bars.columns]

        cols = [
            c for c in self.required_columns(list(bars.columns))
            if not c.startswith(_PRICE_PREFIXES) and c != ctx.price_col
            and c not in cond_cols
        ]
        usable, skipped = partition_usable_columns(bars, cols, min_obs=min_obs)
        result.features_tested = usable
        result.features_skipped = skipped

        prices = bars[ctx.price_col].to_numpy(dtype=np.float64, na_value=np.nan)
        z_mat = (
            np.column_stack([
                bars[c].to_numpy(dtype=np.float64, na_value=np.nan) for c in cond_cols
            ]) if cond_cols else None
        )

        for h_name, h_bars in ctx.horizons.items():
            fr = compute_forward_returns(prices, h_bars)
            finite_fr = fr[np.isfinite(fr)]
            if len(finite_fr) < min_obs:
                continue
            sigma_r_bps = float(np.std(finite_fr)) * 1e4
            kurt = (
                float(stats.kurtosis(finite_fr, fisher=False))
                if self.params["kurtosis_correction"] else 3.0
            )
            i_min = min_info_bits(fee_rt, sigma_r_bps, kurtosis=kurt)

            for feat in usable:
                x = bars[feat].to_numpy(dtype=np.float64, na_value=np.nan)
                if z_mat is not None:
                    xs, rs, *zs = _subsample(
                        x, fr, *(z_mat[:, j] for j in range(z_mat.shape[1])),
                        max_samples=max_samples,
                    )
                    z_sub = np.column_stack(zs)
                else:
                    xs, rs = _subsample(x, fr, max_samples=max_samples)
                    z_sub = None
                if len(xs) < min_obs:
                    continue

                xs, rs = _rank01(xs), _rank01(rs)
                if z_sub is not None:
                    z_sub = np.column_stack([
                        _rank01(z_sub[:, j]) for j in range(z_sub.shape[1])
                    ])
                mi_bits = ksg_mi(xs, rs, k=k)
                extras = {
                    "i_min_bits": round(i_min, 6) if np.isfinite(i_min) else None,
                    "sigma_r_bps": round(sigma_r_bps, 2),
                    "kurtosis": round(kurt, 2),
                    "fee_rt_bps": fee_rt,
                    "n_samples": len(xs),
                }
                if z_sub is not None:
                    cmi_bits = cmi(xs, rs, z_sub, k=k)
                    extras["cmi_bits"] = round(cmi_bits, 6)
                    extras["interaction_info_bits"] = round(cmi_bits - mi_bits, 6)
                    extras["conditioning"] = cond_cols

                result.findings.append(Finding(
                    feature=feat, horizon=h_name, metric="mi_bits",
                    value=round(mi_bits, 6),
                    threshold=round(i_min, 6) if np.isfinite(i_min) else None,
                    informative=bool(np.isfinite(i_min) and mi_bits >= i_min),
                    extras=extras,
                ))

        return result.finalize(time.time() - t0)


@register
class TransferEntropyProcess(EvaluationProcess):
    """Directed TE(feature -> 1-bar return) with reverse-direction control."""

    PARAMS = {
        "features": (None, "list of name prefixes to score; None = all non-meta numeric"),
        "te_method": ("linear", "'linear' (Gaussian AR, fast, recommended) or 'ksg' "
                                "(slow; it_engine's 4-term entropy CMI is biased low on "
                                "autocorrelated targets and often clamps to 0 — exploratory)"),
        "lag": (1, "lags of the source included"),
        "order": (1, "AR order of the target history conditioned on"),
        "ksg_k": (5, "k for the KSG variant"),
        "max_samples": (20000, "stride-subsample guard for the KSG variant"),
        "min_obs": (200, "minimum joint-valid observations"),
        "kurtosis_correction": (True, "scale I_min by realized kurtosis / 3"),
        "fee_rt_bps": (None, "round-trip fee override; None = ctx.costs hyperliquid"),
    }

    def name(self) -> str:
        return "transfer_entropy"

    def evaluate(self, bars, ctx: ProcessContext) -> ProcessResult:
        t0 = time.time()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe=ctx.timeframe, params=dict(self.params),
        )
        min_obs = int(self.params["min_obs"])
        method = str(self.params["te_method"])
        lag, order = int(self.params["lag"]), int(self.params["order"])
        fee_rt = _fee_rt_bps(ctx, self.params["fee_rt_bps"])

        cols = [
            c for c in self.required_columns(list(bars.columns))
            if not c.startswith(_PRICE_PREFIXES) and c != ctx.price_col
        ]
        usable, skipped = partition_usable_columns(bars, cols, min_obs=min_obs)
        result.features_tested = usable
        result.features_skipped = skipped

        prices = bars[ctx.price_col].to_numpy(dtype=np.float64, na_value=np.nan)
        ret = np.full(len(prices), np.nan)
        ret[1:] = np.diff(np.log(prices))
        finite_ret = ret[np.isfinite(ret)]
        if len(finite_ret) < min_obs:
            return result.finalize(time.time() - t0, error="insufficient return data")
        sigma_r_bps = float(np.std(finite_ret)) * 1e4
        kurt = (
            float(stats.kurtosis(finite_ret, fisher=False))
            if self.params["kurtosis_correction"] else 3.0
        )
        i_min = min_info_bits(fee_rt, sigma_r_bps, kurtosis=kurt)

        if method == "ksg":
            def te(src, tgt):
                # Rank transform per marginal (see module docstring); TE is
                # invariant, the joint-space geometry is not.
                return ksg_te(_rank01(src), _rank01(tgt),
                              lag=lag, order=order, k=int(self.params["ksg_k"]))
        else:
            def te(src, tgt):
                return linear_te(src, tgt, lag=lag, order=order)

        for feat in usable:
            x = bars[feat].to_numpy(dtype=np.float64, na_value=np.nan)
            xs, rs = _joint_tail(x, ret, max_samples=int(self.params["max_samples"]))
            if len(xs) < min_obs:
                continue
            te_fwd = te(xs, rs)
            te_rev = te(rs, xs)
            result.findings.append(Finding(
                feature=feat, horizon="1bar", metric="te_bits",
                value=round(te_fwd, 6),
                threshold=round(i_min, 6) if np.isfinite(i_min) else None,
                informative=bool(
                    np.isfinite(i_min) and te_fwd >= i_min and te_fwd > te_rev
                ),
                extras={
                    "te_reverse_bits": round(te_rev, 6),
                    "directionality_ratio": round(te_fwd / te_rev, 3) if te_rev > 1e-12 else None,
                    "i_min_bits": round(i_min, 6) if np.isfinite(i_min) else None,
                    "method": method, "lag": lag, "order": order,
                    "n_samples": len(xs),
                },
            ))

        return result.finalize(time.time() - t0)
