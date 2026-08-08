"""A-2: walk-forward IC for a fitted composite — weights never see their own fold.

§5's hierarchical-combiner result (composite IC BTC .178 / ETH .248 / SOL .359) is the last
unrefuted capital-relevant claim in the record, and its own source flags the symptom:
**monotonically rising fold ICs**, noted as a "possible look-ahead/trend artifact". The
provenance explains it — `models/hierarchical_combiner/weights_BTC.json` carries
`training_date 2026-06-11` while the OOS window was 2026-06-08→10. The weights were fitted
*after* the period they were scored on.

That is not a subtle bias. An IC-weighted composite fitted on data containing its own
evaluation window will rank that window well by construction, and folds later in the sample
look progressively better because a larger share of the fit is behind them — exactly the
monotone rise §5 observed and could not explain.

This module is the honest version of that measurement:

  * weights are re-fitted **per fold, on rows strictly before it**, with an optional embargo
    so an overlapping forward return cannot bridge the boundary;
  * every fold reports its own IC, so the pooled mean is never the only statistic (§4.9: the
    binding failures were day-consistency and concentration, not the average);
  * the monotonicity of the fold-IC series is measured and returned, because that is the
    specific tell this exercise exists to check for.

It deliberately does **not** know about the combiner's layers. It scores an IC-weighted
composite of whatever columns it is given, so the same harness prices the full stack and its
ablations (L1 only, L2 only, L1×L2) on identical folds.

Tests: `tests/test_a2_walkforward.py` — the decisive one plants an edge that dies before the
holdout and asserts this path reports ~0 while an in-sample fit reports a large IC.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def fit_ic_weights(df: pd.DataFrame, features: list[str], target: str) -> dict[str, float]:
    """Spearman IC of each feature against `target`, used directly as its weight.

    The combiner's own scheme (`_ic_weighted_composite`): sign and magnitude both come from
    the measured relation, so a feature that flips sign in training flips in the composite.
    """
    weights: dict[str, float] = {}
    y = df[target].to_numpy(dtype=np.float64)
    for f in features:
        x = df[f].to_numpy(dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 20 or np.std(x[m]) == 0:
            weights[f] = 0.0
            continue
        ic = spearmanr(x[m], y[m]).statistic
        weights[f] = 0.0 if not np.isfinite(ic) else float(ic)
    return weights


def _composite(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Weight-normalised z-score composite, computed with the FOLD's own statistics.

    Standardising within the fold keeps the scoring self-contained: no training-set mean or
    variance leaks into the evaluation, and IC is rank-based so an affine shift is harmless
    anyway — it is the weights that must not leak, and they are passed in.
    """
    total = sum(abs(w) for w in weights.values()) or 1.0
    out = np.zeros(len(df), dtype=np.float64)
    for f, w in weights.items():
        x = df[f].to_numpy(dtype=np.float64)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (x - np.nanmean(x)) / sd
        out += (w / total) * np.nan_to_num(z, nan=0.0)
    return out


def fold_bounds(n: int, n_folds: int, min_train: int) -> list[tuple[int, int]]:
    """Contiguous evaluation folds tiling `[min_train, n)`.

    Contiguous, not random: the rows are a time series, and a shuffled split would let the
    fit see the future of its own fold through autocorrelation.
    """
    if n_folds < 1 or n <= min_train:
        return []
    edges = np.linspace(min_train, n, n_folds + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges, edges[1:]) if b > a]


def walk_forward_ic(df: pd.DataFrame, features: list[str], target: str,
                    n_folds: int = 6, min_train: Optional[int] = None,
                    embargo: int = 0) -> dict:
    """Per-fold IC of an IC-weighted composite, refitted before each fold.

    Returns the per-fold series plus the aggregates that decide durability, and the
    fold-IC trend/monotonicity — the §5 tell.
    """
    missing = [f for f in features if f not in df.columns] + \
              ([target] if target not in df.columns else [])
    if missing:
        return {"error": f"missing columns: {missing}", "folds": [], "n_folds_used": 0,
                "pooled_ic": None, "n_obs": 0}

    cols = list(dict.fromkeys(list(features) + [target]))
    data = df[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    n = len(data)
    if min_train is None:
        min_train = max(100, n // 4)

    bounds = fold_bounds(n, n_folds, min_train)
    if not bounds:
        return {"error": f"not enough rows ({n}) for {n_folds} folds after "
                         f"min_train={min_train}",
                "folds": [], "n_folds_used": 0, "pooled_ic": None, "n_obs": n}

    y_all = data[target].to_numpy(dtype=np.float64)
    folds = []
    for start, end in bounds:
        train_end = max(0, start - int(embargo))
        if train_end < 20:
            continue
        train = data.iloc[:train_end]
        weights = fit_ic_weights(train, features, target)
        test = data.iloc[start:end]
        if len(test) < 10:
            continue
        sig = _composite(test, weights)
        y = y_all[start:end]
        m = np.isfinite(sig) & np.isfinite(y)
        if m.sum() < 10 or np.std(sig[m]) == 0:
            continue
        ic = spearmanr(sig[m], y[m]).statistic
        folds.append({"start": int(start), "end": int(end), "train_end": int(train_end),
                      "n": int(m.sum()), "ic": float(ic) if np.isfinite(ic) else 0.0,
                      "weights": weights})

    if not folds:
        return {"error": "no fold could be evaluated", "folds": [], "n_folds_used": 0,
                "pooled_ic": None, "n_obs": n}

    ics = np.array([f["ic"] for f in folds], dtype=np.float64)
    trend = float(np.polyfit(np.arange(len(ics), dtype=np.float64), ics, 1)[0]) \
        if len(ics) > 1 else 0.0
    total = float(np.sum(np.abs(ics))) or 1.0
    return {
        "error": None,
        "n_obs": n,
        "n_folds_used": len(folds),
        "pooled_ic": float(ics.mean()),
        "ic_std": float(ics.std(ddof=1)) if len(ics) > 1 else 0.0,
        "positive_fold_share": float((ics > 0).mean()),
        "max_fold_share": float(np.max(np.abs(ics)) / total),
        # The §5 tell: fold ICs that climb monotonically are the signature of a fit that
        # keeps absorbing more of its own evaluation window.
        "fold_ic_trend": trend,
        "fold_ic_monotone": bool(np.all(np.diff(ics) > 0) or np.all(np.diff(ics) < 0)),
        "folds": folds,
    }
