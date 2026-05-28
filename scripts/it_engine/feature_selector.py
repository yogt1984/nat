"""
Greedy forward feature selection by conditional MI gain.

Algorithm:
  1. f* = argmax_f  I(f; r_k)         — best single feature
  2. f_next = argmax_f  I(f; r_k | S) — largest CMI gain given selected set S
  3. Stop when marginal gain < I_min(k) or max features reached
  4. Output: ordered feature set with cumulative MI

This is a tractable alternative to full Partial Information Decomposition
(NP-hard for >3 variables).
"""

import logging
import numpy as np
from typing import Optional

from .estimators import ksg_mi, cmi, min_info_bits

log = logging.getLogger(__name__)


def greedy_select(
    features: dict[str, np.ndarray],
    returns: np.ndarray,
    fee_rt_bps: float,
    sigma_r_bps: float,
    max_features: int = 10,
    k: int = 5,
    excluded: Optional[set[str]] = None,
) -> list[dict]:
    """Greedy forward selection of features by information gain.

    Parameters
    ----------
    features     : {name: array} mapping feature names to 1-D arrays
    returns      : 1-D array of forward returns (same length as features)
    fee_rt_bps   : round-trip transaction fee in bps
    sigma_r_bps  : return standard deviation in bps
    max_features : maximum features to select
    k            : KSG nearest neighbors
    excluded     : feature names to skip (e.g. already in portfolio)

    Returns
    -------
    List of dicts, each with:
        name          : feature name
        mi            : marginal MI I(f; r)
        cmi_gain      : conditional MI gain at this step
        cumulative_mi : total MI after including this feature
        cost_viable   : whether cumulative MI > I_min
    """
    excluded = excluded or set()
    r = np.asarray(returns, dtype=np.float64)
    # Compute sample kurtosis for fat-tail correction of cost threshold
    from scipy.stats import kurtosis as _kurtosis
    kurt = float(_kurtosis(r[np.isfinite(r)], fisher=False)) if np.isfinite(r).sum() > 30 else 3.0
    i_min = min_info_bits(fee_rt_bps, sigma_r_bps, kurtosis=kurt)

    candidates = {
        name: arr for name, arr in features.items()
        if name not in excluded and len(arr) == len(r)
    }

    if not candidates:
        return []

    selected = []
    selected_arrays = []
    cumulative = 0.0

    for step in range(max_features):
        best_name = None
        best_gain = -np.inf

        for name, arr in candidates.items():
            if name in {s["name"] for s in selected}:
                continue

            if not selected_arrays:
                # First feature: plain MI
                gain = ksg_mi(arr, r, k=k)
            else:
                # Conditional MI given already-selected features
                z = np.column_stack(selected_arrays)
                gain = cmi(arr, r, z, k=k)

            if gain > best_gain:
                best_gain = gain
                best_name = name

        if best_name is None or best_gain <= 0:
            break

        # Check stopping criterion
        if step > 0 and best_gain < i_min:
            log.info(
                "Stopping at step %d: gain %.6f bits < I_min %.6f bits",
                step, best_gain, i_min,
            )
            break

        cumulative += best_gain
        selected.append({
            "name": best_name,
            "mi": float(ksg_mi(candidates[best_name], r, k=k)),
            "cmi_gain": float(best_gain),
            "cumulative_mi": float(cumulative),
            "cost_viable": cumulative > i_min,
        })
        selected_arrays.append(candidates[best_name])

        log.info(
            "Step %d: %s  gain=%.6f  cumulative=%.6f  viable=%s",
            step, best_name, best_gain, cumulative, cumulative > i_min,
        )

    return selected
