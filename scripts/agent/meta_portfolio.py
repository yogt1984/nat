"""Portfolio assembly and risk parity weighting for the meta-agent.

Pure functions for:
- Inverse-volatility (risk parity) weight computation
- Portfolio metric aggregation (weighted IC, effective N)
- Redundant signal filtering from cross-agent correlation
"""

from __future__ import annotations

import math
from typing import Optional


def compute_risk_parity_weights(signals: list[dict]) -> list[float]:
    """Inverse-volatility weighting using IC history variance.

    For each signal, volatility = std(ic_history). Signals with lower
    IC variance (more stable alpha) get higher weight.

    Falls back to equal weight if ic_history is missing or empty.

    Args:
        signals: list of signal dicts, each may contain 'ic_history' (list[float])

    Returns:
        list of weights summing to 1.0, same length as signals
    """
    if not signals:
        return []

    n = len(signals)
    vols = []
    has_history = False

    for sig in signals:
        ic_hist = sig.get("ic_history", [])
        if len(ic_hist) >= 2:
            has_history = True
            mean_ic = sum(ic_hist) / len(ic_hist)
            variance = sum((x - mean_ic) ** 2 for x in ic_hist) / (len(ic_hist) - 1)
            vol = math.sqrt(variance)
            vols.append(max(vol, 0.01))  # floor to prevent division by zero
        else:
            vols.append(None)

    # Fallback: equal weight if no signal has IC history
    if not has_history:
        w = 1.0 / n
        return [w] * n

    # Fill missing vols with median of known vols
    known_vols = [v for v in vols if v is not None]
    median_vol = sorted(known_vols)[len(known_vols) // 2]
    vols = [v if v is not None else median_vol for v in vols]

    # Inverse-vol weights
    inv_vols = [1.0 / v for v in vols]
    total = sum(inv_vols)
    return [iv / total for iv in inv_vols]


def compute_portfolio_metrics(
    signals: list[dict],
    weights: list[float],
) -> dict:
    """Compute portfolio-level metrics from weighted signals.

    Args:
        signals: list of signal dicts with 'expected_ic' field
        weights: list of portfolio weights (same length as signals)

    Returns:
        dict with portfolio_ic, effective_n, total_signals
    """
    if not signals or not weights:
        return {
            "portfolio_ic": 0.0,
            "effective_n": 0.0,
            "total_signals": 0,
        }

    # Weighted average IC
    portfolio_ic = sum(
        w * sig.get("expected_ic", 0.0)
        for w, sig in zip(weights, signals)
    )

    # Effective N (Herfindahl inverse): 1 / sum(w_i^2)
    sum_w_sq = sum(w * w for w in weights)
    effective_n = 1.0 / sum_w_sq if sum_w_sq > 0 else 0.0

    return {
        "portfolio_ic": round(portfolio_ic, 6),
        "effective_n": round(effective_n, 2),
        "total_signals": len(signals),
    }


def filter_redundant_signals(
    signals: list[dict],
    flagged_pairs: list[dict],
) -> list[dict]:
    """Remove the lower-IC signal from each flagged redundant pair.

    Args:
        signals: list of signal dicts with 'name' and 'expected_ic'
        flagged_pairs: list of dicts with 'signal_a', 'signal_b', 'correlation'

    Returns:
        Filtered signal list with redundant signals removed
    """
    if not flagged_pairs:
        return list(signals)

    # Build lookup
    sig_by_name = {s["name"]: s for s in signals}
    to_remove = set()

    for pair in flagged_pairs:
        a_name = pair.get("signal_a", "")
        b_name = pair.get("signal_b", "")
        a = sig_by_name.get(a_name)
        b = sig_by_name.get(b_name)
        if a is None or b is None:
            continue
        # Remove the one with lower expected IC
        if a.get("expected_ic", 0) >= b.get("expected_ic", 0):
            to_remove.add(b_name)
        else:
            to_remove.add(a_name)

    return [s for s in signals if s["name"] not in to_remove]


def evaluate_promotion(
    portfolio_ic_history: list[float],
    paper_sharpe_min: float = 1.5,
    paper_days: int = 7,
    annualization_factor: float = 252.0,
) -> dict:
    """Evaluate whether the portfolio is ready for promotion.

    Computes portfolio-level Sharpe from IC history and checks if it
    exceeds the threshold for the required number of consecutive days.

    Args:
        portfolio_ic_history: daily portfolio IC values
        paper_sharpe_min: minimum Sharpe ratio for promotion
        paper_days: required consecutive days above threshold
        annualization_factor: trading days per year (default 252)

    Returns:
        dict with sharpe, consecutive_days, recommended (bool), reason
    """
    if len(portfolio_ic_history) < paper_days:
        return {
            "sharpe": 0.0,
            "consecutive_days": 0,
            "recommended": False,
            "reason": f"insufficient data ({len(portfolio_ic_history)}/{paper_days} days)",
        }

    mean_ic = sum(portfolio_ic_history) / len(portfolio_ic_history)
    variance = sum((x - mean_ic) ** 2 for x in portfolio_ic_history) / (
        len(portfolio_ic_history) - 1
    )
    std_ic = math.sqrt(variance) if variance > 0 else 0.0
    sharpe = (mean_ic / std_ic) * math.sqrt(annualization_factor) if std_ic > 0 else 0.0

    # Count consecutive days above threshold (from the end)
    consecutive = 0
    for ic in reversed(portfolio_ic_history):
        if std_ic > 0:
            daily_sharpe = (ic / std_ic) * math.sqrt(annualization_factor)
        else:
            daily_sharpe = 0.0
        if daily_sharpe >= paper_sharpe_min:
            consecutive += 1
        else:
            break

    recommended = consecutive >= paper_days
    reason = (
        f"Sharpe={sharpe:.2f}, {consecutive}/{paper_days} consecutive days"
        + (" — PROMOTE" if recommended else " — not ready")
    )

    return {
        "sharpe": round(sharpe, 4),
        "consecutive_days": consecutive,
        "recommended": recommended,
        "reason": reason,
    }
