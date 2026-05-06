"""
EAMM Module 5: Walk-Forward Evaluator

Evaluates EAMM vs fixed-spread baselines using expanding-window walk-forward.
No lookahead. Trains on past, predicts on future.

Reference: EAMM_SPEC.md §1.8
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .train import train_eamm, predict_spread, TrainResult
from .simulator import SimulationResult, pnl_to_bps


@dataclass
class SplitResult:
    """Result for one walk-forward split."""
    split_idx: int
    train_size: int
    test_size: int
    eamm_pnl_total_bps: float
    eamm_sharpe: float
    eamm_max_drawdown_bps: float
    eamm_fill_rate: float
    eamm_avg_spread: float
    baseline_pnl: dict  # {spread_bps: total_pnl_bps}
    baseline_sharpe: dict  # {spread_bps: sharpe}


@dataclass
class EvaluationResult:
    """Full walk-forward evaluation result."""
    splits: List[SplitResult]
    n_splits: int
    eamm_avg_sharpe: float
    eamm_total_pnl_bps: float
    best_fixed_sharpe: float
    best_fixed_spread: float
    eamm_beats_best_fixed: bool
    spread_levels_bps: List[float]


def walk_forward_evaluate(
    context_matrix: np.ndarray,
    pnl_matrix_bps: np.ndarray,
    fill_rt_matrix: np.ndarray,
    optimal_spread_bps: np.ndarray,
    spread_levels_bps: List[float],
    feature_names: List[str],
    n_splits: int = 5,
    min_train_frac: float = 0.2,
    mode: str = "regression",
) -> EvaluationResult:
    """Run walk-forward evaluation of EAMM.

    Parameters
    ----------
    context_matrix : np.ndarray, shape (N, 19)
    pnl_matrix_bps : np.ndarray, shape (N, K) — PnL at each spread level
    fill_rt_matrix : np.ndarray, shape (N, K) — round-trip fills at each level
    optimal_spread_bps : np.ndarray, shape (N,) — target labels
    spread_levels_bps : list of float, length K
    feature_names : list of str, length 19
    n_splits : int
    min_train_frac : float
    mode : "regression" or "classification"

    Returns
    -------
    EvaluationResult
    """
    N, K = pnl_matrix_bps.shape
    min_train = int(N * min_train_frac)
    test_size = int((N - min_train) / n_splits)
    spreads_arr = np.array(spread_levels_bps)

    splits = []

    for i in range(n_splits):
        test_start = min_train + i * test_size
        test_end = min(test_start + test_size, N)
        if test_start >= N:
            break

        X_tr = context_matrix[:test_start]
        y_tr = optimal_spread_bps[:test_start]
        X_te = context_matrix[test_start:test_end]
        pnl_te = pnl_matrix_bps[test_start:test_end]
        fill_te = fill_rt_matrix[test_start:test_end]

        # Train
        if mode == "classification":
            # Convert continuous spread to class labels
            y_tr_cls = np.array([
                np.argmin(np.abs(spreads_arr - s)) for s in y_tr
            ])
            result = train_eamm(X_tr, y_tr_cls, feature_names,
                                mode="classification", save_dir=None)
            probs = predict_spread(result, X_te)
            predicted_classes = np.argmax(probs, axis=1)
            predicted_spreads = spreads_arr[predicted_classes]
        else:
            result = train_eamm(X_tr, y_tr, feature_names,
                                mode="regression", save_dir=None)
            predicted_spreads = predict_spread(result, X_te)

        # EAMM PnL: for each test row, get PnL at the predicted spread level
        # Map predicted spread to nearest available level
        predicted_classes_nearest = np.array([
            np.argmin(np.abs(spreads_arr - s)) for s in predicted_spreads
        ])
        eamm_pnl = np.array([
            pnl_te[j, predicted_classes_nearest[j]]
            for j in range(len(pnl_te))
        ])
        eamm_fills = np.array([
            fill_te[j, predicted_classes_nearest[j]]
            for j in range(len(fill_te))
        ])

        eamm_total = float(np.nansum(eamm_pnl))
        eamm_sharpe = _sharpe(eamm_pnl)
        eamm_dd = _max_drawdown(eamm_pnl)
        eamm_fill_rate = float(np.nanmean(eamm_fills))
        eamm_avg_spread = float(np.nanmean(predicted_spreads))

        # Baselines: fixed spread at each level
        baseline_pnl = {}
        baseline_sharpe = {}
        for k, s in enumerate(spread_levels_bps):
            bp = pnl_te[:, k]
            baseline_pnl[s] = float(np.nansum(bp))
            baseline_sharpe[s] = _sharpe(bp)

        splits.append(SplitResult(
            split_idx=i,
            train_size=test_start,
            test_size=test_end - test_start,
            eamm_pnl_total_bps=eamm_total,
            eamm_sharpe=eamm_sharpe,
            eamm_max_drawdown_bps=eamm_dd,
            eamm_fill_rate=eamm_fill_rate,
            eamm_avg_spread=eamm_avg_spread,
            baseline_pnl=baseline_pnl,
            baseline_sharpe=baseline_sharpe,
        ))

    # Aggregate
    eamm_avg_sharpe = np.mean([s.eamm_sharpe for s in splits])
    eamm_total_pnl = sum(s.eamm_pnl_total_bps for s in splits)

    # Best fixed baseline (by average Sharpe across splits)
    avg_baseline_sharpes = {}
    for s_bps in spread_levels_bps:
        avg_baseline_sharpes[s_bps] = np.mean([
            sp.baseline_sharpe[s_bps] for sp in splits
        ])
    best_fixed_spread = max(avg_baseline_sharpes, key=avg_baseline_sharpes.get)
    best_fixed_sharpe = avg_baseline_sharpes[best_fixed_spread]

    return EvaluationResult(
        splits=splits,
        n_splits=len(splits),
        eamm_avg_sharpe=float(eamm_avg_sharpe),
        eamm_total_pnl_bps=float(eamm_total_pnl),
        best_fixed_sharpe=float(best_fixed_sharpe),
        best_fixed_spread=float(best_fixed_spread),
        eamm_beats_best_fixed=float(eamm_avg_sharpe) > float(best_fixed_sharpe),
        spread_levels_bps=spread_levels_bps,
    )


def print_evaluation_report(result: EvaluationResult):
    """Print walk-forward evaluation report."""
    print("\n" + "=" * 70)
    print("EAMM WALK-FORWARD EVALUATION")
    print("=" * 70)

    print(f"\n{'Split':>6} | {'Train':>7} | {'Test':>7} | {'EAMM PnL':>10} | "
          f"{'EAMM Sharpe':>11} | {'Avg Spread':>10} | {'Fill Rate':>9}")
    print("-" * 80)
    for s in result.splits:
        print(f"{s.split_idx:>6} | {s.train_size:>7} | {s.test_size:>7} | "
              f"{s.eamm_pnl_total_bps:>+9.1f}bp | {s.eamm_sharpe:>11.2f} | "
              f"{s.eamm_avg_spread:>8.2f}bp | {s.eamm_fill_rate:>8.1%}")

    print(f"\n  EAMM average Sharpe:       {result.eamm_avg_sharpe:.2f}")
    print(f"  EAMM total PnL:           {result.eamm_total_pnl_bps:+.1f} bps")
    print(f"  Best fixed Sharpe:         {result.best_fixed_sharpe:.2f} "
          f"(at {result.best_fixed_spread:.1f} bps)")
    print(f"  EAMM beats best fixed:     {'YES' if result.eamm_beats_best_fixed else 'NO'}")


def _sharpe(pnl: np.ndarray) -> float:
    """Compute Sharpe ratio of PnL series."""
    valid = pnl[~np.isnan(pnl)]
    if len(valid) < 2:
        return 0.0
    std = np.std(valid)
    if std < 1e-12:
        return 0.0
    return float(np.mean(valid) / std * np.sqrt(len(valid)))


def _max_drawdown(pnl: np.ndarray) -> float:
    """Compute max drawdown from PnL series (in cumulative bps)."""
    valid = pnl[~np.isnan(pnl)]
    if len(valid) == 0:
        return 0.0
    cumulative = np.cumsum(valid)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(np.max(drawdown))
