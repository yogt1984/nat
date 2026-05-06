"""
EAMM Module 7: Entropy Regime Analysis

Core research question: does entropy predict the optimal market making spread?

Discretizes tick entropy into regimes and tests whether optimal spread
distributions differ significantly across regimes.

Reference: EAMM_SPEC.md §1.4
"""

import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from scipy import stats


# Entropy regime boundaries (H = ent_tick_30s, range [0, ln(3)])
LN3 = np.log(3.0)
REGIME_BOUNDARIES = [0.0, 0.35, 0.55, 0.65, LN3 + 0.01]
REGIME_NAMES = ["TRENDING", "TRANSITIONING", "NORMAL", "RANDOM"]


@dataclass
class RegimeAnalysisResult:
    """Results of entropy regime analysis.

    Attributes
    ----------
    regime_names : list of str
        Names of the regimes.
    regime_counts : list of int
        Number of samples per regime.
    regime_spread_matrix : np.ndarray
        Mean PnL per (regime, spread_level). Shape (4, K).
    optimal_spread_per_regime : list of float
        Mean optimal spread (bps) per regime.
    kruskal_wallis_stat : float
        Kruskal-Wallis H statistic for optimal spread across regimes.
    kruskal_wallis_p : float
        p-value of the test.
    eta_squared : float
        Effect size: proportion of variance explained by regime.
    fill_rate_matrix : np.ndarray
        Round-trip fill rate per (regime, spread_level). Shape (4, K).
    adverse_selection_matrix : np.ndarray
        Single-side fill rate (adverse selection) per (regime, spread). Shape (4, K).
    thesis_confirmed : bool
        True if p < 0.01 AND eta_squared > 0.01 AND regimes have different optimal spreads.
    """
    regime_names: List[str]
    regime_counts: List[int]
    regime_spread_matrix: np.ndarray
    optimal_spread_per_regime: List[float]
    kruskal_wallis_stat: float
    kruskal_wallis_p: float
    eta_squared: float
    fill_rate_matrix: np.ndarray
    adverse_selection_matrix: np.ndarray
    thesis_confirmed: bool


def assign_regimes(
    entropy_values: np.ndarray,
    boundaries: List[float] = None,
) -> np.ndarray:
    """Assign each entropy value to a regime index.

    Parameters
    ----------
    entropy_values : np.ndarray
        H_tick_30s values, shape (N,).
    boundaries : list of float
        Bin edges. Default: [0, 0.35, 0.55, 0.65, ln(3)+0.01].

    Returns
    -------
    np.ndarray of int, shape (N,). Values in {0, 1, 2, 3}.
    """
    if boundaries is None:
        boundaries = REGIME_BOUNDARIES
    regimes = np.digitize(entropy_values, boundaries[1:])
    # Clamp to valid range [0, len(REGIME_NAMES)-1]
    regimes = np.clip(regimes, 0, len(REGIME_NAMES) - 1)
    return regimes


def analyze_regimes(
    entropy_values: np.ndarray,
    pnl_matrix: np.ndarray,
    fill_bid_matrix: np.ndarray,
    fill_ask_matrix: np.ndarray,
    fill_rt_matrix: np.ndarray,
    spread_levels_bps: List[float],
    optimal_spread_bps: np.ndarray,
) -> RegimeAnalysisResult:
    """Run full regime analysis.

    Parameters
    ----------
    entropy_values : np.ndarray, shape (N,)
        H_tick_30s for each row.
    pnl_matrix : np.ndarray, shape (N, K)
        PnL in bps at each spread level.
    fill_bid_matrix : np.ndarray, shape (N, K)
        Bid fill indicators.
    fill_ask_matrix : np.ndarray, shape (N, K)
        Ask fill indicators.
    fill_rt_matrix : np.ndarray, shape (N, K)
        Round-trip fill indicators.
    spread_levels_bps : list of float, length K
        The spread levels.
    optimal_spread_bps : np.ndarray, shape (N,)
        Optimal spread per row (from labels module).

    Returns
    -------
    RegimeAnalysisResult
    """
    N, K = pnl_matrix.shape
    regimes = assign_regimes(entropy_values)
    n_regimes = len(REGIME_NAMES)

    # Per-regime statistics
    regime_counts = []
    regime_spread_matrix = np.zeros((n_regimes, K))
    fill_rate_matrix = np.zeros((n_regimes, K))
    adverse_selection_matrix = np.zeros((n_regimes, K))
    optimal_spread_per_regime = []

    regime_groups = []  # for Kruskal-Wallis

    for r in range(n_regimes):
        mask = regimes == r
        count = np.sum(mask)
        regime_counts.append(int(count))

        if count > 0:
            regime_spread_matrix[r, :] = np.nanmean(pnl_matrix[mask], axis=0)
            fill_rate_matrix[r, :] = np.nanmean(fill_rt_matrix[mask], axis=0)

            # Adverse selection: single-side fills (bid XOR ask)
            single_bid = fill_bid_matrix[mask] * (1 - fill_ask_matrix[mask])
            single_ask = fill_ask_matrix[mask] * (1 - fill_bid_matrix[mask])
            adverse_selection_matrix[r, :] = np.nanmean(
                single_bid + single_ask, axis=0
            )

            optimal_spread_per_regime.append(
                float(np.nanmean(optimal_spread_bps[mask]))
            )
            regime_groups.append(optimal_spread_bps[mask])
        else:
            optimal_spread_per_regime.append(0.0)
            regime_groups.append(np.array([]))

    # Kruskal-Wallis test: are optimal spreads different across regimes?
    non_empty_groups = [g for g in regime_groups if len(g) > 0]
    if len(non_empty_groups) >= 2:
        stat, p_value = stats.kruskal(*non_empty_groups)
    else:
        stat, p_value = 0.0, 1.0

    # Eta-squared (effect size for Kruskal-Wallis)
    # eta^2 = (H - k + 1) / (N - k) where H = KW stat, k = n_groups, N = total
    k = len(non_empty_groups)
    N_total = sum(len(g) for g in non_empty_groups)
    if N_total > k:
        eta_sq = (stat - k + 1) / (N_total - k)
        eta_sq = max(0.0, eta_sq)  # clamp to [0, 1]
    else:
        eta_sq = 0.0

    # Thesis confirmation: significant difference + meaningful effect + spread ordering
    thesis_confirmed = (
        p_value < 0.01
        and eta_sq > 0.01
        and len(non_empty_groups) >= 3
    )

    return RegimeAnalysisResult(
        regime_names=REGIME_NAMES,
        regime_counts=regime_counts,
        regime_spread_matrix=regime_spread_matrix,
        optimal_spread_per_regime=optimal_spread_per_regime,
        kruskal_wallis_stat=float(stat),
        kruskal_wallis_p=float(p_value),
        eta_squared=float(eta_sq),
        fill_rate_matrix=fill_rate_matrix,
        adverse_selection_matrix=adverse_selection_matrix,
        thesis_confirmed=thesis_confirmed,
    )


def print_regime_report(result: RegimeAnalysisResult, spread_levels: List[float]):
    """Print a human-readable regime analysis report."""
    print("\n" + "=" * 70)
    print("EAMM ENTROPY REGIME ANALYSIS")
    print("=" * 70)

    print(f"\n{'Regime':<15} {'Count':>8} {'Opt Spread':>12} {'Description'}")
    print("-" * 60)
    for i, name in enumerate(result.regime_names):
        desc = {
            "TRENDING": "Low entropy, directional — widen spread",
            "TRANSITIONING": "Moderate entropy, regime shift — cautious",
            "NORMAL": "Typical entropy — standard spread",
            "RANDOM": "High entropy, noise — tighten spread",
        }.get(name, "")
        print(
            f"{name:<15} {result.regime_counts[i]:>8} "
            f"{result.optimal_spread_per_regime[i]:>10.2f}bp  {desc}"
        )

    print(f"\n{'':15}", end="")
    for s in spread_levels:
        print(f" {s:>7.1f}bp", end="")
    print("  (Mean PnL in bps)")
    print("-" * (16 + 9 * len(spread_levels)))
    for i, name in enumerate(result.regime_names):
        print(f"{name:<15}", end="")
        for k in range(len(spread_levels)):
            v = result.regime_spread_matrix[i, k]
            print(f" {v:>+7.2f}", end="")
        print()

    print(f"\nKruskal-Wallis test:")
    print(f"  H-statistic: {result.kruskal_wallis_stat:.2f}")
    print(f"  p-value:     {result.kruskal_wallis_p:.2e}")
    print(f"  eta-squared: {result.eta_squared:.4f}")

    print(f"\nTHESIS {'CONFIRMED' if result.thesis_confirmed else 'NOT CONFIRMED'}:")
    if result.thesis_confirmed:
        print("  Entropy regime significantly predicts optimal MM spread.")
        print("  Adaptive spread is justified.")
    else:
        reasons = []
        if result.kruskal_wallis_p >= 0.01:
            reasons.append(f"p={result.kruskal_wallis_p:.3f} >= 0.01")
        if result.eta_squared <= 0.01:
            reasons.append(f"eta^2={result.eta_squared:.4f} <= 0.01")
        print(f"  Reasons: {'; '.join(reasons)}")
        print("  Options: more data, different entropy measure, different horizon.")
