#!/usr/bin/env python3
"""
Skeptical Validation Suite for NAT Algorithmic Hypotheses
=========================================================

PURPOSE: Before investing months building 8 algorithms, a NautilusTrader
integration, a dashboard, and a research lab — PROVE that the data supports
the core hypotheses. If entropy doesn't cluster, if features don't predict
returns, if signals don't survive transaction costs, then STOP.

This script runs 20+ statistical tests designed to DISPROVE the core
assumptions. Only hypotheses that survive skeptical scrutiny deserve
implementation effort.

Usage:
    python scripts/skeptical_validation.py --data ./data/features
    python scripts/skeptical_validation.py --data ./data/features --output ./reports/validation
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Constants
# ============================================================

ALPHA = 0.05  # Significance level before correction
SEED = 42
N_PERMUTATIONS = 500
FORWARD_HORIZONS = [1, 5, 10, 20, 50, 100]  # In rows (ticks)

ENTROPY_COLS = [
    "ent_permutation_8",
    "ent_permutation_16",
    "ent_permutation_32",
    "ent_book_shape",
    "ent_trade_size",
]
FEATURE_COLS_ALL = [
    "ent_permutation_8",
    "ent_permutation_16",
    "ent_permutation_32",
    "ent_book_shape",
    "ent_trade_size",
    "ent_rate_of_change",
    "ent_zscore",
    "flow_aggressor_ratio",
    "flow_aggressor_momentum",
    "flow_volume_5s",
    "flow_trade_count_5s",
    "imbalance_l5",
    "imbalance_l10",
    "imbalance_persistence",
    "vol_realized_100",
    "vol_realized_20",
    "vol_ratio",
    "raw_spread_bps",
    "composite_regime_signal",
]


# ============================================================
# Result Data Classes
# ============================================================


@dataclass
class TestResult:
    name: str
    passed: bool  # True = hypothesis SURVIVED skeptical test
    statistic: float
    p_value: float
    detail: str
    verdict: str  # SURVIVES / REJECTED / INCONCLUSIVE


@dataclass
class ValidationReport:
    tests: List[TestResult] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def add(self, result: TestResult):
        self.tests.append(result)

    def print_report(self):
        print("\n" + "=" * 80)
        print("  SKEPTICAL VALIDATION REPORT")
        print("=" * 80)

        survived = sum(1 for t in self.tests if t.verdict == "SURVIVES")
        rejected = sum(1 for t in self.tests if t.verdict == "REJECTED")
        inconclusive = sum(1 for t in self.tests if t.verdict == "INCONCLUSIVE")
        total = len(self.tests)

        print(f"\n  Total tests:   {total}")
        print(f"  SURVIVED:      {survived}  ({survived/total*100:.0f}%)")
        print(f"  REJECTED:      {rejected}  ({rejected/total*100:.0f}%)")
        print(f"  INCONCLUSIVE:  {inconclusive}  ({inconclusive/total*100:.0f}%)")
        print()

        # Group by category
        categories = {}
        for t in self.tests:
            cat = t.name.split(":")[0] if ":" in t.name else "General"
            categories.setdefault(cat, []).append(t)

        for cat, tests in sorted(categories.items()):
            print(f"  --- {cat} ---")
            for t in tests:
                icon = {"SURVIVES": "+", "REJECTED": "X", "INCONCLUSIVE": "?"}[
                    t.verdict
                ]
                print(f"  [{icon}] {t.name}")
                print(f"      {t.detail}")
                print(f"      p={t.p_value:.4f}  stat={t.statistic:.4f}  -> {t.verdict}")
                print()

        print("=" * 80)
        if rejected > survived:
            print("  OVERALL: Majority of hypotheses REJECTED.")
            print("  RECOMMENDATION: Revisit core assumptions before building.")
        elif survived > rejected and survived > inconclusive:
            print("  OVERALL: Majority of hypotheses SURVIVED.")
            print("  RECOMMENDATION: Proceed with caution, validate on live data.")
        else:
            print("  OVERALL: Mixed results.")
            print("  RECOMMENDATION: Collect more data and re-test.")
        print("=" * 80 + "\n")

        self.summary = {
            "total": total,
            "survived": survived,
            "rejected": rejected,
            "inconclusive": inconclusive,
            "recommendation": "proceed" if survived > rejected else "revisit",
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "summary": self.summary,
                "tests": [asdict(t) for t in self.tests],
            },
            indent=2,
            default=str,
        )


# ============================================================
# Data Loading
# ============================================================


def load_data(data_dir: str) -> pd.DataFrame:
    """Load all parquet files from data directory."""
    data_path = Path(data_dir)
    files = sorted(data_path.rglob("*.parquet"))

    if not files:
        print(f"ERROR: No parquet files found in {data_dir}")
        sys.exit(1)

    dfs = []
    for f in files:
        try:
            if f.stat().st_size == 0:
                continue
            table = pq.read_table(f)
            df = table.to_pandas()
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not read {f}: {e}")

    if not dfs:
        print("ERROR: No data loaded from any file.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)

    # Derive returns from midprice
    if "raw_mid_price" in df.columns:
        df["returns"] = df["raw_mid_price"].pct_change()
        df["log_returns"] = np.log(df["raw_mid_price"] / df["raw_mid_price"].shift(1))

        # Forward returns at multiple horizons
        for h in FORWARD_HORIZONS:
            df[f"fwd_ret_{h}"] = (
                df["raw_mid_price"].shift(-h) / df["raw_mid_price"] - 1
            )

    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns from {len(dfs)} files")
    return df


# ============================================================
# TEST 1: Entropy Distribution Analysis
# ============================================================


def test_entropy_distribution(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Is entropy multimodal? Are there natural clusters, or is it just noise?"""
    print("\n[1] Entropy Distribution Analysis...")

    for col in ENTROPY_COLS:
        if col not in df.columns:
            continue

        values = df[col].dropna().values

        # Normality test (Jarque-Bera)
        jb_stat, jb_p = stats.jarque_bera(values)
        report.add(TestResult(
            name=f"Entropy Distribution:{col} normality (Jarque-Bera)",
            passed=jb_p < ALPHA,
            statistic=jb_stat,
            p_value=jb_p,
            detail=f"JB={jb_stat:.1f}, skew={stats.skew(values):.3f}, kurt={stats.kurtosis(values):.3f}",
            verdict="SURVIVES" if jb_p < ALPHA else "REJECTED",
        ))

    # GMM BIC model selection for primary entropy
    primary_entropy_col = "ent_permutation_16"
    if primary_entropy_col in df.columns:
        values = df[primary_entropy_col].dropna().values.reshape(-1, 1)
        bic_scores = []
        for n in range(1, 7):
            gmm = GaussianMixture(n_components=n, random_state=SEED, n_init=5)
            gmm.fit(values)
            bic_scores.append((n, gmm.bic(values), gmm.aic(values)))

        best_n = min(bic_scores, key=lambda x: x[1])[0]
        is_multimodal = best_n > 1

        report.add(TestResult(
            name="Entropy Distribution:GMM optimal clusters (BIC)",
            passed=is_multimodal,
            statistic=float(best_n),
            p_value=0.0,
            detail=f"Optimal clusters={best_n}. BIC scores: {[(n, f'{b:.0f}') for n,b,_ in bic_scores]}",
            verdict="SURVIVES" if is_multimodal else "REJECTED",
        ))

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Histogram
        axes[0].hist(values, bins=60, density=True, alpha=0.7, color="steelblue", edgecolor="white")
        axes[0].set_title(f"Entropy Distribution ({primary_entropy_col})")
        axes[0].set_xlabel("Entropy")
        axes[0].set_ylabel("Density")
        axes[0].axvline(np.percentile(values, 25), color="red", ls="--", label="P25")
        axes[0].axvline(np.percentile(values, 75), color="red", ls="--", label="P75")
        axes[0].axvline(0.3, color="orange", ls=":", lw=2, label="Proposed 0.3")
        axes[0].axvline(0.7, color="orange", ls=":", lw=2, label="Proposed 0.7")
        axes[0].legend(fontsize=8)

        # BIC curve
        axes[1].plot([x[0] for x in bic_scores], [x[1] for x in bic_scores], "o-", color="darkred", label="BIC")
        axes[1].plot([x[0] for x in bic_scores], [x[2] for x in bic_scores], "s--", color="navy", label="AIC")
        axes[1].set_title("GMM Model Selection")
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Score (lower = better)")
        axes[1].legend()
        axes[1].axvline(best_n, color="green", ls=":", label=f"Best={best_n}")

        # QQ-plot
        stats.probplot(values.flatten(), dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot (vs Normal)")

        plt.tight_layout()
        plt.savefig(output_dir / "01_entropy_distribution.png", dpi=150)
        plt.close()


# ============================================================
# TEST 2: Entropy Autocorrelation & Persistence
# ============================================================


def test_entropy_persistence(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Is entropy persistent enough to be a regime indicator, or is it just noise?"""
    print("[2] Entropy Persistence Analysis...")

    primary = "ent_permutation_16"
    if primary not in df.columns:
        return

    values = df[primary].dropna().values
    max_lag = min(200, len(values) // 4)

    # Compute autocorrelation
    acf_values = []
    for lag in range(1, max_lag + 1):
        corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
        acf_values.append(corr)

    acf_arr = np.array(acf_values)

    # Find half-life
    half_life_idx = np.where(acf_arr < 0.5)[0]
    half_life = half_life_idx[0] + 1 if len(half_life_idx) > 0 else max_lag

    # Is ACF at lag-1 significantly positive? (regime persistence)
    lag1_acf = acf_arr[0]
    # Fisher z-transform for significance
    n = len(values)
    z = 0.5 * np.log((1 + lag1_acf) / (1 - lag1_acf + 1e-10))
    se = 1.0 / np.sqrt(n - 3)
    z_score = z / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))

    is_persistent = lag1_acf > 0.5 and p_val < ALPHA

    report.add(TestResult(
        name="Entropy Persistence:Lag-1 autocorrelation",
        passed=is_persistent,
        statistic=lag1_acf,
        p_value=p_val,
        detail=f"ACF(1)={lag1_acf:.4f}, half-life={half_life} ticks, z={z_score:.2f}",
        verdict="SURVIVES" if is_persistent else "REJECTED",
    ))

    # Ljung-Box test for overall serial correlation
    lb_lags = min(20, len(values) // 5)
    acf_sq_sum = sum(acf_arr[i] ** 2 / (n - i - 1) for i in range(lb_lags))
    lb_stat = n * (n + 2) * acf_sq_sum
    lb_p = 1 - stats.chi2.cdf(lb_stat, lb_lags)

    report.add(TestResult(
        name="Entropy Persistence:Ljung-Box serial correlation",
        passed=lb_p < ALPHA,
        statistic=lb_stat,
        p_value=lb_p,
        detail=f"LB({lb_lags})={lb_stat:.1f}, significant serial correlation={lb_p < ALPHA}",
        verdict="SURVIVES" if lb_p < ALPHA else "REJECTED",
    ))

    # ADF stationarity test
    from scipy.stats import norm as norm_dist

    # Simple ADF approximation: regress diff(y) on lag(y)
    dy = np.diff(values)
    y_lag = values[:-1]
    # OLS: dy = alpha + beta * y_lag
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    beta = np.linalg.lstsq(X, dy, rcond=None)[0]
    residuals = dy - X @ beta
    se_beta = np.sqrt(np.sum(residuals ** 2) / (len(dy) - 2) / np.sum((y_lag - y_lag.mean()) ** 2))
    adf_stat = beta[1] / se_beta
    # Approximate p-value (MacKinnon critical values for n>250: 1%=-3.43, 5%=-2.86)
    is_stationary = adf_stat < -2.86

    report.add(TestResult(
        name="Entropy Persistence:ADF stationarity",
        passed=is_stationary,
        statistic=adf_stat,
        p_value=0.05 if is_stationary else 0.5,
        detail=f"ADF={adf_stat:.3f}, critical(-2.86 at 5%). Stationary={is_stationary}",
        verdict="SURVIVES" if is_stationary else "INCONCLUSIVE",
    ))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(acf_arr[:100], color="steelblue")
    axes[0].axhline(0.5, color="red", ls="--", label="0.5 threshold")
    axes[0].axhline(0, color="gray", ls="-")
    axes[0].axhline(1.96 / np.sqrt(n), color="gray", ls=":", label="95% CI")
    axes[0].axhline(-1.96 / np.sqrt(n), color="gray", ls=":")
    axes[0].set_title(f"Entropy ACF (half-life={half_life})")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].legend(fontsize=8)

    # Entropy time series
    axes[1].plot(values[:2000], color="steelblue", alpha=0.8, lw=0.5)
    axes[1].set_title("Entropy Time Series (first 2000 ticks)")
    axes[1].set_xlabel("Tick")
    axes[1].set_ylabel("Entropy")
    axes[1].axhline(0.3, color="orange", ls=":", label="0.3 threshold")
    axes[1].axhline(0.7, color="orange", ls=":", label="0.7 threshold")
    axes[1].legend(fontsize=8)

    # Regime duration histogram
    low_mask = values < np.percentile(values, 25)
    run_lengths = []
    count = 0
    for v in low_mask:
        if v:
            count += 1
        else:
            if count > 0:
                run_lengths.append(count)
            count = 0
    if count > 0:
        run_lengths.append(count)

    if run_lengths:
        axes[2].hist(run_lengths, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
        axes[2].set_title(f"Low-Entropy Regime Duration (P25 threshold)")
        axes[2].set_xlabel("Duration (ticks)")
        axes[2].set_ylabel("Count")
        axes[2].axvline(np.mean(run_lengths), color="red", ls="--", label=f"Mean={np.mean(run_lengths):.1f}")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "02_entropy_persistence.png", dpi=150)
    plt.close()


# ============================================================
# TEST 3: Entropy Predicts Returns?
# ============================================================


def test_entropy_return_predictability(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Does entropy level predict forward returns? This is THE core hypothesis."""
    print("[3] Entropy-Return Predictability...")

    primary = "ent_permutation_16"
    if primary not in df.columns:
        return

    entropy = df[primary].values
    n_tests = 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    plot_idx = 0

    for h in FORWARD_HORIZONS:
        fwd_col = f"fwd_ret_{h}"
        if fwd_col not in df.columns:
            continue

        fwd = df[fwd_col].values
        valid = ~(np.isnan(entropy) | np.isnan(fwd))
        e_valid = entropy[valid]
        f_valid = fwd[valid]

        if len(e_valid) < 100:
            continue

        # Split into quintiles
        quintiles = pd.qcut(e_valid, 5, labels=False, duplicates="drop")
        groups = [f_valid[quintiles == q] for q in sorted(np.unique(quintiles))]

        # Kruskal-Wallis test (non-parametric ANOVA)
        if len(groups) >= 2 and all(len(g) > 10 for g in groups):
            kw_stat, kw_p = stats.kruskal(*groups)
            n_tests += 1

            report.add(TestResult(
                name=f"Entropy Predicts Returns:Kruskal-Wallis h={h}",
                passed=kw_p < ALPHA,
                statistic=kw_stat,
                p_value=kw_p,
                detail=f"Entropy quintiles predict {h}-tick returns? KW={kw_stat:.2f}",
                verdict="SURVIVES" if kw_p < ALPHA else "REJECTED",
            ))

        # Spearman rank correlation
        rho, rho_p = stats.spearmanr(e_valid, f_valid)
        n_tests += 1

        report.add(TestResult(
            name=f"Entropy Predicts Returns:Spearman corr h={h}",
            passed=rho_p < ALPHA,
            statistic=rho,
            p_value=rho_p,
            detail=f"Spearman rho={rho:.4f} between entropy and {h}-tick fwd return",
            verdict="SURVIVES" if rho_p < ALPHA else "REJECTED",
        ))

        # Plot: returns by entropy quintile
        if plot_idx < len(axes):
            means = [g.mean() for g in groups]
            stds = [g.std() / np.sqrt(len(g)) for g in groups]
            axes[plot_idx].bar(range(len(means)), means, yerr=stds,
                              color="steelblue", alpha=0.7, edgecolor="white",
                              capsize=3)
            axes[plot_idx].axhline(0, color="gray", ls="-")
            axes[plot_idx].set_title(f"Fwd Return by Entropy Quintile (h={h})")
            axes[plot_idx].set_xlabel("Entropy Quintile (0=low, 4=high)")
            axes[plot_idx].set_ylabel("Mean Forward Return")
            axes[plot_idx].set_xticks(range(len(means)))
            plot_idx += 1

    # Apply Bonferroni correction note
    if n_tests > 0:
        corrected_alpha = ALPHA / n_tests
        report.add(TestResult(
            name="Entropy Predicts Returns:Bonferroni correction note",
            passed=True,
            statistic=float(n_tests),
            p_value=corrected_alpha,
            detail=f"{n_tests} tests run. Bonferroni-corrected alpha={corrected_alpha:.4f}",
            verdict="INCONCLUSIVE",
        ))

    plt.tight_layout()
    plt.savefig(output_dir / "03_entropy_return_predictability.png", dpi=150)
    plt.close()


# ============================================================
# TEST 4: Momentum Behaves Differently in Low vs High Entropy?
# ============================================================


def test_momentum_regime_conditioned(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Does momentum continuation differ by entropy regime? Core of the gating idea."""
    print("[4] Momentum Regime Conditioning...")

    primary = "ent_permutation_16"
    if primary not in df.columns or "returns" not in df.columns:
        return

    entropy = df[primary].values
    returns = df["returns"].values

    # Compute simple momentum (trailing 20-tick return)
    lookback = 20
    momentum = pd.Series(returns).rolling(lookback).sum().values

    valid = ~(np.isnan(entropy) | np.isnan(momentum) | np.isnan(returns))
    entropy_v = entropy[valid]
    momentum_v = momentum[valid]

    # Forward return (next tick)
    fwd = np.roll(returns, -1)[valid]

    # Define regimes using data percentiles (NOT arbitrary 0.3/0.7)
    p25 = np.percentile(entropy_v, 25)
    p75 = np.percentile(entropy_v, 75)

    low_mask = entropy_v < p25
    high_mask = entropy_v > p75

    # In low entropy: does positive momentum predict positive forward returns?
    low_pos_mom = (momentum_v > 0) & low_mask
    low_neg_mom = (momentum_v < 0) & low_mask
    high_pos_mom = (momentum_v > 0) & high_mask
    high_neg_mom = (momentum_v < 0) & high_mask

    results = {}
    for label, mask in [
        ("Low entropy + pos momentum", low_pos_mom),
        ("Low entropy + neg momentum", low_neg_mom),
        ("High entropy + pos momentum", high_pos_mom),
        ("High entropy + neg momentum", high_neg_mom),
    ]:
        fwd_subset = fwd[mask]
        if len(fwd_subset) < 30:
            continue
        t_stat, t_p = stats.ttest_1samp(fwd_subset, 0)
        results[label] = {
            "mean": fwd_subset.mean(),
            "n": len(fwd_subset),
            "t": t_stat,
            "p": t_p,
        }

    # Key test: momentum continuation in low entropy vs high entropy
    low_continuation = fwd[low_pos_mom].mean() if low_pos_mom.sum() > 30 else np.nan
    high_continuation = fwd[high_pos_mom].mean() if high_pos_mom.sum() > 30 else np.nan

    if not np.isnan(low_continuation) and not np.isnan(high_continuation):
        # Mann-Whitney U test: are they different?
        u_stat, u_p = stats.mannwhitneyu(
            fwd[low_pos_mom], fwd[high_pos_mom], alternative="two-sided"
        )
        report.add(TestResult(
            name="Momentum Regime:Low vs High entropy momentum continuation",
            passed=u_p < ALPHA,
            statistic=u_stat,
            p_value=u_p,
            detail=f"Low-ent continuation={low_continuation:.6f}, High-ent={high_continuation:.6f}",
            verdict="SURVIVES" if u_p < ALPHA else "REJECTED",
        ))

    # Sign accuracy: does momentum sign predict forward return sign?
    for label, mask, regime in [("low_entropy", low_mask, "Low"), ("high_entropy", high_mask, "High")]:
        mom_sub = momentum_v[mask]
        fwd_sub = fwd[mask]
        if len(mom_sub) < 50:
            continue
        sign_accuracy = np.mean(np.sign(mom_sub) == np.sign(fwd_sub))
        # Binomial test: is accuracy > 50%?
        n_correct = int(np.sum(np.sign(mom_sub) == np.sign(fwd_sub)))
        n_total = len(mom_sub)
        binom_p = stats.binomtest(n_correct, n_total, 0.5, alternative="greater").pvalue

        report.add(TestResult(
            name=f"Momentum Regime:{regime} entropy sign accuracy",
            passed=binom_p < ALPHA,
            statistic=sign_accuracy,
            p_value=binom_p,
            detail=f"Sign accuracy={sign_accuracy:.4f} ({n_correct}/{n_total}) in {regime} entropy",
            verdict="SURVIVES" if binom_p < ALPHA and sign_accuracy > 0.52 else "REJECTED",
        ))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Conditional mean returns
    labels_plot = list(results.keys())
    means = [results[k]["mean"] for k in labels_plot]
    colors = ["green" if "pos" in k else "red" for k in labels_plot]
    axes[0].barh(range(len(labels_plot)), means, color=colors, alpha=0.7, edgecolor="white")
    axes[0].set_yticks(range(len(labels_plot)))
    axes[0].set_yticklabels(labels_plot, fontsize=8)
    axes[0].set_title("Mean Forward Return by Regime+Momentum")
    axes[0].set_xlabel("Mean 1-tick Forward Return")
    axes[0].axvline(0, color="gray", ls="-")

    # Scatter: momentum vs forward return in low entropy
    if low_mask.sum() > 50:
        sample_idx = np.random.RandomState(SEED).choice(low_mask.sum(), min(500, low_mask.sum()), replace=False)
        axes[1].scatter(momentum_v[low_mask][sample_idx], fwd[low_mask][sample_idx],
                        alpha=0.3, s=5, color="steelblue")
        axes[1].set_title(f"Low Entropy: Momentum vs Fwd Return")
        axes[1].set_xlabel("Trailing Momentum")
        axes[1].set_ylabel("1-tick Forward Return")
        axes[1].axhline(0, color="gray", ls="-")
        axes[1].axvline(0, color="gray", ls="-")

    # Same for high entropy
    if high_mask.sum() > 50:
        sample_idx = np.random.RandomState(SEED).choice(high_mask.sum(), min(500, high_mask.sum()), replace=False)
        axes[2].scatter(momentum_v[high_mask][sample_idx], fwd[high_mask][sample_idx],
                        alpha=0.3, s=5, color="indianred")
        axes[2].set_title(f"High Entropy: Momentum vs Fwd Return")
        axes[2].set_xlabel("Trailing Momentum")
        axes[2].set_ylabel("1-tick Forward Return")
        axes[2].axhline(0, color="gray", ls="-")
        axes[2].axvline(0, color="gray", ls="-")

    plt.tight_layout()
    plt.savefig(output_dir / "04_momentum_regime_conditioning.png", dpi=150)
    plt.close()


# ============================================================
# TEST 5: Feature-Return Correlations + Permutation Tests
# ============================================================


def test_feature_return_correlations(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Which features (if any) predict forward returns? With permutation-based significance."""
    print("[5] Feature-Return Correlations with Permutation Tests...")

    available_features = [c for c in FEATURE_COLS_ALL if c in df.columns]
    fwd_col = "fwd_ret_10"

    if fwd_col not in df.columns or not available_features:
        return

    rng = np.random.RandomState(SEED)
    results = []

    for feat in available_features:
        valid = df[[feat, fwd_col]].dropna()
        if len(valid) < 100:
            continue

        x = valid[feat].values
        y = valid[fwd_col].values

        # Observed correlation
        observed_corr = np.corrcoef(x, y)[0, 1]

        # Permutation test
        null_corrs = np.empty(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            null_corrs[i] = np.corrcoef(x, rng.permutation(y))[0, 1]

        perm_p = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))

        results.append({
            "feature": feat,
            "corr": observed_corr,
            "perm_p": perm_p,
            "null_mean": null_corrs.mean(),
            "null_std": null_corrs.std(),
        })

        report.add(TestResult(
            name=f"Feature Correlation:{feat} -> fwd_ret_10",
            passed=perm_p < ALPHA,
            statistic=observed_corr,
            p_value=perm_p,
            detail=f"r={observed_corr:.4f}, permutation p={perm_p:.4f}, null_std={null_corrs.std():.4f}",
            verdict="SURVIVES" if perm_p < ALPHA else "REJECTED",
        ))

    # Bonferroni correction
    n_features_tested = len(results)
    if n_features_tested > 0:
        bonferroni_alpha = ALPHA / n_features_tested
        surviving_bonferroni = sum(1 for r in results if r["perm_p"] < bonferroni_alpha)

        report.add(TestResult(
            name="Feature Correlation:Bonferroni-corrected survivors",
            passed=surviving_bonferroni > 0,
            statistic=float(surviving_bonferroni),
            p_value=bonferroni_alpha,
            detail=f"{surviving_bonferroni}/{n_features_tested} features survive Bonferroni (alpha={bonferroni_alpha:.4f})",
            verdict="SURVIVES" if surviving_bonferroni > 0 else "REJECTED",
        ))

    # Plot
    if results:
        results_sorted = sorted(results, key=lambda r: abs(r["corr"]), reverse=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart of correlations
        names = [r["feature"] for r in results_sorted]
        corrs = [r["corr"] for r in results_sorted]
        colors = ["green" if r["perm_p"] < ALPHA else "gray" for r in results_sorted]
        axes[0].barh(range(len(names)), corrs, color=colors, alpha=0.7, edgecolor="white")
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names, fontsize=7)
        axes[0].set_title("Feature-Return Correlations (green=significant)")
        axes[0].set_xlabel("Pearson Correlation with 10-tick Fwd Return")
        axes[0].axvline(0, color="gray", ls="-")

        # P-value plot
        p_vals = [r["perm_p"] for r in results_sorted]
        axes[1].barh(range(len(names)), [-np.log10(max(p, 1e-10)) for p in p_vals],
                     color="steelblue", alpha=0.7, edgecolor="white")
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=7)
        axes[1].axvline(-np.log10(ALPHA), color="red", ls="--", label=f"alpha={ALPHA}")
        if n_features_tested > 0:
            axes[1].axvline(-np.log10(bonferroni_alpha), color="darkred", ls=":", label=f"Bonferroni={bonferroni_alpha:.4f}")
        axes[1].set_title("-log10(p-value) from Permutation Test")
        axes[1].set_xlabel("-log10(p)")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / "05_feature_return_correlations.png", dpi=150)
        plt.close()


# ============================================================
# TEST 6: Feature Redundancy & Dimensionality
# ============================================================


def test_feature_redundancy(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Are most features redundant? What is the effective dimensionality?"""
    print("[6] Feature Redundancy & Dimensionality...")

    available = [c for c in FEATURE_COLS_ALL if c in df.columns]
    if len(available) < 3:
        return

    data = df[available].dropna()
    if len(data) < 100:
        return

    # Correlation matrix
    corr_matrix = data.corr()

    # Count highly correlated pairs
    high_corr_pairs = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            c = abs(corr_matrix.iloc[i, j])
            if c > 0.8:
                high_corr_pairs.append((available[i], available[j], corr_matrix.iloc[i, j]))

    redundancy_ratio = len(high_corr_pairs) / max(1, len(available) * (len(available) - 1) / 2)

    report.add(TestResult(
        name="Feature Redundancy:Highly correlated pairs (|r|>0.8)",
        passed=len(high_corr_pairs) < len(available),
        statistic=float(len(high_corr_pairs)),
        p_value=redundancy_ratio,
        detail=f"{len(high_corr_pairs)} pairs with |r|>0.8 out of {len(available)} features. Pairs: {[(a,b,f'{c:.2f}') for a,b,c in high_corr_pairs[:5]]}",
        verdict="SURVIVES" if redundancy_ratio < 0.3 else "REJECTED",
    ))

    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    pca = PCA()
    pca.fit(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim_90 = np.searchsorted(cumvar, 0.90) + 1
    dim_95 = np.searchsorted(cumvar, 0.95) + 1

    report.add(TestResult(
        name="Feature Redundancy:Effective dimensionality (PCA)",
        passed=dim_95 < len(available),
        statistic=float(dim_95),
        p_value=dim_95 / len(available),
        detail=f"90% variance in {dim_90} dims, 95% in {dim_95} dims (out of {len(available)} features)",
        verdict="SURVIVES" if dim_95 < len(available) * 0.7 else "INCONCLUSIVE",
    ))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Correlation heatmap
    im = axes[0].imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(available)))
    axes[0].set_xticklabels(available, rotation=90, fontsize=5)
    axes[0].set_yticks(range(len(available)))
    axes[0].set_yticklabels(available, fontsize=5)
    axes[0].set_title("Feature Correlation Matrix")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Scree plot
    axes[1].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_, color="steelblue", alpha=0.7, edgecolor="white")
    axes[1].plot(range(1, len(cumvar) + 1), cumvar, "r-o", markersize=4, label="Cumulative")
    axes[1].axhline(0.90, color="orange", ls=":", label="90%")
    axes[1].axhline(0.95, color="red", ls=":", label="95%")
    axes[1].set_title("PCA Explained Variance")
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Variance Ratio")
    axes[1].legend(fontsize=8)

    # Top loadings for PC1 and PC2
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        columns=["PC1", "PC2"],
        index=available,
    )
    loadings_sorted = loadings.reindex(loadings["PC1"].abs().sort_values(ascending=True).index)
    loadings_sorted.plot(kind="barh", ax=axes[2], alpha=0.7)
    axes[2].set_title("PCA Loadings (PC1 & PC2)")
    axes[2].set_xlabel("Loading")

    plt.tight_layout()
    plt.savefig(output_dir / "06_feature_redundancy.png", dpi=150)
    plt.close()


# ============================================================
# TEST 7: Regime Classification Stability
# ============================================================


def test_regime_stability(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Is GMM clustering stable across bootstrap samples?"""
    print("[7] Regime Classification Stability (Bootstrap)...")

    primary = "ent_permutation_16"
    if primary not in df.columns:
        return

    values = df[primary].dropna().values.reshape(-1, 1)
    n = len(values)
    n_bootstraps = 50
    n_clusters = 3
    rng = np.random.RandomState(SEED)

    # Reference clustering
    ref_gmm = GaussianMixture(n_components=n_clusters, random_state=SEED, n_init=5)
    ref_labels = ref_gmm.fit_predict(values)

    # Bootstrap ARI scores
    from sklearn.metrics import adjusted_rand_score

    ari_scores = []
    cluster_means_all = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        boot_values = values[idx]

        gmm = GaussianMixture(n_components=n_clusters, random_state=SEED, n_init=3)
        boot_labels = gmm.fit_predict(boot_values)

        # ARI between reference and bootstrap (on shared indices)
        ref_subset = ref_labels[idx]
        ari = adjusted_rand_score(ref_subset, boot_labels)
        ari_scores.append(ari)
        cluster_means_all.append(sorted(gmm.means_.flatten()))

    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    pct_stable = np.mean(np.array(ari_scores) > 0.7)

    report.add(TestResult(
        name="Regime Stability:Bootstrap ARI",
        passed=mean_ari > 0.6,
        statistic=mean_ari,
        p_value=1.0 - pct_stable,
        detail=f"Mean ARI={mean_ari:.3f} +/- {std_ari:.3f}, {pct_stable:.0%} bootstraps with ARI>0.7",
        verdict="SURVIVES" if mean_ari > 0.6 else "REJECTED",
    ))

    # Cluster mean stability
    means_arr = np.array(cluster_means_all)
    mean_cv = np.std(means_arr, axis=0) / (np.abs(np.mean(means_arr, axis=0)) + 1e-10)

    report.add(TestResult(
        name="Regime Stability:Cluster center consistency",
        passed=np.all(mean_cv < 0.2),
        statistic=float(np.max(mean_cv)),
        p_value=float(np.max(mean_cv)),
        detail=f"Cluster center CV: {mean_cv.round(4).tolist()}. Stable = all CV < 0.2",
        verdict="SURVIVES" if np.all(mean_cv < 0.2) else "REJECTED",
    ))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(ari_scores, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(mean_ari, color="red", ls="--", label=f"Mean={mean_ari:.3f}")
    axes[0].axvline(0.7, color="orange", ls=":", label="Stability threshold")
    axes[0].set_title("Bootstrap ARI Distribution")
    axes[0].set_xlabel("Adjusted Rand Index")
    axes[0].legend(fontsize=8)

    for i in range(min(3, means_arr.shape[1])):
        axes[1].hist(means_arr[:, i], bins=15, alpha=0.5, label=f"Cluster {i+1}")
    axes[1].set_title("Cluster Center Distributions (Bootstrap)")
    axes[1].set_xlabel("Entropy Mean")
    axes[1].legend(fontsize=8)

    # Show reference clustering on data
    colors_map = ["steelblue", "indianred", "green", "orange", "purple"]
    for c in range(n_clusters):
        mask = ref_labels == c
        axes[2].hist(values[mask], bins=30, alpha=0.5, color=colors_map[c % len(colors_map)],
                     label=f"Cluster {c} (n={mask.sum()})")
    axes[2].set_title("Reference GMM Clustering")
    axes[2].set_xlabel("Entropy")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "07_regime_stability.png", dpi=150)
    plt.close()


# ============================================================
# TEST 8: Baseline Comparison
# ============================================================


def test_baseline_comparison(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Can any entropy-based signal beat trivial baselines?"""
    print("[8] Baseline Strategy Comparison...")

    if "raw_mid_price" not in df.columns:
        return

    prices = df["raw_mid_price"].values
    returns = np.diff(prices) / prices[:-1]

    if len(returns) < 200:
        return

    # Strategy 1: Buy and hold
    bh_sharpe = np.sqrt(252 * 24 * 3600) * returns.mean() / (returns.std() + 1e-10)

    # Strategy 2: Simple momentum (long if trailing 20 ticks positive)
    lookback = 20
    mom = pd.Series(returns).rolling(lookback).sum().values[lookback:]
    mom_signal = np.sign(mom)
    mom_returns = mom_signal[:-1] * returns[lookback + 1:]
    mom_sharpe = np.sqrt(252 * 24 * 3600) * mom_returns.mean() / (mom_returns.std() + 1e-10)

    # Strategy 3: Mean reversion (fade 2-sigma moves)
    z = (prices - pd.Series(prices).rolling(50).mean().values) / (pd.Series(prices).rolling(50).std().values + 1e-10)
    z = z[50:]
    mr_signal = np.where(z > 2, -1, np.where(z < -2, 1, 0))
    mr_returns = mr_signal[:-1] * returns[50:]
    valid_mr = mr_returns[~np.isnan(mr_returns)]
    mr_sharpe = np.sqrt(252 * 24 * 3600) * valid_mr.mean() / (valid_mr.std() + 1e-10) if len(valid_mr) > 0 else 0

    # Strategy 4: Entropy-gated momentum (the proposed approach)
    primary = "ent_permutation_16"
    if primary in df.columns:
        entropy = df[primary].values
        p25 = np.percentile(entropy, 25)
        p75 = np.percentile(entropy, 75)

        # Simple version: long momentum in low entropy, mean-revert in high
        eg_returns = np.zeros(len(returns))
        for i in range(lookback, len(returns)):
            if entropy[i] < p25:
                # Momentum in low entropy
                trailing = sum(returns[i - lookback:i])
                eg_returns[i] = np.sign(trailing) * returns[i]
            elif entropy[i] > p75:
                # Mean revert in high entropy
                z_val = (prices[i] - np.mean(prices[max(0, i - 50):i])) / (np.std(prices[max(0, i - 50):i]) + 1e-10)
                if abs(z_val) > 1.5:
                    eg_returns[i] = -np.sign(z_val) * returns[i]

        eg_valid = eg_returns[eg_returns != 0]
        eg_sharpe = np.sqrt(252 * 24 * 3600) * eg_valid.mean() / (eg_valid.std() + 1e-10) if len(eg_valid) > 10 else 0
    else:
        eg_sharpe = 0
        eg_valid = np.array([])

    strategies = {
        "Buy & Hold": bh_sharpe,
        "Simple Momentum (20)": mom_sharpe,
        "Mean Reversion (z>2)": mr_sharpe,
        "Entropy-Gated (proposed)": eg_sharpe,
    }

    best_baseline = max(
        [(k, v) for k, v in strategies.items() if k != "Entropy-Gated (proposed)"],
        key=lambda x: x[1],
    )

    beats_baseline = eg_sharpe > best_baseline[1]

    report.add(TestResult(
        name="Baseline:Entropy-gated vs best baseline",
        passed=beats_baseline,
        statistic=eg_sharpe,
        p_value=0.0,
        detail=f"Entropy-gated Sharpe={eg_sharpe:.2f} vs best baseline ({best_baseline[0]}) Sharpe={best_baseline[1]:.2f}",
        verdict="SURVIVES" if beats_baseline else "REJECTED",
    ))

    # Strategy 5: Random trading baseline
    rng = np.random.RandomState(SEED)
    random_sharpes = []
    for _ in range(100):
        random_signal = rng.choice([-1, 0, 1], size=len(returns), p=[0.25, 0.5, 0.25])
        random_ret = random_signal * returns
        random_valid = random_ret[random_ret != 0]
        if len(random_valid) > 10:
            rs = np.sqrt(252 * 24 * 3600) * random_valid.mean() / (random_valid.std() + 1e-10)
            random_sharpes.append(rs)

    random_p95 = np.percentile(random_sharpes, 95) if random_sharpes else 0

    report.add(TestResult(
        name="Baseline:Entropy-gated vs random (95th percentile)",
        passed=eg_sharpe > random_p95,
        statistic=eg_sharpe,
        p_value=np.mean(np.array(random_sharpes) >= eg_sharpe) if random_sharpes else 1.0,
        detail=f"Entropy-gated Sharpe={eg_sharpe:.2f} vs random 95th pct={random_p95:.2f}",
        verdict="SURVIVES" if eg_sharpe > random_p95 else "REJECTED",
    ))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(strategies.keys())
    sharpes = list(strategies.values())
    colors = ["steelblue" if n != "Entropy-Gated (proposed)" else "green" for n in names]
    axes[0].barh(range(len(names)), sharpes, color=colors, alpha=0.7, edgecolor="white")
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].set_title("Strategy Sharpe Ratios")
    axes[0].set_xlabel("Annualized Sharpe")
    axes[0].axvline(0, color="gray", ls="-")

    if random_sharpes:
        axes[1].hist(random_sharpes, bins=30, color="gray", alpha=0.5, edgecolor="white", label="Random strategies")
        axes[1].axvline(eg_sharpe, color="green", ls="--", lw=2, label=f"Entropy-Gated ({eg_sharpe:.2f})")
        axes[1].axvline(random_p95, color="red", ls=":", label=f"Random 95th pct ({random_p95:.2f})")
        axes[1].set_title("Entropy-Gated vs Random Strategies")
        axes[1].set_xlabel("Annualized Sharpe")
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "08_baseline_comparison.png", dpi=150)
    plt.close()


# ============================================================
# TEST 9: Transaction Cost Survival
# ============================================================


def test_transaction_cost_survival(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Can any signal survive realistic transaction costs?"""
    print("[9] Transaction Cost Survival Analysis...")

    if "returns" not in df.columns:
        return

    returns = df["returns"].dropna().values

    # Hyperliquid costs
    maker_fee = 0.0002  # 2 bps
    taker_fee = 0.0005  # 5 bps
    spread_cost = 0.0003  # 3 bps average
    round_trip = 2 * taker_fee + spread_cost  # Conservative

    # What edge is needed to survive costs?
    for win_rate in [0.52, 0.55, 0.60]:
        edge = 2 * win_rate - 1
        if edge <= 0:
            continue
        breakeven_magnitude = round_trip / edge

        # What fraction of actual returns exceed this magnitude?
        abs_returns = np.abs(returns)
        fraction_above = np.mean(abs_returns > breakeven_magnitude)

        report.add(TestResult(
            name=f"Transaction Costs:Breakeven at {win_rate:.0%} win rate",
            passed=fraction_above > 0.5,
            statistic=breakeven_magnitude,
            p_value=fraction_above,
            detail=f"Need |return|>{breakeven_magnitude*100:.3f}%, {fraction_above:.1%} of ticks exceed this",
            verdict="SURVIVES" if fraction_above > 0.3 else "REJECTED",
        ))

    # Net Sharpe estimation
    avg_abs_return = np.mean(np.abs(returns))
    gross_edge = avg_abs_return * 0.04  # Assume 4% edge (52% win rate)
    net_edge = gross_edge - round_trip

    report.add(TestResult(
        name="Transaction Costs:Net edge after costs",
        passed=net_edge > 0,
        statistic=net_edge,
        p_value=0.0,
        detail=f"Gross edge={gross_edge*10000:.1f}bps, costs={round_trip*10000:.1f}bps, net={net_edge*10000:.1f}bps",
        verdict="SURVIVES" if net_edge > 0 else "REJECTED",
    ))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Return distribution with cost overlay
    abs_ret = np.abs(returns) * 100
    axes[0].hist(abs_ret, bins=80, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(round_trip * 100, color="red", ls="--", lw=2, label=f"Round-trip cost ({round_trip*100:.3f}%)")
    axes[0].set_title("Absolute Return Distribution vs Transaction Costs")
    axes[0].set_xlabel("Absolute Return (%)")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(0, np.percentile(abs_ret, 99))
    axes[0].legend(fontsize=8)

    # Breakeven analysis
    win_rates = np.linspace(0.51, 0.65, 30)
    breakevens = [round_trip / (2 * wr - 1) * 100 for wr in win_rates]
    axes[1].plot(win_rates * 100, breakevens, color="darkred", lw=2)
    axes[1].axhline(avg_abs_return * 100, color="steelblue", ls="--", label=f"Avg |return|={avg_abs_return*100:.3f}%")
    axes[1].set_title("Required Win Magnitude vs Win Rate")
    axes[1].set_xlabel("Win Rate (%)")
    axes[1].set_ylabel("Required Avg Win (%)")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, min(max(breakevens), avg_abs_return * 100 * 5))

    plt.tight_layout()
    plt.savefig(output_dir / "09_transaction_costs.png", dpi=150)
    plt.close()


# ============================================================
# TEST 10: Lead-Lag Analysis
# ============================================================


def test_lead_lag(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Do features lead returns, or is the relationship contemporaneous/lagged?"""
    print("[10] Lead-Lag Analysis...")

    available = [c for c in FEATURE_COLS_ALL if c in df.columns]
    if "returns" not in df.columns or not available:
        return

    returns = df["returns"].values
    max_lag = 50

    leading_features = []

    for feat in available:
        feat_vals = df[feat].values
        valid = ~(np.isnan(feat_vals) | np.isnan(returns))
        fv = feat_vals[valid]
        rv = returns[valid]

        if len(fv) < max_lag * 3:
            continue

        # Cross-correlation at different lags
        cross_corrs = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                cc = np.corrcoef(fv[:lag], rv[-lag:])[0, 1]
            elif lag > 0:
                cc = np.corrcoef(fv[lag:], rv[:-lag])[0, 1]
            else:
                cc = np.corrcoef(fv, rv)[0, 1]
            cross_corrs[lag] = cc

        peak_lag = max(cross_corrs, key=lambda k: abs(cross_corrs[k]))
        peak_corr = cross_corrs[peak_lag]

        # Feature leads if peak is at negative lag (feature predates returns)
        feature_leads = peak_lag < 0

        if feature_leads and abs(peak_corr) > 0.01:
            leading_features.append((feat, peak_lag, peak_corr))

    n_leading = len(leading_features)

    report.add(TestResult(
        name="Lead-Lag:Features that lead returns",
        passed=n_leading > 0,
        statistic=float(n_leading),
        p_value=n_leading / max(1, len(available)),
        detail=f"{n_leading}/{len(available)} features lead returns. Top: {leading_features[:3]}",
        verdict="SURVIVES" if n_leading > 0 else "REJECTED",
    ))


# ============================================================
# TEST 11: Composite Regime Signal Validity
# ============================================================


def test_composite_regime(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Does the composite_regime_signal actually differentiate market states?"""
    print("[11] Composite Regime Signal Validity...")

    if "composite_regime_signal" not in df.columns:
        return

    regime = df["composite_regime_signal"].values

    for h in [5, 20, 50]:
        fwd_col = f"fwd_ret_{h}"
        if fwd_col not in df.columns:
            continue

        fwd = df[fwd_col].values
        valid = ~(np.isnan(regime) | np.isnan(fwd))
        r_valid = regime[valid]
        f_valid = fwd[valid]

        # Split regime into terciles
        terciles = pd.qcut(r_valid, 3, labels=False, duplicates="drop")
        groups = [f_valid[terciles == t] for t in sorted(np.unique(terciles))]

        if len(groups) >= 2 and all(len(g) > 30 for g in groups):
            kw_stat, kw_p = stats.kruskal(*groups)

            # Effect size (epsilon-squared)
            n_total = sum(len(g) for g in groups)
            epsilon_sq = (kw_stat - len(groups) + 1) / (n_total - len(groups))

            report.add(TestResult(
                name=f"Regime Signal:Differentiates {h}-tick returns",
                passed=kw_p < ALPHA,
                statistic=kw_stat,
                p_value=kw_p,
                detail=f"KW={kw_stat:.2f}, epsilon^2={epsilon_sq:.4f}, mean returns by tercile: {[f'{g.mean():.6f}' for g in groups]}",
                verdict="SURVIVES" if kw_p < ALPHA and epsilon_sq > 0.01 else "REJECTED",
            ))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(regime[~np.isnan(regime)], bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].set_title("Composite Regime Signal Distribution")
    axes[0].set_xlabel("Regime Score")

    if "fwd_ret_20" in df.columns:
        valid = ~(np.isnan(regime) | np.isnan(df["fwd_ret_20"].values))
        sample_size = min(2000, valid.sum())
        idx = np.random.RandomState(SEED).choice(np.where(valid)[0], sample_size, replace=False)
        axes[1].scatter(regime[idx], df["fwd_ret_20"].values[idx], alpha=0.2, s=3, color="steelblue")
        axes[1].set_title("Regime Score vs 20-tick Forward Return")
        axes[1].set_xlabel("Regime Score")
        axes[1].set_ylabel("Forward Return")
        axes[1].axhline(0, color="gray", ls="-")

    plt.tight_layout()
    plt.savefig(output_dir / "11_composite_regime.png", dpi=150)
    plt.close()


# ============================================================
# TEST 12: Return Distribution Analysis
# ============================================================


def test_return_properties(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Are returns predictable at all, or pure noise?"""
    print("[12] Return Distribution Properties...")

    if "returns" not in df.columns:
        return

    returns = df["returns"].dropna().values

    # Test 1: Returns are mean-zero? (t-test)
    t_stat, t_p = stats.ttest_1samp(returns, 0)
    report.add(TestResult(
        name="Returns:Mean significantly different from zero",
        passed=t_p < ALPHA,
        statistic=t_stat,
        p_value=t_p,
        detail=f"Mean={returns.mean():.8f}, t={t_stat:.3f}. Non-zero mean = drift exists",
        verdict="SURVIVES" if t_p < ALPHA else "INCONCLUSIVE",
    ))

    # Test 2: Return autocorrelation (serial dependence)
    lag1_corr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    n = len(returns)
    se = 1.0 / np.sqrt(n)
    z = lag1_corr / se
    acf_p = 2 * (1 - stats.norm.cdf(abs(z)))

    report.add(TestResult(
        name="Returns:Serial dependence (lag-1 autocorrelation)",
        passed=acf_p < ALPHA,
        statistic=lag1_corr,
        p_value=acf_p,
        detail=f"ACF(1)={lag1_corr:.6f}, z={z:.3f}. Positive = momentum, Negative = reversal",
        verdict="SURVIVES" if acf_p < ALPHA else "REJECTED",
    ))

    # Test 3: Volatility clustering (ARCH effect)
    sq_returns = returns ** 2
    sq_acf = np.corrcoef(sq_returns[:-1], sq_returns[1:])[0, 1]

    report.add(TestResult(
        name="Returns:Volatility clustering (squared return ACF)",
        passed=sq_acf > 0.05,
        statistic=sq_acf,
        p_value=0.0,
        detail=f"ACF of r^2={sq_acf:.4f}. High = vol clusters, exploitable for position sizing",
        verdict="SURVIVES" if sq_acf > 0.05 else "REJECTED",
    ))

    # Test 4: Fat tails
    kurt = stats.kurtosis(returns)
    report.add(TestResult(
        name="Returns:Fat tails (excess kurtosis)",
        passed=kurt > 3,
        statistic=kurt,
        p_value=0.0,
        detail=f"Kurtosis={kurt:.2f} (normal=0). Fat tails = extreme events more likely than normal",
        verdict="SURVIVES" if kurt > 1 else "INCONCLUSIVE",
    ))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(returns, bins=100, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    x = np.linspace(returns.min(), returns.max(), 200)
    axes[0].plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), "r-", label="Normal")
    axes[0].set_title(f"Return Distribution (kurt={kurt:.1f})")
    axes[0].set_xlabel("Return")
    axes[0].legend(fontsize=8)

    # ACF of returns
    acf_vals = [np.corrcoef(returns[:-k], returns[k:])[0, 1] for k in range(1, 51)]
    axes[1].bar(range(1, 51), acf_vals, color="steelblue", alpha=0.7, edgecolor="white")
    axes[1].axhline(1.96 / np.sqrt(n), color="red", ls=":", label="95% CI")
    axes[1].axhline(-1.96 / np.sqrt(n), color="red", ls=":")
    axes[1].set_title("Return ACF")
    axes[1].set_xlabel("Lag")
    axes[1].legend(fontsize=8)

    # ACF of squared returns
    sq_acf_vals = [np.corrcoef(sq_returns[:-k], sq_returns[k:])[0, 1] for k in range(1, 51)]
    axes[2].bar(range(1, 51), sq_acf_vals, color="indianred", alpha=0.7, edgecolor="white")
    axes[2].set_title("Squared Return ACF (Volatility Clustering)")
    axes[2].set_xlabel("Lag")

    plt.tight_layout()
    plt.savefig(output_dir / "12_return_properties.png", dpi=150)
    plt.close()


# ============================================================
# TEST 13: Cross-Entropy Correlation Between Entropy Measures
# ============================================================


def test_entropy_agreement(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Do different entropy measures agree, or are they measuring different things?"""
    print("[13] Entropy Measure Agreement...")

    available = [c for c in ENTROPY_COLS if c in df.columns]
    if len(available) < 2:
        return

    data = df[available].dropna()
    if len(data) < 100:
        return

    corr = data.corr()

    # Average pairwise correlation
    upper_tri = corr.values[np.triu_indices_from(corr.values, k=1)]
    avg_corr = np.mean(upper_tri)
    min_corr = np.min(upper_tri)

    report.add(TestResult(
        name="Entropy Agreement:Pairwise correlation between entropy measures",
        passed=avg_corr > 0.5,
        statistic=avg_corr,
        p_value=0.0,
        detail=f"Avg pairwise r={avg_corr:.3f}, min={min_corr:.3f}. High = redundant, Low = different constructs",
        verdict="SURVIVES" if avg_corr > 0.3 else "INCONCLUSIVE",
    ))

    # Do they agree on regime classification?
    regime_agreement = []
    p25s = {c: np.percentile(data[c], 25) for c in available}
    p75s = {c: np.percentile(data[c], 75) for c in available}

    for i in range(len(data)):
        low_votes = sum(1 for c in available if data[c].iloc[i] < p25s[c])
        high_votes = sum(1 for c in available if data[c].iloc[i] > p75s[c])
        if low_votes >= len(available) // 2 or high_votes >= len(available) // 2:
            regime_agreement.append(1)
        else:
            regime_agreement.append(0)

    agreement_rate = np.mean(regime_agreement)

    report.add(TestResult(
        name="Entropy Agreement:Regime classification consensus",
        passed=agreement_rate > 0.3,
        statistic=agreement_rate,
        p_value=0.0,
        detail=f"{agreement_rate:.1%} of ticks have majority agreement on low/high entropy",
        verdict="SURVIVES" if agreement_rate > 0.25 else "REJECTED",
    ))


# ============================================================
# TEST 14: Data Sufficiency
# ============================================================


def test_data_sufficiency(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Is there enough data to draw conclusions?"""
    print("[14] Data Sufficiency Analysis...")

    n = len(df)

    # Power analysis for detecting small effects
    # For Cohen's d = 0.1, alpha = 0.01, power = 0.8: need ~1570 observations
    required_d01 = 1570
    # For d = 0.2: need ~400
    required_d02 = 400

    report.add(TestResult(
        name="Data Sufficiency:Sample size for small effects (d=0.1)",
        passed=n >= required_d01,
        statistic=float(n),
        p_value=n / required_d01,
        detail=f"Have {n:,} rows, need {required_d01:,} for d=0.1 at alpha=0.01, power=0.8",
        verdict="SURVIVES" if n >= required_d01 else "REJECTED",
    ))

    report.add(TestResult(
        name="Data Sufficiency:Sample size for medium effects (d=0.2)",
        passed=n >= required_d02,
        statistic=float(n),
        p_value=n / required_d02,
        detail=f"Have {n:,} rows, need {required_d02:,} for d=0.2 at alpha=0.01, power=0.8",
        verdict="SURVIVES" if n >= required_d02 else "INCONCLUSIVE",
    ))

    # NaN analysis
    feature_cols = [c for c in FEATURE_COLS_ALL if c in df.columns]
    nan_rates = df[feature_cols].isna().mean()
    high_nan = nan_rates[nan_rates > 0.1]

    report.add(TestResult(
        name="Data Sufficiency:Feature completeness",
        passed=len(high_nan) == 0,
        statistic=float(len(high_nan)),
        p_value=nan_rates.max() if len(nan_rates) > 0 else 0,
        detail=f"{len(high_nan)} features with >10% NaN. Worst: {nan_rates.nlargest(3).to_dict() if len(nan_rates) > 0 else 'N/A'}",
        verdict="SURVIVES" if len(high_nan) == 0 else "REJECTED",
    ))

    # Time span
    if "timestamp" in df.columns:
        try:
            times = pd.to_datetime(df["timestamp"])
            duration = times.max() - times.min()
            report.add(TestResult(
                name="Data Sufficiency:Time span",
                passed=duration > pd.Timedelta(hours=1),
                statistic=duration.total_seconds(),
                p_value=0.0,
                detail=f"Data spans {duration}. Need >1h minimum for meaningful analysis",
                verdict="SURVIVES" if duration > pd.Timedelta(hours=1) else "REJECTED",
            ))
        except Exception:
            pass


# ============================================================
# TEST 15: Spread & Microstructure Dynamics
# ============================================================


def test_microstructure(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Do microstructure features (spread, depth) contain tradeable information?"""
    print("[15] Microstructure Dynamics...")

    if "raw_spread_bps" not in df.columns:
        return

    spread = df["raw_spread_bps"].values
    valid_spread = spread[~np.isnan(spread)]

    # Spread distribution
    report.add(TestResult(
        name="Microstructure:Spread distribution",
        passed=True,
        statistic=np.median(valid_spread),
        p_value=0.0,
        detail=f"Median spread={np.median(valid_spread):.2f}bps, mean={np.mean(valid_spread):.2f}bps, P95={np.percentile(valid_spread, 95):.2f}bps",
        verdict="INCONCLUSIVE",
    ))

    # Does wide spread predict higher volatility?
    if "vol_realized_20" in df.columns:
        vol = df["vol_realized_20"].values
        valid = ~(np.isnan(spread) | np.isnan(vol))
        if valid.sum() > 100:
            corr, p = stats.spearmanr(spread[valid], vol[valid])
            report.add(TestResult(
                name="Microstructure:Spread-volatility relationship",
                passed=p < ALPHA and corr > 0,
                statistic=corr,
                p_value=p,
                detail=f"Spearman r={corr:.4f}. Positive = spread widens with volatility (expected)",
                verdict="SURVIVES" if p < ALPHA and corr > 0.1 else "INCONCLUSIVE",
            ))

    # Does imbalance predict direction?
    for imb_col in ["imbalance_l5", "imbalance_l10"]:
        if imb_col not in df.columns or "fwd_ret_5" not in df.columns:
            continue
        imb = df[imb_col].values
        fwd = df["fwd_ret_5"].values
        valid = ~(np.isnan(imb) | np.isnan(fwd))
        if valid.sum() > 100:
            corr, p = stats.spearmanr(imb[valid], fwd[valid])
            report.add(TestResult(
                name=f"Microstructure:{imb_col} predicts 5-tick return",
                passed=p < ALPHA,
                statistic=corr,
                p_value=p,
                detail=f"r={corr:.4f}. Positive = buy imbalance predicts up move",
                verdict="SURVIVES" if p < ALPHA else "REJECTED",
            ))


# ============================================================
# TEST 16: Entropy-Volatility Relationship
# ============================================================


def test_entropy_volatility(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Is entropy informative about volatility regime?"""
    print("[16] Entropy-Volatility Relationship...")

    primary = "ent_permutation_16"
    vol_col = "vol_realized_20"

    if primary not in df.columns or vol_col not in df.columns:
        return

    entropy = df[primary].values
    vol = df[vol_col].values
    valid = ~(np.isnan(entropy) | np.isnan(vol))

    if valid.sum() < 100:
        return

    e_valid = entropy[valid]
    v_valid = vol[valid]

    # Correlation
    corr, p = stats.spearmanr(e_valid, v_valid)

    report.add(TestResult(
        name="Entropy-Vol:Correlation",
        passed=p < ALPHA,
        statistic=corr,
        p_value=p,
        detail=f"Entropy-Vol Spearman r={corr:.4f}. Negative = low entropy = low vol (regime alignment)",
        verdict="SURVIVES" if p < ALPHA else "REJECTED",
    ))

    # Do entropy clusters have different volatility?
    quintiles = pd.qcut(e_valid, 5, labels=False, duplicates="drop")
    vol_by_quintile = [v_valid[quintiles == q] for q in sorted(np.unique(quintiles))]

    if len(vol_by_quintile) >= 2 and all(len(g) > 20 for g in vol_by_quintile):
        kw_stat, kw_p = stats.kruskal(*vol_by_quintile)
        report.add(TestResult(
            name="Entropy-Vol:Volatility differs by entropy quintile",
            passed=kw_p < ALPHA,
            statistic=kw_stat,
            p_value=kw_p,
            detail=f"KW={kw_stat:.2f}, vol means by quintile: {[f'{g.mean():.4f}' for g in vol_by_quintile]}",
            verdict="SURVIVES" if kw_p < ALPHA else "REJECTED",
        ))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sample_size = min(3000, valid.sum())
    idx = np.random.RandomState(SEED).choice(valid.sum(), sample_size, replace=False)
    axes[0].scatter(e_valid[idx], v_valid[idx], alpha=0.2, s=3, color="steelblue")
    axes[0].set_title(f"Entropy vs Volatility (r={corr:.3f})")
    axes[0].set_xlabel("Entropy")
    axes[0].set_ylabel("Realized Volatility")

    bp = axes[1].boxplot([g for g in vol_by_quintile], labels=[f"Q{i+1}" for i in range(len(vol_by_quintile))])
    axes[1].set_title("Volatility by Entropy Quintile")
    axes[1].set_xlabel("Entropy Quintile")
    axes[1].set_ylabel("Realized Volatility")

    plt.tight_layout()
    plt.savefig(output_dir / "16_entropy_volatility.png", dpi=150)
    plt.close()


# ============================================================
# TEST 17: Effect Size Analysis
# ============================================================


def test_effect_sizes(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Even if statistically significant, are effects economically meaningful?"""
    print("[17] Effect Size Analysis (Cohen's d)...")

    primary = "ent_permutation_16"
    if primary not in df.columns or "fwd_ret_20" not in df.columns:
        return

    entropy = df[primary].values
    fwd = df["fwd_ret_20"].values
    valid = ~(np.isnan(entropy) | np.isnan(fwd))
    e_valid = entropy[valid]
    f_valid = fwd[valid]

    p25 = np.percentile(e_valid, 25)
    p75 = np.percentile(e_valid, 75)

    low_returns = f_valid[e_valid < p25]
    high_returns = f_valid[e_valid > p75]

    if len(low_returns) < 30 or len(high_returns) < 30:
        return

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(low_returns) - 1) * low_returns.std() ** 2 + (len(high_returns) - 1) * high_returns.std() ** 2)
        / (len(low_returns) + len(high_returns) - 2)
    )
    cohens_d = (low_returns.mean() - high_returns.mean()) / (pooled_std + 1e-10)

    # Classification: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)
    if abs(cohens_d) < 0.2:
        size_class = "NEGLIGIBLE"
    elif abs(cohens_d) < 0.5:
        size_class = "SMALL"
    elif abs(cohens_d) < 0.8:
        size_class = "MEDIUM"
    else:
        size_class = "LARGE"

    report.add(TestResult(
        name="Effect Size:Low vs High entropy return difference (Cohen's d)",
        passed=abs(cohens_d) > 0.1,
        statistic=cohens_d,
        p_value=0.0,
        detail=f"d={cohens_d:.4f} ({size_class}). Low entropy mean ret={low_returns.mean():.6f}, High={high_returns.mean():.6f}",
        verdict="SURVIVES" if abs(cohens_d) > 0.2 else "REJECTED",
    ))


# ============================================================
# TEST 18: Walk-Forward Stability
# ============================================================


def test_walk_forward_stability(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Do feature-return relationships hold across time, or only in certain periods?"""
    print("[18] Walk-Forward Stability...")

    primary = "ent_permutation_16"
    if primary not in df.columns or "fwd_ret_10" not in df.columns:
        return

    n = len(df)
    n_folds = 5
    fold_size = n // n_folds

    fold_corrs = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size
        fold = df.iloc[start:end]

        valid = fold[[primary, "fwd_ret_10"]].dropna()
        if len(valid) < 50:
            continue

        corr = valid[primary].corr(valid["fwd_ret_10"])
        fold_corrs.append(corr)

    if len(fold_corrs) < 3:
        return

    # Are correlations consistently positive or negative?
    same_sign = all(c > 0 for c in fold_corrs) or all(c < 0 for c in fold_corrs)
    corr_std = np.std(fold_corrs)

    report.add(TestResult(
        name="Walk-Forward:Entropy-return correlation stability",
        passed=same_sign and corr_std < 0.05,
        statistic=np.mean(fold_corrs),
        p_value=corr_std,
        detail=f"Fold correlations: {[f'{c:.4f}' for c in fold_corrs]}. Same sign={same_sign}, std={corr_std:.4f}",
        verdict="SURVIVES" if same_sign and corr_std < 0.05 else "REJECTED",
    ))


# ============================================================
# TEST 19: Nonlinear Predictability
# ============================================================


def test_nonlinear_predictability(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Is the entropy-return relationship nonlinear (justifying complex models)?"""
    print("[19] Nonlinear Predictability (Mutual Information)...")

    primary = "ent_permutation_16"
    if primary not in df.columns or "fwd_ret_10" not in df.columns:
        return

    valid = df[[primary, "fwd_ret_10"]].dropna()
    if len(valid) < 200:
        return

    x = valid[primary].values
    y = valid["fwd_ret_10"].values

    # Estimate mutual information via binning
    n_bins = 20
    x_bins = pd.qcut(x, n_bins, labels=False, duplicates="drop")
    y_bins = pd.qcut(y, n_bins, labels=False, duplicates="drop")

    # Joint and marginal distributions
    joint = np.zeros((n_bins, n_bins))
    for xb, yb in zip(x_bins, y_bins):
        if not np.isnan(xb) and not np.isnan(yb):
            joint[int(xb), int(yb)] += 1

    joint /= joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))

    # Null MI via permutation
    rng = np.random.RandomState(SEED)
    null_mis = []
    for _ in range(200):
        y_perm = rng.permutation(y_bins)
        joint_null = np.zeros((n_bins, n_bins))
        for xb, yb in zip(x_bins, y_perm):
            if not np.isnan(xb) and not np.isnan(yb):
                joint_null[int(xb), int(yb)] += 1
        joint_null /= joint_null.sum()
        mi_null = 0
        for i in range(joint_null.shape[0]):
            for j in range(joint_null.shape[1]):
                if joint_null[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi_null += joint_null[i, j] * np.log(joint_null[i, j] / (px[i] * py[j]))
        null_mis.append(mi_null)

    mi_p = np.mean(np.array(null_mis) >= mi)

    # Compare MI to linear correlation
    linear_corr = np.corrcoef(x, y)[0, 1]

    report.add(TestResult(
        name="Nonlinear:Mutual information entropy->returns",
        passed=mi_p < ALPHA,
        statistic=mi,
        p_value=mi_p,
        detail=f"MI={mi:.6f}, perm p={mi_p:.4f}, linear r={linear_corr:.4f}. High MI with low r = nonlinear relationship",
        verdict="SURVIVES" if mi_p < ALPHA else "REJECTED",
    ))


# ============================================================
# TEST 20: Synthetic Data Warning
# ============================================================


def test_synthetic_data_warning(df: pd.DataFrame, report: ValidationReport, output_dir: Path):
    """Warn if this looks like synthetic/simulated data."""
    print("[20] Synthetic Data Detection...")

    if "returns" not in df.columns:
        return

    returns = df["returns"].dropna().values

    # Synthetic data often has suspiciously uniform distributions or exact symmetry
    # Test: is the return distribution suspiciously normal?
    _, normality_p = stats.shapiro(returns[:min(5000, len(returns))])

    # Real market data is NEVER normal. If it passes Shapiro-Wilk, it's synthetic.
    is_suspicious = normality_p > 0.01

    report.add(TestResult(
        name="Data Quality:Synthetic data check (Shapiro-Wilk normality)",
        passed=not is_suspicious,
        statistic=normality_p,
        p_value=normality_p,
        detail=f"Shapiro p={normality_p:.4f}. Real markets NEVER pass normality. p>0.01 = SUSPICIOUS",
        verdict="SURVIVES" if not is_suspicious else "REJECTED",
    ))

    # Check timestamp regularity
    if "timestamp_ns" in df.columns:
        ts = df["timestamp_ns"].values
        diffs = np.diff(ts)
        unique_diffs = len(np.unique(diffs))
        regularity = unique_diffs / len(diffs)

        report.add(TestResult(
            name="Data Quality:Timestamp regularity",
            passed=regularity > 0.01,
            statistic=regularity,
            p_value=0.0,
            detail=f"{unique_diffs} unique intervals out of {len(diffs)}. Very regular = synthetic. Real markets have variable gaps",
            verdict="SURVIVES" if regularity > 0.05 else "REJECTED",
        ))


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Skeptical validation of NAT algorithmic hypotheses"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/features",
        help="Path to data directory with parquet files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/skeptical_validation",
        help="Output directory for plots and report",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  NAT SKEPTICAL VALIDATION SUITE")
    print("  Testing core hypotheses BEFORE building algorithms")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    df = load_data(args.data)
    print(f"  Features available: {[c for c in FEATURE_COLS_ALL if c in df.columns]}")
    print(f"  Entropy columns: {[c for c in ENTROPY_COLS if c in df.columns]}")

    # Run all tests
    report = ValidationReport()

    test_entropy_distribution(df, report, output_dir)
    test_entropy_persistence(df, report, output_dir)
    test_entropy_return_predictability(df, report, output_dir)
    test_momentum_regime_conditioned(df, report, output_dir)
    test_feature_return_correlations(df, report, output_dir)
    test_feature_redundancy(df, report, output_dir)
    test_regime_stability(df, report, output_dir)
    test_baseline_comparison(df, report, output_dir)
    test_transaction_cost_survival(df, report, output_dir)
    test_lead_lag(df, report, output_dir)
    test_composite_regime(df, report, output_dir)
    test_return_properties(df, report, output_dir)
    test_entropy_agreement(df, report, output_dir)
    test_data_sufficiency(df, report, output_dir)
    test_microstructure(df, report, output_dir)
    test_entropy_volatility(df, report, output_dir)
    test_effect_sizes(df, report, output_dir)
    test_walk_forward_stability(df, report, output_dir)
    test_nonlinear_predictability(df, report, output_dir)
    test_synthetic_data_warning(df, report, output_dir)

    # Print report
    report.print_report()

    # Save report
    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        f.write(report.to_json())
    print(f"Report saved to {report_path}")

    # Save summary
    summary_path = output_dir / "SUMMARY.txt"
    with open(summary_path, "w") as f:
        f.write("NAT Skeptical Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total tests: {report.summary.get('total', 0)}\n")
        f.write(f"Survived:    {report.summary.get('survived', 0)}\n")
        f.write(f"Rejected:    {report.summary.get('rejected', 0)}\n")
        f.write(f"Inconclusive:{report.summary.get('inconclusive', 0)}\n")
        f.write(f"\nRecommendation: {report.summary.get('recommendation', 'N/A')}\n")
    print(f"Summary saved to {summary_path}")

    print(f"\nPlots saved to {output_dir}/")
    print(f"  {len(list(output_dir.glob('*.png')))} visualization files generated")


if __name__ == "__main__":
    main()
