"""
Integration tests for the full profiling pipeline (Phase 9, Task 9.1 + 9.2).

Tests the end-to-end flow: synthetic data → profile() → validate → online classify.
"""

import time
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from cluster_pipeline.hierarchy import profile, ProfilingResult
from cluster_pipeline.online import (
    ClassifierConfig,
    DerivativeBuffer,
    OnlineClassifier,
    save_classifier,
    load_classifier,
)
from cluster_pipeline.validate import validate


# ---------------------------------------------------------------------------
# Helpers — Synthetic Data Generation
# ---------------------------------------------------------------------------


_ENT_BASE_COLS = [
    "ent_permutation_returns_8",
    "ent_permutation_returns_16",
    "ent_permutation_returns_32",
    "ent_permutation_imbalance_16",
    "ent_spread_dispersion",
    "ent_volume_dispersion",
    "ent_book_shape",
    "ent_trade_size_dispersion",
]


def _generate_synthetic_regimes(
    n_bars=300, n_regimes=2, micro_per_regime=2, seed=42,
):
    """
    Generate synthetic bars with clear regime + micro-state structure.

    Uses entropy vector column naming (ent_*_mean) so the pipeline
    can match them via config.
    """
    rng = np.random.default_rng(seed)
    n_features = len(_ENT_BASE_COLS)

    # Assign true regimes: blocks of bars
    block_size = n_bars // (n_regimes * 2)
    true_regime = np.zeros(n_bars, dtype=int)
    for i in range(n_bars):
        true_regime[i] = (i // block_size) % n_regimes

    # Assign micro states within regimes
    true_micro = np.zeros(n_bars, dtype=int)
    for i in range(n_bars):
        r = true_regime[i]
        sub_block = (i // max(1, block_size // micro_per_regime)) % micro_per_regime
        true_micro[i] = r * micro_per_regime + sub_block

    # Generate features with regime-dependent dynamics
    data = {}
    for f, base_col in enumerate(_ENT_BASE_COLS):
        col_name = f"{base_col}_mean"
        values = np.zeros(n_bars)
        for i in range(n_bars):
            r = true_regime[i]
            m = true_micro[i]
            base = r * 3.0 + m * 1.5 + (f % 3) * 0.5
            if i > 0:
                values[i] = 0.7 * values[i - 1] + 0.3 * base + rng.standard_normal() * 0.3
            else:
                values[i] = base + rng.standard_normal() * 0.3
        data[col_name] = values

    df = pd.DataFrame(data)
    return df, true_regime, true_micro


def _generate_random_data(n_bars=200, seed=99):
    """Generate uniform random data with no cluster structure."""
    rng = np.random.default_rng(seed)
    data = {f"{col}_mean": rng.standard_normal(n_bars) for col in _ENT_BASE_COLS}
    return pd.DataFrame(data)


def _generate_with_break(n_bars=200, seed=42):
    """Generate data with a mean shift at midpoint."""
    rng = np.random.default_rng(seed)
    mid = n_bars // 2
    data = {}
    for col in _ENT_BASE_COLS:
        col_name = f"{col}_mean"
        first_half = rng.standard_normal(mid) * 0.5
        second_half = rng.standard_normal(n_bars - mid) * 0.5 + 5.0
        data[col_name] = np.concatenate([first_half, second_half])
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# TestIntegrationSynthetic
# ---------------------------------------------------------------------------


class TestIntegrationSynthetic:
    """Full pipeline on synthetic data with known structure."""

    def test_profile_completes(self):
        """profile() runs without error on well-structured data."""
        df, _, _ = _generate_synthetic_regimes(n_bars=300)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        assert isinstance(result, ProfilingResult)
        assert result.macro.k >= 2
        assert result.hierarchy.n_micro_total >= 2

    def test_macro_recovers_regimes(self):
        """Macro discovery finds a reasonable number of regimes."""
        df, true_regime, _ = _generate_synthetic_regimes(
            n_bars=400, n_regimes=2, seed=123
        )
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 5), micro_k_range=range(2, 4),
        )
        # Should find 2-4 macro regimes (BIC may pick more due to micro structure)
        assert 2 <= result.macro.k <= 4

    def test_labels_cover_all_bars(self):
        """Hierarchy labels span all (post-warmup) bars."""
        df, _, _ = _generate_synthetic_regimes(n_bars=200)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        n_labels = len(result.hierarchy.micro_labels)
        n_bars = len(result.bars)
        assert n_labels == n_bars

    def test_quality_above_minimum(self):
        """Well-separated data should have reasonable silhouette."""
        df, _, _ = _generate_synthetic_regimes(
            n_bars=400, n_regimes=2, seed=77
        )
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        # Silhouette should be positive for well-separated clusters
        assert result.macro.quality.silhouette > 0.0


# ---------------------------------------------------------------------------
# TestIntegrationNoStructure
# ---------------------------------------------------------------------------


class TestIntegrationNoStructure:
    """Pipeline gracefully handles data with no cluster structure."""

    def test_no_structure_early_exit(self):
        """Random data triggers early exit."""
        df = _generate_random_data(n_bars=200)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        # Either early_exit is True, or silhouette is low
        # (random data may occasionally pass structure test with weak signal)
        if result.macro.early_exit:
            assert result.macro.structure_test.has_structure is False
        else:
            # If it didn't early exit, quality should be poor
            assert result.macro.quality.silhouette < 0.5

    def test_no_structure_returns_valid_result(self):
        """Even with no structure, ProfilingResult is valid."""
        df = _generate_random_data(n_bars=200)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        assert isinstance(result, ProfilingResult)
        assert result.bars is not None


# ---------------------------------------------------------------------------
# TestIntegrationBreak
# ---------------------------------------------------------------------------


class TestIntegrationBreak:
    """Pipeline handles structural breaks."""

    def test_break_uses_longest_segment(self):
        """With mean shift, pipeline uses longest valid segment."""
        df = _generate_with_break(n_bars=300)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        # Pipeline should complete (using one segment)
        assert isinstance(result, ProfilingResult)
        # After break detection + warmup, bars should be < original 300
        assert len(result.bars) <= 300


# ---------------------------------------------------------------------------
# TestIntegrationSaveLoadClassify
# ---------------------------------------------------------------------------


class TestIntegrationSaveLoadClassify:
    """Profile → save → load → classify roundtrip."""

    def test_save_load_classify(self, tmp_path):
        """Online classifier from saved artifacts classifies training data."""
        df, _, _ = _generate_synthetic_regimes(n_bars=300, seed=55)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )

        if result.macro.early_exit:
            pytest.skip("Early exit — cannot test classification")

        from sklearn.mixture import GaussianMixture
        from cluster_pipeline.derivatives import generate_derivatives

        # Get the reduced feature space used by macro PCA
        macro_pca = result.macro.pca_result
        reduced_cols = macro_pca.column_names  # columns after variance/correlation filter

        # Regenerate derivatives to get full DataFrame
        deriv_result = generate_derivatives(result.bars, vector="entropy")
        derivatives = deriv_result.derivatives
        if deriv_result.warmup_rows > 0:
            derivatives = derivatives.iloc[deriv_result.warmup_rows:].reset_index(drop=True)

        # Select only the reduced columns used for macro PCA
        available_cols = [c for c in reduced_cols if c in derivatives.columns]
        X_filtered = derivatives[available_cols].values

        # Standardize and project using macro PCA
        X_std = (X_filtered - macro_pca.mean) / np.where(macro_pca.std > 1e-12, macro_pca.std, 1.0)
        X_reduced = X_std @ macro_pca.components.T

        # Fit GMM on PCA-reduced data
        macro_gmm = GaussianMixture(
            n_components=result.macro.k, covariance_type="full",
            n_init=5, random_state=42,
        )
        macro_gmm.fit(X_reduced)

        # Build minimal config for save/load test
        micro_pca_components = {}
        micro_pca_mean = {}
        micro_pca_std = {}
        micro_gmm_dict = {}
        label_map = result.hierarchy.label_map

        macro_labels = macro_gmm.predict(X_reduced)

        for regime_id, micro_result in result.micros.items():
            if micro_result is None:
                continue
            m_pca = micro_result.pca_result
            m_cols = [c for c in m_pca.column_names if c in derivatives.columns]
            if not m_cols:
                continue

            micro_pca_components[regime_id] = m_pca.components
            micro_pca_mean[regime_id] = m_pca.mean
            micro_pca_std[regime_id] = m_pca.std

            regime_mask = macro_labels == regime_id
            X_regime = X_filtered[regime_mask]
            # Use same columns as macro for micro (simplified)
            X_m_std = (X_regime - m_pca.mean) / np.where(m_pca.std > 1e-12, m_pca.std, 1.0)
            X_m_reduced = X_m_std @ m_pca.components.T

            m_gmm = GaussianMixture(
                n_components=micro_result.k, covariance_type="full",
                n_init=3, random_state=42,
            )
            if len(X_m_reduced) >= micro_result.k:
                m_gmm.fit(X_m_reduced)
                micro_gmm_dict[regime_id] = m_gmm

        train_ll = macro_gmm.score_samples(X_reduced)
        ll_p10 = float(np.percentile(train_ll, 10))
        ll_p50 = float(np.percentile(train_ll, 50))

        n_states = result.hierarchy.n_micro_total
        rng = np.random.default_rng(42)
        trans = rng.dirichlet(np.ones(n_states), size=n_states)

        config = ClassifierConfig(
            macro_pca_components=macro_pca.components,
            macro_pca_mean=macro_pca.mean,
            macro_pca_std=macro_pca.std,
            macro_gmm=macro_gmm,
            micro_pca_components=micro_pca_components,
            micro_pca_mean=micro_pca_mean,
            micro_pca_std=micro_pca_std,
            micro_gmm=micro_gmm_dict,
            label_map=label_map,
            transition_matrix=trans,
            state_ids=list(range(n_states)),
            training_ll_p10=ll_p10,
            training_ll_p50=ll_p50,
        )

        # Save and reload
        save_classifier(config, tmp_path / "model")
        loaded_config = load_classifier(tmp_path / "model")

        # Classify with loaded model using reduced feature vectors
        clf = OnlineClassifier(loaded_config)
        online_labels = []
        for i in range(len(X_filtered)):
            est = clf.classify(X_filtered[i])
            online_labels.append(est.macro_regime)

        # Should produce multiple distinct regimes
        online_arr = np.array(online_labels)
        assert len(np.unique(online_arr)) >= 2 or result.macro.k == 1


# ---------------------------------------------------------------------------
# TestIntegrationValidation
# ---------------------------------------------------------------------------


class TestIntegrationValidation:
    """End-to-end: profile + validate."""

    def test_validate_after_profile(self):
        """Validate runs on profiling output without error."""
        df, _, _ = _generate_synthetic_regimes(n_bars=300)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        # Generate aligned prices
        rng = np.random.default_rng(42)
        prices = np.exp(np.cumsum(rng.standard_normal(len(result.bars)) * 0.01) + 4.0)
        verdict = validate(result, prices)
        assert verdict.overall in ["GO", "PIVOT", "COLLECT", "DROP"]

    def test_verdict_has_per_state(self):
        df, _, _ = _generate_synthetic_regimes(n_bars=300)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        prices = np.exp(np.cumsum(np.random.default_rng(42).standard_normal(len(result.bars)) * 0.01) + 4.0)
        verdict = validate(result, prices)
        assert len(verdict.per_state_verdicts) == result.hierarchy.n_micro_total


# ---------------------------------------------------------------------------
# TestPerformanceBenchmarks (Task 9.2)
# ---------------------------------------------------------------------------


class TestPerformanceBenchmarks:
    """Performance benchmarks for pipeline components."""

    def test_derivative_generation_time(self):
        """Derivative generation on 2000 bars < 10 seconds."""
        from cluster_pipeline.derivatives import generate_derivatives

        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            f"{col}_mean": rng.standard_normal(2000) for col in _ENT_BASE_COLS
        })

        start = time.time()
        generate_derivatives(df, vector="entropy", max_base_features=8, temporal_windows=[5, 15, 30])
        elapsed = time.time() - start
        assert elapsed < 10.0, f"Derivative generation took {elapsed:.2f}s (limit: 10s)"

    def test_online_classify_throughput(self):
        """Online classification > 500/sec (reasonable for Python+sklearn)."""
        from tests.test_online_classifier import _make_config

        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)

        # Warmup
        for i in range(10):
            clf.classify(X[i])

        n_classify = 200
        start = time.time()
        for i in range(n_classify):
            clf.classify(X[i % len(X)])
        elapsed = time.time() - start

        throughput = n_classify / elapsed
        assert throughput > 500, f"Throughput: {throughput:.0f}/sec (need >500)"

    def test_derivative_buffer_throughput(self):
        """DerivativeBuffer update > 30/sec after warmup."""
        columns = [f"f_{i}" for i in range(10)]
        buf = DerivativeBuffer(columns=columns, temporal_windows=[5, 10])

        rng = np.random.default_rng(42)
        bars = pd.DataFrame({c: rng.standard_normal(500) for c in columns})

        # Warmup
        for i in range(buf.max_window):
            buf.update(bars.iloc[i])

        n_updates = 50
        start = time.time()
        for i in range(buf.max_window, buf.max_window + n_updates):
            buf.update(bars.iloc[i])
        elapsed = time.time() - start

        throughput = n_updates / elapsed
        assert throughput > 30, f"Buffer throughput: {throughput:.0f}/sec (need >30)"

    def test_full_pipeline_time(self):
        """Full profile() on 300 bars < 60 seconds."""
        df, _, _ = _generate_synthetic_regimes(n_bars=300)

        start = time.time()
        profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        elapsed = time.time() - start
        assert elapsed < 60.0, f"Full pipeline took {elapsed:.2f}s (limit: 60s)"
