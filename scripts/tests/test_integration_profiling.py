"""
Integration tests for the full profiling pipeline (Phase 9, Task 9.1 + 9.2).

Tests the end-to-end flow: synthetic data → profile() → validate → online classify.
Expanded: characterization, cross-asset lead/lag, report generation, visualization,
edge cases, and extended performance benchmarks.
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
from cluster_pipeline.validate import validate, cross_symbol_consistency
from cluster_pipeline.characterize import (
    characterize_states,
    compute_signatures,
    return_profile,
    cross_asset_lead_lag,
)
from cluster_pipeline.transitions import empirical_transitions
from cluster_pipeline.report import generate_report


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
        df, _, _ = _generate_synthetic_regimes(n_bars=500)
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

    def test_reduction_time(self):
        """Dimensionality reduction on 2000 bars x 200 cols < 3 seconds."""
        from cluster_pipeline.reduce import reduce_all

        rng = np.random.default_rng(42)
        X = rng.standard_normal((2000, 200))

        start = time.time()
        reduce_all(X, n_components=10, methods=["pca"])
        elapsed = time.time() - start
        assert elapsed < 3.0, f"Reduction took {elapsed:.2f}s (limit: 3s)"

    def test_macro_discovery_time(self):
        """Macro regime discovery on 2000 bars < 30 seconds."""
        from cluster_pipeline.hierarchy import discover_macro_regimes

        rng = np.random.default_rng(42)
        n = 2000
        labels_true = np.repeat([0, 1], n // 2)
        X = rng.standard_normal((n, 5))
        X[labels_true == 1] += 3.0
        # discover_macro_regimes expects a DataFrame
        df_macro = pd.DataFrame(X, columns=[f"d_{i}" for i in range(5)])

        start = time.time()
        discover_macro_regimes(
            df_macro, autocorrelation_threshold=0.0,  # low threshold to include all cols
            k_range=range(2, 5), random_state=42,
        )
        elapsed = time.time() - start
        assert elapsed < 30.0, f"Macro discovery took {elapsed:.2f}s (limit: 30s)"

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


# ---------------------------------------------------------------------------
# TestIntegrationDriftDetection
# ---------------------------------------------------------------------------


class TestIntegrationDriftDetection:
    """Profile first half, classify second half with shifted distribution."""

    def test_drift_detection_fires_on_shift(self):
        """Drift detection triggers when distribution shifts between halves."""
        # Generate 2 halves: first is normal, second is shifted
        rng = np.random.default_rng(42)
        n_half = 200
        data_first = {f"{col}_mean": rng.standard_normal(n_half) for col in _ENT_BASE_COLS}
        data_second = {f"{col}_mean": rng.standard_normal(n_half) + 5.0 for col in _ENT_BASE_COLS}

        df_first = pd.DataFrame(data_first)

        # Profile first half
        result = profile(
            df_first, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )

        if result.macro.early_exit:
            pytest.skip("First half has no structure — cannot test drift")

        # Build classifier from profiling result
        from sklearn.mixture import GaussianMixture
        from cluster_pipeline.derivatives import generate_derivatives

        macro_pca = result.macro.pca_result
        available_cols = [c for c in macro_pca.column_names
                          if c in result.bars.columns or f"{c}" in result.bars.columns]

        # Fit GMM on first half
        deriv_first = generate_derivatives(result.bars, vector="entropy")
        df_deriv = deriv_first.derivatives
        if deriv_first.warmup_rows > 0:
            df_deriv = df_deriv.iloc[deriv_first.warmup_rows:].reset_index(drop=True)

        cols_available = [c for c in macro_pca.column_names if c in df_deriv.columns]
        if not cols_available:
            pytest.skip("No matching derivative columns")

        X_first = df_deriv[cols_available].values
        X_std = (X_first - macro_pca.mean) / np.where(macro_pca.std > 1e-12, macro_pca.std, 1.0)
        X_reduced = X_std @ macro_pca.components.T

        macro_gmm = GaussianMixture(
            n_components=result.macro.k, covariance_type="full",
            n_init=5, random_state=42,
        ).fit(X_reduced)

        # Compute training LL baseline
        train_ll = macro_gmm.score_samples(X_reduced)
        ll_p10 = float(np.percentile(train_ll, 10))
        ll_p50 = float(np.percentile(train_ll, 50))

        # Classify shifted data using raw GMM score
        df_second = pd.DataFrame(data_second)
        deriv_second = generate_derivatives(df_second, vector="entropy")
        df_deriv2 = deriv_second.derivatives
        if deriv_second.warmup_rows > 0:
            df_deriv2 = df_deriv2.iloc[deriv_second.warmup_rows:].reset_index(drop=True)

        cols2 = [c for c in cols_available if c in df_deriv2.columns]
        if not cols2 or len(df_deriv2) == 0:
            pytest.skip("No data after warmup")

        X_second = df_deriv2[cols2].values
        X_std2 = (X_second - macro_pca.mean) / np.where(macro_pca.std > 1e-12, macro_pca.std, 1.0)
        X_reduced2 = X_std2 @ macro_pca.components.T

        # Score shifted data — should have much lower log-likelihood
        shifted_ll = macro_gmm.score_samples(X_reduced2)
        mean_shifted_ll = float(np.mean(shifted_ll))

        # Shifted data LL should be well below training p10
        assert mean_shifted_ll < ll_p10, (
            f"Shifted data LL ({mean_shifted_ll:.2f}) not below training p10 ({ll_p10:.2f})"
        )

    def test_no_drift_on_same_distribution(self):
        """No drift when test data comes from the same generating process."""
        from sklearn.mixture import GaussianMixture

        # Generate two sets from the SAME regime structure
        rng = np.random.default_rng(42)
        n = 500
        k = 2
        # 2-regime data in PCA-like space (bypass derivative pipeline)
        labels = np.repeat(np.arange(k), n // k)
        X_train = rng.standard_normal((n, 5))
        X_test = rng.standard_normal((n, 5))
        for i in range(k):
            mask = labels == i
            shift = rng.standard_normal(5) * 3
            X_train[mask] += shift
            X_test[mask] += shift

        gmm = GaussianMixture(n_components=k, n_init=5, random_state=42).fit(X_train)
        train_ll = gmm.score_samples(X_train)
        ll_p10 = float(np.percentile(train_ll, 10))

        test_ll = float(np.mean(gmm.score_samples(X_test)))

        # Same generating process — test LL should be above training p10
        assert test_ll > ll_p10 - 2.0, (
            f"Same-distribution LL ({test_ll:.2f}) below p10 ({ll_p10:.2f})"
        )


# ---------------------------------------------------------------------------
# TestIntegrationRealData
# ---------------------------------------------------------------------------


class TestIntegrationRealData:
    """Integration test with actual collected data (skipped if unavailable)."""

    @pytest.fixture(autouse=True)
    def _check_data(self):
        import os
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "features")
        if not os.path.isdir(data_dir) or not os.listdir(data_dir):
            pytest.skip("No real data available in data/features/")
        self.data_dir = data_dir

    def test_real_data_completes(self):
        """profile() completes on real collected data without crashing."""
        from cluster_pipeline.loader import load_parquet
        from cluster_pipeline.preprocess import aggregate_bars

        df = load_parquet(self.data_dir)
        if len(df) == 0:
            pytest.skip("No data loaded")

        # Use only BTC and limit for speed
        if "symbol" in df.columns:
            df = df[df["symbol"] == "BTC"].reset_index(drop=True)
        if len(df) > 2000:
            df = df.head(2000)
        if len(df) < 100:
            pytest.skip("Not enough raw data")

        bars = aggregate_bars(df, timeframe="15min")
        if len(bars) < 50:
            pytest.skip("Not enough bars after aggregation")

        # Convert to pandas if polars DataFrame
        bars_pd = bars.to_pandas() if hasattr(bars, 'to_pandas') else bars
        result = profile(
            bars_pd, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        assert isinstance(result, ProfilingResult)


# ---------------------------------------------------------------------------
# Helper — Multi-Symbol Synthetic Data
# ---------------------------------------------------------------------------


def _generate_multi_symbol_regimes(
    n_bars=300, n_regimes=2, n_symbols=3, lead_offset=3, seed=42,
):
    """
    Generate per-symbol label arrays with controllable lead/lag.

    Symbol 0 (BTC) enters each regime first; subsequent symbols are delayed
    by ``lead_offset * i`` bars.

    Returns (per_symbol_labels, per_symbol_prices).
    """
    rng = np.random.default_rng(seed)
    block = n_bars // (n_regimes * 3)
    total = n_bars + lead_offset * (n_symbols - 1)
    base = np.array([i // block % n_regimes for i in range(total)])

    names = ["BTC", "ETH", "SOL"][:n_symbols]
    labels = {}
    prices = {}
    for idx, sym in enumerate(names):
        # More offset → transitions appear earlier → leader
        shift = lead_offset * (n_symbols - 1 - idx)
        labels[sym] = base[shift:shift + n_bars].copy()
        # Correlated geometric random walk
        log_returns = rng.normal(0, 0.005, n_bars)
        prices[sym] = np.exp(np.cumsum(log_returns) + 10.0)

    return labels, prices


# ---------------------------------------------------------------------------
# TestIntegrationCharacterization
# ---------------------------------------------------------------------------


class TestIntegrationCharacterization:
    """Test characterization functions on pipeline output."""

    @pytest.fixture(autouse=True)
    def _profile(self):
        from cluster_pipeline.derivatives import generate_derivatives
        df, _, _ = _generate_synthetic_regimes(n_bars=500, seed=77)
        self.result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        self.prices = np.exp(
            np.random.default_rng(77).normal(0, 0.005, len(self.result.bars)).cumsum() + 10
        )
        # Regenerate derivatives aligned to post-warmup bars
        deriv_result = generate_derivatives(df, vector="entropy")
        derivs = deriv_result.derivatives
        if deriv_result.warmup_rows > 0:
            derivs = derivs.iloc[deriv_result.warmup_rows:].reset_index(drop=True)
        self.derivatives = derivs.iloc[:len(self.result.hierarchy.micro_labels)].reset_index(drop=True)

    def test_characterize_states_from_profile(self):
        tm = empirical_transitions(self.result.hierarchy.micro_labels)
        profiles = characterize_states(
            self.derivatives,
            self.result.hierarchy,
            tm,
        )
        assert isinstance(profiles, dict)
        assert len(profiles) == self.result.hierarchy.n_micro_total
        for state_id, sp in profiles.items():
            assert len(sp.centroid) == len(self.derivatives.columns)

    def test_compute_signatures_from_profile(self):
        labels = self.result.hierarchy.micro_labels
        sig = compute_signatures(self.derivatives, labels, state_id=0, lookback=5)
        if sig is not None:
            assert sig.entry_trajectory.shape[1] == self.derivatives.shape[1]
            assert sig.entry_trajectory.shape[0] == 5

    def test_return_profile_from_profile(self):
        labels = self.result.hierarchy.micro_labels
        rp = return_profile(labels, self.prices, state_id=0, horizons=[1, 5, 10])
        assert isinstance(rp.horizons, dict)
        for h in [1, 5, 10]:
            assert h in rp.horizons
            assert rp.horizons[h]["n"] > 0


# ---------------------------------------------------------------------------
# TestIntegrationCrossAssetLeadLag
# ---------------------------------------------------------------------------


class TestIntegrationCrossAssetLeadLag:
    """Test cross-asset lead/lag with multi-symbol synthetic data."""

    def test_leader_detected_with_offset(self):
        labels, prices = _generate_multi_symbol_regimes(
            n_bars=600, lead_offset=5, seed=42,
        )
        result = cross_asset_lead_lag(labels, per_symbol_prices=prices)
        leaders = [s.leader for s in result.per_state.values() if s.leader is not None]
        assert "BTC" in leaders

    def test_no_leader_when_simultaneous(self):
        labels, prices = _generate_multi_symbol_regimes(
            n_bars=300, lead_offset=0, seed=42,
        )
        result = cross_asset_lead_lag(labels)
        for sll in result.per_state.values():
            assert sll.leader is None

    def test_return_correlations_computed(self):
        labels, prices = _generate_multi_symbol_regimes(
            n_bars=300, lead_offset=0, seed=42,
        )
        result = cross_asset_lead_lag(labels, per_symbol_prices=prices)
        has_corr = any(s.return_correlations is not None for s in result.per_state.values())
        assert has_corr

    def test_insufficient_episodes_graceful(self):
        # BTC always state 0, ETH always state 1 → no shared episodes
        labels = {"BTC": np.zeros(200, dtype=int), "ETH": np.ones(200, dtype=int)}
        result = cross_asset_lead_lag(labels)
        for sll in result.per_state.values():
            assert sll.leader is None
            assert sll.n_episodes == 0


# ---------------------------------------------------------------------------
# TestIntegrationReportGeneration
# ---------------------------------------------------------------------------


class TestIntegrationReportGeneration:
    """Test report generation end-to-end."""

    def test_report_from_pipeline_output(self):
        df, _, _ = _generate_synthetic_regimes(n_bars=400, seed=55)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        prices = np.exp(
            np.random.default_rng(55).normal(0, 0.005, len(result.bars)).cumsum() + 10
        )
        report = generate_report(result, prices, vector="entropy", timeframe="15min")
        assert isinstance(report, str)
        assert len(report) > 100
        assert "## Data Summary" in report

    def test_report_with_early_exit(self):
        df = _generate_random_data(n_bars=200, seed=99)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        prices = np.ones(len(result.bars)) * 100
        report = generate_report(result, prices, vector="entropy", timeframe="15min")
        assert isinstance(report, str)

    def test_report_with_cross_symbol(self):
        df, _, _ = _generate_synthetic_regimes(n_bars=400, seed=66)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        # Simulate cross-symbol: use same labels for two "symbols"
        cs = cross_symbol_consistency({
            "BTC": result.hierarchy.micro_labels,
            "ETH": result.hierarchy.micro_labels,
        })
        prices = np.ones(len(result.bars)) * 100
        report = generate_report(
            result, prices, vector="entropy", timeframe="15min",
            cross_symbol_result=cs,
        )
        assert "## Cross-Symbol Consistency" in report


# ---------------------------------------------------------------------------
# TestIntegrationVisualization
# ---------------------------------------------------------------------------


class TestIntegrationVisualization:
    """Test that visualization functions don't crash on pipeline output."""

    @pytest.fixture(autouse=True)
    def _profile(self, tmp_path):
        df, _, _ = _generate_synthetic_regimes(n_bars=300, seed=88)
        self.result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        self.output_dir = tmp_path

    def test_hierarchy_overview_no_crash(self):
        try:
            from visualize_hierarchy import plot_hierarchy_overview
        except ImportError:
            import sys, os
            from visualize_hierarchy import plot_hierarchy_overview
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = plot_hierarchy_overview(self.result, self.output_dir)
        assert path.exists()
        plt.close("all")

    def test_derivative_pca_no_crash(self):
        try:
            from visualize_hierarchy import plot_derivative_pca
        except ImportError:
            import sys, os
            from visualize_hierarchy import plot_derivative_pca
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = plot_derivative_pca(self.result, self.output_dir)
        assert path.exists()
        plt.close("all")

    def test_transition_graph_no_crash(self):
        try:
            from visualize_hierarchy import plot_transition_graph
        except ImportError:
            import sys, os
            from visualize_hierarchy import plot_transition_graph
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = plot_transition_graph(self.result, self.output_dir)
        assert path.exists()
        plt.close("all")

    def test_drift_dashboard_no_crash(self):
        try:
            from visualize_hierarchy import plot_drift_dashboard
        except ImportError:
            import sys, os
            from visualize_hierarchy import plot_drift_dashboard
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(self.result.hierarchy.micro_labels)
        ll_scores = np.random.default_rng(88).normal(-3.0, 1.0, n)
        path = plot_drift_dashboard(
            ll_scores, training_p10=-5.0, training_p50=-3.0,
            output_dir=self.output_dir,
        )
        assert path.exists()
        plt.close("all")


# ---------------------------------------------------------------------------
# TestIntegrationEdgeCases
# ---------------------------------------------------------------------------


class TestIntegrationEdgeCases:
    """Edge cases for the profiling pipeline."""

    def test_single_regime(self):
        """Data with weak structure → pipeline completes without crash."""
        rng = np.random.default_rng(42)
        data = {f"{col}_mean": rng.standard_normal(200) * 0.1 for col in _ENT_BASE_COLS}
        df = pd.DataFrame(data)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 3),
        )
        assert isinstance(result, ProfilingResult)

    def test_very_short_data(self):
        """50 bars — just above derivative warmup minimum."""
        df, _, _ = _generate_synthetic_regimes(n_bars=100, seed=42)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 3), micro_k_range=range(2, 3),
        )
        assert isinstance(result, ProfilingResult)
        assert len(result.hierarchy.micro_labels) <= 100

    def test_all_nan_single_feature(self):
        """One feature column is all NaN — pipeline should still complete."""
        df, _, _ = _generate_synthetic_regimes(n_bars=300, seed=42)
        nan_col = df.columns[0]
        df[nan_col] = np.nan
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 3),
        )
        assert isinstance(result, ProfilingResult)

    def test_many_features_few_bars(self):
        """More features than usual — pipeline handles gracefully with top-N selection."""
        rng = np.random.default_rng(42)
        # Use standard ent_ column naming + extras to get more features
        extra_ent = [f"ent_extra_{i}" for i in range(12)]
        all_base = list(_ENT_BASE_COLS) + extra_ent  # 20 total base features
        cols = [f"{c}_mean" for c in all_base]
        data = {col: rng.standard_normal(120) for col in cols}
        # Add block structure
        for i, col in enumerate(cols[:8]):
            shift = np.zeros(120)
            shift[60:] = 2.0 * (i % 3)
            data[col] = data[col] + shift
        df = pd.DataFrame(data)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 3), micro_k_range=range(2, 3),
        )
        assert isinstance(result, ProfilingResult)


# ---------------------------------------------------------------------------
# TestPerformanceBenchmarksExtended
# ---------------------------------------------------------------------------


class TestPerformanceBenchmarksExtended:
    """Extended performance benchmarks for characterization and lead/lag."""

    def test_characterization_time(self):
        """characterize_states + signatures < 5s on 500 bars."""
        from cluster_pipeline.derivatives import generate_derivatives
        df, _, _ = _generate_synthetic_regimes(n_bars=500, seed=77)
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        tm = empirical_transitions(result.hierarchy.micro_labels)
        deriv_result = generate_derivatives(df, vector="entropy")
        derivs_df = deriv_result.derivatives
        if deriv_result.warmup_rows > 0:
            derivs_df = derivs_df.iloc[deriv_result.warmup_rows:].reset_index(drop=True)
        derivs_df = derivs_df.iloc[:len(result.hierarchy.micro_labels)].reset_index(drop=True)

        start = time.time()
        characterize_states(derivs_df, result.hierarchy, tm)
        for state_id in range(result.hierarchy.n_micro_total):
            compute_signatures(derivs_df, result.hierarchy.micro_labels, state_id=state_id)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Characterization took {elapsed:.1f}s (limit 5s)"

    def test_cross_asset_lead_lag_time(self):
        """cross_asset_lead_lag < 3s on 3 symbols x 1000 bars."""
        labels, prices = _generate_multi_symbol_regimes(n_bars=1000, seed=42)

        start = time.time()
        cross_asset_lead_lag(labels, per_symbol_prices=prices)
        elapsed = time.time() - start

        assert elapsed < 3.0, f"Lead/lag took {elapsed:.1f}s (limit 3s)"

    def test_full_pipeline_with_characterization_time(self):
        """Full profile + characterize + return_profile < 90s on 500 bars."""
        from cluster_pipeline.derivatives import generate_derivatives
        df, _, _ = _generate_synthetic_regimes(n_bars=500, seed=42)
        prices = np.exp(
            np.random.default_rng(42).normal(0, 0.005, 500).cumsum() + 10
        )

        start = time.time()
        result = profile(
            df, vector="entropy", skip_aggregation=True,
            macro_k_range=range(2, 4), micro_k_range=range(2, 4),
        )
        tm = empirical_transitions(result.hierarchy.micro_labels)
        deriv_result = generate_derivatives(df, vector="entropy")
        derivs_df = deriv_result.derivatives
        if deriv_result.warmup_rows > 0:
            derivs_df = derivs_df.iloc[deriv_result.warmup_rows:].reset_index(drop=True)
        derivs_df = derivs_df.iloc[:len(result.hierarchy.micro_labels)].reset_index(drop=True)
        characterize_states(derivs_df, result.hierarchy, tm)
        p = prices[:len(result.hierarchy.micro_labels)]
        for s in range(result.hierarchy.n_micro_total):
            return_profile(result.hierarchy.micro_labels, p, state_id=s)
        elapsed = time.time() - start

        assert elapsed < 90.0, f"Full pipeline took {elapsed:.1f}s (limit 90s)"
