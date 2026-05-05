"""
Tests for Task 7.2: Online Classifier with Drift Detection.

Covers: classification output, probability sum, drift detection,
no false drift, time-in-state tracking, transition prediction,
and edge cases.
"""

import numpy as np
import pytest
from collections import deque
from sklearn.mixture import GaussianMixture

from cluster_pipeline.online import (
    ClassifierConfig,
    OnlineClassifier,
    StateEstimate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_gmm(X, n_components, seed=42):
    """Fit a GMM and return it."""
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        n_init=3, random_state=seed,
    )
    gmm.fit(X)
    return gmm


def _make_config(n_features=10, n_regimes=2, micro_per_regime=2, seed=42):
    """
    Build a ClassifierConfig with synthetic GMMs.

    Creates n_regimes macro clusters and micro_per_regime micro states per regime.
    """
    rng = np.random.default_rng(seed)

    # Generate training data with n_regimes clusters
    n_train = 500
    X_train = np.vstack([
        rng.standard_normal((n_train // n_regimes, n_features)) + i * 3
        for i in range(n_regimes)
    ])

    # Macro PCA: use identity-ish (just standardize)
    macro_mean = X_train.mean(axis=0)
    macro_std = X_train.std(axis=0)
    macro_std[macro_std < 1e-12] = 1.0
    X_std = (X_train - macro_mean) / macro_std

    # PCA: just use first n_features components (identity)
    n_pca = min(5, n_features)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_pca, random_state=seed)
    X_reduced = pca.fit_transform(X_std)
    macro_components = pca.components_  # (n_pca, n_features)

    # Macro GMM
    macro_gmm = _fit_gmm(X_reduced, n_regimes, seed=seed)

    # Predict macro labels
    macro_labels = macro_gmm.predict(X_reduced)

    # Per-regime micro GMMs
    micro_pca_components = {}
    micro_pca_mean = {}
    micro_pca_std = {}
    micro_gmm = {}

    for regime_id in range(n_regimes):
        regime_mask = macro_labels == regime_id
        X_regime = X_train[regime_mask]
        if len(X_regime) < 10:
            continue

        m_mean = X_regime.mean(axis=0)
        m_std = X_regime.std(axis=0)
        m_std[m_std < 1e-12] = 1.0
        X_m_std = (X_regime - m_mean) / m_std

        n_micro_pca = min(3, n_features)
        m_pca = PCA(n_components=n_micro_pca, random_state=seed)
        X_m_reduced = m_pca.fit_transform(X_m_std)

        micro_pca_components[regime_id] = m_pca.components_
        micro_pca_mean[regime_id] = m_mean
        micro_pca_std[regime_id] = m_std
        micro_gmm[regime_id] = _fit_gmm(
            X_m_reduced, micro_per_regime, seed=seed + regime_id
        )

    # Label map: global → (regime, local)
    label_map = {}
    global_id = 0
    for r in range(n_regimes):
        for l in range(micro_per_regime):
            label_map[global_id] = (r, l)
            global_id += 1
    n_states = global_id

    # Transition matrix (row-stochastic)
    trans = rng.dirichlet(np.ones(n_states), size=n_states)
    state_ids = list(range(n_states))

    # Training log-likelihood stats
    train_ll = macro_gmm.score_samples(X_reduced)
    ll_p10 = float(np.percentile(train_ll, 10))
    ll_p50 = float(np.percentile(train_ll, 50))

    return ClassifierConfig(
        macro_pca_components=macro_components,
        macro_pca_mean=macro_mean,
        macro_pca_std=macro_std,
        macro_gmm=macro_gmm,
        micro_pca_components=micro_pca_components,
        micro_pca_mean=micro_pca_mean,
        micro_pca_std=micro_pca_std,
        micro_gmm=micro_gmm,
        label_map=label_map,
        transition_matrix=trans,
        state_ids=state_ids,
        training_ll_p10=ll_p10,
        training_ll_p50=ll_p50,
    ), X_train, macro_mean, macro_std


def _generate_in_distribution(config, n=50, seed=99):
    """Generate vectors that are in-distribution for the classifier."""
    rng = np.random.default_rng(seed)
    # Generate from same distribution as training
    n_features = len(config.macro_pca_mean)
    # Sample near cluster centers
    vectors = []
    for _ in range(n):
        regime = rng.integers(0, 2)
        vec = rng.standard_normal(n_features) + regime * 3
        vectors.append(vec)
    return vectors


def _generate_out_of_distribution(config, n=50, seed=77):
    """Generate vectors far from training distribution."""
    rng = np.random.default_rng(seed)
    n_features = len(config.macro_pca_mean)
    # Very far from any cluster
    return [rng.standard_normal(n_features) * 50 + 100 for _ in range(n)]


# ---------------------------------------------------------------------------
# TestClassification
# ---------------------------------------------------------------------------


class TestClassification:
    """Basic classification tests."""

    def test_returns_state_estimate(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        assert isinstance(result, StateEstimate)

    def test_macro_regime_valid(self):
        cfg, X, _, _ = _make_config(n_regimes=3)
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        assert 0 <= result.macro_regime < 3

    def test_micro_state_valid(self):
        cfg, X, _, _ = _make_config(n_regimes=2, micro_per_regime=3)
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        assert result.micro_state in cfg.label_map

    def test_confidence_bounded(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        assert 0.0 <= result.macro_confidence <= 1.0
        assert 0.0 <= result.micro_confidence <= 1.0

    def test_composite_label_format(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        assert result.composite_label.startswith("R")
        assert "_S" in result.composite_label


# ---------------------------------------------------------------------------
# TestProbabilities
# ---------------------------------------------------------------------------


class TestProbabilities:
    """Tests for probability outputs."""

    def test_all_probabilities_sum_to_one(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        total = sum(result.all_probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_states_have_probability(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        for state_id in cfg.label_map:
            assert state_id in result.all_probabilities

    def test_probabilities_non_negative(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        for i in range(min(20, len(X))):
            result = clf.classify(X[i])
            for p in result.all_probabilities.values():
                assert p >= 0.0


# ---------------------------------------------------------------------------
# TestTimeInState
# ---------------------------------------------------------------------------


class TestTimeInState:
    """Tests for time-in-state tracking."""

    def test_first_classify_time_is_one(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        assert result.time_in_state == 1

    def test_same_state_increments(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        # Classify same vector repeatedly — should stay in same state
        vec = X[0]
        times = []
        for _ in range(10):
            result = clf.classify(vec)
            times.append(result.time_in_state)
        # Should be 1,2,3,...,10
        assert times == list(range(1, 11))

    def test_state_change_resets_time(self):
        cfg, X, _, _ = _make_config(n_regimes=2)
        clf = OnlineClassifier(cfg)
        # Use vectors from different clusters
        vec_r0 = X[0]  # regime 0
        vec_r1 = X[len(X) // 2 + 50]  # regime 1

        # Classify several from regime 0
        for _ in range(5):
            r = clf.classify(vec_r0)
        state_0 = r.micro_state

        # Switch to regime 1
        r = clf.classify(vec_r1)
        if r.micro_state != state_0:
            assert r.time_in_state == 1


# ---------------------------------------------------------------------------
# TestTransitionPrediction
# ---------------------------------------------------------------------------


class TestTransitionPrediction:
    """Tests for likely_next_state prediction."""

    def test_likely_next_state_in_state_ids(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        assert result.likely_next_state in cfg.state_ids or result.transition_prob == 0.0

    def test_transition_prob_bounded(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        assert 0.0 <= result.transition_prob <= 1.0


# ---------------------------------------------------------------------------
# TestDriftDetection
# ---------------------------------------------------------------------------


class TestDriftDetection:
    """Tests for drift detection logic."""

    def test_no_drift_on_training_data(self):
        """In-distribution data should not trigger drift."""
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg, drift_window=50, drift_consecutive=20)
        # Classify training data (in-distribution)
        for i in range(100):
            clf.classify(X[i % len(X)])
        assert not clf.drift_detected

    def test_drift_on_ood_data(self):
        """Far out-of-distribution data should trigger drift."""
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg, drift_window=20, drift_consecutive=15)

        # First warm up with in-distribution
        for i in range(20):
            clf.classify(X[i])

        # Then feed OOD data
        ood_vecs = _generate_out_of_distribution(cfg, n=50)
        for vec in ood_vecs:
            clf.classify(vec)

        assert clf.drift_detected

    def test_drift_warning_in_estimate(self):
        """StateEstimate.drift_warning matches classifier state."""
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg, drift_window=10, drift_consecutive=5)

        # Feed OOD
        ood_vecs = _generate_out_of_distribution(cfg, n=20)
        last_result = None
        for vec in ood_vecs:
            last_result = clf.classify(vec)

        assert last_result.drift_warning == clf.drift_detected

    def test_drift_resets(self):
        """reset_drift() clears the drift state."""
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg, drift_window=10, drift_consecutive=5)

        ood_vecs = _generate_out_of_distribution(cfg, n=20)
        for vec in ood_vecs:
            clf.classify(vec)

        assert clf.drift_detected
        clf.reset_drift()
        assert not clf.drift_detected
        assert clf.bars_below_threshold == 0

    def test_rolling_ll_is_finite(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        assert np.isfinite(result.rolling_log_likelihood)


# ---------------------------------------------------------------------------
# TestValidationErrors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Input validation tests."""

    def test_2d_vector_raises(self):
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        with pytest.raises(ValueError, match="1-D"):
            clf.classify(X[:2])  # 2D array

    def test_wrong_size_still_works(self):
        """Wrong feature size may produce garbage but shouldn't crash
        if it's 1-D and passes through PCA projection."""
        cfg, _, _, _ = _make_config(n_features=10)
        clf = OnlineClassifier(cfg)
        # Shorter vector — will fail at PCA projection
        with pytest.raises(Exception):
            clf.classify(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Determinism tests."""

    def test_same_input_same_output(self):
        cfg, X, _, _ = _make_config()
        clf1 = OnlineClassifier(cfg)
        clf2 = OnlineClassifier(cfg)

        r1 = clf1.classify(X[0])
        r2 = clf2.classify(X[0])

        assert r1.macro_regime == r2.macro_regime
        assert r1.micro_state == r2.micro_state
        assert r1.macro_confidence == r2.macro_confidence
        assert r1.rolling_log_likelihood == r2.rolling_log_likelihood

    def test_sequence_deterministic(self):
        cfg, X, _, _ = _make_config()
        clf1 = OnlineClassifier(cfg)
        clf2 = OnlineClassifier(cfg)

        results1 = [clf1.classify(X[i]).micro_state for i in range(20)]
        results2 = [clf2.classify(X[i]).micro_state for i in range(20)]
        assert results1 == results2


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases."""

    def test_single_regime(self):
        """Works with n_regimes=1."""
        cfg, X, _, _ = _make_config(n_regimes=1, micro_per_regime=3)
        clf = OnlineClassifier(cfg)
        result = clf.classify(X[0])
        assert result.macro_regime == 0

    def test_many_classifications(self):
        """No memory leak or crash over many calls."""
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg)
        for i in range(1000):
            result = clf.classify(X[i % len(X)])
        assert result is not None

    def test_drift_window_size(self):
        """Internal LL history respects drift_window size."""
        cfg, X, _, _ = _make_config()
        clf = OnlineClassifier(cfg, drift_window=10)
        for i in range(100):
            clf.classify(X[i % len(X)])
        assert len(clf._ll_history) == 10
