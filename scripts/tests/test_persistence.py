"""
Tests for Task 7.3: Detector Persistence (Save/Load).

Covers: roundtrip save/load, artifact completeness, drift stats persistence,
classification equivalence, edge cases, and validation errors.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from cluster_pipeline.online import (
    ClassifierConfig,
    OnlineClassifier,
    save_classifier,
    load_classifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_gmm(X, n_components, seed=42):
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        n_init=3, random_state=seed,
    )
    gmm.fit(X)
    return gmm


def _make_config(n_features=10, n_regimes=2, micro_per_regime=2, seed=42):
    """Build a ClassifierConfig with synthetic GMMs."""
    rng = np.random.default_rng(seed)

    n_train = 500
    X_train = np.vstack([
        rng.standard_normal((n_train // n_regimes, n_features)) + i * 3
        for i in range(n_regimes)
    ])

    macro_mean = X_train.mean(axis=0)
    macro_std = X_train.std(axis=0)
    macro_std[macro_std < 1e-12] = 1.0
    X_std = (X_train - macro_mean) / macro_std

    n_pca = min(5, n_features)
    pca = PCA(n_components=n_pca, random_state=seed)
    X_reduced = pca.fit_transform(X_std)
    macro_components = pca.components_

    macro_gmm = _fit_gmm(X_reduced, n_regimes, seed=seed)
    macro_labels = macro_gmm.predict(X_reduced)

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

    label_map = {}
    global_id = 0
    for r in range(n_regimes):
        for l in range(micro_per_regime):
            label_map[global_id] = (r, l)
            global_id += 1
    n_states = global_id

    trans = rng.dirichlet(np.ones(n_states), size=n_states)
    state_ids = list(range(n_states))

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
    ), X_train


# ---------------------------------------------------------------------------
# TestRoundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    """Save → load → classify produces identical results."""

    def test_roundtrip_identical_labels(self, tmp_path):
        cfg, X = _make_config()
        save_classifier(cfg, tmp_path / "model")
        loaded = load_classifier(tmp_path / "model")

        clf_orig = OnlineClassifier(cfg)
        clf_loaded = OnlineClassifier(loaded)

        for i in range(50):
            r1 = clf_orig.classify(X[i])
            r2 = clf_loaded.classify(X[i])
            assert r1.macro_regime == r2.macro_regime
            assert r1.micro_state == r2.micro_state

    def test_roundtrip_probabilities_match(self, tmp_path):
        cfg, X = _make_config()
        save_classifier(cfg, tmp_path / "model")
        loaded = load_classifier(tmp_path / "model")

        clf_orig = OnlineClassifier(cfg)
        clf_loaded = OnlineClassifier(loaded)

        r1 = clf_orig.classify(X[0])
        r2 = clf_loaded.classify(X[0])

        assert r1.macro_confidence == pytest.approx(r2.macro_confidence, abs=1e-10)
        assert r1.rolling_log_likelihood == pytest.approx(
            r2.rolling_log_likelihood, abs=1e-10
        )

    def test_roundtrip_transition_matrix(self, tmp_path):
        cfg, _ = _make_config()
        save_classifier(cfg, tmp_path / "model")
        loaded = load_classifier(tmp_path / "model")

        np.testing.assert_allclose(
            loaded.transition_matrix, cfg.transition_matrix, atol=1e-10
        )

    def test_roundtrip_label_map(self, tmp_path):
        cfg, _ = _make_config()
        save_classifier(cfg, tmp_path / "model")
        loaded = load_classifier(tmp_path / "model")

        assert loaded.label_map == cfg.label_map

    def test_roundtrip_training_stats(self, tmp_path):
        cfg, _ = _make_config()
        save_classifier(cfg, tmp_path / "model")
        loaded = load_classifier(tmp_path / "model")

        assert loaded.training_ll_p10 == pytest.approx(cfg.training_ll_p10)
        assert loaded.training_ll_p50 == pytest.approx(cfg.training_ll_p50)


# ---------------------------------------------------------------------------
# TestArtifactCompleteness
# ---------------------------------------------------------------------------


class TestArtifactCompleteness:
    """All expected files are created."""

    def test_core_files_exist(self, tmp_path):
        cfg, _ = _make_config()
        save_classifier(cfg, tmp_path / "model")
        d = tmp_path / "model"

        assert (d / "pca_macro.npz").exists()
        assert (d / "gmm_macro.npz").exists()
        assert (d / "transitions.json").exists()
        assert (d / "config.json").exists()
        assert (d / "training_stats.json").exists()

    def test_per_regime_files_exist(self, tmp_path):
        cfg, _ = _make_config(n_regimes=3, micro_per_regime=2)
        save_classifier(cfg, tmp_path / "model")
        d = tmp_path / "model"

        for regime_id in cfg.micro_gmm:
            assert (d / f"pca_micro_{regime_id}.npz").exists()
            assert (d / f"gmm_micro_{regime_id}.npz").exists()

    def test_metadata_file_written(self, tmp_path):
        cfg, _ = _make_config()
        metadata = {"n_bars": 1000, "verdict": "GO", "training_range": "2026-01-01/2026-02-01"}
        save_classifier(cfg, tmp_path / "model", metadata=metadata)

        assert (tmp_path / "model" / "metadata.json").exists()
        with open(tmp_path / "model" / "metadata.json") as f:
            loaded_meta = json.load(f)
        assert loaded_meta["verdict"] == "GO"
        assert loaded_meta["n_bars"] == 1000

    def test_no_metadata_file_without_metadata(self, tmp_path):
        cfg, _ = _make_config()
        save_classifier(cfg, tmp_path / "model")
        assert not (tmp_path / "model" / "metadata.json").exists()


# ---------------------------------------------------------------------------
# TestTrainingStats
# ---------------------------------------------------------------------------


class TestTrainingStats:
    """Training stats are correctly saved and loaded."""

    def test_training_stats_json_format(self, tmp_path):
        cfg, _ = _make_config()
        save_classifier(cfg, tmp_path / "model")

        with open(tmp_path / "model" / "training_stats.json") as f:
            stats = json.load(f)

        assert "log_likelihood_p10" in stats
        assert "log_likelihood_p50" in stats
        assert isinstance(stats["log_likelihood_p10"], float)
        assert isinstance(stats["log_likelihood_p50"], float)

    def test_p10_less_than_p50(self, tmp_path):
        cfg, _ = _make_config()
        save_classifier(cfg, tmp_path / "model")

        with open(tmp_path / "model" / "training_stats.json") as f:
            stats = json.load(f)

        assert stats["log_likelihood_p10"] <= stats["log_likelihood_p50"]

    def test_drift_detection_works_after_load(self, tmp_path):
        """Loaded config enables drift detection via training stats."""
        cfg, X = _make_config()
        save_classifier(cfg, tmp_path / "model")
        loaded = load_classifier(tmp_path / "model")

        clf = OnlineClassifier(loaded, drift_window=10, drift_consecutive=5)

        # Feed OOD data
        rng = np.random.default_rng(77)
        n_features = len(loaded.macro_pca_mean)
        for _ in range(20):
            vec = rng.standard_normal(n_features) * 50 + 100
            clf.classify(vec)

        assert clf.drift_detected


# ---------------------------------------------------------------------------
# TestDirectoryHandling
# ---------------------------------------------------------------------------


class TestDirectoryHandling:
    """Tests for directory creation and validation."""

    def test_creates_nested_directory(self, tmp_path):
        cfg, _ = _make_config()
        deep_path = tmp_path / "a" / "b" / "c" / "model"
        save_classifier(cfg, deep_path)
        assert deep_path.exists()
        assert (deep_path / "pca_macro.npz").exists()

    def test_overwrites_existing_files(self, tmp_path):
        cfg1, _ = _make_config(seed=1)
        cfg2, _ = _make_config(seed=2)

        save_classifier(cfg1, tmp_path / "model")
        save_classifier(cfg2, tmp_path / "model")

        loaded = load_classifier(tmp_path / "model")
        # Should match cfg2, not cfg1
        np.testing.assert_allclose(
            loaded.transition_matrix, cfg2.transition_matrix, atol=1e-10
        )

    def test_load_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_classifier(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases."""

    def test_single_regime(self, tmp_path):
        cfg, X = _make_config(n_regimes=1, micro_per_regime=3)
        save_classifier(cfg, tmp_path / "model")
        loaded = load_classifier(tmp_path / "model")

        clf = OnlineClassifier(loaded)
        result = clf.classify(X[0])
        assert result.macro_regime == 0

    def test_many_regimes(self, tmp_path):
        cfg, X = _make_config(n_regimes=4, micro_per_regime=2)
        save_classifier(cfg, tmp_path / "model")
        loaded = load_classifier(tmp_path / "model")

        clf = OnlineClassifier(loaded)
        result = clf.classify(X[0])
        assert 0 <= result.macro_regime < 4

    def test_high_dimensional(self, tmp_path):
        cfg, X = _make_config(n_features=50)
        save_classifier(cfg, tmp_path / "model")
        loaded = load_classifier(tmp_path / "model")

        clf = OnlineClassifier(loaded)
        result = clf.classify(X[0])
        assert isinstance(result.macro_regime, int)
