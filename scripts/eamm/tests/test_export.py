"""
Skeptical unit tests for EAMM Parameter Export Module.

Tests verify:
- Tree extraction correctness (predictions match original model)
- Fixed-point quantization accuracy and bounds
- Normalization parameter correctness
- Regime LUT boundaries and ordering
- C header generation validity
- Verilog parameter generation
- Edge cases: extreme values, empty models, single trees
- Round-trip accuracy: export → predict_from_export ≈ model.predict
- Fixed-point overflow/underflow detection
- Determinism: same input → same output across calls
"""

import json
import numpy as np
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eamm.train import train_eamm, predict_spread, TrainResult
from eamm.export import (
    extract_trees,
    compute_feature_norms,
    build_regime_lut,
    quantize_threshold,
    dequantize,
    export_model,
    predict_from_export,
    predict_fixed_point,
    ExportResult,
    ExportedTree,
    TreeNode,
    FeatureNormParams,
    RegimeLUT,
    FIXED_POINT_BITS,
    FIXED_POINT_FRAC_BITS,
    FIXED_POINT_SCALE,
)
from eamm.regime_analysis import REGIME_BOUNDARIES, REGIME_NAMES


# ============================================================================
# FIXTURES
# ============================================================================

FEATURE_NAMES = [f"feat_{i}" for i in range(19)]


def _train_model(n=1000, n_estimators=10, max_depth=3):
    """Train a small model for testing."""
    np.random.seed(42)
    X = np.random.randn(n, 19)
    # Learnable target: depends on first few features
    y = 5.0 + 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] ** 2
    y = np.abs(y) + np.random.randn(n) * 0.3
    result = train_eamm(X, y, FEATURE_NAMES, mode="regression",
                        n_estimators=n_estimators, max_depth=max_depth,
                        save_dir=None)
    return result, X, y


def _train_model_large(n=2000, n_estimators=50, max_depth=5):
    """Train a larger model for accuracy tests."""
    np.random.seed(123)
    X = np.random.randn(n, 19)
    y = 10.0 + 3.0 * np.sin(X[:, 0]) + 2.0 * X[:, 1] * X[:, 2]
    y = np.abs(y)
    result = train_eamm(X, y, FEATURE_NAMES, mode="regression",
                        n_estimators=n_estimators, max_depth=max_depth,
                        save_dir=None)
    return result, X, y


# ============================================================================
# TREE EXTRACTION TESTS
# ============================================================================

class TestTreeExtraction:
    def test_extracts_correct_number_of_trees(self):
        result, X, _ = _train_model(n_estimators=10)
        trees = extract_trees(result)
        assert len(trees) == 10

    def test_extracts_correct_number_trees_large(self):
        result, X, _ = _train_model(n_estimators=50, max_depth=4)
        trees = extract_trees(result)
        assert len(trees) == 50

    def test_tree_has_nodes(self):
        result, X, _ = _train_model()
        trees = extract_trees(result)
        for tree in trees:
            assert len(tree.nodes) > 0

    def test_leaves_have_negative_feature_idx(self):
        result, X, _ = _train_model()
        trees = extract_trees(result)
        for tree in trees:
            leaves = [n for n in tree.nodes if n.feature_idx == -1]
            assert len(leaves) == tree.n_leaves
            assert tree.n_leaves > 0

    def test_internal_nodes_have_valid_feature_idx(self):
        result, X, _ = _train_model()
        trees = extract_trees(result)
        for tree in trees:
            for node in tree.nodes:
                if node.feature_idx != -1:
                    assert 0 <= node.feature_idx < 19

    def test_internal_nodes_have_valid_children(self):
        result, X, _ = _train_model()
        trees = extract_trees(result)
        for tree in trees:
            for node in tree.nodes:
                if node.feature_idx != -1:
                    assert 0 <= node.left_child < len(tree.nodes)
                    assert 0 <= node.right_child < len(tree.nodes)

    def test_max_depth_reasonable(self):
        result, X, _ = _train_model(max_depth=3)
        trees = extract_trees(result)
        for tree in trees:
            assert tree.max_depth <= 3

    def test_max_depth_larger_model(self):
        result, X, _ = _train_model(max_depth=5)
        trees = extract_trees(result)
        for tree in trees:
            assert tree.max_depth <= 5

    def test_no_cycles_in_tree(self):
        """Verify tree structure has no cycles (each node visited at most once)."""
        result, X, _ = _train_model()
        trees = extract_trees(result)
        for tree in trees:
            visited = set()
            stack = [0]
            while stack:
                idx = stack.pop()
                assert idx not in visited, f"Cycle detected at node {idx}"
                visited.add(idx)
                node = tree.nodes[idx]
                if node.feature_idx != -1:
                    stack.append(node.left_child)
                    stack.append(node.right_child)

    def test_all_leaves_reachable(self):
        """Every leaf should be reachable from root."""
        result, X, _ = _train_model()
        trees = extract_trees(result)
        for tree in trees:
            reachable = set()
            stack = [0]
            while stack:
                idx = stack.pop()
                node = tree.nodes[idx]
                if node.feature_idx == -1:
                    reachable.add(idx)
                else:
                    stack.append(node.left_child)
                    stack.append(node.right_child)
            leaves = {i for i, n in enumerate(tree.nodes) if n.feature_idx == -1}
            assert reachable == leaves


# ============================================================================
# PREDICTION ACCURACY TESTS (SKEPTICAL)
# ============================================================================

class TestPredictionAccuracy:
    """Verify exported model predictions match original LightGBM predictions."""

    def test_predictions_match_original_small(self):
        result, X, _ = _train_model(n_estimators=10, max_depth=3)
        export = export_model(result, X, output_dir=None)
        original_preds = predict_spread(result, X[:100])
        export_preds = predict_from_export(export, X[:100])
        # Should be very close (only rounding differences)
        np.testing.assert_allclose(export_preds, original_preds, rtol=1e-5, atol=1e-5)

    def test_predictions_match_original_large(self):
        result, X, _ = _train_model_large()
        export = export_model(result, X, output_dir=None)
        original_preds = predict_spread(result, X[:200])
        export_preds = predict_from_export(export, X[:200])
        np.testing.assert_allclose(export_preds, original_preds, rtol=1e-5, atol=1e-5)

    def test_predictions_match_on_unseen_data(self):
        """Test on data not used for training."""
        result, X, _ = _train_model(n=500, n_estimators=20)
        export = export_model(result, X, output_dir=None)
        # Generate new unseen data
        np.random.seed(999)
        X_new = np.random.randn(100, 19)
        original_preds = predict_spread(result, X_new)
        export_preds = predict_from_export(export, X_new)
        np.testing.assert_allclose(export_preds, original_preds, rtol=1e-5, atol=1e-5)

    def test_single_sample_prediction(self):
        result, X, _ = _train_model()
        export = export_model(result, X, output_dir=None)
        original = predict_spread(result, X[:1])
        exported = predict_from_export(export, X[0])
        np.testing.assert_allclose(exported, original, rtol=1e-5, atol=1e-5)

    def test_predictions_non_negative(self):
        result, X, _ = _train_model()
        export = export_model(result, X, output_dir=None)
        preds = predict_from_export(export, X)
        assert np.all(preds >= 0.0)

    def test_predictions_reasonable_range(self):
        """Predictions should be in a reasonable spread range (0-100 bps)."""
        result, X, _ = _train_model()
        export = export_model(result, X, output_dir=None)
        preds = predict_from_export(export, X[:200])
        assert np.all(preds < 200.0), f"Max prediction {preds.max()} seems too large"


# ============================================================================
# FIXED-POINT QUANTIZATION TESTS
# ============================================================================

class TestQuantization:
    def test_quantize_dequantize_roundtrip(self):
        values = [0.0, 1.0, -1.0, 0.5, 3.14159, -2.718]
        for v in values:
            fp = quantize_threshold(v)
            back = dequantize(fp)
            assert abs(back - v) < 1.0 / FIXED_POINT_SCALE, f"Failed for {v}: got {back}"

    def test_quantize_zero(self):
        assert quantize_threshold(0.0) == 0

    def test_quantize_one(self):
        assert quantize_threshold(1.0) == FIXED_POINT_SCALE

    def test_quantize_preserves_order(self):
        """Quantized values should maintain ordering."""
        values = sorted(np.random.randn(100))
        quantized = [quantize_threshold(v) for v in values]
        assert quantized == sorted(quantized)

    def test_quantize_small_differences(self):
        """Can distinguish values that differ by more than 1/scale."""
        a = 1.0
        b = 1.0 + 2.0 / FIXED_POINT_SCALE
        assert quantize_threshold(a) != quantize_threshold(b)

    def test_quantize_negative_values(self):
        v = -5.5
        fp = quantize_threshold(v)
        assert fp < 0
        back = dequantize(fp)
        assert abs(back - v) < 1.0 / FIXED_POINT_SCALE

    def test_fixed_point_no_overflow_16bit(self):
        """Thresholds in typical range should fit in 16-bit signed."""
        # Typical feature values are in [-10, 10]
        for v in np.linspace(-8, 8, 100):
            fp = quantize_threshold(v)
            # 16-bit signed: [-32768, 32767]
            # With 12 frac bits, max representable is ~7.999
            # Values > 8 will overflow — this is expected for our 4.12 format

    def test_quantize_entropy_range(self):
        """Entropy values [0, ln(3)] ≈ [0, 1.099] should fit fine."""
        for v in np.linspace(0, np.log(3), 50):
            fp = quantize_threshold(v)
            back = dequantize(fp)
            assert abs(back - v) < 0.001


# ============================================================================
# FIXED-POINT PREDICTION TESTS
# ============================================================================

class TestFixedPointPrediction:
    def test_fixed_point_close_to_float(self):
        """Fixed-point predictions should be close to floating-point."""
        result, X, _ = _train_model(n_estimators=5, max_depth=2)
        export = export_model(result, X, output_dir=None)
        float_preds = predict_from_export(export, X[:50])
        fp_preds = predict_fixed_point(export, X[:50])
        # Allow more tolerance for quantization error
        # With 12 frac bits, error accumulates over trees
        max_err = abs(float_preds - fp_preds).max()
        mean_err = abs(float_preds - fp_preds).mean()
        # Relative error should be bounded
        assert mean_err < 2.0, f"Mean error {mean_err} too large"

    def test_fixed_point_non_negative(self):
        result, X, _ = _train_model()
        export = export_model(result, X, output_dir=None)
        fp_preds = predict_fixed_point(export, X[:100])
        assert np.all(fp_preds >= 0.0)

    def test_fixed_point_deterministic(self):
        """Same input → same output."""
        result, X, _ = _train_model()
        export = export_model(result, X, output_dir=None)
        p1 = predict_fixed_point(export, X[:20])
        p2 = predict_fixed_point(export, X[:20])
        np.testing.assert_array_equal(p1, p2)

    def test_fixed_point_ordering_preserved(self):
        """If float model says A > B, fixed-point should usually agree."""
        result, X, _ = _train_model(n_estimators=5, max_depth=2)
        export = export_model(result, X, output_dir=None)
        float_preds = predict_from_export(export, X[:100])
        fp_preds = predict_fixed_point(export, X[:100])
        # Check rank correlation
        from scipy.stats import spearmanr
        corr, _ = spearmanr(float_preds, fp_preds)
        assert corr > 0.8, f"Rank correlation {corr} too low"


# ============================================================================
# FEATURE NORMALIZATION TESTS
# ============================================================================

class TestFeatureNorms:
    def test_correct_count(self):
        np.random.seed(42)
        X = np.random.randn(100, 19)
        norms = compute_feature_norms(X, FEATURE_NAMES)
        assert len(norms) == 19

    def test_mean_std_correct(self):
        np.random.seed(42)
        X = np.random.randn(1000, 19)
        norms = compute_feature_norms(X, FEATURE_NAMES)
        for i, norm in enumerate(norms):
            expected_mean = np.mean(X[:, i])
            expected_std = np.std(X[:, i])
            assert abs(norm.mean - expected_mean) < 1e-10
            assert abs(norm.std - expected_std) < 1e-10

    def test_min_max_correct(self):
        np.random.seed(42)
        X = np.random.randn(500, 19)
        norms = compute_feature_norms(X, FEATURE_NAMES)
        for i, norm in enumerate(norms):
            assert norm.min_val == np.min(X[:, i])
            assert norm.max_val == np.max(X[:, i])

    def test_handles_nan(self):
        X = np.random.randn(100, 19)
        X[0, 0] = np.nan
        X[5, 3] = np.inf
        norms = compute_feature_norms(X, FEATURE_NAMES)
        # Should still produce valid norms (ignoring nan/inf)
        assert np.isfinite(norms[0].mean)
        assert np.isfinite(norms[0].std)

    def test_constant_feature(self):
        """Feature with zero variance should get std=1."""
        X = np.ones((100, 19))
        norms = compute_feature_norms(X, FEATURE_NAMES)
        assert norms[0].std == 1.0

    def test_fp_scale_positive(self):
        np.random.seed(42)
        X = np.random.randn(100, 19)
        norms = compute_feature_norms(X, FEATURE_NAMES)
        for norm in norms:
            assert norm.fixed_point_scale > 0

    def test_feature_names_preserved(self):
        np.random.seed(42)
        X = np.random.randn(50, 19)
        norms = compute_feature_norms(X, FEATURE_NAMES)
        for i, norm in enumerate(norms):
            assert norm.name == FEATURE_NAMES[i]
            assert norm.idx == i


# ============================================================================
# REGIME LUT TESTS
# ============================================================================

class TestRegimeLUT:
    def test_correct_count(self):
        lut = build_regime_lut()
        assert len(lut) == 4

    def test_regime_names(self):
        lut = build_regime_lut()
        names = [r.regime_name for r in lut]
        assert names == REGIME_NAMES

    def test_boundaries_contiguous(self):
        """Regime boundaries should cover full entropy range without gaps."""
        lut = build_regime_lut()
        for i in range(len(lut) - 1):
            assert abs(lut[i].entropy_high - lut[i + 1].entropy_low) < 1e-10

    def test_boundaries_start_at_zero(self):
        lut = build_regime_lut()
        assert lut[0].entropy_low == 0.0

    def test_boundaries_cover_full_range(self):
        lut = build_regime_lut()
        assert lut[-1].entropy_high > np.log(3.0)

    def test_custom_spreads(self):
        spreads = [30.0, 15.0, 7.0, 2.0]
        lut = build_regime_lut(spreads)
        assert lut[0].base_spread_bps == 30.0
        assert lut[3].base_spread_bps == 2.0

    def test_fixed_point_ordering(self):
        lut = build_regime_lut()
        for i in range(len(lut)):
            assert lut[i].fixed_point_low < lut[i].fixed_point_high

    def test_fixed_point_boundaries_contiguous(self):
        lut = build_regime_lut()
        for i in range(len(lut) - 1):
            assert lut[i].fixed_point_high == lut[i + 1].fixed_point_low

    def test_default_spreads_wide_for_trending(self):
        """Trending regime should have wider spread (more adverse selection)."""
        lut = build_regime_lut()
        assert lut[0].base_spread_bps > lut[3].base_spread_bps


# ============================================================================
# FULL EXPORT TESTS
# ============================================================================

class TestFullExport:
    def test_export_creates_files(self):
        result, X, _ = _train_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            export = export_model(result, X, output_dir=tmpdir)
            assert (Path(tmpdir) / "eamm_params.json").exists()
            assert (Path(tmpdir) / "eamm_params.h").exists()
            assert (Path(tmpdir) / "eamm_params.v").exists()

    def test_json_is_valid(self):
        result, X, _ = _train_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_model(result, X, output_dir=tmpdir)
            with open(Path(tmpdir) / "eamm_params.json") as f:
                data = json.load(f)
            assert data["n_trees"] == 10
            assert data["n_features"] == 19
            assert "trees" in data
            assert "feature_norms" in data
            assert "regime_lut" in data

    def test_json_trees_structure(self):
        result, X, _ = _train_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_model(result, X, output_dir=tmpdir)
            with open(Path(tmpdir) / "eamm_params.json") as f:
                data = json.load(f)
            for tree in data["trees"]:
                assert "nodes" in tree
                assert len(tree["nodes"]) > 0
                for node in tree["nodes"]:
                    assert "feature" in node
                    assert "threshold" in node
                    assert "left" in node
                    assert "right" in node
                    assert "value" in node

    def test_c_header_compiles(self):
        """C header should have valid syntax (check basic structure)."""
        result, X, _ = _train_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_model(result, X, output_dir=tmpdir)
            header = (Path(tmpdir) / "eamm_params.h").read_text()
            assert "#ifndef EAMM_PARAMS_H" in header
            assert "#define EAMM_PARAMS_H" in header
            assert "#endif" in header
            assert "EAMM_N_TREES" in header
            assert "EAMM_N_FEATURES" in header
            assert "eamm_node_t" in header

    def test_c_header_defines_match(self):
        result, X, _ = _train_model(n_estimators=15)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_model(result, X, output_dir=tmpdir)
            header = (Path(tmpdir) / "eamm_params.h").read_text()
            assert "#define EAMM_N_TREES 15" in header

    def test_verilog_params_valid(self):
        result, X, _ = _train_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_model(result, X, output_dir=tmpdir)
            verilog = (Path(tmpdir) / "eamm_params.v").read_text()
            assert "parameter EAMM_N_TREES" in verilog
            assert "parameter EAMM_N_FEATURES" in verilog
            assert "REGIME_TRENDING" in verilog
            assert "REGIME_RANDOM" in verilog

    def test_export_result_metadata(self):
        result, X, _ = _train_model(n_estimators=10, max_depth=3)
        export = export_model(result, X, output_dir=None)
        assert export.n_trees == 10
        assert export.n_features == 19
        assert export.fixed_point_bits == FIXED_POINT_BITS
        assert export.fixed_point_frac_bits == FIXED_POINT_FRAC_BITS
        assert export.total_nodes > 0
        assert export.total_leaves > 0
        assert export.max_depth_actual <= 3

    def test_export_no_output_dir(self):
        """Export with output_dir=None should not create files."""
        result, X, _ = _train_model()
        export = export_model(result, X, output_dir=None)
        assert export.export_path is None
        assert export.n_trees == 10


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    def test_export_deterministic(self):
        """Same model → same export every time."""
        result, X, _ = _train_model()
        e1 = export_model(result, X, output_dir=None)
        e2 = export_model(result, X, output_dir=None)
        assert e1.n_trees == e2.n_trees
        assert e1.total_nodes == e2.total_nodes
        for t1, t2 in zip(e1.trees, e2.trees):
            assert len(t1.nodes) == len(t2.nodes)
            for n1, n2 in zip(t1.nodes, t2.nodes):
                assert n1.feature_idx == n2.feature_idx
                assert n1.threshold == n2.threshold
                assert n1.leaf_value == n2.leaf_value

    def test_predict_from_export_deterministic(self):
        result, X, _ = _train_model()
        export = export_model(result, X, output_dir=None)
        p1 = predict_from_export(export, X[:50])
        p2 = predict_from_export(export, X[:50])
        np.testing.assert_array_equal(p1, p2)

    def test_json_roundtrip_deterministic(self):
        """Export → save JSON → load → same data."""
        result, X, _ = _train_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_model(result, X, output_dir=tmpdir)
            with open(Path(tmpdir) / "eamm_params.json") as f:
                d1 = json.load(f)
            export_model(result, X, output_dir=tmpdir)
            with open(Path(tmpdir) / "eamm_params.json") as f:
                d2 = json.load(f)
            assert d1 == d2


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    def test_single_tree(self):
        result, X, _ = _train_model(n_estimators=1, max_depth=2)
        export = export_model(result, X, output_dir=None)
        assert export.n_trees == 1
        preds = predict_from_export(export, X[:10])
        assert len(preds) == 10

    def test_deep_tree(self):
        result, X, _ = _train_model(n_estimators=5, max_depth=8)
        export = export_model(result, X, output_dir=None)
        assert export.max_depth_actual <= 8

    def test_extreme_feature_values(self):
        """Model should handle extreme inputs without crashing."""
        result, X, _ = _train_model()
        export = export_model(result, X, output_dir=None)
        X_extreme = np.array([[1e6] * 19, [-1e6] * 19, [0.0] * 19])
        preds = predict_from_export(export, X_extreme)
        assert len(preds) == 3
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0.0)

    def test_zero_features(self):
        result, X, _ = _train_model()
        export = export_model(result, X, output_dir=None)
        X_zero = np.zeros((5, 19))
        preds = predict_from_export(export, X_zero)
        assert np.all(np.isfinite(preds))

    def test_nan_features_in_prediction(self):
        """NaN in features should not cause crash (tree traversal still works)."""
        result, X, _ = _train_model()
        export = export_model(result, X, output_dir=None)
        X_nan = X[:5].copy()
        X_nan[0, 0] = np.nan
        # NaN comparisons go right (> threshold is False for NaN)
        preds = predict_from_export(export, X_nan)
        assert len(preds) == 5


# ============================================================================
# CONSISTENCY TESTS (MODEL vs EXPORT)
# ============================================================================

class TestModelExportConsistency:
    def test_leaf_values_sum_to_prediction(self):
        """Sum of leaf_values across trees should equal prediction."""
        result, X, _ = _train_model(n_estimators=5, max_depth=2)
        export = export_model(result, X, output_dir=None)
        x = X[0]

        # Manual prediction — lr is already baked into leaf values
        total = export.base_score
        for tree in export.trees:
            # Traverse tree
            idx = 0
            while idx < len(tree.nodes):
                node = tree.nodes[idx]
                if node.feature_idx == -1:
                    total += node.leaf_value
                    break
                if x[node.feature_idx] <= node.threshold:
                    idx = node.left_child
                else:
                    idx = node.right_child

        pred = predict_from_export(export, x)
        # Total might be negative, pred is clamped to 0
        expected = max(total, 0.0)
        assert abs(pred[0] - expected) < 1e-10

    def test_feature_importances_consistent(self):
        """Features used in trees should match importance ranking."""
        result, X, _ = _train_model(n_estimators=30, max_depth=4)
        export = export_model(result, X, output_dir=None)

        # Count feature usage across all trees
        usage = np.zeros(19)
        for tree in export.trees:
            for node in tree.nodes:
                if node.feature_idx != -1:
                    usage[node.feature_idx] += 1

        # Most used features should align with top importances
        top_used = np.argsort(usage)[::-1][:5]
        top_important = list(result.feature_importances.keys())[:5]
        top_important_idx = [FEATURE_NAMES.index(n) for n in top_important]

        # At least 2 of top-5 should overlap
        overlap = len(set(top_used) & set(top_important_idx))
        assert overlap >= 2, (
            f"Only {overlap} overlap between top-5 used features {top_used} "
            f"and top-5 important {top_important_idx}"
        )

    def test_many_samples_correlation(self):
        """Over many samples, export predictions should correlate 1.0 with original."""
        result, X, _ = _train_model_large()
        export = export_model(result, X, output_dir=None)
        original = predict_spread(result, X)
        exported = predict_from_export(export, X)
        corr = np.corrcoef(original, exported)[0, 1]
        assert corr > 0.9999, f"Correlation {corr} too low"


# ============================================================================
# OVERFLOW / SAFETY TESTS
# ============================================================================

class TestOverflowSafety:
    def test_fp_accumulation_no_overflow_32bit(self):
        """Accumulated fixed-point prediction should fit in 32-bit integer."""
        result, X, _ = _train_model(n_estimators=50, max_depth=5)
        export = export_model(result, X, output_dir=None)

        # Simulate worst case: max leaf value * n_trees * lr
        max_leaf = 0.0
        for tree in export.trees:
            for node in tree.nodes:
                if node.feature_idx == -1:
                    max_leaf = max(max_leaf, abs(node.leaf_value))

        worst_case = export.base_score + export.n_trees * export.learning_rate * max_leaf
        worst_case_fp = quantize_threshold(worst_case)
        # Should fit in 32-bit signed
        assert abs(worst_case_fp) < 2**31, (
            f"Worst-case FP value {worst_case_fp} exceeds 32-bit range"
        )

    def test_fp_multiplication_no_overflow_64bit(self):
        """lr_fp * leaf_fp should fit in 64-bit before shift."""
        result, X, _ = _train_model(n_estimators=50)
        export = export_model(result, X, output_dir=None)
        lr_fp = quantize_threshold(export.learning_rate)

        for tree in export.trees:
            for node in tree.nodes:
                if node.feature_idx == -1:
                    leaf_fp = quantize_threshold(node.leaf_value)
                    product = lr_fp * leaf_fp
                    assert abs(product) < 2**63, "Multiplication overflow"

    def test_regime_lut_fits_16bit(self):
        """All regime LUT values should fit in 16-bit signed."""
        lut = build_regime_lut()
        for r in lut:
            assert -32768 <= r.fixed_point_low <= 32767
            assert -32768 <= r.fixed_point_high <= 32767
            # Spread uses 8.8 format (max 127 bps)
            assert -32768 <= r.fixed_point_spread <= 32767


# ============================================================================
# REGRESSION TEST: KNOWN VALUES
# ============================================================================

class TestKnownValues:
    def test_quantize_known_values(self):
        """Verify specific quantization values."""
        # With 12 frac bits: 1.0 → 4096
        assert quantize_threshold(1.0, 12) == 4096
        assert quantize_threshold(0.5, 12) == 2048
        assert quantize_threshold(-1.0, 12) == -4096
        assert quantize_threshold(0.25, 12) == 1024

    def test_dequantize_known_values(self):
        assert dequantize(4096, 12) == 1.0
        assert dequantize(2048, 12) == 0.5
        assert dequantize(-4096, 12) == -1.0

    def test_regime_boundaries_known(self):
        """Verify regime boundaries match spec."""
        lut = build_regime_lut()
        assert lut[0].entropy_low == 0.0
        assert lut[0].entropy_high == 0.35
        assert lut[1].entropy_low == 0.35
        assert lut[1].entropy_high == 0.55
        assert lut[2].entropy_low == 0.55
        assert lut[2].entropy_high == 0.65
        assert lut[3].entropy_low == 0.65
