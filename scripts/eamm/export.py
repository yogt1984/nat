"""
EAMM Module 8: Parameter Export for Online/FPGA Deployment

Exports trained EAMM models into deterministic, low-latency formats:
  - Decision tree extraction (if/else chains)
  - Fixed-point quantization (16-bit integer arithmetic)
  - Feature normalization parameters
  - Regime boundary lookup tables
  - C header generation for FPGA/embedded deployment

Reference: EAMM_SPEC.md §1.10
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import lightgbm as lgb

from .train import TrainResult
from .regime_analysis import REGIME_BOUNDARIES, REGIME_NAMES


# Fixed-point configuration
FIXED_POINT_BITS = 16
FIXED_POINT_FRAC_BITS = 12  # 4.12 format: 4 integer bits, 12 fractional
FIXED_POINT_SCALE = 2 ** FIXED_POINT_FRAC_BITS


@dataclass
class TreeNode:
    """A single decision tree node."""
    feature_idx: int  # -1 for leaf
    threshold: float
    left_child: int  # index or -1
    right_child: int  # index or -1
    leaf_value: float  # only valid if feature_idx == -1


@dataclass
class ExportedTree:
    """A single exported decision tree."""
    tree_idx: int
    nodes: List[TreeNode]
    n_leaves: int
    max_depth: int


@dataclass
class FeatureNormParams:
    """Normalization parameters for a single feature."""
    name: str
    idx: int
    mean: float
    std: float
    min_val: float
    max_val: float
    fixed_point_scale: int
    fixed_point_offset: int


@dataclass
class RegimeLUT:
    """Regime lookup table entry."""
    regime_name: str
    entropy_low: float
    entropy_high: float
    base_spread_bps: float
    fixed_point_low: int
    fixed_point_high: int
    fixed_point_spread: int


@dataclass
class ExportResult:
    """Complete export result."""
    trees: List[ExportedTree]
    n_trees: int
    n_features: int
    feature_names: List[str]
    feature_norms: List[FeatureNormParams]
    regime_lut: List[RegimeLUT]
    learning_rate: float
    base_score: float
    fixed_point_bits: int
    fixed_point_frac_bits: int
    max_depth_actual: int
    total_nodes: int
    total_leaves: int
    export_path: Optional[str] = None


def extract_trees(train_result: TrainResult) -> List[ExportedTree]:
    """Extract decision trees from a trained LightGBM model.

    Parameters
    ----------
    train_result : TrainResult
        Trained EAMM model.

    Returns
    -------
    List[ExportedTree]
        Extracted trees with nodes in breadth-first order.
    """
    model = train_result.model
    booster = model.booster_

    # Get model dump as JSON
    model_dump = booster.dump_model()
    tree_info = model_dump["tree_info"]

    exported_trees = []
    for t_idx, tree in enumerate(tree_info):
        nodes = []
        max_depth = _parse_tree_recursive(tree["tree_structure"], nodes, depth=0)
        n_leaves = sum(1 for n in nodes if n.feature_idx == -1)
        exported_trees.append(ExportedTree(
            tree_idx=t_idx,
            nodes=nodes,
            n_leaves=n_leaves,
            max_depth=max_depth,
        ))

    return exported_trees


def _parse_tree_recursive(node_dict: dict, nodes: List[TreeNode], depth: int) -> int:
    """Recursively parse a LightGBM tree node into flat list."""
    if "leaf_value" in node_dict:
        # Leaf node
        nodes.append(TreeNode(
            feature_idx=-1,
            threshold=0.0,
            left_child=-1,
            right_child=-1,
            leaf_value=float(node_dict["leaf_value"]),
        ))
        return depth

    # Internal node
    feature_idx = int(node_dict["split_feature"])
    threshold = float(node_dict["threshold"])
    current_idx = len(nodes)

    # Placeholder — will be filled after children are parsed
    nodes.append(TreeNode(
        feature_idx=feature_idx,
        threshold=threshold,
        left_child=-1,
        right_child=-1,
        leaf_value=0.0,
    ))

    # Parse left child
    left_idx = len(nodes)
    left_depth = _parse_tree_recursive(node_dict["left_child"], nodes, depth + 1)
    nodes[current_idx].left_child = left_idx

    # Parse right child
    right_idx = len(nodes)
    right_depth = _parse_tree_recursive(node_dict["right_child"], nodes, depth + 1)
    nodes[current_idx].right_child = right_idx

    return max(left_depth, right_depth)


def compute_feature_norms(
    X: np.ndarray,
    feature_names: List[str],
) -> List[FeatureNormParams]:
    """Compute normalization parameters from training data.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Training feature matrix.
    feature_names : list of str
        Feature names.

    Returns
    -------
    List[FeatureNormParams]
    """
    norms = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        valid = col[np.isfinite(col)]
        if len(valid) == 0:
            mean, std, min_val, max_val = 0.0, 1.0, 0.0, 0.0
        else:
            mean = float(np.mean(valid))
            std = float(np.std(valid))
            if std < 1e-12:
                std = 1.0
            min_val = float(np.min(valid))
            max_val = float(np.max(valid))

        # Fixed-point: scale to fit in 16-bit signed integer
        # Map [min, max] → [0, 2^15 - 1]
        range_val = max_val - min_val
        if range_val < 1e-12:
            range_val = 1.0
        fp_scale = int((2**15 - 1) / range_val)
        fp_offset = int(-min_val * fp_scale)

        norms.append(FeatureNormParams(
            name=name,
            idx=i,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            fixed_point_scale=fp_scale,
            fixed_point_offset=fp_offset,
        ))
    return norms


def build_regime_lut(
    optimal_spreads_per_regime: Optional[List[float]] = None,
) -> List[RegimeLUT]:
    """Build regime lookup table for fallback spread selection.

    Parameters
    ----------
    optimal_spreads_per_regime : list of float, optional
        Mean optimal spread per regime from training data.
        If None, uses conservative defaults.

    Returns
    -------
    List[RegimeLUT]
    """
    # Default spreads if not provided (conservative values)
    if optimal_spreads_per_regime is None:
        optimal_spreads_per_regime = [20.0, 10.0, 5.0, 3.0]  # wide→narrow

    lut = []
    boundaries = REGIME_BOUNDARIES
    for i, name in enumerate(REGIME_NAMES):
        low = boundaries[i]
        high = boundaries[i + 1]
        spread = optimal_spreads_per_regime[i] if i < len(optimal_spreads_per_regime) else 5.0

        # Fixed-point conversion
        # Entropy boundaries use full 4.12 scale (values < 1.1)
        fp_low = int(low * FIXED_POINT_SCALE)
        fp_high = int(high * FIXED_POINT_SCALE)
        # Spread in bps uses 8.8 scale (values up to 127 bps)
        fp_spread = int(spread * 256)

        lut.append(RegimeLUT(
            regime_name=name,
            entropy_low=low,
            entropy_high=high,
            base_spread_bps=spread,
            fixed_point_low=fp_low,
            fixed_point_high=fp_high,
            fixed_point_spread=fp_spread,
        ))
    return lut


def quantize_threshold(value: float, frac_bits: int = FIXED_POINT_FRAC_BITS) -> int:
    """Convert a floating-point threshold to fixed-point integer.

    Parameters
    ----------
    value : float
        The threshold value.
    frac_bits : int
        Number of fractional bits.

    Returns
    -------
    int
        Fixed-point representation.
    """
    scale = 2 ** frac_bits
    return int(round(value * scale))


def dequantize(fixed_val: int, frac_bits: int = FIXED_POINT_FRAC_BITS) -> float:
    """Convert fixed-point integer back to float."""
    scale = 2 ** frac_bits
    return fixed_val / scale


def export_model(
    train_result: TrainResult,
    X_train: np.ndarray,
    optimal_spreads_per_regime: Optional[List[float]] = None,
    output_dir: str = "export",
) -> ExportResult:
    """Export a trained EAMM model for deployment.

    Parameters
    ----------
    train_result : TrainResult
        Trained model.
    X_train : np.ndarray
        Training data (for computing normalization params).
    optimal_spreads_per_regime : list of float, optional
        Mean optimal spread per regime.
    output_dir : str
        Directory to write export files.

    Returns
    -------
    ExportResult
    """
    # Extract trees
    trees = extract_trees(train_result)

    # Compute feature norms
    feature_norms = compute_feature_norms(X_train, train_result.feature_names)

    # Build regime LUT
    regime_lut = build_regime_lut(optimal_spreads_per_regime)

    # Get model metadata
    booster = train_result.model.booster_
    model_dump = booster.dump_model()

    # Learning rate from model params
    lr = model_dump.get("average_output", False)
    # Extract from parameters
    params = booster.params
    learning_rate = float(params.get("learning_rate", 0.05))

    # Base score (initial prediction)
    base_score = 0.0  # LightGBM uses 0 as base

    # Compute stats
    total_nodes = sum(len(t.nodes) for t in trees)
    total_leaves = sum(t.n_leaves for t in trees)
    max_depth_actual = max(t.max_depth for t in trees) if trees else 0

    result = ExportResult(
        trees=trees,
        n_trees=len(trees),
        n_features=len(train_result.feature_names),
        feature_names=train_result.feature_names,
        feature_norms=feature_norms,
        regime_lut=regime_lut,
        learning_rate=learning_rate,
        base_score=base_score,
        fixed_point_bits=FIXED_POINT_BITS,
        fixed_point_frac_bits=FIXED_POINT_FRAC_BITS,
        max_depth_actual=max_depth_actual,
        total_nodes=total_nodes,
        total_leaves=total_leaves,
    )

    # Write files
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # JSON export (portable)
        _write_json_export(result, out_path / "eamm_params.json")

        # C header
        _write_c_header(result, out_path / "eamm_params.h")

        # Verilog parameters
        _write_verilog_params(result, out_path / "eamm_params.v")

        result.export_path = str(out_path)

    return result


def predict_from_export(
    export: ExportResult,
    features: np.ndarray,
) -> np.ndarray:
    """Predict using exported trees (reference implementation).

    This mirrors what the FPGA/C code would do — pure integer arithmetic
    on quantized thresholds, summing leaf values.

    Parameters
    ----------
    export : ExportResult
        Exported model.
    features : np.ndarray, shape (N, D) or (D,)
        Raw features (not normalized).

    Returns
    -------
    np.ndarray, shape (N,)
        Predicted spread in bps.
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)

    N = features.shape[0]
    predictions = np.full(N, export.base_score)

    for tree in export.trees:
        for i in range(N):
            leaf_val = _traverse_tree(tree.nodes, features[i])
            predictions[i] += leaf_val  # LightGBM bakes lr into leaf values

    # Ensure non-negative spread
    predictions = np.maximum(predictions, 0.0)
    return predictions


def _traverse_tree(nodes: List[TreeNode], x: np.ndarray) -> float:
    """Traverse a single tree to get leaf value."""
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.feature_idx == -1:
            return node.leaf_value
        if x[node.feature_idx] <= node.threshold:
            idx = node.left_child
        else:
            idx = node.right_child
    return 0.0  # should not reach here


def predict_fixed_point(
    export: ExportResult,
    features: np.ndarray,
) -> np.ndarray:
    """Predict using fixed-point arithmetic (FPGA simulation).

    Quantizes all thresholds and features to fixed-point, performs
    comparisons in integer domain, accumulates leaf values in fixed-point.

    Parameters
    ----------
    export : ExportResult
    features : np.ndarray, shape (N, D)

    Returns
    -------
    np.ndarray, shape (N,) — predictions in bps (dequantized)
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)

    N = features.shape[0]
    frac_bits = export.fixed_point_frac_bits
    scale = 2 ** frac_bits

    # Quantize base score
    fp_base = quantize_threshold(export.base_score, frac_bits)
    # Quantize learning rate
    fp_lr = quantize_threshold(export.learning_rate, frac_bits)

    predictions_fp = np.full(N, fp_base, dtype=np.int64)

    for tree in export.trees:
        # Quantize all thresholds
        fp_thresholds = [quantize_threshold(n.threshold, frac_bits) for n in tree.nodes]

        for i in range(N):
            # Quantize features for this sample
            fp_features = [quantize_threshold(float(features[i, d]), frac_bits)
                           for d in range(features.shape[1])]

            # Traverse tree in fixed-point
            leaf_val = _traverse_tree_fixed(tree.nodes, fp_thresholds, fp_features)

            # Accumulate leaf value directly (lr already baked in)
            predictions_fp[i] += leaf_val

    # Dequantize
    predictions = predictions_fp.astype(np.float64) / scale
    predictions = np.maximum(predictions, 0.0)
    return predictions


def _traverse_tree_fixed(
    nodes: List[TreeNode],
    fp_thresholds: List[int],
    fp_features: List[int],
) -> int:
    """Traverse tree using fixed-point comparisons."""
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.feature_idx == -1:
            return quantize_threshold(node.leaf_value)
        if fp_features[node.feature_idx] <= fp_thresholds[idx]:
            idx = node.left_child
        else:
            idx = node.right_child
    return 0


def _write_json_export(result: ExportResult, path: Path):
    """Write portable JSON export."""
    data = {
        "format_version": "1.0",
        "model_type": "eamm_lgbm_export",
        "n_trees": result.n_trees,
        "n_features": result.n_features,
        "learning_rate": result.learning_rate,
        "base_score": result.base_score,
        "fixed_point_bits": result.fixed_point_bits,
        "fixed_point_frac_bits": result.fixed_point_frac_bits,
        "feature_names": result.feature_names,
        "feature_norms": [
            {
                "name": fn.name,
                "idx": fn.idx,
                "mean": fn.mean,
                "std": fn.std,
                "min": fn.min_val,
                "max": fn.max_val,
                "fp_scale": fn.fixed_point_scale,
                "fp_offset": fn.fixed_point_offset,
            }
            for fn in result.feature_norms
        ],
        "regime_lut": [
            {
                "name": r.regime_name,
                "entropy_low": r.entropy_low,
                "entropy_high": r.entropy_high,
                "spread_bps": r.base_spread_bps,
                "fp_low": r.fixed_point_low,
                "fp_high": r.fixed_point_high,
                "fp_spread": r.fixed_point_spread,
            }
            for r in result.regime_lut
        ],
        "trees": [
            {
                "idx": t.tree_idx,
                "n_leaves": t.n_leaves,
                "max_depth": t.max_depth,
                "nodes": [
                    {
                        "feature": n.feature_idx,
                        "threshold": n.threshold,
                        "left": n.left_child,
                        "right": n.right_child,
                        "value": n.leaf_value,
                    }
                    for n in t.nodes
                ],
            }
            for t in result.trees
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_c_header(result: ExportResult, path: Path):
    """Generate C header file for embedded/FPGA deployment."""
    lines = [
        "/* EAMM Parameter Export — Auto-generated */",
        "#ifndef EAMM_PARAMS_H",
        "#define EAMM_PARAMS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define EAMM_N_TREES {result.n_trees}",
        f"#define EAMM_N_FEATURES {result.n_features}",
        f"#define EAMM_FP_BITS {result.fixed_point_bits}",
        f"#define EAMM_FP_FRAC_BITS {result.fixed_point_frac_bits}",
        f"#define EAMM_FP_SCALE (1 << EAMM_FP_FRAC_BITS)",
        f"#define EAMM_LEARNING_RATE_FP {quantize_threshold(result.learning_rate)}",
        f"#define EAMM_BASE_SCORE_FP {quantize_threshold(result.base_score)}",
        f"#define EAMM_MAX_DEPTH {result.max_depth_actual}",
        f"#define EAMM_TOTAL_NODES {result.total_nodes}",
        "",
        "/* Regime lookup table */",
        f"#define EAMM_N_REGIMES {len(result.regime_lut)}",
        "",
    ]

    # Regime LUT
    lines.append("static const struct {")
    lines.append("    int16_t entropy_low_fp;")
    lines.append("    int16_t entropy_high_fp;")
    lines.append("    int16_t spread_bps_fp;")
    lines.append(f"}} eamm_regime_lut[EAMM_N_REGIMES] = {{")
    for r in result.regime_lut:
        lines.append(f"    /* {r.regime_name:15s} */ "
                     f"{{{r.fixed_point_low:6d}, {r.fixed_point_high:6d}, {r.fixed_point_spread:6d}}},")
    lines.append("};")
    lines.append("")

    # Feature normalization
    lines.append("/* Feature normalization: fp_value = (raw - offset) * scale */")
    lines.append("static const struct {")
    lines.append("    int32_t scale;")
    lines.append("    int32_t offset;")
    lines.append(f"}} eamm_feature_norm[EAMM_N_FEATURES] = {{")
    for fn in result.feature_norms:
        lines.append(f"    /* {fn.name:15s} */ {{{fn.fixed_point_scale:10d}, {fn.fixed_point_offset:10d}}},")
    lines.append("};")
    lines.append("")

    # Tree nodes (flattened)
    lines.append("/* Tree node: feature_idx, threshold_fp, left_child, right_child, leaf_value_fp */")
    lines.append("typedef struct {")
    lines.append("    int8_t feature_idx;  /* -1 = leaf */")
    lines.append("    int16_t threshold_fp;")
    lines.append("    int16_t left_child;")
    lines.append("    int16_t right_child;")
    lines.append("    int16_t leaf_value_fp;")
    lines.append("} eamm_node_t;")
    lines.append("")

    # Emit tree data
    total_offset = 0
    lines.append(f"/* Tree start offsets */")
    lines.append(f"static const int16_t eamm_tree_offsets[EAMM_N_TREES] = {{")
    offsets = []
    for t in result.trees:
        offsets.append(total_offset)
        total_offset += len(t.nodes)
    lines.append("    " + ", ".join(str(o) for o in offsets))
    lines.append("};")
    lines.append("")

    lines.append(f"static const eamm_node_t eamm_nodes[{result.total_nodes}] = {{")
    for t in result.trees:
        lines.append(f"    /* Tree {t.tree_idx} ({len(t.nodes)} nodes, depth {t.max_depth}) */")
        for n in t.nodes:
            th_fp = quantize_threshold(n.threshold) if n.feature_idx != -1 else 0
            lv_fp = quantize_threshold(n.leaf_value) if n.feature_idx == -1 else 0
            lines.append(f"    {{{n.feature_idx:3d}, {th_fp:6d}, "
                         f"{n.left_child:4d}, {n.right_child:4d}, {lv_fp:6d}}},")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* EAMM_PARAMS_H */")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_verilog_params(result: ExportResult, path: Path):
    """Generate Verilog parameter file for FPGA."""
    lines = [
        "/* EAMM Verilog Parameters — Auto-generated */",
        "",
        f"parameter EAMM_N_TREES = {result.n_trees};",
        f"parameter EAMM_N_FEATURES = {result.n_features};",
        f"parameter EAMM_FP_BITS = {result.fixed_point_bits};",
        f"parameter EAMM_FP_FRAC_BITS = {result.fixed_point_frac_bits};",
        f"parameter EAMM_MAX_DEPTH = {result.max_depth_actual};",
        f"parameter EAMM_LR_FP = {quantize_threshold(result.learning_rate)};",
        "",
        "/* Regime boundaries (fixed-point) */",
    ]
    for i, r in enumerate(result.regime_lut):
        lines.append(f"parameter REGIME_{r.regime_name}_LOW = {r.fixed_point_low};")
        lines.append(f"parameter REGIME_{r.regime_name}_HIGH = {r.fixed_point_high};")
        lines.append(f"parameter REGIME_{r.regime_name}_SPREAD = {r.fixed_point_spread};")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
