"""
Feature vector definitions for NAT cluster analysis.

Maps the 14 feature categories from the Rust ingestor into named vectors.
Each vector groups semantically related features for independent clustering.

Usage:
    from cluster_pipeline.config import FEATURE_VECTORS, extract_vector, list_vectors

    cols = extract_vector(df, "entropy")
    X = df[cols].to_numpy()
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Feature Vector Definitions
#
# Each vector is defined by either:
#   - "prefixes": list of column prefixes (matched with startswith)
#   - "columns": explicit list of column names (for disambiguation)
#
# When both are present, "columns" takes precedence.
# ---------------------------------------------------------------------------

FEATURE_VECTORS: Dict[str, dict] = {
    # ---- Information-theoretic ----
    "entropy": {
        "description": "Tick entropy, permutation entropy, conditional entropy",
        "use_case": "Regime detection, predictability",
        "expected_dim": 24,
        "prefixes": ["ent_"],
        "columns": [
            "ent_permutation_returns_8",
            "ent_permutation_returns_16",
            "ent_permutation_returns_32",
            "ent_permutation_imbalance_16",
            "ent_spread_dispersion",
            "ent_volume_dispersion",
            "ent_book_shape",
            "ent_trade_size_dispersion",
            "ent_rate_of_change_5s",
            "ent_zscore_1m",
            "ent_tick_1s",
            "ent_tick_5s",
            "ent_tick_10s",
            "ent_tick_15s",
            "ent_tick_30s",
            "ent_tick_1m",
            "ent_tick_15m",
            "ent_vol_tick_1s",
            "ent_vol_tick_5s",
            "ent_vol_tick_10s",
            "ent_vol_tick_15s",
            "ent_vol_tick_30s",
            "ent_vol_tick_1m",
            "ent_vol_tick_15m",
        ],
    },

    # ---- Trend & Persistence ----
    "trend": {
        "description": "Momentum, monotonicity, Hurst exponent, R-squared",
        "use_case": "Persistence, mean-reversion detection",
        "expected_dim": 15,
        "prefixes": ["trend_"],
        "columns": [
            "trend_momentum_60",
            "trend_momentum_r2_60",
            "trend_monotonicity_60",
            "trend_momentum_300",
            "trend_momentum_r2_300",
            "trend_monotonicity_300",
            "trend_hurst_300",
            "trend_momentum_600",
            "trend_momentum_r2_600",
            "trend_monotonicity_600",
            "trend_hurst_600",
            "trend_ma_crossover",
            "trend_ma_crossover_norm",
            "trend_ema_short",
            "trend_ema_long",
        ],
    },

    # ---- Price Impact ----
    "illiquidity": {
        "description": "Kyle lambda, Amihud, Hasbrouck, Roll spread",
        "use_case": "Price impact, informed flow detection",
        "expected_dim": 12,
        "prefixes": ["illiq_"],
        "columns": [
            "illiq_kyle_100",
            "illiq_amihud_100",
            "illiq_hasbrouck_100",
            "illiq_roll_100",
            "illiq_kyle_500",
            "illiq_amihud_500",
            "illiq_hasbrouck_500",
            "illiq_roll_500",
            "illiq_kyle_ratio",
            "illiq_amihud_ratio",
            "illiq_composite",
            "illiq_trade_count",
        ],
    },

    # ---- Order Flow Toxicity ----
    "toxicity": {
        "description": "VPIN, adverse selection, effective/realized spread",
        "use_case": "Order flow toxicity, informed trading detection",
        "expected_dim": 10,
        "prefixes": ["toxic_"],
        "columns": [
            "toxic_vpin_10",
            "toxic_vpin_50",
            "toxic_vpin_roc",
            "toxic_adverse_selection",
            "toxic_effective_spread",
            "toxic_realized_spread",
            "toxic_flow_imbalance",
            "toxic_flow_imbalance_abs",
            "toxic_index",
            "toxic_trade_count",
        ],
    },

    # ---- Order Book Shape ----
    "orderflow": {
        "description": "Imbalance (L1/L5/L10), pressure, depth-weighted",
        "use_case": "Directional conviction from order book",
        "expected_dim": 8,
        "prefixes": ["imbalance_"],
        "columns": [
            "imbalance_qty_l1",
            "imbalance_qty_l5",
            "imbalance_qty_l10",
            "imbalance_orders_l5",
            "imbalance_notional_l5",
            "imbalance_depth_weighted",
            "imbalance_pressure_bid",
            "imbalance_pressure_ask",
        ],
    },

    # ---- Volatility ----
    "volatility": {
        "description": "Realized vol, Parkinson, spread vol, vol ratio",
        "use_case": "Risk regime identification",
        "expected_dim": 8,
        "prefixes": ["vol_"],
        "columns": [
            "vol_returns_1m",
            "vol_returns_5m",
            "vol_parkinson_5m",
            "vol_spread_mean_1m",
            "vol_spread_std_1m",
            "vol_midprice_std_1m",
            "vol_ratio_short_long",
            "vol_zscore",
        ],
    },

    # ---- Position Concentration (Hyperliquid-unique) ----
    # Explicit columns because "whale_" prefix overlaps with whale_flow vector.
    "concentration": {
        "description": "Gini, HHI, Top-K concentration, Theil index, whale ratios",
        "use_case": "Position crowding, concentration risk",
        "expected_dim": 15,
        "prefixes": [],
        "columns": [
            "top5_concentration",
            "top10_concentration",
            "top20_concentration",
            "top50_concentration",
            "herfindahl_index",
            "gini_coefficient",
            "theil_index",
            "whale_retail_ratio",
            "whale_fraction",
            "whale_avg_size_ratio",
            "concentration_change_1h",
            "hhi_roc",
            "concentration_trend",
            "position_count",
            "whale_position_count",
        ],
    },

    # ---- Whale Flow (Hyperliquid-unique) ----
    # Explicit columns to avoid overlap with concentration.
    "whale": {
        "description": "Net flow (1h/4h/24h), momentum, intensity, buy ratio",
        "use_case": "Smart money tracking",
        "expected_dim": 12,
        "prefixes": [],
        "columns": [
            "whale_net_flow_1h",
            "whale_net_flow_4h",
            "whale_net_flow_24h",
            "whale_flow_normalized_1h",
            "whale_flow_normalized_4h",
            "whale_flow_momentum",
            "whale_flow_intensity",
            "whale_flow_roc",
            "whale_buy_ratio",
            "whale_directional_agreement",
            "active_whale_count",
            "whale_total_activity",
        ],
    },

    # ---- Liquidation Risk (Hyperliquid-unique) ----
    "liquidation": {
        "description": "Risk mapping above/below price, cascade probability",
        "use_case": "Liquidation cascade prediction",
        "expected_dim": 13,
        "prefixes": ["liquidation_"],
        "columns": [
            "liquidation_risk_above_1pct",
            "liquidation_risk_above_2pct",
            "liquidation_risk_above_5pct",
            "liquidation_risk_above_10pct",
            "liquidation_risk_below_1pct",
            "liquidation_risk_below_2pct",
            "liquidation_risk_below_5pct",
            "liquidation_risk_below_10pct",
            "liquidation_asymmetry",
            "liquidation_intensity",
            "positions_at_risk_count",
            "largest_position_at_risk",
            "nearest_cluster_distance",
        ],
    },

    # ---- Raw Microstructure ----
    "raw": {
        "description": "Midprice, microprice, spread, depth",
        "use_case": "Microstructure baseline",
        "expected_dim": 10,
        "prefixes": ["raw_"],
        "columns": [
            "raw_midprice",
            "raw_spread",
            "raw_spread_bps",
            "raw_microprice",
            "raw_bid_depth_5",
            "raw_ask_depth_5",
            "raw_bid_depth_10",
            "raw_ask_depth_10",
            "raw_bid_orders_5",
            "raw_ask_orders_5",
        ],
    },

    # ---- Trade Flow ----
    "flow": {
        "description": "Volume, VWAP, aggressor ratio, trade intensity",
        "use_case": "Execution pattern analysis",
        "expected_dim": 12,
        "prefixes": ["flow_"],
        "columns": [
            "flow_count_1s",
            "flow_count_5s",
            "flow_count_30s",
            "flow_volume_1s",
            "flow_volume_5s",
            "flow_volume_30s",
            "flow_aggressor_ratio_5s",
            "flow_aggressor_ratio_30s",
            "flow_vwap_5s",
            "flow_vwap_deviation",
            "flow_avg_trade_size_30s",
            "flow_intensity",
        ],
    },

    # ---- Market Context ----
    "context": {
        "description": "Funding rate, open interest, premium, basis",
        "use_case": "Market condition assessment",
        "expected_dim": 9,
        "prefixes": ["ctx_"],
        "columns": [
            "ctx_funding_rate",
            "ctx_funding_zscore",
            "ctx_open_interest",
            "ctx_oi_change_5m",
            "ctx_oi_change_pct_5m",
            "ctx_premium_bps",
            "ctx_volume_24h",
            "ctx_volume_ratio",
            "ctx_mark_oracle_divergence",
        ],
    },

    # ---- Derived / Cross-Domain ----
    "derived": {
        "description": "Regime indicators, composite signals, feature interactions",
        "use_case": "Combined alpha signals",
        "expected_dim": 15,
        "prefixes": ["derived_"],
        "columns": [
            "derived_entropy_trend_interaction",
            "derived_entropy_trend_zscore",
            "derived_trend_strength_60",
            "derived_trend_strength_300",
            "derived_trend_strength_ratio",
            "derived_entropy_volatility_ratio",
            "derived_regime_type_score",
            "derived_illiquidity_trend",
            "derived_informed_trend_score",
            "derived_toxicity_regime",
            "derived_toxic_chop_score",
            "derived_trend_strength_roc",
            "derived_entropy_momentum",
            "derived_regime_indicator",
            "derived_regime_confidence",
        ],
    },

    # ---- Regime Detection ----
    "regime": {
        "description": "Absorption, divergence, churn, range position",
        "use_case": "Accumulation/distribution detection",
        "expected_dim": 20,
        "prefixes": ["regime_"],
        "columns": [
            "regime_absorption_1h",
            "regime_absorption_4h",
            "regime_absorption_24h",
            "regime_absorption_zscore",
            "regime_divergence_1h",
            "regime_divergence_4h",
            "regime_divergence_24h",
            "regime_divergence_zscore",
            "regime_kyle_lambda",
            "regime_churn_1h",
            "regime_churn_4h",
            "regime_churn_24h",
            "regime_churn_zscore",
            "regime_range_pos_4h",
            "regime_range_pos_24h",
            "regime_range_pos_1w",
            "regime_range_width_24h",
            "regime_accumulation_score",
            "regime_distribution_score",
            "regime_clarity",
        ],
    },
}


# ---------------------------------------------------------------------------
# Composite Vectors — unions of base vectors
# ---------------------------------------------------------------------------

COMPOSITE_VECTORS: Dict[str, dict] = {
    "micro": {
        "description": "Core microstructure state (entropy + volatility + flow)",
        "use_case": "Short-horizon regime detection",
        "components": ["entropy", "volatility", "flow"],
    },
    "macro": {
        "description": "Higher-level market structure (regime + whale + context)",
        "use_case": "Long-horizon regime detection",
        "components": ["regime", "whale", "context"],
    },
    "full": {
        "description": "All available numeric features",
        "use_case": "Baseline full-dimensional clustering",
        "components": list(FEATURE_VECTORS.keys()),
    },
}


# ---------------------------------------------------------------------------
# Meta columns — never included in feature vectors
# ---------------------------------------------------------------------------

META_COLUMNS = {"timestamp", "timestamp_ns", "symbol", "sequence_id", "datetime"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_vector_columns(vector_name: str) -> List[str]:
    """Return the canonical column list for a vector (before checking data availability)."""
    if vector_name in FEATURE_VECTORS:
        return list(FEATURE_VECTORS[vector_name]["columns"])
    if vector_name in COMPOSITE_VECTORS:
        cols = []
        for comp in COMPOSITE_VECTORS[vector_name]["components"]:
            cols.extend(FEATURE_VECTORS[comp]["columns"])
        return cols
    raise ValueError(
        f"Unknown vector '{vector_name}'. "
        f"Available: {list(FEATURE_VECTORS.keys()) + list(COMPOSITE_VECTORS.keys())}"
    )


def extract_vector(
    df,
    vector_name: str,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Extract feature columns for a vector from a dataframe.

    Args:
        df: pandas or polars DataFrame
        vector_name: one of the vector names from FEATURE_VECTORS or COMPOSITE_VECTORS
        strict: if True, raise error when any expected column is missing

    Returns:
        (found_columns, missing_columns)
    """
    expected = get_vector_columns(vector_name)

    # Get available columns from the dataframe
    if hasattr(df, "columns"):
        available = set(df.columns)
    else:
        raise TypeError(f"Expected DataFrame, got {type(df)}")

    found = [c for c in expected if c in available]
    missing = [c for c in expected if c not in available]

    if strict and missing:
        raise ValueError(
            f"Vector '{vector_name}': missing {len(missing)} columns: {missing}"
        )

    return found, missing


def extract_vector_data(
    df,
    vector_name: str,
    dropna_thresh: float = 0.8,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract feature matrix for a vector, handling NaN and scaling.

    Args:
        df: pandas DataFrame
        vector_name: vector name
        dropna_thresh: drop rows where fraction of valid values < this

    Returns:
        (X, column_names) where X is a numpy array
    """
    found, missing = extract_vector(df, vector_name)
    if not found:
        raise ValueError(f"Vector '{vector_name}': no columns found in data")

    X = df[found].copy()

    # Drop rows with too many NaN
    min_valid = int(len(found) * dropna_thresh)
    mask = X.notna().sum(axis=1) >= min_valid
    X = X.loc[mask]

    # Fill remaining NaN with column median
    for col in X.columns:
        median_val = X[col].median()
        if np.isnan(median_val):
            median_val = 0.0
        X[col] = X[col].fillna(median_val)

    return X.values, found


def list_vectors(df=None) -> List[dict]:
    """
    List all available feature vectors with metadata.

    If df is provided, also shows how many columns are present in the data.
    """
    result = []

    for name, spec in FEATURE_VECTORS.items():
        info = {
            "name": name,
            "description": spec["description"],
            "use_case": spec["use_case"],
            "expected_dim": spec["expected_dim"],
        }
        if df is not None:
            found, missing = extract_vector(df, name)
            info["found_dim"] = len(found)
            info["missing_dim"] = len(missing)
            info["coverage"] = f"{len(found)}/{spec['expected_dim']}"
        result.append(info)

    for name, spec in COMPOSITE_VECTORS.items():
        expected = get_vector_columns(name)
        info = {
            "name": name,
            "description": spec["description"],
            "use_case": spec["use_case"],
            "expected_dim": len(expected),
        }
        if df is not None:
            found, missing = extract_vector(df, name)
            info["found_dim"] = len(found)
            info["missing_dim"] = len(missing)
            info["coverage"] = f"{len(found)}/{len(expected)}"
        result.append(info)

    return result


def print_vectors(df=None) -> None:
    """Pretty-print available vectors."""
    vectors = list_vectors(df)
    header = f"{'Vector':<16} {'Dim':>4}"
    if df is not None:
        header += f"  {'Found':>6}  {'Description'}"
    else:
        header += f"  {'Description'}"
    print(header)
    print("-" * len(header) + "-" * 30)

    for v in vectors:
        line = f"{v['name']:<16} {v['expected_dim']:>4}"
        if df is not None:
            line += f"  {v['coverage']:>6}"
        line += f"  {v['description']}"
        print(line)


def get_all_vector_names() -> List[str]:
    """Return names of all base + composite vectors."""
    return list(FEATURE_VECTORS.keys()) + list(COMPOSITE_VECTORS.keys())


def get_total_feature_count() -> int:
    """Total number of unique features across all base vectors."""
    all_cols = set()
    for spec in FEATURE_VECTORS.values():
        all_cols.update(spec["columns"])
    return len(all_cols)
