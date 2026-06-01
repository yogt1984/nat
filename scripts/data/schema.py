"""Feature schema validation for NAT parquet files.

Catches column renames, missing features, and data quality issues
at load time instead of at runtime NaN surprises.

Source of truth: 211-column schema from the Rust ingestor's
FeatureComputer::names_all() (features/mod.rs).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# Meta columns (always present)
BASE_COLUMNS = ["timestamp_ns", "symbol", "sequence_id"]

# Base features: 14 categories, always computed by ingestor (154 features)
BASE_FEATURES = {
    "raw": [
        "raw_midprice", "raw_spread", "raw_spread_bps", "raw_microprice",
        "raw_bid_depth_5", "raw_ask_depth_5", "raw_bid_depth_10", "raw_ask_depth_10",
        "raw_bid_orders_5", "raw_ask_orders_5",
    ],
    "imbalance": [
        "imbalance_qty_l1", "imbalance_qty_l5", "imbalance_qty_l10",
        "imbalance_orders_l5", "imbalance_notional_l5", "imbalance_depth_weighted",
        "imbalance_pressure_bid", "imbalance_pressure_ask",
    ],
    "flow": [
        "flow_count_1s", "flow_count_5s", "flow_count_30s",
        "flow_volume_1s", "flow_volume_5s", "flow_volume_30s",
        "flow_aggressor_ratio_5s", "flow_aggressor_ratio_30s",
        "flow_vwap_5s", "flow_vwap_deviation",
        "flow_avg_trade_size_30s", "flow_intensity",
    ],
    "volatility": [
        "vol_returns_1m", "vol_returns_5m", "vol_parkinson_5m",
        "vol_spread_mean_1m", "vol_spread_std_1m", "vol_midprice_std_1m",
        "vol_ratio_short_long", "vol_zscore",
    ],
    "entropy": [
        "ent_permutation_returns_8", "ent_permutation_returns_16",
        "ent_permutation_returns_32", "ent_permutation_imbalance_16",
        "ent_spread_dispersion", "ent_volume_dispersion", "ent_book_shape",
        "ent_trade_size_dispersion", "ent_rate_of_change_5s", "ent_zscore_1m",
        "ent_tick_1s", "ent_tick_5s", "ent_tick_10s", "ent_tick_15s",
        "ent_tick_30s", "ent_tick_1m", "ent_tick_15m",
        "ent_vol_tick_1s", "ent_vol_tick_5s", "ent_vol_tick_10s",
        "ent_vol_tick_15s", "ent_vol_tick_30s", "ent_vol_tick_1m", "ent_vol_tick_15m",
    ],
    "context": [
        "ctx_funding_rate", "ctx_funding_zscore", "ctx_open_interest",
        "ctx_oi_change_5m", "ctx_oi_change_pct_5m", "ctx_premium_bps",
        "ctx_volume_24h", "ctx_volume_ratio", "ctx_mark_oracle_divergence",
        "ctx_funding_momentum_8h", "ctx_funding_acceleration", "ctx_oi_momentum_1h",
    ],
    "trend": [
        "trend_momentum_60", "trend_momentum_r2_60", "trend_monotonicity_60",
        "trend_momentum_300", "trend_momentum_r2_300", "trend_monotonicity_300",
        "trend_hurst_300", "trend_momentum_600", "trend_momentum_r2_600",
        "trend_monotonicity_600", "trend_hurst_600",
        "trend_ma_crossover", "trend_ma_crossover_norm",
        "trend_ema_short", "trend_ema_long",
    ],
    "medium_freq": [
        "mf_ema_1m", "mf_ema_5m", "mf_ema_15m",
        "mf_ema_cross_1m_5m", "mf_ema_cross_5m_15m",
        "mf_rsi_1m", "mf_rsi_5m", "mf_rsi_15m",
        "mf_bb_pctb_1m", "mf_bb_pctb_5m", "mf_bb_pctb_15m",
        "mf_bb_width_1m", "mf_bb_width_5m", "mf_bb_width_15m",
        "mf_atr_5m", "mf_atr_15m",
    ],
    "illiquidity": [
        "illiq_kyle_100", "illiq_amihud_100", "illiq_hasbrouck_100", "illiq_roll_100",
        "illiq_kyle_500", "illiq_amihud_500", "illiq_hasbrouck_500", "illiq_roll_500",
        "illiq_kyle_ratio", "illiq_amihud_ratio", "illiq_composite", "illiq_trade_count",
    ],
    "toxicity": [
        "toxic_vpin_10", "toxic_vpin_50", "toxic_vpin_roc",
        "toxic_adverse_selection", "toxic_effective_spread", "toxic_realized_spread",
        "toxic_flow_imbalance", "toxic_flow_imbalance_abs",
        "toxic_index", "toxic_trade_count",
    ],
    "derived": [
        "derived_entropy_trend_interaction", "derived_entropy_trend_zscore",
        "derived_trend_strength_60", "derived_trend_strength_300",
        "derived_trend_strength_ratio", "derived_entropy_volatility_ratio",
        "derived_regime_type_score", "derived_illiquidity_trend",
        "derived_informed_trend_score", "derived_toxicity_regime",
        "derived_toxic_chop_score", "derived_trend_strength_roc",
        "derived_entropy_momentum", "derived_regime_indicator", "derived_regime_confidence",
    ],
    "micro": [
        "micro_obi_velocity", "micro_obi_acceleration",
        "micro_queue_position_bid", "micro_queue_position_ask",
        "micro_depth_recovery_ratio",
    ],
    "resilience": [
        "resilience_recovery_time_50", "resilience_depth_impact_ratio",
        "resilience_recovery_speed",
    ],
    "hawkes": [
        "hawkes_intensity", "hawkes_baseline", "hawkes_branching_ratio",
    ],
}

# Optional features: 6 categories, NaN-padded when data sources unavailable (73 features)
OPTIONAL_FEATURES = {
    "whale": [
        "whale_net_flow_1h", "whale_net_flow_4h", "whale_net_flow_24h",
        "whale_flow_normalized_1h", "whale_flow_normalized_4h",
        "whale_flow_momentum", "whale_flow_intensity", "whale_flow_roc",
        "whale_buy_ratio", "whale_directional_agreement",
        "active_whale_count", "whale_total_activity",
    ],
    "liquidation": [
        "liquidation_risk_above_1pct", "liquidation_risk_above_2pct",
        "liquidation_risk_above_5pct", "liquidation_risk_above_10pct",
        "liquidation_risk_below_1pct", "liquidation_risk_below_2pct",
        "liquidation_risk_below_5pct", "liquidation_risk_below_10pct",
        "liquidation_asymmetry", "liquidation_intensity",
        "positions_at_risk_count", "largest_position_at_risk",
        "nearest_cluster_distance",
    ],
    "concentration": [
        "top5_concentration", "top10_concentration", "top20_concentration",
        "top50_concentration", "herfindahl_index", "gini_coefficient",
        "theil_index", "whale_retail_ratio", "whale_fraction",
        "whale_avg_size_ratio", "concentration_change_1h", "hhi_roc",
        "concentration_trend", "position_count", "whale_position_count",
    ],
    "regime": [
        "regime_absorption_1h", "regime_absorption_4h", "regime_absorption_24h",
        "regime_absorption_zscore", "regime_divergence_1h", "regime_divergence_4h",
        "regime_divergence_24h", "regime_divergence_zscore",
        "regime_kyle_lambda", "regime_churn_1h", "regime_churn_4h",
        "regime_churn_24h", "regime_churn_zscore",
        "regime_range_pos_4h", "regime_range_pos_24h", "regime_range_pos_1w",
        "regime_range_width_24h", "regime_accumulation_score",
        "regime_distribution_score", "regime_clarity",
        "regime", "regime_prob_accumulation", "regime_prob_markup",
        "regime_prob_distribution", "regime_prob_markdown", "regime_prob_ranging",
        "regime_confidence", "regime_entropy",
    ],
    "cross_symbol": [
        "cross_obi_divergence", "cross_obi_mean", "cross_obi_dispersion",
    ],
}

# Flat list of all expected columns
ALL_BASE = BASE_COLUMNS + [c for cols in BASE_FEATURES.values() for c in cols]
ALL_OPTIONAL = [c for cols in OPTIONAL_FEATURES.values() for c in cols]
ALL_COLUMNS = ALL_BASE + ALL_OPTIONAL


def validate_columns(df_columns: list[str]) -> dict:
    """Check loaded columns against expected schema.

    Returns:
        {
            "missing_base": [...],     # base columns expected but absent
            "missing_optional": [...], # optional columns absent (informational)
            "unexpected": [...],       # columns present but not in schema
            "valid": True/False        # True if no missing base columns
        }
    """
    col_set = set(df_columns)
    all_known = set(ALL_COLUMNS)

    missing_base = [c for c in ALL_BASE if c not in col_set]
    missing_optional = [c for c in ALL_OPTIONAL if c not in col_set]
    unexpected = [c for c in df_columns if c not in all_known]

    return {
        "missing_base": missing_base,
        "missing_optional": missing_optional,
        "unexpected": unexpected,
        "valid": len(missing_base) == 0,
    }


def validate_quality(df: pd.DataFrame) -> dict:
    """Data quality checks on a loaded DataFrame.

    Returns:
        {
            "nan_rates": {"col": 0.05, ...},  # per-column NaN rate (only cols with NaN)
            "high_nan_cols": [...],            # > 50% NaN
            "constant_cols": [...],            # zero variance
            "row_count": int,
            "symbol_counts": {"BTC": 4000, ...}
        }
    """
    n = len(df)
    if n == 0:
        return {
            "nan_rates": {},
            "high_nan_cols": [],
            "constant_cols": [],
            "row_count": 0,
            "symbol_counts": {},
        }

    # NaN rates for numeric columns
    numeric = df.select_dtypes(include="number")
    nan_counts = numeric.isna().sum()
    nan_rates = {col: rate for col, rate in (nan_counts / n).items() if rate > 0}
    high_nan = [col for col, rate in nan_rates.items() if rate > 0.5]

    # Constant columns (zero variance)
    stds = numeric.std()
    constant = [col for col, s in stds.items() if s == 0.0]

    # Symbol distribution
    symbol_counts = {}
    if "symbol" in df.columns:
        symbol_counts = df["symbol"].value_counts().to_dict()

    return {
        "nan_rates": nan_rates,
        "high_nan_cols": high_nan,
        "constant_cols": constant,
        "row_count": n,
        "symbol_counts": symbol_counts,
    }
