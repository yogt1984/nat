# Feature Manifest

191 features extracted from Hyperliquid WebSocket market data at 100ms intervals.

## Overview

- **Base features** (123): Always computed from order book, trades, and market context.
- **Optional features** (68): Require additional data sources or warmup. Padded with NaN when absent.
- **Data contract**: `Features::to_vec()` returns exactly 191 elements. `Features::names_all()` returns matching column names. Parquet schema built from `names_all()` in `output/schema.rs`.
- **Emission rate**: ~10 rows/sec per symbol (30 rows/sec for 3 symbols). Buffer flushes every 10,000 rows (~5.5 min).

## Summary

| # | Category | Count | Prefix | Module | Status | Key References |
|---|----------|-------|--------|--------|--------|----------------|
| 1 | Raw | 10 | `raw_` | `features/raw.rs` | All working | Gatheral & Oomen (2010) |
| 2 | Imbalance | 8 | `imbalance_` | `features/imbalance.rs` | All working | Cont, Stoikov & Talreja (2010) |
| 3 | Flow | 12 | `flow_` | `features/flow.rs` | All working | — |
| 4 | Volatility | 8 | `vol_` | `features/volatility.rs` | 6 working, 2 placeholder | Parkinson (1980) |
| 5 | Entropy | 24 | `ent_` | `features/entropy.rs` | All warmup-dependent | Bandt & Pompe (2002) |
| 6 | Context | 9 | `ctx_` | `features/context.rs` | All working | — |
| 7 | Trend | 15 | `trend_` | `features/trend.rs` | All working | Jegadeesh & Titman (1993) |
| 8 | Illiquidity | 12 | `illiq_` | `features/illiquidity.rs` | All working | Kyle (1985), Amihud (2002) |
| 9 | Toxicity | 10 | `toxic_` | `features/toxicity.rs` | All working | Easley et al. (2012) |
| 10 | Derived | 15 | `derived_` | `features/derived.rs` | All working | — |
| 11 | Whale Flow | 12 | `whale_` | `features/whale_flow.rs` | Optional | — |
| 12 | Liquidation | 13 | `liquidation_` | `features/liquidation.rs` | Optional | — |
| 13 | Concentration | 15 | `top`/`conc_` | `features/concentration.rs` | Optional | — |
| 14 | Regime | 20 | `regime_` | `features/regime/mod.rs` | Optional | — |
| 15 | GMM | 8 | `regime`/`prob_` | `ml/regime.rs` | Optional | — |

All module paths are relative to `rust/ing/src/`.

---

## 1. Raw (10 features)

Direct L2 order book measurements. Module: `features/raw.rs`.

| Parquet Column | Formula | Range | Interpretation |
|---------------|---------|-------|----------------|
| `raw_midprice` | (best_bid + best_ask) / 2 | [0, +inf) | Reference price |
| `raw_spread` | best_ask - best_bid | [0, +inf) | Market tightness |
| `raw_spread_bps` | spread / midprice × 10000 | [0, +inf) | Normalized transaction cost |
| `raw_microprice` | (V_ask × P_bid + V_bid × P_ask) / (V_bid + V_ask) | [0, +inf) | Fair-value estimate (Gatheral & Oomen 2010) |
| `raw_bid_depth_5` | Σ bid volume, levels 1-5 | [0, +inf) | Bid-side liquidity |
| `raw_ask_depth_5` | Σ ask volume, levels 1-5 | [0, +inf) | Ask-side liquidity |
| `raw_bid_depth_10` | Σ bid volume, levels 1-10 | [0, +inf) | Deep bid liquidity |
| `raw_ask_depth_10` | Σ ask volume, levels 1-10 | [0, +inf) | Deep ask liquidity |
| `raw_bid_orders_5` | Order count, bid levels 1-5 | [0, +inf) | Bid fragmentation |
| `raw_ask_orders_5` | Order count, ask levels 1-5 | [0, +inf) | Ask fragmentation |

## 2. Imbalance (8 features)

Order book asymmetry. Module: `features/imbalance.rs`. Ref: Cont, Stoikov & Talreja (2010).

| Parquet Column | Formula | Range | Interpretation |
|---------------|---------|-------|----------------|
| `imbalance_qty_l1` | (bid_vol - ask_vol) / (bid + ask), L1 | [-1, 1] | >0 = bid-heavy at touch |
| `imbalance_qty_l5` | Same formula, levels 1-5 | [-1, 1] | >0 = bid-heavy near touch |
| `imbalance_qty_l10` | Same formula, levels 1-10 | [-1, 1] | >0 = bid-heavy deep book |
| `imbalance_orders_l5` | Order count asymmetry, L1-5 | [-1, 1] | >0 = more bid orders |
| `imbalance_notional_l5` | Dollar-value asymmetry, L1-5 | [-1, 1] | >0 = bid-heavy in USD |
| `imbalance_depth_weighted` | Distance-weighted volume imbalance | [-1, 1] | Near-touch weighted |
| `imbalance_pressure_bid` | Cumulative depth × 1/(1+dist_bps/10) | [0, 1] | Normalized bid support |
| `imbalance_pressure_ask` | Cumulative depth × 1/(1+dist_bps/10) | [0, 1] | Normalized ask resistance |

## 3. Flow (12 features)

Trade arrival patterns and aggressor dynamics. Module: `features/flow.rs`.

| Parquet Column | Formula | Range | Interpretation |
|---------------|---------|-------|----------------|
| `flow_count_1s` | Trade count in 1s | [0, +inf) | Microstructure activity |
| `flow_count_5s` | Trade count in 5s | [0, +inf) | Short-term activity |
| `flow_count_30s` | Trade count in 30s | [0, +inf) | Medium-term activity |
| `flow_volume_1s` | Total volume in 1s | [0, +inf) | Micro liquidity demand |
| `flow_volume_5s` | Total volume in 5s | [0, +inf) | Short-term demand |
| `flow_volume_30s` | Total volume in 30s | [0, +inf) | Medium-term demand |
| `flow_aggressor_ratio_5s` | Buy vol / total vol, 5s | [0, 1] | >0.5 = buy-dominated |
| `flow_aggressor_ratio_30s` | Buy vol / total vol, 30s | [0, 1] | >0.5 = buy-dominated |
| `flow_vwap_5s` | Σ(price × vol) / Σ(vol), 5s | [0, +inf) | Volume-weighted price |
| `flow_vwap_deviation` | (VWAP - last_price) / last_price | (-inf, +inf) | >0 = buying above market |
| `flow_avg_trade_size_30s` | Mean trade size, 30s | [0, +inf) | Larger = institutional |
| `flow_intensity` | Trades/sec, 5s EMA | [0, +inf) | Urgency measure |

Window rationale: 1s = market-maker timescale, 5s = quote update cycle, 30s = informed trader execution.

## 4. Volatility (8 features)

Realized and range-based volatility. Module: `features/volatility.rs`. Ref: Parkinson (1980).

| Parquet Column | Formula | Range | Status |
|---------------|---------|-------|--------|
| `vol_returns_1m` | sqrt(Σ r² / N), 60 ticks | [0, +inf) | Working |
| `vol_returns_5m` | sqrt(Σ r² / N), 300 ticks | [0, +inf) | Working |
| `vol_parkinson_5m` | ln(H/L) / sqrt(4·ln(2)), 300 ticks | [0, +inf) | Working |
| `vol_spread_mean_1m` | Current spread (point-in-time) | [0, +inf) | Working |
| `vol_spread_std_1m` | — | 0.0 | **PLACEHOLDER** |
| `vol_midprice_std_1m` | std(prices), 60 ticks | [0, +inf) | Working |
| `vol_ratio_short_long` | vol_1m / vol_5m | [0, +inf) | >1 = accelerating |
| `vol_zscore` | — | 0.0 | **PLACEHOLDER** |

## 5. Entropy (24 features)

Information content and predictability. Module: `features/entropy.rs`.
Refs: Bandt & Pompe (2002), Shannon (1948), Zunino et al. (2009).

**Permutation entropy** (10 features):

| Parquet Column | Algorithm | Range | Interpretation |
|---------------|-----------|-------|----------------|
| `ent_permutation_returns_8` | Ordinal patterns, d=3, last 8 returns | [0, 1] | 0=deterministic, 1=random |
| `ent_permutation_returns_16` | Same, last 16 returns | [0, 1] | Medium horizon |
| `ent_permutation_returns_32` | Same, last 32 returns | [0, 1] | Longer horizon |
| `ent_permutation_imbalance_16` | Ordinal patterns of L1 imbalance | [0, 1] | Low = persistent imbalance |
| `ent_spread_dispersion` | Shannon entropy, binned spreads (10 bins) | [0, 1] | Low = tight clustering |
| `ent_volume_dispersion` | Shannon entropy, binned trade sizes (10 bins) | [0, 1] | Low = uniform sizes |
| `ent_book_shape` | Shannon entropy of depth proportions | [0, 1] | Low = concentrated depth |
| `ent_trade_size_dispersion` | Shannon entropy, trade sizes (5 bins) | [0, 1] | Low = homogeneous flow |
| `ent_rate_of_change_5s` | Entropy delta, current vs ~50 ticks ago | (-inf, +inf) | Sharp drop = regime onset |
| `ent_zscore_1m` | (current - mean) / std, ~600 tick buffer | (-inf, +inf) | |z|>2 = unusual regime |

**Tick entropy** (7 features — 1s/5s/10s/15s/30s/1m/15m windows):

| Parquet Column | Algorithm | Range |
|---------------|-----------|-------|
| `ent_tick_{1s,5s,10s,15s,30s,1m,15m}` | Shannon entropy of {up,down,neutral} trade directions | [0, ln(3)] |

**Volume-weighted tick entropy** (7 features — same windows):

| Parquet Column | Algorithm | Range |
|---------------|-----------|-------|
| `ent_vol_tick_{1s,5s,10s,15s,30s,1m,15m}` | Volume-weighted direction entropy | [0, ln(3)] |

Interpretation: low tick entropy = trending (one direction dominates), high = random/efficient market. Volume-weighted variant prevents small trades from diluting the signal.

## 6. Context (9 features)

Hyperliquid market metadata from activeAssetCtx. Module: `features/context.rs`.

| Parquet Column | Description | Range |
|---------------|-------------|-------|
| `ctx_funding_rate` | Current perp funding rate | (-inf, +inf) |
| `ctx_funding_zscore` | Funding vs historical distribution | (-inf, +inf) |
| `ctx_open_interest` | Total open contracts (USD) | [0, +inf) |
| `ctx_oi_change_5m` | Absolute OI change, 60 samples (~5 min) | (-inf, +inf) |
| `ctx_oi_change_pct_5m` | OI percent change | (-inf, +inf) |
| `ctx_premium_bps` | Mark vs index price premium (bps) | (-inf, +inf) |
| `ctx_volume_24h` | Rolling 24h traded volume | [0, +inf) |
| `ctx_volume_ratio` | Current vs average 24h volume | [0, +inf) |
| `ctx_mark_oracle_divergence` | Mark price minus oracle price | (-inf, +inf) |

## 7. Trend (15 features)

Trend detection and persistence. Module: `features/trend.rs`.
Refs: Jegadeesh & Titman (1993), Moskowitz et al. (2012), Mandelbrot (1971).

| Parquet Column | Formula | Range | Interpretation |
|---------------|---------|-------|----------------|
| `trend_momentum_{60,300,600}` | Linear regression slope | (-inf, +inf) | >0 = uptrend |
| `trend_momentum_r2_{60,300,600}` | R² of momentum regression | [0, 1] | Higher = cleaner trend |
| `trend_monotonicity_{60,300,600}` | Fraction of ticks in majority direction | [0.5, 1] | >0.7 = strong trend |
| `trend_hurst_{300,600}` | Rescaled range H exponent | [0, 1] | <0.5 = mean-reverting, >0.5 = trending |
| `trend_ma_crossover` | EMA(10) - EMA(50) | (-inf, +inf) | >0 = bullish |
| `trend_ma_crossover_norm` | Crossover / price | (-inf, +inf) | Normalized by price level |
| `trend_ema_short` | EMA, period 10 | [0, +inf) | Short-term average |
| `trend_ema_long` | EMA, period 50 | [0, +inf) | Long-term average |

Window sizes: 60/300/600 ticks ≈ 6s/30s/60s at 100ms emission.

## 8. Illiquidity (12 features)

Market impact and liquidity measures. Module: `features/illiquidity.rs`.
Refs: Kyle (1985), Amihud (2002), Hasbrouck (2009).

| Parquet Column | Formula | Range | Interpretation |
|---------------|---------|-------|----------------|
| `illiq_kyle_lambda_{100,500,1000}` | Cov(ΔP, signed_vol) / Var(signed_vol) | [0, +inf) | Higher = more illiquid |
| `illiq_amihud_lambda_{100,500,1000}` | mean(\|return\| / volume) | [0, +inf) | Higher = more illiquid |
| `illiq_hasbrouck_lambda_{100,500,1000}` | Permanent price impact component | [0, +inf) | Higher = more illiquid |
| `illiq_roll_spread_{100,500,1000}` | 2·sqrt(-Cov(ΔP_t, ΔP_{t-1})) | [0, +inf) | Implied spread (Roll 1984) |

Window sizes: 100/500/1000 trades (short/medium/long market impact horizons).

## 9. Toxicity (10 features)

Order flow quality and informed trading detection. Module: `features/toxicity.rs`.
Refs: Easley et al. (2012), Glosten & Milgrom (1985).

| Parquet Column | Formula | Range | Interpretation |
|---------------|---------|-------|----------------|
| `toxic_vpin_10` | VPIN, 10-bucket window | [0, 1] | >0.5 = toxic flow |
| `toxic_vpin_50` | VPIN, 50-bucket window | [0, 1] | Longer-horizon toxicity |
| `toxic_vpin_roc` | VPIN rate of change | (-inf, +inf) | Rising = increasing toxicity |
| `toxic_adverse_selection` | Effective - realized spread | [0, +inf) | Higher = more adverse selection |
| `toxic_effective_spread` | 2·\|trade_price - mid\| / mid × 10000 | [0, +inf) | Actual execution cost (bps) |
| `toxic_realized_spread` | 2·sign·(trade_price - mid_future) / mid × 10000 | (-inf, +inf) | Market-maker P&L (bps) |
| `toxic_flow_imbalance` | Signed volume imbalance ratio | [-1, 1] | >0 = net buying |
| `toxic_flow_imbalance_abs` | \|flow_imbalance\| | [0, 1] | Magnitude of directional flow |
| `toxic_index` | Composite toxicity score | [0, 1] | Weighted average of indicators |
| `toxic_trade_count` | Trades used in computation | [0, +inf) | Data sufficiency indicator |

## 10. Derived (15 features)

Composite indicators from base features. Module: `features/derived.rs`.

| Parquet Column | Formula | Interpretation |
|---------------|---------|----------------|
| `derived_entropy_trend_interaction` | entropy × (1 - monotonicity) | High = choppy/uncertain |
| `derived_entropy_trend_zscore` | Z-score of entropy×trend interaction | Regime change signal |
| `derived_trend_strength_{60,300}` | momentum × R² | Strong trend = high |
| `derived_trend_strength_ratio` | strength_60 / strength_300 | >1 = short-term acceleration |
| `derived_entropy_volatility_ratio` | entropy / (1 + volatility) | Regime stability |
| `derived_regime_type_score` | Hurst-based regime indicator | >0 = trending regime |
| `derived_illiquidity_trend` | Kyle λ × momentum sign | Informed directional flow |
| `derived_informed_trend_score` | VPIN × trend strength | Toxic trending |
| `derived_toxicity_regime` | toxicity_index × (1 - entropy) | Informed flow in ordered market |
| `derived_toxic_chop_score` | toxicity × entropy | Informed flow in choppy market |
| `derived_trend_strength_roc` | Δ trend_strength | Trend acceleration |
| `derived_entropy_momentum` | Δ entropy over time | Regime transition speed |
| `derived_regime_indicator` | Composite regime signal | Overall regime state |
| `derived_regime_confidence` | Consistency of regime indicators | How clear the regime is |

## 11. Whale Flow (12 features) — Optional

Position change tracking for large accounts. Module: `features/whale_flow.rs`.

| Parquet Column | Description | Range |
|---------------|-------------|-------|
| `whale_net_flow_{1h,4h,24h}` | Net position changes (buys - sells) | (-inf, +inf) |
| `whale_flow_normalized_{1h,4h}` | Flow / average absolute flow | (-inf, +inf) |
| `whale_flow_momentum` | Acceleration of flow | (-inf, +inf) |
| `whale_flow_intensity` | |flow| / time | [0, +inf) |
| `whale_flow_roc` | Rate of change of net flow | (-inf, +inf) |
| `whale_buy_ratio` | Buy volume / total whale volume | [0, 1] |
| `whale_directional_agreement` | Fraction of whales in same direction | [0, 1] |
| `active_whale_count` | Number of active whale accounts | [0, +inf) |
| `whale_total_activity` | Total whale volume (buys + sells) | [0, +inf) |

## 12. Liquidation Risk (13 features) — Optional

Cascade prediction from open positions. Module: `features/liquidation.rs`.

| Parquet Column | Description | Range |
|---------------|-------------|-------|
| `liquidation_risk_above_{1,2,5,10}pct` | Liquidation volume if price rises N% | [0, +inf) |
| `liquidation_risk_below_{1,2,5,10}pct` | Liquidation volume if price falls N% | [0, +inf) |
| `liquidation_asymmetry` | (risk_above - risk_below) / total | [-1, 1] |
| `liquidation_intensity` | Total risk / total OI | [0, 1] |
| `positions_at_risk_count` | Positions near liquidation | [0, +inf) |
| `largest_position_at_risk` | Size of largest at-risk position | [0, +inf) |
| `nearest_cluster_distance` | Distance to nearest liquidation cluster (%) | [0, +inf) |

## 13. Concentration (15 features) — Optional

Position crowding metrics. Module: `features/concentration.rs`.

| Parquet Column | Formula | Range |
|---------------|---------|-------|
| `top{5,10,20,50}_concentration` | Top-N share of total OI | [0, 1] |
| `herfindahl_index` | Σ (share_i)² | [0, 1] |
| `gini_coefficient` | Lorenz curve area ratio | [0, 1] |
| `theil_index` | Σ share_i × ln(share_i / (1/N)) | [0, +inf) |
| `whale_retail_ratio` | Whale OI / retail OI | [0, +inf) |
| `whale_fraction` | Whale accounts / total accounts | [0, 1] |
| `whale_avg_size_ratio` | Avg whale position / avg overall | [1, +inf) |
| `concentration_change_1h` | Δ top-10 concentration, 1 hour | (-inf, +inf) |
| `hhi_roc` | Δ HHI rate of change | (-inf, +inf) |
| `concentration_trend` | Linear trend of concentration | (-inf, +inf) |
| `position_count` | Total tracked positions | [0, +inf) |
| `whale_position_count` | Whale-classified positions | [0, +inf) |

## 14. Regime Detection (20 features) — Optional

Accumulation/distribution cycle metrics. Module: `features/regime/mod.rs`.

| Parquet Column | Description | Windows |
|---------------|-------------|---------|
| `regime_absorption_{1h,4h,24h}` | Volume absorption ratio | 1h/4h/24h |
| `regime_absorption_zscore` | Absorption vs historical | — |
| `regime_divergence_{1h,4h,24h}` | Price-volume divergence | 1h/4h/24h |
| `regime_divergence_zscore` | Divergence vs historical | — |
| `regime_kyle_lambda` | Kyle's lambda at regime timescale | — |
| `regime_churn_{1h,4h,24h}` | Volume turnover rate | 1h/4h/24h |
| `regime_churn_zscore` | Churn vs historical | — |
| `regime_range_pos_{4h,24h,1w}` | Price position within range [0,1] | 4h/24h/1w |
| `regime_range_width_24h` | Range width as % of price | — |
| `regime_accumulation_score` | Composite accumulation indicator | — |
| `regime_distribution_score` | Composite distribution indicator | — |
| `regime_clarity` | How clearly one regime dominates | [0, 1] |

## 15. GMM Classification (8 features) — Optional

Gaussian Mixture Model regime output. Module: `ml/regime.rs`.

| Parquet Column | Description | Range |
|---------------|-------------|-------|
| `regime` | Classified regime (numeric) | {0,1,2,3,4} |
| `regime_prob_accumulation` | P(accumulation) | [0, 1] |
| `regime_prob_markup` | P(markup) | [0, 1] |
| `regime_prob_distribution` | P(distribution) | [0, 1] |
| `regime_prob_markdown` | P(markdown) | [0, 1] |
| `regime_prob_ranging` | P(ranging) | [0, 1] |
| `regime_confidence` | max(probabilities) | [0, 1] |
| `regime_entropy` | -Σ p_i ln(p_i) of regime probs | [0, ln(5)] |

GMM input features: [kyle_lambda, vpin, absorption_zscore, hurst, whale_net_flow_1h].

---

## Placeholder Features

These features are hardcoded to 0.0 and need additional infrastructure to implement:

| Feature | Module | What's Needed |
|---------|--------|---------------|
| `vol_spread_std_1m` | `volatility.rs:91` | Pass spread history buffer to `compute()` |
| `vol_zscore` | `volatility.rs:112` | Hourly volatility history buffer |

Note: Several entropy features (`ent_permutation_imbalance_16`, `ent_spread_dispersion`, `ent_rate_of_change_5s`, `ent_zscore_1m`) return 0.0 during warmup (insufficient buffer data) but work correctly once the buffer fills. These are **not** placeholders.

## Information Redundancy

Feature pairs that carry overlapping information (useful for feature selection):

| Feature A | Feature B | Relationship |
|-----------|-----------|-------------|
| `illiq_kyle_lambda_*` | `regime_kyle_lambda` | Same concept, different time aggregation |
| `ent_volume_dispersion` | `ent_trade_size_dispersion` | Similar inputs, different bin counts (10 vs 5) |
| `flow_aggressor_ratio_5s` | `toxic_flow_imbalance` | Related measures of buy/sell asymmetry |
| `imbalance_qty_l5` | `imbalance_notional_l5` | Volume vs dollar-value (correlated for stable prices) |
| `trend_momentum_*` | `trend_ma_crossover` | Both measure directional tendency |
| `vol_returns_1m` | `vol_midprice_std_1m` | Both measure short-term dispersion |

## Hypothesis Mapping

Which features feed each hypothesis test (see `hypothesis/` module):

| Hypothesis | Primary Features | Secondary |
|-----------|-----------------|-----------|
| H1: Whale flow → returns | `whale_*` (12) | `trend_momentum_*` |
| H2: Entropy × whale interaction | `ent_*` + `whale_*` | `derived_entropy_*` |
| H3: Liquidation cascades | `liquidation_*` (13) | `vol_*`, `ctx_oi_*` |
| H4: Concentration → volatility | `top*_concentration`, `gini_*`, `hhi_*` | `vol_*` |
| H5: Persistence indicator | `trend_hurst_*`, `trend_monotonicity_*` | `ent_tick_*` |

## Adding a New Feature

1. Create struct with `count()`, `names()`, `to_vec()` methods
2. Add to `Features` struct in `features/mod.rs`
3. Add to `to_vec()`, `names_all()`, `count_all()` — if optional, use NaN padding pattern
4. Schema updates automatically via `create_schema()` in `output/schema.rs`
5. Add a row to this manifest
