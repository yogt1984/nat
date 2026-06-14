# Feature Catalogue & Status Report

**Date:** 2026-06-12
**Codebase:** 236 features across 21 categories in `rust/ing-features/src/` (14.8K LOC)
**Resolution:** 100ms emission interval, ~10 rows/sec per symbol
**Symbols:** BTC, ETH, SOL | **Output:** Parquet + zstd compression
**IC scan data:** 3 days (2026-05-19 to 2026-05-21), 207 live features, 6 horizons

---

## 1. Executive Summary

236 features are computed per tick across 21 categories. The feature vector has a fixed length — optional categories are NaN-padded when absent.

| Segment       | Categories | Features | Status                    |
|---------------|------------|----------|---------------------------|
| Base (always) | 14         | 154      | All live                  |
| Optional      | 7          | 82       | All dead (K2/K3 issues)   |
| **Total**     | **21**     | **236**  | **154 live, 82 dead**     |

**Key findings:**
- `imbalance_qty_l1` is the #1 predictor (IC 0.45-0.47 at 1-5s, all symbols)
- Signal half-life ~30s — directional IC decays to ~0.06 at 15min
- Volatility and direction are orthogonal: zero cross-IC between imbalance and vol features
- 82 optional features (whale, liquidation, concentration, regime, GMM, heatmap) are ALL-NaN due to missing API wiring (K2)
- Conditional IC (given mid-cross fills) collapses to ~0 — confirms adverse selection at microstructure frequencies

---

## 2. Data Contract

**File:** `rust/ing-features/src/lib.rs` (474 LOC)

- `Features::to_vec()` always returns exactly 236 `f64` values
- `Features::names_all()` returns matching column names in same order
- `Features::count_all()` = 236, `Features::count()` = 154 (base only)
- Parquet schema built from `names_all()` in `rust/ing/src/output/schema.rs`
- 3 metadata columns prepended: `timestamp_ns`, `symbol`, `sequence_id`

**Vector order:** 154 base features (categories 1-14) + 82 optional features (categories 15-21), NaN-padded when absent.

**Ring buffer sizes:** spread 600, midprice 3000, entropy 600, imbalance 16, OBI 600, depth 600, vol_1m 36000 ticks.

---

## 3. Base Features — Always Computed (154 total)

### 3.1 Raw (10 features)

**File:** `rust/ing-features/src/raw.rs` | **Prefix:** `raw_`
**References:** Gatheral & Oomen (2010)

| Feature | Formula | Range | IC @ 5s |
|---------|---------|-------|---------|
| `raw_midprice` | (best_bid + best_ask) / 2 | [0, +inf) | — |
| `raw_spread` | best_ask - best_bid | [0, +inf) | — |
| `raw_spread_bps` | spread / midprice x 10000 | [0, +inf) | +0.074-0.179 @ 15m |
| `raw_microprice` | (V_ask x P_bid + V_bid x P_ask) / (V_bid + V_ask) | [0, +inf) | — |
| `raw_bid_depth_5` | Sum bid volume, levels 1-5 | [0, +inf) | +0.405-0.420 |
| `raw_ask_depth_5` | Sum ask volume, levels 1-5 | [0, +inf) | -0.404 to -0.424 |
| `raw_bid_depth_10` | Sum bid volume, levels 1-10 | [0, +inf) | +0.368-0.409 |
| `raw_ask_depth_10` | Sum ask volume, levels 1-10 | [0, +inf) | -0.360 to -0.399 |
| `raw_bid_orders_5` | Order count, bid L1-L5 | [0, +inf) | +0.406-0.425 |
| `raw_ask_orders_5` | Order count, ask L1-L5 | [0, +inf) | -0.405 to -0.424 |

**Status:** All operational. Depth asymmetry is a top-10 fast directional predictor.

---

### 3.2 Imbalance (8 features)

**File:** `rust/ing-features/src/imbalance.rs` | **Prefix:** `imbalance_`
**References:** Cont, Stoikov & Talreja (2010)

| Feature | Formula | Range | IC @ 5s |
|---------|---------|-------|---------|
| `imbalance_qty_l1` | (bid_vol - ask_vol) / (bid + ask), L1 | [-1, 1] | **+0.447-0.466** |
| `imbalance_qty_l5` | Same, levels 1-5 | [-1, 1] | **+0.431-0.456** |
| `imbalance_qty_l10` | Same, levels 1-10 | [-1, 1] | +0.410-0.434 |
| `imbalance_orders_l5` | Order count asymmetry, L1-5 | [-1, 1] | +0.435-0.455 |
| `imbalance_notional_l5` | Dollar-value asymmetry, L1-5 | [-1, 1] | +0.431-0.456 |
| `imbalance_depth_weighted` | Distance-weighted volume imbalance | [-1, 1] | +0.410-0.434 |
| `imbalance_pressure_bid` | Cumulative depth x 1/(1+dist_bps/10) | [0, 1] | +0.377-0.404 |
| `imbalance_pressure_ask` | Same, ask side | [0, 1] | -0.387 to -0.417 |

**Status:** ALL LIVE & DOMINANT. `imbalance_qty_l1` is the **#1 feature** across all symbols.

**Temporal stability:**
- Mean IC +0.433 +/- 0.061 (BTC, 23-day rolling)
- Worst day IC: +0.24 (BTC), +0.04 (ETH), +0.17 (SOL)
- June OOS degradation: IC drops from +0.45 (May) to +0.29 (June) at 5s horizon
- 1s horizon stable: +0.49-0.50 in June vs +0.45 in May

**Adverse selection finding:** Conditional IC (given mid-cross fills) drops to ~0. Directional IC is structurally eliminated by execution cost at microstructure frequencies.

---

### 3.3 Flow (12 features)

**File:** `rust/ing-features/src/flow.rs` | **Prefix:** `flow_`

| Feature | Window | Range | IC @ 5s | Type |
|---------|--------|-------|---------|------|
| `flow_count_1s` | 1s | [0, +inf) | — | Activity |
| `flow_count_5s` | 5s | [0, +inf) | — | Activity |
| `flow_count_30s` | 30s | [0, +inf) | +0.281-0.318 | Volatility |
| `flow_volume_1s` | 1s | [0, +inf) | — | Activity |
| `flow_volume_5s` | 5s | [0, +inf) | — | Activity |
| `flow_volume_30s` | 30s | [0, +inf) | — | Activity |
| `flow_aggressor_ratio_5s` | 5s | [0, 1] | +0.109-0.112 | Weak directional |
| `flow_aggressor_ratio_30s` | 30s | [0, 1] | — | Directional |
| `flow_vwap_5s` | 5s | [0, +inf) | — | Reference |
| `flow_vwap_deviation` | 5s | (-inf, +inf) | -0.188 to -0.287 | Mean-reversion |
| `flow_avg_trade_size_30s` | 30s | [0, +inf) | — | Activity |
| `flow_intensity` | 5s EMA | [0, +inf) | +0.181-0.291 | Volatility |

**Status:** All operational. Flow counts predict volatility, not direction. VWAP deviation shows weak mean-reversion at 1s.

---

### 3.4 Volatility (9 features)

**File:** `rust/ing-features/src/volatility.rs` | **Prefix:** `vol_`
**References:** Parkinson (1980), Garman & Klass (1980)

| Feature | Method | Range | IC @ 5s | Notes |
|---------|--------|-------|---------|-------|
| `vol_returns_1m` | Realized (60 ticks) | [0, +inf) | +0.318-0.323 | Vol IC |
| `vol_returns_5m` | Realized (300 ticks) | [0, +inf) | **+0.326-0.328** | Top vol predictor |
| `vol_parkinson_5m` | ln(H/L)/sqrt(4*ln2) | [0, +inf) | +0.312-0.318 | Efficient estimator |
| `vol_garman_klass_5m` | 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2 | [0, +inf) | — | Most efficient classical |
| `vol_spread_mean_1m` | Current spread | [0, +inf) | — | Point-in-time |
| `vol_spread_std_1m` | Spread std (600 ticks) | [0, +inf) | — | Liquidity instability |
| `vol_midprice_std_1m` | Price std (60 ticks) | [0, +inf) | +0.288-0.295 | Vol predictor |
| `vol_ratio_short_long` | vol_1m / vol_5m | [0, +inf) | — | >1 = accelerating |
| `vol_zscore` | (vol_1m - mean_1h)/std_1h, [-10,10] | [-10, 10] | — | Spike detector |

**Status:** All working. Predicts magnitude, NOT direction. Zero directional IC. Warmup: vol_zscore needs ~12s (120 samples).

---

### 3.5 Entropy (27 features)

**File:** `rust/ing-features/src/entropy.rs` | **Prefix:** `ent_`
**References:** Bandt & Pompe (2002), Shannon (1948), Zunino et al. (2009)

**Permutation Entropy (10 features):**

| Feature | Window | Order | Range | IC @ 1s |
|---------|--------|-------|-------|---------|
| `ent_permutation_returns_8` | 8 ticks | m=3 (6 patterns) | [0, 1] | — |
| `ent_permutation_returns_16` | 16 ticks | m=3 | [0, 1] | — |
| `ent_permutation_returns_32` | 32 ticks | m=3 | [0, 1] | +0.312-0.315 (vol) |
| `ent_permutation_imbalance_16` | 16 samples | m=3 | [0, 1] | -0.170 to -0.177 |
| `ent_perm_m5_8` | 8 ticks | m=5 (120 patterns) | [0, 1] | — |
| `ent_perm_m5_16` | 16 ticks | m=5 | [0, 1] | — |
| `ent_perm_m5_32` | 32 ticks | m=5 | [0, 1] | — |
| `ent_spread_dispersion` | current | 10 bins | [0, 1] | — |
| `ent_volume_dispersion` | current | 10 bins | [0, 1] | — |
| `ent_book_shape` | current | depth proportions | [0, 1] | — |

**Tick Entropy (7 features):**

| Feature | Window | Range | Description |
|---------|--------|-------|-------------|
| `ent_tick_1s` | 1s | [0, ln(3)] | Direction entropy (3 states: up/down/neutral) |
| `ent_tick_5s` | 5s | [0, ln(3)] | |
| `ent_tick_10s` | 10s | [0, ln(3)] | |
| `ent_tick_15s` | 15s | [0, ln(3)] | |
| `ent_tick_30s` | 30s | [0, ln(3)] | |
| `ent_tick_1m` | 1m | [0, ln(3)] | |
| `ent_tick_15m` | 15m | [0, ln(3)] | |

**Volume-Weighted Tick Entropy (7 features):** Same windows as tick entropy, weighted by trade volume. Prefix: `ent_vol_tick_*`.

**Other Entropy (3 features):**

| Feature | Description | Range |
|---------|-------------|-------|
| `ent_trade_size_dispersion` | Shannon entropy of trade sizes (5 bins) | [0, 1] |
| `ent_rate_of_change_5s` | Entropy delta (current vs ~50 ticks ago) | (-inf, +inf) |
| `ent_zscore_1m` | (current - mean) / std, 600-tick buffer | (-inf, +inf) |

**Interpretation:** Low entropy = trending (one direction dominates). High = random/efficient market. `ent_book_shape` is the key regime-gating variable (IC lift +22% in low-entropy quintile).

**Status:** All 27 working. Warmup: permutation/z-score features need ~12s buffer fill.

---

### 3.6 Context (12 features)

**File:** `rust/ing-features/src/context.rs` | **Prefix:** `ctx_`
**Source:** Hyperliquid `activeAssetCtx` WebSocket channel

| Feature | Description | Range |
|---------|-------------|-------|
| `ctx_funding_rate` | Perpetual funding rate | (-inf, +inf) |
| `ctx_funding_zscore` | Funding vs historical distribution | (-inf, +inf) |
| `ctx_open_interest` | Total open contracts (USD) | [0, +inf) |
| `ctx_oi_change_5m` | Absolute OI change (~5 min) | (-inf, +inf) |
| `ctx_oi_change_pct_5m` | OI percent change | (-inf, +inf) |
| `ctx_premium_bps` | Mark vs index price premium (bps) | (-inf, +inf) |
| `ctx_volume_24h` | Rolling 24h traded volume | [0, +inf) |
| `ctx_volume_ratio` | Current / avg 24h volume | [0, +inf) |
| `ctx_mark_oracle_divergence` | Mark price - oracle price | (-inf, +inf) |
| `ctx_funding_momentum_8h` | funding(t) - funding(t-8h) | (-inf, +inf) |
| `ctx_funding_acceleration` | d(momentum)/dt over 1h | (-inf, +inf) |
| `ctx_oi_momentum_1h` | OI % change (1h-like lookback) | (-inf, +inf) |

**Status:** All 12 operational. Funding features drive `funding_reversion` algorithm (Sharpe 6.1 ETH).

---

### 3.7 Trend (15 features)

**File:** `rust/ing-features/src/trend.rs` | **Prefix:** `trend_`
**References:** Jegadeesh & Titman (1993), Mandelbrot (1971)

| Feature | Window | Range | Description |
|---------|--------|-------|-------------|
| `trend_momentum_60` | 60 ticks (~6s) | (-inf, +inf) | Linear regression slope |
| `trend_momentum_300` | 300 ticks (~30s) | (-inf, +inf) | Linear regression slope |
| `trend_momentum_600` | 600 ticks (~60s) | (-inf, +inf) | Linear regression slope |
| `trend_momentum_r2_60` | 60 ticks | [0, 1] | R^2 of regression |
| `trend_momentum_r2_300` | 300 ticks | [0, 1] | R^2 of regression |
| `trend_momentum_r2_600` | 600 ticks | [0, 1] | R^2 of regression |
| `trend_monotonicity_60` | 60 ticks | [0.5, 1.0] | Fraction in majority direction |
| `trend_monotonicity_300` | 300 ticks | [0.5, 1.0] | |
| `trend_monotonicity_600` | 600 ticks | [0.5, 1.0] | |
| `trend_hurst_300` | 300 ticks | [0, 1] | R/S Hurst: >0.5=trending, <0.5=reverting |
| `trend_hurst_600` | 600 ticks | [0, 1] | |
| `trend_ma_crossover` | EMA(10)-EMA(50) | (-inf, +inf) | |
| `trend_ma_crossover_norm` | crossover/price x100 | (-inf, +inf) | Normalized |
| `trend_ema_short` | EMA period 10 | [0, +inf) | IC: -0.076 to -0.188 @ 15m |
| `trend_ema_long` | EMA period 50 | [0, +inf) | |

**Status:** All 15 working. Hurst exponent is key for regime detection (H<0.5 flags mean-reversion).

---

### 3.8 Medium Frequency (16 features)

**File:** `rust/ing-features/src/medium_freq.rs` | **Prefix:** `mf_`
**References:** Wilder (1978), Bollinger (2001)
**Bar definition:** Deterministic tick-count: 600/3000/9000 ticks per 1m/5m/15m (not wall-clock)

| Feature | Timeframe | Range | Description |
|---------|-----------|-------|-------------|
| `mf_ema_1m` | 600 ticks | [0, +inf) | Exponential MA |
| `mf_ema_5m` | 3000 ticks | [0, +inf) | |
| `mf_ema_15m` | 9000 ticks | [0, +inf) | |
| `mf_ema_cross_1m_5m` | 1m/5m | (-inf, +inf) bps | (EMA_1m - EMA_5m) / EMA_5m x 10000 |
| `mf_ema_cross_5m_15m` | 5m/15m | (-inf, +inf) bps | |
| `mf_rsi_1m` | 14 bars, 1m | [0, 100] | Wilder RSI |
| `mf_rsi_5m` | 14 bars, 5m | [0, 100] | |
| `mf_rsi_15m` | 14 bars, 15m | [0, 100] | |
| `mf_bb_pctb_1m` | 20-bar, 2-sigma, 1m | (-inf, +inf) | Bollinger %B |
| `mf_bb_pctb_5m` | 20-bar, 2-sigma, 5m | (-inf, +inf) | |
| `mf_bb_pctb_15m` | 20-bar, 2-sigma, 15m | (-inf, +inf) | |
| `mf_bb_width_1m` | 20-bar, 2-sigma, 1m | [0, +inf) | Band width |
| `mf_bb_width_5m` | 20-bar, 5m | [0, +inf) | |
| `mf_bb_width_15m` | 20-bar, 15m | [0, +inf) | |
| `mf_atr_5m` | Wilder ATR(14), 5m | [0, +inf) | Average True Range |
| `mf_atr_15m` | Wilder ATR(14), 15m | [0, +inf) | |

**Status:** All 16 working. EMAs update continuously; RSI/BB/ATR update on bar completion. Warmup-dependent.

---

### 3.9 Illiquidity (12 features)

**File:** `rust/ing-features/src/illiquidity.rs` | **Prefix:** `illiq_`
**References:** Kyle (1985), Amihud (2002), Hasbrouck (2009), Roll (1984)

| Feature | Window | Formula | IC @ 5s |
|---------|--------|---------|---------|
| `illiq_kyle_100` | 100 trades | Cov(dP, signed_vol) / Var(signed_vol) | — |
| `illiq_kyle_500` | 500 trades | Same | — |
| `illiq_kyle_ratio` | 100/500 | kyle_100 / kyle_500 | — |
| `illiq_amihud_100` | 100 trades | Sum|r_i| / Sum|v_i| x 1e6 | — |
| `illiq_amihud_500` | 500 trades | Same | — |
| `illiq_amihud_ratio` | 100/500 | amihud_100 / amihud_500 | — |
| `illiq_hasbrouck_100` | 100 trades | OLS: r_t = c + lambda*x_t + eps | — |
| `illiq_hasbrouck_500` | 500 trades | Same | — |
| `illiq_roll_100` | 100 trades | 2*sqrt(-Cov(dP_t, dP_{t-1})) | — |
| `illiq_roll_500` | 500 trades | Same | — |
| `illiq_composite` | — | (kyle + amihud + hasbrouck) / 3 | — |
| `illiq_trade_count` | — | Trades in buffer | +0.286-0.318 (vol) |

**Status:** All 12 working. Market impact measures — short/long ratios highlight liquidity spikes.

---

### 3.10 Toxicity (10 features)

**File:** `rust/ing-features/src/toxicity.rs` | **Prefix:** `toxic_`
**References:** Easley, Lopez de Prado, O'Hara (2012), Glosten & Milgrom (1985)

| Feature | Formula | Range | IC @ 5s |
|---------|---------|-------|---------|
| `toxic_vpin_10` | 10-bucket VPIN | [0, 1] | — |
| `toxic_vpin_50` | 50-bucket VPIN | [0, 1] | — |
| `toxic_vpin_roc` | VPIN rate of change | (-inf, +inf) | — |
| `toxic_adverse_selection` | Corr(trade_dir, future_return) x 10000 | [0, +inf) | — |
| `toxic_effective_spread` | 2 x mean|trade_price - VWAP| | [0, +inf) | +0.279-0.291 |
| `toxic_realized_spread` | 2 x mean[dir x (price - ref_mid)] | (-inf, +inf) | — |
| `toxic_flow_imbalance` | (buy_vol - sell_vol) / total | [-1, 1] | — |
| `toxic_flow_imbalance_abs` | |flow_imbalance| | [0, 1] | — |
| `toxic_index` | 0.5*VPIN + 0.3*adverse_norm + 0.2*flow_norm | [0, 1] | — |
| `toxic_trade_count` | Trades in buffer | [0, +inf) | — |

**Status:** All 10 working. Predicts volatility, not direction. VPIN is used as gate by `vpin_regime` algorithm.

---

### 3.11 Derived / Interaction (15 features)

**File:** `rust/ing-features/src/derived.rs` | **Prefix:** `derived_`

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `derived_entropy_trend_interaction` | entropy x (1 - monotonicity) | [0, ~0.55] | High = choppy |
| `derived_entropy_trend_zscore` | (interaction - 0.2) / 0.15 | (-inf, +inf) | |z|>2 = extreme |
| `derived_trend_strength_60` | sign(mom60) x (mono60-0.5)x2 x (1-ent) | [-1, 1] | Directional strength |
| `derived_trend_strength_300` | Same, 300-tick | [-1, 1] | |
| `derived_trend_strength_ratio` | strength_60 / strength_300 | (-inf, +inf) | >1 = short-term accel |
| `derived_entropy_volatility_ratio` | entropy / (1 + vol) | [0, ~1.1] | Orderly low-vol |
| `derived_regime_type_score` | vol x (1 - 2*entropy) | [-1, 1] | >0=breakout, <0=chaos |
| `derived_illiquidity_trend` | kyle x |mom60| x 1000 | [0, +inf) | Informed directional flow |
| `derived_informed_trend_score` | kyle x monotonicity | [0, 1] | Persistent informed |
| `derived_toxicity_regime` | toxicity_index x entropy | [0, +inf) | Toxic in choppy market |
| `derived_toxic_chop_score` | VPIN x (1 - monotonicity) | [0, 1] | Toxic when directionless |
| `derived_trend_strength_roc` | strength_60 - strength_300 | [-2, 2] | Trend acceleration |
| `derived_entropy_momentum` | ent_1m - ent_5s | [-1.1, 1.1] | >0 = regime breaking down |
| `derived_regime_indicator` | mean_revert - trending - flow | [-1, 1] | -1=trending, +1=reverting |
| `derived_regime_confidence` | max(agree_trend, agree_revert) x 2 | [0, 1] | Regime consensus |

**Status:** All 15 working. Composite indicators from base features.

---

### 3.12 Microstructure (5 features)

**File:** `rust/ing-features/src/microstructure.rs` | **Prefix:** `micro_`
**References:** Biais, Hillion & Spatt (1995)

| Feature | Window | Range | Description |
|---------|--------|-------|-------------|
| `micro_obi_velocity` | 10 samples (~1s) | (-inf, +inf) | d(OBI_l5)/dt finite difference |
| `micro_obi_acceleration` | 20 samples (~2s) | (-inf, +inf) | d^2(OBI_l5)/dt^2 |
| `micro_queue_position_bid` | 5s | [0, +inf) | Bid depth / avg_trade_size (ticks to fill) |
| `micro_queue_position_ask` | 5s | [0, +inf) | Ask depth / avg_trade_size |
| `micro_depth_recovery_ratio` | 10 samples (~1s) | [0, +inf) | current_depth / depth_1s_ago |

**Status:** All 5 working. OBI inflection predicts direction changes. Used by hierarchical_combiner L2.

---

### 3.13 Resilience (3 features)

**File:** `rust/ing-features/src/resilience.rs` | **Prefix:** `resilience_`

| Feature | Description |
|---------|-------------|
| `resilience_recovery_time_50` | Avg ticks to 50% depth recovery after >20% drop |
| `resilience_depth_impact_ratio` | Avg max depth drop / pre-take depth |
| `resilience_recovery_speed` | Avg depth recovery per tick (normalized) |

**Status:** All 3 working. Tracks events when depth drops >20% in a single tick.

---

### 3.14 Hawkes Trade Intensity (3 features)

**File:** `rust/ing-features/src/hawkes.rs` | **Prefix:** `hawkes_`
**References:** Bacry, Mastromatteo & Muzy (2015)

| Feature | Formula | Range | IC @ 5s |
|---------|---------|-------|---------|
| `hawkes_intensity` | lambda(t) = mu + Sum alpha*exp(-beta*(t-t_i)) | [0, +inf) | +0.247-0.345 (vol) |
| `hawkes_baseline` | mu = trade_count / window | [0, +inf) | — |
| `hawkes_branching_ratio` | (lambda - mu) / mu | [0, +inf) | >1 = cascade |

**Parameters:** alpha=0.5/sec, beta=1.0/sec, window=30s.
**Status:** All 3 working. Strong volatility predictor. Used by hierarchical_combiner L3.

---

## 4. Optional Features — NaN-Padded When Absent (82 total)

All 82 optional features are currently **ALL-NaN** due to missing API wiring (K2 issue).

### 4.1 Whale Flow (12 features) — ALL-NaN

**File:** `rust/ing-features/src/whale_flow.rs` | **Prefix:** `whale_`
**Source:** Hyperliquid position tracking API (not wired)

| Feature | Window | Description |
|---------|--------|-------------|
| `whale_net_flow_1h` | 1h | Sum(position_changes), positive = buying |
| `whale_net_flow_4h` | 4h | |
| `whale_net_flow_24h` | 24h | |
| `whale_flow_normalized_1h` | 1h | flow / rolling_avg_abs_flow |
| `whale_flow_normalized_4h` | 4h | |
| `whale_flow_momentum` | 1h/4h | flow_1h - flow_4h (acceleration) |
| `whale_flow_intensity` | — | |flow_1h| / avg_|flow| |
| `whale_flow_roc` | — | Rate of change of flow |
| `whale_buy_ratio` | — | Fraction of whales buying [0,1] |
| `whale_directional_agreement` | — | Net directional consensus [-1,1] |
| `active_whale_count` | — | Whales making changes |
| `whale_total_activity` | — | Sum of absolute flows |

**Config:** `ing.toml` has `[features.whale_flow]` with threshold=100k USD, but position tracker not implemented.
**Blocks:** `meta_labeling` algorithm (requires `whale_directional_agreement`).

---

### 4.2 Liquidation Risk (13 features) — ALL-NaN

**File:** `rust/ing-features/src/liquidation.rs` | **Prefix:** `liquidation_`
**Source:** Hyperliquid liquidation levels API (not wired)

| Feature | Trigger | Description |
|---------|---------|-------------|
| `liquidation_risk_above_1pct` | Price up 1% | USD at risk (shorts liquidated) |
| `liquidation_risk_above_2pct` | Price up 2% | |
| `liquidation_risk_above_5pct` | Price up 5% | |
| `liquidation_risk_above_10pct` | Price up 10% | |
| `liquidation_risk_below_1pct` | Price down 1% | USD at risk (longs liquidated) |
| `liquidation_risk_below_2pct` | Price down 2% | |
| `liquidation_risk_below_5pct` | Price down 5% | |
| `liquidation_risk_below_10pct` | Price down 10% | |
| `liquidation_asymmetry` | — | risk_above_5 / risk_below_5 [-1,1] |
| `liquidation_intensity` | — | total_risk_5 / total_OI [0,1] |
| `positions_at_risk_count` | Within +/-5% | Count |
| `largest_position_at_risk` | Within +/-5% | Max single position USD |
| `nearest_cluster_distance` | — | % distance to nearest cluster |

**Theory:** Liquidation cascades (Cont & Wagalath 2016, Brunnermeier & Pedersen 2009).

---

### 4.3 Concentration (15 features) — ALL-NaN

**File:** `rust/ing-features/src/concentration.rs` | **Prefix:** `top_` / `conc_`
**Source:** Depends on whale flow (K2 prerequisite)

| Feature | Description | Range |
|---------|-------------|-------|
| `top5_concentration` | Fraction of OI held by top 5 | [0, 1] |
| `top10_concentration` | Top 10 | [0, 1] |
| `top20_concentration` | Top 20 | [0, 1] |
| `top50_concentration` | Top 50 | [0, 1] |
| `conc_herfindahl_index` | Sum(share^2) | [0, 1] |
| `conc_gini_coefficient` | Lorenz curve inequality | [0, 1] |
| `conc_theil_index` | Entropy-based inequality | [0, ln(N)] |
| `conc_whale_retail_ratio` | Whale OI / retail OI | [0, +inf) |
| `conc_whale_fraction` | Whale positions / total | [0, 1] |
| `conc_whale_avg_size_ratio` | Whale avg / retail avg | [1, +inf) |
| `conc_concentration_change_1h` | Delta top10 over 1h | (-inf, +inf) |
| `conc_hhi_roc` | HHI rate of change | (-inf, +inf) |
| `conc_concentration_trend` | 1=rising, 0=stable, -1=falling | {-1, 0, 1} |
| `conc_position_count` | Total positions | [0, +inf) |
| `conc_whale_position_count` | Whale-sized positions | [0, +inf) |

**Blocks:** `meta_labeling` algorithm (requires `conc_hhi`).

---

### 4.4 Regime Detection (20 features) — ALL-NaN / CONSTANT

**File:** `rust/ing-features/src/regime/mod.rs` | **Prefix:** `regime_`

**Absorption (4):** `regime_absorption_1h`, `regime_absorption_4h`, `regime_absorption_24h`, `regime_absorption_zscore` — ALL-NaN

**Divergence (7):** `regime_divergence_1m`, `regime_divergence_5m`, `regime_divergence_15m`, `regime_divergence_1h`, `regime_divergence_4h`, `regime_divergence_24h`, `regime_divergence_zscore` — ALL-NaN

**Kyle lambda (1):** `regime_kyle_lambda` — **CONSTANT** (std=0.000000, K3 issue)

**Churn (4):** `regime_churn_1h`, `regime_churn_4h`, `regime_churn_24h`, `regime_churn_zscore` — ALL-NaN

**Range (4):** `regime_range_pos_4h`, `regime_range_pos_24h`, `regime_range_pos_1w`, `regime_range_width_24h` — ALL-NaN

**Composite (3):** `regime_accumulation_score` (**CONSTANT** 0.4429, K3), `regime_distribution_score` (**CONSTANT**, K3), `regime_clarity` (ALL-NaN)

**Impact:** Hierarchical combiner L1 uses `regime_divergence_1h_last` — currently gets NaN. Resolves when K2 is fixed.

---

### 4.5 GMM Classification (8 features) — ALL-NaN

**File:** `rust/ing/src/ml/regime.rs` | **Prefix:** `regime_prob_` / `regime`

| Feature | Range | Description |
|---------|-------|-------------|
| `regime` | {0,1,2,3,4} | GMM class label |
| `regime_prob_accumulation` | [0, 1] | Class probability |
| `regime_prob_markup` | [0, 1] | |
| `regime_prob_distribution` | [0, 1] | |
| `regime_prob_markdown` | [0, 1] | |
| `regime_prob_ranging` | [0, 1] | |
| `regime_confidence` | [0, 1] | max(prob) |
| `regime_entropy` | [0, ln(5)] | Shannon entropy of probs |

**GMM inputs:** [kyle_lambda, vpin, absorption_zscore, hurst, whale_net_flow_1h] — 5/5 inputs are dead.
**Config:** `gmm_model_path` commented out in `ing.toml`.

---

### 4.6 Cross-Symbol (3 features) — ALL-NaN

**File:** `rust/ing-features/src/cross_symbol.rs` | **Prefix:** `cross_`

| Feature | Description |
|---------|-------------|
| `cross_obi_divergence` | OBI_self - mean(OBI_others) |
| `cross_obi_mean` | mean(OBI_others) |
| `cross_obi_dispersion` | std(all_OBI_values) |

**Architecture:** Shared `CrossSymbolState` via `Arc<RwLock<HashMap>>`. IC +0.331-0.354 when active.
**Note:** May be partially live depending on multi-symbol config — need verification.

---

### 4.7 Heatmap (8 features) — ALL-NaN

**File:** `rust/ing-features/src/heatmap.rs` | **Prefix:** `hm_`
**References:** Cont & Wagalath (2016), Brunnermeier & Pedersen (2009)

| Feature | Description |
|---------|-------------|
| `hm_nearest_cluster_dist` | Min distance to liquidation cluster ($) |
| `hm_cluster_mass_ratio` | H(k*) / mean(H) (concentration) |
| `hm_cascade_chain_length` | Consecutive bins > threshold (domino potential) |
| `hm_asymmetric_cascade_pot` | Up vs down mass ratio [-1, 1] |
| `hm_absorption_capacity` | Book depth / liquidation pressure |
| `hm_cluster_velocity` | d(d_min)/dt (approaching? <0 = yes) |
| `hm_mass_weighted_distance` | Centre of liquidation gravity |
| `hm_heatmap_entropy` | Shannon entropy of mass distribution |

**Discretization:** 200 bins (10 bps each), +/-10% from midprice. Threshold $1M clusters, $100k chains.
**Used by:** `cascade_probability` algorithm.

---

## 5. Signal Hierarchy

### Top 10 Features by Directional IC (@ 5s, all symbols)

| Rank | Feature | BTC IC | ETH IC | SOL IC |
|------|---------|--------|--------|--------|
| 1 | `imbalance_qty_l1` | +0.453 | +0.447 | +0.466 |
| 2 | `imbalance_qty_l5` | +0.456 | +0.431 | +0.453 |
| 3 | `imbalance_notional_l5` | +0.456 | +0.431 | +0.453 |
| 4 | `imbalance_orders_l5` | +0.435 | +0.438 | +0.455 |
| 5 | `imbalance_depth_weighted` | +0.434 | +0.410 | +0.425 |
| 6 | `imbalance_qty_l10` | +0.434 | +0.410 | +0.425 |
| 7 | `raw_bid_orders_5` | +0.425 | +0.406 | +0.416 |
| 8 | `raw_bid_depth_5` | +0.420 | +0.405 | +0.415 |
| 9 | `raw_ask_depth_5` | -0.424 | -0.404 | -0.415 |
| 10 | `imbalance_pressure_ask` | -0.417 | -0.387 | -0.404 |

8/10 are imbalance variants capturing the same underlying order book asymmetry.

### Top 5 Features by Volatility IC (@ 5s)

| Rank | Feature | BTC IC | ETH IC | SOL IC |
|------|---------|--------|--------|--------|
| 1 | `hawkes_intensity` | +0.345 | +0.281 | +0.247 |
| 2 | `vol_returns_5m` | +0.328 | +0.326 | +0.293 |
| 3 | `flow_count_30s` | +0.318 | +0.281 | +0.253 |
| 4 | `vol_returns_1m` | +0.323 | +0.318 | +0.295 |
| 5 | `vol_parkinson_5m` | +0.318 | +0.312 | +0.295 |

**Key insight:** Direction and volatility features are perfectly orthogonal — zero cross-IC.

### Signal Decay Profile (imbalance_qty_l1)

| Horizon | IC (BTC) | IC (ETH) | IC (SOL) |
|---------|----------|----------|----------|
| 1s | +0.447 | +0.440 | +0.460 |
| 5s | +0.453 | +0.447 | +0.466 |
| 30s | ~+0.26 | ~+0.24 | ~+0.22 |
| 5m | ~+0.09 | ~+0.08 | ~+0.07 |
| 15m | ~+0.06 | ~+0.05 | ~+0.04 |

**Half-life:** ~30s (SOL ~15s fastest, BTC ~30s slowest).

---

## 6. Data Quality Issues (K1-K6)

| Issue | Severity | Status | Description | Impact |
|-------|----------|--------|-------------|--------|
| K1 | Critical | FIXED | Docker volume mount — data lost on restart | — |
| K2 | High | OPEN | 56 dead features (whale/liquidation/concentration/GMM) | 82 ALL-NaN features, blocks meta_labeling |
| K3 | Medium | OPEN | regime_accumulation_score constant (0.4429, std=0) | Depends on K2 whale flow |
| K4 | Low | MONITORING | WebSocket gaps 10-12/hr, max 13.4s | Median cadence 100ms OK |
| K5 | Medium | FIXED | 6-day data gap Jun 4-10 (zombie process) | Watchdog added |
| K6 | Low | ACCEPTED | 17 days missing Apr-May, unrecoverable | No backfill source |

**K2 fix effort:** 2-4h to investigate Hyperliquid position/liquidation REST API and wire into ingestor.

---

## 7. Feature Quality Summary

| Category | Count | Status | IC Type | Dominant? |
|----------|-------|--------|---------|-----------|
| Raw | 10 | Live | Directional (depth) | Top-10 |
| Imbalance | 8 | Live | **Directional (0.45+)** | **#1** |
| Flow | 12 | Live | Volatility + weak dir | — |
| Volatility | 9 | Live | Volatility (0.33) | Top-5 vol |
| Entropy | 27 | Live | Regime gating | IC lift +22% |
| Context | 12 | Live | Funding (algo input) | funding_reversion |
| Trend | 15 | Live | Weak long-horizon | — |
| Medium Freq | 16 | Live | Warmup-dependent | — |
| Illiquidity | 12 | Live | Impact measures | — |
| Toxicity | 10 | Live | Volatility | VPIN gating |
| Derived | 15 | Live | Composite | — |
| Microstructure | 5 | Live | OBI dynamics | hier_combiner L2 |
| Resilience | 3 | Live | Recovery speed | — |
| Hawkes | 3 | Live | Volatility (0.35) | hier_combiner L3 |
| Whale Flow | 12 | **Dead** (K2) | — | — |
| Liquidation | 13 | **Dead** (K2) | — | — |
| Concentration | 15 | **Dead** (K2) | — | — |
| Regime | 20 | **Dead** (K2/K3) | — | — |
| GMM | 8 | **Dead** (K2) | — | — |
| Cross-Symbol | 3 | **Dead?** | Directional (0.35) | Verify |
| Heatmap | 8 | **Dead** (K2) | — | — |
| **Total** | **236** | **154 live / 82 dead** | | |

---

## 8. Configuration

**File:** `config/ing.toml`

| Setting | Value | Status |
|---------|-------|--------|
| Emission interval | 100ms | Active |
| Trade buffer | 60s | Active |
| Book levels | 10 | Active |
| Whale flow threshold | 100,000 USD | Configured, not wired |
| GMM regime model | — | Commented out |
| Output format | Parquet + zstd | Active |
| Data directory | `../data/features` | Active (relative path) |
| Redis publish | 500ms interval | Active |
| Dashboard | disabled | — |

---

## 9. File Reference

### Feature Implementation (Rust)

| File | LOC | Category |
|------|-----|----------|
| `rust/ing-features/src/lib.rs` | 474 | Master struct, to_vec(), names_all() |
| `rust/ing-features/src/entropy.rs` | ~400 | 27 entropy features |
| `rust/ing-features/src/medium_freq.rs` | ~350 | 16 MF indicators |
| `rust/ing-features/src/regime/mod.rs` | ~300 | 20 regime features |
| `rust/ing-features/src/concentration.rs` | ~250 | 15 concentration |
| `rust/ing-features/src/illiquidity.rs` | ~250 | 12 illiquidity |
| `rust/ing-features/src/trend.rs` | ~250 | 15 trend |
| `rust/ing-features/src/derived.rs` | ~220 | 15 composites |
| `rust/ing-features/src/whale_flow.rs` | ~200 | 12 whale flow |
| `rust/ing-features/src/liquidation.rs` | ~200 | 13 liquidation |
| `rust/ing-features/src/toxicity.rs` | ~200 | 10 toxicity |
| `rust/ing-features/src/flow.rs` | ~180 | 12 trade flow |
| `rust/ing-features/src/volatility.rs` | ~180 | 9 volatility |
| `rust/ing-features/src/imbalance.rs` | ~150 | 8 imbalance |
| `rust/ing-features/src/heatmap.rs` | ~150 | 8 heatmap |
| `rust/ing-features/src/context.rs` | ~150 | 12 context |
| `rust/ing-features/src/raw.rs` | ~100 | 10 raw |
| `rust/ing-features/src/microstructure.rs` | ~100 | 5 microstructure |
| `rust/ing-features/src/resilience.rs` | ~100 | 3 resilience |
| `rust/ing-features/src/hawkes.rs` | ~80 | 3 Hawkes |
| `rust/ing-features/src/cross_symbol.rs` | ~80 | 3 cross-symbol |

### Key Documentation

| File | Content |
|------|---------|
| `FEATURES.md` | Complete feature manifest with formulas |
| `docs/research/new/9_6/full_ic_scan_report.md` | 207-feature IC scan |
| `docs/research/new/9_6/ic_validation_report.md` | Temporal OOS + conditional IC |
| `docs/korrektur_tasks.md` | K1-K6 data quality issues |
| `config/ing.toml` | Runtime configuration |
| `rust/ing/src/output/schema.rs` | Parquet schema from names_all() |
