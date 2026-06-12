# Algorithm Catalogue & Status Report

**Date:** 2026-06-12
**Codebase:** 26 algorithms, 7,656 LOC in `scripts/algorithms/`
**Test conditions:** Walk-forward OOS, 3-day training window, P20/P80 z-score entry, 100min horizon, 1.61 bps RT fee
**OOS window:** 13 dates (2026-05-07 to 2026-05-23) | **Symbols:** BTC, ETH, SOL | **Bars:** 5min from 100ms ticks

---

## 1. Executive Summary

26 algorithms are implemented in `scripts/algorithms/`, all conforming to the `MicrostructureAlgorithm` ABC. Of these:

- **5 deployable** (Tier 1-2): net positive OOS after costs
- **1 ML deployable**: mean_reversion_detector (OOS AUC 0.55-0.58)
- **1 preliminary**: hierarchical_combiner (2-day data, promising but unvalidated)
- **12 unprofitable** (Tier 3): net negative after 1.61 bps RT costs
- **3 ML failures**: momentum_continuation, meta_labeling, regime_conditioned_lgbm
- **4 infrastructure**: regime_state_machine, change_point_detector, convolver, knn_retrieval (support roles)

Portfolio cross-correlation between winners is <0.35, with jump_detector and funding_reversion near-zero (~0.00). BTC is dominated by 3f_liquidity (Sharpe 9.2-12.1), ETH/SOL by jump_detector (Sharpe 6.2).

---

## 2. Framework Architecture

### Base Class (`scripts/algorithms/base.py`, 113 LOC)

```
MicrostructureAlgorithm ABC
  name() -> str                          # unique identifier
  alg_features() -> list[AlgorithmFeature]  # output descriptors with warmup
  required_columns() -> list[str]        # input feature names
  step(tick: dict) -> dict               # process one tick
  reset() -> None                        # clear internal state
  run_batch(df) -> pd.DataFrame          # vectorized path (default: row-iterate)
```

**Properties:** `bar_level` (False = tick-level, True = 5min bars), `warmup` (max across features).

### Registry (`scripts/algorithms/registry.py`, 29 LOC)

- `@register` decorator auto-registers algorithms by `name()`
- `get_algorithm(name, **kwargs)` instantiates from config
- `list_algorithms()` returns sorted list of 26 names
- `autodiscover.py` imports all `.py` files in directory on startup

### Runner (`scripts/algorithms/runner.py`, 226 LOC)

`AlgorithmRunner` orchestrates batch execution: loads data, runs each algorithm's `run_batch()`, collects results, applies warmup blanking.

### Configuration (`config/algorithms.toml`, 192 lines)

Every algorithm's constructor kwargs are configurable without code changes. Grouped: Order Flow (A), Hawkes (B), Volatility/Jump (C), Toxicity (D), Entropy (E), Cross-Asset (F), Latent State (G), Cascade (H), ML Waves (J), Pattern (I), Ensemble.

---

## 3. Tier 1 — Deployable (Net Positive All Symbols)

### 3.1 jump_detector — Total +23,199 bps

**Lee-Mykland (2008) nonparametric jump test with post-jump mean-reversion.**

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL | Win Rate | Max Daily Loss |
|--------|--------|---------------|--------|-----------|----------|----------------|
| BTC    | 1,678  | +1.03         | 1.6    | +1,722    | 54%      | -1,981         |
| ETH    | 1,678  | +6.47         | 6.2    | +10,861   | 62%      | -3,412         |
| SOL    | 1,678  | +6.33         | 6.2    | +10,616   | 69%      | -3,311         |

**File:** `scripts/algorithms/jump_detector.py` (253 LOC)
**Required columns:** `raw_midprice`
**Parameters:** `window=100, significance=3.0, reversion_horizon=50`
**Warmup:** 100 ticks
**Signal polarity:** low_long (low jump ratio = stable = long)

**Features (4):**
- `alg_jump_statistic` — L(t) = |return| / local_vol (Lee-Mykland test statistic)
- `alg_jump_detected` — 1.0 if L(t) > significance threshold
- `alg_jump_magnitude` — Signed return at jump (0 if no jump)
- `alg_post_jump_reversion` — Price change since last jump (**primary signal**)

**Mathematics:**
- Local volatility: `sigma_BV = sqrt((pi/2) * mean(|r_i| * |r_{i-1}|))` (bipower variation)
- Test statistic: `L(t) = |r_t| / sigma_BV(t)`
- Jump detection: `L(t) > c` (default c=3.0, ~3-sigma event)
- Reversion: `-(log(p_t/p_J) / r_J)` within H ticks of jump
- Full `run_batch()` vectorization

---

### 3.2 3f_liquidity — Total +16,028 bps

**Medium-frequency spread+depth+VWAP composite z-score.**

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL | Win Rate |
|--------|--------|---------------|--------|-----------|----------|
| BTC    | 950    | +5.58         | 9.2    | +5,302    | 62%      |
| ETH    | 915    | +7.83         | 7.8    | +7,162    | 62%      |
| SOL    | 954    | +3.74         | 3.2    | +3,564    | 62%      |

**File:** `scripts/alpha/paper_trader.py` (not a standard algorithm — implemented as alpha pipeline signal)
**Horizon:** 50min bars (vs 100min for standard algorithms)
**Signal:** Equal-weight z-score of `mf_spread_bps + mf_depth_imbalance + mf_vwap_deviation`

**Key finding:** Dominates BTC with Sharpe 9.2-12.1. First profitable signal after transaction costs. Strongest single-symbol algorithm.

---

### 3.3 funding_reversion — Total +14,459 bps

**Crypto-native funding rate mean-reversion with saturation.**

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL | Win Rate | Max Daily Loss |
|--------|--------|---------------|--------|-----------|----------|----------------|
| BTC    | 1,678  | +0.26         | 0.4    | +429      | 38%      | -2,327         |
| ETH    | 1,678  | +6.12         | 6.1    | +10,265   | 54%      | -2,629         |
| SOL    | 1,678  | +2.24         | 1.7    | +3,766    | 54%      | -3,786         |

**File:** `scripts/algorithms/funding_reversion.py` (150 LOC)
**Required columns:** `ctx_funding_rate, ctx_funding_zscore, ctx_premium_bps`
**Parameters:** `zscore_entry=2.0, momentum_span=100, premium_weight=0.3, halflife_window=200`
**Warmup:** 100 ticks
**Signal polarity:** high_long

**Features (4):**
- `alg_funding_signal` — `-sign(zscore) * min(|z|/z_entry, 3)/3` (**primary signal**)
- `alg_funding_momentum` — EMA of funding rate
- `alg_premium_divergence` — Weighted combo: funding_zscore vs premium_bps
- `alg_funding_halflife_ticks` — OU half-life from lag-1 AR on z-score

**Mathematics:**
- Entry: `|zscore| >= threshold -> signal = -sign(zscore) * min(|z|/entry, 3.0) / 3.0`
- Premium divergence: blend `funding_z + premium_bps / 10`
- Half-life: `hl = -ln(2) / ln(rho)` where rho = AR(1) coefficient

---

### 3.4 optimal_entry — Total +13,679 bps

**SPRT (Wald 1947) on Kalman OU-filtered L1 imbalance innovations.**

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL | Win Rate | Max Daily Loss |
|--------|--------|---------------|--------|-----------|----------|----------------|
| BTC    | 1,678  | +0.90         | 1.1    | +1,504    | 46%      | -2,327         |
| ETH    | 1,678  | +5.89         | 5.2    | +9,877    | 62%      | -3,645         |
| SOL    | 1,678  | +1.37         | 1.0    | +2,298    | 54%      | -4,762         |

**File:** `scripts/algorithms/optimal_entry.py` (282 LOC)
**Required columns:** `imbalance_qty_l1`
**Parameters:** `theta=0.1, sigma_process=0.01, sigma_obs=0.1, dt=0.1, sprt_drift=0.001, alpha_error=0.05, beta_error=0.20`
**Warmup:** 50 ticks
**Signal polarity:** high_long
**Known bug:** `run_batch()` hardcodes `sigma_process=0.01` instead of `self._sigma_process`

**Features (3):**
- `alg_sprt_statistic` — Cumulative log-likelihood ratio
- `alg_entry_signal` — +1 long, -1 short, 0 no signal (**primary signal**)
- `alg_cumulative_evidence` — Normalized |S| / A

**Mathematics:**
- H0: `nu ~ N(0, sigma^2)` vs H1: `nu ~ N(mu, sigma^2)` where mu = sprt_drift
- Log-likelihood: `Lambda = (mu/sigma^2) * nu - mu^2/(2*sigma^2)`
- Boundaries: `A = log((1-beta)/alpha)`, `B = log(beta/(1-alpha))`
- Reset S after each decision (continuous monitoring)
- Inherently sequential — loop-based `run_batch()`

---

## 4. Tier 2 — Symbol-Specific Alpha

### 4.1 surprise_signal — Total +3,505 bps

**Entropy regime transition detection via ROC z-score.**

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL | Win Rate | Max Daily Loss |
|--------|--------|---------------|--------|-----------|----------|----------------|
| BTC    | 954    | -4.78         | -8.3   | -4,563    | 15%      | -1,316         |
| ETH    | 1,010  | +2.85         | 3.1    | +2,878    | 54%      | -2,493         |
| SOL    | 981    | +5.29         | 6.7    | +5,190    | 46%      | -518           |

**File:** `scripts/algorithms/surprise_signal.py` (145 LOC)
**Required columns:** `ent_book_shape, ent_tick_5s`
**Parameters:** `roc_window=50`
**Warmup:** 100 ticks
**Signal polarity:** low_long (ordering = long, disordering = short)
**Note:** Fails on BTC (Sharpe -8.3). Deploy ETH/SOL only.

**Features (3):**
- `alg_entropy_surprise` — Z-score of entropy rate-of-change (**primary signal**)
- `alg_entropy_roc` — Entropy rate of change
- `alg_regime_transition_prob` — P(regime transition) = erf(|z| / sqrt(2))

---

## 5. Preliminary — Promising but Insufficient Data

### 5.1 hierarchical_combiner — 3-Layer IC-Weighted Architecture

**File:** `scripts/algorithms/hierarchical_combiner.py` (259 LOC)
**Data:** 2 days only (2026-06-08 to 2026-06-10), walk-forward IC-weighted, 4-fold expanding window

| Symbol | IC     | Sharpe | Horizon |
|--------|--------|--------|---------|
| BTC    | +0.178 | +1.25  | 5h      |
| ETH    | +0.248 | +1.71  | 5h      |
| SOL    | +0.359 | +2.40  | 3.3h    |

**Architecture:**
- **Layer 1 (Slow directional):** regime_divergence_1h, spread_bps, trend_ema_short -> {-1, 0, +1}
- **Layer 2 (Fast entry):** OBI_l1, cross_obi, queue_position_bid, vwap_deviation, obi_velocity -> [-1,+1] gated by L1
- **Layer 3 (Vol sizing):** hawkes_intensity, flow_count_30s, vol_returns_5m -> [0,1] inverse sigmoid

**Required columns (15):**
- L1: `regime_divergence_1h_last, raw_spread_bps_mean, trend_ema_short_last`
- L2: `imbalance_qty_l1_mean, cross_obi_mean_mean, micro_queue_position_bid_mean, flow_vwap_deviation_mean, micro_obi_velocity_mean`
- L3: `hawkes_intensity_mean, flow_count_30s_sum, vol_returns_5m_last`

**Features (4):**
- `alg_hier_directional_bias` — Slow directional state {-1, 0, +1}
- `alg_hier_entry_timing` — Fast entry strength [-1, +1]
- `alg_hier_vol_scale` — Vol-adjusted size scalar [0, 1]
- `alg_hier_composite` — L1 * |L2| * L3 (**primary signal**)

**Assessment:** First algorithm explicitly addressing adverse selection via L2 gating. Needs 7+ days data for statistical confidence. Monotonically increasing IC across folds is suspicious.

---

## 6. ML Algorithms — Deployed

### 6.1 mean_reversion_detector — OOS AUC 0.55-0.58

**LightGBM binary classifier predicting P(reversion) over 100min horizon.**

**File:** `scripts/algorithms/mean_reversion_detector.py` (229 LOC)
**Required columns:** `vol_returns_5m_last, ent_tick_1m_mean, trend_hurst_300_mean, imbalance_qty_l1_mean, toxic_vpin_50_mean, raw_midprice_mean, mf_ema_15m_last`
**Parameters:** `entropy_threshold=0.70, zscore_threshold=2.0, reversion_prob_thresh=0.65`
**Warmup:** 50 bars

| Symbol | OOS AUC | IS AUC | OOS/IS Ratio |
|--------|---------|--------|--------------|
| BTC    | 0.577   | 0.743  | 0.78         |
| ETH    | 0.564   | 0.720  | 0.78         |
| SOL    | 0.552   | 0.746  | 0.74         |

**Features (4):**
- `alg_mr_signal` — `-sign(zscore) * P(reversion)` when gated (**primary signal**)
- `alg_mr_probability` — Model P(reversion) [0, 1]
- `alg_mr_zscore` — Price displacement z-score
- `alg_mr_entropy_gate` — 1 if entropy > 0.70 (ranging regime, **inverted** gate)

**Feature importance (normalized gain):** zscore 35-44%, imbalance_qty_l1 9-24%, trend_hurst_300 10-22%, toxic_vpin_50 9-17%

**Note:** Falls back to neutral (signal=0, prob=0.5) if no trained model files exist.

---

## 7. ML Algorithms — Failed

### 7.1 momentum_continuation — OOS AUC 0.37-0.45 (overfit)

**Logistic regression P(continuation) over 20-bar horizon.**

**File:** `scripts/algorithms/momentum_continuation.py` (203 LOC)
**Failure mode:** Classic overfit — IS AUC 0.65, OOS collapse to 0.37-0.45.
**Entropy gate:** Active when entropy LOW (trending regime). Complementary to mean_reversion_detector.

**Features (3):** `alg_mc_signal`, `alg_mc_confidence`, `alg_mc_entropy_gate`

### 7.2 meta_labeling — OOS AUC 0.44-0.48 (missing features)

**Logistic regression precision filter (De Prado Ch. 3 triple-barrier labels).**

**File:** `scripts/algorithms/meta_labeling.py` (188 LOC)
**Failure mode:** Missing input features — `conc_hhi`, `whale_directional_agreement`, `regime_clarity` are 100% NaN (K2 dead features).

**Features (3):** `alg_meta_probability`, `alg_meta_side`, `alg_meta_size`

### 7.3 regime_conditioned_lgbm — Insufficient regime samples

**Per-regime LightGBM models with global fallback.**

**File:** `scripts/algorithms/regime_conditioned_lgbm.py` (296 LOC)
**Failure mode:** RSM only classifies 37% of bars into distinct regimes; insufficient per-regime training samples.
**Routing:** regime {0,1} -> ranging model, {2,3} -> trending model, {4,5} -> volatile model.

**Features (4):** `alg_rlgbm_signal`, `alg_rlgbm_predicted_return`, `alg_rlgbm_regime_used`, `alg_rlgbm_regime_confidence`

---

## 8. Tier 3 — No Edge After Costs

All net negative aggregate OOS at 100min horizon with 1.61 bps RT.

| #  | Algorithm          | LOC | Total bps  | BTC Sharpe | ETH Sharpe | SOL Sharpe | Primary Feature              | Method                           |
|----|--------------------|-----|------------|------------|------------|------------|------------------------------|----------------------------------|
| 6  | oi_divergence      | 167 | -1,721     | -5.3       | -5.7       | +2.1       | `alg_oi_price_divergence`    | OI vs price trend divergence     |
| 7  | regime_gated       | 126 | -1,748     | -2.4       | -0.4       | -0.0       | `alg_regime_gated_imbalance` | Entropy-gated L1 imbalance       |
| 8  | entropy_momentum   | 178 | -2,600     | -6.4       | -0.2       | -2.7       | `alg_entropy_gated_momentum` | Momentum in low-entropy regime   |
| 9  | propagator         | 295 | -4,118     | -2.4       | -1.2       | -3.9       | `alg_transient_impact`       | Bouchaud impact decomposition    |
| 10 | hawkes_intensity   | 341 | -5,443     | +0.8       | -2.6       | -4.3       | `alg_bid_ask_hawkes_imbalance` | Self-exciting point process    |
| 11 | trade_through      | 170 | -5,739     | -5.1       | -4.2       | +0.2       | `alg_trade_through_imbalance`| Queue depletion (Cont 2013)      |
| 12 | weighted_ofi       | 237 | -6,183     | -4.8       | -0.6       | -3.6       | `alg_weighted_ofi`           | Depth-decay weighted OFI         |
| 14 | switching_ou       | 266 | -6,230     | -3.5       | +0.7       | -6.0       | `alg_switching_ou_state`     | HMM 2-regime OU Kalman           |
| 15 | vpin_regime        | 121 | -7,331     | -4.7       | -1.7       | -3.5       | `alg_vpin_gated_imbalance`   | VPIN-gated imbalance             |
| 16 | kalman_imbalance   | 134 | -7,517     | -2.4       | -0.0       | -7.2       | `alg_kalman_signal_strength` | OU Kalman filtered imbalance     |
| 17 | bipower_jump       | 208 | -32,079    | -14.0      | -9.7       | -9.8       | `alg_jump_ratio`             | Barndorff-Nielsen 2004 BV/RV     |
| 18 | spread_decomp      | 130 | -34,510    | -10.7      | -14.8      | -8.4       | `alg_adverse_component`      | Huang-Stoll 1997 decomposition   |

---

## 9. Infrastructure Algorithms (Non-Trading)

These produce features consumed by other algorithms or the trading system but do not generate tradeable signals themselves.

### 9.1 regime_state_machine (258 LOC)

6-state discrete regime classifier: ACCUMULATION, DISTRIBUTION, TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE_NOISE. Scoring via binary conditions, argmax selection, min 5-bar hysteresis. Outputs `alg_rsm_regime` and `alg_rsm_trade_allowed` (0 in VOLATILE_NOISE). Used as input to `regime_conditioned_lgbm`.

### 9.2 change_point_detector (252 LOC)

Dual method: CUSUM (Page 1954) + Bayesian OCD (Adams & MacKay 2007). Produces `alg_cpd_cusum_signal`, `alg_cpd_run_length`, `alg_cpd_change_prob`, `alg_cpd_regime_age`. Bar-level. Calibration window = 200 bars.

### 9.3 convolver (305 LOC + 455 LOC kernels)

SVD-discovered pattern kernels scoring 6 event types (breakout/turtle/trap, bull/bear). 4-channel decomposition: body, upper_wick, lower_wick, volume. Cosine similarity scoring. Falls back to analytical basis (10 functions) if no trained kernels. Warmup: 20,400 ticks. Pre-discovery phase — no training results yet.

### 9.4 knn_retrieval (245 LOC)

Non-parametric nearest-neighbor state retrieval. Ledoit-Wolf shrinkage covariance, Mahalanobis distance via KD-tree. Ring buffer of 5,000 states. Signal fires only if `|expected_return| > 2 bps cost AND win_rate > 0.60`.

### 9.5 cascade_probability (272 LOC)

Online logistic regression predicting liquidation cascades from 8 heatmap features + 3 interactions. SGD with delayed targets (3,000-tick horizon). Class weighting for rare cascade events.

---

## 10. Ensemble Methods

**File:** `scripts/algorithms/ensemble.py` (262 LOC)

Three combination methods in the `Ensemble` class:

| Method         | Description                                            | Config Key   |
|----------------|--------------------------------------------------------|--------------|
| `equal_weight` | Mean of z-scored primary signals (1/N weights)         | Default      |
| `ic_weight`    | Weight by trailing |IC| (Spearman, lookback=5000)     | ic_lookback  |
| `regime_switch`| Trending algos in low-entropy, reverting in high       | regime_column|

**Default ensemble configuration** (`config/algorithms.toml`):
- Algorithms: `[jump_detector, optimal_entry, funding_reversion, surprise_signal, weighted_ofi]`
- Method: `equal_weight`
- IC lookback: 5000 ticks

**Primary signal features per algorithm:**
```
jump_detector     -> alg_post_jump_reversion
optimal_entry     -> alg_entry_signal
funding_reversion -> alg_funding_signal
surprise_signal   -> alg_entropy_surprise
weighted_ofi      -> alg_weighted_ofi
```

**Regime switch routing:**
- Low entropy (trending): jump_detector, optimal_entry, surprise_signal
- High entropy (ranging): funding_reversion, weighted_ofi

---

## 11. Cross-Algorithm Correlation

From `reports/signal_correlation.json`, pairwise Spearman between primary signals:

### BTC
| Pair                              | Correlation |
|-----------------------------------|-------------|
| 3f_liquidity x jump_detector      | 0.192       |
| 3f_liquidity x surprise_signal    | 0.338       |
| jump_detector x funding_reversion | -0.005      |
| funding_reversion x surprise      | 0.093       |

### ETH
| Pair                              | Correlation |
|-----------------------------------|-------------|
| 3f_liquidity x surprise_signal    | 0.449       |
| jump_detector x funding_reversion | -0.022      |

### SOL
| Pair                              | Correlation |
|-----------------------------------|-------------|
| 3f_liquidity x jump_detector      | 0.205       |
| 3f_liquidity x surprise_signal    | -0.150      |
| jump_detector x funding_reversion | 0.009       |

**Maximum pairwise:** 0.449 (3f_liquidity x surprise_signal on ETH). All pairs < 0.35 target except this one.

**Portfolio implication:** jump_detector and funding_reversion are essentially uncorrelated (~0.00 across all symbols) — ideal blending pair.

---

## 12. Feature Dependency Map

### Input features required (unique across all 26 algorithms):

**Order book (tick):** `imbalance_qty_l1, imbalance_qty_l5, imbalance_qty_l10, raw_bid_depth_5, raw_ask_depth_5, raw_midprice`
**Order book (bar):** `imbalance_qty_l1_mean, cross_obi_mean_mean, micro_queue_position_bid_mean, micro_obi_velocity_mean`
**Trade flow:** `flow_volume_1s, flow_aggressor_ratio_5s, flow_count_1s, flow_intensity, flow_count_30s_sum, flow_vwap_deviation_mean`
**Toxicity:** `toxic_vpin_10_mean, toxic_vpin_50, toxic_vpin_50_mean, toxic_effective_spread, toxic_realized_spread, toxic_index_mean`
**Entropy:** `ent_book_shape, ent_tick_5s, ent_tick_1m_mean, ent_rate_of_change_5s_mean, ent_permutation_returns_16_mean`
**Volatility:** `vol_returns_5m_last, vol_returns_5m_mean, vol_ratio_short_long_last`
**Trend:** `trend_momentum_60, trend_momentum_300, trend_momentum_300_mean, trend_hurst_300_mean, trend_ema_short_last, mf_ema_15m_last, mf_bb_pctb_5m_last`
**Regime:** `regime_divergence_1h_last, regime_accumulation_score_mean, regime_clarity_last`
**Context:** `ctx_funding_rate, ctx_funding_zscore, ctx_premium_bps, ctx_open_interest, ctx_oi_change_5m`
**Whale:** `whale_net_flow_4h_sum, whale_directional_agreement_last`
**Spread:** `raw_spread_bps_mean`
**Concentration:** `conc_hhi_last`
**Heatmap (8):** `hm_nearest_cluster_dist, hm_cluster_mass_ratio, hm_cascade_chain_length, hm_asymmetric_cascade_pot, hm_absorption_capacity, hm_cluster_velocity, hm_mass_weighted_distance, hm_heatmap_entropy`
**Cross-algorithm:** `alg_rsm_regime_last, alg_rsm_confidence_last, alg_conv_best_score_max, illiq_composite`

**Dead features (K2, 100% NaN):** `conc_hhi_last, whale_directional_agreement_last, regime_clarity_last` — blocks meta_labeling deployment.

---

## 13. Evaluation Infrastructure

### Evaluation Harness (`scripts/algorithms/evaluate.py`, 249 LOC)

- IC Analysis: Spearman rank correlation vs forward returns at horizons [1, 5, 10, 50, 100]
- Regime-stratified IC: conditional on entropy regimes (30th percentile)
- Drift analysis: post-fill drift via `MakerFillSimulator` (entry threshold 0.3)

### Swarm Evaluator (`scripts/swarm/evaluator.py`)

Fitness metrics:
- **Sharpe:** `(mean_return / std_return) * sqrt(315,360,000)` (annualized from 100ms ticks)
- **Mean IC:** Spearman over non-overlapping 1000-tick windows
- **Max Drawdown:** Peak-to-trough cumulative return
- **Signal Count/Day:** threshold = 0.5 * std(signal)
- **Turnover:** sum(|delta_position|) / n_days

### Optuna Optimizer (`scripts/swarm/optuna_optimizer.py`)

- Samplers: CMA-ES (continuous 35D), TPE (mixed), NSGA-II (multi-objective Pareto)
- Walk-forward: 2/3 train, 1/3 test (OOS)
- Guard rails: signal count > 50/day, turnover < 100/day, OOS Sharpe > 0, IS/OOS ratio < 3.0
- Deflated Sharpe ratio (Bailey & Lopez de Prado 2014)

### Alpha Pipeline Quality Gates (`config/alpha.toml`)

| Gate | Step          | Key Metric                | Threshold       |
|------|---------------|---------------------------|-----------------|
| G1   | Screening     | FDR-significant features  | >= 5 (PASS)     |
| G2   | Combining     | Composite IC, turnover    | IC >= 0.8*max   |
| G3   | Sizing        | Trade reduction           | >= 50%          |
| G4   | Validation    | OOS Sharpe, IS/OOS ratio  | >= 0.5, >= 0.7  |
| G5   | Regime        | IC improvement per regime | >= 1.5x         |
| G6   | Multi-freq    | Composite vs individual   | Sharpe up, DD down |
| G7   | Portfolio     | Diversification           | DD < 0.8*worst  |
| G8   | Paper         | Live stability            | Sharpe <= 2x BT |

---

## 14. IC & Signal Analysis

### Feature-Level IC Scan (3 days, 207 features, 6 horizons)

**Top directional features:**

| Feature                   | BTC IC  | ETH IC  | SOL IC  | Peak Horizon |
|---------------------------|---------|---------|---------|--------------|
| imbalance_qty_l1          | +0.453  | +0.447  | +0.466  | 1-5s         |
| imbalance_qty_l5          | +0.456  | +0.431  | +0.453  | 5s           |
| imbalance_orders_l5       | +0.435  | +0.438  | +0.455  | 1-5s         |
| imbalance_depth_weighted  | +0.434  | +0.410  | +0.425  | 5s           |
| cross_obi_mean            | +0.354  | +0.334  | +0.331  | 1-5s         |

**Signal decay:** IC 0.45 @ 1s -> 0.26 @ 30s -> 0.09 @ 5m -> 0.06 @ 15m. Half-life ~30s.

**Adverse selection finding:** Conditional IC (given mid-cross fills) drops to ~0. Mid-cross fills structurally eliminate directional IC — confirms adverse selection trap at microstructure frequencies.

**Strategic implication:** Microstructure imbalance signals require limit order execution at 1-5s horizon. Taker execution (11 bps RT) at these horizons is economically impossible (move size 0.5-2 bps << cost 11 bps).

---

## 15. Specced but Unimplemented

### aegis_maker — Regime-Intelligent Market Making

**File:** `docs/research/new/aegis_maker.txt`
**Method:** Avellaneda-Stoikov (2008) enriched with 6 microstructure algorithms (spread_decomp, kalman_imbalance, switching_ou, hawkes_intensity, optimal_entry, funding_reversion).
**Status:** Proposal only.

### EAMM — Entropy-Adaptive Market Making

**Status:** Spec exists (`nat eamm`), not implemented or deployable.

### Deferred ML Algorithms (Trigger-Gated)

From `docs/research/new/ml_algorithms.txt`:
- **HMM with Gaussian Emissions** — Trigger: 60+ days data + deployed mean_reversion_detector
- **Stacking Ensemble** — Trigger: 4+ deployed ML algorithms, max pairwise |rho| < 0.5
- **Online SGD Adaptation** — Trigger: 30%+ Sharpe degradation in any deployed model over 14 days

---

## 16. Known Issues

| Issue | Severity | Algorithm | Description |
|-------|----------|-----------|-------------|
| Bug   | Medium   | optimal_entry | `run_batch()` hardcodes `sigma_process=0.01` vs `self._sigma_process` |
| Data  | High     | meta_labeling | 3 required features 100% NaN (K2 dead features) |
| Data  | Medium   | regime_conditioned_lgbm | RSM classifies only 37% of bars |
| Overfit | Medium | momentum_continuation | IS AUC 0.65 -> OOS 0.37-0.45 |
| Data  | Low      | hierarchical_combiner | Only 2-day OOS, monotonic IC suspicious |
| Missing | Low    | convolver | No trained kernel files, uses analytical fallback |
| Missing | Low    | cascade_probability | Liquidation volume proxied by price moves (no API) |

---

## 17. File Reference

### Algorithm Implementations (by LOC)

| File | LOC | Category |
|------|-----|----------|
| convolver_kernels.py | 455 | Pattern library |
| hawkes_intensity.py | 341 | Hawkes process |
| convolver.py | 305 | Pattern scoring |
| regime_conditioned_lgbm.py | 296 | ML Wave 3a |
| propagator.py | 295 | Impact model |
| optimal_entry.py | 282 | SPRT + Kalman |
| cascade_probability.py | 272 | Online SGD cascade |
| switching_ou.py | 266 | HMM + dual Kalman |
| ensemble.py | 262 | Signal combination |
| hierarchical_combiner.py | 259 | 3-layer combiner |
| regime_state_machine.py | 258 | 6-state RSM |
| change_point_detector.py | 252 | CUSUM + Bayesian OCD |
| evaluate.py | 249 | Evaluation harness |
| knn_retrieval.py | 245 | Nearest-neighbor |
| weighted_ofi.py | 237 | Depth-decay OFI |
| mean_reversion_detector.py | 229 | LightGBM classifier |
| runner.py | 226 | Batch runner |
| calibrate_warmup.py | 226 | Warmup analysis |
| signal_adapter.py | 225 | Signal preprocessing |
| bipower_jump.py | 208 | Bipower variation |
| momentum_continuation.py | 203 | Logistic continuation |
| meta_labeling.py | 188 | Triple-barrier filter |
| entropy_momentum.py | 178 | Entropy-gated momentum |
| trade_through.py | 170 | Queue depletion |
| oi_divergence.py | 167 | OI vs price |
| funding_reversion.py | 150 | Funding mean-reversion |
| surprise_signal.py | 145 | Entropy surprise |
| kalman_imbalance.py | 134 | OU Kalman filter |
| spread_decomp.py | 130 | Spread decomposition |
| regime_gated.py | 126 | Entropy gate |
| vpin_regime.py | 121 | VPIN gate |
| base.py | 113 | ABC |
| regime_retune.py | 92 | Regime-conditional retune |
| autodiscover.py | 50 | Module auto-import |
| registry.py | 29 | Registration mechanism |
| __init__.py | 19 | Public API |
| **Total** | **7,656** | |

### Key Documentation

| File | Content |
|------|---------|
| `docs/research/ALGORITHMS.md` | OOS results, tier classification |
| `reports/best__mf_liquidity_signal.md` | 3f_liquidity detailed analysis |
| `docs/research/new/10_6/hierarchical_combiner_report.md` | Hierarchical combiner 2-day OOS |
| `reports/ml_training_wave1_3_20260608.md` | ML training results |
| `docs/research/new/ml_algorithms.txt` | 10 ML algorithm specs (5,754 lines) |
| `docs/research/new/9_6/full_ic_scan_report.md` | 207-feature IC scan |
| `docs/research/new/9_6/ic_validation_report.md` | Temporal OOS decay analysis |
| `reports/signal_correlation.json` | Cross-algorithm correlations |
| `docs/convolver_implementation/` | 14 documents on SVD pattern discovery |
| `reports/algo_mathematical_foundations.md` | Full mathematical derivations |
| `config/algorithms.toml` | All algorithm parameters |

### Test Suite

| File | Tests | Coverage |
|------|-------|----------|
| `scripts/algorithms/tests/test_algorithms.py` | Smoke tests (all 26 algos) | Output shape, NaN, warmup, reset |
| `scripts/algorithms/tests/test_winning_algos.py` | 25 tests | Mathematical verification of 5 winners |
| `scripts/algorithms/tests/test_ml_algorithms.py` | ML algorithm tests | Model loading, fallback, gating |
| `scripts/algorithms/tests/test_ensemble.py` | Ensemble tests | 3 methods, normalization, missing algos |
