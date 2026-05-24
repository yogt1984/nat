# NAT Terminal Tool Reference — Complete Command & Mathematical Documentation

**Date**: 2026-05-15
**Version**: Comprehensive reference for all ~100 `nat` commands

---

## Architecture Overview

NAT is a quantitative research platform for extracting alpha signals from Hyperliquid
perpetual futures. The data pipeline:

```
Hyperliquid WebSocket (100ms)
  -> Rust ingestor (191 features, 14 categories)
  -> Parquet files (hourly rotation)
  -> Python analysis (spectral, regime, profiling, backtesting)
  -> Signal validation -> Paper trading -> Live execution
```

All features are computed at 10 Hz (100ms intervals) per symbol (BTC, ETH, SOL).
The `nat` CLI unifies all operations under a single entry point.

---

## 1. Data Collection

### `nat start`
Start the Rust ingestor + watchdog + dashboard. Launches the WebSocket client
(`ws/client.rs`) that subscribes to Hyperliquid L2 order book and trade streams.
Each symbol runs in its own tokio task. Features emitted every 100ms via
`tokio::select! { biased; }` — WebSocket messages have priority over emission ticker.

### `nat stop`
Graceful shutdown. Flushes the `ArrowWriter` buffer to disk (unflushed data stays
at 0 bytes until `close()`). Kills stale processes.

### `nat status`
One-line health check. Reports: process alive, last parquet write timestamp,
row count, symbol coverage.

### `nat log`
Tail the ingestor log in real-time.

---

## 2. Feature Vector (191 features)

The ingestor computes 191 features across 15 categories. Key mathematical formulations:

### 2.1 Raw (10 features, `raw_*`)
Direct L2 order book measurements.

| Feature | Formula |
|---------|---------|
| `raw_midprice` | (P_bid + P_ask) / 2 |
| `raw_spread_bps` | (P_ask - P_bid) / midprice x 10000 |
| `raw_microprice` | (V_ask x P_bid + V_bid x P_ask) / (V_bid + V_ask) — Gatheral & Oomen (2010) |
| `raw_bid_depth_5` | sum(V_bid, levels 1-5) |
| `raw_ask_depth_5` | sum(V_ask, levels 1-5) |
| `raw_bid_depth_10`, `raw_ask_depth_10` | sum(V, levels 1-10) |
| `raw_bid_orders_5`, `raw_ask_orders_5` | Order count at levels 1-5 |

### 2.2 Imbalance (8 features, `imbalance_*`)
Order book asymmetry — Cont, Stoikov & Talreja (2010).

```
imbalance(L) = (V_bid(L) - V_ask(L)) / (V_bid(L) + V_ask(L))    in [-1, 1]
```

| Feature | Level | Interpretation |
|---------|-------|----------------|
| `imbalance_qty_l1` | L1 (touch) | Instantaneous directional pressure |
| `imbalance_qty_l5` | L1-5 | Near-touch aggregate |
| `imbalance_qty_l10` | L1-10 | Deep book asymmetry |
| `imbalance_depth_weighted` | Inverse-distance weighted | Price-level-aware imbalance |
| `imbalance_notional_l5` | L1-5 notional | Dollar-weighted |
| `imbalance_orders_l5` | L1-5 order count | Fragmentation-weighted |
| `imbalance_pressure_bid/ask` | One-sided depth change | Directional depth accumulation |

### 2.3 Flow (12 features, `flow_*`)
Trade flow and aggressor analysis.

| Feature | Formula |
|---------|---------|
| `flow_aggressor_ratio_5s` | buy_volume / total_volume over 5s window |
| `flow_aggressor_ratio_30s` | Same, 30s window |
| `flow_net_volume_5s` | buy_volume - sell_volume (5s) |
| `flow_intensity_5s` | trade_count / 5s |
| `flow_vwap_5s` | sum(price x volume) / sum(volume) |
| `flow_count_*` | Trade counts at various windows |

### 2.4 Volatility (8 features, `vol_*`)
Parkinson (1980) and standard deviation measures.

```
vol_parkinson = sqrt(ln(H/L)^2 / (4 ln 2))          -- Parkinson (1980)
vol_returns_Ns = std(log_returns) over N-second window
vol_ratio_short_long = vol_short / vol_long           -- regime change detector
```

### 2.5 Entropy (24 features, `ent_*`)
Information-theoretic measures — Bandt & Pompe (2002).

```
ent_tick_Ns = -sum(p_i log p_i) / log(n_bins)        -- normalized Shannon entropy of tick returns
ent_permutation_returns_m = -sum(p_pi log p_pi) / log(m!)  -- permutation entropy, embedding dim m
ent_spread_dispersion = H(spread histogram)           -- spread randomness
ent_book_shape = H(depth profile)                     -- order book structure randomness
ent_vol_tick_Ns = H(volume per tick)                  -- volume distribution entropy
```

**`ent_book_shape`** is the most important regime feature discovered in this research:
low values indicate structured (non-random) depth profiles, associated with informed
positioning and higher imbalance signal quality (IC +20-45%).

### 2.6 Context (9 features, `ctx_*`)
Exchange-level metadata: funding rate, open interest, 24h volume, mark price.

### 2.7 Trend (15 features, `trend_*`)
Momentum and mean-reversion indicators — Jegadeesh & Titman (1993).

```
trend_ema_short = EMA(midprice, span=12)
trend_ema_long = EMA(midprice, span=26)
trend_momentum_r2_N = R^2 of OLS(price ~ time) over N ticks
trend_monotonicity_N = |sum(sign(delta_p))| / N       -- fraction of same-direction moves
trend_hurst_N = Hurst exponent via R/S analysis over N ticks
```

### 2.8 Illiquidity (12 features, `illiq_*`)
Market impact and fragility — Kyle (1985), Amihud (2002).

```
illiq_kyle_N = cov(delta_p, signed_volume) / var(signed_volume)   -- Kyle lambda
illiq_amihud_N = mean(|r_t| / volume_t)                           -- Amihud (2002)
illiq_hasbrouck_N = sqrt(var(r_t / sqrt(volume_t)))                -- Hasbrouck (2009)
illiq_composite = weighted(kyle, amihud, hasbrouck, roll)          -- multi-measure blend
```

### 2.9 Toxicity (10 features, `toxic_*`)
Informed trading detection — Easley et al. (2012).

```
toxic_vpin_50 = |V_buy - V_sell| / (V_buy + V_sell)  -- VPIN, 50-bucket
toxic_flow_imbalance = signed flow accumulation
toxic_adverse_selection = E[|r_{t+1}| | trade] - E[|r_{t+1}|]
toxic_effective_spread = 2 * |trade_price - midprice| / midprice * 10000
toxic_realized_spread = 2 * sign(trade) * (trade_price - midprice_{t+5s}) / midprice * 10000
toxic_index = composite toxicity score
```

### 2.10 Derived (15 features, `derived_*`)
Cross-category combinations and regime scores.

```
derived_regime_type_score = f(entropy, volatility, flow) -- market state classifier
derived_regime_confidence = GMM posterior probability of dominant state
derived_informed_trend_score = trend_strength * toxicity_signal
derived_illiquidity_trend = illiq_composite * trend_momentum_r2
derived_toxicity_regime = toxicity_index * entropy_state
```

### 2.11-2.15 Optional Categories (68 features)
- **Whale Flow** (12, `whale_*`): Large trade detection, whale-to-total ratio
- **Liquidation** (13, `liquidation_*`): Cascade risk, liquidation proximity
- **Concentration** (15, `top_*`/`conc_*`): Order concentration, Herfindahl index
- **Regime** (20, `regime_*`): Multi-timeframe regime labels, divergence, churn
- **GMM** (8, `prob_*`): Real-time GMM state probabilities

---

## 3. Analysis Tools

### `nat spannung` — Signal Grid Search

**Method**: Sweep over (alpha, beta, flow_feature, illiq_feature, horizon) combinations.

```
Spannung(t) = EWM_{alpha}(flow(t)) / (|EWM_{beta}(illiq(t))| + eps)
```

**Forward returns**: `r_h(t) = ln(p[t+h] / p[t])`

**Information Coefficient (IC)**: Non-overlapping rolling Spearman rank correlation:
```
IC_w = rho_spearman(signal[w], r_h[w])    for each 3000-tick window w
IC = mean(IC_w)
IC_IR = mean(IC_w) / std(IC_w)            -- information ratio
IC_hit_rate = #{sign(IC_w) = sign(IC)} / W
```

**Grid**: 3 flow x 3 illiq x 6 alpha x 5 beta x 5 horizons = 1,350 combos/symbol.

**Output**: `reports/spannung/spannung_{SYM}.json`

### `nat spannung backtest` — Cost-Aware Backtest

**Position rule**:
```
position(t) = sign(imbalance(t))  if |imbalance(t)| >= threshold,  else 0
```

**P&L per interval**:
```
gross_pnl(t) = position(t) * r_h(t)
cost_taker = |delta_position| * 3.5 bps / 10000  per side
cost_maker = |delta_position| * 1.0 bps / 10000  per side
net_pnl = gross_pnl - cost
```

**Sharpe**: `mean(pnl) / std(pnl) * sqrt(intervals_per_year)`

**Max drawdown**: `max(cummax(cumsum(pnl)) - cumsum(pnl))`

**Regime gating**: Conditional P&L/IC computed within regime subsets defined by
median splits on entropy, VPIN, volatility, illiquidity.

**Output**: `reports/spannung/backtest_{SYM}.json`

### `nat spannung horizon` — Bar-Level Horizon Sweep

**Method**: Aggregate ticks into bars (30s, 1m, 2m, 5m, 10m, 15m). At each bar,
compute 6 signal variants:

| Signal | Formula |
|--------|---------|
| `imbalance_mean` | mean(imbalance_qty_l1) over bar |
| `imbalance_last` | last tick imbalance in bar |
| `imbalance_trend` | OLS slope of imbalance within bar |
| `imbalance_persistence` | #{sign(imb) = sign(bar_mean)} / n_ticks |
| `spannung_mean` | mean(EWM_flow / EWM_illiq) over bar |
| `imbalance_x_illiq` | imbalance_mean x illiq_composite_mean |

Forward returns at 1, 2, 4, 8 bars. IC, gross/net Sharpe, breakeven cost computed
for each of 144 combinations per symbol.

**Output**: `reports/spannung/horizon_sweep_{SYM}.json`

### `nat spannung spectral` — Frequency-Domain Analysis

Six analyses at `fs = 10 Hz`, `NPERSEG = 2048` (~205s segments):

**1. Welch PSD**:
```
P_xx(f) = (2 / (fs * N)) * |X(f)|^2     averaged over Hann-windowed segments
Noise slope: beta = d(log P_xx) / d(log f)   via linear regression
Hurst: H = -(beta + 1) / 2
```

**2. Cross-Spectral Coherence** (Carter 1987):
```
C^2(f) = |P_xy(f)|^2 / (P_xx(f) * P_yy(f))    in [0, 1]
phase(f) = arg(P_xy(f))
phase_lead_ms = phase(f) / (2 pi f) * 1000
```
Computed at horizons {1s, 5s, 30s, 60s}.

**3. Autocorrelation** (Wiener-Khinchin):
```
R(tau) = IFFT(|FFT(x)|^2) / R(0)
```
OU process fit: `R(tau) = exp(-theta * tau)` via nonlinear least squares.
```
half_life = ln(2) / theta
```

**4. Band-Filtered IC**: 4th-order zero-phase Butterworth bandpass (SOS form):
```
H(s) = product_{i=1}^{4} 1 / (s^2 + (omega_i/Q) s + omega_i^2)
```
Applied via `sosfiltfilt` (zero-phase). Frequency bands:
- ultra_low: 0.005-0.1 Hz (periods 10-200s)
- low: 0.05-0.5 Hz (periods 2-20s)
- mid: 0.5-2.0 Hz (periods 0.5-2s)
- high: 2.0-4.5 Hz (periods 0.2-0.5s)

Rolling Spearman IC on each filtered band.

**5. Spectral Entropy**:
```
p_k = P_xx(f_k) / sum(P_xx)
H_norm = -sum(p_k ln p_k) / ln(N)    in [0, 1]
```
0 = perfectly periodic, 1 = white noise.

**Output**: `reports/spannung/spectral_{SYM}.json`

### `nat spannung regime` — Systematic Regime Screening

**Phase 1 — Single-factor screening**: For each of 17 microstructure features, split
at quintile thresholds (P20/P40/P60/P80). For each split:
```
mask = (feature < percentile)  or  (feature > percentile)
IC_conditional = spearmanr(signal[mask], r_h[mask])
dIC = IC_conditional - IC_baseline
```
Both raw and ultra-low bandpass-filtered imbalance tested.

**Phase 2 — Multi-factor combinations**: Top 15 single factors combined in 2-way
(C(15,2) = 105) and 3-way AND combinations. Correlation guard: skip pairs with
|Pearson r| > 0.7 between underlying features.

**Pareto filter**: A combo is Pareto-optimal iff no other combo dominates on BOTH
IC and coverage:
```
dominated(A) = exists B: IC(B) >= IC(A) AND coverage(B) >= coverage(A)
                         with at least one strict inequality
```

**Phase 3 — Persistence**: Run-length analysis of boolean regime mask:
```
episodes = contiguous True runs in mask
mean_duration = mean(episode_lengths) / fs
frac_gt_5s = #{episodes > 50 ticks} / #{episodes}
entry_rate = #{episodes} / total_time_minutes
```

**Output**: `reports/spannung/regime_screen_{SYM}.json`

### `nat profile` — Regime Profiling (GMM Clustering)

**Pipeline**: Tick data -> 15-min bars -> entropy feature vector -> GMM clustering.

**Hopkins Statistic** (clustering tendency):
```
H = sum(u_i) / (sum(u_i) + sum(w_i))
```
where u_i = NN distance from random uniform points, w_i = NN distance between data.
H > 0.7 -> clusters exist.

**GMM via EM**: `GaussianMixture(covariance_type="full", n_init=10)`.
```
p(x | theta) = sum_{k=1}^{K} pi_k * N(x | mu_k, Sigma_k)
```
k chosen by BIC minimization: `BIC = -2 ln L + k_params * ln(n)`.

**Quality Gates**:
- Q1 (structural): silhouette >= 0.25, bootstrap ARI >= 0.6 (50 resamples at 80%)
- Q2 (predictive): Kruskal-Wallis H-test on log-returns across clusters at horizons
  {1, 5, 10, 20}. Effect size: `eta^2 = (H - k + 1) / (n - k)`. Pass if p < 0.05
  and eta^2 > 0.01.
- Q3 (operational): self-transition rate >= 0.8, mean duration >= 3 bars

**Verdict**: GO (Q1+Q2+Q3) / PIVOT (Q1+Q2) / COLLECT (Q1 only) / DROP

**Output**: Printed `ProfilingSnapshot` with cluster metrics.

### `nat profile scalp` — Scalping Feature Profiler

For each of ~550 feature columns at configurable timeframe:

**IC**: `IC_h = spearmanr(x, r_h)` at horizons h in {1, 2, 5, 10} bars.
Best horizon: `h* = argmax_h |IC_h|`.

**IC Information Ratio**: Rolling overlapping windows (step = W/2):
```
IC_IR = E[IC_w] / std(IC_w)
```

**Hit rate**: `#{sign(x[t]) = sign(r_1[t])} / n`

**Quintile spread**: `E[r | Q5] - E[r | Q1]` in bps.

**Cost-adjusted net edge**:
```
gross_edge = |quintile_spread| / 2
turnover_factor = max(0.1, 1 - autocorr_1) if autocorr_1 > 0 else 1.0
net_edge = gross_edge - 2 * cost_bps * turnover_factor
```

**Scalp score** (composite, clipped to [0, 1]):
```
score = 0.25 * |IC*|/0.15 + 0.15 * |IC_IR|/1.0 + 0.15 * (hit-0.5)/0.10
      + 0.15 * |q_spread|/20 + 0.10 * ac_penalty + 0.20 * max(net_edge,0)/5

scalp_score = min(score, 1.0) * role_multiplier
```
Role multipliers: noise=0.20, gate=0.70, directional/regime=1.0.

**Role assignment**: directional (IC > min_ic AND hit > min_hit AND net_edge > 0),
gate (IC > min_ic OR hit > min_hit), regime (special pattern), noise (else).

**Walk-forward validation** (`--forward-test`): 5-fold chronological split,
min_train=200 bars. Per fold: re-profile on IS, measure OOS IC. Decision:
- KEEP: sign_consistency >= 80% AND |OOS_IC| > threshold
- MONITOR: sign_consistency >= 60%
- DROP: else

**Output**: `reports/profiler/profile_{SYM}_{TF}.json`,
`reports/profiler/walk_forward_{SYM}_{TF}.json`

---

## 4. Alpha Pipeline

### `nat alpha combine` — Feature Combination

Combines validated features into a multi-signal alpha. Methods:
- **Linear**: `alpha(t) = sum(w_i * z_i(t))` where z_i = z-scored feature
- **IC-weighted**: `w_i = IC_i / sum(|IC_j|)`
- **Rank-average**: `alpha(t) = mean(rank(feature_i(t)))` across selected features

### `nat alpha size` — Cost-Aware Position Sizing

**Expected gain gate**:
```
E[gain] = |delta_z(t)| * |IC| * vol(r) * sqrt(horizon_bars)
Trade only if E[gain] > round_trip_cost * 1.5
```

**Kelly sizing**: `position = clip(signal * scale, -1, +1)` (fractional Kelly).

**Ramp-up**: Positions scaled by 0.5x for the first 2880 bars (~30 days).

### `nat alpha validate` — Walk-Forward Validation

**Purged walk-forward** (Bailey & Lopez de Prado):
- n_splits folds, embargo gap between train/test to prevent leakage
- Fold valid if: OOS/IS Sharpe >= 0.7 AND OOS Sharpe >= 0.3

**Deflated Sharpe Ratio** (multiple-testing correction):
```
E[max_SR] = sigma * [(1-gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(Ne))]
DSR = (observed_SR - E[max_SR]) / std[max_SR]
p_false = Phi(DSR)
```

### `nat alpha regime` — Regime Conditioning

Applies regime filters (entropy, volatility, toxicity states) to the combined alpha.
IC and Sharpe measured per regime slice. Selects regime conditions that improve
signal quality without excessive coverage loss.

### `nat alpha multi-freq` — Multi-Frequency Integration

Combines signals operating at different timescales (e.g., tick-level imbalance +
minute-level entropy regime). Frequency-domain decomposition ensures signals at
different bands don't interfere.

### `nat alpha portfolio` — Portfolio Assembly

**Risk parity (inverse volatility)**:
```
w_i = (1/sigma_i) / sum(1/sigma_j)
```
sigma computed over 2880-bar lookback.

**Correlation adjustment**: If any pair |corr| > 0.8, scale all weights by 0.8
and renormalize.

**Drawdown control**: Scale factor applied when cumulative drawdown exceeds threshold.

### `nat alpha paper` — Paper Trading Simulation

Shadow execution against live or historical data. Tracks realized IC, fill rate,
slippage, drawdown. Compares realized vs predicted metrics.

### `nat alpha deploy` — Deployment Readiness

Checklist: walk-forward passed, paper Sharpe > threshold, no anomalies, risk limits
configured.

---

## 5. Backtest Engine

### `nat backtest` — Strategy Backtest

**Cost model** (Hyperliquid):
```
one_way_cost = fee_bps + slippage_bps     (default: 5.0 + 2.0 = 7.0 bps)
round_trip = 2 * one_way_cost             (14.0 bps)
```

**Sharpe**: `mean(pnl) / std(pnl) * sqrt(min(n_trades, 252))`

**Max drawdown**: `max(cummax(equity) - equity) / cummax(equity) * 100`

### `nat backtest validate` — Walk-Forward Backtest Validation

Purged walk-forward with embargo. Per-fold: train strategy, test OOS.
Aggregate OOS Sharpe must exceed threshold.

### `nat backtest ml` / `nat backtest ml-validate` / `nat backtest ml-quantile`

ML-based backtesting using LightGBM predictions. Quantile thresholds for
entry/exit. Walk-forward validation with embargo. Experiment tracking integration.

---

## 6. Cluster Analysis

### `nat cluster analyze` — Cluster Quality

Runs full clustering pipeline on a symbol: GMM fit, silhouette, Davies-Bouldin,
Calinski-Harabasz, BIC sweep.

**Silhouette**: `s(i) = (b(i) - a(i)) / max(a(i), b(i))`
where a = mean intra-cluster distance, b = mean nearest-cluster distance.

**Davies-Bouldin**: `DB = (1/K) sum max_{j!=i} (sigma_i + sigma_j) / d(c_i, c_j)`

**Calinski-Harabasz**: `CH = [tr(B_k) / (K-1)] / [tr(W_k) / (n-K)]`
where B_k = between-cluster dispersion, W_k = within-cluster dispersion.

### `nat cluster gmm` — GMM Analysis

Gaussian Mixture Model with full covariance. EM algorithm:
```
E-step: gamma(z_nk) = pi_k N(x_n | mu_k, Sigma_k) / sum_j(pi_j N(x_n | mu_j, Sigma_j))
M-step: mu_k = sum(gamma * x) / sum(gamma)
        Sigma_k = sum(gamma * (x - mu_k)(x - mu_k)^T) / sum(gamma)
        pi_k = sum(gamma) / N
```

### `nat cluster hmm-fit` — HMM via Baum-Welch

Hidden Markov Model with Gaussian emissions. Baum-Welch EM for parameter estimation,
Viterbi algorithm for most likely state sequence.

```
Forward: alpha_t(j) = [sum_i alpha_{t-1}(i) a_{ij}] b_j(o_t)
Backward: beta_t(i) = sum_j a_{ij} b_j(o_{t+1}) beta_{t+1}(j)
Viterbi: delta_t(j) = max_i [delta_{t-1}(i) a_{ij}] b_j(o_t)
```

BIC for model selection: `BIC = -2 ln L + k_params * ln(n)`

### `nat cluster explore` — Exploratory Clustering

HDBSCAN (density-based, automatic k), agglomerative (Ward linkage), k-means sweep.
Diagnostic dendrograms and t-SNE/UMAP projections.

---

## 7. Validation

### `nat validate skeptical` — 20+ Statistical Tests

| Category | Tests |
|----------|-------|
| Distribution | Jarque-Bera: `JB = (n/6)(S^2 + K^2/4)`, Shapiro-Wilk |
| Persistence | ACF(1) via Fisher z-transform, Ljung-Box: `Q = n(n+2) sum(rho_k^2/(n-k))`, ADF |
| Predictive | Kruskal-Wallis on quintile groups, Spearman rank correlation |
| Nonlinear | Mutual information: `MI = sum p(x,y) ln(p(x,y)/(p(x)p(y)))` + permutation null |
| Effect size | Cohen's d = `(mu_1 - mu_2) / sigma_pooled` |
| Redundancy | PCA scree (dims for 90%/95% variance) |
| Regime stability | Bootstrap ARI (50 resamples), cluster centre CV |
| Strategy baselines | Sharpe vs B&H, momentum, mean-reversion, random 95th percentile |
| Return properties | t-test (mean != 0), lag-1 ACF, squared-return ACF (ARCH), excess kurtosis |
| Cross-correlation | Lead-lag at lags -50 to +50 |
| Multiple testing | Bonferroni: `alpha_adj = alpha / m`, Deflated Sharpe Ratio |

**Output**: `reports/skeptical_validation/validation_report.json`

---

## 8. Signal Testing

### `nat signal test` — Signal Existence

LightGBM binary classifier. Target: `sign(midprice[t+h] - midprice[t])`.
Dead-zone filter: +/-0.5 bps (ambiguous returns excluded).

Tests:
1. In-sample accuracy (target > 0.55)
2. Walk-forward accuracy (5 expanding-window splits)
3. Cost-adjusted: `round_trip = 2 * taker_fee + spread = 13 bps`

Edge: `accuracy - base_rate`. Sharpe from signed P&L.

### `nat signal test-all`
Full sweep across all symbols and horizons.

---

## 9. EAMM (Entropy-Aware Market Making)

### `nat eamm run` — Full EAMM Pipeline

**Spread simulation**: For each candidate half-spread delta_k (bps):
```
P_bid = P_mid * (1 - delta_k / 10000)
P_ask = P_mid * (1 + delta_k / 10000)
Fill_bid = (future_min <= P_bid)
Fill_ask = (future_max >= P_ask)
PnL(t, k) = fill * (midprice_future - entry_price)
```

**Optimal spread**: `y(t) = argmax_k PnL(t, delta_k)`.
Continuous version: quadratic fit `PnL = a*delta^2 + b*delta + c`, optimum at
`delta* = -b / (2a)` when a < 0 (concave).

**Context vector** (19 dimensions): 7 entropy features, VPIN-50, toxic index,
adverse selection, 2 volatility, flow intensity, aggressor ratio, 2 imbalance,
momentum-60, Hurst-300, spread bps.

**Model**: LightGBM classifier (spread classes) or regressor (continuous optimal spread).

### `nat eamm regime` — Regime-Conditional Spread

Optimal spread computed per GMM/HMM state. Wider spreads in high-entropy (noisy)
regimes, tighter in low-entropy (informed) regimes.

### `nat eamm backtest` — Market-Making Backtest

Stateful simulation with inventory tracking:
```
inventory(t+1) = inventory(t) + fills_bid(t) - fills_ask(t)
PnL(t) = spread_captured - adverse_selection - inventory_risk
```
Parameters: gamma (inventory aversion), q_max (max inventory).

---

## 10. Scalp Edge Scanner

### `nat scan` — Feature Tail Analysis

For each feature, identify distribution tails and measure forward return edge:

```
tail_mask = (feature > P_high) or (feature < P_low)
tail_edge_bps = |mean(r_h[tail])| * 10000
tail_sharpe = mean(r_h[tail]) / std(r_h[tail])
weighted_sharpe = sharpe * sqrt(n_occurrences)    -- rewards quality + frequency
```

Regime edges: forward returns bucketed by volatility quantiles (Q1-Q4).

Stability: half-split test (edge in first half vs second half).

**Output**: `reports/scalp_scanner/scan_{SYM}_{DATE}.json`

---

## 11. Experiment Management

### `nat experiment workflow`
Full pipeline: snapshot data -> train model -> score -> backtest -> record results.

### `nat experiment snapshot`
Create immutable dataset snapshot with metadata (name, description, hash).

### `nat experiment list/get/compare/best`
Query experiment database. Compare metrics across experiments. Find best by
specified metric (Sharpe, IC, drawdown).

---

## 12. Pipeline State Machine

### `nat pipeline start`

```
IDLE -> BUILDING -> INGESTING -> COLLECTING -> ANALYZING -> DONE
```

State persisted in `data/pipeline_state.json` for resume-on-interrupt.
Configurable via `config/pipeline.toml` (ingestion duration, analysis thresholds).

---

## 13. 15-Minute Experiment

### `nat 15m`
Automated end-to-end pipeline: ingest 15 min -> validate -> profile -> cluster -> report.

### `nat 15m offline`
Same pipeline on existing data (no ingestion). Useful for quick re-analysis.

### `nat 15m viz`
Two-page visualization: page 1 = standard panels, page 2 = advanced (toxicity,
trend, OI, illiquidity).

---

## 14. Visualization

### `nat visualize scan` — Scanner Plots
Gaussian KDE overlays on forward return distributions for tail vs mid-distribution.
Conditional return heatmaps.

### `nat visualize data` — Data Quality
Time series of feature values, NaN rates, distribution histograms.

### `nat visualize profile` / `nat visualize hierarchy`
Cluster visualizations: t-SNE/UMAP projections, dendrogram, state transition
heatmaps, per-cluster feature distributions.

### `nat visualize cluster` — Exploratory Clustering
Interactive cluster exploration with subset selection and method comparison.

### `nat visualize skeptical` — Validation Diagnostics
16+ diagnostic PNGs: ACF plots, QQ-plots, PCA scree, feature correlation matrix,
regime transition charts, return distribution analysis.

---

## 15. Infrastructure

### Build (`nat build`)
`make release` — Rust LTO build, single codegen unit, panic=abort, stripped.
Debug build for faster iteration.

### Test (`nat test`)
Rust unit tests, hypothesis tests (H1-H5), live API validation (4 binaries),
Python test suites for pipeline/dashboard/cluster/backtest/scanner.

### Docker (`nat docker`)
docker-compose: redis (6379), ingestor, api (3000), alerts.

### API (`nat api`)
Axum REST/WebSocket server on port 3000. Real-time feature streaming, model
scoring endpoint, alert service (Telegram).

### Experiment Runner (`nat exp`)
Remote experiment management: start/stop ingestor in tmux, health checks,
daily validation, end-of-experiment analysis, Cloudflare tunnel to dashboard.

---

## Summary Statistics

| Dimension | Count |
|-----------|-------|
| Total CLI commands | ~100 |
| Feature categories | 15 |
| Individual features | 191 |
| Statistical tests (skeptical) | 20+ |
| Regime features screened | 17 |
| Frequency bands analyzed | 4 |
| Walk-forward folds (default) | 5 |
| Symbols supported | BTC, ETH, SOL |
| Sampling rate | 10 Hz (100ms) |
| Emission rate | ~30 rows/sec (3 symbols) |

---

## Key Academic References

| Area | Reference |
|------|-----------|
| Order book imbalance | Cont, Stoikov & Talreja (2010) |
| Microprice | Gatheral & Oomen (2010) |
| Kyle lambda | Kyle (1985) |
| Amihud illiquidity | Amihud (2002) |
| VPIN / toxicity | Easley, Lopez de Prado & O'Hara (2012) |
| Permutation entropy | Bandt & Pompe (2002) |
| Parkinson volatility | Parkinson (1980) |
| Market making | Avellaneda & Stoikov (2008) |
| Momentum | Jegadeesh & Titman (1993) |
| Walk-forward validation | Bailey & Lopez de Prado (2014) |
| Fractional Brownian motion | Mandelbrot & Van Ness (1968) |
| Order book dynamics | Bouchaud et al. (2004) |
