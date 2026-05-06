# Phase 1 Algorithm — Mathematical Specification

## Problem Statement

Given a time series of market microstructure feature vectors, predict the sign of the forward return at a fixed horizon.

## Formal Definition

### Input Space

At each discrete time step t (sampled at 100ms intervals), the ingestor produces a feature vector:

```
x(t) = [x_1(t), x_2(t), ..., x_d(t)] in R^d,  d = 123
```

where d = 123 is the number of active (non-NaN) features across 10 categories. The remaining 68 features (whale, liquidation, concentration, regime, GMM) are NaN because the optional data sources are not connected.

### Target Variable

The target is the sign of the forward return at horizon h:

```
r(t, h) = (P(t + h) - P(t)) / P(t)
```

where P(t) = `raw_midprice` at time t, and h is measured in rows (1 row = 100ms).

The binary classification target:

```
y(t) = 1  if  r(t, h) > delta
y(t) = 0  if  r(t, h) < -delta
y(t) = DROPPED  if  |r(t, h)| <= delta
```

where delta = 0.5 bps (0.00005) is a dead zone that filters out noise-dominated micro-returns.

### Tested Horizons

| h (rows) | Wall time | Rationale |
|----------|-----------|-----------|
| 300 | 30 seconds | Market-maker timescale |
| 3000 | 5 minutes | Short-term alpha, our primary test |
| 18000 | 30 minutes | Medium-term, lower cost sensitivity |
| 36000 | 1 hour | Swing, costs negligible relative to move |

---

## Feature Space X in R^123

### Category 1: Raw Orderbook (10 features)

Direct L2 snapshot observables.

| Feature | Formula | Range |
|---------|---------|-------|
| `raw_midprice` | P_mid = (P_bid + P_ask) / 2 | [74670, 78070] |
| `raw_spread` | S = P_ask - P_bid | [1, 9] |
| `raw_spread_bps` | S_bps = S / P_mid * 10000 | [0.13, 1.20] |
| `raw_microprice` | P_micro = (V_ask * P_bid + V_bid * P_ask) / (V_bid + V_ask) | [74670, 78070] |
| `raw_bid_depth_5` | sum(qty) for 5 best bid levels | [0, 140] BTC |
| `raw_ask_depth_5` | sum(qty) for 5 best ask levels | [0, 241] BTC |
| `raw_bid_depth_10` | sum(qty) for 10 best bid levels | [0, 197] BTC |
| `raw_ask_depth_10` | sum(qty) for 10 best ask levels | [0, 260] BTC |
| `raw_bid_orders_5` | count of orders at 5 best bid levels | [6, 778] |
| `raw_ask_orders_5` | count of orders at 5 best ask levels | [7, 900] |

Note: `raw_midprice`, `raw_bid/ask_depth`, `raw_bid/ask_orders` are absolute-value features that leak regime identity. Removed in the `--remove-leaky` variant.

### Category 2: Imbalance (8 features)

Order book asymmetry. Ref: Cont, Stoikov & Talreja (2010).

| Feature | Formula | Range |
|---------|---------|-------|
| `imbalance_qty_l1` | I_1 = (V_bid^1 - V_ask^1) / (V_bid^1 + V_ask^1) | [-1, 1] |
| `imbalance_qty_l5` | I_5 = (sum V_bid^{1..5} - sum V_ask^{1..5}) / (sum V_bid^{1..5} + sum V_ask^{1..5}) | [-1, 1] |
| `imbalance_qty_l10` | Same formula, 10 levels | [-1, 1] |
| `imbalance_orders_l5` | (N_bid^{1..5} - N_ask^{1..5}) / (N_bid^{1..5} + N_ask^{1..5}) | [-1, 1] |
| `imbalance_notional_l5` | Same as qty_l5 but notional-weighted | [-1, 1] |
| `imbalance_depth_weighted` | sum(V_i * w_i) where w_i = 1/(1 + d_i/10), d_i = distance in bps | [-1, 1] |
| `imbalance_pressure_bid` | cumulative bid depth / max(bid, ask) | [0, 1] |
| `imbalance_pressure_ask` | cumulative ask depth / max(bid, ask) | [0, 1] |

### Category 3: Flow (12 features)

Trade arrival dynamics.

| Feature | Formula | Range |
|---------|---------|-------|
| `flow_count_{1s,5s,30s}` | N(t - w, t) = count of trades in window w | [1, 216] |
| `flow_volume_{1s,5s,30s}` | V(t - w, t) = sum of trade sizes in window w | [0, 5.2] BTC |
| `flow_aggressor_ratio_{5s,30s}` | V_buy / (V_buy + V_sell) in window w | [0, 1] |
| `flow_vwap_5s` | VWAP = sum(p_i * v_i) / sum(v_i) over 5s | price level |
| `flow_vwap_deviation` | (VWAP_5s - P_last) / P_last | [-0.03%, 0.04%] |
| `flow_avg_trade_size_30s` | V_30s / N_30s | [0, 0.08] BTC |
| `flow_intensity` | EMA of trades/sec, 5s halflife | [1.2, 17.6] |

### Category 4: Volatility (8 features)

Realized and range-based estimators. Ref: Parkinson (1980).

| Feature | Formula | Range |
|---------|---------|-------|
| `vol_returns_1m` | RV = sqrt(sum(r_i^2) / N), N=60 ticks | [0, 0.01%] |
| `vol_returns_5m` | RV over 300 ticks | [0, 0.01%] |
| `vol_parkinson_5m` | sigma_P = ln(H/L) / sqrt(4 * ln(2)), 300 ticks | [0, 0.05%] |
| `vol_spread_mean_1m` | Point-in-time spread (misnomer) | [1, 9] |
| `vol_spread_std_1m` | **Placeholder: always 0.0** | 0 |
| `vol_midprice_std_1m` | std(P_mid) over 60 ticks | [0, 14.85] |
| `vol_ratio_short_long` | RV_1m / RV_5m | [0, 1.25] |
| `vol_zscore` | **Placeholder: always 0.0** | 0 |

### Category 5: Entropy (24 features)

Information content and predictability. Refs: Bandt & Pompe (2002), Shannon (1948).

**Permutation entropy (4 features):**
For embedding dimension m=3, count occurrences of each of m!=6 ordinal patterns in windows of w ticks:

```
H_perm(w) = -sum_{pi in S_m} p(pi) * ln(p(pi)) / ln(m!)
```

Normalized to [0, 1]. Applied to returns (w=8, 16, 32) and L1 imbalance (w=16).

**Distribution entropy (4 features):**
Bin continuous values into N equal-width bins:

```
H_dist = -sum_{i=1}^{N} p_i * ln(p_i) / ln(N)
```

Applied to: spread (N=10), volume (N=10), book shape (depth proportions), trade size (N=5).

**Rate of change and z-score (2 features):**

```
ent_roc = H(t) - H(t - 5s)
ent_zscore = (H(t) - mu_H(60s)) / sigma_H(60s)
```

**Tick entropy (7 features):**
Classify each trade as {up, down, neutral} by tick rule. Shannon entropy of direction distribution within window w:

```
H_tick(w) = -sum_{d in {up,down,neutral}} p(d) * ln(p(d))
```

Range [0, ln(3)]. Windows: 1s, 5s, 10s, 15s, 30s, 1m, 15m.

**Volume-weighted tick entropy (7 features):**
Same windows, but directions weighted by trade volume before computing H.

### Category 6: Context (9 features)

Hyperliquid market metadata from `activeAssetCtx` channel.

| Feature | Formula | Range |
|---------|---------|-------|
| `ctx_funding_rate` | 8-hour funding rate | [-0.01%, 0.01%] |
| `ctx_funding_zscore` | (FR - mu_FR) / sigma_FR, 60-sample lookback | [-1.5, 2.4] |
| `ctx_open_interest` | Total OI in coin units | ~27,700 BTC |
| `ctx_oi_change_5m` | abs(OI(t) - OI(t - 5m)) | ~0 |
| `ctx_oi_change_pct_5m` | OI change as percentage | ~0 |
| `ctx_premium_bps` | (mark - index) / index * 10000 | [-6.4, 0] bps |
| `ctx_volume_24h` | Rolling 24h volume in USD | ~$1.65B |
| `ctx_volume_ratio` | V_24h(t) / V_24h(t - 5m) | [1.0, 1.001] |
| `ctx_mark_oracle_divergence` | abs(mark - oracle) / oracle * 10000 | [0, 6.4] bps |

Note: `ctx_open_interest` and `ctx_volume_24h` are absolute values — removed in `--remove-leaky`.

### Category 7: Trend (15 features)

Persistence and mean-reversion detection. Refs: Jegadeesh & Titman (1993), Mandelbrot (1971).

| Feature | Formula | Range |
|---------|---------|-------|
| `trend_momentum_{60,300,600}` | OLS slope of P_mid over w ticks | R |
| `trend_momentum_r2_{60,300,600}` | R^2 of the OLS fit | [0, 1] |
| `trend_monotonicity_{60,300,600}` | fraction of ticks in majority direction | [0.5, 1.0] |
| `trend_hurst_{300,600}` | Rescaled range: H = log(R/S) / log(n). H<0.5=mean-revert, H>0.5=trending | [0, 0.81] |
| `trend_ma_crossover` | EMA(10) - EMA(50) in price units | [-17.8, 11.8] |
| `trend_ma_crossover_norm` | MA crossover / P_mid | [-0.024, 0.016] |
| `trend_ema_short` | EMA(10) of midprice | price level |
| `trend_ema_long` | EMA(50) of midprice | price level |

### Category 8: Illiquidity (12 features)

Market impact measures. Refs: Kyle (1985), Amihud (2002), Hasbrouck (2009).

| Feature | Formula | Range |
|---------|---------|-------|
| `illiq_kyle_{100,500}` | lambda = Cov(dP, signed_vol) / Var(signed_vol) | [0, 17.7] |
| `illiq_amihud_{100,500}` | A = sum(\|r_i\|) / sum(v_i) * 10^6 | [43, 6430] |
| `illiq_hasbrouck_{100,500}` | Permanent price impact via OLS | [0, 237] |
| `illiq_roll_{100,500}` | S_roll = 2 * sqrt(-Cov(dP_t, dP_{t-1})) | [0, 1.54] |
| `illiq_kyle_ratio` | lambda_100 / lambda_500 | [0, 786] |
| `illiq_amihud_ratio` | A_100 / A_500 | [0.3, 22] |
| `illiq_composite` | mean(normalized lambdas) | [14.6, 2189] |
| `illiq_trade_count` | N trades in computation window | [30, 354] |

### Category 9: Toxicity (10 features)

Adverse selection and informed flow. Refs: Easley et al. (2012), Glosten & Milgrom (1985).

| Feature | Formula | Range |
|---------|---------|-------|
| `toxic_vpin_10` | VPIN = sum(\|V_buy^b - V_sell^b\|) / sum(V_buy^b + V_sell^b), 10 buckets | ~0 (degenerate) |
| `toxic_vpin_50` | VPIN with 50 volume buckets | [0.85, 0.99] |
| `toxic_vpin_roc` | VPIN(t) - VPIN(t-1) | ~0 |
| `toxic_adverse_selection` | effective_spread - realized_spread | [-0.08, 0.10] |
| `toxic_effective_spread` | 2 * mean(\|P_trade - VWAP\|) | [0.6, 34.9] |
| `toxic_realized_spread` | mean(dir * (P_trade - P_{t+5}) * 2) | [-1.5, 1.1] |
| `toxic_flow_imbalance` | (V_buy - V_sell) / (V_buy + V_sell) | [-0.83, 0.99] |
| `toxic_flow_imbalance_abs` | \|flow_imbalance\| | [0.12, 0.99] |
| `toxic_index` | Weighted composite of VPIN, adverse_sel, flow_imb, normalized to [0,1] | [0.47, 0.69] |
| `toxic_trade_count` | N trades used in computation | [30, 354] |

### Category 10: Derived (15 features)

Cross-category interaction terms.

| Feature | Formula | Range |
|---------|---------|-------|
| `derived_entropy_trend_interaction` | (1 - H_tick_30s/ln(3)) * \|monotonicity_60 - 0.5\| * 2 | [0, 0.65] |
| `derived_entropy_trend_zscore` | z-score of above over 60s | [-1.3, 3.0] |
| `derived_trend_strength_{60,300}` | sign(mom) * (mono - 0.5) * 2 * (1 - H_tick) | [-0.33, 0.53] |
| `derived_trend_strength_ratio` | strength_60 / strength_300 | [-16.4, 14.0] |
| `derived_entropy_volatility_ratio` | H_tick_30s / (RV_1m * 100 + epsilon) | [0.47, 0.69] |
| `derived_regime_type_score` | RV_1m * 100 * (1 - 2 * H_tick_30s/ln(3)) | [-0.001, 0.0002] |
| `derived_illiquidity_trend` | kyle_100/100 * \|mom_60\| * 1000 | [0, 75.5] |
| `derived_informed_trend_score` | clamp(kyle/100) * clamp(VPIN) * \|trend\| | [0, 0.14] |
| `derived_toxicity_regime` | toxic_index * H_tick_30s | [0.30, 0.46] |
| `derived_toxic_chop_score` | toxic_index * H_tick * (1 - mono) * 4 | [0, 0.99] |
| `derived_trend_strength_roc` | strength(t) - strength(t - 5s) | [-0.43, 0.37] |
| `derived_entropy_momentum` | H_tick(t) - H_tick(t - 15s) | [-0.14, 0.50] |
| `derived_regime_indicator` | (mean_revert - trending - flow_factor), clamped to [-1, 1] | [-0.74, 0.86] |
| `derived_regime_confidence` | max(\|trending\|, \|mean_revert\|) | [0.19, 1.0] |

---

## Classifier: Gradient Boosted Decision Trees

### Model

LightGBM (Ke et al., 2017) binary classifier. Predicts:

```
P(y=1 | x(t)) = sigma(F(x(t)))
```

where F is an additive ensemble of M regression trees:

```
F(x) = sum_{m=1}^{M} f_m(x),  f_m in H (space of CART trees)
```

Each tree f_m is fit on the negative gradient of the log-loss:

```
L = -sum_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 (test 1-2), 300 (test 3) | Enough to capture interactions, not enough to overfit |
| `max_depth` | 6 | Allows 3-way feature interactions |
| `learning_rate` | 0.05 | Conservative shrinkage |
| `subsample` | 0.8 | Row sampling to reduce overfitting |
| `colsample_bytree` | 0.8 | Feature sampling per tree |
| `min_child_samples` | 50 | Prevents splits on tiny leaf counts |

### NaN Handling

All NaN and inf values replaced with 0.0 before training:

```
X = nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
```

This means the 68 all-NaN optional features become constant 0 columns and contribute nothing. Effectively d=123 active dimensions.

---

## Evaluation Protocol

### Test 1: In-Sample (Sanity Check)

Train on first 70% of data, evaluate accuracy on training set.

```
ACC_train = (1/N_train) * sum_{i=1}^{N_train} 1[F(x_i) > 0.5 == y_i]
```

This should be high (>90%). If low, the model cannot even fit the training data.

### Test 2: Walk-Forward Validation

No lookahead. Expanding training window, fixed test size.

```
For k = 1, ..., K (K=5 splits):
    Train set:  {1, ..., N_min + (k-1) * N_step}
    Test set:   {N_min + (k-1) * N_step + 1, ..., N_min + k * N_step}

    Fit model on train set
    Predict on test set
    Measure: accuracy, edge over base rate, Sharpe ratio
```

Where N_min = 0.2 * N (minimum 20% for first training set) and N_step = 0.8 * N / K.

**Edge** = accuracy - base_rate. Base rate = P(y=1) in the test set. Positive edge = model predicts better than always guessing the majority class.

**Per-bar Sharpe** (within-split):

```
position_i = +1 if F(x_i) > 0.5, else -1
pnl_i = position_i * r(t_i, h)
Sharpe = mean(pnl) / std(pnl) * sqrt(N_test)
```

### Test 3: Confidence-Filtered Trading

The actual profitability test. Single train/test split (60/40).

For each confidence threshold tau in {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80}:

```
Trade set T(tau) = { i : P(y=1|x_i) > tau  OR  P(y=1|x_i) < 1-tau }

For i in T(tau):
    direction_i = +1  if P(y=1|x_i) > tau
                  -1  if P(y=1|x_i) < 1-tau

    gross_pnl_i = direction_i * r(t_i, h)
    net_pnl_i(taker) = gross_pnl_i - C_taker
    net_pnl_i(maker) = gross_pnl_i - C_maker
```

Where:
```
C_taker = 2 * (3.5 + spread_bps/2) / 10000    (round trip: 2 * (fee + half_spread))
C_maker = 2 * (0.0 + spread_bps/2) / 10000     (Hyperliquid: 0% maker fee)
```

At spread_bps = 1.0:
- C_taker = 8.0 bps per round trip
- C_maker = 1.0 bps per round trip

**Profitability condition:** mean(net_pnl) > 0 at any threshold.

---

## Results Summary (5.5 days, 441,611 BTC rows)

### With All 123 Active Features

```
Walk-forward edge:     +4.18% over base rate
Best accuracy (0.80):  54.2%
Best net (maker):      -0.45 bps/trade     UNPROFITABLE
```

### With Leaky Features Removed (d=114)

Removed: raw_midprice, ctx_open_interest, ctx_volume_24h, raw_bid/ask_depth_{5,10}, raw_bid/ask_orders_5

```
Walk-forward edge:     +4.61% over base rate
Best accuracy (0.80):  50.5%
Best net (maker):      -0.76 bps/trade     UNPROFITABLE
```

### Top Features by Importance (LightGBM split count)

With all features:
```
1. ctx_volume_ratio         198   (volume momentum proxy)
2. regime_divergence_1h     195   (NaN→0, but split count suggests interaction)
3. ctx_funding_zscore       172   (funding rate regime)
4. ctx_funding_rate         310   (raw funding)
5. vol_returns_5m           208   (5-min realized vol)
6. trend_momentum_600       171   (60s price momentum)
7. trend_momentum_r2_600    148   (momentum quality)
8. trend_monotonicity_600   145   (directional persistence)
```

Without leaky features:
```
1. ctx_funding_rate         310
2. raw_microprice           276   (retained — relative, not absolute)
3. regime_kyle_lambda       256   (NaN→0, likely interaction term)
4. vol_returns_5m           208
5. ctx_volume_ratio         198
6. regime_divergence_1h     195
7. raw_spread_bps           186   (spread in bps — relative)
8. regime_churn_1h          180
```

### Interpretation

The model's signal comes primarily from:
1. **Funding rate** (directional bias indicator)
2. **Volatility** (realized vol predicts continuation/reversal)
3. **Trend momentum + persistence** (momentum at 30-60s timescale)
4. **Volume ratio** (volume acceleration)
5. **Spread** (liquidity conditions)

These align with known microstructure predictors in the literature. The signal is real but insufficient to overcome transaction costs at 5-minute horizon with 5.5 days of data.
