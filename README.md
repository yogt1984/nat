```

    ███╗   ██╗ █████╗ ████████╗
    ████╗  ██║██╔══██╗╚══██╔══╝
    ██╔██╗ ██║███████║   ██║
    ██║╚██╗██║██╔══██║   ██║
    ██║ ╚████║██║  ██║   ██║
    ╚═╝  ╚═══╝╚═╝  ╚═╝   ╚═╝

    N E X T - G E N   A L P H A   T E C H N O L O G Y
    ─────────────────────────────────────────────────
    Hyperliquid Analytics & Signal Intelligence Layer

```

# NAT — Quantitative Research Infrastructure for Hyperliquid

**NAT** is a production-grade quantitative research platform designed for extracting alpha signals from Hyperliquid's perpetual futures market. Built in Rust for maximum performance, NAT provides real-time feature extraction, rigorous hypothesis testing, and statistically-validated trading signals.

[![Tests](https://img.shields.io/badge/tests-287%20passing-brightgreen)]()
[![Rust](https://img.shields.io/badge/rust-1.75+-orange)]()
[![License](https://img.shields.io/badge/license-proprietary-blue)]()

---

## Why NAT?

Most crypto analytics tools are **toys**. NAT is **infrastructure**.

| The Problem | NAT's Solution |
|-------------|----------------|
| Delayed data feeds | Sub-millisecond WebSocket ingestion |
| Basic indicators | 163 institutional-grade features |
| No validation | Rigorous hypothesis testing with walk-forward validation |
| Overfitting | Bonferroni correction, OOS/IS ratio checks, MI thresholds |
| Black box signals | Full statistical transparency with confidence intervals |

---

## Core Capabilities

### Real-Time Feature Extraction Engine

NAT processes Hyperliquid's order book and trade stream in real-time, computing **183 features** across 14 categories:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE EXTRACTION PIPELINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WebSocket Stream ──► Order Book State ──► Feature Computation      │
│        │                    │                      │                │
│        ▼                    ▼                      ▼                │
│   ┌─────────┐        ┌───────────┐          ┌──────────┐           │
│   │ Trades  │        │ L2 Book   │          │ Parquet  │           │
│   │ Ticks   │        │ Snapshots │          │ Output   │           │
│   └─────────┘        └───────────┘          └──────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Feature Categories:**

| Category | Count | Key Features | Signal Type |
|----------|-------|--------------|-------------|
| **Entropy** | 24 | Tick entropy, permutation entropy, conditional entropy | Regime detection, predictability |
| **Trend** | 15 | Momentum, monotonicity, Hurst exponent, R² | Persistence, mean-reversion |
| **Illiquidity** | 12 | Kyle's λ, Amihud, Hasbrouck, Roll spread | Price impact, informed flow |
| **Toxicity** | 10 | VPIN, adverse selection, effective spread | Order flow toxicity |
| **Order Flow** | 8 | Imbalance (L1/L5/L10), pressure, depth-weighted | Directional conviction |
| **Volatility** | 8 | Realized vol, Parkinson, Garman-Klass | Risk regime |
| **Concentration** | 15 | Gini, HHI, Top-10/20, Theil, whale ratios | Position crowding |
| **Whale Flow** | 12 | Net flow (1h/4h/24h), momentum, intensity | Smart money tracking |
| **Liquidation** | 13 | Risk mapping, cluster detection, cascade probability | Cascade prediction |
| **Raw Data** | 10 | Midprice, microprice, spread, depth | Microstructure |
| **Trade Flow** | 12 | Volume, VWAP, aggressor ratio, intensity | Execution patterns |
| **Context** | 9 | Funding rate, OI, premium, basis | Market conditions |
| **Derived** | 15 | Regime indicators, composite signals, interactions | Combined alpha |
| **Regime** | 20 | Absorption, divergence, churn, range position | Accumulation/distribution detection |

### Whale Intelligence System

Track and classify large players in real-time:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      WHALE CLASSIFICATION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Tier 1: MEGA WHALE     │  $10M+ positions    │  Market movers    │
│   Tier 2: WHALE          │  $1M-$10M           │  Significant      │
│   Tier 3: LARGE TRADER   │  $100K-$1M          │  Notable          │
│   Tier 4: RETAIL         │  <$100K             │  Noise            │
│                                                                     │
│   Metrics: Net flow, position changes, entry/exit timing           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Liquidation Cascade Detection

Identify clustered liquidation risk before cascades occur:

- Real-time liquidation price mapping
- Cluster detection with configurable thresholds
- Lead-time analysis for actionable signals
- Directional prediction (long vs short squeeze)

---

## Hypothesis Testing Framework

NAT doesn't guess. NAT **validates**.

Every signal passes through a rigorous statistical gauntlet before deployment:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HYPOTHESIS TESTING PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │    H1    │    │    H2    │    │    H3    │    │    H4    │     │
│  │  Whale   │    │ Entropy+ │    │ Liquid.  │    │ Concen.  │     │
│  │  Flow    │    │  Whale   │    │ Cascade  │    │   Vol    │     │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘     │
│       │               │               │               │            │
│       └───────────────┴───────────────┴───────────────┘            │
│                           │                                        │
│                           ▼                                        │
│                    ┌──────────┐                                    │
│                    │    H5    │                                    │
│                    │ Persist. │                                    │
│                    │Indicator │                                    │
│                    └────┬─────┘                                    │
│                         │                                          │
│                         ▼                                          │
│              ┌─────────────────────┐                               │
│              │   FINAL DECISION    │                               │
│              │  GO / PIVOT / NOGO  │                               │
│              └─────────────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The Five Hypotheses

| ID | Hypothesis | Success Criteria | Statistical Tests |
|----|------------|------------------|-------------------|
| **H1** | Whale flow predicts returns | r > 0.05, p < 0.001, MI > 0.02 bits | Pearson, Spearman, Walk-forward |
| **H2** | Entropy + whale interaction | Lift > 10%, p < 0.01 | Chi-squared, Contingency tables |
| **H3** | Liquidation cascades predictable | Precision > 30%, Lift > 2x | Classification metrics, Lead-time |
| **H4** | Concentration predicts volatility | r > 0.2, partial r > 0.1 | Partial correlation, Causality |
| **H5** | Persistence indicator works | WF Sharpe > 0.5, OOS/IS > 0.7 | Walk-forward, Regime analysis |

### Decision Framework

```
╔══════════════════════════════════════════════════════════════════╗
║                      DECISION MATRIX                              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║   0-1 hypotheses pass  ──►  NO-GO   (Insufficient alpha)         ║
║   2-3 hypotheses pass  ──►  PIVOT   (Focus on validated only)    ║
║   4-5 hypotheses pass  ──►  GO      (Full deployment)            ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Statistical Rigor

NAT implements institutional-grade statistical validation:

- **Bonferroni Correction** — Controls family-wise error rate across multiple tests
- **Walk-Forward Validation** — 5-fold expanding window, no look-ahead bias
- **Out-of-Sample Ratio** — OOS/IS > 0.7 required (detects overfitting)
- **Mutual Information** — Non-linear dependency detection beyond correlation
- **Confidence Intervals** — 95% CI on all correlation estimates
- **Regime Analysis** — Separate validation in low-vol vs high-vol environments

---

## Feature Redundancy Analysis

Not all features are created equal. NAT's feature analysis module:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ANALYSIS OUTPUT                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ✓ Correlation matrix (Pearson + Spearman)                         │
│  ✓ Mutual Information matrix                                        │
│  ✓ Hierarchical clustering with dendrogram                         │
│  ✓ Redundancy detection (|r| > 0.9)                                │
│  ✓ Feature ranking by predictive power                             │
│  ✓ Recommended subset (10-15 non-redundant features)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Complete Feature Reference

### 183 Features with Mathematical Definitions

<details>
<summary><b>1. Entropy Features (24)</b> — Regime detection & predictability</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `permutation_entropy` | Ordinal pattern entropy | H = -Σ p(π) log₂ p(π), where π are ordinal patterns |
| `tick_entropy_1s` | Tick direction entropy (1s) | H = -Σ p(d) log₂ p(d), d ∈ {up, down, unchanged} |
| `tick_entropy_5s` | Tick direction entropy (5s) | Same formula, 5s window |
| `tick_entropy_10s` | Tick direction entropy (10s) | Same formula, 10s window |
| `tick_entropy_15s` | Tick direction entropy (15s) | Same formula, 15s window |
| `tick_entropy_30s` | Tick direction entropy (30s) | Same formula, 30s window |
| `tick_entropy_1m` | Tick direction entropy (1m) | Same formula, 60s window |
| `tick_entropy_15m` | Tick direction entropy (15m) | Same formula, 900s window |
| `volume_weighted_tick_entropy_*` | Volume-weighted variant | H = -Σ pᵥ(d) log₂ pᵥ(d), weighted by trade volume |
| `normalized_entropy_*` | Normalized to [0,1] | H_norm = H / log₂(n_categories) |
| `entropy_rate` | Entropy change rate | dH/dt over sliding window |
| `conditional_entropy` | Conditional on previous tick | H(X\|Y) = -Σ p(x,y) log₂ p(x\|y) |

</details>

<details>
<summary><b>2. Trend Features (15)</b> — Momentum & mean-reversion</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `momentum_60` | Price momentum (60 ticks) | m = (Pₜ - Pₜ₋₆₀) / Pₜ₋₆₀ |
| `momentum_300` | Price momentum (300 ticks) | m = (Pₜ - Pₜ₋₃₀₀) / Pₜ₋₃₀₀ |
| `momentum_600` | Price momentum (600 ticks) | m = (Pₜ - Pₜ₋₆₀₀) / Pₜ₋₆₀₀ |
| `r_squared_60` | R² linear regression (60) | R² = 1 - SSᵣₑₛ / SSₜₒₜ |
| `r_squared_300` | R² linear regression (300) | Same, 300-tick window |
| `monotonicity_60` | Monotonic move proportion | M = \|Σ sign(ΔP)\| / n |
| `monotonicity_300` | Monotonicity (300 ticks) | Same formula, 300-tick window |
| `monotonicity_600` | Monotonicity (600 ticks) | Same formula, 600-tick window |
| `hurst_exponent` | Hurst exponent (persistence) | H from R/S analysis: E[R(n)/S(n)] = Cnᴴ |
| `ma_crossover_fast_slow` | MA crossover signal | sign(MA_fast - MA_slow) |
| `trend_strength` | Combined trend indicator | √(momentum² × R²) |
| `price_acceleration` | Second derivative of price | a = Δ²P / Δt² |
| `trend_consistency` | Direction consistency | Ratio of same-direction moves |
| `breakout_indicator` | Distance from range | (P - min) / (max - min) |
| `mean_reversion_score` | Z-score from MA | z = (P - MA) / σ |

</details>

<details>
<summary><b>3. Illiquidity Features (12)</b> — Price impact & market quality</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `kyle_lambda_100` | Kyle's λ (100 trades) | λ = Cov(ΔP, V_signed) / Var(V_signed) |
| `kyle_lambda_500` | Kyle's λ (500 trades) | Same, larger window |
| `amihud_lambda_100` | Amihud illiquidity | λ = E[\|r\| / V] × 10⁶ |
| `amihud_lambda_500` | Amihud (500 trades) | Same, larger window |
| `hasbrouck_lambda_100` | Hasbrouck's λ | λ = √(Var(ΔP) / Var(V)) |
| `hasbrouck_lambda_500` | Hasbrouck (500 trades) | Same, larger window |
| `roll_spread_100` | Roll's implied spread | S = 2√(-Cov(ΔPₜ, ΔPₜ₋₁)) |
| `roll_spread_500` | Roll spread (500) | Same, larger window |
| `depth_impact` | Price impact per depth | ΔP / ΔDepth |
| `effective_spread` | Effective bid-ask spread | 2 × \|P_trade - midprice\| |
| `realized_spread` | Realized spread (5s fwd) | 2 × sign × (P_trade - Pₜ₊₅ₛ) |
| `price_impact_asymmetry` | Buy vs sell impact | λ_buy / λ_sell |

</details>

<details>
<summary><b>4. Toxicity Features (10)</b> — Adverse selection & informed flow</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `vpin_10` | VPIN (10 buckets) | VPIN = Σ\|V_buy - V_sell\| / Σ(V_buy + V_sell) |
| `vpin_50` | VPIN (50 buckets) | Same, 50 volume buckets |
| `adverse_selection` | Adverse selection | AS = E[sign × (Pₜ₊Δ - Pₜ)] |
| `effective_spread_pct` | Effective spread % | 2 × \|P - mid\| / mid × 100 |
| `realized_spread_pct` | Realized spread % | 2 × sign × (P - Pₜ₊₅ₛ) / mid × 100 |
| `flow_imbalance` | Order flow imbalance | (V_buy - V_sell) / (V_buy + V_sell) |
| `toxicity_ratio` | Toxicity ratio | AS / Effective_spread |
| `informed_trade_prob` | Informed trade probability | From PIN model estimation |
| `quote_stuffing_indicator` | Quote stuffing detection | Quote updates / second |
| `spoofing_score` | Cancelled order ratio | Cancelled_volume / Posted_volume |

</details>

<details>
<summary><b>5. Order Flow Imbalance Features (8)</b> — Directional pressure</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `imbalance_qty_l1` | L1 quantity imbalance | I = (Q_bid - Q_ask) / (Q_bid + Q_ask) |
| `imbalance_qty_l5` | L5 cumulative imbalance | Same, summed over 5 levels |
| `imbalance_qty_l10` | L10 cumulative imbalance | Same, summed over 10 levels |
| `imbalance_orders` | Order count imbalance | (N_bid - N_ask) / (N_bid + N_ask) |
| `imbalance_notional` | Notional value imbalance | (V_bid - V_ask) / (V_bid + V_ask) |
| `depth_weighted_imbalance` | Distance-weighted | Σ wᵢ × Iᵢ, wᵢ = 1/distanceᵢ |
| `pressure_bid` | Bid side pressure | Σ Q_bid × (1 / distance) |
| `pressure_ask` | Ask side pressure | Σ Q_ask × (1 / distance) |

</details>

<details>
<summary><b>6. Volatility Features (8)</b> — Risk measurement</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `realized_vol_1m` | Realized volatility (1m) | σ = √(Σ r² × 252 × 24 × 60) |
| `realized_vol_5m` | Realized volatility (5m) | Same, 5-minute window |
| `parkinson_vol` | Parkinson high-low vol | σ = √(1/(4ln2) × (ln(H/L))²) |
| `spread_volatility` | Spread volatility | σ(spread) over window |
| `spread_mean` | Mean bid-ask spread | E[ask - bid] |
| `spread_max` | Maximum spread | max(ask - bid) |
| `vol_ratio` | Volatility ratio | σ₁ₘ / σ₅ₘ |
| `garman_klass_vol` | Garman-Klass volatility | σ² = 0.5(ln H/L)² - (2ln2-1)(ln C/O)² |

</details>

<details>
<summary><b>7. Position Concentration Features (15)</b> — Crowding & inequality</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `top5_concentration` | Top 5 traders' share | Σᵢ₌₁⁵ \|posᵢ\| / Σ\|pos\| |
| `top10_concentration` | Top 10 traders' share | Σᵢ₌₁¹⁰ \|posᵢ\| / Σ\|pos\| |
| `top20_concentration` | Top 20 traders' share | Σᵢ₌₁²⁰ \|posᵢ\| / Σ\|pos\| |
| `top50_concentration` | Top 50 traders' share | Σᵢ₌₁⁵⁰ \|posᵢ\| / Σ\|pos\| |
| `hhi` | Herfindahl-Hirschman Index | HHI = Σ sᵢ², where sᵢ = share |
| `gini_coefficient` | Gini inequality | G = (2Σ i×xᵢ)/(nΣxᵢ) - (n+1)/n |
| `theil_index` | Theil entropy index | T = (1/n) Σ (xᵢ/μ) ln(xᵢ/μ) |
| `whale_long_ratio` | Whale share of longs | Whale_long / Total_long |
| `whale_short_ratio` | Whale share of shorts | Whale_short / Total_short |
| `concentration_change_1h` | Concentration Δ (1h) | HHIₜ - HHIₜ₋₁ₕ |
| `concentration_change_4h` | Concentration Δ (4h) | HHIₜ - HHIₜ₋₄ₕ |
| `position_crowding` | Directional crowding | \|Net_position\| / Total_OI |
| `long_short_ratio` | Long vs short ratio | Total_long / Total_short |
| `whale_dominance` | Whale share of OI | Whale_OI / Total_OI |
| `retail_participation` | Retail share | Retail_OI / Total_OI |

</details>

<details>
<summary><b>8. Whale Flow Features (12)</b> — Smart money tracking</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `whale_net_flow_1h` | Net whale flow (1h) | Σ (whale_buys - whale_sells) |
| `whale_net_flow_4h` | Net whale flow (4h) | Same, 4h window |
| `whale_net_flow_24h` | Net whale flow (24h) | Same, 24h window |
| `whale_flow_normalized_1h` | Normalized flow (1h) | flow₁ₕ / σ(flow₁ₕ) |
| `whale_flow_normalized_4h` | Normalized flow (4h) | flow₄ₕ / σ(flow₄ₕ) |
| `whale_flow_normalized_24h` | Normalized flow (24h) | flow₂₄ₕ / σ(flow₂₄ₕ) |
| `whale_flow_momentum` | Flow momentum | (flow₁ₕ - flow₄ₕ) / flow₄ₕ |
| `whale_intensity` | Trade intensity | whale_trades / total_trades |
| `whale_directional_conviction` | Conviction score | \|flow\| / volume × sign(flow) |
| `mega_whale_activity` | Tier 1 activity | mega_whale_volume / total |
| `whale_accumulation` | Accumulation indicator | Δ whale_positions > 0 |
| `whale_distribution` | Distribution indicator | Δ whale_positions < 0 |

</details>

<details>
<summary><b>9. Liquidation Risk Features (13)</b> — Cascade detection</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `liq_risk_above_1pct` | Liq risk above +1% | Σ liq_notional where P_liq ∈ [P, P×1.01] |
| `liq_risk_above_2pct` | Liq risk above +2% | Same, [P, P×1.02] |
| `liq_risk_above_5pct` | Liq risk above +5% | Same, [P, P×1.05] |
| `liq_risk_above_10pct` | Liq risk above +10% | Same, [P, P×1.10] |
| `liq_risk_below_1pct` | Liq risk below -1% | Σ liq_notional where P_liq ∈ [P×0.99, P] |
| `liq_risk_below_2pct` | Liq risk below -2% | Same, [P×0.98, P] |
| `liq_risk_below_5pct` | Liq risk below -5% | Same, [P×0.95, P] |
| `liq_risk_below_10pct` | Liq risk below -10% | Same, [P×0.90, P] |
| `liq_asymmetry` | Long vs short imbalance | (risk_above - risk_below) / total |
| `liq_intensity` | Liquidation density | total_liq_risk / OI |
| `nearest_cluster_distance` | Distance to cluster | min(\|P - P_cluster\|) / P |
| `cluster_size` | Largest cluster size | max(cluster_notional) |
| `cascade_probability` | Cascade risk estimate | f(cluster_density, leverage) |

</details>

<details>
<summary><b>10. Raw Market Data Features (10)</b> — Microstructure</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `midprice` | Mid price | (best_bid + best_ask) / 2 |
| `spread_bps` | Spread (basis points) | (ask - bid) / mid × 10000 |
| `microprice` | Size-weighted mid | (bid×Q_ask + ask×Q_bid) / (Q_bid + Q_ask) |
| `depth_bid_l1` | Bid depth at L1 | Q_bid at best bid |
| `depth_ask_l1` | Ask depth at L1 | Q_ask at best ask |
| `depth_bid_l5` | Cumulative bid (L5) | Σ Q_bid for levels 1-5 |
| `depth_ask_l5` | Cumulative ask (L5) | Σ Q_ask for levels 1-5 |
| `order_count_bid` | Bid order count | Count of bid orders |
| `order_count_ask` | Ask order count | Count of ask orders |
| `book_pressure` | Book pressure | log(depth_bid / depth_ask) |

</details>

<details>
<summary><b>11. Trade Flow Features (12)</b> — Execution patterns</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `trade_count_1m` | Trade count (1m) | Count of trades |
| `volume_1m` | Volume (1m) | Σ trade_size |
| `volume_notional_1m` | Notional volume (1m) | Σ (price × size) |
| `aggressor_ratio` | Buy aggressor ratio | V_taker_buy / V_total |
| `vwap_1m` | VWAP (1m) | Σ(P×V) / Σ V |
| `vwap_deviation` | Price vs VWAP | (P - VWAP) / VWAP |
| `trade_intensity` | Trades per second | trades / seconds |
| `avg_trade_size` | Average trade size | V_total / n_trades |
| `large_trade_ratio` | Large trade proportion | V_{size>threshold} / V_total |
| `trade_clustering` | Arrival clustering | Variance of inter-arrival times |
| `buy_volume` | Buy volume | Σ V where side = buy |
| `sell_volume` | Sell volume | Σ V where side = sell |

</details>

<details>
<summary><b>12. Market Context Features (9)</b> — External conditions</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `funding_rate` | Current funding rate | From exchange API |
| `funding_rate_annualized` | Annualized funding | rate × 3 × 365 × 100 |
| `open_interest` | Total open interest | Σ \|positions\| |
| `oi_change_1h` | OI change (1h) | OIₜ - OIₜ₋₁ₕ |
| `premium` | Futures premium | (perp_price - spot) / spot |
| `volume_ratio_24h` | Volume vs 24h avg | V₁ₕ / (V₂₄ₕ / 24) |
| `mark_oracle_divergence` | Mark vs oracle | (mark - oracle) / oracle |
| `basis` | Futures basis | futures - spot |
| `time_to_funding` | Time to next funding | seconds remaining |

</details>

<details>
<summary><b>13. Derived/Composite Features (15)</b> — Combined signals</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `entropy_trend_interaction` | Entropy × trend | H × \|momentum\| |
| `flow_volatility_interaction` | Flow × volatility | \|imbalance\| × σ |
| `whale_entropy_regime` | Whale in low entropy | whale_flow × (1 - H_norm) |
| `liquidation_volatility_product` | Liq risk × vol | liq_risk × realized_vol |
| `concentration_momentum` | Concentration × momentum | HHI × momentum |
| `toxicity_adjusted_spread` | Toxicity-adjusted spread | spread × (1 + VPIN) |
| `regime_indicator` | Market regime | f(entropy, vol, trend) |
| `signal_strength` | Combined signal | Weighted sum of signals |
| `risk_adjusted_flow` | Flow / volatility | whale_flow / σ |
| `crowding_momentum` | Crowding × momentum | concentration × momentum |
| `smart_money_indicator` | Whale vs retail | whale_direction - retail_direction |
| `mean_reversion_probability` | Reversion probability | sigmoid(z-score) |
| `breakout_probability` | Breakout probability | 1 - P(reversion) |
| `regime_transition_prob` | Regime change prob | From HMM or entropy change |
| `composite_alpha` | Combined alpha | Σ wᵢ × signalᵢ |

</details>

<details>
<summary><b>14. Regime Detection Features (20)</b> — Accumulation & distribution phases</summary>

| Feature | Description | Formula |
|---------|-------------|---------|
| `regime_absorption_1h` | Absorption ratio (1h) | AR = Σ(Volume) / (\|ΔPrice\| + ε) |
| `regime_absorption_4h` | Absorption ratio (4h) | Same, 4h window |
| `regime_absorption_24h` | Absorption ratio (24h) | Same, 24h window |
| `regime_absorption_zscore` | Absorption z-score | (AR - μ) / σ over history |
| `regime_divergence_1h` | Volume-price divergence (1h) | Actual_ΔP - λ × SignedVolume |
| `regime_divergence_4h` | Volume-price divergence (4h) | Same, 4h window |
| `regime_divergence_24h` | Volume-price divergence (24h) | Same, 24h window |
| `regime_divergence_zscore` | Divergence z-score | (div - μ) / σ over history |
| `regime_kyle_lambda` | Kyle's lambda estimate | λ = Cov(ΔP, SignedVol) / Var(SignedVol) |
| `regime_churn_1h` | Churn rate (1h) | (BuyVol + SellVol) / (\|BuyVol - SellVol\| + ε) |
| `regime_churn_4h` | Churn rate (4h) | Same, 4h window |
| `regime_churn_24h` | Churn rate (24h) | Same, 24h window |
| `regime_churn_zscore` | Churn z-score | (churn - μ) / σ over history |
| `regime_range_pos_4h` | Range position (4h) | (Price - Min) / (Max - Min) |
| `regime_range_pos_24h` | Range position (24h) | Same, 24h window |
| `regime_range_pos_1w` | Range position (1 week) | Same, 1 week window |
| `regime_range_width_24h` | Range width (24h) | (Max - Min) / Midpoint |
| `regime_accumulation_score` | Accumulation score [0,1] | Weighted combination of signals |
| `regime_distribution_score` | Distribution score [0,1] | Weighted combination of signals |
| `regime_clarity` | Regime clarity | \|accumulation - distribution\| |

**Regime Signal Interpretation:**
- **High absorption + negative divergence + low range position** = Accumulation (buying without price rise)
- **High absorption + positive divergence + high range position** = Distribution (selling without price fall)
- **Low churn** = One-directional flow (trending)
- **High churn** = Two-sided activity (position transfer)

</details>

---

## Real-Time Dashboard

Monitor NAT remotely via WebSocket-powered dashboard:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      LIVE MONITORING DASHBOARD                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────┐     ┌─────────────────────┐              │
│   │    LIVE LOGS        │     │    MARKET STATE     │              │
│   │  Real-time tracing  │     │  • Order book depth │              │
│   │  with severity      │     │  • Trade flow       │              │
│   │  filtering          │     │  • Whale activity   │              │
│   └─────────────────────┘     └─────────────────────┘              │
│                                                                     │
│   Access: http://localhost:8080                                     │
│   Remote: cloudflared tunnel --url http://localhost:8080           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Quick Start:**
```bash
make run_and_serve    # Start ingestor + dashboard
make tunnel           # Expose to internet (requires cloudflared)
```

---

## Data Validation

Automated quality checks for collected Parquet data:

| Check | Description | Threshold |
|-------|-------------|-----------|
| **File Integrity** | Parquet schema validation | All files must load |
| **Continuity** | Gap detection between files | < 5 min gaps |
| **NaN Ratio** | Missing value detection | < 1% per column |
| **Feature Ranges** | Outlier detection | Within 6σ of mean |
| **Cross-Symbol** | Price correlation check | BTC-ETH r > 0.5 |
| **Data Rate** | Expected row frequency | > 0.5 rows/sec |
| **Sequence** | Monotonic sequence IDs | No duplicates |

**Quick Start:**
```bash
make validate_data           # Validate all data
make validate_data_recent HOURS=24   # Validate last 24h
```

---

## Visualization Module

Python visualization library for feature exploration:

```python
from scripts.viz import FeaturePlotter, EventPlotter, CorrelationAnalyzer

# Feature time series
plotter = FeaturePlotter(df)
plotter.plot_feature_timeseries(['whale_net_flow_1h', 'vpin_10'])

# Event analysis
events = EventPlotter(df)
events.plot_event_response('whale_flow_spike', window=60)

# Correlation analysis
analyzer = CorrelationAnalyzer(df)
analyzer.plot_correlation_matrix(method='spearman')
analyzer.plot_feature_clusters()
```

**Quick Start:**
```bash
make explore    # Launch Jupyter notebook
```

---

## Architecture

```
nat/
├── rust/ing/                    # Core Rust engine
│   ├── src/
│   │   ├── main.rs              # Entry point & orchestration
│   │   ├── ws/                  # WebSocket client (Hyperliquid)
│   │   ├── rest/                # REST API client
│   │   ├── state/               # Order book state management
│   │   ├── features/            # Feature extraction (183 features)
│   │   │   ├── entropy.rs       # Tick entropy features
│   │   │   ├── trend.rs         # Momentum, Hurst, monotonicity
│   │   │   ├── illiquidity.rs   # Kyle, Amihud, Hasbrouck
│   │   │   ├── toxicity.rs      # VPIN, adverse selection
│   │   │   ├── whale_flow.rs    # Whale tracking features
│   │   │   ├── concentration.rs # Position concentration
│   │   │   ├── liquidation.rs   # Liquidation mapping
│   │   │   ├── regime/          # Regime detection (20 features)
│   │   │   │   ├── absorption.rs    # Volume absorption ratio
│   │   │   │   ├── divergence.rs    # Kyle's lambda divergence
│   │   │   │   ├── churn.rs         # Two-sided activity
│   │   │   │   ├── range.rs         # Price range position
│   │   │   │   ├── composite.rs     # Accumulation/distribution scores
│   │   │   │   └── buffer.rs        # Regime buffer coordinator
│   │   │   └── ...
│   │   ├── hypothesis/          # Statistical testing framework
│   │   │   ├── stats.rs         # Core statistical functions
│   │   │   ├── h1_whale_flow.rs
│   │   │   ├── h2_entropy_whale.rs
│   │   │   ├── h3_liquidation_cascade.rs
│   │   │   ├── h4_concentration_vol.rs
│   │   │   ├── h5_persistence.rs
│   │   │   ├── feature_analysis.rs
│   │   │   └── final_decision.rs
│   │   ├── dashboard/           # Real-time monitoring
│   │   │   ├── server.rs        # Axum WebSocket server
│   │   │   ├── handlers.rs      # HTTP & WS handlers
│   │   │   ├── state.rs         # Broadcast channels
│   │   │   └── tracing.rs       # Log capture layer
│   │   ├── whales/              # Whale registry & classification
│   │   ├── positions/           # Position tracking
│   │   ├── output/              # Parquet writer
│   │   └── metrics/             # Prometheus metrics
│   └── config/                  # Configuration files
├── scripts/
│   ├── validate_data.py         # Data quality validation
│   └── viz/                     # Visualization library
│       ├── loader.py            # Parquet data loader
│       ├── features.py          # Feature plotting
│       ├── events.py            # Event analysis
│       ├── correlations.py      # Correlation matrices
│       └── distributions.py     # Distribution analysis
└── docs/                        # Documentation & research
```

---

## Performance

Built in Rust for production workloads:

| Metric | Performance |
|--------|-------------|
| Feature computation | < 1ms per tick |
| Memory footprint | ~50MB per symbol |
| Throughput | 10,000+ updates/sec |
| Output format | Parquet (columnar, compressed) |

---

## Quick Start

```bash
# Build release version
make release

# Run ingestor
make run

# Run with live dashboard (http://localhost:8080)
make run_and_serve

# Expose dashboard to internet
make tunnel

# Run all tests
make test

# Validate collected data
make validate_data

# Launch feature exploration notebook
make explore

# Run hypothesis tests on data
make test_hypotheses

# Show real-time features in terminal
make show SYMBOL=BTC FREQ=10

# See all available commands
make help
```

---

## Configuration

```toml
[symbols]
assets = ["BTC", "ETH", "SOL"]

[features]
emission_interval_ms = 1000
entropy_windows = [1, 5, 10, 30, 60, 900]

[websocket]
url = "wss://api.hyperliquid.xyz/ws"
reconnect_delay_ms = 1000

[output]
path = "./data"
rotation_interval_secs = 3600
```

---

## Research Output

NAT generates structured output for downstream analysis:

**Parquet Schema:**
- Timestamp (ns precision)
- Symbol
- 183 feature columns (including 20 regime features)
- Metadata (sequence ID, data quality flags)

**Decision Report:**
- Hypothesis test results with confidence intervals
- Strategy estimates (Sharpe, capacity, alpha decay)
- Recommended feature subset
- Honest assessment & next steps

---

## Test Coverage

```
287 tests across:
├── Feature extraction (140+ tests)
│   └── Regime detection (21 tests)
├── Statistical functions (30+ tests)
├── Hypothesis H1-H5 (85+ tests)
├── Feature analysis (14 tests)
└── Final decision (12 tests)
```

---

## Roadmap

- [x] Real-time feature extraction (183 features)
- [x] Whale tracking & classification
- [x] Hypothesis testing framework (H1-H5)
- [x] Feature redundancy analysis
- [x] GO/PIVOT/NO-GO decision engine
- [x] Real-time WebSocket dashboard
- [x] Data validation pipeline
- [x] Visualization library
- [x] Regime detection features (accumulation/distribution)
- [ ] Backtesting infrastructure
- [ ] Paper trading integration
- [ ] Live deployment

---

## Philosophy

> "In God we trust. All others must bring data." — W. Edwards Deming

NAT is built on the principle that **every trading signal must be statistically validated** before deployment. No hunches. No vibes. No "it worked in backtest."

The framework is intentionally skeptical — designed to reject weak signals and prevent overfitting, even at the cost of rejecting some potentially valid hypotheses.

---

## Academic References

NAT's methodology is grounded in peer-reviewed research from quantitative finance, market microstructure, information theory, and statistical learning. Below are the foundational publications underlying our feature extraction and hypothesis testing framework.

### Market Microstructure & Price Impact

| Citation | Contribution |
|----------|--------------|
| Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." *Econometrica*, 53(6), 1315-1335. | Kyle's λ — price impact of informed trading |
| Glosten, L. & Milgrom, P. (1985). "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders." *Journal of Financial Economics*, 14(1), 71-100. | Information asymmetry in bid-ask spreads |
| Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market." *Journal of Finance*, 39(4), 1127-1139. | Roll's implied spread estimator |
| Hasbrouck, J. (1991). "Measuring the Information Content of Stock Trades." *Journal of Finance*, 46(1), 179-207. | Trade informativeness measurement |
| Hasbrouck, J. (1991). "The Summary Informativeness of Stock Trades: An Econometric Analysis." *Review of Financial Studies*, 4(3), 571-595. | VAR-based price impact models |
| O'Hara, M. (1995). *Market Microstructure Theory*. Blackwell Publishers. | Comprehensive microstructure framework |
| Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3(2), 5-39. | Market impact and optimal execution |

### Illiquidity & Toxicity Measures

| Citation | Contribution |
|----------|--------------|
| Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects." *Journal of Financial Markets*, 5(1), 31-56. | Amihud illiquidity ratio |
| Easley, D., Kiefer, N.M., O'Hara, M. & Paperman, J. (1996). "Liquidity, Information, and Infrequently Traded Stocks." *Journal of Finance*, 51(4), 1405-1436. | PIN — Probability of Informed Trading |
| Easley, D., López de Prado, M. & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency World." *Review of Financial Studies*, 25(5), 1457-1493. | VPIN toxicity metric |
| Easley, D., López de Prado, M. & O'Hara, M. (2011). "The Microstructure of the 'Flash Crash'." *Journal of Portfolio Management*, 37(2), 118-128. | Toxicity and liquidity crashes |
| Lee, C. & Ready, M. (1991). "Inferring Trade Direction from Intraday Data." *Journal of Finance*, 46(2), 733-746. | Trade classification algorithm |

### Order Flow & Imbalance

| Citation | Contribution |
|----------|--------------|
| Chordia, T., Roll, R. & Subrahmanyam, A. (2002). "Order Imbalance, Liquidity, and Market Returns." *Journal of Financial Economics*, 65(1), 111-130. | Order imbalance effects on returns |

### Volatility Estimation

| Citation | Contribution |
|----------|--------------|
| Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the Rate of Return." *Journal of Business*, 53(1), 61-65. | High-low volatility estimator |
| Garman, M.B. & Klass, M.J. (1980). "On the Estimation of Security Price Volatilities from Historical Data." *Journal of Business*, 53(1), 67-78. | OHLC volatility estimator |
| Rogers, L.C.G. & Satchell, S.E. (1991). "Estimating Variance from High, Low and Closing Prices." *Annals of Applied Probability*, 1(4), 504-512. | Drift-independent volatility estimator |
| Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327. | GARCH volatility clustering |
| Black, F. (1976). "Studies of Stock Price Volatility Changes." *Proceedings of the Business and Economic Statistics Section*, American Statistical Association, 177-181. | Leverage effect |

### Information Theory & Entropy

| Citation | Contribution |
|----------|--------------|
| Shannon, C.E. (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27, 379-423, 623-656. | Shannon entropy foundation |
| Bandt, C. & Pompe, B. (2002). "Permutation Entropy: A Natural Complexity Measure for Time Series." *Physical Review Letters*, 88(17), 174102. | Permutation entropy for time series |
| Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience. | Mutual information theory |
| Theil, H. (1967). *Economics and Information Theory*. North-Holland, Amsterdam. | Theil entropy index |

### Concentration & Inequality Measures

| Citation | Contribution |
|----------|--------------|
| Gini, C. (1912). "Variabilità e Mutabilità." *Reprinted in Memorie di Metodologica Statistica*. | Gini coefficient |
| Herfindahl, O.C. (1950). "Concentration in the U.S. Steel Industry." Doctoral dissertation, Columbia University. | Herfindahl-Hirschman Index |
| Hirschman, A.O. (1964). "The Paternity of an Index." *American Economic Review*, 54(5), 761-762. | HHI attribution |

### Time Series & Persistence

| Citation | Contribution |
|----------|--------------|
| Hurst, H.E. (1951). "Long-Term Storage Capacity of Reservoirs." *Transactions of the American Society of Civil Engineers*, 116, 770-799. | Hurst exponent — long-range dependence |
| Lo, A.W. & MacKinlay, A.C. (1988). "Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test." *Review of Financial Studies*, 1(1), 41-66. | Variance ratio tests |

### Risk-Adjusted Performance

| Citation | Contribution |
|----------|--------------|
| Sharpe, W.F. (1966). "Mutual Fund Performance." *Journal of Business*, 39(1), 119-138. | Sharpe ratio |
| Fama, E.F. & French, K.R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics*, 33(1), 3-56. | Factor model foundations |

### Statistical Methods

| Citation | Contribution |
|----------|--------------|
| Bonferroni, C.E. (1936). "Teoria Statistica delle Classi e Calcolo delle Probabilità." *Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze*, 8, 3-62. | Bonferroni correction |
| Dunn, O.J. (1961). "Multiple Comparisons Among Means." *Journal of the American Statistical Association*, 56(293), 52-64. | Multiple comparison procedures |
| Pardo, R. (1992). *Design, Testing, and Optimization of Trading Systems*. Wiley. | Walk-forward validation |

### Cryptocurrency & DeFi Markets

| Citation | Contribution |
|----------|--------------|
| Makarov, I. & Schoar, A. (2020). "Trading and Arbitrage in Cryptocurrency Markets." *Journal of Financial Economics*, 135(2), 293-319. | Crypto market microstructure |
| Qin, K., Zhou, L., Gamito, P., Jovanovic, P. & Gervais, A. (2021). "An Empirical Study of DeFi Liquidations: Incentives, Risks, and Instabilities." *ACM Internet Measurement Conference*, 336-350. | DeFi liquidation mechanisms |

---

## License

Proprietary. All rights reserved.

---

<p align="center">
  <b>NAT</b> — Where Alpha Meets Rigor
</p>
