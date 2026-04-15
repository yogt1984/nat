# Hyperliquid Analytics Layer - Task Roadmap

**Created:** 2026-03-05
**Project:** Hyperliquid Perp DEX Analytics Layer
**Goal:** Validate hypotheses about alpha generation using on-chain transparency
**Timeline:** 8 weeks
**Total Tasks:** 24

---

## Executive Summary

This roadmap defines the implementation and testing plan for a Hyperliquid perpetual DEX analytics layer. The project tests whether on-chain transparency (wallet-level positions, whale tracking, liquidation visibility) provides predictive alpha that doesn't exist on centralized exchanges.

**Core Hypotheses:**
- H1: Whale flow predicts returns
- H2: Low entropy + whale agreement strengthens signals
- H3: Liquidation clustering predicts cascades
- H4: Position concentration predicts volatility
- H5: Persistence indicator works on Hyperliquid

**Decision Framework:**
- 0-1 hypotheses pass → NO-GO (insufficient evidence)
- 2-3 hypotheses pass → PIVOT (focus on validated features only)
- 4-5 hypotheses pass → GO (full development)

---

## Roadmap Visualization

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    HYPERLIQUID ANALYTICS LAYER ROADMAP                       ║
║                         24 Tasks • 8 Week Timeline                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PHASE 1: DATA FOUNDATION (Week 1-2)                                        ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │ #1  Set up Hyperliquid API connection                    [START HERE] │ ║
║  │ #2  Implement position tracking per wallet                            │ ║
║  │ #3  Identify whale wallets and compute classification                 │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                         ║
║                                    ▼                                         ║
║  PHASE 2: STANDARD FEATURES (Week 3-4)                                      ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │ #4  Port entropy features (1s, 5s, 10s, 30s, 1m, 15m)    [blocks: #1] │ ║
║  │ #5  Implement trend features (momentum, mono, Hurst)     [blocks: #1] │ ║
║  │ #6  Implement illiquidity features (Kyle, Amihud, Hasb)  [blocks: #1] │ ║
║  │ #7  Implement toxicity features (VPIN, adverse_sel)      [blocks: #1] │ ║
║  │ #8  Implement order flow features (imbalance, pressure)  [blocks: #1] │ ║
║  │ #9  Implement volume features (aggr_ratio, trade_rate)   [blocks: #1] │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                         ║
║                                    ▼                                         ║
║  PHASE 3: DERIVED FEATURES (Week 5)                                         ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │ #10 entropy_gradient (dH/dt)                             [blocks: #4] │ ║
║  │ #11 monotonicity_zscore                                  [blocks: #5] │ ║
║  │ #12 illiquidity_momentum                                 [blocks: #6] │ ║
║  │ #13 ofi_persistence (autocorrelation)                    [blocks: #8] │ ║
║  │ #14 entropy_illiquidity_ratio                         [blocks: #4,#6] │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                         ║
║                                    ▼                                         ║
║  PHASE 4: HYPERLIQUID-UNIQUE FEATURES (Week 5-6)                            ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │ #15 whale_net_flow (accumulation/distribution)        [blocks: #2,#3] │ ║
║  │ #16 liquidation_risk_map ($ at risk by price)            [blocks: #2] │ ║
║  │ #17 position_concentration (Gini, herding)            [blocks: #2,#3] │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                         ║
║                                    ▼                                         ║
║  PHASE 5: HYPOTHESIS TESTING (Week 6-7)                                     ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │ #18 H1: Does whale flow predict returns?                [blocks: #15] │ ║
║  │ #19 H2: Does entropy + whale agreement work?         [blocks: #4,#15] │ ║
║  │ #20 H3: Does liquidation clustering predict cascades?   [blocks: #16] │ ║
║  │ #21 H4: Does position concentration predict vol?        [blocks: #17] │ ║
║  │ #22 H5: Does persistence indicator work on HL?       [blocks: #4-#14] │ ║
║  │ #23 Feature correlation and redundancy analysis     [blocks: #4-#17] │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                         ║
║                                    ▼                                         ║
║  PHASE 6: FINAL DECISION (Week 8)                                           ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │ #24 FINAL DECISION: GO / PIVOT / NO-GO              [blocks: #18-#23] │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Feature Reference Tables

### Standard Features (From Ingestor)

| Feature Category | Features Available | Persistence Signal |
|------------------|-------------------|-------------------|
| **Entropy** | tick_entropy_1s, 5s, 10s, 30s, 1m, 15m | Low entropy → trends persist |
| **Trend** | momentum, monotonicity, hurst_exponent | High monotonicity → persistence |
| **Illiquidity** | kyle_lambda, amihud_lambda, hasbrouck_lambda | High illiquidity + trend → informed flow |
| **Toxicity** | vpin, adverse_selection | Rising VPIN → trend may reverse |
| **Order Flow** | order_flow_imbalance, order_flow_pressure | Sustained imbalance → persistence |
| **Volume** | aggr_ratio_10/50/100, trade_rate_10s | Volume confirmation → persistence |

### Derived Features (To Be Created)

| Derived Feature | Formula | Interpretation |
|-----------------|---------|----------------|
| entropy_gradient | dH/dt (entropy rate of change) | Rising entropy → trend weakening |
| monotonicity_zscore | (mono - rolling_mean) / rolling_std | Unusually high mono → strong trend |
| illiquidity_momentum | d(kyle_lambda)/dt | Increasing illiquidity → information flow |
| ofi_persistence | autocorr(order_flow_imbalance, lag=10) | High autocorr → imbalance persists |
| entropy_illiquidity_ratio | entropy / (1 + kyle_lambda) | Low ratio → informed trending market |

### Hyperliquid-Unique Features

| Feature | Source | Interpretation |
|---------|--------|----------------|
| whale_net_flow | Position changes by whale wallets | Positive = accumulation → bullish |
| liquidation_risk_above | $ at risk if price goes up X% | High risk → potential cascade |
| liquidation_risk_below | $ at risk if price goes down X% | High risk → potential cascade |
| liquidation_asymmetry | above_risk / below_risk | Imbalance predicts direction |
| position_gini | Gini coefficient of positions | High concentration → volatility |
| top_10_share | Top 10 wallets share of OI | Crowding indicator |
| herding_indicator | Correlation of position changes | High herding → reversal risk |

---

## Phase 1: Data Foundation (Week 1-2)

### Task #1: Set up Hyperliquid API connection and basic data ingestion

**Status:** Pending
**Blocked By:** None (START HERE)
**Blocks:** #4, #5, #6, #7, #8, #9

**Goal:** Establish connection to Hyperliquid API and verify data availability.

**Implementation:**
1. Connect to Hyperliquid REST API (https://api.hyperliquid.xyz/info)
2. Connect to WebSocket (wss://api.hyperliquid.xyz/ws)
3. Fetch and parse: market metadata, order book, recent trades
4. Verify wallet addresses are included in trade data
5. Store sample data to Parquet (1 hour of data)

**Acceptance Criteria:**
- Successfully fetch order book for BTC-PERP
- Successfully stream trades with wallet_address field populated
- Parquet file written with correct schema
- No data gaps > 5 seconds in 1-hour sample

**Skeptical Test:**
- Verify wallet_address is ACTUALLY unique per trader (not aggregated)
- Check if position data is real-time or delayed
- Document any rate limits or data restrictions

**Output:** Working ingestion script + sample Parquet file + data availability report

**AI Agent Prompt:**
```
Implement Hyperliquid API connection for the analytics layer.

Requirements:
1. Connect to REST API: https://api.hyperliquid.xyz/info
2. Connect to WebSocket: wss://api.hyperliquid.xyz/ws
3. Fetch order book and trades for BTC-PERP
4. Verify wallet_address field is present in trade data
5. Store 1 hour of data to Parquet format
6. Document any rate limits or restrictions found

Reference existing code:
- /home/onat/Ingestor/src/data/lob_feed_manager.rs (WebSocket patterns)
- /home/onat/Ingestor/src/data/persistence.rs (Parquet storage)

Output:
- src/hyperliquid/api.rs (connection module)
- src/hyperliquid/ingestion.rs (data ingestion)
- tests/test_api_connection.rs
- Report: data_availability_report.md
```

---

### Task #2: Implement position tracking per wallet

**Status:** Pending
**Blocked By:** None
**Blocks:** #15, #16, #17

**Goal:** Track open positions per wallet address over time.

**Implementation:**
1. Use /clearinghouseState endpoint to fetch positions per wallet
2. Design position snapshot schema: timestamp, wallet, symbol, size, entry_price, liquidation_price
3. Implement periodic polling (every 1 minute)
4. Store position history to Parquet
5. Compute position changes between snapshots

**Acceptance Criteria:**
- Track positions for top 100 wallets by position size
- Correctly compute liquidation prices from position data
- Position deltas match trade flow (verify consistency)
- 24 hours of position history stored

**Skeptical Test:**
- Cross-validate: Do position changes match observed trades?
- Check for stale data: Are positions updating in real-time?
- Verify liquidation price formula against actual liquidations

**Output:** Position tracking module + 24h position history + consistency report

**AI Agent Prompt:**
```
Implement position tracking per wallet for Hyperliquid.

Requirements:
1. Use /clearinghouseState endpoint to fetch positions
2. Design schema: timestamp, wallet_address, symbol, size, entry_price, liquidation_price, unrealized_pnl
3. Poll every 1 minute for top 100 wallets
4. Store to Parquet with partitioning by date
5. Compute position_change = position_t - position_{t-1}
6. Verify consistency: position changes should match trade flow

Reference:
- Hyperliquid API docs for clearinghouseState
- /home/onat/Ingestor/src/data/persistence.rs

Output:
- src/hyperliquid/positions.rs
- tests/test_position_tracking.rs
- Sample data: positions_24h.parquet
- Report: consistency_validation.md
```

---

### Task #3: Identify whale wallets and compute whale classification

**Status:** Pending
**Blocked By:** None
**Blocks:** #15, #17

**Goal:** Classify wallets as whales based on position size and trading activity.

**Implementation:**
1. Define whale threshold: position_size > $500K OR 30d_volume > $10M
2. Scan all positions to identify whale wallets
3. Create whale registry with: wallet_address, first_seen, total_volume, max_position, win_rate
4. Track whale count over time
5. Compute whale concentration: top_10_whale_share of total OI

**Acceptance Criteria:**
- Identify at least 50 whale wallets
- Whale registry persisted and updated hourly
- Concentration metric computed correctly (sum of top 10 / total OI)

**Skeptical Test:**
- Are "whales" actually skilled or just large? Compute whale PnL distribution
- Do whale positions correlate with price? (Could be market makers, not directional)
- Check if whale addresses are unique traders or related wallets (clustering)

**Output:** Whale registry + whale_concentration time series + whale skill analysis

**AI Agent Prompt:**
```
Implement whale wallet identification for Hyperliquid.

Requirements:
1. Define whale: position_size > $500K OR 30d_volume > $10M
2. Scan positions to identify whales
3. Create whale registry: wallet_address, first_seen, total_volume, max_position, pnl_estimate
4. Compute whale_concentration = sum(top_10_positions) / total_OI
5. Analyze: Are whales profitable? Compute PnL distribution

Skeptical checks:
- Are whales market makers (balanced) or directional traders?
- Are multiple "whale" addresses actually the same entity?

Output:
- src/hyperliquid/whales.rs
- data/whale_registry.json
- tests/test_whale_classification.rs
- Report: whale_skill_analysis.md
```

---

## Phase 2: Standard Features (Week 3-4)

### Task #4: Port entropy features from Ingestor to Hyperliquid data

**Status:** Pending
**Blocked By:** #1
**Blocks:** #10, #14, #19, #22, #23

**Goal:** Compute tick entropy features on Hyperliquid trade data.

**Features to implement:**
- tick_entropy_1s, tick_entropy_5s, tick_entropy_10s
- tick_entropy_30s, tick_entropy_1m, tick_entropy_15m

**Implementation:**
1. Port entropy.rs logic to work with Hyperliquid trade format
2. Compute permutation entropy on tick direction sequences
3. Validate against known entropy values from Ingestor (same formula)
4. Store entropy features alongside trade data

**Acceptance Criteria:**
- Entropy values in valid range [0, 1]
- Entropy computation matches Ingestor implementation (test on synthetic data)
- Features computed at 1-second resolution

**Skeptical Test:**
- Does entropy distribution on Hyperliquid match Binance? (If very different, investigate why)
- Is entropy actually predictive on Hyperliquid? Compute correlation(entropy_t, abs(return_{t+1min}))
- Check for entropy artifacts from low tick count periods

**Output:** Entropy feature module + entropy distribution comparison report

**AI Agent Prompt:**
```
Port entropy features from Ingestor to Hyperliquid data.

Reference implementation:
- /home/onat/Ingestor/src/features/entropy.rs

Requirements:
1. Implement tick entropy at windows: 1s, 5s, 10s, 30s, 1m, 15m
2. Use permutation entropy on tick direction sequences
3. Validate: test on synthetic data, compare to Ingestor output
4. Store as feature columns in Parquet

Skeptical tests:
- Compare entropy distribution: Hyperliquid vs Binance
- Test: correlation(entropy_t, abs(return_{t+1min}))
- Check for artifacts in low-tick periods

Output:
- src/features/entropy.rs (ported)
- tests/test_entropy.rs
- Report: entropy_comparison.md (HL vs Binance distributions)
```

---

### Task #5: Implement trend features: momentum, monotonicity, Hurst exponent

**Status:** Pending
**Blocked By:** #1
**Blocks:** #11, #22, #23

**Goal:** Compute trend detection features on Hyperliquid price data.

**Features to implement:**
- momentum: Linear regression slope of prices over window
- monotonicity: Fraction of ticks in dominant direction
- hurst_exponent: R/S analysis for trend persistence

**Implementation:**
1. Port trend_features.rs logic to Hyperliquid data
2. Compute features at multiple windows: 60, 300, 600 ticks
3. Validate Hurst: H > 0.5 should correlate with trending periods
4. Store alongside price data

**Acceptance Criteria:**
- Momentum units: price change per tick (normalized)
- Monotonicity in [0.5, 1.0] range
- Hurst in [0, 1] range with mean near 0.5

**Skeptical Test:**
- Does Hurst > 0.5 actually predict trend continuation?
  Test: P(same_direction | Hurst > 0.6) vs P(same_direction | Hurst < 0.4)
- Is monotonicity just a proxy for low volatility? Check correlation
- Compare momentum predictiveness on HL vs Binance

**Output:** Trend feature module + predictiveness analysis report

**AI Agent Prompt:**
```
Implement trend features for Hyperliquid data.

Reference implementation:
- /home/onat/Ingestor/src/features/trend_features.rs

Features:
1. momentum: Linear regression slope of prices
2. monotonicity: Fraction of ticks in dominant direction [0.5, 1.0]
3. hurst_exponent: R/S analysis [0, 1], >0.5 = trending

Windows: 60, 300, 600 ticks

Skeptical tests:
- Test: P(continue | Hurst > 0.6) vs P(continue | Hurst < 0.4)
- Check if monotonicity correlates with volatility
- Compare predictiveness: HL vs Binance

Output:
- src/features/trend.rs
- tests/test_trend_features.rs
- Report: trend_predictiveness.md
```

---

### Task #6: Implement illiquidity features: Kyle, Amihud, Hasbrouck lambda

**Status:** Pending
**Blocked By:** #1
**Blocks:** #12, #14, #22, #23

**Goal:** Compute market impact / illiquidity measures.

**Features to implement:**
- kyle_lambda: Price impact per unit volume (regression slope)
- amihud_lambda: |return| / volume ratio
- hasbrouck_lambda: Permanent price impact estimate

**Implementation:**
1. Port illiquidity.rs logic to Hyperliquid data
2. Compute over rolling windows (100, 500 trades)
3. Handle edge cases: zero volume periods, extreme returns
4. Normalize by asset volatility for comparability

**Acceptance Criteria:**
- Lambda values are positive (higher = more illiquid)
- Spikes during low-volume periods (expected behavior)
- Values comparable across different market conditions

**Skeptical Test:**
- Does high illiquidity predict future volatility? Test correlation
- Does illiquidity + trend actually indicate informed flow?
  Test: P(trend_continues | high_illiq, trend) vs P(trend_continues | low_illiq, trend)
- Are illiquidity measures correlated with each other? (If r > 0.9, redundant)

**Output:** Illiquidity feature module + redundancy analysis

**AI Agent Prompt:**
```
Implement illiquidity features for Hyperliquid.

Reference:
- /home/onat/Ingestor/src/features/illiquidity.rs

Features:
1. kyle_lambda: Regression of price change on signed volume
2. amihud_lambda: |return| / volume ratio
3. hasbrouck_lambda: Permanent price impact (VAR model)

Windows: 100, 500 trades

Skeptical tests:
- correlation(illiquidity, future_volatility)
- Test: P(continue | high_illiq, trend) vs P(continue | low_illiq, trend)
- Check correlation between lambda measures (redundancy)

Output:
- src/features/illiquidity.rs
- tests/test_illiquidity.rs
- Report: illiquidity_analysis.md
```

---

### Task #7: Implement toxicity features: VPIN, adverse selection

**Status:** Pending
**Blocked By:** #1
**Blocks:** #22, #23

**Goal:** Compute order flow toxicity measures.

**Features to implement:**
- vpin: Volume-synchronized probability of informed trading
- adverse_selection: Realized spread vs quoted spread ratio

**Implementation:**
1. Port toxicity.rs logic to Hyperliquid data
2. VPIN: Classify trades as buy/sell, compute volume imbalance in buckets
3. Adverse selection: Compare trade price to subsequent mid-price
4. Rolling computation over 50-bar volume buckets

**Acceptance Criteria:**
- VPIN in [0, 1] range (0 = balanced, 1 = one-sided)
- VPIN spikes before large moves (validate on historical data)
- Adverse selection positive on average (market makers earn spread)

**Skeptical Test:**
- Does VPIN actually predict volatility on Hyperliquid?
  Test: correlation(VPIN_t, abs(return_{t+5min}))
- Does rising VPIN predict reversals as hypothesized?
  Test: P(reversal | VPIN_rising) vs P(reversal | VPIN_falling)
- Is VPIN just a proxy for volume? Check partial correlation

**Output:** Toxicity feature module + predictiveness validation

**AI Agent Prompt:**
```
Implement toxicity features for Hyperliquid.

Reference:
- /home/onat/Ingestor/src/features/toxicity.rs

Features:
1. vpin: Volume-synchronized probability of informed trading [0, 1]
2. adverse_selection: (realized_spread - quoted_spread) / quoted_spread

Implementation:
- Classify trades as buy/sell (tick rule or quote rule)
- VPIN: Volume imbalance in 50-bar buckets
- Adverse selection: Compare trade price to mid-price 1 minute later

Skeptical tests:
- correlation(VPIN, abs(return_{t+5min}))
- P(reversal | VPIN_rising) vs P(reversal | VPIN_falling)
- Partial correlation: VPIN vs returns, controlling for volume

Output:
- src/features/toxicity.rs
- tests/test_toxicity.rs
- Report: toxicity_validation.md
```

---

### Task #8: Implement order flow features: imbalance, pressure

**Status:** Pending
**Blocked By:** #1
**Blocks:** #13, #22, #23

**Goal:** Compute order flow imbalance and pressure metrics.

**Features to implement:**
- order_flow_imbalance: (buy_volume - sell_volume) / total_volume
- order_flow_pressure: Signed cumulative imbalance

**Implementation:**
1. Classify trades as buyer/seller initiated (tick rule or quote rule)
2. Compute imbalance over rolling windows (10s, 30s, 1m, 5m)
3. Compute pressure as cumulative sum of imbalance
4. Normalize by typical volume

**Acceptance Criteria:**
- Imbalance in [-1, 1] range
- Pressure shows mean-reversion over long horizons
- Trade classification accuracy > 85% (validate against known sides)

**Skeptical Test:**
- Does order flow imbalance predict short-term returns?
  Test: correlation(OFI_t, return_{t+1min})
- Is the signal stronger on Hyperliquid than Binance? (Due to transparency)
- Does imbalance persist (autocorrelation) or mean-revert quickly?

**Output:** Order flow feature module + signal strength comparison

**AI Agent Prompt:**
```
Implement order flow features for Hyperliquid.

Features:
1. order_flow_imbalance: (buy_vol - sell_vol) / total_vol, range [-1, 1]
2. order_flow_pressure: Cumulative signed imbalance

Windows: 10s, 30s, 1m, 5m

Implementation:
- Classify trades: tick rule (price > last = buy) or quote rule (price >= ask = buy)
- Validate classification accuracy against known sides (if available)

Skeptical tests:
- correlation(OFI, return_{t+1min})
- Compare signal strength: Hyperliquid vs Binance
- Compute autocorrelation of OFI (persistence)

Output:
- src/features/order_flow.rs
- tests/test_order_flow.rs
- Report: ofi_signal_analysis.md
```

---

### Task #9: Implement volume features: aggressor ratio, trade rate

**Status:** Pending
**Blocked By:** #1
**Blocks:** #22, #23

**Goal:** Compute volume-based features for trend confirmation.

**Features to implement:**
- aggr_ratio_10, aggr_ratio_50, aggr_ratio_100: Buy volume / total volume
- trade_rate_10s: Number of trades per 10 seconds

**Implementation:**
1. Compute buy/sell volume from classified trades
2. Rolling aggressor ratio over 10, 50, 100 trade windows
3. Trade rate as count per time window
4. Volume-weighted variants

**Acceptance Criteria:**
- Aggressor ratio in [0, 1] (0.5 = balanced)
- Trade rate correlates with volatility (expected)
- Consistent computation across time windows

**Skeptical Test:**
- Does high volume actually confirm trends?
  Test: P(trend_continues | high_volume, trend) vs P(trend_continues | low_volume, trend)
- Is aggressor ratio just noise? Check autocorrelation
- Does trade rate predict future volatility? Compute MI

**Output:** Volume feature module + confirmation hypothesis test

**AI Agent Prompt:**
```
Implement volume features for Hyperliquid.

Features:
1. aggr_ratio_{10,50,100}: buy_volume / total_volume over N trades
2. trade_rate_10s: trades per 10-second window

Skeptical tests:
- P(continue | high_vol, trend) vs P(continue | low_vol, trend)
- Autocorrelation of aggressor ratio
- MI(trade_rate, future_volatility)

Output:
- src/features/volume.rs
- tests/test_volume.rs
- Report: volume_confirmation.md
```

---

## Phase 3: Derived Features (Week 5)

### Task #10: Implement derived feature: entropy_gradient (dH/dt)

**Status:** Pending
**Blocked By:** #4
**Blocks:** #22, #23

**Goal:** Compute rate of change of entropy to detect trend weakening.

**Feature:** entropy_gradient = dH/dt (entropy rate of change)
**Interpretation:** Rising entropy → trend weakening

**Implementation:**
1. Compute first difference of tick_entropy_1m: dH = H_t - H_{t-1}
2. Smooth with EMA to reduce noise: entropy_gradient = EMA(dH, span=10)
3. Normalize by typical entropy volatility
4. Store as feature column

**Acceptance Criteria:**
- Gradient centers around 0 (entropy is stationary on average)
- Positive gradient correlates with upcoming volatility increase
- Smoothing reduces noise without excessive lag

**Skeptical Test:**
- Does rising entropy actually precede reversals?
  Test: P(reversal_{t+5min} | entropy_gradient_t > 0.1) vs baseline
- Is entropy_gradient just noise? Compute autocorrelation
- What's the optimal smoothing window? Test 5, 10, 20

**Output:** entropy_gradient feature + reversal prediction test results

**AI Agent Prompt:**
```
Implement entropy_gradient derived feature.

Formula: entropy_gradient = EMA(diff(tick_entropy_1m), span=10)
Interpretation: Rising entropy → trend weakening → reversal likely

Implementation:
1. dH = entropy_t - entropy_{t-1}
2. Smooth: EMA(dH, span=10)
3. Normalize by rolling std

Skeptical tests:
- P(reversal | gradient > 0.1) vs P(reversal | baseline)
- Autocorrelation of gradient (is it just noise?)
- Optimal smoothing: test span = 5, 10, 20

Output:
- src/features/derived/entropy_gradient.rs
- tests/test_entropy_gradient.rs
- Report: entropy_gradient_validation.md
```

---

### Task #11: Implement derived feature: monotonicity_zscore

**Status:** Pending
**Blocked By:** #5
**Blocks:** #22, #23

**Goal:** Detect unusually strong trends via z-scored monotonicity.

**Feature:** monotonicity_zscore = (mono - rolling_mean) / rolling_std
**Interpretation:** Unusually high monotonicity → strong trend

**Implementation:**
1. Compute rolling mean of monotonicity (window=500)
2. Compute rolling std of monotonicity (window=500)
3. Z-score: (current - mean) / std
4. Handle edge cases: std near zero

**Acceptance Criteria:**
- Z-score approximately N(0,1) distributed
- Extreme values (|z| > 2) are rare (< 5% of samples)
- High z-score correlates with strong moves

**Skeptical Test:**
- Does high monotonicity_zscore predict continuation?
  Test: P(continue | zscore > 2) vs P(continue | zscore < 0)
- Is z-score just capturing volatility? Check correlation with realized_vol
- How quickly does signal decay? Test predictiveness at 1m, 5m, 15m horizons

**Output:** monotonicity_zscore feature + predictiveness decay analysis

**AI Agent Prompt:**
```
Implement monotonicity_zscore derived feature.

Formula: zscore = (monotonicity - rolling_mean) / rolling_std
Window: 500 samples for mean/std

Skeptical tests:
- P(continue | zscore > 2) vs P(continue | zscore < 0)
- Correlation with volatility (is it redundant?)
- Signal decay: test at 1m, 5m, 15m horizons

Output:
- src/features/derived/monotonicity_zscore.rs
- tests/test_mono_zscore.rs
- Report: mono_zscore_analysis.md
```

---

### Task #12: Implement derived feature: illiquidity_momentum

**Status:** Pending
**Blocked By:** #6
**Blocks:** #22, #23

**Goal:** Detect increasing information flow via illiquidity rate of change.

**Feature:** illiquidity_momentum = d(kyle_lambda)/dt
**Interpretation:** Increasing illiquidity → information flow arriving

**Implementation:**
1. Compute first difference of kyle_lambda
2. Smooth with EMA (span=10)
3. Normalize by typical illiquidity level
4. Handle extreme values (clip to 3 sigma)

**Acceptance Criteria:**
- Centered around 0 (illiquidity is mean-reverting)
- Positive spikes during news events / large moves
- Not dominated by outliers

**Skeptical Test:**
- Does increasing illiquidity predict larger moves?
  Test: correlation(illiq_momentum_t, abs(return_{t+5min}))
- Does illiquidity momentum add information beyond raw illiquidity?
  Test: partial correlation controlling for kyle_lambda
- Is this feature actually tradeable or just descriptive?

**Output:** illiquidity_momentum feature + incremental value test

**AI Agent Prompt:**
```
Implement illiquidity_momentum derived feature.

Formula: illiq_momentum = EMA(diff(kyle_lambda), span=10) / mean(kyle_lambda)

Skeptical tests:
- correlation(illiq_momentum, abs(return_{t+5min}))
- Partial correlation controlling for raw kyle_lambda
- Is feature tradeable or just descriptive?

Output:
- src/features/derived/illiquidity_momentum.rs
- tests/test_illiq_momentum.rs
- Report: illiq_momentum_value.md
```

---

### Task #13: Implement derived feature: ofi_persistence (order flow autocorrelation)

**Status:** Pending
**Blocked By:** #8
**Blocks:** #22, #23

**Goal:** Measure persistence of order flow imbalance.

**Feature:** ofi_persistence = autocorr(order_flow_imbalance, lag=10)
**Interpretation:** High autocorrelation → imbalance persists → trend likely continues

**Implementation:**
1. Compute rolling autocorrelation of OFI at lag 10
2. Window size: 100 samples for stable estimate
3. Handle edge cases: insufficient data, zero variance
4. Output in [-1, 1] range

**Acceptance Criteria:**
- Positive on average (OFI has some persistence)
- Higher during trending periods
- Stable computation (not noisy)

**Skeptical Test:**
- Does high OFI persistence predict trend continuation?
  Test: P(continue | ofi_persist > 0.3) vs P(continue | ofi_persist < 0.1)
- What lag is optimal? Test lag = 5, 10, 20, 50
- Is this redundant with momentum? Check correlation

**Output:** ofi_persistence feature + optimal lag analysis

**AI Agent Prompt:**
```
Implement ofi_persistence derived feature.

Formula: ofi_persistence = rolling_autocorr(OFI, lag=10, window=100)

Skeptical tests:
- P(continue | persist > 0.3) vs P(continue | persist < 0.1)
- Optimal lag: test 5, 10, 20, 50
- Correlation with momentum (redundancy check)

Output:
- src/features/derived/ofi_persistence.rs
- tests/test_ofi_persistence.rs
- Report: ofi_persistence_analysis.md
```

---

### Task #14: Implement derived feature: entropy_illiquidity_ratio

**Status:** Pending
**Blocked By:** #4, #6
**Blocks:** #22, #23

**Goal:** Combine entropy and illiquidity into single informed-trending indicator.

**Feature:** entropy_illiquidity_ratio = entropy / (1 + kyle_lambda)
**Interpretation:** Low ratio → low entropy + high illiquidity → informed trending market

**Implementation:**
1. Normalize both entropy and kyle_lambda to similar scales
2. Compute ratio: entropy / (1 + normalized_lambda)
3. Handle edge cases: lambda = 0
4. Smooth with short EMA if needed

**Acceptance Criteria:**
- Ratio is positive
- Low ratio correlates with trending periods
- Captures something different than entropy alone

**Skeptical Test:**
- Does low ratio predict continuation better than low entropy alone?
  Test: P(continue | ratio < 0.3) vs P(continue | entropy < 0.3)
- What's the marginal information gain from adding illiquidity?
  Compute MI(ratio, future_return) vs MI(entropy, future_return)
- Is this just redundant? Check correlation between ratio and components

**Output:** entropy_illiquidity_ratio feature + marginal value analysis

**AI Agent Prompt:**
```
Implement entropy_illiquidity_ratio derived feature.

Formula: ratio = entropy / (1 + normalize(kyle_lambda))
Interpretation: Low ratio → informed trending market

Skeptical tests:
- P(continue | ratio < 0.3) vs P(continue | entropy < 0.3)
- MI(ratio, return) vs MI(entropy, return) - marginal gain?
- Correlation of ratio with components (redundancy)

Output:
- src/features/derived/entropy_illiq_ratio.rs
- tests/test_entropy_illiq_ratio.rs
- Report: ratio_marginal_value.md
```

---

## Phase 4: Hyperliquid-Unique Features (Week 5-6)

### Task #15: Implement Hyperliquid-unique feature: whale_net_flow

**Status:** Pending
**Blocked By:** #2, #3
**Blocks:** #18, #19

**Goal:** Compute aggregate whale buying/selling pressure.

**Feature:** whale_net_flow = sum of position changes by whale wallets
**Interpretation:** Positive flow = whales accumulating → bullish signal

**Implementation:**
1. For each whale wallet, compute position_change = position_t - position_{t-1h}
2. Sum across all whale wallets: whale_net_flow = Σ position_changes
3. Normalize by average whale flow magnitude
4. Compute over multiple windows: 1h, 4h, 24h

**Acceptance Criteria:**
- Flow is signed (positive = buying, negative = selling)
- Reasonable magnitude (not dominated by single whale)
- Updates at least hourly

**Skeptical Test:**
- CRITICAL: Does whale flow predict returns?
  Test: correlation(whale_flow_t, return_{t+4h})
- Are whales actually informed or just large noise traders?
  Test: whale_flow vs future_return by market regime
- Is this signal crowded? Check if whale flow is already priced in

**Output:** whale_net_flow feature + predictiveness analysis (CRITICAL HYPOTHESIS TEST)

**AI Agent Prompt:**
```
Implement whale_net_flow feature (CRITICAL - Hyperliquid unique).

Formula: whale_net_flow = sum(position_change for each whale wallet)
Windows: 1h, 4h, 24h

This is the key hypothesis: Does on-chain whale visibility provide alpha?

Skeptical tests (CRITICAL):
- correlation(whale_flow, return_{t+4h}) - MUST be significant
- Whale flow vs return by regime (bull/bear/sideways)
- Is signal already priced in? Check contemporaneous vs lagged correlation

Output:
- src/features/hyperliquid/whale_flow.rs
- tests/test_whale_flow.rs
- Report: WHALE_FLOW_ALPHA_TEST.md (detailed analysis)
```

---

### Task #16: Implement Hyperliquid-unique feature: liquidation_risk_map

**Status:** Pending
**Blocked By:** #2
**Blocks:** #20

**Goal:** Compute dollars at risk of liquidation at each price level.

**Features:**
- liquidation_risk_above: $ at risk if price goes up X%
- liquidation_risk_below: $ at risk if price goes down X%
- liquidation_asymmetry: ratio of above/below risk

**Implementation:**
1. For each position, compute liquidation price
2. Bucket liquidation prices by distance from current price (1%, 2%, 5%, 10%)
3. Sum position sizes at each bucket
4. Compute asymmetry: above_risk / below_risk

**Acceptance Criteria:**
- Risk values in dollars (reasonable magnitude)
- Asymmetry captures long/short imbalance
- Updates with position changes

**Skeptical Test:**
- Do liquidation clusters predict cascades?
  Test: When price approaches cluster, does volatility increase?
- Does liquidation asymmetry predict direction?
  Test: correlation(asymmetry, future_return)
- Is this signal actionable or just descriptive?

**Output:** liquidation_risk_map features + cascade prediction test

**AI Agent Prompt:**
```
Implement liquidation_risk_map feature.

Features:
1. liquidation_risk_above_{1,2,5,10}pct: $ at risk if price rises
2. liquidation_risk_below_{1,2,5,10}pct: $ at risk if price falls
3. liquidation_asymmetry: above_risk / below_risk

Implementation:
- Compute liquidation price for each position
- Bucket by distance: 1%, 2%, 5%, 10% from current
- Sum position sizes in each bucket

Skeptical tests:
- Does approaching cluster increase volatility?
- correlation(asymmetry, future_return)
- Lead time: How far ahead can we detect cascade risk?

Output:
- src/features/hyperliquid/liquidation_map.rs
- tests/test_liquidation_map.rs
- Report: liquidation_cascade_test.md
```

---

### Task #17: Implement Hyperliquid-unique feature: position_concentration

**Status:** Pending
**Blocked By:** #2, #3
**Blocks:** #21

**Goal:** Measure how concentrated positions are across wallets.

**Features:**
- position_gini: Gini coefficient of position sizes (0 = equal, 1 = concentrated)
- top_10_share: Share of OI held by top 10 wallets
- herding_indicator: Correlation of position changes across wallets

**Implementation:**
1. Compute Gini coefficient: G = (Σ|x_i - x_j|) / (2n²μ)
2. Compute top-10 share: sum(top_10_positions) / total_OI
3. Herding: correlation of Δposition across wallets over rolling window

**Acceptance Criteria:**
- Gini in [0, 1] range
- Top-10 share typically 20-60% (validate on data)
- Herding indicator meaningful (not always near 0)

**Skeptical Test:**
- Does high concentration predict volatility?
  Test: correlation(gini, future_volatility)
- Does herding predict reversals? (Everyone on same side = crowded trade)
  Test: P(reversal | herding > 0.5) vs baseline
- Is concentration actually informative or just market structure?

**Output:** position_concentration features + crowding analysis

**AI Agent Prompt:**
```
Implement position_concentration features.

Features:
1. position_gini: Gini coefficient of |position| sizes [0, 1]
2. top_10_share: sum(top_10) / total_OI
3. herding_indicator: cross-sectional correlation of position changes

Skeptical tests:
- correlation(gini, future_volatility)
- P(reversal | herding > 0.5) vs baseline
- Is concentration informative or just structure?

Output:
- src/features/hyperliquid/concentration.rs
- tests/test_concentration.rs
- Report: concentration_analysis.md
```

---

## Phase 5: Hypothesis Testing (Week 6-7)

### Task #18: HYPOTHESIS TEST H1: Does whale flow predict returns?

**Status:** Pending
**Blocked By:** #15
**Blocks:** #24

**Goal:** Test if whale accumulation/distribution predicts future price movement.

**Hypothesis:** whale_net_flow_t → return_{t+Δ} (positive correlation)

**Test Protocol:**
1. Compute whale_net_flow for 1h, 4h, 24h windows
2. Compute forward returns for 1h, 4h, 24h horizons
3. Test correlations for all window/horizon combinations
4. Apply multiple testing correction (Bonferroni)
5. Compute Mutual Information for non-linear relationships
6. Run walk-forward validation (train/test split)

**Success Criteria:**
- correlation > 0.05 with p < 0.001 (after correction)
- MI > 0.02 bits
- Walk-forward OOS correlation > 0.5 * in-sample

**Failure Criteria:**
- correlation < 0.02 or p > 0.01
- MI < 0.01 bits
- Walk-forward degradation > 60%

**Skeptical Checks:**
- Is signal strongest during specific market conditions?
- Does signal decay after discovery (if others use it)?
- Are whales skilled or just coincidentally right?

**Output:** H1 test report with GO/NO-GO decision for whale features

**AI Agent Prompt:**
```
HYPOTHESIS TEST H1: Does whale flow predict returns?

This is a CRITICAL test. Be skeptical and rigorous.

Test protocol:
1. Compute correlations: whale_flow × future_return
   - Windows: 1h, 4h, 24h flow
   - Horizons: 1h, 4h, 24h returns
2. Apply Bonferroni correction (9 tests)
3. Compute MI for each combination
4. Walk-forward validation: 70% train, 30% test, 5 folds

Success: corr > 0.05, p < 0.001, MI > 0.02 bits, OOS > 0.5*IS
Failure: corr < 0.02, p > 0.01, MI < 0.01 bits

Output:
- Correlation matrix with p-values
- MI scores
- Walk-forward results
- Clear GO/NO-GO decision
- Report: H1_WHALE_FLOW_TEST.md
```

---

### Task #19: HYPOTHESIS TEST H2: Does low entropy + whale agreement strengthen signal?

**Status:** Pending
**Blocked By:** #4, #15
**Blocks:** #24

**Goal:** Test if combining entropy and whale signals produces better predictions.

**Hypothesis:** P(up | low_entropy, whale_buying) > P(up | low_entropy) > P(up | baseline)

**Test Protocol:**
1. Define conditions:
   - low_entropy: tick_entropy_1m < 0.4
   - whale_buying: whale_net_flow > threshold
2. Compute conditional probabilities:
   - P(return > 0 | low_entropy, whale_buying)
   - P(return > 0 | low_entropy, NOT whale_buying)
   - P(return > 0 | whale_buying, NOT low_entropy)
   - P(return > 0 | baseline)
3. Test significance with chi-squared
4. Compute information gain: MI(return | entropy, whale) vs MI(return | entropy)

**Success Criteria:**
- Conditional probability lift > 10% vs single factor
- Interaction effect significant (p < 0.01)
- Information gain > 0.01 bits

**Failure Criteria:**
- No significant interaction effect
- Combined signal not better than best single signal
- MI gain < 0.005 bits

**Skeptical Checks:**
- Is this just overfitting to two conditions?
- Does the interaction hold out-of-sample?
- Are there enough samples in the joint condition?

**Output:** H2 interaction test report + decision on signal combination

**AI Agent Prompt:**
```
HYPOTHESIS TEST H2: Does entropy + whale agreement work?

Test protocol:
1. Define conditions:
   - low_entropy: entropy_1m < 0.4
   - whale_buying: whale_flow > 75th percentile
2. Compute conditional probabilities (2x2 table)
3. Chi-squared test for interaction
4. Compute MI(return | entropy, whale) vs MI(return | entropy)

Success: lift > 10%, p < 0.01, MI gain > 0.01 bits
Failure: no interaction, combined <= best single

Skeptical: Sample size in joint condition? OOS validation?

Output:
- Contingency table with probabilities
- Chi-squared results
- MI comparison
- Report: H2_INTERACTION_TEST.md
```

---

### Task #20: HYPOTHESIS TEST H3: Does liquidation clustering predict cascades?

**Status:** Pending
**Blocked By:** #16
**Blocks:** #24

**Goal:** Test if concentrated liquidation levels predict cascade events.

**Hypothesis:** When price approaches liquidation cluster → volatility spike

**Test Protocol:**
1. Define liquidation cluster: > $X million within Y% of current price
2. Define cascade event: > 5% price move within 1 hour
3. Compute: P(cascade | approaching_cluster) vs P(cascade | no_cluster)
4. Test with different thresholds for X and Y
5. Validate on out-of-sample period

**Success Criteria:**
- P(cascade | cluster) > 2x P(cascade | no_cluster)
- Precision > 30% (when we predict cascade, we're right 30%+)
- Signal is actionable (enough lead time)

**Failure Criteria:**
- No significant difference in cascade probability
- Precision < 15%
- Clusters are too rare to be useful

**Skeptical Checks:**
- Are we just detecting volatility, not predicting it?
- Do cascades happen WITHOUT approaching clusters? (False negative rate)
- Is this signal already priced into options/funding?

**Output:** H3 cascade prediction test + trading viability assessment

**AI Agent Prompt:**
```
HYPOTHESIS TEST H3: Liquidation clustering predicts cascades?

Test protocol:
1. Define cluster: > $10M within 2% of current price
2. Define cascade: > 5% move within 1 hour
3. Compute P(cascade | approaching cluster) vs P(cascade | no cluster)
4. Test thresholds: $5M/$10M/$20M × 1%/2%/5%
5. OOS validation

Success: P(cascade|cluster) > 2x P(cascade|no_cluster), precision > 30%
Failure: No difference, precision < 15%

Output:
- Cascade prediction confusion matrix
- Precision/recall by threshold
- Lead time analysis
- Report: H3_CASCADE_TEST.md
```

---

### Task #21: HYPOTHESIS TEST H4: Does position concentration predict volatility?

**Status:** Pending
**Blocked By:** #17
**Blocks:** #24

**Goal:** Test if crowded positioning predicts future volatility.

**Hypothesis:** High Gini coefficient → future volatility increase

**Test Protocol:**
1. Compute position_gini daily
2. Compute realized volatility over next 24h, 7d
3. Test correlation(gini_t, volatility_{t+24h})
4. Test different concentration measures (Gini, top-10, herding)
5. Control for current volatility level

**Success Criteria:**
- correlation > 0.2 with p < 0.01
- Relationship holds after controlling for current vol
- Predictive power at meaningful horizon (24h+)

**Failure Criteria:**
- correlation < 0.1
- Relationship disappears when controlling for current vol
- Only predicts very short-term (not actionable)

**Skeptical Checks:**
- Is concentration just correlated with vol, not predictive?
- Does high concentration predict direction or just magnitude?
- Is this useful for trading or just risk management?

**Output:** H4 concentration-volatility test + practical application assessment

**AI Agent Prompt:**
```
HYPOTHESIS TEST H4: Concentration predicts volatility?

Test protocol:
1. Compute daily position_gini
2. Compute 24h, 7d forward realized volatility
3. Test correlation(gini, future_vol)
4. Control for current volatility (partial correlation)
5. Test Gini, top_10_share, herding separately

Success: corr > 0.2, p < 0.01, holds after controlling for current vol
Failure: corr < 0.1, disappears with controls

Output:
- Correlation matrix
- Partial correlations
- Practical application assessment
- Report: H4_CONCENTRATION_TEST.md
```

---

### Task #22: HYPOTHESIS TEST H5: Does persistence indicator work on Hyperliquid?

**Status:** Pending
**Blocked By:** #4, #5, #6, #7, #8, #9, #10, #11, #12, #13, #14
**Blocks:** #24

**Goal:** Validate that your persistence indicator from Ingestor works on Hyperliquid data.

**Hypothesis:** Persistence indicator predicts trend continuation on Hyperliquid

**Test Protocol:**
1. Compute persistence indicator using all features:
   - entropy, momentum, monotonicity, hurst, OFI, illiquidity
2. Label data with three-bar classification (1min, 5min, 15min horizons)
3. Train persistence model on Hyperliquid data (first 70%)
4. Test on holdout (last 30%)
5. Walk-forward validation (5 folds)
6. Compare to simple momentum baseline

**Success Criteria:**
- Walk-forward Sharpe > 0.5
- OOS Sharpe > 0.7 * IS Sharpe (not overfit)
- Beat momentum baseline by > 20%
- MI(persistence_indicator, future_return) > 0.05 bits

**Failure Criteria:**
- Walk-forward Sharpe < 0.3
- OOS Sharpe < 0.5 * IS Sharpe (overfit)
- Does not beat simple baseline
- MI < 0.02 bits

**Skeptical Checks:**
- Does it work across market regimes?
- Which timeframes work? (5sec, 30sec, 1min, 5min)
- Which features contribute most? (Feature importance)

**Output:** H5 persistence indicator validation report - CRITICAL GO/NO-GO DECISION

**AI Agent Prompt:**
```
HYPOTHESIS TEST H5: Persistence indicator on Hyperliquid

This is the CRITICAL test of your core thesis.

Test protocol:
1. Compute persistence indicator from all features
2. Three-bar labels: 1min, 5min, 15min horizons
3. Train/test: 70/30 split
4. Walk-forward: 5 folds
5. Baseline: simple momentum strategy

Success: WF Sharpe > 0.5, OOS > 0.7*IS, beat baseline by 20%, MI > 0.05
Failure: WF Sharpe < 0.3, OOS < 0.5*IS, worse than baseline, MI < 0.02

Analysis:
- Performance by regime
- Performance by timeframe
- Feature importance ranking

Output:
- Walk-forward results
- Regime breakdown
- Feature importance
- Report: H5_PERSISTENCE_TEST.md (CRITICAL)
```

---

### Task #23: Compute comprehensive feature correlation and redundancy analysis

**Status:** Pending
**Blocked By:** #4, #5, #6, #7, #8, #9, #10, #11, #12, #13, #14, #15, #16, #17
**Blocks:** #24

**Goal:** Identify which features are redundant and which add unique information.

**Analysis:**
1. Compute correlation matrix for all features (20+)
2. Identify highly correlated pairs (|r| > 0.8)
3. Compute MI matrix: MI(feature_i, feature_j) for all pairs
4. Identify feature clusters (hierarchical clustering)
5. For redundant features, determine which to keep

**Output:**
- Correlation heatmap
- Feature clusters visualization
- Recommended feature subset (non-redundant)
- MI contribution of each feature to target

**Decision Criteria:**
- If two features have r > 0.9, keep only the more predictive one
- Aim for feature set of 10-15 non-redundant features
- Each feature must have MI > 0.01 bits with target to be included

**Output:** Feature redundancy report + recommended feature subset

**AI Agent Prompt:**
```
Feature redundancy and correlation analysis.

Analysis:
1. Correlation matrix (all features)
2. Identify pairs with |r| > 0.8
3. MI matrix: MI(feature_i, feature_j)
4. Hierarchical clustering of features
5. MI(feature, target) for each feature

Decision rules:
- r > 0.9 → keep more predictive one
- Target: 10-15 non-redundant features
- Each must have MI > 0.01 bits with target

Output:
- Correlation heatmap (save as image)
- Feature clusters dendrogram
- Recommended feature subset
- Report: FEATURE_REDUNDANCY.md
```

---

## Phase 6: Final Decision (Week 8)

### Task #24: FINAL DECISION: Aggregate hypothesis results and determine project viability

**Status:** Pending
**Blocked By:** #18, #19, #20, #21, #22, #23
**Blocks:** None (FINAL TASK)

**Goal:** Make GO/NO-GO decision on continuing the Hyperliquid analytics project.

**Input:** Results from H1, H2, H3, H4, H5 hypothesis tests

**Decision Matrix:**

| Hypotheses Passed | Decision |
|-------------------|----------|
| 0-1 of 5         | NO-GO: Insufficient evidence of alpha |
| 2-3 of 5         | PIVOT: Focus only on validated features |
| 4-5 of 5         | GO: Full development of analytics layer |

**Report Contents:**
1. Summary table: Each hypothesis, result, confidence
2. Feature ranking by predictive power
3. Recommended feature subset for trading
4. Estimated strategy Sharpe (conservative)
5. Recommended next steps

**Honest Assessment:**
- What worked? What didn't?
- What assumptions were wrong?
- Is there enough edge to justify continued development?
- How much runway until alpha decays?

**Output:** Final viability report with clear GO/PIVOT/NO-GO decision

**AI Agent Prompt:**
```
FINAL DECISION: Project viability assessment

Aggregate all hypothesis test results:
- H1: Whale flow → returns
- H2: Entropy + whale interaction
- H3: Liquidation cascades
- H4: Concentration → volatility
- H5: Persistence indicator

Decision matrix:
- 0-1 pass: NO-GO
- 2-3 pass: PIVOT (focus on what works)
- 4-5 pass: GO (full development)

Report:
1. Summary table with pass/fail and confidence
2. Feature ranking by MI
3. Recommended feature subset
4. Conservative Sharpe estimate
5. Next steps

Be HONEST:
- What assumptions were wrong?
- Is there enough edge?
- What's the alpha decay timeline?

Output:
- FINAL_DECISION.md with clear GO/PIVOT/NO-GO
- Recommended next steps regardless of decision
```

---

## Appendix A: AI Agent Prompt Template

For each task, use this prompt structure:

```
Implement task #{N}: {subject}

Context:
{paste full task description from this document}

Reference code:
- /home/onat/Ingestor/src/features/ (existing feature implementations)
- /home/onat/Ingestor/src/data/ (data handling patterns)
- /home/onat/Ingestor/docs/ (documentation)

Requirements:
1. Write production-quality code
2. Write comprehensive unit tests
3. Run the skeptical tests described
4. Report results in structured format

Skeptical mindset:
- Assume the hypothesis is FALSE until proven otherwise
- Look for alternative explanations
- Check for data leakage and overfitting
- Validate out-of-sample

Output format:
- Code files (properly organized)
- Test files
- Results summary
- GO/NO-GO recommendation for this specific component

Be rigorous. Be skeptical. Report honestly.
```

---

## Appendix B: Success Metrics Summary

| Hypothesis | Test | Success Threshold | Failure Threshold |
|------------|------|-------------------|-------------------|
| H1: Whale Flow | correlation | > 0.05, p < 0.001 | < 0.02, p > 0.01 |
| H1: Whale Flow | MI | > 0.02 bits | < 0.01 bits |
| H2: Interaction | lift | > 10% | no interaction |
| H2: Interaction | MI gain | > 0.01 bits | < 0.005 bits |
| H3: Cascades | precision | > 30% | < 15% |
| H3: Cascades | lift | > 2x | < 1.2x |
| H4: Concentration | correlation | > 0.2, p < 0.01 | < 0.1 |
| H5: Persistence | WF Sharpe | > 0.5 | < 0.3 |
| H5: Persistence | OOS ratio | > 0.7 | < 0.5 |
| H5: Persistence | MI | > 0.05 bits | < 0.02 bits |

---

## Appendix C: Timeline Summary

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| 1-2 | Data Foundation | #1, #2, #3 | API connection, position tracking, whale registry |
| 3-4 | Standard Features | #4-#9 | 6 feature modules ported from Ingestor |
| 5 | Derived Features | #10-#14 | 5 derived feature modules |
| 5-6 | HL-Unique Features | #15-#17 | Whale flow, liquidation map, concentration |
| 6-7 | Hypothesis Testing | #18-#23 | 5 hypothesis test reports + redundancy analysis |
| 8 | Final Decision | #24 | GO/PIVOT/NO-GO decision document |

---

*Document created: 2026-03-05*
*Project: Hyperliquid Analytics Layer*
*Decision deadline: 8 weeks from start*
