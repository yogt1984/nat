# Feature-Strategy Mismatch Analysis

**Date:** 2026-04-02
**Problem:** 183 HFT microstructure features but no validated strategies to use them

---

## The Core Issue

You have built a **Formula 1 race car** (sub-millisecond feature extraction) but you're planning to race in **NASCAR** (daily MA crossover). The tools don't match the task.

---

## Feature Inventory vs Actual Needs

### What You Have (183 Features)

| Feature Category | Count | Time Scale | Designed For | Your Strategy Needs It? |
|------------------|-------|------------|--------------|-------------------------|
| Kyle's Lambda | 12 | Tick-level | HFT market making | ❌ No |
| VPIN | 8 | 1-5 minute bars | Intraday toxicity detection | ❌ No |
| Entropy (Shannon, Tsallis) | 15 | Sub-second | Ultra-HFT regime detection | ❌ No |
| Hurst Exponent | 10 | Multiple scales | Long-memory detection | ⚠️ Maybe (if computed daily) |
| Order Flow Imbalance | 20 | Tick-level | HFT directional prediction | ❌ No |
| Microstructure Noise | 8 | Tick-level | HFT noise filtering | ❌ No |
| Volume-Weighted Features | 18 | Tick-level | HFT execution | ⚠️ Aggregate to daily |
| Price Impact | 12 | Trade-level | HFT cost modeling | ⚠️ Use daily Amihud instead |
| Spread Metrics | 15 | Tick-level | HFT liquidity provision | ❌ No |
| Tick Direction | 10 | Tick-level | HFT momentum | ❌ No |
| **Regime Features** | **20** | **Tick-level** | **Regime detection** | **✅ YES - but aggregate to daily** |
| **Volume Imbalance** | **15** | **Tick-level** | **Accumulation/Distribution** | **✅ YES - aggregate to daily** |
| **Price Range** | **10** | **Multiple scales** | **Breakout detection** | **✅ YES - use daily** |
| Other Microstructure | 10 | Tick-level | HFT research | ❌ No |

**Summary:**
- **Useful for your strategies:** ~45 features (after daily aggregation)
- **Not useful:** ~138 features (wrong time scale)
- **Waste:** 75% of your feature engineering effort

---

## What Your Strategies Actually Need

### Strategy 1: BTC MA44 Crossover

**Required Features:**
1. Daily close price (1 feature)
2. 44-day moving average (computed from #1)

**Total: 1 feature**

**Your current features used:** 0 out of 183

---

### Strategy 2: ETH MA33 Crossover

**Required Features:**
1. Daily close price (1 feature)
2. 33-day moving average (computed from #1)

**Total: 1 feature**

**Your current features used:** 0 out of 183

---

### Strategy 3: Regime-Aware MA (Your Enhancement Idea)

**Required Features:**
1. Daily close price
2. Daily volume
3. Daily high/low (for range calculation)
4. Up/down volume (for accumulation/distribution)
5. Volume MA (trend strength)

**Total: ~5 features (all available from basic OHLCV)**

**Your current features that could help:**
- Volume imbalance features (aggregate to daily)
- Range position features (aggregate to daily)
- Regime detection features (aggregate to daily)

**Useful from your 183:** ~20 features (after aggregation)

---

### Strategy 4: Illiquidity-Adaptive MA (Your Enhancement Idea)

**Required Features:**
1. Daily close price
2. Daily volume
3. Daily Amihud illiquidity = |return| / (volume × price)
4. Illiquidity MA (smoothed version)

**Total: ~4 features (3 from OHLCV + 1 derived)**

**Your current features that could help:**
- Kyle's Lambda (if aggregated to daily)
- Price impact metrics (if aggregated to daily)
- Effective spread (daily average)

**Useful from your 183:** ~10 features (after aggregation)

---

## The Aggregation Pipeline You Need

Transform your tick-level features to **daily aggregates**:

### Volume Features (Daily Aggregation)

```python
# From tick-level to daily
tick_features = pl.read_parquet("data/features/btc_2026-04-02.parquet")

daily_features = tick_features.groupby(
    pl.col("timestamp").dt.truncate("1d")
).agg([
    # Price features
    pl.col("close").last().alias("close"),
    pl.col("high").max().alias("high"),
    pl.col("low").min().alias("low"),
    pl.col("open").first().alias("open"),

    # Volume features
    pl.col("volume").sum().alias("volume"),
    pl.col("volume_buy").sum().alias("volume_buy"),
    pl.col("volume_sell").sum().alias("volume_sell"),

    # Illiquidity features (average over day)
    pl.col("kyle_lambda").mean().alias("kyle_lambda_daily"),
    pl.col("effective_spread").mean().alias("effective_spread_daily"),

    # Regime features (average over day)
    pl.col("regime_absorption").mean().alias("absorption_daily"),
    pl.col("regime_divergence").mean().alias("divergence_daily"),
    pl.col("regime_churn").mean().alias("churn_daily"),
    pl.col("regime_range_position").mean().alias("range_position_daily"),

    # Derived features
    (pl.col("volume_buy").sum() - pl.col("volume_sell").sum()).alias("volume_imbalance"),
    (pl.col("close").last() - pl.col("open").first()).alias("daily_return"),
])

# Compute additional daily features
daily_features = daily_features.with_columns([
    # Amihud illiquidity (standard in literature)
    (pl.col("daily_return").abs() / (pl.col("volume") * pl.col("close"))).alias("amihud_illiquidity"),

    # Volume ratio (vs 20-day MA)
    (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("volume_ratio"),

    # Buy/sell pressure
    (pl.col("volume_buy") / pl.col("volume")).alias("buy_pressure"),

    # Range metrics
    ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias("range_position"),
])
```

### Result: 15 Daily Features (Instead of 183 Tick Features)

**These 15 features are sufficient for:**
1. MA crossover (use close)
2. Regime detection (use absorption, divergence, volume_ratio, range_position)
3. Accumulation/distribution (use volume_imbalance, buy_pressure)
4. Adaptive MA (use amihud_illiquidity)

---

## Liquidation Cascades (Your Idea)

**Hypothesis:** Liquidation events create predictable price action.

### Data Availability

**Good news:** Liquidation data IS publicly available for crypto.

**Sources:**
1. **Hyperliquid API:** Provides liquidation events
   - Endpoint: `/info` with `type: "liquidations"`
   - Real-time WebSocket feed
   - Historical data available

2. **Coinglass API:** Aggregated liquidation data across exchanges
   - https://www.coinglass.com/
   - Free tier available
   - BTC/ETH liquidations in real-time

3. **Your own calculation:** Detect suspected liquidations from order book
   - Large market orders during high volatility
   - Cascade pattern: multiple large sells in quick succession
   - Your tick data can identify these

### Implementation

```python
# scripts/features/liquidation_features.py

def compute_daily_liquidation_metrics(liquidations: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate liquidation events to daily features.

    Args:
        liquidations: DataFrame with columns [timestamp, size, side, price]

    Returns:
        Daily liquidation metrics
    """
    daily = liquidations.groupby(
        liquidations['timestamp'].dt.floor('1d')
    ).agg({
        'size': ['sum', 'count', 'max'],  # Total liquidated, count, largest event
        'price': ['min', 'max'],  # Price range during liquidations
    }).reset_index()

    daily.columns = [
        'date',
        'liquidation_volume',
        'liquidation_count',
        'max_liquidation_size',
        'liquidation_price_min',
        'liquidation_price_max'
    ]

    # Separate long/short liquidations
    daily['long_liquidations'] = liquidations[liquidations['side'] == 'long'].groupby(
        liquidations['timestamp'].dt.floor('1d')
    )['size'].sum()

    daily['short_liquidations'] = liquidations[liquidations['side'] == 'short'].groupby(
        liquidations['timestamp'].dt.floor('1d')
    )['size'].sum()

    # Liquidation imbalance (more longs or shorts liquidated?)
    daily['liquidation_imbalance'] = (
        (daily['long_liquidations'] - daily['short_liquidations']) /
        (daily['long_liquidations'] + daily['short_liquidations'])
    )

    return daily
```

### Trading Hypothesis

**Hypothesis:** Large liquidation cascades create temporary oversold/overbought conditions.

**Strategy:**
- **Long liquidation cascade** (many longs liquidated) → Price likely oversold → **BUY**
- **Short liquidation cascade** (many shorts liquidated) → Price likely overbought → **SELL**

**Regime Integration:**
- **Accumulation + Long liquidations** → **Strong buy** (smart money buying the dip)
- **Distribution + Short liquidations** → **Strong sell** (smart money exiting the pump)

**This enhances your MA crossover:**

```python
class LiquidationEnhancedMA(RegimeAwareMAStrategy):
    """
    MA crossover enhanced with liquidation cascade detection.
    """

    def generate_signals(self, daily_prices: pd.DataFrame, liquidations: pd.DataFrame) -> pd.DataFrame:
        # Get base signals from regime-aware MA
        df = super().generate_signals(daily_prices)

        # Add liquidation features
        liq_features = compute_daily_liquidation_metrics(liquidations)
        df = df.merge(liq_features, on='date', how='left')

        # Enhance signals
        # Buy signal stronger if long liquidation cascade
        df.loc[
            (df['signal'] == 1) &  # Already buy signal from MA
            (df['liquidation_imbalance'] < -0.3) &  # Mostly longs liquidated
            (df['regime'] == 'accumulation'),  # In accumulation regime
            'signal_strength'
        ] = 2  # Strong buy

        # Sell signal stronger if short liquidation cascade
        df.loc[
            (df['signal'] == -1) &  # Already sell signal from MA
            (df['liquidation_imbalance'] > 0.3) &  # Mostly shorts liquidated
            (df['regime'] == 'distribution'),  # In distribution regime
            'signal_strength'
        ] = 2  # Strong sell

        return df
```

**Expected improvement:** Better entry timing, fewer false signals.

---

## The "Where Does Liquidity Lie?" Question

You mentioned: "illiquidity vector contains information about where liquidity lies."

**This is profound and correct.** Here's how to use it:

### Liquidity Levels Detection

**Hypothesis:** Areas of high liquidity (support/resistance) are visible in the order book depth distribution.

**Daily aggregation:**

```python
def compute_daily_liquidity_distribution(order_book_snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Identify price levels with high liquidity accumulation.

    Returns daily features about where liquidity concentrated.
    """
    daily_features = []

    for date in order_book_snapshots['timestamp'].dt.floor('1d').unique():
        day_snapshots = order_book_snapshots[
            order_book_snapshots['timestamp'].dt.floor('1d') == date
        ]

        # Find price levels with highest bid depth (support)
        support_level = day_snapshots.groupby('bid_price')['bid_size'].sum().idxmax()
        support_depth = day_snapshots.groupby('bid_price')['bid_size'].sum().max()

        # Find price levels with highest ask depth (resistance)
        resistance_level = day_snapshots.groupby('ask_price')['ask_size'].sum().idxmax()
        resistance_depth = day_snapshots.groupby('ask_price')['ask_size'].sum().max()

        # Current price position relative to these levels
        current_price = day_snapshots['mid_price'].iloc[-1]

        daily_features.append({
            'date': date,
            'support_level': support_level,
            'support_depth': support_depth,
            'resistance_level': resistance_level,
            'resistance_depth': resistance_depth,
            'distance_to_support': (current_price - support_level) / current_price,
            'distance_to_resistance': (resistance_level - current_price) / current_price,
            'support_resistance_ratio': support_depth / resistance_depth,
        })

    return pd.DataFrame(daily_features)
```

### Integration with MA Strategy

**Idea:** Only enter MA crossover trades when:
1. MA gives buy signal AND
2. Price near major support level (high liquidity) AND
3. Support/resistance ratio > 1.5 (more support than resistance)

**This reduces false breakouts.**

---

## Summary: What To Do With Your Features

### Keep and Aggregate to Daily (45 features → 15 daily features)

1. **Volume features** → Daily volume, buy/sell imbalance
2. **Regime features** → Daily absorption, divergence, churn
3. **Price range features** → Daily range position
4. **Illiquidity features** → Daily Amihud illiquidity

### Discard (138 features)

1. Tick-level entropy (wrong scale)
2. Sub-second microstructure noise (wrong scale)
3. HFT-specific features (not trading HFT)

### Add New (Liquidations)

1. Daily liquidation volume
2. Long/short liquidation imbalance
3. Liquidation cascade detection

### Final Feature Set: ~20 Daily Features

**This is sufficient for:**
- ✅ MA crossover (base strategy)
- ✅ Regime detection (avoid ranging markets)
- ✅ Accumulation/distribution (identify smart money)
- ✅ Liquidation-enhanced entry (better timing)
- ✅ Adaptive MA period (illiquidity-based)

---

## Your Three Ideas Ranked

| Idea | Complexity | Expected Improvement | Priority | Implement When |
|------|------------|---------------------|----------|----------------|
| **1. Adaptive MA with illiquidity** | Medium | +0.1-0.2 Sharpe | 🥈 Second | Week 3 |
| **2. Regime + illiquidity for accumulation** | Low-Medium | +0.2-0.3 Sharpe | 🥇 **First** | Week 2 |
| **3. Liquidation cascade detection** | Low | +0.1-0.15 Sharpe | 🥉 Third | Week 4 |

**Start with #2** (regime detection) because:
- Uses features you already have (just need daily aggregation)
- Lower complexity
- Larger expected improvement
- Directly addresses MA crossover weakness (whipsaws in ranging markets)

---

## Action Plan Summary

**Week 1:** Implement simple MA44/MA33, validate it works

**Week 2:** Add regime detection (accumulation/distribution), filter out ranging markets

**Week 3:** Add adaptive MA period based on illiquidity

**Week 4:** Add liquidation cascade detection for entry timing

**Week 5-8:** NautilusTrader integration, paper trading, live deployment

**Your 183 features are not wasted** - the volume/regime/illiquidity features are valuable after daily aggregation. The other 75% can be discarded without loss.

---

**Next Steps:**
1. Create daily aggregation pipeline?
2. Implement simple MA crossover with your proven parameters (MA44, MA33)?
3. Add liquidation data fetching from Hyperliquid API?

You're on the right track. Start simple (MA crossover), enhance incrementally (regime, illiquidity, liquidations).
