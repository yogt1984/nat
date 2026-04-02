# Strategy Implementation Plan

**Date:** 2026-04-02
**Context:** Pivot from HFT microstructure to validated medium-frequency strategies
**Goal:** Implement profitable MA crossover with adaptive enhancements

---

## Problem Statement

**Current State:**
- 183 high-frequency microstructure features (tick-level)
- Zero validated trading strategies
- No execution framework integrated

**Validated Observations:**
- BTC MA44 crossover beats holding
- ETH MA33 crossover beats holding
- Longer time horizons more reliable
- Have illiquidity and regime features but don't know how to use them

**Goal:** Bridge from feature engineering to profitable trading

---

## Strategy Roadmap (Priority Order)

### Phase 1: Simple MA Crossover Baseline (Week 1-2)

**Goal:** Implement and validate the strategies you KNOW work

#### 1.1 Daily MA Crossover (BTC MA44, ETH MA33)

**Implementation:**
```python
# scripts/strategies/simple_ma_crossover.py

class SimpleMAStrategy:
    """
    Proven MA crossover strategy.

    BTC: 44-day MA
    ETH: 33-day MA

    Entry: Price crosses above MA
    Exit: Price crosses below MA
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ma_period = 44 if symbol == "BTC" else 33

    def generate_signals(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            daily_prices: OHLCV data at daily frequency

        Returns:
            DataFrame with columns: [timestamp, signal, ma_value, price]
        """
        df = daily_prices.copy()
        df['ma'] = df['close'].rolling(window=self.ma_period).mean()

        # Entry signal: price crosses above MA
        df['signal'] = 0
        df.loc[df['close'] > df['ma'], 'signal'] = 1  # Long
        df.loc[df['close'] < df['ma'], 'signal'] = -1  # Flat (or short)

        return df[['timestamp', 'signal', 'ma', 'close']]
```

**Validation Steps:**
1. Backtest on 6 months historical daily data
2. Measure:
   - Total return vs buy-and-hold
   - Sharpe ratio
   - Maximum drawdown
   - Win rate
   - Average holding period
3. Include realistic costs (8 bps round-trip on Hyperliquid)

**Expected Results (based on your observation):**
- Positive alpha vs holding
- Sharpe > 0.5
- Works in trending markets, loses in ranging markets

**Makefile Target:**
```makefile
backtest_simple_ma:
	python scripts/strategies/simple_ma_crossover.py \
		--symbol $(SYMBOL) \
		--data-dir ./data/daily \
		--output ./output/backtest_simple_ma.json
```

---

#### 1.2 Why This Works (and HFT features don't)

**Time Scale Mismatch:**

| Feature Type | Time Scale | Use Case | Your Situation |
|--------------|------------|----------|----------------|
| Kyle's Lambda | Tick-level | HFT market making | ❌ Not relevant for daily MA |
| VPIN | 1-5 minute bars | Intraday mean reversion | ❌ Too fast for MA44 |
| Entropy | Sub-second | Ultra HFT | ❌ Noise at daily scale |
| **Daily MA** | **24-hour bars** | **Trend following** | ✅ **Proven to work** |
| **Daily volume** | **24-hour bars** | **Regime detection** | ✅ **Can enhance MA** |

**Key Insight:** Start with features at the SAME timescale as your strategy (daily).

---

### Phase 2: Regime-Aware MA Crossover (Week 3-4)

**Goal:** Enhance MA with regime detection to avoid ranging markets

#### 2.1 Accumulation/Distribution Detection (Your Idea #1)

**Hypothesis:** MA crossover works in trending regimes (accumulation/distribution) but fails in ranging.

**Daily-Scale Regime Features:**

```python
# scripts/features/daily_regime_features.py

def compute_daily_regime_features(daily_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime features at DAILY scale.

    Unlike tick-level features, these operate on daily bars.
    """
    df = daily_ohlcv.copy()

    # 1. Trend Strength (ADX-like)
    df['range_20d'] = df['high'].rolling(20).max() - df['low'].rolling(20).min()
    df['trend_strength'] = (df['close'] - df['close'].shift(20)).abs() / df['range_20d']

    # 2. Volume Accumulation
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']

    # 3. Price Position in Range (0-1, where 0.5 = mid-range)
    df['range_position'] = (df['close'] - df['low'].rolling(20).min()) / df['range_20d']

    # 4. Directional Volume (accumulation vs distribution)
    df['up_volume'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['down_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
    df['volume_imbalance'] = (
        (df['up_volume'].rolling(20).sum() - df['down_volume'].rolling(20).sum()) /
        df['volume'].rolling(20).sum()
    )

    # 5. Regime Classification
    df['regime'] = 'ranging'  # Default

    # Accumulation: High volume, range position > 0.6, positive volume imbalance
    accumulation_mask = (
        (df['volume_ratio'] > 1.2) &
        (df['range_position'] > 0.6) &
        (df['volume_imbalance'] > 0.2) &
        (df['trend_strength'] > 0.3)
    )
    df.loc[accumulation_mask, 'regime'] = 'accumulation'

    # Distribution: High volume, range position < 0.4, negative volume imbalance
    distribution_mask = (
        (df['volume_ratio'] > 1.2) &
        (df['range_position'] < 0.4) &
        (df['volume_imbalance'] < -0.2) &
        (df['trend_strength'] > 0.3)
    )
    df.loc[distribution_mask, 'regime'] = 'distribution'

    # Trending: Strong directional move
    trending_mask = (df['trend_strength'] > 0.5)
    df.loc[trending_mask, 'regime'] = 'trending'

    return df
```

**Enhanced Strategy:**

```python
class RegimeAwareMAStrategy(SimpleMAStrategy):
    """
    MA crossover that only trades in trending/accumulation/distribution regimes.

    Sits out during ranging markets to avoid whipsaws.
    """

    def generate_signals(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        # Get base MA signals
        df = super().generate_signals(daily_prices)

        # Add regime features
        df = compute_daily_regime_features(df)

        # Filter signals: only trade in favorable regimes
        favorable_regimes = ['trending', 'accumulation', 'distribution']
        df.loc[~df['regime'].isin(favorable_regimes), 'signal'] = 0

        return df
```

**Expected Improvement:**
- Higher Sharpe ratio (fewer losing trades in ranges)
- Lower drawdown
- Lower win rate but higher profit factor

---

#### 2.2 Illiquidity-Based Regime Detection (Your Idea #2)

**Hypothesis:** Illiquidity spikes indicate regime transitions (accumulation starting, distribution ending).

**Daily Illiquidity Metrics:**

```python
def compute_daily_illiquidity(daily_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Illiquidity at daily scale.

    High illiquidity = low activity, potential accumulation zone
    Low illiquidity = high activity, potential distribution
    """
    df = daily_ohlcv.copy()

    # Kyle's Lambda at daily scale
    df['price_impact'] = (df['high'] - df['low']) / df['volume']
    df['kyle_lambda_daily'] = df['price_impact'].rolling(20).mean()

    # Amihud Illiquidity (literature standard for daily data)
    df['amihud_illiquidity'] = df['close'].pct_change().abs() / (df['volume'] * df['close'])
    df['amihud_ma_20'] = df['amihud_illiquidity'].rolling(20).mean()

    # Liquidity regime
    df['liquidity_regime'] = 'normal'
    df.loc[df['amihud_illiquidity'] > df['amihud_ma_20'] * 1.5, 'liquidity_regime'] = 'illiquid'
    df.loc[df['amihud_illiquidity'] < df['amihud_ma_20'] * 0.5, 'liquidity_regime'] = 'liquid'

    return df
```

**Insight:** Combine with regime detection:
- **Illiquid + accumulation pattern** → Strong buy signal (smart money accumulating)
- **Liquid + distribution pattern** → Strong sell signal (smart money exiting)

---

### Phase 3: Adaptive MA Period (Week 5-6)

**Goal:** Dynamically adjust MA period based on market conditions

#### 3.1 Spectral-Based MA Period Optimization (Your Idea #3)

**Hypothesis:** Market's dominant cycle length changes over time; MA should adapt.

**Implementation:**

```python
# scripts/strategies/adaptive_ma.py

import numpy as np
from scipy import signal

class AdaptiveMAStrategy:
    """
    MA crossover where MA period adapts to market's spectral characteristics.

    Uses FFT to find dominant cycle, sets MA period to 1/2 cycle length.
    """

    def __init__(self, symbol: str, lookback: int = 100):
        self.symbol = symbol
        self.lookback = lookback
        self.min_period = 20
        self.max_period = 60

    def compute_dominant_cycle(self, prices: np.ndarray) -> int:
        """
        Find dominant cycle using FFT.

        Returns period in days.
        """
        # Detrend
        detrended = signal.detrend(prices)

        # FFT
        fft = np.fft.rfft(detrended)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(prices))

        # Find dominant frequency (excluding DC component and very long cycles)
        min_freq = 1.0 / self.max_period
        max_freq = 1.0 / self.min_period

        mask = (freqs > min_freq) & (freqs < max_freq)
        if not mask.any():
            return 33  # Default

        dominant_freq = freqs[mask][np.argmax(power[mask])]
        dominant_period = int(1.0 / dominant_freq)

        return np.clip(dominant_period // 2, self.min_period, self.max_period)

    def generate_signals(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        df = daily_prices.copy()

        # Compute adaptive MA period
        df['ma_period'] = df['close'].rolling(self.lookback).apply(
            lambda x: self.compute_dominant_cycle(x.values),
            raw=False
        )

        # Compute MA with adaptive period (requires custom rolling)
        df['ma'] = self._compute_adaptive_ma(df['close'], df['ma_period'])

        # Generate signals
        df['signal'] = 0
        df.loc[df['close'] > df['ma'], 'signal'] = 1
        df.loc[df['close'] < df['ma'], 'signal'] = -1

        return df

    def _compute_adaptive_ma(self, prices: pd.Series, periods: pd.Series) -> pd.Series:
        """Compute MA where period changes dynamically."""
        ma = pd.Series(index=prices.index, dtype=float)

        for i in range(len(prices)):
            period = int(periods.iloc[i]) if not pd.isna(periods.iloc[i]) else 33
            if i >= period:
                ma.iloc[i] = prices.iloc[i-period:i].mean()

        return ma
```

---

#### 3.2 Illiquidity-Conditioned MA Period (Simpler Alternative)

**Hypothesis:** Use longer MA in illiquid markets (less noise), shorter in liquid markets (faster signals).

```python
class IlliquidityAdaptiveMA:
    """
    Simpler adaptive MA: period based on illiquidity level.

    High illiquidity → longer MA (avoid noise)
    Low illiquidity → shorter MA (capture trends)
    """

    def compute_adaptive_period(self, illiquidity: pd.Series) -> pd.Series:
        """
        Map illiquidity to MA period.

        Illiquidity in [0, high] → Period in [min_period, max_period]
        """
        # Normalize illiquidity to [0, 1]
        illiq_norm = (illiquidity - illiquidity.rolling(100).min()) / (
            illiquidity.rolling(100).max() - illiquidity.rolling(100).min()
        )

        # Map to period
        period = self.min_period + illiq_norm * (self.max_period - self.min_period)
        return period.fillna(33).astype(int)
```

**This is simpler and more interpretable than spectral analysis.**

---

### Phase 4: NautilusTrader Integration (Week 7-8)

**Goal:** Execute strategies live on Hyperliquid via NautilusTrader

#### 4.1 Why NautilusTrader?

**Advantages:**
- ✅ Production-grade execution framework
- ✅ Built-in backtesting with realistic simulation
- ✅ Multi-venue support (Hyperliquid via custom adapter)
- ✅ Risk management built-in
- ✅ Event-driven architecture

**Integration Architecture:**

```
┌─────────────────┐
│   NAT Features  │  (Your 183 features for research)
│   (Research)    │
└────────┬────────┘
         │
         │ Daily aggregation
         │
         ▼
┌─────────────────┐
│  Daily Regime   │  (Accumulation/Distribution/Trending)
│   Detection     │
└────────┬────────┘
         │
         │ Regime signals
         │
         ▼
┌─────────────────┐
│ Adaptive MA     │  (BTC MA44, ETH MA33 + adaptive)
│   Strategy      │
└────────┬────────┘
         │
         │ Trade signals
         │
         ▼
┌─────────────────┐
│ NautilusTrader  │  (Execution, risk mgmt, position sizing)
│   Execution     │
└────────┬────────┘
         │
         │ Orders
         │
         ▼
┌─────────────────┐
│  Hyperliquid    │  (Live trading, self-custody)
│   Exchange      │
└─────────────────┘
```

#### 4.2 Implementation Steps

**Step 1: Create Hyperliquid Adapter**

NautilusTrader requires a custom adapter for Hyperliquid. This involves:

```python
# integrations/nautilus/hyperliquid_adapter.py

from nautilus_trader.adapters.base import ExecutionClient
from nautilus_trader.model.commands import SubmitOrder, CancelOrder
from nautilus_trader.model.events import OrderAccepted, OrderFilled

class HyperliquidExecutionClient(ExecutionClient):
    """
    Execution client for Hyperliquid.

    Handles order submission, cancellation, and fills.
    """

    def __init__(self, client_id, venue, msgbus, cache, clock, logger, config):
        super().__init__(client_id, venue, msgbus, cache, clock, logger, config)
        # Initialize Hyperliquid API client
        self.hl_client = HyperliquidClient(
            private_key=config.private_key,
            testnet=config.testnet
        )

    async def _submit_order(self, command: SubmitOrder):
        """Submit order to Hyperliquid."""
        order = command.order

        # Convert Nautilus order to Hyperliquid format
        hl_order = {
            'coin': order.instrument_id.symbol.value,
            'is_buy': order.side == OrderSide.BUY,
            'sz': float(order.quantity),
            'limit_px': float(order.price) if order.order_type == OrderType.LIMIT else None,
            'order_type': {'Limit': 'limit', 'Market': 'market'}[order.order_type.name],
            'reduce_only': False,
        }

        # Submit to Hyperliquid
        response = await self.hl_client.submit_order(hl_order)

        # Emit OrderAccepted event
        self.generate_order_accepted(
            strategy_id=command.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=order.client_order_id,
            venue_order_id=VenueOrderId(str(response['order_id'])),
            ts_event=self._clock.timestamp_ns(),
        )
```

**Step 2: Implement Strategy in Nautilus**

```python
# integrations/nautilus/ma_crossover_strategy.py

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.orders import MarketOrder

class MAStrategyNautilus(Strategy):
    """
    MA crossover strategy in NautilusTrader framework.

    Receives daily bars, computes MA, generates orders.
    """

    def __init__(self, config):
        super().__init__(config)
        self.ma_period = config.ma_period
        self.prices = []

    def on_start(self):
        """Subscribe to daily bars."""
        self.subscribe_bars(
            BarType.from_str(f"{self.instrument_id}-1-DAY-LAST")
        )

    def on_bar(self, bar: Bar):
        """Process new daily bar."""
        self.prices.append(bar.close)

        if len(self.prices) < self.ma_period:
            return  # Not enough data

        # Compute MA
        ma = sum(self.prices[-self.ma_period:]) / self.ma_period

        # Generate signal
        if bar.close > ma and not self.portfolio.is_flat(self.instrument_id):
            # Buy signal
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.calculate_position_size(),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)

        elif bar.close < ma and not self.portfolio.is_flat(self.instrument_id):
            # Exit signal
            self.close_all_positions(self.instrument_id)
```

**Step 3: Backtest with Nautilus**

```python
# backtests/test_ma_nautilus.py

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.data import BacktestDataContainer
from nautilus_trader.model.currencies import USD, BTC

# Load historical data
data_container = BacktestDataContainer()
data_container.add_bars(
    instrument_id="BTC-USD.HYPERLIQUID",
    bar_type=BarType.from_str("BTC-USD.HYPERLIQUID-1-DAY-LAST"),
    bars=load_daily_bars("./data/daily/btc_usd.parquet")
)

# Setup engine
engine = BacktestEngine(
    data_container=data_container,
    venues=[HyperliquidVenue()],
)

# Add strategy
engine.add_strategy(MAStrategyNautilus(
    config=MAStrategyConfig(ma_period=44)
))

# Run backtest
engine.run(
    start=pd.Timestamp("2025-01-01"),
    end=pd.Timestamp("2026-04-01")
)

# Analyze results
analyzer = engine.analyzer
print(analyzer.get_metrics())
```

---

#### 4.3 Live Trading with Self-Custody

**Hyperliquid Integration:**

```python
# config/hyperliquid_config.toml

[exchange]
name = "hyperliquid"
testnet = false

[wallet]
private_key_env = "HYPERLIQUID_PRIVATE_KEY"  # Never commit keys!

[risk]
max_position_size_usd = 10000  # Start small
max_daily_loss_usd = 500
max_open_orders = 2

[strategy.btc_ma44]
symbol = "BTC"
ma_period = 44
position_size_usd = 5000

[strategy.eth_ma33]
symbol = "ETH"
ma_period = 33
position_size_usd = 3000
```

**Live Execution:**

```bash
# Paper trading first (testnet)
python integrations/nautilus/run_live.py \
    --config config/hyperliquid_config.toml \
    --testnet \
    --log-level INFO

# After 2-4 weeks of paper trading validation
python integrations/nautilus/run_live.py \
    --config config/hyperliquid_config.toml \
    --log-level INFO
```

---

## Well-Defined Promising Strategies (Your Question #2)

Here are strategies that are:
1. **Well-documented** in academic literature
2. **Simple to implement**
3. **Validated empirically**
4. **Suitable for daily timeframe**

### Tier 1: Proven, Simple (Start Here)

| Strategy | Time Horizon | Sharpe (Historical) | Complexity | Your Fit |
|----------|--------------|---------------------|------------|----------|
| **MA Crossover** | Daily | 0.4-0.8 | Very Low | ✅ You know this works |
| **Bollinger Band Mean Reversion** | Daily | 0.5-0.9 | Low | ✅ Easy to implement |
| **RSI Mean Reversion** | Daily | 0.3-0.6 | Low | ✅ Complements MA |
| **Volume Breakout** | Daily | 0.4-0.7 | Low | ✅ Uses your volume features |

### Tier 2: Moderate Complexity

| Strategy | Time Horizon | Sharpe (Historical) | Complexity | Your Fit |
|----------|--------------|---------------------|------------|----------|
| **Dual MA + Volume Filter** | Daily | 0.6-1.0 | Medium | ✅ Combines MA + volume |
| **Range Breakout (Donchian)** | Daily | 0.5-0.8 | Medium | ✅ Works in crypto |
| **Momentum (12-month)** | Weekly | 0.4-0.7 | Medium | ⚠️ Requires long history |
| **Pairs Trading (BTC-ETH)** | Daily | 0.3-0.6 | Medium | ✅ You trade both |

### Tier 3: Advanced (Later)

| Strategy | Time Horizon | Sharpe (Historical) | Complexity | Your Fit |
|----------|--------------|---------------------|------------|----------|
| **Regime-Switching MA** | Daily | 0.7-1.2 | High | ✅ Uses your regime features |
| **Adaptive MA (spectral)** | Daily | 0.6-1.0 | High | ✅ Your idea #3 |
| **Volatility Targeting** | Daily | 0.5-0.9 | High | ✅ Risk management |
| **Multi-Asset Risk Parity** | Weekly | 0.6-0.8 | High | ⚠️ Need more assets |

**Recommendation:** Start with Tier 1, master it, then move to Tier 2/3.

---

## Time Horizon vs Reliability (Your Question #3)

**You are absolutely correct.** Here's why:

### Empirical Evidence

| Time Horizon | Sharpe Ratio (Typical) | Why |
|--------------|------------------------|-----|
| **Tick-level** | 0.1-0.3 | 99% noise, 1% signal; txn costs dominate |
| **1-minute** | 0.2-0.4 | High noise, latency sensitive |
| **5-minute** | 0.3-0.5 | Still noisy, many false signals |
| **1-hour** | 0.4-0.7 | Noise reduced, trends emerge |
| **Daily** | **0.5-1.0** | **Clear trends, low txn costs** ✅ |
| **Weekly** | 0.4-0.8 | Very stable, but fewer trades |

### Why Longer Horizons Work Better

1. **Signal-to-Noise Ratio**
   - Daily: Signal dominates (trend persistence)
   - Tick: Noise dominates (bid-ask bounce, spoofing)

2. **Transaction Costs**
   - Daily MA44: ~20 trades/year × 8 bps = 1.6% annual cost
   - HFT: 1000s trades/day × 8 bps = costs destroy alpha

3. **Regime Persistence**
   - Trends last weeks (observable at daily scale)
   - Tick patterns last milliseconds (not tradeable)

4. **Overfitting Risk**
   - Daily: 250 bars/year, hard to overfit
   - Tick: millions of bars, easy to find spurious patterns

**Your 183 HFT features are designed for a game you're not playing.**

---

## Revised Feature Usage Strategy

### What to Keep from Your 183 Features

**Daily Aggregates Only:**

1. **Volume Features** (Keep)
   - Daily volume
   - Volume MA ratios
   - Up/down volume imbalance
   - **Use for:** Regime detection

2. **Price Features** (Keep)
   - Daily OHLC
   - Daily returns
   - Realized volatility (daily)
   - **Use for:** MA calculation, volatility targeting

3. **Illiquidity Features** (Transform to Daily)
   - Amihud illiquidity (daily)
   - Effective spread (daily average)
   - **Use for:** Adaptive MA period, regime detection

4. **Regime Features** (Aggregate to Daily)
   - Daily range position
   - Daily trend strength
   - **Use for:** Filter ranging markets

### What to Discard (For Now)

- ❌ Tick-level entropy (too granular)
- ❌ Sub-second VPIN (irrelevant for daily)
- ❌ Kyle's Lambda per tick (use daily version)
- ❌ Microstructure noise metrics (not needed)

**You need ~10-15 daily features, not 183 tick features.**

---

## Concrete Action Plan (Next 4 Weeks)

### Week 1: Validate Simple MA
```bash
# 1. Create daily data pipeline
python scripts/data/aggregate_to_daily.py \
    --input ./data/features/*.parquet \
    --output ./data/daily/

# 2. Implement simple MA strategy
# (scripts/strategies/simple_ma_crossover.py - code above)

# 3. Backtest
python scripts/strategies/simple_ma_crossover.py \
    --symbol BTC \
    --ma-period 44 \
    --data ./data/daily/btc.parquet \
    --output ./output/backtest_ma44.json

python scripts/strategies/simple_ma_crossover.py \
    --symbol ETH \
    --ma-period 33 \
    --data ./data/daily/eth.parquet \
    --output ./output/backtest_ma33.json

# 4. Verify it beats holding
python scripts/analysis/compare_strategies.py \
    --strategies ./output/backtest_*.json \
    --baseline buy_and_hold
```

### Week 2: Add Regime Detection
```bash
# 1. Implement daily regime features
# (scripts/features/daily_regime_features.py - code above)

# 2. Backtest regime-aware MA
python scripts/strategies/regime_aware_ma.py \
    --symbol BTC \
    --ma-period 44 \
    --regime-filter True \
    --data ./data/daily/btc.parquet

# 3. Compare to simple MA
# Expected: Higher Sharpe, lower drawdown
```

### Week 3: Adaptive MA Period
```bash
# 1. Implement illiquidity-adaptive MA
# (simpler than spectral for first version)

# 2. Backtest
python scripts/strategies/adaptive_ma.py \
    --symbol BTC \
    --adaptation-method illiquidity \
    --data ./data/daily/btc.parquet

# 3. Compare all variants:
#    - Simple MA44
#    - Regime-aware MA44
#    - Adaptive MA (illiquidity)
```

### Week 4: NautilusTrader Integration
```bash
# 1. Install NautilusTrader
pip install nautilus_trader

# 2. Create Hyperliquid adapter
# (integrations/nautilus/hyperliquid_adapter.py - code above)

# 3. Backtest in Nautilus framework
python backtests/test_ma_nautilus.py

# 4. Paper trading on testnet
python integrations/nautilus/run_live.py --testnet
```

---

## Summary: Your Path Forward

### What You Should Do

1. ✅ **Start with MA44/MA33** - You know it works
2. ✅ **Use daily timeframe** - More reliable, lower costs
3. ✅ **Enhance with regime detection** - Your accumulation/distribution idea
4. ✅ **Integrate NautilusTrader** - Professional execution
5. ✅ **Paper trade 2-4 weeks** - Validate before live

### What You Should NOT Do

1. ❌ Don't try to use all 183 features
2. ❌ Don't trade HFT with daily strategies
3. ❌ Don't deploy without paper trading
4. ❌ Don't over-optimize in backtest

### Feature Engineering Pivot

**Old approach:** 183 tick-level microstructure features
**New approach:** 10-15 daily aggregate features for regime detection

**Your HFT infrastructure is valuable for data collection, but strategy operates at daily scale.**

---

## Risk Management (Critical)

Before going live with real money:

1. **Start tiny** - $1K-5K per strategy
2. **Paper trade first** - 4-8 weeks minimum
3. **Max position size** - 5-10% of capital per trade
4. **Daily loss limit** - 2-5% of capital
5. **Review weekly** - Compare to backtest expectations

**You have empirical evidence MA works. Don't overthink it. Start simple, validate, enhance.**

---

**Next Steps:**
1. Should I create the implementation scripts for simple MA crossover?
2. Should I create the daily aggregation pipeline for your parquet files?
3. Should I create the NautilusTrader integration architecture?

Let me know which direction you want to pursue first.
