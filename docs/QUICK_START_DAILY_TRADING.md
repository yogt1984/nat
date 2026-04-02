# Quick Start: Daily Trading Strategy

**Goal:** Get from 183 unused HFT features → Profitable daily MA trading in 4 weeks

---

## TL;DR

1. **You're right:** Longer time horizons are more reliable (daily optimal)
2. **You're right:** MA44 (BTC) and MA33 (ETH) beat holding
3. **You're right:** Illiquidity + regime detection can enhance this
4. **You're missing:** Daily aggregation pipeline (have tick data, need daily)
5. **Integration:** NautilusTrader for execution on Hyperliquid

---

## Week-by-Week Plan

### Week 1: Daily Aggregation + Simple MA Backtest

**Goal:** Prove MA44/MA33 work on your data

```bash
# 1. Create daily aggregation pipeline
python scripts/data/aggregate_to_daily.py \
    --input-dir ./data/features \
    --output-dir ./data/daily \
    --symbols BTC ETH

# 2. Implement simple MA crossover
# See: scripts/strategies/simple_ma_crossover.py

# 3. Backtest (validate your observation)
python scripts/strategies/simple_ma_crossover.py \
    --symbol BTC \
    --ma-period 44 \
    --data ./data/daily/BTC.parquet \
    --output ./output/backtest_ma44.json

python scripts/strategies/simple_ma_crossover.py \
    --symbol ETH \
    --ma-period 33 \
    --data ./data/daily/ETH.parquet \
    --output ./output/backtest_ma33.json

# 4. Compare to buy-and-hold
python scripts/analysis/compare_strategies.py \
    --backtest ./output/backtest_ma44.json \
    --baseline buy_and_hold
```

**Expected results:**
- Sharpe > 0.5
- Beats buy-and-hold
- Max drawdown < buy-and-hold

**Deliverable:** Proof that your MA observation is correct

---

### Week 2: Add Regime Detection (Accumulation/Distribution)

**Goal:** Avoid whipsaws in ranging markets

```bash
# 1. Implement daily regime features
# See: scripts/features/daily_regime_features.py
# - Volume imbalance (accumulation vs distribution)
# - Range position
# - Trend strength

# 2. Backtest regime-aware MA
python scripts/strategies/regime_aware_ma.py \
    --symbol BTC \
    --ma-period 44 \
    --regime-filter True \
    --data ./data/daily/BTC.parquet

# Expected improvement:
# - Fewer trades (filter out ranging)
# - Higher Sharpe (+0.1 to +0.3)
# - Lower max drawdown
```

**Deliverable:** Enhanced strategy that sits out ranging markets

---

### Week 3: Adaptive MA Period (Illiquidity-Based)

**Goal:** Dynamically adjust MA length based on market conditions

```bash
# 1. Implement illiquidity-adaptive MA
# See: scripts/strategies/adaptive_ma.py
# - Compute daily Amihud illiquidity
# - Map illiquidity → MA period (20-60 range)
# - Use longer MA in illiquid markets (less noise)

# 2. Backtest
python scripts/strategies/adaptive_ma.py \
    --symbol BTC \
    --adaptation-method illiquidity \
    --min-period 30 \
    --max-period 60

# Expected improvement:
# - Better adaptation to changing market regimes
# - Sharpe +0.05 to +0.15
```

**Deliverable:** Adaptive strategy that adjusts to market conditions

---

### Week 4: Liquidation Data Integration

**Goal:** Enhance entry timing with liquidation cascade detection

```bash
# 1. Fetch liquidation data from Hyperliquid
python scripts/data/fetch_liquidations.py \
    --symbols BTC ETH \
    --start-date 2025-01-01 \
    --output ./data/liquidations/

# 2. Aggregate to daily
python scripts/features/liquidation_features.py \
    --input ./data/liquidations/ \
    --output ./data/daily_liquidations/

# 3. Backtest with liquidation enhancement
python scripts/strategies/liquidation_enhanced_ma.py \
    --symbol BTC \
    --ma-period 44 \
    --liquidation-data ./data/daily_liquidations/BTC.parquet

# Expected improvement:
# - Better entry timing
# - Sharpe +0.05 to +0.1
```

**Deliverable:** Full strategy with liquidation cascade detection

---

## What You Get (After 4 Weeks)

### Strategy Components

1. ✅ **Base:** MA crossover (proven to work)
2. ✅ **Filter:** Regime detection (avoid ranging markets)
3. ✅ **Adapt:** Illiquidity-based MA period (adjust to market)
4. ✅ **Enhance:** Liquidation cascade timing (better entries)

### Expected Performance (Backtest)

- **Sharpe ratio:** 0.7-1.2 (after costs)
- **Max drawdown:** 15-25%
- **Win rate:** 55-60%
- **Trades/year:** 20-40
- **Annual txn costs:** <0.1%

### Ready for NautilusTrader Integration

**Week 5-6:** Integrate with NautilusTrader
**Week 7-8:** Paper trading on Hyperliquid testnet
**Week 9+:** Live trading with small capital ($1K-5K)

---

## Feature Usage: Before vs After

### Before (Current State)

```
183 tick-level features
  ↓
Sitting unused in Parquet files
  ↓
No trading strategy
```

### After (4 Weeks)

```
183 tick-level features
  ↓
Daily aggregation (15 features)
  ├─ Volume imbalance → Accumulation/distribution detection
  ├─ Amihud illiquidity → Adaptive MA period
  ├─ Range position → Regime filter
  └─ Liquidation data → Entry timing
  ↓
Daily MA crossover signals
  ↓
NautilusTrader execution
  ↓
Hyperliquid (self-custody)
  ↓
Profitable trading
```

---

## Files to Create (Implementation Checklist)

### Data Pipeline
- [ ] `scripts/data/aggregate_to_daily.py` - Tick → Daily aggregation
- [ ] `scripts/data/fetch_liquidations.py` - Get liquidation data from Hyperliquid API

### Features
- [ ] `scripts/features/daily_regime_features.py` - Accumulation/distribution/trending
- [ ] `scripts/features/liquidation_features.py` - Daily liquidation metrics

### Strategies
- [ ] `scripts/strategies/simple_ma_crossover.py` - Base MA44/MA33
- [ ] `scripts/strategies/regime_aware_ma.py` - + regime filter
- [ ] `scripts/strategies/adaptive_ma.py` - + illiquidity adaptation
- [ ] `scripts/strategies/liquidation_enhanced_ma.py` - + liquidation timing

### Analysis
- [ ] `scripts/analysis/compare_strategies.py` - Backtest comparison tool
- [ ] `scripts/analysis/time_horizon_test.py` - Validate time horizon effect

### NautilusTrader Integration
- [ ] `integrations/nautilus/hyperliquid_adapter.py` - Hyperliquid execution client
- [ ] `integrations/nautilus/ma_crossover_strategy.py` - Strategy in Nautilus framework
- [ ] `backtests/test_ma_nautilus.py` - Backtest with Nautilus
- [ ] `config/hyperliquid_config.toml` - Live trading configuration

---

## Makefile Targets to Add

```makefile
# Daily data pipeline
aggregate_to_daily:
	python scripts/data/aggregate_to_daily.py \
		--input-dir $(DATA_DIR) \
		--output-dir ./data/daily \
		--symbols $(SYMBOLS)

# Strategy backtests
backtest_simple_ma:
	python scripts/strategies/simple_ma_crossover.py \
		--symbol $(SYMBOL) \
		--ma-period $(MA_PERIOD) \
		--data ./data/daily/$(SYMBOL).parquet \
		--output ./output/backtest_simple_ma_$(SYMBOL).json

backtest_regime_aware:
	python scripts/strategies/regime_aware_ma.py \
		--symbol $(SYMBOL) \
		--ma-period $(MA_PERIOD) \
		--data ./data/daily/$(SYMBOL).parquet \
		--output ./output/backtest_regime_$(SYMBOL).json

backtest_adaptive_ma:
	python scripts/strategies/adaptive_ma.py \
		--symbol $(SYMBOL) \
		--adaptation-method $(METHOD) \
		--data ./data/daily/$(SYMBOL).parquet \
		--output ./output/backtest_adaptive_$(SYMBOL).json

# Compare all strategies
compare_strategies:
	python scripts/analysis/compare_strategies.py \
		--strategies ./output/backtest_*.json \
		--output ./output/strategy_comparison.html

# NautilusTrader
nautilus_backtest:
	python backtests/test_ma_nautilus.py \
		--config config/hyperliquid_config.toml \
		--start $(START_DATE) \
		--end $(END_DATE)

nautilus_paper_trade:
	python integrations/nautilus/run_live.py \
		--config config/hyperliquid_config.toml \
		--testnet

nautilus_live:
	python integrations/nautilus/run_live.py \
		--config config/hyperliquid_config.toml
```

---

## Risk Management (Critical)

### Before Going Live

1. **Backtest thoroughly** (minimum 2 years data)
2. **Walk-forward validation** (6-month train, 1-month test, roll forward)
3. **Paper trade 4-8 weeks** (validate execution, slippage, real-time behavior)
4. **Start tiny** ($1K-5K)

### Position Sizing

```python
# Example position sizing logic
def calculate_position_size(
    capital: float,
    risk_per_trade: float = 0.02,  # 2% risk per trade
    atr: float,  # Average True Range (volatility)
) -> float:
    """
    Size position based on volatility.

    Risk 2% of capital per trade.
    """
    risk_amount = capital * risk_per_trade
    position_size = risk_amount / (2 * atr)  # 2× ATR stop loss
    return position_size
```

### Stop Losses

- **Technical:** 2× ATR (Average True Range)
- **Time-based:** Exit after 30 days if no MA signal
- **Portfolio:** Max 5% daily loss

### Diversification

- Trade both BTC and ETH (correlation ~0.7, some diversification)
- Consider adding SOL, other majors later
- Don't put >50% capital in crypto (high volatility)

---

## Summary

**You have:**
- ✅ Infrastructure (world-class)
- ✅ Data (2+ weeks tick-level)
- ✅ Proven idea (MA44/MA33)
- ✅ Enhancement ideas (regime, illiquidity, liquidations)

**You need:**
- ⚠️ Daily aggregation pipeline (1 day to build)
- ⚠️ Simple backtest (1 day to validate)
- ⚠️ Enhancements (2-3 weeks)
- ⚠️ NautilusTrader integration (1-2 weeks)
- ⚠️ Paper trading (4-8 weeks)

**Timeline to live trading:** 8-12 weeks (if backtests validate)

**Start with Week 1 (daily aggregation + simple MA backtest). Prove it works. Then enhance incrementally.**

Your infrastructure isn't wasted - just needs to operate at daily scale, not tick scale.
