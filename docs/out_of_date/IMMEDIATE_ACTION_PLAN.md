# Immediate Action Plan - Get Out of High Entropy State

**Current State:** Analysis paralysis, too many options, unclear priorities
**Goal:** Clear sequential path from chaos to working system
**Philosophy:** Validate hypothesis FIRST, build infrastructure SECOND

---

## The Problem: High Entropy Decision Space

You have too many good ideas competing for attention:
- Daily MA strategies (proven to work elsewhere)
- Agent-based evolution system (exciting engineering)
- Liquidity distribution analysis (novel insight)
- NautilusTrader integration (production-grade)
- Online learning MA (adaptive)
- Regime detection (sophisticated)

**All are good. But implementing all simultaneously = 0% completion.**

**This is the "research trap"** - perpetual planning, no execution, no results.

---

## The Solution: Sequential Validation

```
Phase 0: Validate Core Hypothesis (1 day)
  ↓
  Does MA strategy work on YOUR data?
  ├─ NO → Stop, rethink approach ❌
  └─ YES → Proceed to Phase 1 ✅

Phase 1: Simple Enhancement (1 week)
  ↓
  Can regime detection improve it?
  ├─ NO → Try liquidity features
  └─ YES → Quantify improvement

Phase 2: Systematic Search (2 weeks)
  ↓
  Build minimal agent system for parameter search
  ↓
  Find optimal MA periods, thresholds, filters

Phase 3: Production (4 weeks)
  ↓
  NautilusTrader integration, paper trading, live
```

**Each phase has clear success criteria. No phase starts until previous validates.**

---

## PHASE 0: Validate Core Hypothesis (DO THIS TODAY)

### Objective

**Prove or disprove:** "MA crossover beats buy-and-hold on BTC/ETH daily data"

**Time:** 4-6 hours
**Dependencies:** Your tick data + daily aggregation
**Deliverable:** Simple backtest results showing Sharpe ratio

---

### Step 0.1: Aggregate Tick Data to Daily (2 hours)

**If you haven't already:**

```bash
# Run daily aggregation
python scripts/data/aggregate_to_daily.py \
    --input-dir ./data/features \
    --output-dir ./data/daily \
    --symbols BTC ETH

# Verify output
ls -lh data/daily/
# Should see: BTC.parquet, ETH.parquet
```

**Check data quality:**

```python
import polars as pl

# Read BTC daily
btc = pl.read_parquet('data/daily/BTC.parquet')
print(f"BTC: {len(btc)} days, from {btc['timestamp'].min()} to {btc['timestamp'].max()}")

# Verify columns
print(btc.columns)
# Should have: timestamp, open, high, low, close, volume

# Check for gaps
btc = btc.sort('timestamp')
gaps = (btc['timestamp'].diff().dt.total_days() > 2).sum()
print(f"Date gaps: {gaps}")  # Should be 0 or very few
```

**Success criteria:**
- [ ] Have at least 6 months of daily data (180+ days)
- [ ] No major gaps (missing weeks)
- [ ] OHLCV columns present and valid

---

### Step 0.2: Implement Simplest Possible MA Backtest (1 hour)

**Create this file:**

```python
# scripts/backtest/validate_ma_hypothesis.py

import polars as pl
import numpy as np
import argparse
from pathlib import Path

def backtest_simple_ma(
    daily_data: pl.DataFrame,
    ma_period: int,
    transaction_cost: float = 0.0008  # 8 bps
) -> dict:
    """
    Simplest possible MA crossover backtest.

    Rules:
    - Buy when price > MA
    - Sell when price < MA
    - Execute at next day's close (realistic)
    - Transaction cost on every position change

    Returns metrics: Sharpe, total return, max drawdown, win rate
    """
    df = daily_data.sort('timestamp').clone()

    # Compute MA
    df = df.with_columns([
        pl.col('close').rolling_mean(ma_period).alias('ma')
    ])

    # Generate position (1 = long, 0 = flat)
    df = df.with_columns([
        (pl.col('close') > pl.col('ma')).cast(pl.Int8).alias('position')
    ])

    # Compute daily returns
    df = df.with_columns([
        pl.col('close').pct_change().alias('market_return')
    ])

    # Strategy returns (position from previous day)
    df = df.with_columns([
        (pl.col('market_return') * pl.col('position').shift(1)).alias('strategy_return')
    ])

    # Transaction costs (when position changes)
    df = df.with_columns([
        (pl.col('position').diff().abs() * transaction_cost).alias('txn_cost')
    ])

    # Net returns
    df = df.with_columns([
        (pl.col('strategy_return') - pl.col('txn_cost')).alias('net_return')
    ])

    # Drop nulls from MA warmup
    df = df.drop_nulls()

    if len(df) < ma_period:
        return {'error': 'Insufficient data'}

    # Compute metrics
    net_returns = df['net_return'].to_numpy()
    market_returns = df['market_return'].to_numpy()

    # Sharpe ratio (annualized)
    sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252)

    # Buy and hold Sharpe
    bh_sharpe = np.mean(market_returns) / np.std(market_returns) * np.sqrt(252)

    # Cumulative returns
    strategy_cum = (1 + net_returns).cumprod()
    market_cum = (1 + market_returns).cumprod()

    # Total return
    total_return = (strategy_cum[-1] - 1) * 100
    bh_return = (market_cum[-1] - 1) * 100

    # Max drawdown
    running_max = np.maximum.accumulate(strategy_cum)
    drawdown = (strategy_cum - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100

    # Win rate (only on trades, not hold days)
    trades = df.filter(pl.col('position').diff().abs() > 0)
    if len(trades) > 0:
        wins = (trades['net_return'] > 0).sum()
        win_rate = wins / len(trades) * 100
    else:
        win_rate = 0

    # Number of trades per year
    n_trades = (df['position'].diff().abs() > 0).sum()
    years = len(df) / 252
    trades_per_year = n_trades / years if years > 0 else 0

    return {
        'ma_period': ma_period,
        'sharpe_ratio': float(sharpe),
        'buy_hold_sharpe': float(bh_sharpe),
        'sharpe_improvement': float(sharpe - bh_sharpe),
        'total_return_pct': float(total_return),
        'buy_hold_return_pct': float(bh_return),
        'alpha_pct': float(total_return - bh_return),
        'max_drawdown_pct': float(max_drawdown),
        'win_rate_pct': float(win_rate),
        'trades_per_year': int(trades_per_year),
        'n_days': len(df),
    }


def main():
    parser = argparse.ArgumentParser(description='Validate MA hypothesis')
    parser.add_argument('--data', type=str, required=True, help='Path to daily parquet')
    parser.add_argument('--ma-period', type=int, default=44, help='MA period')
    parser.add_argument('--symbol', type=str, default='BTC', help='Symbol name')

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.data}...")
    df = pl.read_parquet(args.data)
    print(f"Loaded {len(df)} days")

    # Run backtest
    print(f"\nBacktesting MA{args.ma_period} on {args.symbol}...\n")
    results = backtest_simple_ma(df, args.ma_period)

    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return

    # Print results
    print("=" * 60)
    print(f"  MA{args.ma_period} Backtest Results ({args.symbol})")
    print("=" * 60)
    print(f"Period: {results['n_days']} days ({results['n_days']/252:.1f} years)")
    print()
    print(f"Strategy Sharpe:        {results['sharpe_ratio']:>8.3f}")
    print(f"Buy & Hold Sharpe:      {results['buy_hold_sharpe']:>8.3f}")
    print(f"Improvement:            {results['sharpe_improvement']:>8.3f}  {'✅' if results['sharpe_improvement'] > 0 else '❌'}")
    print()
    print(f"Strategy Return:        {results['total_return_pct']:>7.1f}%")
    print(f"Buy & Hold Return:      {results['buy_hold_return_pct']:>7.1f}%")
    print(f"Alpha:                  {results['alpha_pct']:>7.1f}%  {'✅' if results['alpha_pct'] > 0 else '❌'}")
    print()
    print(f"Max Drawdown:           {results['max_drawdown_pct']:>7.1f}%")
    print(f"Win Rate:               {results['win_rate_pct']:>7.1f}%")
    print(f"Trades/Year:            {results['trades_per_year']:>8}")
    print("=" * 60)

    # Decision
    print()
    if results['sharpe_ratio'] > 0.3 and results['alpha_pct'] > 0:
        print("✅ HYPOTHESIS VALIDATED: MA strategy beats buy-and-hold")
        print("   → Proceed to Phase 1 (enhancement)")
    elif results['sharpe_ratio'] > 0.3:
        print("⚠️  MARGINAL: MA strategy has positive Sharpe but no alpha")
        print("   → Enhancement needed to beat buy-and-hold")
    else:
        print("❌ HYPOTHESIS REJECTED: MA strategy does not work on this data")
        print("   → Rethink approach or try different parameters")

    return results


if __name__ == '__main__':
    main()
```

---

### Step 0.3: Run Validation (30 minutes)

```bash
# Test BTC with MA44
python scripts/backtest/validate_ma_hypothesis.py \
    --data data/daily/BTC.parquet \
    --ma-period 44 \
    --symbol BTC

# Test ETH with MA33
python scripts/backtest/validate_ma_hypothesis.py \
    --data data/daily/ETH.parquet \
    --ma-period 33 \
    --symbol ETH

# Try a few other MA periods to see if 44/33 are actually optimal
for ma in 20 30 44 50 60; do
    echo "Testing MA${ma}..."
    python scripts/backtest/validate_ma_hypothesis.py \
        --data data/daily/BTC.parquet \
        --ma-period $ma \
        --symbol BTC
done
```

---

### Step 0.4: Interpret Results (30 minutes)

**Scenario A: MA Works** ✅
```
Strategy Sharpe:        0.85
Buy & Hold Sharpe:      0.62
Improvement:            0.23  ✅
Alpha:                  15.3%  ✅
```

**Action:** Proceed to Phase 1 (enhancement)
**Confidence:** HIGH - you have a working base strategy

---

**Scenario B: MA Marginal** ⚠️
```
Strategy Sharpe:        0.42
Buy & Hold Sharpe:      0.58
Improvement:           -0.16  ❌
Alpha:                  -5.2%  ❌
```

**Action:** Try enhancements immediately (regime filter, liquidity features)
**Confidence:** MEDIUM - base doesn't work but enhancements might help

---

**Scenario C: MA Fails** ❌
```
Strategy Sharpe:       -0.15
Buy & Hold Sharpe:      0.58
Improvement:           -0.73  ❌
Alpha:                 -45.8%  ❌
```

**Action:** STOP. Rethink entire approach.
**Options:**
1. Your data period is unusual (check date range)
2. MA doesn't work in crypto anymore (market changed)
3. Need different strategy family (mean reversion, momentum, etc.)

**DO NOT build agent system if base strategy fails.**

---

## PHASE 1: Simple Enhancement (START ONLY IF PHASE 0 PASSES)

### Objective

Improve base MA strategy with 1-2 simple enhancements.

**Time:** 1 week
**Goal:** Sharpe improvement of +0.1 to +0.3

---

### Enhancement Option A: Regime Filter (Simplest)

**Hypothesis:** MA works in trending markets, fails in ranging markets.

**Implementation:**

```python
# Add to backtest script

def compute_regime(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """
    Simple regime detection.

    Trending: High ADX-like metric (range utilization)
    Ranging: Low range utilization
    """
    df = df.with_columns([
        # Range utilization (how much of 20-day range did we move?)
        ((pl.col('close') - pl.col('close').shift(lookback)).abs() /
         (pl.col('high').rolling_max(lookback) - pl.col('low').rolling_min(lookback))).alias('range_utilization')
    ])

    # Trending if range utilization > 0.4
    df = df.with_columns([
        (pl.col('range_utilization') > 0.4).alias('is_trending')
    ])

    return df

# In backtest:
df = compute_regime(df)

# Only trade in trending regime
df = df.with_columns([
    pl.when(pl.col('is_trending'))
      .then(pl.col('position'))
      .otherwise(0)
      .alias('position_filtered')
])

# Use position_filtered instead of position for returns
```

**Test:**
```bash
python scripts/backtest/validate_ma_with_regime.py \
    --data data/daily/BTC.parquet \
    --ma-period 44
```

**Success criteria:**
- [ ] Sharpe improves by >0.1
- [ ] Win rate improves
- [ ] Max drawdown reduces

---

### Enhancement Option B: Volume Filter

**Hypothesis:** MA works better when volume confirms the trend.

```python
# Add volume confirmation
df = df.with_columns([
    (pl.col('volume') / pl.col('volume').rolling_mean(20)).alias('volume_ratio')
])

# Only enter long if volume > 1.2× average
df = df.with_columns([
    pl.when((pl.col('position') == 1) & (pl.col('volume_ratio') > 1.2))
      .then(1)
      .when(pl.col('position') == 0)
      .then(0)
      .otherwise(pl.col('position'))
      .alias('position_filtered')
])
```

---

### Enhancement Option C: Liquidity-Based Entry

**Hypothesis:** Enter near support (high liquidity) for better risk/reward.

(Requires liquidity distribution features from your observation #2)

```python
# If you have support/resistance levels
df = df.with_columns([
    ((pl.col('close') - pl.col('support_level')) / pl.col('close') * 100).alias('distance_to_support_pct')
])

# Only buy if within 3% of support
df = df.with_columns([
    pl.when((pl.col('position') == 1) & (pl.col('distance_to_support_pct') < 3.0))
      .then(1)
      .otherwise(0)
      .alias('position_filtered')
])
```

---

**Pick ONE enhancement, test it, measure improvement.**

**If improvement > 0.1 Sharpe → Keep it and try adding a second**
**If improvement < 0.1 Sharpe → Try different enhancement**

---

## PHASE 2: Systematic Parameter Search (START ONLY IF PHASE 1 WORKS)

### Objective

Find optimal parameters systematically (but without full agent system yet).

**Time:** 2 weeks
**Approach:** Grid search first, then minimal evolution

---

### Step 2.1: Grid Search (3 days)

**Test multiple parameter combinations:**

```python
# scripts/backtest/parameter_grid_search.py

import itertools
import pandas as pd

# Define parameter grid
param_grid = {
    'ma_period': [20, 25, 30, 35, 40, 44, 50, 60, 70, 80],
    'regime_threshold': [0.3, 0.4, 0.5, 0.6],
    'volume_threshold': [1.0, 1.2, 1.5],
    'min_trend_strength': [0.0, 0.2, 0.4],
}

results = []

# Try all combinations
for ma, regime, volume, trend in itertools.product(
    param_grid['ma_period'],
    param_grid['regime_threshold'],
    param_grid['volume_threshold'],
    param_grid['min_trend_strength']
):
    # Run backtest with these parameters
    metrics = backtest_with_params(
        daily_data,
        ma_period=ma,
        regime_threshold=regime,
        volume_threshold=volume,
        min_trend_strength=trend
    )

    results.append({
        'ma_period': ma,
        'regime_threshold': regime,
        'volume_threshold': volume,
        'min_trend_strength': trend,
        **metrics
    })

# Find best by Sharpe
results_df = pd.DataFrame(results)
best = results_df.nlargest(10, 'sharpe_ratio')
print(best)
```

**This tests:**
- 10 MA periods × 4 regime thresholds × 3 volume thresholds × 3 trend strengths
- = 360 combinations
- @ 0.5 seconds each = 3 minutes total

**Success criteria:**
- [ ] Find parameters with Sharpe > 0.6
- [ ] Verify best parameters make intuitive sense
- [ ] Check for overfitting (OOS validation)

---

### Step 2.2: Walk-Forward Validation (2 days)

**Critical:** Don't overfit to one period.

```python
# scripts/backtest/walk_forward.py

def walk_forward_validation(daily_data, param_grid, train_months=6, test_months=1):
    """
    Walk-forward validation.

    Process:
    1. Train on 6 months (find best parameters)
    2. Test on next 1 month (out-of-sample)
    3. Roll forward by 1 month
    4. Repeat
    """
    results = []

    for i in range(0, len(daily_data) - train_months*30, test_months*30):
        # Train period
        train_data = daily_data[i:i+train_months*30]

        # Find best parameters on train
        best_params = grid_search(train_data, param_grid)

        # Test period
        test_data = daily_data[i+train_months*30:i+(train_months+test_months)*30]

        # Evaluate on test
        test_metrics = backtest_with_params(test_data, **best_params)

        results.append({
            'train_start': train_data['timestamp'].min(),
            'test_start': test_data['timestamp'].min(),
            'test_sharpe': test_metrics['sharpe_ratio'],
            **best_params
        })

    # Aggregate OOS performance
    oos_sharpe = np.mean([r['test_sharpe'] for r in results])
    return oos_sharpe, results
```

**Success criteria:**
- [ ] OOS Sharpe > 0.4
- [ ] OOS/IS ratio > 0.6 (OOS not much worse than in-sample)
- [ ] Consistent parameters across periods (stable)

---

### Step 2.3: Minimal Evolution (1 week)

**Only if grid search works, try simple evolution:**

```python
# scripts/evolution/simple_evolution.py

import random
import numpy as np

class SimpleStrategy:
    """Simple parameterized strategy."""

    def __init__(self, ma_period, regime_threshold, volume_threshold):
        self.ma_period = ma_period
        self.regime_threshold = regime_threshold
        self.volume_threshold = volume_threshold
        self.fitness = 0.0  # Sharpe ratio

    def mutate(self):
        """Create mutated copy."""
        new = SimpleStrategy(
            ma_period=int(np.clip(self.ma_period + random.randint(-5, 5), 20, 80)),
            regime_threshold=np.clip(self.regime_threshold + random.uniform(-0.1, 0.1), 0.1, 0.9),
            volume_threshold=np.clip(self.volume_threshold + random.uniform(-0.2, 0.2), 0.8, 2.0)
        )
        return new

# Simple evolution
population = [SimpleStrategy(44, 0.4, 1.2) for _ in range(20)]

for generation in range(10):
    # Evaluate all
    for strategy in population:
        metrics = backtest_with_params(daily_data,
                                      ma_period=strategy.ma_period,
                                      regime_threshold=strategy.regime_threshold,
                                      volume_threshold=strategy.volume_threshold)
        strategy.fitness = metrics['sharpe_ratio']

    # Sort by fitness
    population.sort(key=lambda s: s.fitness, reverse=True)

    # Keep top 5, mutate to create next 15
    next_gen = population[:5]  # Elitism
    for _ in range(15):
        parent = random.choice(population[:10])
        child = parent.mutate()
        next_gen.append(child)

    population = next_gen

    print(f"Gen {generation}: Best Sharpe = {population[0].fitness:.3f}")

# Best strategy
best = population[0]
print(f"Best parameters: MA={best.ma_period}, Regime={best.regime_threshold:.2f}, Volume={best.volume_threshold:.2f}")
```

**This is 100 lines, runs in 5 minutes, tests if evolution helps.**

**Success criteria:**
- [ ] Evolution improves on grid search
- [ ] Best strategy has Sharpe > 0.7
- [ ] Parameters make sense

---

## PHASE 3: Production (START ONLY IF PHASE 2 WORKS)

### NautilusTrader Integration (2 weeks)

**Only proceed if you have:**
- ✅ Validated strategy (Sharpe > 0.5)
- ✅ Robust parameters (walk-forward validated)
- ✅ Understands why it works

**Steps:**
1. Install NautilusTrader
2. Create Hyperliquid adapter
3. Backtest in Nautilus (compare to your simple backtester)
4. Paper trade for 4 weeks
5. If paper trading matches backtest → Go live with $1K-5K

---

## DECISION TREE SUMMARY

```
TODAY:
  ↓
Run Phase 0 (validate MA hypothesis)
  ↓
  ├─ Sharpe > 0.5? → GREAT, proceed to Phase 1
  ├─ Sharpe 0.3-0.5? → MARGINAL, try enhancements in Phase 1
  └─ Sharpe < 0.3? → STOP, rethink approach

Week 1-2: Phase 1 (if Phase 0 passed)
  ↓
Add 1-2 simple enhancements
  ↓
  ├─ Improvement > 0.1 Sharpe? → Proceed to Phase 2
  └─ No improvement? → Try different enhancement or stop

Week 3-4: Phase 2 (if Phase 1 worked)
  ↓
Grid search + Walk-forward validation
  ↓
  ├─ OOS Sharpe > 0.4? → Proceed to Phase 3
  └─ OOS Sharpe < 0.4? → Overfitting, back to Phase 1

Week 5-8: Phase 3 (if Phase 2 validated)
  ↓
NautilusTrader + Paper trading
  ↓
  ├─ Paper matches backtest? → Live trading
  └─ Paper underperforms? → More research
```

---

## WHAT TO BUILD WHEN

| System Component | Build When | Why |
|------------------|------------|-----|
| **Daily aggregation** | Phase 0 (NOW) | Need data to backtest |
| **Simple MA backtest** | Phase 0 (NOW) | Validate hypothesis |
| **Regime detection** | Phase 1 (if Phase 0 works) | First enhancement |
| **Grid search** | Phase 2 (if enhanced strategy works) | Parameter optimization |
| **Walk-forward validation** | Phase 2 | Prevent overfitting |
| **Simple evolution** | Phase 2 (optional) | Test if evolution helps |
| **Full agent system** | NEVER (or Month 4+) | Overkill for single strategy |
| **Web dashboard** | Phase 3 (if deploying live) | Monitor live trading |
| **NautilusTrader** | Phase 3 (if validated) | Production execution |

---

## THE TRAP TO AVOID

**Don't build the agent system yet.** Here's why:

**You're tempted to build:**
- Multi-agent evolution system (2 months)
- Web dashboard (2 weeks)
- PostgreSQL + Redis + Celery (1 week)
- Genetic algorithms (1 week)

**Total: 3 months of engineering**

**But you don't know:**
- Does MA even work on your data? (Unknown)
- What parameters matter? (Unknown)
- Is evolution better than grid search? (Unknown)

**If MA doesn't work (30% chance), you've wasted 3 months.**

**Better approach:**
1. Validate in 1 day (Phase 0)
2. Enhance in 1 week (Phase 1)
3. Optimize in 2 weeks (Phase 2)
4. THEN decide if agent system is worth building

**Maybe you discover:**
- Grid search finds Sharpe 0.85
- Evolution finds Sharpe 0.87
- Improvement: 0.02 Sharpe
- Agent system cost: 2 months

**Not worth it. Ship the grid search strategy and start trading.**

---

## IMMEDIATE NEXT ACTIONS (TODAY)

```bash
# 1. Create daily aggregation script (if not exists)
# See code above

# 2. Run aggregation
make aggregate_to_daily  # or run script directly

# 3. Create validation script
# Copy code from Step 0.2 above

# 4. Run validation
python scripts/backtest/validate_ma_hypothesis.py \
    --data data/daily/BTC.parquet \
    --ma-period 44 \
    --symbol BTC

# 5. Based on results, decide:
# - Proceed to Phase 1? (if Sharpe > 0.3)
# - Rethink approach? (if Sharpe < 0.3)
```

---

## SUMMARY

**You asked:** "What should I implement now?"

**Answer:**

**TODAY:** Run Phase 0 validation (4-6 hours)
- Aggregate to daily
- Backtest simple MA
- Measure Sharpe ratio

**IF Phase 0 passes:**
- Week 1-2: Add regime filter (Phase 1)
- Week 3-4: Grid search + walk-forward (Phase 2)
- Week 5-8: NautilusTrader + paper trading (Phase 3)

**IF Phase 0 fails:**
- Rethink approach entirely
- Try mean reversion instead
- Try different symbols
- Consider lower frequency (weekly)

**DO NOT build agent system until you have a working strategy to evolve.**

**Infrastructure does not create alpha. Validate first, automate later.**
