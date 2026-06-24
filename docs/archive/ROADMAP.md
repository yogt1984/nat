# Systematic Alpha Discovery Framework

## The Problem

You are hand-crafting strategies (funding reversion, MA crossover) and hoping they work.
This is O(1) hypothesis testing. With 191 features x 4 timeframes x 3 symbols = 2,292
candidate signals, hand-crafting is the wrong approach.

Phase 1 proved signal exists (+4.18% edge) but costs kill it (-0.45 bps/trade net).
This means: features contain information, but the way you're converting information
into trades is inefficient.

## The Solution

Build a pipeline that automatically:
1. Measures every feature's predictive power (alpha screening)
2. Filters by statistical significance with multiple testing correction
3. Filters by economic significance (can it pay costs?)
4. Combines survivors into a composite signal
5. Validates out-of-sample with walk-forward (you already built this)
6. Monitors for decay in production

This is what institutional quant shops do. The difference between this and what you have
is that this tests O(N) hypotheses simultaneously with proper correction, instead of
O(1) at a time.

## What You Already Have

- [x] 191-feature ingestor (Rust, 100ms emission)
- [x] Bar aggregation at 5min, 15min, 1h, 4h
- [x] Walk-forward validation with embargo (`scripts/backtest/walk_forward.py`)
- [x] Combinatorial purged CV (`combinatorial_purged_cv()`)
- [x] Deflated Sharpe ratio (`compute_deflated_sharpe()`)
- [x] GMM clustering with quality gates Q1/Q2/Q3
- [x] Cost model with maker/taker fees
- [x] Macro data fetcher (daily OHLCV + indicators)
- [x] Report generator, dashboard, CLI
- [x] Two strategy interfaces (continuous signal + entry/exit)
- [x] Backtest engine with trade tracking and equity curves

## What's Missing

1. Alpha screener (measure predictive power of every feature)
2. Feature combiner (turn ranked features into a composite signal)
3. Cost-aware position sizing (only trade when expected gain > cost)
4. Bridge between the two backtest systems (strategies/ uses pandas continuous signals; backtest/ uses Polars entry/exit conditions)
5. Regime-conditional screening (per-regime alpha measurement)
6. Multi-frequency signal integration (macro direction + micro timing with evidence-based weights)
7. Portfolio assembler (combine per-symbol strategies)
8. Live signal generator + paper trade logger

---

## Step 1: Alpha Screener

**File**: `scripts/alpha/screener.py`

**Input**: Aggregated bars (DataFrame with 191 features + timestamp + symbol)

**Output**: `reports/alpha_screen.json` — per-feature metrics ranked by strength

**Method**: For each feature `f` in each timeframe `tf` in {15min, 1h, 4h}, for each forward horizon `h` in {1h, 4h, 1d}:

```
1. Compute forward return:  r(t, h) = price(t+h) / price(t) - 1
2. Compute rank IC:         IC(t) = Spearman(rank(f), rank(r)) over 7-day rolling windows
3. IC mean:                 avg IC across all windows
4. IC std:                  std of IC across windows
5. IC information ratio:    IR = IC_mean / IC_std
6. IC autocorrelation:      corr(IC(t), IC(t-1))   — measures persistence
7. Turnover:                mean|f(t) - f(t-1)| / std(f)   — measures cost
8. t-statistic:             IC_mean / (IC_std / sqrt(n_windows))
9. Breakeven cost:          IC_mean * vol(r) / turnover   — minimum fee to stay profitable
```

After computing all features, apply Benjamini-Hochberg FDR correction across all t-statistics to get adjusted p-values.

**Quality Gate G1** (at least ONE of these must pass):
- 5+ features with adjusted p < 0.05 AND |IC_mean| > 0.015 AND breakeven > 2 bps
- 3+ features with adjusted p < 0.01 AND |IC_mean| > 0.025
- 1+ feature with |IC_mean| > 0.05 (very strong single signal)

**If G1 fails**:
- Try longer horizons (1d, 1w) — lower frequency = lower costs
- Try nonlinear transforms: rank(f), log(|f|+1), quantile(f)
- Try lagged features: f(t-1), f(t-2) — some features predict with delay
- Try feature interactions: f1 * f2, f1 / f2 for top pairs
- If still fails after all transforms: your features don't contain tradeable alpha at these costs. Options: (a) switch to maker-only execution (1 bps vs 3.5 bps), (b) add new features, (c) this market may not have exploitable inefficiency

**Implementation notes**:
- Run per-symbol (BTC, ETH, SOL separately) — they have different dynamics
- Use `cluster_pipeline.preprocess.aggregate_bars()` for bar construction
- Forward returns use `raw_midprice_mean` column
- Drop first `warmup` bars per session to avoid look-ahead from scaling
- Parallelize across symbols with multiprocessing

---

## Step 2: Feature Combination

**File**: `scripts/alpha/combiner.py`

**Input**: Top-N features from screener (N <= 20)

**Output**: Combined signal `z(t)` per bar, normalized to [-1, +1]

**Method**:
```
1. Standardize each feature: z_i(t) = (f_i(t) - rolling_mean(f_i, 30d)) / rolling_std(f_i, 30d)
2. Compute correlation matrix among selected features
3. Iteratively drop features with |corr| > 0.8, keeping higher IC
4. Combine:
   - Simple: z(t) = mean(z_i(t)) for surviving features, equal weight
   - Better: z(t) = sum(IC_i * z_i(t)) / sum(|IC_i|), IC-weighted
5. Normalize: z(t) = 2 * rank(z_combined(t)) / N - 1   (cross-sectional rank → [-1, +1])
```

Equal-weight is more robust out-of-sample than IC-weighted. Start with equal weight.
Only switch to IC-weighted if equal weight passes Step 4 AND IC-weighted improves OOS Sharpe.

**Quality Gate G2**:
- Combined IC > 0.8 * max(individual ICs) — combination shouldn't destroy signal
- Combined turnover < 2x average individual turnover — shouldn't churn
- Combined signal not > 0.9 correlated with any single feature — must add value

**If G2 fails**:
- Use fewer features (top 3-5 only)
- Switch from IC-weighted to equal weight
- Check for multicollinearity: if 3 entropy features all pass, pick the best one

---

## Step 3: Cost-Aware Position Sizing

**File**: `scripts/alpha/position.py`

**Input**: Combined signal z(t), cost model

**Output**: Position p(t) in [-1, +1]

This is the step Phase 1 skipped. The 4.18% edge was real but eaten by costs because
every signal change triggered a trade. The fix: only change position when the expected
gain from changing exceeds the cost of changing.

**Method**:
```
1. Estimate expected return of signal change:
   E[gain] = |z(t) - z(t-1)| * IC * vol(r) * sqrt(horizon)

2. Cost of changing:
   cost = |p_new - p_old| * cost_per_trade

3. Trade filter:
   Only update p(t) when E[gain] > cost * 1.5   (1.5x safety margin)
   Otherwise: p(t) = p(t-1)   (hold current position)

4. Position sizing (Kelly fraction):
   p(t) = clip(z(t) * IC * vol(r) / var(r), -1, +1)
   In practice: p(t) = clip(z(t) * scale_factor, -1, +1)
   where scale_factor is calibrated so that max position = 1.0

5. Ramp-up: first 30 days use 50% of max position
```

**Quality Gate G3**:
- Trade count drops by 50%+ vs unfiltered signal (proves filter is active)
- Net return INCREASES vs unfiltered (proves filter removes bad trades, not good ones)
- Mean holding time > 2 hours (not churning)

**If G3 fails**:
- Lower cost_per_trade by using maker-only orders
- Increase the 1.5x safety margin to 2x or 3x
- Use coarser bar size (1h instead of 15min)

---

## Step 4: Walk-Forward Validation

**File**: Already exists at `scripts/backtest/walk_forward.py`

**Bridge needed**: The `scripts/strategies/` system uses `compute_features() -> generate_signal()` returning a pandas Series in [-1, +1]. The `scripts/backtest/` system uses Polars DataFrames with `entry_condition() -> exit_condition()` returning booleans. You need an adapter.

**File**: `scripts/alpha/adapter.py`
```python
class ContinuousSignalAdapter:
    """Wraps a continuous signal [-1,+1] for the walk-forward engine."""

    def __init__(self, signal_series, threshold=0.3):
        self.signal = signal_series
        self.threshold = threshold

    def entry_condition(self, df):
        # Enter when |signal| > threshold
        return pl.Series(np.abs(self.signal) > self.threshold)

    def exit_condition(self, df):
        # Exit when signal crosses zero or drops below threshold/2
        return pl.Series(np.abs(self.signal) < self.threshold / 2)
```

**Validation protocol**:
```
1. walk_forward_validation(df, strategy, cost_model,
       n_splits=5, embargo_bars=600, oos_is_threshold=0.7)

2. combinatorial_purged_cv(df, strategy, cost_model,
       n_splits=5, n_test_splits=2, embargo_bars=600)

3. compute_deflated_sharpe(observed_sharpe,
       n_trials=N_features_screened_in_step_1)
   This is CRITICAL — it corrects for having tested N features.
   If you tested 191 features and pick the best, deflated Sharpe
   tells you the probability it's a fluke.
```

**Quality Gate G4** (ALL must pass):
- OOS Sharpe > 0.5 (economically meaningful)
- OOS/IS ratio > 0.7 (not overfitting)
- Deflated Sharpe p-value < 0.05 (survives multiple testing correction)
- Max drawdown < 5% (risk-acceptable)
- Minimum 30 trades in OOS (statistically meaningful)
- Profit factor > 1.2 (more winning $ than losing $)

**If G4 fails**:
- OOS Sharpe < 0.5 but > 0: increase horizon, reduce frequency
- OOS/IS ratio < 0.7: you're overfitting — use fewer features, simpler combination
- Deflated Sharpe fails: you tested too many things — preregister hypotheses next time
- Max DD > 5%: add drawdown control (reduce position 50% when DD > 2%)

---

## Step 5: Regime Conditioning

**File**: `scripts/alpha/regime_filter.py`

**Dependency**: Profiling pipeline must have produced valid regime labels (Q1 pass: silhouette > 0.25, ARI > 0.6)

**Method**:
```
1. Get regime labels from profiling: regime(t) in {0, 1, ..., k-1}
2. Re-run alpha screening (Step 1) WITHIN each regime separately
3. Compare: IC_within_regime vs IC_global
4. If IC_within > 1.5 * IC_global for regime R:
   → Use regime-specific feature weights when in regime R
5. If not:
   → Regimes don't help this feature. Use global weights.
6. Build regime-conditioned signal:
   z(t) = z_regime(R(t))(t)   where R(t) is the current regime
```

**Quality Gate G5**:
- At least 1 regime with IC > 1.5x global IC
- Regime-conditioned OOS Sharpe > global OOS Sharpe (from Step 4)
- If neither: skip regime conditioning entirely, use global signal

This step is OPTIONAL. Only do it if profiling produced real clusters (Q1+Q2 pass).
If clusters don't exist, regime conditioning adds noise, not signal.

---

## Step 6: Multi-Frequency Integration

**File**: `scripts/alpha/multi_freq.py`

**Input**: Micro signal (from Steps 1-5) + Macro signal (from `data/macro.py`)

**Output**: Composite signal incorporating both timescales

**Method**:
```
1. Macro filter (daily data):
   - SMA(50) > SMA(200) AND price > SMA(50): long_allowed = True
   - SMA(50) < SMA(200) AND price < SMA(50): short_allowed = True
   - Otherwise: flat_only = True

2. Micro sizing (intraday data):
   - When long_allowed: p(t) = max(0, micro_signal(t))
   - When short_allowed: p(t) = min(0, micro_signal(t))
   - When flat_only: p(t) = 0

3. Profit-sensitive exit:
   - If unrealized_pnl > 2 * cost_per_trade:
     Tighten VPIN exit threshold from 0.7 to 0.5
     (Lock in profits when microstructure shows danger)
```

This is your existing MacroMicro strategy, but with evidence-based micro weights from Step 1
instead of hand-tuned parameters.

**Quality Gate G6**:
- Composite Sharpe > max(macro_only_Sharpe, micro_only_Sharpe)
- Composite max DD < min(macro_only_DD, micro_only_DD)
- If composite doesn't beat both: use whichever is better alone

---

## Step 7: Portfolio Assembly

**File**: `scripts/alpha/portfolio.py`

Run Steps 1-6 independently for BTC, ETH, SOL.

**Method**:
```
1. Compute per-symbol equity curves from Step 4 OOS results
2. Compute daily return correlation matrix between symbols
3. Risk parity weighting: w_i = (1/vol_i) / sum(1/vol_j)
4. Adjust for correlation: if corr(BTC, ETH) > 0.8,
   reduce combined weight by 20%
5. Drawdown control: when portfolio DD > 2%,
   reduce all positions to 50% until DD recovers to 1%
6. Cross-symbol consensus: optionally, only trade when 2/3
   symbols agree on macro direction
```

**Quality Gate G7**:
- Portfolio Sharpe > max(individual Sharpes)
- Portfolio max DD < 80% of worst individual max DD
- If portfolio doesn't improve: trade best symbol only

---

## Step 8: Paper Trading

**Duration**: 14 days minimum

**File**: `scripts/alpha/paper_trader.py`

**Method**:
```
1. Every 15 minutes:
   a. Read latest parquet data
   b. Compute signal using the full pipeline
   c. Log: timestamp, signal, hypothetical entry/exit, price, features used
   d. Write to data/paper_trades/YYYY-MM-DD.json

2. Daily reconciliation:
   a. Compute paper PnL for the day
   b. Compare vs what backtest would have predicted for same period
   c. Log divergence

3. Monitor signal decay:
   a. Compute rolling 7-day IC of live signal vs realized returns
   b. Compare to backtest IC on training data
   c. Alert if live IC < 50% of backtest IC for 3 consecutive days
```

**Quality Gate G8**:
- Paper Sharpe within 2x of backtest Sharpe (expect some degradation)
- No single day loss > 2%
- Signal IC hasn't decayed > 50% from backtest
- All infrastructure (data feed, signal computation, logging) runs without errors for 14 days

**If G8 fails**:
- Paper much worse than backtest: execution model is wrong (likely fill assumptions)
- Single day > 2% loss: add tighter stop-loss, reduce position size
- IC decay: market regime shifted — re-run Step 1 with recent data
- Infrastructure failures: fix bugs, add monitoring

---

## Step 9: Live Deployment

**Scale-up schedule**:
```
Week 1-2:   1% of capital, maker orders only, observe
Week 3-4:   5% of capital if paper match holds
Month 2-3:  10% of capital if Sharpe holds
Month 4+:   Up to 25% of capital, never more
```

**Kill switches**:
- Daily loss > 1%: halt trading for 24 hours
- Weekly drawdown > 2%: halt and review pipeline
- Monthly drawdown > 5%: kill strategy, re-run full pipeline from Step 1
- IC drops below 0 for 5 consecutive days: halt

---

## Dependency Graph

```
Data Collection (7+ days)
    │
    ▼
Step 1: Alpha Screener ◄── CRITICAL PATH — everything depends on this
    │
    ├── Gate G1 PASS ──► Step 2: Feature Combination
    │                        │
    │                        ├── Gate G2 PASS ──► Step 3: Cost-Aware Sizing
    │                        │                        │
    │                        │                        ├── Gate G3 PASS ──► Step 4: Walk-Forward
    │                        │                        │                        │
    │                        │                        │                   Gate G4 PASS
    │                        │                        │                        │
    │                        │                        │                   ┌────┴────┐
    │                        │                        │                   ▼         ▼
    │                        │                        │              Step 5      Step 6
    │                        │                        │              (Regime)    (Multi-freq)
    │                        │                        │                   │         │
    │                        │                        │                   └────┬────┘
    │                        │                        │                        ▼
    │                        │                        │                   Step 7: Portfolio
    │                        │                        │                        │
    │                        │                        │                   Step 8: Paper Trade
    │                        │                        │                        │
    │                        │                        │                   Step 9: Live
    │                        │                        │
    │                        │                        └── Gate G3 FAIL ──► Lower costs or coarser bars
    │                        │
    │                        └── Gate G2 FAIL ──► Fewer features or equal weight
    │
    └── Gate G1 FAIL ──► Longer horizons, nonlinear transforms, or maker-only execution
```

## Timeline

| Week | Step | Deliverable | Gate |
|------|------|-------------|------|
| 1 | Step 1 | `alpha_screen.json` — ranked features with IC, p-values | G1 |
| 2 | Steps 2-3 | Combined signal with cost filter | G2, G3 |
| 2 | Step 4 | Walk-forward results, deflated Sharpe | G4 |
| 3 | Step 5 | Regime-conditioned signal (if profiling validates) | G5 |
| 3 | Step 6 | Multi-frequency composite signal | G6 |
| 4 | Step 7 | Portfolio weights and risk controls | G7 |
| 5-6 | Step 8 | Paper trading log and reconciliation | G8 |
| 7+ | Step 9 | Live with 1% capital | Ongoing |

## Critical Constraints

1. **Step 1 is the bottleneck.** If no features predict returns after FDR correction, nothing downstream matters. Do Step 1 first and decide based on results.

2. **Deflated Sharpe in Step 4 is non-negotiable.** You are testing 191+ features. Without multiple testing correction, you WILL find spurious signals. The deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) is already implemented in your codebase — use it.

3. **The two backtest systems must be bridged.** `scripts/strategies/` has the strategies. `scripts/backtest/` has walk-forward + CPCV + deflated Sharpe. These don't talk to each other. The adapter in Step 4 fixes this.

4. **Cost-aware sizing (Step 3) is what separates profitable from unprofitable.** Phase 1 had edge but lost money. The fix isn't better features — it's smarter position management that only trades when expected gain clears costs.

5. **Regimes (Step 5) are optional, not foundational.** Only use regime conditioning if profiling produces real clusters. If clusters are noise, regime conditioning adds noise.

6. **Never skip the paper trading step.** Backtests always look better than reality. 14 days of paper trading reveals: fill rate issues, latency problems, signal decay, infrastructure bugs.

## What NOT To Build

- Do not build a live trading engine until Step 8 passes
- Do not add more features until Step 1 proves existing features insufficient
- Do not optimize hyperparameters until Step 4 passes (no signal to optimize)
- Do not build ML models (LightGBM, neural nets) until the linear pipeline (Steps 1-4) either works or provably fails — ML adds complexity without adding signal if the base features lack IC
- Do not split into multiple repos — one pipeline, one source of truth
