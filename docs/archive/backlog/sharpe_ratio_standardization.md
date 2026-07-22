# Sharpe Ratio Standardization

**Priority:** High — affects all performance comparisons, gate decisions (G6, G7), and deployment readiness scoring.

**Branch:** `feat/sharpe-normalization`

---

## Problem

Five distinct Sharpe formulas exist across the codebase. They disagree on annualization, standard deviation convention, and input semantics. Two strategies with identical return distributions can report Sharpe ratios differing by 3-10x depending on which code path evaluates them.

## Audit Findings

### Current implementations

| Location | Formula | Annualization | std | Input |
|----------|---------|---------------|-----|-------|
| `scripts/utils/metrics.py:6` | `mean/std * sqrt(ppy)` | parameterized (default 252) | population (N) | PnL |
| `scripts/backtest/engine.py:452-462` | `sharpe_intraday(pnl, trades_per_day)` | `sqrt(252 * trades_per_day)` **dynamic** | population (N) | per-trade PnL |
| `scripts/eamm/backtest.py:247-251` | `sharpe_intraday(pnl, trades_per_day)` | `sqrt(252 * trades_per_day)` **dynamic** | population (N) | per-trade PnL |
| `scripts/eamm/evaluate.py:196-200` | `sharpe_intraday(pnl, 864_000)` | `sqrt(252 * 864000)` = 46,477x | population (N) | per-bar PnL (10Hz) |
| `scripts/backtest/algorithm_strategy.py:130` | `sharpe_intraday(returns, 864_000)` | `sqrt(252 * 864000)` = 46,477x | population (N) | per-tick returns (10Hz) |
| `scripts/alpha/portfolio.py:246-249` | `sharpe_intraday(pnl, 96)` | `sqrt(252 * 96)` = 155.6x **hardcoded** | population (N) | 15-min bar PnL |
| `scripts/alpha/multi_freq.py:253-256` | `sharpe_intraday(pnl, 96)` | `sqrt(252 * 96)` = 155.6x **hardcoded** | population (N) | 15-min bar PnL |
| `scripts/alpha/paper_trader_daily.py` | `sharpe_daily(pnl)` | `sqrt(252)` = 15.87x | population (N) | daily PnL |
| `scripts/oos_validate.py` | `sharpe_daily(pnl)` | `sqrt(252)` = 15.87x | population (N) | daily PnL |
| `rust/ing-features/src/derived.rs:564-585` | `mean/std * sqrt(252)` | `sqrt(252)` = 15.87x | **sample (N-1)** | daily returns |

### Issues ranked by severity

#### 1. Trade-frequency annualization bias (HIGH)

`engine.py` and `eamm/backtest.py` compute `trades_per_day = len(pnls) / n_days` and annualize by `sqrt(252 * trades_per_day)`. This means:

- Strategy A: 10 trades/day -> annualization = `sqrt(2520)` = 50.2x
- Strategy B: 1 trade/day -> annualization = `sqrt(252)` = 15.9x

Same return distribution, 3.2x different Sharpe. This biases deployment decisions toward high-frequency strategies regardless of actual edge.

**Fix:** Aggregate per-trade PnL to daily PnL, then use `sharpe_daily()`.

#### 2. Population vs sample standard deviation (MEDIUM)

All Python paths use `np.std(pnl)` which divides by N (population). Rust uses N-1 (sample). For N=30 this is a 1.7% difference; for N=10 it's 5.4%.

**Fix:** Use `ddof=1` in `np.std()` consistently. Matches Rust and is statistically correct for sample Sharpe.

#### 3. Hardcoded bars_per_day (MEDIUM)

`portfolio.py` and `multi_freq.py` hardcode `bars_per_day=96` (15-min). `algorithm_strategy.py` and `eamm/evaluate.py` hardcode `864_000` (10Hz). If data frequency changes these will silently produce wrong Sharpes.

**Fix:** Infer `bars_per_day` from data timestamps or accept it as an explicit parameter from the caller.

#### 4. No risk-free rate (LOW for crypto)

All paths compute raw `mean/std`, not `(mean - rf)/std`. For crypto with near-zero holding costs this is acceptable, but the signature should support it.

**Fix:** Add `risk_free_rate=0.0` parameter to `annualized_sharpe()` for correctness.

---

## Implementation Plan

### T1. Fix `scripts/utils/metrics.py` (central utility)

```python
def annualized_sharpe(pnl, periods_per_year=252.0, risk_free_rate=0.0):
    if len(pnl) < 2:
        return 0.0
    arr = np.asarray(pnl, dtype=float)
    excess = arr - risk_free_rate / periods_per_year
    std = float(np.std(excess, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))
```

Key changes: `ddof=1`, optional risk-free subtraction, `np.asarray` guard.

### T2. Fix trade-frequency bias in `engine.py` and `eamm/backtest.py`

Replace per-trade annualization with daily aggregation:

```python
# Aggregate to daily PnL, then use sharpe_daily()
daily_pnl = aggregate_to_daily(pnls, timestamps)
sharpe_ratio = sharpe_daily(daily_pnl)
```

Add `aggregate_to_daily()` helper to `utils/metrics.py`.

### T3. Parameterize bars_per_day in `portfolio.py` and `multi_freq.py`

Replace hardcoded `96` with a parameter derived from data or config. Add assertion that validates the assumption.

### T4. Align Rust `compute_sharpe()` in `derived.rs`

Already uses sample std (N-1) which is correct. Add `risk_free_rate` parameter for parity. Confirm it's only called on daily returns.

### T5. Update hardcoded 10Hz paths

`algorithm_strategy.py:130` and `eamm/evaluate.py:196` — either aggregate to daily or infer frequency from timestamps.

---

## Verification

- [x] All Sharpe computations route through `utils/metrics.py` (grep confirms no inline formulas remain)
- [x] `ddof=1` used in all Python Sharpe paths
- [x] No hardcoded `bars_per_day` without documented justification
- [x] engine.py trade-frequency bias eliminated (aggregates to daily)
- [x] Existing tests pass: portfolio (19), multi_freq (19), adapter (14)
- [x] All _sharpe() helpers aggregate to daily before annualizing (no intraday multiplier inflation)
- [x] No callers of sharpe_intraday() remain (definition kept in metrics.py for backward compat)
- [ ] G6 and G7 gate decisions unchanged or improved on existing backtest data

## Known Remaining Deviations

`scripts/phase1_signal_test.py` lines 204, 486 use `mean/std * sqrt(N)` (t-statistic, not annualized Sharpe). Left unchanged — serves a different purpose (signal significance screening).

## Files Modified

- `scripts/utils/metrics.py` — ddof=1, risk_free_rate param, timeframe helpers, BARS_PER_DAY_10HZ constant (T1)
- `scripts/backtest/engine.py` — daily aggregation replaces trade-frequency annualization (T2)
- `scripts/eamm/backtest.py` — daily aggregation from equity curve, removed _sharpe() (T2)
- `scripts/alpha/portfolio.py` — bars_per_day threaded from timeframe param (T3)
- `scripts/alpha/multi_freq.py` — bars_per_day threaded from timeframe param (T3)
- `rust/ing-features/src/derived.rs` — fixed annualization bug: sqrt(252) was cancelling out (T4)
- `scripts/backtest/algorithm_strategy.py` — magic 864_000 replaced with BARS_PER_DAY_10HZ (T5)
- `scripts/eamm/evaluate.py` — magic 864_000 replaced with BARS_PER_DAY_10HZ (T5)
