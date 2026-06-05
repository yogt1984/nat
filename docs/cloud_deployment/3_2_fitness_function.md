# 3.2 Multi-Objective Fitness Function

## Status: NOT_STARTED

## Goal

Define the fitness function that Optuna optimizes. Must balance profitability
(Sharpe, IC) against risk (drawdown, turnover) using multi-objective optimization
with NSGA-II to produce a Pareto front of non-dominated solutions.

## Prerequisites

- [3.1 Optuna Setup](3_1_optuna_setup.md) — study and sampler configured
- [2.3 Evaluator Worker](2_3_evaluator_worker.md) — fitness computation works

## Multi-Objective Formulation

### Objectives (3 directions)

| Objective | Direction | Weight in Selection |
|-----------|-----------|-------------------|
| OOS Sharpe | maximize | Primary — profitability |
| Max Drawdown | minimize | Risk constraint |
| Mean IC | maximize | Signal quality |

### Hard Constraints (trial rejected if violated)

| Constraint | Threshold | Reason |
|------------|-----------|--------|
| Signal count | > 50/day | Prevents degenerate low-activity configs |
| Turnover | < 100/day | Prevents churning |
| OOS Sharpe | > 0 | Must be profitable out-of-sample |

### Implementation

**File:** `scripts/swarm/fitness.py`

```python
def objective(trial: optuna.Trial) -> tuple[float, float, float]:
    """Multi-objective fitness: (sharpe, drawdown, ic)."""
    # 1. Generate config from trial
    config = suggest_config(trial)

    # 2. Load Parquet data
    df = load_parquet(data_dir, symbol="BTC", days=30)

    # 3. Split: train (days 1-20) / test (days 21-30)
    train_df, test_df = walk_forward_split(df, train_days=20, test_days=10)

    # 4. Run algorithms on train set (for IC calibration)
    train_results = run_algorithms(train_df, config)

    # 5. Run algorithms on test set (OOS evaluation)
    test_results = run_algorithms(test_df, config)

    # 6. Compute OOS fitness
    sharpe = compute_sharpe(test_results)
    drawdown = compute_max_drawdown(test_results)
    ic = compute_mean_ic(test_results)
    signal_count = count_signals(test_results) / 10  # per day
    turnover = compute_turnover(test_results) / 10

    # 7. Check hard constraints
    if signal_count < 50:
        return (0.0, 1.0, 0.0)  # worst case
    if turnover > 100:
        return (0.0, 1.0, 0.0)

    # 8. Report for pruning
    trial.report(sharpe, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return (sharpe, drawdown, ic)
```

### Metric Computation Details

**Sharpe Ratio (annualized):**
```python
def compute_sharpe(results: pd.DataFrame) -> float:
    returns = results["pnl"].pct_change().dropna()
    if returns.std() == 0:
        return 0.0
    bars_per_year = 252 * 24 * 60 / bar_seconds
    return returns.mean() / returns.std() * np.sqrt(bars_per_year)
```

**Mean IC (rank correlation):**
```python
def compute_mean_ic(results: pd.DataFrame) -> float:
    ics = []
    for col in signal_columns:
        ic = results[col].corr(results["forward_return"], method="spearman")
        ics.append(ic)
    return np.nanmean(ics)
```

**Max Drawdown:**
```python
def compute_max_drawdown(results: pd.DataFrame) -> float:
    cumulative = (1 + results["pnl"].pct_change()).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return abs(drawdown.min())
```

## Pareto Front

With 3 objectives, NSGA-II produces a Pareto front of non-dominated solutions.
A config is non-dominated if no other config is better on ALL objectives.

```python
# After optimization
pareto_trials = study.best_trials  # Pareto-optimal set

for trial in pareto_trials:
    print(f"Sharpe={trial.values[0]:.2f}, "
          f"DD={trial.values[1]:.3f}, "
          f"IC={trial.values[2]:.4f}")
```

### Selection from Pareto Front

User picks the operating point based on risk appetite:
- **Aggressive:** highest Sharpe (accept higher drawdown)
- **Conservative:** lowest drawdown (accept lower Sharpe)
- **Balanced:** knee of the Pareto curve (maximum curvature point)

```python
def select_knee(pareto_trials: list) -> optuna.Trial:
    """Select the Pareto knee point using curvature."""
    # Normalize objectives to [0,1]
    # Find point with maximum curvature
    # Return corresponding trial
```

## Cost-Adjusted Metrics

All fitness metrics must subtract trading costs:

```python
# From config/costs.toml
TAKER_BPS = 3.5
MAKER_BPS = 0.2
SLIPPAGE_BPS = 1.0

def cost_adjusted_pnl(signals: pd.Series, prices: pd.Series) -> pd.Series:
    trades = signals.diff().abs()
    costs = trades * prices * (TAKER_BPS + SLIPPAGE_BPS) / 10000
    return raw_pnl - costs
```

## Verification

```bash
# Run single objective evaluation
python scripts/swarm/fitness.py --config config/algorithms.toml --symbol BTC
# Expected: prints (sharpe, drawdown, ic) tuple

# Run multi-objective study (small)
python scripts/swarm/optuna_optimizer.py \
  --study fitness_test --trials 50 --workers 4
# Expected: Pareto front with 5-15 non-dominated solutions
```

## Files Created

- `scripts/swarm/fitness.py`
- Integrates with `scripts/swarm/optuna_optimizer.py` (from 3.1)
