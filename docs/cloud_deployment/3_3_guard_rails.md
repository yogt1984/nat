# 3.3 Guard Rails — Overfit Prevention

## Status: DONE

## Goal

Prevent the optimizer from finding configs that look great in-sample but fail
out-of-sample. This is the most critical component — without it, the entire
evolution system is worse than useless (it will confidently deploy overfit configs).

## Prerequisites

- [3.1 Optuna Setup](3_1_optuna_setup.md) — optimizer running
- [3.2 Fitness Function](3_2_fitness_function.md) — metrics defined

## Guard Rail 1: Walk-Forward Validation

**Never evaluate on data the optimizer has seen.**

### Rolling Window Protocol

```
Day 1       Day 20      Day 30
├───────────┤───────────┤
│  TRAIN    │   TEST    │
│  (in-sample) (OOS)   │

        Day 5       Day 25      Day 35
        ├───────────┤───────────┤
        │  TRAIN    │   TEST    │
        roll forward 5 days
```

Each trial uses a walk-forward split. The optimizer only sees OOS metrics.
Rolling the window forward every week ensures the optimizer can't memorize
a fixed test set.

### Implementation

```python
def walk_forward_split(df: pd.DataFrame, train_days: int = 20,
                       test_days: int = 10) -> tuple:
    """Split data into train/test with no overlap."""
    total_rows = len(df)
    train_frac = train_days / (train_days + test_days)
    split_idx = int(total_rows * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]

def rolling_walk_forward(df: pd.DataFrame, n_folds: int = 5) -> list:
    """Generate N walk-forward folds."""
    folds = []
    fold_size = len(df) // (n_folds + 1)
    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        test_end = min(train_end + fold_size, len(df))
        folds.append((df.iloc[:train_end], df.iloc[train_end:test_end]))
    return folds
```

## Guard Rail 2: Overfit Detection

### In-Sample vs OOS Ratio

```python
def overfit_score(is_sharpe: float, oos_sharpe: float) -> float:
    """Ratio of in-sample to out-of-sample performance.
    Healthy: < 2.0. Overfit: > 3.0. Extreme: > 5.0."""
    if oos_sharpe <= 0:
        return float('inf')
    return is_sharpe / oos_sharpe
```

### Penalty in Fitness Function

```python
def penalized_fitness(trial, config, train_df, test_df):
    is_results = run_algorithms(train_df, config)
    oos_results = run_algorithms(test_df, config)

    is_sharpe = compute_sharpe(is_results)
    oos_sharpe = compute_sharpe(oos_results)

    ratio = overfit_score(is_sharpe, oos_sharpe)

    if ratio > 3.0:
        # Penalize: reduce effective Sharpe
        penalty = (ratio - 2.0) * 0.1  # linear penalty above 2x
        oos_sharpe *= max(0.5, 1.0 - penalty)
        trial.set_user_attr("overfit_flag", True)
        trial.set_user_attr("overfit_ratio", ratio)

    return oos_sharpe
```

### Deflated Sharpe Ratio (DSR)

Account for multiple testing — the more configs you try, the more likely a
high Sharpe is due to chance:

```python
def deflated_sharpe(sharpe: float, n_trials: int,
                    sharpe_std: float = 1.0) -> float:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio.
    Adjusts for multiple testing bias."""
    from scipy.stats import norm
    expected_max = sharpe_std * (
        (1 - 0.5772) * norm.ppf(1 - 1/n_trials)
        + 0.5772 * norm.ppf(1 - 1/(n_trials * np.e))
    )
    return norm.cdf((sharpe - expected_max) / sharpe_std)
```

If DSR < 0.95, the Sharpe is likely noise. Flag the trial.

## Guard Rail 3: Stability Checks

### Parameter Sensitivity

A robust config should perform similarly with small parameter perturbations:

```python
def sensitivity_check(config: dict, best_sharpe: float,
                      perturbation: float = 0.05) -> float:
    """Perturb each param by ±5%, measure Sharpe drop."""
    drops = []
    for key, value in config.items():
        if isinstance(value, (int, float)):
            for direction in [-1, 1]:
                perturbed = copy.deepcopy(config)
                perturbed[key] = value * (1 + direction * perturbation)
                s = evaluate(perturbed)["sharpe"]
                drops.append(abs(s - best_sharpe) / best_sharpe)
    return np.mean(drops)  # < 0.1 = robust, > 0.3 = fragile
```

### Minimum Data Requirement

Reject trials evaluated on less than 14 days of data:

```python
MIN_EVAL_DAYS = 14
if eval_days < MIN_EVAL_DAYS:
    raise ValueError(f"Need >= {MIN_EVAL_DAYS} days, got {eval_days}")
```

### Signal Stationarity

Check that signal IC doesn't decay over the test period:

```python
def ic_stationarity(results: pd.DataFrame, n_windows: int = 5) -> bool:
    """Check IC is roughly stable across time windows."""
    window_size = len(results) // n_windows
    ics = []
    for i in range(n_windows):
        chunk = results.iloc[i*window_size:(i+1)*window_size]
        ic = chunk["signal"].corr(chunk["forward_return"], method="spearman")
        ics.append(ic)
    # Reject if IC drops by >50% from first to last window
    return ics[-1] > 0.5 * ics[0]
```

## Guard Rail Summary

| Guard Rail | What it Prevents | Threshold |
|------------|-----------------|-----------|
| Walk-forward OOS | Evaluating on seen data | Train/test never overlap |
| Overfit ratio | In-sample fishing | IS/OOS Sharpe ratio < 3.0 |
| Deflated Sharpe | Multiple testing bias | DSR > 0.95 |
| Sensitivity check | Fragile optima | Mean perturbation drop < 10% |
| Min data | Small-sample illusions | >= 14 days |
| IC stationarity | Signal decay | Last window IC > 50% of first |

## Verification

```bash
# Run study with guard rails enabled
python scripts/swarm/optuna_optimizer.py \
  --study guarded_test --trials 200 --guard-rails

# Check overfit flags
python -c "
import optuna
study = optuna.load_study('guarded_test', storage='...')
flagged = [t for t in study.trials if t.user_attrs.get('overfit_flag')]
print(f'{len(flagged)}/{len(study.trials)} flagged as overfit')
"
```

## Files Created

- `scripts/swarm/guard_rails.py` — all validation functions
- Integrates with `scripts/swarm/fitness.py` (from 3.2)
