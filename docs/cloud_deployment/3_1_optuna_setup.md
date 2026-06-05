# 3.1 Optuna Setup

## Status: NOT_STARTED

## Goal

Integrate Optuna as the optimization framework to replace random/grid search
from Tier 2 with intelligent Bayesian optimization (CMA-ES / TPE).

## Prerequisites

- Tier 2 complete (evaluator works, config generator works)

## Why Optuna

| Feature | Benefit for NAT |
|---------|----------------|
| CMA-ES sampler | Handles correlated 35D continuous space with covariance adaptation |
| TPE sampler | Good for mixed continuous/categorical (ensemble method) |
| Pruning | Kills bad trials early (saves 50%+ compute) |
| Multi-objective (NSGA-II) | Simultaneous Sharpe vs drawdown optimization |
| PostgreSQL backend | Multiple machines share one study |
| Dashboard | Live parameter importance, Pareto fronts, optimization history |
| Python-native | Direct integration with existing scripts |

### CMA-ES vs PSO for NAT's Parameter Space

**CMA-ES wins** for this use case:
- Rotation-invariant: finds optima along any axis combination
- Adapts covariance matrix: learns parameter correlations automatically
- Better convergence in 35D continuous space
- PSO particles can get stuck in local optima at high dimensions

## Implementation

### Dependencies

```bash
pip install optuna optuna-dashboard psycopg2-binary
```

### Study Creation

**New file:** `scripts/swarm/optuna_optimizer.py`

```python
import optuna
from optuna.samplers import CmaEsSampler, TPESampler

class NATOptimizer:
    def __init__(self, study_name: str, storage: str, sampler: str = "cma"):
        sampler_obj = {
            "cma": CmaEsSampler(seed=42),
            "tpe": TPESampler(seed=42),
        }[sampler]

        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler_obj,
            directions=["maximize", "minimize", "maximize"],
            # Sharpe (max), Drawdown (min), IC (max)
            load_if_exists=True,
        )

    def suggest_config(self, trial: optuna.Trial) -> dict:
        """Map Optuna trial → NAT algorithm config."""
        config = {}
        # Algorithm thresholds
        config["jump_z_threshold"] = trial.suggest_float(
            "jump_z_threshold", 2.0, 5.0)
        config["jump_window"] = trial.suggest_int(
            "jump_window", 50, 500)
        config["sprt_upper"] = trial.suggest_float(
            "sprt_upper", 1.0, 5.0)
        # ... (35 params total)

        # Categorical
        config["ensemble_method"] = trial.suggest_categorical(
            "ensemble_method",
            ["equal_weight", "ic_weight", "regime_switch"])

        return config

    def optimize(self, n_trials: int, n_workers: int = 8):
        """Run optimization with parallel workers."""
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_workers,
            show_progress_bar=True,
        )
```

### PostgreSQL Backend

Optuna uses a PostgreSQL URL for distributed storage:

```python
storage = "postgresql://nat:password@localhost:5432/optuna"
```

Multiple machines can connect to the same study and pull trials concurrently.
Optuna handles locking and deduplication internally.

**docker-compose addition:**

```yaml
optuna-dashboard:
  image: ghcr.io/optuna/optuna-dashboard:latest
  ports:
    - "8070:8080"
  environment:
    - OPTUNA_STORAGE=postgresql://nat:${POSTGRES_PASSWORD}@postgres:5432/optuna
  depends_on:
    - postgres
```

### CLI Integration

```python
# nat evolve start --study my_study --trials 5000 --workers 8 --sampler cma
# nat evolve status --study my_study
# nat evolve best --study my_study --top 5
# nat evolve pareto --study my_study
# nat evolve dashboard  # opens optuna-dashboard
```

### Pruning

Optuna can prune unpromising trials before they complete:

```python
from optuna.pruners import MedianPruner

pruner = MedianPruner(n_startup_trials=50, n_warmup_steps=5)

# Inside objective function:
for step, partial_fitness in enumerate(evaluate_incremental(config)):
    trial.report(partial_fitness["sharpe"], step)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

This is powerful when evaluations process data day-by-day: if a config
performs poorly on days 1-5, skip days 6-30 entirely.

## Verification

```bash
# Create study
python scripts/swarm/optuna_optimizer.py \
  --study test_study --trials 100 --workers 4

# Check results
nat evolve status --study test_study
# Expected: 100 trials, best Sharpe, convergence info

# View dashboard
nat evolve dashboard
# Opens http://localhost:8070 with Optuna UI
```

## Files Created

- `scripts/swarm/optuna_optimizer.py`
- `docker-compose.yml` — optuna-dashboard service
- `nat` — evolve subcommands
