# 2.3 Evaluator Worker

## Status: NOT_STARTED

## Goal

A lightweight Python process that reads Parquet data, runs all algorithms with
a given config, and writes fitness results to a shared database.

## Prerequisites

- [2.1 Shared Ingestor](2_1_shared_ingestor.md) — Parquet data available
- [2.2 Config Generator](2_2_config_generator.md) — config TOML files generated

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ config_i.toml│────►│  Evaluator   │────►│  Results DB  │
└──────────────┘     │              │     └──────────────┘
                     │ 1. Load data │
┌──────────────┐     │ 2. Run algos │
│ *.parquet    │────►│ 3. Compute   │
│ (read-only)  │     │    fitness   │
└──────────────┘     │ 4. Write row │
                     └──────────────┘
```

## Implementation

**New file:** `scripts/swarm/evaluator.py`

```python
class Evaluator:
    def __init__(self, config_path: str, data_dir: str, db_path: str):
        self.config = toml.load(config_path)
        self.data_dir = data_dir
        self.db = sqlite3.connect(db_path)

    def load_data(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Read Parquet files for evaluation window."""
        # Reuse existing scripts/analysis/ patterns

    def run_algorithms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all enabled algorithms via registry."""
        from scripts.algorithms.registry import get_registered
        # For each algorithm: call run_batch(df)
        # Merge outputs into single DataFrame

    def compute_fitness(self, results: pd.DataFrame) -> dict:
        """Compute fitness metrics from algorithm outputs."""
        return {
            "sharpe": self._sharpe(results),
            "mean_ic": self._mean_ic(results),
            "max_drawdown": self._max_drawdown(results),
            "signal_count": self._signal_count(results),
            "turnover": self._turnover(results),
        }

    def evaluate(self, symbol: str = "BTC", hours: int = 24) -> dict:
        """Full evaluation pipeline: load → run → fitness → store."""
        df = self.load_data(symbol, hours)
        results = self.run_algorithms(df)
        fitness = self.compute_fitness(results)
        self._store_result(fitness)
        return fitness
```

### Fitness Metrics

| Metric | Formula | Direction | Constraint |
|--------|---------|-----------|-----------|
| Sharpe | `mean(returns) / std(returns) * sqrt(252)` | maximize | — |
| Mean IC | `mean(spearman_corr(signal, forward_return))` | maximize | — |
| Max Drawdown | `max(peak - trough) / peak` | minimize | < 0.15 |
| Signal Count | `count(abs(signal) > threshold) / days` | — | > 50/day |
| Turnover | `sum(abs(position_change)) / days` | — | < 100/day |

### Results Storage (SQLite)

```sql
CREATE TABLE trials (
    trial_id    INTEGER PRIMARY KEY,
    config_hash TEXT NOT NULL,        -- SHA-256 of config TOML
    config_json TEXT NOT NULL,        -- full config as JSON
    symbol      TEXT NOT NULL,
    eval_hours  INTEGER NOT NULL,
    sharpe      REAL,
    mean_ic     REAL,
    max_drawdown REAL,
    signal_count REAL,
    turnover    REAL,
    eval_time_s REAL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Location:** `data/swarm/results.db`

### Reuse from Existing Code

- `scripts/algorithms/registry.py` — `get_registered()` for algorithm dispatch
- `scripts/algorithms/base.py` — `MicrostructureAlgorithm.run_batch()`
- `scripts/algorithms/ensemble.py` — ensemble methods (equal_weight, ic_weight)
- `scripts/analysis/` — Parquet reading patterns

### Docker Service

```yaml
evaluator:
  build:
    context: .
    dockerfile: docker/Dockerfile.evaluator
  volumes:
    - parquet_data:/app/data/features:ro
    - ./config:/app/config:ro
    - swarm_data:/app/data/swarm
  command: >
    python scripts/swarm/evaluator.py
      --config /app/config/swarm/config_${INSTANCE_ID}.toml
      --data-dir /app/data/features
      --db /app/data/swarm/results.db
```

### Performance Target

- ~5 seconds per evaluation (1 day of Parquet data, BTC only)
- 8 parallel evaluators = ~11,000 evaluations/day
- Memory: < 500MB per worker (Parquet is columnar, loads only needed columns)

## Verification

```bash
# Single evaluation
python scripts/swarm/evaluator.py \
  --config config/algorithms.toml \
  --data-dir data/features \
  --symbol BTC --hours 24

# Check results
sqlite3 data/swarm/results.db "SELECT sharpe, mean_ic FROM trials LIMIT 5"
```

## Files Created

- `scripts/swarm/evaluator.py`
- `scripts/swarm/__init__.py`
- `docker/Dockerfile.evaluator`
