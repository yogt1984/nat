# 2.4 Swarm Orchestrator

## Status: NOT_STARTED

## Goal

Coordinate N evaluator workers: launch them, collect results, rank configs by
fitness, and expose rankings via CLI and Grafana.

## Prerequisites

- [2.1 Shared Ingestor](2_1_shared_ingestor.md) — data pipeline
- [2.2 Config Generator](2_2_config_generator.md) — configs generated
- [2.3 Evaluator Worker](2_3_evaluator_worker.md) — individual eval works

## Architecture

```
nat swarm run --instances 16
       │
       ▼
┌──────────────────┐
│   Orchestrator   │
│                  │
│ 1. Generate N    │
│    configs       │
│ 2. Launch N      │──► docker compose up --scale evaluator=N
│    evaluators    │
│ 3. Monitor       │──► poll results.db every 30s
│    progress      │
│ 4. Rank + report │──► stdout + Grafana
└──────────────────┘
```

## Implementation

**New file:** `scripts/swarm/orchestrator.py`

```python
class SwarmOrchestrator:
    def __init__(self, config_path: str, db_path: str):
        self.config = toml.load(config_path)
        self.db = sqlite3.connect(db_path)

    def run(self, n_instances: int, hours: int = 24):
        """Full swarm run."""
        # 1. Generate configs
        configs = ConfigGenerator(self.config).generate_random(n_instances)
        # 2. Write to data/swarm/configs/
        # 3. Launch evaluators (subprocess or Docker scale)
        # 4. Wait for completion
        # 5. Rank and report

    def status(self) -> dict:
        """Current swarm status: running, completed, pending."""
        # Query results.db for completed trials
        # Return progress + top-5 results

    def results(self, top_n: int = 10) -> pd.DataFrame:
        """Return top configs ranked by Sharpe."""
        return pd.read_sql("""
            SELECT config_hash, sharpe, mean_ic, max_drawdown,
                   signal_count, turnover
            FROM trials
            ORDER BY sharpe DESC
            LIMIT ?
        """, self.db, params=[top_n])

    def export_best(self, output_path: str):
        """Export the best config as a usable TOML file."""
```

### CLI Integration

**File:** `nat` (CLI script) — add subcommands:

```python
# nat swarm run --instances 16 --hours 24
def cmd_swarm_run(args):
    _py(f"scripts/swarm/orchestrator.py run "
        f"--instances {args.instances} --hours {args.hours}")

# nat swarm status
def cmd_swarm_status(args):
    _py("scripts/swarm/orchestrator.py status")

# nat swarm results --top 10
def cmd_swarm_results(args):
    _py(f"scripts/swarm/orchestrator.py results --top {args.top}")

# nat swarm best --export config/best.toml
def cmd_swarm_best(args):
    _py(f"scripts/swarm/orchestrator.py best --export {args.export}")
```

### Grafana Dashboard: Swarm Heatmap

**New file:** `docker/grafana/dashboards/swarm_results.json`

Panels:
- **Sharpe Distribution** — histogram of all trial Sharpe ratios
- **Parameter Importance** — which params correlate most with fitness
- **Top 10 Configs** — table with fitness metrics
- **Convergence Curve** — best Sharpe over time (trial index)
- **Parameter Scatter** — 2D scatter of key params vs Sharpe

Data source: Prometheus exporter reading from results.db, or direct
PostgreSQL datasource in Grafana (if using Postgres backend).

### Prometheus Exporter for Swarm Metrics

**New file:** `scripts/swarm/metrics_exporter.py`

Exposes swarm metrics for Prometheus scraping:

```python
# Gauges
swarm_trials_total          # total completed evaluations
swarm_best_sharpe           # current best Sharpe
swarm_best_ic               # current best IC
swarm_running_evaluators    # number of active workers
swarm_eval_rate             # evaluations per minute
```

## Execution Modes

### Local (subprocess)

```bash
# Run 8 evaluators as parallel subprocesses
nat swarm run --instances 8 --hours 24 --mode local
```

Uses `multiprocessing.Pool` — simplest, works without Docker.

### Docker (compose scale)

```bash
# Run 16 evaluators as Docker containers
nat swarm run --instances 16 --hours 24 --mode docker
```

Uses `docker compose up --scale evaluator=16`.

## Verification

```bash
# Run small swarm
nat swarm run --instances 4 --hours 1

# Check status
nat swarm status
# Expected: 4/4 complete, best Sharpe: X.XX

# View results
nat swarm results --top 4
# Expected: table with 4 rows, sorted by Sharpe

# Export winner
nat swarm best --export config/best_algorithms.toml
diff config/algorithms.toml config/best_algorithms.toml
```

## Files Created

- `scripts/swarm/orchestrator.py`
- `scripts/swarm/metrics_exporter.py`
- `docker/grafana/dashboards/swarm_results.json`
- `nat` — swarm subcommands added
- `config/swarm.toml` — orchestrator settings
