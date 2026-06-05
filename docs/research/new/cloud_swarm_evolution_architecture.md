# Cloud Deployment, Config Swarm & Evolutionary Optimization

**Date:** 2026-06-05
**Status:** Architecture proposal

Three escalating tiers for scaling NAT from a single-machine research tool to an
evolutionary optimization platform.

---

## Tier 1: Continuous Cloud + Observability

**Goal:** NAT running 24/7, research state visible remotely.

### Stack

- **Compute:** Hetzner AX-series dedicated (AX52: 8-core, 64GB, ~EUR60/mo). Bare-metal perf, no cloud tax.
- **Orchestration:** Docker Compose (existing `docker-compose.yml` with redis, ingestor, api, alerts). Add Grafana + Prometheus as services.
- **Observability:** Grafana dashboards reading from existing `metrics-exporter-prometheus`. Panels for: feature emission rate, WS staleness, algorithm IC (daily), agent cycle progress.
- **State persistence:** PostgreSQL for research state (agent cycles, hypothesis results, optimization trials). Migrate from JSON files.
- **Remote access:** Grafana on HTTPS (Caddy reverse proxy), Telegram alerts (already wired).

### Development

1. Dockerize the full stack (ingestor + research agent + dashboard + monitoring)
2. Deploy to Hetzner, verify parity with su-35
3. Add Grafana dashboards for existing Prometheus metrics
4. Wire `nat status --json` output to a Prometheus exporter

### Effort: ~2 days

---

## Tier 2: Config Swarm

**Goal:** N parallel evaluator workers, each with different algorithm parameters, competing on the same data.

### Critical Architecture Insight

Don't run N full NAT instances. Run **1 ingestor** that produces Parquet, and **N lightweight evaluator workers** that read the same data with different configs. This makes the swarm 100x cheaper.

```
1 ingestor --> shared Parquet files
N evaluator workers --> each reads same Parquet, runs algorithms with different configs
```

### Parameter Space (~35 dimensions)

| Category | Parameters | Dims |
|----------|-----------|------|
| Algorithm thresholds | jump z_threshold, SPRT boundaries, funding z_window | ~15 |
| Ensemble | method, ic_lookback, regime_column, weights | ~5 |
| Trading | entry/exit percentiles, bar_seconds, position sizing | ~8 |
| Feature selection | which optional features to enable | ~7 binary |
| Convolver | kernel_path, candle_freq, score_threshold | ~3 |

### Stack

- **Config generation:** Python script that takes base `config/algorithms.toml` + parameter ranges, emits N TOML variants.
- **Runner:** Docker Compose profiles or thin Python orchestrator. Each container gets a unique config volume-mounted. Results written to `data/swarm/{instance_id}/`.
- **Evaluation:** Collector script reads each instance output, computes fitness metrics (OOS IC, Sharpe, signal count, max drawdown), writes to SQLite/Postgres trials table.
- **Dashboard:** Grafana swarm panel (parameter heatmap vs fitness) or Optuna's built-in dashboard.

### New Scripts

- `scripts/swarm/config_generator.py` -- base config + parameter ranges --> N TOML files
- `scripts/swarm/evaluator.py` -- runs all algorithms on shared Parquet with a given config, returns fitness dict
- `scripts/swarm/collector.py` -- aggregates results across instances, ranks by fitness

### CLI

```bash
nat swarm run --instances 16 --hours 24
nat swarm status
nat swarm results --top 10
```

### Effort: ~3-4 days

---

## Tier 3: Metaheuristic Optimization (PSO / CMA-ES)

**Goal:** Automated evolution of configurations toward optimal trading performance.

### Stack

- **Optimizer:** Optuna as the primary framework.
  - Supports CMA-ES (better than PSO for continuous spaces -- faster convergence, rotation-invariant)
  - Supports TPE (Tree-structured Parzen Estimator -- more sample-efficient for expensive evaluations)
  - Built-in **pruning** -- kills bad trials early (critical when each eval takes minutes)
  - **Distributed workers** via PostgreSQL storage -- multiple machines can share a study
  - Built-in **dashboard** (`optuna-dashboard`) -- live visualization of parameter importance, optimization history, Pareto fronts
  - Multi-objective support (Sharpe vs drawdown vs turnover)
  - Python-native, integrates with existing scripts
- **If specifically wanting PSO:** Use DEAP alongside Optuna for raw GA/PSO/ES primitives.

### Why CMA-ES Over PSO

The parameter space is ~35D continuous. CMA-ES adapts its covariance matrix to the landscape -- it learns which parameters are correlated and searches along those axes. PSO can get stuck in local optima in high-D spaces. Optuna's CMA-ES sampler is production-ready.

### Fitness Function

Multi-objective, evaluated on walk-forward OOS:

```python
def fitness(config) -> dict:
    results = run_all_algorithms(parquet_data, config)
    return {
        "sharpe": compute_sharpe(results),          # maximize
        "max_drawdown": compute_drawdown(results),  # minimize
        "mean_ic": compute_mean_ic(results),         # maximize
        "signal_count": count_signals(results),      # constraint: > 50/day
        "turnover": compute_turnover(results),       # constraint: < 100/day
    }
```

### Guard Rails

- Walk-forward validation (train on days 1-20, test on 21-30) built into fitness
- Overfit detection: if in-sample Sharpe > 3x OOS Sharpe, flag and penalize
- Minimum signal count constraint prevents degenerate solutions (configs that never trade)

### Parallelism

Optuna workers run on the Hetzner box (8-16 parallel evaluations). Each evaluation = load Parquet + run algorithms + compute fitness. At ~5s per evaluation on 1 day of data, 16 workers evaluate ~11,000 configs/day.

### CLI

```bash
nat evolve start --study my_study --trials 5000 --workers 8
nat evolve status --study my_study
nat evolve best --study my_study --top 5
nat evolve pareto --study my_study   # multi-objective Pareto front
```

### Effort: ~1-2 weeks (basic CMA-ES loop: 1 week; multi-objective + dashboard + guard rails: 2 weeks)

---

## Development Order

Strictly 1 --> 2 --> 3. Each tier builds on the previous.

```
Tier 1: cloud infra + monitoring
  |
  v
Tier 2: evaluator function + config generation
  |  (Tier 2's evaluator becomes Tier 3's objective function)
  v
Tier 3: Optuna optimizer + distributed workers + Pareto dashboard
```

## Summary

| Tier | Stack | Key Tool | Effort |
|------|-------|----------|--------|
| 1. Cloud + observe | Hetzner + Docker Compose + Grafana | Prometheus | 2 days |
| 2. Config swarm | Shared ingestor + N evaluator workers | Config generator | 3-4 days |
| 3. Evolution | Optuna (CMA-ES/TPE) + PostgreSQL + dashboard | Optuna | 1-2 weeks |
