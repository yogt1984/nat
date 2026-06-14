# Cloud Deployment Architecture — Overview

## Vision

NAT evolves from a single-machine tmux setup into a cloud-native platform with
automated parameter optimization. Three tiers, each building on the previous:

```
Tier 1: Run continuously, observe everything
Tier 2: Run N configs in parallel, compare fitness
Tier 3: Let an optimizer evolve configs toward optimal performance
```

## Tier Dependency Chain

```
Tier 1 (Observe) ──► Tier 2 (Swarm) ──► Tier 3 (Evolve)
   Docker stack         Shared ingestor      Optuna optimizer
   Prometheus/Grafana   Config generator     CMA-ES / NSGA-II
   Production harden    Evaluator workers    Walk-forward OOS
```

## Task Index

### Tier 1: Continuous Cloud + Observability

| Task | Description | Status |
|------|-------------|--------|
| [1_1](1_1_docker_stack.md) | Docker Compose stack + Dockerfiles | DONE |
| [1_2](1_2_prometheus_grafana.md) | Prometheus metrics + Grafana dashboards | DONE |
| [1_3](1_3_production_hardening.md) | Caddy HTTPS, PostgreSQL, auth, backup | DONE |
| [1_4](1_4_testing_verification.md) | Build test, endpoint smoke tests | DONE |

### Tier 2: Config Swarm

| Task | Description | Status |
|------|-------------|--------|
| [2_1](2_1_shared_ingestor.md) | Shared ingestor → Parquet → N evaluators | DONE |
| [2_2](2_2_config_generator.md) | 35D parameter space, TOML templating | DONE |
| [2_3](2_3_evaluator_worker.md) | Lightweight evaluator, fitness dict | DONE |
| [2_4](2_4_swarm_orchestrator.md) | Orchestration, ranking, CLI, heatmap | DONE |

### Tier 3: Evolutionary Optimization

| Task | Description | Status |
|------|-------------|--------|
| [3_1](3_1_optuna_setup.md) | Optuna + CMA-ES/TPE, distributed backend | DONE |
| [3_2](3_2_fitness_function.md) | Multi-objective fitness (Sharpe, DD, IC) | DONE |
| [3_3](3_3_guard_rails.md) | Walk-forward OOS, overfit detection | DONE |

## Estimated Effort

| Tier | Effort | Prerequisite |
|------|--------|-------------|
| 1 | ~2 days | None |
| 2 | ~3-4 days | Tier 1 complete |
| 3 | ~1-2 weeks | Tier 2 complete |

## Infrastructure

- **Target:** Hetzner AX52 (8-core, 64GB, ~EUR60/mo) or equivalent
- **Current:** su-35 (dev/staging)
- **Container runtime:** Docker + Compose v2
