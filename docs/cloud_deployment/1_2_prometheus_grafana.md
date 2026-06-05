# 1.2 Prometheus + Grafana Observability

## Status: DONE

## Goal

Auto-provisioned Prometheus scraping + Grafana dashboard so metrics are visible
the moment `docker compose up` runs. No manual configuration needed.

## Prerequisites

- [1.1 Docker Stack](1_1_docker_stack.md) — services must be running

## Components

### Ingestor Metrics (Rust)

**File:** `rust/ing/src/metrics.rs`

4 metrics registered via the `metrics` crate + `metrics_exporter_prometheus`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `ing_features_emitted_total` | Counter | symbol | Features computed |
| `ing_errors_total` | Counter | symbol, type | Errors by category |
| `ing_feature_latency_seconds` | Histogram | — | Feature compute time |
| `ing_update_latency_seconds` | Histogram | — | WS update processing |

The `PrometheusBuilder` binds to `ING_PROMETHEUS_ADDR` (default: disabled).
In Docker, set to `0.0.0.0:9090` to expose.

### Prometheus Service

**File:** `docker/prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nat-ingestor'
    static_configs:
      - targets: ['nat-ingestor:9090']
        labels:
          service: 'ingestor'
```

**Docker compose:** `prom/prometheus:v2.53.0`, port 9090, 90-day retention.

### Grafana Service

**File:** `docker/grafana/provisioning/datasources/prometheus.yml`

```yaml
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

**File:** `docker/grafana/provisioning/dashboards/dashboards.yml`

Loads JSON dashboards from `/var/lib/grafana/dashboards` every 30s.

**Docker compose:** `grafana/grafana:11.1.0`, port 3002, anonymous viewer enabled.

### Dashboard: NAT Overview

**File:** `docker/grafana/dashboards/nat_overview.json`

7 panels, auto-refresh 5s, 1h time range:

| Panel | Type | Query |
|-------|------|-------|
| Feature Emission Rate | timeseries | `rate(ing_features_emitted_total[1m])` by symbol |
| Error Rate | timeseries | `rate(ing_errors_total[5m]) * 60` by symbol+type |
| Feature Compute Latency | timeseries | p50/p95/p99 via `histogram_quantile` |
| WS Update Latency | timeseries | p50/p95/p99 via `histogram_quantile` |
| Total Features Emitted | stat | `sum(ing_features_emitted_total)` |
| Total Errors | stat | `sum(ing_errors_total)` (red threshold >= 1) |
| Uptime | stat | `time() - process_start_time_seconds` |

## Verification

```bash
# Prometheus scraping
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[].health'
# Expected: "up"

# Grafana alive
curl http://localhost:3002/api/health
# Expected: {"commit":"...","database":"ok","version":"11.1.0"}

# Dashboard provisioned
curl http://localhost:3002/api/dashboards/uid/nat-overview | jq '.dashboard.title'
# Expected: "NAT Overview"
```

## Files Created

- `docker/prometheus/prometheus.yml`
- `docker/grafana/provisioning/datasources/prometheus.yml`
- `docker/grafana/provisioning/dashboards/dashboards.yml`
- `docker/grafana/dashboards/nat_overview.json`
- `docker-compose.yml` — prometheus + grafana services added
