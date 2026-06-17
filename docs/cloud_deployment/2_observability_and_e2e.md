# 2 — Observability & end-to-end stack (T18)

Status: **partial** (metrics exporter + dashboards + lifecycle tab done; per-daemon
`/metrics` endpoints deferred — see below).

Builds on `1_1_docker_stack.md` (services) and `1_2_prometheus_grafana.md` (ingestor
metrics + the NAT Overview dashboard). This doc covers observability for the Python
automation spine (kill-switch, gap-alert, promotion, signal-bridge) and the full
`docker compose up`.

## The metrics exporter (the keystone)

Grafana can only read Prometheus; NAT's business state lives in SQLite/JSON. The
exporter bridges that:

- **Service:** `metrics-exporter` (`docker/Dockerfile.exporter`, pure-stdlib +
  `prometheus-client`), exposes `:9094/metrics`.
- **Code:** `scripts/monitoring/metrics_exporter.py` — refreshes every
  `refresh_s` (config `config/monitoring.toml`).
- **Metrics:**
  - `nat_lifecycle_signals{state}` — signal counts from `signal_lifecycle` (nat.db)
  - `nat_live_cum_pnl_pct`, `nat_live_last_daily_pnl_pct`, `nat_live_pnl_days` —
    from `data/execution/daily_pnl.json` (the bridge's rollup)
  - `nat_paper_sharpe{signal}`, `nat_paper_max_drawdown_bps{signal}` — from
    `data/oos_validation/state.json`
- **Scrape:** `docker/prometheus/prometheus.yml` job `nat-metrics-exporter`.
- **CLI:** `python scripts/monitoring/metrics_exporter.py once|start|health`.

## Grafana dashboards (auto-provisioned)

Dropped in `docker/grafana/dashboards/` (provider loads `/var/lib/grafana/dashboards`):

| Dashboard | uid | Reads |
|-----------|-----|-------|
| NAT Lifecycle Funnel | `nat-lifecycle-funnel` | `nat_lifecycle_signals{state}` |
| NAT Paper Performance | `nat-paper-performance` | `nat_paper_sharpe`, `nat_paper_max_drawdown_bps` |
| NAT Live P&L | `nat-live-pnl` | `nat_live_*` |
| NAT Overview | `nat-overview` | ingestor metrics (throughput / freshness) |

The lifecycle funnel highlights `APPROVAL_PENDING` (the human gate). Paper/live
panels are empty until the paper window accrues (~Aug) — by design.

## Agent dashboard — lifecycle tab

`scripts/agent_dashboard.py` (stdlib HTTP, :8060) gains `/api/lifecycle` and a
"Signal Lifecycle" card: state funnel + signals table + approval-pending callout,
read live from `nat.db`.

## End-to-end stack

`docker compose config` is valid across all 15 services with no port conflicts
(the four Python daemons publish no host ports). Bring-up order is enforced by
healthchecked `depends_on`: `redis → ingestor → gap-alert`; `kill-switch →
signal-bridge`; `prometheus → grafana`; `caddy → {grafana, api}`.

Runbook (on the deploy box):
```bash
nat doctor                                    # preflight: data-dir ownership/writability, binary, disk
nat docker build                              # or: docker compose build
docker compose up -d redis postgres           # data + Optuna store first
docker compose up -d ingestor                 # wired binary
docker compose up -d gap-alert alerts kill-switch   # monitoring/safety
docker compose up -d promotion signal-bridge metrics-exporter
docker compose up -d prometheus grafana caddy api web optuna-dashboard
docker compose ps                             # all 15 healthy?
# Grafana: http://<host>:3002  (anonymous); dashboards auto-provisioned.
```
All **15** services (`docker compose config --services`): redis, postgres, ingestor, gap-alert,
alerts, kill-switch, promotion, signal-bridge, metrics-exporter, prometheus, grafana, caddy, api,
web, optuna-dashboard. Named-service invocations only — never a bare `docker compose up` while
su-35 is streak-frozen (constraint 1).

> **Data-dir ownership gotcha:** the Docker ingestor runs as root and creates root-owned
> `data/` dirs; a native (non-root) ingestor against the same tree stalls silently on writes.
> `nat doctor` detects it; fix with `sudo chown -R <user>:<user> data/`.

## Deferred (follow-on)

- **Per-daemon `/metrics` endpoints** (kill-switch already has a guarded hook; the
  others would each expose `:909x`). Lower marginal value: the exporter already
  carries the business metrics and the daemons' healthchecks cover liveness. Add
  when per-daemon internal counters (cycle latency, halt-trigger counts) are
  wanted on the cloud box.
- **`docker compose up` E2E integration test** with live fixtures — a deploy-box
  activity (heavyweight bring-up of 15 services).
