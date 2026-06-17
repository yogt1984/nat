# Hetzner Deployment Plan (T0b — cloud ingestion + automation stack)

End-to-end plan to go from **nothing → a running NAT cloud stack** on a fresh
Hetzner box. This is the provisioning/hardening/DNS/secrets layer that sits *above*
the existing in-stack docs:

| Doc | Covers |
|-----|--------|
| **this file** | provision → harden → secrets/DNS → deploy → verify → cutover |
| [`docs/cloud_deployment/0_overview.md`](docs/cloud_deployment/0_overview.md) | in-stack runbook: bring-up order, 24h/48h checks |
| [`docs/cloud_deployment/1_1_docker_stack.md`](docs/cloud_deployment/1_1_docker_stack.md) | the Docker services / images |
| [`docs/cloud_deployment/1_2_prometheus_grafana.md`](docs/cloud_deployment/1_2_prometheus_grafana.md) | ingestor metrics + NAT Overview dashboard |
| [`docs/cloud_deployment/2_observability_and_e2e.md`](docs/cloud_deployment/2_observability_and_e2e.md) | metrics exporter + dashboards + bring-up chain |
| [`docs/in_progress/tasks_assigned_12_6_26/01_concentration_viability_assessment.md`](docs/in_progress/tasks_assigned_12_6_26/01_concentration_viability_assessment.md) | the 48h concentration decision matrix |
| [`.env.example`](.env.example) | the full secrets template |

## Why

Cloud deployment is the next P0 action. One bring-up does three things at once:
1. **Fixes the dead features in production** — the wired binary (`[position_tracker]`
   enabled) populates the 40 whale/liquidation/concentration columns. *Verified
   locally Jun-16:* `WsTrade.users IS populated` and all 31 columns go non-NaN ~6 min
   after start, so this is expected to work — the open question is the *viability
   verdict*, not whether it functions.
2. **Stands up a redundant, always-on ingestor** — data continuity is the binding
   constraint; a second box independent of su-35 protects it.
3. **Runs the 48h concentration viability verdict** (viable / noisy / unavailable).

## Guardrails (non-negotiable)

- **su-35: zero contact** until the 7-day clean streak completes. The Hetzner box is
  a *separate* ingestor — that is the whole point (it's the deploy vehicle for the
  wired binary while su-35 stays frozen). su-35 upgrades to the wired binary only
  *after* the streak, at cutover.
- **Named-service invocations only** — never a bare `docker compose up`.
- Gates imported, never invented. **No live capital** before G8 + a healthy
  kill-switch (the bridge ships dry-run by default).

## Provider / sizing

- **Default — Hetzner AX52 dedicated** (8-core / 64 GB, ~€60/mo): ingestion + the
  evaluation swarm.
- **Ingestion-only first — Hetzner CX/CPX VM** (~€10–20/mo): enough until the swarm
  moves over; add swap.
- **Latency:** Hetzner EU adds ~250 ms RTT to Hyperliquid (Tokyo) — fine for research
  ingestion. *Live execution (T21)* needs Tokyo-proximate hosting, decided then.

---

## Phase 0 — Provision & harden the box

1. Create the box (Ubuntu LTS). Add your SSH key; disable password auth.
2. Firewall (`ufw`): allow `22`, `80`, `443`; deny the rest. Plan to keep the
   `prometheus.` vhost off the public internet (see Phase 1 / Ops).
3. Install Docker Engine + the compose plugin. Add your user to the `docker` group.
4. (VM only) add 4–8 GB swap.
5. `git clone` this repo; `cd nat`.

## Phase 1 — Secrets & DNS

1. `cp .env.example .env` and fill, at minimum:
   - `DOMAIN` (your domain), `ACME_EMAIL` (Let's Encrypt) — Caddy auto-TLS.
   - `POSTGRES_PASSWORD`, `GRAFANA_PASSWORD`.
   - **`GRAFANA_ANON=false`** on a public box (the default `true` is for local dev).
   - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — run `scripts/setup_telegram.sh` for
     interactive setup (powers gap-alert + kill-switch pages).
2. DNS A-records → the box IP: the apex `DOMAIN` (→ Grafana) plus `api.`,
   `dashboard.`, `prometheus.` subdomains (see `docker/Caddyfile`). Caddy issues TLS
   automatically on first request.

## Phase 2 — Build & bring up (named services, in order)

```bash
nat doctor                            # preflight: data-dir ownership/writability, binary, disk
nat docker build                      # or: docker compose build
docker compose up -d redis postgres
docker compose up -d ingestor         # WIRED binary; writes ./data/features
docker compose up -d gap-alert alerts # <5min Telegram page on any data gap
docker compose up -d kill-switch
docker compose up -d promotion signal-bridge metrics-exporter
docker compose up -d prometheus grafana caddy api web optuna-dashboard
nat docker ps                         # confirm all 15 services healthy
```

That brings up all **15** services (`docker compose config --services`): redis, postgres,
ingestor, gap-alert, alerts, kill-switch, promotion, signal-bridge, metrics-exporter, prometheus,
grafana, caddy, api, web, optuna-dashboard. `postgres` backs the Optuna studies (optuna-dashboard
depends on it); `alerts` is the Telegram service. Order is health-gated by `depends_on`
(redis→ingestor→gap-alert; kill-switch→signal-bridge; prometheus→grafana; caddy→{grafana,api};
postgres→optuna-dashboard). See `docs/cloud_deployment/2_observability_and_e2e.md` for the full
chain. `nat doctor` up front catches the silent-stall case where a data dir is owned by a different
user than the ingestor process (see Operations).

## 24/7 / always-on (what keeps it up without babysitting)

Once Phase 2 is up, the stack is self-running — no cron/manual restarts:

- **`restart: unless-stopped` on all 15 services** — Docker auto-restarts any service that crashes
  **and on host/Docker reboot**, until you explicitly `docker compose stop` it. This (not a script)
  is what makes the box 24/7.
- **Self-healing/monitoring** — gap-alert pages within ~5 min on any ingestion stall; the
  kill-switch halts on a PnL/IC breach (paper/live only); Prometheus/Grafana track health; Caddy
  renews TLS automatically.
- **Always-on box** — the point of the cloud box: a redundant ingestor independent of su-35, since
  data continuity is the binding constraint.

## Phase 3 — Verify (first 24h)

Per `docs/cloud_deployment/0_overview.md` Step 1:
- `nat gap status` → ingestion fresh, no gap.
- **Dead-feature coverage** — within ~1h the 40 whale/liquidation/concentration
  columns leave 100% NaN (use the one-liner in `0_overview.md`).
- **`WsTrade.users` diagnostic** — `docker compose logs ingestor | grep WsTrade.users`
  must show *"IS populated"* + *"Discovered and promoted whale wallets"*.
- Grafana over HTTPS at `https://$DOMAIN` (Lifecycle Funnel / Paper / Live P&L /
  NAT Overview auto-provisioned); TLS valid.

## Phase 4 — Concentration viability verdict (48h)

Apply the decision matrix (`0_overview.md` Step 2 + the `01_…` doc):

| Wallets tracked | OI coverage | Verdict |
|---|---|---|
| 50+ | >20% | **viable** — no change |
| 20–50 | 5–20% | **noisy** — FEATURES.md disclaimer |
| <20 | <5% | **unavailable** — keep NaN, documented (valid outcome) |

Record the verdict in the `01_…` doc. It unblocks LF3 + the agents' dead-column
skip lists, or documents them as permanently gated.

## Phase 5 — Cutover (only after the 7-day clean-data streak completes)

Compare cloud vs su-35 over overlapping hours: row counts, gap profile (`nat gap`),
feature parity within float noise. The cleaner box becomes primary; the other stays
as redundancy. su-35 upgrades to the wired binary at this point — **not before**.

---

## Operations

- **Retention:** raw parquet grows per-symbol-per-day; set an expiry/downsample
  after N days so the disk doesn't fill (size it from the first 24h).
- **Backups:** `data/nat.db` (lifecycle/research SQLite) + the postgres volume
  (Optuna studies).
- **Security:** `ufw` as above; `GRAFANA_ANON=false`; keep the `prometheus.` vhost
  internal (drop its DNS record or IP-restrict in `docker/Caddyfile`).
- **Monitoring:** gap-alert pages within ~5 min on a stall; kill-switch halts on a
  PnL/IC breach (paper/live only). Health: `nat gap health`, `nat risk status`,
  `python scripts/monitoring/metrics_exporter.py health`.
- **Rollback:** `docker compose up -d --no-deps <service>` after a `git checkout` of
  the prior code, or stop a daemon with its `nat <x> stop`.
- **Data-dir ownership (silent-stall gotcha):** the Docker ingestor runs as **root** and creates
  **root-owned** `data/features` & `data/trades` dirs. If you ever run a **native (non-root)
  ingestor** against the same tree (e.g. at cutover, or a local `nat start`), it **cannot write the
  root-owned dirs and stalls silently** (the writer task dies; no error to the operator). Fix:
  `sudo chown -R <user>:<user> data/`. **`nat doctor` detects this** before `nat start` — run it
  first whenever you switch between the Docker and native ingestor.

## Out of scope

Provisioning the box and running the deploy are operator actions; this plan makes
them turnkey. The data-gated track (streak completion → T11/G1, the conditional-IC
gate) proceeds in parallel and is unaffected by this deployment.
