# 0 — Cloud deployment overview & runbook (T0b / M2)

Status: **ready to execute** (turnkey once the box exists). The wired binary and the
full Docker stack are built and verified; this is the operational runbook to stand
up a redundant cloud ingestor + the automation/observability stack, fix the dead
features in production, and run the T0 viability verdict.

Companion docs: `1_1_docker_stack.md` (services), `1_2_prometheus_grafana.md`
(ingestor metrics), `2_observability_and_e2e.md` (exporter + dashboards + bring-up
order). Concentration decision matrix:
`docs/in_progress/tasks_assigned_12_6_26/01_concentration_viability_assessment.md`.

## Why a cloud box

Data continuity is the binding constraint. The cloud box is (a) a **redundant,
always-on second ingestor** independent of su-35, and (b) the **deployment vehicle
for the wired binary** (`[position_tracker]` enabled) while su-35 stays frozen.

## Provider / sizing

- **Default:** Hetzner **AX52** dedicated (8-core / 64 GB, ~€60/mo) — sized for
  ingestion + the evaluation swarm.
- **Ingestion-only first:** a Hetzner **CX/CPX VM** (~€10–20/mo) is enough until
  the swarm moves over.
- **Latency note:** Hetzner EU adds ~250 ms RTT to Hyperliquid (Tokyo) — fine for
  research ingestion. Live *execution* (T21) needs Tokyo-proximate hosting, decided
  then, not now.

## Hard guardrails

1. **su-35: zero contact** until the 7-day clean streak completes. The cloud box is
   a *separate* ingestor; su-35 upgrades to the wired binary only *after* the streak.
2. **Named-service invocations only** — never a bare `docker compose up` (it could
   touch unintended services). Bring services up by name.
3. Gates/thresholds imported, never invented. No live capital before G8 + a healthy
   kill-switch.

## Prerequisites

- Docker + docker compose on the box; this repo checked out; outbound WSS to
  `api.hyperliquid.xyz`.
- `.env` with `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` (gap-alert + kill-switch pages).
- `config/ing.toml` already has `[position_tracker] enabled = true` (the wired path).
- **Wiring already verified locally (Jun-16):** the wired binary populates all 31
  whale/liquidation/concentration columns ~6 min after start; `WsTrade.users IS
  populated` (wallet discovery works). So the deploy is expected to resolve the dead
  features — the open question is the *viability verdict*, not whether it works.

## Bring-up sequence (named services, in order)

```bash
nat doctor                                         # preflight: data-dir ownership/writability, binary, disk
nat docker build                                   # or: docker compose build
docker compose up -d redis postgres
docker compose up -d ingestor                      # wired binary; writes ./data/features
docker compose up -d gap-alert alerts              # <5min Telegram page on any data gap
docker compose up -d kill-switch                   # risk halt (publishes halt_state.json)
docker compose up -d promotion signal-bridge metrics-exporter
docker compose up -d prometheus grafana caddy api web optuna-dashboard
docker compose ps                                  # confirm all 15 healthy
```
That is the full **15-service** set (`docker compose config --services`): redis, postgres,
ingestor, gap-alert, alerts, kill-switch, promotion, signal-bridge, metrics-exporter, prometheus,
grafana, caddy, api, web, optuna-dashboard. `postgres` backs the Optuna studies; `alerts` is the
Telegram service. `depends_on` health-gates the order (redis→ingestor→gap-alert; kill-switch→
signal-bridge; prometheus→grafana; caddy→{grafana,api}; postgres→optuna-dashboard). Grafana:
`http://<host>:3002` (anonymous) — NAT Overview + Lifecycle Funnel + Paper Performance + Live P&L
auto-provision.

## Step 1 — first 24h: coverage check (resolve the dead features)

Within 1h of the wired ingestor running, confirm the 40 columns leave 100% NaN:

```bash
nat gap status                                     # ingestion fresh, no gap
# per-column non-NaN coverage for the 40 unlocked cols:
python3 -c "import glob,os,pandas as pd; f=sorted(glob.glob('data/features/*/*.parquet'),key=os.path.getmtime)[-1]; df=pd.read_parquet(f); \
cols=[c for c in df.columns if c.startswith(('whale_','liquidation_','top','conc_')) or 'herfindahl' in c or 'gini' in c]; \
print({c: round(df[c].notna().mean(),3) for c in cols})"
# WsTrade.users diagnostic (must say 'IS populated'):
docker compose logs ingestor | grep -iE 'WsTrade.users|Discovered and promoted'
```

Sanity ranges once populated: top5 0.05–0.50, top10 0.10–0.60, HHI 0.01–0.25,
Gini 0.30–0.80.

## Step 2 — 48h: concentration viability verdict

Apply the decision matrix in `01_concentration_viability_assessment.md`:

| Wallets tracked | OI coverage | Verdict |
|---|---|---|
| 50+ | >20% | **viable** — no change |
| 20–50 | 5–20% | **noisy** — add a FEATURES.md disclaimer |
| <20 | <5% | **unavailable** — keep NaN, documented (a valid outcome) |

Record the verdict in the `01_…` doc. It unblocks T9's LF3 (liquidation cascade)
and the agents' dead-column skip lists — or documents them as permanently gated.

## Step 3 — cutover decision (after the 7-day clean-data streak completes)

Compare cloud vs su-35 over overlapping hours: row counts, gap profile (`nat gap`),
feature parity within float noise. The cleaner box becomes primary; the other stays
as redundancy. Only then does su-35 upgrade to the wired binary.

## Disk retention

Raw parquet grows ~per-symbol-per-day; set a retention/expiry (downsample or delete
after N days) so the box doesn't fill. (Track growth from the first 24h.)

## Rollback / monitoring

- Gap-alert pages within ~5 min on any ingestion stall; kill-switch halts on a
  PnL/IC breach (paper/live only). Health: `nat gap status`, `nat risk status`.
- Roll back a service with `docker compose up -d --no-deps <service>` after a
  `git checkout` of the prior image, or stop a daemon with its `nat <x> stop`.
- The 48h viability clock runs on **this box's** data, not su-35's.
- **Data-dir ownership (silent-stall gotcha):** the Docker ingestor runs as **root** and creates
  **root-owned** `data/features` & `data/trades`. If you ever run a **native (non-root) ingestor**
  against the same tree (e.g. at cutover), it can't write the root-owned dirs and **stalls
  silently**. Fix: `sudo chown -R <user>:<user> data/`. **`nat doctor` detects this** — run it
  before switching ingestors.
