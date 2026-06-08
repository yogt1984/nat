# 1.4 Testing & Verification

## Status: DONE

## Goal

Verify the entire Tier 1 stack works end-to-end: Docker builds compile, services
start, metrics flow from ingestor → Prometheus → Grafana.

## Prerequisites

- [1.1 Docker Stack](1_1_docker_stack.md) — DONE
- [1.2 Prometheus + Grafana](1_2_prometheus_grafana.md) — DONE

## Test Plan

### Step 1: Docker Build

```bash
docker compose build ingestor api alerts
```

**Pass criteria:** All 3 images build without error. Verify crate-copy fix works
(ing-types, ing-features both present in build context).

**Known issue:** First build takes 5-10 min (Rust release + LTO). Subsequent
builds use Docker layer cache.

### Step 2: Start Stack

```bash
docker compose up -d
docker compose ps
```

**Pass criteria:** All services show `running` or `healthy`.

**Known issue:** Port 3000 may conflict with other services (e.g. open-webui).
If so, remap API to another port:
```yaml
ports:
  - "3010:3000"  # avoid conflict
```

### Step 3: Automated Smoke Test

All endpoint checks are automated via `nat docker smoke`:

```bash
nat docker smoke
```

Checks 6 services:
- **Ingestor** — HTTP 200 on :8080
- **API** — HTTP 200 on :3010/health
- **Prometheus** — HTTP 200 on :9090/api/v1/targets
- **Grafana** — HTTP 200 on :3002/api/health
- **PostgreSQL** — `pg_isready -U nat`
- **Caddy** — HTTP 200/301 on :80

### Step 4: Metrics Flow Verification

The `nat docker stack` command verifies metrics automatically:

```bash
nat docker stack --no-build
# Output includes: "Metrics  Prometheus scraping 3 series"
```

Manual check if needed:
```bash
curl -s 'http://localhost:9090/api/v1/query?query=ing_features_emitted_total' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['data']['result']), 'series')"
# Expected: 3 series (BTC, ETH, SOL)
```

### Step 5: Teardown

```bash
docker compose down
docker compose down -v  # also remove volumes (optional)
```

## Resolved Issues

- **Rust 1.75 → 1.89:** Cargo.lock v4 requires Rust >= 1.78. Fixed in all Dockerfiles.
- **Build context 1.5GB → 53MB:** Added `.dockerignore` excluding `rust/target/`, `data/`, `web/`, `exploration/`.
- **Port 3000 conflict:** API remapped to 3010 (open-webui uses 3000 on su-35).

## Files Modified

- `README.md` — Docker observability section added
- `CLAUDE.md` — Docker services + env vars updated
- `nat` — `docker stack`, `docker smoke` commands added
