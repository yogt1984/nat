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

### Step 3: Endpoint Smoke Tests

Run each curl and verify expected response:

```bash
# 1. Ingestor dashboard
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080
# Expected: 200

# 2. API health
curl -s http://localhost:3000/health
# Expected: JSON with status ok

# 3. Prometheus targets
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool
# Expected: activeTargets with health="up" for nat-ingestor

# 4. Grafana health
curl -s http://localhost:3002/api/health
# Expected: {"database":"ok"}

# 5. Grafana dashboard exists
curl -s http://localhost:3002/api/dashboards/uid/nat-overview \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['dashboard']['title'])"
# Expected: "NAT Overview"
```

### Step 4: Metrics Flow Verification

Wait 1-2 minutes after startup, then:

```bash
# Check Prometheus has scraped metrics
curl -s 'http://localhost:9090/api/v1/query?query=ing_features_emitted_total' \
  | python3 -m json.tool
# Expected: result array with data points

# Check latency histograms exist
curl -s 'http://localhost:9090/api/v1/query?query=ing_feature_latency_seconds_bucket' \
  | python3 -m json.tool
# Expected: non-empty result
```

### Step 5: Visual Check

Open in browser:
- http://localhost:3002 → Grafana → "NAT Overview" dashboard
- Verify panels show live data (non-empty graphs)
- Check emission rate panel shows per-symbol lines
- Check latency panel shows p50/p95/p99 curves

### Step 6: Teardown

```bash
docker compose down
docker compose down -v  # also remove volumes (optional)
```

## Automated Smoke Script (future)

```bash
#!/bin/bash
# scripts/test_docker_smoke.sh
set -e
docker compose up -d
sleep 30  # wait for startup + first scrape
for url in "localhost:8080" "localhost:3000/health" \
           "localhost:9090/api/v1/targets" "localhost:3002/api/health"; do
  status=$(curl -s -o /dev/null -w "%{http_code}" "http://$url")
  [ "$status" = "200" ] || { echo "FAIL: $url ($status)"; exit 1; }
done
echo "ALL PASS"
docker compose down
```

## Current Blockers

- Docker build in progress (first Rust compile, ~10 min)
- Port 3000 conflict with open-webui on su-35

## Files Modified

- `README.md` — Docker observability section added (uncommitted)
- `CLAUDE.md` — Docker + env vars updated (uncommitted)
