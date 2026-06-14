# 1.3 Production Hardening

## Status: DONE

## Goal

Secure the stack for always-on cloud deployment: HTTPS access, persistent state
in PostgreSQL, proper auth, and automated backups.

## Prerequisites

- [1.1 Docker Stack](1_1_docker_stack.md) — DONE
- [1.2 Prometheus + Grafana](1_2_prometheus_grafana.md) — DONE

## Components

### A. Caddy Reverse Proxy (HTTPS)

**Why:** Grafana and the API should not be exposed over plain HTTP on a public
server. Caddy auto-provisions Let's Encrypt TLS certificates.

**New file:** `docker/Caddyfile`

```
grafana.{$DOMAIN} {
    reverse_proxy grafana:3000
}

api.{$DOMAIN} {
    reverse_proxy api:3000
}

dashboard.{$DOMAIN} {
    reverse_proxy ingestor:8080
}
```

**docker-compose addition:**

```yaml
caddy:
  image: caddy:2-alpine
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./docker/Caddyfile:/etc/caddy/Caddyfile:ro
    - caddy_data:/data
  environment:
    - DOMAIN=${DOMAIN:-localhost}
  depends_on:
    - grafana
    - api
```

### B. PostgreSQL for State Persistence

**Why:** Agent state, pipeline state, and swarm results are currently in JSON
files. PostgreSQL gives durability, querying, and multi-service access.

**docker-compose addition:**

```yaml
postgres:
  image: postgres:16-alpine
  ports:
    - "5432:5432"
  environment:
    - POSTGRES_DB=nat
    - POSTGRES_USER=nat
    - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-nat_dev}
  volumes:
    - postgres_data:/var/lib/postgresql/data
```

**Migration path:**
1. Add `psycopg2` to Python deps
2. Migrate `data/agent/agent_state.json` → `agent_state` table
3. Migrate `data/pipeline_state.json` → `pipeline_state` table
4. Keep JSON files as fallback for local dev

### C. Grafana Auth Hardening

**Current:** Anonymous viewer access, admin password = `nat`.

**Target:**
- Set `GF_SECURITY_ADMIN_PASSWORD` from `.env` file
- Disable anonymous access on production
- Keep anonymous access on local dev (via env toggle)

```yaml
environment:
  - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-nat}
  - GF_AUTH_ANONYMOUS_ENABLED=${GRAFANA_ANON:-true}
```

### D. Automated Backup

**Cron job** (host-level or Docker-based):

```bash
# Daily: dump PostgreSQL + copy Parquet index
pg_dump -h localhost -U nat nat | gzip > /backup/nat_$(date +%Y%m%d).sql.gz
# Weekly: prune backups older than 30 days
find /backup -name "nat_*.sql.gz" -mtime +30 -delete
```

Prometheus data is expendable (re-scraped). Grafana dashboards are in git
(provisioned from files, not DB).

### E. `.env` File Template

**New file:** `.env.example`

```bash
DOMAIN=nat.example.com
POSTGRES_PASSWORD=<strong-password>
GRAFANA_PASSWORD=<strong-password>
TELEGRAM_BOT_TOKEN=<token>
TELEGRAM_CHAT_ID=<chat-id>
REDIS_URL=redis://redis:6379
```

## Verification

```bash
# HTTPS works
curl https://grafana.${DOMAIN}/api/health

# PostgreSQL accessible
docker compose exec postgres psql -U nat -c "SELECT 1"

# Backups running
ls -la /backup/nat_*.sql.gz
```

## Files Created/Modified

- `docker/Caddyfile` — NEW
- `.env.example` — NEW
- `docker-compose.yml` — add caddy, postgres services
- Python scripts — psycopg2 state persistence (multiple files)
