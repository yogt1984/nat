# 1.1 Docker Compose Stack

## Status: DONE

## Goal

Dockerize the full NAT stack (redis, ingestor, api, alerts, web) with correct
multi-crate Rust builds and service wiring.

## Prerequisites

None — this is the foundation.

## Components

### Dockerfiles (multi-stage Rust builds)

All three Dockerfiles use the same pattern:

```dockerfile
# Build stage — full Rust workspace
FROM rust:1.75-bookworm AS builder
WORKDIR /app
COPY rust/Cargo.toml rust/Cargo.lock ./
COPY rust/ing-types ./ing-types        # dependency chain: types → features → ing
COPY rust/ing-features ./ing-features
COPY rust/ing ./ing
COPY rust/api ./api
RUN cargo build --release --bin <target>

# Runtime stage — minimal image
FROM debian:bookworm-slim
COPY --from=builder /app/target/release/<binary> /app/<binary>
```

**Critical fix applied:** Originally only `ing` and `api` crates were copied.
The workspace has 4 crates (`ing-types → ing-features → ing`, `api`) and builds
fail without all of them present.

### docker-compose.yml Services

| Service | Image | Port | Depends On |
|---------|-------|------|-----------|
| redis | redis:7-alpine | 6379 | — |
| ingestor | nat-ingestor | 8080 | redis |
| api | nat-api | 3000 | redis |
| alerts | nat-alerts | — | redis |
| web | nat-web | 3001 | api |

### Environment Variables (ingestor)

```yaml
environment:
  - REDIS_URL=redis://redis:6379
  - ING_DASHBOARD_ENABLED=true
  - ING_PROMETHEUS_ADDR=0.0.0.0:9090
```

`ING_PROMETHEUS_ADDR` override added in `rust/ing/src/config.rs` (line ~207):
```rust
if let Ok(val) = std::env::var("ING_PROMETHEUS_ADDR") {
    config.metrics.prometheus_addr = Some(val);
}
```

### Volume Mounts

```yaml
volumes:
  - ./config:/app/config:ro     # ing.toml, symbols.toml
  - ./data:/app/data             # Parquet output
```

## Verification

```bash
docker compose build ingestor api alerts   # all 3 compile
docker compose up -d                       # services start
docker compose ps                          # all running/healthy
docker compose logs ingestor | head -20    # no crash loops
```

## Files Modified

- `docker/Dockerfile.ingestor` — added ing-types, ing-features COPY
- `docker/Dockerfile.api` — same fix
- `docker/Dockerfile.alerts` — same fix
- `docker-compose.yml` — env vars, port mappings
- `rust/ing/src/config.rs` — ING_PROMETHEUS_ADDR env override (3 lines)
