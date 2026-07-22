# 2.1 Shared Ingestor Architecture

## Status: DONE

## Goal

Decouple data ingestion from evaluation so that 1 WebSocket connection produces
Parquet files consumed by N independent evaluator workers. This makes the swarm
100x cheaper than running N full NAT instances.

## Prerequisites

- Tier 1 complete (Docker stack running, metrics flowing)

## Architecture

```
                    ┌─────────────────┐
                    │   Hyperliquid   │
                    │   WebSocket     │
                    └────────┬────────┘
                             │ 1 connection
                    ┌────────▼────────┐
                    │    Ingestor     │
                    │  (existing)     │
                    │  Parquet output │
                    └────────┬────────┘
                             │ writes to shared volume
                    ┌────────▼────────┐
                    │   /data/features│
                    │   YYYY-MM-DD/   │
                    │   *.parquet     │
                    └──┬──┬──┬──┬──┬──┘
                       │  │  │  │  │  reads (read-only)
              ┌────────┘  │  │  │  └────────┐
              ▼           ▼  ▼  ▼           ▼
         ┌─────────┐ ┌─────────┐  ... ┌─────────┐
         │ Eval #1 │ │ Eval #2 │      │ Eval #N │
         │ config_1│ │ config_2│      │ config_N│
         └────┬────┘ └────┬────┘      └────┬────┘
              │           │                │
              ▼           ▼                ▼
         ┌─────────────────────────────────────┐
         │   Results DB (SQLite / PostgreSQL)  │
         └─────────────────────────────────────┘
```

## Key Design Decisions

### Why not N ingestor instances?

Each ingestor opens a WebSocket, subscribes to orderbook + trades for all
symbols, and computes 236 features every 100ms. Running N instances means:
- N WebSocket connections (API rate limit risk)
- N × feature computation (wasted CPU)
- N × Parquet writes (wasted disk I/O)

The shared approach: 1 ingestor writes Parquet once, N evaluators just read
and run algorithms — which is 100-1000x lighter.

### Data synchronization

Evaluators must handle:
- **Incomplete files:** The ingestor rotates hourly. The current-hour file is
  actively written. Evaluators should only read completed files, or use a
  file-watcher to detect rotation events.
- **Lag:** New evaluators starting mid-day need to process historical files
  from today before running live.

### Volume mount (Docker)

```yaml
# docker-compose.yml
services:
  ingestor:
    volumes:
      - parquet_data:/app/data/features

  evaluator-1:
    volumes:
      - parquet_data:/app/data/features:ro  # read-only
```

All evaluators share the same Docker volume, mounted read-only.

## Implementation Details

### Parquet reader utility

**New file:** `scripts/swarm/parquet_reader.py`

```python
def read_latest_parquet(data_dir: str, symbol: str, hours: int = 24):
    """Read the last N hours of Parquet data for a symbol."""
    # Glob data_dir/YYYY-MM-DD/{symbol}_*.parquet
    # Filter by date range
    # Concatenate into single DataFrame
    # Return sorted by timestamp
```

Reuse existing pattern from `scripts/analysis/` which already reads Parquet.

### File watcher (optional, for live eval)

```python
from watchdog.observers import Observer
# Watch for new .parquet files in data_dir
# Trigger re-evaluation when hourly rotation completes
```

## Verification

```bash
# Start ingestor, wait for 1 hour rotation
docker compose up -d ingestor
# Check Parquet file exists
ls data/features/$(date +%Y-%m-%d)/

# Start evaluator with read-only mount
docker compose run --rm evaluator-1 python scripts/swarm/evaluator.py
# Should read Parquet and produce results
```

## Files Created

- `scripts/swarm/parquet_reader.py` — shared Parquet reading utility
- `docker-compose.yml` — shared volume mount for evaluators
