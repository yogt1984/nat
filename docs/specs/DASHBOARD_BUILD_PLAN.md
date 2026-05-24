# Dashboard Build Plan — Task Breakdown

## Overview

Build a live experiment dashboard that shows data collection progress,
health metrics, and preliminary profiling results. Single page, auto-refreshing,
dark theme, no framework.

**Total: 12 tasks, each ≤ 45 minutes.**

---

## Task 1: State File Schema

**What**: Define the JSON schema that represents the entire dashboard state.
One file that everything reads from and writes to.

**Output**: `scripts/experiment/state.py`
- `ExperimentState` dataclass with nested sections (experiment, data, profiling, events)
- `save_state(state, path)` — write to JSON
- `load_state(path)` — read from JSON, return dataclass
- Default path: `data/experiment_state.json`

**Size**: ~80 lines
**Depends on**: Nothing

---

## Task 2: Data Counter

**What**: Function that scans `data/features/` and returns row count,
file count, disk size, per-symbol status, and last-flush timestamp.

**Output**: `scripts/experiment/metrics.py` → `collect_data_metrics(data_dir) -> DataMetrics`
- Count parquet files
- Sum row counts (read metadata only, no full load)
- Check each symbol has recent rows
- Return structured result

**Size**: ~60 lines
**Depends on**: Nothing (uses pyarrow metadata, no full file read)

---

## Task 3: Health Checker

**What**: Function that validates the most recent N hours of data.
Lightweight version of `validate_data_recent`.

**Output**: `scripts/experiment/metrics.py` → `check_health(data_dir, hours=1) -> HealthMetrics`
- NaN ratio (sample last hour only)
- Gap detection (timestamp continuity)
- Feature count matches expected (191)
- Returns: nan_ratio, n_gaps, longest_gap_s, features_ok

**Size**: ~70 lines
**Depends on**: Existing `cluster_pipeline.loader.load_parquet`

---

## Task 4: Quick Profiler

**What**: Function that runs a fast profiling pass on whatever bars exist.
Subset of the full `profile()` — just structure test + GMM + quality gates.

**Output**: `scripts/experiment/profiler.py` → `quick_profile(data_dir) -> ProfilingSnapshot`
- Load data, aggregate to 15-min bars
- If bars < 50: return "insufficient" status
- If bars 50-100: Hopkins test only
- If bars > 100: full profile + validate
- Returns: k, silhouette, ARI, states summary, transition matrix, gate results, verdict

**Size**: ~100 lines
**Depends on**: Existing `cluster_pipeline` modules (loader, preprocess, derivatives, hierarchy, validate)

---

## Task 5: Event Log

**What**: Simple append-only event log. Each event is a timestamp + type + message.
Stored as JSON lines in `data/experiment_events.jsonl`.

**Output**: `scripts/experiment/events.py`
- `log_event(type, message)` — append one line
- `read_events(n=50)` — read last N events
- Event types: `started`, `stopped`, `health`, `profiling`, `error`

**Size**: ~40 lines
**Depends on**: Nothing

---

## Task 6: Monitor Loop

**What**: The background process that runs every 60 seconds, collects metrics,
and periodically runs profiling.

**Output**: `scripts/experiment/monitor.py` (runnable as `python -m scripts.experiment.monitor`)
- Every 60s: collect data metrics + health check → update state file
- Every 6h (or when bars crosses 50/100/200 thresholds): run quick profiler → update state file
- On each update: log event if something changed (new milestone, health issue, profiling result)
- Graceful shutdown on SIGTERM/SIGINT

**Size**: ~120 lines
**Depends on**: Tasks 1-5

---

## Task 7: API Server

**What**: FastAPI app with 3 endpoints. No auth, read-only.

**Output**: `scripts/experiment/server.py`
- `GET /` — serves the HTML dashboard (static file)
- `GET /api/state` — returns `data/experiment_state.json` contents
- `GET /api/events` — returns last 50 events from JSONL

**Size**: ~40 lines
**Depends on**: Task 1 (state schema), Task 5 (event log)

---

## Task 8: Dashboard HTML — Layout and CSS

**What**: The static HTML file with dark theme styling. No data yet,
just the skeleton with section headers, progress bar placeholder,
and table structure.

**Output**: `scripts/experiment/static/dashboard.html`
- Dark background (#1a1a2e), monospace font
- Sections: header, data collection, health, profiling, quality gates, events
- Progress bar (CSS-only, width set by JS later)
- Table styles for states and transition matrix
- Responsive single column, max-width 800px

**Size**: ~120 lines (HTML + CSS)
**Depends on**: Nothing

---

## Task 9: Dashboard JS — Polling and Rendering

**What**: JavaScript that polls `/api/state` every 60 seconds and fills
in the HTML skeleton with live data.

**Output**: Added to `dashboard.html` as inline `<script>`
- `fetchState()` — GET /api/state, parse JSON
- `renderData(state)` — fill in row count, bars, disk, rate, progress bar width
- `renderHealth(state)` — fill in NaN, gaps, per-symbol indicators
- `renderProfiling(state)` — fill in k, silhouette, states table, transition matrix
- `renderGates(state)` — fill in Q1/Q2/Q3 with ✓/✗/⚠ and verdict
- `renderEvents(state)` — fill in event log list
- Auto-refresh every 60s, show "last updated" timestamp

**Size**: ~150 lines
**Depends on**: Task 8 (HTML skeleton), Task 7 (API endpoints)

---

## Task 10: Makefile Integration

**What**: Add targets that start/stop the monitor and server alongside the ingestor.

**Output**: Modifications to `Makefile` + `scripts/run_experiment.py`
- `make exp_start` now also starts monitor + server in same tmux (3 panes)
- `make exp_stop` stops all three
- `make exp_dashboard` opens `http://localhost:8050` (or just prints URL)
- `make exp_analyze` stops monitor, runs final full profiling, updates state with final verdict

**Size**: ~30 lines Makefile + ~40 lines run_experiment.py changes
**Depends on**: Tasks 6, 7

---

## Task 11: Tunnel Setup

**What**: Expose dashboard externally via Cloudflare tunnel so you can check
from your phone or another machine.

**Output**: Modifications to `Makefile`
- `make exp_tunnel` — runs `cloudflared tunnel --url http://localhost:8050`
- Prints the public URL

**Size**: ~5 lines
**Depends on**: Task 7 (server running)

---

## Task 12: Test and Verify

**What**: Start the full stack locally, verify everything connects.

**Steps**:
- Start ingestor (collects for 2 minutes)
- Start monitor (should write state file after 60s)
- Start server (should serve dashboard)
- Open browser, confirm dashboard shows live data
- Stop ingestor, confirm dashboard shows "STOPPED"
- Run `make exp_analyze`, confirm final verdict appears

**Output**: Documented verification steps in this file (checked off)
**Depends on**: All previous tasks

---

## Execution Order

```
Independent (can be done in any order):
  Task 1: State schema
  Task 2: Data counter
  Task 3: Health checker
  Task 5: Event log
  Task 8: HTML layout

Sequential (depends on above):
  Task 4: Quick profiler     (needs existing pipeline modules)
  Task 6: Monitor loop       (needs 1, 2, 3, 4, 5)
  Task 7: API server         (needs 1, 5)
  Task 9: Dashboard JS       (needs 7, 8)
  Task 10: Makefile          (needs 6, 7)
  Task 11: Tunnel            (needs 7)
  Task 12: Verify            (needs all)
```

**Dependency graph:**

```
[1 State] ──────┬──→ [6 Monitor] ──→ [10 Makefile] ──→ [12 Verify]
[2 Counter] ────┤                          ↑
[3 Health] ─────┤                          │
[4 Profiler] ───┘                          │
[5 Events] ────────→ [7 Server] ───────────┘
[8 HTML] ──────────→ [9 JS] ──→ [10]
                     [11 Tunnel]
```

---

## File Structure When Complete

```
scripts/
  experiment/
    __init__.py          # empty
    state.py             # Task 1: ExperimentState dataclass + save/load
    metrics.py           # Task 2+3: collect_data_metrics + check_health
    profiler.py          # Task 4: quick_profile
    events.py            # Task 5: log_event + read_events
    monitor.py           # Task 6: background loop (entry point)
    server.py            # Task 7: FastAPI app
    static/
      dashboard.html     # Task 8+9: HTML + CSS + JS (single file)
```

---

## Constraints

- No npm, no webpack, no React. Single HTML file with inline CSS and JS.
- No database. State is one JSON file. Events is one JSONL file.
- No WebSocket to browser. Simple polling every 60 seconds is sufficient.
- Monitor and server are separate processes (can restart one without the other).
- All profiling uses existing `cluster_pipeline` modules — no reimplementation.
- Dashboard works without internet (no CDN dependencies except KaTeX if math is needed later).
