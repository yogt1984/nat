# Requirements — Parquet Visualization & Validation Tooling for `nat`

Status: **proposed** (requirements analysis; implementation is a follow-up slice).
Owner: research tooling. Scope: read-only consumers of the existing feature parquet —
**no feature-vector / schema change is implied**.

---

## 1. Context & motivation

We can ingest 236+ features at 100 ms but have no fast, terminal-native way to *look at*
a parquet or *trust* it before feeding it downstream. Today the eyeball/verify loop is:
open Python, write ad-hoc pandas, hand-roll a plot. That is slow and unrepeatable.

This document specifies three CLI capabilities so a researcher can, without leaving the
terminal:

1. **Render** any slice of the feature parquet as a PNG at **1 min / 5 min / 15 min**
   granularity, and have `nat 15m` auto-open an "all features captured" snapshot.
2. **Validate** a single parquet file: `nat data validate path/to/file.parquet`.
3. Explore a **3D interactive mesh** of the data — a **feature-surface-over-time**
   (x = time bars, y = feature by category, z = normalized value).

This directly supports the dead-features / data-continuity work: the cloud-deploy 24h
coverage check (`docs/cloud_deployment/0_overview.md` Step 1) and the `/streak` depth
checks become **one command each** (`nat data validate <today's file>`, `nat viz render`).

### Design decisions (locked)
- **3D mesh = feature-surface-over-time** (Plotly `Surface`/`Mesh3d` → standalone HTML).
- **CLI = extend existing, no breakage.** Top-level `nat validate` already owns
  `skeptical`/`regression`, and `nat 15m` already runs the 15-min experiment, so:
  - PNG snapshots → a new parametrized `nat viz render --tf {1m,5m,15m}` under the
    existing `nat viz` group; `nat 15m viz` is upgraded to call it and auto-open.
  - File validation → `nat data validate <path>` (an optional positional on the existing
    command), **not** a new top-level `nat validate <path>`.
  - 3D mesh → a new `nat viz3d` (alias `nat mesh`).

---

## 2. Scope

**In scope**
- The three capabilities above and their CLI surface.
- Reusing the existing loaders, resampler, schema/quality validators, themed plotters,
  and Plotly 3D pattern (see §7).

**Out of scope**
- Live/streaming dashboards — already exist (`scripts/dashboard.py` :8050,
  `scripts/agent_dashboard.py` :8060, the `web/` Next.js app).
- Web-frontend (`web/`) changes.
- Any new feature computation or change to the Rust feature vector / Parquet schema.
- Trade-level (fills/P&L) visualization — covered by `scripts/trade_visualize.py` and
  `nat viz paper|portfolio`.

---

## 3. Data contract (inputs the tools consume)

Authoritative sources: `FEATURES.md`, `scripts/data/schema.py`, `rust/.../output/schema.rs`.

- **Layout:** `data/features/YYYY-MM-DD/YYYYMMDD_HHMMSS.parquet`, rotated **hourly**,
  zstd-compressed (~30–40 MB/file), ~108k rows/file (3 symbols × ~36k rows/symbol/hour).
- **Time column:** **`timestamp_ns`** — `int64`, **nanosecond** Unix epoch.
  Convert with `pd.to_datetime(df["timestamp_ns"], unit="ns")`.
- **Other metadata:** `symbol` (BTC/ETH/SOL), `sequence_id` (uint64, per-symbol monotonic).
- **Features:** 236+ columns across **14 base** categories (always computed) + **5 optional**
  categories (whale_flow, liquidation_risk, concentration, regime, cross-symbol; the schema
  also carries heatmap). Exact counts come from `data/schema.py` — **do not hardcode**.
- **NaN semantics:** optional categories are **NaN-padded** when unavailable. As of this
  writing the whale/liquidation/concentration columns are dead (100% NaN) in production
  until the wired ingestor deploys — tools **must treat all-NaN columns as "unavailable",
  not as an error**.

---

## 4. Functional requirements

### Capability A — Parquet → PNG snapshots at {1m, 5m, 15m}

- **FR-A1** `nat viz render` SHALL render a multi-panel PNG of feature values resampled to
  a chosen timeframe. CLI:
  ```
  nat viz render --tf {1m,5m,15m} --symbol {BTC,ETH,SOL}
                 [--date YYYY-MM-DD | --hours N]
                 [--features <category|vector|csv-list>]
                 [--open] [--output PATH]
  ```
- **FR-A2** Timeframe resampling SHALL use `cluster_pipeline.preprocess.aggregate_bars`.
  That function already accepts any pandas freq string (so `1min` works); the requirement
  is to add `"1min"` to its `TIMEFRAMES` registry and to the CLI `--tf` choices
  (`1m→1min`, `5m→5min`, `15m→15min`). Per-category aggregation rules are inherited
  unchanged (OHLC for price, sum for volume/counts, mean/std/slope for entropy, etc.).
- **FR-A3** Features SHALL be grouped into panels by category using
  `data/schema.py::BASE_FEATURES`/`OPTIONAL_FEATURES` (and optionally
  `cluster_pipeline.config.FEATURE_VECTORS` for named vectors). `--features` selects a
  single category, a named vector, or an explicit comma-list; default = **all features**
  (per-category panels + a feature heatmap), reusing the `scripts/15m_visualize.py` layout.
- **FR-A4** With `--open`, the tool SHALL open the produced PNG via the platform opener
  (`xdg-open` on Linux). Default (no `--open`) writes to disk and prints the path only.
- **FR-A5** Data selection: `--date` picks one calendar day's files; `--hours N` loads the
  last N hours from the latest available date (reuse `swarm.parquet_reader` semantics).
  Default when neither given = latest available date.
- **FR-A6** `nat 15m viz` SHALL be upgraded to invoke this renderer at `--tf 15m` on the
  latest data **with auto-open**, satisfying "`nat 15m` automatically opens a PNG
  representing all features captured." The existing `nat 15m` experiment
  (ingest→analyze→report) and its other subcommands MUST remain unchanged.
- **FR-A7** All-NaN (unavailable) optional columns SHALL be rendered as an explicit
  "no data" panel/cell, never raising. A requested `--date`/file with no rows SHALL fail
  with a clear, non-traceback error message and a nonzero exit code.

### Capability B — Single-file parquet validation

- **FR-B1** `nat data validate` SHALL accept an **optional positional path**:
  ```
  nat data validate [<path/to/file.parquet>] [--hours N] [--json]
  ```
  - With a path → validate **only that file**.
  - Without a path → **current behavior is unchanged** (validate recent data-dir; existing
    `--hours`).
- **FR-B2** Single-file validation SHALL report, reusing existing logic:
  - **Schema** — `data/schema.py::validate_columns` (missing base/optional, unexpected cols).
  - **Quality** — `data/schema.py::validate_quality` (per-column NaN rate, constant columns,
    row count, per-symbol counts).
  - **Continuity & ranges** — the gap (>5 s), post-warmup NaN-ratio, feature-range, and
    cross-symbol-consistency checks from `scripts/validate_data.py`, adapted to one file.
- **FR-B3** The command SHALL print a single **PASS / WARN / FAIL** verdict with the
  contributing reasons, and exit **nonzero on FAIL** (so it is usable in the cloud-deploy /
  CI coverage check). `--json` SHALL emit the same result as a machine-readable object.
- **FR-B4** A path that is missing, unreadable, or not a parquet SHALL produce a clear
  error (not a traceback) and a nonzero exit.

### Capability C — 3D interactive feature-surface-over-time

- **FR-C1** `nat viz3d` (alias `nat mesh`) SHALL produce an interactive 3D surface:
  ```
  nat viz3d --tf {1m,5m,15m} --symbol BTC
            [--date YYYY-MM-DD | --hours N]
            [--features <category|vector>]
            [--z {zscore,value}] [--open] [--output PATH]
  ```
- **FR-C2** Axes: **x = time bars** (resampled via `aggregate_bars`, FR-A2),
  **y = features** ordered/grouped by category, **z = value** normalized per-feature
  (`zscore` default; `value` for raw). Built as a Plotly `go.Surface` (regular grid) or
  `go.Mesh3d`, reusing the Plotly pattern in `cluster_pipeline/viz.py::plot_scatter_3d`.
- **FR-C3** Output SHALL be a **standalone, self-contained HTML** (`fig.write_html(...)`)
  that opens in a browser; `--open` opens it. The HTML MUST render without a running
  server. `--features` SHOULD be encouraged for legibility (a 236-row y-axis is dense).
- **FR-C4** All-NaN feature rows SHALL be dropped (with a note) so the surface is
  well-defined; an empty selection SHALL error clearly with a nonzero exit.

---

## 5. Non-functional requirements

- **NFR-1 Memory.** Loads SHALL use `load_parquet(..., columns=, max_memory_mb=)` and filter
  by `--symbol` early. Default single-day single-symbol load must stay well under ~500 MB
  (the swarm evaluator budget).
- **NFR-2 Dependencies.** No new runtime deps — matplotlib, seaborn, plotly, pandas,
  pyarrow are already in `requirements.txt`. Plotly is used **only** for Capability C.
- **NFR-3 Theme.** PNGs SHALL use the shared dark theme (`scripts/viz/features.py` `STYLE` +
  `COLORS`) and the house save convention: `dpi=150`, `bbox_inches="tight"`,
  `facecolor` preserved.
- **NFR-4 Determinism.** Same inputs → byte-stable selection/aggregation and stable output
  paths. No wall-clock in computed values; only in default output filenames.
- **NFR-5 Portability of `--open`.** A single shared helper SHALL encapsulate the opener
  (`xdg-open`/`open`/`start`) so all three commands behave identically and degrade to
  "printed path only" if no opener is available.
- **NFR-6 No side effects on data.** Tools are strictly read-only against `data/features/`;
  they write only under `reports/`.

---

## 6. CLI surface (summary)

| Command | Purpose | Key flags | Exit |
|---|---|---|---|
| `nat viz render` | PNG snapshot at 1m/5m/15m | `--tf --symbol --date/--hours --features --open --output` | 0 ok / ≠0 on no-data |
| `nat 15m viz` | all-features 15m PNG, auto-open | (delegates to `viz render --tf 15m --open`) | 0 / ≠0 |
| `nat data validate <path>` | validate one parquet file | `[<path>] --hours --json` | 0 PASS·WARN / ≠0 FAIL·error |
| `nat viz3d` / `nat mesh` | 3D feature-surface-over-time HTML | `--tf --symbol --date/--hours --features --z --open --output` | 0 / ≠0 |

Registration follows the existing `nat` argparse pattern:
`sub.add_parser(name).set_defaults(func=cmd_…)` with nested subparsers (see how
`nat data validate` and `nat viz paper` are wired in the `nat` script).

---

## 7. Reuse map (verified present — build on these, don't rebuild)

| Need | Existing module / symbol |
|---|---|
| Load parquet (symbols / date / columns / mem cap) | `scripts/cluster_pipeline/loader.py::load_parquet` (line 247) |
| Auto latest-date, N-hour load | `scripts/swarm/parquet_reader.py::read_evaluation_data` |
| Resample to 1m/5m/15m bars (per-category aggs) | `scripts/cluster_pipeline/preprocess.py::aggregate_bars` (line 72); add `"1min"` to `TIMEFRAMES` (line ~30) |
| Feature categories / names / vectors | `scripts/data/schema.py` (`BASE_FEATURES` l.20, `OPTIONAL_FEATURES` l.110, `ALL_COLUMNS` l.155); `cluster_pipeline/config.py::FEATURE_VECTORS` |
| Schema + quality validation | `scripts/data/schema.py::validate_columns` (l.158), `::validate_quality` (l.184) |
| Continuity / NaN / range / cross-symbol checks | `scripts/validate_data.py` |
| Multi-panel "all features" PNG layout | `scripts/15m_visualize.py` (2 pages × 6 panels incl. feature heatmap) |
| Themed time-series / heatmap / correlation plotters | `scripts/viz/{features,correlations,distributions,events,common}.py` (dark `STYLE`+`COLORS`) |
| Interactive 3D (Plotly) reference | `scripts/cluster_pipeline/viz.py::plot_scatter_3d` (l.162) |
| CLI registration pattern | `nat` (argparse, nested subparsers, `set_defaults(func=…)`) |
| Output dir / DPI convention | `reports/figures/`, `reports/smoke_test/`; `dpi=150`, `bbox_inches="tight"` |

---

## 8. Output artifacts & conventions

- PNG snapshots → **`reports/figures/snapshots/<symbol>_<tf>_<date>.png`** (DPI 150).
- 3D mesh → **`reports/figures/mesh/<symbol>_<tf>_<date>.html`** (self-contained).
- `--output PATH` overrides the default location for both.
- `--open` uses the shared opener helper (NFR-5).
- All outputs live under `reports/` (gitignored artifact area), never under `data/`.

---

## 9. Acceptance criteria & verification

Per `CLAUDE.md` (planted test before real data; real-parquet smoke before commit):

- **AC-1 Planted (red-first) test.** Synthesize a small parquet with the real column set,
  a known per-feature pattern, and an **injected NaN block + a >5 s time gap**. Assert:
  - (a) `viz render` produces a non-empty PNG and groups panels by category;
  - (b) `viz3d` writes HTML containing a surface trace whose dims = (n_bars × n_features
    selected);
  - (c) `nat data validate <synthetic>` returns **FAIL** and flags the injected gap + NaN,
    with nonzero exit; a clean synthetic returns PASS.
- **AC-2 Real-parquet smoke.** Run all three against the latest `data/features` date for
  BTC before any commit; PNG + HTML open; validate produces a sane verdict.
- **AC-3 CLI dispatch / no regression.** `nat viz render`, `nat viz3d`/`nat mesh`, and
  `nat data validate <path>` parse and route correctly; and the following are **unchanged**:
  `nat 15m`, `nat 15m viz` (now auto-opens), `nat validate skeptical|regression`,
  `nat data validate` (no positional), `nat data schema`.
- **AC-4 NaN/edge handling.** All-NaN optional columns render as "unavailable" (no crash);
  missing date / empty selection / bad path each give a clean error + nonzero exit.

---

## 10. Risks & open questions

- **R-1 y-axis density (Capability C).** A 236-feature surface is visually dense. Mitigation:
  default `viz3d` to a single category/vector and require explicit opt-in for "all".
- **R-2 1-min volume on long windows.** 1-min bars over many hours × 236 features can be
  heavy; rely on `--symbol`/`--features`/`--hours` and `max_memory_mb` (NFR-1).
- **R-3 `aggregate_bars` agg suffixes.** Bars emit `_mean/_std/_last/_open…` suffixed
  columns; the renderer must map back to base feature names for panel/axis labels
  (choose the representative aggregate, default `_mean`/`_last`).
- **OQ-1** Multi-symbol PNG (faceted) vs strict single-`--symbol`? Default single; faceting
  is a later enhancement.
- **OQ-2** Should `nat data validate <path>` accept a directory (validate all files within)?
  Default file-only; directory mode is a later enhancement.

---

## 11. Phasing (implementation follow-up, test-first)

1. **Shared plumbing** — `--tf` registry (`1min`), the opener helper (NFR-5), a small
   "load → resample → select features by category" helper over existing loaders.
2. **Capability B** (`nat data validate <path>`) — smallest, highest utility; wire AC-1(c).
3. **Capability A** (`nat viz render` + `nat 15m viz` upgrade) — reuse `15m_visualize` panels.
4. **Capability C** (`nat viz3d`/`nat mesh`) — Plotly surface + HTML export.
5. Planted tests precede each; real-parquet smoke before commit; conventional commits on a
   feat branch, `merge --no-ff` to master.
