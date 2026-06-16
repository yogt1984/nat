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

1. **Page through** the feature parquet as PNGs at **1 min / 5 min / 15 min** granularity —
   `nat viz render --tf 5m 1` shows the first 5-min window, `… 2` the 5–10 min window, etc.
   — and have `nat 15m` auto-open an "all features captured" overview.
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
  - PNG snapshots → a new parametrized `nat viz render --tf {1m,5m,15m} [INDEX]` under the
    existing `nat viz` group; `nat 15m viz` is upgraded to call it and auto-open.
  - File validation → `nat data validate <path>` (an optional positional on the existing
    command), **not** a new top-level `nat validate <path>`.
  - 3D mesh → a new `nat viz3d` (alias `nat mesh`).
- **Paginated viewer (`--tf` + optional `INDEX`).** `--tf` is the time granularity and an
  optional 1-based positional `INDEX` pages through the data: **no index** = whole-day
  overview (all features as `--tf` bars), **index N** = zoom into the Nth `--tf`-width
  window at tick resolution. Index is **data-relative within the day** (page 1 = first
  available tick). Full semantics in §3b; applies to both `nat viz render` and `nat viz3d`.

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

## 3b. Interval / pagination model (shared by Capabilities A & C)

The ergonomic core of the viewer. `--tf {1m,5m,15m}` is the **time granularity**, with a
**dual role** keyed on an optional **1-based positional `INDEX`**:

| Invocation | Mode | What renders |
|---|---|---|
| `nat viz render --tf 5m` | **Overview** (no index) | the **whole day** resampled to 5-min bars — all features ("all features captured") |
| `nat viz render --tf 5m 1` | **Page 1** | the **first** 5-min window, at tick resolution |
| `nat viz render --tf 5m 2` | **Page 2** | the **5–10 min** window |
| `nat viz render --tf 5m N` | **Page N** | the Nth 5-min window |

**Anchor — data-relative within the day.** Page 1 begins at the **first available tick** of
the selected day; page `N` spans `[t0 + (N-1)·tf, t0 + N·tf)` where `t0` = that first tick.
So "the first 5 min" is literally page 1 — there are **no empty leading pages**. The day is
chosen by `--date` (default: latest available day). *Trade-off accepted:* a given page index
maps to a different wall-clock window if ingestion start time moves day-to-day; clock-aligned
indexing is a documented future option (`--align clock`, see §10).

**Page count.** For a day with span `T` from `t0` to the last tick, there are
`ceil(T / tf)` pages; the **last page may be partial** (data ends mid-window) and SHALL be
rendered with a "partial page" note rather than padded.

**Errors.** `INDEX` is 1-based: `INDEX ≤ 0` → usage error; `INDEX >` page count → a clear
error naming the max valid page (e.g. *"page 300 > 288 5-min pages for BTC on 2026-06-15"*)
and a nonzero exit.

**Optional ergonomic extensions** (mention; not required for v1):
- Negative index (`-1` = last page) — note the argparse leading-dash caveat (accept via
  `--page -1` or a `--` separator).
- Range form `N-M` (or `N:M`) to render a multi-page contact sheet.

---

## 4. Functional requirements

### Capability A — Paginated PNG viewer at {1m, 5m, 15m}

- **FR-A1** `nat viz render` SHALL render a multi-panel PNG of feature values, in one of two
  modes selected by the optional positional `INDEX` (§3b). CLI:
  ```
  nat viz render --tf {1m,5m,15m} [INDEX] --symbol {BTC,ETH,SOL}
                 [--date YYYY-MM-DD]
                 [--features <category|vector|csv-list>]
                 [--open] [--output PATH]
  ```
- **FR-A2 (overview mode, no `INDEX`)** The tool SHALL resample the **whole selected day** to
  `--tf` bars via `cluster_pipeline.preprocess.aggregate_bars` and render the multi-panel
  "all features" overview. `aggregate_bars` already accepts any pandas freq string (so
  `1min` works); the requirement is to register `"1min"` in its `TIMEFRAMES` table and map
  the CLI `--tf` choices (`1m→1min`, `5m→5min`, `15m→15min`). Per-category aggregation rules
  are inherited unchanged (OHLC for price, sum for volume/counts, mean/std/slope for entropy).
- **FR-A3 (page mode, `INDEX = N`)** The tool SHALL slice the **Nth `--tf`-width window** per
  the data-relative anchor in §3b and render the feature time-series **at tick resolution**
  within that window, **auto-decimating** to a bounded plot-point budget (target ≈1–2k points
  per feature) when the window is dense (e.g. 15 min ≈ 9k ticks). Out-of-range / non-positive
  `INDEX` and partial last pages behave exactly as specified in §3b.
- **FR-A4** Features SHALL be grouped into panels by category using
  `data/schema.py::BASE_FEATURES`/`OPTIONAL_FEATURES` (and optionally
  `cluster_pipeline.config.FEATURE_VECTORS` for named vectors). `--features` selects a
  single category, a named vector, or an explicit comma-list; default = **all features**
  (per-category panels + a feature heatmap), reusing the `scripts/15m_visualize.py` layout.
  This grouping applies identically in overview and page modes.
- **FR-A5** With `--open`, the tool SHALL open the produced PNG via the shared opener helper
  (`xdg-open` on Linux; NFR-5). Default (no `--open`) writes to disk and prints the path only.
- **FR-A6** Day selection: `--date YYYY-MM-DD` picks the calendar day whose files are loaded
  and paginated; default = **latest available day**. (`--hours` is intentionally dropped from
  this command — the `INDEX` page supersedes it as the windowing mechanism.)
- **FR-A7** `nat 15m viz` SHALL be upgraded to invoke this renderer in **overview mode** at
  `--tf 15m` (no index) on the latest day **with auto-open**, satisfying "`nat 15m`
  automatically opens a PNG representing all features captured." The existing `nat 15m`
  experiment (ingest→analyze→report) and its other subcommands MUST remain unchanged.
- **FR-A8** All-NaN (unavailable) optional columns SHALL be rendered as an explicit
  "no data" panel/cell, never raising. A requested `--date` with no rows (or an empty page)
  SHALL fail with a clear, non-traceback error message and a nonzero exit code.

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

- **FR-C1** `nat viz3d` (alias `nat mesh`) SHALL produce an interactive 3D surface, using the
  **same `--tf` + `INDEX` paginated model** as Capability A (§3b):
  ```
  nat viz3d --tf {1m,5m,15m} [INDEX] --symbol BTC
            [--date YYYY-MM-DD]
            [--features <category|vector>]
            [--z {zscore,value}] [--open] [--output PATH]
  ```
- **FR-C2** Axes: **x = time** — day bars (overview, no `INDEX`) or the ticks within page
  `N` (page mode); **y = features** ordered/grouped by category; **z = value** normalized
  per-feature (`zscore` default; `value` for raw). Built as a Plotly `go.Surface` (regular
  grid) or `go.Mesh3d`, reusing the Plotly pattern in
  `cluster_pipeline/viz.py::plot_scatter_3d`.
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

| Command | Purpose | Key args / flags | Exit |
|---|---|---|---|
| `nat viz render` | paged PNG viewer at 1m/5m/15m | `--tf [INDEX] --symbol --date --features --open --output` | 0 ok / ≠0 on no-data·bad-index |
| `nat 15m viz` | all-features 15m overview, auto-open | (delegates to `viz render --tf 15m --open`) | 0 / ≠0 |
| `nat data validate <path>` | validate one parquet file | `[<path>] --hours --json` | 0 PASS·WARN / ≠0 FAIL·error |
| `nat viz3d` / `nat mesh` | 3D feature-surface-over-time HTML | `--tf [INDEX] --symbol --date --features --z --open --output` | 0 / ≠0 |

`INDEX` is the optional 1-based page positional (§3b): omit for the whole-day overview, give
`N` to zoom into the Nth `--tf`-width window.

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

- PNG snapshots → **`reports/figures/snapshots/<symbol>_<tf>_<date>[_pN].png`** (DPI 150) —
  the `_pN` page suffix is present in page mode, absent for the overview.
- 3D mesh → **`reports/figures/mesh/<symbol>_<tf>_<date>[_pN].html`** (self-contained).
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
- **AC-3 CLI dispatch / no regression.** `nat viz render` (with and without `INDEX`),
  `nat viz3d`/`nat mesh`, and `nat data validate <path>` parse and route correctly; and the
  following are **unchanged**: `nat 15m`, `nat 15m viz` (now auto-opens), `nat validate
  skeptical|regression`, `nat data validate` (no positional), `nat data schema`.
- **AC-4 Pagination (planted).** On a synthetic day whose first tick is a known
  **non-midnight** time `t0`: assert `--tf 5m 1` covers `[t0, t0+5m)`, `--tf 5m 2` covers
  `[t0+5m, t0+10m)`; an out-of-range `INDEX` errors with the max-page message + nonzero exit;
  the final partial page renders with a "partial" note; and no-`INDEX` yields the full-day
  overview. Output filenames carry `_pN` only in page mode.
- **AC-5 NaN/edge handling.** All-NaN optional columns render as "unavailable" (no crash);
  missing date / empty page / bad path each give a clean error + nonzero exit.

---

## 10. Risks & open questions

- **R-1 y-axis density (Capability C).** A 236-feature surface is visually dense. Mitigation:
  default `viz3d` to a single category/vector and require explicit opt-in for "all".
- **R-2 Whole-day overview volume.** A full day × 236 features is heavy in overview mode;
  it is bounded by resampling to `--tf` bars (a 1-min overview of a 24 h day ≈ 1440 bars) plus
  `--symbol`/`--features` and `max_memory_mb` (NFR-1). Page mode is naturally bounded by `tf`.
- **R-3 `aggregate_bars` agg suffixes.** Overview bars emit `_mean/_std/_last/_open…` suffixed
  columns; the renderer must map back to base feature names for panel/axis labels (choose the
  representative aggregate, default `_mean`/`_last`). Page mode plots raw tick columns, so no
  suffix mapping is needed there.
- **R-4 Page→wall-clock drift.** Because the index anchor is data-relative (§3b), the same
  page index maps to a different wall-clock window if the day's ingestion start moves. Accepted
  for ergonomics; `--align clock` (OQ-3) is the reproducible alternative.
- **OQ-1** Multi-symbol PNG (faceted) vs strict single-`--symbol`? Default single; faceting
  is a later enhancement.
- **OQ-2** Should `nat data validate <path>` accept a directory (validate all files within)?
  Default file-only; directory mode is a later enhancement.
- **OQ-3** Add `--align clock` for wall-clock-aligned pages (and negative / range `INDEX`)?
  Deferred to a later ergonomic pass; v1 ships data-relative single-page indexing.

---

## 11. Phasing (implementation follow-up, test-first)

1. **Shared plumbing** — `--tf` registry (`1min`), the opener helper (NFR-5), and the
   **pagination helper** (§3b): "load day → compute page bounds from `t0` & `tf` → slice page
   N (or whole day) → select features by category" over existing loaders.
2. **Capability B** (`nat data validate <path>`) — smallest, highest utility; wire AC-1(c).
3. **Capability A** (`nat viz render` overview + page modes, `+ nat 15m viz` upgrade) — reuse
   `15m_visualize` panels for overview, tick time-series for pages.
4. **Capability C** (`nat viz3d`/`nat mesh`) — Plotly surface + HTML export, same pagination.
5. Planted tests precede each; real-parquet smoke before commit; conventional commits on a
   feat branch, `merge --no-ff` to master.
