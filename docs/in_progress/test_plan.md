# NAT Manual Test Plan — Terminal Connectivity & Visualization

**Scope:** the human-only checks that the automated suites cannot cover. The repo has heavy automated
coverage (~260 `nat` commands, 118 Python test files, ~55 Rust test modules) but it is almost
entirely synthetic-fixture unit testing. Three things only a human can confirm:

1. **Parquet viewing & validation tooling** *(newest — test first)* — do `nat data validate <file>`,
   `nat viz render`, and `nat viz3d`/`nat mesh` dispatch, validate, and *render* correctly on real
   parquet? (Spec: `docs/requirements/parquet_viz_validation.md`.)
2. **Terminal connectivity** — is every implemented capability actually reachable through `nat`, and
   does each command dispatch without crashing?
3. **Visualization correctness** — do the plots, the nightly HTML report, the web dashboards, and the
   terminal UIs actually *render correctly* (not merely "the function ran")?

**Out of scope (covered elsewhere — do not duplicate):**
- Docker stack health, metrics flow → `docs/cloud_deployment/1_4_testing_verification.md`
- ML-algorithm interface conformance → `docs/research/new/ml_specs/14_verification_checklist.md`
- Live WS ingestion→parquet, Telegram delivery, daemon lifecycle, model-serving latency → not yet
  documented; a future runbook.

**How to use:** work top to bottom, type each command, compare against *Expected*, tick the box.
Record each run in the sign-off log. All commands are run from the repo root (`/home/onat/nat`).

---

## Sign-off log

| Date | Tester | git SHA (`git rev-parse --short HEAD`) | Section A | Section B | Section C | Notes |
|------|--------|----------------------------------------|-----------|-----------|-----------|-------|
|      |        |                                        | ☐ pass    | ☐ pass    | ☐ pass    |       |
|      |        |                                        | ☐ pass    | ☐ pass    | ☐ pass    |       |

---

## Section A — Parquet viewing & validation tooling

Goal: a human confirms the newest data-inspection commands both **dispatch** and produce **correct
output** (a verdict, a PNG, an interactive surface) on *real* parquet — neither of which the planted
unit tests assert. Spec + acceptance criteria: `docs/requirements/parquet_viz_validation.md`.
Run from repo root; pick a real recent file first:

```bash
F=$(ls -t data/features/*/*.parquet | head -1); echo "$F"
```

### A1. Single-file validation — `nat data validate`

- [ ] `nat data validate "$F"` → per-check report ending in **PASS / WARN / FAIL**; `echo $?` matches
  (0 for PASS/WARN, **nonzero on FAIL**). On current production data **FAIL is expected** (dead
  optional columns → NaN Ratio; real gaps → Continuity). Hard checks = integrity/continuity/NaN/
  sequence; soft (ranges/cross-symbol/rate) only warn.
- [ ] `nat data validate "$F" --json` → **clean JSON on stdout** (no progress prose). Has `verdict`,
  `passed`, and a `checks` array with per-check `hard` flags. Pipe through
  `python3 -c "import json,sys; json.load(sys.stdin)"` → no error.
- [ ] `nat data validate` (no path) → directory-mode behaviour **unchanged** (scans the feature dir).
- [ ] `nat data validate /tmp/nope.parquet` → clean error ("Path does not exist"), exit 1, **no
  traceback**.

### A2. Paged PNG viewer — `nat viz render`

Writes PNGs to `reports/figures/snapshots/`. Open each: non-empty, dark GitHub theme, correct
title/axes. `--tf` is the granularity/page width; an optional **1-based index** pages the day
(data-relative: page 1 = first available tick).

- [ ] `nat viz render --tf 15m --symbol BTC` → **overview** (whole day, all-features curated panels;
  basic + advanced PNGs). Title shows the day.
- [ ] `nat viz render --tf 5m 1 --symbol BTC` → **page 1** = first 5-min window (filename `…_w01_…`);
  `nat viz render --tf 5m 2` → the 5–10 min window.
- [ ] `nat viz render --tf 5m 9999 --symbol BTC` → clean error "page 9999 out of range; N 5min
  page(s) available", exit 1.
- [ ] `nat viz render --tf 5m 1 --features flow --symbol BTC` → a **per-feature panel grid** of just
  the `flow` category (filename `feat_…`); `--features raw_midprice,raw_spread` → exactly those two
  panels; capped to top-N by variance (`--max-features`, default 16) when a category is large.
- [ ] `nat viz render --tf 5m 1 --features nope_xyz` → "matched no columns", exit 1.
- [ ] `nat viz render --tf 15m --open` → produced PNG opens in the default viewer (or prints the path
  if headless — no crash).
- [ ] `nat viz render --last 15m --symbol BTC` → freshest-readable tail: prints the analyzed window
  + its **age** ("Last 15m: …–… UTC (15.0 min, N min old)"; "current hour still buffering" when the
  current hour hasn't rotated) and writes the snapshot — does NOT beat the ~1h readable lag.
- [ ] A final partial window is flagged "— partial" in the title — expected, not a defect.

### A3. Interactive 3D feature-surface — `nat viz3d` / `nat mesh`

Writes self-contained HTML to `reports/figures/mesh/`.

- [ ] `nat viz3d --tf 15m --symbol BTC` (and the alias `nat mesh --tf 15m --symbol BTC`) → writes
  `<sym>_<tf>_<date>.html`; open it: a 3D **surface** renders, rotatable/zoomable, x = time,
  y = features (grouped by category), z = value; colourbar present; dark template.
- [ ] **Self-contained / offline:** open with no network — surface still renders (Plotly JS embedded).
  `grep -c "Plotly.newPlot" reports/figures/mesh/*.html` → ≥ 1.
- [ ] `nat viz3d --tf 5m 2 --features entropy --symbol BTC` → page 2; y-axis scoped to entropy
  features (a "capped" note prints if > `--max-features`).
- [ ] `nat viz3d --tf 5m 9999 --symbol BTC` → out-of-range error, exit 1.
- [ ] If `plotly` is absent → clean message ("plotly is required … pip install plotly"), exit 1
  (**not** a traceback).

### A4. 15-minute snapshot auto-open — `nat 15m viz`

- [ ] `nat 15m viz --symbol BTC` → renders the latest-experiment all-features snapshot and
  **auto-opens** the PNG(s). **Quiet by design**: output is *only* the two `Saved:` PNG paths (no
  banner/logs/summary).
- [ ] `nat 15m viz --symbol BTC --no-open` → renders (the two `Saved:` lines) but does **not** open.

---

## Section B — Terminal connectivity

Goal: confirm the full command surface is reachable and dispatches, and that no implemented
capability is stranded outside the CLI.

### B1. Enumerate the surface

- [ ] `nat --json commands | python3 -c "import json,sys; print(len(json.load(sys.stdin)['commands']))"`
  → **Expected:** prints `260` (or higher if commands were added). The JSON form is
  `nat --json commands` (global `--json` flag *before* the subcommand); `nat commands --json` does
  **not** emit JSON.
- [ ] `nat commands` → **Expected:** human-readable list, header reads `Available Commands (260)`.
- [ ] `nat help` → **Expected:** curated grouped help renders, exit code 0, no Python traceback.

### B2. Unwired-capability scan (documented heuristic)

`nat` invokes most subcommands by shelling out to a `scripts/…` file. This snippet lists scripts that
**define their own argparse CLI but are never referenced by the `nat` dispatcher** — i.e. capability
that exists but may be unreachable from the terminal.

```bash
# from repo root — prints candidate unwired scripts
mapfile -t cli_scripts < <(grep -rl "argparse.ArgumentParser" scripts --include="*.py" \
  | sed 's|^scripts/||;s|\.py$||' | sort -u)
for s in "${cli_scripts[@]}"; do
  dotted="scripts.${s//\//.}"; stem="${s##*/}"
  grep -qE "scripts/${s}\.py|${dotted}\b|\b${stem}\.py" nat || echo "scripts/${s}.py"
done | sort
```

- [ ] Run the snippet. **Expected:** ~40–45 candidates. This heuristic **intentionally
  over-reports** — many candidates are wired *indirectly* (a command's backend called via
  `import`, or invoked by another script) and are fine. Treat the output as a triage list, not a
  defect list. For any candidate, confirm reachability with `nat help` / by grepping `nat` for the
  stem before concluding it is stranded.
- [ ] **Confirmed-stranded baseline** (0 references in `nat` as of this writing — user-facing
  capability with no CLI entry). Verify this set is unchanged; investigate any *new* additions:
  - ML model trainers: `train_momentum.py`, `train_meta_labeling.py`, `train_mean_reversion.py`,
    `train_regime_lgbm.py`, `build_meta_training_data.py`
  - ML ops: `ml_health_check.py`, `ml_rollback.py`
  - Validation/gates: `evaluate_wave1_gate.py`, `evaluate_wave2_gate.py`, `oos_validate.py`,
    `check_data_sufficiency.py`, `check_deferred_triggers.py`
  - Terminal UI: `oos_terminal.py`

  > Note: `nat model train` exists but routes to a *different* code path than the per-algorithm
  > trainers above — those four are reachable only by direct `python scripts/train_*.py`.

### B3. Doc / implementation consistency

- [ ] `nat test agent -h` → **Expected (current reality):** **fails** — `test agent` is *not* in the
  dispatcher, even though `CLAUDE.md` advertises it ("350+ tests"). Record as a known doc/impl
  mismatch. (Agent tests run via `pytest scripts/tests/test_agent_*.py`.) If a future build wires it,
  flip this expectation.
- [ ] Spot-check 3–4 other commands named in `CLAUDE.md` / `README` actually dispatch
  (e.g. `nat test`, `nat test validate`, `nat oos30 -h`, `nat algorithm evaluate -h`).

### B4. Per-group dispatch smoke

For each major group, run the representative command. **Expected for every row:** output
renders / JSON parses, exit 0, **no argparse error or Python traceback**. Pure-compute/JSON commands
are safe to run for real; **daemon / live / destructive** commands are checked with `-h` and their
`status` subcommand *only* — never start them from this checklist.

| Group | Command to type | Kind |
|-------|-----------------|------|
| system | `nat --json status` | read (global `--json` is a **prefix** flag — `nat status --json` errors) |
| doctor | `nat doctor` | read (ingestion preflight: data-dir ownership/binary/disk) |
| commands | `nat --json commands >/dev/null` | read |
| config | `nat config show` | read |
| algorithm | `nat algorithm list` | read |
| process | `nat process list` | read |
| data | `nat data validate -h` | read |
| viz render | `nat viz render -h` | read |
| viz3d / mesh | `nat viz3d -h` / `nat mesh -h` | read |
| gauntlet | `nat gauntlet report` | read |
| nightly | `nat nightly report` | read |
| reports | `nat reports latest` | read |
| experiment | `nat experiment list` | read |
| cluster | `nat cluster -h` | read |
| backtest | `nat backtest -h` | read |
| swarm | `nat swarm status` | read |
| evolve | `nat evolve status` | read |
| agent | `nat agent status` + `nat agent -h` | daemon → status/-h only |
| mf-agent | `nat mf-agent status` | daemon → status only |
| macro-agent | `nat macro-agent status` | daemon → status only |
| meta-agent | `nat meta-agent status` | daemon → status only |
| discovery | `nat discovery status` | daemon → status only |
| it-engine | `nat it-engine status` | daemon → status only |
| docker | `nat docker ps` | needs docker → -h if absent |
| start/stop | `nat start -h` / `nat stop -h` | destructive → -h only |
| alg1 live | `nat alg1 -h` | **places live orders** → -h only |

- [ ] All read rows produced clean output.
- [ ] All daemon/destructive rows showed help / status without launching anything.

---

## Section C — Visualization correctness

Goal: a human confirms the visual output is actually correct — axes, legends, colours, layout, live
updates — none of which the unit tests assert.

### C1. Terminal UIs

- [ ] `nat monitor` → **Expected (redefined):** the live **feature probe** (`show_features`) — its
  own WebSocket, prints instantaneous feature values at ~10 Hz, **persists nothing**. Connects,
  shows the refreshing PRICE/TRADE-ACTIVITY display (values fill in after ~warmup), `Ctrl-C` exits.
  Works independent of the persistent ingestor; needs no Redis. (`--hz`, `-s/--symbol`.)
- [ ] `nat monitor tui` → **Expected:** the legacy `rich` dashboard (health/agent/features tabs,
  Redis-backed); with Redis down it shows a graceful "Redis DOWN/not available" state, **not** a
  traceback (it no longer crashes).

### C2. Static plots (`nat visualize …`)

Each command writes PNGs under `reports/figures/` (skeptical → `reports/skeptical_validation/`).
Open the PNGs and check, per category: image is **non-empty**; title / axis labels / legend present
and correct; dark GitHub theme (`#0d1117` background, accent palette from
`scripts/viz/features.py` `STYLE`/`COLORS`); the plotted series visually matches the underlying data
window.

- [ ] `nat visualize data --symbol BTC` → data-quality plots render.
- [ ] `nat visualize scan --symbol BTC` → scanner/edge plots render.
- [ ] `nat visualize cluster --symbol BTC` → PCA/cluster scatter; clusters visually separable,
  points coloured by label.
- [ ] `nat visualize profile` → profiling plots render.
- [ ] `nat visualize hierarchy` → hierarchical-cluster figure renders.
- [ ] `nat visualize skeptical` → diagnostic plots under `reports/skeptical_validation/` render.
- [ ] `nat visualize all` → runs the above batch without error; spot-check a sample of the PNGs.
- [ ] Open a handful: `xdg-open reports/figures/<one>.png` — confirm no truncated/blank/garbled
  images.

### C3. Nightly HTML report

- [ ] `nat nightly report` → prints latest report summary + file paths.
- [ ] `nat nightly open` → opens latest `reports/nightly/<date>.html` in the browser. **Expected:**
  - Self-contained: **every image displays** (all `<img>` are base64 — no broken-image icons, no
    external requests). Sanity check:
    `grep -cE 'src="http|href="http' reports/nightly/<date>.html` → `0`.
  - Status badge (complete / interrupted / partial) and freshness banner are correct.
  - All five sections present: **health, wiring, features, gauntlet, viz**.
  - All embedded figures render: **price overview, equity curves, daily PnL, NaN heatmap**, and the
    three **wiring histograms** (whale / liquidation / concentration).
  - The "NO DATA" placeholders for still-dead wiring columns are **expected**, not a failure (the
    position tracker is disabled by default).

### C4. Web dashboards

Start each, open the URL, confirm it loads, updates, and renders tables/heatmaps correctly. Stop when
done.

- [ ] Ingestor dashboard — `nat run serve` → http://localhost:8080 — live feature counters / data
  freshness update in real time.
- [ ] Pipeline dashboard — `nat dashboard` → http://localhost:8050 — state, log tail, and figure
  list load; no stuck "loading".
- [ ] Agent dashboard — `nat agent dashboard` → http://localhost:8060 — IC heatmap renders with a
  sane colour gradient; registry / cycle tables populate (or show a graceful empty state).

### C5. Out-of-scope visual surfaces (pointers only)

- [ ] Grafana (:3002) and Optuna dashboard (:8070) are part of the Docker stack — verify via
  `docs/cloud_deployment/1_4_testing_verification.md`, not here.

---

## What this plan intentionally does NOT cover

- Live WebSocket ingestion actually filling `data/features/*.parquet`, Telegram alert delivery,
  Docker service health, daemon start/stop lifecycle, and model-serving latency. These need live
  services and are out of scope for a terminal+visualization checklist. The Docker portion is in
  `docs/cloud_deployment/1_4_testing_verification.md`; the rest is a future operational runbook.
