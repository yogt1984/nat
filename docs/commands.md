# `nat` commands in the tutorials & guides

A quick index of the `nat` commands referenced across the tutorial / guide / runbook docs,
**in the order they appear** in each. This is a navigation aid, not the source of truth — the CLI
itself is authoritative: `nat help`, `nat commands`, `nat commands --json` (~298 commands across
65 groups). For per-unit detail, open the linked doc.

---

## [`README.md`](../README.md) — Quick Start (the daily flow)

1. `nat start` · `nat log` · `nat status` · `nat dashboard`
2. `nat doctor` — ingestion preflight (data-dir ownership/writability, binary, disk)
3. `nat data validate` · `nat data validate <file.parquet>`  → PASS/WARN/FAIL
4. `nat viz render --tf 15m` — whole-day all-features PNG snapshot
5. `nat viz render --tf 5m 1 --open` — zoom into the first 5-min page
6. `nat viz render --tf 5m 1 --features flow` — scope a page to a category/vector/list
7. `nat viz render --last 15m --open` — freshest-readable window + its age
8. `nat viz3d --tf 15m --features entropy --open` — interactive 3D feature-surface
9. `nat monitor` — live feature probe (~10 Hz, no ingestion) · `nat monitor tui` — legacy dashboard
10. `nat agent start` · `nat mf_agent start` · `nat macro_agent start` · `nat meta_agent start`
11. `nat agent status` · `nat agent dashboard` · `nat agent registry` · `nat agent report`
12. `nat oos30` — 5 winning algorithms, walk-forward
13. `nat spannung --symbol BTC` · `nat spannung regime --symbol BTC` · `nat profile scalp --symbol BTC`

## [`docs/test_docs/viz_validation_testing_guide.md`](test_docs/viz_validation_testing_guide.md) — testing guide

1. `nat data validate <path>` · `nat data validate` · `nat data validate /tmp/nope.parquet`
2. `nat viz render --tf 15m --symbol BTC`
3. `nat viz render --tf 5m 1` · `--tf 5m 2` · `--tf 5m 9999` (out-of-range)
4. `nat viz render --tf 5m 1 --features flow` · `--features raw_midprice,raw_spread` · `--features nope_xyz`
5. `nat viz render --tf 15m --open`
6. `nat viz3d --tf 15m --symbol BTC` · `nat mesh` (alias) · `nat viz3d --tf 5m 9999`
7. `nat 15m viz --symbol BTC` · `nat 15m viz --symbol BTC --no-open`

## [`docs/in_progress/test_plan.md`](in_progress/test_plan.md) — manual test plan

**Section A — viz/validation tooling:**
`nat data validate <file>` → `nat data validate /tmp/nope.parquet` → `nat viz render --tf 15m` →
`--tf 5m 1/2/9999` → `--features flow/nope_xyz` → `--tf 15m --open` → `--last 15m` →
`nat viz3d --tf 15m` → `nat mesh` → `nat viz3d --tf 5m 2 --features entropy` → `--tf 5m 9999` →
`nat 15m viz [--no-open]`

**Section B — terminal connectivity:**
`nat --json commands` · `nat commands` · `nat help` → `nat test agent -h` (expected-fail) ·
`nat test` · `nat test validate` · `nat oos30 -h` · `nat algorithm evaluate -h` →
`nat --json status` · `nat doctor` · `nat config show` · `nat algorithm list` · `nat process list` ·
`nat data validate -h` · `nat viz render -h` · `nat viz3d -h` · `nat mesh -h` · `nat gauntlet report` ·
`nat nightly report` · `nat reports latest` · `nat experiment list` · `nat cluster -h` · `nat backtest -h` ·
`nat swarm status` · `nat evolve status` · `nat agent status` · `nat mf-agent status` ·
`nat macro-agent status` · `nat meta-agent status` · `nat discovery status` · `nat it-engine status` ·
`nat docker ps` · `nat start -h` · `nat stop -h` · `nat alg1 -h`

**Section C — visualization correctness:**
`nat monitor` · `nat monitor tui` → `nat visualize {data,scan,cluster,profile,hierarchy,skeptical,all}` →
`nat nightly report` · `nat nightly open` → `nat run serve` · `nat dashboard` · `nat agent dashboard`

## [`HETZNER_DEPLOYMENT_PLAN.md`](../HETZNER_DEPLOYMENT_PLAN.md) — cloud deploy

`nat doctor` → `nat docker build` → `docker compose up -d …` (15 services) → `nat docker ps` →
verify: `nat gap status` · `nat risk status` → cutover: `nat gap` (gap profile) → `nat start`
(su-35, post-streak only)

## [`docs/cloud_deployment/0_overview.md`](cloud_deployment/0_overview.md) — in-stack runbook

`nat doctor` → `nat docker build` → bring-up (`docker compose up -d …`) → `nat gap status` ·
`nat risk status`

## [`docs/cloud_deployment/2_observability_and_e2e.md`](cloud_deployment/2_observability_and_e2e.md) — observability + e2e

`nat doctor` → `nat docker build` → bring-up (`docker compose up -d …`, all 15 services)

---

## Spec / contract docs (command references, not step-by-step tutorials)

- [`docs/requirements/parquet_viz_validation.md`](requirements/parquet_viz_validation.md) — the
  viz/validation requirements: `nat data validate <path> [--json]`, `nat data schema`,
  `nat viz render --tf {1m,5m,15m} [INDEX] [--features] [--last]`, `nat viz3d`/`nat mesh`,
  `nat 15m viz`, `nat monitor`(+`tui`), `nat doctor`.
- [`docs/contracts/viz.md`](contracts/viz.md) — the viz contract: the `nat viz <x>` command shape +
  reference exemplars (`nat viz render`, `nat viz3d`, `nat data validate`).

> Counts/commands drift — `nat commands --json` is the machine-readable source of truth.
