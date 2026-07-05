# NAT — Improvement Task Backlog (07/2026)

Produced 2026-07-05 from a full-project analysis: live ops-incident investigation, a
plan-vs-implementation audit, and a code-health review. Companion to
[`03_07_report.md`](03_07_report.md) (strategic directions) — this file is the concrete task
layer. It extends the open-bugs list in [`PLAN.md`](PLAN.md) §0; where a task supersedes or
refines a PLAN.md item, that is noted inline.

**Priorities:** P0 = binding-constraint / act now · P1 = before the next milestone window
(G8/D1, ~Aug) · P2 = hygiene & debt, schedule opportunistically.
**Effort:** S < 2 h · M ≤ 2 d · L > 2 d.
All file:line references verified against the working tree on 2026-07-05.

---

## Recommended execution order

1. **OPS-0…OPS-4** — restart the ingestor, then close the three continuity holes (timeout,
   staleness watchdog, Telegram). Small effort; directly serves the master gate (clean 7-day
   streak).
2. **OPS-5** — deploy the T0b Hetzner ingest box (PLAN.md Q1/P0; still the structural fix).
3. **LOOP-1/LOOP-2** — `sys.path` fix + dockerize the agents; turns the platform back into the
   autonomous system it is designed to be.
4. **COST-1…COST-3 + LOOP-3** — close the false-discovery holes before any G8 paper window.
5. **REGIME-1/REGIME-2** — wire the regime-gating path that decides D1 (conditional-IC).

---

## 0. Context — the Jul-3 → Jul-5 zombie-ingestor incident

Timeline (all times UTC):

- **Jul-3 20:42:04** — all three symbol tasks log `Connecting to WebSocket` … and never log
  again. No reconnect attempts, no errors, nothing from `ing::ws` for the following 29+ h.
- **Jul-3 20:47:54** — last Parquet write (the writer draining its buffer), per `nat gap status`.
- **Jul-3 → Jul-5** — process stays "RUNNING" (kept alive by the positions-poller thread, which
  logs `Connection reset by peer` on every REST call). The cron watchdog
  (`*/5 * * * * pgrep -x ing || restart`) never fires because the process exists.
  `gap_alert.py` detects the gap within its 300 s threshold and warns every 30 s — locally only:
  its own log states `Telegram NOT configured — gap alerts are LOCAL-ONLY`.
- **Jul-5 ~00:05** — network to `api.hyperliquid.xyz` recovers (fresh requests return HTTP 200);
  the hung WS tasks still never recover. Jul-4 = zero data.

Three stacked defects, each individually small (→ OPS-1/2/3/4). Even outside this incident,
coverage is chronically partial: Jun-24 → Jul-3 days hold only **10–18 of 24** hourly Parquet
files, and Jun-27/28 are missing entirely.

---

## 1. OPS — data-continuity layer  `[P0]`

The binding constraint (PLAN.md §0). Every task here is prerequisite work for the clean-streak
master gate.

### OPS-0 · Restart the stalled ingestor — `P0 · S`
- **Problem:** ingestor is a zombie since Jul-3 20:47Z (see §0).
- **Fix:** `nat stop && nat start`; confirm `nat gap status` returns to fresh and hourly files
  resume in `data/features/<today>/`.
- **Verify:** `nat status` shows last-write < 5 min; a new Parquet file appears within the hour.

### OPS-1 · WebSocket connect timeout — `P0 · S`
- **Problem:** `client.connect().await` at `rust/ing/src/main.rs:378` has no timeout. A hung
  TCP/TLS/WS handshake blocks the symbol task forever. The exponential-backoff retry branch only
  fires when `connect()` *returns* `Err` — unreachable on a hang. This is the direct root cause
  of the Jul-3 incident.
- **Fix:** wrap every `connect()`/`reconnect()` await in `tokio::time::timeout` (suggested
  default 30 s, as a `WebSocketConfig` field, not a magic number); a timeout counts as a failed
  attempt and feeds the existing `reconnect_delay_ms` backoff in `rust/ing/src/ws/client.rs`.
- **Verify:** planted test first (per METHODOLOGY): a mock endpoint that accepts TCP but never
  completes the handshake must produce a timeout + retry within N seconds, not a hang. Then a
  live `nat test validate` pass.

### OPS-2 · In-process no-data watchdog + task supervision — `P0 · M`
- **Problem:** two silent-death modes remain even with OPS-1: (a) a connected socket that stops
  delivering messages; (b) a per-symbol task that exits/panics while the process lives. Main
  never joins or monitors the per-symbol task handles, and the ping/pong state in
  `ws/client.rs` (`last_ping_sent`, `pong_received`) is not evaluated while a task is stuck.
- **Fix:** (a) per-symbol staleness check in the main `tokio::select!` health tick — no message
  for N s ⇒ force reconnect; (b) hold the per-symbol `JoinHandle`s; if any task ends or its
  symbol emits nothing for M minutes ⇒ restart the task, or exit the process nonzero so the
  external watchdog restarts it (crash-only design is acceptable here).
- **Verify:** planted test: feed a stream that goes silent ⇒ reconnect observed; kill one symbol
  task ⇒ process either respawns it or exits nonzero. Real-parquet smoke before commit.

### OPS-3 · External watchdog on data freshness, not process existence — `P0 · S`
- **Problem:** the cron watchdog is `pgrep -x ing || restart` — liveness only. A stalled-alive
  ingestor (this incident: 29+ h) is invisible to it.
- **Fix:** extend the cron job (or replace with a small script under `scripts/ops/`) to also
  check last-write age over `data/features/` — the exact freshness logic `nat gap status` /
  `scripts/ops/gap_alert.py` already computes — and `nat stop && nat start` when age exceeds a
  threshold (suggested 10–15 min; config key in `config/ops.toml`, alongside the existing gap
  threshold). Guard against restart loops (e.g. max one forced restart per 30 min).
- **Verify:** touch nothing for > threshold with the ingestor deliberately suspended
  (`kill -STOP`) ⇒ watchdog restarts it; check `logs/cron_restart.log`.

### OPS-4 · Wire Telegram push alerts — `P0 · S`
- **Problem:** `gap_alert.py` supports push alerts but runs local-only; 29 h of warnings went
  unseen. The < 5-min gap alert is an explicit Q1 deliverable (PLAN.md §2).
- **Fix:** set `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` in `.env` (the daemon already reads
  them and logs its alerting mode at startup).
- **Verify:** `grep Telegram logs/gap_alert.log` after daemon restart shows push mode; trigger a
  synthetic gap (threshold-exceeding pause) and receive the message.

### OPS-5 · Deploy the T0b Hetzner ingest box — `P0 · M`
- **Problem:** single-box ingestion is the structural cause of the streak failures. T0b has been
  the declared critical path since 2026-06-24 (PLAN.md §0 do-now #2) and is still undeployed.
- **Fix:** execute root `HETZNER_DEPLOYMENT_PLAN.md` (`nat deploy cloud <ip> --dry-run` first).
  Ship OPS-1…OPS-4 *before or with* it — the cloud box inherits the same zombie failure mode
  otherwise. su-35 stays frozen throughout (guardrail).
- **Verify:** per the deployment plan; then `nat gap status` style freshness check against the
  cloud box's data path, and the Telegram gap alert firing end-to-end from that box.

---

## 2. LOOP — the autonomous research loop isn't running  `[P1]`

OBJECTIVE.md describes a continuous autonomous loop; in practice it has not run unattended since
mid-May. Independent of the data blocker.

### LOOP-1 · Fix `sys.path` in all agent CLI entry points — `P1 · S`
- **Problem:** `nat agent status` (and `mf-agent` / `macro-agent` / `meta-agent` / cascade
  equivalents, incl. `queue`/`registry`/`graveyard`/`report` subcommands) crash with
  `ModuleNotFoundError: No module named 'logging_config'` — reproduced 2026-07-05; import at
  `scripts/agent/base.py:1241`. The daemon entry points never add `scripts/` to `sys.path`.
  PLAN.md's open-bugs list names only `nat agent status`; the blast radius is every agent daemon,
  including on the future cloud box.
- **Fix:** one path-bootstrap (or absolute imports + package install per D3's `nat_paths.py`
  work) applied to `scripts/agent/{daemon,mf_daemon,macro_daemon,cascade_daemon,meta_daemon}.py`.
- **Verify:** `nat agent status`, `nat mf-agent status`, `nat macro-agent status`,
  `nat meta-agent status` all return instead of raising; `pytest scripts/tests/test_agent_*.py`.

### LOOP-2 · Dockerize the agent daemons (T12) — `P1 · M`
- **Problem:** `docker-compose.yml` defines 15 services — none are the agents. Consequence:
  `data/agent/agent_state.json` shows `cycle_count: 1`, `last_cycle_start: null` (stalled since
  2026-05-15), and the discovery orchestrator's state file has never been created. The
  autonomous loop exists as code, not as a running system.
- **Fix:** add `agent-micro`, `agent-mf`, `agent-macro`, `meta-agent` (and decide on
  `cascade_daemon`, see DOCS-1) services mirroring the existing `promotion`/`kill-switch`
  service patterns; health-checked, restart-unless-stopped, state volumes mounted.
- **Verify:** `nat docker up` ⇒ agents cycle (state JSON `cycle_count` advances; structured
  research output lands in `data/research/`); dashboard at :8060 reflects live cycles.

### LOOP-3 · Promotion daemon: measure `infra_stable`; real decay check — `P1 · S/M`
- **Problem:** one of the five G8 paper-trading criteria is hardcoded:
  `scripts/promotion_daemon.py:456` sets `"infra_stable": True` (and the gate defaults truthy at
  `:133`) — a silent pass on a promotion gate. `_check_decay` (`:460-466`) fires only on an
  explicit metadata flag; no per-signal IC-decay tracking. No LIVE signals exist yet, so the
  window to fix this cheaply is *before* the first G8 window (~Aug).
- **Fix:** derive `infra_stable` from observable infra facts over the paper window (e.g. data-gap
  count from the OPS-3 freshness source and ingestor restarts) — criteria themselves imported
  from ROADMAP/G8, not invented here; implement rolling-IC decay from the paper-trading records.
- **Verify:** planted test: a paper window containing a synthetic multi-hour gap must fail the
  `infra_stable` gate; a synthetic decaying-IC series must trip `_check_decay`.

---

## 3. COST — cost-model and gate guardrail leaks  `[P1]`

The "all costs via `load_costs()`" guardrail has real holes. These threaten result validity —
the exact damage class (false discovery shipped) the guardrails exist to prevent.

### COST-1 · Unify the two cost systems — `P1 · S`
- **Problem:** `scripts/backtest/costs.py` is a parallel cost system:
  dataclass default `fee_bps: float = 5.0` (`:37`) vs the authoritative
  `config/costs.toml` `taker_bps = 3.5`; its `from_config()` (`:53-77`) re-implements TOML
  loading with its own fallback instead of delegating to `scripts/utils/costs.py::load_costs()`;
  docstrings still cite "~5 bps taker". Net effect: `CostModel()` → 5.0,
  `CostModel.from_config()` → 3.5 — result depends on call path.
- **Fix:** make `CostModel` source its defaults from `utils.costs` accessors; delete the
  duplicate loader; fix docstrings.
- **Verify:** unit test asserting `CostModel().fee_bps == taker_bps()`; grep CI guard (see
  COST-3) stays green.

### COST-2 · Remove the zero-cost backtest fallback — `P1 · S`
- **Problem:** `scripts/run_backtest.py:162` and `scripts/run_backtest_tracked.py:101` fall back
  to `CostModel(fee_bps=0, slippage_bps=0)` when `--cost-model` isn't a recognized preset — an
  unrecognized/empty value silently produces a cost-free backtest.
- **Fix:** default branch = config-loaded taker model; unknown preset = hard error.
- **Verify:** invoking with a bogus `--cost-model` exits nonzero; regression test comparing
  net-PnL output with/without the fix on a fixture.

### COST-3 · Purge hardcoded fee/slippage literals — `P1 · S`
- **Problem:** none of these import `utils.costs`, and one actively diverges:
  - `scripts/exploration/skeptical_regression_test.py:430,432` — `taker_cost_bps = 8.0`,
    `maker_cost_bps = 1.0` (**8.0 ≠ the config's 7.0 round-trip**);
  - `scripts/phase1_signal_test.py:252,537` — `taker_fee_bps = 3.5` (twice);
  - `scripts/eamm/backtest.py:191` — `taker_fee = 3.5 / 10000.0`;
  - `scripts/exploration/generate_report.py:248` — `cost_per_trade = 0.00035`;
  - `scripts/train_hierarchical.py:192` — `cost_bps = 11e-4`;
  - `scripts/analysis/funding_carry.py:595-603` — `fee_bps=7.0` / `2.0`.
- **Fix:** route all through `load_costs()`; add a CI grep guard (naive pattern over
  `scripts/`, allowlist `utils/costs.py` + tests) so new literals fail fast.
- **Verify:** re-run one affected report (e.g. `skeptical_regression_test`) and confirm the
  numbers shift consistently with 7.0 RT; CI guard red/green demo.

### COST-4 · Wave-gate thresholds → config — `P2 · S`
- **Problem:** `scripts/evaluate_wave1_gate.py:50,52` and `scripts/evaluate_wave2_gate.py:157`
  embed gate literals in code, while equivalent thresholds live in `config/agent.toml` /
  `config/algorithms.toml`. Inconsistent with the "gates imported, not invented" rule —
  config-externalized gates are auditable; in-code ones drift.
- **Fix:** move the existing numbers (unchanged — no new thresholds) into config; scripts read
  them.
- **Verify:** gate decisions identical before/after on the same input (golden-file test).

---

## 4. REGIME — the D1-deciding research path has unwired plumbing  `[P1]`

Everything downstream (live capital, Q-branch continuation) hangs on the conditional-IC verdict
(D1, ~Aug). The documented route to winning it — regime gating, IC 0.45 → 0.55–0.67
(`03_07_report.md` direction #4) — has dead wiring.

### REGIME-1 · GMM 5-D regime classifier: three stacked blockers — `P1 · M`
- **Problem:** PLAN.md's open-bug framing ("fix column names, train, enable") understates it:
  1. `scripts/train_regime_gmm.py:40-51` — `ill_kyle_lambda_300` and `tox_vpin_50` match no
     Rust column (real names: `illiq_kyle_100`/`_500`; VPIN naming differs), and the
     `FEATURE_ALTERNATIVES` fallback dict covers only the *other* two columns — these two
     silently resolve to nothing;
  2. `gmm_model_path` is commented out in `config/ing.toml:37-38`;
  3. `rust/ing/src/main.rs` never invokes `ml::regime` even when a model exists.
- **Fix:** correct the column list against `names_all()` (schema SSOT in
  `rust/ing-features/src/lib.rs`), train, enable the config key, wire the call site. Reminder
  from PLAN.md: do **not** merge branch `936f7cb` (drops whale flow).
- **Verify:** planted test with synthetic regimes recovered by the trained GMM; then live smoke:
  `gmm_*` feature columns populate (non-NaN) in fresh Parquet.

### REGIME-2 · Give `config/kalman.toml` a consumer — `P1 · M`
- **Problem:** zero references to `kalman.toml` anywhere in `scripts/` or `rust/` (verified by
  grep). `scripts/algorithms/kalman_imbalance.py` runs on hardcoded constructor defaults. The
  orphaned `[kalman.regime]` section (`feature = "ent_book_shape"`, percentile gate) is exactly
  the regime-gated Kalman extraction the Jul-3 report rates the highest-leverage direction.
- **Fix:** load the file in `kalman_imbalance.py` (same pattern as `config/algorithms.toml`
  consumers), implement the `ent_book_shape` percentile gate it specifies.
- **Verify:** algorithm-contract conformance (`pytest scripts/tests/test_bar_level_dispatch.py`),
  then `nat algorithm evaluate --algorithm kalman_imbalance --symbol BTC` with/without the
  regime gate and compare conditional IC.

### REGIME-3 · Write the T0 whale-viability verdict — `P1 · S`
- **Problem:** `docs/in_progress/nan_wiring/01_concentration_viability_assessment.md` and
  `05_concentration_viability.md` are still blank decision-matrix templates, though the wiring
  itself is verified (all 31 whale/liq/concentration columns populate). The missing verdict
  blocks LF3 (liquidation cascade) un-gating and the agents' dead-column skip lists per T0's own
  exit criteria.
- **Fix:** run the assessment on the data already in hand, fill the matrices, record the verdict.
- **Verify:** the two docs contain a decision; downstream gates reference it.

---

## 5. DOCS — plan/doc drift (misallocation risk)  `[P2]`

The plan understates what is shipped; the risk is re-doing finished work or mis-prioritizing
Q/D/P time.

### DOCS-1 · Refresh PLAN.md §3 + status corrections — `P2 · S`
Corrections, all verified in code:
- **D2 (CLI modularization) is done** — `nat` is a 19-line shim over `scripts/cli/` (49 files,
  real `register()` protocol) since commit `b916541` (2026-06-21).
- **D3 is essentially done** — `scripts/nat_paths.py` implements the full relocatable-path
  precedence; `packaging/build_deb.sh` builds a real `.deb` (one sits in `build/deb/`). Remaining
  D3 work is the apt-repo step, not "gated on step-1".
- **Kill-switch shipped 2026-06-15** (`scripts/risk/kill_switch.py`, all four thresholds) —
  `03_07_report.md` still lists it as unshipped (~6 h).
- **T5 done** (`scripts/agent/base.py:325`), **T14 ~90 %** (see LOOP-3) — update CLAUDE.md's
  "pending" phrasing.
- **`cascade_daemon.py` exists** — a fifth agent (heatmap-cascade validation), absent from
  PLAN.md/GLOSSARY/CLAUDE.md; document it and decide its place in the agent fleet (it may
  already serve the liquidity-heatmap preprint's "empirical validation pending" note).

### DOCS-2 · Regenerate FEATURES.md from the code manifest — `P2 · S`
- **Problem:** FEATURES.md documents 16 categories / 211 features vs the actual 21 / 236
  (`rust/ing-features/src/lib.rs` doc-manifest is correct and matches CLAUDE.md). Five wired
  categories are missing entirely (microstructure, resilience, hawkes, cross_symbol, heatmap);
  two listed categories have wrong counts (entropy 27→24, context 9→12).
- **Fix:** regenerate from `names_all()` / the `lib.rs` manifest; consider a small generator
  script so it can't drift again.
- **Verify:** doc totals equal `count_all()` = length of `names_all()`.

### DOCS-3 · Fix `docs/commands.md` + CLAUDE.md operational sections — `P2 · S`
- `commands.md` documents underscored commands that error (`nat mf_agent` → real: `nat mf-agent`)
  and a stale count (~298; live is 331 per `nat --json commands`). CLAUDE.md's Docker section
  omits 6 of 15 compose services (gap-alert, kill-switch, promotion, signal-bridge,
  metrics-exporter, web) and 6 of 19 config files (`execution/monitoring/ops/processes/risk/
  tournament.toml`). Also reconcile the `mesh` "empty stub" comment vs its actual alias to
  `viz3d` (`scripts/cli/viz.py:505,524`).

---

## 6. HYG — engineering hygiene  `[P2]`

### HYG-1 · Python lint/format in CI — `P2 · S`
No `ruff`/`black`/`mypy` config or CI step exists for ~109k LOC of Python (CI's Python job runs
pytest only). Add `ruff check` + `ruff format --check` with a minimal `[tool.ruff]`. Highest
leverage single hygiene change; pairs with a `.pre-commit-config.yaml` (`ruff` + `cargo fmt`).

### HYG-2 · Triage the 6 CI-ignored test files — `P2 · S/M`
`.github/workflows/ci.yml:122-127` `--ignore`s `test_integration_profiling.py`,
`test_nat_cli.py`, `test_pipeline_runner.py`, `test_dashboard.py`, `test_visualize_scanner.py`,
`test_model_serving.py`. They exist but never run — silent rot. Determine why (slow/flaky/env),
then fix, mark nightly, or delete.

### HYG-3 · Timeouts on unattended `subprocess.run` calls — `P2 · S`
No-timeout calls in automated paths: `scripts/agent/code_synth.py:152` (LLM synthesis — can hang
an agent cycle), `scripts/oos_validate.py:78,101,282`, `scripts/nightly_report.py:629`,
`scripts/ops/gap_alert.py:296`, `scripts/pipeline_runner.py:174`, `scripts/run_experiment.py:39`.
Pattern to copy: `scripts/discovery_orchestrator.py:216` (`timeout=` + `TimeoutExpired` handler).

### HYG-4 · Deduplicate the Rust hypothesis crate — `P2 · M`
`rust/ing/src/hypothesis/` (10,255 LOC): `normal_cdf` re-implemented in `h2:1175`/`h4:999`
(private original at `stats.rs:411`), `variance` copy-pasted in h2/h3/h4, `make_error_result` in
h2/h3/h4, and per-file decision-plumbing templates. Export the stats fns; extract a
`HypothesisTest` trait for result/decision plumbing.

### HYG-5 · CLI polish — `P2 · S`
Wire or remove `nat test agent` (`scripts/cli/test.py` registers 21 test subcommands, no
`agent` — CLAUDE.md points at pytest instead; make the CLI agree). Give bare group parsers a
`print_help` default (pattern: `app.py:911`) so `nat viz` etc. never dead-end.

### HYG-6 · Enforce clippy; test the promotion-critical scripts — `P2 · M`
CI runs clippy with `-W` (advisory, `.github/workflows/ci.yml:28`) — switch to `-D warnings` or a
`[workspace.lints]` table. Add tests for the untested `train_*` / `run_backtest*` family
(12 modules, incl. `train_regime_gmm.py`, `oos_validate.py`) — they drive model promotion and
P&L numbers.

### HYG-7 · Daemons: `print()` → `logging` — `P2 · S`
Long-running services mix ~2,800 `print()` vs ~900 logging calls repo-wide; worst daemons:
`scripts/tournament/daemon.py` (37 print / 20 log), `scripts/it_engine/daemon.py` (16/14),
`scripts/agent/base.py` (39/42). Standardize daemons on the existing
`scripts/logging_config.py` (levels/timestamps/correlation IDs survive systemd/tmux).

---

## 7. Verified strengths (no action)

For calibration — checked and healthy: zero TODO/FIXME/HACK markers across ~154k LOC; no bare
`except:`; hot-path Rust (ws client, parquet writer, state) clean of production unwraps; typed
error handling (`thiserror`/`anyhow`); `Cargo.lock` + `requirements.lock` committed; CI includes
a Rust perf-regression gate; agent-daemon consolidation genuinely thin (no residual copy-paste);
CLI modular with drift-guard tests (`test_help_coverage.py`, `test_commands_snapshot.py`).
