# NAT — Task Backlog

**The single backlog.** Every actionable task lives here, once. Strategy, gates, and Current Focus
live in [`PLAN.md`](PLAN.md) — this file is its itemized companion. Findings/specs are *not* tasks
(see `research/`, `specs/`). Superseded task docs are in [`archive/`](archive/).

*Consolidated 2026-07-22 from: `PLAN.md`, `specs/process_layer.md` (ex-`PROPOSAL.md`), `07_26_TASKS.md`,
`backlog/`, `in_progress/tasks_assigned_12_6_26/`, `in_progress/{korrektur_tasks,test_plan}.md`,
`in_progress/{convolver_implementation,nan_wiring,cloud_deployment}/`. Items verified DONE in code were
dropped (see the "Verified shipped" note at the bottom); ~60 live items retained.*

## Conventions

- **One entry per task.** New tasks go here — never a new `*_TASKS.md` or dated task file.
- **ID** is stable; reference it from commits/PRs. IDs preserve source labels where recognizable
  (`HF*`, `LF*`, `F*`, `A*`, `OPS*`, `HYG*`, `PROC*`, `K*`).
- **Status:** `TODO` · `WIP` · `BLOCKED` · `DONE` (DONE rows swept to `archive/` quarterly).
- **Data:** `in-hand` (runs on existing parquet) · `streak` (needs the clean-data streak).
- Long specs live in `specs/` or a contract; link them, don't inline.

**Schema per row:** `ID` · title · status · prio(P0/P1/P2) · effort(XS/S/M/L) · data · dep · one-line.

---

## Q — Quant gates *(prove the edge is real and capturable)*

Gate chain: `Q0 → (Q1 ∥ Q2) → Q3 → Q4 → Q5`. No live capital before Q5 + G8 + kill-switch.

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| Q0 | Verify the clean-data streak | BLOCKED | P0 | XS | streak | — | `nat gap status` / `/streak`; master gate for the whole Q-branch. |
| Q1 | T0b Hetzner ingest box (24/7) | TODO | P0 | M | in-hand | REL-1..4 | Removes su-35 SPOF. **Ship the REL reliability cluster first** (Jul-5 zombie lesson). Merges OPS-5 + plan.md T0b. Runbook: `runbooks/HETZNER_DEPLOYMENT_PLAN.md`. |
| Q2 | Longitudinal tool `nat oos --window <N>d` | TODO | P1 | M | in-hand | Q0 | Walk-forward folds + deflated Sharpe + `--json` (generalize `nat oos30`). |
| Q3 | Revalidate 5 winners on ≥30 clean days | BLOCKED | P1 | M | streak | Q1,Q2 | `3f_liquidity`, `jump_detector`, `funding_reversion`, `optimal_entry`, `surprise_signal`. |
| Q4 | Alpha-skeptic kill gate | TODO | P0 | S | in-hand | — | Run each survivor through `alpha-skeptic` **before** the ≥90-day data spend; first pass on data in hand now. |
| Q5 | Conditional-IC > 0.15 go/no-go (~Aug) | BLOCKED | P0 | M | streak | Q3,Q4 | Trading-business gate (IC≈0.45 → ~0.03 under fills). *(Was mislabeled "D1" in PLAN.)* Path to 0.15 = PROC + institutional-GAP execution. |
| Q-K2 | Dead-feature / concentration production verdict | BLOCKED | P1 | M | streak | Q1 | Merges F8/K2/REGIME-3/nan_wiring-05/01_concentration: wiring locally verified; run 48h on T0b, apply 50+wallets/>20%-OI matrix, write FEATURES.md verdict. |
| Q2.7 | Horizon cross-validation of MF-capable tick algos | TODO | P2 | S | streak | Q0 | Re-test `propagator`/`switching_ou`/`bipower` at 5–30min (Tier-3 graveyard may be clock-mismatch). |
| LOOP-3 | Promotion daemon: measure `infra_stable` + real decay | TODO | P1 | S | in-hand | — | Currently hardcoded `True` (`promotion_daemon:456`); wire real half-life (see PROC-10). |
| REGIME-2 | Give `config/kalman.toml` a consumer | TODO | P1 | M | in-hand | — | No code references it; wire into `switching_ou`/`kalman_imbalance` or retire the config. |

## QA — New alpha candidates *(unfreeze the roster: features & algorithms not yet built)*

Institutional-GAP items cross-ref [`research/INSTITUTIONAL_ALGORITHMS.md`](research/INSTITUTIONAL_ALGORITHMS.md).

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| HF1 | Microprice algorithm | TODO | P1 | S | in-hand | — | Feature `microprice.py` exists; no registered algo consuming it. Directly attacks the fill-IC collapse (Stoikov 2018). |
| HF4 | VPIN toxicity gate (shared) | TODO | P1 | S | in-hand | — | Reusable adverse-selection gate (only the failed `vpin_regime` exists). |
| HF2 | Integrated multi-level OFI algorithm | TODO | P2 | M | in-hand | — | Feature `multilevel_ofi.py` exists; wire algo/generator (Cont–Kukanov–Stoikov). |
| HF5 | Avellaneda–Stoikov market making | TODO | P2 | L | in-hand | HF1,HF4,kill-switch | Execution-layer, sim-first (institutional GAP). |
| A4 | Queue-value execution model | TODO | P2 | M | in-hand | HF1 | Expected value of a resting limit order; sim-first (queue-reactive lineage). |
| A2 | Macro/daily mean-reversion algorithm | TODO | P2 | S | in-hand | — | Premium/basis reversion on the settlement-clock feature. |
| LF2 | OI-positioning-extremes algorithm | TODO | P2 | S | in-hand | — | `oi_divergence` failed; extreme-positioning variant not built. |
| LF6 | HAR-RV sizing (non-directional) | WIP | P2 | S | in-hand | — | Feature `har_rv.py` done; sizing wiring into `meta_portfolio`/kill-switch unverified. |
| HF3 | Bivariate Hawkes intensity-imbalance algorithm | TODO | P2 | L | streak | F7 | Needs λ_buy/λ_sell + branching ratio (Rust feature F7). |
| HF6 | Cross-symbol lead-lag scan (Hayashi-Yoshida) | TODO | P2 | XS | in-hand | — | Cheap scan; implement full algo only if lag>200ms survives (see PROC-9). |
| LF3 | Liquidation-cascade reversion | BLOCKED | P2 | M | streak | Q-K2 | K2-gated; a depth-only prototype is possible now. |
| LF4 | Volume-weighted TSM (daily) | TODO | P2 | S | streak | T13 | Needs candle/daily bars + a daily agent. |
| LF5 | Weekend-effect conditioning | TODO | P2 | XS | streak | LF4 | Conditioning layer on LF4. |
| F6 | Cross-symbol features (Rust ingestor) | TODO | P2 | M | streak | — | Shared `MarketState`/aggregator; feature-vector change → plan first. |
| F7 | Bivariate Hawkes features (Rust ingestor) | TODO | P2 | M | streak | — | λ_buy/λ_sell + branching ratio; ingestor change → plan first. |
| T13 | Daily agent + candle daemon | WIP | P2 | L | streak | Q0 | `fetch_candles.py` exists; no `DailyAgent`/candle daemon. Enables LF4/LF5. |

## D — Development / platform *(harden & ship `nat`)*

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| D1 | Viz set + maturity tags | WIP | P1 | M | in-hand | — | Foundation + features/algorithm/paper/portfolio viz shipped. **Remaining: spectral/regime/correlation viz (NAT8) + `[PROVEN]/[PRELIM]/[SPEC]/[LIVE]` command tags (NAT9).** Renders PROC-8 surface. |
| D2 | Modularize the `nat` monolith | DONE | P1 | L | in-hand | — | Verified: ~50 `scripts/cli/*.py` + `app.py` assembler (NAT10). *Kept for provenance; sweep to archive.* |
| D3 | Ship `nat` apt-installable | TODO | P2 | L | in-hand | — | Phase 1 = relocatable paths (XDG/`NAT_HOME`); then pipx/wheel; then `.deb`+apt repo. |
| D4 | Continuous-discovery → cloud research lab | WIP | P2 | L | in-hand | — | Harden `discovery_orchestrator` + 4 agents; surface via `api` + Next.js. Partially built. |
| DOCS-1 | Refresh `PLAN.md` §0/§3 + status corrections | TODO | P2 | S | in-hand | — | Mark D2/kill-switch/T5/T14 done; fix ~4-week staleness of the pinned block (docs-restructure Phase 5). |
| DOCS-3 | Fix `commands.md` + `CLAUDE.md` operational sections | TODO | P2 | S | in-hand | — | Stale command names/counts, missing compose services. |
| P4-2 | Per-hypothesis PDF export | TODO | P2 | M | in-hand | — | No Export-PDF button in `web/` (only live item left in `engineering_backlog`). |
| HYG-3 | Timeouts on unattended `subprocess.run` | TODO | P2 | S | in-hand | — | `code_synth`/`oos_validate`/nightly have none. |
| HYG-4 | Deduplicate the Rust hypothesis crate | TODO | P2 | M | in-hand | — | `normal_cdf`/`variance` copy-pasted across h2/h3/h4. |
| HYG-5 | CLI polish (wire or remove `nat test agent`) | TODO | P2 | S | in-hand | — | No agent subcmd in `cli/test.py` (CLAUDE.md notes it's not wired). |
| HYG-7 | Daemons: `print()` → structured logging | TODO | P2 | S | in-hand | — | tournament/it_engine/base daemons mix `print`. |
| TEST-1 | Manual terminal + viz-correctness checklist | TODO | P2 | S | in-hand | — | Recurring human run of parquet-validate / 260-cmd dispatch / render correctness (ex-`test_plan.md`). |

## P — PhD *(publish, then outreach)*

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| P1 | Polish → camera-ready convolver PDF | TODO | P1 | S | in-hand | — | |
| P2 | SSRN upload | TODO | P1 | XS | in-hand | P1 | 1–3 business days. |
| P3 | arXiv `q-fin.TR` endorsement | TODO | P1 | S | in-hand | P1 | |
| P4 | Prof outreach (gather Tier-1 emails + send) | TODO | P1 | S | in-hand | P2,P3 | Emails not stored — gathering is part of the task. |
| P5 | Track responses; stagger Tier-2 | TODO | P2 | — | in-hand | P4 | 2+ interested → formal apps. EPFL EDFI **Jan 15 / Mar 31 2027**. |

## REL — Reliability & ingestor *(P0 — the binding-constraint machinery; Q1 depends on it)*

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| REL-1 | WebSocket connect timeout | TODO | P0 | S | in-hand | — | `main.rs:378` still raw `connect().await` — **the zombie-ingestor root cause**. |
| REL-2 | In-process no-data watchdog + task supervision | WIP | P0 | M | in-hand | REL-1 | Stale-reconnect present (`main.rs:593-617`) but no `JoinHandle` supervision. Supersedes K5 "fixed" claim (memory notes recurrence). |
| REL-3 | External watchdog on data freshness | WIP | P0 | S | in-hand | — | `ops/gap_alert.py` now has auto-restart bookkeeping; finish + verify. |
| REL-4 | Wire Telegram push alerts (verify delivery) | TODO | P0 | S | in-hand | — | Code reads `TELEGRAM_*`; a <5min alert that doesn't page is worthless — verify end-to-end. |

## INF — Infrastructure / observability / CI

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| LOOP-2 | Dockerize the agent daemons (T12) | TODO | P1 | M | in-hand | — | No `agent-*` services in `docker-compose`. |
| K4 | WebSocket gap monitoring (Prometheus counters) | TODO | P2 | S | in-hand | — | Add >1s/>5s/>10s gap counters + alert threshold. |
| K6 | Historical gap audit + `data/catalog.json` | TODO | P2 | XS | in-hand | — | Audit May31–Jun2 tiny-file anomaly; maintain catalog. |
| F9 | Binance reference feed (cross-venue) | TODO | P2 | L | streak | — | External WS client for basis/lead-lag; separate process. |
| INF-SWARM | Swarm-results Grafana dashboard | TODO | P2 | S | in-hand | — | Add `docker/grafana/dashboards/swarm_results.json` (only unbuilt item in the cloud tree). |
| HYG-1 | Python lint/format in CI (ruff) | TODO | P2 | S | in-hand | — | No `pyproject`/`.pre-commit-config`. |
| HYG-2 | Triage 6 CI-ignored test files | TODO | P2 | S | in-hand | — | `ci.yml:122-127` `--ignore`s all 6. |
| HYG-6 | Enforce `clippy -D`; test train/backtest scripts | TODO | P2 | M | in-hand | — | `ci.yml:28` clippy still advisory `-W`. |

## BUG — Defects *(incl. cost-integrity; fix early)*

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| BUG-2 | `nat agent status` → `ModuleNotFoundError: logging_config` | TODO | P1 | XS | in-hand | — | `base.py:1241` sys.path; blocks agents **incl. on the cloud box**. Merges LOOP-1. |
| BUG-3 | Fix + enable GMM 5D regime classifier | TODO | P1 | M | in-hand | — | Path commented (`ing.toml:38`), bad cols (`:41`). **Do NOT merge `936f7cb`** (drops whale flow). Merges REGIME-1. |
| BUG-1 | Retrain/revalidate 3 ML algos vs current schema | TODO | P1 | S | in-hand | — | `mean_reversion_detector`, `meta_labeling`, `regime_conditioned_lgbm` (artifacts date 2026-06-08). |
| COST-1 | Unify the two cost systems | TODO | P1 | S | in-hand | — | `backtest/costs.py fee_bps=5.0:37` bypasses `load_costs()`. |
| COST-2 | Remove zero-cost backtest fallback | TODO | P1 | S | in-hand | — | `CostModel(fee_bps=0)` at `:162/:101`. |
| COST-3 | Purge hardcoded fee/slippage literals + CI grep guard | TODO | P1 | S | in-hand | — | 8.0/3.5 literals present; add a CI guard (guardrail: all costs via `load_costs()`). |
| COST-4 | Wave-gate thresholds → config | TODO | P2 | S | in-hand | — | Gate literals in `evaluate_wave{1,2}_gate.py`. |
| K3 | `regime_accumulation_score` constant (0.4429) | BLOCKED | P2 | XS | streak | BUG-3 | Likely auto-resolves with the GMM/K2 fix. |

## PROC — Process / IT discovery layer *(raise the ceiling on how NAT discovers edges)*

Detailed spec: [`specs/process_layer.md`](specs/process_layer.md). Critical path:
`PROC-12 → PROC-6 → PROC-7 → PROC-8 → PROC-1`. **Start with PROC-12 + PROC-6.** All run on data in hand.
*(The `process_signal_design` S1–S9 series is folded here: S1/S2→PROC-3, S3→PROC-9, S4→PROC-12/13,
S5→PROC-4, S6→PROC-6, S9→PROC-8/11.)*

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| PROC-12 | Null-calibration layer ⭐ | TODO | P0 | S | in-hand | — | Shuffle-null → bits-above-null / z-score, not raw bits. Foundation. (=S4 permutation_null) |
| PROC-6 | `conditional_predictability` process ⭐ | TODO | P0 | M | in-hand | PROC-12 | `MI(f;label\|Z=z)` as a function of z; argmax = tradeable regime. (=S6) |
| PROC-5 | Schedule the 3-bar classifier as standing eval | TODO | P0 | XS | in-hand | — | `triple_barrier → mi_ksg(target=tb_label)`; audit whether ever run. |
| PROC-7 | Horizon/label MI-surface meta-process | TODO | P1 | M | in-hand | PROC-5,6,12 | Sweep `(horizon,geometry,regime)` → argmax. |
| PROC-13 | FDR/DSR on the process layer + cross-run ledger | TODO | P1 | S | in-hand | PROC-12 | BH q per cell; program-level `(process,target,n_tested,git_sha)` ledger (=B3). |
| PROC-4 | Longitudinal MI tracker (`mi_stability`) | TODO | P1 | M | ≥10 days | PROC-12 | Cross-day MI stability (=S5); planted path runs now. |
| PROC-8 | Predictability surface + viz | TODO | P1 | M | in-hand | PROC-6,7 | Central artifact `combo×horizon×label×regime→MI`; feeds D1. |
| PROC-10 | Predictability half-life | TODO | P1 | S | ≥30 days | PROC-4 | `MI(t)` decay; feeds Q4 + LOOP-3. |
| PROC-3 | MI-maximizing nonlinear combiner (`cmi_select`+synergy) | TODO | P2 | L | in-hand | PROC-12,13 | Synergy-aware selection (=S1+S2), replaces myopic greedy. |
| PROC-9 | Transfer-entropy causal graph (`lead_lag_te`) | TODO | P2 | M | in-hand | PROC-12 | Directed feature/symbol lead-lag (=S3); nonparametric Hasbrouck. Feeds HF6. |
| PROC-1 | Process→algorithm compiler | TODO | P2 | L | in-hand | PROC-6,7,12,13 | Promoted finding → registered `MicrostructureAlgorithm`. |
| PROC-11 | Two-stage regime-then-price system | TODO | P2 | L | in-hand | PROC-6 | Forecast regime → fire signal only in favorable regime. (=S9 signal_book) |
| PROC-2 | Self-explaining edges + reading notes | TODO | P2 | M | in-hand | — | Mechanism annotation per edge + 5 paper reading notes. |
| PROC-17 | Target as a first-class node (`targets.py`) | TODO | P1 | S | in-hand | — | Replace `target_col`; today only `ic_horizon`/`ml_importance` honor targets. Substrate for PROC-5/6/7. |
| PROC-15 | `residualize` transform (pure-innovation) | TODO | P2 | XS | in-hand | — | Orthogonalize a feature vs a conditioning set, no look-ahead (=S7). |
| PROC-18 | `feature_ops` transforms (fractional-diff, etc.) | TODO | P2 | M | in-hand | — | Frac-diff / spectral / robust-norm operators (institutional GAP: frac-diff). |
| PROC-16 | `pca_combo` Marchenko–Pastur denoise param | TODO | P2 | XS | in-hand | — | MP eigenvalue clip on the existing `pca_combo` (=S8). |

---

## Verified shipped (dropped, not carried over)

Confirmed present in code during consolidation, so **not** listed above (sources archived):
CLI modularization (NAT1–7, NAT10 = D2), all 5 F-features (`settlement_clock`, `microprice`,
`multilevel_ofi`, `har_rv`, `realized_moments`), algos `relative_value_pairs`/`vol_squeeze`/
`funding_settlement`, costs+provenance+`signal_lifecycle`+`promotion_daemon`+`kill_switch`+
`nat oos --window`, the convolver pipeline (14 stages), nan-wiring steps 01–04, and the cloud/swarm
stack (Caddy, `nat docker`, swarm CLI, Optuna CMA/TPE/NSGA-II). `process_concept` (process as 3rd
citizen) shipped as `scripts/processes/`. `korrektur` K1/K5 marked fixed (K5 re-opened as REL-2).

*Sources drained into this file are moved to `archive/` (docs-restructure Phase 3). Tier-A finding
reports inside `in_progress/tasks_assigned_12_6_26/` (features_report, algorithms_report,
data_inventory, situation_analysis) are **left in place** — they route to `research/FINDINGS.md` in
Phase 4, not archived. See [`DOCS_RESTRUCTURE_PROPOSAL.md`](DOCS_RESTRUCTURE_PROPOSAL.md).*
