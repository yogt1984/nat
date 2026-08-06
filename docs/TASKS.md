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
- **ID registry rule (P3).** *This file is the ID registry of record.* An ID exists when its row
  exists — mint by adding the row **first**, and only then may a commit, a FINDINGS section, or a
  spec reference it. Consult this file before minting. *(Violating this produced the COST-4/COST-5
  collision of 2026-07-30, now on the permanent record; see the erratum in the BUG/cost table.)*
- **Same-branch status rule (P3).** A branch that completes a task flips its row — status **and**
  merge SHA in the notes — **in the same merge**. "Done in code, open in TASKS" is a defect, not
  bookkeeping debt. *(~25 rows drifted this way between Jul-26 and Jul-31.)*
- **Verify before flipping.** A row goes DONE against *evidence* (merge SHA + the code path), never
  against a claim in another doc. The 2026-08-06 pass found two audit-asserted "DONE" items that had
  no supporting commit — they stayed open.
- **Weekly reconciliation.** Statuses, the FDR ledger, and `PLAN.md` §0 are refreshed weekly;
  DONE rows sweep to `archive/` quarterly.

**Schema per row:** `ID` · title · status · prio(P0/P1/P2) · effort(XS/S/M/L) · data · dep · one-line.

---

## Execution order *(chronological — read this first)*

*Rewritten 2026-08-06 for the post-Q4 strategy. The June ordering it replaced still placed
`HF1`/`A4`/`HF5` in a tier-5 "only if Q5 positive" bucket — reality inverted that on 2026-07-30:
Q4 killed the taker roster, the maker line was pulled forward **as research**, and the PROC
discovery layer shipped end to end. Tiers 1–3 below are entirely data-independent.*

**Where the record actually stands** (evidence in [`research/FINDINGS.md`](research/FINDINGS.md)):
the taker path is arithmetically closed (§2), all five shipped "winners" are refuted (§4.6), and
passive quoting at BTC's touch is structurally negative at every reachable fee tier — breakeven
maker rate is **+0.144 bps** vs a zero-fee best case (§4.11). One maker hypothesis survives:
**wider-spread pairs**, where the half-spread is a multiple of BTC's 0.083 bps.

**1 — Now: the one live maker hypothesis + the universe that tests it** *(data-independent)*
1. `XS-1` — universe candle backfill. The unblocker: `data/candles/` does not exist yet, and both
   `B-5` and the whole Class-3 track need it. `HyperliquidClient.get_meta()` already parses
   `universe`, so this is wiring, not new API work.
2. `B-5` — maker viability on wider-spread pairs. The direct test of §4.11's open question; reuses
   the §4.9 cell grid and criteria **unchanged** (this study moves the universe, never the bar).
   Fold in criterion (e) funding accrual and re-derive the A4 gate threshold per fee tier.
3. `COST-8` — `hyperliquid_maker()` hardcodes the 0.2 bps rebate that §4.11 names the most
   load-bearing unvalidated assumption in the stack. Route it through `load_costs()` before any
   further maker number is produced.

**2 — Class 3: the cross-sectional rotation track** *(data-independent; runs parallel to 1)*
4. `XS-2` — bar-level xs feature library (planted tests first).
5. `PROC-19` — `candles` data level + multi-symbol loading in the process runner (framework
   prerequisite for the three xs processes).
6. `XS-3` / `XS-4` / `XS-5` — rank-predictability, persistence, capacity gate.
7. `XS-6` — rotation OOS study, pre-registered. Promotes to lifecycle DISCOVERED iff it survives.

**3 — Class 1 signal layer** *(data-independent)*
8. `A-2` — combiner revalidation. §5's IC .18/.25/.36 rests on a **2-day** OOS with monotonically
   rising fold ICs; it is the last unrefuted capital-relevant claim in the record. Run it before
   anything is built on top of it.
9. `A-1` / `A-3` — agreement-gate standing eval; conditional-IC surface refresh.
10. `PROC-9` / `PROC-11` / `PROC-18` — TE causal graph, two-stage regime-then-price, feature ops.

**4 — Reliability & the data foundation** *(ops; gates everything paper/live)*
11. `REL-4` — verify Telegram delivery end-to-end (creds are user-side; a <5 min alert that doesn't
    page is worthless).
12. `Q1` — deploy the T0b Hetzner ingest box; kills the su-35 SPOF.
13. `Q0` — verify the clean-data streak (`/streak`) — the master gate for tier 5.

**5 — Streak-gated validation**
14. `PROC-10` — predictability half-life (≥30 clean days); feeds `LOOP-3`.
15. `Q-K2` — dead-feature / concentration production verdict (48 h on T0b).
16. `X-3` — fill-economics reruns on `X-2` F-task data or T0b shadow quotes — **the maker go/no-go**.
17. `Q5` — conditional-IC > 0.15. The trading-business gate; consumes the three-class program.

**6 — Capture & scale** *(only if `Q5` positive)*
18. `QA` — remaining institutional-GAP algos; `G8` paper window → live capital, gated on G8 + a
    healthy kill-switch.

**Parallel tracks** *(independent of the streak)*
- **D (platform):** `D1` finish viz + maturity tags → `D3` apt-packaging → `D4` cloud lab; plus `DOCS-3`, `HYG-*`, `INF-*`. (`D2`, `DOCS-1` done.)
- **P (PhD):** `P1` polish preprint → `P2` SSRN → `P3` arXiv → `P4` prof outreach → `P5` track responses (EPFL EDFI **Jan 15 2027**).

---

## Q — Quant gates *(prove the edge is real and capturable)*

Gate chain, restated post-Q4 (the June chain `Q0 → (Q1∥Q2) → Q3 → Q4 → Q5` was broken by events —
Q4 fired early and killed its own downstream): **`Q4 ✅ → R1 program (XS/A/B/X) → X-3 → Q5`**, with
`Q0 → Q1` running alongside as the ops track that gates paper/live only. No live capital before
Q5 + G8 + kill-switch.

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| Q0 | Verify the clean-data streak | BLOCKED | P0 | XS | streak | — | `nat gap status` / `/streak`; gates tier 5 (paper/live), **not** the R1 research program. |
| Q1 | T0b Hetzner ingest box (24/7) | TODO | P0 | M | in-hand | REL-4 | Removes su-35 SPOF. REL-1/2/3 shipped 2026-07-26/27, so this is unblocked but for the REL-4 alert verification. Runbook: `runbooks/HETZNER_DEPLOYMENT_PLAN.md`. |
| Q2 | Longitudinal tool `nat oos --window <N>d` | DONE | P1 | M | in-hand | — | **Shipped** — `scripts/cli/oos.py:95,148,151` (`--window`, `--json`, walk-forward + DSR). The Q4 skeptics used `nat oos --window 60d/90d` to kill `optimal_entry`/`jump_detector`. Row was stale; TASKS' own "Verified shipped" footer already listed it. |
| Q3 | Revalidate 5 winners on ≥30 clean days | MOOT | P1 | M | streak | — | **Void as scoped** — all five REJECTED in the lifecycle 2026-07-30 (§4.6); nothing to revalidate. The slot passes to whatever the R1 program promotes (`B-5`, `XS-6`, `A-2`). |
| Q4 | Alpha-skeptic kill gate | DONE | P0 | S | in-hand | — | **DONE 2026-07-30, 5/5 KILL** (`8676a41`; FINDINGS §4.6). The ≥90-day revalidation spend it was guarding against is cancelled. Root causes were platform defects → COST-6/7, BUG-4/5. |
| Q5 | Conditional-IC > 0.15 go/no-go | BLOCKED | P0 | M | streak | X-3, A-2 | The trading-business gate (IC≈0.45 → ~0.03 under fills). **Do not call this "D1"** — that ID belongs to the viz task; the mislabel is retired here and in PLAN §0. Path to 0.15 now runs through the R1 program, not the dead taker roster. |
| Q-K2 | Dead-feature / concentration production verdict | BLOCKED | P1 | M | streak | Q1 | Merges F8/K2/REGIME-3/nan_wiring-05/01_concentration: wiring locally verified; run 48h on T0b, apply 50+wallets/>20%-OI matrix, write FEATURES.md verdict. |
| Q2.7 | Horizon cross-validation of MF-capable tick algos | TODO | P2 | S | streak | Q0 | Re-test `propagator`/`switching_ou`/`bipower` at 5–30min (Tier-3 graveyard may be clock-mismatch). |
| LOOP-3 | Promotion daemon: measure `infra_stable` + real decay | TODO | P1 | S | in-hand | — | Currently hardcoded `True` (`promotion_daemon:456`); wire real half-life (see PROC-10). |
| REGIME-2 | Give `config/kalman.toml` a consumer | TODO | P1 | M | in-hand | — | No code references it; wire into `switching_ou`/`kalman_imbalance` or retire the config. |

## QA — New alpha candidates *(unfreeze the roster: features & algorithms not yet built)*

Institutional-GAP items cross-ref [`research/INSTITUTIONAL_ALGORITHMS.md`](research/INSTITUTIONAL_ALGORITHMS.md).

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| HF1 | Microprice algorithm | DONE | P1 | S | in-hand | — | `scripts/algorithms/microprice.py`, `@register`ed (`ae2c63e`). The maker-line anchor; `alg_mp_dev_bps` IC@50t +0.14/+0.24 gated. Measured effect is on **inventory** (liquidation cost −25…−40 %, §4.8), not yet per-fill PnL. |
| HF4 | VPIN toxicity gate (shared) | WIP | P1 | S | in-hand | — | **Not yet a unit.** Used in sims only as a `use_hf4_gate` flag over an externally supplied `gate_open` array (`execution/touch_maker.py:40,94`); no registered reusable `toxicity_gate` algorithm exists. Directionally validated in §4.5 (Sharpe lift 3/3 symbols). Ship it as its own registered unit. |
| HF2 | Integrated multi-level OFI algorithm | TODO | P2 | M | in-hand | — | Feature `multilevel_ofi.py` exists; wire algo/generator (Cont–Kukanov–Stoikov). |
| HF5 | Avellaneda–Stoikov market making | DONE | P2 | L | in-hand | — | `execution/avellaneda_stoikov.py` + HF5b queue coupling (`27c6b8b`). **Verdict recorded, negative** (§4.8): textbook A-S quotes sit behind the touch → fills dominated by price-throughs → −1.89 bps/fill. Exogenous-λ absolute PnL is never citable; only paired deltas. Sim-only. |
| A4 | Queue-value execution model | DONE | P2 | M | in-hand | — | `execution/queue_value.py` (`344a22a`). Produced the platform's only +EV number (+0.036/+0.013 bps per posting, §4.7) and the **EV gate that flips V1's per-fill sign** (−1.66 → +0.67, §4.9). Threshold must be **re-derived per fee tier** — it is non-monotone in the maker rate (§4.11). |
| A2 | Macro/daily mean-reversion algorithm | TODO | P2 | S | in-hand | — | Premium/basis reversion on the settlement-clock feature. |
| LF2 | OI-positioning-extremes algorithm | TODO | P2 | S | in-hand | — | `oi_divergence` failed; extreme-positioning variant not built. |
| LF6 | HAR-RV sizing (non-directional) | WIP | P2 | S | in-hand | — | Feature `har_rv.py` done; sizing wiring into `meta_portfolio`/kill-switch unverified. |
| LF7 | VWAP-SD channel maker (band mean-reversion) | TODO | P2 | M | in-hand | HF4,A4 | Spec `docs/research/new/vwap_sd_channel.txt`. k swept not fixed; single-day priors: k≤1.5 adverse, capture at k≈2.0–2.5, SOL-led; maker rebate +0.4bps RT, binding cost = adverse selection; queue-sim (A4) gates any profit claim. |
| HF3 | Bivariate Hawkes intensity-imbalance algorithm | TODO | P2 | L | streak | F7 | Needs λ_buy/λ_sell + branching ratio (Rust feature F7). |
| HF6 | Cross-symbol lead-lag scan (Hayashi-Yoshida) | TODO | P2 | XS | in-hand | — | Cheap scan; implement full algo only if lag>200ms survives (see PROC-9). |
| LF3 | Liquidation-cascade reversion | BLOCKED | P2 | M | streak | Q-K2 | K2-gated; a depth-only prototype is possible now. |
| LF4 | Volume-weighted TSM (daily) | TODO | P2 | S | streak | T13 | Needs candle/daily bars + a daily agent. |
| LF5 | Weekend-effect conditioning | TODO | P2 | XS | streak | LF4 | Conditioning layer on LF4. |
| F6 | Cross-symbol features (Rust ingestor) | TODO | P2 | M | streak | — | Shared `MarketState`/aggregator; feature-vector change → plan first. |
| F7 | Bivariate Hawkes features (Rust ingestor) | TODO | P2 | M | streak | — | λ_buy/λ_sell + branching ratio; ingestor change → plan first. |
| T13 | Daily agent + candle daemon | WIP | P2 | L | streak | Q0 | `fetch_candles.py` exists; no `DailyAgent`/candle daemon. Enables LF4/LF5. Overlaps `XS-1` — do the universe backfill there first, then this is just the daemon. |
| QA-JD2 | Wire `jump_detector_v2` into the economics harness | DONE | P1 | S | in-hand | — | *(Retro-row.)* `be9cf38` (2026-07-30). **Verdict: the taker-path Lee–Mykland family is dead, v2 included** — v1/v2 indistinguishable over 59 days at SSOT cost (§4.6). v1's threshold miscalibration was *not* the binding failure. Revival via taker: closed. |
| EXP-1 | Touch-maker experiment (pre-registered, multi-day) | DONE | P1 | M | in-hand | HF1,A4 | *(Retro-row.)* `f415d3a` / `60c3c58` (2026-07-31). 8 cells × 173 day-symbol episodes, criteria declared before the run: **all cells FAIL**, binding failures (b) day-consistency and (c) concentration (§4.9). Pre-registration is what stopped another one-lucky-day discovery from shipping. |
| REV-1 | Purge/re-run every §4.1-derived number still cited at SSOT cost | TODO | P1 | M | in-hand | COST-6 | *(Retro-row, and **still open** — no commit found 2026-08-06.)* The §4.1 winners table is fenced with a REFUTED banner in FINDINGS, but §4.1-derived Sharpe/bps numbers may survive in `reports/`, notebooks, preprints and `research/ALGORITHMS.md`. Sweep every citation; re-run at SSOT cost or delete. |

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
| REL-1 | WebSocket connect timeout | DONE | P0 | S | in-hand | — | Shipped as **OPS-1** (`9d9f4d7` / `4155f86`, 2026-07-26): `config.connect_timeout_ms` bounds the TCP/TLS/WS handshake (`rust/ing/src/ws/client.rs:61-67`). The zombie-ingestor root cause is closed. |
| REL-2 | In-process no-data watchdog + task supervision | DONE | P0 | M | in-hand | — | Shipped as **OPS-2** (`8dcf7f4` / `144e92f`, 2026-07-27): per-symbol task supervision with crash-restart on silent death. |
| REL-3 | External watchdog on data freshness | DONE | P0 | S | in-hand | — | Shipped as **OPS-3** (`ca31196` / `8ce5a51`, 2026-07-27): auto-restart of a bare ingestor on gap/stall without systemd. |
| REL-4 | Wire Telegram push alerts (verify delivery) | TODO | P0 | S | in-hand | — | **The only REL item still open** and the last thing between REL and `Q1`. Code reads `TELEGRAM_*`; creds are user-side. Verified 2026-08-06: no commit demonstrates end-to-end delivery. A <5 min alert that doesn't page is worthless — the acceptance test is a phone that buzzes. |

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
> **⚠️ ID erratum (COST-4 / COST-5 collision, 2026-07-30).** The Q4 follow-ups minted `COST-4` and
> `COST-5` for the VIP9-default purge and the CI-guard hardening while both IDs were **already
> taken** by the rows below (wave-gate thresholds; maker volume-tier). The commit subjects
> (`d9f3c1c`, `1334d41`) and `FINDINGS.md` §4.6 are immutable and still say COST-4/COST-5 in that
> sense. **Canonical from now on: those two pieces of work are `COST-6` and `COST-7`** (rows below).
> Root cause — no rule that this file is the ID registry; fixed in Conventions above.

| BUG-2 | `nat agent status` → `ModuleNotFoundError: logging_config` | DONE | P1 | XS | in-hand | — | `0df4797` / `9554cdb` (2026-07-29): `scripts/` on `sys.path` in daemon entry points (`agent/base.py:22-28`). Agent fleet revived, incl. on the cloud box. Merged LOOP-1. |
| BUG-3 | Fix + enable GMM 5D regime classifier | DONE | P1 | M | in-hand | — | Three merges 2026-07-29: `98be9d8` (feature-column fix + planted test), `5e82177` (enabled end-to-end), `12b4bfc` (state GMM inputs corrected, redundant `main.rs` wiring dropped). `936f7cb` was **not** merged, as instructed *(that ref no longer resolves in this checkout — the branch was never merged and has since been pruned; kept for provenance)*. Merged REGIME-1. |
| BUG-1 | Retrain/revalidate 3 ML algos vs current schema | TODO | P1 | S | in-hand | — | **Still open — the Jul-31 audit listed this DONE without evidence; that claim does not survive.** Verified 2026-08-06: no retrain commit exists, and no model artifact exists anywhere in the checkout (`models/` is gitignored, `.gitignore:81`), so the trained state lives off-git and is unauditable. `mean_reversion_detector` degrades to NaN without a loaded model (`:71-77`). Task now also includes: decide whether artifacts should be tracked or rebuilt from a pinned recipe. |
| COST-1 | Unify the two cost systems | DONE | P1 | S | in-hand | — | `3d50d82` / `f568bbf` (2026-07-28). `backtest/costs.py:43` now defaults `fee_bps` to the SSOT `taker_bps()` via `field(default_factory=...)`. |
| COST-2 | Remove zero-cost backtest fallback | DONE | P1 | S | in-hand | — | `3d50d82`. Precisely: the *fallback* is gone; `zero_cost()` survives as a **named, explicit-opt-in** preset (`costs.py:283-289`, registered as `"zero"` with the comment "explicit opt-in only — never an accidental fallback"). That is the intended end state, not a residual. |
| COST-3 | Purge hardcoded fee/slippage literals + CI grep guard | DONE | P1 | S | in-hand | — | `3d50d82` + CI guard `scripts/ops/check_no_hardcoded_costs.sh` (`ci.yml:120`). Known limit, by construction: it scans numeric literals, so wrong-preset *function calls* pass — that hole is what COST-7 closed. |
| COST-6 | Purge VIP9 cost defaults from every eval harness | DONE | P1 | S | in-hand | — | *(ID reassigned — see erratum; commits say "COST-4".)* `d9f3c1c` / `1334d41` (2026-07-30): all harness defaults resolve to the Hyperliquid SSOT (~11 bps RT); VIP9 is explicit-opt-in only. Covered `paper_trader_daily`, `cli/oos`, `cli/gauntlet`, `overnight_sweep`, `mf_liquidity_backtest`, `mf_hypothesis_suite`, `it_multiday`, `paper_trader_surprise`. Bonus: `overnight_sweep` printed `--cost-mode` but ignored it. |
| COST-7 | Harden the CI guard to catch wrong-preset *calls* | DONE | P1 | S | in-hand | COST-6 | *(ID reassigned — see erratum; commits say "COST-5".)* `d9f3c1c`; recurrence blocked by `scripts/tests/test_cost_defaults.py`. |
| COST-8 | `hyperliquid_maker()` hardcodes the 0.2 bps rebate | TODO | **P1** | XS | in-hand | — | **Found 2026-08-06 during this reconciliation.** `backtest/costs.py:269,274` construct `CostModel(fee_bps=0.2, ...)` as literals, bypassing `load_costs()` — a guardrail violation the COST-3 literal-scanner does not flag (it is a preset function, like the COST-6 hole). It hardcodes precisely the **rebate-tier-2 assumption §4.11 names the single most load-bearing unvalidated assumption in the stack** (worth ~1.7 bps/fill, larger than every measured gating effect combined). Route through `load_costs()` + `maker_tier_override` before any further maker number is produced. |
| COST-4 | Wave-gate thresholds → config | TODO | P2 | S | in-hand | — | *(Original COST-4 — this row keeps the ID.)* Gate literals in `evaluate_wave{1,2}_gate.py`. |
| BUG-4 | `optimal_entry` `sigma_process` hardcode → backtest/live parity | DONE | P1 | S | in-hand | — | *(Retro-row: named, fixed and committed 2026-07-30 without ever existing here — exactly the defect the ID-registry rule now prevents.)* `ba7b208` / `902ab1d`. |
| BUG-5 | `jump_detector` `run_batch` self-masking bipower + weak parity test | DONE | P1 | S | in-hand | — | *(Retro-row, same merge `ba7b208` / `902ab1d`.)* Current tick's return no longer embedded in its own bipower denominator; exact step/batch parity. |
| X-1 | `[hyperliquid_staked]` fee tier + reprice §4.7/§4.9 | DONE | P1 | S | in-hand | — | Ladder in `costs.toml` (wood 5 %→diamond 40 %, active tier `none`), tier-aware `utils/costs.py` + `tier_summary()` stamp, guards `tests/test_fee_tiers.py`, driver `execution/fee_tier_reprice.py`. **Verdict (FINDINGS §4.10): no cell flips** — 8 cells × 7 rungs × 179 episodes, 0 survivors; staking discounts don't reach maker rebates, so the maker line is fee-tier-invariant. |
| COST-5 | Quantify the maker *volume*-tier assumption | DONE | P1 | S | in-hand | X-1 | `[hyperliquid_maker_tiers]` ladder + `maker_tier_override`; §4.9 re-simulated at 4 rungs (FINDINGS §4.11). **Breakeven maker rate = +0.144/+0.159 bps — zero fees are ~0.08 bps/posting under water**; SSOT's +0.2 bps is rebate_t2 (≥1.5 % of venue maker volume, unearned). No cell survives at any rate; EV-gated cells are non-monotone (a bigger rebate loosens the gate and buys worse fills). |
| B-5 | Maker viability on wider-spread pairs | TODO | P1 | M | in-hand | XS-1 | The untested direction out of §4.11: breakeven maker rate scales with the half-spread, and every maker experiment so far ran on the 3 tightest symbols. Needs the Class-3 candle universe; no ingestor/streak dependency. |
| K3 | `regime_accumulation_score` constant (0.4429) | BLOCKED | P2 | XS | streak | BUG-3 | Likely auto-resolves with the GMM/K2 fix. |

## PROC — Process / IT discovery layer *(raise the ceiling on how NAT discovers edges)*

Detailed spec: [`specs/process_layer.md`](specs/process_layer.md). Critical path:
`PROC-12 → PROC-6 → PROC-7 → PROC-8 → PROC-1` — **complete as of 2026-08-05** (Phase 0 + Phase 1 of
the spec's roadmap, plus the compiler). The discovery loop now closes end to end: null-calibrated
measurement → conditional predictability → surface → a registered algorithm. **Next on this layer:**
`PROC-17` (targets as a first-class node, the substrate 5/6/7 currently work around), then `PROC-3`
(synergy-aware combiner — the unit that actually tests the orthogonality assumption) and `PROC-4`
(longitudinal MI stability, planted path runs now). All run on data in hand.
*(The `process_signal_design` S1–S9 series is folded here: S1/S2→PROC-3, S3→PROC-9, S4→PROC-12/13,
S5→PROC-4, S6→PROC-6, S9→PROC-8/11.)*

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| PROC-12 | Null-calibration layer ⭐ | DONE | P0 | S | in-hand| —| `it_engine/null_calibration.py` + `tests/test_null_calibration.py` (c0673e8). Shuffle-null → bits-above-null / z / p; thresholds in `config/it_engine.toml`. |
| PROC-6 | `conditional_predictability` process ⭐ | DONE | P0 | M | in-hand| PROC-12| `processes/conditional_predictability.py` + test (1d11990). `MI(f;label\|Z=z)` per bucket; registered. |
| PROC-5 | Schedule the 3-bar classifier as standing eval | DONE | P0 | XS | in-hand| —| `processes/standing.py` + `tests/test_standing_evals.py` (31bedc4). `barrier_3bar_mi` registered + `audit_standing_evals()`. |
| PROC-7 | Horizon/label MI-surface meta-process | DONE | P1 | M | in-hand| PROC-5,6,12| `processes/horizon_label_scan.py` + test (3252581). (horizon × geometry × regime) sweep. |
| PROC-13 | FDR/DSR on the process layer + cross-run ledger | DONE | P1 | S | in-hand| PROC-12| `processes/fdr.py` + `tests/test_process_fdr.py`. BH q per cell + append-only cross-run ledger. |
| PROC-4 | Longitudinal MI tracker (`mi_stability`) | DONE | P1 | M | ≥10 days | PROC-12 | `processes/mi_stability.py` + `tests/test_mi_stability.py` (12 tests). **One fold per calendar day, never pooled** — pooling lets between-day drift masquerade as prediction, and a planted test asserts a feature with zero within-day relation stays non-informative. Per-day null-calibrated (PROC-12) series + mean/cv/slope/frac_days_informative → verdict durable\|non_durable\|insufficient_days. Real run still wants ≥10 clean days; smoke on 13 BTC days (2026-07-12→08-04, 5.06 M rows) runs clean. |
| PROC-8 | Predictability surface + viz | DONE | P1 | M | in-hand| PROC-6,7| `processes/surface.py` + `tests/test_predictability_surface.py` (8850b14). `nat viz predictability`. |
| PROC-10 | Predictability half-life | TODO | P1 | S | ≥30 days | PROC-4 | `MI(t)` decay; feeds Q4 + LOOP-3. |
| PROC-3 | MI-maximizing nonlinear combiner (`cmi_select`+synergy) | DONE | P2 | L | in-hand | PROC-12,13 | `processes/mi_combiner.py` + `tests/test_mi_combiner.py` (19 tests). Pair-aware seed via the chain rule `I((a,b);y)=I(a;y)+I(b;y\|a)` recovers synergistic (XOR) pairs that `greedy_select` structurally cannot — the baseline's failure is asserted alongside. Redundancy penalised via interaction information; purged contiguous cross-fit GBDT emits `combo_mi` out-of-fold; scoring null-calibrated (PROC-12) vs the best single feature. **Decisive guard:** a shuffled label must NOT clear the null. |
| PROC-9 | Transfer-entropy causal graph (`lead_lag_te`) | TODO | P2 | M | in-hand | PROC-12 | Directed feature/symbol lead-lag (=S3); nonparametric Hasbrouck. Feeds HF6. |
| PROC-1 | Process→algorithm compiler | DONE | P2 | L | in-hand| PROC-6,7,12,13| `agent/algo_synth.py` + `tests/test_algo_synth.py` (34 tests). Compiles ONLY null-calibrated (PROC-12 z) + FDR-passed (PROC-13 q) findings carrying an explicit polarity — MI is unsigned, so a direction is never guessed; identifier-validated, no shadowing, deterministic render, kinds threshold/regime_gated (combiner → PROC-3). |
| PROC-11 | Two-stage regime-then-price system | TODO | P2 | L | in-hand | PROC-6 | Forecast regime → fire signal only in favorable regime. (=S9 signal_book) |
| PROC-2 | Self-explaining edges + reading notes | TODO | P2 | M | in-hand | — | Mechanism annotation per edge + 5 paper reading notes. |
| PROC-17 | Target as a first-class node (`targets.py`) | DONE | P1 | S | in-hand | — | `processes/targets.py` + `tests/test_targets.py` (26 tests). `Target` owns resolution (one precedence rule), materialisation, **its own leakage set** (label + `tb_*` siblings; price cols for returns), gate selection (label ⇒ null-z, return ⇒ fee) and signedness — the precondition for PROC-1 polarity. Missing/all-NaN/constant targets RAISE rather than degrade to forward returns. Wired into `ic_horizon`, `ml_importance`, `info_theory`. |
| PROC-15 | `residualize` transform (pure-innovation) | DONE | P2 | XS | in-hand | — | `processes/residualize.py` + `tests/test_residualize.py` (19 tests). `res_f = f - beta'Z`, beta fit on the **training prefix only**; the emitted finding is the **holdout** `\|corr(res,Z)\|` (prefix orthogonality is OLS arithmetic, not evidence). Degenerate/collinear conditioners refused, self-residualization dropped **with a recorded reason**, NaN-in→NaN-out. First real-data datum: BTC 2026-08-04, `imbalance_qty_l5 \| imbalance_qty_l1` — prefix corr 0.000 but holdout **0.192**, beta drifting +0.815→+0.729 *within one day*: linear orthogonalization between the imbalance cousins does not hold out of sample. |
| PROC-18 | `feature_ops` transforms (fractional-diff, etc.) | TODO | P2 | M | in-hand | — | Frac-diff / spectral / robust-norm operators (institutional GAP: frac-diff). |
| PROC-16 | `pca_combo` Marchenko–Pastur denoise param | TODO | P2 | XS | in-hand | — | MP eigenvalue clip on the existing `pca_combo` (=S8). |
| PROC-19 | `candles` data level + multi-symbol process loading | TODO | P1 | S | in-hand | XS-1 | Framework prerequisite for the three `xs_*` processes: the runner currently loads tick parquet for one symbol at a time. Flagged in `specs/maker_system.md` §7 as "its own task before implementation" — this is that row. |
| PROC-20 | `persistence_stats` — momentum runs + band excursion | DONE | P1 | M | in-hand | — | `processes/persistence_stats.py` + `tests/test_persistence_stats.py` (22 tests). Two families at **bar** level: `P(continue \| run length k)` + run-direction markout, and `k·sigma` band touches (markout, time-to-revert) with an embargo. Per-cell permutation null, BH-FDR across the grid, PROC-4 per-day verdicts. **Turns LF7's single-day priors (n=4–31/cell, "PRIORS ONLY") into a runnable study** — unblocks `LF7`. Touch price is a fill proxy that overstates fills; A4 gates any profit claim. Smoke: BTC 8 days @5min → 36 cells, **0 FDR discoveries**, run lengths 565/281/117/65/32 (geometric = random-walk signature). **Full study 2026-08-06 (FINDINGS §5): 185/219 episodes, 330 cells, 0 informative after FDR + day-durability.** Momentum is *anti*-persistent at bar scale (34/36 cells negative; 1 min z 6–10); LF7's k≈2.0–2.5 capture and SOL lead replicate qualitatively but the ordering is SOL>BTC>ETH and no cell survives (SOL k=2.5: +45.8 bps, q=0.33, fracD=0.00). LF7's missing ingredient is **events/day**, not a better k. |

## R1 — Three-class research program *(drained from `THREE_CLASS_RESEARCH_PROPOSAL.md`, 2026-08-06)*

The post-Q4 research program: **C** = Class-3 cross-sectional rotation · **A** = Class-1 directional
bias makers · **B** = Class-2 oscillation harvesters · **X** = shared execution research. Design
authority is [`specs/maker_system.md`](specs/maker_system.md) v2; the program and its kill criteria
are [`THREE_CLASS_RESEARCH_PROPOSAL.md`](THREE_CLASS_RESEARCH_PROPOSAL.md). Shared protocol for
**every** row here: walk-forward + embargo, DSR wherever Sharpe is claimed, **acceptance criteria
pre-registered in the driver/test file before results**, PROC gates for signal claims (null z ≥ 3,
BH-FDR q ≤ .05, ledger `data/processes/fdr_ledger.jsonl`), no post-hoc symbol/parameter/window
selection, every study cites its fee tier, costs only via `load_costs()`.

Track C leads: it is the only track with **zero data debt**, and `data/candles/` does not exist yet.

*`X-1` (staked fee tier) belongs to track X but keeps its existing row in the BUG/cost table above —
one row per ID, per Conventions. Verdict: **no cell flips at any rung** (§4.10).*

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| XS-1 | Universe candle backfill | WIP | **P0** | S | in-hand | — | **The unblocker for `B-5` and all of Track C.** Code shipped 2026-08-06: `--universe` / `--include-delisted` / `--max-symbols` / `--symbol-delay` on `data/fetch_candles.py` (`fetch_universe` + `backfill_universe`), guarded by `tests/test_universe_backfill.py` (27 tests). Live enumeration: **177 listed perps, 55 delisted excluded**. *Two premises in the 2026-08-06 reconciliation were wrong: `data/candles/` already existed (BTC/ETH/SOL × 1m,15m) and the universe is 177, not ~150.* **Remaining: run the bulk backfill** (177 × ≥90 d) — that is what flips this DONE. |
| XS-2 | Bar-level xs feature library | TODO | P1 | M | in-hand | XS-1 | `scripts/xs/features.py` (does not exist yet): permutation entropy, momentum slope×R², Hurst, vol percentile — **each vs the pair's own history** (rolling percentile/z), never cross-sectionally raw. Planted tests first. |
| XS-3 | `xs_rank_predictability` process | TODO | P1 | M | in-hand | XS-2,PROC-19 | Rank-IC of scores vs relative forward returns per rebalance interval; permutation null = **shuffled pair labels**; FDR across score variants. Verdict: is any score family significant at all? |
| XS-4 | `xs_persistence` process | TODO | P1 | S | in-hand | XS-2,PROC-19 | Rank autocorrelation half-life per score. **Must exceed the rebalance cadence or the rotation is churn by construction** — this is the row that can kill the track cheaply. |
| XS-5 | `xs_capacity_gate` process | TODO | P1 | S | in-hand | XS-1 | Spread/depth/volume floors from candle + `meta` data, SSOT-priced; admitted-universe list refreshed daily. Untradeable tails never reach ranking. |
| XS-6 | Rotation OOS study | TODO | P1 | M | in-hand | XS-3,4,5 | Top-k weighted paper rotation on candles, walk-forward, SSOT costs, pre-registered §4.9-style criteria. Promotes to lifecycle DISCOVERED iff it survives. |
| A-1 | `agreement_gate_eval` process + standing eval | TODO | P1 | S | in-hand | — | Conditional IC of the fast signal **given** slow-bias agreement vs disagreement, null-calibrated. Promotes the §5 pilot (the one structure with conditional-IC *above* unconditional) to a monitored fact — or kills it. |
| A-2 | **Combiner revalidation** | TODO | **P0** | M | in-hand | — | *The highest-value single experiment on the board.* §5's composite IC (BTC .178 / ETH .248 / SOL .359) rests on a **2-day** OOS with **monotonically rising fold ICs** — the source itself flags possible look-ahead/trend artifact, L1 dominance (ablation pending), SOL likely inflated, costs assumed not measured. Multi-day walk-forward via `nat oos --window` + DSR, pre-registered. **It is the last unrefuted capital-relevant claim in the record** — settle it before anything is built on top of it. |
| A-3 | Conditional-IC surface refresh | TODO | P1 | S | in-hand | A-2 | PROC-6/7 sweep with the §2 feature contract as conditioning, fee tier priced in. |
| B-1 | `oscillation_admission_eval` process | TODO | P2 | S | in-hand | — | Does admission (Hurst / band power / OU-τ) predict forward oscillation? Null-calibrated forward persistence. **Track B stops here if admission has no forward validity.** |
| B-2 | `band_geometry_scan` process | TODO | P2 | M | in-hand | B-1 | (k × dominant period × regime) capture-vs-adverse surface, FDR'd. **Geometry is read off the surface, never swept** — the LF7 k≈2.0–2.5 prior is confirmed or replaced by the argmax. |
| B-3 | LF7 signal layer | TODO | P2 | M | in-hand | B-2 | Band/channel algorithm per `contracts/algorithm.md`, planted tests, k from B-2. |
| B-4 | Band study OOS | TODO | P2 | M | in-hand | B-3 | Multi-day pre-registered signal-level verdict; economics deferred to X-3. |
| X-2 | **F-task plan** — L1 queue sizes + per-tick side volume | TODO | P1 | S | in-hand | — | **Plan only; implementation is its own approval** (guardrail: plan before any feature-vector/schema change — it ripples to Parquet, `names_all()`, and every reader). This is the data that would replace the §4.7 flow/depth *proxies*, which caveat every maker number NAT has produced. |
| X-3 | Fill-economics reruns on real fill data | BLOCKED | P1 | M | streak | X-2 | **The maker go/no-go.** §4.7/§4.9/HF5b re-run on F-task data or T0b shadow quotes. Data-gated: per §4.9, the decisive unblocker is data, not more sim variants. |

---

## Verified shipped (dropped, not carried over)

Confirmed present in code during consolidation, so **not** listed above (sources archived):
CLI modularization (NAT1–7, NAT10 = D2), all 5 F-features (`settlement_clock`, `microprice`,
`multilevel_ofi`, `har_rv`, `realized_moments`), algos `relative_value_pairs`/`vol_squeeze`/
`funding_settlement`, costs+provenance+`signal_lifecycle`+`promotion_daemon`+`kill_switch`+
`nat oos --window`, the convolver pipeline (14 stages), nan-wiring steps 01–04, and the cloud/swarm
stack (Caddy, `nat docker`, swarm CLI, Optuna CMA/TPE/NSGA-II). `process_concept` (process as 3rd
citizen) shipped as `scripts/processes/`. `korrektur` K1/K5 marked fixed (K5 re-opened as REL-2).

*Sources drained into this file are moved to `archive/` (docs-restructure Phase 3). The Tier-A finding
reports (features_report, algorithms_report, data_inventory, situation_analysis, IC scans) were
merged into [`research/FINDINGS.md`](research/FINDINGS.md) (Phase 4, 2026-07-25) and their sources
archived. See [`DOCS_RESTRUCTURE_PROPOSAL.md`](DOCS_RESTRUCTURE_PROPOSAL.md).*

---

## Reconciliation log

**2026-08-06 — P3 + P1 of [`DOCS_IMPROVEMENT_PLAN_PROPOSAL_V1.md`](DOCS_IMPROVEMENT_PLAN_PROPOSAL_V1.md).**
Every status claim in that audit was re-verified against code and git history rather than trusted;
17 rows flipped, 8 rows added, 1 ID collision resolved by erratum, the execution order rewritten for
the post-Q4 strategy. **Three results the audit itself got wrong:**

1. **`BUG-1` is not done.** The audit listed it DONE. There is no retrain commit, and no model
   artifact exists in the checkout at all (`models/` is gitignored) — the claim had no evidence and
   the row stays open.
2. **`HF4` is not a shipped unit.** It exists only as a `use_hf4_gate` flag over an externally
   supplied `gate_open` array inside the touch-maker sim; no registered `toxicity_gate` algorithm
   exists. WIP, not DONE.
3. **`REV-1` is still open** — the §4.1-derived numbers were fenced in FINDINGS but never swept from
   `reports/`, notebooks, or `research/ALGORITHMS.md`.

**One new defect found while verifying:** `COST-8` — `hyperliquid_maker()` hardcodes the 0.2 bps
rebate (`backtest/costs.py:269,274`), bypassing `load_costs()`, and it is exactly the rebate-tier-2
assumption §4.11 identifies as the most load-bearing unvalidated number in the stack. The COST-3
literal-scanner cannot see it — same blind spot as the VIP9 preset calls that COST-7 closed.

*Next reconciliation due: 2026-08-13 (weekly, per Conventions).*
