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
1. ~~`XS-1`~~ ✅ **DONE 2026-08-07** — 708 series, 3.06 M candles, zero gaps. It also *changed the
   plan*: the venue caps history at ~5000 bars/interval, so 1m reaches 3.5 d and **cannot be
   backfilled further, ever** (FINDINGS §7).
2. `XS-7` — **daily candle refresh cron.** Promoted to first place by that cap: 1m universe history
   can only be *accumulated*, so every day without it is a permanently lost day. Cheap, urgent.
3. `XS-8` — REST `l2Book` universe sampler. **`B-5` cannot proceed without it** (see below).
4. `B-5a` → `B-5b` — maker viability on wider-spread pairs, now correctly split: the arithmetic
   screen first (it can kill the hypothesis without a single simulation), the tick sim only after.
5. ~~`COST-8`~~ ✅ **DONE 2026-08-06** — and it uncovered a sign inversion (a rebate booked as a
   charge, 0.4 bps/side against a +0.144 bps breakeven).

**2 — Class 3: the cross-sectional rotation track** *(data-independent; runs parallel to 1)*
6. `XS-2` — bar-level xs feature library (planted tests first). **Start at 15m/1h, not 1m** — the
   retention cap means 1m has 3.5 d while 15m has 52 d and 1h has 90 d. PROC-20 independently found
   1m/5m momentum anti-persistent with 5m unresolvable, so the coarser bars are the right entry
   anyway. Make the bar a parameter, not an assumption.
7. `PROC-19` — `candles` data level + multi-symbol loading in the process runner (framework
   prerequisite for the three xs processes).
8. `XS-3` / `XS-4` / `XS-5` — rank-predictability, persistence, capacity gate.
9. `XS-6` — rotation OOS study, pre-registered. Promotes to lifecycle DISCOVERED iff it survives.

**3 — Class 1 signal layer** *(data-independent)*
10. ~~`A-2` — combiner revalidation~~ ✅ **DONE 2026-08-08 — REFUTED** (FINDINGS §5.1). Walk-forward
   IC +0.062/+0.099/−0.024 vs §5's .18/.25/.36, and the composite never beats its own best single
   feature. The weights were fitted *after* the window they were scored on (`training_date`
   2026-06-11 vs an OOS of 06-08→10) — the monotone fold ICs were the tell.
11. ~~`A-1` — agreement-gate standing eval~~ ✅ **DONE 2026-08-08 — the gate is HARMFUL**: on ETH/SOL
   the *disagreement* subset carries the signal (lift −0.067/−0.057, z −2.91/−2.71). **§5 is fully
   retired, and with it the last unrefuted capital-relevant claim in the record.** `A-3`
   (conditional-IC surface refresh) remains open but now measures a surface whose headline
   architecture is gone — reconsider before building it.
12. `PROC-9` / `PROC-11` / `PROC-18` — TE causal graph, two-stage regime-then-price, feature ops.

**4 — Reliability & the data foundation** *(ops; gates everything paper/live)*
13. `REL-4` — verify Telegram delivery end-to-end (creds are user-side; a <5 min alert that doesn't
    page is worthless).
14. `Q1` — deploy the T0b Hetzner ingest box; kills the su-35 SPOF.
15. `Q0` — verify the clean-data streak (`/streak`) — the master gate for tier 5.

**5 — Streak-gated validation**
16. `PROC-10` — predictability half-life (≥30 clean days); feeds `LOOP-3`.
17. `Q-K2` — dead-feature / concentration production verdict (48 h on T0b).
18. `X-3` — fill-economics reruns on `X-2` F-task data or T0b shadow quotes — **the maker go/no-go**.
19. `Q5` — conditional-IC > 0.15. The trading-business gate; consumes the three-class program.

**6 — Capture & scale** *(only if `Q5` positive)*
20. `QA` — remaining institutional-GAP algos; `G8` paper window → live capital, gated on G8 + a
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
| A5 | TWAP/VWAP slicing + hysteresis no-trade bands | DONE | **P1** | S | in-hand | — | **Shipped 2026-08-07 (FINDINGS §7.9).** `execution/rebalance.py` + 19 tests: `no_trade_band`, `trade_to_edge` (the Constantinides proportional-cost optimum — move to the *boundary*, not the target), `band_from_cost`, `twap_slices`, `vwap_slices`. Measured on the §7.8 rotation: **cost saving is real and monotone** (turnover 0.199 → 0.018, cost 1.10 % → 0.10 %) but **gross swings 7.25 → 12.58 non-monotonically**, several times the cost saved — so at n=83 the net effect is noise and **no band is adopted** (picking the SR-2.99 cell would be fitting 7 configs on one window, the §4.6 pattern). **Two of my own defects found by disbelieving a good number:** gross was priced on *target* rather than *held* weights (free money — latent at band=0, now regression-tested), and `band_from_cost` returned a dimensionless ratio (~1.4) that as a weight band means *never trade*. **TWAP/VWAP ships with no performance claim** — slicing reduces impact and the cost model has no impact term, so it measures exactly zero until `X-3`. |
| A4 | Queue-value execution model | DONE | P2 | M | in-hand | — | `execution/queue_value.py` (`344a22a`). Produced the platform's only +EV number (+0.036/+0.013 bps per posting, §4.7) and the **EV gate that flips V1's per-fill sign** (−1.66 → +0.67, §4.9). Threshold must be **re-derived per fee tier** — it is non-monotone in the maker rate (§4.11). |
| A2 | Macro/daily mean-reversion algorithm | TODO | P2 | S | in-hand | — | Premium/basis reversion on the settlement-clock feature. |
| LF2 | OI-positioning-extremes algorithm | TODO | P2 | S | in-hand | — | `oi_divergence` failed; extreme-positioning variant not built. |
| WP-1 | Wallet roster (derived, never hardcoded) | TODO | P1 | S | in-hand | — | Step 1 of `specs/wallet_positioning.md`. Sources: venue leaderboard (verify live), `WsTrade.users` aggregation, pinned `config/wallets.toml`. Address validation before any path/API use; retry on transport faults only. |
| WP-2 | Position snapshot collector + clock | TODO | **P1** | S | in-hand | WP-1 | Step 2. `data/fetch_positions.py` on the `fetch_l2.py` pattern; 15-min sweeps → `data/positions/`. **Writes `status=failed` rows** (XS-8 drops them, so unreachable is indistinguishable from flat). systemd unit with `StartLimitIntervalSec` in `[Unit]`. **Every uncollected day is permanently lost** — start it early. |
| WP-3 | Cohort construction | TODO | P1 | M | ≥30 days | WP-2 | Step 3. Walk-forward P&L ranking; cohorts ranked on a window ending strictly *before* `as_of` — an in-sample ranking is the A-2 error (§5.1) in new clothing. Reports rank-stability, since "the cohort is not a cohort" is failure mode 1. |
| WP-4 | `cohort_predictability` process | TODO | P1 | M | ≥30 days | WP-3 | Step 4. Registered `EvaluationProcess`; reuses PROC-17 targets, PROC-12 null (**permute cohort labels**, size-preserving — A-1's lesson), PROC-13 FDR, PROC-4 day folds. Non-overlapping sampling mandatory (A-2's 0.39→0.06 inflation). |
| WP-5 | Pre-registered positioning study | TODO | P1 | S | **≥90 days** | WP-4 | Step 5. Criteria committed to git before the run (XS-6 standard). **Gated on ≥90 days of WP-2 collection** — `n ∝ 1/SR²` and a 24 h horizon gives ~1 obs/day/coin, so running early manufactures exactly the suggestive number that died three times this week. |
| LF6 | HAR-RV sizing (non-directional) | WIP | P2 | S | in-hand | — | Feature `har_rv.py` done; sizing wiring into `meta_portfolio`/kill-switch unverified. |
| LF7 | VWAP-SD channel maker (band mean-reversion) | TODO | P2 | M | in-hand | HF4,A4 | Spec `docs/research/new/vwap_sd_channel.txt`. k swept not fixed; single-day priors: k≤1.5 adverse, capture at k≈2.0–2.5, SOL-led; maker rebate +0.4bps RT, binding cost = adverse selection; queue-sim (A4) gates any profit claim. |
| VW-1 | `vwap_multiscale` offline transform | DONE | **P1** | S | in-hand | — | **Shipped 2026-08-09.** `features/vwap_multiscale.py` + `tests/test_vwap_multiscale.py` (31). 1-min bucketed ring, ~12 KB/symbol vs ~17 MB of trades; refuses partial windows, feed gaps and forward-fills. New columns use the intuitive `(price−vwap)/vwap`; the shipped inverted `flow_vwap_deviation` is left alone and both conventions are pinned in a test. **Smoke found the real blocker: on 2026-08-07 BTC the trade feed has only 49.7 % active minutes and a 586-min hole, so 12 h is 0 % available and 6 h is 6.4 %.** Windows ≤1 h are 82–99 % available. |
| VW-2 | Multi-scale VWAP study | DONE | **P1** | M | in-hand | VW-1 | **Run 2026-08-09: 0/6 windows pass (§7.12).** Binding gate is day-consistency (best frac 0.16 vs ≥0.55) on a 29–32 %-active archive; nested windows fail redundancy (0.81–0.92), 1h/2h fail event rate. On the record, not promoted: 2h pooled z 7.1/4.3/4.0 cross-symbol, least redundant vs 5m (corr 0.21) — **re-run this exact driver after a clean ≥30-day streak.** Methods: strided non-overlapping targets are mandatory (dense rows gave z 50–70 out of nothing); criterion (d) literal residual-corr is degenerate on duplicates — spec to amend. Original scope for reference: Step A2. PROC-4 durability + PROC-20 band structure + PROC-15 redundancy per window, criteria committed **before** the run. A window earns a column only if informative (z≥3), day-durable (≥0.55), FDR-surviving, **non-redundant vs the faster window**, and with an event rate that makes a verdict reachable (PROC-20's ~1.5/day is the counter-example). **Scope narrowed by VW-1's smoke: evaluate 5m/10m/15m/1h only — 6h/12h are not computable on the current archive (§7 gaps), so they cannot earn a column regardless of their information content. Re-open them once `REL-4`→`Q1` produces a clean streak.** |
| VW-3 | Feature-vector migration (schema change) | DONE | P2 | M | in-hand | VW-2 | **Closed 2026-08-09 without migration: 0/6 windows passed VW-2 (§7.12), so per the spec this closes with a FINDINGS entry and the ingestor is untouched.** Re-opens only if the post-streak VW-2 re-run changes the verdict. Original scope for reference: Step B — **only for windows that pass VW-2; if none pass this closes with a FINDINGS entry and the ingestor is untouched.** Guardrail: ripple analysis before code (236→248 max), readers must tolerate absent columns (44 d of existing parquet lack them). Rust `VwapRing` of 1-min buckets — **not** a widened `TradeBuffer`: six O(n) `trades_in_window` scans per symbol at 10 Hz would blow the 80 ms/tick p99 budget. |
| TC-1 | Trend continuation 15m/1h × universe | DONE | P2 | M | in-hand | XS-1 | **Run 2026-08-09: the band does not exist (§7.13).** PROC-20 momentum family, fresh 177-pair fetch (15m ~52 d, 1h ~90 d), gate = next-bar continuation (non-overlapping), one BH family over 1,739 cells, sweep in the FDR ledger. 81 % of cells negative, z ≤ −2 vs z ≥ +2 asymmetry 36:1, 0 durable cells. Trend-following now has a negative record at every horizon 1m→1d; the reversal sign feeds Track B/C mean reversion. `exploration/trend_continuation_study.py` + 7 planted tests. |
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
| REV-1 | Purge/re-run every §4.1-derived number still cited at SSOT cost | DONE | P1 | M | in-hand | COST-6 | **Done 2026-08-07.** Scope turned out narrower and differently shaped than the row assumed. The refuted §4.1 *P&L figures* survive in only 4 files — `FINDINGS.md` and `ALGORITHMS.md`, both already carrying prominent refutation banners, plus two under `archive/` which *should* keep them. **The real exposure was elsewhere:** five artifacts in `reports/` still carried VIP9 (1.61 bps) pricing as unlabelled machine-readable JSON — `best__mf_liquidity_signal.{json,md}`, `mf_liquidity_updated.json`, `it_multiday_btc.json`, `ic_scan_btc.json` — exactly what a future sweep or agent would ingest as current. **Stamped, not deleted:** one is a live input to `analysis/mf_liquidity_backtest.py:430`, so deletion would have broken a code path, and §4.6's lesson is that *unrecorded* negatives return as false positives. Each carries `_superseded` with reason, authority (§4.6) and task; `best__mf_liquidity_signal.json` is noted as less wrong than its siblings since it reports three fee models including Hyperliquid. **Recurrence blocked** by `tests/test_report_provenance.py` (3), which fails any VIP9-priced report lacking a stamp — the COST-7 pattern. |

## D — Development / platform *(harden & ship `nat`)*

| ID | Title | Status | Prio | Eff | Data | Dep | Notes |
|----|-------|--------|------|-----|------|-----|-------|
| D1 | Viz set + maturity tags | WIP | P1 | M | in-hand | — | Foundation + features/algorithm/paper/portfolio viz shipped. **Remaining: spectral/regime/correlation viz (NAT8) + `[PROVEN]/[PRELIM]/[SPEC]/[LIVE]` command tags (NAT9).** Renders PROC-8 surface. |
| D2 | Modularize the `nat` monolith | DONE | P1 | L | in-hand | — | Verified: ~50 `scripts/cli/*.py` + `app.py` assembler (NAT10). *Kept for provenance; sweep to archive.* |
| D3 | Ship `nat` apt-installable | TODO | P2 | L | in-hand | — | Phase 1 = relocatable paths (XDG/`NAT_HOME`); then pipx/wheel; then `.deb`+apt repo. |
| D4 | Continuous-discovery → cloud research lab | WIP | P2 | L | in-hand | — | Harden `discovery_orchestrator` + 4 agents; surface via `api` + Next.js. Partially built. |
| DOCS-1 | Refresh `PLAN.md` §0/§3 + status corrections | TODO | P2 | S | in-hand | — | Mark D2/kill-switch/T5/T14 done; fix ~4-week staleness of the pinned block (docs-restructure Phase 5). |
| DOCS-3 | Fix `commands.md` + `CLAUDE.md` operational sections | DONE | P2 | S | in-hand | — | **Done 2026-08-07 by generation, not transcription.** The drift was worse than the row implied: **26 command groups absent** from `commands.md` and the headline count stale by 80 (`~260` vs **340**, in both `CLAUDE.md:37` and `README.md:47`). Hand-writing 26 groups would have fixed today and rotted by next week — the exact failure `README.md` names when it argues the doc map should be *generated* so it never drifts. So `ops/gen_commands_doc.py` renders the reference from the live argparse tree (`nat --json commands`): 340 commands, 72 groups, 789 lines, with a `--check` mode. **Recurrence blocked** by `tests/test_commands_doc.py` (3) — regeneration must be a no-op, the file must declare itself generated, and the headline count must match the CLI. |
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
| HYG-8 | `requirements.lock` does not describe what the code imports | DONE | **P1** | S | in-hand | — | **Found 2026-08-07 building the first venv on su-75 from `requirements.lock`** — the lock is not reproducible. **(a) `toml`:** ~10 modules import it — `config_utils.py`, `risk/kill_switch.py`, `promotion_daemon.py`, `it_engine/config.py`, `it_engine/null_calibration.py` (PROC-12) — but the only related pin is `tomli>=2.0.0; python_version < "3.11"`, a **no-op on 3.12**, so a clean install cannot import the kill switch or the null-calibration layer. Python 3.11+ ships `tomllib` in the stdlib, so the right fix is likely migrating readers to `tomllib` rather than pinning the unmaintained `toml`. **(b) `optuna`:** imported by `swarm/optuna_optimizer.py`, absent from the lock — the documented `nat evolve` Tier-3 feature cannot run from a clean install. Both surfaced as collection errors (`test_tier2_swarm.py`, `test_tier3_optuna.py`). **DONE 2026-08-07:** `toml>=0.10.2` + `optuna>=4.0.0` in `requirements.txt`, pinned in `requirements.lock`; both suites now collect and pass (74 tests). **The tomllib idea in the original note is refuted:** `swarm/{config_generator,optuna_optimizer,orchestrator}.py` call `toml.dump()` and stdlib `tomllib` is read-only, so `toml` is a genuine dependency, not a legacy one. |
| HYG-9 | **The suite was red on master** — 16 failed / 38 errors | DONE | **P1** | M | in-hand | HYG-8 | **Measured 2026-08-07** on `ba0a253` in a clean venv: `4347 passed, 16 failed, 38 errors`; CI's `--ignore` list covered none of them, so CI was failing on master. **All 54 cleared.** The headline is that the *platform* was healthy — `config/agent.toml` satisfies its validator and every daemon constructs — but the tests had rotted behind two completed migrations (file-registry → SQLite; module-global `ROOT` → `ResearchAgent.root` property). Fixed at the seam, never by weakening an assertion: 38 agent errors (fixtures patched a vanished global); macro/mf registry tests rewritten against `store.append_signal`/`load_registry` and constructed the way `create_runner` does (store **and** agent identity); `test_alpha_pipeline` pointed at production's own `_make_state`; config fixtures completed against a grown `_REQUIRED_KEYS`; snapshot baseline regenerated after verifying the delta (`process standing`, `viz predictability` — both legitimate); curated `nat help` entries. **Three real defects surfaced, all fixed with tests:** (1) **HF1 microprice** never NaN-blanked its `run_batch` warmup — a `contracts/algorithm.md` violation in the maker-line anchor; (2) **`load_agent_config` crashed on `symbols = [...]`**, the array form `config/symbols.toml` itself uses — `"primary" not in <list>` is a membership test that returns True, then `list["primary"] = ...` raises; production dodged it only because `agent.toml` omits `symbols`; (3) **`AlphaPipelineState` history diverged by backend** — the SQLite path wrote to the history table but never updated `_data["history"]`, so `get("history")` returned a full list under JSON and `[]` under SQLite until the object was rebuilt. Also found: `_make_state` derives its DB as `state_file.parent.parent/nat.db`, so a flat test `state_file` put every test's DB in the shared pytest tmp root (fixed in the fixture). |
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
| BUG-1 | Retrain/revalidate 3 ML algos vs current schema | WIP | P1 | S | in-hand | — | **First leg done 2026-08-09 (§7.14): `mean_reversion_detector` retrained on current schema, `--start-date 2026-07-14` — OOS AUC 0.506/0.519/0.530 (BTC/ETH FAIL vs the trainer's own 0.52 gate, SOL barely PASS); at chance on 2 of 3, stays research-only. Remaining: `meta_labeling` (runnable); `regime_conditioned_lgbm` blocked by BUG-6 (regime_ inputs all-NaN since 07-26).** Was: the Jul-31 audit listed this DONE without evidence; that claim did not survive. Verified 2026-08-06: no retrain commit exists, and no model artifact exists anywhere in the checkout (`models/` is gitignored, `.gitignore:81`), so the trained state lives off-git and is unauditable. `mean_reversion_detector` degrades to NaN without a loaded model (`:71-77`). Task now also includes: decide whether artifacts should be tracked or rebuilt from a pinned recipe. |
| BUG-6 | `regime_` category dead in production since 2026-07-26 | TODO | **P0** | M | blocked-freeze | — | **Found 2026-08-09 (FINDINGS §7.11).** `regime_divergence_1h` and `regime_absorption_1h` are **0 finite / 1,444,319 tick rows** — the whole 23-feature optional category is NaN in every file since 07-26. Last alive **07-18**; the 07-19→07-25 outage hides the transition. Not a first occurrence: **46 of 75 days alive**, flapping since 04-19. **It silently hollowed out `A-1`** — only 4 days of that study's stated 25-day window carry a live gate input, which is why every cell returned `insufficient_days`/`non_durable`. Cause undetermined: `OPS-1`/`OPS-2` land 07-27, *after* the first dead file, so neither is implicated. Hypothesis to test on contact: `regime_` needs 1 h/4 h/24 h accumulators (`regime/mod.rs:13`), so frequent supervisor restarts would starve warmup permanently. **Diagnosis blocked by the su-35 freeze** — do not touch; verify on T0b instead. Add a `nat data validate` all-NaN-category check so a dead category pages instead of being discovered by a downstream study. |
| BUG-7 | `agreement_gate_eval` unrunnable via its own CLI defaults | TODO | P1 | XS | in-hand | — | **Found 2026-08-09 (FINDINGS §7.11).** The process declares `data_level = "bars"` but defaults to tick-level names: `slow="regime_divergence_1h"` (bars carry `regime_divergence_1h_last`) and `fast="alg_mp_dev_ema"`, which **exists in no feature file** — `alg_` columns are algorithm runtime outputs, never persisted. So `nat process run agreement_gate_eval` fails on its own defaults while its 15 planted tests pass, because they name their own columns: the planted layer and the real-data path never met (**real-parquet smoke, skipped**). Two adjacent defects: an unrunnable process returns `n_tested: 0, error: null` (success with nothing in it, the XS-10 shape), and `config/processes.toml`'s `max_memory_mb = 4000` silently truncated a 35-day request to **78 bars** — under `min_obs`, so every horizon skipped. `nat process run` exposes no memory flag. |
| COST-1 | Unify the two cost systems | DONE | P1 | S | in-hand | — | `3d50d82` / `f568bbf` (2026-07-28). `backtest/costs.py:43` now defaults `fee_bps` to the SSOT `taker_bps()` via `field(default_factory=...)`. |
| COST-2 | Remove zero-cost backtest fallback | DONE | P1 | S | in-hand | — | `3d50d82`. Precisely: the *fallback* is gone; `zero_cost()` survives as a **named, explicit-opt-in** preset (`costs.py:283-289`, registered as `"zero"` with the comment "explicit opt-in only — never an accidental fallback"). That is the intended end state, not a residual. |
| COST-3 | Purge hardcoded fee/slippage literals + CI grep guard | DONE | P1 | S | in-hand | — | `3d50d82` + CI guard `scripts/ops/check_no_hardcoded_costs.sh` (`ci.yml:120`). Known limit, by construction: it scans numeric literals, so wrong-preset *function calls* pass — that hole is what COST-7 closed. |
| COST-6 | Purge VIP9 cost defaults from every eval harness | DONE | P1 | S | in-hand | — | *(ID reassigned — see erratum; commits say "COST-4".)* `d9f3c1c` / `1334d41` (2026-07-30): all harness defaults resolve to the Hyperliquid SSOT (~11 bps RT); VIP9 is explicit-opt-in only. Covered `paper_trader_daily`, `cli/oos`, `cli/gauntlet`, `overnight_sweep`, `mf_liquidity_backtest`, `mf_hypothesis_suite`, `it_multiday`, `paper_trader_surprise`. Bonus: `overnight_sweep` printed `--cost-mode` but ignored it. |
| COST-7 | Harden the CI guard to catch wrong-preset *calls* | DONE | P1 | S | in-hand | COST-6 | *(ID reassigned — see erratum; commits say "COST-5".)* `d9f3c1c`; recurrence blocked by `scripts/tests/test_cost_defaults.py`. |
| COST-8 | `hyperliquid_maker()` hardcodes the 0.2 bps rebate | DONE | **P1** | XS | in-hand | — | **Fixed 2026-08-06.** Both presets + `from_config(role='maker')` now read the COST-5 tier ladder, so a rung change ripples instead of needing an edit. **Second defect found while fixing: the sign was inverted** — `CostModel.fee_bps` is a COST while `utils.costs.maker_bps()` is a REBATE EARNED, so a 0.2 bps rebate was booked as a 0.2 bps charge (a 0.4 bps/side error against a +0.144 bps breakeven, §4.11 — enough to flip a maker verdict alone). `CostModel` now accepts a negative fee down to `MAX_REBATE_BPS`; the non-negative guard became a floor, keeping its typo-catching intent. Guards `tests/test_maker_cost_preset.py` (13). |
| COST-4 | Wave-gate thresholds → config | TODO | P2 | S | in-hand | — | *(Original COST-4 — this row keeps the ID.)* Gate literals in `evaluate_wave{1,2}_gate.py`. |
| BUG-4 | `optimal_entry` `sigma_process` hardcode → backtest/live parity | DONE | P1 | S | in-hand | — | *(Retro-row: named, fixed and committed 2026-07-30 without ever existing here — exactly the defect the ID-registry rule now prevents.)* `ba7b208` / `902ab1d`. |
| BUG-5 | `jump_detector` `run_batch` self-masking bipower + weak parity test | DONE | P1 | S | in-hand | — | *(Retro-row, same merge `ba7b208` / `902ab1d`.)* Current tick's return no longer embedded in its own bipower denominator; exact step/batch parity. |
| X-1 | `[hyperliquid_staked]` fee tier + reprice §4.7/§4.9 | DONE | P1 | S | in-hand | — | Ladder in `costs.toml` (wood 5 %→diamond 40 %, active tier `none`), tier-aware `utils/costs.py` + `tier_summary()` stamp, guards `tests/test_fee_tiers.py`, driver `execution/fee_tier_reprice.py`. **Verdict (FINDINGS §4.10): no cell flips** — 8 cells × 7 rungs × 179 episodes, 0 survivors; staking discounts don't reach maker rebates, so the maker line is fee-tier-invariant. |
| COST-5 | Quantify the maker *volume*-tier assumption | DONE | P1 | S | in-hand | X-1 | `[hyperliquid_maker_tiers]` ladder + `maker_tier_override`; §4.9 re-simulated at 4 rungs (FINDINGS §4.11). **Breakeven maker rate = +0.144/+0.159 bps — zero fees are ~0.08 bps/posting under water**; SSOT's +0.2 bps is rebate_t2 (≥1.5 % of venue maker volume, unearned). No cell survives at any rate; EV-gated cells are non-monotone (a bigger rebate loosens the gate and buys worse fills). |
| B-5 | Maker viability on wider-spread pairs *(umbrella — see B-5a/B-5b)* | TODO | P1 | M | mixed | XS-8 | The untested direction out of §4.11: breakeven maker rate scales with the half-spread, and every maker experiment so far ran on the 3 tightest symbols. **Corrected 2026-08-07: "needs the Class-3 candle universe" was wrong.** The §4.9 harness consumes tick-level `raw_spread` / `imbalance_qty_l1` / depth from `data/features/` (`touch_maker_experiment.py:38`), and the ingestor covers only BTC/ETH/SOL — candles contain neither spread nor depth, so "swap the universe, keep the grid" is not executable. Split below. |
| B-5a | Wide-pair breakeven **arithmetic screen** | WIP | **P1** | S | in-hand | XS-8 | **Unit shipped + first run 2026-08-08 (FINDINGS §7.10).** `xs/breakeven.py` + `tests/test_xs_breakeven.py` (23). Emits **no survivor count** — only β\*, the exponent at which `E[adverse|fill] = A_btc·(h/h_btc)^β` flips the verdict; median **β\* = 0.698**, so the hypothesis reduces to *does adverse selection scale slower than h^0.70?* Capacity is the harder blade: at a $5k touch floor only 4/177 pairs admit and **none are wide**; at $1k, 18 admit (10 wide, 10 survive @β=0.75). Spread and depth are **uncorrelated** (ρ=−0.107, p=0.16) — the whole universe is thin at the touch, not just the wide pairs. **Stays WIP: the run is ~18 sweeps over ~1.5 h, not the multi-day sample this row requires** → `B-5c`. |
| B-5b | Wide-pair maker **simulation** | BLOCKED | P2 | L | streak | B-5a | Only for pairs surviving `B-5a`. Requires those symbols in `symbols.toml` (already arbitrary-N, `config.rs:195`) **and then collection time** — this is forward data accrual, not in-hand work, and it competes with the ingestor's capacity/streak. Do not start before `B-5a` returns a shortlist. |
| B-5c | Re-run B-5a on a multi-day L2 sample | TODO | **P1** | XS | in-hand | B-5a | **Minted 2026-08-08.** §7.10's first reading is ~18 sweeps over ~1.5 h of one weekday — no intraday, weekday or regime variation. (The 50-pair coverage gap in the 12-sweep read resolved itself as sampling continued, which is itself an argument for length.) The XS-8 sampler now runs as a systemd unit, so this is *waiting*, not building: re-run `xs/breakeven.py` at ≥3 days and check whether the β\* distribution, the depth-floor curve and the ρ(spread,depth)≈0 result hold. **Three suggestive numbers this week did not survive their own study** (§5 orthogonality drift, PROC-4 durability smoke, §5 band markout) — nothing from §7.10 promotes until this runs. |
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
| PROC-19 | `candles` data level + multi-symbol process loading | DONE | P1 | S | in-hand | XS-1 | **Shipped 2026-08-07:** `processes/candles.py` (`load_candles` / `available_candle_symbols` / `CANDLE_PRICE_COL`) + `_run_candles_process` in the runner + `data_level="candles"` and `ctx.symbols` on the process contract; `tests/test_candle_data_level.py` (14). **The shape is the design decision.** The archive holds 177 pairs with unequal histories, and both natural implementations are wrong: an **inner join** truncates the whole panel to the newest listing (one 27-day coin costs 175 pairs their history, silently), while an **outer join with fill** invents prices for pairs that had not listed — a lookahead that would inflate any rank-IC study. So the loader returns a **long frame where absence is absence**, and both wrong shapes are asserted against. Price column is `close`, not the tick default `raw_midprice` (carrying that across would make every price lookup NaN and return empty findings rather than erroring). Missing symbols are **named**, never silently dropped. Real smoke: 177 symbols / 379,688 rows in 0.7 s, cross-section width 175→177 as late listings join, CASHCAT's first bar 2026-07-11 against a panel start of 2026-05-09 — not backfilled. Unblocks `XS-3`/`XS-4`/`XS-5`. |
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
| XS-1 | Universe candle backfill | DONE | **P0** | S | in-hand | — | **The unblocker for `B-5` and all of Track C.** Code shipped 2026-08-06: `--universe` / `--include-delisted` / `--max-symbols` / `--symbol-delay` on `data/fetch_candles.py` (`fetch_universe` + `backfill_universe`), guarded by `tests/test_universe_backfill.py` (27 tests). Live enumeration: **177 listed perps, 55 delisted excluded**. *Two premises in the 2026-08-06 reconciliation were wrong: `data/candles/` already existed (BTC/ETH/SOL × 1m,15m) and the universe is 177, not ~150.* **Backfill run 2026-08-07** (su-75): 177 pairs × {1m, 5m, 15m, 1h} = **708 series, 3,059,200 candles, 98 MB**, and **every series is 100 % complete within its span — zero gaps anywhere**, which is a categorically better substrate than the tick record (§7: 37 % of days missing). 1h reaches the full 90 d on 175/177 (GRAM 36 d and CASHCAT 27 d are recent listings). **But the headline requirement could not be met, and cannot be:** the venue caps candle history at **~5000 bars per interval**, so 1m reaches only **3.5 days** — measured, not inferred (a narrow 2 h window 4 d back returns zero while 1h at 89 d back returns fine). See FINDINGS §7. Depth by bar: 1m 3.5 d · 5m 17.4 d · 15m 52 d · 1h 90 d (208 d available). |
| XS-7 | Daily candle refresh cron | DONE | **P0** | XS | in-hand | XS-1 | **Minted 2026-08-07, and it is urgent for a reason that only became visible when XS-1 ran.** The venue keeps ~5000 bars/interval, so the 1m universe archive can only be **accumulated** — today's 3.5 d snapshot exists solely in our parquet and the venue will drop it. Every day this does not run is a permanently lost day of 1m breadth that no future backfill can recover. Cron `fetch_candles --universe` for {1m, 5m, 15m, 1h} daily; the fetcher is already incremental. **Add a retry pass:** the 2026-08-07 sweep produced 2 spurious `empty` results (ORDI 15m, REZ 5m) that both succeeded on immediate retry — the `empty` bucket currently conflates "venue has none" with "one request hiccupped". **Add a requested-vs-received span check:** the same sweep reported `ok=177 failed=0` for a 1m run that returned 4 % of the requested window; `ok` currently means "rows came back". A depth audit caught it, the tool did not — and a cron that reports clean while collecting 4 % is worse than no cron. **Shipped 2026-08-07:** `span_days()` + `retries` / `short` in `backfill_universe`, systemd `nat-candle-refresh.{service,timer}` (oneshot, `OnCalendar=03:17`, **`Persistent=true`** so a window missed while the box was off fires on next boot — for data that expires at the source, "next time" is not a recovery), 1m swept first so a truncated run still gets the perishable interval, wired into `nat service install/uninstall/status/logs`. Guards `tests/test_candle_refresh.py` (12). Live smoke: BTC/ETH/ATOM at 90 d requested now report `short {'BTC': (3.62, 90), ...}` alongside `ok=3 failed=0` — the blind spot is closed. |
| XS-8 | REST `l2Book` universe sampler | DONE | **P1** | S | in-hand | — | **The missing instrument, and `B-5a`'s hard dependency** — candles carry no spread and no depth, and the ingestor's book feed follows `symbols.toml` (BTC/ETH/SOL), i.e. the three *tightest* symbols on the venue, which is the sampling bias §4.11 named as the reason the maker line is unresolved rather than dead. **Shipped 2026-08-07:** `data/fetch_l2.py` (`parse_l2_book` / `sample_universe` / `write_snapshot`) + `tests/test_l2_sampler.py` (15). Per pair: best bid/ask, mid, spread, **half_spread_bps**, touch size/notional/order-count, 5-level notional. Degenerate books (crossed / locked / one-sided / non-positive px) carry a status and **no spread** — on illiquid pairs they are common and a zero spread reads as free money. One symbol's failure never ends the sweep (XS-1 lesson). **Runs as a loop daemon, not a timer** (`nat-l2-sampler.service`, `Restart=always`, 300 s): the product is the intraday *distribution*, and a single book is n=1 — the exact error PROC-20 corrected in LF7's priors. Snapshots are append-only, one parquet per sweep under a UTC day dir. **First live reading: BTC half-spread 0.078 bps** (cross-checks §4.11's 0.083) **vs CASHCAT 3.10 bps — 40×** — and CASHCAT holds more touch notional ($9,980) than SOL ($63). Also supplies `XS-5`'s spread ceiling / depth floor. |
| XS-9 | Beta-neutral rotation construction | DONE | **P1** | S | in-hand | XS-6 | **Built + run 2026-08-07 (FINDINGS §7.8): 4 of 6 pre-registered criteria now pass, vs 0 of 6.** Diagnosis: XS-6's portfolio was ~2.2 effective bets, not 40 (within-basket corr 0.433/0.323) and 0.802-correlated with a *static* low-beta-minus-high-beta position — but that tilt is **uncompensated** (beta→return IC −0.026, t −1.01) while the signal survives neutralisation and *sharpens* (t −5.48 vs −4.08). Rebuilt beta-neutral + score-proportional: Sharpe **1.06 → 2.12**, IS Sharpe **−0.42 → +2.64** (profitable in both halves now), positive share 0.49 → 0.55, max-day 0.78 → 0.30, turnover 0.34 → 0.20 (so cost *fell*), net beta 0.000. **Still fails (b) DSR and (e) OOS/IS 0.447 → nothing promotes.** Power: n ∝ 1/SR², so t=2 now needs ~325 rebalances ≈ **0.89 yr** (was 2.55) — ~8 more months of `XS-7`. **Caveat: designed after seeing XS-6 fail, on the same 83 days** — mechanism is theory not search (and it cut turnover, which fitting usually doesn't), but the magnitude is an upper bound and this is a 13th trial on one window. |
| XS-10 | Surface the XS layer in `nat` + trial ledger + standing eval | DONE | **P1** | S | in-hand | XS-9 | **Shipped 2026-08-07.** (i) **Contract debt closed:** `scripts/cli/xs.py` — `nat xs universe\|capacity\|rank\|persistence\|trajectory\|ledger`, every entry tagged **[PRELIM]** with an explicit "nothing promoted" banner (NAT9 isn't built, so per `contracts/README.md` the tag lives in help text + docstrings). **A second half of the debt was found while wiring it:** the XS processes were `@register`ed but never *imported* in `processes/__init__.py`, so the decorators never fired and `get_process` could not see them — which is *why* every XS run had to bypass the runner. (ii) **Ledger gap closed:** `nat xs rank` now routes through `run_process` → `_fdr_and_ledger`; `data/processes/fdr_ledger.jsonl` exists and holds its first sweep (n_tested=3, git_sha stamped) where it held **zero** despite 13 trials. (iii) **Standing eval:** `xs/trajectory.py` + `tests/test_xs_trajectory.py` (11) — append-only t-stat trajectory with the six criteria **imported, never redefined**, and a non-positive Sharpe reported as *never resolvable* rather than given a schedule. `--record` re-runs the rotation on the current archive and the **candle-refresh timer now fires it daily**, so the wait measures itself. **Its first automated run already moved the result** (FINDINGS §7.8): 7 pairs of admitted-universe churn from 2.4× more L2 data took Sharpe 2.12 → 2.00 and flipped criterion (d), so the tally is **3 of 6**, not 4. The sequence is the product — if §7.8's in-sample design choice was optimistic the Sharpe decays as n grows, and that shows up early instead of in eight months. |
| XS-2 | Bar-level xs feature library | DONE | P1 | M | in-hand | XS-1 | **Shipped 2026-08-07:** `scripts/xs/features.py` + `tests/test_xs_features.py` (15). Estimators `permutation_entropy` (mirrors `ing-features/src/entropy.rs:373` so the name means one thing across layers), `hurst_rs` (rescaled range — deliberately not the PSD-slope estimator in `spannung_spectral.py`, which needs a long stationary tick series), `momentum_strength` (slope × R²: high only when a move is both large **and** clean), `realized_vol`; plus `rolling_self_percentile`, the transform that makes 177 pairs comparable. **The percentile is strictly causal and asserted so** — perturb the future violently, the past must not move by a float (the PROC-15 attack); a percentile that sees its own future is the classic way to manufacture cross-sectional alpha and would flatter every downstream `xs_*` score invisibly. Smoke: 177 pairs × 90 d of 1h candles in 1.8 s, zero NaNs. **Real-data smoke produced a negative worth having (FINDINGS §7.3): permutation entropy does not discriminate across this universe** (IQR 0.0005 at order 3; raising the order substitutes an undersampling bias that ranks by history length). `XS-3` should rank on hurst/momentum/vol, which do spread. |
| XS-3 | `xs_rank_predictability` process | DONE | P1 | M | in-hand | XS-2,PROC-19 | **Shipped + run 2026-08-07. Track C SURVIVES its kill test** (FINDINGS §7.4): on 177 perps × 90 d, 83 **non-overlapping** daily rebalances, two score families clear BH-FDR — `xs_vol` rank-IC **−0.0690** (z −8.37, q 0.007) and `xs_momentum` **−0.0387** (z −4.56, q 0.007); `xs_hurst` fails at z −2.47. **Both survivors are negative**: low-vol pairs outperform, and recent winners underperform — i.e. cross-sectional *mean reversion*, independently reproducing PROC-20's anti-persistence result by a different method. Two alternatives were tested and one killed a result: log returns reproduce every IC to 4 dp (skew/Jensen ruled out), while **overlapping windows turned out to be inflating `xs_hurst` at 7 d — z −3.48 → −0.59 once re-spaced**, the same defect that invalidated `funding_reversion` in §4.6. Null = label permutation **within** each cross-section (never pooled, never across time). Not established: costs (median half-spread 1.37 bps vs IC 0.069 → `XS-6`), capacity (§7.2), one regime. *Defect fixed en route: `informative` was one-sided in z, inherited from the unsigned MI processes, so a z of −8.4 initially reported as non-informative; now two-sided with explicit `polarity`.* |
| XS-4 | `xs_persistence` process | DONE | P1 | S | in-hand | XS-2,PROC-19 | **Shipped + run 2026-08-07 (FINDINGS §7.5): only `vol` has genuine rank persistence.** Rank autocorrelation to lag 30 on the 177-pair panel. **The short lags are an artifact** — a 168-bar lookback means consecutive daily scores share 6/7 of their input, so `momentum`'s ρ(1d)=0.879 is window overlap, not memory. **Lag 7 is the first disjoint-window lag and it separates them completely:** `vol` 0.691 (still 0.509 at 30 d, fitted half-life ~37.7 d — an extrapolation, it never crossed 0.5 in-window) vs `momentum` **−0.003** and `hurst` 0.018. So **`vol` wins on both axes** — larger \|IC\| (0.069 vs 0.039) *and* a ranking that survives weeks — while a daily momentum rotation is churn by construction, the exact failure this row exists to catch. *Note the specified criterion ("half-life > cadence") is necessary but weak: momentum passes at 1.4 d vs 1 d cadence while having no disjoint-window memory; the meaningful quantities are the ratio and ρ at a disjoint lag.* Perf: argsort-Spearman + `stride` (scipy's is ~40× slower on this hot path). |
| XS-5 | `xs_capacity_gate` process | DONE | P1 | S | in-hand | XS-1,XS-8 | **Shipped + run 2026-08-07 (FINDINGS §7.6): breadth and size trade off directly.** `xs/capacity.py` (`load_l2_snapshots`/`aggregate_l2`/`admit`/`tradability_curve`) + `tests/test_xs_capacity.py` (10). **Mints no thresholds** — the guardrail is gates-imported-not-invented and no measured economics exists yet to derive a ceiling from, so it reports the *curve* and `XS-6` picks the operating point; rejections list **every** failed floor, so relaxing one never looks sufficient. **The instrument choice was the finding:** an L1 touch floor of $10k admits just **3 pairs** (BTC/ETH/SOL — exactly what the ingestor already covers) and would have read as "Class-3 breadth is impossible", but L1 is resting size at an instant; against ADV at 1 % participation, **117 pairs support $1k/pair at ≤2 bps**, 52 support $10k, 10 support $100k. So IR≈IC·√breadth holds at small notional and nowhere else. **The surviving `vol` score does not solve capacity**: corr(vol, half-spread) +0.397 (helpful — low-vol pairs are ~4× tighter) but corr(vol, touch) −0.092, and the low-vol cohort runs from ETH ($365k touch) to ICP ($13). Limits: spread from ~3 h of one day, sampler still accruing. |
| XS-6 | Rotation OOS study | DONE | P1 | M | in-hand | XS-3,4,5 | **Run 2026-08-07, pre-registered (criteria committed `f3eea78` BEFORE the run): 0 of 6 configurations survive** (FINDINGS §7.7). Nothing promotes to DISCOVERED. **It fails on durability, not cost** — turnover 0.17–0.49 of a 2.0 max and costs 1–2.7 % against 8.5 % gross, exactly as §7.5's ≥30-day half-life predicted, so the mechanism worked; every §4.6 casualty died paying full cost against a seconds-scale signal and this one genuinely does not. **The trap it caught:** long-short shows OOS Sharpe **5.04**, which read alone looks outstanding — but **IS Sharpe is negative** (−0.27) and the OOS/IS ratio is −19, i.e. the strategy lost over the first 60 % of its own backtest and made everything in the last 40 %. Criterion (e) catches it only by testing the ratio rather than the OOS level. Also fails positive-share 0.49 (coin flip) and single-day concentration up to **104 % of total P&L** — `surprise_signal`'s §4.6 failure on an unrelated strategy. Long-only is market beta (−19.5 % gross), so any future rotation must be built neutral. *Driver defect fixed mid-study: `.get("taker_bps", 4.5)` fallbacks silently supplied hardcoded fees because the SSOT is nested — the §4.6 guardrail violation, reproduced by me in a new file; now the accessor with no fallback.* |
| A-1 | `agreement_gate_eval` process + standing eval | DONE | P1 | S | in-hand | — | **Shipped + run 2026-08-08 — the gate is HARMFUL, not absent (FINDINGS §5.1).** `processes/agreement_gate_eval.py` + 15 tests, registered as a standing eval. Null permutes **the gate** holding subset **size** fixed — the selection trap that let §5's pilot pass (max(agree,disagree) beats pooled by construction, so the naive check passes on noise). Measured: on ETH/SOL the **disagreement** subset carries the signal; lift −0.067/−0.057 at z −2.91/−2.71, `frac_days_informative`=0.00 everywhere. §5 is now fully retired. **Caveat added 2026-08-09 (BUG-6 / FINDINGS §7.11): the gate input was dead for most of the stated window.** `regime_divergence_1h` is all-NaN from 07-26, so of the 25 days only **07-15→07-18** carry a live slow feature — every row after 07-26 drops out of `valid` silently. That matches the reported `n_days` 0–3 vs `min_days` 3 and the `insufficient_days`/`non_durable` verdicts, but it means the result rests on **≤4 usable day-folds**. An independent run on the alive window (06-21→07-25, 243 bars) reproduces the **direction and not the strength**: 0/9 cells informative, 7/9 lifts negative, largest cells favouring disagreement (BTC 1d agree −0.035 vs disagree +0.259). **Better stated as "no support, underpowered" than "refuted"** — the process itself refuses a durability verdict. Re-run once BUG-6 is fixed. |
| A-2 | **Combiner revalidation** | DONE | **P0** | M | in-hand | — | **Run 2026-08-08 — REFUTED (FINDINGS §5.1).** `alpha/walkforward_ic.py` + 15 tests; weights refitted per fold on rows strictly before it, scored on **non-overlapping** observations. Walk-forward IC **+0.062 / +0.099 / −0.024** vs §5's .178/.248/.359, and the composite **never beats its own best single feature** (`trend_ema_short` ~0.20). Mechanism is in the file dates: `weights_BTC.json` `training_date 2026-06-11` vs an OOS window of 06-08→10 — fitted **after** the period it was scored on, which produces monotone fold ICs by construction. Also found: at §5's ~5 h horizon 25 days yield **28 non-overlapping obs**, so that horizon is not currently testable. |
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
