# Three-Class Research Program — Proposal

**Status:** APPROVED & DRAINED (2026-08-06). The 16 rows now live in
[`TASKS.md`](TASKS.md) § "R1 — Three-class research program" (IDs `XS-1..6`, `A-1..3`, `B-1..4`,
`X-1..3`), and `PLAN.md` §0 has been refreshed to the R1 order — **`TASKS.md` is the execution
record from here; this file remains the program's rationale and kill criteria.** `X-1` is DONE
(§4.10). Sequencing amended by evidence since approval: `XS-1` leads outright because
`B-5` (wider-spread maker viability, the one hypothesis §4.11 left alive) also depends on the
candle universe.
*(Original header: PROPOSAL, 2026-07-31. On approval: rows drain into `TASKS.md`, `PLAN.md` §0
refreshes, and studies execute in the order below.)* **Initially research only** — paper and
live phases are specified for completeness but explicitly out of scope until their gates.
**Spec basis:** `specs/maker_system.md` v2. **Empirical basis:** `research/FINDINGS.md`.
**Gates are imported, never invented** (Q5, G8, P6/§4.9 criteria, PROC null/FDR).

---

## 0. Principle

Every claim climbs the maturity ladder: **research → validated → paper → live**, one gate at
a time, and each study is pre-registered (criteria in the test/driver file before results).
Failed studies are recorded in FINDINGS with the same care as successes — the Q4 lesson is
that unrecorded negatives return as false positives. No live capital before G8 + healthy
kill-switch; approval (`nat lifecycle approve`) is the sole human gate.

## 1. Program structure — four tracks

| Track | Object | Data dependency |
|---|---|---|
| **C** | Class 3 — cross-sectional rotation (scanner + selection) | **none** (REST candles) |
| **A** | Class 1 — directional bias makers (signal + gating layers) | in-hand parquet |
| **B** | Class 2 — oscillation harvesters (admission + geometry) | in-hand parquet |
| **X** | Shared execution research (fee tiers, fill data, F-task) | mixed |

Track C leads: it is the only track with zero data debt, and its output (which pairs matter)
sharpens every later study in A/B.

## 2. Phase R1 — Research studies (enumerated; drain into TASKS.md as-is)

### Track C — Class 3 (all in-hand via REST candles)

| ID | Study | Size | Deliverable / criterion |
|---|---|---|---|
| XS-1 | Universe candle backfill: extend `fetch_candles.py` driver to the full perp universe (enumerate via `meta`), 1 m + 1 h, incremental daily cron | S | `data/candles/` ≥ 90 d × universe; coverage report |
| XS-2 | Bar-level xs feature library: permutation entropy, momentum slope×R², Hurst, vol percentile — each vs own history (rolling percentile/z); planted tests | M | `scripts/xs/features.py`; planted red→green |
| XS-3 | `xs_rank_predictability` process: rank-IC of scores vs relative forward returns per rebalance interval; permutation null (shuffled pair labels) + FDR | M | registered process; verdict: any score family significant? |
| XS-4 | `xs_persistence` process: rank autocorrelation half-life per score | S | half-life > candidate cadence, else cadence lengthens or track stops |
| XS-5 | `xs_capacity_gate`: spread/depth/volume floors from candle+meta data, SSOT-priced | S | admitted-universe list, refreshed daily |
| XS-6 | Rotation OOS study: top-k weighted paper rotation on candles, walk-forward, SSOT costs, pre-registered P6-style criteria | M | FINDINGS §; promotes to lifecycle DISCOVERED iff survives |

### Track A — Class 1

| ID | Study | Size | Deliverable / criterion |
|---|---|---|---|
| A-1 | `agreement_gate_eval` process + standing eval: conditional IC of fast signal given slow-bias agreement vs disagreement, null-calibrated | S | replicates §5 pilot as a monitored fact, or kills it |
| A-2 | **Combiner revalidation** (the highest-value single experiment): multi-day walk-forward via `nat oos --window` + DSR, pre-registered; the 2-day §5 numbers (IC .18/.25/.36) stand or die | M | conditional-IC vs the Q5 bar at the priced fee tier |
| A-3 | Conditional-IC surface refresh: PROC-6/7 sweep with the §2 feature contract as conditioning, staked fee tier priced in | S | updated predictability surface artifact |

### Track B — Class 2

| ID | Study | Size | Deliverable / criterion |
|---|---|---|---|
| B-1 | `oscillation_admission_eval` process: does admission (hurst/band-power/OU-τ) predict forward oscillation? Null-calibrated persistence | S | admission validity verdict |
| B-2 | `band_geometry_scan` process: (k × dominant period × regime) capture-vs-adverse surface, FDR'd; geometry is read off, never swept | M | the LF7 k-prior confirmed or replaced by surface argmax |
| B-3 | LF7 signal layer: band/channel algorithm per `contracts/algorithm.md`, planted tests, k from B-2 | M | registered algorithm, signal-level IC record |
| B-4 | Band study OOS: multi-day pre-registered signal-level verdict (economics deferred to X-3) | M | FINDINGS §; DISCOVERED iff survives |

### Track X — Shared execution research

| ID | Study | Size | Deliverable / criterion |
|---|---|---|---|
| X-1 | `[hyperliquid_staked]` tier in `costs.toml` (+ guard tests) and reprice §4.7/§4.9 grids | S | does any maker cell flip at the discounted tier? |
| X-2 | **F-task plan** (schema change → plan first, per guardrail): L1 queue sizes + per-tick side volume columns; ripple analysis (Parquet schema, readers, `names_all()`) | S (plan only) | reviewed plan; implementation is its own approval |
| X-3 | Fill-economics reruns (§4.7/§4.9/HF5b) on F-task data or T0b shadow quotes | M | **data-gated** — the maker go/no-go |

## 3. Shared OOS protocol (applies to every study above)

- Walk-forward folds with embargo; deflated Sharpe reported wherever Sharpe is claimed
  (`nat oos --window` machinery; DSR reporting per the G4 convention).
- **Pre-registration**: acceptance criteria in the driver/test file before results — the
  §4.9 set for capital-relevant claims: per-fill/per-period EV > 0, positive-day share
  ≥ 0.55, max single-day ≤ 30 %, proxy-sensitivity sign stability.
- PROC gates for signal claims: null-calibration z ≥ 3, BH-FDR q ≤ .05; every sweep lands in
  the program FDR ledger (`data/processes/fdr_ledger.jsonl`) — multiple testing is accounted
  **across** the program, not per study.
- No post-hoc symbol, parameter, or window selection: symbol splits declared up front;
  geometry from surfaces; failures recorded in FINDINGS.
- Every study cites its fee tier; costs only via `load_costs()`.

## 4. Phase R2 — Validation (promotion gates; not new numbers)

A finding promotes DISCOVERED → VALIDATED in the lifecycle only via imported gates:
Class 1/2 capital claims: **Q5 conditional-IC > 0.15** at the priced tier + §4.9 criteria on
multi-day OOS. Class 3: xs rank-IC significant after FDR **and** rank half-life > rebalance
cadence + XS-6 criteria. Everything provenance-stamped (`git_sha`) in the lifecycle DB.

## 5. Phase P — Paper (out of scope now; prerequisites listed)

VALIDATED findings only. Prerequisites: healthy kill-switch, T0b box for shadow quoting
(maker fills), G8 scorecard wiring. Maker paper trading without real fill data is not
evidence — shadow quotes on T0b are the honest instrument.

## 6. Phase L — Live (out of scope; unchanged gates)

G8 + kill-switch + `nat lifecycle approve` (human). Nothing in this proposal weakens this.

## 7. Sequencing

```
now →  X-1 ─┐
       XS-1 → XS-2 → XS-3/XS-4/XS-5 → XS-6      (Track C: independent, leads)
       A-1 → A-2 → A-3                           (Track A: in-hand parquet)
       B-1 → B-2 → B-3 → B-4                     (Track B: after A-1 frees attention)
       X-2 (plan) ────────────→ X-3 (data-gated)
```
Parallelism: C and A can run concurrently; B follows A's first results; X-1 first (one
afternoon, reprices everything). Q0/Q1 (streak, T0b) remain the user's side and gate only
Phase P and X-3 — **nothing in R1 waits on them.**

## 8. TASKS.md / PLAN.md drain (on approval)

- Add the 16 rows above to `TASKS.md` under a "Three-class research program" block
  (IDs XS-1..6, A-1..3, B-1..4, X-1..3), statuses TODO, sources → this proposal.
- Refresh `PLAN.md` §0 Current Focus: R1 order as above; note Q4 aftermath closed.
- `specs/maker_system.md` stays the design authority; this proposal is the program.

## 9. Kill criteria for the program itself

- Track C stops if XS-3 finds no score family significant after FDR on ≥ 90 d, or XS-4
  half-life < any practical cadence.
- Track A stops (for capital purposes) if A-2 kills the combiner numbers — reverts to
  signal research only.
- Track B stops if B-1 shows admission has no forward validity.
- Track X-3 verdict is final for the maker economics question on current venue mechanics.
