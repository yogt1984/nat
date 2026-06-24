# Priorities — 2026-06-14

Derived from `STATE_14_6_2026.md` + the `docs/in_progress/` corpus (`tasks_assigned_12_6_26/plan.md`
rev 6, `Q/*`, `P/*`, `test_plan.md`, `data_inventory.md`, `situation_analysis.md`).
**Ordering rule** (from plan.md's hard constraints + the binding finding): *data continuity is the
binding constraint — every clean day appreciates all existing research; resolve execution feasibility
before scaling; front-load all non-data-gated work into the accumulation window.* Gates are **imported,
not invented** (G4 = walk-forward + deflated Sharpe; G8 = 14-day paper, 5 criteria; kill thresholds =
ROADMAP Step 9). Task IDs follow `plan.md`'s reconciled T-numbering (with Q/P cross-refs).

> **Status (updated 2026-06-15).** Shipped since this doc was written: **T2** provenance
> (`scripts/provenance.py`; costs already at `scripts/utils/costs.py`), **T3–T5** signal-lifecycle
> spine (`signal_lifecycle.py` + `nat lifecycle` + agent hook, seeded), **T6** CLI help
> (`nat help --grep` + group-level help), **T7** viz foundation + **NAT4/NAT5**
> (`nat viz features|algorithm`). Also `test_algorithm_smoke.py` + `.claude` scaffolding-drift fixes.
> Remaining P0: T0/T0b (ops/deploy), Q1.3 (partial), kill-switch. P1+ stays gated on the Jun-17 streak.

## Implementation priorities

### P0 — Now, non-data-gated (Jun 11–17 window; zero su-35 contact)
1. **T0 / Q1.1 — resolve the 82 dead features** (40 unlocked by committed wiring). Enable
   `[position_tracker]`, deploy to the **T0b cloud box, not su-35**; 24h coverage check → 48h viability
   verdict. *Highest leverage — unblocks the FDR screen's full 191-feature coverage and LF3.*
2. **T0b / M2 — continuous cloud ingestion (Hetzner)** with Telegram <5min gap alerting. Redundant second
   ingestor; doubles as the wired-binary deployment vehicle while su-35 is streak-frozen.
3. **[✓ DONE] T2 + Q1.4 — costs + provenance** (`scripts/utils/costs.py` `load_costs()`, `scripts/provenance.py` git SHA
   + data fingerprint). Single cost source; reproducibility. **Shared with P1.5** — one job, both tracks.
4. **Q1.3 — SQLite research store** (~25h): atomic writes, schema versioning, contract tests.
5. **[✓ DONE] T3–T5 — signal lifecycle spine** (DB tables + `signal_lifecycle.py` + `nat lifecycle` CLI +
   agent integration; seeded 4 winners VALIDATED / 2 DISCOVERED). The spine the promotion daemon will drive.
6. **T16 / Q3.1 — kill-switch daemon** (can start early; *must* ship before any bridge). 4 thresholds,
   Telegram <60s, `nat risk status/resume`.
7. **[✓ DONE] CLI/viz foundations** — T6 (NAT1/2 help) + T7 (NAT3 viz lib) + NAT4/5
   (`nat viz features|algorithm`). The cheap integration substrate (METHODOLOGY: enabling-infra-first).

### P1 — Jun-17 gated (execute the moment the 7-day streak + Q1.1 land)
8. **T11 / Q2.3 — full alpha screen with BH-FDR (Gate G1)** across 191 feats × 3 symbols, plus
   `nat process run ic_horizon|mi_ksg` for cross-method evidence. **Critical path.**
9. **T11 / Q2.1 — revalidate `hierarchical_combiner`** on 7 clean days (its 2-day monotone-IC folds are
   unverified) + ablation. **This is the execution-feasibility test.**
10. **T11b — process transforms** (`pca_combo --score-with ic_horizon`, triple_barrier) — only
    decision-grade on ≥7 days.
11. **Q2.5 — spannung Kalman** on the ultra-low band (zero/low-fee viability).

### P2 — Validation → paper (contingent on gate passage)
12. **Q2.4/Q2.5 — combine + cost-aware size + walk-forward (Gate G4).**
13. **T14 — promotion daemon** (data-sufficiency guard; DISCOVERED→VALIDATED→PAPER automatic).
14. **T15 / T17 — approval viz (NAT6/7) + signal bridge daemon** (dry-run, risk-parity sizing).
15. **Q3.2 — 14-day paper window (Gate G8)** — calendar-bound; first APPROVAL_PENDING ~Aug.

### P3 — Deferred / contingent (do NOT pull forward)
Q4.1/T21 live deployment (Sep–Oct, human-gated, post-G8) · Q3.3 regime/multi-freq (skip if no OOS gain)
· T20 CLI polish (NAT8/9/10) · 3D-viz greenfield · HF3 Hawkes (10h) · LF3 cascade (gated on T0 verdict).

## Testing priorities
The METHODOLOGY pyramid, applied per item:
1. **Planted-signal first** (mandatory before any real-data use) — `nat test process` (48 contracts),
   `pytest scripts/tests/test_algorithm_smoke.py -k <name>`. *Three estimator bugs were caught only this way.*
2. **Real-parquet smoke before commit** — spot-check values on the latest day; schema contract intact
   (`names_all()` == 236, NaN-padding when sources absent).
3. **Gate tests** — G1 report (T11); G4 (OOS Sharpe>0.5, OOS/IS>0.7, deflated p<0.05, maxDD<5%, ≥30 trades,
   PF>1.2); G8 (5 criteria over 14 days). **No new thresholds.**
4. **CI** — fmt/clippy/cargo test + pytest + vitest + criterion 10% gate, green before merge.
5. **Manual `test_plan.md` pass** — Section A (terminal connectivity: 260 commands dispatch) + Section B
   (visualization correctness) as a **pre-paper release gate**; record in the sign-off log. Note the known
   `nat test agent` doc/impl mismatch.
6. **Live wiring validation** — T0 per-column coverage + sanity ranges on the cloud box; T0b 48h zero-gap
   + forced-disconnect alert fires.

## Explicit non-goals right now
No su-35 contact mid-streak · no live capital before G8 + a healthy kill-switch · no invented thresholds ·
no 3D-viz greenfield until conditional-IC is positive · defer Hawkes / cascade / aegis_maker.
