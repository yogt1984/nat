# PLAN — Next 30 Days (2026-06-14 → 2026-07-14)

**Combines:** `docs/in_progress/tasks_assigned_12_6_26/plan.md` (rev 6, T0–T21 sequencer) · `Q/*` (gates)
· `P/*` (academic) · `docs/in_progress/test_plan.md` (manual gate).
**Read with:** `STATE_14_6_2026.md`, `PRIORITIES.md`, `MASTER_PLAN.md`.

**Goal for the window:** convert the data-accumulation wait into a validated, autonomously-promotable
pipeline — land the **G1 alpha-screen verdict + `hierarchical_combiner` revalidation** on clean data, with
the lifecycle / promotion / kill-switch machinery live behind it.

**Hard constraints (from plan.md):** zero su-35 contact until the streak completes (Jun 17) · gates
imported not invented · all costs via `load_costs()` · no live capital.

---

## Week 1 — Jun 14–17 · Stage A (zero su-35 risk; cloud + pure Python)
**Theme: earn the substrate while the streak finishes.**

> **Progress (2026-06-15):** T2, T3–T5, T6, T7 (+ NAT4/NAT5) all merged & tested. Remaining in Week 1:
> T0/T0b (ops — deploy the wired binary to the cloud box), Q1.3 (SQLite store, partial). Streak completes Jun 17.
- **T0 / Q1.1 — dead-features resolution.** Enable `[position_tracker]`; deploy the wired binary to the
  **T0b cloud box**; 24h coverage check; **48h viability verdict due ~Jun 15** (viable / noisy / unavailable).
  *Test:* whale/liq coverage <100% NaN within 1h; `names_all()` == 236; verdict recorded in `01_concentration…`.
- **T0b / M2 — cloud ingestion** (Hetzner AX52): Tier-1 docker stack + wired binary + Telegram <5min gap
  alert. *Test:* 48h zero gaps >60s; alert fires on forced disconnect; feature parity vs su-35.
- **[✓ DONE] T2 + Q1.4 + P1.5 — costs + provenance** (`scripts/utils/costs.py`, `scripts/provenance.py`).
  *Test:* `load_costs("hyperliquid")` reproduces current `nat oos30` exactly; fingerprint deterministic.
- **[✓ DONE] T3–T5 — lifecycle spine** (DB tables, `signal_lifecycle.py`, `nat lifecycle`, agent integration;
  seeded 4 VALIDATED / 2 DISCOVERED). *Test:* scripted transition walk; illegal transition raises;
  register_signal → DISCOVERED row.
- **[✓ DONE] T6 — CLI quick wins** (NAT1/NAT2). **[✓ DONE] T7 — viz foundation** (NAT3 + NAT4/NAT5: `nat viz features|algorithm`).
- **Start Q1.3** (SQLite store) if capacity. **P-track:** P1.5 done here; begin P1 preprint draft (long lead).
- **Streak watch:** daily data check only; **do not touch su-35.** Re-check the Jun 10–11 thin-day risk
  against the Jun-17 target.

**Week-1 exit:** T0 verdict recorded · cloud ingestor unattended · costs/provenance live · `nat lifecycle`
functional · streak intact to Jun 17.

## Week 2 — Jun 17–24 · Stage B (streak complete → first real validation)
**Theme: does the edge survive clean data?**
- **su-35 cutover** to the wired binary (post-streak); T0b cutover decision (cloud vs su-35 primary).
- **T11 / Q2.3 — alpha screen + BH-FDR (Gate G1)** across 191 feats × 3 symbols + `nat process run
  ic_horizon|mi_ksg`. **Critical path.** *Test:* G1 report generated.
- **T11 / Q2.1 — `hierarchical_combiner` revalidation** on 7 clean days + ablation (L2/L3 vs L1).
  *Test:* ≥4 folds @500+ bars; IC not monotonically rising; combiner → VALIDATED only if G4 passes,
  else REJECTED with reason recorded.
- **T11b — process transforms** (`pca_combo --score-with ic_horizon`, triple_barrier) on the clean window.
- **Q2.5 — spannung Kalman** ultra-low band.
- **T8 / T9 — feature gaps F1–F5 + algo candidates** (HF4 VPIN gate, LF1 funding-settlement, HF1 microprice)
  interleaved.
- **T12 — dockerize agents**; **T13 — daily agent + candle fetcher.**

**Week-2 exit (the pivotal gate):** **G1 verdict + combiner-revalidation verdict** — the first read on
whether the edge is real on clean data. *If G1 fails, stop and re-scope — do not loosen gates.*

## Week 3 — Jun 24–Jul 1 · Stage C (automation + risk layer)
**Theme: make promotion autonomous and safe.**
- **T16 / Q3.1 — kill-switch daemon** (ships before any bridge). *Test:* each of 4 thresholds fires;
  Telegram <60s; resume rules enforced.
- **T14 — promotion daemon** (data-sufficiency guard; DISCOVERED→VALIDATED via G4 → PAPER).
  *Test:* seeded signal auto-starts paper; insufficient-data case refuses OOS with logged reason.
- **T15 — approval viz** (NAT6/NAT7; must precede the first APPROVAL_PENDING).
- **T17 — signal bridge daemon** (dry-run; checks `halt_state.json` every cycle; risk-parity sizing).
- **Q2.4/Q2.5 — combine + size + walk-forward (Gate G4)** if G1 passed.

**Week-3 exit:** lifecycle runs unattended discover→paper behind a healthy kill-switch; ≥1 signal VALIDATED
or in paper.

## Week 4 — Jul 1–14 · Stage C→D (observability + paper window)
**Theme: prove it under observation.**
- **T18 — monitoring + E2E** (Prometheus on Python services; Grafana lifecycle/paper/P&L dashboards; full
  `docker compose up`). *Test:* E2E ingestion → discovery → OOS → paper → approve → dry-run trade.
- **T19 — 14-day paper window (Gate G8)** starts as soon as T14 promotes (paper clock starts here; first
  APPROVAL_PENDING expected ~Aug).
- **Manual `test_plan.md` pass** — Section A (260 commands dispatch) + Section B (viz correctness) as the
  pre-paper release gate; record in the sign-off log.
- **Begin T20** CLI polish (NAT8/NAT9; NAT10 only after NAT1–9 stabilize). **P-track:** continue preprint.

## 30-day exit criteria
1. T0 dead-feature verdict + conditional-IC direction known.
2. **G1 alpha-screen + `hierarchical_combiner` revalidation** verdicts recorded.
3. Lifecycle + promotion daemon + kill-switch + bridge (dry-run) live and dockerized.
4. ≥1 signal in (or queued for) the 14-day paper window.
5. `test_plan.md` Section A+B signed off; CI green.
6. Costs / provenance / SQLite-store infra landed; preprint draft underway.

## Parallel P-track (writing — fills analyst time)
P1.5 provenance (Week 1, shared with Q1.4) · P1 preprint drafting ongoing (~40h, camera-ready target Aug).
P2–P4 deferred (Sep+).

## What slips past 30 days (by design)
14-day paper *completion* (→ Aug) · live deployment T21 (Sep–Oct) · Q3.3 regime/multi-freq (skip if no OOS
gain) · 3D-viz greenfield · HF3 Hawkes / LF3 cascade.
