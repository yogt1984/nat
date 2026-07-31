# NAT — Q / D / P Plan

**Single source of truth for the three branches: Q (Quant), D (Development), P (PhD).**
Supersedes the Jun-14 planning layer (`PLAN`/`PRIORITIES`/`MASTER_PLAN`/`STATE_14_6_2026`) and the
Jun-22 `QDP_ROADMAP.md` — all harvested into this file and moved to [`archive/`](archive/).
Durable companions (not merged, not archived): `OBJECTIVE.md` (mission), `METHODOLOGY.md` (method),
`GLOSSARY.md`, `contracts/`, `commands.md`.

**Itemized backlog:** every actionable task lives in [`TASKS.md`](TASKS.md) (the single backlog) —
this plan holds the strategic gate-level view; `TASKS.md` holds the rows. Process/IT-layer detail in
[`specs/process_layer.md`](specs/process_layer.md).

*Last consolidated: 2026-06-24. Backlog consolidated into `TASKS.md`: 2026-07-22.*

---

## 0. Current Focus  *(pinned — update this block as state changes)*

**Binding constraint:** data continuity. Every clean day appreciates all existing research; the
whole Q-branch is gated on it.

**Master gate — a clean 7-day data streak.** The Jun-17 su-35 target was **missed** (Jun-19 check:
window only 7–16 h/day, no complete day), so re-establishing a clean streak — most likely on the
**T0b cloud box** — is now the gating objective, and the dedicated ingest box is the critical path.
Verify current status with `nat gap status` / `/streak` (local read; does **not** contact su-35).
A clean streak lifts the su-35 freeze and unblocks Q2/Q3. **Hard rule: zero su-35 contact until a
clean streak completes.**

**Do-now sequence:**
1. **Q0** — verify the streak outcome.
2. **T0b cloud deploy (P0)** — provision the Hetzner box and deploy the redundant ingestor
   (`nat deploy cloud <ip> --dry-run` first). Removes single-point su-35 risk; the vehicle is ready
   (root `HETZNER_DEPLOYMENT_PLAN.md`).
3. **Q4 — alpha-skeptic gate on existing data** — ✅ **DONE 2026-07-30, verdict 5/5 KILL**
   (`FINDINGS.md` §4.6): all five "winners" were false discoveries (wrong-venue cost tier +
   sweep harness never ran the algorithms' own logic); all REJECTED in the lifecycle. The
   90-day spend it was guarding against is cancelled for these signals.
4. **Conditional-IC > 0.15 (D1, ~Aug)** — the go/no-go for the whole trading business: IC≈0.45
   collapses to ~0.03 under realistic fills.

**Actionable now (no data needed):** T0b provisioning; the D-branch (D1/D2/D3); the entire P-branch
(P1→P5); the maker line (GAP-04/HF1/A4 — the surviving execution path per Q4); the open bugs below.

**Blocked on data/streak:** Q3 revalidation; the G1 alpha screen; `hierarchical_combiner`
revalidation; combine → paper → live.

**Open bugs (P1):**
- Retrain/revalidate the 3 ML algos (`mean_reversion_detector`, `meta_labeling`,
  `regime_conditioned_lgbm`) against the current schema (artifacts date to 2026-06-08).
- Fix `nat agent status` → `ModuleNotFoundError: logging_config` (blocks running agents, incl. on
  the cloud box).
- Fix + enable the GMM 5D regime classifier: correct `train_regime_gmm.py` column names, train,
  enable. **Do NOT merge branch `936f7cb`** (it drops whale flow).

**Dated milestones:** Jun-17 streak gate · D1 conditional-IC verdict + first G8 window ~Aug ·
preprint camera-ready ~Aug, SSRN/arXiv ~Sep · D2 prof-interest checkpoint ~Nov · EPFL EDFI
deadlines **Jan 15 / Mar 31 2027** · live capital Sep–Oct (only if D1 positive).

---

## 1. Guardrails  *(non-negotiable — imported, not invented)*

- **Gates imported, never invented:** G4 = walk-forward + deflated Sharpe; G8 = 14-day paper, 5
  criteria; kill thresholds = ROADMAP Step 9.
- **All costs via `load_costs()`** (`config/costs.toml`). Never hardcode a fee or slippage.
- **No live capital before G8 + a healthy kill-switch.**
- **Plan before any feature-vector / schema change** (it ripples to Parquet and every reader).
- **Planted (synthetic) test before any real-data use.**
- **su-35: zero contact until the clean-data streak completes.**

---

## 2. Q — Quant  *(prove the edge is real and capturable)*

Aim: re-test the promising algorithms over a longer window with a dedicated tool and a dedicated
ingest box, then adversarially try to kill them. No live capital until paper passes G8.

- **Q0 — Verify the streak.** `nat gap status` / `/streak`. ⏳ do first; gating.
- **Q1 — T0b Hetzner ingest box.** 24/7 ingestor + Telegram <5 min gap alert (per root
  `HETZNER_DEPLOYMENT_PLAN.md`). Removes single-point su-35 risk.
- **Q2 — Longitudinal tool.** Generalize `nat oos30` → `nat oos --window <N>d` (walk-forward folds
  + deflated Sharpe, `--json`). Depends on Q0.
- **Q3 — Extended revalidation** of the 5 winners — **MOOT as scoped** (Q4 killed all five,
  2026-07-30; nothing to revalidate). Q3's slot passes to whatever the maker line (GAP-04/HF1/A4)
  + PROC discovery layer promote next.
- **Q4 — Adversarial kill gate.** ✅ **DONE 2026-07-30: 5/5 KILL** (`FINDINGS.md` §4.6, lifecycle
  REJECTED ×4 + surprise_signal never registered). *"Q2/Q3 only matter if Q4 survives"* — it did
  not; the ~90-day revalidation spend is cancelled for these signals.

**Sequencing:** Q0 → (Q1 ∥ Q2) → Q3 → Q4.
**Tension to resolve:** `archive/tasks_22_6_26__2.md` argues Q4 should run *before* the Q1 data
investment — don't accumulate ~90 days for edges that die under refutation on data already in hand.
Recommended: run a first Q4 pass on existing data now (it's in the do-now sequence above).

---

## 3. D — Development  *(harden/ship `nat`, converge to a cloud lab)*

Most of the Jun-12 CLI plan already shipped. Remaining:

- **D1 — Viz set + maturity tags.** `nat viz portfolio/paper/spectral/regime`; `[PROVEN] / [PRELIM]
  / [SPEC] / [LIVE]` tags. ~20h.
- **D2 — Modularize the `nat` monolith** (5,113 lines) → `scripts/cli/*.py` with a `register()`
  protocol. ~12h. Prerequisite for packaging.
- **D3 — Ship `nat` apt-installable.** Phased: (1) **relocatable paths** (XDG, `NAT_HOME`) — the
  real blocker, pairs with D2; (2) interim `pipx`/wheel; (3) native `.deb` + self-hosted apt repo
  (see `packaging/README.md`).
- **D4 — Continuous-discovery → cloud research lab.** Harden `discovery_orchestrator` + the 4
  agents; surface via the `api` crate + Next.js (per `cloud_deployment/`). Partially built.

**Gate:** D3 is gated on its step-1 relocatable paths ("nothing installs cleanly until this is done").

---

## 4. P — PhD  *(publish, then outreach)*

Goal: PhD at ETH Zürich / EPFL in spectral microstructure / quant finance. **Decision:** the
convolver preprint is already a complete contribution — **publish it now**; develop Spannung as
paper #2 (don't gate outreach behind more writing).

- **P1 — Polish** → camera-ready PDF (~1d).
- **P2 — SSRN** upload (1–3 business days).
- **P3 — arXiv `q-fin.TR`** endorsement.
- **P4 — Outreach:** gather Tier-1 prof emails + send (~5h). *Prof emails are not stored — gathering
  them is part of P4.*
- **P5 — Track responses,** stagger Tier-2; milestone = 2+ interested → formal applications.

**Deadlines:** EPFL EDFI **Jan 15 2027** (R1) / **Mar 31 2027** (R2), Sep-2027 entry; ETH rolling.

**Artifact locations:**
- Preprints (5, all **Yigit Onat** — `yionat@gmail.com`, each `.tex` + compiled `.pdf` in
  `research/`): `convolver_preprint`, `microstructure_alpha_preprint`,
  `liquidity_heatmap_preprint` (liquidation-density heatmap → cascade price-movement model),
  `prism_preprint` (Prism's Perception Pressure / Resonance narrative metrics),
  `prism_signal_preprint` (Prism's `P(K,t)` as an exogenous, orthogonal alpha signal integrated into
  the microstructure book — timescale separation, effective breadth `N_eff`, partial adverse-selection
  evasion; methods-only, ties together the convolver/microstructure/process papers).
- Findings / build appendix: `synthesis/{microstructure_alpha_findings,build_implementation_spec}.{tex,pdf}`.
- Guide + prof list + email template: `phd_related/phd_application_guide.tex`; one-pager
  `phd_related/phd_application_summary.{tex,pdf}`.

---

## 5. Synergies & decision points

The three branches compete for **time allocation**, not resources — they share infrastructure,
data, and methodology. Decision gates:
- **D1 (~Aug):** conditional-IC verdict — "is there a trading business?" (gates Q-branch live work).
- **D2 (~Nov):** prof-interest checkpoint — "do professors want this?" (gates P-branch effort).
- Live capital only after D1 positive **and** G8 **and** a healthy kill-switch.

---

## 6. Companions & references  *(durable — not part of this plan, do not archive)*

- **Mission / method:** `OBJECTIVE.md`, `METHODOLOGY.md`, `GLOSSARY.md`.
- **Unit contracts:** `contracts/` — feature / algorithm / **process** / viz.
- **Process definitions:** `contracts/process.md` (contract) + `specs/process_layer.md` (the
  PROC-1..18 spec) + the `scripts/processes/` framework (7 shipped processes). (Original concept
  docs archived under `archive/in_progress/tasks_assigned_12_6_26/`.)
- **Docs improvement plan:** `DOCS_IMPROVEMENT_PLAN_PROPOSAL_V1.md` (2026-07-31, PROPOSAL) —
  audit of PLAN/TASKS/specs after the Jul-30/31 sprint: ~25 stale rows, one ID collision
  (COST-4), D1 naming clash, funding-carry gap, missing risk/capital + monitoring sections.
  Remedies P1–P4 (ID-registry + same-branch-status conventions, TASKS reconciliation, PLAN §0
  rewrite, spec v3). P1 blocked on the in-flight TASKS.md edit landing.
- **Three-class research program:** `THREE_CLASS_RESEARCH_PROPOSAL.md` (2026-07-31, PROPOSAL) —
  the research-first program over the three classes: 16 enumerated studies in four tracks
  (XS-1..6 rotation scanner [data-independent, leads], A-1..3 Class-1 incl. combiner
  revalidation, B-1..4 Class-2/LF7, X-1..3 execution research incl. staked fee tier + F-task
  plan), shared OOS/pre-registration protocol, promotion gates, per-track kill criteria.
  Paper/live phases specified but out of scope until their gates. Drains into TASKS.md rows
  on approval.
- **Maker system:** `specs/maker_system.md` (v2, 2026-07-31) — post-Q4 execution doctrine
  (maker-only, taker = emergency transition), the orthogonal combiner feature contract, **three
  algorithm classes** (directional bias makers / oscillation harvesters / cross-sectional
  rotation over the full perp universe, two-tier: REST-candle scanner + deep tick stack on
  selected pairs), the three-level regime router, per-class process definitions (incl. the new
  cross-sectional process kind), and pre-registered acceptance criteria. Build order inside;
  the Class-3 Tier-W scanner is fully data-independent and buildable now; fill-economics
  claims blocked on the F-task (L1 queue sizes + side volume) or T0b shadow quoting.
- **Empirical findings:** `research/FINDINGS.md` — the consolidated measured record (IC scans,
  conditional-IC adverse-selection result, OOS tiers, data audits), with provenance.
- **CLI:** `commands.md`.
- **Consolidated specs/findings (appendix):** `synthesis/`.
- **Deployment runbook:** root `HETZNER_DEPLOYMENT_PLAN.md` + `cloud_deployment/`.
- **Superseded sources** (harvested into this file, kept for provenance): [`archive/`](archive/).
