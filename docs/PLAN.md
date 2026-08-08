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

## 0. Current Focus  *(pinned — update this block as state changes; refreshed 2026-08-06)*

**Naming, fixed here:** the conditional-IC go/no-go is **`Q5`**, everywhere and only. It was
previously also called "D1", which is the *viz* task in the D-branch. That collision is retired.

**Where the record stands.** Three things are now settled and should not be re-litigated:
the **taker path is arithmetically closed** (0.5–2 bps move vs ~11 bps RT, §2); **all five shipped
"winners" are refuted** (Q4, 2026-07-30, 5/5 KILL — wrong-venue cost tier plus a sweep harness that
never ran the algorithms' own logic, §4.6); and **passive quoting at BTC's touch is structurally
negative at every reachable fee tier** — breakeven maker rate is +0.144 bps against a zero-fee best
case (§4.10–4.11). The deployable tier is empty. What survives is the *instruments* (A4 EV gate,
queue sims, HF1 center), the PROC discovery layer (complete end to end 2026-08-05), and the
methodology that caught all of the above.

**The binding constraint has moved.** It is no longer "clean days" in general — the R1 research
program below runs entirely on data in hand. Data continuity now gates specifically **paper, live,
and the fill-economics verdict (X-3)**. The su-35 freeze and the T0b deployment stand unchanged:
**hard rule — zero su-35 contact until a clean streak completes**; verify with `nat gap status` /
`/streak` (local read, does not contact su-35).

**Do-now sequence** *(all of 1–4 are data-independent)*:
1. **`XS-1` — universe candle backfill.** The unblocker: `data/candles/` does not exist, and both
   `B-5` and all of Track C need it. `get_meta()` already parses the universe, so it is wiring.
2. **`B-5` — maker viability on wider-spread pairs.** The one maker hypothesis §4.11 left alive:
   breakeven scales with the half-spread, and every maker experiment so far ran on the three
   *tightest* symbols on the venue. §4.9 criteria imported unchanged.
3. ~~**`A-2` — combiner revalidation.**~~ ✅ **DONE 2026-08-08 — REFUTED, with `A-1`** (FINDINGS
   §5.1). The composite loses to a single feature under honest walk-forward and the agreement gate
   is *harmful*, not merely absent. **No capital-relevant claim in the record is now unrefuted**,
   and §2's adverse-selection collapse stands unopposed. The surviving open questions — B-5a's β
   conditional and Track C's beta-neutral rotation — are **time-blocked, not work-blocked**
   (XS-9's own power arithmetic: ~325 rebalances ≈ 0.89 yr), which makes the data clocks
   (`XS-7`/`XS-8`) and `REL-4`→`Q1` the work that actually shortens the wait.
4. **`COST-8`** — stop hardcoding the 0.2 bps maker rebate; it is the most load-bearing
   unvalidated number in the stack (§4.11), worth ~1.7 bps/fill.

**Ops track, in parallel:** `REL-4` (verify Telegram actually pages — the last open REL item) →
`Q1` T0b provisioning (`nat deploy cloud <ip> --dry-run` first) → `Q0` streak verification.

**Blocked on data/streak:** `X-3` fill economics (the maker go/no-go), `PROC-10` half-life,
`Q-K2` concentration verdict, `Q5` itself, and everything paper → live.

**Open bugs (P1):** `BUG-1` — the 3 ML algos are still unretrained *and* their artifacts live
off-git (`models/` is gitignored), so the trained state is unauditable from a clean checkout.
`REV-1` — §4.1-derived numbers fenced in FINDINGS but never swept from `reports/`/notebooks/
`ALGORITHMS.md`. `COST-8` (above). *(REL-1/2/3, BUG-2/3/4/5, COST-1/2/3/6/7 all shipped Jul-26→30 —
verified against merge SHAs 2026-08-06; see the TASKS reconciliation log.)*

**Dated milestones:** `Q5` conditional-IC verdict + first G8 window — **slipped past the ~Aug
target**, and honestly so: it now depends on the R1 program plus fill data that does not exist yet ·
preprint camera-ready ~Aug, SSRN/arXiv ~Sep · D2 prof-interest checkpoint ~Nov · EPFL EDFI
deadlines **Jan 15 / Mar 31 2027** · live capital only if `Q5` is positive — no date is meaningful
until X-3 has fill data.

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

- **Q0 — Verify the streak.** `nat gap status` / `/streak`. Gates paper/live and X-3 — **not** the
  R1 research program, which runs on data in hand.
- **Q1 — T0b Hetzner ingest box.** 24/7 ingestor + Telegram <5 min gap alert (per root
  `HETZNER_DEPLOYMENT_PLAN.md`). Removes single-point su-35 risk. Unblocked but for `REL-4`
  (REL-1/2/3 shipped as OPS-1/2/3, Jul-26/27).
- **Q2 — Longitudinal tool.** ✅ **DONE** — `nat oos --window <N>d` with walk-forward folds,
  deflated Sharpe and `--json` (`scripts/cli/oos.py:95,148,151`). The Q4 skeptics used it to kill
  `optimal_entry` and `jump_detector` on stored data.
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
- ~~**D2 — Modularize the `nat` monolith**~~ ✅ **DONE** — ~50 `scripts/cli/*.py` + an `app.py`
  assembler (NAT10). Was the prerequisite for packaging; D3 is now unblocked.
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
- **Q5:** conditional-IC verdict — "is there a trading business?" (gates Q-branch live work).
  *Formerly also written "D1" — that name is retired; `D1` is the viz task only.* Date slipped
  past ~Aug: it now consumes the R1 program and needs fill data (`X-2`/`X-3`).
- **D2-decision (~Nov):** prof-interest checkpoint — "do professors want this?" (gates P-branch
  effort). *Note: distinct from backlog row `D2` (CLI modularization, done) — decision points and
  task IDs share a namespace here for historical reasons; see `GLOSSARY.md`.*
- Live capital only after Q5 positive **and** G8 **and** a healthy kill-switch.

---

## 6. Companions & references  *(durable — not part of this plan, do not archive)*

- **Mission / method:** `OBJECTIVE.md`, `METHODOLOGY.md`, `GLOSSARY.md`.
- **Unit contracts:** `contracts/` — feature / algorithm / **process** / viz.
- **Process definitions:** `contracts/process.md` (contract) + `specs/process_layer.md` (the
  PROC-1..18 spec) + the `scripts/processes/` framework (7 shipped processes). (Original concept
  docs archived under `archive/in_progress/tasks_assigned_12_6_26/`.)
- **Docs improvement plan:** `DOCS_IMPROVEMENT_PLAN_PROPOSAL_V1.md` (2026-07-31, PROPOSAL) —
  audit of PLAN/TASKS/specs after the Jul-30/31 sprint. **P3 + P1 + P2 executed 2026-08-06**
  (conventions incl. the ID-registry rule; ~25 rows reconciled against merge SHAs; COST-4/5
  collision resolved by erratum → COST-6/7; this §0 rewritten; D1/Q5 clash retired) — see the
  reconciliation log at the foot of `TASKS.md`, which also records **three claims in the audit
  that did not survive verification** (BUG-1, HF4, REV-1 are all still open). **P4 remains:**
  `maker_system.md` v3 — risk/capital section, acceptance criterion (e) funding-inclusive
  accounting, warm-up table, Class-3 weight discipline, standing-monitoring phase.
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
