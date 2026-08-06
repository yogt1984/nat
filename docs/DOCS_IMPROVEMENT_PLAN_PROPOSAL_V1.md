# Docs Improvement Plan — Proposal V1

**Status:** PROPOSAL (2026-07-31). **Scope:** `PLAN.md`, `TASKS.md`, `specs/maker_system.md`,
`THREE_CLASS_RESEARCH_PROPOSAL.md`, `specs/process_layer.md`. **Trigger:** full audit of the
planning docs after the Jul-30/31 sprint (Q4 kill gate → maker line → three-class program)
revealed that **the sprint itself broke the documentation system** — the single backlog no
longer describes reality, and one ID collision entered the permanent record.

**Execution status (2026-08-06): P3 ✅ · P1 ✅ · P2 ✅ · P4 ⏳ remaining.** P3 (ID-registry +
same-branch-status conventions), P1 (row reconciliation, COST-4/5 erratum → COST-6/7, missing
done-records, three-class drain, execution-order rewrite) and P2 (PLAN §0 rewrite, D1/Q5 clash
retired) landed together. **P4 — `maker_system.md` v3 — is still open:** risk/capital section,
acceptance criterion (e) funding-inclusive accounting, warm-up table, Class-3 weight discipline,
standing-monitoring phase.

**Audit corrections found during execution** — three Part-I claims below did **not** survive
verification against code and git, and the rows stayed open: **BUG-1** (listed DONE; no retrain
commit, no artifact in the checkout, `models/` gitignored), **HF4** (listed DONE; exists only as a
flag over an externally supplied gate array, not a registered unit), **REV-1** (never executed).
The audit was also *understated* on the collision: the VIP9 purge consumed **both** COST-4 and
COST-5, not COST-4 alone. Full record: `TASKS.md` § Reconciliation log.

*(Original: on approval, P3 lands first (prevents recurrence), P1+P2 land as one reconciliation
branch, P4 as a spec revision. P1 is blocked until the in-flight `TASKS.md` working-tree edit (LF7
row) is committed.)*

---

## Part I — Audit findings

### A. CRITICAL — status drift: the backlog no longer describes reality

~25 `TASKS.md` rows are stale, most completed Jul-29..31:

| Rows | Listed | Reality (evidence) |
|---|---|---|
| REL-1/2/3 | TODO/WIP | DONE — OPS-1/2/3 merged (connect timeout, task supervision, freshness watchdog) |
| BUG-1/2/3 | TODO | DONE (ML retrain, sys.path fix, GMM 5D fixed + enabled) |
| COST-1/2/3 | TODO | DONE (cost SSOT unification + CI guard) |
| Q4 | TODO | **DONE 2026-07-30, 5/5 KILL** (`FINDINGS.md` §4.6) — the pivotal event; the row doesn't know |
| Q3 | BLOCKED "revalidate 5 winners" | MOOT — all five REJECTED in the lifecycle; PLAN knows, TASKS doesn't |
| PROC-12/6/5/13/7/8 | TODO ("start with PROC-12") | ALL DONE — the entire process-layer critical path shipped |
| HF1 / A4 / HF5 | TODO (tier 5, "only if Q5 positive") | DONE as sim-first research (maker line, §4.7–4.9) |
| PLAN §0 "Open bugs" | lists BUG-1/2/3 | all fixed |

**Internal contradiction:** row Q2 says TODO "generalize `nat oos30` → `nat oos --window`"
while TASKS' own "Verified shipped" footer lists `nat oos --window` as shipped. The Q4
skeptics *used* `nat oos --window 60d/90d`. The row is wrong.

### B. CRITICAL — ID-namespace collision (self-inflicted, on the record)

- `TASKS.md` defines **COST-4 = wave-gate thresholds → config**. The Q4 follow-ups re-minted
  **COST-4/COST-5** for the VIP9-default purge (FINDINGS §4.6 + commit d9f3c1c). Two tasks
  now share an ID in the permanent record.
- **BUG-4/BUG-5** were named, fixed, and committed (ba7b208) without ever existing as rows.
- The three-class proposal's **XS-1..6 / A-1..3 / B-1..4 / X-1..3** IDs are not reserved in
  the registry.
- Root cause: **no rule that TASKS.md is the ID registry of record, consulted before minting.**

### C. MAJOR — ordering & naming decay

- TASKS' 26-step execution order still places HF1/HF5/A4 in tier 5 "only if Q5 positive";
  reality inverted this post-Q4 (maker line pulled forward *as research*). The sequence
  describes June's strategy.
- **"D1" means two things**: PLAN §0 item 4 calls the conditional-IC gate "D1" while the
  D-branch D1 is the viz task; TASKS' Q5 row footnotes the mislabel but PLAN still carries it.
- The gate chain `Q0 → (Q1∥Q2) → Q3 → Q4 → Q5` is broken by events (Q4 done, Q3 moot) and
  never restated.
- Dated milestones reference Jun-17 (past) and an ~Aug verdict resting on a streak that does
  not exist yet.

### D. SUBSTANTIVE spec gaps (missing content, not staleness)

1. **Funding carry is modeled nowhere.** The Q4 autopsy itself proved `funding_reversion`'s
   backtest never charged funding — and neither do the maker sims, the touch-maker
   experiment, nor the proposal's acceptance criteria. Held maker inventory accrues funding
   continuously; at a ±0.03 bps/posting margin this is potentially decisive.
2. **No risk/capital section** in `maker_system.md`: per-class budgets, per-pair caps, total
   exposure, margin/liquidation distance, program-level drawdown kill.
3. **Warm-up unquantified**: "pre-warm before capital" without numbers, though
   `regime_divergence_1h` needs ≥1 h and Hurst-300 its full window — the Tier-D rotation
   cadence math depends on these.
4. **Class-3 score weights undisciplined**: `fit_C1/fit_C2 = f(…)` placeholders with no rule
   for choosing weights — an overfit door.
5. **No standing-monitoring phase**: PROC-4/10 (MI stability / half-life) exist as tasks but
   no post-validation decay monitoring is wired into the program ladder.
6. **No review cadence**: nothing schedules reconciliation — which is exactly how finding A
   happened. `specs/process_layer.md` statuses have drifted the same way.

## Part II — Improvement plan

### P3 (first — prevents recurrence): two conventions + a guard

Add to the `TASKS.md` Conventions block:
- **ID registry rule:** an ID exists when its row exists. Mint by adding the row *first*;
  only then may commits/FINDINGS/specs reference it.
- **Same-branch status rule:** a branch that completes a task flips its row (status + commit
  SHA in notes) in the same merge. "Done in code, open in TASKS" is a defect.
- Optional CI guard (S): grep merged commit subjects for `\b[A-Z]{1,5}-\d+\b`, warn when the
  named row was not touched in the merge. Advisory, not blocking.

### P1 (reconciliation pass — blocked on the in-flight TASKS.md edit landing)

One branch, one commit:
- Flip the ~25 completed rows to DONE with merge SHAs in notes (REL-1/2/3, BUG-1/2/3,
  COST-1/2/3, Q4, PROC-12/6/5/13/7/8, HF1, A4, HF5; Q3 → MOOT with pointer to §4.6).
- Fix the Q2 contradiction (row → DONE; scope residue, if any, becomes a new row).
- Resolve the COST-4 collision: rename the VIP9-purge usage to **COST-6/COST-7** via an
  erratum line in FINDINGS §4.6 and a note on the TASKS row; historic commit messages stay
  as-is (immutable), the erratum is the pointer.
- Add missing done-records as rows marked DONE (BUG-4, BUG-5, REV-1, QA-JD2, the
  touch-maker experiment) — the backlog should show what happened, not only what's next.
- Add missing live rows: **F-task** (L1 queue sizes + per-tick side volume; schema change,
  plan-first), **candles data-level runner extension** (spec §7 note), REL-4 residual
  (Telegram creds = user-side).
- Drain the 16 three-class rows (XS-1..6, A-1..3, B-1..4, X-1..3) under a new program block.
- Rewrite the execution-order section for the post-Q4 strategy (R1 program order; maker
  economics gated on fill data; tier-5 list corrected).

### P2 (PLAN §0 rewrite — the real DOCS-1)

- New Current Focus: three-class R1 (X-1 fee tier → XS track → A-2 combiner revalidation),
  T0b provisioning, streak status; Q4 aftermath one-liner.
- Kill the D1 naming collision: the conditional-IC gate is **Q5** everywhere.
- Refresh milestones against the actual streak state; remove the fixed bugs block; restate
  the gate chain post-Q4 (`Q0 → Q1 → Q5'` where Q5' consumes the three-class program).

### P4 (spec v3 — `maker_system.md` + proposal amendment)

- **New §: Risk & Capital** — per-class/per-pair budgets, total exposure cap,
  margin/liquidation distance model, program drawdown kill; leverage stated as a multiplier
  applied only after per-notional edge is proven (conversation 2026-07-31).
- **Acceptance criterion (e): funding-inclusive accounting** — every capital-relevant sim
  charges funding accrual on held inventory (rate from data, not assumption). Applies
  retroactively to any §4.7–4.9 rerun.
- **Warm-up table**: per-feature warm-up horizons; Tier-D "capital-eligible after
  max(warmups)" rule with concrete numbers.
- **Class-3 weight discipline**: score weights come only from marginal
  `xs_rank_predictability` evidence per component; joint weight optimization is prohibited
  (overfit door closed by construction).
- **Standing-monitoring phase** appended to the program ladder: validated findings enter
  PROC-4/10 decay monitoring; a decay breach demotes via the lifecycle.
- **Weekly reconciliation ritual** (one line in TASKS conventions + PLAN §0): statuses,
  ledger, and §0 refreshed weekly; quarterly DONE sweep to archive (already conventioned,
  never executed — first sweep due).

## Part III — Execution order & sizing

| Step | Blocked on | Size |
|---|---|---|
| P3 conventions (+ optional CI guard) | — | XS (+S) |
| P2 PLAN §0 rewrite | — | S |
| P4 spec v3 + proposal amendment | — | S/M |
| P1 TASKS reconciliation | user commits the in-flight TASKS.md edit | M |

All four are documentation-only; no code paths change. Every edit cites merge SHAs so the
reconciliation is itself auditable.
