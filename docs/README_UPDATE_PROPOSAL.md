# README Update Proposal — V1

**Status:** proposal, not applied. No README edit has been made.
**Author date:** 2026-08-08.

## Provenance of this audit

| Fact | Value |
|---|---|
| README last modified | `93ee007` — *docs: extend README with maker-line findings + three-class direction*, **2026-07-31 15:16 +0200** |
| HEAD at audit time | `7bcf5eb` — *Merge branch 'feat/b5a-breakeven-screen'*, 2026-08-08 |
| Commits since | **84** (43 non-merge) |
| README size | 1,693 lines |

Every claim below was checked against a live source, named per item: the generated CLI
reference (`docs/commands.md`, itself regenerated from `nat --json commands` on 2026-08-07),
the algorithm/process registries loaded in-process, `rust/ing-features/src/lib.rs`,
`docs/research/FINDINGS.md`, and `docs/PLAN.md` §0. Nothing here is inferred from prose.

**Relationship to the existing docs work:** `DOCS_IMPROVEMENT_PLAN_PROPOSAL_V1.md` audits
`PLAN`/`TASKS`/`specs` and never mentions the README. This proposal is the missing surface.
It also inherits DOCS-3's conclusion — *the reference is derived, not remembered* — and
applies it one level up.

---

## Part I — Audit findings

Ordered by severity. Severity is defined by damage on being believed, not by size of edit.

### A. CRITICAL — the README still instructs the reader to run the refuted configuration

The Q4 kill gate (2026-07-30, `FINDINGS.md` §4.6) killed all five "winners" on two defects:
a wrong-venue cost tier (1.61 bps RT = Binance VIP9, vs Hyperliquid's ~11 bps all-in) and a
sweep harness that never ran each algorithm's own logic. `dd438a0` and `93ee007` fenced the
*results tables*. They did not touch the *operating instructions*, which are what a new
reader actually executes.

| Line(s) | Text as it stands | Problem |
|---|---|---|
| 813–819 | `**Configuration:** … Fee model: 1.61 bps round-trip (Binance VIP9 taker)` | The exact defect REV-1 exists to purge, documented as current config, with no fence. `config/costs.toml` is the SSOT and `load_costs()` is a hard guardrail. |
| 97–98 | `# 6. Run paper trading (after 30 days of data)` / `nat oos30  # all 5 winning algorithms, walk-forward` | "winning" — all five are REJECTED in the lifecycle. |
| 806–811 | `# All 5 winning algorithms at once (via nat CLI)` | Same. |
| 822–830 | `nat oos30  # runs all 5 winning algos in 3 steps:` | Same, plus it presents the workflow as the project's OOS path. |
| 1222 | `nat oos30  # all 5 winning algorithms` | Same. |

A reader following Quick Start step 6 reproduces the refuted result and gets no signal that
it is refuted until §"Top Performer Algorithms", 240 lines later.

### B. CRITICAL — `surprise_signal` is un-fenced, and contradicts the same file twice

Line 555 still reads, verbatim:

> **Performance:** +3,505 bps total. Sharpe 6.7 (SOL). Captures SOL's more volatile
> microstructure transitions exceptionally well.

The other four algorithms in that section each carry a strikethrough + `**REFUTED**` note.
This one was missed. It contradicts line 350 (`| 2 | surprise_signal | +3,505 | … | **KILL** |`)
and line 1598 (`~~surprise_signal: Sharpe 6.7 SOL~~ — KILL (87.6% of edge from one day)`)
**within the same document**. A live claim that its own file refutes twice is the worst
possible state — a reader who stops at §5 leaves with the false number.

### C. CRITICAL — documented commands that do not exist

Checked against `docs/commands.md` and by grepping the `nat` parser directly. There are no
underscore aliases; these fail outright.

| README says (line) | Reality |
|---|---|
| `nat alpha_pipeline_start / _resume / _status / _gates / _step N` (724–728, 1203–1207) | Group is `alpha`: **`nat alpha pipeline-start`**, `pipeline-resume`, `pipeline-status`, `pipeline-gates`, `pipeline-step`. Zero occurrences of the underscore form in `nat`. |
| `nat mf_agent`, `nat macro_agent`, `nat meta_agent` (86–89, 592–598, 1148–1150) | `nat mf-agent`, `nat macro-agent`, `nat meta-agent` |
| `nat it_engine {start,stop,status}` (766–771, 1231) | `nat it-engine` |
| `nat validate api / positions / whales / entropy` (208–211) | The `validate` group is now `{regression, skeptical}` only. The four validation binaries are reached via `nat test validate`. |
| `nat cascade {start,once,status,stop,report}` (1236) | Group no longer exists. |

Five wrong invocations in a Quick Start is the kind of error that costs a new reader their
first hour and their trust in the rest of the document.

### D. MAJOR — every headline count is stale

| Claim | README | Measured | Source of truth |
|---|---|---|---|
| CLI commands | "~280 commands" (TOC 52, §1127, structure 1452) | **340 commands / 72 groups** | `docs/commands.md` header, generated 2026-08-07 |
| Algorithms | "25 algorithms" (banner 15, TOC 41, §255–257, 1195, 1507) | **32 registered** | `algorithms.registry.list_algorithms()` |
| Processes | *not mentioned* | **15 registered** | `processes.registry.list_processes()` |
| Feature categories | "15 categories" (§174, 194, 1462) | **21** (14 base + 7 optional) | `rust/ing-features/src/lib.rs` |
| Base / optional split | "138 base … 98 optional" (237) | **154 / 82** | `FEATURES.md` line 7–8 |
| Agent tests | "350 agent tests" (114, 1325) | suite is **4,408 passing, 0 failed** (2026-08-07, `nat`-wide) | commit *fix: suite green* |

The seven algorithms missing from the catalog table are not incidental — they are
`funding_settlement`, `hierarchical_combiner`, `jump_detector_v2`, `microprice`,
`toxic_vwap_reversion`, `vol_squeeze`, `vwap_reversion`: i.e. **precisely the maker-line and
combiner units that the README's own "Current Direction" section depends on.** The document
advertises a maker pivot while its algorithm catalog contains only the pre-pivot library.

### E. MAJOR — "236" and "209" both appear as the feature count

The vector is 236 (`Features::count_all()`). `209` survives in four places from an older
revision: the architecture diagram (130), the ingestion-layer flow diagram (194), the IT
Engine section ("across all 209 features", 764), and the project-structure tree twice
(1454 `FEATURES.md # 209-feature manifest`, 1462 `features/ # 15 feature modules (209 features)`).

Separately, the §"Feature Vector (236 Dimensions)" table lists 15 categories that **sum to
192**, and omits six modules entirely: `microstructure`, `hawkes`, `medium_freq`,
`resilience`, `cross_symbol`, `heatmap`. A schema table that does not sum to its own stated
contract is a correctness problem, because the contract (`to_vec()` length ==
`names_all()` length == Parquet schema width) is the thing the table is documenting.

### F. MAJOR — three whole subsystems are absent

The README's architecture map predates the two layers that `PLAN.md` §0 names as what
survives Q4.

**1. The PROC discovery layer** — complete end-to-end 2026-08-05. 15 registered processes
(`ic_horizon`, `mi_ksg`, `mi_combiner`, `mi_stability`, `transfer_entropy`,
`conditional_predictability`, `residualize`, `pca_combo`, `spectral`, `triple_barrier`,
`persistence_stats`, `horizon_label_scan`, `ml_importance`, `xs_rank_predictability`,
`xs_persistence`), a `nat process {list,run,results,show,standing}` group, three data levels
(bars / ticks / **candles**, PROC-19), a process→algorithm compiler (PROC-1), and a
program-level FDR ledger (PROC-13, `data/processes/fdr_ledger.jsonl`). PLAN §0: *"What
survives is the instruments …, the PROC discovery layer (complete end to end 2026-08-05),
and the methodology that caught all of the above."* The README does not contain the word.

**2. The Class-3 / XS cross-sectional layer** — the README's own "Current Direction" table
says Class 3 is *"fully data-independent — leads the program"*, and then documents none of
it. Shipped since 2026-08-06: `scripts/xs/` (breakeven, capacity, features, rotation,
trajectory), `scripts/data/fetch_candles.py --universe`, `scripts/data/fetch_l2.py`, and a
`nat xs {universe,capacity,rank,persistence,trajectory,ledger}` group (7 commands, all
`[PRELIM]`-tagged behind a "nothing promoted" banner).

**3. The candle archive** — 177 listed perps × {1m,5m,15m,1h} = **708 series, 3,059,200
candles, 98 MB, zero gaps** (XS-1, §7.1). This is the first dataset on the platform with no
integrity caveat, and it is the substrate the whole current program runs on. The README's
data section documents only the tick/Parquet path.

Missing CLI groups beyond those: `gauntlet`, `tournament`, `daily`, `nightly`, `fetch`,
`audit`, `deploy`, `service`, `signal`, `metrics`, `package`, `alg`, `alg1`, `visualize`.

### G. MAJOR — the hand-maintained CLI reference is the exact failure DOCS-3 just fixed

`docs/commands.md` now carries this header:

> **Generated** … **Do not edit by hand** … This file was hand-maintained until 2026-08-07,
> by which point 26 command groups were missing and the headline count was stale by 80. A
> reference that disagrees with the CLI is worse than no reference, because it is trusted.

README §"The `nat` CLI (~280 commands)" is ~150 lines of hand-maintained command reference
with the same disease (stale by 60, five groups wrong, 15 missing). It should not be
regenerated — it should be **deleted and replaced by a pointer**, because a second reference
is a second thing to drift.

### H. MODERATE — Key Findings stops at §4.9; five newer sections are missing

The README's findings tables end at the 2026-07-31 touch-maker experiment. Since then
`FINDINGS.md` has gained:

| § | Date | Result |
|---|---|---|
| 4.10 | 08-03 | **X-1: the maker line is fee-tier-invariant.** Staking discounts apply to fees paid, not to maker rebates; 179 day-symbol episodes re-priced, **no cell flips**. |
| 4.11 | 08-04 | **COST-5: zero fees are not free money.** Breakeven maker rate = E[adverse\|fill] − half-spread = **+0.144 / +0.159 bps**. At zero fees a resting BTC quote is still ~0.08 bps/posting under water — the venue must *pay* ~0.15 bps. |
| 7.1 | 08-07 | **XS-1 candle universe:** 708 series, 3.06 M candles, zero gaps; ~5,000-bar retention cap per interval (1m reaches only 3.5 d — 1m breadth must be **accrued, not fetched**). |
| 7.2 | 08-07 | **XS-8 spreads (n=1):** universe median half-spread **1.372 bps = 17.7× BTC**; 169 of 177 pairs wider. *NAT has been studying the extreme tight tail of its own venue.* |
| 7.3 | 08-07 | **XS-2 negative:** permutation entropy carries no cross-sectional information at bar scale (IQR 0.0005–0.0025). Contradicts `specs/maker_system.md` §5. |
| 7.4 | 08-07 | **XS-3: Track C survives its pre-registered kill test.** `xs_vol` rank-IC −0.0690 (z −8.37), `xs_momentum` −0.0387 (z −4.56), both BH q 0.007. **Both signs negative** — the "momentum" score is cross-sectional mean-reversion, independently reproducing PROC-20. |
| 7.5 | 08-07 | **XS-4:** only `vol` ranks persist (ρ₇d 0.691, half-life ~37.7 d); momentum/hurst half-lives 1.4–1.5 d. |
| 7.6 | 08-07 | **XS-5 capacity:** touch notional is the wrong instrument; at 1 % of ADV, 117 pairs support $1 k/day at ≤2 bps but only 52 support $10 k. Breadth and size trade off directly. |
| 7.7 | 08-07 | **XS-6: 0 of 6 configs survive** pre-registered criteria. It fails on **durability, not cost** (turnover 0.17–0.49 of max; cost 1–2.7 % vs 8.5 % gross). |
| 7.8 | 08-07 | **XS-9 post-mortem:** within-basket ρ 0.433 → 40 names = **≈2.2 effective bets**; −0.33 beta tilt, P&L 0.802-correlated with a *static* low-beta-minus-high-beta position. But beta earns nothing (t −1.01) while the signal **sharpens under neutralisation** (t −5.48 vs −4.08). Implementation defect, not signal defect. |
| 7.9 | 08-07 | **A5 hysteresis bands: cost saving real, net effect undecidable.** Turnover 0.199→0.018, cost 1.10→0.10 %, but gross swings 7.25→12.58 non-monotonically. The apparent winner (SR 2.99) is **reported and not adopted** — 7 configs on one window is the §4.6 pattern. |

Plus two units with no findings entry yet: **XS-10** (standing t-stat trajectory — seeded at
83 periods, t = 1.01, **325 needed, 242 remaining**) and **B-5a** (wide-pair breakeven screen
— reports the indifference exponent β\* = ln((h+rebate)/A_btc)/ln(h/h_btc), **not** a survivor
count; universe median β\* = 0.69, falsifiable by one tick measurement on one wide pair).

### I. MODERATE — the banner over-promises

Line 18: *"From order book to deployment — zero human intervention"*, and line 29's
*"register validated signals — without human intervention"*. Against `PLAN.md` §0: *"The
deployable tier is empty."* The lifecycle's sole human gate (`nat lifecycle approve`) is
also documented correctly elsewhere in the same README (914, 1646), so the banner
contradicts the body. This is the one place the README reads as marketing, and it is the
first thing anyone sees.

Similarly, §"Hypothesis Testing (H1-H5)" (1361–1373) presents five "Confirmed" verdicts with
no test window and no point-in-time stamp, in a document whose every other results table now
carries one.

### J. MINOR

- Line 15 banner: "236 features · 100ms resolution · 25 algorithms" — count drift as in D.
- Makefile quickref (108–117) duplicates the `nat` surface; `CLAUDE.md` states `nat` is the
  primary interface. Both are maintained, one is authoritative.
- §"Top Performer Algorithms" carries ~220 lines of full mathematical derivation for five
  refuted algorithms (358–557), sitting *above* the current direction. The math is worth
  keeping — the derivations are correct; it is the *P&L claims* that were refuted — but it
  belongs in `docs/research/ALGORITHMS.md`, not in the position of honour.
- `config/` described as "12 files" (1561) while the config table above it lists 17.
- "70+ test files" / "600+ algorithm tests" / "38 Optuna tests" — all hand-counted, all
  drifting, all replaceable by one number from the suite.

---

## Part II — Proposed plan

Five changes, ordered so that each one makes the next smaller. R1–R2 are the ones that
matter; R3–R5 are cleanup that becomes cheap once R2 lands.

### R1 — Truth pass (do first; ~1 h)

Non-negotiable corrections, no restructuring:

1. Fence every `nat oos30` mention and delete the `1.61 bps` fee-model block (finding A).
   Replace the "Configuration" block with a pointer to `config/costs.toml` +
   `load_costs()`, which is the guardrail the README should be teaching.
2. Strike and annotate `surprise_signal`'s performance line to match its four siblings (B).
3. Fix the five wrong command families (C).
4. Rewrite the banner and the intro paragraph to state the actual position: a research
   platform with an empty deployable tier, a refutation record, and a maker/cross-sectional
   program in flight (I). Keep `nat lifecycle approve` as the named human gate.
5. Stamp the H1–H5 table with its test window, or move it under a "historical" heading.

**Acceptance:** no un-fenced number in the README that `FINDINGS.md` §4.6 refuted; every
command in the README resolves in `docs/commands.md`.

### R2 — Replace the CLI section with a pointer (~30 min, deletes ~140 lines)

Cut §"The `nat` CLI (~280 commands)" down to:

- one line — **340 commands across 72 groups**, generated;
- a link to `docs/commands.md` and `nat commands --json` as the machine-readable SSOT;
- a curated **~15-command "what a newcomer actually runs"** block (start/stop/status/log,
  data validate, viz render, agent start/status, process list/run, xs universe/rank,
  lifecycle status, test).

Rationale is DOCS-3's, quoted verbatim in finding G. Two references drift; one does not.

### R3 — Add the three missing subsystems (~2 h)

New sections, placed *before* the legacy algorithm math:

- **Process Discovery Layer (PROC)** — the 15 processes, `nat process`, the three data
  levels, PROC-1 compiler, PROC-13 FDR ledger. One paragraph on why it exists: multiple
  testing accounted **across** the program, which is the §4.6 failure in miniature.
- **Class-3 Cross-Sectional Layer (XS)** — `nat xs`, the candle archive, the L2 sampler,
  and the XS-1→XS-10 chain with its verdicts. Every entry `[PRELIM]`; keep the "nothing
  promoted" framing the CLI group already uses.
- **The candle archive** — fold into the data section as a second substrate alongside the
  tick/Parquet path, with the retention cap stated (1m must be accrued, not fetched).

### R4 — Refresh Key Findings + Current Direction (~1 h)

Append §4.10, §4.11 and §7.1–7.9 as two compact tables (draft rows in finding H — they are
written to be pasted). Update the three-class table's Status column:

| Class | Status now |
|---|---|
| 1 — Directional bias makers | signal layer buildable; **economics closed at every reachable fee tier** (§4.10–4.11, breakeven +0.144 bps) — blocked on X-3 fill data |
| 2 — Oscillation harvesters | admission + geometry studies in-hand; unchanged |
| 3 — Cross-sectional rotation | **kill test passed (XS-3), rotation refuted (XS-6), cause diagnosed (XS-9), standing trajectory tracking 83/325 periods (XS-10)** |

And add the B-5a line: the wide-pair maker hypothesis is now a *falsifiable exponent*
(β\* = 0.69 at the universe median), not an open question — resolvable by one tick-data
measurement (B-5b).

### R5 — Move the refuted derivations (~30 min)

Relocate lines 358–557 to `docs/research/ALGORITHMS.md` under a clearly-dated "refuted,
retained for the mechanism record" heading, leaving a one-paragraph summary + link. The
README drops ~200 lines and the current program moves above the fold.

**Net effect: 1,693 → ~1,350 lines, with three more subsystems documented.**

---

## Part III — Recurrence prevention

Every finding in D and E is a hand-maintained count that drifted. The fix is the same one
DOCS-3 applied to `commands.md`, one level up.

**Proposal: a README-drift test** (`scripts/tests/test_readme_counts.py`) asserting that
the counts appearing in README prose match their registries:

| Assertion | Source |
|---|---|
| command count + group count | `nat --json commands` |
| algorithm count | `algorithms.registry.list_algorithms()` |
| process count | `processes.registry.list_processes()` |
| feature count | `Features::count_all()` via `FEATURES.md` header |
| no command string in README absent from `docs/commands.md` | parse fenced `nat …` lines |

The last assertion is the highest-value one — it would have caught findings C entirely, and
it is mechanical: extract `nat <word> <word>` from README code fences, check membership.

A weaker alternative, if the test is judged too brittle: mark the counts with HTML comment
anchors and regenerate them in the same script that produces `commands.md`. Either way the
principle from DOCS-3 holds — **derive it or drop it; do not remember it.**

---

## Part IV — Execution order & sizing

| Step | Change | Size | Blocking? |
|---|---|---|---|
| 1 | **R1** truth pass | ~1 h | No — do immediately; it is a correctness fix |
| 2 | **R2** CLI → pointer | ~30 min | No |
| 3 | **R5** move refuted math | ~30 min | No (mechanical move) |
| 4 | **R3** PROC + XS + candles | ~2 h | No |
| 5 | **R4** findings + direction | ~1 h | No |
| 6 | **Part III** drift test | ~1 h | Should land with or before R1, else R1 drifts too |

All six are data-independent and touch no code path — R1 and R2 are pure documentation
corrections, R3–R5 describe units already shipped and tested. Nothing here waits on the
streak, on su-35, or on X-3.

Suggested commits (conventional, feat branches, `merge --no-ff`, per the repo convention):

```
docs(readme): R1 — purge the refuted operating instructions
docs(readme): R2 — the CLI reference is generated, not restated
docs(readme): R5 — move refuted derivations to ALGORITHMS.md
docs(readme): R3 — document the PROC and XS layers
docs(readme): R4 — findings through §7.9, three-class status refreshed
test(docs): pin README counts to their registries
```

---

## Appendix — verification commands

```bash
git log -1 --format='%ai %h %s' -- README.md      # README age
head -12 docs/commands.md                          # 340 / 72, generation notice
python3 -c "import sys;sys.path.insert(0,'scripts');
from algorithms.registry import list_algorithms; print(len(list_algorithms()))"   # 32
python3 -c "import sys;sys.path.insert(0,'scripts');
from processes.registry import list_processes; print(len(list_processes()))"      # 15
grep -n 'count_all() = ' rust/ing-features/src/lib.rs                             # 236
grep -c 'alpha_pipeline_start' nat                                                # 0
grep -n '^### 4.10\|^### 7\.' docs/research/FINDINGS.md                           # newer sections
```
