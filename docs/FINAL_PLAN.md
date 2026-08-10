---
title: Final Plan
purpose: The terminal sequence — what is still worth building, the pre-registered rule for stopping, and what acceptance of failure actually means
type: planning
status: living
branch: Q
updated: 2026-08-10
---

# FINAL_PLAN — the last things to try, and how to stop

**This document does not carry work items.** [`TASKS.md`](TASKS.md) is the ID registry of record
and every ID below is a pointer into it; duplicating a row here would create exactly the drift
the registry rule exists to prevent. What this document adds is the thing `TASKS.md` cannot
express: **a terminal sequence with a committed end**, written before the result is known.

It exists because "I tried and failed" is a claim, and a claim needs the same standard as any
other in this repo — pre-registered, falsifiable, and decided against criteria fixed in advance
rather than against how the last run felt.

---

## 1. Where the record stands

Six months (first commit 2026-02-17), 1,154 commits, ~232k LOC, ~10,000 configurations tested,
**14 refuted mechanisms, zero unrefuted capital-relevant claims.** The full record is
[`research/FINDINGS.md`](research/FINDINGS.md); the one-page version is
[`research/FINDINGS_SUMMARY.md`](research/FINDINGS_SUMMARY.md).

The pattern behind every refutation is one line, and it is not about skill:

> **The predicted move is comparable to the cost.**

Taker: a 0.5–2 bps move at 1–5 s against an 11 bps round trip — a cost ratio of ~0.1, knowable on
day one without a single measurement. Maker: breakeven at BTC's touch is **+0.144 bps**, i.e. the
venue must *pay* you before a resting quote is level, and 0 of 8 cells survive at every reachable
fee rung (§4.11).

**What this is not.** It is not a failure to search hard enough. Nearly everything tested belongs
to one mechanism family — short-horizon order-flow pressure in one venue's book — and that family
is closed by arithmetic. The mistake was **target selection**, not rigor, and it was available to
be avoided in week one.

---

## 2. What is still worth building

Four items, in dependency order. IDs are live in [`TASKS.md`](TASKS.md) §Execution-order tier 0.

| # | ID | What | Why it is on this list |
|---|---|---|---|
| 1 | **`WP-2`** | Position snapshot collector | **S effort, and it starts a clock nothing else can start.** `WP-5` needs ≥90 days, so the earliest verdict is **2026-11-08** and it slips one-for-one with every day of delay. Same shape as the `XS-7` retention cap: when the constraint is accrual, delay is irreversible. |
| 2 | **`COST-9` → `LF8`** | Charge funding in the sim; then funding carry as a *held* position | The only untried family **decidable inside the window** — data already on disk. `funding_reversion` was refuted as a *directional* signal (§4.6), which says nothing about *collecting* funding while hedged. `COST-9` first, or the study is circular: funding is currently charged in **no** simulation, in either direction. |
| 3 | **`WP-3`** | Cohort construction | Carries the **early kill**. If cohort-membership autocorrelation fails, family 5's positioning branch is dead *permanently* at ~30 days — no need to wait for `WP-5`. This is the cheapest-refuting-measurement discipline applied to the schedule itself. |
| 4 | **`XS-11`** | Momentum IC conditioned on liquidity | Cheap filler, no new data. *Undecidable* is the **pre-recorded** expected verdict — run it because the answer is worth having, not because it is expected to land. |

**Why family 5 (`WP-*`) is the one that earns the wait.** It is the only place NAT holds a
structural advantage rather than competing on speed or terms: on this venue positions, entry
prices and liquidation levels are **computable in advance**, where a CEX makes them invisible. H3
(liquidation cascade) was *confirmed* in the hypothesis suite and has **never been tested on live
data**, because the features are K2 dead columns (`rust/ing/src/state/mod.rs:127`,
`position_state: None`). An untested confirmed mechanism is a strange thing to be sitting on.

**Running free, requiring no decision:** ingestion, the `XS-7` daily candle refresh, and the
`XS-10` trajectory recorder. Track C resolves itself at ~0.89 yr. **Do not re-fit it.**

**Explicitly not before the review:** `F9` (real work, ranks below family 5); any new feature,
config or CLI surface (10,000 configurations returned zero, and each extra trial raises the DSR
bar by √(2·ln N) for free); the maker book line beyond `B-5c` (§4.11 closed it at every reachable
rung); and the `LOOP-4` ideation layer — **idea supply is not the binding constraint.**

---

## 3. Routes already closed — do not reopen them

| Route | Why it is closed | §|
|---|---|---|
| Taker capture at 1–5 s | Cost ratio ~0.1. Arithmetic, permanent | §2 |
| Naive maker fills | IC 0.45 → 0.03/−0.06/−0.03; the fill *requires* the adverse move | §2 |
| Passive quoting at BTC's touch | Breakeven +0.144 bps; 0/8 cells at every rung | §4.11 |
| Fee-tier optimisation | HYPE staking is tier-invariant; discounts apply to fees, never rebates | §4.10 |
| **Becoming a market maker** | **No program exists to apply to.** The first viable rung (rebate_t2) needs ~\$86 M/day in maker fills, and 0/8 cells survive even there | **§7.15** |
| The five shipped "winners" | Wrong-venue cost tier + a harness that never ran their own entry logic | §4.6 |
| Zero fees as a fix | Not the relevant threshold — a *negative* fee is, and even rebate_t3 does not clear the criteria | §4.11 |

---

## 4. The stop rule — `REV-2`, 2026-10-10

Committed now, before the result. **The review decides *continuation*, not *truth*.**

At 2026-10-10 `WP-2` holds ~60 days of history: `WP-3`/`WP-4` (≥30 d) are answerable and **`WP-5`
is not** — it needs ≥90 d, i.e. **2026-11-08**. Declaring failure on 10-10 would file
*undecidable* as *refuted*, which is the single error the death-reason table in
[`research/MECHANISM_FAMILIES.md`](research/MECHANISM_FAMILIES.md) exists to prevent, and the most
expensive one available to this project.

| Outcome at 2026-10-10 | Verdict | Action |
|---|---|---|
| `WP-3` cohort autocorrelation fails | Family 5 positioning branch **dead, permanent** | Stop. Do not wait for `WP-5` |
| Cohorts stable | Live question, 4 weeks out | Finish `WP-5` to 2026-11-08 **whatever else has died** |
| `LF8` negative at honest cost | Family 4 **dead, permanent** | Close it; it does not gate the rest |
| Everything undecidable | The constraint is **span**, not idea quality | A *collection* decision, not a research one |

**One exception to the calendar.** The liquidation branch (H3 / K2) is **event-limited, not
span-limited** — levels are computable from a single snapshot; what is needed is cascades, and
those cluster in volatility. Count events. Do not schedule it by date.

**Terminal condition.** If `WP-5` and `LF8` both die for a **permanent** reason — *arithmetic* or
*mechanism-absent*, never *undecidable* or *blocked* — the search is closed. NAT becomes a
methodology and dataset project, and `DOCS-4` + `P6` are the remaining deliverables.

---

## 5. What "accept the failure" actually means

Precision here matters, because the rough sentence and the true one differ.

**Not defensible:** *"I tried to find alpha and failed."* Too vague to be worth saying, and it
implies a swing and a miss.

**Defensible, once the sequence above is run:**

> *I tested one venue's order book at retail cost terms across ~10,000 configurations and 14
> mechanisms; established the space is closed by arithmetic rather than by my execution of it;
> tested the on-chain families where the venue gave me a structural advantage; and documented
> every death with its cause.*

The difference between those two sentences is `WP-2` — **S effort and ~30 days** for the early
kill. That is the whole price of never having to qualify the claim.

**What acceptance is not.** Stopping a project that returned zero after six months, with a
documented reason and a rule fixed in advance, is the correct response to evidence. The actual
failure mode is the opposite: continuing indefinitely, the trial ledger growing, the DSR bar
rising, until a false positive is manufactured. That outcome is **guaranteed, not risked** — and
the ledger is the only thing standing between the two.

---

## 6. What survives acceptance

These pay whether or not anything is ever found, and they are currently **banked but
unrealised**. Build them *during* the wait, not after.

- **`DOCS-4`** — `reproduce.sh` + a frozen public data slice. 232k LOC and no way for a stranger
  to regenerate a single number; a reviewer who cannot run it reads only prose.
- **`P6`** — the negative-results writeup, *"Fourteen mechanisms that did not work on a perp
  DEX"*. Organised by **mechanism and death reason**, never by error code. Two honest
  contributions: the graveyard (negative results are unpublishable in journals, so everyone
  re-derives the same walls) and the **descriptive** measurements, which are the stronger
  academic angle — universe median half-spread 1.372 bps = 17.7× BTC, spread ⟂ depth
  (ρ = −0.107), median touch notional \$33–391 with 4/177 pairs holding \$5 k, ~5000-bar
  retention making 1 m history unbackfillable (§7.1, §7.2, §7.10). Declare the survivorship gap
  (55 delisted perps absent) rather than let a referee find it.

**Publish the method and the graveyard. Never publish a live edge.**

The durable asset was never the alpha. It is a falsification harness that killed its own best
result — five signals promoted to VALIDATED, one backtesting at Sharpe 9.2, shown to lose
**−11.1 bps** on BTC across 57 OOS dates once priced honestly. A method that deletes its own
winners on its own evidence is rarer than a winning backtest, and considerably harder to fake.

---

## 7. The transferable rule

If the answer is no and the next target is elsewhere, exactly one habit carries over:

> **Run expected-move ÷ round-trip-cost on the new venue before building anything. Ratio ≥ 3, or
> do not start.**

Twenty minutes. It would have redirected this entire project in week one. Everything else in this
repository is downstream of not having done it.

---

*Registry: [`TASKS.md`](TASKS.md) — tier 0 of the execution order carries the sequence, `REV-2`
the review. Evidence: [`research/FINDINGS.md`](research/FINDINGS.md). Taxonomy:
[`research/MECHANISM_FAMILIES.md`](research/MECHANISM_FAMILIES.md). Nothing in this file is a
source; when it disagrees with FINDINGS, FINDINGS wins.*
