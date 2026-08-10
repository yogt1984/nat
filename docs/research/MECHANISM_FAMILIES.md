---
title: Mechanism Families
purpose: Taxonomy of edge sources by economic mechanism — who is on the other side and why they lose
type: reference
status: living
branch: Q
updated: 2026-08-10
---

# Mechanism Families — the edge-source taxonomy

**Companion to [`INSTITUTIONAL_ALGORITHMS.md`](INSTITUTIONAL_ALGORITHMS.md).** That document
organises the literature by *mathematical technique* (microstructure / execution / vol / ML /
crypto). This one organises it by **economic source**: who is on the other side of the trade,
and why they lose. Every status claim traces to [`FINDINGS.md`](FINDINGS.md) — when the two
disagree, FINDINGS wins.

---

## Why this document exists

A technique taxonomy hid a structural problem. Hawkes processes, order-flow imbalance, VPIN,
jump tests, OU mean-reversion and book pressure are six *different techniques* — and all six
draw on **one economic source**: short-horizon order-flow pressure in a single venue's book.

That is why they failed the same way. [`FINDINGS_SUMMARY.md`](FINDINGS_SUMMARY.md) states the
pattern behind every refutation in the record:

> *"Every refuted result shares one property: the predicted move is comparable to the cost."*

**Two techniques applied to one edge source are not diversification.** Reorganising by mechanism
makes the concentration visible, and makes the untried directions nameable.

As of 2026-08-10 the record holds **14 refuted mechanisms and zero unrefuted capital-relevant
claims** — after roughly 10,000 tested configurations (PROF-1 alone swept 9,668 cells, 0
survivors). Parameter breadth has a *measured* return of ~0 here. Breadth across **families** is
the expansion axis with positive expected value.

## How a family dies — and whether it stays dead

Death reasons are not equivalent, and conflating them is how a live question gets buried:

| Death reason | Permanent? | Implication |
|---|---|---|
| **Arithmetic** — predicted move ≈ cost | Yes, at that horizon | Only a cost change reopens it |
| **Mechanism absent** — measured, not there | Yes | Closed |
| **Undecidable** — underpowered | **No** | Revisit at n; a *collection* task, not a research one |
| **Blocked** — data missing | **No** | Acquire the data |

Only the first two exhaust a family. Track C and oscillation harvesting sit in the third row,
not the second — the record says so explicitly ("underpowered, *not* absent"; "starved of
events"). Treating them as refuted would discard live questions.

---

## The families

| # | Family — economic source | Canonical literature | NAT status |
|---|---|---|---|
| 1 | **Liquidity provision** — paid to bear inventory + adverse selection | Grossman & Miller (1988); Ho & Stoll (1981); Avellaneda & Stoikov (2008); Guéant, Lehalle & Fernandez-Tapia (2013) | Heavily worked. Blocked on an **unearned rebate tier** + fill data (§4.7–§4.11) |
| 2 | **Adverse-selection avoidance** — don't be the informed trader's counterparty | Kyle (1985); Glosten & Milgrom (1985); Easley et al. (1996) PIN; Easley, López de Prado & O'Hara (2012) VPIN | Survives **as a gate only** — VPIN lifts Sharpe 3/3 symbols but carries no direction (§4.5) |
| 3 | **Price-discovery lag** — learn a price before another venue does | Hasbrouck (1995) information share; Gonzalo & Granger (1995); Makarov & Schoar (2020) | **Untried.** Needs the `F9` cross-venue feed (specced, unbuilt) |
| 4 | **Risk transfer / carry** — paid a premium to hold what others shed | Perpetual funding mechanics; Alexander et al. on perpetual basis | **Untried as a carry position.** `funding_reversion` was refuted as a *directional* signal; funding is charged **nowhere** in any sim (§4.6) |
| 5 | **Forced / constrained flow** — someone must trade regardless of price | Coval & Stafford (2007) fire sales; Shleifer & Vishny (1997) limits of arbitrage | **Untried in practice.** H3 (cascade prediction) was *confirmed* in the hypothesis suite, but the features are K2 dead columns (§7, §8) |
| 6 | **Slow-moving capital / segmentation** | Duffie (2010), *Asset price dynamics with slow-moving capital* | Untried |
| 7 | **Attention & flow-driven demand** — listings, inclusion, retail flow | Shleifer (1986); Harris & Gurel (1986) index inclusion; Barber & Odean (2008) attention | **Untried — data already on disk.** 177 listed / 55 delisted, listing events in the candle archive (§7.1) |
| 8 | **Statistical relative value** — cointegrated mispricing between instruments | Gatev, Goetzmann & Rouwenhorst (2006); Avellaneda & Lee (2010) | `relative_value_pairs` registered but **never evaluated** |
| 9 | **Behavioural under/over-reaction** — genuine mis-pricing, no intermediary role | Jegadeesh & Titman (1993); De Bondt & Thaler (1985); Lehmann (1990) | **The naked signature is exhausted, the mechanisms are not** — see [Signatures vs mechanisms](#signatures-are-not-mechanisms). Trading the sign of recent returns is dead at bar scale on this universe (PROC-20 §5, TC-1 §7.13, XS-3 §7.4) |
| 10 | **Volatility risk premium** — sell insurance | Carr & Wu (2009); Bakshi & Kapadia (2003) | **Inaccessible** — no options data |
| 11 | **Microstructure noise / bid-ask bounce** | Roll (1984); Hasbrouck (1993) | Exhausted (mean-reversion suite) |
| 12 | **Latency / queue priority** — pure speed | Budish, Cramton & Shim (2015) | **Not viable** — REST/WS access, no colocation |

### Signatures are not mechanisms

**Momentum and mean-reversion are not families.** They describe the *sign of autocorrelation* —
an observable. A family names **who pays you**. Several unrelated mechanisms produce the same
signature, and the distinction decides whether a result generalises:

| Signature | Produced by | Who pays, and why |
|---|---|---|
| **Reversion** | Family 1 — liquidity provision | You absorb an imbalance; the reversion **is** the compensation (Grossman & Miller) |
| | Family 5 — forced flow | A liquidation overshoots fair value and snaps back once forced selling ends (Coval & Stafford) |
| | Family 11 — microstructure noise | Bid–ask bounce. Mechanical, no economic content (Roll) |
| | Family 9 — behavioural | Genuine over-reaction (De Bondt & Thaler) |
| **Momentum** | Family 3 — price-discovery lag | One venue leads; "continuation" is information arriving (Hasbrouck) |
| | Family 6 — slow-moving capital | Capital arrives gradually, so price trends while it does (Duffie) |
| | Family 7 — attention | Flow arrives in waves |
| | Family 9 — behavioural | Under-reaction |

**What NAT exhausted was the naked signature** — take the sign of recent returns, at bar scale,
on this universe, at taker cost. Three instruments agree, and that is settled. It says nothing
about the mechanisms that *generate* reversion, which are largely untried.

**And the strongest result in the record is a signpost, not a dead end.** "The universe reverts
8-to-1" (TC-1) is *precisely the condition under which liquidity provision and forced-flow
harvesting pay*, because both are compensated **by** reversion. The finding therefore points
directly at families **1** and **5** — uncomfortably, the two NAT cannot currently execute
(family 1 blocked on an unearned rebate tier, family 5 on K2 dead columns). But *blocked* is row
four of the death table, not row two.

**The question to ask any reversion or momentum proposal is therefore not "what horizon?" but
"paid by whom?"** A proposal that cannot name the counterparty and their reason for losing is
proposing a signature, and belongs in family 9 — where the record says the naked version does
not pay.

### The structural advantage not yet used

**Hyperliquid is on-chain.** Wallet positions, liquidation levels and cohort composition are
*observable*; on a centralised venue they are invisible. That makes families **5** and **7**
unusually tractable here — Coval & Stafford must *infer* forced flow from returns, whereas this
venue largely lets you **see** it.

This is a structural edge in the literal sense: it comes from the venue's architecture, not from
a better estimator, and it cannot be lifted from an equities paper.

---

## Ranked shortlist — untried, by cost to attempt

1. **Listing & delisting dynamics** (family 7) — *free*. The events are already in the candle
   archive. In a universe that measurably mean-reverts, listings are one of the few places with
   a **structural** reason for one-sided flow.
2. **Funding carry as a position** (family 4) — data exists. Note the asymmetry: because funding
   is modelled nowhere, it is simultaneously an **unpriced cost** in every existing backtest and
   an **untested edge**.
3. **Liquidation-cascade mechanics** (family 5) — mechanism already confirmed (H3); blocked only
   by dead feature columns, i.e. a plumbing problem, not a research one.
4. **Positioning / on-chain cohorts** (family 5/7) — specced as `WP-1..5`.
5. **Cross-venue basis / lead-lag** (family 3) — strongest academic support for crypto
   specifically (Makarov & Schoar documented large cross-venue dislocations), but requires
   building `F9` first, so it is real work rather than free.

---

## How to use this

- **A proposal names its family.** From an *untried* family it gets budget priority; from an
  *exhausted* family it must state what changed — a new cost tier, horizon, or instrument set —
  or it is rejected before any test is run.
- **Allocate trials per family, not per idea**, so one prolific family cannot consume the
  multiple-testing budget. Every trial raises the deflated-Sharpe bar by √(2·ln N) without
  adding independent information.
- **Breadth across families; deliberate poverty of parameters within one.** A proposal shaped
  like "sweep k ∈ {10, 20, 40}" is trial-count inflation wearing the costume of thoroughness.

---

*Related: [`FINDINGS.md`](FINDINGS.md) (the measured record) ·
[`FINDINGS_SUMMARY.md`](FINDINGS_SUMMARY.md) (one page) ·
[`INSTITUTIONAL_ALGORITHMS.md`](INSTITUTIONAL_ALGORITHMS.md) (technique view + gap audit) ·
[`PAPERS_IDEAS.md`](PAPERS_IDEAS.md) (reading bibliography) ·
[`ALGORITHMS.md`](ALGORITHMS.md) (implemented catalogue).*
