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

*Numbers are stable identifiers, not positions — 6, 10 and 12 are deliberately absent (see the footnote below) and are never reused.*

| # | Family — economic source | Canonical literature | NAT status |
|---|---|---|---|
| 1 | **Liquidity provision** — paid to bear inventory + adverse selection | Grossman & Miller (1988); Ho & Stoll (1981); Avellaneda & Stoikov (2008); Guéant, Lehalle & Fernandez-Tapia (2013) | Heavily worked. Blocked on an **unearned rebate tier** + fill data (§4.7–§4.11) |
| 2 | **Adverse-selection avoidance** — don't be the informed trader's counterparty | Kyle (1985); Glosten & Milgrom (1985); Easley et al. (1996) PIN; Easley, López de Prado & O'Hara (2012) VPIN | Survives **as a gate only** — VPIN lifts Sharpe 3/3 symbols but carries no direction (§4.5) |
| 3 | **Price-discovery lag** — learn a price before another venue does | Hasbrouck (1995) information share; Gonzalo & Granger (1995); Makarov & Schoar (2020) | **Untried.** Needs the `F9` cross-venue feed (specced, unbuilt) |
| 4 | **Funding reflexivity** — crowded positioning pays funding → forces unwinding → moves price → changes positioning. A *feedback loop*, not a carry trade | Perpetual funding mechanics; Alexander et al. on perpetual basis | **Untried.** `funding_reversion` was refuted as a *directional signal*, which is a third thing again; funding is charged **nowhere** in any sim (§4.6), so it is simultaneously an unpriced cost and an untested edge |
| 5 | **Deterministic liquidation** — an engine executes at a price anyone can compute from public positions. *Not* a fire sale: no discretion, no delay, trigger known in advance | Coval & Stafford (2007) is the nearest equity analogue and is **weaker** — their seller chooses; Shleifer & Vishny (1997) | **Untried in practice, and the best-supported family here.** H3 *confirmed* in the hypothesis suite; blocked only by K2 dead columns — a plumbing problem, not a research one (§7, §8) |
| 6 | **Gradual information diffusion** — a slow-to-update holder sells to you early and buys from you late; diffusion is *slower in less-visible assets* | Hong & Stein (1999); Hong, Lim & Stein (2000) *bad news travels slowly*; Da, Gurun & Warachka (2014) frog-in-the-pan; Duffie (2010) slow-moving capital | **Untried, and NOT excluded by the reversion results** — see [Why the reversion finding does not close it](#why-the-reversion-finding-does-not-close-family-6). Test is `XS-11` |
| 7 | **Attention & flow-driven demand** — listings, inclusion, retail flow | Shleifer (1986); Harris & Gurel (1986) index inclusion; Barber & Odean (2008) attention | **Untried — data already on disk.** 177 listed / 55 delisted, listing events in the candle archive (§7.1) |
| 8 | **Statistical relative value** — cointegrated mispricing between instruments | Gatev, Goetzmann & Rouwenhorst (2006); Avellaneda & Lee (2010) | `relative_value_pairs` registered but **never evaluated** |
| 9 | **Behavioural under/over-reaction** — genuine mis-pricing, no intermediary role | Jegadeesh & Titman (1993); De Bondt & Thaler (1985); Lehmann (1990) | **The naked signature is exhausted, the mechanisms are not** — see [Signatures vs mechanisms](#signatures-are-not-mechanisms). Trading the sign of recent returns is dead at bar scale on this universe (PROC-20 §5, TC-1 §7.13, XS-3 §7.4) |
| 11 | **Microstructure noise / bid-ask bounce** | Roll (1984); Hasbrouck (1993) | Exhausted (mean-reversion suite) |

**Not carried in the table** — two families are real in the literature but *operationally
inaccessible here*: volatility risk premium (Carr & Wu 2009 — no options data) and
latency/queue priority (Budish, Cramton & Shim 2015 — REST/WS access, no colocation).
Revisit only if the access changes. *Family 6 was in this list until 2026-08-10 and should
not have been: "slow-moving capital" is testable cross-sectionally with data already on
disk, and is now carried as gradual information diffusion.*

**A caution on the skeleton.** Ten of the families are drawn from *equity* market
microstructure. Perps differ structurally — 24/7 so no open/close/overnight effects, no
settlement, funding as a continuous forced payment, retail-dominated leverage, no NBBO — so
this taxonomy should be treated as a starting frame that crypto-native mechanisms may not fit.
Families 4 and 5 were already respecified once for exactly this reason.

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
| | Family 6 — gradual information diffusion | News reaches a thin pair's holders over days, so price drifts while it does (Hong & Stein) — **untested here**, see [why](#why-the-reversion-finding-does-not-close-family-6) |
| | Family 7 — attention | Flow arrives in waves |
| | Family 9 — behavioural | Under-reaction |

**What NAT exhausted was the naked signature** — take the sign of recent returns, at bar scale,
on this universe, at taker cost. Three instruments agree, and that is settled. It says nothing
about the mechanisms that *generate* reversion, which are largely untried — and it is **pooled**
over a universe whose liquidity spans a 27× spread range, which is exactly the pooling that
hides family 6.

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

### Why the reversion finding does not close family 6

"The universe reverts 8-to-1" is the most robust result in the record and it is **silent about
this family**, for two separate reasons.

**Horizon.** PROC-20 tested 1 m/5 m, TC-1 tested 15 m/1 h, XS-3's longest was 7 days. Diffusion
momentum lives at **weeks to months**. Reversal at minutes and momentum at months are not in
conflict — they are the standard term structure of autocorrelation (Lehmann 1990 short-horizon
reversal; Jegadeesh & Titman 3–12 month momentum; De Bondt & Thaler multi-year reversal). NAT
has measured the short end and generalised from it.

**Pooling.** XS-3 ran over all 177 pairs with **no liquidity conditioning**, and XS-6 traded
only the 119 pairs inside a 2 bps spread ceiling. A liquidity-*dependent* effect is therefore
invisible to both: diluted toward zero in the first, excluded outright from the second. A
pooled IC of −0.039 is perfectly consistent with, say, +0.10 in the illiquid third and −0.10 in
the liquid third.

**The mechanism, stated so it can fail.** Information reaches holders of thin, low-attention
perps more slowly than holders of BTC, so their prices adjust over days rather than seconds.
The counterparty is the slow-to-update holder. The crypto analogue of Hong–Lim–Stein's "low
analyst coverage" is a thin, rarely-quoted pair.

**The literature is genuinely split on the payoff, and that split is the study.** Hong, Lim &
Stein find illiquidity *creates* momentum; Avramov, Cheng & Hameed (2016) find momentum is only
*realisable* in liquid names. Both can hold — the effect real and untradeable — which is this
project's single most repeated outcome, so the test must report tradability separately from
existence.

**The test** (`PROC-6` already computes exactly this — MI(f; y | Z = z) as a function of z,
which is the pooled-vs-conditional fix):

1. Condition momentum IC on a **liquidity bucket** (spread or ADV tercile from XS-8 / the
   candle archive), at **1-week and 1-month** horizons — not the minute scale already tested.
2. Run the breakeven arithmetic **per bucket**, so the verdict is one of *real and tradeable*,
   *real and untradeable*, or *absent* — never a single pooled number.
3. Expect the power bar to bite: 90 days of 1 h candles is ~12 non-overlapping weeks, so an
   *undecidable* verdict is the likely honest outcome and must be reported as such rather than
   as refutation.

**Known biases, both working against a clean read:** cost is worst exactly where the edge is
predicted (illiquid pairs run 3–27 bps half-spread against BTC's 0.078), and survivorship is
worst there too — delisting happens in the thin tail, and the 55 delisted perps are absent from
the archive.

### The structural advantage not yet used

**Hyperliquid is on-chain.** Wallet positions, liquidation levels and cohort composition are
*observable*; on a centralised venue they are invisible. That makes families **5** and **7**
unusually tractable here — Coval & Stafford must *infer* forced flow from returns, whereas this
venue largely lets you **see** it.

This is a structural edge in the literal sense: it comes from the venue's architecture, not from
a better estimator, and it cannot be lifted from an equities paper.

---

## Ranked shortlist — untried, by **expected value**

*Revised 2026-08-10. The first version ranked by cost-to-attempt and presented it as priority,
which put cheap-but-weak first and buried the best-evidenced item last. Cost is a tiebreak, not
the ordering.*

1. **Deterministic liquidation** (family 5) — the mechanism is already *confirmed* (H3), the
   venue makes the flow **observable** rather than inferred, and the market condition that pays
   it — reversion — is the most robustly measured result in the record (TC-1, PROC-20, XS-3).
   Blocked only by K2 dead columns: plumbing, not research.
2. **Cross-venue dislocation** (family 3) — strongest crypto-*specific* evidence base; Makarov &
   Schoar document large, persistent cross-venue price differences. Costs building the `F9`
   feed, which is real work, and it is still worth ranking above cheaper items.
3. **Funding reflexivity** (family 4) — data already exists, and the term is unmodelled in
   *both* directions, so the first measurement is informative whichever way it lands.
4. **On-chain cohorts** (families 5/7) — specced as `WP-1..5`; shares family 5's observability
   advantage.
5. **Information diffusion × liquidity** (family 6) — the only shortlist item whose test needs
   **no new data**: PROC-6 already computes conditional MI, and the candle archive already
   carries the universe. Ranked below the four above because the horizon it lives at
   (weeks–months) is the one where 90 days of history has the least power, so *undecidable* is
   the likely verdict. Cheap enough that the answer is worth having anyway.
6. **Listing & delisting dynamics** (family 7) — **demoted to a collection task.** The 90-day
   archive contains exactly **two** observable listing events (GRAM 2026-07-02, CASHCAT
   2026-07-11). The first version of this document called it "free — data already on disk",
   which was true of the *data* and false of the *sample*: n = 2 supports no study. Needs venue
   listing history, or forward accumulation. Recorded because the error is instructive — cheap
   is not the same as ready, and ranking by cost is how that mistake gets made.

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
