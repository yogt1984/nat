# Wallet Positioning — collector, cohort construction, and the predictability test

**Status:** SPEC (2026-08-08). **Owner:** Onat.
**Empirical basis:** `research/FINDINGS.md` §2 (adverse selection), §4.6 (Q4 kill gate), §5.1
(combiner retired), §7.10 (B-5a). **Contracts:** `contracts/process.md`, `contracts/feature.md`.
**Pattern to copy:** `scripts/data/fetch_l2.py` (XS-8) — the same collector shape, the same
failure discipline, the same systemd clock.

---

## 0. Why this, and why now

Every refuted result in the record shares one property: **the predicted move is comparable to
the cost.** Taker at 1–5 s (0.5–2 bps vs 11 bps RT). VWAP reversion at ~50 trades/day. Passive
quoting where the half-spread is 0.083 bps against 0.23 bps of adverse selection. That family
is closed by arithmetic (§2, §4.6, §4.11, §5.1).

Wallet positioning changes over **hours to days**, so a 50–500 bps move meets the same ~11 bps
cost — a coverage ratio of 5–50× instead of 0.1×. And it is not a latency race: seeing a
*public* position confers no advantage on whoever sees it 10 µs sooner.

**The structural asymmetry.** Hyperliquid is a fully on-chain perp DEX, so per-account
positions, entry prices, leverage and therefore **liquidation prices** are queryable. Every
venue the microstructure literature is built on hides this — PIN and VPIN exist precisely
because identity must be *inferred* when it cannot be observed. Here it can be observed. The
asymmetry is that the data is public but underused, not that it is private.

**What already exists** (do not rebuild): `HyperliquidClient.get_positions(wallet)` wraps
`clearinghouseState` and returns a typed `Position` (coin, signed size, entry, value, uPnL,
**liquidation_price**, leverage, margin). The parquet already carries 15 `whale_*` columns,
`top5_concentration`, `herfindahl_index`, `position_count` and 8 `hm_*` liquidation-map
columns. `nat2` recorded the fact that makes cohorts tractable: **2.9 % of wallets hold 21.9 %
of notional.**

**What must not be assumed.** H1–H6 ("whale flow → returns" among them) were confirmed in the
era whose confirmations later collapsed 5-for-5 under the Q4 gate. This spec **re-tests from
zero** and cites those results as hypotheses, never as support.

---

## 1. The claim being tested, stated so it can fail

> The current positioning of an empirically-identified profitable cohort carries information
> about forward returns, over horizons where the move exceeds cost, and that information is
> durable across days rather than concentrated in a few episodes.

Three ways it dies, all of which the design must be able to report:

1. the cohort is not identifiable (P&L ranking is unstable — yesterday's winners are not
   today's);
2. the cohort is identifiable but its positioning carries no forward information;
3. it carries information that is not durable (§4.9's binding failure — day-consistency and
   concentration, not the pooled mean).

---

## 2. Build order

Each step ships independently, is tested before the next begins, and **step 4 is the first one
allowed to look at forward returns.** Steps 1–3 build the substrate; a step that peeks early is
how §5.1 happened.

| # | Unit | File | Effort | Gate to proceed |
|---|---|---|---|---|
| 1 | Wallet roster | `scripts/data/wallet_roster.py` | S | roster reproducible, sourced not hardcoded |
| 2 | Position snapshot collector | `scripts/data/fetch_positions.py` | S | sweeps land, failures survivable, clock installed |
| 3 | Cohort construction | `scripts/wallets/cohorts.py` | M | cohorts are causal + stable enough to be a cohort |
| 4 | `cohort_predictability` process | `scripts/processes/cohort_predictability.py` | M | null-calibrated, day-durable, FDR'd |
| 5 | Pre-registered study | `scripts/exploration/wallet_positioning_study.py` | S | criteria committed **before** the run |

---

## 3. Step 1 — wallet roster

**What.** Produce the list of wallets to track, with provenance. Sources, in order of
preference:

- venue leaderboard endpoint if one is exposed (verify against live API — do not assume);
- wallets observed in `WsTrade.users` on the tick feed (already populated, per the T0 verdict) —
  aggregated over a window and ranked by traded notional;
- an explicit pinned list in `config/wallets.toml` for known addresses.

**How.** `fetch_roster(source=..., info_fn=None) -> list[WalletRef]`, with `info_fn` injected so
the suite runs offline. A `WalletRef` carries `address`, `source`, `first_seen`, `notional_seen`.

**Non-negotiable:** the roster is **derived, never hardcoded**. A frozen list rots the moment the
cohort turns over, and cohort turnover is one of the hypotheses under test (§3 failure mode 1).

**Tests** (`tests/test_wallet_roster.py`):
- address validation — anything not a 0x-prefixed 40-hex string is rejected *before* it can
  reach a filesystem path or an API call (the XS-1 lesson: names become paths);
- a source scan asserts no literal address list in the module;
- malformed payloads raise rather than returning a short roster (a silently truncated roster
  narrows every downstream cohort claim);
- transient HTTP errors retry with backoff, schema errors do not (the 429 that killed the XS-8
  sampler at startup);
- de-duplication across sources, deterministic ordering.

---

## 4. Step 2 — position snapshot collector

**What.** `scripts/data/fetch_positions.py`, modelled directly on `fetch_l2.py`. One sweep =
one `clearinghouseState` call per wallet → one parquet under `data/positions/YYYY-MM-DD/`.

**Schema (one row per wallet × coin held):**

```
ts_ms · wallet · coin · size (signed) · entry_price · position_value · unrealized_pnl
liquidation_price · leverage · margin_used · account_value · status
```

`status ∈ {ok, empty, failed}` — **a wallet that fails is written, not dropped.** XS-8 writes
only its OK rows, so `aggregate_l2` cannot distinguish "frequently unreachable" from "less
history", and a wallet that stops responding must not silently look like a wallet that closed
its positions. That distinction is the whole point here.

**Cadence.** Positions move on hours; 15 min is ample and 5 min is wasteful against a
rate-limited endpoint. Start at **15 min**, `--symbol-delay`-equivalent between wallets.

**Clock.** A systemd `--user` unit (`nat-position-sampler.service`) with `Restart=always` and
`StartLimitIntervalSec=0` **in `[Unit]`** — it is ignored in `[Service]`, which is why the
existing daemons never actually had their restart limiter disabled (fixed 2026-08-08).
Boot persistence via linger. **Every day not collected is permanently lost**, exactly as for
1 m candles (§7.1).

**Tests** (`tests/test_fetch_positions.py`), all offline via an injected fetcher:
- one wallet's failure does not abort the sweep; `ok/empty/failed/rejected` account for every
  requested wallet **exactly once** (assert the arithmetic);
- a failed wallet appears in the parquet with `status=failed`, not omitted;
- an empty account (no positions) is distinguishable from a failed one;
- signed size survives the round trip (a short must not become a long);
- `liquidation_price=None` is preserved as null, never coerced to 0 — a zero liquidation price
  would corrupt every downstream cascade-distance calculation;
- re-running is idempotent per `(ts_ms, wallet, coin)`;
- rate-limit delay is applied between wallets;
- truncation via `--max-wallets` is reported, never silent.

**Real smoke before commit:** one live sweep of ≤5 wallets; assert schema, non-empty, and that
a known short shows negative size.

---

## 5. Step 3 — cohort construction

**What.** `scripts/wallets/cohorts.py` — turn a snapshot history into cohorts, using **only
information available at time t**.

- `realised_pnl(wallet, t0, t1)` from the snapshot series (position value + uPnL deltas), with
  deposits/withdrawals **read from the venue ledger and subtracted** — see the amendment below;
- `rank_cohorts(as_of, lookback) -> {top: [...], bottom: [...]}` — ranked on the lookback
  window **ending strictly before `as_of`**;
- `cohort_net_positioning(snapshots, cohort, coin) -> signed notional`, normalised by cohort
  account value so one whale does not become the cohort.

**The causality rule.** A cohort ranked on a window that includes the evaluation period is the
A-2 error in new clothing — the combiner's weights were fitted three days *after* the window
they were scored on and that alone produced its result. Cohorts must be re-ranked walk-forward.

> **Amendment 2026-08-13 — flows are read, not inferred.** This step originally planned to infer
> deposits/withdrawals from the snapshot residual and flag large jumps as unattributable. Two
> things came out of the data. First, the residual `Δaccount_value − Δ uPnL` is not separable:
> across 20,065 real WP-2 intervals its 99th percentile is **0.43 of account value** and **6.6 %
> of intervals move account value by >2 % net of uPnL**, so flagging would file 6.6 % of the
> panel as unknown. Second — not known when this spec was written — the venue exposes
> `userNonFundingLedgerUpdates`, which returns deposits and transfers **explicitly**. So
> `WP-3` gained a Part B (`scripts/data/fetch_ledger.py`) and realised P&L became
> `Δaccount_value − Δ uPnL − net_perp_flow`, exact rather than flagged.
>
> The flagging discipline moved rather than disappeared: it now applies to **unrecognised ledger
> delta types**, which raise instead of contributing zero. That earned itself on the first
> universe backfill — six unknown types appeared, and the most common of them,
> `accountClassTransfer` (spot↔perp collateral), had **1,574 occurrences**. Zeroed silently it
> would have corrupted realised P&L for most wallets with no symptom. Three types
> (`rewardsClaim`, `borrowLend`, `accountActivationGas`) remain deliberately unresolved: their
> perp effect cannot be established until the position panel is long enough to reconcile an
> event against its `account_value` step, and a guessed sign is worse than a flag.

**Tests** (`tests/test_cohorts.py`):
- **the decisive one:** a wallet that is profitable *only after* `as_of` must not appear in the
  top cohort (leakage), with a control asserting an in-sample ranking *does* select it —
  without that control the leakage test proves nothing (the A-2 pattern);
- rank stability is *measured and reported*, not assumed: cohort-membership autocorrelation
  across rebalances, since failure mode 1 is "the cohort is not a cohort";
- deposits/withdrawals do not register as P&L;
- normalisation — doubling one wallet's size must not double the cohort signal;
- an empty or single-member cohort is refused, not scored.

---

## 6. Step 4 — `cohort_predictability` process

**What.** A registered `EvaluationProcess` (`contracts/process.md`) measuring
`IC(cohort_net_positioning; forward_return)` per coin × horizon, and MI against the same target.

**Reused wholesale — build none of this again:**
- `processes/targets.py` (PROC-17) for the target and its leakage set;
- `it_engine/null_calibration.py` (PROC-12) for the null — **permute the cohort labels**, i.e.
  ask whether *this* cohort beats an arbitrary same-size group of wallets. That is A-1's
  size-preserving-gate null, and it is the difference between a finding and a selection effect;
- `processes/fdr.py` (PROC-13) across the (cohort × coin × horizon) grid;
- `mi_stability._day_key` (PROC-4) for per-day folds and the `durable | non_durable` verdict;
- `alpha/walkforward_ic.py` (A-2) if any weights are fitted.

**Horizons:** 1 h / 4 h / 24 h. At 5-min bars these are 12/48/288, so **non-overlapping
sampling is mandatory** — A-2's first run printed IC 0.39–0.46, *higher* than the claim it was
auditing, purely because consecutive observations shared 59/60 of their forward window.

**Tests** (`tests/test_cohort_predictability.py`):
- planted: a cohort whose positioning drives forward returns is recovered, with the correct
  sign;
- **mirror:** random cohorts over the same wallets report nothing;
- an intermittent edge reports `frac_days_informative ≈ 0.5` and `non_durable`;
- overlapping vs non-overlapping sampling is asserted to differ, so the trap cannot silently
  return;
- thresholds imported (`load_null_config`, `DEFAULT_FDR_ALPHA`), no literals.

---

## 7. Step 5 — the pre-registered study

**Criteria committed to git BEFORE the run** (§4.9 discipline; XS-6's criteria were committed in
`f3eea78` before its run and that is the standard):

- (a) pooled IC > 0.05 at ≥1 horizon;
- (b) `frac_days_informative` ≥ 0.55;
- (c) no single day > 30 % of the |IC| mass;
- (d) survives BH-FDR across the full grid at q ≤ 0.05;
- (e) **beats a same-size random-cohort null** (z ≥ 3) — not merely beats zero;
- (f) cohort rank half-life exceeds the rebalance cadence (else it is churn, per `XS-4`);
- (g) verdict sign stable across the top-10 % / top-25 % cohort definitions.

**Minimum sample before the study runs at all.** Positioning at a 24 h horizon gives ~1
independent observation per day per coin. `n ∝ 1/SR²` (XS-9's arithmetic) implies **≥ 90 days**
for any durable verdict. Running it at two weeks would produce exactly the kind of suggestive
number that died three times this week. **Collect first; the study is scheduled, not immediate.**

---

## 8. What this spec deliberately excludes

No trading rule, no algorithm, no capital path. A surviving finding enters the maturity ladder
at DISCOVERED and goes through the same gates as anything else — no live capital before G8 and a
healthy kill-switch. Liquidation-cascade prediction (`LF3`) and positioning-extreme signals
(`LF2`) become *buildable* once step 2 has run for a quarter, but they are separate units with
their own specs.

## 9. Honest risks

- **The data is public.** This is an underused asymmetry, not a private one; anyone can run it.
- **Rate limits.** The same REST endpoint threw 429s during the XS-8 install; the roster may
  need to be capped and rotated, and that cap must be reported (§ silent truncation).
- **Survivorship in the roster.** Wallets that blew up stop appearing. A cohort ranked on
  survivors is biased upward, and the collector's `status=failed` rows are what make that
  measurable rather than invisible.
- **Reflexivity.** If enough participants trade the same public positioning, the edge decays —
  `PROC-10`'s predictability half-life is the instrument for that.
- **The prior should be sceptical.** Eight consecutive studies have come back negative, and the
  H1–H6 whale confirmations belong to the era the Q4 gate invalidated. This spec is a way to
  find out, not a reason to expect a yes.
