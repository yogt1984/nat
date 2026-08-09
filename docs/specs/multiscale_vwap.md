# Multi-scale VWAP — measure offline, migrate what earns its place

**Status:** SPEC (2026-08-09). **Owner:** Onat.
**Guardrail this exists to satisfy:** *"Plan before any feature-vector / schema change — it
ripples to Parquet and every reader."*
**Empirical basis:** `research/FINDINGS.md` §1 (`flow_vwap_deviation` IC −0.29/−0.21/−0.19 @1 s),
§5 (band capture at k ≈ 2.0–2.5, OU τ½ 5–7 s, IC in the 0.005–0.1 Hz band), §5 PROC-20
(0/330 band cells survive — starved of events, not absent), §7.10 (B-5a).
**Related:** `LF7` (`research/new/vwap_sd_channel.txt`), `PROC-20` (`processes/persistence_stats.py`).

---

## 0. The question, and why the cheap order matters

The ingestor publishes exactly one VWAP: a **5-second** trade VWAP plus its deviation. LF7's
channel needs an **hour-scale** one. PROC-20 found band capture concentrating at k ≈ 2.0–2.5 on
hour-scale windows using a *bar-derived* midline, because the tick-level slow VWAP does not
exist as a feature.

The obvious move — emit 5 m / 10 m / 15 m / 1 h / 6 h / 12 h continuously — is a **12-column
feature-vector change** (236 → 248) that ripples to `to_vec()`, `names_all()`, `count_all()`,
`output/schema.rs` and every Parquet reader, and cannot be undone cheaply once files exist.

**We do not know that any slow window carries information.** §1 measured the *fast* deviation
only. So the order is:

> **Measure all six windows offline from `data/trades/`, then migrate only the windows that
> earn their column.**

If none carry information, the migration never happens and the cost was one afternoon. If two
do, twelve columns become four. That asymmetry is the whole argument.

**The offline data exists:** `data/trades/` holds **44 days, 1,193 files, 203 MB**, schema
`timestamp_ns · symbol · tid · price · size · is_buy` — every field a trade VWAP needs.

---

## 1. Phase A — offline measurement (no schema change)

### A1. `vwap_multiscale` transform

`scripts/features/vwap_multiscale.py` — bucketed accumulators over raw trades.

**Design: 1-minute buckets, not raw scans.** VWAP is a ratio of two sums and sums decompose:

```
vwap(N minutes) = Σ notional over the last N buckets / Σ volume over the last N buckets
bucket = (Σ price·size, Σ size)      # 16 bytes
```

720 buckets covers 12 h in **~12 KB per symbol**, against ~17 MB of raw trades, and reading any
window is O(N/60) instead of an O(n) scan with allocation. Exact for any window that is a whole
number of buckets — 5 m/10 m/15 m/1 h/6 h/12 h all are. This is the same structure Phase B would
put in Rust, so measuring it here also **validates the production design** before it is written.

**Output columns**, per window `w ∈ {5m, 10m, 15m, 1h, 6h, 12h}`:

```
vwap_{w}            volume-weighted price over the window
vwap_dev_{w}        (price − vwap) / vwap        ← intuitive sign, see §3
vwap_dev_z_{w}      deviation / rolling σ of deviation   (the k of LF7's k·σ)
```

**Sign convention.** The shipped `flow_vwap_deviation` is `(vwap − price)/price` — **inverted**
relative to how the quantity is normally read, which is why §1's −0.29 means *price below VWAP
predicts further decline*. New columns use `(price − vwap)/vwap`. Two conventions in one row is
worse than one wrong one, so the ingestor's existing column is left alone and the discrepancy
is documented at both sites rather than silently propagated.

**Tests** (`tests/test_vwap_multiscale.py`):
- **arithmetic**: planted trades with known prices/sizes give the hand-computed VWAP for each
  window — the one test that must be exact, since everything downstream is a ratio of it;
- **bucket equivalence**: bucketed VWAP equals a brute-force scan over the same trades to
  1e-9, on random trade streams — this is the claim the production design rests on;
- **boundary**: a trade exactly on a bucket edge is counted once, in one bucket;
- **warm-up**: a window with less than its full span reports **NaN**, never a partial-window
  VWAP silently labelled as the full one;
- **gaps**: a 6-hour hole (which §7 guarantees) yields NaN for the windows spanning it, not a
  VWAP computed from whatever survived;
- **empty buckets** (no trades in a minute) contribute 0 notional AND 0 volume — never a
  divide-by-zero, never an implicit forward-fill;
- **sign**: price above VWAP ⇒ `vwap_dev > 0`, asserted against the *inverted* legacy column so
  the two conventions are pinned in a test rather than in prose;
- **determinism** and no input mutation.

**Real-data smoke before commit:** one day of BTC from `data/trades/`; assert the 5 m column
tracks the ingestor's `flow_vwap_5s` in *shape* (correlation, after sign correction), which
cross-validates the offline path against the production one.

### A2. The study

`scripts/exploration/vwap_multiscale_study.py`, run over the full 44-day archive × BTC/ETH/SOL.

**What it measures**, reusing existing machinery rather than new statistics:
- **information**: `PROC-4` (`mi_stability`) per window — is the deviation informative, and is
  it *durable across days* rather than a pooled average;
- **band structure**: `PROC-20` (`persistence_stats`) per window — markout by k, time-to-revert,
  event counts per day. PROC-20's verdict was that the band effect is starved of events at
  ~1.5/day; six windows on the same tape multiply the event count, which is the actual test;
- **redundancy**: `PROC-15` (`residualize`) — does `vwap_dev_1h` add anything beyond
  `vwap_dev_15m`, and beyond `imbalance_qty_l1`? Six correlated windows are one axis wearing six
  hats until shown otherwise (the §5 fat-tail result applies).

**Pre-registered acceptance — a window earns a column iff:**

- (a) its deviation is informative against a permutation null (PROC-12, z ≥ 3) at ≥1 horizon;
- (b) `frac_days_informative` ≥ 0.55 (PROC-4) — §4.9's binding failure was day-consistency;
- (c) it survives BH-FDR across the (window × horizon × symbol) grid (PROC-13);
- (d) it is **not redundant**: holdout `|corr(residual, faster_window)| < 0.5` after PROC-15,
  i.e. it says something the cheaper column does not;
- (e) its band-touch **event rate** is high enough to make a durable verdict reachable —
  PROC-20's ~1.5/day is the counter-example to avoid.

Criteria are committed to git **before** the run (the XS-6 standard, `f3eea78`).

---

## 2. Phase B — migration, only for windows that passed

**Do not start Phase B unless ≥1 window passes A2.** If none do, this spec closes with a
FINDINGS entry and the ingestor is untouched — a good outcome, cheaply obtained.

### B1. Plan the schema change (its own review)

Per the guardrail, the feature-vector change gets a written ripple analysis before code:
`Features` struct → `to_vec()` / `names_all()` / `count_all()` → `output/schema.rs` → every
reader (`cluster_pipeline/loader.py`, the process runner, the algorithm registry, the dashboard).
**Readers must tolerate the columns' absence**, since 44 days of existing Parquet will not have
them — the count is not a constant to assert against.

### B2. Rust implementation

`ing-features`: a `VwapRing` of 1-minute buckets on `MarketState`, mirroring A1's design so the
Python tests double as the specification. **Not** a widened `TradeBuffer`: keeping 12 h of raw
trades and running `trades_in_window` (an O(n) scan *with allocation*) six times per symbol at
10 Hz would blow the 80 ms/tick p99 budget, which is a correctness problem for the emission
loop and not merely a slow feature.

**Tests:** a Rust unit test asserting the ring reproduces a brute-force scan (the same property
A1 tests in Python, so the two implementations are pinned to each other); a NaN-until-warm test;
and a benchmark asserting the per-tick cost stays inside the emission budget.

### B3. Warm-up and restart

A 12 h window needs 12 h of trades. On restart those columns are NaN for half a day, and §7's
uptime record makes that routine rather than exceptional. NaN-padding follows the existing
optional-category pattern (`count_all()` with explicit NaN), and the warm-up state is
observable — a column that is NaN because it is cold must be distinguishable from one that is
NaN because it is broken.

---

## 3. What this spec deliberately does not do

- **No trading rule.** A surviving window is a *feature*; LF7 remains the algorithm, gated by A4
  as before.
- **No change to `flow_vwap_5s` / `flow_vwap_deviation`.** They ship, they are measured, and
  re-signing them would invalidate §1's IC record. The inversion is documented, not fixed.
- **No new statistics.** Every verdict comes from PROC-4/12/13/15/20, already built and tested.

## 4. Honest risks

- **The six windows are probably one axis.** They are nested sums of the same tape, so
  correlation will be high; criterion (d) exists because "six new features" is the likely
  illusion, and the honest outcome may be one or two columns.
- **VWAP is a trade-weighted anchor**, so it inherits the trade feed's gaps. §7's continuity
  problem propagates directly into the slow windows, and the 44-day archive is not gap-free.
- **Bounce.** Roll (1984): bid-ask bounce alone produces negative return autocovariance with
  amplitude ≈ the spread. If a window's "oscillation" has amplitude comparable to the spread,
  it is bounce and there is no edge by construction. The amplitude-to-spread ratio — not the
  reversion — is the quantity that decides tradeability, and A2 must report it.
- **The prior should be sceptical.** PROC-20 already tested bar-derived band structure over 62
  days and found 0/330 cells surviving. This spec's bet is that the *tick-level* slow VWAP is a
  better anchor and that more windows mean more events — not that the effect was missed.
