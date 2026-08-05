# Orthogonality sweep — do the eight axes survive a holdout?

**Study date:** 2026-08-05 · **Data window:** 2026-05-18 → 2026-08-04 · **Status:** point-in-time
result, no capital implication.
**Driver:** `scripts/exploration/orthogonality_sweep.py` (uses PROC-15 `residualize` unchanged).
**Artifact:** `reports/orthogonality_sweep.json`.
**Supersedes:** the "Orthogonalization does not survive a holdout — first datum" bullet in
[`FINDINGS.md`](FINDINGS.md) §5, whose headline numbers are now known to be a tail day.

---

## The question

[`FINDINGS.md`](FINDINGS.md) §1 classifies the feature set into **eight independent signal axes**,
and [`../specs/maker_system.md`](../specs/maker_system.md) §2 turns that into a contract — *one
representative per orthogonal axis*. Both are enforced by correlation dedup on **full-sample**
statistics, i.e. a backward-looking measurement used as a forward-looking claim.

PROC-15's first run (BTC 2026-08-04) appeared to expose that: residualizing `imbalance_qty_l5`
against `imbalance_qty_l1` with a prefix-fitted β left **0.192** correlation on the holdout, with β
drifting +0.8148 → +0.7290 inside one day. This sweep replicates that measurement across every
available day × symbol.

## Method

Per (day, symbol): fit `res_f = f − β'Z` on the leading 70 % of rows with `Z = imbalance_qty_l1`,
then measure `|corr(res, Z)|` on the untouched 30 %. Prefix correlation is zero by OLS
construction and is only a sanity check; the holdout number is the entire content.

`flow_vwap_deviation` was carried as a **control** — §1 calls it a distinct (mean-reverting) axis,
so a sound method should leave it clean while redundant cousins drift. A sweep where everything
looks non-orthogonal is a broken method, not a finding.

**Coverage:** 152 of 216 episodes over **51 days × BTC/ETH/SOL**. The 64 missing are the known
data gaps (38 symbol-days absent, 23 below the 50 k-tick floor, 3 from a malformed
`2026-05-12-clean` directory) — §7's continuity problem, not method failure.

## Result

| target | median \|corr\| | p90 | frac > 0.10 | R²_fit | median β drift |
|---|---|---|---|---|---|
| `raw_bid_depth_5` | 0.098 | 0.279 | 0.50 | 0.427 | 0.119 |
| `raw_ask_depth_5` | 0.097 | 0.303 | 0.49 | 0.421 | 0.121 |
| `micro_queue_position_bid` | 0.096 | 0.520 | 0.49 | 0.048 | 0.377 |
| **`flow_vwap_deviation` — CONTROL** | **0.093** | 0.176 | 0.46 | 0.078 | 0.329 |
| `imbalance_orders_l5` | 0.083 | 0.242 | 0.44 | 0.789 | 0.052 |
| `imbalance_depth_weighted` | 0.079 | 0.219 | 0.40 | 0.619 | 0.056 |
| `imbalance_qty_l10` | 0.079 | 0.218 | 0.40 | 0.613 | 0.056 |
| `cross_obi_mean` | 0.076 | 0.170 | 0.38 | 0.250 | 0.119 |
| `imbalance_qty_l5` | 0.074 | 0.196 | 0.34 | 0.788 | 0.036 |
| `imbalance_notional_l5` | 0.074 | 0.196 | 0.34 | 0.788 | 0.036 |
| `micro_obi_velocity` | 0.056 | 0.146 | 0.23 | 0.046 | 0.262 |
| `flow_aggressor_ratio_5s` | 0.029 | 0.091 | 0.07 | 0.021 | 0.208 |
| `ent_permutation_imbalance_16` | 0.021 | 0.054 | 0.01 | 0.030 | 0.120 |

`imbalance_qty_l5` holdout distribution: p10 0.017 · **median 0.074** · p90 0.196 · max 0.564.

## What it says

1. **The single-day datum was a tail, and the earlier reading was wrong.** BTC 2026-08-04's 0.192
   sits at the **89th percentile** of the 152-episode distribution. The "β drifts ~10 % in a day"
   claim is a BTC median of **2.5 %**, and 3.6–5.6 % pooled across the imbalance cousins.
2. **The control refutes the cousin-specific story.** `flow_vwap_deviation` — the axis §1 calls
   independent — shows a *higher* median residual correlation (0.093) than the supposedly
   redundant `imbalance_qty_l5` (0.074). The leftover correlation is a general non-stationarity
   effect across the 70/30 split, not evidence that the imbalance block is inseparable.
   **This is the load-bearing part of the study:** without the control the same numbers would
   have read as "orthogonality is shaky everywhere" and been wrong in a way nothing downstream
   could catch — the shape of the VIP9 and five-winners failures.
3. **The axis contract mostly survives**, and two axes are now empirically clean rather than
   asserted: `ent_permutation_imbalance_16` (0.021, 1 % of episodes above 0.10) and
   `flow_aggressor_ratio_5s` (0.029, 7 %) separate from book pressure on essentially every day.
4. **One axis should be retired.** `raw_bid_depth_5` / `raw_ask_depth_5` are the worst pairs
   (0.098, half the episodes above 0.10, p90 ≈ 0.29). This is mechanical — imbalance is *built*
   from those depths — so "raw depth asymmetry" is not a distinct axis from "book imbalance" and
   the §1 list should stop treating it as one.
5. **The tail deserves respect.** p90 of 0.17–0.30 and a third to a half of episodes above 0.10
   for the depth/imbalance block mean decorrelation-based sizing should assume less
   diversification than the full-sample number implies — the same lesson §4.9 taught about
   day-consistency, in a different guise.

## Limits

One linear method (OLS), one conditioning variable, one split geometry (70/30, no rolling re-fit),
and a sample shaped by the §7 data gaps. Nothing here measures predictive value — only whether the
*separation* between features holds out of sample. A rolling-β variant and a multi-conditioner
version (residualize against the whole selected set, per PROC-3) are the obvious extensions.
