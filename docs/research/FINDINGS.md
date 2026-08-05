# NAT — Consolidated Empirical Findings

**What this is:** the single consolidated record of everything NAT has *measured* — about the data,
the market, and its own code. Merged from the Tier-A finding reports (per
[`archive/in_progress/INDEX.md`](../archive/in_progress/INDEX.md) classification); sources preserved in
[`archive/`](../archive/) with full provenance (table at the bottom). Findings are point-in-time:
each block states its test window. Nothing here is a plan — plans live in [`PLAN.md`](../PLAN.md) /
[`TASKS.md`](../TASKS.md).

*Consolidated 2026-07-25. Companion catalogues (living, not merged):
[`ALGORITHMS.md`](ALGORITHMS.md) (implemented algos + OOS), [`../../FEATURES.md`](../../FEATURES.md)
(feature manifest), [`INSTITUTIONAL_ALGORITHMS.md`](INSTITUTIONAL_ALGORITHMS.md) (gap audit).*

---

## 0. Synthesis — what the record collectively says

1. **A large, universal, structurally-validated signal exists.** Order-book imbalance carries
   IC ≈ 0.45 at 1–5 s on all three symbols, 24/7, in both vol regimes, with bootstrap CI width
   ~0.02. It is not noise and not symbol-specific (§1, §3).
2. **The signal cannot be monetized by the fill models tested.** Taker: the 1–5 s move (0.5–2 bps)
   is smaller than taker cost (11 bps RT). Maker: conditioning on the directionally-correct
   mid-cross fill collapses IC from ~0.45 to ~0.03/−0.06/−0.03 — adverse selection is structural,
   not a tuning problem (§2). **This is the project's central result and its binding research
   question** (gate Q5).
3. **Most algorithm "failures" were horizon mismatches, not logic failures.** 10 of 12 Tier-3
   losers are tick-intrinsic logic evaluated at 100 min — 12–200× their native half-life (§4.3).
   The salvage path is maker-side execution (microprice, MM, toxicity gating), not faster taker.
4. **What survives OOS is a diversified 4–5 algo book at MF/macro horizons** — jump_detector,
   3f_liquidity, funding_reversion, optimal_entry (+ surprise_signal ex-BTC) spanning 4 logic
   families with max pairwise ρ 0.449 and a ~0.00 jump×funding pair (§4.1–4.2).
5. **Regime gating measurably works.** `ent_book_shape` lifts imbalance IC by +22 % (low-entropy
   quintile; 0.45 → 0.55–0.67 in the Spannung replication), and the hierarchical combiner's
   direction-gated L2 is the first architecture that structurally addresses adverse selection (§5).
6. **Data continuity is the operational binding constraint.** 20 of 54 calendar days missing
   (~37 %), a 6-day zombie-ingestor gap with no error logs, 56–82 dead (all-NaN) feature columns,
   and a longest clean streak of 12 days (§7). Every downstream verdict (30-day OOS, HMM, full
   confidence) is data-gated.
7. **Sample-size arithmetic is unforgiving.** Convolver trap events need ~39 k candles minimum;
   across-regime validation needs 6–24 months; hourly-pattern discovery is infeasible (~5 years of
   data) (§6). Claims must be sized to the data actually in hand.

---

## 1. The signal — feature information content

*Source: full IC scan 2026-06-09 (3 days 2026-05-19→21, BTC/ETH/SOL, 2.17 M ticks/symbol @100 ms,
Spearman rank IC subsampled every ~10 s, 21,708 points, 6 horizons, p<0.01).*

- **207 live features scanned** (48 all-NaN at scan time). Classification (BTC/ETH/SOL):
  dir_fast 29/29/28 · dir_medium 11/11/10 · dir_slow 7/3/14 · volatility 68/75/68 ·
  weak 11/9/7 · no_signal 33/32/32 · all_nan 48/48/48.
- **Order-book imbalance dominates** directional prediction at 1–5 s, IC 0.40–0.47 on all three
  symbols — universal, not symbol-specific. Top features (BTC/ETH/SOL @ peak):
  `imbalance_qty_l1` +0.453/+0.447/+0.466 · `imbalance_qty_l5` +0.456/+0.431/+0.453 ·
  `imbalance_notional_l5` +0.456/+0.431/+0.453 · `imbalance_orders_l5` +0.435/+0.438/+0.455 ·
  `imbalance_depth_weighted` +0.434/+0.410/+0.425 · `raw_ask_depth_5` −0.424/−0.404/−0.415 ·
  `raw_bid_depth_5` +0.420/+0.405/+0.406 · `cross_obi_mean` +0.354/+0.334/+0.331 ·
  `micro_queue_position_bid` +0.307/+0.282/+0.291 · `flow_vwap_deviation` −0.287/−0.206/−0.188 (1 s) ·
  `micro_obi_velocity` +0.192/+0.181/+0.208 (1 s) · `ent_permutation_imbalance_16` −0.170/−0.173/−0.177 (1 s) ·
  `flow_aggressor_ratio_5s` +0.112/+0.109/+0.111.
- **Eight independent signal axes** (the 29 fast-directional features are heavily correlated):
  (1) book imbalance ~0.45 · (2) raw depth asymmetry ~0.42 · (3) cross-symbol imbalance ~0.35 ·
  (4) queue dynamics ~0.31 · (5) VWAP deviation (mean-reverting) ~0.25 · (6) OBI velocity ~0.19 ·
  (7) imbalance entropy ~0.17 · (8) aggressor flow ~0.11.
- **Decay / half-life** (peak directional IC, BTC/ETH/SOL): 1 s 0.447/0.447/0.466 → 5 s
  0.456/0.438/0.412 → 30 s 0.261/0.219/0.190 → 1 m 0.187/0.160/0.135 → 5 m 0.090/0.074/0.056 →
  15 m 0.056/0.031/0.036. Half-life ≈ 30 s BTC, ~20 s ETH, ~15 s SOL (thinner book decays faster).
  By 5 minutes, ~80 % of the signal is gone.
- **Direction and volatility are cleanly orthogonal:** imbalance features carry zero vol-IC and
  vol features zero directional IC, consistently across symbols. Top vol features @5 s (BTC/ETH/SOL):
  `hawkes_intensity` +0.345/+0.281/+0.247 · `vol_returns_5m` +0.328/+0.326/+0.293 ·
  `vol_parkinson_5m` +0.318/+0.312/+0.281 · `flow_count_30s` +0.318/+0.281/+0.253 ·
  `illiq_trade_count` +0.318/+0.286/+0.261 · `toxic_effective_spread` +0.291/+0.279/+0.254 ·
  `flow_intensity` +0.291 (best short-horizon vol predictor).
- **Slow directional features grow with horizon:** `raw_spread_bps` BTC 15 m +0.139 / ETH +0.179;
  `trend_ema_short` ETH 15 m −0.188; `regime_range_pos_24h` SOL 15 m −0.145;
  `regime_divergence_1h` 0.02 @1 s → 0.07 @15 m → **0.21 @100 min** (slow accumulation, not
  microstructure). VPIN and tick-entropy have **no** directional IC at any horizon (magnitude only).
- **Why the earlier 25-algorithm sweep failed** (ic_horizon_analysis): every algorithm was
  evaluated at 100-minute horizons on 5-min bars, destroying the 1–5 s signal (0.45 → <0.07
  residual). *"The features work — the horizon was wrong."*

## 2. The barrier — adverse selection (conditional IC)

*Source: ic_validation_report 2026-06-09 (31 dates 2026-04-19→06-04, 23 valid; mid-cross fill
within 50 ticks; `reports/ic_validation.json`).*

The central negative result, `imbalance_qty_l1` @5 s:

| Conditioning | BTC | ETH | SOL |
|---|---|---|---|
| Unconditional | +0.453 | +0.438 | +0.412 |
| Any fill event | +0.526 | +0.516 | +0.460 |
| **Buy fill (imb > 0)** | **+0.032** | **−0.061** | **−0.032** |
| **Sell fill (imb < 0)** | **−0.047** | **−0.027** | **−0.021** |

- **Mechanism:** signal says buy → place bid → the fill requires mid to *drop* to the bid (an
  adverse move) → at fill time the "up" prediction is already invalidated → post-fill IC ≈ 0.
  "Any-fill" IC *exceeds* unconditional (ticks near fills have higher |imbalance|), but directional
  conditioning eliminates the edge on all three symbols.
- **Fill prevalence:** buy fills 20.8 %/24.1 %/30.8 % of ticks (BTC/ETH/SOL); SOL's more frequent
  fills → more trades but worse PnL — *more fills = more adverse selection*.
- **Taker arithmetic** (ic_horizon): expected 1–5 s move ≈ 0.5–2 bps vs 11 bps RT taker cost —
  taker capture is arithmetically impossible; only maker execution can in principle capture it,
  and the table above is what naive maker fills do to the signal.

## 3. Robustness & drift

*Source: ic_validation_report 2026-06-09.*

- **Validation matrix:** BTC passes **7/8**, ETH **6/8**, SOL **6/8**. Passes: per-day mean IC
  (+0.43/+0.40/+0.35 vs >0.30), per-day std (0.06/0.10/0.07 <0.10), all intraday windows, low-vol
  IC, high-vol IC, bootstrap CI lower bound. Fails: worst-single-day ETH +0.04 / SOL +0.17
  (threshold >0.20); **temporal-OOS delta fails all three** (−0.17/−0.16/−0.17 vs <0.10).
- Per-day rolling IC (mean±std, 5 s): BTC +0.433±0.061 · ETH +0.400±0.098 · SOL +0.355±0.071.
  Feature rank ordering is preserved across all symbols — the hierarchy is structural.
- Intraday: present 24/7, no dead zones; weakest 12–16 UTC, strongest 04–08 UTC
  (BTC +0.384→+0.460 range).
- Vol regimes (median split): low-vol +0.458/+0.435/+0.391 vs high-vol +0.422/+0.394/+0.342 —
  works in both, slightly stronger in calm markets.
- Bootstrap 95 % CI (1000 resamples): `imbalance_qty_l1` BTC [0.442, 0.465] — CI width ~0.02.
- **Drift warning:** May 19–21 → Jun 2–4, 5 s IC drops +0.45→+0.29 (BTC) on all symbols, while
  1 s IC is stable or *improves* (BTC +0.447→+0.502). The signal's half-life is **shortening**;
  June spreads 0.2–0.3 bps vs 0.1 bps in May — faster price discovery in thinner books.

## 4. Algorithms out of sample

### 4.1 Winners (13-date OOS 2026-05-07→23, 100 min horizon, 1.61 bps RT, P20/P80 z-entry)

> ⚠️ **REFUTED 2026-07-30 — see §4.6.** Retained for provenance only. Every number below was
> produced at the wrong venue cost (1.61 bps Binance VIP9 vs Hyperliquid ~11 bps all-in) by a
> harness that applied one generic P20/P80 entry to every candidate instead of running each
> algorithm's own logic. At SSOT cost all five are deeply net-negative; all were REJECTED in
> the signal lifecycle on 2026-07-30.

*Source: algorithms_report 2026-06-12.*

| Algorithm | Total bps | BTC bps/tr · Sharpe | ETH | SOL | Note |
|---|---|---|---|---|---|
| jump_detector | +23,199 | +1.03 · 1.6 (54 % WR) | +6.47 · 6.2 (62 %) | +6.33 · 6.2 (69 %) | Lee–Mykland; 1,678 tr/sym |
| 3f_liquidity | +16,028 | +5.58 · 9.2 (62 %) | +7.83 · 7.8 (62 %) | +3.74 · 3.2 (62 %) | 50-min bars (non-standard) |
| funding_reversion | +14,459 | +0.26 · 0.4 (38 %) | +6.12 · 6.1 (54 %) | +2.24 · 1.7 (54 %) | funding z mean-reversion |
| optimal_entry | +13,679 | +0.90 · 1.1 (46 %) | +5.89 · 5.2 (62 %) | +1.37 · 1.0 (54 %) | SPRT/Kalman; known σ bug |
| surprise_signal | +3,505 | −4.78 · **−8.3** (15 %) | +2.85 · 3.1 (54 %) | +5.29 · 6.7 (46 %) | Tier-2: **deploy ETH/SOL only** |

- **Correlation structure:** max pairwise ρ 0.449 (3f × surprise, ETH) — the only pair >0.35;
  jump × funding ≈ 0.00 on all symbols (ideal blend). The 4 deployables span **4 logic families ×
  3 horizon bands** (event/MF, composite/MF, carry/macro, order-flow/tick).
- ML: `mean_reversion_detector` OOS AUC 0.577/0.564/0.552 (OOS/IS ≈ 0.78 — honest);
  `momentum_continuation` IS 0.65 → OOS 0.37–0.45 (**overfit**); `meta_labeling` OOS 0.44–0.48
  (its inputs `conc_hhi`, `whale_directional_agreement`, `regime_clarity` are 100 % NaN — K2-blocked);
  `regime_conditioned_lgbm` — RSM classifies only 37 % of bars.
- ⚠️ Census discrepancy (kept honestly): situation_analysis quotes 3f BTC Sharpe **12.1**;
  algorithms_report quotes **9.2** — different windows/bar choices; also 191-feature vs 236-feature
  censuses across the two reports. Treat per-report numbers with their own conditions.

### 4.2 Losers (same OOS window; all net-negative after 1.61 bps RT)

oi_divergence −1,721 · regime_gated −1,748 · entropy_momentum −2,600 · propagator −4,118 ·
hawkes_intensity −5,443 · trade_through −5,739 · weighted_ofi −6,183 · switching_ou −6,230 ·
vpin_regime −7,331 · kalman_imbalance −7,517 · bipower_jump −32,079 · spread_decomp −34,510 (bps).
Also failed elsewhere (situation_analysis): surprise_signal walk-forward Sharpe −6.23,
vol_gated_divergence −8.52, weighted_ofi rolling −14.6, sweep_taker −20.5, sweep_maker −5.7;
alpha screen: **551 features at 1-day horizon → 0 significant after FDR**.

### 4.3 The horizon-mismatch diagnosis

*Source: algorithm_classification 2026-06-12.*

**10 of the 12 Tier-3 failures are tick-intrinsic logic tested at 100 min** — 12–200× their native
half-life (imbalance IC 0.45 @1 s → 0.09 @5 m → 0.06 @15 m). The logic is real at its native
horizon (per §1) but not salvageable by faster *taker* retest (§2) — the salvage path is
**maker-side** (microprice quoting HF1, MM HF5, toxicity gating HF4). Retest candidates at 5–30 min:
propagator, switching_ou, bipower_jump (task Q2.7). Coverage gaps: **Daily row empty** (no
implementation), macro thin (funding_reversion only). Promotion rule derived: prefer unoccupied
(family × horizon) cells; same-cell additions require a correlation check.

### 4.4 Single-day gauntlet (2026-06-01, 4.7 h, 19 algos — regime-shift evidence, not a verdict)

*Source: gauntlet_analysis 2026-06-01 (3-day training, 1.61 bps RT).*

Top: hawkes_intensity +955.8 bps (26 trades, +36.8 bps/tr, 2/3 symbols) · propagator +482.2 (one
+1150 ETH print) · spread_decomp +456.3 (**3/3 symbols**) · optimal_entry +441.0 (3/3) ·
funding_reversion +272.9. Bottom: jump_detector −444.7 · bipower_jump −500.4 · trade_through
−1,101 · oi_divergence −1,151. **Three zero-trade algorithms** (convolver, entropy_momentum,
cascade_probability) — convolver's diagnosed cause: `bar_agg="mean"` dilutes a 60 s score spike
0.8 → ~0.13 under 5-min bars (fix: `bar_agg="max"`). Key observation: hawkes/spread_decomp were
Tier-3 in May but ranked #1/#3 on Jun-1 (lookahead-fix commits `1470d9c`/`5a9aa79`, regime change,
or single-day noise — unresolved). Single day: no statistical significance claimed.

### 4.5 VWAP-reversion × VPIN gate (2026-07-29 — GAP-01/03, 58-day walk-forward, SSOT costs)

*Source: `paper_trader_generic --cost-mode config`, `toxic_vwap_reversion` (GAP-03, VPIN-gated
fade) vs `vwap_reversion` (GAP-01, ungated fade). Both trade their signed `_signal`, so the only
difference is the gate. Costs from the unified SSOT (COST-1/2/3).*

Net bps / Sharpe (58 days, BTC/ETH/SOL):

| Algorithm | BTC | ETH | SOL |
|---|---|---|---|
| `vwap_reversion` (ungated) | −25566 / −15.1 | −41052 / −13.8 | −29929 / −8.3 |
| `toxic_vwap_reversion` (gated) | −30958 / −8.1 | −32701 / −8.1 | −22651 / −6.1 |

1. **Neither survives taker costs** — deeply net-negative on all three symbols (~50 trades/day ×
   taker fees bleed the tick-derived edge dry). Reconfirms Spannung Phase B with the fixed cost model.
2. **The VPIN gate is directionally validated** — `toxic_vwap_reversion` beats the ungated baseline on
   **Sharpe on all three symbols** (BTC −8.1 vs −15.1; ETH −8.1 vs −13.8; SOL −6.1 vs −8.3) and on
   total bps for ETH/SOL: the gate removes the adverse-selection loss tail exactly as theorised — but
   the improvement is far too small to overcome taker fees. "Directionally correct but insufficient."
3. **Tick-level IC ≠ tradeable edge:** ungated IC@50t (+0.1215) actually *exceeds* gated (+0.107); the
   gate's value is a net-of-cost / Sharpe effect, invisible in raw IC.

**Implication:** the taker path for VWAP-reversion is dead. The only viable route is maker / zero-fee
execution — GAP-04 `microprice_maker_sim` (HF1) and GAP-05 Q2.6 OU-Kalman — which attack the
fill-conditional collapse instead of paying taker fees. This is the Q5 decision structure.

### 4.6 Q4 alpha-skeptic kill gate (2026-07-30) — **the §4.1 winners table is refuted**

*Five parallel adversarial passes (alpha-skeptic agents), one per "deployable winner", run on data
in hand before the ≥90-day spend, exactly as `docs/TASKS.md` Q4 sequences it. **All five verdicts:
KILL.** These supersede §4.1's Tier-1/Tier-2 claims and the "4 deployable" experiment-status
summary. Per-run evidence: `.claude/agent-memory/alpha-skeptic/` verdict files.*

| Algorithm | Verdict | Lethal findings (full evidence in the per-run records) |
|---|---|---|
| `3f_liquidity` | **KILL** | §4.1's Sharpe 9.2/7.8 was priced at **1.61 bps RT (Binance VIP9)**, not Hyperliquid (~11 bps all-in). At honest cost on 57 OOS dates (05-07→07-30): BTC **−11.1**, ETH −6.4, SOL −8.3. Original `mf_*` inputs no longer exist in the codebase; the registered implementation is a different construction that is negative even at 1.61 bps. `reports/rolling_3f_liquidity.json` (Jun-9) already showed ≈ −9.5 and sat unreconciled while the signal was promoted to VALIDATED (Jun-15). |
| `funding_reversion` | **KILL** | Same cost tier bug; at realistic cost on 59–62 dates: BTC **−6.1**, ETH −3.4, SOL −3.1. P20/P80 harness enters on ~45% of bars with 95%-overlapping 100-min windows → n_eff ≈ 84, so the original claim was never significant. Funding was longs-crowded for the entire data history (no shorts-crowded stretch exists to test symmetry). The backtest never charged funding carry (gate column never created). |
| `surprise_signal` | **KILL** | `paper_trader_surprise.py` hardcodes `binance_vip9_rt_bps()`; at real cost all 3 symbols negative on the original window. Extending 20→~60 days *at the cheap cost* alone: ETH 3.52→0.98, SOL 5.27→**0.19** (fails G4 0.5). **87.6% of ETH's original edge came from one day** (2026-05-23). BTC direction reproducibly wrong → "deploy ETH/SOL only" was post-hoc symbol selection. Look-ahead attack REFUTED — the entropy path is causal; failure is economic. Never entered the lifecycle DB. |
| `optimal_entry` | **KILL** | §4.1's harness never ran the SPRT/Kalman logic — it applied a generic P20/P80 z-entry to every candidate (identical 1,678 trade counts across algos betray the fixed schedule). `run_batch()` hardcodes `sigma_process=0.01` (documented as a known bug 2026-06-12, still live) → backtest is blind to the parameter, backtest/live parity broken. Platform's own `nat oos --window 60d`: OOS Sharpe **−4.5 to −6.3**, maxDD 86–119%, DSR 0.03–0.06 — fails every G4 criterion on stored data. Sole input is `imbalance_qty_l1`, the documented 0.45→0.03 fill-collapse feature. |
| `jump_detector` | **KILL** | Platform's own `nat oos --window 90d`: OOS Sharpe BTC **−3.0** / ETH −4.5 / SOL −5.2, DSR 0.036–0.095, maxDD 58–84× the gate — fails G4 on stored data at the *optimistic* cost. July rerun at SSOT cost: Sharpe −18.4/−8.5/−6.3; BTC gross edge negative **before** cost. The c=3.0 threshold at 10 Hz fires **~13,900×/day** (EVT-correct critical value ≈ 7.2, per v2's own docstring) with a symmetric up/down split — a generic large-return filter, not jump detection. `run_batch` embeds the current tick's return in its own bipower denominator (self-masking; conservative direction, but backtest≠live — v1's parity test can't see it, only v2 has rtol=1e-9 parity). 5-second native mechanism traded at 100-min holds (the §4.3 failure mode). Effective spread at jump ticks 1.57× baseline — costs are elevated exactly where it trades. |

**Systemic root causes (each is a platform defect, not a per-algorithm accident):**

1. **Wrong-venue cost default** — the sweep/eval harnesses default to `cost_mode="binance_vip9"`
   (1.61 bps RT): `paper_trader_daily.py`, `cli/oos.py`, `cli/gauntlet.py`, `overnight_sweep.py`,
   `mf_liquidity_backtest.py`, `mf_hypothesis_suite.py`, `it_multiday.py`; `paper_trader_surprise.py`
   hardcodes the VIP9 helper call. NAT trades Hyperliquid (7 bps RT taker + ~2 bps/side slippage).
   **Every historical backtest number produced through these paths is invalid until re-run.** The
   COST-3 CI guard scans numeric literals, so wrong-preset *function calls* pass it.
2. **Sweep harness didn't run the algorithms** — the §4.1 table came from one generic P20/P80
   z-entry over each candidate's primary column, not from each algorithm's own decision logic.
   Winners selected from ~26 candidates with **no algorithm-level FDR/DSR** (feature-level FDR only).
3. **Governance**: two signals were promoted to VALIDATED (2026-06-15) while contradicting artifacts
   existed in-repo, and one "deploy" recommendation lived only in prose. The lifecycle DB and the
   empirical record drifted apart with no reconciliation step.

**Verdict for Q5 planning:** the deployable-winners tier is **empty** as of 2026-07-30 — 5/5 KILL,
with the wrong-venue cost tier confirmed on all five. The honest survivors of the record are the
*mechanism* findings (VPIN gate directionality §4.5, conditional-IC barrier §2, maker-path economics
GAP-04) — not any shipped taker algorithm. The path to Q5's conditional-IC > 0.15 runs through maker
execution + the PROC discovery layer, exactly as §4.5 concluded. Two named revival candidates exist,
both as NEW signals (REJECTED is terminal): `jump_detector_v2` (EVT threshold ≈7.2 + exact
step/batch parity, not yet wired into the economics harness) and the GAP-04/HF1 maker path.

Follow-ups filed: **COST-4** (flip every eval-harness default from `binance_vip9` to the Hyperliquid
SSOT: `paper_trader_daily.py`, `cli/oos.py`, `cli/gauntlet.py`, `overnight_sweep.py`,
`mf_liquidity_backtest.py`, `mf_hypothesis_suite.py`, `it_multiday.py`, `paper_trader_surprise.py`),
**COST-5** (harden the CI guard to catch wrong-preset *calls*, not just literals), **BUG-4**
(`optimal_entry` `sigma_process` hardcode → backtest/live parity), **BUG-5** (`jump_detector`
`run_batch` self-masking bipower + weak parity test), **REV-1** (purge/re-run every §4.1-derived
number still cited at SSOT costs), **QA-JD2** (wire `jump_detector_v2` into `paper_trader_generic`
and re-run July at SSOT cost — the cheapest test of whether the Lee-Mykland family has any life).

**COST-4/5 landed 2026-07-30** (commit d9f3c1c): all harness defaults now resolve to the SSOT
(~11 bps RT); VIP9 is explicit-opt-in only; recurrence blocked by `test_cost_defaults.py` + the
extended CI guard. Bonus fix: `overnight_sweep` printed `--cost-mode` but ignored it.

**QA-JD2 answered 2026-07-30: the taker-path Lee-Mykland family is dead, v2 included.**
`jump_detector_v2` (EVT threshold ≈7.2, exact step/batch parity) wired into
`paper_trader_generic` and run side-by-side with v1 over the full 59-day window at SSOT cost:

| net bps / Sharpe | BTC | ETH | SOL |
|---|---|---|---|
| `jump_detector_v2` | −43,928 / −5.3 | −36,709 / −3.3 | −41,713 / −4.1 |
| `jump_detector` (v1) | −48,854 / −5.3 | −42,409 / −3.5 | −35,202 / −3.1 |

Indistinguishable — v1's threshold miscalibration was *not* the binding failure; at taker cost
the reversion signal has no edge under the standard harness regardless of detection quality.
*Caveat:* this tests v2's continuous REV signal through the generic P20/P80 bar harness, not an
event-native maker execution at detected jumps — that residual idea belongs to the GAP-04 maker
line, not to any taker deployment. Lee-Mykland revival via taker: closed.

### 4.7 First queue-value maker replay (2026-07-30 — A4, HF1 anchor; BTC, ~29 h)

*Source: `scripts/execution/queue_value.py` (`replay_from_frame`, conservative FIFO queue sim),
BTC 2026-07-29→30, 1.06 M ticks, 5,288 postings/side at 200-tick cadence, 300-tick horizon,
50-tick markout, costs via `load_costs()`.*

| Side | fill rate | capture (½spread + rebate) | E[adverse \| fill] | **EV / posting** |
|---|---|---|---|---|
| BID | 0.583 | 0.078 + 0.20 = 0.278 bps | +0.217 bps | **+0.036 bps** |
| ASK | 0.580 | 0.078 + 0.20 = 0.278 bps | +0.256 bps | **+0.013 bps** |

**First marginally-+EV execution result on the platform** — and the structure is exactly what
§4.5 predicted: the half-spread is nearly worthless (0.08 bps at BTC's tight touch); the maker
rebate (0.20 bps) covers adverse selection with a sliver left. **Not a validated edge:** one
symbol, ~1 day, and the replay's queue/volume inputs are PROXIES (exec volume from the rolling
1 s flow window split by the 5 s aggressor ratio — likely overstates per-tick volume and hence
fill rate; queue-ahead = 0.4·depth_5, an assumption; naive 5 s markout). The conservative
cancel rule (cancellations never advance the queue) biases the other way. Next (HF5): condition
postings on the HF1 microprice center (`alg_mp_dev_bps`, IC@50t +0.14/+0.24 gated) — quote the
side the fair value favors, skew away from adverse flow — and test whether EV widens beyond
the rebate sliver. Sim-only; no live path exists.

### 4.8 Avellaneda–Stoikov composition + first calibrated episode (2026-07-30 — HF5, sim-first)

*Source: `scripts/execution/avellaneda_stoikov.py` (ASQuoter/ASSim/calibrate_kappa), BTC
2026-07-29→30, 1.07 M ticks; HF1 center (`alg_mp_dev_ema`), HF4 VPIN gate (70 % open),
κ calibrated from crossing rates at 5 offsets, γ=0.02, τ=100 ticks, q_max=5, seed 42.*

**What the episode validated:** the composition works mechanically — inventory hard-capped at
±5 with mean ≈ 0, gate applied, reservation skew active. The **HF1 microprice center cut the
terminal liquidation cost ~40 %** vs a mid-centered control (10.6 vs 17.7 bps) by leaning
inventory away from adverse drift (mean q −0.56 vs +0.02).

**What the episode refuted (about the instrument itself):** the exogenous λ(d) fill model is
**not a valid economics instrument at these parameters.** The crossing-based calibration
(fill rates 0.56→0.43 across 0.05→0.8 bps offsets ⇒ A=0.55/tick, κ=0.32/bps) measures *price
volatility*, not queue consumption — at A=0.55 per tick the sim "fills" 291 k times and prints
≈ +950 k bps for both centers: pure fantasy from fills uncorrelated with adverse flow. Absolute
PnL from `ASSim` must never be cited as economics; only paired-comparison deltas (center vs
center, skew vs no-skew) are meaningful, and even those are weak while fills are exogenous.

**Standing verdict + next step:** the honest maker-economics instrument remains the A4 queue
replay (§4.7, conservative FIFO, EV ≈ +0.01–0.04 bps/posting). The required upgrade before any
HF5 economics claim: **couple `ASSim`'s fills to `QueueSim`** (post the A-S quotes into the
queue engine, fills only by depletion/price-through) so adverse selection enters the fill
process itself. Sim-only; no live path; G8 + kill-switch gate any graduation.

**HF5b addendum (same day): the coupling built and run — the honest instrument is NEGATIVE
for textbook A-S spreads.** `ASQueueSim` (deterministic, no RNG: fills only by touch-zone
FIFO depletion or price-through, conservative join `l1_fraction·depth` on every post,
requoting resets priority, crossing quotes never placed). Same BTC window, γ=0.02, κ=20,
τ=100, requote 1 s, HF4 gate:

| center | pnl (bps) | fills | per-fill | mean q | liq cost |
|---|---|---|---|---|---|
| HF1 microprice | −6,015 | 3,187 | **−1.89 bps** | −0.04 | 10.6 |
| mid (control) | −4,922 | 3,172 | **−1.55 bps** | +0.23 | 14.1 |

Mechanism: risk-widened A-S quotes sit *behind* the touch for most of the tape, so their
fills are dominated by price-throughs — adverse by construction. Bracketed with §4.7
(touch-joined postings: marginally +EV, rebate-carried), the maker economics now point at a
specific posture: **live at the touch, harvest the rebate, gate per posting on the A4 EV rule
— not textbook A-S spread-widening at these vol parameters.** HF1's center consistently
improves *inventory* (mean q ≈ 0, liquidation cost −25 % here, −40 % in the exogenous run)
but has not yet shown a per-fill PnL edge. Proxy caveats of §4.7 apply (flow/depth proxies,
one symbol, ~1 day). Next candidate experiment: touch-pegged quoting (GAP-04 posture) + HF1
inventory skew + per-posting A4 EV gate, on multi-day data.

### 4.9 Touch-maker experiment (2026-07-31 — pre-registered, multi-day): **all cells FAIL**

*Source: `scripts/execution/touch_maker.py` + `touch_maker_experiment.py`; 173 day-symbol
episodes (~58 days × BTC/ETH/SOL), 8 pre-registered cells (touch-pegged quoting × HF1
side-selection × inventory skew × HF4 gate × A4 EV gate), criteria declared before the run:
(a) pooled per-fill > 0, (b) positive-day share ≥ 0.55, (c) max single-day ≤ 30 % of total,
(d) l1-fraction sensitivity. Artifact: `reports/touch_maker_experiment.json`.*

| cell | fills | per-fill (bps) | pos-day % | max-day % | verdict |
|---|---|---|---|---|---|
| V1 touch both sides | 1,154,890 | −1.66 | 35 | 100 | FAIL(a,b,c) |
| V1 + **EV gate** | 21,244 | **+0.67** | 47 | 118 | FAIL(b,c) |
| V2 + HF1 side-select | 417,684 | +6.14 | 49 | 71 | FAIL(b,c) |
| V2 + EV gate | 20,816 | +3.95 | 51 | 164 | FAIL(b,c) |
| V3 skew only | 773,033 | −1.50 | 0 | 100 | FAIL(a,b,c) |
| V4 all + EV | 18,570 | −1.52 | 38 | 100 | FAIL(a,b,c) |

**What was learned (the experiment answered its question):**
1. **Always-on touch quoting is bled dry by adverse selection** (−1.66/fill, 35 % positive
   days) — §4.8's mechanism operates at the touch too, not only behind it.
2. **The A4 EV gate is validated as a filter**: it flips V1's per-fill sign (−1.66 → +0.67)
   while cutting fills 55×. Capture-vs-adverse gating works; it just doesn't produce
   day-over-day consistency on this data.
3. **HF1 side-selection prints large but non-maker PnL**: +6.1/fill on 418 k fills is
   directional inventory riding trends (49 % positive days, 71 % single-day concentration) —
   and implausible as maker income under the flow proxies. Treat as proxy artifact until
   better data; the honest signal here is that HF1's direction has value, already known (§HF1
   IC), not that a maker harvests it.
4. **The binding failures everywhere are (b) day-consistency and (c) concentration** — the
   two criteria that killed `surprise_signal` in §4.6. Pre-registration prevented shipping
   another one-lucky-day discovery.

**Conclusion for the maker line:** with snapshot proxies (rolling-window flow splits,
depth-fraction queue joins), no touch-maker configuration clears the pre-registered bar. The
maker route to Q5 remains **unproven, not disproven** — the decisive unblocker is data, not
more sim variants: L1 queue sizes + per-tick side volume from the ingestor (a planned F-task
schema change), or shadow quoting on the T0b box once it exists. Until then the maker line's
validated assets are the instruments themselves (A4 EV gate, queue sims, HF1 center) and the
posture knowledge (§4.7–4.9). Sim-only throughout; nothing here touches capital gates.

### 4.10 Fee-tier repricing (2026-08-03 — X-1): **the maker line is fee-tier-invariant**

*Source: `scripts/execution/fee_tier_reprice.py`; ladder in `config/costs.toml`
`[hyperliquid_staked]`, taken from the [venue fee docs](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees)
(wood 5 % · bronze 10 % · silver 15 % · gold 20 % · platinum 30 % · diamond 40 %), verified
2026-08-03. Part A: the §4.7 A4 replay re-run on BTC 2026-07-29→30 (1.294 M ticks, 6,469
postings/side). Part B: the §4.9 8-cell grid over 70 days × BTC/ETH/SOL = **179 day-symbol
episodes**, §4.9 acceptance criteria **imported unchanged** — this study moves prices, never
the bar. Artifact: `reports/fee_tier_reprice.json`; every cell stamped with `tier_summary()`.*

**The mechanic that decides the whole study:** staking discounts apply to *fees paid*, and per
the venue docs **not to maker rebates**. A resting quote's economics are half-spread + rebate −
adverse selection; the discount reaches none of those. It reaches only the taker legs — which,
under the §1 doctrine, a maker pays only when liquidating inventory.

**A. §4.7 EV per posting** (measured, cost-free: fill 0.563/0.559, half-spread 0.0832 bps,
E[adverse|fill] 0.228/0.242 bps — reproduces the §4.7 structure on a longer window):

| tier | taker (bps) | rebate | EV BID | EV ASK | *if rebates were discounted too* |
|---|---|---|---|---|---|
| none | 3.50 | 0.200 | **+0.0313** | **+0.0228** | (same) |
| gold | 2.80 | 0.200 | **+0.0313** | **+0.0228** | +0.0088 / +0.0004 |
| diamond | 2.10 | 0.200 | **+0.0313** | **+0.0228** | **−0.0137 / −0.0219** |

Identical at every rung. The pessimistic column runs the *wrong* way — it crosses zero between
gold and platinum, i.e. if the venue ever discounted rebates, staking harder would make the
maker line worse.

**B. §4.9 grid — per-fill bps and verdicts** (base = tier `none`; the intermediate rungs
interpolate monotonically and are in the artifact):

| cell | fills | per-fill `none` | per-fill `diamond` | pos-day % | max-day % | verdict (all tiers) |
|---|---|---|---|---|---|---|
| V1 touch both sides | 1,168,018 | −1.787 | −1.738 | 35 | 100 | FAIL(a,b,c) |
| V1 + **EV gate** | 21,596 | +0.390 | +0.540 | 48 | 201 | FAIL(b,c) |
| V2 + HF1 side-select | 421,937 | +6.422 | +7.137 | 47 | 67 | FAIL(b,c) |
| V2 + EV gate | 21,523 | +3.833 | +4.434 | 50 | 163 | FAIL(b,c) |
| V3 skew only | 781,944 | −1.504 | −1.503 | 0 | 100 | FAIL(a,b,c) |
| V3 skew + EV gate | 28,680 | −2.095 | −2.079 | 27 | 100 | FAIL(a,b,c) |
| V4 all | 148,020 | −1.498 | −1.494 | 4 | 100 | FAIL(a,b,c) |
| V4 all + EV | 19,230 | −1.497 | −1.465 | 37 | 100 | FAIL(a,b,c) |

**No cell flips at any tier** — 8 cells × 7 rungs, plus the pessimistic sensitivity: 0 survivors.

**What was learned:**
1. **The fee tier is not a live hypothesis for the maker line.** The whole ladder moves per-fill
   PnL by ≤ 0.72 bps (largest: V2 +6.42 → +7.14 at diamond, +11 %), while the binding failures
   are criteria (b) day-consistency and (c) concentration — structural properties of the fill
   distribution that no price level touches. §4.9's conclusion stands unchanged at every tier.
2. **Where the discount does bite, it is not maker income.** Its only channel is terminal
   liquidation, so the benefit scales with leftover inventory — largest exactly in V2, the cell
   §4.9 already flagged as *directional inventory riding trends, not maker income*. The tier
   subsidises the part of the P&L that isn't the strategy.
3. **The pessimistic sensitivity is the informative one.** Discounting the rebate cuts fills
   23 % (21,596 → 16,724 in V1+EV) because the A4 gate needs capture > adverse: a smaller
   rebate closes the gate more often. Per-fill *rises* (+0.390 → +0.755) and pos-day share
   reaches 0.547 — the closest any configuration has come to (b) — but still fails, and on
   strictly less volume. Selectivity improves quality without producing consistency.
4. **The base re-run reproduces §4.9** on 179 episodes vs 173 (V1 −1.79 vs −1.66, V1+EV +0.39
   vs +0.67, V2 +6.42 vs +6.14, V4+EV −1.50 vs −1.52): the verdicts are stable to a 6-episode
   data extension, which is a mild robustness datum for §4.9 itself.

**The fee risk that actually matters is upstream of this study.** The SSOT prices a 0.2 bps
maker *rebate*, which presumes a maker volume tier; the venue's base perp maker rate is a
+1.5 bps *fee*. At the base rate the §4.7 EV (+0.031 bps, carried entirely by a 0.20 bps
rebate) is deeply negative — the maker line's survival rests on a volume-tier assumption
that has never been tested, and unlike the staking discount it is not a rounding effect.
Quantifying it is the next honest cost question (a `[hyperliquid_maker_tiers]` follow-up),
not more staking arithmetic. Proxy caveats of §4.7 apply throughout; the final day of the
window was still being written at run time. Sim-only; no live path; nothing here touches
capital gates.

### 4.11 Maker-ladder sweep (2026-08-04 — COST-5): **zero fees are not free money**

*Source: `scripts/execution/fee_tier_reprice.py --maker-sweep`; ladder in `config/costs.toml`
`[hyperliquid_maker_tiers]` from the [venue fee docs](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees)
(verified 2026-08-04). Convention: **positive = rebate earned, negative = fee paid**. §4.9 grid
re-simulated in full at each rung — 179 day-symbol episodes × 8 cells × 4 rungs — because the
maker rate enters both the per-fill cash and the A4 gate's capture term and is therefore *not*
repriceable. Criteria imported unchanged. Artifact: `reports/maker_tier_sweep.json`.*

**The ladder NAT has been assuming.** The venue runs two ladders: volume tiers (0.015 % → 0.000 %
at >$500 M/14 d — reaching **zero at best**) and rebate tiers (>0.5 % / >1.5 % / >3.0 % of
venue-wide 14-day *maker* volume → −0.001 / −0.002 / −0.003 %). The SSOT's 0.2 bps rebate is
**rebate tier 2 — it presumes ≥1.5 % of all Hyperliquid maker volume**, and every maker number in
§4.7–§4.10 was priced at it.

**A. §4.7 EV per posting, closed-form in the maker rate** (BTC 2026-07-29→30):

| maker tier | rate (bps) | EV BID | EV ASK |
|---|---|---|---|
| base (0.015 % fee) | −1.50 | −0.925 | −0.928 |
| vol_t3 (>$100M) | −0.40 | −0.306 | −0.313 |
| **zero_fee (>$500M)** | **0.00** | **−0.081** | **−0.089** |
| rebate_t1 (>0.5 %) | +0.10 | −0.025 | −0.033 |
| rebate_t2 (>1.5 %) — the SSOT | +0.20 | +0.031 | +0.023 |
| rebate_t3 (>3 %) | +0.30 | +0.088 | +0.079 |

**Breakeven maker rate = E[adverse|fill] − half-spread = +0.144 bps (bid) / +0.159 bps (ask).**
Zero fees leave a resting quote ~0.08 bps/posting *under water*: at BTC's touch the half-spread
(0.083 bps) is roughly a third of the adverse selection (0.228/0.242 bps), so not-being-charged
is not enough — the venue must **pay** you ~0.15 bps before a passive quote breaks even. Every
positive maker number on this platform lives above rebate tier 1.

**B. §4.9 grid, per-fill bps (fills)** — `rebate_t2` column from §4.10 for reference:

| cell | base −1.5 | zero 0.0 | reb_t1 +0.1 | reb_t2 +0.2 | reb_t3 +0.3 |
|---|---|---|---|---|---|
| V1 touch both | −3.49 (1.17 M) | −1.99 (1.17 M) | −1.89 (1.17 M) | −1.79 (1.17 M) | −1.69 (1.17 M) |
| V1 + EV gate | −3.64 (1 k) | +0.54 (12 k) | **+2.01 (15 k)** | +0.39 (22 k) | −0.84 (26 k) |
| V2 + HF1 side | +4.79 (423 k) | +6.29 (423 k) | +6.39 (423 k) | +6.42 (422 k) | +6.59 (423 k) |
| V2 + EV gate | −14.08 (0.4 k) | +8.92 (12 k) | +4.82 (16 k) | +3.83 (22 k) | +4.95 (27 k) |
| V3 skew | −3.20 (784 k) | −1.70 (784 k) | −1.60 (784 k) | −1.50 (782 k) | −1.40 (784 k) |
| V4 all + EV | −10.29 (0.4 k) | −1.28 (10 k) | −1.55 (13 k) | −1.50 (19 k) | −1.37 (26 k) |

**No cell survives at any maker rate** — 8 cells × 4 rungs, 0 survivors, same binding failures
(b) day-consistency and (c) concentration as §4.9/§4.10. At the *base* rate (a +1.5 bps fee, i.e.
what an account below $5 M/14 d actually pays) the whole grid is deeply negative and the EV gate
closes almost completely (1 k fills vs 22 k at tier 2).

**What was learned:**
1. **Un-gated cells are exactly linear in the maker rate** (V1: −3.49 → −1.69 across 1.8 bps of
   ladder, i.e. 1.0 bps of rate ⇒ 1.0 bps/fill), which is the arithmetic sanity check that the
   rate enters where it should and nowhere else.
2. **EV-gated cells are NON-monotone** — V1+EV peaks at rebate_t1 (+2.01) and *falls* to −0.84 at
   rebate_t3. Mechanism, visible in the fill counts: a larger rebate raises the gate's capture
   term, so `capture > adverse` admits progressively more marginal postings (1 k → 12 k → 15 k →
   22 k → 26 k fills). **A better fee tier loosens the filter and buys worse fills.** "Better fees
   ⇒ better maker P&L" is false for any capture-gated strategy; the gate threshold must be
   re-derived per fee tier, not inherited.
3. **The fee tier NAT assumes is doing more work than any signal in the stack.** Moving from the
   rate an ordinary account pays (−1.5) to the assumed one (+0.2) is worth ~1.7 bps/fill — larger
   than every gating and side-selection effect measured in §4.7–§4.10 combined. This is the
   single most load-bearing unvalidated assumption in the maker line.
4. Combined with §4.10: **neither fee ladder rescues the maker line.** Staking cannot touch it
   (rebates are exempt); the maker ladder moves the level but not the verdict, and its attainable
   rung (zero, via volume) is below breakeven at BTC's touch.

**Consequence for the research program.** Passive quoting at a touch this tight is structurally
negative unless the venue pays a rebate NAT has no claim to. The untested surface is the other
direction — **wider-spread pairs**, where the half-spread is a multiple of BTC's 0.083 bps and
the breakeven rate is correspondingly easier. That is the Class-3 scanner's universe (~150 perps,
REST candles, no ingestor and no streak dependency), and no maker experiment has ever been run
outside the three tightest symbols on the venue. Sim-only; proxy caveats of §4.7 apply throughout.

## 5. Combination, gating & regime

- **Hierarchical combiner** (2026-06-10; ⚠️ 2-day OOS 06-08→10, 4-fold, 100-bar embargo):
  composite IC BTC **+0.178** (Sharpe +1.25, dir-acc 0.557) · ETH **+0.248** (+1.71, 0.576) ·
  SOL **+0.359** (+2.40, 0.594; 3.3 h horizon). L1 slow bias × L2 fast timing (zeroed on
  disagreement) × L3 inverse-vol sizing. **Directional gating works:** L2 conditional-on-agreement
  IC exceeds unconditional — the first architecture structurally addressing §2. Honest caveats in
  the source: monotonically rising fold ICs (possible look-ahead/trend artifact), L1 dominance
  (ablation pending), SOL likely inflated, costs assumed not measured.
- **Spannung arc** (situation_analysis): 1,350-combo grid — raw L1 imbalance IC 0.45, EWM smoothing
  *hurts*; causality clean (smooth lag decay, no look-ahead); aging IC 0.48 (IS) → 0.47 (24 h) →
  0.45 (48 h) → 0.36 (3 wk); **unprofitable at taker fees** (0.17–0.37 bps edge vs 7 bps RT);
  spectral: the entire IC lives in the 0.005–0.1 Hz band (10–200 s periods), OU half-life 5–7 s,
  dominant coherence at 68 s.
- **Entropy gate:** `ent_book_shape` lifts imbalance IC +22 % in the low-entropy quintile
  (0.45 → 0.55–0.67, replicated cross-symbol) — the measured basis for regime gating (→ PROC-6).
- **Clustering:** 54 vector×timeframe combos → 22 pass existence, 6 pass stability, **only
  orderflow@5min passes predictive value** (KW p=3e-6 but η² = 0.057). Clusters exist
  geometrically; predictive power is minimal.

- **Orthogonalization does not survive a holdout — first datum** (2026-08-05, PROC-15
  `processes/residualize.py`; BTC 2026-08-04, 246,037 rows, 70/30 prefix split).
  Residualizing `imbalance_qty_l5` against `imbalance_qty_l1` (`res_f = f − β'Z`, β fit on
  the prefix only):

  | segment | \|corr(res, Z)\| | OLS β |
  |---|---|---|
  | prefix (fit) | 0.0000 | +0.8148 |
  | holdout | **0.1919** | +0.7290 |

  The prefix zero is OLS arithmetic, so the holdout number is the whole content: **β drifts
  ~10 % inside a single day**, leaving ~0.19 residual correlation with the very variable it
  was orthogonalized against. Same pass, `imbalance_depth_weighted | imbalance_qty_l1`:
  holdout 0.183 (R²_fit 0.731). `flow_vwap_deviation | imbalance_qty_l1` is the control —
  β ≈ 0, R²_fit 0.103, holdout 0.083 — i.e. the axis §1 calls mean-reverting really is
  near-independent of book pressure, while the imbalance cousins are not stably separable from
  each other.

  **Why it matters:** §1's "eight independent signal axes" and the combiner contract's *one
  representative per orthogonal axis* (`specs/maker_system.md` §2) are enforced today by
  correlation dedup on **full-sample** statistics. This says those statistics do not hold on
  a 70/30 split of one day — so "orthogonal" is currently a full-sample property being used
  as if it were a forward one. **Not a verdict:** one symbol, one day, one linear method.
  What would settle it is the same residualization swept across days × symbols × pairs
  (cheap — it is an XS process), reporting holdout \|corr\| per pair; if the drift replicates,
  either the axes need rolling re-fits or the orthogonality claim must be restated as
  regime-local rather than structural.

## 6. Convolver & data-volume arithmetic

*Source: convolver_data_analysis 2026-06-03 (BTC, 12,059 60 s candles ≈ 8.4 days).*

- Discovery run: 1,775 events / 6 types; 6 of 158 kernels survive BH-FDR; **0 of 6 robust OOS**
  (walk-forward decay <0.50) — yet temporal shape stability is high (turtle_bull ρ=0.920,
  trap_bull ρ=0.941). Verdict: *methodology sound, constraint is data volume* — "the convolver is
  broken" is FALSE.
- Sample arithmetic: SVD stability needs ≥100 events/(type,channel); trap events run 11–14/1000
  candles → **~39 k candles minimum**; multi-timeframe scaling for 200 OOS traps: 60 s ≈ 1 month ·
  5 min ≈ 5 months · 15 min ≈ 1.5 years · 1 h ≈ 5 years · 4 h ≈ 20 years. Hourly+ discovery is
  infeasible.
- Regime coverage: 35 days captures 1–2 regimes; across-regime validation needs **6–24 months**.
  Event rates are non-stationary (≈60 breakouts/1000 candles trending vs ≈15 sideways).
- Expected |ρ| < 0.2 vs existing microstructure winners (OHLCV shapes vs tick book) — highest
  value as ML input features, not a standalone algorithm.

## 7. Data-integrity record

*Sources: data_inventory 2026-06-12 · korrektur audit 2026-06-10 · features_report.*

- **Inventory (as of 2026-06-12):** 9.4 GB total; `data/features/` 8.4 GB, 671 parquet files,
  34 date dirs over 54 calendar days (Apr 19 – Jun 12) → **20 days missing (~37 %)**; 22 good days
  (>200 MB); expected full day ≈ 500–680 MB. Longest clean streak **May 18–29 (12 days)**.
- **Defect audit (K1–K6):** K1 Docker volume-mount data loss (Critical, fixed 06-10) · **K2 56
  dead features → 82 all-NaN columns** (whale 12, liquidation 13, concentration 15, GMM 8,
  heatmap 8; ~24 % column bloat; blocks meta_labeling; open at audit time — *wiring later verified
  locally Jun-16, production verdict still pending, task Q-K2)* · K3 `regime_accumulation_score`
  constant 0.4429 std=0 (despite the scan listing it as slow-directional — likely K2 fallback) ·
  K4 WebSocket gaps 10–12/hr >1 s, max 13.4 s (median cadence a clean 100 ms) · **K5 six-day gap
  Jun 4–10: process alive, zero writes, no error logs** (watchdog added; recurred Jul-5 → REL tasks) ·
  K6 ~17 days Apr–May unrecoverable (largest Apr 25 – May 6).
- Anomaly: May 31 – Jun 2 produced 125/57/83 tiny fragment files (~13/6/10 MB vs ~500 MB normal).
- **Sufficiency verdicts (Jun-12):** walk-forward 4-fold SUFFICIENT · OOS30 22/30 dates · 30-day
  rolling IC 12/30 · HMM (60 d) 22/60 · full confidence (90 d) ETA ~Sep. All ETAs assumed
  uninterrupted ingestion — an assumption the K5/Jul-5 recurrences broke.

## 8. Platform & hypothesis-suite metrics

*Source: project_state_report 2026-06-09.*

- Feature engine p99 latency **<80 ms/tick** (emission-budget integration test); GMM online
  inference <1 ms; 236 features = 154 base + 82 optional.
- **Hypothesis suite: all 5 confirmed** under the NOGO/PIVOT/GO matrix — H1 whale flow→returns,
  H2 entropy×whale interaction, H3 liquidation-cascade prediction, H4 concentration→volatility,
  H5 persistence.
- Two Rust algorithm implementations (`regime_gated`, `kalman_imbalance`) were **dummy stubs
  returning NaN** at report time. Test asymmetry: 4,100+ unit tests vs 2 integration tests.
- Raw-trade collection (TradeParquetWriter → `data/trades/`) started 2026-06-09 — the input the
  execution-model research (§2) needs.

---

## Provenance

| Merged source (now in `archive/`) | Date | Sections |
|---|---|---|
| `in_progress/research/new/9_6/full_ic_scan_report.md` | 2026-06-09 | §1 |
| `in_progress/research/new/9_6/ic_horizon_analysis.md` | 2026-06-09 | §1, §2 |
| `in_progress/research/new/9_6/ic_validation_report.md` | 2026-06-09 | §2, §3 |
| `in_progress/research/new/10_6/hierarchical_combiner_report.md` | 2026-06-10 | §5 |
| `in_progress/research/new/9_6/project_state_report.md` | 2026-06-09 | §8 |
| `in_progress/research/new/convolver_data_analysis.txt` | 2026-06-03 | §6 |
| `research/gauntlet_analysis_2026_06_01.md` | 2026-06-01 | §4.4 |
| `in_progress/tasks_assigned_12_6_26/features_report.md` | 2026-06-12 | §1, §7 |
| `in_progress/tasks_assigned_12_6_26/algorithms_report.md` | 2026-06-12 | §4.1–4.2 |
| `in_progress/tasks_assigned_12_6_26/algorithm_classification.md` | 2026-06-12 | §4.3 |
| `in_progress/tasks_assigned_12_6_26/situation_analysis.md` | 2026-06-12 | §4, §5 |
| `in_progress/tasks_assigned_12_6_26/data_inventory.md` | 2026-06-12 | §7 |
| `archive/in_progress/korrektur_tasks.md` | 2026-06-10 | §7 |

*Updates after 2026-06-12 (from project memory, not a merged report): position-tracker wiring
verified locally Jun-16; Q0 streak check failed Jun-19; zombie-ingestor recurrence Jul-5 → REL-1..4
in `TASKS.md`.*
