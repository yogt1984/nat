# NAT — Consolidated Empirical Findings

**What this is:** the single consolidated record of everything NAT has *measured* — about the data,
the market, and its own code. Merged from the Tier-A finding reports (per
[`archive/in_progress/INDEX.md`](../archive/in_progress/INDEX.md) classification); sources preserved in
[`archive/`](../archive/) with full provenance (table at the bottom). Findings are point-in-time:
each block states its test window. Nothing here is a plan — plans live in [`PLAN.md`](../PLAN.md) /
[`TASKS.md`](../TASKS.md).

**One-page overview:** [`FINDINGS_SUMMARY.md`](FINDINGS_SUMMARY.md) — every row links back here;
this file is the source when the two disagree.

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

> **⚠️ ID erratum (added 2026-08-06).** The two follow-ups named "COST-4" and "COST-5" in this
> section were minted against IDs **already in use** in `TASKS.md` (wave-gate thresholds; maker
> volume-tier quantification). Their canonical IDs are now **COST-6** (harness VIP9-default purge)
> and **COST-7** (CI-guard hardening). Commit subjects `d9f3c1c` / `1334d41` are immutable and still
> read COST-4/COST-5 — this line is the pointer. The collision is what prompted the ID-registry
> rule now in `TASKS.md` § Conventions: an ID exists when its row exists.

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

- ⚠️ **REFUTED 2026-08-08 — both halves. See §5.1 below.** The composite loses to a single
  feature under honest walk-forward (A-2), and the gating mechanism is not merely absent but
  **harmful** (A-1). The bullet is retained as the claim that was tested, not as a finding.

  **Hierarchical combiner** (2026-06-10; ⚠️ 2-day OOS 06-08→10, 4-fold, 100-bar embargo):
  composite IC BTC **+0.178** (Sharpe +1.25, dir-acc 0.557) · ETH **+0.248** (+1.71, 0.576) ·
  SOL **+0.359** (+2.40, 0.594; 3.3 h horizon). L1 slow bias × L2 fast timing (zeroed on
  disagreement) × L3 inverse-vol sizing. **Directional gating works:** L2 conditional-on-agreement
  IC exceeds unconditional — the first architecture structurally addressing §2. Honest caveats in
  the source: monotonically rising fold ICs (possible look-ahead/trend artifact), L1 dominance
  (ablation pending), SOL likely inflated, costs assumed not measured.

### 5.1 The combiner, retired (A-2 + A-1, 2026-08-08) — **the last unrefuted capital-relevant claim**

*Sources: `alpha/walkforward_ic.py` + `exploration/a2_combiner_revalidation.py` (A-2);
`processes/agreement_gate_eval.py` + `exploration/a1_agreement_gate_study.py` (A-1). 25 days
(2026-07-14→08-07), 5-min bars, BTC/ETH/SOL. Criteria pre-registered in the driver docstrings
before either run.*

Until this week the combiner was the only capital-relevant claim in the record that had never
been tested to destruction. It made **two** separate claims and both are now refuted.

**A-2 — the composite loses to one of its own inputs.** Walk-forward IC, weights refitted per
fold on rows strictly before it, evaluated on **non-overlapping** observations:

| symbol | walk-forward IC | §5 claimed | positive folds | max-fold share | verdict |
|---|---|---|---|---|---|
| BTC | +0.062 | .178 | 0.60 | 0.36 | FAIL(c,d) |
| ETH | +0.099 | .248 | 0.80 | 0.31 | FAIL(c,d) |
| SOL | **−0.024** | .359 | 0.20 | 0.40 | FAIL(a,b,c,d) |

Criterion (d) is the one that matters: **the composite never beats its own best single
feature.** `trend_ema_short` alone scores ~0.20 against the three-layer stack's 0.06–0.10, and
SOL — §5's strongest claim — comes back negative. The hierarchy destroys information.

**The mechanism behind §5's number is in the file dates.**
`models/hierarchical_combiner/weights_BTC.json` carries `training_date 2026-06-11`; the OOS
window was 2026-06-08→10. **The weights were fitted after the period they were scored on.**
That produces monotonically rising fold ICs by construction — each successive fold has more of
its own fit behind it — which is exactly the artifact §5 observed and could not explain.

**A-1 — the gate is not neutral, it is harmful.** Conditional IC given fast/slow agreement,
against a null that permutes *the gate* while holding the subset size fixed:

| symbol | hz | IC uncond. | IC **agree** | IC **disagree** | lift | z |
|---|---|---|---|---|---|---|
| BTC | 30m | −0.033 | −0.043 | −0.024 | −0.010 | −0.45 |
| ETH | 30m | −0.020 | **−0.087** | **+0.040** | −0.067 | **−2.91** |
| SOL | 30m | +0.011 | **−0.047** | **+0.073** | −0.057 | **−2.71** |
| SOL | 2h | −0.004 | −0.050 | +0.047 | −0.046 | −2.11 |

On ETH and SOL the **disagreement** subset carries the signal and agreement destroys it;
`frac_days_informative = 0.00` everywhere. §5 claimed conditional IC *above* unconditional.

**Why the original passed, and the null that catches it.** The trap is selection, not
estimation: split any sample on any condition and report the better half, and a lift appears —
the agreement subset is smaller and differently distributed, so `max(agree, disagree)` beats
the pooled figure by construction. "Agreement IC > unconditional IC" therefore passes on pure
noise, and §5's pilot is that shape. A-1's null permutes which observations count as agreeing
while holding the fast signal, the target and the **subset size** fixed — size matters because
IC's sampling variance depends on n, so a resizing null would measure sample size rather than
structure. A planted test with two independent signals confirms the raw lift is often positive
there while the calibrated one is not.

**A second finding fell out of A-2: the data cannot currently test §5's own horizon.** At its
~5 h horizon, 25 days of gap-affected bars yield **28 non-overlapping observations** — no
walk-forward verdict is possible on 28 points, so the results above are at 30 min, the longest
horizon this sample supports. An initial run *without* non-overlapping sampling printed pooled
IC 0.39–0.46 — *higher* than §5's claim, since consecutive bars share 59 of 60 bars of their
forward window; that inflation is recorded here because it is the same error in a new costume.

**Consequence.** Nothing capital-relevant in the record is now unrefuted. §2's adverse-selection
collapse stands unopposed: the one architecture claiming to address it structurally does not.
The surviving open questions are B-5a's β conditional (§7.10) and Track C's beta-neutral
rotation (§7.8), both of which are **time-blocked rather than work-blocked** — XS-9's own power
arithmetic needs ~325 rebalances ≈ 0.89 yr.

*Limits:* A-1 used `imbalance_qty_l1 × regime_divergence_1h` — the §2 contract's axis
representatives — not the combiner's exact L1/L2 composites; a defender of §5 could fairly ask
for those. A-2 priced no costs, because it did not need to: the IC fails before costs are
applied. Both are single-window studies on 25 days.

⚠️ **Amended 2026-08-09 — the A-1 half is weaker than stated above; see [§7.11](#711-the-regime_-category-is-dead-in-production-2026-08-09--and-it-silently-hollowed-out-a-1).**
`regime_divergence_1h` is all-NaN from 2026-07-26, so only **07-15→07-18** of A-1's 25-day
window carries a live gate input and the result rests on **≤4 usable day-folds** — which is
precisely why every cell came back `insufficient_days`/`non_durable`. A second run on the
alive window reproduces the *direction* (0/9 informative, 7/9 lifts negative) but not the
strength. **A-1 is better read as "no support, underpowered" than as "refuted."** A-2 is
unaffected — it does not use `regime_` at all — so the section's headline conclusion stands on
A-2 alone.
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

- **Orthogonality holds out of sample, with a fat tail — and one axis should be retired**
  (2026-08-05; `exploration/orthogonality_sweep.py` over PROC-15 `residualize`; **152 episodes,
  51 days 2026-05-18→08-04 × BTC/ETH/SOL**; artifact `reports/orthogonality_sweep.json`).
  Per episode: fit `res_f = f − β'Z` on the leading 70 % with `Z = imbalance_qty_l1`, measure
  `|corr(res, Z)|` on the untouched 30 %. Prefix correlation is 0 by OLS construction, so the
  holdout number is the entire content. `flow_vwap_deviation` was carried as a **control** — §1
  calls it a distinct axis, so a sound method should leave it clean while redundant cousins drift.

  | target | median \|corr\| | p90 | frac > 0.10 | R²_fit | median β drift |
  |---|---|---|---|---|---|
  | `raw_bid_depth_5` / `raw_ask_depth_5` | 0.098 / 0.097 | 0.28 / 0.30 | 0.50 / 0.49 | 0.43 / 0.42 | 0.12 |
  | `micro_queue_position_bid` | 0.096 | 0.520 | 0.49 | 0.048 | 0.377 |
  | **`flow_vwap_deviation` — CONTROL** | **0.093** | 0.176 | 0.46 | 0.078 | 0.329 |
  | `imbalance_orders_l5` | 0.083 | 0.242 | 0.44 | 0.789 | 0.052 |
  | `imbalance_qty_l5` / `_notional_l5` | 0.074 | 0.196 | 0.34 | 0.788 | 0.036 |
  | `cross_obi_mean` | 0.076 | 0.170 | 0.38 | 0.250 | 0.119 |
  | `micro_obi_velocity` | 0.056 | 0.146 | 0.23 | 0.046 | 0.262 |
  | `flow_aggressor_ratio_5s` | 0.029 | 0.091 | 0.07 | 0.021 | 0.208 |
  | `ent_permutation_imbalance_16` | 0.021 | 0.054 | 0.01 | 0.030 | 0.120 |

  1. **The control is what makes this readable.** `flow_vwap_deviation` — the axis §1 calls
     independent — drifts *more* (0.093) than the supposedly redundant `imbalance_qty_l5`
     (0.074). The residual correlation is therefore **general non-stationarity across the split,
     not cousin-specific redundancy**. Without the control the same numbers would have read as
     "orthogonality is shaky everywhere" and been wrong in a way nothing downstream could catch —
     the shape of the VIP9 and five-winners failures (§4.6).
  2. **A single-day precursor was a tail, not a signal.** The first PROC-15 run (BTC 2026-08-04)
     printed 0.192 with β drifting +0.815→+0.729; that value sits at the **89th percentile** here
     (p10 0.017 · median 0.074 · p90 0.196 · max 0.564) and the drift is a BTC median of 2.5 %.
     Recorded because the corrected reading is the finding.
  3. **The axis contract mostly survives, and two axes are now measured rather than asserted:**
     `ent_permutation_imbalance_16` (0.021, 1 % of episodes > 0.10) and `flow_aggressor_ratio_5s`
     (0.029, 7 %) separate from book pressure on essentially every day.
  4. **Retire "raw depth asymmetry" as a separate axis.** `raw_bid/ask_depth_5` are the worst
     pairs (≈0.098, half the episodes > 0.10, p90 ≈ 0.29) — mechanical, since imbalance is *built*
     from those depths. §1's eight-axis list should stop counting it as distinct.
  5. **Respect the tail.** p90 0.17–0.30 and a third to a half of episodes above 0.10 for the
     depth/imbalance block: decorrelation-based sizing should assume less diversification than the
     full-sample number implies — §4.9's day-consistency lesson in another guise.

  *Limits:* one linear method, one conditioning variable, one split geometry (70/30, no rolling
  re-fit); sample shaped by §7's gaps (64 of 216 episodes missing: 38 symbol-days absent, 23
  sub-threshold, 3 from a malformed `2026-05-12-clean` directory). Measures separation only, never
  predictive value. Extensions: rolling-β, and residualizing against a whole selected set (PROC-3).

- **Bar-scale momentum is ANTI-persistent, and no band cell clears the bar — negative result**
  (2026-08-06; `processes/persistence_stats.py` (PROC-20) via
  `exploration/persistence_study.py`; **185 of 219 episodes, ~62 days × BTC/ETH/SOL × {1 min,
  5 min}, 330 cells** at the config null budget; artifact `reports/persistence_study.json`).
  Two families, one run: `P(continue | run length k)` against a **sign-permutation** null
  (permuting the sign series and recomputing runs, so 0.5 is the null value), and `k·σ` band
  touches from a rolling VWAP midline with an embargo, marked out in the reverting direction.

  **A. Momentum does not persist here — the opposite does.** Excess over null at k=1:

  | tf | BTC | ETH | SOL |
  |---|---|---|---|
  | 1 min | −0.0205 (z 10.5, n 18.6 k) | −0.0300 (z 8.6, n 19.6 k) | −0.0138 (z 6.5, n 19.9 k) |
  | 5 min | −0.0263 (z 0.6) | −0.0293 (z 0.7) | −0.0093 (z 2.0) |

  **34 of 36 cells negative**, growing to −0.12 at k=6. At 1 min the pooled z of 6–10 on ~19 k
  events per symbol makes the sign unambiguous; at 5 min it is the same sign but unresolvable.
  The tradeable structure at bar scale is *reversion*, not continuation — consistent with §5's
  OU/spectral picture and with the bounce mechanics of §2.

  **B. LF7's band priors partially replicate, none survive.** 5 min bars, 2 h horizon, markout
  in the reverting direction (bps, n in brackets):

  | sym | k=1.0 | k=1.5 | k=2.0 | k=2.5 | k=3.0 |
  |---|---|---|---|---|---|
  | BTC | +0.3 (199) | +14.3 (155) | +15.2 (111) | +22.6 (82) | +10.6 (60) |
  | ETH | −4.8 (211) | −10.8 (165) | −9.0 (115) | +1.1 (91) | +31.0 (64) |
  | SOL | −7.3 (220) | +12.0 (173) | +23.4 (124) | **+45.8 (93)** | +31.3 (68) |

  LF7 (`research/new/vwap_sd_channel.txt`, single day, n = 4–31/cell) predicted adverse at
  k ≤ 1.5, capture at k ≈ 2.0–2.5, and SOL > ETH > BTC. **Two of three hold** — capture does
  concentrate at k ≈ 2.0–2.5 and SOL leads by a wide margin — but the ordering is
  **SOL > BTC > ETH**, with ETH adverse across most of the grid. At 1 min/30 min almost every
  cell is adverse, matching the shallow-touch continuation signature.

  **The verdict: 0 of 330 cells are informative after FDR + day-durability.** SOL's headline
  +45.8 bps is z = 2.87 with **BH q = 0.33** and `frac_days_informative = 0.00` — 93 events
  across 62 days is ~1.5/day, so no day carries enough to be durable. The 1 min momentum cells
  are the only ones with real pooled significance and they fail day-consistency (q ≈ 0.08–0.12).
  **Nothing here is promotable**, and the grid's shape — 330 cells, thin per-cell counts, large
  bps values — is precisely the one that manufactures headline numbers.

  *Consequence for LF7:* its k-prior survives as a prior and nothing more; the missing
  ingredient is **events per day**, not a better k. More events means lower k, more symbols, or
  the wider-spread pairs `B-5` targets — which is the same conclusion §4.11 reached from the
  cost side. *Limits:* touch price is a fill proxy that **overstates** fills (price must trade
  through a resting order), so A4's queue sim gates any profit claim; markouts are gross of
  fees and funding; 34 of 219 episodes lost to §7's gaps.

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

*Sources: data_inventory 2026-06-12 · korrektur audit 2026-06-10 · features_report · candle
universe backfill 2026-08-07.*

### 7.1 The candle universe (XS-1, 2026-08-07) — complete, but shallow by venue design

*Source: `scripts/data/fetch_candles.py --universe` on su-75; per-pair depth audit over the
resulting parquet. 177 listed perps (55 delisted excluded) × {1m, 5m, 15m, 1h}.*

**What was captured: 708 series, 3,059,200 candles, 98 MB, and every series is 100 % complete
within its span — zero gaps anywhere.** That is a categorically better substrate than the tick
record above (37 % of calendar days missing, 82 all-NaN columns), and it is the first dataset on
the platform with no integrity caveat attached.

**The constraint that matters more than the volume — a ~5000-bar retention cap per interval:**

| interval | candles retained | span reachable | measured depth (177 pairs) |
|---|---|---|---|
| 1m | ~5,000 | ~3.5 d | **3.5 d** (0 pairs reach 7 d) |
| 5m | ~5,000 | ~17 d | 17.4 d |
| 15m | ~5,000 | ~52 d | 52 d (175/177 ≥ 30 d) |
| 1h | ~5,000 | ~208 d | 90 d requested, 175/177 full |

Measured, not inferred: a narrow 2 h window **4 days back at 1m returns zero candles**, while 1h
at 89 days back returns normally. The cap is on bar *count*, so reachable history scales with bar
size. The two 1h pairs short of 90 d (GRAM 36 d, CASHCAT 27 d) are recent listings, not gaps.

**Three consequences:**

1. **1m universe history cannot be backfilled — only accumulated.** `specs/maker_system.md` §5
   specifies Tier-W as "REST 1 m candles for the full perp universe"; that is not achievable
   retroactively and never will be. The 3.5 d captured on 2026-08-07 now exists *only* in our
   parquet — the venue will drop it. Every day without a refresh cron is a permanently lost day of
   1m breadth (task `XS-7`, promoted to P0 on this finding).
2. **Cross-sectional work should start at 15m/1h, not 1m.** 52 d and 90 d respectively, versus
   3.5 d. This agrees with §5's PROC-20 result from the opposite direction: 1m/5m momentum is
   *anti*-persistent and the 5m cells were statistically unresolvable.
3. **A "successful" sweep is not a complete one.** The backfill reported `ok=177 failed=0 empty=0`
   for the 1m run that returned 4 % of the requested span — `ok` means "rows came back", not "the
   requested window was satisfied". The gap was caught only by a separate depth audit. Any sweep
   that reports coverage should compare **requested vs received span**, which is the §4.9 lesson
   (a silent cap reads as "covered everything") reappearing in the data layer.

*Minor:* two pairs returned empty once (ORDI 15m, REZ 5m) and both succeeded on immediate retry —
the `empty` bucket conflates "venue has none" with a transient request failure; `XS-7` adds a retry
pass.

### 7.2 The universe's spreads (XS-8, first reading 2026-08-07) — **n = 1, recorded as a prior**

*Source: `scripts/data/fetch_l2.py` (XS-8), one `l2Book` sweep of all 177 listed perps at
08:40 UTC, 177/177 OK, 0 degenerate, 0 failed. **This is a single snapshot.** The sampler
exists because half-spread moves all day and one book is an n=1 estimate — the error PROC-20
corrected in LF7's priors. Nothing below is a verdict; it is the first datum in an
accumulating distribution.*

Half-spread, bps of mid:

| p0 | p5 | p25 | **p50** | p75 | p95 | p100 |
|---|---|---|---|---|---|---|
| 0.056 | 0.084 | 0.750 | **1.372** | 2.239 | 6.904 | 26.810 |

**NAT has been studying the extreme tight tail of its own venue.** BTC sits at 0.0776 bps
(an independent cross-check of §4.11's 0.083, measured by a different instrument on a
different day). The universe median is **17.7× wider**, and **169 of 177 pairs are wider
than BTC**. SOL — one of the three symbols every maker experiment has used — ranks in the
**tightest five** (ALGO 0.056, DOT 0.061, SOL 0.069, FIL 0.072, DOGE 0.072). §4.11 inferred
this sampling bias from the cost side; it is now measured.

**The conditional that B-5a must resolve.** §4.11's relation is
`breakeven_maker_rate = E[adverse|fill] − half_spread`. *If* adverse selection stayed at
BTC's measured 0.228 bps, then 156 of 177 pairs would cover it on half-spread alone —
before any rebate, at zero fees. That "if" is carrying the entire result and is almost
certainly false in the helpful direction: spreads are wide *because* market makers price
inventory and toxicity risk into them, so `E[adverse|fill]` should scale with the spread.
If it scales proportionally, the ratio is unchanged and nothing improves. **E[adverse|fill]
is measured only for BTC/ETH/SOL**, so B-5a must bound it for wide pairs and state the
bound, not bury it.

**Capacity is the other blade, and it cuts the opposite way.** The widest pairs are
nearly empty at the touch: XAI 12.94 bps with **$20** of bid notional, DYM 7.17 bps with
$613, HMSTR 26.81 bps with $3,326. MEME ($10.3 k) and BOME ($12.5 k) are the exceptions.
A large per-fill edge on $20 of size is not a business, so the joint requirement — wide
enough *and* deep enough — is far more restrictive than either margin alone. This is
exactly the admission test `XS-5` exists to apply.

*Limits:* one sweep, one moment (08:40 UTC), no intraday or weekday variation, and
quoted spreads are not fill economics — they say what a resting order could earn, never
what it would be filled against. Sim and adverse-selection work still gate any claim.

### 7.3 Permutation entropy does not rank this universe (XS-2, 2026-08-07) — negative

*Source: `scripts/xs/features.py` on the XS-1 candle archive — 177 pairs, 1h log returns
(≈2,160 bars each), repeated at orders 3/4/5/6 and on 15m.*

`specs/maker_system.md` §5 lists permutation entropy as one of Tier-W's per-pair scores.
Measured, **it carries no cross-sectional information at bar scale**:

| order | min | median | max | **IQR** |
|---|---|---|---|---|
| 3 | 0.9980 | 0.9996 | 1.0000 | **0.0005** |
| 4 | 0.9953 | 0.9985 | 0.9995 | 0.0008 |
| 5 | 0.9768 | 0.9941 | 0.9960 | 0.0013 |
| 6 | 0.8970 | 0.9719 | 0.9757 | 0.0025 |

An IQR of 0.0005–0.0025 means the middle half of the universe is indistinguishable, and
a rank built on it is noise. **Raising the order does not rescue it** — it substitutes a
different defect: 6! = 720 patterns against ~2,156 windows is 3 windows per pattern, so
the estimate is undersampled and biased *downward* for pairs with less history. At order
6 the score would rank partly by history length, which on a universe with recent listings
is exactly the wrong thing to rank by. 15m behaves the same (IQR 0.0004 at order 3).

This is coherent with §5's PROC-20 result rather than surprising: bar-scale returns are
ordinally near-random for every pair, which is what an entropy near 1.0 says.

**Consequence:** `XS-3` should rank on `hurst_rs` (0.454–0.608, median 0.530),
`momentum_strength` (−0.21 to +1.11) and `realized_vol` (0.0015–0.0512), which do spread
across the universe. The estimator is kept — it is correct and matches the Rust tick-level
implementation (`ing-features/src/entropy.rs:373`), where it may still separate — but it
is not a Class-3 score. Cheap to have learned before building a ranking process on it.

### 7.4 Cross-sectional rank predictability (XS-3, 2026-08-07) — **Track C survives its kill test**

*Source: `processes/xs_rank_predictability.py` on the XS-1 archive — 177 perps × 90 days of
1h candles, 168-bar lookback, 83 **non-overlapping** daily rebalances, mean universe 175.6
pairs, 200 within-cross-section label permutations, BH-FDR across the three score families.
A signal result, filed here to keep the XS thread together.*

`THREE_CLASS_RESEARCH_PROPOSAL.md` §9 made this terminal: *"Track C stops if XS-3 finds no
score family significant after FDR."* It did not stop.

| score | rank-IC (1 d) | z | BH q | verdict |
|---|---|---|---|---|
| `xs_vol` | **−0.0690** | −8.37 | 0.007 | informative |
| `xs_momentum` | **−0.0387** | −4.56 | 0.007 | informative |
| `xs_hurst` | −0.0216 | −2.47 | 0.015 | fails z ≥ 3 |

**Both survivors are NEGATIVE, and the signs are the finding.** Low-volatility pairs
outperform high-volatility ones cross-sectionally, and recent winners *underperform* — the
"momentum" score is a cross-sectional **mean-reversion** signal with its sign inverted.
That independently reproduces §5's PROC-20 result, which found bar-scale momentum
*anti*-persistent in 34 of 36 cells by a completely different method. Two instruments,
one conclusion.

**Two alternative explanations were tested and one of them killed a result:**

1. **Return skew / Jensen — ruled out.** Re-running on log returns reproduces every IC to
   four decimals (−0.0387 / −0.0216 / −0.0690), as it must: Spearman is invariant to
   monotone transforms. The vol ranking is not an artifact of simple returns being
   right-skewed for volatile assets.
2. **Overlapping windows — a real defect, and it removed `hurst`.** A first 7-day pass
   spaced rebalances 24 h apart, so consecutive windows shared 86 % of their data. It
   reported `xs_hurst` at z −3.48, "informative". Re-spaced to non-overlapping (11
   windows), the same score gives **z −0.59, q 0.82** — nothing. This is precisely the
   defect that invalidated `funding_reversion` in §4.6 (95 %-overlapping windows,
   n_eff ≈ 84), reproduced and caught. `xs_vol` survives the same correction at z −3.76.
   **The headline 1 d results are non-overlapping by construction** (24 h horizon at 24 h
   spacing) and unaffected.

**What this does NOT establish**, and no one should read into it:

- **No costs.** Rank-IC is signal-level. A daily rotation across this universe crosses a
  median half-spread of 1.37 bps (§7.2) — ~2.7 bps round trip against an IC of 0.069 — so
  whether anything survives fees is `XS-6`'s question, not this one, and the taker
  arithmetic in §2 is not encouraging.
- **No capacity.** §7.2: the widest-spread pairs are nearly empty at the touch (XAI $20).
- **Survivorship, and it runs *against* the finding.** The archive holds currently-listed
  perps; the 55 delisted are absent, and failed coins are disproportionately high-vol. So
  the high-vol cohort here is missing its worst members, which biases *toward* zero — the
  true low-vol effect should be stronger, not weaker.
- **One 90-day window, one regime.** §6's arithmetic still applies: across-regime
  validation needs 6–24 months.

*Process defect found and fixed by this run:* `informative` was tested one-sided
(`z >= threshold`), inherited from the MI processes where the statistic is unsigned. Rank-IC
is signed, so every one of these results was initially reported as non-informative despite
z = −8.4. Now two-sided in |z|, with the direction carried as an explicit `polarity` — which
is also what PROC-1's compiler requires before it will emit an algorithm.

### 7.5 Rank persistence (XS-4, 2026-08-07) — **only `vol` has any**

*Source: `processes/xs_persistence.py` on the same panel — 177 perps, scores recomputed
every 24 h from a 168-bar (7-day) trailing window, rank autocorrelation to lag 30
rebalances (= 30 days).*

XS-3's ICs are necessary, not sufficient: a ranking that reshuffles before the next
rebalance is churn, paying the full spread to chase a signal already gone.

| score | ρ(1 d) | **ρ(7 d)** | ρ(30 d) | fitted half-life |
|---|---|---|---|---|
| `vol` | 0.968 | **0.691** | **0.509** | ~37.7 d |
| `momentum` | 0.879 | **−0.003** | 0.022 | 1.4 d |
| `hurst` | 0.615 | 0.018 | −0.007 | 1.5 d |

**The short-lag column is an artifact and must not be read as persistence.** Scores use a
168-bar lookback, so consecutive daily scores share 6/7 of their input. `momentum`'s
ρ(1 d) = 0.879 is *window overlap*, not memory. **Lag 7 is the first lag at which the two
windows are disjoint**, and it separates the scores completely: `vol` retains 0.691 and is
still at 0.509 after 30 days, while `momentum` and `hurst` sit at zero.

**Conclusion: `vol` is the only Class-3 score with genuine rank persistence, and it wins on
both axes** — the larger |IC| (0.069 vs 0.039, §7.4) *and* a ranking that survives weeks
rather than one overlapping window. A vol-ranked rotation would trade rarely, which is what
makes the cost question (`XS-6`) answerable at all; `momentum` has the smaller edge and no
memory once its window turns over, so a daily momentum rotation is churn by construction —
exactly the failure mode `XS-4` was specified to catch.

*Caveats:* the 37.7 d half-life for `vol` is an **extrapolation** — its autocorrelation
never actually crossed 0.5 inside the 30-lag window (`crossing=inf`), so read it as
"≥30 days, fitted 37.7". `vol`'s persistence is also the least surprising result in this
document: volatility clustering is the most robust stylised fact in finance, and finding it
is a sanity check on the instrument as much as a discovery about the venue. The criterion
as specified in `TASKS.md` — "half-life > cadence" — is necessary but weak: `momentum`
technically passes at 1.4 d against a 1 d cadence while having no disjoint-window memory at
all, so the meaningful quantity is the **ratio** (`vol` 37×, `momentum` 1.4×) and, more
honestly, ρ at a disjoint lag.

### 7.6 Capacity (XS-5, 2026-08-07) — **breadth and size trade off directly**

*Source: `xs/capacity.py` over 32 XS-8 sweeps × 177 pairs (~3 h of one day), joined to
30 days of candle dollar-volume. Spread figures are intraday-limited; ADV is not.*

**Touch notional is the wrong instrument for a daily rotation, and using it would have
produced a false verdict.** At an L1 floor of $10 k only **3 pairs** qualify — BTC, ETH,
SOL, i.e. exactly the three symbols the ingestor already covers, which would have read as
"Class-3 breadth is impossible". But L1 is resting size at one instant; a daily rebalance
works orders against a whole day's volume. Median touch across the universe is a few
hundred dollars while median **ADV is $330 k** (p95 $15.7 M).

Pairs supporting a given per-pair daily trade at **1 % participation of ADV**:

| size / pair | ≤1 bps | ≤2 bps | ≤5 bps | any spread |
|---|---|---|---|---|
| $1,000 | 49 | **117** | 156 | 162 |
| $10,000 | 37 | **52** | 57 | 58 |
| $100,000 | 10 | 10 | 11 | 12 |
| $1,000,000 | 5 | 5 | 5 | 5 |

**This is the Class-3 thesis meeting its constraint quantitatively.** The premise is
IR ≈ IC·√breadth — 150 pairs at modest IC beating 3 at high IC. That holds at **small
notional and nowhere else**: ~$1 k/pair keeps 117 pairs (≈$117 k deployed), $10 k/pair
keeps 52 (≈$520 k), and by $100 k/pair breadth has collapsed to 10 and the √N advantage
with it. Track C is capacity-viable, but as a small-notional wide-breadth strategy — which
is the appropriate shape for a first deployment in any case.

**The surviving score does not solve capacity for you.** `vol` (§7.4–7.5) correlates with
*spread* — corr(vol, half-spread) = **+0.397**, low-vol pairs median 0.601 bps vs high-vol
2.632 — which helps, since a rotation crosses the spread. But corr(vol, touch notional) =
**−0.092**: volatility says essentially nothing about depth. The low-vol cohort runs from
ETH at $365 k of touch to ICP at $13. So the liquidity gate has to be applied explicitly;
tilting to low vol does not implicitly select tradeable pairs.

*Design note:* this module deliberately mints **no thresholds** — the guardrail is "gates
imported, not invented", and there is no measured economics yet from which a spread ceiling
could be derived. It reports the curve; `XS-6` picks the operating point against measured
P&L. *Limits:* spread from ~3 h of one day (the sampler is still accruing), 1 %
participation is a convention not a measurement, and ADV itself is elevated for pairs in
the middle of a move.

### 7.7 Rotation OOS study (XS-6, 2026-08-07) — **0 of 6 survive; it fails on durability, not cost**

*Source: `exploration/xs_rotation_study.py`, **pre-registered** (criteria committed in
`f3eea78` before the run). 119 pairs admitted at ≤2 bps, daily top-k rotation on `vol`
rank, walk-forward 60/40, costs = each pair's measured half-spread + SSOT taker + slippage
via `load_costs()`. Artifact: `reports/xs_rotation_study.json`.*

| config | gross | cost | **net** | SR | SR_is | SR_oos | pos share | max-day |
|---|---|---|---|---|---|---|---|---|
| k=10 long-only | −19.47 % | 1.20 | −20.67 % | −3.02 | | | 0.48 | 0.32 |
| k=20 long-only | −19.05 % | 1.01 | −20.06 % | −2.60 | | | 0.47 | 0.37 |
| k=20 long-short | +8.73 % | 2.42 | **+6.31 %** | 0.85 | **−0.54** | +4.20 | 0.49 | **1.04** |
| k=40 long-short | +8.46 % | 1.91 | **+6.55 %** | 1.25 | **−0.27** | +5.04 | 0.49 | 0.66 |

**Verdict: NONE.** Every configuration fails the criteria declared beforehand.

**Three things this establishes, none of them the obvious one.**

1. **Cost is not the killer — and that is a genuine, validated prediction.** §7.5 said
   `vol`'s ≥30-day rank half-life implies low turnover; measured turnover is **0.17–0.49**
   against a theoretical maximum of 2.0, and costs consume only 1–2.7 % against 8.5 % gross.
   Every strategy refuted in §4.6 died paying full cost against a seconds-to-minutes signal;
   this one genuinely doesn't. The mechanism worked. It just isn't profitable.
2. **Long-only measures market beta, not the signal.** Its −19.5 % gross is the universe
   falling over the window; the cross-sectional signal only appears market-neutral
   (long-short +8.7 % gross). Any future rotation must be constructed neutral.
3. **The "OOS Sharpe 5.0" is the most misleading number in this document.** Read alone it
   looks outstanding. But **IS Sharpe is negative in every configuration** (−1.79 / −0.54 /
   −0.27) — the strategy lost money over the first 60 % of its own backtest and made
   everything in the last 40 %. That is a regime change, not validation, and criterion (e)
   catches it only because it tests the *ratio* (negative: −0.39, −7.75, −19.0) rather than
   the OOS level. Pre-registration is what stopped this from being written up as a win.

Supporting failures: positive-period share **0.49** (a coin flip) and single-day P&L
concentration up to **104 % of the total** — remove one day and the edge is gone. That is
`surprise_signal`'s §4.6 failure (87.6 % of edge from one day) reproduced on an unrelated
strategy.

**A cross-cutting pattern now visible.** §4.9's touch-maker grid failed on criteria (b)
day-consistency and (c) concentration. §7.4's `hurst` died to window overlap. XS-6 fails on
exactly (b), (c) and IS/OOS. **NAT's candidate strategies keep failing *durability*, not
edge size and not cost.** That suggests consistency should be the primary design objective
of the next candidate rather than a check applied at the end.

**Track C status:** the signal is real (§7.4), persistent (§7.5) and capacity-viable at
small notional (§7.6) — and still **not tradeable** on this evidence. Nothing promotes to
lifecycle DISCOVERED. *Limits:* 90 days is short — the 60/40 split leaves ~36 in-sample and
~24 out-of-sample rebalances, so a regime change between halves is unsurprising and §6's
6–24-month arithmetic applies. Funding accrual on held inventory is still not modelled
anywhere. A longer window is the one cheap thing that could change this verdict, and `XS-7`
is already accumulating it.

### 7.8 XS-6 post-mortem (XS-9, 2026-08-07) — **the breadth was an illusion; the beta was uncompensated**

*Source: diagnostic decomposition of §7.7's portfolio, then a beta-neutral rebuild. Same
119-pair universe, same 83 rebalances, same SSOT costs.*

**The strategy was never 40 bets.** Within-basket pairwise correlation is **0.433** (long)
and **0.323** (short), so 40 names behave as ≈ 40/(1+39·0.433) ≈ **2.2** effective bets.
Long-basket beta 0.81 vs short-basket 1.14 gives the "market-neutral" portfolio a
persistent **−0.33 beta tilt**, and its P&L is **0.802 correlated with a *static*
low-beta-minus-high-beta position**. Daily rebalancing was rearranging a position that was
80 % a standing factor bet — which is why IR = IC√BR overpredicted, why t = 0.49, and why
one day carried 104 % of P&L. One bet, 83 observations, fat tails.

**But that beta exposure earns nothing.** Measured over the same 83 rebalances:

| test | mean IC | t |
|---|---|---|
| raw vol → relative return | −0.0715 | −4.08 |
| **beta → relative return** | −0.0264 | **−1.01** |
| **vol \| beta → beta-neutral return** | −0.0650 | **−5.48** |

Beta does not predict relative returns, while the signal **survives neutralisation and
sharpens** (t −5.48 vs −4.08 raw). So the tilt is pure uncompensated variance sitting on
top of a real cross-sectional signal — an implementation defect, not a signal defect.
*(A refuted alternative, recorded so it is not retried: swapping the score to idiosyncratic
vol changes nothing, corr(total_vol, idio_vol) = 0.941.)*

**Rebuilding beta-neutral and score-proportional** (Grinold-optimal weights rather than
equal-weight top-k), criteria unchanged:

| | §7.7 top-k raw | **beta-neutral** |
|---|---|---|
| net / gross / cost | +5.49 / +7.39 / 1.90 % | **+7.77 / +8.90 / 1.12 %** |
| Sharpe (is / oos) | +1.06 (−0.42 / +4.79) | **+2.12 (+2.64 / +1.18)** |
| t | +0.50 | **+1.01** |
| positive share · max-day · turnover | 0.49 · 0.78 · 0.34 | **0.55 · 0.30 · 0.20** |
| \|net beta\| | 0.406 | **0.000** |

**4 of 6 pre-registered criteria now pass** — (a) Sharpe, (c) consistency, (d)
concentration, (f) cost stress — against 0 of 6 before. It still **fails (b) DSR and (e)
OOS/IS** (0.447), so **nothing promotes**. Note the failure changed character: IS Sharpe
went from −0.42 to +2.64, so the strategy is now profitable in *both* halves and (e) fails
in the ordinary overfit direction rather than the §7.7 regime-change direction.

**The power arithmetic moves the most.** Doubling the Sharpe quarters the data requirement,
since n ∝ 1/SR²: at SR 2.12 the daily Sharpe is 0.111, so t = 2 needs n = (2/0.111)² ≈
**325 rebalances ≈ 0.89 years**, down from 2.55. We have 83, so ≈ 8 more months of the
`XS-7` archive would settle it.

**The honesty caveat, which is not small.** XS-9's construction was designed *after* seeing
§7.7 fail, on the same 83 days. The mechanism is theory rather than search — beta does not
predict (t −1.01), therefore hedge it — which is a far stronger position than a parameter
sweep, and it cut turnover and cost too, which fitting does not usually do. But the
*magnitude* of the improvement is measured in-sample with respect to that design choice and
should be treated as an upper bound. The multiple-testing burden also rises: §7.7 declared
12 trials, and this is a 13th on the same window. The clean test is the next 8 months.

**First automated re-measurement (2026-08-07, XS-10) already moved the numbers.** The
trajectory tracker's first scheduled run re-derived the admitted universe from 2.4x more L2
data (14,165 snapshot rows vs ~4,700) and re-ran the same construction on the same 83
rebalances. **7 pairs churned** in and out of the >=2 bps admission (120 -> 119 admitted;
`INIT`/`MOVE`/`ZETA`/`ZORA` out, `DYDX`/`STX`/`SUPER` in), which moved gross **+8.90% ->
+8.175%**, Sharpe **2.12 -> 2.00**, and flipped criterion (d) from pass (0.300) to fail
(0.333) — so the tally is now **3 of 6**, not 4.

That is not a code change; it is the same strategy on a better-measured universe. It says
the §7.8 figures were sensitive to an admission set itself estimated from ~3 hours of
spread data, and it arrived on the tracker's *first* automated run — which is the argument
for having built it rather than leaving the re-run to memory. Treat 2.12 as the optimistic
end of a range whose lower end is not yet known.

*Driver defect found and fixed mid-study:* the first version read costs with
`costs.get("taker_bps", 4.5)` fallbacks. The SSOT is nested
(`costs["hyperliquid"]["taker_bps"] = 3.5`), so the lookup missed and **hardcoded literals
supplied the fees** — the exact guardrail violation behind §4.6's wrong-venue pricing,
reproduced by me in a new file. Now uses the `taker_bps()` accessor with no fallback: a
missing key raises. Costs were *overstated* by 1 bps/side, so the negative verdict was
conservative, and the corrected numbers are the ones tabled above.

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

### 7.9 Hysteresis bands (A5, 2026-08-07) — **cost saving real, net effect undecidable**

*Source: `execution/rebalance.py` applied to the §7.8 rotation; 83 rebalances, 119 pairs.
"edge" = trade to the no-trade boundary (Constantinides), "full" = trade fully or not at all.*

| band | mode | gross | cost | net | SR | turnover |
|---|---|---|---|---|---|---|
| 0.000 | — | 8.10 | 1.10 | 6.99 | 1.98 | 0.199 |
| 0.002 | edge | 7.25 | 0.67 | 6.58 | 1.91 | 0.121 |
| 0.005 | edge | 7.39 | 0.43 | 6.96 | 2.02 | 0.078 |
| 0.010 | edge | 10.13 | 0.23 | 9.90 | 2.07 | 0.041 |
| 0.010 | full | 12.58 | 0.48 | 12.10 | **2.99** | 0.087 |
| 0.020 | edge | 10.53 | 0.10 | 10.42 | 2.37 | 0.018 |

**The cost saving is real, mechanical and monotone**: turnover falls 0.199 → 0.018 and cost
1.10 % → 0.10 %, exactly as intended. **The net effect is not decidable on this data.** A
band changes *which positions are held*, and gross swings 7.25 → 12.58 — non-monotone, and
several times larger than the ~1 pp of cost being saved. At 83 rebalances that variation is
noise, so **selecting a band here would be fitting it**, which is why the apparent winner
(band 0.010 "full", SR 2.99) is reported and **not adopted**: 7 configurations on one window
is the §4.6 pattern. The band must come from cost/edge a priori, not from this table.

*Two defects of mine, found by disbelieving a good number.* The first table showed cost
falling with **gross unchanged at 8.10 across every band** — free money, therefore a bug:
`run_rotation` priced gross on the *target* weights while the portfolio held the
band-adjusted ones. Identical when band = 0, so it stayed latent until A5 and is now
regression-tested. The second: `band_from_cost` returns a **dimensionless** cost/edge ratio
(~1.4 at NAT's costs), not a position band; applied directly to a unit-gross book of ~120
names whose typical weight is ~0.008, it means *never trade* — a units error that would
present as a strategy mysteriously ceasing to rebalance. It now takes `position_scale`.

**TWAP/VWAP slicing ships as a primitive with no performance claim.** Slicing exists to
reduce market impact; NAT's cost model is spread + fee + slippage per unit turnover with
**no impact term**, so it measures as exactly zero here. That is not a win being left on the
table — it is unpriceable until `X-3` has fill data.

### 7.10 Wide-pair breakeven screen (B-5a, first run 2026-08-08) — **the hypothesis survives, capacity nearly kills it; n = 1 hour**

*Source: `scripts/xs/breakeven.py` via `exploration/b5a_breakeven_study.py` over
`xs.capacity.aggregate_l2` on **~18 XS-8 sweeps across ~1.5 hours** (2026-08-08, 2,751
snapshot rows). **177 of 177 pairs** cleared the ≥12-sweep floor — an earlier read at 12
sweeps had only 127, and the missing 50 were request failures that resolved as sampling
continued, so the coverage bias that read carried is gone. Rebate from the COST-5 ladder
(`rebate_t2`, +0.2 bps).
**This is one hour of one day, not the "several days" B-5a asks for — recorded as a first
reading, not a verdict.***

**What was actually tested, and why it is the decisive question.** §4.11 established the
maker rule `posting is +EV ⟺ half_spread + rebate > E[adverse | fill]`, and §7.2 then showed
NAT has only ever tested it on the extreme tight tail of its own venue (169/177 pairs are
wider than BTC). The tempting inference — wide pairs cover adverse selection — rests entirely
on assuming `E[adverse|fill]` stays at BTC's measured 0.228 bps as the spread widens. It
should not: spreads are wide *because* makers price toxicity and inventory risk into them.
So the screen refuses to emit a survivor count and instead reports the exponent at which the
verdict flips, parameterising `E[adverse|fill](h) = A_btc·(h/h_btc)^β` pinned through BTC's
measured point:

| β | E[adverse] at the median pair | survivors |
|---|---|---|
| 0.00 (constant — the optimistic reading) | 0.228 bps | 177 / 177 |
| 0.50 | 0.961 bps | 177 / 177 |
| **0.75** | 1.976 bps | **29 / 177** |
| 1.00 (proportional — the pessimistic reading) | 4.062 bps | 6 / 177 |

**Median β\* = 0.698**, and it held at 0.696 on the shorter 127-pair read. The whole hypothesis is now one number: *does adverse selection scale
more slowly than `h^0.70`?* One tick-data measurement on one wide pair settles it — that is
`B-5b`, and it is a measurement rather than a program of simulations.

**Capacity is the harder blade.** Joint wide-AND-deep, via `XS-5`'s floors:

| touch floor | admitted | of which wide (>1 bps) | survive @ β=0.75 |
|---|---|---|---|
| $0 | 177 | 131 | 29 |
| $500 | 33 | 18 | 15 |
| $1,000 | 18 | 10 | 10 |
| **$5,000** | **4** | **0** | 2 |

So the tradeable version is **≈10 wide pairs at roughly $1,000 of touch size**, conditional on
β < 0.75. That is the capacity ceiling `B-5b` must justify before any simulation is worth
running.

**A genuine surprise: spread and depth are uncorrelated** (Spearman −0.107, p = 0.16, n=177).
It is not that wide pairs are especially thin — **the whole universe is thin at the touch**.
Median touch notional by spread quartile: $391 (tightest 25 %) · $33 · $45 · $121 (widest).
The only pairs carrying real size are BTC ($268 k), ETH ($172 k), SOL ($27 k) and XRP ($9.7 k) — all tight, and all already studied.
This weakens §7.2's "wide pairs are nearly empty" reading: emptiness is universe-wide, not a
property of width.

**Cross-check:** BTC measured **0.0769 bps** here against §4.11's 0.0832 anchor — a third
independent instrument agreeing on the same quantity.

**How to read the optimistic row.** "177/177 survive at β ≤ 0.5" is *not* evidence; it is the
assumption restated, and it is exactly the trap §7.2 named. The defensible prior is β nearer 1
than 0, where 6 pairs survive.

*Limits:* ~18 sweeps over ~1.5 hours of one weekday; no intraday, weekday or regime variation;
quoted spreads are not fill economics; the per-pair β\* is only interpretable
as robustness for pairs genuinely wider than the anchor (near `h ≈ h_btc`, `ln(h/h_btc) → 0`
and β\* explodes — BTC's −2.48 and SOL's −0.65 are artifacts, not scores). The survivors-by-β
table is the sound output.

### 7.11 The `regime_` category is dead in production (2026-08-09) — **and it silently hollowed out A-1**

*Source: per-day scan of `regime_divergence_1h` finiteness across all 75 day-directories in
`data/features/`, one mid-day file per day, cross-checked against `regime_absorption_1h`.
Found while attempting to run `A-1` through `nat process run`.*

**`regime_divergence_1h` is 100 % NaN in every file written since 2026-07-26** — 0 finite
values in 1,444,319 tick rows over the recent window. `regime_absorption_1h` is dead alongside
it, so this is the whole optional `regime_` category (23 features), not one column.

| last ALIVE | outage | first DEAD | dead since |
|---|---|---|---|
| **2026-07-18** | 07-19 → 07-25 (no files) | **2026-07-26** | 13 consecutive data-days |

The break is bracketed by a 7-day ingestion outage, so the *transition* is unobservable in the
data: the category was alive going into the gap and dead coming out. It has flapped before —
46 of 75 days alive, with alternating runs since 2026-04-19 — so this is a recurrence, not a
first occurrence.

**The cause is not determinable from the repo, and the freeze forbids finding out.** The two
commits nearest the boundary are `OPS-1` (connect-timeout) and `OPS-2` (per-symbol crash-restart
supervision), both **2026-07-27 — after the first dead file**, so neither is established as the
trigger. One hypothesis worth testing when contact is allowed: `regime_` needs 1 h/4 h/24 h
rolling windows (`regime/mod.rs:13`), so a supervisor that restarts symbol tasks frequently
would reset those accumulators and hold the whole category at NaN indefinitely — a
warmup-starvation failure that presents exactly as this does. Recorded as a hypothesis, not a
diagnosis; confirming it requires inspecting the running ingestor, which the **su-35 freeze
forbids**.

**Consequence for §5.1 — A-1 had at most four usable days, not 25.** A-1 conditions the fast
signal on `regime_divergence_1h`; its stated window is 2026-07-14→08-07, and within that
window only **07-15, 07-16, 07-17 and 07-18** carry a live slow feature. Every row from 07-26
onward drops out of `valid` because the gate input is NaN, silently. That is consistent with
what A-1 reported — `n_days` of 0–3 against `min_days = 3`, every cell `insufficient_days` or
`non_durable`, `frac_days_informative = 0.00` — but it means the "25 days" framing in §5.1's
limits paragraph describes the *requested* window, not the *usable* one.

An independent run on the alive window (2026-06-21→07-25, 243 bars, 237 with both features
finite) reproduces the direction and not the strength: **0 of 9 cells informative, 7 of 9 lifts
negative**, with the three largest-magnitude cells all favouring *disagreement* (BTC 1 d: agree
−0.035 vs disagree +0.259; ETH 4 h: +0.027 vs +0.217; SOL 4 h: −0.123 vs +0.150; SOL 4 h z
−2.64, which would not survive BH across nine cells and points against the claim in any case).

**So the A-1 half of §5.1 is better stated as "no support, on a test too underpowered to
settle it" than as "refuted."** The direction is consistent across two independent windows and
nothing supports the original claim — but a durability verdict on ≤4 day-folds is not a
refutation, and A-1's own process refuses to issue one. The A-2 half is unaffected: it does not
use `regime_` at all.

**Three defects in the A-1 unit itself, all found by the same attempt:**

1. **Its registered defaults cannot run.** `AgreementGateEval` declares `data_level = "bars"`
   but defaults to `slow="regime_divergence_1h"` and `fast="alg_mp_dev_ema"`. Bar frames carry
   `{col}_{agg}` (`regime_divergence_1h_last`), and **no `alg_` column exists in any feature
   file** — algorithm outputs are computed at runtime and never persisted. `nat process run
   agreement_gate_eval` therefore fails on its own defaults. The 15 planted tests pass because
   they construct frames with names of their own choosing, so the planted layer and the
   real-data path never met — the `real-parquet smoke before commit` step, skipped.
2. **Unrunnable reports as success.** With no horizon clearing `min_obs`, the loop `continue`s
   for every horizon and the run returns `n_tested: 0, error: null` — a clean exit with nothing
   in it. Same shape as the conformance test that silently validated a subset (XS-10).
3. **The loader cap is invisible and binding.** `config/processes.toml` sets
   `max_memory_mb = 4000`, which truncated a 35-day request to **78 bars** — below
   `min_obs = 100`, so the process skipped silently. The same window at a 16 GB budget yields
   243 bars. `nat process run` exposes no memory flag.

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
