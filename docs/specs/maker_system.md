# Maker System — Execution Doctrine, Combiner Contract & Algorithm Classes

**Status:** SPEC v2 (2026-07-31; v1 same day — adds Class 3 + per-class process definitions).
**Owner:** Onat. **Empirical basis:** `research/FINDINGS.md` §1–§5, §4.5–4.9.
**Contracts:** `contracts/algorithm.md`, `contracts/process.md`.
**Related:** LF7 (`research/new/vwap_sd_channel.txt`), GAP specs 04/07, HF1/A4/HF5 modules.

The post-Q4 design: every shipped taker strategy was refuted (§4.6); the surviving assets are
the imbalance/microprice signal record (§1–§3), the maker instruments (HF1 center, A4 EV gate,
queue sims), and the measured posture knowledge (§4.7–§4.9). This spec freezes the system
those facts point at — **before** the next build phase writes code against it.

---

## 1. Execution doctrine — maker-only, taker as state transition

Quotes are the only strategy. A taker order is never an alpha decision; it is an **emergency
state transition**, permitted in exactly three cases:

| Trigger | Condition | Action |
|---|---|---|
| Inventory emergency | `\|q\| > q_max` AND reduce-only quoting unfilled for `T_unwind` | taker-flatten the excess |
| Kill-switch / staleness | gate inputs degraded, watchdog fired, kill-switch open | flatten, stop quoting |
| Episode end | session close / shutdown | terminal liquidation |

All taker transitions are charged at the SSOT taker cost (`load_costs()`), as the sims already
do. Rationale: §2 of FINDINGS — the taker path is arithmetically dead at fees for the fast
signal; the fee-free-equivalent path is the maker rebate, and its enemy is adverse selection.

**Fee tiers are SSOT state, never assumptions.** The HYPE-staking discount enters as a
`[hyperliquid_staked]` tier in `config/costs.toml`; every experiment reports which tier it
priced. The maker margin (±0.01–0.04 bps/posting, §4.7) is decided by hundredths of a bp.

*Shipped 2026-08-03 (X-1, `research/FINDINGS.md` §4.10).* The ladder is live
(wood 5 % → diamond 40 %), tier `none` until HYPE is actually staked, `tier_summary()`
stamped into every artifact. **The decisive venue mechanic: staking discounts apply to fees
paid, not to maker rebates** ([venue docs](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees)) —
so the discount cannot touch the maker line's income, only the taker legs. The maker
economics are therefore *tier-invariant*, and the fee tier is no longer a live hypothesis
for rescuing them.

## 2. Combiner feature contract — one representative per orthogonal axis

Orthogonality is operational: different **mechanism × horizon × role**, verified by
interaction-information (PROC-3) and correlation dedup before any pair is believed distinct.
FINDINGS §1 measured that the 29 fast-directional features are one correlated block — the
contract is therefore *one representative per axis*:

| Role | Axis | Representative | IC (provenance) | Horizon |
|---|---|---|---|---|
| Fast direction | book pressure | `alg_mp_dev_ema` (HF1) | 0.40–0.47 measured (§1); +0.14/+0.24 gated @50 t | 1 s–1 min, τ½ 15–30 s |
| Slow bias | regime drift | `regime_divergence_1h` + `trend_hurst_300` | 0.15–0.21 measured (§1 — only slow grower) | 30 min–3 h |
| Reversion anchor | oscillation | VWAP/band dev + spectral band power | 0.12–0.29 measured (§1, §4.5); IC in 0.005–0.1 Hz, OU τ½ 5–7 s (§5) | 10 s–5 min |
| Carry posture | positioning | funding z (bias tilt ONLY) | ~0.05–0.1 **hypothesis** (standalone refuted §4.6) | hours |
| Gate — toxicity | informed flow | VPIN percentile | gate: Sharpe lift 3/3 symbols (§4.5) | pull |
| Gate — predictability | disorder | `ent_book_shape` | gate: +22 % IC lift, replicated (§5) | allow |
| Gate — fill quality | queue economics | A4 EV (capture > causal adverse est.) | veto: flips per-fill sign (§4.9) | per posting |
| Sizing only | vol & liquidity | `vol_returns_5m`, Kyle λ, touch depth | 0.29–0.35 on \|r\|, **exactly 0 on sign** (§1) | continuous |

All measured ICs: Spearman rank vs forward returns, overlap-subsampled, p<.01, survived §3's
battery (per-day stability, intraday/vol-regime splits, bootstrap CIs, rank-order structure).

**Maker translation:** slow bias selects the preferred side → fast direction times the queue
join → gates veto postings → vol/liquidity size quotes and spacing. **Agreement-gating**
(quote only when fast and slow agree) is mandatory: the one structure with measured
conditional-IC *above* unconditional (§5) — the anti-adverse-selection core.

## 3. Class 1 — directional bias makers

Monetize *persistent* microstructure pressure: quote the favored side at/near the touch.

- **Signal:** combiner output (§2) → side preference + join timing.
- **Quoting:** touch-pegged (width never derived — §4.8); reservation-skew for inventory.
- **Gates:** VPIN, entropy, A4 EV — all veto-capable per posting.
- **Inventory:** soft bound `q_soft` → reduce-only; hard `q_max` → doctrine §1.
- **Status:** §4.9 baseline measured (EV-gated per-fill +0.67 but day-consistency FAIL);
  economics claims **blocked on fill data** (F-task or T0b shadow quoting); signal/gating
  layers buildable now.

## 4. Class 2 — regime-conditional oscillation harvesters

Monetize *anti-persistent* oscillation: symmetric ladder around a fair-value center, earn the
amplitude. (LF7 is the founding member.) Rests on the **most persistent measurable in the
data** — volatility/regime state (vol-IC 0.29–0.35 while its direction-IC is zero, §1) — so
the class predicts *amplitude*, never direction.

- **Admission:** `hurst < 0.5 − δ`, band power concentrated in the measured 10–200 s band,
  OU half-life in [τ_min, τ_max] (record: 5–7 s), VPIN gate open. From the GMM 5D state.
- **Geometry from the spectrum, never swept:** center = rolling VWAP or HF1 EMA; spacing
  `k·σ_band` at the dominant period (LF7 prior: capture at k≈2.0–2.5, adverse ≤1.5); ladder
  depth bounded by `q_max`; every level passes the A4 EV gate.
- **Lifecycle:** fill at level → opposite quote at center/next band. **Regime flip = kill:**
  hurst crosses 0.5 or band power collapses → pull grid, passive unwind, taker per §1 only.

## 5. Class 3 — cross-sectional rotation (the universe selector)

Continuously measure every pair's entropy, momentum strength, and vol regime **against its
own history**, rank cross-sectionally, and allocate to the top pairs — Class 3 decides
*where* Classes 1/2 run and with how much capital. Breadth is the point: IR ≈ IC·√breadth —
150 pairs × modest xs-IC beats 3 pairs × high IC.

**Two-tier architecture (no schema changes — verified against the codebase):**
- **Tier W (wide, cheap):** REST 1 m candles for the **full Hyperliquid perp universe**
  (~150+ pairs) via the existing `scripts/data/fetch_candles.py` (multi-symbol, incremental,
  → `data/candles/`). Per pair, bar-level: permutation entropy of 1 m returns, momentum
  strength (slope × R², Hurst), vol regime — each as percentile/z vs the pair's own history.
- **Tier D (deep, existing):** the 100 ms tick stack + Class 1/2 execution runs only on the
  selected pairs. `symbols.toml` is already arbitrary-N (`config.rs:195`, `main.rs:251`);
  each deep symbol costs ~170–225 MB/day + one WS connection. Rotation = editing the SSOT
  symbol list with **warm-up discipline**: select → ingest → warm features → only then route
  capital. Cross-symbol features are N-agnostic already (`cross_symbol.rs`).

**Per-pair scores (dual, because the router needs a mode, not just a rank):**
- `fit_C1 = f(momentum_strength_pct, hurst > 0.5, low entropy_pct)` — persistent candidates
- `fit_C2 = f(hurst < 0.5, band_power_pct, OU τ½ in range)` — oscillatory candidates

**Admission (xs_capacity_gate):** spread ceiling, depth floor, volume floor (SSOT-costed
tradability) + minimum history for stable percentiles. Untradeable tails never reach ranking.

**Allocation: top-k weighted** (k ≈ 3–5): weights ∝ rank-score × inverse-vol, rebalanced
slowly (daily+) with hysteresis (minimum holding period, max turnover) — rank churn is the
class's main failure mode, gated by `xs_persistence` (below).

## 6. Regime router (three levels)

`Class 3 → (pair, mode, budget)` → per pair: persistent regime → Class 1; anti-persistent →
Class 2; toxic/disordered/unknown → **quote nothing**. Hysteresis at both levels (pair
rotation and mode switching); no thrash. This is the architecture the GMM was trained for.

## 7. Process definitions per class (the measurement layer)

Every class claim is established by registered `EvaluationProcess` units (null-calibrated
per PROC-12, BH-FDR per PROC-13, pre-registered where capital-relevant). Existing + new:

**Class 1:**
- `conditional_predictability` *(exists)* — MI(f; label | regime) per bucket: which gate
  states carry the edge.
- `horizon_label_scan` *(exists)* — (horizon × barrier geometry × regime) surface.
- `agreement_gate_eval` *(new)* — conditional IC of the fast signal GIVEN slow-bias
  agreement vs disagreement, as a standing FDR-gated eval (promotes the §5 pilot result to a
  monitored fact).

**Class 2:**
- `spectral` *(exists)* — PSD / Hurst / OU half-life / band-IC per symbol.
- `oscillation_admission_eval` *(new)* — does "admitted now" predict "oscillatory next"?
  Null-calibrated forward persistence of the admission state + forward band-capture proxy.
- `band_geometry_scan` *(new)* — (k × dominant-period × regime) capture-vs-adverse surface;
  reuses the `horizon_label_scan` pattern + FDR. Geometry is *read off this surface*, never swept.

**Class 3 (cross-sectional — a new process kind):**
- `xs_rank_predictability` *(new)* — per rebalance interval: Spearman rank-IC of the scanner
  scores vs relative forward performance across the admitted universe; permutation null =
  shuffle pair labels; FDR across score variants.
- `xs_persistence` *(new)* — rank autocorrelation half-life per score; **must exceed the
  rebalance cadence** or the rotation is churn by construction.
- `xs_capacity_gate` *(new)* — data-driven tradability floors (spread/depth/volume), SSOT-priced.
- Infra note: these need a `candles` data level + multi-symbol loading in the process runner —
  a small framework extension, flagged as its own task before implementation.

## 8. Acceptance criteria — pre-registered, imported from the record

Any configuration **survives** only if, on a multi-day pre-registered run at the priced fee
tier (criteria declared before results, §4.9 discipline):

- (a) pooled per-fill EV > 0 · (b) positive-day share ≥ 0.55 · (c) no single day > 30 % of
  total PnL · (d) verdict sign stable under fill-proxy sensitivity.
- Signal-layer claims additionally pass the PROC gates (null z ≥ 3, BH-FDR q ≤ .05).
- Class-3 additionally: xs rank-IC significant after FDR AND rank half-life > rebalance cadence.
- **No live capital before G8 + healthy kill-switch** (unchanged, non-negotiable).

## 9. Build order & dependencies

1. ~~**COSTS:** `[hyperliquid_staked]` tier in `costs.toml`; re-run §4.7/§4.9 grids at it.~~
   **DONE 2026-08-03 (X-1, FINDINGS §4.10): no cell flips at any rung** — staking discounts
   do not reach maker rebates, so the maker line is tier-invariant. The live cost question it
   surfaced instead: the SSOT's 0.2 bps maker *rebate* presumes an untested volume tier, while
   the venue's base perp maker rate is a +1.5 bps *fee* — quantify that before any maker
   economics claim.
2. **Class-3 Tier-W scanner** — *fully buildable now*: needs only REST candles (no ingestor,
   no streak, no fill data). `fetch_candles.py` universe extension + `xs_scan` + the three
   xs processes. The highest-value data-independent unit on the board.
3. **Combiner revalidation** (Class-1 signal layer): multi-day, pre-registered, `nat oos
   --window` + DSR — §5's 2-day numbers must not be trusted until this passes.
4. **Class-2 signal layer** (LF7 + spectral admission): geometry from `band_geometry_scan`.
5. **F-task (blocks all fill-economics claims):** L1 queue sizes + per-tick side volume —
   schema change: **plan first** per guardrail. Alternative: T0b shadow quoting.
6. Router + paper integration, behind the lifecycle ladder and G8.

## 10. Provenance

Every number here traces to `research/FINDINGS.md` (§1 signal record, §2 adverse selection,
§3 robustness/drift, §4.5 VWAP/VPIN, §4.6 kill gate, §4.7 queue EV, §4.8 spread verdict,
§4.9 touch-maker experiment, §5 combiner/entropy/spectral) or to code refs verified 2026-07-31
(`fetch_candles.py`, `config.rs:195`, `main.rs:251`, `cross_symbol.rs`). Anything not
traceable is hypothesis, and says so.
