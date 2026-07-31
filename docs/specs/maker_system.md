# Maker System — Execution Doctrine, Combiner Contract & Algorithm Classes

**Status:** SPEC (2026-07-31). **Owner:** Onat. **Empirical basis:** `research/FINDINGS.md`
§1–§5, §4.5–4.9. **Contracts:** `contracts/algorithm.md`, `contracts/process.md`.
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
do. Rationale: §2 — the taker path is arithmetically dead at fees for the fast signal; the
fee-free-equivalent path is the maker rebate, and its enemy is adverse selection, not fees.

**Fee tiers are SSOT state, never assumptions.** The HYPE-staking discount enters as a
`[hyperliquid_staked]` tier in `config/costs.toml`; every experiment reports which tier it
priced. The maker margin (±0.01–0.04 bps/posting, §4.7) is decided by hundredths of a bp —
fee-tier modeling is first-class here.

## 2. Combiner feature contract — one representative per orthogonal axis

Orthogonality is operational: different **mechanism × horizon × role**, verified by
interaction-information (PROC-3) and correlation dedup before any pair is believed distinct.
§1 measured that the 29 fast-directional features are one correlated block — the contract is
therefore *one representative per axis*:

| Role | Axis | Representative | Horizon | Basis |
|---|---|---|---|---|
| Fast direction | book pressure | `alg_mp_dev_ema` (HF1) | 1 s–1 min | IC 0.45 @1–5 s (§1); +0.14/+0.24 gated @50 t |
| Slow bias | regime drift | `regime_divergence_1h` + `trend_hurst_300` | 30 min–3 h | 0.21 @100 min — only measured slow grower (§1) |
| Reversion anchor | oscillation | VWAP/band deviation + spectral band power | 10 s–5 min | IC 0.12–0.25 (§1, §4.5); IC lives in 0.005–0.1 Hz (§5) |
| Carry posture | positioning | funding rate/z (bias tilt only) | hours | §4.6: never standalone (n_eff, one-sided regime) |
| Gate — toxicity | informed flow | VPIN percentile | pull | Sharpe lift on 3/3 symbols (§4.5) |
| Gate — predictability | disorder | `ent_book_shape` | allow | +22 % IC in low-entropy quintile, replicated (§5) |
| Gate — fill quality | queue economics | A4 EV (capture > causal adverse est.) | per-posting | flips per-fill sign (§4.9) |
| Sizing only | vol & liquidity | `vol_returns_5m`, Kyle λ, touch depth | continuous | vol has ZERO directional IC (§1) — must never pick a side |

**Maker translation of the combiner:** slow bias selects the preferred side → fast direction
times the queue join → gates veto postings → vol/liquidity size quotes and spacing.
**Agreement-gating** (quote only when fast and slow agree) is mandatory: it is the one
structure with measured conditional-IC *above* unconditional (§5) — the anti-adverse-selection
core. Combined targets are judged by the platform gates (Q5: conditional-IC > 0.15 at the
priced fee tier), through the PROC surface + FDR, pre-registered.

## 3. Class 1 — directional bias makers

Monetize *persistent* microstructure pressure: quote the favored side at/near the touch,
capture spread + rebate + drift alignment.

- **Signal:** combiner output (§2) → side preference + join timing.
- **Quoting:** touch-pegged (width is never derived — §4.8 killed spread-widening);
  reservation-skew for inventory (HF5 quoter, skew only).
- **Gates:** VPIN, entropy, A4 EV — all veto-capable per posting.
- **Inventory:** soft bound `q_soft` → reduce-only; hard `q_max` → doctrine §1.
- **Status:** measured baseline exists (§4.9: EV-gated per-fill +0.67 bps but day-consistency
  FAIL). **Blocked for economics claims on fill data** (F-task L1/side-volume columns or T0b
  shadow quoting); signal/gating layers buildable and testable now.

## 4. Class 2 — regime-conditional oscillation harvesters

Monetize *anti-persistent* oscillation: symmetric ladder around a fair-value center, earn the
amplitude. (LF7 is the founding member.)

- **Admission (the class trades only when the world oscillates):** `hurst < 0.5 − δ`,
  spectral power concentrated in the measured 10–200 s band, OU half-life in [τ_min, τ_max]
  (record: 5–7 s), VPIN gate open. Admission comes from the regime classifier (GMM 5D already
  contains hurst/vol/λ/VPIN).
- **Geometry from the spectrum, never swept:** center = rolling VWAP or HF1 EMA;
  level spacing `k·σ_band` at the dominant period (LF7 prior: capture at k≈2.0–2.5, adverse
  ≤1.5); ladder depth bounded by `q_max`; every level passes the A4 EV gate.
- **Lifecycle:** fill at level → opposite quote at center/next band (the round trip is the
  trade). **Regime flip is the kill condition:** hurst crosses 0.5 or band power collapses →
  pull the grid, passive unwind, taker per doctrine §1 only. Trend days are the tail risk —
  the concentration criterion below exists for exactly this failure.

## 5. Regime router

One classifier state routes capital: persistent regime → Class 1; anti-persistent → Class 2;
toxic/disordered/unknown → quote nothing. The router is itself gated (hysteresis on regime
transitions; no thrash). This is the two-state architecture the GMM was trained for.

## 6. Acceptance criteria — pre-registered, imported from the record

Any configuration of either class **survives** only if, on a multi-day pre-registered run at
the priced fee tier (criteria declared in the driver before results, §4.9 discipline):

- (a) pooled per-fill EV > 0 · (b) positive-day share ≥ 0.55 · (c) no single day > 30 % of
  total PnL · (d) verdict sign stable under fill-proxy sensitivity (`l1_fraction` extremes)
- Signal-layer claims additionally pass the PROC gates (null-calibration z ≥ 3, BH-FDR q ≤ .05).
- **No live capital before G8 + healthy kill-switch** (unchanged, non-negotiable).

## 7. Build order & dependencies

1. **COSTS:** add `[hyperliquid_staked]` tier to `costs.toml`; re-run §4.7/§4.9 grids at it
   (cheapest experiment; re-prices everything).
2. **Combiner revalidation** (Class-1 signal layer): multi-day, pre-registered, `nat oos
   --window` + DSR — the §5 numbers are 2-day OOS and must not be trusted until this passes.
3. **Class-2 signal layer** (LF7 + spectral admission): band geometry from the `spectral`
   process; planted tests on spacing/admission; economics deferred to (4).
4. **F-task (blocking for all fill-economics claims):** ingestor emits L1 queue sizes +
   per-tick side volume — a feature-vector/schema change: **plan first** per guardrail.
   Alternative unblocker: shadow quoting on T0b once provisioned.
5. Router + paper integration, behind the lifecycle ladder and G8.

## 8. Provenance

Every number cited here traces to `research/FINDINGS.md` (§1 signal record, §2 adverse
selection, §3 robustness/drift, §4.5 VWAP/VPIN, §4.6 kill gate, §4.7 queue EV, §4.8 spread
verdict, §4.9 touch-maker experiment, §5 combiner/entropy/spectral). Anything not traceable
to a section there is hypothesis, and says so.
