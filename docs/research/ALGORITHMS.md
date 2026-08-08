# Algorithm Catalogue

> ⚠️ **PERFORMANCE CLAIMS REFUTED 2026-07-30** (Q4 kill gate — `FINDINGS.md` §4.6). The tier
> assignments and P&L below were measured at the wrong venue cost (1.61 bps Binance VIP9;
> Hyperliquid reality is ~11 bps all-in) through a harness that never ran each algorithm's own
> entry logic. At SSOT cost every "Tier 1/2" algorithm is deeply net-negative; all five were
> REJECTED in the signal lifecycle. The catalogue is retained as a mechanism/reference document —
> treat every number as historical, not current.

**Last updated:** 2026-07-30 (refutation banner; performance tables are the 2026-05-23 record)
**Test conditions:** Walk-forward OOS, 3-day training window, P20/P80 z-score entry, 100min horizon, 1.61 bps RT fee (Binance VIP9 — **wrong venue, see banner**)
**OOS window:** 13 dates (2026-05-07 to 2026-05-23)
**Symbols:** BTC, ETH, SOL | **Bars:** 5min from 100ms ticks

---

## Tier 1 — Deployable (net positive all 3 symbols) — **REFUTED, see banner**

### 1. jump_detector — Total +23,199 bps

**Post-jump mean-reversion via Lee-Mykland nonparametric test.**
Detects jumps using bipower variation volatility, then enters mean-reversion after the jump dissipates. Low jump fraction = stable regime = long.

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL (bps) | Win Rate | Max Daily Loss |
|--------|--------|--------------|--------|-----------------|----------|----------------|
| BTC    | 1,678  | +1.03        | 1.6    | +1,722          | 54%      | -1,981         |
| ETH    | 1,678  | +6.47        | 6.2    | +10,861         | 62%      | -3,412         |
| SOL    | 1,678  | +6.33        | 6.2    | +10,616         | 69%      | -3,311         |

- **Primary feature:** `alg_post_jump_reversion`
- **Signal polarity:** low_long (low jump ratio = stable = long)
- **Source:** `scripts/algorithms/jump_detector.py`

#### Variant: jump_detector_v2 — [PRELIM], no OOS results yet

Statistical upgrade of the winner, kept as a separate registered algorithm so the v1 baseline
stays frozen for A/B: EVT (Gumbel) detection threshold per Lee-Mykland 2008 asymptotics
(≈7.2 at 1-day blocks vs legacy fixed 3.0), staggered (skip-one) bipower variation
(consecutive-jump masking + bid-ask-bounce robustness), post-shock vol floor, directional
reversion routing (`alg_jd2_rev_up`/`alg_jd2_rev_down`), magnitude-adaptive reversion horizon.
Planted 17/17 green; real-parquet smoke green (16.7M ticks). **Maturity: PRELIM** — promotion
blocked on the v1-vs-v2 A/B over the next ≥30-clean-day window (Q3).
- **Primary feature:** `alg_jd2_reversion`
- **Source:** `scripts/algorithms/jump_detector_v2.py` · config `[jump_detector_v2]`

---

### 2. 3f_liquidity — Total +16,028 bps

**Equal-weight z-score composite of spread + depth + VWAP deviation.**
The baseline MF liquidity signal from Experiment Report 1. Strongest on BTC.

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL (bps) | Win Rate | Max Daily Loss |
|--------|--------|--------------|--------|-----------------|----------|----------------|
| BTC    | 950    | +5.58        | 9.2    | +5,302          | 62%      | —              |
| ETH    | 915    | +7.83        | 7.8    | +7,162          | 62%      | —              |
| SOL    | 954    | +3.74        | 3.2    | +3,564          | 62%      | —              |

- **Primary feature:** z-score of `mf_spread_bps + mf_depth_imbalance + mf_vwap_deviation`
- **Source:** `scripts/alpha/paper_trader.py`

---

### 3. funding_reversion — Total +14,459 bps

**Funding rate z-score mean-reversion with saturation.**
Crypto-native signal. Enters opposite to extreme funding rates using `-sign(z) * min(|z|/z_entry, 3)/3`. Dominates ETH.

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL (bps) | Win Rate | Max Daily Loss |
|--------|--------|--------------|--------|-----------------|----------|----------------|
| BTC    | 1,678  | +0.26        | 0.4    | +429            | 38%      | -2,327         |
| ETH    | 1,678  | +6.12        | 6.1    | +10,265         | 54%      | -2,629         |
| SOL    | 1,678  | +2.24        | 1.7    | +3,766          | 54%      | -3,786         |

- **Primary feature:** `alg_funding_signal`
- **Signal polarity:** high_long
- **Source:** `scripts/algorithms/funding_reversion.py`

---

### 4. optimal_entry — Total +13,679 bps

**SPRT on Kalman OU-filtered L1 imbalance innovations.**
Sequential hypothesis test for entry timing. Accumulates log-likelihood ratio until decision boundaries are crossed.

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL (bps) | Win Rate | Max Daily Loss |
|--------|--------|--------------|--------|-----------------|----------|----------------|
| BTC    | 1,678  | +0.90        | 1.1    | +1,504          | 46%      | -2,327         |
| ETH    | 1,678  | +5.89        | 5.2    | +9,877          | 62%      | -3,645         |
| SOL    | 1,678  | +1.37        | 1.0    | +2,298          | 54%      | -4,762         |

- **Primary feature:** `alg_entry_signal`
- **Signal polarity:** high_long
- **Source:** `scripts/algorithms/optimal_entry.py`
- **Known bug:** `run_batch()` hardcodes `sigma_process=0.01` instead of using `self._sigma_process`

---

## Tier 2 — Symbol-specific alpha — **REFUTED, see banner**

### 5. surprise_signal — Total +3,505 bps

**Entropy regime transition detection via ROC z-score.**
Captures microstructure regime shifts through entropy rate-of-change. Strong on SOL, decent on ETH, fails on BTC.

| Symbol | Trades | Net bps/trade | Sharpe | Total PnL (bps) | Win Rate | Max Daily Loss |
|--------|--------|--------------|--------|-----------------|----------|----------------|
| BTC    | 954    | -4.78        | -8.3   | -4,563          | 15%      | -1,316         |
| ETH    | 1,010  | +2.85        | 3.1    | +2,878          | 54%      | -2,493         |
| SOL    | 981    | +5.29        | 6.7    | +5,190          | 46%      | -518           |

- **Primary feature:** `alg_entropy_surprise`
- **Signal polarity:** low_long (ordering = long, disordering = short)
- **Source:** `scripts/algorithms/surprise_signal.py`

---

## Tier 3 — No edge after costs

All algorithms below are net negative in aggregate across the OOS window.

| # | Algorithm | Total (bps) | BTC Sharpe | ETH Sharpe | SOL Sharpe | Primary Feature |
|---|-----------|-------------|-----------|-----------|-----------|-----------------|
| 6 | oi_divergence | -1,721 | -5.3 | -5.7 | +2.1 | `alg_oi_price_divergence` |
| 7 | regime_gated | -1,748 | -2.4 | -0.4 | -0.0 | `alg_regime_gated_imbalance` |
| 8 | entropy_momentum | -2,600 | -6.4 | -0.2 | -2.7 | `alg_entropy_gated_momentum` |
| 9 | propagator | -4,118 | -2.4 | -1.2 | -3.9 | `alg_transient_impact` |
| 10 | hawkes_intensity | -5,443 | +0.8 | -2.6 | -4.3 | `alg_bid_ask_hawkes_imbalance` |
| 11 | trade_through | -5,739 | -5.1 | -4.2 | +0.2 | `alg_trade_through_imbalance` |
| 12 | weighted_ofi | -6,183 | -4.8 | -0.6 | -3.6 | `alg_weighted_ofi` |
| 14 | switching_ou | -6,230 | -3.5 | +0.7 | -6.0 | `alg_switching_ou_state` |
| 15 | vpin_regime | -7,331 | -4.7 | -1.7 | -3.5 | `alg_vpin_gated_imbalance` |
| 16 | kalman_imbalance | -7,517 | -2.4 | -0.0 | -7.2 | `alg_kalman_signal_strength` |
| 17 | bipower_jump | -32,079 | -14.0 | -9.7 | -9.8 | `alg_jump_ratio` |
| 18 | spread_decomp | -34,510 | -10.7 | -14.8 | -8.4 | `alg_adverse_component` |

All implementations in `scripts/algorithms/*.py`. Generic paper trader: `scripts/alpha/paper_trader_generic.py`.

---

## Portfolio Notes

> ⚠️ Historical, and **refuted** — see the banner. Retained because the *correlation
> structure* is a mechanism observation independent of the mispriced P&L.

- **Complementarity:** 3f_liquidity dominates BTC (Sharpe 9.2), jump_detector dominates ETH/SOL (Sharpe 6.2). Near-zero Spearman correlation between jump_detector and funding_reversion — ideal for blending.
- **Full mathematical derivations:** `reports/algo_mathematical_foundations.md`
- **Verification tests:** `scripts/algorithms/tests/test_winning_algos.py` (25 tests, all passing)
- **Detailed results:** `reports/experiment_report_2.md`, `reports/algo_paper_trade_comparison.json`

---

## Mathematical formulations

*Relocated from `README.md` on 2026-08-08. The derivations below are correct and were never
in question — it is the **P&L claims** attached to them that the Q4 kill gate refuted. They
live here so the root README does not lead with 200 lines of math for five rejected
algorithms, and so the mechanism record survives intact.*

### 1. `3f_liquidity` — three-feature liquidity composite

Constructs a composite liquidity score from spread, depth, and VWAP deviation, then
z-scores it over a rolling training window. Extreme z-scores (P20/P80) trigger
mean-reversion entries: wide spreads and thin depth revert as liquidity providers refill.

Given 5-minute bars, the three inputs are:

```
  f₁ = raw_spread_bps       (bid-ask spread in basis points)
  f₂ = raw_ask_depth_5      (ask-side volume, levels 1-5)
  f₃ = flow_vwap_deviation  (price deviation from volume-weighted average)
```

Each is z-scored over a rolling W-day training window, and the composite is their sum:

```
  z_i(t) = ( f_i(t) − μ_i ) / σ_i          μ_i, σ_i from the training window
  C(t)   = z₁(t) + z₂(t) + z₃(t)
```

Entry logic, with percentile thresholds from the training distribution:

```
  Long   if  C(t) ≤ P₂₀(C_train)     (liquidity stress → reversion expected)
  Short  if  C(t) ≥ P₈₀(C_train)     (excess liquidity → reversion expected)
  Flat   otherwise
```

Exit: fixed 100-minute horizon (20 bars).

**Refuted:** −11.1 BTC at SSOT cost, and not reproducible (`FINDINGS.md` §4.6).

---

### 2. `jump_detector` — Lee-Mykland nonparametric jump detection

Detects intraday price jumps by comparing each log-return against a locally-estimated
diffusion volatility that is robust to other jumps in the window, then trades the post-jump
mean-reversion.

Let r_t = ln(p_t / p_{t−1}). Local volatility via bipower variation (robust to jumps, unlike
standard deviation):

```
  σ̂_BV(t) = √( (π/2) · mean_{i=2}^{K} |r_{t−i}| · |r_{t−i+1}| )
```

The constant π/2 = 1/μ₁² corrects for the expected product of adjacent half-normal variables
(μ₁ = E[|Z|] = √(2/π) for Z ~ N(0,1)). The test statistic is:

```
  L(t) = |r_t| / σ̂_BV(t)
```

A jump is declared when L(t) > c. The exact critical value follows a Gumbel distribution
under continuous-record asymptotics — which the legacy fixed c = 3.0 ignores, and which
`jump_detector_v2` implements properly (≈7.2 at 1-day blocks).

After a detected jump at tick t_J with return r_J and price p_J:

```
  REV(t) = − ln(p_t / p_J) / r_J       for 0 < t − t_J ≤ H
```

The negation makes +1 = "fully reverted", so a positive signal directly indicates reversion.

**Parameters (v1):** window K = 100 ticks, significance c = 3.0, reversion horizon H = 50 ticks.

**Refuted:** at c = 3.0 the threshold fires ~13,900×/day — a noise filter, not jump
detection. Fails G4 at every cost tier (`FINDINGS.md` §4.6).

**Reference:** Lee, S.S. & Mykland, P.A. (2008). Jumps in financial markets: a new
nonparametric test and jump dynamics. *Review of Financial Studies*, 21(6), 2535-2563.

---

### 3. `optimal_entry` — SPRT on Kalman innovation

A Sequential Probability Ratio Test (Wald 1947) on the innovation sequence of a Kalman
filter tracking an OU process on order-book imbalance. The SPRT minimizes expected sample
size for a given error rate, so it provides statistically optimal entry timing.

A Kalman filter tracks latent OU dynamics on `imbalance_qty_l1`; the one-step-ahead
innovation is ν_t = z_t − ẑ_{t|t−1}. The SPRT tests:

```
  H₀: ν_t ~ N(0, σ̂²)         (no signal — noise only)
  H₁: ν_t ~ N(μ, σ̂²)         (drift present — entry opportunity)
```

with μ = 0.001 the minimum detectable drift and σ̂² an EMA estimate of innovation variance:

```
  σ̂²(t) = α · ν_t² + (1 − α) · σ̂²(t−1)        α = 0.02
```

The per-tick log-likelihood-ratio increment (closed-form Gaussian ratio) and its cumulative
statistic:

```
  Λ_t = (μ / σ̂²) · ν_t  −  μ² / (2σ̂²)
  S_n = S_{n−1} + Λ_t
```

Wald's optimal decision boundaries:

```
  A = log((1 − β) / α) ≈  2.77     (accept H₁ — fire entry signal)
  B = log(β / (1 − α)) ≈ −1.55     (accept H₀ — no entry)
```

When S_n ≥ A the entry direction is sign(ν_t) and S resets to 0; when S_n ≤ B, no signal and
S resets.

**Parameters:** OU theta = 0.1, process noise = 0.01, observation noise = 0.1, α = 0.05, β = 0.20.

**Refuted:** the sweep never ran the SPRT logic at all — it applied a generic P20/P80 entry.
Fails G4 on stored data (`FINDINGS.md` §4.6).

**References:** Wald, A. (1947). *Sequential Analysis*. Wiley. Shiryaev, A.N. (1978).
*Optimal Stopping Rules*. Springer.

---

### 4. `funding_reversion` — funding-rate mean reversion

Perpetual funding rates mean-revert: when funding is extremely positive (longs pay shorts),
the rate tends toward zero, creating a predictable price movement. Premium divergence — the
gap between spot and futures price — acts as a confirming signal.

Given funding z-score z_t and premium p_t (bps), the entry activates only when funding is
extreme (|z| ≥ z_entry, default 2.0):

```
  signal(t) = −sign(z_t) · min(|z_t| / z_entry, 3) / 3       if |z_t| ≥ z_entry
            = 0                                               otherwise
```

Contrarian by construction: short when funding is extremely positive (crowded longs), long
when extremely negative; magnitude clamped to [−1, 1] by the min/3 normalization.

Funding momentum tracks the EMA of the raw rate, and premium divergence combines the two:

```
  EMA(t) = α · rate(t) + (1 − α) · EMA(t−1)       α = 2/(span+1), span = 100
  D(t)   = (1 − w) · z_t + w · (p_t / 10)         w = 0.3
```

The premium is scaled by 1/10 to bring it to a magnitude comparable with the z-score.

**Parameters:** z-entry = 2.0, momentum span = 100 ticks, premium weight = 0.3.

**Refuted:** wrong-venue cost, n_eff ≈ 84, and a one-sided funding regime over the test
window (`FINDINGS.md` §4.6).

---

### 5. `surprise_signal` — entropy regime-transition detection

Markets alternate between disordered (high-entropy) and ordered (low-entropy) states. A
sudden entropy drop signals a transition from noise to structure. This algorithm computes
the rate-of-change of a composite entropy measure and z-scores it.

```
  E(t)   = 0.5 · ent_book_shape(t) + 0.5 · ent_tick_5s(t)
  ROC(t) = Ē₅(t) − Ē₅(t − W)                    Ē₅ = 5-tick moving average of E
```

The surprise z-score normalizes ROC against its own recent distribution (rolling 2W window,
min 20 observations), and the transition probability applies a sigmoid:

```
  surprise(t)     = (ROC(t) − μ_ROC) / σ_ROC
  P_transition(t) = 1 / (1 + exp(−|surprise(t)| + τ))       τ = 2.0
```

The sign of surprise indicates direction: negative surprise (entropy dropping) = market
ordering = potential trend formation.

**Parameters:** ROC window = 50 ticks, transition threshold τ = 2.0.

**Refuted:** 87.6 % of the claimed edge came from a single day (`FINDINGS.md` §4.6).

**References:** Bandt, C. & Pompe, B. (2002). Permutation entropy. *Physical Review
Letters*, 88(17), 174102. Schreiber, T. (2000). Measuring information transfer. *Physical
Review Letters*, 85(2), 461-464.
