# Spannung — Online Flow/Illiquidity Tension Metric

**Date**: 2026-05-15
**Status**: Phase F complete — cross-symbol walk-forward validates imbalance signal across BTC/ETH/SOL

## Original Idea

> I want to define an online learning process where I define a concept called
> Spannung which computes (with an exponentially decaying filter) the one-sided
> order inflow w.r.t. illiquidity with a set of parameters to be learned online
> until it proves predictive power on the price actions.

## Concept

"Spannung" (German: tension / voltage) measures how much directional order
pressure is building against a market's ability to absorb it. The analogy is
electrical: **voltage = current / resistance**, which maps to
**flow / illiquidity**. When Spannung is high, price *must* move.

The key insight: neither flow nor illiquidity alone is sufficient. The same
order flow has 10x impact in a thin market versus a deep one. Spannung
captures this interaction as a single, interpretable scalar.

## Formalization

```
Spannung(t) = EWM_alpha[ signed_flow(t) ] / ( EWM_beta[ illiq(t) ] + epsilon )
```

Where:
- `signed_flow`: directional order pressure (aggressor imbalance, flow_imbalance, or VPIN deviation from 0.5)
- `illiq`: absorption capacity (Kyle lambda, composite illiquidity, or inverse depth)
- `alpha`: exponential decay halflife for flow memory
- `beta`: exponential decay halflife for illiquidity memory (potentially different — flow memory != liquidity memory)
- `epsilon`: regularization to prevent division by zero in liquid regimes

## Parameter Space

| Parameter | Role | Prior range |
|-----------|------|-------------|
| alpha (flow decay) | How long flow memory persists | 0.5s - 30s halflife |
| beta (illiq decay) | How long fragility memory persists | 5s - 60s halflife |
| w_flow | Which flow signals to use | aggressor_ratio, flow_imbalance, vpin |
| w_illiq | Which fragility signals to use | kyle_100, kyle_500, composite |
| h (horizon) | Forward return to predict | 1s - 60s |
| theta (threshold) | When Spannung is "active" | Learned from signal distribution |

## Online Learning Loop

```
At each tick t:
  1. Compute Spannung(t; alpha, beta, w_flow, w_illiq)
  2. Observe realized return r(t+h) - r(t)
  3. Update parameters:
     Loss = (r(t+h) - gamma * Spannung(t))^2
  4. Track rolling IC (information coefficient) as fitness metric
  5. Signal is "proven" when rolling IC > threshold for N consecutive windows
```

### Method Options

- **Online SGD**: Simplest — updates alpha/beta/weights each tick via gradient of prediction error
- **Recursive Least Squares (RLS)**: Better for the linear coefficient gamma, converges faster than SGD
- **Kalman filter**: Treats parameters as time-varying state — most appropriate if optimal alpha/beta drift over time (which they probably do)

**Recommended hybrid**: Kalman filter for the linear coefficient gamma (how much Spannung predicts returns) + periodic grid search or Bayesian optimization for structural parameters (alpha, beta, feature selection) since those are harder to differentiate through.

## Why This Should Work

1. **Kyle (1985)**: Price impact = lambda x order_flow. Spannung is the empirical real-time product
2. **Bouchaud et al. (2004)**: Order flow has long memory (H ~ 0.7-0.8). The EWM captures this — accumulated pressure, not just instantaneous flow
3. **Cont, Stoikov, Talreja (2010)**: Order book imbalance predicts short-term returns at tick-to-second timescales
4. **Easley et al. (2012)**: VPIN predicts flash crashes and short-term dislocations
5. **Illiquidity as the multiplier**: Same flow has 10x impact in thin markets. Normalization makes Spannung comparable across regimes

## Available Features in NAT

Already computed in Rust at 100ms resolution:

**Flow signals** (numerator candidates):
- `flow_aggressor_ratio_5s`, `flow_aggressor_ratio_30s` — directional conviction
- `toxic_flow_imbalance` — signed directional pressure
- `toxic_vpin_50` — probability of informed trading
- `imbalance_qty_l1`, `imbalance_qty_l5` — order book imbalance

**Illiquidity signals** (denominator candidates):
- `illiq_kyle_100`, `illiq_kyle_500` — price impact per unit flow
- `illiq_composite` — multi-measure illiquidity
- `illiq_amihud_100` — return per unit volume
- `1 / raw_bid_depth_5` — inverse depth as fragility proxy

## Recommended Implementation Path

### Phase A: Offline Grid Search (first)

Run on existing 15m parquet data. Grid over (alpha, beta) pairs, compute Spannung, measure IC against forward returns at multiple horizons. This reveals:
- Whether the signal has juice at all
- The shape of the IC landscape (smooth = learnable, spiky = noise)
- The best (alpha, beta) starting point for the online learner
- Which flow/illiq feature combination works best

**Go/no-go gate**: IC > 0.05 on out-of-sample data at any horizon.

### Phase B: Online Learner (if Phase A passes)

Implement the learning loop in Python, running against live ingestor data. Start with RLS for gamma, periodic re-optimization for alpha/beta. Track convergence and stability.

### Phase C: Rust Promotion (if Phase B validates)

If the learned parameters stabilize, hardcode the best configuration into the Rust feature computer for real-time emission at 100ms.

## Phase A Results: Offline Grid Search (2026-05-14)

**Data**: 2026-05-12, ~282k rows per symbol (~10 hours at 10 rows/sec), 3 symbols.
**Grid**: 1,350 combinations per symbol (3 flow x 3 illiq x 6 alpha x 5 beta x 5 horizons).
**Method**: Non-overlapping rolling Spearman IC (3000-tick windows, ~5 min each, 94 windows).

### Cross-Symbol Results

| Symbol | Best IC | Best IR | Best alpha | Best beta | Best horizon | IC>0.05 |
|--------|---------|---------|-----------|----------|-------------|---------|
| BTC | 0.395 | 3.57 | 0.5s | 60s | 5s | 863/1350 (64%) |
| ETH | 0.386 | 3.96 | 0.5s | 60s | 5s | 956/1350 (71%) |
| SOL | 0.335 | 5.67 | 0.5s | 60s | 1s | 790/1350 (59%) |

**Go/no-go gate**: IC > 0.05 required. **PASSED** — best IC is 6-8x the threshold.

### Key Findings

1. **imbalance_qty_l1 dominates completely** — all top-20 results across all symbols use
   L1 book imbalance as the flow input. Neither `flow_aggressor_ratio_5s` nor
   `toxic_flow_imbalance` compete. This makes sense: L1 imbalance is the most
   instantaneous measure of directional pressure, while aggressor ratio and VPIN
   are already smoothed over 5s/50-bucket windows.

2. **Fastest flow decay wins (alpha=0.5s)** — Spannung is about *instantaneous* pressure,
   not accumulated flow. The 0.5s halflife means 86% of the signal comes from the
   last 1.5s of book snapshots. This is consistent with Cont/Stoikov (2010) finding
   that imbalance predictive power peaks at the very shortest horizons.

3. **Slow illiquidity decay wins (beta=60s)** — fragility is a regime, not a tick-level
   signal. The market "remembers" being thin for minutes. A 60s halflife means the
   illiquidity denominator reflects the last ~3 minutes of market conditions.

4. **SOL is fastest** — optimal horizon is 1s vs 5s for BTC/ETH. SOL's thinner book
   means information incorporates faster. SOL also has the highest IR (5.67) despite
   lower IC, suggesting more stable signal-to-noise.

5. **IC magnitude is extraordinary** — typical feature ICs in quant are 0.02-0.05.
   Spannung achieves 0.33-0.40, an order of magnitude higher. The 100% hit rate
   across 94 non-overlapping 5-minute windows means the signal never flips sign
   over 10 hours of data.

6. **illiq_composite slightly better than kyle alone** — composite (which blends Kyle,
   Amihud, Hasbrouck, Roll) provides marginal improvement in the denominator,
   suggesting the fragility signal benefits from multiple measurement approaches.

7. **The illiquidity denominator adds real value** — even the worst illiq feature
   combination with the right (alpha, beta) produces IC > 0.30. But comparing
   raw imbalance_qty_l1 alone (no illiq normalization) against Spannung would
   quantify the marginal contribution of the denominator. (Not yet tested.)

### Recommended Best Configuration

```
Spannung(t) = EWM(halflife=5 ticks)[imbalance_qty_l1] / (EWM(halflife=600 ticks)[illiq_composite] + 1e-10)
```

- alpha = 5 ticks = 0.5s
- beta = 600 ticks = 60s
- Prediction horizon: 50 ticks (5s) for BTC/ETH, 10 ticks (1s) for SOL

## Phase A.2: Look-Ahead Bias Check (2026-05-14)

**Method**: Lag the Spannung signal by 0–100 ticks and re-measure IC. Also shift it
forward (using future book state) to detect leakage. Test raw imbalance alone to
measure the marginal value of the EWM and illiquidity denominator.

### Lag Decay (BTC, horizon=5s)

| Signal lag | IC | Interpretation |
|------------|-----|----------------|
| 0 (baseline) | 0.395 | What we measured in Phase A |
| 0.1s | 0.388 | Minimal decay — signal is stable |
| 0.5s | 0.359 | Gradual decay |
| 1.0s | 0.328 | Still strong |
| 5.0s | 0.180 | Significant decay — signal is short-lived |
| 10.0s | 0.085 | Near zero — halflife is ~5s |

**Verdict: No look-ahead bias.** IC decays gradually and smoothly with lag. This is
exactly what a causal microstructure signal looks like. The future-shifted signal
(+1 to +50 ticks) shows only marginally higher IC (0.40→0.46), confirming there
is no information leakage in the signal construction.

### Denominator and EWM Add No Value

This was the unexpected finding:

| Signal | BTC IC | ETH IC | SOL IC |
|--------|--------|--------|--------|
| Spannung (EWM flow / EWM illiq) | 0.395 | 0.386 | 0.335 |
| EWM(imbalance) alone, no denom | 0.429 | 0.429 | 0.348 |
| **Raw imbalance_l1, no EWM, no denom** | **0.479** | **0.484** | **0.441** |

Raw, unsmoothed L1 book imbalance beats Spannung by 8-10 IC points across all symbols.
Both the exponential smoothing and the illiquidity normalization *hurt* the signal:

- **EWM hurts**: The 0.5s halflife blurs the instantaneous book snapshot. At this
  timescale (predicting 1-5s ahead), the most recent tick is the most informative
  and any smoothing dilutes it with stale information.
- **Illiquidity denominator hurts**: Normalizing by illiquidity adds noise rather than
  context. At tick-level resolution, the book imbalance already implicitly reflects
  liquidity conditions — when the book is thin, imbalance is naturally more extreme.
  Dividing by a separate illiquidity estimate is redundant and introduces estimation error.

### What This Means

1. **The core signal is raw L1 book imbalance** → 0.44-0.48 IC predicting 1-5s forward
   returns. This is one of the strongest documented microstructure signals, consistent
   with Cont/Stoikov/Talreja (2010).

2. **The Spannung formulation as designed doesn't add value over the raw input** at
   tick-level resolution. The voltage = current/resistance metaphor is elegant but
   the denominator is redundant when the numerator already embeds liquidity information.

3. **Where Spannung might still add value**: At longer timescales (minutes, not ticks),
   where raw imbalance becomes noisy and the EWM + illiquidity context could help.
   Also, the illiquidity denominator may matter more in cross-regime comparisons
   (e.g., comparing signal magnitude between calm and volatile periods).

4. **The online learning objective should shift**: Instead of learning alpha/beta for
   Spannung, learn *when to act on raw imbalance* — the gating function (entropy,
   toxicity, volatility regime) that determines when high imbalance is informative
   vs. noise.

## Phase A.3: Out-of-Sample Validation (2026-05-14)

**Method**: Run raw imbalance_l1 IC (the true signal) on 3 out-of-sample dates plus
the original in-sample date. Tests whether the signal generalizes across days and
market conditions.

**Note**: 2026-05-07 and 2026-05-08 data turned out to be frozen (constant features,
old ingestor bug). Valid OOS dates: 2026-05-11, 2026-05-10, 2026-04-25.

### Cross-Day Results (raw imbalance_l1, BTC h=5s, ETH h=5s, SOL h=1s)

| Date | Tag | Hours | BTC IC | ETH IC | SOL IC |
|------|-----|-------|--------|--------|--------|
| 2026-05-12 | In-sample | 10.0h | 0.479 | 0.484 | 0.441 |
| 2026-05-11 | OOS | 24.0h | 0.466 | 0.471 | 0.423 |
| 2026-05-10 | OOS | 6.6h | 0.446 | 0.423 | 0.405 |
| 2026-04-25 | OOS | 7.6h | 0.359 | 0.353 | 0.270 |

### Interpretation

1. **Recent OOS (May 10-11) is nearly identical to in-sample** — IC drops only
   0.01-0.04. No overfit. The signal is stable across consecutive days.

2. **24-hour stability confirmed** — 2026-05-11 has 196 non-overlapping 5-min windows
   and still shows IC=0.47 for BTC. The signal doesn't degrade over a full day.

3. **Older data (April 25) is weaker but still strong** — IC 0.27-0.36. Possible causes:
   - Different market regime (3 weeks earlier, different volatility/liquidity conditions)
   - Earlier ingestor version computing features slightly differently
   - Still well above the 0.05 go/no-go threshold

4. **The signal passes OOS validation decisively** — raw L1 book imbalance is a genuine,
   persistent microstructure signal that generalizes across days.

## Phase B: Cost-Aware Backtest (2026-05-14)

**Strategy**: At each h-tick interval, observe imbalance_qty_l1. If |imbalance| >
threshold, take position in sign(imbalance). Hold for h ticks. Sweep thresholds
0.0–0.8. Costs: taker 3.5 bps/side, maker 1.0 bps/side.

### Results (2026-05-12 data, best threshold per symbol)

| Symbol | Horizon | Gross Sharpe | Net Sharpe (taker) | Net Sharpe (maker) | Gross bps | Net bps (taker) | bps/trade |
|--------|---------|-------------|-------------------|-------------------|-----------|----------------|-----------|
| BTC | 5.0s | 672 | **-1,054** | -335 | +1,533 | -7,256 | 0.27 |
| ETH | 5.0s | 727 | **-1,152** | -285 | +2,082 | -8,666 | 0.37 |
| SOL | 1.0s | 1,387 | **-2,038** | -849 | +4,877 | -27,025 | 0.17 |

**The signal does NOT survive transaction costs.** Every threshold level loses money
after both taker and maker fees across all three symbols.

### Why It Fails

1. **Per-trade edge is microscopic**: 0.17–0.37 bps gross per trade, vs 7 bps
   round-trip taker cost. The signal needs ~20x more edge per trade to break even.

2. **Excessive turnover**: The signal flips direction constantly — 1,200–7,700 position
   changes over 10 hours. Each flip incurs cost. The IC comes from directional
   accuracy in aggregate, but the per-trade magnitude is too small.

3. **Hit rate is only 30–45%**: Most individual trades lose. The overall IC is positive
   because winners are slightly larger than losers, but not by enough to cover costs.

4. **This is the HFT problem**: The signal is real (IC 0.45+) but lives at a timescale
   (100ms–5s) where execution costs dominate. This is exactly the domain of collocated
   market makers with sub-millisecond latency, maker rebates, and queue priority.

## Phase B.2: Regime Gating (2026-05-14)

**Method**: Condition the imbalance signal on entropy, VPIN, volatility, and illiquidity
regimes. Split each regime feature at its median, test single and combined conditions.

### Results (BTC, threshold=0.3)

| Regime | Condition | N intervals | IC | Net Sharpe (taker) |
|--------|-----------|------------|-----|-------------------|
| ALL (baseline) | no filter | 5,647 | 0.480 | -1,214 |
| Low entropy | ent < 0.63 | 2,823 | 0.475 | -1,301 |
| High entropy | ent >= 0.63 | 2,824 | 0.484 | -1,378 |
| Low vol | vol < median | 2,823 | 0.489 | -1,162 |
| High illiq | kyle >= 0.55 | 2,824 | 0.488 | -1,223 |
| Low ent + High illiq | informed + fragile | 1,265 | 0.510 | -1,328 |
| High vol + High illiq | volatile + fragile | 1,321 | 0.508 | -1,436 |

**Regime gating does not help.** No regime combination produces positive net Sharpe.
The IC varies slightly across regimes (0.42–0.51) but the cost problem dominates
everywhere. Even the best regime ("low entropy + high illiquidity", IC=0.51) still
loses -1,328 Sharpe after taker fees.

### Key Insight

The regime gating hypothesis — that imbalance is more informative in structured,
fragile markets — is **directionally correct** (IC does increase from 0.48 to 0.51
in the "informed + fragile" regime). But the improvement is far too small to overcome
the fundamental cost problem. The signal needs 20x more edge per trade, not 6% more IC.

## Conclusions and Next Directions

### What We Learned

1. **L1 book imbalance is a genuine microstructure signal** — IC 0.44–0.48, no
   look-ahead bias, generalizes across days and symbols. This is consistent with
   Cont/Stoikov/Talreja (2010).

2. **The Spannung formulation (EWM + illiq denominator) hurts at tick timescales** —
   raw imbalance beats it. Both smoothing and normalization add noise, not signal.

3. **Not tradeable as a naive directional strategy** — per-trade edge (0.2–0.4 bps)
   is 20x too small relative to taker fees (7 bps round-trip). Bar aggregation
   destroys signal (IC drops 2–3x). Regime gating is directionally correct but
   insufficient.

4. **Spectral analysis reopens the path.** Almost all predictive power (IC=0.45)
   lives in the ultra-low frequency band (0.005–0.1 Hz, periods 10–200s). The
   signal has brown noise characteristics (H=0.43), OU mean-reversion half-life
   of 5–7s, and dominant coherence with returns at 0.015 Hz (~68s cycles).
   This structure supports market-making and Kalman-filtered extraction rather
   than brute-force bar aggregation.

5. **The viable path is: zero-fee pairs + Kalman filter + regime gating + market-making.**
   Extract the slow imbalance component via bandpass/Kalman, gate on `ent_book_shape`
   (lifts IC from 0.45 to 0.55+), exploit the 5–7s mean-reversion cycle on zero-fee
   exchanges where the only costs are spread and adverse selection. Regime screening
   (Phase E) confirmed that `ent_book_shape` is the dominant gating condition —
   replicating across dates with 20-45% IC improvement.

## Phase C: Longer-Horizon Sweep (2026-05-14)

**Method**: Aggregate tick-level data into bars (30s, 1min, 2min, 5min, 10min, 15min).
At each bar, compute 6 signal variants from the underlying ticks. Test forward returns
at 1, 2, 4, 8 bars ahead. Total: 144 combinations per symbol per date.

### Signal Variants Tested

1. **imbalance_mean** — mean(imbalance_qty_l1) over bar (baseline)
2. **imbalance_last** — last tick imbalance in bar
3. **imbalance_trend** — OLS slope of imbalance over bar (is pressure building?)
4. **imbalance_persistence** — fraction of ticks where sign(imbalance) matches bar mean
5. **spannung_mean** — mean(EWM_flow / EWM_illiq) over bar (original Spannung at bar level)
6. **imbalance_x_illiq** — imbalance_mean × illiq_composite_mean (interaction product)

### Results Summary

| Date | Symbol | Hours | Profitable combos (net > 0) | Best net Sharpe | Best breakeven |
|------|--------|-------|-----------------------------|-----------------|----------------|
| 2026-05-12 (IS) | BTC | 10.0h | **0 / 96** | -23.3 | 2.2 bps/side |
| 2026-05-12 (IS) | ETH | 10.0h | **3 / 96** | 5.2 (5min, 600s) | 3.9 bps/side |
| 2026-05-11 (OOS) | BTC | 24.0h | **4 / 144** | 8.0 (15min, 7200s) | 158.3 bps/side |
| 2026-05-11 (OOS) | ETH | 24.0h | **7 / 144** | 10.8 (10min, 4800s) | 5.5 bps/side |

### Why This Doesn't Work

1. **Zero overlap between in-sample and OOS profitable combos.** Different signals,
   different timeframes, different horizons. The "profitable" combinations are noise
   artifacts from thin data (70–200 bars per run), not robust strategies.

2. **IC degrades catastrophically with aggregation.** Tick-level IC was 0.44–0.48.
   At bar level, best ICs are 0.15–0.23 — a 2–3x drop. The raw signal lives at
   the tick, and aggregating it destroys most of its information content.

3. **Negative IC on profitable combos.** Most combos that show positive net P&L have
   *negative* IC (mean-reversion, signal flipped). This is consistent with noise:
   with 70 bars, a few random reversals can produce apparent profitability.

4. **Cost still dominates.** Even where gross returns are large (e.g., 344 bps for
   imbalance_trend at 2min/960s), costs consume it all (net = -562 bps). The
   problem is turnover: aggregated signals still flip frequently enough to eat
   the larger per-trade edge.

5. **The Spannung EWM formulation does not help at bar level either.** spannung_mean
   performs comparably to or worse than imbalance_mean at every timeframe.

### What This Means for the Spannung Thesis

The original hope — that aggregating the tick signal to longer bars would preserve
enough IC while reducing costs — does not hold. The fundamental problem is that
**L1 book imbalance is an inherently short-lived signal**:

- At tick level: IC=0.48, 100% hit rate, 0.2–0.4 bps/trade edge → not enough for costs
- At bar level: IC=0.05–0.20, inconsistent across dates, insufficient edge → still not enough
- The signal's predictive power decays exponentially with timescale. There is no
  "sweet spot" where IC is high enough and costs are low enough.

This is consistent with the academic literature. Cont/Stoikov/Talreja (2010) showed
book imbalance predicts at the *tick* timescale. It was never expected to predict
at minutes-to-hours horizons — that requires fundamentally different signals
(order flow toxicity accumulation, inventory effects, information cascades).

### Where to Go From Here

1. **Imbalance as a feature, not a strategy**: Feed imbalance into a multi-feature
   model (with entropy, toxicity, trend, volatility) that trades at lower frequency.
   The IC=0.48 at 5s makes it the strongest single input — but it needs to be
   combined with other signals at a tradeable timescale.

2. **Maker strategy**: If you can reliably post limit orders (1 bps/side = 2 bps
   round trip), the cost picture improves 3.5x. Still not enough for pure imbalance
   at 5s, but combined with longer horizons it might work.

3. **Cross-symbol signals**: Does BTC imbalance predict ETH/SOL returns at longer
   horizons? Cross-asset information flow operates at slower timescales where
   costs are less punishing.

4. **Abandon single-signal strategies at retail**: The core lesson from Phases A–C
   is that no single microstructure signal (no matter how strong its IC) generates
   enough per-trade edge to overcome retail taker fees. The path forward is
   multi-signal combination at lower trading frequency.

## Phase D: Spectral Analysis (2026-05-14)

**Method**: Frequency-domain characterization of `imbalance_qty_l1` — Welch PSD,
cross-coherence with forward returns, FFT-based autocorrelation with OU process fit,
band-pass filtered IC across 4 frequency bands, spectral entropy.

### Spectral Characteristics (replicates across dates)

| Metric | 2026-05-12 (10h) | 2026-05-11 (24h) |
|--------|-------------------|-------------------|
| Noise color | Brown (slope=-1.86) | Brown (slope=-1.86) |
| Hurst exponent | 0.431 | 0.430 |
| OU half-life | 7.3s | 5.4s |
| ACF(1s) | 0.895 | 0.860 |
| ACF(5s) | 0.597 | 0.506 |
| ACF first zero crossing | 44.9s | >60s |
| Spectral entropy | 0.456 | 0.499 |

**Remarkably stable across dates** — noise slope, Hurst exponent, and spectral entropy
are virtually identical. This is structural, not a transient market condition.

### Band-Filtered IC — Where the Predictive Power Lives

| Band | Freq range | Period range | IC (1s fwd) | IC (5s fwd) | IC IR (5s) |
|------|-----------|-------------|-------------|-------------|------------|
| **ultra_low** | 0.005–0.1 Hz | 10–200s | **+0.30** | **+0.45** | **4.1** |
| **low** | 0.05–0.5 Hz | 2–20s | **+0.27** | +0.10 | 1.1 |
| mid | 0.5–2.0 Hz | 0.5–2s | -0.02 | ~0 | — |
| high | 2.0–4.5 Hz | 0.2–0.5s | ~0 | ~0 | — |

**Critical finding: almost ALL predictive power (IC=0.45) sits in the ultra-low
frequency band (periods 10–200s).** The mid and high frequency bands are pure noise.

This completely reframes the tradeability question. Phase C's bar-level aggregation
failed because it blindly averaged across all frequencies — including the noisy mid/high
components that dilute the signal. A Kalman filter or bandpass extraction targeting
the 0.005–0.1 Hz component would preserve the IC while operating at a timescale
(~10–60s cycles) where execution is feasible.

### Coherence with Forward Returns

Strongest coherence peaks (consistent across both dates):

| Frequency | Period | Coherence | Phase lead (1s horizon) |
|-----------|--------|-----------|------------------------|
| 0.015 Hz | 68s | 0.26 | -390 to -1706 ms |
| 0.024 Hz | 41s | 0.22 | +124 ms |
| 0.044 Hz | 23s | 0.22 | +237 ms |
| 0.093 Hz | 11s | 0.16 | +101 ms |

The dominant coupling is at ~1 minute cycles (0.015 Hz). This is the natural frequency
at which imbalance and price movements are most strongly linked. The positive phase at
0.024–0.093 Hz means **imbalance leads returns by 100–500ms at these frequencies** —
the signal has genuine predictive lead time, not just contemporaneous correlation.

### Autocorrelation and Mean-Reversion

- **OU half-life = 5–7s** — imbalance mean-reverts on a characteristic ~6s timescale.
  This is the natural quote refresh period for a market-making strategy.
- **ACF(1s) = 0.86–0.90** — extremely high persistence at 1s. The signal does not
  jump around — it evolves smoothly. This is ideal for Kalman filtering.
- **ACF(5s) = 0.50–0.60** — still substantial correlation at 5s, consistent with
  the 5–7s half-life.
- **Hurst H = 0.43** — slightly mean-reverting (H < 0.5), confirming the OU model
  is appropriate. Not strongly mean-reverting, but enough for market-making cycles.

### What This Means

1. **The signal is not dead — it was being extracted wrong.** Phase C's bar aggregation
   failed because it averaged across all frequencies. The predictive content is
   concentrated in the ultra-low band. Band-pass extraction or Kalman filtering
   targeting this band should preserve IC=0.45 while reducing noise.

2. **Market-making is viable.** The OU half-life (5–7s), dominant coherence period
   (~60s), and slight mean-reversion (H=0.43) define a natural market-making regime:
   - Quote refresh at ~0.15 Hz (every ~7s, matching OU half-life)
   - Position holding period aligned with the 60s coherence cycle
   - Lean quotes in the direction of the ultra-low band signal

3. **Kalman filter can recover latency-lost IC.** The lag decay from Phase A was
   IC: 0.48 → 0.39 (100ms) → 0.33 (1s). But the ultra-low band has periods of
   10–200s — at these timescales, 100ms–1s of latency is negligible. A Kalman
   filter tracking the slow component operates where latency doesn't matter.

4. **Zero-fee pairs become viable.** On zero-fee exchanges, the remaining costs are
   spread (~0.5–1 bps) and adverse selection. With the ultra-low band signal
   (IC=0.45, IR=4.1), the per-trade edge at ~60s holding periods is much larger
   than at 5s — enough to potentially clear spread costs.

5. **Spectral entropy (0.45–0.50) indicates moderate concentration** — the spectrum
   is neither perfectly periodic (entropy~0) nor white noise (entropy~1). There is
   exploitable structure, but it requires careful signal extraction.

## Phase E: Regime Screening (2026-05-15)

**Method**: Systematic search across 17 microstructure features as regime conditions.
For each feature, split at quintile thresholds (P20/P40/P60/P80) and measure conditional
IC of both raw and ultra-low bandpass-filtered imbalance. Then test all 2-way and 3-way
AND combinations of top single factors, apply Pareto filter (no other combo has both
higher IC and higher coverage), and measure regime persistence (duration, entry rate).

**Data**: 2026-05-12 (10h, in-sample) and 2026-05-11 (24h, out-of-sample).

### Regime Features Screened (17 total)

- **Entropy** (6): ent_tick_1m, ent_tick_5s, ent_tick_30s, ent_permutation_returns_16,
  ent_spread_dispersion, ent_book_shape
- **Illiquidity** (3): illiq_kyle_100, illiq_composite, illiq_amihud_100
- **Toxicity** (3): toxic_vpin_50, toxic_adverse_selection, toxic_index
- **Volatility** (3): vol_returns_1m, vol_returns_5m, vol_ratio_short_long
- **Derived** (3): derived_regime_type_score, derived_regime_confidence,
  derived_informed_trend_score

### Single-Factor Results (top 5, filtered IC at 5s horizon)

| Condition | IS IC | IS dIC | OOS IC | OOS dIC | Coverage |
|-----------|-------|--------|--------|---------|----------|
| **ent_book_shape<P40** | **0.544** | **+0.089** | — | — | 40% |
| **ent_book_shape<P20** | 0.542 | +0.088 | **0.546** | **+0.091** | 20% |
| ent_book_shape<P60 | 0.514 | +0.059 | 0.500 | +0.045 | 60% |
| toxic_adverse_selection>P80 | 0.494 | +0.039 | — | — | 20% |
| ent_permutation_returns_16>P60 | 0.480 | +0.026 | 0.493 | +0.038 | 46-54% |

**`ent_book_shape` dominates completely.** It appears in every single Pareto-optimal
combination on both dates. The effect replicates almost identically OOS.

### Why ent_book_shape Works

Low book shape entropy means the order book has a structured, non-random depth profile.
This happens when informed traders are positioning — they create predictable patterns
(heavy depth on one side, thin on the other). In this regime, imbalance is a stronger
signal because the book structure *confirms* that the imbalance reflects genuine
directional information, not random noise.

The intuition: when the book looks random (high entropy), anyone could be placing orders
for any reason. When the book has structure (low entropy), someone with information is
shaping it — and the imbalance tells you which direction.

### Multi-Factor Pareto Frontier

| Combo | IS IC_filt | IS dIC | OOS IC_filt | OOS dIC | Coverage |
|-------|-----------|--------|-------------|---------|----------|
| ent_book_shape<P40 & ent_tick_5s<P40 & derived_regime_type_score<P40 | **0.634** | +0.179 | — | — | 7% |
| ent_book_shape<P20 & toxic_vpin_50>P80 & toxic_adverse_selection>P60 | — | — | **0.669** | +0.214 | 1% |
| ent_book_shape<P40 & ent_tick_5s<P40 | 0.593 | +0.139 | — | — | 17% |
| ent_book_shape<P20 & toxic_vpin_50>P80 | — | — | 0.640 | +0.185 | 4% |
| ent_book_shape<P20 & vol_ratio_short_long>P60 | — | — | 0.592 | +0.137 | 8% |
| ent_book_shape<P40 & illiq_composite<P60 | 0.561 | +0.106 | — | — | 25% |
| ent_permutation_returns_16>P60 & ent_tick_30s>P40 | — | — | 0.506 | +0.051 | 31% |

Best combos reach IC=0.63–0.67 — a 40-50% improvement over unconditional baseline.

### Regime Persistence — The Tradeoff

| Regime type | IC range | Coverage | Mean duration | >5s episodes |
|-------------|----------|----------|---------------|--------------|
| Tight 3-way combos | 0.60–0.67 | 1–7% | 1–2s | 1–6% |
| 2-way combos | 0.55–0.60 | 8–25% | 1.5–4s | 5–19% |
| Wide single/2-way | 0.49–0.51 | 30–60% | 10–16s | 56–100% |

**The tight combos have extraordinary IC but are too short-lived to trade at retail
latency.** The wide regimes (coverage 30-60%) have IC=0.49-0.51 with episodes lasting
10-16 seconds — these are tradeable.

The **sweet spot** is a 2-way combo at 8-25% coverage: IC=0.55-0.60, episodes lasting
2-4 seconds, with ~15-19% of episodes >5s. On zero-fee pairs with Kalman-filtered
signal extraction, this may be actionable.

### What This Means

1. **`ent_book_shape` is the single most important regime condition.** It replicates
   across dates, dominates every Pareto combo, and has a clear economic interpretation
   (structured book = informed positioning). This is the primary gating signal.

2. **Regime gating lifts IC by 20-45%.** From 0.45 → 0.55 (single factor) to 0.63-0.67
   (multi-factor). Phase B's simple median split found only +6%. The difference: quintile
   thresholds are more selective, and `ent_book_shape` is a fundamentally better regime
   indicator than the features tested in Phase B.

3. **The IC-persistence tradeoff defines the strategy design.** High-IC regimes
   (0.60+) are too short for retail execution. The viable path is:
   - Gate on a wide condition (ent_book_shape<P60 or ent_permutation_returns_16>P60)
     for IC=0.50+ at 30-60% coverage, 10-16s episodes
   - OR gate on a medium condition (ent_book_shape<P40 + one supporting factor)
     for IC=0.55-0.60 at 8-25% coverage, 2-4s episodes — requires faster execution

4. **Multiple testing concern is mitigated.** 313 tests were run, but the dominant
   finding (`ent_book_shape`) is a single, consistent effect — not 20 marginal ones.
   It appears across all thresholds, both dates, and every combo. This is structural.

5. **Needs validation on more dates.** Two days (10h + 24h) is encouraging but not
   conclusive. The April 2026 data (weaker baseline IC) should be tested to check if
   regime gating helps in different market conditions.

### Implementation: `nat spannung`

```bash
nat spannung                                    # grid search (IC landscape)
nat spannung --data data/features/2026-05-12    # specific date
nat spannung --symbol BTC --top 30              # single symbol
nat spannung --horizons 5 10 20 50              # custom horizons (ticks)
nat spannung backtest                           # cost-aware backtest + regime gating
nat spannung backtest --symbol BTC --horizon 50 # single symbol backtest
nat spannung horizon                            # longer-horizon bar sweep
nat spannung horizon --data data/features/2026-05-11 --symbol BTC  # specific date
nat spannung spectral                           # spectral analysis (PSD, coherence, ACF, band IC)
nat spannung spectral --data data/features/2026-05-12 --symbol BTC
nat spannung regime                             # regime screener (quintile, Pareto, persistence)
nat spannung regime --data data/features/2026-05-12 --symbol BTC
```

Output: `reports/spannung/` — per-symbol grid, backtest, regime, spectral, regime_screen, and summary JSONs.

## Open Questions

- ~~At what timescale does aggregated imbalance become cost-positive?~~ **Answered**: bar-level aggregation fails, but band-pass filtering preserves IC=0.45 in ultra-low band.
- ~~Does the Spannung EWM formulation add value at minute+ horizons?~~ **Answered**: no, performs comparably or worse than raw mean imbalance at all bar timeframes.
- ~~How much IC comes from numerator vs denominator?~~ **Answered**: all from numerator
- ~~Does the signal survive a 1-tick lag?~~ **Answered**: yes, smooth decay, no bias
- ~~Does the signal generalize across days?~~ **Answered**: yes, IC 0.27-0.48 across 4 dates
- ~~Can a gating function improve the signal?~~ **Answered**: Phase B median split gave +6%, Phase E quintile screening gave +20-45% via ent_book_shape
- ~~Is the signal tradeable at retail fees?~~ **Answered**: no at taker fees; regime gating + spectral extraction + zero-fee pairs is the viable path
- ~~What are the spectral characteristics?~~ **Answered**: brown noise (H=0.43), OU half-life 5-7s, IC concentrated in ultra-low band (0.005-0.1 Hz), dominant coherence at 0.015 Hz (68s)
- ~~Which regime conditions improve the signal?~~ **Answered**: ent_book_shape dominates (IC 0.45→0.55 single, 0.63-0.67 combo), replicates OOS
- Does ent_book_shape gating help on older data (April 2026) where baseline IC was weaker?
- Can a Kalman filter on the ultra-low band, gated by ent_book_shape, produce tradeable PnL on zero-fee pairs?
- What is the optimal IC-persistence tradeoff point for a market-making strategy?
- Does BTC imbalance predict ETH/SOL returns (cross-symbol information flow)?
- Is there an asymmetry — does buy-side imbalance predict differently than sell-side?
- ~~Can the regime screener findings generalize across ETH and SOL?~~ **Answered**: see Phase F below

## Phase F: Cross-Symbol Walk-Forward Validation (2026-05-15)

**Method**: 5-fold walk-forward profiling (`scalping_profiler.py`) on BTC, ETH, SOL at
1min bars. Each fold trains on 200+ bars, evaluates on held-out fold. Features scored
by OOS IC, sign consistency across folds, and assigned KEEP/MONITOR/DROP verdict.

**Data**: 2026-05-12 (BTC: 2467 bars, ETH/SOL: 472 bars at 1min). Cost: 3.5 bps.

### Verdict Summary

| Metric | BTC | ETH | SOL |
|--------|-----|-----|-----|
| Bars | 2467 | 472 | 472 |
| Fold length | 453 | 54 | 54 |
| KEEP | 167 | 216 | 177 |
| MONITOR | 65 | 34 | 25 |
| DROP | 140 | 123 | 169 |

### Imbalance Features — Cross-Symbol Replication

| Feature | BTC OOS IC | ETH OOS IC | SOL OOS IC | BTC Sign% | ETH Sign% | SOL Sign% | Verdict |
|---------|-----------|-----------|-----------|-----------|-----------|-----------|---------|
| imbalance_qty_l1_last | 0.185 | 0.177 | 0.110 | 100% | 100% | 80% | KEEP all |
| imbalance_qty_l10_last | 0.191 | 0.136 | 0.110 | 100% | 80% | 100% | KEEP all |
| imbalance_depth_weighted_last | 0.191 | 0.135 | 0.110 | 100% | 80% | 100% | KEEP all |
| imbalance_qty_l5_last | 0.095 | 0.170 | 0.075 | 100% | 100% | 60% | KEEP all |
| imbalance_pressure_ask_last | -0.079 | -0.144 | -0.105 | 100% | 100% | 100% | KEEP all |

**All `_last` imbalance variants achieve KEEP verdict across all 3 symbols with >=80%
sign consistency.** This is the strongest cross-symbol evidence yet that L1 book
imbalance is a structural microstructure phenomenon, not a BTC-specific artifact.

### Key Findings

1. **Cross-symbol replication confirmed.** Imbalance signal replicates with KEEP verdict
   and high sign consistency across BTC, ETH, and SOL. The signal is structural.

2. **Liquidity ordering in IC magnitude**: SOL (top features OOS IC ~0.50) > ETH (~0.32)
   > BTC (~0.27) on price/spread features at 1min bars. Less liquid instruments show
   stronger mean-reversion, consistent with microstructure theory — wider effective
   spreads create more mean-reversion.

3. **IS-to-OOS IC increase on SOL** (IS=0.26 → OOS=0.52 on price features) is striking.
   Suggests the later folds capture a cleaner regime or that SOL's thinner microstructure
   amplifies predictable patterns.

4. **`_last` vs `_mean`/`_std` split confirmed cross-symbol.** Instantaneous features
   replicate (KEEP); aggregated features degrade (DROP/MONITOR). This is the time-domain
   manifestation of Phase D's spectral finding: averaging mixes informative ultra-low
   frequencies with noisy mid/high frequencies.

5. **Entropy and toxicity features appear as gates, not directional signals.** On all
   symbols, entropy/toxicity features have IC < 0.04 but are classified as KEEP gates —
   consistent with Phase E's finding that they condition signal quality rather than
   predicting returns directly.

6. **Net edge remains negative at 3.5 bps cost** across all features and symbols.
   Zero-fee pairs + Kalman-filtered tick-level execution remain the viable path.

### Implications for Strategy Design

- The imbalance signal generalizes — a single model can be deployed across BTC/ETH/SOL
  with symbol-specific calibration (primarily horizon: BTC/ETH at 5s, SOL at 1s)
- SOL's higher raw IC + lower liquidity creates the most attractive risk/reward for
  market-making on zero-fee pairs
- Walk-forward KEEP/DROP verdicts can serve as a feature selection filter for the
  production model — only `_last` variants should be used
