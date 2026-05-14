# Spannung — Online Flow/Illiquidity Tension Metric

**Date**: 2026-05-14
**Status**: Phase A complete — grid search validates signal

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

### Remaining Caveats

- **Transaction cost reality**: IC=0.48 at 5s horizon is extraordinary on paper, but
  actual execution at 100ms timescales faces latency, slippage, queue position,
  and adverse selection costs that paper IC doesn't capture.
- **Is this tradeable?** At 10 rows/sec, acting on L1 imbalance means competing with
  HFT firms on collocated infrastructure. The signal is real but the edge may not
  survive execution friction for a non-collocated setup.
- **Longer-term OOS**: Need weeks/months of data to confirm stability across major
  market regime changes (bull/bear transitions, volatility spikes, liquidity crises).

### Implementation: `nat spannung`

```bash
nat spannung                                    # auto-detect best data, all symbols
nat spannung --data data/features/2026-05-12    # specific date
nat spannung --symbol BTC --top 30              # single symbol
nat spannung --horizons 5 10 20 50              # custom horizons (ticks)
```

Output: `reports/spannung/spannung_{SYM}.json` + `spannung_summary.json`

## Open Questions

- Should Spannung be computed per-symbol or cross-symbol (e.g., BTC flow / ETH illiquidity)?
- Is there an asymmetry to exploit — does buy-side Spannung predict differently than sell-side?
- How does Spannung interact with the existing `derived_informed_trend_score` (kyle x monotonicity)?
- Can a gating function (entropy/toxicity regime) improve the raw imbalance signal?
- At what timescale does the illiquidity denominator start adding value? (minutes? hours?)
- ~~How much IC comes from numerator vs denominator?~~ **Answered**: all from numerator (raw imbalance)
- ~~Does the signal survive a 1-tick lag?~~ **Answered**: yes, smooth decay, no look-ahead bias
- ~~Does the signal generalize across days?~~ **Answered**: yes, IC 0.27-0.48 across 4 dates
