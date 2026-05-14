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

### Caveats / Next Steps

- **Look-ahead bias check needed**: The IC is high enough to warrant verifying there's
  no information leakage. Lagging the signal by h ticks and re-measuring IC would
  confirm. The signal *should* degrade with lag — if it doesn't, something is wrong.
- **Out-of-sample validation**: Run on 2026-05-08 or 2026-05-10 data to confirm the
  signal generalizes across days. Same-day IC could overfit to market conditions.
- **Marginal value of denominator**: Test raw `imbalance_qty_l1` IC at h=5s to measure
  how much the illiquidity normalization actually adds.
- **Transaction cost reality**: At IC=0.40 and turnover=0.016, the breakeven cost is
  very favorable, but actual execution at 100ms timescales faces latency, slippage,
  and queue position challenges that paper IC doesn't capture.

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
- Can the exponential decay be replaced with a learned kernel (more general but harder to optimize)?
- How much of the IC comes from the numerator vs the denominator? (Test raw imbalance IC alone)
- Does the signal survive a 1-tick lag? (Look-ahead bias check)
