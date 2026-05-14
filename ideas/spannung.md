# Spannung — Online Flow/Illiquidity Tension Metric

**Date**: 2026-05-14
**Status**: Concept

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

## Open Questions

- Should Spannung be computed per-symbol or cross-symbol (e.g., BTC flow / ETH illiquidity)?
- Is there an asymmetry to exploit — does buy-side Spannung predict differently than sell-side?
- How does Spannung interact with the existing `derived_informed_trend_score` (kyle x monotonicity)?
- Can the exponential decay be replaced with a learned kernel (more general but harder to optimize)?
