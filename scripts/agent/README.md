# NAT Research Agent

Autonomous hypothesis-driven alpha discovery for crypto perpetual futures microstructure.

## What the Agent Tests

The agent tests whether **order book imbalance** predicts short-horizon returns
(1-5 seconds) when **conditioned on regime variables**. Each hypothesis has the form:

> *Signal feature F, gated by regime variable G {<,>} threshold T, predicts h-second forward returns.*

### Signal Features (directional predictors)

Order book imbalance at varying depth levels. Formulation from
Cont, Stoikov & Talreja (2010), "A stochastic model for order book dynamics":

```
                 V_bid(d) - V_ask(d)
  OBI(d)   =   ----------------------
                 V_bid(d) + V_ask(d)
```

where `V_bid(d) = sum of bid volume at levels 1..d` and `d in {1, 5, 10}`.

| Feature | Depth | Interpretation |
|---------|-------|----------------|
| `imbalance_qty_l1` | L1 (touch) | Instantaneous buy/sell pressure at best quote |
| `imbalance_qty_l5` | L1-5 | Near-book pressure (most predictive in our data) |
| `imbalance_qty_l10` | L1-10 | Deep book pressure (slower, more persistent) |
| `imbalance_depth_weighted` | Weighted | `sum(V * 1/(1 + dist_bps/10))` — near-touch emphasis |
| `imbalance_notional_l5` | L1-5 (USD) | Dollar-value asymmetry |
| `flow_aggressor_ratio_5s` | Trades | Buy-initiated volume / total volume, 5s window |
| `toxic_flow_imbalance` | Trades | VPIN-derived toxic flow directional imbalance |

### Regime Gates (conditional filters)

The agent tests whether restricting signal evaluation to specific **microstructure
regimes** improves predictability. Each gate partitions the data by a quintile
threshold on a conditioning variable.

**Entropy features** — information content of the order book and trade flow.
Based on Shannon (1948) and Bandt & Pompe (2002), "Permutation entropy":

```
  H(X)  =  - sum_{i=1}^{n} p_i * log(p_i)
```

| Gate Feature | Formula | Hypothesis |
|-------------|---------|------------|
| `ent_book_shape` | Shannon entropy of depth proportions across price levels | Low entropy = concentrated liquidity = predictable flow |
| `ent_tick_{5s,30s,1m}` | Shannon entropy of {up, down, neutral} tick directions | Low entropy = directional persistence |
| `ent_permutation_returns_16` | Bandt-Pompe ordinal pattern entropy, order m=3, lag=1, 16 returns | Low = deterministic return structure |
| `ent_spread_dispersion` | Shannon entropy of binned spread values | Low = tight spread clustering |

**Illiquidity features** — market impact and transaction cost proxies.
Based on Kyle (1985), "Continuous auctions and insider trading" and
Amihud (2002), "Illiquidity and stock returns":

| Gate Feature | Formula | Hypothesis |
|-------------|---------|------------|
| `illiq_kyle_100` | Kyle's lambda: `Cov(dP, dV) / Var(dV)` over 100 ticks | High lambda = thin market, signals have more impact |
| `illiq_composite` | Weighted combination of Kyle + Amihud + spread | Composite illiquidity regime |
| `illiq_amihud_100` | Amihud ratio: `abs(r) / volume` over 100 ticks | Price impact per unit volume |

**Toxicity features** — adverse selection and informed trading.
Based on Easley, Lopez de Prado & O'Hara (2012), "Flow toxicity and
liquidity in a high-frequency world":

| Gate Feature | Formula | Hypothesis |
|-------------|---------|------------|
| `toxic_vpin_50` | Volume-synchronized PIN, 50-bucket | High VPIN = informed trading active |
| `toxic_adverse_selection` | Fill-weighted slippage of recent trades | High = toxic flow period |
| `toxic_index` | Composite toxicity score | Aggregate informed-trading regime |

**Volatility features** — realized variance and vol-of-vol:

| Gate Feature | Formula | Hypothesis |
|-------------|---------|------------|
| `vol_returns_{1m,5m}` | `sqrt(sum(r^2) / N)` over 1m/5m windows (Parkinson, 1980) | Low vol = mean-reversion regime |
| `vol_ratio_short_long` | `vol_1m / vol_5m` | >1 = vol accelerating, regime shift |

**Derived regime features** — HMM state and confidence:

| Gate Feature | Source | Hypothesis |
|-------------|--------|------------|
| `derived_regime_type_score` | Gaussian HMM state posterior | Regime-specific signal quality |
| `derived_regime_confidence` | Max posterior probability | High confidence = stable regime |

### Search Space

The systematic generator produces hypotheses from the Cartesian product:

```
  7 signal features  x  17 gate features  x  4 thresholds  x  2 directions  =  952 hypotheses
```

Thresholds are unconditional quintile breakpoints: `{P20, P40, P60, P80}`.
Directions are `{<, >}` — e.g., `ent_book_shape < P20` selects the lowest-entropy 20% of observations.

## Inputs

| Input | Source | Format |
|-------|--------|--------|
| Order book snapshots | Hyperliquid L2 WebSocket, 100ms emission | Parquet in `data/features/YYYY-MM-DD/` |
| 191 features per tick | Rust ingestor (`ing`) real-time computation | Columns in Parquet files |
| Spectral reports | `nat spannung spectral` (PSD, ACF, coherence) | JSON in `reports/spannung/` |
| Regime screen reports | `nat spannung regime` (quintile screening) | JSON in `reports/spannung/` |
| Walk-forward reports | `nat profile scalp --forward-test` (5-fold CV) | JSON in `reports/profiler/` |

### Data Requirements

- Minimum 4 hours of gap-free data per symbol per date
- At least 2 dates for temporal replication
- 3 symbols (BTC, ETH, SOL) for cross-asset replication

## Outputs

| Output | Path | Content |
|--------|------|---------|
| Hypothesis log | `data/agent/hypotheses.json` | Append-only log of all tested claims |
| Signal registry | `data/agent/registry.json` | Validated signals (passed all 3 gates) |
| Data manifest | `data/agent/manifest.json` | Available dates, hours, symbols |
| Generator stats | `data/agent/generator_stats.json` | Per-generator hit rates (Beta prior) |
| Agent state | `data/agent/agent_state.json` | Daemon phase, cycle count, current experiment |

### Registered Signal Schema

```json
{
  "name": "imbalance_qty_l1 gated by ent_book_shape<P40 predicts 5s returns",
  "features": ["imbalance_qty_l1"],
  "regime_gate": "ent_book_shape<P40",
  "extraction": "raw",
  "horizon_s": 5.0,
  "expected_ic": 0.488,
  "symbols": ["BTC", "ETH", "SOL"],
  "status": "validated",
  "discovery_date": "2026-05-15",
  "hypothesis_id": "HYP-SYS-25ba20e3"
}
```

## Three-Gate Replication Protocol

Every hypothesis must pass three independent gates before registration.
This protocol is designed to control the false discovery rate under
multiple testing (Harvey, Liu & Zhu, 2016, "... and the Cross-Section of
Expected Returns").

```
  DISCOVERY  -->  TEMPORAL REPLICATION  -->  SYMBOL REPLICATION  -->  REGISTER
      |                  |                         |
      v                  v                         v
  GRAVEYARD          GRAVEYARD                 GRAVEYARD
```

### Gate 1: Discovery

Run the test protocol on the primary symbol (BTC) with the latest data:

```
  nat spannung regime --data data/features/{latest} --symbol BTC
  nat profile scalp --symbol BTC --data data/features/{latest} --forward-test
```

**Pass condition**: Information Coefficient `|IC| >= 0.10`.

The IC is the Spearman rank correlation between the signal and forward returns:

```
                    6 * sum(d_i^2)
  IC  =  1  -  -----------------------
                   n * (n^2 - 1)
```

where `d_i = rank(signal_i) - rank(return_i)` (Spearman, 1904).

### Gate 2: Temporal Replication

Re-run discovery on 2 additional dates from the manifest:

```
  nat spannung regime --data data/features/{date_k} --symbol BTC    for k in {1,2}
```

**Pass condition**: `nat` command succeeds on >= `min_oos_dates` (default 1) other dates.

This tests out-of-sample temporal stability. A signal that only works on one
date is likely a statistical artifact (White, 2000, "A reality check for data
snooping").

### Gate 3: Symbol Replication

Re-run on ETH and SOL:

```
  nat spannung regime --data data/features/{latest} --symbol {ETH,SOL}
```

**Pass condition**: IC gate passes on >= `min_symbols - 1` (default 1) other symbols.

Cross-asset replication is the strongest evidence of a genuine microstructure
effect vs. symbol-specific overfitting. The theoretical basis is that order
book imbalance is a **structural** property of limit order markets, not specific
to any single asset (Gueant, Lehalle & Fernandez-Tapia, 2012, "Dealing with
the inventory risk").

## Five Hypothesis Generators

### 1. Systematic Screener (`generators/systematic.py`)

Exhaustive (feature x gate x threshold x direction) search.
Priority boosted for `ent_book_shape` gates (empirically strongest).

### 2. Spectral Anomaly Detector (`generators/spectral.py`)

Monitors frequency-domain characteristics against expected baselines.
Emits hypotheses when anomalies are detected:

- **PSD slope** outside `[-2.2, -1.5]` (expected: ~-1.85 for fractional Brownian motion)

  ```
    S(f) ~ f^beta,   beta in [-2.2, -1.5]   (brown noise)
  ```

  Reference: Mandelbrot & Van Ness (1968), "Fractional Brownian motions,
  fractional noises and applications".

- **OU half-life** outside `[2, 15]` seconds (expected: BTC ~7.3s, ETH ~5.3s, SOL ~3.3s)

  ```
    dX_t = -theta * X_t * dt + sigma * dW_t
    t_half = ln(2) / theta
  ```

  Reference: Uhlenbeck & Ornstein (1930), "On the theory of the Brownian motion".

### 3. Regime Transition Detector (`generators/regime.py`)

Tests whether IC improves after HMM state transitions.
Uses Gaussian HMM with Baum-Welch EM fitting (Rabiner, 1989,
"A tutorial on hidden Markov models").

```
  H_0: IC(post-transition) = IC(pre-transition)
  H_1: IC(post-transition) > IC(pre-transition)
```

### 4. Cross-Asset Prober (`generators/cross_asset.py`)

Tests lead-lag relationships between BTC, ETH, and SOL at the
68-second coherence frequency (empirically identified dominant frequency).

```
  C_xy(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))
```

where `S_xy` is the cross-spectral density. Reference: Priestley (1981),
"Spectral Analysis and Time Series".

### 5. Failure Recycler (`generators/recycler.py`)

Re-examines graveyard hypotheses when failure conditions change:
- `insufficient_data` -- re-queue when new data accumulates (50% margin)
- `no_replication` -- re-queue when new dates become available
- `cost_killed` -- re-queue when complementary signals are discovered

## Generator Budget Allocation

Generator selection uses a Beta-prior multi-armed bandit
(Thompson, 1933, "On the likelihood that one unknown probability
exceeds another"):

```
  weight(g) = (successes(g) + 1) / (attempts(g) + 2)     # E[Beta(a, b)]
  budget_fraction(g) = weight(g) / sum(weights)
```

This steers hypothesis generation toward productive generators
without abandoning exploration (the +1/+2 Laplace prior ensures
all generators maintain nonzero probability).

## Priority Scoring

```
  priority = ic_gain * novelty * data_readiness

  novelty = 1 / (1 + max_corr_with_registry)     # [0, 1], higher = more novel
  data_readiness = min(available_h, required_h) / required_h
```

The `ent_book_shape` gates receive a +0.3 priority boost (empirically
established as the strongest single regime factor across BTC, ETH, and
SOL). Low-threshold conditions (`< P20`, `< P40`) receive a +0.1 boost
(low-entropy regimes show larger IC uplift).

## CLI

```bash
nat agent start       # launch daemon (cycles every hour)
nat agent stop        # graceful shutdown (SIGTERM, finishes current experiment)
nat agent once        # run single cycle (for testing)
nat agent status      # phase, cycle count, registry size, generator stats
nat agent queue       # queued hypotheses by priority
nat agent registry    # validated signals
nat agent graveyard   # failed hypotheses with reasons
```

## References

1. Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31-56.
2. Bandt, C. & Pompe, B. (2002). Permutation entropy: A natural complexity measure for time series. *Physical Review Letters*, 88(17), 174102.
3. Cont, R., Stoikov, S. & Talreja, R. (2010). A stochastic model for order book dynamics. *Operations Research*, 58(3), 549-563.
4. Easley, D., Lopez de Prado, M. & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies*, 25(5), 1457-1493.
5. Gueant, O., Lehalle, C.A. & Fernandez-Tapia, J. (2012). Dealing with the inventory risk: A solution to the market making problem. *Mathematics and Financial Economics*, 4(7), 477-507.
6. Harvey, C.R., Liu, Y. & Zhu, H. (2016). ... and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5-68.
7. Kyle, A.S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.
8. Mandelbrot, B.B. & Van Ness, J.W. (1968). Fractional Brownian motions, fractional noises and applications. *SIAM Review*, 10(4), 422-437.
9. Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61-65.
10. Priestley, M.B. (1981). *Spectral Analysis and Time Series*. Academic Press.
11. Rabiner, L.R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.
12. Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.
13. Spearman, C. (1904). The proof and measurement of association between two things. *American Journal of Psychology*, 15(1), 72-101.
14. Thompson, W.R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3-4), 285-294.
15. Uhlenbeck, G.E. & Ornstein, L.S. (1930). On the theory of the Brownian motion. *Physical Review*, 36(5), 823-841.
16. White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.
17. Zunino, L. et al. (2009). Forbidden patterns, permutation entropy and stock market inefficiency. *Physica A*, 388(14), 2854-2864.
