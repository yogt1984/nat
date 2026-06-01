# Gauntlet Analysis — 2026-06-01

**Date:** 2026-06-01
**Test window:** 00:00-04:40 UTC (4.7h, 56 bars/symbol)
**Training:** 3-day walk-forward (2026-05-28, 2026-05-29, 2026-05-31)
**Cost model:** 1.61 bps RT (Binance VIP9)
**Symbols:** BTC, ETH, SOL
**Algorithms tested:** 19 (18 generic + 3f_liquidity)

---

## Ranking Summary

| Rank | Algorithm | BTC | ETH | SOL | Total | Trades | Sym+ | bps/trade |
|------|-----------|-----|-----|-----|-------|--------|------|-----------|
| 1 | hawkes_intensity | -40.0 | +386.2 | +609.6 | +955.8 | 26 | 2/3 | +36.8 |
| 2 | propagator | -249.7 | +1150.2 | -418.3 | +482.2 | 53 | 1/3 | +9.1 |
| 3 | spread_decomp | +176.4 | +229.0 | +50.9 | +456.3 | 45 | 3/3 | +10.1 |
| 4 | optimal_entry | +37.6 | +209.4 | +194.0 | +441.0 | 108 | 3/3 | +4.1 |
| 5 | funding_reversion | -11.2 | +159.8 | +124.3 | +272.9 | 105 | 2/3 | +2.6 |
| 6 | kalman_imbalance | -8.5 | -19.2 | +270.6 | +242.9 | 30 | 1/3 | +8.1 |
| 7 | switching_ou | -31.0 | +78.8 | +16.4 | +64.2 | 39 | 2/3 | +1.6 |
| 8 | 3f_liquidity | +233.2 | -396.8 | +196.5 | +32.9 | 84 | 2/3 | +0.4 |
| 9 | surprise_signal | -2.5 | -335.2 | +321.8 | -15.9 | 103 | 1/3 | -0.2 |
| 10 | weighted_ofi | -194.2 | -243.8 | +410.5 | -27.5 | 91 | 1/3 | -0.3 |
| 11 | vpin_regime | -209.9 | -234.5 | +280.5 | -163.9 | 81 | 1/3 | -2.0 |
| 12 | regime_gated | -111.1 | +35.1 | -343.3 | -419.3 | 54 | 1/3 | -7.8 |
| 13 | jump_detector | -11.2 | +60.7 | -494.2 | -444.7 | 105 | 1/3 | -4.2 |
| 14 | bipower_jump | -61.1 | +114.9 | -554.2 | -500.4 | 78 | 1/3 | -6.4 |
| 15 | trade_through | -141.3 | -486.9 | -472.8 | -1101.0 | 47 | 0/3 | -23.4 |
| 16 | oi_divergence | -360.3 | -357.3 | -433.6 | -1151.2 | 93 | 0/3 | -12.4 |
| -- | convolver | 0 | 0 | 0 | 0 | 0 | -- | -- |
| -- | entropy_momentum | 0 | 0 | 0 | 0 | 0 | -- | -- |
| -- | cascade_probability | -- | -- | -- | -- | -- | -- | -- |

---

## Key Findings

### Consistency vs magnitude

Only two algorithms were profitable on all 3 symbols: **spread_decomp** and **optimal_entry**. This is the harder bar to clear — total PnL can be dominated by a single lucky print, but cross-symbol consistency requires genuine signal.

Hawkes_intensity ranked #1 by total PnL (+955.8 bps) but on only 26 trades, making it statistically thin. Propagator's +482 bps was almost entirely one +1150 ETH print. These need multi-day validation via `nat gauntlet run` before drawing conclusions.

### Regime shift from May results

The ALGORITHMS.md catalogue from 2026-05-23 ranked hawkes_intensity and spread_decomp as Tier 3 (no edge). On June 1 data they rank #1 and #3. Possible explanations:
1. **Lookahead bias fixes** (commits `1470d9c`, `5a9aa79`) corrected downward bias in backtests
2. **Market regime changed** between May and June
3. **Single-day noise** — one good day does not establish alpha

The gauntlet multi-day sweep will disambiguate.

### Dead signals

Three algorithms produced zero trades: convolver (no pattern match above threshold), entropy_momentum (no entropy regime triggered), cascade_probability (failed silently). These need investigation — they may have parameter issues with the 5-min bar aggregation used by the paper trader.

---

## Deep Dive: hawkes_intensity

### Functional logic

The Hawkes intensity algorithm models trade arrivals as a self-exciting point process where each trade increases the probability of future trades, with that excitement decaying exponentially over time (~7s half-life). It maintains two separate excitement processes — one for bid-side and one for ask-side — by splitting incoming trade events proportionally to the order book pressure on each side. The trading signal is the normalized difference between ask-side and bid-side intensities: when ask-side excitement significantly outpaces bid-side, it means buyers are clustering (informed buying cascade), so go long — and vice versa. It trades infrequently because the signal only fires when there is a genuine directional asymmetry in the burst dynamics, not on routine order flow noise.

### Mathematical framework

**Conditional intensity (Hawkes process with exponential kernel):**

    lambda(t) = mu + alpha * A(t)

where A(t) is the recursive excitement state:

    A(t) = exp(-beta * dt) * A(t-1) + N(t)

- mu: baseline intensity, estimated as rolling 300-tick mean of `flow_intensity`
- alpha: self-excitation amplitude (default 0.5, auto-tuned per regime)
- beta: decay rate (default 0.1/s, half-life = ln(2)/beta ~ 6.9s)
- N(t): event count from `flow_count_1s`

**Bid/ask decomposition:**

    bid_frac = |pressure_bid| / (|pressure_bid| + |pressure_ask| + eps)
    N_bid = N(t) * bid_frac
    N_ask = N(t) * (1 - bid_frac)

    lambda_bid = mu/2 + alpha * A_bid(t)
    lambda_ask = mu/2 + alpha * A_ask(t)

**Trading signal (Hawkes imbalance):**

    HI(t) = (lambda_ask - lambda_bid) / (lambda_ask + lambda_bid + eps)   in (-1, 1)

Positive = ask-side intensity dominates = buy pressure = long.

### Parameters

| Parameter | Default | Auto-tuned | Method |
|-----------|---------|------------|--------|
| beta (decay) | 0.1/s | Yes | -ln(lag-1 autocorr) / dt |
| alpha (excitation) | 0.5 | Yes | branching_ratio * beta, where branching = var/mean - 1 |
| baseline_window | 300 ticks | No | Fixed |

### Output features

| Feature | Range | Role |
|---------|-------|------|
| `alg_hawkes_intensity` | [0, inf) | Total event arrival rate |
| `alg_hawkes_baseline` | [0, inf) | Background rate (rolling mean) |
| `alg_hawkes_excitement` | [0, 1) | Fraction from self-excitation (phi -> 1 = cascade) |
| `alg_bid_ask_hawkes_imbalance` | (-1, 1) | **The trading signal** |

### Why it may work

Informed traders cluster. When someone with alpha starts buying, it triggers follow-on buys (other algos detecting flow, stops getting hit, liquidation cascades). Standard OFI treats each event equally; Hawkes gives exponentially more weight to recent bursts, capturing self-reinforcing dynamics of informed flow.

The 26-trade count means high selectivity — the signal fires only during genuine directional cascades, not minor imbalance fluctuations. This produces high bps/trade (+36.8) but low statistical power per day.

### Research directions

1. **MLE-fitted half-life per symbol.** The current method-of-moments estimate is rough. Hawkes MLE has a closed-form recursive likelihood — fit per symbol to learn whether BTC cascades decay faster than SOL cascades.
2. **Multi-scale kernels.** Replace single exponential with fast + slow: `alpha_fast * exp(-beta_fast * t) + alpha_slow * exp(-beta_slow * t)`. Captures both HFT reaction (sub-second) and retail cascade (30s+).
3. **Volume conditioning.** Weight events by `ctx_volume_ratio` so high-volume bursts excite more. Gate the signal on volume regime — cascades in quiet markets may be noise.
4. **Excitement threshold.** Only trade when `alg_hawkes_excitement > threshold` (e.g., 0.3), filtering out periods where the signal is dominated by baseline rather than self-excitation.

### References

- Bacry, E., Mastromatteo, I. & Muzy, J.-F. (2015) — "Hawkes processes in finance", Market Microstructure and Liquidity 1(1), 1550005.
- Lu, X. & Abergel, F. (2018) — "High-dimensional Hawkes processes for limit order books", Quantitative Finance 18(2), 177-188.

---

## Deep Dive: spread_decomp

### Functional logic

The spread decomposition algorithm breaks the bid-ask spread into two components: adverse selection (cost of trading against informed flow) and inventory/order-processing (the market maker's operational cost). It computes `adverse = effective_spread - lagged_realized_spread`, where the realized spread is lagged by one tick to avoid lookahead bias. When the adverse selection component is low, the current flow is mostly uninformed (retail, rebalancing) — the market maker is not being picked off — so it is a safer time to enter positions. The algorithm trades `low_long`: low adverse selection triggers a long entry because you are trading alongside uninformed flow rather than against smart money.

### Mathematical framework

**Effective spread** (observed at trade time):

    S_eff(t) = 2 * |P_trade(t) - M(t)|

where M(t) is the midprice at trade time.

**Realized spread** (observed after the trade):

    S_real(t) = 2 * sign(trade) * (P_trade(t) - M(t+dt))

Positive realized spread = market maker profited from the trade.

**Adverse selection component:**

    AS(t) = S_eff(t) - S_real(t-1)

The lag on realized spread is critical — `S_real(t)` at time t requires knowing the future midprice `M(t+dt)`, so using it contemporaneously would introduce lookahead bias. The fix (commit `5a9aa79`) lags it by one tick.

**Trend (EMA of adverse selection):**

    AS_trend(t) = alpha * AS(t) + (1 - alpha) * AS_trend(t-1)

where alpha = 2 / (span + 1), span = 100 ticks.

**Regime detection:**

    spread_regime(t) = 1 if AS(t) > P70(AS over 300 ticks) else 0

High regime = high informed trading = dangerous to enter.

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| ema_span | 100 | EMA window for adverse trend |
| regime_window | 300 | Rolling window for percentile |
| regime_percentile | 70 | Threshold for high-informed regime |

### Output features

| Feature | Range | Role |
|---------|-------|------|
| `alg_adverse_component` | (-inf, inf) | Adverse selection component |
| `alg_adverse_trend` | (-inf, inf) | EMA-smoothed adverse selection |
| `alg_spread_regime` | {0, 1} | 1 = high informed trading regime |

### Why it may work

The adverse selection component directly measures the information content of recent order flow. When AS is low, market makers are comfortable (their spreads are being realized) — this indicates uninformed flow dominance, which is precisely when a directional signal (from another source) is least likely to be front-run or adversely selected against.

Cross-symbol consistency (3/3 positive) suggests this is measuring a genuine microstructure property rather than fitting to idiosyncratic price patterns.

### Research directions

1. **Use as a meta-filter.** Rather than trading spread_decomp alone, use `alg_spread_regime = 0` (low informed trading) as a gate for other algorithms. E.g., only trade hawkes_intensity when adverse selection is below P30.
2. **Intraday seasonality.** Adverse selection likely varies by hour (higher around news, funding payments). Deseasonalize before computing the signal.
3. **Multi-level decomposition.** Current implementation uses effective/realized spread from L1. Extending to L5/L10 depth levels could capture informed flow that doesn't hit the best bid/ask.

### References

- Huang, R.D. & Stoll, H.R. (1997) — "The components of the bid-ask spread: A general approach", Review of Financial Studies 10(4), 995-1034.
- Hendershott, T., Jones, C.M. & Menkveld, A.J. (2011) — "Does algorithmic trading improve liquidity?", Journal of Finance 66(1), 1-33.

---

## Complementarity: hawkes_intensity + spread_decomp

These two signals measure orthogonal properties of the order flow:

| Property | hawkes_intensity | spread_decomp |
|----------|-----------------|---------------|
| Question answered | "Who is clustering — buyers or sellers?" | "Is the current flow informed or uninformed?" |
| Signal type | Directional (momentum from self-excitation) | Regime filter (adverse selection safety) |
| Trade frequency | Low (26 trades / 4.7h) | Medium (45 trades / 4.7h) |
| Strength | High conviction per trade (+36.8 bps/trade) | Consistent across symbols (3/3) |

A combined strategy — **hawkes direction gated by low adverse selection** — could be stronger than either alone. The hypothesis: trade the hawkes imbalance signal only when `alg_spread_regime = 0`, filtering out periods where informed traders are active and the cascade signal is likely adversely selected against.

This is testable as a new algorithm or as a hypothesis for the research agent.

---

## Caveats

1. **Single day, 4.7 hours.** All results here are from one test date. Statistical significance requires multi-day validation via `nat gauntlet run`.
2. **Low trade counts.** Hawkes had 26 trades — a handful of lucky trades could dominate. Need >= 100 trades across multiple dates for confidence.
3. **Regime shift from May.** Both hawkes and spread_decomp were Tier 3 (negative) in the May 23 ALGORITHMS.md. Either the lookahead bias fixes changed things, or the market regime shifted, or this is noise.
4. **Paper trader mechanics.** The generic paper trader uses z-score percentile thresholds (P20/P80) and 5-min bars. Different bar sizes or entry thresholds could change rankings materially.

---

## Next Steps

1. Run `nat gauntlet run` overnight to get multi-day statistics with Sharpe ratios
2. If hawkes and spread_decomp hold up, implement the combined hawkes+spread gate hypothesis
3. Investigate parameter learning for hawkes (MLE half-life, volume conditioning)
4. Debug cascade_probability, convolver, entropy_momentum zero-trade issues
