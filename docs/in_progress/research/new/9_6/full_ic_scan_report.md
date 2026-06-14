# Full 236-Feature IC Horizon Scan — Cross-Symbol Results

**Date**: 2026-06-09
**Data**: 3 days (2026-05-19 to 2026-05-21), 2.17M ticks per symbol, 100ms resolution
**Method**: Spearman rank IC, subsampled every 100 ticks (~10s), 21,708 evaluation points
**Symbols**: BTC, ETH, SOL (scanned independently)

## Summary

207 live features (48 all-NaN) were scanned at 6 horizons (1s, 5s, 30s, 1m, 5m, 15m) for both directional IC (vs signed return) and volatility IC (vs |return|). The order book imbalance cluster dominates directional prediction at 1-5s with IC 0.40-0.47 across all 3 symbols. The signal is universal, not symbol-specific.

## Feature Classification

| Category | BTC | ETH | SOL | Description |
|----------|-----|-----|-----|-------------|
| **dir_fast** | 29 | 29 | 28 | Directional signal at 1-30s |
| **dir_medium** | 11 | 11 | 10 | Directional at 1-5min |
| **dir_slow** | 7 | 3 | 14 | Directional at 5-15min |
| **volatility** | 68 | 75 | 68 | Predicts magnitude, not direction |
| **weak** | 11 | 9 | 7 | Marginal signal |
| **no_signal** | 33 | 32 | 32 | Nothing at any horizon |
| **all_nan** | 48 | 48 | 48 | No data produced |

## Top Directional Features (cross-symbol)

Peak IC at 1-5s horizon, all p < 0.01:

| Feature | BTC | ETH | SOL | Horizon |
|---------|-----|-----|-----|---------|
| **imbalance_qty_l1** | +0.453 | +0.447 | +0.466 | 1-5s |
| **imbalance_qty_l5** | +0.456 | +0.431 | +0.453 | 5s |
| **imbalance_notional_l5** | +0.456 | +0.431 | +0.453 | 5s |
| **imbalance_orders_l5** | +0.435 | +0.438 | +0.455 | 1-5s |
| **imbalance_depth_weighted** | +0.434 | +0.410 | +0.425 | 5s |
| **imbalance_qty_l10** | +0.434 | +0.410 | +0.424 | 5s |
| **raw_ask_depth_5** | -0.424 | -0.404 | -0.415 | 5s |
| **raw_ask_orders_5** | -0.421 | -0.405 | -0.424 | 1-5s |
| **raw_bid_depth_5** | +0.420 | +0.405 | +0.406 | 5s |
| **imbalance_pressure_ask** | -0.417 | -0.387 | -0.418 | 5s |
| **raw_bid_orders_5** | +0.416 | +0.406 | +0.425 | 1-5s |
| **raw_bid_depth_10** | +0.409 | +0.379 | +0.368 | 5s |
| **imbalance_pressure_bid** | +0.404 | +0.377 | +0.395 | 5s |
| **raw_ask_depth_10** | -0.399 | -0.360 | -0.396 | 5s |
| **cross_obi_mean** | +0.354 | +0.334 | +0.331 | 1-5s |
| **micro_queue_position_ask** | -0.312 | -0.292 | -0.291 | 1-5s |
| **micro_queue_position_bid** | +0.307 | +0.282 | +0.291 | 5s |
| **flow_vwap_deviation** | -0.287 | -0.206 | -0.188 | 1s |
| **cross_obi_divergence** | +0.240 | +0.163 | +0.202 | 5s |
| **micro_obi_velocity** | +0.192 | +0.181 | +0.208 | 1s |
| **ent_permutation_imbalance_16** | -0.170 | -0.173 | -0.177 | 1s |
| **flow_aggressor_ratio_5s** | +0.112 | +0.109 | +0.111 | 1-5s |

### Signal Interpretation

- **Positive imbalance** (more bids than asks) predicts upward price movement within 1-5 seconds
- **Higher ask depth** predicts downward moves — sell pressure interpretation
- **VWAP deviation** is mean-reverting: price above VWAP predicts down
- **OBI velocity** (rate of change of imbalance) adds incremental signal over level
- **Cross-symbol OBI mean** captures market-wide imbalance shifts

## Signal Decay by Symbol

```
Peak directional IC at each horizon:

         BTC      ETH      SOL
  1s    0.447    0.447    0.466
  5s    0.456    0.438    0.412
 30s    0.261    0.219    0.190
  1m    0.187    0.160    0.135
  5m    0.090    0.074    0.056
 15m    0.056    0.031    0.036

Half-life:  ~30s     ~20s     ~15s
```

SOL decays fastest (thinner book, faster price discovery). BTC retains signal longest.

## Top Volatility Features (cross-symbol)

Peak IC at 5s horizon, all p < 0.01:

| Feature | BTC | ETH | SOL | Type |
|---------|-----|-----|-----|------|
| **hawkes_intensity** | +0.345 | +0.281 | +0.247 | Hawkes process |
| **vol_returns_5m** | +0.328 | +0.326 | +0.293 | Realized vol |
| **vol_parkinson_5m** | +0.318 | +0.312 | +0.281 | Range-based vol |
| **flow_count_30s** | +0.318 | +0.281 | +0.253 | Trade intensity |
| **illiq_trade_count** | +0.318 | +0.286 | +0.261 | Microstructure |
| **vol_returns_1m** | +0.318 | +0.323 | +0.289 | Realized vol |
| **ent_permutation_returns_32** | +0.315 | +0.312 | +0.286 | Entropy |
| **vol_midprice_std_1m** | +0.295 | +0.288 | +0.244 | Midprice vol |
| **flow_intensity** | +0.291 | +0.224 | +0.181 | Trade intensity |
| **toxic_effective_spread** | +0.291 | +0.279 | +0.254 | Toxicity |

### Directional vs Volatility: Clean Separation

Imbalance features have **zero volatility IC** — they predict direction, not magnitude.
Volatility features have **zero directional IC** — they predict magnitude, not direction.
This orthogonality is consistent across all 3 symbols.

## Medium/Slow Directional Features

Features with directional IC that grows with horizon (opposite of imbalance):

| Feature | BTC 5m | BTC 15m | ETH 15m | SOL 15m | Interpretation |
|---------|--------|---------|---------|---------|----------------|
| **raw_spread_bps** | +0.074 | +0.139 | +0.179 | +0.086 | Wide spread → uncertainty resolves up |
| **trend_ema_short** | -0.076 | -0.142 | -0.188 | — | Mean reversion at multi-minute scale |
| **regime_range_pos_24h** | -0.048 | -0.109 | -0.100 | -0.145 | High in range → mean reverts down |
| **regime_accumulation_score** | +0.048 | +0.109 | — | — | Accumulation predicts upward drift |

## All-NaN Features (48)

These features exist in the Rust schema but produce no data. Same 48 across all symbols:

- **Whale flow** (12): whale_net_flow_1h/4h/24h, whale_flow_normalized_1h/4h, whale_flow_momentum, whale_flow_intensity, whale_flow_roc, whale_buy_ratio, whale_directional_agreement, active_whale_count, whale_total_activity
- **Liquidation risk** (13): liquidation_risk_above/below_1/2/5/10pct, liquidation_asymmetry, liquidation_intensity, positions_at_risk_count, largest_position_at_risk, nearest_cluster_distance
- **Concentration** (15): top5/10/20/50_concentration, herfindahl_index, gini_coefficient, theil_index, whale_retail_ratio, whale_fraction, whale_avg_size_ratio, concentration_change_1h, hhi_roc, concentration_trend, position_count, whale_position_count
- **GMM regime** (8): regime, regime_prob_accumulation/markup/distribution/markdown/ranging, regime_confidence, regime_entropy

## Independent Signal Groups

Many of the 29 fast-directional features are highly correlated. The likely independent signal axes are:

1. **Order book imbalance** (~IC 0.45) — imbalance_qty_l1 as representative. All L1/L5/L10/depth-weighted/notional variants capture the same signal.
2. **Raw depth asymmetry** (~IC 0.42) — raw_bid_depth_5 vs raw_ask_depth_5. Correlated with imbalance but measures absolute levels, not ratios.
3. **Cross-symbol imbalance** (~IC 0.35) — cross_obi_mean. Market-wide order flow signal, partially independent of single-symbol imbalance.
4. **Queue dynamics** (~IC 0.31) — micro_queue_position_bid/ask. Where in the queue the best orders sit.
5. **VWAP deviation** (~IC 0.25) — flow_vwap_deviation. Mean-reversion signal, mechanistically different from order book state.
6. **OBI velocity** (~IC 0.19) — micro_obi_velocity. Rate of change of imbalance, captures momentum of order flow shifts.
7. **Aggressor flow** (~IC 0.11) — flow_aggressor_ratio_5s. Who is hitting whom in recent trades.
8. **Imbalance entropy** (~IC 0.17) — ent_permutation_imbalance_16. Predictability of the imbalance sequence itself.

## Implications for Limit Order Strategy

1. **Primary signal**: Order book imbalance at L1 (IC 0.45-0.47 at 1-5s). Place limit buy when imbalance is positive (more bids), limit sell when negative.
2. **Vol gate**: Use hawkes_intensity or flow_count_30s to filter for high-magnitude periods, reducing adverse selection on fills.
3. **Secondary signals**: Cross-symbol OBI, queue position, and VWAP deviation provide incremental alpha for order placement refinement.
4. **Symbol-specific tuning**: SOL requires faster reaction (15s half-life) vs BTC (30s half-life). Different stale-signal timeouts per symbol.
5. **48 dead features** need investigation — whale/liquidation/concentration data pipeline is broken.

## Methodology

- Spearman rank correlation (non-parametric, robust to outliers)
- Forward returns: `(price[t+h] / price[t] - 1) * 10000` in basis points
- Subsampled every 100 ticks (~10s) to reduce autocorrelation
- Significance: p < 0.01 (two-sided test)
- Data: Hyperliquid perpetual futures, 100ms tick resolution, 3 consecutive full days
- Full results saved to `reports/full_ic_scan_{btc,eth,sol}.json`
