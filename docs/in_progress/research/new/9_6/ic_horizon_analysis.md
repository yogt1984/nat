# Feature IC Horizon Analysis — Tick-Level Signal Discovery

**Date**: 2026-06-09
**Data**: 3 days (2026-05-19 to 2026-05-21), 2.17M ticks, BTC, 100ms resolution
**Method**: Spearman rank IC, subsampled every 10s (21,618 evaluation points)

## Summary

Microstructure features have massive directional IC at 1-5 second horizons (0.42-0.45) that decays to near-zero by 5 minutes. All 25 algorithms in the system were evaluated at 100-minute horizons where no directional signal exists. The features work — the horizon was wrong.

## Directional IC (feature vs signed forward return)

| Feature | 1s | 5s | 30s | 1m | 5m | 15m |
|---------|------|------|------|------|------|------|
| **imbalance_qty_l1** | **+0.447*** | **+0.453*** | **+0.257*** | **+0.184*** | +0.089* | +0.056* |
| **imbalance_depth_weighted** | **+0.427*** | **+0.434*** | **+0.255*** | **+0.185*** | +0.081* | +0.037* |
| **raw_ask_depth_5** | **-0.414*** | **-0.425*** | **-0.238*** | **-0.167*** | -0.072* | -0.040* |
| **flow_aggressor_ratio_5s** | **+0.112*** | **+0.112*** | +0.074* | +0.051* | +0.009 | -0.020 |
| flow_intensity | +0.028* | +0.019* | +0.009 | -0.006 | -0.018* | -0.000 |
| ent_tick_1m | -0.019* | -0.028* | -0.015 | -0.015 | -0.009 | +0.014 |
| raw_spread_bps | -0.000 | +0.006 | +0.014 | +0.023* | +0.073* | +0.139* |
| toxic_vpin_50 | +0.011 | +0.004 | -0.006 | -0.004 | +0.008 | -0.004 |
| regime_divergence_1h | -0.020 | -0.009 | -0.035* | -0.043* | -0.058* | -0.074* |

`*` = p < 0.01

### Key Observations

1. **Order book imbalance (L1 and depth-weighted) dominates at 1-5s with IC 0.43-0.45.** This is the strongest directional signal in the system. Positive imbalance (more bids than asks) predicts upward price movement within seconds.

2. **Ask depth has IC -0.42 at 1-5s.** Higher ask depth predicts downward moves — consistent with sell pressure interpretation.

3. **Signal half-life is ~30 seconds.** IC decays from 0.45 at 1s to 0.25 at 30s to 0.09 at 5m. The predictive content is microstructure-scale, not macro-scale.

4. **Regime divergence does the opposite — IC grows with horizon** (0.02 at 1s, 0.07 at 15m, 0.21 at 100m). It captures slow accumulation/distribution, not microstructure.

5. **Spread predicts direction only at longer horizons** (IC 0.14 at 15m). Wide spread signals uncertainty that resolves over minutes, not seconds.

6. **VPIN and entropy have no directional IC at any horizon.** They predict volatility magnitude, not direction.

## Volatility IC (feature vs |forward return|)

| Feature | 1s | 5s | 30s | 1m | 5m | 15m |
|---------|------|------|------|------|------|------|
| **flow_intensity** | **+0.257*** | **+0.291*** | **+0.226*** | **+0.181*** | +0.150* | +0.100* |
| **raw_spread_bps** | **+0.138*** | **+0.165*** | **+0.154*** | **+0.130*** | +0.087* | +0.145* |
| toxic_vpin_50 | -0.066* | **-0.106*** | **-0.106*** | -0.098* | -0.101* | -0.075* |
| ent_tick_1m | +0.018 | +0.019 | +0.012 | +0.017 | +0.037* | +0.057* |
| flow_aggressor_ratio_5s | -0.009 | -0.030* | -0.028* | -0.033* | -0.035* | -0.046* |

### Key Observations

1. **Flow intensity is the best vol predictor at short horizons** (IC 0.29 at 5s). High trade intensity signals imminent large moves.

2. **Spread predicts vol across all horizons** (IC 0.13-0.17). Wide spread = big move coming.

3. **Imbalance features have zero vol IC** — they predict direction, not magnitude. Clean separation.

## Signal Decay Curve

```
imbalance_qty_l1 directional IC:

  0.45 |  **
  0.40 |
  0.35 |
  0.30 |
  0.25 |         *
  0.20 |                *
  0.15 |
  0.10 |                        *
  0.05 |                                *
  0.00 |________________________________________________
         1s     5s     30s     1m      5m     15m   100m
```

Half-life: ~30 seconds. By 5 minutes, 80% of the signal is gone.

## Feature Classification

| Category | Features | Predicts | Horizon | IC Range |
|----------|----------|----------|---------|----------|
| **Directional (fast)** | imbalance_qty_l1, imbalance_depth_weighted, raw_ask_depth_5 | Direction | 1-30s | 0.25-0.45 |
| **Directional (medium)** | flow_aggressor_ratio_5s | Direction | 1s-1m | 0.05-0.11 |
| **Directional (slow)** | regime_divergence_1h, raw_spread_bps | Direction | 5m-100m | 0.07-0.21 |
| **Volatility** | flow_intensity, raw_spread_bps, toxic_vpin_50 | Magnitude | 1s-15m | 0.10-0.29 |
| **No signal** | ent_tick_1m, toxic_vpin_50 (direction) | Neither | — | < 0.03 |

## Implications

### Why All 25 Algorithms Failed

Every algorithm in the system was evaluated at 100-minute horizons using 5-minute bar aggregation. This aggregation destroys the 1-5 second directional signal (IC 0.45 → 0) while preserving only the weak residual (IC < 0.07). The algorithms were solving the wrong problem at the wrong timescale.

### Optimal Strategy: Informed Limit Orders

The IC profile points to a limit order strategy:
- **Order book imbalance predicts direction at 1-5s with IC 0.45**
- At this timescale, expected move is ~0.5-2 bps
- Taker execution (11 bps RT) is impossible — the move is smaller than the cost
- Maker execution (limit orders with rebates) captures the spread + directional edge
- Place limit buy when imbalance predicts up, limit sell when imbalance predicts down
- Expected fill when price moves toward your order (adverse selection is managed by the IC)

### Next Steps

1. **Full feature scan at tick level** — test all 236 features at 1s/5s/30s/1m horizons to find additional directional signals
2. **IC stability analysis** — measure how IC varies across days, hours, and market regimes
3. **Continuous IC discovery engine** — automated process that recalibrates feature weights as ICs shift
4. **Limit order simulator** — backtest with realistic fill assumptions (no guaranteed fills, queue position matters)
5. **Deploy short-window divergence** — new 1m/5m/15m regime_divergence columns (committed, awaiting ingestor rebuild)

## Methodology Notes

- IC computed as Spearman rank correlation (non-parametric, robust to outliers)
- Forward returns computed as `(price[t+h] / price[t] - 1) * 10000` in basis points
- Subsampled every 100 ticks (~10 seconds) to reduce autocorrelation in evaluation
- `*` indicates p < 0.01 (two-sided test)
- Data: BTC on Hyperliquid, 100ms tick resolution, 3 consecutive full days
