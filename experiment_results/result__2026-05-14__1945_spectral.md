# Spannung Spectral Analysis — Experiment Results

**Date**: 2026-05-14
**Duration**: Phase D, single session
**Status**: Concluded — exploitable spectral structure found

## Discovery

L1 book imbalance has a concentrated power spectrum (brown noise, H=0.43) with
almost all predictive power (IC=0.45) in the ultra-low frequency band (0.005–0.1 Hz,
periods 10–200s). The signal mean-reverts with OU half-life 5–7s and has dominant
coherence with returns at 0.015 Hz (~68s cycles). These characteristics replicate
across dates, suggesting structural market microstructure — not transient.

## Key Numbers

| Metric | 2026-05-12 (10h) | 2026-05-11 (24h) |
|--------|-------------------|-------------------|
| Noise slope | -1.86 (brown) | -1.86 (brown) |
| Hurst exponent | 0.431 | 0.430 |
| OU half-life | 7.3s | 5.4s |
| Spectral entropy | 0.456 | 0.499 |
| ACF(1s) | 0.895 | 0.860 |
| ACF(5s) | 0.597 | 0.506 |
| ACF(10s) | 0.386 | 0.287 |
| First zero crossing | 44.9s | >60s |

## Band-Filtered IC (both dates consistent)

| Band | Freq range | IC (1s fwd) | IC (5s fwd) | IC IR (5s) |
|------|-----------|-------------|-------------|------------|
| ultra_low | 0.005–0.1 Hz | +0.30 | **+0.45** | **4.1** |
| low | 0.05–0.5 Hz | +0.27 | +0.10 | 1.1 |
| mid | 0.5–2.0 Hz | -0.02 | ~0 | — |
| high | 2.0–4.5 Hz | ~0 | ~0 | — |

## Coherence Peaks (imbalance → returns)

| Frequency | Period | Coherence | Phase lead |
|-----------|--------|-----------|------------|
| 0.015 Hz | 68s | 0.26 | 100–500 ms |
| 0.024 Hz | 41s | 0.22 | 100–200 ms |
| 0.044 Hz | 23s | 0.22 | 200–500 ms |

## Implications

1. **Predictive power is frequency-localized** — ultra-low band carries IC=0.45, mid/high bands are noise
2. **Bar aggregation failed because it averaged across all bands** — bandpass extraction preserves the signal
3. **OU half-life (5–7s) defines natural market-making refresh rate**
4. **Kalman filter on slow component** can recover latency-lost IC since ultra-low periods (10–200s) dwarf execution latency
5. **Zero-fee pairs + spectral extraction = viable path** — spread (~0.5–1 bps) is the only remaining cost

## Tools

```bash
nat spannung spectral --data data/features/2026-05-12 --symbol BTC
nat spannung spectral --data data/features/2026-05-11 --symbol BTC
```
