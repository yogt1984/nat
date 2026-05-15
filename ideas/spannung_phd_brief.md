# Spannung Research — PhD Brief & Findings Summary

**Date**: 2026-05-15
**Purpose**: Summary of findings for academic inquiry (ETH Zürich PhD)

## Research Arc Summary

### Phase A — Signal Discovery
Raw L1 book imbalance predicts 1–5s forward returns with IC=0.44–0.48 across BTC,
ETH, SOL. ~10x typical quant feature ICs. 100% positive IC windows over 10h. The
Spannung formulation (EWM flow / illiq) hurts — raw imbalance beats it by 8–10 IC
points. The denominator is redundant at tick resolution.

### Phase A.2 — Causality Validation
Smooth lag decay (0.48 → 0.39 at 100ms → 0.33 at 1s → 0.08 at 10s) confirms causal
signal with no look-ahead bias. Forward-shifted signal shows only marginal improvement.

### Phase A.3 — Out-of-Sample Generalization
IC=0.42–0.48 across 4 dates spanning 3 weeks. Persistent and structural.

### Phase B — Cost Reality
Per-trade edge 0.2–0.4 bps vs 7 bps taker round-trip. 20x gap. Regime gating with
simple median splits improves IC by 6% — insufficient.

### Phase C — Longer Horizons Fail
Bar-level aggregation (30s–15min) destroys IC (drops 2–3x). Zero IS/OOS overlap.
The signal is inherently tick-level.

### Phase D — Spectral Breakthrough
- Brown noise spectrum (slope=-1.86), Hurst H=0.43 (slightly mean-reverting)
- **IC=0.45 concentrated entirely in ultra-low band (0.005–0.1 Hz, periods 10–200s)**
- Mid and high frequency bands carry zero predictive power
- OU mean-reversion half-life = 5–7s
- Dominant coherence with returns at 0.015 Hz (68s cycles), phase lead 100–500ms
- All characteristics replicate identically across dates

### Phase E — Regime Screening Breakthrough
- `ent_book_shape` (order book shape entropy) is the dominant regime condition
- Single factor: IC lifts from 0.45 → 0.55 at 20–40% coverage
- Best combos: IC=0.63–0.67 at 1–7% coverage
- **Replicates perfectly across IS and OOS**
- Economic interpretation: structured book = informed positioning = imbalance is informative
- IC-persistence tradeoff: wide regimes (50%+ coverage) last 10–16s with IC=0.50,
  tight combos (1–7%) last 1–2s with IC=0.65

## Key Achievements

1. Built a complete research platform: 5 tools (`nat spannung`, `backtest`, `horizon`,
   `spectral`, `regime`)
2. Discovered that L1 book imbalance predictive power is frequency-localized in the
   ultra-low band
3. Identified `ent_book_shape` as a replicating regime condition with strong economic
   interpretation
4. Established the spectral/OU characterization of the signal (H=0.43, half-life 5–7s,
   dominant coherence at 68s)
5. Mapped the full IC-persistence Pareto frontier for regime-gated trading

## PhD Brief — ETH Zürich

**Subject: PhD Inquiry — Spectral Microstructure of Order Book Imbalance and Regime-Conditional Market Making**

Dear Professor [Name],

I am writing to inquire about PhD opportunities in your group. I have been conducting
independent research on the frequency-domain characteristics of order book imbalance
in cryptocurrency perpetual futures, and I believe the findings open a research direction
that intersects market microstructure theory, signal processing, and optimal market making.

### Summary of findings

Using a proprietary real-time data pipeline (Rust ingestor at 100ms resolution, 191
engineered features), I analyzed the spectral structure of L1 order book imbalance
across BTC, ETH, and SOL perpetual futures. The key results:

1. **Frequency localization of predictive power.** The imbalance signal exhibits brown
   noise characteristics (spectral slope beta=-1.86, Hurst H=0.43). Critically, the
   information coefficient (Spearman IC=0.45, IR=4.1) with forward returns is entirely
   concentrated in the ultra-low frequency band (0.005–0.1 Hz, periods 10–200s). Higher
   frequency components carry no predictive power. This explains why naive time-domain
   aggregation (e.g., bar-level averaging) destroys the signal — it mixes informative
   low-frequency content with high-frequency noise.

2. **Ornstein-Uhlenbeck dynamics.** The autocorrelation structure fits an OU process
   with half-life 5–7 seconds and first zero crossing at 45–60 seconds. The dominant
   cross-spectral coherence with returns occurs at 0.015 Hz (68-second cycles), with
   imbalance leading returns by 100–500ms at predictive frequencies. These parameters
   are remarkably stable across days.

3. **Order book entropy as a regime indicator.** A systematic screen of 17 microstructure
   features as regime conditions identified order book shape entropy as the dominant
   predictor of signal quality. Low book shape entropy (structured, non-random depth
   profile) lifts IC from 0.45 to 0.55+ (single condition) and to 0.63–0.67
   (multi-condition combinations). This replicates out-of-sample and has a clear
   economic interpretation: structured books reflect informed positioning, amplifying
   the information content of imbalance.

4. **The IC-persistence Pareto frontier.** High-IC regime windows (0.60+) persist for
   only 1–2 seconds, while moderate-IC regimes (0.50) persist for 10–16 seconds,
   creating a tradeoff between signal strength and executability that has implications
   for optimal market-making strategy design.

### Proposed research directions

- Theoretical framework connecting spectral PSD characteristics to optimal Kalman
  filter design for microstructure signals (bridging Cont/Stoikov (2010) with
  frequency-domain signal extraction)
- Regime-dependent market making: optimal quoting strategies that adapt refresh rate
  and spread to OU parameters and book entropy state
- Information-theoretic characterization of order book states — why does low shape
  entropy predict signal quality, and what does this imply about the information
  revelation process in limit order books?
- Extension to cross-asset information flow at spectral timescales

### Background

I hold [your degree] in [field] and have professional experience in embedded systems
engineering and quantitative trading systems. I built the full research infrastructure
(real-time Rust ingestor, Python analysis pipeline with spectral/regime tools) and can
provide the complete dataset and reproducible results.

I would welcome the opportunity to discuss how this work might fit within your group's
research agenda.

Best regards,
[Your name]

## Target Professors / Groups at ETH

- **Didier Sornette** (Chair of Entrepreneurial Risks) — complex systems, financial
  bubbles, prediction
- **Patrick Cheridito** (D-MATH, Insurance Mathematics) — stochastic processes,
  financial modeling
- **Markus Leippold** (UZH/SFI, close to ETH) — fintech, text-as-data, market
  microstructure
- **Damien Challet** (if he has ETH affiliations) — market microstructure, order
  book dynamics
- **SFI Swiss Finance Institute** faculty — several work on microstructure and
  algorithmic trading

Adapt the letter to match each professor's specific research focus. The spectral /
signal-processing angle is novel — most microstructure research stays in the time domain.

## Academic References

- Cont, Stoikov, Talreja (2010) — "A Stochastic Model for Order Book Dynamics"
- Kyle (1985) — "Continuous Auctions and Insider Trading"
- Bouchaud et al. (2004) — "Fluctuations and Response in Financial Markets"
- Easley et al. (2012) — "Flow Toxicity and Liquidity in a High-Frequency World"
- Mandelbrot & Van Ness (1968) — "Fractional Brownian Motions, Fractional Noises..."
- Avellaneda & Stoikov (2008) — "High-frequency Trading in a Limit Order Book"
- Gueant, Lehalle, Fernandez-Tapia (2012) — "Optimal Portfolio Liquidation with
  Limit Orders"
