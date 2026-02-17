# Relevant Publications and Ideas for Entropy-Based Regime Detection Agent Swarm

## 1. Market Microstructure & Order Book Dynamics

### Foundational Theory
- **Glosten & Milgrom (1985)** - "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders"
  - Establishes information asymmetry in order flow; foundational for understanding when adverse selection dominates

- **Kyle (1985)** - "Continuous Auctions and Insider Trading"
  - Lambda (Kyle's lambda) as price impact measure; relates to entropy through information revelation rate

- **Avellaneda & Stoikov (2008)** - "High-frequency trading in a limit order book"
  - ASMM foundation paper; optimal spread as function of inventory risk and volatility
  - Key parameters: $\gamma$ (risk aversion), $\sigma$ (volatility), $k$ (order arrival intensity)

### Order Book Features & Information Content
- **Cont, Stoikov & Talreja (2010)** - "A Stochastic Model for Order Book Dynamics"
  - Queue dynamics, order flow imbalance as predictive feature

- **Cartea, Jaimungal & Penalva (2015)** - "Algorithmic and High-Frequency Trading" (Book)
  - Comprehensive treatment of microstructure features

- **Lehalle & Laruelle (2018)** - "Market Microstructure in Practice" (Book, 2nd ed)
  - Practical feature engineering for order books

## 2. Entropy & Information Theory in Finance

### Market Entropy Measures
- **Zunino et al. (2009)** - "Forbidden patterns, permutation entropy and stock market inefficiency"
  - Permutation entropy for detecting market efficiency regimes

- **Risso (2008)** - "The informational efficiency and the financial crashes"
  - Shannon entropy as market efficiency proxy

- **Pincus (1991)** - "Approximate entropy as a measure of system complexity"
  - ApEn for detecting regime changes; applicable to price/volume series

- **Richman & Moorman (2000)** - "Physiological time-series analysis using approximate entropy and sample entropy"
  - Sample entropy (SampEn) - improved ApEn without self-matching bias

### Entropy and Predictability
- **Bandt & Pompe (2002)** - "Permutation Entropy: A Natural Complexity Measure for Time Series"
  - Ordinal patterns; computationally efficient; robust to noise

- **Costa, Goldberger & Peng (2005)** - "Multiscale entropy analysis of biological signals"
  - Multi-scale entropy; captures complexity across time horizons

- **Shternshis, Mazzarisi & Marmi (2022)** - "Measuring market efficiency: The Shannon entropy of high-frequency financial time series"
  - Direct application to HFT data

## 3. Regime Detection & Switching Models

### Hidden Markov Models
- **Hamilton (1989)** - "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle"
  - Foundational regime-switching paper

- **Guidolin & Timmermann (2007)** - "Asset allocation under multivariate regime switching"
  - Multi-asset regime models

### Machine Learning Approaches
- **Nystrup et al. (2017)** - "Regime-Based Versus Static Asset Allocation"
  - HMM vs static; practical regime detection

- **Mulvey & Liu (2016)** - "Identifying Economic Regimes: Reducing Downside Risks for University Endowments and Foundations"
  - Ensemble methods for regime detection

- **Gu, Kelly & Xiu (2020)** - "Empirical Asset Pricing via Machine Learning"
  - Comprehensive ML for financial prediction; feature importance

## 4. Mean Reversion vs Trend Following

### Mean Reversion Theory
- **Poterba & Summers (1988)** - "Mean reversion in stock prices: Evidence and Implications"
  - Statistical evidence for mean reversion at various horizons

- **Lo & MacKinlay (1988)** - "Stock Market Prices Do Not Follow Random Walks"
  - Variance ratio tests; autocorrelation structure

### Trend Following
- **Moskowitz, Ooi & Pedersen (2012)** - "Time Series Momentum"
  - Cross-asset momentum; 12-month lookback

- **Hurst, Ooi & Pedersen (2017)** - "A Century of Evidence on Trend-Following Investing"
  - Long-term evidence; crisis alpha

### Conditional Strategy Selection
- **Baz et al. (2015)** - "Dissecting Investment Strategies in the Cross Section and Time Series"
  - Decomposition of carry, momentum, value

## 5. Genetic Algorithms & Evolutionary Computation in Trading

### Foundational
- **Holland (1975)** - "Adaptation in Natural and Artificial Systems"
  - Genetic algorithm foundations

- **Koza (1992)** - "Genetic Programming"
  - GP for strategy evolution

### Trading Applications
- **Neely, Weller & Dittmar (1997)** - "Is Technical Analysis in the Foreign Exchange Market Profitable?"
  - GP for trading rule discovery

- **Chen & Yeh (2001)** - "Evolving Traders and the Business School with Genetic Programming"
  - Agent-based markets with GP traders

- **Cont (2001)** - "Empirical properties of asset returns: stylized facts and statistical issues"
  - Stylized facts any evolved strategy must respect

## 6. Multi-Agent Systems & Swarm Intelligence

### Agent-Based Modeling in Finance
- **LeBaron (2006)** - "Agent-based Computational Finance"
  - Survey of ABM in finance

- **Farmer & Foley (2009)** - "The economy needs agent-based modelling"
  - Nature perspective on ABM necessity

### Swarm Intelligence
- **Bonabeau, Dorigo & Theraulaz (1999)** - "Swarm Intelligence: From Natural to Artificial Systems"
  - Ant colony, particle swarm foundations

- **Kennedy & Eberhart (1995)** - "Particle Swarm Optimization"
  - PSO for parameter optimization

## 7. Feature Engineering for Order Books

### Specific Features
- **Cao, Chen & Hansch (2009)** - "Order imbalance and order book information"
  - Order imbalance metrics; predictive power

- **Kercheval & Zhang (2015)** - "Modelling high-frequency limit order book dynamics with support vector machines"
  - Comprehensive LOB feature set; 144 features

- **Sirignano (2019)** - "Deep Learning for Limit Order Books"
  - Neural network approaches; spatial-temporal features

### Kalman Filtering in Finance
- **Wells (1996)** - "The Kalman Filter in Finance"
  - Kalman for state estimation in markets

- **Harvey (1990)** - "Forecasting, Structural Time Series Models and the Kalman Filter"
  - State-space models for time series

## 8. Spectral Methods & Fourier Analysis

### Financial Applications
- **Granger & Morgenstern (1963)** - "Spectral Analysis of New York Stock Market Prices"
  - Early spectral analysis in finance

- **Ramsey & Zhang (1997)** - "The analysis of foreign exchange data using waveform dictionaries"
  - Wavelet methods for FX

- **Mandelbrot (1997)** - "Fractals and Scaling in Finance"
  - Multi-fractal analysis; Hurst exponent

## 9. Hypervolume & Decision Boundary Learning

### Classification in High-Dimensional Spaces
- **Vapnik (1995)** - "The Nature of Statistical Learning Theory"
  - SVM; margin maximization; kernel methods for Hilbert spaces

- **Schölkopf & Smola (2002)** - "Learning with Kernels"
  - Kernel methods in Hilbert spaces; one-class SVM for novelty detection

### Active Learning & Decision Boundaries
- **Settles (2009)** - "Active Learning Literature Survey"
  - Efficient boundary exploration

- **Deb et al. (2002)** - "A fast and elitist multiobjective genetic algorithm: NSGA-II"
  - Multi-objective optimization; Pareto frontiers

## 10. Real-Time Systems & Low-Latency Architecture

### System Design
- **Arndt & Saeger (2019)** - "Building Low Latency Applications with C++"
  - Cache optimization, lock-free structures

- **Butenhof (1997)** - "Programming with POSIX Threads"
  - Concurrent programming foundations

### Rust-Specific
- **Klabnik & Nichols (2019)** - "The Rust Programming Language"
  - Official Rust book; ownership, lifetimes

- **Blandy & Orendorff (2021)** - "Programming Rust" (2nd ed)
  - Advanced Rust patterns

---

## Key Ideas to Synthesize

### A. Entropy-Regime Hypothesis
**Core Conjecture**: Market entropy $H(t)$ acts as a sufficient statistic for optimal strategy selection:
- $H(t) < H_{low}$: Information asymmetry high, trends persist → Trend Following
- $H(t) > H_{high}$: Noise dominates, prices mean-revert → Mean Reversion (ASMM)
- $H_{low} < H(t) < H_{high}$: Uncertain regime → Reduce exposure or N/A

### B. Feature Space Geometry
The feature space $\mathcal{F}$ has natural structure:
1. **Manifold Hypothesis**: Profitable states lie on lower-dimensional manifolds
2. **Clustering**: Natural clustering by regime (use UMAP/t-SNE for visualization)
3. **Decision Boundaries**: Learn separating hyperplanes/surfaces via SVM or neural nets

### C. Evolutionary Architecture
Agent genotypes encode:
1. Feature selection (which subset of $\mathcal{F}$ to observe)
2. Entropy computation method (permutation, sample, multi-scale)
3. Threshold parameters ($H_{low}$, $H_{high}$)
4. Strategy parameters ($\theta_{MR}$, $\theta_{TF}$)

Fitness function: Risk-adjusted returns (Sharpe, Sortino, or Calmar ratio)

### D. Continuous Learning Loop
```
Ingest → Features → Entropy → Regime → Strategy → Execute → Profit/Loss → Update Genotype
   ↑                                                                              |
   └──────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority (Suggested)

1. **Phase 1**: Ingestor (Rust) - deterministic, reproducible feature extraction
2. **Phase 2**: Entropy computation module - multiple entropy measures
3. **Phase 3**: Backtesting framework - historical regime labeling
4. **Phase 4**: Supervised learning - hypervolume boundaries from labeled data
5. **Phase 5**: Genetic layer - evolving agent parameters
6. **Phase 6**: Live trading with paper money - validation
7. **Phase 7**: Production deployment with risk limits
