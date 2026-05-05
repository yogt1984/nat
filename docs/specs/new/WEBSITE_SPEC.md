# NAT Research Agent Website — Full Specification

## Vision

A living research paper that updates itself. An autonomous agent reads academic papers,
extracts algorithms, implements them as trading strategies, evaluates them against real
Hyperliquid microstructure data, and publishes results — including failures — with full
mathematical rigor. The website is a real-time window into this process.

---

## System Architecture

```
                        ┌─────────────────────────┐
                        │    PAPER QUEUE           │
                        │  (curated seed list +    │
                        │   citation expansion)    │
                        └────────────┬────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────┐
│                    RESEARCH ORCHESTRATOR                      │
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌───────────────────┐  │
│  │   READER     │   │   CODER     │   │   EVALUATOR       │  │
│  │   (LLM)      │ → │   (LLM)     │ → │   (deterministic) │  │
│  │              │   │             │   │                   │  │
│  │  PDF → JSON  │   │ JSON →      │   │  walk-forward     │  │
│  │  extraction  │   │ Strategy    │   │  backtest,        │  │
│  │  of formula, │   │ subclass    │   │  Q1/Q2/Q3 gates,  │  │
│  │  intuition,  │   │ (sandboxed) │   │  cost model       │  │
│  │  parameters  │   │             │   │                   │  │
│  └─────────────┘   └─────────────┘   └─────────┬─────────┘  │
│                                                 │            │
│                                       ┌─────────▼─────────┐  │
│  ┌─────────────┐                      │   EVENT LOG       │  │
│  │  REFLECTOR   │ ◄──────────────────  │   (SQLite)        │  │
│  │  (LLM)       │                      └─────────┬─────────┘  │
│  │              │                                │            │
│  │ Analyzes     │ → next paper / variation        │            │
│  │ failures,    │   selection                     │            │
│  │ patterns     │                                │            │
│  └─────────────┘                                 │            │
└──────────────────────────────────────────────────┤────────────┘
                                                   │
                                        ┌──────────▼─────────┐
                                        │  WEBSITE RENDERER   │
                                        │  FastAPI + Static   │
                                        │  HTML/JS/KaTeX      │
                                        └─────────────────────┘
```

---

## Agent Specifications

### Agent 1: The Reader

**Role**: Extract structured, actionable information from academic papers.

**Input**: PDF file path or arXiv URL

**System prompt**:

```
You are a quantitative researcher extracting trading strategy ideas from
academic papers. You work for a system that will attempt to implement and
backtest every idea you extract.

You have access to the following market data (100ms resolution, Hyperliquid perps):
- L2 order book: bid/ask prices and quantities at 10 levels
- Trade flow: individual trades with timestamps, price, size, aggressor side
- 191 computed features across 15 categories (see FEATURES.md)
- Symbols: BTC, ETH, SOL perpetual futures
- Maker fee: 0%, Taker fee: 3.5 bps

Your job:
1. Read the paper carefully
2. Identify every testable quantitative claim
3. For each claim, extract the exact formula/algorithm
4. Assess whether it can be implemented with the available data
5. Note what the paper claims vs what is realistically testable
6. Flag edge conditions, parameter sensitivity, and known limitations
7. Distinguish between CONCURRENT relationships (explains current price)
   and PREDICTIVE relationships (forecasts future price). Only predictive
   relationships are useful for trading.

Output format: JSON with the following structure:

{
  "paper": {
    "title": "...",
    "authors": "...",
    "year": 2014,
    "venue": "Journal of Financial Economics",
    "arxiv_id": "1011.6402" (if available)
  },
  "core_thesis": "One sentence: what does the paper claim?",
  "strategies": [
    {
      "name": "short_snake_case_name",
      "description": "What the strategy does in plain English",
      "type": "signal|filter|execution",
      "formula": {
        "definition": "LaTeX string of the core formula",
        "variables": {
          "var_name": "description and how to compute from available data"
        },
        "parameters": {
          "param_name": {"paper_value": 5, "test_range": [1, 5, 10, 20]}
        }
      },
      "prediction_target": "What does this predict? (return sign, magnitude, volatility, etc.)",
      "horizon": "What timescale? Map to our data: rows (100ms) or bars (15min)",
      "paper_claims": {
        "metric": "R^2 / accuracy / Sharpe / etc.",
        "value": 0.65,
        "conditions": "contemporaneous, 5-minute aggregation, US equities"
      },
      "our_adaptation": {
        "changes_needed": "What must change for Hyperliquid crypto perps?",
        "expected_degradation": "Why will results likely be worse than the paper?",
        "potential_advantages": "Why might results be better? (e.g., less competition)"
      },
      "required_features": ["list of NAT feature names needed"],
      "missing_features": ["features the paper uses that we don't have"],
      "feasibility": "HIGH|MEDIUM|LOW",
      "edge_conditions": ["list of things that could make this fail"]
    }
  ],
  "related_papers": [
    {"title": "...", "relevance": "extends/contradicts/builds-on this work"}
  ]
}

Be skeptical. Most paper results don't survive out-of-sample testing.
Flag when a paper uses lookahead bias, survivorship bias, or unrealistic
cost assumptions. If a strategy requires data we don't have, mark it
feasibility: LOW and explain what's missing.
```

### Agent 2: The Coder

**Role**: Implement a strategy from the Reader's extraction.

**Input**: Reader's JSON extraction for one strategy

**System prompt**:

```
You are implementing a trading strategy as a Python class. You must conform
EXACTLY to the Strategy interface below. Do not add imports beyond
numpy, pandas, scipy. Do not access the filesystem, network, or any
external state.

STRATEGY INTERFACE:

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class StrategyMeta:
    name: str                    # short identifier
    paper: str                   # paper reference
    description: str             # what it does
    horizon: str                 # "1min", "5min", "15min", "1h", "4h"
    required_columns: List[str]  # NAT feature columns needed
    parameters: Dict             # tunable parameters with defaults

class Strategy:
    meta: StrategyMeta

    def compute_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute strategy-specific derived features from aggregated bars.

        Args:
            bars: DataFrame with columns from NAT feature set.
                  Index is sequential bar number.
                  Columns include: raw_midprice_mean, raw_spread_bps_mean,
                  imbalance_qty_l1_mean, flow_volume_5s_mean, etc.
                  Suffix _mean, _std, _min, _max from bar aggregation.

        Returns:
            DataFrame with computed features. Same index as input.
            Column names must be prefixed with strategy name.
            First N rows may be NaN (warmup period).
        """
        raise NotImplementedError

    def generate_signal(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate trading signal from computed features.

        Args:
            features: Output of compute_features()

        Returns:
            Series with values in [-1.0, +1.0].
            +1.0 = maximum long conviction
            -1.0 = maximum short conviction
             0.0 = no signal / flat
            NaN  = insufficient data (warmup)
        """
        raise NotImplementedError

    def warmup_bars(self) -> int:
        """Number of initial bars needed before signal is valid."""
        raise NotImplementedError
```

RULES:
1. Implement compute_features() and generate_signal() ONLY
2. All computation must use vectorized numpy/pandas operations (no Python loops over bars)
3. Handle NaN inputs gracefully (propagate NaN, don't crash)
4. Signal must be bounded to [-1, +1]
5. Include docstrings explaining the math for each computation step
6. Include the paper's formula as a comment above the implementation
7. Name intermediate variables to match the paper's notation

ALSO GENERATE:
- A test function that creates synthetic data where the strategy SHOULD
  produce a known signal (e.g., sine wave for momentum, step function for
  breakout detection). This validates the implementation against known input.

Output the complete Python file as a single code block.
```

### Agent 3: The Evaluator

**Role**: Run the strategy through the evaluation pipeline.

This is NOT an LLM. This is deterministic Python code that:
1. Loads the generated strategy class
2. Runs its unit tests (synthetic data validation)
3. If tests pass: runs walk-forward backtest on real data
4. Computes metrics: accuracy, edge, Sharpe, max drawdown, win rate
5. Applies cost model (maker at 0 bps, taker at 3.5 bps, plus half-spread)
6. Runs Q1/Q2/Q3 quality gates (if regime-dependent)
7. Applies multiple testing correction (Bonferroni on cumulative experiment count)
8. Produces verdict: GO / PIVOT / COLLECT / DROP

### Agent 4: The Reflector

**Role**: Analyze experiment results and guide future research direction.

**Input**: Full experiment history (all past results)

**System prompt**:

```
You are a senior quantitative researcher reviewing the results of an
automated strategy testing pipeline. Your job is to find patterns in
what works and what doesn't, and recommend what to test next.

You will receive:
1. The complete experiment log (strategy name, paper source, parameters,
   metrics, verdict, failure reason if applicable)
2. The current paper queue (what hasn't been tested yet)
3. Available data characteristics (N bars, symbols, date range)

Your tasks:
1. PATTERN ANALYSIS: What do successful strategies have in common?
   (horizon, feature type, regime dependency, etc.)
2. FAILURE ANALYSIS: Why did failed strategies fail? Group by failure mode:
   - Insufficient signal (low Sharpe)
   - Cost-dominated (positive gross, negative net)
   - Regime-dependent (works in some states, not others)
   - Data-insufficient (COLLECT verdict)
   - Implementation error (crashed or produced constant signal)
3. RECOMMENDATIONS: Suggest the next 3 experiments to run, ordered by
   expected value. For each:
   - What to test and why
   - Which paper to draw from (existing queue or new search)
   - What parameter variations to try
   - What you expect to learn even if it fails
4. META-OBSERVATIONS: Any structural insights about the market from
   the experiment history (e.g., "all momentum strategies work in
   State 1 but fail in State 0 — the market has distinct regimes
   for momentum vs mean-reversion")

Be concise. Lead with the recommendation. Support with evidence from
the experiment log. Do not recommend strategies similar to ones that
already failed unless you have a specific reason to believe the
variation will produce different results.

Output format:

{
  "patterns": {
    "success_factors": ["..."],
    "failure_modes": {"mode": count, ...},
    "regime_insights": "..."
  },
  "next_experiments": [
    {
      "priority": 1,
      "paper": "...",
      "strategy": "...",
      "variation": "what's different from previous attempts",
      "expected_outcome": "...",
      "rationale": "..."
    }
  ],
  "meta_observations": "...",
  "paper_search_queries": ["search terms for finding new relevant papers"]
}
```

---

## Event Schema

Every action in the system produces an event stored in SQLite:

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO 8601
    type TEXT NOT NULL,                -- event type
    payload TEXT NOT NULL,             -- JSON
    experiment_id TEXT,                -- links related events
    paper_id TEXT                      -- source paper reference
);

CREATE INDEX idx_events_type ON events(type);
CREATE INDEX idx_events_experiment ON events(experiment_id);
CREATE INDEX idx_events_timestamp ON events(timestamp);
```

Event types:

| Type | When | Payload |
|------|------|---------|
| `data_checkpoint` | Every 6 hours | rows, bars, nan_ratio, disk_size, symbols_healthy |
| `paper_queued` | Paper added to queue | paper metadata, source (seed/citation/search) |
| `paper_read` | Reader finishes extraction | full Reader JSON output |
| `strategy_generated` | Coder produces code | strategy name, code hash, test results |
| `strategy_test_failed` | Unit tests fail | strategy name, error message, traceback |
| `experiment_started` | Backtest begins | experiment_id, strategy, parameters, data range |
| `experiment_result` | Backtest completes | all metrics (Sharpe, edge, accuracy, etc.) |
| `gate_evaluation` | Quality gates run | Q1/Q2/Q3 details, verdict |
| `agent_decision` | System makes a choice | action (GO/PIVOT/DROP), reasoning |
| `strategy_deployed` | Strategy goes to paper trading | strategy name, position sizing |
| `strategy_retired` | Strategy removed from paper trading | reason, final metrics |
| `reflection` | Reflector analyzes history | patterns, recommendations |
| `profiling_complete` | Regime profiling runs | k, states, transition matrix, quality |
| `error` | Something breaks | component, error, recovery action |

---

## Website Pages

### Page 1: Live Feed (Homepage)

The real-time event stream. Each event rendered as a card with timestamp,
type icon, and formatted payload. Color-coded by type:
- Blue: data events
- Green: successful results
- Red: failures and drops
- Yellow: decisions and pivots
- Purple: reflections and insights

Auto-refreshes every 30 seconds via polling.

Header bar shows:
- Agent status: RUNNING / IDLE / ERROR
- Uptime
- Total rows collected
- Experiments: N run / N passed / N deployed

### Page 2: The Model (Static Reference)

Permanent page explaining the mathematical framework. Content sections:

**2.1 Feature Space**
- Definition: x(t) in R^d, d=123 active features
- Table of all feature categories with formulas
- Paper references for each category
- LaTeX-rendered equations using KaTeX

**2.2 Derivative Operator**
- Temporal derivatives: velocity, acceleration, z-score, slope, relative volatility
- Spectral derivatives: FFT decomposition, low/high power, spectral ratio, dominant period
- Cross-feature derivatives: ratios, correlations, divergences
- Full formulas for each, showing how raw features transform to dynamics

**2.3 Dimensionality Reduction**
- Ledoit-Wolf shrinkage: formula for regularized covariance
- PCA eigendecomposition: how components are selected
- Variance explained threshold: why 95%

**2.4 Clustering Objective**
- GMM log-likelihood: the exact optimization
- BIC model selection: formula and interpretation
- Block bootstrap stability: procedure and ARI metric

**2.5 Hierarchical Structure**
- Macro → break detection → micro: the full cascade
- PELT algorithm: penalty function, PCA pre-reduction
- How labels compose across levels

**2.6 Validation Framework**
- Q1 (structural): null hypothesis, test statistic, rejection criterion
- Q2 (predictive): Kruskal-Wallis H-test, eta-squared effect size
- Q3 (operational): self-transition rate, duration threshold
- Decision tree: Q1 fail→DROP, Q2 fail→COLLECT, Q3 fail→PIVOT, all→GO

**2.7 Online Classification**
- How new bars are classified in real-time
- Drift detection: rolling log-likelihood vs training baseline
- When to retrain: trigger conditions

### Page 3: The Data

Updated continuously from data_checkpoint events.

**3.1 Collection Status**
- Total rows, bars, date range
- Per-symbol health indicators
- Collection rate trend (rows/hour over time)
- Disk usage

**3.2 Feature Statistics**
- Correlation matrix heatmap (123x123)
- PCA scree plot (explained variance per component)
- Stationarity: ADF test p-values per feature
- Distribution plots: empirical CDF + best-fit parametric

**3.3 Data Quality**
- NaN ratio per feature category
- Continuity gaps (timeline with gap markers)
- Feature range violations (values outside expected bounds)

### Page 4: The Regimes

Updated after each profiling_complete event.

**4.1 Structure Test**
- Hopkins statistic with null distribution visualization
- Dip test results
- Conclusion: "Structure exists / does not exist"

**4.2 Optimal k**
- BIC curve for k=1..10
- Bootstrap k stability (histogram of modal k across resamples)
- Selected k with justification

**4.3 State Definitions**

For each state i = 0..k-1:
- Centroid in PCA space (3D scatter of first 3 components, states colored)
- Centroid projected to top-5 interpretable features
- Feature radar chart (normalized feature values for this state)
- Covariance ellipse in 2D PCA projection
- Plain-language interpretation: "State 0: high entropy, low momentum,
  high volatility → choppy, directionless, volatile"
- Bar count, proportion of total time
- Mean duration and duration distribution

**4.4 Transition Matrix**

- k×k heatmap with probability values
- Directed graph visualization (nodes = states, edges = transitions,
  width proportional to probability)
- Self-transition rates highlighted on diagonal
- Expected duration per state: E[d_i] = 1 / (1 - P(i→i))
- Stationary distribution: pi = eigenvector of P^T with eigenvalue 1
- Mean return per state at each horizon (table)

**4.5 Micro-States**

For each macro regime, show the within-regime decomposition:
- Micro-state centroids and interpretations
- Micro transition matrix
- Micro→macro transition analysis: "When State 0 enters micro-0a,
  it exits to State 1 within 4.2 bars on average"

### Page 5: The Library

One entry per processed paper. Each entry shows:

- Paper citation and link
- Core thesis (one sentence)
- Strategies extracted (count)
- For each strategy tested:
  - Formula (LaTeX rendered)
  - Our adaptation (what changed for Hyperliquid)
  - Result: metrics table (Sharpe, edge, accuracy, cost impact)
  - Verdict with reasoning
  - Comparison: paper's claimed performance vs our result
- Related papers (citation graph edges)

Filterable by: verdict (GO/PIVOT/DROP), feature category, horizon, date

### Page 6: Experiment History

Table of all experiments with sortable columns:
- ID, name, paper source, horizon, Sharpe, edge, verdict, date

Click-through to experiment detail page showing:
- Full walk-forward results (per-split table)
- OOS equity curve
- Feature importance (if model-based)
- Cost sensitivity curve (net P&L vs assumed cost, 0-10 bps)
- Quality gate breakdown
- Statistical significance (bootstrap CI on Sharpe)
- Agent's reasoning for the verdict
- If PIVOT: what variation was tried next and its result

Aggregate statistics at top:
- Total experiments, pass rate, best Sharpe
- Success rate by algorithm family
- Success rate by horizon
- Success rate by regime condition

### Page 7: The Spectral View

- Power spectral density spectrograms per key feature (time × frequency × power)
- Dominant period detection results
- Spectral ratio as regime indicator (overlay on regime labels)
- Frequency-domain early warning analysis

### Page 8: Cross-Symbol Consistency

- Pairwise ARI matrix (BTC/ETH/SOL)
- Temporal lag cross-correlation of regime labels
- Consensus timeline (stacked regime labels, agreement highlighted)
- Agreement statistics and interpretation

### Page 9: Decision Theory

- Multi-armed bandit: posterior distributions per algorithm family
- Explore/exploit balance over time
- Multiple testing correction: adjusted significance levels
- Capital allocation framework (Kelly criterion derivation)
- Stopping rules and power analysis

### Page 10: Paper Trading (when strategies deploy)

Per-deployed-strategy:
- Live equity curve
- Trade log (entry/exit, P&L, regime at entry)
- Backtest vs live comparison (Sharpe ratio)
- Drawdown chart
- Strategy health: is it degrading?

---

## API Endpoints

FastAPI server running on su-35, exposed via Cloudflare Tunnel.

```
GET  /api/status              → agent state, uptime, data summary
GET  /api/events              → event stream (params: since, type, limit)
GET  /api/experiments          → experiment list with metrics
GET  /api/experiments/{id}     → single experiment detail
GET  /api/papers               → processed papers with extractions
GET  /api/papers/{id}          → single paper detail + linked experiments
GET  /api/profiling/latest     → latest profiling result (states, transitions)
GET  /api/profiling/history    → all profiling runs over time
GET  /api/strategies/active    → currently deployed paper trading strategies
GET  /api/strategies/{id}/pnl  → equity curve for a deployed strategy
GET  /api/data/health          → latest data quality metrics
GET  /api/reflection/latest    → most recent Reflector analysis
```

All endpoints return JSON. No authentication (read-only, public).
CORS headers set to allow any origin.

---

## Frontend Technology

- Single HTML file per page, no build step, no framework
- KaTeX for LaTeX math rendering (loaded from CDN)
- Vanilla JS for API polling (fetch every 30 seconds)
- CSS: dark theme, monospace body, serif for math blocks
- Max content width: 720px (paper-like reading experience)
- Minimal interactivity: hover tooltips on matrices, click-through on tables
- No charts library for v1 (tables and text); Chart.js or uPlot for v2

---

## Paper Seed List

Initial 20 papers for the Reader to process:

**Microstructure & Order Flow**
1. Cont, Kukanov, Stoikov (2014) — Order flow imbalance as predictor
2. Cont, Stoikov, Talreja (2010) — Stochastic model for order book dynamics
3. Cartea, Jaimungal, Penalva (2015) — Algorithmic and HFT (Chapter 4: optimal execution)
4. Bouchaud, Farmer, Lillo (2009) — How markets slowly digest changes in supply and demand

**Information & Toxicity**
5. Easley, Lopez de Prado, O'Hara (2012) — VPIN and the flash crash
6. Kyle (1985) — Continuous auctions and insider trading
7. Glosten, Milgrom (1985) — Bid/ask/transaction prices with asymmetric information

**Regime Detection**
8. Hamilton (1989) — Regime switching models
9. Bulla, Bulla (2006) — Stylized facts of financial time series and HMMs
10. Ang, Timmermann (2012) — Regime changes and financial markets

**Entropy & Information Theory**
11. Bandt, Pompe (2002) — Permutation entropy
12. Zunino, Zanin, Tabak (2009) — Forbidden patterns and financial time series
13. Risso (2008) — Informational efficiency and the financial crisis

**Volatility & Risk**
14. Parkinson (1980) — Extreme value method for estimating variance
15. Garman, Klass (1980) — On the estimation of security price volatilities
16. Amihud (2002) — Illiquidity and stock returns

**ML & Modern Approaches**
17. Kolm, Turiel, Westray (2023) — Deep order flow imbalance
18. Zhang, Zohren, Roberts (2019) — DeepLOB: deep learning for limit order books
19. Sirignano (2019) — Deep learning for limit order books

**Crypto-Specific**
20. Makarov, Schoar (2020) — Trading and arbitrage in cryptocurrency markets

---

## Implementation Priority

```
Phase 0 (Day 1-2): Foundation
  - SQLite event store schema
  - Event logger class (Python, writes events)
  - Strategy interface definition
  - FastAPI server (3 core endpoints: status, events, experiments)
  - Single-page frontend (feed + status)
  - Cloudflare tunnel

Phase 1 (Day 3-5): Agent Pipeline
  - Reader agent (prompt + arXiv/PDF ingestion)
  - Coder agent (prompt + sandboxed execution)
  - Evaluator integration (connect to existing backtest)
  - Process first 3 papers from seed list

Phase 2 (Week 2): Mathematical Pages
  - The Model page (static, LaTeX content from existing formulation doc)
  - The Regimes page (dynamic, from profiling results)
  - Transition matrix visualization
  - State definition cards

Phase 3 (Week 3): Full Pipeline
  - Reflector agent
  - Experiment history page with detail views
  - Library page (papers + linked experiments)
  - Process remaining seed papers

Phase 4 (Week 4): Paper Trading
  - Strategy deployment to paper trading
  - Live equity curve page
  - Performance monitoring
  - Strategy retirement logic

Phase 5 (Ongoing): Autonomous Operation
  - Citation graph paper discovery
  - arXiv keyword search for new papers
  - Automated re-evaluation as new data arrives
  - Multiple testing correction dashboard
```

---

## File Structure

```
scripts/
  website/
    server.py              # FastAPI application
    events.py              # Event logger + SQLite store
    strategy.py            # Strategy base class + interface
    orchestrator.py        # Research loop coordinator
    agents/
      reader.py            # Paper reading agent
      coder.py             # Strategy code generation agent
      reflector.py         # Pattern analysis agent
    evaluator.py           # Connects generated strategies to backtest engine
    sandbox.py             # Safe execution environment for generated code
  strategies/
    generated/             # Auto-generated strategy files
    tests/                 # Auto-generated strategy tests

frontend/
    index.html             # Live feed (homepage)
    model.html             # The Model (mathematical reference)
    data.html              # Data health
    regimes.html           # Regime discovery results
    library.html           # Processed papers
    experiments.html       # Experiment history
    experiment.html        # Single experiment detail
    spectral.html          # Spectral analysis
    cross_symbol.html      # Cross-symbol consistency
    decisions.html         # Decision theory
    trading.html           # Paper trading performance
    css/
      style.css            # Dark theme, monospace, paper-like layout
    js/
      api.js               # Polling + data fetching
      render.js            # Event card rendering
      math.js              # KaTeX integration helpers
      matrix.js            # Transition matrix heatmap renderer
      charts.js            # Minimal chart rendering (v2)

papers/
    seed/                  # PDF files of seed papers
    extractions/           # Reader JSON outputs
    queue.json             # Paper processing queue

data/
    events.db              # SQLite event store
    experiments/           # Per-experiment result artifacts
    strategies/            # Deployed strategy state
```
