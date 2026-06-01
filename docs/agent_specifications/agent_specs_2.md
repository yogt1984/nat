# NAT Evolutionary Strategy Agent — Technical Specification

**Date**: 2026-06-01
**Status**: PROPOSED
**Purpose**: Evolutionary search over signal combinations, regime gates, and risk
parameters to discover diverse, complementary trading strategies

---

## 1. Problem Statement

NAT has 18 registered algorithms, each producing 1-8 features. Manual exploration
(run gauntlet, rank by PnL, pick top 5) covers a tiny fraction of the combinatorial
space:

- Signal subsets: 2^18 = 262,144 combinations
- Per-signal weights: continuous, 18-dimensional
- Regime gate conditions: 5+ features x threshold x direction
- Risk parameters: position sizing, stops, holding period
- Timescale: tick to 15-minute bars

An evolutionary search explores this space systematically, with built-in defenses
against overfitting, and discovers not just the single best strategy but a diverse
portfolio of complementary specialists.

---

## 2. Genotype: Structured Strategy Genome

Fixed-length, interpretable genome where every gene has economic meaning. No trees,
no neural nets, no variable-length representations.

```
Genome (42 genes):

SIGNAL SELECTION — 18 genes, binary
  use[i]: bool                    one per registered algorithm
                                  (hawkes, optimal_entry, funding, ...)

SIGNAL WEIGHTS — 18 genes, continuous
  w[i]: float in [-1, 1]         sign encodes polarity override
                                  magnitude encodes conviction
                                  only active (use[i]=True) signals matter

REGIME GATE — 4 genes, mixed
  gate_feature:    enum           ent_book_shape | switching_ou_regime |
                                  spread_regime | vol_returns_1m |
                                  hawkes_excitement | none
  gate_threshold:  float [0, 1]  percentile threshold on gate_feature
  gate_direction:  enum           above | below
  gate_mode:       enum           hard (binary) | soft (linear scaling)

ENTRY / EXIT — 4 genes, continuous + integer
  entry_z:           float [1.0, 4.0]    z-score of composite signal to enter
  exit_z:            float [0.1, 1.5]    z-score to exit (or signal flip)
  max_holding_ticks: int [10, 6000]      forced exit: 1s to 10 min
  cooldown_ticks:    int [0, 600]        minimum gap between trades

RISK MANAGEMENT — 4 genes, continuous
  position_frac:        float [0.01, 0.50]   fraction of capital per trade
  stop_loss_atr:        float [0.5, 5.0]     stop loss in ATR multiples
  max_drawdown_pct:     float [1.0, 20.0]    portfolio kill switch
  correlation_penalty:  float [0.0, 1.0]     reduce size when correlated
                                             with existing portfolio
```

### 2.1 Why This Genotype

| Property | Benefit |
|----------|---------|
| Fixed length (42) | No bloat, bounded search space |
| Every gene interpretable | Can explain why a strategy works |
| Mixed discrete + continuous | Matches the problem structure |
| Signal selection bits | Natural modularity — evolution discovers good combos |
| Bounded ranges | Prevents degenerate solutions |
| Small parameter count | Overfitting risk manageable (42 params, not 50K) |

### 2.2 Genome Encoding

```python
@dataclass
class StrategyGenome:
    # Signal selection and weights
    signal_mask: np.ndarray       # bool[18]
    signal_weights: np.ndarray    # float[18], in [-1, 1]

    # Regime gate
    gate_feature: str             # feature name or "none"
    gate_threshold: float         # percentile [0, 1]
    gate_direction: str           # "above" or "below"
    gate_mode: str                # "hard" or "soft"

    # Entry / exit
    entry_z: float
    exit_z: float
    max_holding_ticks: int
    cooldown_ticks: int

    # Risk
    position_frac: float
    stop_loss_atr: float
    max_drawdown_pct: float
    correlation_penalty: float

    def active_signals(self) -> list[str]:
        """Algorithm names where signal_mask is True."""
        return [ALGO_NAMES[i] for i in range(18) if self.signal_mask[i]]

    def n_active(self) -> int:
        return int(self.signal_mask.sum())

    def composite_signal(self, tick_features: dict) -> float:
        """Weighted sum of active algorithm outputs."""
        total = 0.0
        for i, name in enumerate(ALGO_NAMES):
            if not self.signal_mask[i]:
                continue
            val = tick_features.get(ALGO_PRIMARY[name], 0.0)
            total += self.signal_weights[i] * val
        return total
```

### 2.3 Mutation Operators

```python
def mutate(genome: StrategyGenome, p_flip=0.1, sigma=0.1) -> StrategyGenome:
    """Produce a child genome via mutation."""
    child = copy.deepcopy(genome)

    # Flip signal bits with probability p_flip
    for i in range(18):
        if random.random() < p_flip:
            child.signal_mask[i] = not child.signal_mask[i]

    # Ensure at least 1 signal active
    if child.n_active() == 0:
        child.signal_mask[random.randint(0, 17)] = True

    # Perturb continuous genes with Gaussian noise
    child.signal_weights += np.random.normal(0, sigma, 18)
    child.signal_weights = np.clip(child.signal_weights, -1, 1)

    child.entry_z = clip(child.entry_z + gauss(0, 0.2), 1.0, 4.0)
    child.exit_z = clip(child.exit_z + gauss(0, 0.1), 0.1, 1.5)
    child.max_holding_ticks = clip(int(child.max_holding_ticks * lognormal(0, 0.2)), 10, 6000)
    child.cooldown_ticks = clip(int(child.cooldown_ticks + gauss(0, 30)), 0, 600)

    child.position_frac = clip(child.position_frac + gauss(0, 0.03), 0.01, 0.50)
    child.stop_loss_atr = clip(child.stop_loss_atr + gauss(0, 0.3), 0.5, 5.0)

    # Occasionally mutate discrete genes
    if random.random() < 0.05:
        child.gate_feature = random.choice(GATE_FEATURES)
    if random.random() < 0.05:
        child.gate_direction = random.choice(["above", "below"])
    if random.random() < 0.05:
        child.gate_mode = random.choice(["hard", "soft"])

    child.gate_threshold = clip(child.gate_threshold + gauss(0, 0.1), 0, 1)

    return child
```

### 2.4 Crossover Operator

```python
def crossover(a: StrategyGenome, b: StrategyGenome) -> StrategyGenome:
    """Uniform crossover: each gene picked from parent A or B."""
    child = copy.deepcopy(a)

    for i in range(18):
        if random.random() < 0.5:
            child.signal_mask[i] = b.signal_mask[i]
            child.signal_weights[i] = b.signal_weights[i]

    # Swap blocks of continuous genes
    if random.random() < 0.5:
        child.entry_z, child.exit_z = b.entry_z, b.exit_z
        child.max_holding_ticks = b.max_holding_ticks
        child.cooldown_ticks = b.cooldown_ticks

    if random.random() < 0.5:
        child.gate_feature = b.gate_feature
        child.gate_threshold = b.gate_threshold
        child.gate_direction = b.gate_direction
        child.gate_mode = b.gate_mode

    if random.random() < 0.5:
        child.position_frac = b.position_frac
        child.stop_loss_atr = b.stop_loss_atr

    return child
```

---

## 3. Fitness Function

### 3.1 Anti-Overfit Design

```
fitness(genome) =
    deflated_sharpe_oos(genome)
    - lambda_c * n_active_signals(genome)
    - lambda_t * max(0, MIN_TRADES - n_trades)

where:
    deflated_sharpe_oos = walk-forward OOS Sharpe ratio
                          adjusted for number of strategies tested
                          (Harvey & Liu 2015 correction)

    n_active_signals    = count(signal_mask == True)
    lambda_c            = 0.05  (complexity penalty per signal)
    n_trades            = total trades in OOS evaluation period
    MIN_TRADES          = 30
    lambda_t            = 0.5   (penalty per missing trade below minimum)
```

### 3.2 Walk-Forward Evaluation

Each fitness evaluation runs the genome through the existing gauntlet infrastructure:

```python
def evaluate_fitness(genome: StrategyGenome, dates: list[str]) -> float:
    """Walk-forward OOS fitness for a strategy genome.

    Uses purged walk-forward: train on dates[:-1], test on dates[-1].
    Repeats for each date as the test fold.
    """
    oos_sharpes = []

    for fold_idx in range(len(dates)):
        test_date = dates[fold_idx]
        train_dates = [d for i, d in enumerate(dates) if i != fold_idx]

        # Calibrate z-score thresholds on training data
        z_params = calibrate_zscore(genome, train_dates)

        # Evaluate on test date (no in-sample leakage)
        result = run_paper_trader(
            genome=genome,
            z_params=z_params,
            date=test_date,
            cost_model="hyperliquid_taker",
        )

        if result.n_trades >= MIN_TRADES:
            oos_sharpes.append(result.sharpe)

    if len(oos_sharpes) < 2:
        return -np.inf  # insufficient data

    # Deflated Sharpe (Harvey & Liu 2015)
    raw_sharpe = np.mean(oos_sharpes)
    n_tested = get_total_strategies_tested()  # global counter
    deflated = deflated_sharpe_ratio(raw_sharpe, n_tested, len(oos_sharpes))

    # Complexity penalty
    penalty = LAMBDA_C * genome.n_active()

    return deflated - penalty
```

### 3.3 Deflated Sharpe Ratio

```python
def deflated_sharpe_ratio(
    sharpe: float,
    n_strategies: int,
    n_folds: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Harvey & Liu (2015) correction for multiple testing.

    Adjusts the Sharpe ratio downward based on how many strategies
    were tested to find this one.
    """
    from scipy.stats import norm

    # Expected maximum Sharpe under null (all strategies are noise)
    e_max = norm.ppf(1 - 1 / n_strategies) if n_strategies > 1 else 0

    # Variance of Sharpe estimator (Lo 2002)
    var_sharpe = (1 + 0.5 * sharpe**2 - skew * sharpe
                  + (kurtosis - 3) / 4 * sharpe**2) / n_folds

    # Test statistic: is observed Sharpe significantly above expected max?
    if var_sharpe <= 0:
        return 0.0
    psi = (sharpe - e_max) / np.sqrt(var_sharpe)
    p_value = 1 - norm.cdf(psi)

    # Return adjusted Sharpe (0 if not significant at 5%)
    if p_value > 0.05:
        return 0.0
    return sharpe - e_max
```

---

## 4. Evolutionary Algorithm: MAP-Elites + CMA-ES

### 4.1 MAP-Elites: Diversity Preservation

MAP-Elites maintains a grid of behaviorally diverse strategies. Instead of
converging to a single optimum, it fills a map of niches — each the best
strategy of its behavioral type.

```
Behavioral Descriptors (grid axes):

  Axis 1: TIMESCALE
    mean holding period in seconds
    Bins: [0-2s, 2-10s, 10-60s, 60-300s, 300s+]
    5 bins

  Axis 2: SIGNAL FAMILY
    dominant signal category (by weight mass):
    Bins: [microstructure, entropy, macro/funding, mixed]
    4 bins

  Axis 3: TRADE FREQUENCY
    trades per hour
    Bins: [0-5, 5-20, 20-60, 60+]
    4 bins

Total grid: 5 x 4 x 4 = 80 cells
Each cell holds one genome (the best of its behavioral type)
```

### 4.2 MAP-Elites Loop

```python
class MAPElites:
    def __init__(self, dates: list[str]):
        self.grid: dict[tuple, StrategyGenome] = {}
        self.fitness_cache: dict[tuple, float] = {}
        self.dates = dates
        self.n_tested = 0

    def cell_key(self, genome: StrategyGenome, result: TradeResult) -> tuple:
        """Map a genome to its behavioral cell."""
        # Axis 1: timescale
        ht = result.mean_holding_seconds
        ts = 0 if ht < 2 else 1 if ht < 10 else 2 if ht < 60 else 3 if ht < 300 else 4

        # Axis 2: signal family
        micro = {"hawkes_intensity", "spread_decomp", "optimal_entry",
                 "kalman_imbalance", "switching_ou", "weighted_ofi",
                 "trade_through", "propagator", "vpin_regime"}
        entropy = {"surprise_signal", "entropy_momentum"}
        macro = {"funding_reversion", "oi_divergence"}

        active = set(genome.active_signals())
        w_micro = sum(abs(genome.signal_weights[i]) for i, n in enumerate(ALGO_NAMES)
                      if n in active & micro)
        w_entropy = sum(abs(genome.signal_weights[i]) for i, n in enumerate(ALGO_NAMES)
                        if n in active & entropy)
        w_macro = sum(abs(genome.signal_weights[i]) for i, n in enumerate(ALGO_NAMES)
                      if n in active & macro)
        total_w = w_micro + w_entropy + w_macro + 1e-12
        if w_micro / total_w > 0.6:
            sf = 0
        elif w_entropy / total_w > 0.4:
            sf = 1
        elif w_macro / total_w > 0.4:
            sf = 2
        else:
            sf = 3

        # Axis 3: trade frequency
        tph = result.trades_per_hour
        tf = 0 if tph < 5 else 1 if tph < 20 else 2 if tph < 60 else 3

        return (ts, sf, tf)

    def run(self, n_init=200, n_generations=100, children_per_gen=50):
        """Full MAP-Elites run."""

        # Phase 1: Random initialization
        for _ in range(n_init):
            genome = random_genome()
            self._evaluate_and_place(genome)

        # Phase 2: Evolutionary improvement
        for gen in range(n_generations):
            children = []
            occupied = list(self.grid.values())

            for _ in range(children_per_gen):
                # Select parent from occupied cells (uniform random)
                parent = random.choice(occupied)

                # 70% mutation, 30% crossover
                if random.random() < 0.7:
                    child = mutate(parent)
                else:
                    parent2 = random.choice(occupied)
                    child = crossover(parent, parent2)

                children.append(child)

            # Evaluate all children (parallelizable)
            for child in children:
                self._evaluate_and_place(child)

            # Report progress
            n_filled = len(self.grid)
            best_fit = max(self.fitness_cache.values()) if self.fitness_cache else -np.inf
            mean_fit = np.mean(list(self.fitness_cache.values())) if self.fitness_cache else 0
            print(f"Gen {gen:3d} | cells: {n_filled}/80 | "
                  f"best: {best_fit:.3f} | mean: {mean_fit:.3f} | "
                  f"tested: {self.n_tested}")

    def _evaluate_and_place(self, genome: StrategyGenome):
        """Evaluate genome and place in grid if it improves its cell."""
        self.n_tested += 1
        fitness_val = evaluate_fitness(genome, self.dates)

        if fitness_val == -np.inf:
            return  # insufficient trades

        # Get behavioral result for cell assignment
        result = get_last_trade_result()  # from evaluate_fitness
        key = self.cell_key(genome, result)

        if key not in self.grid or fitness_val > self.fitness_cache[key]:
            self.grid[key] = genome
            self.fitness_cache[key] = fitness_val
```

### 4.3 CMA-ES Inner Optimization

For the top cells in the MAP-Elites grid, refine continuous parameters:

```python
import cma

def cma_refine(genome: StrategyGenome, dates: list[str], max_evals=100):
    """CMA-ES optimization of continuous genes with discrete genes fixed."""

    def pack(g: StrategyGenome) -> np.ndarray:
        """Pack continuous genes into a vector."""
        return np.array([
            *g.signal_weights[g.signal_mask],  # only active weights
            g.entry_z, g.exit_z,
            np.log(g.max_holding_ticks),  # log-scale for integers
            np.log(g.cooldown_ticks + 1),
            g.gate_threshold,
            g.position_frac,
            g.stop_loss_atr,
        ])

    def unpack(vec: np.ndarray, template: StrategyGenome) -> StrategyGenome:
        """Unpack vector back into genome (discrete genes unchanged)."""
        g = copy.deepcopy(template)
        n_active = g.n_active()
        g.signal_weights[g.signal_mask] = np.clip(vec[:n_active], -1, 1)
        idx = n_active
        g.entry_z = clip(vec[idx], 1.0, 4.0); idx += 1
        g.exit_z = clip(vec[idx], 0.1, 1.5); idx += 1
        g.max_holding_ticks = clip(int(np.exp(vec[idx])), 10, 6000); idx += 1
        g.cooldown_ticks = clip(int(np.exp(vec[idx]) - 1), 0, 600); idx += 1
        g.gate_threshold = clip(vec[idx], 0, 1); idx += 1
        g.position_frac = clip(vec[idx], 0.01, 0.50); idx += 1
        g.stop_loss_atr = clip(vec[idx], 0.5, 5.0)
        return g

    x0 = pack(genome)
    sigma0 = 0.3

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        "maxfevals": max_evals,
        "bounds": [[-1]*len(x0), [1]*len(x0)],  # approximate
    })

    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        for sol in solutions:
            candidate = unpack(sol, genome)
            f = evaluate_fitness(candidate, dates)
            fitnesses.append(-f)  # CMA-ES minimizes
        es.tell(solutions, fitnesses)

    best_vec = es.result.xbest
    return unpack(best_vec, genome)
```

---

## 5. Multi-Agent Cooperative Co-Evolution

### 5.1 Portfolio Fitness (Marginal Contribution)

Individual fitness measures standalone quality. Portfolio fitness measures
what an agent ADDS to the ensemble:

```python
def portfolio_fitness(
    candidate: StrategyGenome,
    existing_portfolio: list[StrategyGenome],
    dates: list[str],
) -> float:
    """Marginal Shapley contribution to portfolio Sharpe.

    fitness = sharpe(portfolio + candidate) - sharpe(portfolio)

    Penalizes candidates correlated with existing members.
    """
    # Evaluate portfolio without candidate
    base_returns = portfolio_returns(existing_portfolio, dates)
    base_sharpe = sharpe_ratio(base_returns)

    # Evaluate portfolio with candidate
    candidate_returns = strategy_returns(candidate, dates)
    combined_returns = base_returns + candidate_returns  # additive
    combined_sharpe = sharpe_ratio(combined_returns)

    marginal = combined_sharpe - base_sharpe

    # Correlation penalty
    if len(existing_portfolio) > 0:
        corrs = [np.corrcoef(candidate_returns, strategy_returns(s, dates))[0, 1]
                 for s in existing_portfolio]
        max_corr = max(abs(c) for c in corrs)
        marginal *= (1 - candidate.correlation_penalty * max_corr)

    return marginal
```

### 5.2 Agent Communication Protocol

Agents share state for coordination and collective risk management:

```python
@dataclass
class SharedState:
    """Broadcast from portfolio manager to all agents each tick."""
    # Regime context (from switching_ou, spread_decomp)
    regime_fast_prob: float        # P(fast mean-reversion)
    adverse_selection: float      # spread_decomp adverse trend
    hawkes_excitement: float      # current excitement level

    # Portfolio state
    aggregate_position: float     # net position across all agents
    portfolio_drawdown: float     # current drawdown from peak (%)
    n_active_agents: int          # how many agents hold positions

    # Risk limits
    position_budget: float        # remaining position capacity
    drawdown_headroom: float      # distance to kill switch (%)


@dataclass
class AgentOutput:
    """Each agent emits per tick."""
    signal: float                 # directional conviction [-1, 1]
    confidence: float             # desired capital fraction [0, 1]
    veto: bool                    # emergency: flatten all positions
    reason: str                   # human-readable rationale
```

### 5.3 Veto Mechanism

Any agent can trigger portfolio-wide risk-off:

```python
def portfolio_step(agents: list[Agent], tick: dict, shared: SharedState):
    outputs = [agent.step(tick, shared) for agent in agents]

    # Veto check — any agent can kill all positions
    if any(out.veto for out in outputs):
        flatten_all_positions()
        vetoing = [a.name for a, o in zip(agents, outputs) if o.veto]
        log(f"VETO by {vetoing} — all positions flattened")
        return

    # Aggregate signals weighted by confidence and portfolio fitness
    total_signal = sum(
        out.signal * out.confidence * agent.portfolio_weight
        for agent, out in zip(agents, outputs)
    )

    # Apply position limits from shared state
    target_position = clip(
        total_signal * shared.position_budget,
        -shared.position_budget,
        +shared.position_budget,
    )

    execute_target(target_position)
```

---

## 6. Overfitting Defenses (5 Layers)

| Layer | Mechanism | Implementation |
|-------|-----------|----------------|
| 1. Walk-forward | Fitness computed ONLY on OOS folds | Purged CV with embargo bars |
| 2. Deflated Sharpe | Corrects for number of strategies tested | Harvey & Liu (2015), global counter |
| 3. Complexity penalty | Fewer active signals = better | lambda_c * n_active in fitness |
| 4. Minimum trade count | Strategies with < 30 trades = -inf | Hard floor in fitness function |
| 5. FDR on final population | BH correction on MAP-Elites grid | Existing agent FDR infrastructure |

### 6.1 Generation Budget

Hard stop at 100 generations. With 50 children per generation and 200 initial
candidates, total strategies tested = 200 + 100*50 = 5,200. The deflated Sharpe
correction accounts for this search depth.

### 6.2 Temporal Holdout

Reserve the most recent 20% of dates as a final holdout, never seen during
evolution. After the MAP-Elites run completes, evaluate the entire grid on
the holdout. Strategies that degrade > 50% are flagged.

---

## 7. Computational Cost

| Operation | Time | Count | Total |
|-----------|------|-------|-------|
| Single fitness evaluation | ~5 min | 5,200 | ~430 hours |
| With 4x parallel workers | ~5 min | 5,200 | ~108 hours |
| CMA-ES refinement (top 20 cells) | ~8 hours each | 20 | ~160 hours |
| **Total (4 workers, sequential)** | | | **~270 hours (~11 days)** |
| **Total (8 workers, cloud burst)** | | | **~6 days** |

This is a batch job, not real-time. Run on su-35 over 1-2 weeks, or burst to
cloud VMs for faster turnaround.

### 7.1 Caching

Many genomes share signal subsets. Cache algorithm outputs per date:

```python
# Key: (algorithm_name, date, symbol)
# Value: DataFrame of algorithm outputs
ALGO_CACHE: dict[tuple, pd.DataFrame] = {}

def get_algo_output(algo: str, date: str, symbol: str) -> pd.DataFrame:
    key = (algo, date, symbol)
    if key not in ALGO_CACHE:
        ALGO_CACHE[key] = run_algorithm_batch(algo, date, symbol)
    return ALGO_CACHE[key]
```

This means fitness evaluation only recomputes the composite signal and trade
simulation — the expensive per-algorithm batch runs are cached. Reduces per-eval
time from ~5 min to ~30 seconds after warmup.

---

## 8. Implementation Stages

### Stage 1: Exhaustive Signal Combination Search (2-3 days to implement)

No evolutionary machinery needed. Brute-force enumerate:

```
All 2-signal combos: C(18,2) = 153
All 3-signal combos: C(18,3) = 816
Total: 969 combinations
```

For each combination, run equal-weighted composite through the gauntlet.
Rank by deflated Sharpe. This reveals the fitness landscape and identifies
which signal families are complementary vs redundant.

**Files:**
- `scripts/evolution/combination_sweep.py` — enumerate, evaluate, rank
- Uses existing `paper_trader_generic.py` infrastructure

**Output:** `reports/combination_sweep.json` — ranked list of all combos
with per-symbol Sharpe, trade count, correlation matrix.

### Stage 2: MAP-Elites on Promising Region (3-4 days to implement)

Take the top 50 combinations from Stage 1. Seed the MAP-Elites grid.
Evolve regime gates and risk parameters around them.

**Files:**
- `scripts/evolution/genome.py` — StrategyGenome dataclass + mutation/crossover
- `scripts/evolution/map_elites.py` — MAP-Elites loop + behavioral descriptors
- `scripts/evolution/fitness.py` — walk-forward evaluation + deflated Sharpe
- `scripts/evolution/report.py` — grid visualization + summary

**Output:** MAP-Elites grid (80 cells) with diverse strategies, each the
best of its behavioral type.

### Stage 3: CMA-ES Refinement + Portfolio Assembly (2-3 days)

For the top 20 cells in the grid, run CMA-ES to fine-tune continuous
parameters. Then assemble the final portfolio using marginal Shapley
contribution.

**Files:**
- `scripts/evolution/cma_refine.py` — CMA-ES inner loop
- `scripts/evolution/portfolio.py` — cooperative fitness + portfolio assembly
- Reuses `scripts/agent/meta_portfolio.py` for risk parity weights

**Output:** Final portfolio of 5-10 complementary strategies with
allocations and expected Sharpe.

### Stage 4: Agent Communication + Live Framework (3-4 days)

SharedState protocol, veto mechanism, live portfolio execution loop.
Integration with existing agent infrastructure.

**Files:**
- `scripts/evolution/agent_protocol.py` — SharedState, AgentOutput
- `scripts/evolution/portfolio_manager.py` — tick-level orchestration
- Integration with `scripts/agent/base.py` ResearchAgent

---

## 9. NAT CLI Integration

```bash
nat evolve sweep                    # Stage 1: brute-force combination search
nat evolve sweep --max-signals 4    # limit to 2-4 signal combos
nat evolve sweep report             # print ranked results

nat evolve run                      # Stage 2-3: MAP-Elites + CMA-ES
nat evolve run --generations 100    # generation budget
nat evolve run --workers 4          # parallel fitness evaluations
nat evolve run --dates 7            # use last 7 days of data
nat evolve stop                     # graceful stop + save state
nat evolve report                   # print MAP-Elites grid + best strategies
nat evolve report --cell 2,1,3      # detail for specific grid cell

nat evolve portfolio                # Stage 3: assemble portfolio from grid
nat evolve portfolio --top 20       # CMA-ES refine top 20 cells
nat evolve portfolio --backtest     # run portfolio on holdout data

nat evolve live                     # Stage 4: live portfolio manager
nat evolve live --paper             # paper trading mode
```

---

## 10. Connection to Existing NAT Infrastructure

| Evolutionary component | NAT module | Reuse path |
|---|---|---|
| Algorithm outputs | `scripts/algorithms/*.py` | `run_batch()` on parquet data |
| Fitness evaluation | `scripts/alpha/paper_trader_generic.py` | Walk-forward paper trading |
| Deflated Sharpe | `scripts/alpha/alpha_pipeline.py` | Existing Harvey & Liu implementation |
| FDR correction | `scripts/agent/base.py` | BH procedure in ResearchAgent |
| Portfolio assembly | `scripts/agent/meta_portfolio.py` | Risk parity + Ledoit-Wolf |
| Cost model | `config/costs.toml` | Single source of truth |
| Parameter calibration | `scripts/calibration/` (proposed) | adaptive_calibrator |
| Report generation | `scripts/agent/research_output.py` | Structured JSON emitter |

---

## 11. Expected Outcomes

### What MAP-Elites Should Discover

Based on existing gauntlet results, the grid will likely fill these niches:

| Cell (timescale, family, freq) | Expected strategy |
|---|---|
| (2-10s, micro, 20-60/h) | hawkes + optimal_entry + spread_decomp gate |
| (60-300s, macro, 5-20/h) | funding_reversion + switching_ou gate |
| (10-60s, micro, 5-20/h) | kalman_imbalance + ent_book_shape gate |
| (2-10s, entropy, 20-60/h) | surprise_signal + hawkes excitement gate |
| (60-300s, mixed, 0-5/h) | funding + optimal_entry + low frequency |

### What We Don't Know (and Evolution Will Reveal)

1. Do 4+ signal combinations outperform the best 2-3 signal combos?
2. Which regime gate produces the best complexity-adjusted fitness?
3. Is there a viable high-frequency niche (0-2s holding) at current fees?
4. Do microstructure and macro signals synergize or just diversify?
5. What is the optimal portfolio size (3 agents? 5? 10?)?

---

## 12. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Overfitting despite defenses | High | 5-layer defense + temporal holdout |
| Computational cost (~11 days) | Medium | Algo output caching reduces to ~3 days |
| Non-stationarity | High | MAP-Elites diversity = some agents survive shifts |
| Implementation complexity | Medium | Stage 1 requires zero new framework code |
| Single-developer maintenance | Medium | Clean separation: genome / fitness / evolution |

---

## 13. References

**Evolutionary algorithms:**
- Hansen, N. (2006) — "The CMA Evolution Strategy: A Tutorial", arXiv
- Mouret, J.B. & Clune, J. (2015) — "Illuminating search spaces by mapping
  elites", arXiv:1504.04909 (MAP-Elites)
- Pugh, J.K., Soros, L.B. & Stanley, K.O. (2016) — "Quality Diversity: A New
  Frontier for Evolutionary Computation", Frontiers in Robotics and AI

**Evolutionary finance:**
- Dempster, M.A.H. & Romahi, Y.S. (2002) — "Intraday FX trading: an evolutionary
  reinforcement learning approach", IDEAL 2002
- Lohpetch, D. & Corne, D. (2010) — "Outperforming buy-and-hold with evolved
  technical trading rules", EvoApplications 2010
- Chen, S.H. (2012) — "Genetic Programming and Financial Trading: How Much about
  What We Know is What We Know?", Handbook of Financial Engineering

**Multiple testing in finance:**
- Harvey, C.R. & Liu, Y. (2015) — "...and the Cross-Section of Expected Returns",
  Review of Financial Studies 29(1), 5-68 (deflated Sharpe ratio)
- Bailey, D.H. & Lopez de Prado, M. (2014) — "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality",
  Journal of Portfolio Management 40(5), 94-107

**Multi-agent systems:**
- Spooner, T., Fearnley, J., Savani, R. & Koukorinis, A. (2018) — "Market Making
  via Reinforcement Learning", AAMAS 2018
- Cont, R. & Bouchaud, J.P. (2000) — "Herd behavior and aggregate fluctuations
  in financial markets", Macroeconomic Dynamics 4(2), 170-196
