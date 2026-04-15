# Agent-Based Quantitative Research System Architecture

**Date:** 2026-04-02
**Vision:** Autonomous multi-agent system for trading algorithm discovery, testing, and evolution with real-time web-based monitoring

---

## Executive Summary

Build a system where **agents autonomously generate, test, and evolve trading strategies** while you monitor their progress via web dashboard. Agents operate 24/7, systematically exploring the strategy space, rigorously validating results, and learning from successes/failures.

**Core Concept:**
```
Genotype (Strategy DNA) → Agent (Executor) → Phenotype (Backtest) → Fitness (Sharpe) → Evolution (Next Gen)
                                    ↓
                            Web Dashboard (Monitor)
```

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        WEB DASHBOARD                            │
│  - Agent status and activity                                    │
│  - Real-time performance metrics                                │
│  - Strategy genealogy (evolution tree)                          │
│  - Best strategies leaderboard                                  │
│  - Resource usage (CPU, memory)                                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │ WebSocket / REST API
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR SERVICE                         │
│  - Task queue management (Redis/Celery)                         │
│  - Agent lifecycle management                                   │
│  - Results aggregation                                          │
│  - Database persistence (PostgreSQL/TimescaleDB)                │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│  AGENT 1  │   │  AGENT 2  │   │  AGENT N  │
│           │   │           │   │           │
│ - Genotype│   │ - Genotype│   │ - Genotype│
│ - Backtest│   │ - Backtest│   │ - Backtest│
│ - Evaluate│   │ - Evaluate│   │ - Evaluate│
│ - Report  │   │ - Report  │   │ - Report  │
└─────┬─────┘   └─────┬─────┘   └─────┬─────┘
      │               │               │
      └───────────────┼───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │   SHARED RESOURCES          │
        │ - Daily data (Parquet)      │
        │ - Feature cache             │
        │ - Model registry            │
        │ - Experiment database       │
        └─────────────────────────────┘
```

---

## Component 1: Genotype System (Strategy DNA)

### What is a Genotype?

A **genotype** is a high-level, parameterized representation of a trading strategy. Think of it as "strategy DNA" that can be:
- **Evolved** (genetic algorithms)
- **Mutated** (random parameter changes)
- **Crossed over** (combine two successful strategies)
- **Evaluated** (fitness = Sharpe ratio)

### Genotype Schema

```python
# genotypes/base_genotype.py

from dataclasses import dataclass
from typing import Dict, Any, List
from enum import Enum

class StrategyFamily(Enum):
    """High-level strategy categories."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    REGIME_SWITCHING = "regime_switching"
    HYBRID = "hybrid"

@dataclass
class Genotype:
    """
    Strategy DNA - high-level parameterized strategy representation.

    Can be evolved, mutated, crossed over, and evaluated.
    """
    # Identifier
    id: str  # UUID
    generation: int  # Evolution generation number
    parent_ids: List[str]  # Parent genotypes (for genealogy)

    # Strategy Definition
    family: StrategyFamily
    parameters: Dict[str, Any]  # Strategy-specific parameters

    # Performance Metrics (fitness)
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trades_per_year: int = 0
    oos_is_ratio: float = 0.0  # Out-of-sample / In-sample

    # Metadata
    created_at: str
    evaluated: bool = False
    validation_status: str = "pending"  # pending, passed, failed

    def to_phenotype(self) -> "Phenotype":
        """
        Convert genotype (DNA) to phenotype (executable strategy).

        This is where the magic happens - genotype parameters
        are translated into actual trading logic.
        """
        raise NotImplementedError("Subclass must implement")

    def mutate(self, mutation_rate: float = 0.1) -> "Genotype":
        """Create mutated copy of this genotype."""
        raise NotImplementedError("Subclass must implement")

    @staticmethod
    def crossover(parent1: "Genotype", parent2: "Genotype") -> "Genotype":
        """Create offspring from two parent genotypes."""
        raise NotImplementedError("Subclass must implement")
```

### Example: MA Crossover Genotype

```python
# genotypes/ma_crossover_genotype.py

@dataclass
class MACrossoverGenotype(Genotype):
    """
    Moving Average Crossover strategy genotype.

    Parameters (genes):
    - ma_period: MA length (20-60 days)
    - entry_threshold: How far above MA to enter (0.0-0.05)
    - exit_threshold: How far below MA to exit (0.0-0.05)
    - regime_filter: Whether to filter by regime (bool)
    - min_trend_strength: Minimum trend strength to trade (0.0-1.0)
    - position_size_method: equal_weight, volatility_scaled, kelly
    """

    def __init__(self, **kwargs):
        # Default parameters
        default_params = {
            'ma_period': 44,
            'entry_threshold': 0.0,
            'exit_threshold': 0.0,
            'regime_filter': False,
            'min_trend_strength': 0.0,
            'position_size_method': 'equal_weight',
            'stop_loss_atr_multiple': None,  # No stop loss by default
        }

        params = {**default_params, **kwargs.get('parameters', {})}
        super().__init__(
            family=StrategyFamily.TREND_FOLLOWING,
            parameters=params,
            **{k: v for k, v in kwargs.items() if k != 'parameters'}
        )

    def to_phenotype(self) -> "Phenotype":
        """Convert to executable strategy."""
        from strategies.ma_crossover import MACrossoverStrategy

        return MACrossoverStrategy(
            ma_period=self.parameters['ma_period'],
            entry_threshold=self.parameters['entry_threshold'],
            exit_threshold=self.parameters['exit_threshold'],
            regime_filter=self.parameters['regime_filter'],
            min_trend_strength=self.parameters['min_trend_strength'],
            position_size_method=self.parameters['position_size_method'],
            stop_loss_atr_multiple=self.parameters.get('stop_loss_atr_multiple'),
        )

    def mutate(self, mutation_rate: float = 0.1) -> "MACrossoverGenotype":
        """
        Create mutated copy.

        Each parameter has mutation_rate chance of being changed.
        """
        import random
        import copy

        new_genotype = copy.deepcopy(self)
        new_genotype.id = str(uuid.uuid4())
        new_genotype.generation = self.generation + 1
        new_genotype.parent_ids = [self.id]
        new_genotype.evaluated = False

        params = new_genotype.parameters

        # Mutate ma_period (±10%)
        if random.random() < mutation_rate:
            params['ma_period'] = int(
                np.clip(
                    params['ma_period'] + random.randint(-5, 5),
                    20, 60
                )
            )

        # Mutate entry_threshold
        if random.random() < mutation_rate:
            params['entry_threshold'] += random.uniform(-0.01, 0.01)
            params['entry_threshold'] = np.clip(params['entry_threshold'], 0.0, 0.05)

        # Mutate exit_threshold
        if random.random() < mutation_rate:
            params['exit_threshold'] += random.uniform(-0.01, 0.01)
            params['exit_threshold'] = np.clip(params['exit_threshold'], 0.0, 0.05)

        # Mutate regime_filter (flip with small probability)
        if random.random() < mutation_rate / 2:
            params['regime_filter'] = not params['regime_filter']

        # Mutate min_trend_strength
        if random.random() < mutation_rate:
            params['min_trend_strength'] += random.uniform(-0.1, 0.1)
            params['min_trend_strength'] = np.clip(params['min_trend_strength'], 0.0, 1.0)

        return new_genotype

    @staticmethod
    def crossover(parent1: "MACrossoverGenotype", parent2: "MACrossoverGenotype") -> "MACrossoverGenotype":
        """
        Create offspring from two parents.

        Randomly inherit parameters from each parent.
        """
        import random

        child_params = {}
        for key in parent1.parameters.keys():
            # 50% chance to inherit from each parent
            if random.random() < 0.5:
                child_params[key] = parent1.parameters[key]
            else:
                child_params[key] = parent2.parameters[key]

        return MACrossoverGenotype(
            id=str(uuid.uuid4()),
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
            parameters=child_params,
            created_at=datetime.now().isoformat(),
        )
```

---

## Component 2: Agent System

### Agent Types

**1. Generator Agent**
- Generates new genotypes (random or informed)
- Seeds initial population
- Creates diversity

**2. Evaluator Agent**
- Takes genotype from queue
- Converts to phenotype (executable strategy)
- Runs backtest on daily data
- Computes fitness metrics
- Reports results

**3. Evolver Agent**
- Reads population from database
- Selects top performers (fitness)
- Creates next generation via mutation/crossover
- Submits new genotypes to queue

**4. Monitor Agent**
- Tracks agent health
- Monitors resource usage
- Detects stuck agents
- Reports to dashboard

### Agent Base Class

```python
# agents/base_agent.py

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import logging

@dataclass
class AgentStatus:
    """Agent status for monitoring."""
    agent_id: str
    agent_type: str
    status: str  # idle, working, error, stopped
    current_task: Optional[str]
    tasks_completed: int
    tasks_failed: int
    uptime_seconds: float
    last_heartbeat: str
    metrics: Dict[str, Any]  # Agent-specific metrics

class BaseAgent(ABC):
    """
    Base class for all agents.

    Provides:
    - Task queue integration
    - Database connection
    - Logging
    - Status reporting
    - Heartbeat mechanism
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        task_queue,  # Redis/Celery queue
        database,    # PostgreSQL connection
        config: Dict[str, Any]
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.task_queue = task_queue
        self.database = database
        self.config = config

        self.logger = logging.getLogger(f"Agent.{agent_type}.{agent_id}")
        self.status = AgentStatus(
            agent_id=agent_id,
            agent_type=agent_type,
            status="idle",
            current_task=None,
            tasks_completed=0,
            tasks_failed=0,
            uptime_seconds=0.0,
            last_heartbeat=datetime.now().isoformat(),
            metrics={}
        )

        self.start_time = datetime.now()
        self.running = False

    async def start(self):
        """Start agent main loop."""
        self.running = True
        self.logger.info(f"Agent {self.agent_id} starting")

        # Start heartbeat task
        asyncio.create_task(self._heartbeat_loop())

        # Start main work loop
        try:
            await self._work_loop()
        except Exception as e:
            self.logger.error(f"Agent crashed: {e}", exc_info=True)
            self.status.status = "error"
            raise
        finally:
            self.running = False

    async def _work_loop(self):
        """Main work loop - get tasks and execute."""
        while self.running:
            try:
                # Get task from queue
                task = await self.task_queue.get_task(
                    agent_type=self.agent_type,
                    timeout=5.0
                )

                if task is None:
                    # No tasks available, idle
                    self.status.status = "idle"
                    await asyncio.sleep(1.0)
                    continue

                # Execute task
                self.status.status = "working"
                self.status.current_task = task.id

                self.logger.info(f"Processing task {task.id}")

                result = await self.execute_task(task)

                # Report result
                await self.task_queue.complete_task(task.id, result)
                self.status.tasks_completed += 1
                self.logger.info(f"Task {task.id} completed")

            except Exception as e:
                self.logger.error(f"Task failed: {e}", exc_info=True)
                self.status.tasks_failed += 1
                await self.task_queue.fail_task(task.id, str(e))

    async def _heartbeat_loop(self):
        """Send heartbeat every 10 seconds."""
        while self.running:
            self.status.last_heartbeat = datetime.now().isoformat()
            self.status.uptime_seconds = (datetime.now() - self.start_time).total_seconds()

            # Report status to database
            await self.database.update_agent_status(self.status)

            await asyncio.sleep(10.0)

    @abstractmethod
    async def execute_task(self, task) -> Dict[str, Any]:
        """Execute specific task. Implemented by subclasses."""
        pass

    async def stop(self):
        """Gracefully stop agent."""
        self.logger.info(f"Agent {self.agent_id} stopping")
        self.running = False
        self.status.status = "stopped"
```

### Evaluator Agent (Most Important)

```python
# agents/evaluator_agent.py

import polars as pl
from pathlib import Path
from typing import Dict, Any

class EvaluatorAgent(BaseAgent):
    """
    Agent that evaluates trading strategies.

    Tasks:
    1. Receive genotype from queue
    2. Convert to phenotype (executable strategy)
    3. Load daily data
    4. Run backtest
    5. Compute fitness metrics
    6. Store results in database
    7. Update genotype with fitness
    """

    async def execute_task(self, task) -> Dict[str, Any]:
        """
        Evaluate a genotype.

        Task format:
        {
            'type': 'evaluate',
            'genotype_id': 'uuid',
            'genotype': <serialized genotype>,
            'symbols': ['BTC', 'ETH'],
            'start_date': '2024-01-01',
            'end_date': '2026-04-01',
        }
        """
        genotype_data = task['genotype']
        genotype = self._deserialize_genotype(genotype_data)

        self.logger.info(f"Evaluating genotype {genotype.id} (generation {genotype.generation})")

        # Convert to executable strategy
        strategy = genotype.to_phenotype()

        # Load daily data
        daily_data = {}
        for symbol in task['symbols']:
            data_path = Path(f"./data/daily/{symbol}.parquet")
            if data_path.exists():
                daily_data[symbol] = pl.read_parquet(data_path)
                self.logger.info(f"Loaded {len(daily_data[symbol])} days for {symbol}")

        if not daily_data:
            raise ValueError("No daily data available")

        # Run backtest for each symbol
        results = {}
        for symbol, data in daily_data.items():
            # Filter date range
            data = data.filter(
                (pl.col('timestamp') >= task['start_date']) &
                (pl.col('timestamp') <= task['end_date'])
            )

            # Generate signals
            signals = strategy.generate_signals(data)

            # Backtest
            backtest_result = self._run_backtest(data, signals)
            results[symbol] = backtest_result

        # Aggregate metrics across symbols
        aggregated_metrics = self._aggregate_results(results)

        # Update genotype with fitness
        genotype.sharpe_ratio = aggregated_metrics['sharpe_ratio']
        genotype.max_drawdown = aggregated_metrics['max_drawdown']
        genotype.win_rate = aggregated_metrics['win_rate']
        genotype.trades_per_year = aggregated_metrics['trades_per_year']
        genotype.evaluated = True
        genotype.validation_status = "passed" if aggregated_metrics['sharpe_ratio'] > 0.3 else "failed"

        # Store in database
        await self.database.update_genotype(genotype)
        await self.database.store_backtest_results(genotype.id, results)

        self.logger.info(
            f"Genotype {genotype.id}: Sharpe={genotype.sharpe_ratio:.3f}, "
            f"MaxDD={genotype.max_drawdown:.3f}, WinRate={genotype.win_rate:.3f}"
        )

        return {
            'genotype_id': genotype.id,
            'fitness': genotype.sharpe_ratio,
            'metrics': aggregated_metrics,
        }

    def _run_backtest(self, data: pl.DataFrame, signals: pl.DataFrame) -> Dict[str, Any]:
        """
        Run backtest given data and signals.

        Returns metrics: sharpe, max_drawdown, win_rate, etc.
        """
        # Merge data with signals
        df = data.join(signals, on='timestamp', how='left')

        # Compute returns
        df = df.with_columns([
            pl.col('close').pct_change().alias('market_return'),
        ])

        # Strategy returns (position * market_return, shifted by 1)
        df = df.with_columns([
            (pl.col('market_return') * pl.col('signal').shift(1)).alias('strategy_return'),
        ])

        # Transaction costs (8 bps per trade)
        df = df.with_columns([
            (pl.col('signal').diff().abs() * 0.0008).alias('txn_cost'),
        ])

        df = df.with_columns([
            (pl.col('strategy_return') - pl.col('txn_cost')).alias('net_return'),
        ])

        # Compute metrics
        df = df.drop_nulls()

        sharpe = df['net_return'].mean() / df['net_return'].std() * np.sqrt(252)

        # Cumulative returns for drawdown
        cumulative = (1 + df['net_return']).cum_prod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        trades = df.filter(pl.col('signal').diff().abs() > 0)
        if len(trades) > 0:
            wins = (trades['net_return'] > 0).sum()
            win_rate = wins / len(trades)
        else:
            win_rate = 0.0

        # Trades per year
        n_trades = (df['signal'].diff().abs() > 0).sum()
        years = len(df) / 252
        trades_per_year = n_trades / years if years > 0 else 0

        return {
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'trades_per_year': int(trades_per_year),
            'total_return': float((cumulative.tail(1).item() - 1) * 100),  # %
            'n_trades': int(n_trades),
        }

    def _aggregate_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple symbols."""
        # Equal-weighted average
        sharpe_ratios = [r['sharpe_ratio'] for r in results.values()]
        max_drawdowns = [r['max_drawdown'] for r in results.values()]
        win_rates = [r['win_rate'] for r in results.values()]
        trades_per_year = [r['trades_per_year'] for r in results.values()]

        return {
            'sharpe_ratio': np.mean(sharpe_ratios),
            'max_drawdown': np.mean(max_drawdowns),
            'win_rate': np.mean(win_rates),
            'trades_per_year': int(np.mean(trades_per_year)),
            'by_symbol': results,
        }
```

---

## Component 3: Web Dashboard

### Technology Stack

**Backend:**
- FastAPI (REST API + WebSocket)
- PostgreSQL (agent status, genotypes, results)
- Redis (task queue, caching)

**Frontend:**
- React + TypeScript
- Chart.js or Plotly (performance charts)
- WebSocket (real-time updates)
- Tailwind CSS (styling)

### Dashboard Pages

**1. Overview Page**
```
┌─────────────────────────────────────────────────────────────┐
│  NAT Agent Research Dashboard                     [Settings]│
├─────────────────────────────────────────────────────────────┤
│  Active Agents: 8/10        Tasks Queued: 23               │
│  Strategies Evaluated: 1,247   Best Sharpe: 1.23           │
│  Current Generation: 15        Uptime: 3d 7h 42m           │
├─────────────────────────────────────────────────────────────┤
│  Agent Status                                               │
│  ┌──────┬──────────────┬─────────┬──────────┬────────────┐ │
│  │ ID   │ Type         │ Status  │ Task     │ Completed  │ │
│  ├──────┼──────────────┼─────────┼──────────┼────────────┤ │
│  │ A001 │ Evaluator    │ Working │ Gen15-42 │ 127        │ │
│  │ A002 │ Evaluator    │ Idle    │ -        │ 103        │ │
│  │ A003 │ Evolver      │ Working │ Gen15    │ 15         │ │
│  │ A004 │ Generator    │ Idle    │ -        │ 234        │ │
│  └──────┴──────────────┴─────────┴──────────┴────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Top Strategies (by Sharpe)                                 │
│  ┌──────────┬─────────┬────────┬─────────┬───────────────┐ │
│  │ ID       │ Sharpe  │ MaxDD  │ WinRate │ Family        │ │
│  ├──────────┼─────────┼────────┼─────────┼───────────────┤ │
│  │ G15-042  │ 1.23    │ -18%   │ 62%     │ Trend+Regime  │ │
│  │ G14-089  │ 1.18    │ -22%   │ 58%     │ Adaptive MA   │ │
│  │ G15-013  │ 1.12    │ -16%   │ 60%     │ Hybrid        │ │
│  └──────────┴─────────┴────────┴─────────┴───────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**2. Strategy Explorer Page**
- Search/filter genotypes
- View parameters
- See performance metrics
- Download backtest results
- View genealogy (parent → child)

**3. Evolution Tree Page**
- Visual graph of strategy evolution
- Color-coded by fitness
- Interactive (click to see details)
- Track lineages

**4. Real-Time Activity Page**
- Live feed of agent actions
- "Agent A001 evaluating genotype G15-042..."
- "Genotype G15-042 completed: Sharpe 1.23"
- "Evolver creating generation 16..."

**5. Performance Analytics Page**
- Sharpe ratio distribution over time
- Best strategies per generation
- Convergence metrics
- Diversity metrics (are all strategies similar?)

### API Endpoints

```python
# dashboard/api/main.py

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio

app = FastAPI(title="NAT Agent Dashboard")

# REST API Endpoints

@app.get("/api/agents/status")
async def get_agents_status():
    """Get current status of all agents."""
    agents = await database.get_all_agent_status()
    return agents

@app.get("/api/genotypes/top")
async def get_top_genotypes(limit: int = 10, metric: str = "sharpe_ratio"):
    """Get top performing genotypes."""
    genotypes = await database.get_top_genotypes(metric=metric, limit=limit)
    return genotypes

@app.get("/api/genotypes/{genotype_id}")
async def get_genotype_details(genotype_id: str):
    """Get detailed information about a genotype."""
    genotype = await database.get_genotype(genotype_id)
    backtest_results = await database.get_backtest_results(genotype_id)

    return {
        'genotype': genotype,
        'backtest_results': backtest_results,
        'genealogy': await database.get_genealogy(genotype_id),
    }

@app.get("/api/evolution/stats")
async def get_evolution_stats():
    """Get evolution statistics."""
    stats = await database.get_evolution_stats()
    return stats

@app.post("/api/agents/spawn")
async def spawn_agent(agent_type: str, config: dict):
    """Spawn a new agent."""
    agent_id = await orchestrator.spawn_agent(agent_type, config)
    return {'agent_id': agent_id}

@app.post("/api/agents/{agent_id}/stop")
async def stop_agent(agent_id: str):
    """Stop an agent."""
    await orchestrator.stop_agent(agent_id)
    return {'status': 'stopped'}

# WebSocket for Real-Time Updates

@app.websocket("/ws/activity")
async def websocket_activity(websocket: WebSocket):
    """
    WebSocket endpoint for real-time activity feed.

    Streams agent actions and results as they happen.
    """
    await websocket.accept()

    try:
        # Subscribe to activity feed (Redis pub/sub)
        async for message in activity_feed.subscribe():
            await websocket.send_json(message)

    except WebSocketDisconnect:
        pass

@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metrics.

    Sends updated metrics every second.
    """
    await websocket.accept()

    try:
        while True:
            metrics = await database.get_current_metrics()
            await websocket.send_json(metrics)
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        pass
```

---

## Component 4: Evolution Engine

### Genetic Algorithm

```python
# evolution/genetic_algorithm.py

from typing import List
import numpy as np

class EvolutionEngine:
    """
    Genetic algorithm for strategy evolution.

    Process:
    1. Evaluate population (fitness = Sharpe ratio)
    2. Select top performers (elitism)
    3. Breed next generation (crossover + mutation)
    4. Add random diversity (random new genotypes)
    5. Submit to evaluation queue
    """

    def __init__(
        self,
        population_size: int = 100,
        elitism_rate: float = 0.1,  # Keep top 10%
        crossover_rate: float = 0.6,  # 60% from breeding
        mutation_rate: float = 0.2,  # 20% from mutation
        random_rate: float = 0.1,  # 10% completely random
    ):
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_rate = random_rate

    async def evolve_generation(
        self,
        current_population: List[Genotype],
        generation: int
    ) -> List[Genotype]:
        """
        Create next generation from current population.

        Returns new population of genotypes to evaluate.
        """
        # Sort by fitness (Sharpe ratio)
        current_population.sort(key=lambda g: g.sharpe_ratio, reverse=True)

        next_generation = []

        # 1. Elitism - keep top performers
        n_elite = int(self.population_size * self.elitism_rate)
        elite = current_population[:n_elite]
        next_generation.extend(elite)

        # 2. Crossover - breed top performers
        n_crossover = int(self.population_size * self.crossover_rate)
        for _ in range(n_crossover):
            # Tournament selection (pick 5 random, take best 2)
            tournament = np.random.choice(current_population[:50], size=5, replace=False)
            tournament = sorted(tournament, key=lambda g: g.sharpe_ratio, reverse=True)
            parent1, parent2 = tournament[0], tournament[1]

            # Crossover
            child = type(parent1).crossover(parent1, parent2)
            child.generation = generation
            next_generation.append(child)

        # 3. Mutation - mutate top performers
        n_mutation = int(self.population_size * self.mutation_rate)
        for _ in range(n_mutation):
            parent = np.random.choice(current_population[:30])
            mutant = parent.mutate(mutation_rate=0.2)
            mutant.generation = generation
            next_generation.append(mutant)

        # 4. Random diversity - completely new genotypes
        n_random = self.population_size - len(next_generation)
        for _ in range(n_random):
            random_genotype = self._create_random_genotype(generation)
            next_generation.append(random_genotype)

        return next_generation

    def _create_random_genotype(self, generation: int) -> Genotype:
        """Create random genotype for diversity."""
        # Randomly choose strategy family
        family = np.random.choice(list(StrategyFamily))

        # Create genotype with random parameters
        if family == StrategyFamily.TREND_FOLLOWING:
            return MACrossoverGenotype(
                id=str(uuid.uuid4()),
                generation=generation,
                parent_ids=[],
                parameters={
                    'ma_period': np.random.randint(20, 61),
                    'entry_threshold': np.random.uniform(0.0, 0.05),
                    'exit_threshold': np.random.uniform(0.0, 0.05),
                    'regime_filter': np.random.choice([True, False]),
                    'min_trend_strength': np.random.uniform(0.0, 0.5),
                    'position_size_method': np.random.choice(['equal_weight', 'volatility_scaled']),
                },
                created_at=datetime.now().isoformat(),
            )
        # Add more strategy families...
```

---

## Implementation Plan: Task Sequence

### Phase 1: Foundation (Week 1-2)

**Goal:** Get basic agent system working with simple genotypes

**Tasks:**

1. **Database Schema** (2 days)
   ```sql
   -- genotypes table
   CREATE TABLE genotypes (
       id UUID PRIMARY KEY,
       generation INTEGER,
       family VARCHAR(50),
       parameters JSONB,
       sharpe_ratio FLOAT,
       max_drawdown FLOAT,
       win_rate FLOAT,
       evaluated BOOLEAN,
       created_at TIMESTAMP
   );

   -- agent_status table
   CREATE TABLE agent_status (
       agent_id VARCHAR(50) PRIMARY KEY,
       agent_type VARCHAR(50),
       status VARCHAR(20),
       current_task VARCHAR(100),
       tasks_completed INTEGER,
       last_heartbeat TIMESTAMP,
       metrics JSONB
   );

   -- backtest_results table
   CREATE TABLE backtest_results (
       id SERIAL PRIMARY KEY,
       genotype_id UUID REFERENCES genotypes(id),
       symbol VARCHAR(10),
       sharpe_ratio FLOAT,
       trades JSONB,
       equity_curve JSONB,
       created_at TIMESTAMP
   );
   ```

2. **Task Queue Setup** (1 day)
   - Install Redis
   - Install Celery
   - Create task definitions

3. **Base Agent Implementation** (2 days)
   - `agents/base_agent.py`
   - Heartbeat mechanism
   - Task queue integration

4. **Simple Genotype** (2 days)
   - Implement `MACrossoverGenotype`
   - Test mutation
   - Test crossover

5. **Evaluator Agent** (3 days)
   - Implement backtest logic
   - Test on daily data
   - Verify metrics calculation

6. **Basic Dashboard** (3 days)
   - FastAPI backend
   - Simple HTML frontend
   - Display agent status

**Deliverable:** Single evaluator agent can receive genotype, backtest it, report results via web dashboard.

---

### Phase 2: Evolution Engine (Week 3-4)

**Goal:** Implement genetic algorithm and multi-agent coordination

**Tasks:**

7. **Generator Agent** (2 days)
   - Create random genotypes
   - Submit to queue

8. **Evolver Agent** (3 days)
   - Read population from database
   - Run genetic algorithm
   - Create next generation

9. **Multi-Agent Orchestrator** (3 days)
   - Spawn/stop agents dynamically
   - Load balancing
   - Error recovery

10. **Evolution Metrics** (2 days)
    - Track diversity
    - Track convergence
    - Genealogy tracking

11. **Enhanced Dashboard** (3 days)
    - Evolution tree visualization
    - Performance charts
    - Real-time activity feed

**Deliverable:** 5-10 agents autonomously evolving strategies, visible via dashboard.

---

### Phase 3: Advanced Features (Week 5-8)

**Goal:** Add sophisticated strategy types and validation

**Tasks:**

12. **More Genotype Types** (5 days)
    - Regime-switching genotype
    - Mean reversion genotype
    - Hybrid genotype
    - Multi-symbol genotype

13. **Walk-Forward Validation** (3 days)
    - OOS/IS ratio calculation
    - Time-series split
    - Robustness testing

14. **Multi-Objective Optimization** (4 days)
    - Optimize for Sharpe AND drawdown
    - Pareto frontier
    - Diverse strategy portfolio

15. **Strategy Composition** (5 days)
    - Combine multiple strategies
    - Portfolio allocation
    - Correlation analysis

16. **Advanced Dashboard Features** (5 days)
    - Strategy comparison
    - Parameter sensitivity analysis
    - Export to live trading
    - Paper trading integration

**Deliverable:** Production-ready system discovering diverse, validated strategies.

---

## Performance Evaluation Criteria

### Agent-Level Metrics

```python
class AgentPerformanceMetrics:
    """Metrics for evaluating agent performance."""

    # Throughput
    tasks_per_hour: float
    avg_task_duration_seconds: float

    # Quality
    success_rate: float  # % of tasks completed successfully
    avg_strategy_sharpe: float  # Average fitness of evaluated strategies

    # Reliability
    uptime_percentage: float
    errors_per_hour: float

    # Resource Usage
    cpu_usage_percent: float
    memory_usage_mb: float
```

### System-Level Metrics

```python
class SystemPerformanceMetrics:
    """Metrics for overall system performance."""

    # Evolution Progress
    current_generation: int
    best_sharpe_ever: float
    best_sharpe_this_generation: float
    avg_sharpe_this_generation: float

    # Diversity
    unique_strategy_families: int
    parameter_diversity_score: float  # Variance in parameters

    # Convergence
    improvement_rate: float  # % improvement per generation
    plateau_generations: int  # Generations without improvement

    # Efficiency
    strategies_evaluated_per_day: int
    cpu_hours_per_strategy: float

    # Discovery
    strategies_above_threshold: int  # Sharpe > 0.5
    production_ready_strategies: int  # Passed all validation
```

### Validation Criteria (Before Production)

A strategy is **production-ready** if:

1. ✅ **Sharpe ratio > 0.5** (after transaction costs)
2. ✅ **Max drawdown < 25%**
3. ✅ **Win rate > 50%**
4. ✅ **OOS/IS ratio > 0.6** (robust out-of-sample)
5. ✅ **Tested on multiple symbols** (BTC, ETH minimum)
6. ✅ **Tested on multiple time periods** (walk-forward)
7. ✅ **No obvious overfitting** (parameter sensitivity analysis)
8. ✅ **Passes hypothesis tests** (from your existing framework)

---

## Technology Stack Summary

| Component | Technology | Why |
|-----------|------------|-----|
| **Database** | PostgreSQL + TimescaleDB | Time-series data, JSONB for flexibility |
| **Task Queue** | Redis + Celery | Distributed task processing |
| **API Backend** | FastAPI | Async, fast, WebSocket support |
| **Dashboard Frontend** | React + TypeScript | Interactive, real-time updates |
| **Visualization** | Plotly.js | Interactive charts |
| **Agent Runtime** | Python asyncio | Concurrent execution |
| **Data Storage** | Parquet (daily data) | Fast, columnar |
| **Messaging** | Redis Pub/Sub | Real-time activity feed |
| **Monitoring** | Prometheus + Grafana (optional) | System metrics |

---

## Realistic Timeline

| Phase | Duration | Deliverable | Complexity |
|-------|----------|-------------|------------|
| **Phase 1: Foundation** | 2 weeks | Single agent + basic dashboard | Medium |
| **Phase 2: Evolution** | 2 weeks | Multi-agent system evolving strategies | High |
| **Phase 3: Advanced** | 4 weeks | Production-ready with validation | Very High |
| **Testing & Refinement** | 2 weeks | Stable, documented, deployed | Medium |
| **TOTAL** | **10-12 weeks** | Autonomous strategy discovery system | - |

---

## Next Steps

**I can implement this system in phases. Which would you like to start with?**

1. **Phase 1 Foundation** - Get basic agent + dashboard working (2 weeks)
2. **Just the genotype system** - Design strategy DNA (3 days)
3. **Just the dashboard** - Web UI first, agents later (1 week)
4. **Proof of concept** - Minimal viable system (1 week)

This is ambitious but achievable. You already have 80% of the infrastructure (data, features, backtesting). This adds the autonomous discovery layer on top.

**This would be genuinely novel in the quant space** - most firms don't have this level of automation and transparency.
