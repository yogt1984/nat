# Agent System Implementation: Task Sequence

**Goal:** Build autonomous multi-agent trading research system from ground up
**Timeline:** 10-12 weeks full implementation, 2 weeks for proof-of-concept
**Approach:** Incremental - each task produces testable deliverable

---

## Task Dependency Graph

```
Foundation Layer (Week 1-2)
├─ T1: Database Schema ──────────┐
├─ T2: Task Queue Setup ─────────┤
├─ T3: Daily Data Pipeline ──────┤
└─ T4: Base Genotype Class ──────┤
                                 ▼
Agent Layer (Week 3-4)           T5: Base Agent Class
├─ T6: Evaluator Agent ──────────┤
├─ T7: Generator Agent ──────────┤
└─ T8: Evolver Agent ────────────┤
                                 ▼
Dashboard Layer (Week 5-6)       T9: Agent Orchestrator
├─ T10: FastAPI Backend ─────────┤
├─ T11: WebSocket Real-time ─────┤
├─ T12: React Frontend ──────────┤
└─ T13: Visualization ───────────┤
                                 ▼
Evolution Layer (Week 7-8)       T14: System Integration
├─ T15: Genetic Algorithm ───────┤
├─ T16: Multi-Objective Opt ─────┤
└─ T17: Walk-Forward Valid ──────┤
                                 ▼
Production Layer (Week 9-10)     T18: Advanced Features
├─ T19: More Genotypes ──────────┤
├─ T20: Validation Pipeline ─────┤
└─ T21: Export to Trading ───────┘
```

---

## Detailed Task Breakdown

### PHASE 1: FOUNDATION (Week 1-2)

---

#### **T1: Database Schema Design and Implementation**

**Priority:** P0 (Critical - everything depends on this)
**Duration:** 2 days
**Dependencies:** None

**Objectives:**
1. Design schema for genotypes, agents, results
2. Set up PostgreSQL with TimescaleDB extension
3. Create migration scripts
4. Write database access layer

**Deliverables:**

```sql
-- File: database/schema.sql

-- Genotypes table
CREATE TABLE genotypes (
    id UUID PRIMARY KEY,
    generation INTEGER NOT NULL,
    family VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,

    -- Fitness metrics
    sharpe_ratio FLOAT DEFAULT 0.0,
    max_drawdown FLOAT DEFAULT 0.0,
    win_rate FLOAT DEFAULT 0.0,
    trades_per_year INTEGER DEFAULT 0,
    oos_is_ratio FLOAT DEFAULT 0.0,

    -- Genealogy
    parent_ids UUID[] DEFAULT '{}',

    -- Status
    evaluated BOOLEAN DEFAULT FALSE,
    validation_status VARCHAR(20) DEFAULT 'pending',

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    evaluated_at TIMESTAMP,

    INDEX idx_generation (generation),
    INDEX idx_sharpe (sharpe_ratio DESC),
    INDEX idx_family (family)
);

-- Agent status table
CREATE TABLE agent_status (
    agent_id VARCHAR(50) PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    current_task VARCHAR(100),

    -- Performance
    tasks_completed INTEGER DEFAULT 0,
    tasks_failed INTEGER DEFAULT 0,
    uptime_seconds FLOAT DEFAULT 0.0,

    -- Health
    last_heartbeat TIMESTAMP DEFAULT NOW(),
    cpu_usage_percent FLOAT,
    memory_usage_mb FLOAT,

    -- Custom metrics (agent-specific)
    metrics JSONB,

    created_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_type (agent_type),
    INDEX idx_status (status)
);

-- Backtest results table
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    genotype_id UUID REFERENCES genotypes(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,

    -- Metrics
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    win_rate FLOAT,
    total_return FLOAT,
    n_trades INTEGER,

    -- Detailed results
    trades JSONB,  -- Array of trade details
    equity_curve JSONB,  -- Time series of equity

    created_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_genotype (genotype_id),
    INDEX idx_symbol (symbol)
);

-- Evolution history table (track each generation)
CREATE TABLE evolution_history (
    id SERIAL PRIMARY KEY,
    generation INTEGER NOT NULL,

    -- Statistics
    population_size INTEGER,
    best_sharpe FLOAT,
    avg_sharpe FLOAT,
    worst_sharpe FLOAT,
    diversity_score FLOAT,

    -- Top genotypes
    top_genotype_ids UUID[],

    created_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_generation (generation)
);

-- Activity log (for real-time feed)
CREATE TABLE activity_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    agent_id VARCHAR(50),
    event_type VARCHAR(50),  -- 'task_started', 'task_completed', 'genotype_evaluated', etc.
    message TEXT,
    data JSONB,

    INDEX idx_timestamp (timestamp DESC),
    INDEX idx_agent (agent_id),
    INDEX idx_event_type (event_type)
);
```

**Database Access Layer:**

```python
# File: database/db.py

import asyncpg
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class Database:
    """Async database access layer."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(self.connection_string)

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()

    # Genotype operations

    async def insert_genotype(self, genotype: dict) -> str:
        """Insert new genotype, return ID."""
        async with self.pool.acquire() as conn:
            genotype_id = await conn.fetchval(
                """
                INSERT INTO genotypes (
                    id, generation, family, parameters,
                    parent_ids, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                genotype['id'],
                genotype['generation'],
                genotype['family'],
                json.dumps(genotype['parameters']),
                genotype.get('parent_ids', []),
                datetime.now()
            )
        return genotype_id

    async def update_genotype(self, genotype: dict):
        """Update genotype with fitness metrics."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE genotypes SET
                    sharpe_ratio = $2,
                    max_drawdown = $3,
                    win_rate = $4,
                    trades_per_year = $5,
                    oos_is_ratio = $6,
                    evaluated = $7,
                    validation_status = $8,
                    evaluated_at = $9
                WHERE id = $1
                """,
                genotype['id'],
                genotype.get('sharpe_ratio', 0.0),
                genotype.get('max_drawdown', 0.0),
                genotype.get('win_rate', 0.0),
                genotype.get('trades_per_year', 0),
                genotype.get('oos_is_ratio', 0.0),
                genotype.get('evaluated', False),
                genotype.get('validation_status', 'pending'),
                datetime.now()
            )

    async def get_top_genotypes(
        self,
        metric: str = 'sharpe_ratio',
        limit: int = 10
    ) -> List[dict]:
        """Get top performing genotypes."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM genotypes
                WHERE evaluated = TRUE
                ORDER BY {metric} DESC
                LIMIT $1
                """,
                limit
            )
        return [dict(row) for row in rows]

    async def get_generation(self, generation: int) -> List[dict]:
        """Get all genotypes from a generation."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM genotypes WHERE generation = $1",
                generation
            )
        return [dict(row) for row in rows]

    # Agent operations

    async def update_agent_status(self, status: dict):
        """Update agent status (upsert)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_status (
                    agent_id, agent_type, status, current_task,
                    tasks_completed, tasks_failed, uptime_seconds,
                    last_heartbeat, cpu_usage_percent, memory_usage_mb,
                    metrics
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (agent_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    current_task = EXCLUDED.current_task,
                    tasks_completed = EXCLUDED.tasks_completed,
                    tasks_failed = EXCLUDED.tasks_failed,
                    uptime_seconds = EXCLUDED.uptime_seconds,
                    last_heartbeat = EXCLUDED.last_heartbeat,
                    cpu_usage_percent = EXCLUDED.cpu_usage_percent,
                    memory_usage_mb = EXCLUDED.memory_usage_mb,
                    metrics = EXCLUDED.metrics
                """,
                status['agent_id'],
                status['agent_type'],
                status['status'],
                status.get('current_task'),
                status.get('tasks_completed', 0),
                status.get('tasks_failed', 0),
                status.get('uptime_seconds', 0.0),
                datetime.now(),
                status.get('cpu_usage_percent'),
                status.get('memory_usage_mb'),
                json.dumps(status.get('metrics', {}))
            )

    async def log_activity(
        self,
        agent_id: str,
        event_type: str,
        message: str,
        data: dict = None
    ):
        """Log activity for real-time feed."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO activity_log (
                    agent_id, event_type, message, data
                ) VALUES ($1, $2, $3, $4)
                """,
                agent_id,
                event_type,
                message,
                json.dumps(data) if data else None
            )
```

**Tests:**

```python
# File: tests/test_database.py

import pytest
import asyncio
from database.db import Database

@pytest.fixture
async def db():
    """Create test database connection."""
    db = Database("postgresql://localhost/nat_test")
    await db.connect()
    yield db
    await db.close()

@pytest.mark.asyncio
async def test_insert_and_retrieve_genotype(db):
    """Test genotype CRUD operations."""
    genotype = {
        'id': 'test-genotype-1',
        'generation': 1,
        'family': 'trend_following',
        'parameters': {'ma_period': 44},
    }

    # Insert
    genotype_id = await db.insert_genotype(genotype)
    assert genotype_id == 'test-genotype-1'

    # Update with fitness
    genotype['sharpe_ratio'] = 1.23
    genotype['evaluated'] = True
    await db.update_genotype(genotype)

    # Retrieve
    top = await db.get_top_genotypes(limit=1)
    assert len(top) == 1
    assert top[0]['sharpe_ratio'] == 1.23
```

**Success Criteria:**
- [ ] PostgreSQL running with schema created
- [ ] All CRUD operations work
- [ ] Tests passing
- [ ] Can insert/update/query genotypes
- [ ] Can track agent status

---

#### **T2: Task Queue Setup**

**Priority:** P0
**Duration:** 1 day
**Dependencies:** None

**Objectives:**
1. Install and configure Redis
2. Set up Celery for distributed task processing
3. Define task types
4. Create task queue interface

**Deliverables:**

```python
# File: tasks/queue.py

from celery import Celery
from typing import Dict, Any
import json

# Celery app
app = Celery(
    'nat_agents',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Task definitions

@app.task(name='tasks.evaluate_genotype')
def evaluate_genotype_task(genotype_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task: Evaluate a genotype (run backtest).

    This is a Celery task that will be picked up by evaluator agents.
    """
    # This is just the task definition
    # Actual execution happens in EvaluatorAgent.execute_task()
    return genotype_data

@app.task(name='tasks.evolve_generation')
def evolve_generation_task(generation: int) -> Dict[str, Any]:
    """Task: Create next generation from current population."""
    return {'generation': generation}

@app.task(name='tasks.generate_genotypes')
def generate_genotypes_task(n: int, generation: int) -> Dict[str, Any]:
    """Task: Generate N random genotypes."""
    return {'n': n, 'generation': generation}


# Task queue interface (for agents to use)

class TaskQueue:
    """Interface for agents to interact with task queue."""

    def __init__(self, celery_app: Celery):
        self.app = celery_app

    async def submit_task(self, task_type: str, data: Dict[str, Any]) -> str:
        """Submit task to queue, return task ID."""
        if task_type == 'evaluate':
            result = evaluate_genotype_task.delay(data)
        elif task_type == 'evolve':
            result = evolve_generation_task.delay(data)
        elif task_type == 'generate':
            result = generate_genotypes_task.delay(data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return result.id

    async def get_task_result(self, task_id: str, timeout: float = None):
        """Get result of completed task."""
        result = self.app.AsyncResult(task_id)
        return result.get(timeout=timeout)

    async def get_task_status(self, task_id: str) -> str:
        """Get status of task (PENDING, STARTED, SUCCESS, FAILURE)."""
        result = self.app.AsyncResult(task_id)
        return result.status
```

**Configuration:**

```python
# File: config/celery_config.py

broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/1'

task_routes = {
    'tasks.evaluate_genotype': {'queue': 'evaluation'},
    'tasks.evolve_generation': {'queue': 'evolution'},
    'tasks.generate_genotypes': {'queue': 'generation'},
}

worker_prefetch_multiplier = 1  # Only fetch one task at a time
task_acks_late = True  # Acknowledge task after completion
```

**Tests:**

```python
# File: tests/test_task_queue.py

import pytest
from tasks.queue import TaskQueue, app

@pytest.mark.asyncio
async def test_submit_and_retrieve_task():
    """Test task submission and retrieval."""
    queue = TaskQueue(app)

    # Submit task
    task_id = await queue.submit_task('evaluate', {'genotype_id': 'test-1'})
    assert task_id is not None

    # Check status
    status = await queue.get_task_status(task_id)
    assert status in ['PENDING', 'STARTED', 'SUCCESS']
```

**Success Criteria:**
- [ ] Redis running
- [ ] Celery workers can start
- [ ] Tasks can be submitted
- [ ] Tasks can be retrieved
- [ ] Multiple queues working (evaluation, evolution, generation)

---

#### **T3: Daily Data Aggregation Pipeline**

**Priority:** P1 (High - needed for backtesting)
**Duration:** 2 days
**Dependencies:** Your existing tick data

**Objectives:**
1. Convert tick-level Parquet to daily OHLCV
2. Add regime features (daily scale)
3. Add liquidation data (if available)
4. Create incremental update mechanism

**Deliverables:**

```python
# File: scripts/data/aggregate_to_daily.py

import polars as pl
from pathlib import Path
from typing import List
import argparse

def aggregate_tick_to_daily(
    tick_files: List[Path],
    output_path: Path,
    symbol: str
):
    """
    Aggregate tick-level features to daily OHLCV + features.

    Args:
        tick_files: List of Parquet files with tick data
        output_path: Where to write daily Parquet
        symbol: Symbol name (BTC, ETH, etc.)
    """
    print(f"Aggregating {len(tick_files)} files for {symbol}...")

    # Read all tick files
    dfs = [pl.read_parquet(f) for f in tick_files]
    df = pl.concat(dfs)

    print(f"Loaded {len(df)} ticks")

    # Aggregate to daily
    daily = df.groupby(
        pl.col('timestamp').dt.truncate('1d')
    ).agg([
        # OHLCV
        pl.col('close').last().alias('close'),
        pl.col('high').max().alias('high'),
        pl.col('low').min().alias('low'),
        pl.col('open').first().alias('open'),
        pl.col('volume').sum().alias('volume'),

        # Volume features
        pl.col('volume_buy').sum().alias('volume_buy'),
        pl.col('volume_sell').sum().alias('volume_sell'),

        # Regime features (average over day)
        pl.col('regime_absorption').mean().alias('absorption'),
        pl.col('regime_divergence').mean().alias('divergence'),
        pl.col('regime_churn').mean().alias('churn'),
        pl.col('regime_range_position').mean().alias('range_position'),

        # Illiquidity (if available)
        pl.col('kyle_lambda').mean().alias('kyle_lambda_daily'),
    ]).sort('timestamp')

    # Add derived features
    daily = daily.with_columns([
        # Returns
        pl.col('close').pct_change().alias('return'),

        # Volume imbalance
        ((pl.col('volume_buy') - pl.col('volume_sell')) / pl.col('volume')).alias('volume_imbalance'),

        # Buy pressure
        (pl.col('volume_buy') / pl.col('volume')).alias('buy_pressure'),

        # Amihud illiquidity
        (pl.col('return').abs() / (pl.col('volume') * pl.col('close'))).alias('amihud_illiquidity'),

        # Range position
        ((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low'))).alias('range_pos'),

        # ATR (Average True Range) - for position sizing
        pl.max_horizontal([
            pl.col('high') - pl.col('low'),
            (pl.col('high') - pl.col('close').shift(1)).abs(),
            (pl.col('low') - pl.col('close').shift(1)).abs(),
        ]).alias('true_range'),
    ])

    # Add moving averages of features
    daily = daily.with_columns([
        pl.col('volume').rolling_mean(20).alias('volume_ma_20'),
        pl.col('amihud_illiquidity').rolling_mean(20).alias('illiquidity_ma_20'),
        pl.col('true_range').rolling_mean(14).alias('atr_14'),
    ])

    # Volume ratio
    daily = daily.with_columns([
        (pl.col('volume') / pl.col('volume_ma_20')).alias('volume_ratio'),
    ])

    # Write to Parquet
    daily.write_parquet(output_path)
    print(f"Wrote {len(daily)} daily bars to {output_path}")

    return daily


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--symbols', nargs='+', default=['BTC', 'ETH'])

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in args.symbols:
        # Find all tick files for this symbol
        tick_files = sorted(input_dir.glob(f"{symbol}_*.parquet"))

        if not tick_files:
            print(f"No files found for {symbol}")
            continue

        # Aggregate
        output_path = output_dir / f"{symbol}.parquet"
        aggregate_tick_to_daily(tick_files, output_path, symbol)
```

**Makefile Target:**

```makefile
# Add to Makefile

aggregate_to_daily:
	python scripts/data/aggregate_to_daily.py \
		--input-dir ./data/features \
		--output-dir ./data/daily \
		--symbols BTC ETH
```

**Success Criteria:**
- [ ] Can aggregate tick → daily
- [ ] Daily data includes OHLCV + 15 features
- [ ] Multiple symbols supported
- [ ] Incremental updates possible
- [ ] Output format suitable for backtesting

---

#### **T4: Base Genotype Class**

**Priority:** P0
**Duration:** 2 days
**Dependencies:** None

**Objectives:**
1. Define genotype interface
2. Implement serialization/deserialization
3. Create mutation/crossover methods
4. Add validation

**Deliverables:** (See AGENT_BASED_RESEARCH_ARCHITECTURE.md for full code)

```python
# File: genotypes/base.py
# (Implementation in architecture doc)

# File: genotypes/ma_crossover.py
# (Implementation in architecture doc)
```

**Tests:**

```python
# File: tests/test_genotype.py

import pytest
from genotypes.ma_crossover import MACrossoverGenotype

def test_genotype_creation():
    """Test creating a genotype."""
    g = MACrossoverGenotype(
        id='test-1',
        generation=1,
        parent_ids=[],
        parameters={'ma_period': 44},
        created_at='2026-04-02T00:00:00'
    )

    assert g.id == 'test-1'
    assert g.generation == 1
    assert g.parameters['ma_period'] == 44

def test_genotype_mutation():
    """Test mutation creates different genotype."""
    g1 = MACrossoverGenotype(
        id='test-1',
        generation=1,
        parent_ids=[],
        parameters={'ma_period': 44},
        created_at='2026-04-02T00:00:00'
    )

    g2 = g1.mutate(mutation_rate=1.0)  # Force mutation

    assert g2.id != g1.id
    assert g2.generation == 2
    assert g2.parent_ids == [g1.id]
    # Parameters should be different
    assert g2.parameters['ma_period'] != g1.parameters['ma_period']

def test_genotype_crossover():
    """Test crossover combines parents."""
    g1 = MACrossoverGenotype(
        id='parent-1',
        generation=1,
        parent_ids=[],
        parameters={'ma_period': 30},
        created_at='2026-04-02T00:00:00'
    )

    g2 = MACrossoverGenotype(
        id='parent-2',
        generation=1,
        parent_ids=[],
        parameters={'ma_period': 60},
        created_at='2026-04-02T00:00:00'
    )

    child = MACrossoverGenotype.crossover(g1, g2)

    assert child.generation == 2
    assert set(child.parent_ids) == {'parent-1', 'parent-2'}
    # Child should have parameters in parent range
    assert 30 <= child.parameters['ma_period'] <= 60
```

**Success Criteria:**
- [ ] Genotype class implemented
- [ ] Mutation works (creates variations)
- [ ] Crossover works (combines parents)
- [ ] Serialization/deserialization works
- [ ] All tests passing

---

### Summary of Phase 1 Tasks

| Task | Duration | Priority | Dependencies | Deliverable |
|------|----------|----------|--------------|-------------|
| T1: Database Schema | 2 days | P0 | None | PostgreSQL schema + access layer |
| T2: Task Queue | 1 day | P0 | None | Redis + Celery working |
| T3: Daily Data Pipeline | 2 days | P1 | Tick data | Daily OHLCV + features |
| T4: Base Genotype | 2 days | P0 | None | Genotype class with mutation/crossover |

**Total Phase 1:** 7 days (1.5 weeks)

**End State:** Foundation ready for implementing agents.

---

## Would You Like Me To Continue with Phase 2-4 Tasks?

I can create detailed task breakdowns for:
- **Phase 2:** Agent implementation (Evaluator, Generator, Evolver)
- **Phase 3:** Dashboard and real-time monitoring
- **Phase 4:** Evolution engine and advanced features

Or would you prefer I **start implementing Phase 1** right now?

Each task includes:
- Objectives
- Complete code
- Tests
- Success criteria
- Dependencies

This ensures systematic, testable progress toward the autonomous research system.
