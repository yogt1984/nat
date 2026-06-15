# Statistical Dashboard Architecture Specification

**Status:** Specification
**Created:** 2026-04-05
**Purpose:** Define architecture for statistical analysis publication website
**Principle:** Compute once, publish continuously, consume by many

---

## 1. Executive Summary

### 1.1 Purpose

Build a website that:
1. **Continuously computes** statistical characteristics of market data
2. **Publishes** these statistics via API and visual dashboard
3. **Serves as reference** for autonomous agents and human analysts
4. **Creates audit trail** of market conditions over time

### 1.2 Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Single Source of Truth** | All agents and humans reference the same published statistics |
| **Immutable Snapshots** | Once published, snapshots never change (versioned) |
| **Freshness Guarantees** | Every statistic has TTL; stale data is rejected |
| **Separation of Concerns** | Compute backend ≠ API layer ≠ Frontend ≠ Agent consumers |
| **Progressive Disclosure** | Simple metrics upfront, details on demand |

### 1.3 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STATISTICAL DASHBOARD SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────────┐ │
│  │   NAT Data   │───▶│  Statistical     │───▶│  Publication Layer         │ │
│  │   Ingestion  │    │  Compute Engine  │    │  (API + Database)          │ │
│  │  (Existing)  │    │                  │    │                            │ │
│  └──────────────┘    └──────────────────┘    └────────────────────────────┘ │
│         │                    │                          │                    │
│         │                    │                          │                    │
│         ▼                    ▼                          ▼                    │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────────┐ │
│  │   Parquet    │    │  Snapshot Store  │    │  Consumers                 │ │
│  │   Files      │    │  (TimescaleDB)   │    │  ├─ Web Dashboard          │ │
│  │   (Raw)      │    │                  │    │  ├─ Trading Agents         │ │
│  └──────────────┘    └──────────────────┘    │  ├─ Alert System           │ │
│                                              │  └─ External APIs          │ │
│                                              └────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Architecture

### 2.1 Data Flow

```
                    WRITE PATH                          READ PATH

NAT Parquet Files                              Web Dashboard
      │                                              ▲
      ▼                                              │
┌─────────────┐                              ┌───────┴───────┐
│  Compute    │                              │   REST API    │
│  Scheduler  │                              │   /api/v1/*   │
└─────────────┘                              └───────────────┘
      │                                              ▲
      │  Triggers                                    │
      ▼                                              │
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Realtime   │  │   Hourly    │  │      Daily          │  │
│  │  Computer   │  │  Computer   │  │     Computer        │  │
│  │  (1-5 min)  │  │   (1 hr)    │  │     (24 hr)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │               │                    │              │
│         └───────────────┴────────────────────┘              │
│                         │                                    │
│                         ▼                                    │
│              ┌─────────────────────┐                        │
│              │   Snapshot Writer   │                        │
│              │   (Atomic Commits)  │                        │
│              └─────────────────────┘                        │
│                         │                                    │
│                         ▼                                    │
│              ┌─────────────────────┐                        │
│              │    TimescaleDB      │◀───────────────────────┤
│              │  (Immutable Store)  │                        │
│              └─────────────────────┘                        │
│                                                              │
│                    COMPUTE ENGINE                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Snapshot Types

| Snapshot Type | Update Frequency | TTL | Primary Use |
|--------------|------------------|-----|-------------|
| **Realtime** | 1-5 minutes | 5 minutes | Current market state, agent decisions |
| **Hourly** | 60 minutes | 2 hours | Cluster stats, correlations |
| **Daily** | 24 hours | 48 hours | Distribution analysis, feature selection |
| **Weekly** | 7 days | 14 days | Deep analysis, model retraining triggers |

### 2.3 Database Schema

```sql
-- PostgreSQL with TimescaleDB extension

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- CORE TABLES
-- ============================================================

-- Snapshot metadata (parent for all snapshot types)
CREATE TABLE snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_type VARCHAR(20) NOT NULL,  -- 'realtime', 'hourly', 'daily', 'weekly'
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ NOT NULL,
    data_start TIMESTAMPTZ NOT NULL,
    data_end TIMESTAMPTZ NOT NULL,
    computation_time_ms INTEGER,
    version VARCHAR(20) NOT NULL DEFAULT '1.0',
    status VARCHAR(20) NOT NULL DEFAULT 'active',  -- 'active', 'superseded', 'invalid'

    CONSTRAINT valid_snapshot_type CHECK (snapshot_type IN ('realtime', 'hourly', 'daily', 'weekly'))
);

CREATE INDEX idx_snapshots_type_created ON snapshots(snapshot_type, created_at DESC);
CREATE INDEX idx_snapshots_valid_until ON snapshots(valid_until);

-- ============================================================
-- REALTIME SNAPSHOT TABLES
-- ============================================================

-- Current entropy state
CREATE TABLE realtime_entropy (
    snapshot_id UUID PRIMARY KEY REFERENCES snapshots(snapshot_id),

    -- Current values
    entropy_current FLOAT NOT NULL,
    entropy_percentile FLOAT NOT NULL,
    entropy_trend_5m FLOAT,  -- Slope over last 5 minutes
    entropy_trend_15m FLOAT,

    -- Regime probabilities (must sum to 1)
    regime_prob_trending FLOAT NOT NULL,
    regime_prob_ranging FLOAT NOT NULL,
    regime_prob_uncertain FLOAT NOT NULL,

    -- Cluster assignment (if applicable)
    cluster_id INTEGER,
    cluster_confidence FLOAT,

    CONSTRAINT entropy_range CHECK (entropy_current >= 0 AND entropy_current <= 1),
    CONSTRAINT regime_probs_sum CHECK (
        ABS(regime_prob_trending + regime_prob_ranging + regime_prob_uncertain - 1.0) < 0.01
    )
);

-- Current volatility state
CREATE TABLE realtime_volatility (
    snapshot_id UUID PRIMARY KEY REFERENCES snapshots(snapshot_id),

    realized_vol_5m FLOAT NOT NULL,
    realized_vol_15m FLOAT NOT NULL,
    realized_vol_1h FLOAT NOT NULL,
    realized_vol_4h FLOAT,

    vol_percentile FLOAT NOT NULL,
    vol_regime VARCHAR(20),  -- 'low', 'normal', 'high', 'extreme'

    parkinson_vol FLOAT,
    garman_klass_vol FLOAT,

    CONSTRAINT vol_regime_valid CHECK (vol_regime IN ('low', 'normal', 'high', 'extreme'))
);

-- Current market state
CREATE TABLE realtime_market_state (
    snapshot_id UUID PRIMARY KEY REFERENCES snapshots(snapshot_id),

    -- Price data
    price_current FLOAT NOT NULL,
    price_change_1h FLOAT,
    price_change_4h FLOAT,
    price_change_24h FLOAT,

    -- Whale flow
    whale_flow_1h FLOAT,
    whale_flow_4h FLOAT,
    whale_flow_24h FLOAT,
    whale_flow_direction VARCHAR(10),  -- 'buying', 'selling', 'neutral'

    -- Liquidation
    liq_risk_long FLOAT,
    liq_risk_short FLOAT,
    liq_asymmetry FLOAT,
    liq_cluster_nearest_pct FLOAT,  -- Distance to nearest liquidation cluster

    -- Funding & basis
    funding_rate FLOAT,
    funding_percentile FLOAT,
    basis_annualized FLOAT,

    -- Order book
    spread_bps FLOAT,
    depth_imbalance FLOAT,
    microprice FLOAT
);

-- Current feature values (top N features only)
CREATE TABLE realtime_features (
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    feature_name VARCHAR(100) NOT NULL,
    feature_value FLOAT NOT NULL,
    feature_percentile FLOAT,
    feature_zscore FLOAT,

    PRIMARY KEY (snapshot_id, feature_name)
);

-- ============================================================
-- HOURLY SNAPSHOT TABLES
-- ============================================================

-- Cluster statistics
CREATE TABLE hourly_cluster_stats (
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    cluster_id INTEGER NOT NULL,

    -- Cluster definition
    entropy_min FLOAT NOT NULL,
    entropy_max FLOAT NOT NULL,
    entropy_mean FLOAT NOT NULL,

    -- Observations
    n_observations INTEGER NOT NULL,
    pct_of_total FLOAT NOT NULL,

    -- Volatility within cluster
    vol_mean FLOAT,
    vol_std FLOAT,
    vol_percentile_25 FLOAT,
    vol_percentile_75 FLOAT,

    -- Returns within cluster
    return_mean FLOAT,
    return_std FLOAT,
    return_skew FLOAT,
    return_kurtosis FLOAT,
    sharpe_ratio FLOAT,

    -- Trend characteristics
    momentum_persistence FLOAT,
    sign_persistence FLOAT,

    PRIMARY KEY (snapshot_id, cluster_id)
);

-- Feature correlations (stored as matrix)
CREATE TABLE hourly_correlations (
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    feature_1 VARCHAR(100) NOT NULL,
    feature_2 VARCHAR(100) NOT NULL,
    pearson_corr FLOAT NOT NULL,
    spearman_corr FLOAT,

    PRIMARY KEY (snapshot_id, feature_1, feature_2)
);

-- Transition counts (last 24h)
CREATE TABLE hourly_transitions (
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    from_cluster INTEGER NOT NULL,
    to_cluster INTEGER NOT NULL,
    transition_count INTEGER NOT NULL,
    transition_probability FLOAT NOT NULL,
    avg_return_after FLOAT,

    PRIMARY KEY (snapshot_id, from_cluster, to_cluster)
);

-- ============================================================
-- DAILY SNAPSHOT TABLES
-- ============================================================

-- Entropy distribution analysis
CREATE TABLE daily_entropy_distribution (
    snapshot_id UUID PRIMARY KEY REFERENCES snapshots(snapshot_id),

    -- Basic statistics
    mean FLOAT NOT NULL,
    median FLOAT NOT NULL,
    std FLOAT NOT NULL,
    skewness FLOAT,
    kurtosis FLOAT,

    -- Percentiles (stored as JSON array)
    percentiles JSONB NOT NULL,  -- {5: 0.2, 10: 0.25, 25: 0.35, ...}

    -- Clustering results
    optimal_n_clusters INTEGER NOT NULL,
    cluster_means JSONB NOT NULL,  -- [0.35, 0.55, 0.75]
    cluster_weights JSONB NOT NULL,  -- [0.3, 0.4, 0.3]
    cluster_stds JSONB,

    -- Modality tests
    is_multimodal BOOLEAN NOT NULL,
    dip_test_pvalue FLOAT,

    -- Temporal properties
    acf_half_life INTEGER,
    hurst_exponent FLOAT,
    is_stationary BOOLEAN,
    adf_pvalue FLOAT
);

-- Feature predictiveness rankings
CREATE TABLE daily_feature_rankings (
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    feature_rank INTEGER NOT NULL,
    feature_name VARCHAR(100) NOT NULL,

    -- Predictive metrics
    correlation_1p FLOAT,
    correlation_5p FLOAT,
    correlation_20p FLOAT,
    mutual_info FLOAT,

    -- Causal validation
    granger_pvalue FLOAT,
    feature_leads_returns BOOLEAN,
    oos_is_ratio FLOAT,
    is_causal BOOLEAN,

    -- Stability
    bootstrap_selection_rate FLOAT,
    is_stable BOOLEAN,

    PRIMARY KEY (snapshot_id, feature_rank)
);

-- Selected features (final set)
CREATE TABLE daily_selected_features (
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    feature_name VARCHAR(100) NOT NULL,
    selection_reason VARCHAR(255),
    importance_score FLOAT,

    PRIMARY KEY (snapshot_id, feature_name)
);

-- Recommendations
CREATE TABLE daily_recommendations (
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    recommendation_id SERIAL,
    recommendation_type VARCHAR(20) NOT NULL,  -- 'INFO', 'WARNING', 'ACTION'
    message TEXT NOT NULL,
    action_suggested TEXT,

    PRIMARY KEY (snapshot_id, recommendation_id)
);

-- ============================================================
-- AGENT CONSUMPTION TRACKING
-- ============================================================

-- Track which snapshots agents consumed
CREATE TABLE agent_snapshot_consumption (
    consumption_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    snapshot_id UUID REFERENCES snapshots(snapshot_id),
    consumed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decision_made VARCHAR(50),  -- 'TRADE_LONG', 'TRADE_SHORT', 'NO_TRADE'
    decision_reason TEXT
);

CREATE INDEX idx_consumption_agent ON agent_snapshot_consumption(agent_id, consumed_at DESC);
CREATE INDEX idx_consumption_snapshot ON agent_snapshot_consumption(snapshot_id);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('agent_snapshot_consumption', 'consumed_at');

-- ============================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================

-- Latest realtime snapshot
CREATE VIEW latest_realtime AS
SELECT
    s.*,
    e.*,
    v.*,
    m.*
FROM snapshots s
JOIN realtime_entropy e ON s.snapshot_id = e.snapshot_id
JOIN realtime_volatility v ON s.snapshot_id = v.snapshot_id
JOIN realtime_market_state m ON s.snapshot_id = m.snapshot_id
WHERE s.snapshot_type = 'realtime'
  AND s.status = 'active'
  AND s.valid_until > NOW()
ORDER BY s.created_at DESC
LIMIT 1;

-- Latest daily analysis
CREATE VIEW latest_daily AS
SELECT
    s.*,
    d.*
FROM snapshots s
JOIN daily_entropy_distribution d ON s.snapshot_id = d.snapshot_id
WHERE s.snapshot_type = 'daily'
  AND s.status = 'active'
ORDER BY s.created_at DESC
LIMIT 1;

-- Feature rankings with causal validation
CREATE VIEW causal_features AS
SELECT
    fr.*,
    s.created_at as analysis_date
FROM daily_feature_rankings fr
JOIN snapshots s ON fr.snapshot_id = s.snapshot_id
WHERE fr.is_causal = TRUE
  AND fr.is_stable = TRUE
  AND s.status = 'active'
ORDER BY s.created_at DESC, fr.feature_rank ASC;
```

---

## 3. API Specification

### 3.1 REST API Design

**Base URL:** `https://stats.nat.trading/api/v1`

**Authentication:** API key in header (`X-API-Key: <key>`)

**Rate Limits:**
- Public: 60 requests/minute
- Authenticated: 600 requests/minute
- Agents: 6000 requests/minute (special tier)

### 3.2 Endpoints

#### 3.2.1 Snapshot Endpoints

```yaml
# Get latest realtime snapshot
GET /snapshot/realtime
Response:
  snapshot_id: uuid
  created_at: timestamp
  valid_until: timestamp
  ttl_seconds: integer

  entropy:
    current: float
    percentile: float
    trend_5m: float
    regime_probabilities:
      trending: float
      ranging: float
      uncertain: float
    cluster:
      id: integer
      confidence: float

  volatility:
    realized_5m: float
    realized_1h: float
    percentile: float
    regime: string

  market_state:
    price: float
    whale_flow:
      1h: float
      4h: float
      direction: string
    liquidation:
      risk_long: float
      risk_short: float
      asymmetry: float
    funding_rate: float
    spread_bps: float

  features:
    - name: string
      value: float
      percentile: float

# Get latest hourly snapshot
GET /snapshot/hourly
Response:
  snapshot_id: uuid
  created_at: timestamp
  valid_until: timestamp

  clusters:
    - cluster_id: integer
      entropy_range: [float, float]
      n_observations: integer
      volatility:
        mean: float
        std: float
      returns:
        mean: float
        sharpe: float
      trend:
        momentum_persistence: float
        sign_persistence: float

  transitions:
    matrix: [[float]]
    expected_durations: [float]

  correlations:
    highly_correlated_pairs:
      - [feature1, feature2, correlation]

# Get latest daily snapshot
GET /snapshot/daily
Response:
  snapshot_id: uuid
  created_at: timestamp
  valid_until: timestamp

  entropy_distribution:
    mean: float
    std: float
    percentiles: {5: float, 10: float, ...}
    clusters:
      optimal_n: integer
      means: [float]
      weights: [float]
    is_multimodal: boolean
    temporal:
      half_life: integer
      hurst: float
      is_stationary: boolean

  feature_rankings:
    - rank: integer
      name: string
      correlation: float
      mutual_info: float
      is_causal: boolean
      is_stable: boolean

  selected_features:
    - name: string
      reason: string

  recommendations:
    - type: string
      message: string
      action: string

# Get specific snapshot by ID
GET /snapshot/{snapshot_id}
Response: Full snapshot data based on type
```

#### 3.2.2 Historical Data Endpoints

```yaml
# Get historical snapshots
GET /history/snapshots
Parameters:
  type: string (required) - 'realtime', 'hourly', 'daily'
  start: timestamp (required)
  end: timestamp (required)
  limit: integer (default: 100, max: 1000)
Response:
  snapshots:
    - snapshot_id: uuid
      created_at: timestamp
      summary: object (type-specific summary)

# Get historical metric series
GET /history/metric
Parameters:
  metric: string (required) - e.g., 'entropy.current', 'volatility.realized_5m'
  start: timestamp (required)
  end: timestamp (required)
  resolution: string - '1m', '5m', '1h', '1d'
Response:
  metric: string
  resolution: string
  data:
    - timestamp: timestamp
      value: float

# Get historical cluster assignments
GET /history/clusters
Parameters:
  start: timestamp (required)
  end: timestamp (required)
Response:
  data:
    - timestamp: timestamp
      cluster_id: integer
      entropy: float
      confidence: float
```

#### 3.2.3 Analysis Endpoints

```yaml
# Get current regime analysis
GET /analysis/regime
Response:
  current_regime: string
  confidence: float
  duration_in_regime: integer (periods)
  expected_remaining_duration: float
  transition_probabilities:
    to_trending: float
    to_ranging: float
    to_uncertain: float
  historical_performance:
    regime: string
    avg_return: float
    sharpe: float

# Get feature importance for current regime
GET /analysis/features
Parameters:
  regime: string (optional) - filter by regime
Response:
  regime: string
  features:
    - name: string
      importance: float
      current_value: float
      signal: string  # 'bullish', 'bearish', 'neutral'

# Get predictive power summary
GET /analysis/predictive
Response:
  overall_predictability: float  # 0-1 score
  best_features:
    - name: string
      correlation: float
      is_causal: boolean
  regime_conditional:
    trending:
      predictability: float
      best_feature: string
    ranging:
      predictability: float
      best_feature: string
  recommendation: string
```

#### 3.2.4 Agent Endpoints

```yaml
# Register agent consumption (for audit trail)
POST /agent/consume
Body:
  agent_id: string (required)
  snapshot_id: uuid (required)
  decision: string
  reason: string
Response:
  consumption_id: uuid
  recorded_at: timestamp

# Get agent decision history
GET /agent/{agent_id}/history
Parameters:
  start: timestamp
  end: timestamp
  limit: integer
Response:
  decisions:
    - consumption_id: uuid
      snapshot_id: uuid
      decision: string
      reason: string
      timestamp: timestamp
      snapshot_summary: object

# Validate snapshot freshness
GET /agent/validate/{snapshot_id}
Response:
  is_valid: boolean
  is_fresh: boolean
  ttl_remaining_seconds: integer
  superseded_by: uuid (if applicable)
```

### 3.3 WebSocket API

For real-time updates without polling:

```yaml
# Connect to WebSocket
WS /ws/v1/stream

# Subscribe to channels
Send: {"action": "subscribe", "channels": ["realtime", "regime_change", "alerts"]}

# Receive updates
Receive (realtime):
  {
    "channel": "realtime",
    "snapshot_id": "uuid",
    "timestamp": "2026-04-05T10:30:00Z",
    "data": { ... realtime snapshot ... }
  }

Receive (regime_change):
  {
    "channel": "regime_change",
    "timestamp": "2026-04-05T10:30:00Z",
    "previous_regime": "ranging",
    "new_regime": "trending",
    "confidence": 0.75
  }

Receive (alerts):
  {
    "channel": "alerts",
    "timestamp": "2026-04-05T10:30:00Z",
    "alert_type": "volatility_spike",
    "message": "Volatility jumped to 95th percentile",
    "severity": "warning"
  }
```

---

## 4. Compute Engine Specification

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       COMPUTE ENGINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    SCHEDULER                             │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │    │
│  │  │Realtime │  │ Hourly  │  │  Daily  │  │ Weekly  │    │    │
│  │  │ (cron)  │  │ (cron)  │  │ (cron)  │  │ (cron)  │    │    │
│  │  │ */5 *   │  │ 0 *     │  │ 0 0     │  │ 0 0 * 0 │    │    │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │    │
│  └───────┼────────────┼────────────┼────────────┼──────────┘    │
│          │            │            │            │                │
│          ▼            ▼            ▼            ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    TASK QUEUE (Redis)                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│          │            │            │            │                │
│          ▼            ▼            ▼            ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    WORKER POOL                           │    │
│  │                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │    │
│  │  │   Worker 1   │  │   Worker 2   │  │   Worker N   │   │    │
│  │  │              │  │              │  │              │   │    │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │   │    │
│  │  │ │ Entropy  │ │  │ │ Cluster  │ │  │ │ Feature  │ │   │    │
│  │  │ │ Computer │ │  │ │ Computer │ │  │ │ Computer │ │   │    │
│  │  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │   │    │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │   │    │
│  │  │ │   Vol    │ │  │ │ Predict  │ │  │ │  Causal  │ │   │    │
│  │  │ │ Computer │ │  │ │ Computer │ │  │ │ Computer │ │   │    │
│  │  │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│          │                                                       │
│          ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  SNAPSHOT ASSEMBLER                      │    │
│  │  - Collects results from all computers                   │    │
│  │  - Validates completeness                                │    │
│  │  - Atomic write to database                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Computer Modules

```python
# computers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class BaseComputer(ABC):
    """Base class for all statistical computers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute statistics from data

        Args:
            data: DataFrame with raw feature data

        Returns:
            Dictionary of computed statistics
        """
        pass

    @abstractmethod
    def validate_output(self, result: Dict[str, Any]) -> bool:
        """Validate computed results before storage"""
        pass

    def get_required_columns(self) -> list:
        """Return list of required input columns"""
        return []


# computers/entropy.py
class EntropyComputer(BaseComputer):
    """Compute entropy-related statistics"""

    def get_required_columns(self) -> list:
        return ['normalized_entropy_15m', 'returns']

    def compute(self, data: pd.DataFrame) -> Dict[str, Any]:
        entropy = data['normalized_entropy_15m'].dropna()

        result = {
            'current': {
                'value': entropy.iloc[-1],
                'percentile': (entropy < entropy.iloc[-1]).mean(),
                'trend_5m': self._compute_trend(entropy, periods=5),
                'trend_15m': self._compute_trend(entropy, periods=15)
            },
            'distribution': {
                'mean': entropy.mean(),
                'std': entropy.std(),
                'percentiles': {
                    str(p): entropy.quantile(p/100)
                    for p in [5, 10, 25, 50, 75, 90, 95]
                }
            },
            'regime_probabilities': self._compute_regime_probs(entropy),
            'cluster': self._compute_cluster_assignment(entropy)
        }

        return result

    def _compute_regime_probs(self, entropy: pd.Series) -> Dict[str, float]:
        """Compute regime probabilities using fitted GMM"""
        # Load pre-fitted GMM model
        gmm = self._load_gmm_model()

        current = entropy.iloc[-1]
        probs = gmm.predict_proba([[current]])[0]

        # Map GMM components to regime names
        # (assumes components ordered by mean: low, medium, high)
        return {
            'trending': probs[0],      # Low entropy
            'uncertain': probs[1],     # Medium entropy
            'ranging': probs[2]        # High entropy
        }

    def _compute_cluster_assignment(self, entropy: pd.Series) -> Dict[str, Any]:
        """Assign current state to cluster with confidence"""
        gmm = self._load_gmm_model()

        current = entropy.iloc[-1]
        cluster_id = gmm.predict([[current]])[0]
        probs = gmm.predict_proba([[current]])[0]
        confidence = probs[cluster_id]

        return {
            'id': int(cluster_id),
            'confidence': float(confidence)
        }

    def validate_output(self, result: Dict[str, Any]) -> bool:
        # Check entropy is in valid range
        if not 0 <= result['current']['value'] <= 1:
            return False

        # Check probabilities sum to 1
        probs = result['regime_probabilities']
        prob_sum = probs['trending'] + probs['uncertain'] + probs['ranging']
        if abs(prob_sum - 1.0) > 0.01:
            return False

        return True


# computers/volatility.py
class VolatilityComputer(BaseComputer):
    """Compute volatility-related statistics"""

    def get_required_columns(self) -> list:
        return ['returns', 'high', 'low', 'close', 'open']

    def compute(self, data: pd.DataFrame) -> Dict[str, Any]:
        returns = data['returns'].dropna()

        result = {
            'realized': {
                '5m': self._realized_vol(returns, periods=5),
                '15m': self._realized_vol(returns, periods=15),
                '1h': self._realized_vol(returns, periods=60),
                '4h': self._realized_vol(returns, periods=240)
            },
            'parkinson': self._parkinson_vol(data),
            'garman_klass': self._garman_klass_vol(data),
            'percentile': self._vol_percentile(returns),
            'regime': self._classify_vol_regime(returns)
        }

        return result

    def _realized_vol(self, returns: pd.Series, periods: int) -> float:
        """Annualized realized volatility"""
        recent = returns.iloc[-periods:]
        return float(recent.std() * np.sqrt(252 * 24 * 4))  # Assuming 15m data

    def _parkinson_vol(self, data: pd.DataFrame, periods: int = 60) -> float:
        """Parkinson volatility estimator (uses high-low)"""
        recent = data.iloc[-periods:]
        log_hl = np.log(recent['high'] / recent['low'])
        return float(np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2).mean()) * np.sqrt(252 * 24 * 4))

    def _classify_vol_regime(self, returns: pd.Series) -> str:
        """Classify current volatility regime"""
        current_vol = returns.iloc[-60:].std()
        historical_vol = returns.std()

        percentile = (returns.rolling(60).std() < current_vol).mean()

        if percentile < 0.25:
            return 'low'
        elif percentile < 0.75:
            return 'normal'
        elif percentile < 0.95:
            return 'high'
        else:
            return 'extreme'


# computers/predictive.py
class PredictiveComputer(BaseComputer):
    """Compute feature predictiveness statistics"""

    def compute(self, data: pd.DataFrame) -> Dict[str, Any]:
        returns = data['returns']
        features = [c for c in data.columns
                   if c not in ['returns', 'timestamp', 'close', 'open', 'high', 'low']]

        rankings = []

        for feature in features:
            feature_data = data[feature].dropna()
            aligned_returns = returns.loc[feature_data.index].shift(-1)

            result = {
                'name': feature,
                'correlation_1p': feature_data.corr(aligned_returns),
                'correlation_5p': feature_data.corr(returns.shift(-5)),
                'mutual_info': self._mutual_info(feature_data, aligned_returns),
                'is_causal': self._test_causality(feature_data, returns),
                'is_stable': self._test_stability(feature_data, aligned_returns)
            }
            rankings.append(result)

        # Sort by absolute correlation
        rankings.sort(key=lambda x: abs(x['correlation_1p'] or 0), reverse=True)

        # Add rank
        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return {
            'rankings': rankings[:20],  # Top 20
            'selected_features': [r['name'] for r in rankings
                                 if r['is_causal'] and r['is_stable']][:10]
        }
```

### 4.3 Snapshot Assembly

```python
# assembler.py
from typing import Dict, Any, List
import uuid
from datetime import datetime, timedelta

class SnapshotAssembler:
    """
    Assembles computed statistics into immutable snapshots
    """

    def __init__(self, db_connection, computers: List[BaseComputer]):
        self.db = db_connection
        self.computers = computers

    def create_realtime_snapshot(self, data: pd.DataFrame) -> str:
        """
        Create and store a realtime snapshot

        Returns: snapshot_id
        """
        snapshot_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        valid_until = created_at + timedelta(minutes=5)

        # Run all computers
        results = {}
        for computer in self.computers:
            if computer.name in ['EntropyComputer', 'VolatilityComputer', 'MarketStateComputer']:
                result = computer.compute(data)

                if not computer.validate_output(result):
                    raise ValueError(f"Validation failed for {computer.name}")

                results[computer.name] = result

        # Atomic write to database
        with self.db.transaction():
            # Insert snapshot metadata
            self.db.execute("""
                INSERT INTO snapshots (snapshot_id, snapshot_type, created_at,
                                       valid_until, data_start, data_end, status)
                VALUES (%s, 'realtime', %s, %s, %s, %s, 'active')
            """, (snapshot_id, created_at, valid_until,
                  data.index.min(), data.index.max()))

            # Mark previous realtime snapshots as superseded
            self.db.execute("""
                UPDATE snapshots
                SET status = 'superseded'
                WHERE snapshot_type = 'realtime'
                  AND status = 'active'
                  AND snapshot_id != %s
            """, (snapshot_id,))

            # Insert entropy data
            self._insert_entropy(snapshot_id, results['EntropyComputer'])

            # Insert volatility data
            self._insert_volatility(snapshot_id, results['VolatilityComputer'])

            # Insert market state
            self._insert_market_state(snapshot_id, results['MarketStateComputer'])

        return snapshot_id

    def _insert_entropy(self, snapshot_id: str, data: Dict[str, Any]):
        """Insert entropy data into realtime_entropy table"""
        self.db.execute("""
            INSERT INTO realtime_entropy (
                snapshot_id, entropy_current, entropy_percentile,
                entropy_trend_5m, entropy_trend_15m,
                regime_prob_trending, regime_prob_ranging, regime_prob_uncertain,
                cluster_id, cluster_confidence
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            snapshot_id,
            data['current']['value'],
            data['current']['percentile'],
            data['current']['trend_5m'],
            data['current']['trend_15m'],
            data['regime_probabilities']['trending'],
            data['regime_probabilities']['ranging'],
            data['regime_probabilities']['uncertain'],
            data['cluster']['id'],
            data['cluster']['confidence']
        ))
```

---

## 5. Frontend Specification

### 5.1 Page Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  NAT Statistical Dashboard                    [API Docs] [Login]│
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  CURRENT MARKET STATE                    Updated: 10:30:05  ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │                                                              ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ ││
│  │  │   ENTROPY    │ │  VOLATILITY  │ │      REGIME          │ ││
│  │  │              │ │              │ │                      │ ││
│  │  │    0.47      │ │    1.82%     │ │    UNCERTAIN         │ ││
│  │  │  52nd %ile   │ │  HIGH (85%)  │ │  (confidence: 42%)   │ ││
│  │  │              │ │              │ │                      │ ││
│  │  │   ▼ -0.02    │ │   ▲ +0.3%    │ │  P(trend): 31%       │ ││
│  │  │   (falling)  │ │   (rising)   │ │  P(range): 27%       │ ││
│  │  └──────────────┘ └──────────────┘ └──────────────────────┘ ││
│  │                                                              ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ ││
│  │  │  WHALE FLOW  │ │  LIQ RISK    │ │     FUNDING          │ ││
│  │  │              │ │              │ │                      │ ││
│  │  │   +$2.3M     │ │   MODERATE   │ │     0.012%           │ ││
│  │  │  (buying)    │ │  asym: 0.15  │ │   65th %ile          │ ││
│  │  └──────────────┘ └──────────────┘ └──────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  ENTROPY DISTRIBUTION                                        ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │                                                              ││
│  │         ▄▄▄                                                  ││
│  │       ▄█████▄                                                ││
│  │      ▄████████                 Current: ●                    ││
│  │    ▄████████████▄                                            ││
│  │  ▄██████████████████▄                                        ││
│  │ ▄████████████████████████▄                                   ││
│  │ └────────────────────────────────────────────────────────    ││
│  │  0.0       0.25       0.50       0.75       1.0              ││
│  │                                                              ││
│  │  Clusters: [0.38] [0.52] [0.71]                             ││
│  │  Multimodal: Yes (dip test p=0.02)                          ││
│  │  Half-life: 12 periods (~3 hours)                           ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌────────────────────────────┐ ┌──────────────────────────────┐│
│  │  TOP PREDICTIVE FEATURES   │ │  RECOMMENDATIONS             ││
│  │  ────────────────────────  │ │  ────────────────────────    ││
│  │                            │ │                              ││
│  │  1. whale_net_flow_4h     │ │  ⚠ Entropy in transition     ││
│  │     corr: 0.08 ✓ causal   │ │    zone. Wait for clarity.   ││
│  │                            │ │                              ││
│  │  2. liq_asymmetry         │ │  ℹ Volatility elevated.      ││
│  │     corr: 0.06 ✓ causal   │ │    Reduce position size.     ││
│  │                            │ │                              ││
│  │  3. funding_rate          │ │  ✓ Top features pass         ││
│  │     corr: 0.05 ✗ unstable │ │    causal tests.             ││
│  │                            │ │                              ││
│  │  4. momentum_300          │ │                              ││
│  │     corr: 0.04 ✗ spurious │ │                              ││
│  │                            │ │                              ││
│  └────────────────────────────┘ └──────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  CLUSTER PERFORMANCE                                         ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │                                                              ││
│  │  Cluster │ Entropy  │ % Data │ Sharpe │ Momentum │ Strategy ││
│  │  ────────┼──────────┼────────┼────────┼──────────┼───────── ││
│  │     0    │ 0.25-0.42│   28%  │  0.82  │   0.65   │ TREND    ││
│  │     1    │ 0.42-0.62│   45%  │  0.12  │   0.48   │ WAIT     ││
│  │     2    │ 0.62-0.85│   27%  │  0.45  │   0.31   │ FADE     ││
│  │                                                              ││
│  │  Current cluster: 1 (entropy=0.47)                          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  TRANSITION MATRIX                     Last 24h transitions  ││
│  │  ─────────────────────────────────────────────────────────  ││
│  │                                                              ││
│  │           To:    0      1      2                             ││
│  │        ┌──────┬──────┬──────┬──────┐                         ││
│  │  From: │  0   │ 0.72 │ 0.23 │ 0.05 │  Expected duration: 3.6h││
│  │        ├──────┼──────┼──────┼──────┤                         ││
│  │        │  1   │ 0.18 │ 0.64 │ 0.18 │  Expected duration: 2.8h││
│  │        ├──────┼──────┼──────┼──────┤                         ││
│  │        │  2   │ 0.08 │ 0.27 │ 0.65 │  Expected duration: 2.9h││
│  │        └──────┴──────┴──────┴──────┘                         ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  [Historical Data] [API Documentation] [Download Report]        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Frontend Framework** | React + TypeScript | Type safety, component reuse |
| **State Management** | React Query | Server state, caching, auto-refresh |
| **Charting** | D3.js + Recharts | Custom visualizations + standard charts |
| **Styling** | Tailwind CSS | Rapid prototyping, consistent design |
| **Real-time** | WebSocket | Live updates without polling |
| **API Client** | Auto-generated from OpenAPI | Type-safe API calls |

### 5.3 Key Components

```typescript
// components/CurrentState.tsx
interface CurrentStateProps {
  snapshot: RealtimeSnapshot;
}

export const CurrentState: React.FC<CurrentStateProps> = ({ snapshot }) => {
  const ttlSeconds = Math.max(0,
    (new Date(snapshot.valid_until).getTime() - Date.now()) / 1000
  );

  return (
    <div className="grid grid-cols-3 gap-4">
      <MetricCard
        title="Entropy"
        value={snapshot.entropy.current.toFixed(2)}
        subtitle={`${(snapshot.entropy.percentile * 100).toFixed(0)}th percentile`}
        trend={snapshot.entropy.trend_5m}
        trendLabel={snapshot.entropy.trend_5m < 0 ? 'falling' : 'rising'}
      />

      <MetricCard
        title="Volatility"
        value={`${(snapshot.volatility.realized_5m * 100).toFixed(2)}%`}
        subtitle={`${snapshot.volatility.regime.toUpperCase()}`}
        badge={snapshot.volatility.regime === 'extreme' ? 'danger' : undefined}
      />

      <RegimeCard
        regime={getTopRegime(snapshot.entropy.regime_probabilities)}
        probabilities={snapshot.entropy.regime_probabilities}
        confidence={snapshot.entropy.cluster.confidence}
      />

      <FreshnessIndicator ttlSeconds={ttlSeconds} />
    </div>
  );
};

// components/EntropyDistribution.tsx
export const EntropyDistribution: React.FC<{
  distribution: DailyEntropyDistribution;
  currentValue: number;
}> = ({ distribution, currentValue }) => {
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Entropy Distribution</h3>

      {/* Histogram with KDE overlay */}
      <DistributionChart
        percentiles={distribution.percentiles}
        clusterMeans={distribution.clusters.means}
        currentValue={currentValue}
      />

      {/* Cluster markers */}
      <div className="flex justify-center gap-4 mt-4">
        {distribution.clusters.means.map((mean, i) => (
          <ClusterMarker
            key={i}
            clusterId={i}
            mean={mean}
            weight={distribution.clusters.weights[i]}
            isActive={isInCluster(currentValue, i, distribution)}
          />
        ))}
      </div>

      {/* Temporal properties */}
      <div className="grid grid-cols-3 gap-2 mt-4 text-sm text-gray-600">
        <div>
          <span className="font-medium">Half-life:</span>{' '}
          {distribution.temporal.half_life} periods
        </div>
        <div>
          <span className="font-medium">Hurst:</span>{' '}
          {distribution.temporal.hurst.toFixed(2)}
        </div>
        <div>
          <span className="font-medium">Stationary:</span>{' '}
          {distribution.temporal.is_stationary ? 'Yes' : 'No'}
        </div>
      </div>
    </div>
  );
};

// hooks/useRealtimeSnapshot.ts
export const useRealtimeSnapshot = () => {
  return useQuery({
    queryKey: ['snapshot', 'realtime'],
    queryFn: () => api.getRealtimeSnapshot(),
    refetchInterval: 60_000,  // Refetch every minute
    staleTime: 30_000,        // Consider stale after 30s
  });
};

// hooks/useWebSocket.ts
export const useSnapshotWebSocket = (
  onUpdate: (snapshot: RealtimeSnapshot) => void
) => {
  useEffect(() => {
    const ws = new WebSocket('wss://stats.nat.trading/ws/v1/stream');

    ws.onopen = () => {
      ws.send(JSON.stringify({
        action: 'subscribe',
        channels: ['realtime']
      }));
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.channel === 'realtime') {
        onUpdate(message.data);
      }
    };

    return () => ws.close();
  }, [onUpdate]);
};
```

---

## 6. Deployment Architecture

### 6.1 Infrastructure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         CLOUDFLARE                                   │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │     CDN     │  │     WAF     │  │   DDoS      │                  │    │
│  │  │  (static)   │  │  (security) │  │  Protection │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      KUBERNETES CLUSTER                              │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  INGRESS (nginx)                                              │   │    │
│  │  │  - SSL termination                                            │   │    │
│  │  │  - Rate limiting                                              │   │    │
│  │  │  - Request routing                                            │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │           │                    │                    │                │    │
│  │           ▼                    ▼                    ▼                │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │    │
│  │  │  Frontend    │    │   API        │    │   WebSocket          │   │    │
│  │  │  (3 pods)    │    │  (5 pods)    │    │   (3 pods)           │   │    │
│  │  │              │    │              │    │                      │   │    │
│  │  │  React SPA   │    │  FastAPI     │    │  FastAPI + WS        │   │    │
│  │  │  nginx       │    │  uvicorn     │    │  uvicorn             │   │    │
│  │  └──────────────┘    └──────────────┘    └──────────────────────┘   │    │
│  │                              │                    │                  │    │
│  │                              ▼                    ▼                  │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                        REDIS                                  │   │    │
│  │  │  - API response cache                                         │   │    │
│  │  │  - WebSocket pub/sub                                          │   │    │
│  │  │  - Rate limiting state                                        │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                              │                                       │    │
│  │                              ▼                                       │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                    TIMESCALEDB                                │   │    │
│  │  │  - Primary database                                           │   │    │
│  │  │  - 3-node HA cluster                                          │   │    │
│  │  │  - Continuous aggregates                                      │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                    COMPUTE ENGINE                             │   │    │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │   │    │
│  │  │  │ Scheduler  │  │  Workers   │  │   Assembler            │  │   │    │
│  │  │  │ (1 pod)    │  │  (5 pods)  │  │   (2 pods)             │  │   │    │
│  │  │  │            │  │            │  │                        │  │   │    │
│  │  │  │ Celery     │  │  Celery    │  │   Atomic writes        │  │   │    │
│  │  │  │ Beat       │  │  Workers   │  │   Validation           │  │   │    │
│  │  │  └────────────┘  └────────────┘  └────────────────────────┘  │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      NAT DATA SOURCE                                 │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │  Existing NAT infrastructure                                    │ │    │
│  │  │  - Hyperliquid WebSocket                                        │ │    │
│  │  │  - Feature computation                                          │ │    │
│  │  │  - Parquet storage                                              │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Resource Requirements

| Component | CPU | Memory | Storage | Replicas |
|-----------|-----|--------|---------|----------|
| Frontend | 0.2 | 256MB | - | 3 |
| API | 0.5 | 512MB | - | 5 |
| WebSocket | 0.3 | 256MB | - | 3 |
| Compute Workers | 2.0 | 4GB | - | 5 |
| Scheduler | 0.2 | 256MB | - | 1 |
| Assembler | 0.5 | 1GB | - | 2 |
| Redis | 1.0 | 2GB | 10GB | 3 (cluster) |
| TimescaleDB | 4.0 | 16GB | 500GB | 3 (HA) |

### 6.3 Monitoring

```yaml
# Prometheus metrics to track

# API metrics
api_request_duration_seconds{endpoint, method}
api_request_total{endpoint, method, status}
api_error_total{endpoint, error_type}

# Snapshot metrics
snapshot_creation_duration_seconds{type}
snapshot_creation_total{type, status}
snapshot_freshness_seconds{type}  # Time since last valid snapshot
snapshot_ttl_remaining_seconds{type}

# Compute metrics
compute_duration_seconds{computer}
compute_error_total{computer, error_type}
compute_queue_depth

# WebSocket metrics
websocket_connections_active
websocket_messages_sent_total{channel}
websocket_errors_total

# Database metrics
db_query_duration_seconds{query_type}
db_connections_active
db_replication_lag_seconds

# Alerts
- Snapshot staleness > 2x TTL
- API error rate > 1%
- Compute worker backlog > 10 tasks
- Database replication lag > 10s
- WebSocket connection drop rate > 5%
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goal:** Basic infrastructure + realtime statistics

| Task | Description | Output |
|------|-------------|--------|
| Database setup | Deploy TimescaleDB, create schema | Working database |
| Basic API | GET /snapshot/realtime endpoint | API returning mock data |
| Entropy computer | Compute basic entropy stats | Entropy values in DB |
| Volatility computer | Compute volatility stats | Volatility values in DB |
| Basic frontend | Display current state cards | Visible dashboard |

**Deliverable:** Dashboard showing live entropy + volatility

### Phase 2: Core Analytics (Week 3-4)

**Goal:** Cluster analysis + predictive features

| Task | Description | Output |
|------|-------------|--------|
| GMM fitting | Fit entropy distribution model | Cluster assignments |
| Cluster stats | Per-cluster volatility/returns | Cluster profiles |
| Feature rankings | Predictive power analysis | Feature leaderboard |
| Causal validation | Granger + lead-lag tests | Causal flags |
| Distribution charts | Entropy histogram + KDE | Visual distribution |

**Deliverable:** Dashboard with cluster analysis + feature rankings

### Phase 3: Agent Support (Week 5-6)

**Goal:** API for agent consumption

| Task | Description | Output |
|------|-------------|--------|
| Full API | All endpoints documented | OpenAPI spec |
| WebSocket | Real-time streaming | Live updates |
| Consumption logging | Track agent decisions | Audit trail |
| Snapshot validation | Freshness checks | Valid/invalid flags |
| Rate limiting | Prevent abuse | Throttled API |

**Deliverable:** Production-ready API for agents

### Phase 4: Polish (Week 7-8)

**Goal:** Reliability + UX

| Task | Description | Output |
|------|-------------|--------|
| Monitoring | Prometheus + Grafana | Dashboards |
| Alerting | PagerDuty integration | On-call alerts |
| Documentation | User guide + API docs | Documentation site |
| Performance | Query optimization | <100ms p99 |
| Testing | Integration + load tests | CI pipeline |

**Deliverable:** Production-ready system

---

## 8. Success Criteria

### 8.1 Technical Metrics

| Metric | Target | Critical |
|--------|--------|----------|
| API latency (p50) | <50ms | <200ms |
| API latency (p99) | <200ms | <1000ms |
| Snapshot freshness | <5min | <15min |
| Uptime | 99.9% | 99% |
| Data accuracy | 100% | 100% |

### 8.2 Functional Requirements

- [ ] Realtime snapshot updates every 5 minutes
- [ ] Hourly cluster statistics
- [ ] Daily distribution analysis
- [ ] Feature predictiveness rankings
- [ ] Causal validation flags
- [ ] WebSocket streaming
- [ ] Agent consumption logging
- [ ] Historical data API

### 8.3 Quality Requirements

- [ ] All statistics reproducible
- [ ] Snapshots immutable once created
- [ ] Full audit trail of agent decisions
- [ ] Graceful degradation on component failure
- [ ] Zero data loss on database failure

---

## Appendix A: API Error Codes

| Code | Name | Description |
|------|------|-------------|
| 1001 | SNAPSHOT_NOT_FOUND | Requested snapshot does not exist |
| 1002 | SNAPSHOT_EXPIRED | Snapshot TTL exceeded |
| 1003 | SNAPSHOT_SUPERSEDED | Newer snapshot available |
| 2001 | INVALID_AGENT_ID | Agent ID format invalid |
| 2002 | AGENT_NOT_REGISTERED | Agent must register first |
| 3001 | RATE_LIMIT_EXCEEDED | Too many requests |
| 3002 | INVALID_API_KEY | API key invalid or expired |
| 4001 | COMPUTE_IN_PROGRESS | Snapshot computation ongoing |
| 4002 | COMPUTE_FAILED | Snapshot computation failed |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Snapshot** | Immutable point-in-time statistical summary |
| **TTL** | Time-to-live; duration snapshot is considered valid |
| **Cluster** | Natural grouping in entropy distribution |
| **Regime** | Market state (trending, ranging, uncertain) |
| **Causal** | Feature that leads (predicts) returns, not lags |
| **Stable** | Feature that appears consistently in bootstrap samples |

---

**Document Version:** 1.0
**Created:** 2026-04-05
**Status:** Specification
**Next Step:** Phase 1 implementation
