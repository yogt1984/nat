# PRIORITIES: API + Alerts Infrastructure

**Date:** 2026-03-23
**Goal:** Build real-time API and alerting system for feature consumption
**Timeline:** 7 days
**Status:** Planning

---

## Overview

Build infrastructure to expose computed features via REST API, WebSocket streaming, and Telegram alerts. This serves both personal monitoring and potential SaaS pivot.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TARGET ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐   │
│  │   Ingestor   │────►│    Redis     │────►│     API Server       │   │
│  │   (Rust)     │     │   Pub/Sub    │     │     (Axum)           │   │
│  │              │     │   + Cache    │     │                      │   │
│  └──────────────┘     └──────────────┘     │  ┌────────────────┐  │   │
│         │                    │              │  │  REST API      │  │   │
│         │                    │              │  │  /api/features │  │   │
│         ▼                    │              │  │  /api/whales   │  │   │
│  ┌──────────────┐            │              │  │  /api/regime   │  │   │
│  │   Parquet    │            │              │  └────────────────┘  │   │
│  │   Storage    │            │              │                      │   │
│  └──────────────┘            │              │  ┌────────────────┐  │   │
│                              │              │  │  WebSocket     │  │   │
│                              │              │  │  /ws/stream    │  │   │
│                              │              │  └────────────────┘  │   │
│                              │              └──────────────────────┘   │
│                              │                         │               │
│                              │                         ▼               │
│                              │              ┌──────────────────────┐   │
│                              └─────────────►│   Alert Service      │   │
│                                             │   (Telegram/Discord) │   │
│                                             └──────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Redis Integration (Day 1-2)

### 1.1 Install and Configure Redis

**Task:** Set up Redis for pub/sub and caching

**Steps:**
```bash
# Install Redis
sudo apt install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify
redis-cli ping  # Should return PONG
```

**Configuration (`/etc/redis/redis.conf`):**
```conf
# Bind to localhost only (security)
bind 127.0.0.1

# Disable persistence (we don't need it, features are ephemeral)
save ""
appendonly no

# Set max memory (features are small)
maxmemory 256mb
maxmemory-policy allkeys-lru
```

**Acceptance Criteria:**
- [ ] Redis running on localhost:6379
- [ ] `redis-cli ping` returns PONG
- [ ] Memory limit configured

---

### 1.2 Add Redis Client to Ingestor

**Task:** Add Redis dependency and publish features

**File:** `rust/ing/Cargo.toml`
```toml
[dependencies]
# Add these
redis = { version = "0.25", features = ["tokio-comp", "connection-manager"] }
```

**File:** `rust/ing/src/redis_publisher.rs` (NEW)
```rust
//! Redis Feature Publisher
//!
//! Publishes computed features to Redis for consumption by API server and alerts.

use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use serde::Serialize;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Feature snapshot for Redis publication
#[derive(Debug, Clone, Serialize)]
pub struct FeatureSnapshot {
    pub timestamp_ms: u64,
    pub symbol: String,
    pub features: serde_json::Value,
    pub regime: Option<RegimeSnapshot>,
    pub whale_summary: Option<WhaleSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegimeSnapshot {
    pub accumulation_score: f64,
    pub distribution_score: f64,
    pub clarity: f64,
    pub range_position_24h: f64,
    pub absorption_zscore: f64,
    pub churn_zscore: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WhaleSummary {
    pub net_flow_1h: f64,
    pub net_flow_1h_zscore: f64,
    pub net_flow_4h: f64,
    pub intensity: f64,
}

/// Redis publisher configuration
#[derive(Debug, Clone)]
pub struct RedisConfig {
    pub url: String,
    pub channel_prefix: String,
    pub cache_ttl_seconds: u64,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            channel_prefix: "nat".to_string(),
            cache_ttl_seconds: 60,
        }
    }
}

/// Redis feature publisher
pub struct RedisPublisher {
    conn: ConnectionManager,
    config: RedisConfig,
}

impl RedisPublisher {
    /// Create new publisher
    pub async fn new(config: RedisConfig) -> Result<Self, redis::RedisError> {
        let client = redis::Client::open(config.url.as_str())?;
        let conn = ConnectionManager::new(client).await?;

        info!("Redis publisher connected to {}", config.url);

        Ok(Self { conn, config })
    }

    /// Publish feature snapshot
    pub async fn publish(&mut self, snapshot: &FeatureSnapshot) -> Result<(), redis::RedisError> {
        let channel = format!("{}:features:{}", self.config.channel_prefix, snapshot.symbol);
        let json = serde_json::to_string(snapshot).unwrap_or_default();

        // Publish to channel (for real-time subscribers)
        self.conn.publish::<_, _, ()>(&channel, &json).await?;

        // Also cache latest value (for REST API)
        let cache_key = format!("{}:latest:{}", self.config.channel_prefix, snapshot.symbol);
        self.conn.set_ex::<_, _, ()>(
            &cache_key,
            &json,
            self.config.cache_ttl_seconds
        ).await?;

        debug!("Published features for {} to Redis", snapshot.symbol);

        Ok(())
    }

    /// Publish alert trigger
    pub async fn publish_alert(&mut self, alert: &AlertTrigger) -> Result<(), redis::RedisError> {
        let channel = format!("{}:alerts", self.config.channel_prefix);
        let json = serde_json::to_string(alert).unwrap_or_default();

        self.conn.publish::<_, _, ()>(&channel, &json).await?;

        info!("Published alert: {:?}", alert.alert_type);

        Ok(())
    }
}

/// Alert trigger for pub/sub
#[derive(Debug, Clone, Serialize)]
pub struct AlertTrigger {
    pub timestamp_ms: u64,
    pub symbol: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub enum AlertType {
    WhaleAccumulation,
    WhaleDistribution,
    LiquidationCluster,
    RegimeChange,
    EntropyDrop,
    ConcentrationSpike,
    Custom(String),
}

#[derive(Debug, Clone, Serialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}
```

**Acceptance Criteria:**
- [ ] Redis dependency added to Cargo.toml
- [ ] RedisPublisher struct implemented
- [ ] Pub/sub channel: `nat:features:{symbol}`
- [ ] Cache key: `nat:latest:{symbol}`
- [ ] Alert channel: `nat:alerts`

---

### 1.3 Integrate Publisher into Main Loop

**Task:** Call RedisPublisher from main ingestor loop

**File:** `rust/ing/src/main.rs` (MODIFY)

**Integration Points:**
1. Initialize RedisPublisher at startup
2. After computing features, create FeatureSnapshot
3. Publish to Redis
4. Check alert conditions and publish alerts

```rust
// In main() or run() function:

// Initialize Redis publisher
let redis_config = RedisConfig::default();
let mut redis_publisher = RedisPublisher::new(redis_config).await
    .expect("Failed to connect to Redis");

// In the feature emission loop (after compute_features()):
let snapshot = FeatureSnapshot {
    timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
    symbol: symbol.to_string(),
    features: serde_json::to_value(&features).unwrap_or_default(),
    regime: features.regime.as_ref().map(|r| RegimeSnapshot {
        accumulation_score: r.accumulation_score,
        distribution_score: r.distribution_score,
        clarity: r.regime_clarity(),
        range_position_24h: r.range_position_24h,
        absorption_zscore: r.absorption_zscore,
        churn_zscore: r.churn_zscore,
    }),
    whale_summary: features.whale_flow.as_ref().map(|w| WhaleSummary {
        net_flow_1h: w.net_flow_1h,
        net_flow_1h_zscore: w.net_flow_normalized_1h,
        net_flow_4h: w.net_flow_4h,
        intensity: w.whale_intensity,
    }),
};

if let Err(e) = redis_publisher.publish(&snapshot).await {
    warn!("Failed to publish to Redis: {}", e);
}

// Check for alert conditions
if let Some(alert) = check_alert_conditions(&features, &symbol) {
    if let Err(e) = redis_publisher.publish_alert(&alert).await {
        warn!("Failed to publish alert: {}", e);
    }
}
```

**Acceptance Criteria:**
- [ ] Features published to Redis every emission interval
- [ ] Can verify with: `redis-cli SUBSCRIBE nat:features:BTC`
- [ ] Latest features cached for REST access
- [ ] No performance degradation (async publish)

---

### 1.4 Alert Condition Checker

**Task:** Implement alert condition detection

**File:** `rust/ing/src/alerts.rs` (NEW)
```rust
//! Alert Condition Detection
//!
//! Checks features against thresholds and generates alerts.

use crate::features::Features;
use crate::redis_publisher::{AlertTrigger, AlertType, AlertSeverity};

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Whale flow z-score threshold for accumulation alert
    pub whale_accumulation_zscore: f64,
    /// Whale flow z-score threshold for distribution alert
    pub whale_distribution_zscore: f64,
    /// Liquidation cascade probability threshold
    pub cascade_probability: f64,
    /// Entropy drop threshold (below this = predictable)
    pub entropy_low: f64,
    /// Concentration z-score spike threshold
    pub concentration_zscore: f64,
    /// Regime clarity threshold for regime change alert
    pub regime_clarity: f64,
    /// Cooldown between same alert type (seconds)
    pub cooldown_seconds: u64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            whale_accumulation_zscore: 2.0,
            whale_distribution_zscore: -2.0,
            cascade_probability: 0.7,
            entropy_low: 0.5,
            concentration_zscore: 2.5,
            regime_clarity: 0.6,
            cooldown_seconds: 300, // 5 minutes
        }
    }
}

/// Alert state tracker (for cooldowns)
pub struct AlertTracker {
    config: AlertConfig,
    last_alerts: std::collections::HashMap<String, u64>, // alert_key -> timestamp
}

impl AlertTracker {
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            last_alerts: std::collections::HashMap::new(),
        }
    }

    /// Check features and return any triggered alerts
    pub fn check(&mut self, features: &Features, symbol: &str, timestamp_ms: u64) -> Vec<AlertTrigger> {
        let mut alerts = Vec::new();

        // Check whale accumulation
        if let Some(ref wf) = features.whale_flow {
            if wf.net_flow_normalized_1h >= self.config.whale_accumulation_zscore {
                if self.can_fire("whale_acc", symbol, timestamp_ms) {
                    alerts.push(AlertTrigger {
                        timestamp_ms,
                        symbol: symbol.to_string(),
                        alert_type: AlertType::WhaleAccumulation,
                        severity: AlertSeverity::Warning,
                        message: format!(
                            "Whale accumulation detected: z-score {:.2}, flow ${:.0}",
                            wf.net_flow_normalized_1h,
                            wf.net_flow_1h
                        ),
                        data: serde_json::json!({
                            "flow_1h": wf.net_flow_1h,
                            "flow_zscore": wf.net_flow_normalized_1h,
                            "flow_4h": wf.net_flow_4h,
                            "intensity": wf.whale_intensity,
                        }),
                    });
                    self.mark_fired("whale_acc", symbol, timestamp_ms);
                }
            }

            // Check whale distribution
            if wf.net_flow_normalized_1h <= self.config.whale_distribution_zscore {
                if self.can_fire("whale_dist", symbol, timestamp_ms) {
                    alerts.push(AlertTrigger {
                        timestamp_ms,
                        symbol: symbol.to_string(),
                        alert_type: AlertType::WhaleDistribution,
                        severity: AlertSeverity::Warning,
                        message: format!(
                            "Whale distribution detected: z-score {:.2}, flow ${:.0}",
                            wf.net_flow_normalized_1h,
                            wf.net_flow_1h
                        ),
                        data: serde_json::json!({
                            "flow_1h": wf.net_flow_1h,
                            "flow_zscore": wf.net_flow_normalized_1h,
                        }),
                    });
                    self.mark_fired("whale_dist", symbol, timestamp_ms);
                }
            }
        }

        // Check regime change
        if let Some(ref regime) = features.regime {
            if regime.regime_clarity() >= self.config.regime_clarity {
                let regime_type = if regime.accumulation_score > regime.distribution_score {
                    "ACCUMULATION"
                } else {
                    "DISTRIBUTION"
                };

                if self.can_fire(&format!("regime_{}", regime_type.to_lowercase()), symbol, timestamp_ms) {
                    alerts.push(AlertTrigger {
                        timestamp_ms,
                        symbol: symbol.to_string(),
                        alert_type: AlertType::RegimeChange,
                        severity: AlertSeverity::Info,
                        message: format!(
                            "Regime detected: {} (clarity: {:.2})",
                            regime_type,
                            regime.regime_clarity()
                        ),
                        data: serde_json::json!({
                            "accumulation_score": regime.accumulation_score,
                            "distribution_score": regime.distribution_score,
                            "clarity": regime.regime_clarity(),
                            "range_position": regime.range_position_24h,
                        }),
                    });
                    self.mark_fired(&format!("regime_{}", regime_type.to_lowercase()), symbol, timestamp_ms);
                }
            }
        }

        // Check liquidation cascade risk
        if let Some(ref liq) = features.liquidation_risk {
            if liq.cascade_probability >= self.config.cascade_probability {
                if self.can_fire("cascade", symbol, timestamp_ms) {
                    let direction = if liq.liq_asymmetry > 0.0 { "UPWARD" } else { "DOWNWARD" };
                    alerts.push(AlertTrigger {
                        timestamp_ms,
                        symbol: symbol.to_string(),
                        alert_type: AlertType::LiquidationCluster,
                        severity: AlertSeverity::Critical,
                        message: format!(
                            "Liquidation cascade risk: {:.0}% probability, {} pressure",
                            liq.cascade_probability * 100.0,
                            direction
                        ),
                        data: serde_json::json!({
                            "cascade_probability": liq.cascade_probability,
                            "asymmetry": liq.liq_asymmetry,
                            "risk_above_1pct": liq.liq_risk_above_1pct,
                            "risk_below_1pct": liq.liq_risk_below_1pct,
                        }),
                    });
                    self.mark_fired("cascade", symbol, timestamp_ms);
                }
            }
        }

        // Check entropy drop
        if features.entropy.tick_entropy_1m < self.config.entropy_low {
            if self.can_fire("entropy", symbol, timestamp_ms) {
                alerts.push(AlertTrigger {
                    timestamp_ms,
                    symbol: symbol.to_string(),
                    alert_type: AlertType::EntropyDrop,
                    severity: AlertSeverity::Info,
                    message: format!(
                        "Low entropy detected: {:.3} (market predictable)",
                        features.entropy.tick_entropy_1m
                    ),
                    data: serde_json::json!({
                        "entropy_1m": features.entropy.tick_entropy_1m,
                        "entropy_5s": features.entropy.tick_entropy_5s,
                    }),
                });
                self.mark_fired("entropy", symbol, timestamp_ms);
            }
        }

        alerts
    }

    fn can_fire(&self, alert_type: &str, symbol: &str, now_ms: u64) -> bool {
        let key = format!("{}:{}", alert_type, symbol);
        match self.last_alerts.get(&key) {
            Some(&last_time) => {
                let elapsed_seconds = (now_ms - last_time) / 1000;
                elapsed_seconds >= self.config.cooldown_seconds
            }
            None => true,
        }
    }

    fn mark_fired(&mut self, alert_type: &str, symbol: &str, now_ms: u64) {
        let key = format!("{}:{}", alert_type, symbol);
        self.last_alerts.insert(key, now_ms);
    }
}
```

**Acceptance Criteria:**
- [ ] Whale accumulation/distribution alerts
- [ ] Regime change alerts
- [ ] Liquidation cascade alerts
- [ ] Entropy drop alerts
- [ ] Cooldown prevents spam
- [ ] Alerts published to `nat:alerts` channel

---

## Phase 2: API Server (Day 3-4)

### 2.1 Create API Server Crate

**Task:** Create new Rust crate for API server

```bash
cd rust
cargo new api --name nat-api
```

**File:** `rust/api/Cargo.toml`
```toml
[package]
name = "nat-api"
version = "0.1.0"
edition = "2021"

[dependencies]
# Web framework
axum = { version = "0.7", features = ["ws", "macros"] }
tokio = { version = "1", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }

# Redis
redis = { version = "0.25", features = ["tokio-comp", "connection-manager"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Utilities
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4"] }

# Configuration
config = "0.14"
dotenvy = "0.15"
```

**Acceptance Criteria:**
- [ ] New crate created at `rust/api/`
- [ ] Dependencies added
- [ ] Compiles with `cargo check`

---

### 2.2 API Server Main Structure

**File:** `rust/api/src/main.rs`
```rust
//! NAT API Server
//!
//! REST API and WebSocket server for feature consumption.

mod config;
mod routes;
mod state;
mod redis_client;

use axum::{
    Router,
    routing::{get, post},
    http::Method,
};
use tower_http::cors::{CorsLayer, Any};
use tower_http::trace::TraceLayer;
use std::sync::Arc;
use tracing::info;

use crate::config::ApiConfig;
use crate::state::AppState;
use crate::redis_client::RedisClient;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nat_api=debug,tower_http=debug".into())
        )
        .init();

    // Load config
    let config = ApiConfig::load().expect("Failed to load config");

    // Initialize Redis client
    let redis = RedisClient::new(&config.redis_url)
        .await
        .expect("Failed to connect to Redis");

    // Create app state
    let state = Arc::new(AppState::new(redis, config.clone()));

    // Build router
    let app = Router::new()
        // Health check
        .route("/health", get(routes::health::health_check))

        // Feature endpoints
        .route("/api/features/:symbol", get(routes::features::get_latest))
        .route("/api/features/:symbol/history", get(routes::features::get_history))

        // Whale endpoints
        .route("/api/whales/:symbol", get(routes::whales::get_whale_summary))

        // Regime endpoints
        .route("/api/regime/:symbol", get(routes::regime::get_regime_state))

        // WebSocket streaming
        .route("/ws/stream/:symbol", get(routes::ws::websocket_handler))
        .route("/ws/alerts", get(routes::ws::alerts_websocket_handler))

        // Add state
        .with_state(state)

        // Add middleware
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers(Any)
        );

    // Start server
    let addr = format!("{}:{}", config.host, config.port);
    info!("Starting API server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

**Acceptance Criteria:**
- [ ] Server starts on configured port
- [ ] CORS enabled for browser access
- [ ] Tracing/logging configured
- [ ] Routes registered

---

### 2.3 API Routes Implementation

**File:** `rust/api/src/routes/features.rs`
```rust
//! Feature API routes

use axum::{
    extract::{Path, State, Query},
    Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct FeatureResponse {
    pub timestamp_ms: u64,
    pub symbol: String,
    pub features: serde_json::Value,
    pub regime: Option<serde_json::Value>,
    pub whale_summary: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// GET /api/features/:symbol
/// Returns latest feature snapshot for a symbol
pub async fn get_latest(
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
) -> Result<Json<FeatureResponse>, (StatusCode, Json<ErrorResponse>)> {
    let symbol = symbol.to_uppercase();

    match state.redis.get_latest_features(&symbol).await {
        Ok(Some(snapshot)) => Ok(Json(snapshot)),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("No data for symbol: {}", symbol),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Redis error: {}", e),
            }),
        )),
    }
}

#[derive(Debug, Deserialize)]
pub struct HistoryQuery {
    /// Number of minutes of history
    pub minutes: Option<u32>,
    /// Maximum number of records
    pub limit: Option<u32>,
}

/// GET /api/features/:symbol/history
/// Returns historical features (from Parquet or cache)
pub async fn get_history(
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<Vec<FeatureResponse>>, (StatusCode, Json<ErrorResponse>)> {
    // For now, return recent cached values
    // TODO: Implement Parquet history access

    let minutes = query.minutes.unwrap_or(60);
    let limit = query.limit.unwrap_or(100);

    match state.redis.get_feature_history(&symbol.to_uppercase(), minutes, limit).await {
        Ok(history) => Ok(Json(history)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Error fetching history: {}", e),
            }),
        )),
    }
}
```

**File:** `rust/api/src/routes/regime.rs`
```rust
//! Regime API routes

use axum::{
    extract::{Path, State},
    Json,
    http::StatusCode,
};
use serde::Serialize;
use std::sync::Arc;
use crate::state::AppState;
use crate::routes::features::ErrorResponse;

#[derive(Debug, Serialize)]
pub struct RegimeResponse {
    pub timestamp_ms: u64,
    pub symbol: String,
    pub regime_type: String,
    pub accumulation_score: f64,
    pub distribution_score: f64,
    pub clarity: f64,
    pub range_position_24h: f64,
    pub absorption_zscore: f64,
    pub divergence_zscore: f64,
    pub churn_zscore: f64,
    pub interpretation: String,
}

/// GET /api/regime/:symbol
/// Returns current regime state with interpretation
pub async fn get_regime_state(
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
) -> Result<Json<RegimeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let symbol = symbol.to_uppercase();

    match state.redis.get_latest_features(&symbol).await {
        Ok(Some(snapshot)) => {
            if let Some(regime) = snapshot.regime {
                let regime: serde_json::Value = regime;

                let acc_score = regime["accumulation_score"].as_f64().unwrap_or(0.0);
                let dist_score = regime["distribution_score"].as_f64().unwrap_or(0.0);
                let clarity = regime["clarity"].as_f64().unwrap_or(0.0);
                let range_pos = regime["range_position_24h"].as_f64().unwrap_or(0.5);

                let (regime_type, interpretation) = interpret_regime(
                    acc_score, dist_score, clarity, range_pos
                );

                Ok(Json(RegimeResponse {
                    timestamp_ms: snapshot.timestamp_ms,
                    symbol,
                    regime_type,
                    accumulation_score: acc_score,
                    distribution_score: dist_score,
                    clarity,
                    range_position_24h: range_pos,
                    absorption_zscore: regime["absorption_zscore"].as_f64().unwrap_or(0.0),
                    divergence_zscore: regime["divergence_zscore"].as_f64().unwrap_or(0.0),
                    churn_zscore: regime["churn_zscore"].as_f64().unwrap_or(0.0),
                    interpretation,
                }))
            } else {
                Err((
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: "Regime data not yet available (need ~60 minutes of data)".to_string(),
                    }),
                ))
            }
        }
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("No data for symbol: {}", symbol),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Redis error: {}", e),
            }),
        )),
    }
}

fn interpret_regime(acc: f64, dist: f64, clarity: f64, range_pos: f64) -> (String, String) {
    if clarity < 0.3 {
        return ("UNCLEAR".to_string(), "No clear regime detected. Market in transition or ranging.".to_string());
    }

    if acc > dist {
        let regime = "ACCUMULATION".to_string();
        let interp = if range_pos < 0.3 {
            "Strong accumulation at range lows. Smart money likely buying. Potential markup phase approaching."
        } else if range_pos < 0.5 {
            "Accumulation detected in lower half of range. Buying pressure with room to run."
        } else {
            "Accumulation detected but price elevated. Watch for distribution transition."
        };
        (regime, interp.to_string())
    } else {
        let regime = "DISTRIBUTION".to_string();
        let interp = if range_pos > 0.7 {
            "Strong distribution at range highs. Smart money likely selling. Potential markdown phase approaching."
        } else if range_pos > 0.5 {
            "Distribution detected in upper half of range. Selling pressure building."
        } else {
            "Distribution detected but price already low. Watch for accumulation transition."
        };
        (regime, interp.to_string())
    }
}
```

**File:** `rust/api/src/routes/ws.rs`
```rust
//! WebSocket routes for real-time streaming

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    response::IntoResponse,
};
use futures_util::{StreamExt, SinkExt};
use std::sync::Arc;
use tracing::{debug, info, error};
use crate::state::AppState;

/// GET /ws/stream/:symbol
/// WebSocket endpoint for real-time feature streaming
pub async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
) -> impl IntoResponse {
    let symbol = symbol.to_uppercase();
    info!("WebSocket connection request for {}", symbol);

    ws.on_upgrade(move |socket| handle_feature_stream(socket, state, symbol))
}

async fn handle_feature_stream(socket: WebSocket, state: Arc<AppState>, symbol: String) {
    let (mut sender, mut receiver) = socket.split();

    // Subscribe to Redis channel for this symbol
    let channel = format!("nat:features:{}", symbol);

    // Create a subscription
    let mut pubsub = match state.redis.subscribe(&channel).await {
        Ok(ps) => ps,
        Err(e) => {
            error!("Failed to subscribe to {}: {}", channel, e);
            return;
        }
    };

    info!("WebSocket connected and subscribed to {}", channel);

    // Handle incoming messages (for ping/pong or client commands)
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Close(_)) => break,
                Ok(Message::Ping(data)) => {
                    // Pong is handled automatically by axum
                    debug!("Received ping");
                }
                _ => {}
            }
        }
    });

    // Forward Redis messages to WebSocket
    loop {
        tokio::select! {
            msg = pubsub.on_message() => {
                if let Some(payload) = msg {
                    let text: String = payload.get_payload().unwrap_or_default();
                    if sender.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
            }
            _ = recv_task => {
                break;
            }
        }
    }

    info!("WebSocket disconnected from {}", symbol);
}

/// GET /ws/alerts
/// WebSocket endpoint for alert streaming
pub async fn alerts_websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    info!("WebSocket connection request for alerts");
    ws.on_upgrade(move |socket| handle_alerts_stream(socket, state))
}

async fn handle_alerts_stream(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    let channel = "nat:alerts";

    let mut pubsub = match state.redis.subscribe(channel).await {
        Ok(ps) => ps,
        Err(e) => {
            error!("Failed to subscribe to alerts: {}", e);
            return;
        }
    };

    info!("Alerts WebSocket connected");

    loop {
        tokio::select! {
            msg = pubsub.on_message() => {
                if let Some(payload) = msg {
                    let text: String = payload.get_payload().unwrap_or_default();
                    if sender.send(Message::Text(text)).await.is_err() {
                        break;
                    }
                }
            }
        }
    }
}
```

**Acceptance Criteria:**
- [ ] `GET /api/features/:symbol` returns latest features
- [ ] `GET /api/regime/:symbol` returns regime with interpretation
- [ ] `GET /api/whales/:symbol` returns whale summary
- [ ] WebSocket `/ws/stream/:symbol` streams real-time features
- [ ] WebSocket `/ws/alerts` streams alerts

---

### 2.4 API Configuration

**File:** `rust/api/src/config.rs`
```rust
//! API Server Configuration

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub redis_url: String,
    pub cors_origins: Vec<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 3000,
            redis_url: "redis://127.0.0.1:6379".to_string(),
            cors_origins: vec!["*".to_string()],
        }
    }
}

impl ApiConfig {
    pub fn load() -> Result<Self, config::ConfigError> {
        // Load from environment or config file
        let config = config::Config::builder()
            .set_default("host", "0.0.0.0")?
            .set_default("port", 3000)?
            .set_default("redis_url", "redis://127.0.0.1:6379")?
            .add_source(config::Environment::with_prefix("NAT_API"))
            .build()?;

        config.try_deserialize()
    }
}
```

**File:** `config/api.toml` (NEW)
```toml
host = "0.0.0.0"
port = 3000
redis_url = "redis://127.0.0.1:6379"
cors_origins = ["*"]
```

**Acceptance Criteria:**
- [ ] Configuration loads from environment variables
- [ ] Default values work out of the box
- [ ] Port configurable via `NAT_API_PORT`

---

## Phase 3: Telegram Alerts (Day 5-6)

### 3.1 Create Telegram Bot

**Task:** Set up Telegram bot for alerts

**Steps:**
1. Message @BotFather on Telegram
2. Send `/newbot`
3. Choose name: "NAT Alerts"
4. Choose username: `nat_alerts_bot` (must be unique)
5. Save the API token

**Store token:**
```bash
# Add to .env file
echo "TELEGRAM_BOT_TOKEN=your_token_here" >> .env
echo "TELEGRAM_CHAT_ID=your_chat_id" >> .env
```

**Get your chat ID:**
1. Start a chat with your bot
2. Send any message
3. Visit: `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Find your `chat.id` in the response

---

### 3.2 Alert Service Implementation

**File:** `rust/api/src/telegram.rs`
```rust
//! Telegram Alert Service
//!
//! Subscribes to Redis alerts and sends to Telegram.

use reqwest::Client;
use serde::Serialize;
use tracing::{info, error, warn};

pub struct TelegramBot {
    client: Client,
    token: String,
    chat_id: String,
}

#[derive(Serialize)]
struct SendMessageRequest {
    chat_id: String,
    text: String,
    parse_mode: String,
}

impl TelegramBot {
    pub fn new(token: String, chat_id: String) -> Self {
        Self {
            client: Client::new(),
            token,
            chat_id,
        }
    }

    pub async fn send_alert(&self, alert: &AlertMessage) -> Result<(), reqwest::Error> {
        let text = format_alert_message(alert);

        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.token);

        let request = SendMessageRequest {
            chat_id: self.chat_id.clone(),
            text,
            parse_mode: "HTML".to_string(),
        };

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            info!("Alert sent to Telegram");
        } else {
            warn!("Telegram API error: {:?}", response.text().await);
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AlertMessage {
    pub symbol: String,
    pub alert_type: String,
    pub severity: String,
    pub message: String,
    pub timestamp: String,
    pub data: serde_json::Value,
}

fn format_alert_message(alert: &AlertMessage) -> String {
    let emoji = match alert.severity.as_str() {
        "Critical" => "🚨",
        "Warning" => "⚠️",
        "Info" => "ℹ️",
        _ => "📊",
    };

    let type_emoji = match alert.alert_type.as_str() {
        "WhaleAccumulation" => "🐋⬆️",
        "WhaleDistribution" => "🐋⬇️",
        "LiquidationCluster" => "💥",
        "RegimeChange" => "🔄",
        "EntropyDrop" => "🎯",
        "ConcentrationSpike" => "📊",
        _ => "📌",
    };

    format!(
        "{emoji} <b>NAT Alert: {type_emoji} {alert_type}</b>\n\n\
        <b>Symbol:</b> {symbol}\n\
        <b>Message:</b> {message}\n\n\
        <b>Details:</b>\n<pre>{data}</pre>\n\n\
        <i>{timestamp}</i>",
        emoji = emoji,
        type_emoji = type_emoji,
        alert_type = alert.alert_type,
        symbol = alert.symbol,
        message = alert.message,
        data = serde_json::to_string_pretty(&alert.data).unwrap_or_default(),
        timestamp = alert.timestamp,
    )
}

/// Run the alert service (subscribes to Redis and forwards to Telegram)
pub async fn run_alert_service(
    redis_url: &str,
    telegram_token: String,
    telegram_chat_id: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = redis::Client::open(redis_url)?;
    let mut pubsub = client.get_async_pubsub().await?;

    pubsub.subscribe("nat:alerts").await?;

    let bot = TelegramBot::new(telegram_token, telegram_chat_id);

    info!("Alert service started, listening for alerts...");

    loop {
        let msg = pubsub.on_message().next().await;

        if let Some(msg) = msg {
            let payload: String = msg.get_payload().unwrap_or_default();

            match serde_json::from_str::<serde_json::Value>(&payload) {
                Ok(alert_json) => {
                    let alert = AlertMessage {
                        symbol: alert_json["symbol"].as_str().unwrap_or("").to_string(),
                        alert_type: format!("{:?}", alert_json["alert_type"]),
                        severity: format!("{:?}", alert_json["severity"]),
                        message: alert_json["message"].as_str().unwrap_or("").to_string(),
                        timestamp: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                        data: alert_json["data"].clone(),
                    };

                    if let Err(e) = bot.send_alert(&alert).await {
                        error!("Failed to send Telegram alert: {}", e);
                    }
                }
                Err(e) => {
                    warn!("Failed to parse alert: {}", e);
                }
            }
        }
    }
}
```

**Acceptance Criteria:**
- [ ] Bot created and token saved
- [ ] Chat ID obtained
- [ ] Alerts formatted with emojis and HTML
- [ ] Messages sent successfully

---

### 3.3 Alert Service Binary

**File:** `rust/api/src/bin/alert_service.rs`
```rust
//! Standalone alert service
//!
//! Run with: cargo run --bin alert-service

use nat_api::telegram::run_alert_service;
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("nat_api=info")
        .init();

    dotenvy::dotenv().ok();

    let redis_url = std::env::var("REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let telegram_token = std::env::var("TELEGRAM_BOT_TOKEN")
        .expect("TELEGRAM_BOT_TOKEN must be set");
    let telegram_chat_id = std::env::var("TELEGRAM_CHAT_ID")
        .expect("TELEGRAM_CHAT_ID must be set");

    info!("Starting alert service...");

    if let Err(e) = run_alert_service(&redis_url, telegram_token, telegram_chat_id).await {
        eprintln!("Alert service error: {}", e);
    }
}
```

**Update Makefile:**
```makefile
# Add to Makefile
alert_service:
	@echo "Starting Telegram alert service..."
	cd rust && cargo run --bin alert-service
```

**Acceptance Criteria:**
- [ ] Alert service runs as separate process
- [ ] Receives alerts from Redis
- [ ] Sends formatted messages to Telegram
- [ ] Handles errors gracefully

---

## Phase 4: Integration & Testing (Day 7)

### 4.1 Update Makefile

**Add to `Makefile`:**
```makefile
# =============================================================================
# API SERVER
# =============================================================================

# Start API server
api:
	@echo "Starting NAT API server..."
	cd rust && cargo run --release --bin nat-api

# Start alert service
alerts:
	@echo "Starting Telegram alert service..."
	cd rust && cargo run --release --bin alert-service

# Start everything (ingestor + API + alerts)
serve_all: release
	@echo "Starting full stack..."
	@echo "  - Ingestor: ingesting data + publishing to Redis"
	@echo "  - API Server: http://localhost:3000"
	@echo "  - Alert Service: sending to Telegram"
	@echo ""
	@tmux new-session -d -s nat 'cd rust && ./target/release/ing ../config/ing.toml' \; \
		split-window -h 'cd rust && ./target/release/nat-api' \; \
		split-window -v 'cd rust && ./target/release/alert-service' \; \
		attach

# Test Redis connection
test_redis:
	@echo "Testing Redis connection..."
	redis-cli ping
	@echo "Subscribing to features (Ctrl+C to exit)..."
	redis-cli SUBSCRIBE nat:features:BTC
```

---

### 4.2 Integration Test Script

**File:** `scripts/test_api.sh`
```bash
#!/bin/bash
# Test API endpoints

API_URL="${API_URL:-http://localhost:3000}"

echo "Testing NAT API at $API_URL"
echo "================================"

# Health check
echo -n "Health check: "
curl -s "$API_URL/health" | jq .

# Features endpoint
echo -n "Features (BTC): "
curl -s "$API_URL/api/features/BTC" | jq .

# Regime endpoint
echo -n "Regime (BTC): "
curl -s "$API_URL/api/regime/BTC" | jq .

# Whales endpoint
echo -n "Whales (BTC): "
curl -s "$API_URL/api/whales/BTC" | jq .

echo ""
echo "WebSocket test: wscat -c ws://localhost:3000/ws/stream/BTC"
```

---

### 4.3 Docker Compose (Optional)

**File:** `docker-compose.yml`
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

  ingestor:
    build:
      context: ./rust
      dockerfile: Dockerfile.ingestor
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./config:/app/config

  api:
    build:
      context: ./rust
      dockerfile: Dockerfile.api
    depends_on:
      - redis
    ports:
      - "3000:3000"
    environment:
      - REDIS_URL=redis://redis:6379
      - NAT_API_PORT=3000

  alerts:
    build:
      context: ./rust
      dockerfile: Dockerfile.alerts
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}

volumes:
  redis_data:
```

---

## Summary Checklist

### Phase 1: Redis (Day 1-2)
- [ ] Redis installed and running
- [ ] RedisPublisher implemented
- [ ] Integrated into ingestor main loop
- [ ] Alert conditions checked
- [ ] Features published to `nat:features:{symbol}`
- [ ] Alerts published to `nat:alerts`

### Phase 2: API Server (Day 3-4)
- [ ] API crate created
- [ ] REST endpoints implemented
- [ ] WebSocket streaming working
- [ ] CORS configured
- [ ] Tested with curl/wscat

### Phase 3: Telegram Alerts (Day 5-6)
- [ ] Bot created with BotFather
- [ ] Token and chat ID configured
- [ ] Alert service running
- [ ] Messages formatted nicely
- [ ] Tested end-to-end

### Phase 4: Integration (Day 7)
- [ ] Makefile updated
- [ ] All services start together
- [ ] Test script passing
- [ ] Documentation updated
- [ ] (Optional) Docker Compose

---

## API Reference (Final)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/features/:symbol` | GET | Latest features |
| `/api/features/:symbol/history` | GET | Historical features |
| `/api/regime/:symbol` | GET | Regime state + interpretation |
| `/api/whales/:symbol` | GET | Whale activity summary |
| `/ws/stream/:symbol` | WS | Real-time feature stream |
| `/ws/alerts` | WS | Real-time alert stream |

---

## Next Steps After Completion

1. **Run preliminary hypothesis tests** with collected data
2. **Monitor alerts** for 1-2 weeks to validate signal quality
3. **Iterate on thresholds** based on false positive/negative rate
4. **Build frontend dashboard** (if pursuing SaaS)
5. **Add authentication** (API keys, rate limiting)
