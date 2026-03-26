//! Redis Feature Publisher
//!
//! Publishes computed features to Redis for consumption by API server and alerts.
//!
//! Channels:
//! - `nat:features:{symbol}` - Real-time feature snapshots
//! - `nat:alerts` - Alert triggers
//!
//! Cache keys:
//! - `nat:latest:{symbol}` - Latest feature snapshot (for REST API)

use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use serde::Serialize;
use tracing::{debug, error, info, warn};

use crate::features::{Features, RegimeFeatures, WhaleFlowFeatures};

/// Feature snapshot for Redis publication
#[derive(Debug, Clone, Serialize)]
pub struct FeatureSnapshot {
    pub timestamp_ms: u64,
    pub symbol: String,
    /// Full feature set as JSON
    pub features: serde_json::Value,
    /// Regime summary (if available)
    pub regime: Option<RegimeSummary>,
    /// Whale activity summary (if available)
    pub whale: Option<WhaleSummary>,
    /// Key metrics for quick access
    pub metrics: QuickMetrics,
}

/// Regime state summary
#[derive(Debug, Clone, Serialize)]
pub struct RegimeSummary {
    pub accumulation_score: f64,
    pub distribution_score: f64,
    pub clarity: f64,
    pub regime_type: String,
    pub range_position_24h: f64,
    pub absorption_zscore: f64,
    pub divergence_zscore: f64,
    pub churn_zscore: f64,
}

impl RegimeSummary {
    pub fn from_features(regime: &RegimeFeatures) -> Self {
        let regime_type = if regime.regime_clarity() < 0.3 {
            "UNCLEAR".to_string()
        } else if regime.accumulation_score > regime.distribution_score {
            "ACCUMULATION".to_string()
        } else {
            "DISTRIBUTION".to_string()
        };

        Self {
            accumulation_score: regime.accumulation_score,
            distribution_score: regime.distribution_score,
            clarity: regime.regime_clarity(),
            regime_type,
            range_position_24h: regime.range_position_24h,
            absorption_zscore: regime.absorption_zscore,
            divergence_zscore: regime.divergence_zscore,
            churn_zscore: regime.churn_zscore,
        }
    }
}

/// Whale activity summary
#[derive(Debug, Clone, Serialize)]
pub struct WhaleSummary {
    pub net_flow_1h: f64,
    pub net_flow_1h_zscore: f64,
    pub net_flow_4h: f64,
    pub net_flow_24h: f64,
    pub intensity: f64,
    pub direction: String,
}

impl WhaleSummary {
    pub fn from_features(whale: &WhaleFlowFeatures) -> Self {
        let direction = if whale.whale_flow_normalized_1h > 1.0 {
            "ACCUMULATING".to_string()
        } else if whale.whale_flow_normalized_1h < -1.0 {
            "DISTRIBUTING".to_string()
        } else {
            "NEUTRAL".to_string()
        };

        Self {
            net_flow_1h: whale.whale_net_flow_1h,
            net_flow_1h_zscore: whale.whale_flow_normalized_1h,
            net_flow_4h: whale.whale_net_flow_4h,
            net_flow_24h: whale.whale_net_flow_24h,
            intensity: whale.whale_flow_intensity,
            direction,
        }
    }
}

/// Quick access metrics
#[derive(Debug, Clone, Serialize)]
pub struct QuickMetrics {
    pub midprice: f64,
    pub spread_bps: f64,
    pub vpin: f64,
    pub tick_entropy: f64,
    pub realized_vol_1m: f64,
    pub imbalance_l1: f64,
}

impl QuickMetrics {
    pub fn from_features(features: &Features) -> Self {
        Self {
            midprice: features.raw.midprice,
            spread_bps: features.raw.spread_bps,
            vpin: features.toxicity.vpin_10,
            tick_entropy: features.entropy.tick_entropy_1m,
            realized_vol_1m: features.volatility.returns_1m,
            imbalance_l1: features.imbalance.qty_l1,
        }
    }
}

/// Redis publisher configuration
#[derive(Debug, Clone)]
pub struct RedisConfig {
    pub url: String,
    pub channel_prefix: String,
    pub cache_ttl_seconds: u64,
    pub enabled: bool,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            channel_prefix: "nat".to_string(),
            cache_ttl_seconds: 60,
            enabled: true,
        }
    }
}

impl RedisConfig {
    /// Load from environment variables
    pub fn from_env() -> Self {
        Self {
            url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string()),
            channel_prefix: std::env::var("REDIS_PREFIX")
                .unwrap_or_else(|_| "nat".to_string()),
            cache_ttl_seconds: std::env::var("REDIS_CACHE_TTL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(60),
            enabled: std::env::var("REDIS_ENABLED")
                .map(|s| s.to_lowercase() != "false" && s != "0")
                .unwrap_or(true),
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

    /// Try to create publisher, return None if connection fails
    pub async fn try_new(config: RedisConfig) -> Option<Self> {
        if !config.enabled {
            info!("Redis publishing disabled");
            return None;
        }

        match Self::new(config).await {
            Ok(publisher) => Some(publisher),
            Err(e) => {
                warn!("Redis connection failed (publishing disabled): {}", e);
                None
            }
        }
    }

    /// Publish feature snapshot
    pub async fn publish_features(
        &mut self,
        symbol: &str,
        features: &Features,
        timestamp_ms: u64,
    ) -> Result<(), redis::RedisError> {
        let snapshot = FeatureSnapshot {
            timestamp_ms,
            symbol: symbol.to_string(),
            features: serde_json::to_value(features.to_vec()).unwrap_or_default(),
            regime: features.regime.as_ref().map(RegimeSummary::from_features),
            whale: features.whale_flow.as_ref().map(WhaleSummary::from_features),
            metrics: QuickMetrics::from_features(features),
        };

        self.publish(&snapshot).await
    }

    /// Publish feature snapshot
    async fn publish(&mut self, snapshot: &FeatureSnapshot) -> Result<(), redis::RedisError> {
        let channel = format!("{}:features:{}", self.config.channel_prefix, snapshot.symbol);
        let json = serde_json::to_string(snapshot).unwrap_or_default();

        // Publish to channel (for real-time subscribers)
        let _: () = self.conn.publish(&channel, &json).await?;

        // Also cache latest value (for REST API)
        let cache_key = format!("{}:latest:{}", self.config.channel_prefix, snapshot.symbol);
        let _: () = self.conn.set_ex(&cache_key, &json, self.config.cache_ttl_seconds).await?;

        debug!("Published features for {} to Redis", snapshot.symbol);

        Ok(())
    }

    /// Publish alert trigger
    pub async fn publish_alert(&mut self, alert: &AlertTrigger) -> Result<(), redis::RedisError> {
        let channel = format!("{}:alerts", self.config.channel_prefix);
        let json = serde_json::to_string(alert).unwrap_or_default();

        let _: () = self.conn.publish(&channel, &json).await?;

        info!("Published alert: {:?} for {}", alert.alert_type, alert.symbol);

        Ok(())
    }

    /// Get channel prefix
    pub fn prefix(&self) -> &str {
        &self.config.channel_prefix
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redis_config_default() {
        let config = RedisConfig::default();
        assert_eq!(config.url, "redis://127.0.0.1:6379");
        assert_eq!(config.channel_prefix, "nat");
        assert!(config.enabled);
    }

    #[test]
    fn test_regime_summary_accumulation() {
        let regime = RegimeFeatures {
            accumulation_score: 0.8,
            distribution_score: 0.3,
            ..Default::default()
        };

        let summary = RegimeSummary::from_features(&regime);
        assert_eq!(summary.regime_type, "ACCUMULATION");
    }

    #[test]
    fn test_regime_summary_distribution() {
        let regime = RegimeFeatures {
            accumulation_score: 0.2,
            distribution_score: 0.7,
            ..Default::default()
        };

        let summary = RegimeSummary::from_features(&regime);
        assert_eq!(summary.regime_type, "DISTRIBUTION");
    }

    #[test]
    fn test_alert_serialization() {
        let alert = AlertTrigger {
            timestamp_ms: 1234567890,
            symbol: "BTC".to_string(),
            alert_type: AlertType::WhaleAccumulation,
            severity: AlertSeverity::Warning,
            message: "Test alert".to_string(),
            data: serde_json::json!({"flow": 1000000}),
        };

        let json = serde_json::to_string(&alert).unwrap();
        assert!(json.contains("WhaleAccumulation"));
        assert!(json.contains("BTC"));
    }
}
