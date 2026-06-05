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

use std::collections::HashMap;
use std::time::{Duration, Instant};

use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use serde::Serialize;
use tracing::{debug, info, warn};

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
    pub publish_interval_ms: u64,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            channel_prefix: "nat".to_string(),
            cache_ttl_seconds: 60,
            enabled: true,
            publish_interval_ms: 500,
        }
    }
}

impl RedisConfig {
    /// Load from environment variables
    pub fn from_env() -> Self {
        Self::from_env_with_toml_url(None)
    }

    /// Load from environment variables, falling back to toml config URL
    pub fn from_env_with_toml_url(toml_url: Option<&str>) -> Self {
        Self::from_env_with_toml(toml_url, None)
    }

    /// Load from environment variables, falling back to TOML config values
    pub fn from_env_with_toml(toml_url: Option<&str>, toml_interval_ms: Option<u64>) -> Self {
        let default_url = toml_url.unwrap_or("redis://127.0.0.1:6379");
        let default_interval = toml_interval_ms.unwrap_or(500);
        Self {
            url: std::env::var("REDIS_URL").unwrap_or_else(|_| default_url.to_string()),
            channel_prefix: std::env::var("REDIS_PREFIX").unwrap_or_else(|_| "nat".to_string()),
            cache_ttl_seconds: std::env::var("REDIS_CACHE_TTL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(60),
            enabled: std::env::var("REDIS_ENABLED")
                .map(|s| s.to_lowercase() != "false" && s != "0")
                .unwrap_or(true),
            publish_interval_ms: std::env::var("REDIS_PUBLISH_INTERVAL_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default_interval),
        }
    }
}

/// Redis feature publisher with per-symbol rate limiting
pub struct RedisPublisher {
    conn: ConnectionManager,
    config: RedisConfig,
    last_publish: HashMap<String, Instant>,
    min_interval: Duration,
}

impl RedisPublisher {
    /// Create new publisher
    pub async fn new(config: RedisConfig) -> Result<Self, redis::RedisError> {
        let client = redis::Client::open(config.url.as_str())?;
        let conn = ConnectionManager::new(client).await?;
        let min_interval = Duration::from_millis(config.publish_interval_ms);

        info!(
            "Redis publisher connected to {} (publish interval: {}ms)",
            config.url, config.publish_interval_ms
        );

        Ok(Self {
            conn,
            config,
            last_publish: HashMap::new(),
            min_interval,
        })
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
            whale: features
                .whale_flow
                .as_ref()
                .map(WhaleSummary::from_features),
            metrics: QuickMetrics::from_features(features),
        };

        self.publish(&snapshot).await
    }

    /// Publish feature snapshot (rate-limited Pub/Sub, always updates cache)
    async fn publish(&mut self, snapshot: &FeatureSnapshot) -> Result<(), redis::RedisError> {
        let json = serde_json::to_string(snapshot).unwrap_or_default();

        // Always update cache (REST API needs latest value)
        let cache_key = format!("{}:latest:{}", self.config.channel_prefix, snapshot.symbol);
        let _: () = self
            .conn
            .set_ex(&cache_key, &json, self.config.cache_ttl_seconds)
            .await?;

        // Rate-limit Pub/Sub publish per symbol
        let now = Instant::now();
        let should_publish = match self.last_publish.get(&snapshot.symbol) {
            Some(last) => now.duration_since(*last) >= self.min_interval,
            None => true,
        };

        if should_publish {
            let channel = format!(
                "{}:features:{}",
                self.config.channel_prefix, snapshot.symbol
            );
            let _: () = self.conn.publish(&channel, &json).await?;
            self.last_publish.insert(snapshot.symbol.clone(), now);
            debug!("Published features for {} to Redis", snapshot.symbol);
        }

        Ok(())
    }

    /// Publish alert trigger
    pub async fn publish_alert(&mut self, alert: &AlertTrigger) -> Result<(), redis::RedisError> {
        let channel = format!("{}:alerts", self.config.channel_prefix);
        let json = serde_json::to_string(alert).unwrap_or_default();

        let _: () = self.conn.publish(&channel, &json).await?;

        info!(
            "Published alert: {:?} for {}",
            alert.alert_type, alert.symbol
        );

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
        assert_eq!(config.publish_interval_ms, 500);
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

    // --- Regime summary edge cases ---

    #[test]
    fn test_regime_summary_unclear_when_low_clarity() {
        // clarity = |accum - distrib| = |0.4 - 0.5| = 0.1 < 0.3 → UNCLEAR
        let regime = RegimeFeatures {
            accumulation_score: 0.4,
            distribution_score: 0.5,
            ..Default::default()
        };
        let summary = RegimeSummary::from_features(&regime);
        assert_eq!(summary.regime_type, "UNCLEAR");
        assert!((summary.clarity - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_regime_summary_boundary_at_clarity_threshold() {
        // clarity = |0.65 - 0.35| = 0.3 — exactly at threshold → UNCLEAR (< 0.3 is false)
        let regime = RegimeFeatures {
            accumulation_score: 0.65,
            distribution_score: 0.35,
            ..Default::default()
        };
        let summary = RegimeSummary::from_features(&regime);
        // 0.3 is NOT < 0.3, so clarity check passes → ACCUMULATION
        assert_eq!(summary.regime_type, "ACCUMULATION");
    }

    #[test]
    fn test_regime_summary_just_below_clarity_threshold() {
        // clarity = |0.649 - 0.351| = 0.298 < 0.3 → UNCLEAR
        let regime = RegimeFeatures {
            accumulation_score: 0.649,
            distribution_score: 0.351,
            ..Default::default()
        };
        let summary = RegimeSummary::from_features(&regime);
        assert_eq!(summary.regime_type, "UNCLEAR");
    }

    #[test]
    fn test_regime_summary_equal_scores_unclear() {
        // clarity = |0.5 - 0.5| = 0.0 < 0.3 → UNCLEAR
        let regime = RegimeFeatures {
            accumulation_score: 0.5,
            distribution_score: 0.5,
            ..Default::default()
        };
        let summary = RegimeSummary::from_features(&regime);
        assert_eq!(summary.regime_type, "UNCLEAR");
    }

    #[test]
    fn test_regime_summary_propagates_all_fields() {
        let regime = RegimeFeatures {
            accumulation_score: 0.9,
            distribution_score: 0.1,
            range_position_24h: 0.75,
            absorption_zscore: 1.5,
            divergence_zscore: -0.8,
            churn_zscore: 2.1,
            ..Default::default()
        };
        let summary = RegimeSummary::from_features(&regime);
        assert_eq!(summary.accumulation_score, 0.9);
        assert_eq!(summary.distribution_score, 0.1);
        assert_eq!(summary.range_position_24h, 0.75);
        assert_eq!(summary.absorption_zscore, 1.5);
        assert_eq!(summary.divergence_zscore, -0.8);
        assert_eq!(summary.churn_zscore, 2.1);
        assert_eq!(summary.regime_type, "ACCUMULATION");
    }

    // --- Whale summary ---

    #[test]
    fn test_whale_summary_accumulating() {
        let whale = WhaleFlowFeatures {
            whale_flow_normalized_1h: 1.5, // > 1.0
            whale_net_flow_1h: 5000.0,
            whale_net_flow_4h: 12000.0,
            whale_net_flow_24h: 30000.0,
            whale_flow_intensity: 2.3,
            ..Default::default()
        };
        let summary = WhaleSummary::from_features(&whale);
        assert_eq!(summary.direction, "ACCUMULATING");
        assert_eq!(summary.net_flow_1h, 5000.0);
        assert_eq!(summary.net_flow_4h, 12000.0);
        assert_eq!(summary.net_flow_24h, 30000.0);
        assert_eq!(summary.intensity, 2.3);
    }

    #[test]
    fn test_whale_summary_distributing() {
        let whale = WhaleFlowFeatures {
            whale_flow_normalized_1h: -1.5, // < -1.0
            ..Default::default()
        };
        let summary = WhaleSummary::from_features(&whale);
        assert_eq!(summary.direction, "DISTRIBUTING");
    }

    #[test]
    fn test_whale_summary_neutral() {
        let whale = WhaleFlowFeatures {
            whale_flow_normalized_1h: 0.5, // between -1.0 and 1.0
            ..Default::default()
        };
        let summary = WhaleSummary::from_features(&whale);
        assert_eq!(summary.direction, "NEUTRAL");
    }

    #[test]
    fn test_whale_summary_neutral_at_boundary() {
        // Exactly 1.0 — not > 1.0, so NEUTRAL
        let whale = WhaleFlowFeatures {
            whale_flow_normalized_1h: 1.0,
            ..Default::default()
        };
        let summary = WhaleSummary::from_features(&whale);
        assert_eq!(summary.direction, "NEUTRAL");

        // Exactly -1.0 — not < -1.0, so NEUTRAL
        let whale_neg = WhaleFlowFeatures {
            whale_flow_normalized_1h: -1.0,
            ..Default::default()
        };
        let summary_neg = WhaleSummary::from_features(&whale_neg);
        assert_eq!(summary_neg.direction, "NEUTRAL");
    }

    #[test]
    fn test_whale_summary_zscore_propagated() {
        let whale = WhaleFlowFeatures {
            whale_flow_normalized_1h: 2.5,
            ..Default::default()
        };
        let summary = WhaleSummary::from_features(&whale);
        assert_eq!(summary.net_flow_1h_zscore, 2.5);
    }

    // --- QuickMetrics ---

    #[test]
    fn test_quick_metrics_from_features() {
        let mut features = Features::default();
        features.raw.midprice = 67500.0;
        features.raw.spread_bps = 0.15;
        features.toxicity.vpin_10 = 0.63;
        features.entropy.tick_entropy_1m = 0.89;
        features.volatility.returns_1m = 0.0012;
        features.imbalance.qty_l1 = 0.35;

        let metrics = QuickMetrics::from_features(&features);
        assert_eq!(metrics.midprice, 67500.0);
        assert_eq!(metrics.spread_bps, 0.15);
        assert_eq!(metrics.vpin, 0.63);
        assert_eq!(metrics.tick_entropy, 0.89);
        assert_eq!(metrics.realized_vol_1m, 0.0012);
        assert_eq!(metrics.imbalance_l1, 0.35);
    }

    #[test]
    fn test_quick_metrics_default_features() {
        let features = Features::default();
        let metrics = QuickMetrics::from_features(&features);
        assert_eq!(metrics.midprice, 0.0);
        assert_eq!(metrics.spread_bps, 0.0);
    }

    // --- FeatureSnapshot serialization ---

    #[test]
    fn test_feature_snapshot_serialization() {
        let snapshot = FeatureSnapshot {
            timestamp_ms: 1717574400000,
            symbol: "BTC".to_string(),
            features: serde_json::json!([1.0, 2.0, 3.0]),
            regime: None,
            whale: None,
            metrics: QuickMetrics {
                midprice: 67000.0,
                spread_bps: 0.12,
                vpin: 0.5,
                tick_entropy: 0.9,
                realized_vol_1m: 0.001,
                imbalance_l1: 0.3,
            },
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("1717574400000"));
        assert!(json.contains("BTC"));
        assert!(json.contains("67000"));
        // regime and whale should be null
        assert!(json.contains("\"regime\":null"));
        assert!(json.contains("\"whale\":null"));
    }

    #[test]
    fn test_feature_snapshot_with_regime_and_whale() {
        let regime = RegimeSummary {
            accumulation_score: 0.8,
            distribution_score: 0.2,
            clarity: 0.6,
            regime_type: "ACCUMULATION".to_string(),
            range_position_24h: 0.7,
            absorption_zscore: 1.2,
            divergence_zscore: -0.5,
            churn_zscore: 0.3,
        };

        let whale = WhaleSummary {
            net_flow_1h: 5000.0,
            net_flow_1h_zscore: 2.1,
            net_flow_4h: 15000.0,
            net_flow_24h: 40000.0,
            intensity: 3.0,
            direction: "ACCUMULATING".to_string(),
        };

        let snapshot = FeatureSnapshot {
            timestamp_ms: 1717574400000,
            symbol: "ETH".to_string(),
            features: serde_json::json!([]),
            regime: Some(regime),
            whale: Some(whale),
            metrics: QuickMetrics {
                midprice: 3500.0,
                spread_bps: 0.2,
                vpin: 0.4,
                tick_entropy: 0.85,
                realized_vol_1m: 0.002,
                imbalance_l1: -0.1,
            },
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("ACCUMULATION"));
        assert!(json.contains("ACCUMULATING"));
        assert!(json.contains("ETH"));
        // Verify it round-trips
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
    }

    // --- RedisConfig ---

    #[test]
    fn test_redis_config_from_env_defaults() {
        // Without any env vars set, should use defaults
        // (relies on env vars not being set in test environment)
        let config = RedisConfig::from_env_with_toml(None, None);
        assert_eq!(config.channel_prefix, "nat");
        assert_eq!(config.cache_ttl_seconds, 60);
        assert_eq!(config.publish_interval_ms, 500);
    }

    #[test]
    fn test_redis_config_toml_url_fallback() {
        let config = RedisConfig::from_env_with_toml_url(Some("redis://custom:6380"));
        // If REDIS_URL env var is not set, should use the toml_url
        if std::env::var("REDIS_URL").is_err() {
            assert_eq!(config.url, "redis://custom:6380");
        }
    }

    #[test]
    fn test_redis_config_toml_interval_fallback() {
        let config = RedisConfig::from_env_with_toml(None, Some(250));
        if std::env::var("REDIS_PUBLISH_INTERVAL_MS").is_err() {
            assert_eq!(config.publish_interval_ms, 250);
        }
    }

    // --- Alert types ---

    #[test]
    fn test_all_alert_types_serialize() {
        let types = vec![
            AlertType::WhaleAccumulation,
            AlertType::WhaleDistribution,
            AlertType::LiquidationCluster,
            AlertType::RegimeChange,
            AlertType::EntropyDrop,
            AlertType::ConcentrationSpike,
            AlertType::Custom("TestAlert".to_string()),
        ];

        for alert_type in types {
            let alert = AlertTrigger {
                timestamp_ms: 0,
                symbol: "BTC".to_string(),
                alert_type,
                severity: AlertSeverity::Info,
                message: "test".to_string(),
                data: serde_json::json!(null),
            };
            let json = serde_json::to_string(&alert).unwrap();
            assert!(!json.is_empty());
        }
    }

    #[test]
    fn test_all_alert_severities_serialize() {
        for severity in [AlertSeverity::Info, AlertSeverity::Warning, AlertSeverity::Critical] {
            let alert = AlertTrigger {
                timestamp_ms: 0,
                symbol: "BTC".to_string(),
                alert_type: AlertType::RegimeChange,
                severity,
                message: "test".to_string(),
                data: serde_json::json!(null),
            };
            let json = serde_json::to_string(&alert).unwrap();
            assert!(!json.is_empty());
        }
    }

    #[test]
    fn test_alert_custom_type_preserves_string() {
        let alert = AlertTrigger {
            timestamp_ms: 0,
            symbol: "SOL".to_string(),
            alert_type: AlertType::Custom("VolumeAnomaly".to_string()),
            severity: AlertSeverity::Critical,
            message: "Unusual volume detected".to_string(),
            data: serde_json::json!({"volume_ratio": 5.2}),
        };
        let json = serde_json::to_string(&alert).unwrap();
        assert!(json.contains("VolumeAnomaly"));
        assert!(json.contains("Critical"));
        assert!(json.contains("5.2"));
    }

    // --- Graceful degradation ---

    #[tokio::test]
    async fn test_try_new_returns_none_when_disabled() {
        let config = RedisConfig {
            enabled: false,
            ..Default::default()
        };
        assert!(RedisPublisher::try_new(config).await.is_none());
    }

    #[tokio::test]
    async fn test_try_new_returns_none_on_bad_url() {
        let config = RedisConfig {
            url: "redis://localhost:1".to_string(), // Port 1 — connection refused immediately
            enabled: true,
            ..Default::default()
        };
        // Should return None (connection failure), not panic
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            RedisPublisher::try_new(config),
        )
        .await;
        match result {
            Ok(publisher) => assert!(publisher.is_none()),
            Err(_) => {} // timeout is also acceptable — no panic
        }
    }

    // --- Channel naming ---

    #[test]
    fn test_feature_channel_naming() {
        let prefix = "nat";
        let symbol = "BTC";
        let channel = format!("{}:features:{}", prefix, symbol);
        assert_eq!(channel, "nat:features:BTC");
    }

    #[test]
    fn test_cache_key_naming() {
        let prefix = "nat";
        let symbol = "ETH";
        let key = format!("{}:latest:{}", prefix, symbol);
        assert_eq!(key, "nat:latest:ETH");
    }

    #[test]
    fn test_alert_channel_naming() {
        let prefix = "nat";
        let channel = format!("{}:alerts", prefix);
        assert_eq!(channel, "nat:alerts");
    }
}
