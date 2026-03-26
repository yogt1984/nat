//! Redis Client Wrapper
//!
//! Provides async Redis operations for the API server.

use redis::aio::ConnectionManager;
use redis::{AsyncCommands, Client};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Feature snapshot from Redis cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSnapshot {
    pub timestamp_ms: u64,
    pub symbol: String,
    pub features: serde_json::Value,
    pub regime: Option<serde_json::Value>,
    pub whale: Option<serde_json::Value>,
    pub metrics: Option<serde_json::Value>,
}

/// Redis client for API server
pub struct RedisClient {
    conn: ConnectionManager,
    prefix: String,
}

impl RedisClient {
    /// Create new Redis client
    pub async fn new(url: &str) -> Result<Self, redis::RedisError> {
        let client = Client::open(url)?;
        let conn = ConnectionManager::new(client).await?;

        info!("Redis client connected to {}", url);

        Ok(Self {
            conn,
            prefix: "nat".to_string(),
        })
    }

    /// Get latest feature snapshot for a symbol
    pub async fn get_latest_features(
        &self,
        symbol: &str,
    ) -> Result<Option<FeatureSnapshot>, redis::RedisError> {
        let key = format!("{}:latest:{}", self.prefix, symbol);
        let mut conn = self.conn.clone();

        let json: Option<String> = conn.get(&key).await?;

        match json {
            Some(data) => {
                debug!("Got cached features for {}", symbol);
                match serde_json::from_str(&data) {
                    Ok(snapshot) => Ok(Some(snapshot)),
                    Err(e) => {
                        tracing::warn!("Failed to parse cached features: {}", e);
                        Ok(None)
                    }
                }
            }
            None => Ok(None),
        }
    }

    /// Get list of symbols with cached data
    pub async fn get_active_symbols(&self) -> Result<Vec<String>, redis::RedisError> {
        let pattern = format!("{}:latest:*", self.prefix);
        let mut conn = self.conn.clone();

        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&pattern)
            .query_async(&mut conn)
            .await?;

        let symbols: Vec<String> = keys
            .into_iter()
            .filter_map(|k| {
                k.strip_prefix(&format!("{}:latest:", self.prefix))
                    .map(|s| s.to_string())
            })
            .collect();

        Ok(symbols)
    }

    /// Create a pub/sub subscription
    pub async fn subscribe(&self, channel: &str) -> Result<redis::aio::PubSub, redis::RedisError> {
        let client = redis::Client::open(format!(
            "{}",
            std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
        ))?;
        let mut pubsub = client.get_async_pubsub().await?;
        pubsub.subscribe(channel).await?;
        Ok(pubsub)
    }

    /// Get connection for pub/sub operations
    pub fn get_connection(&self) -> ConnectionManager {
        self.conn.clone()
    }
}
