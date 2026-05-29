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
        let client = redis::Client::open(
            std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string()),
        )?;
        let mut pubsub = client.get_async_pubsub().await?;
        pubsub.subscribe(channel).await?;
        Ok(pubsub)
    }

    /// Get connection for pub/sub operations
    pub fn get_connection(&self) -> ConnectionManager {
        self.conn.clone()
    }

    /// Ensure a consumer group exists for a stream (create if missing).
    /// Uses MKSTREAM to create the stream itself if it doesn't exist.
    pub async fn ensure_consumer_group(
        &self,
        stream: &str,
        group: &str,
    ) -> Result<(), redis::RedisError> {
        let mut conn = self.conn.clone();
        // XGROUP CREATE <stream> <group> $ MKSTREAM — idempotent via BUSYGROUP check
        let result: redis::RedisResult<()> = redis::cmd("XGROUP")
            .arg("CREATE")
            .arg(stream)
            .arg(group)
            .arg("$")
            .arg("MKSTREAM")
            .query_async(&mut conn)
            .await;
        match result {
            Ok(_) => Ok(()),
            Err(e) if e.to_string().contains("BUSYGROUP") => Ok(()), // already exists
            Err(e) => Err(e),
        }
    }

    /// Read new messages from a stream consumer group (blocking, timeout_ms=0 blocks forever).
    /// Returns Vec of (message_id, payload_field_value).
    pub async fn xread_group(
        &self,
        group: &str,
        consumer: &str,
        stream: &str,
        count: usize,
        block_ms: usize,
    ) -> Result<Vec<(String, String)>, redis::RedisError> {
        let mut conn = self.conn.clone();
        let result: redis::Value = redis::cmd("XREADGROUP")
            .arg("GROUP")
            .arg(group)
            .arg(consumer)
            .arg("COUNT")
            .arg(count)
            .arg("BLOCK")
            .arg(block_ms)
            .arg("STREAMS")
            .arg(stream)
            .arg(">")
            .query_async(&mut conn)
            .await?;

        Ok(parse_xread_response(result))
    }

    /// Acknowledge messages after processing.
    pub async fn xack(
        &self,
        stream: &str,
        group: &str,
        ids: &[String],
    ) -> Result<(), redis::RedisError> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.clone();
        let mut cmd = redis::cmd("XACK");
        cmd.arg(stream).arg(group);
        for id in ids {
            cmd.arg(id.as_str());
        }
        cmd.query_async(&mut conn).await
    }
}

/// Parse XREADGROUP response into (id, event_payload) pairs.
pub fn parse_xread_response(value: redis::Value) -> Vec<(String, String)> {
    let mut results = Vec::new();
    // Response shape: [[stream_name, [[id, [field, value, ...]], ...]]]
    if let redis::Value::Bulk(streams) = value {
        for stream_entry in streams {
            if let redis::Value::Bulk(parts) = stream_entry {
                if parts.len() >= 2 {
                    if let redis::Value::Bulk(messages) = &parts[1] {
                        for msg in messages {
                            if let redis::Value::Bulk(msg_parts) = msg {
                                if msg_parts.len() >= 2 {
                                    let id = match &msg_parts[0] {
                                        redis::Value::Data(b) => {
                                            String::from_utf8_lossy(b).to_string()
                                        }
                                        _ => continue,
                                    };
                                    // Fields: [field_name, field_value, ...]
                                    if let redis::Value::Bulk(fields) = &msg_parts[1] {
                                        // We only have one field "event"
                                        if fields.len() >= 2 {
                                            if let redis::Value::Data(b) = &fields[1] {
                                                let payload =
                                                    String::from_utf8_lossy(b).to_string();
                                                results.push((id, payload));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    results
}
