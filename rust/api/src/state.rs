//! Application State

use crate::config::ApiConfig;
use crate::redis_client::RedisClient;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Cached research data — avoids re-reading JSON from disk on every request.
pub struct ResearchCache {
    pub hypotheses: Vec<serde_json::Value>,
    pub cycles: Vec<serde_json::Value>,
    pub loaded_at: Instant,
    pub ttl: Duration,
}

impl ResearchCache {
    pub fn new(ttl: Duration) -> Self {
        Self {
            hypotheses: Vec::new(),
            cycles: Vec::new(),
            loaded_at: Instant::now() - ttl - Duration::from_secs(1), // force initial load
            ttl,
        }
    }

    pub fn is_stale(&self) -> bool {
        self.loaded_at.elapsed() > self.ttl
    }
}

/// Shared application state
pub struct AppState {
    pub redis: RedisClient,
    pub config: ApiConfig,
    pub research_cache: Arc<RwLock<ResearchCache>>,
}

impl AppState {
    pub fn new(redis: RedisClient, config: ApiConfig) -> Self {
        Self {
            redis,
            config,
            research_cache: Arc::new(RwLock::new(ResearchCache::new(Duration::from_secs(30)))),
        }
    }
}
