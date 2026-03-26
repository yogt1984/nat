//! Application State

use crate::config::ApiConfig;
use crate::redis_client::RedisClient;

/// Shared application state
pub struct AppState {
    pub redis: RedisClient,
    pub config: ApiConfig,
}

impl AppState {
    pub fn new(redis: RedisClient, config: ApiConfig) -> Self {
        Self { redis, config }
    }
}
