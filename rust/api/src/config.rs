//! API Server Configuration

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub redis_url: String,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 3000,
            redis_url: "redis://127.0.0.1:6379".to_string(),
        }
    }
}

impl ApiConfig {
    pub fn from_env() -> Self {
        Self {
            host: std::env::var("NAT_API_HOST")
                .unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: std::env::var("NAT_API_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3000),
            redis_url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string()),
        }
    }
}
