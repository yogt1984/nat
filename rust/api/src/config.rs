//! API Server Configuration

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub redis_url: String,
    /// Root directory for research output JSON files (hypotheses, cycles).
    /// Used as fallback if research_db_path is not available.
    pub research_data_dir: String,
    /// Root directory for IT engine state files (MI, CMI, interaction info).
    pub it_engine_data_dir: String,
    /// Path to nat.db SQLite database (primary research data source).
    pub research_db_path: String,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 3000,
            redis_url: "redis://127.0.0.1:6379".to_string(),
            research_data_dir: "../data/research".to_string(),
            it_engine_data_dir: "../data/it_engine".to_string(),
            research_db_path: "../data/nat.db".to_string(),
        }
    }
}

impl ApiConfig {
    pub fn from_env() -> Self {
        Self::from_env_with_defaults(None)
    }

    pub fn from_env_with_defaults(redis_url_default: Option<&str>) -> Self {
        Self {
            host: std::env::var("NAT_API_HOST")
                .unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: std::env::var("NAT_API_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3000),
            redis_url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| redis_url_default.unwrap_or("redis://127.0.0.1:6379").to_string()),
            research_data_dir: std::env::var("NAT_RESEARCH_DIR")
                .unwrap_or_else(|_| "../data/research".to_string()),
            it_engine_data_dir: std::env::var("NAT_IT_ENGINE_DIR")
                .unwrap_or_else(|_| "../data/it_engine".to_string()),
            research_db_path: std::env::var("NAT_RESEARCH_DB")
                .unwrap_or_else(|_| "../data/nat.db".to_string()),
        }
    }
}
