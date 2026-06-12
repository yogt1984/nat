//! Shared configuration types

use serde::Deserialize;

fn default_emission_interval() -> u64 {
    100
}
fn default_trade_buffer_seconds() -> u64 {
    60
}
fn default_book_levels() -> usize {
    10
}
fn default_price_buffer_size() -> usize {
    1000
}
fn default_whale_threshold_usd() -> f64 {
    100_000.0
}

/// Configuration for trade-size-based whale flow detection
#[derive(Debug, Clone, Deserialize)]
pub struct WhaleFlowTradeConfig {
    /// Minimum trade notional (USD) to classify as whale activity
    #[serde(default = "default_whale_threshold_usd")]
    pub whale_threshold_usd: f64,
}

impl Default for WhaleFlowTradeConfig {
    fn default() -> Self {
        Self {
            whale_threshold_usd: default_whale_threshold_usd(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct FeaturesConfig {
    #[serde(default = "default_emission_interval")]
    pub emission_interval_ms: u64,
    #[serde(default = "default_trade_buffer_seconds")]
    pub trade_buffer_seconds: u64,
    #[serde(default = "default_book_levels")]
    pub book_levels: usize,
    #[serde(default = "default_price_buffer_size")]
    pub price_buffer_size: usize,
    #[serde(default)]
    pub gmm_model_path: Option<String>,
    /// Trade-size whale flow detection (enabled when present)
    #[serde(default)]
    pub whale_flow: Option<WhaleFlowTradeConfig>,
}
