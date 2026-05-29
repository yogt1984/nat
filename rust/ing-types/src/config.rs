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
}
