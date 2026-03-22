//! ING Library - Hyperliquid Market Data Ingestor
//!
//! Core library providing market data ingestion, feature extraction,
//! and hypothesis testing capabilities.

pub mod config;
pub mod dashboard;
pub mod error;
pub mod features;
pub mod hypothesis;
pub mod metrics;
pub mod output;
pub mod positions;
pub mod rest;
pub mod state;
pub mod whales;
pub mod ws;

/// Feature vector emitted at regular intervals
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub timestamp_ns: i64,
    pub symbol: String,
    pub sequence_id: u64,
    pub features: features::Features,
}
