//! Shared data types for the NAT ingestor.
//!
//! This crate contains pure data types with no external dependencies beyond serde.
//! It exists to enable incremental compilation and future crate splitting:
//! `ing-types` → `ing-features` → `ing` (binary).

pub mod config;
pub mod messages;
pub mod regime;
pub mod ring_buffer;
pub mod state;

pub use config::FeaturesConfig;
pub use messages::*;
pub use regime::{Regime, RegimeFeatures, GmmClassificationFeatures};
pub use ring_buffer::RingBuffer;
pub use state::{OrderBook, TradeBuffer, Trade, MarketContext};
