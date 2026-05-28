//! Shared data types for the NAT ingestor.
//!
//! This crate contains pure data types with no external dependencies beyond serde.
//! It exists to enable incremental compilation and future crate splitting:
//! `ing-types` → `ing-features` → `ing` (binary).

pub mod messages;
pub mod ring_buffer;

pub use messages::*;
pub use ring_buffer::RingBuffer;
