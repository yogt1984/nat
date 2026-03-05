//! Whale identification and tracking module
//!
//! Identifies large traders ("whales") based on position size and trading activity,
//! tracks their behavior, and computes concentration metrics.

mod classifier;
mod registry;
mod metrics;

pub use classifier::*;
pub use registry::*;
pub use metrics::*;
