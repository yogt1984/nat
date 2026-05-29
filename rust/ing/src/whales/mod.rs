//! Whale identification and tracking module
//!
//! Identifies large traders ("whales") based on position size and trading activity,
//! tracks their behavior, and computes concentration metrics.

mod classifier;
mod metrics;
mod registry;

pub use classifier::*;
pub use metrics::*;
pub use registry::*;
