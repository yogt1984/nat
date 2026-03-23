//! Machine Learning Module
//!
//! Provides real-time regime classification and ML inference capabilities.

pub mod regime;

pub use regime::{Regime, RegimeClassifier, RegimeFeatures, GmmParams};
