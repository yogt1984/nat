//! Machine Learning Module
//!
//! Provides real-time regime classification and ML inference capabilities.

pub mod regime;

pub use regime::{GmmClassificationFeatures, GmmParams, Regime, RegimeClassifier, RegimeFeatures};
