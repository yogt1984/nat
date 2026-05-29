//! Feature Computation Module — re-exports from `ing-features` crate.
//!
//! The feature extraction logic lives in `ing-features`. This module provides
//! backward-compatible re-exports so that `crate::features::*` paths continue
//! to work throughout the `ing` crate.

// Re-export everything from the extracted crate
pub use ing_features::*;

// Re-export GMM classifier types that live in ing::ml
pub use crate::ml::regime::{
    GmmParams, RegimeClassifier, RegimeFeatures as GmmClassificationFeatures,
};
