//! Microstructure Algorithm Framework
//!
//! Pluggable algorithms that compute derived features from base features.
//! Each algorithm declares its output features and computes them tick-by-tick.
//!
//! Currently all implementations are **dummies** (return NaN). Real logic
//! is developed in Python first; once validated, the Rust impl replaces the stub.

pub mod regime_gated;
pub mod kalman_imbalance;
pub mod multi_level_imb;

use crate::features::Features;

/// Descriptor for a single algorithm-derived feature
pub struct AlgFeatureDesc {
    pub name: &'static str,
    pub warmup_ticks: usize,
}

/// Trait for microstructure algorithms that compute derived features.
///
/// Mirrors the Python `MicrostructureAlgorithm` ABC in `scripts/algorithms/base.py`.
/// The `step()` method receives the full base `Features` struct and returns
/// a fixed-length `Vec<f64>` of algorithm-specific derived features.
pub trait MicrostructureAlgorithm: Send {
    /// Unique algorithm name (used in config and column prefixes)
    fn name(&self) -> &'static str;

    /// Descriptors for each output feature
    fn alg_feature_descs(&self) -> &'static [AlgFeatureDesc];

    /// Number of output features
    fn count(&self) -> usize {
        self.alg_feature_descs().len()
    }

    /// Output feature names (for parquet schema)
    fn names(&self) -> Vec<&'static str> {
        self.alg_feature_descs().iter().map(|d| d.name).collect()
    }

    /// Compute algorithm features from base features.
    /// Must return exactly `count()` values.
    fn step(&mut self, features: &Features) -> Vec<f64>;

    /// Reset internal state
    fn reset(&mut self);
}

/// Build algorithm instances from config
pub fn create_algorithms(enabled: &[String]) -> Vec<Box<dyn MicrostructureAlgorithm>> {
    let mut algs: Vec<Box<dyn MicrostructureAlgorithm>> = Vec::new();
    for name in enabled {
        match name.as_str() {
            "regime_gated" => algs.push(Box::new(regime_gated::RegimeGated::new())),
            "kalman_imbalance" => algs.push(Box::new(kalman_imbalance::KalmanImbalance::new())),
            "multi_level_imb" => algs.push(Box::new(multi_level_imb::MultiLevelImbalance::new())),
            other => tracing::warn!("Unknown algorithm '{}', skipping", other),
        }
    }
    algs
}

/// Total count of all algorithm features across all enabled algorithms
pub fn total_alg_feature_count(algs: &[Box<dyn MicrostructureAlgorithm>]) -> usize {
    algs.iter().map(|a| a.count()).sum()
}

/// All algorithm feature names (for parquet schema extension)
pub fn all_alg_feature_names(algs: &[Box<dyn MicrostructureAlgorithm>]) -> Vec<&'static str> {
    algs.iter().flat_map(|a| a.names()).collect()
}

/// Run all algorithms on a feature vector, collecting outputs into a single Vec
pub fn run_all(
    algs: &mut [Box<dyn MicrostructureAlgorithm>],
    features: &Features,
) -> Vec<f64> {
    algs.iter_mut().flat_map(|a| a.step(features)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_algorithms_known() {
        let algs = create_algorithms(&[
            "regime_gated".into(),
            "kalman_imbalance".into(),
            "multi_level_imb".into(),
        ]);
        assert_eq!(algs.len(), 3);
    }

    #[test]
    fn test_create_algorithms_unknown_skipped() {
        let algs = create_algorithms(&["regime_gated".into(), "nonexistent".into()]);
        assert_eq!(algs.len(), 1);
    }

    #[test]
    fn test_total_feature_count() {
        let algs = create_algorithms(&[
            "regime_gated".into(),
            "kalman_imbalance".into(),
            "multi_level_imb".into(),
        ]);
        // regime_gated: 3, kalman: 4, multi_level: 3
        assert_eq!(total_alg_feature_count(&algs), 10);
    }

    #[test]
    fn test_all_names_unique() {
        let algs = create_algorithms(&[
            "regime_gated".into(),
            "kalman_imbalance".into(),
            "multi_level_imb".into(),
        ]);
        let names = all_alg_feature_names(&algs);
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            assert!(seen.insert(name), "Duplicate alg feature name: {}", name);
        }
    }

    #[test]
    fn test_run_all_returns_nan_dummies() {
        let mut algs = create_algorithms(&[
            "regime_gated".into(),
            "kalman_imbalance".into(),
            "multi_level_imb".into(),
        ]);
        let features = Features::default();
        let values = run_all(&mut algs, &features);
        assert_eq!(values.len(), 10);
        assert!(values.iter().all(|v| v.is_nan()), "Dummy impls should return NaN");
    }
}
