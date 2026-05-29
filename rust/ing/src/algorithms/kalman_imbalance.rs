//! Kalman-Filtered Imbalance Algorithm (DUMMY)
//!
//! OU Kalman filter on L1 imbalance, extracting slow component.
//! Real logic in Python: `scripts/algorithms/kalman_imbalance.py`

use super::{AlgFeatureDesc, MicrostructureAlgorithm};
use crate::features::Features;

static DESCS: &[AlgFeatureDesc] = &[
    AlgFeatureDesc {
        name: "alg_kalman_filtered_imb",
        warmup_ticks: 50,
    },
    AlgFeatureDesc {
        name: "alg_kalman_uncertainty",
        warmup_ticks: 50,
    },
    AlgFeatureDesc {
        name: "alg_kalman_innovation",
        warmup_ticks: 50,
    },
    AlgFeatureDesc {
        name: "alg_kalman_signal_strength",
        warmup_ticks: 50,
    },
];

pub struct KalmanImbalance {
    _tick_count: u64,
}

impl Default for KalmanImbalance {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanImbalance {
    pub fn new() -> Self {
        Self { _tick_count: 0 }
    }
}

impl MicrostructureAlgorithm for KalmanImbalance {
    fn name(&self) -> &'static str {
        "kalman_imbalance"
    }
    fn alg_feature_descs(&self) -> &'static [AlgFeatureDesc] {
        DESCS
    }

    fn step(&mut self, _features: &Features) -> Vec<f64> {
        self._tick_count += 1;
        vec![f64::NAN; self.count()]
    }

    fn reset(&mut self) {
        self._tick_count = 0;
    }
}
