//! Multi-Level Imbalance Algorithm (DUMMY)
//!
//! Weighted combination of L1/L5/L10 order book imbalance.
//! Real logic in Python: `scripts/algorithms/multi_level_imb.py`

use super::{AlgFeatureDesc, MicrostructureAlgorithm};
use crate::features::Features;

static DESCS: &[AlgFeatureDesc] = &[
    AlgFeatureDesc { name: "alg_composite_imbalance", warmup_ticks: 10 },
    AlgFeatureDesc { name: "alg_l1_l5_divergence", warmup_ticks: 10 },
    AlgFeatureDesc { name: "alg_depth_agreement", warmup_ticks: 10 },
];

pub struct MultiLevelImbalance {
    _tick_count: u64,
}

impl MultiLevelImbalance {
    pub fn new() -> Self {
        Self { _tick_count: 0 }
    }
}

impl MicrostructureAlgorithm for MultiLevelImbalance {
    fn name(&self) -> &'static str { "multi_level_imb" }
    fn alg_feature_descs(&self) -> &'static [AlgFeatureDesc] { DESCS }

    fn step(&mut self, _features: &Features) -> Vec<f64> {
        self._tick_count += 1;
        vec![f64::NAN; self.count()]
    }

    fn reset(&mut self) {
        self._tick_count = 0;
    }
}
