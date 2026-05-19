//! Regime-Gated Imbalance Algorithm (DUMMY)
//!
//! Gates raw L1 imbalance by `ent_book_shape` percentile.
//! Real logic in Python: `scripts/algorithms/regime_gated.py`

use super::{AlgFeatureDesc, MicrostructureAlgorithm};
use crate::features::Features;

static DESCS: &[AlgFeatureDesc] = &[
    AlgFeatureDesc { name: "alg_regime_gated_imbalance", warmup_ticks: 100 },
    AlgFeatureDesc { name: "alg_regime_gate_active", warmup_ticks: 100 },
    AlgFeatureDesc { name: "alg_regime_zscore", warmup_ticks: 100 },
];

pub struct RegimeGated {
    _tick_count: u64,
}

impl RegimeGated {
    pub fn new() -> Self {
        Self { _tick_count: 0 }
    }
}

impl MicrostructureAlgorithm for RegimeGated {
    fn name(&self) -> &'static str { "regime_gated" }
    fn alg_feature_descs(&self) -> &'static [AlgFeatureDesc] { DESCS }

    fn step(&mut self, _features: &Features) -> Vec<f64> {
        self._tick_count += 1;
        vec![f64::NAN; self.count()]
    }

    fn reset(&mut self) {
        self._tick_count = 0;
    }
}
