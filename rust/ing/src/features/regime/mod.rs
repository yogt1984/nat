//! Regime Detection Features
//!
//! Low-frequency features for detecting market regimes (accumulation, distribution,
//! markup, markdown). These operate on minute-scale data rather than tick data.
//!
//! # Features
//! - **Absorption**: Volume per unit price change (high = accumulation/distribution)
//! - **Divergence**: Deviation from expected price impact (Kyle's lambda)
//! - **Churn**: Two-sided trading activity (high = position transfer)
//! - **Range Position**: Location within recent price range
//!
//! # Time Windows
//! - Regime features: 1h (60), 4h (240), 24h (1440) minutes
//! - Range features: 4h (240), 24h (1440), 1 week (10080) minutes

mod absorption;
mod buffer;
mod churn;
mod composite;
mod divergence;
mod range;

pub use absorption::AbsorptionComputer;
pub use buffer::{RegimeBuffer, RegimeConfig};
pub use churn::ChurnComputer;
pub use composite::{compute_accumulation_score, compute_distribution_score};
pub use divergence::DivergenceComputer;
pub use range::RangeComputer;

/// Regime detection features computed at minute-scale resolution.
///
/// These features are designed to detect accumulation/distribution phases
/// that unfold over hours to days, distinct from microstructure features.
#[derive(Debug, Clone, Default)]
pub struct RegimeFeatures {
    // === Absorption Ratio ===
    /// Absorption ratio (1 hour window)
    pub absorption_1h: f64,
    /// Absorption ratio (4 hour window)
    pub absorption_4h: f64,
    /// Absorption ratio (24 hour window)
    pub absorption_24h: f64,
    /// Absorption z-score (vs 1-week history)
    pub absorption_zscore: f64,

    // === Volume-Price Divergence ===
    /// Volume-price divergence (1 hour)
    pub divergence_1h: f64,
    /// Volume-price divergence (4 hour)
    pub divergence_4h: f64,
    /// Volume-price divergence (24 hour)
    pub divergence_24h: f64,
    /// Divergence z-score
    pub divergence_zscore: f64,
    /// Kyle's lambda estimate
    pub kyle_lambda: f64,

    // === Churn Rate ===
    /// Churn rate (1 hour)
    pub churn_1h: f64,
    /// Churn rate (4 hour)
    pub churn_4h: f64,
    /// Churn rate (24 hour)
    pub churn_24h: f64,
    /// Churn z-score
    pub churn_zscore: f64,

    // === Range Position ===
    /// Position in 4-hour range [0, 1]
    pub range_position_4h: f64,
    /// Position in 24-hour range [0, 1]
    pub range_position_24h: f64,
    /// Position in 1-week range [0, 1]
    pub range_position_1w: f64,
    /// Range width (24h) as percentage
    pub range_width_24h: f64,

    // === Composite Scores ===
    /// Accumulation score [0, 1]
    pub accumulation_score: f64,
    /// Distribution score [0, 1]
    pub distribution_score: f64,
}

impl RegimeFeatures {
    /// Number of regime features
    pub fn count() -> usize {
        20
    }

    /// Feature names for serialization
    pub fn names() -> Vec<&'static str> {
        vec![
            "regime_absorption_1h",
            "regime_absorption_4h",
            "regime_absorption_24h",
            "regime_absorption_zscore",
            "regime_divergence_1h",
            "regime_divergence_4h",
            "regime_divergence_24h",
            "regime_divergence_zscore",
            "regime_kyle_lambda",
            "regime_churn_1h",
            "regime_churn_4h",
            "regime_churn_24h",
            "regime_churn_zscore",
            "regime_range_pos_4h",
            "regime_range_pos_24h",
            "regime_range_pos_1w",
            "regime_range_width_24h",
            "regime_accumulation_score",
            "regime_distribution_score",
            "regime_clarity",
        ]
    }

    /// Convert to vector of f64
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.absorption_1h,
            self.absorption_4h,
            self.absorption_24h,
            self.absorption_zscore,
            self.divergence_1h,
            self.divergence_4h,
            self.divergence_24h,
            self.divergence_zscore,
            self.kyle_lambda,
            self.churn_1h,
            self.churn_4h,
            self.churn_24h,
            self.churn_zscore,
            self.range_position_4h,
            self.range_position_24h,
            self.range_position_1w,
            self.range_width_24h,
            self.accumulation_score,
            self.distribution_score,
            self.regime_clarity(),
        ]
    }

    /// Regime clarity: how distinct is the current regime signal
    pub fn regime_clarity(&self) -> f64 {
        (self.accumulation_score - self.distribution_score).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_features_count() {
        assert_eq!(RegimeFeatures::count(), 20);
        assert_eq!(RegimeFeatures::names().len(), 20);
        assert_eq!(RegimeFeatures::default().to_vec().len(), 20);
    }
}
