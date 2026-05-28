//! Regime Buffer - Coordinator for Regime Feature Computation
//!
//! Aggregates minute-bar data and computes all regime features.
//! Fed by the main ingestor at minute intervals.

use super::{
    AbsorptionComputer, ChurnComputer, DivergenceComputer, RangeComputer, RegimeFeatures,
    compute_accumulation_score, compute_distribution_score,
};

/// Configuration for regime feature computation.
#[derive(Debug, Clone)]
pub struct RegimeConfig {
    /// Window sizes for regime features (minutes)
    pub regime_windows: Vec<usize>,
    /// Window sizes for range features (minutes)
    pub range_windows: Vec<usize>,
    /// History size for z-score computation
    pub history_size: usize,
    /// Window for Kyle's lambda estimation
    pub lambda_window: usize,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            // 1h, 4h, 24h in minutes
            regime_windows: vec![60, 240, 1440],
            // 4h, 24h, 1 week in minutes
            range_windows: vec![240, 1440, 10080],
            // 1 week of minute data
            history_size: 10080,
            // 1 hour for lambda estimation
            lambda_window: 60,
        }
    }
}

/// Coordinates all regime feature computation.
#[derive(Debug, Clone)]
pub struct RegimeBuffer {
    /// Absorption ratio computer
    absorption: AbsorptionComputer,
    /// Volume-price divergence computer
    divergence: DivergenceComputer,
    /// Churn rate computer
    churn: ChurnComputer,
    /// Range position computer
    range: RangeComputer,
    /// Configuration
    config: RegimeConfig,
    /// Minutes processed
    minutes_processed: u64,
}

impl RegimeBuffer {
    /// Create new regime buffer with configuration.
    pub fn new(config: RegimeConfig) -> Self {
        // Use largest window from config
        let max_regime = *config.regime_windows.iter().max().unwrap_or(&1440);
        let max_range = *config.range_windows.iter().max().unwrap_or(&10080);
        let max_window = max_regime.max(max_range);

        Self {
            absorption: AbsorptionComputer::new(max_window, config.history_size),
            divergence: DivergenceComputer::new(max_window, config.history_size, config.lambda_window),
            churn: ChurnComputer::new(max_window, config.history_size),
            range: RangeComputer::new(max_window),
            config,
            minutes_processed: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RegimeConfig::default())
    }

    /// Update with new minute bar data.
    ///
    /// # Arguments
    /// * `price` - Closing price of the minute bar
    /// * `volume` - Total volume in the minute bar
    /// * `buy_volume` - Buy-side volume
    /// * `sell_volume` - Sell-side volume
    pub fn update(&mut self, price: f64, volume: f64, buy_volume: f64, sell_volume: f64) {
        self.absorption.update(price, volume);
        self.divergence.update(price, buy_volume, sell_volume);
        self.churn.update(buy_volume, sell_volume);
        self.range.update(price);
        self.minutes_processed += 1;
    }

    /// Compute all regime features.
    pub fn compute(&self) -> RegimeFeatures {
        let windows = &self.config.regime_windows;
        let range_windows = &self.config.range_windows;

        // Absorption at multiple windows
        let absorption_1h = self.absorption.compute(windows.get(0).copied().unwrap_or(60));
        let absorption_4h = self.absorption.compute(windows.get(1).copied().unwrap_or(240));
        let absorption_24h = self.absorption.compute(windows.get(2).copied().unwrap_or(1440));
        let absorption_zscore = self.absorption.compute_zscore(windows.get(2).copied().unwrap_or(1440));

        // Divergence at multiple windows
        let divergence_1h = self.divergence.compute(windows.get(0).copied().unwrap_or(60));
        let divergence_4h = self.divergence.compute(windows.get(1).copied().unwrap_or(240));
        let divergence_24h = self.divergence.compute(windows.get(2).copied().unwrap_or(1440));
        let divergence_zscore = self.divergence.compute_zscore(windows.get(2).copied().unwrap_or(1440));
        let kyle_lambda = self.divergence.get_kyle_lambda();

        // Churn at multiple windows
        let churn_1h = self.churn.compute(windows.get(0).copied().unwrap_or(60));
        let churn_4h = self.churn.compute(windows.get(1).copied().unwrap_or(240));
        let churn_24h = self.churn.compute(windows.get(2).copied().unwrap_or(1440));
        let churn_zscore = self.churn.compute_zscore(windows.get(2).copied().unwrap_or(1440));

        // Range position at multiple windows
        let range_position_4h = self.range.compute_position(range_windows.get(0).copied().unwrap_or(240));
        let range_position_24h = self.range.compute_position(range_windows.get(1).copied().unwrap_or(1440));
        let range_position_1w = self.range.compute_position(range_windows.get(2).copied().unwrap_or(10080));
        let range_width_24h = self.range.compute_width(range_windows.get(1).copied().unwrap_or(1440));

        // Composite scores using 24h metrics
        let accumulation_score = compute_accumulation_score(
            absorption_zscore,
            divergence_zscore,
            churn_zscore,
            range_position_24h,
        );
        let distribution_score = compute_distribution_score(
            absorption_zscore,
            divergence_zscore,
            churn_zscore,
            range_position_24h,
        );

        RegimeFeatures {
            absorption_1h,
            absorption_4h,
            absorption_24h,
            absorption_zscore,
            divergence_1h,
            divergence_4h,
            divergence_24h,
            divergence_zscore,
            kyle_lambda,
            churn_1h,
            churn_4h,
            churn_24h,
            churn_zscore,
            range_position_4h,
            range_position_24h,
            range_position_1w,
            range_width_24h,
            accumulation_score,
            distribution_score,
        }
    }

    /// Check if enough data for full computation.
    pub fn is_ready(&self) -> bool {
        let min_window = *self.config.regime_windows.iter().min().unwrap_or(&60);
        self.absorption.is_ready(min_window)
    }

    /// Check if ready for specific window.
    pub fn is_ready_for(&self, window: usize) -> bool {
        self.absorption.is_ready(window)
    }

    /// Get minutes processed.
    pub fn minutes_processed(&self) -> u64 {
        self.minutes_processed
    }

    /// Get current buffer length.
    pub fn len(&self) -> usize {
        self.absorption.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.absorption.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_buffer_basic() {
        let mut buffer = RegimeBuffer::with_defaults();

        // Simulate some minute bars
        for i in 0..100 {
            let price = 100.0 + (i as f64 * 0.1).sin();
            let volume = 1000.0 + (i as f64 * 0.05).cos() * 100.0;
            let buy_vol = volume * 0.55;
            let sell_vol = volume * 0.45;

            buffer.update(price, volume, buy_vol, sell_vol);
        }

        assert_eq!(buffer.minutes_processed(), 100);
        assert!(buffer.is_ready_for(60));
        assert!(!buffer.is_ready_for(200)); // Not enough data yet

        let features = buffer.compute();
        assert!(features.absorption_1h > 0.0);
        assert!(features.churn_1h > 0.0);
    }

    #[test]
    fn test_accumulation_detection() {
        let mut buffer = RegimeBuffer::with_defaults();

        // Simulate accumulation pattern:
        // - High buy volume
        // - Price stays flat (absorbed)
        for _ in 0..1500 {
            let price = 100.0; // Flat price
            let buy_vol = 2000.0; // Heavy buying
            let sell_vol = 500.0;
            let volume = buy_vol + sell_vol;

            buffer.update(price, volume, buy_vol, sell_vol);
        }

        let features = buffer.compute();

        // Absorption should be very high (lots of volume, no price movement)
        assert!(features.absorption_24h > 1_000_000.0);

        // Range position should be neutral (price hasn't moved)
        // Note: With flat prices, range is tiny so position may vary
    }

    #[test]
    fn test_feature_count() {
        let features = RegimeFeatures::default();
        assert_eq!(features.to_vec().len(), RegimeFeatures::count());
    }
}
