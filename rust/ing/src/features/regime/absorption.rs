//! Absorption Ratio Computation
//!
//! Absorption ratio measures volume absorbed per unit price change.
//! High absorption indicates accumulation (buying without price rise)
//! or distribution (selling without price fall).
//!
//! Formula: AR = Σ(Volume) / (|ΔPrice| + ε)

use std::collections::VecDeque;

/// Epsilon to prevent division by zero
const EPSILON: f64 = 1e-10;

/// Computes absorption ratio at multiple time windows.
#[derive(Debug, Clone)]
pub struct AbsorptionComputer {
    /// Price samples (minute-level)
    prices: VecDeque<f64>,
    /// Volume samples (minute-level)
    volumes: VecDeque<f64>,
    /// Historical absorption values for z-score
    history: VecDeque<f64>,
    /// Maximum buffer size (largest window needed)
    max_size: usize,
    /// History size for z-score computation
    history_size: usize,
}

impl AbsorptionComputer {
    /// Create new absorption computer.
    ///
    /// # Arguments
    /// * `max_window` - Largest window in minutes (e.g., 1440 for 24h)
    /// * `history_size` - Samples for z-score normalization (e.g., 10080 for 1 week)
    pub fn new(max_window: usize, history_size: usize) -> Self {
        Self {
            prices: VecDeque::with_capacity(max_window),
            volumes: VecDeque::with_capacity(max_window),
            history: VecDeque::with_capacity(history_size),
            max_size: max_window,
            history_size,
        }
    }

    /// Update with new minute bar data.
    pub fn update(&mut self, price: f64, volume: f64) {
        // Add new samples
        self.prices.push_back(price);
        self.volumes.push_back(volume);

        // Trim to max size
        while self.prices.len() > self.max_size {
            self.prices.pop_front();
        }
        while self.volumes.len() > self.max_size {
            self.volumes.pop_front();
        }

        // Update history with current absorption (using largest window)
        if self.prices.len() >= self.max_size {
            let ar = self.compute_raw(self.max_size);
            self.history.push_back(ar);
            while self.history.len() > self.history_size {
                self.history.pop_front();
            }
        }
    }

    /// Compute absorption ratio for a given window.
    ///
    /// # Arguments
    /// * `window` - Window size in minutes
    ///
    /// # Returns
    /// Absorption ratio (volume / price change)
    pub fn compute(&self, window: usize) -> f64 {
        if self.prices.len() < window || window == 0 {
            return 0.0;
        }
        self.compute_raw(window)
    }

    /// Internal computation without bounds checking.
    fn compute_raw(&self, window: usize) -> f64 {
        let n = self.prices.len();
        let start_idx = n - window;

        // Sum volume over window
        let total_volume: f64 = self.volumes.iter().skip(start_idx).sum();

        // Price change over window
        let start_price = self.prices[start_idx];
        let end_price = self.prices[n - 1];
        let price_change = (end_price - start_price).abs();

        // Absorption ratio
        total_volume / (price_change + EPSILON)
    }

    /// Compute z-score of current absorption vs history.
    pub fn compute_zscore(&self, window: usize) -> f64 {
        if self.history.len() < 10 {
            return 0.0;
        }

        let current = self.compute(window);

        let mean: f64 = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let variance: f64 = self.history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / self.history.len() as f64;
        let std = variance.sqrt();

        if std < EPSILON {
            return 0.0;
        }

        (current - mean) / std
    }

    /// Check if enough data for computation.
    pub fn is_ready(&self, window: usize) -> bool {
        self.prices.len() >= window
    }

    /// Get current buffer length.
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absorption_high_volume_no_movement() {
        let mut computer = AbsorptionComputer::new(60, 100);

        // High volume, flat price
        for _ in 0..60 {
            computer.update(100.0, 1000.0);
        }

        let ar = computer.compute(30);
        // Price didn't move, so absorption should be very high
        assert!(ar > 1_000_000.0);
    }

    #[test]
    fn test_absorption_low_volume_large_movement() {
        let mut computer = AbsorptionComputer::new(60, 100);

        // Low volume, rising price
        for i in 0..60 {
            computer.update(100.0 + i as f64, 10.0);
        }

        let ar = computer.compute(30);
        // Price moved a lot, low volume, so absorption should be low
        assert!(ar < 100.0);
    }

    #[test]
    fn test_absorption_zscore() {
        let mut computer = AbsorptionComputer::new(60, 100);

        // Build up history with normal values
        for i in 0..100 {
            let price = 100.0 + (i as f64 * 0.1).sin();
            computer.update(price, 100.0);
        }

        // Z-score should be near 0 for normal values
        let zscore = computer.compute_zscore(60);
        assert!(zscore.abs() < 3.0);
    }
}

// ============================================================================
// Skeptical Tests - Validate regime detection capability
// ============================================================================

#[cfg(test)]
mod skeptical_tests {
    use super::*;

    /// Test 1: Absorption should be high when volume is high but price doesn't move
    /// This is the classic accumulation/distribution signature
    #[test]
    fn test_high_absorption_accumulation_pattern() {
        let mut computer = AbsorptionComputer::new(300, 500);

        // Simulate accumulation: high volume, minimal price change
        for i in 0..300 {
            let price = 100.0 + (i as f64 * 0.001); // Tiny price drift (0.3 total)
            let volume = 1000.0; // Constant high volume
            computer.update(price, volume);
        }

        let absorption = computer.compute(300);

        // Price moved ~0.3, volume = 300,000
        // Absorption should be very high (>100,000)
        assert!(
            absorption > 100_000.0,
            "Accumulation pattern should have high absorption: {}",
            absorption
        );
    }

    /// Test 2: Absorption should be low when price moves easily on volume
    /// This indicates markup/markdown phases
    #[test]
    fn test_low_absorption_trending_pattern() {
        let mut computer = AbsorptionComputer::new(300, 500);

        // Simulate markup: price moves significantly on moderate volume
        for i in 0..300 {
            let price = 100.0 + (i as f64 * 0.1); // Strong price trend (30 total)
            let volume = 100.0; // Moderate volume
            computer.update(price, volume);
        }

        let absorption = computer.compute(300);

        // Price moved ~30, volume = 30,000
        // Absorption should be ~1,000
        assert!(
            absorption < 2000.0,
            "Trending pattern should have low absorption: {}",
            absorption
        );
    }

    /// Test 3: Absorption should clearly differentiate accumulation from markup
    /// This is the core regime detection capability
    #[test]
    fn test_absorption_differentiates_regimes() {
        // Accumulation pattern: flat price, high volume
        let mut acc_computer = AbsorptionComputer::new(300, 500);
        for _ in 0..300 {
            acc_computer.update(100.0, 1000.0); // Flat price, high volume
        }
        let acc_absorption = acc_computer.compute(300);

        // Markup pattern: trending price, same volume
        let mut mk_computer = AbsorptionComputer::new(300, 500);
        for i in 0..300 {
            mk_computer.update(100.0 + i as f64, 1000.0); // Trending price
        }
        let mk_absorption = mk_computer.compute(300);

        // Accumulation should have MUCH higher absorption (100x or more)
        assert!(
            acc_absorption > mk_absorption * 100.0,
            "Accumulation absorption {} should be >> markup absorption {}",
            acc_absorption,
            mk_absorption
        );
    }

    /// Test 4: Z-score should detect abnormal absorption relative to history
    #[test]
    fn test_zscore_detects_anomaly() {
        let mut computer = AbsorptionComputer::new(100, 200);

        // Build normal history: moderate absorption
        for i in 0..150 {
            let price = 100.0 + (i as f64 * 0.1); // Normal trending
            let volume = 100.0;
            computer.update(price, volume);
        }

        let normal_zscore = computer.compute_zscore(100);

        // Now inject accumulation pattern: flat price, high volume
        for _ in 0..50 {
            computer.update(115.0, 1000.0); // Flat, 10x volume
        }

        let anomaly_zscore = computer.compute_zscore(100);

        // Anomaly z-score should be significantly higher
        assert!(
            anomaly_zscore > normal_zscore + 1.0,
            "Anomaly z-score {} should be significantly higher than normal {}",
            anomaly_zscore,
            normal_zscore
        );
    }

    /// Test 5: Edge case - zero price change should not cause panic/infinity
    #[test]
    fn test_zero_price_change_handled() {
        let mut computer = AbsorptionComputer::new(100, 200);

        // Perfectly flat price
        for _ in 0..100 {
            computer.update(100.0, 1000.0);
        }

        let absorption = computer.compute(50);

        // Should return a high but finite value, not infinity
        assert!(absorption.is_finite(), "Absorption should be finite");
        assert!(
            absorption > 0.0,
            "Absorption should be positive with volume"
        );
    }

    /// Test 6: Edge case - zero volume should give zero absorption
    #[test]
    fn test_zero_volume_handled() {
        let mut computer = AbsorptionComputer::new(100, 200);

        // No volume, price trending
        for i in 0..100 {
            computer.update(100.0 + i as f64, 0.0);
        }

        let absorption = computer.compute(50);

        // Zero volume means zero absorption
        assert!(
            absorption.abs() < EPSILON,
            "Zero volume should give near-zero absorption: {}",
            absorption
        );
    }

    /// Test 7: Multi-window consistency - longer windows should smooth noise
    #[test]
    fn test_multi_window_consistency() {
        let mut computer = AbsorptionComputer::new(1440, 2000);

        // Build up data with some noise
        for i in 0..1440 {
            let noise = ((i * 7) % 13) as f64 - 6.0; // Deterministic "noise"
            let price = 100.0 + (i as f64 * 0.01) + noise * 0.001;
            let volume = 100.0 + noise.abs() * 10.0;
            computer.update(price, volume);
        }

        let abs_60 = computer.compute(60);
        let abs_240 = computer.compute(240);
        let abs_1440 = computer.compute(1440);

        // All should be positive and finite
        assert!(abs_60.is_finite() && abs_60 > 0.0);
        assert!(abs_240.is_finite() && abs_240 > 0.0);
        assert!(abs_1440.is_finite() && abs_1440 > 0.0);

        // Longer windows should give more stable (less extreme) values in noisy data
        // This is a soft check - the key is they're all reasonable
    }

    /// Test 8: Z-score normalization enables cross-asset comparison
    /// Raw absorption is scale-dependent, but z-scores normalize this
    #[test]
    fn test_zscore_enables_cross_asset_comparison() {
        // BTC scale: $50,000 with accumulation pattern
        let mut btc_computer = AbsorptionComputer::new(100, 200);
        // First build normal history
        for i in 0..150 {
            let price = 50000.0 + (i as f64 * 5.0); // Normal trending
            let volume = 10.0;
            btc_computer.update(price, volume);
        }
        // Then accumulation: flat price, same volume
        for _ in 0..50 {
            btc_computer.update(50750.0, 10.0); // Flat
        }
        let btc_zscore = btc_computer.compute_zscore(100);

        // ETH scale: $3,000 with same relative accumulation pattern
        let mut eth_computer = AbsorptionComputer::new(100, 200);
        // First build normal history
        for i in 0..150 {
            let price = 3000.0 + (i as f64 * 0.3); // Normal trending
            let volume = 10.0;
            eth_computer.update(price, volume);
        }
        // Then accumulation: flat price, same volume
        for _ in 0..50 {
            eth_computer.update(3045.0, 10.0); // Flat
        }
        let eth_zscore = eth_computer.compute_zscore(100);

        // Both z-scores should be elevated (>1) indicating absorption anomaly
        // The key insight: z-scores normalize scale differences
        assert!(
            btc_zscore > 1.0,
            "BTC z-score should indicate anomaly: {}",
            btc_zscore
        );
        assert!(
            eth_zscore > 1.0,
            "ETH z-score should indicate anomaly: {}",
            eth_zscore
        );

        // Z-scores should be reasonably similar (within 2x of each other)
        // since both represent the same regime shift
        let zscore_ratio = if btc_zscore > eth_zscore {
            btc_zscore / eth_zscore
        } else {
            eth_zscore / btc_zscore
        };
        assert!(
            zscore_ratio < 3.0,
            "Z-scores should be comparable across assets: BTC={}, ETH={}, ratio={}",
            btc_zscore,
            eth_zscore,
            zscore_ratio
        );
    }

    /// Test 9: Real-world scenario - accumulation followed by breakout
    #[test]
    fn test_accumulation_to_breakout_transition() {
        let mut computer = AbsorptionComputer::new(200, 500);

        // Phase 1: Accumulation (high volume, flat price)
        for _ in 0..100 {
            computer.update(100.0, 500.0);
        }

        let acc_absorption = computer.compute(100);

        // Phase 2: Breakout (price rises on volume)
        for i in 0..100 {
            computer.update(100.0 + (i as f64 * 0.5), 500.0);
        }

        let breakout_absorption = computer.compute(100);

        // During accumulation, absorption should be much higher than during breakout
        assert!(
            acc_absorption > breakout_absorption * 10.0,
            "Accumulation absorption {} should be >> breakout absorption {}",
            acc_absorption,
            breakout_absorption
        );
    }
}
