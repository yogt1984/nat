//! Churn Rate Computation
//!
//! Churn rate measures two-sided trading activity.
//! High churn indicates position transfer between participants.
//!
//! Formula: Churn = (BuyVol + SellVol) / (|BuyVol - SellVol| + ε)
//!
//! - Churn ≈ 1: All volume is one-directional
//! - Churn >> 1: Volume is balanced (high two-sided activity)

use std::collections::VecDeque;

const EPSILON: f64 = 1e-10;

/// Computes churn rate measuring two-sided trading activity.
#[derive(Debug, Clone)]
pub struct ChurnComputer {
    /// Buy volume samples
    buy_volumes: VecDeque<f64>,
    /// Sell volume samples
    sell_volumes: VecDeque<f64>,
    /// Historical churn values for z-score
    history: VecDeque<f64>,
    /// Maximum buffer size
    max_size: usize,
    /// History size
    history_size: usize,
}

impl ChurnComputer {
    /// Create new churn computer.
    pub fn new(max_window: usize, history_size: usize) -> Self {
        Self {
            buy_volumes: VecDeque::with_capacity(max_window),
            sell_volumes: VecDeque::with_capacity(max_window),
            history: VecDeque::with_capacity(history_size),
            max_size: max_window,
            history_size,
        }
    }

    /// Update with new minute bar data.
    pub fn update(&mut self, buy_volume: f64, sell_volume: f64) {
        self.buy_volumes.push_back(buy_volume);
        self.sell_volumes.push_back(sell_volume);

        // Trim buffers
        while self.buy_volumes.len() > self.max_size {
            self.buy_volumes.pop_front();
        }
        while self.sell_volumes.len() > self.max_size {
            self.sell_volumes.pop_front();
        }

        // Update history
        if self.buy_volumes.len() >= self.max_size {
            let churn = self.compute_raw(self.max_size);
            self.history.push_back(churn);
            while self.history.len() > self.history_size {
                self.history.pop_front();
            }
        }
    }

    /// Compute churn rate for a given window.
    pub fn compute(&self, window: usize) -> f64 {
        if self.buy_volumes.len() < window || window == 0 {
            return 1.0; // Default to neutral
        }
        self.compute_raw(window)
    }

    fn compute_raw(&self, window: usize) -> f64 {
        let n = self.buy_volumes.len();
        let start = n - window;

        let total_buy: f64 = self.buy_volumes.iter().skip(start).sum();
        let total_sell: f64 = self.sell_volumes.iter().skip(start).sum();

        let total = total_buy + total_sell;
        let net = (total_buy - total_sell).abs();

        total / (net + EPSILON)
    }

    /// Compute z-score of churn.
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

    /// Compute churn signal (log-transformed for symmetry).
    ///
    /// The signal is centered at churn=2 (neutral point):
    /// - `> 0` means churn > 2 (high two-sided activity, distribution/ranging)
    /// - `< 0` means churn < 2 (directional flow, accumulation/trending)
    ///
    /// # Arguments
    /// * `window` - Window size in minutes
    ///
    /// # Returns
    /// Log-transformed churn signal
    pub fn compute_signal(&self, window: usize) -> f64 {
        let churn = self.compute(window);
        // Avoid log(0) - minimum churn is 1.0
        let safe_churn = churn.max(1.0);
        (safe_churn / 2.0).ln()
    }

    /// Check if ready for computation.
    pub fn is_ready(&self, window: usize) -> bool {
        self.buy_volumes.len() >= window
    }

    pub fn len(&self) -> usize {
        self.buy_volumes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buy_volumes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_churn_one_directional() {
        let mut computer = ChurnComputer::new(60, 100);

        // All buying, no selling
        for _ in 0..60 {
            computer.update(1000.0, 0.0);
        }

        let churn = computer.compute(30);
        // Churn should be close to 1.0 (one-directional)
        assert!((churn - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_churn_balanced() {
        let mut computer = ChurnComputer::new(60, 100);

        // Equal buying and selling
        for _ in 0..60 {
            computer.update(500.0, 500.0);
        }

        let churn = computer.compute(30);
        // Churn should be very high (balanced volume)
        assert!(churn > 1000.0);
    }

    #[test]
    fn test_churn_accumulation_signature() {
        let mut computer = ChurnComputer::new(60, 100);

        // Slight buy bias with high total volume
        for _ in 0..60 {
            computer.update(600.0, 400.0);
        }

        let churn = computer.compute(30);
        // Churn should be elevated (5.0 = 1000/200)
        assert!(churn > 3.0);
        assert!(churn < 10.0);
    }
}

// ============================================================================
// Skeptical Tests - Validate regime detection capability
// ============================================================================

#[cfg(test)]
mod skeptical_tests {
    use super::*;

    /// Test 1: Pure one-sided flow should have churn = 1
    /// This represents strong directional conviction (markup/markdown)
    #[test]
    fn test_one_sided_flow_churn_equals_one() {
        let mut computer = ChurnComputer::new(100, 200);

        // All buys, no sells - pure directional flow
        for _ in 0..100 {
            computer.update(10000.0, 0.0);
        }

        let churn = computer.compute(100);

        // Churn = total / |net| = 10000 / 10000 = 1.0
        assert!(
            (churn - 1.0).abs() < 0.01,
            "One-sided flow should have churn=1.0, got {}",
            churn
        );

        // Signal should be negative (directional)
        let signal = computer.compute_signal(100);
        assert!(
            signal < 0.0,
            "Directional flow should have negative signal, got {}",
            signal
        );
    }

    /// Test 2: Perfectly balanced flow should have very high churn
    /// This indicates high two-sided activity (distribution or ranging)
    #[test]
    fn test_balanced_flow_high_churn() {
        let mut computer = ChurnComputer::new(100, 200);

        // Equal buying and selling - perfectly balanced
        for _ in 0..100 {
            computer.update(5000.0, 5000.0);
        }

        let churn = computer.compute(100);

        // Net volume is ~0, so churn should be very high (capped by epsilon)
        assert!(
            churn > 1000.0,
            "Balanced flow should have very high churn, got {}",
            churn
        );

        // Signal should be positive (two-sided)
        let signal = computer.compute_signal(100);
        assert!(
            signal > 0.0,
            "Balanced flow should have positive signal, got {}",
            signal
        );
    }

    /// Test 3: 60/40 split should have moderate churn = 5
    /// Mathematical verification: (6000+4000) / |6000-4000| = 10000/2000 = 5
    #[test]
    fn test_moderate_imbalance_churn() {
        let mut computer = ChurnComputer::new(100, 200);

        // 60% buy, 40% sell
        for _ in 0..100 {
            computer.update(600.0, 400.0);
        }

        let churn = computer.compute(100);

        // Expected: (60000 + 40000) / (60000 - 40000) = 100000 / 20000 = 5.0
        assert!(
            (churn - 5.0).abs() < 0.1,
            "60/40 split should have churn=5.0, got {}",
            churn
        );
    }

    /// Test 4: Churn should differentiate accumulation from distribution
    /// - Accumulation: consistent buying (LOW churn, directional)
    /// - Distribution: selling into buying (HIGH churn, two-sided)
    #[test]
    fn test_churn_differentiates_accumulation_from_distribution() {
        // Accumulation pattern: strong buy bias (80/20)
        let mut acc_computer = ChurnComputer::new(100, 200);
        for _ in 0..100 {
            acc_computer.update(8000.0, 2000.0); // Heavy buying
        }
        let acc_churn = acc_computer.compute(100);
        // Expected: 10000 / 6000 ≈ 1.67

        // Distribution pattern: selling into buying (55/45)
        let mut dist_computer = ChurnComputer::new(100, 200);
        for _ in 0..100 {
            dist_computer.update(5500.0, 4500.0); // Near balanced
        }
        let dist_churn = dist_computer.compute(100);
        // Expected: 10000 / 1000 = 10.0

        // Distribution should have MUCH higher churn than accumulation
        assert!(
            dist_churn > acc_churn * 2.0,
            "Distribution churn {} should be >> accumulation churn {}",
            dist_churn,
            acc_churn
        );

        // Verify expected ranges
        assert!(
            acc_churn < 2.5,
            "Accumulation churn should be low: {}",
            acc_churn
        );
        assert!(
            dist_churn > 5.0,
            "Distribution churn should be high: {}",
            dist_churn
        );
    }

    /// Test 5: Churn signal should be negative for directional flow
    #[test]
    fn test_churn_signal_directional() {
        let mut computer = ChurnComputer::new(100, 200);

        // Strong directional flow (90/10)
        for _ in 0..100 {
            computer.update(9000.0, 1000.0);
        }

        let churn = computer.compute(100);
        let signal = computer.compute_signal(100);

        // Churn = 10000 / 8000 = 1.25, which is < 2
        assert!(churn < 2.0, "Directional flow should have churn < 2: {}", churn);

        // Signal = ln(1.25 / 2) = ln(0.625) ≈ -0.47
        assert!(
            signal < 0.0,
            "Directional flow should have negative signal: {}",
            signal
        );
    }

    /// Test 6: Churn signal should be positive for balanced flow
    #[test]
    fn test_churn_signal_balanced() {
        let mut computer = ChurnComputer::new(100, 200);

        // Near-balanced flow (52/48)
        for _ in 0..100 {
            computer.update(5200.0, 4800.0);
        }

        let churn = computer.compute(100);
        let signal = computer.compute_signal(100);

        // Churn = 10000 / 400 = 25, which is > 2
        assert!(churn > 2.0, "Balanced flow should have churn > 2: {}", churn);

        // Signal = ln(25 / 2) = ln(12.5) ≈ 2.5
        assert!(
            signal > 0.0,
            "Balanced flow should have positive signal: {}",
            signal
        );
    }

    /// Test 7: Z-score should detect regime change
    #[test]
    fn test_zscore_detects_regime_change() {
        let mut computer = ChurnComputer::new(100, 200);

        // Build history with directional flow (low churn)
        for _ in 0..150 {
            computer.update(8000.0, 2000.0); // 80/20 split
        }

        let baseline_zscore = computer.compute_zscore(100);

        // Now inject distribution pattern (high churn)
        for _ in 0..50 {
            computer.update(5100.0, 4900.0); // Near balanced
        }

        let elevated_zscore = computer.compute_zscore(100);

        // Z-score should increase significantly
        assert!(
            elevated_zscore > baseline_zscore + 1.0,
            "Elevated z-score {} should be > baseline {} + 1.0",
            elevated_zscore,
            baseline_zscore
        );
    }

    /// Test 8: Churn is volume-scale independent
    /// Same ratio should give same churn regardless of absolute volume
    #[test]
    fn test_volume_scale_independence() {
        // High volume market (BTC-like)
        let mut high_vol = ChurnComputer::new(100, 200);
        for _ in 0..100 {
            high_vol.update(100_000.0, 50_000.0); // 2:1 ratio
        }
        let high_churn = high_vol.compute(100);

        // Low volume market (altcoin-like)
        let mut low_vol = ChurnComputer::new(100, 200);
        for _ in 0..100 {
            low_vol.update(100.0, 50.0); // Same 2:1 ratio
        }
        let low_churn = low_vol.compute(100);

        // Same ratio should produce same churn
        // Expected: 150000 / 50000 = 3.0 for both
        assert!(
            (high_churn - low_churn).abs() < 0.01,
            "Churn should be volume-scale independent: high={}, low={}",
            high_churn,
            low_churn
        );
    }

    /// Test 9: Edge case - zero volume should not panic
    #[test]
    fn test_zero_volume_handled() {
        let mut computer = ChurnComputer::new(100, 200);

        // No trading activity
        for _ in 0..100 {
            computer.update(0.0, 0.0);
        }

        let churn = computer.compute(100);
        let signal = computer.compute_signal(100);

        // Should return safe defaults, not panic or NaN
        assert!(churn.is_finite(), "Churn should be finite with zero volume");
        assert!(signal.is_finite(), "Signal should be finite with zero volume");
    }

    /// Test 10: Multi-window consistency
    #[test]
    fn test_multi_window_consistency() {
        let mut computer = ChurnComputer::new(1440, 2000);

        // Build up data with varying flow
        for i in 0..1440 {
            let buy_ratio = 0.5 + 0.2 * ((i as f64 * 0.01).sin()); // Oscillate 30-70%
            let total = 1000.0;
            computer.update(total * buy_ratio, total * (1.0 - buy_ratio));
        }

        let churn_60 = computer.compute(60);
        let churn_240 = computer.compute(240);
        let churn_1440 = computer.compute(1440);

        // All should be positive and finite
        assert!(churn_60.is_finite() && churn_60 >= 1.0);
        assert!(churn_240.is_finite() && churn_240 >= 1.0);
        assert!(churn_1440.is_finite() && churn_1440 >= 1.0);

        // Longer windows should be more stable (closer to mean)
        // With oscillating data, all should be elevated but not extreme
        assert!(
            churn_1440 > 1.5 && churn_1440 < 100.0,
            "1440-window churn should be moderate: {}",
            churn_1440
        );
    }

    /// Test 11: Real-world scenario - markup to distribution transition
    #[test]
    fn test_markup_to_distribution_transition() {
        let mut computer = ChurnComputer::new(200, 500);

        // Phase 1: Markup (strong directional buying)
        for _ in 0..100 {
            computer.update(9000.0, 1000.0); // 90% buys
        }
        let markup_churn = computer.compute(100);

        // Phase 2: Distribution (selling into buying)
        for _ in 0..100 {
            computer.update(5200.0, 4800.0); // Near balanced
        }
        let distribution_churn = computer.compute(100);

        // Churn should increase significantly during distribution
        assert!(
            distribution_churn > markup_churn * 5.0,
            "Distribution churn {} should be >> markup churn {}",
            distribution_churn,
            markup_churn
        );
    }

    /// Test 12: Verify signal transformation properties
    #[test]
    fn test_signal_transformation_properties() {
        let mut computer = ChurnComputer::new(100, 200);

        // Test at neutral point (churn = 2)
        // Need 2:1 total/net ratio => buy=2, sell=1 gives total=3, net=1, churn=3
        // For churn=2: total = 2*net, so if net=X, total=2X, buy+sell=2X, |buy-sell|=X
        // buy=1.5X, sell=0.5X gives churn=2
        for _ in 0..100 {
            computer.update(750.0, 250.0); // 3:1 ratio gives churn=2
        }

        let churn = computer.compute(100);
        let signal = computer.compute_signal(100);

        // Churn = 1000/500 = 2.0, signal = ln(1) = 0
        assert!(
            (churn - 2.0).abs() < 0.1,
            "Expected churn≈2.0, got {}",
            churn
        );
        assert!(
            signal.abs() < 0.1,
            "At churn=2, signal should be ≈0, got {}",
            signal
        );
    }
}
