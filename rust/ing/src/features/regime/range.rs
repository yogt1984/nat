//! Range Position Computation
//!
//! Measures where current price sits within recent price range.
//!
//! Formula: Position = (Price - Min) / (Max - Min)
//!
//! - Position near 0: Price at range lows (potential accumulation zone)
//! - Position near 1: Price at range highs (potential distribution zone)
//! - Range width: Volatility measure

use std::collections::VecDeque;

const EPSILON: f64 = 1e-10;

/// Computes range position at multiple time windows.
#[derive(Debug, Clone)]
pub struct RangeComputer {
    /// Price samples
    prices: VecDeque<f64>,
    /// Maximum buffer size
    max_size: usize,
}

impl RangeComputer {
    /// Create new range computer.
    ///
    /// # Arguments
    /// * `max_window` - Largest window in minutes (e.g., 10080 for 1 week)
    pub fn new(max_window: usize) -> Self {
        Self {
            prices: VecDeque::with_capacity(max_window),
            max_size: max_window,
        }
    }

    /// Update with new price.
    pub fn update(&mut self, price: f64) {
        self.prices.push_back(price);

        while self.prices.len() > self.max_size {
            self.prices.pop_front();
        }
    }

    /// Compute range position for a given window.
    ///
    /// # Returns
    /// Position in range [0, 1] where 0 = at low, 1 = at high
    pub fn compute_position(&self, window: usize) -> f64 {
        if self.prices.len() < window || window == 0 {
            return 0.5; // Default to middle
        }

        let n = self.prices.len();
        let start = n - window;

        let mut min = f64::MAX;
        let mut max = f64::MIN;

        for i in start..n {
            let p = self.prices[i];
            if p < min {
                min = p;
            }
            if p > max {
                max = p;
            }
        }

        let range = max - min;
        if range < EPSILON {
            return 0.5;
        }

        let current = self.prices[n - 1];
        (current - min) / range
    }

    /// Compute range width as percentage of midpoint.
    ///
    /// # Returns
    /// Range width as decimal (0.05 = 5% range)
    pub fn compute_width(&self, window: usize) -> f64 {
        if self.prices.len() < window || window == 0 {
            return 0.0;
        }

        let n = self.prices.len();
        let start = n - window;

        let mut min = f64::MAX;
        let mut max = f64::MIN;

        for i in start..n {
            let p = self.prices[i];
            if p < min {
                min = p;
            }
            if p > max {
                max = p;
            }
        }

        let midpoint = (max + min) / 2.0;
        if midpoint < EPSILON {
            return 0.0;
        }

        (max - min) / midpoint
    }

    /// Compute range position velocity (rate of change).
    ///
    /// Measures how fast price is moving within the range.
    /// Positive = moving toward high, Negative = moving toward low.
    ///
    /// # Arguments
    /// * `window` - Window for range calculation
    /// * `lookback` - How far back to compare position
    ///
    /// # Returns
    /// Position change per bar (e.g., 0.1 = 10% of range per bar)
    pub fn compute_velocity(&self, window: usize, lookback: usize) -> f64 {
        if self.prices.len() < window || lookback == 0 || lookback >= window {
            return 0.0;
        }

        let current_pos = self.compute_position(window);

        // Compute position from `lookback` bars ago
        // We need to simulate what position was back then
        let n = self.prices.len();
        let start = n - window;

        // Find min/max over the window (same as current)
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        for i in start..n {
            let p = self.prices[i];
            if p < min { min = p; }
            if p > max { max = p; }
        }

        let range = max - min;
        if range < EPSILON {
            return 0.0;
        }

        // Position `lookback` bars ago
        let past_idx = n - 1 - lookback;
        if past_idx < start {
            return 0.0;
        }
        let past_price = self.prices[past_idx];
        let past_pos = (past_price - min) / range;

        // Velocity = change in position / lookback
        (current_pos - past_pos) / lookback as f64
    }

    /// Compute distance from range extremes.
    ///
    /// # Returns
    /// (distance_from_low, distance_from_high) as percentages of range
    pub fn compute_extreme_distances(&self, window: usize) -> (f64, f64) {
        let position = self.compute_position(window);
        (position, 1.0 - position)
    }

    /// Check if price is at range extreme (within threshold).
    ///
    /// # Arguments
    /// * `window` - Window for range calculation
    /// * `threshold` - How close to extreme counts (e.g., 0.1 = within 10%)
    ///
    /// # Returns
    /// -1 = at low, 0 = middle, 1 = at high
    pub fn at_extreme(&self, window: usize, threshold: f64) -> i32 {
        let position = self.compute_position(window);
        if position <= threshold {
            -1 // At low
        } else if position >= 1.0 - threshold {
            1 // At high
        } else {
            0 // Middle
        }
    }

    /// Check if ready for computation.
    pub fn is_ready(&self, window: usize) -> bool {
        self.prices.len() >= window
    }

    pub fn len(&self) -> usize {
        self.prices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_position_at_high() {
        let mut computer = RangeComputer::new(100);

        // Price starts low and ends high
        for i in 0..60 {
            computer.update(100.0 + i as f64);
        }

        let position = computer.compute_position(30);
        // Should be near 1.0 (at the high)
        assert!(position > 0.9);
    }

    #[test]
    fn test_range_position_at_low() {
        let mut computer = RangeComputer::new(100);

        // Price starts high and ends low
        for i in 0..60 {
            computer.update(160.0 - i as f64);
        }

        let position = computer.compute_position(30);
        // Should be near 0.0 (at the low)
        assert!(position < 0.1);
    }

    #[test]
    fn test_range_position_middle() {
        let mut computer = RangeComputer::new(100);

        // Add range then return to middle
        computer.update(100.0);
        computer.update(110.0);
        computer.update(90.0);
        computer.update(100.0); // Back to middle

        let position = computer.compute_position(4);
        // Should be near 0.5
        assert!((position - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_range_width() {
        let mut computer = RangeComputer::new(100);

        // Create 10% range around 100
        computer.update(95.0);
        computer.update(105.0);
        computer.update(100.0);

        let width = computer.compute_width(3);
        // Range = 10, midpoint = 100, width = 0.1
        assert!((width - 0.1).abs() < 0.01);
    }
}

// ============================================================================
// Skeptical Tests - Validate regime detection capability
// ============================================================================

#[cfg(test)]
mod skeptical_tests {
    use super::*;

    /// Test 1: Position exactly 0 at range low
    #[test]
    fn test_position_exactly_zero_at_low() {
        let mut computer = RangeComputer::new(100);

        // Create range, end at low
        computer.update(100.0); // High
        computer.update(90.0);  // Low (current)

        let position = computer.compute_position(2);

        assert!(
            position.abs() < EPSILON,
            "At range low, position should be 0.0, got {}",
            position
        );
    }

    /// Test 2: Position exactly 1 at range high
    #[test]
    fn test_position_exactly_one_at_high() {
        let mut computer = RangeComputer::new(100);

        // Create range, end at high
        computer.update(90.0);  // Low
        computer.update(100.0); // High (current)

        let position = computer.compute_position(2);

        assert!(
            (position - 1.0).abs() < EPSILON,
            "At range high, position should be 1.0, got {}",
            position
        );
    }

    /// Test 3: Position 0.5 at exact midpoint
    #[test]
    fn test_position_exactly_half_at_midpoint() {
        let mut computer = RangeComputer::new(100);

        computer.update(80.0);  // Low
        computer.update(120.0); // High
        computer.update(100.0); // Exact midpoint

        let position = computer.compute_position(3);

        assert!(
            (position - 0.5).abs() < EPSILON,
            "At midpoint, position should be 0.5, got {}",
            position
        );
    }

    /// Test 4: Flat price returns 0.5 (no range)
    #[test]
    fn test_flat_price_returns_middle() {
        let mut computer = RangeComputer::new(100);

        // All same price
        for _ in 0..50 {
            computer.update(100.0);
        }

        let position = computer.compute_position(50);

        assert!(
            (position - 0.5).abs() < EPSILON,
            "Flat price should return 0.5, got {}",
            position
        );
    }

    /// Test 5: Range width scales with volatility
    #[test]
    fn test_range_width_scales_with_volatility() {
        // Low volatility: 2% range
        let mut low_vol = RangeComputer::new(100);
        low_vol.update(99.0);
        low_vol.update(101.0);
        low_vol.update(100.0);
        let low_width = low_vol.compute_width(3);

        // High volatility: 20% range
        let mut high_vol = RangeComputer::new(100);
        high_vol.update(90.0);
        high_vol.update(110.0);
        high_vol.update(100.0);
        let high_width = high_vol.compute_width(3);

        // High vol width should be 10x low vol
        let ratio = high_width / low_width;
        assert!(
            (ratio - 10.0).abs() < 0.1,
            "Width ratio should be ~10x: low={}, high={}, ratio={}",
            low_width,
            high_width,
            ratio
        );
    }

    /// Test 6: Velocity positive during uptrend
    #[test]
    fn test_velocity_positive_uptrend() {
        let mut computer = RangeComputer::new(100);

        // Steady uptrend
        for i in 0..50 {
            computer.update(100.0 + i as f64);
        }

        let velocity = computer.compute_velocity(50, 10);

        assert!(
            velocity > 0.0,
            "Uptrend should have positive velocity: {}",
            velocity
        );
    }

    /// Test 7: Velocity negative during downtrend
    #[test]
    fn test_velocity_negative_downtrend() {
        let mut computer = RangeComputer::new(100);

        // Steady downtrend
        for i in 0..50 {
            computer.update(150.0 - i as f64);
        }

        let velocity = computer.compute_velocity(50, 10);

        assert!(
            velocity < 0.0,
            "Downtrend should have negative velocity: {}",
            velocity
        );
    }

    /// Test 8: Velocity near zero in ranging market
    #[test]
    fn test_velocity_near_zero_ranging() {
        let mut computer = RangeComputer::new(100);

        // Oscillating price (ranging) - use full cycle to end near start
        for i in 0..63 {
            // ~2 full cycles (2*pi*2 ≈ 12.6, so 63 steps at 0.2 = 12.6)
            let price = 100.0 + 10.0 * ((i as f64 * 0.2).sin());
            computer.update(price);
        }

        let velocity = computer.compute_velocity(50, 20);

        // Velocity should be small (oscillating, not trending)
        // Tolerance of 0.1 accounts for not-perfectly-aligned cycles
        assert!(
            velocity.abs() < 0.1,
            "Ranging market should have near-zero velocity: {}",
            velocity
        );
    }

    /// Test 9: Extreme detection at lows
    #[test]
    fn test_at_extreme_low() {
        let mut computer = RangeComputer::new(100);

        // Range with current at low
        computer.update(110.0);
        computer.update(100.0);
        computer.update(90.0); // At low

        let extreme = computer.at_extreme(3, 0.1);

        assert_eq!(extreme, -1, "Should detect price at low extreme");
    }

    /// Test 10: Extreme detection at highs
    #[test]
    fn test_at_extreme_high() {
        let mut computer = RangeComputer::new(100);

        // Range with current at high
        computer.update(90.0);
        computer.update(100.0);
        computer.update(110.0); // At high

        let extreme = computer.at_extreme(3, 0.1);

        assert_eq!(extreme, 1, "Should detect price at high extreme");
    }

    /// Test 11: Middle returns 0 for at_extreme
    #[test]
    fn test_at_extreme_middle() {
        let mut computer = RangeComputer::new(100);

        computer.update(90.0);
        computer.update(110.0);
        computer.update(100.0); // Middle

        let extreme = computer.at_extreme(3, 0.1);

        assert_eq!(extreme, 0, "Middle should not be extreme");
    }

    /// Test 12: Position is scale-independent
    #[test]
    fn test_position_scale_independence() {
        // BTC at $50,000 with 2% range
        let mut btc = RangeComputer::new(100);
        btc.update(49000.0);
        btc.update(51000.0);
        btc.update(50500.0); // 75% of range

        let btc_pos = btc.compute_position(3);

        // ETH at $3,000 with same relative range
        let mut eth = RangeComputer::new(100);
        eth.update(2940.0);
        eth.update(3060.0);
        eth.update(3030.0); // 75% of range

        let eth_pos = eth.compute_position(3);

        // Both should be at ~0.75 position
        assert!(
            (btc_pos - 0.75).abs() < 0.01,
            "BTC position should be ~0.75: {}",
            btc_pos
        );
        assert!(
            (eth_pos - 0.75).abs() < 0.01,
            "ETH position should be ~0.75: {}",
            eth_pos
        );
        assert!(
            (btc_pos - eth_pos).abs() < 0.01,
            "Positions should be equal: BTC={}, ETH={}",
            btc_pos,
            eth_pos
        );
    }

    /// Test 13: Multi-window shows different perspectives
    /// Different windows capture different market contexts
    #[test]
    fn test_multi_window_perspectives() {
        let mut computer = RangeComputer::new(500);

        // Phase 1: Establish a low base (200 bars at 100)
        for _ in 0..200 {
            computer.update(100.0);
        }

        // Phase 2: Rally to 200 (100 bars)
        for i in 0..100 {
            computer.update(100.0 + i as f64); // Rally from 100 to 199
        }
        computer.update(200.0); // Peak

        // Phase 3: Pullback to 150 (50 bars)
        for i in 1..=50 {
            computer.update(200.0 - i as f64); // 199, 198, ... 150
        }

        // Total: 200 + 100 + 1 + 50 = 351 bars, current price = 150

        // Short window (50 bars): sees recent pullback from ~200 to 150
        // Range: 150 (low) to 200 (high), current at 150 = position 0.0
        let pos_short = computer.compute_position(50);

        // Long window (300 bars): sees base at 100, rally to 200, pullback to 150
        // Range: 100 (low) to 200 (high), current at 150 = position 0.5
        let pos_long = computer.compute_position(300);

        // Verify short window shows position at low end (we just pulled back)
        assert!(
            pos_short < 0.1,
            "Short window should show we're at bottom of recent range: {}",
            pos_short
        );

        // Verify long window shows higher position (remembers old lows)
        assert!(
            pos_long > 0.4,
            "Long window should show we're elevated vs old lows: {}",
            pos_long
        );

        // The key insight: same price reads differently depending on context
        assert!(
            pos_long > pos_short,
            "Long window {} should show higher position than short window {}",
            pos_long,
            pos_short
        );
    }

    /// Test 14: Wyckoff accumulation zone detection
    /// Price near range lows with decreasing volatility
    #[test]
    fn test_wyckoff_accumulation_zone() {
        let mut computer = RangeComputer::new(100);

        // Create range, settle near lows
        computer.update(90.0);  // Low
        computer.update(110.0); // High
        computer.update(95.0);  // Near low
        computer.update(93.0);  // Even closer to low
        computer.update(92.0);  // At accumulation zone

        let position = computer.compute_position(5);
        let (dist_low, dist_high) = computer.compute_extreme_distances(5);

        // Should be in accumulation zone (near low)
        assert!(
            position < 0.2,
            "Should be in accumulation zone (near low): {}",
            position
        );
        assert!(
            dist_low < 0.2,
            "Distance from low should be small: {}",
            dist_low
        );
        assert!(
            dist_high > 0.8,
            "Distance from high should be large: {}",
            dist_high
        );
    }

    /// Test 15: Wyckoff distribution zone detection
    /// Price near range highs
    #[test]
    fn test_wyckoff_distribution_zone() {
        let mut computer = RangeComputer::new(100);

        // Create range, settle near highs
        computer.update(90.0);  // Low
        computer.update(110.0); // High
        computer.update(105.0); // Near high
        computer.update(107.0); // Even closer to high
        computer.update(108.0); // At distribution zone

        let position = computer.compute_position(5);
        let (dist_low, dist_high) = computer.compute_extreme_distances(5);

        // Should be in distribution zone (near high)
        assert!(
            position > 0.8,
            "Should be in distribution zone (near high): {}",
            position
        );
        assert!(
            dist_high < 0.2,
            "Distance from high should be small: {}",
            dist_high
        );
        assert!(
            dist_low > 0.8,
            "Distance from low should be large: {}",
            dist_low
        );
    }

    /// Test 16: Spring detection (false breakdown)
    /// Price briefly goes below range then recovers
    #[test]
    fn test_spring_pattern() {
        let mut computer = RangeComputer::new(100);

        // Establish range
        for _ in 0..20 {
            computer.update(100.0);
        }
        computer.update(95.0);  // Low of range
        computer.update(105.0); // High of range

        // Spring: break below range then recover
        computer.update(93.0);  // Below range (spring)
        computer.update(98.0);  // Recover into range
        computer.update(102.0); // Strong recovery

        let position = computer.compute_position(25);

        // After spring and recovery, should be in upper half
        assert!(
            position > 0.5,
            "After spring recovery, should be above middle: {}",
            position
        );
    }

    /// Test 17: Edge case - very small range
    #[test]
    fn test_very_small_range() {
        let mut computer = RangeComputer::new(100);

        // Tiny range (0.0001%)
        computer.update(100.0000);
        computer.update(100.0001);
        computer.update(100.00005);

        let position = computer.compute_position(3);
        let width = computer.compute_width(3);

        // Should still work without panic/NaN
        assert!(position.is_finite(), "Position should be finite");
        assert!(width.is_finite(), "Width should be finite");
        assert!(position >= 0.0 && position <= 1.0, "Position should be [0,1]");
    }
}
