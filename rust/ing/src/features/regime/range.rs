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
