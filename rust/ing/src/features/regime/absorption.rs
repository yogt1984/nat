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
