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
