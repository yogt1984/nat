//! Volume-Price Divergence Computation
//!
//! Measures deviation from expected price impact using Kyle's lambda.
//!
//! Kyle's model: ΔP = λ * SignedVolume
//! Divergence = Actual_ΔP - Expected_ΔP
//!
//! Negative divergence (price rises less than expected given buying) = accumulation
//! Positive divergence (price falls less than expected given selling) = distribution

use std::collections::VecDeque;

const EPSILON: f64 = 1e-10;

/// Computes volume-price divergence using Kyle's lambda estimation.
#[derive(Debug, Clone)]
pub struct DivergenceComputer {
    /// Price samples
    prices: VecDeque<f64>,
    /// Buy volume samples
    buy_volumes: VecDeque<f64>,
    /// Sell volume samples
    sell_volumes: VecDeque<f64>,
    /// Historical divergence values for z-score
    history: VecDeque<f64>,
    /// Current Kyle's lambda estimate
    kyle_lambda: f64,
    /// Maximum buffer size
    max_size: usize,
    /// History size for z-score
    history_size: usize,
    /// Window for lambda estimation
    lambda_window: usize,
}

impl DivergenceComputer {
    /// Create new divergence computer.
    pub fn new(max_window: usize, history_size: usize, lambda_window: usize) -> Self {
        Self {
            prices: VecDeque::with_capacity(max_window),
            buy_volumes: VecDeque::with_capacity(max_window),
            sell_volumes: VecDeque::with_capacity(max_window),
            history: VecDeque::with_capacity(history_size),
            kyle_lambda: 0.0,
            max_size: max_window,
            history_size,
            lambda_window,
        }
    }

    /// Update with new minute bar data.
    pub fn update(&mut self, price: f64, buy_volume: f64, sell_volume: f64) {
        self.prices.push_back(price);
        self.buy_volumes.push_back(buy_volume);
        self.sell_volumes.push_back(sell_volume);

        // Trim buffers
        while self.prices.len() > self.max_size {
            self.prices.pop_front();
        }
        while self.buy_volumes.len() > self.max_size {
            self.buy_volumes.pop_front();
        }
        while self.sell_volumes.len() > self.max_size {
            self.sell_volumes.pop_front();
        }

        // Update Kyle's lambda estimate
        if self.prices.len() >= self.lambda_window {
            self.update_kyle_lambda();
        }

        // Update history
        if self.prices.len() >= self.max_size {
            let div = self.compute_raw(self.max_size);
            self.history.push_back(div);
            while self.history.len() > self.history_size {
                self.history.pop_front();
            }
        }
    }

    /// Update Kyle's lambda using rolling regression.
    ///
    /// λ = Cov(ΔP, SignedVol) / Var(SignedVol)
    fn update_kyle_lambda(&mut self) {
        let n = self.prices.len();
        if n < self.lambda_window {
            return;
        }

        let start = n - self.lambda_window;

        // Compute price changes and signed volumes
        let mut delta_prices = Vec::with_capacity(self.lambda_window - 1);
        let mut signed_vols = Vec::with_capacity(self.lambda_window - 1);

        for i in start + 1..n {
            let dp = self.prices[i] - self.prices[i - 1];
            let sv = self.buy_volumes[i] - self.sell_volumes[i];
            delta_prices.push(dp);
            signed_vols.push(sv);
        }

        if delta_prices.is_empty() {
            return;
        }

        // Compute means
        let mean_dp: f64 = delta_prices.iter().sum::<f64>() / delta_prices.len() as f64;
        let mean_sv: f64 = signed_vols.iter().sum::<f64>() / signed_vols.len() as f64;

        // Compute covariance and variance
        let mut cov = 0.0;
        let mut var_sv = 0.0;

        for (dp, sv) in delta_prices.iter().zip(signed_vols.iter()) {
            cov += (dp - mean_dp) * (sv - mean_sv);
            var_sv += (sv - mean_sv).powi(2);
        }

        if var_sv > EPSILON {
            // Exponential moving average for stability
            let new_lambda = cov / var_sv;
            self.kyle_lambda = 0.9 * self.kyle_lambda + 0.1 * new_lambda;
        }
    }

    /// Compute divergence for a given window.
    ///
    /// Divergence = Actual_ΔP - λ * SignedVolume
    pub fn compute(&self, window: usize) -> f64 {
        if self.prices.len() < window || window == 0 {
            return 0.0;
        }
        self.compute_raw(window)
    }

    fn compute_raw(&self, window: usize) -> f64 {
        let n = self.prices.len();
        let start = n - window;

        // Actual price change
        let actual_dp = self.prices[n - 1] - self.prices[start];

        // Total signed volume
        let total_buy: f64 = self.buy_volumes.iter().skip(start).sum();
        let total_sell: f64 = self.sell_volumes.iter().skip(start).sum();
        let signed_vol = total_buy - total_sell;

        // Expected price change
        let expected_dp = self.kyle_lambda * signed_vol;

        // Divergence: how much price deviated from expectation
        actual_dp - expected_dp
    }

    /// Compute z-score of divergence.
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

    /// Get current Kyle's lambda estimate.
    pub fn get_kyle_lambda(&self) -> f64 {
        self.kyle_lambda
    }

    /// Check if ready for computation.
    pub fn is_ready(&self, window: usize) -> bool {
        self.prices.len() >= window && self.kyle_lambda.abs() > EPSILON
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
    fn test_kyle_lambda_estimation() {
        let mut computer = DivergenceComputer::new(100, 100, 50);

        // Simulate market with known lambda
        let true_lambda = 0.001;
        let mut price = 100.0;

        for _ in 0..100 {
            let buy_vol = 1000.0 + (rand_like() * 500.0);
            let sell_vol = 1000.0 + (rand_like() * 500.0);
            let signed_vol = buy_vol - sell_vol;

            // Price moves according to Kyle's model
            price += true_lambda * signed_vol;

            computer.update(price, buy_vol, sell_vol);
        }

        // Lambda should be close to true value (with some error due to noise)
        let estimated = computer.get_kyle_lambda();
        assert!(estimated > 0.0); // Should be positive
    }

    #[test]
    fn test_divergence_accumulation() {
        let mut computer = DivergenceComputer::new(100, 100, 50);

        // Set up lambda
        computer.kyle_lambda = 0.001;

        // Simulate accumulation: high buying but price stays flat
        for _ in 0..60 {
            computer.update(100.0, 2000.0, 500.0); // Buying >> Selling
        }

        let divergence = computer.compute(30);
        // Expected: price should have risen given buying, but didn't
        // So divergence should be negative
        assert!(divergence < 0.0);
    }

    // Simple pseudo-random for tests
    fn rand_like() -> f64 {
        static mut SEED: u64 = 12345;
        unsafe {
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            (SEED as f64 / u64::MAX as f64)
        }
    }
}
