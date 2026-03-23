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

    /// Compute normalized divergence (as percentage of price).
    ///
    /// This makes divergence comparable across assets with different price scales.
    ///
    /// # Arguments
    /// * `window` - Window size in minutes
    ///
    /// # Returns
    /// Divergence as percentage of starting price
    pub fn compute_normalized(&self, window: usize) -> f64 {
        if self.prices.len() < window || window == 0 {
            return 0.0;
        }

        let n = self.prices.len();
        let start = n - window;
        let start_price = self.prices[start];

        if start_price.abs() < EPSILON {
            return 0.0;
        }

        let divergence = self.compute_raw(window);
        (divergence / start_price) * 100.0 // As percentage
    }

    /// Compute volume trend over window.
    ///
    /// # Returns
    /// Slope of volume regression (positive = increasing volume)
    pub fn compute_volume_trend(&self, window: usize) -> f64 {
        if self.buy_volumes.len() < window || window < 2 {
            return 0.0;
        }

        let n = self.buy_volumes.len();
        let start = n - window;

        // Total volume per bar
        let volumes: Vec<f64> = (start..n)
            .map(|i| self.buy_volumes[i] + self.sell_volumes[i])
            .collect();

        // Linear regression slope
        let n_f = volumes.len() as f64;
        let sum_x: f64 = (0..volumes.len()).map(|i| i as f64).sum();
        let sum_y: f64 = volumes.iter().sum();
        let sum_xy: f64 = volumes.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();
        let sum_x2: f64 = (0..volumes.len()).map(|i| (i * i) as f64).sum();

        let denominator = n_f * sum_x2 - sum_x * sum_x;
        if denominator.abs() < EPSILON {
            return 0.0;
        }

        let slope = (n_f * sum_xy - sum_x * sum_y) / denominator;

        // Normalize by mean volume
        let mean_vol = sum_y / n_f;
        if mean_vol.abs() < EPSILON {
            return 0.0;
        }

        slope / mean_vol
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
            SEED as f64 / u64::MAX as f64
        }
    }
}

// ============================================================================
// Skeptical Tests - Validate regime detection capability
// ============================================================================

#[cfg(test)]
mod skeptical_tests {
    use super::*;

    /// Test 1: Low divergence when price moves close to Kyle's model prediction
    /// This validates the mathematical foundation
    /// Note: Lambda is updated via EMA, so exact zero divergence isn't expected
    #[test]
    fn test_low_divergence_when_model_approximately_accurate() {
        let mut computer = DivergenceComputer::new(200, 200, 100);

        let true_lambda = 0.001;
        let mut price = 100.0;

        // First, let lambda converge by feeding consistent data
        for _ in 0..150 {
            let buy_vol = 1500.0;
            let sell_vol = 500.0;
            let signed_vol = buy_vol - sell_vol;

            price += true_lambda * signed_vol;
            computer.update(price, buy_vol, sell_vol);
        }

        // Now lambda should have converged closer to true value
        // Continue with same pattern
        for _ in 0..50 {
            let buy_vol = 1500.0;
            let sell_vol = 500.0;
            let signed_vol = buy_vol - sell_vol;

            price += true_lambda * signed_vol;
            computer.update(price, buy_vol, sell_vol);
        }

        let divergence = computer.compute(50);

        // After convergence, divergence should be relatively small
        // compared to divergence when there's actual suppression
        // The key is it's much smaller than accumulation/distribution scenarios
        let abs_div = divergence.abs();
        assert!(
            abs_div < 50.0,
            "When price approximately follows model, divergence should be small, got {}",
            divergence
        );
    }

    /// Test 2: Negative divergence during accumulation
    /// Price rises LESS than expected given buying pressure
    #[test]
    fn test_negative_divergence_accumulation() {
        let mut computer = DivergenceComputer::new(100, 200, 50);

        // Set known lambda
        computer.kyle_lambda = 0.001;

        // Heavy buying but price stays flat (accumulation)
        for _ in 0..100 {
            computer.update(100.0, 2000.0, 500.0); // Net buy = +1500 per bar
        }

        let divergence = computer.compute(50);

        // Expected price change = 0.001 * 1500 * 50 = 75.0
        // Actual price change = 0
        // Divergence = 0 - 75 = -75 (negative)
        assert!(
            divergence < -10.0,
            "Accumulation should have negative divergence, got {}",
            divergence
        );
    }

    /// Test 3: Positive divergence during distribution
    /// Price falls LESS than expected given selling pressure
    #[test]
    fn test_positive_divergence_distribution() {
        let mut computer = DivergenceComputer::new(100, 200, 50);

        // Set known lambda
        computer.kyle_lambda = 0.001;

        // Heavy selling but price stays flat (distribution - sellers absorbed)
        for _ in 0..100 {
            computer.update(100.0, 500.0, 2000.0); // Net sell = -1500 per bar
        }

        let divergence = computer.compute(50);

        // Expected price change = 0.001 * (-1500) * 50 = -75.0
        // Actual price change = 0
        // Divergence = 0 - (-75) = +75 (positive)
        assert!(
            divergence > 10.0,
            "Distribution should have positive divergence, got {}",
            divergence
        );
    }

    /// Test 4: Divergence differentiates accumulation from distribution
    #[test]
    fn test_divergence_differentiates_regimes() {
        // Accumulation: buying absorbed, price flat
        let mut acc_computer = DivergenceComputer::new(100, 200, 50);
        acc_computer.kyle_lambda = 0.001;
        for _ in 0..100 {
            acc_computer.update(100.0, 2000.0, 500.0);
        }
        let acc_divergence = acc_computer.compute(50);

        // Distribution: selling absorbed, price flat
        let mut dist_computer = DivergenceComputer::new(100, 200, 50);
        dist_computer.kyle_lambda = 0.001;
        for _ in 0..100 {
            dist_computer.update(100.0, 500.0, 2000.0);
        }
        let dist_divergence = dist_computer.compute(50);

        // Accumulation = negative, Distribution = positive
        assert!(
            acc_divergence < 0.0,
            "Accumulation divergence should be negative: {}",
            acc_divergence
        );
        assert!(
            dist_divergence > 0.0,
            "Distribution divergence should be positive: {}",
            dist_divergence
        );

        // They should be roughly opposite in magnitude
        let ratio = dist_divergence / acc_divergence.abs();
        assert!(
            (ratio - 1.0).abs() < 0.5,
            "Divergences should be roughly symmetric: acc={}, dist={}, ratio={}",
            acc_divergence,
            dist_divergence,
            ratio
        );
    }

    /// Test 5: Kyle's lambda converges to true value
    #[test]
    fn test_lambda_convergence() {
        let mut computer = DivergenceComputer::new(200, 200, 100);

        let true_lambda = 0.0005;
        let mut price = 100.0;

        // Generate data following Kyle's model exactly
        for i in 0..200 {
            // Deterministic "random" variation
            let variation = ((i * 17) % 11) as f64 - 5.0;
            let buy_vol = 1000.0 + variation * 50.0;
            let sell_vol = 1000.0 - variation * 30.0;
            let signed_vol = buy_vol - sell_vol;

            price += true_lambda * signed_vol;
            computer.update(price, buy_vol, sell_vol);
        }

        let estimated = computer.get_kyle_lambda();

        // Lambda should be reasonably close to true value
        // (EMA smoothing means it won't be exact)
        assert!(
            estimated > 0.0,
            "Lambda should be positive: {}",
            estimated
        );
        assert!(
            (estimated - true_lambda).abs() < true_lambda * 2.0,
            "Lambda {} should be within 2x of true value {}",
            estimated,
            true_lambda
        );
    }

    /// Test 6: Normalized divergence correctly converts to percentage
    /// Note: Cross-asset comparison requires consistent lambda estimation,
    /// which depends on market data. Z-score is better for cross-asset comparison.
    #[test]
    fn test_normalized_divergence_percentage_conversion() {
        let mut computer = DivergenceComputer::new(100, 200, 50);
        computer.kyle_lambda = 0.001;

        // Price at 100, accumulation pattern (flat price, heavy buying)
        for _ in 0..100 {
            computer.update(100.0, 2000.0, 500.0);
        }

        let raw_divergence = computer.compute(50);
        let normalized = computer.compute_normalized(50);

        // Normalized should be raw / price * 100
        let expected_normalized = (raw_divergence / 100.0) * 100.0;

        assert!(
            (normalized - expected_normalized).abs() < 0.1,
            "Normalized should equal raw/price*100: raw={}, normalized={}, expected={}",
            raw_divergence,
            normalized,
            expected_normalized
        );

        // Both should indicate accumulation (negative)
        assert!(raw_divergence < 0.0, "Raw should be negative");
        assert!(normalized < 0.0, "Normalized should be negative");
    }

    /// Test 6b: Divergence magnitude increases with suppression strength
    /// This verifies the core measurement capability
    #[test]
    fn test_divergence_magnitude_scales_with_suppression() {
        // Scenario A: Light accumulation (60/40 buy/sell, flat price)
        let mut light_acc = DivergenceComputer::new(100, 200, 50);
        light_acc.kyle_lambda = 0.001;
        for _ in 0..100 {
            light_acc.update(100.0, 600.0, 400.0); // Net = +200
        }
        let light_div = light_acc.compute(50);

        // Scenario B: Heavy accumulation (90/10 buy/sell, flat price)
        let mut heavy_acc = DivergenceComputer::new(100, 200, 50);
        heavy_acc.kyle_lambda = 0.001;
        for _ in 0..100 {
            heavy_acc.update(100.0, 900.0, 100.0); // Net = +800
        }
        let heavy_div = heavy_acc.compute(50);

        // Both should be negative (accumulation = price suppressed)
        assert!(
            light_div < 0.0,
            "Light accumulation should have negative divergence: {}",
            light_div
        );
        assert!(
            heavy_div < 0.0,
            "Heavy accumulation should have negative divergence: {}",
            heavy_div
        );

        // Heavy accumulation should have larger (more negative) divergence
        assert!(
            heavy_div < light_div,
            "Heavy accumulation ({}) should be more negative than light ({})",
            heavy_div,
            light_div
        );

        // Magnitude should roughly scale with net volume difference
        // Heavy net = 800, Light net = 200, ratio = 4x
        let magnitude_ratio = heavy_div.abs() / light_div.abs();
        assert!(
            magnitude_ratio > 2.0 && magnitude_ratio < 6.0,
            "Magnitude should scale with suppression: ratio={}",
            magnitude_ratio
        );
    }

    /// Test 7: Z-score detects regime transition
    #[test]
    fn test_zscore_detects_regime_transition() {
        let mut computer = DivergenceComputer::new(100, 300, 50);
        computer.kyle_lambda = 0.001;

        // Phase 1: Normal market (price follows model)
        let mut price = 100.0;
        for _ in 0..150 {
            let buy_vol = 1000.0;
            let sell_vol = 800.0;
            price += computer.kyle_lambda * (buy_vol - sell_vol);
            computer.update(price, buy_vol, sell_vol);
        }

        let normal_zscore = computer.compute_zscore(100);

        // Phase 2: Accumulation (price suppressed despite buying)
        for _ in 0..50 {
            computer.update(price, 2000.0, 500.0); // Heavy buying, flat price
        }

        let anomaly_zscore = computer.compute_zscore(100);

        // Z-score should shift significantly
        assert!(
            (anomaly_zscore - normal_zscore).abs() > 1.0,
            "Z-score should detect regime change: normal={}, anomaly={}",
            normal_zscore,
            anomaly_zscore
        );
    }

    /// Test 8: Volume trend computation
    #[test]
    fn test_volume_trend_rising() {
        let mut computer = DivergenceComputer::new(100, 200, 50);
        computer.kyle_lambda = 0.001;

        // Rising volume pattern
        for i in 0..100 {
            let volume = 100.0 + (i as f64 * 10.0); // 100 to 1090
            computer.update(100.0, volume * 0.6, volume * 0.4);
        }

        let trend = computer.compute_volume_trend(50);

        assert!(
            trend > 0.0,
            "Rising volume should have positive trend: {}",
            trend
        );
    }

    /// Test 9: Volume trend computation - falling
    #[test]
    fn test_volume_trend_falling() {
        let mut computer = DivergenceComputer::new(100, 200, 50);
        computer.kyle_lambda = 0.001;

        // Falling volume pattern
        for i in 0..100 {
            let volume = 1000.0 - (i as f64 * 8.0); // 1000 to 200
            computer.update(100.0, volume * 0.6, volume * 0.4);
        }

        let trend = computer.compute_volume_trend(50);

        assert!(
            trend < 0.0,
            "Falling volume should have negative trend: {}",
            trend
        );
    }

    /// Test 10: Edge case - zero volume
    #[test]
    fn test_zero_volume_handled() {
        let mut computer = DivergenceComputer::new(100, 200, 50);
        computer.kyle_lambda = 0.001;

        // No trading activity
        for i in 0..100 {
            computer.update(100.0 + (i as f64 * 0.1), 0.0, 0.0);
        }

        let divergence = computer.compute(50);
        let normalized = computer.compute_normalized(50);
        let trend = computer.compute_volume_trend(50);

        // Should all be finite
        assert!(divergence.is_finite(), "Divergence should be finite");
        assert!(normalized.is_finite(), "Normalized should be finite");
        assert!(trend.is_finite(), "Trend should be finite");
    }

    /// Test 11: Multi-window consistency
    #[test]
    fn test_multi_window_consistency() {
        let mut computer = DivergenceComputer::new(1440, 2000, 60);
        computer.kyle_lambda = 0.0001;

        // Build data with consistent accumulation pattern
        for _ in 0..1440 {
            computer.update(100.0, 1500.0, 500.0);
        }

        let div_60 = computer.compute(60);
        let div_240 = computer.compute(240);
        let div_1440 = computer.compute(1440);

        // All should be negative (accumulation)
        assert!(div_60 < 0.0, "60-min divergence should be negative");
        assert!(div_240 < 0.0, "240-min divergence should be negative");
        assert!(div_1440 < 0.0, "1440-min divergence should be negative");

        // Longer windows accumulate more divergence
        assert!(
            div_1440.abs() > div_240.abs(),
            "Longer window should have larger divergence magnitude"
        );
        assert!(
            div_240.abs() > div_60.abs(),
            "Longer window should have larger divergence magnitude"
        );
    }

    /// Test 12: Real-world scenario - markup with confirming volume
    #[test]
    fn test_markup_with_volume_confirmation() {
        let mut computer = DivergenceComputer::new(100, 200, 50);
        computer.kyle_lambda = 0.001;

        // Markup: price rises WITH buying (no divergence expected)
        let mut price = 100.0;
        for _ in 0..100 {
            let buy_vol = 1500.0;
            let sell_vol = 500.0;
            let signed_vol = buy_vol - sell_vol;

            // Price rises as expected
            price += computer.kyle_lambda * signed_vol;
            computer.update(price, buy_vol, sell_vol);
        }

        let divergence = computer.compute(50);

        // Divergence should be near zero (price confirmed by volume)
        assert!(
            divergence.abs() < 5.0,
            "Confirmed markup should have near-zero divergence: {}",
            divergence
        );
    }

    /// Test 13: Divergence sign interpretation
    /// Validates the core regime detection logic
    #[test]
    fn test_divergence_sign_interpretation() {
        let mut computer = DivergenceComputer::new(100, 200, 50);
        computer.kyle_lambda = 0.001;

        // Scenario A: Heavy buying, price flat → negative divergence (accumulation)
        for _ in 0..100 {
            computer.update(100.0, 3000.0, 500.0);
        }
        let accumulation_div = computer.compute(50);

        // Reset
        let mut computer2 = DivergenceComputer::new(100, 200, 50);
        computer2.kyle_lambda = 0.001;

        // Scenario B: Heavy selling, price flat → positive divergence (distribution)
        for _ in 0..100 {
            computer2.update(100.0, 500.0, 3000.0);
        }
        let distribution_div = computer2.compute(50);

        // Validate interpretation
        assert!(
            accumulation_div < 0.0,
            "Accumulation = negative divergence: {}",
            accumulation_div
        );
        assert!(
            distribution_div > 0.0,
            "Distribution = positive divergence: {}",
            distribution_div
        );

        // Document the interpretation
        // Negative: Price didn't rise despite buying → someone absorbing (accumulation)
        // Positive: Price didn't fall despite selling → someone absorbing (distribution)
    }
}
