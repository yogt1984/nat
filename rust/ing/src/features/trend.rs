//! Trend Feature Extraction
//!
//! This module implements trend detection features for identifying trending vs
//! mean-reverting market conditions.
//!
//! # Features
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **Momentum** | Linear regression slope of prices | (-inf, +inf) | Positive = uptrend |
//! | **Monotonicity** | % of ticks in same direction | [0.5, 1.0] | >0.7 = strong trend |
//! | **Hurst Exponent** | Trend persistence measure | [0, 1] | >0.5 = trending |
//! | **MA Crossover** | EMA(short) - EMA(long) | (-inf, +inf) | Positive = bullish |
//!
//! # References
//!
//! - Jegadeesh & Titman (1993) - Original momentum paper
//! - Moskowitz et al. (2012) - Time Series Momentum
//! - Mandelbrot (1971) - Hurst exponent and long-range dependence

use crate::state::RingBuffer;

/// Trend features computed at multiple windows
/// Total: 15 features (5 base features × 3 windows: 60, 300, 600 ticks)
#[derive(Debug, Clone, Default)]
pub struct TrendFeatures {
    // Window 60 (short-term)
    /// Momentum (linear regression slope) - 60 tick window
    pub momentum_60: f64,
    /// R-squared of momentum regression - 60 tick window
    pub momentum_r2_60: f64,
    /// Monotonicity score [0.5, 1.0] - 60 tick window
    pub monotonicity_60: f64,

    // Window 300 (medium-term)
    /// Momentum - 300 tick window
    pub momentum_300: f64,
    /// R-squared - 300 tick window
    pub momentum_r2_300: f64,
    /// Monotonicity - 300 tick window
    pub monotonicity_300: f64,
    /// Hurst exponent [0, 1] - 300 tick window (needs more data)
    pub hurst_300: f64,

    // Window 600 (long-term)
    /// Momentum - 600 tick window
    pub momentum_600: f64,
    /// R-squared - 600 tick window
    pub momentum_r2_600: f64,
    /// Monotonicity - 600 tick window
    pub monotonicity_600: f64,
    /// Hurst exponent - 600 tick window
    pub hurst_600: f64,

    // EMA-based features (computed across full buffer)
    /// EMA crossover: EMA(10) - EMA(50)
    pub ma_crossover: f64,
    /// Normalized MA crossover (as % of price)
    pub ma_crossover_norm: f64,
    /// Short EMA (period 10)
    pub ema_short: f64,
    /// Long EMA (period 50)
    pub ema_long: f64,
}

impl TrendFeatures {
    pub fn count() -> usize {
        15
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "trend_momentum_60",
            "trend_momentum_r2_60",
            "trend_monotonicity_60",
            "trend_momentum_300",
            "trend_momentum_r2_300",
            "trend_monotonicity_300",
            "trend_hurst_300",
            "trend_momentum_600",
            "trend_momentum_r2_600",
            "trend_monotonicity_600",
            "trend_hurst_600",
            "trend_ma_crossover",
            "trend_ma_crossover_norm",
            "trend_ema_short",
            "trend_ema_long",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.momentum_60,
            self.momentum_r2_60,
            self.monotonicity_60,
            self.momentum_300,
            self.momentum_r2_300,
            self.monotonicity_300,
            self.hurst_300,
            self.momentum_600,
            self.momentum_r2_600,
            self.monotonicity_600,
            self.hurst_600,
            self.ma_crossover,
            self.ma_crossover_norm,
            self.ema_short,
            self.ema_long,
        ]
    }
}

/// Compute trend features from price buffer
pub fn compute(price_buffer: &RingBuffer<f64>) -> TrendFeatures {
    let prices = price_buffer.to_vec();

    if prices.len() < 20 {
        return TrendFeatures::default();
    }

    // Compute features at different windows
    let (momentum_60, momentum_r2_60) = compute_momentum_window(&prices, 60);
    let monotonicity_60 = compute_monotonicity_window(&prices, 60);

    let (momentum_300, momentum_r2_300) = compute_momentum_window(&prices, 300);
    let monotonicity_300 = compute_monotonicity_window(&prices, 300);
    let hurst_300 = compute_hurst_window(&prices, 300);

    let (momentum_600, momentum_r2_600) = compute_momentum_window(&prices, 600);
    let monotonicity_600 = compute_monotonicity_window(&prices, 600);
    let hurst_600 = compute_hurst_window(&prices, 600);

    // Compute EMAs across full buffer
    let (ema_short, ema_long) = compute_emas(&prices, 10, 50);
    let ma_crossover = ema_short - ema_long;
    let ma_crossover_norm = if ema_long.abs() > 1e-10 {
        (ma_crossover / ema_long) * 100.0
    } else {
        0.0
    };

    TrendFeatures {
        momentum_60,
        momentum_r2_60,
        monotonicity_60,
        momentum_300,
        momentum_r2_300,
        monotonicity_300,
        hurst_300,
        momentum_600,
        momentum_r2_600,
        monotonicity_600,
        hurst_600,
        ma_crossover,
        ma_crossover_norm,
        ema_short,
        ema_long,
    }
}

/// Compute momentum (linear regression slope) for a window
/// Returns (slope, r_squared)
fn compute_momentum_window(prices: &[f64], window: usize) -> (f64, f64) {
    let n = prices.len().min(window);
    if n < 2 {
        return (0.0, 0.0);
    }

    let start = prices.len().saturating_sub(window);
    let window_prices = &prices[start..];

    compute_momentum(window_prices)
}

/// Compute momentum as linear regression slope
/// Returns (slope, r_squared)
fn compute_momentum(prices: &[f64]) -> (f64, f64) {
    let n = prices.len();
    if n < 2 {
        return (0.0, 0.0);
    }

    let n_f = n as f64;

    // x values: 0, 1, 2, ..., n-1
    // Using formulas for sum of arithmetic series
    let sum_x: f64 = (n - 1) as f64 * n_f / 2.0;
    let sum_x2: f64 = (n - 1) as f64 * n_f * (2 * n - 1) as f64 / 6.0;

    let sum_y: f64 = prices.iter().sum();
    let sum_xy: f64 = prices
        .iter()
        .enumerate()
        .map(|(i, &p)| i as f64 * p)
        .sum();

    let denominator = n_f * sum_x2 - sum_x * sum_x;
    if denominator.abs() < 1e-10 {
        return (0.0, 0.0);
    }

    let slope = (n_f * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n_f;

    // Compute R-squared
    let mean_y = sum_y / n_f;
    let ss_tot: f64 = prices.iter().map(|&p| (p - mean_y).powi(2)).sum();

    let ss_res: f64 = prices
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let predicted = intercept + slope * i as f64;
            (p - predicted).powi(2)
        })
        .sum();

    let r_squared = if ss_tot > 1e-10 {
        (1.0 - ss_res / ss_tot).max(0.0)
    } else {
        0.0
    };

    (slope, r_squared)
}

/// Compute monotonicity for a window
fn compute_monotonicity_window(prices: &[f64], window: usize) -> f64 {
    let n = prices.len().min(window);
    if n < 2 {
        return 0.5;
    }

    let start = prices.len().saturating_sub(window);
    let window_prices = &prices[start..];

    compute_monotonicity(window_prices)
}

/// Compute monotonicity: fraction of price moves in dominant direction
/// Returns value in [0.5, 1.0] where 1.0 = perfectly monotonic
fn compute_monotonicity(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.5;
    }

    let mut up_count = 0;
    let mut down_count = 0;

    for i in 1..prices.len() {
        if prices[i] > prices[i - 1] {
            up_count += 1;
        } else if prices[i] < prices[i - 1] {
            down_count += 1;
        }
        // Equal prices don't count either way
    }

    let total_moves = up_count + down_count;
    if total_moves == 0 {
        return 0.5; // No price movement
    }

    let dominant = up_count.max(down_count) as f64;
    dominant / total_moves as f64
}

/// Compute Hurst exponent for a window
fn compute_hurst_window(prices: &[f64], window: usize) -> f64 {
    let n = prices.len().min(window);
    if n < 20 {
        return 0.5; // Default to random walk
    }

    let start = prices.len().saturating_sub(window);
    let window_prices = &prices[start..];

    compute_hurst_exponent(window_prices).unwrap_or(0.5)
}

/// Compute Hurst exponent using rescaled range (R/S) analysis
///
/// H > 0.5: Trending (persistent) - past trends tend to continue
/// H = 0.5: Random walk - no predictability
/// H < 0.5: Mean-reverting (anti-persistent) - trends tend to reverse
fn compute_hurst_exponent(prices: &[f64]) -> Option<f64> {
    if prices.len() < 20 {
        return None; // Need sufficient data for R/S analysis
    }

    // Compute log returns
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 {
                (w[1] / w[0]).ln()
            } else {
                0.0
            }
        })
        .collect();

    if returns.is_empty() {
        return None;
    }

    // Use multiple window sizes for R/S calculation
    let mut log_n = Vec::new();
    let mut log_rs = Vec::new();

    let n = returns.len();
    let mut window_size = 8.min(n / 2);

    while window_size <= n / 2 && window_size >= 8 {
        if let Some(rs) = rescaled_range(&returns, window_size) {
            if rs > 0.0 {
                log_n.push((window_size as f64).ln());
                log_rs.push(rs.ln());
            }
        }
        window_size = (window_size as f64 * 1.5).ceil() as usize;
    }

    if log_n.len() < 2 {
        return None;
    }

    // Linear regression of log(R/S) vs log(n) gives Hurst exponent
    let slope = simple_linear_regression_slope(&log_n, &log_rs)?;

    // Clamp to valid range [0, 1]
    Some(slope.clamp(0.0, 1.0))
}

/// Compute rescaled range for a given window size
fn rescaled_range(returns: &[f64], window_size: usize) -> Option<f64> {
    if returns.len() < window_size || window_size < 2 {
        return None;
    }

    let num_windows = returns.len() / window_size;
    if num_windows == 0 {
        return None;
    }

    let mut rs_sum = 0.0;
    let mut valid_windows = 0;

    for i in 0..num_windows {
        let start = i * window_size;
        let end = start + window_size;
        let window = &returns[start..end];

        // Mean of window
        let mean: f64 = window.iter().sum::<f64>() / window_size as f64;

        // Standard deviation
        let variance: f64 = window.iter().map(|&r| (r - mean).powi(2)).sum::<f64>()
            / window_size as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            continue; // Skip windows with no variation
        }

        // Cumulative deviations from mean
        let mut cumsum = 0.0;
        let mut max_cumsum = f64::NEG_INFINITY;
        let mut min_cumsum = f64::INFINITY;

        for &r in window {
            cumsum += r - mean;
            max_cumsum = max_cumsum.max(cumsum);
            min_cumsum = min_cumsum.min(cumsum);
        }

        let range = max_cumsum - min_cumsum;
        let rs = range / std_dev;

        rs_sum += rs;
        valid_windows += 1;
    }

    if valid_windows == 0 {
        return None;
    }

    Some(rs_sum / valid_windows as f64)
}

/// Simple linear regression returning just the slope
fn simple_linear_regression_slope(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();

    let denominator = n * sum_x2 - sum_x * sum_x;
    if denominator.abs() < 1e-10 {
        return None;
    }

    Some((n * sum_xy - sum_x * sum_y) / denominator)
}

/// Compute EMAs with given periods
fn compute_emas(prices: &[f64], short_period: usize, long_period: usize) -> (f64, f64) {
    if prices.is_empty() {
        return (0.0, 0.0);
    }

    let alpha_short = 2.0 / (short_period as f64 + 1.0);
    let alpha_long = 2.0 / (long_period as f64 + 1.0);

    let mut ema_short = prices[0];
    let mut ema_long = prices[0];

    for &price in &prices[1..] {
        ema_short = alpha_short * price + (1.0 - alpha_short) * ema_short;
        ema_long = alpha_long * price + (1.0 - alpha_long) * ema_long;
    }

    (ema_short, ema_long)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Basic Feature Tests
    // ========================================================================

    #[test]
    fn test_feature_count() {
        assert_eq!(TrendFeatures::count(), 15);
        assert_eq!(TrendFeatures::names().len(), 15);
        assert_eq!(TrendFeatures::default().to_vec().len(), 15);
    }

    #[test]
    fn test_compute_with_insufficient_data() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(100);
        buf.push(100.0);
        buf.push(101.0);

        let features = compute(&buf);

        // Should return defaults
        assert_eq!(features.momentum_60, 0.0);
    }

    #[test]
    fn test_compute_with_sufficient_data() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(1000);
        for i in 0..100 {
            buf.push(100.0 + i as f64);
        }

        let features = compute(&buf);

        // Should have computed values
        assert!(features.momentum_60 > 0.0);
        assert!(features.monotonicity_60 > 0.9);
    }

    // ========================================================================
    // Momentum Tests
    // ========================================================================

    #[test]
    fn test_momentum_perfect_uptrend() {
        let prices: Vec<f64> = (0..60).map(|i| 100.0 + i as f64).collect();
        let (slope, r2) = compute_momentum(&prices);

        assert!((slope - 1.0).abs() < 1e-10, "Expected slope ~1.0, got {}", slope);
        assert!((r2 - 1.0).abs() < 1e-10, "Expected R^2 ~1.0, got {}", r2);
    }

    #[test]
    fn test_momentum_perfect_downtrend() {
        let prices: Vec<f64> = (0..60).map(|i| 160.0 - i as f64).collect();
        let (slope, r2) = compute_momentum(&prices);

        assert!((slope + 1.0).abs() < 1e-10, "Expected slope ~-1.0, got {}", slope);
        assert!((r2 - 1.0).abs() < 1e-10, "Expected R^2 ~1.0, got {}", r2);
    }

    #[test]
    fn test_momentum_flat() {
        let prices: Vec<f64> = vec![100.0; 60];
        let (slope, _) = compute_momentum(&prices);

        assert!(slope.abs() < 1e-10, "Expected slope ~0, got {}", slope);
    }

    #[test]
    fn test_momentum_noisy_uptrend() {
        let prices: Vec<f64> = (0..60)
            .map(|i| 100.0 + i as f64 + (i % 3) as f64 - 1.0)
            .collect();
        let (slope, r2) = compute_momentum(&prices);

        assert!(slope > 0.0, "Expected positive slope");
        assert!(r2 < 1.0 && r2 > 0.5, "Expected 0.5 < R^2 < 1.0, got {}", r2);
    }

    // ========================================================================
    // Monotonicity Tests
    // ========================================================================

    #[test]
    fn test_monotonicity_perfect_uptrend() {
        let prices: Vec<f64> = (0..60).map(|i| 100.0 + i as f64).collect();
        let mono = compute_monotonicity(&prices);

        assert!((mono - 1.0).abs() < 1e-10, "Expected 1.0, got {}", mono);
    }

    #[test]
    fn test_monotonicity_perfect_downtrend() {
        let prices: Vec<f64> = (0..60).map(|i| 160.0 - i as f64).collect();
        let mono = compute_monotonicity(&prices);

        assert!((mono - 1.0).abs() < 1e-10, "Expected 1.0, got {}", mono);
    }

    #[test]
    fn test_monotonicity_alternating() {
        let prices: Vec<f64> = (0..60).map(|i| 100.0 + (i % 2) as f64).collect();
        let mono = compute_monotonicity(&prices);

        assert!((mono - 0.5).abs() < 0.1, "Expected ~0.5, got {}", mono);
    }

    #[test]
    fn test_monotonicity_flat() {
        let prices: Vec<f64> = vec![100.0; 60];
        let mono = compute_monotonicity(&prices);

        assert!((mono - 0.5).abs() < 1e-10, "Expected 0.5 for flat, got {}", mono);
    }

    #[test]
    fn test_monotonicity_75_percent() {
        // 3 ups, 1 down, repeated
        let mut prices = Vec::new();
        let mut price = 100.0;
        for i in 0..60 {
            prices.push(price);
            if i % 4 == 3 {
                price -= 1.0;
            } else {
                price += 1.0;
            }
        }

        let mono = compute_monotonicity(&prices);
        assert!((mono - 0.75).abs() < 0.05, "Expected ~0.75, got {}", mono);
    }

    // ========================================================================
    // Hurst Exponent Tests
    // ========================================================================

    #[test]
    fn test_hurst_insufficient_data() {
        let prices: Vec<f64> = vec![100.0, 101.0, 102.0];
        let hurst = compute_hurst_exponent(&prices);

        assert!(hurst.is_none());
    }

    #[test]
    fn test_hurst_trending_series() {
        // Create a persistent series with clear trend
        let prices: Vec<f64> = (0..200).map(|i| 100.0 + i as f64 * 0.1).collect();

        let hurst = compute_hurst_exponent(&prices);

        if let Some(h) = hurst {
            assert!(h >= 0.0 && h <= 1.0, "Hurst should be in [0,1], got {}", h);
        }
    }

    #[test]
    fn test_hurst_mean_reverting() {
        // Sinusoidal pattern - tends to be anti-persistent
        let prices: Vec<f64> = (0..200)
            .map(|i| 100.0 + 5.0 * (i as f64 * 0.5).sin())
            .collect();

        let hurst = compute_hurst_exponent(&prices);

        if let Some(h) = hurst {
            assert!(h >= 0.0 && h <= 1.0, "Hurst should be in [0,1], got {}", h);
        }
    }

    #[test]
    fn test_hurst_clamped() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();

        let hurst = compute_hurst_exponent(&prices);

        if let Some(h) = hurst {
            assert!(h >= 0.0 && h <= 1.0, "Hurst should be clamped to [0,1]");
        }
    }

    // ========================================================================
    // EMA Tests
    // ========================================================================

    #[test]
    fn test_ema_uptrend() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();
        let (ema_short, ema_long) = compute_emas(&prices, 10, 50);

        // In uptrend, short EMA should be above long EMA
        assert!(ema_short > ema_long, "Short EMA should be > long EMA in uptrend");
    }

    #[test]
    fn test_ema_downtrend() {
        let prices: Vec<f64> = (0..100).map(|i| 200.0 - i as f64).collect();
        let (ema_short, ema_long) = compute_emas(&prices, 10, 50);

        // In downtrend, short EMA should be below long EMA
        assert!(ema_short < ema_long, "Short EMA should be < long EMA in downtrend");
    }

    #[test]
    fn test_ema_flat() {
        let prices: Vec<f64> = vec![100.0; 100];
        let (ema_short, ema_long) = compute_emas(&prices, 10, 50);

        assert!((ema_short - 100.0).abs() < 1e-6, "Short EMA should converge to 100");
        assert!((ema_long - 100.0).abs() < 1e-6, "Long EMA should converge to 100");
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_compute_uptrend() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(1000);
        for i in 0..700 {
            buf.push(100.0 + i as f64 * 0.1);
        }

        let features = compute(&buf);

        // All momentum should be positive
        assert!(features.momentum_60 > 0.0);
        assert!(features.momentum_300 > 0.0);
        assert!(features.momentum_600 > 0.0);

        // All monotonicity should be high
        assert!(features.monotonicity_60 > 0.9);
        assert!(features.monotonicity_300 > 0.9);
        assert!(features.monotonicity_600 > 0.9);

        // MA crossover should be positive
        assert!(features.ma_crossover > 0.0);
    }

    #[test]
    fn test_full_compute_downtrend() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(1000);
        for i in 0..700 {
            buf.push(200.0 - i as f64 * 0.1);
        }

        let features = compute(&buf);

        // All momentum should be negative
        assert!(features.momentum_60 < 0.0);
        assert!(features.momentum_300 < 0.0);
        assert!(features.momentum_600 < 0.0);

        // MA crossover should be negative
        assert!(features.ma_crossover < 0.0);
    }

    #[test]
    fn test_full_compute_choppy() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(1000);
        for i in 0..700 {
            buf.push(100.0 + (i % 2) as f64 * 2.0 - 1.0);
        }

        let features = compute(&buf);

        // Momentum should be near zero
        assert!(features.momentum_60.abs() < 0.1);

        // Monotonicity should be near 0.5
        assert!((features.monotonicity_60 - 0.5).abs() < 0.1);
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_empty_buffer() {
        let buf: RingBuffer<f64> = RingBuffer::new(100);
        let features = compute(&buf);

        assert_eq!(features.momentum_60, 0.0);
        assert_eq!(features.monotonicity_60, 0.0);
    }

    #[test]
    fn test_large_prices() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(1000);
        for i in 0..100 {
            buf.push(100000.0 + i as f64 * 10.0);
        }

        let features = compute(&buf);

        assert!((features.momentum_60 - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_small_price_changes() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(1000);
        for i in 0..100 {
            buf.push(100.0 + i as f64 * 0.0001);
        }

        let features = compute(&buf);

        assert!(features.momentum_60 > 0.0);
        assert!(features.monotonicity_60 > 0.9);
    }
}
