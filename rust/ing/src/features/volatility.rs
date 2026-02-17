//! Volatility features

use crate::state::{OrderBook, RingBuffer};

/// Volatility features (8 features)
#[derive(Debug, Clone, Default)]
pub struct VolatilityFeatures {
    /// Realized volatility from 1-minute returns
    pub returns_1m: f64,
    /// Realized volatility from 5-minute returns
    pub returns_5m: f64,
    /// Parkinson volatility (high-low based)
    pub parkinson_5m: f64,
    /// Mean spread over 1 minute
    pub spread_mean_1m: f64,
    /// Spread standard deviation over 1 minute
    pub spread_std_1m: f64,
    /// Midprice standard deviation over 1 minute
    pub midprice_std_1m: f64,
    /// Volatility ratio (short/long)
    pub ratio_short_long: f64,
    /// Volatility z-score vs 1-hour mean
    pub zscore: f64,
}

impl VolatilityFeatures {
    pub fn count() -> usize { 8 }

    pub fn names() -> Vec<&'static str> {
        vec![
            "vol_returns_1m",
            "vol_returns_5m",
            "vol_parkinson_5m",
            "vol_spread_mean_1m",
            "vol_spread_std_1m",
            "vol_midprice_std_1m",
            "vol_ratio_short_long",
            "vol_zscore",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.returns_1m,
            self.returns_5m,
            self.parkinson_5m,
            self.spread_mean_1m,
            self.spread_std_1m,
            self.midprice_std_1m,
            self.ratio_short_long,
            self.zscore,
        ]
    }
}

/// Compute volatility features
pub fn compute(
    price_buffer: &RingBuffer<f64>,
    order_book: &OrderBook,
) -> VolatilityFeatures {
    // Realized volatility from returns
    let returns = price_buffer.returns();

    // 1-minute vol (assuming 100ms intervals, ~600 samples per minute)
    let returns_1m = if returns.len() >= 60 {
        compute_realized_vol(&returns[returns.len()-60..])
    } else {
        compute_realized_vol(&returns)
    };

    // 5-minute vol
    let returns_5m = if returns.len() >= 300 {
        compute_realized_vol(&returns[returns.len()-300..])
    } else {
        compute_realized_vol(&returns)
    };

    // Parkinson volatility (simplified - using price range)
    let parkinson_5m = if price_buffer.len() >= 300 {
        let prices: Vec<f64> = price_buffer.last_n(300)
            .into_iter()
            .cloned()
            .collect();
        compute_parkinson_vol(&prices)
    } else {
        compute_parkinson_vol(&price_buffer.to_vec())
    };

    // Spread statistics (would need spread buffer in real impl)
    let spread_mean_1m = order_book.spread().unwrap_or(0.0);
    let spread_std_1m = 0.0;  // Would need historical spreads

    // Midprice std
    let midprice_std_1m = if price_buffer.len() >= 60 {
        let prices: Vec<f64> = price_buffer.last_n(60)
            .into_iter()
            .cloned()
            .collect();
        std_dev(&prices)
    } else {
        price_buffer.std()
    };

    // Ratio short/long
    let ratio_short_long = if returns_5m > 0.0 {
        returns_1m / returns_5m
    } else {
        1.0
    };

    // Z-score (simplified - would need longer history)
    let zscore = 0.0;  // Would need hourly vol history

    VolatilityFeatures {
        returns_1m,
        returns_5m,
        parkinson_5m,
        spread_mean_1m,
        spread_std_1m,
        midprice_std_1m,
        ratio_short_long,
        zscore,
    }
}

/// Compute realized volatility from returns
fn compute_realized_vol(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let sum_sq: f64 = returns.iter().map(|r| r * r).sum();
    (sum_sq / returns.len() as f64).sqrt()
}

/// Compute Parkinson volatility estimator
fn compute_parkinson_vol(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let high = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let low = prices.iter().cloned().fold(f64::INFINITY, f64::min);

    if low > 0.0 {
        let log_ratio = (high / low).ln();
        log_ratio / (4.0 * std::f64::consts::LN_2).sqrt()
    } else {
        0.0
    }
}

/// Compute standard deviation
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;

    variance.sqrt()
}
