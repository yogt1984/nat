//! Toxicity Feature Extraction
//!
//! This module implements order flow toxicity measures that detect informed
//! trading and adverse selection in the market.
//!
//! # Features
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **VPIN** | Volume-synchronized probability of informed trading | [0, 1] | Higher = more toxic |
//! | **Adverse Selection** | Information asymmetry measure | [0, +inf) | Higher = more informed flow |
//! | **Toxicity Index** | Composite toxicity score | [0, 1] | Higher = more dangerous |
//!
//! # References
//!
//! - Easley, Lopez de Prado, O'Hara (2012) - Flow Toxicity and Liquidity
//! - Glosten & Milgrom (1985) - Bid, Ask and Transaction Prices
//! - Kyle (1985) - Continuous Auctions and Insider Trading

use crate::state::{TradeBuffer, Trade};

/// Minimum trades required for reliable computation
const MIN_TRADES: usize = 20;
const MIN_BUCKETS_VPIN: usize = 10;

/// Toxicity features
/// Total: 10 features
#[derive(Debug, Clone, Default)]
pub struct ToxicityFeatures {
    // VPIN variants
    /// VPIN with 10 volume buckets
    pub vpin_10: f64,
    /// VPIN with 50 volume buckets
    pub vpin_50: f64,
    /// VPIN rate of change (current - previous window)
    pub vpin_roc: f64,

    // Adverse selection measures
    /// Adverse selection component (realized spread decomposition)
    pub adverse_selection: f64,
    /// Effective spread (trade price vs midpoint proxy)
    pub effective_spread: f64,
    /// Realized spread (short-term price impact)
    pub realized_spread: f64,

    // Order flow imbalance toxicity
    /// Order flow imbalance (buy - sell) / total
    pub flow_imbalance: f64,
    /// Absolute flow imbalance
    pub flow_imbalance_abs: f64,

    // Composite measures
    /// Composite toxicity index [0, 1]
    pub toxicity_index: f64,
    /// Trade count used
    pub trade_count: usize,
}

impl ToxicityFeatures {
    pub fn count() -> usize {
        10
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "toxic_vpin_10",
            "toxic_vpin_50",
            "toxic_vpin_roc",
            "toxic_adverse_selection",
            "toxic_effective_spread",
            "toxic_realized_spread",
            "toxic_flow_imbalance",
            "toxic_flow_imbalance_abs",
            "toxic_index",
            "toxic_trade_count",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.vpin_10,
            self.vpin_50,
            self.vpin_roc,
            self.adverse_selection,
            self.effective_spread,
            self.realized_spread,
            self.flow_imbalance,
            self.flow_imbalance_abs,
            self.toxicity_index,
            self.trade_count as f64,
        ]
    }
}

/// Compute toxicity features from trade buffer
pub fn compute(trade_buffer: &TradeBuffer) -> ToxicityFeatures {
    let trades: Vec<_> = trade_buffer.iter().collect();
    let trade_count = trades.len();

    if trade_count < MIN_TRADES {
        return ToxicityFeatures {
            trade_count,
            ..Default::default()
        };
    }

    // Extract trade data with directions
    let trade_data = extract_trade_data(&trades);

    // Compute VPIN at different bucket sizes
    let vpin_10 = compute_vpin(&trade_data, 10);
    let vpin_50 = compute_vpin(&trade_data, 50);

    // VPIN rate of change (compare first half vs second half)
    let vpin_roc = compute_vpin_roc(&trade_data);

    // Adverse selection measures
    let adverse_selection = compute_adverse_selection(&trade_data);
    let (effective_spread, realized_spread) = compute_spread_components(&trade_data);

    // Order flow imbalance
    let (flow_imbalance, flow_imbalance_abs) = compute_flow_imbalance(&trade_data);

    // Composite toxicity index
    let toxicity_index = compute_toxicity_index(vpin_50, adverse_selection, flow_imbalance_abs);

    ToxicityFeatures {
        vpin_10,
        vpin_50,
        vpin_roc,
        adverse_selection,
        effective_spread,
        realized_spread,
        flow_imbalance,
        flow_imbalance_abs,
        toxicity_index,
        trade_count,
    }
}

/// Extracted trade data for toxicity computation
struct TradeData {
    prices: Vec<f64>,
    volumes: Vec<f64>,
    directions: Vec<i8>, // +1 buy, -1 sell
    is_buy: Vec<bool>,
}

/// Extract trade data with tick-rule classification
fn extract_trade_data(trades: &[&Trade]) -> TradeData {
    let mut prices = Vec::with_capacity(trades.len());
    let mut volumes = Vec::with_capacity(trades.len());
    let mut directions = Vec::with_capacity(trades.len());
    let mut is_buy = Vec::with_capacity(trades.len());
    let mut last_price: Option<f64> = None;

    for trade in trades {
        prices.push(trade.price);
        volumes.push(trade.size);
        is_buy.push(trade.is_buy);

        // Tick rule classification: use price change, fallback to aggressor side
        let direction = match last_price {
            Some(prev) if trade.price > prev => 1i8,
            Some(prev) if trade.price < prev => -1i8,
            _ => if trade.is_buy { 1i8 } else { -1i8 },
        };
        directions.push(direction);
        last_price = Some(trade.price);
    }

    TradeData {
        prices,
        volumes,
        directions,
        is_buy,
    }
}

/// Compute VPIN (Volume-synchronized Probability of Informed Trading)
///
/// VPIN buckets trades by volume, then measures the average absolute
/// order imbalance across buckets.
///
/// VPIN = Σ|V_buy - V_sell| / (n_buckets * bucket_volume)
fn compute_vpin(data: &TradeData, n_buckets: usize) -> f64 {
    if data.volumes.is_empty() || n_buckets == 0 {
        return 0.0;
    }

    let total_volume: f64 = data.volumes.iter().sum();
    if total_volume <= 0.0 {
        return 0.0;
    }

    let bucket_size = total_volume / n_buckets as f64;
    if bucket_size <= 0.0 {
        return 0.0;
    }

    let mut buckets_buy = Vec::with_capacity(n_buckets);
    let mut buckets_sell = Vec::with_capacity(n_buckets);

    let mut current_buy = 0.0;
    let mut current_sell = 0.0;
    let mut current_volume = 0.0;

    for i in 0..data.volumes.len() {
        let vol = data.volumes[i];
        let dir = data.directions[i];

        if dir > 0 {
            current_buy += vol;
        } else {
            current_sell += vol;
        }
        current_volume += vol;

        // Check if bucket is full
        if current_volume >= bucket_size {
            buckets_buy.push(current_buy);
            buckets_sell.push(current_sell);
            current_buy = 0.0;
            current_sell = 0.0;
            current_volume = 0.0;
        }
    }

    // Add final partial bucket if significant
    if current_volume > bucket_size * 0.5 {
        buckets_buy.push(current_buy);
        buckets_sell.push(current_sell);
    }

    if buckets_buy.len() < MIN_BUCKETS_VPIN {
        return 0.0;
    }

    // Compute VPIN: average absolute imbalance
    let total_imbalance: f64 = buckets_buy
        .iter()
        .zip(&buckets_sell)
        .map(|(b, s)| (b - s).abs())
        .sum();

    let total_bucket_volume: f64 = buckets_buy
        .iter()
        .zip(&buckets_sell)
        .map(|(b, s)| b + s)
        .sum();

    if total_bucket_volume > 0.0 {
        total_imbalance / total_bucket_volume
    } else {
        0.0
    }
}

/// Compute VPIN rate of change (momentum of toxicity)
fn compute_vpin_roc(data: &TradeData) -> f64 {
    let n = data.volumes.len();
    if n < 40 {
        return 0.0;
    }

    let mid = n / 2;

    // Create first half data
    let first_half = TradeData {
        prices: data.prices[..mid].to_vec(),
        volumes: data.volumes[..mid].to_vec(),
        directions: data.directions[..mid].to_vec(),
        is_buy: data.is_buy[..mid].to_vec(),
    };

    // Create second half data
    let second_half = TradeData {
        prices: data.prices[mid..].to_vec(),
        volumes: data.volumes[mid..].to_vec(),
        directions: data.directions[mid..].to_vec(),
        is_buy: data.is_buy[mid..].to_vec(),
    };

    let vpin_first = compute_vpin(&first_half, 10);
    let vpin_second = compute_vpin(&second_half, 10);

    vpin_second - vpin_first
}

/// Compute adverse selection component
///
/// Adverse selection = correlation between trade direction and subsequent price move
/// High adverse selection means trades predict future price movements (informed trading)
fn compute_adverse_selection(data: &TradeData) -> f64 {
    let n = data.prices.len();
    if n < 10 {
        return 0.0;
    }

    // Look at 5-trade ahead price moves
    let lookahead = 5.min(n / 4);
    if lookahead == 0 {
        return 0.0;
    }

    let mut sum_direction_return = 0.0;
    let mut count = 0;

    for i in 0..(n - lookahead) {
        let direction = data.directions[i] as f64;
        let price_now = data.prices[i];
        let price_future = data.prices[i + lookahead];

        if price_now > 0.0 {
            let future_return = (price_future - price_now) / price_now;
            // Positive correlation = adverse selection (trades predict moves)
            sum_direction_return += direction * future_return;
            count += 1;
        }
    }

    if count > 0 {
        // Scale to make interpretable (multiply by 10000 for bps-like scale)
        (sum_direction_return / count as f64) * 10000.0
    } else {
        0.0
    }
}

/// Compute effective and realized spread components
///
/// Effective spread: 2 * |trade_price - midpoint| (using VWAP as proxy)
/// Realized spread: effective spread - price impact
fn compute_spread_components(data: &TradeData) -> (f64, f64) {
    let n = data.prices.len();
    if n < 10 {
        return (0.0, 0.0);
    }

    // Use VWAP as midpoint proxy
    let total_notional: f64 = data.prices.iter().zip(&data.volumes).map(|(p, v)| p * v).sum();
    let total_volume: f64 = data.volumes.iter().sum();

    if total_volume <= 0.0 {
        return (0.0, 0.0);
    }

    let vwap = total_notional / total_volume;

    // Compute effective spread: average |trade_price - vwap| * 2
    let sum_deviation: f64 = data.prices.iter().map(|p| (p - vwap).abs()).sum();
    let effective_spread = 2.0 * sum_deviation / n as f64;

    // Compute realized spread using 5-trade price reversal
    let lookahead = 5.min(n / 4);
    if lookahead == 0 {
        return (effective_spread, 0.0);
    }

    let mut sum_realized = 0.0;
    let mut count = 0;

    for i in 0..(n - lookahead) {
        let direction = data.directions[i] as f64;
        let trade_price = data.prices[i];
        let future_price = data.prices[i + lookahead];

        // Realized spread = direction * (trade_price - future_price) * 2
        // Positive if price reverts (market maker profit)
        sum_realized += direction * (trade_price - future_price) * 2.0;
        count += 1;
    }

    let realized_spread = if count > 0 {
        sum_realized / count as f64
    } else {
        0.0
    };

    (effective_spread, realized_spread)
}

/// Compute order flow imbalance
fn compute_flow_imbalance(data: &TradeData) -> (f64, f64) {
    let mut buy_volume = 0.0;
    let mut sell_volume = 0.0;

    for i in 0..data.volumes.len() {
        if data.directions[i] > 0 {
            buy_volume += data.volumes[i];
        } else {
            sell_volume += data.volumes[i];
        }
    }

    let total = buy_volume + sell_volume;
    if total <= 0.0 {
        return (0.0, 0.0);
    }

    let imbalance = (buy_volume - sell_volume) / total;
    (imbalance, imbalance.abs())
}

/// Compute composite toxicity index [0, 1]
fn compute_toxicity_index(vpin: f64, adverse_selection: f64, flow_imbalance_abs: f64) -> f64 {
    // Normalize components to [0, 1] range
    let vpin_norm = vpin.clamp(0.0, 1.0);

    // Adverse selection: assume typical range [-100, 100] bps
    let adverse_norm = (adverse_selection.abs() / 100.0).clamp(0.0, 1.0);

    // Flow imbalance already in [0, 1]
    let flow_norm = flow_imbalance_abs.clamp(0.0, 1.0);

    // Weighted average (VPIN is the primary measure)
    0.5 * vpin_norm + 0.3 * adverse_norm + 0.2 * flow_norm
}

// ============================================================================
// Skeptical Tests Module
// ============================================================================

/// Module for skeptical statistical tests on toxicity features
pub mod skeptical_tests {
    //! Skeptical tests to validate toxicity feature effectiveness
    //!
    //! These tests verify that:
    //! 1. High VPIN predicts large price moves
    //! 2. VPIN is not just a proxy for volatility
    //! 3. Adverse selection correlates with whale activity

    /// Result of VPIN predictive power test
    #[derive(Debug, Clone)]
    pub struct VpinPredictiveTest {
        pub correlation_with_future_vol: f64,
        pub high_vpin_move_magnitude: f64,
        pub low_vpin_move_magnitude: f64,
        pub lift: f64,
        pub sample_size: usize,
        pub significant: bool,
    }

    /// Result of VPIN vs volatility independence test
    #[derive(Debug, Clone)]
    pub struct VpinVolatilityIndependenceTest {
        pub correlation: f64,
        pub partial_correlation: f64, // VPIN-future_vol controlling for current_vol
        pub is_independent: bool,     // partial_corr significantly different from 0
        pub sample_size: usize,
    }

    /// Result of adverse selection and whale activity test
    #[derive(Debug, Clone)]
    pub struct AdverseSelectionWhaleTest {
        pub correlation: f64,
        pub adverse_selection_high_whale: f64,
        pub adverse_selection_low_whale: f64,
        pub lift: f64,
        pub sample_size: usize,
        pub significant: bool,
    }

    /// Test if high VPIN predicts large price moves
    pub fn test_vpin_predictive_power(
        vpin_values: &[f64],
        future_price_moves: &[f64], // Absolute price moves
        threshold_percentile: f64,
    ) -> VpinPredictiveTest {
        let n = vpin_values.len().min(future_price_moves.len());

        if n < 50 {
            return VpinPredictiveTest {
                correlation_with_future_vol: 0.0,
                high_vpin_move_magnitude: 0.0,
                low_vpin_move_magnitude: 0.0,
                lift: 1.0,
                sample_size: n,
                significant: false,
            };
        }

        // Compute correlation
        let correlation = pearson_correlation(&vpin_values[..n], &future_price_moves[..n]);

        // Compute threshold
        let mut sorted_vpin: Vec<f64> = vpin_values[..n].to_vec();
        sorted_vpin.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = ((n as f64) * threshold_percentile / 100.0) as usize;
        let vpin_threshold = sorted_vpin.get(threshold_idx.min(n - 1)).copied().unwrap_or(0.0);

        // Compare high vs low VPIN price moves
        let mut high_vpin_sum = 0.0;
        let mut high_vpin_count = 0;
        let mut low_vpin_sum = 0.0;
        let mut low_vpin_count = 0;

        for i in 0..n {
            if vpin_values[i] >= vpin_threshold {
                high_vpin_sum += future_price_moves[i].abs();
                high_vpin_count += 1;
            } else {
                low_vpin_sum += future_price_moves[i].abs();
                low_vpin_count += 1;
            }
        }

        let high_vpin_move = if high_vpin_count > 0 {
            high_vpin_sum / high_vpin_count as f64
        } else {
            0.0
        };

        let low_vpin_move = if low_vpin_count > 0 {
            low_vpin_sum / low_vpin_count as f64
        } else {
            0.0
        };

        let lift = if low_vpin_move > 0.0 {
            high_vpin_move / low_vpin_move
        } else {
            1.0
        };

        VpinPredictiveTest {
            correlation_with_future_vol: correlation,
            high_vpin_move_magnitude: high_vpin_move,
            low_vpin_move_magnitude: low_vpin_move,
            lift,
            sample_size: n,
            significant: lift > 1.2 && correlation > 0.1,
        }
    }

    /// Test if VPIN provides information beyond current volatility
    pub fn test_vpin_volatility_independence(
        vpin_values: &[f64],
        current_volatility: &[f64],
        future_volatility: &[f64],
    ) -> VpinVolatilityIndependenceTest {
        let n = vpin_values
            .len()
            .min(current_volatility.len())
            .min(future_volatility.len());

        if n < 50 {
            return VpinVolatilityIndependenceTest {
                correlation: 0.0,
                partial_correlation: 0.0,
                is_independent: false,
                sample_size: n,
            };
        }

        let vpin = &vpin_values[..n];
        let current_vol = &current_volatility[..n];
        let future_vol = &future_volatility[..n];

        // Simple correlation: VPIN vs current volatility
        let correlation = pearson_correlation(vpin, current_vol);

        // Partial correlation: VPIN vs future_vol, controlling for current_vol
        // Using regression residuals approach
        let partial_correlation = compute_partial_correlation(vpin, future_vol, current_vol);

        // VPIN is independent if partial correlation is significant
        // (provides info beyond current volatility)
        let is_independent = partial_correlation.abs() > 0.1;

        VpinVolatilityIndependenceTest {
            correlation,
            partial_correlation,
            is_independent,
            sample_size: n,
        }
    }

    /// Test if adverse selection correlates with whale activity
    pub fn test_adverse_selection_whale_correlation(
        adverse_selection: &[f64],
        whale_activity: &[f64], // e.g., whale volume fraction
        threshold_percentile: f64,
    ) -> AdverseSelectionWhaleTest {
        let n = adverse_selection.len().min(whale_activity.len());

        if n < 50 {
            return AdverseSelectionWhaleTest {
                correlation: 0.0,
                adverse_selection_high_whale: 0.0,
                adverse_selection_low_whale: 0.0,
                lift: 1.0,
                sample_size: n,
                significant: false,
            };
        }

        let correlation = pearson_correlation(&adverse_selection[..n], &whale_activity[..n]);

        // Compute whale threshold
        let mut sorted_whale: Vec<f64> = whale_activity[..n].to_vec();
        sorted_whale.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = ((n as f64) * threshold_percentile / 100.0) as usize;
        let whale_threshold = sorted_whale.get(threshold_idx.min(n - 1)).copied().unwrap_or(0.0);

        // Compare adverse selection in high vs low whale periods
        let mut high_whale_sum = 0.0;
        let mut high_whale_count = 0;
        let mut low_whale_sum = 0.0;
        let mut low_whale_count = 0;

        for i in 0..n {
            if whale_activity[i] >= whale_threshold {
                high_whale_sum += adverse_selection[i].abs();
                high_whale_count += 1;
            } else {
                low_whale_sum += adverse_selection[i].abs();
                low_whale_count += 1;
            }
        }

        let high_whale_adverse = if high_whale_count > 0 {
            high_whale_sum / high_whale_count as f64
        } else {
            0.0
        };

        let low_whale_adverse = if low_whale_count > 0 {
            low_whale_sum / low_whale_count as f64
        } else {
            0.0
        };

        let lift = if low_whale_adverse > 0.0 {
            high_whale_adverse / low_whale_adverse
        } else {
            1.0
        };

        AdverseSelectionWhaleTest {
            correlation,
            adverse_selection_high_whale: high_whale_adverse,
            adverse_selection_low_whale: low_whale_adverse,
            lift,
            sample_size: n,
            significant: correlation.abs() > 0.2 || lift > 1.3,
        }
    }

    /// Compute Pearson correlation coefficient
    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < 2 {
            return 0.0;
        }

        let mean_x: f64 = x[..n].iter().sum::<f64>() / n as f64;
        let mean_y: f64 = y[..n].iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom < 1e-15 {
            return 0.0;
        }

        cov / denom
    }

    /// Compute partial correlation of x and y controlling for z
    fn compute_partial_correlation(x: &[f64], y: &[f64], z: &[f64]) -> f64 {
        let n = x.len().min(y.len()).min(z.len());
        if n < 10 {
            return 0.0;
        }

        // Regress x on z, get residuals
        let x_residuals = regress_residuals(&x[..n], &z[..n]);

        // Regress y on z, get residuals
        let y_residuals = regress_residuals(&y[..n], &z[..n]);

        // Correlation of residuals
        pearson_correlation(&x_residuals, &y_residuals)
    }

    /// Simple linear regression residuals
    fn regress_residuals(y: &[f64], x: &[f64]) -> Vec<f64> {
        let n = y.len().min(x.len());
        if n < 2 {
            return y.to_vec();
        }

        let mean_x: f64 = x[..n].iter().sum::<f64>() / n as f64;
        let mean_y: f64 = y[..n].iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
        }

        let slope = if var_x > 1e-15 { cov / var_x } else { 0.0 };
        let intercept = mean_y - slope * mean_x;

        // Compute residuals
        (0..n)
            .map(|i| y[i] - (intercept + slope * x[i]))
            .collect()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Feature Count Tests
    // ========================================================================

    #[test]
    fn test_feature_count() {
        assert_eq!(ToxicityFeatures::count(), 10);
        assert_eq!(ToxicityFeatures::names().len(), 10);
        assert_eq!(ToxicityFeatures::default().to_vec().len(), 10);
    }

    // ========================================================================
    // VPIN Tests
    // ========================================================================

    #[test]
    fn test_vpin_balanced_flow() {
        // Equal buy and sell volume should give low VPIN
        let data = TradeData {
            prices: vec![100.0; 100],
            volumes: vec![1.0; 100],
            directions: (0..100).map(|i| if i % 2 == 0 { 1i8 } else { -1i8 }).collect(),
            is_buy: (0..100).map(|i| i % 2 == 0).collect(),
        };

        let vpin = compute_vpin(&data, 10);
        assert!(vpin < 0.2, "Balanced flow should have low VPIN, got {}", vpin);
    }

    #[test]
    fn test_vpin_unbalanced_flow() {
        // All buy orders should give high VPIN
        let data = TradeData {
            prices: vec![100.0; 100],
            volumes: vec![1.0; 100],
            directions: vec![1i8; 100], // All buys
            is_buy: vec![true; 100],
        };

        let vpin = compute_vpin(&data, 10);
        assert!(vpin > 0.8, "All-buy flow should have high VPIN, got {}", vpin);
    }

    #[test]
    fn test_vpin_insufficient_data() {
        let data = TradeData {
            prices: vec![100.0; 5],
            volumes: vec![1.0; 5],
            directions: vec![1i8; 5],
            is_buy: vec![true; 5],
        };

        let vpin = compute_vpin(&data, 10);
        assert_eq!(vpin, 0.0, "Insufficient data should return 0");
    }

    // ========================================================================
    // Adverse Selection Tests
    // ========================================================================

    #[test]
    fn test_adverse_selection_informed_flow() {
        // Create simple data where ALL buys are followed by price increases
        let n = 100;
        let mut prices = Vec::with_capacity(n);
        let directions: Vec<i8> = vec![1i8; n]; // All buys

        // Prices steadily increase (buys correctly predict future increases)
        for i in 0..n {
            prices.push(100.0 + i as f64 * 0.5);
        }

        let data = TradeData {
            prices,
            volumes: vec![1.0; n],
            directions,
            is_buy: vec![true; n],
        };

        let adverse = compute_adverse_selection(&data);
        // With all buys and steadily increasing prices, future returns are positive
        // direction (+1) * positive_return = positive contribution
        assert!(adverse > 0.0, "All-buy with increasing prices should have positive adverse selection, got {}", adverse);
    }

    #[test]
    fn test_adverse_selection_random_flow() {
        // Random directions with trending price
        let n = 100;
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.01).collect();
        let directions: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1i8 } else { -1i8 }).collect();

        let data = TradeData {
            prices,
            volumes: vec![1.0; n],
            directions,
            is_buy: (0..n).map(|i| i % 2 == 0).collect(),
        };

        let adverse = compute_adverse_selection(&data);
        // Should be close to zero (no correlation)
        assert!(adverse.abs() < 50.0, "Random flow should have low adverse selection, got {}", adverse);
    }

    // ========================================================================
    // Flow Imbalance Tests
    // ========================================================================

    #[test]
    fn test_flow_imbalance_all_buys() {
        let data = TradeData {
            prices: vec![100.0; 50],
            volumes: vec![1.0; 50],
            directions: vec![1i8; 50],
            is_buy: vec![true; 50],
        };

        let (imbalance, imbalance_abs) = compute_flow_imbalance(&data);
        assert!((imbalance - 1.0).abs() < 1e-10, "All buys should give imbalance of 1");
        assert!((imbalance_abs - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_flow_imbalance_balanced() {
        let data = TradeData {
            prices: vec![100.0; 50],
            volumes: vec![1.0; 50],
            directions: (0..50).map(|i| if i % 2 == 0 { 1i8 } else { -1i8 }).collect(),
            is_buy: (0..50).map(|i| i % 2 == 0).collect(),
        };

        let (imbalance, _) = compute_flow_imbalance(&data);
        assert!(imbalance.abs() < 0.1, "Balanced flow should have low imbalance, got {}", imbalance);
    }

    // ========================================================================
    // Toxicity Index Tests
    // ========================================================================

    #[test]
    fn test_toxicity_index_high() {
        let index = compute_toxicity_index(0.9, 80.0, 0.8);
        assert!(index > 0.7, "High toxicity inputs should give high index, got {}", index);
    }

    #[test]
    fn test_toxicity_index_low() {
        let index = compute_toxicity_index(0.1, 5.0, 0.1);
        assert!(index < 0.3, "Low toxicity inputs should give low index, got {}", index);
    }

    #[test]
    fn test_toxicity_index_bounds() {
        // Test clamping
        let index = compute_toxicity_index(2.0, 500.0, 2.0);
        assert!(index <= 1.0, "Index should be clamped to 1.0");
        assert!(index >= 0.0, "Index should be non-negative");
    }

    // ========================================================================
    // Skeptical Tests
    // ========================================================================

    #[test]
    fn test_vpin_predictive_power() {
        use skeptical_tests::test_vpin_predictive_power;

        // Create data where high VPIN predicts large moves
        let n = 200;
        let vpin: Vec<f64> = (0..n).map(|i| if i % 4 == 0 { 0.8 } else { 0.2 }).collect();
        let future_moves: Vec<f64> = (0..n).map(|i| if i % 4 == 0 { 2.0 } else { 0.5 }).collect();

        let result = test_vpin_predictive_power(&vpin, &future_moves, 75.0);

        assert!(result.lift > 1.0, "High VPIN should predict larger moves, lift = {}", result.lift);
        assert!(result.high_vpin_move_magnitude > result.low_vpin_move_magnitude);
    }

    #[test]
    fn test_vpin_volatility_independence() {
        use skeptical_tests::test_vpin_volatility_independence;

        // Create data where VPIN adds info beyond current vol
        let n = 100;
        let vpin: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin().abs()).collect();
        let current_vol: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).cos().abs()).collect();
        let future_vol: Vec<f64> = (0..n).map(|i| {
            // Future vol depends on both VPIN and current vol
            (i as f64 * 0.1).sin().abs() * 0.5 + (i as f64 * 0.05).cos().abs() * 0.3 + 0.2
        }).collect();

        let result = test_vpin_volatility_independence(&vpin, &current_vol, &future_vol);

        assert!(result.sample_size >= 50);
        // Just verify it computes without panic
    }

    #[test]
    fn test_adverse_selection_whale_correlation() {
        use skeptical_tests::test_adverse_selection_whale_correlation;

        // Create data where high whale activity correlates with high adverse selection
        let n = 100;
        // About 25% high whale activity (so 75th percentile split works)
        let whale_activity: Vec<f64> = (0..n).map(|i| if i % 4 == 0 { 0.9 } else { 0.2 }).collect();
        // Higher adverse selection when whale activity is high
        let adverse_selection: Vec<f64> = (0..n).map(|i| {
            if i % 4 == 0 { 80.0 } else { 15.0 }
        }).collect();

        let result = test_adverse_selection_whale_correlation(&adverse_selection, &whale_activity, 75.0);

        // High whale periods should have higher adverse selection
        assert!(result.adverse_selection_high_whale > result.adverse_selection_low_whale,
            "High whale ({:.1}) should have more adverse selection than low whale ({:.1})",
            result.adverse_selection_high_whale, result.adverse_selection_low_whale);
    }
}
