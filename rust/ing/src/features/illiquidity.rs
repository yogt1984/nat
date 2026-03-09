//! Illiquidity Feature Extraction
//!
//! This module implements market impact and illiquidity measures that help
//! identify informed trading flow and market conditions.
//!
//! # Features
//!
//! | Feature | Description | Range | Interpretation |
//! |---------|-------------|-------|----------------|
//! | **Kyle's Lambda** | Price impact per unit volume | [0, +inf) | Higher = more illiquid |
//! | **Amihud's Lambda** | |return| / volume ratio | [0, +inf) | Higher = more illiquid |
//! | **Hasbrouck's Lambda** | Permanent price impact | [0, +inf) | Higher = more illiquid |
//!
//! # References
//!
//! - Kyle (1985) - Continuous auctions and insider trading
//! - Amihud (2002) - Illiquidity and stock returns
//! - Hasbrouck (2009) - Trading costs and returns for US equities

use crate::state::{TradeBuffer, Trade};

/// Minimum observations required for reliable computation
const MIN_OBSERVATIONS: usize = 20;
const MIN_OBSERVATIONS_KYLE: usize = 30;
const MIN_OBSERVATIONS_HASBROUCK: usize = 50;

/// Illiquidity features computed at multiple windows
/// Total: 12 features (4 base features × 3 windows: 100, 500, 1000 trades)
#[derive(Debug, Clone, Default)]
pub struct IlliquidityFeatures {
    // Window 100 trades (short-term)
    /// Kyle's lambda - 100 trade window
    pub kyle_lambda_100: f64,
    /// Amihud's lambda - 100 trade window
    pub amihud_lambda_100: f64,
    /// Hasbrouck's lambda - 100 trade window
    pub hasbrouck_lambda_100: f64,
    /// Roll spread estimate - 100 trade window
    pub roll_spread_100: f64,

    // Window 500 trades (medium-term)
    /// Kyle's lambda - 500 trade window
    pub kyle_lambda_500: f64,
    /// Amihud's lambda - 500 trade window
    pub amihud_lambda_500: f64,
    /// Hasbrouck's lambda - 500 trade window
    pub hasbrouck_lambda_500: f64,
    /// Roll spread estimate - 500 trade window
    pub roll_spread_500: f64,

    // Cross-window derived features
    /// Kyle lambda ratio (short/long) - detects illiquidity changes
    pub kyle_ratio: f64,
    /// Amihud lambda ratio (short/long)
    pub amihud_ratio: f64,
    /// Average illiquidity (mean of normalized lambdas)
    pub illiquidity_composite: f64,
    /// Trade count used in computation
    pub trade_count: usize,
}

impl IlliquidityFeatures {
    pub fn count() -> usize {
        12
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "illiq_kyle_100",
            "illiq_amihud_100",
            "illiq_hasbrouck_100",
            "illiq_roll_100",
            "illiq_kyle_500",
            "illiq_amihud_500",
            "illiq_hasbrouck_500",
            "illiq_roll_500",
            "illiq_kyle_ratio",
            "illiq_amihud_ratio",
            "illiq_composite",
            "illiq_trade_count",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.kyle_lambda_100,
            self.amihud_lambda_100,
            self.hasbrouck_lambda_100,
            self.roll_spread_100,
            self.kyle_lambda_500,
            self.amihud_lambda_500,
            self.hasbrouck_lambda_500,
            self.roll_spread_500,
            self.kyle_ratio,
            self.amihud_ratio,
            self.illiquidity_composite,
            self.trade_count as f64,
        ]
    }
}

/// Compute illiquidity features from trade buffer
pub fn compute(trade_buffer: &TradeBuffer) -> IlliquidityFeatures {
    // Get all trades (use full buffer - typically 60 seconds)
    let trades: Vec<_> = trade_buffer.iter().collect();
    let trade_count = trades.len();

    if trade_count < MIN_OBSERVATIONS {
        return IlliquidityFeatures {
            trade_count,
            ..Default::default()
        };
    }

    // Extract price, volume, and direction data
    let (prices, volumes, directions) = extract_trade_data(&trades);

    // Compute features at different windows
    let kyle_lambda_100 = compute_kyle_lambda(&prices, &volumes, &directions, 100);
    let amihud_lambda_100 = compute_amihud_lambda(&prices, &volumes, 100);
    let hasbrouck_lambda_100 = compute_hasbrouck_lambda(&prices, &volumes, &directions, 100);
    let roll_spread_100 = compute_roll_spread(&prices, 100);

    let kyle_lambda_500 = compute_kyle_lambda(&prices, &volumes, &directions, 500);
    let amihud_lambda_500 = compute_amihud_lambda(&prices, &volumes, 500);
    let hasbrouck_lambda_500 = compute_hasbrouck_lambda(&prices, &volumes, &directions, 500);
    let roll_spread_500 = compute_roll_spread(&prices, 500);

    // Compute ratios (short/long) - values > 1 indicate increasing illiquidity
    let kyle_ratio = if kyle_lambda_500 > 1e-10 {
        kyle_lambda_100 / kyle_lambda_500
    } else {
        1.0
    };

    let amihud_ratio = if amihud_lambda_500 > 1e-10 {
        amihud_lambda_100 / amihud_lambda_500
    } else {
        1.0
    };

    // Composite illiquidity score (normalized average)
    let illiquidity_composite = compute_composite(
        kyle_lambda_100,
        amihud_lambda_100,
        hasbrouck_lambda_100,
    );

    IlliquidityFeatures {
        kyle_lambda_100,
        amihud_lambda_100,
        hasbrouck_lambda_100,
        roll_spread_100,
        kyle_lambda_500,
        amihud_lambda_500,
        hasbrouck_lambda_500,
        roll_spread_500,
        kyle_ratio,
        amihud_ratio,
        illiquidity_composite,
        trade_count,
    }
}

/// Extract price, volume, and direction from trades
fn extract_trade_data(trades: &[&Trade]) -> (Vec<f64>, Vec<f64>, Vec<i8>) {
    let mut prices = Vec::with_capacity(trades.len());
    let mut volumes = Vec::with_capacity(trades.len());
    let mut directions = Vec::with_capacity(trades.len());
    let mut last_price: Option<f64> = None;

    for trade in trades {
        prices.push(trade.price);
        volumes.push(trade.size);

        // Compute direction based on price change or aggressor side
        let direction = match last_price {
            Some(prev) if trade.price > prev => 1i8,
            Some(prev) if trade.price < prev => -1i8,
            Some(_) => if trade.is_buy { 1i8 } else { -1i8 },
            None => if trade.is_buy { 1i8 } else { -1i8 },
        };
        directions.push(direction);
        last_price = Some(trade.price);
    }

    (prices, volumes, directions)
}

/// Compute Kyle's Lambda: regression of price change on signed volume
///
/// Lambda = Cov(ΔP, signed_volume) / Var(signed_volume)
///
/// Interpretation: Price impact per unit of signed volume
/// Higher values indicate lower liquidity (larger price impact)
fn compute_kyle_lambda(
    prices: &[f64],
    volumes: &[f64],
    directions: &[i8],
    window: usize,
) -> f64 {
    let n = prices.len().min(window);
    if n < MIN_OBSERVATIONS_KYLE {
        return 0.0;
    }

    let start = prices.len().saturating_sub(window);
    let prices = &prices[start..];
    let volumes = &volumes[start..];
    let directions = &directions[start..];

    // Compute price changes and signed volumes
    let mut price_changes = Vec::with_capacity(prices.len() - 1);
    let mut signed_volumes = Vec::with_capacity(volumes.len() - 1);

    for i in 1..prices.len() {
        let price_change = prices[i] - prices[i - 1];
        let signed_vol = volumes[i] * directions[i] as f64;
        price_changes.push(price_change);
        signed_volumes.push(signed_vol);
    }

    if price_changes.len() < MIN_OBSERVATIONS_KYLE {
        return 0.0;
    }

    // Compute means
    let mean_price_change: f64 = price_changes.iter().sum::<f64>() / price_changes.len() as f64;
    let mean_signed_vol: f64 = signed_volumes.iter().sum::<f64>() / signed_volumes.len() as f64;

    // Compute covariance and variance
    let mut covariance = 0.0;
    let mut vol_variance = 0.0;

    for (pc, sv) in price_changes.iter().zip(&signed_volumes) {
        let pc_diff = pc - mean_price_change;
        let sv_diff = sv - mean_signed_vol;
        covariance += pc_diff * sv_diff;
        vol_variance += sv_diff * sv_diff;
    }

    let n_minus_1 = (price_changes.len() - 1) as f64;
    covariance /= n_minus_1;
    vol_variance /= n_minus_1;

    if vol_variance.abs() < 1e-15 {
        return 0.0;
    }

    // Kyle's lambda = Cov / Var, take absolute value (should be positive)
    (covariance / vol_variance).abs()
}

/// Compute Amihud's Lambda: |return| / volume ratio
///
/// Lambda = Σ|r_i| / Σ|v_i|
///
/// Interpretation: Average absolute return per unit volume
/// Higher values indicate lower liquidity
fn compute_amihud_lambda(prices: &[f64], volumes: &[f64], window: usize) -> f64 {
    let n = prices.len().min(window);
    if n < MIN_OBSERVATIONS {
        return 0.0;
    }

    let start = prices.len().saturating_sub(window);
    let prices = &prices[start..];
    let volumes = &volumes[start..];

    let mut sum_abs_returns = 0.0;
    let mut sum_volume = 0.0;

    for i in 1..prices.len() {
        if prices[i - 1] > 0.0 {
            // Use simple return: (P_t - P_{t-1}) / P_{t-1}
            let abs_return = ((prices[i] - prices[i - 1]) / prices[i - 1]).abs();
            sum_abs_returns += abs_return;
        }
        sum_volume += volumes[i];
    }

    if sum_volume < 1e-10 {
        return 0.0;
    }

    // Scale by 1e6 for readability (returns are typically small)
    (sum_abs_returns / sum_volume) * 1e6
}

/// Compute Hasbrouck's Lambda: permanent price impact estimate
///
/// Uses VAR(1) model approach: r_t = c + λ*x_t + ε_t
/// where x_t is signed order flow
///
/// Interpretation: Permanent component of price impact
fn compute_hasbrouck_lambda(
    prices: &[f64],
    volumes: &[f64],
    directions: &[i8],
    window: usize,
) -> f64 {
    let n = prices.len().min(window);
    if n < MIN_OBSERVATIONS_HASBROUCK {
        return 0.0;
    }

    let start = prices.len().saturating_sub(window);
    let prices = &prices[start..];
    let volumes = &volumes[start..];
    let directions = &directions[start..];

    // Compute returns and signed order flow
    let mut returns = Vec::with_capacity(prices.len() - 1);
    let mut signed_flows = Vec::with_capacity(volumes.len() - 1);

    for i in 1..prices.len() {
        if prices[i - 1] > 0.0 {
            let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
            let signed_flow = volumes[i] * directions[i] as f64;
            returns.push(ret);
            signed_flows.push(signed_flow);
        }
    }

    if returns.len() < MIN_OBSERVATIONS {
        return 0.0;
    }

    // Simple regression of returns on signed flow
    let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let mean_flow: f64 = signed_flows.iter().sum::<f64>() / signed_flows.len() as f64;

    let mut cov_ret_flow = 0.0;
    let mut var_flow = 0.0;

    for (ret, flow) in returns.iter().zip(&signed_flows) {
        let ret_diff = ret - mean_ret;
        let flow_diff = flow - mean_flow;
        cov_ret_flow += ret_diff * flow_diff;
        var_flow += flow_diff * flow_diff;
    }

    if var_flow.abs() < 1e-15 {
        return 0.0;
    }

    // Hasbrouck lambda = Cov / Var, scaled for readability
    ((cov_ret_flow / var_flow) * 1e6).abs()
}

/// Compute Roll's spread estimate
///
/// Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
///
/// Based on autocorrelation of price changes
fn compute_roll_spread(prices: &[f64], window: usize) -> f64 {
    let n = prices.len().min(window);
    if n < MIN_OBSERVATIONS {
        return 0.0;
    }

    let start = prices.len().saturating_sub(window);
    let prices = &prices[start..];

    // Compute price changes
    let price_changes: Vec<f64> = prices
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();

    if price_changes.len() < 2 {
        return 0.0;
    }

    // Compute autocovariance at lag 1
    let n_changes = price_changes.len();
    let mut sum_product = 0.0;

    for i in 1..n_changes {
        sum_product += price_changes[i] * price_changes[i - 1];
    }

    let autocovariance = sum_product / (n_changes - 1) as f64;

    // Roll spread = 2 * sqrt(-cov) if cov < 0, else 0
    if autocovariance < 0.0 {
        2.0 * (-autocovariance).sqrt()
    } else {
        0.0
    }
}

/// Compute composite illiquidity score
fn compute_composite(kyle: f64, amihud: f64, hasbrouck: f64) -> f64 {
    // Simple average of the three measures
    // In production, could use z-scores or percentile ranks
    let count = 3.0;
    let sum = kyle + amihud + hasbrouck;
    sum / count
}

// ============================================================================
// Skeptical Tests Module
// ============================================================================

/// Module for skeptical statistical tests on illiquidity features
pub mod skeptical_tests {
    //! Skeptical tests to validate illiquidity feature effectiveness
    //!
    //! These tests verify that:
    //! 1. High illiquidity predicts future volatility
    //! 2. Illiquidity + trend indicates informed flow
    //! 3. The three lambda measures aren't redundant (r < 0.9)

    /// Result of correlation analysis between illiquidity and volatility
    #[derive(Debug, Clone)]
    pub struct IlliquidityVolatilityTest {
        pub correlation: f64,
        pub p_value: f64,
        pub sample_size: usize,
        pub significant: bool,
    }

    /// Result of redundancy analysis between lambda measures
    #[derive(Debug, Clone)]
    pub struct RedundancyTest {
        pub kyle_amihud_corr: f64,
        pub kyle_hasbrouck_corr: f64,
        pub amihud_hasbrouck_corr: f64,
        pub max_correlation: f64,
        pub is_redundant: bool, // true if any correlation > 0.9
    }

    /// Test if high illiquidity predicts future volatility
    pub fn test_illiquidity_volatility_prediction(
        illiquidity_values: &[f64],
        future_volatility: &[f64],
    ) -> IlliquidityVolatilityTest {
        if illiquidity_values.len() != future_volatility.len() || illiquidity_values.len() < 30 {
            return IlliquidityVolatilityTest {
                correlation: 0.0,
                p_value: 1.0,
                sample_size: illiquidity_values.len(),
                significant: false,
            };
        }

        let correlation = pearson_correlation(illiquidity_values, future_volatility);
        let n = illiquidity_values.len();

        // t-statistic for correlation significance
        let t_stat = if (1.0 - correlation * correlation).abs() > 1e-10 {
            correlation * ((n - 2) as f64).sqrt() / (1.0 - correlation * correlation).sqrt()
        } else {
            0.0
        };

        // Approximate p-value (two-tailed)
        let p_value = approximate_t_pvalue(t_stat, n - 2);

        IlliquidityVolatilityTest {
            correlation,
            p_value,
            sample_size: n,
            significant: p_value < 0.01 && correlation > 0.0,
        }
    }

    /// Test redundancy between the three lambda measures
    pub fn test_lambda_redundancy(
        kyle_values: &[f64],
        amihud_values: &[f64],
        hasbrouck_values: &[f64],
    ) -> RedundancyTest {
        let n = kyle_values.len().min(amihud_values.len()).min(hasbrouck_values.len());

        if n < 30 {
            return RedundancyTest {
                kyle_amihud_corr: 0.0,
                kyle_hasbrouck_corr: 0.0,
                amihud_hasbrouck_corr: 0.0,
                max_correlation: 0.0,
                is_redundant: false,
            };
        }

        let kyle_amihud_corr = pearson_correlation(&kyle_values[..n], &amihud_values[..n]).abs();
        let kyle_hasbrouck_corr = pearson_correlation(&kyle_values[..n], &hasbrouck_values[..n]).abs();
        let amihud_hasbrouck_corr = pearson_correlation(&amihud_values[..n], &hasbrouck_values[..n]).abs();

        let max_correlation = kyle_amihud_corr
            .max(kyle_hasbrouck_corr)
            .max(amihud_hasbrouck_corr);

        RedundancyTest {
            kyle_amihud_corr,
            kyle_hasbrouck_corr,
            amihud_hasbrouck_corr,
            max_correlation,
            is_redundant: max_correlation > 0.9,
        }
    }

    /// Test if illiquidity + trend indicates informed flow
    /// Returns conditional probability lift
    pub fn test_informed_flow_detection(
        illiquidity_values: &[f64],
        trend_values: &[f64], // positive = uptrend
        future_returns: &[f64],
        illiquidity_threshold_percentile: f64,
    ) -> InformedFlowTest {
        let n = illiquidity_values.len()
            .min(trend_values.len())
            .min(future_returns.len());

        if n < 100 {
            return InformedFlowTest {
                p_continue_high_illiq_trend: 0.5,
                p_continue_low_illiq_trend: 0.5,
                lift: 1.0,
                sample_size_high: 0,
                sample_size_low: 0,
                significant: false,
            };
        }

        // Compute illiquidity threshold
        let mut sorted_illiq: Vec<f64> = illiquidity_values[..n].to_vec();
        sorted_illiq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = ((n as f64) * illiquidity_threshold_percentile / 100.0) as usize;
        let illiq_threshold = sorted_illiq.get(threshold_idx).copied().unwrap_or(0.0);

        // Count continuation events
        let mut high_illiq_continue = 0;
        let mut high_illiq_total = 0;
        let mut low_illiq_continue = 0;
        let mut low_illiq_total = 0;

        for i in 0..n {
            let has_trend = trend_values[i].abs() > 0.0;
            let trend_direction = trend_values[i].signum();
            let return_direction = future_returns[i].signum();
            let continues = trend_direction == return_direction;

            if has_trend {
                if illiquidity_values[i] >= illiq_threshold {
                    high_illiq_total += 1;
                    if continues {
                        high_illiq_continue += 1;
                    }
                } else {
                    low_illiq_total += 1;
                    if continues {
                        low_illiq_continue += 1;
                    }
                }
            }
        }

        let p_high = if high_illiq_total > 0 {
            high_illiq_continue as f64 / high_illiq_total as f64
        } else {
            0.5
        };

        let p_low = if low_illiq_total > 0 {
            low_illiq_continue as f64 / low_illiq_total as f64
        } else {
            0.5
        };

        let lift = if p_low > 0.0 { p_high / p_low } else { 1.0 };

        InformedFlowTest {
            p_continue_high_illiq_trend: p_high,
            p_continue_low_illiq_trend: p_low,
            lift,
            sample_size_high: high_illiq_total,
            sample_size_low: low_illiq_total,
            significant: lift > 1.1 && high_illiq_total >= 30 && low_illiq_total >= 30,
        }
    }

    /// Result of informed flow detection test
    #[derive(Debug, Clone)]
    pub struct InformedFlowTest {
        pub p_continue_high_illiq_trend: f64,
        pub p_continue_low_illiq_trend: f64,
        pub lift: f64, // p_high / p_low
        pub sample_size_high: usize,
        pub sample_size_low: usize,
        pub significant: bool,
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

    /// Approximate p-value for t-distribution (two-tailed)
    fn approximate_t_pvalue(t: f64, df: usize) -> f64 {
        // Using normal approximation for large df
        if df > 30 {
            let z = t.abs();
            // Approximate using error function
            let p = 1.0 - 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
            return 2.0 * p; // Two-tailed
        }

        // For smaller df, use a rough approximation
        let x = df as f64 / (df as f64 + t * t);
        let p = 0.5 * incomplete_beta(df as f64 / 2.0, 0.5, x);
        2.0 * p.min(1.0 - p)
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        // Horner form approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Incomplete beta function approximation (very rough)
    fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }

        // Simple numerical integration
        let n_steps = 100;
        let dx = x / n_steps as f64;
        let mut sum = 0.0;

        for i in 0..n_steps {
            let t = (i as f64 + 0.5) * dx;
            sum += t.powf(a - 1.0) * (1.0 - t).powf(b - 1.0);
        }

        sum * dx / beta(a, b)
    }

    /// Beta function
    fn beta(a: f64, b: f64) -> f64 {
        (gamma(a) * gamma(b)) / gamma(a + b)
    }

    /// Gamma function approximation (Stirling)
    fn gamma(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }
        if x < 0.5 {
            return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x));
        }

        let x = x - 1.0;
        let g = 7.0;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        let mut sum = c[0];
        for i in 1..9 {
            sum += c[i] / (x + i as f64);
        }

        let t = x + g + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{TradeBuffer, Trade};

    fn create_test_buffer_with_trades(trades: Vec<Trade>) -> TradeBuffer {
        let mut buffer = TradeBuffer::new(300); // 5 minute window
        // Manually insert trades (bypassing add() since we have Trade directly)
        for trade in trades {
            // We need to access the internal trades field
            // Since we can't, we'll create a different approach
        }
        buffer
    }

    // ========================================================================
    // Feature Count Tests
    // ========================================================================

    #[test]
    fn test_feature_count() {
        assert_eq!(IlliquidityFeatures::count(), 12);
        assert_eq!(IlliquidityFeatures::names().len(), 12);
        assert_eq!(IlliquidityFeatures::default().to_vec().len(), 12);
    }

    // ========================================================================
    // Kyle's Lambda Tests
    // ========================================================================

    #[test]
    fn test_kyle_lambda_insufficient_data() {
        let prices = vec![100.0; 10];
        let volumes = vec![1.0; 10];
        let directions = vec![1i8; 10];

        let lambda = compute_kyle_lambda(&prices, &volumes, &directions, 100);
        assert_eq!(lambda, 0.0, "Should return 0 for insufficient data");
    }

    #[test]
    fn test_kyle_lambda_positive_impact() {
        // Create data where buying (positive direction) increases price
        let n = 100;
        let mut prices = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        let mut directions = Vec::with_capacity(n);

        let mut price = 100.0;
        for i in 0..n {
            let dir = if i % 2 == 0 { 1i8 } else { -1i8 };
            let vol = 1.0;
            // Price moves in direction of trade
            price += dir as f64 * 0.1;

            prices.push(price);
            volumes.push(vol);
            directions.push(dir);
        }

        let lambda = compute_kyle_lambda(&prices, &volumes, &directions, 100);
        assert!(lambda > 0.0, "Kyle's lambda should be positive, got {}", lambda);
    }

    #[test]
    fn test_kyle_lambda_no_impact() {
        // Random price with no correlation to volume direction
        let n = 100;
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.01).sin()).collect();
        let volumes: Vec<f64> = vec![1.0; n];
        let directions: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1i8 } else { -1i8 }).collect();

        let lambda = compute_kyle_lambda(&prices, &volumes, &directions, 100);
        // Should be small but not necessarily zero
        assert!(lambda < 1.0, "Kyle's lambda should be small for random data, got {}", lambda);
    }

    // ========================================================================
    // Amihud's Lambda Tests
    // ========================================================================

    #[test]
    fn test_amihud_lambda_insufficient_data() {
        let prices = vec![100.0; 5];
        let volumes = vec![1.0; 5];

        let lambda = compute_amihud_lambda(&prices, &volumes, 100);
        assert_eq!(lambda, 0.0, "Should return 0 for insufficient data");
    }

    #[test]
    fn test_amihud_lambda_high_impact() {
        // Large price changes relative to volume = high illiquidity
        let n = 50;
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect(); // 0.5% per trade
        let volumes: Vec<f64> = vec![0.1; n]; // Small volume

        let lambda = compute_amihud_lambda(&prices, &volumes, 100);
        assert!(lambda > 0.0, "Amihud should be positive for high impact");
    }

    #[test]
    fn test_amihud_lambda_low_impact() {
        // Small price changes relative to volume = low illiquidity
        let n = 50;
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.001).collect(); // Tiny moves
        let volumes: Vec<f64> = vec![100.0; n]; // Large volume

        let lambda = compute_amihud_lambda(&prices, &volumes, 100);
        // Should be smaller than high impact case
        assert!(lambda >= 0.0, "Amihud should be non-negative");
    }

    // ========================================================================
    // Hasbrouck's Lambda Tests
    // ========================================================================

    #[test]
    fn test_hasbrouck_lambda_insufficient_data() {
        let prices = vec![100.0; 30];
        let volumes = vec![1.0; 30];
        let directions = vec![1i8; 30];

        let lambda = compute_hasbrouck_lambda(&prices, &volumes, &directions, 100);
        assert_eq!(lambda, 0.0, "Should return 0 for insufficient data");
    }

    #[test]
    fn test_hasbrouck_lambda_permanent_impact() {
        // Create data with permanent price impact from signed flow
        let n = 100;
        let mut prices = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        let mut directions = Vec::with_capacity(n);

        let mut price = 100.0;
        for i in 0..n {
            let dir = if i % 3 == 0 { -1i8 } else { 1i8 };
            let vol = 1.0 + (i % 5) as f64 * 0.1;
            // Permanent impact: price changes with signed flow
            price += dir as f64 * vol * 0.01;

            prices.push(price);
            volumes.push(vol);
            directions.push(dir);
        }

        let lambda = compute_hasbrouck_lambda(&prices, &volumes, &directions, 100);
        assert!(lambda > 0.0, "Hasbrouck should detect permanent impact, got {}", lambda);
    }

    // ========================================================================
    // Roll Spread Tests
    // ========================================================================

    #[test]
    fn test_roll_spread_bid_ask_bounce() {
        // Simulate bid-ask bounce: price alternates
        let n = 100;
        let prices: Vec<f64> = (0..n).map(|i| {
            if i % 2 == 0 { 100.0 } else { 100.1 } // Bounce between bid and ask
        }).collect();

        let spread = compute_roll_spread(&prices, 100);
        assert!(spread > 0.0, "Roll spread should detect bid-ask bounce, got {}", spread);
    }

    #[test]
    fn test_roll_spread_trending() {
        // Trending market: positive autocorrelation, no bid-ask bounce
        let n = 100;
        let prices: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.1).collect();

        let spread = compute_roll_spread(&prices, 100);
        // Trending should have low or zero Roll spread
        assert!(spread >= 0.0, "Roll spread should be non-negative");
    }

    // ========================================================================
    // Composite Tests
    // ========================================================================

    #[test]
    fn test_composite_calculation() {
        let composite = compute_composite(1.0, 2.0, 3.0);
        assert!((composite - 2.0).abs() < 1e-10, "Composite should be average");
    }

    // ========================================================================
    // Skeptical Tests
    // ========================================================================

    #[test]
    fn test_redundancy_detection() {
        use skeptical_tests::test_lambda_redundancy;

        // Test with highly correlated data
        let n = 100;
        let kyle: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let amihud: Vec<f64> = (0..n).map(|i| i as f64 * 1.1 + 0.5).collect(); // Highly correlated
        let hasbrouck: Vec<f64> = (0..n).map(|i| (i as f64).sin() * 10.0 + 50.0).collect(); // Less correlated

        let result = test_lambda_redundancy(&kyle, &amihud, &hasbrouck);

        assert!(result.kyle_amihud_corr > 0.9, "Kyle-Amihud should be highly correlated");
        assert!(result.is_redundant, "Should detect redundancy");
    }

    #[test]
    fn test_no_redundancy() {
        use skeptical_tests::test_lambda_redundancy;

        // Test with uncorrelated data
        let n = 100;
        let kyle: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() * 10.0).collect();
        let amihud: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).cos() * 10.0).collect();
        let hasbrouck: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin() * 10.0 + 5.0).collect();

        let result = test_lambda_redundancy(&kyle, &amihud, &hasbrouck);

        assert!(result.max_correlation < 0.9, "Should not detect redundancy for uncorrelated data");
        assert!(!result.is_redundant, "Should not be redundant");
    }

    #[test]
    fn test_volatility_prediction() {
        use skeptical_tests::test_illiquidity_volatility_prediction;

        // Create data where high illiquidity predicts high volatility
        let n = 100;
        let illiquidity: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let volatility: Vec<f64> = (0..n).map(|i| (i % 10) as f64 * 0.5 + 0.1).collect(); // Correlated

        let result = test_illiquidity_volatility_prediction(&illiquidity, &volatility);

        assert!(result.correlation > 0.5, "Should detect correlation, got {}", result.correlation);
    }

    #[test]
    fn test_informed_flow() {
        use skeptical_tests::test_informed_flow_detection;

        // Create synthetic data where high illiquidity + trend predicts continuation
        let n = 200;
        let mut illiquidity = Vec::with_capacity(n);
        let mut trend = Vec::with_capacity(n);
        let mut future_returns = Vec::with_capacity(n);

        for i in 0..n {
            // High illiquidity every 4th sample
            let illiq = if i % 4 == 0 { 10.0 } else { 1.0 };
            // Consistent positive trend
            let t = 1.0;
            // High illiquidity: 90% continuation rate
            // Low illiquidity: 40% continuation rate
            let ret = if illiq > 5.0 {
                if i % 10 == 0 { -1.0 } else { 1.0 }  // 90% continues
            } else {
                if i % 5 < 2 { 1.0 } else { -1.0 }  // 40% continues
            };

            illiquidity.push(illiq);
            trend.push(t);
            future_returns.push(ret);
        }

        let result = test_informed_flow_detection(&illiquidity, &trend, &future_returns, 75.0);

        // With our synthetic data, high illiquidity should show higher continuation
        assert!(result.p_continue_high_illiq_trend > result.p_continue_low_illiq_trend,
            "High illiq continuation ({:.2}) should be > low illiq ({:.2})",
            result.p_continue_high_illiq_trend, result.p_continue_low_illiq_trend);
        assert!(result.lift > 1.0, "Lift should be > 1, got {:.2}", result.lift);
    }
}
