//! Position Concentration Feature Extraction
//!
//! This module computes concentration metrics that measure how positions
//! are distributed among market participants - a key Hyperliquid-unique feature.
//!
//! # Features
//!
//! | Feature | Formula | Interpretation |
//! |---------|---------|----------------|
//! | **top10_concentration** | Σ(top10) / total_OI | % held by top 10 |
//! | **top20_concentration** | Σ(top20) / total_OI | % held by top 20 |
//! | **herfindahl_index** | Σ(share²) | Market concentration |
//! | **gini_coefficient** | Inequality measure | 0=equal, 1=monopoly |
//!
//! # Theory
//!
//! Position concentration reveals market structure:
//! - **High concentration**: Few players dominate, more predictable moves
//! - **Rising concentration**: Whales accumulating, potential breakout
//! - **Falling concentration**: Distribution phase, potential reversal
//!
//! On Hyperliquid, we can see all positions, enabling real-time
//! concentration analysis that's impossible on CEXs.

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// Configuration for concentration computation
#[derive(Debug, Clone)]
pub struct ConcentrationConfig {
    /// Number of top positions to track
    pub top_n_positions: Vec<usize>,
    /// History size for change calculation
    pub history_size: usize,
    /// Minimum position value to include (USD)
    pub min_position_value: f64,
}

impl Default for ConcentrationConfig {
    fn default() -> Self {
        Self {
            top_n_positions: vec![5, 10, 20, 50],
            history_size: 60, // 1 hour at 1-minute updates
            min_position_value: 1000.0,
        }
    }
}

/// Position concentration features
/// Total: 15 features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcentrationFeatures {
    // === Top-N Concentration ===
    /// Fraction of OI held by top 5 positions
    pub top5_concentration: f64,
    /// Fraction of OI held by top 10 positions
    pub top10_concentration: f64,
    /// Fraction of OI held by top 20 positions
    pub top20_concentration: f64,
    /// Fraction of OI held by top 50 positions
    pub top50_concentration: f64,

    // === Inequality Metrics ===
    /// Herfindahl-Hirschman Index (sum of squared shares)
    /// 0 = perfectly distributed, 1 = single holder
    pub herfindahl_index: f64,
    /// Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    pub gini_coefficient: f64,
    /// Theil index (entropy-based inequality)
    pub theil_index: f64,

    // === Whale vs Retail ===
    /// Ratio of whale OI to retail OI
    pub whale_retail_ratio: f64,
    /// Fraction of positions that are whales (by count)
    pub whale_fraction: f64,
    /// Average position size of whales vs retail
    pub whale_avg_size_ratio: f64,

    // === Concentration Dynamics ===
    /// Change in top10 concentration over 1h
    pub concentration_change_1h: f64,
    /// Rate of change of HHI
    pub hhi_roc: f64,
    /// Is concentration increasing? (1 = yes, 0 = stable, -1 = decreasing)
    pub concentration_trend: f64,

    // === Position Count Metrics ===
    /// Total number of positions
    pub position_count: f64,
    /// Number of whale-sized positions
    pub whale_position_count: f64,
}

impl Default for ConcentrationFeatures {
    fn default() -> Self {
        Self {
            top5_concentration: 0.0,
            top10_concentration: 0.0,
            top20_concentration: 0.0,
            top50_concentration: 0.0,
            herfindahl_index: 0.0,
            gini_coefficient: 0.0,
            theil_index: 0.0,
            whale_retail_ratio: 1.0,
            whale_fraction: 0.0,
            whale_avg_size_ratio: 1.0,
            concentration_change_1h: 0.0,
            hhi_roc: 0.0,
            concentration_trend: 0.0,
            position_count: 0.0,
            whale_position_count: 0.0,
        }
    }
}

impl ConcentrationFeatures {
    pub fn count() -> usize {
        15
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "top5_concentration",
            "top10_concentration",
            "top20_concentration",
            "top50_concentration",
            "herfindahl_index",
            "gini_coefficient",
            "theil_index",
            "whale_retail_ratio",
            "whale_fraction",
            "whale_avg_size_ratio",
            "concentration_change_1h",
            "hhi_roc",
            "concentration_trend",
            "position_count",
            "whale_position_count",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.top5_concentration,
            self.top10_concentration,
            self.top20_concentration,
            self.top50_concentration,
            self.herfindahl_index,
            self.gini_coefficient,
            self.theil_index,
            self.whale_retail_ratio,
            self.whale_fraction,
            self.whale_avg_size_ratio,
            self.concentration_change_1h,
            self.hhi_roc,
            self.concentration_trend,
            self.position_count,
            self.whale_position_count,
        ]
    }
}

/// A single position for concentration analysis
#[derive(Debug, Clone)]
pub struct Position {
    /// Position value in USD (absolute)
    pub value_usd: f64,
    /// Whether this is classified as a whale
    pub is_whale: bool,
}

/// Buffer for tracking concentration over time
#[derive(Debug)]
pub struct ConcentrationBuffer {
    config: ConcentrationConfig,
    /// Historical top10 concentration values
    top10_history: VecDeque<f64>,
    /// Historical HHI values
    hhi_history: VecDeque<f64>,
    /// Previous HHI for ROC calculation
    prev_hhi: f64,
}

impl ConcentrationBuffer {
    /// Create a new concentration buffer
    pub fn new(config: ConcentrationConfig) -> Self {
        Self {
            config,
            top10_history: VecDeque::with_capacity(100),
            hhi_history: VecDeque::with_capacity(100),
            prev_hhi: 0.0,
        }
    }

    /// Compute concentration features from current positions
    pub fn compute(&mut self, positions: &[Position], total_oi: f64) -> ConcentrationFeatures {
        if positions.is_empty() || total_oi <= 0.0 {
            return ConcentrationFeatures::default();
        }

        // Filter and sort positions by value (descending)
        let mut valid_positions: Vec<f64> = positions
            .iter()
            .filter(|p| p.value_usd >= self.config.min_position_value)
            .map(|p| p.value_usd)
            .collect();

        if valid_positions.is_empty() {
            return ConcentrationFeatures::default();
        }

        valid_positions.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let n = valid_positions.len();

        // Top-N concentration
        let top5 = compute_top_n_concentration(&valid_positions, 5, total_oi);
        let top10 = compute_top_n_concentration(&valid_positions, 10, total_oi);
        let top20 = compute_top_n_concentration(&valid_positions, 20, total_oi);
        let top50 = compute_top_n_concentration(&valid_positions, 50, total_oi);

        // Inequality metrics
        let hhi = compute_herfindahl(&valid_positions, total_oi);
        let gini = compute_gini(&valid_positions);
        let theil = compute_theil(&valid_positions);

        // Whale vs retail analysis
        let whale_positions: Vec<f64> = positions
            .iter()
            .filter(|p| p.is_whale && p.value_usd >= self.config.min_position_value)
            .map(|p| p.value_usd)
            .collect();

        let retail_positions: Vec<f64> = positions
            .iter()
            .filter(|p| !p.is_whale && p.value_usd >= self.config.min_position_value)
            .map(|p| p.value_usd)
            .collect();

        let whale_oi: f64 = whale_positions.iter().sum();
        let retail_oi: f64 = retail_positions.iter().sum();

        let whale_retail_ratio = if retail_oi > 1e-6 {
            whale_oi / retail_oi
        } else if whale_oi > 1e-6 {
            10.0 // Cap when no retail
        } else {
            1.0
        };

        let whale_fraction = if n > 0 {
            whale_positions.len() as f64 / n as f64
        } else {
            0.0
        };

        let whale_avg = if !whale_positions.is_empty() {
            whale_oi / whale_positions.len() as f64
        } else {
            0.0
        };

        let retail_avg = if !retail_positions.is_empty() {
            retail_oi / retail_positions.len() as f64
        } else {
            1.0 // Avoid div by zero
        };

        let whale_avg_size_ratio = if retail_avg > 1e-6 {
            whale_avg / retail_avg
        } else {
            1.0
        };

        // Update history
        self.top10_history.push_back(top10);
        while self.top10_history.len() > self.config.history_size {
            self.top10_history.pop_front();
        }

        self.hhi_history.push_back(hhi);
        while self.hhi_history.len() > self.config.history_size {
            self.hhi_history.pop_front();
        }

        // Concentration dynamics
        let concentration_change_1h = if self.top10_history.len() >= 2 {
            let oldest = self.top10_history.front().copied().unwrap_or(top10);
            top10 - oldest
        } else {
            0.0
        };

        let hhi_roc = hhi - self.prev_hhi;
        self.prev_hhi = hhi;

        // Trend: smoothed direction of concentration change
        let concentration_trend = if self.top10_history.len() >= 10 {
            let recent: Vec<f64> = self.top10_history.iter().rev().take(10).copied().collect();
            compute_trend_direction(&recent)
        } else {
            0.0
        };

        ConcentrationFeatures {
            top5_concentration: top5,
            top10_concentration: top10,
            top20_concentration: top20,
            top50_concentration: top50,
            herfindahl_index: hhi,
            gini_coefficient: gini,
            theil_index: theil,
            whale_retail_ratio,
            whale_fraction,
            whale_avg_size_ratio,
            concentration_change_1h,
            hhi_roc,
            concentration_trend,
            position_count: n as f64,
            whale_position_count: whale_positions.len() as f64,
        }
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.top10_history.clear();
        self.hhi_history.clear();
        self.prev_hhi = 0.0;
    }
}

/// Compute top-N concentration ratio
fn compute_top_n_concentration(sorted_positions: &[f64], n: usize, total_oi: f64) -> f64 {
    if total_oi <= 0.0 {
        return 0.0;
    }

    let top_n_sum: f64 = sorted_positions.iter().take(n).sum();
    (top_n_sum / total_oi).min(1.0)
}

/// Compute Herfindahl-Hirschman Index
fn compute_herfindahl(positions: &[f64], total_oi: f64) -> f64 {
    if total_oi <= 0.0 || positions.is_empty() {
        return 0.0;
    }

    positions
        .iter()
        .map(|&p| {
            let share = p / total_oi;
            share * share
        })
        .sum()
}

/// Compute Gini coefficient
fn compute_gini(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    if mean <= 0.0 {
        return 0.0;
    }

    // Sort values ascending for Gini calculation
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Gini = (2 * Σ(i * x_i) - (n+1) * Σx_i) / (n * Σx_i)
    let sum_x: f64 = sorted.iter().sum();
    if sum_x <= 0.0 {
        return 0.0;
    }

    let weighted_sum: f64 = sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| (i + 1) as f64 * x)
        .sum();

    let gini = (2.0 * weighted_sum - (n + 1) as f64 * sum_x) / (n as f64 * sum_x);
    gini.max(0.0).min(1.0)
}

/// Compute Theil index (generalized entropy)
fn compute_theil(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    if mean <= 0.0 {
        return 0.0;
    }

    // Theil T = (1/n) * Σ(x_i/μ * ln(x_i/μ))
    let theil: f64 = values
        .iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| {
            let ratio = x / mean;
            ratio * ratio.ln()
        })
        .sum::<f64>()
        / n as f64;

    theil.max(0.0)
}

/// Compute trend direction from recent values
/// Returns: 1 = increasing, -1 = decreasing, 0 = stable
fn compute_trend_direction(recent_values: &[f64]) -> f64 {
    if recent_values.len() < 2 {
        return 0.0;
    }

    // Simple linear regression slope
    let n = recent_values.len() as f64;
    let sum_x: f64 = (0..recent_values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = recent_values.iter().sum();
    let sum_xy: f64 = recent_values
        .iter()
        .enumerate()
        .map(|(i, &y)| i as f64 * y)
        .sum();
    let sum_x2: f64 = (0..recent_values.len()).map(|i| (i * i) as f64).sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return 0.0;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;

    // Normalize to -1 to 1 range based on typical concentration changes
    (slope * 100.0).tanh()
}

/// Standalone compute function for simple usage
pub fn compute(positions: &[Position], total_oi: f64) -> ConcentrationFeatures {
    let config = ConcentrationConfig::default();
    let mut buffer = ConcentrationBuffer::new(config);
    buffer.compute(positions, total_oi)
}

// ============================================================================
// Skeptical Tests Module
// ============================================================================

pub mod skeptical_tests {
    //! Skeptical tests to validate concentration feature effectiveness
    //!
    //! These tests verify:
    //! 1. Concentration predicts volatility
    //! 2. Concentration change predicts direction
    //! 3. High concentration is informative

    /// Result of concentration-volatility test
    #[derive(Debug, Clone)]
    pub struct ConcentrationVolatilityTest {
        /// Correlation between concentration and future volatility
        pub correlation: f64,
        /// Volatility in high concentration regime
        pub vol_high_concentration: f64,
        /// Volatility in low concentration regime
        pub vol_low_concentration: f64,
        /// Volatility ratio (high/low)
        pub volatility_ratio: f64,
        pub sample_size: usize,
        pub significant: bool,
    }

    /// Result of concentration change direction test
    #[derive(Debug, Clone)]
    pub struct ConcentrationDirectionTest {
        /// P(up | concentration increasing)
        pub prob_up_increasing: f64,
        /// P(up | concentration decreasing)
        pub prob_up_decreasing: f64,
        /// P(up | concentration stable)
        pub prob_up_stable: f64,
        /// Predictive lift
        pub lift: f64,
        pub sample_size: usize,
        pub significant: bool,
    }

    /// Result of whale accumulation test
    #[derive(Debug, Clone)]
    pub struct WhaleAccumulationTest {
        /// Correlation between whale_retail_ratio change and future returns
        pub correlation: f64,
        /// Average return when whales accumulating
        pub return_whale_accumulating: f64,
        /// Average return when whales distributing
        pub return_whale_distributing: f64,
        pub sample_size: usize,
        pub significant: bool,
    }

    /// Test if concentration predicts volatility
    pub fn test_concentration_volatility(
        concentration: &[f64],
        realized_volatility: &[f64],
        threshold: f64, // Split threshold (e.g., median)
    ) -> ConcentrationVolatilityTest {
        let n = concentration.len().min(realized_volatility.len());

        if n < 50 {
            return ConcentrationVolatilityTest {
                correlation: 0.0,
                vol_high_concentration: 0.0,
                vol_low_concentration: 0.0,
                volatility_ratio: 1.0,
                sample_size: n,
                significant: false,
            };
        }

        // Split by concentration level
        let mut vol_high = Vec::new();
        let mut vol_low = Vec::new();

        for i in 0..n {
            if concentration[i] >= threshold {
                vol_high.push(realized_volatility[i]);
            } else {
                vol_low.push(realized_volatility[i]);
            }
        }

        let mean_high = if !vol_high.is_empty() {
            vol_high.iter().sum::<f64>() / vol_high.len() as f64
        } else {
            0.0
        };

        let mean_low = if !vol_low.is_empty() {
            vol_low.iter().sum::<f64>() / vol_low.len() as f64
        } else {
            0.0
        };

        let ratio = if mean_low > 1e-10 {
            mean_high / mean_low
        } else {
            1.0
        };

        // Correlation
        let correlation = pearson_correlation(&concentration[..n], &realized_volatility[..n]);

        // Significant if clear difference
        let significant = (ratio - 1.0).abs() > 0.2
            && vol_high.len() >= 20
            && vol_low.len() >= 20;

        ConcentrationVolatilityTest {
            correlation,
            vol_high_concentration: mean_high,
            vol_low_concentration: mean_low,
            volatility_ratio: ratio,
            sample_size: n,
            significant,
        }
    }

    /// Test if concentration change predicts direction
    pub fn test_concentration_direction(
        concentration_change: &[f64],
        future_returns: &[f64],
        change_threshold: f64,
    ) -> ConcentrationDirectionTest {
        let n = concentration_change.len().min(future_returns.len());

        if n < 50 {
            return ConcentrationDirectionTest {
                prob_up_increasing: 0.5,
                prob_up_decreasing: 0.5,
                prob_up_stable: 0.5,
                lift: 1.0,
                sample_size: n,
                significant: false,
            };
        }

        let mut increasing_up = 0;
        let mut increasing_total = 0;
        let mut decreasing_up = 0;
        let mut decreasing_total = 0;
        let mut stable_up = 0;
        let mut stable_total = 0;

        for i in 0..n {
            let change = concentration_change[i];
            let ret = future_returns[i];

            if change > change_threshold {
                increasing_total += 1;
                if ret > 0.0 {
                    increasing_up += 1;
                }
            } else if change < -change_threshold {
                decreasing_total += 1;
                if ret > 0.0 {
                    decreasing_up += 1;
                }
            } else {
                stable_total += 1;
                if ret > 0.0 {
                    stable_up += 1;
                }
            }
        }

        let prob_increasing = if increasing_total > 0 {
            increasing_up as f64 / increasing_total as f64
        } else {
            0.5
        };

        let prob_decreasing = if decreasing_total > 0 {
            decreasing_up as f64 / decreasing_total as f64
        } else {
            0.5
        };

        let prob_stable = if stable_total > 0 {
            stable_up as f64 / stable_total as f64
        } else {
            0.5
        };

        let lift = if prob_stable > 0.1 {
            prob_increasing.max(1.0 - prob_decreasing) / prob_stable
        } else {
            1.0
        };

        let significant = (prob_increasing - prob_decreasing).abs() > 0.1
            && increasing_total >= 15
            && decreasing_total >= 15;

        ConcentrationDirectionTest {
            prob_up_increasing: prob_increasing,
            prob_up_decreasing: prob_decreasing,
            prob_up_stable: prob_stable,
            lift,
            sample_size: n,
            significant,
        }
    }

    /// Test if whale accumulation predicts returns
    pub fn test_whale_accumulation(
        whale_ratio_change: &[f64],
        future_returns: &[f64],
        horizon: usize,
    ) -> WhaleAccumulationTest {
        let n = whale_ratio_change.len().min(future_returns.len());

        if n < horizon + 50 {
            return WhaleAccumulationTest {
                correlation: 0.0,
                return_whale_accumulating: 0.0,
                return_whale_distributing: 0.0,
                sample_size: n,
                significant: false,
            };
        }

        // Align: ratio_change at t predicts return at t+horizon
        let change_aligned: Vec<f64> = whale_ratio_change[..(n - horizon)].to_vec();
        let returns_aligned: Vec<f64> = future_returns[horizon..n].to_vec();

        let correlation = pearson_correlation(&change_aligned, &returns_aligned);

        // Split by accumulating vs distributing
        let mut ret_accumulating = Vec::new();
        let mut ret_distributing = Vec::new();

        for i in 0..change_aligned.len() {
            if change_aligned[i] > 0.01 {
                ret_accumulating.push(returns_aligned[i]);
            } else if change_aligned[i] < -0.01 {
                ret_distributing.push(returns_aligned[i]);
            }
        }

        let mean_accumulating = if !ret_accumulating.is_empty() {
            ret_accumulating.iter().sum::<f64>() / ret_accumulating.len() as f64
        } else {
            0.0
        };

        let mean_distributing = if !ret_distributing.is_empty() {
            ret_distributing.iter().sum::<f64>() / ret_distributing.len() as f64
        } else {
            0.0
        };

        let significant = correlation.abs() > 0.05
            && ret_accumulating.len() >= 20
            && ret_distributing.len() >= 20;

        WhaleAccumulationTest {
            correlation,
            return_whale_accumulating: mean_accumulating,
            return_whale_distributing: mean_distributing,
            sample_size: n - horizon,
            significant,
        }
    }

    /// Pearson correlation coefficient
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
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_position(value: f64, is_whale: bool) -> Position {
        Position {
            value_usd: value,
            is_whale,
        }
    }

    // ========================================================================
    // Feature Count Tests
    // ========================================================================

    #[test]
    fn test_feature_count() {
        assert_eq!(ConcentrationFeatures::count(), 15);
        assert_eq!(ConcentrationFeatures::names().len(), 15);
        assert_eq!(ConcentrationFeatures::default().to_vec().len(), 15);
    }

    // ========================================================================
    // Top-N Concentration Tests
    // ========================================================================

    #[test]
    fn test_top_n_concentration() {
        let positions = vec![
            make_position(1_000_000.0, true),  // 10%
            make_position(900_000.0, true),    // 9%
            make_position(800_000.0, true),    // 8%
            make_position(700_000.0, false),   // 7%
            make_position(600_000.0, false),   // 6%
            // Rest is smaller positions
            make_position(100_000.0, false),
            make_position(100_000.0, false),
            make_position(100_000.0, false),
            make_position(100_000.0, false),
            make_position(100_000.0, false),
        ];

        let total_oi = 4_500_000.0;
        let features = compute(&positions, total_oi);

        // Top 5 = 1M + 0.9M + 0.8M + 0.7M + 0.6M = 4M
        // 4M / 4.5M = 0.889
        assert!(features.top5_concentration > 0.85,
            "Top 5 concentration should be ~89%, got {}", features.top5_concentration);

        // Top 10 includes all = 4.5M / 4.5M = 1.0
        assert!(features.top10_concentration > 0.99,
            "Top 10 should include all, got {}", features.top10_concentration);
    }

    #[test]
    fn test_equal_distribution() {
        // 100 equal positions
        let positions: Vec<Position> = (0..100)
            .map(|_| make_position(10_000.0, false))
            .collect();

        let total_oi = 1_000_000.0;
        let features = compute(&positions, total_oi);

        // Top 10 = 10% of positions = 10% of OI
        assert!((features.top10_concentration - 0.1).abs() < 0.01,
            "Top 10 of equal distribution should be 10%, got {}", features.top10_concentration);

        // Gini should be 0 (perfect equality)
        assert!(features.gini_coefficient < 0.05,
            "Gini should be near 0 for equal distribution, got {}", features.gini_coefficient);

        // HHI should be 0.01 (1/100 for each = 100 * (0.01)² = 0.01)
        assert!(features.herfindahl_index < 0.02,
            "HHI should be low for equal distribution, got {}", features.herfindahl_index);
    }

    #[test]
    fn test_high_concentration() {
        // One dominant position
        let positions = vec![
            make_position(9_000_000.0, true),  // 90%
            make_position(100_000.0, false),
            make_position(100_000.0, false),
            make_position(100_000.0, false),
            make_position(100_000.0, false),
            make_position(100_000.0, false),
            make_position(100_000.0, false),
            make_position(100_000.0, false),
            make_position(100_000.0, false),
            make_position(100_000.0, false),
        ];

        let total_oi = 10_000_000.0;
        let features = compute(&positions, total_oi);

        // Top 1 = 90%
        assert!(features.top5_concentration > 0.9,
            "Top 5 should be dominated by single whale, got {}", features.top5_concentration);

        // Gini should be high (near 1)
        assert!(features.gini_coefficient > 0.7,
            "Gini should be high for concentrated distribution, got {}", features.gini_coefficient);

        // HHI should be high (0.9² = 0.81)
        assert!(features.herfindahl_index > 0.75,
            "HHI should be high for concentrated distribution, got {}", features.herfindahl_index);
    }

    // ========================================================================
    // Whale vs Retail Tests
    // ========================================================================

    #[test]
    fn test_whale_retail_ratio() {
        let positions = vec![
            // Whales: 3M total
            make_position(1_500_000.0, true),
            make_position(1_000_000.0, true),
            make_position(500_000.0, true),
            // Retail: 1M total
            make_position(200_000.0, false),
            make_position(200_000.0, false),
            make_position(200_000.0, false),
            make_position(200_000.0, false),
            make_position(200_000.0, false),
        ];

        let total_oi = 4_000_000.0;
        let features = compute(&positions, total_oi);

        // Whale/Retail = 3M/1M = 3.0
        assert!((features.whale_retail_ratio - 3.0).abs() < 0.1,
            "Whale/retail ratio should be 3.0, got {}", features.whale_retail_ratio);

        // Whale fraction = 3/8 = 0.375
        assert!((features.whale_fraction - 0.375).abs() < 0.01,
            "Whale fraction should be 0.375, got {}", features.whale_fraction);

        // Whale count = 3
        assert_eq!(features.whale_position_count, 3.0,
            "Should have 3 whale positions");
    }

    // ========================================================================
    // Dynamics Tests
    // ========================================================================

    #[test]
    fn test_concentration_change() {
        let config = ConcentrationConfig::default();
        let mut buffer = ConcentrationBuffer::new(config);

        // Create many small positions so top10 concentration is meaningful
        // First snapshot: distributed (top 10 holds ~50% of 10M OI)
        let mut positions1: Vec<Position> = (0..10)
            .map(|_| make_position(500_000.0, false))  // 10 * 500k = 5M
            .collect();
        // Add 50 smaller positions (50 * 100k = 5M)
        positions1.extend((0..50).map(|_| make_position(100_000.0, false)));

        let features1 = buffer.compute(&positions1, 10_000_000.0);
        let top10_first = features1.top10_concentration;

        // Second snapshot: concentrated (top 10 holds ~70% of 10M OI)
        let mut positions2: Vec<Position> = (0..10)
            .map(|_| make_position(700_000.0, true))  // 10 * 700k = 7M
            .collect();
        // Add 30 smaller positions (30 * 100k = 3M)
        positions2.extend((0..30).map(|_| make_position(100_000.0, false)));

        let features2 = buffer.compute(&positions2, 10_000_000.0);

        // Verify top10 increased
        assert!(features2.top10_concentration > top10_first,
            "Top10 should increase: {} -> {}", top10_first, features2.top10_concentration);

        // Concentration change should be positive
        assert!(features2.concentration_change_1h > 0.0,
            "Concentration should be increasing, got change {}", features2.concentration_change_1h);
    }

    // ========================================================================
    // Empty Input Tests
    // ========================================================================

    #[test]
    fn test_empty_positions() {
        let features = compute(&[], 1_000_000.0);

        assert_eq!(features.top10_concentration, 0.0);
        assert_eq!(features.gini_coefficient, 0.0);
        assert_eq!(features.position_count, 0.0);
    }

    #[test]
    fn test_zero_oi() {
        let positions = vec![make_position(100_000.0, false)];
        let features = compute(&positions, 0.0);

        assert_eq!(features.top10_concentration, 0.0);
    }

    // ========================================================================
    // Gini Coefficient Tests
    // ========================================================================

    #[test]
    fn test_gini_perfect_equality() {
        let values = vec![100.0, 100.0, 100.0, 100.0];
        let gini = compute_gini(&values);

        assert!(gini.abs() < 0.01,
            "Gini of equal values should be 0, got {}", gini);
    }

    #[test]
    fn test_gini_high_inequality() {
        let values = vec![1.0, 1.0, 1.0, 97.0];
        let gini = compute_gini(&values);

        assert!(gini > 0.7,
            "Gini of highly unequal values should be high, got {}", gini);
    }

    // ========================================================================
    // HHI Tests
    // ========================================================================

    #[test]
    fn test_hhi_monopoly() {
        let positions = vec![1_000_000.0];
        let hhi = compute_herfindahl(&positions, 1_000_000.0);

        assert!((hhi - 1.0).abs() < 0.01,
            "HHI of monopoly should be 1.0, got {}", hhi);
    }

    #[test]
    fn test_hhi_duopoly() {
        let positions = vec![500_000.0, 500_000.0];
        let hhi = compute_herfindahl(&positions, 1_000_000.0);

        // 0.5² + 0.5² = 0.5
        assert!((hhi - 0.5).abs() < 0.01,
            "HHI of equal duopoly should be 0.5, got {}", hhi);
    }

    // ========================================================================
    // Skeptical Tests
    // ========================================================================

    #[test]
    fn test_concentration_volatility_relationship() {
        use skeptical_tests::test_concentration_volatility;

        let n = 200;
        // High concentration -> high volatility
        let concentration: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 0.6 } else { 0.3 })
            .collect();

        let volatility: Vec<f64> = (0..n)
            .map(|i| {
                if concentration[i] > 0.5 { 0.05 } else { 0.02 }
            })
            .collect();

        let result = test_concentration_volatility(&concentration, &volatility, 0.5);

        assert!(result.volatility_ratio > 2.0,
            "High concentration should have higher vol, got ratio {}", result.volatility_ratio);
    }

    #[test]
    fn test_concentration_direction_relationship() {
        use skeptical_tests::test_concentration_direction;

        let n = 200;
        // Increasing concentration -> up moves
        let concentration_change: Vec<f64> = (0..n)
            .map(|i| {
                if i % 3 == 0 { 0.05 }
                else if i % 3 == 1 { -0.05 }
                else { 0.0 }
            })
            .collect();

        let future_returns: Vec<f64> = (0..n)
            .map(|i| {
                if concentration_change[i] > 0.02 { 0.02 }
                else if concentration_change[i] < -0.02 { -0.02 }
                else { 0.001 }
            })
            .collect();

        let result = test_concentration_direction(&concentration_change, &future_returns, 0.02);

        assert!(result.prob_up_increasing > result.prob_up_decreasing,
            "Increasing concentration should predict up, got inc={}, dec={}",
            result.prob_up_increasing, result.prob_up_decreasing);
    }

    #[test]
    fn test_whale_accumulation_relationship() {
        use skeptical_tests::test_whale_accumulation;

        let n = 200;
        let horizon = 4;

        // Whale accumulation predicts positive returns
        let whale_ratio_change: Vec<f64> = (0..n)
            .map(|i| ((i as f64 * 0.1).sin() * 0.1))
            .collect();

        let future_returns: Vec<f64> = (0..n)
            .map(|i| {
                if i >= horizon {
                    whale_ratio_change[i - horizon] * 0.5 + (i as f64 * 0.2).cos() * 0.01
                } else {
                    0.0
                }
            })
            .collect();

        let result = test_whale_accumulation(&whale_ratio_change, &future_returns, horizon);

        assert!(result.sample_size > 100,
            "Should have enough samples, got {}", result.sample_size);
    }

    // ========================================================================
    // Theil Index Tests
    // ========================================================================

    #[test]
    fn test_theil_equal() {
        let values = vec![100.0, 100.0, 100.0, 100.0];
        let theil = compute_theil(&values);

        assert!(theil.abs() < 0.01,
            "Theil of equal values should be 0, got {}", theil);
    }

    #[test]
    fn test_theil_unequal() {
        let values = vec![10.0, 20.0, 30.0, 140.0];
        let theil = compute_theil(&values);

        assert!(theil > 0.1,
            "Theil of unequal values should be positive, got {}", theil);
    }
}
