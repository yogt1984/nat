//! Liquidation Risk Map Feature Extraction
//!
//! This module computes liquidation risk features that map the dollar value
//! at risk of liquidation at different price levels - a key Hyperliquid-unique feature.
//!
//! # Features
//!
//! | Feature | Formula | Interpretation |
//! |---------|---------|----------------|
//! | **liquidation_risk_above_Xpct** | Σ(short positions liquidated if price ↑ X%) | Cascade risk on rally |
//! | **liquidation_risk_below_Xpct** | Σ(long positions liquidated if price ↓ X%) | Cascade risk on dump |
//! | **liquidation_asymmetry** | risk_above / risk_below | Directional bias |
//! | **liquidation_intensity** | total_risk / total_OI | Overall leverage risk |
//!
//! # Theory
//!
//! Liquidation cascades occur when:
//! 1. Price approaches a cluster of liquidation prices
//! 2. Forced liquidations push price further in that direction
//! 3. This triggers more liquidations, creating a cascade
//!
//! On Hyperliquid, we can see liquidation prices for all positions,
//! allowing us to predict cascade risk before it happens.
//!
//! # Position Mechanics
//!
//! - **Long positions**: Liquidation price is BELOW entry (liquidated when price falls)
//! - **Short positions**: Liquidation price is ABOVE entry (liquidated when price rises)

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Configuration for liquidation risk computation
#[derive(Debug, Clone)]
pub struct LiquidationRiskConfig {
    /// Price distance buckets (percentages)
    pub distance_buckets: Vec<f64>,
    /// Minimum position value to include (USD)
    pub min_position_value: f64,
}

impl Default for LiquidationRiskConfig {
    fn default() -> Self {
        Self {
            distance_buckets: vec![1.0, 2.0, 5.0, 10.0],
            min_position_value: 1000.0, // Ignore tiny positions
        }
    }
}

/// Liquidation risk features
/// Total: 13 features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidationRiskFeatures {
    // === Risk Above (shorts liquidated on price rise) ===
    /// USD at risk if price rises 1%
    pub liquidation_risk_above_1pct: f64,
    /// USD at risk if price rises 2%
    pub liquidation_risk_above_2pct: f64,
    /// USD at risk if price rises 5%
    pub liquidation_risk_above_5pct: f64,
    /// USD at risk if price rises 10%
    pub liquidation_risk_above_10pct: f64,

    // === Risk Below (longs liquidated on price fall) ===
    /// USD at risk if price falls 1%
    pub liquidation_risk_below_1pct: f64,
    /// USD at risk if price falls 2%
    pub liquidation_risk_below_2pct: f64,
    /// USD at risk if price falls 5%
    pub liquidation_risk_below_5pct: f64,
    /// USD at risk if price falls 10%
    pub liquidation_risk_below_10pct: f64,

    // === Derived Metrics ===
    /// Asymmetry ratio: risk_above_5pct / risk_below_5pct
    /// > 1 means more shorts at risk (bullish pressure on squeeze)
    /// < 1 means more longs at risk (bearish pressure on cascade)
    pub liquidation_asymmetry: f64,

    /// Total risk at 5% as fraction of total OI
    pub liquidation_intensity: f64,

    /// Number of positions at risk within 5%
    pub positions_at_risk_count: f64,

    /// Largest single position at risk within 5% (USD)
    pub largest_position_at_risk: f64,

    /// Distance to nearest significant liquidation cluster (%)
    pub nearest_cluster_distance: f64,
}

impl LiquidationRiskFeatures {
    pub fn count() -> usize {
        13
    }

    pub fn names() -> Vec<&'static str> {
        vec![
            "liquidation_risk_above_1pct",
            "liquidation_risk_above_2pct",
            "liquidation_risk_above_5pct",
            "liquidation_risk_above_10pct",
            "liquidation_risk_below_1pct",
            "liquidation_risk_below_2pct",
            "liquidation_risk_below_5pct",
            "liquidation_risk_below_10pct",
            "liquidation_asymmetry",
            "liquidation_intensity",
            "positions_at_risk_count",
            "largest_position_at_risk",
            "nearest_cluster_distance",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.liquidation_risk_above_1pct,
            self.liquidation_risk_above_2pct,
            self.liquidation_risk_above_5pct,
            self.liquidation_risk_above_10pct,
            self.liquidation_risk_below_1pct,
            self.liquidation_risk_below_2pct,
            self.liquidation_risk_below_5pct,
            self.liquidation_risk_below_10pct,
            self.liquidation_asymmetry,
            self.liquidation_intensity,
            self.positions_at_risk_count,
            self.largest_position_at_risk,
            self.nearest_cluster_distance,
        ]
    }
}

impl Default for LiquidationRiskFeatures {
    fn default() -> Self {
        Self {
            liquidation_risk_above_1pct: 0.0,
            liquidation_risk_above_2pct: 0.0,
            liquidation_risk_above_5pct: 0.0,
            liquidation_risk_above_10pct: 0.0,
            liquidation_risk_below_1pct: 0.0,
            liquidation_risk_below_2pct: 0.0,
            liquidation_risk_below_5pct: 0.0,
            liquidation_risk_below_10pct: 0.0,
            liquidation_asymmetry: 1.0, // Balanced when no data
            liquidation_intensity: 0.0,
            positions_at_risk_count: 0.0,
            largest_position_at_risk: 0.0,
            nearest_cluster_distance: f64::MAX,
        }
    }
}

/// A position with liquidation information
#[derive(Debug, Clone)]
pub struct LiquidationPosition {
    /// Position value in USD (absolute)
    pub position_value_usd: f64,
    /// Liquidation price
    pub liquidation_price: f64,
    /// Whether this is a long position
    pub is_long: bool,
    /// Entry price
    pub entry_price: f64,
}

impl LiquidationPosition {
    /// Create from position snapshot data
    pub fn from_snapshot(
        size: f64,
        entry_price: f64,
        liquidation_price: f64,
        position_value_usd: f64,
    ) -> Self {
        Self {
            position_value_usd: position_value_usd.abs(),
            liquidation_price,
            is_long: size > 0.0,
            entry_price,
        }
    }

    /// Calculate distance from current price to liquidation as percentage
    pub fn distance_from_price(&self, current_price: f64) -> f64 {
        if current_price <= 0.0 {
            return f64::MAX;
        }
        ((self.liquidation_price - current_price) / current_price * 100.0).abs()
    }

    /// Check if this position would be liquidated by price moving in given direction
    /// direction > 0 means price going up, direction < 0 means price going down
    pub fn is_at_risk(&self, current_price: f64, price_change_pct: f64) -> bool {
        if price_change_pct > 0.0 {
            // Price going up - shorts get liquidated (liq price is above current)
            !self.is_long && self.liquidation_price <= current_price * (1.0 + price_change_pct / 100.0)
        } else {
            // Price going down - longs get liquidated (liq price is below current)
            self.is_long && self.liquidation_price >= current_price * (1.0 + price_change_pct / 100.0)
        }
    }
}

/// Compute liquidation risk features from positions
pub fn compute(
    positions: &[LiquidationPosition],
    current_price: f64,
    total_oi: f64,
    config: &LiquidationRiskConfig,
) -> LiquidationRiskFeatures {
    if positions.is_empty() || current_price <= 0.0 {
        return LiquidationRiskFeatures::default();
    }

    // Filter positions above minimum value
    let valid_positions: Vec<&LiquidationPosition> = positions
        .iter()
        .filter(|p| p.position_value_usd >= config.min_position_value)
        .collect();

    if valid_positions.is_empty() {
        return LiquidationRiskFeatures::default();
    }

    // Compute risk at each bucket
    let mut risk_above = HashMap::new();
    let mut risk_below = HashMap::new();

    for &bucket in &config.distance_buckets {
        risk_above.insert(bucket as i32, 0.0f64);
        risk_below.insert(bucket as i32, 0.0f64);
    }

    let mut positions_at_risk_5pct = 0;
    let mut largest_at_risk_5pct = 0.0f64;
    let mut nearest_distance = f64::MAX;

    for pos in &valid_positions {
        let distance = pos.distance_from_price(current_price);

        // Track nearest liquidation
        if distance < nearest_distance {
            nearest_distance = distance;
        }

        // Bucket the risk
        for &bucket in &config.distance_buckets {
            let bucket_key = bucket as i32;

            if pos.is_long {
                // Longs liquidated when price falls - risk below
                // Check if liquidation price is within bucket% below current
                let threshold_price = current_price * (1.0 - bucket / 100.0);
                if pos.liquidation_price >= threshold_price && pos.liquidation_price < current_price {
                    *risk_below.get_mut(&bucket_key).unwrap() += pos.position_value_usd;
                }
            } else {
                // Shorts liquidated when price rises - risk above
                // Check if liquidation price is within bucket% above current
                let threshold_price = current_price * (1.0 + bucket / 100.0);
                if pos.liquidation_price <= threshold_price && pos.liquidation_price > current_price {
                    *risk_above.get_mut(&bucket_key).unwrap() += pos.position_value_usd;
                }
            }
        }

        // Count positions at risk within 5%
        if distance <= 5.0 {
            positions_at_risk_5pct += 1;
            if pos.position_value_usd > largest_at_risk_5pct {
                largest_at_risk_5pct = pos.position_value_usd;
            }
        }
    }

    // Extract values for standard buckets
    let risk_above_1 = *risk_above.get(&1).unwrap_or(&0.0);
    let risk_above_2 = *risk_above.get(&2).unwrap_or(&0.0);
    let risk_above_5 = *risk_above.get(&5).unwrap_or(&0.0);
    let risk_above_10 = *risk_above.get(&10).unwrap_or(&0.0);

    let risk_below_1 = *risk_below.get(&1).unwrap_or(&0.0);
    let risk_below_2 = *risk_below.get(&2).unwrap_or(&0.0);
    let risk_below_5 = *risk_below.get(&5).unwrap_or(&0.0);
    let risk_below_10 = *risk_below.get(&10).unwrap_or(&0.0);

    // Compute asymmetry (risk_above / risk_below at 5%)
    let asymmetry = if risk_below_5 > 1e-6 {
        risk_above_5 / risk_below_5
    } else if risk_above_5 > 1e-6 {
        10.0 // Cap at 10 when denominator is near zero
    } else {
        1.0 // Balanced when both are zero
    };

    // Compute intensity (total risk at 5% / total OI)
    let total_risk_5 = risk_above_5 + risk_below_5;
    let intensity = if total_oi > 1e-6 {
        total_risk_5 / total_oi
    } else {
        0.0
    };

    // Find nearest cluster (significant concentration)
    let cluster_distance = find_nearest_cluster(&valid_positions, current_price, config);

    LiquidationRiskFeatures {
        liquidation_risk_above_1pct: risk_above_1,
        liquidation_risk_above_2pct: risk_above_2,
        liquidation_risk_above_5pct: risk_above_5,
        liquidation_risk_above_10pct: risk_above_10,
        liquidation_risk_below_1pct: risk_below_1,
        liquidation_risk_below_2pct: risk_below_2,
        liquidation_risk_below_5pct: risk_below_5,
        liquidation_risk_below_10pct: risk_below_10,
        liquidation_asymmetry: asymmetry,
        liquidation_intensity: intensity,
        positions_at_risk_count: positions_at_risk_5pct as f64,
        largest_position_at_risk: largest_at_risk_5pct,
        nearest_cluster_distance: cluster_distance.min(nearest_distance),
    }
}

/// Find distance to nearest liquidation cluster
/// A cluster is defined as > $1M within 0.5% price range
fn find_nearest_cluster(
    positions: &[&LiquidationPosition],
    current_price: f64,
    _config: &LiquidationRiskConfig,
) -> f64 {
    const CLUSTER_THRESHOLD: f64 = 1_000_000.0; // $1M
    const BUCKET_SIZE_PCT: f64 = 0.5; // 0.5% buckets

    if positions.is_empty() || current_price <= 0.0 {
        return f64::MAX;
    }

    // Create buckets from -20% to +20% in 0.5% increments
    let num_buckets = 80; // 40 below, 40 above
    let mut buckets = vec![0.0f64; num_buckets];

    for pos in positions {
        let distance_pct = (pos.liquidation_price - current_price) / current_price * 100.0;

        // Map to bucket index (0 = -20%, 40 = 0%, 80 = +20%)
        let bucket_idx = ((distance_pct / BUCKET_SIZE_PCT) + 40.0) as i32;

        if bucket_idx >= 0 && (bucket_idx as usize) < num_buckets {
            buckets[bucket_idx as usize] += pos.position_value_usd;
        }
    }

    // Find nearest cluster
    let center_bucket = 40; // 0% distance
    let mut nearest_cluster_distance = f64::MAX;

    for (idx, &value) in buckets.iter().enumerate() {
        if value >= CLUSTER_THRESHOLD {
            let bucket_distance = ((idx as i32 - center_bucket) as f64).abs() * BUCKET_SIZE_PCT;
            if bucket_distance < nearest_cluster_distance {
                nearest_cluster_distance = bucket_distance;
            }
        }
    }

    nearest_cluster_distance
}

// ============================================================================
// Skeptical Tests Module
// ============================================================================

pub mod skeptical_tests {
    //! Skeptical tests to validate liquidation risk feature effectiveness
    //!
    //! These tests verify:
    //! 1. Liquidation clusters predict volatility spikes
    //! 2. Liquidation asymmetry predicts price direction
    //! 3. Signal provides actionable lead time

    /// Result of cluster volatility prediction test
    #[derive(Debug, Clone)]
    pub struct ClusterVolatilityTest {
        /// Volatility when price approaches cluster
        pub vol_near_cluster: f64,
        /// Volatility when price is away from cluster
        pub vol_away_from_cluster: f64,
        /// Ratio (vol_near / vol_away)
        pub volatility_lift: f64,
        /// Sample count near cluster
        pub samples_near: usize,
        /// Sample count away from cluster
        pub samples_away: usize,
        pub significant: bool,
    }

    /// Result of asymmetry direction prediction test
    #[derive(Debug, Clone)]
    pub struct AsymmetryDirectionTest {
        /// P(price up | high asymmetry) - asymmetry > 1.5
        pub prob_up_given_high_asymmetry: f64,
        /// P(price up | low asymmetry) - asymmetry < 0.67
        pub prob_up_given_low_asymmetry: f64,
        /// P(price up | balanced) - 0.67 < asymmetry < 1.5
        pub prob_up_baseline: f64,
        /// Lift from using asymmetry
        pub predictive_lift: f64,
        pub sample_size: usize,
        pub significant: bool,
    }

    /// Result of cascade prediction precision test
    #[derive(Debug, Clone)]
    pub struct CascadePrecisionTest {
        /// Number of predicted cascades (price approached cluster)
        pub predicted_cascades: usize,
        /// Number of actual cascades (> 5% move within 1h)
        pub actual_cascades: usize,
        /// Precision: actual / predicted
        pub precision: f64,
        /// Recall: actual cascades we caught / total actual cascades
        pub recall: f64,
        /// Lead time in periods before cascade
        pub avg_lead_time: f64,
        pub significant: bool,
    }

    /// Test if price approaching liquidation clusters increases volatility
    pub fn test_cluster_volatility(
        cluster_distance: &[f64],
        realized_volatility: &[f64],
        threshold_pct: f64,
    ) -> ClusterVolatilityTest {
        let n = cluster_distance.len().min(realized_volatility.len());

        if n < 50 {
            return ClusterVolatilityTest {
                vol_near_cluster: 0.0,
                vol_away_from_cluster: 0.0,
                volatility_lift: 1.0,
                samples_near: 0,
                samples_away: 0,
                significant: false,
            };
        }

        let mut vol_near = Vec::new();
        let mut vol_away = Vec::new();

        for i in 0..n {
            if cluster_distance[i] <= threshold_pct {
                vol_near.push(realized_volatility[i]);
            } else {
                vol_away.push(realized_volatility[i]);
            }
        }

        if vol_near.is_empty() || vol_away.is_empty() {
            return ClusterVolatilityTest {
                vol_near_cluster: 0.0,
                vol_away_from_cluster: 0.0,
                volatility_lift: 1.0,
                samples_near: vol_near.len(),
                samples_away: vol_away.len(),
                significant: false,
            };
        }

        let mean_near = vol_near.iter().sum::<f64>() / vol_near.len() as f64;
        let mean_away = vol_away.iter().sum::<f64>() / vol_away.len() as f64;

        let lift = if mean_away > 1e-10 {
            mean_near / mean_away
        } else {
            1.0
        };

        // Significant if volatility near cluster is > 1.5x volatility away
        let significant = lift > 1.5 && vol_near.len() >= 20;

        ClusterVolatilityTest {
            vol_near_cluster: mean_near,
            vol_away_from_cluster: mean_away,
            volatility_lift: lift,
            samples_near: vol_near.len(),
            samples_away: vol_away.len(),
            significant,
        }
    }

    /// Test if liquidation asymmetry predicts price direction
    pub fn test_asymmetry_direction(
        asymmetry: &[f64],
        future_returns: &[f64],
        horizon: usize,
    ) -> AsymmetryDirectionTest {
        let n = asymmetry.len().min(future_returns.len());

        if n < horizon + 50 {
            return AsymmetryDirectionTest {
                prob_up_given_high_asymmetry: 0.5,
                prob_up_given_low_asymmetry: 0.5,
                prob_up_baseline: 0.5,
                predictive_lift: 1.0,
                sample_size: n,
                significant: false,
            };
        }

        let mut high_asym_up = 0;
        let mut high_asym_total = 0;
        let mut low_asym_up = 0;
        let mut low_asym_total = 0;
        let mut balanced_up = 0;
        let mut balanced_total = 0;

        for i in 0..(n - horizon) {
            let ret = future_returns[i + horizon];
            let asym = asymmetry[i];

            if asym > 1.5 {
                // High asymmetry (more shorts at risk) - expect squeeze up
                high_asym_total += 1;
                if ret > 0.0 {
                    high_asym_up += 1;
                }
            } else if asym < 0.67 {
                // Low asymmetry (more longs at risk) - expect cascade down
                low_asym_total += 1;
                if ret > 0.0 {
                    low_asym_up += 1;
                }
            } else {
                // Balanced
                balanced_total += 1;
                if ret > 0.0 {
                    balanced_up += 1;
                }
            }
        }

        let prob_high = if high_asym_total > 0 {
            high_asym_up as f64 / high_asym_total as f64
        } else {
            0.5
        };

        let prob_low = if low_asym_total > 0 {
            low_asym_up as f64 / low_asym_total as f64
        } else {
            0.5
        };

        let prob_balanced = if balanced_total > 0 {
            balanced_up as f64 / balanced_total as f64
        } else {
            0.5
        };

        // Lift: how much better is high asymmetry at predicting up vs baseline
        let lift = if prob_balanced > 0.1 {
            prob_high / prob_balanced
        } else {
            1.0
        };

        // Significant if clear difference and enough samples
        let significant = (prob_high - prob_low).abs() > 0.1
            && high_asym_total >= 20
            && low_asym_total >= 20;

        AsymmetryDirectionTest {
            prob_up_given_high_asymmetry: prob_high,
            prob_up_given_low_asymmetry: prob_low,
            prob_up_baseline: prob_balanced,
            predictive_lift: lift,
            sample_size: n - horizon,
            significant,
        }
    }

    /// Test cascade prediction precision
    pub fn test_cascade_precision(
        cluster_distance: &[f64],
        price_moves: &[f64], // Actual price moves in next period
        cluster_threshold: f64, // Distance to be "near cluster"
        cascade_threshold: f64, // % move to count as cascade
    ) -> CascadePrecisionTest {
        let n = cluster_distance.len().min(price_moves.len());

        if n < 50 {
            return CascadePrecisionTest {
                predicted_cascades: 0,
                actual_cascades: 0,
                precision: 0.0,
                recall: 0.0,
                avg_lead_time: 0.0,
                significant: false,
            };
        }

        let mut predicted = 0;
        let mut actual = 0;
        let mut true_positives = 0;
        let mut lead_times = Vec::new();

        // Find all actual cascades
        let actual_cascade_indices: Vec<usize> = (0..n)
            .filter(|&i| price_moves[i].abs() >= cascade_threshold)
            .collect();

        let total_actual_cascades = actual_cascade_indices.len();

        for i in 0..n {
            let near_cluster = cluster_distance[i] <= cluster_threshold;
            let is_cascade = price_moves[i].abs() >= cascade_threshold;

            if near_cluster {
                predicted += 1;
                if is_cascade {
                    true_positives += 1;
                }
            }

            if is_cascade {
                actual += 1;
            }
        }

        // Calculate lead time for true positives
        // Look backwards from each cascade to find when we first signaled "near cluster"
        for &cascade_idx in &actual_cascade_indices {
            let mut lead = 0;
            for j in (0..cascade_idx).rev() {
                if cluster_distance[j] <= cluster_threshold {
                    lead = cascade_idx - j;
                    break;
                }
            }
            if lead > 0 {
                lead_times.push(lead as f64);
            }
        }

        let precision = if predicted > 0 {
            true_positives as f64 / predicted as f64
        } else {
            0.0
        };

        let recall = if total_actual_cascades > 0 {
            true_positives as f64 / total_actual_cascades as f64
        } else {
            0.0
        };

        let avg_lead = if !lead_times.is_empty() {
            lead_times.iter().sum::<f64>() / lead_times.len() as f64
        } else {
            0.0
        };

        // Significant if precision > 30% as specified in TASKS.md
        let significant = precision > 0.30 && predicted >= 10;

        CascadePrecisionTest {
            predicted_cascades: predicted,
            actual_cascades: actual,
            precision,
            recall,
            avg_lead_time: avg_lead,
            significant,
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_long_position(value_usd: f64, entry: f64, liq_price: f64) -> LiquidationPosition {
        LiquidationPosition {
            position_value_usd: value_usd,
            liquidation_price: liq_price,
            is_long: true,
            entry_price: entry,
        }
    }

    fn make_short_position(value_usd: f64, entry: f64, liq_price: f64) -> LiquidationPosition {
        LiquidationPosition {
            position_value_usd: value_usd,
            liquidation_price: liq_price,
            is_long: false,
            entry_price: entry,
        }
    }

    // ========================================================================
    // Feature Count Tests
    // ========================================================================

    #[test]
    fn test_feature_count() {
        assert_eq!(LiquidationRiskFeatures::count(), 13);
        assert_eq!(LiquidationRiskFeatures::names().len(), 13);
        assert_eq!(LiquidationRiskFeatures::default().to_vec().len(), 13);
    }

    // ========================================================================
    // Risk Above Tests (Shorts at risk on price rise)
    // ========================================================================

    #[test]
    fn test_risk_above_shorts() {
        let config = LiquidationRiskConfig::default();
        let current_price = 50000.0;

        // Shorts with liquidation prices above current
        let positions = vec![
            // Short, liq at 50500 (1% above) - at risk within 1%
            make_short_position(100_000.0, 49000.0, 50500.0),
            // Short, liq at 51000 (2% above) - at risk within 2%
            make_short_position(200_000.0, 48000.0, 51000.0),
            // Short, liq at 52500 (5% above) - at risk within 5%
            make_short_position(300_000.0, 47000.0, 52500.0),
        ];

        let features = compute(&positions, current_price, 10_000_000.0, &config);

        assert!(features.liquidation_risk_above_1pct > 0.0,
            "Should have risk above 1%, got {}", features.liquidation_risk_above_1pct);
        assert!(features.liquidation_risk_above_2pct >= features.liquidation_risk_above_1pct,
            "2% risk should include 1% risk");
        assert!(features.liquidation_risk_above_5pct >= features.liquidation_risk_above_2pct,
            "5% risk should include 2% risk");
    }

    // ========================================================================
    // Risk Below Tests (Longs at risk on price fall)
    // ========================================================================

    #[test]
    fn test_risk_below_longs() {
        let config = LiquidationRiskConfig::default();
        let current_price = 50000.0;

        // Longs with liquidation prices below current
        let positions = vec![
            // Long, liq at 49500 (1% below) - at risk within 1%
            make_long_position(100_000.0, 52000.0, 49500.0),
            // Long, liq at 49000 (2% below) - at risk within 2%
            make_long_position(200_000.0, 54000.0, 49000.0),
            // Long, liq at 47500 (5% below) - at risk within 5%
            make_long_position(300_000.0, 55000.0, 47500.0),
        ];

        let features = compute(&positions, current_price, 10_000_000.0, &config);

        assert!(features.liquidation_risk_below_1pct > 0.0,
            "Should have risk below 1%, got {}", features.liquidation_risk_below_1pct);
        assert!(features.liquidation_risk_below_2pct >= features.liquidation_risk_below_1pct,
            "2% risk should include 1% risk");
        assert!(features.liquidation_risk_below_5pct >= features.liquidation_risk_below_2pct,
            "5% risk should include 2% risk");
    }

    // ========================================================================
    // Asymmetry Tests
    // ========================================================================

    #[test]
    fn test_asymmetry_more_shorts_at_risk() {
        let config = LiquidationRiskConfig::default();
        let current_price = 50000.0;

        // More shorts at risk than longs
        let positions = vec![
            // Large shorts at risk
            make_short_position(500_000.0, 48000.0, 52000.0), // 4% above
            make_short_position(500_000.0, 47000.0, 51500.0), // 3% above
            // Small long at risk
            make_long_position(100_000.0, 52000.0, 48000.0), // 4% below
        ];

        let features = compute(&positions, current_price, 10_000_000.0, &config);

        assert!(features.liquidation_asymmetry > 1.0,
            "Asymmetry should be > 1 when more shorts at risk, got {}", features.liquidation_asymmetry);
    }

    #[test]
    fn test_asymmetry_more_longs_at_risk() {
        let config = LiquidationRiskConfig::default();
        let current_price = 50000.0;

        // More longs at risk than shorts
        let positions = vec![
            // Large longs at risk
            make_long_position(500_000.0, 52000.0, 48000.0), // 4% below
            make_long_position(500_000.0, 53000.0, 48500.0), // 3% below
            // Small short at risk
            make_short_position(100_000.0, 48000.0, 52000.0), // 4% above
        ];

        let features = compute(&positions, current_price, 10_000_000.0, &config);

        assert!(features.liquidation_asymmetry < 1.0,
            "Asymmetry should be < 1 when more longs at risk, got {}", features.liquidation_asymmetry);
    }

    // ========================================================================
    // Intensity Tests
    // ========================================================================

    #[test]
    fn test_liquidation_intensity() {
        let config = LiquidationRiskConfig::default();
        let current_price = 50000.0;
        let total_oi = 10_000_000.0;

        // 1M at risk within 5%
        let positions = vec![
            make_long_position(500_000.0, 52000.0, 48000.0),
            make_short_position(500_000.0, 48000.0, 52000.0),
        ];

        let features = compute(&positions, current_price, total_oi, &config);

        // Intensity should be ~0.1 (1M / 10M)
        assert!(features.liquidation_intensity > 0.05,
            "Intensity should reflect risk/OI ratio, got {}", features.liquidation_intensity);
    }

    // ========================================================================
    // Position Count Tests
    // ========================================================================

    #[test]
    fn test_positions_at_risk_count() {
        let config = LiquidationRiskConfig::default();
        let current_price = 50000.0;

        let positions = vec![
            // 3 positions within 5%
            make_long_position(100_000.0, 52000.0, 49000.0),  // 2% below
            make_long_position(100_000.0, 53000.0, 48000.0),  // 4% below
            make_short_position(100_000.0, 48000.0, 51500.0), // 3% above
            // 1 position outside 5%
            make_long_position(100_000.0, 60000.0, 40000.0),  // 20% below
        ];

        let features = compute(&positions, current_price, 10_000_000.0, &config);

        assert_eq!(features.positions_at_risk_count, 3.0,
            "Should count 3 positions within 5%, got {}", features.positions_at_risk_count);
    }

    // ========================================================================
    // Empty Input Tests
    // ========================================================================

    #[test]
    fn test_empty_positions() {
        let config = LiquidationRiskConfig::default();
        let features = compute(&[], 50000.0, 10_000_000.0, &config);

        assert_eq!(features.liquidation_risk_above_5pct, 0.0);
        assert_eq!(features.liquidation_risk_below_5pct, 0.0);
        assert_eq!(features.liquidation_asymmetry, 1.0);
    }

    #[test]
    fn test_zero_price() {
        let config = LiquidationRiskConfig::default();
        let positions = vec![
            make_long_position(100_000.0, 50000.0, 45000.0),
        ];

        let features = compute(&positions, 0.0, 10_000_000.0, &config);

        assert_eq!(features.liquidation_risk_above_5pct, 0.0);
    }

    // ========================================================================
    // Skeptical Tests
    // ========================================================================

    #[test]
    fn test_cluster_volatility_relationship() {
        use skeptical_tests::test_cluster_volatility;

        // Create data where volatility is higher near clusters
        let n = 200;
        let cluster_distance: Vec<f64> = (0..n)
            .map(|i| if i % 10 < 3 { 2.0 } else { 8.0 }) // 30% near cluster
            .collect();

        let realized_volatility: Vec<f64> = (0..n)
            .map(|i| {
                if cluster_distance[i] <= 3.0 {
                    0.05 // 5% vol near cluster
                } else {
                    0.02 // 2% vol away
                }
            })
            .collect();

        let result = test_cluster_volatility(&cluster_distance, &realized_volatility, 3.0);

        assert!(result.volatility_lift > 2.0,
            "Volatility should be higher near clusters, got lift {}", result.volatility_lift);
        assert!(result.significant, "Should be significant with clear relationship");
    }

    #[test]
    fn test_asymmetry_direction_relationship() {
        use skeptical_tests::test_asymmetry_direction;

        // Create data where high asymmetry predicts up moves
        // Note: test aligns asymmetry[i] with future_returns[i + horizon]
        // So we need returns[i] to be based on asymmetry[i - horizon]
        let n = 200;
        let horizon = 1;

        let asymmetry: Vec<f64> = (0..n)
            .map(|i| {
                if i % 3 == 0 { 2.0 }      // High asymmetry (shorts at risk)
                else if i % 3 == 1 { 0.5 } // Low asymmetry (longs at risk)
                else { 1.0 }               // Balanced
            })
            .collect();

        // Returns at i are based on asymmetry at i-horizon (to match test alignment)
        let future_returns: Vec<f64> = (0..n)
            .map(|i| {
                if i < horizon {
                    0.001 // No prior asymmetry data
                } else {
                    let prior_asym = asymmetry[i - horizon];
                    if prior_asym > 1.5 { 0.02 }       // Up when prior high asym
                    else if prior_asym < 0.67 { -0.02 } // Down when prior low asym
                    else { 0.001 }                      // Flat when balanced
                }
            })
            .collect();

        let result = test_asymmetry_direction(&asymmetry, &future_returns, horizon);

        assert!(result.prob_up_given_high_asymmetry > result.prob_up_given_low_asymmetry,
            "High asymmetry should predict up better than low asymmetry, got high={}, low={}",
            result.prob_up_given_high_asymmetry, result.prob_up_given_low_asymmetry);
    }

    #[test]
    fn test_cascade_precision() {
        use skeptical_tests::test_cascade_precision;

        // Create data with some cascade events
        let n = 100;
        let cluster_distance: Vec<f64> = (0..n)
            .map(|i| if i % 20 < 5 { 2.0 } else { 10.0 })
            .collect();

        let price_moves: Vec<f64> = (0..n)
            .map(|i| {
                if cluster_distance[i] <= 3.0 && i % 20 == 3 {
                    6.0 // Cascade when near cluster
                } else {
                    1.0 // Normal move
                }
            })
            .collect();

        let result = test_cascade_precision(&cluster_distance, &price_moves, 3.0, 5.0);

        assert!(result.predicted_cascades > 0, "Should predict some cascades");
        // Note: Precision depends on synthetic data quality
    }

    // ========================================================================
    // Distance Calculation Tests
    // ========================================================================

    #[test]
    fn test_distance_from_price() {
        let pos = make_long_position(100_000.0, 50000.0, 45000.0);
        let distance = pos.distance_from_price(50000.0);

        // 45000 is 10% below 50000
        assert!((distance - 10.0).abs() < 0.1,
            "Distance should be 10%, got {}", distance);
    }

    // ========================================================================
    // Largest Position At Risk Tests
    // ========================================================================

    #[test]
    fn test_largest_position_at_risk() {
        let config = LiquidationRiskConfig::default();
        let current_price = 50000.0;

        let positions = vec![
            make_long_position(100_000.0, 52000.0, 49000.0),  // 2% below
            make_long_position(500_000.0, 53000.0, 48000.0),  // 4% below - largest
            make_short_position(200_000.0, 48000.0, 51500.0), // 3% above
        ];

        let features = compute(&positions, current_price, 10_000_000.0, &config);

        assert_eq!(features.largest_position_at_risk, 500_000.0,
            "Should identify largest position at risk, got {}", features.largest_position_at_risk);
    }
}
