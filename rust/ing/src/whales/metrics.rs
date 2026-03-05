//! Whale metrics and analysis
//!
//! Computes concentration metrics, skill analysis, and behavioral patterns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{WhaleClassification, WhaleTier};

/// Concentration metrics for whale analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WhaleConcentration {
    /// Timestamp of calculation
    pub timestamp_ms: i64,
    /// Total open interest (USD)
    pub total_oi_usd: f64,
    /// Sum of top 10 whale positions (USD)
    pub top10_position_usd: f64,
    /// Top 10 concentration ratio (top10 / total)
    pub top10_ratio: f64,
    /// Herfindahl-Hirschman Index (sum of squared market shares)
    pub hhi: f64,
    /// Gini coefficient (0 = equal, 1 = one whale has all)
    pub gini: f64,
    /// Number of whales in calculation
    pub whale_count: usize,
}

impl WhaleConcentration {
    /// Calculate concentration metrics from whale positions
    pub fn calculate(whales: &[WhaleClassification], total_oi: f64, timestamp_ms: i64) -> Self {
        if whales.is_empty() || total_oi <= 0.0 {
            return Self {
                timestamp_ms,
                total_oi_usd: total_oi,
                ..Default::default()
            };
        }

        // Get position sizes
        let mut positions: Vec<f64> = whales.iter()
            .map(|w| w.stats.current_position_usd.abs())
            .collect();

        // Sort descending for top N calculation
        positions.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Top 10 concentration
        let top10_position: f64 = positions.iter().take(10).sum();
        let top10_ratio = top10_position / total_oi;

        // HHI: sum of squared market shares
        let hhi: f64 = positions.iter()
            .map(|p| {
                let share = p / total_oi;
                share * share
            })
            .sum();

        // Gini coefficient
        let gini = calculate_gini(&positions);

        Self {
            timestamp_ms,
            total_oi_usd: total_oi,
            top10_position_usd: top10_position,
            top10_ratio,
            hhi,
            gini,
            whale_count: whales.len(),
        }
    }

    /// Interpret concentration level
    pub fn concentration_level(&self) -> ConcentrationLevel {
        if self.top10_ratio > 0.5 {
            ConcentrationLevel::VeryHigh
        } else if self.top10_ratio > 0.3 {
            ConcentrationLevel::High
        } else if self.top10_ratio > 0.15 {
            ConcentrationLevel::Moderate
        } else {
            ConcentrationLevel::Low
        }
    }
}

/// Interpretation of concentration level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConcentrationLevel {
    Low,
    Moderate,
    High,
    VeryHigh,
}

/// Whale skill/performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WhaleSkillMetrics {
    /// Wallet address
    pub address: String,
    /// Total realized PnL
    pub total_pnl: f64,
    /// Win rate (profitable trades / total trades)
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Sharpe-like ratio (avg return / std dev)
    pub risk_adjusted_return: f64,
    /// Average trade duration (seconds)
    pub avg_trade_duration_secs: f64,
    /// Directional accuracy (% of trades in correct direction)
    pub directional_accuracy: f64,
    /// Is this whale skilled or just large?
    pub skill_tier: SkillTier,
}

/// Skill classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SkillTier {
    #[default]
    /// Consistently losing money
    Unskilled,
    /// Breaking even or slight gains
    Average,
    /// Consistent profits
    Skilled,
    /// Exceptional risk-adjusted returns
    Elite,
    /// Likely market maker (profits from spread)
    MarketMaker,
}

impl WhaleSkillMetrics {
    /// Calculate skill metrics from trade history
    pub fn calculate(
        address: &str,
        pnl_history: &[f64],
        win_count: usize,
        loss_count: usize,
        gross_profit: f64,
        gross_loss: f64,
    ) -> Self {
        let total_trades = win_count + loss_count;
        let total_pnl: f64 = pnl_history.iter().sum();

        let win_rate = if total_trades > 0 {
            win_count as f64 / total_trades as f64
        } else {
            0.0
        };

        let profit_factor = if gross_loss.abs() > 0.0 {
            gross_profit / gross_loss.abs()
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Calculate risk-adjusted return (simplified Sharpe)
        let avg_return = if !pnl_history.is_empty() {
            total_pnl / pnl_history.len() as f64
        } else {
            0.0
        };

        let std_dev = if pnl_history.len() > 1 {
            let variance: f64 = pnl_history.iter()
                .map(|p| (p - avg_return).powi(2))
                .sum::<f64>() / (pnl_history.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let risk_adjusted_return = if std_dev > 0.0 {
            avg_return / std_dev
        } else {
            0.0
        };

        // Classify skill tier
        let skill_tier = Self::classify_skill(win_rate, profit_factor, risk_adjusted_return, total_pnl);

        Self {
            address: address.to_string(),
            total_pnl,
            win_rate,
            profit_factor,
            risk_adjusted_return,
            avg_trade_duration_secs: 0.0, // Would need trade history to calculate
            directional_accuracy: win_rate, // Simplified
            skill_tier,
        }
    }

    fn classify_skill(win_rate: f64, profit_factor: f64, sharpe: f64, total_pnl: f64) -> SkillTier {
        // Elite: high win rate AND high profit factor AND positive Sharpe
        if win_rate > 0.6 && profit_factor > 2.0 && sharpe > 1.0 {
            return SkillTier::Elite;
        }

        // Skilled: consistent profits
        if win_rate > 0.5 && profit_factor > 1.5 && total_pnl > 0.0 {
            return SkillTier::Skilled;
        }

        // Average: breaking even
        if profit_factor > 0.8 && profit_factor < 1.2 {
            return SkillTier::Average;
        }

        // Market maker detection (very high win rate, small average profit)
        if win_rate > 0.7 && profit_factor > 1.0 && profit_factor < 1.3 {
            return SkillTier::MarketMaker;
        }

        // Default to unskilled
        if total_pnl < 0.0 || profit_factor < 0.8 {
            return SkillTier::Unskilled;
        }

        SkillTier::Average
    }
}

/// Aggregate skill analysis across all whales
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WhaleSkillAnalysis {
    pub timestamp_ms: i64,
    pub total_whales: usize,
    pub skilled_count: usize,
    pub unskilled_count: usize,
    pub elite_count: usize,
    pub market_maker_count: usize,
    pub avg_whale_pnl: f64,
    pub median_whale_pnl: f64,
    pub pnl_std_dev: f64,
    /// Correlation between position size and skill
    pub size_skill_correlation: f64,
}

impl WhaleSkillAnalysis {
    pub fn from_metrics(metrics: &[WhaleSkillMetrics], timestamp_ms: i64) -> Self {
        if metrics.is_empty() {
            return Self { timestamp_ms, ..Default::default() };
        }

        let mut skilled = 0;
        let mut unskilled = 0;
        let mut elite = 0;
        let mut market_makers = 0;

        let mut pnls: Vec<f64> = Vec::new();

        for m in metrics {
            pnls.push(m.total_pnl);
            match m.skill_tier {
                SkillTier::Skilled => skilled += 1,
                SkillTier::Unskilled => unskilled += 1,
                SkillTier::Elite => elite += 1,
                SkillTier::MarketMaker => market_makers += 1,
                SkillTier::Average => {}
            }
        }

        let avg_pnl = pnls.iter().sum::<f64>() / pnls.len() as f64;

        // Median
        pnls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_pnl = if pnls.len() % 2 == 0 {
            (pnls[pnls.len() / 2 - 1] + pnls[pnls.len() / 2]) / 2.0
        } else {
            pnls[pnls.len() / 2]
        };

        // Std dev
        let variance: f64 = pnls.iter()
            .map(|p| (p - avg_pnl).powi(2))
            .sum::<f64>() / pnls.len() as f64;
        let std_dev = variance.sqrt();

        Self {
            timestamp_ms,
            total_whales: metrics.len(),
            skilled_count: skilled,
            unskilled_count: unskilled,
            elite_count: elite,
            market_maker_count: market_makers,
            avg_whale_pnl: avg_pnl,
            median_whale_pnl: median_pnl,
            pnl_std_dev: std_dev,
            size_skill_correlation: 0.0, // Would need position data to calculate
        }
    }
}

/// Calculate Gini coefficient
fn calculate_gini(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    if mean <= 0.0 {
        return 0.0;
    }

    // Sort values
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate Gini using the formula: G = (2 * sum(i * x_i) - (n+1) * sum(x_i)) / (n * sum(x_i))
    let sum_x: f64 = sorted.iter().sum();
    let weighted_sum: f64 = sorted.iter()
        .enumerate()
        .map(|(i, x)| (i + 1) as f64 * x)
        .sum();

    (2.0 * weighted_sum - (n + 1.0) * sum_x) / (n * sum_x)
}

/// Whale herding metric - do whales trade together?
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WhaleHerdingMetric {
    pub timestamp_ms: i64,
    /// Percentage of whales with same directional bias
    pub directional_agreement: f64,
    /// Average correlation between whale position changes
    pub position_correlation: f64,
    /// Number of whales that changed position in same direction
    pub same_direction_count: usize,
    /// Herding classification
    pub herding_level: HerdingLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum HerdingLevel {
    /// Whales trading independently
    #[default]
    Independent,
    /// Some coordination
    Moderate,
    /// Strong coordination
    High,
    /// Nearly all whales moving together
    Extreme,
}

impl WhaleHerdingMetric {
    /// Calculate herding from position changes
    pub fn calculate(position_changes: &[(String, f64)], timestamp_ms: i64) -> Self {
        if position_changes.is_empty() {
            return Self { timestamp_ms, ..Default::default() };
        }

        let long_count = position_changes.iter()
            .filter(|(_, change)| *change > 0.0)
            .count();
        let short_count = position_changes.iter()
            .filter(|(_, change)| *change < 0.0)
            .count();

        let total_directional = long_count + short_count;
        let max_same = long_count.max(short_count);

        let directional_agreement = if total_directional > 0 {
            max_same as f64 / total_directional as f64
        } else {
            0.5
        };

        let herding_level = if directional_agreement > 0.8 {
            HerdingLevel::Extreme
        } else if directional_agreement > 0.65 {
            HerdingLevel::High
        } else if directional_agreement > 0.55 {
            HerdingLevel::Moderate
        } else {
            HerdingLevel::Independent
        };

        Self {
            timestamp_ms,
            directional_agreement,
            position_correlation: 0.0, // Would need time series to calculate
            same_direction_count: max_same,
            herding_level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gini_coefficient() {
        // Perfect equality
        let equal = vec![100.0, 100.0, 100.0, 100.0];
        assert!(calculate_gini(&equal).abs() < 0.01);

        // High inequality
        let unequal = vec![1.0, 1.0, 1.0, 97.0];
        assert!(calculate_gini(&unequal) > 0.7);
    }

    #[test]
    fn test_concentration_calculation() {
        let whales = vec![
            WhaleClassification {
                address: "0x1".to_string(),
                tier: WhaleTier::LargeWhale,
                whale_score: 90.0,
                stats: super::super::WalletStats {
                    current_position_usd: 5_000_000.0,
                    ..Default::default()
                },
                classified_at_ms: 0,
            },
            WhaleClassification {
                address: "0x2".to_string(),
                tier: WhaleTier::MediumWhale,
                whale_score: 60.0,
                stats: super::super::WalletStats {
                    current_position_usd: 2_000_000.0,
                    ..Default::default()
                },
                classified_at_ms: 0,
            },
        ];

        let concentration = WhaleConcentration::calculate(&whales, 100_000_000.0, 0);

        assert_eq!(concentration.whale_count, 2);
        assert!(concentration.top10_ratio > 0.0);
        assert!(concentration.hhi > 0.0);
    }

    #[test]
    fn test_herding_metric() {
        // All whales going long
        let changes = vec![
            ("0x1".to_string(), 100.0),
            ("0x2".to_string(), 50.0),
            ("0x3".to_string(), 200.0),
        ];

        let herding = WhaleHerdingMetric::calculate(&changes, 0);
        assert_eq!(herding.herding_level, HerdingLevel::Extreme);
        assert_eq!(herding.directional_agreement, 1.0);
    }

    #[test]
    fn test_skill_classification() {
        let metrics = WhaleSkillMetrics::calculate(
            "0x123",
            &[100.0, 50.0, -30.0, 80.0, 20.0],
            4,  // 4 wins
            1,  // 1 loss
            250.0,  // gross profit
            30.0,   // gross loss
        );

        assert!(metrics.win_rate > 0.5);
        assert!(metrics.profit_factor > 1.0);
    }
}
