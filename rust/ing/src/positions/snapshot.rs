//! Position snapshot data structures

use serde::{Deserialize, Serialize};

/// A snapshot of a wallet's position at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSnapshot {
    /// Timestamp in milliseconds
    pub timestamp_ms: i64,
    /// Wallet address
    pub wallet: String,
    /// Asset symbol (e.g., "BTC", "ETH")
    pub symbol: String,
    /// Signed position size (negative = short)
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Liquidation price (None if no position)
    pub liquidation_price: Option<f64>,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Position value in USD
    pub position_value: f64,
    /// Leverage used
    pub leverage: f64,
    /// Whether position is cross margin
    pub is_cross_margin: bool,
}

impl PositionSnapshot {
    /// Check if this is a long position
    pub fn is_long(&self) -> bool {
        self.size > 0.0
    }

    /// Check if this is a short position
    pub fn is_short(&self) -> bool {
        self.size < 0.0
    }

    /// Check if position is empty (no position)
    pub fn is_empty(&self) -> bool {
        self.size.abs() < 1e-10
    }

    /// Get absolute position size
    pub fn abs_size(&self) -> f64 {
        self.size.abs()
    }

    /// Calculate notional value (size * entry_price)
    pub fn notional_value(&self) -> f64 {
        self.size.abs() * self.entry_price
    }

    /// Calculate distance to liquidation as a percentage
    pub fn liquidation_distance_pct(&self) -> Option<f64> {
        self.liquidation_price.map(|liq| {
            if self.entry_price > 0.0 {
                ((liq - self.entry_price) / self.entry_price * 100.0).abs()
            } else {
                0.0
            }
        })
    }
}

/// Change in position between two snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionDelta {
    /// Timestamp of the new snapshot
    pub timestamp_ms: i64,
    /// Wallet address
    pub wallet: String,
    /// Asset symbol
    pub symbol: String,
    /// Previous position size
    pub prev_size: f64,
    /// Current position size
    pub curr_size: f64,
    /// Change in position size
    pub size_delta: f64,
    /// Previous entry price
    pub prev_entry_price: f64,
    /// Current entry price
    pub curr_entry_price: f64,
    /// Change in unrealized PnL
    pub pnl_delta: f64,
    /// Type of position change
    pub change_type: PositionChangeType,
}

impl PositionDelta {
    /// Create a delta from two snapshots
    pub fn from_snapshots(prev: &PositionSnapshot, curr: &PositionSnapshot) -> Self {
        let size_delta = curr.size - prev.size;
        let pnl_delta = curr.unrealized_pnl - prev.unrealized_pnl;

        let change_type = Self::classify_change(prev.size, curr.size, size_delta);

        Self {
            timestamp_ms: curr.timestamp_ms,
            wallet: curr.wallet.clone(),
            symbol: curr.symbol.clone(),
            prev_size: prev.size,
            curr_size: curr.size,
            size_delta,
            prev_entry_price: prev.entry_price,
            curr_entry_price: curr.entry_price,
            pnl_delta,
            change_type,
        }
    }

    fn classify_change(prev: f64, curr: f64, delta: f64) -> PositionChangeType {
        let prev_empty = prev.abs() < 1e-10;
        let curr_empty = curr.abs() < 1e-10;

        if prev_empty && curr_empty {
            PositionChangeType::NoChange
        } else if prev_empty && !curr_empty {
            if curr > 0.0 {
                PositionChangeType::OpenLong
            } else {
                PositionChangeType::OpenShort
            }
        } else if !prev_empty && curr_empty {
            PositionChangeType::Close
        } else if prev > 0.0 && curr > 0.0 {
            if delta > 0.0 {
                PositionChangeType::IncreaseLong
            } else if delta < 0.0 {
                PositionChangeType::DecreaseLong
            } else {
                PositionChangeType::NoChange
            }
        } else if prev < 0.0 && curr < 0.0 {
            if delta < 0.0 {
                PositionChangeType::IncreaseShort
            } else if delta > 0.0 {
                PositionChangeType::DecreaseShort
            } else {
                PositionChangeType::NoChange
            }
        } else {
            // Sign flip
            PositionChangeType::Flip
        }
    }

    /// Check if this is a significant change (> 1% of position)
    pub fn is_significant(&self, threshold_pct: f64) -> bool {
        if self.prev_size.abs() < 1e-10 {
            return self.curr_size.abs() > 1e-10;
        }
        (self.size_delta.abs() / self.prev_size.abs()) * 100.0 > threshold_pct
    }
}

/// Type of position change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionChangeType {
    NoChange,
    OpenLong,
    OpenShort,
    IncreaseLong,
    DecreaseLong,
    IncreaseShort,
    DecreaseShort,
    Close,
    Flip,
}

impl PositionChangeType {
    /// Returns true if this change increases exposure
    pub fn increases_exposure(&self) -> bool {
        matches!(self,
            PositionChangeType::OpenLong |
            PositionChangeType::OpenShort |
            PositionChangeType::IncreaseLong |
            PositionChangeType::IncreaseShort
        )
    }

    /// Returns true if this change decreases exposure
    pub fn decreases_exposure(&self) -> bool {
        matches!(self,
            PositionChangeType::DecreaseLong |
            PositionChangeType::DecreaseShort |
            PositionChangeType::Close
        )
    }
}

/// Aggregate position statistics for a wallet
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WalletPositionStats {
    /// Wallet address
    pub wallet: String,
    /// Total account value
    pub account_value: f64,
    /// Total position value (absolute)
    pub total_position_value: f64,
    /// Number of open positions
    pub position_count: usize,
    /// Largest position by notional value
    pub largest_position_symbol: Option<String>,
    /// Largest position size
    pub largest_position_value: f64,
    /// Total unrealized PnL
    pub total_unrealized_pnl: f64,
    /// Weighted average leverage
    pub weighted_avg_leverage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_delta_open_long() {
        let prev = PositionSnapshot {
            timestamp_ms: 1000,
            wallet: "0x123".to_string(),
            symbol: "BTC".to_string(),
            size: 0.0,
            entry_price: 0.0,
            liquidation_price: None,
            unrealized_pnl: 0.0,
            position_value: 0.0,
            leverage: 1.0,
            is_cross_margin: true,
        };

        let curr = PositionSnapshot {
            timestamp_ms: 2000,
            wallet: "0x123".to_string(),
            symbol: "BTC".to_string(),
            size: 1.0,
            entry_price: 50000.0,
            liquidation_price: Some(45000.0),
            unrealized_pnl: 100.0,
            position_value: 50000.0,
            leverage: 10.0,
            is_cross_margin: true,
        };

        let delta = PositionDelta::from_snapshots(&prev, &curr);
        assert_eq!(delta.change_type, PositionChangeType::OpenLong);
        assert_eq!(delta.size_delta, 1.0);
    }

    #[test]
    fn test_position_delta_increase_short() {
        let prev = PositionSnapshot {
            timestamp_ms: 1000,
            wallet: "0x123".to_string(),
            symbol: "ETH".to_string(),
            size: -5.0,
            entry_price: 3000.0,
            liquidation_price: Some(3500.0),
            unrealized_pnl: 0.0,
            position_value: 15000.0,
            leverage: 5.0,
            is_cross_margin: true,
        };

        let curr = PositionSnapshot {
            timestamp_ms: 2000,
            wallet: "0x123".to_string(),
            symbol: "ETH".to_string(),
            size: -10.0,
            entry_price: 3000.0,
            liquidation_price: Some(3300.0),
            unrealized_pnl: 500.0,
            position_value: 30000.0,
            leverage: 5.0,
            is_cross_margin: true,
        };

        let delta = PositionDelta::from_snapshots(&prev, &curr);
        assert_eq!(delta.change_type, PositionChangeType::IncreaseShort);
        assert_eq!(delta.size_delta, -5.0);
    }

    #[test]
    fn test_liquidation_distance() {
        let snapshot = PositionSnapshot {
            timestamp_ms: 1000,
            wallet: "0x123".to_string(),
            symbol: "BTC".to_string(),
            size: 1.0,
            entry_price: 50000.0,
            liquidation_price: Some(45000.0),
            unrealized_pnl: 0.0,
            position_value: 50000.0,
            leverage: 10.0,
            is_cross_margin: true,
        };

        let dist = snapshot.liquidation_distance_pct().unwrap();
        assert!((dist - 10.0).abs() < 0.01); // 10% from entry
    }
}
