//! Trade buffer for maintaining rolling window of trades

use std::collections::VecDeque;
use crate::ws::WsTrade;

/// A trade with parsed numeric values
#[derive(Debug, Clone)]
pub struct Trade {
    pub price: f64,
    pub size: f64,
    pub is_buy: bool,
    pub timestamp: u64,
    pub tid: u64,
}

impl From<WsTrade> for Trade {
    fn from(ws_trade: WsTrade) -> Self {
        Self {
            price: ws_trade.price(),
            size: ws_trade.size(),
            is_buy: ws_trade.is_buy(),
            timestamp: ws_trade.time,
            tid: ws_trade.tid,
        }
    }
}

/// Rolling buffer of trades with time-based eviction
#[derive(Debug)]
pub struct TradeBuffer {
    trades: VecDeque<Trade>,
    window_seconds: u64,
}

impl TradeBuffer {
    /// Create a new trade buffer with the given time window
    pub fn new(window_seconds: u64) -> Self {
        Self {
            trades: VecDeque::with_capacity(10_000),
            window_seconds,
        }
    }

    /// Add a trade to the buffer
    pub fn add(&mut self, ws_trade: WsTrade) {
        let trade = Trade::from(ws_trade);
        self.evict_old(trade.timestamp);
        self.trades.push_back(trade);
    }

    /// Evict trades older than the window
    fn evict_old(&mut self, current_time: u64) {
        let cutoff = current_time.saturating_sub(self.window_seconds * 1000);
        while let Some(front) = self.trades.front() {
            if front.timestamp < cutoff {
                self.trades.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get number of trades in the buffer
    pub fn len(&self) -> usize {
        self.trades.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.trades.is_empty()
    }

    /// Get trades in the last N seconds
    pub fn trades_in_window(&self, seconds: u64) -> Vec<&Trade> {
        if self.trades.is_empty() {
            return Vec::new();
        }

        let latest_time = self.trades.back().map(|t| t.timestamp).unwrap_or(0);
        let cutoff = latest_time.saturating_sub(seconds * 1000);

        self.trades.iter()
            .filter(|t| t.timestamp >= cutoff)
            .collect()
    }

    /// Count trades in the last N seconds
    pub fn count_in_window(&self, seconds: u64) -> usize {
        self.trades_in_window(seconds).len()
    }

    /// Total volume in the last N seconds
    pub fn volume_in_window(&self, seconds: u64) -> f64 {
        self.trades_in_window(seconds)
            .iter()
            .map(|t| t.size)
            .sum()
    }

    /// Total buy volume in the last N seconds
    pub fn buy_volume_in_window(&self, seconds: u64) -> f64 {
        self.trades_in_window(seconds)
            .iter()
            .filter(|t| t.is_buy)
            .map(|t| t.size)
            .sum()
    }

    /// Total sell volume in the last N seconds
    pub fn sell_volume_in_window(&self, seconds: u64) -> f64 {
        self.trades_in_window(seconds)
            .iter()
            .filter(|t| !t.is_buy)
            .map(|t| t.size)
            .sum()
    }

    /// Aggressor ratio (buy volume / total volume) in the last N seconds
    pub fn aggressor_ratio_in_window(&self, seconds: u64) -> f64 {
        let trades = self.trades_in_window(seconds);
        let total_volume: f64 = trades.iter().map(|t| t.size).sum();
        let buy_volume: f64 = trades.iter().filter(|t| t.is_buy).map(|t| t.size).sum();

        if total_volume > 0.0 {
            buy_volume / total_volume
        } else {
            0.5  // Neutral when no trades
        }
    }

    /// VWAP in the last N seconds
    pub fn vwap_in_window(&self, seconds: u64) -> Option<f64> {
        let trades = self.trades_in_window(seconds);
        if trades.is_empty() {
            return None;
        }

        let total_notional: f64 = trades.iter().map(|t| t.price * t.size).sum();
        let total_volume: f64 = trades.iter().map(|t| t.size).sum();

        if total_volume > 0.0 {
            Some(total_notional / total_volume)
        } else {
            None
        }
    }

    /// Average trade size in the last N seconds
    pub fn avg_trade_size_in_window(&self, seconds: u64) -> f64 {
        let trades = self.trades_in_window(seconds);
        if trades.is_empty() {
            return 0.0;
        }

        let total_volume: f64 = trades.iter().map(|t| t.size).sum();
        total_volume / trades.len() as f64
    }

    /// Trade intensity (trades per second) in the last N seconds
    pub fn intensity_in_window(&self, seconds: u64) -> f64 {
        let count = self.count_in_window(seconds);
        if seconds > 0 {
            count as f64 / seconds as f64
        } else {
            0.0
        }
    }

    /// Get all trade sizes in window (for entropy calculation)
    pub fn trade_sizes_in_window(&self, seconds: u64) -> Vec<f64> {
        self.trades_in_window(seconds)
            .iter()
            .map(|t| t.size)
            .collect()
    }

    /// Get all trade prices in window
    pub fn trade_prices_in_window(&self, seconds: u64) -> Vec<f64> {
        self.trades_in_window(seconds)
            .iter()
            .map(|t| t.price)
            .collect()
    }

    /// Get the most recent trade
    pub fn last_trade(&self) -> Option<&Trade> {
        self.trades.back()
    }

    /// Iterate over all trades
    pub fn iter(&self) -> impl Iterator<Item = &Trade> {
        self.trades.iter()
    }

    /// Compute tick directions (up/down/neutral) for trades in window
    /// Returns tuples of (direction, volume) where direction is:
    /// +1 for price increase, -1 for price decrease, 0 for unchanged
    pub fn tick_directions_in_window(&self, seconds: u64) -> Vec<(i8, f64)> {
        let trades = self.trades_in_window(seconds);
        if trades.len() < 2 {
            // Need at least 2 trades to compute direction
            return trades.iter()
                .map(|t| {
                    // Use is_buy as direction for single trade
                    let dir = if t.is_buy { 1i8 } else { -1i8 };
                    (dir, t.size)
                })
                .collect();
        }

        let mut result = Vec::with_capacity(trades.len());
        let mut last_price: Option<f64> = None;

        for trade in trades {
            let direction = match last_price {
                Some(prev) if trade.price > prev => 1i8,
                Some(prev) if trade.price < prev => -1i8,
                Some(_) => {
                    // Price unchanged, use aggressor side
                    if trade.is_buy { 1i8 } else { -1i8 }
                }
                None => {
                    // First trade, use aggressor side
                    if trade.is_buy { 1i8 } else { -1i8 }
                }
            };

            last_price = Some(trade.price);
            result.push((direction, trade.size));
        }

        result
    }

    /// Compute tick entropy over a time window
    /// Entropy = -Σ p * ln(p) where p is probability of each direction
    pub fn tick_entropy_in_window(&self, seconds: u64) -> Option<f64> {
        let directions = self.tick_directions_in_window(seconds);
        if directions.is_empty() {
            return None;
        }

        // Count directions: down (-1) -> 0, neutral (0) -> 1, up (+1) -> 2
        let mut counts = [0u32; 3];
        for (dir, _) in &directions {
            let idx = match *dir {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => continue,
            };
            counts[idx] += 1;
        }

        let total: u32 = counts.iter().sum();
        if total == 0 {
            return None;
        }

        // Compute Shannon entropy
        let entropy: f64 = counts.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total as f64;
                -p * p.ln()
            })
            .sum();

        Some(entropy)
    }

    /// Compute volume-weighted tick entropy over a time window
    /// Uses volume as weight for each direction
    pub fn volume_tick_entropy_in_window(&self, seconds: u64) -> Option<f64> {
        let directions = self.tick_directions_in_window(seconds);
        if directions.is_empty() {
            return None;
        }

        // Sum volumes by direction
        let mut volumes = [0.0f64; 3];
        for (dir, vol) in &directions {
            let idx = match *dir {
                -1 => 0,
                0 => 1,
                1 => 2,
                _ => continue,
            };
            volumes[idx] += vol;
        }

        let total: f64 = volumes.iter().sum();
        if total <= 0.0 {
            return None;
        }

        // Compute Shannon entropy weighted by volume
        let entropy: f64 = volumes.iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| {
                let p = v / total;
                -p * p.ln()
            })
            .sum();

        Some(entropy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_buffer() -> TradeBuffer {
        let mut buffer = TradeBuffer::new(60);

        // Add some test trades
        let base_time = 1704067200000u64;

        buffer.trades.push_back(Trade {
            price: 50000.0,
            size: 1.0,
            is_buy: true,
            timestamp: base_time,
            tid: 1,
        });

        buffer.trades.push_back(Trade {
            price: 50001.0,
            size: 2.0,
            is_buy: false,
            timestamp: base_time + 1000,
            tid: 2,
        });

        buffer.trades.push_back(Trade {
            price: 50002.0,
            size: 1.5,
            is_buy: true,
            timestamp: base_time + 2000,
            tid: 3,
        });

        buffer
    }

    #[test]
    fn test_volume() {
        let buffer = create_test_buffer();
        let volume = buffer.volume_in_window(60);
        assert!((volume - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_aggressor_ratio() {
        let buffer = create_test_buffer();
        // Buy: 1.0 + 1.5 = 2.5, Total: 4.5
        let ratio = buffer.aggressor_ratio_in_window(60);
        assert!((ratio - (2.5 / 4.5)).abs() < 1e-10);
    }

    #[test]
    fn test_vwap() {
        let buffer = create_test_buffer();
        // (50000*1 + 50001*2 + 50002*1.5) / 4.5
        let expected = (50000.0 * 1.0 + 50001.0 * 2.0 + 50002.0 * 1.5) / 4.5;
        let vwap = buffer.vwap_in_window(60).unwrap();
        assert!((vwap - expected).abs() < 1e-6);
    }

    #[test]
    fn test_tick_directions_empty() {
        let buffer = TradeBuffer::new(60);
        let directions = buffer.tick_directions_in_window(60);
        assert!(directions.is_empty());
    }

    #[test]
    fn test_tick_directions_uptrend() {
        let mut buffer = TradeBuffer::new(60);
        let base_time = 1704067200000u64;

        // Add ascending price trades (all up ticks)
        buffer.trades.push_back(Trade {
            price: 50000.0,
            size: 1.0,
            is_buy: true,
            timestamp: base_time,
            tid: 1,
        });
        buffer.trades.push_back(Trade {
            price: 50001.0,
            size: 1.0,
            is_buy: true,
            timestamp: base_time + 1000,
            tid: 2,
        });
        buffer.trades.push_back(Trade {
            price: 50002.0,
            size: 1.0,
            is_buy: true,
            timestamp: base_time + 2000,
            tid: 3,
        });

        let directions = buffer.tick_directions_in_window(60);
        assert_eq!(directions.len(), 3);

        // First trade uses is_buy (+1), second and third are upticks (+1)
        assert_eq!(directions[0].0, 1);
        assert_eq!(directions[1].0, 1);
        assert_eq!(directions[2].0, 1);
    }

    #[test]
    fn test_tick_directions_downtrend() {
        let mut buffer = TradeBuffer::new(60);
        let base_time = 1704067200000u64;

        // Add descending price trades (all down ticks)
        buffer.trades.push_back(Trade {
            price: 50002.0,
            size: 1.0,
            is_buy: false,
            timestamp: base_time,
            tid: 1,
        });
        buffer.trades.push_back(Trade {
            price: 50001.0,
            size: 1.0,
            is_buy: false,
            timestamp: base_time + 1000,
            tid: 2,
        });
        buffer.trades.push_back(Trade {
            price: 50000.0,
            size: 1.0,
            is_buy: false,
            timestamp: base_time + 2000,
            tid: 3,
        });

        let directions = buffer.tick_directions_in_window(60);
        assert_eq!(directions.len(), 3);

        // First trade uses is_buy (-1), second and third are downticks (-1)
        assert_eq!(directions[0].0, -1);
        assert_eq!(directions[1].0, -1);
        assert_eq!(directions[2].0, -1);
    }

    #[test]
    fn test_tick_entropy_uniform() {
        let mut buffer = TradeBuffer::new(60);
        let base_time = 1704067200000u64;

        // Add equal up and down ticks
        for i in 0..10 {
            buffer.trades.push_back(Trade {
                price: 50000.0 + (i % 2) as f64, // Alternates: 50000, 50001, 50000, 50001...
                size: 1.0,
                is_buy: i % 2 == 0,
                timestamp: base_time + i * 1000,
                tid: i,
            });
        }

        let entropy = buffer.tick_entropy_in_window(60);
        assert!(entropy.is_some());

        // For equal up/down distribution, entropy should be close to ln(2) ≈ 0.693
        let e = entropy.unwrap();
        assert!(e > 0.5, "Entropy should be high for uniform distribution: {}", e);
    }

    #[test]
    fn test_tick_entropy_single_direction() {
        let mut buffer = TradeBuffer::new(60);
        let base_time = 1704067200000u64;

        // Add only upward ticks
        for i in 0..10 {
            buffer.trades.push_back(Trade {
                price: 50000.0 + i as f64,
                size: 1.0,
                is_buy: true,
                timestamp: base_time + i * 1000,
                tid: i,
            });
        }

        let entropy = buffer.tick_entropy_in_window(60);
        assert!(entropy.is_some());

        // For single direction, entropy should be 0
        let e = entropy.unwrap();
        assert!(e < 0.01, "Entropy should be ~0 for single direction: {}", e);
    }

    #[test]
    fn test_volume_tick_entropy() {
        let mut buffer = TradeBuffer::new(60);
        let base_time = 1704067200000u64;

        // Add trades with different volumes
        buffer.trades.push_back(Trade {
            price: 50000.0,
            size: 10.0, // Large up
            is_buy: true,
            timestamp: base_time,
            tid: 1,
        });
        buffer.trades.push_back(Trade {
            price: 50001.0,
            size: 1.0, // Small up (price increased)
            is_buy: true,
            timestamp: base_time + 1000,
            tid: 2,
        });
        buffer.trades.push_back(Trade {
            price: 50000.0,
            size: 1.0, // Small down
            is_buy: false,
            timestamp: base_time + 2000,
            tid: 3,
        });

        let entropy = buffer.volume_tick_entropy_in_window(60);
        assert!(entropy.is_some());

        // With volume weighting: up=11.0, down=1.0
        // Volume-weighted entropy should be lower than uniform tick entropy
        let e = entropy.unwrap();
        assert!(e > 0.0, "Volume entropy should be positive: {}", e);
    }
}
