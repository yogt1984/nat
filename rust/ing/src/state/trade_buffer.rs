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
}
