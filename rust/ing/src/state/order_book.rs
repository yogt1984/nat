//! Order book state management

use crate::ws::{WsBook, WsLevel};

/// A single price level in the order book
#[derive(Debug, Clone, Default)]
pub struct Level {
    pub price: f64,
    pub size: f64,
    pub order_count: u32,
}

impl Level {
    pub fn notional(&self) -> f64 {
        self.price * self.size
    }
}

/// Order book state
#[derive(Debug)]
pub struct OrderBook {
    max_levels: usize,
    bids: Vec<Level>,
    asks: Vec<Level>,
    last_update_time: u64,
}

impl OrderBook {
    /// Create a new order book
    pub fn new(max_levels: usize) -> Self {
        Self {
            max_levels,
            bids: Vec::with_capacity(max_levels),
            asks: Vec::with_capacity(max_levels),
            last_update_time: 0,
        }
    }

    /// Update the order book from a WebSocket message
    pub fn update(&mut self, book: &WsBook) {
        self.bids = book.levels.0.iter()
            .take(self.max_levels)
            .map(|l| Level {
                price: l.price(),
                size: l.size(),
                order_count: l.n,
            })
            .collect();

        self.asks = book.levels.1.iter()
            .take(self.max_levels)
            .map(|l| Level {
                price: l.price(),
                size: l.size(),
                order_count: l.n,
            })
            .collect();

        self.last_update_time = book.time;
    }

    /// Get the best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get the best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Get the best bid level
    pub fn best_bid_level(&self) -> Option<&Level> {
        self.bids.first()
    }

    /// Get the best ask level
    pub fn best_ask_level(&self) -> Option<&Level> {
        self.asks.first()
    }

    /// Calculate midprice
    pub fn midprice(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Calculate spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.midprice()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid * 10000.0),
            _ => None,
        }
    }

    /// Calculate microprice (volume-weighted mid)
    pub fn microprice(&self) -> Option<f64> {
        match (self.best_bid_level(), self.best_ask_level()) {
            (Some(bid), Some(ask)) => {
                let total_size = bid.size + ask.size;
                if total_size > 0.0 {
                    Some((bid.price * ask.size + ask.price * bid.size) / total_size)
                } else {
                    self.midprice()
                }
            }
            _ => None,
        }
    }

    /// Get total bid depth up to N levels
    pub fn bid_depth(&self, levels: usize) -> f64 {
        self.bids.iter().take(levels).map(|l| l.size).sum()
    }

    /// Get total ask depth up to N levels
    pub fn ask_depth(&self, levels: usize) -> f64 {
        self.asks.iter().take(levels).map(|l| l.size).sum()
    }

    /// Get total bid notional up to N levels
    pub fn bid_notional(&self, levels: usize) -> f64 {
        self.bids.iter().take(levels).map(|l| l.notional()).sum()
    }

    /// Get total ask notional up to N levels
    pub fn ask_notional(&self, levels: usize) -> f64 {
        self.asks.iter().take(levels).map(|l| l.notional()).sum()
    }

    /// Get total bid order count up to N levels
    pub fn bid_order_count(&self, levels: usize) -> u32 {
        self.bids.iter().take(levels).map(|l| l.order_count).sum()
    }

    /// Get total ask order count up to N levels
    pub fn ask_order_count(&self, levels: usize) -> u32 {
        self.asks.iter().take(levels).map(|l| l.order_count).sum()
    }

    /// Get bid levels
    pub fn bids(&self) -> &[Level] {
        &self.bids
    }

    /// Get ask levels
    pub fn asks(&self) -> &[Level] {
        &self.asks
    }

    /// Get last update timestamp
    pub fn last_update_time(&self) -> u64 {
        self.last_update_time
    }

    /// Check if the order book has data
    pub fn is_valid(&self) -> bool {
        !self.bids.is_empty() && !self.asks.is_empty()
    }

    /// Calculate volume imbalance at level
    pub fn volume_imbalance(&self, levels: usize) -> f64 {
        let bid_depth = self.bid_depth(levels);
        let ask_depth = self.ask_depth(levels);
        let total = bid_depth + ask_depth;
        if total > 0.0 {
            (bid_depth - ask_depth) / total
        } else {
            0.0
        }
    }

    /// Calculate order count imbalance at level
    pub fn order_imbalance(&self, levels: usize) -> f64 {
        let bid_orders = self.bid_order_count(levels) as f64;
        let ask_orders = self.ask_order_count(levels) as f64;
        let total = bid_orders + ask_orders;
        if total > 0.0 {
            (bid_orders - ask_orders) / total
        } else {
            0.0
        }
    }

    /// Calculate depth-weighted imbalance
    pub fn depth_weighted_imbalance(&self) -> f64 {
        let mid = match self.midprice() {
            Some(m) => m,
            None => return 0.0,
        };

        let mut bid_pressure = 0.0;
        let mut ask_pressure = 0.0;

        for (i, level) in self.bids.iter().enumerate() {
            let distance = (mid - level.price).abs() / mid;
            let weight = 1.0 / (1.0 + distance * 100.0);  // Decay with distance
            bid_pressure += level.size * weight;
        }

        for (i, level) in self.asks.iter().enumerate() {
            let distance = (level.price - mid).abs() / mid;
            let weight = 1.0 / (1.0 + distance * 100.0);
            ask_pressure += level.size * weight;
        }

        let total = bid_pressure + ask_pressure;
        if total > 0.0 {
            (bid_pressure - ask_pressure) / total
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_book() -> OrderBook {
        let mut book = OrderBook::new(10);

        // Simulate an update
        let ws_book = WsBook {
            coin: "BTC".to_string(),
            levels: (
                vec![
                    WsLevel { px: "50000.0".to_string(), sz: "1.0".to_string(), n: 3 },
                    WsLevel { px: "49999.0".to_string(), sz: "2.0".to_string(), n: 5 },
                ],
                vec![
                    WsLevel { px: "50001.0".to_string(), sz: "1.5".to_string(), n: 2 },
                    WsLevel { px: "50002.0".to_string(), sz: "3.0".to_string(), n: 4 },
                ],
            ),
            time: 1704067200000,
        };

        book.update(&ws_book);
        book
    }

    #[test]
    fn test_midprice() {
        let book = create_test_book();
        assert_eq!(book.midprice(), Some(50000.5));
    }

    #[test]
    fn test_spread() {
        let book = create_test_book();
        assert_eq!(book.spread(), Some(1.0));
    }

    #[test]
    fn test_volume_imbalance() {
        let book = create_test_book();
        // bid: 1.0 + 2.0 = 3.0, ask: 1.5 + 3.0 = 4.5
        // imbalance = (3.0 - 4.5) / 7.5 = -0.2
        let imb = book.volume_imbalance(2);
        assert!((imb - (-0.2)).abs() < 1e-10);
    }
}
