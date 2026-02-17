//! Raw order book features

use crate::state::OrderBook;

/// Raw order book features (10 features)
#[derive(Debug, Clone, Default)]
pub struct RawFeatures {
    /// Midprice: (best_bid + best_ask) / 2
    pub midprice: f64,
    /// Spread: best_ask - best_bid
    pub spread: f64,
    /// Spread in basis points
    pub spread_bps: f64,
    /// Microprice: volume-weighted mid
    pub microprice: f64,
    /// Total bid depth, levels 1-5
    pub bid_depth_5: f64,
    /// Total ask depth, levels 1-5
    pub ask_depth_5: f64,
    /// Total bid depth, levels 1-10
    pub bid_depth_10: f64,
    /// Total ask depth, levels 1-10
    pub ask_depth_10: f64,
    /// Total bid order count, levels 1-5
    pub bid_orders_5: u32,
    /// Total ask order count, levels 1-5
    pub ask_orders_5: u32,
}

impl RawFeatures {
    pub fn count() -> usize { 10 }

    pub fn names() -> Vec<&'static str> {
        vec![
            "raw_midprice",
            "raw_spread",
            "raw_spread_bps",
            "raw_microprice",
            "raw_bid_depth_5",
            "raw_ask_depth_5",
            "raw_bid_depth_10",
            "raw_ask_depth_10",
            "raw_bid_orders_5",
            "raw_ask_orders_5",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.midprice,
            self.spread,
            self.spread_bps,
            self.microprice,
            self.bid_depth_5,
            self.ask_depth_5,
            self.bid_depth_10,
            self.ask_depth_10,
            self.bid_orders_5 as f64,
            self.ask_orders_5 as f64,
        ]
    }
}

/// Compute raw features from order book
pub fn compute(order_book: &OrderBook) -> RawFeatures {
    RawFeatures {
        midprice: order_book.midprice().unwrap_or(0.0),
        spread: order_book.spread().unwrap_or(0.0),
        spread_bps: order_book.spread_bps().unwrap_or(0.0),
        microprice: order_book.microprice().unwrap_or(0.0),
        bid_depth_5: order_book.bid_depth(5),
        ask_depth_5: order_book.ask_depth(5),
        bid_depth_10: order_book.bid_depth(10),
        ask_depth_10: order_book.ask_depth(10),
        bid_orders_5: order_book.bid_order_count(5),
        ask_orders_5: order_book.ask_order_count(5),
    }
}
