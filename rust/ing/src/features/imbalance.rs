//! Order book imbalance features

use crate::state::OrderBook;

/// Imbalance features (8 features)
#[derive(Debug, Clone, Default)]
pub struct ImbalanceFeatures {
    /// Volume imbalance at level 1
    pub qty_l1: f64,
    /// Volume imbalance at levels 1-5
    pub qty_l5: f64,
    /// Volume imbalance at levels 1-10
    pub qty_l10: f64,
    /// Order count imbalance at levels 1-5
    pub orders_l5: f64,
    /// Notional imbalance at levels 1-5
    pub notional_l5: f64,
    /// Depth-weighted imbalance
    pub depth_weighted: f64,
    /// Cumulative bid pressure
    pub pressure_bid: f64,
    /// Cumulative ask pressure
    pub pressure_ask: f64,
}

impl ImbalanceFeatures {
    pub fn count() -> usize { 8 }

    pub fn names() -> Vec<&'static str> {
        vec![
            "imbalance_qty_l1",
            "imbalance_qty_l5",
            "imbalance_qty_l10",
            "imbalance_orders_l5",
            "imbalance_notional_l5",
            "imbalance_depth_weighted",
            "imbalance_pressure_bid",
            "imbalance_pressure_ask",
        ]
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.qty_l1,
            self.qty_l5,
            self.qty_l10,
            self.orders_l5,
            self.notional_l5,
            self.depth_weighted,
            self.pressure_bid,
            self.pressure_ask,
        ]
    }
}

/// Compute imbalance features from order book
pub fn compute(order_book: &OrderBook) -> ImbalanceFeatures {
    let mid = order_book.midprice().unwrap_or(0.0);

    // Volume imbalances
    let qty_l1 = order_book.volume_imbalance(1);
    let qty_l5 = order_book.volume_imbalance(5);
    let qty_l10 = order_book.volume_imbalance(10);

    // Order count imbalance
    let orders_l5 = order_book.order_imbalance(5);

    // Notional imbalance
    let bid_notional = order_book.bid_notional(5);
    let ask_notional = order_book.ask_notional(5);
    let total_notional = bid_notional + ask_notional;
    let notional_l5 = if total_notional > 0.0 {
        (bid_notional - ask_notional) / total_notional
    } else {
        0.0
    };

    // Depth-weighted imbalance
    let depth_weighted = order_book.depth_weighted_imbalance();

    // Pressure scores (cumulative depth * distance weight)
    let (pressure_bid, pressure_ask) = compute_pressure(order_book, mid);

    ImbalanceFeatures {
        qty_l1,
        qty_l5,
        qty_l10,
        orders_l5,
        notional_l5,
        depth_weighted,
        pressure_bid,
        pressure_ask,
    }
}

/// Compute bid and ask pressure scores
fn compute_pressure(order_book: &OrderBook, mid: f64) -> (f64, f64) {
    if mid <= 0.0 {
        return (0.0, 0.0);
    }

    let mut bid_pressure = 0.0;
    let mut cumulative_bid = 0.0;
    for level in order_book.bids() {
        cumulative_bid += level.size;
        let distance_bps = (mid - level.price) / mid * 10000.0;
        // Weight decreases with distance
        let weight = 1.0 / (1.0 + distance_bps / 10.0);
        bid_pressure += cumulative_bid * weight;
    }

    let mut ask_pressure = 0.0;
    let mut cumulative_ask = 0.0;
    for level in order_book.asks() {
        cumulative_ask += level.size;
        let distance_bps = (level.price - mid) / mid * 10000.0;
        let weight = 1.0 / (1.0 + distance_bps / 10.0);
        ask_pressure += cumulative_ask * weight;
    }

    // Normalize
    let max_pressure = bid_pressure.max(ask_pressure).max(1.0);
    (bid_pressure / max_pressure, ask_pressure / max_pressure)
}
