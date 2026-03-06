//! Real-time Feature Display
//!
//! Streams live features from Hyperliquid to the terminal.
//! No file output - purely for monitoring and validation.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use std::collections::VecDeque;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use tokio_tungstenite::{connect_async, tungstenite::Message};

const WS_URL: &str = "wss://api.hyperliquid.xyz/ws";
const FEATURE_INTERVAL_MS: u64 = 1000; // Emit features every 1 second

/// Trade sample
#[derive(Debug, Clone)]
struct Trade {
    timestamp_ms: u64,
    price: f64,
    size: f64,
    is_buy: bool,
}

/// Rolling trade buffer for feature computation
struct TradeBuffer {
    trades: VecDeque<Trade>,
    last_price: Option<f64>,
}

impl TradeBuffer {
    fn new() -> Self {
        Self {
            trades: VecDeque::with_capacity(50_000),
            last_price: None,
        }
    }

    fn add(&mut self, price: f64, size: f64, is_buy: bool, timestamp_ms: u64) {
        self.trades.push_back(Trade {
            timestamp_ms,
            price,
            size,
            is_buy,
        });
        self.last_price = Some(price);

        // Prune old trades (keep last 15 minutes)
        let cutoff = timestamp_ms.saturating_sub(900_000);
        while let Some(front) = self.trades.front() {
            if front.timestamp_ms < cutoff {
                self.trades.pop_front();
            } else {
                break;
            }
        }
    }

    fn count_in_window(&self, window_secs: u64) -> usize {
        if self.trades.is_empty() {
            return 0;
        }
        let latest = self.trades.back().map(|t| t.timestamp_ms).unwrap_or(0);
        let cutoff = latest.saturating_sub(window_secs * 1000);
        self.trades.iter().filter(|t| t.timestamp_ms >= cutoff).count()
    }

    fn volume_in_window(&self, window_secs: u64) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        let latest = self.trades.back().map(|t| t.timestamp_ms).unwrap_or(0);
        let cutoff = latest.saturating_sub(window_secs * 1000);
        self.trades.iter()
            .filter(|t| t.timestamp_ms >= cutoff)
            .map(|t| t.size)
            .sum()
    }

    fn buy_volume_in_window(&self, window_secs: u64) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }
        let latest = self.trades.back().map(|t| t.timestamp_ms).unwrap_or(0);
        let cutoff = latest.saturating_sub(window_secs * 1000);
        self.trades.iter()
            .filter(|t| t.timestamp_ms >= cutoff && t.is_buy)
            .map(|t| t.size)
            .sum()
    }

    fn vwap_in_window(&self, window_secs: u64) -> Option<f64> {
        if self.trades.is_empty() {
            return None;
        }
        let latest = self.trades.back().map(|t| t.timestamp_ms).unwrap_or(0);
        let cutoff = latest.saturating_sub(window_secs * 1000);
        let trades: Vec<_> = self.trades.iter()
            .filter(|t| t.timestamp_ms >= cutoff)
            .collect();

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

    fn tick_entropy(&self, window_secs: u64) -> Option<f64> {
        if self.trades.is_empty() {
            return None;
        }
        let latest = self.trades.back().map(|t| t.timestamp_ms).unwrap_or(0);
        let cutoff = latest.saturating_sub(window_secs * 1000);
        let trades: Vec<_> = self.trades.iter()
            .filter(|t| t.timestamp_ms >= cutoff)
            .collect();

        if trades.len() < 2 {
            return None;
        }

        // Compute directions
        let mut counts = [0u32; 3]; // down, neutral, up
        let mut last_price: Option<f64> = None;

        for trade in &trades {
            let direction = match last_price {
                Some(prev) if trade.price > prev => 2, // up
                Some(prev) if trade.price < prev => 0, // down
                Some(_) => if trade.is_buy { 2 } else { 0 }, // neutral uses side
                None => if trade.is_buy { 2 } else { 0 },
            };
            counts[direction] += 1;
            last_price = Some(trade.price);
        }

        let total: u32 = counts.iter().sum();
        if total == 0 {
            return None;
        }

        let entropy: f64 = counts.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total as f64;
                -p * p.ln()
            })
            .sum();

        Some(entropy)
    }

    fn aggressor_ratio(&self, window_secs: u64) -> f64 {
        let total = self.volume_in_window(window_secs);
        let buy = self.buy_volume_in_window(window_secs);
        if total > 0.0 {
            buy / total
        } else {
            0.5
        }
    }

    fn trade_intensity(&self, window_secs: u64) -> f64 {
        let count = self.count_in_window(window_secs);
        if window_secs > 0 {
            count as f64 / window_secs as f64
        } else {
            0.0
        }
    }
}

/// Order book state
struct OrderBook {
    best_bid: f64,
    best_ask: f64,
    bid_depth: f64,
    ask_depth: f64,
}

impl OrderBook {
    fn new() -> Self {
        Self {
            best_bid: 0.0,
            best_ask: 0.0,
            bid_depth: 0.0,
            ask_depth: 0.0,
        }
    }

    fn update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        if let Some((price, _)) = bids.first() {
            self.best_bid = *price;
        }
        if let Some((price, _)) = asks.first() {
            self.best_ask = *price;
        }
        self.bid_depth = bids.iter().map(|(_, sz)| sz).sum();
        self.ask_depth = asks.iter().map(|(_, sz)| sz).sum();
    }

    fn midprice(&self) -> f64 {
        if self.best_bid > 0.0 && self.best_ask > 0.0 {
            (self.best_bid + self.best_ask) / 2.0
        } else {
            0.0
        }
    }

    fn spread(&self) -> f64 {
        if self.best_bid > 0.0 && self.best_ask > 0.0 {
            self.best_ask - self.best_bid
        } else {
            0.0
        }
    }

    fn spread_bps(&self) -> f64 {
        let mid = self.midprice();
        if mid > 0.0 {
            (self.spread() / mid) * 10_000.0
        } else {
            0.0
        }
    }

    fn imbalance(&self) -> f64 {
        let total = self.bid_depth + self.ask_depth;
        if total > 0.0 {
            (self.bid_depth - self.ask_depth) / total
        } else {
            0.0
        }
    }
}

/// Feature snapshot for display
struct FeatureSnapshot {
    timestamp: DateTime<Utc>,
    symbol: String,
    // Price features
    midprice: f64,
    spread_bps: f64,
    // Volume features
    trade_count_1s: usize,
    trade_count_5s: usize,
    trade_count_30s: usize,
    volume_1s: f64,
    volume_5s: f64,
    volume_30s: f64,
    // Flow features
    aggressor_ratio_5s: f64,
    aggressor_ratio_30s: f64,
    imbalance: f64,
    // Entropy features
    tick_entropy_1s: f64,
    tick_entropy_5s: f64,
    tick_entropy_30s: f64,
    tick_entropy_1m: f64,
    // Derived
    trade_intensity_5s: f64,
    vwap_30s: f64,
}

impl FeatureSnapshot {
    fn print_header() {
        println!("\n{}", "=".repeat(120));
        println!("NAT Feature Ingestor - Real-time Display (Ctrl+C to stop)");
        println!("{}", "=".repeat(120));
        println!();
    }

    fn print(&self, update_count: u64) {
        // Clear screen and move cursor to top
        print!("\x1B[2J\x1B[1;1H");
        io::stdout().flush().unwrap();

        println!("{}",  "=".repeat(100));
        println!("  NAT Real-time Features | {} | Update #{}",
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC"), update_count);
        println!("{}", "=".repeat(100));
        println!();

        // Price section
        println!("  PRICE");
        println!("  {:<20} {:>15.2}  {:<20} {:>12.2} bps",
            "Midprice:", self.midprice,
            "Spread:", self.spread_bps);
        println!();

        // Trade activity section
        println!("  TRADE ACTIVITY");
        println!("  {:<20} {:>15}  {:<20} {:>15}  {:<20} {:>15}",
            "Trades (1s):", self.trade_count_1s,
            "Trades (5s):", self.trade_count_5s,
            "Trades (30s):", self.trade_count_30s);
        println!("  {:<20} {:>15.4}  {:<20} {:>15.4}  {:<20} {:>15.4}",
            "Volume (1s):", self.volume_1s,
            "Volume (5s):", self.volume_5s,
            "Volume (30s):", self.volume_30s);
        println!("  {:<20} {:>15.2}/s",
            "Trade intensity:", self.trade_intensity_5s);
        println!();

        // Flow section
        println!("  ORDER FLOW");
        println!("  {:<20} {:>15.4}  {:<20} {:>15.4}  {:<20} {:>15.4}",
            "Aggressor (5s):", self.aggressor_ratio_5s,
            "Aggressor (30s):", self.aggressor_ratio_30s,
            "Book imbalance:", self.imbalance);
        println!("  {:<20} {:>15.2}",
            "VWAP (30s):", self.vwap_30s);
        println!();

        // Entropy section
        println!("  TICK ENTROPY (regime detection)");
        println!("  {:<20} {:>15.4}  {:<20} {:>15.4}  {:<20} {:>15.4}  {:<20} {:>15.4}",
            "Entropy (1s):", self.tick_entropy_1s,
            "Entropy (5s):", self.tick_entropy_5s,
            "Entropy (30s):", self.tick_entropy_30s,
            "Entropy (1m):", self.tick_entropy_1m);

        // Entropy interpretation
        let regime = if self.tick_entropy_30s < 0.3 {
            "TRENDING (low entropy)"
        } else if self.tick_entropy_30s > 0.6 {
            "RANDOM (high entropy)"
        } else {
            "TRANSITIONAL"
        };
        println!("  {:<20} {:>15}", "Regime:", regime);
        println!();

        println!("{}", "-".repeat(100));
        println!("  Symbol: {} | Features: 24 entropy + flow + volume | Press Ctrl+C to stop", self.symbol);
        println!("{}", "=".repeat(100));
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let symbol = std::env::args().nth(1).unwrap_or_else(|| "BTC".to_string());

    println!("Connecting to Hyperliquid WebSocket...");

    let (mut ws_stream, _) = connect_async(WS_URL)
        .await
        .context("Failed to connect to WebSocket")?;

    println!("Connected. Subscribing to {} data...", symbol);

    // Subscribe to trades
    let trade_sub = serde_json::json!({
        "method": "subscribe",
        "subscription": {
            "type": "trades",
            "coin": &symbol
        }
    });
    ws_stream.send(Message::Text(serde_json::to_string(&trade_sub)?)).await?;

    // Subscribe to order book
    let book_sub = serde_json::json!({
        "method": "subscribe",
        "subscription": {
            "type": "l2Book",
            "coin": &symbol
        }
    });
    ws_stream.send(Message::Text(serde_json::to_string(&book_sub)?)).await?;

    println!("Subscribed. Starting feature display...\n");

    let mut trade_buffer = TradeBuffer::new();
    let mut order_book = OrderBook::new();
    let mut last_feature_emit = Instant::now();
    let mut update_count = 0u64;
    let feature_interval = Duration::from_millis(FEATURE_INTERVAL_MS);

    FeatureSnapshot::print_header();

    loop {
        tokio::select! {
            msg = ws_stream.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(response) = serde_json::from_str::<serde_json::Value>(&text) {
                            let channel = response["channel"].as_str().unwrap_or("");

                            match channel {
                                "trades" => {
                                    if let Some(trades) = response["data"].as_array() {
                                        for trade in trades {
                                            let price = trade["px"].as_str()
                                                .and_then(|s| s.parse::<f64>().ok())
                                                .unwrap_or(0.0);
                                            let size = trade["sz"].as_str()
                                                .and_then(|s| s.parse::<f64>().ok())
                                                .unwrap_or(0.0);
                                            let side = trade["side"].as_str().unwrap_or("B");
                                            let is_buy = side == "B";
                                            let timestamp = trade["time"].as_u64().unwrap_or(0);

                                            trade_buffer.add(price, size, is_buy, timestamp);
                                        }
                                    }
                                }
                                "l2Book" => {
                                    if let Some(data) = response["data"].as_object() {
                                        let mut bids: Vec<(f64, f64)> = Vec::new();
                                        let mut asks: Vec<(f64, f64)> = Vec::new();

                                        if let Some(levels) = data["levels"].as_array() {
                                            if levels.len() >= 2 {
                                                // Parse bids
                                                if let Some(bid_levels) = levels[0].as_array() {
                                                    for level in bid_levels {
                                                        let px = level["px"].as_str()
                                                            .and_then(|s| s.parse::<f64>().ok())
                                                            .unwrap_or(0.0);
                                                        let sz = level["sz"].as_str()
                                                            .and_then(|s| s.parse::<f64>().ok())
                                                            .unwrap_or(0.0);
                                                        if px > 0.0 {
                                                            bids.push((px, sz));
                                                        }
                                                    }
                                                }
                                                // Parse asks
                                                if let Some(ask_levels) = levels[1].as_array() {
                                                    for level in ask_levels {
                                                        let px = level["px"].as_str()
                                                            .and_then(|s| s.parse::<f64>().ok())
                                                            .unwrap_or(0.0);
                                                        let sz = level["sz"].as_str()
                                                            .and_then(|s| s.parse::<f64>().ok())
                                                            .unwrap_or(0.0);
                                                        if px > 0.0 {
                                                            asks.push((px, sz));
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        order_book.update(&bids, &asks);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    Some(Ok(Message::Ping(data))) => {
                        ws_stream.send(Message::Pong(data)).await?;
                    }
                    Some(Ok(Message::Close(_))) => {
                        println!("\nWebSocket closed by server.");
                        break;
                    }
                    Some(Err(e)) => {
                        eprintln!("\nWebSocket error: {}", e);
                        break;
                    }
                    None => {
                        println!("\nWebSocket stream ended.");
                        break;
                    }
                    _ => {}
                }
            }

            _ = tokio::time::sleep(Duration::from_millis(50)) => {
                // Check if it's time to emit features
                if last_feature_emit.elapsed() >= feature_interval {
                    update_count += 1;

                    let snapshot = FeatureSnapshot {
                        timestamp: Utc::now(),
                        symbol: symbol.clone(),
                        midprice: order_book.midprice(),
                        spread_bps: order_book.spread_bps(),
                        trade_count_1s: trade_buffer.count_in_window(1),
                        trade_count_5s: trade_buffer.count_in_window(5),
                        trade_count_30s: trade_buffer.count_in_window(30),
                        volume_1s: trade_buffer.volume_in_window(1),
                        volume_5s: trade_buffer.volume_in_window(5),
                        volume_30s: trade_buffer.volume_in_window(30),
                        aggressor_ratio_5s: trade_buffer.aggressor_ratio(5),
                        aggressor_ratio_30s: trade_buffer.aggressor_ratio(30),
                        imbalance: order_book.imbalance(),
                        tick_entropy_1s: trade_buffer.tick_entropy(1).unwrap_or(0.0),
                        tick_entropy_5s: trade_buffer.tick_entropy(5).unwrap_or(0.0),
                        tick_entropy_30s: trade_buffer.tick_entropy(30).unwrap_or(0.0),
                        tick_entropy_1m: trade_buffer.tick_entropy(60).unwrap_or(0.0),
                        trade_intensity_5s: trade_buffer.trade_intensity(5),
                        vwap_30s: trade_buffer.vwap_in_window(30).unwrap_or(0.0),
                    };

                    snapshot.print(update_count);
                    last_feature_emit = Instant::now();
                }
            }
        }
    }

    Ok(())
}
