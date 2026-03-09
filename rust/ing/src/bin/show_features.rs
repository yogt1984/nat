//! Real-time Feature Display
//!
//! Streams live features from Hyperliquid to the terminal.
//! No file output - purely for monitoring and validation.
//!
//! Usage: show_features [SYMBOL] [FREQ_HZ]
//!   SYMBOL: Trading pair (default: BTC)
//!   FREQ_HZ: Update frequency in Hz (default: 1, max: 50)

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use std::collections::VecDeque;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use tokio_tungstenite::{connect_async, tungstenite::Message};

const WS_URL: &str = "wss://api.hyperliquid.xyz/ws";
const DEFAULT_FREQ_HZ: u64 = 1;
const MAX_FREQ_HZ: u64 = 50;

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

/// Price buffer for trend computation
struct PriceBuffer {
    prices: VecDeque<f64>,
    capacity: usize,
}

impl PriceBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            prices: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, price: f64) {
        if self.prices.len() >= self.capacity {
            self.prices.pop_front();
        }
        self.prices.push_back(price);
    }

    fn len(&self) -> usize {
        self.prices.len()
    }

    fn to_vec(&self) -> Vec<f64> {
        self.prices.iter().cloned().collect()
    }

    /// Compute momentum (linear regression slope) for last n prices
    fn momentum(&self, window: usize) -> f64 {
        let n = self.prices.len().min(window);
        if n < 2 {
            return 0.0;
        }

        let start = self.prices.len().saturating_sub(window);
        let prices: Vec<f64> = self.prices.iter().skip(start).cloned().collect();

        let n_f = prices.len() as f64;
        let sum_x: f64 = (prices.len() - 1) as f64 * n_f / 2.0;
        let sum_x2: f64 = (prices.len() - 1) as f64 * n_f * (2 * prices.len() - 1) as f64 / 6.0;
        let sum_y: f64 = prices.iter().sum();
        let sum_xy: f64 = prices.iter().enumerate().map(|(i, &p)| i as f64 * p).sum();

        let denominator = n_f * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (n_f * sum_xy - sum_x * sum_y) / denominator
    }

    /// Compute monotonicity (fraction in dominant direction)
    fn monotonicity(&self, window: usize) -> f64 {
        let n = self.prices.len().min(window);
        if n < 2 {
            return 0.5;
        }

        let start = self.prices.len().saturating_sub(window);
        let prices: Vec<f64> = self.prices.iter().skip(start).cloned().collect();

        let mut up = 0;
        let mut down = 0;
        for i in 1..prices.len() {
            if prices[i] > prices[i - 1] {
                up += 1;
            } else if prices[i] < prices[i - 1] {
                down += 1;
            }
        }

        let total = up + down;
        if total == 0 {
            0.5
        } else {
            up.max(down) as f64 / total as f64
        }
    }

    /// Compute simple EMAs
    fn emas(&self, short_period: usize, long_period: usize) -> (f64, f64) {
        if self.prices.is_empty() {
            return (0.0, 0.0);
        }

        let alpha_short = 2.0 / (short_period as f64 + 1.0);
        let alpha_long = 2.0 / (long_period as f64 + 1.0);

        let mut ema_short = *self.prices.front().unwrap();
        let mut ema_long = ema_short;

        for &price in self.prices.iter().skip(1) {
            ema_short = alpha_short * price + (1.0 - alpha_short) * ema_short;
            ema_long = alpha_long * price + (1.0 - alpha_long) * ema_long;
        }

        (ema_short, ema_long)
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
    // Trend features
    momentum_60: f64,
    momentum_300: f64,
    monotonicity_60: f64,
    monotonicity_300: f64,
    ma_crossover: f64,
    price_samples: usize,
}

impl FeatureSnapshot {
    fn print_header(freq_hz: u64) {
        println!("\n{}", "=".repeat(120));
        println!("NAT Feature Ingestor - Real-time Display @ {} Hz (Ctrl+C to stop)", freq_hz);
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

        // Trend section
        println!("  TREND FEATURES");
        println!("  {:<20} {:>15.6}  {:<20} {:>15.6}",
            "Momentum (60):", self.momentum_60,
            "Momentum (300):", self.momentum_300);
        println!("  {:<20} {:>15.4}  {:<20} {:>15.4}",
            "Monotonicity (60):", self.monotonicity_60,
            "Monotonicity (300):", self.monotonicity_300);
        println!("  {:<20} {:>15.4}  {:<20} {:>15}",
            "MA Crossover:", self.ma_crossover,
            "Price samples:", self.price_samples);

        // Trend interpretation
        let trend = if self.momentum_60 > 0.0 && self.monotonicity_60 > 0.7 {
            "STRONG UPTREND"
        } else if self.momentum_60 < 0.0 && self.monotonicity_60 > 0.7 {
            "STRONG DOWNTREND"
        } else if self.monotonicity_60 < 0.55 {
            "CHOPPY / RANGING"
        } else if self.momentum_60 > 0.0 {
            "WEAK UPTREND"
        } else {
            "WEAK DOWNTREND"
        };
        println!("  {:<20} {:>15}", "Trend:", trend);
        println!();

        println!("{}", "-".repeat(100));
        println!("  Symbol: {} | Features: 77 (entropy + flow + volume + trend) | Press Ctrl+C to stop", self.symbol);
        println!("{}", "=".repeat(100));
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let symbol = args.get(1)
        .filter(|s| !s.parse::<u64>().is_ok()) // Skip if it's a number (freq)
        .cloned()
        .unwrap_or_else(|| "BTC".to_string());

    let freq_hz: u64 = args.iter()
        .skip(1)
        .find_map(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_FREQ_HZ)
        .min(MAX_FREQ_HZ)
        .max(1);

    let feature_interval_ms = 1000 / freq_hz;

    println!("Configuration:");
    println!("  Symbol: {}", symbol);
    println!("  Frequency: {} Hz ({} ms interval)", freq_hz, feature_interval_ms);
    println!();
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
    let mut price_buffer = PriceBuffer::new(1000); // For trend features
    let mut last_feature_emit = Instant::now();
    let mut last_price_sample = Instant::now();
    let mut update_count = 0u64;
    let feature_interval = Duration::from_millis(feature_interval_ms);
    let price_sample_interval = Duration::from_millis(100); // Sample price every 100ms

    FeatureSnapshot::print_header(freq_hz);

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
                // Sample price at regular intervals for trend features
                if last_price_sample.elapsed() >= price_sample_interval {
                    let midprice = order_book.midprice();
                    if midprice > 0.0 {
                        price_buffer.push(midprice);
                    }
                    last_price_sample = Instant::now();
                }

                // Check if it's time to emit features
                if last_feature_emit.elapsed() >= feature_interval {
                    update_count += 1;

                    // Compute trend features
                    let (ema_short, ema_long) = price_buffer.emas(10, 50);
                    let ma_crossover = ema_short - ema_long;

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
                        // Trend features
                        momentum_60: price_buffer.momentum(60),
                        momentum_300: price_buffer.momentum(300),
                        monotonicity_60: price_buffer.monotonicity(60),
                        monotonicity_300: price_buffer.monotonicity(300),
                        ma_crossover,
                        price_samples: price_buffer.len(),
                    };

                    snapshot.print(update_count);
                    last_feature_emit = Instant::now();
                }
            }
        }
    }

    Ok(())
}
