//! Market state management

mod order_book;
mod trade_buffer;
mod context;
mod ring_buffer;

pub use order_book::OrderBook;
pub use trade_buffer::{TradeBuffer, Trade};
pub use context::MarketContext;
pub use ring_buffer::RingBuffer;

use crate::config::FeaturesConfig;
use crate::features::{Features, FeatureComputer, RegimeBuffer, RegimeConfig};
use crate::ws::WsMessage;

/// Aggregated market state for a single symbol
pub struct MarketState {
    symbol: String,
    order_book: OrderBook,
    trade_buffer: TradeBuffer,
    context: MarketContext,
    feature_computer: FeatureComputer,
    price_buffer: RingBuffer<f64>,
    initialized: bool,
    /// Regime detection buffer (minute-level features)
    regime_buffer: RegimeBuffer,
    /// Current minute timestamp (floored to minute)
    current_minute: u64,
    /// Accumulated volume this minute
    minute_volume: f64,
    /// Accumulated buy volume this minute
    minute_buy_volume: f64,
    /// Accumulated sell volume this minute
    minute_sell_volume: f64,
    /// Last price seen this minute
    minute_last_price: Option<f64>,
}

impl MarketState {
    /// Create a new market state
    pub fn new(symbol: &str, config: &FeaturesConfig) -> Self {
        Self {
            symbol: symbol.to_string(),
            order_book: OrderBook::new(config.book_levels),
            trade_buffer: TradeBuffer::new(config.trade_buffer_seconds),
            context: MarketContext::new(),
            feature_computer: FeatureComputer::new(config),
            price_buffer: RingBuffer::new(config.price_buffer_size),
            initialized: false,
            regime_buffer: RegimeBuffer::new(RegimeConfig::default()),
            current_minute: 0,
            minute_volume: 0.0,
            minute_buy_volume: 0.0,
            minute_sell_volume: 0.0,
            minute_last_price: None,
        }
    }

    /// Update state from a WebSocket message
    pub fn update(&mut self, msg: &WsMessage) {
        match msg {
            WsMessage::Book(book) => {
                self.order_book.update(book);
                if let Some(mid) = self.order_book.midprice() {
                    self.price_buffer.push(mid);
                    self.minute_last_price = Some(mid);
                }
                self.initialized = true;
            }
            WsMessage::Trades(trades) => {
                for trade in trades {
                    // Accumulate volumes for minute bar
                    let volume = trade.sz.parse::<f64>().unwrap_or(0.0);
                    self.minute_volume += volume;
                    if trade.side == "B" {
                        self.minute_buy_volume += volume;
                    } else {
                        self.minute_sell_volume += volume;
                    }

                    // Check for minute boundary using trade timestamp
                    let trade_minute = trade.time / 60_000; // ms to minutes
                    if self.current_minute == 0 {
                        self.current_minute = trade_minute;
                    } else if trade_minute > self.current_minute {
                        // Minute boundary crossed - flush to regime buffer
                        self.flush_minute_bar();
                        self.current_minute = trade_minute;
                    }

                    self.trade_buffer.add(trade.clone());
                }
            }
            WsMessage::AssetCtx(ctx) => {
                self.context.update(ctx);
            }
            WsMessage::Unknown(_) => {}
        }
    }

    /// Flush accumulated minute bar data to regime buffer
    fn flush_minute_bar(&mut self) {
        if let Some(price) = self.minute_last_price {
            // Only update if we have data
            if self.minute_volume > 0.0 {
                self.regime_buffer.update(
                    price,
                    self.minute_volume,
                    self.minute_buy_volume,
                    self.minute_sell_volume,
                );
            }
        }

        // Reset accumulators
        self.minute_volume = 0.0;
        self.minute_buy_volume = 0.0;
        self.minute_sell_volume = 0.0;
        // Keep minute_last_price for next bar's close reference
    }

    /// Compute features from current state
    pub fn compute_features(&self) -> Option<Features> {
        if !self.initialized {
            return None;
        }

        let mut features = self.feature_computer.compute(
            &self.order_book,
            &self.trade_buffer,
            &self.context,
            &self.price_buffer,
        );

        // Add regime features if buffer has enough data
        if self.regime_buffer.is_ready() {
            features.regime = Some(self.regime_buffer.compute());
        }

        Some(features)
    }

    /// Get regime buffer for external access (e.g., monitoring)
    pub fn regime_buffer(&self) -> &RegimeBuffer {
        &self.regime_buffer
    }

    /// Get minutes of regime data collected
    pub fn regime_minutes(&self) -> u64 {
        self.regime_buffer.minutes_processed()
    }

    /// Get the symbol
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Check if state is initialized (has received at least one book update)
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}
