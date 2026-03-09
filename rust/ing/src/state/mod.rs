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
use crate::features::{Features, FeatureComputer};
use crate::ws::{WsMessage, WsBook, WsTrade, WsAssetCtx};

/// Aggregated market state for a single symbol
pub struct MarketState {
    symbol: String,
    order_book: OrderBook,
    trade_buffer: TradeBuffer,
    context: MarketContext,
    feature_computer: FeatureComputer,
    price_buffer: RingBuffer<f64>,
    initialized: bool,
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
        }
    }

    /// Update state from a WebSocket message
    pub fn update(&mut self, msg: &WsMessage) {
        match msg {
            WsMessage::Book(book) => {
                self.order_book.update(book);
                if let Some(mid) = self.order_book.midprice() {
                    self.price_buffer.push(mid);
                }
                self.initialized = true;
            }
            WsMessage::Trades(trades) => {
                for trade in trades {
                    self.trade_buffer.add(trade.clone());
                }
            }
            WsMessage::AssetCtx(ctx) => {
                self.context.update(ctx);
            }
            WsMessage::Unknown(_) => {}
        }
    }

    /// Compute features from current state
    pub fn compute_features(&self) -> Option<Features> {
        if !self.initialized {
            return None;
        }

        Some(self.feature_computer.compute(
            &self.order_book,
            &self.trade_buffer,
            &self.context,
            &self.price_buffer,
        ))
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
