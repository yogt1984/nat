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
use crate::features::{Features, FeatureComputer, RegimeBuffer, RegimeConfig, GmmClassificationFeatures};
use crate::ml::regime::RegimeClassifier;
use crate::ws::WsMessage;
use std::path::Path;

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
    /// GMM regime classifier (optional, loaded from model file)
    regime_classifier: Option<RegimeClassifier>,
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
        // Load GMM classifier if model path is provided
        let regime_classifier = config.gmm_model_path.as_ref().and_then(|path| {
            match RegimeClassifier::load(Path::new(path)) {
                Ok(classifier) => {
                    tracing::info!(symbol = symbol, path = path, "Loaded GMM regime classifier");
                    Some(classifier)
                }
                Err(e) => {
                    tracing::warn!(
                        symbol = symbol,
                        path = path,
                        error = %e,
                        "Failed to load GMM regime classifier, classification disabled"
                    );
                    None
                }
            }
        });

        Self {
            symbol: symbol.to_string(),
            order_book: OrderBook::new(config.book_levels),
            trade_buffer: TradeBuffer::new(config.trade_buffer_seconds),
            context: MarketContext::new(),
            feature_computer: FeatureComputer::new(config),
            price_buffer: RingBuffer::new(config.price_buffer_size),
            initialized: false,
            regime_buffer: RegimeBuffer::new(RegimeConfig::default()),
            regime_classifier,
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
    pub fn compute_features(&mut self) -> Option<Features> {
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
            let regime_features = self.regime_buffer.compute();

            // Run GMM classification if classifier is available
            if let Some(ref classifier) = self.regime_classifier {
                // Extract 5D features for GMM input:
                // [kyle_lambda, vpin, absorption_zscore, hurst, whale_net_flow]
                let gmm_input = [
                    features.illiquidity.kyle_lambda_100,   // Kyle's Lambda (closest to 300)
                    features.toxicity.vpin_50,              // VPIN
                    regime_features.absorption_zscore,      // Absorption z-score
                    features.trend.hurst_300,               // Hurst exponent
                    features.whale_flow
                        .as_ref()
                        .map(|wf| wf.whale_net_flow_1h)
                        .unwrap_or(0.0),  // Whale net flow (0 if not available)
                ];

                let (regime, probs) = classifier.classify(&gmm_input);
                features.gmm_classification = Some(GmmClassificationFeatures::from_classification(regime, &probs));
            }

            features.regime = Some(regime_features);
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

    /// Check if GMM classifier is loaded
    pub fn has_gmm_classifier(&self) -> bool {
        self.regime_classifier.is_some()
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::regime::GmmParams;

    fn default_config() -> FeaturesConfig {
        FeaturesConfig {
            emission_interval_ms: 100,
            trade_buffer_seconds: 60,
            book_levels: 10,
            price_buffer_size: 1000,
            gmm_model_path: None,
        }
    }

    #[test]
    fn test_market_state_creation() {
        let config = default_config();
        let state = MarketState::new("BTC", &config);

        assert_eq!(state.symbol(), "BTC");
        assert!(!state.is_initialized());
        assert!(!state.has_gmm_classifier());
    }

    #[test]
    fn test_market_state_without_gmm() {
        let config = default_config();
        let state = MarketState::new("ETH", &config);

        // Without GMM model, classification should not be available
        assert!(!state.has_gmm_classifier());

        // Features should still compute (without GMM classification)
        // Note: Would need to initialize state with book update first
    }

    #[test]
    fn test_features_count_includes_gmm() {
        use crate::features::{Features, GmmClassificationFeatures};

        // Verify GMM features are counted
        let gmm_count = GmmClassificationFeatures::count();
        assert_eq!(gmm_count, 8, "GMM classification should have 8 features");

        // Verify total includes GMM
        let total_all = Features::count_all();
        assert!(total_all > Features::count(), "count_all should include optional features");
    }

    #[test]
    fn test_features_names_include_gmm() {
        use crate::features::{Features, GmmClassificationFeatures};

        let gmm_names = GmmClassificationFeatures::names();
        assert!(gmm_names.contains(&"regime"), "Should include regime field");
        assert!(gmm_names.contains(&"regime_confidence"), "Should include confidence");
        assert!(gmm_names.contains(&"regime_entropy"), "Should include entropy");

        let all_names = Features::names_all();
        for gmm_name in &gmm_names {
            assert!(
                all_names.contains(gmm_name),
                "names_all should include GMM name: {}",
                gmm_name
            );
        }
    }

    #[test]
    fn test_features_to_vec_includes_gmm() {
        use crate::features::{Features, GmmClassificationFeatures};
        use crate::ml::regime::Regime;

        let mut features = Features::default();

        // Without GMM, to_vec should work
        let vec_without = features.to_vec();
        let base_len = vec_without.len();

        // With GMM, to_vec should include 8 more features
        let gmm_output = GmmClassificationFeatures::from_classification(
            Regime::Accumulation,
            &[0.5, 0.1, 0.1, 0.1, 0.2],
        );
        features.gmm_classification = Some(gmm_output);

        let vec_with = features.to_vec();
        assert_eq!(
            vec_with.len(),
            base_len + 8,
            "to_vec with GMM should have 8 more elements"
        );
    }

    #[test]
    fn test_gmm_classifier_mock() {
        use crate::ml::regime::{RegimeClassifier, Regime};

        // Test with mock params
        let params = GmmParams::default();
        let classifier = RegimeClassifier::new(
            params,
            vec![
                Regime::Accumulation,
                Regime::Markup,
                Regime::Distribution,
                Regime::Markdown,
                Regime::Ranging,
            ],
        );

        // Classify some features
        let features = [0.0, 0.0, 0.0, 0.5, 0.0];
        let (regime, probs) = classifier.classify(&features);

        // Probabilities should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Probabilities should sum to 1, got {}",
            sum
        );

        // Regime should be valid
        assert_ne!(regime, Regime::Unknown, "Should classify to a valid regime");
    }
}
