//! Market state management

mod context;
mod order_book;
mod ring_buffer;
mod trade_buffer;

pub use context::MarketContext;
pub use order_book::OrderBook;
pub use ring_buffer::RingBuffer;
pub use trade_buffer::{Trade, TradeBuffer};

use crate::algorithms::{self, MicrostructureAlgorithm};
use crate::config::FeaturesConfig;
use crate::features::{
    CrossSymbolState, FeatureComputer, Features, GmmClassificationFeatures, RegimeBuffer,
    RegimeConfig,
};
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
    /// Cross-symbol shared OBI state (optional, set when multi-symbol tracking is active)
    cross_symbol_state: Option<CrossSymbolState>,
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
    /// Pluggable microstructure algorithms (compute alg_features each tick)
    algorithms: Vec<Box<dyn MicrostructureAlgorithm>>,
}

impl MarketState {
    /// Create a new market state
    pub fn new(symbol: &str, config: &FeaturesConfig) -> Self {
        Self::new_with_algorithms(symbol, config, Vec::new())
    }

    /// Create a new market state with pluggable algorithms
    pub fn new_with_algorithms(
        symbol: &str,
        config: &FeaturesConfig,
        algorithms: Vec<Box<dyn MicrostructureAlgorithm>>,
    ) -> Self {
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
            cross_symbol_state: None,
            current_minute: 0,
            minute_volume: 0.0,
            minute_buy_volume: 0.0,
            minute_sell_volume: 0.0,
            minute_last_price: None,
            algorithms,
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
                // Update cross-symbol shared state with current OBI_l5
                if let Some(ref css) = self.cross_symbol_state {
                    let obi_l5 = self.order_book.volume_imbalance(5);
                    css.update(&self.symbol, obi_l5);
                }
                if !self.initialized {
                    self.initialized = true;
                    tracing::info!(
                        symbol = %self.symbol,
                        midprice = ?self.order_book.midprice(),
                        "Market state initialized (first Book message processed)"
                    );
                }
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

    /// Compute features from current state.
    /// Returns (base_features, algorithm_values).
    pub fn compute_features(&mut self) -> Option<(Features, Vec<f64>)> {
        if !self.initialized {
            return None;
        }

        let mut features = self.feature_computer.compute(
            &self.order_book,
            &self.trade_buffer,
            &self.context,
            &self.price_buffer,
        );

        // Add cross-symbol features if shared state is available
        if let Some(ref css) = self.cross_symbol_state {
            features.cross_symbol = Some(css.compute(&self.symbol));
        }

        // Add regime features if buffer has enough data
        if self.regime_buffer.is_ready() {
            let regime_features = self.regime_buffer.compute();

            // Run GMM classification if classifier is available
            if let Some(ref classifier) = self.regime_classifier {
                // Feature vector must match train_regime_gmm.py FEATURE_COLUMNS order.
                // Currently 4D (whale flow excluded — all-NaN until feature is wired).
                let gmm_input = vec![
                    features.illiquidity.kyle_lambda_100, // illiq_kyle_100
                    features.toxicity.vpin_50,            // toxic_vpin_50
                    regime_features.absorption_zscore,    // regime_absorption_zscore
                    features.trend.hurst_300,             // trend_hurst_300
                ];

                let (regime, probs) = classifier.classify(&gmm_input);
                features.gmm_classification = Some(GmmClassificationFeatures::from_classification(
                    regime, &probs,
                ));
            }

            features.regime = Some(regime_features);
        }

        // Run microstructure algorithms
        let alg_values = algorithms::run_all(&mut self.algorithms, &features);

        Some((features, alg_values))
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

    /// Set shared cross-symbol state for cross-symbol feature computation
    pub fn set_cross_symbol_state(&mut self, state: CrossSymbolState) {
        self.cross_symbol_state = Some(state);
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
        assert!(
            total_all > Features::count(),
            "count_all should include optional features"
        );
    }

    #[test]
    fn test_features_names_include_gmm() {
        use crate::features::{Features, GmmClassificationFeatures};

        let gmm_names = GmmClassificationFeatures::names();
        assert!(gmm_names.contains(&"regime"), "Should include regime field");
        assert!(
            gmm_names.contains(&"regime_confidence"),
            "Should include confidence"
        );
        assert!(
            gmm_names.contains(&"regime_entropy"),
            "Should include entropy"
        );

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

        // to_vec always returns fixed length (NaN for missing optional features)
        let vec_without = features.to_vec();
        assert_eq!(
            vec_without.len(),
            Features::count_all(),
            "to_vec should always return count_all() elements"
        );

        // GMM slots should be NaN when not set
        // GMM is followed by cross_symbol (3) and heatmap (8) in to_vec order
        use crate::features::{CrossSymbolFeatures, HeatmapFeatures};
        let gmm_start = Features::count_all()
            - HeatmapFeatures::count()
            - CrossSymbolFeatures::count()
            - GmmClassificationFeatures::count();
        assert!(
            vec_without[gmm_start].is_nan(),
            "GMM features should be NaN when not set"
        );

        // With GMM set, length stays the same but values are filled
        let gmm_output = GmmClassificationFeatures::from_classification(
            Regime::Accumulation,
            &[0.5, 0.1, 0.1, 0.1, 0.2],
        );
        features.gmm_classification = Some(gmm_output);

        let vec_with = features.to_vec();
        assert_eq!(
            vec_with.len(),
            Features::count_all(),
            "to_vec length must be constant regardless of optional features"
        );
        assert!(
            !vec_with[gmm_start].is_nan(),
            "GMM features should be filled when set"
        );
    }

    #[test]
    fn test_gmm_classifier_mock() {
        use crate::ml::regime::{Regime, RegimeClassifier};

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

        // Classify some features (4D: kyle_lambda, vpin, absorption, hurst)
        let features = [0.0, 0.0, 0.0, 0.5];
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

    // ========================================================================
    // End-to-end integration: WsMessage -> MarketState -> Features -> validate
    // ========================================================================

    fn make_book(bid_px: f64, ask_px: f64, depth: usize, timestamp_ms: u64) -> WsMessage {
        let mut bids = Vec::with_capacity(depth);
        let mut asks = Vec::with_capacity(depth);
        for i in 0..depth {
            bids.push(ing_types::WsLevel {
                px: format!("{:.1}", bid_px - i as f64 * 0.5),
                sz: format!("{:.1}", 5.0 + i as f64),
                n: 3,
            });
            asks.push(ing_types::WsLevel {
                px: format!("{:.1}", ask_px + i as f64 * 0.5),
                sz: format!("{:.1}", 5.0 + i as f64),
                n: 3,
            });
        }
        WsMessage::Book(ing_types::WsBook {
            coin: "BTC".to_string(),
            levels: (bids, asks),
            time: timestamp_ms,
        })
    }

    fn make_trade(price: f64, size: f64, is_buy: bool, timestamp_ms: u64, tid: u64) -> WsMessage {
        WsMessage::Trades(vec![ing_types::WsTrade {
            coin: "BTC".to_string(),
            side: if is_buy { "B" } else { "A" }.to_string(),
            px: format!("{:.1}", price),
            sz: format!("{:.2}", size),
            hash: format!("0x{:016x}", tid),
            time: timestamp_ms,
            tid,
            users: None,
        }])
    }

    fn make_asset_ctx(funding: f64, oi: f64, oracle_px: f64, timestamp_ms: u64) -> WsMessage {
        WsMessage::AssetCtx(ing_types::WsAssetCtx {
            coin: "BTC".to_string(),
            ctx: ing_types::AssetCtxData {
                day_ntl_vlm: "50000000.0".to_string(),
                funding: format!("{:.8}", funding),
                open_interest: format!("{:.2}", oi),
                oracle_px: format!("{:.1}", oracle_px),
                prev_day_px: format!("{:.1}", oracle_px - 100.0),
                mark_px: Some(format!("{:.1}", oracle_px + 0.5)),
                premium: Some("0.00001".to_string()),
            },
        })
    }

    #[test]
    fn test_e2e_uninit_returns_none() {
        let config = default_config();
        let mut state = MarketState::new("BTC", &config);
        assert!(state.compute_features().is_none(), "Should return None before first Book");
    }

    #[test]
    fn test_e2e_single_book_produces_features() {
        let config = default_config();
        let mut state = MarketState::new("BTC", &config);

        state.update(&make_book(50000.0, 50001.0, 10, 1_700_000_000_000));

        let (features, alg_values) = state.compute_features().expect("Should produce features");
        let vec = features.to_vec();

        // Feature vector length contract
        assert_eq!(
            vec.len(),
            Features::count_all(),
            "to_vec must return exactly count_all() elements"
        );
        assert_eq!(
            Features::names_all().len(),
            Features::count_all(),
            "names_all must match count_all"
        );

        // Raw features should be populated
        assert!(features.raw.midprice > 0.0, "Midprice should be positive");
        assert!(features.raw.spread > 0.0, "Spread should be positive");
        assert!((features.raw.midprice - 50000.5).abs() < 0.1, "Midprice ~ 50000.5");

        // No algorithms registered, so alg_values is empty
        assert!(alg_values.is_empty());
    }

    #[test]
    fn test_e2e_multi_tick_sequence() {
        let config = default_config();
        let mut state = MarketState::new("BTC", &config);

        let base_time: u64 = 1_700_000_000_000;
        let mut all_vecs = Vec::new();

        // Simulate 50 ticks: book updates + trades + context
        for i in 0u64..50 {
            let t = base_time + i * 100;
            let mid = 50000.0 + (i as f64 * 0.1); // slow drift up

            // Book every tick
            state.update(&make_book(mid - 0.5, mid + 0.5, 10, t));

            // Trades every 3rd tick
            if i % 3 == 0 {
                state.update(&make_trade(mid, 0.5, i % 2 == 0, t, i));
            }

            // Context once at start
            if i == 0 {
                state.update(&make_asset_ctx(0.0001, 1_000_000.0, mid, t));
            }

            let (features, _) = state.compute_features().expect("Should produce features");
            all_vecs.push(features.to_vec());
        }

        // All vectors same length
        let expected_len = Features::count_all();
        for (i, vec) in all_vecs.iter().enumerate() {
            assert_eq!(vec.len(), expected_len, "Tick {} wrong vector length", i);
        }

        // Base features (first 154) should have no NaN after warmup (tick 10+)
        let base_count = Features::count();
        for (i, vec) in all_vecs.iter().enumerate().skip(10) {
            let base_nans = vec[..base_count].iter().filter(|v| v.is_nan()).count();
            // Allow some NaN in entropy/trend features that need longer warmup
            assert!(
                base_nans < base_count / 2,
                "Tick {} has {} NaN out of {} base features (too many)",
                i, base_nans, base_count
            );
        }

        // Midprice should be monotonically increasing (we drifted up)
        for i in 1..all_vecs.len() {
            assert!(
                all_vecs[i][0] >= all_vecs[i - 1][0],
                "Midprice should be non-decreasing (tick {})",
                i
            );
        }
    }

    #[test]
    fn test_e2e_optional_features_are_nan() {
        let config = default_config();
        let mut state = MarketState::new("BTC", &config);

        state.update(&make_book(50000.0, 50001.0, 10, 1_700_000_000_000));
        let (features, _) = state.compute_features().unwrap();
        let vec = features.to_vec();

        // Optional categories (whale, liquidation, concentration, regime, gmm,
        // cross_symbol, heatmap) should all be NaN since no data sources
        assert!(features.whale_flow.is_none());
        assert!(features.liquidation_risk.is_none());
        assert!(features.concentration.is_none());
        assert!(features.regime.is_none());
        assert!(features.gmm_classification.is_none());
        assert!(features.cross_symbol.is_none());
        assert!(features.heatmap.is_none());

        // Their slots in to_vec should be NaN
        let base_count = Features::count();
        let optional_slice = &vec[base_count..];
        assert!(
            optional_slice.iter().all(|v| v.is_nan()),
            "All optional feature slots should be NaN when data sources absent"
        );
    }

    #[test]
    fn test_e2e_symbol_preserved() {
        let config = default_config();
        for sym in &["BTC", "ETH", "SOL"] {
            let state = MarketState::new(sym, &config);
            assert_eq!(state.symbol(), *sym);
        }
    }
}
