//! Feature Computation Module
//!
//! Extracts 220 features from Hyperliquid WebSocket market data across 20 categories.
//! See `FEATURES.md` at the project root for the full feature manifest with formulas,
//! interpretation, and paper references.
//!
//! # Feature Categories
//!
//! | Category | Count | Prefix | Status | Key Reference |
//! |----------|-------|--------|--------|---------------|
//! | Raw | 10 | `raw_` | All working | Gatheral & Oomen (2010) |
//! | Imbalance | 8 | `imbalance_` | All working | Cont, Stoikov & Talreja (2010) |
//! | Flow | 12 | `flow_` | All working | — |
//! | Volatility | 9 | `vol_` | All working | Parkinson (1980), Garman & Klass (1980) |
//! | Entropy | 24 | `ent_` | All warmup-dependent | Bandt & Pompe (2002) |
//! | Context | 12 | `ctx_` | All working | — |
//! | Trend | 15 | `trend_` | All working | Jegadeesh & Titman (1993) |
//! | Illiquidity | 12 | `illiq_` | All working | Kyle (1985) |
//! | Toxicity | 10 | `toxic_` | All working | Easley et al. (2012) |
//! | Derived | 15 | `derived_` | All working | — |
//! | Microstructure | 5 | `micro_` | All working | Biais et al. (1995) |
//! | Resilience | 3 | `resilience_` | All working | Biais et al. (1995) |
//! | Hawkes | 3 | `hawkes_` | All working | Bacry et al. (2015) |
//! | *Whale Flow* | 12 | `whale_` | Optional (NaN if absent) | — |
//! | *Liquidation* | 13 | `liquidation_` | Optional (NaN if absent) | — |
//! | *Concentration* | 15 | `top`/mixed | Optional (NaN if absent) | — |
//! | *Regime* | 20 | `regime_` | Optional (NaN if absent) | — |
//! | *GMM* | 8 | `regime`/`prob_` | Optional (NaN if absent) | — |
//! | *Cross-Symbol* | 3 | `cross_` | Optional (NaN if absent) | — |
//! | *Heatmap* | 8 | `hm_` | Optional (NaN if absent) | Cont & Wagalath (2016) |
//!
//! Base features (138) are always computed. Optional features (79) require
//! additional data sources or warmup time and are NaN-padded when absent.
//!
//! # Data Contract
//!
//! `Features::to_vec()` always returns exactly `count_all()` = 220 elements.
//! `Features::names_all()` returns the corresponding column names.
//! The Parquet schema is built from `names_all()` in `output/schema.rs`.

mod raw;
mod imbalance;
mod flow;
mod volatility;
mod entropy;
mod context;
mod trend;
mod illiquidity;
mod toxicity;
mod derived;
mod microstructure;
mod resilience;
mod hawkes;
pub mod cross_symbol;
pub mod whale_flow;
pub mod liquidation;
pub mod concentration;
pub mod regime;
pub mod heatmap;

pub use raw::RawFeatures;
pub use imbalance::ImbalanceFeatures;
pub use flow::FlowFeatures;
pub use volatility::VolatilityFeatures;
pub use entropy::EntropyFeatures;
pub use context::ContextFeatures;
pub use trend::TrendFeatures;
pub use illiquidity::IlliquidityFeatures;
pub use toxicity::ToxicityFeatures;
pub use derived::DerivedFeatures;
pub use microstructure::MicrostructureFeatures;
pub use resilience::{ResilienceFeatures, ResilienceTracker};
pub use hawkes::HawkesFeatures;
pub use cross_symbol::{CrossSymbolFeatures, CrossSymbolState};
pub use whale_flow::{WhaleFlowFeatures, WhaleFlowBuffer, WhaleFlowConfig, WhalePositionChange};
pub use liquidation::{LiquidationRiskFeatures, LiquidationRiskConfig, LiquidationPosition};
pub use heatmap::{HeatmapFeatures, HeatmapBuffer, HeatmapConfig};
pub use concentration::{ConcentrationFeatures, ConcentrationBuffer, ConcentrationConfig, Position as ConcentrationPosition};
pub use regime::{RegimeFeatures, RegimeBuffer, RegimeConfig, AbsorptionComputer, DivergenceComputer, ChurnComputer, RangeComputer};

use ing_types::FeaturesConfig;
use ing_types::GmmClassificationFeatures;
use ing_types::{OrderBook, TradeBuffer, MarketContext, RingBuffer};

/// All computed features
#[derive(Debug, Clone, Default)]
pub struct Features {
    pub raw: RawFeatures,
    pub imbalance: ImbalanceFeatures,
    pub flow: FlowFeatures,
    pub volatility: VolatilityFeatures,
    pub entropy: EntropyFeatures,
    pub context: ContextFeatures,
    pub trend: TrendFeatures,
    pub illiquidity: IlliquidityFeatures,
    pub toxicity: ToxicityFeatures,
    pub derived: DerivedFeatures,
    /// OBI dynamics + queue position proxy (5.2 + 5.5)
    pub microstructure: MicrostructureFeatures,
    /// Order book resilience after large takes (5.1)
    pub resilience: ResilienceFeatures,
    /// Hawkes self-exciting trade intensity (5.3)
    pub hawkes: HawkesFeatures,
    /// Whale flow features (Hyperliquid-unique, requires position tracking)
    pub whale_flow: Option<WhaleFlowFeatures>,
    /// Liquidation risk features (Hyperliquid-unique, requires position tracking)
    pub liquidation_risk: Option<LiquidationRiskFeatures>,
    /// Position concentration features (Hyperliquid-unique, requires position tracking)
    pub concentration: Option<ConcentrationFeatures>,
    /// Regime detection features (accumulation/distribution, minute-level)
    pub regime: Option<RegimeFeatures>,
    /// GMM regime classification output (regime label, probabilities, confidence)
    pub gmm_classification: Option<GmmClassificationFeatures>,
    /// Cross-symbol OBI divergence (5.4, requires multi-symbol shared state)
    pub cross_symbol: Option<CrossSymbolFeatures>,
    /// Liquidity heatmap features (spatial liquidation density, cascade detection)
    pub heatmap: Option<HeatmapFeatures>,
}

impl Features {
    /// Get total number of base features (always computed)
    pub fn count() -> usize {
        RawFeatures::count() +
        ImbalanceFeatures::count() +
        FlowFeatures::count() +
        VolatilityFeatures::count() +
        EntropyFeatures::count() +
        ContextFeatures::count() +
        TrendFeatures::count() +
        IlliquidityFeatures::count() +
        ToxicityFeatures::count() +
        DerivedFeatures::count() +
        MicrostructureFeatures::count() +
        ResilienceFeatures::count() +
        HawkesFeatures::count()
    }

    /// Get total number of features including whale flow
    pub fn count_with_whale_flow() -> usize {
        Self::count() + WhaleFlowFeatures::count()
    }

    /// Get total number of features including all Hyperliquid-unique features
    pub fn count_with_hyperliquid_features() -> usize {
        Self::count() + WhaleFlowFeatures::count() + LiquidationRiskFeatures::count() + ConcentrationFeatures::count()
    }

    /// Get total number of features including all optional features
    pub fn count_all() -> usize {
        Self::count()
            + WhaleFlowFeatures::count()
            + LiquidationRiskFeatures::count()
            + ConcentrationFeatures::count()
            + RegimeFeatures::count()
            + GmmClassificationFeatures::count()
            + CrossSymbolFeatures::count()
            + HeatmapFeatures::count()
    }

    /// Convert to flat vector of f64 (fixed-length, NaN for missing optional features)
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(Self::count_all());
        // Base features (always present)
        v.extend(self.raw.to_vec());
        v.extend(self.imbalance.to_vec());
        v.extend(self.flow.to_vec());
        v.extend(self.volatility.to_vec());
        v.extend(self.entropy.to_vec());
        v.extend(self.context.to_vec());
        v.extend(self.trend.to_vec());
        v.extend(self.illiquidity.to_vec());
        v.extend(self.toxicity.to_vec());
        v.extend(self.derived.to_vec());
        v.extend(self.microstructure.to_vec());
        v.extend(self.resilience.to_vec());
        v.extend(self.hawkes.to_vec());
        // Optional features (NaN when not yet available)
        match &self.whale_flow {
            Some(wf) => v.extend(wf.to_vec()),
            None => v.extend(std::iter::repeat(f64::NAN).take(WhaleFlowFeatures::count())),
        }
        match &self.liquidation_risk {
            Some(lr) => v.extend(lr.to_vec()),
            None => v.extend(std::iter::repeat(f64::NAN).take(LiquidationRiskFeatures::count())),
        }
        match &self.concentration {
            Some(c) => v.extend(c.to_vec()),
            None => v.extend(std::iter::repeat(f64::NAN).take(ConcentrationFeatures::count())),
        }
        match &self.regime {
            Some(r) => v.extend(r.to_vec()),
            None => v.extend(std::iter::repeat(f64::NAN).take(RegimeFeatures::count())),
        }
        match &self.gmm_classification {
            Some(g) => v.extend(g.to_vec()),
            None => v.extend(std::iter::repeat(f64::NAN).take(GmmClassificationFeatures::count())),
        }
        match &self.cross_symbol {
            Some(cs) => v.extend(cs.to_vec()),
            None => v.extend(std::iter::repeat(f64::NAN).take(CrossSymbolFeatures::count())),
        }
        match &self.heatmap {
            Some(hm) => v.extend(hm.to_vec()),
            None => v.extend(std::iter::repeat(f64::NAN).take(HeatmapFeatures::count())),
        }
        v
    }

    /// Get feature names (base features only)
    pub fn names() -> Vec<&'static str> {
        let mut names = Vec::with_capacity(Self::count());
        names.extend(RawFeatures::names());
        names.extend(ImbalanceFeatures::names());
        names.extend(FlowFeatures::names());
        names.extend(VolatilityFeatures::names());
        names.extend(EntropyFeatures::names());
        names.extend(ContextFeatures::names());
        names.extend(TrendFeatures::names());
        names.extend(IlliquidityFeatures::names());
        names.extend(ToxicityFeatures::names());
        names.extend(DerivedFeatures::names());
        names.extend(MicrostructureFeatures::names());
        names.extend(ResilienceFeatures::names());
        names.extend(HawkesFeatures::names());
        names
    }

    /// Get all feature names including whale flow
    pub fn names_with_whale_flow() -> Vec<&'static str> {
        let mut names = Self::names();
        names.extend(WhaleFlowFeatures::names());
        names
    }

    /// Get all feature names including all Hyperliquid-unique features
    pub fn names_with_hyperliquid_features() -> Vec<&'static str> {
        let mut names = Self::names();
        names.extend(WhaleFlowFeatures::names());
        names.extend(LiquidationRiskFeatures::names());
        names.extend(ConcentrationFeatures::names());
        names
    }

    /// Get all feature names including all optional features
    pub fn names_all() -> Vec<&'static str> {
        let mut names = Self::names();
        names.extend(WhaleFlowFeatures::names());
        names.extend(LiquidationRiskFeatures::names());
        names.extend(ConcentrationFeatures::names());
        names.extend(RegimeFeatures::names());
        names.extend(GmmClassificationFeatures::names());
        names.extend(CrossSymbolFeatures::names());
        names.extend(HeatmapFeatures::names());
        names
    }

    /// Set whale flow features
    pub fn with_whale_flow(mut self, whale_flow: WhaleFlowFeatures) -> Self {
        self.whale_flow = Some(whale_flow);
        self
    }

    /// Set liquidation risk features
    pub fn with_liquidation_risk(mut self, liquidation_risk: LiquidationRiskFeatures) -> Self {
        self.liquidation_risk = Some(liquidation_risk);
        self
    }

    /// Set concentration features
    pub fn with_concentration(mut self, concentration: ConcentrationFeatures) -> Self {
        self.concentration = Some(concentration);
        self
    }

    /// Set regime features
    pub fn with_regime(mut self, regime: RegimeFeatures) -> Self {
        self.regime = Some(regime);
        self
    }

    /// Set GMM classification features
    pub fn with_gmm_classification(mut self, gmm: GmmClassificationFeatures) -> Self {
        self.gmm_classification = Some(gmm);
        self
    }

    /// Set cross-symbol features
    pub fn with_cross_symbol(mut self, cross_symbol: CrossSymbolFeatures) -> Self {
        self.cross_symbol = Some(cross_symbol);
        self
    }

    /// Set heatmap features
    pub fn with_heatmap(mut self, heatmap: HeatmapFeatures) -> Self {
        self.heatmap = Some(heatmap);
        self
    }
}

/// Feature computer that manages all feature calculations
pub struct FeatureComputer {
    _config: FeaturesConfig,
    spread_buffer: RingBuffer<f64>,
    midprice_buffer: RingBuffer<f64>,
    entropy_buffer: RingBuffer<f64>,
    imbalance_buffer: RingBuffer<f64>,
    /// OBI_l5 history for microstructure OBI dynamics (5.2)
    obi_buffer: RingBuffer<f64>,
    /// Total L5 depth history for microstructure depth recovery (5.2)
    depth_buffer: RingBuffer<f64>,
    /// Resilience tracker state machine (5.1)
    resilience_tracker: ResilienceTracker,
    /// Hourly vol_1m history for z-score normalization (1 hour at 100ms = 36,000 samples)
    vol_1m_buffer: RingBuffer<f64>,
}

impl FeatureComputer {
    /// Create a new feature computer
    pub fn new(config: &FeaturesConfig) -> Self {
        Self {
            _config: config.clone(),
            spread_buffer: RingBuffer::new(600),    // 1 minute at 100ms
            midprice_buffer: RingBuffer::new(3000), // 5 minutes at 100ms
            entropy_buffer: RingBuffer::new(600),   // 1 minute at 100ms
            imbalance_buffer: RingBuffer::new(16),  // 16 samples for permutation entropy
            obi_buffer: RingBuffer::new(600),       // 1 minute at 100ms for OBI dynamics
            depth_buffer: RingBuffer::new(600),     // 1 minute at 100ms for depth recovery
            resilience_tracker: ResilienceTracker::new(),
            vol_1m_buffer: RingBuffer::new(36_000), // 1 hour at 100ms for vol z-score
        }
    }

    /// Compute all features from current state
    pub fn compute(
        &mut self,
        order_book: &OrderBook,
        trade_buffer: &TradeBuffer,
        market_context: &MarketContext,
        price_buffer: &RingBuffer<f64>,
    ) -> Features {
        // Update history buffers
        if let Some(spread) = order_book.spread() {
            self.spread_buffer.push(spread);
        }
        if let Some(mid) = order_book.midprice() {
            self.midprice_buffer.push(mid);
        }
        let imbalance_l1 = order_book.volume_imbalance(1);
        self.imbalance_buffer.push(imbalance_l1);

        // Update OBI and depth buffers for microstructure features
        let obi_l5 = order_book.volume_imbalance(5);
        self.obi_buffer.push(obi_l5);
        let total_depth = order_book.bid_depth(5) + order_book.ask_depth(5);
        self.depth_buffer.push(total_depth);

        let raw = raw::compute(order_book);
        let imbalance = imbalance::compute(order_book);
        let flow = flow::compute(trade_buffer);
        let volatility = volatility::compute(
            price_buffer, order_book,
            &self.spread_buffer, &self.vol_1m_buffer,
        );

        // Push current vol_1m into hourly history for z-score computation
        self.vol_1m_buffer.push(volatility.returns_1m);

        let entropy = entropy::compute(
            price_buffer, order_book, trade_buffer,
            &self.imbalance_buffer, &self.spread_buffer, &self.entropy_buffer,
        );

        // Update entropy history buffer with a representative entropy value (tick_1m)
        self.entropy_buffer.push(entropy.tick_entropy_1m);

        let context = context::compute(market_context);
        let trend = trend::compute(price_buffer);
        let illiquidity = illiquidity::compute(trade_buffer);
        // Use previous mid-price for causal realized spread (Huang & Stoll 1997)
        let prev_mid = if self.midprice_buffer.len() >= 2 {
            self.midprice_buffer.get(self.midprice_buffer.len() - 2).copied()
        } else {
            None
        };
        let toxicity = toxicity::compute(trade_buffer, prev_mid);

        // Compute derived features from base features
        let derived = derived::compute(
            &entropy,
            &trend,
            &volatility,
            &illiquidity,
            &toxicity,
            &flow,
        );

        // New Phase 5 base features
        let micro = microstructure::compute(
            order_book, trade_buffer,
            &self.obi_buffer, &self.depth_buffer,
        );
        let resilience_features = self.resilience_tracker.update(total_depth);
        let hawkes_features = hawkes::compute(trade_buffer);

        Features {
            raw,
            imbalance,
            flow,
            volatility,
            entropy,
            context,
            trend,
            illiquidity,
            toxicity,
            derived,
            microstructure: micro,
            resilience: resilience_features,
            hawkes: hawkes_features,
            whale_flow: None, // Computed separately via WhaleFlowBuffer
            liquidation_risk: None, // Computed separately via liquidation::compute()
            concentration: None, // Computed separately via ConcentrationBuffer
            regime: None, // Computed separately via RegimeBuffer at minute intervals
            gmm_classification: None, // Computed when regime features are ready
            cross_symbol: None, // Computed separately via CrossSymbolState
            heatmap: None, // Computed separately via HeatmapBuffer
        }
    }
}
