//! Feature computation module

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
pub mod whale_flow;
pub mod liquidation;
pub mod concentration;

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
pub use whale_flow::{WhaleFlowFeatures, WhaleFlowBuffer, WhaleFlowConfig, WhalePositionChange};
pub use liquidation::{LiquidationRiskFeatures, LiquidationRiskConfig, LiquidationPosition};
pub use concentration::{ConcentrationFeatures, ConcentrationBuffer, ConcentrationConfig, Position as ConcentrationPosition};

use crate::config::FeaturesConfig;
use crate::state::{OrderBook, TradeBuffer, MarketContext, RingBuffer};

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
    /// Whale flow features (Hyperliquid-unique, requires position tracking)
    pub whale_flow: Option<WhaleFlowFeatures>,
    /// Liquidation risk features (Hyperliquid-unique, requires position tracking)
    pub liquidation_risk: Option<LiquidationRiskFeatures>,
    /// Position concentration features (Hyperliquid-unique, requires position tracking)
    pub concentration: Option<ConcentrationFeatures>,
}

impl Features {
    /// Get total number of base features (excluding whale flow)
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
        DerivedFeatures::count()
    }

    /// Get total number of features including whale flow
    pub fn count_with_whale_flow() -> usize {
        Self::count() + WhaleFlowFeatures::count()
    }

    /// Get total number of features including all Hyperliquid-unique features
    pub fn count_with_hyperliquid_features() -> usize {
        Self::count() + WhaleFlowFeatures::count() + LiquidationRiskFeatures::count() + ConcentrationFeatures::count()
    }

    /// Convert to flat vector of f64
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(Self::count());
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
        if let Some(ref wf) = self.whale_flow {
            v.extend(wf.to_vec());
        }
        if let Some(ref lr) = self.liquidation_risk {
            v.extend(lr.to_vec());
        }
        if let Some(ref c) = self.concentration {
            v.extend(c.to_vec());
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
}

/// Feature computer that manages all feature calculations
pub struct FeatureComputer {
    config: FeaturesConfig,
    spread_buffer: RingBuffer<f64>,
    midprice_buffer: RingBuffer<f64>,
    entropy_buffer: RingBuffer<f64>,
}

impl FeatureComputer {
    /// Create a new feature computer
    pub fn new(config: &FeaturesConfig) -> Self {
        Self {
            config: config.clone(),
            spread_buffer: RingBuffer::new(600),   // 1 minute at 100ms
            midprice_buffer: RingBuffer::new(3000), // 5 minutes at 100ms
            entropy_buffer: RingBuffer::new(600),
        }
    }

    /// Compute all features from current state
    pub fn compute(
        &self,
        order_book: &OrderBook,
        trade_buffer: &TradeBuffer,
        market_context: &MarketContext,
        price_buffer: &RingBuffer<f64>,
    ) -> Features {
        // Update internal buffers (would need &mut self in real impl)
        // For now, we'll compute from the passed buffers

        let raw = raw::compute(order_book);
        let imbalance = imbalance::compute(order_book);
        let flow = flow::compute(trade_buffer);
        let volatility = volatility::compute(price_buffer, order_book);
        let entropy = entropy::compute(price_buffer, order_book, trade_buffer);
        let context = context::compute(market_context);
        let trend = trend::compute(price_buffer);
        let illiquidity = illiquidity::compute(trade_buffer);
        let toxicity = toxicity::compute(trade_buffer);

        // Compute derived features from base features
        let derived = derived::compute(
            &entropy,
            &trend,
            &volatility,
            &illiquidity,
            &toxicity,
            &flow,
        );

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
            whale_flow: None, // Computed separately via WhaleFlowBuffer
            liquidation_risk: None, // Computed separately via liquidation::compute()
            concentration: None, // Computed separately via ConcentrationBuffer
        }
    }
}
