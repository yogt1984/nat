//! Hypothesis Testing Module
//!
//! This module implements rigorous statistical hypothesis tests for validating
//! the Hyperliquid analytics layer's core hypotheses.
//!
//! # Hypotheses
//!
//! | ID | Hypothesis | Success Criteria |
//! |----|------------|------------------|
//! | H1 | Whale flow predicts returns | corr > 0.05, p < 0.001, MI > 0.02 |
//! | H2 | Entropy + whale interaction | lift > 10%, MI gain > 0.01 |
//! | H3 | Liquidation cascades | precision > 30%, lift > 2x |
//! | H4 | Concentration predicts vol | corr > 0.2, p < 0.01 |
//! | H5 | Persistence indicator works | Sharpe > 0.5, OOS > 0.7*IS |
//!
//! # Decision Framework
//!
//! - 0-1 hypotheses pass → NO-GO
//! - 2-3 hypotheses pass → PIVOT
//! - 4-5 hypotheses pass → GO

pub mod stats;
pub mod h1_whale_flow;
pub mod h2_entropy_whale;
pub mod h3_liquidation_cascade;
pub mod h4_concentration_vol;
pub mod h5_persistence;

pub use stats::{
    pearson_correlation, spearman_correlation, mutual_information,
    t_test_correlation, bonferroni_correct, CorrelationResult,
    WalkForwardResult, walk_forward_correlation,
};

pub use h1_whale_flow::{
    H1TestResult, H1TestConfig, H1Decision,
    run_h1_whale_flow_test,
};

pub use h2_entropy_whale::{
    H2TestResult, H2TestConfig, H2Decision,
    run_h2_entropy_whale_test,
    ContingencyTable, InteractionTable,
};

pub use h3_liquidation_cascade::{
    H3TestResult, H3TestConfig, H3Decision,
    run_h3_liquidation_cascade_test,
    ClassificationMetrics, ThresholdResult,
    LeadTimeAnalysis, DirectionAnalysis,
};

pub use h4_concentration_vol::{
    H4TestResult, H4TestConfig, H4Decision,
    run_h4_concentration_vol_test,
    ConcentrationMeasureResult, RegimeAnalysis as H4RegimeAnalysis, CausalityAnalysis,
};

pub use h5_persistence::{
    H5TestResult, H5TestConfig, H5Decision,
    run_h5_persistence_test,
    FeatureRow, ThreeBarLabel, HorizonResult, FoldResult,
    RegimeAnalysis as H5RegimeAnalysis, RegimeResult, FeatureImportance,
    compute_persistence_indicator, compute_future_returns, sharpe_ratio,
};

/// Overall hypothesis test outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypothesisDecision {
    /// Strong evidence FOR the hypothesis
    Accept,
    /// Strong evidence AGAINST the hypothesis
    Reject,
    /// Insufficient evidence either way
    Inconclusive,
}

impl std::fmt::Display for HypothesisDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HypothesisDecision::Accept => write!(f, "ACCEPT"),
            HypothesisDecision::Reject => write!(f, "REJECT"),
            HypothesisDecision::Inconclusive => write!(f, "INCONCLUSIVE"),
        }
    }
}
