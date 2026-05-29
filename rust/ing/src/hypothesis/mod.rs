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

pub mod data_loader;
pub mod feature_analysis;
pub mod final_decision;
pub mod h1_whale_flow;
pub mod h2_entropy_whale;
pub mod h3_liquidation_cascade;
pub mod h4_concentration_vol;
pub mod h5_persistence;
pub mod stats;

pub use stats::{
    bonferroni_correct, mutual_information, pearson_correlation, spearman_correlation,
    t_test_correlation, walk_forward_correlation, CorrelationResult, WalkForwardResult,
};

pub use h1_whale_flow::{run_h1_whale_flow_test, H1Decision, H1TestConfig, H1TestResult};

pub use h2_entropy_whale::{
    run_h2_entropy_whale_test, ContingencyTable, H2Decision, H2TestConfig, H2TestResult,
    InteractionTable,
};

pub use h3_liquidation_cascade::{
    run_h3_liquidation_cascade_test, ClassificationMetrics, DirectionAnalysis, H3Decision,
    H3TestConfig, H3TestResult, LeadTimeAnalysis, ThresholdResult,
};

pub use h4_concentration_vol::{
    run_h4_concentration_vol_test, CausalityAnalysis, ConcentrationMeasureResult, H4Decision,
    H4TestConfig, H4TestResult, RegimeAnalysis as H4RegimeAnalysis,
};

pub use h5_persistence::{
    compute_future_returns, compute_persistence_indicator, run_h5_persistence_test, sharpe_ratio,
    FeatureImportance, FeatureRow, FoldResult, H5Decision, H5TestConfig, H5TestResult,
    HorizonResult, RegimeAnalysis as H5RegimeAnalysis, RegimeResult, ThreeBarLabel,
};

pub use feature_analysis::{
    compute_correlation_matrix, compute_mi_matrix, run_feature_analysis, AnalysisSummary,
    FeatureAnalysisConfig, FeatureAnalysisResult, FeatureCluster, FeaturePairCorrelation,
    FeatureStats,
};

pub use final_decision::{
    run_final_decision, ConfidenceLevel, DecisionInput, FinalDecision, FinalDecisionResult,
    HonestAssessment, HypothesisSummary, NextSteps, StrategyEstimate,
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
