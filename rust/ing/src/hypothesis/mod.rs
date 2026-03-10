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

pub use stats::{
    pearson_correlation, spearman_correlation, mutual_information,
    t_test_correlation, bonferroni_correct, CorrelationResult,
    WalkForwardResult, walk_forward_correlation,
};

pub use h1_whale_flow::{
    H1TestResult, H1TestConfig, H1Decision,
    run_h1_whale_flow_test,
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
