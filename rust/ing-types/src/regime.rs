//! Regime classification types
//!
//! Pure data types for GMM-based market regime classification output.
//! The classifier itself lives in `ing::ml::regime`; these types are
//! shared across `ing-features` and `ing`.

use serde::{Deserialize, Serialize};

const EPSILON: f64 = 1e-10;

/// Market regime labels
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Regime {
    /// Smart money accumulation phase
    Accumulation = 0,
    /// Price markup / trending up
    Markup = 1,
    /// Smart money distribution phase
    Distribution = 2,
    /// Price markdown / trending down
    Markdown = 3,
    /// Range-bound / consolidating
    Ranging = 4,
    /// Classification failed or insufficient data
    #[default]
    Unknown = 5,
}

impl Regime {
    /// Convert from cluster index using label mapping
    pub fn from_cluster(cluster: usize, cluster_labels: &[Regime]) -> Regime {
        cluster_labels
            .get(cluster)
            .copied()
            .unwrap_or(Regime::Unknown)
    }

    /// String representation for logging/display
    pub fn as_str(&self) -> &'static str {
        match self {
            Regime::Accumulation => "accumulation",
            Regime::Markup => "markup",
            Regime::Distribution => "distribution",
            Regime::Markdown => "markdown",
            Regime::Ranging => "ranging",
            Regime::Unknown => "unknown",
        }
    }

    /// Numeric code for Parquet storage
    pub fn as_code(&self) -> f64 {
        *self as u8 as f64
    }

    /// Convert from numeric code
    pub fn from_code(code: f64) -> Self {
        match code as u8 {
            0 => Regime::Accumulation,
            1 => Regime::Markup,
            2 => Regime::Distribution,
            3 => Regime::Markdown,
            4 => Regime::Ranging,
            _ => Regime::Unknown,
        }
    }

    /// Expected characteristics for each regime
    pub fn expected_characteristics(&self) -> &'static str {
        match self {
            Regime::Accumulation => "Low λ, low VPIN, high absorption, positive whale flow",
            Regime::Markup => "High λ, low VPIN, high Hurst, positive momentum",
            Regime::Distribution => "Low λ, high VPIN, high absorption, negative whale flow",
            Regime::Markdown => "High λ, high VPIN, high Hurst, negative momentum",
            Regime::Ranging => "Moderate λ, low VPIN, low Hurst, neutral flow",
            Regime::Unknown => "Insufficient data for classification",
        }
    }
}

/// Regime classification output features for Parquet storage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegimeFeatures {
    /// Current regime (encoded as f64: 0-5)
    pub regime: f64,
    /// Probability of accumulation regime
    pub prob_accumulation: f64,
    /// Probability of markup regime
    pub prob_markup: f64,
    /// Probability of distribution regime
    pub prob_distribution: f64,
    /// Probability of markdown regime
    pub prob_markdown: f64,
    /// Probability of ranging regime
    pub prob_ranging: f64,
    /// Classification confidence (max probability)
    pub regime_confidence: f64,
    /// Regime entropy (uncertainty measure)
    pub regime_entropy: f64,
}

impl RegimeFeatures {
    /// Number of features in this struct
    pub fn count() -> usize {
        8
    }

    /// Feature names for Parquet schema
    pub fn names() -> Vec<&'static str> {
        vec![
            "regime",
            "regime_prob_accumulation",
            "regime_prob_markup",
            "regime_prob_distribution",
            "regime_prob_markdown",
            "regime_prob_ranging",
            "regime_confidence",
            "regime_entropy",
        ]
    }

    /// Convert to vector for Parquet row
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.regime,
            self.prob_accumulation,
            self.prob_markup,
            self.prob_distribution,
            self.prob_markdown,
            self.prob_ranging,
            self.regime_confidence,
            self.regime_entropy,
        ]
    }

    /// Create from classification result
    pub fn from_classification(regime: Regime, probs: &[f64]) -> Self {
        let confidence = probs.iter().cloned().fold(0.0f64, f64::max);

        // Shannon entropy: H = -Σ p_i * log(p_i)
        let entropy: f64 = -probs
            .iter()
            .filter(|&&p| p > EPSILON)
            .map(|&p| p * p.ln())
            .sum::<f64>();

        Self {
            regime: regime.as_code(),
            prob_accumulation: probs.first().copied().unwrap_or(0.0),
            prob_markup: probs.get(1).copied().unwrap_or(0.0),
            prob_distribution: probs.get(2).copied().unwrap_or(0.0),
            prob_markdown: probs.get(3).copied().unwrap_or(0.0),
            prob_ranging: probs.get(4).copied().unwrap_or(0.0),
            regime_confidence: confidence,
            regime_entropy: entropy,
        }
    }

    /// Get regime as enum
    pub fn get_regime(&self) -> Regime {
        Regime::from_code(self.regime)
    }

    /// Check if classification is confident (above threshold)
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.regime_confidence >= threshold
    }

    /// Check if regime is bullish (accumulation or markup)
    pub fn is_bullish(&self) -> bool {
        matches!(self.get_regime(), Regime::Accumulation | Regime::Markup)
    }

    /// Check if regime is bearish (distribution or markdown)
    pub fn is_bearish(&self) -> bool {
        matches!(self.get_regime(), Regime::Distribution | Regime::Markdown)
    }
}

/// Type alias for backward compatibility
pub type GmmClassificationFeatures = RegimeFeatures;
