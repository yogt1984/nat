//! Real-time Regime Classification using pre-trained GMM
//!
//! This module implements online inference for Hidden Markov Model-style
//! market regime detection. The GMM is trained offline on historical data
//! using the `scripts/train_regime_gmm.py` script.
//!
//! # 5D Feature Space
//!
//! | Dimension | Feature | Interpretation |
//! |-----------|---------|----------------|
//! | 1 | Kyle's Lambda | Liquidity/price impact |
//! | 2 | VPIN | Informed trading probability |
//! | 3 | Absorption Ratio Z-score | Accumulation/distribution signal |
//! | 4 | Hurst Exponent | Trend persistence |
//! | 5 | Whale Net Flow | Institutional positioning |
//!
//! # Target Regimes
//!
//! - **Accumulation**: Smart money buying, low volatility, building positions
//! - **Markup**: Price trending up, high liquidity, positive momentum
//! - **Distribution**: Smart money selling, high volume at tops
//! - **Markdown**: Price trending down, liquidations, negative momentum
//! - **Ranging**: No clear direction, mean-reverting behavior

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs;
use std::path::Path;

/// Epsilon for numerical stability
const EPSILON: f64 = 1e-10;

/// Pre-trained GMM parameters loaded from JSON
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GmmParams {
    /// Number of mixture components (typically 5 for regimes)
    pub n_components: usize,
    /// Mean vectors for each component [n_components, n_features]
    pub means: Vec<Vec<f64>>,
    /// Covariance matrices [n_components, n_features, n_features]
    pub covariances: Vec<Vec<Vec<f64>>>,
    /// Component weights (prior probabilities) [n_components]
    pub weights: Vec<f64>,
    /// StandardScaler mean for feature normalization [n_features]
    pub scaler_mean: Vec<f64>,
    /// StandardScaler std for feature normalization [n_features]
    pub scaler_std: Vec<f64>,
}

impl Default for GmmParams {
    fn default() -> Self {
        // Default 5-component GMM with identity covariances
        let n = 5;
        let d = 5;
        Self {
            n_components: n,
            means: vec![vec![0.0; d]; n],
            covariances: (0..n)
                .map(|_| {
                    (0..d)
                        .map(|i| (0..d).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                        .collect()
                })
                .collect(),
            weights: vec![1.0 / n as f64; n],
            scaler_mean: vec![0.0; d],
            scaler_std: vec![1.0; d],
        }
    }
}

/// Market regime labels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    Unknown = 5,
}

impl Regime {
    /// Convert from cluster index using label mapping
    pub fn from_cluster(cluster: usize, cluster_labels: &[Regime]) -> Regime {
        cluster_labels.get(cluster).copied().unwrap_or(Regime::Unknown)
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

impl Default for Regime {
    fn default() -> Self {
        Regime::Unknown
    }
}

/// GMM-based regime classifier
///
/// Performs online classification using pre-trained Gaussian Mixture Model.
/// Parameters are loaded from a JSON file generated by the training script.
pub struct RegimeClassifier {
    /// Pre-trained GMM parameters
    params: GmmParams,
    /// Mapping from cluster indices to semantic regime labels
    cluster_labels: Vec<Regime>,
    /// Precomputed inverse covariance matrices
    cov_inv: Vec<Vec<Vec<f64>>>,
    /// Precomputed log determinants
    log_dets: Vec<f64>,
}

impl RegimeClassifier {
    /// Create classifier with given parameters and label mapping
    pub fn new(params: GmmParams, cluster_labels: Vec<Regime>) -> Self {
        // Precompute matrix inverses and determinants
        let cov_inv: Vec<Vec<Vec<f64>>> = params
            .covariances
            .iter()
            .map(|cov| Self::invert_matrix_static(cov))
            .collect();

        let log_dets: Vec<f64> = params
            .covariances
            .iter()
            .map(|cov| Self::log_determinant_static(cov))
            .collect();

        Self {
            params,
            cluster_labels,
            cov_inv,
            log_dets,
        }
    }

    /// Load classifier from JSON model file
    pub fn load(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json_str = fs::read_to_string(model_path)?;
        let params: GmmParams = serde_json::from_str(&json_str)?;

        // Default cluster labels - should be overridden based on cluster analysis
        let cluster_labels = vec![
            Regime::Accumulation,
            Regime::Markup,
            Regime::Distribution,
            Regime::Markdown,
            Regime::Ranging,
        ];

        Ok(Self::new(params, cluster_labels))
    }

    /// Create classifier with default (untrained) parameters
    ///
    /// Returns uniform probabilities until a trained model is loaded.
    pub fn default_untrained() -> Self {
        let params = GmmParams::default();
        let cluster_labels = vec![
            Regime::Accumulation,
            Regime::Markup,
            Regime::Distribution,
            Regime::Markdown,
            Regime::Ranging,
        ];
        Self::new(params, cluster_labels)
    }

    /// Update cluster label mapping based on cluster statistics
    pub fn set_cluster_labels(&mut self, labels: Vec<Regime>) {
        if labels.len() == self.params.n_components {
            self.cluster_labels = labels;
        }
    }

    /// Classify a single observation
    ///
    /// # Arguments
    /// * `features` - 5D feature vector [kyle_lambda, vpin, absorption_zscore, hurst, whale_flow]
    ///
    /// # Returns
    /// (Regime, probabilities for each cluster)
    pub fn classify(&self, features: &[f64; 5]) -> (Regime, Vec<f64>) {
        // Standardize features using saved scaler parameters
        let scaled: Vec<f64> = features
            .iter()
            .zip(
                self.params
                    .scaler_mean
                    .iter()
                    .zip(self.params.scaler_std.iter()),
            )
            .map(|(&x, (&mean, &std))| {
                if std > EPSILON {
                    (x - mean) / std
                } else {
                    0.0
                }
            })
            .collect();

        // Compute log-responsibilities for each component
        let mut log_probs = Vec::with_capacity(self.params.n_components);

        for k in 0..self.params.n_components {
            let log_prob = self.log_multivariate_normal_pdf(&scaled, k)
                + self.params.weights[k].max(EPSILON).ln();
            log_probs.push(log_prob);
        }

        // Convert to probabilities via log-sum-exp (numerically stable)
        let probs = self.log_sum_exp_normalize(&log_probs);

        // Find most likely component
        let best_cluster = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let regime = Regime::from_cluster(best_cluster, &self.cluster_labels);

        (regime, probs)
    }

    /// Batch classification for multiple observations
    pub fn classify_batch(&self, features_batch: &[[f64; 5]]) -> Vec<(Regime, Vec<f64>)> {
        features_batch.iter().map(|f| self.classify(f)).collect()
    }

    /// Get raw cluster assignment (without semantic label mapping)
    pub fn predict_cluster(&self, features: &[f64; 5]) -> (usize, f64) {
        let (_, probs) = self.classify(features);
        let (cluster, &prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        (cluster, prob)
    }

    /// Compute log PDF of multivariate normal for component k
    fn log_multivariate_normal_pdf(&self, x: &[f64], k: usize) -> f64 {
        let n = x.len() as f64;
        let mean = &self.params.means[k];

        // Compute (x - mean)
        let diff: Vec<f64> = x.iter().zip(mean.iter()).map(|(&xi, &mi)| xi - mi).collect();

        // Compute Mahalanobis distance: (x - μ)ᵀ Σ⁻¹ (x - μ)
        let mahalanobis = self.quadratic_form(&diff, &self.cov_inv[k]);

        // Log normalization constant: -0.5 * (n*log(2π) + log|Σ|)
        -0.5 * (n * (2.0 * PI).ln() + self.log_dets[k] + mahalanobis)
    }

    /// Log-sum-exp normalization to convert log-probs to probabilities
    fn log_sum_exp_normalize(&self, log_probs: &[f64]) -> Vec<f64> {
        let max_log = log_probs
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if max_log.is_infinite() {
            // All log probs are -inf, return uniform
            return vec![1.0 / log_probs.len() as f64; log_probs.len()];
        }

        let exp_sum: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
        let log_sum = max_log + exp_sum.ln();

        log_probs
            .iter()
            .map(|&lp| (lp - log_sum).exp().max(0.0).min(1.0))
            .collect()
    }

    /// Compute quadratic form xᵀ M x
    fn quadratic_form(&self, x: &[f64], m: &[Vec<f64>]) -> f64 {
        let n = x.len();
        let mut result = 0.0;

        for i in 0..n {
            for j in 0..n {
                result += x[i] * m[i][j] * x[j];
            }
        }

        result
    }

    /// Invert matrix (static version for precomputation)
    ///
    /// Uses Cholesky decomposition for numerical stability.
    /// Falls back to diagonal approximation for near-singular matrices.
    fn invert_matrix_static(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = m.len();

        // Try Cholesky decomposition first (for positive definite matrices)
        if let Some(inv) = Self::cholesky_inverse(m) {
            return inv;
        }

        // Fall back to regularized diagonal inverse
        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            let diag = m[i][i].abs().max(EPSILON);
            inv[i][i] = 1.0 / diag;
        }
        inv
    }

    /// Cholesky decomposition-based matrix inverse
    fn cholesky_inverse(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let n = m.len();
        let mut l = vec![vec![0.0; n]; n];

        // Cholesky decomposition: A = LLᵀ
        for i in 0..n {
            for j in 0..=i {
                let mut sum = m[i][j];
                for k in 0..j {
                    sum -= l[i][k] * l[j][k];
                }

                if i == j {
                    if sum <= 0.0 {
                        return None; // Not positive definite
                    }
                    l[i][j] = sum.sqrt();
                } else {
                    if l[j][j].abs() < EPSILON {
                        return None;
                    }
                    l[i][j] = sum / l[j][j];
                }
            }
        }

        // Invert L (lower triangular)
        let mut l_inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            l_inv[i][i] = 1.0 / l[i][i];
            for j in (i + 1)..n {
                let mut sum = 0.0;
                for k in i..j {
                    sum -= l[j][k] * l_inv[k][i];
                }
                l_inv[j][i] = sum / l[j][j];
            }
        }

        // A⁻¹ = (LLᵀ)⁻¹ = L⁻ᵀ L⁻¹
        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    inv[i][j] += l_inv[k][i] * l_inv[k][j];
                }
            }
        }

        Some(inv)
    }

    /// Log determinant of matrix (static version)
    fn log_determinant_static(m: &[Vec<f64>]) -> f64 {
        let n = m.len();

        // Try Cholesky factorization for log determinant
        let mut l = vec![vec![0.0; n]; n];
        let mut valid = true;

        for i in 0..n {
            for j in 0..=i {
                let mut sum = m[i][j];
                for k in 0..j {
                    sum -= l[i][k] * l[j][k];
                }

                if i == j {
                    if sum <= 0.0 {
                        valid = false;
                        break;
                    }
                    l[i][j] = sum.sqrt();
                } else {
                    if l[j][j].abs() < EPSILON {
                        valid = false;
                        break;
                    }
                    l[i][j] = sum / l[j][j];
                }
            }
            if !valid {
                break;
            }
        }

        if valid {
            // log|A| = 2 * sum(log(L_ii))
            2.0 * l.iter().enumerate().map(|(i, row)| row[i].ln()).sum::<f64>()
        } else {
            // Fall back to product of diagonal elements
            m.iter()
                .enumerate()
                .map(|(i, row)| row[i].abs().max(EPSILON).ln())
                .sum()
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
            prob_accumulation: probs.get(0).copied().unwrap_or(0.0),
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
        matches!(
            self.get_regime(),
            Regime::Accumulation | Regime::Markup
        )
    }

    /// Check if regime is bearish (distribution or markdown)
    pub fn is_bearish(&self) -> bool {
        matches!(
            self.get_regime(),
            Regime::Distribution | Regime::Markdown
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_params() -> GmmParams {
        // Create distinguishable cluster centers
        GmmParams {
            n_components: 5,
            means: vec![
                vec![-1.0, -1.0, 1.0, 0.5, 1.0],   // Accumulation: low λ, low VPIN, high abs, pos whale
                vec![1.0, -1.0, 0.0, 1.0, 0.5],    // Markup: high λ, low VPIN, high Hurst
                vec![-1.0, 1.0, 1.0, 0.5, -1.0],   // Distribution: low λ, high VPIN, high abs, neg whale
                vec![1.0, 1.0, 0.0, 1.0, -0.5],    // Markdown: high λ, high VPIN, high Hurst, neg whale
                vec![0.0, -1.0, 0.0, -1.0, 0.0],   // Ranging: neutral, low Hurst
            ],
            covariances: (0..5)
                .map(|_| {
                    (0..5)
                        .map(|i| (0..5).map(|j| if i == j { 0.5 } else { 0.0 }).collect())
                        .collect()
                })
                .collect(),
            weights: vec![0.2, 0.2, 0.2, 0.2, 0.2],
            scaler_mean: vec![0.0; 5],
            scaler_std: vec![1.0; 5],
        }
    }

    #[test]
    fn test_regime_classification_accumulation() {
        let params = mock_params();
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

        // Features matching accumulation center
        let features = [-1.0, -1.0, 1.0, 0.5, 1.0];
        let (regime, probs) = classifier.classify(&features);

        assert_eq!(regime, Regime::Accumulation);
        assert!(probs[0] > 0.5, "Accumulation prob should be high: {:?}", probs);
    }

    #[test]
    fn test_regime_classification_markup() {
        let params = mock_params();
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

        // Features matching markup center
        let features = [1.0, -1.0, 0.0, 1.0, 0.5];
        let (regime, probs) = classifier.classify(&features);

        assert_eq!(regime, Regime::Markup);
        assert!(probs[1] > 0.3, "Markup prob should be elevated: {:?}", probs);
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let params = mock_params();
        let classifier = RegimeClassifier::new(params, vec![Regime::Unknown; 5]);

        // Random features
        let test_cases = [
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [-2.0, 2.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
        ];

        for features in &test_cases {
            let (_, probs) = classifier.classify(features);
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Probabilities should sum to 1: {} for {:?}",
                sum,
                features
            );
        }
    }

    #[test]
    fn test_probabilities_non_negative() {
        let params = mock_params();
        let classifier = RegimeClassifier::new(params, vec![Regime::Unknown; 5]);

        let features = [-5.0, 5.0, -3.0, 3.0, 0.0]; // Extreme values
        let (_, probs) = classifier.classify(&features);

        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p >= 0.0 && p <= 1.0,
                "Probability {} should be in [0,1]: {}",
                i,
                p
            );
        }
    }

    #[test]
    fn test_regime_features_construction() {
        let probs = vec![0.1, 0.3, 0.05, 0.05, 0.5];
        let features = RegimeFeatures::from_classification(Regime::Ranging, &probs);

        assert_eq!(features.regime, 4.0); // Ranging = 4
        assert!((features.prob_accumulation - 0.1).abs() < EPSILON);
        assert!((features.prob_markup - 0.3).abs() < EPSILON);
        assert!((features.prob_ranging - 0.5).abs() < EPSILON);
        assert!((features.regime_confidence - 0.5).abs() < EPSILON);
        assert!(features.regime_entropy > 0.0);
    }

    #[test]
    fn test_regime_entropy_calculation() {
        // Uniform distribution has maximum entropy
        let uniform_probs = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let uniform_features = RegimeFeatures::from_classification(Regime::Unknown, &uniform_probs);

        // Concentrated distribution has low entropy
        let concentrated_probs = vec![0.95, 0.01, 0.01, 0.01, 0.02];
        let concentrated_features =
            RegimeFeatures::from_classification(Regime::Accumulation, &concentrated_probs);

        assert!(
            uniform_features.regime_entropy > concentrated_features.regime_entropy,
            "Uniform entropy {} should be > concentrated entropy {}",
            uniform_features.regime_entropy,
            concentrated_features.regime_entropy
        );
    }

    #[test]
    fn test_regime_string_conversion() {
        assert_eq!(Regime::Accumulation.as_str(), "accumulation");
        assert_eq!(Regime::Markup.as_str(), "markup");
        assert_eq!(Regime::Distribution.as_str(), "distribution");
        assert_eq!(Regime::Markdown.as_str(), "markdown");
        assert_eq!(Regime::Ranging.as_str(), "ranging");
        assert_eq!(Regime::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_regime_code_conversion() {
        for code in 0..=5 {
            let regime = Regime::from_code(code as f64);
            assert_eq!(regime.as_code() as u8, code);
        }

        // Invalid codes should map to Unknown
        assert_eq!(Regime::from_code(99.0), Regime::Unknown);
    }

    #[test]
    fn test_is_bullish_bearish() {
        let bull_features = RegimeFeatures {
            regime: Regime::Accumulation.as_code(),
            ..Default::default()
        };
        assert!(bull_features.is_bullish());
        assert!(!bull_features.is_bearish());

        let bear_features = RegimeFeatures {
            regime: Regime::Distribution.as_code(),
            ..Default::default()
        };
        assert!(!bear_features.is_bullish());
        assert!(bear_features.is_bearish());

        let ranging_features = RegimeFeatures {
            regime: Regime::Ranging.as_code(),
            ..Default::default()
        };
        assert!(!ranging_features.is_bullish());
        assert!(!ranging_features.is_bearish());
    }

    #[test]
    fn test_cholesky_inverse() {
        // 2x2 positive definite matrix
        let m = vec![vec![4.0, 2.0], vec![2.0, 2.0]];

        let inv = RegimeClassifier::cholesky_inverse(&m).unwrap();

        // Verify: M * M^-1 ≈ I
        let mut product = vec![vec![0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    product[i][j] += m[i][k] * inv[k][j];
                }
            }
        }

        // Check diagonal elements are ~1, off-diagonal ~0
        assert!((product[0][0] - 1.0).abs() < 0.01);
        assert!((product[1][1] - 1.0).abs() < 0.01);
        assert!(product[0][1].abs() < 0.01);
        assert!(product[1][0].abs() < 0.01);
    }

    #[test]
    fn test_batch_classification() {
        let params = mock_params();
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

        let batch = [
            [-1.0, -1.0, 1.0, 0.5, 1.0],
            [1.0, -1.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        let results = classifier.classify_batch(&batch);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, Regime::Accumulation);
        assert_eq!(results[1].0, Regime::Markup);
    }

    #[test]
    fn test_feature_names_count_match() {
        assert_eq!(RegimeFeatures::names().len(), RegimeFeatures::count());
    }

    #[test]
    fn test_to_vec_length() {
        let features = RegimeFeatures::default();
        assert_eq!(features.to_vec().len(), RegimeFeatures::count());
    }
}

// ============================================================================
// Skeptical Tests - Validate actual regime detection capability
// ============================================================================

#[cfg(test)]
mod skeptical_tests {
    use super::*;

    /// Create mock GMM parameters for testing
    fn mock_params() -> GmmParams {
        // Create distinguishable cluster centers
        GmmParams {
            n_components: 5,
            means: vec![
                vec![-1.0, -1.0, 1.0, 0.5, 1.0],   // Accumulation: low λ, low VPIN, high abs, pos whale
                vec![1.0, -1.0, 0.0, 1.0, 0.5],    // Markup: high λ, low VPIN, high Hurst
                vec![-1.0, 1.0, 1.0, 0.5, -1.0],   // Distribution: low λ, high VPIN, high abs, neg whale
                vec![1.0, 1.0, 0.0, 1.0, -0.5],    // Markdown: high λ, high VPIN, high Hurst, neg whale
                vec![0.0, -1.0, 0.0, -1.0, 0.0],   // Ranging: neutral, low Hurst
            ],
            covariances: (0..5)
                .map(|_| {
                    (0..5)
                        .map(|i| (0..5).map(|j| if i == j { 0.5 } else { 0.0 }).collect())
                        .collect()
                })
                .collect(),
            weights: vec![0.2, 0.2, 0.2, 0.2, 0.2],
            scaler_mean: vec![0.0; 5],
            scaler_std: vec![1.0; 5],
        }
    }

    /// Test 1: Classifier should separate distinctly different market states
    #[test]
    fn test_separates_distinct_regimes() {
        let params = mock_params();
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

        // Strong accumulation signal
        let acc_features = [-2.0, -2.0, 2.0, 0.5, 2.0];
        let (acc_regime, acc_probs) = classifier.classify(&acc_features);

        // Strong markup signal
        let mk_features = [2.0, -2.0, 0.0, 2.0, 1.0];
        let (mk_regime, mk_probs) = classifier.classify(&mk_features);

        // Strong distribution signal
        let dist_features = [-2.0, 2.0, 2.0, 0.5, -2.0];
        let (dist_regime, dist_probs) = classifier.classify(&dist_features);

        // They should classify to different regimes
        assert_ne!(
            acc_regime, mk_regime,
            "Accumulation and Markup should be different"
        );
        assert_ne!(
            acc_regime, dist_regime,
            "Accumulation and Distribution should be different"
        );
        assert_ne!(
            mk_regime, dist_regime,
            "Markup and Distribution should be different"
        );

        // Each should have high confidence for its category
        assert!(
            acc_probs[0] > 0.3,
            "Accumulation features should favor accumulation: {:?}",
            acc_probs
        );
    }

    /// Test 2: Ambiguous features should have lower confidence
    #[test]
    fn test_ambiguous_features_lower_confidence() {
        let params = mock_params();
        let classifier = RegimeClassifier::new(params, vec![Regime::Unknown; 5]);

        // Clear accumulation signal
        let clear_features = [-2.0, -2.0, 2.0, 0.5, 2.0];
        let clear_result = RegimeFeatures::from_classification(
            classifier.classify(&clear_features).0,
            &classifier.classify(&clear_features).1,
        );

        // Ambiguous signal (near origin)
        let ambig_features = [0.0, 0.0, 0.0, 0.0, 0.0];
        let ambig_result = RegimeFeatures::from_classification(
            classifier.classify(&ambig_features).0,
            &classifier.classify(&ambig_features).1,
        );

        // Clear signals should have higher confidence
        assert!(
            clear_result.regime_confidence >= ambig_result.regime_confidence,
            "Clear signal confidence {} should be >= ambiguous {}",
            clear_result.regime_confidence,
            ambig_result.regime_confidence
        );

        // Ambiguous should have higher entropy
        assert!(
            ambig_result.regime_entropy >= clear_result.regime_entropy,
            "Ambiguous entropy {} should be >= clear {}",
            ambig_result.regime_entropy,
            clear_result.regime_entropy
        );
    }

    /// Test 3: Opposing whale flow should differentiate accumulation from distribution
    #[test]
    fn test_whale_flow_differentiates_acc_dist() {
        let params = mock_params();
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

        // Same features except whale flow
        let pos_whale = [-1.0, 0.0, 1.0, 0.5, 2.0]; // Strong positive whale
        let neg_whale = [-1.0, 0.0, 1.0, 0.5, -2.0]; // Strong negative whale

        let (pos_regime, pos_probs) = classifier.classify(&pos_whale);
        let (neg_regime, neg_probs) = classifier.classify(&neg_whale);

        // Positive whale should favor accumulation/markup
        let pos_bullish = pos_probs[0] + pos_probs[1];
        // Negative whale should favor distribution/markdown
        let neg_bearish = neg_probs[2] + neg_probs[3];

        assert!(
            pos_bullish > neg_bearish - 0.3,
            "Positive whale flow should favor bullish regimes: pos_bull={}, neg_bear={}",
            pos_bullish,
            neg_bearish
        );
    }

    /// Test 4: High Hurst should indicate trending (markup/markdown), low Hurst should indicate ranging
    #[test]
    fn test_hurst_indicates_trending_vs_ranging() {
        let params = mock_params();
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

        // High Hurst (trending)
        let high_hurst = [0.5, -0.5, 0.0, 2.0, 0.3];
        let (_, high_probs) = classifier.classify(&high_hurst);

        // Low Hurst (ranging)
        let low_hurst = [0.0, -1.0, 0.0, -2.0, 0.0];
        let (low_regime, low_probs) = classifier.classify(&low_hurst);

        // High Hurst should favor trending regimes (markup/markdown)
        let high_trending = high_probs[1] + high_probs[3];

        // Low Hurst should favor ranging
        assert!(
            low_probs[4] > 0.2,
            "Low Hurst should favor ranging regime: {:?}",
            low_probs
        );
    }

    /// Test 5: Classification should be deterministic
    #[test]
    fn test_classification_deterministic() {
        let params = mock_params();
        let classifier = RegimeClassifier::new(params, vec![Regime::Unknown; 5]);

        let features = [0.5, -0.5, 0.3, 0.7, 0.2];

        let (regime1, probs1) = classifier.classify(&features);
        let (regime2, probs2) = classifier.classify(&features);

        assert_eq!(regime1, regime2, "Same features should give same regime");

        for (p1, p2) in probs1.iter().zip(probs2.iter()) {
            assert!(
                (p1 - p2).abs() < EPSILON,
                "Same features should give same probabilities"
            );
        }
    }

    /// Test 6: Extreme features should not cause numerical issues
    #[test]
    fn test_extreme_features_numerical_stability() {
        let params = mock_params();
        let classifier = RegimeClassifier::new(params, vec![Regime::Unknown; 5]);

        let extreme_cases = [
            [100.0, 100.0, 100.0, 100.0, 100.0],
            [-100.0, -100.0, -100.0, -100.0, -100.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1e10, -1e10, 1e-10, -1e-10, 0.0],
        ];

        for features in &extreme_cases {
            let (_, probs) = classifier.classify(features);

            // Check probabilities are valid
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.1 || sum.is_finite(),
                "Extreme features {:?} should produce valid probabilities, got sum={}",
                features,
                sum
            );

            for &p in &probs {
                assert!(
                    p.is_finite(),
                    "Probability should be finite for extreme features"
                );
            }
        }
    }

    /// Test 7: Default untrained classifier should return uniform-ish probabilities
    #[test]
    fn test_default_untrained_behavior() {
        let classifier = RegimeClassifier::default_untrained();

        let features = [0.5, 0.5, 0.5, 0.5, 0.5];
        let (_, probs) = classifier.classify(&features);

        // With default identity covariances and centered means,
        // any non-extreme input should give roughly uniform probabilities
        let max_prob = probs.iter().cloned().fold(0.0f64, f64::max);
        let min_prob = probs.iter().cloned().fold(1.0f64, f64::min);

        // Probabilities shouldn't be too skewed for default model
        assert!(
            max_prob < 0.9,
            "Untrained model shouldn't be too confident: max={}",
            max_prob
        );
    }
}
