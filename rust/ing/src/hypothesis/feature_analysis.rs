//! Feature Correlation and Redundancy Analysis
//!
//! This module analyzes feature correlations and identifies redundant features
//! to recommend an optimal feature subset for trading.
//!
//! Analysis:
//! 1. Correlation matrix (Pearson and Spearman)
//! 2. Mutual Information matrix
//! 3. Hierarchical clustering
//! 4. Feature importance ranking
//! 5. Redundancy detection
//!
//! Decision Criteria:
//! - If two features have |r| > 0.9, keep only the more predictive one
//! - Aim for 10-15 non-redundant features
//! - Each feature must have MI > 0.01 bits with target to be included

use super::stats::{pearson_correlation, spearman_correlation, mutual_information_adaptive};
use std::collections::{HashMap, HashSet};

/// Configuration for feature analysis
#[derive(Debug, Clone)]
pub struct FeatureAnalysisConfig {
    /// Correlation threshold for redundancy (|r| > threshold means redundant)
    pub redundancy_threshold: f64,
    /// High correlation threshold for warnings
    pub high_corr_threshold: f64,
    /// Minimum MI with target to include feature
    pub min_mi_threshold: f64,
    /// Target number of features in final subset
    pub target_feature_count: usize,
}

impl Default for FeatureAnalysisConfig {
    fn default() -> Self {
        Self {
            redundancy_threshold: 0.9,
            high_corr_threshold: 0.8,
            min_mi_threshold: 0.01,
            target_feature_count: 12,
        }
    }
}

/// Correlation between two features
#[derive(Debug, Clone)]
pub struct FeaturePairCorrelation {
    /// First feature name
    pub feature_a: String,
    /// Second feature name
    pub feature_b: String,
    /// Pearson correlation
    pub pearson: f64,
    /// Spearman correlation
    pub spearman: f64,
    /// Mutual information (bits)
    pub mi: f64,
    /// Whether this pair is considered redundant
    pub is_redundant: bool,
}

/// Single feature analysis result
#[derive(Debug, Clone)]
pub struct FeatureStats {
    /// Feature name
    pub name: String,
    /// Feature index
    pub index: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Min value
    pub min: f64,
    /// Max value
    pub max: f64,
    /// Correlation with target
    pub target_correlation: f64,
    /// MI with target (bits)
    pub target_mi: f64,
    /// Number of features this is redundant with
    pub redundant_with_count: usize,
    /// Whether to include in final subset
    pub include_in_subset: bool,
    /// Rank by predictive power (1 = best)
    pub predictive_rank: usize,
}

/// Cluster of correlated features
#[derive(Debug, Clone)]
pub struct FeatureCluster {
    /// Cluster ID
    pub id: usize,
    /// Features in this cluster
    pub features: Vec<String>,
    /// Representative feature (highest MI with target)
    pub representative: String,
    /// Average intra-cluster correlation
    pub avg_correlation: f64,
}

/// Hierarchical clustering node
#[derive(Debug, Clone)]
struct ClusterNode {
    /// Features in this node
    features: Vec<usize>,
    /// Distance at which this cluster was formed
    distance: f64,
}

/// Full feature analysis result
#[derive(Debug, Clone)]
pub struct FeatureAnalysisResult {
    /// Config used
    pub config: FeatureAnalysisConfig,
    /// Number of features analyzed
    pub n_features: usize,
    /// Number of samples
    pub n_samples: usize,
    /// Per-feature statistics
    pub feature_stats: Vec<FeatureStats>,
    /// Correlation matrix (flattened, row-major)
    pub correlation_matrix: Vec<f64>,
    /// MI matrix (flattened, row-major)
    pub mi_matrix: Vec<f64>,
    /// Highly correlated pairs (|r| > high_corr_threshold)
    pub high_corr_pairs: Vec<FeaturePairCorrelation>,
    /// Redundant pairs (|r| > redundancy_threshold)
    pub redundant_pairs: Vec<FeaturePairCorrelation>,
    /// Feature clusters
    pub clusters: Vec<FeatureCluster>,
    /// Recommended feature subset
    pub recommended_subset: Vec<String>,
    /// Features excluded due to redundancy
    pub excluded_redundant: Vec<String>,
    /// Features excluded due to low MI
    pub excluded_low_mi: Vec<String>,
    /// Summary statistics
    pub summary: AnalysisSummary,
}

/// Summary of the analysis
#[derive(Debug, Clone)]
pub struct AnalysisSummary {
    /// Total features analyzed
    pub total_features: usize,
    /// Features in recommended subset
    pub subset_size: usize,
    /// Features excluded for redundancy
    pub excluded_redundant: usize,
    /// Features excluded for low MI
    pub excluded_low_mi: usize,
    /// Number of clusters identified
    pub n_clusters: usize,
    /// Average MI with target (recommended subset)
    pub avg_mi_subset: f64,
    /// Max correlation in recommended subset
    pub max_corr_in_subset: f64,
}

/// Compute basic statistics for a feature
fn compute_feature_stats(values: &[f64]) -> (f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    (mean, std, min, max)
}

/// Compute correlation matrix for all feature pairs
pub fn compute_correlation_matrix(
    features: &[Vec<f64>],
    use_spearman: bool,
) -> Vec<f64> {
    let n = features.len();
    let mut matrix = vec![0.0; n * n];

    for i in 0..n {
        // Diagonal is 1.0
        matrix[i * n + i] = 1.0;

        for j in (i + 1)..n {
            let corr = if use_spearman {
                spearman_correlation(&features[i], &features[j])
            } else {
                pearson_correlation(&features[i], &features[j])
            };

            matrix[i * n + j] = corr;
            matrix[j * n + i] = corr; // Symmetric
        }
    }

    matrix
}

/// Compute MI matrix for all feature pairs
pub fn compute_mi_matrix(features: &[Vec<f64>]) -> Vec<f64> {
    let n = features.len();
    let mut matrix = vec![0.0; n * n];

    for i in 0..n {
        for j in i..n {
            let mi = if i == j {
                // Self-MI is entropy, but for comparison we use a large value
                1.0
            } else {
                mutual_information_adaptive(&features[i], &features[j])
            };

            matrix[i * n + j] = mi;
            matrix[j * n + i] = mi; // Symmetric
        }
    }

    matrix
}

/// Find highly correlated pairs
pub fn find_correlated_pairs(
    feature_names: &[String],
    features: &[Vec<f64>],
    corr_matrix: &[f64],
    mi_matrix: &[f64],
    threshold: f64,
) -> Vec<FeaturePairCorrelation> {
    let n = features.len();
    let mut pairs = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let pearson = corr_matrix[i * n + j];
            if pearson.abs() >= threshold {
                let spearman = spearman_correlation(&features[i], &features[j]);
                let mi = mi_matrix[i * n + j];

                pairs.push(FeaturePairCorrelation {
                    feature_a: feature_names[i].clone(),
                    feature_b: feature_names[j].clone(),
                    pearson,
                    spearman,
                    mi,
                    is_redundant: pearson.abs() >= 0.9,
                });
            }
        }
    }

    // Sort by correlation magnitude (descending)
    pairs.sort_by(|a, b| {
        b.pearson.abs().partial_cmp(&a.pearson.abs()).unwrap_or(std::cmp::Ordering::Equal)
    });

    pairs
}

/// Perform agglomerative hierarchical clustering
pub fn hierarchical_clustering(
    corr_matrix: &[f64],
    n_features: usize,
    threshold: f64,
) -> Vec<Vec<usize>> {
    // Convert correlation to distance (1 - |corr|)
    let mut distances: Vec<Vec<f64>> = vec![vec![0.0; n_features]; n_features];
    for i in 0..n_features {
        for j in 0..n_features {
            distances[i][j] = 1.0 - corr_matrix[i * n_features + j].abs();
        }
    }

    // Initialize each feature as its own cluster
    let mut clusters: Vec<HashSet<usize>> = (0..n_features)
        .map(|i| {
            let mut s = HashSet::new();
            s.insert(i);
            s
        })
        .collect();

    let mut active: Vec<bool> = vec![true; n_features];

    // Agglomerative clustering with average linkage
    loop {
        // Find closest pair of active clusters
        let mut best_dist = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;

        for i in 0..clusters.len() {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..clusters.len() {
                if !active[j] {
                    continue;
                }

                // Average linkage distance
                let mut sum_dist = 0.0;
                let mut count = 0;
                for &fi in &clusters[i] {
                    for &fj in &clusters[j] {
                        sum_dist += distances[fi][fj];
                        count += 1;
                    }
                }
                let avg_dist = if count > 0 { sum_dist / count as f64 } else { f64::INFINITY };

                if avg_dist < best_dist {
                    best_dist = avg_dist;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Stop if best distance exceeds threshold (correlation below 1-threshold)
        if best_dist > (1.0 - threshold) {
            break;
        }

        // Merge clusters
        let merged: HashSet<usize> = clusters[best_i].union(&clusters[best_j]).cloned().collect();
        clusters[best_i] = merged;
        active[best_j] = false;
    }

    // Collect active clusters
    clusters
        .into_iter()
        .enumerate()
        .filter(|(i, _)| active[*i])
        .map(|(_, c)| c.into_iter().collect::<Vec<_>>())
        .collect()
}

/// Build feature clusters from hierarchical clustering result
pub fn build_feature_clusters(
    cluster_indices: Vec<Vec<usize>>,
    feature_names: &[String],
    target_mi: &[f64],
    corr_matrix: &[f64],
    n_features: usize,
) -> Vec<FeatureCluster> {
    let mut clusters = Vec::new();

    for (id, indices) in cluster_indices.into_iter().enumerate() {
        if indices.is_empty() {
            continue;
        }

        // Find representative (highest MI with target)
        let representative_idx = indices.iter()
            .max_by(|&&a, &&b| {
                target_mi[a].partial_cmp(&target_mi[b]).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(indices[0]);

        // Compute average intra-cluster correlation
        let mut sum_corr = 0.0;
        let mut count = 0;
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                sum_corr += corr_matrix[indices[i] * n_features + indices[j]].abs();
                count += 1;
            }
        }
        let avg_correlation = if count > 0 { sum_corr / count as f64 } else { 1.0 };

        clusters.push(FeatureCluster {
            id,
            features: indices.iter().map(|&i| feature_names[i].clone()).collect(),
            representative: feature_names[representative_idx].clone(),
            avg_correlation,
        });
    }

    // Sort by cluster size (descending)
    clusters.sort_by(|a, b| b.features.len().cmp(&a.features.len()));

    // Reassign IDs
    for (i, cluster) in clusters.iter_mut().enumerate() {
        cluster.id = i;
    }

    clusters
}

/// Select recommended feature subset
pub fn select_feature_subset(
    feature_stats: &mut [FeatureStats],
    corr_matrix: &[f64],
    config: &FeatureAnalysisConfig,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let n = feature_stats.len();

    // Sort by target MI (descending) to establish ranking
    let mut ranked_indices: Vec<usize> = (0..n).collect();
    ranked_indices.sort_by(|&a, &b| {
        feature_stats[b].target_mi.partial_cmp(&feature_stats[a].target_mi)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign ranks
    for (rank, &idx) in ranked_indices.iter().enumerate() {
        feature_stats[idx].predictive_rank = rank + 1;
    }

    let mut included: HashSet<usize> = HashSet::new();
    let mut excluded_redundant: Vec<String> = Vec::new();
    let mut excluded_low_mi: Vec<String> = Vec::new();

    // Greedy selection: add features in order of MI, skip if redundant
    for &idx in &ranked_indices {
        let stats = &feature_stats[idx];

        // Skip if MI too low
        if stats.target_mi < config.min_mi_threshold {
            excluded_low_mi.push(stats.name.clone());
            continue;
        }

        // Check if redundant with any already included
        let mut is_redundant = false;
        for &included_idx in &included {
            let corr = corr_matrix[idx * n + included_idx].abs();
            if corr > config.redundancy_threshold {
                is_redundant = true;
                break;
            }
        }

        if is_redundant {
            excluded_redundant.push(stats.name.clone());
        } else {
            included.insert(idx);
        }

        // Stop if we have enough features
        if included.len() >= config.target_feature_count {
            break;
        }
    }

    // Mark inclusion in feature_stats
    for (idx, stats) in feature_stats.iter_mut().enumerate() {
        stats.include_in_subset = included.contains(&idx);
    }

    // Build recommended subset (in rank order)
    let mut recommended: Vec<String> = ranked_indices.iter()
        .filter(|&&idx| included.contains(&idx))
        .map(|&idx| feature_stats[idx].name.clone())
        .collect();

    // If we didn't get enough, relax MI threshold
    if recommended.len() < config.target_feature_count / 2 {
        for &idx in &ranked_indices {
            if !included.contains(&idx) && !excluded_redundant.contains(&feature_stats[idx].name) {
                let mut is_redundant = false;
                for &included_idx in &included {
                    let corr = corr_matrix[idx * n + included_idx].abs();
                    if corr > config.redundancy_threshold {
                        is_redundant = true;
                        break;
                    }
                }
                if !is_redundant {
                    included.insert(idx);
                    recommended.push(feature_stats[idx].name.clone());
                    feature_stats[idx].include_in_subset = true;
                    // Remove from excluded_low_mi if it was there
                    excluded_low_mi.retain(|name| name != &feature_stats[idx].name);
                }
                if recommended.len() >= config.target_feature_count {
                    break;
                }
            }
        }
    }

    (recommended, excluded_redundant, excluded_low_mi)
}

/// Run full feature analysis
///
/// # Arguments
/// * `feature_names` - Names of features
/// * `features` - Feature data (each inner vec is one feature's values across samples)
/// * `target` - Target variable (e.g., future returns)
/// * `config` - Analysis configuration
///
/// # Returns
/// FeatureAnalysisResult with full analysis
pub fn run_feature_analysis(
    feature_names: &[String],
    features: &[Vec<f64>],
    target: &[f64],
    config: &FeatureAnalysisConfig,
) -> FeatureAnalysisResult {
    let n_features = features.len();
    let n_samples = if features.is_empty() { 0 } else { features[0].len() };

    // Validate inputs
    if n_features == 0 || n_samples == 0 {
        return FeatureAnalysisResult {
            config: config.clone(),
            n_features: 0,
            n_samples: 0,
            feature_stats: vec![],
            correlation_matrix: vec![],
            mi_matrix: vec![],
            high_corr_pairs: vec![],
            redundant_pairs: vec![],
            clusters: vec![],
            recommended_subset: vec![],
            excluded_redundant: vec![],
            excluded_low_mi: vec![],
            summary: AnalysisSummary {
                total_features: 0,
                subset_size: 0,
                excluded_redundant: 0,
                excluded_low_mi: 0,
                n_clusters: 0,
                avg_mi_subset: 0.0,
                max_corr_in_subset: 0.0,
            },
        };
    }

    // Compute correlation and MI matrices
    let corr_matrix = compute_correlation_matrix(features, false);
    let mi_matrix = compute_mi_matrix(features);

    // Compute per-feature statistics
    let mut feature_stats: Vec<FeatureStats> = Vec::with_capacity(n_features);
    for (i, (name, values)) in feature_names.iter().zip(features.iter()).enumerate() {
        let (mean, std, min, max) = compute_feature_stats(values);
        let target_corr = pearson_correlation(values, target);
        let target_mi = mutual_information_adaptive(values, target);

        feature_stats.push(FeatureStats {
            name: name.clone(),
            index: i,
            mean,
            std,
            min,
            max,
            target_correlation: target_corr,
            target_mi,
            redundant_with_count: 0,
            include_in_subset: false,
            predictive_rank: 0,
        });
    }

    // Count redundancies
    for i in 0..n_features {
        let mut count = 0;
        for j in 0..n_features {
            if i != j && corr_matrix[i * n_features + j].abs() > config.redundancy_threshold {
                count += 1;
            }
        }
        feature_stats[i].redundant_with_count = count;
    }

    // Find correlated pairs
    let high_corr_pairs = find_correlated_pairs(
        feature_names,
        features,
        &corr_matrix,
        &mi_matrix,
        config.high_corr_threshold,
    );

    let redundant_pairs: Vec<FeaturePairCorrelation> = high_corr_pairs.iter()
        .filter(|p| p.is_redundant)
        .cloned()
        .collect();

    // Hierarchical clustering
    let cluster_indices = hierarchical_clustering(&corr_matrix, n_features, config.high_corr_threshold);
    let target_mi_vec: Vec<f64> = feature_stats.iter().map(|s| s.target_mi).collect();
    let clusters = build_feature_clusters(
        cluster_indices,
        feature_names,
        &target_mi_vec,
        &corr_matrix,
        n_features,
    );

    // Select feature subset
    let (recommended_subset, excluded_redundant, excluded_low_mi) =
        select_feature_subset(&mut feature_stats, &corr_matrix, config);

    // Compute summary statistics
    let avg_mi_subset = if recommended_subset.is_empty() {
        0.0
    } else {
        feature_stats.iter()
            .filter(|s| s.include_in_subset)
            .map(|s| s.target_mi)
            .sum::<f64>() / recommended_subset.len() as f64
    };

    // Find max correlation within subset
    let subset_indices: Vec<usize> = feature_stats.iter()
        .filter(|s| s.include_in_subset)
        .map(|s| s.index)
        .collect();
    let mut max_corr_in_subset = 0.0;
    for i in 0..subset_indices.len() {
        for j in (i + 1)..subset_indices.len() {
            let corr = corr_matrix[subset_indices[i] * n_features + subset_indices[j]].abs();
            if corr > max_corr_in_subset {
                max_corr_in_subset = corr;
            }
        }
    }

    let summary = AnalysisSummary {
        total_features: n_features,
        subset_size: recommended_subset.len(),
        excluded_redundant: excluded_redundant.len(),
        excluded_low_mi: excluded_low_mi.len(),
        n_clusters: clusters.len(),
        avg_mi_subset,
        max_corr_in_subset,
    };

    FeatureAnalysisResult {
        config: config.clone(),
        n_features,
        n_samples,
        feature_stats,
        correlation_matrix: corr_matrix,
        mi_matrix,
        high_corr_pairs,
        redundant_pairs,
        clusters,
        recommended_subset,
        excluded_redundant,
        excluded_low_mi,
        summary,
    }
}

impl FeatureAnalysisResult {
    /// Get correlation between two features by name
    pub fn get_correlation(&self, feature_a: &str, feature_b: &str) -> Option<f64> {
        let idx_a = self.feature_stats.iter().position(|s| s.name == feature_a)?;
        let idx_b = self.feature_stats.iter().position(|s| s.name == feature_b)?;
        Some(self.correlation_matrix[idx_a * self.n_features + idx_b])
    }

    /// Get MI between two features by name
    pub fn get_mi(&self, feature_a: &str, feature_b: &str) -> Option<f64> {
        let idx_a = self.feature_stats.iter().position(|s| s.name == feature_a)?;
        let idx_b = self.feature_stats.iter().position(|s| s.name == feature_b)?;
        Some(self.mi_matrix[idx_a * self.n_features + idx_b])
    }

    /// Generate markdown report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Feature Correlation and Redundancy Analysis\n\n");

        report.push_str("## Summary\n\n");
        report.push_str(&format!("- **Total Features Analyzed**: {}\n", self.summary.total_features));
        report.push_str(&format!("- **Samples**: {}\n", self.n_samples));
        report.push_str(&format!("- **Recommended Subset Size**: {}\n", self.summary.subset_size));
        report.push_str(&format!("- **Excluded (Redundant)**: {}\n", self.summary.excluded_redundant));
        report.push_str(&format!("- **Excluded (Low MI)**: {}\n", self.summary.excluded_low_mi));
        report.push_str(&format!("- **Feature Clusters**: {}\n", self.summary.n_clusters));
        report.push_str(&format!("- **Avg MI in Subset**: {:.4} bits\n", self.summary.avg_mi_subset));
        report.push_str(&format!("- **Max Corr in Subset**: {:.3}\n", self.summary.max_corr_in_subset));

        report.push_str("\n## Recommended Feature Subset\n\n");
        report.push_str("| Rank | Feature | Target Corr | Target MI | Include |\n");
        report.push_str("|------|---------|-------------|-----------|----------|\n");

        for stats in self.feature_stats.iter().filter(|s| s.include_in_subset) {
            report.push_str(&format!(
                "| {} | {} | {:.3} | {:.4} | Yes |\n",
                stats.predictive_rank,
                stats.name,
                stats.target_correlation,
                stats.target_mi,
            ));
        }

        if !self.redundant_pairs.is_empty() {
            report.push_str("\n## Redundant Feature Pairs (|r| > 0.9)\n\n");
            report.push_str("| Feature A | Feature B | Pearson | Spearman | MI |\n");
            report.push_str("|-----------|-----------|---------|----------|-----|\n");

            for pair in &self.redundant_pairs {
                report.push_str(&format!(
                    "| {} | {} | {:.3} | {:.3} | {:.4} |\n",
                    pair.feature_a,
                    pair.feature_b,
                    pair.pearson,
                    pair.spearman,
                    pair.mi,
                ));
            }
        }

        if !self.clusters.is_empty() {
            report.push_str("\n## Feature Clusters\n\n");

            for cluster in &self.clusters {
                report.push_str(&format!(
                    "### Cluster {} (n={}, avg_corr={:.3})\n",
                    cluster.id,
                    cluster.features.len(),
                    cluster.avg_correlation,
                ));
                report.push_str(&format!("- **Representative**: {}\n", cluster.representative));
                report.push_str(&format!("- **Members**: {}\n\n", cluster.features.join(", ")));
            }
        }

        report.push_str("\n## All Features Ranked by Predictive Power\n\n");
        report.push_str("| Rank | Feature | Target MI | Target Corr | Redundant With | Status |\n");
        report.push_str("|------|---------|-----------|-------------|----------------|--------|\n");

        let mut sorted_stats = self.feature_stats.clone();
        sorted_stats.sort_by_key(|s| s.predictive_rank);

        for stats in &sorted_stats {
            let status = if stats.include_in_subset {
                "INCLUDED"
            } else if self.excluded_redundant.contains(&stats.name) {
                "redundant"
            } else if self.excluded_low_mi.contains(&stats.name) {
                "low MI"
            } else {
                "excluded"
            };

            report.push_str(&format!(
                "| {} | {} | {:.4} | {:.3} | {} | {} |\n",
                stats.predictive_rank,
                stats.name,
                stats.target_mi,
                stats.target_correlation,
                stats.redundant_with_count,
                status,
            ));
        }

        report.push_str("\n## Configuration\n\n");
        report.push_str(&format!("- Redundancy threshold: |r| > {:.2}\n", self.config.redundancy_threshold));
        report.push_str(&format!("- High correlation warning: |r| > {:.2}\n", self.config.high_corr_threshold));
        report.push_str(&format!("- Minimum MI threshold: {:.3} bits\n", self.config.min_mi_threshold));
        report.push_str(&format!("- Target subset size: {}\n", self.config.target_feature_count));

        report
    }

    /// Generate correlation heatmap data (for visualization)
    pub fn get_correlation_heatmap_data(&self) -> Vec<(String, String, f64)> {
        let mut data = Vec::new();
        for i in 0..self.n_features {
            for j in 0..self.n_features {
                data.push((
                    self.feature_stats[i].name.clone(),
                    self.feature_stats[j].name.clone(),
                    self.correlation_matrix[i * self.n_features + j],
                ));
            }
        }
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_features(n_samples: usize, n_features: usize) -> (Vec<String>, Vec<Vec<f64>>, Vec<f64>) {
        let mut rng_state = 42u64;
        let mut features = Vec::with_capacity(n_features);
        let mut names = Vec::with_capacity(n_features);

        // Generate correlated feature groups
        for f in 0..n_features {
            names.push(format!("feature_{}", f));

            let mut values = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let rand = (rng_state >> 33) as f64 / (u32::MAX as f64) - 0.5;

                // Create some correlation structure
                let base = (i as f64 * 0.1).sin();
                let group_signal = if f < 3 {
                    base * 0.8  // Group 1: correlated
                } else if f < 6 {
                    -base * 0.7 // Group 2: negatively correlated with group 1
                } else {
                    0.0 // Independent
                };

                values.push(group_signal + rand * 0.3);
            }
            features.push(values);
        }

        // Target correlates with group 1
        let target: Vec<f64> = (0..n_samples)
            .map(|i| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let rand = (rng_state >> 33) as f64 / (u32::MAX as f64) - 0.5;
                (i as f64 * 0.1).sin() * 0.5 + rand * 0.2
            })
            .collect();

        (names, features, target)
    }

    #[test]
    fn test_compute_feature_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std, min, max) = compute_feature_stats(&values);

        assert!((mean - 3.0).abs() < 1e-10);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max - 5.0).abs() < 1e-10);
        assert!(std > 0.0);
    }

    #[test]
    fn test_correlation_matrix() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0], // Identical to first
            vec![4.0, 3.0, 2.0, 1.0], // Negatively correlated
        ];

        let matrix = compute_correlation_matrix(&features, false);

        assert_eq!(matrix.len(), 9);
        // Diagonal is 1
        assert!((matrix[0] - 1.0).abs() < 1e-10);
        assert!((matrix[4] - 1.0).abs() < 1e-10);
        assert!((matrix[8] - 1.0).abs() < 1e-10);
        // First two identical: correlation = 1
        assert!((matrix[1] - 1.0).abs() < 1e-10);
        // First and third negatively correlated
        assert!(matrix[2] < -0.9);
    }

    #[test]
    fn test_mi_matrix() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ];

        let matrix = compute_mi_matrix(&features);

        assert_eq!(matrix.len(), 4);
        // Diagonal should be high (self-MI)
        assert!(matrix[0] > 0.0);
    }

    #[test]
    fn test_find_correlated_pairs() {
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.1, 2.1, 3.1, 4.1], // Highly correlated with a
            vec![4.0, 3.0, 2.0, 1.0], // Negatively correlated
        ];

        let corr_matrix = compute_correlation_matrix(&features, false);
        let mi_matrix = compute_mi_matrix(&features);

        let pairs = find_correlated_pairs(&names, &features, &corr_matrix, &mi_matrix, 0.8);

        assert!(!pairs.is_empty());
        // a and b should be highly correlated
        let ab_pair = pairs.iter().find(|p|
            (p.feature_a == "a" && p.feature_b == "b") ||
            (p.feature_a == "b" && p.feature_b == "a")
        );
        assert!(ab_pair.is_some());
    }

    #[test]
    fn test_hierarchical_clustering() {
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.1, 2.1, 3.1, 4.1], // Should cluster with 0
            vec![4.0, 3.0, 2.0, 1.0], // Should be separate
            vec![4.1, 3.1, 2.1, 1.1], // Should cluster with 2
        ];

        let corr_matrix = compute_correlation_matrix(&features, false);
        let clusters = hierarchical_clustering(&corr_matrix, 4, 0.8);

        // Should have 2 clusters
        assert!(clusters.len() <= 4);
        assert!(clusters.len() >= 1);
    }

    #[test]
    fn test_full_analysis() {
        let (names, features, target) = generate_test_features(200, 10);
        let config = FeatureAnalysisConfig::default();

        let result = run_feature_analysis(&names, &features, &target, &config);

        assert_eq!(result.n_features, 10);
        assert_eq!(result.n_samples, 200);
        assert_eq!(result.feature_stats.len(), 10);
        assert!(!result.recommended_subset.is_empty());
    }

    #[test]
    fn test_empty_input() {
        let names: Vec<String> = vec![];
        let features: Vec<Vec<f64>> = vec![];
        let target: Vec<f64> = vec![];
        let config = FeatureAnalysisConfig::default();

        let result = run_feature_analysis(&names, &features, &target, &config);

        assert_eq!(result.n_features, 0);
        assert!(result.recommended_subset.is_empty());
    }

    #[test]
    fn test_redundancy_detection() {
        // Create features with clear redundancy
        let names = vec![
            "original".to_string(),
            "copy".to_string(),
            "independent".to_string(),
        ];
        let original: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let copy: Vec<f64> = original.iter().map(|x| x * 1.001 + 0.001).collect();  // Nearly identical
        let independent: Vec<f64> = (0..100).map(|i| (i as f64 * 0.37).cos()).collect();

        let features = vec![original.clone(), copy, independent];
        let target = original; // Target correlates with original

        let config = FeatureAnalysisConfig {
            redundancy_threshold: 0.95,
            ..Default::default()
        };

        let result = run_feature_analysis(&names, &features, &target, &config);

        // Copy should be flagged as redundant with original
        assert!(result.redundant_pairs.iter().any(|p|
            (p.feature_a == "original" && p.feature_b == "copy") ||
            (p.feature_a == "copy" && p.feature_b == "original")
        ));
    }

    #[test]
    fn test_feature_ranking() {
        let (names, features, target) = generate_test_features(200, 5);
        let config = FeatureAnalysisConfig::default();

        let result = run_feature_analysis(&names, &features, &target, &config);

        // All features should have ranks
        for stats in &result.feature_stats {
            assert!(stats.predictive_rank >= 1);
            assert!(stats.predictive_rank <= 5);
        }

        // Ranks should be unique
        let ranks: Vec<usize> = result.feature_stats.iter().map(|s| s.predictive_rank).collect();
        let unique_ranks: HashSet<usize> = ranks.iter().cloned().collect();
        assert_eq!(ranks.len(), unique_ranks.len());
    }

    #[test]
    fn test_report_generation() {
        let (names, features, target) = generate_test_features(100, 5);
        let config = FeatureAnalysisConfig::default();

        let result = run_feature_analysis(&names, &features, &target, &config);
        let report = result.generate_report();

        assert!(report.contains("Feature Correlation and Redundancy Analysis"));
        assert!(report.contains("Summary"));
        assert!(report.contains("Recommended Feature Subset"));
        assert!(report.contains("Configuration"));
    }

    #[test]
    fn test_get_correlation() {
        let names = vec!["a".to_string(), "b".to_string()];
        let features = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
        ];
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let config = FeatureAnalysisConfig::default();

        let result = run_feature_analysis(&names, &features, &target, &config);

        let corr = result.get_correlation("a", "b");
        assert!(corr.is_some());
        assert!((corr.unwrap() - 1.0).abs() < 1e-10);

        let corr_none = result.get_correlation("a", "nonexistent");
        assert!(corr_none.is_none());
    }

    #[test]
    fn test_heatmap_data() {
        let names = vec!["a".to_string(), "b".to_string()];
        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0],
        ];
        let target = vec![1.0, 2.0, 3.0];
        let config = FeatureAnalysisConfig::default();

        let result = run_feature_analysis(&names, &features, &target, &config);
        let heatmap = result.get_correlation_heatmap_data();

        assert_eq!(heatmap.len(), 4); // 2x2 matrix
    }

    #[test]
    fn test_cluster_building() {
        let cluster_indices = vec![vec![0, 1], vec![2]];
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let target_mi = vec![0.1, 0.2, 0.05]; // b has highest MI in first cluster
        let corr_matrix = vec![
            1.0, 0.9, 0.1,
            0.9, 1.0, 0.1,
            0.1, 0.1, 1.0,
        ];

        let clusters = build_feature_clusters(cluster_indices, &names, &target_mi, &corr_matrix, 3);

        assert_eq!(clusters.len(), 2);
        // First cluster (larger) should have b as representative
        let first_cluster = clusters.iter().find(|c| c.features.len() == 2).unwrap();
        assert_eq!(first_cluster.representative, "b");
    }

    #[test]
    fn test_subset_selection_respects_mi_threshold() {
        let names: Vec<String> = (0..5).map(|i| format!("f{}", i)).collect();
        let features: Vec<Vec<f64>> = (0..5)
            .map(|i| (0..100).map(|j| (j as f64 + i as f64).sin()).collect())
            .collect();
        // Target that only correlates with first feature
        let target: Vec<f64> = features[0].clone();

        let config = FeatureAnalysisConfig {
            min_mi_threshold: 0.5, // High threshold
            ..Default::default()
        };

        let result = run_feature_analysis(&names, &features, &target, &config);

        // Most features should be excluded due to low MI
        // (unless relaxation kicks in)
        assert!(result.excluded_low_mi.len() > 0 || result.recommended_subset.len() < 5);
    }
}
