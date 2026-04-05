# Statistical Analysis Framework: Data-First Algorithmic Research

**Status:** Strategic Foundation
**Created:** 2026-04-05
**Purpose:** Define statistical features to compute BEFORE algorithm design
**Core Principle:** Data dictates algorithms, not the reverse

---

## Executive Summary

**The Fundamental Shift:**

```
WRONG (Previous Approach):
Hypothesis → Algorithm Design → Data Collection → Validation
"Low entropy means trending" → Design entropy-gated strategy → Collect data → Test

RIGHT (This Approach):
Data Collection → Statistical Analysis → Pattern Discovery → Hypothesis Formation → Algorithm Design
Collect data → Analyze entropy distribution → Discover natural clusters → Form testable hypotheses → Design algorithms
```

This document specifies the **statistical features to compute** before any algorithmic work begins. The analysis is organized hierarchically, starting with **entropy as the primary organizing principle**, then analyzing properties **within each entropy cluster**.

---

## Part I: Hierarchical Statistical Analysis Architecture

### The Hierarchy

```
LEVEL 0: RAW DATA INGESTION
    │
    ▼
LEVEL 1: ENTROPY DISTRIBUTION ANALYSIS (Primary Clustering)
    │   - How is entropy distributed?
    │   - What are the natural clusters?
    │   - How persistent are entropy regimes?
    │
    ▼
LEVEL 2: WITHIN-CLUSTER CHARACTERIZATION
    │   For each entropy cluster:
    │   - Volatility characteristics
    │   - Trend continuity
    │   - Feature correlations
    │   - Return distributions
    │
    ▼
LEVEL 3: CROSS-CLUSTER DYNAMICS
    │   - Transition probabilities
    │   - Change-point detection
    │   - Lead-lag relationships
    │
    ▼
LEVEL 4: PREDICTIVE RELATIONSHIP ANALYSIS
    │   - Which features lead returns?
    │   - Cluster-conditional predictability
    │   - Causal validation
    │
    ▼
LEVEL 5: DIMENSIONALITY REDUCTION & FEATURE SELECTION
    │   - PCA within clusters
    │   - Redundancy elimination
    │   - Stable feature identification
    │
    ▼
OUTPUT: Statistical Summary Database + Hypothesis Generation
```

---

## Part II: Level 1 - Entropy Distribution Analysis

### 1.1 Primary Entropy Statistics

**Before any threshold can be set, we MUST know:**

```python
class EntropyDistributionAnalysis:
    """
    FIRST THING TO COMPUTE: What does entropy actually look like?
    """

    def __init__(self, entropy_series, timeframe='15m'):
        self.entropy = entropy_series
        self.timeframe = timeframe

    def compute_basic_distribution(self):
        """
        Basic distributional properties
        """
        return {
            # Central tendency
            'mean': self.entropy.mean(),
            'median': self.entropy.median(),
            'mode': self._estimate_mode(),

            # Dispersion
            'std': self.entropy.std(),
            'iqr': self.entropy.quantile(0.75) - self.entropy.quantile(0.25),
            'range': self.entropy.max() - self.entropy.min(),

            # Shape
            'skewness': self.entropy.skew(),
            'kurtosis': self.entropy.kurtosis(),

            # Percentiles (CRITICAL for threshold selection)
            'percentiles': {
                'p5': self.entropy.quantile(0.05),
                'p10': self.entropy.quantile(0.10),
                'p25': self.entropy.quantile(0.25),
                'p50': self.entropy.quantile(0.50),
                'p75': self.entropy.quantile(0.75),
                'p90': self.entropy.quantile(0.90),
                'p95': self.entropy.quantile(0.95)
            }
        }

    def detect_natural_clusters(self):
        """
        Find natural clusters in entropy - DON'T ASSUME 0.3/0.7

        Methods:
        1. Gaussian Mixture Model (GMM) with BIC selection
        2. Kernel Density Estimation (KDE) mode detection
        3. Dip test for multimodality
        """
        results = {}

        # Method 1: GMM with optimal component selection
        from sklearn.mixture import GaussianMixture

        bic_scores = []
        for n_components in range(1, 6):
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(self.entropy.values.reshape(-1, 1))
            bic_scores.append({
                'n_components': n_components,
                'bic': gmm.bic(self.entropy.values.reshape(-1, 1)),
                'aic': gmm.aic(self.entropy.values.reshape(-1, 1))
            })

        optimal_n = min(bic_scores, key=lambda x: x['bic'])['n_components']

        # Fit optimal GMM
        gmm = GaussianMixture(n_components=optimal_n, random_state=42)
        gmm.fit(self.entropy.values.reshape(-1, 1))

        results['gmm'] = {
            'optimal_n_clusters': optimal_n,
            'cluster_means': sorted(gmm.means_.flatten()),
            'cluster_stds': [np.sqrt(v[0][0]) for v in gmm.covariances_],
            'cluster_weights': gmm.weights_.tolist(),
            'bic_scores': bic_scores
        }

        # Method 2: KDE mode detection
        from scipy.signal import find_peaks
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(self.entropy.dropna())
        x = np.linspace(self.entropy.min(), self.entropy.max(), 1000)
        density = kde(x)

        peaks, properties = find_peaks(density, prominence=0.1)
        modes = x[peaks]

        results['kde'] = {
            'n_modes': len(modes),
            'mode_locations': modes.tolist(),
            'mode_densities': density[peaks].tolist()
        }

        # Method 3: Dip test for unimodality
        from diptest import diptest
        dip_stat, dip_pvalue = diptest(self.entropy.dropna().values)

        results['dip_test'] = {
            'statistic': dip_stat,
            'p_value': dip_pvalue,
            'is_multimodal': dip_pvalue < 0.05
        }

        return results

    def analyze_temporal_properties(self):
        """
        How does entropy evolve over time?
        """
        return {
            # Autocorrelation structure
            'acf': self._compute_acf(nlags=100),
            'pacf': self._compute_pacf(nlags=50),
            'acf_half_life': self._compute_acf_half_life(),

            # Stationarity
            'adf_statistic': self._adf_test(),
            'adf_pvalue': self._adf_test(return_pvalue=True),
            'kpss_statistic': self._kpss_test(),
            'is_stationary': self._is_stationary(),

            # Long memory
            'hurst_exponent': self._hurst_exponent(),

            # Regime persistence
            'avg_regime_duration': self._avg_regime_duration(),
            'regime_duration_distribution': self._regime_duration_dist()
        }

    def _compute_acf_half_life(self):
        """Time for autocorrelation to decay to 0.5"""
        import statsmodels.api as sm
        acf = sm.tsa.acf(self.entropy.dropna(), nlags=100)
        half_life_idx = np.where(acf < 0.5)[0]
        return half_life_idx[0] if len(half_life_idx) > 0 else ">100"

    def _hurst_exponent(self, max_lag=100):
        """R/S analysis for persistence measurement"""
        series = self.entropy.dropna().values
        lags = range(2, min(max_lag, len(series)//4))
        rs_values = []

        for lag in lags:
            rs = self._rs_statistic(series, lag)
            if rs > 0:
                rs_values.append(rs)
            else:
                rs_values.append(np.nan)

        # Fit log(R/S) = H * log(n) + c
        valid_idx = ~np.isnan(rs_values)
        if sum(valid_idx) < 10:
            return None

        log_lags = np.log(list(lags))[valid_idx]
        log_rs = np.log(np.array(rs_values)[valid_idx])

        H, _ = np.polyfit(log_lags, log_rs, 1)
        return H

    def _avg_regime_duration(self, threshold_percentile=25):
        """How long do low/high entropy regimes last?"""
        low_thresh = self.entropy.quantile(threshold_percentile/100)
        high_thresh = self.entropy.quantile(1 - threshold_percentile/100)

        low_mask = self.entropy < low_thresh
        high_mask = self.entropy > high_thresh

        return {
            'low_entropy_avg_duration': self._compute_run_lengths(low_mask).mean(),
            'high_entropy_avg_duration': self._compute_run_lengths(high_mask).mean(),
            'middle_avg_duration': self._compute_run_lengths(~low_mask & ~high_mask).mean()
        }
```

### 1.2 Expected Outputs

After running Level 1 analysis, we expect to know:

| Question | Example Output | Implication |
|----------|----------------|-------------|
| Is entropy bimodal? | "No, GMM optimal = 1 component" | Binary thresholds are WRONG |
| Where are natural clusters? | "Modes at 0.42, 0.58, 0.71" | Use THESE as boundaries |
| How persistent is entropy? | "Half-life = 8 periods (2h)" | Minimum holding period |
| What's the 25th percentile? | "0.38" | Data-driven low threshold |
| Is entropy stationary? | "ADF p=0.01, yes" | Can use standard methods |

---

## Part III: Level 2 - Within-Cluster Characterization

### 3.1 Cluster-Conditional Analysis Framework

**For EACH entropy cluster identified in Level 1, compute:**

```python
class WithinClusterAnalysis:
    """
    Analyze properties WITHIN each entropy cluster

    This is the key insight: characteristics differ by regime
    """

    def __init__(self, full_data, cluster_assignments):
        self.data = full_data
        self.clusters = cluster_assignments
        self.n_clusters = len(np.unique(cluster_assignments))

    def analyze_all_clusters(self):
        """
        Comprehensive analysis for each cluster
        """
        results = {}

        for cluster_id in range(self.n_clusters):
            cluster_mask = self.clusters == cluster_id
            cluster_data = self.data[cluster_mask]

            results[f'cluster_{cluster_id}'] = {
                # Basic info
                'n_observations': len(cluster_data),
                'percentage_of_data': len(cluster_data) / len(self.data) * 100,
                'entropy_range': (cluster_data['entropy'].min(),
                                  cluster_data['entropy'].max()),

                # Volatility characterization
                'volatility': self._analyze_volatility(cluster_data),

                # Trend/continuity characterization
                'trend_continuity': self._analyze_trend_continuity(cluster_data),

                # Return distribution
                'return_distribution': self._analyze_returns(cluster_data),

                # Feature correlations within cluster
                'feature_correlations': self._analyze_feature_correlations(cluster_data),

                # Predictive features within cluster
                'predictive_features': self._analyze_predictive_power(cluster_data),

                # PCA within cluster
                'pca_analysis': self._analyze_pca(cluster_data)
            }

        return results
```

### 3.2 Volatility Characterization Per Cluster

```python
def _analyze_volatility(self, cluster_data):
    """
    How does volatility behave in this entropy regime?
    """
    returns = cluster_data['returns']

    return {
        # Basic volatility measures
        'realized_vol_mean': cluster_data['realized_vol_5m'].mean(),
        'realized_vol_std': cluster_data['realized_vol_5m'].std(),
        'parkinson_vol_mean': cluster_data.get('parkinson_vol', returns.std()),

        # Volatility distribution
        'vol_percentiles': {
            'p10': cluster_data['realized_vol_5m'].quantile(0.10),
            'p50': cluster_data['realized_vol_5m'].quantile(0.50),
            'p90': cluster_data['realized_vol_5m'].quantile(0.90)
        },

        # Volatility clustering (GARCH effects)
        'vol_autocorrelation': cluster_data['realized_vol_5m'].autocorr(lag=1),
        'vol_half_life': self._compute_acf_half_life(cluster_data['realized_vol_5m']),

        # Volatility-return relationship
        'vol_return_correlation': cluster_data['realized_vol_5m'].corr(
            cluster_data['returns'].abs()
        ),

        # High volatility events
        'extreme_vol_frequency': (
            cluster_data['realized_vol_5m'] >
            cluster_data['realized_vol_5m'].quantile(0.95)
        ).mean(),

        # Comparison to other clusters (computed later)
        'relative_vol': None  # Filled in post-processing
    }
```

### 3.3 Trend Continuity Per Cluster

```python
def _analyze_trend_continuity(self, cluster_data):
    """
    How do trends behave in this entropy regime?

    Key question: Does momentum continue or reverse?
    """
    returns = cluster_data['returns']
    momentum = cluster_data['momentum_300']

    # Autocorrelation of returns (momentum persistence)
    return_acf = [returns.autocorr(lag=k) for k in range(1, 21)]

    # Sign persistence (how often does direction continue?)
    sign_persistence = (np.sign(returns) == np.sign(returns.shift(1))).mean()

    # Momentum-return relationship
    forward_return = returns.shift(-1)
    momentum_return_corr = momentum.corr(forward_return)

    # Conditional returns
    positive_momentum = momentum > 0
    negative_momentum = momentum < 0

    return {
        # Autocorrelation structure
        'return_autocorrelations': return_acf,
        'acf_lag1': return_acf[0] if return_acf else None,
        'acf_lag5': return_acf[4] if len(return_acf) > 4 else None,

        # Sign persistence
        'sign_persistence_rate': sign_persistence,
        'consecutive_same_sign_avg': self._avg_consecutive_signs(returns),

        # Momentum predictability
        'momentum_return_correlation': momentum_return_corr,
        'momentum_leads_returns': self._test_lead_lag(momentum, forward_return),

        # Conditional returns
        'return_given_positive_momentum': forward_return[positive_momentum].mean(),
        'return_given_negative_momentum': forward_return[negative_momentum].mean(),
        'win_rate_following_momentum': (
            (np.sign(momentum) == np.sign(forward_return)).mean()
        ),

        # Trend strength measures
        'avg_trend_strength': cluster_data.get('trend_strength', momentum.abs()).mean(),
        'avg_r_squared': cluster_data.get('r_squared_300', 0).mean(),
        'avg_hurst': cluster_data.get('hurst_exponent', 0.5).mean(),

        # Reversal frequency
        'reversal_rate': (np.sign(returns) != np.sign(returns.shift(1))).mean()
    }
```

### 3.4 Return Distribution Per Cluster

```python
def _analyze_returns(self, cluster_data):
    """
    What do returns look like in this entropy regime?
    """
    returns = cluster_data['returns']

    return {
        # Central tendency
        'mean': returns.mean(),
        'median': returns.median(),
        'mean_annualized': returns.mean() * 252 * 24 * 4,  # Adjust for timeframe

        # Dispersion
        'std': returns.std(),
        'mad': (returns - returns.mean()).abs().mean(),
        'iqr': returns.quantile(0.75) - returns.quantile(0.25),

        # Shape
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),

        # Tails
        'tail_index': self._estimate_tail_index(returns),
        'var_95': returns.quantile(0.05),
        'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
        'var_99': returns.quantile(0.01),

        # Normality tests
        'shapiro_pvalue': self._shapiro_test(returns),
        'jarque_bera_pvalue': self._jb_test(returns),
        'is_normal': self._shapiro_test(returns) > 0.05,

        # Performance metrics
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252 * 24 * 4),
        'sortino_ratio': returns.mean() / returns[returns < 0].std() * np.sqrt(252 * 24 * 4),

        # Win/loss statistics
        'positive_return_pct': (returns > 0).mean(),
        'avg_positive_return': returns[returns > 0].mean(),
        'avg_negative_return': returns[returns < 0].mean(),
        'win_loss_ratio': abs(returns[returns > 0].mean() / returns[returns < 0].mean())
    }
```

### 3.5 Feature Correlations Per Cluster

```python
def _analyze_feature_correlations(self, cluster_data):
    """
    How do features relate to each other within this cluster?

    Key insight: Correlations may differ by regime
    """
    # Select numeric features
    feature_cols = [c for c in cluster_data.columns
                    if cluster_data[c].dtype in ['float64', 'float32', 'int64']]

    features = cluster_data[feature_cols].dropna()

    # Correlation matrices
    pearson_corr = features.corr(method='pearson')
    spearman_corr = features.corr(method='spearman')

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(pearson_corr.columns)):
        for j in range(i+1, len(pearson_corr.columns)):
            corr = pearson_corr.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr_pairs.append({
                    'feature_1': pearson_corr.columns[i],
                    'feature_2': pearson_corr.columns[j],
                    'correlation': corr
                })

    return {
        'pearson_correlation_matrix': pearson_corr.to_dict(),
        'spearman_correlation_matrix': spearman_corr.to_dict(),
        'highly_correlated_pairs': high_corr_pairs,
        'n_redundant_features': len([p for p in high_corr_pairs if abs(p['correlation']) > 0.8]),

        # Correlation stability (bootstrap)
        'correlation_stability': self._correlation_stability(features)
    }
```

### 3.6 PCA Per Cluster

```python
def _analyze_pca(self, cluster_data):
    """
    Dimensionality analysis within cluster

    How many independent dimensions of variation exist?
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Select features
    feature_cols = [c for c in cluster_data.columns
                    if c not in ['returns', 'timestamp', 'close', 'open', 'high', 'low']]

    features = cluster_data[feature_cols].dropna()

    if len(features) < 10:
        return {'error': 'Insufficient data for PCA'}

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # PCA
    pca = PCA()
    pca.fit(features_scaled)

    # Effective dimensionality
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim_90 = np.searchsorted(cumvar, 0.90) + 1
    dim_95 = np.searchsorted(cumvar, 0.95) + 1
    dim_99 = np.searchsorted(cumvar, 0.99) + 1

    # Principal component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=feature_cols
    )

    # Top features per component
    top_features_per_pc = {}
    for i in range(min(5, len(pca.components_))):
        pc_loadings = abs(loadings[f'PC{i+1}'])
        top_features_per_pc[f'PC{i+1}'] = pc_loadings.nlargest(5).index.tolist()

    return {
        'n_features_original': len(feature_cols),
        'effective_dim_90pct': dim_90,
        'effective_dim_95pct': dim_95,
        'effective_dim_99pct': dim_99,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist()[:10],
        'cumulative_variance': cumvar.tolist()[:10],
        'top_features_per_component': top_features_per_pc,
        'dimensionality_ratio': dim_95 / len(feature_cols)
    }
```

---

## Part IV: Level 3 - Cross-Cluster Dynamics

### 4.1 Transition Analysis

```python
class CrossClusterDynamics:
    """
    How do clusters relate to each other over time?
    """

    def __init__(self, cluster_sequence, cluster_data):
        self.sequence = cluster_sequence
        self.data = cluster_data
        self.n_clusters = len(np.unique(cluster_sequence))

    def compute_transition_matrix(self):
        """
        Probability of transitioning between clusters
        """
        transitions = np.zeros((self.n_clusters, self.n_clusters))

        for i in range(len(self.sequence) - 1):
            from_cluster = self.sequence[i]
            to_cluster = self.sequence[i + 1]
            transitions[from_cluster, to_cluster] += 1

        # Normalize rows
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_matrix = transitions / row_sums

        return {
            'transition_matrix': transition_matrix.tolist(),
            'self_transition_probs': np.diag(transition_matrix).tolist(),
            'expected_durations': [1 / (1 - p) if p < 1 else float('inf')
                                   for p in np.diag(transition_matrix)],
            'stationary_distribution': self._stationary_distribution(transition_matrix)
        }

    def analyze_transition_returns(self):
        """
        What happens to returns during transitions?
        """
        transitions = []

        for i in range(len(self.sequence) - 1):
            from_cluster = self.sequence[i]
            to_cluster = self.sequence[i + 1]

            if from_cluster != to_cluster:
                # Transition occurred
                returns_before = self.data['returns'].iloc[max(0, i-5):i].mean()
                returns_after = self.data['returns'].iloc[i:min(len(self.data), i+5)].mean()

                transitions.append({
                    'from': from_cluster,
                    'to': to_cluster,
                    'idx': i,
                    'returns_before': returns_before,
                    'returns_after': returns_after
                })

        # Aggregate by transition type
        transition_returns = {}
        for from_c in range(self.n_clusters):
            for to_c in range(self.n_clusters):
                if from_c != to_c:
                    key = f'{from_c}_to_{to_c}'
                    relevant = [t for t in transitions
                               if t['from'] == from_c and t['to'] == to_c]

                    if relevant:
                        transition_returns[key] = {
                            'count': len(relevant),
                            'avg_return_before': np.mean([t['returns_before'] for t in relevant]),
                            'avg_return_after': np.mean([t['returns_after'] for t in relevant]),
                            'return_change': np.mean([t['returns_after'] - t['returns_before']
                                                     for t in relevant])
                        }

        return transition_returns

    def detect_change_points(self, method='cusum'):
        """
        Detect structural breaks in entropy series
        """
        entropy = self.data['entropy'].values

        if method == 'cusum':
            # CUSUM for change-point detection
            mu = np.mean(entropy[:100])  # Initial mean estimate
            sigma = np.std(entropy[:100])
            k = 0.5 * sigma
            h = 4  # Threshold

            cusum_pos = np.zeros(len(entropy))
            cusum_neg = np.zeros(len(entropy))

            for i in range(1, len(entropy)):
                z = (entropy[i] - mu) / sigma
                cusum_pos[i] = max(0, cusum_pos[i-1] + z - k)
                cusum_neg[i] = max(0, cusum_neg[i-1] - z - k)

            change_points = np.where((cusum_pos > h) | (cusum_neg > h))[0]

            return {
                'method': 'cusum',
                'change_points': change_points.tolist(),
                'n_change_points': len(change_points),
                'avg_segment_length': len(entropy) / (len(change_points) + 1)
            }

        return None
```

---

## Part V: Level 4 - Predictive Relationship Analysis

### 5.1 Return Prediction Analysis

```python
class PredictiveAnalysis:
    """
    Which features predict returns, and in which clusters?

    CRITICAL: This determines which features to use in algorithms
    """

    def __init__(self, data, cluster_assignments):
        self.data = data
        self.clusters = cluster_assignments

    def analyze_feature_predictiveness(self, forward_periods=[1, 5, 20]):
        """
        For each feature, how well does it predict forward returns?
        """
        results = {}

        for period in forward_periods:
            fwd_returns = self.data['returns'].shift(-period)

            for feature in self.data.columns:
                if feature in ['returns', 'timestamp', 'close']:
                    continue

                key = f'{feature}_vs_return_{period}p'

                results[key] = {
                    # Linear relationship
                    'pearson_corr': self.data[feature].corr(fwd_returns),
                    'spearman_corr': self.data[feature].corr(fwd_returns, method='spearman'),

                    # Statistical significance
                    'pearson_pvalue': self._correlation_pvalue(
                        self.data[feature], fwd_returns
                    ),

                    # Non-linear relationship
                    'mutual_information': self._mutual_info(
                        self.data[feature], fwd_returns
                    ),

                    # Lead-lag structure
                    'feature_leads_returns': self._feature_leads(
                        self.data[feature], self.data['returns']
                    ),
                    'optimal_lag': self._find_optimal_lag(
                        self.data[feature], self.data['returns']
                    )
                }

        # Rank features by predictiveness
        feature_ranks = self._rank_features(results)

        return {
            'detailed_results': results,
            'feature_rankings': feature_ranks
        }

    def cluster_conditional_predictiveness(self, forward_period=1):
        """
        Does predictiveness differ by cluster?

        THIS IS KEY: A feature may predict in low-entropy but not high-entropy
        """
        fwd_returns = self.data['returns'].shift(-forward_period)

        results = {}

        for cluster_id in np.unique(self.clusters):
            cluster_mask = self.clusters == cluster_id
            cluster_data = self.data[cluster_mask]
            cluster_fwd = fwd_returns[cluster_mask]

            cluster_results = {}

            for feature in self.data.columns:
                if feature in ['returns', 'timestamp', 'close']:
                    continue

                corr = cluster_data[feature].corr(cluster_fwd)
                mi = self._mutual_info(cluster_data[feature], cluster_fwd)

                cluster_results[feature] = {
                    'correlation': corr,
                    'mutual_info': mi,
                    'n_samples': len(cluster_data)
                }

            results[f'cluster_{cluster_id}'] = cluster_results

        # Find regime-dependent features
        regime_dependent = self._find_regime_dependent_features(results)

        return {
            'cluster_results': results,
            'regime_dependent_features': regime_dependent
        }

    def _find_regime_dependent_features(self, cluster_results):
        """
        Identify features with different predictiveness across clusters
        """
        dependent = []

        features = list(list(cluster_results.values())[0].keys())

        for feature in features:
            correlations = [cluster_results[c][feature]['correlation']
                           for c in cluster_results.keys()]

            # High variance across clusters = regime-dependent
            corr_std = np.std(correlations)

            if corr_std > 0.1:  # Significant variation
                dependent.append({
                    'feature': feature,
                    'correlation_std': corr_std,
                    'correlations_by_cluster': dict(zip(cluster_results.keys(), correlations)),
                    'max_corr_cluster': list(cluster_results.keys())[np.argmax(np.abs(correlations))]
                })

        return sorted(dependent, key=lambda x: x['correlation_std'], reverse=True)
```

### 5.2 Causal Validation

```python
def validate_causality(self, feature, returns, max_lag=20):
    """
    Is the relationship causal or spurious?

    Tests:
    1. Granger causality - does feature help predict returns?
    2. Lead-lag - does feature LEAD (not lag) returns?
    3. Permutation test - is correlation spurious?
    4. Out-of-sample - does relationship hold OOS?
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    results = {}

    # 1. Granger causality
    data = pd.DataFrame({'feature': feature, 'returns': returns}).dropna()
    try:
        gc_result = grangercausalitytests(data[['returns', 'feature']], maxlag=max_lag)
        pvalues = {lag: gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)}
        results['granger_causes'] = min(pvalues.values()) < 0.01
        results['granger_pvalues'] = pvalues
    except:
        results['granger_causes'] = None

    # 2. Lead-lag analysis
    cross_corrs = {}
    for lag in range(-max_lag, max_lag+1):
        if lag < 0:
            corr = feature.iloc[:lag].corr(returns.iloc[-lag:])
        elif lag > 0:
            corr = feature.iloc[lag:].corr(returns.iloc[:-lag])
        else:
            corr = feature.corr(returns)
        cross_corrs[lag] = corr

    peak_lag = max(cross_corrs, key=lambda k: abs(cross_corrs[k]))
    results['peak_lag'] = peak_lag
    results['feature_leads'] = peak_lag < 0  # Negative lag = feature leads
    results['cross_correlations'] = cross_corrs

    # 3. Permutation test
    observed_corr = feature.corr(returns)
    null_corrs = []
    for _ in range(1000):
        shuffled = np.random.permutation(returns)
        null_corrs.append(feature.corr(pd.Series(shuffled)))

    results['permutation_pvalue'] = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))
    results['is_spurious'] = results['permutation_pvalue'] > 0.01

    # 4. Out-of-sample test
    n = len(feature)
    train_end = int(n * 0.6)

    train_corr = feature.iloc[:train_end].corr(returns.iloc[:train_end])
    test_corr = feature.iloc[train_end:].corr(returns.iloc[train_end:])

    results['train_correlation'] = train_corr
    results['test_correlation'] = test_corr
    results['oos_is_ratio'] = test_corr / train_corr if train_corr != 0 else 0
    results['holds_oos'] = results['oos_is_ratio'] > 0.5

    # Final verdict
    results['is_causal'] = (
        results.get('granger_causes', False) and
        results['feature_leads'] and
        not results['is_spurious'] and
        results['holds_oos']
    )

    return results
```

---

## Part VI: Level 5 - Feature Selection & Reduction

### 6.1 Principled Feature Selection

```python
class FeatureSelection:
    """
    Select final features for algorithm development

    Criteria:
    1. Non-redundant (correlation < 0.8)
    2. Predictive (MI > threshold)
    3. Stable (appears in >80% of bootstraps)
    4. Causal (passes causality tests)
    """

    def __init__(self, data, returns, max_features=10):
        self.data = data
        self.returns = returns
        self.max_features = max_features

    def full_pipeline(self):
        """
        Complete feature selection pipeline
        """
        print("="*60)
        print("FEATURE SELECTION PIPELINE")
        print("="*60)

        # Stage 1: Remove redundant features
        print("\nStage 1: Removing redundant features...")
        features_stage1 = self._remove_redundancy()
        print(f"  {len(self.data.columns)} -> {len(features_stage1)} features")

        # Stage 2: Filter by relevance (MI)
        print("\nStage 2: Filtering by relevance...")
        features_stage2 = self._filter_relevance(features_stage1)
        print(f"  {len(features_stage1)} -> {len(features_stage2)} features")

        # Stage 3: Stability selection
        print("\nStage 3: Stability selection...")
        features_stage3 = self._stability_selection(features_stage2)
        print(f"  {len(features_stage2)} -> {len(features_stage3)} features")

        # Stage 4: Causal validation
        print("\nStage 4: Causal validation...")
        final_features = self._causal_filter(features_stage3)
        print(f"  {len(features_stage3)} -> {len(final_features)} features")

        print("\n" + "="*60)
        print("FINAL SELECTED FEATURES:")
        for f in final_features:
            print(f"  - {f}")
        print("="*60)

        return final_features

    def _remove_redundancy(self, threshold=0.8):
        """Remove features with |correlation| > threshold"""
        corr_matrix = self.data.corr().abs()

        # Find pairs to remove
        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    # Drop the one with lower MI with returns
                    mi1 = self._mutual_info_single(self.data.iloc[:, i])
                    mi2 = self._mutual_info_single(self.data.iloc[:, j])

                    if mi1 > mi2:
                        to_drop.add(corr_matrix.columns[j])
                    else:
                        to_drop.add(corr_matrix.columns[i])

        return [c for c in self.data.columns if c not in to_drop]

    def _filter_relevance(self, features, mi_threshold=0.01):
        """Keep features with significant mutual information"""
        from sklearn.feature_selection import mutual_info_regression

        X = self.data[features].dropna()
        y = self.returns.loc[X.index]

        mi_scores = mutual_info_regression(X, y)

        # Permutation test for significance
        significant = []
        for i, feature in enumerate(features):
            null_mi = []
            for _ in range(100):
                shuffled_y = np.random.permutation(y)
                null_mi.append(mutual_info_regression(
                    X[[feature]], shuffled_y
                )[0])

            pvalue = np.mean(np.array(null_mi) >= mi_scores[i])

            if pvalue < 0.05 / len(features):  # Bonferroni correction
                significant.append(feature)

        return significant

    def _stability_selection(self, features, n_bootstrap=100, threshold=0.8):
        """Keep features appearing in >threshold of bootstraps"""
        from sklearn.linear_model import LassoCV

        X = self.data[features].dropna()
        y = self.returns.loc[X.index]

        feature_counts = {f: 0 for f in features}

        for _ in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot = X.iloc[idx]
            y_boot = y.iloc[idx]

            # LASSO selection
            lasso = LassoCV(cv=5, random_state=None)
            lasso.fit(X_boot, y_boot)

            # Count selected features
            for f, coef in zip(features, lasso.coef_):
                if abs(coef) > 1e-6:
                    feature_counts[f] += 1

        # Keep stable features
        stable = [f for f, count in feature_counts.items()
                  if count / n_bootstrap > threshold]

        return stable[:self.max_features]

    def _causal_filter(self, features):
        """Keep only features that pass causal tests"""
        pa = PredictiveAnalysis(self.data, None)

        causal_features = []
        for feature in features:
            result = pa.validate_causality(
                self.data[feature],
                self.returns
            )

            if result['is_causal']:
                causal_features.append(feature)

        return causal_features
```

---

## Part VII: Update Frequency Recommendations

### 7.1 When to Update Statistical Analysis

| Analysis Level | Update Frequency | Trigger for Out-of-Cycle Update |
|---------------|------------------|--------------------------------|
| **Level 1: Entropy Distribution** | Monthly | Dip test shows change in modality |
| **Level 2: Within-Cluster Stats** | Weekly | >20% change in cluster composition |
| **Level 3: Transition Dynamics** | Weekly | Transition matrix changes >10% |
| **Level 4: Predictive Analysis** | Weekly | OOS/IS ratio drops below 0.5 |
| **Level 5: Feature Selection** | Monthly | Any upstream analysis changes significantly |

### 7.2 Continuous Monitoring Signals

```python
class StatisticalMonitor:
    """
    Monitor for drift requiring re-analysis
    """

    def __init__(self, baseline_stats):
        self.baseline = baseline_stats

    def check_entropy_drift(self, recent_entropy, window=1000):
        """
        Has entropy distribution changed?
        """
        # KS test for distribution change
        from scipy.stats import ks_2samp

        baseline_entropy = self.baseline['entropy_values'][-window:]
        recent_entropy = recent_entropy[-window:]

        stat, pvalue = ks_2samp(baseline_entropy, recent_entropy)

        return {
            'distribution_changed': pvalue < 0.01,
            'ks_statistic': stat,
            'p_value': pvalue,
            'action': 'RERUN_LEVEL_1' if pvalue < 0.01 else 'NONE'
        }

    def check_correlation_stability(self, recent_data):
        """
        Have feature correlations changed?
        """
        baseline_corr = self.baseline['correlation_matrix']
        recent_corr = recent_data.corr()

        # Frobenius norm of difference
        diff = baseline_corr - recent_corr
        frobenius_diff = np.sqrt((diff ** 2).sum().sum())

        # Normalized by matrix size
        normalized_diff = frobenius_diff / baseline_corr.shape[0]

        return {
            'correlations_changed': normalized_diff > 0.1,
            'frobenius_diff': frobenius_diff,
            'normalized_diff': normalized_diff,
            'action': 'RERUN_LEVEL_2' if normalized_diff > 0.1 else 'NONE'
        }

    def check_predictive_decay(self, recent_data, returns):
        """
        Has feature predictiveness decayed?
        """
        baseline_predictive = self.baseline['feature_correlations_with_returns']

        recent_predictive = {}
        for feature in baseline_predictive.keys():
            recent_predictive[feature] = recent_data[feature].corr(returns)

        # Compare
        decayed_features = []
        for feature in baseline_predictive:
            baseline = baseline_predictive[feature]
            recent = recent_predictive[feature]

            if baseline != 0:
                ratio = recent / baseline
                if ratio < 0.5:
                    decayed_features.append({
                        'feature': feature,
                        'baseline': baseline,
                        'recent': recent,
                        'ratio': ratio
                    })

        return {
            'predictiveness_decayed': len(decayed_features) > 0,
            'decayed_features': decayed_features,
            'action': 'RERUN_LEVEL_4' if len(decayed_features) > 0 else 'NONE'
        }
```

### 7.3 Automated Update Pipeline

```python
class StatisticalUpdatePipeline:
    """
    Automated pipeline to update statistics on schedule
    """

    def __init__(self, data_source, output_path):
        self.data_source = data_source
        self.output_path = output_path

    def daily_update(self):
        """
        Daily: Quick health checks only
        """
        recent_data = self.data_source.get_recent(periods=1000)

        checks = {
            'data_quality': self._check_data_quality(recent_data),
            'outlier_count': self._count_outliers(recent_data),
            'missing_rate': recent_data.isnull().sum().sum() / recent_data.size
        }

        self._log_checks('daily', checks)

        return checks

    def weekly_update(self):
        """
        Weekly: Full statistical refresh
        """
        print("Starting weekly statistical update...")

        # Get expanded data window
        data = self.data_source.get_recent(periods=10000)

        # Run Levels 2-4
        level2 = WithinClusterAnalysis(data, self._get_clusters(data))
        level3 = CrossClusterDynamics(self._get_clusters(data), data)
        level4 = PredictiveAnalysis(data, self._get_clusters(data))

        results = {
            'timestamp': datetime.now().isoformat(),
            'level2': level2.analyze_all_clusters(),
            'level3': level3.compute_transition_matrix(),
            'level4': level4.analyze_feature_predictiveness()
        }

        # Save results
        self._save_results('weekly', results)

        # Check for significant changes
        changes = self._compare_to_baseline(results)
        if changes['significant']:
            self._trigger_alert(changes)

        return results

    def monthly_update(self):
        """
        Monthly: Deep analysis including entropy distribution
        """
        print("Starting monthly statistical update...")

        # Get full data window (6+ months)
        data = self.data_source.get_full_history()

        # Run all levels
        level1 = EntropyDistributionAnalysis(data['entropy'])
        level2 = WithinClusterAnalysis(data, self._get_clusters(data))
        level3 = CrossClusterDynamics(self._get_clusters(data), data)
        level4 = PredictiveAnalysis(data, self._get_clusters(data))
        level5 = FeatureSelection(data, data['returns'])

        results = {
            'timestamp': datetime.now().isoformat(),
            'level1': {
                'distribution': level1.compute_basic_distribution(),
                'clusters': level1.detect_natural_clusters(),
                'temporal': level1.analyze_temporal_properties()
            },
            'level2': level2.analyze_all_clusters(),
            'level3': level3.compute_transition_matrix(),
            'level4': level4.analyze_feature_predictiveness(),
            'level5': level5.full_pipeline()
        }

        # Update baseline
        self._update_baseline(results)

        # Save results
        self._save_results('monthly', results)

        return results
```

---

## Part VIII: Output Format for Web Dashboard / Ingestion

### 8.1 Database Schema

```sql
-- Statistical summary tables

CREATE TABLE entropy_distribution (
    analysis_date DATE PRIMARY KEY,
    mean FLOAT,
    median FLOAT,
    std FLOAT,
    skewness FLOAT,
    kurtosis FLOAT,
    percentile_5 FLOAT,
    percentile_25 FLOAT,
    percentile_50 FLOAT,
    percentile_75 FLOAT,
    percentile_95 FLOAT,
    n_clusters INT,
    cluster_means JSON,
    cluster_weights JSON,
    is_multimodal BOOLEAN,
    acf_half_life INT,
    hurst_exponent FLOAT,
    updated_at TIMESTAMP
);

CREATE TABLE cluster_statistics (
    analysis_date DATE,
    cluster_id INT,
    n_observations INT,
    pct_of_data FLOAT,
    entropy_min FLOAT,
    entropy_max FLOAT,
    vol_mean FLOAT,
    vol_std FLOAT,
    return_mean FLOAT,
    return_std FLOAT,
    sharpe_ratio FLOAT,
    sign_persistence FLOAT,
    momentum_correlation FLOAT,
    effective_dim_95 INT,
    PRIMARY KEY (analysis_date, cluster_id)
);

CREATE TABLE feature_predictiveness (
    analysis_date DATE,
    feature_name VARCHAR(100),
    forward_period INT,
    pearson_corr FLOAT,
    spearman_corr FLOAT,
    mutual_info FLOAT,
    granger_pvalue FLOAT,
    feature_leads_returns BOOLEAN,
    oos_is_ratio FLOAT,
    is_causal BOOLEAN,
    PRIMARY KEY (analysis_date, feature_name, forward_period)
);

CREATE TABLE transition_matrix (
    analysis_date DATE,
    from_cluster INT,
    to_cluster INT,
    probability FLOAT,
    avg_return_after FLOAT,
    PRIMARY KEY (analysis_date, from_cluster, to_cluster)
);

CREATE TABLE selected_features (
    analysis_date DATE,
    feature_rank INT,
    feature_name VARCHAR(100),
    selection_reason VARCHAR(255),
    PRIMARY KEY (analysis_date, feature_rank)
);
```

### 8.2 JSON Output Format

```python
def generate_statistical_summary_json(results):
    """
    Generate JSON for web dashboard or API consumption
    """
    return {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'data_range': {
                'start': results['data_start'],
                'end': results['data_end'],
                'n_observations': results['n_observations']
            },
            'version': '1.0'
        },

        'entropy_analysis': {
            'distribution': results['level1']['distribution'],
            'clusters': {
                'optimal_n': results['level1']['clusters']['gmm']['optimal_n_clusters'],
                'means': results['level1']['clusters']['gmm']['cluster_means'],
                'weights': results['level1']['clusters']['gmm']['cluster_weights']
            },
            'temporal': {
                'half_life': results['level1']['temporal']['acf_half_life'],
                'hurst': results['level1']['temporal']['hurst_exponent'],
                'is_stationary': results['level1']['temporal']['is_stationary']
            }
        },

        'cluster_profiles': results['level2'],

        'dynamics': {
            'transition_matrix': results['level3']['transition_matrix'],
            'expected_durations': results['level3']['expected_durations'],
            'change_points': results['level3'].get('change_points', [])
        },

        'predictive_features': {
            'top_features': results['level4']['feature_rankings'][:10],
            'cluster_conditional': results['level4']['cluster_results'],
            'regime_dependent': results['level4']['regime_dependent_features']
        },

        'selected_features': results['level5'],

        'recommendations': self._generate_recommendations(results)
    }

def _generate_recommendations(self, results):
    """
    Based on statistical analysis, what should we do?
    """
    recs = []

    # Entropy clustering recommendation
    n_clusters = results['level1']['clusters']['gmm']['optimal_n_clusters']
    if n_clusters == 1:
        recs.append({
            'type': 'WARNING',
            'message': 'Entropy is NOT multimodal. Binary thresholds will lose information.',
            'action': 'Use continuous regime probability, not binary classification.'
        })
    elif n_clusters >= 3:
        means = results['level1']['clusters']['gmm']['cluster_means']
        recs.append({
            'type': 'INFO',
            'message': f'Found {n_clusters} natural entropy clusters at {means}',
            'action': f'Use data-driven thresholds based on cluster boundaries.'
        })

    # Predictability recommendation
    top_features = results['level4']['feature_rankings'][:5]
    causal_count = sum(1 for f in top_features if f['is_causal'])
    if causal_count < 3:
        recs.append({
            'type': 'WARNING',
            'message': f'Only {causal_count}/5 top features pass causal tests.',
            'action': 'Be cautious about predictive claims. May be spurious.'
        })

    # Regime stability recommendation
    half_life = results['level1']['temporal']['acf_half_life']
    if half_life != ">100" and half_life < 10:
        recs.append({
            'type': 'WARNING',
            'message': f'Entropy half-life is only {half_life} periods.',
            'action': 'Regimes are short-lived. Consider shorter holding periods.'
        })

    return recs
```

---

## Part IX: Implementation Checklist

### Before ANY Algorithm Development:

- [ ] **Run Level 1 Analysis**
  - [ ] Compute entropy percentiles (actual, not assumed)
  - [ ] Detect natural clusters (GMM with BIC)
  - [ ] Measure temporal persistence (ACF half-life)
  - [ ] Test stationarity

- [ ] **Run Level 2 Analysis (per cluster)**
  - [ ] Volatility characteristics
  - [ ] Trend continuity (momentum persistence)
  - [ ] Return distribution (skew, kurtosis, tails)
  - [ ] Feature correlations
  - [ ] PCA (effective dimensionality)

- [ ] **Run Level 3 Analysis**
  - [ ] Compute transition matrix
  - [ ] Analyze transition returns
  - [ ] Detect change points

- [ ] **Run Level 4 Analysis**
  - [ ] Feature predictiveness (overall)
  - [ ] Cluster-conditional predictiveness
  - [ ] Causal validation (Granger, lead-lag, permutation, OOS)

- [ ] **Run Level 5 Analysis**
  - [ ] Remove redundant features
  - [ ] Filter by relevance
  - [ ] Stability selection
  - [ ] Final causal filter

- [ ] **Generate Summary**
  - [ ] Export to database/JSON
  - [ ] Generate recommendations
  - [ ] Document findings

### Expected Timeline:

| Phase | Duration | Output |
|-------|----------|--------|
| Data collection | Ongoing | 6+ months of tick data |
| Level 1 analysis | 1-2 days | Entropy distribution report |
| Level 2 analysis | 2-3 days | Cluster profile reports |
| Level 3 analysis | 1 day | Transition dynamics report |
| Level 4 analysis | 2-3 days | Predictiveness analysis |
| Level 5 analysis | 1-2 days | Selected feature set |
| Documentation | 1 day | Final statistical summary |
| **Total** | **~2 weeks** | Complete statistical foundation |

---

## Conclusion

This framework ensures that:

1. **Data comes first** - We characterize the data before designing algorithms
2. **Entropy clusters organize analysis** - All statistics computed within natural regimes
3. **Causality is validated** - No spurious correlations make it through
4. **Features are principled** - Selection based on predictiveness, stability, and non-redundancy
5. **Updates are systematic** - Clear schedule for refreshing analysis

**The key insight you identified is correct:** Building a statistical analysis tool that generates data about our features BEFORE developing hypotheses is the only rigorous approach. The data will tell us what algorithms are realistic to implement.

---

**Document Version:** 1.0
**Created:** 2026-04-05
**Next Step:** Implement Level 1 analysis on existing data
