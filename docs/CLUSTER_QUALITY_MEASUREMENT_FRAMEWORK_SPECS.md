# Cluster Quality Measurement Framework Specification

**Version:** 1.0
**Date:** 2026-03-27
**Status:** Draft

---

## 1. Overview

### 1.1 Purpose

This framework provides rigorous, quantitative methods to measure whether extracted features contain meaningful structure for Hidden Markov Model (HMM) regime detection. Rather than assuming regimes exist, we first validate that the feature space exhibits clear, stable, and economically meaningful clusters.

### 1.2 Philosophy

```
Traditional Approach:
  Assume regimes → Extract features → Fit HMM → Hope it works

Our Approach:
  Extract features → Measure cluster quality → Validate predictive power →
  Refine if needed → Only then build HMM
```

### 1.3 Key Questions This Framework Answers

1. **Do natural clusters exist in the feature space?**
2. **How separable are these clusters?**
3. **Are clusters stable across time periods?**
4. **Do clusters have predictive value for returns/volatility?**
5. **Which feature subsets produce the clearest structure?**

---

## 2. Cluster Quality Metrics

### 2.1 Internal Validation Metrics

These metrics measure cluster quality using only the feature data (no external labels).

#### 2.1.1 Silhouette Score

**Definition:** Measures how similar a point is to its own cluster compared to other clusters.

```
For each point i:
  a(i) = average distance to points in same cluster
  b(i) = minimum average distance to points in other clusters
  s(i) = (b(i) - a(i)) / max(a(i), b(i))

Silhouette Score = mean(s(i)) for all points
```

**Interpretation:**
| Score | Interpretation |
|-------|----------------|
| > 0.7 | Strong cluster structure |
| 0.5 - 0.7 | Reasonable structure |
| 0.25 - 0.5 | Weak structure, overlapping clusters |
| < 0.25 | No substantial structure |

**Implementation:**
```python
from sklearn.metrics import silhouette_score, silhouette_samples

def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute silhouette metrics.

    Returns:
        overall: Global silhouette score
        per_cluster: Silhouette by cluster (detect weak clusters)
        per_sample: Full distribution (detect outliers)
    """
    overall = silhouette_score(X, labels)
    samples = silhouette_samples(X, labels)

    per_cluster = {}
    for label in np.unique(labels):
        mask = labels == label
        per_cluster[label] = samples[mask].mean()

    return {
        "overall": overall,
        "per_cluster": per_cluster,
        "std": samples.std(),
        "pct_negative": (samples < 0).mean(),  # Misclassified points
    }
```

#### 2.1.2 Davies-Bouldin Index

**Definition:** Ratio of within-cluster scatter to between-cluster separation. Lower is better.

```
For clusters i and j:
  S_i = average distance of points in i to centroid of i
  D_ij = distance between centroids of i and j
  R_ij = (S_i + S_j) / D_ij

DB = (1/k) * Σ max(R_ij) for j≠i
```

**Interpretation:**
| Score | Interpretation |
|-------|----------------|
| < 0.5 | Excellent separation |
| 0.5 - 1.0 | Good separation |
| 1.0 - 2.0 | Moderate overlap |
| > 2.0 | Poor separation |

**Implementation:**
```python
from sklearn.metrics import davies_bouldin_score

def compute_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    return davies_bouldin_score(X, labels)
```

#### 2.1.3 Calinski-Harabasz Index (Variance Ratio)

**Definition:** Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.

```
CH = [B / (k-1)] / [W / (n-k)]

Where:
  B = between-cluster sum of squares
  W = within-cluster sum of squares
  k = number of clusters
  n = number of samples
```

**Implementation:**
```python
from sklearn.metrics import calinski_harabasz_score

def compute_calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    return calinski_harabasz_score(X, labels)
```

#### 2.1.4 Gap Statistic

**Definition:** Compares within-cluster dispersion to expected dispersion under null reference (uniform random).

```
Gap(k) = E*[log(W_k)] - log(W_k)

Where:
  W_k = within-cluster sum of squares for k clusters
  E* = expectation under null reference distribution
```

**Interpretation:** Choose k where Gap(k) is maximized and significantly above random.

**Implementation:**
```python
def compute_gap_statistic(X: np.ndarray, max_clusters: int = 10, n_refs: int = 20) -> dict:
    """
    Compute gap statistic for cluster number selection.
    """
    from sklearn.cluster import KMeans

    def compute_Wk(X, labels):
        """Within-cluster sum of squares."""
        W = 0
        for label in np.unique(labels):
            cluster_points = X[labels == label]
            centroid = cluster_points.mean(axis=0)
            W += ((cluster_points - centroid) ** 2).sum()
        return W

    # Compute for real data
    Wks = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        Wks.append(np.log(compute_Wk(X, labels)))

    # Compute for reference (uniform random)
    Wks_ref = []
    for k in range(1, max_clusters + 1):
        ref_Wks = []
        for _ in range(n_refs):
            # Generate uniform random data in same bounding box
            X_ref = np.random.uniform(
                X.min(axis=0), X.max(axis=0), size=X.shape
            )
            kmeans = KMeans(n_clusters=k, random_state=None, n_init=3)
            labels = kmeans.fit_predict(X_ref)
            ref_Wks.append(np.log(compute_Wk(X_ref, labels)))
        Wks_ref.append((np.mean(ref_Wks), np.std(ref_Wks)))

    # Gap = E[log(W_ref)] - log(W)
    gaps = [ref[0] - wk for ref, wk in zip(Wks_ref, Wks)]
    gap_stds = [ref[1] for ref in Wks_ref]

    return {
        "gaps": gaps,
        "gap_stds": gap_stds,
        "optimal_k": np.argmax(gaps) + 1,
    }
```

### 2.2 Stability Metrics

These metrics measure whether clusters are robust and reproducible.

#### 2.2.1 Bootstrap Stability

**Definition:** Consistency of cluster assignments across bootstrap samples.

```
For B bootstrap iterations:
  1. Sample with replacement
  2. Cluster the sample
  3. Match clusters to reference clustering
  4. Compute agreement (Adjusted Rand Index)

Stability = mean(ARI) across bootstraps
```

**Interpretation:**
| Score | Interpretation |
|-------|----------------|
| > 0.8 | Highly stable clusters |
| 0.6 - 0.8 | Moderately stable |
| 0.4 - 0.6 | Unstable, sensitive to sampling |
| < 0.4 | Very unstable, possibly spurious |

**Implementation:**
```python
from sklearn.metrics import adjusted_rand_score

def compute_bootstrap_stability(
    X: np.ndarray,
    cluster_func: callable,
    n_bootstraps: int = 100,
    sample_frac: float = 0.8,
) -> dict:
    """
    Measure cluster stability via bootstrap resampling.

    Args:
        X: Feature matrix
        cluster_func: Function that takes X and returns labels
        n_bootstraps: Number of bootstrap iterations
        sample_frac: Fraction of data to sample each iteration
    """
    n_samples = len(X)
    sample_size = int(n_samples * sample_frac)

    # Reference clustering on full data
    reference_labels = cluster_func(X)

    ari_scores = []
    for _ in range(n_bootstraps):
        # Bootstrap sample
        idx = np.random.choice(n_samples, sample_size, replace=True)
        X_sample = X[idx]

        # Cluster the sample
        sample_labels = cluster_func(X_sample)

        # Compare to reference (on overlapping points)
        # Use the sampled indices to get reference labels
        ref_subset = reference_labels[idx]

        ari = adjusted_rand_score(ref_subset, sample_labels)
        ari_scores.append(ari)

    return {
        "mean_ari": np.mean(ari_scores),
        "std_ari": np.std(ari_scores),
        "min_ari": np.min(ari_scores),
        "pct_stable": np.mean(np.array(ari_scores) > 0.6),
    }
```

#### 2.2.2 Temporal Stability

**Definition:** Do the same clusters appear in different time periods?

```
Split data into time windows:
  - Train: First 70%
  - Test: Last 30%

Cluster on train, predict on test.
Measure if test points cluster similarly.
```

**Implementation:**
```python
def compute_temporal_stability(
    X: np.ndarray,
    timestamps: np.ndarray,
    cluster_func: callable,
    train_frac: float = 0.7,
) -> dict:
    """
    Measure if clusters are stable across time.
    """
    # Sort by time
    sort_idx = np.argsort(timestamps)
    X_sorted = X[sort_idx]

    # Split
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X_sorted[:split_idx], X_sorted[split_idx:]

    # Cluster train
    train_labels = cluster_func(X_train)

    # For GMM, we can predict on test
    # For other methods, use nearest centroid

    # Compute centroids from training
    centroids = []
    for label in np.unique(train_labels):
        centroids.append(X_train[train_labels == label].mean(axis=0))
    centroids = np.array(centroids)

    # Assign test points to nearest centroid
    from sklearn.metrics import pairwise_distances
    distances = pairwise_distances(X_test, centroids)
    test_labels = distances.argmin(axis=1)

    # Cluster test independently
    test_labels_independent = cluster_func(X_test)

    # Compare
    ari = adjusted_rand_score(test_labels, test_labels_independent)

    # Also check cluster proportions
    train_props = np.bincount(train_labels) / len(train_labels)
    test_props = np.bincount(test_labels, minlength=len(train_props)) / len(test_labels)

    prop_diff = np.abs(train_props - test_props).mean()

    return {
        "temporal_ari": ari,
        "proportion_drift": prop_diff,
        "train_proportions": train_props.tolist(),
        "test_proportions": test_props.tolist(),
    }
```

#### 2.2.3 Cross-Validation Stability

**Definition:** K-fold stability measurement.

```python
def compute_cv_stability(
    X: np.ndarray,
    cluster_func: callable,
    n_folds: int = 5,
) -> dict:
    """
    K-fold cross-validation for cluster stability.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_labels = []
    for train_idx, test_idx in kf.split(X):
        labels = cluster_func(X[train_idx])
        fold_labels.append((train_idx, labels))

    # Compare all pairs of folds
    ari_pairs = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            # Find overlapping indices
            idx_i, labels_i = fold_labels[i]
            idx_j, labels_j = fold_labels[j]

            # This is tricky - folds don't overlap
            # Instead, cluster full data with each fold's model and compare
            pass  # Simplified: use bootstrap stability instead

    return {"cv_stability": np.mean(ari_pairs) if ari_pairs else None}
```

### 2.3 External Validation Metrics

These metrics validate clusters against external outcomes (returns, volatility).

#### 2.3.1 Forward Return Differentiation

**Definition:** Do different clusters have statistically different forward returns?

```
For each cluster k:
  R_k = forward returns of points in cluster k

Test: ANOVA or Kruskal-Wallis across clusters
Null hypothesis: All clusters have same mean return
```

**Implementation:**
```python
from scipy import stats

def compute_return_differentiation(
    labels: np.ndarray,
    forward_returns: np.ndarray,
    horizons: List[int] = [60, 300, 3600],  # seconds
) -> dict:
    """
    Test if clusters have different forward returns.

    Args:
        labels: Cluster assignments
        forward_returns: Dict of horizon -> returns array
        horizons: Return horizons to test
    """
    results = {}

    for horizon in horizons:
        returns = forward_returns.get(horizon)
        if returns is None:
            continue

        # Group returns by cluster
        groups = []
        for label in np.unique(labels):
            if label == -1:  # Skip noise
                continue
            group_returns = returns[labels == label]
            groups.append(group_returns)

        if len(groups) < 2:
            continue

        # ANOVA (assumes normality)
        f_stat, anova_p = stats.f_oneway(*groups)

        # Kruskal-Wallis (non-parametric)
        h_stat, kw_p = stats.kruskal(*groups)

        # Effect size (eta-squared)
        ss_between = sum(len(g) * (g.mean() - returns.mean())**2 for g in groups)
        ss_total = ((returns - returns.mean())**2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Mean returns by cluster
        cluster_means = {
            label: returns[labels == label].mean()
            for label in np.unique(labels) if label != -1
        }

        results[horizon] = {
            "anova_f": f_stat,
            "anova_p": anova_p,
            "kruskal_h": h_stat,
            "kruskal_p": kw_p,
            "eta_squared": eta_squared,
            "cluster_means": cluster_means,
            "significant": kw_p < 0.05,
        }

    return results
```

#### 2.3.2 Volatility Regime Detection

**Definition:** Do clusters correspond to different volatility regimes?

```python
def compute_volatility_differentiation(
    labels: np.ndarray,
    forward_volatility: np.ndarray,
) -> dict:
    """
    Test if clusters have different forward volatility.
    """
    groups = []
    for label in np.unique(labels):
        if label == -1:
            continue
        groups.append(forward_volatility[labels == label])

    if len(groups) < 2:
        return {"significant": False}

    # Levene's test for equality of variances
    levene_stat, levene_p = stats.levene(*groups)

    # Kruskal-Wallis for median differences
    h_stat, kw_p = stats.kruskal(*groups)

    return {
        "levene_stat": levene_stat,
        "levene_p": levene_p,
        "kruskal_h": h_stat,
        "kruskal_p": kw_p,
        "significant": kw_p < 0.05 or levene_p < 0.05,
    }
```

#### 2.3.3 Transition Matrix Analysis

**Definition:** Do cluster transitions follow meaningful patterns?

```python
def compute_transition_matrix(
    labels: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """
    Analyze cluster transition patterns.
    """
    # Sort by time
    sort_idx = np.argsort(timestamps)
    sorted_labels = labels[sort_idx]

    # Count transitions
    n_clusters = len(np.unique(sorted_labels[sorted_labels != -1]))
    transition_counts = np.zeros((n_clusters, n_clusters))

    for i in range(len(sorted_labels) - 1):
        if sorted_labels[i] == -1 or sorted_labels[i+1] == -1:
            continue
        transition_counts[sorted_labels[i], sorted_labels[i+1]] += 1

    # Normalize to probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_probs = np.divide(
        transition_counts, row_sums,
        where=row_sums > 0,
        out=np.zeros_like(transition_counts)
    )

    # Metrics
    # Self-transition rate (regime persistence)
    self_transition = np.diag(transition_probs).mean()

    # Entropy of transitions (predictability)
    def row_entropy(row):
        row = row[row > 0]
        return -np.sum(row * np.log(row)) if len(row) > 0 else 0

    transition_entropy = np.mean([row_entropy(row) for row in transition_probs])

    return {
        "transition_matrix": transition_probs.tolist(),
        "self_transition_rate": self_transition,
        "transition_entropy": transition_entropy,
        "avg_regime_duration": 1 / (1 - self_transition + 1e-10),
    }
```

---

## 3. Composite Quality Score

### 3.1 Weighted Quality Index

Combine multiple metrics into a single actionable score.

```python
@dataclass
class ClusterQualityScore:
    """Composite cluster quality assessment."""

    # Internal metrics (0-1 scale)
    silhouette: float          # Higher = better
    davies_bouldin: float      # Lower = better (inverted)

    # Stability metrics (0-1 scale)
    bootstrap_stability: float
    temporal_stability: float

    # External metrics (0-1 scale)
    return_significance: float  # 1 if p < 0.05, scaled otherwise
    volatility_significance: float

    def compute_composite(self, weights: dict = None) -> float:
        """
        Compute weighted composite score.

        Default weights emphasize stability and predictive power
        over internal metrics alone.
        """
        if weights is None:
            weights = {
                "silhouette": 0.15,
                "davies_bouldin": 0.10,
                "bootstrap_stability": 0.20,
                "temporal_stability": 0.20,
                "return_significance": 0.20,
                "volatility_significance": 0.15,
            }

        # Normalize davies_bouldin (invert, cap at 2)
        db_normalized = 1 - min(self.davies_bouldin, 2) / 2

        score = (
            weights["silhouette"] * self.silhouette +
            weights["davies_bouldin"] * db_normalized +
            weights["bootstrap_stability"] * self.bootstrap_stability +
            weights["temporal_stability"] * self.temporal_stability +
            weights["return_significance"] * self.return_significance +
            weights["volatility_significance"] * self.volatility_significance
        )

        return score

    def get_grade(self) -> str:
        """Human-readable quality grade."""
        score = self.compute_composite()
        if score >= 0.8:
            return "A - Excellent: Ready for HMM"
        elif score >= 0.6:
            return "B - Good: Minor refinements needed"
        elif score >= 0.4:
            return "C - Fair: Significant refinements needed"
        elif score >= 0.2:
            return "D - Poor: Consider different features"
        else:
            return "F - Failed: No meaningful structure"
```

### 3.2 Quality Thresholds for HMM Readiness

| Metric | Minimum for HMM | Target |
|--------|-----------------|--------|
| Silhouette | 0.3 | 0.5+ |
| Davies-Bouldin | < 1.5 | < 1.0 |
| Bootstrap Stability | 0.6 | 0.8+ |
| Temporal Stability | 0.5 | 0.7+ |
| Return p-value | < 0.10 | < 0.01 |
| Composite Score | 0.4 | 0.6+ |

---

## 4. Agentic Feature Refinement Loop

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     REFINEMENT AGENT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                               │
│  │ Feature Pool │  All available features from NAT             │
│  │ (200+)       │  - entropy, flow, orderbook, regime, etc.    │
│  └──────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Selector   │───▶│  Evaluator   │───▶│   Refiner    │      │
│  │              │    │              │    │              │      │
│  │ Select subset│    │ Cluster &    │    │ Modify based │      │
│  │ of features  │    │ measure      │    │ on feedback  │      │
│  └──────────────┘    │ quality      │    └──────────────┘      │
│         ▲            └──────────────┘           │               │
│         │                   │                   │               │
│         │                   ▼                   │               │
│         │            ┌──────────────┐           │               │
│         │            │  Quality OK? │           │               │
│         │            └──────────────┘           │               │
│         │                   │                   │               │
│         │         NO        │       YES         │               │
│         └───────────────────┘        │          │               │
│                                      ▼          │               │
│                              ┌──────────────┐   │               │
│                              │    OUTPUT    │◀──┘               │
│                              │ Best features│                   │
│                              │ + quality    │                   │
│                              └──────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Refinement Strategies

```python
class RefinementStrategy:
    """Base class for feature refinement strategies."""

    @abstractmethod
    def refine(self, current_features: List[str], quality: ClusterQualityScore) -> List[str]:
        pass

class AddInteractionFeatures(RefinementStrategy):
    """Add interaction terms when clusters overlap."""

    def refine(self, features, quality):
        if quality.silhouette < 0.3:
            # Clusters overlap - try interactions
            new_features = []
            for f1, f2 in combinations(features[:5], 2):
                new_features.append(f"{f1}_x_{f2}")  # Multiplication
                new_features.append(f"{f1}_div_{f2}")  # Ratio
            return features + new_features
        return features

class IncreaseWindowSize(RefinementStrategy):
    """Increase time windows when clusters are unstable."""

    def refine(self, features, quality):
        if quality.bootstrap_stability < 0.6:
            # Noisy features - use longer windows
            return [f.replace("_60", "_300").replace("_300", "_600")
                    for f in features]
        return features

class RemoveCorrelatedFeatures(RefinementStrategy):
    """Remove redundant features when many exist."""

    def refine(self, features, quality, X):
        if len(features) > 20:
            # Too many features - remove correlated
            corr = np.corrcoef(X.T)
            to_remove = set()
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    if abs(corr[i,j]) > 0.9:
                        to_remove.add(features[j])
            return [f for f in features if f not in to_remove]
        return features

class TryDifferentSubset(RefinementStrategy):
    """Switch to different feature category."""

    def __init__(self, subsets: List[str]):
        self.subsets = subsets
        self.tried = set()

    def refine(self, features, quality):
        if quality.return_significance < 0.5:
            # Current features don't predict returns
            for subset in self.subsets:
                if subset not in self.tried:
                    self.tried.add(subset)
                    return get_features_for_subset(subset)
        return features
```

### 4.3 Agent Implementation

```python
class FeatureRefinementAgent:
    """
    Iteratively refines feature selection to maximize cluster quality.

    Stopping conditions:
    1. Quality score exceeds threshold
    2. Maximum iterations reached
    3. No improvement for N iterations
    """

    def __init__(
        self,
        feature_pool: List[str],
        quality_threshold: float = 0.6,
        max_iterations: int = 50,
        patience: int = 10,
    ):
        self.feature_pool = feature_pool
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.patience = patience

        self.strategies = [
            AddInteractionFeatures(),
            IncreaseWindowSize(),
            RemoveCorrelatedFeatures(),
            TryDifferentSubset(["entropy", "flow", "regime", "illiquidity"]),
        ]

        self.history = []

    def evaluate(self, X: np.ndarray, returns: np.ndarray) -> ClusterQualityScore:
        """Full quality evaluation of feature set."""
        # Cluster
        gmm = GaussianMixture(n_components=5, random_state=42)
        labels = gmm.fit_predict(X)

        # Internal metrics
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)

        # Stability
        def cluster_func(X_sub):
            return GaussianMixture(n_components=5, random_state=42).fit_predict(X_sub)

        bootstrap = compute_bootstrap_stability(X, cluster_func)
        temporal = compute_temporal_stability(X, timestamps, cluster_func)

        # External
        return_diff = compute_return_differentiation(labels, returns)

        return ClusterQualityScore(
            silhouette=sil,
            davies_bouldin=db,
            bootstrap_stability=bootstrap["mean_ari"],
            temporal_stability=temporal["temporal_ari"],
            return_significance=1.0 if return_diff[3600]["kruskal_p"] < 0.05 else 0.0,
            volatility_significance=0.5,  # Placeholder
        )

    def select_strategy(self, quality: ClusterQualityScore) -> RefinementStrategy:
        """Choose refinement strategy based on quality weaknesses."""

        if quality.silhouette < 0.3:
            return self.strategies[0]  # AddInteractionFeatures
        elif quality.bootstrap_stability < 0.6:
            return self.strategies[1]  # IncreaseWindowSize
        elif quality.return_significance < 0.5:
            return self.strategies[3]  # TryDifferentSubset
        else:
            return self.strategies[2]  # RemoveCorrelatedFeatures

    def run(self, data: pl.DataFrame, returns: np.ndarray) -> dict:
        """
        Main refinement loop.

        Returns:
            best_features: Optimal feature set
            quality: Final quality score
            history: Full optimization history
        """
        current_features = self.feature_pool[:10]  # Start with first 10
        best_quality = None
        best_features = None
        no_improvement = 0

        for iteration in range(self.max_iterations):
            # Extract features
            X = extract_features(data, current_features)

            # Evaluate
            quality = self.evaluate(X, returns)
            composite = quality.compute_composite()

            # Track history
            self.history.append({
                "iteration": iteration,
                "features": current_features.copy(),
                "quality": quality,
                "composite": composite,
            })

            # Check if best
            if best_quality is None or composite > best_quality.compute_composite():
                best_quality = quality
                best_features = current_features.copy()
                no_improvement = 0
            else:
                no_improvement += 1

            # Check stopping conditions
            if composite >= self.quality_threshold:
                print(f"Quality threshold reached at iteration {iteration}")
                break

            if no_improvement >= self.patience:
                print(f"No improvement for {self.patience} iterations, stopping")
                break

            # Refine
            strategy = self.select_strategy(quality)
            current_features = strategy.refine(current_features, quality)

            print(f"Iteration {iteration}: composite={composite:.3f}, "
                  f"strategy={strategy.__class__.__name__}")

        return {
            "best_features": best_features,
            "quality": best_quality,
            "history": self.history,
        }
```

---

## 5. Implementation Plan

### 5.1 Phase 1: Core Metrics (Priority: High)

**File:** `scripts/cluster_quality.py`

```
[ ] Implement silhouette computation with per-cluster breakdown
[ ] Implement Davies-Bouldin index
[ ] Implement Calinski-Harabasz index
[ ] Implement Gap statistic
[ ] Create unified QualityMetrics dataclass
```

### 5.2 Phase 2: Stability Metrics (Priority: High)

**File:** `scripts/cluster_stability.py`

```
[ ] Implement bootstrap stability
[ ] Implement temporal stability (train/test split)
[ ] Implement cross-symbol stability (BTC vs ETH)
```

### 5.3 Phase 3: External Validation (Priority: High)

**File:** `scripts/cluster_validation.py`

```
[ ] Implement forward return differentiation (ANOVA + Kruskal-Wallis)
[ ] Implement volatility regime detection
[ ] Implement transition matrix analysis
[ ] Add statistical significance reporting
```

### 5.4 Phase 4: Composite Scoring (Priority: Medium)

**File:** `scripts/cluster_score.py`

```
[ ] Implement ClusterQualityScore dataclass
[ ] Implement weighted composite scoring
[ ] Implement quality grading (A/B/C/D/F)
[ ] Create quality report generator
```

### 5.5 Phase 5: Agentic Refinement (Priority: Low)

**File:** `scripts/feature_refinement_agent.py`

```
[ ] Implement base RefinementStrategy
[ ] Implement specific strategies (interaction, window, subset)
[ ] Implement FeatureRefinementAgent
[ ] Add logging and visualization of refinement history
```

---

## 6. Usage Examples

### 6.1 Basic Quality Check

```bash
# Measure cluster quality for entropy features
python scripts/cluster_quality.py \
    --data-dir rust/data/features \
    --subset entropy \
    --output analysis/quality_entropy.json
```

### 6.2 Full Quality Report

```bash
# Generate comprehensive quality report
python scripts/cluster_quality.py \
    --data-dir rust/data/features \
    --all-subsets \
    --include-stability \
    --include-returns \
    --output analysis/full_quality_report.json
```

### 6.3 Agentic Refinement

```bash
# Run agentic feature refinement
python scripts/feature_refinement_agent.py \
    --data-dir rust/data/features \
    --quality-threshold 0.6 \
    --max-iterations 50 \
    --output analysis/refined_features.json
```

---

## 7. Success Criteria

### 7.1 Framework Validation

The framework is successful if:

1. **Reproducible:** Same data produces same quality scores
2. **Discriminative:** Can distinguish good clusters from random
3. **Predictive:** Quality scores correlate with downstream HMM performance
4. **Actionable:** Refinement suggestions lead to improvement

### 7.2 Feature Validation

Features are ready for HMM when:

| Criterion | Threshold |
|-----------|-----------|
| Composite Quality Score | >= 0.5 |
| Silhouette Score | >= 0.3 |
| Bootstrap Stability | >= 0.6 |
| Temporal Stability | >= 0.5 |
| Return Differentiation p-value | < 0.05 |
| Number of stable clusters | 3-7 |

---

## 8. References

1. Rousseeuw, P.J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis"
2. Davies, D.L. & Bouldin, D.W. (1979). "A Cluster Separation Measure"
3. Tibshirani, R., Walther, G., & Hastie, T. (2001). "Estimating the number of clusters in a data set via the gap statistic"
4. Ben-Hur, A., Elisseeff, A., & Guyon, I. (2001). "A stability based method for discovering structure in clustered data"
5. Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle"

---

## Appendix A: Metric Summary Table

| Metric | Type | Range | Better | Use Case |
|--------|------|-------|--------|----------|
| Silhouette | Internal | [-1, 1] | Higher | Overall cluster quality |
| Davies-Bouldin | Internal | [0, ∞) | Lower | Cluster separation |
| Calinski-Harabasz | Internal | [0, ∞) | Higher | Variance ratio |
| Gap Statistic | Internal | [0, ∞) | Higher | Optimal k selection |
| Bootstrap Stability | Stability | [0, 1] | Higher | Robustness to sampling |
| Temporal Stability | Stability | [0, 1] | Higher | Consistency over time |
| Return p-value | External | [0, 1] | Lower | Predictive power |
| Composite Score | Combined | [0, 1] | Higher | Overall readiness |
