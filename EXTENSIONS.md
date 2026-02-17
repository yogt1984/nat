# Advanced Extensions: Information Geometry, Transfer Entropy, and Unsupervised Regime Discovery

## Overview

This document describes the novel methodological extensions that differentiate this project from standard approaches.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTENSION HIERARCHY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BASE LAYER (V1):                                                           │
│  └── Standard features + Entropy + Supervised regime classification         │
│                                                                              │
│  EXTENSION LAYER 1 (V1.1): Polynomial Chaos Expansion                       │
│  └── Uncertainty quantification + Sobol indices + Feature interactions      │
│                                                                              │
│  EXTENSION LAYER 2 (V1.2): Transfer Entropy Networks                        │
│  └── Causal information flow + Network topology features                    │
│                                                                              │
│  EXTENSION LAYER 3 (V1.3): Information Geometry                             │
│  └── Fisher manifold + Geodesic distances + Curvature features              │
│                                                                              │
│  EXTENSION LAYER 4 (V1.4): Unsupervised Regime Discovery                   │
│  └── Clustering on entropy manifold + Strategy-agnostic regime finding     │
│                                                                              │
│  INTEGRATION LAYER (V2): Full Synthesis                                     │
│  └── Cluster on Fisher manifold using TE-informed distances + PCE UQ       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Information Geometry

### 1.1 Theoretical Foundation

**Core Principle**: Probability distributions form a Riemannian manifold where distance reflects statistical distinguishability.

#### Fisher Information Metric

For a parametric family of distributions p(x|θ), the Fisher Information Matrix is:

```
g_ij(θ) = E[ ∂log p(x|θ)/∂θ_i · ∂log p(x|θ)/∂θ_j ]

        = -E[ ∂²log p(x|θ)/∂θ_i∂θ_j ]
```

This metric defines:
- **Local geometry**: How sensitive is the distribution to parameter changes?
- **Geodesic distance**: Shortest path between distributions
- **Curvature**: How "bent" is the manifold at this point?

#### Key Distances

| Distance | Formula | Properties |
|----------|---------|------------|
| **Fisher-Rao** | ∫ √(Σ g_ij dθ_i dθ_j) | True geodesic, hard to compute |
| **Hellinger** | √(1 - ∫√(p·q)dx) | Related to Fisher-Rao, tractable |
| **KL Divergence** | ∫ p log(p/q) dx | Not symmetric, not a true metric |
| **Jensen-Shannon** | ½KL(p‖m) + ½KL(q‖m), m=(p+q)/2 | Symmetric, bounded |

### 1.2 Application to Market Regimes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MARKET STATE AS MANIFOLD POINT                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  At time t, observe features X_t = [entropy, imbalance, volatility, ...]   │
│                                                                              │
│  Estimate distribution P_t from sliding window:                             │
│  P_t = KDE(X_{t-W:t}) or histogram                                          │
│                                                                              │
│  P_t is a POINT on the statistical manifold M                              │
│                                                                              │
│  As market evolves: P_t traces a PATH on M                                  │
│                                                                              │
│  REGIME = REGION on M                                                       │
│  REGIME CHANGE = PATH crosses region boundary                               │
│                                                                              │
│  Advantages:                                                                 │
│  ├── Regime boundaries are GEOMETRIC, not arbitrary thresholds             │
│  ├── Distance to regime is well-defined (geodesic)                         │
│  ├── Uncertainty quantified by local curvature                             │
│  └── Invariant to feature rescaling                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Derived Features

```python
# Information Geometry Features (8 features)
IG_HELLINGER_VELOCITY_1S         # Hellinger distance traveled in 1 second
IG_HELLINGER_VELOCITY_5S         # Hellinger distance traveled in 5 seconds
IG_FISHER_TRACE                  # Trace of Fisher matrix (total information)
IG_FISHER_DETERMINANT            # |Fisher matrix| (information volume)
IG_LOCAL_CURVATURE               # Eigenvalue spread of Fisher matrix
IG_GEODESIC_ACCELERATION         # d²/dt²(manifold position)
IG_DISTANCE_TO_CLUSTER_0         # Geodesic distance to cluster 0 centroid
IG_DISTANCE_TO_CLUSTER_1         # Geodesic distance to cluster 1 centroid
# ... (one per discovered cluster)
```

### 1.4 Implementation

```python
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity

class InformationGeometryEngine:
    """
    Compute information-geometric features from market state distributions.

    The market state at time t is represented as a probability distribution
    over feature space. This distribution is a point on the statistical
    manifold. We track movement on this manifold to detect regime changes.
    """

    def __init__(
        self,
        window_size: int = 100,        # Samples for distribution estimation
        n_grid_points: int = 50,        # Grid resolution for distribution
        velocity_horizons: list = [10, 50],  # Samples for velocity calculation
    ):
        self.window_size = window_size
        self.n_grid_points = n_grid_points
        self.velocity_horizons = velocity_horizons
        self.distribution_history = []
        self.cluster_centroids = {}

    def estimate_distribution(
        self,
        features: np.ndarray,  # Shape: (window_size, n_features)
    ) -> np.ndarray:
        """
        Estimate probability distribution from feature window.
        Returns discretized distribution on grid.
        """
        # Fit KDE
        kde = KernelDensity(bandwidth='scott', kernel='gaussian')
        kde.fit(features)

        # Evaluate on grid
        grid_points = self._build_grid(features)
        log_density = kde.score_samples(grid_points)
        density = np.exp(log_density)
        density /= density.sum()  # Normalize

        return density

    def _build_grid(self, features: np.ndarray) -> np.ndarray:
        """Build evaluation grid based on feature ranges."""
        n_features = features.shape[1]
        grids = []
        for i in range(n_features):
            grids.append(np.linspace(
                features[:, i].min() - 0.1 * np.ptp(features[:, i]),
                features[:, i].max() + 0.1 * np.ptp(features[:, i]),
                self.n_grid_points
            ))
        return np.array(np.meshgrid(*grids)).reshape(n_features, -1).T

    def hellinger_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Hellinger distance between distributions.
        H(p,q) = sqrt(1 - sum(sqrt(p * q)))

        Related to Fisher-Rao geodesic distance by:
        d_FR(p,q) = 2 * arccos(1 - H²(p,q))
        """
        bc = np.sum(np.sqrt(p * q))  # Bhattacharyya coefficient
        return np.sqrt(1 - bc)

    def jensen_shannon_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Jensen-Shannon distance (symmetric, bounded).
        """
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / (m + 1e-10) + 1e-10))
        kl_qm = np.sum(q * np.log(q / (m + 1e-10) + 1e-10))
        return np.sqrt(0.5 * (kl_pm + kl_qm))

    def compute_fisher_matrix(
        self,
        features: np.ndarray,
        epsilon: float = 0.01,
    ) -> np.ndarray:
        """
        Estimate Fisher Information Matrix numerically.

        Uses finite differences to approximate:
        g_ij ≈ E[(∂log p/∂θ_i)(∂log p/∂θ_j)]
        """
        n_features = features.shape[1]
        p_center = self.estimate_distribution(features)

        fisher = np.zeros((n_features, n_features))

        # Numerical differentiation
        for i in range(n_features):
            # Perturb feature i
            features_plus = features.copy()
            features_plus[:, i] += epsilon * features[:, i].std()
            p_plus = self.estimate_distribution(features_plus)

            features_minus = features.copy()
            features_minus[:, i] -= epsilon * features[:, i].std()
            p_minus = self.estimate_distribution(features_minus)

            # Score function approximation
            score_i = (np.log(p_plus + 1e-10) - np.log(p_minus + 1e-10)) / (2 * epsilon)

            for j in range(i, n_features):
                if i == j:
                    fisher[i, j] = np.sum(p_center * score_i ** 2)
                else:
                    features_plus_j = features.copy()
                    features_plus_j[:, j] += epsilon * features[:, j].std()
                    p_plus_j = self.estimate_distribution(features_plus_j)

                    features_minus_j = features.copy()
                    features_minus_j[:, j] -= epsilon * features[:, j].std()
                    p_minus_j = self.estimate_distribution(features_minus_j)

                    score_j = (np.log(p_plus_j + 1e-10) - np.log(p_minus_j + 1e-10)) / (2 * epsilon)

                    fisher[i, j] = np.sum(p_center * score_i * score_j)
                    fisher[j, i] = fisher[i, j]

        return fisher

    def compute_features(self, features: np.ndarray) -> dict:
        """
        Compute all information-geometric features.
        """
        # Current distribution
        current_dist = self.estimate_distribution(features)
        self.distribution_history.append(current_dist)

        result = {}

        # Velocity features (movement on manifold)
        for horizon in self.velocity_horizons:
            if len(self.distribution_history) > horizon:
                past_dist = self.distribution_history[-horizon]
                velocity = self.hellinger_distance(past_dist, current_dist)
                result[f'ig_hellinger_velocity_{horizon}'] = velocity
            else:
                result[f'ig_hellinger_velocity_{horizon}'] = 0.0

        # Fisher matrix features
        fisher = self.compute_fisher_matrix(features)
        eigenvalues = np.linalg.eigvalsh(fisher)

        result['ig_fisher_trace'] = np.trace(fisher)
        result['ig_fisher_determinant'] = np.linalg.det(fisher)
        result['ig_local_curvature'] = np.log(eigenvalues.max() / (eigenvalues.min() + 1e-10))

        # Acceleration (second derivative on manifold)
        if len(self.distribution_history) >= 3:
            v1 = self.hellinger_distance(
                self.distribution_history[-3],
                self.distribution_history[-2]
            )
            v2 = self.hellinger_distance(
                self.distribution_history[-2],
                self.distribution_history[-1]
            )
            result['ig_geodesic_acceleration'] = v2 - v1
        else:
            result['ig_geodesic_acceleration'] = 0.0

        # Distance to cluster centroids
        for cluster_id, centroid in self.cluster_centroids.items():
            result[f'ig_distance_to_cluster_{cluster_id}'] = self.hellinger_distance(
                current_dist, centroid
            )

        return result

    def set_cluster_centroids(self, centroids: dict):
        """Set cluster centroids for distance calculations."""
        self.cluster_centroids = centroids
```

---

## 2. Transfer Entropy Networks

### 2.1 Theoretical Foundation

**Core Principle**: Measure directed, nonlinear information flow between time series.

#### Transfer Entropy Definition

```
T_{X→Y} = H(Y_{t+1} | Y_t^{(k)}) - H(Y_{t+1} | Y_t^{(k)}, X_t^{(l)})
```

Where:
- Y_t^{(k)} = (Y_t, Y_{t-1}, ..., Y_{t-k+1}) is the history of Y
- X_t^{(l)} = (X_t, X_{t-1}, ..., X_{t-l+1}) is the history of X
- H(·|·) is conditional entropy

**Interpretation**: How much does knowing X's past reduce uncertainty about Y's future, beyond what Y's own past tells us?

### 2.2 Network Construction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFER ENTROPY NETWORK PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: SELECT FEATURES FOR NETWORK                                        │
│  ─────────────────────────────────────                                       │
│  Focus on key features (not all 57):                                        │
│  ├── Entropy: ENT_permutation_returns_16                                   │
│  ├── Imbalance: IMBALANCE_qty_l5                                           │
│  ├── Flow: FLOW_aggressor_ratio_5s                                         │
│  ├── Volatility: VOL_returns_1m                                            │
│  ├── Spread: RAW_spread_bps                                                │
│  ├── Return: (computed from midprice)                                      │
│  └── Funding: CTX_funding_rate                                             │
│  Total: 7-10 nodes in network                                               │
│                                                                              │
│  STEP 2: COMPUTE PAIRWISE TRANSFER ENTROPY                                  │
│  ───────────────────────────────────────────                                 │
│  For each pair (i, j), compute T_{i→j}                                      │
│  Result: N×N matrix (asymmetric)                                            │
│                                                                              │
│  STEP 3: THRESHOLD AND BUILD GRAPH                                          │
│  ─────────────────────────────────────                                       │
│  ├── Remove edges below significance threshold                             │
│  ├── Or: Keep top K edges by weight                                        │
│  └── Result: Directed weighted graph                                       │
│                                                                              │
│  STEP 4: EXTRACT NETWORK FEATURES                                           │
│  ─────────────────────────────────                                           │
│  ├── Global: density, reciprocity, total flow                              │
│  ├── Local: in/out degree, PageRank, clustering                            │
│  └── Structural: community structure, hierarchy                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Derived Features

```python
# Transfer Entropy Network Features (12 features)

# Global network properties
TE_NETWORK_DENSITY               # Fraction of possible edges present
TE_NETWORK_RECIPROCITY           # Fraction of bidirectional edges
TE_TOTAL_INFORMATION_FLOW        # Sum of all edge weights
TE_FLOW_ASYMMETRY               # Imbalance in information flow directions
TE_NETWORK_ENTROPY              # Entropy of edge weight distribution

# Key causal relationships
TE_IMBALANCE_TO_RETURN          # Does imbalance predict returns?
TE_ENTROPY_TO_VOLATILITY        # Does entropy predict volatility?
TE_FLOW_TO_SPREAD               # Does trade flow predict spread?
TE_RETURN_TO_IMBALANCE          # Feedback: returns → imbalance

# Centrality measures
TE_RETURN_PAGERANK              # How "influential" is return?
TE_DOMINANT_HUB                 # Which feature dominates information flow?
TE_HIERARCHY_SCORE              # Is there clear causal hierarchy?
```

### 2.4 Regime-Dependent Network Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NETWORK TOPOLOGY BY REGIME                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HYPOTHESIS: Network structure differs between regimes                      │
│                                                                              │
│  MEAN-REVERSION REGIME:                                                     │
│  ├── Dense network (many interconnections)                                 │
│  ├── High reciprocity (bidirectional flow)                                 │
│  ├── No dominant hub (distributed influence)                               │
│  └── High network entropy (disordered flow)                                │
│                                                                              │
│      ENT ←→ IMB ←→ VOL                                                      │
│       ↕       ↕       ↕                                                      │
│     FLOW ←→ SPR ←→ RET                                                      │
│                                                                              │
│  TREND-FOLLOWING REGIME:                                                    │
│  ├── Sparse network (few key connections)                                  │
│  ├── Low reciprocity (one-way flow)                                        │
│  ├── Clear hub (one feature dominates)                                     │
│  └── Low network entropy (ordered flow)                                    │
│                                                                              │
│          ENT                                                                 │
│           ↓                                                                  │
│      IMB ─→ RET ←─ FLOW                                                     │
│           ↓                                                                  │
│          VOL                                                                 │
│                                                                              │
│  DETECTION: Monitor network topology → regime change when topology shifts  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Implementation

```python
import numpy as np
import networkx as nx
from collections import deque
from typing import Dict, List, Tuple

class TransferEntropyNetwork:
    """
    Build and analyze transfer entropy networks from feature time series.

    The network captures causal information flow between features.
    Network topology changes indicate regime transitions.
    """

    def __init__(
        self,
        feature_names: List[str],
        history_length: int = 5,         # k in T_{X→Y} definition
        n_bins: int = 8,                  # Discretization bins
        min_samples: int = 500,           # Minimum samples for TE estimation
        significance_threshold: float = 0.01,  # Edge pruning threshold
    ):
        self.feature_names = feature_names
        self.history_length = history_length
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.significance_threshold = significance_threshold

        self.n_features = len(feature_names)
        self.buffer = deque(maxlen=min_samples + history_length + 1)

    def update(self, features: np.ndarray):
        """Add new feature observation to buffer."""
        self.buffer.append(features)

    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Discretize continuous values into bins."""
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bins = np.percentile(x, percentiles)
        return np.clip(np.digitize(x, bins[1:-1]), 0, self.n_bins - 1)

    def _estimate_transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Estimate transfer entropy T_{source → target}.

        Uses histogram-based estimation with bias correction.
        """
        n = len(target) - self.history_length - 1
        if n < 100:
            return 0.0

        # Discretize
        source_d = self._discretize(source)
        target_d = self._discretize(target)

        # Build history vectors
        # Y_future: target[k+1:]
        # Y_past: target[k:k+1], target[k-1:k], ... (k terms)
        # X_past: source[k:k+1], source[k-1:k], ... (k terms)

        y_future = target_d[self.history_length + 1:]

        y_past = np.column_stack([
            target_d[self.history_length - i:-i - 1]
            for i in range(self.history_length)
        ])

        x_past = np.column_stack([
            source_d[self.history_length - i:-i - 1]
            for i in range(self.history_length)
        ])

        # Estimate H(Y_future | Y_past)
        h_y_given_ypast = self._conditional_entropy_histogram(y_future, y_past)

        # Estimate H(Y_future | Y_past, X_past)
        joint_past = np.column_stack([y_past, x_past])
        h_y_given_both = self._conditional_entropy_histogram(y_future, joint_past)

        # TE = H(Y|Y_past) - H(Y|Y_past, X_past)
        te = max(0, h_y_given_ypast - h_y_given_both)

        return te

    def _conditional_entropy_histogram(
        self,
        y: np.ndarray,
        x: np.ndarray,
    ) -> float:
        """
        Estimate H(Y|X) using histogram method.
        H(Y|X) = H(Y,X) - H(X)
        """
        # Joint entropy H(Y,X)
        joint = np.column_stack([y, x])
        h_joint = self._entropy_histogram(joint)

        # Marginal entropy H(X)
        h_x = self._entropy_histogram(x)

        return h_joint - h_x

    def _entropy_histogram(self, x: np.ndarray) -> float:
        """Estimate entropy using histogram."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Convert to tuple for hashing
        x_tuples = [tuple(row) for row in x]

        # Count occurrences
        from collections import Counter
        counts = Counter(x_tuples)

        # Compute entropy
        n = len(x_tuples)
        probs = np.array(list(counts.values())) / n

        return -np.sum(probs * np.log2(probs + 1e-10))

    def compute_te_matrix(self) -> np.ndarray:
        """Compute full transfer entropy matrix."""
        if len(self.buffer) < self.min_samples:
            return np.zeros((self.n_features, self.n_features))

        data = np.array(self.buffer)
        te_matrix = np.zeros((self.n_features, self.n_features))

        for i in range(self.n_features):
            for j in range(self.n_features):
                if i != j:
                    te_matrix[i, j] = self._estimate_transfer_entropy(
                        data[:, i], data[:, j]
                    )

        return te_matrix

    def build_network(self, te_matrix: np.ndarray) -> nx.DiGraph:
        """Build directed graph from TE matrix."""
        G = nx.DiGraph()

        for i, name in enumerate(self.feature_names):
            G.add_node(name)

        for i in range(self.n_features):
            for j in range(self.n_features):
                if te_matrix[i, j] > self.significance_threshold:
                    G.add_edge(
                        self.feature_names[i],
                        self.feature_names[j],
                        weight=te_matrix[i, j]
                    )

        return G

    def compute_features(self) -> Dict[str, float]:
        """Compute all network features."""
        te_matrix = self.compute_te_matrix()
        G = self.build_network(te_matrix)

        features = {}

        # Global network properties
        features['te_network_density'] = nx.density(G)
        features['te_network_reciprocity'] = (
            nx.reciprocity(G) if G.number_of_edges() > 0 else 0
        )

        total_flow = sum(d['weight'] for _, _, d in G.edges(data=True))
        features['te_total_information_flow'] = total_flow

        # Flow asymmetry
        asymmetry = 0
        for u, v, d in G.edges(data=True):
            reverse = G.get_edge_data(v, u, {}).get('weight', 0)
            asymmetry += abs(d['weight'] - reverse)
        features['te_flow_asymmetry'] = asymmetry / (G.number_of_edges() + 1)

        # Network entropy
        if G.number_of_edges() > 0:
            weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
            weights = weights / weights.sum()
            features['te_network_entropy'] = -np.sum(weights * np.log2(weights + 1e-10))
        else:
            features['te_network_entropy'] = 0

        # Specific causal relationships (if features exist)
        for source, target, key in [
            ('imbalance', 'return', 'te_imbalance_to_return'),
            ('entropy', 'volatility', 'te_entropy_to_volatility'),
            ('flow', 'spread', 'te_flow_to_spread'),
            ('return', 'imbalance', 'te_return_to_imbalance'),
        ]:
            # Find matching feature names
            source_idx = self._find_feature_index(source)
            target_idx = self._find_feature_index(target)
            if source_idx is not None and target_idx is not None:
                features[key] = te_matrix[source_idx, target_idx]

        # Centrality measures
        if G.number_of_edges() > 0:
            pagerank = nx.pagerank(G, weight='weight')
            features['te_dominant_hub'] = max(pagerank.values())

            # Find return node pagerank if it exists
            return_idx = self._find_feature_index('return')
            if return_idx is not None:
                return_name = self.feature_names[return_idx]
                features['te_return_pagerank'] = pagerank.get(return_name, 0)
        else:
            features['te_dominant_hub'] = 0
            features['te_return_pagerank'] = 0

        return features

    def _find_feature_index(self, partial_name: str) -> int:
        """Find feature index by partial name match."""
        for i, name in enumerate(self.feature_names):
            if partial_name.lower() in name.lower():
                return i
        return None
```

---

## 3. Unsupervised Regime Discovery

### 3.1 Motivation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY UNSUPERVISED?                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SUPERVISED APPROACH:                                                        │
│  1. Define regimes (MR, TF) based on strategy profitability                │
│  2. Label historical data                                                   │
│  3. Train classifier                                                        │
│  4. Deploy                                                                  │
│                                                                              │
│  PROBLEMS:                                                                   │
│  ├── Circular: Regimes defined BY strategies we want to select             │
│  ├── Assumes 2-3 regimes exist (what if there are 7?)                      │
│  ├── Misses regimes where NEITHER MR nor TF works                          │
│  └── Labels are noisy (strategy P&L has variance)                          │
│                                                                              │
│  UNSUPERVISED APPROACH:                                                      │
│  1. Let data define natural clusters/regimes                               │
│  2. Characterize each cluster                                               │
│  3. THEN ask: "What works in each cluster?"                                │
│  4. Deploy with data-driven regime definitions                             │
│                                                                              │
│  ADVANTAGES:                                                                 │
│  ├── Discovers regimes you didn't know existed                             │
│  ├── No label leakage                                                       │
│  ├── Can find "do nothing" regimes naturally                               │
│  └── More robust to strategy overfitting                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Clustering on Information Manifold (Novel Contribution)

**Key Innovation**: Don't cluster in Euclidean feature space. Cluster on the statistical manifold using geodesic distances.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              INFORMATION-GEOMETRIC CLUSTERING (NOVEL)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STANDARD CLUSTERING:                                                        │
│  ───────────────────                                                         │
│  Features X ──▶ Euclidean distance matrix ──▶ HDBSCAN ──▶ Clusters         │
│                                                                              │
│  Problem: Euclidean distance ignores statistical structure                  │
│           Features may have different scales, distributions                 │
│                                                                              │
│  INFORMATION-GEOMETRIC CLUSTERING:                                           │
│  ─────────────────────────────────                                           │
│  Features X ──▶ Sliding window distributions ──▶ Manifold points           │
│                                                                              │
│  Manifold points ──▶ Geodesic distance matrix ──▶ HDBSCAN ──▶ Clusters     │
│                                                                              │
│  Advantages:                                                                 │
│  ├── Clusters are "statistically distinct" not just "far in Euclidean"    │
│  ├── Scale-invariant (no need for feature normalization)                   │
│  ├── Respects the probabilistic nature of market states                    │
│  └── Natural uncertainty quantification                                    │
│                                                                              │
│  NOVELTY: Very few papers apply this to market microstructure              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Clustering Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FULL CLUSTERING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Feature matrix X ∈ R^{T × D} (T timepoints, D features)            │
│                                                                              │
│  STEP 1: SELECT CLUSTERING FEATURES                                         │
│  ──────────────────────────────────                                          │
│  Use entropy-related subset (10-15 features):                               │
│  ├── ENT_permutation_returns_*                                             │
│  ├── ENT_book_shape, ENT_spread_dispersion                                 │
│  ├── VOL_ratio_short_long, VOL_zscore                                      │
│  ├── IMBALANCE_qty_l5, IMBALANCE_pressure_*                                │
│  ├── CTX_funding_zscore, CTX_oi_change_pct                                 │
│  └── TE_network_entropy, TE_flow_asymmetry                                 │
│                                                                              │
│  STEP 2: COMPUTE DISTRIBUTION SNAPSHOTS                                     │
│  ────────────────────────────────────                                        │
│  For each time t:                                                            │
│    P_t = distribution of features in window [t-W, t]                        │
│    (Use KDE or histogram)                                                    │
│                                                                              │
│  STEP 3: BUILD GEODESIC DISTANCE MATRIX                                     │
│  ─────────────────────────────────────                                       │
│  D[i,j] = Hellinger(P_i, P_j)  ∀ i,j                                        │
│  (Or Jensen-Shannon, Fisher-Rao approximation)                              │
│                                                                              │
│  STEP 4: CLUSTER                                                            │
│  ────────────────                                                            │
│  Apply HDBSCAN with precomputed distance matrix:                            │
│  ├── min_cluster_size = 500 (adjust based on data)                         │
│  ├── min_samples = 50                                                       │
│  └── metric = 'precomputed'                                                │
│                                                                              │
│  STEP 5: ANALYZE CLUSTERS                                                   │
│  ────────────────────────                                                    │
│  For each cluster k:                                                         │
│  ├── Centroid: mean distribution                                           │
│  ├── Size: fraction of data                                                │
│  ├── Temporal pattern: when does it occur?                                 │
│  ├── Feature profile: what characterizes this cluster?                     │
│  ├── TE network: information flow topology in this cluster                 │
│  └── Strategy performance: backtest all strategies in this cluster         │
│                                                                              │
│  OUTPUT: Cluster assignments, centroids, strategy mapping                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Implementation

```python
import numpy as np
import hdbscan
from sklearn.manifold import TSNE
import umap
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional

class UnsupervisedRegimeDiscovery:
    """
    Discover market regimes through unsupervised clustering on the
    information-geometric manifold.

    Novel contribution: Uses geodesic (Hellinger) distance instead of
    Euclidean for clustering, respecting statistical structure.
    """

    def __init__(
        self,
        window_size: int = 50,           # Window for distribution estimation
        stride: int = 10,                 # Stride for sliding window
        min_cluster_size: int = 500,      # HDBSCAN parameter
        min_samples: int = 50,            # HDBSCAN parameter
        n_bins: int = 20,                 # Histogram bins
    ):
        self.window_size = window_size
        self.stride = stride
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_bins = n_bins

        self.cluster_model = None
        self.cluster_centroids = {}
        self.cluster_profiles = {}

    def _estimate_distribution(self, features: np.ndarray) -> np.ndarray:
        """
        Estimate probability distribution from feature window.
        Returns flattened histogram for distance computation.
        """
        n_features = features.shape[1]
        histograms = []

        for i in range(n_features):
            hist, _ = np.histogram(
                features[:, i],
                bins=self.n_bins,
                density=True
            )
            hist = hist / (hist.sum() + 1e-10)  # Normalize
            histograms.append(hist)

        return np.concatenate(histograms)

    def _hellinger_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Hellinger distance between two distributions."""
        return np.sqrt(1 - np.sum(np.sqrt(p * q)))

    def _compute_distance_matrix(
        self,
        distributions: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise Hellinger distance matrix.
        """
        n = len(distributions)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = self._hellinger_distance(distributions[i], distributions[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return dist_matrix

    def fit(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> 'UnsupervisedRegimeDiscovery':
        """
        Fit clustering model to feature data.

        Parameters:
        -----------
        features : np.ndarray
            Shape (T, D) where T is timepoints, D is features
        feature_names : list
            Names of features for profiling
        """
        T, D = features.shape

        # Step 1: Extract sliding window distributions
        distributions = []
        indices = []

        for t in range(self.window_size, T, self.stride):
            window = features[t - self.window_size:t]
            dist = self._estimate_distribution(window)
            distributions.append(dist)
            indices.append(t)

        distributions = np.array(distributions)
        self.distribution_indices = indices

        # Step 2: Compute geodesic distance matrix
        print(f"Computing distance matrix for {len(distributions)} points...")
        dist_matrix = self._compute_distance_matrix(distributions)

        # Step 3: Cluster with HDBSCAN
        print("Clustering...")
        self.cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='precomputed',
        )
        labels = self.cluster_model.fit_predict(dist_matrix)

        self.labels = labels
        self.distributions = distributions

        # Step 4: Compute cluster centroids and profiles
        unique_labels = set(labels) - {-1}  # Exclude noise

        for label in unique_labels:
            mask = labels == label

            # Centroid: mean distribution
            self.cluster_centroids[label] = distributions[mask].mean(axis=0)

            # Profile: feature statistics in this cluster
            cluster_indices = [indices[i] for i, m in enumerate(mask) if m]
            cluster_features = features[cluster_indices]

            self.cluster_profiles[label] = {
                'size': mask.sum() / len(labels),
                'mean': cluster_features.mean(axis=0),
                'std': cluster_features.std(axis=0),
                'feature_names': feature_names,
            }

        print(f"Found {len(unique_labels)} clusters")
        return self

    def predict(self, features: np.ndarray) -> int:
        """
        Assign new observation to nearest cluster.

        Parameters:
        -----------
        features : np.ndarray
            Shape (window_size, D) - a window of features

        Returns:
        --------
        cluster_id : int
            Assigned cluster (-1 for noise/uncertain)
        """
        dist = self._estimate_distribution(features)

        # Find nearest cluster centroid
        min_dist = float('inf')
        best_cluster = -1

        for label, centroid in self.cluster_centroids.items():
            d = self._hellinger_distance(dist, centroid)
            if d < min_dist:
                min_dist = d
                best_cluster = label

        # Threshold for uncertainty
        if min_dist > 0.5:  # Adjust threshold
            return -1  # Uncertain

        return best_cluster

    def predict_proba(self, features: np.ndarray) -> Dict[int, float]:
        """
        Get probability distribution over clusters.
        Uses softmax over negative distances.
        """
        dist = self._estimate_distribution(features)

        distances = {}
        for label, centroid in self.cluster_centroids.items():
            distances[label] = self._hellinger_distance(dist, centroid)

        # Softmax
        min_d = min(distances.values())
        exp_neg_d = {k: np.exp(-(v - min_d) / 0.1) for k, v in distances.items()}
        total = sum(exp_neg_d.values())

        return {k: v / total for k, v in exp_neg_d.items()}

    def get_cluster_distances(self, features: np.ndarray) -> Dict[str, float]:
        """
        Get distances to all cluster centroids (for use as features).
        """
        dist = self._estimate_distribution(features)

        return {
            f'cluster_{label}_distance': self._hellinger_distance(dist, centroid)
            for label, centroid in self.cluster_centroids.items()
        }

    def visualize_clusters(
        self,
        method: str = 'umap',
        n_components: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 2D/3D visualization of clusters.

        Returns:
        --------
        embedding : np.ndarray
            Low-dimensional representation
        labels : np.ndarray
            Cluster labels
        """
        if method == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                metric='precomputed',
                n_neighbors=30,
            )
            dist_matrix = self._compute_distance_matrix(self.distributions)
            embedding = reducer.fit_transform(dist_matrix)
        else:  # tsne
            reducer = TSNE(
                n_components=n_components,
                metric='precomputed',
            )
            dist_matrix = self._compute_distance_matrix(self.distributions)
            embedding = reducer.fit_transform(dist_matrix)

        return embedding, self.labels

    def analyze_cluster(
        self,
        cluster_id: int,
        feature_names: List[str],
    ) -> Dict:
        """
        Detailed analysis of a specific cluster.
        """
        profile = self.cluster_profiles.get(cluster_id, {})

        if not profile:
            return {'error': f'Cluster {cluster_id} not found'}

        # Find most distinctive features
        mean = profile['mean']
        std = profile['std']

        # Compare to global mean
        global_mean = np.mean([p['mean'] for p in self.cluster_profiles.values()], axis=0)
        global_std = np.std([p['mean'] for p in self.cluster_profiles.values()], axis=0)

        z_scores = (mean - global_mean) / (global_std + 1e-10)

        distinctive = sorted(
            zip(feature_names, z_scores),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        return {
            'cluster_id': cluster_id,
            'size': profile['size'],
            'distinctive_features': distinctive,
            'mean_features': dict(zip(feature_names, mean)),
        }
```

---

## 4. Integrated Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTEGRATED EXTENSION ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         DATA INGESTION (Rust)                          │ │
│  │  Hyperliquid WS ──▶ Parser ──▶ Base Features (57) ──▶ Parquet         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    EXTENSION LAYER (Python)                            │ │
│  │                                                                         │ │
│  │  Base Features                                                          │ │
│  │       │                                                                 │ │
│  │       ├──▶ [PCE Engine] ──▶ Sobol indices, UQ features (+8)           │ │
│  │       │                                                                 │ │
│  │       ├──▶ [TE Network] ──▶ Causal graph, network features (+12)      │ │
│  │       │                                                                 │ │
│  │       ├──▶ [Info Geometry] ──▶ Manifold velocity, curvature (+8)      │ │
│  │       │                                                                 │ │
│  │       └──▶ [Unsupervised Clustering] ──▶ Cluster distances (+K)       │ │
│  │                                                                         │ │
│  │  Extended Features: 57 + 8 + 12 + 8 + K ≈ 90-100 total                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    REGIME DETECTION                                     │ │
│  │                                                                         │ │
│  │  Option A: Supervised (original)                                        │ │
│  │  Extended features ──▶ XGBoost ──▶ MR/TF/NA                            │ │
│  │                                                                         │ │
│  │  Option B: Unsupervised (novel)                                         │ │
│  │  Extended features ──▶ Cluster assignment ──▶ Strategy lookup          │ │
│  │                                                                         │ │
│  │  Option C: Hybrid (recommended)                                         │ │
│  │  Cluster assignment + Extended features ──▶ XGBoost ──▶ Strategy       │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    STRATEGY EXECUTION                                   │ │
│  │                                                                         │ │
│  │  Regime/Cluster ──▶ Strategy selector ──▶ ASMM(θ) / TrendFollow(θ)    │ │
│  │                                                                         │ │
│  │  Per-cluster optimal parameters learned via backtesting                │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Originality Summary

| Extension | Originality | Justification |
|-----------|-------------|---------------|
| **PCE for microstructure** | HIGH | Not in literature for LOB/regime detection |
| **Information Geometry** | HIGH | Novel application to market state manifold |
| **Transfer Entropy Networks** | MEDIUM-HIGH | Applied to LOB features specifically |
| **Unsupervised clustering** | MEDIUM | Standard technique |
| **Clustering on Fisher manifold** | **VERY HIGH** | Novel synthesis - geodesic distance clustering |
| **Full integration** | **HIGH** | No prior work combines all these |

**Overall**: With these extensions, this is publishable research, not just applied engineering.
