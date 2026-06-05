# Specification: Nearest-Neighbor State Retrieval (#6)

Non-parametric, buffer-based algorithm that finds historical market states most similar to the current state using Mahalanobis distance. Predicts forward returns from time-decay weighted neighbor outcomes. No offline training required — adapts continuously as the buffer grows.

---

## Thesis

Market microstructure exhibits recurrent patterns. When the current state resembles a past state that led to a profitable outcome, the same trade is likely profitable now. Unlike parametric models, KNN makes no distributional assumptions and adapts to non-stationarity through its sliding buffer.

---

## Mahalanobis Distance

Standard Euclidean distance treats all features equally, ignoring correlations. Mahalanobis distance whitens features via the inverse covariance matrix:

```
d_M(x, y) = sqrt((x - y)^T * Sigma^{-1} * (x - y))
```

Implemented via Cholesky decomposition: `Sigma = L * L^T`, then `d_M(x, y) = ||L^{-1}(x - y)||_2`. This reduces to Euclidean distance in the whitened space, enabling efficient KD-tree lookup.

---

## Ledoit-Wolf Shrinkage

The sample covariance is ill-conditioned with few samples relative to features. Ledoit-Wolf shrinkage regularizes toward the identity:

```
Sigma_shrunk = (1 - alpha) * S + alpha * mu * I
```

where `mu = trace(S) / p` and `alpha` is estimated from the data. This ensures positive-definiteness for Cholesky.

---

## KD-Tree Rebuild Protocol

The KD-tree and covariance are rebuilt every `refit_interval` bars (default 100):

1. Extract all feature vectors from the ring buffer
2. Compute Ledoit-Wolf shrinkage covariance
3. Cholesky decomposition -> whitening matrix
4. Whiten all buffer vectors
5. Build `scipy.spatial.cKDTree` on whitened vectors

Between rebuilds, queries use the existing tree (stale by at most `refit_interval` bars).

---

## Time-Decay Weighting

Neighbors are weighted by recency:

```
w_i = exp(-ln(2) * age_i / halflife)
```

where `age_i = buffer_len - 1 - index_i`. Default halflife=500 bars (~42 hours at 5-min bars). This ensures the prediction is dominated by recent market behavior.

---

## Parameters

| Parameter | Default | Config Key | Description |
|-----------|---------|------------|-------------|
| `k` | 20 | `[knn_retrieval].k` | Number of nearest neighbors |
| `buffer_size` | 5000 | `[knn_retrieval].buffer_size` | Ring buffer capacity |
| `time_decay_halflife` | 500 | `[knn_retrieval].time_decay_halflife` | Decay halflife (bars) |
| `refit_interval` | 100 | `[knn_retrieval].refit_interval` | Bars between tree rebuilds |
| `cost_threshold_bps` | 2.0 | `[knn_retrieval].cost_threshold_bps` | Min expected return for signal |
| `win_rate_threshold` | 0.60 | `[knn_retrieval].win_rate_threshold` | Min win rate for signal |

---

## Output Features

| Name | Range | Warmup | Description |
|------|-------|--------|-------------|
| `alg_knn_signal` | [-1, 1] | 0 | Directional signal (0 during buffer fill) |
| `alg_knn_expected_return` | (-inf, inf) | 0 | Weighted mean forward return of K neighbors |
| `alg_knn_win_rate` | [0, 1] | 0 | Fraction of profitable neighbors |
| `alg_knn_confidence` | [0, 1] | 0 | Inverse distance confidence |

---

## References

- Cover, T.M. & Hart, P.E. (1967). "Nearest Neighbor Pattern Classification." *IEEE Transactions on Information Theory*, 13(1), 21-27.
- Mahalanobis, P.C. (1936). "On the Generalized Distance in Statistics." *Proceedings of the National Institute of Sciences of India*, 2(1), 49-55.
- Ledoit, O. & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*, 88(2), 365-411.

---

## File Locations

| Purpose | Path |
|---------|------|
| Algorithm | `scripts/algorithms/knn_retrieval.py` |
| Config | `config/algorithms.toml` -> `[knn_retrieval]` |
| Paper trader | `scripts/alpha/paper_trader_generic.py` -> `ALGO_CONFIG` |
| Daily runner | `scripts/alpha/paper_trader_daily.py` -> `DAILY_ALGOS` |
| Unit tests | `scripts/algorithms/tests/test_knn_unit.py` |
| Integration tests | `scripts/algorithms/tests/test_knn_integration.py` |
