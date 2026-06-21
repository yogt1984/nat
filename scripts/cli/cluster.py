"""`nat cluster` — cluster analysis (GMM/HMM, exploration, predictive quality)."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from cli.common import ROOT, DATA_DEFAULT, BOLD, W, R, G, _banner, _py, _p, _sym, _data


def cmd_cluster_analyze(args):
    hours = getattr(args, 'hours', 24)
    _banner(f"Analyzing cluster quality: {_sym(args)}")
    _py(f"scripts/analyze_clusters.py --data-dir {_data(args)} --symbol {_sym(args)} --hours {hours}")


def cmd_cluster_gmm(args):
    _banner(f"Analyzing with GMM model: {_sym(args)}")
    _py(
        f"scripts/analyze_clusters.py --data-dir {_data(args)} --symbol {_sym(args)} "
        f"--model models/regime_gmm.json"
    )


def cmd_cluster_all(args):
    _banner("Analyzing all symbols")
    for sym in ["BTC", "ETH", "SOL"]:
        print(f"\n  {BOLD}{sym}{W}\n")
        _py(f"scripts/analyze_clusters.py --data-dir {_data(args)} --symbol {sym} "
            f"--output reports/cluster_{sym}.txt")


def cmd_cluster_quality(args):
    _banner("Q3 predictive quality test")
    _py("scripts/q3_predictive_quality.py")


def cmd_cluster_explore(args):
    _banner(f"Exploring clusters: {_sym(args)}")
    cmd = f"scripts/explore_clusters.py --data-dir {_data(args)} --symbol {_sym(args)}"
    subset = getattr(args, 'subset', None)
    if subset:
        cmd += f" --subset {subset}"
    method = getattr(args, 'method', 'gmm')
    cmd += f" --cluster-method {method}"
    _py(cmd)


def cmd_cluster_hmm(args):
    _banner(f"HMM fitting: {_sym(args)}")
    n_states = getattr(args, 'n_states', 3)
    n_iter = getattr(args, 'n_iter', 100)
    cov = getattr(args, 'covariance', 'full')
    timeframe = getattr(args, 'timeframe', '15min')
    output = getattr(args, 'output', 'reports/hmm_fit.json')

    sys.path.insert(0, str(ROOT / "scripts"))
    from cluster_pipeline.loader import load_parquet
    from cluster_pipeline.preprocess import aggregate_bars
    from cluster_pipeline.transitions import fit_hmm, compare_hmm_gmm, empirical_transitions
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    import json, numpy as np, polars as pl

    print(f"  Loading data from {_data(args)}...")
    df = load_parquet(str(_data(args)))
    df = df.filter(pl.col("symbol") == _sym(args))
    df = aggregate_bars(df, timeframe=timeframe)
    print(f"  {len(df)} bars after {timeframe} aggregation")

    # Select numeric feature columns
    meta = {"bar_start", "bar_end", "symbol", "tick_count", "bar_index"}
    feat_cols = [c for c in df.columns if c not in meta and df[c].dtype in (pl.Float64, pl.Float32)]
    X = df.select(feat_cols).to_numpy()
    # Fill NaN with column median
    for i in range(X.shape[1]):
        col = X[:, i]
        mask = ~np.isfinite(col)
        if mask.any():
            X[mask, i] = np.nanmedian(col)

    # PCA reduce
    n_components = min(10, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"  PCA: {X.shape[1]} features → {n_components} components "
          f"({pca.explained_variance_ratio_.sum():.1%} variance)")

    # GMM baseline
    gmm = GaussianMixture(n_components=n_states, random_state=42, n_init=3).fit(X_pca)
    gmm_labels = gmm.predict(X_pca)
    emp = empirical_transitions(gmm_labels)

    # HMM fit with GMM initialization
    result = fit_hmm(
        X_pca, n_states=n_states, n_iter=n_iter,
        covariance_type=cov,
        init_transmat=emp.matrix, init_means=gmm.means_,
    )
    if result is None:
        _p("x", R, "HMM fitting failed or insufficient data")
        return

    # Compare
    comparison = compare_hmm_gmm(result.smoothed_labels, gmm_labels)

    _g = lambda p: f"\033[32mPASS\033[0m" if p else f"\033[31mFAIL\033[0m"
    print(f"\n  HMM Results ({n_states} states):")
    print(f"    Converged:           {result.convergence}")
    print(f"    Log-likelihood:      {result.log_likelihood:.4f}")
    print(f"    BIC:                 {result.bic:.1f}")
    print(f"    ARI vs GMM:          {comparison['ari']:.4f}  [{_g(comparison['ari'] > 0.5)}]")
    print(f"    Smoothness ratio:    {comparison['smoothness_ratio']:.3f} "
          f"(HMM={comparison['hmm_transitions']}, GMM={comparison['gmm_transitions']})")
    print(f"    Stationary dist:     {np.array2string(result.stationary_distribution, precision=3)}")

    # Diagonal dominance check
    diag_ok = all(result.transition_matrix[i, i] > 0.5 for i in range(n_states))
    print(f"    Diagonal dominant:   [{_g(diag_ok)}]")

    # Save
    out = {
        "symbol": _sym(args), "n_states": n_states, "timeframe": timeframe,
        "n_bars": len(df), "converged": result.convergence,
        "log_likelihood": result.log_likelihood, "bic": result.bic,
        "transition_matrix": result.transition_matrix.tolist(),
        "stationary_distribution": result.stationary_distribution.tolist(),
        "comparison": comparison,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    _p("+", G, f"Saved to {output}")


def register(sub):
    cl_p = sub.add_parser('cluster', help='Cluster analysis')
    cl_p.set_defaults(func=lambda a: cl_p.print_help())
    csub = cl_p.add_subparsers(dest='subcmd')
    ca = csub.add_parser('analyze', help='Analyze cluster quality')
    ca.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    ca.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    ca.add_argument('--hours', type=int, default=24, help='Hours of data to analyze')
    ca.set_defaults(func=cmd_cluster_analyze)
    cg = csub.add_parser('gmm', help='Analyze with GMM')
    cg.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    cg.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    cg.set_defaults(func=cmd_cluster_gmm)
    call = csub.add_parser('all', help='Analyze all symbols')
    call.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    call.set_defaults(func=cmd_cluster_all)
    csub.add_parser('quality', help='Q3 predictive quality test').set_defaults(func=cmd_cluster_quality)
    ce = csub.add_parser('explore', help='Exploratory clustering (PCA/UMAP/t-SNE)')
    ce.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    ce.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    ce.add_argument('--subset', default=None, help='Feature subset (entropy, orderflow, etc.)')
    ce.add_argument('--method', default='gmm', choices=['gmm', 'kmeans', 'hdbscan', 'dbscan'], help='Clustering algorithm')
    ce.set_defaults(func=cmd_cluster_explore)
    ch = csub.add_parser('hmm-fit', help='Fit HMM on feature data (Baum-Welch)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Fits a Hidden Markov Model to PCA-reduced feature data using Baum-Welch EM,
        initialized from GMM cluster assignments.

        Mathematics:
          E-step (forward-backward):
            α_t(i) = P(o_1..o_t, s_t=i)     forward probabilities
            β_t(i) = P(o_{t+1}..o_T | s_t=i) backward probabilities
            γ_t(i) = P(s_t=i | O, λ)         state posteriors
          M-step:
            A_ij = Σ ξ_t(i,j) / Σ γ_t(i)    transition matrix update
            μ_i = Σ γ_t(i)·o_t / Σ γ_t(i)   mean update
          Model selection:
            BIC = -2·ln(L) + k·ln(n)         Bayesian Information Criterion
          Comparison:
            ARI (Adjusted Rand Index) vs GMM labels
            Smoothness ratio: HMM transitions / GMM transitions

        Example:
          nat cluster hmm-fit --symbol BTC --n-states 3 --timeframe 15min
        """))
    ch.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    ch.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    ch.add_argument('--n-states', type=int, default=3, help='Number of hidden states')
    ch.add_argument('--n-iter', type=int, default=100, help='Max EM iterations')
    ch.add_argument('--covariance', default='full', choices=['full', 'diag', 'spherical', 'tied'], help='Covariance type')
    ch.add_argument('--timeframe', default='15min', help='Bar aggregation timeframe')
    ch.add_argument('--output', default='reports/hmm_fit.json', help='Output JSON path')
    ch.set_defaults(func=cmd_cluster_hmm)


__all__ = ["cmd_cluster_analyze", "cmd_cluster_gmm", "cmd_cluster_all",
           "cmd_cluster_quality", "cmd_cluster_explore", "cmd_cluster_hmm",
           "register"]
