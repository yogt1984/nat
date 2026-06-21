"""`nat visualize` — visualization suite (scanner/data/profiling/hierarchy/cluster)."""

from __future__ import annotations

import sys

from cli.common import ROOT, DATA_DEFAULT, G, B, _banner, _p, _py


def cmd_visualize_scan(args):
    """Generate scanner visualization plots."""
    _banner("Scanner Visualizations")
    symbol = getattr(args, 'symbol', 'BTC')
    data_dir = getattr(args, 'data', str(DATA_DEFAULT))
    report_path = getattr(args, 'report', None)
    output = getattr(args, 'output', str(ROOT / 'reports' / 'figures' / 'scanner'))
    dpi = getattr(args, 'dpi', 150)

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "scripts" / "exploration"))  # visualize_scanner lives here
    from visualize_scanner import run_all
    paths = run_all(data_dir=data_dir, symbol=symbol, output_dir=output,
                    report_path=report_path, dpi=dpi, plots="scanner")
    _p("+", G, f"Saved {len(paths)} figures to {output}/")


def cmd_visualize_data(args):
    """Generate data quality plots."""
    _banner("Data Quality Visualizations")
    symbol = getattr(args, 'symbol', 'BTC')
    data_dir = getattr(args, 'data', str(DATA_DEFAULT))
    output = getattr(args, 'output', str(ROOT / 'reports' / 'figures' / 'scanner'))
    dpi = getattr(args, 'dpi', 150)

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "scripts" / "exploration"))  # visualize_scanner lives here
    from visualize_scanner import run_all
    paths = run_all(data_dir=data_dir, symbol=symbol, output_dir=output,
                    dpi=dpi, plots="data")
    _p("+", G, f"Saved {len(paths)} figures to {output}/")


def cmd_visualize_profile(args):
    """Run existing cluster profiling visualizations."""
    _banner("Cluster Profiling Visualizations")
    _py(f"{ROOT / 'scripts' / 'visualize_profiling.py'}")


def cmd_visualize_hierarchy(args):
    """Generate hierarchical profiling visualizations (Phase 8)."""
    _banner("Hierarchical Profiling Visualizations")
    data_dir = getattr(args, 'data', str(DATA_DEFAULT))
    output = getattr(args, 'output', str(ROOT / 'reports' / 'figures'))
    vector = getattr(args, 'vector', 'entropy')
    timeframe = getattr(args, 'timeframe', '15min')
    _py(f"{ROOT / 'scripts' / 'visualize_hierarchy.py'} --data {data_dir} --output {output} --vector {vector} --timeframe {timeframe}")


def cmd_visualize_cluster(args):
    """Generate cluster exploration plots (PCA, UMAP, t-SNE scatter with clustering)."""
    _banner("Cluster Exploration Visualizations")
    data_dir = getattr(args, 'data', str(DATA_DEFAULT))
    output = getattr(args, 'output', str(ROOT / 'reports' / 'figures' / 'clusters'))
    symbol = getattr(args, 'symbol', 'BTC')
    subset = getattr(args, 'subset', None)
    method = getattr(args, 'cluster_method', 'gmm')
    cmd = f"{ROOT / 'scripts' / 'explore_clusters.py'} --data-dir {data_dir} --output-dir {output} --cluster-method {method}"
    if symbol:
        cmd += f" --symbol {symbol}"
    if subset:
        cmd += f" --subset {subset}"
    _py(cmd)


def cmd_visualize_skeptical(args):
    """Generate skeptical validation diagnostic plots (20 tests, 16+ PNGs)."""
    _banner("Skeptical Validation Visualizations")
    data_dir = getattr(args, 'data', str(DATA_DEFAULT))
    output = getattr(args, 'output', str(ROOT / 'reports' / 'skeptical_validation'))
    _py(f"{ROOT / 'scripts' / 'skeptical_validation.py'} --data {data_dir} --output {output}")


def cmd_visualize_all(args):
    """Generate all visualizations."""
    _banner("All Visualizations")
    symbol = getattr(args, 'symbol', 'BTC')
    data_dir = getattr(args, 'data', str(DATA_DEFAULT))
    report_path = getattr(args, 'report', None)
    output = getattr(args, 'output', str(ROOT / 'reports' / 'figures' / 'scanner'))
    dpi = getattr(args, 'dpi', 150)

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "scripts" / "exploration"))  # visualize_scanner lives here
    from visualize_scanner import run_all
    paths = run_all(data_dir=data_dir, symbol=symbol, output_dir=output,
                    report_path=report_path, dpi=dpi, plots="all")
    _p("+", G, f"Saved {len(paths)} scanner/data figures to {output}/")
    print()
    _p("...", B, "Running cluster profiling visualizations...")
    _py(f"{ROOT / 'scripts' / 'visualize_profiling.py'}")


def register(sub):
    # ── visualize ──
    viz_p = sub.add_parser('visualize', help='Visualization suite')
    viz_p.add_argument('--symbol', default='BTC', help='Trading symbol (default: BTC)')
    viz_p.add_argument('--data', default=str(DATA_DEFAULT), help='Feature data directory')
    viz_p.add_argument('--report', default=None, help='Scanner JSON report path')
    viz_p.add_argument('--output', default=str(ROOT / 'reports' / 'figures' / 'scanner'), help='Output directory for plots')
    viz_p.add_argument('--dpi', type=int, default=150, help='Plot DPI resolution')
    viz_p.set_defaults(func=cmd_visualize_all)
    vsub = viz_p.add_subparsers(dest='subcmd')
    vsub.add_parser('scan', help='Scanner plots (1-7)').set_defaults(func=cmd_visualize_scan)
    vsub.add_parser('data', help='Data quality plots (8-10)').set_defaults(func=cmd_visualize_data)
    vsub.add_parser('profile', help='Cluster profiling plots').set_defaults(func=cmd_visualize_profile)
    hier_p = vsub.add_parser('hierarchy', help='Hierarchical profiling plots (Phase 8)')
    hier_p.add_argument('--vector', default='entropy', help='Feature vector (entropy, orderflow, etc.)')
    hier_p.add_argument('--timeframe', default='15min', help='Bar aggregation timeframe')
    hier_p.set_defaults(func=cmd_visualize_hierarchy)
    cluster_p = vsub.add_parser('cluster', help='Cluster exploration plots (PCA/UMAP/t-SNE)')
    cluster_p.add_argument('--subset', default=None, help='Feature subset (entropy, orderflow, etc.)')
    cluster_p.add_argument('--cluster-method', default='gmm', choices=['gmm', 'kmeans', 'hdbscan', 'dbscan'], help='Clustering algorithm')
    cluster_p.set_defaults(func=cmd_visualize_cluster)
    vsub.add_parser('skeptical', help='Skeptical validation diagnostic plots').set_defaults(func=cmd_visualize_skeptical)
    vsub.add_parser('all', help='All visualizations').set_defaults(func=cmd_visualize_all)


__all__ = [
    "cmd_visualize_scan", "cmd_visualize_data", "cmd_visualize_profile",
    "cmd_visualize_hierarchy", "cmd_visualize_cluster", "cmd_visualize_skeptical",
    "cmd_visualize_all", "register",
]
