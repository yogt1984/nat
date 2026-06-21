"""`nat metrics` — the metric catalogue (IC / IT / stats), extracted from the monolith.

First domain split onto the register(sub) protocol: it owns its handlers + the
METRICS data + its parser registration, and exposes `register(sub)` that the
app's build_parser calls at the same ordinal position as the old inline block.
"""

from __future__ import annotations

import sys

from cli.common import ROOT, BOLD, W, R, _output, _json_mode, _p

METRICS = [
    {"name": "ic", "category": "core", "estimator": None,
     "formula": "IC = Spearman ρ(signal_t, fwd_return[t→t+k])",
     "definition": "Rank correlation of a signal with forward return (predictive power).",
     "used_by": "signal test, profile scalp, algorithm evaluate, mf/macro-agent"},
    {"name": "dic", "category": "core", "estimator": None,
     "formula": "dIC = IC_gated − IC_ungated",
     "definition": "IC improvement from a regime gate (conditioning lift).",
     "used_by": "agent discovery gate, regime conditioning"},
    {"name": "sharpe", "category": "core", "estimator": None,
     "formula": "Sharpe = mean(r) / std(r) · √(periods_per_year)",
     "definition": "Risk-adjusted return of a strategy's P&L series.",
     "used_by": "backtest, oos, swarm fitness, evolve"},
    {"name": "deflated_sharpe", "category": "core", "estimator": None,
     "formula": "DSR = Φ( (SR − SR0)·√(n−1) / √(1 − γ3·SR + (γ4−1)/4·SR²) )",
     "definition": "Sharpe deflated for multiple-testing / non-normality (Bailey & López de Prado 2014).",
     "used_by": "backtest gate G4, oos --window, evolve guard rails"},
    {"name": "ksg_mi", "category": "information-theory", "estimator": "ksg_mi",
     "formula": "I(X;Y) via KSG Algorithm 1 (k-NN, Chebyshev ε-ball), bits",
     "definition": "Mutual information — total (linear+nonlinear) dependence between X and Y.",
     "used_by": "process mi_ksg, it-engine greedy selection"},
    {"name": "cmi", "category": "information-theory", "estimator": "cmi",
     "formula": "I(X;Y|Z) = H(X,Z)+H(Y,Z)−H(X,Y,Z)−H(Z)",
     "definition": "Conditional MI — X↔Y dependence after removing what Z explains.",
     "used_by": "it-engine feature_selector (greedy CMI-gain), interaction info"},
    {"name": "interaction_info", "category": "information-theory", "estimator": "interaction_info",
     "formula": "II(X;Y;Z) = I(X;Y|Z) − I(X;Y)",
     "definition": "Synergy (>0) vs redundancy (<0) among three variables.",
     "used_by": "it-engine, feature-interaction discovery"},
    {"name": "transfer_entropy", "category": "information-theory", "estimator": "ksg_te",
     "formula": "TE(X→Y) = I(X_past; Y_present | Y_past)",
     "definition": "Directed information flow from source to target (lead-lag causality).",
     "used_by": "process transfer_entropy, it-engine daemon"},
    {"name": "linear_te", "category": "information-theory", "estimator": "linear_te",
     "formula": "TE = ½·log( var(ε_reduced) / var(ε_full) )",
     "definition": "Gaussian/AR approximation of transfer entropy (fast linear proxy).",
     "used_by": "it-engine (cheap TE pre-screen)"},
    {"name": "min_info_bits", "category": "information-theory", "estimator": "min_info_bits",
     "formula": "I_min = −½·log₂(1 − (fee_RT/σ_r)²) · (κ/3)",
     "definition": "Cost gate: minimum MI (bits) a signal needs to beat transaction costs.",
     "used_by": "it-engine greedy stop criterion, cost gate"},
]


def _estimator_doc(estimator_name):
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from it_engine import estimators as _est
        fn = getattr(_est, estimator_name, None)
        return (fn.__doc__ or "").strip() if fn else None
    except Exception:
        return None


def cmd_metrics_list(args):
    """List the metrics catalogue (name, category, definition, where-used)."""
    cat = getattr(args, 'category', None)
    rows = [m for m in METRICS if not cat or m["category"] == cat]

    def _human(_d):
        items = _d["metrics"]
        print(f"\n  {BOLD}Metrics catalogue{W}  ({len(items)})\n")
        print(f"  {'NAME':<18} {'CATEGORY':<20} DEFINITION")
        for m in items:
            print(f"  {m['name']:<18} {m['category']:<20} {m['definition']}")
        print(f"\n  Detail: nat metrics show <name>   ·   full glossary: nat --math\n")

    _output({"metrics": rows, "count": len(rows)}, args, _human)
    return 0


def cmd_metrics_show(args):
    """Show one metric: formula, definition, where-used, + live estimator docstring."""
    name = getattr(args, 'name')
    m = next((x for x in METRICS if x["name"] == name), None)
    if not m:
        if _json_mode(args):
            _output({"error": f"unknown metric '{name}'",
                     "known": [x["name"] for x in METRICS]}, args)
        else:
            _p("x", R, f"Unknown metric '{name}'. Try: nat metrics ls")
        return 1
    doc = _estimator_doc(m["estimator"]) if m["estimator"] else None
    payload = {**m, "estimator_doc": doc}

    def _human(_m):
        print(f"\n  {BOLD}{_m['name']}{W}  ({_m['category']})\n")
        print(f"  Definition: {_m['definition']}")
        print(f"  Formula:    {_m['formula']}")
        print(f"  Used by:    {_m['used_by']}")
        if _m.get("estimator"):
            print(f"  Estimator:  scripts/it_engine/estimators.py:{_m['estimator']}()")
        if _m.get("estimator_doc"):
            print(f"\n  {BOLD}Implementation notes{W}\n")
            for line in _m["estimator_doc"].splitlines():
                print(f"    {line}")
        print()

    _output(payload, args, _human)
    return 0


def register(sub):
    """Register the `metrics` group on the given subparsers action (verbatim)."""
    # ── metrics (catalogue of IC / IT / stats metrics) ──
    met_p = sub.add_parser('metrics', help='Metric catalogue (IC, MI, TE, Sharpe, …)')
    met_p.set_defaults(func=lambda a: met_p.print_help())
    metsub = met_p.add_subparsers(dest='subcmd')
    metls = metsub.add_parser('ls', aliases=['list'], help='List all metrics (name/category/definition)')
    metls.add_argument('--category', default=None,
                       help='Filter: core | information-theory')
    metls.add_argument('--json', action='store_true', help='JSON output')
    metls.set_defaults(func=cmd_metrics_list)
    metshow = metsub.add_parser('show', help='Show one metric (formula + estimator docstring)')
    metshow.add_argument('name', help='Metric name (see nat metrics ls)')
    metshow.add_argument('--json', action='store_true', help='JSON output')
    metshow.set_defaults(func=cmd_metrics_show)


__all__ = ["cmd_metrics_list", "cmd_metrics_show", "register"]
