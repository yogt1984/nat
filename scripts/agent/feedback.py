"""
OOS Feedback Summarizer — closes the learning loop.

Reads OOS validation results, registry IC trends, and failure patterns
to produce a compact natural-language summary for the LLM generator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("nat.feedback")

ROOT = Path(__file__).resolve().parent.parent.parent
OOS_STATE_FILE = ROOT / "data" / "oos_validation" / "state.json"


def build_feedback_summary(store, agent: str) -> str:
    """Build a compact feedback summary for LLM context (< 2000 tokens)."""
    parts = []

    # 1. Feature performance (which features appear in winners vs losers)
    feat_perf = _feature_performance(store, agent)
    if feat_perf:
        top_winners = sorted(feat_perf.items(), key=lambda x: -x[1]["win_rate"])[:5]
        top_losers = sorted(feat_perf.items(), key=lambda x: x[1]["win_rate"])[:5]
        parts.append("FEATURE WIN RATES:")
        for feat, stats in top_winners:
            parts.append(f"  {feat}: {stats['win_rate']:.0%} ({stats['wins']}/{stats['total']})")
        if top_losers and top_losers[0][1]["win_rate"] < 0.05:
            parts.append("FEATURES THAT NEVER WORK:")
            for feat, stats in top_losers[:3]:
                if stats["win_rate"] < 0.05:
                    parts.append(f"  {feat}: {stats['win_rate']:.0%} ({stats['total']} attempts)")

    # 2. Failure patterns
    failures = _failure_patterns(store, agent)
    if failures:
        parts.append("\nFAILURE PATTERNS:")
        for (gen, reason), count in sorted(failures.items(), key=lambda x: -x[1])[:5]:
            parts.append(f"  {gen} -> {reason}: {count} times")

    # 3. OOS validation summary
    oos_summary = _oos_summary()
    if oos_summary:
        parts.append(f"\nOOS VALIDATION:\n{oos_summary}")

    # 4. IC trend
    trend = _ic_trend(store, agent)
    if trend:
        parts.append(f"\nREGISTRY IC TREND: {trend}")

    return "\n".join(parts) if parts else "No feedback data available yet."


def _feature_performance(store, agent: str) -> dict:
    """Count how many times each feature appears in winners vs losers."""
    feat_stats = {}

    try:
        hypotheses = store.load_hypotheses(agent)
    except Exception:
        return {}

    for h in hypotheses:
        status = h.get("status", "")
        results = h.get("results", {})

        # Extract features from the hypothesis
        features = []
        if isinstance(results, dict):
            features = results.get("features", [])
        claim = h.get("claim", "")

        # Infer feature from claim if not in results
        if not features and "feature=" in claim:
            try:
                feat = claim.split("feature=")[1].split(",")[0].split(")")[0].strip()
                features = [feat]
            except Exception:
                pass

        is_win = status == "registered"
        for feat in features:
            if feat not in feat_stats:
                feat_stats[feat] = {"wins": 0, "total": 0, "win_rate": 0.0}
            feat_stats[feat]["total"] += 1
            if is_win:
                feat_stats[feat]["wins"] += 1
            feat_stats[feat]["win_rate"] = (
                feat_stats[feat]["wins"] / feat_stats[feat]["total"]
            )

    return feat_stats


def _failure_patterns(store, agent: str) -> dict:
    """Group failures by (generator, failure_reason)."""
    patterns = {}

    try:
        hypotheses = store.load_hypotheses(agent)
    except Exception:
        return {}

    for h in hypotheses:
        if h.get("status") != "failed":
            continue
        gen = h.get("generator", "unknown")
        reason = h.get("failure_reason", "unknown")
        key = (gen, reason)
        patterns[key] = patterns.get(key, 0) + 1

    return patterns


def _oos_summary() -> str:
    """Summarize OOS validation state."""
    if not OOS_STATE_FILE.exists():
        return ""

    try:
        with open(OOS_STATE_FILE) as f:
            state = json.load(f)
    except Exception:
        return ""

    lines = []
    algos = state.get("algos", {})
    for algo_name in ["3f", "jump_detector", "funding_reversion", "optimal_entry"]:
        algo = algos.get(algo_name, {})
        symbols = algo.get("symbols", {})
        for sym in ["BTC", "ETH", "SOL"]:
            metrics = symbols.get(sym, {}).get("metrics", {})
            sharpe = metrics.get("current_sharpe", 0)
            pnl = metrics.get("cumulative_pnl_bps", 0)
            degrad = metrics.get("degradation", False)
            if sharpe != 0 or pnl != 0:
                flag = " [DEGRADING]" if degrad else ""
                lines.append(f"  {algo_name} {sym}: Sharpe={sharpe:+.1f}, PnL={pnl:+.0f} bps{flag}")

    return "\n".join(lines)


def _ic_trend(store, agent: str) -> str:
    """Compute overall IC trend from registry."""
    try:
        registry = store.load_registry(agent)
    except Exception:
        return ""

    if not registry:
        return "empty registry"

    # Check IC history trends
    improving = 0
    degrading = 0
    for sig in registry:
        ic_hist = sig.get("ic_history", [])
        if len(ic_hist) >= 3:
            recent = ic_hist[-3:]
            if recent[-1] > recent[0]:
                improving += 1
            else:
                degrading += 1

    if improving > degrading:
        return f"improving ({improving} signals gaining, {degrading} losing)"
    elif degrading > improving:
        return f"degrading ({degrading} signals losing, {improving} gaining)"
    return f"stable ({len(registry)} signals)"
