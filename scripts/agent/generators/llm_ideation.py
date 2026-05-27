"""LLM Hypothesis Generator — Claude-powered ideation.

Calls Claude API with context from registry, graveyard, generator stats,
arXiv ideas, and OOS feedback to generate novel testable hypotheses.

Plugs into the standard generator framework: generate(manifest, queue, stats).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from ..hypothesis import Hypothesis, GeneratorStats
from ..hypothesis_queue import HypothesisQueue

log = logging.getLogger(__name__)

Hypothesis.register_prefix("llm_ideation", "LLM")

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Compact feature category summary (avoids sending 209 names)
FEATURE_CATEGORIES = """
15 categories, 209 total features:
- raw (8): midprice, spread_bps, bid/ask_depth_5, bid/ask_depth_10, trade_count_5s, volume_5s
- imbalance (6): qty L1/L5/L10, depth_weighted, notional, bid_lift_ratio
- flow (6): ofi, aggressor_ratio, vwap_deviation, net_volume, taker_imbalance, kyle_lambda
- vol (5): returns 1m/5m, ratio_short_long, garman_klass, parkinson
- entropy (7): book_shape, tick 5s/30s/1m, permutation_returns, spread_dispersion, surprise
- illiq (4): kyle_100, amihud_100, composite, roll_spread
- toxic (4): vpin_50, adverse_selection, index, pin
- derived (14): regime cluster, PCA, trend score, mean_revert score, interactions
- whale_flow (4): net_flow, pressure, accumulation, distribution
- liquidation (4): cluster_risk, cascade_prob, open_interest_delta, funding_rate
- concentration (3): herfindahl, top_trader_share, gini
- regime (3): type_score, confidence, persistence
- gmm (3): state, posterior, transition_prob
- cross_symbol (5): btc_eth_corr, lead_lag, beta, spillover, contagion
- algorithm outputs (130+): from 20 registered algorithms (entropy_momentum, hawkes, jump_detector, etc.)
"""

SYSTEM_PROMPT = """You are a quantitative researcher designing testable hypotheses for a crypto perpetual futures alpha discovery system.

The system tests hypotheses on BTC, ETH, SOL using 100ms microstructure features. Each hypothesis specifies a signal feature, optional regime gate, and test protocol.

{feature_categories}

RULES:
- Output exactly one JSON object (no markdown, no explanation)
- The claim must be a testable statement about a signal predicting short-term returns
- test_protocol must use real nat CLI commands: "spannung regime --data {{data_dir}} --symbol {{sym}}"
- Features in thresholds must exist in the manifest above
- Priority 0.0 to 1.0 (higher = more promising)
- Be creative but grounded — propose combinations not yet tried

JSON schema:
{{
  "claim": "human-readable hypothesis statement",
  "test_protocol": ["nat command 1", "nat command 2"],
  "thresholds": {{
    "min_ic": 0.10,
    "min_dIC": 0.05,
    "min_coverage": 0.10,
    "horizon_s": 5.0,
    "symbols": ["BTC", "ETH", "SOL"],
    "regime_gate": "feature>P80 or feature<P20 (optional)"
  }},
  "priority": 0.5,
  "reasoning": "why this hypothesis is worth testing"
}}"""


def _build_registry_summary(store, agent: str) -> str:
    """Top registered signals with IC and features."""
    try:
        registry = store.load_registry(agent)
    except Exception:
        return "No registry data available."

    if not registry:
        return "Registry is empty — no signals registered yet."

    lines = [f"{len(registry)} registered signal(s):"]
    for sig in sorted(registry, key=lambda s: -s.get("expected_ic", 0))[:10]:
        ic = sig.get("expected_ic", 0)
        feats = sig.get("features", [])
        status = sig.get("status", "?")
        lines.append(f"  IC={ic:.3f} status={status} features={feats}")
    return "\n".join(lines)


def _build_graveyard_summary(store, agent: str) -> str:
    """Failed hypotheses grouped by failure reason."""
    try:
        hypotheses = store.load_hypotheses(agent)
    except Exception:
        return "No hypothesis data available."

    failed = [h for h in hypotheses if h.get("status") == "failed"]
    if not failed:
        return "No failed hypotheses yet."

    # Group by failure reason
    reasons = {}
    for h in failed:
        reason = h.get("failure_reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

    lines = [f"{len(failed)} failed hypotheses:"]
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count}")

    # Last 5 failed claims for context
    lines.append("\nRecent failures:")
    for h in failed[-5:]:
        lines.append(f"  - {h.get('claim', '?')[:80]}")

    return "\n".join(lines)


def _build_gen_stats_summary(stats: dict) -> str:
    """Generator hit rates."""
    if not stats:
        return "No generator stats available."

    lines = ["Generator hit rates:"]
    for name, gs in sorted(stats.items(), key=lambda x: -x[1].get("weight", 0)):
        att = gs.get("attempts", 0)
        wins = gs.get("successes", 0)
        w = gs.get("weight", 0)
        lines.append(f"  {name}: {wins}/{att} ({w:.3f})")
    return "\n".join(lines)


def _load_arxiv_ideas(store) -> str:
    """Load recent ideas from arXiv processing."""
    try:
        conn = store._conn
        rows = conn.execute(
            "SELECT ideas FROM arxiv_papers WHERE processed = 1 AND ideas IS NOT NULL "
            "ORDER BY processed_at DESC LIMIT 5"
        ).fetchall()
        all_ideas = []
        for r in rows:
            ideas = json.loads(r["ideas"]) if r["ideas"] else []
            all_ideas.extend(ideas)
        if all_ideas:
            return "Recent arXiv ideas:\n" + "\n".join(f"  - {i}" for i in all_ideas[:10])
    except Exception:
        pass
    return ""


def _load_feedback(store, agent: str) -> str:
    """Load OOS feedback summary."""
    try:
        from ..feedback import build_feedback_summary
        return build_feedback_summary(store, agent)
    except Exception:
        return ""


def generate(
    manifest: dict,
    queue: HypothesisQueue,
    stats: Optional[GeneratorStats] = None,
) -> list[Hypothesis]:
    """Generate hypotheses using Claude API."""
    # Load config
    try:
        import tomllib
        with open(ROOT / "config" / "agent.toml", "rb") as f:
            config = tomllib.load(f)
        llm_config = config.get("agent", {}).get("llm", {})
    except Exception:
        log.warning("Could not load LLM config from agent.toml")
        return []

    if not llm_config.get("enabled", False):
        log.debug("LLM generator disabled in config")
        return []

    # Get store from queue (it holds the SQLite connection)
    store = getattr(queue, "_store", None)
    agent = getattr(queue, "_agent", "microstructure")

    # Build context
    registry_ctx = _build_registry_summary(store, agent) if store else ""
    graveyard_ctx = _build_graveyard_summary(store, agent) if store else ""
    gen_stats_ctx = _build_gen_stats_summary(stats or {})
    arxiv_ctx = _load_arxiv_ideas(store) if store else ""
    feedback_ctx = _load_feedback(store, agent) if store else ""

    # Build user message
    dates = sorted(manifest.get("dates", {}).keys())
    data_dir = f"data/features/{dates[-1]}" if dates else "data/features/latest"

    user_msg = f"""Generate one novel hypothesis for alpha discovery.

Data directory: {data_dir}

CURRENT REGISTRY:
{registry_ctx}

FAILED HYPOTHESES:
{graveyard_ctx}

GENERATOR PERFORMANCE:
{gen_stats_ctx}

{arxiv_ctx}

{feedback_ctx}

Avoid hypotheses similar to recent failures. Focus on unexplored feature combinations or novel regime conditions."""

    # Initialize LLM client
    try:
        from ..llm_client import LLMClient
        llm_config["agent_name"] = agent
        llm = LLMClient(llm_config, store)
    except Exception as e:
        log.warning("Could not initialize LLM client: %s", e)
        return []

    # Call Claude
    system = SYSTEM_PROMPT.format(feature_categories=FEATURE_CATEGORIES)
    response = llm.call(system, user_msg, tag="ideation")
    if not response:
        return []

    # Parse JSON response
    try:
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        if text.startswith("json"):
            text = text[4:]
        data = json.loads(text.strip())
    except json.JSONDecodeError as e:
        log.warning("LLM returned invalid JSON: %s", e)
        log.debug("Raw response: %s", response[:500])
        return []

    # Validate and create hypothesis
    claim = data.get("claim", "")
    test_protocol = data.get("test_protocol", [])
    thresholds = data.get("thresholds", {})
    priority = float(data.get("priority", 0.5))
    reasoning = data.get("reasoning", "")

    if not claim or not test_protocol:
        log.warning("LLM response missing claim or test_protocol")
        return []

    # Substitute data_dir in test protocol
    test_protocol = [cmd.replace("{data_dir}", data_dir) for cmd in test_protocol]

    h = Hypothesis.create(
        claim=claim,
        generator="llm_ideation",
        test_protocol=test_protocol,
        priority=min(1.0, max(0.0, priority)),
        thresholds=thresholds,
    )

    log.info("LLM generated hypothesis: %s (priority=%.2f, reason=%s)",
             h.id, priority, reasoning[:80])
    return [h]
