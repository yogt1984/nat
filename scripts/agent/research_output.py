"""Structured hypothesis output — self-contained JSON per hypothesis and cycle.

Emits to data/research/hypotheses/{id}.json and data/research/cycles/{cycle_id}.json.
Each file contains enough detail to render a full research page on the website.

Schema:
    Hypothesis record:
        id, agent, generator, claim, math, status, failure_reason,
        gates (per-gate: name, metric, threshold, p_value, passed, message),
        features, regime_gate, horizon_s, thresholds,
        timestamps (created, completed), parent_id

    Cycle summary:
        cycle_id, agent, started, completed, duration_s,
        n_tested, n_registered, n_fdr_rejected, n_chained,
        fdr_q, hypotheses (list of ids with status),
        generator_stats
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("nat.agent")

# Redis channel for real-time research events
_RESEARCH_CHANNEL = "nat:research:events"

# Lazy-initialized Redis connection for publishing
_redis_conn = None


def _get_redis():
    """Lazy-connect to Redis for event publishing. Returns None if unavailable."""
    global _redis_conn
    if _redis_conn is not None:
        return _redis_conn
    try:
        import redis as redis_lib
        url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")
        _redis_conn = redis_lib.Redis.from_url(url, socket_connect_timeout=2)
        _redis_conn.ping()
        return _redis_conn
    except Exception:
        log.debug("Redis not available, research events will not be published")
        _redis_conn = None
        return None


def publish_research_event(event_type: str, payload: dict) -> bool:
    """Publish a research event to Redis for WebSocket streaming.

    Event types: hypothesis_started, gate_passed, gate_failed,
                 hypothesis_registered, cycle_completed

    Returns True if published, False if Redis unavailable.
    """
    conn = _get_redis()
    if conn is None:
        return False
    msg = json.dumps({"event": event_type, **payload}, default=str)
    try:
        conn.publish(_RESEARCH_CHANNEL, msg)
        return True
    except Exception:
        log.debug("Failed to publish research event: %s", event_type)
        return False

# Default output root
_OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "research"

# LaTeX derivations per generator type
GENERATOR_MATH = {
    "systematic": (
        r"H_0: \text{IC}(\text{signal} \mid \text{gate}) = 0 \quad "
        r"\text{vs} \quad H_1: |\text{IC}| \geq \text{IC}_{\min}" "\n\n"
        r"\text{IC} = \text{Spearman}(\text{signal}_t, r_{t+h}) \quad "
        r"h = \text{horizon\_s}" "\n\n"
        r"\Delta\text{IC} = \text{IC}_{\text{gated}} - \text{IC}_{\text{ungated}} "
        r"\geq \Delta\text{IC}_{\min}"
    ),
    "spectral": (
        r"\text{PSD}(f) = |\mathcal{F}\{x(t)\}|^2" "\n\n"
        r"\text{Band IC}(f_1, f_2) = \text{Spearman}("
        r"\mathcal{F}^{-1}\{\hat{x}(f) \cdot \mathbb{1}_{[f_1,f_2]}\}, r_{t+h})" "\n\n"
        r"\text{SNR}(f) = \frac{\text{PSD}_{\text{signal}}(f)}"
        r"{\text{PSD}_{\text{noise}}(f)}"
    ),
    "regime": (
        r"P(\text{regime} = k \mid x_t) \propto \pi_k \cdot "
        r"\mathcal{N}(x_t; \mu_k, \Sigma_k)" "\n\n"
        r"\text{IC}_k = \text{Spearman}(\text{signal}_t, r_{t+h} "
        r"\mid \text{regime} = k)" "\n\n"
        r"\text{Coverage}_k = P(\text{regime} = k) \geq c_{\min}"
    ),
    "cross_asset": (
        r"\rho_{ij}(h) = \text{Spearman}(x^{(i)}_t, r^{(j)}_{t+h})" "\n\n"
        r"\text{Lead-lag: } \arg\max_h |\rho_{ij}(h)| \neq 0 "
        r"\Rightarrow \text{predictive cross-signal}"
    ),
    "recycler": (
        r"\text{Recycled from } H_{\text{parent}} \text{ with modified thresholds:}" "\n\n"
        r"\theta' = \theta \cdot (1 \pm \delta), \quad "
        r"\delta \in \{0.1, 0.2, 0.5\}"
    ),
    "ensemble": (
        r"s_{\text{ens}}(t) = \sum_{i=1}^{N} w_i \cdot s_i(t)" "\n\n"
        r"w_i \propto \text{IC}_i \cdot (1 - \max_j |\rho_{ij}|)" "\n\n"
        r"\text{IC}_{\text{ens}} \geq \max_i \text{IC}_i \text{ (diversification gain)}"
    ),
    "momentum": (
        r"\text{mom}(t, \tau) = \frac{p_t - p_{t-\tau}}{p_{t-\tau}}" "\n\n"
        r"\text{R}^2_{\text{trend}} = 1 - \frac{\text{Var}(r_t - \hat{r}_t)}"
        r"{\text{Var}(r_t)}" "\n\n"
        r"H = \frac{\log(R/S)}{\log(n)} \quad \text{(Hurst exponent)}"
    ),
    "vol_breakout": (
        r"\sigma_{\text{short}} / \sigma_{\text{long}} > \tau_{\text{breakout}}" "\n\n"
        r"z_{\sigma} = \frac{\sigma_t - \bar{\sigma}}{\text{std}(\sigma)}" "\n\n"
        r"\text{IC} = \text{Spearman}(z_{\sigma}, |r_{t+h}|)"
    ),
    "flow_cluster": (
        r"C_k = \text{KMeans}(\text{flow features}, k)" "\n\n"
        r"\text{IC}_k = \text{Spearman}(\mathbb{1}_{C_k}, r_{t+h})" "\n\n"
        r"\text{Select } k^* = \arg\max_k |\text{IC}_k|"
    ),
    "funding_meanrev": (
        r"f_t = \text{funding rate}" "\n\n"
        r"z_f = \frac{f_t - \bar{f}_{\tau}}{\text{std}(f_{\tau})}" "\n\n"
        r"\text{Signal} = -z_f \quad \text{(mean-reversion)}"
    ),
    "oi_divergence": (
        r"\Delta\text{OI}_t = \text{OI}_t - \text{OI}_{t-1}" "\n\n"
        r"\text{Divergence} = \text{sign}(\Delta\text{OI}_t) \neq "
        r"\text{sign}(r_t)" "\n\n"
        r"\text{IC} = \text{Spearman}(\Delta\text{OI}_t, r_{t+h})"
    ),
    "whale_momentum": (
        r"\text{WF}_t = \sum_{|\text{size}| > q_{99}} \text{sign}(\text{side}) "
        r"\cdot |\text{size}|" "\n\n"
        r"\text{IC} = \text{Spearman}(\text{WF}_t, r_{t+h})"
    ),
    "it_discovery": (
        r"I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}" "\n\n"
        r"\hat{I}_{\text{KSG}}(X; Y) = \psi(k) - \langle\psi(n_x + 1) "
        r"+ \psi(n_y + 1)\rangle + \psi(N)" "\n\n"
        r"\text{Select features by } \arg\max_S \hat{I}(S; r_{t+h})"
    ),
}


def _parse_gate_results(results: Optional[dict]) -> list[dict]:
    """Parse gate results from hypothesis results dict into structured records."""
    if not results:
        return []

    gates = []
    for gr in results.get("gate_results", []):
        msg = gr.get("msg", "")
        gate = {
            "name": _infer_gate_name(msg),
            "passed": gr.get("passed", False),
            "message": msg,
            "metric": _extract_metric(msg),
            "threshold": _extract_threshold(msg),
            "p_value": _extract_pvalue(msg),
        }
        gates.append(gate)

    # Cost check gate
    if "cost_check" in results:
        msg = results["cost_check"]
        gates.append({
            "name": "cost",
            "passed": "PASS" in msg,
            "message": msg,
            "metric": _extract_metric(msg),
            "threshold": _extract_threshold(msg),
            "p_value": None,
        })

    # Correlation check gate
    if "correlation_check" in results:
        msg = results["correlation_check"]
        gates.append({
            "name": "correlation",
            "passed": "REDUNDANT" not in msg,
            "message": msg,
            "metric": _extract_metric(msg),
            "threshold": None,
            "p_value": None,
        })

    # Symbol replication gate
    if "symbol_replication" in results:
        sr = results["symbol_replication"]
        gates.append({
            "name": "symbol_replication",
            "passed": sr.get("n_pass", 0) > 0,
            "message": f"passed={sr.get('passed', [])}, failed={sr.get('failed', [])}",
            "metric": sr.get("n_pass", 0),
            "threshold": None,
            "p_value": None,
        })

    return gates


def _infer_gate_name(msg: str) -> str:
    """Infer gate name from message content."""
    if "KEEP=" in msg:
        return "walkforward"
    if "dIC=" in msg and "IC=" in msg and msg.index("dIC=") < msg.index("IC="):
        return "dIC"
    if "coverage=" in msg:
        return "coverage"
    if "avg_ret=" in msg:
        return "cost"
    if "max_corr=" in msg:
        return "correlation"
    if "IC=" in msg:
        return "IC"
    return "unknown"


def _extract_metric(msg: str) -> Optional[float]:
    """Extract the primary metric value from a gate message."""
    patterns = [
        (r"IC=([+-]?\d+\.?\d*)", float),
        (r"dIC=([+-]?\d+\.?\d*)", float),
        (r"avg_ret=([+-]?\d+\.?\d*)", float),
        (r"max_corr=([+-]?\d+\.?\d*)", float),
        (r"coverage=(\d+\.?\d*%?)", lambda s: float(s.rstrip("%")) / 100 if "%" in s else float(s)),
    ]
    for pattern, converter in patterns:
        m = re.search(pattern, msg)
        if m:
            try:
                return converter(m.group(1))
            except (ValueError, ZeroDivisionError):
                pass
    return None


def _extract_threshold(msg: str) -> Optional[float]:
    """Extract threshold from 'vs min=...' patterns."""
    m = re.search(r"vs min=([+-]?\d+\.?\d*)", msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"threshold=([+-]?\d+\.?\d*)", msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_pvalue(msg: str) -> Optional[float]:
    """Extract p-value from 'p=...' patterns."""
    m = re.search(r"p=([+-]?\d+\.?\d*(?:e[+-]?\d+)?)", msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def build_hypothesis_record(
    hypothesis,
    agent_type: str,
    output_root: Optional[Path] = None,
) -> dict:
    """Build a structured hypothesis record from a Hypothesis object.

    Returns the record dict (also writes to disk if output_root is provided).
    """
    h = hypothesis
    gates = _parse_gate_results(h.results)
    features = _extract_features_from_thresholds(h.thresholds)
    math = GENERATOR_MATH.get(h.generator, "")

    record = {
        "schema_version": 1,
        "id": h.id,
        "agent": agent_type,
        "generator": h.generator,
        "claim": h.claim,
        "math": math,
        "status": h.status,
        "failure_reason": h.failure_reason,
        "gates": gates,
        "features": features,
        "regime_gate": h.thresholds.get("regime_gate"),
        "horizon_s": h.thresholds.get("horizon_s"),
        "thresholds": h.thresholds,
        "parent_id": h.parent_id,
        "timestamps": {
            "created": h.created,
            "completed": h.completed,
        },
    }

    if output_root is not None:
        _write_record(output_root / "hypotheses", h.id, record)

    return record


def build_cycle_summary(
    cycle_id: str,
    agent_type: str,
    started: str,
    duration_s: float,
    hypotheses: list,
    n_registered: int,
    n_fdr_rejected: int,
    n_chained: int,
    fdr_q: float,
    generator_stats: dict,
    output_root: Optional[Path] = None,
) -> dict:
    """Build a structured cycle summary.

    Args:
        hypotheses: list of Hypothesis objects tested in this cycle
    """
    completed = datetime.now(timezone.utc).isoformat()

    hyp_summaries = []
    for h in hypotheses:
        hyp_summaries.append({
            "id": h.id,
            "generator": h.generator,
            "claim": h.claim[:100],
            "status": h.status,
            "failure_reason": h.failure_reason,
        })

    gen_stats = {}
    for name, gs in generator_stats.items():
        gen_stats[name] = {
            "attempts": gs.attempts,
            "successes": gs.successes,
            "hit_rate": gs.hit_rate,
            "weight": gs.weight,
        }

    summary = {
        "schema_version": 1,
        "cycle_id": cycle_id,
        "agent": agent_type,
        "started": started,
        "completed": completed,
        "duration_s": round(duration_s, 1),
        "n_tested": len(hypotheses),
        "n_registered": n_registered,
        "n_fdr_rejected": n_fdr_rejected,
        "n_chained": n_chained,
        "fdr_q": fdr_q,
        "hypotheses": hyp_summaries,
        "generator_stats": gen_stats,
    }

    if output_root is not None:
        _write_record(output_root / "cycles", cycle_id, summary)

    return summary


def _extract_features_from_thresholds(thresholds: dict) -> list[str]:
    """Extract feature names from threshold config."""
    gate = thresholds.get("regime_gate", "")
    features = []
    if gate:
        m = re.match(r"^([a-z_0-9]+)[<>]", gate)
        if m:
            features.append(m.group(1))
    return features


def _write_record(directory: Path, name: str, data: dict) -> None:
    """Write a JSON record to disk."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.debug("Wrote research record: %s", path)
