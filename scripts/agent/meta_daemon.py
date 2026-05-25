#!/usr/bin/env python3
"""NAT Meta-Agent — Orchestrator for multi-agent research network.

Coordinates the three research agents (microstructure, medium-frequency, macro)
without generating or testing hypotheses itself. Responsibilities:

    1. BUDGET      — Thompson-bandit compute allocation across agents
    2. CORRELATION  — Cross-agent signal redundancy detection
    3. PORTFOLIO    — Risk parity assembly of uncorrelated signals
    4. PROMOTION    — Portfolio-level Sharpe evaluation

NOT a ResearchAgent subclass — it sits above the research agents.

Usage:
    python scripts/agent/meta_daemon.py start         # daemon loop
    python scripts/agent/meta_daemon.py once          # single cycle
    python scripts/agent/meta_daemon.py status        # current state
    python scripts/agent/meta_daemon.py portfolio     # signal portfolio
    python scripts/agent/meta_daemon.py correlation   # cross-agent corr matrix
    python scripts/agent/meta_daemon.py budget        # agent budget allocation
    python scripts/agent/meta_daemon.py report        # full report
"""

from __future__ import annotations

import argparse
import json
import logging
import signal as signal_mod
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from agent.hypothesis import GeneratorStats  # noqa: E402
from agent.meta_portfolio import (  # noqa: E402
    compute_risk_parity_weights,
    compute_portfolio_metrics,
    filter_redundant_signals,
    evaluate_promotion,
)
from data.state import StateStore  # noqa: E402

log = logging.getLogger("nat.meta_agent")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "cycle_interval_s": 21600,          # 6 hours
    "correlation_threshold": 0.70,
    "min_portfolio_signals": 2,
    "promotion": {
        "paper_sharpe_min": 1.5,
        "paper_days": 7,
    },
}

DB_PATH = ROOT / "data" / "nat.db"

# Legacy JSON paths — used only for auto-migration and CLI display
META_STATE_PATH = ROOT / "data" / "agent_meta" / "meta_state.json"
CORRELATION_PATH = ROOT / "data" / "agent_meta" / "correlation.json"
PORTFOLIO_PATH = ROOT / "data" / "agent_meta" / "portfolio.json"

# Sub-agent identifiers (matching their agent_type in base.py)
SUB_AGENTS = ["microstructure", "medium_freq", "macro"]


def load_config() -> dict:
    """Load meta-agent config from TOML or return defaults."""
    config_path = ROOT / "config" / "agent.toml"
    if config_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(config_path, "rb") as f:
            toml_data = tomllib.load(f)
        meta_cfg = toml_data.get("meta_agent", {})
        merged = {**DEFAULT_CONFIG, **meta_cfg}
        # Merge nested promotion section
        if "promotion" in meta_cfg:
            merged["promotion"] = {
                **DEFAULT_CONFIG["promotion"],
                **meta_cfg["promotion"],
            }
        # Also pull [agent.promotion] for shared thresholds
        agent_promo = toml_data.get("agent", {}).get("promotion", {})
        if agent_promo and "promotion" not in meta_cfg:
            merged["promotion"] = {**DEFAULT_CONFIG["promotion"], **agent_promo}
        return merged
    return dict(DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# MetaAgent
# ---------------------------------------------------------------------------

class MetaAgent:
    """Orchestrator that coordinates micro, MF, and macro agents.

    NOT a ResearchAgent — doesn't generate or test hypotheses.
    Runs on a 6-hour cycle by default.
    """

    AGENTS = list(SUB_AGENTS)

    def __init__(self, config: dict | None = None, store: StateStore | None = None):
        self.config = config or load_config()
        self._store = store or StateStore(DB_PATH)
        self._shutdown = False
        self._auto_migrate()
        self._state = self._load_state()
        signal_mod.signal(signal_mod.SIGTERM, self._handle_signal)
        signal_mod.signal(signal_mod.SIGINT, self._handle_signal)

    # --- State persistence ---------------------------------------------------

    def _auto_migrate(self) -> None:
        """One-time import of legacy JSON state into SQLite."""
        self._store.migrate_from_json(
            "meta",
            state_path=META_STATE_PATH,
        )

    def _load_state(self) -> dict:
        data = self._store.load_state("meta")
        if data:
            data["history"] = self._store.load_history("meta", limit=200)
            return data
        return {
            "phase": "IDLE",
            "cycle_count": 0,
            "last_cycle_at": None,
            "budget": {},
            "history": [],
        }

    def _save_state(self) -> None:
        self._store.save_state("meta", self._state)

    def _handle_signal(self, signum, frame):
        log.info("Received signal %d, stopping after current cycle", signum)
        self._shutdown = True

    # --- Main loop -----------------------------------------------------------

    def run(self) -> None:
        """Daemon loop — run cycles at configured interval."""
        log.info("Meta-agent starting (cycle=%ds)", self.config["cycle_interval_s"])
        while not self._shutdown:
            self.run_cycle()
            interval = self.config["cycle_interval_s"]
            log.info("Meta-agent sleeping for %ds", interval)
            deadline = time.time() + interval
            while time.time() < deadline and not self._shutdown:
                time.sleep(1)
        self._state["phase"] = "STOPPED"
        self._save_state()
        log.info("Meta-agent stopped")

    def run_cycle(self) -> None:
        """Single orchestration cycle."""
        now = datetime.now(timezone.utc).isoformat()
        self._state["phase"] = "RUNNING"
        self._state["last_cycle_at"] = now
        self._save_state()

        log.info("=" * 60)
        log.info("META-AGENT CYCLE %d", self._state["cycle_count"] + 1)
        log.info("=" * 60)

        # Step 1: Update agent-level stats
        agent_stats = self.update_agent_stats()

        # Step 2: Allocate compute budget
        budget = self.allocate_budget(agent_stats)

        # Step 3: Cross-agent correlation monitoring
        flagged = self.monitor_cross_correlation()

        # Step 4: Assemble portfolio
        self.assemble_portfolio(flagged)

        # Step 5: Evaluate promotion
        self.evaluate_promotions()

        self._state["cycle_count"] += 1
        self._state["phase"] = "IDLE"
        self._state["budget"] = budget
        self._store.append_history("meta", {
            "from": "RUNNING", "to": "IDLE",
            "msg": json.dumps({"budget": budget}),
            "at": now,
        })
        self._save_state()

        log.info("Meta-agent cycle %d complete", self._state["cycle_count"])

    # --- Step 1: Agent stats -------------------------------------------------

    def update_agent_stats(self) -> dict[str, dict]:
        """Read each agent's generator stats, compute agent-level success rate."""
        agent_stats = {}
        for agent_name in self.AGENTS:
            raw = self._store.load_gen_stats(agent_name)
            total_attempts = sum(g.get("attempts", 0) for g in raw.values())
            total_successes = sum(g.get("successes", 0) for g in raw.values())

            # Thompson weight at agent level
            weight = (total_successes + 1) / (total_attempts + 2)
            agent_stats[agent_name] = {
                "attempts": total_attempts,
                "successes": total_successes,
                "weight": round(weight, 4),
            }
            log.info("  %s: %d attempts, %d successes, weight=%.3f",
                     agent_name, total_attempts, total_successes, weight)

        # Persist agent stats to meta state
        self._store.save_state("meta_stats", agent_stats)
        return agent_stats

    # --- Step 2: Budget allocation -------------------------------------------

    def allocate_budget(self, agent_stats: dict[str, dict]) -> dict[str, float]:
        """Thompson sampling at agent level → normalized compute shares."""
        if not agent_stats:
            n = len(self.AGENTS)
            return {a: round(1.0 / n, 4) for a in self.AGENTS}

        weights = {}
        for agent_name in self.AGENTS:
            stats = agent_stats.get(agent_name, {})
            weights[agent_name] = stats.get("weight", 0.5)

        total = sum(weights.values())
        if total <= 0:
            n = len(self.AGENTS)
            return {a: round(1.0 / n, 4) for a in self.AGENTS}

        budget = {a: round(w / total, 4) for a, w in weights.items()}
        log.info("Budget allocation: %s", budget)
        return budget

    # --- Step 3: Cross-agent correlation -------------------------------------

    def monitor_cross_correlation(self) -> list[dict]:
        """Load all registries, compute pairwise Spearman across agents."""
        threshold = self.config.get("correlation_threshold", 0.70)

        # Load all registries with agent labels
        all_signals = []
        all_regs = self._store.all_registries()
        for agent_name in self.AGENTS:
            for sig in all_regs.get(agent_name, []):
                if sig.get("status") == "retired":
                    continue
                sig_copy = dict(sig)
                sig_copy["_agent"] = agent_name
                all_signals.append(sig_copy)

        if len(all_signals) < 2:
            log.info("Cross-correlation: fewer than 2 active signals, skipping")
            self._save_correlation([], [])
            return []

        # Cross-agent pairs only (intra-agent already handled by each agent)
        pairs = []
        flagged = []
        for i in range(len(all_signals)):
            for j in range(i + 1, len(all_signals)):
                a = all_signals[i]
                b = all_signals[j]
                if a["_agent"] == b["_agent"]:
                    continue  # skip same-agent pairs
                corr = self._compute_signal_correlation(a, b)
                pair = {
                    "signal_a": a["name"],
                    "agent_a": a["_agent"],
                    "signal_b": b["name"],
                    "agent_b": b["_agent"],
                    "correlation": round(corr, 4) if corr is not None else None,
                }
                pairs.append(pair)
                if corr is not None and abs(corr) > threshold:
                    flagged.append(pair)
                    log.warning("  CROSS-AGENT REDUNDANCY: %s (%s) vs %s (%s) rho=%.3f",
                                a["name"][:30], a["_agent"],
                                b["name"][:30], b["_agent"], corr)

        self._save_correlation(pairs, flagged)
        log.info("Cross-correlation: %d pairs checked, %d flagged", len(pairs), len(flagged))
        return flagged

    def _compute_signal_correlation(self, sig_a: dict, sig_b: dict) -> float | None:
        """Compute Spearman correlation between two signals' feature values."""
        try:
            import numpy as np
            import pandas as pd
            from agent.runner import _load_feature_data, _extract_gated_signal
        except ImportError:
            return None

        feat_a = (sig_a.get("features") or [None])[0]
        feat_b = (sig_b.get("features") or [None])[0]
        if feat_a is None or feat_b is None:
            return None

        # Find latest data directory
        data_root = ROOT / "data" / "features"
        if not data_root.exists():
            return None
        dates = sorted(d.name for d in data_root.iterdir() if d.is_dir())
        if not dates:
            return None

        symbol = "BTC"
        for date in reversed(dates[-2:]):
            data_dir = f"data/features/{date}"
            df = _load_feature_data(data_dir, symbol)
            if df is None or len(df) < 100:
                continue

            vals_a = _extract_gated_signal(df, feat_a, sig_a.get("regime_gate"))
            vals_b = _extract_gated_signal(df, feat_b, sig_b.get("regime_gate"))
            if vals_a is None or vals_b is None:
                continue

            valid = ~(np.isnan(vals_a) | np.isnan(vals_b))
            if valid.sum() < 100:
                continue

            corr = pd.Series(vals_a[valid]).corr(
                pd.Series(vals_b[valid]), method="spearman"
            )
            return float(corr) if not np.isnan(corr) else None

        return None

    def _save_correlation(self, pairs: list[dict], flagged: list[dict]) -> None:
        CORRELATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CORRELATION_PATH, "w") as f:
            json.dump({
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "pairs": pairs,
                "flagged": flagged,
            }, f, indent=2)

    # --- Step 4: Portfolio assembly ------------------------------------------

    def assemble_portfolio(self, flagged_pairs: list[dict]) -> dict:
        """Build risk-parity-weighted portfolio from all agent registries."""
        all_signals = []
        all_regs = self._store.all_registries()
        for agent_name in self.AGENTS:
            for sig in all_regs.get(agent_name, []):
                if sig.get("status") == "retired":
                    continue
                sig_copy = dict(sig)
                sig_copy["_agent"] = agent_name
                all_signals.append(sig_copy)

        # Filter redundant signals
        filtered = filter_redundant_signals(all_signals, flagged_pairs)

        min_signals = self.config.get("min_portfolio_signals", 2)
        if len(filtered) < min_signals:
            log.info("Portfolio: %d signals < minimum %d, portfolio empty",
                     len(filtered), min_signals)
            portfolio = {
                "signals": [],
                "total_signals": 0,
                "effective_n": 0.0,
                "portfolio_ic": 0.0,
                "assembled_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_portfolio(portfolio)
            return portfolio

        # Compute risk parity weights
        weights = compute_risk_parity_weights(filtered)
        metrics = compute_portfolio_metrics(filtered, weights)

        portfolio_signals = []
        for sig, w in zip(filtered, weights):
            portfolio_signals.append({
                "name": sig["name"],
                "agent": sig.get("_agent", "unknown"),
                "weight": round(w, 6),
                "expected_ic": sig.get("expected_ic", 0.0),
                "features": sig.get("features", []),
                "regime_gate": sig.get("regime_gate"),
                "status": sig.get("status", "validated"),
            })

        portfolio = {
            "signals": portfolio_signals,
            "assembled_at": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        self._save_portfolio(portfolio)

        log.info("Portfolio: %d signals, IC=%.4f, effective_N=%.1f",
                 metrics["total_signals"], metrics["portfolio_ic"],
                 metrics["effective_n"])
        return portfolio

    def _save_portfolio(self, portfolio: dict) -> None:
        PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PORTFOLIO_PATH, "w") as f:
            json.dump(portfolio, f, indent=2)

    # --- Step 5: Promotion evaluation ----------------------------------------

    def evaluate_promotions(self) -> dict:
        """Evaluate portfolio-level Sharpe for paper→live promotion."""
        promo_config = self.config.get("promotion", {})
        paper_sharpe_min = promo_config.get("paper_sharpe_min", 1.5)
        paper_days = promo_config.get("paper_days", 7)

        # Build IC history from past portfolio snapshots
        ic_history = self._load_portfolio_ic_history()

        result = evaluate_promotion(
            ic_history,
            paper_sharpe_min=paper_sharpe_min,
            paper_days=paper_days,
        )

        if result["recommended"]:
            log.info("PROMOTION RECOMMENDED: %s", result["reason"])
        else:
            log.info("Promotion: %s", result["reason"])

        return result

    def _load_portfolio_ic_history(self) -> list[float]:
        """Load historical portfolio IC values from state history."""
        history = self._state.get("portfolio_ic_history", [])
        # Append current portfolio IC if available
        if PORTFOLIO_PATH.exists():
            with open(PORTFOLIO_PATH) as f:
                portfolio = json.load(f)
            ic = portfolio.get("portfolio_ic", 0.0)
            if ic != 0.0:
                history.append(ic)
                self._state["portfolio_ic_history"] = history[-60:]
                self._save_state()
        return history

    # --- CLI helpers ---------------------------------------------------------

    def print_status(self) -> None:
        print(f"Phase:       {self._state.get('phase', 'UNKNOWN')}")
        print(f"Cycles:      {self._state.get('cycle_count', 0)}")
        print(f"Last cycle:  {self._state.get('last_cycle_at', 'never')}")

        budget = self._state.get("budget", {})
        if budget:
            print(f"\nBudget allocation:")
            for agent, share in sorted(budget.items()):
                print(f"  {agent:18s}  {share:.1%}")

        print(f"\nAgent registries:")
        all_regs = self._store.all_registries()
        for agent_name in self.AGENTS:
            n = sum(1 for s in all_regs.get(agent_name, [])
                    if s.get("status") != "retired")
            print(f"  {agent_name:18s}  {n} active signals")

    def print_portfolio(self) -> None:
        if not PORTFOLIO_PATH.exists():
            print("(no portfolio assembled yet)")
            return
        with open(PORTFOLIO_PATH) as f:
            p = json.load(f)
        print(f"Portfolio assembled at: {p.get('assembled_at', '?')}")
        print(f"Total signals: {p.get('total_signals', 0)}")
        print(f"Portfolio IC:  {p.get('portfolio_ic', 0):.4f}")
        print(f"Effective N:   {p.get('effective_n', 0):.1f}")
        print()
        for sig in p.get("signals", []):
            print(f"  w={sig['weight']:.3f}  IC={sig['expected_ic']:.3f}  "
                  f"[{sig['agent']:12s}]  {sig['name']}")

    def print_correlation(self) -> None:
        if not CORRELATION_PATH.exists():
            print("(no correlation data yet)")
            return
        with open(CORRELATION_PATH) as f:
            data = json.load(f)
        print(f"Computed at: {data.get('computed_at', '?')}")
        pairs = data.get("pairs", [])
        flagged = data.get("flagged", [])
        print(f"Pairs checked: {len(pairs)}")
        print(f"Flagged (|rho| > threshold): {len(flagged)}")
        if flagged:
            print()
            for p in flagged:
                print(f"  rho={p['correlation']:+.3f}  "
                      f"{p['signal_a'][:25]:25s} ({p['agent_a']}) vs "
                      f"{p['signal_b'][:25]:25s} ({p['agent_b']})")

    def print_budget(self) -> None:
        # Load agent stats from SQLite
        stats = self._store.load_state("meta_stats") or {}

        budget = self._state.get("budget", {})
        if not budget:
            n = len(self.AGENTS)
            budget = {a: round(1.0 / n, 4) for a in self.AGENTS}

        print(f"{'Agent':18s}  {'Attempts':>8s}  {'Success':>7s}  "
              f"{'Weight':>7s}  {'Budget':>7s}")
        print("-" * 55)
        for agent in self.AGENTS:
            s = stats.get(agent, {})
            print(f"{agent:18s}  {s.get('attempts', 0):8d}  "
                  f"{s.get('successes', 0):7d}  "
                  f"{s.get('weight', 0.5):7.3f}  "
                  f"{budget.get(agent, 0):7.1%}")

    def print_report(self) -> None:
        print("=" * 60)
        print("  NAT Meta-Agent Report")
        print("=" * 60)

        print()
        self.print_status()

        print(f"\n{'─' * 60}")
        print("BUDGET ALLOCATION")
        print(f"{'─' * 60}")
        self.print_budget()

        print(f"\n{'─' * 60}")
        print("PORTFOLIO")
        print(f"{'─' * 60}")
        self.print_portfolio()

        print(f"\n{'─' * 60}")
        print("CROSS-AGENT CORRELATION")
        print(f"{'─' * 60}")
        self.print_correlation()
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from logging_config import setup_logging
    setup_logging("nat.meta_agent")

    parser = argparse.ArgumentParser(description="NAT Meta-Agent (Orchestrator)")
    parser.add_argument("action", choices=[
        "start", "stop", "once", "status",
        "portfolio", "correlation", "budget", "report",
    ], help="Action to perform")
    args = parser.parse_args()

    agent = MetaAgent()

    if args.action == "start":
        agent.run()
    elif args.action == "once":
        agent.run_cycle()
    elif args.action == "stop":
        print("Send SIGTERM to the running meta-agent process to stop it.")
    elif args.action == "status":
        agent.print_status()
    elif args.action == "portfolio":
        agent.print_portfolio()
    elif args.action == "correlation":
        agent.print_correlation()
    elif args.action == "budget":
        agent.print_budget()
    elif args.action == "report":
        agent.print_report()


if __name__ == "__main__":
    main()
