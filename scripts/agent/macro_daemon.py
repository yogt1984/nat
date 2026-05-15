#!/usr/bin/env python3
"""NAT Macro Agent — alpha discovery at 1h-24h horizons.

Main loop:
    1. UPDATE MANIFEST  — scan data/features/, write manifest.json
    2. GENERATE         — funding_meanrev, oi_divergence, whale_momentum
    3. EXECUTE          — pop highest-priority, run 4-gate protocol
    4. MONITOR          — check registered signals for IC decay
    5. SLEEP            — wait until next cycle

Discovers alpha signals from 1h-resampled bars using funding rate,
OI divergence, and whale flow features. Separate state from micro/MF agents.

Usage:
    python scripts/agent/macro_daemon.py start
    python scripts/agent/macro_daemon.py status
    python scripts/agent/macro_daemon.py once
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from agent.base import ResearchAgent, BaseRunner, AgentPhase  # noqa: E402
from agent.hypothesis import Hypothesis, GeneratorStats  # noqa: E402

log = logging.getLogger("nat.agent_macro")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "cycle_interval_s": 14400,         # 4 hours between cycles
    "max_experiments_per_cycle": 6,
    "max_cycle_runtime_s": 7200,
    "timeframe": "1h",
    "generators_enabled": [
        "funding_meanrev",
        "oi_divergence",
        "whale_momentum",
    ],
    "gates": {
        "min_ic": 0.07,
        "min_dIC": 0.02,
        "fdr_q": 0.05,
        "min_oos_dates": 2,
        "min_symbols": 2,
    },
    "decay": {
        "ic_decay_ratio": 0.5,
        "consecutive_days_limit": 14,
    },
}

MACRO_STATE_PATH = ROOT / "data" / "agent_macro" / "agent_state.json"
MACRO_STATS_PATH = ROOT / "data" / "agent_macro" / "generator_stats.json"
MACRO_REGISTRY_PATH = ROOT / "data" / "agent_macro" / "registry.json"


def load_config() -> dict:
    """Load macro agent config from TOML or return defaults."""
    config_path = ROOT / "config" / "agent.toml"
    if config_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(config_path, "rb") as f:
            return {**DEFAULT_CONFIG, **tomllib.load(f).get("agent_macro", {})}
    return DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# MacroAgent
# ---------------------------------------------------------------------------

class MacroAgent(ResearchAgent):
    """Macro research agent for 1h-24h alpha discovery.

    Discovers signals from 1h-resampled bars using funding rate mean-reversion,
    OI divergence, and whale flow momentum. Separate state from micro/MF agents.
    """

    agent_type = "macro"
    default_generators = ["funding_meanrev", "oi_divergence", "whale_momentum"]

    # Prefixes registered at generator import time

    # --- Path overrides ------------------------------------------------------

    @property
    def root(self) -> Path:
        return ROOT

    @property
    def state_path(self) -> Path:
        return MACRO_STATE_PATH

    @property
    def queue_path(self) -> Path:
        return ROOT / "data" / "agent_macro" / "hypotheses.json"

    @property
    def stats_path(self) -> Path:
        return MACRO_STATS_PATH

    # --- Config --------------------------------------------------------------

    def load_config(self) -> dict:
        return load_config()

    # --- Abstract hook implementations ---------------------------------------

    def get_generator(self, name: str):
        """Lazy-import macro generator functions."""
        try:
            if name == "funding_meanrev":
                from agent.generators.macro.funding_meanrev import generate
                return generate
            elif name == "oi_divergence":
                from agent.generators.macro.oi_divergence import generate
                return generate
            elif name == "whale_momentum":
                from agent.generators.macro.whale_momentum import generate
                return generate
        except ImportError as e:
            log.debug("Macro generator %s not yet implemented: %s", name, e)
        return None

    def create_runner(self, hypothesis: Hypothesis, manifest: dict) -> BaseRunner:
        from agent.macro_runner import MacroRunner
        return MacroRunner(hypothesis, manifest)

    def pre_execute(self, hypothesis: Hypothesis) -> None:
        """Inject adaptive IC threshold from macro registry."""
        adaptive_min_ic = self._compute_adaptive_ic()
        hypothesis.thresholds.setdefault("min_ic", 0.07)
        if adaptive_min_ic > hypothesis.thresholds["min_ic"]:
            hypothesis.thresholds["min_ic"] = adaptive_min_ic

    def on_fdr_reject(self, hypothesis: Hypothesis) -> None:
        """Remove FDR-rejected signal from macro registry."""
        self._remove_from_registry(hypothesis.id)

    def run_monitor(self) -> None:
        """Check macro registered signals for IC decay."""
        if not MACRO_REGISTRY_PATH.exists():
            return
        with open(MACRO_REGISTRY_PATH) as f:
            registry = json.load(f)
        if not registry:
            return

        decay_cfg = self.config.get("decay", {})
        decay_ratio = decay_cfg.get("ic_decay_ratio", 0.5)
        decay_days_limit = decay_cfg.get("consecutive_days_limit", 14)

        n_retired = 0
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        modified = False

        for sig in registry:
            if sig.get("status") == "retired":
                continue
            rolling_ic = self._compute_rolling_ic(sig)
            if rolling_ic is not None:
                expected_ic = sig.get("expected_ic", 0)
                threshold = expected_ic * decay_ratio
                sig.setdefault("ic_history", [])
                sig["ic_history"].append({"date": today, "ic": rolling_ic})
                sig["ic_history"] = sig["ic_history"][-30:]
                sig["latest_ic"] = rolling_ic
                modified = True

                if rolling_ic < threshold:
                    days = sig.get("decay_days", 0) + 1
                    sig["decay_days"] = days
                    log.warning("  MACRO IC DECAY: %s IC=%.3f < %.3f, day %d/%d",
                                sig["name"][:40], rolling_ic, threshold,
                                days, decay_days_limit)
                    if days >= decay_days_limit:
                        sig["status"] = "retired"
                        sig["retired_reason"] = "ic_decay"
                        sig["retired_date"] = today
                        n_retired += 1
                        log.info("  MACRO AUTO-RETIRED: %s", sig["name"][:40])
                else:
                    if sig.get("decay_days", 0) > 0:
                        log.info("  MACRO IC recovered: %s IC=%.3f, resetting decay",
                                 sig["name"][:40], rolling_ic)
                    sig["decay_days"] = 0

        if modified or n_retired > 0:
            with open(MACRO_REGISTRY_PATH, "w") as f:
                json.dump(registry, f, indent=2)

        if n_retired > 0:
            self.state.set("total_signals_registered",
                          max(0, self.state.get("total_signals_registered", 0) - n_retired))

        n_active = sum(1 for s in registry if s.get("status") != "retired")
        log.info("Macro Monitor: %d active signals, %d retired this cycle", n_active, n_retired)

    # --- Domain-specific methods ---------------------------------------------

    @staticmethod
    def _remove_from_registry(hypothesis_id: str) -> None:
        """Remove a signal from the macro registry by its hypothesis_id."""
        if not MACRO_REGISTRY_PATH.exists():
            return
        with open(MACRO_REGISTRY_PATH) as f:
            registry = json.load(f)
        registry = [s for s in registry if s.get("hypothesis_id") != hypothesis_id]
        with open(MACRO_REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)

    def _compute_adaptive_ic(self) -> float:
        """Compute adaptive IC threshold: max(floor, median(registry_ic) * 0.8)."""
        floor_ic = self.config.get("gates", {}).get("min_ic", 0.07)
        if not MACRO_REGISTRY_PATH.exists():
            return floor_ic
        with open(MACRO_REGISTRY_PATH) as f:
            registry = json.load(f)
        ics = [s.get("expected_ic", 0) for s in registry
               if s.get("status") != "retired"]
        if not ics:
            return floor_ic
        ics.sort()
        median_ic = ics[len(ics) // 2]
        adaptive = max(floor_ic, median_ic * 0.8)
        log.info("Macro Adaptive IC: median=%.3f -> threshold=%.3f (floor=%.2f)",
                 median_ic, adaptive, floor_ic)
        return adaptive

    def _compute_rolling_ic(self, sig: dict) -> float | None:
        """Compute rolling IC for a macro signal on the latest data.

        Loads the latest Parquet data, resamples to 1h bars, and computes
        Spearman IC between the signal feature and N-bar forward returns.
        """
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            return None

        from agent.runner import _load_feature_data

        data_root = ROOT / "data" / "features"
        if not data_root.exists():
            return None
        dates = sorted(d.name for d in data_root.iterdir() if d.is_dir())
        if not dates:
            return None

        features = sig.get("features", [])
        if not features:
            return None
        feature = features[0]
        symbol = sig.get("symbols", ["BTC"])[0]
        horizon_s = sig.get("horizon_s", 3600.0)

        for date in reversed(dates[-2:]):
            data_dir = f"data/features/{date}"
            df = _load_feature_data(data_dir, symbol)
            if df is None or len(df) < 500:
                continue

            # Resample to 1h bars
            try:
                from cluster_pipeline.preprocess import aggregate_bars
                bars = aggregate_bars(df, "1h")
            except (ImportError, Exception):
                continue

            # Find the signal feature column
            feat_col = None
            for suffix in ("_last", "_mean", "_sum", ""):
                candidate = f"{feature}{suffix}" if suffix else feature
                if candidate in bars.columns:
                    feat_col = candidate
                    break
            if feat_col is None:
                continue

            signal_vals = bars[feat_col].to_numpy(dtype=float)

            # Forward returns at bar horizon
            horizon_bars = max(1, int(horizon_s / 3600))
            if "raw_midprice_mean" in bars.columns:
                mid_col = "raw_midprice_mean"
            elif "raw_midprice_last" in bars.columns:
                mid_col = "raw_midprice_last"
            else:
                continue
            mid = bars[mid_col].to_numpy(dtype=float)
            fwd = np.full_like(mid, np.nan)
            if len(mid) > horizon_bars:
                fwd[:-horizon_bars] = (mid[horizon_bars:] - mid[:-horizon_bars]) / mid[:-horizon_bars]

            valid = ~(np.isnan(signal_vals) | np.isnan(fwd))
            if valid.sum() < 20:  # Fewer bars at 1h resolution
                continue

            ic = pd.Series(signal_vals[valid]).corr(
                pd.Series(fwd[valid]), method="spearman",
            )
            return float(ic) if not np.isnan(ic) else None

        return None

    def print_report(self) -> None:
        """Print a full summary of the macro agent."""
        s = self.state

        print("=" * 60)
        print("  NAT Macro Agent Report")
        print("=" * 60)

        print(f"\nPhase:       {s.phase.value}")
        print(f"Cycles:      {s.get('cycle_count', 0)}")
        print(f"Tested:      {s.get('total_hypotheses_tested', 0)} hypotheses")
        print(f"Registered:  {s.get('total_signals_registered', 0)} signals")
        print(f"Queue depth: {self.queue.depth}")
        print(f"Graveyard:   {len(self.queue.graveyard)}")

        print(f"\n{'─' * 60}")
        print("MACRO REGISTRY")
        print(f"{'─' * 60}")
        if MACRO_REGISTRY_PATH.exists():
            with open(MACRO_REGISTRY_PATH) as f:
                registry = json.load(f)
            if registry:
                for sig in registry:
                    print(f"  IC={sig['expected_ic']:.3f}  gate={sig.get('regime_gate', '-'):30s}  "
                          f"{sig['name'][:50]}")
            else:
                print("  (empty)")
        else:
            print("  (no registry file)")

        print(f"\n{'─' * 60}")
        print("GRAVEYARD BREAKDOWN")
        print(f"{'─' * 60}")
        reasons = {}
        for h in self.queue.graveyard:
            r = h.failure_reason or "unknown"
            reasons[r] = reasons.get(r, 0) + 1
        if reasons:
            for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"  {r:20s}  {count}")
        else:
            print("  (no failures)")

        print(f"\n{'─' * 60}")
        print("GENERATOR PERFORMANCE")
        print(f"{'─' * 60}")
        for name, gs in self.gen_stats.items():
            print(f"  {name:18s}  attempts={gs.attempts:3d}  "
                  f"successes={gs.successes:3d}  "
                  f"hit_rate={gs.hit_rate:.0%}  "
                  f"weight={gs.weight:.3f}")

        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="NAT Macro Agent")
    parser.add_argument("action", choices=["start", "status", "once", "queue",
                                            "registry", "graveyard", "report"],
                        help="Action to perform")
    args = parser.parse_args()

    agent = MacroAgent()

    if args.action == "start":
        agent.run()
    elif args.action == "once":
        agent.run_cycle()
    elif args.action == "status":
        agent.print_status()
    elif args.action == "queue":
        for h in agent.queue.peek(20):
            print(f"  [{h.priority:6.3f}] {h.id}  {h.generator:18s}  {h.claim[:60]}")
    elif args.action == "registry":
        if MACRO_REGISTRY_PATH.exists():
            with open(MACRO_REGISTRY_PATH) as f:
                for sig in json.load(f):
                    print(f"  IC={sig['expected_ic']:.3f}  {sig['status']:10s}  "
                          f"{','.join(sig['symbols']):12s}  {sig['name']}")
        else:
            print("  (empty)")
    elif args.action == "graveyard":
        for h in agent.queue.graveyard[-20:]:
            print(f"  {h.id}  {h.failure_reason or '??':20s}  {h.claim[:50]}")
    elif args.action == "report":
        agent.print_report()


if __name__ == "__main__":
    main()
