#!/usr/bin/env python3
"""NAT Medium-Frequency Agent — alpha discovery at 1min-1h horizons.

Main loop:
    1. UPDATE MANIFEST  — scan data/features/, write manifest.json
    2. GENERATE         — momentum, vol_breakout, flow_cluster generators
    3. EXECUTE          — pop highest-priority, run 4-gate protocol
    4. MONITOR          — check paper trading metrics for registered signals
    5. SLEEP            — wait until next cycle

Discovers alpha signals resampled to 5min bars. Runs alongside the
microstructure agent with fully separate state, queue, and registry.

Usage:
    python scripts/agent/mf_daemon.py start     # run main loop
    python scripts/agent/mf_daemon.py status    # print current state
    python scripts/agent/mf_daemon.py once      # single cycle (for testing)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from agent.base import ResearchAgent, BaseRunner, AgentPhase  # noqa: E402
from agent.hypothesis import Hypothesis, GeneratorStats  # noqa: E402

log = logging.getLogger("nat.agent_mf")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "cycle_interval_s": 7200,          # 2 hours between cycles
    "max_experiments_per_cycle": 8,
    "max_cycle_runtime_s": 5400,
    "timeframe": "5min",
    "generators_enabled": [
        "momentum",
        "vol_breakout",
        "flow_cluster",
    ],
    "gates": {
        "min_ic": 0.08,
        "min_dIC": 0.03,
        "fdr_q": 0.05,
        "min_oos_dates": 2,
        "min_symbols": 2,
    },
    "decay": {
        "ic_decay_ratio": 0.5,
        "consecutive_days_limit": 14,
    },
}

MF_STATE_PATH = ROOT / "data" / "agent_mf" / "agent_state.json"
MF_STATS_PATH = ROOT / "data" / "agent_mf" / "generator_stats.json"
MF_REGISTRY_PATH = ROOT / "data" / "agent_mf" / "registry.json"


def load_config() -> dict:
    """Load medium-frequency agent config from TOML or return defaults."""
    config_path = ROOT / "config" / "agent.toml"
    if config_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(config_path, "rb") as f:
            return {**DEFAULT_CONFIG, **tomllib.load(f).get("agent_mf", {})}
    return DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# MediumFrequencyAgent
# ---------------------------------------------------------------------------

class MediumFrequencyAgent(ResearchAgent):
    """Medium-frequency research agent for 1min-1h alpha discovery.

    Discovers signals from 5min-resampled bars using momentum, vol breakout,
    and flow clustering generators. Separate state from microstructure agent.
    """

    agent_type = "medium_freq"
    default_generators = ["momentum", "vol_breakout", "flow_cluster"]

    # Prefixes registered at generator import time (momentum.py, vol_breakout.py, flow_cluster.py)

    # --- Path overrides (separate from microstructure agent) -----------------

    @property
    def root(self) -> Path:
        return ROOT

    @property
    def state_path(self) -> Path:
        return MF_STATE_PATH

    @property
    def queue_path(self) -> Path:
        return ROOT / "data" / "agent_mf" / "hypotheses.json"

    @property
    def stats_path(self) -> Path:
        return MF_STATS_PATH

    # --- Config --------------------------------------------------------------

    def load_config(self) -> dict:
        return load_config()

    # --- Abstract hook implementations ---------------------------------------

    def get_generator(self, name: str):
        """Lazy-import medium-frequency generator functions."""
        try:
            if name == "momentum":
                from agent.generators.medium_freq.momentum import generate
                return generate
            elif name == "vol_breakout":
                from agent.generators.medium_freq.vol_breakout import generate
                return generate
            elif name == "flow_cluster":
                from agent.generators.medium_freq.flow_cluster import generate
                return generate
        except ImportError as e:
            log.debug("MF generator %s not yet implemented: %s", name, e)
        return None

    def create_runner(self, hypothesis: Hypothesis, manifest: dict) -> BaseRunner:
        from agent.mf_runner import MediumFrequencyRunner
        return MediumFrequencyRunner(hypothesis, manifest)

    def pre_execute(self, hypothesis: Hypothesis) -> None:
        """Inject adaptive IC threshold from MF registry."""
        adaptive_min_ic = self._compute_adaptive_ic()
        hypothesis.thresholds.setdefault("min_ic", 0.08)
        if adaptive_min_ic > hypothesis.thresholds["min_ic"]:
            hypothesis.thresholds["min_ic"] = adaptive_min_ic

    def on_fdr_reject(self, hypothesis: Hypothesis) -> None:
        """Remove FDR-rejected signal from MF registry."""
        self._remove_from_registry(hypothesis.id)

    def run_monitor(self) -> None:
        """Check MF registered signals for IC decay and promotion."""
        if not MF_REGISTRY_PATH.exists():
            return
        with open(MF_REGISTRY_PATH) as f:
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
                    log.warning("  MF IC DECAY: %s IC=%.3f < %.3f, day %d/%d",
                                sig["name"][:40], rolling_ic, threshold,
                                days, decay_days_limit)
                    if days >= decay_days_limit:
                        sig["status"] = "retired"
                        sig["retired_reason"] = "ic_decay"
                        sig["retired_date"] = today
                        n_retired += 1
                        log.info("  MF AUTO-RETIRED: %s", sig["name"][:40])
                else:
                    if sig.get("decay_days", 0) > 0:
                        log.info("  MF IC recovered: %s IC=%.3f, resetting decay",
                                 sig["name"][:40], rolling_ic)
                    sig["decay_days"] = 0

        if modified or n_retired > 0:
            with open(MF_REGISTRY_PATH, "w") as f:
                json.dump(registry, f, indent=2)

        if n_retired > 0:
            self.state.set("total_signals_registered",
                          max(0, self.state.get("total_signals_registered", 0) - n_retired))

        n_active = sum(1 for s in registry if s.get("status") != "retired")
        log.info("MF Monitor: %d active signals, %d retired this cycle", n_active, n_retired)

    # --- Domain-specific methods ---------------------------------------------

    @staticmethod
    def _remove_from_registry(hypothesis_id: str) -> None:
        """Remove a signal from the MF registry by its hypothesis_id."""
        if not MF_REGISTRY_PATH.exists():
            return
        with open(MF_REGISTRY_PATH) as f:
            registry = json.load(f)
        registry = [s for s in registry if s.get("hypothesis_id") != hypothesis_id]
        with open(MF_REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)

    def _compute_adaptive_ic(self) -> float:
        """Compute adaptive IC threshold: max(floor, median(registry_ic) * 0.8)."""
        floor_ic = self.config.get("gates", {}).get("min_ic", 0.08)
        if not MF_REGISTRY_PATH.exists():
            return floor_ic
        with open(MF_REGISTRY_PATH) as f:
            registry = json.load(f)
        ics = [s.get("expected_ic", 0) for s in registry
               if s.get("status") != "retired"]
        if not ics:
            return floor_ic
        ics.sort()
        median_ic = ics[len(ics) // 2]
        adaptive = max(floor_ic, median_ic * 0.8)
        log.info("MF Adaptive IC: median=%.3f -> threshold=%.3f (floor=%.2f)",
                 median_ic, adaptive, floor_ic)
        return adaptive

    def _compute_rolling_ic(self, sig: dict) -> float | None:
        """Compute rolling IC for an MF signal on the latest data.

        Loads the latest Parquet data, resamples to 5min bars, and computes
        Spearman IC between the signal feature and N-bar forward returns.
        """
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            return None

        from agent.runner import _load_feature_data, _extract_gated_signal

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
        gate = sig.get("regime_gate")
        symbol = sig.get("symbols", ["BTC"])[0]
        horizon_s = sig.get("horizon_s", 300.0)

        for date in reversed(dates[-2:]):
            data_dir = f"data/features/{date}"
            df = _load_feature_data(data_dir, symbol)
            if df is None or len(df) < 500:
                continue

            # Resample to 5min bars
            try:
                from cluster_pipeline.preprocess import aggregate_bars
                bars = aggregate_bars(df, "5min")
            except (ImportError, Exception):
                continue

            # Find the signal feature column (may have aggregation suffix)
            feat_col = None
            for suffix in ("_last", "_mean", ""):
                candidate = f"{feature}{suffix}" if suffix else feature
                if candidate in bars.columns:
                    feat_col = candidate
                    break
            if feat_col is None:
                continue

            signal_vals = bars[feat_col].to_numpy(dtype=float)

            # Forward returns at bar horizon
            horizon_bars = max(1, int(horizon_s / 300))
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
            if valid.sum() < 50:
                continue

            ic = pd.Series(signal_vals[valid]).corr(
                pd.Series(fwd[valid]), method="spearman",
            )
            return float(ic) if not np.isnan(ic) else None

        return None

    def print_report(self) -> None:
        """Print a full summary of the MF agent."""
        s = self.state

        print("=" * 60)
        print("  NAT Medium-Frequency Agent Report")
        print("=" * 60)

        print(f"\nPhase:       {s.phase.value}")
        print(f"Cycles:      {s.get('cycle_count', 0)}")
        print(f"Tested:      {s.get('total_hypotheses_tested', 0)} hypotheses")
        print(f"Registered:  {s.get('total_signals_registered', 0)} signals")
        print(f"Queue depth: {self.queue.depth}")
        print(f"Graveyard:   {len(self.queue.graveyard)}")

        print(f"\n{'─' * 60}")
        print("MF REGISTRY")
        print(f"{'─' * 60}")
        if MF_REGISTRY_PATH.exists():
            with open(MF_REGISTRY_PATH) as f:
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
            print(f"  {name:15s}  attempts={gs.attempts:3d}  "
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

    parser = argparse.ArgumentParser(description="NAT Medium-Frequency Agent")
    parser.add_argument("action", choices=["start", "status", "once", "queue",
                                            "registry", "graveyard", "report"],
                        help="Action to perform")
    args = parser.parse_args()

    agent = MediumFrequencyAgent()

    if args.action == "start":
        agent.run()
    elif args.action == "once":
        agent.run_cycle()
    elif args.action == "status":
        agent.print_status()
    elif args.action == "queue":
        for h in agent.queue.peek(20):
            print(f"  [{h.priority:6.3f}] {h.id}  {h.generator:12s}  {h.claim[:60]}")
    elif args.action == "registry":
        if MF_REGISTRY_PATH.exists():
            with open(MF_REGISTRY_PATH) as f:
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
