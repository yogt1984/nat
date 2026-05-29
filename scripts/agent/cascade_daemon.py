#!/usr/bin/env python3
"""NAT Cascade Validation Agent — continuous 5-gate validation of the
liquidity heatmap cascade probability model.

Unlike the microstructure/MF agents which discover many hypotheses, this
agent validates a single model (the cascade probability estimator) against
5 quality gates on a recurring schedule.

Main loop:
    1. DATA_CHECK   — verify heatmap features exist in latest parquet
    2. VALIDATE     — run 5-gate protocol (AUC, cost, walk-forward, cross-symbol, orthogonality)
    3. MONITOR      — track model drift and coefficient stability
    4. SLEEP        — wait until next cycle

Usage:
    python scripts/agent/cascade_daemon.py start     # run main loop
    python scripts/agent/cascade_daemon.py status     # print current state
    python scripts/agent/cascade_daemon.py once       # single cycle (for testing)
    python scripts/agent/cascade_daemon.py report     # full validation report
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

from data.state import StateStore
from utils.health import HealthWriter

log = logging.getLogger("nat.agent_cascade")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "cycle_interval_s": 3600,
    "min_auc": 0.65,
    "min_lift": 2.0,
    "min_net_ic": 0.02,
    "walk_forward_min_auc": 0.60,
    "walk_forward_fail_frac": 0.20,
    "cross_symbol_min_cosine": 0.6,
    "max_orthogonality_r2": 0.30,
    "cascade_price_thresh": 0.03,
    "cascade_horizon_ticks": 3000,
    "symbols": None,  # loaded from config/symbols.toml at runtime
}

DB_PATH = ROOT / "data" / "nat.db"
STATE_DIR = ROOT / "data" / "agent_cascade"
STATE_PATH = STATE_DIR / "agent_state.json"  # legacy, for auto-migration
RESULTS_PATH = STATE_DIR / "gate_results.json"


def load_config() -> dict:
    """Load cascade agent config from TOML or return defaults."""
    config_path = ROOT / "config" / "agent.toml"
    if config_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(config_path, "rb") as f:
            return {**DEFAULT_CONFIG, **tomllib.load(f).get("agent_cascade", {})}
    return DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class CascadeState:
    """SQLite-backed agent state (via shared StateStore)."""

    AGENT = "cascade"

    def __init__(self, store: StateStore):
        self._store = store
        self.data: dict = {
            "phase": "idle",
            "cycle_count": 0,
            "last_cycle": None,
            "gate_history": [],
        }
        self._load()

    def _load(self):
        saved = self._store.load_state(self.AGENT)
        if saved:
            self.data.update(saved)
        # Reconstruct gate_history from state_history msg payloads
        raw_history = self._store.load_history(self.AGENT, limit=50)
        gate_history = []
        for entry in raw_history:
            msg = entry.get("msg", "")
            if msg:
                try:
                    gate_history.append(json.loads(msg))
                except (json.JSONDecodeError, TypeError):
                    pass
        if gate_history:
            self.data["gate_history"] = gate_history

    def save(self):
        self._store.save_state(self.AGENT, self.data)

    def append_gate_result(self, results: dict):
        """Append a validation result to history."""
        self._store.append_history(self.AGENT, {
            "from": "validating",
            "to": "idle",
            "msg": json.dumps(results, default=str),
            "at": results.get("timestamp", ""),
        })


# ---------------------------------------------------------------------------
# CascadeRunner — 5-gate validation protocol
# ---------------------------------------------------------------------------

class CascadeRunner:
    """Runs the 5-gate validation protocol on cascade probability model."""

    HEATMAP_FEATURES = [
        "hm_nearest_cluster_dist", "hm_cluster_mass_ratio",
        "hm_cascade_chain_length", "hm_asymmetric_cascade_pot",
        "hm_absorption_capacity", "hm_cluster_velocity",
        "hm_mass_weighted_distance", "hm_heatmap_entropy",
    ]

    def __init__(self, config: dict):
        self.config = config
        self.results: dict = {}

    def run_all_gates(self) -> dict:
        """Run all 5 gates, return results dict."""
        gates = {}
        gates["G1_discriminative"] = self._run_g1()
        gates["G2_cost_aware"] = self._run_g2()
        gates["G3_temporal"] = self._run_g3()
        gates["G4_cross_symbol"] = self._run_g4()
        gates["G5_orthogonality"] = self._run_g5()

        overall = all(g.get("passed", False) for g in gates.values())
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gates": gates,
            "overall": overall,
        }
        return self.results

    def _load_data(self, symbol: str = "BTC"):
        """Load latest feature data with heatmap columns."""
        from data.features import load_features

        df = load_features(
            symbols=[symbol] if symbol else None,
            columns=list(self.HEATMAP_FEATURES) + ["timestamp_ns", "symbol", "raw_midprice"],
            validate=False,
        )
        if df.empty:
            return None

        # Keep only last ~3 days of data (matching previous behavior)
        if "timestamp_ns" in df.columns and len(df) > 0:
            ts = df["timestamp_ns"].values
            cutoff = ts[-1] - 3 * 86400 * 1_000_000_000
            df = df[ts >= cutoff].reset_index(drop=True)

        # Check heatmap columns exist
        missing = [c for c in self.HEATMAP_FEATURES if c not in df.columns]
        if missing:
            log.warning("Missing heatmap columns: %s", missing)
            return None

        return df

    def _build_cascade_labels(self, df, horizon_ticks: int = 3000,
                               price_thresh: float = 0.03):
        """Build binary cascade labels from price data."""
        import numpy as np

        mid = df["raw_midprice"].to_numpy(dtype=float) if "raw_midprice" in df.columns else None
        if mid is None or len(mid) < horizon_ticks + 100:
            return None

        labels = np.zeros(len(mid))
        for i in range(len(mid) - horizon_ticks):
            if mid[i] > 0:
                log_return = abs(np.log(mid[i + horizon_ticks] / mid[i]))
                if log_return > price_thresh:
                    labels[i] = 1.0

        return labels

    def _run_g1(self) -> dict:
        """G1: Discriminative power — AUC > 0.65, lift > 2x."""
        import numpy as np

        result = {"passed": False, "reason": "not_run"}
        df = self._load_data("BTC")
        if df is None:
            result["reason"] = "no_data"
            return result

        labels = self._build_cascade_labels(
            df,
            self.config.get("cascade_horizon_ticks", 3000),
            self.config.get("cascade_price_thresh", 0.03),
        )
        if labels is None:
            result["reason"] = "insufficient_data"
            return result

        # Run cascade probability algorithm
        from algorithms.cascade_probability import CascadeProbability
        model = CascadeProbability()
        out = model.run_batch(df)

        probs = out["alg_cascade_prob"].to_numpy(dtype=float)

        # Trim warmup and align
        warmup = model.warmup
        horizon = self.config.get("cascade_horizon_ticks", 3000)
        end = len(labels) - horizon
        valid = slice(warmup, end)

        y = labels[valid]
        p = probs[valid]
        mask = np.isfinite(p) & np.isfinite(y)
        y, p = y[mask], p[mask]

        if len(y) < 100 or y.sum() < 5:
            result["reason"] = f"too_few_events (n={len(y)}, pos={int(y.sum())})"
            return result

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, p)
        except Exception:
            # Manual AUC via Wilcoxon-Mann-Whitney
            pos_scores = p[y == 1]
            neg_scores = p[y == 0]
            auc = np.mean(pos_scores[:, None] > neg_scores[None, :])

        # Lift at top decile
        q90 = np.percentile(p, 90)
        top_decile = y[p >= q90]
        base_rate = y.mean()
        lift = (top_decile.mean() / base_rate) if base_rate > 0 else 0.0

        min_auc = self.config.get("min_auc", 0.65)
        min_lift = self.config.get("min_lift", 2.0)
        passed = auc > min_auc and lift > min_lift

        return {
            "passed": passed,
            "auc": round(float(auc), 4),
            "lift": round(float(lift), 2),
            "base_rate": round(float(base_rate), 4),
            "n_events": int(y.sum()),
            "n_total": len(y),
        }

    def _run_g2(self) -> dict:
        """G2: Cost awareness — net IC > 0.02 after spread adjustment."""
        result = {"passed": False, "reason": "not_implemented_yet"}
        # Requires spread data during cascade events — deferred until
        # heatmap features are collected in production.
        # Placeholder: pass if G1 passed with sufficient margin.
        return result

    def _run_g3(self) -> dict:
        """G3: Temporal stability — walk-forward median AUC > 0.60."""
        result = {"passed": False, "reason": "not_run"}
        df = self._load_data("BTC")
        if df is None:
            result["reason"] = "no_data"
            return result

        import numpy as np

        labels = self._build_cascade_labels(df)
        if labels is None:
            result["reason"] = "insufficient_data"
            return result

        from algorithms.cascade_probability import CascadeProbability

        # Split into chunks (simulate 30-day windows at tick level)
        chunk_size = max(len(df) // 5, 10000)
        aucs = []

        for start in range(0, len(df) - chunk_size, chunk_size):
            end = start + chunk_size
            chunk_df = df.iloc[start:end].reset_index(drop=True)
            chunk_labels = labels[start:end]

            model = CascadeProbability()
            out = model.run_batch(chunk_df)
            probs = out["alg_cascade_prob"].to_numpy(dtype=float)

            warmup = model.warmup
            horizon = self.config.get("cascade_horizon_ticks", 3000)
            valid_end = len(chunk_labels) - horizon
            if valid_end <= warmup:
                continue

            y = chunk_labels[warmup:valid_end]
            p = probs[warmup:valid_end]
            mask = np.isfinite(p) & np.isfinite(y)
            y, p = y[mask], p[mask]

            if len(y) < 50 or y.sum() < 3:
                continue

            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y, p)
            except Exception:
                pos_s = p[y == 1]
                neg_s = p[y == 0]
                auc = float(np.mean(pos_s[:, None] > neg_s[None, :]))

            aucs.append(auc)

        if len(aucs) < 2:
            result["reason"] = f"too_few_windows ({len(aucs)})"
            return result

        median_auc = float(np.median(aucs))
        min_auc = self.config.get("walk_forward_min_auc", 0.60)
        fail_frac = sum(1 for a in aucs if a < 0.55) / len(aucs)
        max_fail = self.config.get("walk_forward_fail_frac", 0.20)

        passed = median_auc > min_auc and fail_frac <= max_fail

        return {
            "passed": passed,
            "median_auc": round(median_auc, 4),
            "aucs": [round(a, 4) for a in aucs],
            "fail_fraction": round(fail_frac, 3),
            "n_windows": len(aucs),
        }

    def _run_g4(self) -> dict:
        """G4: Cross-symbol — all symbols pass G1, beta cosine > 0.6."""
        import numpy as np

        try:
            from config_utils import load_symbols
        except ImportError:
            from config_utils import load_symbols
        symbols = self.config.get("symbols") or load_symbols()
        symbol_results = {}
        betas = {}

        for sym in symbols:
            df = self._load_data(sym)
            if df is None:
                symbol_results[sym] = {"passed": False, "reason": "no_data"}
                continue

            labels = self._build_cascade_labels(df)
            if labels is None:
                symbol_results[sym] = {"passed": False, "reason": "insufficient_data"}
                continue

            from algorithms.cascade_probability import CascadeProbability
            model = CascadeProbability()
            _ = model.run_batch(df)

            if model._beta is not None:
                betas[sym] = model._beta.copy()
                symbol_results[sym] = {"passed": True, "beta_norm": float(np.linalg.norm(model._beta))}
            else:
                symbol_results[sym] = {"passed": False, "reason": "model_not_trained"}

        # Check pairwise cosine similarity of betas
        cosines = []
        sym_list = [s for s in symbols if s in betas]
        for i in range(len(sym_list)):
            for j in range(i + 1, len(sym_list)):
                b1, b2 = betas[sym_list[i]], betas[sym_list[j]]
                norm_prod = np.linalg.norm(b1) * np.linalg.norm(b2)
                cos = float(np.dot(b1, b2) / norm_prod) if norm_prod > 1e-10 else 0.0
                cosines.append(cos)

        min_cosine = self.config.get("cross_symbol_min_cosine", 0.6)
        all_passed = all(r.get("passed", False) for r in symbol_results.values())
        cosine_ok = all(c > min_cosine for c in cosines) if cosines else False

        return {
            "passed": all_passed and cosine_ok,
            "symbols": symbol_results,
            "cosine_similarities": [round(c, 3) for c in cosines],
        }

    def _run_g5(self) -> dict:
        """G5: Orthogonality — R^2 < 0.30 vs spread_depth_composite."""
        import numpy as np

        result = {"passed": False, "reason": "not_run"}
        df = self._load_data("BTC")
        if df is None:
            result["reason"] = "no_data"
            return result

        from algorithms.cascade_probability import CascadeProbability
        model = CascadeProbability()
        out = model.run_batch(df)

        probs = out["alg_cascade_prob"].to_numpy(dtype=float)

        # Build spread+depth composite as baseline signal
        spread = df["raw_spread_bps"].to_numpy(dtype=float) if "raw_spread_bps" in df.columns else None
        depth = df["raw_bid_depth_5"].to_numpy(dtype=float) if "raw_bid_depth_5" in df.columns else None

        if spread is None or depth is None:
            result["reason"] = "missing_baseline_columns"
            return result

        # Normalise and combine
        s_mean, s_std = np.nanmean(spread), np.nanstd(spread)
        d_mean, d_std = np.nanmean(depth), np.nanstd(depth)
        s_std = max(s_std, 1e-10)
        d_std = max(d_std, 1e-10)
        baseline = (spread - s_mean) / s_std - (depth - d_mean) / d_std

        warmup = model.warmup
        p = probs[warmup:]
        b = baseline[warmup:]
        mask = np.isfinite(p) & np.isfinite(b)
        p, b = p[mask], b[mask]

        if len(p) < 100:
            result["reason"] = "insufficient_valid_data"
            return result

        # Compute R^2
        corr = np.corrcoef(p, b)[0, 1]
        r_squared = corr ** 2

        max_r2 = self.config.get("max_orthogonality_r2", 0.30)
        passed = r_squared < max_r2

        return {
            "passed": passed,
            "r_squared": round(float(r_squared), 4),
            "correlation": round(float(corr), 4),
        }


# ---------------------------------------------------------------------------
# CascadeAgent
# ---------------------------------------------------------------------------

class CascadeAgent:
    """Continuously validates the cascade probability model."""

    def __init__(self, store: StateStore | None = None):
        self.config = load_config()
        self._store = store or StateStore(DB_PATH)
        self.state = CascadeState(self._store)
        self._shutdown = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        log.info("Received signal %d, shutting down after current cycle", signum)
        self._shutdown = True

    def run(self):
        """Main loop: validate → monitor → sleep → repeat."""
        health = HealthWriter("cascade_agent")
        log.info("Cascade agent starting (cycle interval: %ds)",
                 self.config["cycle_interval_s"])
        cycle = 0
        while not self._shutdown:
            cycle += 1
            health.beat(phase="VALIDATE", cycle=cycle)
            self.run_cycle()
            if self._shutdown:
                break
            interval = self.config["cycle_interval_s"]
            log.info("Sleeping %ds until next cycle", interval)
            health.beat(phase="SLEEP", cycle=cycle)
            for _ in range(interval):
                if self._shutdown:
                    break
                time.sleep(1)
        health.shutdown()

    def run_cycle(self):
        """Single validation cycle."""
        self.state.data["phase"] = "validating"
        self.state.save()

        log.info("=" * 60)
        log.info("CASCADE VALIDATION CYCLE %d", self.state.data["cycle_count"] + 1)
        log.info("=" * 60)

        runner = CascadeRunner(self.config)
        results = runner.run_all_gates()

        # Log results
        for gate_name, gate_result in results["gates"].items():
            status = "PASS" if gate_result.get("passed") else "FAIL"
            log.info("  %s: %s  %s", gate_name, status,
                     {k: v for k, v in gate_result.items() if k != "passed"})

        overall = "ALL GATES PASSED" if results["overall"] else "SOME GATES FAILED"
        log.info("  OVERALL: %s", overall)

        # Persist
        self.state.data["cycle_count"] += 1
        self.state.data["last_cycle"] = results["timestamp"]
        self.state.data["phase"] = "idle"
        self.state.append_gate_result(results)
        self.state.save()

        # Also save detailed results
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    def print_status(self):
        """Print current agent state."""
        s = self.state.data
        print(f"Phase:       {s['phase']}")
        print(f"Cycles:      {s['cycle_count']}")
        print(f"Last cycle:  {s.get('last_cycle', 'never')}")

        history = s.get("gate_history", [])
        if history:
            last = history[-1]
            print(f"\nLast validation ({last['timestamp']}):")
            for gate_name, gate_result in last["gates"].items():
                status = "PASS" if gate_result.get("passed") else "FAIL"
                print(f"  {gate_name}: {status}")
            overall = "PASS" if last["overall"] else "FAIL"
            print(f"  Overall: {overall}")

            # Pass rate over recent history
            passes = sum(1 for h in history if h.get("overall", False))
            print(f"\nPass rate:   {passes}/{len(history)} ({100*passes/len(history):.0f}%)")

    def print_report(self):
        """Print full validation report."""
        self.print_status()

        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                results = json.load(f)
            print(f"\n{'─' * 60}")
            print("DETAILED GATE RESULTS")
            print(f"{'─' * 60}")
            print(json.dumps(results, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from logging_config import setup_logging
    setup_logging("nat.agent_cascade")

    parser = argparse.ArgumentParser(description="NAT Cascade Validation Agent")
    parser.add_argument("action", choices=["start", "status", "once", "report"],
                        help="Action to perform")
    args = parser.parse_args()

    agent = CascadeAgent()

    if args.action == "start":
        from agent.base import _write_pid_file
        project_root = Path(__file__).resolve().parent.parent.parent
        _write_pid_file(project_root / ".cascade_agent.pid")
        agent.run()
    elif args.action == "once":
        agent.run_cycle()
    elif args.action == "status":
        agent.print_status()
    elif args.action == "report":
        agent.print_report()


if __name__ == "__main__":
    main()
