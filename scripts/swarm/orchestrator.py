"""
Swarm orchestrator — coordinate N parallel evaluator workers.

Two execution modes:
  - local: multiprocessing.Pool (simplest, no Docker needed)
  - docker: docker compose --scale evaluator=N

Usage:
    python scripts/swarm/orchestrator.py run --instances 16 --hours 24
    python scripts/swarm/orchestrator.py status
    python scripts/swarm/orchestrator.py results --top 10
    python scripts/swarm/orchestrator.py best --export config/best_algorithms.toml
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import toml

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_BASE_CONFIG = "config/algorithms.toml"
DEFAULT_RANGES_CONFIG = "config/swarm_ranges.toml"
DEFAULT_DATA_DIR = "data/features"
DEFAULT_DB_PATH = "data/swarm/results.db"
DEFAULT_CONFIG_DIR = "data/swarm/configs"


def _run_single_eval(args_tuple):
    """Worker function for ProcessPoolExecutor (must be top-level for pickle)."""
    config_path, data_dir, db_path, symbol, hours = args_tuple
    # Import inside worker to avoid issues with multiprocessing
    try:
        from .evaluator import Evaluator
    except ImportError:
        from swarm.evaluator import Evaluator
    ev = Evaluator(config_path, data_dir, db_path)
    return ev.evaluate(symbol=symbol, hours=hours)


class SwarmOrchestrator:
    """Coordinate config generation, parallel evaluation, and ranking."""

    def __init__(
        self,
        base_config: str = DEFAULT_BASE_CONFIG,
        ranges_config: str = DEFAULT_RANGES_CONFIG,
        data_dir: str = DEFAULT_DATA_DIR,
        db_path: str = DEFAULT_DB_PATH,
        config_dir: str = DEFAULT_CONFIG_DIR,
    ):
        self.base_config = base_config
        self.ranges_config = ranges_config
        self.data_dir = data_dir
        self.db_path = db_path
        self.config_dir = config_dir

    def run(
        self,
        n_instances: int = 16,
        hours: int = 24,
        symbol: str = "BTC",
        seed: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> list[dict]:
        """Full swarm run: generate configs -> evaluate in parallel -> rank.

        Args:
            n_instances: Number of config variants to evaluate.
            hours: Hours of data per evaluation.
            symbol: Symbol to evaluate.
            seed: Random seed for config generation.
            max_workers: Max parallel workers (default: min(n_instances, CPU count)).

        Returns:
            List of fitness dicts, one per config, sorted by Sharpe desc.
        """
        try:
            from .config_generator import ConfigGenerator
        except ImportError:
            from swarm.config_generator import ConfigGenerator

        # 1. Generate configs
        logger.info("Generating %d configs...", n_instances)
        gen = ConfigGenerator(self.base_config, self.ranges_config)
        configs = gen.generate_random(n_instances, seed=seed)
        paths = gen.write_configs(configs, self.config_dir)
        logger.info("Configs written to %s/", self.config_dir)

        # 2. Evaluate in parallel
        if max_workers is None:
            import os
            max_workers = min(n_instances, os.cpu_count() or 4)

        logger.info(
            "Launching %d evaluations (%d parallel workers)...",
            n_instances, max_workers,
        )

        work_items = [
            (str(p), self.data_dir, self.db_path, symbol, hours)
            for p in paths
        ]

        results = []
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_single_eval, item): i
                for i, item in enumerate(work_items)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    fitness = future.result()
                    fitness["config_index"] = idx
                    results.append(fitness)
                    logger.info(
                        "  [%d/%d] config_%04d: Sharpe=%.4f IC=%.4f",
                        len(results), n_instances, idx,
                        fitness.get("sharpe", float("nan")),
                        fitness.get("mean_ic", float("nan")),
                    )
                except Exception as e:
                    logger.error("  [%d/%d] config_%04d FAILED: %s",
                                 len(results) + 1, n_instances, idx, e)

        elapsed = time.time() - t0
        logger.info(
            "Swarm complete: %d/%d succeeded in %.1fs (%.1f eval/min)",
            len(results), n_instances, elapsed,
            len(results) / (elapsed / 60) if elapsed > 0 else 0,
        )

        # 3. Sort by Sharpe
        import math
        results.sort(
            key=lambda r: r.get("sharpe", float("-inf"))
            if not (isinstance(r.get("sharpe"), float) and math.isnan(r.get("sharpe", 0)))
            else float("-inf"),
            reverse=True,
        )

        return results

    def status(self) -> dict:
        """Current swarm status from the results database."""
        if not Path(self.db_path).exists():
            return {"status": "no_runs", "total_trials": 0}

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*), MAX(sharpe), AVG(sharpe), MIN(created_at), MAX(created_at) "
                "FROM trials"
            ).fetchone()

        total, best_sharpe, avg_sharpe, first, last = row
        return {
            "status": "completed" if total > 0 else "no_runs",
            "total_trials": total,
            "best_sharpe": round(best_sharpe, 4) if best_sharpe else None,
            "avg_sharpe": round(avg_sharpe, 4) if avg_sharpe else None,
            "first_trial": first,
            "last_trial": last,
        }

    def results(self, top_n: int = 10) -> list[dict]:
        """Return top configs ranked by Sharpe."""
        if not Path(self.db_path).exists():
            return []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT trial_id, config_hash, symbol, eval_hours,
                          sharpe, mean_ic, max_drawdown, signal_count,
                          turnover, n_rows, eval_time_s, created_at
                   FROM trials
                   ORDER BY sharpe DESC
                   LIMIT ?""",
                (top_n,),
            ).fetchall()

        return [dict(r) for r in rows]

    def export_best(self, output_path: str) -> Optional[str]:
        """Export the best config as a usable TOML file."""
        if not Path(self.db_path).exists():
            logger.error("No results database found")
            return None

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT config_json, sharpe FROM trials ORDER BY sharpe DESC LIMIT 1"
            ).fetchone()

        if not row:
            logger.error("No trials in database")
            return None

        cfg = json.loads(row[0])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            toml.dump(cfg, f)

        logger.info("Best config (Sharpe=%.4f) exported to %s", row[1], output_path)
        return output_path


# ── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Swarm orchestrator")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run a swarm evaluation")
    p_run.add_argument("--instances", type=int, default=16)
    p_run.add_argument("--hours", type=int, default=24)
    p_run.add_argument("--symbol", default="BTC")
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--workers", type=int, default=None)
    p_run.add_argument("--base", default=DEFAULT_BASE_CONFIG)
    p_run.add_argument("--ranges", default=DEFAULT_RANGES_CONFIG)
    p_run.add_argument("--json", action="store_true")

    # status
    sub.add_parser("status", help="Show swarm status")

    # results
    p_res = sub.add_parser("results", help="Show top results")
    p_res.add_argument("--top", type=int, default=10)
    p_res.add_argument("--json", action="store_true")

    # best
    p_best = sub.add_parser("best", help="Export best config")
    p_best.add_argument("--export", default="config/best_algorithms.toml")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    orch = SwarmOrchestrator(
        base_config=getattr(args, "base", DEFAULT_BASE_CONFIG),
        ranges_config=getattr(args, "ranges", DEFAULT_RANGES_CONFIG),
    )

    if args.command == "run":
        results = orch.run(
            n_instances=args.instances,
            hours=args.hours,
            symbol=args.symbol,
            seed=args.seed,
            max_workers=args.workers,
        )
        if getattr(args, "json", False):
            print(json.dumps(results, default=str))
        else:
            print(f"\nTop 5 results:")
            for i, r in enumerate(results[:5]):
                print(
                    f"  #{i+1}: Sharpe={r.get('sharpe', 'N/A'):>8} "
                    f"IC={r.get('mean_ic', 'N/A'):>8} "
                    f"DD={r.get('max_drawdown', 'N/A'):>8}"
                )

    elif args.command == "status":
        s = orch.status()
        print(json.dumps(s, indent=2, default=str))

    elif args.command == "results":
        rows = orch.results(top_n=args.top)
        if getattr(args, "json", False):
            print(json.dumps(rows, default=str))
        else:
            if not rows:
                print("No results yet.")
            else:
                print(f"{'#':>3} {'Hash':>10} {'Sharpe':>8} {'IC':>8} "
                      f"{'MaxDD':>8} {'Sig/d':>7} {'Time':>6}")
                print("-" * 60)
                for i, r in enumerate(rows):
                    print(
                        f"{i+1:>3} {r['config_hash'][:10]:>10} "
                        f"{r['sharpe']:>8.4f} {r['mean_ic']:>8.4f} "
                        f"{r['max_drawdown']:>8.4f} {r['signal_count']:>7.0f} "
                        f"{r['eval_time_s']:>5.1f}s"
                    )

    elif args.command == "best":
        path = orch.export_best(args.export)
        if path:
            print(f"Best config exported to {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
