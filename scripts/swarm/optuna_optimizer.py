"""
Optuna-based optimizer for NAT algorithm parameters (Tier 3).

Replaces random/grid search from Tier 2 with intelligent Bayesian optimization:
- CMA-ES for correlated 35D continuous space (covariance adaptation)
- TPE for mixed continuous/categorical parameters
- NSGA-II for multi-objective (Sharpe max, drawdown min, IC max)

Walk-forward train/test split ensures out-of-sample evaluation.
Guard rails prevent overfit configs from being promoted.

Usage:
    python scripts/swarm/optuna_optimizer.py optimize \
        --study my_study --trials 500 --sampler cma

    # Multi-objective with Pareto front:
    python scripts/swarm/optuna_optimizer.py optimize \
        --study pareto --trials 1000 --sampler nsga2

    # Distributed: multiple machines against same PostgreSQL study:
    python scripts/swarm/optuna_optimizer.py optimize \
        --study distributed --trials 200 \
        --storage postgresql://nat:pass@host:5432/optuna
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import toml
from optuna.pruners import MedianPruner
from optuna.samplers import CmaEsSampler, NSGAIISampler, TPESampler

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Defaults ───────────────────────────────────────────────────────

DEFAULT_BASE_CONFIG = "config/algorithms.toml"
DEFAULT_RANGES_CONFIG = "config/swarm_ranges.toml"
DEFAULT_DATA_DIR = "data/features"
DEFAULT_STORAGE = "sqlite:///data/swarm/optuna.db"

# Hard constraints (spec 3_2)
MIN_SIGNAL_COUNT_PER_DAY = 50
MAX_TURNOVER_PER_DAY = 100

# Overfit thresholds (spec 3_3)
OVERFIT_WARN_RATIO = 2.0
OVERFIT_REJECT_RATIO = 3.0

SAMPLERS = {
    "cma": lambda seed: CmaEsSampler(seed=seed),
    "tpe": lambda seed: TPESampler(seed=seed),
    "nsga2": lambda seed: NSGAIISampler(seed=seed),
}


class NATOptimizer:
    """Optuna optimizer with walk-forward OOS evaluation and guard rails.

    Wraps ConfigGenerator (parameter suggestion) and Evaluator (fitness)
    into an Optuna study. Algorithms run on the full data window for proper
    warmup, but fitness is measured only on the OOS portion.
    """

    def __init__(
        self,
        study_name: str = "nat_evolve",
        storage: str = DEFAULT_STORAGE,
        sampler: str = "cma",
        seed: int = 42,
        base_config: str = DEFAULT_BASE_CONFIG,
        ranges_config: str = DEFAULT_RANGES_CONFIG,
        data_dir: str = DEFAULT_DATA_DIR,
        eval_hours: int = 720,
        train_frac: float = 2 / 3,
        symbol: str = "BTC",
        guard_rails: bool = True,
    ):
        if sampler not in SAMPLERS:
            raise ValueError(
                f"Unknown sampler '{sampler}'. Choose from: {list(SAMPLERS)}"
            )

        self.base_config = base_config
        self.ranges_config = ranges_config
        self.data_dir = data_dir
        self.eval_hours = eval_hours
        self.train_frac = train_frac
        self.symbol = symbol
        self.guard_rails = guard_rails
        self.multi_objective = sampler == "nsga2"

        # Ensure storage directory exists for SQLite
        if storage.startswith("sqlite:///"):
            Path(storage.replace("sqlite:///", "")).parent.mkdir(
                parents=True, exist_ok=True,
            )

        sampler_obj = SAMPLERS[sampler](seed)
        pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=3)

        if self.multi_objective:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler_obj,
                pruner=pruner,
                directions=["maximize", "minimize", "maximize"],
                load_if_exists=True,
            )
        else:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler_obj,
                pruner=pruner,
                direction="maximize",
                load_if_exists=True,
            )

    @classmethod
    def from_study(
        cls,
        study_name: str,
        storage: str = DEFAULT_STORAGE,
        base_config: str = DEFAULT_BASE_CONFIG,
        ranges_config: str = DEFAULT_RANGES_CONFIG,
    ) -> "NATOptimizer":
        """Load an existing study for read operations (status, best, export)."""
        instance = object.__new__(cls)
        instance.study = optuna.load_study(
            study_name=study_name, storage=storage,
        )
        instance.multi_objective = len(instance.study.directions) > 1
        instance.base_config = base_config
        instance.ranges_config = ranges_config
        instance.data_dir = None
        instance.eval_hours = None
        instance.train_frac = None
        instance.symbol = None
        instance.guard_rails = None
        return instance

    # ── Objective ──────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial):
        """Walk-forward OOS evaluation with guard rails.

        1. Suggest config from trial (via ConfigGenerator)
        2. Run algorithms on full data (warmup on train period)
        3. Compute fitness on OOS portion only
        4. Apply hard constraints and overfit detection
        """
        try:
            from swarm.config_generator import ConfigGenerator, config_hash
            from swarm.evaluator import Evaluator
        except ImportError:
            from .config_generator import ConfigGenerator, config_hash
            from .evaluator import Evaluator

        worst = (0.0, 1.0, 0.0) if self.multi_objective else 0.0

        # 1. Generate config from trial
        gen = ConfigGenerator(self.base_config, self.ranges_config)
        config = gen.generate_from_optuna(trial)
        cfg_hash = config_hash(config)
        trial.set_user_attr("config_hash", cfg_hash)

        # Write config for Evaluator
        config_dir = Path("data/swarm/optuna_configs")
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"trial_{trial.number:06d}.toml"
        config["_meta"] = {"config_hash": cfg_hash, "trial": trial.number}
        with open(config_path, "w") as f:
            toml.dump(config, f)

        # 2. Load data
        evaluator = Evaluator(
            str(config_path), self.data_dir, "data/swarm/results.db",
        )
        try:
            df = evaluator.load_data(self.symbol, self.eval_hours)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Trial %d: data load failed: %s", trial.number, e)
            return worst

        if len(df) < 1000:
            logger.warning(
                "Trial %d: insufficient data (%d rows)", trial.number, len(df),
            )
            return worst

        # 3. Run algorithms on full data (proper warmup on train period)
        algo_results = evaluator.run_algorithms(df)
        if not algo_results:
            logger.warning(
                "Trial %d: no algorithms produced output", trial.number,
            )
            return worst

        ensemble_df = evaluator.run_ensemble(algo_results, df)

        # 4. Walk-forward split: train (warmup) | test (OOS)
        split_idx = int(len(df) * self.train_frac)
        oos_base = df.iloc[split_idx:].reset_index(drop=True)
        oos_ens = ensemble_df.iloc[split_idx:].reset_index(drop=True)
        oos = evaluator.compute_fitness(oos_ens, oos_base)

        sharpe = oos.get("sharpe", float("nan"))
        dd = oos.get("max_drawdown", float("nan"))
        ic = oos.get("mean_ic", float("nan"))
        sig_count = oos.get("signal_count", 0.0)
        turnover = oos.get("turnover", 0.0)

        # Store all OOS metrics
        for k, v in oos.items():
            trial.set_user_attr(f"oos_{k}", v)
        trial.set_user_attr("n_rows_total", len(df))
        trial.set_user_attr("n_rows_oos", len(oos_base))

        # 5. Hard constraints
        if math.isnan(sharpe) or sharpe <= 0:
            trial.set_user_attr("reject_reason", "non_positive_sharpe")
            return worst
        if sig_count < MIN_SIGNAL_COUNT_PER_DAY:
            trial.set_user_attr("reject_reason", "low_signal_count")
            return worst
        if turnover > MAX_TURNOVER_PER_DAY:
            trial.set_user_attr("reject_reason", "high_turnover")
            return worst

        # 6. Overfit detection (IS vs OOS Sharpe ratio)
        if self.guard_rails:
            is_base = df.iloc[:split_idx].reset_index(drop=True)
            is_ens = ensemble_df.iloc[:split_idx].reset_index(drop=True)
            is_fitness = evaluator.compute_fitness(is_ens, is_base)
            is_sharpe = is_fitness.get("sharpe", float("nan"))
            trial.set_user_attr("is_sharpe", is_sharpe)

            if not math.isnan(is_sharpe) and is_sharpe > 0 and sharpe > 0:
                ratio = is_sharpe / sharpe
                trial.set_user_attr("overfit_ratio", round(ratio, 2))
                if ratio > OVERFIT_REJECT_RATIO:
                    trial.set_user_attr("overfit_flag", True)
                    penalty = (ratio - OVERFIT_WARN_RATIO) * 0.1
                    sharpe *= max(0.5, 1.0 - penalty)
                    trial.set_user_attr("penalized_sharpe", round(sharpe, 4))

        logger.info(
            "Trial %d: Sharpe=%.4f DD=%.4f IC=%.4f Sig=%.0f/d",
            trial.number, sharpe, dd, ic, sig_count,
        )

        if self.multi_objective:
            return (sharpe, dd, ic)
        return sharpe

    # ── Public API ─────────────────────────────────────────────────

    def optimize(
        self,
        n_trials: int = 500,
        n_jobs: int = 1,
        timeout: Optional[int] = None,
    ):
        """Run optimization loop.

        Args:
            n_trials: Number of trials to run.
            n_jobs: Parallel trial threads (1 = sequential).
                    For true parallelism, run multiple processes against
                    the same PostgreSQL-backed study.
            timeout: Optional timeout in seconds.
        """
        logger.info(
            "Starting: %d trials, %d jobs, sampler=%s, symbol=%s, "
            "eval=%dh (train=%.0f%%)",
            n_trials, n_jobs, type(self.study.sampler).__name__,
            self.symbol, self.eval_hours, self.train_frac * 100,
        )

        t0 = time.time()
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True,
        )
        elapsed = time.time() - t0

        n_complete = sum(
            1 for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        logger.info(
            "Done: %d/%d complete in %.0fs (%.1f trials/min)",
            n_complete, len(self.study.trials), elapsed,
            n_complete / (elapsed / 60) if elapsed > 0 else 0,
        )

    def status(self) -> dict:
        """Current study status."""
        trials = self.study.trials
        if not trials:
            return {
                "study_name": self.study.study_name,
                "status": "no_trials",
                "total": 0,
            }

        complete = [
            t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned = [
            t for t in trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        failed = [
            t for t in trials if t.state == optuna.trial.TrialState.FAIL
        ]

        result = {
            "study_name": self.study.study_name,
            "sampler": type(self.study.sampler).__name__,
            "multi_objective": self.multi_objective,
            "total": len(trials),
            "complete": len(complete),
            "pruned": len(pruned),
            "failed": len(failed),
            "overfit_flagged": sum(
                1 for t in complete if t.user_attrs.get("overfit_flag")
            ),
            "rejected": sum(
                1 for t in complete if t.user_attrs.get("reject_reason")
            ),
        }

        if self.multi_objective and complete:
            pareto = self.study.best_trials
            result["pareto_size"] = len(pareto)
            if pareto:
                result["best_sharpe"] = round(
                    max(t.values[0] for t in pareto), 4,
                )
        elif complete:
            sharpes = [t.value for t in complete if t.value is not None]
            if sharpes:
                result["best_sharpe"] = round(max(sharpes), 4)
                result["best_trial"] = self.study.best_trial.number
                result["avg_sharpe"] = round(float(np.mean(sharpes)), 4)
                result["std_sharpe"] = round(float(np.std(sharpes)), 4)

        return result

    def best_configs(self, top_n: int = 5) -> list[dict]:
        """Top configs sorted by Sharpe."""
        if self.multi_objective:
            trials = sorted(
                self.study.best_trials,
                key=lambda t: t.values[0],
                reverse=True,
            )[:top_n]
        else:
            trials = sorted(
                [
                    t for t in self.study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None
                ],
                key=lambda t: t.value,
                reverse=True,
            )[:top_n]

        results = []
        for t in trials:
            entry = {"trial": t.number, "params": t.params}
            if self.multi_objective:
                entry.update(
                    sharpe=round(t.values[0], 4),
                    drawdown=round(t.values[1], 6),
                    ic=round(t.values[2], 6),
                )
            else:
                entry["sharpe"] = round(t.value, 4)
            for k in ("oos_max_drawdown", "oos_mean_ic", "oos_signal_count",
                       "overfit_ratio", "overfit_flag"):
                if k in t.user_attrs:
                    entry[k] = t.user_attrs[k]
            results.append(entry)

        return results

    def pareto_front(self) -> list[dict]:
        """Pareto-optimal trials (multi-objective only)."""
        if not self.multi_objective:
            raise ValueError("pareto_front() requires NSGA-II sampler")
        return [
            {
                "trial": t.number,
                "sharpe": round(t.values[0], 4),
                "drawdown": round(t.values[1], 6),
                "ic": round(t.values[2], 6),
                "overfit_ratio": t.user_attrs.get("overfit_ratio"),
            }
            for t in self.study.best_trials
        ]

    def export_best(self, output_path: str) -> str:
        """Export best config as usable TOML file."""
        if self.multi_objective:
            trials = self.study.best_trials
            if not trials:
                raise ValueError("No completed trials")
            best = max(trials, key=lambda t: t.values[0])
        else:
            best = self.study.best_trial

        config = self._params_to_config(best.params)
        config["_meta"] = {
            "source": "optuna",
            "study": self.study.study_name,
            "trial": best.number,
        }
        if self.multi_objective:
            config["_meta"].update(
                sharpe=best.values[0],
                drawdown=best.values[1],
                ic=best.values[2],
            )
        else:
            config["_meta"]["sharpe"] = best.value

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            toml.dump(config, f)

        logger.info(
            "Best config (trial %d) exported to %s", best.number, output_path,
        )
        return output_path

    def _params_to_config(self, params: dict) -> dict:
        """Reconstruct nested config from flat Optuna param names.

        Optuna params are named 'section.param' (e.g. 'jump_detector.z_threshold').
        This rebuilds the nested TOML structure on top of the base config.
        """
        config = copy.deepcopy(toml.load(self.base_config))
        for full_name, value in params.items():
            parts = full_name.split(".", 1)
            if len(parts) == 2:
                section, param = parts
                if section not in config:
                    config[section] = {}
                config[section][param] = value
            else:
                config[full_name] = value
        return config


# ── Standalone utilities ───────────────────────────────────────────


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    sharpe_std: float = 1.0,
) -> float:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    Adjusts observed Sharpe for multiple-testing bias — the more configs
    tried, the more likely a high Sharpe is due to chance.

    Returns the DSR in [0, 1]: the probability the edge is real after the
    adjustment. HIGHER is better; DSR < 0.95 -> Sharpe likely noise.

    NOTE: not currently wired into the objective/guard rails — exercised only
    by tests. Parallel to scripts/backtest/walk_forward.compute_deflated_sharpe
    (the implementation gated at G4).
    """
    from scipy.stats import norm

    if n_trials < 2 or sharpe_std <= 0:
        return 0.0
    expected_max = sharpe_std * (
        (1 - 0.5772) * norm.ppf(1 - 1 / n_trials)
        + 0.5772 * norm.ppf(1 - 1 / (n_trials * np.e))
    )
    return float(norm.cdf((sharpe - expected_max) / sharpe_std))


# ── CLI ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="NAT Optuna optimizer — evolutionary config search",
    )
    sub = parser.add_subparsers(dest="command")

    # optimize
    p = sub.add_parser("optimize", help="Run optimization")
    p.add_argument("--study", default="nat_evolve", help="Study name")
    p.add_argument("--storage", default=DEFAULT_STORAGE, help="Optuna storage URL")
    p.add_argument("--sampler", choices=list(SAMPLERS), default="cma")
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel threads (for true parallelism, use PostgreSQL + multiple processes)")
    p.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symbol", default="BTC")
    p.add_argument("--hours", type=int, default=720,
                   help="Total eval window in hours (default 720 = 30 days)")
    p.add_argument("--train-frac", type=float, default=2 / 3,
                   help="Fraction of data for train/warmup (rest is OOS)")
    p.add_argument("--no-guard-rails", action="store_true",
                   help="Disable overfit detection")
    p.add_argument("--base", default=DEFAULT_BASE_CONFIG)
    p.add_argument("--ranges", default=DEFAULT_RANGES_CONFIG)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)

    # status
    p = sub.add_parser("status", help="Study status")
    p.add_argument("--study", default="nat_evolve")
    p.add_argument("--storage", default=DEFAULT_STORAGE)

    # best
    p = sub.add_parser("best", help="Top configs")
    p.add_argument("--study", default="nat_evolve")
    p.add_argument("--storage", default=DEFAULT_STORAGE)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--json", action="store_true")

    # pareto
    p = sub.add_parser("pareto", help="Pareto front (NSGA-II studies)")
    p.add_argument("--study", default="nat_evolve")
    p.add_argument("--storage", default=DEFAULT_STORAGE)
    p.add_argument("--json", action="store_true")

    # export
    p = sub.add_parser("export", help="Export best config as TOML")
    p.add_argument("--study", default="nat_evolve")
    p.add_argument("--storage", default=DEFAULT_STORAGE)
    p.add_argument("--output", default="config/evolved_algorithms.toml")
    p.add_argument("--base", default=DEFAULT_BASE_CONFIG)
    p.add_argument("--ranges", default=DEFAULT_RANGES_CONFIG)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "optimize":
        opt = NATOptimizer(
            study_name=args.study,
            storage=args.storage,
            sampler=args.sampler,
            seed=args.seed,
            base_config=args.base,
            ranges_config=args.ranges,
            data_dir=args.data_dir,
            eval_hours=args.hours,
            train_frac=args.train_frac,
            symbol=args.symbol,
            guard_rails=not args.no_guard_rails,
        )
        opt.optimize(
            n_trials=args.trials, n_jobs=args.jobs, timeout=args.timeout,
        )
        print(json.dumps(opt.status(), indent=2))

    elif args.command == "status":
        try:
            opt = NATOptimizer.from_study(args.study, args.storage)
        except KeyError:
            print(json.dumps({"error": f"Study '{args.study}' not found"}))
            sys.exit(1)
        print(json.dumps(opt.status(), indent=2))

    elif args.command == "best":
        try:
            opt = NATOptimizer.from_study(args.study, args.storage)
        except KeyError:
            print(f"Study '{args.study}' not found")
            sys.exit(1)
        configs = opt.best_configs(top_n=args.top)
        if args.json:
            print(json.dumps(configs, indent=2, default=str))
        else:
            for i, c in enumerate(configs):
                print(f"\n#{i + 1} Trial {c['trial']}: Sharpe={c['sharpe']}")
                if "drawdown" in c:
                    print(f"  DD={c['drawdown']:.4f} IC={c['ic']:.4f}")
                if "overfit_ratio" in c:
                    print(f"  Overfit ratio: {c['overfit_ratio']:.1f}")

    elif args.command == "pareto":
        try:
            opt = NATOptimizer.from_study(args.study, args.storage)
        except KeyError:
            print(f"Study '{args.study}' not found")
            sys.exit(1)
        if not opt.multi_objective:
            print("Error: study is not multi-objective. "
                  "Use --sampler nsga2 when optimizing.")
            sys.exit(1)
        front = opt.pareto_front()
        if args.json:
            print(json.dumps(front, indent=2, default=str))
        else:
            print(f"Pareto front: {len(front)} non-dominated solutions\n")
            fmt = "{:>3} {:>6} {:>8} {:>8} {:>8} {:>8}"
            print(fmt.format("#", "Trial", "Sharpe", "DD", "IC", "Overfit"))
            print("-" * 50)
            for i, p in enumerate(
                sorted(front, key=lambda x: x["sharpe"], reverse=True)
            ):
                of = p.get("overfit_ratio")
                of_s = f"{of:.1f}" if isinstance(of, (int, float)) else "-"
                print(fmt.format(
                    i + 1, p["trial"],
                    f"{p['sharpe']:.4f}", f"{p['drawdown']:.4f}",
                    f"{p['ic']:.4f}", of_s,
                ))

    elif args.command == "export":
        try:
            opt = NATOptimizer.from_study(
                args.study, args.storage,
                base_config=args.base, ranges_config=args.ranges,
            )
        except KeyError:
            print(f"Study '{args.study}' not found")
            sys.exit(1)
        path = opt.export_best(args.output)
        print(f"Exported to {path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
