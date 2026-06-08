"""
Evaluator worker — run algorithms with a given config and score the results.

Each evaluator:
1. Loads Parquet data for a symbol/time window
2. Instantiates algorithms from the config
3. Runs them via the existing AlgorithmRunner + Ensemble
4. Computes fitness metrics (Sharpe, IC, drawdown, signal count, turnover)
5. Writes results to a shared SQLite database

Usage:
    python scripts/swarm/evaluator.py \
        --config data/swarm/configs/config_0001.toml \
        --data-dir data/features \
        --symbol BTC --hours 24

Performance target: ~5s per evaluation (1 day BTC), <500 MB memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import toml

logger = logging.getLogger(__name__)

# ── SQLite schema ───────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash  TEXT NOT NULL,
    config_json  TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    eval_hours   INTEGER NOT NULL,
    sharpe       REAL,
    mean_ic      REAL,
    max_drawdown REAL,
    signal_count REAL,
    turnover     REAL,
    n_rows       INTEGER,
    eval_time_s  REAL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_trials_sharpe ON trials(sharpe DESC);
CREATE INDEX IF NOT EXISTS idx_trials_hash ON trials(config_hash);
"""


class Evaluator:
    """Run algorithms with a config and compute fitness metrics."""

    def __init__(
        self,
        config_path: str,
        data_dir: str,
        db_path: str = "data/swarm/results.db",
    ):
        self.config = toml.load(config_path)
        self.config_path = config_path
        self.data_dir = data_dir
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)

    def load_data(self, symbol: str, hours: int) -> pd.DataFrame:
        """Read Parquet files for the evaluation window."""
        try:
            from .parquet_reader import read_evaluation_data
        except ImportError:
            from swarm.parquet_reader import read_evaluation_data
        return read_evaluation_data(
            self.data_dir,
            symbol=symbol,
            hours=hours,
            max_memory_mb=500.0,
        )

    def run_algorithms(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Run all enabled algorithms via the registry.

        Returns dict mapping algorithm name -> features DataFrame.
        """
        # Import here to avoid heavy load at module level
        from algorithms import get_algorithm, list_algorithms
        from algorithms.runner import AlgorithmRunner

        algo_names = self.config.get("ensemble", {}).get(
            "algorithms",
            ["jump_detector", "optimal_entry", "funding_reversion",
             "surprise_signal", "weighted_ofi"],
        )

        results = {}
        for name in algo_names:
            if name not in list_algorithms():
                logger.warning("Algorithm '%s' not registered, skipping", name)
                continue

            # Pass config kwargs, filtering to accepted constructor params
            kwargs = self.config.get(name, {})
            try:
                import inspect
                sig = inspect.signature(get_algorithm(name).__class__.__init__)
                valid_params = set(sig.parameters.keys()) - {"self"}
                kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
                algo = get_algorithm(name, **kwargs)
                runner = AlgorithmRunner(algo)
                result = runner.run_on_dataframe(df)
                results[name] = result.features_df
                logger.info(
                    "  %s: %d rows, %.2fs",
                    name, result.n_ticks, result.elapsed_s,
                )
            except Exception as e:
                logger.warning("Algorithm '%s' failed: %s", name, e)
                continue

        return results

    def run_ensemble(
        self,
        algo_results: dict[str, pd.DataFrame],
        base_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine algorithm signals via ensemble method."""
        from algorithms.ensemble import Ensemble

        ens_cfg = self.config.get("ensemble", {})
        algo_names = list(algo_results.keys())

        ensemble = Ensemble(
            algorithms=algo_names,
            method=ens_cfg.get("method", "equal_weight"),
            ic_lookback=ens_cfg.get("ic_lookback", 5000),
            regime_column=ens_cfg.get("regime_column", "ent_book_shape"),
        )

        # Compute forward returns for IC-based methods
        forward_returns = None
        if ens_cfg.get("method") in ("ic_weight", "regime_switch"):
            if "raw_midprice" in base_df.columns:
                mid = base_df["raw_midprice"].values
                forward_returns = np.roll(mid, -1) / mid - 1
                forward_returns[-1] = np.nan

        return ensemble.combine(
            algo_results,
            base_df=base_df,
            forward_returns=forward_returns,
        )

    def compute_fitness(
        self,
        ensemble_df: pd.DataFrame,
        base_df: pd.DataFrame,
    ) -> dict:
        """Compute fitness metrics from ensemble signal.

        Returns dict with: sharpe, mean_ic, max_drawdown, signal_count, turnover.
        """
        signal = ensemble_df["ens_signal"].values
        n = len(signal)

        # Forward returns from midprice
        if "raw_midprice" not in base_df.columns:
            return self._nan_fitness()

        mid = base_df["raw_midprice"].values
        fwd_ret = np.roll(mid, -1) / mid - 1
        fwd_ret[-1] = np.nan

        # Strategy returns: signal * forward_return (long/short proportional)
        mask = np.isfinite(signal) & np.isfinite(fwd_ret)
        strat_ret = np.where(mask, signal * fwd_ret, np.nan)

        # Sharpe ratio (annualized, assuming 100ms ticks)
        valid_ret = strat_ret[np.isfinite(strat_ret)]
        if len(valid_ret) < 100:
            return self._nan_fitness()

        ticks_per_year = 365.25 * 24 * 3600 * 10  # 100ms ticks
        mean_r = np.mean(valid_ret)
        std_r = np.std(valid_ret)
        sharpe = (mean_r / std_r * np.sqrt(ticks_per_year)) if std_r > 1e-15 else 0.0

        # Mean IC (rank correlation of signal vs forward return, rolling 1000-tick windows)
        ic_values = []
        window = 1000
        step = window  # non-overlapping
        for i in range(0, n - window, step):
            s = signal[i:i + window]
            r = fwd_ret[i:i + window]
            m = np.isfinite(s) & np.isfinite(r)
            if m.sum() > 50:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(s[m], r[m])
                if np.isfinite(ic):
                    ic_values.append(ic)
        mean_ic = np.mean(ic_values) if ic_values else 0.0

        # Max drawdown on cumulative strategy returns
        cum_ret = np.nancumsum(valid_ret)
        running_max = np.maximum.accumulate(cum_ret)
        drawdown = running_max - cum_ret
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Signal count per day
        threshold = np.nanstd(signal) * 0.5 if np.nanstd(signal) > 0 else 0
        active = np.sum(np.abs(signal[np.isfinite(signal)]) > threshold)
        hours = n / (10 * 3600)  # ticks to hours
        days = max(hours / 24, 1 / 24)
        signal_count = active / days

        # Turnover: average absolute position change per day
        pos_changes = np.abs(np.diff(signal[np.isfinite(signal)]))
        turnover = np.sum(pos_changes) / days if len(pos_changes) > 0 else 0.0

        return {
            "sharpe": round(float(sharpe), 4),
            "mean_ic": round(float(mean_ic), 6),
            "max_drawdown": round(float(max_dd), 6),
            "signal_count": round(float(signal_count), 1),
            "turnover": round(float(turnover), 1),
        }

    def store_result(
        self,
        fitness: dict,
        symbol: str,
        hours: int,
        n_rows: int,
        eval_time: float,
    ):
        """Write trial result to SQLite."""
        cfg_hash = self.config.get("_meta", {}).get(
            "config_hash",
            hashlib.sha256(
                json.dumps(self.config, sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
        )
        cfg_json = json.dumps(
            {k: v for k, v in self.config.items() if k != "_meta"},
            default=str,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO trials
                   (config_hash, config_json, symbol, eval_hours,
                    sharpe, mean_ic, max_drawdown, signal_count, turnover,
                    n_rows, eval_time_s)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cfg_hash, cfg_json, symbol, hours,
                    fitness.get("sharpe"),
                    fitness.get("mean_ic"),
                    fitness.get("max_drawdown"),
                    fitness.get("signal_count"),
                    fitness.get("turnover"),
                    n_rows,
                    round(eval_time, 2),
                ),
            )
        logger.info(
            "Trial stored: hash=%s sharpe=%.4f ic=%.4f dd=%.4f",
            cfg_hash,
            fitness.get("sharpe", 0),
            fitness.get("mean_ic", 0),
            fitness.get("max_drawdown", 0),
        )

    def evaluate(self, symbol: str = "BTC", hours: int = 24) -> dict:
        """Full pipeline: load -> run algos -> ensemble -> fitness -> store."""
        t0 = time.time()

        logger.info("Loading data: %s, %dh", symbol, hours)
        df = self.load_data(symbol, hours)
        n_rows = len(df)

        logger.info("Running algorithms on %d rows...", n_rows)
        algo_results = self.run_algorithms(df)

        if not algo_results:
            logger.error("No algorithms produced output")
            return self._nan_fitness()

        logger.info("Running ensemble (%d algorithms)...", len(algo_results))
        ensemble_df = self.run_ensemble(algo_results, df)

        logger.info("Computing fitness...")
        fitness = self.compute_fitness(ensemble_df, df)

        elapsed = time.time() - t0
        self.store_result(fitness, symbol, hours, n_rows, elapsed)

        fitness["eval_time_s"] = round(elapsed, 2)
        fitness["n_rows"] = n_rows
        return fitness

    @staticmethod
    def _nan_fitness() -> dict:
        return {
            "sharpe": float("nan"),
            "mean_ic": float("nan"),
            "max_drawdown": float("nan"),
            "signal_count": 0.0,
            "turnover": 0.0,
        }


# ── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Swarm evaluator worker")
    parser.add_argument("--config", required=True, help="Config TOML path")
    parser.add_argument("--data-dir", default="data/features", help="Parquet data dir")
    parser.add_argument("--db", default="data/swarm/results.db", help="Results DB path")
    parser.add_argument("--symbol", default="BTC", help="Symbol to evaluate")
    parser.add_argument("--hours", type=int, default=24, help="Hours of data")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ev = Evaluator(args.config, args.data_dir, args.db)
    fitness = ev.evaluate(symbol=args.symbol, hours=args.hours)

    if args.json:
        print(json.dumps(fitness))
    else:
        print(f"Sharpe:      {fitness.get('sharpe', 'N/A')}")
        print(f"Mean IC:     {fitness.get('mean_ic', 'N/A')}")
        print(f"Max DD:      {fitness.get('max_drawdown', 'N/A')}")
        print(f"Signals/day: {fitness.get('signal_count', 'N/A')}")
        print(f"Turnover:    {fitness.get('turnover', 'N/A')}")
        print(f"Eval time:   {fitness.get('eval_time_s', 'N/A')}s")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
