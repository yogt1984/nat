#!/usr/bin/env python3
"""
Alpha Pipeline Orchestrator

Chains the 9 alpha modules (screener → combiner → position → validation →
regime → multi_freq → portfolio → paper → deployer) with quality gates
between each step. Persists state to JSON for resume-on-interrupt.

Usage:
    python scripts/alpha/alpha_pipeline.py start  [--config config/alpha.toml]
    python scripts/alpha/alpha_pipeline.py resume [--force-gate]
    python scripts/alpha/alpha_pipeline.py status
    python scripts/alpha/alpha_pipeline.py gates
    python scripts/alpha/alpha_pipeline.py run-step N
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import signal
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    IDLE = "IDLE"
    SCREENING = "SCREENING"       # Step 1: screener.py
    COMBINING = "COMBINING"       # Step 2: combiner.py
    SIZING = "SIZING"             # Step 3: position.py
    VALIDATING = "VALIDATING"     # Step 4: adapter.py
    REGIME = "REGIME"             # Step 5: regime_filter.py
    MULTI_FREQ = "MULTI_FREQ"     # Step 6: multi_freq.py
    PORTFOLIO = "PORTFOLIO"       # Step 7: portfolio.py
    PAPER = "PAPER"               # Step 8: paper_trader.py
    DEPLOYING = "DEPLOYING"       # Step 9: deployer.py
    DONE = "DONE"
    GATE_FAILED = "GATE_FAILED"   # Soft stop — quality gate did not pass
    ERROR = "ERROR"


# Ordered processing phases (for iteration and step numbering)
STEP_PHASES = [
    Phase.SCREENING, Phase.COMBINING, Phase.SIZING, Phase.VALIDATING,
    Phase.REGIME, Phase.MULTI_FREQ, Phase.PORTFOLIO, Phase.PAPER, Phase.DEPLOYING,
]

GATE_NAMES = {
    Phase.SCREENING: "G1",
    Phase.COMBINING: "G2",
    Phase.SIZING: "G3",
    Phase.VALIDATING: "G4",
    Phase.REGIME: "G5",
    Phase.MULTI_FREQ: "G6",
    Phase.PORTFOLIO: "G7",
    Phase.PAPER: "G8",
    Phase.DEPLOYING: "G9",
}

STEP_LABELS = {
    Phase.SCREENING: "Screening",
    Phase.COMBINING: "Combining",
    Phase.SIZING: "Position Sizing",
    Phase.VALIDATING: "Walk-Forward Validation",
    Phase.REGIME: "Regime Conditioning",
    Phase.MULTI_FREQ: "Multi-Frequency",
    Phase.PORTFOLIO: "Portfolio Assembly",
    Phase.PAPER: "Paper Trading",
    Phase.DEPLOYING: "Deployment Readiness",
}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(config_path: str = "config/alpha.toml") -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Alpha config not found: {config_path}")

    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class AlphaPipelineState:
    """Persistent pipeline state — survives restarts."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return self._defaults()

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        return {
            "phase": Phase.IDLE.value,
            "started_at": None,
            "finished_at": None,
            "current_step": 0,
            "gates": {},
            "artifacts": {},
            "step_outputs": {},
            "error": None,
            "history": [],
        }

    def save(self) -> None:
        with open(self.state_file, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    def transition(self, phase: Phase, message: str = "") -> None:
        old = self._data["phase"]
        self._data["phase"] = phase.value
        self._data["history"].append({
            "from": old,
            "to": phase.value,
            "at": datetime.now(timezone.utc).isoformat(),
            "message": message,
        })
        if len(self._data["history"]) > 200:
            self._data["history"] = self._data["history"][-100:]
        self.save()

    @property
    def current(self) -> Phase:
        return Phase(self._data["phase"])

    def record_gate(self, name: str, verdict: str, metrics: dict, advice: str = "") -> None:
        self._data["gates"][name] = {
            "verdict": verdict,
            "metrics": metrics,
            "advice": advice,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def set_artifact(self, name: str, path: str) -> None:
        self._data["artifacts"][name] = path
        self.save()

    def get_artifact(self, name: str) -> Optional[str]:
        return self._data.get("artifacts", {}).get(name)

    def set_output(self, name: str, data: dict) -> None:
        self._data["step_outputs"][name] = data
        self.save()

    def get_output(self, name: str) -> Optional[dict]:
        return self._data.get("step_outputs", {}).get(name)

    def reset(self) -> None:
        self._data = self._defaults()
        self.save()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_file: str) -> logging.Logger:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("alpha_pipeline")
    log.setLevel(logging.INFO)

    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        log.addHandler(ch)

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def evaluate_gate(
    gate_name: str,
    pass_count: int,
    total_count: int,
    weak_min: int,
    metrics: dict,
    advice_on_fail: str,
) -> tuple[str, dict, str]:
    """Evaluate a gate with PASS/WEAK/FAIL logic.

    Returns (verdict, metrics, advice).
    """
    if pass_count >= total_count:
        return "PASS", metrics, ""
    elif pass_count >= weak_min:
        return "WEAK", metrics, advice_on_fail
    else:
        return "FAIL", metrics, advice_on_fail


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

# Global shutdown flag for graceful interrupt
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True


def run_pipeline(
    config: Dict[str, Any],
    ps: AlphaPipelineState,
    log: logging.Logger,
    force_gates: bool = False,
) -> None:
    """Execute the alpha pipeline state machine."""

    import numpy as np

    pipe_cfg = config["pipeline"]
    gates_cfg = config["gates"]
    report_dir = Path(pipe_cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path("data/alpha")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    phase = ps.current

    # --- IDLE → SCREENING ---
    if phase == Phase.IDLE:
        ps.set("started_at", datetime.now(timezone.utc).isoformat())
        ps.transition(Phase.SCREENING, "Starting alpha pipeline")
        phase = Phase.SCREENING

    # --- STEP 1: SCREENING ---
    if phase == Phase.SCREENING:
        log.info("=" * 60)
        log.info("  STEP 1: Feature Screening")
        log.info("=" * 60)

        from alpha.screener import screen_features, save_results

        scr_cfg = config["screener"]
        result = screen_features(
            data_dir=pipe_cfg["data_dir"],
            timeframe=pipe_cfg["timeframe"],
            symbols=pipe_cfg["symbols"],
            fdr_alpha=scr_cfg["fdr_alpha"],
            min_ic=scr_cfg["min_ic"],
            min_breakeven_bps=scr_cfg["min_breakeven_bps"],
            price_col=scr_cfg.get("price_col", "raw_midprice"),
        )
        path = save_results(result, report_dir)
        ps.set_artifact("screen", str(path))
        ps.set_output("screen", {
            "n_significant": result.n_significant,
            "n_tested": result.total_tests,
        })

        # G1: significant features
        n_sig = result.n_significant
        verdict, metrics, advice = evaluate_gate(
            "G1", n_sig, gates_cfg["g1_min_significant"],
            weak_min=gates_cfg["g1_weak_significant"],
            metrics={"n_significant": n_sig, "n_tested": result.total_tests},
            advice_on_fail="Lower FDR alpha, try different timeframe, or collect more data.",
        )
        ps.record_gate("G1", verdict, metrics, advice)
        log.info("G1 Screening: %s (%d significant features)", verdict, n_sig)
        if advice:
            log.warning("  Advice: %s", advice)

        if verdict == "FAIL" and not force_gates:
            ps.transition(Phase.GATE_FAILED, f"G1 FAIL: {n_sig} significant features")
            return

        ps.transition(Phase.COMBINING, "G1 passed")
        phase = Phase.COMBINING

    # --- STEP 2: COMBINING ---
    if phase == Phase.COMBINING:
        log.info("=" * 60)
        log.info("  STEP 2: Signal Combination")
        log.info("=" * 60)

        from alpha.combiner import run_combine

        comb_cfg = config["combiner"]
        screen_path = ps.get_artifact("screen") or str(report_dir / "alpha_screen.json")
        combine_output = str(report_dir / "alpha_combine.json")

        signal, comb_result = run_combine(
            screen_path=screen_path,
            data_dir=pipe_cfg["data_dir"],
            symbol=pipe_cfg["primary_symbol"],
            top_n=comb_cfg["top_n"],
            max_corr=comb_cfg["max_corr"],
            timeframe=pipe_cfg["timeframe"],
            method=comb_cfg["method"],
            output=combine_output,
        )
        np.save(str(artifacts_dir / "signal.npy"), signal)
        ps.set_artifact("signal_npy", str(artifacts_dir / "signal.npy"))
        ps.set_artifact("combine", combine_output)
        ps.set_output("combine", {
            "combined_ic": comb_result.combined_ic,
            "max_individual_ic": comb_result.max_individual_ic,
            "combined_turnover": comb_result.combined_turnover,
            "n_features": len(comb_result.features_after_dedup),
        })

        # G2: composite signal quality (3 sub-gates)
        sub_pass = sum([
            comb_result.combined_ic > gates_cfg["g2_ic_ratio"] * comb_result.max_individual_ic,
            comb_result.combined_turnover < gates_cfg["g2_max_turnover_ratio"] * comb_result.avg_individual_turnover,
            comb_result.max_single_corr < gates_cfg["g2_max_single_corr"],
        ])
        verdict, metrics, advice = evaluate_gate(
            "G2", sub_pass, 3, weak_min=2,
            metrics={
                "combined_ic": comb_result.combined_ic,
                "ic_ratio": comb_result.combined_ic / max(comb_result.max_individual_ic, 1e-10),
                "turnover_ratio": comb_result.combined_turnover / max(comb_result.avg_individual_turnover, 1e-10),
                "max_single_corr": comb_result.max_single_corr,
                "sub_pass": sub_pass,
            },
            advice_on_fail="Try ic_weighted method, relax max_corr, or increase top_n.",
        )
        ps.record_gate("G2", verdict, metrics, advice)
        log.info("G2 Combining: %s (%d/3 sub-gates)", verdict, sub_pass)
        if advice:
            log.warning("  Advice: %s", advice)

        if verdict == "FAIL" and not force_gates:
            ps.transition(Phase.GATE_FAILED, f"G2 FAIL: {sub_pass}/3 sub-gates")
            return

        ps.transition(Phase.SIZING, "G2 passed")
        phase = Phase.SIZING

    # --- STEP 3: POSITION SIZING ---
    if phase == Phase.SIZING:
        log.info("=" * 60)
        log.info("  STEP 3: Position Sizing")
        log.info("=" * 60)

        from alpha.position import run_position_sizing

        pos_cfg = config["position"]
        signal = np.load(ps.get_artifact("signal_npy"))
        comb_out = ps.get_output("combine") or {}
        ic_estimate = comb_out.get("combined_ic", 0.03)

        # Compute return_vol from data
        try:
            from cluster_pipeline.loader import load_parquet
            from cluster_pipeline.preprocess import aggregate_bars
            df = load_parquet(
                pipe_cfg["data_dir"],
                start_date=pipe_cfg.get("start_date"),
                end_date=pipe_cfg.get("end_date"),
                max_memory_mb=pipe_cfg.get("max_memory_mb"),
            )
            bars = aggregate_bars(df, timeframe=pipe_cfg["timeframe"])
            price_col = config["screener"].get("price_col", "raw_midprice")
            price_mean_col = f"{price_col}_mean"
            if price_mean_col in bars.columns:
                prices = bars[price_mean_col].to_numpy()
                returns = np.diff(prices) / prices[:-1]
                return_vol = float(np.nanstd(returns))
            else:
                return_vol = 0.001
        except Exception:
            return_vol = 0.001
            log.warning("Could not compute return_vol from data, using default 0.001")

        position, pos_result = run_position_sizing(
            signal=signal,
            ic_estimate=ic_estimate,
            return_vol=return_vol,
            cost_multiplier=pos_cfg["cost_multiplier"],
            scale=pos_cfg["scale"],
            ramp_bars=pos_cfg["ramp_bars"],
            ramp_fraction=pos_cfg["ramp_fraction"],
            bar_minutes=pos_cfg["bar_minutes"],
            output=str(report_dir / "alpha_position.json"),
        )
        np.save(str(artifacts_dir / "position.npy"), position)
        ps.set_artifact("position_npy", str(artifacts_dir / "position.npy"))
        ps.set_artifact("position", str(report_dir / "alpha_position.json"))
        ps.set_output("position", {
            "trade_reduction_pct": pos_result.trade_reduction_pct,
            "mean_holding_hours": pos_result.mean_holding_hours,
        })

        # G3: cost-aware filtering (2 sub-gates)
        sub_pass = sum([
            pos_result.trade_reduction_pct >= gates_cfg["g3_min_trade_reduction_pct"],
            pos_result.mean_holding_hours >= gates_cfg["g3_min_holding_hours"],
        ])
        verdict, metrics, advice = evaluate_gate(
            "G3", sub_pass, 2, weak_min=1,
            metrics={
                "trade_reduction_pct": pos_result.trade_reduction_pct,
                "mean_holding_hours": pos_result.mean_holding_hours,
                "sub_pass": sub_pass,
            },
            advice_on_fail="Increase cost_multiplier or widen entry threshold.",
        )
        ps.record_gate("G3", verdict, metrics, advice)
        log.info("G3 Sizing: %s (%.0f%% reduction, %.1fh hold)", verdict,
                 pos_result.trade_reduction_pct, pos_result.mean_holding_hours)
        if advice:
            log.warning("  Advice: %s", advice)

        if verdict == "FAIL" and not force_gates:
            ps.transition(Phase.GATE_FAILED, f"G3 FAIL: {sub_pass}/2 sub-gates")
            return

        ps.transition(Phase.VALIDATING, "G3 passed")
        phase = Phase.VALIDATING

    # --- STEP 4: WALK-FORWARD VALIDATION ---
    if phase == Phase.VALIDATING:
        log.info("=" * 60)
        log.info("  STEP 4: Walk-Forward Validation")
        log.info("=" * 60)

        from alpha.adapter import run_validation

        val_cfg = config["validation"]
        signal = np.load(ps.get_artifact("signal_npy"))
        screen_out = ps.get_output("screen") or {}
        n_trials = screen_out.get("n_tested", 1998)

        # Load data as polars DataFrame
        try:
            import polars as pl
            from cluster_pipeline.loader import load_parquet
            from cluster_pipeline.preprocess import aggregate_bars
            df_pd = load_parquet(
                pipe_cfg["data_dir"],
                start_date=pipe_cfg.get("start_date"),
                end_date=pipe_cfg.get("end_date"),
                max_memory_mb=pipe_cfg.get("max_memory_mb"),
            )
            bars = aggregate_bars(df_pd, timeframe=pipe_cfg["timeframe"])
            df_pl = pl.from_pandas(bars) if not isinstance(bars, pl.DataFrame) else bars
        except Exception as e:
            log.error("Failed to load data for validation: %s", e)
            ps.transition(Phase.ERROR, f"Data load failed: {e}")
            return

        val_results = run_validation(
            df=df_pl,
            signal=signal,
            n_trials=n_trials,
            entry_threshold=val_cfg["entry_threshold"],
            n_splits=val_cfg["n_splits"],
            embargo_bars=val_cfg["embargo_bars"],
            directions=val_cfg.get("directions", ["long", "short"]),
            output=str(report_dir / "alpha_validation.json"),
        )
        ps.set_artifact("validation", str(report_dir / "alpha_validation.json"))

        # Find best direction result
        best = max(val_results, key=lambda r: r.oos_sharpe) if val_results else None
        if best is not None:
            ps.set_output("validate", {
                "oos_sharpe": best.oos_sharpe,
                "is_sharpe": best.is_sharpe,
                "oos_is_ratio": best.oos_is_ratio,
                "max_drawdown_pct": best.max_drawdown_pct,
                "total_oos_trades": best.total_oos_trades,
                "profit_factor": best.profit_factor,
                "direction": best.direction,
            })

            # G4: 6 sub-gates
            sub_pass = sum([
                best.oos_sharpe >= gates_cfg["g4_min_oos_sharpe"],
                best.oos_is_ratio >= gates_cfg["g4_min_oos_is_ratio"],
                best.deflated_sharpe_p <= gates_cfg["g4_max_deflated_sharpe_p"],
                best.max_drawdown_pct <= gates_cfg["g4_max_drawdown_pct"],
                best.total_oos_trades >= gates_cfg["g4_min_trades"],
                best.profit_factor >= gates_cfg["g4_min_profit_factor"],
            ])
            verdict, metrics, advice = evaluate_gate(
                "G4", sub_pass, 6, weak_min=4,
                metrics={
                    "oos_sharpe": best.oos_sharpe,
                    "oos_is_ratio": best.oos_is_ratio,
                    "deflated_sharpe_p": best.deflated_sharpe_p,
                    "max_drawdown_pct": best.max_drawdown_pct,
                    "total_oos_trades": best.total_oos_trades,
                    "profit_factor": best.profit_factor,
                    "direction": best.direction,
                    "sub_pass": sub_pass,
                },
                advice_on_fail="Try different timeframe, increase data, or relax entry threshold.",
            )
        else:
            verdict, metrics, advice = "FAIL", {"error": "no validation results"}, "Validation produced no results."
            sub_pass = 0

        ps.record_gate("G4", verdict, metrics, advice)
        log.info("G4 Validation: %s (%d/6 sub-gates)", verdict, sub_pass)
        if advice:
            log.warning("  Advice: %s", advice)

        if verdict == "FAIL" and not force_gates:
            ps.transition(Phase.GATE_FAILED, f"G4 FAIL: {sub_pass}/6 sub-gates")
            return

        ps.transition(Phase.REGIME, "G4 passed")
        phase = Phase.REGIME

    # --- STEP 5: REGIME CONDITIONING ---
    if phase == Phase.REGIME:
        log.info("=" * 60)
        log.info("  STEP 5: Regime Conditioning")
        log.info("=" * 60)

        from alpha.regime_filter import run_regime_filter

        reg_cfg = config["regime"]
        screen_path = ps.get_artifact("screen") or str(report_dir / "alpha_screen.json")

        reg_result = run_regime_filter(
            data_dir=pipe_cfg["data_dir"],
            screen_path=screen_path,
            model_path=reg_cfg.get("model_path", "") or None,
            timeframe=pipe_cfg["timeframe"],
            symbol=pipe_cfg["primary_symbol"],
            top_n=reg_cfg["top_n"],
            improvement_threshold=reg_cfg["improvement_threshold"],
            output=str(report_dir / "alpha_regime.json"),
        )
        ps.set_artifact("regime", str(report_dir / "alpha_regime.json"))
        ps.set_output("regime", {
            "n_regimes": reg_result.n_regimes,
            "conditioned_regimes": reg_result.conditioned_regimes,
        })

        # G5: at least one improving regime
        has_improving = reg_result.gate_has_improving_regime
        verdict = "PASS" if has_improving else "FAIL"
        metrics = {
            "n_regimes": reg_result.n_regimes,
            "conditioned_regimes": reg_result.conditioned_regimes,
            "improvement_ratios": reg_result.improvement_ratios,
        }
        advice = "" if has_improving else "No regime shows IC improvement > 1.5x. Try different gate features."
        ps.record_gate("G5", verdict, metrics, advice)
        log.info("G5 Regime: %s (%d conditioned regimes)", verdict, len(reg_result.conditioned_regimes))
        if advice:
            log.warning("  Advice: %s", advice)

        if verdict == "FAIL" and not force_gates:
            ps.transition(Phase.GATE_FAILED, f"G5 FAIL: no improving regime")
            return

        ps.transition(Phase.MULTI_FREQ, "G5 passed")
        phase = Phase.MULTI_FREQ

    # --- STEP 6: MULTI-FREQUENCY ---
    if phase == Phase.MULTI_FREQ:
        log.info("=" * 60)
        log.info("  STEP 6: Multi-Frequency Integration")
        log.info("=" * 60)

        from alpha.multi_freq import run_multi_freq

        mf_cfg = config.get("multi_freq", {})
        mf_result = run_multi_freq(
            data_dir=pipe_cfg["data_dir"],
            symbol=pipe_cfg["primary_symbol"],
            timeframe=pipe_cfg["timeframe"],
            signal_path=mf_cfg.get("signal_path", "") or None,
            output=str(report_dir / "alpha_multi_freq.json"),
        )
        ps.set_artifact("multi_freq", str(report_dir / "alpha_multi_freq.json"))
        ps.set_output("multi_freq", {
            "micro_sharpe": mf_result.micro_sharpe,
            "composite_sharpe": mf_result.composite_sharpe,
            "composite_max_dd": mf_result.composite_max_dd,
        })

        # G6: composite improves over individual (2 sub-gates)
        sub_pass = sum([
            mf_result.gate_sharpe_improves,
            mf_result.gate_dd_improves,
        ])
        verdict, metrics, advice = evaluate_gate(
            "G6", sub_pass, 2, weak_min=1,
            metrics={
                "micro_sharpe": mf_result.micro_sharpe,
                "macro_sharpe": mf_result.macro_sharpe,
                "composite_sharpe": mf_result.composite_sharpe,
                "composite_max_dd": mf_result.composite_max_dd,
                "sub_pass": sub_pass,
            },
            advice_on_fail="Macro filter may not add value. Consider skipping this step.",
        )
        ps.record_gate("G6", verdict, metrics, advice)
        log.info("G6 Multi-Freq: %s (%d/2 sub-gates)", verdict, sub_pass)
        if advice:
            log.warning("  Advice: %s", advice)

        if verdict == "FAIL" and not force_gates:
            ps.transition(Phase.GATE_FAILED, f"G6 FAIL: {sub_pass}/2 sub-gates")
            return

        ps.transition(Phase.PORTFOLIO, "G6 passed")
        phase = Phase.PORTFOLIO

    # --- STEP 7: PORTFOLIO ASSEMBLY ---
    if phase == Phase.PORTFOLIO:
        log.info("=" * 60)
        log.info("  STEP 7: Portfolio Assembly")
        log.info("=" * 60)

        from alpha.portfolio import run_portfolio

        port_result = run_portfolio(
            data_dir=pipe_cfg["data_dir"],
            symbols=pipe_cfg["symbols"],
            timeframe=pipe_cfg["timeframe"],
            output=str(report_dir / "alpha_portfolio.json"),
        )
        ps.set_artifact("portfolio", str(report_dir / "alpha_portfolio.json"))
        ps.set_output("portfolio", {
            "portfolio_sharpe": port_result.portfolio_sharpe,
            "portfolio_max_dd": port_result.portfolio_max_dd,
            "max_individual_sharpe": port_result.max_individual_sharpe,
        })

        # G7: diversification benefit (2 sub-gates)
        sub_pass = sum([
            port_result.gate_sharpe_improves,
            port_result.gate_dd_improves,
        ])
        verdict, metrics, advice = evaluate_gate(
            "G7", sub_pass, 2, weak_min=1,
            metrics={
                "portfolio_sharpe": port_result.portfolio_sharpe,
                "max_individual_sharpe": port_result.max_individual_sharpe,
                "portfolio_max_dd": port_result.portfolio_max_dd,
                "sub_pass": sub_pass,
            },
            advice_on_fail="Symbols may be too correlated. Consider single-symbol deployment.",
        )
        ps.record_gate("G7", verdict, metrics, advice)
        log.info("G7 Portfolio: %s (%d/2 sub-gates)", verdict, sub_pass)
        if advice:
            log.warning("  Advice: %s", advice)

        if verdict == "FAIL" and not force_gates:
            ps.transition(Phase.GATE_FAILED, f"G7 FAIL: {sub_pass}/2 sub-gates")
            return

        ps.transition(Phase.PAPER, "G7 passed")
        phase = Phase.PAPER

    # --- STEP 8: PAPER TRADING ---
    if phase == Phase.PAPER:
        log.info("=" * 60)
        log.info("  STEP 8: Paper Trading Simulation")
        log.info("=" * 60)

        from alpha.paper_trader import run_paper_simulation

        paper_cfg = config.get("paper", {})
        val_out = ps.get_output("validate") or {}
        backtest_sharpe = val_out.get("oos_sharpe", paper_cfg.get("backtest_sharpe", 1.0))
        comb_out = ps.get_output("combine") or {}
        backtest_ic = comb_out.get("combined_ic", paper_cfg.get("backtest_ic", 0.03))

        paper_result = run_paper_simulation(
            data_dir=pipe_cfg["data_dir"],
            symbol=pipe_cfg["primary_symbol"],
            timeframe=pipe_cfg["timeframe"],
            backtest_sharpe=backtest_sharpe,
            backtest_ic=backtest_ic,
            output=str(report_dir / "alpha_paper.json"),
        )
        ps.set_artifact("paper", str(report_dir / "alpha_paper.json"))
        ps.set_output("paper", {
            "paper_sharpe": paper_result.paper_sharpe,
            "sharpe_ratio": paper_result.sharpe_ratio,
            "max_daily_loss_pct": paper_result.max_daily_loss_pct,
            "n_days": paper_result.n_days,
        })

        # G8: paper trading stability (4 sub-gates)
        sub_pass = sum([
            paper_result.gate_sharpe_within_2x,
            paper_result.gate_no_big_daily_loss,
            paper_result.gate_ic_stable,
            paper_result.gate_infra_stable,
        ])
        verdict, metrics, advice = evaluate_gate(
            "G8", sub_pass, 4, weak_min=3,
            metrics={
                "paper_sharpe": paper_result.paper_sharpe,
                "sharpe_ratio": paper_result.sharpe_ratio,
                "max_daily_loss_pct": paper_result.max_daily_loss_pct,
                "n_days": paper_result.n_days,
                "sub_pass": sub_pass,
            },
            advice_on_fail="Paper trading metrics unstable. Investigate IC decay or daily losses.",
        )
        ps.record_gate("G8", verdict, metrics, advice)
        log.info("G8 Paper: %s (%d/4 sub-gates)", verdict, sub_pass)
        if advice:
            log.warning("  Advice: %s", advice)

        if verdict == "FAIL" and not force_gates:
            ps.transition(Phase.GATE_FAILED, f"G8 FAIL: {sub_pass}/4 sub-gates")
            return

        ps.transition(Phase.DEPLOYING, "G8 passed")
        phase = Phase.DEPLOYING

    # --- STEP 9: DEPLOYMENT READINESS ---
    if phase == Phase.DEPLOYING:
        log.info("=" * 60)
        log.info("  STEP 9: Deployment Readiness Check")
        log.info("=" * 60)

        from alpha.deployer import check_readiness

        paper_path = ps.get_artifact("paper") or str(report_dir / "alpha_paper.json")
        readiness = check_readiness(paper_report_path=paper_path)

        ps.set_artifact("deployer", "readiness_check")
        ps.set_output("deployer", {
            "overall_ready": readiness.overall_ready,
            "blockers": readiness.blockers,
        })

        # G9: deployment readiness
        verdict = "PASS" if readiness.overall_ready else "FAIL"
        metrics = {
            "overall_ready": readiness.overall_ready,
            "blockers": readiness.blockers,
        }
        advice = "" if readiness.overall_ready else f"Blockers: {'; '.join(readiness.blockers)}"
        ps.record_gate("G9", verdict, metrics, advice)
        log.info("G9 Deployer: %s", verdict)
        if advice:
            log.warning("  Blockers: %s", advice)

        if verdict == "FAIL" and not force_gates:
            ps.transition(Phase.GATE_FAILED, f"G9 FAIL: not ready for deployment")
            return

        ps.set("finished_at", datetime.now(timezone.utc).isoformat())
        ps.transition(Phase.DONE, "All gates passed — ready for deployment")
        phase = Phase.DONE

    # --- DONE ---
    if phase == Phase.DONE:
        log.info("=" * 60)
        log.info("  ALPHA PIPELINE COMPLETE")
        log.info("=" * 60)
        _print_gate_summary(ps, log)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_gate_summary(ps: AlphaPipelineState, log: Optional[logging.Logger] = None) -> None:
    """Print gate verdict table."""
    gates = ps.get("gates", {})
    lines = ["\n  Gate Summary:", "  " + "-" * 58]

    for p in STEP_PHASES:
        gname = GATE_NAMES[p]
        label = STEP_LABELS[p]
        if gname in gates:
            g = gates[gname]
            verdict = g["verdict"]
            metrics = g.get("metrics", {})
            # Build a short metric summary
            if "sub_pass" in metrics:
                detail = f"{metrics['sub_pass']} sub-gates"
            elif "n_significant" in metrics:
                detail = f"{metrics['n_significant']} features"
            elif "overall_ready" in metrics:
                detail = "ready" if metrics["overall_ready"] else "not ready"
            else:
                detail = ""
            icon = {"PASS": "+", "WEAK": "~", "FAIL": "x"}.get(verdict, "?")
            line = f"  [{icon}] {gname} {label:<25s} {verdict:<6s} {detail}"
        else:
            line = f"  [ ] {gname} {label:<25s} --"
        lines.append(line)

    text = "\n".join(lines)
    if log:
        for line in lines:
            log.info(line)
    else:
        print(text)


def _print_status(ps: AlphaPipelineState) -> None:
    """Print pipeline status."""
    print(f"\n  Alpha Pipeline Status")
    print(f"  " + "-" * 40)
    print(f"  Phase:      {ps.current.value}")
    print(f"  Started:    {ps.get('started_at', 'N/A')}")
    print(f"  Finished:   {ps.get('finished_at', 'N/A')}")

    error = ps.get("error")
    if error:
        print(f"  Error:      {error}")

    # Artifacts
    artifacts = ps.get("artifacts", {})
    if artifacts:
        print(f"\n  Artifacts:")
        for name, path in artifacts.items():
            print(f"    {name}: {path}")

    _print_gate_summary(ps)
    print()


def _print_gates_detail(ps: AlphaPipelineState) -> None:
    """Print detailed gate report."""
    gates = ps.get("gates", {})
    if not gates:
        print("\n  No gates evaluated yet.\n")
        return

    for p in STEP_PHASES:
        gname = GATE_NAMES[p]
        label = STEP_LABELS[p]
        if gname not in gates:
            continue
        g = gates[gname]
        print(f"\n  {gname}: {label}")
        print(f"  " + "-" * 40)
        print(f"  Verdict:    {g['verdict']}")
        print(f"  Evaluated:  {g.get('evaluated_at', 'N/A')}")
        if g.get("advice"):
            print(f"  Advice:     {g['advice']}")
        print(f"  Metrics:")
        for k, v in g.get("metrics", {}).items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    print()


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ps = AlphaPipelineState(config["pipeline"]["state_file"])
    log = setup_logging(config["pipeline"]["log_file"])

    if ps.current not in (Phase.IDLE, Phase.DONE, Phase.ERROR, Phase.GATE_FAILED):
        print(f"Pipeline is in phase {ps.current.value}. Use 'resume' or reset first.")
        sys.exit(1)

    ps.reset()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        run_pipeline(config, ps, log)
    except Exception as e:
        log.error("Pipeline error: %s\n%s", e, traceback.format_exc())
        ps.set("error", str(e))
        ps.transition(Phase.ERROR, f"Unhandled: {e}")


def cmd_resume(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ps = AlphaPipelineState(config["pipeline"]["state_file"])
    log = setup_logging(config["pipeline"]["log_file"])

    if ps.current in (Phase.IDLE, Phase.DONE):
        print(f"Nothing to resume (phase={ps.current.value}). Use 'start'.")
        sys.exit(1)

    # If GATE_FAILED and --force-gate, move to the next phase
    if ps.current == Phase.GATE_FAILED and args.force_gate:
        # Find which gate failed and advance past it
        gates = ps.get("gates", {})
        for p in STEP_PHASES:
            gname = GATE_NAMES[p]
            if gname in gates and gates[gname]["verdict"] == "FAIL":
                idx = STEP_PHASES.index(p)
                if idx + 1 < len(STEP_PHASES):
                    next_phase = STEP_PHASES[idx + 1]
                else:
                    next_phase = Phase.DONE
                ps.transition(next_phase, f"Force-gate: skipping {gname}")
                break

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("Resuming pipeline from phase: %s", ps.current.value)
    try:
        run_pipeline(config, ps, log, force_gates=args.force_gate)
    except Exception as e:
        log.error("Pipeline error: %s\n%s", e, traceback.format_exc())
        ps.set("error", str(e))
        ps.transition(Phase.ERROR, f"Unhandled: {e}")


def cmd_status(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ps = AlphaPipelineState(config["pipeline"]["state_file"])
    _print_status(ps)


def cmd_gates(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ps = AlphaPipelineState(config["pipeline"]["state_file"])
    _print_gates_detail(ps)


def cmd_run_step(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ps = AlphaPipelineState(config["pipeline"]["state_file"])
    log = setup_logging(config["pipeline"]["log_file"])

    step = args.step
    if step < 1 or step > 9:
        print("Step must be 1-9.")
        sys.exit(1)

    target_phase = STEP_PHASES[step - 1]
    ps._data["phase"] = target_phase.value
    ps.save()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("Running single step %d (%s)", step, target_phase.value)
    try:
        run_pipeline(config, ps, log, force_gates=True)
    except Exception as e:
        log.error("Step %d error: %s\n%s", step, e, traceback.format_exc())
        ps.set("error", str(e))
        ps.transition(Phase.ERROR, f"Step {step} failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NAT Alpha Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1. Screening      Feature IC + FDR correction
  2. Combining       Top-N → composite signal
  3. Sizing          Cost-aware position filter
  4. Validation      Walk-forward + deflated Sharpe
  5. Regime          Regime-conditioned IC boost
  6. Multi-Freq      Macro trend filter
  7. Portfolio       Risk-parity multi-asset
  8. Paper           Simulated live trading
  9. Deployer        Readiness check
        """,
    )
    parser.add_argument("--config", default="config/alpha.toml", help="Config file path")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("start", help="Start fresh pipeline run")
    resume_p = sub.add_parser("resume", help="Resume from last phase")
    resume_p.add_argument("--force-gate", action="store_true",
                          help="Continue past GATE_FAILED")
    sub.add_parser("status", help="Show pipeline status and gate verdicts")
    sub.add_parser("gates", help="Detailed gate report with metrics")
    step_p = sub.add_parser("run-step", help="Run a single step (1-9)")
    step_p.add_argument("step", type=int, help="Step number (1-9)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "start": cmd_start,
        "resume": cmd_resume,
        "status": cmd_status,
        "gates": cmd_gates,
        "run-step": cmd_run_step,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
