#!/usr/bin/env python3
"""
NAT Automated Pipeline Runner — State Machine Orchestrator.

Manages the full lifecycle:
    IDLE → INGESTING → COLLECTING → ANALYZING → DONE

Usage:
    # Start the full pipeline (ingest for 7 days, then analyze)
    python scripts/pipeline_runner.py start

    # Check current status
    python scripts/pipeline_runner.py status

    # Skip to analysis on already-collected data
    python scripts/pipeline_runner.py analyze

    # Force-stop the ingestor
    python scripts/pipeline_runner.py stop

    # Resume after interruption (picks up from saved state)
    python scripts/pipeline_runner.py resume
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import signal
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Pipeline states
# ---------------------------------------------------------------------------


class State(str, Enum):
    IDLE = "IDLE"
    BUILDING = "BUILDING"
    INGESTING = "INGESTING"
    COLLECTING = "COLLECTING"  # flush + validate collected data
    ANALYZING = "ANALYZING"
    DONE = "DONE"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------


def load_config(config_path: str = "config/pipeline.toml") -> Dict[str, Any]:
    """Load pipeline configuration from TOML."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class PipelineState:
    """Persistent state for the pipeline, survives restarts."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            "state": State.IDLE.value,
            "started_at": None,
            "ingest_started_at": None,
            "ingest_target_end": None,
            "ingest_pid": None,
            "ingest_stopped_at": None,
            "analyze_started_at": None,
            "analyze_finished_at": None,
            "last_health_check": None,
            "health_checks_ok": 0,
            "health_checks_fail": 0,
            "restarts": 0,
            "total_rows": 0,
            "total_files": 0,
            "decision": None,
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

    def transition(self, new_state: State, message: str = "") -> None:
        old = self._data["state"]
        self._data["state"] = new_state.value
        self._data["history"].append({
            "from": old,
            "to": new_state.value,
            "at": datetime.datetime.utcnow().isoformat(),
            "message": message,
        })
        self.save()

    @property
    def current(self) -> State:
        return State(self._data["state"])


# ---------------------------------------------------------------------------
# Ingestor management
# ---------------------------------------------------------------------------


def build_ingestor(project_root: Path, log: logging.Logger) -> bool:
    """Build the Rust ingestor in release mode."""
    log.info("Building ingestor (release)...")
    result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "ing"],
        cwd=project_root / "rust",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("Build failed:\n%s", result.stderr[-2000:])
        return False
    log.info("Build succeeded.")
    return True


def start_ingestor(
    project_root: Path,
    ingestor_config: str,
    log_file: Path,
    log: logging.Logger,
) -> Optional[subprocess.Popen]:
    """Start the Rust ingestor as a background process."""
    binary = project_root / "rust" / "target" / "release" / "ing"
    config_path = project_root / ingestor_config

    if not binary.exists():
        log.error("Ingestor binary not found: %s", binary)
        return None

    if not config_path.exists():
        log.error("Ingestor config not found: %s", config_path)
        return None

    log_file.parent.mkdir(parents=True, exist_ok=True)

    log.info("Starting ingestor: %s %s", binary, config_path)
    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            [str(binary), str(config_path)],
            cwd=str(project_root),
            stdout=lf,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # new process group so we can kill cleanly
        )

    log.info("Ingestor started, PID=%d", proc.pid)
    return proc


def stop_ingestor(pid: Optional[int], log: logging.Logger) -> None:
    """Stop the ingestor process gracefully, then force if needed."""
    if pid is None:
        return

    try:
        os.kill(pid, 0)  # check if alive
    except OSError:
        log.info("Ingestor PID %d already stopped.", pid)
        return

    log.info("Sending SIGTERM to ingestor PID %d...", pid)
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        return

    # Wait up to 10 seconds for graceful shutdown
    for _ in range(20):
        try:
            os.kill(pid, 0)
            time.sleep(0.5)
        except OSError:
            log.info("Ingestor stopped gracefully.")
            return

    log.warning("Ingestor didn't stop, sending SIGKILL...")
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def is_ingestor_alive(pid: Optional[int]) -> bool:
    """Check if the ingestor process is still running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def health_check(data_dir: str, max_gap: int, log: logging.Logger) -> Dict[str, Any]:
    """
    Check data freshness — are new parquet files being written?

    Returns dict with status, file_count, total_rows, latest_file, gap_seconds.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return {"ok": False, "reason": "data directory does not exist"}

    parquet_files = sorted(data_path.rglob("*.parquet"), key=os.path.getmtime)
    if not parquet_files:
        return {"ok": False, "reason": "no parquet files found"}

    latest = parquet_files[-1]
    mtime = os.path.getmtime(latest)
    gap = time.time() - mtime

    # Count total rows via pyarrow metadata
    total_rows = 0
    try:
        import pyarrow.parquet as pq
        for f in parquet_files:
            meta = pq.read_metadata(str(f))
            total_rows += meta.num_rows
    except Exception:
        pass

    ok = gap < max_gap
    if not ok:
        log.warning(
            "Data stale: latest file %s is %.0f seconds old (max: %d)",
            latest.name, gap, max_gap,
        )

    return {
        "ok": ok,
        "file_count": len(parquet_files),
        "total_rows": total_rows,
        "latest_file": str(latest),
        "gap_seconds": gap,
    }


# ---------------------------------------------------------------------------
# Data collection / validation
# ---------------------------------------------------------------------------


def collect_data(data_dir: str, log: logging.Logger) -> Dict[str, Any]:
    """Validate and summarize collected data after ingestion stops."""
    sys.path.insert(0, str(Path(__file__).parent))

    from cluster_pipeline.loader import scan_schema, validate_schema, load_parquet

    log.info("Scanning collected data in %s...", data_dir)
    schema = scan_schema(data_dir)
    log.info(
        "Found %d files, %d total rows, %d columns",
        schema["file_count"], schema["total_rows"], len(schema["columns"]),
    )

    # Light validation on first 10k rows
    df_sample = load_parquet(data_dir, max_rows=10_000)
    val = validate_schema(df_sample, require_meta=True)

    if not val["valid"]:
        log.error("Schema validation failed: %s", val["errors"])
    else:
        log.info("Schema valid. Vectors available: %s", val["vectors_available"])

    if val["warnings"]:
        for w in val["warnings"][:5]:
            log.warning("  %s", w)

    return {
        "file_count": schema["file_count"],
        "total_rows": schema["total_rows"],
        "columns": len(schema["columns"]),
        "symbols": schema["symbols"],
        "valid": val["valid"],
        "vectors_available": val["vectors_available"],
        "vectors_complete": val["vectors_complete"],
        "errors": val["errors"],
    }


# ---------------------------------------------------------------------------
# Analysis phase
# ---------------------------------------------------------------------------


def run_analysis(
    data_dir: str,
    config: Dict[str, Any],
    report_dir: str,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Run the full cluster analysis pipeline on collected data.

    This mirrors the notebook logic but runs non-interactively,
    saving figures and a JSON report to report_dir.
    """
    sys.path.insert(0, str(Path(__file__).parent))

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cluster_pipeline.loader import load_parquet
    from cluster_pipeline.preprocess import aggregate_bars, preprocess, bar_summary
    from cluster_pipeline.cluster import full_analysis, predictive_quality, compute_linkage
    from cluster_pipeline.reduce import reduce_all, pca_optimal_components, top_pca_features
    from cluster_pipeline.viz import generate_all_plots, plot_comparison_grid, plot_scatter_2d

    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    fig_dir = report_path / "figures"
    fig_dir.mkdir(exist_ok=True)

    acfg = config["analysis"]
    thresholds = acfg["thresholds"]
    timeframe = acfg["timeframe"]
    vectors = acfg["vectors"]
    scaler = acfg["scaler"]
    k_range = range(acfg["k_min"], acfg["k_max"] + 1)
    n_bootstrap = acfg["n_bootstrap"]
    seed = acfg["random_state"]
    dpi = config["output"].get("figure_dpi", 150)

    # Load data
    log.info("Loading data from %s...", data_dir)
    df = load_parquet(data_dir)
    log.info("Loaded %d rows", len(df))

    # Aggregate bars
    log.info("Aggregating bars at %s...", timeframe)
    bars = aggregate_bars(df, timeframe=timeframe)
    summary = bar_summary(bars)
    log.info("Bars: %d, Features: %d", summary["n_bars"], summary["n_features"])

    # Forward returns for predictive quality
    midprice_col = None
    for candidate in ["raw_midprice_close", "raw_midprice_mean", "raw_microprice_close"]:
        if candidate in bars.columns:
            midprice_col = candidate
            break

    forward_returns = None
    if midprice_col:
        forward_returns = bars[midprice_col].pct_change().shift(-1).values
        log.info("Forward returns from %s", midprice_col)

    # Per-vector analysis
    results = {}
    for vname in vectors:
        log.info("Analyzing vector: %s", vname)

        try:
            X, feat_cols, meta = preprocess(
                bars, vector=vname, scaler=scaler, clip_sigma=5.0,
            )
        except ValueError as e:
            log.warning("  Skip %s: %s", vname, e)
            results[vname] = {"status": "skip", "reason": str(e)}
            continue

        if X.shape[0] < max(k_range) + 1:
            log.warning("  Skip %s: only %d bars", vname, X.shape[0])
            results[vname] = {"status": "skip", "reason": "too few bars"}
            continue

        analysis = full_analysis(
            X, k_range=k_range, n_bootstrap=n_bootstrap,
            column_names=feat_cols, random_state=seed,
        )

        q = analysis["quality"]
        b = analysis["bootstrap_stability"]
        t = analysis["temporal_stability"]
        best_k = analysis["best_k"]
        labels = analysis["best_result"].labels
        modality = analysis["multimodality"]
        n_mm = sum(1 for m in modality if m["multimodal"])

        log.info("  k=%d sil=%.4f boot_ari=%.4f temp_ari=%.4f",
                 best_k, q.silhouette, b.mean_ari, t.mean_ari)

        # Q1-Q3
        q1 = q.silhouette > thresholds["silhouette"] and best_k > 1
        q2 = b.mean_ari > thresholds["bootstrap_ari"] and t.mean_ari > thresholds["temporal_ari"]
        q3 = None
        pq = None

        if forward_returns is not None:
            n = min(len(labels), len(forward_returns))
            fr = forward_returns[:n]
            lab = labels[:n]
            mask = ~np.isnan(fr)
            if mask.sum() >= 10:
                pq = predictive_quality(lab[mask], fr[mask])
                q3 = pq["kruskal_wallis_p"] < thresholds["kruskal_p"] and \
                     pq["eta_squared"] > thresholds["eta_squared"]

        # Dimensionality reduction + plots
        projections = reduce_all(X, n_components=2, random_state=seed)
        embedding_2d = projections["pca"].embedding

        X_link = X[:500] if len(X) > 500 else X
        linkage_mat = compute_linkage(X_link, method="ward")

        plots = generate_all_plots(
            X, labels, embedding_2d,
            feature_names=feat_cols,
            sweep=analysis["sweep"],
            linkage_matrix=linkage_mat,
            method_name=f"PCA — {vname}",
        )

        # Save figures
        for pname, fig in plots.items():
            fig_path = fig_dir / f"{vname}_{pname}.png"
            fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        if "tsne" in projections:
            fig_tsne = plot_scatter_2d(
                projections["tsne"].embedding, labels,
                title=f"t-SNE — {vname} (k={best_k})",
            )
            fig_tsne.savefig(fig_dir / f"{vname}_tsne.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig_tsne)

        results[vname] = {
            "status": "ok",
            "best_k": best_k,
            "silhouette": float(q.silhouette),
            "davies_bouldin": float(q.davies_bouldin),
            "bootstrap_ari": float(b.mean_ari),
            "temporal_ari": float(t.mean_ari),
            "n_multimodal": n_mm,
            "n_features": len(feat_cols),
            "n_bars": X.shape[0],
            "cluster_sizes": {str(k): v for k, v in q.cluster_sizes.items()},
            "q1_pass": q1,
            "q2_pass": q2,
            "q3_pass": q3,
            "predictive": {
                "kruskal_p": pq["kruskal_wallis_p"] if pq else None,
                "eta_squared": pq["eta_squared"] if pq else None,
                "self_transition": pq["self_transition_rate"] if pq else None,
            } if pq else None,
        }

    # Decision gate
    ok_results = {k: v for k, v in results.items() if v["status"] == "ok"}
    n_ok = len(ok_results)
    q1_yes = sum(1 for v in ok_results.values() if v["q1_pass"])
    q2_yes = sum(1 for v in ok_results.values() if v["q2_pass"])
    q3_yes = sum(1 for v in ok_results.values() if v.get("q3_pass"))

    if n_ok > 0 and q1_yes > n_ok / 2 and q2_yes > n_ok / 2 and q3_yes > 0:
        decision = "GO"
    elif n_ok > 0 and q1_yes > n_ok / 2 and q2_yes > 0:
        decision = "PIVOT"
    else:
        decision = "NO-GO"

    # Find best vector
    best_vector = None
    if ok_results:
        best_vector = max(ok_results, key=lambda k: ok_results[k]["silhouette"])

    gate = {
        "decision": decision,
        "best_vector": best_vector,
        "q1_pass": q1_yes,
        "q2_pass": q2_yes,
        "q3_pass": q3_yes,
        "n_vectors_ok": n_ok,
        "n_vectors_total": len(vectors),
    }

    log.info("DECISION GATE: %s (Q1=%d/%d Q2=%d/%d Q3=%d/%d)",
             decision, q1_yes, n_ok, q2_yes, n_ok, q3_yes, n_ok)

    # Save JSON report
    report = {
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "data_dir": data_dir,
        "timeframe": timeframe,
        "bar_summary": {
            "n_bars": summary["n_bars"],
            "n_features": summary["n_features"],
            "symbols": summary["symbols"],
            "time_range": summary["time_range"],
        },
        "vectors": results,
        "decision_gate": gate,
    }

    report_file = report_path / "analysis_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report saved: %s", report_file)

    # Best vector comparison grid
    if best_vector and ok_results:
        log.info("Generating comparison grid for best vector: %s", best_vector)
        # Re-run preprocess for best vector to get embedding
        X_best, cols_best, _ = preprocess(
            bars, vector=best_vector, scaler=scaler, clip_sigma=5.0,
        )
        proj_best = reduce_all(X_best, n_components=2, random_state=seed)
        emb_best = proj_best["pca"].embedding
        analysis_best = full_analysis(
            X_best, k_range=k_range, n_bootstrap=10,
            column_names=cols_best, random_state=seed,
        )
        labels_best = analysis_best["best_result"].labels

        entropy_vals = None
        for ec in ["ent_tick_1m_mean", "ent_tick_5s_mean"]:
            if ec in bars.columns:
                entropy_vals = bars[ec].values[:len(labels_best)]
                break

        symbol_vals = bars["symbol"].values[:len(labels_best)] if "symbol" in bars.columns else None
        fr_vals = forward_returns[:len(labels_best)] if forward_returns is not None else None

        fig_grid = plot_comparison_grid(
            emb_best,
            labels=labels_best,
            entropy=entropy_vals,
            forward_returns=fr_vals,
            symbols=symbol_vals,
            title=f"Best Vector: {best_vector} / {timeframe}",
        )
        fig_grid.savefig(fig_dir / "best_comparison_grid.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig_grid)

    return gate


# ---------------------------------------------------------------------------
# Main state machine
# ---------------------------------------------------------------------------


def setup_logging(log_file: Optional[str]) -> logging.Logger:
    """Configure logging to both console and file."""
    log = logging.getLogger("pipeline")
    log.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # File
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


def run_pipeline(config: Dict[str, Any], ps: PipelineState, log: logging.Logger) -> None:
    """
    Execute the full pipeline state machine.

    Transitions: IDLE → BUILDING → INGESTING → COLLECTING → ANALYZING → DONE
    """
    project_root = Path(__file__).resolve().parent.parent
    icfg = config["ingestion"]
    data_dir = icfg["data_dir"]
    ingestor_config = icfg["ingestor_config"]
    duration_days = icfg["duration_days"]
    health_interval = icfg["health_check_interval"]
    max_gap = icfg["max_gap_seconds"]
    report_dir = config["output"]["report_dir"]
    log_file = Path(config["state"]["log_file"]).parent / "ingestor.log"

    # Handle signal for clean shutdown
    ingestor_proc = None
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log.info("Shutdown signal received (signal=%d)", signum)
        shutdown_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    state = ps.current

    # --- IDLE → BUILDING ---
    if state == State.IDLE:
        ps.set("started_at", datetime.datetime.utcnow().isoformat())
        ps.transition(State.BUILDING, "Starting pipeline")
        state = State.BUILDING

    # --- BUILDING ---
    if state == State.BUILDING:
        if not build_ingestor(project_root, log):
            ps.transition(State.ERROR, "Build failed")
            ps.set("error", "Rust ingestor build failed")
            return
        ps.transition(State.INGESTING, "Build complete")
        state = State.INGESTING

    # --- INGESTING ---
    if state == State.INGESTING:
        # Calculate end time
        if ps.get("ingest_started_at") is None:
            now = datetime.datetime.utcnow()
            end_time = now + datetime.timedelta(days=duration_days)
            ps.set("ingest_started_at", now.isoformat())
            ps.set("ingest_target_end", end_time.isoformat())
            log.info("Ingestion target: %s → %s (%d days)", now, end_time, duration_days)
        else:
            end_time = datetime.datetime.fromisoformat(ps.get("ingest_target_end"))
            log.info("Resuming ingestion. Target end: %s", end_time)

        # Start ingestor (or check if already running from resume)
        existing_pid = ps.get("ingest_pid")
        if existing_pid and is_ingestor_alive(existing_pid):
            log.info("Ingestor already running (PID %d), attaching...", existing_pid)
        else:
            ingestor_proc = start_ingestor(project_root, ingestor_config, log_file, log)
            if ingestor_proc is None:
                ps.transition(State.ERROR, "Failed to start ingestor")
                return
            ps.set("ingest_pid", ingestor_proc.pid)

        pid = ps.get("ingest_pid")

        # Monitor loop
        while not shutdown_requested:
            now = datetime.datetime.utcnow()

            # Check if duration elapsed
            if now >= end_time:
                log.info("Ingestion duration reached. Stopping ingestor...")
                stop_ingestor(pid, log)
                ps.set("ingest_stopped_at", now.isoformat())
                ps.transition(State.COLLECTING, "Ingestion complete")
                state = State.COLLECTING
                break

            # Health check
            hc = health_check(data_dir, max_gap, log)
            ps.set("last_health_check", now.isoformat())

            if hc["ok"]:
                ps.set("health_checks_ok", ps.get("health_checks_ok", 0) + 1)
                ps.set("total_rows", hc.get("total_rows", 0))
                ps.set("total_files", hc.get("file_count", 0))
            else:
                ps.set("health_checks_fail", ps.get("health_checks_fail", 0) + 1)
                log.warning("Health check failed: %s", hc.get("reason", "unknown"))

                # Restart if ingestor died
                if not is_ingestor_alive(pid):
                    log.warning("Ingestor not running, restarting...")
                    ps.set("restarts", ps.get("restarts", 0) + 1)
                    ingestor_proc = start_ingestor(
                        project_root, ingestor_config, log_file, log,
                    )
                    if ingestor_proc:
                        pid = ingestor_proc.pid
                        ps.set("ingest_pid", pid)
                    else:
                        log.error("Failed to restart ingestor")

            # Status log
            elapsed = now - datetime.datetime.fromisoformat(ps.get("ingest_started_at"))
            remaining = end_time - now
            log.info(
                "Ingesting: %s elapsed, %s remaining | files=%d rows=%d | health ok=%d fail=%d restarts=%d",
                str(elapsed).split(".")[0],
                str(remaining).split(".")[0],
                ps.get("total_files", 0),
                ps.get("total_rows", 0),
                ps.get("health_checks_ok", 0),
                ps.get("health_checks_fail", 0),
                ps.get("restarts", 0),
            )

            # Sleep until next health check
            time.sleep(health_interval)

        # Handle shutdown during ingestion
        if shutdown_requested and state == State.INGESTING:
            log.info("Graceful shutdown during ingestion. State saved for resume.")
            stop_ingestor(pid, log)
            return

    # --- COLLECTING ---
    if state == State.COLLECTING:
        log.info("Collecting and validating data...")
        collection = collect_data(data_dir, log)
        ps.set("total_rows", collection["total_rows"])
        ps.set("total_files", collection["file_count"])

        if not collection["valid"]:
            log.error("Data validation failed: %s", collection["errors"])
            ps.transition(State.ERROR, "Data validation failed")
            ps.set("error", str(collection["errors"]))
            return

        log.info(
            "Collection OK: %d files, %d rows, vectors=%s",
            collection["file_count"], collection["total_rows"],
            collection["vectors_available"],
        )
        ps.transition(State.ANALYZING, "Data validated")
        state = State.ANALYZING

    # --- ANALYZING ---
    if state == State.ANALYZING:
        ps.set("analyze_started_at", datetime.datetime.utcnow().isoformat())
        log.info("Starting cluster analysis...")

        try:
            gate = run_analysis(data_dir, config, report_dir, log)
            ps.set("decision", gate["decision"])
            ps.set("analyze_finished_at", datetime.datetime.utcnow().isoformat())
            ps.transition(State.DONE, f"Analysis complete: {gate['decision']}")
        except Exception as e:
            log.exception("Analysis failed")
            ps.transition(State.ERROR, f"Analysis failed: {e}")
            ps.set("error", str(e))
            return

    # --- DONE ---
    if ps.current == State.DONE:
        log.info("=" * 60)
        log.info("  PIPELINE COMPLETE")
        log.info("  Decision: %s", ps.get("decision"))
        log.info("  Report:   %s", Path(report_dir) / "analysis_report.json")
        log.info("  Figures:  %s", Path(report_dir) / "figures")
        log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ps = PipelineState(config["state"]["state_file"])
    log = setup_logging(config["state"]["log_file"])

    if ps.current not in (State.IDLE, State.DONE, State.ERROR):
        print(f"Pipeline is in state {ps.current.value}. Use 'resume' or 'stop' first.")
        sys.exit(1)

    # Reset state for fresh start
    ps.transition(State.IDLE, "Fresh start")
    run_pipeline(config, ps, log)


def cmd_resume(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ps = PipelineState(config["state"]["state_file"])
    log = setup_logging(config["state"]["log_file"])

    if ps.current in (State.IDLE, State.DONE):
        print(f"Nothing to resume (state={ps.current.value}). Use 'start'.")
        sys.exit(1)

    log.info("Resuming pipeline from state: %s", ps.current.value)
    run_pipeline(config, ps, log)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Skip ingestion, run analysis directly on existing data."""
    config = load_config(args.config)
    ps = PipelineState(config["state"]["state_file"])
    log = setup_logging(config["state"]["log_file"])

    ps.transition(State.COLLECTING, "Direct analysis (skip ingestion)")
    run_pipeline(config, ps, log)


def cmd_stop(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ps = PipelineState(config["state"]["state_file"])
    log = setup_logging(config["state"]["log_file"])

    pid = ps.get("ingest_pid")
    if pid and is_ingestor_alive(pid):
        stop_ingestor(pid, log)
        ps.set("ingest_stopped_at", datetime.datetime.utcnow().isoformat())
        ps.transition(State.COLLECTING, "Manual stop → collecting")
        print(f"Ingestor stopped (PID {pid}). State: COLLECTING")
        print("Run 'python scripts/pipeline_runner.py resume' to continue to analysis.")
    else:
        print("No running ingestor found.")
        print(f"Current state: {ps.current.value}")


def cmd_status(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ps = PipelineState(config["state"]["state_file"])

    print(f"\nPipeline State: {ps.current.value}")
    print(f"{'─' * 50}")

    if ps.get("started_at"):
        print(f"  Started:        {ps.get('started_at')}")
    if ps.get("ingest_started_at"):
        print(f"  Ingest started: {ps.get('ingest_started_at')}")
    if ps.get("ingest_target_end"):
        print(f"  Ingest target:  {ps.get('ingest_target_end')}")

        # Time remaining
        try:
            end = datetime.datetime.fromisoformat(ps.get("ingest_target_end"))
            now = datetime.datetime.utcnow()
            if now < end:
                remaining = end - now
                print(f"  Time remaining: {str(remaining).split('.')[0]}")
            else:
                print(f"  Time remaining: ELAPSED")
        except Exception:
            pass

    pid = ps.get("ingest_pid")
    alive = is_ingestor_alive(pid) if pid else False
    print(f"  Ingestor PID:   {pid} ({'RUNNING' if alive else 'stopped'})")
    print(f"  Files:          {ps.get('total_files', 0)}")
    print(f"  Rows:           {ps.get('total_rows', 0):,}")
    print(f"  Health OK:      {ps.get('health_checks_ok', 0)}")
    print(f"  Health Fail:    {ps.get('health_checks_fail', 0)}")
    print(f"  Restarts:       {ps.get('restarts', 0)}")

    if ps.get("decision"):
        print(f"  Decision:       {ps.get('decision')}")
    if ps.get("error"):
        print(f"  Error:          {ps.get('error')}")

    # History
    history = ps.get("history", [])
    if history:
        print(f"\n  State History (last 10):")
        for h in history[-10:]:
            print(f"    {h['at']}  {h['from']} → {h['to']}  {h.get('message', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NAT Automated Pipeline — Ingest, Collect, Analyze",
    )
    parser.add_argument(
        "--config", default="config/pipeline.toml",
        help="Pipeline config file (default: config/pipeline.toml)",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("start", help="Start fresh pipeline (ingest → analyze)")
    sub.add_parser("resume", help="Resume from saved state after interruption")
    sub.add_parser("analyze", help="Skip ingestion, analyze existing data")
    sub.add_parser("stop", help="Stop the ingestor and save state")
    sub.add_parser("status", help="Show current pipeline status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "start": cmd_start,
        "resume": cmd_resume,
        "analyze": cmd_analyze,
        "stop": cmd_stop,
        "status": cmd_status,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
