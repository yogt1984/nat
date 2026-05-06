"""
Task 6: Background monitor loop.

Runs every 60 seconds, collects metrics, periodically runs profiling.
Updates experiment_state.json for the dashboard to read.

Usage:
    python -m scripts.experiment.monitor
    python -m scripts.experiment.monitor --interval 60 --profile-every 21600
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add scripts to path
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from experiment.state import ExperimentState, ExperimentInfo, load_state, save_state, DEFAULT_STATE_PATH
from experiment.metrics import collect_data_metrics, check_health, DEFAULT_DATA_DIR
from experiment.profiler import quick_profile
from experiment.events import log_event

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_running = True


def _shutdown(sig, frame):
    global _running
    logger.info("Shutdown signal received")
    _running = False


def run_monitor(
    interval: int = 60,
    profile_every: int = 21600,  # 6 hours
    data_dir: Path = DEFAULT_DATA_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
):
    """Main monitor loop."""
    global _running

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(f"Monitor started. Interval={interval}s, profile_every={profile_every}s")
    log_event("started", "Monitor started")

    # Load or create state
    state = load_state(state_path)
    if not state.experiment.started:
        state.experiment.started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state.experiment.status = "COLLECTING"
    save_state(state, state_path)

    last_profile_time = 0
    last_bars_milestone = 0
    bar_milestones = [50, 100, 200, 500, 1000]

    while _running:
        try:
            # Collect data metrics
            data = collect_data_metrics(data_dir)
            state.data = data

            # Health check
            health = check_health(data_dir, hours=1)
            state.health = health

            # Check if we should profile
            now = time.time()
            should_profile = False

            # Time-based: every profile_every seconds
            if now - last_profile_time >= profile_every:
                should_profile = True

            # Milestone-based: when bars crosses a threshold
            for milestone in bar_milestones:
                if data.bars_15m >= milestone > last_bars_milestone:
                    should_profile = True
                    last_bars_milestone = milestone
                    log_event("health", f"Milestone: {milestone} bars reached")
                    break

            if should_profile and data.bars_15m >= 50:
                logger.info(f"Running profiling on {data.bars_15m} bars...")
                log_event("profiling", f"Starting profiling on {data.bars_15m} bars")

                snapshot = quick_profile(data_dir)
                state.profiling = snapshot
                last_profile_time = now

                if snapshot.status == "complete":
                    log_event("profiling",
                              f"k={snapshot.k}, sil={snapshot.silhouette}, "
                              f"ARI={snapshot.bootstrap_ari}, verdict={snapshot.current_verdict}")
                    logger.info(f"Profiling complete: k={snapshot.k}, verdict={snapshot.current_verdict}")
                elif snapshot.status == "insufficient":
                    log_event("profiling", f"Insufficient bars ({snapshot.n_bars_used})")

            # Update events in state (last 20)
            from experiment.events import read_events
            state.events = read_events(20)

            # Save state
            state.updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_state(state, state_path)

        except Exception as e:
            logger.error(f"Monitor loop error: {e}")
            log_event("error", str(e))

        # Sleep in small increments for responsive shutdown
        for _ in range(interval):
            if not _running:
                break
            time.sleep(1)

    # Final state update
    state.experiment.status = "DONE" if state.profiling.current_verdict else "IDLE"
    state.updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_state(state, state_path)
    log_event("stopped", "Monitor stopped")
    logger.info("Monitor stopped")


def main():
    parser = argparse.ArgumentParser(description="NAT Experiment Monitor")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--profile-every", type=int, default=21600, help="Profiling interval in seconds")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--state-path", type=str, default=str(DEFAULT_STATE_PATH))
    args = parser.parse_args()

    run_monitor(
        interval=args.interval,
        profile_every=args.profile_every,
        data_dir=Path(args.data_dir),
        state_path=Path(args.state_path),
    )


if __name__ == "__main__":
    main()
