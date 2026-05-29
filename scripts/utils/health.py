"""Lightweight health heartbeat for Python daemons.

Each daemon writes a JSON heartbeat file periodically. External monitoring
(Docker healthcheck, cron, Prometheus node-exporter textfile) checks file
freshness to determine if the daemon is alive and making progress.

Usage in a daemon:
    from utils.health import HealthWriter

    health = HealthWriter("agent_micro", port=None)  # file-based only
    # In cycle loop:
    health.beat(phase="EXECUTE", cycle=42, extra={"hypotheses": 5})
    # On shutdown:
    health.shutdown()

Checking health externally:
    python -m utils.health --check agent_micro --max-age 120
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HEALTH_DIR = Path(__file__).resolve().parents[2] / "data" / "health"


class HealthWriter:
    """Writes periodic heartbeat files for a named daemon."""

    def __init__(self, daemon_name: str, health_dir: Path | None = None):
        self._name = daemon_name
        self._dir = Path(health_dir) if health_dir else HEALTH_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{daemon_name}.json"
        self._start_ts = time.monotonic()

    @property
    def path(self) -> Path:
        return self._path

    def beat(
        self,
        phase: str = "unknown",
        cycle: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write a heartbeat with current state."""
        payload = {
            "daemon": self._name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": time.time(),
            "uptime_s": round(time.monotonic() - self._start_ts, 1),
            "phase": phase,
            "cycle": cycle,
            "pid": _pid(),
        }
        if extra:
            payload["extra"] = extra
        # Atomic write via tmp + rename
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.rename(self._path)

    def shutdown(self) -> None:
        """Mark daemon as stopped."""
        self.beat(phase="STOPPED", cycle=-1)


def check_health(
    daemon_name: str,
    max_age_s: float = 120.0,
    health_dir: Path | None = None,
) -> tuple[bool, str]:
    """Check if a daemon's heartbeat is fresh.

    Returns (healthy, message).
    """
    d = Path(health_dir) if health_dir else HEALTH_DIR
    path = d / f"{daemon_name}.json"
    if not path.exists():
        return False, f"no heartbeat file: {path}"
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return False, f"corrupt heartbeat: {e}"

    epoch = data.get("epoch", 0)
    age = time.time() - epoch
    phase = data.get("phase", "unknown")

    if phase == "STOPPED":
        return False, f"daemon stopped (phase=STOPPED)"
    if age > max_age_s:
        return False, f"stale heartbeat: {age:.0f}s old (max {max_age_s:.0f}s)"
    return True, f"healthy: phase={phase}, cycle={data.get('cycle', '?')}, age={age:.0f}s"


def _pid() -> int:
    import os
    return os.getpid()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check daemon health")
    parser.add_argument("--check", type=str, required=True, help="Daemon name")
    parser.add_argument("--max-age", type=float, default=120.0, help="Max heartbeat age (s)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    healthy, msg = check_health(args.check, max_age_s=args.max_age)
    if args.json:
        print(json.dumps({"healthy": healthy, "message": msg}))
    else:
        status = "OK" if healthy else "UNHEALTHY"
        print(f"[{status}] {args.check}: {msg}")
    raise SystemExit(0 if healthy else 1)


if __name__ == "__main__":
    main()
