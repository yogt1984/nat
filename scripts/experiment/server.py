"""
Task 7: API server.

FastAPI app serving the dashboard HTML and JSON API endpoints.

Usage:
    python -m scripts.experiment.server
    python -m scripts.experiment.server --port 8050
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add scripts to path
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    print("ERROR: fastapi and uvicorn required. Install with:")
    print("  pip install fastapi uvicorn")
    sys.exit(1)

from experiment.state import load_state, DEFAULT_STATE_PATH
from experiment.events import read_events

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="NAT Experiment Dashboard", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML."""
    html_path = STATIC_DIR / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text())


@app.get("/api/state")
async def get_state():
    """Return current experiment state as JSON."""
    state = load_state(DEFAULT_STATE_PATH)
    from dataclasses import asdict
    return JSONResponse(asdict(state))


@app.get("/api/events")
async def get_events(n: int = 50):
    """Return last N events."""
    events = read_events(n)
    return JSONResponse(events)


@app.get("/api/heartbeat")
async def heartbeat():
    """Live heartbeat: check ingestor process + tail recent log lines."""
    import subprocess
    import os
    from datetime import datetime, timezone

    # Check ingestor process
    result = subprocess.run(["pgrep", "-x", "ing"], capture_output=True, text=True)
    pid = int(result.stdout.strip().split("\n")[0]) if result.returncode == 0 and result.stdout.strip() else None

    # Find most recent log file
    log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    log_lines = []
    if log_dir.exists():
        logs = sorted(log_dir.glob("ingestor_*.log"), key=os.path.getmtime, reverse=True)
        if logs:
            # Read last 10 lines, strip ANSI codes
            import re
            ansi_re = re.compile(r'\x1b\[[0-9;]*m')
            try:
                with open(logs[0], 'rb') as f:
                    # Seek to end, read last ~4KB
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 4096))
                    tail = f.read().decode('utf-8', errors='replace')
                    lines = tail.strip().split('\n')
                    log_lines = [ansi_re.sub('', l) for l in lines[-8:]]
            except Exception:
                pass

    # Check data freshness
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "features"
    latest_mtime = 0
    if data_dir.exists():
        for p in data_dir.rglob("*.parquet"):
            mt = p.stat().st_mtime
            if mt > latest_mtime:
                latest_mtime = mt

    data_age_s = (datetime.now(timezone.utc).timestamp() - latest_mtime) if latest_mtime > 0 else -1

    return JSONResponse({
        "alive": pid is not None,
        "pid": pid,
        "data_age_s": round(data_age_s, 1),
        "log_lines": log_lines,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def main():
    parser = argparse.ArgumentParser(description="NAT Dashboard Server")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Dashboard: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
