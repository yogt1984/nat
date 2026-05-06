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


def main():
    parser = argparse.ArgumentParser(description="NAT Dashboard Server")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Dashboard: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
