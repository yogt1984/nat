#!/usr/bin/env python3
"""
NAT Pipeline Dashboard — Read-only web view of pipeline state, logs, and results.

Serves:
    /              — Dashboard page (state, logs, figures)
    /api/state     — Pipeline state JSON
    /api/logs      — Last N lines of pipeline log
    /api/results   — Analysis report JSON
    /api/figures   — List of available figure filenames
    /figures/<name> — Serve individual figure PNGs

Usage:
    python scripts/dashboard.py                          # defaults
    python scripts/dashboard.py --port 8050              # custom port
    python scripts/dashboard.py --config config/pipeline.toml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8050
DEFAULT_HOST = "0.0.0.0"
DEFAULT_LOG_LINES = 200
MAX_LOG_LINES = 5000


def load_config(config_path: str) -> Dict[str, Any]:
    """Load pipeline.toml and resolve paths relative to project root."""
    project_root = Path(config_path).resolve().parent.parent
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    # Resolve relative paths
    cfg["_project_root"] = str(project_root)
    cfg["_state_file"] = str(project_root / cfg["state"]["state_file"])
    cfg["_log_file"] = str(project_root / cfg["state"]["log_file"])
    cfg["_report_dir"] = str(project_root / cfg["output"]["report_dir"])
    cfg["_data_dir"] = str(project_root / cfg["ingestion"]["data_dir"])
    return cfg


# ---------------------------------------------------------------------------
# Data readers (pure functions, no side effects)
# ---------------------------------------------------------------------------


def read_state(state_file: str) -> Dict[str, Any]:
    """Read pipeline state JSON. Returns empty-state dict if file missing."""
    p = Path(state_file)
    if not p.exists():
        return {"state": "IDLE", "error": None, "history": [],
                "message": "No state file found — pipeline has not been started."}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"state": "UNKNOWN", "error": str(e), "history": []}


def read_log_tail(log_file: str, n_lines: int = DEFAULT_LOG_LINES) -> List[str]:
    """Read last n_lines from log file. Returns empty list if missing."""
    p = Path(log_file)
    if not p.exists():
        return []
    try:
        with open(p) as f:
            return list(deque(f, maxlen=n_lines))
    except OSError:
        return []


def list_figures(report_dir: str) -> List[str]:
    """List PNG/SVG figure files in report directory."""
    p = Path(report_dir)
    if not p.is_dir():
        return []
    extensions = {".png", ".svg", ".jpg", ".jpeg"}
    return sorted(
        f.name for f in p.iterdir()
        if f.suffix.lower() in extensions
    )


def read_report(report_dir: str) -> Optional[Dict[str, Any]]:
    """Read analysis_report.json if it exists."""
    p = Path(report_dir) / "analysis_report.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def data_dir_stats(data_dir: str, project_root: str = "") -> Dict[str, Any]:
    """Quick stats on collected data: file count, total size, date range."""
    p = Path(data_dir)
    # Compute relative path from project root for display
    try:
        rel_path = str(p.relative_to(project_root)) if project_root else str(p)
    except ValueError:
        rel_path = str(p)
    if not p.is_dir():
        return {"exists": False, "path": rel_path, "n_files": 0, "total_bytes": 0, "dates": []}
    parquets = sorted(p.rglob("*.parquet"))
    total = sum(f.stat().st_size for f in parquets)
    dates = sorted({f.parent.name for f in parquets if f.parent != p})
    return {
        "exists": True,
        "path": rel_path,
        "n_files": len(parquets),
        "total_bytes": total,
        "total_mb": round(total / (1024 * 1024), 2),
        "dates": dates,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NAT Pipeline Dashboard</title>
<style>
  :root { --bg: #0d1117; --fg: #c9d1d9; --card: #161b22; --border: #30363d;
          --accent: #58a6ff; --green: #3fb950; --red: #f85149; --yellow: #d29922; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
         background: var(--bg); color: var(--fg); padding: 20px; }
  h1 { color: var(--accent); margin-bottom: 20px; font-size: 1.4em; }
  h2 { color: var(--fg); margin: 16px 0 8px; font-size: 1.1em; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 16px; }
  .card-full { grid-column: 1 / -1; }
  .state-badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 0.9em; }
  .state-IDLE { background: var(--border); color: var(--fg); }
  .state-INGESTING, .state-BUILDING { background: #1f6feb33; color: var(--accent); }
  .state-COLLECTING, .state-ANALYZING { background: #d2992233; color: var(--yellow); }
  .state-DONE { background: #3fb95033; color: var(--green); }
  .state-ERROR { background: #f8514933; color: var(--red); }
  .decision-GO { color: var(--green); font-weight: bold; }
  .decision-PIVOT { color: var(--yellow); font-weight: bold; }
  .decision-NO-GO { color: var(--red); font-weight: bold; }
  .kv { display: flex; justify-content: space-between; padding: 4px 0;
        border-bottom: 1px solid var(--border); font-size: 0.85em; }
  .kv:last-child { border-bottom: none; }
  .kv .label { color: #8b949e; }
  .log-box { background: #010409; border: 1px solid var(--border); border-radius: 4px;
             padding: 12px; max-height: 400px; overflow-y: auto; font-size: 0.78em;
             line-height: 1.5; white-space: pre-wrap; word-break: break-all; }
  .figures-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
  .figures-grid img { width: 100%; border-radius: 4px; border: 1px solid var(--border); cursor: pointer; }
  .figures-grid img:hover { border-color: var(--accent); }
  .refresh-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; font-size: 0.8em; color: #8b949e; }
  .refresh-bar button { background: var(--card); color: var(--fg); border: 1px solid var(--border);
                         border-radius: 4px; padding: 4px 12px; cursor: pointer; font-family: inherit; }
  .refresh-bar button:hover { border-color: var(--accent); }
  #auto-label { margin-left: 8px; }
  .history-table { width: 100%; font-size: 0.8em; border-collapse: collapse; }
  .history-table th, .history-table td { padding: 4px 8px; text-align: left; border-bottom: 1px solid var(--border); }
  .history-table th { color: #8b949e; }
  .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
           background: rgba(0,0,0,0.85); z-index: 100; justify-content: center; align-items: center; }
  .modal.active { display: flex; }
  .modal img { max-width: 90%; max-height: 90%; border-radius: 4px; }
</style>
</head>
<body>

<h1>NAT Pipeline Dashboard</h1>
<div id="data-path-bar" style="background:var(--card);border:1px solid var(--border);border-radius:4px;padding:8px 16px;margin-bottom:12px;font-size:0.85em;">
  Data folder: <span id="data-path" style="color:var(--accent)">loading...</span>
  &mdash; <span id="data-path-status"></span>
</div>

<div class="refresh-bar">
  <span>Last refresh: <span id="last-refresh">—</span>
    <label><input type="checkbox" id="auto-refresh" checked> <span id="auto-label">Auto (10s)</span></label>
  </span>
  <button onclick="refreshAll()">Refresh Now</button>
</div>

<div class="grid">
  <!-- State card -->
  <div class="card" id="state-card">
    <h2>Pipeline State</h2>
    <div id="state-content">Loading...</div>
  </div>

  <!-- Data card -->
  <div class="card" id="data-card">
    <h2>Data Collection</h2>
    <div id="data-content">Loading...</div>
  </div>

  <!-- Decision gate -->
  <div class="card" id="decision-card">
    <h2>Decision Gate</h2>
    <div id="decision-content">No analysis results yet.</div>
  </div>

  <!-- Vector results -->
  <div class="card" id="vectors-card">
    <h2>Vector Results</h2>
    <div id="vectors-content">No analysis results yet.</div>
  </div>

  <!-- Logs -->
  <div class="card card-full">
    <h2>Pipeline Log (last <span id="log-count">0</span> lines)</h2>
    <div class="log-box" id="log-box">No logs yet.</div>
  </div>

  <!-- Transition history -->
  <div class="card card-full" id="history-card">
    <h2>State Transition History</h2>
    <div id="history-content"></div>
  </div>

  <!-- Figures -->
  <div class="card card-full">
    <h2>Analysis Figures</h2>
    <div class="figures-grid" id="figures-grid">No figures generated yet.</div>
  </div>
</div>

<!-- Image modal -->
<div class="modal" id="img-modal" onclick="this.classList.remove('active')">
  <img id="modal-img" src="" alt="figure">
</div>

<script>
const REFRESH_INTERVAL = 10000;
let timer = null;

function kv(label, value) {
  return `<div class="kv"><span class="label">${label}</span><span>${value ?? '—'}</span></div>`;
}

function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(1) + ' MB';
}

async function fetchJSON(url) {
  try { const r = await fetch(url); return await r.json(); }
  catch(e) { return null; }
}

async function refreshState() {
  const s = await fetchJSON('/api/state');
  if (!s) return;
  const el = document.getElementById('state-content');
  const state = s.state || 'UNKNOWN';
  let html = `<div style="margin-bottom:12px"><span class="state-badge state-${state}">${state}</span></div>`;
  html += kv('Started', s.started_at);
  html += kv('Ingest started', s.ingest_started_at);
  html += kv('Ingest target end', s.ingest_target_end);
  html += kv('Ingest stopped', s.ingest_stopped_at);
  html += kv('Analysis started', s.analyze_started_at);
  html += kv('Analysis finished', s.analyze_finished_at);
  html += kv('Health OK / Fail', `${s.health_checks_ok ?? 0} / ${s.health_checks_fail ?? 0}`);
  html += kv('Restarts', s.restarts ?? 0);
  html += kv('Total rows', (s.total_rows ?? 0).toLocaleString());
  html += kv('Total files', s.total_files ?? 0);
  if (s.error) html += kv('Error', `<span style="color:var(--red)">${s.error}</span>`);
  el.innerHTML = html;

  // History
  const hist = s.history || [];
  if (hist.length > 0) {
    let tbl = '<table class="history-table"><tr><th>From</th><th>To</th><th>At</th><th>Message</th></tr>';
    hist.slice().reverse().forEach(h => {
      tbl += `<tr><td>${h.from}</td><td>${h.to}</td><td>${h.at || ''}</td><td>${h.message || ''}</td></tr>`;
    });
    tbl += '</table>';
    document.getElementById('history-content').innerHTML = tbl;
  }
}

async function refreshData() {
  const d = await fetchJSON('/api/data');
  if (!d) return;
  // Update top-level data path bar
  const pathEl = document.getElementById('data-path');
  const statusEl = document.getElementById('data-path-status');
  pathEl.textContent = d.path || '—';
  if (d.exists) {
    statusEl.innerHTML = `<span style="color:var(--green)">${d.n_files} files, ${formatBytes(d.total_bytes)}</span>`;
  } else {
    statusEl.innerHTML = '<span style="color:var(--red)">directory does not exist</span>';
  }
  // Update data card
  const el = document.getElementById('data-content');
  let html = kv('Path', d.path || '—');
  html += kv('Directory exists', d.exists ? 'Yes' : 'No');
  html += kv('Parquet files', d.n_files);
  html += kv('Total size', formatBytes(d.total_bytes));
  html += kv('Collection dates', (d.dates || []).join(', ') || '—');
  el.innerHTML = html;
}

async function refreshResults() {
  const r = await fetchJSON('/api/results');
  const del = document.getElementById('decision-content');
  const vel = document.getElementById('vectors-content');
  if (!r) { del.textContent = vel.textContent = 'No analysis results yet.'; return; }

  // Decision gate
  const g = r.decision_gate || {};
  let dhtml = `<div style="margin-bottom:8px;font-size:1.3em" class="decision-${g.decision || ''}">${g.decision || '—'}</div>`;
  dhtml += kv('Best vector', g.best_vector);
  dhtml += kv('Q1 (structure)', `${g.q1_pass ?? 0} / ${g.n_vectors_ok ?? 0}`);
  dhtml += kv('Q2 (stability)', `${g.q2_pass ?? 0} / ${g.n_vectors_ok ?? 0}`);
  dhtml += kv('Q3 (predictive)', `${g.q3_pass ?? 0} / ${g.n_vectors_ok ?? 0}`);
  dhtml += kv('Vectors OK / Total', `${g.n_vectors_ok ?? 0} / ${g.n_vectors_total ?? 0}`);
  if (r.bar_summary) {
    dhtml += kv('Bars', r.bar_summary.n_bars);
    dhtml += kv('Features', r.bar_summary.n_features);
    dhtml += kv('Time range', r.bar_summary.time_range);
  }
  del.innerHTML = dhtml;

  // Per-vector
  const vecs = r.vectors || {};
  if (Object.keys(vecs).length === 0) { vel.textContent = 'No vectors analyzed.'; return; }
  let vhtml = '<table class="history-table"><tr><th>Vector</th><th>Status</th><th>k</th><th>Silhouette</th><th>Q1</th><th>Q2</th><th>Q3</th></tr>';
  for (const [name, v] of Object.entries(vecs)) {
    const ok = v.status === 'ok';
    vhtml += `<tr><td>${name}</td><td style="color:${ok?'var(--green)':'var(--red)'}">${v.status}</td>`;
    vhtml += `<td>${ok ? v.best_k : '—'}</td><td>${ok ? v.silhouette?.toFixed(3) : '—'}</td>`;
    vhtml += `<td>${ok ? (v.q1_pass?'Y':'N') : '—'}</td><td>${ok ? (v.q2_pass?'Y':'N') : '—'}</td>`;
    vhtml += `<td>${ok ? (v.q3_pass?'Y':'N') : '—'}</td></tr>`;
  }
  vhtml += '</table>';
  vel.innerHTML = vhtml;
}

async function refreshLogs() {
  const data = await fetchJSON('/api/logs');
  if (!data) return;
  const box = document.getElementById('log-box');
  const lines = data.lines || [];
  document.getElementById('log-count').textContent = lines.length;
  box.textContent = lines.join('');
  box.scrollTop = box.scrollHeight;
}

async function refreshFigures() {
  const data = await fetchJSON('/api/figures');
  if (!data || !data.figures || data.figures.length === 0) return;
  const grid = document.getElementById('figures-grid');
  grid.innerHTML = data.figures.map(f =>
    `<img src="/figures/${f}" alt="${f}" title="${f}" onclick="openModal(this.src)">`
  ).join('');
}

function openModal(src) {
  document.getElementById('modal-img').src = src;
  document.getElementById('img-modal').classList.add('active');
}

async function refreshAll() {
  await Promise.all([refreshState(), refreshData(), refreshResults(), refreshLogs(), refreshFigures()]);
  document.getElementById('last-refresh').textContent = new Date().toLocaleTimeString();
}

function startAutoRefresh() {
  if (timer) clearInterval(timer);
  timer = setInterval(refreshAll, REFRESH_INTERVAL);
}

document.getElementById('auto-refresh').addEventListener('change', function() {
  if (this.checked) startAutoRefresh();
  else if (timer) { clearInterval(timer); timer = null; }
});

refreshAll();
startAutoRefresh();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server (stdlib only — no Flask/FastAPI dependency)
# ---------------------------------------------------------------------------

from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse


class DashboardHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler. Config injected via server.dashboard_config."""

    def log_message(self, format, *args):
        # Quieter logging — single line
        sys.stderr.write(f"[dashboard] {args[0]}\n")

    def _cfg(self) -> Dict[str, Any]:
        return self.server.dashboard_config  # type: ignore[attr-defined]

    def _json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html: str, status: int = 200) -> None:
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file_response(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        cfg = self._cfg()

        if path == "" or path == "/":
            self._html_response(DASHBOARD_HTML)

        elif path == "/api/state":
            self._json_response(read_state(cfg["_state_file"]))

        elif path == "/api/logs":
            qs = urllib.parse.parse_qs(parsed.query)
            n = min(int(qs.get("n", [str(DEFAULT_LOG_LINES)])[0]), MAX_LOG_LINES)
            lines = read_log_tail(cfg["_log_file"], n)
            self._json_response({"lines": lines, "n": len(lines)})

        elif path == "/api/results":
            report = read_report(cfg["_report_dir"])
            if report is None:
                self._json_response(None)
            else:
                self._json_response(report)

        elif path == "/api/figures":
            figs = list_figures(cfg["_report_dir"])
            self._json_response({"figures": figs})

        elif path == "/api/data":
            stats = data_dir_stats(cfg["_data_dir"], cfg["_project_root"])
            self._json_response(stats)

        elif path.startswith("/figures/"):
            name = path[len("/figures/"):]
            # Prevent path traversal
            if "/" in name or "\\" in name or ".." in name:
                self.send_error(403)
                return
            fig_path = Path(cfg["_report_dir"]) / name
            ext = fig_path.suffix.lower()
            ct = {".png": "image/png", ".svg": "image/svg+xml",
                  ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
            self._file_response(fig_path, ct)

        else:
            self.send_error(404)


def run_server(config: Dict[str, Any], host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Start the dashboard HTTP server."""
    server = HTTPServer((host, port), DashboardHandler)
    server.dashboard_config = config  # type: ignore[attr-defined]
    print(f"NAT Pipeline Dashboard: http://{host}:{port}")
    print(f"  State file : {config['_state_file']}")
    print(f"  Log file   : {config['_log_file']}")
    print(f"  Report dir : {config['_report_dir']}")
    print(f"  Data dir   : {config['_data_dir']}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard.")
        server.server_close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="NAT Pipeline Dashboard")
    parser.add_argument("--config", default="config/pipeline.toml",
                        help="Path to pipeline.toml (default: config/pipeline.toml)")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_server(cfg, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
