#!/usr/bin/env python3
"""NAT Agent Dashboard — Read-only web view of agent state, hypotheses, and registry.

Serves:
    /                — Dashboard page
    /api/state       — Agent state JSON
    /api/queue       — Queued hypotheses
    /api/registry    — Registered signals
    /api/graveyard   — Failed hypotheses with reasons
    /api/heatmap     — (signal x gate) IC matrix data
    /api/cache       — Cache stats

Usage:
    python scripts/agent_dashboard.py                  # default port 8060
    python scripts/agent_dashboard.py --port 8060
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "data" / "agent"

# ---------------------------------------------------------------------------
# Data readers (pure functions)
# ---------------------------------------------------------------------------

def read_state() -> dict:
    path = STATE_DIR / "agent_state.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"phase": "UNKNOWN", "cycle_count": 0}


def read_hypotheses() -> list[dict]:
    path = STATE_DIR / "hypotheses.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def read_registry() -> list[dict]:
    path = STATE_DIR / "registry.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def read_gen_stats() -> dict:
    path = STATE_DIR / "generator_stats.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def get_queue(hypotheses: list[dict], limit: int = 50) -> list[dict]:
    queued = [h for h in hypotheses if h.get("status") == "queued"]
    queued.sort(key=lambda h: h.get("priority", 0), reverse=True)
    return queued[:limit]


def get_graveyard(hypotheses: list[dict]) -> list[dict]:
    return [h for h in hypotheses if h.get("status") == "failed"]


def get_tested(hypotheses: list[dict]) -> list[dict]:
    return [h for h in hypotheses if h.get("status") != "queued"]


def build_heatmap_data(hypotheses: list[dict]) -> dict:
    """Build (signal_feature x gate_feature) IC matrix from tested hypotheses."""
    # Parse claims to extract signal and gate features
    rows = []
    signal_features = set()
    gate_features = set()

    for h in hypotheses:
        if h.get("status") == "queued":
            continue
        claim = h.get("claim", "")
        thresholds = h.get("thresholds", {})
        gate = thresholds.get("regime_gate", "")

        # Extract signal feature from claim: "X gated by Y predicts..."
        parts = claim.split(" gated by ")
        if len(parts) < 2:
            continue
        signal_feat = parts[0].strip()
        signal_features.add(signal_feat)

        # Parse gate: "ent_book_shape<P20"
        gate_label = gate if gate else "ungated"
        gate_features.add(gate_label)

        # Extract IC from results
        ic = None
        results = h.get("results") or {}
        for gr in results.get("gate_results", []):
            msg = gr.get("msg", "")
            if "IC=" in msg:
                try:
                    ic = float(msg.split("IC=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
                break

        rows.append({
            "signal": signal_feat,
            "gate": gate_label,
            "ic": ic,
            "status": h.get("status"),
            "failure_reason": h.get("failure_reason"),
        })

    return {
        "signals": sorted(signal_features),
        "gates": sorted(gate_features),
        "cells": rows,
    }


def get_cache_stats() -> dict:
    cache_dir = STATE_DIR / "cache"
    if not cache_dir.exists():
        return {"entries": 0, "size_kb": 0}
    files = list(cache_dir.glob("*.meta.json"))
    total_size = sum(f.stat().st_size for f in cache_dir.glob("*.json"))
    return {"entries": len(files), "size_kb": round(total_size / 1024, 1)}


def get_summary(hypotheses: list[dict], registry: list[dict], state: dict) -> dict:
    """Compute summary stats for the dashboard header."""
    tested = [h for h in hypotheses if h.get("status") != "queued"]
    statuses = {}
    for h in tested:
        st = h.get("status", "unknown")
        reason = h.get("failure_reason")
        key = f"{st}({reason})" if reason else st
        statuses[key] = statuses.get(key, 0) + 1

    return {
        "phase": state.get("phase", "UNKNOWN"),
        "cycle_count": state.get("cycle_count", 0),
        "total_tested": len(tested),
        "total_queued": len(hypotheses) - len(tested),
        "total_registered": len(registry),
        "status_breakdown": statuses,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NAT Agent Dashboard</title>
<style>
:root {
    --bg: #0d1117; --fg: #c9d1d9; --card: #161b22; --border: #30363d;
    --accent: #58a6ff; --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --purple: #bc8cff; --orange: #f0883e;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', monospace; background: var(--bg); color: var(--fg); padding: 16px; }
h1 { font-size: 20px; margin-bottom: 12px; color: var(--accent); }
h2 { font-size: 15px; margin-bottom: 8px; color: var(--fg); border-bottom: 1px solid var(--border); padding-bottom: 4px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
.card-full { grid-column: 1 / -1; }
.stat { display: inline-block; margin-right: 20px; }
.stat-value { font-size: 22px; font-weight: bold; }
.stat-label { font-size: 11px; color: #8b949e; text-transform: uppercase; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.badge-idle { background: var(--border); color: var(--fg); }
.badge-execute { background: var(--accent); color: #000; }
.badge-sleeping { background: var(--purple); color: #000; }
.badge-stopped { background: var(--red); color: #fff; }
.badge-error { background: var(--red); color: #fff; }
.badge-pass { background: var(--green); color: #000; }
.badge-fail { background: var(--red); color: #fff; }
.badge-redundant { background: var(--orange); color: #000; }
.badge-queued { background: var(--border); color: var(--fg); }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 4px 8px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 10px; }
td { font-family: monospace; }
.heatmap { display: grid; gap: 2px; margin-top: 8px; }
.hm-cell { width: 100%; aspect-ratio: 1; border-radius: 3px; display: flex; align-items: center;
           justify-content: center; font-size: 9px; font-weight: bold; cursor: default; }
.hm-label { font-size: 10px; color: #8b949e; text-align: center; overflow: hidden;
            text-overflow: ellipsis; white-space: nowrap; }
.refresh-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.refresh-bar button { background: var(--accent); color: #000; border: none; padding: 4px 12px;
                      border-radius: 4px; cursor: pointer; font-size: 12px; }
.refresh-bar label { font-size: 11px; color: #8b949e; }
.breakdown { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.breakdown-item { font-size: 11px; }
</style>
</head>
<body>

<div class="refresh-bar">
    <h1>NAT Agent Dashboard</h1>
    <button onclick="refreshAll()">Refresh</button>
    <label><input type="checkbox" id="autoRefresh" checked> Auto (10s)</label>
    <span id="lastRefresh" style="font-size:11px;color:#8b949e;"></span>
</div>

<div class="grid">
    <!-- Summary stats -->
    <div class="card card-full" id="summary-card">
        <div id="summary">Loading...</div>
    </div>

    <!-- Registry -->
    <div class="card card-full">
        <h2>Registry (Validated Signals)</h2>
        <table><thead><tr><th>IC</th><th>Gate</th><th>Signal</th><th>Symbols</th><th>Discovered</th></tr></thead>
        <tbody id="registry-body"></tbody></table>
    </div>

    <!-- Heatmap -->
    <div class="card card-full">
        <h2>Signal x Gate IC Heatmap</h2>
        <div id="heatmap">Loading...</div>
    </div>

    <!-- Graveyard -->
    <div class="card">
        <h2>Graveyard (Failed)</h2>
        <table><thead><tr><th>Reason</th><th>Claim</th></tr></thead>
        <tbody id="graveyard-body"></tbody></table>
    </div>

    <!-- Queue -->
    <div class="card">
        <h2>Queue (Top 20)</h2>
        <table><thead><tr><th>Pri</th><th>Generator</th><th>Claim</th></tr></thead>
        <tbody id="queue-body"></tbody></table>
    </div>

    <!-- Generator stats -->
    <div class="card">
        <h2>Generator Performance</h2>
        <table><thead><tr><th>Generator</th><th>Attempts</th><th>Successes</th><th>Hit Rate</th></tr></thead>
        <tbody id="genstats-body"></tbody></table>
    </div>

    <!-- Cache stats -->
    <div class="card">
        <h2>Cache</h2>
        <div id="cache-stats">Loading...</div>
    </div>
</div>

<script>
const API = '';

function phaseBadge(phase) {
    const cls = {IDLE:'idle',EXECUTE:'execute',SLEEPING:'sleeping',STOPPED:'stopped',ERROR:'error'}[phase]||'idle';
    return `<span class="badge badge-${cls}">${phase}</span>`;
}

function statusBadge(status, reason) {
    if (status === 'replicated') return '<span class="badge badge-pass">REGISTERED</span>';
    if (reason === 'redundant') return '<span class="badge badge-redundant">REDUNDANT</span>';
    if (status === 'failed') return `<span class="badge badge-fail">${(reason||'failed').toUpperCase()}</span>`;
    return `<span class="badge badge-queued">${status}</span>`;
}

function icColor(ic, status) {
    if (status === 'replicated') return '#3fb950';
    if (status === 'failed') return '#f85149';
    if (ic === null || ic === undefined) return '#30363d';
    // Green gradient for IC
    const intensity = Math.min(Math.max((ic - 0.3) / 0.4, 0), 1);
    const r = Math.round(60 + (1 - intensity) * 180);
    const g = Math.round(60 + intensity * 130);
    return `rgb(${r},${g},60)`;
}

async function refreshSummary() {
    const [state, hyps, reg] = await Promise.all([
        fetch(API+'/api/state').then(r=>r.json()),
        fetch(API+'/api/graveyard').then(r=>r.json()),
        fetch(API+'/api/registry').then(r=>r.json()),
    ]);
    // Also fetch full summary
    const summary = await fetch(API+'/api/state').then(r=>r.json());
    const tested = summary.total_hypotheses_tested || 0;
    const registered = reg.length;
    const queued = summary._queue_depth || 0;

    document.getElementById('summary').innerHTML = `
        <div class="stat"><div class="stat-value">${phaseBadge(summary.phase)}</div><div class="stat-label">Status</div></div>
        <div class="stat"><div class="stat-value">${summary.cycle_count||0}</div><div class="stat-label">Cycles</div></div>
        <div class="stat"><div class="stat-value">${tested}</div><div class="stat-label">Tested</div></div>
        <div class="stat"><div class="stat-value">${registered}</div><div class="stat-label">Registered</div></div>
        <div class="stat"><div class="stat-value">${queued}</div><div class="stat-label">Queued</div></div>
    `;
}

async function refreshRegistry() {
    const data = await fetch(API+'/api/registry').then(r=>r.json());
    const rows = data.map(s => `<tr>
        <td style="color:var(--green);font-weight:bold">${(s.expected_ic||0).toFixed(3)}</td>
        <td>${s.regime_gate||'none'}</td>
        <td>${(s.name||'').substring(0,60)}</td>
        <td>${(s.symbols||[]).join(', ')}</td>
        <td>${s.discovery_date||''}</td>
    </tr>`).join('');
    document.getElementById('registry-body').innerHTML = rows || '<tr><td colspan=5 style="color:#8b949e">No signals registered</td></tr>';
}

async function refreshHeatmap() {
    const data = await fetch(API+'/api/heatmap').then(r=>r.json());
    if (!data.signals.length || !data.gates.length) {
        document.getElementById('heatmap').innerHTML = '<p style="color:#8b949e">No tested hypotheses yet</p>';
        return;
    }
    // Build lookup
    const lookup = {};
    data.cells.forEach(c => { lookup[c.signal+'|'+c.gate] = c; });

    const nCols = data.gates.length + 1;
    let html = `<div class="heatmap" style="grid-template-columns: 140px repeat(${data.gates.length}, 1fr);">`;
    // Header row
    html += '<div class="hm-label"></div>';
    data.gates.forEach(g => { html += `<div class="hm-label" title="${g}">${g.replace(/_/g,' ').substring(0,18)}</div>`; });
    // Data rows
    data.signals.forEach(sig => {
        html += `<div class="hm-label" title="${sig}">${sig.replace(/_/g,' ').substring(0,18)}</div>`;
        data.gates.forEach(gate => {
            const cell = lookup[sig+'|'+gate];
            const ic = cell ? cell.ic : null;
            const st = cell ? cell.status : null;
            const bg = icColor(ic, st);
            const label = ic !== null ? ic.toFixed(2) : '';
            const title = cell ? `${sig} | ${gate}\\nIC=${ic}\\nStatus: ${st}${cell.failure_reason?' ('+cell.failure_reason+')':''}` : 'untested';
            html += `<div class="hm-cell" style="background:${bg}" title="${title}">${label}</div>`;
        });
    });
    html += '</div>';
    document.getElementById('heatmap').innerHTML = html;
}

async function refreshGraveyard() {
    const data = await fetch(API+'/api/graveyard').then(r=>r.json());
    const rows = data.map(h => `<tr>
        <td>${statusBadge(h.status, h.failure_reason)}</td>
        <td>${(h.claim||'').substring(0,55)}</td>
    </tr>`).join('');
    document.getElementById('graveyard-body').innerHTML = rows || '<tr><td colspan=2 style="color:#8b949e">No failures</td></tr>';
}

async function refreshQueue() {
    const data = await fetch(API+'/api/queue').then(r=>r.json());
    const rows = data.slice(0,20).map(h => `<tr>
        <td>${(h.priority||0).toFixed(2)}</td>
        <td>${h.generator||''}</td>
        <td>${(h.claim||'').substring(0,50)}</td>
    </tr>`).join('');
    document.getElementById('queue-body').innerHTML = rows || '<tr><td colspan=3 style="color:#8b949e">Queue empty</td></tr>';
}

async function refreshGenStats() {
    const data = await fetch('/api/state').then(r=>r.json());
    // gen_stats might be in a separate endpoint; use state for now
    const gs = data._gen_stats || {};
    let rows = '';
    for (const [name, stats] of Object.entries(gs)) {
        const attempts = stats.attempts || 0;
        const successes = stats.successes || 0;
        const rate = attempts > 0 ? ((successes/attempts)*100).toFixed(0)+'%' : '-';
        rows += `<tr><td>${name}</td><td>${attempts}</td><td>${successes}</td><td>${rate}</td></tr>`;
    }
    document.getElementById('genstats-body').innerHTML = rows || '<tr><td colspan=4 style="color:#8b949e">No stats yet</td></tr>';
}

async function refreshCache() {
    const data = await fetch(API+'/api/cache').then(r=>r.json());
    document.getElementById('cache-stats').innerHTML = `
        <div class="stat"><div class="stat-value">${data.entries}</div><div class="stat-label">Entries</div></div>
        <div class="stat"><div class="stat-value">${data.size_kb} KB</div><div class="stat-label">Size</div></div>
    `;
}

async function refreshAll() {
    await Promise.all([refreshSummary(), refreshRegistry(), refreshHeatmap(),
                       refreshGraveyard(), refreshQueue(), refreshGenStats(), refreshCache()]);
    document.getElementById('lastRefresh').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

refreshAll();
setInterval(() => { if (document.getElementById('autoRefresh').checked) refreshAll(); }, 10000);
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class AgentDashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(f"[agent-dashboard] {args[0]}\n")

    def _json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html: str) -> None:
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "" or path == "/":
            self._html_response(DASHBOARD_HTML)

        elif path == "/api/state":
            state = read_state()
            # Enrich with queue depth and gen stats
            hyps = read_hypotheses()
            state["_queue_depth"] = len([h for h in hyps if h.get("status") == "queued"])
            state["_gen_stats"] = read_gen_stats()
            self._json_response(state)

        elif path == "/api/queue":
            hyps = read_hypotheses()
            self._json_response(get_queue(hyps))

        elif path == "/api/registry":
            self._json_response(read_registry())

        elif path == "/api/graveyard":
            hyps = read_hypotheses()
            self._json_response(get_graveyard(hyps))

        elif path == "/api/heatmap":
            hyps = read_hypotheses()
            self._json_response(build_heatmap_data(hyps))

        elif path == "/api/cache":
            self._json_response(get_cache_stats())

        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description="NAT Agent Dashboard")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), AgentDashboardHandler)
    print(f"Agent dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
