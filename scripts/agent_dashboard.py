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
    /api/graveyard_sankey — Generator → failure_reason flow data
    /api/cross_symbol_ic  — Per-symbol IC for cross-symbol scatter
    /api/ic_decay    — IC history curves for registered signals
    /api/correlation_matrix — Pairwise correlation across registered signals

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


def build_graveyard_sankey(hypotheses: list[dict]) -> dict:
    """Build generator → failure_reason flow counts for Sankey diagram."""
    flows: dict[tuple[str, str], int] = {}
    generators = set()
    reasons = set()
    for h in hypotheses:
        if h.get("status") != "failed":
            continue
        gen = h.get("generator", "unknown")
        reason = h.get("failure_reason", "unknown")
        generators.add(gen)
        reasons.add(reason)
        flows[(gen, reason)] = flows.get((gen, reason), 0) + 1

    links = []
    for (gen, reason), count in sorted(flows.items(), key=lambda x: -x[1]):
        links.append({"source": gen, "target": reason, "count": count})

    return {
        "generators": sorted(generators),
        "reasons": sorted(reasons),
        "links": links,
    }


def build_cross_symbol_ic(hypotheses: list[dict]) -> dict:
    """Extract per-symbol IC from hypothesis results for cross-symbol scatter."""
    points = []
    for h in hypotheses:
        if h.get("status") == "queued":
            continue
        results = h.get("results") or {}
        sym_results = results.get("symbol_replication") or {}
        if not sym_results:
            continue
        # Extract IC per symbol
        ics = {}
        for symbol, data in sym_results.items():
            if isinstance(data, dict):
                ic = data.get("ic")
                if ic is not None:
                    ics[symbol] = ic
            elif isinstance(data, (int, float, bool)):
                # Older format: just pass/fail, try to get IC from gate_results
                pass
        if len(ics) >= 2:
            points.append({
                "id": h.get("id", ""),
                "generator": h.get("generator", ""),
                "claim": h.get("claim", "")[:60],
                "ics": ics,
            })

    return {"points": points}


def build_ic_decay_data(registry: list[dict]) -> dict:
    """Extract IC history curves from registered signals."""
    curves = []
    for sig in registry:
        if sig.get("status") == "retired":
            continue
        ic_history = sig.get("ic_history", [])
        if not ic_history:
            continue
        discovery_ic = sig.get("expected_ic", 0)
        # ic_history entries are {date, ic} dicts
        points = []
        for entry in ic_history:
            if isinstance(entry, dict):
                points.append({"date": entry.get("date", ""), "ic": entry.get("ic", 0)})
            elif isinstance(entry, (int, float)):
                points.append({"date": "", "ic": entry})
        curves.append({
            "name": sig.get("name", "")[:50],
            "discovery_ic": discovery_ic,
            "retirement_threshold": discovery_ic * 0.5,
            "points": points,
        })

    return {"curves": curves}


def build_correlation_matrix(registry: list[dict]) -> dict:
    """Build pairwise correlation data from registry signals."""
    signals = [s for s in registry if s.get("status") != "retired"]
    if len(signals) < 2:
        return {"signals": [], "matrix": []}

    names = [s.get("name", "")[:40] for s in signals]
    corr_data = []
    for s in signals:
        corr_with = s.get("correlation_with", {})
        corr_data.append(corr_with)

    # Build NxN matrix from correlation_with fields
    matrix = []
    for i, sig_i in enumerate(signals):
        row = []
        for j, sig_j in enumerate(signals):
            if i == j:
                row.append(1.0)
            else:
                # Check if i has correlation with j
                corr_i = sig_i.get("correlation_with", {})
                corr_j = sig_j.get("correlation_with", {})
                name_j = sig_j.get("name", "")
                name_i = sig_i.get("name", "")
                c = corr_i.get(name_j, corr_j.get(name_i))
                row.append(c)
        matrix.append(row)

    return {"signals": names, "matrix": matrix}


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

    <!-- 3.4 Graveyard Sankey -->
    <div class="card card-full">
        <h2>Graveyard Analysis (Generator &rarr; Failure Reason)</h2>
        <div id="sankey" style="min-height:200px;">Loading...</div>
    </div>

    <!-- 3.5 Cross-symbol IC scatter -->
    <div class="card card-full">
        <h2>Cross-Symbol IC Scatter</h2>
        <div id="cross-symbol" style="display:flex;gap:12px;flex-wrap:wrap;">Loading...</div>
    </div>

    <!-- 3.6 IC decay curves -->
    <div class="card card-full">
        <h2>IC Decay Curves</h2>
        <svg id="ic-decay" width="100%" height="250" style="overflow:visible;"></svg>
    </div>

    <!-- 3.7 Correlation clustermap -->
    <div class="card card-full">
        <h2>Signal Correlation Matrix</h2>
        <div id="corr-matrix">Loading...</div>
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

// --- 3.4 Graveyard Sankey ---
async function refreshSankey() {
    const data = await fetch(API+'/api/graveyard_sankey').then(r=>r.json());
    if (!data.links.length) {
        document.getElementById('sankey').innerHTML = '<p style="color:#8b949e">No failures yet</p>';
        return;
    }
    // Render as horizontal bar-flow diagram
    const maxCount = Math.max(...data.links.map(l=>l.count));
    const colors = ['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff','#f0883e','#8b949e'];
    const genColors = {};
    data.generators.forEach((g,i) => genColors[g] = colors[i % colors.length]);

    let html = '<div style="display:flex;flex-direction:column;gap:4px;">';
    data.links.forEach(l => {
        const pct = Math.max((l.count / maxCount) * 100, 8);
        const c = genColors[l.source] || '#8b949e';
        html += `<div style="display:flex;align-items:center;gap:8px;font-size:12px;">
            <span style="width:120px;text-align:right;color:${c};font-weight:600">${l.source}</span>
            <span style="flex:none;color:#8b949e">&rarr;</span>
            <div style="background:${c}33;border-left:3px solid ${c};height:20px;width:${pct}%;display:flex;align-items:center;padding-left:6px;border-radius:0 3px 3px 0;">
                <span style="font-weight:bold;color:${c}">${l.count}</span>
            </div>
            <span style="color:#8b949e">${l.target}</span>
        </div>`;
    });
    html += '</div>';
    document.getElementById('sankey').innerHTML = html;
}

// --- 3.5 Cross-symbol IC scatter ---
async function refreshCrossSymbol() {
    const data = await fetch(API+'/api/cross_symbol_ic').then(r=>r.json());
    if (!data.points.length) {
        document.getElementById('cross-symbol').innerHTML = '<p style="color:#8b949e">No cross-symbol data yet</p>';
        return;
    }
    const symbols = ['BTC','ETH','SOL'];
    const pairs = [];
    for (let i=0; i<symbols.length; i++)
        for (let j=i+1; j<symbols.length; j++)
            pairs.push([symbols[i], symbols[j]]);

    const genColors = {};
    const colors = ['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff','#f0883e'];
    let ci = 0;

    let html = '';
    pairs.forEach(([sx, sy]) => {
        const W = 220, H = 220, pad = 35;
        let svg = `<svg width="${W}" height="${H}" style="background:var(--card);border:1px solid var(--border);border-radius:6px;">`;
        // Axes
        svg += `<line x1="${pad}" y1="${H-pad}" x2="${W-5}" y2="${H-pad}" stroke="#30363d"/>`;
        svg += `<line x1="${pad}" y1="5" x2="${pad}" y2="${H-pad}" stroke="#30363d"/>`;
        svg += `<text x="${W/2}" y="${H-5}" fill="#8b949e" font-size="10" text-anchor="middle">${sx} IC</text>`;
        svg += `<text x="10" y="${H/2}" fill="#8b949e" font-size="10" text-anchor="middle" transform="rotate(-90,10,${H/2})">${sy} IC</text>`;
        // Diagonal
        svg += `<line x1="${pad}" y1="${H-pad}" x2="${W-5}" y2="5" stroke="#30363d" stroke-dasharray="4"/>`;
        // Points
        data.points.forEach(p => {
            const icx = p.ics[sx], icy = p.ics[sy];
            if (icx===undefined || icy===undefined) return;
            if (!genColors[p.generator]) genColors[p.generator] = colors[ci++ % colors.length];
            const x = pad + (icx + 0.5) * (W - pad - 5);
            const y = (H - pad) - (icy + 0.5) * (H - pad - 5);
            svg += `<circle cx="${x}" cy="${y}" r="4" fill="${genColors[p.generator]}" opacity="0.8">
                <title>${p.claim}\\n${sx}=${icx?.toFixed(3)} ${sy}=${icy?.toFixed(3)}</title></circle>`;
        });
        svg += '</svg>';
        html += svg;
    });
    document.getElementById('cross-symbol').innerHTML = html;
}

// --- 3.6 IC decay curves ---
async function refreshICDecay() {
    const data = await fetch(API+'/api/ic_decay').then(r=>r.json());
    const svg = document.getElementById('ic-decay');
    if (!data.curves.length) {
        svg.innerHTML = '<text x="20" y="30" fill="#8b949e" font-size="13">No IC history data</text>';
        return;
    }
    const W = svg.clientWidth || 700, H = 250, pad = 45;
    const colors = ['#58a6ff','#3fb950','#d29922','#bc8cff','#f0883e','#f85149'];

    // Find IC range
    let allICs = [];
    data.curves.forEach(c => c.points.forEach(p => allICs.push(p.ic)));
    const minIC = Math.min(0, ...allICs);
    const maxIC = Math.max(0.1, ...allICs) * 1.1;

    let content = '';
    // Y axis ticks
    for (let t = 0; t <= 1; t += 0.25) {
        const ic = minIC + t * (maxIC - minIC);
        const y = H - pad - t * (H - 2 * pad);
        content += `<line x1="${pad}" y1="${y}" x2="${W-5}" y2="${y}" stroke="#30363d" stroke-dasharray="2"/>`;
        content += `<text x="${pad-5}" y="${y+3}" fill="#8b949e" font-size="9" text-anchor="end">${ic.toFixed(2)}</text>`;
    }

    data.curves.forEach((curve, ci) => {
        const col = colors[ci % colors.length];
        const n = curve.points.length;
        if (n < 2) return;
        // Threshold line
        const thY = H - pad - ((curve.retirement_threshold - minIC) / (maxIC - minIC)) * (H - 2 * pad);
        content += `<line x1="${pad}" y1="${thY}" x2="${W-5}" y2="${thY}" stroke="${col}" stroke-dasharray="4" opacity="0.4"/>`;
        // Curve
        let path = '';
        curve.points.forEach((p, i) => {
            const x = pad + (i / (n - 1)) * (W - pad - 5);
            const y = H - pad - ((p.ic - minIC) / (maxIC - minIC)) * (H - 2 * pad);
            path += (i === 0 ? 'M' : 'L') + `${x},${y}`;
        });
        content += `<path d="${path}" fill="none" stroke="${col}" stroke-width="2"/>`;
        // Legend
        content += `<text x="${W - 5}" y="${20 + ci * 14}" fill="${col}" font-size="10" text-anchor="end">${curve.name.substring(0,35)}</text>`;
    });
    svg.innerHTML = content;
}

// --- 3.7 Correlation clustermap ---
async function refreshCorrMatrix() {
    const data = await fetch(API+'/api/correlation_matrix').then(r=>r.json());
    if (!data.signals.length) {
        document.getElementById('corr-matrix').innerHTML = '<p style="color:#8b949e">Need 2+ signals for correlation</p>';
        return;
    }
    const n = data.signals.length;
    const cellSz = Math.min(50, Math.max(25, 300 / n));

    function corrColor(c) {
        if (c === null || c === undefined) return '#30363d';
        // Blue (negative) → gray (0) → red (positive)
        const v = Math.max(-1, Math.min(1, c));
        if (v >= 0) {
            const t = v;
            return `rgb(${60+Math.round(t*195)},${60+Math.round((1-t)*60)},60)`;
        } else {
            const t = -v;
            return `rgb(${60+Math.round((1-t)*20)},${60+Math.round((1-t)*60)},${60+Math.round(t*195)})`;
        }
    }

    let html = `<div class="heatmap" style="grid-template-columns: 140px repeat(${n}, ${cellSz}px);">`;
    // Header
    html += '<div class="hm-label"></div>';
    data.signals.forEach(s => html += `<div class="hm-label" title="${s}" style="writing-mode:vertical-rl;transform:rotate(180deg);height:80px;">${s.substring(0,15)}</div>`);
    // Rows
    data.signals.forEach((sig, i) => {
        html += `<div class="hm-label" title="${sig}">${sig.substring(0,18)}</div>`;
        data.matrix[i].forEach((c, j) => {
            const bg = corrColor(c);
            const label = c !== null && c !== undefined ? (c === 1 ? '1.0' : c.toFixed(2)) : '';
            html += `<div class="hm-cell" style="background:${bg};width:${cellSz}px;height:${cellSz}px;" title="${sig} vs ${data.signals[j]}: ${label}">${label}</div>`;
        });
    });
    html += '</div>';
    document.getElementById('corr-matrix').innerHTML = html;
}

async function refreshAll() {
    await Promise.all([refreshSummary(), refreshRegistry(), refreshHeatmap(),
                       refreshGraveyard(), refreshQueue(), refreshGenStats(), refreshCache(),
                       refreshSankey(), refreshCrossSymbol(), refreshICDecay(), refreshCorrMatrix()]);
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

        elif path == "/api/graveyard_sankey":
            hyps = read_hypotheses()
            self._json_response(build_graveyard_sankey(hyps))

        elif path == "/api/cross_symbol_ic":
            hyps = read_hypotheses()
            self._json_response(build_cross_symbol_ic(hyps))

        elif path == "/api/ic_decay":
            registry = read_registry()
            self._json_response(build_ic_decay_data(registry))

        elif path == "/api/correlation_matrix":
            registry = read_registry()
            self._json_response(build_correlation_matrix(registry))

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
