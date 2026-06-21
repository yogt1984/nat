"""`nat health` — comprehensive system health check (all components)."""

from __future__ import annotations

import json as _json
import time
from datetime import datetime

from cli.common import (
    ROOT, DATA_DEFAULT,
    G, R, Y,
    _pid, _sh, _json_mode, _output, _banner, _p,
)


def cmd_health(args):
    """Comprehensive health check across all components."""
    health = {"timestamp": datetime.now().isoformat(), "components": {}}

    # Ingestor
    pid = _pid()
    health["components"]["ingestor"] = {"healthy": pid is not None, "pid": pid}

    # Data freshness per symbol
    data_health = {}
    if DATA_DEFAULT.exists():
        for sym in ["BTC", "ETH", "SOL"]:
            sym_files = list(DATA_DEFAULT.rglob(f"*{sym}*.parquet"))
            if sym_files:
                newest = max(f.stat().st_mtime for f in sym_files)
                age_s = time.time() - newest
                data_health[sym] = {"files": len(sym_files), "last_write_s": round(age_s, 1),
                                    "healthy": age_s < 600}
            else:
                data_health[sym] = {"files": 0, "healthy": False}
    health["components"]["data"] = data_health

    # Agent phase
    agent_state_path = ROOT / "data" / "agent" / "agent_state.json"
    if agent_state_path.exists():
        try:
            state = _json.loads(agent_state_path.read_text())
            health["components"]["agent"] = {
                "healthy": True, "phase": state.get("phase", "UNKNOWN"),
                "cycle": state.get("cycle", 0),
            }
        except _json.JSONDecodeError:
            health["components"]["agent"] = {"healthy": False, "error": "corrupt state"}
    else:
        health["components"]["agent"] = {"healthy": False, "error": "no state file"}

    # Pipeline state
    pipe_path = ROOT / "data" / "pipeline_state.json"
    if pipe_path.exists():
        try:
            state = _json.loads(pipe_path.read_text())
            health["components"]["pipeline"] = {
                "healthy": True, "state": state.get("state", "UNKNOWN"),
            }
        except _json.JSONDecodeError:
            health["components"]["pipeline"] = {"healthy": False, "error": "corrupt state"}
    else:
        health["components"]["pipeline"] = {"state": "IDLE"}

    # Watchdog
    health["components"]["watchdog"] = {
        "active": "pgrep -x ing" in (_sh("crontab -l 2>/dev/null").stdout or "")
    }

    # Dashboard
    health["components"]["dashboard"] = {
        "running": _sh("tmux has-session -t nat-dashboard 2>/dev/null").returncode == 0
    }

    # Redis
    redis_ok = _sh("redis-cli ping 2>/dev/null").stdout.strip() == "PONG"
    health["components"]["redis"] = {"connected": redis_ok}

    # Overall
    health["healthy"] = all(
        c.get("healthy", c.get("active", c.get("connected", True)))
        for c in health["components"].values() if isinstance(c, dict) and "healthy" in c
    )

    if _json_mode(args):
        _output(health, args)
        return

    _banner("System Health")
    for name, info in health["components"].items():
        if isinstance(info, dict):
            ok = info.get("healthy", info.get("active", info.get("connected", None)))
            icon, color = ("*", G) if ok else ("x", R) if ok is False else ("-", Y)
            detail = ", ".join(f"{k}={v}" for k, v in info.items()
                               if k not in ("healthy", "active", "connected"))
            _p(icon, color, f"{name}: {detail}")
        else:
            # data_health is a dict of symbols
            for sym, sym_info in info.items():
                ok = sym_info.get("healthy", False)
                icon, color = ("*", G) if ok else ("x", R)
                _p(icon, color, f"data/{sym}: {sym_info.get('files', 0)} files, "
                   f"last write {sym_info.get('last_write_s', '?')}s ago")


def register(sub):
    # ── health ──
    sub.add_parser('health', help='Comprehensive system health check (all components)').set_defaults(func=cmd_health)


__all__ = ["cmd_health", "register"]
