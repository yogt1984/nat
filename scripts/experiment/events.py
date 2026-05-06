"""
Task 5: Simple append-only event log.

Events stored as JSON lines in data/experiment_events.jsonl.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

DEFAULT_EVENTS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "experiment_events.jsonl"


def log_event(event_type: str, message: str, path: Path = DEFAULT_EVENTS_PATH) -> Dict[str, str]:
    """Append one event to the log. Returns the event dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "type": event_type,
        "msg": message,
    }
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")
    return event


def read_events(n: int = 50, path: Path = DEFAULT_EVENTS_PATH) -> List[Dict[str, str]]:
    """Read the last N events. Most recent first."""
    if not path.exists():
        return []
    lines = path.read_text().strip().split("\n")
    lines = [l for l in lines if l.strip()]
    events = []
    for line in lines[-n:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    events.reverse()
    return events
