"""Agent state machine and configuration loading.

Provides:
- AgentPhase enum — lifecycle phases for research agents
- AgentState class — persistent state backed by SQLite
- Config utilities — deep merge, validation, loading with inheritance
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger("nat.agent")


# ---------------------------------------------------------------------------
# Agent state (persistent)
# ---------------------------------------------------------------------------

class AgentPhase(str, Enum):
    IDLE = "IDLE"
    MANIFEST = "MANIFEST"
    GENERATE = "GENERATE"
    EXECUTE = "EXECUTE"
    MONITOR = "MONITOR"
    SLEEPING = "SLEEPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class AgentState:
    """Persistent agent state backed by SQLite (via StateStore)."""

    _DEFAULTS = {
        "phase": AgentPhase.IDLE.value,
        "cycle_count": 0,
        "total_hypotheses_tested": 0,
        "total_signals_registered": 0,
        "current_hypothesis": None,
        "started_at": None,
        "last_cycle_at": None,
    }

    def __init__(self, *, store, agent: str = "agent"):
        self._store = store
        self._agent = agent
        self._data = self._load()

    def _load(self) -> dict:
        loaded = self._store.load_state(self._agent)
        if loaded:
            loaded["history"] = self._store.load_history(
                self._agent, limit=200)
            return loaded
        return {**self._DEFAULTS, "history": []}

    def save(self) -> None:
        self._store.save_state(self._agent, self._data)

    def transition(self, phase: AgentPhase, msg: str = "") -> None:
        old = self._data["phase"]
        self._data["phase"] = phase.value
        entry = {
            "from": old, "to": phase.value,
            "at": datetime.now(timezone.utc).isoformat(),
            "msg": msg,
        }
        self._data.setdefault("history", []).append(entry)
        # Keep in-memory history bounded
        if len(self._data["history"]) > 500:
            self._data["history"] = self._data["history"][-200:]
        self._store.save_state(self._agent, self._data)
        self._store.append_history(self._agent, entry)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self.save()

    @property
    def phase(self) -> AgentPhase:
        return AgentPhase(self._data["phase"])


# ---------------------------------------------------------------------------
# Config inheritance
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (neither dict is mutated)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# Keys recognised at the top level and in common subsections.
# Unknown keys trigger a warning (not an error) so legacy configs keep working.
_KNOWN_TOP_KEYS = {
    "cycle_interval_s", "max_experiments_per_cycle", "max_cycle_runtime_s",
    "timeframe", "generators_enabled",
    "gates", "cost", "decay", "promotion", "symbols", "paths",
}
_KNOWN_GATE_KEYS = {
    "min_ic", "min_dIC", "min_coverage", "fdr_q",
    "min_walkforward_sign_consistency", "min_oos_dates", "min_symbols",
}


_REQUIRED_KEYS = {"cycle_interval_s", "max_experiments_per_cycle", "generators_enabled"}


def validate_config(config: dict, section: str) -> list[str]:
    """Validate agent config after defaults merge.

    Returns warnings for unknown keys.
    Raises ValueError if required keys are missing.
    """
    # Required-key check (hard error)
    missing = _REQUIRED_KEYS - config.keys()
    if missing:
        raise ValueError(
            f"[{section}] missing required keys: {sorted(missing)}"
        )

    # Unknown-key check (soft warning)
    warnings = []
    for key in config:
        if key not in _KNOWN_TOP_KEYS:
            warnings.append(f"[{section}] unknown key: {key!r}")
    gates = config.get("gates", {})
    for key in gates:
        if key not in _KNOWN_GATE_KEYS:
            warnings.append(f"[{section}.gates] unknown key: {key!r}")
    return warnings


def load_agent_config(config_path: Path, section: str,
                      base_config: dict) -> dict:
    """Load agent config with inheritance: base_config -> [defaults] -> [section].

    Deep-merges nested subsections (gates, decay, symbols, paths) so that
    a section only needs to override the keys that differ from [defaults].
    Injects symbols.primary from config/symbols.toml if not set.
    """
    if not config_path.exists():
        return dict(base_config)
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    defaults = raw.get("defaults", {})
    section_cfg = raw.get(section, {})
    merged = _deep_merge(dict(base_config), defaults)
    merged = _deep_merge(merged, section_cfg)

    # Inject canonical symbols if not provided by TOML
    if "symbols" not in merged or "primary" not in merged.get("symbols", {}):
        try:
            from config_utils import load_symbols
        except ImportError:
            from config_utils import load_symbols
        merged.setdefault("symbols", {})["primary"] = load_symbols()

    for w in validate_config(merged, section):
        log.warning("Config: %s", w)
    return merged
