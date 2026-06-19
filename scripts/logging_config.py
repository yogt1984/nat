"""Centralized logging configuration for NAT Python subsystems.

Provides:
- JSON-formatted log output for machine parsing
- File rotation (daily, configurable retention)
- Correlation ID context (cycle_id, hypothesis_id)
- Consistent format across all daemons and scripts

Usage:
    from logging_config import setup_logging, set_context

    setup_logging("nat.agent")                    # stderr + file
    setup_logging("nat.agent", file_only=True)    # file only
    set_context(cycle_id="CYC-001", hypothesis_id="HYP-SYS-abc123")
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Thread-local context for correlation IDs
_context = threading.local()

# Default log directory (relocatable: env → repo → XDG, via nat_paths)
try:
    import nat_paths
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nat_paths

_DEFAULT_LOG_DIR = nat_paths.data_root() / "logs"


def set_context(**kwargs) -> None:
    """Set correlation context for the current thread.

    Common keys: cycle_id, hypothesis_id, agent, symbol.
    """
    if not hasattr(_context, "data"):
        _context.data = {}
    _context.data.update(kwargs)


def clear_context() -> None:
    """Clear all correlation context."""
    _context.data = {}


def get_context() -> dict:
    """Get the current correlation context."""
    return getattr(_context, "data", {})


class JSONFormatter(logging.Formatter):
    """Emit structured JSON log lines with context fields."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge correlation context
        ctx = get_context()
        if ctx:
            entry["ctx"] = ctx

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable format for stderr with optional context."""

    FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    DATEFMT = "%H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        ctx = get_context()
        if ctx:
            ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items())
            return f"{base} [{ctx_str}]"
        return base


def setup_logging(
    name: str = "nat",
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    file_only: bool = False,
    json_stderr: bool = False,
    retention_days: int = 30,
) -> logging.Logger:
    """Configure logging for a NAT subsystem.

    Args:
        name: Logger name (e.g. "nat.agent", "nat.pipeline")
        level: Log level (default INFO)
        log_dir: Directory for log files (default data/logs/)
        file_only: If True, suppress stderr output
        json_stderr: If True, use JSON format on stderr (for production)
        retention_days: Number of daily log files to keep

    Returns:
        Configured logger instance.
    """
    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers on repeated calls
    logger.handlers.clear()

    # File handler: daily rotation, JSON format
    log_file = log_dir / "nat.jsonl"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        str(log_file),
        when="midnight",
        interval=1,
        backupCount=retention_days,
        utc=True,
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Stderr handler (unless file_only)
    if not file_only:
        stderr_handler = logging.StreamHandler(sys.stderr)
        if json_stderr:
            stderr_handler.setFormatter(JSONFormatter())
        else:
            stderr_handler.setFormatter(HumanFormatter())
        stderr_handler.setLevel(level)
        logger.addHandler(stderr_handler)

    # Don't propagate to root (avoids duplicate output from basicConfig)
    logger.propagate = False

    return logger
