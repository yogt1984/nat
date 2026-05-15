"""Computation cache for deterministic nat commands.

Caches JSON report output keyed by (command, data_dir, symbol). Commands like
`nat spannung regime --data X --symbol Y` always produce the same report for
the same inputs, so re-running them is pure waste.

Storage: data/agent/cache/<hex_key>.json with metadata sidecar.
TTL-based expiry (default 7 days).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Commands whose output is deterministic for a given (data_dir, symbol)
CACHEABLE_PREFIXES = [
    "spannung regime",
    "spannung spectral",
    "spannung backtest",
    "profile scalp",
]


def _is_cacheable(cmd_parts: list[str]) -> bool:
    """Check if a nat command produces deterministic output."""
    cmd_str = " ".join(cmd_parts)
    return any(prefix in cmd_str for prefix in CACHEABLE_PREFIXES)


def _cache_key(cmd_parts: list[str]) -> str:
    """Compute a stable hash key for a command invocation.

    Normalizes by sorting flags so that argument order doesn't matter.
    Only the command and its value-bearing flags contribute to the key.
    """
    # Extract the meaningful parts: subcommand + flag-value pairs
    parts = list(cmd_parts)
    # Separate positional args from flag-value pairs
    subcommand = []
    flags = {}
    i = 0
    while i < len(parts):
        if parts[i].startswith("--"):
            key = parts[i]
            if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                flags[key] = parts[i + 1]
                i += 2
            else:
                flags[key] = ""
                i += 1
        else:
            subcommand.append(parts[i])
            i += 1

    # Build a canonical string: subcommand + sorted flags
    canonical = " ".join(subcommand)
    for k in sorted(flags.keys()):
        canonical += f" {k} {flags[k]}"

    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class ReportCache:
    """File-backed cache for nat command reports."""

    def __init__(self, cache_dir: Path, ttl_seconds: int = 7 * 86400):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def get(self, cmd_parts: list[str]) -> Optional[dict]:
        """Return cached report if available and not expired."""
        if not _is_cacheable(cmd_parts):
            return None

        key = _cache_key(cmd_parts)
        meta_path = self.cache_dir / f"{key}.meta.json"
        data_path = self.cache_dir / f"{key}.json"

        if not meta_path.exists() or not data_path.exists():
            self._misses += 1
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._misses += 1
            return None

        # Check TTL
        cached_at = meta.get("cached_at", 0)
        if time.time() - cached_at > self.ttl_seconds:
            log.debug("Cache expired for %s (age=%.0fh)",
                      " ".join(cmd_parts), (time.time() - cached_at) / 3600)
            self._misses += 1
            return None

        try:
            with open(data_path) as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._misses += 1
            return None

        self._hits += 1
        log.info("Cache HIT for: nat %s [key=%s]", " ".join(cmd_parts), key)
        return report

    def put(self, cmd_parts: list[str], report: dict) -> None:
        """Store a report in the cache."""
        if not _is_cacheable(cmd_parts):
            return

        key = _cache_key(cmd_parts)
        data_path = self.cache_dir / f"{key}.json"
        meta_path = self.cache_dir / f"{key}.meta.json"

        meta = {
            "cmd": " ".join(cmd_parts),
            "key": key,
            "cached_at": time.time(),
        }

        with open(data_path, "w") as f:
            json.dump(report, f)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        log.debug("Cached: nat %s [key=%s]", " ".join(cmd_parts), key)

    def invalidate(self, cmd_parts: list[str]) -> bool:
        """Remove a specific cache entry. Returns True if entry existed."""
        key = _cache_key(cmd_parts)
        data_path = self.cache_dir / f"{key}.json"
        meta_path = self.cache_dir / f"{key}.meta.json"
        existed = data_path.exists()
        data_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return existed

    def clear(self) -> int:
        """Remove all cache entries. Returns count of entries removed."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count // 2  # each entry is 2 files (data + meta)

    def evict_expired(self) -> int:
        """Remove expired entries. Returns count evicted."""
        now = time.time()
        evicted = 0
        for meta_path in self.cache_dir.glob("*.meta.json"):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if now - meta.get("cached_at", 0) > self.ttl_seconds:
                    key = meta_path.stem.replace(".meta", "")
                    meta_path.unlink()
                    data_path = self.cache_dir / f"{key}.json"
                    data_path.unlink(missing_ok=True)
                    evicted += 1
            except (json.JSONDecodeError, OSError):
                meta_path.unlink(missing_ok=True)
                evicted += 1
        return evicted

    @property
    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "entries": len(list(self.cache_dir.glob("*.meta.json"))),
        }
