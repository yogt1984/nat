"""Priority queue for hypotheses, backed by a JSON file."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .hypothesis import Hypothesis

log = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/agent/hypotheses.json")


class HypothesisQueue:
    """Priority queue backed by SQLite or JSON.

    Dual-mode: pass ``store`` + ``agent`` for SQLite, or ``path`` for JSON.
    The queue is rebuilt on startup by filtering status == "queued"
    and sorting by descending priority.
    """

    def __init__(self, path: Path = DEFAULT_PATH, *,
                 store=None, agent: str = "agent"):
        self._store = store
        self._agent = agent
        self.path = path
        if path and not store:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._all: list[Hypothesis] = self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> list[Hypothesis]:
        if self._store:
            return [Hypothesis.from_dict(d)
                    for d in self._store.load_hypotheses(self._agent)]
        if self.path and self.path.exists():
            with open(self.path) as f:
                raw = json.load(f)
            return [Hypothesis.from_dict(h) for h in raw]
        return []

    def save(self) -> None:
        if self._store:
            for h in self._all:
                self._store.upsert_hypothesis(self._agent, h.to_dict())
            return
        with open(self.path, "w") as f:
            json.dump([h.to_dict() for h in self._all], f, indent=2, default=str)

    # -- queue operations ---------------------------------------------------

    def push(self, h: Hypothesis) -> None:
        """Add a hypothesis to the queue."""
        existing_claims = {x.claim for x in self._all if x.status in ("queued", "running")}
        if h.claim in existing_claims:
            log.debug("Skipping duplicate claim: %s", h.claim[:60])
            return
        self._all.append(h)
        if self._store:
            self._store.upsert_hypothesis(self._agent, h.to_dict())
        else:
            self.save()

    def pop(self, manifest: Optional[dict] = None) -> Optional[Hypothesis]:
        """Pop the highest-priority runnable hypothesis."""
        queued = [h for h in self._all if h.status == "queued"]
        queued.sort(key=lambda h: h.priority, reverse=True)
        for h in queued:
            if manifest and not self._is_runnable(h, manifest):
                continue
            h.status = "running"
            if self._store:
                self._store.upsert_hypothesis(self._agent, h.to_dict())
            else:
                self.save()
            return h
        return None

    def peek(self, n: int = 10) -> list[Hypothesis]:
        """Preview top-N queued hypotheses by priority."""
        queued = [h for h in self._all if h.status == "queued"]
        queued.sort(key=lambda h: h.priority, reverse=True)
        return queued[:n]

    def update(self, h: Hypothesis) -> None:
        """Update a hypothesis in-place and persist."""
        for i, existing in enumerate(self._all):
            if existing.id == h.id:
                self._all[i] = h
                if self._store:
                    self._store.upsert_hypothesis(self._agent, h.to_dict())
                else:
                    self.save()
                return
        self._all.append(h)
        if self._store:
            self._store.upsert_hypothesis(self._agent, h.to_dict())
        else:
            self.save()

    # -- queries ------------------------------------------------------------

    @property
    def queued(self) -> list[Hypothesis]:
        return [h for h in self._all if h.status == "queued"]

    @property
    def running(self) -> list[Hypothesis]:
        return [h for h in self._all if h.status == "running"]

    @property
    def passed(self) -> list[Hypothesis]:
        return [h for h in self._all if h.status in ("passed", "replicated")]

    @property
    def graveyard(self) -> list[Hypothesis]:
        return [h for h in self._all if h.status == "failed"]

    @property
    def depth(self) -> int:
        return len(self.queued)

    def by_generator(self) -> dict[str, int]:
        """Count queued hypotheses per generator."""
        counts: dict[str, int] = {}
        for h in self.queued:
            counts[h.generator] = counts.get(h.generator, 0) + 1
        return counts

    # -- data readiness check -----------------------------------------------

    @staticmethod
    def _is_runnable(h: Hypothesis, manifest: dict) -> bool:
        """Check if the hypothesis has enough data to run."""
        required = h.thresholds.get("min_hours", 4)
        required_symbols = h.thresholds.get("symbols", ["BTC"])
        for sym in required_symbols:
            available = manifest.get("symbols", {}).get(sym, {}).get("hours", 0)
            if available < required:
                return False
        return True
