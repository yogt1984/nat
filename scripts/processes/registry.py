"""
Process registry — discover and instantiate analytical processes by name.

Mirrors `scripts/algorithms/registry.py`.
"""

from __future__ import annotations

from typing import Type

from .base import Process

_REGISTRY: dict[str, Type[Process]] = {}


def register(cls: Type[Process]) -> Type[Process]:
    """Class decorator to register a process."""
    instance = cls()
    _REGISTRY[instance.name()] = cls
    return cls


def get_process(name: str, **kwargs) -> Process:
    """Instantiate a process by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown process '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)


def list_processes() -> list[str]:
    """List all registered process names."""
    return sorted(_REGISTRY.keys())


def list_processes_by_kind(kind: str) -> list[str]:
    """List registered process names filtered by kind ('evaluation'|'transform')."""
    return sorted(n for n, cls in _REGISTRY.items() if cls.kind == kind)
