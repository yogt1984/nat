"""
Algorithm registry — discover and instantiate algorithms by name.
"""

from __future__ import annotations

from typing import Type
from .base import MicrostructureAlgorithm

_REGISTRY: dict[str, Type[MicrostructureAlgorithm]] = {}


def register(cls: Type[MicrostructureAlgorithm]) -> Type[MicrostructureAlgorithm]:
    """Class decorator to register an algorithm."""
    instance = cls()
    _REGISTRY[instance.name()] = cls
    return cls


def get_algorithm(name: str, **kwargs) -> MicrostructureAlgorithm:
    """Instantiate an algorithm by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)


def list_algorithms() -> list[str]:
    """List all registered algorithm names."""
    return sorted(_REGISTRY.keys())
