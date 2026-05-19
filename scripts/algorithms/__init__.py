# Microstructure Algorithm Framework
#
# Pluggable algorithms that compute derived features from base ingestor features.
# Python-first for research; Rust dummies get replaced once validated.

from .base import AlgorithmFeature, MicrostructureAlgorithm
from .registry import register, get_algorithm, list_algorithms
from .runner import AlgorithmRunner, AlgorithmResult
from .autodiscover import discover_all

# Auto-register all algorithm modules on package import
discover_all()

__all__ = [
    "AlgorithmFeature", "MicrostructureAlgorithm",
    "register", "get_algorithm", "list_algorithms",
    "AlgorithmRunner", "AlgorithmResult",
    "discover_all",
]
