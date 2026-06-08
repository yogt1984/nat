"""
Config generator — produce N algorithm config variants for swarm evaluation.

Reads a base config (config/algorithms.toml) and parameter ranges
(config/swarm_ranges.toml), then generates random or grid-search configs.
Each config is a full TOML file usable by the algorithm framework.

Usage:
    python scripts/swarm/config_generator.py \
        --base config/algorithms.toml \
        --ranges config/swarm_ranges.toml \
        --count 16 --output data/swarm/configs/
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import random
import sys
from itertools import product
from pathlib import Path
from typing import Any, Optional

import toml

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """Generate algorithm config variants from base + parameter ranges."""

    def __init__(self, base_config: str, ranges_config: str):
        self.base = toml.load(base_config)
        self.ranges = toml.load(ranges_config)

    def generate_random(self, n: int, seed: Optional[int] = None) -> list[dict]:
        """Generate N random configs by sampling each parameter from its range."""
        rng = random.Random(seed)
        configs = []
        for _ in range(n):
            cfg = copy.deepcopy(self.base)
            for section, params in self.ranges.items():
                if section not in cfg:
                    cfg[section] = {}
                for param_name, spec in params.items():
                    if not isinstance(spec, dict) or "type" not in spec:
                        continue
                    cfg[section][param_name] = self._sample(spec, rng)
            configs.append(cfg)
        return configs

    def generate_grid(self, resolution: int = 3) -> list[dict]:
        """Generate grid-search configs over all parameters.

        Resolution is the number of points per dimension. Use sparingly —
        35D at resolution=3 would be 3^35 ~ 5e16 configs. Intended for
        small subspaces only.
        """
        axes = []
        axis_keys = []

        for section, params in self.ranges.items():
            for param_name, spec in params.items():
                if not isinstance(spec, dict) or "type" not in spec:
                    continue
                axis_keys.append((section, param_name))
                axes.append(self._grid_points(spec, resolution))

        configs = []
        for point in product(*axes):
            cfg = copy.deepcopy(self.base)
            for (section, param_name), value in zip(axis_keys, point):
                if section not in cfg:
                    cfg[section] = {}
                cfg[section][param_name] = value
            configs.append(cfg)

        return configs

    def generate_from_optuna(self, trial) -> dict:
        """Generate a config from an Optuna trial (Tier 3 integration).

        Args:
            trial: optuna.trial.Trial instance.

        Returns:
            Full config dict with trial-suggested parameters.
        """
        cfg = copy.deepcopy(self.base)

        for section, params in self.ranges.items():
            if section not in cfg:
                cfg[section] = {}
            for param_name, spec in params.items():
                if not isinstance(spec, dict) or "type" not in spec:
                    continue
                full_name = f"{section}.{param_name}"
                cfg[section][param_name] = self._suggest_optuna(
                    trial, full_name, spec,
                )

        return cfg

    def write_configs(
        self,
        configs: list[dict],
        output_dir: str,
    ) -> list[Path]:
        """Write configs as output_dir/config_{i}.toml with hash metadata."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []

        for i, cfg in enumerate(configs):
            cfg_hash = config_hash(cfg)
            cfg["_meta"] = {"config_hash": cfg_hash, "config_index": i}
            path = out / f"config_{i:04d}.toml"
            with open(path, "w") as f:
                toml.dump(cfg, f)
            paths.append(path)

        logger.info("Wrote %d configs to %s", len(configs), output_dir)
        return paths

    # ── sampling helpers ────────────────────────────────────────

    @staticmethod
    def _sample(spec: dict, rng: random.Random) -> Any:
        t = spec["type"]
        if t == "float":
            lo, hi = spec["min"], spec["max"]
            if spec.get("log"):
                return math.exp(rng.uniform(math.log(lo), math.log(hi)))
            return rng.uniform(lo, hi)
        elif t == "int":
            return rng.randint(spec["min"], spec["max"])
        elif t == "categorical":
            return rng.choice(spec["choices"])
        elif t == "bool":
            return rng.random() < 0.5
        else:
            raise ValueError(f"Unknown param type: {t}")

    @staticmethod
    def _grid_points(spec: dict, resolution: int) -> list:
        t = spec["type"]
        if t == "float":
            lo, hi = spec["min"], spec["max"]
            if spec.get("log"):
                pts = [math.exp(x) for x in
                       _linspace(math.log(lo), math.log(hi), resolution)]
            else:
                pts = _linspace(lo, hi, resolution)
            return pts
        elif t == "int":
            lo, hi = spec["min"], spec["max"]
            step = max(1, (hi - lo) // (resolution - 1)) if resolution > 1 else 1
            return list(range(lo, hi + 1, step))[:resolution]
        elif t == "categorical":
            return spec["choices"]
        elif t == "bool":
            return [True, False]
        else:
            raise ValueError(f"Unknown param type: {t}")

    @staticmethod
    def _suggest_optuna(trial, name: str, spec: dict) -> Any:
        t = spec["type"]
        if t == "float":
            return trial.suggest_float(
                name, spec["min"], spec["max"],
                log=spec.get("log", False),
            )
        elif t == "int":
            return trial.suggest_int(name, spec["min"], spec["max"])
        elif t == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        elif t == "bool":
            return trial.suggest_categorical(name, [True, False])
        else:
            raise ValueError(f"Unknown param type: {t}")


def config_hash(cfg: dict) -> str:
    """Deterministic SHA-256 hash of a config dict (ignoring _meta)."""
    clean = {k: v for k, v in cfg.items() if k != "_meta"}
    raw = json.dumps(clean, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


# ── CLI ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate swarm config variants",
    )
    parser.add_argument(
        "--base", default="config/algorithms.toml",
        help="Base algorithm config",
    )
    parser.add_argument(
        "--ranges", default="config/swarm_ranges.toml",
        help="Parameter ranges config",
    )
    parser.add_argument(
        "--count", type=int, default=16,
        help="Number of configs to generate",
    )
    parser.add_argument(
        "--output", default="data/swarm/configs",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--mode", choices=["random", "grid"], default="random",
        help="Generation mode",
    )
    parser.add_argument(
        "--resolution", type=int, default=3,
        help="Grid resolution per dimension (grid mode only)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    gen = ConfigGenerator(args.base, args.ranges)

    if args.mode == "random":
        configs = gen.generate_random(args.count, seed=args.seed)
    else:
        configs = gen.generate_grid(resolution=args.resolution)
        if args.count and len(configs) > args.count:
            rng = random.Random(args.seed)
            configs = rng.sample(configs, args.count)

    paths = gen.write_configs(configs, args.output)
    print(f"Generated {len(paths)} configs in {args.output}/")

    # Show a summary of what varies
    for section, params in gen.ranges.items():
        for param_name, spec in params.items():
            if not isinstance(spec, dict) or "type" not in spec:
                continue
            vals = [c.get(section, {}).get(param_name) for c in configs]
            if spec["type"] in ("float", "int"):
                print(f"  {section}.{param_name}: [{min(vals):.4g}, {max(vals):.4g}]")
            elif spec["type"] == "categorical":
                from collections import Counter
                counts = Counter(vals)
                print(f"  {section}.{param_name}: {dict(counts)}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
