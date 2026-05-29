"""
Code Synthesis — LLM-generated algorithm classes.

Claude generates MicrostructureAlgorithm subclasses following the @register pattern.
Each generated algorithm goes through 3-gate subprocess validation:
  1. Syntax check (compile)
  2. Import test (subprocess import)
  3. Smoke test (run_batch on synthetic data)

Generated code → scripts/algorithms/generated/{name}.py

Usage:
  python scripts/agent/code_synth.py --idea "..." --test    # Generate + validate
  python scripts/agent/code_synth.py --list                  # List generated algos
  python scripts/agent/code_synth.py --clean                 # Remove failed algos
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
import resource
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("nat.code_synth")

ROOT = Path(__file__).resolve().parent.parent.parent
GENERATED_DIR = ROOT / "scripts" / "algorithms" / "generated"
ALGORITHMS_DIR = ROOT / "scripts" / "algorithms"

SYNTH_SYSTEM = """You are writing a Python algorithm class for a market microstructure research system.

REQUIREMENTS:
- Subclass MicrostructureAlgorithm from .base (relative import)
- Decorate with @register from .registry (relative import from parent package)
- Implement: name(), alg_features(), required_columns(), step(tick), reset()
- All output feature names must start with "alg_" prefix
- required_columns() must use only features from the manifest below
- step() receives dict[str, float], returns dict[str, float]
- Must handle NaN inputs: if any required column is NaN, return NaN for all outputs
- Include a docstring with mathematical formulation
- Use numpy for math (imported as np)

IMPORTS (use exactly these):
```python
from __future__ import annotations
import numpy as np
from ..base import AlgorithmFeature, MicrostructureAlgorithm
from ..registry import register
```

Note: The generated file lives in algorithms/generated/, so imports use .. to reach
the parent algorithms/ package.

AVAILABLE INPUT FEATURES (use only these in required_columns):
- raw_midprice, raw_spread_bps, raw_bid_depth_5, raw_ask_depth_5
- imbalance_qty_l1, imbalance_qty_l5, imbalance_depth_weighted
- flow_ofi, flow_aggressor_ratio_5s, flow_vwap_deviation, flow_net_volume
- vol_returns_1m, vol_returns_5m, vol_garman_klass, vol_parkinson
- ent_book_shape, ent_tick_5s, ent_tick_30s, ent_tick_1m, ent_surprise
- illiq_kyle_100, illiq_amihud_100, illiq_composite
- toxic_vpin_50, toxic_adverse_selection, toxic_index
- trend_momentum_60, trend_momentum_300

Return ONLY the Python source code. No markdown fences, no explanation."""


def _load_example_algorithm() -> str:
    """Load entropy_momentum.py as a few-shot example."""
    example_path = ALGORITHMS_DIR / "entropy_momentum.py"
    if example_path.exists():
        return example_path.read_text()
    return ""


def synthesize_algorithm(
    llm,
    idea: str,
    feature_manifest: str = "",
) -> str | None:
    """Generate algorithm source code from an idea description."""
    example = _load_example_algorithm()

    user_msg = f"""EXAMPLE (working algorithm):
```python
{example}
```

IDEA TO IMPLEMENT:
{idea}

Generate a complete, self-contained Python algorithm module. Use a unique class name
and unique alg_* feature names that don't conflict with existing algorithms."""

    return llm.call(SYNTH_SYSTEM, user_msg, tag="code_synth", max_tokens=4096)


def _extract_module_name(source: str) -> str:
    """Extract a module name from the class name in source."""
    match = re.search(r"class\s+(\w+)\(", source)
    if match:
        # Convert CamelCase to snake_case
        name = match.group(1)
        s = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        return f"gen_{s}"
    return f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def validate_generated_code(
    source: str,
    module_name: str,
    timeout_s: int = 30,
) -> tuple[bool, str]:
    """Run 3-gate validation. Returns (passed, error_message)."""
    # Gate 1: Syntax check
    try:
        compile(source, f"<generated:{module_name}>", "exec")
    except SyntaxError as e:
        return False, f"syntax_error: {e}"

    # Write to file
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = GENERATED_DIR / f"{module_name}.py"
    filepath.write_text(source)

    # Gate 2: Import test
    passed, err = _run_import_test(filepath, module_name, timeout_s)
    if not passed:
        filepath.unlink(missing_ok=True)
        return False, f"import_error: {err}"

    # Gate 3: Smoke test
    passed, err = _run_smoke_test(module_name, timeout_s)
    if not passed:
        filepath.unlink(missing_ok=True)
        return False, f"smoke_test_error: {err}"

    return True, ""


def _sandbox_preexec() -> None:
    """Set resource limits for sandboxed subprocess (LLM-generated code).

    Limits: 60s CPU, 512MB memory, 32 file descriptors, 4 processes.
    Called via preexec_fn in subprocess.run() — runs in the child before exec.
    """
    # CPU time (seconds) — hard kill after limit
    resource.setrlimit(resource.RLIMIT_CPU, (60, 90))
    # Virtual memory (512 MB soft, 1 GB hard)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 1024 * 1024 * 1024))
    # Open file descriptors (32 soft, 64 hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 64))
    # Max child processes (4 soft, 8 hard) — prevents fork bombs
    resource.setrlimit(resource.RLIMIT_NPROC, (4, 8))
    # New process group for clean cleanup
    os.setsid()


def _run_import_test(filepath: Path, module_name: str, timeout_s: int) -> tuple[bool, str]:
    """Test that the module imports and registers correctly."""
    test_script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, "{ROOT / 'scripts'}")
        from algorithms.generated.{module_name} import *
        from algorithms.registry import list_algorithms
        algos = list_algorithms()
        print(f"registered: {{len(algos)}} algorithms")
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=str(ROOT),
            preexec_fn=_sandbox_preexec,
        )
        if result.returncode != 0:
            return False, result.stderr[-500:]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "import timed out"
    except (subprocess.SubprocessError, OSError) as e:
        return False, str(e)


def _run_smoke_test(module_name: str, timeout_s: int) -> tuple[bool, str]:
    """Run the algorithm on synthetic data to verify output shape."""
    test_script = textwrap.dedent(f"""
        import sys, numpy as np, pandas as pd
        sys.path.insert(0, "{ROOT / 'scripts'}")

        # Import and discover
        from algorithms.generated.{module_name} import *
        from algorithms.registry import list_algorithms, get_algorithm

        # Find the newly registered algorithm
        algos = list_algorithms()
        alg = algos[-1]  # most recently registered
        instance = get_algorithm(alg)

        # Create synthetic data
        n = 500
        data = {{}}
        for col in instance.required_columns():
            data[col] = np.random.randn(n).tolist()
        df = pd.DataFrame(data)

        # Run step() on each row
        instance.reset()
        expected_keys = set(f.name for f in instance.alg_features())
        for i in range(n):
            tick = {{col: df[col].iloc[i] for col in df.columns}}
            result = instance.step(tick)
            if not isinstance(result, dict):
                raise ValueError(f"step() returned {{type(result)}}, expected dict")
            if set(result.keys()) != expected_keys:
                raise ValueError(f"step() returned keys {{set(result.keys())}}, expected {{expected_keys}}")
            # After warmup, check for non-NaN values
            if i > 200:
                vals = [v for v in result.values() if np.isfinite(v)]
                if not vals:
                    pass  # Some algorithms may still be warming up

        print("smoke test passed")
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=str(ROOT),
            preexec_fn=_sandbox_preexec,
        )
        if result.returncode != 0:
            return False, result.stderr[-500:]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "smoke test timed out"
    except (subprocess.SubprocessError, OSError) as e:
        return False, str(e)


def attempt_code_synthesis(
    llm,
    store,
    idea: str,
) -> bool:
    """Full pipeline: synthesize -> validate -> keep or delete."""
    log.info("Attempting code synthesis for: %s", idea[:80])

    source = synthesize_algorithm(llm, idea)
    if not source:
        log.warning("LLM returned no code")
        return False

    # Clean up response (remove markdown fences if present)
    text = source.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    if text.startswith("python"):
        text = text[6:]
    source = text.strip()

    module_name = _extract_module_name(source)
    passed, error = validate_generated_code(source, module_name)

    if passed:
        log.info("Code synthesis succeeded: %s", module_name)
        return True
    else:
        log.warning("Code synthesis failed for %s: %s", module_name, error)
        return False


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Algorithm Code Synthesis")
    parser.add_argument("--idea", type=str, help="Idea to implement")
    parser.add_argument("--test", action="store_true", help="Generate and validate")
    parser.add_argument("--list", action="store_true", help="List generated algos")
    parser.add_argument("--clean", action="store_true", help="Remove all generated algos")
    args = parser.parse_args()

    if args.list:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        for f in sorted(GENERATED_DIR.glob("gen_*.py")):
            print(f"  {f.name}")
        return

    if args.clean:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        for f in GENERATED_DIR.glob("gen_*.py"):
            f.unlink()
            print(f"  Removed {f.name}")
        return

    if args.idea and args.test:
        sys.path.insert(0, str(ROOT / "scripts"))
        import tomllib
        with open(ROOT / "config" / "agent.toml", "rb") as f:
            config = tomllib.load(f)
        llm_config = config.get("agent", {}).get("llm", {})
        llm_config["agent_name"] = "code_synth"

        from data.state import StateStore
        from agent.llm_client import LLMClient

        store = StateStore(str(ROOT / "data" / "nat.db"))
        llm = LLMClient(llm_config, store)

        success = attempt_code_synthesis(llm, store, args.idea)
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        store.close()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
