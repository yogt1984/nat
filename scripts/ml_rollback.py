#!/usr/bin/env python3
"""
ML Algorithm Rollback Utility.

Quick removal, re-enabling, and model version rollback for ML algorithms.

Usage:
    python scripts/ml_rollback.py disable momentum_continuation
    python scripts/ml_rollback.py enable momentum_continuation
    python scripts/ml_rollback.py rollback-model momentum_continuation
    python scripts/ml_rollback.py list-models momentum_continuation
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAILY_FILE = ROOT / "scripts" / "alpha" / "paper_trader_daily.py"
MODELS_DIR = ROOT / "models"


def _read_daily_algos(filepath: Path) -> list[str]:
    """Parse DAILY_ALGOS list from paper_trader_daily.py."""
    text = filepath.read_text()
    match = re.search(r'DAILY_ALGOS\s*=\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []
    body = match.group(1)
    return re.findall(r'"([^"]+)"', body)


def _write_daily_algos(filepath: Path, algos: list[str]) -> None:
    """Rewrite DAILY_ALGOS list in paper_trader_daily.py."""
    text = filepath.read_text()

    # Build replacement list (4 per line for readability)
    items = [f'"{a}"' for a in algos]
    lines = []
    for i in range(0, len(items), 4):
        chunk = ", ".join(items[i:i+4])
        lines.append(f"    {chunk},")
    body = "\n".join(lines)
    replacement = f"DAILY_ALGOS = [\n{body}\n]"

    text = re.sub(r'DAILY_ALGOS\s*=\s*\[.*?\]', replacement, text, flags=re.DOTALL)
    filepath.write_text(text)


def disable(algo_name: str, filepath: Path = DAILY_FILE) -> bool:
    """Remove algorithm from DAILY_ALGOS. Returns True if removed."""
    algos = _read_daily_algos(filepath)
    if algo_name not in algos:
        print(f"{algo_name} not found in DAILY_ALGOS")
        return False
    algos.remove(algo_name)
    _write_daily_algos(filepath, algos)
    print(f"DISABLED {algo_name} — removed from DAILY_ALGOS ({len(algos)} remaining)")
    return True


def enable(algo_name: str, filepath: Path = DAILY_FILE) -> bool:
    """Add algorithm back to DAILY_ALGOS. Returns True if added."""
    algos = _read_daily_algos(filepath)
    if algo_name in algos:
        print(f"{algo_name} already in DAILY_ALGOS")
        return False
    algos.append(algo_name)
    _write_daily_algos(filepath, algos)
    print(f"ENABLED {algo_name} — added to DAILY_ALGOS ({len(algos)} total)")
    return True


def list_models(algo_name: str, models_dir: Path = MODELS_DIR) -> list[dict]:
    """List all model files for an algorithm with metadata summary."""
    model_dir = models_dir / algo_name
    if not model_dir.exists():
        print(f"No models found for {algo_name} (dir does not exist)")
        return []

    sys.path.insert(0, str(ROOT / "scripts"))
    from utils.model_io import list_models as _list

    models = _list(model_dir)
    if not models:
        print(f"No models found for {algo_name}")
        return []

    print(f"Models for {algo_name}:")
    for m in models:
        print(f"  {m['model_file']:40s}  trained={m['training_date']}")

    return models


def rollback_model(algo_name: str, models_dir: Path = MODELS_DIR) -> bool:
    """Archive newest model, making the previous one active.

    Returns True if rollback succeeded.
    """
    model_dir = models_dir / algo_name
    if not model_dir.exists():
        print(f"No model directory for {algo_name}")
        return False

    sys.path.insert(0, str(ROOT / "scripts"))
    from utils.model_io import list_models as _list

    models = _list(model_dir)
    if len(models) < 2:
        print(f"ERROR: only {len(models)} model(s) for {algo_name} — cannot rollback")
        return False

    newest = models[0]
    archive_dir = model_dir / "archived"
    archive_dir.mkdir(exist_ok=True)

    # Move model file
    src = model_dir / newest["model_file"]
    dst = archive_dir / newest["model_file"]
    shutil.move(str(src), str(dst))

    # Move metadata file if exists
    meta_file = newest.get("metadata_file", "")
    if meta_file and (model_dir / meta_file).exists():
        shutil.move(str(model_dir / meta_file), str(archive_dir / meta_file))

    from utils.model_io import get_latest_model
    now_active = get_latest_model(model_dir)
    print(f"ROLLED BACK {algo_name}")
    print(f"  Archived: {newest['model_file']}")
    print(f"  Now active: {now_active.name if now_active else 'none'}")
    return True


def main():
    parser = argparse.ArgumentParser(description="ML algorithm rollback utility")
    sub = parser.add_subparsers(dest="action", required=True)

    p_dis = sub.add_parser("disable", help="Remove algorithm from trading")
    p_dis.add_argument("algo", help="Algorithm name")

    p_en = sub.add_parser("enable", help="Re-enable algorithm for trading")
    p_en.add_argument("algo", help="Algorithm name")

    p_rb = sub.add_parser("rollback-model", help="Revert to previous model version")
    p_rb.add_argument("algo", help="Algorithm name")

    p_ls = sub.add_parser("list-models", help="List model versions")
    p_ls.add_argument("algo", help="Algorithm name")

    args = parser.parse_args()

    if args.action == "disable":
        disable(args.algo)
    elif args.action == "enable":
        enable(args.algo)
    elif args.action == "rollback-model":
        rollback_model(args.algo)
    elif args.action == "list-models":
        list_models(args.algo)


if __name__ == "__main__":
    main()
