#!/usr/bin/env python3
"""
List Saved Models

Displays all saved models with their metadata and performance metrics.

Usage:
    python scripts/list_models.py
    python scripts/list_models.py --model-dir ./custom/models
"""

import argparse
import sys
from pathlib import Path
from tabulate import tabulate

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_io import list_models


def main():
    parser = argparse.ArgumentParser(description="List saved models")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./models"),
        help="Directory containing models (default: ./models)",
    )

    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"Model directory not found: {args.model_dir}")
        print()
        print("Train models first:")
        print("  python scripts/train_baseline.py --snapshot baseline_30d --model elasticnet")
        return

    models = list_models(args.model_dir)

    if not models:
        print(f"No models found in {args.model_dir}")
        print()
        print("Train models first:")
        print("  python scripts/train_baseline.py --snapshot baseline_30d --model elasticnet")
        return

    print()
    print("=" * 100)
    print("SAVED MODELS")
    print("=" * 100)
    print()

    # Prepare table data
    table_data = []
    for model in models:
        test_r2 = model.get("test_r2")
        if test_r2 is not None:
            test_r2_str = f"{test_r2:.4f}"
        else:
            test_r2_str = "N/A"

        table_data.append([
            model["model_name"],
            model["model_type"],
            model["training_date"][:19] if len(model["training_date"]) > 19 else model["training_date"],
            test_r2_str,
            model["model_file"],
        ])

    headers = ["Model Name", "Type", "Trained", "Test R²", "Filename"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()
    print(f"Total models: {len(models)}")
    print(f"Model directory: {args.model_dir}")
    print()


if __name__ == "__main__":
    main()
