#!/usr/bin/env python3
"""
Experiment Governance and Versioning

Manages dataset snapshots, experiment manifests, and reproducibility.

Usage:
    # Freeze a dataset snapshot
    python scripts/experiment_governance.py snapshot --data-dir ./data/features --name "baseline_30d"

    # Create experiment manifest
    python scripts/experiment_governance.py experiment --snapshot baseline_30d --model lightgbm

    # List experiments
    python scripts/experiment_governance.py list
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import shutil


class ExperimentGovernance:
    """Manages experiment versioning and reproducibility."""

    def __init__(self, base_dir: Path = Path("./experiments")):
        self.base_dir = base_dir
        self.snapshots_dir = base_dir / "snapshots"
        self.manifests_dir = base_dir / "manifests"

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        data_dir: Path,
        name: str,
        description: Optional[str] = None,
    ) -> Dict:
        """
        Create a frozen dataset snapshot.

        Args:
            data_dir: Directory containing Parquet files
            name: Snapshot name
            description: Optional description

        Returns:
            Snapshot metadata dict
        """
        print(f"Creating snapshot '{name}'...")

        snapshot_dir = self.snapshots_dir / name
        if snapshot_dir.exists():
            raise ValueError(f"Snapshot '{name}' already exists")

        snapshot_dir.mkdir(parents=True)

        # Copy Parquet files
        files = list(data_dir.glob("*.parquet"))
        print(f"Copying {len(files)} files...")

        file_hashes = {}
        for file in files:
            dest = snapshot_dir / file.name
            shutil.copy2(file, dest)

            # Compute hash
            with open(file, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            file_hashes[file.name] = file_hash

        # Create metadata
        metadata = {
            "name": name,
            "description": description or "",
            "created_at": datetime.now().isoformat(),
            "source_dir": str(data_dir),
            "num_files": len(files),
            "file_hashes": file_hashes,
            "snapshot_hash": self._compute_snapshot_hash(file_hashes),
        }

        # Save metadata
        metadata_path = snapshot_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Snapshot created: {snapshot_dir}")
        print(f"Snapshot hash: {metadata['snapshot_hash']}")

        return metadata

    def create_experiment_manifest(
        self,
        snapshot_name: str,
        model_name: str,
        features: List[str],
        label: str,
        cost_model: Dict,
        notes: Optional[str] = None,
    ) -> Dict:
        """
        Create experiment manifest.

        Args:
            snapshot_name: Dataset snapshot name
            model_name: Model identifier
            features: List of feature names
            label: Label definition
            cost_model: Cost model parameters
            notes: Optional experiment notes

        Returns:
            Experiment manifest dict
        """
        # Verify snapshot exists
        snapshot_dir = self.snapshots_dir / snapshot_name
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot '{snapshot_name}' not found")

        # Load snapshot metadata
        with open(snapshot_dir / "metadata.json") as f:
            snapshot_meta = json.load(f)

        # Create experiment ID
        experiment_id = f"{snapshot_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create manifest
        manifest = {
            "experiment_id": experiment_id,
            "created_at": datetime.now().isoformat(),
            "snapshot": {
                "name": snapshot_name,
                "hash": snapshot_meta["snapshot_hash"],
            },
            "model": {
                "name": model_name,
                "version": "1.0",
            },
            "features": {
                "names": features,
                "version": self._compute_hash(str(features)),
            },
            "label": {
                "definition": label,
                "version": self._compute_hash(label),
            },
            "cost_model": cost_model,
            "notes": notes or "",
        }

        # Save manifest
        manifest_path = self.manifests_dir / f"{experiment_id}.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Experiment manifest created: {manifest_path}")
        return manifest

    def list_snapshots(self) -> List[Dict]:
        """List all snapshots."""
        snapshots = []
        for snapshot_dir in self.snapshots_dir.iterdir():
            if snapshot_dir.is_dir():
                metadata_path = snapshot_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        snapshots.append(json.load(f))
        return snapshots

    def list_experiments(self) -> List[Dict]:
        """List all experiments."""
        experiments = []
        for manifest_path in self.manifests_dir.glob("*.json"):
            with open(manifest_path) as f:
                experiments.append(json.load(f))
        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)

    def _compute_snapshot_hash(self, file_hashes: Dict[str, str]) -> str:
        """Compute hash of entire snapshot."""
        combined = "".join(sorted(file_hashes.values()))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _compute_hash(self, data: str) -> str:
        """Compute hash of string data."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Experiment governance")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Snapshot command
    snapshot_parser = subparsers.add_parser("snapshot", help="Create dataset snapshot")
    snapshot_parser.add_argument("--data-dir", type=Path, required=True)
    snapshot_parser.add_argument("--name", type=str, required=True)
    snapshot_parser.add_argument("--description", type=str)

    # Experiment command
    exp_parser = subparsers.add_parser("experiment", help="Create experiment manifest")
    exp_parser.add_argument("--snapshot", type=str, required=True)
    exp_parser.add_argument("--model", type=str, required=True)
    exp_parser.add_argument("--features", type=str, nargs="+", required=True)
    exp_parser.add_argument("--label", type=str, required=True)
    exp_parser.add_argument("--notes", type=str)

    # List command
    subparsers.add_parser("list", help="List snapshots and experiments")

    args = parser.parse_args()

    gov = ExperimentGovernance()

    if args.command == "snapshot":
        gov.create_snapshot(args.data_dir, args.name, args.description)

    elif args.command == "experiment":
        # Simplified cost model for now
        cost_model = {
            "spread_bps": 2.0,
            "slippage_bps": 1.0,
            "fee_bps": 2.5,
        }

        gov.create_experiment_manifest(
            args.snapshot,
            args.model,
            args.features,
            args.label,
            cost_model,
            args.notes,
        )

    elif args.command == "list":
        print("\n=== SNAPSHOTS ===")
        for snapshot in gov.list_snapshots():
            print(f"  {snapshot['name']}: {snapshot['num_files']} files, created {snapshot['created_at']}")

        print("\n=== EXPERIMENTS ===")
        for exp in gov.list_experiments():
            print(f"  {exp['experiment_id']}: {exp['model']['name']} on {exp['snapshot']['name']}")


if __name__ == "__main__":
    main()
