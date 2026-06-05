"""
Tests for Parquet schema versioning and forward/backward compatibility.

Verifies that the Python loader:
  - reads schema_version from Parquet metadata
  - warns on version mismatch or legacy (unversioned) files
  - pads missing columns with NaN
  - keeps unknown columns (alg_* or newer features)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cluster_pipeline.loader import (
    CURRENT_SCHEMA_VERSION,
    normalize_schema,
    read_schema_version,
)


def _write_parquet(
    path: Path,
    columns: list[str],
    n_rows: int = 10,
    schema_version: int | None = None,
) -> Path:
    """Write a minimal Parquet file with optional schema_version metadata."""
    data = {col: np.random.default_rng(42).standard_normal(n_rows) for col in columns}
    # Ensure required meta columns have correct types
    if "timestamp_ns" in columns:
        data["timestamp_ns"] = np.arange(n_rows, dtype=np.int64)
    if "symbol" in columns:
        data["symbol"] = ["BTC"] * n_rows
    if "sequence_id" in columns:
        data["sequence_id"] = np.arange(n_rows, dtype=np.uint64)

    table = pa.Table.from_pandas(pd.DataFrame(data))

    if schema_version is not None:
        metadata = table.schema.metadata or {}
        metadata[b"schema_version"] = str(schema_version).encode()
        table = table.replace_schema_metadata(metadata)

    pq.write_table(table, str(path))
    return path


class TestReadSchemaVersion:
    def test_returns_none_for_legacy_file(self, tmp_path):
        path = _write_parquet(tmp_path / "legacy.parquet", ["raw_midprice"])
        assert read_schema_version(path) is None

    def test_returns_int_for_versioned_file(self, tmp_path):
        path = _write_parquet(
            tmp_path / "v1.parquet", ["raw_midprice"], schema_version=1
        )
        assert read_schema_version(path) == 1

    def test_returns_correct_version_number(self, tmp_path):
        path = _write_parquet(
            tmp_path / "v5.parquet", ["raw_midprice"], schema_version=5
        )
        assert read_schema_version(path) == 5

    def test_returns_none_for_nonexistent_file(self):
        assert read_schema_version(Path("/nonexistent/file.parquet")) is None


class TestNormalizeSchema:
    def test_pads_missing_columns(self, tmp_path):
        path = _write_parquet(
            tmp_path / "partial.parquet",
            ["timestamp_ns", "symbol", "raw_midprice"],
            schema_version=CURRENT_SCHEMA_VERSION,
        )
        df = pd.DataFrame({
            "timestamp_ns": np.arange(10, dtype=np.int64),
            "symbol": ["BTC"] * 10,
            "raw_midprice": np.random.default_rng(42).standard_normal(10),
        })

        result = normalize_schema(df, [path])

        # raw_spread should have been padded
        assert "raw_spread" in result.columns
        assert result["raw_spread"].isna().all()

    def test_keeps_existing_columns(self, tmp_path):
        path = _write_parquet(
            tmp_path / "full.parquet",
            ["timestamp_ns", "symbol", "raw_midprice"],
            schema_version=CURRENT_SCHEMA_VERSION,
        )
        df = pd.DataFrame({
            "timestamp_ns": np.arange(5, dtype=np.int64),
            "symbol": ["BTC"] * 5,
            "raw_midprice": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = normalize_schema(df, [path])

        # Original data preserved
        assert list(result["raw_midprice"]) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_keeps_unknown_columns(self, tmp_path):
        path = _write_parquet(
            tmp_path / "extra.parquet",
            ["timestamp_ns", "symbol", "raw_midprice", "alg_custom_signal"],
            schema_version=CURRENT_SCHEMA_VERSION,
        )
        df = pd.DataFrame({
            "timestamp_ns": np.arange(5, dtype=np.int64),
            "symbol": ["BTC"] * 5,
            "raw_midprice": np.ones(5),
            "alg_custom_signal": np.ones(5) * 0.5,
        })

        result = normalize_schema(df, [path])

        assert "alg_custom_signal" in result.columns
        assert (result["alg_custom_signal"] == 0.5).all()

    def test_warns_on_legacy_file(self, tmp_path, caplog):
        path = _write_parquet(
            tmp_path / "legacy.parquet", ["timestamp_ns", "symbol", "raw_midprice"]
        )
        df = pd.DataFrame({
            "timestamp_ns": np.arange(5, dtype=np.int64),
            "symbol": ["BTC"] * 5,
            "raw_midprice": np.ones(5),
        })

        with caplog.at_level(logging.WARNING, logger="cluster_pipeline.loader"):
            normalize_schema(df, [path])

        assert any("no schema_version" in m.lower() for m in caplog.messages)

    def test_warns_on_old_version(self, tmp_path, caplog):
        path = _write_parquet(
            tmp_path / "old.parquet",
            ["timestamp_ns", "symbol", "raw_midprice"],
            schema_version=0,
        )
        df = pd.DataFrame({
            "timestamp_ns": np.arange(5, dtype=np.int64),
            "symbol": ["BTC"] * 5,
            "raw_midprice": np.ones(5),
        })

        with caplog.at_level(logging.WARNING, logger="cluster_pipeline.loader"):
            normalize_schema(df, [path])

        assert any("version 0" in m for m in caplog.messages)

    def test_warns_on_future_version(self, tmp_path, caplog):
        path = _write_parquet(
            tmp_path / "future.parquet",
            ["timestamp_ns", "symbol", "raw_midprice"],
            schema_version=CURRENT_SCHEMA_VERSION + 1,
        )
        df = pd.DataFrame({
            "timestamp_ns": np.arange(5, dtype=np.int64),
            "symbol": ["BTC"] * 5,
            "raw_midprice": np.ones(5),
        })

        with caplog.at_level(logging.WARNING, logger="cluster_pipeline.loader"):
            normalize_schema(df, [path])

        future_v = CURRENT_SCHEMA_VERSION + 1
        assert any(f"version {future_v}" in m for m in caplog.messages)

    def test_no_warning_on_current_version(self, tmp_path, caplog):
        from data.schema import ALL_BASE, ALL_OPTIONAL

        all_cols = ALL_BASE + ALL_OPTIONAL
        path = _write_parquet(
            tmp_path / "current.parquet",
            all_cols,
            schema_version=CURRENT_SCHEMA_VERSION,
        )
        df = pd.DataFrame(
            {col: np.random.default_rng(42).standard_normal(5) for col in all_cols}
        )
        df["timestamp_ns"] = np.arange(5, dtype=np.int64)
        df["symbol"] = "BTC"
        df["sequence_id"] = np.arange(5, dtype=np.uint64)

        with caplog.at_level(logging.WARNING, logger="cluster_pipeline.loader"):
            normalize_schema(df, [path])

        warn_msgs = [m for m in caplog.messages if "version" in m.lower()]
        assert len(warn_msgs) == 0
