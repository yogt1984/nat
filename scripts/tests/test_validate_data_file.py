"""Planted tests for single-file parquet validation (Capability B).

Contract (docs/requirements/parquet_viz_validation.md §FR-B):
  - `nat data validate <path>` validates ONE parquet file.
  - find_parquet_files(<file>) returns just that file (ignores --hours).
  - A clean file → verdict PASS; a file with an injected >5s gap + post-warmup
    NaN block → verdict FAIL (hard checks Continuity + NaN Ratio fail).
  - Verdict is PASS / WARN / FAIL; FAIL exits nonzero.

These are red-first: they exercise `verdict`, `HARD_CHECKS`, and single-file
`find_parquet_files`, which the implementation must add to validate_data.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import validate_data as vd


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #

START_NS = 1_781_532_000_000_000_000  # 2026-06-15 14:00:00 UTC, arbitrary t0
STEP_NS = 100_000_000                 # 100 ms emission cadence


def _clean_frame(seconds: int = 130, symbol: str = "BTC") -> pd.DataFrame:
    """A continuous 100 ms feature frame with no gaps and no NaN."""
    n = seconds * 10
    ts = START_NS + np.arange(n, dtype=np.int64) * STEP_NS
    rng = np.linspace(0, 1, n)
    return pd.DataFrame(
        {
            "timestamp_ns": ts,
            "symbol": symbol,
            "sequence_id": np.arange(n, dtype=np.int64),
            "raw_midprice": 60_000.0 + 100.0 * np.sin(rng * 6.28),
            "raw_spread": 0.5 + 0.1 * rng,
            "vpin_10": 0.4 + 0.05 * np.cos(rng * 6.28),  # stays in [0, 1]
        }
    )


def _write(df: pd.DataFrame, path) -> str:
    df.to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def clean_file(tmp_path):
    return _write(_clean_frame(), tmp_path / "20260615_140000.parquet")


@pytest.fixture
def broken_file(tmp_path):
    """Inject a >5 s gap and a post-warmup NaN block (both HARD failures)."""
    df = _clean_frame()
    ts_col = "timestamp_ns"
    t0 = df[ts_col].min()
    # Gap: drop rows spanning 100 s..110 s so the diff at the seam is ~10 s.
    drop = (df[ts_col] >= t0 + 100 * 1_000_000_000) & (df[ts_col] < t0 + 110 * 1_000_000_000)
    df = df.loc[~drop].reset_index(drop=True)
    # NaN block: blank a feature for 80 s..90 s (post the 60 s warmup window).
    nanmask = (df[ts_col] >= t0 + 80 * 1_000_000_000) & (df[ts_col] < t0 + 90 * 1_000_000_000)
    df.loc[nanmask, "raw_midprice"] = np.nan
    return _write(df, tmp_path / "20260615_150000.parquet")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_find_parquet_files_accepts_single_file(clean_file):
    from pathlib import Path

    files = vd.find_parquet_files(Path(clean_file))
    assert files == [Path(clean_file)]
    # --hours must not filter away an explicitly named file
    files_hours = vd.find_parquet_files(Path(clean_file), hours=1)
    assert files_hours == [Path(clean_file)]


def test_clean_file_passes(clean_file):
    from pathlib import Path

    report = vd.validate(Path(clean_file), vd.ValidationConfig())
    assert report.verdict == "PASS"
    assert report.passed is True
    by_name = {c.name: c for c in report.checks}
    assert by_name["Continuity"].passed
    assert by_name["NaN Ratio"].passed


def test_broken_file_fails_on_gap_and_nan(broken_file):
    from pathlib import Path

    report = vd.validate(Path(broken_file), vd.ValidationConfig())
    assert report.verdict == "FAIL"
    by_name = {c.name: c for c in report.checks}
    assert not by_name["Continuity"].passed
    assert not by_name["NaN Ratio"].passed


def test_verdict_levels_are_classified():
    # The implementation must declare which checks are hard (FAIL) vs soft (WARN).
    assert "Continuity" in vd.HARD_CHECKS
    assert "NaN Ratio" in vd.HARD_CHECKS
    assert "Feature Ranges" not in vd.HARD_CHECKS  # soft → WARN, not FAIL


def test_report_to_dict_is_json_serializable(clean_file):
    import json
    from pathlib import Path

    report = vd.validate(Path(clean_file), vd.ValidationConfig())
    d = report.to_dict()
    assert d["verdict"] == "PASS"
    assert isinstance(d["checks"], list) and d["checks"]
    json.dumps(d)  # must not raise
