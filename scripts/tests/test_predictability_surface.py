"""Planted test for PROC-8 — the predictability surface artifact + viz.

The surface is the single first-class object the platform revolves around: axes
(combo × horizon × label-definition × regime), value = null-calibrated, FDR-corrected
MI. It aggregates PROC-6/7 findings (and mi_ksg label-mode runs) into one queryable
parquet, and `nat viz predictability` renders it.

Contract:
  (a) fixed schema — every row carries combo_id/horizon/label_def/regime_bin/mi_bits/
      z_null/bh_q/informative/maturity + provenance (run_id, git_sha, symbol);
  (b) aggregation consumes the three source processes and ONLY label-mode mi_ksg rows;
  (c) deterministic — same records in, byte-identical frame out;
  (d) parquet round-trip preserves everything;
  (e) index aggregation reads persisted runs (newest per process×symbol wins);
  (f) the terminal render shows the argmax WITH its q and marks FDR-passed cells;
  (g) maturity tags: FDR-passed cells are [PRELIM], the rest [SPEC] — never [PROVEN]
      from statistics alone (that requires downstream validation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from processes.base import Finding, ProcessResult
from processes.surface import (
    SURFACE_COLUMNS,
    build_surface,
    aggregate_from_index,
    load_surface,
    render_surface,
    save_surface,
    surface_rows_from_record,
)


def _hls_record(run_id="proc_hls_BTC_1", git_sha="abc123"):
    """A horizon_label_scan record: 2 features x 2 horizons, one informative cell."""
    findings = []
    for feat, h, bucket, bits, z, p, q, info in [
        ("f_good", "4bar", 0, 0.31, 6.2, 0.005, 0.02, True),
        ("f_good", "12bar", 0, 0.09, 2.1, 0.08, 0.32, False),
        ("f_noise", "4bar", 1, 0.01, 0.4, 0.55, 0.80, False),
    ]:
        findings.append({
            "feature": feat, "horizon": h, "metric": "cond_mi_bits",
            "value": bits, "p_value": p, "p_adjusted": q, "threshold": 3.0,
            "informative": info,
            "extras": {"pt_mult": 2.0, "sl_mult": 1.0, "conditioning": "cond_z",
                       "bucket": bucket, "z_range": (0.0, 0.25), "n": 400,
                       "raw_bits": bits + 0.01, "z": z},
        })
    return {
        "run_id": run_id, "process": "horizon_label_scan", "kind": "evaluation",
        "symbol": "BTC", "timeframe": "15min", "params": {},
        "provenance": {"git_sha": git_sha, "generated_at": "2026-07-30T10:00:00+00:00"},
        "findings": findings, "summary": {},
    }


def _cp_record():
    """A conditional_predictability record (forward-return label)."""
    return {
        "run_id": "proc_cp_ETH_1", "process": "conditional_predictability",
        "kind": "evaluation", "symbol": "ETH", "timeframe": "15min", "params": {},
        "provenance": {"git_sha": "def456", "generated_at": "2026-07-30T11:00:00+00:00"},
        "findings": [{
            "feature": "imbalance_qty_l1", "horizon": "h1", "metric": "cond_mi_bits",
            "value": 0.12, "p_value": 0.01, "p_adjusted": 0.04, "threshold": 3.0,
            "informative": True,
            "extras": {"conditioning": "vol_z", "bucket": 2, "n_buckets": 4,
                       "z_range": (0.5, 0.75), "n": 300, "raw_bits": 0.13, "z": 4.1},
        }],
        "summary": {},
    }


def _mi_label_record():
    """A label-mode mi_ksg record + one forward-return finding that must be EXCLUDED."""
    return {
        "run_id": "proc_mi_ksg_BTC_1", "process": "mi_ksg", "kind": "evaluation",
        "symbol": "BTC", "timeframe": "15min", "params": {},
        "provenance": {"git_sha": "abc123", "generated_at": "2026-07-30T12:00:00+00:00"},
        "findings": [
            {"feature": "toxic_vpin_50_mean", "horizon": "label", "metric": "mi_bits",
             "value": 0.15, "p_value": 0.02, "p_adjusted": 0.05, "threshold": 3.0,
             "informative": True,
             "extras": {"target": "tb_label", "gate": "null_z", "bits_above_null": 0.14,
                        "z": 3.5, "p": 0.02, "null_mean": 0.01, "n_samples": 350}},
            # forward-return finding — different family, no null z: must not enter
            {"feature": "toxic_vpin_50_mean", "horizon": "h1", "metric": "mi_bits",
             "value": 0.05, "p_value": None, "informative": False,
             "extras": {"i_min_bits": 0.03}},
        ],
        "summary": {},
    }


class TestSchema:
    def test_columns_are_exactly_the_schema(self):
        df = build_surface([_hls_record(), _cp_record(), _mi_label_record()])
        assert list(df.columns) == SURFACE_COLUMNS

    def test_row_fields_mapped(self):
        df = build_surface([_hls_record()])
        top = df[(df.combo_id == "f_good") & (df.horizon == "4bar")].iloc[0]
        assert top["label_def"] == "tb(2.0,1.0)"
        assert top["horizon"] == "4bar"
        assert top["regime_var"] == "cond_z"
        assert top["regime_bin"] == "b0"
        assert top["mi_bits"] == 0.31
        assert top["z_null"] == 6.2
        assert top["bh_q"] == 0.02
        assert bool(top["informative"]) is True
        assert top["git_sha"] == "abc123"
        assert top["run_id"] == "proc_hls_BTC_1"

    def test_label_mode_mi_ksg_mapped_and_fwd_excluded(self):
        df = build_surface([_mi_label_record()])
        assert len(df) == 1                       # the h1 forward-return row excluded
        row = df.iloc[0]
        assert row["label_def"] == "tb_label"
        assert row["horizon"] == "label"
        assert row["regime_bin"] == "all"
        assert row["mi_bits"] == 0.15

    def test_cp_record_label_def_is_fwd_ret(self):
        df = build_surface([_cp_record()])
        assert df.iloc[0]["label_def"] == "fwd_ret"
        assert df.iloc[0]["regime_bin"] == "b2"

    def test_maturity_prelim_or_spec_never_proven(self):
        df = build_surface([_hls_record(), _cp_record(), _mi_label_record()])
        assert set(df["maturity"]) <= {"PRELIM", "SPEC"}
        assert (df.loc[df.informative, "maturity"] == "PRELIM").all()
        assert (df.loc[~df.informative.astype(bool), "maturity"] == "SPEC").all()


class TestDeterminism:
    def test_same_records_same_frame(self):
        recs = [_hls_record(), _cp_record(), _mi_label_record()]
        d1 = build_surface(recs)
        d2 = build_surface(list(reversed(recs)))   # input order must not matter
        pd.testing.assert_frame_equal(d1, d2)


class TestRoundTrip:
    def test_parquet_round_trip(self, tmp_path):
        df = build_surface([_hls_record(), _cp_record()])
        path = save_surface(df, path=tmp_path / "surface.parquet")
        back = load_surface(path)
        pd.testing.assert_frame_equal(df, back)

    def test_load_missing_returns_empty_with_schema(self, tmp_path):
        df = load_surface(tmp_path / "nope.parquet")
        assert df.empty
        assert list(df.columns) == SURFACE_COLUMNS


class TestIndexAggregation:
    def _save(self, tmp_path, rec):
        from processes import persistence
        r = ProcessResult(
            run_id=rec["run_id"], process=rec["process"], kind=rec["kind"],
            symbol=rec["symbol"], timeframe=rec["timeframe"], params={},
        )
        r.findings = [Finding(**{k: v for k, v in f.items()}) for f in rec["findings"]]
        r.provenance = rec["provenance"]
        r.finalize(1.0)
        persistence.save_result(r, out_dir=tmp_path / "json", db_path=tmp_path / "nat.db")

    def test_aggregates_saved_runs(self, tmp_path):
        self._save(tmp_path, _hls_record())
        self._save(tmp_path, _cp_record())
        df, path = aggregate_from_index(
            db_path=tmp_path / "nat.db", out_path=tmp_path / "surface.parquet",
        )
        assert path.exists()
        assert set(df["combo_id"]) == {"f_good", "f_noise", "imbalance_qty_l1"}

    def test_latest_run_per_process_symbol_wins(self, tmp_path):
        old = _hls_record(run_id="proc_hls_BTC_old", git_sha="old000")
        old["provenance"]["generated_at"] = "2026-07-29T00:00:00+00:00"
        new = _hls_record(run_id="proc_hls_BTC_new", git_sha="new111")
        self._save(tmp_path, old)
        self._save(tmp_path, new)
        df, _ = aggregate_from_index(
            db_path=tmp_path / "nat.db", out_path=tmp_path / "surface.parquet",
        )
        assert set(df["run_id"]) == {"proc_hls_BTC_new"}

    def test_empty_index_gives_empty_surface(self, tmp_path):
        df, path = aggregate_from_index(
            db_path=tmp_path / "nat.db", out_path=tmp_path / "surface.parquet",
        )
        assert df.empty and list(df.columns) == SURFACE_COLUMNS


class TestRender:
    def test_render_shows_argmax_with_q_and_fdr_marks(self):
        df = build_surface([_hls_record(), _cp_record(), _mi_label_record()])
        out = render_surface(df)
        assert "f_good" in out
        assert "q=0.02" in out                     # argmax never shown without its q
        assert "*" in out                          # FDR-passed marker
        assert "PRELIM" in out

    def test_render_empty_is_graceful(self):
        out = render_surface(build_surface([]))
        assert "empty" in out.lower() or "no " in out.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
