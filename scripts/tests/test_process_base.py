"""Process framework contract tests: registry, NaN partition, describe()."""

import numpy as np
import pandas as pd
import pytest

from processes import (
    EvaluationProcess,
    TransformProcess,
    get_process,
    list_processes,
    list_processes_by_kind,
    partition_usable_columns,
)
from processes.base import make_run_id
from processes.synthetic import make_planted_frame


def test_stage1_processes_registered():
    names = list_processes()
    for expected in ("ic_horizon", "mi_ksg", "transfer_entropy", "spectral", "ml_importance"):
        assert expected in names


def test_registry_round_trip_and_kinds():
    proc = get_process("ic_horizon", min_ic=0.05)
    assert proc.name() == "ic_horizon"
    assert proc.kind == "evaluation"
    assert proc.params["min_ic"] == 0.05
    assert "ic_horizon" in list_processes_by_kind("evaluation")

    with pytest.raises(KeyError):
        get_process("nonexistent_process")


def test_unknown_param_rejected():
    with pytest.raises(TypeError, match="unknown params"):
        get_process("ic_horizon", bogus_param=1)


def test_describe_shape():
    for name in list_processes():
        spec = get_process(name).describe()
        assert spec["name"] == name
        assert spec["kind"] in ("evaluation", "transform")
        assert spec["data_level"] in ("bars", "ticks")
        assert isinstance(spec["doc"], str) and spec["doc"]
        assert isinstance(spec["params"], dict)
        for pspec in spec["params"].values():
            assert "default" in pspec and "doc" in pspec


def test_partition_usable_columns_reasons():
    df = make_planted_frame(n=200)
    df["feat_sparse"] = np.nan
    df.loc[df.index[:10], "feat_sparse"] = 1.0 + np.arange(10)
    df["feat_text"] = "x"

    usable, skipped = partition_usable_columns(
        df,
        ["feat_signal", "feat_noise", "feat_dead", "feat_const",
         "feat_sparse", "feat_text", "feat_missing"],
        min_obs=50,
    )
    assert usable == ["feat_signal", "feat_noise"]
    reasons = {s["feature"]: s["reason"] for s in skipped}
    assert reasons["feat_dead"] == "all_nan"
    assert reasons["feat_const"] == "constant"
    assert reasons["feat_missing"] == "missing"
    assert reasons["feat_text"] == "non_numeric"
    assert reasons["feat_sparse"].startswith("n_valid=10")


def test_required_columns_pattern_filter():
    proc = get_process("ic_horizon", features=["ent_", "ofi_"])
    available = ["ent_tick_1m_mean", "ofi_imbalance_mean", "vol_returns_mean",
                 "symbol", "bar_start"]
    cols = proc.required_columns(available)
    assert cols == ["ent_tick_1m_mean", "ofi_imbalance_mean"]


def test_required_columns_excludes_meta():
    proc = get_process("ic_horizon")
    cols = proc.required_columns(["timestamp_ns", "symbol", "bar_start", "feat_a"])
    assert cols == ["feat_a"]


def test_make_run_id_format():
    from datetime import datetime, timezone
    rid = make_run_id("ic_horizon", "BTC",
                      now=datetime(2026, 6, 12, 14, 30, tzinfo=timezone.utc))
    assert rid == "proc_ic_horizon_BTC_20260612T143000Z"


def test_abcs_not_instantiable():
    with pytest.raises(TypeError):
        EvaluationProcess()
    with pytest.raises(TypeError):
        TransformProcess()
