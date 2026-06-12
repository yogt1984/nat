"""ML importance process contracts: planted feature ranks first with a
profitable confidence-filtered strategy; noise does not; missing lightgbm
degrades to a structured error."""

import sys

import numpy as np
import pytest

from processes import get_process
from processes.synthetic import make_planted_frame, make_test_context

pytest.importorskip("lightgbm")


@pytest.fixture(scope="module")
def ml_result():
    df = make_planted_frame(n=3000, ic=0.30, horizon=4, seed=7)
    rng = np.random.default_rng(13)
    df["feat_shuffled"] = rng.permutation(df["feat_signal"].to_numpy())
    proc = get_process("ml_importance", top_k=1, n_splits=4)
    return proc.evaluate(df, make_test_context(horizons={"4bar": 4}))


def test_planted_feature_ranks_first_and_informative(ml_result):
    f = next(f for f in ml_result.findings if f.feature == "feat_signal")
    assert f.extras["rank"] == 1, ml_result.summary
    assert f.extras["conf_filtered"]["net_pnl_bps"] > 0
    assert f.extras["conf_filtered"]["n_trades"] > 0
    assert f.informative


def test_noise_and_shuffled_not_informative(ml_result):
    for feat in ("feat_noise", "feat_shuffled"):
        f = next(f for f in ml_result.findings if f.feature == feat)
        assert not f.informative, f"{feat} rank={f.extras['rank']}"


def test_wf_accuracy_above_chance(ml_result):
    f = next(f for f in ml_result.findings if f.feature == "feat_signal")
    assert f.extras["wf_accuracy"] > 0.52


def test_dead_columns_skipped(ml_result):
    reasons = {s["feature"]: s["reason"] for s in ml_result.features_skipped}
    assert reasons["feat_dead"] == "all_nan"


def test_missing_lightgbm_structured_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)  # import -> ImportError
    df = make_planted_frame(n=500)
    proc = get_process("ml_importance")
    result = proc.evaluate(df, make_test_context(horizons={"4bar": 4}))
    assert result.summary["error"] == "lightgbm not installed"
    assert result.findings == []
