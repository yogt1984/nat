"""Planted-signal contract for the ic_horizon process.

A feature constructed to predict the 4-bar forward return must be flagged
informative at that horizon (and the IC decay curve must peak there); a
shuffled copy and pure noise must not; dead/constant columns are skipped
with reasons, never raised on.
"""

import numpy as np
import pytest

from processes import get_process
from processes.synthetic import make_planted_frame, make_test_context

PLANTED_HORIZON = 4


@pytest.fixture(scope="module")
def planted_result():
    df = make_planted_frame(n=3000, ic=0.20, horizon=PLANTED_HORIZON, seed=7)
    rng = np.random.default_rng(13)
    df["feat_shuffled"] = rng.permutation(df["feat_signal"].to_numpy())
    proc = get_process("ic_horizon", min_breakeven_bps=0.0)
    return proc.evaluate(df, make_test_context())


def _findings(result, feature):
    return {f.horizon: f for f in result.findings if f.feature == feature}


def test_planted_signal_informative_at_planted_horizon(planted_result):
    by_h = _findings(planted_result, "feat_signal")
    f = by_h["4bar"]
    assert f.informative, (
        f"planted signal not flagged: ic={f.value}, p_adj={f.p_adjusted}"
    )
    assert abs(f.value) > 0.05


def test_ic_decay_curve_peaks_at_planted_horizon(planted_result):
    by_h = _findings(planted_result, "feat_signal")
    decay = by_h["4bar"].extras["ic_decay"]
    assert max(decay, key=lambda h: abs(decay[h])) == "4bar", decay


def test_shuffled_and_noise_not_informative(planted_result):
    for feat in ("feat_shuffled", "feat_noise"):
        for f in _findings(planted_result, feat).values():
            assert not f.informative, (
                f"{feat}@{f.horizon} falsely flagged: ic={f.value}, p_adj={f.p_adjusted}"
            )


def test_dead_and_constant_skipped_with_reasons(planted_result):
    reasons = {s["feature"]: s["reason"] for s in planted_result.features_skipped}
    assert reasons["feat_dead"] == "all_nan"
    assert reasons["feat_const"] == "constant"
    assert "feat_dead" not in planted_result.features_tested


def test_result_record_shape(planted_result):
    d = planted_result.to_dict()
    assert d["schema_version"] == 1
    assert d["process"] == "ic_horizon"
    assert d["kind"] == "evaluation"
    assert d["run_id"].startswith("proc_ic_horizon_SYN_")
    assert d["summary"]["n_tested"] == len(d["features_tested"])
    assert d["summary"]["error"] is None
    assert d["summary"]["top"], "summary.top must rank findings"
    f = d["findings"][0]
    for key in ("feature", "horizon", "metric", "value", "p_adjusted", "informative"):
        assert key in f


def test_expanding_curve_and_halflife_present(planted_result):
    f = _findings(planted_result, "feat_signal")["4bar"]
    assert len(f.extras["ic_expanding"]) >= 2
    assert "ic_decay_halflife_bars" in f.extras


def test_target_col_mode():
    df = make_planted_frame(n=2000, ic=0.25, horizon=2, seed=11)
    prices = df["raw_midprice_close"].to_numpy()
    fwd = np.full(len(df), np.nan)
    fwd[:-2] = prices[2:] / prices[:-2] - 1.0
    df["my_label"] = fwd

    proc = get_process("ic_horizon", target_col="my_label", min_breakeven_bps=0.0)
    result = proc.evaluate(df, make_test_context())
    horizons = {f.horizon for f in result.findings}
    assert horizons == {"label"}
    sig = [f for f in result.findings if f.feature == "feat_signal"]
    assert sig and abs(sig[0].value) > 0.1
