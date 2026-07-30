"""Skeptical suite for PROC-5 Part A — mi_ksg honoring an explicit target column.

Today mi_ksg ALWAYS scores features against forward returns; it cannot evaluate the
3-bar triple-barrier label (`tb_label`). PROC-5 adds a label mode: with `target_col`
set, mi_ksg scores MI(feature; label) and — because a label is not a tradeable return,
so the cost-viability i_min is meaningless — gates informativeness via the PROC-12
null-calibration (bits-above-null / z), not the fee gate.

These tests are adversarial: they try to make the label mode lie (self-prediction,
sibling leakage, the KSG floor masquerading as signal, degenerate targets) and assert
it doesn't. The forward-return mode MUST stay bit-for-bit unchanged (no regression).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from processes.base import ProcessContext
from processes.info_theory import MutualInfoProcess


# --------------------------------------------------------------------------- #
# fixtures / builders                                                         #
# --------------------------------------------------------------------------- #

def _ctx(target_col=None, horizons=None):
    return ProcessContext(
        symbol="BTC", timeframe="15min", price_col="raw_midprice",
        horizons=horizons or {"h1": 1, "h2": 4}, costs={}, target_col=target_col,
    )


def _labelled_bars(n=1500, seed=0, snr=3.0):
    """Bars with a 3-class tb_label that `f_good` determines and `f_noise` doesn't."""
    rng = np.random.default_rng(seed)
    f_good = rng.standard_normal(n)
    f_noise = rng.standard_normal(n)
    drive = snr * f_good + rng.standard_normal(n)
    # tb_label in {-1, 0, +1} by terciles of the driver
    q1, q2 = np.quantile(drive, [1 / 3, 2 / 3])
    label = np.where(drive <= q1, -1.0, np.where(drive >= q2, 1.0, 0.0))
    price = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1)
    return pd.DataFrame({
        "raw_midprice": price,
        "f_good": f_good,
        "f_noise": f_noise,
        "tb_label": label,
        "tb_ret": drive,                 # a sibling barrier column (leaky by construction)
        "tb_hit_bars": rng.integers(1, 16, n).astype(float),
    })


def _run(bars, *, target_col=None, ctx_target=None, **params):
    params.setdefault("min_obs", 100)
    params.setdefault("null_shuffles", 40)
    params.setdefault("seed", 1)
    if target_col is not None:
        params["target_col"] = target_col
    proc = MutualInfoProcess(**params)
    return proc.evaluate(bars, _ctx(target_col=ctx_target))


# --------------------------------------------------------------------------- #
# regression: forward-return mode must be untouched                           #
# --------------------------------------------------------------------------- #

class TestForwardReturnModeUnchanged:
    def test_no_target_still_uses_forward_returns(self):
        bars = _labelled_bars(seed=1)
        res = _run(bars, min_obs=100)
        horizons = {f.horizon for f in res.findings}
        assert horizons <= {"h1", "h2"}          # per-horizon, never a "label" pseudo-horizon
        assert "label" not in horizons
        assert all(f.metric == "mi_bits" for f in res.findings)

    def test_no_target_scores_tb_label_as_a_feature(self):
        # Without a target, tb_label is just another feature vs forward returns.
        bars = _labelled_bars(seed=2)
        res = _run(bars, min_obs=100)
        assert any(f.feature == "tb_label" for f in res.findings)


# --------------------------------------------------------------------------- #
# label mode: shape + exclusions                                              #
# --------------------------------------------------------------------------- #

class TestLabelModeShape:
    def test_label_mode_uses_single_label_horizon(self):
        bars = _labelled_bars(seed=3)
        res = _run(bars, target_col="tb_label")
        assert res.findings, "label mode must produce findings"
        assert {f.horizon for f in res.findings} == {"label"}
        assert all(f.metric == "mi_bits" for f in res.findings)

    def test_target_column_not_scored_against_itself(self):
        bars = _labelled_bars(seed=4)
        res = _run(bars, target_col="tb_label")
        assert not any(f.feature == "tb_label" for f in res.findings)

    def test_sibling_barrier_columns_excluded(self):
        # tb_ret / tb_hit_bars are derived from the same barrier event -> trivial leakage.
        bars = _labelled_bars(seed=5)
        res = _run(bars, target_col="tb_label")
        scored = {f.feature for f in res.findings}
        assert "tb_ret" not in scored
        assert "tb_hit_bars" not in scored
        assert "f_good" in scored and "f_noise" in scored

    def test_ctx_target_col_honored_when_param_absent(self):
        bars = _labelled_bars(seed=6)
        proc = MutualInfoProcess(min_obs=100, null_shuffles=40, seed=1)
        res = proc.evaluate(bars, _ctx(target_col="tb_label"))
        assert {f.horizon for f in res.findings} == {"label"}

    def test_param_target_col_overrides_ctx(self):
        bars = _labelled_bars(seed=7)
        # param says tb_label; ctx says something else -> param wins, no crash
        res = _run(bars, target_col="tb_label", ctx_target="tb_ret")
        assert {f.horizon for f in res.findings} == {"label"}
        assert not any(f.feature == "tb_label" for f in res.findings)


# --------------------------------------------------------------------------- #
# label mode: the null-calibration gate (honesty)                             #
# --------------------------------------------------------------------------- #

class TestLabelModeGate:
    def test_informative_feature_detected(self):
        bars = _labelled_bars(seed=8, snr=4.0)
        res = _run(bars, target_col="tb_label")
        good = [f for f in res.findings if f.feature == "f_good"][0]
        assert good.informative, good
        assert good.extras["z"] >= 3.0
        assert good.extras["bits_above_null"] > 0.0
        assert good.extras["p"] <= 0.05

    def test_pure_noise_feature_not_informative(self):
        bars = _labelled_bars(seed=9)
        res = _run(bars, target_col="tb_label")
        noise = [f for f in res.findings if f.feature == "f_noise"][0]
        assert not noise.informative, noise
        assert noise.extras["z"] < 3.0

    def test_cost_gate_not_applied_in_label_mode(self):
        # A label has no bps sigma; the fee-based i_min must NOT drive the verdict.
        bars = _labelled_bars(seed=10)
        res = _run(bars, target_col="tb_label")
        for f in res.findings:
            # threshold in label mode is the null-z gate, not a fee i_min
            assert "i_min_bits" not in f.extras or f.extras.get("gate") == "null_z"
            assert "z" in f.extras and "bits_above_null" in f.extras

    def test_extras_carry_null_calibration_fields(self):
        bars = _labelled_bars(seed=11)
        res = _run(bars, target_col="tb_label")
        f = res.findings[0]
        for key in ("bits_above_null", "z", "p", "null_mean", "n_samples"):
            assert key in f.extras, (key, f.extras)

    def test_value_is_raw_mi_bits(self):
        bars = _labelled_bars(seed=12)
        res = _run(bars, target_col="tb_label")
        f = [f for f in res.findings if f.feature == "f_good"][0]
        # value == raw MI; bits_above_null == value - null_mean
        assert abs(f.value - (f.extras["bits_above_null"] + f.extras["null_mean"])) < 1e-6


# --------------------------------------------------------------------------- #
# label mode: determinism                                                     #
# --------------------------------------------------------------------------- #

class TestDeterminism:
    def test_seeded_runs_reproducible(self):
        bars = _labelled_bars(seed=13)
        r1 = _run(bars, target_col="tb_label", seed=7)
        r2 = _run(bars, target_col="tb_label", seed=7)
        v1 = {(f.feature, f.value, f.extras["z"]) for f in r1.findings}
        v2 = {(f.feature, f.value, f.extras["z"]) for f in r2.findings}
        assert v1 == v2

    def test_different_seed_changes_null_but_not_raw_mi(self):
        bars = _labelled_bars(seed=14)
        r1 = _run(bars, target_col="tb_label", seed=1)
        r2 = _run(bars, target_col="tb_label", seed=2)
        raw1 = {f.feature: f.value for f in r1.findings}
        raw2 = {f.feature: f.value for f in r2.findings}
        assert raw1 == raw2                       # raw MI does not depend on the shuffle seed


# --------------------------------------------------------------------------- #
# label mode: degenerate / adversarial inputs                                 #
# --------------------------------------------------------------------------- #

class TestDegenerateInputs:
    def test_missing_target_column_errors_cleanly(self):
        bars = _labelled_bars(seed=15).drop(columns=["tb_label"])
        res = _run(bars, target_col="tb_label")
        assert res.findings == []
        assert res.summary.get("error")
        assert "tb_label" in res.summary["error"]

    def test_all_nan_target_produces_no_findings_no_crash(self):
        bars = _labelled_bars(seed=16)
        bars["tb_label"] = np.nan
        res = _run(bars, target_col="tb_label")
        assert res.findings == []           # nothing to score, but no exception

    def test_constant_target_not_informative(self):
        bars = _labelled_bars(seed=17)
        bars["tb_label"] = 0.0
        res = _run(bars, target_col="tb_label")
        # a constant label carries no information; nothing should be flagged
        assert not any(f.informative for f in res.findings)

    def test_too_few_rows_skips(self):
        bars = _labelled_bars(n=50, seed=18)
        res = _run(bars, target_col="tb_label", min_obs=200)
        assert res.findings == []

    def test_nan_feature_rows_dropped_not_fatal(self):
        bars = _labelled_bars(seed=19)
        bars.loc[bars.index[:300], "f_good"] = np.nan
        res = _run(bars, target_col="tb_label")
        # f_good still scored on its finite rows (enough remain past min_obs)
        assert any(f.feature == "f_good" for f in res.findings)

    def test_binary_label_supported(self):
        bars = _labelled_bars(seed=20)
        bars["tb_label"] = (bars["tb_ret"] > 0).astype(float)   # 2-class
        res = _run(bars, target_col="tb_label")
        assert {f.horizon for f in res.findings} == {"label"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
