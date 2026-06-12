"""Planted-signal contracts for the information-theoretic processes.

MI: a feature carrying ~0.05 bits about forward returns must clear a cost
gate calibrated to ~0.01 bits; pure noise and a shuffled copy must not.
TE: an AR(1)-coupled (source -> target) pair must be flagged in the causal
direction only.
"""

import numpy as np
import pytest

from processes import get_process
from processes.synthetic import (
    make_ar1_coupled,
    make_planted_frame,
    make_test_context,
)


def _fee_for_imin(target_bits: float, sigma_r_bps: float) -> float:
    """Invert I_min = -0.5*log2(1 - (fee/sigma)^2) for the fee."""
    r = np.sqrt(1.0 - 2.0 ** (-2.0 * target_bits))
    return float(r * sigma_r_bps)


@pytest.fixture(scope="module")
def mi_result():
    # ic=0.35 -> true MI ~ -0.5*log2(1-0.1225) = 0.094 bits; gate at 0.04 sits
    # well above the KSG noise floor (~0.01-0.02 bits at n=3000) and well
    # below the planted signal.
    df = make_planted_frame(n=3000, ic=0.35, horizon=4, seed=7)
    rng = np.random.default_rng(13)
    df["feat_shuffled"] = rng.permutation(df["feat_signal"].to_numpy())
    ctx = make_test_context(horizons={"4bar": 4})

    prices = df["raw_midprice_close"].to_numpy()
    fwd = prices[4:] / prices[:-4] - 1.0
    sigma_r_bps = float(np.std(fwd)) * 1e4
    fee = _fee_for_imin(0.04, sigma_r_bps)

    proc = get_process("mi_ksg", fee_rt_bps=fee, kurtosis_correction=False)
    return proc.evaluate(df, ctx)


def test_mi_planted_signal_informative(mi_result):
    f = next(f for f in mi_result.findings if f.feature == "feat_signal")
    assert f.value >= f.threshold, f"MI {f.value} below gate {f.threshold}"
    assert f.informative


def test_mi_noise_and_shuffled_not_informative(mi_result):
    for feat in ("feat_noise", "feat_shuffled"):
        f = next(f for f in mi_result.findings if f.feature == feat)
        assert not f.informative, f"{feat}: MI {f.value} vs gate {f.threshold}"


def test_mi_dead_columns_skipped(mi_result):
    reasons = {s["feature"]: s["reason"] for s in mi_result.features_skipped}
    assert reasons["feat_dead"] == "all_nan"
    assert reasons["feat_const"] == "constant"


def test_mi_conditioning_extras():
    df = make_planted_frame(n=1500, ic=0.25, horizon=4, seed=7)
    df["feat_cond"] = np.random.default_rng(3).normal(size=len(df))
    ctx = make_test_context(horizons={"4bar": 4})
    proc = get_process("mi_ksg", conditioning=["feat_cond"], fee_rt_bps=1.0,
                       max_samples=1500)
    result = proc.evaluate(df, ctx)
    f = next(f for f in result.findings if f.feature == "feat_signal")
    assert "cmi_bits" in f.extras and "interaction_info_bits" in f.extras
    assert "feat_cond" not in [g.feature for g in result.findings]


def _te_frame_and_fee():
    source, target = make_ar1_coupled(n=4000, coupling=0.6, phi=0.3, seed=7)
    ret = target * 0.001
    prices = 100.0 * np.exp(np.cumsum(ret))
    import pandas as pd
    df = pd.DataFrame({
        "raw_midprice_close": prices,
        "feat_src": source,
        "feat_noise": np.random.default_rng(5).normal(size=len(source)),
    })
    fee = _fee_for_imin(0.01, float(np.std(ret)) * 1e4)
    return df, fee


def test_te_causal_direction_flagged():
    df, fee = _te_frame_and_fee()
    proc = get_process("transfer_entropy", fee_rt_bps=fee, kurtosis_correction=False)
    result = proc.evaluate(df, make_test_context(horizons={"1bar": 1}))

    src = next(f for f in result.findings if f.feature == "feat_src")
    assert src.informative, (
        f"TE fwd={src.value} rev={src.extras['te_reverse_bits']} gate={src.threshold}"
    )
    assert src.value > src.extras["te_reverse_bits"]

    noise = next(f for f in result.findings if f.feature == "feat_noise")
    assert not noise.informative


def test_te_ksg_variant_runs():
    # Smoke only: the KSG variant inherits it_engine's 4-term entropy CMI,
    # which is biased low on autocorrelated targets and often clamps to 0
    # (documented in the te_method param) — linear is the recommended method.
    df, fee = _te_frame_and_fee()
    proc = get_process("transfer_entropy", te_method="ksg", max_samples=800,
                       fee_rt_bps=fee, kurtosis_correction=False)
    result = proc.evaluate(df, make_test_context(horizons={"1bar": 1}))
    src = next(f for f in result.findings if f.feature == "feat_src")
    assert src.value >= 0.0 and np.isfinite(src.value)
    assert src.extras["method"] == "ksg"
