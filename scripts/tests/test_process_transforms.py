"""Transform process contracts.

Triple barrier: exact agreement with an independent naive reference
implementation on a random path, NaN tail, vertical-barrier zeros, and
chaining into ic_horizon via target_col.

PCA combo: explained-variance ordering, holdout orthogonality, planted
factor recovered as an informative component, and a mandatory no-lookahead
check (perturbing the holdout never changes the train-segment output).
"""

import numpy as np
import pandas as pd
import pytest

from processes import get_process
from processes.synthetic import make_planted_frame, make_test_context

PRICE = "raw_midprice_close"


# ── triple barrier ───────────────────────────────────────────────────────────

def _naive_triple_barrier(p, vol, pt, sl, hold):
    """Reference implementation: explicit per-bar forward scan."""
    n = len(p)
    label = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    hit = np.full(n, np.nan)
    for t in range(n):
        if not np.isfinite(vol[t]) or t + hold >= n:
            continue
        up, dn = p[t] * np.exp(pt * vol[t]), p[t] * np.exp(-sl * vol[t])
        for k in range(1, hold + 1):
            if p[t + k] >= up:
                label[t], ret[t], hit[t] = 1.0, np.log(p[t + k] / p[t]), k
                break
            if p[t + k] <= dn:
                label[t], ret[t], hit[t] = -1.0, np.log(p[t + k] / p[t]), k
                break
        else:
            label[t], ret[t], hit[t] = 0.0, np.log(p[t + hold] / p[t]), hold
    return label, ret, hit


@pytest.fixture(scope="module")
def tb_setup():
    df = make_planted_frame(n=1200, seed=3)
    params = dict(pt_mult=1.5, sl_mult=1.0, max_holding_bars=10, vol_window=40)
    proc = get_process("triple_barrier", **params)
    derived, result = proc.transform(df, make_test_context())
    return df, params, derived, result


def test_triple_barrier_matches_reference(tb_setup):
    df, params, derived, _ = tb_setup
    p = df[PRICE].to_numpy()
    log_ret = pd.Series(np.concatenate([[np.nan], np.diff(np.log(p))]))
    vol = log_ret.rolling(params["vol_window"],
                          min_periods=params["vol_window"] // 2).std().to_numpy()
    ref_label, ref_ret, ref_hit = _naive_triple_barrier(
        p, vol, params["pt_mult"], params["sl_mult"], params["max_holding_bars"])

    np.testing.assert_array_equal(derived["tb_label"].to_numpy(), ref_label)
    np.testing.assert_allclose(derived["tb_ret"].to_numpy(), ref_ret, equal_nan=True)
    np.testing.assert_array_equal(derived["tb_hit_bars"].to_numpy(), ref_hit)


def test_triple_barrier_tail_and_warmup_nan(tb_setup):
    _, params, derived, _ = tb_setup
    lab = derived["tb_label"].to_numpy()
    assert np.all(np.isnan(lab[-params["max_holding_bars"]:]))   # no future data
    assert np.isnan(lab[0])                                      # vol not formed


def test_triple_barrier_classes_and_summary(tb_setup):
    _, _, derived, result = tb_setup
    lab = derived["tb_label"].dropna().unique()
    assert set(lab) <= {-1.0, 0.0, 1.0}
    f = result.findings[0]
    counts = f.extras["counts"]
    assert sum(counts.values()) == f.extras["n_labeled"] > 0
    assert 0.0 <= f.value <= 1.0


def test_triple_barrier_chains_into_ic_horizon(tb_setup):
    df, _, derived, _ = tb_setup
    chained = df.copy()
    chained["tb_label"] = derived["tb_label"]
    proc = get_process("ic_horizon", target_col="tb_label", min_breakeven_bps=0.0)
    result = proc.evaluate(chained, make_test_context())
    assert {f.horizon for f in result.findings} == {"label"}
    assert any(f.feature == "feat_signal" for f in result.findings)


# ── pca combo ────────────────────────────────────────────────────────────────

def _factor_frame(n=2000, seed=5):
    """3 noisy copies of one return-driving factor + 1 independent series."""
    rng = np.random.default_rng(seed)
    factor = np.zeros(n)
    for t in range(1, n):  # persistent factor
        factor[t] = 0.95 * factor[t - 1] + rng.normal()
    factor = factor / np.std(factor)
    eps = 2e-4 * factor + 5e-5 * rng.normal(size=n)  # factor drives returns
    prices = 100.0 * np.exp(np.cumsum(eps))
    return pd.DataFrame({
        "bar_start": pd.date_range("2026-01-01", periods=n, freq="15min"),
        "symbol": "SYN",
        PRICE: prices,
        "feat_a": factor + 0.3 * rng.normal(size=n),
        "feat_b": factor + 0.3 * rng.normal(size=n),
        "feat_c": -factor + 0.3 * rng.normal(size=n),
        "feat_ind": rng.normal(size=n),
    })


@pytest.fixture(scope="module")
def pca_setup():
    df = _factor_frame()
    proc = get_process("pca_combo", n_components=3)
    derived, result = proc.transform(df, make_test_context(horizons={"1bar": 1, "4bar": 4}))
    return df, derived, result


def test_pca_explained_variance_ordered(pca_setup):
    _, _, result = pca_setup
    ev = [f.extras["explained_var"] for f in result.findings]
    assert ev == sorted(ev, reverse=True)
    assert ev[0] > 0.5  # the shared factor dominates


def test_pca_holdout_orthogonality(pca_setup):
    _, _, result = pca_setup
    assert result.summary["orthogonality"] is not None
    assert result.summary["orthogonality"] < 0.15


def test_pca_factor_component_informative(pca_setup):
    _, _, result = pca_setup
    pc1 = next(f for f in result.findings if f.feature == "pc_1")
    assert abs(pc1.value) >= 0.05, pc1.extras
    assert pc1.informative
    assert set(pc1.extras["loadings_top5"]) <= {"feat_a", "feat_b", "feat_c", "feat_ind"}


def test_pca_no_lookahead(pca_setup):
    df, derived, _ = pca_setup
    n = len(df)
    split = int(n * 0.7)

    corrupted = df.copy()
    rng = np.random.default_rng(42)
    for col in ("feat_a", "feat_b", "feat_c", "feat_ind"):
        corrupted.loc[corrupted.index[split:], col] = rng.normal(size=n - split)

    proc = get_process("pca_combo", n_components=3)
    derived2, _ = proc.transform(corrupted, make_test_context(horizons={"1bar": 1}))

    # Holdout perturbation must not move the train-segment projections one bit
    pc_cols = ["pc_1", "pc_2", "pc_3"]
    np.testing.assert_allclose(
        derived.loc[: split - 1, pc_cols].to_numpy(),
        derived2.loc[: split - 1, pc_cols].to_numpy(),
    )


def test_pca_too_few_features_structured_error():
    df = make_planted_frame(n=500)[["bar_start", "symbol", PRICE, "feat_signal"]]
    proc = get_process("pca_combo")
    derived, result = proc.transform(df, make_test_context())
    assert result.summary["error"] and "usable features" in result.summary["error"]
    assert derived.empty or derived.columns.tolist() == []
