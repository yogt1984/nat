"""Tests for realized higher moments (F5)."""

import numpy as np
import pandas as pd

from features.realized_moments import (
    REALIZED_MOMENTS_FEATURES,
    compute_realized_moments,
)


def _path(n, *, dt_s=10.0, symbol="BTC", seed=1, ann_vol=0.5,
          jump_prob=0.0, jump_size=-0.03, mid0=50_000.0):
    """GBM mid path on a dt_s grid; optional one-sided jumps for skew tests."""
    rng = np.random.default_rng(seed)
    step_vol = ann_vol * np.sqrt(dt_s / (365.0 * 86_400))
    r = rng.normal(0, 1, n) * step_vol
    if jump_prob > 0:
        r = r + (rng.random(n) < jump_prob) * jump_size
    mid = mid0 * np.exp(np.cumsum(r))
    ts = pd.Timestamp("2026-05-01", tz="UTC").value + np.arange(n, dtype=np.int64) * int(dt_s * 1e9)
    return pd.DataFrame({"timestamp_ns": ts, "symbol": symbol, "raw_midprice": mid})


def test_outputs_present_and_shape_preserved():
    df = _path(int(3 * 86_400 / 10))
    out = compute_realized_moments(df)
    assert len(out) == len(df)
    for c in REALIZED_MOMENTS_FEATURES:
        assert c in out.columns


def test_gaussian_moments_near_zero():
    out = compute_realized_moments(_path(int(5 * 86_400 / 10), seed=7))
    assert abs(out["rm_skew_1h"].dropna().mean()) < 0.3
    assert abs(out["rm_kurt_1h"].dropna().mean()) < 1.0     # excess kurt ~0


def test_negative_jumps_create_left_skew_and_downside_asymmetry():
    out = compute_realized_moments(
        _path(int(5 * 86_400 / 10), jump_prob=0.01, jump_size=-0.03, seed=3))
    assert out["rm_skew_1h"].dropna().mean() < 0
    sj = out["rm_signed_jump_1h"].dropna()
    assert sj.mean() < 0 and sj.between(-1.0, 1.0).all()


def test_per_symbol_independent():
    a = _path(int(3 * 86_400 / 10), symbol="BTC", seed=1)
    b = _path(int(3 * 86_400 / 10), symbol="ETH", jump_prob=0.01, jump_size=-0.04, seed=2)
    out = compute_realized_moments(pd.concat([a, b], ignore_index=True))
    btc = out[out.symbol == "BTC"]["rm_skew_1h"].dropna().mean()
    eth = out[out.symbol == "ETH"]["rm_skew_1h"].dropna().mean()
    assert eth < btc      # ETH carries the negative jumps → more left-skewed


def test_no_lookahead():
    df = _path(int(3 * 86_400 / 10), seed=4)
    full = compute_realized_moments(df)
    half = compute_realized_moments(df.iloc[: len(df) // 2].copy())
    n = len(half) // 2     # an early region well inside both frames
    np.testing.assert_allclose(
        full["rm_signed_jump_1h"].iloc[:n].to_numpy(),
        half["rm_signed_jump_1h"].iloc[:n].to_numpy(),
        rtol=1e-9, atol=1e-9, equal_nan=True,
    )
